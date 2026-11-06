"""Complete raw-score panels, frozen objectives and pending-aware qLogNEHVI."""

import hashlib
import math
import statistics
from dataclasses import dataclass
from functools import partial

import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from .plan_schema import digest


@dataclass(frozen=True)
class Observation:
    config: dict
    mean: tuple
    target: tuple
    variance: tuple
    panel_identity: str
    synthetic: bool = False


def validate_receipt(receipt, *, split, protocol, allow_synthetic=False):
    """Check one episode receipt before the ledger marks it done (irreversible).

    Same measurement rules as aggregate_panel, plus agent steps for live runs.
    """
    if receipt.get("split") != split or receipt.get("protocol") != protocol:
        raise ValueError("Receipt split/protocol mismatch")
    if receipt.get("synthetic", False) and not allow_synthetic:
        raise ValueError("Synthetic receipt outside an isolated smoke protocol")
    for field in ("raw_score", "total_tokens", "latency_seconds"):
        v = receipt.get(field)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"Missing or nonfinite raw measurement: {field}")
        if field != "raw_score" and v < 0:
            raise ValueError("Costs must be nonnegative")
    if not receipt.get("synthetic", False):
        steps = receipt.get("agent_steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Missing agent_steps")


def aggregate_panel(space, config, rows, *, task_ids, protocol, allow_synthetic=False,
                    allow_partial=False, quality_scale=None):
    """No quality fallback: callers must supply explicit, unmodified raw_score.

    ``allow_partial`` (staged pruning): ``rows`` may cover a subset of the panel (at least two
    tasks); the mean is over the measured tasks and the noise is var / n, so a pruned
    configuration enters the GP with its larger standard error. The panel identity stays
    the full panel, so complete and pruned observations share one GP."""
    rows = list(rows)
    expected = tuple(map(str, task_ids))
    ids = [str(r.get("task_id")) for r in rows]
    if len(set(expected)) != len(expected) or len(expected) < 2:
        raise ValueError("Panel requires distinct tasks and at least two measurements")
    if len(ids) != len(set(ids)):
        raise ValueError("Incomplete or duplicate task panel")
    if allow_partial:
        if len(ids) < 2 or not set(ids) <= set(expected):
            raise ValueError("Partial panel needs at least two tasks of the frozen panel")
    elif set(ids) != set(expected):
        raise ValueError("Incomplete or duplicate task panel")
    if any(r.get("split") != "TRAIN" or r.get("protocol") != protocol for r in rows):
        raise ValueError("Panel split/protocol mismatch")
    synthetic = any(r.get("synthetic", False) for r in rows)
    if synthetic and (not allow_synthetic or not all(r.get("synthetic") for r in rows)):
        raise ValueError(
            "Synthetic measurements require an explicit isolated smoke protocol"
        )
    columns = []
    for field in ("raw_score", "total_tokens", "latency_seconds"):
        values = [r.get(field) for r in rows]
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in values
        ):
            raise ValueError(f"Missing or nonfinite raw measurement: {field}")
        if field != "raw_score" and any(v < 0 for v in values):
            raise ValueError("Costs must be nonnegative")
        columns.append(values)
    means = tuple(statistics.mean(v) for v in columns)
    var = [statistics.variance(v) / len(v) for v in columns]
    if allow_partial and quality_scale and len(ids) < len(expected):
        # Partial panel (pruned configuration): floor the quality SE at the Agresti-Coull SE of
        # the mean reward rescaled to [0, 1], as in the pruning decision (user 2026-11-04).
        n, z = len(ids), 1.96
        p = min(1.0, max(0.0, means[0] / quality_scale))
        p_ac = (p * n + z * z / 2) / (n + z * z)
        var[0] = max(var[0], quality_scale**2 * p_ac * (1 - p_ac) / (n + z * z))
    target = (means[0], -math.log1p(means[1]), -math.log1p(means[2]))
    noise = (
        max(var[0], 1e-4),
        max(var[1] / (1 + means[1]) ** 2, 1e-8),
        max(var[2] / (1 + means[2]) ** 2, 1e-8),
    )
    return Observation(
        space.canonicalize(config),
        means,
        target,
        noise,
        digest([protocol, sorted(expected), synthetic]),
        synthetic,
    )


@dataclass(frozen=True)
class ObjectiveProtocol:
    lo: tuple
    span: tuple
    ref: tuple = (-0.1, -0.1, -0.1)

    @classmethod
    def from_initial(cls, observations):
        y = torch.tensor([o.target for o in observations], dtype=torch.double)
        lo, hi = y.min(0).values, y.max(0).values
        span = torch.maximum(hi - lo, torch.tensor([1.0, 1e-3, 1e-3]))
        return cls(tuple(lo.tolist()), tuple(span.tolist()))

    def tensors(self, observations):
        y = torch.tensor([o.target for o in observations], dtype=torch.double)
        v = torch.tensor([o.variance for o in observations], dtype=torch.double)
        span = torch.tensor(self.span, dtype=torch.double)
        lo = torch.tensor(self.lo, dtype=torch.double)
        return (y - lo) / span, v / span.square()


def fit_model(codec, observations, objective, *, seed=0, maxiter=150, timeout=120):
    if len({o.panel_identity for o in observations}) != 1:
        raise ValueError(
            "GP observations must share the exact TRAIN panel and protocol"
        )
    if len({codec.key(o.config) for o in observations}) != len(observations):
        raise ValueError("Repeated complete config observations")
    x = torch.tensor([codec.encode(o.config) for o in observations], dtype=torch.double)
    y, noise = objective.tensors(observations)
    models = []
    for i in range(3):
        m = MixedSingleTaskGP(
            x,
            y[:, i : i + 1],
            train_Yvar=noise[:, i : i + 1],
            cat_dims=list(codec.cat_dims),
            outcome_transform=Standardize(m=1),
            cont_kernel_factory=partial(
                get_covar_module_with_dim_scaled_prior, use_rbf_kernel=False
            ),
        )
        for kernel in m.modules():
            if isinstance(kernel, ScaleKernel):
                kernel.register_prior(
                    "outputscale_prior", GammaPrior(2.0, 1.0), "outputscale"
                )
        models.append(m)
    model = ModelListGP(*models)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fit_gpytorch_mll(
            SumMarginalLogLikelihood(model.likelihood, model),
            optimizer_kwargs={"options": {"maxiter": maxiter}, "timeout_sec": timeout},
        )
    model.eval()
    return model


class AcquisitionSnapshot:
    """A frozen GP/objective/MC/pending snapshot; duplicate draws retain weights.

    Caches contain detached scalars only. A pending update creates a new cache;
    the caller shares a ProposalBudget across all snapshots in one round.
    """

    def __init__(
        self,
        codec,
        model,
        observations,
        objective,
        *,
        seed=0,
        mc_samples=64,
        pending=(),
        budget=None,
    ):
        self.codec, self.model, self.observations = codec, model, observations
        self.objective, self.seed, self.mc_samples = objective, seed, mc_samples
        self.pending = list(pending)
        self.budget = budget
        self.cache, self.forwards = {}, 0
        state_hash = hashlib.sha256()
        for name, value in sorted(model.state_dict().items()):
            state_hash.update(name.encode())
            state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        self.identity = digest(
            [
                codec.space.identity,
                state_hash.hexdigest(),
                [o.__dict__ for o in observations],
                objective.__dict__,
                seed,
                mc_samples,
                [codec.key(c) for c in pending],
            ]
        )
        x = torch.tensor(
            [codec.encode(o.config) for o in observations], dtype=torch.double
        )
        xp = (
            torch.tensor([codec.encode(c) for c in pending], dtype=torch.double)
            if pending
            else None
        )
        self.acquisition = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=list(objective.ref),
            X_baseline=x,
            sampler=SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed),
            X_pending=xp,
            prune_baseline=False,
            cache_root=True,
        )

    @property
    def unique_inputs(self):
        return len(self.cache)

    def with_pending(self, pending):
        return AcquisitionSnapshot(
            self.codec,
            self.model,
            self.observations,
            self.objective,
            seed=self.seed,
            mc_samples=self.mc_samples,
            pending=pending,
            budget=self.budget,
        )

    def posterior(self, configs):
        x = torch.tensor([self.codec.encode(c) for c in configs], dtype=torch.double)
        with torch.no_grad():
            return self.model.posterior(x).mean

    def score(self, configs):
        keys = [self.codec.key(c) for c in configs]
        missing = {k: c for k, c in zip(keys, configs) if k not in self.cache}
        items = list(missing.items())
        for begin in range(0, len(items), 32):
            chunk = items[begin : begin + 32]
            if self.budget:
                self.budget.admit([k for k, _ in chunk])
            x = torch.tensor(
                [self.codec.encode(c) for _, c in chunk], dtype=torch.double
            )
            with torch.no_grad():
                values = self.acquisition(x.unsqueeze(-2)).tolist()
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError("Nonfinite acquisition; no silent fallback")
            self.forwards += len(chunk)
            self.cache.update({k: float(v) for (k, _), v in zip(chunk, values)})
        return [self.cache[k] for k in keys]
