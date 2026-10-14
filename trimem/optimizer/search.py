"""Pure ask step for quality / total-LLM-token multi-objective search.

RAG-Stack's ask/evaluate/tell separation informs the interface, but this is an
independent implementation using BoTorch directly. Importing this module does
not import torch, connect to Postgres, call a model, or evaluate a candidate.
Latency is deliberately absent from both objectives and constraints in v1.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .schema import CandidateSpec
from .space import SearchSpace


@dataclass(frozen=True)
class SearchData:
    """One comparable cohort aggregate, in raw units.

    Variances are variance *of the mean* (SEM squared), not sample variance.
    The archive owns task/cohort isolation and pooling repeated measurements.
    A quality value is a measured downstream outcome, never a predicted label.
    """

    action: CandidateSpec
    quality_mean: float
    total_tokens_mean: float
    quality_variance_of_mean: float
    tokens_variance_of_mean: float
    n: int = 1

    def __post_init__(self) -> None:
        values = (
            self.quality_mean,
            self.total_tokens_mean,
            self.quality_variance_of_mean,
            self.tokens_variance_of_mean,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("search observations must be finite")
        if self.total_tokens_mean < 0:
            raise ValueError("total token consumption cannot be negative")
        if self.quality_variance_of_mean < 0 or self.tokens_variance_of_mean < 0:
            raise ValueError("variance of mean cannot be negative")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1:
            raise ValueError("n must be a positive integer")


@dataclass(frozen=True)
class SearchSettings:
    """Frozen objective geometry and bounded CPU search settings.

    The raw reference is (reference_quality, -reference_tokens). It is an HV
    reporting boundary, *not* a token budget. Pick it before searching; do not
    recompute it from the current trial extrema. Noise floors are raw variances.
    """

    reference_quality: float
    reference_tokens: float
    quality_scale: float = 1.0
    token_scale: float = 1000.0
    initial_per_plan: int = 2
    seed: int = 0
    max_fit_iterations: int = 50
    mc_samples: int = 64
    pool_size: int = 256
    acquisition_batch_size: int = 32
    allow_repeats: bool = False
    quality_noise_floor: float = 1e-6
    token_noise_floor: float = 1.0
    fallback_on_error: bool = True
    # "per_plan": initial_per_plan points per plan family (pre-v5 behaviour).
    # "uniform": no_memory_v1 baseline first (not counted), then initial_random
    # uniformly random actions over the whole space (RAG-Stack style).
    initial_strategy: str = "per_plan"
    initial_random: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.reference_quality):
            raise ValueError("reference_quality must be finite")
        for field in (
            "reference_tokens",
            "quality_scale",
            "token_scale",
            "quality_noise_floor",
            "token_noise_floor",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be positive and finite")
        for field in (
            "initial_per_plan",
            "max_fit_iterations",
            "mc_samples",
            "pool_size",
            "acquisition_batch_size",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.initial_strategy not in ("per_plan", "uniform"):
            raise ValueError("initial_strategy must be per_plan or uniform")
        if (
            isinstance(self.initial_random, bool)
            or not isinstance(self.initial_random, int)
            or self.initial_random < 0
        ):
            raise ValueError("initial_random must be a nonnegative integer")

    @property
    def reference_point(self) -> tuple[float, float]:
        return (
            self.reference_quality / self.quality_scale,
            -self.reference_tokens / self.token_scale,
        )


@dataclass(frozen=True)
class Proposal:
    action: CandidateSpec
    strategy: str
    diagnostic: str
    acquisition_value: float | None = None


class SpaceExhausted(ValueError):
    """All legal candidates have been measured or explicitly excluded."""


def objective_arrays(
    observations: Sequence[SearchData], settings: SearchSettings
) -> tuple[list[list[float]], list[list[float]]]:
    """Prepare maximize(Q, -T) means and positive fixed observation noise."""
    means, variances = [], []
    for row in observations:
        means.append(
            [
                row.quality_mean / settings.quality_scale,
                -row.total_tokens_mean / settings.token_scale,
            ]
        )
        variances.append(
            [
                max(row.quality_variance_of_mean, settings.quality_noise_floor)
                / settings.quality_scale**2,
                max(row.tokens_variance_of_mean, settings.token_noise_floor)
                / settings.token_scale**2,
            ]
        )
    return means, variances


def _stratified_pool(
    actions: Sequence[CandidateSpec], size: int, rng: random.Random
) -> list[CandidateSpec]:
    """Bound acquisition work while keeping all available plan families visible."""
    if len(actions) <= size:
        return list(actions)
    groups: dict[str, list[CandidateSpec]] = {}
    for action in actions:
        groups.setdefault(action.plan_id, []).append(action)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    rng.shuffle(keys)
    result: list[CandidateSpec] = []
    while len(result) < size:
        for key in keys:
            if groups[key]:
                result.append(groups[key].pop())
                if len(result) == size:
                    return result
    return result


def _balanced_choice(
    actions: Sequence[CandidateSpec],
    observations: Sequence[SearchData],
    rng: random.Random,
) -> CandidateSpec:
    counts = Counter(row.action.plan_id for row in observations)
    minimum = min(counts[action.plan_id] for action in actions)
    plans = sorted({a.plan_id for a in actions if counts[a.plan_id] == minimum})
    plan = rng.choice(plans)
    return rng.choice([a for a in actions if a.plan_id == plan])


def suggest(
    space: SearchSpace,
    observations: Sequence[SearchData],
    excluded_ids: Iterable[str],
    settings: SearchSettings,
) -> Proposal:
    """Suggest one full action, never evaluate it or mutate an archive.

    Explicit exclusions (including pending trial IDs represented by action hash)
    always win. Completed actions are excluded unless allow_repeats is explicit.
    Observations outside a fixed-plan subspace are ignored. The caller records
    the proposal and its diagnostic in its durable trial reservation.
    """
    relevant = [row for row in observations if space.contains(row.action)]
    seen = {row.action.action_hash for row in relevant}
    if len(seen) != len(relevant):
        raise ValueError("aggregate repeated observations per action before suggest")
    excluded = set(excluded_ids)
    unavailable = excluded | (set() if settings.allow_repeats else seen)
    available = [a for a in space.candidates if a.action_hash not in unavailable]
    if not available:
        raise SpaceExhausted("no unevaluated/unreserved legal actions remain")
    rng = random.Random(settings.seed + sum(row.n for row in relevant) + len(excluded))
    if settings.initial_strategy == "uniform":
        # Mutually exclusive with per-plan initialization: once the uniform
        # budget is spent, go straight to the surrogate (never stratify).
        proposal = _uniform_initialization(space, available, seen | excluded, settings)
        if proposal is not None:
            return proposal
    else:
        proposal = _per_plan_initialization(
            space, available, seen, seen | excluded, settings, rng
        )
        if proposal is not None:
            return proposal
    if len(relevant) < 2:
        return Proposal(
            _balanced_choice(available, relevant, rng),
            "stratified_fallback",
            "at least two distinct measured actions are required to fit the surrogate",
        )
    pool = _stratified_pool(available, settings.pool_size, rng)
    try:
        index, value = _qlognehvi(space, relevant, pool, settings)
        return Proposal(
            pool[index],
            "qlognehvi",
            f"CPU float64; measured={len(relevant)}; candidates={len(pool)}; "
            "objectives=(quality,-total_llm_tokens)",
            value,
        )
    except Exception as error:
        if not settings.fallback_on_error:
            raise
        # Fallback is explicit and inspectable; it is never reported as BO.
        return Proposal(
            _balanced_choice(available, relevant, rng),
            "stratified_fallback",
            f"qLogNEHVI unavailable/failed: {type(error).__name__}: {error}"[:1000],
        )


def _per_plan_initialization(
    space: SearchSpace,
    available: Sequence[CandidateSpec],
    seen: set[str],
    taken: set[str],
    settings: SearchSettings,
    rng: random.Random,
) -> Proposal | None:
    """Each plan gets one point before any plan gets its second. no-memory first."""
    by_plan: dict[str, list[CandidateSpec]] = {}
    for action in available:
        by_plan.setdefault(action.plan_id, []).append(action)
    counts = Counter(
        action.plan_id for action in space.candidates if action.action_hash in taken
    )
    sizes = Counter(action.plan_id for action in space.candidates)
    underexplored = [
        plan
        for plan in space.plan_ids
        if plan in by_plan
        and counts[plan] < min(settings.initial_per_plan, sizes[plan])
    ]
    if not underexplored:
        return None
    plan = min(underexplored, key=lambda key: (counts[key], space.plan_ids.index(key)))
    fresh = [a for a in by_plan[plan] if a.action_hash not in seen]
    return Proposal(
        rng.choice(fresh or by_plan[plan]),
        "stratified_initialization",
        f"plan={plan}; initialized={counts[plan]}; target="
        f"{min(settings.initial_per_plan, sizes[plan])}",
    )


def _uniform_initialization(
    space: SearchSpace,
    available: Sequence[CandidateSpec],
    taken: set[str],
    settings: SearchSettings,
) -> Proposal | None:
    """No-memory baseline, then ``initial_random`` uniform draws over the space.

    Pending reservations count as taken, so controllers started one after the
    other each draw a fresh action. The draw is seeded by (seed, #taken), which
    is deterministic and differs for each successive reservation.
    """
    fresh = [a for a in available if a.action_hash not in taken]
    baseline = [a for a in fresh if a.plan_id == "no_memory_v1"]
    if baseline:
        return Proposal(
            baseline[0], "baseline_no_memory", "no-memory baseline; not counted"
        )
    drawn = sum(
        1
        for action in space.candidates
        if action.plan_id != "no_memory_v1" and action.action_hash in taken
    )
    pool = [a for a in fresh if a.plan_id != "no_memory_v1"]
    if drawn >= settings.initial_random or not pool:
        return None
    rng = random.Random(f"uniform-init:{settings.seed}:{drawn}")
    return Proposal(
        rng.choice(pool),
        "uniform_initialization",
        f"draw={drawn + 1}/{settings.initial_random}; available={len(pool)}",
    )


def _qlognehvi(
    space: SearchSpace,
    observations: Sequence[SearchData],
    candidates: Sequence[CandidateSpec],
    settings: SearchSettings,
) -> tuple[int, float]:
    """Fit two mixed-variable GPs, then score finite candidates with q=1."""
    try:
        import torch
        from botorch.acquisition.multi_objective.logei import (
            qLogNoisyExpectedHypervolumeImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.mlls.sum_marginal_log_likelihood import (
            SumMarginalLogLikelihood,
        )
    except ImportError as error:
        raise ImportError(
            "optional CPU optimizer dependencies are missing; install "
            "requirements-trimem-optimizer.txt in an isolated environment"
        ) from error

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Preserve the embedding application's CPU RNG and never initialize CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(settings.seed)
            tensor_args = {"device": "cpu", "dtype": torch.double}
            x = torch.tensor(
                [space.encode(row.action) for row in observations], **tensor_args
            )
            means, variances = objective_arrays(observations, settings)
            y = torch.tensor(means, **tensor_args)
            yvar = torch.tensor(variances, **tensor_args)
            models = []
            for column in range(2):
                kwargs = {
                    "train_X": x,
                    "train_Y": y[:, column : column + 1],
                    "train_Yvar": yvar[:, column : column + 1],
                    "outcome_transform": Standardize(m=1),
                }
                if space.categorical_dims:
                    model = MixedSingleTaskGP(
                        cat_dims=list(space.categorical_dims), **kwargs
                    )
                else:
                    model = SingleTaskGP(**kwargs)
                models.append(model)
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": settings.max_fit_iterations}},
            )
            model.eval()
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([settings.mc_samples]), seed=settings.seed
            )
            acquisition = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=list(settings.reference_point),
                X_baseline=x,
                sampler=sampler,
                prune_baseline=False,
            )
            scored: list[float] = []
            with torch.no_grad():
                for start in range(0, len(candidates), settings.acquisition_batch_size):
                    batch = candidates[start : start + settings.acquisition_batch_size]
                    query = torch.tensor(
                        [space.encode(a) for a in batch], **tensor_args
                    )
                    scores = acquisition(query.unsqueeze(1)).reshape(-1)
                    scored.extend(scores.tolist())
            if not all(math.isfinite(value) for value in scored):
                raise FloatingPointError("acquisition returned non-finite scores")
            best = max(range(len(scored)), key=scored.__getitem__)
            return best, scored[best]
    finally:
        torch.set_num_threads(previous_threads)
