"""Finite-domain score-function PR, optimizing E[qLogNEHVI] directly (v0.4.3).

Categorical fields keep bo_pr's ``softmax((theta - 0.5) / tau)``.

Ordinal fields use the Generalized PR discrete reparameterization (Zhang et
al. 2026, Table 1), extended to full support as an ordered-logit model:

    P(Z >= d_{k+1}) = sigmoid((u - m_k) / tau),   m_k = (d_k + d_{k+1}) / 2.

When only one threshold is unsaturated, this is exactly the paper's two-level
``D(theta)``. The levels ``d_k`` are the GP distance coordinate (log2 for the
count fields), rescaled so that the mean level gap is 1. For an equidistant
field this equals bo_pr's integer scale, so tau = 0.1 keeps its meaning.

The optimizer follows bo_pr ``gen_candidates_torch``: every theta lies in
[0, 1], Adam lr = 0.025, 200 iterations. v0.4.4 default: no lr decay (bo_pr's
default). ``decay=True`` gives bo_pr's option: a fresh Adam with lr / i**0.7
after iteration i.

Stagnation never stops a run; every run continues to its step limit. Steps with
no new submittable key are counted (``stagnant_steps``). v0.4.4 default: no
re-anchoring. ``patience=k`` re-anchors after k stagnant steps at the best
submittable draw that has not been an anchor yet (Sobol if none).
"""

import copy
import math
import time
from dataclasses import dataclass

import torch

from .plan_codec import feature_grid
from .plan_schema import FIELD_NAMES, FIELDS

TAU = 0.1
LR = 0.025
MAX_STEPS = 200
MAX_SAMPLES = 128
MAX_STARTS = 25


@dataclass(frozen=True)
class PRStart:
    source: str
    config: dict


class ProposalBudget:
    def __init__(self, max_unique=10000, seconds=None):
        # seconds=None: no proposal time limit (user 2026-11-03); the unique
        # acquisition-input budget still bounds the work.
        self.max_unique, self.seconds = max_unique, seconds
        self.started = time.monotonic()
        self.keys = set()

    def check(self):
        if self.seconds is not None and time.monotonic() - self.started >= self.seconds:
            raise RuntimeError("proposal time budget exhausted")

    def admit(self, keys):
        self.check()
        new = set(keys) - self.keys
        if len(self.keys) + len(new) > self.max_unique:
            raise RuntimeError("unique acquisition budget exhausted")
        self.keys.update(new)


def ordinal_levels(space, name):
    """Level positions d_k in the GP distance coordinate, mean gap 1."""
    i = FIELD_NAMES.index(name)
    full, grid = FIELDS[i][2], feature_grid(i)
    g = [grid[full.index(v)] for v in space.domains[name]]
    n = len(g)
    return torch.tensor(
        [(n - 1) * (x - g[0]) / (g[-1] - g[0]) for x in g], dtype=torch.double
    )


def ordinal_log_probs(u, levels, tau=TAU):
    """Full-support ordered-logit log P(Z = d_k) for k = 0..n-1.

    log(sigmoid(a) - sigmoid(b)) = logsig(a) + logsig(-b) + log1p(-exp(b - a)),
    with a = (u - m_{k-1}) / tau and b = (u - m_k) / tau. The last term does
    not depend on u.
    """
    mids = (levels[:-1] + levels[1:]) / 2
    a = (u - mids) / tau
    inf = torch.tensor([math.inf], dtype=torch.double)
    upper = torch.cat([inf, a])
    lower = torch.cat([a, -inf])
    gaps = (mids[:-1] - mids[1:]) / tau
    const = torch.cat(
        [
            torch.zeros(1, dtype=torch.double),
            torch.log1p(-torch.exp(gaps)),
            torch.zeros(1, dtype=torch.double),
        ]
    )
    return (
        torch.nn.functional.logsigmoid(upper)
        + torch.nn.functional.logsigmoid(-lower)
        + const
    )


class PRDistribution(torch.nn.Module):
    def __init__(self, space, start, *, seed=0, tau=TAU):
        super().__init__()
        self.space, self.tau = space, tau
        self.generator = torch.Generator().manual_seed(seed)
        self.theta = torch.nn.ParameterDict()
        self.levels = {
            name: ordinal_levels(space, name)
            for name, kind, _ in FIELDS
            if kind == "ord" and len(space.domains[name]) > 1
        }
        self.reset(start, create=True)

    def reset(self, start, *, create=False):
        """Soft initialization around ``start``; E[changed fields] ~= 1.5."""
        space, tau = self.space, self.tau
        self.start = space.canonicalize(start)
        m = sum(
            v is not None and len(space.domains[n]) > 1 for n, v in self.start.items()
        )
        self.rho = min(0.3, 1.5 / max(m, 1))
        for name, kind, _ in FIELDS:
            values = space.domains[name]
            n = len(values)
            if n == 1:
                continue
            value = self.start[name]
            if kind == "cat":
                p = torch.full((n,), 1 / n, dtype=torch.double)
                if value is not None:
                    p.fill_(self.rho / (n - 1))
                    p[values.index(value)] = 1 - self.rho
                theta = (0.5 + tau * p.log()).clamp(0, 1)
            else:
                # An inactive ordinal has no current value; draw its anchor.
                j = (
                    values.index(value)
                    if value is not None
                    else int(torch.randint(n, (1,), generator=self.generator))
                )
                up = j == 0 or (
                    j < n - 1 and bool(torch.randint(2, (1,), generator=self.generator))
                )
                d = self.levels[name]
                mids = (d[:-1] + d[1:]) / 2
                logit = math.log(self.rho / (1 - self.rho))
                # One neighbor gets probability rho; the far side is negligible.
                u = mids[j] + tau * logit if up else mids[j - 1] - tau * logit
                theta = (u / (n - 1)).clamp(0, 1)
            if create:
                self.theta[name] = torch.nn.Parameter(theta.clone())
            else:
                with torch.no_grad():
                    self.theta[name].copy_(theta)

    def field_log_probs(self, name, kind):
        t = self.theta[name]
        if kind == "cat":
            return torch.log_softmax((t - 0.5) / self.tau, dim=0)
        d = self.levels[name]
        return ordinal_log_probs(t * (len(d) - 1), d, self.tau)

    def sample(self, count):
        raw, terms = [{} for _ in range(count)], {}
        for name, kind, _ in FIELDS:
            values = self.space.domains[name]
            if len(values) == 1:
                for c in raw:
                    c[name] = values[0]
                continue
            logp = self.field_log_probs(name, kind)
            idx = torch.multinomial(
                logp.detach().exp(), count, replacement=True, generator=self.generator
            )
            terms[name] = logp[idx]
            for c, j in zip(raw, idx.tolist()):
                c[name] = values[j]
        configs = [self.space.canonicalize(c) for c in raw]
        logprob = torch.zeros(count, dtype=torch.double)
        for name, term in terms.items():
            active = torch.tensor(
                [c[name] is not None for c in configs], dtype=torch.double
            )
            logprob = logprob + term * active
        return configs, logprob

    def project(self):
        with torch.no_grad():
            for p in self.theta.values():
                p.clamp_(0, 1)


def _optimizer(state, lr, decay):
    """bo_pr: iteration i (1-based) uses lr, then lr / (i - 1)**0.7 if decay."""
    local = state["local"]
    rate = lr / local**0.7 if decay and local >= 1 else lr
    return torch.optim.Adam(state["dist"].parameters(), lr=rate)


def _reanchor(space, state, forbidden, seed):
    anchored = set(state["anchors"])
    ranked = sorted(
        (
            (k, c, r)
            for k, (c, r) in state["seen"].items()
            if k not in forbidden and k not in anchored
        ),
        key=lambda row: (-row[2], row[0]),
    )
    if ranked:
        key, target, reason = ranked[0][0], ranked[0][1], "best unanchored draw"
    else:
        fresh = space.sample(1, seed=seed, avoid=forbidden | anchored)
        if not fresh:
            state["stop"] = "space exhausted"
            return
        target, reason = fresh[0], "sobol"
        key = space.key(target)
    state["dist"].reset(target)
    state["anchors"].append(key)
    state["reanchors"].append({"step": state["step"], "key": key, "reason": reason})
    state.update(local=0, ema=0.0, stalled=0, optimizer=None)


def refine(
    space,
    starts,
    snapshot,
    *,
    seed=0,
    steps=MAX_STEPS,
    samples=MAX_SAMPLES,
    keep=8,
    forbidden=(),
    checkpoint=None,
    seconds=None,
    lr=LR,
    decay=False,
    patience=None,
):
    """Round-robin updates; with an optional time slice every run still advances.

    Checkpoints contain RNG, Adam, prior EMA, anchors and all unique draws.
    Duplicate samples are scored once (snapshot cache) but used repeatedly in
    the Monte Carlo gradient.
    """
    if (
        len(starts) > MAX_STARTS
        or not 1 <= steps <= MAX_STEPS
        or not 2 <= samples <= MAX_SAMPLES
    ):
        raise ValueError("PR limits exceed v0.4.3 protocol")
    forbidden = set(forbidden)
    states = []
    for i, start in enumerate(starts):
        dist = PRDistribution(space, start.config, seed=seed + i * 1009)
        if not list(dist.parameters()):
            continue
        state = {
            "dist": dist,
            "optimizer": None,
            "step": 0,
            "local": 0,
            "ema": 0.0,
            "stalled": 0,
            "stagnant_steps": 0,
            "seen": {},
            "draws": 0,
            "source": start.source,
            "anchors": [space.key(start.config)],
            "reanchors": [],
            "stop": None,
        }
        if checkpoint:
            saved = checkpoint[i]
            if (
                saved["start_key"] != space.key(start.config)
                or saved["source"] != start.source
                or saved.get("snapshot") != getattr(snapshot, "identity", None)
            ):
                raise ValueError("PR checkpoint start identity mismatch")
            dist.load_state_dict(saved["parameters"])
            dist.generator.set_state(saved["rng"])
            dist.start = copy.deepcopy(saved["anchor_config"])
            state.update(
                {
                    k: copy.deepcopy(saved[k])
                    for k in (
                        "step",
                        "local",
                        "ema",
                        "stalled",
                        "stagnant_steps",
                        "seen",
                        "draws",
                        "anchors",
                        "reanchors",
                        "stop",
                    )
                }
            )
            if saved["optimizer"] is not None:
                state["optimizer"] = _optimizer(state, lr, decay)
                state["optimizer"].load_state_dict(saved["optimizer"])
        states.append(state)
    began = time.monotonic()
    budget_stop = None
    for _ in range(steps):
        for run, state in enumerate(states):
            if state["step"] >= steps or state["stop"]:
                continue
            if seconds is not None and time.monotonic() - began >= seconds:
                budget_stop = "PR time slice exhausted"
                break
            configs, logp = state["dist"].sample(samples)
            try:
                reward = torch.tensor(snapshot.score(configs), dtype=torch.double)
            except RuntimeError as exc:
                if "budget" not in str(exc):
                    raise
                budget_stop = str(exc)
                break
            if not torch.isfinite(reward).all():
                raise RuntimeError("nonfinite PR rewards")
            local = state["local"]
            baseline = (
                (reward.sum() - reward) / (samples - 1)
                if local == 0
                else state["ema"] / (1 - 0.7**local)
            )
            loss = -((reward - baseline).detach() * logp).mean()
            if decay or state["optimizer"] is None:
                state["optimizer"] = _optimizer(state, lr, decay)
            state["optimizer"].zero_grad()
            loss.backward()
            state["optimizer"].step()
            state["dist"].project()
            state["ema"] = 0.7 * state["ema"] + 0.3 * reward.mean().item()
            keys = [space.key(c) for c in configs]
            fresh = set(keys) - set(state["seen"]) - forbidden
            blocked_fraction = sum(k in forbidden for k in keys) / samples
            for k, c, r in zip(keys, configs, reward.tolist()):
                state["seen"][k] = (c, r)
            state["draws"] += samples
            state["step"] += 1
            state["local"] += 1
            state["stalled"] = (
                state["stalled"] + 1 if not fresh or blocked_fraction >= 0.8 else 0
            )
            state["stagnant_steps"] += not fresh
            # Re-anchor even at the last step so a resumed run stays identical.
            if patience is not None and state["stalled"] >= patience:
                _reanchor(
                    space, state, forbidden, seed + 50000 + run * 1009 + state["step"]
                )
        if budget_stop:
            break
    saved, runs, candidates = [], [], {}
    for start, state in zip(starts, states):
        saved.append(
            {
                **{
                    k: copy.deepcopy(state[k])
                    for k in (
                        "step",
                        "local",
                        "ema",
                        "stalled",
                        "stagnant_steps",
                        "seen",
                        "draws",
                        "anchors",
                        "reanchors",
                        "stop",
                        "source",
                    )
                },
                "start_key": space.key(start.config),
                "anchor_config": copy.deepcopy(state["dist"].start),
                "parameters": copy.deepcopy(state["dist"].state_dict()),
                "snapshot": getattr(snapshot, "identity", None),
                "optimizer": (
                    None
                    if state["optimizer"] is None
                    else copy.deepcopy(state["optimizer"].state_dict())
                ),
                "rng": state["dist"].generator.get_state(),
            }
        )
        top = sorted(
            ((k, c, r) for k, (c, r) in state["seen"].items() if k not in forbidden),
            key=lambda row: (-row[2], row[0]),
        )[:keep]
        for k, c, r in top:
            candidates.setdefault(k, c)
        runs.append(
            {
                "source": state["source"],
                "start": start.config,
                "steps": state["step"],
                "draws": state["draws"],
                "unique_draws": len(state["seen"]),
                "reanchors": len(state["reanchors"]),
                "stagnant_steps": state["stagnant_steps"],
                "reanchor_log": copy.deepcopy(state["reanchors"]),
                "stop": state["stop"]
                or budget_stop
                or ("step limit" if state["step"] >= steps else None),
                "retained": len(top),
            }
        )
    return {
        "candidates": list(candidates.values()),
        "runs": runs,
        "checkpoint": saved,
        "stop": budget_stop,
        "seconds": time.monotonic() - began,
    }
