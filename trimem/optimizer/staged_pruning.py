"""Staged profiling with 3-D Pareto pruning (syftr-style, improved; user decision 2026-10-19).

Pure module (no I/O, no botorch): the caller runs the episodes and owns its state file. Used by the τ² pool round
(``trimem.experiments.tau2.pool_profile``); written to be integrated into the ALFWorld / ScienceWorld pool round
(``bo_pool_driver``) afterwards.

Protocol (``PROTOCOL``):

* Nested checkpoints. Every staged configuration is evaluated on nested query subsets ``S_1 ⊂ S_2 ⊂ ... ⊂ S_K``
  (τ²: 50 -> 100 -> 178 tasks, each task with all its trials; pool datasets: 50 -> 100 -> 200 of each
  configuration's own stratified sample). The subsets are prefixes of ONE stratified order (``stage_order``):
  every prefix is (within one query per stratum) proportional to the strata, and prefixes are nested by
  construction. The seed is the caller's (record it).
* Synchronous per BO batch: stage s runs for all live configurations of the batch, then the batch is decided.
* Prune rule at every checkpoint before the last (fixed checkpoints only, no continuous peeking; v2, user decision
  2026-10-20): the configuration's OPTIMISTIC corner ``(quality + z SE, tokens - z SE, latency - z SE)`` (z = 1.96)
  is compared with two kinds of comparators. (a) FULLY evaluated configurations: their POINT estimates (as in v1).
  (b) STAGE comparators -- NoMemory and every other configuration evaluated on at least the same checkpoint
  (pruned ones included: their evidence on that subset stands), each restricted to the SAME stage subset: their
  PESSIMISTIC corner ``(quality - z SE, tokens + z SE, latency + z SE)``. If any comparator weakly dominates the
  corner (quality >=, tokens <=, latency <=), the configuration is pruned. Corner-vs-corner keeps the rule valid
  before anything is fully evaluated (round 0).
* Nothing but NoMemory is protected (v2): the initial design round (plan defaults + Sobol) is staged and pruned
  like every BO round. NoMemory is never pruned (router candidate and baseline).
* Plan guard (``guard_plans``): a checkpoint never prunes the last live configuration of a plan that has no fully
  evaluated configuration; the configuration of that plan with the best optimistic corner is kept (logged), so
  every plan keeps >= 1 fully evaluated configuration and the router's best-per-plan candidate always exists.
* Pruned configurations remain BO observations with their partial-n means and their (larger) per-configuration SE
  as heteroscedastic GP noise (``gp_noise``, same form as the unpaired path:
  ``session_bo_agentenv.load_points_unpaired``). Selection (Opt-OneConfig / SLOs) and router candidates use fully
  evaluated configurations only.
* Budget: the total episode budget of the unpruned plan is kept; episodes saved by pruning buy more BO
  configurations (``next_batch_size``), capped at ``max_configs``. A batch is only started when every member can
  still be evaluated in full within the budget, so the budget is never exceeded.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace

PROTOCOL = "staged-pareto-prune-v2"  # v1: round 0 protected, front = full points only
Z = 1.96


# ------------------------------------------------------------------ nested stratified stages


def stage_order(
    ids: Sequence[str], stratum_of: Callable[[str], str] | dict, seed: int | str
) -> list[str]:
    """One stratified order of ``ids`` whose every prefix is proportional to the strata (within one id per
    stratum): ids are shuffled within their stratum (``random.Random(f"{seed}|{stratum}")``) and the j-th of a
    stratum of size N_h gets the position key (j + 0.5) / N_h; ties -> stratum name. Nested stages are prefixes."""
    get = stratum_of.__getitem__ if isinstance(stratum_of, dict) else stratum_of
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be distinct")
    by: dict[str, list[str]] = {}
    for i in ids:
        by.setdefault(str(get(i)), []).append(i)
    keyed = []
    for h, xs in by.items():
        xs = sorted(xs)
        random.Random(f"{seed}|{h}").shuffle(xs)
        keyed += [((j + 0.5) / len(xs), h, j, x) for j, x in enumerate(xs)]
    return [x for *_, x in sorted(keyed)]


def stage_subsets(order: Sequence[str], sizes: Sequence[int]) -> list[list[str]]:
    """Nested prefixes of ``order`` for strictly increasing ``sizes``; the last size must be the whole order."""
    sizes = list(sizes)
    if (
        not sizes
        or any(b <= a for a, b in zip(sizes, sizes[1:], strict=False))
        or sizes[0] < 2
    ):
        raise ValueError("stage sizes must be strictly increasing and >= 2")
    if sizes[-1] != len(order):
        raise ValueError(
            f"the last stage ({sizes[-1]}) must be the full set ({len(order)})"
        )
    return [list(order[:n]) for n in sizes]


def seed_of(*parts: str) -> int:
    """Deterministic seed from labels (record it with the stages)."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ dominance / prune decision


@dataclass(frozen=True)
class Obj:
    """Three objectives of one configuration: quality (max), tokens (min), latency (min); with SEs and n."""

    quality: float
    tokens: float
    latency: float
    quality_se: float = 0.0
    tokens_se: float = 0.0
    latency_se: float = 0.0
    n: int = 0


def optimistic_corner(o: Obj, z: float = Z) -> tuple[float, float, float]:
    return (
        o.quality + z * o.quality_se,
        o.tokens - z * o.tokens_se,
        o.latency - z * o.latency_se,
    )


def weakly_dominates(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> bool:
    """a is at least as good as b in every objective (quality max, tokens min, latency min)."""
    return a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2]


def point(o: Obj) -> tuple[float, float, float]:
    return (o.quality, o.tokens, o.latency)


def pareto_front(full: dict[str, Obj]) -> list[str]:
    """Keys of the (strict) Pareto front of fully evaluated configurations' point estimates (reporting)."""

    def strictly(a, b):
        return weakly_dominates(a, b) and a != b

    return [
        k
        for k, o in full.items()
        if not any(strictly(point(p), point(o)) for j, p in full.items() if j != k)
    ]


def pessimistic_corner(o: Obj, z: float = Z) -> tuple[float, float, float]:
    return (
        o.quality - z * o.quality_se,
        o.tokens + z * o.tokens_se,
        o.latency + z * o.latency_se,
    )


FULL_POINT = "full-point"
STAGE_PESSIMISTIC = "stage-pessimistic"


@dataclass(frozen=True)
class Decision:
    prune: bool
    stage: int  # checkpoint size n at which the decision was taken
    corner: tuple[float, float, float]
    dominated_by: (
        str | None
    )  # key of the first dominating comparator (full ones first, then stage ones)
    reason: str
    # every dominating comparator: (key, kind FULL_POINT | STAGE_PESSIMISTIC, its point / pessimistic corner)
    dominators: tuple = ()
    comparators: int = 0  # comparators considered (full + stage)
    guarded: bool = False  # kept by the plan guard although dominated

    def as_dict(self) -> dict:
        return asdict(self)


def decide(
    o: Obj,
    full: dict[str, Obj],
    *,
    stage: int,
    last: bool,
    protected: bool = False,
    z: float = Z,
    partial: dict[str, Obj] | None = None,
) -> Decision:
    """Prune decision at a checkpoint. Never at the last checkpoint, never for protected configurations (NoMemory).
    ``full``: fully evaluated configurations (POINT estimates); ``partial``: stage comparators measured on the SAME
    stage subset as ``o`` (NoMemory restricted to it, other configurations evaluated at >= this stage): their
    PESSIMISTIC corners. All dominators are recorded; ``dominated_by`` is the first (full keys sorted, then stage
    keys sorted)."""
    c = optimistic_corner(o, z)
    partial = partial or {}
    n_cmp = len(full) + len(partial)
    if last or protected:
        return Decision(
            False, stage, c, None, "last stage" if last else "protected", (), n_cmp
        )
    doms = [
        (k, FULL_POINT, point(full[k]))
        for k in sorted(full)
        if weakly_dominates(point(full[k]), c)
    ]
    doms += [
        (k, STAGE_PESSIMISTIC, pessimistic_corner(partial[k], z))
        for k in sorted(partial)
        if weakly_dominates(pessimistic_corner(partial[k], z), c)
    ]
    if doms:
        kind = doms[0][1]
        return Decision(
            True,
            stage,
            c,
            doms[0][0],
            f"optimistic corner (z={z}) weakly dominated by "
            + (
                "a fully evaluated configuration's point estimate"
                if kind == FULL_POINT
                else "the pessimistic corner of a comparator on the same stage subset"
            ),
            tuple(doms),
            n_cmp,
        )
    return Decision(False, stage, c, None, "not dominated", (), n_cmp)


def guard_plans(
    decisions: dict[str, Decision],
    objs: dict[str, Obj],
    plan_of: dict[str, str],
    covered: set[str],
    z: float = Z,
) -> dict[str, Decision]:
    """Plan guard: for every plan NOT in ``covered`` (plans with a fully evaluated configuration or a live
    configuration outside this checkpoint's decisions) whose configurations decided here would ALL be pruned, keep
    the one with the best optimistic corner (quality max, then tokens min, latency min, key) -- so a plan always
    keeps >= 1 configuration that reaches full evaluation (the router's best-per-plan candidate). Returns new
    decisions; the kept one has ``prune=False, guarded=True`` and its dominators recorded."""
    out = dict(decisions)
    by: dict[str, list[str]] = {}
    for k in sorted(decisions):
        by.setdefault(plan_of[k], []).append(k)
    for plan, ks in by.items():
        if plan in covered or not all(decisions[k].prune for k in ks):
            continue

        def rank(k: str) -> tuple:
            c = optimistic_corner(objs[k], z)
            return (-c[0], c[1], c[2], k)

        keep = min(ks, key=rank)
        d = decisions[keep]
        out[keep] = replace(
            d,
            prune=False,
            guarded=True,
            reason=f"plan guard: last live configuration of plan {plan} without a fully evaluated one "
            f"(would be pruned: {d.reason})",
        )
    return out


# ------------------------------------------------------------------ GP observation of partial configurations


def log_var(mean: float, se: float) -> float:
    """Delta-method variance of ln(mean) (the cost objectives are -ln mean tokens / latency)."""
    if mean <= 0:
        raise ValueError("log of a non-positive mean")
    return (se / mean) ** 2


def gp_noise(o: Obj, floor: float = 1e-8) -> tuple[float, float, float]:
    """Heteroscedastic GP noise of an observation (full or partial-n): (SE_q^2, var ln tokens, var ln latency) --
    the same form as ``session_bo_agentenv.load_points_unpaired``; a partial configuration's larger SE enters
    the GP as larger noise."""
    return (
        max(o.quality_se**2, floor),
        max(log_var(o.tokens, o.tokens_se), floor),
        max(log_var(o.latency, o.latency_se), floor),
    )


# ------------------------------------------------------------------ budget


@dataclass(frozen=True)
class Budget:
    total_episodes: int  # the unpruned plan's profiling episodes (every configuration incl. round 0; NoMemory outside)
    full_cost: int  # episodes of one configuration evaluated in full (queries x trials)
    max_configs: int

    @classmethod
    def unpruned(cls, n_configs: int, full_cost: int, max_configs: int) -> Budget:
        return cls(n_configs * full_cost, full_cost, max_configs)


def episodes_used(stage_reached: dict[str, int], trials: int = 1) -> int:
    """Episodes spent: sum over configurations of (queries of the last stage run) x trials."""
    return sum(n * trials for n in stage_reached.values())


def next_batch_size(budget: Budget, used: int, n_configs: int, batch: int) -> int:
    """Configurations the next BO batch may start: at most ``batch``, at most ``max_configs - n_configs``, and only
    as many as can each still be evaluated in FULL within the remaining budget (pruning can only save)."""
    if used > budget.total_episodes:
        raise ValueError(f"budget exceeded: {used} > {budget.total_episodes}")
    return max(
        0,
        min(
            batch,
            budget.max_configs - n_configs,
            (budget.total_episodes - used) // budget.full_cost,
        ),
    )


def stage_cost(sizes: Sequence[int], trials: int = 1) -> list[int]:
    """Incremental episodes of each stage (nested: stage s adds sizes[s] - sizes[s-1] queries)."""
    return [(b - a) * trials for a, b in zip([0, *sizes[:-1]], sizes, strict=True)]
