"""task-complementary-v1: per-query hypervolume improvement as the DLS score (paper eq. task-complementarity).

C(x | S) = sum_q p_q [H_q(S u {x}) - H_q(S)], H_q = exact 3-D hypervolume of the normalized (quality, tokens,
latency) vectors on TRAIN query q, reference point 0. Vectors of S and of candidates both come from the same member
of a configuration-query response model (profiling_response.ResponseModel); A_TC = member mean of C. Pure: no IO,
no model fitting.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

PROTOCOL = "task-complementary-v1"
STATUSES = ("active", "cold_start", "degraded")
OPTIONS = {
    "members": 32,
    "trees": 64,
    "panel_size": 256,
    "epoch_rounds": 1,  # gate re-evaluated every round (a degraded round-1 gate must not idle a short campaign)
    "budget": 4096,  # distinct A_TC evaluations of the DLS walks per decision (arbitration is unbounded)
    "specialist_starts": 2,
    "pool_starts": 8,
    "seed": 20261027,
}
_EPS = 1e-12
# tc-complementary-v1 (user 2026-10-27): a dls pick proposed by an active scorer is pruned at a checkpoint only when
# the response model, refit with the pick's own observed prefix, no longer gives it a plausible complementary gain:
# the 90% member quantile of A_TC is at most PRUNE_EPS (about 1% of the A_TC of a typical pick, ~2e-3).
PRUNE_RULE = "tc-complementary-v1"
PRUNE_EPS = 1e-5
PRUNE_QUANTILE = 0.9


def prune_verdict(
    member_gains, *, eps: float = PRUNE_EPS, quantile: float = PRUNE_QUANTILE
) -> dict:
    """Member-wise A_TC of one configuration -> {prune, a_tc, a_tc_q10, a_tc_q90, reason}."""
    g = np.asarray(member_gains, dtype=float)
    if g.ndim != 1 or not len(g) or not np.isfinite(g).all():
        raise ValueError("member gains must be a finite 1-D array")
    q = float(np.quantile(g, quantile))
    prune = q <= eps
    return {
        "rule": PRUNE_RULE,
        "prune": bool(prune),
        "a_tc": float(g.mean()),
        "a_tc_q10": float(np.quantile(g, 0.1)),
        f"a_tc_q{int(round(quantile * 100))}": q,
        "eps": eps,
        "reason": f"{PRUNE_RULE}: q{int(round(quantile * 100))} A_TC {q:.3g} "
        + ("<=" if prune else ">")
        + f" {eps:g} with the observed prefix",
    }


# ---- exact 3-D hypervolume -----------------------------------------------------------------------------------


def _area2d(xy: np.ndarray) -> np.ndarray:
    """Dominated area of [..., n, 2] points w.r.t. the origin (maximization)."""
    order = np.argsort(-xy[..., 0], axis=-1, kind="stable")
    xs = np.take_along_axis(xy[..., 0], order, axis=-1)
    ys = np.maximum.accumulate(np.take_along_axis(xy[..., 1], order, axis=-1), axis=-1)
    nxt = np.concatenate([xs[..., 1:], np.zeros(xs.shape[:-1] + (1,))], axis=-1)
    return np.sum((xs - nxt) * ys, axis=-1)


def _hv(points: np.ndarray) -> np.ndarray:
    """Hypervolume of [..., K, 3] points in [0, 1]^3 w.r.t. the origin; zero rows are padding.

    z-slab sweep: the slab between the i-th and (i+1)-th largest z is dominated by the i+1 highest-z points, whose
    2-D staircase area is computed for every i at once on the points pre-sorted by x (O(K^2), no per-slab sort)."""
    points = np.asarray(points, dtype=float)
    k = points.shape[-2]
    if k == 0:
        return np.zeros(points.shape[:-2])
    xo = np.argsort(-points[..., 0], axis=-1, kind="stable")
    srt = np.take_along_axis(points, xo[..., None], axis=-2)  # x descending
    x, y, z = srt[..., 0], srt[..., 1], srt[..., 2]
    zs = -np.sort(-z, axis=-1)  # z descending
    znext = np.concatenate([zs[..., 1:], np.zeros(zs.shape[:-1] + (1,))], axis=-1)
    # member[..., i, j]: point j (x order) lies in the top-(i+1) z set, i.e. z_j >= zs_i (ties share a zero slab)
    member = z[..., None, :] >= zs[..., :, None]
    stair = np.maximum.accumulate(np.where(member, y[..., None, :], 0.0), axis=-1)
    dx = x - np.concatenate([x[..., 1:], np.zeros(x.shape[:-1] + (1,))], axis=-1)
    area = np.einsum("...ij,...j->...i", stair, dx)
    return np.einsum("...i,...i->...", zs - znext, area)


def hv3d(points: np.ndarray) -> float:
    return float(_hv(np.clip(np.asarray(points, dtype=float), 0, None)))


def hvi(front: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Exact improvement of adding ``cand`` [..., 3] to ``front`` [..., K, 3] (leading axes broadcast)."""
    front, cand = np.asarray(front, dtype=float), np.asarray(cand, dtype=float)
    clipped = np.minimum(front, cand[..., None, :])
    out = np.prod(cand, axis=-1) - _hv(clipped)
    out = np.where(np.abs(out) < _EPS, 0.0, out)
    return np.maximum(out, 0.0)


# ---- normalization -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scales:
    """Campaign-fixed ln-cost bounds (1% / 99% of round-0 TRAIN episodes); never refit."""

    l_T: float
    u_T: float
    l_L: float
    u_L: float

    @classmethod
    def freeze(cls, rows: list[dict]) -> Scales:
        if not rows:
            raise ValueError("cannot freeze scales without TRAIN rows")
        lt = np.log([max(float(r["tokens"]), 1e-9) for r in rows])
        ll = np.log([max(float(r["latency_ms"]), 1e-9) for r in rows])
        bounds = []
        for v in (lt, ll):
            lo, hi = float(np.quantile(v, 0.01)), float(np.quantile(v, 0.99))
            bounds += [lo, hi if hi - lo > 1e-9 else lo + 1.0]
        return cls(*bounds)

    def to_dict(self) -> dict:
        return {"l_T": self.l_T, "u_T": self.u_T, "l_L": self.l_L, "u_L": self.u_L}

    @classmethod
    def from_dict(cls, d: dict) -> Scales:
        return cls(float(d["l_T"]), float(d["u_T"]), float(d["l_L"]), float(d["u_L"]))

    def gate_norm(self) -> dict:
        """``profiling_response.utility`` scales whose unit-vector preferences give y-space errors."""
        return {
            "quality": {"s": 1.0},
            "tokens": {"s": self.u_T - self.l_T},
            "latency_ms": {"s": self.u_L - self.l_L},
        }


def normalize(raw: np.ndarray, scales: Scales) -> np.ndarray:
    """[..., 3] raw (Q, T, L) -> [..., 3] in [0, 1], larger is better."""
    raw = np.asarray(raw, dtype=float)
    q = np.clip(raw[..., 0], 0, 1)
    t = (scales.u_T - np.log(np.maximum(raw[..., 1], 1e-9))) / (scales.u_T - scales.l_T)
    lat = (scales.u_L - np.log(np.maximum(raw[..., 2], 1e-9))) / (
        scales.u_L - scales.l_L
    )
    return np.stack([q, np.clip(t, 0, 1), np.clip(lat, 0, 1)], axis=-1)


# ---- objective -----------------------------------------------------------------------------------------------


def nondominated(front: np.ndarray) -> np.ndarray:
    """[..., K, 3] -> [..., K', 3]: each slice's non-dominated points first, zero padding, K' = max count.
    Dominated points add no volume, so hypervolumes and improvements are unchanged."""
    ge = np.all(front[..., None, :, :] >= front[..., :, None, :], axis=-1)
    gt = np.any(front[..., None, :, :] > front[..., :, None, :], axis=-1)
    keep = ~np.any(ge & gt, axis=-1)  # [..., K]: no j dominates i
    keep &= np.any(front > 0, axis=-1)
    order = np.argsort(~keep, axis=-1, kind="stable")
    out = np.take_along_axis(
        np.where(keep[..., None], front, 0.0), order[..., None], axis=-2
    )
    return out[..., : max(int(keep.sum(axis=-1).max(initial=0)), 1), :]


class TCObjective:
    """Per-query fronts of S under each member; ``front_raw`` is [M, S, Q, 3] raw predictions."""

    def __init__(
        self, front_raw: np.ndarray, p: np.ndarray, scales: Scales, *, _full=None
    ):
        self.p = np.asarray(p, dtype=float)
        if self.p.ndim != 1 or np.any(self.p <= 0) or not np.isclose(self.p.sum(), 1):
            raise ValueError("query weights must be positive and sum to one")
        self.scales = scales
        if _full is None:
            front_raw = np.asarray(front_raw, dtype=float)
            if front_raw.ndim != 4 or front_raw.shape[2] != len(self.p):
                raise ValueError("front must be [members, configs, queries, 3]")
            _full = normalize(front_raw, scales).transpose(0, 2, 1, 3)  # [M,Q,S,3]
        self.full = _full  # every member of S, in order (exclusive contributions)
        self.front = nondominated(_full)  # pruned per (member, query) for scoring

    def per_query(self, raw: np.ndarray, chunk: int = 2) -> np.ndarray:
        """[M, C, Q, 3] raw candidates -> [M, C, Q] per-query HVI."""
        y = normalize(raw, self.scales)
        parts = [
            hvi(self.front[:, None], y[:, i : i + chunk])
            for i in range(0, y.shape[1], chunk)
        ]
        return np.concatenate(parts, axis=1) if parts else np.zeros(y.shape[:3])

    def member_gain(self, raw: np.ndarray) -> np.ndarray:
        return np.einsum("mcq,q->mc", self.per_query(raw), self.p)

    def gain(self, raw: np.ndarray) -> np.ndarray:
        return self.member_gain(raw).mean(axis=0)

    def include(self, raw_one: np.ndarray) -> TCObjective:
        raw_one = np.asarray(raw_one, dtype=float)
        if raw_one.ndim != 4 or raw_one.shape[1] != 1:
            raise ValueError("include takes exactly one configuration [M, 1, Q, 3]")
        y = normalize(raw_one, self.scales).transpose(0, 2, 1, 3)
        return TCObjective(
            None, self.p, self.scales, _full=np.concatenate([self.full, y], axis=2)
        )

    def exclusive(self) -> np.ndarray:
        """[S]: member mean of sum_q p_q [H_q(S) - H_q(S minus s)]."""
        out = []
        for s in range(self.full.shape[2]):
            rest = nondominated(np.delete(self.full, s, axis=2))
            out.append(
                np.einsum("mq,q->m", hvi(rest, self.full[:, :, s]), self.p).mean()
            )
        return np.array(out)


# ---- support -------------------------------------------------------------------------------------------------


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def tc_supported(
    config: dict, observed: list[dict], levels: Callable[[str, str], list]
) -> bool:
    """Plan seen in training; numeric knobs at most one grid level beyond the same-plan observed range;
    categorical values seen under any plan. Looser than profiling_driver.model_supported (which stays)."""
    same = [c for c in observed if c["plan"] == config["plan"]]
    if not same:
        return False
    for knob, value in config.items():
        if knob == "plan":
            continue
        if _numeric(value):
            seen = [c[knob] for c in same if knob in c and _numeric(c[knob])]
            if not seen:
                return False
            grid = list(levels(config["plan"], knob))
            if value in grid and all(v in grid for v in seen):
                idx = [grid.index(v) for v in seen]
                if not min(idx) - 1 <= grid.index(value) <= max(idx) + 1:
                    return False
            elif not min(seen) <= value <= max(seen):
                return False
        elif value not in {c.get(knob) for c in observed}:
            return False
    return True


# ---- DLS scorer ----------------------------------------------------------------------------------------------


@dataclass
class TCScorer:
    """A_TC for the dls slot. ``s_configs`` = completed configurations + NoMemory (S); ``extra`` = this batch's
    earlier picks (kriging believers), entered as predicted front members only."""

    model: Any
    qids: list[str]
    p: np.ndarray
    scales: Scales
    s_configs: list[dict]
    observed: list[dict]
    levels: Callable[[str, str], list]
    status: str
    key: Callable[[dict], str]
    strata: dict[str, str] | None = None
    budget: int = OPTIONS["budget"]
    info: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"unknown task-complementary status {self.status!r}")

    @property
    def active(self) -> bool:
        return self.status == "active"

    def _predict(self, configs: list[dict]) -> np.ndarray:
        todo = [c for c in configs if self.key(c) not in self._cache]
        if todo:
            pred = self.model.predict(todo, list(self.qids))
            for i, c in enumerate(todo):
                self._cache[self.key(c)] = pred[:, i : i + 1]
        return np.concatenate([self._cache[self.key(c)] for c in configs], axis=1)

    def objective(self, extra: list[dict]) -> TCObjective:
        keys, front = set(), []
        for c in list(self.s_configs) + list(extra):
            if self.key(c) not in keys:
                keys.add(self.key(c))
                front.append(c)
        return TCObjective(self._predict(front), self.p, self.scales)

    def score_fn(self, extra: list[dict]):
        obj = self.objective(extra)
        values: dict[str, float] = {}
        audit = {
            "budget": self.budget,
            "evaluations": 0,
            "unsupported": 0,
            "budget_exhausted": False,
            "front_size": obj.full.shape[2],
            "front_nondominated_max": obj.front.shape[2],
            "seconds": 0.0,
        }

        def f(cfgs: list[dict], bounded: bool = True) -> list[float]:
            started = time.monotonic()
            todo, seen = [], set()
            for c in cfgs:
                k = self.key(c)
                if k in values or k in seen:
                    continue
                seen.add(k)
                if not tc_supported(c, self.observed, self.levels):
                    values[k] = -math.inf
                    audit["unsupported"] += 1
                elif bounded and audit["evaluations"] >= self.budget:
                    audit["budget_exhausted"] = True
                else:
                    todo.append(c)
                    audit["evaluations"] += 1
            for i in range(0, len(todo), 8):
                chunk = todo[i : i + 8]
                for c, v in zip(chunk, obj.gain(self._predict(chunk)), strict=True):
                    values[self.key(c)] = float(v)
            audit["seconds"] += time.monotonic() - started
            return [values.get(self.key(c), -math.inf) for c in cfgs]

        return f, audit

    def specialists(self, extra: list[dict], k: int) -> list[dict]:
        """Up to k members of S with the largest positive exclusive per-query contribution (NoMemory excluded)."""
        obj = self.objective(extra)
        ex = obj.exclusive()[: len(self.s_configs)]
        ranked = sorted(
            range(len(self.s_configs)),
            key=lambda i: (-ex[i], self.key(self.s_configs[i])),
        )
        return [
            self.s_configs[i]
            for i in ranked
            if ex[i] > _EPS and self.s_configs[i]["plan"] != "NoMemory"
        ][:k]

    def explain(self, config: dict, extra: list[dict]) -> dict:
        obj = self.objective(extra)
        per_q = obj.per_query(self._predict([config]))[:, 0]  # [M, Q]
        member = per_q @ obj.p
        mean_q = per_q.mean(axis=0)
        out = {
            "a_tc": float(member.mean()),
            "a_tc_q10": float(np.quantile(member, 0.1)),
            "a_tc_q50": float(np.quantile(member, 0.5)),
            "a_tc_q90": float(np.quantile(member, 0.9)),
            "positive_queries": int(np.sum(mean_q > _EPS)),
            "queries": len(self.qids),
            "support": "strict-or-relaxed"
            if tc_supported(config, self.observed, self.levels)
            else "unsupported",
        }
        if self.strata is not None:
            by: dict[str, float] = {}
            for q, v, w in zip(self.qids, mean_q, obj.p, strict=True):
                by[self.strata[q]] = by.get(self.strata[q], 0.0) + float(v * w)
            out["by_stratum"] = by
        return out
