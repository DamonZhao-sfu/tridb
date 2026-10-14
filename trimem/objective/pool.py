"""D1: per-iteration candidate pool, semantic passes, the 13 candidates and their features.

One tri-modal retrieval builds a pool of M improving descendants for the parent; one SemScore
pass and one SemFilter pass over the pool (TriMEM plan nodes, cached) give every row a score and a
verdict. A candidate (family, k) is the top-k of the pool under the family's rule; (none, 0)
injects nothing. Features are computed exactly from the pool rows, so the objective can score
every candidate without generating anything.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import psycopg

from trimem.objective.space import GRAPH_SPACE, POOL_M_V2, GraphConfig, pool_sql
from trimem.replay.memory import SEM_PREDICATE

FAMILIES = ("trimodal", "hard", "rank")
KS = (1, 3, 5, 8)
POOL_M = 16

POOL_SQL = """
SELECT d.id, d.fitness, s.fitness AS parent_fitness, d.changes, d.code, coalesce(d.tokens, 0),
       1 - (d.emb <=> %(q)s::vector) AS sim, d.session
FROM (SELECT id, vid, fitness FROM attempt
      WHERE task = %(task)s AND cell IS NULL AND emb IS NOT NULL
      ORDER BY emb <=> %(q)s::vector LIMIT 24) s,
     LATERAL trimem.expand(s.vid, 'lineage', 8) e
     JOIN attempt d ON d.vid = e.dst
WHERE d.is_valid AND d.fitness > s.fitness AND d.fitness >= %(floor)s
  AND d.cell IS NULL AND d.code IS NOT NULL
ORDER BY d.fitness - s.fitness DESC, d.id LIMIT %(lim)s
"""
# The predicate is inlined as a literal: the SemFilter/SemScore planner hooks need a constant, and
# psycopg auto-prepares a statement after 5 executions, which turns a bound value into a Param.
_PRED = "'" + SEM_PREDICATE.replace("'", "''") + "'"
SCORE_SQL = f"SELECT id, trimem.sem_score({_PRED}, changes) FROM attempt WHERE id = ANY(%(ids)s)"
FILTER_SQL = f"SELECT id FROM attempt WHERE id = ANY(%(ids)s) AND trimem.sem_filter({_PRED}, changes)"


@dataclass
class PoolRow:
    id: int
    fitness: float
    parent_fitness: float
    changes: str | None
    code: str
    tokens: int
    sim: float
    session: str
    score: float | None = None
    verdict: bool = False

    @property
    def gain(self) -> float:
        return self.fitness - self.parent_fitness


@dataclass
class Candidate:
    family: str
    k: int
    rows: list[PoolRow]
    features: dict = field(default_factory=dict)
    graph: GraphConfig | None = None  # v2 space: the traversal configuration the rows came from

    @property
    def name(self) -> str:
        base = f"{self.family}:{self.k}"
        return base if self.graph is None or self.family == "none" else f"{base}@{self.graph.name}"


@dataclass
class PoolResult:
    rows: list[PoolRow]
    candidates: list[Candidate]
    ret_ms: float
    sem_tokens: int
    sem_ms: float


def _sem_tokens(conn: psycopg.Connection, pv: str) -> int:
    r = conn.execute(
        "SELECT coalesce(sum(prompt_tokens + completion_tokens), 0) FROM trimem.sem_cache "
        "WHERE prompt_version = %s",
        (pv,),
    ).fetchone()
    return int(r[0])


def build_pool(
    conn: psycopg.Connection,
    *,
    task: str,
    q: list[float],
    parent_fitness: float,
    prompt_version: str,
    m: int = POOL_M,
) -> PoolResult:
    t0 = time.perf_counter()
    params = {
        "task": task,
        "q": "[" + ",".join(f"{x:.6g}" for x in q) + "]",
        "floor": 0.8 * parent_fitness if math.isfinite(parent_fitness) else -1e300,  # NaN parent: no floor
        "lim": 3 * m,
    }
    rows, seen = [], set()
    for r in conn.execute(
        POOL_SQL, params
    ).fetchall():  # a row reached from two seeds: keep its best gain
        if r[0] not in seen:
            seen.add(r[0])
            rows.append(PoolRow(*r))
    rows = rows[:m]
    ret_ms = (time.perf_counter() - t0) * 1000
    tok0, t1 = _sem_tokens(conn, prompt_version), time.perf_counter()
    if rows:
        ids = [r.id for r in rows]
        scores = dict(conn.execute(SCORE_SQL, {"ids": ids}).fetchall())
        passed = {r[0] for r in conn.execute(FILTER_SQL, {"ids": ids}).fetchall()}
        for r in rows:
            r.score = scores.get(r.id)
            r.verdict = r.id in passed
    sem_ms = (time.perf_counter() - t1) * 1000
    sem_tokens = _sem_tokens(conn, prompt_version) - tok0
    return PoolResult(rows, candidates(rows), ret_ms, sem_tokens, sem_ms)


def _order(rows: list[PoolRow], family: str) -> list[PoolRow]:
    if family == "trimodal":
        return rows
    if family == "hard":
        return [r for r in rows if r.verdict]
    return sorted(
        rows,
        key=lambda r: (
            -(r.gain * (0.5 + (r.score if r.score is not None else 0.5))),
            r.id,
        ),
    )


def candidates(rows: list[PoolRow]) -> list[Candidate]:
    out = [Candidate("none", 0, [])]
    for fam in FAMILIES:
        ordered = _order(rows, fam)
        for k in KS:
            out.append(Candidate(fam, k, ordered[:k]))
    return out


def featurize(
    c: Candidate,
    *,
    parent_fitness: float,
    iteration: int,
    horizon: int,
    parent_tokens: int,
    parent_session: str | None = None,
) -> dict:
    """Task-agnostic features (no task id), used by every estimator."""
    rows = c.rows
    gains = [r.fitness - parent_fitness for r in rows]
    own = [r.gain for r in rows]
    sims = [r.sim for r in rows]
    scores = [r.score for r in rows if r.score is not None]
    f = {
        "fam_none": float(c.family == "none"),
        "fam_trimodal": float(c.family == "trimodal"),
        "fam_hard": float(c.family == "hard"),
        "fam_rank": float(c.family == "rank"),
        "k": float(len(rows)),
        "k_requested": float(c.k),
        "log1p_k": math.log1p(len(rows)),
        "parent_fitness": parent_fitness,
        "progress": iteration / max(horizon, 1),
        "gain_vs_parent_max": max(gains, default=0.0),
        "gain_vs_parent_mean": float(np.mean(gains)) if gains else 0.0,
        "own_gain_max": max(own, default=0.0),
        "own_gain_mean": float(np.mean(own)) if own else 0.0,
        "sim_max": max(sims, default=0.0),
        "sim_mean": float(np.mean(sims)) if sims else 0.0,
        "sim_gap": (sims[0] - sims[-1]) if len(sims) > 1 else 0.0,
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "verdict_frac": (sum(r.verdict for r in rows) / len(rows)) if rows else 0.0,
        "tokens_injected": float(sum(r.tokens for r in rows)),
        "parent_tokens": float(parent_tokens),
        "other_session_frac": (
            sum(r.session != parent_session for r in rows) / len(rows)
        )
        if rows
        else 0.0,
    }
    if len(rows) > 1:
        g = np.array([r.gain for r in rows])
        f["gain_std"] = float(g.std())
    else:
        f["gain_std"] = 0.0
    return f


FEATURE_ORDER = (
    "fam_none",
    "fam_trimodal",
    "fam_hard",
    "fam_rank",
    "k",
    "log1p_k",
    "parent_fitness",
    "progress",
    "gain_vs_parent_max",
    "gain_vs_parent_mean",
    "own_gain_max",
    "own_gain_mean",
    "sim_max",
    "sim_mean",
    "sim_gap",
    "score_mean",
    "verdict_frac",
    "tokens_injected",
    "parent_tokens",
    "other_session_frac",
    "gain_std",
)


def vector(f: dict) -> np.ndarray:
    return np.array([float(f.get(k, 0.0)) for k in FEATURE_ORDER], dtype=float)


# ---------------------------------------------------------------- v2: graph configuration space
FEATURE_ORDER_V2 = FEATURE_ORDER + ("best_gap", "edge_similar", "dir_up", "hops2", "log_fanout", "pool_n")
NONE_FEATURES = ("parent_fitness", "progress", "best_gap", "parent_tokens")


@dataclass
class PoolsResult:
    pools: dict[str, list[PoolRow]]
    candidates: list[Candidate]
    ret_ms: float
    sem_tokens: int
    sem_ms: float
    union_n: int


def build_pools(conn: psycopg.Connection, *, task: str, q: list[float], parent_fitness: float,
                prompt_version: str, configs: tuple[GraphConfig, ...] = GRAPH_SPACE,
                m: int = POOL_M_V2, require_changes: bool = False) -> PoolsResult:
    """One pool per graph configuration, then ONE SemScore and ONE SemFilter pass over the union."""
    t0 = time.perf_counter()
    # telemetry floor 0.8 x parent for positive scores; for negative (ALE minimisation, signed) 1.25 x
    floor = parent_fitness - 0.2 * abs(parent_fitness)  # = 0.8 x parent for positive scores, signed for ALE
    if not math.isfinite(floor):  # failed-evaluation parent (NaN / inf): no floor
        floor = -1e300
    params = {"task": task, "q": "[" + ",".join(f"{x:.6g}" for x in q) + "]", "floor": floor, "m": m}
    pools: dict[str, list[PoolRow]] = {}
    union: dict[int, PoolRow] = {}
    for g in configs:
        rows = []
        sql = pool_sql(g)
        if require_changes:  # summaries are what gets injected: a row without one carries nothing
            sql = sql.replace("AND d.code IS NOT NULL", "AND d.code IS NOT NULL AND d.changes IS NOT NULL", 1)
        for r in conn.execute(sql, params).fetchall():
            row = union.setdefault(r[0], PoolRow(*r))
            rows.append(row)
        pools[g.name] = rows
    ret_ms = (time.perf_counter() - t0) * 1000
    tok0, t1 = _sem_tokens(conn, prompt_version), time.perf_counter()
    if union:
        ids = list(union)
        scores = dict(conn.execute(SCORE_SQL, {"ids": ids}).fetchall())
        passed = {r[0] for r in conn.execute(FILTER_SQL, {"ids": ids}).fetchall()}
        for i, r in union.items():
            r.score = scores.get(i)
            r.verdict = i in passed
    sem_ms = (time.perf_counter() - t1) * 1000
    sem_tokens = _sem_tokens(conn, prompt_version) - tok0
    cands = [Candidate("none", 0, [])]
    for g in configs:
        for fam in FAMILIES:
            ordered = _order(pools[g.name], fam)
            for k in KS:
                cands.append(Candidate(fam, k, ordered[:k], graph=g))
    return PoolsResult(pools, cands, ret_ms, sem_tokens, sem_ms, len(union))


def featurize_v2(c: Candidate, *, parent_fitness: float, best_fitness: float, iteration: int, horizon: int,
                 parent_tokens: int, pool_n: int, union_n: int, scale: float = 1.0) -> dict:
    """`scale` divides every score-valued feature: |seed score| for ALE, whose problems differ by up to
    four orders of magnitude (and some are negative); 1 for math (scores already in [0, ~1])."""
    f = featurize(c, parent_fitness=parent_fitness, iteration=iteration, horizon=horizon,
                  parent_tokens=parent_tokens)
    if scale != 1.0:
        for key in ("parent_fitness", "gain_vs_parent_max", "gain_vs_parent_mean", "own_gain_max",
                    "own_gain_mean", "gain_std"):
            f[key] = f[key] / scale
    g = c.graph or GraphConfig()
    f.update({
        "best_gap": (best_fitness - parent_fitness) / scale,
        "edge_similar": float(c.family != "none" and g.edge == "similar"),
        "edge_mixed": float(c.family != "none" and g.edge == "mixed"),
        "dir_up": float(c.family != "none" and g.dir == "up"),
        "hops2": float(c.family != "none" and g.hops == 2),
        "log_fanout": math.log(g.fanout) if c.family != "none" else 0.0,
        "pool_n": float(pool_n),
        # share of this iteration's semantic-pass tokens attributable to the candidate's pool
        "sem_share": (pool_n / union_n) if union_n and c.family in ("hard", "rank") else 0.0,
    })
    return f


def vector_v2(f: dict) -> np.ndarray:
    d = {"log_fanout": math.log(8)} | f if f.get("fam_none", 0.0) == 0.0 else f
    return np.array([float(d.get(k, 0.0)) for k in FEATURE_ORDER_V2], dtype=float)


def vector_none(f: dict) -> np.ndarray:
    return np.array([float(f.get(k, 0.0)) for k in NONE_FEATURES], dtype=float)


FEATURE_ORDER_V3 = FEATURE_ORDER_V2 + ("edge_mixed",)


def vector_v3(f: dict) -> np.ndarray:
    d = {"log_fanout": math.log(8)} | f if f.get("fam_none", 0.0) == 0.0 else f
    return np.array([float(d.get(k, 0.0)) for k in FEATURE_ORDER_V3], dtype=float)
