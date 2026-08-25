"""W1 harness: parity gate, then quality, then latency — in that order.

    python3 -m experiments.e2.w1_runner --dsn ... --scope ... --limit 500
    python3 -m experiments.e2.w1_runner --dsn ... --scope ... --full

ORDER IS THE METHODOLOGY
------------------------
1. **Parity** against the exact oracle. A cell whose parity fails prints no latency.
   A fast wrong answer is not a result.
2. **Quality** — nDCG and harmful@k against reward-graded relevance, plus recall
   against the oracle's own top-k.
3. **Latency** — first-row and total, reported only for cells that passed (1).

Without a baseline system in play, `oracle_recall` measures the OPERATOR's own
approximation loss (HNSW `ef_search`, `tjs.graph_work_budget` censoring, `term_cond`
early termination) and never a cross-system comparison. The ablation arms are what
carry the quality claim: `fused` against `vector_only`, `graph_only` and `filter_only`,
all inside the same engine.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.groundtruth import (
    QUERY_PATH,
    QUERY_REPAIR,
    QUERY_REUSE,
    QUERY_STUCK,
    DecisionPoint,
    GroundTruth,
)
from bench.agent_memory.gem_eg.metrics import Accumulator, harmful_at_k, ndcg_at_k, parity
from bench.agent_memory.gem_eg.oracle import Oracle, Predicate, QuerySpec
from bench.agent_memory.gem_eg.query import Knobs, W1Engine
from bench.agent_memory.gem_eg.store import DEFAULT_DSN, EgStore

DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
DEFAULT_OUT = Path("results/e2/w1")

#: Each query's traversal shape. The entry kind differs because W1.a is seeded by a
#: Task description (the paper's own formulation) while the rest are seeded by code.
SHAPES: dict[str, dict[str, Any]] = {
    QUERY_REUSE: {"relation": "hier", "hops": 2, "entry": "task"},
    QUERY_STUCK: {"relation": "lineage", "hops": 3, "entry": "node"},
    QUERY_REPAIR: {"relation": "lineage", "hops": 2, "entry": "node"},
    QUERY_PATH: {"relation": "child_of", "hops": 4, "entry": "node"},
}

#: Ablation arms, all within TriDB. Each name states which leg is REMOVED, and each
#: one actually removes it — an earlier `graph_only` still selected entries by ANN, so
#: it measured a ranking change rather than the absence of the vector leg.
#:
#:   fused        vector + graph + relational, ranked by similarity
#:   no_graph     vector + relational              (ANN over the whole table)
#:   no_vector    graph + relational               (entry by task, ranked by reward)
#:   filter_only  relational                       (predicate, ranked by reward)
#:   reward_rank  fused shape, ranked by REWARD instead of similarity. Not an ablation:
#:                the separate G4 axis, isolating what tjs_open's vector-only ranking
#:                costs against the ordering the paper's reuse query actually wants.
ARMS = ("fused", "no_graph", "no_vector", "filter_only", "reward_rank")


@dataclass
class Cell:
    query_id: str
    arm: str
    knobs: Knobs
    acc: Accumulator


def build_spec(dp: DecisionPoint, arm: str, *, k: int, m_seeds: int) -> QuerySpec:
    """The query definition. Nothing here comes from Knobs — see Knobs' docstring."""
    shape = SHAPES[dp.query_id]
    # The RESULT is always something worth using — for the repair query too. The seed
    # is a failure; what we want back is the fix, not more failures. An earlier version
    # inverted this and asked for invalid nodes, which scored nDCG 0.0 by construction.
    predicate = Predicate(
        kind="node",
        require_valid=True,
        require_fitness=True,
        exclude_sessions=frozenset({dp.session_uid}),
    )
    mode = {
        "fused": "ann_then_traverse",
        "reward_rank": "ann_then_traverse",
        "no_graph": "ann_only",
        "no_vector": "task_scoped",
        "filter_only": "filter_only",
    }[arm]
    hops = shape["hops"]
    # `no_graph` searches the whole node table, so its ANN entry kind is always `node`
    # even for W1.a, whose fused form enters through a Task.
    entry = "node" if arm == "no_graph" else shape["entry"]
    return QuerySpec(
        query_id=dp.query_id,
        decision_point=dp.dp_id,
        seed_uid=dp.seed_uid,
        relation=shape["relation"],
        hops=hops,
        k=k,
        ann_entry_kind=entry,
        ann_m_seeds=m_seeds,
        predicate=predicate,
        mode=mode,
        rank_by="reward" if arm == "reward_rank" else "similarity",
        meta={"task_uid": dp.task_uid},
    )


def sample(points: list[DecisionPoint], limit: int | None, seed: int) -> list[DecisionPoint]:
    """Stratified by (query, backend, domain) so no stratum is silently dropped.

    A uniform sample would under-represent shinkaevolve (1,872 nodes) against
    openevolve_native (4,267) and make a per-backend breakdown unreadable.
    """
    if limit is None or limit >= len(points):
        return points
    buckets: dict[tuple[str, str, str], list[DecisionPoint]] = {}
    for dp in points:
        buckets.setdefault((dp.query_id, dp.backend, dp.domain), []).append(dp)
    rng = random.Random(seed)
    out: list[DecisionPoint] = []
    per_bucket = max(1, limit // max(1, len(buckets)))
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda d: d.dp_id)
        rng.shuffle(bucket)
        out.extend(bucket[:per_bucket])
    out.sort(key=lambda d: d.dp_id)
    return out[:limit]


def run(
    *,
    corpus: Corpus,
    gt: GroundTruth,
    oracle: Oracle,
    engine: W1Engine,
    points: list[DecisionPoint],
    knobs: Knobs,
    k: int,
    m_seeds: int,
    arms: tuple[str, ...],
    raw: Any = None,
) -> dict[str, Cell]:
    cells: dict[str, Cell] = {}
    order = list(points)
    started = time.perf_counter()
    for index, dp in enumerate(order, 1):
        if index % 200 == 0:
            rate = index / max(1e-9, time.perf_counter() - started)
            eta = (len(order) - index) / max(1e-9, rate)
            print(
                f"  {index:,}/{len(order):,} points  {rate:.1f}/s  eta {eta / 60:.0f}m",
                flush=True,
            )
        for arm in arms:
            key = f"{dp.query_id}|{arm}"
            cell = cells.get(key)
            if cell is None:
                cell = cells[key] = Cell(dp.query_id, arm, knobs, Accumulator())
            spec = build_spec(dp, arm, k=k, m_seeds=m_seeds)

            expected = oracle.run(spec)
            observed = engine.run(spec, knobs)

            acc = cell.acc
            acc.n += 1
            # Distances are recomputed here, OUTSIDE the timed region, because
            # tjs_open returns bare ids. They are only used for the tie-tolerant gate.
            seed_vec = corpus.vector(dp.seed_uid)
            obs_d = (
                [float(x) for x in corpus.cosine_distance(seed_vec, observed.ids)]
                if seed_vec is not None
                else None
            )
            exp_d = list(expected.distances) if expected.distances else None
            exact, recall, tie_ok = parity(
                observed.ids,
                expected.topk,
                observed_distances=obs_d,
                expected_distances=exp_d,
            )
            acc.parity_exact += int(exact)
            acc.parity_tie_equivalent += int(tie_ok)
            acc.recall_sum += recall
            if observed.first_row_ms is not None:
                acc.first_row_ms.append(observed.first_row_ms)
            acc.total_ms.append(observed.total_ms)
            acc.examined.append(observed.candidates_examined)
            acc.graph_examined.append(observed.graph_examined)
            acc.stage1_ms.append(observed.stage1_ms)
            acc.stage2_ms.append(observed.stage2_ms)
            acc.merge_ms.append(observed.merge_ms)
            if observed.first_candidate_ms is not None:
                acc.first_candidate_ms.append(observed.first_candidate_ms)
            acc.censored += int(observed.graph_censored)

            # Quality is scored against the ENGINE's answer, not the oracle's: the
            # question is what a consumer would actually receive.
            grades = gt.grades(dp, observed.ids)
            # Arm-INDEPENDENT: the best any retriever could have done here. See
            # GroundTruth.ideal_grades for why deriving it per arm was wrong.
            ideal = gt.ideal_grades(dp, k)
            if not ideal or max(ideal, default=0) <= 0:
                acc.n_empty_ideal += 1
            else:
                acc.n_scored += 1
                ordered = [grades[u] for u in observed.ids]
                acc.ndcg_sum += ndcg_at_k(ordered, ideal, k)
                acc.harmful_sum += harmful_at_k(ordered, k)
                # The ceiling: this formulation's own reachable pool, ranked perfectly,
                # scored against the same arm-independent ideal. A low nDCG with a low
                # ceiling means the query cannot SEE the good nodes; a low nDCG with a
                # high ceiling means it sees them and orders them badly. Those are
                # different findings and must not collapse into one number.
                pool_grades = sorted(
                    gt.grades(dp, list(expected.eligible)).values(), reverse=True
                )
                acc.ceiling_sum += ndcg_at_k(pool_grades[:k], ideal, k)
            acc.trivial_hits += sum(
                1 for u in observed.ids if gt.is_trivial_hit(u, dp.session_uid)
            )
            if raw is not None:
                # Every returned id, so any re-scoring later is offline and instant
                # rather than another three-hour run.
                raw.write(
                    json.dumps(
                        {
                            "dp_id": dp.dp_id,
                            "arm": arm,
                            "ids": observed.ids,
                            "oracle_topk": list(expected.topk),
                            "grades": [grades[u] for u in observed.ids],
                            "ideal": ideal,
                            "first_row_ms": observed.first_row_ms,
                            "total_ms": observed.total_ms,
                            "examined": observed.candidates_examined,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
    return cells


def _fmt(value: float | int | None) -> str:
    """`None` means the arm produced no rows at all — a fact, not a missing number."""
    return "n/a" if value is None else f"{value:.3f}"


def _assert_vectors_agree(engine: W1Engine, corpus: Corpus, npz: Path) -> None:
    """The engine ranks with the DB's vectors; the oracle ranks with the npz's.

    Two sources of truth for the same numbers, and nothing ties them together: a
    re-embed that updates the database without re-running `export_vectors.py` leaves
    the oracle scoring against the PREVIOUS embedding. That failure is silent and
    looks exactly like a defect in the operator -- measured once, it drove W1.a's
    parity from 0.963 to 0.000 and its oracle_recall from 0.996 to 0.094, with no
    error anywhere. A parity gate cannot catch it, because parity is the thing it
    breaks.

    So compare them before measuring anything, and fail closed.
    """
    mismatched: list[str] = []
    missing: list[str] = []
    for uid, row in corpus.vector_row.items():
        literal = engine._vectors.get(uid)
        if literal is None:
            missing.append(uid)
            continue
        db = np.asarray(json.loads(literal), dtype=np.float32)
        db /= np.linalg.norm(db) or 1.0
        if float(np.abs(db - corpus.vectors[row]).max()) > 1e-5:
            mismatched.append(uid)
    if mismatched or missing:
        raise SystemExit(
            f"vectors.npz disagrees with {engine.store.__class__.__name__}: "
            f"{len(mismatched)} differ, {len(missing)} absent from the database "
            f"(e.g. {(mismatched or missing)[:3]}). "
            f"Re-run tools/evotrace/export_vectors.py after any re-embed."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--queries", default=",".join(SHAPES))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--full", action="store_true", help="every decision point")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m-seeds", type=int, default=4)
    parser.add_argument("--term-cond", type=int, default=0)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--graph-budget", type=int, default=65536)
    args = parser.parse_args(argv)

    wanted = set(args.queries.split(","))
    arms = tuple(a for a in args.arms.split(",") if a in ARMS)
    knobs = Knobs(
        term_cond=args.term_cond,
        ef_search=args.ef_search,
        graph_work_budget=args.graph_budget,
    )

    corpus = Corpus.load(args.normalized, vectors=args.normalized / "vectors.npz")
    gt = GroundTruth(corpus)
    oracle = Oracle(corpus)
    points = [dp for dp in gt.all_points() if dp.query_id in wanted]
    chosen = points if args.full else sample(points, args.limit, args.seed)

    store = EgStore.connect(args.dsn)
    store.bootstrap_edge_types()
    engine = W1Engine(store, args.scope)
    engine.load_identity()
    _assert_vectors_agree(engine, corpus, args.normalized / "vectors.npz")

    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "w1_raw.jsonl"
    started = time.perf_counter()
    raw = raw_path.open("w", encoding="utf-8")
    try:
        cells = run(
            corpus=corpus,
            gt=gt,
            oracle=oracle,
            engine=engine,
            points=chosen,
            knobs=knobs,
            k=args.k,
            m_seeds=args.m_seeds,
            arms=arms,
            raw=raw,
        )
    finally:
        raw.close()
        store.close()
    seconds = time.perf_counter() - started

    report = {
        "scope": args.scope,
        "corpus": corpus.summary(),
        "query_definition": {"k": args.k, "m_seeds": args.m_seeds, "shapes": SHAPES},
        "knobs": knobs.__dict__,
        "decision_points": {
            "available": len(points),
            "measured": len(chosen),
            "sampling": "full" if args.full else f"stratified(limit={args.limit},seed={args.seed})",
        },
        "seconds": round(seconds, 1),
        "cells": {key: cell.acc.summary() for key, cell in sorted(cells.items())},
        "claim_boundary": (
            "Single system. `oracle_recall` is this operator's own approximation loss "
            "(HNSW + graph budget + early termination), not a cross-system comparison; "
            "latency has no baseline to be compared against and is characterisation only."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "w1_report.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"decision points : {len(chosen):,} of {len(points):,}   ({seconds:.1f}s)")
    header = (
        f"{'query|arm':26} {'n':>6} {'parity':>7} {'recall':>7} {'nDCG':>7}"
        f" {'ceil':>7} {'rank%':>6} {'harm':>6} {'p50ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for key, cell in sorted(cells.items()):
        s = cell.acc.summary()
        gate = (
            "PASS"
            if s["parity_tie_equivalent_rate"] == 1.0
            else f"{s['parity_tie_equivalent_rate']:.3f}"
        )
        # A failed parity gate prints no latency: a fast wrong answer is not a result.
        if s["parity_tie_equivalent_rate"] == 1.0:
            lat = f"{_fmt(s['first_row_ms_p50']):>8} {_fmt(s['first_row_ms_p95']):>8}"
        else:
            lat = f"{'--':>8} {'--':>8}"
        eff = s["rank_efficiency"]
        print(
            f"{key:26} {s['n']:>6,} {gate:>7} {s['oracle_recall']:>7.4f}"
            f" {s['ndcg']:>7.4f} {s['ndcg_ceiling']:>7.4f}"
            f" {(f'{eff:.2f}' if eff is not None else 'n/a'):>6}"
            f" {s['harmful']:>6.3f} {lat.split()[0]:>8}"
        )
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
