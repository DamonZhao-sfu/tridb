"""Three-way traversal check: oracle vs GEM vs polyglot, on the experiment's own shape.

The existing gate (`gate_polyglot_parity.py`) compares the two SYSTEMS to each other,
which cannot say which one is wrong when they disagree. This adds the oracle -- the
exhaustive definition both are supposed to compute -- as the third leg, so a
disagreement is attributable.

Motivating measurement: on `math:heilbronn_triangle` at p=1.0 the two arms returned
overlapping-but-different programs (Jaccard 0.68) while reporting very different
traversal work -- GEM `graph_examined=372`, polyglot `candidates_examined=2751`,
against a true 2-hop reach of 1,667 nodes. Either GEM is terminating early and
missing eligible nodes, or the counters mean different things. This script decides.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.groundtruth import GroundTruth
from bench.agent_memory.gem_eg.oracle import Oracle
from bench.agent_memory.gem_eg.query import Knobs, W1Engine
from bench.agent_memory.gem_eg.store import EgStore

DEFAULT_DSN = "postgresql://127.0.0.1:55432/evotrace_eg"
DEFAULT_SCOPE = "evotrace:349117b0"
DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
#: The four tasks the p-sweep actually ran.
SWEEP_TASKS = (
    "math:heilbronn_triangle",
    "math:heilbronn_convex_13",
    "math:circle_packing",
    "math:third_autocorr_ineq",
)


class _StubParent:
    """Only `metrics['combined_score']` is read when building the predicate."""

    def __init__(self, fitness: float | None) -> None:
        self.metrics = {"combined_score": fitness} if fitness is not None else {}
        self.id = "stub"


def recall(observed: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    return len(set(observed) & set(expected)) / len(expected)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--scope", default=DEFAULT_SCOPE)
    ap.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260826)
    # k=10 is what the agent experiment injects, NOT the gate's default of 5. A
    # bound that only bites at larger k would be invisible at 5.
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-polyglot", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from bench.agent_memory.gem_oe.retrievers import GemRetriever

    corpus = Corpus.load(args.normalized, vectors=args.normalized / "vectors.npz")
    oracle = Oracle(corpus)
    points = [
        dp
        for dp in GroundTruth(corpus).all_points()
        if dp.query_id == "W1.a" and dp.task_uid in SWEEP_TASKS
    ]
    if not points:
        raise SystemExit("no W1.a decision points on the sweep's tasks")
    chosen = random.Random(args.seed).sample(points, min(args.limit, len(points)))

    store = EgStore.connect(args.dsn)
    store.bootstrap_edge_types()
    gem = GemRetriever(store=store, scope_id=args.scope, knobs=Knobs())
    engine = W1Engine(store, args.scope)
    engine.load_identity()

    backend = None
    if not args.no_polyglot:
        import numpy as np

        from bench.agent_memory.gem_oe.polyglot_backend import PolyglotBackend

        backend = PolyglotBackend()

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for dp in chosen:
        spec = gem._spec(
            dp.task_uid, _StubParent(dp.parent_fitness), args.k, dp.iteration
        )
        exact = oracle.run(spec)
        expected = list(exact.topk)

        result = engine.run(spec, gem.knobs)
        gem_ids = list(result.ids)

        poly_ids: list[str] = []
        poly_tel: dict[str, Any] = {}
        if backend is not None:
            literal = engine._vectors.get(spec.seed_uid)
            qv = np.asarray(json.loads(literal), dtype=np.float32)
            qv /= np.linalg.norm(qv) or 1.0
            ids, poly_tel = backend.reuse_query(spec, qv)
            poly_ids = list(ids)

        rows.append(
            {
                "task_uid": dp.task_uid,
                "iteration": dp.iteration,
                "parent_fitness": dp.parent_fitness,
                "eligible": len(exact.eligible_set),
                "oracle_traversed": exact.traversed,
                "oracle_n": len(expected),
                "gem_recall": recall(gem_ids, expected),
                "poly_recall": recall(poly_ids, expected) if backend else None,
                "gem_graph_examined": result.graph_examined,
                "gem_termination": result.termination_reason,
                "poly_candidates": poly_tel.get("candidates_examined"),
                "jaccard_gem_poly": (
                    len(set(gem_ids) & set(poly_ids))
                    / len(set(gem_ids) | set(poly_ids))
                    if backend and (gem_ids or poly_ids)
                    else None
                ),
            }
        )

    def avg(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else None

    print(
        f"{len(rows)} decision points, k={args.k}, {time.perf_counter() - started:.1f}s\n"
    )
    print(
        f"{'task':<26}{'n':>4}{'合格池':>8}{'GEM召回':>9}{'poly召回':>10}"
        f"{'GEM走过':>9}{'poly走过':>10}{'两者交集':>9}"
    )
    for task in sorted({r["task_uid"] for r in rows}):
        sub = [r for r in rows if r["task_uid"] == task]

        def m(key: str) -> float:
            vals = [r[key] for r in sub if r[key] is not None]
            return statistics.mean(vals) if vals else float("nan")

        print(
            f"{task.split(':')[-1]:<26}{len(sub):>4}{m('eligible'):>8.0f}"
            f"{m('gem_recall'):>9.3f}{m('poly_recall'):>10.3f}"
            f"{m('gem_graph_examined'):>9.0f}{m('poly_candidates'):>10.0f}"
            f"{m('jaccard_gem_poly'):>9.2f}"
        )

    print(
        f"\n总体  GEM 召回 {avg('gem_recall'):.3f}   "
        f"polyglot 召回 {avg('poly_recall') if avg('poly_recall') is not None else float('nan'):.3f}"
    )
    print("召回 = 与 oracle（穷举 2 跳 + 谓词 + 距离排序 top-k）的重合比例。")
    print("低于 1.000 的一方就是偏离共同定义的一方。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
