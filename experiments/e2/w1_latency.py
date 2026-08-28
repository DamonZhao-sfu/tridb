"""Per-leg latency: what each modality costs, and what fusing them saves.

    python3 -m experiments.e2.w1_latency --dsn ... --scope ... --limit 200

WHY THIS IS A SEPARATE HARNESS
------------------------------
`tjs_open` is one C call that interleaves the bounded traversal, the per-candidate
relational probe and the distance computation. It exposes no internal timers, so the
graph / relational / vector split cannot be read out of a fused run. Adding timers
inside `tjs_pg.c` would be the most direct route and is a real option, but it changes
the thing being measured and needs a rebuild.

Instead this measures the same query **built up one leg at a time**, each leg executed
separately and timed:

    ann      seedless tjs_open over the entry kind        -- vector leg (entry)
    graph    gph_traverse_bounded, drained                -- graph leg
    filter   the predicate over the MATERIALISED reach set -- relational leg
    vector   ORDER BY embedding <=> q LIMIT k over survivors -- vector leg (ranking)
    fused    the real tjs_open call                       -- all three, interleaved

The point is not to reconstruct tjs_open's internals. It is that
``ann + graph + filter + vector`` is exactly what an *unfused* pipeline has to pay:
each stage runs to completion and materialises its whole intermediate result before
the next one starts. So

    sum(legs) - fused   =   what interleaving saves

and `rows_materialised` at each boundary is the intermediate-result blow-up that a
composed system carries and a fused operator never creates.

HONESTY NOTES
-------------
* The legs are separate executions, so they do not sum to the fused time by
  construction, and they are an UPPER bound on it. That gap is the result, not an error.
* The relational leg uses `id = ANY(array)` over the materialised reach set, which is
  what a composed pipeline would do. It is deliberately NOT an attempt to replicate the
  operator's per-candidate SPI probe.
* Run this with nothing else touching the database. Numbers taken while another sweep
  is running measure contention.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.groundtruth import GroundTruth
from bench.agent_memory.gem_eg.query import RELATION_EDGE_TYPE, Knobs, W1Engine
from bench.agent_memory.gem_eg.store import DEFAULT_DSN, EgStore
from experiments.e2.w1_runner import SHAPES, build_spec, sample

DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
DEFAULT_OUT = Path("results/e2/w1/latency")

LEGS = ("ann", "graph", "filter", "vector")


@dataclass
class LegSample:
    dp_id: str
    query_id: str
    ann_ms: float = 0.0
    graph_ms: float = 0.0
    filter_ms: float = 0.0
    vector_ms: float = 0.0
    fused_ms: float = 0.0
    fused_first_row_ms: float | None = None
    rows_after_graph: int = 0
    rows_after_filter: int = 0
    rows_returned: int = 0
    graph_examined: int = 0
    candidates_examined: int = 0

    #: The fused call's own post-ANN portion, for the comparison that means something.
    fused_expand_ms: float = 0.0

    @property
    def unfused_ms(self) -> float:
        return self.ann_ms + self.graph_ms + self.filter_ms + self.vector_ms

    @property
    def unfused_expand_ms(self) -> float:
        """Everything AFTER the ANN entry, run as separate stages.

        The ANN is paid identically by both sides — the fused call performs it too — so
        including it buries the effect being measured under a large shared constant.
        This is the part fusion actually changes: traverse, filter, rank.
        """
        return self.graph_ms + self.filter_ms + self.vector_ms

    @property
    def expand_saving_ms(self) -> float:
        return self.unfused_expand_ms - self.fused_expand_ms

    @property
    def fusion_saving_ms(self) -> float:
        return self.unfused_ms - self.fused_ms


@dataclass
class Bucket:
    samples: list[LegSample] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}

        def stat(values: list[float]) -> dict[str, float]:
            ordered = sorted(values)
            return {
                "p50": round(statistics.median(ordered), 3),
                "p95": round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 3),
                "mean": round(statistics.fmean(ordered), 3),
            }

        legs = {leg: stat([getattr(s, f"{leg}_ms") for s in self.samples]) for leg in LEGS}
        fused = stat([s.fused_ms for s in self.samples])
        unfused = stat([s.unfused_ms for s in self.samples])
        saving = stat([s.fusion_saving_ms for s in self.samples])
        total_leg_mean = sum(legs[leg]["mean"] for leg in LEGS) or 1.0
        return {
            "n": len(self.samples),
            "legs_ms": legs,
            # Share of the UNFUSED pipeline's cost each modality accounts for. Not a
            # share of the fused time — the fused operator never runs the legs apart.
            "leg_share_of_unfused": {
                leg: round(legs[leg]["mean"] / total_leg_mean, 4) for leg in LEGS
            },
            "unfused_ms": unfused,
            "fused_ms": fused,
            "fusion_saving_ms": saving,
            "fusion_speedup": round(unfused["mean"] / fused["mean"], 3) if fused["mean"] else None,
            # The comparison that isolates the effect: both sides pay the same ANN, so
            # only the post-entry portion can differ.
            "unfused_expand_ms": stat([s.unfused_expand_ms for s in self.samples]),
            "fused_expand_ms": stat([s.fused_expand_ms for s in self.samples]),
            "expand_saving_ms": stat([s.expand_saving_ms for s in self.samples]),
            "expand_speedup": (
                round(
                    statistics.fmean([s.unfused_expand_ms for s in self.samples])
                    / statistics.fmean([s.fused_expand_ms for s in self.samples]),
                    3,
                )
                if statistics.fmean([s.fused_expand_ms for s in self.samples])
                else None
            ),
            # Share of TOTAL fused latency spent in the ANN entry. When this is ~1.0 the
            # workload is ANN-bound and no traversal optimisation can move the number.
            "ann_share_of_fused": round(
                statistics.fmean([s.ann_ms for s in self.samples])
                / max(1e-9, statistics.fmean([s.fused_ms for s in self.samples])),
                4,
            ),
            "fused_first_row_ms": stat(
                [s.fused_first_row_ms for s in self.samples if s.fused_first_row_ms is not None]
            )
            if any(s.fused_first_row_ms is not None for s in self.samples)
            else None,
            "rows_materialised": {
                "after_graph_p50": round(
                    statistics.median([s.rows_after_graph for s in self.samples]), 1
                ),
                "after_filter_p50": round(
                    statistics.median([s.rows_after_filter for s in self.samples]), 1
                ),
                "returned_p50": round(
                    statistics.median([s.rows_returned for s in self.samples]), 1
                ),
            },
            "work": {
                "graph_examined_p50": round(
                    statistics.median([s.graph_examined for s in self.samples]), 1
                ),
                "candidates_examined_p50": round(
                    statistics.median([s.candidates_examined for s in self.samples]), 1
                ),
            },
        }


class LegTimer:
    def __init__(self, engine: W1Engine, knobs: Knobs) -> None:
        self.engine = engine
        self.conn = engine.conn
        self.knobs = knobs

    def measure(self, spec: Any, dp_id: str) -> LegSample | None:
        query_vec = self.engine._vectors.get(spec.seed_uid)
        if query_vec is None:
            return None
        out = LegSample(dp_id=dp_id, query_id=spec.query_id)

        with self.engine.settings(self.knobs):
            # -- vector leg (entry) --------------------------------------
            t = time.perf_counter()
            entry_vids = self.engine.stage1_entries(spec, self.knobs, query_vec)
            out.ann_ms = (time.perf_counter() - t) * 1000.0
            if not entry_vids:
                return None

            # -- graph leg -----------------------------------------------
            edge_type = self.engine.store.edge_type_id(RELATION_EDGE_TYPE[spec.relation])
            reached: list[int] = []
            t = time.perf_counter()
            for entry_vid in entry_vids:
                rows = self.conn.execute(
                    "SELECT graph_store.gph_traverse_bounded(%s, %s, %s, %s)",
                    (entry_vid, spec.hops, edge_type, self.knobs.graph_work_budget),
                ).fetchall()
                reached.extend(int(r[0]) for r in rows if r[0] is not None)
            out.graph_ms = (time.perf_counter() - t) * 1000.0
            out.rows_after_graph = len(reached)

            survivors: list[int] = []
            if reached:
                # -- relational leg --------------------------------------
                t = time.perf_counter()
                rows = self.conn.execute(
                    "SELECT id FROM gem_eg_vertex WHERE id = ANY(%s) AND ("
                    + spec.predicate.to_sql()
                    + ")",
                    (reached,),
                ).fetchall()
                out.filter_ms = (time.perf_counter() - t) * 1000.0
                survivors = [int(r[0]) for r in rows]
            out.rows_after_filter = len(survivors)

            if survivors:
                # -- vector leg (ranking) --------------------------------
                t = time.perf_counter()
                rows = self.conn.execute(
                    "SELECT id FROM gem_eg_vertex"
                    " WHERE id = ANY(%s) AND embedding IS NOT NULL"
                    " ORDER BY embedding <=> %s::vector, uid LIMIT %s",
                    (survivors, query_vec, spec.k),
                ).fetchall()
                out.vector_ms = (time.perf_counter() - t) * 1000.0

        fused = self.engine.run(spec, self.knobs)
        out.fused_ms = fused.total_ms
        out.fused_expand_ms = fused.stage2_ms + fused.merge_ms
        out.fused_first_row_ms = fused.first_row_ms
        out.rows_returned = len(fused.ids)
        out.graph_examined = fused.graph_examined
        out.candidates_examined = fused.candidates_examined
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=200, help="decision points per query")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m-seeds", type=int, default=4)
    args = parser.parse_args(argv)

    corpus = Corpus.load(args.normalized, vectors=args.normalized / "vectors.npz")
    gt = GroundTruth(corpus)
    store = EgStore.connect(args.dsn)
    store.bootstrap_edge_types()
    engine = W1Engine(store, args.scope)
    engine.load_identity()
    timer = LegTimer(engine, Knobs())

    buckets: dict[str, Bucket] = {q: Bucket() for q in SHAPES}
    try:
        for query_id in SHAPES:
            points = sample(
                [dp for dp in gt.all_points() if dp.query_id == query_id],
                args.limit,
                args.seed,
            )
            specs = [build_spec(dp, "fused", k=args.k, m_seeds=args.m_seeds) for dp in points]
            # Warm the cache first: a cold first query would land entirely in the leg
            # that happened to run first and skew its share.
            for spec, dp in list(zip(specs, points))[: args.warmup]:
                timer.measure(spec, dp.dp_id)
            for _ in range(args.repeats):
                for spec, dp in zip(specs, points):
                    got = timer.measure(spec, dp.dp_id)
                    if got is not None:
                        buckets[query_id].samples.append(got)
            print(f"  {query_id}: {len(buckets[query_id].samples):,} samples", flush=True)
    finally:
        store.close()

    report = {
        "scope": args.scope,
        "k": args.k,
        "m_seeds": args.m_seeds,
        "repeats": args.repeats,
        "per_query": {q: b.summary() for q, b in buckets.items() if b.samples},
        "method": (
            "Legs are executed SEPARATELY and timed; they are an upper bound on the "
            "fused time, and the gap is what interleaving saves. The relational leg "
            "filters a materialised reach set (id = ANY), which is what a composed "
            "pipeline does — not a replica of tjs_open's per-candidate probe."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "w1_latency.json"
    path.write_text(json.dumps(report, indent=2))

    head = (
        f"{'query':10} {'n':>6} {'ann':>8} {'graph':>7} {'filt':>7} {'vec':>7}"
        f" {'ANN%':>6} | {'unfus_exp':>9} {'fus_exp':>8} {'saved':>7} {'x':>6}"
    )
    print("\n" + head)
    print("-" * len(head))
    for query_id, summary in report["per_query"].items():
        legs = summary["legs_ms"]
        print(
            f"{query_id:10} {summary['n']:>6,}"
            f" {legs['ann']['p50']:>8.2f} {legs['graph']['p50']:>7.2f}"
            f" {legs['filter']['p50']:>7.2f} {legs['vector']['p50']:>7.2f}"
            f" {summary['ann_share_of_fused']:>6.1%} |"
            f" {summary['unfused_expand_ms']['p50']:>9.2f}"
            f" {summary['fused_expand_ms']['p50']:>8.2f}"
            f" {summary['expand_saving_ms']['p50']:>7.2f}"
            f" {summary['expand_speedup']:>6.2f}"
        )
    print(f"\nreport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
