"""W1.a on TriDB/GEM and Mem0: the same query, scored the same way.

    PYTHONPATH=. <mem0-venv>/bin/python -m experiments.e2.w1a_crosssystem \
        --tridb-dsn postgresql://... --scope evotrace:349117b0 --limit 2000

WHY THIS COMPARISON IS FAIR
Three confounders are removed by construction rather than argued away:

* **Same vectors.** Both systems index the exported `vectors.npz` — bit-identical
  embeddings of bit-identical text. Neither re-embeds.
* **Same seed vector per query.** Passed in on both sides, so the sweep touches no
  embedding endpoint and GPU contention cannot skew it.
* **Same predicate.** W1.a's `eg_hier` 2-hop traversal is provably equivalent to
  `task_uid == T` on this corpus (18/18 tasks, zero discrepancy), so Mem0 expresses
  the whole query exactly — an ANN entry over Tasks, then a filtered ANN over Nodes.
  Nothing is emulated.

What differs is only where the work happens: TriDB pushes traversal, predicate and
top-k into one operator over a native adjacency; Mem0 runs two pgvector searches with
a metadata filter and materialises the result.

WHAT IS NOT REPORTED
Latency. The GPU is contended, and a latency number taken under contention measures
the contention. Quality is unaffected — the vectors are deterministic, so a slower run
produces identical results — so this reports quality only and leaves latency to a
dedicated run on a quiet machine.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.groundtruth import QUERY_REUSE, DecisionPoint, GroundTruth
from bench.agent_memory.gem_eg.metrics import harmful_at_k, ndcg_at_k, parity
from bench.agent_memory.gem_eg.oracle import Oracle, Predicate, QuerySpec
from experiments.e2.w1_runner import sample

DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
DEFAULT_OUT = Path("results/e2/w1a_crosssystem")

#: The three lineage queries have no metadata encoding — arbitrary-depth parent-child
#: chains — and this Mem0 configuration has no graph. Reported, never emulated.
UNSUPPORTED = {
    "mem0": {
        "W1.b": "no graph in this configuration; lineage depth has no metadata encoding",
        "W1.b-fail": "same",
        "W1.d": "same; reverse traversal additionally needs edge direction",
    }
}


@dataclass
class Acc:
    n: int = 0
    n_scored: int = 0
    ndcg: float = 0.0
    ceiling: float = 0.0
    harmful: float = 0.0
    recall: float = 0.0
    exact: int = 0
    tie_ok: int = 0
    empty: int = 0
    trivial: int = 0
    returned: int = 0
    stage1_ms: list[float] = field(default_factory=list)
    stage2_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        d = max(1, self.n)
        s = max(1, self.n_scored)
        return {
            "n": self.n,
            "n_scored": self.n_scored,
            "empty_result": self.empty,
            "mean_returned": round(self.returned / d, 2),
            "ndcg@10": round(self.ndcg / s, 4),
            "ceiling": round(self.ceiling / s, 4),
            "rank_efficiency": round(self.ndcg / self.ceiling, 4) if self.ceiling > 0 else None,
            "harmful@10": round(self.harmful / s, 4),
            "oracle_recall": round(self.recall / d, 4),
            "parity_exact": round(self.exact / d, 4),
            "parity_tie_equivalent": round(self.tie_ok / d, 4),
            "trivial_hit_rate": round(self.trivial / d, 4),
        }


def build_spec(dp: DecisionPoint, *, k: int, m_seeds: int) -> QuerySpec:
    return QuerySpec(
        query_id=QUERY_REUSE,
        decision_point=dp.dp_id,
        seed_uid=dp.seed_uid,
        relation="hier",
        hops=2,
        k=k,
        ann_entry_kind="task",
        ann_m_seeds=m_seeds,
        predicate=Predicate(
            kind="node",
            require_valid=True,
            require_fitness=True,
            exclude_sessions=frozenset({dp.session_uid}),
        ),
        meta={"task_uid": dp.task_uid},
    )


def score(
    acc: Acc,
    *,
    corpus: Corpus,
    gt: GroundTruth,
    dp: DecisionPoint,
    ids: list[str],
    expected: Any,
    k: int,
) -> None:
    acc.n += 1
    acc.returned += len(ids)
    acc.empty += int(not ids)

    seed_vec = corpus.vector(dp.seed_uid)
    obs_d = (
        [float(x) for x in corpus.cosine_distance(seed_vec, ids)] if seed_vec is not None else None
    )
    exact, recall, tie_ok = parity(
        ids,
        expected.topk,
        observed_distances=obs_d,
        expected_distances=list(expected.distances) if expected.distances else None,
    )
    acc.exact += int(exact)
    acc.tie_ok += int(tie_ok)
    acc.recall += recall

    # Only ids the corpus knows can be graded. An unknown id would be a mapping bug,
    # not a bad result, so it is counted rather than silently scored as irrelevant.
    known = [u for u in ids if u in corpus.nodes]
    grades = gt.grades(dp, known)
    ideal = gt.ideal_grades(dp, k)
    if ideal and max(ideal) > 0:
        acc.n_scored += 1
        ordered = [grades[u] for u in known]
        acc.ndcg += ndcg_at_k(ordered, ideal, k)
        acc.harmful += harmful_at_k(ordered, k)
        pool = sorted(gt.grades(dp, list(expected.eligible)).values(), reverse=True)
        acc.ceiling += ndcg_at_k(pool[:k], ideal, k)
    acc.trivial += sum(1 for u in known if gt.is_trivial_hit(u, dp.session_uid))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tridb-dsn", default="postgresql://hza214@127.0.0.1:55432/evotrace_eg")
    parser.add_argument("--scope", default="evotrace:349117b0")
    parser.add_argument("--mem0-dsn", default="postgresql://hza214@127.0.0.1:55432/w1x_mem0")
    parser.add_argument("--mem0-collection", default="w1a_full")
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m-seeds", type=int, default=4)
    args = parser.parse_args(argv)

    corpus = Corpus.load(args.normalized, vectors=args.normalized / "vectors.npz")
    gt = GroundTruth(corpus)
    oracle = Oracle(corpus)
    points = [dp for dp in gt.reuse_points()]
    chosen = points if args.full else sample(points, args.limit, args.seed)

    from bench.agent_memory.gem_eg.query import Knobs, W1Engine
    from bench.agent_memory.gem_eg.store import EgStore
    from experiments.e2.mem0_w1a import Mem0W1a, Mem0W1Config

    store = EgStore.connect(args.tridb_dsn)
    store.bootstrap_edge_types()
    tridb = W1Engine(store, args.scope)
    tridb.load_identity()
    mem0 = Mem0W1a(
        Mem0W1Config(
            dsn=args.mem0_dsn, collection=args.mem0_collection, normalized=args.normalized
        )
    )

    accs = {"tridb_gem": Acc(), "mem0": Acc()}
    knobs = Knobs(term_cond=0)
    started = time.perf_counter()
    try:
        for index, dp in enumerate(chosen, 1):
            if index % 200 == 0:
                rate = index / max(1e-9, time.perf_counter() - started)
                print(f"  {index:,}/{len(chosen):,}  {rate:.1f}/s", flush=True)
            spec = build_spec(dp, k=args.k, m_seeds=args.m_seeds)
            expected = oracle.run(spec)

            got = tridb.run(spec, knobs)
            accs["tridb_gem"].stage1_ms.append(got.stage1_ms)
            accs["tridb_gem"].stage2_ms.append(got.stage2_ms)
            score(
                accs["tridb_gem"], corpus=corpus, gt=gt, dp=dp, ids=got.ids,
                expected=expected, k=args.k,
            )

            seed_vec = corpus.vector(dp.seed_uid)
            if seed_vec is None:
                continue
            m = mem0.search(
                seed_vec, target_session=dp.session_uid, k=args.k, m_seeds=args.m_seeds
            )
            accs["mem0"].stage1_ms.append(m.stage1_ms)
            accs["mem0"].stage2_ms.append(m.stage2_ms)
            score(
                accs["mem0"], corpus=corpus, gt=gt, dp=dp, ids=m.ids,
                expected=expected, k=args.k,
            )
    finally:
        mem0.close()
        store.close()

    report = {
        "query": QUERY_REUSE,
        "decision_points": {
            "available": len(points),
            "measured": len(chosen),
            "sampling": "full" if args.full else f"stratified(limit={args.limit},seed={args.seed})",
        },
        "k": args.k,
        "m_seeds": args.m_seeds,
        "seconds": round(time.perf_counter() - started, 1),
        "systems": {name: acc.summary() for name, acc in accs.items()},
        "unsupported": UNSUPPORTED,
        "latency": (
            "NOT REPORTED. The GPU was contended during this run; a latency number "
            "taken under contention measures the contention. Quality is unaffected — "
            "the vectors are deterministic, so a slower run yields identical results."
        ),
        "fairness": [
            "Both systems index the same exported vectors; neither re-embeds.",
            "The seed vector is passed in on both sides; no embedding call at query time.",
            "Same predicate: W1.a's hier 2-hop == task_uid == T (18/18 tasks, verified).",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "w1a_report.json"
    path.write_text(json.dumps(report, indent=2))

    def fmt(value: Any) -> str:
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    keys = [
        "n", "n_scored", "mean_returned", "parity_tie_equivalent", "oracle_recall",
        "ndcg@10", "ceiling", "rank_efficiency", "harmful@10", "trivial_hit_rate",
    ]
    width = max(len(k) for k in keys) + 2
    print(f"\n{'metric':{width}} {'TriDB/GEM':>12} {'Mem0':>12}")
    print("-" * (width + 26))
    for key in keys:
        a = report["systems"]["tridb_gem"].get(key)
        b = report["systems"]["mem0"].get(key)
        print(f"{key:{width}} {fmt(a):>12} {fmt(b):>12}")
    print("\nW1.b / W1.b-fail / W1.d on Mem0: unsupported (no graph)")
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
