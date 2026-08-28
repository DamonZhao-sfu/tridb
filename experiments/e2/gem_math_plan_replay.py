"""Deterministic Track-C replay for GEM+ EvoTrace Math/ALE physical plans.

The runner performs two warm-ups and seven measured warm repetitions per
``(query, plan)`` by default.  It writes raw rows, the selected-query manifest, and a
summary containing p50/p95 total and stage latency.  Stored node vectors are replayed,
so ``embed_ms=0`` is explicitly labelled offline; live embedding is measured by the
agent runner.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.oracle import (
    Oracle,
    Predicate,
    QuerySpec,
    logical_spec_hash,
    recall_at_k,
)
from bench.agent_memory.gem_eg.physical import PhysicalExecutor, PhysicalPlan
from bench.agent_memory.gem_eg.query import Knobs, W1Engine
from bench.agent_memory.gem_eg.store import DEFAULT_DSN, EgStore

DEFAULT_SCOPE = "evotrace:349117b0"
DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
DEFAULT_VECTORS = DEFAULT_NORMALIZED / "vectors.npz"
STAGES = (
    "embed_ms",
    "ann_ms",
    "graph_ms",
    "predicate_ms",
    "dedup_rank_ms",
    "hydrate_ms",
    "executor_overhead_ms",
    "retriever_total_ms",
)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * q))
    return round(float(ordered[index]), 6)


def _select_queries(corpus: Corpus, task_uid: str, limit: int | None) -> list[str]:
    rows = [
        uid
        for uid, node in corpus.nodes.items()
        if node["task_uid"] == task_uid
        and node.get("is_valid")
        and node.get("fitness") is not None
        and math.isfinite(float(node["fitness"]))
        and corpus.vector(uid) is not None
    ]
    rows.sort(key=lambda uid: (float(corpus.nodes[uid]["fitness"]), uid))
    if limit is None or len(rows) <= limit:
        return rows
    if limit <= 1:
        return rows[:1]
    # Even fitness-rank spacing is deterministic and records its concrete IDs in the
    # manifest, so a capped run remains reproducible rather than "first N" biased.
    indexes = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)]
    return [rows[i] for i in indexes]


def _spec(corpus: Corpus, uid: str) -> QuerySpec:
    node = corpus.nodes[uid]
    artifact = node.get("artifact_uid")
    duplicates = frozenset(
        other_uid
        for other_uid, other in corpus.nodes.items()
        if artifact is not None and other.get("artifact_uid") == artifact
    )
    vector = corpus.vector(uid)
    assert vector is not None
    return QuerySpec(
        query_id="gem_math_plan_replay",
        decision_point=uid,
        seed_uid=uid,
        relation="lineage",
        hops=3,
        predicate=Predicate(
            kind="node",
            require_valid=True,
            require_fitness=True,
            require_finite_fitness=True,
            fitness_gt=float(node["fitness"]),
            include_tasks=frozenset({node["task_uid"]}),
            exclude_sessions=frozenset({node["session_uid"]}),
            exclude_nodes=duplicates,
        ),
        k=10,
        rank_by="seed_fitness",
        query_vector=tuple(float(value) for value in vector),
        ann_entry_kind="node",
        ann_m_seeds=4,
        meta={"task_uid": node["task_uid"], "source": "stored_vector_replay"},
    )


def _hydrate(conn: Any, uids: list[str]) -> float:
    started = time.perf_counter()
    if uids:
        conn.execute(
            "SELECT n.node_uid,a.payload FROM gem_eg_node n"
            " JOIN gem_eg_artifact a ON a.artifact_uid=n.artifact_uid"
            " WHERE n.node_uid=ANY(%s)",
            (uids,),
        ).fetchall()
    return (time.perf_counter() - started) * 1000.0


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task_uid"], row["physical_plan"])].append(row)
        domain = row["task_uid"].split(":", 1)[0]
        groups[(f"__all_{domain}__", row["physical_plan"])].append(row)
        groups[("__all__", row["physical_plan"])].append(row)
    out: list[dict[str, Any]] = []
    for (task, plan), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "task_uid": task,
            "physical_plan": plan,
            "measured_rows": len(group),
            "queries": len({row["query_uid"] for row in group}),
            "exact_order_rate": round(
                statistics.fmean(float(row["exact_order"]) for row in group), 6
            ),
            "recall_at_10": round(
                statistics.fmean(float(row["recall_at_10"]) for row in group), 6
            ),
        }
        for stage in STAGES:
            values = [float(row[stage]) for row in group]
            item[f"{stage}_p50"] = _percentile(values, 0.50)
            item[f"{stage}_p95"] = _percentile(values, 0.95)
        out.append(item)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--scope", default=DEFAULT_SCOPE)
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--queries-per-task", type=int, default=None)
    parser.add_argument(
        "--domains", nargs="+", choices=("math", "ale"), default=("math",)
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="optional task UID subset for smoke/sensitivity runs",
    )
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--graph-work-budget", type=int, default=65536)
    parser.add_argument("--exact-ann", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 7:
        parser.error("headline Track-S cells require at least 7 measured repetitions")

    corpus = Corpus.load(args.normalized, vectors=args.vectors)
    domains = set(args.domains)
    tasks = sorted(
        uid for uid, row in corpus.tasks.items() if row.get("domain") in domains
    )
    if args.tasks:
        unknown = sorted(set(args.tasks) - set(tasks))
        if unknown:
            parser.error(f"unknown/tasks outside selected domains: {unknown}")
        tasks = sorted(args.tasks)
    selected = {
        task: _select_queries(corpus, task, args.queries_per_task) for task in tasks
    }
    store = EgStore.connect(args.dsn)
    engine = W1Engine(store, args.scope)
    engine.load_identity()
    executor = PhysicalExecutor(engine)
    plans = list(PhysicalPlan)
    for plan in plans:
        executor.ensure_ready(plan)
    oracle = Oracle(corpus)
    knobs = Knobs(
        ef_search=args.ef_search,
        graph_work_budget=args.graph_work_budget,
        ann_exact=args.exact_ann,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "gem_math_ale_plan_replay_v2",
        "scope_id": args.scope,
        "domains": sorted(domains),
        "tasks": tasks,
        "selected_queries": selected,
        "selection": "all eligible"
        if args.queries_per_task is None
        else "fitness-rank spaced",
        "warmups": args.warmups,
        "measured_repetitions": args.repetitions,
        "plans": [plan.value for plan in plans],
        "logical_contract": {
            "m": 4,
            "h": 3,
            "k": 10,
            "similarity_threshold": None,
            "ranking": "best_reaching_seed_distance ASC, fitness DESC, node_uid ASC",
        },
        "offline_embedding_replay": True,
        "embed_ms": 0.0,
        "knobs": {
            "ef_search": args.ef_search,
            "graph_work_budget": args.graph_work_budget,
            "exact_ann": args.exact_ann,
            "aivg_seed_prefixes": list(knobs.aivg_seed_prefixes),
        },
        "environment": {
            "postgres": store.conn.execute("SELECT version()").fetchone()[0],
            "extensions": dict(
                store.conn.execute(
                    "SELECT extname,extversion FROM pg_extension"
                    " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY 1"
                ).fetchall()
            ),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    measured: list[dict[str, Any]] = []
    raw_path = args.out / "raw.jsonl"
    try:
        with raw_path.open("w", encoding="utf-8") as raw:
            for task in tasks:
                for query_index, uid in enumerate(selected[task]):
                    spec = _spec(corpus, uid)
                    expected = oracle.run(spec)
                    rng = random.Random(42 + query_index)
                    for repetition in range(-args.warmups, args.repetitions):
                        order = plans.copy()
                        rng.shuffle(order)
                        for plan in order:
                            result = engine.run(spec, knobs, plan.value)
                            hydrate_ms = _hydrate(store.conn, result.ids)
                            total = result.total_ms + hydrate_ms
                            classified = (
                                result.ann_ms
                                + result.graph_ms
                                + result.predicate_ms
                                + result.dedup_rank_ms
                                + hydrate_ms
                            )
                            row = {
                                "task_uid": task,
                                "query_uid": uid,
                                "logical_spec_hash": logical_spec_hash(spec),
                                "physical_plan": plan.value,
                                "repetition": repetition,
                                "warmup": repetition < 0,
                                "ids": result.ids,
                                "oracle_ids": list(expected.topk),
                                "exact_order": result.ids == list(expected.topk),
                                "recall_at_10": recall_at_k(result.ids, expected.topk),
                                "termination_reason": result.termination_reason,
                                "graph_censored": result.graph_censored,
                                "embed_ms": 0.0,
                                "ann_ms": result.ann_ms,
                                "graph_ms": result.graph_ms,
                                "predicate_ms": result.predicate_ms,
                                "dedup_rank_ms": result.dedup_rank_ms,
                                "hydrate_ms": hydrate_ms,
                                "executor_overhead_ms": max(0.0, total - classified),
                                "retriever_total_ms": total,
                                "first_row_ms": result.first_row_ms,
                                "ann_candidates": result.ann_candidates,
                                "ann_prefixes": result.ann_prefixes,
                                "seeds_consumed": result.seeds_consumed,
                                "graph_examined": result.graph_examined,
                                "predicate_probes": result.predicate_probes,
                                "predicate_passed": result.predicate_passed,
                                "reverse_membership_probes": result.reverse_membership_probes,
                                "dedup_hits": result.dedup_hits,
                            }
                            raw.write(json.dumps(row) + "\n")
                            raw.flush()
                            if repetition >= 0:
                                measured.append(row)
    finally:
        store.close()
    summary = _summary(measured)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
