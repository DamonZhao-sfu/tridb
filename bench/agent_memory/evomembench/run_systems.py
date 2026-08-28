"""Off-target systems-scale harness for fused versus bounded staged execution."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import threading
import time
from typing import Any, Iterable

from bench.agent_memory.evomembench.modeling import experience_query
from bench.agent_memory.evomembench.scale import (
    BatchExperienceStrategy,
    ScaleRecord,
    iter_scaled_experiences,
)
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.store import vec_literal
from bench.agent_memory.gem.types import InteractionEvent, RetrievalMode
from bench.agent_memory.serving import CallLedger, OpenAIEmbeddingClient, PhasedEmbedder


class DeterministicScaleEmbedder:
    """Smoke-only embedder; never valid for an agent-quality table."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append(
                [
                    ((digest[index % len(digest)] / 255.0) * 2.0) - 1.0
                    for index in range(self.dim)
                ]
            )
        return out


def _batches(values: Iterable[ScaleRecord], size: int) -> Iterable[list[ScaleRecord]]:
    batch: list[ScaleRecord] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_batch(
    memory: TriDBGovernedMemory,
    records: list[ScaleRecord],
    *,
    valid_from: str,
    primary_scope: str,
) -> dict[str, Any]:
    by_scope: dict[str, list[ScaleRecord]] = defaultdict(list)
    for record in records:
        by_scope[record.unit.scope_id].append(record)
    totals = {
        "units": 0,
        "edges": 0,
        "seconds": 0.0,
        "experience_records": len(records),
        "query_eligible_experiences": sum(
            record.unit.scope_id == primary_scope
            and record.unit.metadata.get("validity_state") == "active"
            for record in records
        ),
    }
    for scope_id, scoped in by_scope.items():
        events = [
            InteractionEvent(
                scope_id=scope_id,
                external_id=record.unit.source_external_ids[0],
                content=record.unit.memory_payload,
                kind="systems_decoy",
                event_time=valid_from,
                event_order=record.index,
            )
            for record in scoped
        ]
        result = memory.ingest(
            events,
            strategy=BatchExperienceStrategy(
                [record.unit for record in scoped], valid_from=valid_from
            ),
            scope_id=scope_id,
        )
        if not result.committed:
            raise RuntimeError(result.aborted_reason)
        totals["units"] += result.delta.units_created
        totals["edges"] += result.delta.edges_created
        totals["seconds"] += result.cost.seconds
    return totals


def _staged_reference(
    memory: TriDBGovernedMemory,
    *,
    scope_id: str,
    embedding: list[float],
    cutoff: int,
    k: int,
    m_seeds: int,
    hops: int,
    graph_work_budget: int,
) -> tuple[list[int], dict[str, Any]]:
    """Bounded staged baseline with the same vector/bridge output budget.

    This is a diagnostic comparator, not a correctness oracle: the fused operator
    owns an incremental relaxed-order HNSW stream, whereas this baseline performs
    separate bounded vector, graph, and relational statements.
    """
    conn = memory.store.conn
    predicate = (
        "scope_id=%s AND state='active' AND metadata->>'node_kind'='experience'"
        " AND (metadata->>'experience_ordinal')::integer < %s"
        " AND metadata->>'validity_state'='active'"
    )
    began = time.perf_counter()
    seed_window = max(k, m_seeds * 8, m_seeds + 32)
    vector_rows = [
        (int(row[0]), float(row[1]))
        for row in conn.execute(
            f"SELECT id, embedding <=> %s::vector AS distance"
            f" FROM gem_unit WHERE {predicate}"
            " ORDER BY distance, id LIMIT %s",
            (
                vec_literal(embedding),
                scope_id,
                cutoff,
                seed_window,
            ),
        ).fetchall()
    ]
    seeds = [unit_id for unit_id, _distance in vector_rows[:m_seeds]]
    edge_type = memory.store.edge_type_id("association")
    graph_candidate_ids = set(seeds)
    graph_examined = 0
    graph_censored = False
    budget_remaining = graph_work_budget
    for seed in seeds:
        if budget_remaining <= 0:
            graph_censored = True
            break
        before = int(
            conn.execute("SELECT graph_store.gph_visits()::bigint").fetchone()[0]
        )
        reached = conn.execute(
            "SELECT graph_store.gph_traverse_bounded(%s, %s, %s, %s)::bigint",
            (seed, hops, edge_type, budget_remaining),
        ).fetchall()
        after = int(
            conn.execute("SELECT graph_store.gph_visits()::bigint").fetchone()[0]
        )
        consumed = max(0, after - before)
        graph_examined += consumed
        budget_remaining = max(0, budget_remaining - consumed)
        graph_censored = graph_censored or bool(
            conn.execute(
                "SELECT graph_store.gph_traverse_bounded_censored()"
            ).fetchone()[0]
        )
        graph_candidate_ids.update(int(row[0]) for row in reached)

    graph_rows: list[tuple[int, float]] = []
    if graph_candidate_ids:
        graph_rows = [
            (int(row[0]), float(row[1]))
            for row in conn.execute(
                f"SELECT id, embedding <=> %s::vector AS distance"
                f" FROM gem_unit WHERE {predicate} AND id = ANY(%s)"
                " ORDER BY distance, id",
                (
                    vec_literal(embedding),
                    scope_id,
                    cutoff,
                    list(graph_candidate_ids),
                ),
            ).fetchall()
        ]
    ids = _merge_membership_candidates(vector_rows, graph_rows, k=k)
    return ids, {
        "mode": "same_pg_bounded_staged_membership_baseline",
        "seconds": time.perf_counter() - began,
        "vector_seeds": len(seeds),
        "vector_candidates_materialized": len(vector_rows),
        "graph_candidates_materialized": len(graph_rows),
        "graph_examined": graph_examined,
        "graph_censored": graph_censored,
        "graph_work_budget": graph_work_budget,
        "materializes_bounded_intermediate": True,
        "product_path": False,
        "correctness_oracle": False,
    }


def _merge_membership_candidates(
    vector_rows: list[tuple[int, float]],
    graph_rows: list[tuple[int, float]],
    *,
    k: int,
) -> list[int]:
    """Apply tjs membership's bridge-cap merge to two bounded candidate lists."""
    if k < 1:
        raise ValueError("k must be positive")
    vectors = sorted(dict(vector_rows).items(), key=lambda item: (item[1], item[0]))
    bridges = sorted(dict(graph_rows).items(), key=lambda item: (item[1], item[0]))
    bridge_cap = k // 2
    if bridge_cap == 0 and bridges:
        bridge_cap = 1
    selected: list[tuple[int, float]] = []
    selected_ids: set[int] = set()

    def admit(rows: list[tuple[int, float]], limit: int) -> None:
        for unit_id, distance in rows:
            if len(selected) >= limit:
                return
            if unit_id not in selected_ids:
                selected.append((unit_id, distance))
                selected_ids.add(unit_id)

    admit(bridges, min(k, bridge_cap))
    admit(vectors, k)
    admit(bridges, k)
    selected.sort(key=lambda item: (item[1], item[0]))
    return [unit_id for unit_id, _distance in selected]


def _footprint(memory: TriDBGovernedMemory) -> dict[str, int | None]:
    conn = memory.store.conn
    row = conn.execute(
        "SELECT pg_relation_size('gem_unit')::bigint,"
        " pg_indexes_size('gem_unit')::bigint,"
        " pg_total_relation_size('gem_field_value')::bigint,"
        " pg_total_relation_size('gem_edge')::bigint"
    ).fetchone()
    native = conn.execute(
        "SELECT graph_store.gph_vertex_count()::bigint,"
        " graph_store.gph_edge_count()::bigint,"
        " pg_total_relation_size('graph_store.gstore')::bigint"
    ).fetchone()
    return {
        "gem_unit_heap_bytes": int(row[0]),
        "gem_unit_index_bytes": int(row[1]),
        "gem_field_total_bytes": int(row[2]),
        "gem_edge_metadata_total_bytes": int(row[3]),
        "native_graph_vertices_global": int(native[0]),
        "native_graph_edges_global": int(native[1]),
        "native_graph_relation_bytes_global": int(native[2]),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 < fraction <= 1:
        raise ValueError("percentile requires values and 0 < fraction <= 1")
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _open_loop(
    *,
    dsn: str,
    dim: int,
    query: Any,
    qps: float,
    duration: float,
    concurrency: int,
    graph_scoring: str,
    graph_work_budget: int,
) -> dict[str, Any]:
    """Open-loop arrivals; every worker owns its PostgreSQL connection."""
    if qps <= 0 or duration <= 0 or concurrency < 1:
        raise ValueError("invalid open-loop operating point")
    if qps * duration < 1:
        raise ValueError("open-loop duration must admit at least one request")
    tls = threading.local()
    connections: list[TriDBGovernedMemory] = []
    connections_lock = threading.Lock()

    def invoke(admitted_at: float) -> dict[str, Any]:
        worker = getattr(tls, "memory", None)
        if worker is None:
            worker = TriDBGovernedMemory.connect(dsn, dim=dim)
            worker.store.conn.execute(
                "SELECT set_config('tjs.graph_scoring', %s, false)",
                (graph_scoring,),
            )
            worker.store.conn.execute(
                "SELECT set_config('tjs.graph_work_budget', %s, false)",
                (str(graph_work_budget),),
            )
            tls.memory = worker
            with connections_lock:
                connections.append(worker)
        began = time.perf_counter()
        result = worker.retrieve(query)
        completed = time.perf_counter()
        return {
            "committed": result.committed,
            "queue_seconds": began - admitted_at,
            "service_seconds": completed - began,
            "e2e_seconds": completed - admitted_at,
            "returned": len({hit.unit_id for hit in result.hits}),
            "censored": bool(result.probes.get("graph_censored")),
        }

    count = int(qps * duration)
    started = time.perf_counter()
    futures = []
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for index in range(count):
                admitted_at = started + index / qps
                remaining = admitted_at - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                futures.append(executor.submit(invoke, admitted_at))
            samples = [future.result() for future in futures]
    finally:
        for connection in connections:
            connection.close()
    e2e = [sample["e2e_seconds"] for sample in samples]
    elapsed = time.perf_counter() - started
    observation_window = max(duration, elapsed)
    return {
        "offered_qps": qps,
        "duration_seconds": duration,
        "concurrency": concurrency,
        "requests": count,
        "committed": sum(bool(sample["committed"]) for sample in samples),
        "achieved_qps": count / observation_window,
        "latency_seconds_p50": median(e2e),
        "latency_seconds_p95": _percentile(e2e, 0.95),
        "latency_seconds_p99": _percentile(e2e, 0.99),
        "timeout_or_error_fraction": sum(not sample["committed"] for sample in samples)
        / count,
        "censor_fraction": sum(bool(sample["censored"]) for sample in samples) / count,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing existing systems output: {output}")
    if args.graph_work_budget < 128:
        raise ValueError("graph_work_budget must be >= 128")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.embedding_mode == "deterministic_smoke":
        embedder: Any = DeterministicScaleEmbedder(args.embedding_dim)
        ledger: CallLedger | None = None
    else:
        ledger = CallLedger()
        embedding_client = OpenAIEmbeddingClient(
            args.embedding_base_url,
            args.api_key,
            args.embedding_model,
            batch_size=args.embedding_batch_size,
            timeout=args.timeout,
            ledger=ledger,
        )
        embedder = PhasedEmbedder(embedding_client)
    memory = TriDBGovernedMemory.connect(
        args.dsn, dim=args.embedding_dim, embedder=embedder
    )
    memory.init_schema()
    memory.store.conn.execute(
        "SELECT set_config('tjs.graph_scoring', %s, false)",
        (args.graph_scoring,),
    )
    memory.store.conn.execute(
        "SELECT set_config('tjs.graph_work_budget', %s, false)",
        (str(args.graph_work_budget),),
    )
    valid_from = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    loaded = 0
    loaded_eligible = 0
    try:
        target_embedding = [
            float(value) for value in embedder.encode([args.task_signature])[0]
        ]
        for target_size in args.history_sizes:
            if target_size <= loaded:
                raise ValueError("history sizes must be strictly increasing")
            generated = iter_scaled_experiences(
                count=target_size - loaded,
                primary_scope=args.scope_id,
                seed=args.seed + loaded,
            )
            wal_before = memory.store.conn.execute(
                "SELECT pg_current_wal_lsn()"
            ).fetchone()[0]
            construction = {
                "units": 0,
                "edges": 0,
                "seconds": 0.0,
                "experience_records": 0,
                "query_eligible_experiences": 0,
            }
            for batch in _batches(generated, args.ingest_batch_size):
                observed = _load_batch(
                    memory,
                    batch,
                    valid_from=valid_from,
                    primary_scope=args.scope_id,
                )
                for key in construction:
                    construction[key] += observed[key]
            loaded_eligible += construction["query_eligible_experiences"]
            wal_after = memory.store.conn.execute(
                "SELECT pg_current_wal_lsn()"
            ).fetchone()[0]
            wal_bytes = int(
                memory.store.conn.execute(
                    "SELECT pg_wal_lsn_diff(%s,%s)", (wal_after, wal_before)
                ).fetchone()[0]
            )
            loaded = target_size
            cutoff = target_size + 1
            query = experience_query(
                scope_id=args.scope_id,
                task_signature=args.task_signature,
                cutoff_ordinal=cutoff,
                mode=RetrievalMode.FUSED,
                top_k=args.k,
                hops=args.hops,
                reinforce=False,
                validity_states=("active",),
            )
            query = replace(
                query,
                embedding=target_embedding,
                m_seeds=args.m_seeds,
                term_cond=args.term_cond,
            )
            began = time.perf_counter()
            fused = memory.retrieve(query)
            fused_seconds = time.perf_counter() - began
            if not fused.committed:
                raise RuntimeError(fused.aborted_reason)
            fused_ids = list(dict.fromkeys(hit.unit_id for hit in fused.hits))
            staged_ids, staged = _staged_reference(
                memory,
                scope_id=args.scope_id,
                embedding=target_embedding,
                cutoff=cutoff,
                k=args.k,
                m_seeds=args.m_seeds,
                hops=args.hops,
                graph_work_budget=args.graph_work_budget,
            )
            union = set(fused_ids) | set(staged_ids)
            rows.append(
                {
                    "history_size": target_size,
                    "construction": construction,
                    "relational_filter_population": {
                        "source": "deterministic_generator_labels_not_a_database_scan",
                        "total_experiences": loaded,
                        "eligible_experiences": loaded_eligible,
                        "eligibility_fraction": loaded_eligible / loaded,
                    },
                    "wal_bytes": wal_bytes,
                    "wal_scope_note": (
                        "server-global WAL delta over the construction interval; "
                        "run without concurrent writers for attribution"
                    ),
                    "footprint": _footprint(memory),
                    "fused": {
                        "seconds": fused_seconds,
                        "ids": fused_ids,
                        "probes": dict(fused.probes),
                        "cost": asdict(fused.cost),
                        "streaming_contract": "Open/Next/Close with early termination",
                    },
                    "staged_reference": {**staged, "ids": staged_ids},
                    "returned_set_overlap_jaccard": (
                        len(set(fused_ids) & set(staged_ids)) / len(union)
                        if union
                        else 1.0
                    ),
                    "hardware_claim": "x86 stock-PG off-target only",
                    "warm_open_loop": [
                        _open_loop(
                            dsn=args.dsn,
                            dim=args.embedding_dim,
                            query=query,
                            qps=qps,
                            duration=args.qps_duration,
                            concurrency=args.concurrency,
                            graph_scoring=args.graph_scoring,
                            graph_work_budget=args.graph_work_budget,
                        )
                        for qps in args.qps
                    ],
                }
            )
    finally:
        memory.close()
    result = {
        "schema_version": "evomembench_gem_systems_v0.1.0",
        "status": "complete",
        "agent_outcome_measured": False,
        "outcome_valid": None,
        "scope_id": args.scope_id,
        "embedding_mode": args.embedding_mode,
        "embedding": {
            "model": args.embedding_model,
            "dimension": args.embedding_dim,
            "revision": (
                args.embedding_revision if args.embedding_mode == "real" else None
            ),
        },
        "embedding_calls": None if ledger is None else ledger.summary(),
        "embedding_tokens": None if ledger is None else ledger.tokens(),
        "operating_point": {
            "k": args.k,
            "m_seeds": args.m_seeds,
            "hops": args.hops,
            "term_cond": args.term_cond,
            "graph_scoring": args.graph_scoring,
            "graph_work_budget": args.graph_work_budget,
        },
        "points": rows,
        "limitations": [
            "systems-only synthetic histories do not support an agent accuracy claim",
            "returned-set overlap is diagnostic and is not a correctness or relevance metric",
            "the staged comparator materializes only explicitly bounded intermediates and is not a product path",
            "TTFR/time-to-k client cursor instrumentation is not yet available",
            "cold-cache OS/page-cache eviction is not performed by this harness",
            "GX10 ARM64/CUDA/128GB sign-off is not performed on this host",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--task-signature", required=True)
    parser.add_argument("--history-sizes", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--ingest-batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m-seeds", type=int, default=8)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--term-cond", type=int, default=64)
    parser.add_argument(
        "--graph-scoring",
        choices=["membership"],
        default="membership",
        help="staged comparison currently supports membership semantics only",
    )
    parser.add_argument("--graph-work-budget", type=int, default=65536)
    parser.add_argument("--qps", nargs="+", type=float, default=[1.0, 5.0, 10.0])
    parser.add_argument("--qps-duration", type=float, default=10.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument(
        "--embedding-mode",
        choices=["real", "deterministic_smoke"],
        default="real",
    )
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--embedding-revision",
        default="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    )
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
