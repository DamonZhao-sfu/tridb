"""Matched-quality live comparison of fused tjs_open versus bounded staged execution.

Both systems read the exact same completed GEM scope.  The staged baseline performs a
bounded vector window, bounded native-graph pulls, and a relationally filtered final
rank as separate client/server stages; it never materializes an unbounded intermediate.
Query embedding is computed once and excluded from both database-path timings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Sequence

from psycopg import sql

from bench.agent_memory.gem.store import GemStore, vec_literal
from bench.agent_memory.memoryarena.dataset import load_export
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-namespace", default="memoryarena_agent")
    parser.add_argument("--max-tasks", type=int, default=221)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m-seeds", type=int, default=4)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--term-cond", type=int, default=128)
    parser.add_argument("--graph-work-budget", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args(argv)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _quality(
    selected_ids: Sequence[int], relevant_ids: Sequence[int], *, k: int
) -> dict[str, float | int]:
    """Grade one result against the release's all-prior dependency protocol."""
    relevant = set(relevant_ids)
    selected = list(selected_ids[:k])
    matched = [unit_id in relevant for unit_id in selected]
    recall = sum(matched) / len(relevant) if relevant else 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, is_relevant in enumerate(matched)
        if is_relevant
    )
    ideal_count = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return {
        "dependency_recall_at_10": recall,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "returned": len(selected),
        "constraint_violations": sum(not value for value in matched),
    }


def _task_bootstrap(
    task_deltas: Mapping[str, Sequence[float]], *, iterations: int, seed: int
) -> dict[str, Any]:
    task_ids = sorted(task_deltas)
    means = [statistics.fmean(task_deltas[task]) for task in task_ids]
    generator = random.Random(seed)
    samples = [
        statistics.fmean(means[generator.randrange(len(means))] for _ in means)
        for _ in range(iterations)
    ]
    return {
        "n_tasks": len(task_ids),
        "mean_delta_ms_staged_minus_fused": statistics.fmean(means),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "method": "paired_task_cluster_percentile_bootstrap",
        "iterations": iterations,
        "seed": seed,
    }


def _filter_sql(conn: Any, scope_id: str, cutoff: int) -> str:
    predicate = sql.SQL(
        "scope_id={} AND state='active' AND "
        "(metadata->>'session_ordinal')::integer < {}"
    ).format(sql.Literal(scope_id), sql.Literal(cutoff))
    return predicate.as_string(conn)


def _pull_ids(
    conn: Any,
    query: str,
    params: Sequence[Any],
    *,
    limit: int,
    setup: Sequence[tuple[str, str]] = (),
) -> tuple[list[int], float | None, float]:
    started = time.perf_counter()
    first_row_ms: float | None = None
    ids: list[int] = []
    with conn.transaction():
        for name, value in setup:
            conn.execute("SELECT set_config(%s, %s, true)", (name, value))
        cursor_name = f"ma_stage_{time.perf_counter_ns()}"
        with conn.cursor(name=cursor_name) as cursor:
            cursor.itersize = 1
            cursor.execute(query, tuple(params))
            for row in cursor:
                if row[0] is None:
                    continue
                if first_row_ms is None:
                    first_row_ms = (time.perf_counter() - started) * 1000.0
                ids.append(int(row[0]))
                if len(ids) >= limit:
                    break
    return ids, first_row_ms, (time.perf_counter() - started) * 1000.0


def _fused(
    conn: Any,
    *,
    vector: Sequence[float],
    scope_id: str,
    cutoff: int,
    k: int,
    term_cond: int,
    m_seeds: int,
    hops: int,
    graph_work_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    ids: list[int] = []
    first_row_ms: float | None = None
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
        conn.execute(
            "SELECT set_config('tjs.graph_work_budget', %s, true)",
            (str(graph_work_budget),),
        )
        cursor_name = f"ma_fused_{time.perf_counter_ns()}"
        with conn.cursor(name=cursor_name) as cursor:
            cursor.itersize = 1
            cursor.execute(
                "SELECT tjs_open('gem_unit'::regclass,%s,%s,%s,%s,'id',%s,"
                " %s::vector,NULL,0)",
                (
                    k,
                    term_cond,
                    m_seeds,
                    hops,
                    _filter_sql(conn, scope_id, cutoff),
                    vec_literal(vector),
                ),
            )
            for row in cursor:
                if row[0] is None:
                    continue
                if first_row_ms is None:
                    first_row_ms = (time.perf_counter() - started) * 1000.0
                ids.append(int(row[0]))
                if len(ids) >= k:
                    break
        probes = conn.execute(
            "SELECT tjs_open_candidates_examined(),"
            " tjs_open_graph_examined(), tjs_open_graph_censored(),"
            " tjs_open_termination_reason(), tjs_open_budget_capped(),"
            " tjs_open_bridges_injected()"
        ).fetchone()
    return {
        "selected_ids": ids,
        "first_row_ms": first_row_ms,
        "time_to_k_ms": (time.perf_counter() - started) * 1000.0,
        "candidates_examined": int(probes[0]) if probes[0] is not None else None,
        "visited_nodes": None,
        "visited_edges": int(probes[1]) if probes[1] is not None else None,
        "graph_censored": bool(probes[2]),
        "termination_reason": str(probes[3]),
        "budget_capped": probes[4],
        "bridges_injected": int(probes[5]),
    }


def _staged(
    conn: Any,
    *,
    vector: Sequence[float],
    scope_id: str,
    cutoff: int,
    edge_type: int,
    k: int,
    m_seeds: int,
    hops: int,
    graph_work_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_window = max(m_seeds * 8, m_seeds + 32)
    vector_ids, vector_first_ms, vector_ms = _pull_ids(
        conn,
        "SELECT id FROM gem_unit WHERE scope_id=%s AND state='active'"
        " AND (metadata->>'session_ordinal')::integer < %s"
        " ORDER BY embedding <=> %s::vector LIMIT %s",
        (scope_id, cutoff, vec_literal(vector), seed_window),
        limit=seed_window,
        setup=(("hnsw.iterative_scan", "strict_order"),),
    )
    seeds = vector_ids[:m_seeds]
    candidate_ids = list(vector_ids)
    seen = set(candidate_ids)
    graph_started = time.perf_counter()
    graph_edges = 0
    graph_censored = False
    remaining_budget = graph_work_budget
    for seed_id in seeds:
        if remaining_budget <= 0:
            graph_censored = True
            break
        before = int(conn.execute("SELECT graph_store.gph_visits()").fetchone()[0])
        reached, _, _ = _pull_ids(
            conn,
            "SELECT graph_store.gph_traverse_bounded(%s,%s,%s,%s)",
            (seed_id, hops, edge_type, remaining_budget),
            limit=remaining_budget + 1,
        )
        after = int(conn.execute("SELECT graph_store.gph_visits()").fetchone()[0])
        used = after - before
        graph_edges += used
        remaining_budget -= used
        graph_censored = graph_censored or bool(
            conn.execute(
                "SELECT graph_store.gph_traverse_bounded_censored()"
            ).fetchone()[0]
        )
        for unit_id in reached:
            if unit_id not in seen:
                seen.add(unit_id)
                candidate_ids.append(unit_id)
    graph_ms = (time.perf_counter() - graph_started) * 1000.0
    final_started = time.perf_counter()
    pre_final_ms = (final_started - started) * 1000.0
    if candidate_ids:
        selected_ids, final_first_ms, _ = _pull_ids(
            conn,
            "SELECT id FROM gem_unit WHERE id=ANY(%s) AND scope_id=%s"
            " AND state='active' AND (metadata->>'session_ordinal')::integer < %s"
            " ORDER BY embedding <=> %s::vector LIMIT %s",
            (candidate_ids, scope_id, cutoff, vec_literal(vector), k),
            limit=k,
        )
    else:
        selected_ids = []
        final_first_ms = None
    final_rank_ms = (time.perf_counter() - final_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "selected_ids": selected_ids,
        # A staged consumer cannot emit its final first row before all preceding
        # stages finish.  This is the end-to-end TTFR, not vector-stage TTFR.
        "first_row_ms": (
            pre_final_ms + final_first_ms if final_first_ms is not None else None
        ),
        "time_to_k_ms": total_ms,
        "vector_stage_first_row_ms": vector_first_ms,
        "vector_stage_ms": vector_ms,
        "graph_stage_ms": graph_ms,
        "final_filter_rank_ms": final_rank_ms,
        "vector_window_candidates": len(vector_ids),
        "bounded_union_candidates": len(candidate_ids),
        "visited_nodes": None,
        "visited_edges": graph_edges,
        "graph_censored": graph_censored,
    }


def _scope_digest(conn: Any, scope_ids: Sequence[str]) -> str:
    rows = conn.execute(
        "SELECT scope_id,id,state,metadata::text FROM gem_unit"
        " WHERE scope_id=ANY(%s) ORDER BY scope_id,id",
        (list(scope_ids),),
    ).fetchall()
    payload = json.dumps(rows, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if min(args.k, args.m_seeds, args.hops, args.repetitions) < 1:
        raise ValueError("k, m-seeds, hops, and repetitions must be positive")
    if min(args.term_cond, args.graph_work_budget, args.warmups) < 0:
        raise ValueError(
            "term-cond, graph-work-budget, and warmups must be non-negative"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_export(args.export_dir)
    tasks = corpus.tasks[: args.max_tasks]
    decisions = [session for task in tasks for session in task.sessions[1:]]

    ledger = CallLedger()
    client = OpenAIEmbeddingClient(
        args.embedding_base_url,
        "EMPTY",
        args.embedding_model,
        batch_size=64,
        timeout=180.0,
        ledger=ledger,
    )
    if client.discover_models() != [args.embedding_model]:
        raise RuntimeError("embedding endpoint identity mismatch")
    embedder = PhasedEmbedder(client)
    with embedder.phase("query"):
        vectors = embedder.encode([session.question for session in decisions])
    if not vectors or len(vectors[0]) != args.dim:
        raise RuntimeError("embedding dimension mismatch")

    store = GemStore.connect(args.dsn, dim=args.dim)
    conn = store.conn
    scope_by_task = {
        task.task_uid: (
            f"{args.source_namespace}:{args.source_run_id}:gem_fused:{task.task_uid}"
        )
        for task in tasks
    }
    scope_ids = list(scope_by_task.values())
    expected_units = sum(len(task.sessions) for task in tasks)
    observed_units = int(
        conn.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id=ANY(%s)", (scope_ids,)
        ).fetchone()[0]
    )
    if observed_units != expected_units:
        raise RuntimeError(
            f"source scopes incomplete: {observed_units} units != {expected_units}"
        )
    unit_rows = conn.execute(
        "SELECT scope_id,id,(metadata->>'session_ordinal')::integer"
        " FROM gem_unit WHERE scope_id=ANY(%s)",
        (scope_ids,),
    ).fetchall()
    unit_by_scope_ordinal: dict[str, dict[int, int]] = defaultdict(dict)
    for scope_id, unit_id, ordinal in unit_rows:
        scope_units = unit_by_scope_ordinal[str(scope_id)]
        ordinal = int(ordinal)
        if ordinal in scope_units:
            raise RuntimeError(f"duplicate source ordinal: {scope_id}:{ordinal}")
        scope_units[ordinal] = int(unit_id)
    for task in tasks:
        scope_id = scope_by_task[task.task_uid]
        expected_ordinals = set(range(len(task.sessions)))
        if set(unit_by_scope_ordinal[scope_id]) != expected_ordinals:
            raise RuntimeError(f"source ordinal mismatch: {scope_id}")
    edge_type_row = conn.execute(
        "SELECT id FROM graph_store.edge_type WHERE name='association'"
    ).fetchone()
    if edge_type_row is None:
        raise RuntimeError("association native edge type is absent")
    edge_type = int(edge_type_row[0])
    scope_digest_before = _scope_digest(conn, scope_ids)
    global_before = conn.execute(
        "SELECT count(*),graph_store.gph_vertex_count(),"
        " graph_store.gph_visible_edge_count() FROM gem_unit"
    ).fetchone()
    # Load both extension libraries before reading their custom GUC defaults.
    conn.execute("SELECT tjs_open_candidates_examined(),'[0]'::vector").fetchone()
    guc_names = (
        "hnsw.ef_search",
        "hnsw.max_scan_tuples",
        "hnsw.scan_mem_multiplier",
        "tjs.vector_scan_budget",
        "tjs.filter_probe",
    )
    engine_settings = {
        name: conn.execute("SELECT current_setting(%s, true)", (name,)).fetchone()[0]
        for name in guc_names
    }

    rows_path = args.output_dir / "measurements.jsonl"
    task_deltas: defaultdict[str, list[float]] = defaultdict(list)
    fused_times: list[float] = []
    fused_first: list[float] = []
    staged_times: list[float] = []
    staged_first: list[float] = []
    fused_recalls: list[float] = []
    staged_recalls: list[float] = []
    fused_ndcgs: list[float] = []
    staged_ndcgs: list[float] = []
    quality_matches = 0
    exact_order_matches = 0
    exact_set_matches = 0
    constraint_violations = 0
    comparisons = 0
    try:
        with rows_path.open("x", encoding="utf-8") as output:
            for query_index, (session, vector) in enumerate(
                zip(decisions, vectors, strict=True)
            ):
                scope_id = scope_by_task[session.task_uid]
                for phase_index in range(args.warmups + args.repetitions):
                    measured = phase_index >= args.warmups
                    order = (
                        ("fused", "staged")
                        if (query_index + phase_index) % 2 == 0
                        else ("staged", "fused")
                    )
                    observed: dict[str, dict[str, Any]] = {}
                    for system in order:
                        if system == "fused":
                            observed[system] = _fused(
                                conn,
                                vector=vector,
                                scope_id=scope_id,
                                cutoff=session.ordinal,
                                k=args.k,
                                term_cond=args.term_cond,
                                m_seeds=args.m_seeds,
                                hops=args.hops,
                                graph_work_budget=args.graph_work_budget,
                            )
                        else:
                            observed[system] = _staged(
                                conn,
                                vector=vector,
                                scope_id=scope_id,
                                cutoff=session.ordinal,
                                edge_type=edge_type,
                                k=args.k,
                                m_seeds=args.m_seeds,
                                hops=args.hops,
                                graph_work_budget=args.graph_work_budget,
                            )
                    relevant_ids = [
                        unit_by_scope_ordinal[scope_id][ordinal]
                        for ordinal in range(session.ordinal)
                    ]
                    for system in ("fused", "staged"):
                        observed[system]["quality"] = _quality(
                            observed[system]["selected_ids"],
                            relevant_ids,
                            k=args.k,
                        )
                    if not measured:
                        continue
                    exact = (
                        observed["fused"]["selected_ids"]
                        == observed["staged"]["selected_ids"]
                    )
                    comparisons += 1
                    exact_order_matches += int(exact)
                    exact_set_matches += int(
                        set(observed["fused"]["selected_ids"])
                        == set(observed["staged"]["selected_ids"])
                    )
                    fused_quality = observed["fused"]["quality"]
                    staged_quality = observed["staged"]["quality"]
                    quality_match = (
                        abs(
                            float(fused_quality["dependency_recall_at_10"])
                            - float(staged_quality["dependency_recall_at_10"])
                        )
                        <= 1e-12
                        and abs(
                            float(fused_quality["ndcg_at_10"])
                            - float(staged_quality["ndcg_at_10"])
                        )
                        <= 1e-12
                    )
                    quality_matches += int(quality_match)
                    constraint_violations += int(
                        fused_quality["constraint_violations"]
                    ) + int(staged_quality["constraint_violations"])
                    fused_recalls.append(
                        float(fused_quality["dependency_recall_at_10"])
                    )
                    staged_recalls.append(
                        float(staged_quality["dependency_recall_at_10"])
                    )
                    fused_ndcgs.append(float(fused_quality["ndcg_at_10"]))
                    staged_ndcgs.append(float(staged_quality["ndcg_at_10"]))
                    fused_ms = float(observed["fused"]["time_to_k_ms"])
                    staged_ms = float(observed["staged"]["time_to_k_ms"])
                    fused_times.append(fused_ms)
                    staged_times.append(staged_ms)
                    if observed["fused"]["first_row_ms"] is not None:
                        fused_first.append(float(observed["fused"]["first_row_ms"]))
                    if observed["staged"]["first_row_ms"] is not None:
                        staged_first.append(float(observed["staged"]["first_row_ms"]))
                    task_deltas[session.task_uid].append(staged_ms - fused_ms)
                    output.write(
                        json.dumps(
                            {
                                "query_index": query_index,
                                "repetition": phase_index - args.warmups,
                                "execution_order": order,
                                "task_uid": session.task_uid,
                                "session_uid": session.session_uid,
                                "cutoff_ordinal": session.ordinal,
                                "query_sha256": hashlib.sha256(
                                    session.question.encode()
                                ).hexdigest(),
                                "exact_selected_order_match": exact,
                                "quality_match": quality_match,
                                "fused": observed["fused"],
                                "staged": observed["staged"],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    output.flush()
    finally:
        store.close()

    # Reopen so the read-only postcondition is checked independently of the
    # measurement connection's backend-local probe counters.
    verification_store = GemStore.connect(args.dsn, dim=args.dim)
    try:
        scope_digest_after = _scope_digest(verification_store.conn, scope_ids)
        global_after = verification_store.conn.execute(
            "SELECT count(*),graph_store.gph_vertex_count(),"
            " graph_store.gph_visible_edge_count() FROM gem_unit"
        ).fetchone()
        server = verification_store.conn.execute(
            "SELECT version(),current_database()"
        ).fetchone()
        extensions = verification_store.conn.execute(
            "SELECT extname,extversion FROM pg_extension"
            " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY 1"
        ).fetchall()
    finally:
        verification_store.close()
    if scope_digest_after != scope_digest_before or tuple(global_after) != tuple(
        global_before
    ):
        raise RuntimeError("read-only benchmark changed source database state")
    if quality_matches != comparisons or constraint_violations:
        failure = {
            "status": "failed_matched_quality_gate",
            "comparisons": comparisons,
            "quality_matches": quality_matches,
            "constraint_violations": constraint_violations,
            "exact_order_matches_diagnostic": exact_order_matches,
            "exact_set_matches_diagnostic": exact_set_matches,
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            "no matched benchmark-quality point: "
            f"{quality_matches}/{comparisons} quality matches, "
            f"constraint violations={constraint_violations}"
        )

    summary = {
        "schema_version": "memoryarena_fused_vs_staged_v0.2.0",
        "status": "pass_matched_quality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "claim_boundary": (
            "same-process staged execution baseline, not a separately deployed polyglot; "
            "database-path latency excludes the one common precomputed query embedding"
        ),
        "dataset": {
            "config": corpus.config,
            "manifest_sha256": corpus.manifest_sha256,
            "tasks": len(tasks),
            "queries": len(decisions),
        },
        "protocol": {
            "k": args.k,
            "m_seeds": args.m_seeds,
            "hops": args.hops,
            "term_cond": args.term_cond,
            "graph_work_budget": args.graph_work_budget,
            "seed_window": max(args.m_seeds * 8, args.m_seeds + 32),
            "warmups_per_query": args.warmups,
            "repetitions_per_query": args.repetitions,
            "execution_order": "query-level alternating fused/staged",
            "staged_bound": (
                "vector window is seed_window; graph work is edge-step budget; final "
                "candidate union is bounded by window plus graph budget"
            ),
        },
        "quality_gate": {
            "metric": (
                "per-query Dependency Recall@10 and NDCG@10 equality under the "
                "released all-prior dependency protocol"
            ),
            "comparisons": comparisons,
            "matches": quality_matches,
            "match_fraction": quality_matches / comparisons,
            "constraint_violations": constraint_violations,
            "fused_dependency_recall_at_10": _distribution(fused_recalls),
            "staged_dependency_recall_at_10": _distribution(staged_recalls),
            "fused_ndcg_at_10": _distribution(fused_ndcgs),
            "staged_ndcg_at_10": _distribution(staged_ndcgs),
            "diagnostics_not_gates": {
                "exact_selected_order_matches": exact_order_matches,
                "exact_selected_set_matches": exact_set_matches,
            },
            "annotation": (
                "Matched quality does not mean identical IDs: the public release "
                "labels every prior session relevant, so order/identity differences "
                "among eligible prior sessions do not change these benchmark metrics."
            ),
        },
        "latency_ms": {
            "fused_time_to_k": _distribution(fused_times),
            "staged_time_to_k": _distribution(staged_times),
            "fused_time_to_first_row": _distribution(fused_first),
            "staged_time_to_first_row": _distribution(staged_first),
            "paired": _task_bootstrap(
                task_deltas,
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            ),
        },
        "embedding": {
            "model": args.embedding_model,
            "revision": args.embedding_revision,
            "base_url": args.embedding_base_url,
            "calls": ledger.summary(),
            "tokens": ledger.tokens(),
            "timing": "precomputed once and excluded equally from both database paths",
        },
        "engine": {
            "version": server[0],
            "database": server[1],
            "extensions": {name: version for name, version in extensions},
            "settings": engine_settings,
            "hardware_claim": "x86_64 stock-PG only; no GX10 sign-off",
        },
        "read_only_gate": {
            "scope_digest_before": scope_digest_before,
            "scope_digest_after": scope_digest_after,
            "global_counts_before": list(global_before),
            "global_counts_after": list(global_after),
            "violations": 0,
        },
        "artifacts": {"measurements": rows_path.name},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS queries={len(decisions)} comparisons={comparisons} "
        f"quality_match={quality_matches}/{comparisons} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
