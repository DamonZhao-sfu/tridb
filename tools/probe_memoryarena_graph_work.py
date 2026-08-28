"""Measure distinct graph reach and edge steps without mutating frozen run artifacts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from psycopg import sql

from bench.agent_memory.gem.store import GemStore, vec_literal
from bench.agent_memory.memoryarena.dataset import load_export
from bench.agent_memory.serving import CallLedger, OpenAIEmbeddingClient, PhasedEmbedder


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-namespace", required=True)
    parser.add_argument("--custom-library", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, required=True)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--m-seeds", type=int, default=4)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--term-cond", type=int, default=128)
    parser.add_argument("--graph-work-budget", type=int, default=65536)
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


def _scope_digest(conn: Any, scope_ids: Sequence[str]) -> str:
    rows = conn.execute(
        "SELECT scope_id,id,state,metadata::text FROM gem_unit"
        " WHERE scope_id=ANY(%s) ORDER BY scope_id,id",
        (list(scope_ids),),
    ).fetchall()
    return hashlib.sha256(
        json.dumps(rows, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def _filter(conn: Any, scope_id: str, cutoff: int) -> str:
    return (
        sql.SQL(
            "scope_id={} AND state='active' AND "
            "(metadata->>'session_ordinal')::integer < {}"
        )
        .format(sql.Literal(scope_id), sql.Literal(cutoff))
        .as_string(conn)
    )


def _create_probe_functions(conn: Any, library: Path) -> None:
    library_sql = sql.Literal(str(library.resolve())).as_string(conn)
    statements = (
        "CREATE OR REPLACE FUNCTION public.tjs_open_reach_probe("
        "regclass,integer,integer,integer,integer,text,text,vector,bigint,integer) "
        f"RETURNS SETOF bigint AS {library_sql},'tjs_open_pg' LANGUAGE C VOLATILE",
        "CREATE OR REPLACE FUNCTION public.tjs_open_graph_reached_probe() "
        f"RETURNS bigint AS {library_sql},'tjs_open_graph_reached_pg' LANGUAGE C VOLATILE",
        "CREATE OR REPLACE FUNCTION public.tjs_open_graph_examined_probe() "
        f"RETURNS bigint AS {library_sql},'tjs_open_graph_examined_pg' LANGUAGE C VOLATILE",
        "CREATE OR REPLACE FUNCTION public.tjs_open_graph_censored_probe() "
        f"RETURNS boolean AS {library_sql},'tjs_open_graph_censored_pg' LANGUAGE C VOLATILE",
        "CREATE OR REPLACE FUNCTION public.tjs_open_candidates_examined_probe() "
        f"RETURNS bigint AS {library_sql},'tjs_open_candidates_examined_pg' LANGUAGE C VOLATILE",
    )
    for statement in statements:
        conn.execute(statement)


def _drop_probe_functions(conn: Any) -> None:
    for name, signature in (
        (
            "tjs_open_reach_probe",
            "regclass,integer,integer,integer,integer,text,text,vector,bigint,integer",
        ),
        ("tjs_open_graph_reached_probe", ""),
        ("tjs_open_graph_examined_probe", ""),
        ("tjs_open_graph_censored_probe", ""),
        ("tjs_open_candidates_examined_probe", ""),
    ):
        conn.execute(f"DROP FUNCTION IF EXISTS public.{name}({signature})")


def _fused_probe(
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
    selected: list[int] = []
    started = time.perf_counter()
    with conn.transaction():
        conn.execute("SELECT set_config('hnsw.iterative_scan','relaxed_order',true)")
        conn.execute(
            "SELECT set_config('tjs.graph_work_budget',%s,true)",
            (str(graph_work_budget),),
        )
        with conn.cursor(name=f"ma_nodes_{time.perf_counter_ns()}") as cursor:
            cursor.itersize = 1
            cursor.execute(
                "SELECT tjs_open_reach_probe('gem_unit'::regclass,%s,%s,%s,%s,"
                "'id',%s,%s::vector,NULL,0)",
                (
                    k,
                    term_cond,
                    m_seeds,
                    hops,
                    _filter(conn, scope_id, cutoff),
                    vec_literal(vector),
                ),
            )
            for row in cursor:
                if row[0] is not None:
                    selected.append(int(row[0]))
                if len(selected) >= k:
                    break
        probes = conn.execute(
            "SELECT tjs_open_graph_reached_probe(),"
            "tjs_open_graph_examined_probe(),tjs_open_graph_censored_probe(),"
            "tjs_open_candidates_examined_probe()"
        ).fetchone()
    return {
        "selected_ids": selected,
        "visited_nodes": int(probes[0]),
        "visited_edges": int(probes[1]),
        "graph_censored": bool(probes[2]),
        "candidates_examined": int(probes[3]),
        "time_to_k_ms": (time.perf_counter() - started) * 1000.0,
    }


def _graph_probe(
    conn: Any,
    *,
    anchor: int,
    edge_type: int,
    hops: int,
    graph_work_budget: int,
) -> dict[str, Any]:
    before = int(conn.execute("SELECT graph_store.gph_visits()").fetchone()[0])
    reached = [
        int(row[0])
        for row in conn.execute(
            "SELECT graph_store.gph_traverse_bounded(%s,%s,%s,%s)",
            (anchor, hops, edge_type, graph_work_budget),
        )
        if row[0] is not None
    ]
    after = int(conn.execute("SELECT graph_store.gph_visits()").fetchone()[0])
    return {
        "visited_nodes": len(set(reached) | {anchor}),
        "visited_edges": after - before,
        "graph_censored": bool(
            conn.execute(
                "SELECT graph_store.gph_traverse_bounded_censored()"
            ).fetchone()[0]
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {args.output_dir}")
    if not args.custom_library.is_file():
        raise FileNotFoundError(args.custom_library)
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
        raise RuntimeError(f"source scopes incomplete: {observed_units}!={expected_units}")
    rows = conn.execute(
        "SELECT scope_id,id,(metadata->>'session_ordinal')::integer FROM gem_unit"
        " WHERE scope_id=ANY(%s)",
        (scope_ids,),
    ).fetchall()
    units: dict[str, dict[int, int]] = defaultdict(dict)
    for scope_id, unit_id, ordinal in rows:
        units[str(scope_id)][int(ordinal)] = int(unit_id)
    edge_type = int(
        conn.execute(
            "SELECT id FROM graph_store.edge_type WHERE name='association'"
        ).fetchone()[0]
    )
    digest_before = _scope_digest(conn, scope_ids)
    counts_before = list(
        conn.execute(
            "SELECT count(*),graph_store.gph_vertex_count(),"
            "graph_store.gph_visible_edge_count() FROM gem_unit"
        ).fetchone()
    )
    _create_probe_functions(conn, args.custom_library)
    output_path = args.output_dir / "measurements.jsonl"
    try:
        with output_path.open("x", encoding="utf-8") as output:
            for session, vector in zip(decisions, vectors, strict=True):
                scope_id = scope_by_task[session.task_uid]
                anchor = units[scope_id][session.ordinal - 1]
                fused = _fused_probe(
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
                graph = _graph_probe(
                    conn,
                    anchor=anchor,
                    edge_type=edge_type,
                    hops=args.hops,
                    graph_work_budget=args.graph_work_budget,
                )
                selected_rows = conn.execute(
                    "SELECT count(*) FROM gem_unit WHERE id=ANY(%s) AND scope_id=%s"
                    " AND (metadata->>'session_ordinal')::integer < %s",
                    (fused["selected_ids"], scope_id, session.ordinal),
                ).fetchone()
                if int(selected_rows[0]) != len(fused["selected_ids"]):
                    raise RuntimeError(f"fused cutoff/scope violation: {session.session_uid}")
                output.write(
                    json.dumps(
                        {
                            "task_uid": session.task_uid,
                            "session_uid": session.session_uid,
                            "cutoff_ordinal": session.ordinal,
                            "fused": fused,
                            "graph_relational": graph,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                output.flush()
    finally:
        _drop_probe_functions(conn)
    digest_after = _scope_digest(conn, scope_ids)
    counts_after = list(
        conn.execute(
            "SELECT count(*),graph_store.gph_vertex_count(),"
            "graph_store.gph_visible_edge_count() FROM gem_unit"
        ).fetchone()
    )
    store.close()
    if digest_before != digest_after or counts_before != counts_after:
        raise RuntimeError("graph probe changed frozen source data")
    observed = [json.loads(line) for line in output_path.read_text().splitlines()]

    def values(arm: str, field: str) -> list[float]:
        return [float(row[arm][field]) for row in observed]

    summary = {
        "schema_version": "memoryarena_graph_work_probe_v0.1.0",
        "status": "pass",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "dataset": {
            "config": corpus.config,
            "manifest_sha256": corpus.manifest_sha256,
            "tasks": len(tasks),
            "queries": len(decisions),
        },
        "claim_boundary": (
            "read-only system replay with a workspace-built node probe; frozen "
            "receipts/outcomes are unchanged; no GX10 sign-off"
        ),
        "work": {
            "gem_fused": {
                "visited_nodes": _distribution(values("fused", "visited_nodes")),
                "visited_edges": _distribution(values("fused", "visited_edges")),
                "candidates_examined": _distribution(
                    values("fused", "candidates_examined")
                ),
                "graph_censored": sum(
                    row["fused"]["graph_censored"] for row in observed
                ),
            },
            "graph_relational": {
                "visited_nodes": _distribution(
                    values("graph_relational", "visited_nodes")
                ),
                "visited_edges": _distribution(
                    values("graph_relational", "visited_edges")
                ),
                "graph_censored": sum(
                    row["graph_relational"]["graph_censored"] for row in observed
                ),
            },
        },
        "probe": {
            "definition": "distinct graph-reach vertices including seed vertices",
            "library": str(args.custom_library.resolve()),
            "library_sha256": hashlib.sha256(
                args.custom_library.read_bytes()
            ).hexdigest(),
            "embedding_model": args.embedding_model,
            "embedding_revision": args.embedding_revision,
            "calls": ledger.summary(),
        },
        "read_only_gate": {
            "scope_digest_before": digest_before,
            "scope_digest_after": digest_after,
            "global_counts_before": counts_before,
            "global_counts_after": counts_after,
            "violations": 0,
        },
        "artifacts": {"measurements": output_path.name},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS queries={len(decisions)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
