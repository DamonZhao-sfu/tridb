"""Independently audit and aggregate a fused-vs-staged MemoryArena run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem.store import GemStore
from bench.agent_memory.memoryarena.dataset import load_export


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--source-namespace", default="memoryarena_agent")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            result.append(value)
    return result


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


def _bootstrap(
    deltas: Mapping[str, Sequence[float]], *, iterations: int, seed: int
) -> dict[str, Any]:
    task_ids = sorted(deltas)
    task_means = [statistics.fmean(deltas[task_id]) for task_id in task_ids]
    generator = random.Random(seed)
    samples = [
        statistics.fmean(
            task_means[generator.randrange(len(task_means))] for _ in task_means
        )
        for _ in range(iterations)
    ]
    return {
        "n_tasks": len(task_ids),
        "mean_delta_ms_staged_minus_fused": statistics.fmean(task_means),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "method": "paired_task_cluster_percentile_bootstrap",
        "iterations": iterations,
        "seed": seed,
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
    summary_path = args.run_dir / "summary.json"
    measurements_path = args.run_dir / "measurements.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _rows(measurements_path)
    expected = int(summary["dataset"]["queries"]) * int(
        summary["protocol"]["repetitions_per_query"]
    )
    keys = {(int(row["query_index"]), int(row["repetition"])) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(
            f"measurement completeness failed: rows={len(rows)} unique={len(keys)} "
            f"expected={expected}"
        )

    quality_matches = sum(bool(row["quality_match"]) for row in rows)
    constraint_violations = sum(
        int(row[system]["quality"]["constraint_violations"])
        for row in rows
        for system in ("fused", "staged")
    )
    exact_order = sum(bool(row["exact_selected_order_match"]) for row in rows)
    exact_set = sum(
        set(row["fused"]["selected_ids"]) == set(row["staged"]["selected_ids"])
        for row in rows
    )
    if quality_matches != expected or constraint_violations:
        raise RuntimeError(
            f"quality gate failed: matches={quality_matches}/{expected}, "
            f"constraint_violations={constraint_violations}"
        )

    task_deltas: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        task_deltas[str(row["task_uid"])].append(
            float(row["staged"]["time_to_k_ms"])
            - float(row["fused"]["time_to_k_ms"])
        )
    iterations = int(summary["latency_ms"]["paired"]["iterations"])
    seed = int(summary["latency_ms"]["paired"]["seed"])
    paired = _bootstrap(task_deltas, iterations=iterations, seed=seed)
    if paired != summary["latency_ms"]["paired"]:
        raise RuntimeError("paired latency bootstrap does not reproduce summary")

    corpus = load_export(args.export_dir)
    tasks = corpus.tasks[: int(summary["dataset"]["tasks"])]
    scope_ids = [
        f"{args.source_namespace}:{summary['source_run_id']}:gem_fused:{task.task_uid}"
        for task in tasks
    ]
    store = GemStore.connect(args.dsn, dim=1024)
    try:
        live_digest = _scope_digest(store.conn, scope_ids)
        live_counts = list(
            store.conn.execute(
                "SELECT count(*),graph_store.gph_vertex_count(),"
                " graph_store.gph_visible_edge_count() FROM gem_unit"
            ).fetchone()
        )
    finally:
        store.close()
    read_gate = summary["read_only_gate"]
    if not (
        read_gate["violations"] == 0
        and read_gate["scope_digest_before"] == read_gate["scope_digest_after"]
        and read_gate["global_counts_before"] == read_gate["global_counts_after"]
        and live_digest == read_gate["scope_digest_after"]
        and live_counts == read_gate["global_counts_after"]
    ):
        raise RuntimeError("read-only/current-snapshot gate failed")

    def values(system: str, field: str) -> list[float]:
        return [
            float(row[system][field])
            for row in rows
            if row[system].get(field) is not None
        ]

    audit = {
        "schema_version": "memoryarena_fused_vs_staged_audit_v0.1.0",
        "status": "pass",
        "run_id": summary["run_id"],
        "claim_boundary": summary["claim_boundary"],
        "gates": {
            "expected_measurements": expected,
            "observed_measurements": len(rows),
            "unique_query_repetitions": len(keys),
            "matched_quality": quality_matches,
            "constraint_violations": constraint_violations,
            "read_only_violations": 0,
            "live_scope_digest_match": True,
            "live_global_counts_match": True,
        },
        "quality_diagnostics": {
            "exact_selected_order_matches": exact_order,
            "exact_selected_order_fraction": exact_order / expected,
            "exact_selected_set_matches": exact_set,
            "exact_selected_set_fraction": exact_set / expected,
            "annotation": summary["quality_gate"]["annotation"],
        },
        "latency_ms": {
            "fused_time_to_first_row": _distribution(
                values("fused", "first_row_ms")
            ),
            "fused_time_to_k": _distribution(values("fused", "time_to_k_ms")),
            "staged_time_to_first_row": _distribution(
                values("staged", "first_row_ms")
            ),
            "staged_time_to_k": _distribution(values("staged", "time_to_k_ms")),
            "paired": paired,
            "staged_vector_stage": _distribution(
                values("staged", "vector_stage_ms")
            ),
            "staged_graph_stage": _distribution(values("staged", "graph_stage_ms")),
            "staged_final_filter_rank": _distribution(
                values("staged", "final_filter_rank_ms")
            ),
        },
        "work": {
            "fused_candidates_examined": _distribution(
                values("fused", "candidates_examined")
            ),
            "fused_visited_edges": _distribution(values("fused", "visited_edges")),
            "fused_visited_nodes": None,
            "fused_visited_nodes_note": (
                "N/A: installed tjs_pg 0.2.0 exposes edge steps but no distinct-node probe"
            ),
            "fused_bridges_injected": _distribution(
                values("fused", "bridges_injected")
            ),
            "fused_termination_reasons": dict(
                Counter(str(row["fused"]["termination_reason"]) for row in rows)
            ),
            "fused_graph_censored": sum(
                bool(row["fused"]["graph_censored"]) for row in rows
            ),
            "staged_vector_window_candidates": _distribution(
                values("staged", "vector_window_candidates")
            ),
            "staged_bounded_union_candidates": _distribution(
                values("staged", "bounded_union_candidates")
            ),
            "staged_visited_edges": _distribution(values("staged", "visited_edges")),
            "staged_visited_nodes": None,
            "staged_visited_nodes_note": (
                "N/A: measurements record edge steps and bounded union, not distinct "
                "graph-reached vertices"
            ),
            "staged_graph_censored": sum(
                bool(row["staged"]["graph_censored"]) for row in rows
            ),
        },
        "artifacts": {
            "summary": {"path": "summary.json", "sha256": _sha256(summary_path)},
            "measurements": {
                "path": "measurements.jsonl",
                "sha256": _sha256(measurements_path),
            },
        },
    }
    output = args.output or args.run_dir / "audit.json"
    output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PASS measurements={expected} quality={quality_matches} readonly=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
