#!/usr/bin/env python3
"""Create five-system Track C aggregates, figures, receipts, and final report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.table5_track_c.aggregate import (
    aggregate,
    aggregate_answer_quality,
    aggregate_decomposition,
    aggregate_resources,
)
from bench.agent_memory.table5_track_c.figures import (
    plot_add_ecdf,
    plot_add_p99,
    plot_latency_quality_frontier,
    plot_search_box,
    plot_search_ecdf,
    plot_search_p99,
    plot_success_rate,
)
from bench.agent_memory.table5_track_c.protocol import benchmark_code_sha256
from tools.table5_track_c_goal_audit import REPOSITORY, audit


DEFAULT_SCHEDULE = (
    REPOSITORY
    / "bench/agent_memory/table5_track_c/manifests/completion_v2_schedule.json"
)
LABELS = {
    "tridb_gem": "TriDB/GEM",
    "mem0": "Mem0 2.0.18",
    "memos": "MemOS 2.0.30",
    "cognee": "Cognee 1.5.0",
    "graphiti_zep_oss_proxy": "Graphiti (Zep OSS proxy)",
}
SPECIAL_LABELS = {"all": "All systems"}
POSTPROCESS_COVERAGE_CHECK_NAMES = (
    "metric_rows",
    "formal_cardinality",
    "retrieval_quality_rows",
    "decomposition_rows",
    "resource_rows",
    "time_breakdown_rows",
    "observable_call_rows",
    "operational_rows",
    "visibility_rows",
    "answer_quality_rows",
    "noninferiority_rows",
    "paper_reference_rows",
    "source_provenance_rows",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rows_with_display_names(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach unambiguous user-facing names to every exported system id."""

    labeled: list[dict[str, Any]] = []
    for row in rows:
        output: dict[str, Any] = {}
        for key, value in row.items():
            output[key] = value
            if key not in {"system", "candidate", "reference"}:
                continue
            identifier = str(value)
            output[f"{key}_display_name"] = LABELS.get(
                identifier, SPECIAL_LABELS.get(identifier, identifier)
            )
        labeled.append(output)
    return labeled


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty final CSV: {path.name}")
    labeled = _rows_with_display_names(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labeled[0]))
        writer.writeheader()
        writer.writerows(labeled)


def _raw_artifact_specs(
    *,
    root: Path,
    protocol: dict[str, Any],
    schedule_path: Path,
    protocol_path: Path,
) -> list[tuple[str, Path]]:
    """Return the exact pre-analysis evidence set for the frozen matrix."""

    specs: list[tuple[str, Path]] = []
    for system in protocol["systems"]:
        for workload in protocol["workloads"]:
            for round_name in protocol["rounds"]:
                run = root / "runs" / system / workload / round_name
                specs.extend(
                    (role, run / filename)
                    for role, filename in (
                        ("formal_request_jsonl", "formal.jsonl"),
                        ("warmup_request_jsonl", "warmup.jsonl"),
                        ("late_outcome_jsonl", "late_outcomes.jsonl"),
                        ("stage_span_jsonl", "spans.jsonl"),
                        ("run_receipt", "run_receipt.json"),
                    )
                )
                specs.append(
                    (
                        "formal_driver_log",
                        root / "logs" / f"{workload}_{round_name}_{system}.log",
                    )
                )
        for round_name in protocol["rounds"]:
            quality = root / "quality" / system / round_name
            specs.extend(
                (role, quality / filename)
                for role, filename in (
                    ("quality_answers", "answers.jsonl"),
                    ("quality_judges", "judges.jsonl"),
                    ("quality_summary", "quality_summary.json"),
                    ("quality_receipt", "quality_receipt.json"),
                )
            )
            specs.append(
                (
                    "quality_driver_log",
                    root / "logs" / f"quality_{round_name}_{system}.log",
                )
            )
        specs.append(("conformance_receipt", root / "conformance" / f"{system}.json"))
        specs.append(
            (
                "source_receipt",
                REPOSITORY / str(protocol["source_receipts"][system]),
            )
        )
    specs.extend(
        [
            ("hardware_receipt", root / "hardware_receipt.json"),
            ("model_receipt", root / "model_receipt.json"),
            ("tridb_invariant_receipt", root / "tridb_invariant_receipt.json"),
            ("model_service_log", root / "logs/answer_model.log"),
            ("model_service_log", root / "logs/embedding_model.log"),
            ("schedule", schedule_path),
            ("protocol", protocol_path),
            ("canonical_plan", REPOSITORY / str(protocol["plan"])),
            ("dataset", Path(str(protocol["dataset"]["path"]))),
        ]
    )
    return sorted(specs, key=lambda item: (str(item[1].resolve()), item[0]))


def _write_raw_artifact_manifest(
    *,
    root: Path,
    protocol: dict[str, Any],
    schedule_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    specs = _raw_artifact_specs(
        root=root,
        protocol=protocol,
        schedule_path=schedule_path,
        protocol_path=protocol_path,
    )
    missing = [str(path.resolve()) for _, path in specs if not path.is_file()]
    if missing:
        raise RuntimeError(f"raw artifact manifest is missing inputs: {missing}")
    entries = [
        {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for role, path in specs
    ]
    counts = Counter(entry["role"] for entry in entries)
    canonical = json.dumps(
        entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = {
        "schema_version": "table5_track_c_raw_artifact_manifest_v0.1.0",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_root": str(root.resolve()),
        "schedule_sha256": _sha256(schedule_path),
        "protocol_sha256": _sha256(protocol_path),
        "artifact_count": len(entries),
        "by_role": dict(sorted(counts.items())),
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
        "artifacts": entries,
    }
    _write_json(root / "raw_artifact_manifest.json", manifest)
    return manifest


def _time_breakdown(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("**/run_receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        breakdown = receipt.get("time_breakdown") or {}
        for stage, values in sorted((breakdown.get("stages") or {}).items()):
            latency = values.get("per_request_union_ms") or {}
            rows.append(
                {
                    "system": receipt["system"],
                    "phase": receipt["phase"],
                    "build_id": receipt["build_id"],
                    "stage": stage,
                    "request_count": values.get("request_count"),
                    "raw_request_count": values.get("raw_request_count"),
                    "span_count": values.get("span_count"),
                    "successful_spans": values.get("successful_spans"),
                    "stage_work_ms": values.get("stage_work_ms"),
                    "client_visible_stage_work_ms": values.get(
                        "client_visible_stage_work_ms"
                    ),
                    "raw_stage_work_ms": values.get("raw_stage_work_ms"),
                    "post_request_work_ms": values.get("post_request_work_ms"),
                    "per_request_mean_ms": latency.get("mean"),
                    "per_request_p50_ms": latency.get("p50"),
                    "per_request_p90_ms": latency.get("p90"),
                    "per_request_p99_ms": latency.get("p99"),
                }
            )
        unattributed = breakdown.get("unattributed_per_request_ms") or {}
        rows.append(
            {
                "system": receipt["system"],
                "phase": receipt["phase"],
                "build_id": receipt["build_id"],
                "stage": "unattributed",
                "request_count": breakdown.get("formal_requests"),
                "raw_request_count": None,
                "span_count": None,
                "successful_spans": None,
                "stage_work_ms": None,
                "client_visible_stage_work_ms": None,
                "raw_stage_work_ms": None,
                "post_request_work_ms": breakdown.get("post_request_stage_work_ms"),
                "per_request_mean_ms": unattributed.get("mean"),
                "per_request_p50_ms": unattributed.get("p50"),
                "per_request_p90_ms": unattributed.get("p90"),
                "per_request_p99_ms": unattributed.get("p99"),
            }
        )
    return rows


def _observable_call_rows(runs_root: Path) -> list[dict[str, Any]]:
    """Aggregate honest client-boundary counts without inventing internals."""
    rows: list[dict[str, Any]] = []
    for receipt_path in sorted(runs_root.glob("**/run_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        records = []
        with (receipt_path.parent / "formal.jsonl").open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
        observed = [
            record.get("observable_call_counts")
            for record in records
            if isinstance(record.get("observable_call_counts"), dict)
        ]
        count = len(observed)
        totals = {
            kind: sum(int(value.get(kind) or 0) for value in observed)
            for kind in (
                "http_client",
                "database_client",
                "opaque_native_api",
                "total",
            )
        }
        rows.append(
            {
                "system": receipt["system"],
                "phase": receipt["phase"],
                "build_id": receipt["build_id"],
                "formal_requests": len(records),
                "requests_with_call_observation": count,
                "observation_coverage": count / len(records) if records else None,
                "http_client_boundaries_total": totals["http_client"],
                "http_client_boundaries_mean_per_request": (
                    totals["http_client"] / count if count else None
                ),
                "database_client_boundaries_total": totals["database_client"],
                "database_client_boundaries_mean_per_request": (
                    totals["database_client"] / count if count else None
                ),
                "opaque_native_api_boundaries_total": totals["opaque_native_api"],
                "opaque_native_api_boundaries_mean_per_request": (
                    totals["opaque_native_api"] / count if count else None
                ),
                "all_observed_boundaries_total": totals["total"],
                "all_observed_boundaries_mean_per_request": (
                    totals["total"] / count if count else None
                ),
                "partial_timeout_observations": sum(
                    value.get("scope") == "partial_client_boundaries_before_timeout"
                    for value in observed
                ),
                "exact_internal_http_round_trips": None,
                "exact_internal_database_round_trips": None,
                "interpretation": (
                    "client-boundary counts only; compound native internals are "
                    "unavailable and are not inferred"
                ),
            }
        )
    return rows


def _failure_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receipt_path in sorted(runs_root.glob("**/run_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        counts: Counter[tuple[str, bool]] = Counter()
        with (receipt_path.parent / "formal.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("success") is True:
                    continue
                counts[
                    (str(record.get("error") or "unknown"), bool(record.get("timeout")))
                ] += 1
        for (error, timed_out), count in sorted(counts.items()):
            rows.append(
                {
                    "system": receipt["system"],
                    "phase": receipt["phase"],
                    "build_id": receipt["build_id"],
                    "timeout": timed_out,
                    "error": error,
                    "count": count,
                }
            )
    if not rows:
        rows.append(
            {
                "system": "all",
                "phase": "all",
                "build_id": "pooled",
                "timeout": False,
                "error": "no formal request failures observed",
                "count": 0,
            }
        )
    return rows


def _operational_rows(metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "system": row["system"],
            "phase": row["phase"],
            "target_qps": 10.0,
            "scheduled_qps": row.get("scheduled_qps"),
            "admission_qps": row.get("admission_qps"),
            "completion_qps": row.get("actual_qps"),
            "admission_lag_mean_ms": row.get("admission_lag_mean_ms"),
            "admission_lag_p99_ms": row.get("admission_lag_p99_ms"),
            "queue_p99_ms": row.get("queue_p99_ms"),
            "formal_admissions": row.get("formal_requests"),
            "successful": row.get("successful"),
            "failed": row.get("failed"),
            "success_rate": row.get("success_rate"),
            "error_classes": row.get("error_classes"),
        }
        for row in metrics
        if row.get("build_id") == "pooled"
    ]


def _visibility_rows(decomposition: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "system",
        "phase",
        "build_id",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "commit_observed_count",
        "failed_after_commit_count",
        "visibility_requested_count",
        "visibility_probe_started_count",
        "visibility_probe_count",
        "visibility_probe_pass_rate",
        "visibility_probe_false_count",
        "timeout_after_commit_count",
        "timeout_during_visibility_probe_count",
        "visibility_failures",
        "client_timeout_count",
        "late_worker_outcome_count",
        "late_worker_final_success_count",
        "late_worker_final_failed_count",
        "late_worker_commit_observed_count",
        "late_worker_visibility_pass_count",
        "post_timeout_work_mean_ms",
        "post_timeout_work_max_ms",
    )
    return [
        {field: row.get(field) for field in fields}
        for row in decomposition
        if row.get("build_id") == "pooled"
        and row.get("phase") == "add:source_to_searchable"
    ]


def _paper_reference_rows(
    protocol: dict[str, Any], metrics: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    pooled = {
        (str(row["system"]), str(row["phase"])): row
        for row in metrics
        if row.get("build_id") == "pooled"
    }
    rows: list[dict[str, Any]] = []
    reference = (protocol.get("paper_reference") or {}).get("values_ms") or {}
    boundary = (protocol.get("paper_reference") or {}).get("comparison_boundary")
    for system, phases in reference.items():
        for phase, values in phases.items():
            local = pooled.get((str(system), str(phase)))
            if local is None:
                raise RuntimeError(
                    f"paper-reference comparison lacks pooled row: {system} {phase}"
                )
            row: dict[str, Any] = {
                "system": system,
                "phase": phase,
                "comparison_boundary": boundary,
            }
            for metric in ("mean", "p90", "p99"):
                paper_value = float(values[metric])
                local_value = local.get(f"service_{metric}_ms")
                row[f"paper_{metric}_ms"] = paper_value
                row[f"local_{metric}_ms"] = local_value
                row[f"relative_delta_{metric}"] = (
                    (float(local_value) - paper_value) / paper_value
                    if local_value is not None
                    else None
                )
            rows.append(row)
    return rows


def _source_provenance_rows(audit_result: dict[str, Any]) -> list[dict[str, Any]]:
    receipts = audit_result.get("source_receipts") or {}
    if set(receipts) != set(LABELS):
        raise RuntimeError("source provenance does not cover the five systems")
    rows: list[dict[str, Any]] = []
    for system in LABELS:
        receipt = receipts[system]
        runtime = receipt.get("runtime") or {}
        checks = runtime.get("checks") or {}
        observed = runtime.get("observed") or {}
        rows.append(
            {
                "system": system,
                "display_name": LABELS[system],
                "valid": receipt.get("valid"),
                "receipt_status": receipt.get("status"),
                "receipt_path": receipt.get("path"),
                "receipt_sha256": receipt.get("sha256"),
                "expected_receipt_sha256": receipt.get("expected_sha256"),
                "receipt_sha256_matches": receipt.get("sha256_matches"),
                "receipt_identity_matches": receipt.get("identity_matches"),
                "live_runtime_valid": runtime.get("valid"),
                "live_checks_passed": sum(value is True for value in checks.values()),
                "live_checks_total": len(checks),
                "failed_live_checks": json.dumps(
                    [name for name, value in checks.items() if value is not True]
                ),
                "source_head": observed.get("head"),
                "tracked_dirty_entries": observed.get("tracked_dirty_entries"),
                "package_version": observed.get("package_version")
                or observed.get("graphiti_core_version"),
                "postgres_version": observed.get("postgres_version"),
                "neo4j_version": observed.get("neo4j_version"),
                "dependency_manifest_sha256": observed.get(
                    "dependency_manifest_sha256"
                ),
                "environment_freeze_sha256": observed.get("environment_freeze_sha256"),
                "errors": json.dumps(runtime.get("errors") or []),
            }
        )
    if not all(row["valid"] is True for row in rows):
        raise RuntimeError("refusing report with invalid source provenance")
    return rows


def _pooled(rows: Sequence[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    by_system = {
        str(row["system"]): row
        for row in rows
        if row["phase"] == phase and row["build_id"] == "pooled"
    }
    return [by_system[system] for system in LABELS if system in by_system]


def _final_coverage_checks(
    protocol: dict[str, Any],
    *,
    metrics: Sequence[dict[str, Any]],
    retrieval_quality: Sequence[dict[str, Any]],
    decomposition: Sequence[dict[str, Any]],
    resources: Sequence[dict[str, Any]],
    time_breakdown: Sequence[dict[str, Any]],
    observable_calls: Sequence[dict[str, Any]],
    operational: Sequence[dict[str, Any]],
    visibility: Sequence[dict[str, Any]],
    answer_quality: Sequence[dict[str, Any]],
    noninferiority: Sequence[dict[str, Any]],
    paper_reference: Sequence[dict[str, Any]],
    source_provenance: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    """Require exact final-table coverage, not merely non-empty outputs."""
    systems = tuple(str(value) for value in protocol.get("systems") or ())
    workloads = tuple(str(value) for value in protocol.get("workloads") or ())
    rounds = tuple(str(value) for value in protocol.get("rounds") or ())
    phase_for = {
        "search": "search",
        "add_native": "add:native",
        "add_source_to_searchable": "add:source_to_searchable",
    }
    expected_build = {
        (system, phase_for[workload], f"tc5v2_{system}_{workload}_{round_name}")
        for system in systems
        for workload in workloads
        for round_name in rounds
        if workload in phase_for
    }
    expected_pooled = {
        (system, phase_for[workload], "pooled")
        for system in systems
        for workload in workloads
        if workload in phase_for
    }
    expected_metric = expected_build | expected_pooled
    expected_search_build = {
        (system, f"tc5v2_{system}_search_{round_name}")
        for system in systems
        for round_name in rounds
    }
    expected_search_quality = expected_search_build | {
        (system, "pooled") for system in systems
    }

    def run_keys(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str, str]]:
        return [
            (str(row.get("system")), str(row.get("phase")), str(row.get("build_id")))
            for row in rows
        ]

    metric_keys = run_keys(metrics)
    decomposition_keys = run_keys(decomposition)
    resource_keys = run_keys(resources)
    call_keys = run_keys(observable_calls)
    time_keys = set(run_keys(time_breakdown))
    unattributed_keys = {
        key
        for key, row in zip(run_keys(time_breakdown), time_breakdown, strict=True)
        if row.get("stage") == "unattributed"
    }
    operational_keys = [
        (str(row.get("system")), str(row.get("phase"))) for row in operational
    ]
    visibility_keys = run_keys(visibility)
    retrieval_keys = [
        (str(row.get("system")), str(row.get("build_id"))) for row in retrieval_quality
    ]
    answer_overall = [
        (str(row.get("system")), str(row.get("build_id")))
        for row in answer_quality
        if row.get("category") == "overall"
    ]
    expected_formal = {
        "search": int((protocol.get("protocol") or {}).get("search_formal") or 0),
        "add:native": int((protocol.get("protocol") or {}).get("add_formal") or 0),
        "add:source_to_searchable": int(
            (protocol.get("protocol") or {}).get("add_formal") or 0
        ),
    }
    formal_cardinality = all(
        row.get("formal_requests")
        == expected_formal.get(str(row.get("phase")), 0)
        * (len(rounds) if row.get("build_id") == "pooled" else 1)
        for row in metrics
    )
    expected_references = {
        ("tridb_gem", system) for system in systems if system != "tridb_gem"
    }
    observed_references = [
        (str(row.get("candidate")), str(row.get("reference"))) for row in noninferiority
    ]
    quality_policy = protocol.get("quality") or {}
    expected_paper = {
        (str(system), str(phase))
        for system, phases in (
            (protocol.get("paper_reference") or {}).get("values_ms") or {}
        ).items()
        for phase in phases
    }
    observed_paper = [
        (str(row.get("system")), str(row.get("phase"))) for row in paper_reference
    ]
    checks = {
        "metric_rows": len(metric_keys) == len(expected_metric)
        and set(metric_keys) == expected_metric,
        "formal_cardinality": formal_cardinality,
        "retrieval_quality_rows": len(retrieval_keys) == len(expected_search_quality)
        and set(retrieval_keys) == expected_search_quality,
        "decomposition_rows": len(decomposition_keys) == len(expected_metric)
        and set(decomposition_keys) == expected_metric,
        "resource_rows": len(resource_keys) == len(expected_build)
        and set(resource_keys) == expected_build,
        "time_breakdown_rows": time_keys == expected_build
        and unattributed_keys == expected_build
        and all(
            row.get("stage") == "unattributed"
            or (
                all(
                    isinstance(row.get(field), (int, float))
                    for field in (
                        "stage_work_ms",
                        "client_visible_stage_work_ms",
                        "raw_stage_work_ms",
                        "post_request_work_ms",
                    )
                )
                and float(row["stage_work_ms"])
                == float(row["client_visible_stage_work_ms"])
                and float(row["raw_stage_work_ms"])
                >= float(row["client_visible_stage_work_ms"])
                and abs(
                    float(row["post_request_work_ms"])
                    - (
                        float(row["raw_stage_work_ms"])
                        - float(row["client_visible_stage_work_ms"])
                    )
                )
                <= 1e-6
            )
            for row in time_breakdown
        ),
        "observable_call_rows": len(call_keys) == len(expected_build)
        and set(call_keys) == expected_build
        and all(row.get("observation_coverage") == 1.0 for row in observable_calls),
        "operational_rows": len(operational_keys) == len(expected_pooled)
        and set(operational_keys)
        == {(system, phase) for system, phase, _ in expected_pooled}
        and all(
            all(
                isinstance(row.get(field), (int, float))
                for field in (
                    "target_qps",
                    "scheduled_qps",
                    "admission_qps",
                    "admission_lag_p99_ms",
                )
            )
            and float(row["target_qps"]) == 10.0
            and float(row["scheduled_qps"]) == 10.0
            and abs(float(row["admission_qps"]) - 10.0) / 10.0 <= 0.01
            and float(row["admission_lag_p99_ms"]) <= 100.0
            for row in operational
        ),
        "visibility_rows": len(visibility_keys) == len(systems)
        and set(visibility_keys)
        == {(system, "add:source_to_searchable", "pooled") for system in systems},
        "answer_quality_rows": len(answer_overall) == len(expected_search_quality)
        and set(answer_overall) == expected_search_quality,
        "noninferiority_rows": len(observed_references) == len(expected_references)
        and set(observed_references) == expected_references
        and all(
            row.get("method") == quality_policy.get("non_inferiority_method")
            and row.get("pairing_keys")
            == ",".join(quality_policy.get("pairing_keys") or ())
            and row.get("cluster") == quality_policy.get("bootstrap_cluster")
            and row.get("matched_records") == expected_formal["search"] * len(rounds)
            and row.get("clusters")
            == (protocol.get("dataset") or {}).get("conversations")
            and row.get("bootstrap_iterations")
            == quality_policy.get("bootstrap_iterations")
            and row.get("bootstrap_seed") == quality_policy.get("bootstrap_seed")
            and row.get("one_sided_confidence_level")
            == quality_policy.get("one_sided_confidence_level")
            and row.get("margin")
            == quality_policy.get("non_inferiority_margin_percentage_points") / 100
            and isinstance(row.get("one_sided_lower_bound"), (int, float))
            and isinstance(row.get("passes_noninferiority"), bool)
            for row in noninferiority
        ),
        "paper_reference_rows": len(observed_paper) == len(expected_paper)
        and set(observed_paper) == expected_paper,
        "source_provenance_rows": len(source_provenance) == len(systems)
        and {str(row.get("system")) for row in source_provenance} == set(systems),
    }
    if set(checks) != set(POSTPROCESS_COVERAGE_CHECK_NAMES):
        raise RuntimeError("postprocess coverage check schema drifted")
    return checks


def _number(value: Any, *, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _rate(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def _estimate_ci(row: dict[str, Any], stem: str) -> str:
    estimate = row.get(f"{stem}_ms")
    low = row.get(f"{stem}_ci_low_ms")
    high = row.get(f"{stem}_ci_high_ms")
    if estimate is None:
        return "—"
    if low is None or high is None:
        return _number(estimate)
    return f"{_number(estimate)} [{_number(low)}, {_number(high)}]"


def _service_table(rows: Sequence[dict[str, Any]], phase: str) -> str:
    rendered = [
        "| System | mean ms [95% CI] | P90 ms [95% CI] | P99 ms [95% CI] | Success | Completion QPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in _pooled(rows, phase):
        rendered.append(
            "| {label} | {mean} | {p90} | {p99} | {success} | {qps} |".format(
                label=LABELS[str(row["system"])],
                mean=_estimate_ci(row, "service_mean"),
                p90=_estimate_ci(row, "service_p90"),
                p99=_estimate_ci(row, "service_p99"),
                success=_rate(row.get("success_rate")),
                qps=_number(row.get("actual_qps")),
            )
        )
    return "\n".join(rendered)


def _user_visible_table(rows: Sequence[dict[str, Any]], phase: str) -> str:
    rendered = [
        "| System | All-admission mean ms [95% CI] | P90 ms [95% CI] | P99 ms [95% CI] | Admissions | Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in _pooled(rows, phase):
        rendered.append(
            "| {label} | {mean} | {p90} | {p99} | {total} | {failed} |".format(
                label=LABELS[str(row["system"])],
                mean=_estimate_ci(row, "user_visible_mean"),
                p90=_estimate_ci(row, "user_visible_p90"),
                p99=_estimate_ci(row, "user_visible_p99"),
                total=row.get("formal_requests", "—"),
                failed=row.get("failed", "—"),
            )
        )
    return "\n".join(rendered)


def _quality_table(rows: Sequence[dict[str, Any]]) -> str:
    pooled = {
        str(row["system"]): row
        for row in rows
        if row.get("build_id") == "pooled" and row.get("category") == "overall"
    }
    rendered = [
        "| System | Answer score | Judge coverage | Mean context tokens | Token-count coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, label in LABELS.items():
        row = pooled.get(system)
        if row is None:
            continue
        mean_tokens = row.get("mean_context_tokens")
        rendered.append(
            "| {label} | {score:.2%} | {coverage:.2%} | {tokens} | "
            "{token_coverage:.2%} |".format(
                label=label,
                score=float(row["score"]),
                coverage=float(row["coverage"]),
                tokens=(
                    f"{float(mean_tokens):.1f}" if mean_tokens is not None else "—"
                ),
                token_coverage=float(row.get("context_token_coverage") or 0.0),
            )
        )
    return "\n".join(rendered)


def _operational_table(rows: Sequence[dict[str, Any]]) -> str:
    rendered = [
        "| System | Workload | Target QPS | Scheduled QPS | Admission QPS | Completion QPS | Admission lag P99 ms | Queue P99 ms | Success | Failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    phase_labels = {
        "search": "Search",
        "add:native": "Native Add",
        "add:source_to_searchable": "Source-to-searchable Add",
    }
    for row in _operational_rows(rows):
        rendered.append(
            "| {label} | {phase} | 10.000 | {scheduled} | {admission} | {completion} | {lag} | {queue} | {success} | {failed} |".format(
                label=LABELS[str(row["system"])],
                phase=phase_labels.get(str(row["phase"]), str(row["phase"])),
                scheduled=_number(row.get("scheduled_qps")),
                admission=_number(row.get("admission_qps")),
                completion=_number(row.get("completion_qps")),
                lag=_number(row.get("admission_lag_p99_ms")),
                queue=_number(row.get("queue_p99_ms")),
                success=_rate(row.get("success_rate")),
                failed=row.get("failed", "—"),
            )
        )
    return "\n".join(rendered)


def _visibility_table(rows: Sequence[dict[str, Any]]) -> str:
    rendered = [
        "| System | Admissions | Commit observed | Probe started | Probe pass | False probes | Failed after commit | Timeout in probe | Late final success | Late final failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _visibility_rows(rows):
        rendered.append(
            "| {label} | {total} | {committed} | {started} | {rate} | {false} | {failed} | {timeout} | {late_success} | {late_failed} |".format(
                label=LABELS[str(row["system"])],
                total=row.get("total_requests", "—"),
                committed=row.get("commit_observed_count", "—"),
                started=row.get("visibility_probe_started_count", "—"),
                rate=_rate(row.get("visibility_probe_pass_rate")),
                false=row.get("visibility_probe_false_count", "—"),
                failed=row.get("failed_after_commit_count", "—"),
                timeout=row.get("timeout_during_visibility_probe_count", "—"),
                late_success=row.get("late_worker_final_success_count", "—"),
                late_failed=row.get("late_worker_final_failed_count", "—"),
            )
        )
    return "\n".join(rendered)


def _paper_reference_table(
    protocol: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> str:
    rendered = [
        "| System | Workload | Paper mean / local / delta | Paper P90 / local / delta | Paper P99 / local / delta |",
        "|---|---|---:|---:|---:|",
    ]
    phase_labels = {"search": "Search", "add:native": "Native Add"}

    def cell(row: dict[str, Any], metric: str) -> str:
        return "{paper:.1f} / {local} / {delta}".format(
            paper=float(row[f"paper_{metric}_ms"]),
            local=_number(row.get(f"local_{metric}_ms")),
            delta=(
                "—"
                if row.get(f"relative_delta_{metric}") is None
                else f"{float(row[f'relative_delta_{metric}']):+.2%}"
            ),
        )

    for row in rows:
        rendered.append(
            "| {label} | {phase} | {mean} | {p90} | {p99} |".format(
                label=LABELS[str(row["system"])],
                phase=phase_labels[str(row["phase"])],
                mean=cell(row, "mean"),
                p90=cell(row, "p90"),
                p99=cell(row, "p99"),
            )
        )
    return "\n".join(rendered)


def _noninferiority_table(rows: Sequence[dict[str, Any]]) -> str:
    rendered = [
        "| Reference | TriDB score | Reference score | Observed delta | One-sided 95% lower bound | Margin | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rendered.append(
            "| {reference} | {candidate} | {baseline} | {delta} | {lower} | {margin} | {gate} |".format(
                reference=LABELS[str(row["reference"])],
                candidate=_rate(row.get("candidate_score")),
                baseline=_rate(row.get("reference_score")),
                delta=_rate(row.get("delta")),
                lower=_rate(row.get("one_sided_lower_bound")),
                margin=_rate(row.get("margin")),
                gate="PASS" if row.get("passes_noninferiority") else "FAIL",
            )
        )
    return "\n".join(rendered)


def _source_provenance_table(rows: Sequence[dict[str, Any]]) -> str:
    rendered = [
        "| System | Frozen receipt | Live identity | Source commit | Runtime version |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        runtime = (
            row.get("package_version")
            or row.get("postgres_version")
            or row.get("neo4j_version")
            or "—"
        )
        rendered.append(
            "| {label} | {receipt} | {live} ({passed}/{total}) | {head} | {runtime} |".format(
                label=row["display_name"],
                receipt="PASS" if row.get("receipt_sha256_matches") else "FAIL",
                live="PASS" if row.get("live_runtime_valid") else "FAIL",
                passed=row.get("live_checks_passed", "—"),
                total=row.get("live_checks_total", "—"),
                head=row.get("source_head") or "—",
                runtime=runtime,
            )
        )
    return "\n".join(rendered)


def _report(
    protocol: dict[str, Any],
    metrics: Sequence[dict[str, Any]],
    answer_quality: Sequence[dict[str, Any]],
    noninferiority: Sequence[dict[str, Any]],
    decomposition: Sequence[dict[str, Any]],
    paper_reference: Sequence[dict[str, Any]],
    source_provenance: Sequence[dict[str, Any]],
    audit_result: dict[str, Any],
    raw_manifest: dict[str, Any],
) -> str:
    return f"""# Track C five-system controlled-model comparison

## Material Passport

- Dataset: LoCoMo, SHA-256 `{protocol["dataset"]["sha256"]}`
- Frozen plan: `{protocol["plan"]}`, SHA-256 `{audit_result["plan_sha256"]}`
- Formal coverage: {audit_result["complete_valid_runs"]}/45 runs; three independent fresh states per system/workload
- Search: 1,787 formal requests/build; Add: 2,000 formal admissions/build
- Load: 10 QPS absolute no-drift open loop; 60 s timeout; no harness retry
- Statistics: linear-interpolation percentiles; 2,000-replicate 95% bootstrap CI clustered by LoCoMo conversation across fresh builds
- Models: Qwen3-32B-FP8 revision `{protocol["models"]["answer_revision"]}` (thinking off) + Qwen3-Embedding-0.6B revision `{protocol["models"]["embedding_revision"]}` (1024-d), served from frozen offline cache with vLLM `{protocol["models"]["runtime_versions"]["vllm"]}`
- Hardware claim: same-host controlled comparison; not an H800 numerical reproduction and not a GX10 sign-off
- Zep boundary: `Graphiti (Zep OSS proxy)` is not production Zep and is not a reproduced Zep Table 5 row
- Proxy decision provenance: frozen protocol `{protocol["zep_naming"]["decision"]}`; result label `{protocol["zep_naming"]["result_label"]}`
- Raw evidence manifest: {raw_manifest["artifact_count"]} artifacts; combined SHA-256 `{raw_manifest["combined_sha256"]}`

## Protocol deviations and resource-control boundary

This run serializes formal system/workload points and binds both controlled-model services to GPU0. Before model launch, the hashed driver requires physical GPU0 to have zero compute processes and refuses rather than terminating an external process. This is a prelaunch contamination gate, not a claim that NVIDIA exclusive mode is enabled. Before creating a point, the hashed launcher refuses any pre-existing per-build PostgreSQL database, filesystem/Neo4j state, or dataset namespace; each run receipt records the exact fresh-state identities and the independent auditor re-derives them from system/workload/round. It records the GPU launch gate, runner's effective CPU affinity, cgroup cpuset, cgroup memory limit, and model-service environments in `hardware_receipt.json`. It does **not** enforce or claim a benchmark-specific CPU affinity, RAM cap, or GPU exclusive mode across shared PostgreSQL and separately managed backends, so pre-registration fairness rule 10.5 is only partially satisfied. Results remain a same-host controlled comparison, not an H800 reproduction or GX10 sign-off.

## Frozen source and live runtime provenance

{_source_provenance_table(source_provenance)}

Every immutable install/source receipt is rechecked against the live checkout, dependency environment, package/backend version, and—where applicable—PostgreSQL extension binaries before freeze, before each formal point, and during final audit. Full hashes and failed-check fields are retained in `aggregates/source_provenance.csv`.

## TriDB architecture invariant

The machine-verifiable `tridb_invariant_receipt.json` binds the frozen source hashes, loaded PostgreSQL extension identities, and fresh conformance probes. Its gate status is `{audit_result["tridb_invariant_gate"]["valid"]}`. A passing gate means the measured TriDB/GEM path used FUSED/TOPIC `tjs_open`, the native adjacency store, PostgreSQL GenericXLog, and bounded Open/Next/Close traversal with disclosed early-termination work. It also records that `gem_edge` is metadata rather than topology. This evidence applies to the stock-PostgreSQL same-host run only; it is not an H800 numerical reproduction or a GX10 build/sign-off.

## Search

### Table 5 headline: successful service completions

{_service_table(metrics, "search")}

### All-admission user-visible latency

{_user_visible_table(metrics, "search")}

Search latency excludes final answer generation. The Table 5 headline conditions service latency on successful completions. The all-admission table measures scheduled admission through completion for every request, including failures and client-visible timeouts; a timeout therefore contributes approximately the configured 60-second timeout plus admission lag rather than disappearing from the latency distribution.

## Native Add

### Table 5 headline: successful service completions

{_service_table(metrics, "add:native")}

### All-admission user-visible latency

{_user_visible_table(metrics, "add:native")}

## Source-to-searchable Add

### Table 5 headline: successful service completions

{_service_table(metrics, "add:source_to_searchable")}

### All-admission user-visible latency

{_user_visible_table(metrics, "add:source_to_searchable")}

Native Add and Source-to-searchable are isolated workloads and are never pooled together. Visibility-probe failures remain failures in the Source-to-searchable denominator.

## Operational integrity and failures

{_operational_table(metrics)}

Scheduled QPS verifies the immutable 100 ms target grid, while Admission QPS is recomputed from the real per-request admission timestamps and must remain within the frozen tolerance. Completion QPS includes drain time and can fall below the target under overload. Admission lag and queue latency are reported separately instead of being hidden inside successful-only service latency. Exact per-build failure strings, timeout flags, and counts are retained in `aggregates/failures.csv`; pooled error distributions are also retained in `aggregates/operational_integrity.csv`.

### Source-to-searchable visibility outcomes

{_visibility_table(decomposition)}

The visibility table uses every formal admission. A request that commits and then fails or times out during its probe remains a failed admission and is not removed from the denominator. The immutable client-visible timeout stays a failure even when its worker later commits or completes a visibility probe; those drained-worker outcomes and post-timeout work are reported separately, not relabelled as successful latency samples. Full fields are in `aggregates/visibility_integrity.csv`.

## Paper Table 5 reference context

{_paper_reference_table(protocol, paper_reference)}

Paper values come from [Mandol Table 5](https://arxiv.org/pdf/2606.29778). Relative delta is `(local - paper) / paper`; it is context only, not reproduction error, because Track C uses controlled Qwen models and this host is not the paper's H800 environment. Only Mem0 and MemOS Search/Native-Add are paired. Graphiti is deliberately not paired with the paper Zep row; Cognee, TriDB/GEM, and Source-to-searchable have no corresponding Table 5 cells.

## Quality and interpretation gate

{_quality_table(answer_quality)}

### TriDB/GEM quality non-inferiority gate

{_noninferiority_table(noninferiority)}

Retrieval evidence metrics are in `aggregates/quality_gate.csv`; answer/judge results and the pre-registered 2 percentage-point non-inferiority checks are in `aggregates/answer_quality.csv` and `aggregates/quality_noninferiority.csv`. Each gate is a paired one-sided 95% lower confidence bound from 2,000 bootstrap resamples clustered by LoCoMo conversation across all fresh builds. Retrieval, answer, and judge failures score zero. A latency advantage may only be described as same-quality when the lower bound is at least -2 percentage points.

Context tokenization is performed after retrieval and outside the Search latency clock. Token-count coverage is reported explicitly; missing tokenization or answer/judge calls remain failures and are never silently dropped.

## Time and resource breakdown

Per-request stage spans are flattened into `aggregates/time_breakdown.csv`; an explicit `unattributed` row reports service time not covered by observable child spans. Compound native APIs remain `fusion`/`framework_other` rather than receiving invented embedding/vector/graph subdivisions. Build cost and host/GPU sampling are in `aggregates/resource_and_build_cost.csv`; shared model-server GPU/host totals are not attributed exclusively to an adapter.

`aggregates/observable_call_counts.csv` separately reports the HTTP/model-client, database-client, and opaque native-API boundaries visible to the harness for every build. These are client-observed call boundaries, not claimed internal transport round trips. Exact HTTP/database round trips hidden inside Mem0, MemOS, Cognee, or Graphiti compound APIs remain null; one native API call is never converted into one alleged database round trip. Timed-out requests are marked as partial observations.

`aggregates/latency_decomposition.csv` reports Add progress evidence across both successful and failed requests: commit observed, probe started, failed-after-commit, timeout-after-commit, timeout-during-probe, and the final outcome of every worker that drained after a client timeout. It also reports per-system coverage, totals, means, and provenance for native memory/node/edge creation counts. A null count means the native API did not expose a defensible per-request delta; it is never treated as zero. Visibility pass rate uses every completed probe, including probes that returned false on a failed request; it is not restricted to successful requests.

## Figures

- `figures/search_latency_ecdf.pdf`
- `figures/search_p99_bar.pdf`
- `figures/add_latency_ecdf.pdf`
- `figures/add_p99_bar.pdf`
- `figures/formal_success_rate.pdf`
- `figures/latency_quality_frontier.pdf`

All raw request outcomes and spans remain under `runs/`; no timeout or outlier was deleted. The controlled-model services are synchronously stopped before postprocess so their stdout/stderr logs cannot change during hashing. `raw_artifact_manifest.json` enumerates and hashes every formal/warmup/late-outcome/trace JSONL, formal/quality driver and model-service log, run and quality receipt, answer/judge record set, conformance and source receipt, frozen control file, and dataset used by the analysis. The independent final goal auditor re-derives this exact file set and every SHA-256 from disk.
"""


def postprocess(schedule_path: Path) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule.get("status") != "frozen":
        raise RuntimeError("post-processing requires a frozen schedule")
    if benchmark_code_sha256() != schedule.get("benchmark_code_sha256"):
        raise RuntimeError("post-processing code differs from the frozen benchmark")
    if _sha256(Path(__file__).resolve()) != schedule.get("postprocess_script_sha256"):
        raise RuntimeError("post-processing script differs from the frozen schedule")
    goal_audit_path = REPOSITORY / "tools/table5_track_c_goal_audit.py"
    if _sha256(goal_audit_path) != schedule.get("goal_audit_script_sha256"):
        raise RuntimeError("goal auditor differs from the frozen schedule")
    protocol_path = REPOSITORY / schedule["protocol_receipt"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = REPOSITORY / protocol["result_root"]
    runs_root = root / "runs"
    quality_root = root / "quality"
    initial_audit = audit(schedule_path)
    if initial_audit["complete_valid_runs"] != 45:
        raise RuntimeError("post-processing requires all 45 valid formal runs")
    if not initial_audit["prerequisite_complete"]:
        raise RuntimeError(
            "post-processing requires source/model/hardware/conformance gates"
        )
    if not initial_audit["quality_complete"]:
        raise RuntimeError("post-processing requires all 15 quality receipts")
    raw_manifest = _write_raw_artifact_manifest(
        root=root,
        protocol=protocol,
        schedule_path=schedule_path,
        protocol_path=protocol_path,
    )

    metrics, retrieval_quality = aggregate(runs_root)
    decomposition = aggregate_decomposition(runs_root)
    resources = aggregate_resources(runs_root)
    quality_policy = protocol.get("quality") or {}
    answer_quality, noninferiority = aggregate_answer_quality(
        quality_root,
        bootstrap_iterations=int(quality_policy["bootstrap_iterations"]),
        bootstrap_seed=int(quality_policy["bootstrap_seed"]),
        confidence_level=float(quality_policy["one_sided_confidence_level"]),
        noninferiority_margin=(
            float(quality_policy["non_inferiority_margin_percentage_points"]) / 100
        ),
    )
    paper_reference = _paper_reference_rows(protocol, metrics)
    source_provenance = _source_provenance_rows(initial_audit)
    time_breakdown = _time_breakdown(runs_root)
    observable_calls = _observable_call_rows(runs_root)
    operational = _operational_rows(metrics)
    visibility = _visibility_rows(decomposition)
    failures = _failure_rows(runs_root)
    coverage_checks = _final_coverage_checks(
        protocol,
        metrics=metrics,
        retrieval_quality=retrieval_quality,
        decomposition=decomposition,
        resources=resources,
        time_breakdown=time_breakdown,
        observable_calls=observable_calls,
        operational=operational,
        visibility=visibility,
        answer_quality=answer_quality,
        noninferiority=noninferiority,
        paper_reference=paper_reference,
        source_provenance=source_provenance,
    )
    if not all(coverage_checks.values()):
        failed = [name for name, passed in coverage_checks.items() if not passed]
        raise RuntimeError(f"postprocess final coverage is incomplete: {failed}")

    aggregate_dir = root / "aggregates"
    _write_csv(aggregate_dir / "request_metrics.csv", metrics)
    _write_csv(
        aggregate_dir / "search_latency.csv",
        [row for row in metrics if row["phase"] == "search"],
    )
    _write_csv(
        aggregate_dir / "add_native_latency.csv",
        [row for row in metrics if row["phase"] == "add:native"],
    )
    _write_csv(
        aggregate_dir / "add_source_to_searchable_latency.csv",
        [row for row in metrics if row["phase"] == "add:source_to_searchable"],
    )
    _write_csv(aggregate_dir / "quality_gate.csv", retrieval_quality)
    _write_csv(aggregate_dir / "answer_quality.csv", answer_quality)
    _write_csv(aggregate_dir / "quality_noninferiority.csv", noninferiority)
    _write_csv(aggregate_dir / "latency_decomposition.csv", decomposition)
    _write_csv(aggregate_dir / "time_breakdown.csv", time_breakdown)
    _write_csv(
        aggregate_dir / "observable_call_counts.csv",
        observable_calls,
    )
    _write_csv(aggregate_dir / "resource_and_build_cost.csv", resources)
    _write_csv(aggregate_dir / "operational_integrity.csv", operational)
    _write_csv(aggregate_dir / "visibility_integrity.csv", visibility)
    _write_csv(aggregate_dir / "failures.csv", failures)
    _write_csv(aggregate_dir / "paper_reference_delta.csv", paper_reference)
    _write_csv(aggregate_dir / "source_provenance.csv", source_provenance)

    figure_dir = root / "figures"
    plot_search_ecdf(runs_root, figure_dir)
    plot_add_ecdf(runs_root, figure_dir)
    plot_search_box(runs_root, figure_dir)
    plot_search_p99(metrics, figure_dir)
    plot_add_p99(metrics, figure_dir)
    plot_success_rate(metrics, figure_dir)
    if not plot_latency_quality_frontier(metrics, quality_root, figure_dir):
        raise RuntimeError("latency-quality frontier contains no points")

    report_path = root / "REPORT.md"
    report_path.write_text(
        _report(
            protocol,
            metrics,
            answer_quality,
            noninferiority,
            decomposition,
            paper_reference,
            source_provenance,
            initial_audit,
            raw_manifest,
        ),
        encoding="utf-8",
    )
    output_paths = [
        root / relative for relative in protocol["required_deliverables"]
    ] + [
        aggregate_dir / "request_metrics.csv",
        aggregate_dir / "latency_decomposition.csv",
        figure_dir / "search_latency_box.pdf",
    ]
    missing = [str(path) for path in output_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"post-processing missed required files: {missing}")
    receipt = {
        "schema_version": "table5_track_c_completion_postprocess_v0.2.0",
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "schedule": str(schedule_path.resolve()),
        "schedule_sha256": _sha256(schedule_path),
        "protocol_sha256": _sha256(REPOSITORY / schedule["protocol_receipt"]),
        "input_goal_audit": {
            "complete_valid_runs": initial_audit["complete_valid_runs"],
            "quality_complete": initial_audit["quality_complete"],
            "prerequisite_complete": initial_audit["prerequisite_complete"],
        },
        "coverage_checks": coverage_checks,
        "outputs": {
            str(path.relative_to(root)): _sha256(path) for path in output_paths
        },
    }
    _write_json(root / "postprocess_receipt.json", receipt)
    final_audit = audit(schedule_path)
    _write_json(root / "audits/final/goal_audit.json", final_audit)
    if not final_audit["goal_complete"]:
        raise RuntimeError("final goal audit did not pass after post-processing")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    args = parser.parse_args()
    receipt = postprocess(args.schedule.resolve())
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
