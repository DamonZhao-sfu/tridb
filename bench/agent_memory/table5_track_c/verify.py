"""Completeness and protocol-integrity verifier for Track C raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from .dataset import EXPECTED_FORMAL_QUESTION_COUNT, LoCoMoCorpus, load_locomo
from .protocol import (
    _add_items,
    _warmup_queries,
    _write_json,
    add_receipt_contract,
    request_observability_contract,
    search_receipt_contract,
)
from .scheduler import (
    DEFAULT_ADMISSION_LAG_P99_MAX_MS,
    DEFAULT_ADMISSION_QPS_RELATIVE_TOLERANCE,
    DEFAULT_MAX_IN_FLIGHT,
    percentile,
)
from .stats import read_jsonl
from .tracing import observable_call_counts_match_spans


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _corpus(path: str) -> LoCoMoCorpus:
    return load_locomo(path)


def _expected_input_keys(
    receipt: dict[str, Any], phase: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    corpus = _corpus(str(receipt["dataset"]["path"]))
    if phase == "search":
        warmup = _warmup_queries(corpus)
        formal = corpus.formal_queries
        return (
            [(item.sample_id, item.question_id) for item in warmup],
            [(item.sample_id, item.question_id) for item in formal],
        )
    definition = "native" if phase == "add:native" else "source_to_searchable"
    warmup, formal = _add_items(corpus, definition)
    return (
        [(item.sample_id, item.event_id) for item in warmup],
        [(item.sample_id, item.event_id) for item in formal],
    )


def _timings_consistent(records: list[dict[str, Any]]) -> bool:
    for record in records:
        try:
            scheduled = int(record["scheduled_at_ns"])
            admitted = int(record["admitted_at_ns"])
            started = int(record["started_at_ns"])
            completed = int(record["completed_at_ns"])
            if not (scheduled <= admitted <= started <= completed):
                return False
            expected = {
                "admission_lag_ms": (admitted - scheduled) / 1_000_000,
                "queue_latency_ms": (started - scheduled) / 1_000_000,
                "service_latency_ms": (completed - started) / 1_000_000,
                "user_visible_latency_ms": (completed - scheduled) / 1_000_000,
            }
            if any(
                abs(float(record[key]) - value) > 1e-6
                for key, value in expected.items()
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _admission_schedule_checks(
    records: list[dict[str, Any]],
    *,
    target_qps: float,
    relative_tolerance: float,
    lag_p99_max_ms: float,
) -> dict[str, bool]:
    """Prove that real admissions, not only synthetic timestamps, met load."""
    if len(records) < 2 or target_qps <= 0 or relative_tolerance < 0:
        return {
            "admission_order_exact": False,
            "actual_admission_qps_within_tolerance": False,
            "admission_lag_p99_within_limit": False,
        }
    try:
        request_order = _in_request_order(records)
        admitted = [int(record["admitted_at_ns"]) for record in request_order]
        lags = [float(record["admission_lag_ms"]) for record in request_order]
        span_ns = max(admitted) - min(admitted)
        actual_qps = (
            (len(admitted) - 1) / (span_ns / 1_000_000_000) if span_ns > 0 else None
        )
        lag_p99 = percentile(lags, 99)
    except (KeyError, TypeError, ValueError):
        actual_qps = None
        lag_p99 = None
        admitted = []
    return {
        "admission_order_exact": len(admitted) == len(records)
        and all(
            admitted[index] <= admitted[index + 1] for index in range(len(admitted) - 1)
        ),
        "actual_admission_qps_within_tolerance": actual_qps is not None
        and abs(actual_qps - target_qps) / target_qps <= relative_tolerance,
        "admission_lag_p99_within_limit": lag_p99 is not None
        and lag_p99 <= lag_p99_max_ms,
    }


def _indices_exact(records: list[dict[str, Any]], expected: int) -> bool:
    indices = [record.get("request_index") for record in records]
    return all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ) and sorted(indices) == list(range(expected))


def _in_request_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            record["request_index"]
            if isinstance(record.get("request_index"), int)
            and not isinstance(record.get("request_index"), bool)
            else -1
        ),
    )


def _add_progress_evidence(records: list[dict[str, Any]], *, visibility: bool) -> bool:
    required = {
        "commit_observed",
        "visibility_requested",
        "visibility_probe_started",
        "committed_at_ns",
        "searchable_at_ns",
        "visibility_probe",
        "visibility_error",
    }
    for record in records:
        observed = record.get("receipt") or {}
        if not required <= set(observed):
            return False
        committed = observed.get("committed_at_ns") is not None
        if observed.get("commit_observed") is not committed:
            return False
        if observed.get("visibility_requested") is not visibility:
            return False
        if not isinstance(observed.get("visibility_probe_started"), bool):
            return False
        if not observed["visibility_probe_started"] and (
            observed.get("visibility_probe") is not None
            or observed.get("searchable_at_ns") is not None
        ):
            return False
        if not visibility and (
            observed["visibility_probe_started"]
            or observed.get("visibility_probe") is not None
            or observed.get("searchable_at_ns") is not None
        ):
            return False
        if (
            visibility
            and record.get("success")
            and (
                not committed
                or not observed["visibility_probe_started"]
                or observed.get("visibility_probe") is not True
                or observed.get("searchable_at_ns") is None
            )
        ):
            return False
    return True


def _late_outcome_checks(
    *,
    late_path: Path,
    late_records: list[dict[str, Any]],
    request_records: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> dict[str, bool]:
    """Bind every client timeout to exactly one drained worker outcome."""

    timed_out = [record for record in request_records if record.get("timeout") is True]

    def key(record: dict[str, Any]) -> tuple[str, int] | None:
        phase = record.get("phase")
        index = record.get("request_index")
        if (
            not isinstance(phase, str)
            or not phase
            or not isinstance(index, int)
            or isinstance(index, bool)
        ):
            return None
        return phase, index

    timed_out_by_key = {key(record): record for record in timed_out}
    late_by_key = {key(record): record for record in late_records}
    identities_match = True
    timings_match = True
    outcomes_explicit = True
    for late in late_records:
        request = timed_out_by_key.get(key(late))
        if request is None:
            identities_match = timings_match = False
            continue
        identities_match = identities_match and (
            late.get("system") == request.get("system") == receipt.get("system")
            and late.get("build_id")
            == request.get("build_id")
            == receipt.get("build_id")
            and late.get("trace_id") == request.get("trace_id")
            and all(
                late.get(field) == request.get(field)
                for field in ("sample_id", "question_id", "event_id")
                if field in request
            )
        )
        try:
            client_timeout = int(late["client_timed_out_at_ns"])
            worker_complete = int(late["worker_completed_at_ns"])
            post_timeout_ms = float(late["post_timeout_work_ms"])
            timings_match = timings_match and (
                client_timeout == int(request["completed_at_ns"])
                and worker_complete >= client_timeout
                and abs(
                    post_timeout_ms - (worker_complete - client_timeout) / 1_000_000
                )
                <= 1e-6
            )
        except (KeyError, TypeError, ValueError):
            timings_match = False
        final_success = late.get("final_success")
        final_error = late.get("final_error")
        outcomes_explicit = outcomes_explicit and (
            isinstance(final_success, bool)
            and (
                (final_success is True and final_error is None)
                or (
                    final_success is False
                    and isinstance(final_error, str)
                    and bool(final_error)
                )
            )
            and isinstance(late.get("receipt"), dict)
        )

    phase_counts = Counter(str(record.get("phase")) for record in late_records)
    post_timeout_values: list[float] = []
    try:
        post_timeout_values = [
            float(record["post_timeout_work_ms"]) for record in late_records
        ]
    except (KeyError, TypeError, ValueError):
        pass
    expected_summary = {
        "schema_version": "table5_track_c_late_outcome_summary_v0.1.0",
        "path": str(late_path.resolve()),
        "sha256": _sha256(late_path),
        "records": len(late_records),
        "by_phase": dict(sorted(phase_counts.items())),
        "final_success": sum(
            record.get("final_success") is True for record in late_records
        ),
        "final_failed": sum(
            record.get("final_success") is False for record in late_records
        ),
        "post_timeout_work_ms": {
            "total": sum(post_timeout_values),
            "max": max(post_timeout_values, default=None),
        },
    }
    return {
        "late_outcome_jsonl_exists": late_path.is_file(),
        "late_outcome_schema_frozen": all(
            record.get("schema_version") == "table5_track_c_late_outcome_v0.1.0"
            for record in late_records
        ),
        "one_late_outcome_per_timeout": len(late_records) == len(timed_out)
        and None not in timed_out_by_key
        and None not in late_by_key
        and len(timed_out_by_key) == len(timed_out)
        and len(late_by_key) == len(late_records)
        and set(late_by_key) == set(timed_out_by_key),
        "late_outcome_identity_linkage": identities_match,
        "late_outcome_timings_recompute": timings_match,
        "late_outcome_final_state_explicit": outcomes_explicit,
        "late_outcome_summary_recomputed": receipt.get("late_outcome_summary")
        == expected_summary,
    }


def verify_run(receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    phase = str(receipt["phase"])
    run_dir = receipt_path.parent
    formal_path = run_dir / "formal.jsonl"
    warmup_path = run_dir / "warmup.jsonl"
    spans_path = run_dir / "spans.jsonl"
    late_path = run_dir / "late_outcomes.jsonl"
    expected_formal = EXPECTED_FORMAL_QUESTION_COUNT if phase == "search" else 2_000
    expected_warmup = 307 if phase == "search" else 10
    known_phase = phase in {"search", "add:native", "add:source_to_searchable"}
    fresh_state = receipt.get("execution", {}).get("fresh_state_launcher") or {}
    fresh_policy = (
        "hashed launcher refuses every pre-existing per-point database, volume, "
        "or namespace before creation"
    )
    checks: dict[str, bool] = {
        "known_phase": known_phase,
        "receipt_schema_frozen": receipt.get("schema_version")
        in {"table5_track_c_run_v0.2.0", "table5_track_c_run_v0.3.0"},
        "receipt_complete": receipt.get("status") == "complete",
        "formal_jsonl_exists": formal_path.is_file(),
        "warmup_jsonl_exists": warmup_path.is_file(),
        "checksum_frozen": receipt.get("dataset", {}).get("sha256")
        == "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "qps_frozen": receipt.get("protocol", {}).get("qps") == 10.0,
        "timeout_frozen": receipt.get("protocol", {}).get("timeout_seconds") == 60.0,
        "max_in_flight_frozen": receipt.get("protocol", {}).get("max_in_flight")
        == DEFAULT_MAX_IN_FLIGHT,
        "scheduled_interval_frozen": receipt.get("protocol", {}).get(
            "scheduled_interval_ns"
        )
        == 100_000_000,
        "admission_qps_tolerance_frozen": receipt.get("protocol", {}).get(
            "admission_qps_relative_tolerance"
        )
        == DEFAULT_ADMISSION_QPS_RELATIVE_TOLERANCE,
        "admission_lag_limit_frozen": receipt.get("protocol", {}).get(
            "admission_lag_p99_max_ms"
        )
        == DEFAULT_ADMISSION_LAG_P99_MAX_MS,
        "harness_retry_disabled": receipt.get("protocol", {}).get("retries") == 0
        and receipt.get("protocol", {}).get("harness_retries") == 0,
        "top_k_frozen": receipt.get("protocol", {}).get("top_k") == 35,
        "evaluation_labels_hidden_from_adapter": receipt.get("protocol", {}).get(
            "evaluation_fields_visible_to_adapter"
        )
        is False,
        "call_observability_policy_explicit": receipt.get("protocol", {}).get(
            "per_request_call_observability"
        )
        == (receipt.get("protocol", {}).get("stage_profiling") is True)
        and receipt.get("protocol", {}).get("call_observation_fields")
        == [
            "http_client",
            "database_client",
            "opaque_native_api",
            "total",
            "source",
            "scope",
            "exact_internal_http_round_trips",
            "exact_internal_database_round_trips",
            "internal_round_trip_policy",
        ]
        and receipt.get("protocol", {}).get("internal_round_trip_policy")
        == "client-boundary counts only; exact HTTP/database round trips inside "
        "compound native APIs are unavailable and never inferred",
        "benchmark_code_hash_present": bool(
            receipt.get("execution", {}).get("benchmark_code_sha256")
        ),
        "launcher_script_hash_present": bool(
            receipt.get("execution", {}).get("launcher_script_sha256")
        ),
        "formal_schedule_hash_present": bool(
            receipt.get("execution", {}).get("formal_schedule_sha256")
        ),
        "protocol_receipt_hash_present": bool(
            receipt.get("execution", {}).get("protocol_receipt_sha256")
        ),
        "fresh_state_policy_frozen": receipt.get("protocol", {}).get(
            "fresh_state_policy"
        )
        == fresh_policy
        and receipt.get("protocol", {}).get("fresh_state_receipt_schema")
        == "table5_track_c_fresh_state_launcher_v0.1.0",
        "fresh_state_launcher_receipt": fresh_state.get("schema_version")
        == "table5_track_c_fresh_state_launcher_v0.1.0"
        and fresh_state.get("policy") == fresh_policy
        and fresh_state.get("build_id") == receipt.get("build_id")
        and fresh_state.get("preexisting_check_passed") is True
        and all(
            isinstance(fresh_state.get(field), str) and bool(fresh_state[field].strip())
            for field in ("kind", "primary_identity", "secondary_identity")
        ),
        "search_build_finalized": phase != "search"
        or (
            receipt.get("build_mode") == "fresh_ingest"
            and isinstance(receipt.get("build_finalize"), dict)
            and isinstance(receipt.get("build_wall_seconds"), (int, float))
            and receipt["build_wall_seconds"] >= 0
        ),
        "add_state_fresh": phase == "search"
        or receipt.get("state_mode") == "fresh_empty",
    }
    formal = read_jsonl([formal_path]) if formal_path.exists() else []
    warmup = read_jsonl([warmup_path]) if warmup_path.exists() else []
    spans = read_jsonl([spans_path]) if spans_path.exists() else []
    late_outcomes = read_jsonl([late_path]) if late_path.exists() else []
    indices = [record.get("request_index") for record in formal]
    warmup_indices = [record.get("request_index") for record in warmup]
    # JsonlSink flushes each request as soon as it completes so a crash leaves
    # the largest possible auditable prefix.  Under open-loop concurrency that
    # is completion order, not admission order.  Reconstruct the frozen input
    # order from the unique request index before checking identities/schedule.
    formal_in_request_order = _in_request_order(formal)
    warmup_in_request_order = _in_request_order(warmup)
    expected_formal_phase = {
        "search": "formal_search",
        "add:native": "formal_add_native",
        "add:source_to_searchable": "formal_add_source_to_searchable",
    }.get(phase)
    expected_warmup_phase = {
        "search": "warmup_search",
        "add:native": "warmup_add_native",
        "add:source_to_searchable": "warmup_add_source_to_searchable",
    }.get(phase)
    if known_phase:
        warmup_expected_keys, formal_expected_keys = _expected_input_keys(
            receipt, phase
        )
    else:
        warmup_expected_keys, formal_expected_keys = [], []
    identity_key = "question_id" if phase == "search" else "event_id"
    checks.update(
        {
            "formal_cardinality": len(formal) == expected_formal,
            "warmup_cardinality": len(warmup) == expected_warmup,
            "formal_schema_frozen": all(
                record.get("schema_version") == "table5_track_c_request_v0.3.0"
                for record in formal
            ),
            "warmup_schema_frozen": all(
                record.get("schema_version") == "table5_track_c_request_v0.3.0"
                for record in warmup
            ),
            "formal_indices_exact": _indices_exact(formal, expected_formal),
            "one_record_per_formal_request": len(set(indices)) == len(indices),
            "warmup_indices_exact": _indices_exact(warmup, expected_warmup),
            "one_record_per_warmup_request": len(set(warmup_indices))
            == len(warmup_indices),
            "formal_system_consistent": all(
                record.get("system") == receipt.get("system") for record in formal
            ),
            "formal_build_consistent": all(
                record.get("build_id") == receipt.get("build_id") for record in formal
            ),
            "warmup_system_consistent": all(
                record.get("system") == receipt.get("system") for record in warmup
            ),
            "warmup_build_consistent": all(
                record.get("build_id") == receipt.get("build_id") for record in warmup
            ),
            "formal_phase_consistent": all(
                record.get("phase") == expected_formal_phase for record in formal
            ),
            "warmup_phase_consistent": all(
                record.get("phase") == expected_warmup_phase for record in warmup
            ),
            "formal_inputs_exact": [
                (str(record.get("sample_id")), str(record.get(identity_key)))
                for record in formal_in_request_order
            ]
            == formal_expected_keys,
            "warmup_inputs_exact": [
                (str(record.get("sample_id")), str(record.get(identity_key)))
                for record in warmup_in_request_order
            ]
            == warmup_expected_keys,
            "formal_timings_recompute": _timings_consistent(formal),
            "warmup_timings_recompute": _timings_consistent(warmup),
            "all_outcomes_explicit": all(
                isinstance(record.get("success"), bool)
                and isinstance(record.get("timeout"), bool)
                and (
                    (
                        record["success"]
                        and record["timeout"] is False
                        and record.get("error") is None
                    )
                    or (
                        record["success"] is False
                        and isinstance(record.get("error"), str)
                        and bool(record["error"])
                    )
                )
                for record in formal
            ),
            "per_request_harness_retry_zero": all(
                record.get("harness_retries") == 0 for record in formal
            ),
            "native_retry_counter_explicitly_unavailable": all(
                "system_internal_retries" in record for record in formal
            ),
        }
    )
    stage_profiling = receipt.get("protocol", {}).get("stage_profiling") is True
    checks["formal_request_observability"] = request_observability_contract(
        formal, stage_profiling=stage_profiling
    )
    checks["warmup_request_observability"] = request_observability_contract(
        warmup, stage_profiling=stage_profiling
    )
    checks["observable_call_counts_recomputed_from_spans"] = not stage_profiling or (
        spans_path.is_file()
        and observable_call_counts_match_spans([*formal, *warmup], spans)
    )
    checks.update(
        _late_outcome_checks(
            late_path=late_path,
            late_records=late_outcomes,
            request_records=[*formal, *warmup],
            receipt=receipt,
        )
    )
    if len(formal) > 1:
        schedule_steps = [
            formal_in_request_order[index]["scheduled_at_ns"]
            - formal_in_request_order[index - 1]["scheduled_at_ns"]
            for index in range(1, len(formal_in_request_order))
        ]
        checks["absolute_100ms_schedule"] = set(schedule_steps) == {100_000_000}
    else:
        checks["absolute_100ms_schedule"] = False
    checks.update(
        {
            f"formal_{name}": passed
            for name, passed in _admission_schedule_checks(
                formal,
                target_qps=10.0,
                relative_tolerance=DEFAULT_ADMISSION_QPS_RELATIVE_TOLERANCE,
                lag_p99_max_ms=DEFAULT_ADMISSION_LAG_P99_MAX_MS,
            ).items()
        }
    )

    if phase == "add:source_to_searchable":
        checks["source_to_searchable_progress_evidence_explicit"] = (
            _add_progress_evidence(formal, visibility=True)
        )
        checks["successful_visibility_probes_passed"] = all(
            not record["success"]
            or (record.get("receipt") or {}).get("visibility_probe") is True
            for record in formal
        )
        checks["visibility_failure_receipts_retained"] = all(
            "visibility probe failed" not in str(record.get("error") or "").lower()
            or (record.get("receipt") or {}).get("visibility_probe") is False
            for record in formal
        )
    if phase == "add:native":
        checks["native_add_progress_evidence_explicit"] = _add_progress_evidence(
            formal, visibility=False
        )
        checks["native_visibility_not_mixed"] = all(
            (record.get("receipt") or {}).get("visibility_probe") is None
            for record in formal
            if record["success"]
        )
    if phase in {"add:native", "add:source_to_searchable"}:
        checks["successful_add_creation_receipt_contract"] = add_receipt_contract(
            formal
        )
    if phase == "search":
        checks["successful_search_receipt_contract"] = search_receipt_contract(formal)
    error_counts = Counter(str(record.get("error") or "success") for record in formal)
    return {
        "receipt": str(receipt_path),
        "system": receipt.get("system"),
        "build_id": receipt.get("build_id"),
        "phase": phase,
        "checks": checks,
        "passed": all(checks.values()),
        "formal_records": len(formal),
        "warmup_records": len(warmup),
        "raw_sha256": {
            "formal_jsonl": _sha256(formal_path),
            "warmup_jsonl": _sha256(warmup_path),
            "late_outcomes_jsonl": _sha256(late_path),
            "run_receipt": _sha256(receipt_path),
        },
        "outcomes": dict(sorted(error_counts.items())),
    }


def verify_tree(
    root: str | Path,
    *,
    expected_code_hash: str | None = None,
    expected_launcher_hash: str | None = None,
    expected_schedule_hash: str | None = None,
    expected_protocol_hash: str | None = None,
) -> dict[str, Any]:
    root = Path(root)
    runs = [verify_run(path) for path in sorted(root.glob("**/run_receipt.json"))]
    completed_runs = [run for run in runs if run["checks"]["receipt_complete"]]
    by_system_phase: Counter[tuple[str, str]] = Counter(
        (str(run["system"]), str(run["phase"]))
        for run in completed_runs
        if run["passed"]
    )
    expected_systems = {
        "tridb_gem",
        "mem0",
        "memos",
        "cognee",
        "graphiti_zep_oss_proxy",
    }
    phase_build_stems = {
        "search": "search",
        "add:native": "add_native",
        "add:source_to_searchable": "add_source_to_searchable",
    }
    expected_identities = {
        (system, phase, f"tc5v2_{system}_{stem}_{round_name}")
        for system in expected_systems
        for phase, stem in phase_build_stems.items()
        for round_name in ("b1", "b2", "b3")
    }
    actual_identities = {
        (str(run["system"]), str(run["phase"]), str(run["build_id"])) for run in runs
    }
    expected_paths = {
        f"{system}/{stem}/{round_name}/run_receipt.json"
        for system in expected_systems
        for stem in phase_build_stems.values()
        for round_name in ("b1", "b2", "b3")
    }
    actual_paths = {
        str(Path(run["receipt"]).resolve().relative_to(root.resolve())) for run in runs
    }
    formal_tree_exact = (
        len(runs) == 45
        and actual_identities == expected_identities
        and actual_paths == expected_paths
    )
    coverage = {
        system: {
            phase: by_system_phase[(system, phase)]
            for phase in ("search", "add:native", "add:source_to_searchable")
        }
        for system in sorted(expected_systems)
    }
    coverage_complete = all(
        count == 3 for phases in coverage.values() for count in phases.values()
    )
    code_hashes = sorted(
        {
            str(run_receipt.get("execution", {}).get("benchmark_code_sha256"))
            for path in sorted(root.glob("**/run_receipt.json"))
            for run_receipt in [json.loads(path.read_text(encoding="utf-8"))]
            if run_receipt.get("status") == "complete"
        }
    )
    frozen_hash_matches = bool(expected_code_hash) and code_hashes == [
        expected_code_hash
    ]
    launcher_hashes = sorted(
        {
            str(run_receipt.get("execution", {}).get("launcher_script_sha256"))
            for path in sorted(root.glob("**/run_receipt.json"))
            for run_receipt in [json.loads(path.read_text(encoding="utf-8"))]
            if run_receipt.get("status") == "complete"
        }
    )
    schedule_hashes = sorted(
        {
            str(run_receipt.get("execution", {}).get("formal_schedule_sha256"))
            for path in sorted(root.glob("**/run_receipt.json"))
            for run_receipt in [json.loads(path.read_text(encoding="utf-8"))]
            if run_receipt.get("status") == "complete"
        }
    )
    protocol_hashes = sorted(
        {
            str(run_receipt.get("execution", {}).get("protocol_receipt_sha256"))
            for path in sorted(root.glob("**/run_receipt.json"))
            for run_receipt in [json.loads(path.read_text(encoding="utf-8"))]
            if run_receipt.get("status") == "complete"
        }
    )
    return {
        "schema_version": "table5_track_c_verification_v0.3.0",
        "root": str(root.resolve()),
        "runs_found": len(runs),
        "formal_tree_exact": formal_tree_exact,
        "expected_identities": [list(value) for value in sorted(expected_identities)],
        "actual_identities": [list(value) for value in sorted(actual_identities)],
        "expected_receipt_paths": sorted(expected_paths),
        "actual_receipt_paths": sorted(actual_paths),
        "runs": runs,
        "all_found_runs_pass": bool(runs) and all(run["passed"] for run in runs),
        "coverage": coverage,
        "coverage_complete_for_five_systems": coverage_complete,
        "completed_run_benchmark_code_hashes": code_hashes,
        "one_frozen_benchmark_code_hash": len(code_hashes) == 1,
        "expected_benchmark_code_hash": expected_code_hash,
        "frozen_benchmark_code_hash_matches": frozen_hash_matches,
        "completed_run_launcher_hashes": launcher_hashes,
        "expected_launcher_hash": expected_launcher_hash,
        "frozen_launcher_hash_matches": bool(expected_launcher_hash)
        and launcher_hashes == [expected_launcher_hash],
        "completed_run_schedule_hashes": schedule_hashes,
        "expected_schedule_hash": expected_schedule_hash,
        "frozen_schedule_hash_matches": bool(expected_schedule_hash)
        and schedule_hashes == [expected_schedule_hash],
        "completed_run_protocol_hashes": protocol_hashes,
        "expected_protocol_hash": expected_protocol_hash,
        "frozen_protocol_hash_matches": bool(expected_protocol_hash)
        and protocol_hashes == [expected_protocol_hash],
        "graphiti_proxy_expected_in_track_c": True,
        "production_zep_expected_in_track_c": False,
        "zep_note": (
            "Graphiti is measured only as 'Graphiti (Zep OSS proxy)'; "
            "the row is not production Zep or a reproduced Zep Table 5 row."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-launcher-hash", required=True)
    parser.add_argument("--expected-schedule-hash", required=True)
    parser.add_argument("--expected-protocol-hash", required=True)
    args = parser.parse_args(argv)
    result = verify_tree(
        args.root,
        expected_code_hash=args.expected_code_hash,
        expected_launcher_hash=args.expected_launcher_hash,
        expected_schedule_hash=args.expected_schedule_hash,
        expected_protocol_hash=args.expected_protocol_hash,
    )
    output = Path(args.output) if args.output else Path(args.root) / "verification.json"
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(
        not (
            result["all_found_runs_pass"]
            and result["formal_tree_exact"]
            and result["coverage_complete_for_five_systems"]
            and result["frozen_benchmark_code_hash_matches"]
            and result["frozen_launcher_hash_matches"]
            and result["frozen_schedule_hash_matches"]
            and result["frozen_protocol_hash_matches"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
