#!/usr/bin/env python3
"""Audit the full five-system Track C goal without mutating experiment data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from bench.agent_memory.table5_track_c.dataset import LoCoMoCorpus, load_locomo
from bench.agent_memory.table5_track_c.protocol import (
    _add_items,
    _warmup_queries,
    add_receipt_contract,
    benchmark_code_sha256,
    request_observability_contract,
    search_receipt_contract,
)
from bench.agent_memory.table5_track_c.quality import ANSWER_PROMPT, JUDGE_PROMPT
from bench.agent_memory.table5_track_c.tracing import (
    STAGE_CATEGORIES,
    observable_call_counts_match_spans,
    summarize_spans,
)
from bench.agent_memory.table5_track_c.tridb_invariants import INVARIANT_CHECK_NAMES
from experiments.graphiti_track_c_formal.hashing import graphiti_adapter_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE = (
    REPOSITORY
    / "bench/agent_memory/table5_track_c/manifests/completion_v2_schedule.json"
)
HASHED_SCRIPTS = {
    "run_one_script_sha256": "scripts/table5_track_c_completion_v2_run_one.sh",
    "formal_driver_script_sha256": (
        "scripts/table5_track_c_completion_v2_formal_driver.sh"
    ),
    "conformance_driver_script_sha256": (
        "scripts/table5_track_c_completion_v2_conformance.sh"
    ),
    "quality_driver_script_sha256": (
        "scripts/table5_track_c_completion_v2_quality_driver.sh"
    ),
    "postprocess_script_sha256": "tools/table5_track_c_completion_v2_postprocess.py",
    "execution_driver_script_sha256": (
        "scripts/table5_track_c_completion_v2_execute.sh"
    ),
    "goal_audit_script_sha256": "tools/table5_track_c_goal_audit.py",
    "memos_backend_script_sha256": "scripts/table5_track_c_memos_neo4j.sh",
    "graphiti_backend_script_sha256": "scripts/table5_track_c_graphiti_neo4j.sh",
}
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
EXPECTED_DISPLAY_NAMES = {
    "tridb_gem": "TriDB/GEM",
    "mem0": "Mem0 2.0.18",
    "memos": "MemOS 2.0.30",
    "cognee": "Cognee 1.5.0",
    "graphiti_zep_oss_proxy": "Graphiti (Zep OSS proxy)",
    "all": "All systems",
}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_path_is_within(path: Path, root: Path) -> bool:
    """Compare path containment after resolving equivalent symlink spellings."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _path_entry_is_within(path: Path, root: Path) -> bool:
    """Check the directory entry location without following its final symlink."""
    try:
        return path.parent.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_manifest_observation(protocol: dict[str, Any]) -> dict[str, Any]:
    declared = protocol.get("dataset") or {}
    path = Path(str(declared.get("path") or ""))
    result: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "observed": None,
    }
    try:
        corpus = load_locomo(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    observed = {
        "path": str(path),
        "sha256": corpus.sha256,
        "conversations": len(corpus.sample_ids),
        "events": len(corpus.all_events),
        "questions_total": sum(
            len(queries) for queries in corpus.queries_by_sample.values()
        ),
        "formal_search": len(corpus.formal_queries),
    }
    result["resolved_path"] = str(corpus.path)
    result["observed"] = observed
    result["valid"] = observed == declared and path.resolve() == corpus.path
    return result


@lru_cache(maxsize=4)
def _canonical_corpus(path: str) -> LoCoMoCorpus:
    return load_locomo(path)


def _canonical_input_keys(
    protocol: dict[str, Any], workload: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    corpus = _canonical_corpus(str(protocol["dataset"]["path"]))
    if workload == "search":
        warmup = _warmup_queries(corpus)
        formal = corpus.formal_queries
        return (
            [(item.sample_id, item.question_id) for item in warmup],
            [(item.sample_id, item.question_id) for item in formal],
        )
    definition = "native" if workload == "add_native" else "source_to_searchable"
    warmup, formal = _add_items(corpus, definition)
    return (
        [(item.sample_id, item.event_id) for item in warmup],
        [(item.sample_id, item.event_id) for item in formal],
    )


def _input_keys_exact(
    records: list[dict[str, Any]],
    expected: list[tuple[str, str]],
    identity_key: str,
) -> bool:
    if not all(
        isinstance(row.get("request_index"), int)
        and not isinstance(row.get("request_index"), bool)
        for row in records
    ):
        return False
    request_order = sorted(records, key=lambda row: int(row["request_index"]))
    observed = [
        (str(row.get("sample_id")), str(row.get(identity_key))) for row in request_order
    ]
    return observed == expected


def _jsonl(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors = 0
    if not path.is_file():
        return {"exists": False, "records": records, "errors": errors}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if not isinstance(value, dict):
                errors += 1
                continue
            records.append(value)
    return {"exists": True, "records": records, "errors": errors}


def _phase_names(workload: str) -> tuple[str, str, str]:
    if workload == "search":
        return "search", "formal_search", "warmup_search"
    if workload == "add_native":
        return "add:native", "formal_add_native", "warmup_add_native"
    if workload == "add_source_to_searchable":
        return (
            "add:source_to_searchable",
            "formal_add_source_to_searchable",
            "warmup_add_source_to_searchable",
        )
    raise ValueError(f"unknown workload: {workload}")


def _fresh_state_launcher_checks(
    receipt: dict[str, Any],
    protocol: dict[str, Any],
    *,
    system: str,
    workload: str,
    round_name: str,
) -> dict[str, bool]:
    build_id = f"tc5v2_{system}_{workload}_{round_name}"
    database_name = build_id.replace("graphiti_zep_oss_proxy", "graphiti")
    database_name = database_name.replace("add_source_to_searchable", "source")
    database_name = database_name.replace("add_native", "native")
    external = Path("/localhome/hza214/agent-memory-table5")
    volume = external / "volumes/completion_v2" / system / build_id
    expected = {
        "tridb_gem": {
            "kind": "postgres_database",
            "primary_identity": database_name,
            "secondary_identity": "database_owner:hza214",
        },
        "mem0": {
            "kind": "postgres_database_and_filesystem",
            "primary_identity": database_name,
            "secondary_identity": str(volume),
        },
        "cognee": {
            "kind": "postgres_database_and_dataset_namespace",
            "primary_identity": database_name,
            "secondary_identity": f"dataset_prefix:{build_id}",
        },
        "memos": {
            "kind": "filesystem_and_neo4j_state",
            "primary_identity": str(volume),
            "secondary_identity": str(external / "volumes/memos_neo4j" / build_id),
        },
        "graphiti_zep_oss_proxy": {
            "kind": "neo4j_state_directory",
            "primary_identity": str(
                external / "volumes/completion_v2/graphiti_neo4j" / build_id
            ),
            "secondary_identity": (
                "systemd_unit:tridb-table5-graphiti-completion-v2-neo4j.service"
            ),
        },
    }.get(system)
    observed = (receipt.get("execution") or {}).get("fresh_state_launcher") or {}
    policy = (protocol.get("protocol") or {}).get("fresh_state_policy")
    return {
        "known_system": expected is not None,
        "schema": observed.get("schema_version")
        == "table5_track_c_fresh_state_launcher_v0.1.0",
        "policy": observed.get("policy") == policy,
        "build_id": observed.get("build_id") == build_id,
        "kind": expected is not None and observed.get("kind") == expected["kind"],
        "primary_identity": expected is not None
        and observed.get("primary_identity") == expected["primary_identity"],
        "secondary_identity": expected is not None
        and observed.get("secondary_identity") == expected["secondary_identity"],
        "preexisting_check_passed": observed.get("preexisting_check_passed") is True,
    }


def _root_trace_ids(records: list[dict[str, Any]], phase: str) -> set[str]:
    return {
        str(row["trace_id"])
        for row in records
        if row.get("phase") == phase
        and row.get("category") == "request"
        and row.get("parent_span_id") is None
        and isinstance(row.get("trace_id"), str)
        and row["trace_id"]
    }


def _span_tree_checks(
    records: list[dict[str, Any]],
    *,
    formal_traces: set[str],
    warmup_traces: set[str],
    formal_phase: str,
    warmup_phase: str,
    system: str,
    build_id: str,
) -> dict[str, bool]:
    expected_traces = formal_traces | warmup_traces
    roots = [row for row in records if row.get("category") == "request"]
    children = [row for row in records if row.get("category") != "request"]
    keys = [(row.get("trace_id"), row.get("span_id")) for row in records]
    span_ids_by_trace: dict[str, set[str]] = {}
    root_identity: dict[str, tuple[str, int]] = {}
    for row in records:
        trace_id = str(row.get("trace_id") or "")
        span_id = str(row.get("span_id") or "")
        span_ids_by_trace.setdefault(trace_id, set()).add(span_id)
    for row in roots:
        trace_id = str(row.get("trace_id") or "")
        request_index = row.get("request_index")
        if isinstance(request_index, int) and not isinstance(request_index, bool):
            root_identity[trace_id] = (str(row.get("phase") or ""), request_index)

    def lower_hex(value: Any, length: int) -> bool:
        return (
            isinstance(value, str)
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        )

    def timing(row: dict[str, Any]) -> bool:
        try:
            started = int(row["started_at_ns"])
            completed = int(row["completed_at_ns"])
            duration = float(row["duration_ms"])
            return (
                started <= completed
                and abs(duration - (completed - started) / 1_000_000) <= 1e-6
            )
        except (KeyError, TypeError, ValueError):
            return False

    def outcome(row: dict[str, Any]) -> bool:
        success = row.get("success")
        timed_out = row.get("timeout")
        error = row.get("error")
        return (
            isinstance(success, bool)
            and isinstance(timed_out, bool)
            and (
                (success and error is None)
                or (not success and isinstance(error, str) and bool(error))
            )
        )

    child_traces = {str(row.get("trace_id") or "") for row in children}
    return {
        "schema": all(
            row.get("schema_version") == "table5_track_c_span_v0.1.0" for row in records
        ),
        "trace_ids": all(lower_hex(row.get("trace_id"), 32) for row in records),
        "span_ids": all(lower_hex(row.get("span_id"), 16) for row in records),
        "unique_span_keys": len(keys) == len(set(keys)),
        "exact_trace_set": {str(row.get("trace_id") or "") for row in records}
        == expected_traces,
        "one_root_per_trace": len(roots) == len(expected_traces)
        and set(root_identity) == expected_traces,
        "root_parent": all(row.get("parent_span_id") is None for row in roots),
        "child_parent": all(
            isinstance(row.get("parent_span_id"), str)
            and row["parent_span_id"] != row.get("span_id")
            and row["parent_span_id"]
            in span_ids_by_trace.get(str(row.get("trace_id") or ""), set())
            for row in children
        ),
        "categories": all(row.get("category") in STAGE_CATEGORIES for row in records),
        "child_categories": all(row.get("category") != "request" for row in children),
        "identity": all(
            row.get("system") == system and row.get("build_id") == build_id
            for row in records
        ),
        "phase": all(
            row.get("phase") in {formal_phase, warmup_phase} for row in records
        ),
        "request_identity": all(
            root_identity.get(str(row.get("trace_id") or ""))
            == (row.get("phase"), row.get("request_index"))
            for row in records
        ),
        "timings": all(timing(row) for row in records),
        "outcomes": all(outcome(row) for row in records),
        "formal_child_coverage": formal_traces <= child_traces,
        "warmup_child_coverage": warmup_traces <= child_traces,
    }


def _request_checks(
    scan: dict[str, Any],
    *,
    expected_count: int,
    expected_phase: str,
    expected_system: str,
    expected_build: str,
    expected_interval_ns: int | None = None,
    expected_qps: float | None = None,
    admission_qps_relative_tolerance: float | None = None,
    admission_lag_p99_max_ms: float | None = None,
) -> dict[str, bool]:
    records = scan["records"]
    indices = [row.get("request_index") for row in records]
    trace_ids = [row.get("trace_id") for row in records]
    integer_indices = all(
        isinstance(value, int) and not isinstance(value, bool) for value in indices
    )
    request_order = (
        sorted(records, key=lambda row: int(row["request_index"]))
        if integer_indices
        else []
    )

    def timings_consistent(row: dict[str, Any]) -> bool:
        try:
            scheduled = int(row["scheduled_at_ns"])
            admitted = int(row["admitted_at_ns"])
            started = int(row["started_at_ns"])
            completed = int(row["completed_at_ns"])
            if not scheduled <= admitted <= started <= completed:
                return False
            expected = {
                "admission_lag_ms": (admitted - scheduled) / 1_000_000,
                "queue_latency_ms": (started - scheduled) / 1_000_000,
                "service_latency_ms": (completed - started) / 1_000_000,
                "user_visible_latency_ms": (completed - scheduled) / 1_000_000,
            }
            return all(
                abs(float(row[key]) - value) <= 1e-6 for key, value in expected.items()
            )
        except (KeyError, TypeError, ValueError):
            return False

    schedule_exact = True
    if expected_interval_ns is not None:
        schedule_exact = len(request_order) == expected_count and all(
            int(request_order[index]["scheduled_at_ns"])
            - int(request_order[index - 1]["scheduled_at_ns"])
            == expected_interval_ns
            for index in range(1, len(request_order))
        )
    checks = {
        "exists": scan["exists"],
        "parse": scan["errors"] == 0,
        "count": len(records) == expected_count,
        "indices": integer_indices and sorted(indices) == list(range(expected_count)),
        "unique_indices": len(set(indices)) == len(indices),
        "trace_ids": all(isinstance(value, str) and value for value in trace_ids),
        "unique_trace_ids": len(set(trace_ids)) == len(trace_ids),
        "phase": {row.get("phase") for row in records} == {expected_phase},
        "system": {row.get("system") for row in records} == {expected_system},
        "build": {row.get("build_id") for row in records} == {expected_build},
        "schema": all(
            row.get("schema_version") == "table5_track_c_request_v0.3.0"
            for row in records
        ),
        "timings": all(timings_consistent(row) for row in records),
        "absolute_schedule": schedule_exact,
        "outcomes": all(
            isinstance(row.get("success"), bool)
            and isinstance(row.get("timeout"), bool)
            and (
                (
                    row["success"] is True
                    and row["timeout"] is False
                    and row.get("error") is None
                )
                or (
                    row["success"] is False
                    and isinstance(row.get("error"), str)
                    and bool(row["error"])
                )
            )
            for row in records
        ),
        "harness_retries": all(row.get("harness_retries") == 0 for row in records),
        "native_retry_field": all("system_internal_retries" in row for row in records),
        "call_and_retry_observability": request_observability_contract(
            records, stage_profiling=True
        ),
    }
    if expected_qps is not None:
        try:
            admitted = [int(row["admitted_at_ns"]) for row in request_order]
            admission_span_ns = max(admitted) - min(admitted)
            actual_admission_qps = (
                (len(admitted) - 1) / (admission_span_ns / 1_000_000_000)
                if len(admitted) > 1 and admission_span_ns > 0
                else None
            )
            ordered_lags = sorted(float(row["admission_lag_ms"]) for row in records)
            rank = (len(ordered_lags) - 1) * 0.99
            lower = int(rank)
            upper = min(lower + 1, len(ordered_lags) - 1)
            weight = rank - lower
            lag_p99 = (
                ordered_lags[lower] * (1.0 - weight) + ordered_lags[upper] * weight
            )
        except (KeyError, TypeError, ValueError):
            admitted = []
            actual_admission_qps = None
            lag_p99 = None
        checks.update(
            {
                "admission_order_exact": len(admitted) == expected_count
                and all(
                    admitted[index] <= admitted[index + 1]
                    for index in range(len(admitted) - 1)
                ),
                "actual_admission_qps_within_tolerance": (
                    actual_admission_qps is not None
                    and admission_qps_relative_tolerance is not None
                    and abs(actual_admission_qps - expected_qps) / expected_qps
                    <= admission_qps_relative_tolerance
                ),
                "admission_lag_p99_within_limit": (
                    lag_p99 is not None
                    and admission_lag_p99_max_ms is not None
                    and lag_p99 <= admission_lag_p99_max_ms
                ),
            }
        )
    return checks


def _late_outcome_checks(
    scan: dict[str, Any],
    *,
    path: Path,
    request_records: list[dict[str, Any]],
    receipt: dict[str, Any],
    expected_system: str,
    expected_build: str,
) -> dict[str, bool]:
    """Independently prove timeout-to-drained-worker correspondence."""

    records = scan["records"]
    timed_out = [row for row in request_records if row.get("timeout") is True]

    def key(row: dict[str, Any]) -> tuple[str, int] | None:
        phase = row.get("phase")
        index = row.get("request_index")
        if (
            not isinstance(phase, str)
            or not phase
            or not isinstance(index, int)
            or isinstance(index, bool)
        ):
            return None
        return phase, index

    timeout_by_key = {key(row): row for row in timed_out}
    late_by_key = {key(row): row for row in records}
    identity = True
    timings = True
    outcomes = True
    post_timeout_values: list[float] = []
    for row in records:
        request = timeout_by_key.get(key(row))
        if request is None:
            identity = timings = False
            continue
        identity = identity and (
            row.get("system") == request.get("system") == expected_system
            and row.get("build_id") == request.get("build_id") == expected_build
            and row.get("trace_id") == request.get("trace_id")
            and all(
                row.get(field) == request.get(field)
                for field in ("sample_id", "question_id", "event_id")
                if field in request
            )
        )
        try:
            client_timeout = int(row["client_timed_out_at_ns"])
            worker_complete = int(row["worker_completed_at_ns"])
            post_timeout_ms = float(row["post_timeout_work_ms"])
            post_timeout_values.append(post_timeout_ms)
            timings = timings and (
                client_timeout == int(request["completed_at_ns"])
                and worker_complete >= client_timeout
                and abs(
                    post_timeout_ms - (worker_complete - client_timeout) / 1_000_000
                )
                <= 1e-6
            )
        except (KeyError, TypeError, ValueError):
            timings = False
        final_success = row.get("final_success")
        final_error = row.get("final_error")
        outcomes = outcomes and (
            isinstance(final_success, bool)
            and (
                (final_success is True and final_error is None)
                or (
                    final_success is False
                    and isinstance(final_error, str)
                    and bool(final_error)
                )
            )
            and isinstance(row.get("receipt"), dict)
        )

    phases: dict[str, int] = {}
    for row in records:
        phase = str(row.get("phase"))
        phases[phase] = phases.get(phase, 0) + 1
    expected_summary = {
        "schema_version": "table5_track_c_late_outcome_summary_v0.1.0",
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "records": len(records),
        "by_phase": dict(sorted(phases.items())),
        "final_success": sum(row.get("final_success") is True for row in records),
        "final_failed": sum(row.get("final_success") is False for row in records),
        "post_timeout_work_ms": {
            "total": sum(post_timeout_values),
            "max": max(post_timeout_values, default=None),
        },
    }
    return {
        "exists": scan["exists"],
        "parse": scan["errors"] == 0,
        "schema": all(
            row.get("schema_version") == "table5_track_c_late_outcome_v0.1.0"
            for row in records
        ),
        "one_per_timeout": len(records) == len(timed_out)
        and None not in timeout_by_key
        and None not in late_by_key
        and len(timeout_by_key) == len(timed_out)
        and len(late_by_key) == len(records)
        and set(late_by_key) == set(timeout_by_key),
        "identity": identity,
        "timings": timings,
        "outcomes": outcomes,
        "summary": receipt.get("late_outcome_summary") == expected_summary,
    }


def _quality_point(
    root: Path,
    schedule: dict[str, Any],
    system: str,
    round_name: str,
) -> dict[str, Any]:
    output = root / "quality" / system / round_name
    receipt_path = output / "quality_receipt.json"
    build_id = f"tc5v2_{system}_search_{round_name}"
    result: dict[str, Any] = {
        "path": str(receipt_path.resolve()),
        "valid": False,
        "checks": {},
    }
    if not receipt_path.is_file():
        result["status"] = "MISSING"
        return result
    try:
        receipt = _load_json(receipt_path)
        answers_path = output / "answers.jsonl"
        judges_path = output / "judges.jsonl"
        summary_path = output / "quality_summary.json"
        answers = _jsonl(answers_path)
        judges = _jsonl(judges_path)
        summary = _load_json(summary_path) if summary_path.is_file() else {}
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "UNREADABLE"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def exact_records(scan: dict[str, Any], schema: str) -> bool:
        records = scan["records"]
        indices = [row.get("request_index") for row in records]
        return (
            scan["exists"]
            and scan["errors"] == 0
            and len(records) == 1_787
            and sorted(indices) == list(range(1_787))
            and len(set(indices)) == len(indices)
            and {row.get("system") for row in records} == {system}
            and {row.get("build_id") for row in records} == {build_id}
            and all(row.get("schema_version") == schema for row in records)
        )

    formal_path = root / "runs" / system / "search" / round_name / "formal.jsonl"

    def score(records: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [row for row in records if row.get("success") is True]
        correct = sum(row.get("correct") is True for row in successful)
        return {
            "total": len(records),
            "evaluated": len(successful),
            "correct": correct,
            "score": correct / len(records) if records else None,
            "conditional_score": correct / len(successful) if successful else None,
            "coverage": len(successful) / len(records) if records else 0.0,
        }

    judges_by_category: dict[str, list[dict[str, Any]]] = {}
    for row in judges["records"]:
        judges_by_category.setdefault(str(row.get("category")), []).append(row)
    context_tokens = [
        int(row["context_tokens"])
        for row in answers["records"]
        if row.get("context_tokens") is not None
    ]
    context_tokens_total = sum(context_tokens)
    context_token_records = len(context_tokens)
    recomputed_summary = {
        "schema_version": "table5_track_c_quality_summary_v0.2.0",
        "answer_prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
        "answer_prompt_source": (
            "neutralized common-context adaptation of MemOS 2.0.30 "
            "evaluation/scripts/locomo/prompts.py"
        ),
        "overall": score(judges["records"]),
        "by_category": {
            category: score(records)
            for category, records in sorted(judges_by_category.items())
        },
        "answer_generation_success": sum(
            row.get("success") is True for row in answers["records"]
        )
        / len(answers["records"])
        if answers["records"]
        else 0.0,
        "context_tokens_total": context_tokens_total,
        "context_token_records": context_token_records,
        "context_token_coverage": context_token_records / len(answers["records"])
        if answers["records"]
        else 0.0,
        "mean_context_tokens": context_tokens_total / context_token_records
        if context_token_records
        else None,
    }
    answer_by_index = {int(row["request_index"]): row for row in answers["records"]}
    judge_by_index = {int(row["request_index"]): row for row in judges["records"]}

    def non_empty(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    answer_outcomes = all(
        isinstance(row.get("retrieval_success"), bool)
        and non_empty(row.get("sample_id"))
        and non_empty(str(row.get("question_id") or ""))
        and (
            row.get("success") is True
            and row.get("retrieval_success") is True
            and non_empty(row.get("generated_answer"))
            and isinstance(row.get("context_count"), int)
            and row["context_count"] >= 0
            and isinstance(row.get("context_tokens"), int)
            and row["context_tokens"] >= 0
            and row.get("error") is None
            or row.get("success") is False
            and non_empty(row.get("error"))
        )
        for row in answers["records"]
    )
    judge_outcomes = all(
        isinstance(row.get("answer_success"), bool)
        and non_empty(row.get("sample_id"))
        and non_empty(str(row.get("question_id") or ""))
        and (
            row.get("success") is True
            and row.get("answer_success") is True
            and row.get("label") in {"CORRECT", "WRONG"}
            and row.get("correct") is (row.get("label") == "CORRECT")
            and row.get("error") is None
            or row.get("success") is False
            and row.get("correct") is False
            and row.get("label") == "WRONG"
            and non_empty(row.get("error"))
        )
        for row in judges["records"]
    )
    identity_linkage = answer_by_index.keys() == judge_by_index.keys() and all(
        judge_by_index[index].get(field) == answer.get(field)
        for index, answer in answer_by_index.items()
        for field in ("system", "build_id", "sample_id", "question_id", "category")
    )
    success_linkage = identity_linkage and all(
        judge_by_index[index].get("answer_success") is (answer.get("success") is True)
        for index, answer in answer_by_index.items()
    )
    checks = {
        "receipt_complete": receipt.get("status") == "complete",
        "receipt_schema": receipt.get("schema_version")
        == "table5_track_c_quality_run_v0.2.0",
        "system": receipt.get("system") == system,
        "build_id": receipt.get("build_id") == build_id,
        "formal_records": receipt.get("formal_records") == 1_787,
        "model": receipt.get("model") == "Qwen/Qwen3-32B",
        "endpoint": receipt.get("endpoint") == "http://127.0.0.1:8000/v1",
        "workers": receipt.get("workers") == 32,
        "answer_prompt": receipt.get("answer_prompt_sha256")
        == hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
        "judge_prompt": receipt.get("judge_prompt_sha256")
        == hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
        "benchmark_code": receipt.get("benchmark_code_sha256")
        == schedule.get("benchmark_code_sha256"),
        "formal_input_hash": receipt.get("formal_sha256") == _sha256(formal_path),
        "answers_exact": exact_records(answers, "table5_track_c_answer_v0.2.0"),
        "judges_exact": exact_records(judges, "table5_track_c_judge_v0.2.0"),
        "answer_outcome_contract": answer_outcomes,
        "judge_outcome_contract": judge_outcomes,
        "answer_judge_identity_linkage": identity_linkage,
        "answer_judge_success_linkage": success_linkage,
        "answers_hash": receipt.get("answers_sha256") == _sha256(answers_path),
        "judges_hash": receipt.get("judges_sha256") == _sha256(judges_path),
        "summary_hash": receipt.get("summary_sha256") == _sha256(summary_path),
        "summary_schema": summary.get("schema_version")
        == "table5_track_c_quality_summary_v0.2.0",
        "summary_system": summary.get("system") == system,
        "summary_build": summary.get("build_id") == build_id,
        "summary_total": (summary.get("overall") or {}).get("total") == 1_787,
        "summary_evidence_recomputed": all(
            summary.get(key) == value for key, value in recomputed_summary.items()
        ),
        "receipt_overall_matches_summary": receipt.get("overall")
        == summary.get("overall"),
        "summary_context_tokens": isinstance(summary.get("context_tokens_total"), int)
        and summary["context_tokens_total"] >= 0
        and isinstance(summary.get("context_token_records"), int)
        and 0 <= summary["context_token_records"] <= 1_787
        and summary.get("context_token_coverage")
        == summary["context_token_records"] / 1_787
        and (
            summary.get("mean_context_tokens")
            == summary["context_tokens_total"] / summary["context_token_records"]
            if summary["context_token_records"]
            else summary.get("mean_context_tokens") is None
        ),
    }
    result["checks"] = checks
    result["valid"] = all(checks.values())
    result["status"] = "COMPLETE_VALID" if result["valid"] else "COMPLETE_INVALID"
    result["receipt_sha256"] = _sha256(receipt_path)
    return result


def _expected_hashes(schedule: dict[str, Any]) -> tuple[dict[str, str], bool]:
    names = (
        "benchmark_code_sha256",
        "graphiti_adapter_sha256",
        "run_one_script_sha256",
        "formal_driver_script_sha256",
        "conformance_driver_script_sha256",
        "quality_driver_script_sha256",
        "postprocess_script_sha256",
        "execution_driver_script_sha256",
        "goal_audit_script_sha256",
        "protocol_receipt_sha256",
        "formal_schedule_sha256",
    )
    raw = {
        name: (
            schedule.get("_schedule_sha256")
            if name == "formal_schedule_sha256"
            else schedule.get(name)
        )
        for name in names
    }
    expected = {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(value, str) and len(value) == 64
    }
    return expected, len(expected) == len(raw) and bool(raw)


def _graphiti_adapter_sha256() -> str:
    return graphiti_adapter_sha256(REPOSITORY)


def _graphiti_conformance_checks(
    receipt: dict[str, Any], *, expected_adapter_sha: str, expected_commit: str
) -> dict[str, bool]:
    write = receipt.get("write_phase") or {}
    read = receipt.get("read_phase") or {}
    write_schema = write.get("schema_gate") or {}
    read_schema = read.get("schema_gate") or {}
    write_runtime = write.get("neo4j_runtime") or {}
    read_runtime = read.get("neo4j_runtime") or {}
    write_invocation = write_runtime.get("systemd_invocation_id")
    read_invocation = read_runtime.get("systemd_invocation_id")
    runtime_ids_valid = all(
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
        for value in (write_invocation, read_invocation)
    )
    write_searches = [
        value.get("result") or {}
        for value in write.get("search_results") or []
        if isinstance(value, dict)
    ]
    read_searches = [
        value.get("result") or {}
        for value in read.get("search_results") or []
        if isinstance(value, dict)
    ]
    additions = write.get("additions") or []
    fifo_tickets: dict[str, list[int]] = {}
    for addition in additions:
        if not isinstance(addition, dict):
            continue
        group_id = addition.get("write_group_id")
        ticket = addition.get("write_group_fifo_ticket")
        if isinstance(group_id, str) and isinstance(ticket, int):
            fifo_tickets.setdefault(group_id, []).append(ticket)

    def isolated(searches: list[dict[str, Any]]) -> bool:
        if len(searches) != 2:
            return False
        group_a = " ".join(str(value) for value in searches[0].get("contexts") or [])
        group_b = " ".join(str(value) for value in searches[1].get("contexts") or [])
        return (
            "Borealis" not in group_a
            and "Maple Works" not in group_a
            and "Asteria" not in group_b
            and "Cedar Labs" not in group_b
        )

    return {
        "status": receipt.get("status") == "complete",
        "schema": receipt.get("schema_version")
        == "graphiti_track_c_conformance_v0.3.0",
        "label": receipt.get("display_label") == "Graphiti (Zep OSS proxy)",
        "boundary": receipt.get("interpretation_boundary") == "not production Zep",
        "adapter_write": write_schema.get("adapter_sha256") == expected_adapter_sha,
        "adapter_read": read_schema.get("adapter_sha256") == expected_adapter_sha,
        "source_write": (write_schema.get("graphiti_source") or {}).get("commit")
        == expected_commit,
        "source_read": (read_schema.get("graphiti_source") or {}).get("commit")
        == expected_commit,
        "persistent_counts": bool(write.get("counts"))
        and write.get("counts") == read.get("counts"),
        "persistent_groups": bool(write.get("group_counts"))
        and write.get("group_counts") == read.get("group_counts"),
        "write_events_exact": write.get("events") == len(additions) == 4,
        "add_creation_contract": add_receipt_contract(
            [{"success": True, "receipt": value} for value in additions]
        ),
        "formal_write_fifo_contract": bool(additions)
        and all(
            isinstance(value, dict)
            and value.get("write_concurrency_policy")
            == "per_group_fifo_cross_group_parallel"
            and isinstance(value.get("write_group_fifo_wait_ms"), (int, float))
            and value.get("write_group_fifo_wait_ms") >= 0
            and bool(value.get("write_concurrency_provenance"))
            for value in additions
        )
        and fifo_tickets == {"conformance_a": [0, 1], "conformance_b": [0, 1]},
        "search_before_restart": len(write_searches) == 2
        and search_receipt_contract(
            [{"success": True, "receipt": value} for value in write_searches]
        )
        and all(int(value.get("result_count") or 0) > 0 for value in write_searches),
        "search_after_restart": len(read_searches) == 2
        and search_receipt_contract(
            [{"success": True, "receipt": value} for value in read_searches]
        )
        and all(int(value.get("result_count") or 0) > 0 for value in read_searches),
        "scope_isolation_before_restart": isolated(write_searches),
        "scope_isolation_after_restart": isolated(read_searches),
        "runtime_restart": runtime_ids_valid
        and write_invocation != read_invocation
        and isinstance(write_runtime.get("main_pid"), int)
        and write_runtime["main_pid"] > 0
        and isinstance(read_runtime.get("main_pid"), int)
        and read_runtime["main_pid"] > 0,
    }


def _generic_conformance_checks(receipt: dict[str, Any]) -> dict[str, bool]:
    """Recompute ordinary adapter conformance instead of trusting its status."""
    searches = receipt.get("searches") or []
    target_add = receipt.get("target_add") or {}
    scope_add = receipt.get("scope_control_add") or {}
    return {
        "status": receipt.get("status") == "passed",
        "events_exact": (receipt.get("build") or {}).get("events") == 58,
        "five_searches": len(searches) == 5,
        "search_receipt_contract": search_receipt_contract(
            [{"success": True, "receipt": value} for value in searches]
        ),
        "all_searches_nonempty": bool(searches)
        and all(int(value.get("result_count") or 0) > 0 for value in searches),
        "add_creation_receipt_contract": add_receipt_contract(
            [
                {"success": True, "receipt": target_add},
                {"success": True, "receipt": scope_add},
            ]
        ),
        "target_visible_on_add": target_add.get("visibility_probe") is True,
        "control_visible_on_add": scope_add.get("visibility_probe") is True,
        "restart_schema_gate": isinstance(receipt.get("restart_schema_gate"), dict),
        "restart_persistent": receipt.get("restart_visible") is True,
        "scope_isolated": receipt.get("scope_isolated") is True,
    }


def audit_conformance_receipt(
    schedule: dict[str, Any],
    protocol: dict[str, Any],
    root: Path,
    system: str,
) -> dict[str, Any]:
    """Recompute one conformance receipt before formal measurements start."""
    path = root / "conformance" / f"{system}.json"
    result: dict[str, Any] = {
        "system": system,
        "path": str(path.resolve()),
        "passed": False,
    }
    if system not in (protocol.get("systems") or []):
        result["error"] = "system is absent from the frozen protocol"
        return result
    if not path.is_file():
        result["error"] = "conformance receipt is absent"
        return result
    try:
        receipt = _load_json(path)
        expected_status = "complete" if system == "graphiti_zep_oss_proxy" else "passed"
        binding_checks = {
            "status": receipt.get("status") == expected_status,
            "benchmark_code": (receipt.get("execution") or {}).get(
                "benchmark_code_sha256"
            )
            == schedule.get("benchmark_code_sha256"),
            "formal_schedule": (receipt.get("execution") or {}).get(
                "formal_schedule_sha256"
            )
            == schedule.get("_schedule_sha256"),
            "protocol_receipt": (receipt.get("execution") or {}).get(
                "protocol_receipt_sha256"
            )
            == schedule.get("protocol_receipt_sha256"),
        }
        if system == "graphiti_zep_oss_proxy":
            semantic_checks = _graphiti_conformance_checks(
                receipt,
                expected_adapter_sha=str(schedule.get("graphiti_adapter_sha256") or ""),
                expected_commit=str(
                    ((protocol.get("source_identity") or {}).get(system) or {}).get(
                        "source_commit"
                    )
                    or ""
                ),
            )
        else:
            semantic_checks = _generic_conformance_checks(receipt)
        result["binding_checks"] = binding_checks
        result["checks"] = semantic_checks
        result["sha256"] = _sha256(path)
        result["passed"] = all(binding_checks.values()) and all(
            semantic_checks.values()
        )
    except (OSError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _source_identity_matches(
    receipt: dict[str, Any], system: str, expected: dict[str, Any]
) -> bool:
    if system == "graphiti_zep_oss_proxy":
        observed = {
            "schema_version": receipt.get("schema_version"),
            "status": receipt.get("status"),
            "installed": receipt.get("installed"),
            "display_label": receipt.get("display_label"),
            "interpretation_boundary": receipt.get("interpretation_boundary"),
            "source_commit": (receipt.get("source") or {}).get("commit"),
            "package_version": (receipt.get("environment") or {}).get(
                "graphiti_core_version"
            ),
            "backend": receipt.get("backend"),
            "neo4j_version": (receipt.get("backend_identity") or {}).get(
                "neo4j_version"
            ),
            "cypher_shell_version": (receipt.get("backend_identity") or {}).get(
                "cypher_shell_version"
            ),
        }
    else:
        observed = {key: receipt.get(key) for key in expected}
    return observed == expected and receipt.get("system") == system


def _command_bytes(
    command: list[str], *, cwd: Path | None = None
) -> tuple[int, bytes, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        return -1, b"", f"{type(exc).__name__}: {exc}"
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr.decode(errors="replace")[-1_000:],
    )


def _git_runtime_observation(
    source: Path, expected_commit: str, *, require_tracked_clean: bool
) -> dict[str, Any]:
    head_rc, head_bytes, head_error = _command_bytes(
        ["git", "-C", str(source), "rev-parse", "HEAD"]
    )
    status_rc, status_bytes, status_error = _command_bytes(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]
    )
    head = head_bytes.decode(errors="replace").strip()
    tracked_dirty_entries = len(status_bytes.decode(errors="replace").splitlines())
    checks = {
        "source_directory": source.is_dir(),
        "git_head_readable": head_rc == 0,
        "git_head": head_rc == 0 and head == expected_commit,
        "git_status_readable": status_rc == 0,
        "tracked_source_clean": status_rc == 0
        and (not require_tracked_clean or tracked_dirty_entries == 0),
    }
    return {
        "checks": checks,
        "observed": {
            "source": str(source),
            "head": head or None,
            "tracked_dirty_entries": tracked_dirty_entries,
        },
        "errors": [
            value
            for returncode, value in (
                (head_rc, head_error),
                (status_rc, status_error),
            )
            if returncode != 0 and value
        ],
        "valid": all(checks.values()),
    }


def _python_distribution_version(python: Path, distribution: str) -> tuple[int, str]:
    code = f"import importlib.metadata as m; print(m.version({distribution!r}))"
    returncode, stdout, _ = _command_bytes([str(python), "-c", code])
    return returncode, stdout.decode(errors="replace").strip()


def _python_distributions(python: Path) -> tuple[int, list[dict[str, str]]]:
    code = """
import importlib.metadata
import json

items = sorted({
    distribution.metadata["Name"]: distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}.items())
print(json.dumps([{"name": name, "version": version} for name, version in items]))
"""
    returncode, stdout, _ = _command_bytes([str(python), "-c", code])
    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        result = []
    return returncode, result if isinstance(result, list) else []


def _postgres_runtime_observation(receipt: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(receipt.get("source_path") or ""))
    expected_commit = str(receipt.get("repository_base_commit") or "")
    git = _git_runtime_observation(source, expected_commit, require_tracked_clean=False)
    python = Path(str(receipt.get("python") or ""))
    psql = shutil.which("psql") or "psql"
    pg_config = Path("/usr/lib/postgresql/16/bin/pg_config")
    if not pg_config.is_file():
        pg_config = Path(shutil.which("pg_config") or "")
    server_rc, server_stdout, server_error = _command_bytes(
        [
            psql,
            "-h",
            "127.0.0.1",
            "-p",
            "55432",
            "-U",
            "hza214",
            "-d",
            "postgres",
            "-At",
            "-F",
            "\t",
            "-c",
            ("select current_setting('server_version'), current_setting('block_size')"),
        ]
    )
    server_fields = server_stdout.decode(errors="replace").strip().split("\t")
    extension_rc, extension_stdout, extension_error = _command_bytes(
        [
            psql,
            "-h",
            "127.0.0.1",
            "-p",
            "55432",
            "-U",
            "hza214",
            "-d",
            "postgres",
            "-At",
            "-F",
            "\t",
            "-c",
            (
                "select extname, extversion from pg_extension "
                "where extname in ('vector','graph_store_am','tjs_pg') "
                "order by extname"
            ),
        ]
    )
    observed_extensions: dict[str, str] = {}
    for line in extension_stdout.decode(errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) == 2:
            observed_extensions[fields[0]] = fields[1]
    pkglib_rc, pkglib_stdout, pkglib_error = _command_bytes(
        [str(pg_config), "--pkglibdir"]
    )
    pkglibdir = Path(pkglib_stdout.decode(errors="replace").strip())
    expected_binary_hashes = receipt.get("extension_binary_sha256") or {}
    observed_binary_hashes = {
        name: _sha256(pkglibdir / name) for name in expected_binary_hashes
    }
    checks = {
        **git["checks"],
        "python_executable": python.is_file() and os.access(python, os.X_OK),
        "postgres_reachable": server_rc == 0 and len(server_fields) == 2,
        "postgres_version": server_rc == 0
        and server_fields[0].split()[0] == str(receipt.get("postgres")),
        "postgres_block_size": server_rc == 0
        and len(server_fields) == 2
        and server_fields[1] == str(receipt.get("postgres_block_size_bytes")),
        "extensions_readable": extension_rc == 0,
        "extensions": observed_extensions == (receipt.get("extensions") or {}),
        "pg_config_readable": pkglib_rc == 0 and pkglibdir.is_dir(),
        "extension_binaries": observed_binary_hashes == expected_binary_hashes,
    }
    return {
        "checks": checks,
        "observed": {
            **git["observed"],
            "postgres_version": server_fields[0] if server_fields else None,
            "postgres_block_size_bytes": (
                server_fields[1] if len(server_fields) == 2 else None
            ),
            "extensions": observed_extensions,
            "pkglibdir": str(pkglibdir),
            "extension_binary_sha256": observed_binary_hashes,
        },
        "errors": [
            *git["errors"],
            *(
                value
                for returncode, value in (
                    (server_rc, server_error),
                    (extension_rc, extension_error),
                    (pkglib_rc, pkglib_error),
                )
                if returncode != 0 and value
            ),
        ],
        "valid": all(checks.values()),
    }


def _external_python_runtime_observation(
    receipt: dict[str, Any], system: str
) -> dict[str, Any]:
    source = Path(str(receipt.get("source_path") or ""))
    expected_commit = str(receipt.get("source_commit") or "")
    git = _git_runtime_observation(source, expected_commit, require_tracked_clean=True)
    python = Path(str(receipt.get("python") or ""))
    manifest = Path(str(receipt.get("dependency_manifest") or ""))
    uv = Path("/localhome/hza214/.local/bin/uv")
    freeze_rc, freeze_stdout, freeze_error = _command_bytes(
        [str(uv), "pip", "freeze", "--python", str(python)]
    )
    manifest_bytes = manifest.read_bytes() if manifest.is_file() else b""
    distribution = {
        "mem0": "mem0ai",
        "memos": "MemoryOS",
        "cognee": "cognee",
    }[system]
    version_rc, observed_version = _python_distribution_version(python, distribution)
    expected_manifest_sha = str(receipt.get("dependency_manifest_sha256") or "")
    checks = {
        **git["checks"],
        "python_executable": python.is_file() and os.access(python, os.X_OK),
        "dependency_manifest": manifest.is_file(),
        "dependency_manifest_sha256": _sha256(manifest) == expected_manifest_sha,
        "environment_freeze_readable": freeze_rc == 0,
        "environment_freeze_exact": freeze_rc == 0 and freeze_stdout == manifest_bytes,
        "package_version_readable": version_rc == 0,
        "package_version": observed_version == str(receipt.get("package_version")),
    }
    observed: dict[str, Any] = {
        **git["observed"],
        "python": str(python),
        "dependency_manifest": str(manifest),
        "dependency_manifest_sha256": _sha256(manifest),
        "environment_freeze_sha256": hashlib.sha256(freeze_stdout).hexdigest()
        if freeze_rc == 0
        else None,
        "package_distribution": distribution,
        "package_version": observed_version or None,
    }
    if system == "memos":
        tarball = Path(
            "/localhome/hza214/agent-memory-table5/downloads/"
            "neo4j-community-5.26.6-unix.tar.gz"
        )
        neo4j_home = Path(
            "/localhome/hza214/agent-memory-table5/runtime/neo4j-community-5.26.6"
        )
        neo4j_rc, neo4j_stdout, neo4j_error = _command_bytes(
            [str(neo4j_home / "bin/neo4j"), "--version"]
        )
        cypher_rc, cypher_stdout, cypher_error = _command_bytes(
            [str(neo4j_home / "bin/cypher-shell"), "--version"]
        )
        neo4j_version = neo4j_stdout.decode(errors="replace").strip()
        cypher_version = cypher_stdout.decode(errors="replace").strip()
        expected_neo4j = str(receipt.get("neo4j_version") or "").removesuffix(
            "-community"
        )
        checks.update(
            {
                "neo4j_tarball": tarball.is_file(),
                "neo4j_tarball_sha256": _sha256(tarball)
                == receipt.get("neo4j_tarball_sha256"),
                "neo4j_version": neo4j_rc == 0 and neo4j_version == expected_neo4j,
                "cypher_shell_version": cypher_rc == 0
                and cypher_version == f"Cypher-Shell {expected_neo4j}",
            }
        )
        observed.update(
            {
                "neo4j_tarball": str(tarball),
                "neo4j_tarball_sha256": _sha256(tarball),
                "neo4j_version": neo4j_version or None,
                "cypher_shell_version": cypher_version or None,
            }
        )
        git["errors"].extend(
            value
            for returncode, value in (
                (neo4j_rc, neo4j_error),
                (cypher_rc, cypher_error),
            )
            if returncode != 0 and value
        )
    return {
        "checks": checks,
        "observed": observed,
        "errors": [
            *git["errors"],
            *([freeze_error] if freeze_rc != 0 and freeze_error else []),
        ],
        "valid": all(checks.values()),
    }


def _graphiti_runtime_observation(receipt: dict[str, Any]) -> dict[str, Any]:
    source_receipt = receipt.get("source") or {}
    environment = receipt.get("environment") or {}
    backend = receipt.get("backend_identity") or {}
    source = Path(str(source_receipt.get("root") or ""))
    git = _git_runtime_observation(
        source,
        str(source_receipt.get("commit") or ""),
        require_tracked_clean=True,
    )
    venv = Path(str(environment.get("venv") or ""))
    python = Path(str(environment.get("python_executable") or ""))
    version_rc, version = _python_distribution_version(python, "graphiti-core")
    distributions_rc, distributions = _python_distributions(python)
    module_code = (
        "import pathlib, graphiti_core; "
        "print(pathlib.Path(graphiti_core.__file__).resolve())"
    )
    module_rc, module_stdout, module_error = _command_bytes(
        [str(python), "-c", module_code]
    )
    module = module_stdout.decode(errors="replace").strip()
    neo4j_home = Path(str(backend.get("home") or ""))
    neo4j_rc, neo4j_stdout, neo4j_error = _command_bytes(
        [str(neo4j_home / "bin/neo4j"), "--version"]
    )
    cypher_rc, cypher_stdout, cypher_error = _command_bytes(
        [str(neo4j_home / "bin/cypher-shell"), "--version"]
    )
    checks = {
        **git["checks"],
        "pyproject_sha256": _sha256(source / "pyproject.toml")
        == source_receipt.get("pyproject_sha256"),
        "uv_lock_sha256": _sha256(source / "uv.lock")
        == source_receipt.get("uv_lock_sha256"),
        "python_executable": python.is_file() and os.access(python, os.X_OK),
        # A normal venv's ``bin/python`` is itself a symlink to the system
        # interpreter. Resolve the directory aliases (/localhome versus
        # /local-scratch/localhome), but do not follow that final symlink out
        # of the recorded environment.
        "python_in_recorded_venv": _path_entry_is_within(python, venv),
        "python_resolved": str(python.resolve())
        == environment.get("python_executable_resolved"),
        "package_version": version_rc == 0
        and version == environment.get("graphiti_core_version"),
        "package_module": module_rc == 0
        and module == environment.get("graphiti_core_module")
        and _resolved_path_is_within(Path(module), source),
        "environment_distributions": distributions_rc == 0
        and distributions == environment.get("distributions"),
        "neo4j_version": neo4j_rc == 0
        and neo4j_stdout.decode(errors="replace").strip()
        == backend.get("neo4j_version"),
        "cypher_shell_version": cypher_rc == 0
        and cypher_stdout.decode(errors="replace").strip()
        == backend.get("cypher_shell_version"),
    }
    return {
        "checks": checks,
        "observed": {
            **git["observed"],
            "venv": str(venv),
            "python": str(python),
            "python_resolved": str(python.resolve()),
            "graphiti_core_version": version or None,
            "graphiti_core_module": module or None,
            "distribution_count": len(distributions),
            "neo4j_home": str(neo4j_home),
            "neo4j_version": neo4j_stdout.decode(errors="replace").strip() or None,
            "cypher_shell_version": cypher_stdout.decode(errors="replace").strip()
            or None,
        },
        "errors": [
            *git["errors"],
            *(
                value
                for returncode, value in (
                    (module_rc, module_error),
                    (neo4j_rc, neo4j_error),
                    (cypher_rc, cypher_error),
                )
                if returncode != 0 and value
            ),
        ],
        "valid": all(checks.values()),
    }


def _source_runtime_observation(receipt: dict[str, Any], system: str) -> dict[str, Any]:
    if system == "tridb_gem":
        return _postgres_runtime_observation(receipt)
    if system in {"mem0", "memos", "cognee"}:
        return _external_python_runtime_observation(receipt, system)
    if system == "graphiti_zep_oss_proxy":
        return _graphiti_runtime_observation(receipt)
    return {
        "checks": {"known_system": False},
        "observed": {"system": system},
        "errors": [f"unsupported system: {system}"],
        "valid": False,
    }


def audit_source_receipt(
    schedule: dict[str, Any], protocol: dict[str, Any], system: str
) -> dict[str, Any]:
    source_receipts = protocol.get("source_receipts") or {}
    expected_identity = (protocol.get("source_identity") or {}).get(system) or {}
    relative = source_receipts.get(system)
    path = REPOSITORY / str(relative or "")
    expected_sha = (schedule.get("source_receipt_sha256") or {}).get(system)
    value: dict[str, Any] = {
        "system": system,
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": _sha256(path),
        "expected_sha256": expected_sha,
        "identity_matches": False,
        "sha256_matches": False,
        "runtime": {"valid": False},
        "valid": False,
    }
    if not path.is_file():
        return value
    try:
        receipt = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        value["error"] = f"{type(exc).__name__}: {exc}"
        return value
    value["status"] = receipt.get("status")
    value["identity_matches"] = _source_identity_matches(
        receipt, system, expected_identity
    )
    value["sha256_matches"] = value["sha256"] == expected_sha
    value["runtime"] = _source_runtime_observation(receipt, system)
    value["valid"] = (
        value["identity_matches"]
        and value["sha256_matches"]
        and value["runtime"]["valid"]
    )
    return value


def _completion_model_receipt_valid(
    model: dict[str, Any], schedule: dict[str, Any]
) -> bool:
    answer = (model.get("answer") or {}).get("data") or []
    embedding = (model.get("embedding") or {}).get("data") or []
    checks = model.get("checks") or {}
    required_checks = {
        "answer_identity_exact",
        "embedding_identity_exact",
        "embedding_dimension_1024",
        "thinking_disabled",
        "deterministic_chat_probe",
        "answer_revision_exact",
        "embedding_revision_exact",
        "runtime_versions_exact",
        "offline_launch_exact",
        "local_snapshots_present",
    }
    vector_sha = model.get("embedding_vector_sha256")
    runtime_versions = model.get("runtime_versions") or {}
    snapshots = model.get("local_snapshots") or {}
    commands = model.get("model_service_commands") or {}
    environments = model.get("model_service_environments") or {}
    answer_revision = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
    embedding_revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    return (
        model.get("schema_version") == "table5_track_c_completion_model_receipt_v0.2.0"
        and model.get("status") == "passed"
        and model.get("schedule_sha256") == schedule.get("_schedule_sha256")
        and model.get("protocol_sha256") == schedule.get("protocol_receipt_sha256")
        and model.get("answer_endpoint") == "http://127.0.0.1:8000/v1"
        and model.get("embedding_endpoint") == "http://127.0.0.1:8001/v1"
        and model.get("answer_thinking") is False
        and model.get("answer_revision") == answer_revision
        and model.get("embedding_revision") == embedding_revision
        and runtime_versions
        == {
            "vllm": "0.15.0",
            "transformers": "4.57.0",
            "huggingface_hub": "0.36.2",
            "torch": "2.9.1",
        }
        and (snapshots.get("answer") or {}).get("revision") == answer_revision
        and (snapshots.get("answer") or {}).get("exists") is True
        and (snapshots.get("embedding") or {}).get("revision") == embedding_revision
        and (snapshots.get("embedding") or {}).get("exists") is True
        and commands.get("answer", "").count(answer_revision) >= 2
        and commands.get("embedding", "").count(embedding_revision) >= 2
        and all(
            token in str(environments.get(service) or "")
            for service in ("answer", "embedding")
            for token in (
                "HF_HUB_OFFLINE=1",
                "TRANSFORMERS_OFFLINE=1",
                "HF_HOME=/localhome/hza214/.cache/huggingface",
            )
        )
        and len(answer) == 1
        and answer[0].get("id") == "Qwen/Qwen3-32B"
        and answer[0].get("root") == "Qwen/Qwen3-32B-FP8"
        and answer[0].get("max_model_len") == 32768
        and len(embedding) == 1
        and embedding[0].get("id") == "Qwen/Qwen3-Embedding-0.6B"
        and embedding[0].get("root") == "Qwen/Qwen3-Embedding-0.6B"
        and embedding[0].get("max_model_len") == 8192
        and model.get("embedding_dimension") == 1024
        and isinstance(vector_sha, str)
        and len(vector_sha) == 64
        and model.get("chat_content") == "TOKEN_OK"
        and model.get("chat_reasoning_content") in (None, "")
        and isinstance(checks, dict)
        and set(checks) == required_checks
        and all(value is True for value in checks.values())
    )


def _completion_hardware_receipt_valid(
    hardware: dict[str, Any], schedule: dict[str, Any]
) -> bool:
    cgroup = hardware.get("runner_cgroup") or {}
    gpu_launch = hardware.get("gpu_launch_gate") or {}
    model_bindings = hardware.get("model_service_bindings") or {}
    resource_claim = hardware.get("resource_control_claim") or {}
    return (
        hardware.get("schema_version")
        == "table5_track_c_completion_hardware_receipt_v0.3.0"
        and hardware.get("status") == "complete"
        and hardware.get("schedule_sha256") == schedule.get("_schedule_sha256")
        and hardware.get("protocol_sha256") == schedule.get("protocol_receipt_sha256")
        and hardware.get("paper_h800_match") is False
        and hardware.get("gx10_signoff") is False
        and hardware.get("claim") == "same-host controlled comparison only"
        and all(
            isinstance(hardware.get(field), str) and bool(hardware[field].strip())
            for field in ("platform", "machine", "cpu", "memory", "gpus")
        )
        and isinstance(hardware.get("runner_cpu_affinity"), list)
        and bool(hardware["runner_cpu_affinity"])
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in hardware["runner_cpu_affinity"]
        )
        and all(
            isinstance(cgroup.get(field), str) and bool(cgroup[field].strip())
            for field in (
                "membership",
                "path",
                "cpuset_cpus_effective",
                "cpuset_cpus_effective_source",
                "memory_max",
                "memory_max_source",
            )
        )
        and gpu_launch.get("target_physical_index") == 0
        and str(gpu_launch.get("target_uuid") or "").startswith("GPU-")
        and isinstance(gpu_launch.get("checked_at"), str)
        and bool(gpu_launch["checked_at"].strip())
        and gpu_launch.get("prelaunch_compute_processes") == []
        and gpu_launch.get("prelaunch_empty") is True
        and gpu_launch.get("policy")
        == (
            "refuse model launch when physical GPU 0 already has any compute "
            "process; never terminate external processes"
        )
        and model_bindings.get("answer_unit")
        == "tridb-table5-completion-v2-answer.service"
        and model_bindings.get("embedding_unit")
        == "tridb-table5-completion-v2-embedding.service"
        and "CUDA_VISIBLE_DEVICES=0"
        in str(model_bindings.get("answer_environment") or "")
        and "CUDA_VISIBLE_DEVICES=0"
        in str(model_bindings.get("embedding_environment") or "")
        and all(
            token in str(model_bindings.get(f"{service}_environment") or "")
            for service in ("answer", "embedding")
            for token in (
                "HF_HUB_OFFLINE=1",
                "TRANSFORMERS_OFFLINE=1",
                "HF_HOME=/localhome/hza214/.cache/huggingface",
            )
        )
        and str(model_bindings.get("answer_command") or "").count(
            "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
        )
        >= 2
        and str(model_bindings.get("embedding_command") or "").count(
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
        )
        >= 2
        and resource_claim.get("cpu_affinity_fixed") is False
        and resource_claim.get("ram_cap_fixed") is False
        and resource_claim.get("model_gpu_binding_fixed") is True
        and resource_claim.get("gpu_exclusive_mode") is False
        and isinstance(resource_claim.get("interpretation"), str)
        and bool(resource_claim["interpretation"].strip())
    )


def _tridb_invariant_gate(
    root: Path,
    schedule_path: Path,
    schedule: dict[str, Any],
    protocol_path: Path,
) -> dict[str, Any]:
    path = root / "tridb_invariant_receipt.json"
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "valid": False,
        "checks": {},
    }
    if not path.is_file():
        return result
    try:
        receipt = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return result
    conformance_path = root / "conformance/tridb_gem.json"
    source_manifest = REPOSITORY / str(
        (_load_json(protocol_path).get("source_receipts") or {}).get("tridb_gem") or ""
    )
    receipt_checks = receipt.get("checks") or {}
    claims = receipt.get("claims") or {}
    file_hashes = receipt.get("source_files_sha256") or {}
    source_runtime = receipt.get("source_runtime_audit") or {}
    source_files_valid = bool(file_hashes) and all(
        isinstance(relative, str)
        and isinstance(digest, str)
        and len(digest) == 64
        and _sha256(REPOSITORY / relative) == digest
        for relative, digest in file_hashes.items()
    )
    checks = {
        "schema": receipt.get("schema_version")
        == "table5_track_c_tridb_invariant_v0.1.0",
        "status": receipt.get("status") == "passed",
        "schedule": receipt.get("schedule") == str(schedule_path.resolve()),
        "schedule_sha": receipt.get("schedule_sha256") == _sha256(schedule_path),
        "protocol": receipt.get("protocol") == str(protocol_path.resolve()),
        "protocol_sha": receipt.get("protocol_sha256") == _sha256(protocol_path),
        "benchmark_code": receipt.get("benchmark_code_sha256")
        == schedule.get("benchmark_code_sha256")
        == benchmark_code_sha256(),
        "conformance_path": receipt.get("conformance_receipt")
        == str(conformance_path.resolve()),
        "conformance_sha": receipt.get("conformance_receipt_sha256")
        == _sha256(conformance_path),
        "source_manifest_path": receipt.get("source_manifest")
        == str(source_manifest.resolve()),
        "source_manifest_sha": receipt.get("source_manifest_sha256")
        == _sha256(source_manifest)
        == (schedule.get("source_receipt_sha256") or {}).get("tridb_gem"),
        "source_files": source_files_valid,
        "source_runtime": source_runtime.get("valid") is True
        and source_runtime.get("returncode") == 0,
        "invariant_checks": set(receipt_checks) == set(INVARIANT_CHECK_NAMES)
        and all(receipt_checks.get(name) is True for name in INVARIANT_CHECK_NAMES),
        "claims": claims
        == {
            "tr1_open_next_close": True,
            "native_graph": True,
            "same_postgres_process": True,
            "one_postgres_wal": True,
            "full_intermediate_materialization": False,
            "paper_h800_match": False,
            "gx10_signoff": False,
        },
        "interpretation_boundary": "not an H800 numerical reproduction"
        in str(receipt.get("interpretation") or "")
        and "not a GX10 build/sign-off" in str(receipt.get("interpretation") or ""),
    }
    result["checks"] = checks
    result["valid"] = all(checks.values())
    result["sha256"] = _sha256(path)
    return result


def _postprocess_gate(
    root: Path,
    schedule_path: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    path = root / "postprocess_receipt.json"
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "valid": False,
        "checks": {},
    }
    if not path.is_file():
        return result
    try:
        receipt = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return result
    expected_outputs = set(protocol["required_deliverables"]) | {
        "aggregates/request_metrics.csv",
        "aggregates/latency_decomposition.csv",
        "figures/search_latency_box.pdf",
    }
    outputs = receipt.get("outputs") or {}
    output_hashes_valid = set(outputs) == expected_outputs and all(
        isinstance(digest, str)
        and len(digest) == 64
        and _sha256(root / relative) == digest
        for relative, digest in outputs.items()
    )
    input_audit = receipt.get("input_goal_audit") or {}
    coverage = receipt.get("coverage_checks") or {}
    checks = {
        "schema": receipt.get("schema_version")
        == "table5_track_c_completion_postprocess_v0.2.0",
        "status": receipt.get("status") == "complete",
        "schedule": receipt.get("schedule") == str(schedule_path.resolve()),
        "schedule_sha": receipt.get("schedule_sha256") == _sha256(schedule_path),
        "protocol_sha": receipt.get("protocol_sha256") == _sha256(protocol_path),
        "input_formal": input_audit.get("complete_valid_runs") == 45,
        "input_quality": input_audit.get("quality_complete") is True,
        "input_prerequisites": input_audit.get("prerequisite_complete") is True,
        "coverage": set(coverage) == set(POSTPROCESS_COVERAGE_CHECK_NAMES)
        and all(
            coverage.get(name) is True for name in POSTPROCESS_COVERAGE_CHECK_NAMES
        ),
        "outputs": output_hashes_valid,
    }
    result["checks"] = checks
    result["valid"] = all(checks.values())
    result["sha256"] = _sha256(path)
    return result


def _claim_boundary_gate(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Independently reject mislabeled proxy or hardware claims in final outputs."""

    report_path = root / "REPORT.md"
    result: dict[str, Any] = {
        "report": str(report_path.resolve()),
        "valid": False,
        "checks": {},
    }
    if not report_path.is_file():
        return result
    try:
        report = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    aggregate_dir = root / "aggregates"
    csv_paths = sorted(aggregate_dir.glob("*.csv"))
    csv_checks: dict[str, bool] = {}
    graphiti_system_rows = 0
    graphiti_reference_rows = 0
    for path in csv_paths:
        try:
            with path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
        except (OSError, csv.Error):
            csv_checks[path.name] = False
            continue
        valid = bool(rows)
        for row in rows:
            for field in ("system", "candidate", "reference"):
                if field not in row:
                    continue
                identifier = str(row[field])
                expected = EXPECTED_DISPLAY_NAMES.get(identifier, identifier)
                valid = valid and row.get(f"{field}_display_name") == expected
                valid = valid and identifier != "zep"
                if identifier == "graphiti_zep_oss_proxy":
                    if field == "system":
                        graphiti_system_rows += 1
                    elif field == "reference":
                        graphiti_reference_rows += 1
        csv_checks[path.name] = valid

    zep_naming = protocol.get("zep_naming") or {}
    declared_names = protocol.get("display_names") or {}
    checks = {
        "protocol_display_names": declared_names
        == {
            key: value for key, value in EXPECTED_DISPLAY_NAMES.items() if key != "all"
        },
        "protocol_proxy_claims_disabled": zep_naming.get("production_zep_claim_allowed")
        is False
        and zep_naming.get("table5_zep_reproduction_claim_allowed") is False,
        "report_proxy_label": "`Graphiti (Zep OSS proxy)` is not production Zep"
        in report,
        "report_proxy_not_paired": (
            "Graphiti is deliberately not paired with the paper Zep row" in report
        ),
        "report_hardware_boundary": (
            "not an H800 numerical reproduction and not a GX10 sign-off" in report
        ),
        "report_has_no_local_zep_table_row": "| Zep |" not in report
        and "| Production Zep |" not in report,
        "report_hides_internal_proxy_id": "graphiti_zep_oss_proxy" not in report,
        "csv_files_present": bool(csv_paths),
        "csv_display_names": bool(csv_checks) and all(csv_checks.values()),
        "csv_graphiti_system_rows_present": graphiti_system_rows > 0,
        "csv_graphiti_reference_row_present": graphiti_reference_rows > 0,
    }
    result.update(
        {
            "checks": checks,
            "csv_checks": csv_checks,
            "graphiti_system_rows": graphiti_system_rows,
            "graphiti_reference_rows": graphiti_reference_rows,
            "valid": all(checks.values()),
        }
    )
    return result


def _raw_artifact_specs(
    *,
    root: Path,
    protocol: dict[str, Any],
    schedule_path: Path,
    protocol_path: Path,
) -> list[tuple[str, Path]]:
    """Independently derive every raw file required before analysis."""

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


def _raw_artifact_manifest_gate(
    root: Path,
    schedule_path: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    path = root / "raw_artifact_manifest.json"
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "valid": False,
        "checks": {},
    }
    if not path.is_file():
        return result
    try:
        manifest = _load_json(path)
        specs = _raw_artifact_specs(
            root=root,
            protocol=protocol,
            schedule_path=schedule_path,
            protocol_path=protocol_path,
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return result
    try:
        files_exist = all(artifact.is_file() for _, artifact in specs)
        expected_entries = [
            {
                "role": role,
                "path": str(artifact.resolve()),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            }
            for role, artifact in specs
            if artifact.is_file()
        ]
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    by_role: dict[str, int] = {}
    for entry in expected_entries:
        role = str(entry["role"])
        by_role[role] = by_role.get(role, 0) + 1
    canonical = json.dumps(
        expected_entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    observed_entries = manifest.get("artifacts")
    checks = {
        "schema": manifest.get("schema_version")
        == "table5_track_c_raw_artifact_manifest_v0.1.0",
        "status": manifest.get("status") == "complete",
        "result_root": manifest.get("result_root") == str(root.resolve()),
        "schedule": manifest.get("schedule_sha256") == _sha256(schedule_path),
        "protocol": manifest.get("protocol_sha256") == _sha256(protocol_path),
        "all_raw_files_exist": files_exist,
        "artifact_count": manifest.get("artifact_count") == len(specs),
        "by_role": manifest.get("by_role") == dict(sorted(by_role.items())),
        "entries_exact": observed_entries == expected_entries,
        "combined_sha256": manifest.get("combined_sha256")
        == hashlib.sha256(canonical).hexdigest(),
    }
    result.update(
        {
            "checks": checks,
            "valid": all(checks.values()),
            "sha256": _sha256(path),
            "expected_artifacts": len(specs),
            "observed_artifacts": (
                len(observed_entries) if isinstance(observed_entries, list) else 0
            ),
        }
    )
    return result


def _protocol_gate(
    schedule_path: Path,
    schedule: dict[str, Any],
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    dataset = Path(protocol["dataset"]["path"])
    dataset_observation = _dataset_manifest_observation(protocol)
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_systems = [
        "tridb_gem",
        "mem0",
        "memos",
        "cognee",
        "graphiti_zep_oss_proxy",
    ]
    source_receipts = protocol.get("source_receipts") or {}
    source_identity = protocol.get("source_identity") or {}
    source_hashes = schedule.get("source_receipt_sha256") or {}
    checks = {
        "schedule_schema": schedule.get("schema_version")
        == "table5_track_c_completion_schedule_v0.1.0",
        "protocol_schema": protocol.get("schema_version")
        == "table5_track_c_completion_protocol_v0.1.0",
        "schedule_frozen": schedule.get("status") == "frozen",
        "protocol_frozen": protocol.get("status") == "frozen",
        "schedule_sha": schedule.get("_schedule_sha256") == _sha256(schedule_path),
        "protocol_sha": schedule.get("protocol_receipt_sha256")
        == _sha256(protocol_path),
        "plan": protocol.get("plan")
        == "haikaidocs/table5_mem0_zep_memos_cognee_tridb_gem_reproduction_plan_2026-08-19.md"
        and schedule.get("plan_sha256")
        == _sha256(REPOSITORY / str(protocol.get("plan") or "")),
        "result_root": schedule.get("result_root") == protocol.get("result_root"),
        "dataset_exists": dataset.is_file(),
        "dataset_sha": _sha256(dataset) == protocol["dataset"]["sha256"],
        "dataset_manifest_recomputed": dataset_observation["valid"],
        "systems": schedule.get("systems") == expected_systems
        and protocol.get("systems") == expected_systems,
        "workloads": schedule.get("workloads")
        == ["search", "add_native", "add_source_to_searchable"]
        and protocol.get("workloads")
        == ["search", "add_native", "add_source_to_searchable"],
        "rounds": schedule.get("rounds") == ["b1", "b2", "b3"]
        and protocol.get("rounds") == ["b1", "b2", "b3"],
        "formal_run_count": schedule.get("expected_formal_runs") == 45
        and protocol.get("expected_formal_runs") == 45,
        "round_order": schedule.get("round_order")
        == {
            "b1": [
                "tridb_gem",
                "mem0",
                "memos",
                "cognee",
                "graphiti_zep_oss_proxy",
            ],
            "b2": [
                "mem0",
                "memos",
                "cognee",
                "graphiti_zep_oss_proxy",
                "tridb_gem",
            ],
            "b3": [
                "memos",
                "cognee",
                "graphiti_zep_oss_proxy",
                "tridb_gem",
                "mem0",
            ],
        },
        "formal_state_policy": schedule.get("failed_run_retry") == "never_automatic"
        and schedule.get("historical_qps_sweep_reused_as_formal_build") is False
        and schedule.get("historical_qps_sweep_exclusion_reason")
        == "Search build and measurement receipts have different benchmark hashes "
        "and are not bound to one frozen schedule/protocol receipt.",
        "source_receipt_paths": list(source_receipts) == expected_systems,
        "source_identity": list(source_identity) == expected_systems
        and all(
            isinstance(source_identity[system], dict) for system in expected_systems
        ),
        "source_hashes_frozen": list(source_hashes) == expected_systems
        and all(
            isinstance(source_hashes[system], str) and len(source_hashes[system]) == 64
            for system in expected_systems
        ),
        "models": protocol.get("models")
        == {
            "answer_artifact": "Qwen/Qwen3-32B-FP8",
            "answer_revision": "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
            "answer_served_name": "Qwen/Qwen3-32B",
            "answer_endpoint": "http://127.0.0.1:8000/v1",
            "answer_thinking": False,
            "answer_max_model_len": 32768,
            "embedding_artifact": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            "embedding_served_name": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_endpoint": "http://127.0.0.1:8001/v1",
            "embedding_dimensions": 1024,
            "embedding_max_model_len": 8192,
            "offline_cache_required": True,
            "runtime_versions": {
                "vllm": "0.15.0",
                "transformers": "4.57.0",
                "huggingface_hub": "0.36.2",
                "torch": "2.9.1",
            },
        },
        "qps": (protocol.get("protocol") or {}).get("qps") == 10.0,
        "timeout": (protocol.get("protocol") or {}).get("timeout_seconds") == 60.0,
        "formal_cardinality": (protocol.get("protocol") or {}).get("search_formal")
        == 1_787
        and (protocol.get("protocol") or {}).get("add_formal") == 2_000,
        "warmup_cardinality": (protocol.get("protocol") or {}).get("search_warmup")
        == 307
        and (protocol.get("protocol") or {}).get("add_warmup") == 10,
        "execution_policy": (protocol.get("protocol") or {}).get("scheduler")
        == "absolute no-drift open-loop"
        and (protocol.get("protocol") or {}).get("scheduled_interval_ns") == 100_000_000
        and (protocol.get("protocol") or {}).get("admission_qps_relative_tolerance")
        == 0.01
        and (protocol.get("protocol") or {}).get("admission_lag_p99_max_ms") == 100.0
        and (protocol.get("protocol") or {}).get("max_in_flight") == 768
        and (protocol.get("protocol") or {}).get("search_top_k") == 35
        and (protocol.get("protocol") or {}).get("stage_profiling") is True
        and (protocol.get("protocol") or {}).get("per_request_call_observability")
        is True
        and (protocol.get("protocol") or {}).get(
            "fresh_state_per_system_workload_round"
        )
        is True
        and (protocol.get("protocol") or {}).get("one_measured_system_at_a_time")
        is True
        and (protocol.get("protocol") or {}).get("automatic_failed_run_retry") is False,
        "graphiti_add_concurrency_policy": (protocol.get("protocol") or {}).get(
            "graphiti_add_concurrency_policy"
        )
        == "same group_id FIFO with max one active native add_episode; different "
        "group_ids may run concurrently; FIFO wait remains inside measured service "
        "latency; post-commit visibility probe is outside the write lock",
        "fresh_state_policy": (protocol.get("protocol") or {}).get("fresh_state_policy")
        == (
            "hashed launcher refuses every pre-existing per-point database, "
            "volume, or namespace before creation"
        )
        and (protocol.get("protocol") or {}).get("fresh_state_receipt_schema")
        == "table5_track_c_fresh_state_launcher_v0.1.0",
        "call_observability": (protocol.get("protocol") or {}).get(
            "call_observation_fields"
        )
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
        and (protocol.get("protocol") or {}).get("internal_round_trip_policy")
        == "client-boundary counts only; exact HTTP/database round trips inside "
        "compound native APIs are unavailable and never inferred",
        "add_progress_receipt": (protocol.get("protocol") or {}).get(
            "add_progress_receipt_on_timeout"
        )
        is True
        and (protocol.get("protocol") or {}).get("add_progress_fields")
        == [
            "commit_observed",
            "visibility_requested",
            "visibility_probe_started",
            "committed_at_ns",
            "searchable_at_ns",
            "visibility_probe",
            "visibility_error",
        ],
        "late_timeout_outcome_policy": (protocol.get("protocol") or {}).get(
            "late_timeout_outcome_policy"
        )
        == (
            "one sidecar record after every timed-out worker drains; the "
            "client-visible timeout record is immutable"
        )
        and (protocol.get("protocol") or {}).get("late_outcome_schema")
        == "table5_track_c_late_outcome_v0.1.0",
        "add_creation_receipt": (protocol.get("protocol") or {}).get(
            "add_creation_fields"
        )
        == [
            "created_memory_count",
            "created_node_count",
            "created_edge_count",
            "creation_counts_available",
            "creation_count_source",
        ]
        and (protocol.get("protocol") or {}).get("unavailable_creation_count_policy")
        == "explicit null with non-empty native provenance; never coerce to zero",
        "benchmark_code": schedule.get("benchmark_code_sha256")
        == benchmark_code_sha256(),
        "graphiti_adapter": schedule.get("graphiti_adapter_sha256")
        == _graphiti_adapter_sha256(),
        "branch": branch == protocol.get("branch"),
        "graphiti_label": (protocol.get("display_names") or {}).get(
            "graphiti_zep_oss_proxy"
        )
        == "Graphiti (Zep OSS proxy)",
        "graphiti_user_decision": (protocol.get("zep_naming") or {}).get("decision")
        == "user_approved_local_proxy_2026_08_21",
        "graphiti_result_label": (protocol.get("zep_naming") or {}).get("result_label")
        == "Graphiti (Zep OSS proxy)",
        "no_production_zep_claim": (protocol.get("zep_naming") or {}).get(
            "production_zep_claim_allowed"
        )
        is False
        and (protocol.get("zep_naming") or {}).get(
            "table5_zep_reproduction_claim_allowed"
        )
        is False,
        "hardware_claim": (protocol.get("hardware_claim") or {}).get(
            "h800_numerical_reproduction"
        )
        is False
        and (protocol.get("hardware_claim") or {}).get("gx10_signoff") is False,
        "quality_policy": protocol.get("quality")
        == {
            "answer_and_judge_outside_latency_clock": True,
            "non_inferiority_margin_percentage_points": 2.0,
            "non_inferiority_method": "paired_cluster_bootstrap",
            "pairing_keys": ["round", "sample_id", "question_id"],
            "bootstrap_cluster": "sample_id",
            "bootstrap_iterations": 2_000,
            "bootstrap_seed": 20_260_819,
            "one_sided_confidence_level": 0.95,
            "failure_scoring": (
                "retrieval, answer, or judge failure scores incorrect (0)"
            ),
            "receipt": "quality/<system>/<round>/quality_receipt.json",
        },
        "resource_policy": protocol.get("resource_policy")
        == {
            "measured_system_concurrency": (
                "one formal system/workload point at a time"
            ),
            "model_gpu_binding": (
                "answer and embedding vLLM services use CUDA_VISIBLE_DEVICES=0"
            ),
            "model_gpu_prelaunch": (
                "physical GPU0 must have zero compute processes before "
                "controlled-model launch; never terminate external processes "
                "and do not claim exclusive mode"
            ),
            "cpu_affinity": ("observed inherited affinity; no benchmark-specific pin"),
            "ram_cap": ("observed inherited cgroup limit; no benchmark-specific cap"),
            "cache_policy": (
                "warm model and warm process/index; no host page-cache drop"
            ),
            "protocol_deviation": (
                "Pre-registration fairness rule 10.5 requested fixed CPU affinity "
                "and RAM cap. Shared PostgreSQL and separately managed backends "
                "are not placed in one benchmark-only cgroup, so this run records "
                "but does not claim those two controls."
            ),
        },
        "statistics": protocol.get("statistics")
        == {
            "percentile_method": "linear interpolation",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_iterations": 2_000,
            "bootstrap_seed": 20_260_819,
            "bootstrap_cluster": (
                "sample_id (LoCoMo conversation; pooled across fresh builds)"
            ),
            "outlier_policy": "retain all formal admissions",
            "service_latency_population": "successful completions",
            "user_visible_latency_population": (
                "all admissions including failures and timeouts"
            ),
        },
        "paper_reference": protocol.get("paper_reference")
        == {
            "source": "Mandol arXiv:2606.29778 Table 5",
            "comparison_boundary": (
                "context only; Track C uses different controlled models and "
                "non-H800 hardware"
            ),
            "values_ms": {
                "mem0": {
                    "search": {"mean": 1089.0, "p90": 1397.0, "p99": 4637.0},
                    "add:native": {
                        "mean": 888.0,
                        "p90": 1650.0,
                        "p99": 2841.0,
                    },
                },
                "memos": {
                    "search": {"mean": 440.5, "p90": 528.4, "p99": 777.1},
                    "add:native": {
                        "mean": 191.9,
                        "p90": 211.6,
                        "p99": 376.4,
                    },
                },
            },
            "excluded": {
                "graphiti_zep_oss_proxy": (
                    "not production Zep and not comparable to the paper Zep row"
                ),
                "cognee": "no Cognee row in paper Table 5",
                "tridb_gem": "no TriDB/GEM row in paper Table 5",
                "add:source_to_searchable": (
                    "paper Table 5 does not report this endpoint definition"
                ),
            },
        },
    }
    checks.update(
        {
            field: schedule.get(field) == _sha256(REPOSITORY / relative)
            for field, relative in HASHED_SCRIPTS.items()
        }
    )
    return {"valid": all(checks.values()), "checks": checks}


_ADD_PROGRESS_FIELDS = {
    "commit_observed",
    "visibility_requested",
    "visibility_probe_started",
    "committed_at_ns",
    "searchable_at_ns",
    "visibility_probe",
    "visibility_error",
}


def _add_workload_semantics(workload: str, records: list[dict[str, Any]]) -> bool:
    if workload not in {"add_native", "add_source_to_searchable"}:
        raise ValueError(f"not an Add workload: {workload}")

    if not add_receipt_contract(records):
        return False

    for row in records:
        observed = row.get("receipt") or {}
        if not _ADD_PROGRESS_FIELDS <= set(observed):
            return False
        committed = observed.get("committed_at_ns") is not None
        if observed.get("commit_observed") is not committed:
            return False
        if not isinstance(observed.get("visibility_requested"), bool):
            return False
        if not isinstance(observed.get("visibility_probe_started"), bool):
            return False
        if not observed["visibility_probe_started"] and (
            observed.get("visibility_probe") is not None
            or observed.get("searchable_at_ns") is not None
        ):
            return False

        if workload == "add_native":
            if (
                observed["visibility_requested"]
                or observed["visibility_probe_started"]
                or observed.get("visibility_probe") is not None
                or observed.get("searchable_at_ns") is not None
            ):
                return False
            continue

        if not observed["visibility_requested"]:
            return False
        if row.get("success") and (
            not committed
            or not observed["visibility_probe_started"]
            or observed.get("visibility_probe") is not True
            or observed.get("searchable_at_ns") is None
        ):
            return False
        if "visibility probe failed" in str(row.get("error") or "").lower() and (
            observed.get("visibility_probe") is not False
        ):
            return False
    return True


def audit_run(
    schedule: dict[str, Any],
    protocol: dict[str, Any],
    root: Path,
    system: str,
    workload: str,
    round_name: str,
) -> dict[str, Any]:
    run_dir = root / "runs" / system / workload / round_name
    receipt_path = run_dir / "run_receipt.json"
    expected_build = f"tc5v2_{system}_{workload}_{round_name}"
    receipt_phase, formal_phase, warmup_phase = _phase_names(workload)
    formal_count = 1_787 if workload == "search" else 2_000
    warmup_count = 307 if workload == "search" else 10
    if not receipt_path.is_file():
        return {
            "system": system,
            "workload": workload,
            "round": round_name,
            "status": "MISSING",
            "valid": False,
            "receipt": str(receipt_path.resolve()),
            "checks": {},
        }

    try:
        receipt = _load_json(receipt_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "system": system,
            "workload": workload,
            "round": round_name,
            "status": "UNREADABLE",
            "valid": False,
            "receipt": str(receipt_path.resolve()),
            "error": f"{type(exc).__name__}: {exc}",
            "checks": {},
        }

    formal = _jsonl(run_dir / "formal.jsonl")
    warmup = _jsonl(run_dir / "warmup.jsonl")
    spans = _jsonl(run_dir / "spans.jsonl")
    late_path = run_dir / "late_outcomes.jsonl"
    late_outcomes = _jsonl(late_path)
    identity_key = "question_id" if workload == "search" else "event_id"
    try:
        expected_warmup_keys, expected_formal_keys = _canonical_input_keys(
            protocol, workload
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        expected_warmup_keys, expected_formal_keys = [], []
    formal_checks = _request_checks(
        formal,
        expected_count=formal_count,
        expected_phase=formal_phase,
        expected_system=system,
        expected_build=expected_build,
        expected_interval_ns=100_000_000,
        expected_qps=float(protocol["protocol"]["qps"]),
        admission_qps_relative_tolerance=float(
            protocol["protocol"]["admission_qps_relative_tolerance"]
        ),
        admission_lag_p99_max_ms=float(
            protocol["protocol"]["admission_lag_p99_max_ms"]
        ),
    )
    warmup_checks = _request_checks(
        warmup,
        expected_count=warmup_count,
        expected_phase=warmup_phase,
        expected_system=system,
        expected_build=expected_build,
    )
    late_checks = _late_outcome_checks(
        late_outcomes,
        path=late_path,
        request_records=[*formal["records"], *warmup["records"]],
        receipt=receipt,
        expected_system=system,
        expected_build=expected_build,
    )
    formal_traces = {
        str(row["trace_id"])
        for row in formal["records"]
        if isinstance(row.get("trace_id"), str) and row["trace_id"]
    }
    warmup_traces = {
        str(row["trace_id"])
        for row in warmup["records"]
        if isinstance(row.get("trace_id"), str) and row["trace_id"]
    }
    span_tree_checks = _span_tree_checks(
        spans["records"],
        formal_traces=formal_traces,
        warmup_traces=warmup_traces,
        formal_phase=formal_phase,
        warmup_phase=warmup_phase,
        system=system,
        build_id=expected_build,
    )
    span_tree_checks["observable_call_counts_recomputed"] = (
        observable_call_counts_match_spans(
            [*formal["records"], *warmup["records"]], spans["records"]
        )
    )
    try:
        recomputed_breakdown = summarize_spans(run_dir / "spans.jsonl")
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        recomputed_breakdown = None
    execution = receipt.get("execution") or {}
    expected_hashes, hashes_frozen = _expected_hashes(schedule)
    observed_hashes = {
        "benchmark_code_sha256": execution.get("benchmark_code_sha256"),
        "run_one_script_sha256": execution.get("launcher_script_sha256"),
        "formal_schedule_sha256": execution.get("formal_schedule_sha256"),
        "protocol_receipt_sha256": execution.get("protocol_receipt_sha256"),
    }
    if system == "graphiti_zep_oss_proxy":
        observed_hashes["graphiti_adapter_sha256"] = (
            receipt.get("final_stats") or {}
        ).get("adapter_sha256")
    hash_checks = {
        key: observed_hashes.get(key) == value
        for key, value in expected_hashes.items()
        if key in observed_hashes
    }
    fresh_state_checks = _fresh_state_launcher_checks(
        receipt,
        protocol,
        system=system,
        workload=workload,
        round_name=round_name,
    )
    summary = receipt.get("formal_summary") or {}
    breakdown = receipt.get("time_breakdown") or {}
    formal_results = formal["records"]
    if workload == "search":
        workload_semantics = all(
            "generated_answer" not in (row.get("receipt") or {})
            and "answer" not in (row.get("receipt") or {})
            for row in formal_results
        ) and search_receipt_contract(formal_results)
    elif workload == "add_native":
        workload_semantics = _add_workload_semantics(workload, formal_results)
    else:
        workload_semantics = _add_workload_semantics(workload, formal_results)
    graphiti_write_concurrency = True
    if system == "graphiti_zep_oss_proxy" and workload != "search":
        final_policy = (receipt.get("final_stats") or {}).get(
            "formal_write_concurrency"
        ) or {}
        successful_receipts = [
            row.get("receipt") or {}
            for row in formal_results
            if row.get("success") is True
        ]
        formal_fifo_spans = [
            row
            for row in spans["records"]
            if row.get("phase") == formal_phase
            and row.get("operation") == "graphiti.group_fifo_wait"
            and row.get("category") == "framework_other"
            and (row.get("attributes") or {}).get("boundary")
            == "native per-group admission queue"
        ]
        graphiti_write_concurrency = (
            final_policy.get("policy") == "per_group_fifo_cross_group_parallel"
            and final_policy.get("queue_boundary")
            == "native add_episode only; inside measured service latency; "
            "post-commit visibility probe is outside this lock"
            and final_policy.get("same_group_max_active") == 1
            and final_policy.get("cross_group_parallel") is True
            and len(formal_fifo_spans) == formal_count
            and {str(value.get("trace_id")) for value in formal_fifo_spans}
            == formal_traces
            and all(
                value.get("write_concurrency_policy")
                == "per_group_fifo_cross_group_parallel"
                and isinstance(value.get("write_group_fifo_ticket"), int)
                and isinstance(value.get("write_group_fifo_wait_ms"), (int, float))
                and value.get("write_group_fifo_wait_ms") >= 0
                for value in successful_receipts
            )
        )
    checks = {
        "schedule_frozen": schedule.get("status") == "frozen",
        "protocol_frozen": protocol.get("status") == "frozen",
        "all_hashes_frozen": hashes_frozen,
        "receipt_complete": receipt.get("status") == "complete",
        "receipt_schema": receipt.get("schema_version") == "table5_track_c_run_v0.3.0",
        "system": receipt.get("system") == system,
        "phase": receipt.get("phase") == receipt_phase,
        "build_id": receipt.get("build_id") == expected_build,
        "dataset": (receipt.get("dataset") or {}).get("sha256")
        == protocol["dataset"]["sha256"],
        "qps": (receipt.get("protocol") or {}).get("qps")
        == protocol["protocol"]["qps"],
        "timeout": (receipt.get("protocol") or {}).get("timeout_seconds")
        == protocol["protocol"]["timeout_seconds"],
        "harness_retry": (receipt.get("protocol") or {}).get("harness_retries") == 0,
        "max_in_flight": (receipt.get("protocol") or {}).get("max_in_flight") == 768,
        "scheduled_interval": (receipt.get("protocol") or {}).get(
            "scheduled_interval_ns"
        )
        == protocol["protocol"]["scheduled_interval_ns"]
        == 100_000_000,
        "admission_qps_tolerance": (receipt.get("protocol") or {}).get(
            "admission_qps_relative_tolerance"
        )
        == protocol["protocol"]["admission_qps_relative_tolerance"]
        == 0.01,
        "admission_lag_limit": (receipt.get("protocol") or {}).get(
            "admission_lag_p99_max_ms"
        )
        == protocol["protocol"]["admission_lag_p99_max_ms"]
        == 100.0,
        "top_k": (receipt.get("protocol") or {}).get("top_k") == 35,
        "labels_hidden": (receipt.get("protocol") or {}).get(
            "evaluation_fields_visible_to_adapter"
        )
        is False,
        "stage_profiling": (receipt.get("protocol") or {}).get("stage_profiling")
        is True,
        "call_observability_policy": (receipt.get("protocol") or {}).get(
            "per_request_call_observability"
        )
        is True
        and (receipt.get("protocol") or {}).get("call_observation_fields")
        == (protocol.get("protocol") or {}).get("call_observation_fields")
        and (receipt.get("protocol") or {}).get("internal_round_trip_policy")
        == (protocol.get("protocol") or {}).get("internal_round_trip_policy"),
        "late_timeout_outcome_policy": (receipt.get("protocol") or {}).get(
            "late_timeout_outcome_policy"
        )
        == (protocol.get("protocol") or {}).get("late_timeout_outcome_policy")
        and (receipt.get("protocol") or {}).get("late_outcome_schema")
        == (protocol.get("protocol") or {}).get("late_outcome_schema"),
        "late_outcomes": all(late_checks.values()),
        "formal_summary_total": summary.get("total") == formal_count,
        "formal_jsonl": all(formal_checks.values()),
        "warmup_jsonl": all(warmup_checks.values()),
        "formal_inputs_exact": _input_keys_exact(
            formal["records"], expected_formal_keys, identity_key
        ),
        "warmup_inputs_exact": _input_keys_exact(
            warmup["records"], expected_warmup_keys, identity_key
        ),
        "span_exists": spans["exists"],
        "span_parse": spans["errors"] == 0,
        "span_tree": all(span_tree_checks.values()),
        "formal_root_span_per_request": _root_trace_ids(spans["records"], formal_phase)
        == formal_traces,
        "warmup_root_span_per_request": _root_trace_ids(spans["records"], warmup_phase)
        == warmup_traces,
        "time_breakdown": isinstance(receipt.get("time_breakdown"), dict)
        and breakdown.get("formal_requests") == formal_count
        and breakdown.get("formal_requests_with_stage_spans") == formal_count
        and breakdown == recomputed_breakdown,
        "span_identity": all(
            row.get("system") == system and row.get("build_id") == expected_build
            for row in spans["records"]
        ),
        "workload_semantics": workload_semantics,
        "graphiti_write_concurrency": graphiti_write_concurrency,
        "branch": execution.get("git_branch") == "hza214/table5-track-c-20260819",
        "fresh_search_build": workload != "search"
        or (
            receipt.get("build_mode") == "fresh_ingest"
            and isinstance(receipt.get("build_finalize"), dict)
            and isinstance(receipt.get("build_wall_seconds"), (int, float))
        ),
        "fresh_add_state": workload == "search"
        or receipt.get("state_mode") == "fresh_empty",
        "fresh_state_launcher": all(fresh_state_checks.values()),
        "hashes_match": bool(hash_checks) and all(hash_checks.values()),
    }
    valid = all(checks.values())
    status = (
        "COMPLETE_VALID"
        if valid
        else "UNFROZEN"
        if schedule.get("status") != "frozen" or protocol.get("status") != "frozen"
        else "RUNNING"
        if receipt.get("status") == "running"
        else "COMPLETE_INVALID"
    )
    return {
        "system": system,
        "workload": workload,
        "round": round_name,
        "status": status,
        "valid": valid,
        "receipt": str(receipt_path.resolve()),
        "formal_records": len(formal["records"]),
        "successful": summary.get("successful"),
        "failed": summary.get("failed"),
        "service_mean_ms": (summary.get("service_latency_ms") or {}).get("mean"),
        "service_p90_ms": (summary.get("service_latency_ms") or {}).get("p90"),
        "service_p99_ms": (summary.get("service_latency_ms") or {}).get("p99"),
        "completion_qps": summary.get("completion_throughput_qps")
        or summary.get("actual_qps"),
        "checks": checks,
        "request_checks": {"formal": formal_checks, "warmup": warmup_checks},
        "late_outcome_checks": late_checks,
        "fresh_state_checks": fresh_state_checks,
        "span_checks": span_tree_checks,
        "hash_checks": hash_checks,
    }


def audit(schedule_path: Path) -> dict[str, Any]:
    schedule = _load_json(schedule_path)
    schedule["_schedule_sha256"] = _sha256(schedule_path)
    protocol_path = REPOSITORY / schedule["protocol_receipt"]
    protocol = _load_json(protocol_path)
    root = REPOSITORY / protocol["result_root"]
    protocol_gate = _protocol_gate(schedule_path, schedule, protocol_path, protocol)
    systems = protocol["systems"]
    workloads = protocol["workloads"]
    rounds = protocol["rounds"]
    points = [
        audit_run(schedule, protocol, root, system, workload, round_name)
        for workload in workloads
        for round_name in rounds
        for system in systems
    ]
    source_receipts = {
        system: audit_source_receipt(schedule, protocol, system) for system in systems
    }
    conformance = {
        system: audit_conformance_receipt(schedule, protocol, root, system)
        for system in systems
    }
    quality = {
        f"{system}:{round_name}": _quality_point(root, schedule, system, round_name)
        for system in systems
        for round_name in rounds
    }
    deliverables = {
        relative: (root / relative).is_file()
        for relative in protocol["required_deliverables"]
    }
    postprocess_gate = _postprocess_gate(root, schedule_path, protocol_path, protocol)
    claim_boundary_gate = _claim_boundary_gate(root, protocol)
    raw_artifact_manifest_gate = _raw_artifact_manifest_gate(
        root, schedule_path, protocol_path, protocol
    )
    tridb_invariant_gate = _tridb_invariant_gate(
        root, schedule_path, schedule, protocol_path
    )
    model_path = root / "model_receipt.json"
    hardware_path = root / "hardware_receipt.json"
    model_gate: dict[str, Any] = {"path": str(model_path.resolve()), "valid": False}
    hardware_gate: dict[str, Any] = {
        "path": str(hardware_path.resolve()),
        "valid": False,
    }
    if model_path.is_file():
        try:
            model = _load_json(model_path)
            model_gate["valid"] = _completion_model_receipt_valid(model, schedule)
            model_gate["sha256"] = _sha256(model_path)
        except (OSError, json.JSONDecodeError):
            pass
    if hardware_path.is_file():
        try:
            hardware = _load_json(hardware_path)
            hardware_gate["valid"] = _completion_hardware_receipt_valid(
                hardware, schedule
            )
            hardware_gate["sha256"] = _sha256(hardware_path)
        except (OSError, json.JSONDecodeError):
            pass
    prerequisite_complete = (
        protocol_gate["valid"]
        and all(value["valid"] for value in source_receipts.values())
        and all(value["passed"] for value in conformance.values())
        and model_gate["valid"]
        and hardware_gate["valid"]
        and tridb_invariant_gate["valid"]
    )
    return {
        "schema_version": "table5_track_c_goal_audit_v0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schedule": str(schedule_path.resolve()),
        "schedule_status": schedule.get("status"),
        "protocol": str(protocol_path.resolve()),
        "protocol_status": protocol.get("status"),
        "plan": str((REPOSITORY / str(protocol.get("plan") or "")).resolve()),
        "plan_sha256": schedule.get("plan_sha256"),
        "result_root": str(root.resolve()),
        "expected_formal_runs": len(points),
        "complete_valid_runs": sum(bool(point["valid"]) for point in points),
        "points": points,
        "source_receipts": source_receipts,
        "protocol_gate": protocol_gate,
        "conformance": conformance,
        "quality": quality,
        "deliverables": deliverables,
        "postprocess_gate": postprocess_gate,
        "claim_boundary_gate": claim_boundary_gate,
        "raw_artifact_manifest_gate": raw_artifact_manifest_gate,
        "model_gate": model_gate,
        "hardware_gate": hardware_gate,
        "tridb_invariant_gate": tridb_invariant_gate,
        "prerequisite_complete": prerequisite_complete,
        "quality_complete": all(value["valid"] for value in quality.values()),
        "deliverables_complete": all(deliverables.values()),
        "goal_complete": (
            schedule.get("status") == "frozen"
            and protocol.get("status") == "frozen"
            and all(point["valid"] for point in points)
            and prerequisite_complete
            and all(value["valid"] for value in quality.values())
            and all(deliverables.values())
            and postprocess_gate["valid"]
            and claim_boundary_gate["valid"]
            and raw_artifact_manifest_gate["valid"]
        ),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Track C completion-v2 goal audit",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Schedule status: `{result['schedule_status']}`",
        f"Protocol status: `{result['protocol_status']}`",
        f"Plan SHA-256: `{result.get('plan_sha256')}`",
        f"Formal runs: `{result['complete_valid_runs']}/{result['expected_formal_runs']}`",
        f"Goal complete: `{str(result['goal_complete']).lower()}`",
        "",
        "| System | Workload | Round | Status | Formal | Success | Mean ms | P90 ms | P99 ms |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for point in result["points"]:
        values = [
            point["system"],
            point["workload"],
            point["round"],
            point["status"],
            point.get("formal_records", "—"),
            point.get("successful", "—"),
            point.get("service_mean_ms", "—"),
            point.get("service_p90_ms", "—"),
            point.get("service_p99_ms", "—"),
        ]
        rendered = ["—" if value is None else str(value) for value in values]
        lines.append("| " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Prerequisites complete: `{str(result['prerequisite_complete']).lower()}`",
            f"- Quality complete: `{str(result['quality_complete']).lower()}`",
            f"- Deliverables complete: `{str(result['deliverables_complete']).lower()}`",
            f"- Postprocess hashes valid: `{str(result['postprocess_gate']['valid']).lower()}`",
            f"- Claim boundaries valid: `{str(result['claim_boundary_gate']['valid']).lower()}`",
            f"- Raw artifact manifest valid: `{str(result['raw_artifact_manifest_gate']['valid']).lower()}`",
            f"- TriDB architecture invariant: `{str(result['tridb_invariant_gate']['valid']).lower()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--protocol",
        action="store_true",
        help="audit only the frozen schedule/protocol/code identity gate",
    )
    parser.add_argument(
        "--point",
        nargs=3,
        metavar=("SYSTEM", "WORKLOAD", "ROUND"),
        help="audit exactly one formal point and return success only when valid",
    )
    parser.add_argument(
        "--source",
        metavar="SYSTEM",
        help=(
            "audit one immutable source receipt and its live source, dependency, "
            "runtime, and binary identity"
        ),
    )
    parser.add_argument(
        "--conformance",
        metavar="SYSTEM",
        help="recompute one conformance receipt and its frozen code bindings",
    )
    args = parser.parse_args()
    if (
        sum(
            bool(value)
            for value in (args.protocol, args.point, args.source, args.conformance)
        )
        > 1
    ):
        parser.error(
            "--protocol, --point, --source, and --conformance are mutually exclusive"
        )
    if args.protocol:
        schedule_path = args.schedule.resolve()
        schedule = _load_json(schedule_path)
        schedule["_schedule_sha256"] = _sha256(schedule_path)
        protocol_path = REPOSITORY / schedule["protocol_receipt"]
        protocol = _load_json(protocol_path)
        result = _protocol_gate(schedule_path, schedule, protocol_path, protocol)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.source:
        schedule_path = args.schedule.resolve()
        schedule = _load_json(schedule_path)
        schedule["_schedule_sha256"] = _sha256(schedule_path)
        protocol = _load_json(REPOSITORY / schedule["protocol_receipt"])
        result = audit_source_receipt(schedule, protocol, args.source)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.conformance:
        schedule_path = args.schedule.resolve()
        schedule = _load_json(schedule_path)
        schedule["_schedule_sha256"] = _sha256(schedule_path)
        protocol = _load_json(REPOSITORY / schedule["protocol_receipt"])
        root = REPOSITORY / protocol["result_root"]
        result = audit_conformance_receipt(schedule, protocol, root, args.conformance)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.point:
        schedule_path = args.schedule.resolve()
        schedule = _load_json(schedule_path)
        schedule["_schedule_sha256"] = _sha256(schedule_path)
        protocol = _load_json(REPOSITORY / schedule["protocol_receipt"])
        root = REPOSITORY / protocol["result_root"]
        result = audit_run(schedule, protocol, root, *args.point)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    result = audit(args.schedule.resolve())
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "goal_audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "GOAL_AUDIT.md").write_text(
            _markdown(result), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["goal_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
