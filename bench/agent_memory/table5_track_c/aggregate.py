"""Aggregate verified Track C JSONL into per-build and pooled CSV tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Sequence

from .protocol import _write_json, benchmark_code_sha256, evidence_quality
from .quality import verify_quality_tree
from .scheduler import percentile
from .stats import read_jsonl, summarize
from .verify import verify_tree


def _clusters(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Cluster repeated measurements by LoCoMo conversation.

    Pooled rows contain the same conversation from three independent fresh
    builds.  Keeping those repeats in one cluster preserves the pre-registered
    conversation-level dependence instead of treating build repeats as new
    independent conversations.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("bootstrap record lacks a non-empty sample_id")
        grouped[sample_id].append(record)
    return list(grouped.values())


def _bootstrap_ci(
    records: Sequence[dict[str, Any]],
    metric: Callable[[Sequence[dict[str, Any]]], float | None],
    *,
    iterations: int = 2_000,
    seed: int = 20_260_819,
) -> tuple[float | None, float | None]:
    clusters = _clusters(records)
    if not clusters:
        return None, None
    generator = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [
            item
            for _ in clusters
            for item in clusters[generator.randrange(len(clusters))]
        ]
        value = metric(sample)
        if value is not None:
            values.append(value)
    return percentile(values, 2.5), percentile(values, 97.5)


def _successful_latency(
    records: Sequence[dict[str, Any]], percent: float
) -> float | None:
    values = [
        float(record["service_latency_ms"])
        for record in records
        if record.get("success") is True
    ]
    return percentile(values, percent)


def _successful_mean(records: Sequence[dict[str, Any]]) -> float | None:
    values = [
        float(record["service_latency_ms"])
        for record in records
        if record.get("success") is True
    ]
    return fmean(values) if values else None


def _user_visible_latency(
    records: Sequence[dict[str, Any]], percent: float
) -> float | None:
    """Return admission-to-completion latency across every recorded outcome."""
    values = [
        float(record["user_visible_latency_ms"])
        for record in records
        if record.get("user_visible_latency_ms") is not None
    ]
    return percentile(values, percent)


def _user_visible_mean(records: Sequence[dict[str, Any]]) -> float | None:
    values = [
        float(record["user_visible_latency_ms"])
        for record in records
        if record.get("user_visible_latency_ms") is not None
    ]
    return fmean(values) if values else None


def _success_rate(records: Sequence[dict[str, Any]]) -> float | None:
    return (
        sum(record.get("success") is True for record in records) / len(records)
        if records
        else None
    )


def _metric_row(
    system: str,
    phase: str,
    build_id: str,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize(records)
    metrics: dict[str, Callable[[Sequence[dict[str, Any]]], float | None]] = {
        "mean": _successful_mean,
        "p90": lambda values: _successful_latency(values, 90),
        "p99": lambda values: _successful_latency(values, 99),
        "user_visible_mean": _user_visible_mean,
        "user_visible_p90": lambda values: _user_visible_latency(values, 90),
        "user_visible_p99": lambda values: _user_visible_latency(values, 99),
        "success_rate": _success_rate,
    }
    cis = {name: _bootstrap_ci(records, function) for name, function in metrics.items()}
    service = summary["service_latency_ms"]
    user_visible = summary["user_visible_latency_ms"]
    return {
        "system": system,
        "phase": phase,
        "build_id": build_id,
        "formal_requests": summary["total"],
        "successful": summary["successful"],
        "failed": summary["failed"],
        "success_rate": summary["success_rate"],
        "success_rate_ci_low": cis["success_rate"][0],
        "success_rate_ci_high": cis["success_rate"][1],
        "service_mean_ms": service["mean"],
        "service_mean_ci_low_ms": cis["mean"][0],
        "service_mean_ci_high_ms": cis["mean"][1],
        "service_p50_ms": service["p50"],
        "service_p90_ms": service["p90"],
        "service_p90_ci_low_ms": cis["p90"][0],
        "service_p90_ci_high_ms": cis["p90"][1],
        "service_p95_ms": service["p95"],
        "service_p99_ms": service["p99"],
        "service_p99_ci_low_ms": cis["p99"][0],
        "service_p99_ci_high_ms": cis["p99"][1],
        "service_max_ms": service["max"],
        "actual_qps": summary["actual_qps"],
        "scheduled_qps": summary["scheduled_qps"],
        "admission_qps": summary["admission_qps"],
        "admission_lag_mean_ms": summary["admission_lag_ms"]["mean"],
        "admission_lag_p99_ms": summary["admission_lag_ms"]["p99"],
        "queue_p99_ms": summary["queue_latency_ms"]["p99"],
        # Unlike service latency above, these fields retain failed and timed-out
        # admissions. They expose overload instead of making a low-success
        # system appear fast by conditioning solely on its successful tail.
        "user_visible_mean_ms": user_visible["mean"],
        "user_visible_mean_ci_low_ms": cis["user_visible_mean"][0],
        "user_visible_mean_ci_high_ms": cis["user_visible_mean"][1],
        "user_visible_p50_ms": user_visible["p50"],
        "user_visible_p90_ms": user_visible["p90"],
        "user_visible_p90_ci_low_ms": cis["user_visible_p90"][0],
        "user_visible_p90_ci_high_ms": cis["user_visible_p90"][1],
        "user_visible_p95_ms": user_visible["p95"],
        "user_visible_p99_ms": user_visible["p99"],
        "user_visible_p99_ci_low_ms": cis["user_visible_p99"][0],
        "user_visible_p99_ci_high_ms": cis["user_visible_p99"][1],
        "user_visible_max_ms": user_visible["max"],
        "error_classes": json.dumps(summary["errors"], sort_keys=True),
    }


def aggregate(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(root)
    grouped: dict[tuple[str, str], list[tuple[str, list[dict[str, Any]]]]] = (
        defaultdict(list)
    )
    metric_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for receipt_path in sorted(root.glob("**/run_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            continue
        records = read_jsonl([receipt_path.parent / "formal.jsonl"])
        system = str(receipt["system"])
        phase = str(receipt["phase"])
        build_id = str(receipt["build_id"])
        metric_rows.append(_metric_row(system, phase, build_id, records))
        grouped[(system, phase)].append((build_id, records))
        if phase == "search":
            quality_rows.append(
                {
                    "system": system,
                    "build_id": build_id,
                    **evidence_quality(records),
                }
            )

    for (system, phase), builds in sorted(grouped.items()):
        pooled = [record for _, records in builds for record in records]
        metric_rows.append(_metric_row(system, phase, "pooled", pooled))
        if phase == "search":
            quality_rows.append(
                {
                    "system": system,
                    "build_id": "pooled",
                    **evidence_quality(pooled),
                }
            )
    return metric_rows, quality_rows


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _decomposition_row(
    system: str,
    phase: str,
    build_id: str,
    records: Sequence[dict[str, Any]],
    late_outcomes: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    successful = [record for record in records if record.get("success") is True]
    failed = [record for record in records if record.get("success") is not True]

    def values(function: Callable[[dict[str, Any]], float | None]) -> list[float]:
        result = []
        for record in successful:
            value = function(record)
            if value is not None:
                result.append(float(value))
        return result

    def receipt(record: dict[str, Any]) -> dict[str, Any]:
        return record.get("receipt") or {}

    internal_cost = values(
        lambda record: (
            1_000 * float(receipt(record)["cost"]["seconds"])
            if (receipt(record).get("cost") or {}).get("seconds") is not None
            else None
        )
    )
    model_call_elapsed = values(
        lambda record: (
            1_000
            * sum(
                float(call.get("elapsed_seconds") or 0.0)
                for call in receipt(record).get("model_calls") or []
            )
            if receipt(record).get("model_calls") is not None
            else None
        )
    )
    committed = values(
        lambda record: (
            (int(receipt(record)["committed_at_ns"]) - int(record["started_at_ns"]))
            / 1_000_000
            if receipt(record).get("committed_at_ns") is not None
            else None
        )
    )
    searchable = values(
        lambda record: (
            (int(receipt(record)["searchable_at_ns"]) - int(record["started_at_ns"]))
            / 1_000_000
            if receipt(record).get("searchable_at_ns") is not None
            else None
        )
    )
    result_count = values(
        lambda record: (
            float(receipt(record)["result_count"])
            if receipt(record).get("result_count") is not None
            else None
        )
    )
    creation_count_fields = (
        "created_memory_count",
        "created_node_count",
        "created_edge_count",
    )
    creation_counts = {
        field: values(
            lambda record, field=field: (
                float(receipt(record)[field])
                if receipt(record).get(field) is not None
                else None
            )
        )
        for field in creation_count_fields
    }
    creation_sources: dict[str, int] = defaultdict(int)
    for record in successful:
        source = receipt(record).get("creation_count_source")
        if source is not None:
            creation_sources[str(source)] += 1

    def probe_values(key: str) -> list[float]:
        return values(
            lambda record: (
                float(receipt(record)["probes"][key])
                if (receipt(record).get("probes") or {}).get(key) is not None
                else None
            )
        )

    termination_reasons: dict[str, int] = defaultdict(int)
    for record in successful:
        reason = (receipt(record).get("probes") or {}).get("termination_reason")
        if reason is not None:
            termination_reasons[str(reason)] += 1
    visibility = [
        receipt(record).get("visibility_probe")
        for record in records
        if receipt(record).get("visibility_probe") is not None
    ]
    commit_observed = [
        record for record in records if receipt(record).get("commit_observed") is True
    ]
    visibility_requested = [
        record
        for record in records
        if receipt(record).get("visibility_requested") is True
    ]
    visibility_started = [
        record
        for record in records
        if receipt(record).get("visibility_probe_started") is True
    ]
    post_timeout_work_ms = [
        float(record["post_timeout_work_ms"])
        for record in late_outcomes
        if record.get("post_timeout_work_ms") is not None
    ]
    return {
        "system": system,
        "phase": phase,
        "build_id": build_id,
        "total_requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "mean_result_count": _mean(result_count),
        "creation_counts_available_rate": (
            sum(
                receipt(record).get("creation_counts_available") is True
                for record in successful
            )
            / len(successful)
            if successful
            else 0.0
        ),
        "creation_count_sources": json.dumps(
            dict(sorted(creation_sources.items())), sort_keys=True
        ),
        **{
            f"{field.removesuffix('_count')}_count_coverage": (
                len(creation_counts[field]) / len(successful) if successful else 0.0
            )
            for field in creation_count_fields
        },
        **{
            f"{field.removesuffix('_count')}_total": sum(creation_counts[field])
            if creation_counts[field]
            else None
            for field in creation_count_fields
        },
        **{
            f"{field.removesuffix('_count')}_mean": _mean(creation_counts[field])
            for field in creation_count_fields
        },
        "internal_cost_coverage": len(internal_cost) / len(successful)
        if successful
        else 0.0,
        "internal_cost_mean_ms": _mean(internal_cost),
        "model_call_timing_coverage": len(model_call_elapsed) / len(successful)
        if successful
        else 0.0,
        "model_call_elapsed_mean_ms": _mean(model_call_elapsed),
        "commit_timing_coverage": len(committed) / len(successful)
        if successful
        else 0.0,
        "commit_mean_ms": _mean(committed),
        "searchable_timing_coverage": len(searchable) / len(successful)
        if successful
        else 0.0,
        "searchable_mean_ms": _mean(searchable),
        "visibility_probe_count": len(visibility),
        "visibility_probe_pass_rate": (
            sum(value is True for value in visibility) / len(visibility)
            if visibility
            else None
        ),
        "visibility_probe_false_count": sum(value is False for value in visibility),
        "add_progress_evidence_count": sum(
            "commit_observed" in receipt(record) for record in records
        ),
        "commit_observed_count": len(commit_observed),
        "failed_after_commit_count": sum(
            record.get("success") is not True for record in commit_observed
        ),
        "visibility_requested_count": len(visibility_requested),
        "visibility_probe_started_count": len(visibility_started),
        "timeout_after_commit_count": sum(
            record.get("timeout") is True for record in commit_observed
        ),
        "timeout_during_visibility_probe_count": sum(
            record.get("timeout") is True
            and receipt(record).get("visibility_probe") is None
            for record in visibility_started
        ),
        "visibility_failures": sum(
            "visibility probe failed" in str(record.get("error") or "").lower()
            for record in records
        ),
        "client_timeout_count": sum(
            record.get("timeout") is True for record in records
        ),
        "late_worker_outcome_count": len(late_outcomes),
        "late_worker_final_success_count": sum(
            record.get("final_success") is True for record in late_outcomes
        ),
        "late_worker_final_failed_count": sum(
            record.get("final_success") is False for record in late_outcomes
        ),
        "late_worker_commit_observed_count": sum(
            (record.get("receipt") or {}).get("committed_at_ns") is not None
            for record in late_outcomes
        ),
        "late_worker_visibility_pass_count": sum(
            (record.get("receipt") or {}).get("visibility_probe") is True
            for record in late_outcomes
        ),
        "post_timeout_work_mean_ms": _mean(post_timeout_work_ms),
        "post_timeout_work_max_ms": max(post_timeout_work_ms, default=None),
        "candidates_examined_mean": _mean(probe_values("candidates_examined")),
        "graph_examined_mean": _mean(probe_values("graph_examined")),
        "bridges_injected_mean": _mean(probe_values("bridges_injected")),
        "termination_reasons": json.dumps(
            dict(sorted(termination_reasons.items())), sort_keys=True
        ),
    }


def aggregate_decomposition(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    grouped: dict[
        tuple[str, str],
        list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    ] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for receipt_path in sorted(root.glob("**/run_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        records = read_jsonl([receipt_path.parent / "formal.jsonl"])
        late_path = receipt_path.parent / "late_outcomes.jsonl"
        late_outcomes = (
            [
                record
                for record in read_jsonl([late_path])
                if str(record.get("phase") or "").startswith("formal_")
            ]
            if late_path.is_file()
            else []
        )
        system = str(receipt["system"])
        phase = str(receipt["phase"])
        build_id = str(receipt["build_id"])
        rows.append(_decomposition_row(system, phase, build_id, records, late_outcomes))
        grouped[(system, phase)].append((build_id, records, late_outcomes))
    for (system, phase), builds in sorted(grouped.items()):
        pooled = [record for _, records, _ in builds for record in records]
        pooled_late = [record for _, _, late in builds for record in late]
        rows.append(_decomposition_row(system, phase, "pooled", pooled, pooled_late))
    return rows


def aggregate_resources(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    rows = []
    for receipt_path in sorted(root.glob("**/run_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        phase = str(receipt["phase"])
        resources = (
            receipt.get("search_resources")
            if phase == "search"
            else receipt.get("resources")
        ) or {}
        final_stats = receipt.get("final_stats") or {}
        build_finalize = receipt.get("build_finalize") or {}
        build_resources = receipt.get("build_resources") or {}
        storage = build_finalize if phase == "search" else final_stats
        database_bytes = storage.get("database_bytes")
        if database_bytes is None and storage.get("neo4j_store_bytes") is not None:
            database_bytes = int(storage["neo4j_store_bytes"]) + int(
                storage.get("sqlite_user_metadata_bytes") or 0
            )
        rows.append(
            {
                "system": receipt["system"],
                "phase": phase,
                "build_id": receipt["build_id"],
                "build_wall_seconds": receipt.get("build_wall_seconds"),
                "database_or_store_bytes": database_bytes,
                "build_peak_runner_process_rss_bytes": build_resources.get(
                    "peak_runner_process_rss_bytes"
                ),
                "build_peak_host_memory_used_bytes": build_resources.get(
                    "peak_host_memory_used_bytes"
                ),
                "build_peak_gpu_memory_used_bytes": json.dumps(
                    build_resources.get("peak_gpu_memory_used_bytes")
                ),
                "peak_runner_process_rss_bytes": resources.get(
                    "peak_runner_process_rss_bytes"
                ),
                "peak_host_memory_used_bytes": resources.get(
                    "peak_host_memory_used_bytes"
                ),
                "peak_gpu_memory_used_bytes": json.dumps(
                    resources.get("peak_gpu_memory_used_bytes")
                ),
                "resource_scope_note": resources.get("scope_note"),
                "max_queue_depth": (receipt.get("scheduler_observed") or {}).get(
                    "max_queue_depth"
                ),
                "max_active": (receipt.get("scheduler_observed") or {}).get(
                    "max_active"
                ),
                "build_summary": json.dumps(
                    receipt.get("build"), sort_keys=True, default=str
                ),
                "build_finalize": json.dumps(
                    receipt.get("build_finalize"), sort_keys=True, default=str
                ),
                "final_stats": json.dumps(final_stats, sort_keys=True, default=str),
            }
        )
    return rows


def aggregate_answer_quality(
    root: str | Path,
    *,
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 20_260_819,
    confidence_level: float = 0.95,
    noninferiority_margin: float = 0.02,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool scores and run paired conversation-cluster non-inferiority tests."""
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    root = Path(root)
    rows: list[dict[str, Any]] = []
    outcomes: dict[str, dict[tuple[str, str, str], int]] = defaultdict(dict)
    pooled: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "evaluated": 0,
            "correct": 0,
            "context_tokens_total": 0,
            "context_token_records": 0,
        }
    )
    for path in sorted(root.glob("**/quality_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        system = str(payload["system"])
        build_id = str(payload["build_id"])
        round_name = path.parent.name
        judges_path = path.parent / "judges.jsonl"
        if not judges_path.is_file():
            raise RuntimeError(f"missing raw judge evidence beside {path}")
        for judge in read_jsonl([judges_path]):
            sample_id = judge.get("sample_id")
            question_id = judge.get("question_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise RuntimeError(f"judge record lacks sample_id in {judges_path}")
            if question_id is None or str(question_id) == "":
                raise RuntimeError(f"judge record lacks question_id in {judges_path}")
            key = (round_name, sample_id, str(question_id))
            if key in outcomes[system]:
                raise RuntimeError(f"duplicate paired quality key {key} for {system}")
            outcomes[system][key] = int(
                judge.get("success") is True and judge.get("correct") is True
            )
        categories = {"overall": payload["overall"], **payload.get("by_category", {})}
        for category, score in categories.items():
            total = int(score["total"])
            evaluated = int(score["evaluated"])
            correct = int(score["correct"])
            rows.append(
                {
                    "system": system,
                    "build_id": build_id,
                    "category": str(category),
                    "total": total,
                    "evaluated": evaluated,
                    "correct": correct,
                    "score": correct / total if total else None,
                    "conditional_score": correct / evaluated if evaluated else None,
                    "coverage": evaluated / total if total else 0.0,
                    "context_tokens_total": payload.get("context_tokens_total")
                    if category == "overall"
                    else None,
                    "context_token_records": payload.get("context_token_records")
                    if category == "overall"
                    else None,
                    "context_token_coverage": payload.get("context_token_coverage")
                    if category == "overall"
                    else None,
                    "mean_context_tokens": payload.get("mean_context_tokens")
                    if category == "overall"
                    else None,
                }
            )
            bucket = pooled[(system, str(category))]
            bucket["total"] += total
            bucket["evaluated"] += evaluated
            bucket["correct"] += correct
            if category == "overall":
                bucket["context_tokens_total"] += int(
                    payload.get("context_tokens_total") or 0
                )
                bucket["context_token_records"] += int(
                    payload.get("context_token_records") or 0
                )

    for (system, category), counts in sorted(pooled.items()):
        total = counts["total"]
        evaluated = counts["evaluated"]
        correct = counts["correct"]
        context_tokens_total = counts["context_tokens_total"]
        context_token_records = counts["context_token_records"]
        rows.append(
            {
                "system": system,
                "build_id": "pooled",
                "category": category,
                **counts,
                "score": correct / total if total else None,
                "conditional_score": correct / evaluated if evaluated else None,
                "coverage": evaluated / total if total else 0.0,
                "context_tokens_total": context_tokens_total
                if category == "overall"
                else None,
                "context_token_records": context_token_records
                if category == "overall"
                else None,
                "context_token_coverage": context_token_records / total
                if category == "overall" and total
                else None,
                "mean_context_tokens": context_tokens_total / context_token_records
                if category == "overall" and context_token_records
                else None,
            }
        )

    overall = {
        row["system"]: row
        for row in rows
        if row["build_id"] == "pooled" and row["category"] == "overall"
    }
    candidate = overall.get("tridb_gem")
    gates = []
    if candidate is not None:
        for reference in (
            "mem0",
            "memos",
            "cognee",
            "graphiti_zep_oss_proxy",
        ):
            if reference not in overall:
                continue
            candidate_outcomes = outcomes.get("tridb_gem") or {}
            reference_outcomes = outcomes.get(reference) or {}
            if not candidate_outcomes or set(candidate_outcomes) != set(
                reference_outcomes
            ):
                raise RuntimeError(
                    "paired quality evidence is missing or mismatched for "
                    f"tridb_gem versus {reference}"
                )
            keys_by_cluster: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
            for key in sorted(candidate_outcomes):
                keys_by_cluster[key[1]].append(key)
            cluster_ids = sorted(keys_by_cluster)
            generator = random.Random(bootstrap_seed)
            bootstrap_deltas: list[float] = []
            for _ in range(bootstrap_iterations):
                sampled_keys = [
                    key
                    for _ in cluster_ids
                    for key in keys_by_cluster[
                        cluster_ids[generator.randrange(len(cluster_ids))]
                    ]
                ]
                bootstrap_deltas.append(
                    sum(
                        candidate_outcomes[key] - reference_outcomes[key]
                        for key in sampled_keys
                    )
                    / len(sampled_keys)
                )
            delta = float(candidate["score"]) - float(overall[reference]["score"])
            paired_delta = sum(
                candidate_outcomes[key] - reference_outcomes[key]
                for key in candidate_outcomes
            ) / len(candidate_outcomes)
            if abs(delta - paired_delta) > 1e-12:
                raise RuntimeError(
                    f"summary/raw quality delta mismatch for {reference}: "
                    f"{delta} != {paired_delta}"
                )
            lower_bound = percentile(bootstrap_deltas, 100.0 * (1.0 - confidence_level))
            gates.append(
                {
                    "candidate": "tridb_gem",
                    "reference": reference,
                    "candidate_score": candidate["score"],
                    "reference_score": overall[reference]["score"],
                    "delta": delta,
                    "margin": noninferiority_margin,
                    "method": "paired_cluster_bootstrap",
                    "pairing_keys": "round,sample_id,question_id",
                    "cluster": "sample_id",
                    "matched_records": len(candidate_outcomes),
                    "clusters": len(cluster_ids),
                    "bootstrap_iterations": bootstrap_iterations,
                    "bootstrap_seed": bootstrap_seed,
                    "one_sided_confidence_level": confidence_level,
                    "one_sided_lower_bound": lower_bound,
                    "passes_noninferiority": lower_bound is not None
                    and lower_bound >= -noninferiority_margin,
                }
            )
    return rows, gates


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-root", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-launcher-hash", required=True)
    parser.add_argument("--expected-schedule-hash", required=True)
    parser.add_argument("--expected-protocol-hash", required=True)
    args = parser.parse_args(argv)
    current_code_hash = benchmark_code_sha256()
    if current_code_hash != args.expected_code_hash:
        raise RuntimeError(
            "aggregation code does not match the frozen run hash: "
            f"{current_code_hash} != {args.expected_code_hash}"
        )
    verification = verify_tree(
        args.root,
        expected_code_hash=args.expected_code_hash,
        expected_launcher_hash=args.expected_launcher_hash,
        expected_schedule_hash=args.expected_schedule_hash,
        expected_protocol_hash=args.expected_protocol_hash,
    )
    verification_passed = bool(
        verification["all_found_runs_pass"]
        and verification["formal_tree_exact"]
        and verification["coverage_complete_for_five_systems"]
        and verification["frozen_benchmark_code_hash_matches"]
        and verification["frozen_launcher_hash_matches"]
        and verification["frozen_schedule_hash_matches"]
        and verification["frozen_protocol_hash_matches"]
    )
    if not verification_passed:
        raise RuntimeError(
            "refusing to aggregate incomplete or protocol-invalid formal runs"
        )
    expected_quality_runs = {
        (str(run["system"]), str(run["build_id"]))
        for run in verification["runs"]
        if run["phase"] == "search" and run["passed"]
    }
    quality_verification = verify_quality_tree(
        args.quality_root,
        expected_runs=expected_quality_runs,
        expected_code_hash=args.expected_code_hash,
    )
    quality_verification_passed = bool(
        quality_verification["coverage_exact"]
        and quality_verification["all_found_runs_pass"]
    )
    if not quality_verification_passed:
        raise RuntimeError("refusing to aggregate incomplete or invalid quality runs")
    metrics, quality = aggregate(args.root)
    decomposition = aggregate_decomposition(args.root)
    resources = aggregate_resources(args.root)
    output = Path(args.output_dir)
    _write_csv(output / "request_metrics.csv", metrics)
    _write_csv(output / "quality_gate.csv", quality)
    _write_csv(output / "latency_decomposition.csv", decomposition)
    _write_csv(output / "resource_and_build_cost.csv", resources)
    answer_quality: list[dict[str, Any]] = []
    noninferiority: list[dict[str, Any]] = []
    answer_quality, noninferiority = aggregate_answer_quality(args.quality_root)
    _write_csv(output / "answer_quality.csv", answer_quality)
    _write_csv(output / "quality_noninferiority.csv", noninferiority)
    output_files = (
        "request_metrics.csv",
        "quality_gate.csv",
        "latency_decomposition.csv",
        "resource_and_build_cost.csv",
        "answer_quality.csv",
        "quality_noninferiority.csv",
    )
    _write_json(
        output / "aggregate_receipt.json",
        {
            "schema_version": "table5_track_c_aggregate_v0.2.0",
            "runs_root": str(Path(args.root).resolve()),
            "quality_root": (
                str(Path(args.quality_root).resolve()) if args.quality_root else None
            ),
            "benchmark_code_sha256": current_code_hash,
            "verification_passed": verification_passed,
            "quality_verification_passed": quality_verification_passed,
            "quality_runs_found": quality_verification["runs_found"],
            "runs_found": verification["runs_found"],
            "coverage": verification["coverage"],
            "bootstrap": {
                "iterations": 2_000,
                "seed": 20_260_819,
                "cluster": "sample_id (LoCoMo conversation; pooled across fresh builds)",
            },
            "metric_rows": len(metrics),
            "quality_rows": len(quality),
            "decomposition_rows": len(decomposition),
            "resource_rows": len(resources),
            "answer_quality_rows": len(answer_quality),
            "noninferiority_rows": len(noninferiority),
            "output_sha256": {
                filename: _sha256(output / filename) for filename in output_files
            },
        },
    )
    print(
        json.dumps(
            {
                "metric_rows": len(metrics),
                "quality_rows": len(quality),
                "decomposition_rows": len(decomposition),
                "resource_rows": len(resources),
                "answer_quality_rows": len(answer_quality),
                "noninferiority_rows": len(noninferiority),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
