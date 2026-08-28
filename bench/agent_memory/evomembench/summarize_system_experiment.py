"""Aggregate the locked three-arm EvoMemBench systems experiment.

The report keeps retrieval-only and agent end-to-end latency separate.  The
multi-system runner deliberately does not repeat answer generation; after exact
injection parity its end-to-end value is reconstructed by replacing Full GEM's
retrieval component with the measured live multi-system retrieval component.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from bench.agent_memory.evomembench.system_protocol import verify_trace


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(verify_trace(row) for row in rows):
        raise ValueError(f"trace digest verification failed: {path}")
    return rows


def _plain_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _quantile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    observed = [float(value) for value in values]
    return {
        "n": len(observed),
        "mean": mean(observed) if observed else None,
        "p50": median(observed) if observed else None,
        "p95": _quantile(observed, 0.95),
        "p99": _quantile(observed, 0.99),
    }


def _sum_numeric(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows]
    present = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(present) if present else None


def _mean_numeric(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    total = _sum_numeric(rows, field)
    count = sum(isinstance(row.get(field), (int, float)) for row in rows)
    return total / count if total is not None and count else None


def _latency_value(row: Mapping[str, Any], *, multi: bool) -> float:
    latency = row["latency_ms"]
    key = "total" if multi else "memory_or_prompt_assembly"
    return float(latency[key])


def _arm_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    arm = str(rows[0]["arm"]) if rows else "unknown"
    multi = arm == "multi_system"
    tokens = [row.get("tokens", {}) for row in rows]
    intermediates = [row.get("intermediate", {}) for row in rows]
    end_to_end = [
        float(row["latency_ms"]["end_to_end"])
        for row in rows
        if row.get("status") == "complete" and "end_to_end" in row.get("latency_ms", {})
    ]
    statuses = [str(row["status"]) for row in rows]
    latency_breakdown_keys = (
        "query_embedding",
        "database_retrieval",
        "prompt_serialization",
        "prompt_tokenization",
        "model_ttft",
        "model_prefill",
        "model_decode",
        "model_server_end_to_end",
        "generation",
    )
    return {
        "queries": len(rows),
        "status_counts": dict(Counter(statuses)),
        "error_fraction": (
            sum(status not in {"complete", "context_overflow"} for status in statuses)
            / len(rows)
            if rows
            else None
        ),
        "timeout_fraction": (
            sum("timeout" in status.casefold() for status in statuses) / len(rows)
            if rows
            else None
        ),
        "context_overflow_fraction": (
            sum(row["status"] == "context_overflow" for row in rows) / len(rows)
            if rows
            else None
        ),
        "retrieval_or_assembly_ms": _distribution(
            _latency_value(row, multi=multi) for row in rows
        ),
        "measured_end_to_end_ms": _distribution(end_to_end),
        "observed_wall_clock_ms": _distribution(
            float(row["latency_ms"]["end_to_end"])
            for row in rows
            if "end_to_end" in row.get("latency_ms", {})
        ),
        "latency_breakdown_ms": {
            key: _distribution(
                row["latency_ms"][key]
                for row in rows
                if key in row.get("latency_ms", {})
                and (
                    key
                    not in {
                        "model_ttft",
                        "model_prefill",
                        "model_decode",
                        "model_server_end_to_end",
                        "generation",
                    }
                    or row.get("status") == "complete"
                )
            )
            for key in latency_breakdown_keys
        },
        "tokens": {
            key: {
                "mean": _mean_numeric(tokens, key),
                "total": _sum_numeric(tokens, key),
            }
            for key in (
                "answer_prompt",
                "answer_completion",
                "model_prefill",
                "memory_injection",
                "prefix_cache_hit",
            )
        },
        "intermediate": {
            key: {"mean": _mean_numeric(intermediates, key)}
            for key in sorted({key for row in intermediates for key in row})
        },
    }


def _paired(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_by_id = {str(row["target_id"]): row for row in left}
    right_by_id = {str(row["target_id"]): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("paired system arms have different target IDs")
    return [(left_by_id[key], right_by_id[key]) for key in sorted(left_by_id)]


def _paired_ratio(values: Iterable[tuple[float, float]]) -> dict[str, Any]:
    pairs = [(left, right) for left, right in values if left > 0 and right >= 0]
    ratios = [right / left for left, right in pairs]
    differences = [right - left for left, right in pairs]
    return {
        "n": len(pairs),
        "ratio_right_over_left": _distribution(ratios),
        "absolute_difference_right_minus_left": _distribution(differences),
        "right_faster_fraction": (
            sum(right < left for left, right in pairs) / len(pairs) if pairs else None
        ),
        "left_faster_fraction": (
            sum(left < right for left, right in pairs) / len(pairs) if pairs else None
        ),
    }


def _cluster(row: Mapping[str, Any]) -> str:
    probes = row.get("probes", {})
    if probes.get("context_id") is not None:
        return str(probes["context_id"])
    if probes.get("source_environment") and probes.get("target_environment"):
        return f"{probes['source_environment']}__to__{probes['target_environment']}"
    return str(row["scope_id"])


def _clustered_log_ratio(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    left_value: Any,
    right_value: Any,
    repetitions: int = 10_000,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    raw_ratios: list[float] = []
    for left, right in pairs:
        left_observed = float(left_value(left))
        right_observed = float(right_value(right))
        if left_observed <= 0 or right_observed <= 0:
            continue
        ratio = right_observed / left_observed
        raw_ratios.append(ratio)
        grouped.setdefault(_cluster(left), []).append(math.log(ratio))
    if not raw_ratios:
        return {"n": 0, "clusters": 0, "median_ratio": None, "bootstrap_95_ci": None}
    rng = random.Random(42)
    clusters = sorted(grouped)
    draws: list[float] = []
    for _ in range(repetitions):
        sampled = [rng.choice(clusters) for _ in clusters]
        logs = [value for cluster in sampled for value in grouped[cluster]]
        draws.append(math.exp(median(logs)))
    draws.sort()
    return {
        "n": len(raw_ratios),
        "clusters": len(clusters),
        "median_ratio": median(raw_ratios),
        "geometric_mean_ratio": math.exp(mean(math.log(value) for value in raw_ratios)),
        "cluster_bootstrap_95_ci": [
            draws[int(0.025 * (repetitions - 1))],
            draws[int(0.975 * (repetitions - 1))],
        ],
        "cluster_bootstrap_statistic": "median paired log-ratio",
        "bootstrap_seed": 42,
        "bootstrap_repetitions": repetitions,
    }


def _full_over_comparator_gate(comparison: Mapping[str, Any]) -> dict[str, Any]:
    right_over_left = comparison["ratio_right_over_left"]
    clustered = comparison.get("clustered_log_ratio", {})
    observed = right_over_left.get("p50")
    interval = clustered.get("cluster_bootstrap_95_ci")
    full_over = None if observed in (None, 0) else 1 / float(observed)
    full_over_interval = (
        [1 / float(interval[1]), 1 / float(interval[0])]
        if interval and float(interval[0]) > 0
        else None
    )
    faster = comparison.get("left_faster_fraction")
    passed = bool(
        faster is not None
        and float(faster) >= 0.8
        and full_over_interval is not None
        and float(full_over_interval[1]) < 1
    )
    return {
        "full_gem_faster_fraction": faster,
        "median_ratio_full_gem_over_comparator": full_over,
        "paired_cluster_bootstrap_95_ci_full_gem_over_comparator": full_over_interval,
        "passed": passed,
        "threshold": "faster on >=80% paired queries and ratio CI upper <1",
    }


def _tool_successes(quality: Mapping[str, Any], arm: str) -> tuple[float, int]:
    source_arm = "gem_fused" if arm in {"full_gem", "multi_system"} else arm
    cells = quality["cross_environment_transfer"][source_arm]["cells"]
    total = sum(int(cell["n"]) for cell in cells.values())
    successes = sum(
        int(cell["n"]) * float(cell["success_rate"]) for cell in cells.values()
    )
    return successes, total


def _efficiency_point(
    *,
    outcomes: float,
    targets: int,
    arm_summary: Mapping[str, Any],
    construction_latency_ms: float | None,
    construction_tokens: float | None,
) -> dict[str, Any]:
    prompt_tokens = arm_summary["tokens"]["answer_prompt"]["total"]
    completion_tokens = arm_summary["tokens"]["answer_completion"]["total"]
    answer_tokens = (
        float(prompt_tokens or 0) + float(completion_tokens or 0)
        if prompt_tokens is not None or completion_tokens is not None
        else None
    )
    e2e = arm_summary.get(
        "parity_conditioned_reconstructed_end_to_end_ms",
        arm_summary.get(
            "observed_wall_clock_ms", arm_summary["measured_end_to_end_ms"]
        ),
    )
    read_latency_ms = (
        float(e2e["mean"]) * int(e2e["n"]) if e2e.get("mean") is not None else None
    )
    lifecycle_tokens = (
        answer_tokens + construction_tokens
        if answer_tokens is not None and construction_tokens is not None
        else None
    )
    lifecycle_latency_ms = (
        read_latency_ms + construction_latency_ms
        if read_latency_ms is not None and construction_latency_ms is not None
        else None
    )
    return {
        "targets": targets,
        "correct_or_successful_tasks": outcomes,
        "quality_rate": outcomes / targets if targets else None,
        "answer_tokens_total": answer_tokens,
        "answer_input_tokens_total": prompt_tokens,
        "construction_tokens_total": construction_tokens,
        "lifecycle_tokens_total": lifecycle_tokens,
        "read_latency_ms_total": read_latency_ms,
        "construction_latency_ms_total": construction_latency_ms,
        "lifecycle_latency_ms_total": lifecycle_latency_ms,
        "correct_or_success_per_million_answer_input_tokens": (
            outcomes * 1_000_000 / float(prompt_tokens) if prompt_tokens else None
        ),
        "correct_or_success_per_wall_clock_second": (
            outcomes * 1000 / read_latency_ms if read_latency_ms else None
        ),
        "amortized_lifecycle_tokens_per_success": (
            lifecycle_tokens / outcomes
            if lifecycle_tokens is not None and outcomes > 0
            else None
        ),
        "amortized_lifecycle_latency_ms_per_success": (
            lifecycle_latency_ms / outcomes
            if lifecycle_latency_ms is not None and outcomes > 0
            else None
        ),
    }


def _pareto(points: Mapping[str, Mapping[str, Any]]) -> list[str]:
    eligible = {
        arm: row
        for arm, row in points.items()
        if all(
            row.get(key) is not None
            for key in (
                "quality_rate",
                "lifecycle_tokens_total",
                "lifecycle_latency_ms_total",
            )
        )
    }
    frontier = []
    for arm, row in eligible.items():
        dominated = any(
            other != arm
            and float(candidate["quality_rate"]) >= float(row["quality_rate"])
            and float(candidate["lifecycle_tokens_total"])
            <= float(row["lifecycle_tokens_total"])
            and float(candidate["lifecycle_latency_ms_total"])
            <= float(row["lifecycle_latency_ms_total"])
            and (
                float(candidate["quality_rate"]) > float(row["quality_rate"])
                or float(candidate["lifecycle_tokens_total"])
                < float(row["lifecycle_tokens_total"])
                or float(candidate["lifecycle_latency_ms_total"])
                < float(row["lifecycle_latency_ms_total"])
            )
            for other, candidate in eligible.items()
        )
        if not dominated:
            frontier.append(arm)
    return sorted(frontier)


def _parity_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "complete":
        raise ValueError(f"incomplete multi-system replay: {path.parent}")
    if not receipt.get("all_parity_passed"):
        raise ValueError(f"multi-system parity gate failed: {path.parent}")
    return {
        "queries": int(receipt["queries"]),
        "passed": int(receipt["parity_passed"]),
        "fraction": float(receipt["parity_fraction"]),
        "all_passed": True,
    }


def _multi_system_load(receipt: Mapping[str, Any]) -> dict[str, Any]:
    loads = list(receipt.get("load_metrics", []))
    return {
        "snapshots": len(loads),
        "units_total": sum(int(row.get("units", 0)) for row in loads),
        "edges_total": sum(int(row.get("edges", 0)) for row in loads),
        "load_seconds": _distribution(
            float(row.get("load_seconds", 0.0)) for row in loads
        ),
        "load_seconds_total": sum(float(row.get("load_seconds", 0.0)) for row in loads),
        "physical_index_bytes": None,
        "physical_index_bytes_reason": (
            "N/A: Milvus and Neo4j do not expose namespace-isolated physical index "
            "bytes for these shared services"
        ),
        "raw_load_metrics": loads,
    }


def _know_construction(root: Path, arm: str, read_queries: int) -> dict[str, Any]:
    rows = _plain_rows(root / "updates" / f"{arm}.jsonl")
    seconds = [
        float(row.get("construction_seconds", row.get("cost", {}).get("seconds", 0)))
        for row in rows
    ]
    embedding_tokens = [
        int(
            row.get(
                "embedding_tokens", row.get("cost", {}).get("embed_input_tokens", 0)
            )
        )
        for row in rows
    ]
    extraction_prompt = [
        int(row.get("cost", {}).get("prompt_tokens", 0)) for row in rows
    ]
    extraction_completion = [
        int(row.get("cost", {}).get("completion_tokens", 0)) for row in rows
    ]
    total_ms = sum(seconds) * 1000
    return {
        "updates": len(rows),
        "latency_ms": _distribution(value * 1000 for value in seconds),
        "total_latency_ms": total_ms,
        "amortized_latency_ms_per_reuse_query": (
            total_ms / read_queries if read_queries else None
        ),
        "embedding_input_tokens_total": sum(embedding_tokens),
        "extraction_prompt_tokens_total": sum(extraction_prompt),
        "extraction_completion_tokens_total": sum(extraction_completion),
        "wal_bytes_total": sum(int(row.get("wal_bytes", 0)) for row in rows),
        "history_append_bytes_total": sum(
            int(row.get("history_append_bytes", 0)) for row in rows
        ),
    }


def _tool_construction(root: Path, arm: str, read_queries: int) -> dict[str, Any]:
    rows = _plain_rows(root / "updates" / f"{arm}.jsonl")
    seconds = [float(row.get("construction_seconds", 0.0)) for row in rows]
    total_ms = sum(seconds) * 1000
    return {
        "updates": len(rows),
        "committed_updates": sum(row.get("status") == "committed" for row in rows),
        "latency_ms": _distribution(value * 1000 for value in seconds),
        "total_latency_ms": total_ms,
        "amortized_latency_ms_per_target_query": (
            total_ms / read_queries if read_queries else None
        ),
        "extraction_prompt_tokens_total": sum(
            int(row.get("construction_input_tokens", 0)) for row in rows
        ),
        "extraction_completion_tokens_total": sum(
            int(row.get("construction_output_tokens", 0)) for row in rows
        ),
        "embedding_input_tokens_total": sum(
            int(row.get("embedding_tokens", 0)) for row in rows
        ),
        "llm_calls_total": sum(int(row.get("llm_calls", 0)) for row in rows),
        "embedding_calls_total": sum(
            int(row.get("embedding_calls", 0)) for row in rows
        ),
    }


def _track(
    *,
    full: Sequence[dict[str, Any]],
    long: Sequence[dict[str, Any]],
    multi: Sequence[dict[str, Any]],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    full_long = _paired(full, long)
    full_multi = _paired(full, multi)
    full_long_completed = [
        (full_row, long_row)
        for full_row, long_row in full_long
        if full_row.get("status") == "complete" and long_row.get("status") == "complete"
    ]
    if int(parity["queries"]) != len(full_multi):
        raise ValueError("parity receipt count differs from paired trace count")
    reconstructed_multi_e2e: list[float] = []
    for full_row, multi_row in full_multi:
        full_e2e = float(full_row["latency_ms"]["end_to_end"])
        full_retrieval = _latency_value(full_row, multi=False)
        multi_retrieval = _latency_value(multi_row, multi=True)
        reconstructed_multi_e2e.append(
            max(0.0, full_e2e - full_retrieval) + multi_retrieval
        )
    full_summary = _arm_summary(full)
    long_summary = _arm_summary(long)
    multi_summary = _arm_summary(multi)
    multi_summary["tokens"] = full_summary["tokens"]
    multi_summary["token_accounting"] = (
        "inherited from Full GEM after 100% ordered-ID and injection-SHA parity"
    )
    reconstructed_distribution = _distribution(reconstructed_multi_e2e)
    multi_summary["parity_conditioned_reconstructed_end_to_end_ms"] = (
        reconstructed_distribution
    )
    full_long_latency = _paired_ratio(
        (
            float(full_row["latency_ms"]["end_to_end"]),
            float(long_row["latency_ms"]["end_to_end"]),
        )
        for full_row, long_row in full_long_completed
    )
    full_multi_retrieval = _paired_ratio(
        (
            _latency_value(full_row, multi=False),
            _latency_value(multi_row, multi=True),
        )
        for full_row, multi_row in full_multi
    )
    full_multi_e2e = _paired_ratio(
        (
            float(full_row["latency_ms"]["end_to_end"]),
            reconstructed,
        )
        for (full_row, _), reconstructed in zip(
            full_multi, reconstructed_multi_e2e, strict=True
        )
    )
    return {
        "arms": {
            "full_gem": full_summary,
            "long_context": long_summary,
            "multi_system": multi_summary,
        },
        "parity": dict(parity),
        "comparisons": {
            "long_context_to_full_gem_end_to_end": {
                **full_long_latency,
                "clustered_log_ratio": _clustered_log_ratio(
                    full_long_completed,
                    left_value=lambda row: row["latency_ms"]["end_to_end"],
                    right_value=lambda row: row["latency_ms"]["end_to_end"],
                ),
            },
            "long_context_to_full_gem_answer_prompt_tokens": _paired_ratio(
                (
                    float(full_row["tokens"]["answer_prompt"]),
                    float(long_row["tokens"]["answer_prompt"]),
                )
                for full_row, long_row in full_long
            ),
            "multi_system_to_full_gem_retrieval": {
                **full_multi_retrieval,
                "clustered_log_ratio": _clustered_log_ratio(
                    full_multi,
                    left_value=lambda row: _latency_value(row, multi=False),
                    right_value=lambda row: _latency_value(row, multi=True),
                ),
            },
            "multi_system_to_full_gem_reconstructed_end_to_end": full_multi_e2e,
            "multi_system_to_full_gem_peak_application_materialized_ids": _paired_ratio(
                (
                    float(
                        full_row["intermediate"]["peak_application_materialized_ids"]
                    ),
                    float(multi_row["intermediate"]["peak_materialized_ids"]),
                )
                for full_row, multi_row in full_multi
            ),
        },
    }


def summarize(
    *,
    know_run: Path,
    know_multi: Path,
    tool_run: Path,
    tool_multi: Path,
    scale_summary: Path | None = None,
) -> dict[str, Any]:
    know_full = [
        row
        for row in _rows(know_run / "traces" / "full_gem.jsonl")
        if int(row["history_size"]) > 0
    ]
    know_long = [
        row
        for row in _rows(know_run / "traces" / "long_context.jsonl")
        if int(row["history_size"]) > 0
    ]
    know_system = _rows(know_multi / "traces" / "multi_system.jsonl")
    tool_full = [
        row
        for row in _rows(tool_run / "system_traces" / "gem_fused.jsonl")
        if row.get("probes", {}).get("phase") == "transfer"
    ]
    tool_long = [
        row
        for row in _rows(tool_run / "system_traces" / "long_context.jsonl")
        if row.get("probes", {}).get("phase") == "transfer"
    ]
    tool_system = _rows(tool_multi / "traces" / "multi_system.jsonl")
    know_track = _track(
        full=know_full,
        long=know_long,
        multi=know_system,
        parity=_parity_receipt(know_multi / "run_receipt.json"),
    )
    know_track["construction"] = {
        arm: _know_construction(know_run, arm, len(know_full))
        for arm in ("full_gem", "long_context")
    }
    know_track["native_database_footprint"] = json.loads(
        (know_run / "run_receipt.json").read_text()
    ).get("database_footprint_after")
    know_track["multi_system_load"] = _multi_system_load(
        json.loads((know_multi / "run_receipt.json").read_text())
    )
    know_quality = (
        json.loads((know_run / "quality_summary.json").read_text())
        if (know_run / "quality_summary.json").is_file()
        else None
    )
    tool_quality = (
        json.loads((tool_run / "summary.json").read_text())
        if (tool_run / "summary.json").is_file()
        else None
    )
    tool_track = _track(
        full=tool_full,
        long=tool_long,
        multi=tool_system,
        parity=_parity_receipt(tool_multi / "run_receipt.json"),
    )
    tool_track["construction"] = {
        "source_bank_cost_source": "canonical per-update committed traces",
        "full_gem": _tool_construction(tool_run, "gem_fused", len(tool_full)),
        "long_context": _tool_construction(tool_run, "long_context", len(tool_long)),
    }
    tool_receipt = json.loads((tool_run / "run_receipt.json").read_text())
    tool_track["native_database_footprint_before"] = tool_receipt.get(
        "database_footprint_before"
    )
    tool_track["native_database_footprint_after"] = tool_receipt.get(
        "database_footprint_after"
    )
    tool_track["multi_system_load"] = _multi_system_load(
        json.loads((tool_multi / "run_receipt.json").read_text())
    )
    scale = None
    if scale_summary is not None:
        scale = json.loads(scale_summary.read_text())
        if scale.get("status") != "complete":
            raise ValueError("systems-scale summary is incomplete")
    efficiency: dict[str, Any] | None = None
    claim_gates: dict[str, Any] | None = None
    if know_quality is not None and tool_quality is not None:
        know_points: dict[str, Any] = {}
        for arm in ("full_gem", "long_context"):
            construction = know_track["construction"][arm]
            construction_tokens = float(
                construction["embedding_input_tokens_total"]
                + construction["extraction_prompt_tokens_total"]
                + construction["extraction_completion_tokens_total"]
            )
            quality_row = know_quality["arms"][arm]
            outcomes = float(quality_row["cross_episode_accuracy"]) * int(
                quality_row["reuse_decisions"]
            )
            know_points[arm] = _efficiency_point(
                outcomes=outcomes,
                targets=int(quality_row["reuse_decisions"]),
                arm_summary=know_track["arms"][arm],
                construction_latency_ms=float(construction["total_latency_ms"]),
                construction_tokens=construction_tokens,
            )
        multi_summary = dict(know_track["arms"]["multi_system"])
        multi_summary["measured_end_to_end_ms"] = multi_summary[
            "parity_conditioned_reconstructed_end_to_end_ms"
        ]
        full_quality = know_quality["arms"]["full_gem"]
        know_points["multi_system"] = _efficiency_point(
            outcomes=float(full_quality["cross_episode_accuracy"])
            * int(full_quality["reuse_decisions"]),
            targets=int(full_quality["reuse_decisions"]),
            arm_summary=multi_summary,
            construction_latency_ms=None,
            construction_tokens=None,
        )

        tool_points: dict[str, Any] = {}
        for arm in ("full_gem", "long_context"):
            outcomes, targets = _tool_successes(tool_quality, arm)
            construction = tool_track["construction"][arm]
            construction_latency_ms = float(construction["total_latency_ms"])
            construction_tokens = float(
                construction["embedding_input_tokens_total"]
                + construction["extraction_prompt_tokens_total"]
                + construction["extraction_completion_tokens_total"]
            )
            tool_points[arm] = _efficiency_point(
                outcomes=outcomes,
                targets=targets,
                arm_summary=tool_track["arms"][arm],
                construction_latency_ms=construction_latency_ms,
                construction_tokens=construction_tokens,
            )
        outcomes, targets = _tool_successes(tool_quality, "multi_system")
        tool_multi_summary = dict(tool_track["arms"]["multi_system"])
        tool_multi_summary["measured_end_to_end_ms"] = tool_multi_summary[
            "parity_conditioned_reconstructed_end_to_end_ms"
        ]
        tool_points["multi_system"] = _efficiency_point(
            outcomes=outcomes,
            targets=targets,
            arm_summary=tool_multi_summary,
            construction_latency_ms=None,
            construction_tokens=None,
        )
        efficiency = {
            "CrossEp-Know": {
                "arms": know_points,
                "quality_token_latency_pareto_frontier": _pareto(know_points),
            },
            "CrossEp-Tool": {
                "arms": tool_points,
                "quality_token_latency_pareto_frontier": _pareto(tool_points),
            },
            "multi_system_lifecycle_note": (
                "N/A until construction/load latency is normalized across Milvus, "
                "Neo4j, and PostgreSQL; it is not reported as zero"
            ),
        }

        latency_gates = {
            "CrossEp-Know_vs_Long_Context": _full_over_comparator_gate(
                know_track["comparisons"]["long_context_to_full_gem_end_to_end"]
            ),
            "CrossEp-Know_vs_Multi_System": _full_over_comparator_gate(
                know_track["comparisons"]["multi_system_to_full_gem_retrieval"]
            ),
            "CrossEp-Tool_vs_Long_Context": _full_over_comparator_gate(
                tool_track["comparisons"]["long_context_to_full_gem_end_to_end"]
            ),
            "CrossEp-Tool_vs_Multi_System": _full_over_comparator_gate(
                tool_track["comparisons"]["multi_system_to_full_gem_retrieval"]
            ),
        }
        scale_latency_gates = []
        if scale is not None:
            for point in scale["points"]:
                paired = point["paired_latency"]
                interval = paired["paired_query_bootstrap_95_ci_full_gem_over_multi"]
                scale_latency_gates.append(
                    {
                        "history_size": point["history_size"],
                        "full_gem_faster_fraction": paired[
                            "full_gem_faster_query_fraction"
                        ],
                        "median_ratio_full_gem_over_multi": paired[
                            "median_ratio_full_gem_over_multi"
                        ],
                        "paired_bootstrap_95_ci_full_gem_over_multi": interval,
                        "passed": paired["full_gem_faster_query_fraction"] >= 0.8
                        and interval[1] < 1,
                    }
                )
        intermediate_gates = {
            track_name: {
                "median_reduction_factor": track["comparisons"][
                    "multi_system_to_full_gem_peak_application_materialized_ids"
                ]["ratio_right_over_left"]["p50"],
                "passed": float(
                    track["comparisons"][
                        "multi_system_to_full_gem_peak_application_materialized_ids"
                    ]["ratio_right_over_left"]["p50"]
                    or 0
                )
                >= 5,
            }
            for track_name, track in (
                ("CrossEp-Know", know_track),
                ("CrossEp-Tool", tool_track),
            )
        }
        token_gates = {
            track_name: {
                "median_ratio_long_context_over_full_gem": track["comparisons"][
                    "long_context_to_full_gem_answer_prompt_tokens"
                ]["ratio_right_over_left"]["p50"],
                "construction_reported": bool(track.get("construction")),
                "passed": float(
                    track["comparisons"][
                        "long_context_to_full_gem_answer_prompt_tokens"
                    ]["ratio_right_over_left"]["p50"]
                    or 0
                )
                > 1
                and bool(track.get("construction")),
            }
            for track_name, track in (
                ("CrossEp-Know", know_track),
                ("CrossEp-Tool", tool_track),
            )
        }
        parity_pass = all(
            track["parity"]["all_passed"] for track in (know_track, tool_track)
        ) and (
            scale is None
            or all(point["parity"]["all_passed"] for point in scale["points"])
        )
        claim_gates = {
            "1_full_gem_multi_system_parity_100pct": parity_pass,
            "2_know_memory_utility": know_quality["claim_gates"],
            "3_tool_memory_utility": tool_quality["claim_gates"],
            "4_full_gem_long_context_quality_noninferiority": {
                "know": know_quality["claim_gates"]["long_context_noninferiority_pass"],
                "tool": tool_quality["claim_gates"]["long_context_noninferiority_pass"],
            },
            "5_latency": {
                "native": latency_gates,
                "systems_scale": scale_latency_gates,
            },
            "6_peak_application_materialized_ids": intermediate_gates,
            "7_answer_prompt_token_reduction_with_construction": token_gates,
            "parameter_tuning_after_failure": False,
        }
    return {
        "schema_version": "evomembench_gem_system_report_v0.1.0",
        "latency_semantics": {
            "full_gem_vs_long_context": "measured agent end-to-end",
            "full_gem_vs_multi_system_retrieval": "measured retrieval-only",
            "multi_system_end_to_end": (
                "reconstructed after exact injection parity; Full-GEM non-retrieval "
                "time plus measured multi-system retrieval time"
            ),
        },
        "tracks": {
            "CrossEp-Know": know_track,
            "CrossEp-Tool": tool_track,
        },
        "systems_scale": scale,
        "efficiency_normalized": efficiency,
        "claim_gates": claim_gates,
        "quality": {
            "know": know_quality,
            "tool": (tool_quality),
        },
        "unavailable_retrieval_quality_metrics": {
            key: "N/A: EvoMemBench has no independent source-experience qrels"
            for key in ("recall_at_k", "ndcg_at_k", "mrr", "graph_path_recall")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--know-run-dir", type=Path, required=True)
    parser.add_argument("--know-multi-dir", type=Path, required=True)
    parser.add_argument("--tool-run-dir", type=Path, required=True)
    parser.add_argument("--tool-multi-dir", type=Path, required=True)
    parser.add_argument("--scale-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        know_run=args.know_run_dir,
        know_multi=args.know_multi_dir,
        tool_run=args.tool_run_dir,
        tool_multi=args.tool_multi_dir,
        scale_summary=args.scale_summary,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing existing summary: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
