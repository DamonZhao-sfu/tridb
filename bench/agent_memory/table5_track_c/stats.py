"""Summaries for raw Track C JSONL without deleting failures or outliers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from .scheduler import percentile


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    return records


def _metric(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successful = [record for record in records if record.get("success") is True]
    failures = [record for record in records if record.get("success") is not True]
    by_error: dict[str, int] = defaultdict(int)
    for record in failures:
        by_error[str(record.get("error") or "unknown")] += 1

    def values(key: str, source: Sequence[dict[str, Any]]) -> list[float]:
        return [float(record[key]) for record in source if record.get(key) is not None]

    rate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        rate_groups[str(record.get("build_id") or "__single_run__")].append(record)

    def grouped_rate(
        *, start_key: str, end_key: str, completed_requests: bool
    ) -> float | None:
        numerator = 0
        duration_ns = 0
        for group in rate_groups.values():
            if len(group) < 1 + (not completed_requests):
                continue
            if any(start_key not in row or end_key not in row for row in group):
                continue
            span_ns = max(int(row[end_key]) for row in group) - min(
                int(row[start_key]) for row in group
            )
            if span_ns <= 0:
                continue
            numerator += len(group) if completed_requests else len(group) - 1
            duration_ns += span_ns
        return (
            numerator / (duration_ns / 1_000_000_000)
            if numerator > 0 and duration_ns > 0
            else None
        )

    completion_throughput_qps = grouped_rate(
        start_key="scheduled_at_ns",
        end_key="completed_at_ns",
        completed_requests=True,
    )
    scheduled_qps = grouped_rate(
        start_key="scheduled_at_ns",
        end_key="scheduled_at_ns",
        completed_requests=False,
    )
    admission_qps = grouped_rate(
        start_key="admitted_at_ns",
        end_key="admitted_at_ns",
        completed_requests=False,
    )
    return {
        "total": total,
        "successful": len(successful),
        "failed": len(failures),
        "success_rate": len(successful) / total if total else 0.0,
        "actual_qps": completion_throughput_qps,
        "completion_throughput_qps": completion_throughput_qps,
        "scheduled_qps": scheduled_qps,
        "admission_qps": admission_qps,
        "errors": dict(sorted(by_error.items())),
        # Table 5 headline follows the frozen artifact and uses successful
        # service completions; failures remain explicit in the denominator.
        "service_latency_ms": _metric(values("service_latency_ms", successful)),
        "admission_lag_ms": _metric(values("admission_lag_ms", records)),
        "queue_latency_ms": _metric(values("queue_latency_ms", records)),
        "user_visible_latency_ms": _metric(values("user_visible_latency_ms", records)),
    }
