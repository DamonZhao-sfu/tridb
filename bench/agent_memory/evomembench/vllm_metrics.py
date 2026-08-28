"""Attributable per-request timing deltas from an exclusive vLLM endpoint."""

from __future__ import annotations

from dataclasses import dataclass
import time
import urllib.request
from typing import Mapping


METRICS = (
    "time_to_first_token_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
    "e2e_request_latency_seconds",
)


@dataclass(frozen=True)
class MetricsSnapshot:
    sums: Mapping[str, float]
    counts: Mapping[str, int]
    collection_seconds: float


def parse_metrics(
    text: str, *, model: str, collection_seconds: float = 0.0
) -> MetricsSnapshot:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    model_marker = f'model_name="{model}"'
    for line in text.splitlines():
        if not line.startswith("vllm:") or model_marker not in line:
            continue
        metric_and_labels, _, raw_value = line.rpartition(" ")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        metric_name = metric_and_labels.split("{", 1)[0].removeprefix("vllm:")
        for base in METRICS:
            if metric_name == f"{base}_sum":
                sums[base] = value
            elif metric_name == f"{base}_count":
                counts[base] = int(value)
    return MetricsSnapshot(
        sums=sums,
        counts=counts,
        collection_seconds=collection_seconds,
    )


def collect(base_url: str, *, model: str, timeout: float = 10.0) -> MetricsSnapshot:
    began = time.perf_counter()
    with urllib.request.urlopen(  # noqa: S310 - operator-supplied local endpoint
        base_url.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        text = response.read().decode()
    elapsed = time.perf_counter() - began
    return parse_metrics(text, model=model, collection_seconds=elapsed)


def attributable_delta(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    *,
    expected_requests: int,
) -> dict[str, float | int | bool | dict[str, int]]:
    count_deltas = {
        metric: int(after.counts.get(metric, 0) - before.counts.get(metric, 0))
        for metric in METRICS
    }
    complete = all(
        metric in before.sums
        and metric in after.sums
        and metric in before.counts
        and metric in after.counts
        and count_deltas[metric] == expected_requests
        for metric in METRICS
    )
    payload: dict[str, float | int | bool | dict[str, int]] = {
        "attributable": complete,
        "expected_requests": expected_requests,
        "count_deltas": count_deltas,
        "telemetry_overhead_ms": (before.collection_seconds + after.collection_seconds)
        * 1000,
    }
    if complete:
        for metric in METRICS:
            payload[f"{metric.removesuffix('_seconds')}_ms"] = max(
                0.0, (after.sums[metric] - before.sums[metric]) * 1000
            )
    return payload


def collect_attributable_delta(
    base_url: str,
    before: MetricsSnapshot,
    *,
    model: str,
    expected_requests: int,
    timeout: float = 2.0,
    poll_interval: float = 0.05,
) -> dict[str, float | int | bool | dict[str, int]]:
    """Wait briefly for vLLM's completed-request histograms to become visible."""
    began = time.perf_counter()
    deadline = began + timeout
    while True:
        after = collect(base_url, model=model, timeout=max(0.1, timeout))
        elapsed = time.perf_counter() - began
        after = MetricsSnapshot(
            sums=after.sums,
            counts=after.counts,
            collection_seconds=elapsed,
        )
        result = attributable_delta(
            before,
            after,
            expected_requests=expected_requests,
        )
        deltas = result["count_deltas"]
        if result["attributable"] is True or any(
            int(deltas[metric]) > expected_requests for metric in METRICS
        ):
            return result
        if time.perf_counter() >= deadline:
            return result
        time.sleep(poll_interval)
