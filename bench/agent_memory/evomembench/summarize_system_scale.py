"""Summarize the frozen EvoMemBench 1k/10k/100k/1M systems-scale run."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Iterable, Sequence

from bench.agent_memory.evomembench.system_protocol import verify_trace


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or not all(verify_trace(row) for row in rows):
        raise ValueError(f"missing or invalid scale traces: {path}")
    return rows


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}

    def quantile(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "n": len(ordered),
        "mean": mean(ordered),
        "p50": median(ordered),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
    }


def _query_id(target_id: str) -> str:
    return target_id.rsplit(":", 1)[0]


def _paired_query_medians(
    full: Sequence[dict[str, Any]], multi: Sequence[dict[str, Any]]
) -> list[tuple[str, float, float, float, float]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"full": [], "multi": []}
    )
    for row in full:
        grouped[_query_id(str(row["target_id"]))]["full"].append(row)
    for row in multi:
        grouped[_query_id(str(row["target_id"]))]["multi"].append(row)
    pairs: list[tuple[str, float, float, float, float]] = []
    for query_id, arms in sorted(grouped.items()):
        full_ids = {str(row["target_id"]) for row in arms["full"]}
        multi_ids = {str(row["target_id"]) for row in arms["multi"]}
        if full_ids != multi_ids:
            raise ValueError(f"scale paired repetitions differ: {query_id}")
        full_latency = median(float(row["latency_ms"]["total"]) for row in arms["full"])
        multi_latency = median(
            float(row["latency_ms"]["total"]) for row in arms["multi"]
        )
        full_peak = median(
            float(row["intermediate"]["peak_application_materialized_ids"])
            for row in arms["full"]
        )
        multi_peak = median(
            float(row["intermediate"]["peak_materialized_ids"]) for row in arms["multi"]
        )
        pairs.append((query_id, full_latency, multi_latency, full_peak, multi_peak))
    return pairs


def _bootstrap_ratio(
    pairs: Sequence[tuple[str, float, float, float, float]],
    *,
    repetitions: int,
) -> dict[str, Any]:
    ratios = [multi / full for _, full, multi, _, _ in pairs if full > 0]
    if len(ratios) != len(pairs):
        raise ValueError("scale latency must be positive")
    rng = random.Random(42)
    draws = [median(rng.choice(ratios) for _ in ratios) for _ in range(repetitions)]
    draws.sort()
    interval = [
        draws[int(0.025 * (repetitions - 1))],
        draws[int(0.975 * (repetitions - 1))],
    ]
    return {
        "queries": len(ratios),
        "median_ratio_multi_over_full": median(ratios),
        "paired_query_bootstrap_95_ci": interval,
        "median_ratio_full_gem_over_multi": 1 / median(ratios),
        "paired_query_bootstrap_95_ci_full_gem_over_multi": [
            1 / interval[1],
            1 / interval[0],
        ],
        "full_gem_faster_query_fraction": sum(ratio > 1 for ratio in ratios)
        / len(ratios),
        "bootstrap_seed": 42,
        "bootstrap_repetitions": repetitions,
    }


def summarize(run_dir: Path, *, bootstrap_repetitions: int = 10_000) -> dict[str, Any]:
    receipt = json.loads((run_dir / "run_receipt.json").read_text())
    if receipt.get("status") != "complete":
        raise ValueError("scale run is incomplete")
    expected_rows = int(receipt["queries"]) * int(receipt["measured_repetitions"])
    points: list[dict[str, Any]] = []
    for point in receipt["points"]:
        history_size = int(point["history_size"])
        full = _rows(run_dir / "traces" / f"full_gem_{history_size}.jsonl")
        multi = _rows(run_dir / "traces" / f"multi_system_{history_size}.jsonl")
        if len(full) != expected_rows or len(multi) != expected_rows:
            raise ValueError(f"scale trace count mismatch at {history_size}")
        if int(point["parity_queries"]) != int(receipt["queries"]):
            raise ValueError(f"scale parity count mismatch at {history_size}")
        pairs = _paired_query_medians(full, multi)
        if len(pairs) != int(receipt["queries"]):
            raise ValueError(f"scale paired query count mismatch at {history_size}")
        peak_ratios = [
            multi_peak / full_peak
            for _, _, _, full_peak, multi_peak in pairs
            if full_peak > 0
        ]
        points.append(
            {
                "history_size": history_size,
                "parity": {
                    "queries": int(point["parity_queries"]),
                    "fraction": 1.0,
                    "all_passed": True,
                },
                "latency_ms": {
                    "full_gem": _distribution(
                        float(row["latency_ms"]["total"]) for row in full
                    ),
                    "multi_system": _distribution(
                        float(row["latency_ms"]["total"]) for row in multi
                    ),
                },
                "paired_latency": _bootstrap_ratio(
                    pairs, repetitions=bootstrap_repetitions
                ),
                "peak_application_materialized_ids": {
                    "multi_over_full_query_median_ratio": median(peak_ratios),
                    "full_gem_reduction_factor_vs_multi": median(peak_ratios),
                },
                "long_context": point["long_context"],
                "construction": point["construction"],
                "snapshot": point["snapshot"],
                "footprint": point["footprint"],
                "multi_system_load": point["multi_system_load"],
            }
        )
    return {
        "schema_version": "evomembench_gem_system_scale_summary_v0.1.0",
        "status": "complete",
        "formal": bool(receipt.get("formal")),
        "run_id": receipt["run_id"],
        "query_manifest_queries": int(receipt["queries"]),
        "warmups_per_query": int(receipt["warmups"]),
        "measured_repetitions_per_query": int(receipt["measured_repetitions"]),
        "points": points,
        "agent_outcome_measured": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing scale summary: {args.output}")
    report = summarize(args.run_dir, bootstrap_repetitions=args.bootstrap_repetitions)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
