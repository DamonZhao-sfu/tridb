"""Compare two complete E0 runs and emit a machine-readable stability receipt."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from tools.e0.common import artifact_record, environment_record, write_json


def _csv_by(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return {tuple(row[key] for key in keys): row for row in csv.DictReader(handle)}


def _ratio(a: float, b: float) -> float:
    low, high = sorted((a, b))
    return float("inf") if low == 0 else high / low


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def compare(run_a: Path, run_b: Path) -> dict[str, Any]:
    summaries = [
        json.loads((root / "summary.json").read_text()) for root in (run_a, run_b)
    ]
    manifests = [
        json.loads((root / "run_manifest.json").read_text()) for root in (run_a, run_b)
    ]
    queries = [
        _csv_by(root / "metrics_query.csv", ("dataset", "query_id"))
        for root in (run_a, run_b)
    ]
    plans = [
        _csv_by(root / "metrics_plan.csv", ("dataset", "query_id", "plan_id"))
        for root in (run_a, run_b)
    ]
    if queries[0].keys() != queries[1].keys():
        raise ValueError("query sets differ")
    if plans[0].keys() != plans[1].keys():
        raise ValueError("plan sets differ")

    query_ratios = [
        _ratio(
            float(queries[0][key]["plan_spread"]), float(queries[1][key]["plan_spread"])
        )
        for key in queries[0]
    ]
    plan_ratios = [
        _ratio(
            float(plans[0][key]["latency_p50_ms"]),
            float(plans[1][key]["latency_p50_ms"]),
        )
        for key in plans[0]
    ]
    winner_matches = sum(
        queries[0][key]["winner_shape"] == queries[1][key]["winner_shape"]
        for key in queries[0]
    )

    raw_checks = []
    for root in (run_a, run_b):
        rows = [
            json.loads(line)
            for line in (root / "observations.jsonl").read_text().splitlines()
            if line.strip()
        ]
        raw_checks.append(
            {
                "observations": len(rows),
                "errors": sum(row.get("status") != "ok" for row in rows),
                "censored": sum(bool(row.get("graph_censored")) for row in rows),
            }
        )

    medians = [float(value["combined"]["plan_spread"]["median"]) for value in summaries]
    return {
        "schema_version": "e0-repeatability-v0.3.0",
        "environment": environment_record(),
        "runs": [
            {
                "directory": str(root),
                "summary": artifact_record(root / "summary.json"),
                "observations": artifact_record(root / "observations.jsonl"),
                "seconds": manifest["seconds"],
                "design_complete": summary["design_complete"],
                "median_plan_spread": median,
                **check,
            }
            for root, manifest, summary, median, check in zip(
                (run_a, run_b), manifests, summaries, medians, raw_checks
            )
        ],
        "combined_median_spread_ratio": _ratio(*medians),
        "query_spread_run_ratio": _distribution(query_ratios),
        "plan_p50_latency_run_ratio": _distribution(plan_ratios),
        "winner_shape_agreement": {
            "matching_queries": winner_matches,
            "queries": len(queries[0]),
            "fraction": winner_matches / len(queries[0]),
        },
        "valid": all(summary["design_complete"] for summary in summaries)
        and all(
            check["errors"] == 0 and check["censored"] == 0 for check in raw_checks
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.run_a, args.run_b)
    write_json(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
