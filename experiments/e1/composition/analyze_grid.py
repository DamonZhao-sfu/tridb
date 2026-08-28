"""Reduce a complete E1 grid into per-point and matched-quality summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.e0.common import artifact_record, write_json

from .analysis import matched_quality_pairs
from .config import load_config


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reduce_grid(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("run_kind") != "full_grid":
            raise ValueError("grid analysis accepts only full_grid observations")
        if row.get("status") != "ok" or not row.get("result_validity"):
            raise ValueError("grid contains an error, censor, or invalid result")
        grouped[
            (row["dataset"], row["query_id"], row["system"], row["point_id"])
        ].append(row)

    points = []
    for (dataset, query_id, system, point_id), samples in sorted(grouped.items()):
        first = samples[0]
        qualities = {json.dumps(row["quality"], sort_keys=True) for row in samples}
        results = {json.dumps(row["result_ids"], sort_keys=True) for row in samples}
        latencies = [float(row["client_wall_latency_ns"]) / 1e6 for row in samples]
        quality = {
            name: statistics.median(float(row["quality"][name]) for row in samples)
            for name in ("mrr", "recall_at_20", "hit_at_1", "hit_at_5")
        }
        points.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "system": system,
                "point_id": point_id,
                "shape": first["shape"],
                "k": int(first["k"]),
                "hops": int(first["hops"]),
                "predicate_placement": first["predicate_placement"],
                "ann_effort": int(first["ann_effort"]),
                "template": first["template"],
                "annotation_status": first["annotation_status"],
                "repetitions": len(samples),
                "distinct_quality_values": len(qualities),
                "distinct_result_sets": len(results),
                "latency_p50_ms": statistics.median(latencies),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
                "mrr": float(quality["mrr"]),
                "recall_at_20": float(quality["recall_at_20"]),
                "hit_at_1": float(quality["hit_at_1"]),
                "hit_at_5": float(quality["hit_at_5"]),
                "store_rpc_count": statistics.median(
                    int(row["store_rpc_count"]) for row in samples
                ),
                "payload_bytes": statistics.median(
                    sum(row["payload_bytes_by_boundary"].values()) for row in samples
                ),
            }
        )
    return points


def _pair_score(pair: dict[str, Any]) -> float:
    tridb = (pair["tridb_mrr"] + pair["tridb_recall_at_20"]) / 2
    polyglot = (pair["polyglot_mrr"] + pair["polyglot_recall_at_20"]) / 2
    return min(tridb, polyglot)


def representative_pairs(
    tridb: list[dict[str, Any]],
    polyglot: list[dict[str, Any]],
    *,
    mrr_epsilon: float,
    recall_epsilon: float,
    iso_plan: bool,
) -> list[dict[str, Any]]:
    pairs = matched_quality_pairs(
        tridb,
        polyglot,
        mrr_epsilon=mrr_epsilon,
        recall_epsilon=recall_epsilon,
        require_positive=True,
    )
    if iso_plan:
        pairs = [
            pair
            for pair in pairs
            if pair["tridb_point_id"] == pair["polyglot_point_id"]
        ]
    if not pairs:
        return []
    for pair in pairs:
        pair["common_quality_score"] = _pair_score(pair)
    best_score = max(pair["common_quality_score"] for pair in pairs)
    selections = {
        "fast": min(
            pairs,
            key=lambda pair: max(
                pair["tridb_latency_p50_ms"], pair["polyglot_latency_p50_ms"]
            ),
        ),
        "balanced": min(
            (
                pair
                for pair in pairs
                if pair["common_quality_score"] >= 0.9 * best_score
            ),
            key=lambda pair: max(
                pair["tridb_latency_p50_ms"], pair["polyglot_latency_p50_ms"]
            ),
        ),
        "high_quality": min(
            (pair for pair in pairs if pair["common_quality_score"] == best_score),
            key=lambda pair: max(
                pair["tridb_latency_p50_ms"], pair["polyglot_latency_p50_ms"]
            ),
        ),
    }
    return [{"operating_label": label, **pair} for label, pair in selections.items()]


def _bootstrap_ci(values: list[float], *, seed: int = 20260818) -> list[float] | None:
    if len(values) < 2:
        return None
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    medians = np.median(
        generator.choice(array, size=(10_000, len(array)), replace=True), axis=1
    )
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["analysis"], row["operating_label"])].append(
            float(row["polyglot_over_tridb_p50"])
        )
    result = {}
    for (analysis, label), values in sorted(grouped.items()):
        result.setdefault(analysis, {})[label] = {
            "queries": len(values),
            "median_speedup": statistics.median(values),
            "geomean_speedup": math.exp(
                statistics.mean(math.log(value) for value in values)
            ),
            "bootstrap_median_95ci": _bootstrap_ci(values),
            "fraction_tridb_faster": sum(value > 1 for value in values) / len(values),
        }
    return result


def analyze(
    config_path: Path, observations_path: Path, output_dir: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    rows = [
        json.loads(line)
        for line in observations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    points = reduce_grid(rows)
    by_query: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        by_query[(point["dataset"], point["query_id"], point["system"])].append(point)
    matched = config["comparison"]["matched_quality"]
    pair_rows = []
    query_keys = sorted({(point["dataset"], point["query_id"]) for point in points})
    exclusions = []
    for dataset, query_id in query_keys:
        tridb = by_query[(dataset, query_id, "tridb_live")]
        polyglot = by_query[(dataset, query_id, "polyglot_tuned")]
        for analysis_name, iso_plan in (("iso_plan", True), ("pareto_envelope", False)):
            selected = representative_pairs(
                tridb,
                polyglot,
                mrr_epsilon=float(matched["mrr_epsilon"]),
                recall_epsilon=float(matched["recall_at_20_epsilon"]),
                iso_plan=iso_plan,
            )
            if not selected:
                exclusions.append(
                    {
                        "dataset": dataset,
                        "query_id": query_id,
                        "analysis": analysis_name,
                    }
                )
            for pair in selected:
                pair_rows.append(
                    {
                        "dataset": dataset,
                        "query_id": query_id,
                        "analysis": analysis_name,
                        **pair,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "grid_points.csv", points)
    _write_csv(output_dir / "matched_pairs.csv", pair_rows)
    summary = {
        "schema_version": "e1-composition-grid-analysis-v0.1.0",
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "ANALYZED",
        },
        "inputs": [artifact_record(config_path), artifact_record(observations_path)],
        "observations": len(rows),
        "points": len(points),
        "queries": len(query_keys),
        "matched_pair_rows": len(pair_rows),
        "excluded_query_analyses": exclusions,
        "p99_evaluated": False,
        "comparison": summarize_pairs(pair_rows),
    }
    write_json(output_dir / "grid_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e1/composition_v0.1.yaml")
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = analyze(args.config, args.observations, args.output_dir)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
