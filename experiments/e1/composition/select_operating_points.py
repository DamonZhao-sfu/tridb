"""Freeze E1 operating points using calibration queries only."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.e0.common import artifact_record, write_json

from .analyze_grid import representative_pairs
from .config import load_config
from .staging import (
    load_query_specs,
    load_staged_config,
    point_signature,
    signature_id,
    staged_dataset,
    stratified_query_split,
)


def _load_rows(paths: list[Path]):
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _signature_dict(signature: tuple[str, int, str, int]) -> dict[str, Any]:
    shape, k, placement, effort = signature
    return {
        "shape": shape,
        "k": k,
        "predicate_placement": placement,
        "ann_effort": effort,
    }


def select(staged_path: Path) -> dict[str, Any]:
    staged = load_staged_config(staged_path)
    base_path = Path(staged["base_config"])
    base = load_config(base_path)
    dataset_name = staged_dataset(staged, base)
    spec = base["datasets"][dataset_name]
    queries = load_query_specs(Path(spec["queries"]))
    split_cfg = staged["split"]
    split = stratified_query_split(
        queries,
        seed=str(split_cfg["seed"]),
        calibration_fraction=float(split_cfg["calibration_fraction"]),
    )
    calibration_ids = set(split["calibration"])
    repetitions = int(staged["calibration"]["repetitions"])
    paths = [
        *(Path(value) for value in staged.get("reuse_observations", [])),
        Path(staged["outputs"]["calibration_observations"]),
    ]
    selected_rows: dict[
        tuple[str, str, tuple[str, int, str, int], int], dict[str, Any]
    ] = {}
    for row in _load_rows(paths):
        if row.get("query_id") not in calibration_ids:
            continue
        repetition = int(row["repetition"])
        if repetition >= repetitions:
            continue
        if (
            row.get("status") != "ok"
            or not row.get("result_validity")
            or row.get("graph_censored")
        ):
            raise ValueError("calibration contains an unusable row")
        key = (row["query_id"], row["system"], point_signature(row), repetition)
        if key in selected_rows:
            raise ValueError(f"duplicate calibration row: {key}")
        selected_rows[key] = row

    signatures = {point_signature(row) for row in selected_rows.values()}
    expected = len(calibration_ids) * len(signatures) * 2 * repetitions
    if len(signatures) != 100 or len(selected_rows) != expected:
        raise RuntimeError(
            f"calibration incomplete: signatures={len(signatures)}, "
            f"rows={len(selected_rows)}, expected={expected}"
        )

    per_query: dict[tuple[str, str, tuple[str, int, str, int]], dict[str, Any]] = {}
    for query_id in calibration_ids:
        for system in base["systems"]["headline"]:
            for signature in signatures:
                samples = [
                    selected_rows[(query_id, system, signature, repetition)]
                    for repetition in range(repetitions)
                ]
                per_query[(query_id, system, signature)] = {
                    "latency_p50_ms": statistics.median(
                        row["client_wall_latency_ns"] / 1e6 for row in samples
                    ),
                    "mrr": statistics.median(row["quality"]["mrr"] for row in samples),
                    "recall_at_20": statistics.median(
                        row["quality"]["recall_at_20"] for row in samples
                    ),
                }

    aggregate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signature_by_id = {signature_id(signature): signature for signature in signatures}
    for system in base["systems"]["headline"]:
        for signature in signatures:
            query_points = [
                per_query[(query_id, system, signature)] for query_id in calibration_ids
            ]
            aggregate[system].append(
                {
                    "point_id": signature_id(signature),
                    "latency_p50_ms": statistics.median(
                        point["latency_p50_ms"] for point in query_points
                    ),
                    "mrr": statistics.mean(point["mrr"] for point in query_points),
                    "recall_at_20": statistics.mean(
                        point["recall_at_20"] for point in query_points
                    ),
                }
            )

    matched = base["comparison"]["matched_quality"]
    frozen: dict[str, dict[str, Any]] = {}
    for analysis, iso_plan in (("iso_plan", True), ("pareto_envelope", False)):
        pairs = representative_pairs(
            aggregate["tridb_live"],
            aggregate["polyglot_tuned"],
            mrr_epsilon=float(matched["mrr_epsilon"]),
            recall_epsilon=float(matched["recall_at_20_epsilon"]),
            iso_plan=iso_plan,
        )
        if len(pairs) != 3:
            raise RuntimeError(f"{analysis}: could not freeze three operating labels")
        frozen[analysis] = {}
        for pair in pairs:
            label = pair["operating_label"]
            frozen[analysis][label] = {
                "tridb_live": _signature_dict(signature_by_id[pair["tridb_point_id"]]),
                "polyglot_tuned": _signature_dict(
                    signature_by_id[pair["polyglot_point_id"]]
                ),
                "calibration_quality": {
                    "tridb_mrr": pair["tridb_mrr"],
                    "polyglot_mrr": pair["polyglot_mrr"],
                    "tridb_recall_at_20": pair["tridb_recall_at_20"],
                    "polyglot_recall_at_20": pair["polyglot_recall_at_20"],
                },
            }

    result = {
        "schema_version": "e1-operating-points-v0.2.0",
        "valid_for_headline_claims": False,
        "selection_uses_calibration_queries_only": True,
        "split": split,
        "calibration_repetitions_used": repetitions,
        "inputs": [artifact_record(path) for path in paths if path.exists()],
        "operating_points": frozen,
    }
    output = Path(staged["outputs"]["operating_points"])
    write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-config", type=Path, default=Path("configs/e1/staged_v0.2.yaml")
    )
    args = parser.parse_args(argv)
    result = select(args.staged_config)
    print(json.dumps(result["operating_points"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
