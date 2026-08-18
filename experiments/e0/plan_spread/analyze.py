"""Reduce E0 observations into plan/query metrics and diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.e0.common import artifact_record, environment_record, write_json

from .config import enumerate_plans, load_config


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reduce_observations(
    rows: Iterable[dict[str, Any]],
    *,
    eps_hit: float,
    eps_mrr: float,
    min_equivalent_plans: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(row["dataset"], row["query_id"], row["plan_id"])].append(row)

    plan_rows: list[dict[str, Any]] = []
    for (dataset, query_id, plan_id), samples in sorted(grouped.items()):
        first = samples[0]
        latencies = [float(row["latency_ms"]) for row in samples]
        mechanism: dict[str, Any] = {
            "round_trips_p50": statistics.median(
                float(row.get("round_trips", 0)) for row in samples
            ),
            "bytes_shipped_p50": statistics.median(
                float(row.get("bytes_shipped", 0)) for row in samples
            ),
            "serialization_fraction_p50": statistics.median(
                float(row.get("serialization_fraction", 0.0)) for row in samples
            ),
        }
        for stage in ("ann_ms", "traverse_ms", "filter_ms", "merge_ms"):
            mechanism[f"stage_{stage}_p50"] = statistics.median(
                float(row.get("stage_latency_ms", {}).get(stage, 0.0))
                for row in samples
            )
        for cardinality in (
            "seeds",
            "reached",
            "predicate_matches",
            "candidates",
        ):
            values = [
                row.get("intermediate_cardinality", {}).get(cardinality)
                for row in samples
            ]
            numeric = [float(value) for value in values if value is not None]
            mechanism[f"cardinality_{cardinality}_p50"] = (
                None if not numeric else statistics.median(numeric)
            )
        plan_rows.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "plan_id": plan_id,
                "shape": first["shape"],
                "k": int(first["k"]),
                "hops": int(first["hops"]),
                "predicate_placement": first["predicate_placement"],
                "is_default": bool(first.get("is_default", False)),
                "template": first.get("template", "unknown"),
                "annotation_status": first.get("annotation_status", "unknown"),
                "query_hop_limit": int(first.get("query_hop_limit", 0)),
                "backend": first["backend"],
                "valid_for_system_latency_claims": bool(
                    first.get("valid_for_system_latency_claims", False)
                ),
                "repetitions": len(samples),
                "latency_p50_ms": statistics.median(latencies),
                "latency_p95_ms": _percentile(latencies, 95),
                "latency_p99_ms": _percentile(latencies, 99),
                **mechanism,
                **{name: float(first["quality"][name]) for name in first["quality"]},
            }
        )

    by_query: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        by_query[(row["dataset"], row["query_id"])].append(row)

    query_rows: list[dict[str, Any]] = []
    for (dataset, query_id), plans in sorted(by_query.items()):
        best_hit = max(plan["hit_at_1"] for plan in plans)
        hit_equivalent = [
            plan for plan in plans if plan["hit_at_1"] >= best_hit - eps_hit
        ]
        best_mrr = max(plan["mrr"] for plan in hit_equivalent)
        equivalent = [
            plan for plan in hit_equivalent if plan["mrr"] >= best_mrr - eps_mrr
        ]
        fastest = min(equivalent, key=lambda plan: plan["latency_p50_ms"])
        slowest = max(equivalent, key=lambda plan: plan["latency_p50_ms"])
        minimum = float(fastest["latency_p50_ms"])
        spread_defined = len(equivalent) >= min_equivalent_plans
        spread = (
            None
            if not spread_defined
            else (
                math.inf if minimum == 0 else float(slowest["latency_p50_ms"]) / minimum
            )
        )
        default = next((plan for plan in plans if plan["is_default"]), None)
        default_equivalent = default in equivalent if default is not None else False
        query_rows.append(
            {
                "dataset": dataset,
                "query_id": query_id,
                "template": fastest["template"],
                "annotation_status": fastest["annotation_status"],
                "query_hop_limit": fastest["query_hop_limit"],
                "backend": fastest["backend"],
                "valid_for_system_latency_claims": fastest[
                    "valid_for_system_latency_claims"
                ],
                "plans": len(plans),
                "quality_equivalent_plans": len(equivalent),
                "plan_spread_defined": spread_defined,
                "best_hit_at_1": best_hit,
                "best_mrr": best_mrr,
                "plan_spread": spread,
                "fastest_plan_id": fastest["plan_id"],
                "slowest_plan_id": slowest["plan_id"],
                "winner_shape": fastest["shape"],
                "default_quality_equivalent": default_equivalent,
                "default_suboptimality": (
                    None
                    if not default_equivalent
                    else float(default["latency_p50_ms"]) / minimum
                ),
                "default_hit_at_1": None if default is None else default["hit_at_1"],
                "default_mrr": None if default is None else default["mrr"],
            }
        )
    return plan_rows, query_rows


def _summary(
    query_rows: list[dict[str, Any]], stop_below: float, *, design_complete: bool
) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        by_dataset[row["dataset"]].append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        spreads = [
            float(row["plan_spread"])
            for row in rows
            if row["plan_spread"] is not None and math.isfinite(row["plan_spread"])
        ]
        defaults = [
            float(row["default_suboptimality"])
            for row in rows
            if row["default_suboptimality"] is not None
        ]
        return {
            "queries": len(rows),
            "queries_with_defined_spread": len(spreads),
            "plan_spread": {
                "median": None if not spreads else statistics.median(spreads),
                "p95": _percentile(spreads, 95),
                "max": None if not spreads else max(spreads),
            },
            "default_suboptimality": {
                "quality_equivalent_queries": len(defaults),
                "median": None if not defaults else statistics.median(defaults),
                "p95": _percentile(defaults, 95),
            },
            "winner_shape": dict(Counter(row["winner_shape"] for row in rows)),
        }

    combined = summarize(query_rows)
    median = combined["plan_spread"]["median"]
    valid_for_claims = bool(query_rows) and all(
        row["valid_for_system_latency_claims"] for row in query_rows
    )
    stop_evaluable = (
        design_complete
        and valid_for_claims
        and all(row["plan_spread_defined"] for row in query_rows)
    )
    return {
        "schema_version": "e0-plan-summary-v0.2.0",
        "valid_for_system_latency_claims": valid_for_claims,
        "design_complete": design_complete,
        "combined": combined,
        "datasets": {
            name: summarize(rows) for name, rows in sorted(by_dataset.items())
        },
        "stop_condition": {
            "threshold": stop_below,
            "evaluated": stop_evaluable,
            "triggered": (
                None if not stop_evaluable or median is None else median < stop_below
            ),
            "reference_diagnostic_would_trigger": (
                None if median is None else median < stop_below
            ),
        },
    }


def _figures(query_rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    datasets = sorted({row["dataset"] for row in query_rows})
    live = bool(query_rows) and all(
        row["valid_for_system_latency_claims"] for row in query_rows
    )
    backends = {str(row.get("backend", "unknown")) for row in query_rows}
    title_prefix = (
        "TRIDB LIVE (STOCK PG X86_64)"
        if backends == {"tridb_live"}
        else "POLYGLOT LIVE"
        if live
        else "REFERENCE BACKEND — not system latency"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset in datasets:
        values = sorted(
            float(row["plan_spread"])
            for row in query_rows
            if row["dataset"] == dataset
            and row["plan_spread"] is not None
            and math.isfinite(row["plan_spread"])
        )
        if values:
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                label=dataset,
            )
    ax.axvline(2.0, color="black", linestyle="--", linewidth=1, label="P0 threshold")
    ax.set(
        xlabel="quality-equivalent plan spread (max/min)",
        ylabel="ECDF",
        title=f"{title_prefix} — plan spread",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    path = figure_dir / "plan_spread_ecdf.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    values = [row for row in query_rows if row["default_suboptimality"] is not None]
    ax.bar(
        range(len(values)),
        [row["default_suboptimality"] for row in values],
        color="#4c78a8",
    )
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set(
        xlabel="query",
        ylabel="default / optimal p50",
        title=f"{title_prefix} — default suboptimality",
    )
    path = figure_dir / "default_suboptimality.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    shapes = ["vector_first", "filter_first", "traverse_first"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(datasets))
    for shape in shapes:
        heights = [
            sum(
                row["dataset"] == dataset and row["winner_shape"] == shape
                for row in query_rows
            )
            for dataset in datasets
        ]
        ax.bar(datasets, heights, bottom=bottom, label=shape)
        bottom += np.asarray(heights)
    ax.set(ylabel="winning queries", title=f"{title_prefix} — fastest equivalent shape")
    ax.legend()
    path = figure_dir / "winner_shape.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels, groups = [], []
    for dataset in datasets:
        for hop in sorted(
            {row["query_hop_limit"] for row in query_rows if row["dataset"] == dataset}
        ):
            group = [
                row["plan_spread"]
                for row in query_rows
                if row["dataset"] == dataset
                and row["query_hop_limit"] == hop
                and row["plan_spread"] is not None
                and math.isfinite(row["plan_spread"])
            ]
            if group:
                labels.append(f"{dataset}\nh={hop}")
                groups.append(group)
    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=True)
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1)
    ax.set(ylabel="plan spread", title=f"{title_prefix} — spread by audited hop")
    path = figure_dir / "spread_by_hop.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def analyze(raw_path: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quality = config["quality"]
    plan_rows, query_rows = reduce_observations(
        rows,
        eps_hit=float(quality["eps_hit"]),
        eps_mrr=float(quality["eps_mrr"]),
        min_equivalent_plans=int(
            config["stop_conditions"]["require_at_least_equivalent_plans"]
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "metrics_plan.csv"
    query_path = output_dir / "metrics_query.csv"
    _write_csv(plan_path, plan_rows)
    _write_csv(query_path, query_rows)
    expected_observations = 0
    for dataset_name, spec in config["datasets"].items():
        query_count = sum(
            1
            for line in Path(spec["queries"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        expected_observations += (
            query_count
            * len(enumerate_plans(config, dataset_name))
            * int(config["repetitions"])
        )
    run_manifest_path = raw_path.parent / "run_manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.exists()
        else {}
    )
    design_complete = bool(
        run_manifest.get("complete")
        and run_manifest.get("backend") in {"polyglot_live", "tridb_live"}
        and set(run_manifest.get("datasets", [])) == set(config["datasets"])
        and run_manifest.get("query_limit") is None
        and run_manifest.get("plan_limit") is None
        and int(run_manifest.get("repetitions", 0)) == int(config["repetitions"])
        and len(rows) == expected_observations
    )
    summary = _summary(
        query_rows,
        float(config["stop_conditions"]["plan_spread_median_below"]),
        design_complete=design_complete,
    )
    write_json(output_dir / "summary.json", summary)
    figures = _figures(query_rows, output_dir)
    manifest = {
        "schema_version": "e0-plan-analysis-v0.2.0",
        "environment": environment_record(),
        "input": artifact_record(raw_path),
        "config": artifact_record(config_path),
        "outputs": [
            artifact_record(plan_path),
            artifact_record(query_path),
            artifact_record(output_dir / "summary.json"),
            *[artifact_record(path) for path in figures],
        ],
        "observations": len(rows),
        "plans": len(plan_rows),
        "queries": len(query_rows),
        "expected_observations": expected_observations,
        "design_complete": design_complete,
        "valid_for_system_latency_claims": summary["valid_for_system_latency_claims"],
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e0/plan_space_v0.2.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = analyze(args.raw, args.config, args.output_dir)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "observations",
                    "plans",
                    "queries",
                    "valid_for_system_latency_claims",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
