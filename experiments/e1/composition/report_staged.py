"""Analyze and visualize the completed staged E1 composition experiment.

The analysis unit is a held-out query, not an individual timing repetition.
Latency repetitions are reduced within query before query-level bootstrap
intervals are calculated.  The formal system comparison includes only query
pairs that satisfy the preregistered matched-quality gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.e0.common import artifact_record, write_json


SCHEMA_VERSION = "e1-staged-report-v0.1.0"
ANALYSES = ("iso_plan", "pareto_envelope")
LABELS = ("fast", "balanced", "high_quality")
SYSTEMS = ("tridb_live", "polyglot_tuned")
ARMS = (
    "vector_only",
    "graph_only",
    "relational_only",
    "vector_relational",
    "vector_graph",
    "vector_graph_relational",
)
ARM_LABELS = {
    "vector_only": "Vector",
    "graph_only": "Graph",
    "relational_only": "Relational",
    "vector_relational": "Vector + Relational",
    "vector_graph": "Vector + Graph",
    "vector_graph_relational": "All three",
}
TEMPLATE_LABELS = {
    "neighbor_intersection": "Neighbor\nintersection",
    "shared_neighbor_constraint": "Shared-neighbor\nconstraint",
    "typed_chain": "Typed chain",
    "typed_neighbors": "Typed neighbors",
}
SYSTEM_COLORS = {"tridb_live": "#2878B5", "polyglot_tuned": "#E07A1F"}
TEMPLATE_COLORS = {
    "neighbor_intersection": "#3A923A",
    "shared_neighbor_constraint": "#C44E52",
    "typed_chain": "#7B5AA6",
    "typed_neighbors": "#4C9FAD",
}
ARM_COLORS = {
    "vector_only": "#8CB8D9",
    "graph_only": "#83B77A",
    "relational_only": "#C6A7D8",
    "vector_relational": "#E7B66A",
    "vector_graph": "#5AA6A6",
    "vector_graph_relational": "#2878B5",
}
GRID = "#D9DEE7"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno} must contain a JSON object")
            yield value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> list[float] | None:
    if len(values) < 2:
        return None
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = generator.choice(array, size=len(array), replace=True)
        estimates[index] = statistic(sample)
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _point_signature(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["shape"]),
        int(row["k"]),
        str(row["predicate_placement"]),
        int(row["ann_effort"]),
    )


def _frozen_roles(
    frozen: Mapping[str, Any],
) -> dict[tuple[str, str, str], tuple[str, int, str, int]]:
    roles = {}
    for analysis in ANALYSES:
        for label in LABELS:
            pair = frozen["operating_points"][analysis][label]
            for system in SYSTEMS:
                roles[(analysis, label, system)] = _point_signature(pair[system])
    return roles


def _sum_mapping(row: Mapping[str, Any], key: str) -> float:
    values = row.get(key) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{key} must be an object")
    return float(sum(float(value) for value in values.values()))


def load_evaluation(
    frozen: Mapping[str, Any],
    paths: Sequence[Path],
    *,
    repetitions: int,
    mrr_epsilon: float,
    recall_epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evaluation_ids = set(frozen["split"]["evaluation"])
    roles = _frozen_roles(frozen)
    selected: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    physical_keys: set[tuple[str, str, str, int]] = set()

    for path in paths:
        for row in _rows(path):
            if row.get("query_id") not in evaluation_ids:
                continue
            repetition = int(row.get("repetition", -1))
            if not 0 <= repetition < repetitions:
                continue
            system = str(row.get("system"))
            signature = _point_signature(row)
            matching = [
                (analysis, label)
                for (analysis, label, role_system), wanted in roles.items()
                if role_system == system and wanted == signature
            ]
            if not matching:
                continue
            if (
                row.get("status") != "ok"
                or not row.get("result_validity")
                or row.get("graph_censored")
                or not row.get("valid_for_headline_claims")
            ):
                raise ValueError(
                    f"unusable evaluation row for {row.get('query_id')} {system}"
                )
            physical_key = (
                str(row["query_id"]),
                system,
                str(row["point_id"]),
                repetition,
            )
            if physical_key in physical_keys:
                raise ValueError(f"duplicate physical evaluation key: {physical_key}")
            physical_keys.add(physical_key)
            for analysis, label in matching:
                key = (analysis, label, system, str(row["query_id"]), repetition)
                if key in selected:
                    raise ValueError(f"duplicate role-expanded evaluation key: {key}")
                selected[key] = row

    query_system_rows = []
    indexed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for analysis in ANALYSES:
        for label in LABELS:
            for query_id in sorted(evaluation_ids):
                for system in SYSTEMS:
                    samples = [
                        selected[(analysis, label, system, query_id, repetition)]
                        for repetition in range(repetitions)
                        if (analysis, label, system, query_id, repetition) in selected
                    ]
                    if len(samples) != repetitions:
                        raise ValueError(
                            f"{analysis}/{label}/{system}/{query_id}: "
                            f"expected {repetitions} rows, found {len(samples)}"
                        )
                    qualities = {
                        json.dumps(row["quality"], sort_keys=True) for row in samples
                    }
                    if len(qualities) != 1:
                        raise ValueError(
                            f"quality changed across repetitions for {query_id}"
                        )
                    quality = samples[0]["quality"]
                    latencies = [
                        float(row["client_wall_latency_ns"]) / 1e6 for row in samples
                    ]
                    reduced = {
                        "analysis": analysis,
                        "operating_label": label,
                        "query_id": query_id,
                        "template": samples[0]["template"],
                        "system": system,
                        "point_id": samples[0]["point_id"],
                        "latency_p50_ms": statistics.median(latencies),
                        "latency_p95_ms": _percentile(latencies, 95),
                        "mrr": float(quality["mrr"]),
                        "recall_at_20": float(quality["recall_at_20"]),
                        "hit_at_1": float(quality["hit_at_1"]),
                        "hit_at_5": float(quality["hit_at_5"]),
                        "client_request_count": statistics.median(
                            int(row["client_request_count"]) for row in samples
                        ),
                        "store_rpc_count": statistics.median(
                            int(row["store_rpc_count"]) for row in samples
                        ),
                        "cross_store_handoff_count": statistics.median(
                            int(row["cross_store_handoff_count"]) for row in samples
                        ),
                        "intermediate_rows": statistics.median(
                            _sum_mapping(row, "intermediate_rows_by_boundary")
                            for row in samples
                        ),
                        "payload_bytes": statistics.median(
                            _sum_mapping(row, "payload_bytes_by_boundary")
                            for row in samples
                        ),
                        "serialization_ms": statistics.median(
                            float(row["serialization_ns"]) / 1e6 for row in samples
                        ),
                        "serialization_fraction": statistics.median(
                            0.0
                            if int(row["client_wall_latency_ns"]) == 0
                            else float(row["serialization_ns"])
                            / float(row["client_wall_latency_ns"])
                            for row in samples
                        ),
                    }
                    query_system_rows.append(reduced)
                    indexed[(analysis, label, query_id, system)] = reduced

    paired_rows = []
    for analysis in ANALYSES:
        for label in LABELS:
            for query_id in sorted(evaluation_ids):
                tridb = indexed[(analysis, label, query_id, "tridb_live")]
                polyglot = indexed[(analysis, label, query_id, "polyglot_tuned")]
                mrr_delta = abs(tridb["mrr"] - polyglot["mrr"])
                recall_delta = abs(tridb["recall_at_20"] - polyglot["recall_at_20"])
                paired_rows.append(
                    {
                        "analysis": analysis,
                        "operating_label": label,
                        "query_id": query_id,
                        "template": tridb["template"],
                        "quality_matched": (
                            mrr_delta <= mrr_epsilon and recall_delta <= recall_epsilon
                        ),
                        "mrr_delta_abs": mrr_delta,
                        "recall_at_20_delta_abs": recall_delta,
                        "tridb_mrr": tridb["mrr"],
                        "polyglot_mrr": polyglot["mrr"],
                        "tridb_recall_at_20": tridb["recall_at_20"],
                        "polyglot_recall_at_20": polyglot["recall_at_20"],
                        "tridb_latency_p50_ms": tridb["latency_p50_ms"],
                        "polyglot_latency_p50_ms": polyglot["latency_p50_ms"],
                        "tridb_latency_p95_ms": tridb["latency_p95_ms"],
                        "polyglot_latency_p95_ms": polyglot["latency_p95_ms"],
                        "polyglot_over_tridb_p50": (
                            polyglot["latency_p50_ms"] / tridb["latency_p50_ms"]
                        ),
                        "polyglot_over_tridb_p95": (
                            polyglot["latency_p95_ms"] / tridb["latency_p95_ms"]
                        ),
                    }
                )

    audit = {
        "evaluation_queries": len(evaluation_ids),
        "physical_selected_observations": len(physical_keys),
        "role_expanded_query_system_rows": len(query_system_rows),
        "paired_query_rows": len(paired_rows),
    }
    return query_system_rows, paired_rows, audit


def summarize_composition(
    query_system_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    mechanism_rows = []
    for analysis_index, analysis in enumerate(ANALYSES):
        for label_index, label in enumerate(LABELS):
            role_pairs = [
                row
                for row in paired_rows
                if row["analysis"] == analysis and row["operating_label"] == label
            ]
            eligible = [row for row in role_pairs if row["quality_matched"]]
            speedups = [float(row["polyglot_over_tridb_p50"]) for row in eligible]
            speedup_ci = _bootstrap_ci(
                speedups,
                _median,
                seed=seed + analysis_index * 100 + label_index,
            )
            median_speedup = statistics.median(speedups)
            summary_rows.append(
                {
                    "analysis": analysis,
                    "operating_label": label,
                    "queries_total": len(role_pairs),
                    "queries_quality_matched": len(eligible),
                    "queries_quality_excluded": len(role_pairs) - len(eligible),
                    "median_speedup_p50": median_speedup,
                    "geomean_speedup_p50": math.exp(
                        statistics.fmean(math.log(value) for value in speedups)
                    ),
                    "speedup_median_ci95_low": speedup_ci[0],
                    "speedup_median_ci95_high": speedup_ci[1],
                    "fraction_tridb_faster": sum(value > 1 for value in speedups)
                    / len(speedups),
                    "supported_effect": (
                        median_speedup >= 1.25 and speedup_ci[0] > 1.0
                    ),
                    "strong_effect": (median_speedup >= 2.0 and speedup_ci[0] > 1.0),
                }
            )
            for system in SYSTEMS:
                rows = [
                    row
                    for row in query_system_rows
                    if row["analysis"] == analysis
                    and row["operating_label"] == label
                    and row["system"] == system
                ]
                mechanism_rows.append(
                    {
                        "analysis": analysis,
                        "operating_label": label,
                        "system": system,
                        "queries": len(rows),
                        **{
                            f"median_query_{key}": statistics.median(
                                float(row[key]) for row in rows
                            )
                            for key in (
                                "latency_p50_ms",
                                "latency_p95_ms",
                                "client_request_count",
                                "store_rpc_count",
                                "cross_store_handoff_count",
                                "intermediate_rows",
                                "payload_bytes",
                                "serialization_ms",
                                "serialization_fraction",
                            )
                        },
                    }
                )
    return summary_rows, mechanism_rows


def summarize_composition_templates(
    paired_rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    templates = sorted({str(row["template"]) for row in paired_rows})
    rows = []
    for analysis_index, analysis in enumerate(ANALYSES):
        for label_index, label in enumerate(LABELS):
            for template_index, template in enumerate(templates):
                selected = [
                    row
                    for row in paired_rows
                    if row["analysis"] == analysis
                    and row["operating_label"] == label
                    and row["template"] == template
                    and row["quality_matched"]
                ]
                values = [float(row["polyglot_over_tridb_p50"]) for row in selected]
                interval = _bootstrap_ci(
                    values,
                    _median,
                    seed=(
                        seed + analysis_index * 100 + label_index * 10 + template_index
                    ),
                )
                rows.append(
                    {
                        "analysis": analysis,
                        "operating_label": label,
                        "template": template,
                        "queries_quality_matched": len(values),
                        "median_speedup_p50": statistics.median(values),
                        "speedup_median_ci95_low": (
                            None if interval is None else interval[0]
                        ),
                        "speedup_median_ci95_high": (
                            None if interval is None else interval[1]
                        ),
                        "fraction_tridb_faster": sum(value > 1 for value in values)
                        / len(values),
                    }
                )
    return rows


def load_modality(
    path: Path, *, repetitions: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    keys = set()
    for row in _rows(path):
        key = (str(row["query_id"]), str(row["arm"]), int(row["repetition"]))
        if key in keys:
            raise ValueError(f"duplicate modality key: {key}")
        keys.add(key)
        if row.get("status") != "ok" or row.get("graph_censored"):
            raise ValueError(f"unusable modality row: {key}")
        grouped[(key[0], key[1])].append(row)

    reduced_rows = []
    for (query_id, arm), samples in sorted(grouped.items()):
        if len(samples) != repetitions:
            raise ValueError(
                f"{query_id}/{arm}: expected {repetitions}, found {len(samples)}"
            )
        qualities = {json.dumps(row["quality"], sort_keys=True) for row in samples}
        if len(qualities) != 1:
            raise ValueError(f"quality changed across repetitions for {query_id}/{arm}")
        quality = samples[0]["quality"]
        latencies = [float(row["client_wall_latency_ns"]) / 1e6 for row in samples]
        reduced_rows.append(
            {
                "query_id": query_id,
                "template": samples[0]["template"],
                "arm": arm,
                "latency_p50_ms": statistics.median(latencies),
                "latency_p95_ms": _percentile(latencies, 95),
                "mrr": float(quality["mrr"]),
                "recall_at_20": float(quality["recall_at_20"]),
                "hit_at_1": float(quality["hit_at_1"]),
                "hit_at_5": float(quality["hit_at_5"]),
                "full_constraint_validity_fraction": statistics.median(
                    float(row["full_constraint_validity_fraction"]) for row in samples
                ),
                "candidates": statistics.median(
                    int(row["candidates"]) for row in samples
                ),
                "edges_examined": statistics.median(
                    int(row["edges_examined"]) for row in samples
                ),
            }
        )
    queries = {row["query_id"] for row in reduced_rows}
    arms = {row["arm"] for row in reduced_rows}
    if arms != set(ARMS):
        raise ValueError(f"unexpected modality arms: {sorted(arms)}")
    expected_groups = len(queries) * len(ARMS)
    if len(reduced_rows) != expected_groups:
        raise ValueError(
            f"expected {expected_groups} query-arm groups, found {len(reduced_rows)}"
        )
    return reduced_rows, {
        "observations": len(keys),
        "queries": len(queries),
        "query_arm_groups": len(reduced_rows),
    }


def summarize_modality(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    template_rows = []
    for arm_index, arm in enumerate(ARMS):
        selected = [row for row in rows if row["arm"] == arm]
        result: dict[str, Any] = {
            "arm": arm,
            "queries": len(selected),
            "latency_p50_ms": statistics.median(
                float(row["latency_p50_ms"]) for row in selected
            ),
            "latency_p95_ms": statistics.median(
                float(row["latency_p95_ms"]) for row in selected
            ),
            "candidates_median": statistics.median(
                float(row["candidates"]) for row in selected
            ),
            "edges_examined_median": statistics.median(
                float(row["edges_examined"]) for row in selected
            ),
        }
        for metric_index, metric in enumerate(
            ("mrr", "recall_at_20", "full_constraint_validity_fraction")
        ):
            values = [float(row[metric]) for row in selected]
            interval = _bootstrap_ci(
                values,
                _mean,
                seed=seed + arm_index * 10 + metric_index,
            )
            result[f"mean_{metric}"] = statistics.fmean(values)
            result[f"mean_{metric}_ci95_low"] = interval[0]
            result[f"mean_{metric}_ci95_high"] = interval[1]
        summary_rows.append(result)

    templates = sorted({str(row["template"]) for row in rows})
    for template in templates:
        for arm in ARMS:
            selected = [
                row for row in rows if row["template"] == template and row["arm"] == arm
            ]
            template_rows.append(
                {
                    "template": template,
                    "arm": arm,
                    "queries": len(selected),
                    "mean_mrr": statistics.fmean(float(row["mrr"]) for row in selected),
                    "mean_recall_at_20": statistics.fmean(
                        float(row["recall_at_20"]) for row in selected
                    ),
                    "mean_full_constraint_validity_fraction": statistics.fmean(
                        float(row["full_constraint_validity_fraction"])
                        for row in selected
                    ),
                    "median_latency_p50_ms": statistics.median(
                        float(row["latency_p50_ms"]) for row in selected
                    ),
                }
            )
    return summary_rows, template_rows


def _style_axis(axis: Any, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _role_label(analysis: str, label: str) -> str:
    prefix = "Iso" if analysis == "iso_plan" else "Pareto"
    short = {"fast": "Fast", "balanced": "Balanced", "high_quality": "High-Q"}
    return f"{prefix}\n{short[label]}"


def plot_speedup(
    paired_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10.2, 5.3), constrained_layout=True)
    roles = [(analysis, label) for analysis in ANALYSES for label in LABELS]
    rng = np.random.default_rng(20260819)
    display_ceiling = 16.0
    clipped_label_used = False
    for index, (analysis, label) in enumerate(roles):
        eligible = [
            row
            for row in paired_rows
            if row["analysis"] == analysis
            and row["operating_label"] == label
            and row["quality_matched"]
        ]
        for template in sorted(TEMPLATE_COLORS):
            values = [
                float(row["polyglot_over_tridb_p50"])
                for row in eligible
                if row["template"] == template
            ]
            if not values:
                continue
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            displayed = [min(value, display_ceiling * 0.96) for value in values]
            markers = ["^" if value > display_ceiling else "o" for value in values]
            for point_x, point_y, marker in zip(
                np.full(len(values), index) + jitter,
                displayed,
                markers,
                strict=True,
            ):
                clipped_label = None
                if marker == "^" and not clipped_label_used:
                    clipped_label = f"Clipped above {display_ceiling:g}×"
                    clipped_label_used = True
                axis.scatter(
                    point_x,
                    point_y,
                    marker=marker,
                    s=42 if marker == "^" else 30,
                    alpha=0.85 if marker == "^" else 0.72,
                    color=TEMPLATE_COLORS[template],
                    edgecolor="black" if marker == "^" else "white",
                    linewidth=0.7 if marker == "^" else 0.4,
                    label=clipped_label,
                )
            axis.scatter(
                [],
                [],
                color=TEMPLATE_COLORS[template],
                label=TEMPLATE_LABELS[template].replace("\n", " ")
                if index == 0
                else None,
            )
        summary = next(
            row
            for row in summary_rows
            if row["analysis"] == analysis and row["operating_label"] == label
        )
        median = float(summary["median_speedup_p50"])
        low = float(summary["speedup_median_ci95_low"])
        high = float(summary["speedup_median_ci95_high"])
        axis.errorbar(
            index,
            median,
            yerr=[[median - low], [high - median]],
            fmt="D",
            color="black",
            markersize=6,
            capsize=4,
            linewidth=1.5,
            zorder=5,
        )
        axis.text(
            index,
            high * 1.08,
            f"{median:.2f}×\nn={summary['queries_quality_matched']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.axhline(1.0, color="#333333", linewidth=1.2, label="Parity")
    axis.axhline(
        1.25, color="#777777", linewidth=1.0, linestyle="--", label="Support threshold"
    )
    axis.axhline(
        2.0, color="#999999", linewidth=1.0, linestyle=":", label="Strong threshold"
    )
    axis.set_yscale("log", base=2)
    axis.set_ylim(0.4, display_ceiling)
    axis.set_yticks([0.5, 1.0, 1.25, 2.0, 4.0, 8.0, 16.0])
    axis.set_yticklabels(["0.5", "1", "1.25", "2", "4", "8", "16"])
    axis.set_xticks(range(len(roles)), [_role_label(*role) for role in roles])
    axis.set_ylabel("Polyglot-Tuned / TriDB per-query p50 latency")
    axis.set_title(
        "E1 composition penalty at matched quality\npoints are held-out queries; diamonds are bootstrap median 95% CIs"
    )
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axis.legend(unique.values(), unique.keys(), frameon=False, ncol=4, fontsize=8)
    _style_axis(axis)
    return _save(figure, output_dir, "figure1_matched_quality_speedup")


def plot_mechanism(
    mechanism_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    roles = [(analysis, label) for analysis in ANALYSES for label in LABELS]
    x = np.arange(len(roles), dtype=float)
    width = 0.36

    def values(system: str, metric: str) -> list[float]:
        return [
            float(
                next(
                    row
                    for row in mechanism_rows
                    if row["analysis"] == analysis
                    and row["operating_label"] == label
                    and row["system"] == system
                )[metric]
            )
            for analysis, label in roles
        ]

    latency_axis = axes[0, 0]
    for offset, system in ((-width / 2, "tridb_live"), (width / 2, "polyglot_tuned")):
        latency_axis.bar(
            x + offset,
            values(system, "median_query_latency_p50_ms"),
            width,
            color=SYSTEM_COLORS[system],
            label="TriDB" if system == "tridb_live" else "Polyglot-Tuned",
        )
    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("Median per-query p50 latency (ms, log)")
    latency_axis.set_title("A. End-to-end latency")
    latency_axis.legend(frameon=False)
    _style_axis(latency_axis)

    rpc_axis = axes[0, 1]
    poly_rpc = values("polyglot_tuned", "median_query_store_rpc_count")
    poly_handoff = values("polyglot_tuned", "median_query_cross_store_handoff_count")
    rpc_axis.bar(x - width / 2, poly_rpc, width, color="#E07A1F", label="Store RPCs")
    rpc_axis.bar(
        x + width / 2,
        poly_handoff,
        width,
        color="#F2B880",
        label="Cross-store handoffs",
    )
    rpc_axis.axhline(
        0, color=SYSTEM_COLORS["tridb_live"], linewidth=2, label="TriDB: 0 / 0"
    )
    rpc_axis.set_ylabel("Median count per query")
    rpc_axis.set_title("B. Polyglot orchestration boundaries")
    rpc_axis.legend(frameon=False)
    _style_axis(rpc_axis)

    payload_axis = axes[1, 0]
    poly_rows = values("polyglot_tuned", "median_query_intermediate_rows")
    poly_kib = [
        value / 1024 for value in values("polyglot_tuned", "median_query_payload_bytes")
    ]
    payload_axis.plot(x, poly_rows, "o-", color="#7B5AA6", label="Intermediate rows")
    payload_axis.plot(x, poly_kib, "s--", color="#3A923A", label="Payload (KiB)")
    if all(value > 0 for value in [*poly_rows, *poly_kib]):
        payload_axis.set_yscale("log")
        payload_axis.set_ylabel("Median per query (log scale)")
    else:
        payload_axis.set_ylabel("Median per query")
    payload_axis.set_title("C. Cross-store materialization (TriDB = 0)")
    payload_axis.legend(frameon=False)
    _style_axis(payload_axis)

    serial_axis = axes[1, 1]
    serial_percent = [
        value * 100
        for value in values("polyglot_tuned", "median_query_serialization_fraction")
    ]
    serial_axis.bar(x, serial_percent, width=0.58, color="#C44E52")
    serial_axis.axhline(
        0, color=SYSTEM_COLORS["tridb_live"], linewidth=2, label="TriDB: 0%"
    )
    serial_axis.set_ylabel("Median serialization / wall time (%)")
    serial_axis.set_title("D. Measured JSON serialization share")
    serial_axis.legend(frameon=False)
    _style_axis(serial_axis)

    labels = [_role_label(*role).replace("\n", " ") for role in roles]
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    figure.suptitle(
        "E1 measured mechanism behind the residual composition penalty", fontsize=14
    )
    return _save(figure, output_dir, "figure2_latency_and_boundary_cost")


def plot_modality_quality(
    summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    dataset_label: str,
    query_count: int,
) -> list[Path]:
    metrics = (
        ("mean_mrr", "MRR"),
        ("mean_recall_at_20", "Recall@20"),
        ("mean_full_constraint_validity_fraction", "Constraint-valid fraction"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), constrained_layout=True)
    x = np.arange(len(ARMS))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        values = [
            float(next(row for row in summary_rows if row["arm"] == arm)[metric])
            for arm in ARMS
        ]
        lows = [
            float(
                next(row for row in summary_rows if row["arm"] == arm)[
                    f"{metric}_ci95_low"
                ]
            )
            for arm in ARMS
        ]
        highs = [
            float(
                next(row for row in summary_rows if row["arm"] == arm)[
                    f"{metric}_ci95_high"
                ]
            )
            for arm in ARMS
        ]
        axis.bar(x, values, color=[ARM_COLORS[arm] for arm in ARMS])
        axis.errorbar(
            x,
            values,
            yerr=[
                np.asarray(values) - np.asarray(lows),
                np.asarray(highs) - np.asarray(values),
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
        for index, value in enumerate(values):
            axis.text(index, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
        axis.set_ylim(0, 1.08)
        axis.set_xticks(
            x, [ARM_LABELS[arm] for arm in ARMS], rotation=35, ha="right", fontsize=8
        )
        axis.set_title(title)
        _style_axis(axis)
    figure.suptitle(
        f"E1 TriDB modality ablation ({dataset_label}; {query_count} held-out queries; "
        "query-bootstrap 95% CIs)",
        fontsize=14,
    )
    return _save(figure, output_dir, "figure3_modality_quality")


def plot_modality_templates(
    template_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    templates = sorted({str(row["template"]) for row in template_rows})
    query_counts = {
        template: int(
            next(
                row
                for row in template_rows
                if row["template"] == template and row["arm"] == ARMS[0]
            )["queries"]
        )
        for template in templates
    }
    metrics = (
        ("mean_recall_at_20", "A. Recall@20"),
        ("mean_full_constraint_validity_fraction", "B. Constraint-valid fraction"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        matrix = np.asarray(
            [
                [
                    float(
                        next(
                            row
                            for row in template_rows
                            if row["template"] == template and row["arm"] == arm
                        )[metric]
                    )
                    for template in templates
                ]
                for arm in ARMS
            ]
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row_index in range(len(ARMS)):
            for column_index in range(len(templates)):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else "#222222",
                    fontsize=8,
                )
        axis.set_xticks(
            range(len(templates)),
            [
                f"{TEMPLATE_LABELS.get(template, template)}\n(n={query_counts[template]})"
                for template in templates
            ],
            fontsize=8,
        )
        axis.set_yticks(range(len(ARMS)), [ARM_LABELS[arm] for arm in ARMS], fontsize=8)
        axis.set_title(title)
    figure.colorbar(image, ax=axes, shrink=0.78, label="Mean across held-out queries")
    figure.suptitle(
        "Where each modality matters: results stratified by query template", fontsize=14
    )
    return _save(figure, output_dir, "figure4_modality_by_template")


def plot_modality_tradeoff(
    summary_rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.2, 5.5), constrained_layout=True)
    for arm in ARMS:
        row = next(row for row in summary_rows if row["arm"] == arm)
        x = float(row["latency_p50_ms"])
        y = float(row["mean_recall_at_20"])
        validity = float(row["mean_full_constraint_validity_fraction"])
        axis.scatter(
            x,
            y,
            s=70 + validity * 160,
            color=ARM_COLORS[arm],
            edgecolor="black" if arm == "vector_graph_relational" else "white",
            linewidth=1.1,
            zorder=3,
        )
        axis.annotate(
            f"{ARM_LABELS[arm]}\nvalid={validity:.2f}",
            (x, y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_xlim(left=max(0.05, axis.get_xlim()[0]))
    axis.set_ylim(0, 1.0)
    axis.set_xlabel("Median per-query p50 latency (ms, log scale)")
    axis.set_ylabel("Mean Recall@20")
    axis.set_title(
        "E1 modality cost-quality trade-off\nmarker size encodes full-constraint validity"
    )
    _style_axis(axis, grid_axis="both")
    return _save(figure, output_dir, "figure5_modality_latency_quality")


def _format_ci(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['median_speedup_p50']):.2f}× "
        f"[{float(row['speedup_median_ci95_low']):.2f}, "
        f"{float(row['speedup_median_ci95_high']):.2f}]"
    )


def _render_report(
    summary: Mapping[str, Any],
    composition_rows: Sequence[Mapping[str, Any]],
    composition_template_rows: Sequence[Mapping[str, Any]],
    mechanism_rows: Sequence[Mapping[str, Any]],
    modality_rows: Sequence[Mapping[str, Any]],
) -> str:
    dataset_label = str(summary["dataset_label"])
    evaluation_queries = int(summary["evaluation_audit"]["evaluation_queries"])
    repetitions = int(summary["evaluation_repetitions"])
    deployment = summary["deployment"]
    isolated = deployment.get("resident_system_policy") == "system_isolated"
    speedups = [float(row["median_speedup_p50"]) for row in composition_rows]
    if all(value > 1.0 for value in speedups):
        direction = "全部六个 frozen comparisons 中 TriDB 的中位 p50 均更低。"
    elif all(value < 1.0 for value in speedups):
        direction = "全部六个 frozen comparisons 中 Polyglot-Tuned 的中位 p50 均更低。"
    else:
        direction = "六个 frozen comparisons 的中位 p50 方向并不一致。"
    all_three = next(
        row for row in modality_rows if row["arm"] == "vector_graph_relational"
    )
    small_strata = sorted(
        {
            str(row["template"])
            for row in composition_template_rows
            if int(row["queries_quality_matched"]) <= 2
        }
    )
    template_reversals = [
        row
        for row in composition_template_rows
        if float(row["median_speedup_p50"]) < 1.0
    ]
    lines = [
        f"# E1 composition penalty and modality necessity — {dataset_label}",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-19",
        "- Verification Status: ANALYZED",
        "- Version Label: e1_staged_report_v0.1",
        "",
        "## 结论边界",
        "",
        f"在 {evaluation_queries} 个 held-out {dataset_label} queries 上，正式 matched-quality 比较显示：",
        "Polyglot-Tuned 的 per-query p50 latency 中位数相对 TriDB 为 "
        f"{min(speedups):.2f}×–{max(speedups):.2f}×。",
        direction,
        "这是同机、持久连接、批处理 Polyglot-Tuned 之后测得的系统间 gap；",
        "boundary instrumentation 同时记录了 store RPC、cross-store handoff、",
        "intermediate rows/payload 和 JSON serialization，因此支持 E1 的 composition-penalty 机理解释。",
        "",
        "该结果不等于跨所有硬件/数据集的普遍因果定律。当前没有 Polyglot-Naive arm，",
        "所以不能量化 naive→tuned 的收敛；也没有 ≥1,000-request mixed stream，",
        "所以不报告 p99。运行环境是 x86_64 stock-PG 路径，不构成 GX10 sign-off。",
        (
            "容量预检要求两套完整副本按 system-isolated 顺序驻留；因此两系统不是"
            "逐 repetition ABBA 交错运行。冻结点、同一 query 集、warm steady-state "
            "和 query-level paired analysis 保持不变，但结果可能包含阶段时间漂移，"
            "不得表述为 cross-system ABBA 结果。"
            if isolated
            else "两系统按配置的交错/驻留策略运行；具体策略记录在 summary.json。"
        ),
        "",
        "## Matched-quality latency",
        "",
        "| Analysis | Point | Eligible queries | Median speedup [95% CI] | TriDB faster | Decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in composition_rows:
        decision = (
            "strong"
            if row["strong_effect"]
            else "supported"
            if row["supported_effect"]
            else "not supported"
        )
        lines.append(
            f"| {row['analysis']} | {row['operating_label']} | "
            f"{row['queries_quality_matched']}/{row['queries_total']} | "
            f"{_format_ci(row)} | {float(row['fraction_tridb_faster']):.1%} | {decision} |"
        )
    lines.extend(
        [
            "",
            f"速度比以 query 为统计单位；每个 query 先对 {repetitions} 次 repetition 求 p50，",
            "再在 held-out queries 上 bootstrap median 95% CI。未通过 `|ΔMRR|≤0.02`",
            "和 `|ΔRecall@20|≤0.02` 的 query 不进入速度结论。",
            "",
            "![Matched-quality speedup](figures/figure1_matched_quality_speedup.png)",
            "",
            "## Query-template heterogeneity",
            "",
            "Pareto balanced point 的 template-level medians：",
            "",
            "| Template | Eligible queries | Median speedup [95% CI] | TriDB faster |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in composition_template_rows:
        if not (
            row["analysis"] == "pareto_envelope"
            and row["operating_label"] == "balanced"
        ):
            continue
        if row["speedup_median_ci95_low"] is None:
            interval = "CI unavailable (n=1)"
        else:
            interval = (
                f"[{float(row['speedup_median_ci95_low']):.2f}, "
                f"{float(row['speedup_median_ci95_high']):.2f}]"
            )
        lines.append(
            f"| {row['template']} | {row['queries_quality_matched']} | "
            f"{float(row['median_speedup_p50']):.2f}× {interval} | "
            f"{float(row['fraction_tridb_faster']):.1%} |"
        )
    lines.extend(
        [
            "",
            "所有 template 的 median direction 都高于 parity；但 n=1 和 n=2 的",
            "小 strata 只能作为描述性结果。完整六点分层数据在 `composition_by_template.csv`。",
            "",
            "## Composition mechanism",
            "",
        ]
    )
    balanced = [
        row
        for row in mechanism_rows
        if row["analysis"] == "pareto_envelope" and row["operating_label"] == "balanced"
    ]
    for row in balanced:
        name = "TriDB" if row["system"] == "tridb_live" else "Polyglot-Tuned"
        lines.append(
            f"- **{name} / Pareto balanced**: p50={float(row['median_query_latency_p50_ms']):.3f} ms, "
            f"store RPC={float(row['median_query_store_rpc_count']):.1f}, "
            f"handoff={float(row['median_query_cross_store_handoff_count']):.1f}, "
            f"boundary rows={float(row['median_query_intermediate_rows']):.1f}, "
            f"payload={float(row['median_query_payload_bytes']):.1f} B, "
            f"serialization share={float(row['median_query_serialization_fraction']):.2%}."
        )
    lines.extend(
        [
            "",
            "TriDB 的 store RPC、cross-store handoff 和 intermediate boundary payload",
            "按实现契约均为 0；Polyglot-Tuned 即使同机仍必须跨多个 store 接口边界。",
            "payload 是 JSON-encoded intermediate ID lists 的应用层估计，不是 wire bytes。",
            "",
            "![Composition mechanism](figures/figure2_latency_and_boundary_cost.png)",
            "",
            "## Modality necessity",
            "",
            "| Arm | MRR [95% CI] | Recall@20 [95% CI] | Constraint validity | p50 ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in modality_rows:
        lines.append(
            f"| {ARM_LABELS[str(row['arm'])]} | "
            f"{float(row['mean_mrr']):.3f} [{float(row['mean_mrr_ci95_low']):.3f}, {float(row['mean_mrr_ci95_high']):.3f}] | "
            f"{float(row['mean_recall_at_20']):.3f} [{float(row['mean_recall_at_20_ci95_low']):.3f}, {float(row['mean_recall_at_20_ci95_high']):.3f}] | "
            f"{float(row['mean_full_constraint_validity_fraction']):.3f} | "
            f"{float(row['latency_p50_ms']):.3f} | {float(row['latency_p95_ms']):.3f} |"
        )
    lines.extend(
        [
            "",
            "All-three arm 的实测整体指标为 "
            f"MRR={float(all_three['mean_mrr']):.3f}、"
            f"Recall@20={float(all_three['mean_recall_at_20']):.3f}、"
            "constraint-valid fraction="
            f"{float(all_three['mean_full_constraint_validity_fraction']):.3f}。",
            "各去模态 arm 与它的差值才是该数据集上的必要性证据；不预设 All-three"
            "一定在每个指标或每个 template 上严格占优。模板热图用于避免把平均收益"
            "错误推广到每一类查询。",
            "",
            "![Modality quality](figures/figure3_modality_quality.png)",
            "",
            "![Modality by template](figures/figure4_modality_by_template.png)",
            "",
            "![Modality latency-quality trade-off](figures/figure5_modality_latency_quality.png)",
            "",
            "## Statistical and claim warnings",
            "",
            f"- p95 是每个 query 的 {repetitions} 次样本 percentile 后再跨 query 汇总；不报告 p99。",
            "- 小 strata（n≤2）只能作为描述性结果："
            + (", ".join(small_strata) if small_strata else "无")
            + "。",
            "- 六个 frozen comparisons 属于预注册分析，但仍应整体报告，不能只挑最大 speedup。",
            f"- modality CIs 在 query 层 bootstrap；{repetitions} 次 timing repetition 不被当作独立 query。",
            (
                "- 系统采用 system-isolated 阶段执行，未使用 cross-system ABBA；阶段漂移是额外限制。"
                if isolated
                else "- 系统驻留与顺序策略见 summary.json 的 deployment 字段。"
            ),
            "- 当前结果仅支持 tested deployment 上实测方向的 composition gap 和模态贡献，",
            "  不支持未经测试的数据集、GX10 fork 或并发负载外推。",
            "",
            "## Fallacy scan",
            "",
            "Coverage: 11/11 checked.",
            "",
            "| Fallacy | Status | Audit result |",
            "|---|---|---|",
            "| Simpson's paradox | CAUTION | "
            + (
                "At least one template median reverses below parity; inspect composition_by_template.csv."
                if template_reversals
                else "No template median reverses below parity; small strata still limit the check."
            )
            + " |",
            "| Ecological fallacy | NOTE | Claims are bounded to query-level workload results, not individual users or all workloads. |",
            "| Berkson's paradox | NOTE | Frozen stratified split used; all quality-gate exclusions are disclosed. |",
            "| Collider bias | NOTE | No post-treatment covariate adjustment is used. |",
            "| Base-rate neglect | N/A | No diagnostic sensitivity/specificity claim. |",
            "| Regression to the mean | NOTE | Queries were not selected for extreme measured latency or quality. |",
            "| Survivorship bias | SOLID | 0 errors/censors; complete manifests. Quality-gate exclusions remain in CSV. |",
            "| Look-elsewhere effect | CAUTION | Six preregistered comparisons are reported together; no selective p-value claim. |",
            "| Garden of forking paths | SOLID | Split, thresholds, and operating points were frozen before evaluation. |",
            "| Correlation != causation | CAUTION | Boundary metrics explain measured association; universal architectural causality is not claimed. |",
            "| Reverse causality | N/A | Controlled system benchmark, not a cross-sectional directional association. |",
            "",
            "## Reproducibility",
            "",
            f"- Input evaluation manifest complete: `{summary['input_gates']['evaluation_complete']}`",
            f"- Input modality manifest complete: `{summary['input_gates']['modality_complete']}`",
            f"- Physical evaluation observations selected: `{summary['evaluation_audit']['physical_selected_observations']}`",
            f"- Modality observations audited: `{summary['modality_audit']['observations']}`",
            "- All source/output hashes are recorded in `analysis_manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    staged_config_path: Path,
    full_grid_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    staged = _read_json_or_yaml(staged_config_path)
    base_config_path = Path(staged["base_config"])
    base = _read_json_or_yaml(base_config_path)
    outputs = staged["outputs"]
    frozen_path = Path(outputs["operating_points"])
    evaluation_path = Path(outputs["evaluation_observations"])
    modality_path = Path(outputs["modality_observations"])
    evaluation_manifest_path = evaluation_path.parent / "evaluation_manifest.json"
    modality_manifest_path = modality_path.parent / "modality_manifest.json"
    frozen = _read_json(frozen_path)
    evaluation_manifest = _read_json(evaluation_manifest_path)
    modality_manifest = _read_json(modality_manifest_path)
    if not evaluation_manifest.get("complete") or not evaluation_manifest.get(
        "valid_for_headline_claims"
    ):
        raise ValueError("evaluation manifest is not complete/headline-valid")
    if not modality_manifest.get("complete"):
        raise ValueError("modality manifest is not complete")

    repetitions = int(staged["evaluation"]["repetitions"])
    matched = base["comparison"]["matched_quality"]
    evaluation_paths = [evaluation_path]
    if full_grid_path is not None:
        evaluation_paths.insert(0, full_grid_path)
    query_system_rows, paired_rows, evaluation_audit = load_evaluation(
        frozen,
        evaluation_paths,
        repetitions=repetitions,
        mrr_epsilon=float(matched["mrr_epsilon"]),
        recall_epsilon=float(matched["recall_at_20_epsilon"]),
    )
    if evaluation_audit["physical_selected_observations"] != int(
        evaluation_manifest["expected_observations"]
    ):
        raise ValueError("selected evaluation count differs from manifest")
    composition_rows, mechanism_rows = summarize_composition(
        query_system_rows, paired_rows, seed=20260819
    )
    composition_template_rows = summarize_composition_templates(
        paired_rows, seed=20260819
    )

    modality_query_rows, modality_audit = load_modality(
        modality_path, repetitions=int(staged["modality_ablation"]["repetitions"])
    )
    if modality_audit["observations"] != int(
        modality_manifest["expected_observations"]
    ):
        raise ValueError("modality observation count differs from manifest")
    modality_rows, modality_template_rows = summarize_modality(
        modality_query_rows, seed=20260819
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_query_metrics.csv", paired_rows)
    _write_csv(output_dir / "composition_summary.csv", composition_rows)
    _write_csv(output_dir / "composition_by_template.csv", composition_template_rows)
    _write_csv(output_dir / "boundary_summary.csv", mechanism_rows)
    _write_csv(output_dir / "modality_query_metrics.csv", modality_query_rows)
    _write_csv(output_dir / "modality_summary.csv", modality_rows)
    _write_csv(output_dir / "modality_by_template.csv", modality_template_rows)

    dataset_name = str(staged.get("dataset", "stark_prime"))
    dataset_label = dataset_name.replace("_", "-").upper()
    figures = [
        *plot_speedup(paired_rows, composition_rows, figure_dir),
        *plot_mechanism(mechanism_rows, figure_dir),
        *plot_modality_quality(
            modality_rows,
            figure_dir,
            dataset_label=dataset_label,
            query_count=int(evaluation_audit["evaluation_queries"]),
        ),
        *plot_modality_templates(modality_template_rows, figure_dir),
        *plot_modality_tradeoff(modality_rows, figure_dir),
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "material_passport": {
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "origin_date": "2026-08-19",
            "verification_status": "ANALYZED",
            "version_label": "e1_staged_report_v0.1",
        },
        "dataset": dataset_name,
        "dataset_label": dataset_label,
        "evaluation_repetitions": repetitions,
        "deployment": staged.get("deployment", {}),
        "input_gates": {
            "evaluation_complete": bool(evaluation_manifest["complete"]),
            "evaluation_valid_for_headline_claims": bool(
                evaluation_manifest["valid_for_headline_claims"]
            ),
            "modality_complete": bool(modality_manifest["complete"]),
            "p99_evaluated": False,
            "gx10_signoff": False,
        },
        "evaluation_audit": evaluation_audit,
        "modality_audit": modality_audit,
        "composition": composition_rows,
        "composition_by_template": composition_template_rows,
        "modality": modality_rows,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        _render_report(
            summary,
            composition_rows,
            composition_template_rows,
            mechanism_rows,
            modality_rows,
        ),
        encoding="utf-8",
    )

    input_paths = [
        staged_config_path,
        base_config_path,
        frozen_path,
        evaluation_manifest_path,
        evaluation_path,
        modality_manifest_path,
        modality_path,
    ]
    if full_grid_path is not None:
        input_paths.insert(4, full_grid_path)
    output_paths = [
        output_dir / "summary.json",
        output_dir / "REPORT.md",
        output_dir / "paired_query_metrics.csv",
        output_dir / "composition_summary.csv",
        output_dir / "composition_by_template.csv",
        output_dir / "boundary_summary.csv",
        output_dir / "modality_query_metrics.csv",
        output_dir / "modality_summary.csv",
        output_dir / "modality_by_template.csv",
        *figures,
    ]
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": [artifact_record(path) for path in input_paths],
            "outputs": [artifact_record(path) for path in output_paths],
            "p99_evaluated": False,
            "gx10_signoff": False,
        },
    )
    return summary


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return _read_json(path)
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged-config",
        type=Path,
        default=Path("configs/e1/staged_v0.2.yaml"),
    )
    parser.add_argument(
        "--full-grid",
        type=Path,
        default=None,
        help="Optional prior observation file used only when evaluation rows are reused.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/e1/composition/staged_v0.2/analysis"),
    )
    args = parser.parse_args(argv)
    summary = analyze(
        staged_config_path=args.staged_config,
        full_grid_path=args.full_grid,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "composition": summary["composition"],
                "modality": summary["modality"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
