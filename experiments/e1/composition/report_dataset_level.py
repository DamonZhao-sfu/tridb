"""Build a non-pooled, dataset-level E1 comparison from completed reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.e1.composition.report_staged import (
    ANALYSES,
    ARMS,
    ARM_LABELS,
    LABELS,
    _role_label,
)
from tools.e0.common import artifact_record, write_json

SCHEMA_VERSION = "e1-dataset-level-report-v0.1.0"
METRICS = (
    ("mean_mrr", "MRR"),
    ("mean_recall_at_20", "Recall@20"),
    ("mean_full_constraint_validity_fraction", "Constraint validity"),
)
COLORS = ("#2878B5", "#E07A1F", "#3A923A", "#7B5AA6")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _normalize(
    inputs: Sequence[tuple[str, Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = []
    composition = []
    modality_delta = []
    for name, path in inputs:
        summary = _read(path)
        query_count = int(summary["evaluation_audit"]["evaluation_queries"])
        datasets.append(
            {
                "dataset": name,
                "evaluation_queries": query_count,
                "summary": str(path),
            }
        )
        for row in summary["composition"]:
            composition.append(
                {
                    "dataset": name,
                    "evaluation_queries": query_count,
                    **row,
                }
            )
        modality = {str(row["arm"]): row for row in summary["modality"]}
        missing = sorted(set(ARMS) - set(modality))
        if missing:
            raise ValueError(f"{path} is missing modality arms: {missing}")
        all_three = modality["vector_graph_relational"]
        for arm in ARMS:
            for metric, label in METRICS:
                full_value = float(all_three[metric])
                arm_value = float(modality[arm][metric])
                modality_delta.append(
                    {
                        "dataset": name,
                        "evaluation_queries": query_count,
                        "arm": arm,
                        "metric": metric,
                        "metric_label": label,
                        "all_three_value": full_value,
                        "arm_value": arm_value,
                        "drop_from_all_three": full_value - arm_value,
                    }
                )
    return datasets, composition, modality_delta


def _plot_composition(
    datasets: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    roles = [(analysis, label) for analysis in ANALYSES for label in LABELS]
    x = np.arange(len(roles), dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    for index, dataset in enumerate(datasets):
        name = str(dataset["dataset"])
        selected = [
            next(
                row
                for row in rows
                if row["dataset"] == name
                and row["analysis"] == analysis
                and row["operating_label"] == label
            )
            for analysis, label in roles
        ]
        values = np.asarray([float(row["median_speedup_p50"]) for row in selected])
        lows = np.asarray([float(row["speedup_median_ci95_low"]) for row in selected])
        highs = np.asarray([float(row["speedup_median_ci95_high"]) for row in selected])
        offset = (index - (len(datasets) - 1) / 2) * 0.10
        axis.errorbar(
            x + offset,
            values,
            yerr=[values - lows, highs - values],
            fmt="o-",
            capsize=3,
            linewidth=1.5,
            color=COLORS[index % len(COLORS)],
            label=f"{name} (n={dataset['evaluation_queries']})",
        )
    axis.axhline(1.0, color="#333333", linewidth=1.1, label="Parity")
    axis.axhline(1.25, color="#888888", linestyle="--", linewidth=0.9)
    axis.set_yscale("log", base=2)
    axis.set_xticks(x, [_role_label(*role).replace("\n", " ") for role in roles])
    axis.set_ylabel("Polyglot-Tuned / TriDB per-query p50")
    axis.set_title("E1 dataset-level matched-quality latency (datasets are not pooled)")
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.legend(frameon=False)
    return _save(figure, output_dir, "figure1_dataset_level_latency")


def _plot_modality_delta(
    datasets: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    arms = [arm for arm in ARMS if arm != "vector_graph_relational"]
    x = np.arange(len(arms), dtype=float)
    width = 0.8 / len(datasets)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        for index, dataset in enumerate(datasets):
            name = str(dataset["dataset"])
            values = [
                float(
                    next(
                        row
                        for row in rows
                        if row["dataset"] == name
                        and row["arm"] == arm
                        and row["metric"] == metric
                    )["drop_from_all_three"]
                )
                for arm in arms
            ]
            offset = (index - (len(datasets) - 1) / 2) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[index % len(COLORS)],
                label=name,
            )
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_xticks(x, [ARM_LABELS[arm] for arm in arms], rotation=35, ha="right")
        axis.set_ylabel(f"All-three minus arm ({label})")
        axis.set_title(label)
        axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axes[0].legend(frameon=False)
    figure.suptitle(
        "E1 modality necessity by dataset (positive values mean quality lost when modalities are removed)"
    )
    return _save(figure, output_dir, "figure2_dataset_level_modality_drop")


def build(inputs: Sequence[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("dataset-level comparison requires at least two datasets")
    names = [name for name, _ in inputs]
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    datasets, composition, modality_delta = _normalize(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        *_plot_composition(datasets, composition, output_dir),
        *_plot_modality_delta(datasets, modality_delta, output_dir),
    ]
    composition_path = output_dir / "dataset_level_composition.csv"
    modality_path = output_dir / "dataset_level_modality_delta.csv"
    _write_csv(composition_path, composition)
    _write_csv(modality_path, modality_delta)
    report = [
        "# E1 dataset-level TriDB vs Polyglot-Tuned and modality comparison",
        "",
        "每个 dataset 独立做 query-level reduction、matched-quality gate 与 bootstrap；"
        "本报告不把不同 dataset 的 queries 混池。",
        "",
        "![Dataset-level latency](figure1_dataset_level_latency.png)",
        "",
        "![Dataset-level modality loss](figure2_dataset_level_modality_drop.png)",
        "",
        "正的 modality drop 表示去掉相应模态后指标低于 All-three；零或负值不构成该模态"
        "在该平均指标上的必要性证据。数据明细见两个 CSV。",
        "",
        "MAG 的 system-isolated 结果没有逐 repetition cross-system ABBA；因此跨 dataset"
        "复现可以增强外部有效性，但不能消除阶段漂移这一限制。",
        "",
    ]
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "datasets": datasets,
        "composition": composition,
        "modality_delta": modality_delta,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "inputs": [artifact_record(path) for _, path in inputs],
            "outputs": [
                artifact_record(path)
                for path in [
                    summary_path,
                    report_path,
                    composition_path,
                    modality_path,
                    *figures,
                ]
            ],
            "datasets_pooled": False,
            "gx10_signoff": False,
        },
    )
    return summary


def _dataset_arg(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("dataset must be NAME=SUMMARY_JSON")
    return name, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_dataset_arg, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/e1/composition/dataset_level_v0.1"),
    )
    args = parser.parse_args(argv)
    summary = build(args.dataset, args.output_dir)
    print(json.dumps({"datasets": summary["datasets"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
