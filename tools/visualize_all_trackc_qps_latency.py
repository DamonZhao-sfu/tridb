#!/usr/bin/env python3
"""Visualize selected Track C Search/Native-Add receipts across QPS.

The controlled Qwen3-32B cohort is plotted separately from sparse Qwen3.8
exploratory points.  Missing and zero-success points remain visible in the
coverage matrix instead of being silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.export_all_trackc_latency_xlsx import (  # noqa: E402
    receipt_row,
    timeout_count,
)


DEFAULT_SOURCE = REPO / "bench/out"
DEFAULT_OUTPUT = REPO / "haikaidocs/qps_latency_visualizations_2026-08-22"
CONTROLLED = "Qwen3-32B-FP8 controlled"
SYSTEM_ORDER = [
    "TriDB/GEM",
    "Mem0",
    "MemOS",
    "Cognee",
    "Mandol",
    "Graphiti (Zep OSS proxy; not production Zep)",
    "EverMemOS (EverOS OSS)",
    "EverMemOS (paper-era official-network fork)",
]
SHORT_LABELS = {
    "Graphiti (Zep OSS proxy; not production Zep)": "Graphiti (Zep OSS proxy)",
    "EverMemOS (EverOS OSS)": "EverMemOS (EverOS OSS)",
    "EverMemOS (paper-era official-network fork)": "EverMemOS (paper-era)",
}
COLORS = {
    "TriDB/GEM": "#0072B2",
    "Mem0": "#D55E00",
    "MemOS": "#CC79A7",
    "Cognee": "#009E73",
    "Mandol": "#E69F00",
    "Graphiti (Zep OSS proxy; not production Zep)": "#7F7F7F",
    "EverMemOS (EverOS OSS)": "#56B4E9",
    "EverMemOS (paper-era official-network fork)": "#6A3D9A",
}


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def selected_rows(source: Path) -> list[dict[str, Any]]:
    rows = [
        receipt_row(path, source)
        for path in sorted(source.glob("**/run_receipt.json"))
    ]
    return [row for row in rows if row["selected"]]


def metric(row: dict[str, Any], name: str) -> float | None:
    summary = row["summary"]
    if name == "service_mean_s":
        value = nested(summary, "service_latency_ms", "mean")
        return None if value is None else float(value) / 1000.0
    if name == "user_visible_p99_s":
        value = nested(summary, "user_visible_latency_ms", "p99")
        return None if value is None else float(value) / 1000.0
    if name == "success_rate_pct":
        value = summary.get("success_rate")
        return None if value is None else float(value) * 100.0
    if name == "completion_qps":
        value = summary.get("completion_throughput_qps") or summary.get(
            "actual_qps"
        )
        return None if value is None else float(value)
    raise KeyError(name)


def write_csv(output: Path, rows: list[dict[str, Any]]) -> Path:
    path = output / "selected_qps_metrics.csv"
    headers = [
        "system",
        "workload",
        "target_qps",
        "model_cohort",
        "status",
        "formal_total",
        "successful",
        "failed",
        "timeouts",
        "success_rate",
        "service_mean_s_success_only",
        "service_p99_s_success_only",
        "user_visible_p99_s_all_admissions",
        "completion_qps",
        "receipt_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=headers)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                SYSTEM_ORDER.index(item["system_label"]),
                item["workload"],
                float(item["qps"]),
            ),
        ):
            summary = row["summary"]
            service_p99 = nested(summary, "service_latency_ms", "p99")
            writer.writerow(
                {
                    "system": row["system_label"],
                    "workload": row["workload"],
                    "target_qps": row["qps"],
                    "model_cohort": row["model"]["cohort"],
                    "status": row["status"],
                    "formal_total": summary.get("total"),
                    "successful": summary.get("successful"),
                    "failed": summary.get("failed"),
                    "timeouts": timeout_count(summary),
                    "success_rate": summary.get("success_rate"),
                    "service_mean_s_success_only": metric(
                        row, "service_mean_s"
                    ),
                    "service_p99_s_success_only": (
                        None
                        if service_p99 is None
                        else float(service_p99) / 1000.0
                    ),
                    "user_visible_p99_s_all_admissions": metric(
                        row, "user_visible_p99_s"
                    ),
                    "completion_qps": metric(row, "completion_qps"),
                    "receipt_path": row["path"],
                }
            )
    return path


def plot_dashboard(
    rows: list[dict[str, Any]], output: Path, stem: str, title: str
) -> None:
    metrics = [
        ("service_mean_s", "Service mean (s)\nsuccessful requests only"),
        ("user_visible_p99_s", "User-visible P99 (s)\nall admissions"),
        ("success_rate_pct", "Success rate (%)"),
        ("completion_qps", "Completion throughput (QPS)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.6), sharex="col")
    handles: dict[str, Any] = {}
    for row_index, workload in enumerate(("Search", "Native Add")):
        workload_rows = [row for row in rows if row["workload"] == workload]
        for column_index, (metric_name, label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for system in SYSTEM_ORDER:
                points = sorted(
                    [
                        row
                        for row in workload_rows
                        if row["system_label"] == system
                        and metric(row, metric_name) is not None
                    ],
                    key=lambda row: float(row["qps"]),
                )
                if not points:
                    continue
                exploratory = any(
                    row["model"]["cohort"] != CONTROLLED for row in points
                )
                (line,) = axis.plot(
                    [float(row["qps"]) for row in points],
                    [metric(row, metric_name) for row in points],
                    marker="o",
                    linewidth=2.1,
                    markersize=5.5,
                    linestyle="--" if exploratory else "-",
                    color=COLORS[system],
                    label=SHORT_LABELS.get(system, system),
                )
                handles.setdefault(system, line)
            axis.set_xticks([1, 5, 10])
            axis.grid(True, alpha=0.25)
            axis.set_xlabel("Target QPS")
            axis.set_ylabel(label)
            if metric_name in {"service_mean_s", "user_visible_p99_s"}:
                positive = [
                    metric(row, metric_name)
                    for row in workload_rows
                    if metric(row, metric_name) not in (None, 0)
                ]
                if positive and max(positive) / min(positive) >= 50:
                    axis.set_yscale("log")
                    axis.set_ylabel(f"{label}\n(log scale)")
            if metric_name == "success_rate_pct":
                axis.set_ylim(-3, 105)
            if metric_name == "completion_qps":
                axis.plot([1, 5, 10], [1, 5, 10], ":", color="#444444")
            axis.set_title(f"{workload}: {label.splitlines()[0]}")
    ordered_handles = [handles[system] for system in SYSTEM_ORDER if system in handles]
    ordered_labels = [
        SHORT_LABELS.get(system, system)
        for system in SYSTEM_ORDER
        if system in handles
    ]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.085,
        "Solid: controlled Qwen3-32B-FP8 cohort; dashed: Qwen3.8 exploratory. "
        "Missing points are not interpolated.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.95))
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_coverage(rows: list[dict[str, Any]], output: Path) -> None:
    columns = [
        (workload, qps)
        for workload in ("Search", "Native Add")
        for qps in (1, 5, 10)
    ]
    selected = {
        (row["system_label"], row["workload"], int(row["qps"])): row
        for row in rows
    }
    values: list[list[int]] = []
    annotations: list[list[str]] = []
    for system in SYSTEM_ORDER:
        value_row: list[int] = []
        text_row: list[str] = []
        for workload, qps in columns:
            row = selected.get((system, workload, qps))
            if row is None:
                value_row.append(0)
                text_row.append("missing")
                continue
            summary = row["summary"]
            successful = int(summary.get("successful") or 0)
            total = int(summary.get("total") or 0)
            if successful == 0:
                value_row.append(1)
            elif successful < total:
                value_row.append(2)
            else:
                value_row.append(3)
            cohort = "32B" if row["model"]["cohort"] == CONTROLLED else "3.8"
            text_row.append(f"{successful}/{total}\n{cohort}")
        values.append(value_row)
        annotations.append(text_row)

    fig, axis = plt.subplots(figsize=(13.5, 6.2))
    cmap = ListedColormap(["#D9D9D9", "#E57373", "#FFD166", "#76C893"])
    axis.imshow(values, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(
        [f"{workload}\nQPS {qps}" for workload, qps in columns]
    )
    axis.set_yticks(range(len(SYSTEM_ORDER)))
    axis.set_yticklabels([SHORT_LABELS.get(system, system) for system in SYSTEM_ORDER])
    for row_index, text_row in enumerate(annotations):
        for column_index, text in enumerate(text_row):
            axis.text(column_index, row_index, text, ha="center", va="center", fontsize=8)
    axis.set_title(
        "Collected Track C coverage (successful/formal admissions)",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Green = all successful; yellow = partial success; red = zero success; gray = missing. "
        "32B and 3.8 identify model cohorts.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output / "coverage_matrix.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "coverage_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    controlled = [row for row in rows if row["model"]["cohort"] == CONTROLLED]
    exploratory = [row for row in rows if row not in controlled]
    lines = [
        "# Track C QPS latency visualization inventory",
        "",
        "- Dataset: LoCoMo, 10 conversations, 5,882 events.",
        "- Search: 1,787 formal queries; Native Add: 2,000 formal admissions.",
        f"- Selected points: {len(rows)} ({len(controlled)} controlled Qwen3-32B, "
        f"{len(exploratory)} Qwen3.8 exploratory).",
        "- Service latency is computed over successful requests only.",
        "- User-visible latency and success rate retain all admissions/timeouts.",
        "- Graphiti is an OSS proxy for Zep, not production Zep.",
        "- Sparse exploratory points must not be mixed into controlled rankings.",
        "",
        "## Files",
        "",
        "- `qwen32_controlled_dashboard.png/.pdf`: controlled comparison.",
        "- `all_selected_dashboard.png/.pdf`: every selected collected point.",
        "- `coverage_matrix.png/.pdf`: missing/partial/zero-success coverage.",
        "- `selected_qps_metrics.csv`: exact plotted values and receipt paths.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(source)
    controlled = [row for row in rows if row["model"]["cohort"] == CONTROLLED]
    write_csv(output, rows)
    plot_dashboard(
        controlled,
        output,
        "qwen32_controlled_dashboard",
        "Track C controlled Qwen3-32B-FP8: latency, reliability, and throughput",
    )
    plot_dashboard(
        rows,
        output,
        "all_selected_dashboard",
        "All selected collected Track C points (model cohorts separated by line style)",
    )
    plot_coverage(rows, output)
    write_summary(output, rows)
    print(
        f"wrote {len(rows)} selected points ({len(controlled)} controlled) to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
