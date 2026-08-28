"""Render per-task quality and latency figures for the Math plan replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLANS = ("vfwd", "rrev", "aivg")
PLAN_LABELS = ("VFWD", "RREV", "AIVG")
PLAN_COLORS = ("#4C78A8", "#E45756", "#59A14F")
STAGES = (
    ("ann_ms_p50", "ANN", "#4C78A8"),
    ("graph_ms_p50", "Graph", "#F28E2B"),
    ("predicate_ms_p50", "Predicate", "#E15759"),
    ("dedup_rank_ms_p50", "Dedup/top-k", "#76B7B2"),
    ("hydrate_ms_p50", "Hydrate", "#59A14F"),
    ("executor_overhead_ms_p50", "Overhead", "#B07AA1"),
)


def _task_label(task_uid: str) -> str:
    return task_uid.removeprefix("math:").replace("_", " ")


def _load(path: Path) -> tuple[list[str], dict[tuple[str, str], dict]]:
    rows = json.loads(path.read_text())
    rows = [row for row in rows if row["task_uid"] != "__all_math__"]
    tasks = list(dict.fromkeys(row["task_uid"] for row in rows))
    return tasks, {(row["task_uid"], row["physical_plan"]): row for row in rows}


def quality_figure(tasks: list[str], lookup: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2), constrained_layout=True)
    metrics = (
        ("exact_order_rate", "Exact-order accuracy"),
        ("recall_at_10", "Recall@10"),
    )
    image = None
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        values = np.array(
            [[lookup[(task, plan)][metric] for plan in PLANS] for task in tasks]
        )
        image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(PLANS)), PLAN_LABELS)
        ax.set_yticks(range(len(tasks)), [_task_label(task) for task in tasks])
        for row in range(len(tasks)):
            for col in range(len(PLANS)):
                ax.text(
                    col,
                    row,
                    f"{values[row, col]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if values[row, col] >= 0.5 else "black",
                )
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.75, label="Quality (higher is better)")
    fig.suptitle("EvoTrace Math retrieval quality by physical plan", fontsize=14)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def latency_figure(tasks: list[str], lookup: dict, output: Path) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7.4), sharey=True, constrained_layout=True
    )
    y = np.arange(len(tasks))
    bar_height = 0.23
    for ax, (metric, title) in zip(
        axes,
        (
            ("retriever_total_ms_p50", "Median (p50)"),
            ("retriever_total_ms_p95", "Tail (p95)"),
        ),
        strict=True,
    ):
        for offset, (plan, label, color) in enumerate(
            zip(PLANS, PLAN_LABELS, PLAN_COLORS, strict=True)
        ):
            values = np.array([lookup[(task, plan)][metric] for task in tasks])
            positions = y + (offset - 1) * bar_height
            bars = ax.barh(
                positions, values, height=bar_height, label=label, color=color
            )
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Retriever latency (ms, log scale)")
        ax.set_title(title)
        ax.grid(axis="x", which="both", alpha=0.25)
    axes[0].set_yticks(y, [_task_label(task) for task in tasks])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("EvoTrace Math end-to-end retrieval latency by task", fontsize=14)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def breakdown_figure(tasks: list[str], lookup: dict, output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    axes = axes.flatten()
    for ax, task in zip(axes, tasks, strict=True):
        left = np.zeros(len(PLANS))
        totals = np.array(
            [lookup[(task, plan)]["retriever_total_ms_p50"] for plan in PLANS]
        )
        stage_sum = np.array(
            [
                sum(lookup[(task, plan)][metric] for metric, _, _ in STAGES)
                for plan in PLANS
            ]
        )
        for metric, label, color in STAGES:
            values = np.array([lookup[(task, plan)][metric] for plan in PLANS])
            shares = (
                np.divide(
                    values, stage_sum, out=np.zeros_like(values), where=stage_sum > 0
                )
                * 100
            )
            ax.barh(PLAN_LABELS, shares, left=left, label=label, color=color)
            left += shares
        for row, total in enumerate(totals):
            ax.text(101, row, f"{total:.1f} ms", va="center", fontsize=8)
        ax.set_xlim(0, 118)
        ax.set_title(_task_label(task))
        ax.set_xlabel("Share of measured p50 latency (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6)
    fig.suptitle(
        "Median retrieval latency breakdown (bar share; label = absolute total)",
        fontsize=14,
    )
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks, lookup = _load(args.summary)
    quality_figure(tasks, lookup, args.out / "retrieval_quality_by_task")
    latency_figure(tasks, lookup, args.out / "retrieval_latency_by_task")
    breakdown_figure(tasks, lookup, args.out / "retrieval_latency_breakdown_by_task")


if __name__ == "__main__":
    main()
