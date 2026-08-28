"""Render paper-aligned quality and latency views for the three-arm Know run.

The EvoMemBench paper reports CrossEp-Know answer accuracy by knowledge
category and Easy/Medium/Hard tier.  It uses token usage as its efficiency
metric and uses an accuracy-versus-cost scatter for one of its overview
figures.  This script preserves the paper's quality aggregation and adapts the
cost axis to the latency measurements collected by the TriDB experiment.

Polyglot answer quality and end-to-end latency are derived artifacts in the
current run.  The figures therefore use hollow/hatched marks for Polyglot and
never present those fields as independent answer-generation measurements.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable


ARMS = ("memory_off", "full_gem", "polyglot")
ARM_LABELS = {
    "memory_off": "No Memory",
    "full_gem": "GEM",
    "polyglot": "Polyglot",
}
COLORS = {
    "memory_off": "#4C78A8",
    "full_gem": "#F58518",
    "polyglot": "#54A24B",
}
CATEGORIES = (
    "Domain Knowledge Reasoning",
    "Empirical Discovery & Simulation",
    "Procedural Task Execution",
    "Rule System Application",
)
CATEGORY_SHORT = {
    "Domain Knowledge Reasoning": "Domain\nknowledge",
    "Empirical Discovery & Simulation": "Empirical\ndiscovery",
    "Procedural Task Execution": "Procedural\ntasks",
    "Rule System Application": "Rule\nsystems",
}
TIERS = ("Easy", "Medium", "Hard")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_graded(run_root: Path, arm: str) -> list[dict[str, Any]]:
    source_arm = arm if arm != "polyglot" else "full_gem"
    rows: list[dict[str, Any]] = []
    for shard in (0, 1):
        path = run_root / f"shard_{shard}" / "graded" / f"{source_arm}.jsonl"
        rows.extend(_jsonl(path))
    return rows


def _load_tiers(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["context_id"]: row for row in csv.DictReader(handle)}


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return centre - half, centre + half


def _quality(rows: list[dict[str, Any]], *, reuse_only: bool) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if not reuse_only or int(row["metadata"].get("ordinal", 0)) > 0
    ]
    successes = sum(int(row.get("score", 0)) for row in selected)
    yes = sum(
        value == "yes"
        for row in selected
        for value in row.get("requirement_status", [])
    )
    no = sum(
        value == "no" for row in selected for value in row.get("requirement_status", [])
    )
    low, high = _wilson(successes, len(selected))
    return {
        "n": len(selected),
        "successes": successes,
        "strict_answer_accuracy": successes / len(selected),
        "wilson_95_ci": [low, high],
        "rubric_yes": yes,
        "rubric_no": no,
        "rubric_satisfaction_rate": yes / (yes + no),
        "rows_without_requirement_status": sum(
            not row.get("requirement_status") for row in selected
        ),
    }


def _paper_tier_metrics(
    rows: list[dict[str, Any]], tiers: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Match upstream tier_table_kit aggregation exactly.

    First average binary score within context, then average context means in
    each category/tier cell.  The Overall tier is the unweighted mean of the
    four category cells, matching generate_difficulty_table.py.
    """
    by_context: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        context_id = str(row["metadata"]["context_id"])
        by_context[context_id].append(float(row.get("score", 0)))

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for context_id, scores in by_context.items():
        tier_row = tiers[context_id]
        category = tier_row["context_category"]
        tier = tier_row["difficulty"]
        buckets[(category, tier)].append(sum(scores) / len(scores))

    cells: dict[str, dict[str, dict[str, float | int]]] = {}
    for category in CATEGORIES:
        cells[category] = {}
        for tier in TIERS:
            values = buckets[(category, tier)]
            cells[category][tier] = {
                "contexts": len(values),
                "accuracy": sum(values) / len(values),
            }

    overall: dict[str, dict[str, float | int]] = {}
    for tier in TIERS:
        values = [float(cells[category][tier]["accuracy"]) for category in CATEGORIES]
        overall[tier] = {
            "category_cells": len(values),
            "accuracy": sum(values) / len(values),
        }
    cells["Overall"] = overall
    return cells


def _write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(
    run_root: Path, summary_path: Path, tiers_path: Path
) -> dict[str, Any]:
    summary = _json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("three-arm summary is not complete")
    if summary.get("parity", {}).get("fraction") != 1.0:
        raise ValueError(
            "Polyglot/GEM parity is not 100%; quality inheritance is invalid"
        )

    tiers = _load_tiers(tiers_path)
    graded = {arm: _load_graded(run_root, arm) for arm in ARMS}
    for arm, rows in graded.items():
        if len(rows) != 884:
            raise ValueError(f"{arm}: expected 884 graded rows, observed {len(rows)}")

    quality = {
        arm: {
            "all_episodes": _quality(graded[arm], reuse_only=False),
            "reuse_decisions": _quality(graded[arm], reuse_only=True),
            "paper_difficulty_table": _paper_tier_metrics(graded[arm], tiers),
            "provenance": (
                "inherited from GEM after 100% prompt parity"
                if arm == "polyglot"
                else "measured by the local EvoMemBench evaluator adapter"
            ),
        }
        for arm in ARMS
    }

    return {
        "schema_version": "evomembench_three_system_paper_aligned_visualization_v0.1.0",
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_root": str(run_root.resolve()),
        "summary_path": str(summary_path.resolve()),
        "tiers_path": str(tiers_path.resolve()),
        "dataset": summary["dataset"],
        "quality": quality,
        "latency_ms": summary["latency_ms"],
        "paired_effect": summary["quality"]["gem_vs_no_memory"],
        "parity": summary["parity"],
        "judge": summary["judge"],
        "hardware_claim": summary["hardware_claim"],
        "paper_alignment": {
            "primary_quality": (
                "strict binary answer accuracy, reported by category and "
                "Easy/Medium/Hard tier"
            ),
            "paper_efficiency": "total LLM input and output token usage",
            "this_visualization_efficiency": (
                "measured or explicitly reconstructed latency from the TriDB run"
            ),
            "supplemental_quality": (
                "rubric satisfaction rate from the upstream repository stats utility; "
                "not the paper's primary Table 5 metric"
            ),
        },
        "limitations": [
            "Polyglot quality is inherited from GEM after exact prompt parity, not independently generated.",
            "Polyglot end-to-end latency is reconstructed from measured components, not directly timed.",
            "The local Qwen3.8-27B-FP8 judge is not the paper-equivalent closed judge.",
            "Judge/adapter failures are retained as score 0 under the frozen lossless protocol.",
            "This is an x86_64 dual-GPU off-target run, not a GX10/ARM64 sign-off.",
        ],
    }


def _plot_setup() -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK SC"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    return plt


def _bar_style(arm: str) -> dict[str, Any]:
    style: dict[str, Any] = {
        "color": COLORS[arm],
        "edgecolor": "#263238",
        "linewidth": 0.7,
    }
    if arm == "polyglot":
        style.update({"facecolor": "white", "hatch": "///", "linewidth": 1.2})
    return style


def render_overview(metrics: dict[str, Any], output_dir: Path) -> None:
    plt = _plot_setup()
    from matplotlib.ticker import PercentFormatter

    quality = metrics["quality"]
    latency = metrics["latency_ms"]
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)
    figure.suptitle(
        "EvoMemBench CrossEp-Know: three-system quality and latency",
        fontsize=16,
        fontweight="bold",
    )

    x = list(range(len(ARMS)))
    width = 0.34
    all_rates = [quality[arm]["all_episodes"]["strict_answer_accuracy"] for arm in ARMS]
    reuse_rates = [
        quality[arm]["reuse_decisions"]["strict_answer_accuracy"] for arm in ARMS
    ]
    axes[0, 0].bar(
        [value - width / 2 for value in x],
        all_rates,
        width,
        label="All episodes (n=884)",
        color="#9ecae1",
        edgecolor="#263238",
        linewidth=0.7,
    )
    reuse_bars = axes[0, 0].bar(
        [value + width / 2 for value in x],
        reuse_rates,
        width,
        label="Reuse only (n=764)",
        color=[COLORS[arm] for arm in ARMS],
        edgecolor="#263238",
        linewidth=0.7,
    )
    reuse_bars[2].set_facecolor("white")
    reuse_bars[2].set_hatch("///")
    reuse_bars[2].set_linewidth(1.2)
    reuse_cis = [quality[arm]["reuse_decisions"]["wilson_95_ci"] for arm in ARMS]
    axes[0, 0].errorbar(
        [value + width / 2 for value in x],
        reuse_rates,
        yerr=[
            [reuse_rates[index] - reuse_cis[index][0] for index in x],
            [reuse_cis[index][1] - reuse_rates[index] for index in x],
        ],
        fmt="none",
        ecolor="#263238",
        capsize=4,
        linewidth=1.0,
    )
    axes[0, 0].set_xticks(x, [ARM_LABELS[arm] for arm in ARMS])
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_ylim(0, 0.15)
    axes[0, 0].set_title("(a) Strict answer accuracy")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for index, value in enumerate(reuse_rates):
        axes[0, 0].text(
            index + width / 2,
            value + 0.003,
            f"{100 * value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    tier_x = list(range(len(TIERS)))
    tier_width = 0.23
    for arm_index, arm in enumerate(ARMS):
        values = [
            quality[arm]["paper_difficulty_table"]["Overall"][tier]["accuracy"]
            for tier in TIERS
        ]
        bars = axes[0, 1].bar(
            [value + (arm_index - 1) * tier_width for value in tier_x],
            values,
            tier_width,
            label=ARM_LABELS[arm],
            **_bar_style(arm),
        )
        for bar, value in zip(bars, values, strict=True):
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.004,
                f"{100 * value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axes[0, 1].set_xticks(tier_x, TIERS)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 1].set_ylim(0, 0.23)
    axes[0, 1].set_title("(b) Paper-style difficulty tiers (all 884)")
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=3)

    e2e = {
        "memory_off": latency["no_memory_end_to_end_measured"],
        "full_gem": latency["gem_end_to_end_measured"],
        "polyglot": latency["polyglot_end_to_end_parity_reconstructed"],
    }
    for quantile_index, quantile in enumerate(("p50", "p95")):
        positions = [value + (quantile_index - 0.5) * width for value in x]
        values = [e2e[arm][quantile] / 1000.0 for arm in ARMS]
        bars = axes[1, 0].bar(
            positions,
            values,
            width,
            label=quantile.upper(),
            color=[COLORS[arm] for arm in ARMS],
            alpha=1.0 if quantile == "p50" else 0.45,
            edgecolor="#263238",
            linewidth=0.7,
            hatch=None if quantile == "p50" else "..",
        )
        bars[2].set_facecolor("white")
        bars[2].set_hatch("///" if quantile == "p50" else "///..")
        bars[2].set_linewidth(1.2)
    axes[1, 0].set_xticks(x, [ARM_LABELS[arm] for arm in ARMS])
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].set_ylim(0, 47)
    axes[1, 0].set_title("(c) End-to-end latency on reuse decisions")
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].text(
        0.98,
        0.96,
        "Polyglot = reconstructed",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    retrieval = {
        "full_gem": latency["gem_retrieval_measured"],
        "polyglot": latency["polyglot_retrieval_measured"],
    }
    retrieval_arms = ("full_gem", "polyglot")
    retrieval_x = list(range(len(retrieval_arms)))
    for quantile_index, quantile in enumerate(("p50", "p95")):
        positions = [value + (quantile_index - 0.5) * width for value in retrieval_x]
        values = [retrieval[arm][quantile] for arm in retrieval_arms]
        bars = axes[1, 1].bar(
            positions,
            values,
            width,
            label=quantile.upper(),
            color=[COLORS[arm] for arm in retrieval_arms],
            alpha=1.0 if quantile == "p50" else 0.45,
            edgecolor="#263238",
            linewidth=0.7,
            hatch=None if quantile == "p50" else "..",
        )
        for bar, value in zip(bars, values, strict=True):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                value + 6,
                f"{value:.1f}",
                ha="center",
                fontsize=8,
            )
    axes[1, 1].set_xticks(retrieval_x, [ARM_LABELS[arm] for arm in retrieval_arms])
    axes[1, 1].set_ylabel("milliseconds")
    axes[1, 1].set_ylim(0, 290)
    axes[1, 1].set_title("(d) Database retrieval latency (No Memory = N/A)")
    axes[1, 1].legend(frameon=False, fontsize=9)

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"quality_latency_overview.{suffix}", bbox_inches="tight"
        )
    plt.close(figure)


def render_quality_cost_scatter(metrics: dict[str, Any], output_dir: Path) -> None:
    plt = _plot_setup()
    from matplotlib.ticker import PercentFormatter

    quality = metrics["quality"]
    latency = metrics["latency_ms"]
    e2e = {
        "memory_off": latency["no_memory_end_to_end_measured"],
        "full_gem": latency["gem_end_to_end_measured"],
        "polyglot": latency["polyglot_end_to_end_parity_reconstructed"],
    }
    markers = {"memory_off": "o", "full_gem": "s", "polyglot": "D"}
    figure, axis = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    for arm in ARMS:
        x = e2e[arm]["p50"] / 1000.0
        y = quality[arm]["reuse_decisions"]["strict_answer_accuracy"]
        axis.scatter(
            [x],
            [y],
            marker=markers[arm],
            s=130,
            color=COLORS[arm] if arm != "polyglot" else "white",
            edgecolor=COLORS[arm] if arm == "polyglot" else "#263238",
            linewidth=1.8 if arm == "polyglot" else 0.9,
            zorder=3,
            label=(
                ARM_LABELS[arm]
                if arm != "polyglot"
                else "Polyglot (quality inherited; E2E reconstructed)"
            ),
        )
        axis.annotate(
            f"{ARM_LABELS[arm]}\n{x:.2f}s, {100 * y:.2f}%",
            (x, y),
            xytext=(8, 10 if arm != "polyglot" else -31),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("P50 end-to-end latency (seconds; lower is better)")
    axis.set_ylabel("Strict answer accuracy on reuse decisions (higher is better)")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(20.4, 23.7)
    axis.set_ylim(0.078, 0.105)
    axis.set_title("Paper-inspired quality–cost view (latency replaces token cost)")
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"quality_latency_scatter.{suffix}", bbox_inches="tight"
        )
    plt.close(figure)


def render_difficulty_table(metrics: dict[str, Any], output_dir: Path) -> None:
    plt = _plot_setup()
    import numpy as np

    columns = [
        (category, tier) for category in (*CATEGORIES, "Overall") for tier in TIERS
    ]
    matrix = np.array(
        [
            [
                100
                * float(
                    metrics["quality"][arm]["paper_difficulty_table"][category][tier][
                        "accuracy"
                    ]
                )
                for category, tier in columns
            ]
            for arm in ARMS
        ]
    )
    figure, axis = plt.subplots(figsize=(15.5, 3.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=32)
    labels = [
        f"{CATEGORY_SHORT.get(category, category)}\n{tier}"
        for category, tier in columns
    ]
    axis.set_xticks(range(len(columns)), labels, fontsize=8)
    axis.set_yticks(range(len(ARMS)), [ARM_LABELS[arm] for arm in ARMS])
    axis.tick_params(axis="x", rotation=0)
    axis.grid(False)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value > 17 else "#263238",
                fontsize=8,
                fontweight="bold" if row_index == 0 else "normal",
            )
    axis.set_title(
        "CROSSEP-KNOW strict answer accuracy (%) — Table 5 aggregation",
        pad=12,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.015)
    colorbar.set_label("accuracy (%)")
    axis.text(
        1.0,
        -0.34,
        "Polyglot row inherits GEM quality after 100% prompt parity.",
        transform=axis.transAxes,
        ha="right",
        fontsize=8,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"quality_by_category_and_difficulty.{suffix}",
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_tables(metrics: dict[str, Any], output_dir: Path) -> None:
    overall_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        all_result = metrics["quality"][arm]["all_episodes"]
        reuse = metrics["quality"][arm]["reuse_decisions"]
        latency = metrics["latency_ms"]
        if arm == "memory_off":
            e2e = latency["no_memory_end_to_end_measured"]
            retrieval = None
            latency_provenance = "measured"
        elif arm == "full_gem":
            e2e = latency["gem_end_to_end_measured"]
            retrieval = latency["gem_retrieval_measured"]
            latency_provenance = "measured"
        else:
            e2e = latency["polyglot_end_to_end_parity_reconstructed"]
            retrieval = latency["polyglot_retrieval_measured"]
            latency_provenance = "E2E reconstructed; retrieval measured"
        overall_rows.append(
            {
                "system": ARM_LABELS[arm],
                "all_strict_accuracy_pct": 100 * all_result["strict_answer_accuracy"],
                "reuse_strict_accuracy_pct": 100 * reuse["strict_answer_accuracy"],
                "reuse_passes": reuse["successes"],
                "reuse_n": reuse["n"],
                "reuse_rubric_satisfaction_pct": 100
                * reuse["rubric_satisfaction_rate"],
                "e2e_p50_ms": e2e["p50"],
                "e2e_p95_ms": e2e["p95"],
                "retrieval_p50_ms": "" if retrieval is None else retrieval["p50"],
                "retrieval_p95_ms": "" if retrieval is None else retrieval["p95"],
                "quality_provenance": metrics["quality"][arm]["provenance"],
                "latency_provenance": latency_provenance,
            }
        )
    _write_csv(
        output_dir / "three_system_summary.csv",
        list(overall_rows[0]),
        overall_rows,
    )

    difficulty_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for category in (*CATEGORIES, "Overall"):
            for tier in TIERS:
                result = metrics["quality"][arm]["paper_difficulty_table"][category][
                    tier
                ]
                difficulty_rows.append(
                    {
                        "system": ARM_LABELS[arm],
                        "category": category,
                        "difficulty": tier,
                        "accuracy_pct": 100 * float(result["accuracy"]),
                        "aggregation": "mean of context means",
                        "quality_provenance": metrics["quality"][arm]["provenance"],
                    }
                )
    _write_csv(
        output_dir / "paper_style_difficulty_quality.csv",
        list(difficulty_rows[0]),
        difficulty_rows,
    )


def _write_readme(metrics: dict[str, Any], output_dir: Path) -> None:
    quality = metrics["quality"]
    latency = metrics["latency_ms"]
    delta = metrics["paired_effect"]["paired_mean_delta"] * 100
    ci = [
        value * 100 for value in metrics["paired_effect"]["clustered_bootstrap_95_ci"]
    ]
    no_memory = quality["memory_off"]["reuse_decisions"]
    gem = quality["full_gem"]["reuse_decisions"]
    polyglot = quality["polyglot"]["reuse_decisions"]
    text = f"""# EvoMemBench 三系统质量与延迟图

## 最简单的结论

- 这次 764 个真正发生跨会话复用的任务中，No Memory 严格正确率是 **{100 * no_memory["strict_answer_accuracy"]:.2f}%**（{no_memory["successes"]}/{no_memory["n"]}），GEM 是 **{100 * gem["strict_answer_accuracy"]:.2f}%**（{gem["successes"]}/{gem["n"]}）。差值为 **{delta:+.2f} 个百分点**，按 context 聚类 bootstrap 的 95% CI 是 **[{ci[0]:+.2f}, {ci[1]:+.2f}]**，跨过 0，所以目前不能说 GEM 提升了质量。
- Polyglot 图中的严格正确率是 **{100 * polyglot["strict_answer_accuracy"]:.2f}%**，但它不是独立生成答案得到的分数；它在 764/764 条 prompt 完全一致后继承 GEM 的答案质量。
- No Memory 的 P50 E2E 是 **{latency["no_memory_end_to_end_measured"]["p50"] / 1000:.2f}s**，GEM 是 **{latency["gem_end_to_end_measured"]["p50"] / 1000:.2f}s**，Polyglot 是 **{latency["polyglot_end_to_end_parity_reconstructed"]["p50"] / 1000:.2f}s**。Polyglot 的 E2E 是重建值，不是独立端到端计时。
- 数据库检索本身：GEM P50 **{latency["gem_retrieval_measured"]["p50"]:.2f}ms**，Polyglot P50 **{latency["polyglot_retrieval_measured"]["p50"]:.2f}ms**；Polyglot/GEM 的中位倍率为 **{latency["polyglot_over_gem_retrieval"]["ratio_right_over_left"]["p50"]:.2f}×**。因此这轮最强的正面结论是 GEM 的检索链路明显更快，而不是答案质量更高。

## 论文怎样评价 quality

EvoMemBench 论文对 CROSSEP-KNOW 使用 **answer accuracy**。官方 evaluator 是严格二值打分：一题的所有 rubric 都满足才记 1，否则记 0。论文 Table 5 再按四类知识和 Easy/Medium/Hard 展开；difficulty 是用 DeepSeek-V3.2 no-memory baseline 的 context 分数，在每个类别内按三等分得到。官方仓库的统计脚本还给出 `Rubric%`（满足的 rubric 条目比例），但它是辅助诊断，不是论文 Table 5 的主质量指标。

本地 `Rubric%` 很高（No Memory **{100 * no_memory["rubric_satisfaction_rate"]:.2f}%**，GEM **{100 * gem["rubric_satisfaction_rate"]:.2f}%**），而严格正确率只有约 9%。意思是模型通常答对了大部分要求，但只要漏一个要求，整题仍然失败。这个差距适合定位问题，不能替代论文的严格 accuracy。

## 图怎么读

- `quality_latency_overview.*`：总览。上排是严格质量及论文 difficulty 分层，下排是 E2E 和数据库检索延迟。
- `quality_by_category_and_difficulty.*`：照论文 Table 5 的聚合方式，把四类知识 × 三档难度画成热力表。
- `quality_latency_scatter.*`：仿论文 Figure 2 的质量–成本散点；论文 x 轴是 token usage，这里为了系统性能比较换成 P50 E2E latency。
- 实心是直接测量；Polyglot 的空心/斜线表示继承质量或重建 E2E。No Memory 没有检索，所以 retrieval 不能写成 0ms，而应写 N/A。

## 为什么不能直接和论文数字比较

本次答案模型/裁判是本地 Qwen3.8-27B-FP8，论文主表的统一 memory backbone 是 DeepSeek-V3.2；当前 summary 也明确标记本地 judge 不是 paper-equivalent closed judge。再加上本次是 x86_64 双 GPU off-target run，因此只能比较本轮三个 arm，不能把绝对 accuracy 当成论文复现或 GX10 sign-off。

## 来源

- Paper: https://arxiv.org/abs/2605.18421
- Official repository: https://github.com/DSAIL-Memory/EvoMemBench
"""
    (output_dir / "README.md").write_text(text)


def render(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_tables(metrics, output_dir)
    render_overview(metrics, output_dir)
    render_quality_cost_scatter(metrics, output_dir)
    render_difficulty_table(metrics, output_dir)
    _write_readme(metrics, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--tiers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(build_metrics(args.run_root, args.summary, args.tiers), args.output)


if __name__ == "__main__":
    main()
