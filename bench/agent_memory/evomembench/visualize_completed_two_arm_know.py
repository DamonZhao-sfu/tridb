"""Visualize the completed No Memory and GEM slice of an EvoMemBench Know run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any, Iterable


ARMS = ("memory_off", "full_gem")
ARM_LABELS = {"memory_off": "No Memory", "full_gem": "GEM"}
COLORS = {"memory_off": "#457b9d", "full_gem": "#2a9d8f"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load(run_root: Path, kind: str, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard in (0, 1):
        rows.extend(_jsonl(run_root / f"shard_{shard}" / kind / f"{arm}.jsonl"))
    return rows


def _percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[
        max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    ]


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    observed = list(float(value) for value in values)
    return {
        "n": len(observed),
        "mean": sum(observed) / len(observed),
        "p50": median(observed),
        "p95": _percentile(observed, 0.95),
    }


def _wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
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


def _paired_binomial_p(off_only: int, gem_only: int) -> float:
    discordant = off_only + gem_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(off_only, gem_only) + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def _cluster_bootstrap_delta(
    ids: list[str], graded: dict[str, dict[str, dict[str, Any]]], draws: int = 20_000
) -> tuple[float, float]:
    by_context: dict[str, list[str]] = defaultdict(list)
    for task_id in ids:
        context_id = str(graded["memory_off"][task_id]["metadata"]["context_id"])
        by_context[context_id].append(task_id)
    contexts = sorted(by_context)
    rng = random.Random(20260828)
    deltas: list[float] = []
    for _ in range(draws):
        sampled = [rng.choice(contexts) for _ in contexts]
        numerator = 0
        denominator = 0
        for context_id in sampled:
            context_ids = by_context[context_id]
            numerator += sum(
                int(graded["full_gem"][task_id]["score"])
                - int(graded["memory_off"][task_id]["score"])
                for task_id in context_ids
            )
            denominator += len(context_ids)
        deltas.append(100.0 * numerator / denominator)
    deltas.sort()
    return deltas[int(0.025 * draws)], deltas[int(0.975 * draws) - 1]


def build_metrics(run_root: Path) -> dict[str, Any]:
    graded_rows = {arm: _load(run_root, "graded", arm) for arm in ARMS}
    trace_rows = {arm: _load(run_root, "traces", arm) for arm in ARMS}
    graded = {
        arm: {str(row["idx"]): row for row in rows} for arm, rows in graded_rows.items()
    }
    traces = {
        arm: {str(row["target_id"]).split(":")[-1]: row for row in rows}
        for arm, rows in trace_rows.items()
    }
    paired_ids = sorted(set(graded["memory_off"]) & set(graded["full_gem"]))
    if len(paired_ids) != 884:
        raise ValueError(f"expected 884 paired tasks, observed {len(paired_ids)}")
    reuse_ids = [
        task_id
        for task_id in paired_ids
        if int(graded["memory_off"][task_id]["metadata"]["ordinal"]) > 0
    ]
    transitions = Counter(
        (
            int(graded["memory_off"][task_id]["score"]),
            int(graded["full_gem"][task_id]["score"]),
        )
        for task_id in reuse_ids
    )
    off_only = transitions[(1, 0)]
    gem_only = transitions[(0, 1)]
    quality: dict[str, Any] = {
        "metric": "official strict binary score (judge parse failures retained as score 0)",
        "all_tasks": {"n": len(paired_ids)},
        "reuse_tasks": {"n": len(reuse_ids)},
        "paired_transitions_reuse": {
            "both_fail": transitions[(0, 0)],
            "no_memory_only_pass": off_only,
            "gem_only_pass": gem_only,
            "both_pass": transitions[(1, 1)],
        },
        "paired_exact_p_reuse": _paired_binomial_p(off_only, gem_only),
    }
    for scope, ids in (("all_tasks", paired_ids), ("reuse_tasks", reuse_ids)):
        for arm in ARMS:
            successes = sum(int(graded[arm][task_id]["score"]) for task_id in ids)
            low, high = _wilson(successes, len(ids))
            quality[scope][arm] = {
                "successes": successes,
                "rate": successes / len(ids),
                "wilson_95ci": [low, high],
            }
    delta = 100.0 * (
        quality["reuse_tasks"]["full_gem"]["rate"]
        - quality["reuse_tasks"]["memory_off"]["rate"]
    )
    quality["reuse_delta_percentage_points"] = delta
    quality["reuse_delta_cluster_bootstrap_95ci_pp"] = list(
        _cluster_bootstrap_delta(reuse_ids, graded)
    )
    quality["evaluator_failures"] = {
        arm: sum(bool(row.get("evaluator_error")) for row in graded_rows[arm])
        for arm in ARMS
    }

    categories: dict[str, Any] = {}
    for category in sorted(
        {
            graded["memory_off"][task_id]["metadata"]["context_category"]
            for task_id in reuse_ids
        }
    ):
        ids = [
            task_id
            for task_id in reuse_ids
            if graded["memory_off"][task_id]["metadata"]["context_category"] == category
        ]
        categories[str(category)] = {
            "n": len(ids),
            **{
                arm: sum(int(graded[arm][task_id]["score"]) for task_id in ids)
                / len(ids)
                for arm in ARMS
            },
        }
    quality["reuse_by_category"] = categories

    complete_reuse_ids = [
        task_id
        for task_id in reuse_ids
        if traces["memory_off"][task_id]["status"] == "complete"
        and traces["full_gem"][task_id]["status"] == "complete"
    ]
    latency: dict[str, Any] = {"paired_complete_reuse_n": len(complete_reuse_ids)}
    for arm in ARMS:
        latency[arm] = {}
        for metric in ("end_to_end", "model_ttft", "generation"):
            latency[arm][f"{metric}_ms"] = _distribution(
                traces[arm][task_id]["latency_ms"][metric]
                for task_id in complete_reuse_ids
            )
    gem_retrieval = {
        f"{metric}_ms": _distribution(
            traces["full_gem"][task_id]["latency_ms"][metric]
            for task_id in complete_reuse_ids
        )
        for metric in (
            "query_embedding",
            "database_retrieval",
            "memory_or_prompt_assembly",
        )
    }
    operator_work: dict[str, Any] = {}
    for metric in (
        "vector_candidates_examined",
        "graph_edges_examined",
        "relational_candidates_examined",
        "relational_candidates_survived",
        "final_results",
    ):
        values = [
            traces["full_gem"][task_id]["intermediate"].get(metric)
            for task_id in complete_reuse_ids
        ]
        numeric = [value for value in values if isinstance(value, (int, float))]
        operator_work[metric] = (
            _distribution(numeric)
            if numeric
            else {"n": 0, "status": "not instrumented"}
        )

    return {
        "schema_version": "evomembench_completed_two_arm_visualization_v0.1.0",
        "status": "completed_two_arm_slice; parent three-arm run failed in Polyglot replay",
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "run_root": str(run_root.resolve()),
        "quality": quality,
        "latency": latency,
        "gem_retrieval": gem_retrieval,
        "gem_operator_work": operator_work,
        "limitations": [
            "Off-target experiment: not a GX10 sign-off.",
            "The official judge produced 41 parse failures per arm; the frozen lossless protocol retains each as score 0.",
            "Latency uses 760 paired, complete reuse rows; four reuse rows with context_overflow are excluded.",
            "The dataset does not provide an easy/medium/hard label; category is shown instead.",
            "The parent three-arm run failed during Polyglot injection parity, so this report makes no Polyglot claim.",
        ],
    }


def render(metrics: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    quality = metrics["quality"]
    latency = metrics["latency"]
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "EvoMemBench CrossEp-Know: No Memory vs GEM", fontsize=16, fontweight="bold"
    )
    arm_x = [0, 1]
    rates = [quality["reuse_tasks"][arm]["rate"] for arm in ARMS]
    cis = [quality["reuse_tasks"][arm]["wilson_95ci"] for arm in ARMS]
    yerr = [
        [rates[index] - cis[index][0] for index in arm_x],
        [cis[index][1] - rates[index] for index in arm_x],
    ]
    axes[0, 0].bar(arm_x, rates, color=[COLORS[arm] for arm in ARMS], width=0.62)
    axes[0, 0].errorbar(arm_x, rates, yerr=yerr, fmt="none", color="#1d3557", capsize=5)
    axes[0, 0].set_xticks(arm_x, [ARM_LABELS[arm] for arm in ARMS])
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_ylim(0, max(c[1] for c in cis) * 1.32)
    axes[0, 0].set_title("Official quality on reuse decisions (n=764)")
    for index, arm in enumerate(ARMS):
        result = quality["reuse_tasks"][arm]
        axes[0, 0].text(
            index,
            rates[index] + 0.006,
            f"{100 * rates[index]:.2f}%\n({result['successes']}/764)",
            ha="center",
        )
    delta = quality["reuse_delta_percentage_points"]
    delta_ci = quality["reuse_delta_cluster_bootstrap_95ci_pp"]
    axes[0, 0].text(
        0.5,
        0.98,
        f"GEM − No Memory: {delta:+.2f} pp\ncluster-bootstrap 95% CI [{delta_ci[0]:+.2f}, {delta_ci[1]:+.2f}]",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    transition = quality["paired_transitions_reuse"]
    transition_labels = ["Both fail", "No Memory only", "GEM only", "Both pass"]
    transition_values = [
        transition["both_fail"],
        transition["no_memory_only_pass"],
        transition["gem_only_pass"],
        transition["both_pass"],
    ]
    axes[0, 1].barh(
        transition_labels,
        transition_values,
        color=["#adb5bd", COLORS["memory_off"], COLORS["full_gem"], "#6a4c93"],
    )
    axes[0, 1].set_title("Paired quality outcomes")
    axes[0, 1].set_xlabel("tasks")
    for index, value in enumerate(transition_values):
        axes[0, 1].text(value + 5, index, str(value), va="center")
    axes[0, 1].text(
        0.98,
        0.05,
        f"paired exact p={quality['paired_exact_p_reuse']:.3f}",
        transform=axes[0, 1].transAxes,
        ha="right",
        fontsize=9,
    )

    width = 0.34
    p50 = [latency[arm]["end_to_end_ms"]["p50"] / 1000.0 for arm in ARMS]
    p95 = [latency[arm]["end_to_end_ms"]["p95"] / 1000.0 for arm in ARMS]
    axes[1, 0].bar(
        [x - width / 2 for x in arm_x],
        p50,
        width,
        label="p50",
        color=[COLORS[arm] for arm in ARMS],
    )
    axes[1, 0].bar(
        [x + width / 2 for x in arm_x],
        p95,
        width,
        label="p95",
        color=[COLORS[arm] for arm in ARMS],
        alpha=0.48,
        hatch="//",
    )
    axes[1, 0].set_xticks(arm_x, [ARM_LABELS[arm] for arm in ARMS])
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].set_title(
        f"End-to-end latency (paired complete reuse n={latency['paired_complete_reuse_n']})"
    )
    axes[1, 0].legend()
    for index in arm_x:
        axes[1, 0].text(
            index - width / 2,
            p50[index] + 0.5,
            f"{p50[index]:.2f}",
            ha="center",
            fontsize=9,
        )
        axes[1, 0].text(
            index + width / 2,
            p95[index] + 0.5,
            f"{p95[index]:.2f}",
            ha="center",
            fontsize=9,
        )

    stages = (
        "query_embedding_ms",
        "database_retrieval_ms",
        "memory_or_prompt_assembly_ms",
    )
    stage_labels = ("Embedding", "DB retrieval", "Memory path total")
    stage_p50 = [metrics["gem_retrieval"][stage]["p50"] for stage in stages]
    stage_p95 = [metrics["gem_retrieval"][stage]["p95"] for stage in stages]
    x = list(range(len(stages)))
    axes[1, 1].bar(
        [value - width / 2 for value in x],
        stage_p50,
        width,
        label="p50",
        color="#2a9d8f",
    )
    axes[1, 1].bar(
        [value + width / 2 for value in x],
        stage_p95,
        width,
        label="p95",
        color="#e9c46a",
    )
    axes[1, 1].set_xticks(x, stage_labels)
    axes[1, 1].set_ylabel("milliseconds")
    axes[1, 1].set_title("GEM retrieval-path latency")
    axes[1, 1].legend()
    for index, value in enumerate(stage_p50):
        axes[1, 1].text(
            index - width / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=9
        )
    for index, value in enumerate(stage_p95):
        axes[1, 1].text(
            index + width / 2, value + 0.8, f"{value:.2f}", ha="center", fontsize=9
        )
    fig.savefig(output_dir / "no_memory_vs_gem_dashboard.png", dpi=200)
    fig.savefig(output_dir / "no_memory_vs_gem_dashboard.svg")
    plt.close(fig)

    categories = quality["reuse_by_category"]
    labels = list(categories)
    y = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    axes[0].barh(
        [value + 0.18 for value in y],
        [categories[label]["memory_off"] for label in labels],
        0.36,
        label="No Memory",
        color=COLORS["memory_off"],
    )
    axes[0].barh(
        [value - 0.18 for value in y],
        [categories[label]["full_gem"] for label in labels],
        0.36,
        label="GEM",
        color=COLORS["full_gem"],
    )
    axes[0].set_yticks(y, [f"{label} (n={categories[label]['n']})" for label in labels])
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_title("Reuse quality by benchmark category")
    axes[0].legend()

    work = metrics["gem_operator_work"]
    work_keys = ["vector_candidates_examined", "graph_edges_examined", "final_results"]
    work_labels = ["Vector candidates", "Graph edges", "Final results"]
    work_p50 = [work[key]["p50"] for key in work_keys]
    work_p95 = [work[key]["p95"] for key in work_keys]
    x = list(range(len(work_keys)))
    axes[1].bar(
        [value - width / 2 for value in x],
        work_p50,
        width,
        label="p50",
        color="#457b9d",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        work_p95,
        width,
        label="p95",
        color="#e76f51",
    )
    axes[1].set_xticks(x, work_labels)
    axes[1].set_ylabel("items examined / returned")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_title("GEM operator work (relation scan not instrumented)")
    axes[1].legend()
    fig.savefig(output_dir / "quality_categories_and_operator_work.png", dpi=200)
    fig.savefig(output_dir / "quality_categories_and_operator_work.svg")
    plt.close(fig)

    readme = f"""# EvoMemBench No Memory vs GEM visualization

This is the completed two-arm slice of `{Path(metrics["run_root"]).name}`. The parent
three-arm run failed later in Polyglot replay; that does not invalidate the already
completed No Memory and GEM artifacts, but this report makes no Polyglot claim.

## Main results

- Official quality on reuse decisions (`n=764`): No Memory
  `{100 * quality["reuse_tasks"]["memory_off"]["rate"]:.2f}%`
  (`{quality["reuse_tasks"]["memory_off"]["successes"]}/764`), GEM
  `{100 * quality["reuse_tasks"]["full_gem"]["rate"]:.2f}%`
  (`{quality["reuse_tasks"]["full_gem"]["successes"]}/764`).
- Paired difference (GEM - No Memory): `{delta:+.2f}` percentage points;
  context-cluster bootstrap 95% CI `[{delta_ci[0]:+.2f}, {delta_ci[1]:+.2f}]`;
  paired exact p=`{quality["paired_exact_p_reuse"]:.3f}`.
- End-to-end latency p50/p95 (`n={latency["paired_complete_reuse_n"]}` complete paired reuse rows):
  No Memory `{p50[0]:.2f}/{p95[0]:.2f}s`, GEM `{p50[1]:.2f}/{p95[1]:.2f}s`.
- GEM retrieval path p50: embedding `{stage_p50[0]:.2f}ms`, database retrieval
  `{stage_p50[1]:.2f}ms`, total `{stage_p50[2]:.2f}ms`.
- Judge parsing failures: 41 arm-results for No Memory and 41 for GEM. The frozen
  lossless evaluator retains them as score 0 and preserves the denominator.

## Interpretation boundary

The quality interval crosses zero, so this run does not establish a GEM quality benefit
or harm. It is off-target and is not a GX10 performance sign-off. EvoMemBench does not
provide easy/medium/hard labels; the second figure uses its native context categories.
Relational candidate counts are absent from the current trace instrumentation and must
not be interpreted as zero.

`metrics.json` is the machine-readable source for both figures.
"""
    (output_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(build_metrics(args.run_root), args.output_dir)


if __name__ == "__main__":
    main()
