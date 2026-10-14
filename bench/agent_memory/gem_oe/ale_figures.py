"""Render the ALE p-sweep as figures 1-3, in the house style of gem_bench.figures.

Reads a matrix output directory directly -- receipts, ``best_program_info.json`` and
``retrieval_telemetry.jsonl`` -- rather than a pre-aggregated summary, because the
p-sweep has no ``paper_sections.json`` equivalent.  Every plotted value is also
written to a long-form CSV, so the bitmap is never the only artifact.

Only cells whose receipt says ``complete`` are plotted.  An arm that is still
running contributes fewer tasks and a smaller n, and the manifest records both, so
a partial arm cannot be mistaken for a finished one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCHEMA_VERSION = "tridb_ale_psweep_figures_v0.1.0"
BLUE = "#2878B5"
ORANGE = "#E07A1F"
GREEN = "#3A923A"
PURPLE = "#7B5AA6"
GREY = "#7A7A7A"
RED = "#C03A3A"
GRID = "#D9DEE7"

CONTROL_ARM = "nocontext"
ARM_COLOR: dict[str, str] = {
    "gem": BLUE,
    "polyglot": ORANGE,
    "cognee": GREEN,
    "nocontext": GREY,
}
ARM_LABEL: dict[str, str] = {
    "gem": "GEM",
    "polyglot": "Polyglot",
    "cognee": "Cognee",
    "nocontext": "no memory",
}
# Stage keys a retriever may report.  A retriever that reports none is drawn as one
# fused pass -- an absent breakdown is not the same as a zero-cost stage.
STAGE_KEYS: tuple[tuple[str, str], ...] = (
    ("ann_ms", "ANN"),
    ("traverse_ms", "graph traverse"),
    ("filter_rank_ms", "filter + rank"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _style_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def _append(
    records: list[dict[str, Any]],
    *,
    figure: str,
    task: str,
    arm: str,
    frequency: Any,
    metric: str,
    value: Any,
    unit: str,
) -> None:
    records.append(
        {
            "figure": figure,
            "task": task,
            "arm": arm,
            "injection_frequency": frequency,
            "metric": metric,
            "value": value,
            "unit": unit,
        }
    )


# --------------------------------------------------------------------------- load


def load_cells(run_dir: Path) -> list[dict[str, Any]]:
    """Every ``complete`` cell in ``run_dir``, with its score and its telemetry."""
    cells: list[dict[str, Any]] = []
    for cell_dir in sorted(run_dir.glob("ale_*")):
        if not cell_dir.is_dir() or cell_dir.name.endswith(".outage_0827"):
            continue
        receipt_path = cell_dir / "run_receipt.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            continue
        best_path = cell_dir / "openevolve/best/best_program_info.json"
        best = None
        if best_path.exists():
            metrics = json.loads(best_path.read_text(encoding="utf-8")).get("metrics")
            if isinstance(metrics, Mapping):
                best = metrics.get("combined_score")
        telemetry: list[dict[str, Any]] = []
        telemetry_path = cell_dir / "retrieval_telemetry.jsonl"
        if telemetry_path.exists():
            telemetry = [
                json.loads(line)
                for line in telemetry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        cells.append(
            {
                "dir": cell_dir,
                "task": receipt["task_uid"].split(":", 1)[1],
                "arm": receipt["arm"],
                "frequency": receipt.get("injection_frequency"),
                "seed_fitness": receipt.get("seed_fitness"),
                "best": None if best is None else float(best),
                "iterations": receipt.get("iterations_traced"),
                "telemetry": telemetry,
            }
        )
    if not cells:
        raise ValueError(f"no complete cells under {run_dir}")
    return cells


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


# ------------------------------------------------------------------------ fig 1


def _figure1(
    cells: Sequence[Mapping[str, Any]],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    """Best score per memory arm, relative to that task's own no-memory control."""
    control = {
        cell["task"]: cell["best"] for cell in cells if cell["arm"] == CONTROL_ARM
    }
    memory_arms = sorted(
        {str(cell["arm"]) for cell in cells if cell["arm"] != CONTROL_ARM}
    )
    frequencies = sorted(
        {
            float(cell["frequency"])
            for cell in cells
            if cell["arm"] != CONTROL_ARM and cell["frequency"] is not None
        }
    )
    relative: dict[tuple[str, str, float], float] = {}
    for cell in cells:
        base = control.get(cell["task"])
        if cell["arm"] == CONTROL_ARM or base in (None, 0) or cell["best"] is None:
            continue
        value = (cell["best"] - base) / abs(base) * 100.0
        relative[(cell["task"], str(cell["arm"]), float(cell["frequency"]))] = value
        _append(
            records,
            figure="1",
            task=cell["task"],
            arm=str(cell["arm"]),
            frequency=cell["frequency"],
            metric="best_combined_score",
            value=cell["best"],
            unit="score",
        )
        _append(
            records,
            figure="1",
            task=cell["task"],
            arm=str(cell["arm"]),
            frequency=cell["frequency"],
            metric="relative_to_no_memory",
            value=round(value, 4),
            unit="percent",
        )
    for task, base in control.items():
        _append(
            records,
            figure="1",
            task=task,
            arm=CONTROL_ARM,
            frequency="",
            metric="best_combined_score",
            value=base,
            unit="score",
        )

    def rank(task: str) -> float:
        values = [v for (t, _, _), v in relative.items() if t == task]
        return max(values) if values else float("-inf")

    tasks = sorted(control, key=rank, reverse=True)
    # Reversed so the topmost bar in each group is the first legend entry: the y
    # axis is inverted to read tasks top-down, which flips the slot order too.
    slots = [(arm, freq) for arm in memory_arms for freq in frequencies][::-1]
    height = 0.8 / max(len(slots), 1)
    fig, ax = plt.subplots(
        figsize=(7.4, 0.62 * len(tasks) + 2.0), constrained_layout=True
    )
    positions = list(range(len(tasks)))
    for slot_index, (arm, freq) in enumerate(slots):
        offset = (slot_index - (len(slots) - 1) / 2) * height
        values = [relative.get((task, arm, freq), 0.0) for task in tasks]
        present = [(task, arm, freq) in relative for task in tasks]
        ax.barh(
            [p - offset for p in positions],
            values,
            height * 0.9,
            color=ARM_COLOR.get(arm, PURPLE),
            alpha=0.45 + 0.275 * frequencies.index(freq),
            label=f"{ARM_LABEL.get(arm, arm)} p={freq:g}",
            edgecolor="white",
            linewidth=0.4,
        )
        for position, value, ok in zip(positions, values, present, strict=True):
            if not ok:
                ax.text(
                    0.0,
                    position - offset,
                    "  n/a",
                    va="center",
                    fontsize=6,
                    color=GREY,
                )
            elif abs(value) < 0.5:
                # A sub-half-percent bar is a few pixels wide on an axis that has to
                # reach +57%. Without the number these rows read as "no data", which
                # is the one thing they are not: ahc026 and ahc046 are ties.
                ax.text(
                    0.0,
                    position - offset,
                    f"  {value:+.2f}%",
                    va="center",
                    fontsize=6,
                    color=GREY,
                )
    ax.axvline(0.0, color=RED, linewidth=1.1, zorder=3)
    ax.set_yticks(positions, tasks)
    ax.invert_yaxis()
    ax.set_xlabel("Best score relative to the same task's no-memory control (%)")
    ax.set_title(
        "Figure 1: does memory make the optimizer better?\n"
        "positive = the memory arm beat its own control"
    )
    # Bars are plotted in reverse slot order (inverted y axis), so the legend
    # handles come back reversed too; flip them back to match what the eye reads.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        frameon=False,
        fontsize=7.5,
        ncol=len(memory_arms),
        loc="lower right",
    )
    _style_axis(ax, grid_axis="x")
    return _save(fig, output_dir, "figure1_score_vs_control")


# ------------------------------------------------------------------------ fig 2


def _latency_by_task(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        if cell["arm"] == CONTROL_ARM:
            continue
        for entry in cell["telemetry"]:
            grouped[str(cell["arm"])][cell["task"]].append(float(entry["total_ms"]))
    return grouped


def _figure2(
    cells: Sequence[Mapping[str, Any]],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    """Retrieval latency per call: median bar, p95 whisker, one group per task."""
    grouped = _latency_by_task(cells)
    arms = sorted(grouped)
    tasks = sorted({task for per_arm in grouped.values() for task in per_arm})
    width = 0.8 / max(len(arms), 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    positions = list(range(len(tasks)))
    for arm_index, arm in enumerate(arms):
        offset = (arm_index - (len(arms) - 1) / 2) * width
        medians, lows, highs = [], [], []
        for task in tasks:
            values = grouped[arm].get(task, [])
            if not values:
                medians.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
                continue
            p50 = _quantile(values, 0.50)
            p95 = _quantile(values, 0.95)
            medians.append(p50)
            lows.append(max(p50 - _quantile(values, 0.05), 0.0))
            highs.append(max(p95 - p50, 0.0))
            for metric, value in (
                ("retrieval_p05_ms", _quantile(values, 0.05)),
                ("retrieval_p50_ms", p50),
                ("retrieval_p95_ms", p95),
                ("retrieval_mean_ms", statistics.mean(values)),
                ("retrieval_calls", len(values)),
            ):
                _append(
                    records,
                    figure="2",
                    task=task,
                    arm=arm,
                    frequency="all",
                    metric=metric,
                    value=round(value, 3),
                    unit="milliseconds" if metric.endswith("_ms") else "calls",
                )
        ax.bar(
            [p + offset for p in positions],
            medians,
            width * 0.88,
            yerr=[lows, highs],
            capsize=2.5,
            error_kw={"linewidth": 0.9, "ecolor": "#4A4A4A"},
            color=ARM_COLOR.get(arm, PURPLE),
            label=f"{ARM_LABEL.get(arm, arm)} (n={sum(len(v) for v in grouped[arm].values())})",
        )
    ax.set_xticks(positions, tasks, rotation=30, ha="right")
    ax.set_ylabel("Retrieval latency per call (ms)")
    ax.set_title(
        "Figure 2: what the lookup costs\n"
        "bar = median, whisker = p05 to p95 over every retrieval in every complete cell"
    )
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    return _save(fig, output_dir, "figure2_retrieval_latency")


# ------------------------------------------------------------------------ fig 3


def _figure3(
    cells: Sequence[Mapping[str, Any]],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    """Where the time goes, and how many candidates each arm had to look at."""
    stages: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    totals: dict[str, list[float]] = defaultdict(list)
    candidates: dict[str, list[float]] = defaultdict(list)
    for cell in cells:
        if cell["arm"] == CONTROL_ARM:
            continue
        arm = str(cell["arm"])
        for entry in cell["telemetry"]:
            totals[arm].append(float(entry["total_ms"]))
            candidates[arm].append(float(entry.get("candidates_examined", 0) or 0))
            for key, _ in STAGE_KEYS:
                if key in entry:
                    stages[arm][key].append(float(entry[key]))
    arms = sorted(totals)
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(9.6, 4.4), constrained_layout=True, width_ratios=(1.35, 1.0)
    )
    positions = list(range(len(arms)))
    for arm_index, arm in enumerate(arms):
        reported = [(key, label) for key, label in STAGE_KEYS if stages[arm].get(key)]
        if reported:
            bottom = 0.0
            for depth, (key, label) in enumerate(reported):
                value = statistics.mean(stages[arm][key])
                ax_left.bar(
                    arm_index,
                    value,
                    0.56,
                    bottom=bottom,
                    color=ARM_COLOR.get(arm, PURPLE),
                    alpha=0.42 + 0.29 * depth,
                    edgecolor="white",
                    linewidth=0.8,
                )
                # A stage worth a few percent of the bar cannot hold a two-line
                # label inside it; that one gets written beside the segment.
                if value >= 0.14 * sum(
                    statistics.mean(stages[arm][k]) for k, _ in reported
                ):
                    ax_left.text(
                        arm_index,
                        bottom + value / 2,
                        f"{label}\n{value:.0f} ms",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
                else:
                    ax_left.text(
                        arm_index + 0.32,
                        bottom + value / 2,
                        f"{label} {value:.0f} ms",
                        ha="left",
                        va="center",
                        fontsize=7,
                        color="#333333",
                    )
                bottom += value
                _append(
                    records,
                    figure="3",
                    task="all",
                    arm=arm,
                    frequency="all",
                    metric=f"stage_{key}_mean",
                    value=round(value, 3),
                    unit="milliseconds",
                )
        else:
            value = statistics.mean(totals[arm])
            ax_left.bar(
                arm_index,
                value,
                0.56,
                color=ARM_COLOR.get(arm, PURPLE),
                edgecolor="white",
                linewidth=0.8,
            )
            ax_left.text(
                arm_index,
                value / 2,
                f"one fused pass\n{value:.0f} ms",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )
        _append(
            records,
            figure="3",
            task="all",
            arm=arm,
            frequency="all",
            metric="total_mean_ms",
            value=round(statistics.mean(totals[arm]), 3),
            unit="milliseconds",
        )
    ax_left.set_xticks(positions, [ARM_LABEL.get(a, a) for a in arms])
    ax_left.set_ylabel("Mean retrieval latency (ms)")
    ax_left.set_title("Where the time goes\n(an arm reporting no stages is one pass)")
    _style_axis(ax_left)

    means = [statistics.mean(candidates[arm]) for arm in arms]
    ax_right.bar(
        positions,
        [max(value, 0.0) for value in means],
        0.56,
        color=[ARM_COLOR.get(arm, PURPLE) for arm in arms],
    )
    for arm_index, (arm, value) in enumerate(zip(arms, means, strict=True)):
        ax_right.text(
            arm_index,
            max(value, 0.0),
            f"  {value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        _append(
            records,
            figure="3",
            task="all",
            arm=arm,
            frequency="all",
            metric="candidates_examined_mean",
            value=round(value, 2),
            unit="candidates/call",
        )
    ax_right.set_xticks(positions, [ARM_LABEL.get(a, a) for a in arms])
    ax_right.set_ylabel("Candidates examined per call (mean)")
    ax_right.set_title(
        "How much the predicate had to touch\n(0 = pushed into traversal)"
    )
    _style_axis(ax_right)
    fig.suptitle("Figure 3: the mechanism behind Figure 2", fontsize=11)
    return _save(fig, output_dir, "figure3_retrieval_anatomy")


# ----------------------------------------------------------------------- driver


def render(*, run_dir: Path, output_dir: Path) -> dict[str, Any]:
    cells = load_cells(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    outputs: list[Path] = []
    outputs += _figure1(cells, output_dir, records)
    outputs += _figure2(cells, output_dir, records)
    outputs += _figure3(cells, output_dir, records)

    csv_path = output_dir / "figure_data.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "figure",
                "task",
                "arm",
                "injection_frequency",
                "metric",
                "value",
                "unit",
            ),
        )
        writer.writeheader()
        writer.writerows(records)
    outputs.append(csv_path)

    per_arm: dict[str, dict[str, Any]] = {}
    for cell in cells:
        entry = per_arm.setdefault(
            str(cell["arm"]), {"cells": 0, "tasks": set(), "retrievals": 0}
        )
        entry["cells"] += 1
        entry["tasks"].add(cell["task"])
        entry["retrievals"] += len(cell["telemetry"])
    incomplete = sorted(
        path.parent.name
        for path in run_dir.glob("ale_*/run_receipt.json")
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "complete"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "sources": sorted(
            {
                str((cell["dir"] / "run_receipt.json").resolve()): _sha256(
                    cell["dir"] / "run_receipt.json"
                )
                for cell in cells
            }
        ),
        "arms": {
            arm: {
                "cells_plotted": entry["cells"],
                "tasks": sorted(entry["tasks"]),
                "retrieval_calls": entry["retrievals"],
            }
            for arm, entry in sorted(per_arm.items())
        },
        "cells_excluded": incomplete,
        "figures": sorted(
            path.name for path in outputs if path.suffix in {".png", ".pdf"}
        ),
        "data": csv_path.name,
        "comparability": (
            "Only cells whose own receipt says complete are plotted. An arm with "
            "fewer tasks or a smaller retrieval n is partial, not faster or better; "
            "read `arms` before comparing columns."
        ),
    }
    manifest_path = output_dir / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("bench/out/oe/ale_p4_0827_0110"),
        help="matrix output directory holding the ale_<task>__<arm>__... cells",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where the figures land; defaults to <run-dir>/figures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.run_dir / "figures"
    manifest = render(run_dir=args.run_dir, output_dir=output_dir)
    print(json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
