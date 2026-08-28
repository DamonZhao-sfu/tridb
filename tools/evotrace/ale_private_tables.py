"""Paper-aligned ALE private-performance tables and figures.

arXiv:2605.20086 §B.7 reports, per problem and per framework, the change in AtCoder
performance points from the seed to the run's final public-best program when
that program is re-scored on the held-out private test set.  Table 11 is that grid;
Table 12 counts the trajectories (aligned / mild overfit / severe overfit / no
movement).  :mod:`tools.evotrace.ale_private_rescore` produces exactly those two
numbers per cell; this module lays them out.

One deliberate departure.  The paper's columns are four FRAMEWORKS; ours are memory
arms at three injection frequencies, with the framework, model and seed held fixed.
That makes a difference between our columns attributable to memory rather than to
four different systems -- but it also means this is an analogue of the paper's
table, not a reproduction of it, and the two must not be read as the same
measurement.  The per-cell counts differ too: the paper scored 30 run/problem pairs,
this scores one per cell of the matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCHEMA_VERSION = "tridb_ale_private_tables_v0.2.0"
PERFORMANCE_THRESHOLDS = (400, 1600, 2000, 2400)
EXPECTED_PROBLEMS = {
    "ahc008",
    "ahc011",
    "ahc015",
    "ahc016",
    "ahc024",
    "ahc025",
    "ahc026",
    "ahc027",
    "ahc039",
    "ahc046",
}

# The categorical slots plot_p_sweep validated for this arm set (all-pairs pairlist,
# worst CVD dE 9.2, worst normal-vision dE 16.3). Reused unchanged so a reader moving
# between the p-sweep figures and these keeps the same colour for the same arm.
ARM_COLOR = {
    "nocontext": "#2a78d6",
    "gem": "#eb6834",
    "polyglot": "#1baf7a",
    "cognee": "#4a3aa7",
}
ARM_LABEL = {
    "nocontext": "No context",
    "gem": "GEM",
    "polyglot": "Polyglot",
    "cognee": "Cognee",
}
ARM_ORDER = ("nocontext", "gem", "polyglot", "cognee")
FREQ_MARKER = {0.1: "o", 0.5: "s", 1.0: "^"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9dee7"

#: The paper's grading: a public gain the private set does not confirm is
#: overfitting, severe past 200 performance points.
SEVERE_POINTS = 200


def _frequency(cell: str) -> float | None:
    """`ale_ahc008__gem__r100__p050` -> 0.5. The control carries no frequency."""
    match = re.search(r"__p(\d{3})$", cell)
    return None if match is None else int(match.group(1)) / 100.0


def _column(row: Mapping[str, Any]) -> str:
    arm = str(row["arm"])
    physical_plan = row.get("physical_plan")
    if physical_plan:
        return f"{ARM_LABEL.get(arm, arm)} {str(physical_plan).upper()}"
    freq = _frequency(str(row["cell"]))
    return (
        ARM_LABEL.get(arm, arm)
        if freq is None
        else f"{ARM_LABEL.get(arm, arm)} p={freq:g}"
    )


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} holds no re-scored cells")
    for row in rows:
        row["frequency"] = _frequency(str(row["cell"]))
        row["column"] = _column(row)
        # Public delta is meaningless across problems in absolute terms -- ahc016
        # scores in the tens of millions, ahc046 in the thousands -- so the scatter
        # needs it relative to where the run started.
        seed = float(row.get("public_seed_fitness") or 0.0)
        row["public_delta_pct"] = (
            None if seed == 0 else float(row["public_delta"]) / abs(seed) * 100.0
        )
    return rows


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    required = {
        "problem",
        "arm",
        "frequency",
        "column",
        "seed_sha256",
        "final_sha256",
        "private_case_count",
        "seed_private_rank",
        "seed_private_performance",
        "final_private_rank",
        "final_private_performance",
        "private_performance_delta_from_seed",
        "private_performance_delta_vs_nocontext",
        "generalization_label",
    }
    seen = set()
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"{row.get('cell')}: missing fields {sorted(missing)}")
        key = (row["problem"], row["column"])
        if key in seen:
            raise ValueError(f"duplicate task/group row: {key}")
        seen.add(key)
        groups[str(row["column"])].add(str(row["problem"]))
        if int(row["private_case_count"]) <= 0:
            raise ValueError(f"{row.get('cell')}: invalid private case count")
    for group, problems in groups.items():
        if problems != EXPECTED_PROBLEMS:
            raise ValueError(
                f"{group}: incomplete problem coverage; missing "
                f"{sorted(EXPECTED_PROBLEMS - problems)}"
            )
    if "No context" not in groups:
        raise ValueError("No context control group is required")
    return {group: len(problems) for group, problems in groups.items()}


def _columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen = {str(r["column"]) for r in rows}
    ordered = []
    for arm in ARM_ORDER:
        label = ARM_LABEL[arm]
        if label in seen:
            ordered.append(label)
        for freq in (0.1, 0.5, 1.0):
            candidate = f"{label} p={freq:g}"
            if candidate in seen:
                ordered.append(candidate)
    return ordered + sorted(seen - set(ordered))


def _ordered_group_names(groups: Mapping[str, Any]) -> list[str]:
    return [name for name in _columns([{"column": name} for name in groups])]


# ------------------------------------------------------------------- table 11


def table11(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], dict]:
    problems = sorted({str(r["problem"]) for r in rows})
    columns = _columns(rows)
    grid: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        grid[(str(row["problem"]), str(row["column"]))] = row
    return problems, columns, grid


def _render_table11(problems, columns, grid) -> str:
    width = max(len(c) for c in columns) + 2
    head = "problem".ljust(11) + "".join(c.rjust(width) for c in columns)
    lines = [head, "-" * len(head)]
    for problem in problems:
        cells = []
        for column in columns:
            row = grid.get((problem, column))
            if row is None:
                cells.append("—".rjust(width))
                continue
            delta = int(row["private_performance_delta_from_seed"])
            # `*` marks the paper's bold: a public gain the private set contradicts.
            mark = (
                "*" if str(row["generalization_label"]).startswith("overfit") else " "
            )
            cells.append(f"{delta:+,}{mark}".rjust(width))
        lines.append(problem.ljust(11) + "".join(cells))
    lines.append("")
    lines.append("* = overfitting (public up, private down)   — = cell not scored")
    return "\n".join(lines)


# ------------------------------------------------------------------- table 12


VERDICTS = ("aligned", "overfit(mild)", "overfit(severe)", "no movement")
VERDICT_LABELS = {
    "aligned": "Aligned",
    "overfit(mild)": "Mild overfit",
    "overfit(severe)": "Severe overfit",
    "no movement": "No movement",
}
VERDICT_COLORS = {
    "aligned": "#1b8f67",
    "overfit(mild)": "#e5a93d",
    "overfit(severe)": "#cf4f45",
    "no movement": "#9b9a96",
}


def table12(rows: Sequence[Mapping[str, Any]]) -> dict[str, Counter]:
    """Verdict counts per arm *and frequency*, not collapsed across the sweep."""
    per_arm: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per_arm[str(row["column"])][str(row["generalization_label"])] += 1
    return per_arm


def _render_table12(per_arm: Mapping[str, Counter]) -> str:
    head = (
        f"{'arm/frequency':<20}{'scored':>8}{'aligned':>9}"
        f"{'mild':>7}{'severe':>8}{'no move':>9}"
    )
    lines = [head, "-" * len(head)]
    for group in _ordered_group_names(per_arm):
        counts = per_arm[group]
        lines.append(
            f"{group:<20}{sum(counts.values()):>8}"
            f"{counts['aligned']:>9}{counts['overfit(mild)']:>7}"
            f"{counts['overfit(severe)']:>8}{counts['no movement']:>9}"
        )
    return "\n".join(lines)


# -------------------------------------------------------------------- scatter


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def paper_table11_figure(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    """Render the paper's Table 11 layout for the EvoTrace memory groups."""
    problems, columns, grid = table11(rows)
    values = []
    for problem in problems:
        row_values = []
        for column in columns:
            row = grid.get((problem, column))
            row_values.append(
                "—"
                if row is None
                else f"{int(row['private_performance_delta_from_seed']):+,}"
            )
        values.append(row_values)

    fig, ax = plt.subplots(
        figsize=(8.8, 4.9), constrained_layout=True, facecolor=SURFACE
    )
    ax.axis("off")
    ax.set_title(
        "ALE private generalization: seed → final public-best",
        fontsize=11,
        color=INK,
        pad=13,
        weight="semibold",
    )
    table = ax.table(
        cellText=values,
        rowLabels=problems,
        colLabels=columns,
        cellLoc="right",
        rowLoc="left",
        colLoc="center",
        bbox=(0.04, 0.11, 0.92, 0.80),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor(SURFACE)
        cell.set_edgecolor("#d8d7d2")
        cell.set_linewidth(0.55)
        cell.get_text().set_color(INK)
        if row_index == 0 or column_index == -1:
            cell.get_text().set_weight("semibold")
        if row_index == 0:
            cell.set_facecolor("#f0f1f3")

    for row_index, problem in enumerate(problems, start=1):
        for column_index, column in enumerate(columns):
            row = grid.get((problem, column))
            if row is None:
                continue
            if str(row["generalization_label"]).startswith("overfit"):
                cell = table[(row_index, column_index)]
                cell.get_text().set_weight("bold")
                cell.get_text().set_color("#9e2f2a")
                cell.set_facecolor("#fff0ed")

    fig.text(
        0.5,
        0.035,
        "Cell = Δ private performance points. Bold red = public improved but private worsened.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    return _save_figure(fig, output_dir, "table11_private_delta")


def paper_table12_figure(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    """Render the paper's Table 12 count layout for the EvoTrace groups."""
    counts_by_group = table12(rows)
    groups = _ordered_group_names(counts_by_group)
    columns = ("Scored", "Aligned", "Mild", "Severe", "No movement")
    values = []
    for group in groups:
        counts = counts_by_group[group]
        values.append(
            [
                sum(counts.values()),
                counts["aligned"],
                counts["overfit(mild)"],
                counts["overfit(severe)"],
                counts["no movement"],
            ]
        )

    height = max(2.7, 1.45 + 0.52 * len(groups))
    fig, ax = plt.subplots(
        figsize=(8.8, height), constrained_layout=True, facecolor=SURFACE
    )
    ax.axis("off")
    ax.set_title(
        "ALE public→private trajectory counts",
        fontsize=11,
        color=INK,
        pad=13,
        weight="semibold",
    )
    table = ax.table(
        cellText=values,
        rowLabels=groups,
        colLabels=columns,
        cellLoc="center",
        rowLoc="left",
        colLoc="center",
        bbox=(0.04, 0.16, 0.92, 0.70),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor(SURFACE)
        cell.set_edgecolor("#d8d7d2")
        cell.set_linewidth(0.55)
        cell.get_text().set_color(INK)
        if row_index == 0 or column_index == -1:
            cell.get_text().set_weight("semibold")
        if row_index == 0:
            cell.set_facecolor("#f0f1f3")
    return _save_figure(fig, output_dir, "table12_verdict_counts")


def verdict_bars(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    """Plot Table 12 as stacked counts without changing its classification."""
    counts_by_group = table12(rows)
    groups = _ordered_group_names(counts_by_group)
    fig, ax = plt.subplots(
        figsize=(8.2, 4.8), constrained_layout=True, facecolor=SURFACE
    )
    bottoms = [0] * len(groups)
    for verdict in VERDICTS:
        values = [counts_by_group[group][verdict] for group in groups]
        bars = ax.bar(
            groups,
            values,
            bottom=bottoms,
            width=0.68,
            color=VERDICT_COLORS[verdict],
            edgecolor=SURFACE,
            linewidth=0.8,
            label=VERDICT_LABELS[verdict],
        )
        for bar, value in zip(bars, values):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if verdict != "overfit(mild)" else INK,
                    weight="semibold",
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_ylabel("Tasks", fontsize=9.5, color=INK)
    ax.set_title(
        "Held-out generalization by memory arm",
        fontsize=11,
        color=INK,
        pad=10,
        weight="semibold",
    )
    ax.set_ylim(0, max(bottoms, default=0) + 0.7)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
    _style(ax)
    return _save_figure(fig, output_dir, "figure12_verdict_counts")


def scatter(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    """Public gain against private performance change."""
    points = [r for r in rows if r["public_delta_pct"] is not None]
    fig, ax = plt.subplots(
        figsize=(8.2, 5.6), constrained_layout=True, facecolor=SURFACE
    )
    xs = [float(r["public_delta_pct"]) for r in points]
    ys = [float(r["private_performance_delta_from_seed"]) for r in points]
    if xs and ys:
        pad_x = max(max(map(abs, xs)) * 0.12, 1.0)
        pad_y = max(max(map(abs, ys)) * 0.12, 20.0)
        left, right = min(xs) - pad_x, max(xs) + pad_x
        low, high = min(ys) - pad_y, max(ys) + pad_y
        # The quadrant IS the finding, so it is drawn rather than described: a run
        # that gained on the public score and lost performance on held-out cases.
        ax.axhspan(
            low,
            0,
            xmin=(0 - left) / (right - left),
            facecolor="#e34948",
            alpha=0.06,
            zorder=0,
        )
        ax.set_xlim(left, right)
        ax.set_ylim(low, high)
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.axvline(0, color=MUTED, linewidth=1.0, zorder=2)
    seen: set[tuple[str, float | None]] = set()
    for row in points:
        arm = str(row["arm"])
        freq = row["frequency"]
        key = (arm, freq)
        label = None
        if key not in seen:
            seen.add(key)
            label = ARM_LABEL.get(arm, arm) + ("" if freq is None else f" p={freq:g}")
        ax.scatter(
            row["public_delta_pct"],
            row["private_performance_delta_from_seed"],
            s=64,
            color=ARM_COLOR.get(arm, "#4a3aa7"),
            marker=FREQ_MARKER.get(freq, "D"),
            edgecolor=SURFACE,
            linewidth=1.4,
            zorder=3,
            label=label,
        )
    # One label per problem, on its most extreme cell. Labelling every cell that
    # crosses the threshold printed the same problem name three times on top of
    # itself wherever its arms landed together, which is most of them.
    extreme: dict[str, Mapping[str, Any]] = {}
    for row in points:
        if abs(float(row["private_performance_delta_from_seed"])) < SEVERE_POINTS:
            continue
        current = extreme.get(str(row["problem"]))
        if current is None or abs(
            float(row["private_performance_delta_from_seed"])
        ) > abs(float(current["private_performance_delta_from_seed"])):
            extreme[str(row["problem"])] = row
    for problem, row in extreme.items():
        ax.annotate(
            problem,
            (row["public_delta_pct"], row["private_performance_delta_from_seed"]),
            xytext=(
                7,
                5 if float(row["private_performance_delta_from_seed"]) >= 0 else -11,
            ),
            textcoords="offset points",
            fontsize=7.5,
            color=MUTED,
        )
    ax.set_xlabel(
        "Public score change from seed to final best (% of seed)",
        fontsize=9.5,
        color=INK,
    )
    ax.set_ylabel("Private performance change", fontsize=9.5, color=INK)
    ax.set_title(
        "Public gains that the held-out set does not confirm\n"
        "shaded quadrant: public up, private performance down",
        fontsize=11,
        color=INK,
        pad=10,
    )
    # Below the axes, not inside them: an in-frame legend sat on the very cells the
    # figure exists to show, and the fourth quadrant is where they land.
    ax.legend(
        frameon=False,
        fontsize=8,
        labelcolor=MUTED,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    _style(ax)
    return _save_figure(fig, output_dir, "figure5_public_vs_private")


# --------------------------------------------------------------------- driver


def _long_form(rows: Sequence[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        base = {
            "problem": row["problem"],
            "arm": row["arm"],
            "injection_frequency": "" if row["frequency"] is None else row["frequency"],
            "cell": row["cell"],
            "verdict": row["generalization_label"],
        }
        for metric, value, unit in (
            ("public_seed_fitness", row["public_seed_fitness"], "score"),
            ("public_final_fitness", row["public_final_fitness"], "score"),
            ("public_delta", row["public_delta"], "score"),
            ("public_delta_pct", row["public_delta_pct"], "percent"),
            (
                "seed_private_performance",
                row["seed_private_performance"],
                "performance_point",
            ),
            (
                "final_private_performance",
                row["final_private_performance"],
                "performance_point",
            ),
            (
                "private_performance_delta_from_seed",
                row["private_performance_delta_from_seed"],
                "performance_point",
            ),
            (
                "private_performance_delta_vs_nocontext",
                row["private_performance_delta_vs_nocontext"],
                "performance_point",
            ),
        ):
            yield {**base, "metric": metric, "value": value, "unit": unit}


def aggregate_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["column"])].append(row)
    output = []
    for group in _ordered_group_names(grouped):
        members = grouped[group]
        performances = [int(row["final_private_performance"]) for row in members]
        seed_deltas = [
            int(row["private_performance_delta_from_seed"]) for row in members
        ]
        control_deltas = [
            int(row["private_performance_delta_vs_nocontext"])
            for row in members
            if row["private_performance_delta_vs_nocontext"] is not None
        ]
        record = {
            "group": group,
            "arm": members[0]["arm"],
            "injection_frequency": members[0]["frequency"],
            "tasks": len(members),
            "mean_private_performance": sum(performances) / len(performances),
            "mean_private_performance_delta_from_seed": sum(seed_deltas)
            / len(seed_deltas),
            "mean_private_performance_delta_vs_nocontext": (
                None
                if not control_deltas
                else sum(control_deltas) / len(control_deltas)
            ),
        }
        for threshold in PERFORMANCE_THRESHOLDS:
            count = sum(value >= threshold for value in performances)
            record[f"performance_ge_{threshold}_count"] = count
            record[f"performance_ge_{threshold}_fraction"] = count / len(performances)
        output.append(record)
    return output


def paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "problem": row["problem"],
            "arm": row["arm"],
            "injection_frequency": row["frequency"],
            "final_private_performance": row["final_private_performance"],
            "private_performance_delta_vs_nocontext": row[
                "private_performance_delta_vs_nocontext"
            ],
        }
        for row in rows
        if row["arm"] != "nocontext"
    ]


def render(*, private_json: Path, output_dir: Path) -> dict[str, Any]:
    rows = load_rows(private_json)
    group_coverage = validate_rows(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    problems, columns, grid = table11(rows)
    per_arm = table12(rows)

    t11 = _render_table11(problems, columns, grid)
    t12 = _render_table12(per_arm)
    (output_dir / "table11_performance_delta.txt").write_text(
        t11 + "\n", encoding="utf-8"
    )
    (output_dir / "table12_verdict_counts.txt").write_text(t12 + "\n", encoding="utf-8")

    with (output_dir / "table11_performance_delta.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["problem", *columns])
        for problem in problems:
            writer.writerow(
                [problem]
                + [
                    ""
                    if (problem, column) not in grid
                    else grid[(problem, column)]["private_performance_delta_from_seed"]
                    for column in columns
                ]
            )
    with (output_dir / "table12_verdict_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm_frequency", "scored", *VERDICTS])
        for group in _ordered_group_names(per_arm):
            counts = per_arm[group]
            writer.writerow(
                [group, sum(counts.values()), *[counts[v] for v in VERDICTS]]
            )

    aggregates = aggregate_groups(rows)
    with (output_dir / "group_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    (output_dir / "group_summary.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8"
    )

    paired = paired_rows(rows)
    with (output_dir / "paired_vs_nocontext.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    long_rows = list(_long_form(rows))
    with (output_dir / "private_figure_data.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "problem",
                "arm",
                "injection_frequency",
                "cell",
                "verdict",
                "metric",
                "value",
                "unit",
            ),
        )
        writer.writeheader()
        writer.writerows(long_rows)

    figures = (
        paper_table11_figure(rows, output_dir)
        + paper_table12_figure(rows, output_dir)
        + verdict_bars(rows, output_dir)
        + scatter(rows, output_dir)
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(private_json.resolve()),
        "source_sha256": hashlib.sha256(private_json.read_bytes()).hexdigest(),
        "cells_scored": len(rows),
        "group_coverage": group_coverage,
        "completeness_gate": "passed",
        "problems": problems,
        "columns": columns,
        "verdicts": {arm: dict(counts) for arm, counts in sorted(per_arm.items())},
        "severe_threshold_performance_points": SEVERE_POINTS,
        "performance_distribution_thresholds": PERFORMANCE_THRESHOLDS,
        "method": (
            "arXiv:2605.20086 B.7: change in AtCoder performance from "
            "the seed to the run's final public-best program, re-scored on the "
            "held-out private test set. aligned = public and private both up; "
            "mild overfit = public up, private down by <= 200 points; severe = "
            "more than 200; no movement otherwise."
        ),
        "not_a_reproduction": (
            "The paper's columns are four frameworks; these are memory arms at "
            "three injection frequencies with framework, model and seed fixed. "
            "Same metric, different question -- do not present as the paper's "
            "Table 11."
        ),
        "outputs": sorted(p.name for p in figures)
        + [
            "table11_performance_delta.txt",
            "table11_performance_delta.csv",
            "table12_verdict_counts.txt",
            "table12_verdict_counts.csv",
            "group_summary.csv",
            "group_summary.json",
            "paired_vs_nocontext.csv",
            "private_figure_data.csv",
        ],
    }
    (output_dir / "private_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    print(t11)
    print()
    print(t12)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-json", type=Path, default=Path("results/e2/ale_private.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/e2/private"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = render(private_json=args.private_json, output_dir=args.output_dir)
    print()
    print(json.dumps({k: v for k, v in manifest.items() if k != "columns"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
