"""Is the arm difference bigger than the run-to-run spread?

One dot per repeat, a bar for the mean, per (task, arm). The question this figure
exists to answer is not "which arm scored higher" -- a single pair of runs always
answers that -- but "would the ordering survive another draw of the seed". Three
identical `nocontext` cells on this workload already differed by 0.148, which is
larger than most arm gaps measured at one seed, so the dots are the finding and
the bar is only a summary of them.

Raw score, not normalised: raw is the primary metric, and each task keeps its own
panel because raw scores are only comparable within a task.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CELL_GLOB = ("math_*", "ale_*")
#: Slot 4 of the documented order is yellow, which fails the all-pairs floors
#: against orange (normal-vision dE 13.7). Small multiples use the all-pairs
#: pairlist, so the fourth arm takes the theme's violet step instead, re-validated
#: as a set: worst all-pairs CVD dE 9.2, worst normal-vision dE 16.3.
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
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def read_cell(d: Path) -> dict | None:
    trace = d / "evolution_trace.jsonl"
    if not trace.exists():
        return None
    best, n = 0.0, 0
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = (row.get("child_metrics") or {}).get("combined_score")
        if score is None:
            continue
        n += 1
        best = max(best, score)
    if not n:
        return None
    try:
        receipt = json.loads((d / "run_receipt.json").read_text())
    except (OSError, json.JSONDecodeError):
        receipt = {}
    # best-so-far INCLUDES the seed. The trace records children only, so a run whose
    # every child scored worse than the program it started from reported a "best"
    # below its own starting point -- measured on ale:ahc008's `nocontext` cell,
    # which read as -555,288 when the run had simply never improved on the seed.
    seed_fitness = receipt.get("seed_fitness")
    if seed_fitness is not None:
        best = max(best, float(seed_fitness))
    name = d.name
    if "__nocontext" in name:
        arm = "nocontext"
    elif "__gem__" in name:
        arm = "gem"
    elif "__polyglot__" in name:
        arm = "polyglot"
    elif "__cognee__" in name:
        arm = "cognee"
    else:
        return None
    return {
        "task": name.split("__")[0].removeprefix("math_").removeprefix("ale_"),
        "arm": arm,
        "seed": receipt.get("resolved_seed"),
        "best": best,
        "iters": receipt.get("iterations_traced") or n,
        "status": receipt.get("status"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-iters", type=int, default=30)
    args = ap.parse_args()

    cells = sorted(
        {c for pat in CELL_GLOB for c in args.matrix.glob(pat) if c.is_dir()}
    )
    rows = [
        r for r in (read_cell(d) for d in cells) if r and r["iters"] >= args.min_iters
    ]
    tasks = sorted({r["task"] for r in rows})
    if not tasks:
        raise SystemExit("no cell reached the minimum iteration count")

    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.6 * len(tasks) + 0.6, 4.8),
        squeeze=False,
        facecolor=SURFACE,
    )
    for col, task in enumerate(tasks):
        ax = axes[0][col]
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d2")
        ax.grid(axis="y", color="#ecebe6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)

        present = [
            a
            for a in ARM_ORDER
            if any(r["task"] == task and r["arm"] == a for r in rows)
        ]
        for i, arm in enumerate(present):
            vals = [r["best"] for r in rows if r["task"] == task and r["arm"] == arm]
            mean = statistics.mean(vals)
            ax.hlines(
                mean, i - 0.28, i + 0.28, color=ARM_COLOR[arm], linewidth=2, zorder=3
            )
            # Repeats jittered apart so two equal scores do not read as one run.
            for j, v in enumerate(sorted(vals)):
                off = (j - (len(vals) - 1) / 2) * 0.11
                ax.plot(
                    i + off,
                    v,
                    "o",
                    markersize=8,
                    color=ARM_COLOR[arm],
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                    zorder=4,
                )
            spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
            ax.annotate(
                f"±{spread / 2:.3f}" if spread else "n=1",
                (i, mean),
                textcoords="offset points",
                xytext=(0, -20),
                ha="center",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([ARM_LABEL[a] for a in present], fontsize=9)
        ax.set_xlim(-0.6, len(present) - 0.4)
        ax.set_title(task, fontsize=10, color=INK, pad=8)
        if col == 0:
            ax.set_ylabel("Best score (raw)", fontsize=10, color=INK)

    fig.suptitle(
        "Three repeats per arm: does the ordering survive the seed?",
        fontsize=13,
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.915,
        "One dot per seed, bar = mean, ±half the observed range. An arm gap "
        "smaller than the dots is not a result.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}  ({len(rows)} cells, {len(tasks)} tasks)")

    print(
        f"\n{'task':<24}{'arm':<12}{'n':>3}{'均值':>10}{'最小':>10}{'最大':>10}{'极差':>9}"
    )
    for task in tasks:
        for arm in ARM_ORDER:
            vals = sorted(
                r["best"] for r in rows if r["task"] == task and r["arm"] == arm
            )
            if not vals:
                continue
            print(
                f"{task:<24}{ARM_LABEL[arm]:<12}{len(vals):>3}"
                f"{statistics.mean(vals):>10.4f}{vals[0]:>10.4f}{vals[-1]:>10.4f}"
                f"{vals[-1] - vals[0]:>9.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
