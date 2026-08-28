"""Figure: raw score and buggy share against one injection axis.

Two axes, one figure shape, because they answer different questions and were
measured in different batches:

* ``--x-axis p`` -- how OFTEN memory is injected (arXiv:2606.29823 Figure 4's p).
  Every injecting iteration carries the full 10 programs.
* ``--x-axis rate`` -- how MANY programs each injection carries, as a share of the
  10 reference slots. Injection happens every iteration.

They are orthogonal, and an early version of this figure labelled both as
percentages, which made 10 programs at p=1 and 1 program at p=1 indistinguishable
in the legend.

Two measures of different scale, so two ROWS of small multiples rather than one
dual-axis plot: the raw score is the primary metric and the buggy share is the
diagnostic that arXiv:2606.29823's Figure 4 puts on its y-axis. Columns are tasks,
and each column carries its own y range on the score row -- circle_packing scores
around 2.6 while heilbronn_triangle tops out below 1.0, so a shared axis would
flatten three panels to read one.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Categorical slots 1-3 of the validated reference palette. Small multiples use
#: the all-pairs pairlist, which caps a passing set at three slots -- and there are
#: exactly three arms. Verified with validate_palette.js --pairs all: worst CVD
#: dE 9.2, worst normal-vision dE 24.0. Aqua sits below 3:1 on the light surface,
#: so every series is ALSO direct-labelled; that relief is not optional.
#: A cell directory is `<domain>_<task>__<arm>[__rNNN][__pNNN][__sSEED]`. Matching
#: `math_*` silently skipped every ALE cell -- 30 of them -- and an empty table
#: reads exactly like "the runs failed".
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
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
PROBS = [0.1, 0.5, 1.0]
#: Injection sizes in the count sweep, as a share of the 10 reference slots.
RATES = [0.1, 0.5, 1.0]
AXIS_LABEL = {
    "p": "injection probability p",
    "rate": "injected programs (share of 10 slots)",
}
AXIS_NOTE = {
    "p": (
        "p = share of iterations that receive memory; the rest render no reference\n"
        "programs at all. 10 programs per injection."
    ),
    "rate": (
        "Share of the prompt's 10 reference slots filled from memory: 0.1 = 1\n"
        "program, 1 = all 10. Injection happens every iteration."
    ),
}


def _span(ax) -> float:
    lo, hi = ax.get_ylim()
    return hi - lo


def _label_endpoints(ax, endpoints: list[tuple[float, float, str]]) -> None:
    """Draw endpoint labels, pushed apart when the series converge.

    At p=1 three arms can land within a percentage point of each other, and
    matplotlib will happily print all three strings on the same pixels. A single
    pairwise nudge is not enough once there are more than two: the labels are laid
    out as one stack -- sorted, separated by a minimum gap, then slid back inside
    the frame as a block, and only compressed to fit when even that overflows.
    """
    if not endpoints:
        return
    lo, hi = ax.get_ylim()
    span = hi - lo
    gap = 0.075 * span
    items = sorted(endpoints)
    placed: list[float] = []
    for y, _, _ in items:
        placed.append(y if not placed else max(y, placed[-1] + gap))
    ceiling = hi - 0.02 * span
    floor = lo + 0.02 * span
    # Slide the stack down only as far as the overflow needs and the bottom allows.
    # An unconditional "spread evenly between floor and ceiling" fires whenever the
    # lowest series sits near zero -- which on the zero-score row is most panels --
    # and throws every label to the far side of the axis from its own point.
    overflow = placed[-1] - ceiling
    if overflow > 0:
        placed = [
            value - min(overflow, max(placed[0] - floor, 0.0)) for value in placed
        ]
    if placed[-1] > ceiling or placed[0] < floor - gap:
        # Still not fitting: the labels genuinely cannot hold `gap` inside the
        # frame, so spread them over what there is.
        room = ceiling - floor
        last = max(len(placed) - 1, 1)
        placed = [floor + room * index / last for index in range(len(placed))]
    for (y, x, text), target in zip(items, placed, strict=True):
        ax.annotate(
            text,
            (x, y),
            xytext=(x + 0.06, target),
            fontsize=8,
            color=MUTED,
            va="center",
        )


def read_cell(d: Path) -> dict | None:
    trace = d / "evolution_trace.jsonl"
    if not trace.exists():
        return None
    best, n, zero = 0.0, 0, 0
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
        zero += score == 0
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
    # Explicit, not an else-branch. The count-sweep batch also carries `none`
    # cells (the run's own programs, no external memory), and an else-branch
    # labelled them `polyglot` -- circle_packing then showed two Polyglot rows
    # when only one Polyglot cell exists.
    if "__nocontext" in name:
        arm = "nocontext"
    elif "__none" in name:
        arm = "none"
    elif "__gem__" in name:
        arm = "gem"
    elif "__polyglot__" in name:
        arm = "polyglot"
    elif "__cognee__" in name:
        arm = "cognee"
    else:
        return None
    freq = receipt.get("injection_frequency")
    # The count sweep predates the frequency axis and its receipts carry only the
    # rate; the frequency sweep carries both. Reading each from its own key keeps
    # one figure able to render either batch.
    rate = receipt.get("injection_rate_requested")
    # Iterations that actually rendered a program, among those whose gate was
    # open. GEM and the polyglot stack sit near 100% because both push
    # `fitness >= parent_fitness` INTO retrieval; Cognee ranks by similarity and
    # filters afterwards, so a query can return 60 chunks with no eligible row --
    # measured at 8 of 11 iterations on third_autocorr_ineq. Without this, a low
    # Cognee score reads as poor retrieval quality when the arm never injected.
    injected_rounds = gated_rounds = 0
    trace = d / "injection_trace.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("gate_open"):
                gated_rounds += 1
                if entry.get("rendered"):
                    injected_rounds += 1
    return {
        "inject_rate": (injected_rounds / gated_rounds) if gated_rounds else None,
        "task": name.split("__")[0].removeprefix("math_").removeprefix("ale_"),
        "seed": receipt.get("resolved_seed"),
        "arm": arm,
        # The receipt is the authority: a failed iteration writes no trace row, so
        # trace length undercounts by however many the run threw away.
        "p": None if arm == "nocontext" else freq,
        "rate": None if arm == "nocontext" else rate,
        "best": best,
        "buggy": zero / n,
        "iters": receipt.get("iterations_traced") or n,
        "status": receipt.get("status"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-iters", type=int, default=30)
    ap.add_argument(
        "--only-complete",
        action="store_true",
        help="Plot only tasks whose every arm/p cell finished. A task missing one "
        "cell draws a line that stops early, which reads as a measured trend "
        "rather than as absent data.",
    )
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--x-axis", choices=["p", "rate"], default="p")
    ap.add_argument(
        "--cols", type=int, default=5, help="tasks per block; ten across is unreadable"
    )
    ap.add_argument(
        "--programs",
        type=int,
        default=None,
        help="programs per injection, for the caption (math 10, ALE 4)",
    )
    args = ap.parse_args()

    rows = [
        r
        for r in (
            read_cell(d)
            for d in sorted({c for pat in CELL_GLOB for c in args.matrix.glob(pat)})
            if d.is_dir()
        )
        if r
    ]
    rows = [r for r in rows if r["iters"] >= args.min_iters]
    # `none` has no injection axis to sit on -- it is the own-programs-only arm.
    # Dropped from the figure and counted, rather than silently absent.
    dropped_none = sum(1 for r in rows if r["arm"] == "none")
    if dropped_none:
        print(f"excluding {dropped_none} `none` cell(s): no injection axis")
    rows = [r for r in rows if r["arm"] != "none"]
    tasks = sorted({r["task"] for r in rows})
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]
    if args.only_complete:
        # "Every cell present for this task finished", not a hardcoded count. The
        # matrix grew from 7 cells per task to 10 when the Cognee arm was added,
        # and a fixed 7 then excluded every task instead of none.
        full = {
            t
            for t in tasks
            if all(r["status"] == "complete" for r in rows if r["task"] == t)
        }
        dropped = [t for t in tasks if t not in full]
        if dropped:
            print(
                f"skipping {len(dropped)} task(s) still running: {', '.join(dropped)}"
            )
        tasks = [t for t in tasks if t in full]
    rows = [r for r in rows if r["task"] in tasks]
    if not tasks:
        raise SystemExit("no cell reached the minimum iteration count")

    # The figure narrows with the task count, so a fixed title size overflows at
    # two columns. Both strings are sized against the actual canvas width.
    width = 3.5 * len(tasks)
    title_size = min(13.0, 3.7 * width / len(tasks) * 0.98)
    sub_size = min(8.5, title_size - 3.5)

    axis = args.x_axis
    per_injection = args.programs or (
        4 if any(t.startswith("ahc") for t in tasks) else 10
    )
    ticks = PROBS if axis == "p" else RATES

    # Wrap into blocks instead of one row per measure. Ten ALE tasks side by side
    # produced a 7,000-pixel-wide figure that nothing can read; each block is a
    # (score, zero-score) pair over at most `--cols` tasks.
    cols = min(args.cols, len(tasks))
    blocks = (len(tasks) + cols - 1) // cols
    fig, axes = plt.subplots(
        2 * blocks,
        cols,
        figsize=(3.5 * cols, 6.6 * blocks),
        squeeze=False,
        facecolor=SURFACE,
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)
    for idx, task in enumerate(tasks):
        block, col = divmod(idx, cols)
        sub = [r for r in rows if r["task"] == task]
        # Averaged, not "the first one found". The count-sweep batch expanded
        # `nocontext` over the rate axis before that was fixed, so a task can carry
        # three no-context cells that are the SAME configuration run three times --
        # 0.9488 / 0.9361 / 0.9081 on circle_packing. Picking one silently turns a
        # repeat set into a reference line, and which one it picked depended on
        # directory order.
        no_ctx = [r for r in sub if r["arm"] == "nocontext"]
        base = (
            {k: statistics.mean([r[k] for r in no_ctx]) for k in ("best", "buggy")}
            if no_ctx
            else None
        )
        base_n = len(no_ctx)
        for row_i, (key, name) in enumerate(
            [("best", "Best score (raw)"), ("buggy", "Zero-score iterations")]
        ):
            ax = axes[2 * block + row_i][col]
            ax.set_visible(True)
            ax.set_facecolor(SURFACE)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#d8d7d2")
            ax.grid(axis="y", color="#ecebe6", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)

            endpoints: list[tuple[float, float, str]] = []
            for arm in ("gem", "polyglot", "cognee"):
                pts = sorted(
                    (
                        (r[axis], r[key])
                        for r in sub
                        if r["arm"] == arm and r.get(axis) is not None
                    ),
                    key=lambda t: t[0],
                )
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(
                    xs,
                    ys,
                    color=ARM_COLOR[arm],
                    linewidth=2,
                    marker="o",
                    markersize=8,
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                    zorder=3,
                    label=ARM_LABEL[arm],
                )
                # Direct label at the last point -- the relief the contrast WARN
                # on the aqua slot obligates, and it removes the legend lookup.
                # Collected, not drawn: two arms that converge at p=1 print on top
                # of each other, which is exactly where the reader most needs to
                # tell them apart.
                endpoints.append((ys[-1], xs[-1], ARM_LABEL[arm]))

            if base is not None:
                ax.axhline(
                    base[key],
                    color=ARM_COLOR["nocontext"],
                    linewidth=2,
                    linestyle=(0, (5, 3)),
                    zorder=2,
                )
                # Above or below the line depending on where the data sits, so the
                # label never lands on a series that runs along the same level.
                near = [
                    y for y, _, _ in endpoints if abs(y - base[key]) < 0.08 * _span(ax)
                ]
                ax.annotate(
                    "No context" if base_n < 2 else f"No context (n={base_n})",
                    (ticks[0], base[key]),
                    textcoords="offset points",
                    xytext=(0, -14 if near else 5),
                    fontsize=8,
                    color=ARM_COLOR["nocontext"],
                )

            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v:g}" for v in ticks])
            ax.set_xlim(0.0, 1.28)
            if row_i == 1:
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
                ax.set_xlabel(AXIS_LABEL[axis], fontsize=9, color=MUTED)
            if col == 0:
                ax.set_ylabel(name, fontsize=9.5, color=INK)
            if row_i == 0:
                ax.set_title(task, fontsize=10, color=INK, pad=8)

            # LAST, because the minimum gap between labels is a fraction of the y
            # span and both of the statements above move it: the no-context axhline
            # sits far below the memory arms (0.72 against 0.98 on
            # third_autocorr_ineq) and stretches the axis after autoscaling, and the
            # zero-score row is pinned to 0-1 outright. Placed before either, the
            # labels were separated against a span that no longer existed by the
            # time they were drawn, and printed on top of each other anyway.
            _label_endpoints(ax, endpoints)

    fig.suptitle(
        "Injection probability p: what the agent gains, and what it stops getting wrong"
        if axis == "p"
        else "Injection size: what the agent gains, and what it stops getting wrong",
        fontsize=title_size,
        color=INK,
        x=0.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.945 if blocks > 1 else 0.925,
        AXIS_NOTE[axis].replace(
            "10 programs per injection", f"{per_injection} programs per injection"
        ),
        ha="center",
        fontsize=sub_size,
        color=MUTED,
    )
    # Derived from the rows, never a fixed tuple. A hard-coded triple drew Cognee's
    # purple line in every panel and then left it out of the key -- the same failure
    # plot_latency.py's `collect` comment records, in the figure instead of the table.
    drawn = {r["arm"] for r in rows if r["arm"] in ARM_LABEL}
    legend_arms = [a for a in ("nocontext", "gem", "polyglot", "cognee") if a in drawn]
    handles = [
        plt.Line2D(
            [],
            [],
            color=ARM_COLOR[a],
            linewidth=2,
            linestyle=(0, (5, 3)) if a == "nocontext" else "-",
            marker="" if a == "nocontext" else "o",
            markersize=8,
            label=ARM_LABEL[a],
        )
        for a in legend_arms
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=MUTED,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.915))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}  ({len(rows)} cells, {len(tasks)} tasks)")

    print(
        f"\n{'task':<22}{'arm':<12}{'p':>5}{'best':>9}{'buggy':>8}"
        f"{'inject':>8}{'iters':>7}"
    )
    for r in sorted(rows, key=lambda r: (r["task"], r["arm"], r.get(axis) or 0)):
        p = "--" if r.get(axis) is None else f"{r[axis]:g}"
        inj = "--" if r.get("inject_rate") is None else f"{r['inject_rate']:.0%}"
        print(
            f"{r['task']:<22}{ARM_LABEL[r['arm']]:<12}{p:>5}"
            f"{r['best']:>9.4f}{r['buggy']:>7.0%}{inj:>8}{r['iters']:>7}"
        )
    print(
        "\n  inject = share of gate-open iterations that actually rendered a "
        "program.\n  GEM and Polyglot push `fitness >= parent` into retrieval and "
        "sit near 100%;\n  Cognee ranks by similarity and filters afterwards, so "
        "it can return 60 chunks\n  and no eligible row. Without this column a "
        "low score reads as poor retrieval\n  quality when the arm simply never "
        "injected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
