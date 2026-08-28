"""Retrieval latency: GEM's single bounded operator vs polyglot's three serial stages.

Two panels, because the two things a reader wants are on different scales and a
dual axis would be the wrong answer to both: total time per retrieval, and where
polyglot's time goes. GEM has no stage split to show -- one `tjs_open` pair inside
one transaction is the whole query -- so the breakdown panel is polyglot's alone
and is labelled as such rather than drawn as an empty GEM bar.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Slots 1-3 of the validated palette; see plot_p_sweep.py for the validator run.
#: A cell directory is `<domain>_<task>__<arm>[__rNNN][__pNNN][__sSEED]`. Matching
#: `math_*` silently skipped every ALE cell -- 30 of them -- and an empty table
#: reads exactly like "the runs failed".
CELL_GLOB = ("math_*", "ale_*")

ARM_STYLE = {
    "gem": ("#eb6834", "GEM (TriDB)"),
    "polyglot": ("#1baf7a", "Polyglot"),
    # Slot 4 of the documented order is yellow, which fails the all-pairs floors
    # against orange; the theme's violet step passes as a set (CVD dE 9.2).
    "cognee": ("#4a3aa7", "Cognee"),
}
STAGE_C = {"ann_ms": "#2a78d6", "traverse_ms": "#eb6834", "filter_rank_ms": "#1baf7a"}
STAGE_LABEL = {
    "ann_ms": "ANN (Milvus)",
    "traverse_ms": "Traverse (Neo4j)",
    "filter_rank_ms": "Filter + rank (pgvector)",
}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def collect(matrix: Path) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    for d in sorted({c for pat in CELL_GLOB for c in matrix.glob(pat)}):
        tel = d / "retrieval_telemetry.jsonl"
        if not tel.exists():
            continue
        name = d.name
        # Explicit, not an else-branch: `none` and `cognee` cells both land in
        # this directory, and an else-branch silently relabelled them.
        if "__gem__" in name:
            arm = "gem"
        elif "__polyglot__" in name:
            arm = "polyglot"
        elif "__cognee__" in name:
            arm = "cognee"
        else:
            continue
        task = name.split("__")[0].removeprefix("math_").removeprefix("ale_")
        rows = [
            json.loads(line) for line in tel.read_text().splitlines() if line.strip()
        ]
        out.setdefault(task, {}).setdefault(arm, []).extend(rows)
    return out


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d2")
    ax.grid(axis="y", color="#ecebe6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = collect(args.matrix)
    tasks = sorted(data)
    if not tasks:
        raise SystemExit("no retrieval telemetry found")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE)
    style(ax1)
    style(ax2)

    xs = range(len(tasks))
    w = 0.36
    arms = [
        a for a in ("gem", "polyglot", "cognee") if any(a in data[t] for t in tasks)
    ]
    w = 0.86 / max(len(arms), 1)
    for off, arm in enumerate(arms):
        color, label = ARM_STYLE[arm]
        med = [
            statistics.median(
                [r["total_ms"] for r in data[t].get(arm, []) if "total_ms" in r] or [0]
            )
            for t in tasks
        ]
        pos = [x + (off - (len(arms) - 1) / 2) * w for x in xs]
        # 2px surface gap between adjacent bars, per the mark spec.
        bars = ax1.bar(
            pos,
            med,
            w * 0.94,
            color=color,
            label=label,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        ax1.bar_label(bars, fmt="%.0f", padding=3, fontsize=8.5, color=MUTED)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8.5)
    ax1.set_ylabel("Median retrieval latency (ms)", fontsize=10, color=INK)
    ax1.set_title("One retrieval, end to end", fontsize=11, color=INK, pad=8)
    ax1.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")

    bottoms = [0.0] * len(tasks)
    for stage in ("ann_ms", "traverse_ms", "filter_rank_ms"):
        vals = [
            statistics.median(
                [r[stage] for r in data[t].get("polyglot", []) if stage in r] or [0]
            )
            for t in tasks
        ]
        ax2.bar(
            list(xs),
            vals,
            0.5,
            bottom=bottoms,
            color=STAGE_C[stage],
            label=STAGE_LABEL[stage],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=8.5)
    ax2.set_ylabel("Median stage latency (ms)", fontsize=10, color=INK)
    ax2.set_title(
        "Where polyglot's time goes (GEM has no stage split)",
        fontsize=11,
        color=INK,
        pad=8,
    )
    ax2.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")

    fig.suptitle(
        "Retrieval latency: one bounded operator vs three serial systems",
        fontsize=13,
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Polyglot cannot return a row until stage 3 finishes, so its first row and "
        "its last row arrive together.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")

    # One row per (task, arm), not a hardcoded GEM-vs-polyglot pair: the arms
    # present depend on which cells the matrix holds, and a fixed pair silently
    # dropped Cognee entirely.
    print(
        f"\n{'task':<22}{'arm':<10}{'median':>10}{'p90':>10}{'vs GEM':>8}{'queries':>9}"
    )
    for t in tasks:
        base = statistics.median(
            [r["total_ms"] for r in data[t].get("gem", []) if "total_ms" in r] or [0]
        )
        for arm in arms:
            vals = sorted(
                r["total_ms"] for r in data[t].get(arm, []) if "total_ms" in r
            )
            if not vals:
                continue
            med = statistics.median(vals)
            p90 = vals[min(len(vals) - 1, int(len(vals) * 0.9))]
            ratio = f"{med / base:.1f}x" if base and arm != "gem" else "--"
            print(
                f"{t:<22}{ARM_STYLE[arm][1]:<10}{med:>9.1f}ms{p90:>9.1f}ms"
                f"{ratio:>8}{len(vals):>9}"
            )

    print(f"\n{'task':<22}{'ann':>9}{'traverse':>10}{'filter':>9}   (polyglot 三段)")
    for t in tasks:
        st = {
            k: statistics.median(
                [r[k] for r in data[t].get("polyglot", []) if k in r] or [0]
            )
            for k in ("ann_ms", "traverse_ms", "filter_rank_ms")
        }
        print(
            f"{t:<22}{st['ann_ms']:>8.1f}ms{st['traverse_ms']:>9.1f}ms"
            f"{st['filter_rank_ms']:>8.1f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
