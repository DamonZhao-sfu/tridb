"""Export the numbers behind results/e2 fig1, fig2 and fig3 as one long-form CSV.

The three figures are drawn by :mod:`tools.e2.plot_p_sweep` (fig1 and fig2, one
per injection axis) and :mod:`tools.e2.plot_latency` (fig3), from two matrix
directories.  This exporter imports those modules' own readers rather than
re-deriving anything, so a row here cannot drift from the bar it was plotted as --
`best` including the seed fitness, `buggy` as a share of traced rows, and the
median over every retrieval in every cell of a task+arm all come from the
plotters.

Schema follows the ALE figure export
(``bench/out/oe/ale_p4_0827_0110/figures/figure_data.csv``) with one column added:
fig2 sweeps the injection RATE while fig1 and fig3 sweep the FREQUENCY, and folding
the two into one column is exactly the mistake plot_p_sweep's docstring records.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.e2.plot_latency import collect as collect_latency  # noqa: E402
from tools.e2.plot_p_sweep import CELL_GLOB, read_cell  # noqa: E402

SCHEMA_VERSION = "tridb_e2_math_figure_data_v0.1.0"

#: The four math tasks the three figures show. fig2's matrix also holds
#: `first_autocorr_ineq`, `second_autocorr_ineq` and `uncertainty_ineq`, but only
#: at a single injection rate, so they carry no sweep and are not plotted.
TASKS = (
    "circle_packing",
    "heilbronn_convex_13",
    "heilbronn_triangle",
    "third_autocorr_ineq",
)

FIELDS = (
    "figure",
    "task",
    "arm",
    "injection_frequency",
    "injection_rate",
    "metric",
    "value",
    "unit",
    "source",
)


def _cells(matrix: Path) -> list[dict[str, Any]]:
    seen = {c for pattern in CELL_GLOB for c in matrix.glob(pattern)}
    rows = [read_cell(d) for d in sorted(seen)]
    return [r for r in rows if r is not None and r["task"] in TASKS]


def _row(**kw: Any) -> dict[str, Any]:
    return {field: kw.get(field, "") for field in FIELDS}


def _sweep_rows(matrix: Path, *, figure: str, axis: str) -> Iterable[dict[str, Any]]:
    """fig1 (axis='p') and fig2 (axis='rate') share one shape and one reader."""
    cells = _cells(matrix)
    source = str(matrix)
    for cell in cells:
        if cell["arm"] == "nocontext":
            continue
        common = {
            "figure": figure,
            "task": cell["task"],
            "arm": cell["arm"],
            "injection_frequency": "" if cell["p"] is None else cell["p"],
            "injection_rate": "" if cell["rate"] is None else cell["rate"],
            "source": source,
        }
        yield _row(**common, metric="best_score", value=cell["best"], unit="score")
        yield _row(
            **common,
            metric="zero_score_fraction",
            value=round(cell["buggy"], 6),
            unit="fraction",
        )
        yield _row(
            **common, metric="iterations_traced", value=cell["iters"], unit="iterations"
        )
        if cell["inject_rate"] is not None:
            # Share of gate-open iterations that actually rendered a program.
            # Cognee ranks by similarity and filters afterwards, so a query can
            # come back full and still inject nothing; without this a low Cognee
            # score reads as bad retrieval when the arm never injected at all.
            yield _row(
                **common,
                metric="injection_render_rate",
                value=round(cell["inject_rate"], 6),
                unit="fraction",
            )

    # The control is a flat reference line, averaged over its repeats -- that mean
    # is the value the figure draws, so it is exported as its own row rather than
    # left for the reader to recompute.
    for task in TASKS:
        control = [c for c in cells if c["task"] == task and c["arm"] == "nocontext"]
        if not control:
            continue
        common = {
            "figure": figure,
            "task": task,
            "arm": "nocontext",
            "source": source,
        }
        for metric, key, unit in (
            ("best_score", "best", "score"),
            ("zero_score_fraction", "buggy", "fraction"),
        ):
            yield _row(
                **common,
                metric=metric,
                value=round(statistics.mean([c[key] for c in control]), 6),
                unit=f"{unit} (mean of n={len(control)})",
            )
        for index, cell in enumerate(control, start=1):
            yield _row(
                **common,
                metric=f"best_score_repeat_{index}",
                value=cell["best"],
                unit="score",
            )
            yield _row(
                **common,
                metric=f"zero_score_fraction_repeat_{index}",
                value=round(cell["buggy"], 6),
                unit="fraction",
            )


def _latency_rows(matrix: Path) -> Iterable[dict[str, Any]]:
    data = collect_latency(matrix)
    source = str(matrix)
    for task in TASKS:
        per_arm = data.get(task, {})
        for arm in ("gem", "polyglot", "cognee"):
            values = sorted(
                r["total_ms"] for r in per_arm.get(arm, []) if "total_ms" in r
            )
            if not values:
                continue
            common = {
                "figure": "3",
                "task": task,
                "arm": arm,
                # Pooled over all three frequency cells: one retrieval is one
                # retrieval whatever iteration asked for it.
                "injection_frequency": "all",
                "injection_rate": 1.0,
                "source": source,
            }
            p90 = values[min(len(values) - 1, int(len(values) * 0.9))]
            yield _row(
                **common,
                metric="retrieval_median_ms",
                value=round(statistics.median(values), 3),
                unit="milliseconds",
            )
            yield _row(
                **common,
                metric="retrieval_p90_ms",
                value=round(p90, 3),
                unit="milliseconds",
            )
            yield _row(
                **common,
                metric="retrieval_mean_ms",
                value=round(statistics.mean(values), 3),
                unit="milliseconds",
            )
            yield _row(
                **common, metric="retrieval_calls", value=len(values), unit="calls"
            )
            if arm == "polyglot":
                for stage in ("ann_ms", "traverse_ms", "filter_rank_ms"):
                    staged = [r[stage] for r in per_arm[arm] if stage in r]
                    if not staged:
                        continue
                    yield _row(
                        **common,
                        metric=f"stage_{stage}_median",
                        value=round(statistics.median(staged), 3),
                        unit="milliseconds",
                    )


def export(*, p_matrix: Path, rate_matrix: Path, out: Path) -> dict[str, Any]:
    rows = [
        *_sweep_rows(p_matrix, figure="1", axis="p"),
        *_sweep_rows(rate_matrix, figure="2", axis="rate"),
        *_latency_rows(p_matrix),
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "tasks": list(TASKS),
        "figures": {
            "1": {
                "title": "Injection probability p",
                "matrix": str(p_matrix),
                "plotter": "tools/e2/plot_p_sweep.py --x-axis p",
                "png": "results/e2/fig1_injection_probability.png",
            },
            "2": {
                "title": "Injection size (share of the 10 reference slots)",
                "matrix": str(rate_matrix),
                "plotter": "tools/e2/plot_p_sweep.py --x-axis rate",
                "png": "results/e2/fig2_injection_size.png",
            },
            "3": {
                "title": "Retrieval latency",
                "matrix": str(p_matrix),
                "plotter": "tools/e2/plot_latency.py",
                "png": "results/e2/fig3_latency.png",
            },
        },
        "notes": (
            "best_score includes the seed fitness, as the plotters do: a run whose "
            "every child scored worse than its starting program otherwise reports a "
            "best below where it began. zero_score_fraction is over traced rows. "
            "fig1 and fig3 come from the frequency batch, fig2 from the earlier "
            "count batch -- the axes are orthogonal and the batches are different, "
            "so a fig1 row and a fig2 row at the same task are not two points on "
            "one curve."
        ),
        "out": str(out),
    }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p-matrix",
        type=Path,
        default=Path("bench/out/oe/freq_0826_1045"),
        help="frequency sweep; feeds fig1 and fig3",
    )
    parser.add_argument(
        "--rate-matrix",
        type=Path,
        default=Path("bench/out/oe/all7_0825_2218"),
        help="injection-count sweep; feeds fig2",
    )
    parser.add_argument("--out", type=Path, default=Path("results/e2/figure_data.csv"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export(
        p_matrix=args.p_matrix, rate_matrix=args.rate_matrix, out=args.out
    )
    manifest_path = args.out.with_name("figure_data_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
