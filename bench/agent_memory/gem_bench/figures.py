"""Render TriDB/GEM analogues of [AM] Fig. 2, 3, 9, 10 and 11.

Fig. 2/3/10/11 consume the benchmark's canonical ``paper_sections.json``;
they never re-derive timing attribution from JSONL records.  Fig. 9 consumes
``scale_results.json`` produced by :mod:`.scaling`.  Every plotted value is
also exported in a long-form CSV so the bitmap is not the only artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bench.agent_memory.serving import _atomic_write_json, _sha256

SCHEMA_VERSION = "tridb_gem_paper_figures_v0.1.0"
BLUE = "#2878B5"
ORANGE = "#E07A1F"
GREEN = "#3A923A"
PURPLE = "#7B5AA6"
GRID = "#D9DEE7"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _selected_rows(
    sections: Mapping[str, Any], section: str, points: set[str] | None
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in sections[section]["rows"]]
    if points is not None:
        rows = [row for row in rows if row["operating_point"] in points]
    if not rows:
        requested = "all points" if points is None else sorted(points)
        raise ValueError(f"no {section} rows selected for {requested}")
    return rows


def _labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["operating_point"]) for row in rows]


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


def _summary_for(input_dir: Path, point: str) -> dict[str, Any] | None:
    path = input_dir / point / "summary.json"
    return _load_json(path) if path.exists() else None


def _append_metric(
    records: list[dict[str, Any]],
    *,
    figure: str,
    point: str,
    metric: str,
    value: Any,
    unit: str,
) -> None:
    records.append(
        {
            "figure": figure,
            "operating_point": point,
            "metric": metric,
            "value": value,
            "unit": unit,
        }
    )


def _figure2(
    rows: Sequence[Mapping[str, Any]],
    input_dir: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    for index, row in enumerate(rows):
        point = str(row["operating_point"])
        latency = float(row["mean_qa_wallclock_per_query_seconds"])
        accuracy = float(row["accuracy"])
        summary = _summary_for(input_dir, point)
        yerr = None
        if summary is not None:
            interval = summary.get("accuracy", {}).get("wilson_95")
            if interval and len(interval) == 2:
                low, high = map(float, interval)
                yerr = [[accuracy - low], [high - accuracy]]
                _append_metric(
                    records,
                    figure="2",
                    point=point,
                    metric="accuracy_wilson_95_low",
                    value=low,
                    unit="fraction",
                )
                _append_metric(
                    records,
                    figure="2",
                    point=point,
                    metric="accuracy_wilson_95_high",
                    value=high,
                    unit="fraction",
                )
        ax.errorbar(
            [latency],
            [accuracy],
            yerr=yerr,
            fmt="o",
            markersize=9,
            capsize=4,
            color=(BLUE, ORANGE, GREEN, PURPLE)[index % 4],
            label=point,
        )
        ax.annotate(
            point, (latency, accuracy), xytext=(7, 7), textcoords="offset points"
        )
        _append_metric(
            records,
            figure="2",
            point=point,
            metric="mean_qa_wallclock_per_query",
            value=latency,
            unit="seconds/query",
        )
        _append_metric(
            records,
            figure="2",
            point=point,
            metric="accuracy",
            value=accuracy,
            unit="fraction",
        )
    ax.set_xlabel("Mean serving latency (seconds/query; construction excluded)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Figure 2 analogue: serving latency vs. accuracy")
    _style_axis(ax, grid_axis="both")
    return _save(fig, output_dir, "figure2_latency_accuracy")


def _figure3(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    metrics = (
        ("construction_wallclock_seconds", "Construction total", BLUE),
        ("retrieval_per_query_seconds", "Retrieval / query", ORANGE),
        ("generation_per_query_seconds", "Generation / query", GREEN),
    )
    x = list(range(len(rows)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    for metric_index, (key, label, color) in enumerate(metrics):
        offset = (metric_index - 1) * width
        values = [float(row[key]) for row in rows]
        ax.bar(
            [position + offset for position in x],
            values,
            width,
            label=label,
            color=color,
        )
        for row, value in zip(rows, values, strict=True):
            _append_metric(
                records,
                figure="3",
                point=str(row["operating_point"]),
                metric=key,
                value=value,
                unit="seconds",
            )
    positive = [float(row[key]) for row in rows for key, _, _ in metrics if row[key]]
    if positive and max(positive) / min(positive) >= 100:
        ax.set_yscale("log")
        ax.set_ylabel("Seconds (log scale)")
    else:
        ax.set_ylabel("Seconds")
    ax.set_xticks(x, _labels(rows))
    ax.set_title("Figure 3 analogue: construction, retrieval and generation")
    ax.legend(frameon=False)
    _style_axis(ax)
    return _save(fig, output_dir, "figure3_phase_breakdown")


def _figure10(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    x = list(range(len(rows)))
    ttft = [float(row["ttft_p50_seconds"]) for row in rows]
    stream = [float(row["post_first_token_streaming_p50_seconds"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.bar(x, ttft, color=BLUE, label="Wait to first token (p50)")
    ax.bar(x, stream, bottom=ttft, color=ORANGE, label="Post-first-token stream (p50)")
    for row, first, after in zip(rows, ttft, stream, strict=True):
        point = str(row["operating_point"])
        _append_metric(
            records,
            figure="10",
            point=point,
            metric="effective_ttft_p50",
            value=first,
            unit="seconds",
        )
        _append_metric(
            records,
            figure="10",
            point=point,
            metric="post_first_token_streaming_p50",
            value=after,
            unit="seconds",
        )
    ax.set_xticks(x, _labels(rows))
    ax.set_ylabel("Seconds")
    ax.set_title("Figure 10 analogue: effective time to first token")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _style_axis(ax)
    return _save(fig, output_dir, "figure10_effective_ttft")


def _figure11(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    x = list(range(len(rows)))
    width = 0.32
    p50 = [float(row["qa_p50_seconds"]) for row in rows]
    p95 = [float(row["qa_p95_seconds"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    ax.bar([v - width / 2 for v in x], p50, width, color=BLUE, label="p50")
    bars = ax.bar([v + width / 2 for v in x], p95, width, color=PURPLE, label="p95")
    for bar, row, median, tail in zip(bars, rows, p50, p95, strict=True):
        ratio = float(row["qa_p95_over_p50"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f" {ratio:.2f}×",
            ha="center",
            va="bottom",
        )
        point = str(row["operating_point"])
        for metric, value in (
            ("qa_latency_p50", median),
            ("qa_latency_p95", tail),
            ("qa_latency_p95_over_p50", ratio),
        ):
            _append_metric(
                records,
                figure="11",
                point=point,
                metric=metric,
                value=value,
                unit="ratio" if metric.endswith("over_p50") else "seconds",
            )
    ax.set_xticks(x, _labels(rows))
    ax.set_ylabel("End-to-end QA latency (seconds)")
    ax.set_title("Figure 11 analogue: median and tail latency")
    ax.legend(frameon=False)
    _style_axis(ax)
    return _save(fig, output_dir, "figure11_tail_latency")


def _median_by_budget(
    scale_points: Iterable[Mapping[str, Any]], key: Sequence[str]
) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for point in scale_points:
        value: Any = point
        for component in key:
            value = value[component]
        grouped[int(point["requested_tokens"])].append(float(value))
    budgets = sorted(grouped)
    return budgets, [statistics.median(grouped[budget]) for budget in budgets]


def _range_by_budget(
    scale_points: Iterable[Mapping[str, Any]], key: Sequence[str]
) -> tuple[list[float], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for point in scale_points:
        value: Any = point
        for component in key:
            value = value[component]
        grouped[int(point["requested_tokens"])].append(float(value))
    lows = []
    highs = []
    for budget in sorted(grouped):
        median = statistics.median(grouped[budget])
        lows.append(median - min(grouped[budget]))
        highs.append(max(grouped[budget]) - median)
    return lows, highs


def _figure9(
    scale: Mapping[str, Any],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    points = list(scale.get("points") or [])
    if not points:
        raise ValueError("scale_results.json contains no points")
    budgets, construction = _median_by_budget(points, ("construction", "seconds"))
    construction_low, construction_high = _range_by_budget(
        points, ("construction", "seconds")
    )
    _, embed_tokens = _median_by_budget(points, ("tokens", "construction_embed_tokens"))
    _, prompt_tokens = _median_by_budget(
        points, ("tokens", "construction_prompt_tokens")
    )
    _, completion_tokens = _median_by_budget(
        points, ("tokens", "construction_completion_tokens")
    )
    _, footprint = _median_by_budget(points, ("footprint", "logical_total_bytes"))
    _, retrieval_p50 = _median_by_budget(
        points, ("retrieval", "end_to_end_seconds", "p50")
    )
    _, retrieval_p95 = _median_by_budget(
        points, ("retrieval", "end_to_end_seconds", "p95")
    )

    x = [budget / 1024 for budget in budgets]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    isolated = all(
        bool(point.get("footprint", {}).get("physical_isolated")) for point in points
    )
    isolation_label = "isolated DBs" if isolated else "shared DB; logical footprint"
    fig.suptitle(
        f"Figure 9 analogue ({isolation_label}; median across repeats)",
        fontsize=13,
    )
    axes[0, 0].errorbar(
        x,
        construction,
        yerr=(construction_low, construction_high),
        marker="o",
        capsize=3,
        color=BLUE,
    )
    axes[0, 0].set_ylabel("Construction wallclock (s)")
    axes[0, 0].set_title("(a) Construction")

    llm_tokens = [
        prompt + completion
        for prompt, completion in zip(prompt_tokens, completion_tokens, strict=True)
    ]
    axes[0, 1].stackplot(
        x,
        embed_tokens,
        llm_tokens,
        labels=("Embedding input", "LLM prompt + completion"),
        colors=(GREEN, ORANGE),
        alpha=0.85,
    )
    axes[0, 1].set_ylabel("Construction tokens")
    axes[0, 1].set_title("(b) Model-token consumption")
    axes[0, 1].legend(frameon=False, fontsize=8)

    footprint_mib = [value / (1024 * 1024) for value in footprint]
    axes[1, 0].plot(x, footprint_mib, marker="o", color=PURPLE)
    axes[1, 0].set_ylabel("Scoped logical footprint (MiB)")
    axes[1, 0].set_title("(c) Database footprint")

    axes[1, 1].plot(x, retrieval_p50, marker="o", color=BLUE, label="p50")
    axes[1, 1].plot(x, retrieval_p95, marker="s", color=ORANGE, label="p95")
    axes[1, 1].set_ylabel("End-to-end retrieval (s)")
    axes[1, 1].set_title("(d) Retrieval latency")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Input history (K tokens)")
        ax.set_xticks(x, [f"{value:g}" for value in x])
        _style_axis(ax)

    for index, budget in enumerate(budgets):
        point = f"{budget // 1024}K"
        for metric, values, unit in (
            ("construction_wallclock_median", construction, "seconds"),
            ("construction_embed_tokens_median", embed_tokens, "tokens"),
            ("construction_llm_tokens_median", llm_tokens, "tokens"),
            ("logical_footprint_median", footprint, "bytes"),
            ("retrieval_p50_median_across_repeats", retrieval_p50, "seconds"),
            ("retrieval_p95_median_across_repeats", retrieval_p95, "seconds"),
        ):
            _append_metric(
                records,
                figure="9",
                point=point,
                metric=metric,
                value=values[index],
                unit=unit,
            )
        for metric, value in (
            (
                "construction_wallclock_min",
                construction[index] - construction_low[index],
            ),
            (
                "construction_wallclock_max",
                construction[index] + construction_high[index],
            ),
        ):
            _append_metric(
                records,
                figure="9",
                point=point,
                metric=metric,
                value=value,
                unit="seconds",
            )
    return _save(fig, output_dir, "figure9_scaling")


def _scale_label(scale: Mapping[str, Any]) -> str:
    label = scale.get("operating_point")
    if label:
        return str(label)
    points = list(scale.get("points") or [])
    if points and points[0].get("operating_point"):
        return str(points[0]["operating_point"])
    raise ValueError("comparison scale result has no operating_point")


def _figure9_comparison(
    scales: Sequence[Mapping[str, Any]],
    output_dir: Path,
    records: list[dict[str, Any]],
) -> list[Path]:
    """Render Figure 9 for multiple deterministic operating points."""
    if len(scales) < 2:
        raise ValueError("Figure 9 comparison requires at least two scale results")
    colors = (BLUE, ORANGE, GREEN, PURPLE)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    all_points: list[Mapping[str, Any]] = []
    tick_values: set[float] = set()

    for scale_index, scale in enumerate(scales):
        points = list(scale.get("points") or [])
        if not points:
            raise ValueError("comparison scale_results.json contains no points")
        all_points.extend(points)
        label = _scale_label(scale)
        color = colors[scale_index % len(colors)]
        budgets, construction = _median_by_budget(points, ("construction", "seconds"))
        construction_low, construction_high = _range_by_budget(
            points, ("construction", "seconds")
        )
        _, embed_tokens = _median_by_budget(
            points, ("tokens", "construction_embed_tokens")
        )
        _, prompt_tokens = _median_by_budget(
            points, ("tokens", "construction_prompt_tokens")
        )
        _, completion_tokens = _median_by_budget(
            points, ("tokens", "construction_completion_tokens")
        )
        _, footprint = _median_by_budget(points, ("footprint", "logical_total_bytes"))
        _, retrieval_p50 = _median_by_budget(
            points, ("retrieval", "end_to_end_seconds", "p50")
        )
        _, retrieval_p95 = _median_by_budget(
            points, ("retrieval", "end_to_end_seconds", "p95")
        )
        model_tokens = [
            embed + prompt + completion
            for embed, prompt, completion in zip(
                embed_tokens, prompt_tokens, completion_tokens, strict=True
            )
        ]
        footprint_mib = [value / (1024 * 1024) for value in footprint]
        x = [budget / 1024 for budget in budgets]
        tick_values.update(x)

        axes[0, 0].errorbar(
            x,
            construction,
            yerr=(construction_low, construction_high),
            marker="o",
            capsize=3,
            color=color,
            label=label,
        )
        axes[0, 1].plot(x, model_tokens, marker="o", color=color, label=label)
        axes[1, 0].plot(x, footprint_mib, marker="o", color=color, label=label)
        axes[1, 1].plot(x, retrieval_p50, marker="o", color=color, label=f"{label} p50")
        axes[1, 1].plot(
            x,
            retrieval_p95,
            marker="s",
            linestyle="--",
            color=color,
            label=f"{label} p95",
        )

        for index, budget in enumerate(budgets):
            point_label = f"{label}:{budget // 1024}K"
            for metric, values, unit in (
                ("construction_wallclock_median", construction, "seconds"),
                ("construction_model_tokens_median", model_tokens, "tokens"),
                ("logical_footprint_median", footprint, "bytes"),
                ("retrieval_p50_median_across_repeats", retrieval_p50, "seconds"),
                ("retrieval_p95_median_across_repeats", retrieval_p95, "seconds"),
            ):
                _append_metric(
                    records,
                    figure="9",
                    point=point_label,
                    metric=metric,
                    value=values[index],
                    unit=unit,
                )

    isolated = all(
        bool(point.get("footprint", {}).get("physical_isolated"))
        for point in all_points
    )
    isolation_label = "isolated DBs" if isolated else "shared DB; logical footprint"
    fig.suptitle(
        f"Figure 9 analogue comparison ({isolation_label}; median across repeats)",
        fontsize=13,
    )
    axes[0, 0].set_ylabel("Construction wallclock (s)")
    axes[0, 0].set_title("(a) Construction")
    axes[0, 1].set_ylabel("Construction tokens")
    axes[0, 1].set_title("(b) Total model-token consumption")
    axes[1, 0].set_ylabel("Scoped logical footprint (MiB)")
    axes[1, 0].set_title("(c) Database footprint")
    axes[1, 1].set_ylabel("End-to-end retrieval (s)")
    axes[1, 1].set_title("(d) Retrieval latency")

    ticks = sorted(tick_values)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Input history (K tokens)")
        ax.set_xticks(ticks, [f"{value:g}" for value in ticks])
        ax.legend(frameon=False, fontsize=7)
        _style_axis(ax)
    return _save(fig, output_dir, "figure9_scaling_comparison")


def render_figures(
    *,
    input_dir: Path,
    output_dir: Path,
    points: Sequence[str] | None = None,
    scale_results: Path | None = None,
    scale_comparison_results: Sequence[Path] | None = None,
) -> dict[str, Any]:
    if scale_results is not None and scale_comparison_results:
        raise ValueError(
            "pass either scale_results or scale_comparison_results, not both"
        )
    sections_path = input_dir / "paper_sections.json"
    sections = _load_json(sections_path)
    selected = None if points is None else set(points)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    outputs: list[Path] = []
    outputs += _figure2(
        _selected_rows(sections, "section_4_1", selected),
        input_dir,
        output_dir,
        records,
    )
    outputs += _figure3(
        _selected_rows(sections, "section_4_2", selected), output_dir, records
    )
    rows48 = _selected_rows(sections, "section_4_8", selected)
    outputs += _figure10(rows48, output_dir, records)
    outputs += _figure11(rows48, output_dir, records)
    scale_payload = None
    if scale_results is not None:
        scale_payload = _load_json(scale_results)
        outputs += _figure9(scale_payload, output_dir, records)
    comparison_scale_payloads = None
    if scale_comparison_results:
        comparison_scale_payloads = [
            _load_json(path) for path in scale_comparison_results
        ]
        outputs += _figure9_comparison(comparison_scale_payloads, output_dir, records)

    csv_path = output_dir / "figure_data.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("figure", "operating_point", "metric", "value", "unit"),
        )
        writer.writeheader()
        writer.writerows(records)
    outputs.append(csv_path)

    source_paths = [sections_path]
    if scale_results is not None:
        source_paths.append(scale_results)
    if scale_comparison_results:
        source_paths.extend(scale_comparison_results)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in source_paths
        ],
        "selected_points": sorted(selected) if selected is not None else "all",
        "figures": sorted(
            path.name for path in outputs if path.suffix in {".png", ".pdf"}
        ),
        "data": csv_path.name,
        "comparability": (
            "TriDB/GEM same-class metrics on current hardware; these are not "
            "the paper's H100 measurements or its other memory systems."
        ),
    }
    if scale_payload is not None:
        manifest["figure9"] = {
            "repeats": scale_payload.get("repeats"),
            "aggregation": "median; construction error bars are min-max",
            "physical_isolated": all(
                bool(point.get("footprint", {}).get("physical_isolated"))
                for point in scale_payload.get("points", [])
            ),
            "footprint": "scope-attributable logical bytes",
        }
    if comparison_scale_payloads is not None:
        manifest["figure9"] = {
            "operating_points": [
                _scale_label(payload) for payload in comparison_scale_payloads
            ],
            "aggregation": "median; construction error bars are min-max",
            "physical_isolated": all(
                bool(point.get("footprint", {}).get("physical_isolated"))
                for payload in comparison_scale_payloads
                for point in payload.get("points", [])
            ),
            "footprint": "scope-attributable logical bytes",
        }
    manifest_path = output_dir / "plot_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def render_scale_figure(*, scale_results: Path, output_dir: Path) -> dict[str, Any]:
    """Render Figure 9 without requiring a completed QA/core benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    scale_payload = _load_json(scale_results)
    outputs = _figure9(scale_payload, output_dir, records)
    csv_path = output_dir / "figure9_data.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("figure", "operating_point", "metric", "value", "unit"),
        )
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": [
            {
                "path": str(scale_results.resolve()),
                "sha256": _sha256(scale_results),
            }
        ],
        "selected_points": "Figure 9 scaling only",
        "figures": sorted(path.name for path in outputs),
        "data": csv_path.name,
        "comparability": (
            "TriDB/GEM same-class metrics on current hardware; these are not "
            "the paper's H100 measurements or its other memory systems."
        ),
        "figure9": {
            "repeats": scale_payload.get("repeats"),
            "aggregation": "median; construction error bars are min-max",
            "physical_isolated": all(
                bool(point.get("footprint", {}).get("physical_isolated"))
                for point in scale_payload.get("points", [])
            ),
            "footprint": "scope-attributable logical bytes",
        },
    }
    manifest_path = output_dir / "plot_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--points", nargs="*")
    parser.add_argument("--scale-results", type=Path)
    parser.add_argument(
        "--scale-comparison-results",
        nargs="+",
        type=Path,
        help="two or more scale_results.json files to overlay in Figure 9",
    )
    parser.add_argument(
        "--scale-only",
        action="store_true",
        help="render Figure 9 without requiring paper_sections.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scale_only:
        if args.scale_results is None:
            parser.error("--scale-only requires --scale-results")
        manifest = render_scale_figure(
            scale_results=args.scale_results, output_dir=args.output_dir
        )
    else:
        if args.input_dir is None:
            parser.error("--input-dir is required unless --scale-only is used")
        manifest = render_figures(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            points=args.points,
            scale_results=args.scale_results,
            scale_comparison_results=args.scale_comparison_results,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
