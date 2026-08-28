"""Render publication-ready Track C latency and reliability figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import read_jsonl
from .protocol import _write_json, benchmark_code_sha256

SYSTEM_ORDER = (
    "tridb_gem",
    "mem0",
    "memos",
    "cognee",
    "graphiti_zep_oss_proxy",
)
SYSTEM_LABELS = {
    "tridb_gem": "TriDB/GEM",
    "mem0": "Mem0",
    "memos": "MemOS",
    "cognee": "Cognee",
    "graphiti_zep_oss_proxy": "Graphiti (Zep OSS proxy)",
}
SYSTEM_COLORS = {
    "tridb_gem": "#0072B2",
    "mem0": "#D55E00",
    "memos": "#009E73",
    "cognee": "#CC79A7",
    "graphiti_zep_oss_proxy": "#F0E442",
}


def _save(figure: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_receipts(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(root.glob("**/run_receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") == "complete":
            yield path, receipt


def _phase_latencies(root: Path, phase: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for path, receipt in _complete_receipts(root):
        if receipt.get("phase") != phase:
            continue
        system = str(receipt["system"])
        for record in read_jsonl([path.parent / "formal.jsonl"]):
            if record.get("success") is True:
                values[system].append(float(record["service_latency_ms"]))
    return values


def plot_search_ecdf(root: Path, output: Path) -> None:
    values = _phase_latencies(root, "search")
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for system in SYSTEM_ORDER:
        ordered = np.sort(np.asarray(values.get(system, []), dtype=float))
        if ordered.size == 0:
            continue
        y = np.arange(1, ordered.size + 1) / ordered.size
        axis.step(
            ordered,
            y,
            where="post",
            label=SYSTEM_LABELS[system],
            color=SYSTEM_COLORS[system],
            linewidth=1.8,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Successful Search service latency (ms, log scale)")
    axis.set_ylabel("Empirical CDF")
    axis.set_ylim(0, 1.01)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    _save(figure, output, "search_latency_ecdf")


def plot_add_ecdf(root: Path, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), sharey=True)
    for axis, phase, title in zip(
        axes,
        ("add:native", "add:source_to_searchable"),
        ("Native Add", "Source-to-searchable Add"),
        strict=True,
    ):
        values = _phase_latencies(root, phase)
        for system in SYSTEM_ORDER:
            ordered = np.sort(np.asarray(values.get(system, []), dtype=float))
            if ordered.size == 0:
                continue
            y = np.arange(1, ordered.size + 1) / ordered.size
            axis.step(
                ordered,
                y,
                where="post",
                label=SYSTEM_LABELS[system],
                color=SYSTEM_COLORS[system],
                linewidth=1.7,
            )
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("Successful service latency (ms, log scale)")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_ylim(0, 1.01)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, frameon=False)
    _save(figure, output, "add_latency_ecdf")


def plot_search_box(root: Path, output: Path) -> None:
    values = _phase_latencies(root, "search")
    systems = [system for system in SYSTEM_ORDER if values.get(system)]
    figure, axis = plt.subplots(figsize=(7.1, 4.5))
    boxes = axis.boxplot(
        [values[system] for system in systems],
        tick_labels=[SYSTEM_LABELS[system] for system in systems],
        showfliers=True,
        whis=(5, 95),
        patch_artist=True,
    )
    for patch, system in zip(boxes["boxes"], systems, strict=True):
        patch.set_facecolor(SYSTEM_COLORS[system])
        patch.set_alpha(0.7)
    axis.set_yscale("log")
    axis.set_ylabel("Successful Search service latency (ms, log scale)")
    axis.grid(axis="y", which="both", alpha=0.25)
    _save(figure, output, "search_latency_box")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else float("nan")


def _pooled(metrics: Sequence[dict[str, Any]], phase: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["system"]): row
        for row in metrics
        if row.get("phase") == phase and row.get("build_id") == "pooled"
    }


def plot_search_p99(metrics: Sequence[dict[str, Any]], output: Path) -> None:
    rows = _pooled(metrics, "search")
    systems = [system for system in SYSTEM_ORDER if system in rows]
    y = np.asarray([_float(rows[system], "service_p99_ms") for system in systems])
    low = np.asarray(
        [_float(rows[system], "service_p99_ci_low_ms") for system in systems]
    )
    high = np.asarray(
        [_float(rows[system], "service_p99_ci_high_ms") for system in systems]
    )
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    positions = np.arange(len(systems))
    axis.bar(
        positions,
        y,
        color=[SYSTEM_COLORS[system] for system in systems],
        yerr=np.vstack((y - low, high - y)),
        capsize=4,
    )
    axis.set_xticks(positions, [SYSTEM_LABELS[system] for system in systems])
    axis.set_ylabel("Successful Search P99 service latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output, "search_p99_bar")


def plot_add_p99(metrics: Sequence[dict[str, Any]], output: Path) -> None:
    native = _pooled(metrics, "add:native")
    searchable = _pooled(metrics, "add:source_to_searchable")
    systems = [
        system for system in SYSTEM_ORDER if system in native and system in searchable
    ]
    positions = np.arange(len(systems))
    width = 0.36
    native_y = [_float(native[system], "service_p99_ms") for system in systems]
    searchable_y = [_float(searchable[system], "service_p99_ms") for system in systems]
    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.bar(
        positions - width / 2,
        native_y,
        width,
        label="Native Add",
        color="#56B4E9",
    )
    axis.bar(
        positions + width / 2,
        searchable_y,
        width,
        label="Source-to-searchable",
        color="#E69F00",
    )
    axis.set_xticks(positions, [SYSTEM_LABELS[system] for system in systems])
    axis.set_ylabel("Successful Add P99 service latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    _save(figure, output, "add_p99_bar")


def plot_success_rate(metrics: Sequence[dict[str, Any]], output: Path) -> None:
    phases = ("search", "add:native", "add:source_to_searchable")
    labels = ("Search", "Native Add", "Source-to-searchable")
    positions = np.arange(len(SYSTEM_ORDER))
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.8, 4.5))
    for phase_index, (phase, label) in enumerate(zip(phases, labels, strict=True)):
        rows = _pooled(metrics, phase)
        values = [
            100.0 * _float(rows[system], "success_rate")
            if system in rows
            else float("nan")
            for system in SYSTEM_ORDER
        ]
        axis.bar(
            positions + (phase_index - 1) * width,
            values,
            width,
            label=label,
        )
    axis.set_xticks(positions, [SYSTEM_LABELS[system] for system in SYSTEM_ORDER])
    axis.set_ylabel("Formal request success rate (%)")
    axis.set_ylim(0, 101)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    _save(figure, output, "formal_success_rate")


def _quality_scores(root: Path) -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for path in sorted(root.glob("**/quality_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        system = str(payload.get("system") or path.parent.parent.name)
        overall = payload.get("overall") or {}
        totals[system] += int(overall.get("total") or 0)
        correct[system] += int(overall.get("correct") or 0)
    return {
        system: correct[system] / total for system, total in totals.items() if total > 0
    }


def plot_latency_quality_frontier(
    metrics: Sequence[dict[str, Any]], quality_root: Path, output: Path
) -> bool:
    search = _pooled(metrics, "search")
    scores = _quality_scores(quality_root)
    systems = [
        system for system in SYSTEM_ORDER if system in search and system in scores
    ]
    if not systems:
        return False
    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    for system in systems:
        axis.scatter(
            _float(search[system], "service_p99_ms"),
            100.0 * scores[system],
            s=80,
            color=SYSTEM_COLORS[system],
            label=SYSTEM_LABELS[system],
        )
        axis.annotate(
            SYSTEM_LABELS[system],
            (
                _float(search[system], "service_p99_ms"),
                100.0 * scores[system],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )
    axis.set_xscale("log")
    axis.set_xlabel(
        "Successful Search P99 service latency (ms, log scale; lower is better)"
    )
    axis.set_ylabel("LoCoMo answer score (%)")
    axis.grid(True, which="both", alpha=0.25)
    _save(figure, output, "latency_quality_frontier")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quality-root", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    args = parser.parse_args(argv)
    current_code_hash = benchmark_code_sha256()
    if current_code_hash != args.expected_code_hash:
        raise RuntimeError("figure code does not match the frozen run hash")
    root = Path(args.root)
    output = Path(args.output_dir)
    metrics_path = Path(args.metrics)
    aggregate_receipt = json.loads(
        (metrics_path.parent / "aggregate_receipt.json").read_text(encoding="utf-8")
    )
    if not (
        aggregate_receipt.get("benchmark_code_sha256") == args.expected_code_hash
        and aggregate_receipt.get("verification_passed") is True
        and aggregate_receipt.get("quality_verification_passed") is True
    ):
        raise RuntimeError("figures require a verified aggregate receipt")
    metrics = _read_metrics(metrics_path)
    plot_search_ecdf(root, output)
    plot_add_ecdf(root, output)
    plot_search_box(root, output)
    plot_search_p99(metrics, output)
    plot_add_p99(metrics, output)
    plot_success_rate(metrics, output)
    frontier = plot_latency_quality_frontier(metrics, Path(args.quality_root), output)
    if not frontier:
        raise RuntimeError("latency-quality frontier has no verified points")
    stems = [
        "search_latency_ecdf",
        "add_latency_ecdf",
        "search_latency_box",
        "search_p99_bar",
        "add_p99_bar",
        "formal_success_rate",
        "latency_quality_frontier",
    ]
    receipt = {
        "schema_version": "table5_track_c_figure_receipt_v1",
        "output_dir": str(output.resolve()),
        "runs_root": str(root.resolve()),
        "quality_root": str(Path(args.quality_root).resolve()),
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": _sha256(metrics_path),
        "benchmark_code_sha256": current_code_hash,
        "figures": {
            f"{stem}.{suffix}": _sha256(output / f"{stem}.{suffix}")
            for stem in stems
            for suffix in ("pdf", "png")
        },
    }
    _write_json(output / "figure_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
