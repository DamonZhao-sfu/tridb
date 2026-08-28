"""Produce the canonical top-level artifacts locked by the v0.1.0 plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from bench.agent_memory.evomembench.system_protocol import verify_trace


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _trace_paths(root: Path) -> list[Path]:
    paths = [
        *sorted((root / "know" / "traces").glob("*.jsonl")),
        *sorted((root / "know_multi_system" / "traces").glob("*.jsonl")),
        *sorted((root / "tool" / "system_traces").glob("*.jsonl")),
        *sorted((root / "tool_multi_system" / "traces").glob("*.jsonl")),
        *sorted((root / "scale" / "traces").glob("*.jsonl")),
    ]
    if not paths:
        raise ValueError("no system traces found")
    return paths


def _combined_traces(root: Path) -> Iterator[dict[str, Any]]:
    for path in _trace_paths(root):
        for row in _jsonl(path):
            if not verify_trace(row):
                raise ValueError(f"trace digest failed: {path}")
            yield row


def _combined_updates(root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted((root / "know" / "updates").glob("*.jsonl")):
        arm = path.stem
        for row in _jsonl(path):
            yield {
                "schema_version": "evomembench_canonical_update_v0.1.0",
                "track": "CrossEp-Know",
                "arm": arm,
                "source_artifact": str(path.relative_to(root)),
                "record": row,
            }
    tool_update_paths = sorted((root / "tool" / "updates").glob("*.jsonl"))
    if not tool_update_paths:
        raise ValueError("no CrossEp-Tool construction/update traces found")
    for path in tool_update_paths:
        arm = path.stem
        for row in _jsonl(path):
            yield {
                "schema_version": "evomembench_canonical_update_v0.1.0",
                "track": "CrossEp-Tool",
                "arm": arm,
                "phase": "source_construction",
                "source_artifact": str(path.relative_to(root)),
                "record": row,
            }
    scale_path = root / "scale" / "per_update.jsonl"
    for row in _jsonl(scale_path):
        yield {
            "schema_version": "evomembench_canonical_update_v0.1.0",
            "track": "Systems-Scale",
            "arm": "full_gem_construction",
            "source_artifact": str(scale_path.relative_to(root)),
            "record": row,
        }


def _parity(root: Path) -> dict[str, Any]:
    native = {}
    for track, folder in (
        ("CrossEp-Know", "know_multi_system"),
        ("CrossEp-Tool", "tool_multi_system"),
    ):
        receipt = json.loads((root / folder / "run_receipt.json").read_text())
        native[track] = {
            "queries": receipt["queries"],
            "passed": receipt["parity_passed"],
            "fraction": receipt["parity_fraction"],
            "all_passed": receipt["all_parity_passed"],
        }
    scale = []
    for path in sorted((root / "scale" / "parity").glob("scale_*.jsonl")):
        rows = list(_jsonl(path))
        history_size = int(path.stem.split("_")[-1])
        scale.append(
            {
                "history_size": history_size,
                "queries": len(rows),
                "set_parity_fraction": sum(row["set_parity"] for row in rows)
                / len(rows),
                "order_parity_fraction": sum(row["order_parity"] for row in rows)
                / len(rows),
                "injection_parity_fraction": sum(
                    row["injection_parity"] for row in rows
                )
                / len(rows),
                "all_passed": all(row["passed"] for row in rows),
                "source_artifact": str(path.relative_to(root)),
            }
        )
    return {
        "schema_version": "evomembench_canonical_parity_v0.1.0",
        "native": native,
        "systems_scale": sorted(scale, key=lambda row: row["history_size"]),
        "all_passed": all(row["all_passed"] for row in native.values())
        and len(scale) == 4
        and all(row["all_passed"] for row in scale),
    }


def _polyline_svg(
    *, title: str, x_labels: list[str], series: dict[str, list[float]], output: Path
) -> None:
    width, height = 900, 520
    left, top, plot_width, plot_height = 90, 70, 760, 360
    values = [value for rows in series.values() for value in rows]
    maximum = max(values) if values else 1.0
    colors = ("#2463eb", "#dc2626", "#059669")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="black"/>',
    ]
    denominator = max(1, len(x_labels) - 1)
    for index, label in enumerate(x_labels):
        x = left + plot_width * index / denominator
        lines.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 28}" text-anchor="middle" font-size="12">{label}</text>'
        )
    for series_index, (name, rows) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = []
        for index, value in enumerate(rows):
            x = left + plot_width * index / denominator
            y = top + plot_height * (1 - value / maximum)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{left + plot_width - 160}" y="{55 + series_index * 20}" fill="{color}" font-size="13">{name}</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n")


def _figures(root: Path, summary: dict[str, Any]) -> None:
    figures = root / "figures"
    figures.mkdir()
    scale = summary["systems_scale"]
    points = scale["points"]
    _polyline_svg(
        title="Systems-scale retrieval latency (P50 ms)",
        x_labels=[str(point["history_size"]) for point in points],
        series={
            "Full GEM": [point["latency_ms"]["full_gem"]["p50"] for point in points],
            "Multi-system": [
                point["latency_ms"]["multi_system"]["p50"] for point in points
            ],
        },
        output=figures / "scale_latency_p50.svg",
    )
    (figures / "README.md").write_text(
        "# Figures\n\n"
        "`scale_latency_p50.svg` is generated only from verified per-query traces. "
        "It is an off-target diagnostic unless the Material Passport passes the "
        "GX10 headline gate.\n"
    )


def finalize(root: Path) -> dict[str, Any]:
    required_absent = [
        root / name
        for name in (
            "per_query.jsonl",
            "per_update.jsonl",
            "parity.json",
            "native_summary.json",
            "figures",
        )
        if (root / name).exists()
    ]
    if required_absent:
        raise FileExistsError(
            f"refusing existing canonical artifacts: {required_absent}"
        )
    system_summary = json.loads((root / "system_summary.json").read_text())
    per_query_count = _write_jsonl(root / "per_query.jsonl", _combined_traces(root))
    per_update_count = _write_jsonl(root / "per_update.jsonl", _combined_updates(root))
    parity = _parity(root)
    (root / "parity.json").write_text(
        json.dumps(parity, ensure_ascii=False, indent=2) + "\n"
    )
    native = {
        "schema_version": "evomembench_native_summary_v0.1.0",
        "latency_semantics": system_summary["latency_semantics"],
        "tracks": system_summary["tracks"],
        "quality": system_summary["quality"],
        "efficiency_normalized": system_summary["efficiency_normalized"],
        "claim_gates": system_summary["claim_gates"],
        "unavailable_retrieval_quality_metrics": system_summary[
            "unavailable_retrieval_quality_metrics"
        ],
    }
    (root / "native_summary.json").write_text(
        json.dumps(native, ensure_ascii=False, indent=2) + "\n"
    )
    _figures(root, system_summary)
    return {
        "schema_version": "evomembench_canonical_artifact_receipt_v0.1.0",
        "status": "complete",
        "per_query_rows": per_query_count,
        "per_update_rows": per_update_count,
        "parity_all_passed": parity["all_passed"],
        "canonical_artifacts": [
            "hardware.json",
            "per_query.jsonl",
            "per_update.jsonl",
            "parity.json",
            "native_summary.json",
            "scale_summary.json",
            "figures/",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.run_root / "canonical_artifact_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing existing artifact receipt: {receipt_path}")
    receipt = finalize(args.run_root)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
