"""Package a multi-operating-point GEM comparison into a Git-friendly bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem_bench.export import (
    CSV_COLUMNS,
    _core_metrics,
    _load,
    _scale_metrics,
)
from bench.agent_memory.serving import _atomic_write_json, _sha256

SCHEMA_VERSION = "tridb_gem_comparison_bundle_v0.1.0"
FIGURE_STEMS = (
    "figure2_latency_accuracy",
    "figure3_phase_breakdown",
    "figure9_scaling_comparison",
    "figure10_effective_ttft",
    "figure11_tail_latency",
)


def _scale_label(scale: Mapping[str, Any]) -> str:
    label = scale.get("operating_point")
    if label:
        return str(label)
    points = list(scale.get("points") or [])
    if points and points[0].get("operating_point"):
        return str(points[0]["operating_point"])
    raise ValueError("scale result has no operating_point")


def _copy(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _headline_table(sections: Mapping[str, Any], points: Sequence[str]) -> str:
    selected = {
        str(row["operating_point"]): row
        for row in sections["section_4_1"]["rows"]
        if row["operating_point"] in points
    }
    lines = [
        "| Operating point | Taxonomy | Accuracy | Mean QA | Mean retrieval |",
        "|---|---|---:|---:|---:|",
    ]
    for point in points:
        row = selected[point]
        lines.append(
            f"| `{point}` | {row['paradigm']} | {float(row['accuracy']):.3f} | "
            f"{float(row['mean_qa_wallclock_per_query_seconds']):.3f} s | "
            f"{float(row['mean_retrieval_per_query_seconds']) * 1000:.2f} ms |"
        )
    return "\n".join(lines)


def _readme(
    *,
    sections: Mapping[str, Any],
    points: Sequence[str],
    scale_labels: Sequence[str],
) -> str:
    point_args = " ".join(points)
    scale_args = " ".join(
        f"results/agent_memory_comparison/raw/{label}_scale_results.json"
        for label in scale_labels
    )
    return f"""# TriDB/GEM multi-system-proxy comparison

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Verification Status: `ANALYZED`
- Bundle schema: `{SCHEMA_VERSION}`
- Generated: `{datetime.now(UTC).isoformat()}`

This bundle compares TriDB/GEM operating points under one serial benchmark run.
`II_embedrag` is a TriDB implementation. `IIIa_graphrag_like` and
`IIIb_mem0_like` are taxonomy/cost-shape proxies, **not** the official GraphRAG
or Mem0 systems. `gem_conformant` is the GEM contribution point.

## Headline results

{_headline_table(sections, points)}

Figure 2/3/10/11 contain all four operating points. Figure 9 contains only
`{"` and `".join(scale_labels)}` because the current scaling runner supports
deterministic construction points only. Shared-database Figure 9 rows have
`physical_isolated=false`; their footprint is logical, not isolated physical
storage.

## Contents

- `metrics.csv`: tidy core and scaling metrics.
- `figures/`: PNG and PDF versions of Figure 2/3/9/10/11 analogues.
- `raw/paper_sections.json`: canonical core aggregates.
- `raw/<point>/summary.json`: per-point detailed aggregates.
- `raw/*_scale_results.json`: all Figure 9 repetitions and probes.
- `MANIFEST.json`: SHA-256 and byte size for every artifact.

## Re-render from this tracked bundle

```bash
.venv/bin/python -m bench.agent_memory.gem_bench.figures \\
  --input-dir results/agent_memory_comparison/raw \\
  --output-dir results/agent_memory_comparison/figures \\
  --points {point_args} \\
  --scale-comparison-results {scale_args}
```

## Full experiment command

See `raw/run_manifest.json` for pinned models and settings. From the repository
root with PostgreSQL and both vLLM endpoints running:

```bash
.venv/bin/python -m bench.agent_memory.gem_bench \\
  --input data/longmemeval/memoryagentbench_longmemeval_sstar.json \\
  --output-dir bench/out/gem_four_way_full \\
  --dsn postgresql://USER@127.0.0.1:55432/gem_bench \\
  --points {point_args} --top-k 10
```

Absolute latency and energy are measurements of the current machine, not the
paper's H100 environment. The local Qwen3-32B judge is a protocol variant, so
accuracy must not be presented as an exact reproduction of the paper's bars.
"""


def export_comparison_bundle(
    *,
    core_dir: Path,
    scale_results: Sequence[Path],
    figures_dir: Path,
    output_dir: Path,
    points: Sequence[str],
) -> dict[str, Any]:
    sections_path = core_dir / "paper_sections.json"
    sections = _load(sections_path)
    scales = [_load(path) for path in scale_results]
    scale_labels = [_scale_label(scale) for scale in scales]

    for stem in FIGURE_STEMS:
        for suffix in ("png", "pdf"):
            _copy(
                figures_dir / f"{stem}.{suffix}",
                output_dir / "figures" / f"{stem}.{suffix}",
            )
    _copy(sections_path, output_dir / "raw" / "paper_sections.json")
    _copy(core_dir / "run_manifest.json", output_dir / "raw" / "run_manifest.json")
    if (figures_dir / "plot_manifest.json").exists():
        _copy(
            figures_dir / "plot_manifest.json",
            output_dir / "raw" / "plot_manifest.json",
        )

    metrics: list[dict[str, Any]] = []
    for point in points:
        summary_path = core_dir / point / "summary.json"
        summary = _load(summary_path)
        metrics.extend(_core_metrics(sections, summary, point))
        _copy(summary_path, output_dir / "raw" / point / "summary.json")
    for source, scale, label in zip(scale_results, scales, scale_labels, strict=True):
        metrics.extend(_scale_metrics(scale, label))
        _copy(source, output_dir / "raw" / f"{label}_scale_results.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(metrics)
    (output_dir / "README.md").write_text(
        _readme(sections=sections, points=points, scale_labels=scale_labels),
        encoding="utf-8",
    )

    artifacts = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "operating_points": list(points),
        "figure9_operating_points": scale_labels,
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
        "comparability": {
            "paper_hardware_match": False,
            "official_external_systems": False,
            "serial_execution": True,
            "figure9_physical_isolated": all(
                bool(point.get("footprint", {}).get("physical_isolated"))
                for scale in scales
                for point in scale.get("points", [])
            ),
        },
    }
    _atomic_write_json(output_dir / "MANIFEST.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--core-dir", required=True, type=Path)
    parser.add_argument("--scale-results", required=True, nargs="+", type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--points", required=True, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_comparison_bundle(
        core_dir=args.core_dir,
        scale_results=args.scale_results,
        figures_dir=args.figures_dir,
        output_dir=args.output_dir,
        points=args.points,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
