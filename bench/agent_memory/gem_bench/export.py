"""Package GEM paper metrics and figures into a Git-friendly results bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.agent_memory.serving import _atomic_write_json, _sha256

SCHEMA_VERSION = "tridb_gem_results_bundle_v0.1.0"
FIGURE_STEMS = (
    "figure2_latency_accuracy",
    "figure3_phase_breakdown",
    "figure9_scaling",
    "figure10_effective_ttft",
    "figure11_tail_latency",
)
CSV_COLUMNS = (
    "figure",
    "section",
    "operating_point",
    "requested_input_tokens",
    "actual_input_tokens",
    "repeats",
    "aggregation",
    "metric",
    "value",
    "unit",
    "physical_isolated",
    "notes",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _metric(
    *,
    figure: int,
    section: str,
    point: str,
    metric: str,
    value: Any,
    unit: str,
    requested: int | str = "",
    actual: int | str = "",
    repeats: int | str = "",
    aggregation: str = "single run",
    isolated: bool | str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "figure": figure,
        "section": section,
        "operating_point": point,
        "requested_input_tokens": requested,
        "actual_input_tokens": actual,
        "repeats": repeats,
        "aggregation": aggregation,
        "metric": metric,
        "value": value,
        "unit": unit,
        "physical_isolated": isolated,
        "notes": notes,
    }


def _core_metrics(
    sections: Mapping[str, Any], summary: Mapping[str, Any], point: str
) -> list[dict[str, Any]]:
    by_section: dict[str, Mapping[str, Any]] = {}
    for section in ("section_4_1", "section_4_2", "section_4_8"):
        selected = [
            row for row in sections[section]["rows"] if row["operating_point"] == point
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one {section} row for {point}; got {len(selected)}"
            )
        by_section[section] = selected[0]

    rows: list[dict[str, Any]] = []
    row41 = by_section["section_4_1"]
    for metric, unit in (
        ("accuracy", "fraction"),
        ("mean_qa_wallclock_per_query_seconds", "seconds/query"),
        ("mean_retrieval_per_query_seconds", "seconds/query"),
        ("mean_generation_per_query_seconds", "seconds/query"),
        ("queries", "queries"),
    ):
        rows.append(
            _metric(
                figure=2,
                section="4.1",
                point=point,
                metric=metric,
                value=row41[metric],
                unit=unit,
            )
        )
    wilson = summary.get("accuracy", {}).get("wilson_95") or []
    if len(wilson) == 2:
        for name, value in zip(
            ("accuracy_wilson_95_low", "accuracy_wilson_95_high"),
            wilson,
            strict=True,
        ):
            rows.append(
                _metric(
                    figure=2,
                    section="4.1",
                    point=point,
                    metric=name,
                    value=value,
                    unit="fraction",
                )
            )

    row42 = by_section["section_4_2"]
    for metric, unit in (
        ("construction_wallclock_seconds", "seconds"),
        ("retrieval_per_query_seconds", "seconds/query"),
        ("generation_per_query_seconds", "seconds/query"),
        ("lifecycle_wallclock_seconds", "seconds"),
        ("construction_calls", "calls"),
        ("qa_calls", "calls"),
        ("construction_tokens", "tokens"),
        ("qa_tokens", "tokens"),
        ("total_kilojoules", "kilojoules"),
        ("joules_per_correct", "joules/correct"),
    ):
        rows.append(
            _metric(
                figure=3,
                section="4.2",
                point=point,
                metric=metric,
                value=row42.get(metric),
                unit=unit,
            )
        )

    row48 = by_section["section_4_8"]
    figure_for = {
        "ttft_p50_seconds": 10,
        "post_first_token_streaming_p50_seconds": 10,
        "total_p50_seconds": 10,
        "qa_p50_seconds": 11,
        "qa_p95_seconds": 11,
        "qa_p95_over_p50": 11,
        "ttft_p95_over_p50": 11,
    }
    for metric, figure in figure_for.items():
        rows.append(
            _metric(
                figure=figure,
                section="4.8",
                point=point,
                metric=metric,
                value=row48[metric],
                unit="ratio" if metric.endswith("over_p50") else "seconds",
            )
        )
    return rows


def _nested(point: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = point
    for key in path:
        value = value[key]
    return float(value)


def _scale_metrics(scale: Mapping[str, Any], point: str) -> list[dict[str, Any]]:
    raw_points = list(scale.get("points") or [])
    if not raw_points:
        raise ValueError("scale_results.json has no points")
    repeats = int(scale.get("repeats") or 1)
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for raw in raw_points:
        grouped.setdefault(int(raw["requested_tokens"]), []).append(raw)
    rows: list[dict[str, Any]] = []
    paths = (
        ("construction_wallclock_seconds", ("construction", "seconds"), "seconds"),
        (
            "construction_embed_tokens",
            ("tokens", "construction_embed_tokens"),
            "tokens",
        ),
        (
            "construction_llm_prompt_tokens",
            ("tokens", "construction_prompt_tokens"),
            "tokens",
        ),
        (
            "construction_llm_completion_tokens",
            ("tokens", "construction_completion_tokens"),
            "tokens",
        ),
        ("logical_total_bytes", ("footprint", "logical_total_bytes"), "bytes"),
        (
            "retrieval_end_to_end_p50_seconds",
            ("retrieval", "end_to_end_seconds", "p50"),
            "seconds",
        ),
        (
            "retrieval_end_to_end_p95_seconds",
            ("retrieval", "end_to_end_seconds", "p95"),
            "seconds",
        ),
    )
    for budget in sorted(grouped):
        points = grouped[budget]
        actual = int(points[0]["actual_input_tokens"])
        isolated = all(
            bool(item.get("footprint", {}).get("physical_isolated")) for item in points
        )
        note = (
            "isolated physical database"
            if isolated
            else "shared DB: logical footprint; retrieval is preliminary"
        )
        for metric, path, unit in paths:
            values = [_nested(item, path) for item in points]
            rows.append(
                _metric(
                    figure=9,
                    section="4.7",
                    point=point,
                    requested=budget,
                    actual=actual,
                    repeats=repeats,
                    aggregation="median across repeats",
                    metric=metric,
                    value=statistics.median(values),
                    unit=unit,
                    isolated=isolated,
                    notes=note,
                )
            )
            if metric == "construction_wallclock_seconds":
                for suffix, value in (("min", min(values)), ("max", max(values))):
                    rows.append(
                        _metric(
                            figure=9,
                            section="4.7",
                            point=point,
                            requested=budget,
                            actual=actual,
                            repeats=repeats,
                            aggregation=suffix,
                            metric=metric,
                            value=value,
                            unit=unit,
                            isolated=isolated,
                            notes=note,
                        )
                    )
    return rows


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _results_table(scale: Mapping[str, Any]) -> str:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for point in scale["points"]:
        grouped.setdefault(int(point["requested_tokens"]), []).append(point)
    lines = [
        "| Budget | Actual tokens | Construct median | Logical footprint | Retrieval p50 / p95 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in sorted(grouped):
        points = grouped[budget]
        construction = statistics.median(
            _nested(point, ("construction", "seconds")) for point in points
        )
        footprint = statistics.median(
            _nested(point, ("footprint", "logical_total_bytes")) for point in points
        )
        p50 = statistics.median(
            _nested(point, ("retrieval", "end_to_end_seconds", "p50"))
            for point in points
        )
        p95 = statistics.median(
            _nested(point, ("retrieval", "end_to_end_seconds", "p95"))
            for point in points
        )
        lines.append(
            f"| {budget // 1024}K | {int(points[0]['actual_input_tokens']):,} | "
            f"{construction:.3f} s | {footprint / 2**20:.3f} MiB | "
            f"{p50 * 1000:.2f} / {p95 * 1000:.2f} ms |"
        )
    return "\n".join(lines)


def _readme(point: str, sections: Mapping[str, Any], scale: Mapping[str, Any]) -> str:
    row41 = next(
        row
        for row in sections["section_4_1"]["rows"]
        if row["operating_point"] == point
    )
    return f"""# TriDB/GEM agent-memory characterization results

## Material Passport

- Paper: arXiv:2606.06448, *Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads*
- System: `{point}` only
- Bundle schema: `{SCHEMA_VERSION}`
- Verification status: `ANALYZED`
- Generated: `{datetime.now(UTC).isoformat()}`

This directory is the Git-friendly snapshot of the local TriDB/GEM analogues of
Figures 2, 3, 9, 10 and 11. It contains plotted aggregates, the raw JSON needed
to re-render the figures, and PNG/PDF outputs. It does not contain the large
LongMemEval source corpus or model weights.

## Contents

- `metrics.csv`: one tidy CSV containing every reported core/scaling metric.
- `figures/`: PNG (300 DPI) and vector PDF versions of all five figures.
- `raw/paper_sections.json`: Figure 2/3/10/11 aggregate source.
- `raw/gem_conformant/summary.json`: accuracy interval and detailed core metrics.
- `raw/scale_results.json`: all 15 Figure 9 observations and retrieval probes.
- `raw/run_manifest.json`, `raw/scale_input_manifest.json`, and
  `raw/plot_manifest.json`: provenance metadata.
- `MANIFEST.json`: SHA-256 and byte size for every committed artifact.

## Current headline results

- Accuracy: `{float(row41["accuracy"]):.3f}`.
- Mean QA latency: `{float(row41["mean_qa_wallclock_per_query_seconds"]):.3f}` seconds/query.
- Figure 9 uses `{int(scale.get("repeats") or 1)}` repeats and median aggregation.

{_results_table(scale)}

Figure 9 in this snapshot used a shared, non-empty benchmark database.
`physical_isolated=false`, so the plotted footprint is scope-attributable
logical bytes. Physical relation deltas remain in `raw/scale_results.json` but
must not be presented as per-user physical footprint. Retrieval latency is a
preliminary shared-index measurement.

## Re-render this tracked snapshot

From the repository root, with `requirements-agent-memory.txt` installed:

```bash
.venv/bin/python -m bench.agent_memory.gem_bench.figures \\
  --input-dir results/agent_memory_characterization/raw \\
  --output-dir results/agent_memory_characterization/figures \\
  --points gem_conformant \\
  --scale-results results/agent_memory_characterization/raw/scale_results.json
```

## Re-run the experiments

Prepare the database, LongMemEval dataset, answer endpoint and embedding
endpoint as documented in `README.md` section 8, then run:

```bash
# Figure 2/3/10/11: one GEM system, 5 histories x 60 questions
make gem-paper-core \\
  GEM_PAPER_OUT=bench/out/gem_paper

# Figure 9: 64K..1M, three repeats. Use one empty DB per point for physical size.
make gem-paper-scale \\
  GEM_FIGURE_INPUT=bench/out/gem_paper \\
  GEM_SCALE_REPEATS=3 \\
  GEM_SCALE_DSN_TEMPLATE='postgresql://user@127.0.0.1:55432/gem_scale_{{budget_k}}k_r{{repeat}}'

# Refresh this commit-ready snapshot
make gem-paper-export \\
  GEM_EXPORT_CORE=bench/out/gem_paper \\
  GEM_EXPORT_SCALE=bench/out/gem_scaling/scale_results.json
```

For a quick Figure 9 connectivity check:

```bash
make gem-paper-smoke
```

## Git upload

Inspect the bundle, then stage it together with the reproduction code:

```bash
git add results/agent_memory_characterization \\
  bench/agent_memory/gem_bench/figures.py \\
  bench/agent_memory/gem_bench/scaling.py \\
  bench/agent_memory/gem_bench/export.py \\
  bench/agent_memory/gem_bench/README.md \\
  bench/agent_memory/gem_bench/__init__.py \\
  tests/test_gem_paper_figures.py \\
  Makefile README.md requirements-agent-memory.txt
```

Do not claim these absolute timings reproduce the paper's H100 numbers. They
are same-class metrics collected on the current machine with a local judge.
"""


def export_bundle(
    *,
    core_dir: Path,
    scale_results: Path,
    scale_input_manifest: Path | None,
    figures_dir: Path,
    output_dir: Path,
    point: str,
) -> dict[str, Any]:
    sections_path = core_dir / "paper_sections.json"
    summary_path = core_dir / point / "summary.json"
    sections = _load(sections_path)
    summary = _load(summary_path)
    scale = _load(scale_results)

    figures_output = output_dir / "figures"
    raw_output = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in FIGURE_STEMS:
        for suffix in ("png", "pdf"):
            _copy_required(
                figures_dir / f"{stem}.{suffix}",
                figures_output / f"{stem}.{suffix}",
            )

    raw_files = (
        (sections_path, raw_output / "paper_sections.json"),
        (summary_path, raw_output / point / "summary.json"),
        (scale_results, raw_output / "scale_results.json"),
        (core_dir / "run_manifest.json", raw_output / "run_manifest.json"),
        (figures_dir / "plot_manifest.json", raw_output / "plot_manifest.json"),
    )
    for source, destination in raw_files:
        if source.exists():
            _copy_required(source, destination)
    if scale_input_manifest is not None:
        _copy_required(
            scale_input_manifest,
            raw_output / "scale_input_manifest.json",
        )

    metrics = [
        *_core_metrics(sections, summary, point),
        *_scale_metrics(scale, point),
    ]
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(metrics)

    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme(point, sections, scale), encoding="utf-8")

    artifacts = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "operating_point": point,
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifacts)
        ],
        "source_boundaries": {
            "large_dataset_included": False,
            "model_weights_included": False,
            "paper_hardware_match": False,
            "figure9_physical_isolated": all(
                bool(item.get("footprint", {}).get("physical_isolated"))
                for item in scale["points"]
            ),
        },
    }
    manifest_path = output_dir / "MANIFEST.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--core-dir", required=True, type=Path)
    parser.add_argument("--scale-results", required=True, type=Path)
    parser.add_argument("--scale-input-manifest", type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--point", default="gem_conformant")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_bundle(
        core_dir=args.core_dir,
        scale_results=args.scale_results,
        scale_input_manifest=args.scale_input_manifest,
        figures_dir=args.figures_dir,
        output_dir=args.output_dir,
        point=args.point,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
