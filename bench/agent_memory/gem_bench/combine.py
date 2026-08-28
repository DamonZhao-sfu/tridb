"""Combine completed operating points from compatible benchmark run roots."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem_bench import report as reportmod
from bench.agent_memory.gem_bench.summarize import paper_sections
from bench.agent_memory.serving import _atomic_write_json, _sha256


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _compatibility_key(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must match for a cross-run comparison."""

    models = manifest.get("models", {})
    model_key = {
        key: models.get(key)
        for key in (
            "answer",
            "answer_base_url",
            "construction",
            "construction_colocated_with_generation",
            "embedding",
            "embedding_base_url",
            "embedding_dim",
            "judge",
            "judge_base_url",
            "thinking_enabled",
        )
    }
    # vLLM regenerates model/permission IDs and creation timestamps when an
    # endpoint restarts.  Pin the served identity and capacity, not those
    # ephemeral discovery fields.
    for endpoint in ("answer_endpoint_models", "embedding_endpoint_models"):
        model_key[endpoint] = [
            {
                "id": item.get("id"),
                "root": item.get("root"),
                "max_model_len": item.get("max_model_len"),
                "owned_by": item.get("owned_by"),
            }
            for item in models.get(endpoint, [])
        ]
    return {
        "schema_version": manifest.get("schema_version"),
        "input_sha256": manifest.get("input", {}).get("sha256"),
        "models": model_key,
        "generation": manifest.get("generation"),
        "evaluation": manifest.get("evaluation"),
        "caps": manifest.get("caps"),
        "workload": manifest.get("workload"),
        "tridb": manifest.get("tridb"),
    }


def combine_runs(
    *,
    source_dirs: Sequence[Path],
    output_dir: Path,
    points: Sequence[str],
) -> dict[str, Any]:
    """Combine exactly one completed summary per point and record provenance."""

    if not source_dirs:
        raise ValueError("at least one source directory is required")
    if len(set(points)) != len(points):
        raise ValueError("operating points must be unique")

    manifests = [(root, _load(root / "run_manifest.json")) for root in source_dirs]
    expected = _compatibility_key(manifests[0][1])
    for root, manifest in manifests[1:]:
        if _compatibility_key(manifest) != expected:
            raise ValueError(f"incompatible benchmark manifest: {root}")

    summaries: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for point in points:
        matches = [
            (root, root / point / "summary.json")
            for root, _manifest in manifests
            if (root / point / "summary.json").exists()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one completed summary for {point}; got {len(matches)}"
            )
        root, summary_path = matches[0]
        summary = _load(summary_path)
        actual = summary.get("operating_point", {}).get("key")
        if actual != point:
            raise ValueError(
                f"{summary_path} identifies {actual!r}, expected {point!r}"
            )
        destination = output_dir / point / "summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, destination)
        summaries.append(summary)
        sources.append(
            {
                "operating_point": point,
                "run_root": str(root.resolve()),
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": _sha256(summary_path),
                "run_started_at": dict(manifests)[root].get("started_at"),
            }
        )

    manifest = copy.deepcopy(manifests[0][1])
    point_metadata = {
        str(item["key"]): item
        for _root, source_manifest in manifests
        for item in source_manifest.get("operating_points", [])
    }
    manifest["operating_points"] = [point_metadata[point] for point in points]
    manifest["combined_at"] = datetime.now(UTC).isoformat()
    manifest["combined_from_multiple_serial_runs"] = len(source_dirs) > 1
    manifest["source_runs"] = sources
    manifest["energy"] = {
        "available": all(
            summary.get("energy", {}).get("total_joules") is not None
            for summary in summaries
        ),
        "method": "per-point source-run phase attribution",
        "note": (
            "Each point retains the energy measured in its source serial run; "
            "source summary hashes are recorded in source_runs."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "run_manifest.json", manifest)
    sections = paper_sections(summaries)
    reportmod.write(sections, manifest, output_dir)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--points", required=True, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = combine_runs(
        source_dirs=args.source_dirs,
        output_dir=args.output_dir,
        points=args.points,
    )
    print(json.dumps(manifest["source_runs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
