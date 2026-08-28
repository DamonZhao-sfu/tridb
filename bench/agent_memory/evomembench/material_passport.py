"""Generate an auditable Material Passport for an EvoMemBench run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_manifest() -> tuple[dict[str, str], str]:
    repo = Path(__file__).parents[3]
    candidates = [
        *sorted((repo / "bench/agent_memory/evomembench").glob("*")),
        *sorted((repo / "bench/agent_memory/gem").glob("*.py")),
        repo / "bench/agent_memory/gem/schema.sql",
        repo / "src/tjs_pg/tjs_pg.c",
        repo / "src/tjs_pg/tjs_pg--0.2.0.sql",
        repo / "src/graph_store/graph_am.c",
        repo / "tests/test_evomembench_cross_episode.py",
        repo / "tests/test_gem_unit.py",
        repo / "scripts/expc_serve_qwen38.sh",
        repo / "scripts/expc_serve_embed_gpu1.sh",
        *sorted((repo / "scripts").glob("evomembench*.sh")),
    ]
    manifest = {
        str(path.relative_to(repo)): _sha256(path)
        for path in candidates
        if path.is_file()
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(canonical).hexdigest()


def build(root: Path) -> dict[str, Any]:
    receipt_path = root / "run_receipt.json"
    summary_path = root / "summary.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete" or not summary_path.exists():
        raise ValueError("Material Passport requires a complete run and summary")
    summary = json.loads(summary_path.read_text())
    summary_schema = str(summary.get("schema_version", ""))
    systems_only = summary_schema.startswith("evomembench_gem_systems_")
    outcome_valid = None if systems_only else bool(summary.get("outcome_valid", True))
    artefacts = {
        str(path.relative_to(root)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name
        not in {"MATERIAL_PASSPORT.json", "MATERIAL_PASSPORT.md", "SHA256SUMS"}
    }
    implementation_files, implementation_manifest_sha256 = _implementation_manifest()
    return {
        "schema_version": "evomembench_material_passport_v0.1.0",
        "verification_status": (
            "executed_systems_only"
            if systems_only
            else (
                "executed_and_summarized"
                if outcome_valid
                else "executed_systems_only_outcome_invalid"
            )
        ),
        "outcome_valid": outcome_valid,
        "run_id": receipt["run_id"],
        "source_revision": receipt.get("source_revision"),
        "manifest_sha256": receipt.get("manifest_sha256"),
        "arms": receipt.get("arms"),
        "sample_counts": {
            key: receipt.get(key)
            for key in (
                "contexts",
                "predictions",
                "cross_episode_decisions",
                "episodes_per_environment",
                "sample_errors",
            )
            if key in receipt
        },
        "summary_schema": summary_schema,
        "implementation_manifest_sha256": implementation_manifest_sha256,
        "implementation_files": implementation_files,
        "hardware_evidence": (
            (root / "hardware.csv").read_text().splitlines()
            if (root / "hardware.csv").exists()
            else None
        ),
        "known_limitations": [
            "EvoMemBench has no independent source-experience relevance qrels",
            "x86 stock-PG results are not GX10 ARM64/CUDA/128GB sign-off",
            "the upstream pinned revision has no observed top-level license",
            *(
                [
                    "agent-outcome fields are invalid under the run's causal-comparison gate"
                ]
                if outcome_valid is False
                else []
            ),
            *(
                ["this run measures systems behavior and contains no agent outcome"]
                if systems_only
                else []
            ),
        ],
        "artefacts": artefacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    passport = build(args.run_dir)
    (args.run_dir / "MATERIAL_PASSPORT.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=2) + "\n"
    )
    (args.run_dir / "MATERIAL_PASSPORT.md").write_text(
        "\n".join(
            [
                "# Material Passport",
                "",
                f"- Verification status: `{passport['verification_status']}`",
                f"- Run: `{passport['run_id']}`",
                f"- Source revision: `{passport['source_revision']}`",
                f"- Manifest SHA-256: `{passport['manifest_sha256']}`",
                f"- Summary schema: `{passport['summary_schema']}`",
                f"- Content-addressed artefacts: {len(passport['artefacts'])}",
                "",
                "## Known limitations",
                "",
                *[f"- {item}" for item in passport["known_limitations"]],
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
