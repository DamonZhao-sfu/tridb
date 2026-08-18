"""Verify normalized E0 datasets and their pinned embedding artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.e0 import QWEN3_EMBEDDING_REVISION
from tools.e0.common import environment_record, sha256_file, write_json

EXPECTED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EXPECTED_DIMENSION = 1024


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _embedding_status(dataset_dir: Path, expected_rows: int) -> dict[str, Any]:
    data_path = dataset_dir / "embeddings.parquet"
    manifest_path = dataset_dir / "embeddings.manifest.json"
    manifest = _read(manifest_path)
    checks = {
        "data_exists": data_path.exists(),
        "manifest_exists": bool(manifest),
        "row_count_matches": manifest.get("rows") == expected_rows,
        "model_matches": manifest.get("model") == EXPECTED_MODEL,
        "dimension_matches": manifest.get("dimensions") == [EXPECTED_DIMENSION],
        "model_revision_matches": manifest.get("artifact_revision")
        == QWEN3_EMBEDDING_REVISION,
    }
    if data_path.exists() and manifest:
        checks["sha256_matches"] = manifest.get("output", {}).get(
            "sha256"
        ) == sha256_file(data_path)
    else:
        checks["sha256_matches"] = False
    return {"checks": checks, "ready": all(checks.values()), "manifest": manifest}


# E0 scopes. The plan-space measurement is frozen to STARK-PRIME only (decision D2,
# docs/e0_plan_space_execution_v0.1.0.md §4) -- the OpenEvolve leg belongs to P1 and must
# not gate P0. `all` keeps the original both-datasets meaning so nothing that already
# depends on it changes behaviour.
SCOPES = {
    "all": ("openevolve", "stark_prime"),
    "stark_prime": ("stark_prime",),
    "openevolve": ("openevolve",),
}


def verify(root: Path, scope: str = "all") -> dict[str, Any]:
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {sorted(SCOPES)}")
    specs = {
        "openevolve": root / "openevolve" / "normalized" / "seed_42",
        "stark_prime": root / "stark_prime" / "normalized",
    }
    datasets: dict[str, Any] = {}
    for name, directory in specs.items():
        normalization = _read(directory / "manifest.json")
        expected_rows = int(normalization.get("counts", {}).get("nodes", -1))
        normalized_ready = bool(
            normalization.get("ready_for_e0")
            if name == "openevolve"
            else normalization.get("ready_for_embedding")
        )
        embeddings = _embedding_status(directory, expected_rows)
        datasets[name] = {
            "path": str(directory),
            "normalized_ready": normalized_ready,
            "embeddings": embeddings,
            "ready": normalized_ready and embeddings["ready"],
        }
    in_scope = SCOPES[scope]
    result = {
        "schema_version": "e0-data-preparation-v0.2.0",
        "environment": environment_record(),
        "datasets": datasets,
        "scope": scope,
        "scope_datasets": list(in_scope),
        # Per-dataset readiness, so a scoped run can gate on exactly what it uses instead
        # of on a global flag that a different dataset can hold down.
        "ready_by_dataset": {name: datasets[name]["ready"] for name in datasets},
        # Scoped verdict: the one the E0 runner checks.
        "ready_for_scope": all(datasets[name]["ready"] for name in in_scope),
        # Unscoped verdict, kept for continuity: TRUE only when every dataset is ready.
        "ready_for_e0": all(row["ready"] for row in datasets.values()),
    }
    write_json(root / "preparation_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/e0"))
    parser.add_argument(
        "--scope",
        default="all",
        choices=sorted(SCOPES),
        help="which datasets must be ready for the exit status (D2: E0 uses stark_prime)",
    )
    args = parser.parse_args(argv)
    result = verify(args.root, args.scope)
    for name, status in result["datasets"].items():
        marker = "*" if name in result["scope_datasets"] else " "
        print(
            f"{marker} {name}: normalized={status['normalized_ready']} "
            f"embeddings={status['embeddings']['ready']} ready={status['ready']}"
        )
    print(f"scope={result['scope']} ready_for_scope={result['ready_for_scope']}")
    print("ready_for_e0=", result["ready_for_e0"], "(all datasets)")
    return 0 if result["ready_for_scope"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
