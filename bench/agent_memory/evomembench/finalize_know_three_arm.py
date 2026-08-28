"""Fail-closed root receipt and Material Passport for the v0.2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def finalize(root: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if (root / "run_receipt.json").exists() or (
        root / "MATERIAL_PASSPORT.json"
    ).exists():
        raise FileExistsError("root final artifacts already exist")
    protocol = _json(protocol_path)
    protocol_version = str(protocol["schema_version"]).rsplit("_v", 1)[-1]
    plan = Path(__file__).resolve().parents[3] / protocol["plan"]["path"]
    if _sha256(plan) != protocol["plan"]["sha256"]:
        raise ValueError("v0.2 plan hash differs from the frozen protocol")
    dataset_manifest = (
        Path(__file__).resolve().parents[3] / protocol["dataset_manifest"]["path"]
    )
    if _sha256(dataset_manifest) != protocol["dataset_manifest"]["sha256"]:
        raise ValueError("dataset manifest hash differs from the frozen protocol")
    summary = _json(root / "three_arm_summary.json")
    if summary.get("status") != "complete":
        raise ValueError("three-arm summary is incomplete")
    shards = [_json(root / f"shard_{index}" / "run_receipt.json") for index in range(2)]
    polyglot = [
        _json(root / f"polyglot_{index}" / "run_receipt.json") for index in range(2)
    ]
    if any(row.get("status") != "complete" for row in shards + polyglot):
        raise ValueError("a shard or Polyglot replay is incomplete")
    if any(not row.get("all_parity_passed") for row in polyglot):
        raise ValueError("Polyglot parity is not 100%")
    judge_receipts = [
        _json(root / f"shard_{index}" / "graded" / f"{arm}.receipt.json")
        for index in range(2)
        for arm in ("memory_off", "full_gem")
    ]
    if any(
        row.get("status") != "complete"
        or row.get("denominator_preserved") is not True
        or int(row.get("graded", -1)) != int(row.get("predictions", -2))
        for row in judge_receipts
    ):
        raise ValueError("official judge denominator gate failed")
    if not (root / "gpu_layout_admission.log").is_file():
        raise FileNotFoundError("GPU layout admission evidence is missing")
    receipt = {
        "schema_version": (
            f"evomembench_know_three_arm_root_receipt_v{protocol_version}"
        ),
        "status": "complete",
        "run_id": root.name,
        "protocol_sha256": _sha256(protocol_path),
        "plan_sha256": _sha256(plan),
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "arms": protocol["arms"],
        "forbidden_arms": protocol["forbidden_arms"],
        "context_shards": 2,
        "contexts": 120,
        "episodes_per_agent_arm": 884,
        "reuse_decisions": 764,
        "polyglot_queries": sum(int(row["queries"]) for row in polyglot),
        "parity_fraction": 1.0,
        "answer_replicas": 2,
        "embedding_execution": protocol["models"]["embedding_execution"],
        "completed_at_unix": time.time(),
    }
    passport = {
        "schema_version": (
            f"evomembench_know_three_arm_material_passport_v{protocol_version}"
        ),
        "verification_status": "executed_and_summarized",
        "origin_skill": "academic-research-suite/experiment-agent",
        "run_id": root.name,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": receipt["protocol_sha256"],
        "dataset_revision": protocol["dataset_manifest"]["source_revision"],
        "quality_denominator_preserved": True,
        "polyglot_parity_100pct": True,
        "gpu_layout": "two symmetric answer-plus-embedding GPU service pairs",
        "embedding": "two pinned Qwen3-Embedding-0.6B GPU BF16 replicas",
        "hardware_claim": "off_target_x86_dual_gpu_not_gx10_signoff",
        "claim_gates": summary["claim_gates"],
    }
    return receipt, passport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    receipt, passport = finalize(args.run_root, args.protocol)
    (args.run_root / "run_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    (args.run_root / "MATERIAL_PASSPORT.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
