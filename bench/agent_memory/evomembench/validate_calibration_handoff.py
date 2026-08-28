"""Validate a completed Know calibration before a downstream queue consumes it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _task_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str((row.get("metadata") or {}).get("task_id", row.get("idx"))) for row in rows
    ]


def validate(root: Path) -> dict[str, Any]:
    receipt_path = root / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete":
        raise ValueError("calibration generation receipt is not complete")
    arms = tuple(str(value) for value in receipt.get("arms", ()))
    predictions = int(receipt.get("predictions", 0))
    if not arms or predictions <= 0 or predictions % len(arms):
        raise ValueError(
            "calibration receipt has an invalid arm/prediction denominator"
        )
    expected = predictions // len(arms)
    arm_reports: dict[str, Any] = {}
    for arm in arms:
        prediction_rows = _rows(root / "predictions" / f"{arm}.jsonl")
        retrieval_rows = _rows(root / "receipts" / f"{arm}.jsonl")
        graded_rows = _rows(root / "graded" / f"{arm}.jsonl")
        prediction_ids = _task_ids(prediction_rows)
        graded_ids = _task_ids(graded_rows)
        grade_receipt_path = root / "graded" / f"{arm}.receipt.json"
        grade_receipt = json.loads(grade_receipt_path.read_text())
        checks = {
            "prediction_rows": len(prediction_rows) == expected,
            "prediction_ids_unique": len(set(prediction_ids)) == expected,
            "retrieval_receipt_rows": len(retrieval_rows) == expected,
            "graded_rows": len(graded_rows) == expected,
            "graded_ids_unique": len(set(graded_ids)) == expected,
            "graded_ids_exact": set(graded_ids) == set(prediction_ids),
            "grade_receipt_complete": grade_receipt.get("status") == "complete",
            "grade_receipt_predictions": grade_receipt.get("predictions") == expected,
            "grade_receipt_graded": grade_receipt.get("graded") == expected,
            "denominator_preserved": grade_receipt.get("denominator_preserved") is True,
        }
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise ValueError(f"{arm} calibration handoff failed: {failed}")
        arm_reports[arm] = {
            "expected": expected,
            "prediction_task_ids_sha256": hashlib.sha256(
                json.dumps(sorted(prediction_ids), separators=(",", ":")).encode()
            ).hexdigest(),
            "checks": checks,
        }
    return {
        "schema_version": "evomembench_calibration_handoff_v0.1.0",
        "status": "complete",
        "passed": True,
        "run_id": receipt.get("run_id"),
        "expected_per_arm": expected,
        "arms": arm_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.run_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing existing handoff receipt: {args.output}")
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
