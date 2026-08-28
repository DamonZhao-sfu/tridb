"""Execute a fresh-state QPS=1/5/10 Track C sweep from an explicit manifest.

The point commands are argv arrays (never shell strings).  Provisioning stays
in the point wrapper because Search snapshots and Add databases have different
fresh-state rules for each native system.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SWEEP_QPS = (1.0, 5.0, 10.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _manifest_qps(payload: dict[str, Any]) -> tuple[float, ...]:
    requested = tuple(float(value) for value in payload.get("qps", SWEEP_QPS))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("manifest qps must be a non-empty unique list")
    if any(value not in SWEEP_QPS for value in requested):
        raise ValueError(f"manifest qps must be a subset of {SWEEP_QPS}")
    return requested


def _manifest_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = list(payload.get("points") or [])
    if explicit:
        return explicit
    matrix = payload.get("matrix") or {}
    systems = list(matrix.get("systems") or [])
    workloads = list(matrix.get("workloads") or [])
    runner = str(matrix.get("point_runner") or "")
    output_root = str(matrix.get("output_root") or "")
    if not systems or not workloads or not runner or not output_root:
        return []
    return [
        {
            "system": system,
            "workload": workload,
            "qps": qps,
            "run_dir": str(
                Path(output_root) / "runs" / system / workload / f"qps_{int(qps)}"
            ),
            "command": [
                runner,
                system,
                workload,
                "--qps",
                str(int(qps)),
                "--profile-stages",
            ],
        }
        for qps in _manifest_qps(payload)
        for workload in workloads
        for system in systems
    ]


def validate_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    expected_qps = _manifest_qps(payload)
    points = _manifest_points(payload)
    if not points:
        raise ValueError("sweep manifest has no points")
    grouped: dict[tuple[str, str], set[float]] = {}
    seen: set[tuple[str, str, float]] = set()
    for point in points:
        system = str(point.get("system") or "")
        workload = str(point.get("workload") or "")
        qps = float(point.get("qps"))
        command = point.get("command")
        run_dir_value = str(point.get("run_dir") or "")
        key = (system, workload, qps)
        if not system or workload not in {"search", "add_native"}:
            raise ValueError(f"invalid system/workload in sweep point: {key}")
        if qps not in expected_qps:
            raise ValueError(f"QPS must be one of {expected_qps}, got {qps}")
        if key in seen:
            raise ValueError(f"duplicate sweep point: {key}")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) and value for value in command)
        ):
            raise ValueError(f"command must be a non-empty argv array: {key}")
        if "--profile-stages" not in command:
            raise ValueError(f"point does not enable --profile-stages: {key}")
        if "--qps" not in command:
            raise ValueError(f"point does not pass --qps: {key}")
        command_qps = float(command[command.index("--qps") + 1])
        if command_qps != qps:
            raise ValueError(f"manifest/command QPS mismatch: {key}")
        if not run_dir_value:
            raise ValueError(f"point has no run_dir: {key}")
        seen.add(key)
        grouped.setdefault((system, workload), set()).add(qps)
    for key, observed in grouped.items():
        if observed != set(expected_qps):
            raise ValueError(
                f"incomplete QPS coverage for {key}: {sorted(observed)} != "
                f"{list(expected_qps)}"
            )
    return points


def run_manifest(
    manifest: str | Path, receipt_path: str | Path, *, dry_run: bool = False
) -> dict[str, Any]:
    manifest = Path(manifest)
    receipt_path = Path(receipt_path)
    manifest_payload = json.loads(os.path.expandvars(manifest.read_text()))
    points = validate_manifest(manifest_payload)
    requested_qps = _manifest_qps(manifest_payload)
    receipt: dict[str, Any] = {
        "schema_version": "table5_track_c_sweep_v0.1.0",
        "status": "running",
        "started_at": _now(),
        "manifest": str(manifest.resolve()),
        "qps": list(requested_qps),
        "repeats": 1,
        "points": [],
    }
    _write(receipt_path, receipt)
    try:
        for point in points:
            run_dir = Path(point["run_dir"])
            if run_dir.exists():
                raise FileExistsError(f"refusing existing point run_dir: {run_dir}")
            point_receipt = {
                "system": point["system"],
                "workload": point["workload"],
                "qps": float(point["qps"]),
                "run_dir": str(run_dir),
                "command": point["command"],
                "status": "dry_run" if dry_run else "running",
                "started_at": _now(),
            }
            receipt["points"].append(point_receipt)
            _write(receipt_path, receipt)
            if dry_run:
                continue
            completed = subprocess.run(point["command"], check=False)
            point_receipt["returncode"] = completed.returncode
            native_receipt_path = run_dir / "run_receipt.json"
            if completed.returncode != 0 or not native_receipt_path.exists():
                raise RuntimeError(
                    f"sweep point failed: {point['system']} {point['workload']} "
                    f"QPS={point['qps']}"
                )
            native = json.loads(native_receipt_path.read_text())
            valid = (
                native.get("status") == "complete"
                and float(native.get("protocol", {}).get("qps")) == float(point["qps"])
                and native.get("protocol", {}).get("stage_profiling") is True
                and (run_dir / "spans.jsonl").exists()
                and "time_breakdown" in native
            )
            if not valid:
                raise RuntimeError(f"invalid profiled receipt: {native_receipt_path}")
            point_receipt.update(
                {
                    "status": "complete",
                    "completed_at": _now(),
                    "run_receipt": str(native_receipt_path.resolve()),
                }
            )
            _write(receipt_path, receipt)
        receipt["status"] = "dry_run" if dry_run else "complete"
        receipt["completed_at"] = _now()
        _write(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = _now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        _write(receipt_path, receipt)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_manifest(args.manifest, args.receipt, dry_run=args.dry_run)
    print(json.dumps({"status": result["status"], "receipt": args.receipt}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
