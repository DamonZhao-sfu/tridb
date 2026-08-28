"""Run the approved one-seed GEM+ Math physical-plan agent matrix sequentially.

Primary: three tasks x three plans.  Secondary: one task x three plans.  Every cell
uses seed 42, injection frequency 0.5, injection rate 0.5, changes-only injection,
and the frozen 40-iteration OpenEvolve configuration from ``run_arm``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PRIMARY = (
    "math:heilbronn_triangle",
    "math:heilbronn_convex_13",
    "math:circle_packing",
)
SECONDARY = ("math:third_autocorr_ineq",)
PLANS = ("vfwd", "rrev", "aivg")
SEED = 42
INJECTION_FREQUENCY = 0.5


def _complete(cell: Path) -> bool:
    receipt = cell / "run_receipt.json"
    if not receipt.is_file():
        return False
    row = json.loads(receipt.read_text())
    return row.get("status") == "complete" and row.get("resolved_seed") == SEED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dsn", default="postgresql://127.0.0.1:55432/evotrace_eg")
    parser.add_argument("--scope", default="evotrace:349117b0")
    parser.add_argument("--llm-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--llm-model", default="qwen3.8")
    parser.add_argument(
        "--embedding-endpoint", default="http://127.0.0.1:8001/v1/embeddings"
    )
    parser.add_argument(
        "--embedding-receipt",
        type=Path,
        default=None,
        help="required passing live-node embedding gate receipt for a real run",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cells = []
    for task in PRIMARY + SECONDARY:
        tier = "primary" if task in PRIMARY else "secondary"
        for plan in PLANS:
            cell = args.out / f"{task.replace(':', '_')}__gem__{plan}__s{SEED}"
            cells.append(
                {
                    "task_uid": task,
                    "tier": tier,
                    "plan": plan,
                    "seed": SEED,
                    "path": str(cell),
                }
            )
    manifest = {
        "schema_version": "gem_math_plan_agent_matrix_v1",
        "cells": cells,
        "agent_seed": SEED,
        "injection_frequency": INJECTION_FREQUENCY,
        "injection_rate": 0.5,
        "inject_as": "changes",
        "iterations": 40,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "dry_run" if args.dry_run else "running",
    }
    manifest_path = args.out / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if not args.dry_run:
        gate_path = args.embedding_receipt or (args.out / "embedding_gate.json")
        gate = json.loads(gate_path.read_text()) if gate_path.is_file() else None
        if not gate or not gate.get("passed"):
            manifest["status"] = "blocked"
            manifest["blocker"] = (
                f"live node embedding parity gate missing/failed: {gate_path}"
            )
            manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(manifest["blocker"], file=sys.stderr)
            return 2

    failures = []
    for row in cells:
        cell = Path(row["path"])
        if not args.force and _complete(cell):
            continue
        command = [
            args.python,
            "-m",
            "bench.agent_memory.gem_oe.run_arm",
            "--task",
            row["task_uid"],
            "--arm",
            "gem",
            "--out",
            str(cell),
            "--dsn",
            args.dsn,
            "--scope",
            args.scope,
            "--seed-layer",
            "node",
            "--physical-plan",
            row["plan"],
            "--m-seeds",
            "4",
            "--seed",
            str(SEED),
            "--injection-frequency",
            str(INJECTION_FREQUENCY),
            "--injection-rate",
            "0.5",
            "--inject-as",
            "changes",
            "--llm-base",
            args.llm_base,
            "--llm-model",
            args.llm_model,
            "--embedding-endpoint",
            args.embedding_endpoint,
        ]
        if args.dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures.append({**row, "returncode": completed.returncode})
            if not args.dry_run:
                break

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["failures"] = failures
    manifest["status"] = (
        "failed" if failures else ("dry_run" if args.dry_run else "complete")
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
