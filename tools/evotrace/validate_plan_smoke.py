"""Fail-closed validation for the six-cell Math/ALE plan smoke run."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

TASKS = ("math:circle_packing", "ale:ahc008")
PLANS = ("vfwd", "rrev", "aivg")
STAGES = (
    "embed_ms",
    "ann_ms",
    "graph_ms",
    "predicate_ms",
    "dedup_rank_ms",
    "hydrate_ms",
    "executor_overhead_ms",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing artifact: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty artifact: {path}")
    return rows


def _cell(run: Path, task: str, plan: str) -> Path:
    return run / "cells" / f"{task.replace(':', '_')}__gem__{plan}__s42"


def validate(run: Path) -> dict[str, Any]:
    manifest_path = run / "matrix_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {(task, plan) for task in TASKS for plan in PLANS}
    observed = {
        (row.get("task_uid"), row.get("physical_plan"))
        for row in manifest.get("results", [])
    }
    if manifest.get("status") != "complete" or observed != expected:
        raise ValueError("smoke matrix is not an exact complete six-cell matrix")

    rows: list[dict[str, Any]] = []
    for task, plan in sorted(expected):
        cell = _cell(run, task, plan)
        receipt = json.loads((cell / "run_receipt.json").read_text())
        required = {
            "status": "complete",
            "task_uid": task,
            "physical_plan": plan,
            "resolved_seed": 42,
            "iterations_expected": 2,
            "iterations_traced": 2,
            "injection_frequency": 0.5,
            "injection_rate_requested": 0.5,
            "max_injected": 5,
            "inject_as": "changes",
            "injection_render_contract": "external_program_code_is_changes_description",
        }
        mismatches = {
            key: {"expected": value, "observed": receipt.get(key)}
            for key, value in required.items()
            if receipt.get(key) != value
        }
        expected_language = "cpp" if task.startswith("ale:") else "python"
        if receipt.get("language") != expected_language:
            mismatches["language"] = {
                "expected": expected_language,
                "observed": receipt.get("language"),
            }
        score = receipt.get("live_seed_fitness")
        if score is None or not math.isfinite(float(score)):
            mismatches["live_seed_fitness"] = {
                "expected": "finite score",
                "observed": score,
            }
        if mismatches:
            raise ValueError(f"receipt mismatch for {task}/{plan}: {mismatches}")

        evolution = _jsonl(cell / "evolution_trace.jsonl")
        injections = _jsonl(cell / "injection_trace.jsonl")
        telemetry = _jsonl(cell / "retrieval_telemetry.jsonl")
        if len(evolution) != 2 or len(injections) != 2:
            raise ValueError(f"incomplete two-iteration traces for {task}/{plan}")
        gate_open = sum(bool(row.get("gate_open")) for row in injections)
        if gate_open != 1:
            raise ValueError(
                f"actual injection gate fraction is not 0.5 for {task}/{plan}"
            )
        for evo in evolution:
            child_score = evo.get("child_metrics", {}).get("combined_score")
            if child_score is None or not math.isfinite(float(child_score)):
                raise ValueError(f"non-finite evaluated child score for {task}/{plan}")

        max_reconciliation_error = 0.0
        for telemetry_row in telemetry:
            if telemetry_row.get("physical_plan") != plan:
                raise ValueError(f"telemetry plan mismatch for {task}/{plan}")
            total = float(telemetry_row.get("retriever_total_ms", 0.0))
            stage_sum = sum(float(telemetry_row.get(stage, 0.0)) for stage in STAGES)
            error = abs(stage_sum - total) / max(total, 1e-9)
            max_reconciliation_error = max(max_reconciliation_error, error)
            if error > 0.05:
                raise ValueError(
                    f"stage reconciliation exceeds 5% for {task}/{plan}: {error:.6f}"
                )
        rows.append(
            {
                "task_uid": task,
                "physical_plan": plan,
                "evaluated_iterations": len(evolution),
                "injection_gate_fraction": gate_open / len(injections),
                "telemetry_rows": len(telemetry),
                "max_stage_reconciliation_error": max_reconciliation_error,
            }
        )
    return {
        "schema_version": "evotrace_math_ale_plan_smoke_gate_v1",
        "passed": True,
        "cells_checked": len(rows),
        "rows": rows,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = validate(args.run)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        receipt = {
            "schema_version": "evotrace_math_ale_plan_smoke_gate_v1",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
