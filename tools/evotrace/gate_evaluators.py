"""Run every approved task's local evaluator once on its canonical seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

from bench.agent_memory.gem_oe.run_arm import resolve_task
from experiments.e2.gem_math_ale_plan_agent_matrix import TASKS
from tools.evotrace.wrap_evaluator import TEMPLATE

DSN = "postgresql://127.0.0.1:55432/evotrace_eg"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--raw", type=Path, default=Path("data/evotrace/raw"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--math-timeout", type=int, default=240)
    parser.add_argument("--ale-timeout", type=int, default=600)
    args = parser.parse_args(argv)
    work = args.out.parent / "evaluator_seed_smoke"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    passed = True
    for index, task_uid in enumerate(TASKS):
        task = resolve_task(args.dsn, task_uid, args.raw)
        task_dir = work / task_uid.replace(":", "_")
        task_dir.mkdir(exist_ok=True)
        seed_path = task_dir / (
            "initial_program.cpp"
            if task_uid.startswith("ale:")
            else "initial_program.py"
        )
        seed_path.write_text(task["seed_code"], encoding="utf-8")
        evaluator_path = Path(task["evaluator"])
        timeout = args.ale_timeout if task_uid.startswith("ale:") else args.math_timeout
        wrapped = task_dir / "wrapped_evaluator.py"
        wrapped.write_text(
            TEMPLATE.format(evaluator=str(evaluator_path), timeout=timeout),
            encoding="utf-8",
        )
        started = time.perf_counter()
        try:
            result = _load(wrapped, f"seed_gate_{index}").evaluate(str(seed_path))
        except BaseException as exc:  # noqa: BLE001
            result = {"error": f"{type(exc).__name__}: {exc}"}
        seconds = time.perf_counter() - started
        score = result.get("combined_score")
        ok = (
            not result.get("error")
            and score is not None
            and math.isfinite(float(score))
            and (
                not task_uid.startswith("ale:")
                or result.get("judge_result") == "ACCEPTED"
            )
        )
        passed = passed and ok
        rows.append(
            {
                "task_uid": task_uid,
                "task_key": task["task_key"],
                "domain": task["domain"],
                "evaluator": str(evaluator_path),
                "evaluator_sha256": _sha256(evaluator_path),
                "seed_artifact_uid": task["seed_artifact_uid"],
                "stored_seed_fitness": task["seed_fitness"],
                "live_result": result,
                "seconds": round(seconds, 3),
                "passed": ok,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        if not ok:
            break
    receipt = {
        "schema_version": "evotrace_evaluator_seed_gate_v1",
        "tasks_expected": len(TASKS),
        "tasks_checked": len(rows),
        "passed": passed and len(rows) == len(TASKS),
        "rows": rows,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
