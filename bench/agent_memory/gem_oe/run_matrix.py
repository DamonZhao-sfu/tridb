"""Run the (task x arm) matrix, one cell per process, across the available replicas.

    python3 -m bench.agent_memory.gem_oe.run_matrix \
        --arms none gem --endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
        --out bench/out/oe/phase2

Cells are independent, so they are pinned round-robin onto the serving replicas: two
GPUs means two cells in flight, not one cell split across two GPUs (the host has no
NVLink, so tensor-parallel would pay a PCIe all-reduce per layer for no benefit here).

Every cell writes its own receipt; this driver only schedules and collects. It makes NO
latency claim: cells share the replicas with each other, so per-request timings here
are progress instrumentation, not evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: The math tasks that ship a run-local evaluator. `math:signal_processing` is absent
#: on purpose: no session of it published one, so it cannot be scored here at all.
MATH_TASKS = [
    "math:circle_packing",
    "math:first_autocorr_ineq",
    "math:heilbronn_convex_13",
    "math:heilbronn_triangle",
    "math:second_autocorr_ineq",
    "math:third_autocorr_ineq",
    "math:uncertainty_ineq",
]

RUNNER = "bench.agent_memory.gem_oe.run_arm"


def cell_dir(out: Path, task: str, arm: str) -> Path:
    return out / f"{task.replace(':', '_')}__{arm}"


def run_cell(
    python: str, task: str, arm: str, endpoint: str, out: Path, extra: list[str]
) -> dict[str, Any]:
    target = cell_dir(out, task, arm)
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, "-m", RUNNER,
        "--task", task, "--arm", arm,
        "--out", str(target),
        "--llm-base", endpoint,
        *extra,
    ]
    log = target / "cell.log"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=None)
    return {
        "task": task,
        "arm": arm,
        "endpoint": endpoint,
        "returncode": proc.returncode,
        "seconds": round(time.perf_counter() - started, 1),
        "out": str(target),
        "log": str(log),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", default=MATH_TASKS)
    ap.add_argument("--arms", nargs="+", default=["none", "gem"])
    ap.add_argument(
        "--endpoints", nargs="+", default=["http://127.0.0.1:8001/v1"],
        help="one per serving replica; cells are pinned round-robin",
    )
    ap.add_argument(
        "--python", default=".venv-e0/bin/python",
        help="interpreter that has openevolve installed",
    )
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    cells = [(t, a) for t in args.tasks for a in args.arms]
    plan = [
        (task, arm, args.endpoints[i % len(args.endpoints)])
        for i, (task, arm) in enumerate(cells)
    ]
    print(f"{len(plan)} cells over {len(args.endpoints)} replica(s)")
    for task, arm, endpoint in plan:
        print(f"  {task:<28} {arm:<10} -> {endpoint}")
    if args.dry_run:
        extra = [*extra, "--dry-run"]

    started = time.perf_counter()
    # One worker per replica: more would queue requests behind each other on the same
    # server and make every cell slower without finishing any sooner.
    with ThreadPoolExecutor(max_workers=len(args.endpoints)) as pool:
        results = list(pool.map(
            lambda item: run_cell(args.python, item[0], item[1], item[2], args.out, extra),
            plan,
        ))

    ok = [r for r in results if r["returncode"] == 0]
    summary = {
        "cells": results,
        "cells_ok": len(ok),
        "cells_failed": len(results) - len(ok),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "arms": args.arms,
        "tasks": args.tasks,
        "endpoints": args.endpoints,
        "latency_claim": "NONE. Cells share the replicas; timings are progress only.",
    }
    (args.out / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    print()
    for row in results:
        flag = "ok  " if row["returncode"] == 0 else f"FAIL({row['returncode']})"
        print(f"  {flag} {row['task']:<28} {row['arm']:<10} {row['seconds']:>8.1f}s")
    print(f"\n{len(ok)}/{len(results)} cells ok -> {args.out / 'matrix_summary.json'}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
