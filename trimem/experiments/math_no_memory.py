"""Latest-prompt convex13 no-memory control, preserving the 20 -> 50 protocol."""

import argparse
import fcntl
import hashlib
import os
import subprocess
import time

from trimem.experiments import math_trajectory_extend as extension
from trimem.experiments import math_trajectory_run as base

OUT = base.ROOT / "results/trimem_v0.1.0/math_no_memory_v4_20261013"
TASK = "heilbronn_convex_13"
NAME = TASK + "__NoMemory__s42"


def cell_path(horizon):
    return OUT / f"it{horizon}" / NAME


def command(horizon):
    args = base.command(TASK, "VGR")
    args[args.index("--arm") + 1] = "nocontext"
    args[args.index("--max-iterations") + 1] = str(horizon)
    args[args.index("--out") + 1] = str(cell_path(horizon))
    if horizon == 50:
        args += [
            "--resume-from",
            str(cell_path(20) / "openevolve/checkpoints/checkpoint_20"),
        ]
    return args


def report():
    data = extension.merge_rows(
        extension.summarize_stage(OUT / "it20", TASK, "NoMemory", [], {}),
        extension.summarize_stage(OUT / "it50", TASK, "NoMemory", [], {}),
    )
    data["reference_policy"] = (
        "No inspirations, previous attempts or top programs; current parent and feedback retained"
    )
    base.write(OUT / "summary.json", {"updated_unix": time.time(), "cells": [data]})
    return data


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=base.ROOT, text=True
    ).strip()
    extension.branch_contract(
        branch, base.read(base.ROOT / "trimem/experiments/defaults.json", {})
    )
    files = [
        "bench/agent_memory/gem_oe/run_arm.py",
        "bench/agent_memory/gem_oe/trajectory_prompt.py",
        "trimem/experiments/trajectory_cell.py",
        "trimem/experiments/math_no_memory.py",
    ]
    base.write(
        OUT / "manifest.json",
        {
            "branch": branch,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=base.ROOT, text=True
            ).strip(),
            "source_sha256": {
                p: hashlib.sha256((base.ROOT / p).read_bytes()).hexdigest()
                for p in files
            },
            "task": TASK,
            "seed": 42,
            "checkpoints": [10, 20, 30, 40, 50],
            "reference_programs": 0,
            "input_tokens": 32768,
            "output_tokens": 32768,
            "thinking": False,
            "commands": [command(20), command(50)],
            "historical_runs_preserved": True,
        },
    )
    state = {
        "supervisor_pid": os.getpid(),
        "status": "starting",
        "started_unix": time.time(),
    }
    allowed = sorted(os.sched_getaffinity(0))
    cpus = [c for c in [26, 27] if c in allowed] or allowed[:2]
    os.sched_setaffinity(0, cpus)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    env = os.environ | {
        k: "1"
        for k in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TRIMEM_EVAL_THREADS",
            "TRIMEM_EVAL_SLOTS",
        ]
    }
    process = None
    try:
        for horizon in [20, 50]:
            path = cell_path(horizon)
            if extension.complete_cell(path, horizon):
                continue
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(
                    f"Existing incomplete output must be recovered explicitly: {path}"
                )
            path.mkdir(parents=True)
            with (path / "run.log").open("a") as log:
                process = subprocess.Popen(
                    command(horizon),
                    cwd=base.ROOT,
                    env=env
                    | {"TRIMEM_EXPERIMENT_CELL": str(path), "PYTHONUNBUFFERED": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            state.update(
                status="running", stage=horizon, cell_pid=process.pid, cpus=cpus
            )
            while process.poll() is None:
                state["last_heartbeat_unix"] = time.time()
                base.write(OUT / "status.json", state)
                report()
                time.sleep(15)
            if process.returncode or not extension.complete_cell(path, horizon):
                raise RuntimeError(
                    f"Stage {horizon} failed: {base.read(path / 'run_receipt.json', {})}"
                )
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        base.stop_child(process)
        raise
    finally:
        state["finished_unix"] = time.time()
        base.write(OUT / "status.json", state)
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    elif args.report:
        report()
    else:
        parser.error("Choose --run or --report")
