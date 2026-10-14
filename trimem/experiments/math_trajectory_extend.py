"""Four-task 20-round barrier, then checkpoint-preserving continuation through50.

Original cell directories are read-only after completion. Circle-packing20 is
added to the existing matrix. New21..50 outputs live in a separate stage root.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from trimem.experiments import math_trajectory_run as base

TASKS = [
    "heilbronn_convex_13",
    "second_autocorr_ineq",
    "heilbronn_triangle",
    "circle_packing",
]
PLANS = ["VGR", "SemRankOnly", "P1", "P2"]
STAGE20 = base.OUT
STAGE50 = base.ROOT / "results/trimem_v0.1.0/math_trajectory_v4_20261013_it50"
OUT = base.ROOT / "results/trimem_v0.1.0/math_trajectory_extension_20261013"


def cells():
    return [(task, plan, f"{task}__{plan}__s42") for task in TASKS for plan in PLANS]


def cell_command(task, plan, horizon):
    if horizon not in {20, 50}:
        raise ValueError("Only20 and50 stage horizons are supported")
    command = base.command(task, plan)
    name = f"{task}__{plan}__s42"
    command[command.index("--out") + 1] = str(
        (STAGE20 if horizon == 20 else STAGE50) / name
    )
    command[command.index("--max-iterations") + 1] = str(horizon)
    if horizon == 50:
        command += [
            "--resume-from",
            str(STAGE20 / name / "openevolve/checkpoints/checkpoint_20"),
        ]
    return command


def dispatch_iterations(path):
    return {
        int(r["iteration"])
        for r in base.rows(path / "generation_prompt_index.jsonl")
        if r.get("iteration") is not None
    }


def complete_cell(path, horizon):
    receipt = base.read(path / "run_receipt.json", {})
    expected = 20 if horizon == 20 else 30
    if (
        receipt.get("status") != "complete"
        or receipt.get("iterations_traced") != expected
    ):
        return False
    if horizon == 50 and receipt.get("resumed_at_iteration") != 20:
        return False
    expected_rounds = set(range(1, 21)) if horizon == 20 else set(range(21, 51))
    if dispatch_iterations(path) != expected_rounds:
        return False
    metadata = base.read(
        path / f"openevolve/checkpoints/checkpoint_{horizon}/metadata.json", {}
    )
    return metadata.get("last_iteration") == horizon


def barrier20():
    return {
        "required_cells": 16,
        "completed_cells": [
            name for _, _, name in cells() if complete_cell(STAGE20 / name, 20)
        ],
        "pending_cells": [
            name for _, _, name in cells() if not complete_cell(STAGE20 / name, 20)
        ],
    }


def stage_pending(horizon):
    if horizon == 50 and barrier20()["pending_cells"]:
        raise RuntimeError(
            "All16 cells must finish20 before any continuation is admitted"
        )
    root = STAGE20 if horizon == 20 else STAGE50
    pending = []
    for task, plan, name in cells():
        path = root / name
        if complete_cell(path, horizon):
            continue
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(
                f"Incomplete existing cell cannot be overwritten: {path}"
            )
        pending.append((task, plan, name))
    return pending


def limit():
    value = base.read(OUT / "control.json", {}).get("max_parallel", 6)
    if type(value) is not int or not 1 <= value <= 12:
        raise ValueError("max_parallel must be an integer from1 to12")
    return value


def branch_contract(current, defaults, override=None):
    expected = defaults.get("default_code_branch")
    if not isinstance(expected, str) or not expected:
        raise RuntimeError("defaults.json must define default_code_branch")
    required = override or expected
    if current != required:
        raise RuntimeError(
            f"Experiment branch mismatch: current={current!r}, required={required!r}"
        )
    return {
        "current": current,
        "default": expected,
        "override": override,
        "passed": True,
    }


def preflight():
    import psycopg
    from psycopg.rows import dict_row

    from bench.agent_memory.gem_oe.run_arm import resolve_task_trimem

    OUT.mkdir(parents=True, exist_ok=True)
    result = {"timestamp": time.time(), "errors": [], "tasks": {}}
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/v1/models", timeout=10
        ) as response:
            model = next(
                m
                for m in json.load(response)["data"]
                if m["id"] == "Qwen/Qwen3.8-27B-FP8"
            )
        result["model"] = model
        if model.get("max_model_len", 0) < 65536:
            raise RuntimeError("Model context below32768input +32768output")
        with psycopg.connect(base.DSN, connect_timeout=5, row_factory=dict_row) as conn:
            result["load_meta"] = conn.execute("SELECT * FROM load_meta").fetchone()
        if (
            result["load_meta"]["code_embedding"]["embedding_version"]
            != "qwen3-full-code-v1"
        ):
            raise RuntimeError("Unexpected code embedding contract")
        for task in TASKS:
            resolved = resolve_task_trimem(base.DSN, "math:" + task)
            task_info = {
                "seed_sha256": base.sha(resolved["seed_code"]),
                "system_sha256": base.sha(resolved["spec"]),
                "evaluator": resolved["evaluator"],
                "evaluator_sha256": hashlib.sha256(
                    Path(resolved["evaluator"]).read_bytes()
                ).hexdigest(),
            }
            result["tasks"][task] = task_info
            for plan in PLANS:
                source = STAGE20 / f"{task}__{plan}__s42"
                if complete_cell(source, 20) and (
                    hashlib.sha256(
                        (source / "initial_program.py").read_bytes()
                    ).hexdigest()
                    != task_info["seed_sha256"]
                ):
                    raise RuntimeError(
                        f"Task seed changed since original20: {task}/{plan}"
                    )
        result["barrier20"] = barrier20()
        result["pending20"] = stage_pending(20)
    except Exception as error:  # noqa: BLE001 - persist exact preflight blocker
        result["errors"].append(f"{type(error).__name__}: {error}")
    result["git"] = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=base.ROOT, text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=base.ROOT, text=True
        ).strip(),
    }
    try:
        result["branch_contract"] = branch_contract(
            result["git"]["branch"],
            base.read(base.ROOT / "trimem/experiments/defaults.json", {}),
            os.environ.get("TRIMEM_EXPERIMENT_BRANCH_OVERRIDE"),
        )
    except RuntimeError as error:
        result["errors"].append(str(error))
    result["source_sha256"] = {
        str(path.relative_to(base.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__),
            base.ROOT / "trimem/experiments/defaults.json",
            base.ROOT / "trimem/experiments/trajectory_cell.py",
            base.ROOT / "trimem/experiments/trajectory_worker.py",
            base.ROOT / "bench/agent_memory/gem_oe/run_arm.py",
            base.ROOT / "bench/agent_memory/gem_oe/trajectory_prompt.py",
            base.ROOT / "bench/agent_memory/gem_oe/trajectory_retriever.py",
            base.ROOT / "trimem/trajectory_query.py",
        ]
    }
    result["passed"] = not result["errors"]
    base.write(OUT / "preflight.json", result)
    interface = {
        "version": "trajectory-four-task-20-to50-v1",
        "original20_root": str(STAGE20),
        "continuation_root": str(STAGE50),
        "coordinator_root": str(OUT),
        "combined_summary": str(OUT / "summary.json"),
        "semantic_logs": [
            str(STAGE20 / "semantic_api.jsonl"),
            str(OUT / "semantic_api.jsonl"),
        ],
        "checkpoints": [10, 20, 30, 40, 50],
        "cells": [
            {
                "task": task,
                "plan": plan,
                "name": name,
                "stage20": str(STAGE20 / name),
                "stage50": str(STAGE50 / name),
            }
            for task, plan, name in cells()
        ],
    }
    base.write(OUT / "data_interface.json", interface)
    base.write(
        OUT / "manifest.json",
        {
            "tasks": TASKS,
            "plans": PLANS,
            "barrier": "All16 cells must complete20 before any iteration21+",
            "continuation_rounds": "21..50 only, fresh output directories",
            "checkpoints": [10, 20, 30, 40, 50],
            "max_parallel_default": 6,
            "original20_results_immutable": True,
            "input_tokens": 32768,
            "inspiration_tokens": 16384,
            "inspiration_max": 4,
            "output_tokens": 32768,
            "thinking": False,
            "git": result["git"],
            "branch_contract": result.get("branch_contract"),
            "source_sha256": result["source_sha256"],
            "commands20": [cell_command(t, p, 20) for t, p, _ in cells()],
            "commands50": [cell_command(t, p, 50) for t, p, _ in cells()],
        },
    )
    return result


def summarize_stage(root, task, plan, semantic, owners):
    previous = base.OUT
    try:
        base.OUT = root
        return base.summarize_cell(task, plan, semantic, owners)
    finally:
        base.OUT = previous


def merge_rows(first, continuation):
    """Stitch chronological stages without rerunning or counting the initial20 twice."""
    out = dict(first)
    out["stage20_status"] = first["status"]
    out["stage50_status"] = continuation["status"]
    out["status"] = (
        continuation["status"]
        if continuation["status"] != "pending"
        else first["status"] + " @20; pending21..50"
    )
    out["paths"] = {"stage20": first["path"], "stage50": continuation["path"]}
    out["seed_quality_source"] = (
        "original20 cell live seed evaluation; never re-evaluated as a new continuation seed"
    )
    out["execution_mode"] = (
        first.get("execution_mode", "unrecorded")
        + " → "
        + continuation.get("execution_mode", "pending")
    )
    curve = [dict(p) for p in first["curve"] if p["iteration"] <= 20]
    best = next(
        (p["quality"] for p in reversed(curve) if p["quality"] is not None), None
    )
    for point in continuation["curve"]:
        if point["iteration"] <= 20:
            continue
        score = point["quality"]
        if score is not None:
            best = score if best is None else max(best, score)
        curve.append({"iteration": point["iteration"], "quality": best})
    out["curve"] = curve
    for horizon in [10, 20, 30, 40, 50]:
        out[f"quality{horizon}"] = next(
            (p["quality"] for p in curve if p["iteration"] == horizon), None
        )
    sums = [
        "attempted_rounds",
        "retrieval_decisions",
        "child_count",
        "generation_prompt_tokens",
        "generation_completion_tokens",
        "generation_usage_unknown_calls",
        "generation_api_seconds",
        "retrieval_seconds_inclusive",
        "embedding_seconds",
        "evaluation_seconds_children",
        "generation_failed_or_invalid_calls",
    ]
    for key in sums:
        values = [
            r.get(key)
            for r in [first, continuation]
            if isinstance(r.get(key), (int, float))
        ]
        out[key] = sum(values) if values else None
    for key in ["prompt_audits", "retrieval_audits", "full_prompts"]:
        out[key] = first.get(key, []) + continuation.get(key, [])
    out["semantic_usage_stages"] = {
        "stage20": first["semantic_usage"],
        "stage50": continuation["semantic_usage"],
    }
    out["semantic_usage"] = {}
    for op in ["SemFilter", "SemScore"]:
        records = [r["semantic_usage"][op] for r in [first, continuation]]
        out["semantic_usage"][op] = {
            key: sum(r[key] for r in records if isinstance(r.get(key), (int, float)))
            if any(isinstance(r.get(key), (int, float)) for r in records)
            else None
            for key in [
                "observed_api_calls",
                "prompt_tokens",
                "completion_tokens",
                "unknown_usage_calls",
                "ambiguous_api_calls_excluded",
            ]
        }
        out["semantic_usage"][op]["note"] = (
            "Observed uniquely-attributed API usage only, summed across non-overlapping20/21..50 stages. Ambiguous/missing usage is unavailable."
        )
    return out


def report():
    semantic = base.rows(STAGE20 / "semantic_api.jsonl") + base.rows(
        OUT / "semantic_api.jsonl"
    )
    owners = {}
    for task, plan, name in cells():
        op = (
            "SemFilter"
            if plan in {"P1", "P2"}
            else "SemScore"
            if plan == "SemRankOnly"
            else None
        )
        if op:
            for root in [STAGE20, STAGE50]:
                for row in base.rows(root / name / "trajectory_retrieval.jsonl"):
                    owners.setdefault((op, row.get("context_sha256")), set()).add(name)
    data = [
        merge_rows(
            summarize_stage(STAGE20, t, p, semantic, owners),
            summarize_stage(STAGE50, t, p, semantic, owners),
        )
        for t, p, _ in cells()
    ]
    base.write(
        OUT / "summary.json",
        {
            "updated_unix": time.time(),
            "checkpoints": [10, 20, 30, 40, 50],
            "cells": data,
        },
    )
    fields = [
        "task",
        "plan",
        "status",
        "attempted_rounds",
        "child_count",
        "seed_quality",
        "quality10",
        "quality20",
        "quality30",
        "quality40",
        "quality50",
        "generation_prompt_tokens",
        "generation_completion_tokens",
        "generation_api_seconds",
        "retrieval_seconds_inclusive",
        "execution_mode",
    ]
    with (OUT / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)
    from trimem.experiments.trajectory_dashboard import render

    render(data, base.HTML, STAGE20 / "historical_best.json")
    return data


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not base.read(OUT / "launch_authorized.json", {}).get("prerequisites_passed"):
        raise RuntimeError("Coordinator prerequisite authorization is required")
    checked = preflight()
    if not checked["passed"]:
        raise RuntimeError(str(checked["errors"]))
    allowed = sorted(os.sched_getaffinity(0))
    pool = allowed[-min(12, max(1, len(allowed) // 2)) :]
    os.sched_setaffinity(0, pool)
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    env = os.environ | {
        key: "1"
        for key in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TRIMEM_EVAL_THREADS",
            "TRIMEM_EVAL_SLOTS",
            "TRIMEM_EMBEDDING_THREADS",
        ]
    }
    env["PYTHONUNBUFFERED"] = "1"
    processes = subprocess.check_output(
        ["ps", "-eo", "pid,args"], text=True
    ).splitlines()
    if any(
        (
            "-m trimem.worker " in line
            or "-m trimem.experiments.trajectory_worker " in line
        )
        and "55450" in line
        for line in processes
    ):
        raise RuntimeError(
            "Existing semantic worker on55450 must finish before starting extension"
        )
    state = {
        "supervisor_pid": os.getpid(),
        "status": "starting",
        "started_unix": time.time(),
        "active_cells": {},
        "completed_cells": [
            f"20:{name}" for name in checked["barrier20"]["completed_cells"]
        ],
        "failed_cells": {},
        "cpu_pool": pool,
        "stage": 20,
    }
    active = {}
    worker = None
    try:
        with (OUT / "semantic_worker.log").open("a") as log:
            worker = subprocess.Popen(
                [
                    str(base.ROOT / ".venv-e0/bin/python"),
                    "-m",
                    "trimem.experiments.trajectory_worker",
                    "--dsn",
                    base.DSN,
                    "--log",
                    str(OUT / "semantic_api.jsonl"),
                    "--concurrency",
                    "8",
                    "--max-take",
                    "8",
                ],
                cwd=base.ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        state["worker_pid"] = worker.pid
        for horizon in [20, 50]:
            pending = stage_pending(horizon)
            state.update(stage=horizon, pending_cells=[x[2] for x in pending])
            if horizon == 50:
                barrier = barrier20()
                if barrier["pending_cells"]:
                    raise RuntimeError(
                        "Barrier20 failed immediately before continuation"
                    )
                base.write(
                    OUT / "barrier20.json",
                    dict(barrier, passed=True, timestamp=time.time()),
                )
            root = STAGE20 if horizon == 20 else STAGE50
            root.mkdir(parents=True, exist_ok=True)
            while pending or active:
                if worker.poll() is not None:
                    raise RuntimeError("Semantic worker exited")
                for name, process in list(active.items()):
                    if process.poll() is None:
                        continue
                    if process.returncode != 0 or not complete_cell(
                        root / name, horizon
                    ):
                        state["failed_cells"][f"{horizon}:{name}"] = base.read(
                            root / name / "run_receipt.json", {}
                        )
                    else:
                        state["completed_cells"].append(f"{horizon}:{name}")
                    del active[name]
                    del state["active_cells"][name]
                if state["failed_cells"] and not active:
                    raise RuntimeError(
                        f"Stage{horizon} has failed cells; preserving all outputs, barrier not advanced"
                    )
                if not state["failed_cells"]:
                    for task, plan, name in pending[: max(0, limit() - len(active))]:
                        cell = root / name
                        if cell.exists() and any(cell.iterdir()):
                            raise RuntimeError(f"Refusing overwrite: {cell}")
                        cell.mkdir(parents=True, exist_ok=True)
                        occupied = {v["slot"] for v in state["active_cells"].values()}
                        slot = next(i for i in range(12) if i not in occupied)
                        cpus = sorted(
                            {
                                pool[(2 * slot) % len(pool)],
                                pool[(2 * slot + 1) % len(pool)],
                            }
                        )
                        device = f"cuda:{slot % 2}"
                        with (cell / "run.log").open("a") as log:
                            process = subprocess.Popen(
                                [
                                    "taskset",
                                    "-c",
                                    ",".join(map(str, cpus)),
                                    *cell_command(task, plan, horizon),
                                ],
                                cwd=base.ROOT,
                                env=env
                                | {
                                    "TRIMEM_EXPERIMENT_CELL": str(cell),
                                    "TRIMEM_EMBEDDING_DEVICE": device,
                                },
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        active[name] = process
                        state["active_cells"][name] = {
                            "pid": process.pid,
                            "task": task,
                            "plan": plan,
                            "horizon": horizon,
                            "slot": slot,
                            "cpus": cpus,
                            "embedding_device": device,
                            "started_unix": time.time(),
                            "output": str(cell),
                        }
                        base.write(
                            cell / "resource_config.json",
                            dict(
                                state["active_cells"][name],
                                execution_mode="parallel",
                                resume_from=str(
                                    STAGE20
                                    / name
                                    / "openevolve/checkpoints/checkpoint_20"
                                )
                                if horizon == 50
                                else None,
                            ),
                        )
                        pending.remove((task, plan, name))
                        base.write(OUT / "status.json", state)
                summary = report()
                by_name = {f"{r['task']}__{r['plan']}__s42": r for r in summary}
                for name in active:
                    state["active_cells"][name].update(
                        attempted_rounds_combined=by_name[name]["attempted_rounds"],
                        child_count_combined=by_name[name]["child_count"],
                    )
                    for row in base.rows(root / name / "llm_telemetry.jsonl"):
                        if (row.get("prompt_tokens") or 0) > 32768 or row.get(
                            "thinking_enabled"
                        ) is True:
                            raise RuntimeError(
                                f"Prompt/thinking contract violation: {name}"
                            )
                state.update(
                    status="running",
                    last_heartbeat_unix=time.time(),
                    max_parallel=limit(),
                    pending_cells=[x[2] for x in pending],
                )
                base.write(OUT / "status.json", state)
                if pending or active:
                    time.sleep(15)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        for process in active.values():
            base.stop_child(process)
        raise
    finally:
        base.stop_child(worker)
        state["finished_unix"] = time.time()
        base.write(OUT / "status.json", state)
        report()


def main():
    os.chdir(base.ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        result = preflight()
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "errors": result["errors"],
                    "pending20": result.get("barrier20"),
                }
            )
        )
        return 0 if result["passed"] else 1
    if args.report:
        report()
        return 0
    if args.run:
        run()
        return 0
    parser.error("Choose --preflight, --report, or --run")


if __name__ == "__main__":
    sys.exit(main())
