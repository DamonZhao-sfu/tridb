"""Queue ALE behind Math50, then evaluate exact public checkpoints on held-out data.

Private scoring is a separate process and output tree, with no route back into
OpenEvolve. Five10-round stages preserve checkpoint provenance and token accounting.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from trimem.experiments import math_trajectory_extend as math_run
from trimem.experiments import math_trajectory_run as base

TASKS = ["ahc008", "ahc011", "ahc024", "ahc025", "ahc026"]
PLANS = ["VGR", "SemRankOnly", "P1", "P2"]
CHECKPOINTS = [10, 20, 30, 40, 50]
PRIVATE_WORKERS = 4
OUT = base.ROOT / "results/trimem_v0.1.0/ale_trajectory_v4_20261013"


def cells():
    return [(task, plan, f"{task}__{plan}__s42") for task in TASKS for plan in PLANS]


def stage_path(task, plan, iteration):
    return OUT / f"it{iteration}" / f"{task}__{plan}__s42"


def math_barrier():
    missing = [
        name
        for _, _, name in math_run.cells()
        if not math_run.complete_cell(math_run.STAGE50 / name, 50)
    ]
    state = base.read(math_run.OUT / "status.json", {})
    return {
        "passed": not missing and state.get("status") == "complete",
        "missing_cells": missing,
        "math_status": state.get("status", "unavailable"),
        "required": "all16 Math cells complete50 and Math supervisor complete",
    }


def command(task, plan, iteration):
    if task not in TASKS or plan not in PLANS or iteration not in CHECKPOINTS:
        raise ValueError("Unknown ALE task/plan/checkpoint")
    args = base.command(task, plan)
    args[args.index("-m") + 1] = "trimem.experiments.ale_trajectory_run"
    args.insert(args.index("--task"), "--cell")
    args[args.index("--task") + 1] = "ale:" + task
    args[args.index("--out") + 1] = str(stage_path(task, plan, iteration))
    args[args.index("--max-iterations") + 1] = str(iteration)
    args[args.index("--eval-timeout") + 1] = "600"
    args += ["--language", "cpp", "--no-simplify-nudge"]
    if iteration > 10:
        args += [
            "--resume-from",
            str(
                stage_path(task, plan, iteration - 10)
                / f"openevolve/checkpoints/checkpoint_{iteration - 10}"
            ),
        ]
    return args


def complete_stage(path, iteration):
    receipt = base.read(path / "run_receipt.json", {})
    if receipt.get("status") != "complete" or receipt.get("iterations_traced") != 10:
        return False
    if iteration > 10 and receipt.get("resumed_at_iteration") != iteration - 10:
        return False
    if math_run.dispatch_iterations(path) != set(range(iteration - 9, iteration + 1)):
        return False
    metadata = base.read(
        path / f"openevolve/checkpoints/checkpoint_{iteration}/metadata.json", {}
    )
    return metadata.get("last_iteration") == iteration


def checkpoint_selection(path, iteration):
    """Read only checkpoint_N's public winner, never a later checkpoint or latest glob."""
    ck = path / f"openevolve/checkpoints/checkpoint_{iteration}"
    info = base.read(ck / "best_program_info.json", {})
    metadata = base.read(ck / "metadata.json", {})
    if (
        metadata.get("last_iteration") != iteration
        or info.get("current_iteration") != iteration
    ):
        raise ValueError("Requested checkpoint identity does not match metadata")
    uid = info.get("id")
    if not isinstance(uid, str) or not uid or "/" in uid or uid.startswith("gemext:"):
        raise ValueError("Public winner must be an own-run checkpoint program")
    program = base.read(ck / "programs" / f"{uid}.json", {})
    found = program.get("iteration_found")
    if not isinstance(found, int) or found > iteration or found < 0:
        raise ValueError("Checkpoint winner contains future or missing iteration")
    code = program.get("code")
    if not isinstance(code, str) or not code:
        raise ValueError("Checkpoint winner has no complete source code")
    metrics = program.get("metrics") or {}
    if metrics.get("combined_score") != (info.get("metrics") or {}).get(
        "combined_score"
    ):
        raise ValueError("Winner program and checkpoint metrics disagree")
    return {
        "checkpoint": iteration,
        "checkpoint_path": str(ck),
        "program_id": uid,
        "program_iteration": found,
        "public_quality": metrics.get("combined_score"),
        "public_metrics": metrics,
        "source_code": code,
        "code_sha256": base.sha(code),
        "selection_policy": "best_program_id selected solely by public evaluation at exact checkpoint",
    }


def isolated_evaluator(source, destination):
    """Change only evaluator concurrency from8 to1, preserving all50 public cases."""
    text = source.read_text()
    if not re.search(r"^NUM_CASES\s*=\s*50\s*$", text, flags=re.MULTILINE):
        raise ValueError("ALE evaluator must use50 public cases")
    changed, count = re.subn(
        r"^WORKERS\s*=\s*\d+\s*$", "WORKERS = 1", text, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError("Expected exactly one ALE evaluator WORKERS constant")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(changed)
    return {
        "source": str(source),
        "source_sha256": base.sha(text),
        "copy": str(destination),
        "copy_sha256": base.sha(changed),
        "public_cases": 50,
        "workers": 1,
        "changed_semantics": False,
    }


def cell_main(argv):
    from bench.agent_memory.gem_oe import run_arm
    from trimem.experiments import trajectory_cell  # installs instrumentation in parent

    trajectory_cell.install()
    target = Path(argv[argv.index("--out") + 1])
    original = run_arm.resolve_task_trimem

    def resolve(dsn, task_uid):
        task = original(dsn, task_uid)
        copied = target / "ale_public_evaluator.py"
        record = isolated_evaluator(Path(task["evaluator"]), copied)
        base.write(target / "public_evaluator_contract.json", record)
        task["evaluator"] = str(copied)
        return task

    run_arm.resolve_task_trimem = resolve
    os.environ["TRIMEM_ALE_CELL"] = "1"
    return run_arm.main(argv)


def private_path(task, plan, iteration):
    return OUT / "private" / f"{task}__{plan}__s42__it{iteration}.json"


def private_score(task, plan, iteration):
    from trimem.experiments import ale_private

    path = stage_path(task, plan, iteration)
    selected = checkpoint_selection(path, iteration)
    target = private_path(task, plan, iteration)
    if target.exists():
        saved = base.read(target, {})
        if saved.get("code_sha256") != selected["code_sha256"]:
            raise RuntimeError(
                "Private evaluation snapshot differs from existing result"
            )
        return 0 if saved.get("status") == "complete" else 1
    OUT.joinpath("private").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ALE_BENCH_DATA", "/localhome/hza214/ale-bench-data")
    os.environ.setdefault("ALE_BENCH_NO_CONTAINER_USER", "1")
    os.environ.setdefault(
        "DOCKER_HOST", f"unix:///run/user/{os.getuid()}/podman/podman.sock"
    )
    ale_private.CACHE = OUT / "private_cache.json"
    cache = base.read(ale_private.CACHE, {})
    result = {key: value for key, value in selected.items() if key != "source_code"}
    result.update(
        task=task,
        plan=plan,
        iteration=iteration,
        private_feedback_to_agent=False,
        started_unix=time.time(),
        status="running",
    )
    base.write(
        target.with_suffix(".selection.json"),
        dict(result, source_code=selected["source_code"]),
    )
    try:
        seed = (stage_path(task, plan, 10) / "initial_program.py").read_text()
        workers = int(os.getenv("TRIMEM_ALE_PRIVATE_WORKERS", str(PRIVATE_WORKERS)))
        baseline = ale_private.score(task, seed, cache, workers=workers)
        judged = ale_private.score(
            task, selected["source_code"], cache, workers=workers
        )
        result.update(
            status="complete",
            private_abs=judged["private_abs"],
            rank=judged["rank"],
            performance=judged["performance"],
            judge=judged["judge"],
            cases=judged["cases"],
            delta_performance=judged["performance"] - baseline["performance"],
            seed_private=baseline,
            private_eval=judged,
        )
    except Exception as error:  # noqa: BLE001 - private failure remains separate from search
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result["finished_unix"] = time.time()
    base.write(target, result)
    return 0 if result["status"] == "complete" else 1


def report():
    data = []
    for task, plan, name in cells():
        checkpoints = []
        for iteration in CHECKPOINTS:
            path = stage_path(task, plan, iteration)
            record = {
                "iteration": iteration,
                "status": "pending",
                "public_quality": None,
                "private_abs": None,
                "rank": None,
                "performance": None,
                "delta_performance": None,
                "judge": None,
            }
            if complete_stage(path, iteration):
                try:
                    selection = checkpoint_selection(path, iteration)
                    record.update(
                        {
                            key: value
                            for key, value in selection.items()
                            if key != "source_code"
                        },
                        status="public_complete_private_pending",
                    )
                except ValueError as error:
                    record.update(status="selection_error", error=str(error))
            private = base.read(private_path(task, plan, iteration), {})
            if private:
                record.update(private)
            checkpoints.append(record)
        data.append(
            {
                "task": task,
                "plan": plan,
                "checkpoints": checkpoints,
                "paths": {
                    f"it{n}": str(stage_path(task, plan, n)) for n in CHECKPOINTS
                },
                "private_feedback_to_agent": False,
            }
        )
    summary = {
        "updated_unix": time.time(),
        "cells": data,
        "semantic_logs": [str(OUT / "semantic_api.jsonl")],
        "generation_usage_source": "llm_telemetry.jsonl in each non-overlapping10-round stage",
        "cost_aggregation": "sum stage1..N APIusage, eachstageexactly10newrounds; private evaluation generates noLLM tokens",
    }
    base.write(OUT / "summary.json", summary)
    try:
        from trimem.experiments.trajectory_quality_cost import render_report

        render_report()
    except ImportError:
        pass  # Scheduler and raw result persistence do not depend on an optional renderer.
    return summary


def preflight():
    from bench.agent_memory.gem_oe.run_arm import resolve_task_trimem

    result = {
        "timestamp": time.time(),
        "errors": [],
        "tasks": {},
        "math_barrier": math_barrier(),
    }
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=base.ROOT, text=True
        ).strip()
        result["branch"] = math_run.branch_contract(
            branch,
            base.read(base.ROOT / "trimem/experiments/defaults.json", {}),
            os.getenv("TRIMEM_EXPERIMENT_BRANCH_OVERRIDE"),
        )
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=base.ROOT, text=True
        ).strip()
        for task in TASKS:
            resolved = resolve_task_trimem(base.DSN, "ale:" + task)
            source = Path(resolved["evaluator"])
            body = source.read_text()
            if not re.search(r"^NUM_CASES\s*=\s*50\s*$", body, flags=re.MULTILINE):
                raise RuntimeError("Wrong public casecount")
            result["tasks"][task] = {
                "seed_sha256": base.sha(resolved["seed_code"]),
                "system_sha256": base.sha(resolved["spec"]),
                "evaluator": str(source),
                "evaluator_sha256": base.sha(body),
            }
        socket = Path(f"/run/user/{os.getuid()}/podman/podman.sock")
        if not socket.exists():
            raise RuntimeError(f"Missing Podman socket: {socket}")
        if not Path("/localhome/hza214/ale-bench-data").exists():
            raise RuntimeError("Missing ALE-Bench dataset")
        import importlib.metadata
        import urllib.request

        import docker

        result["ale_bench_version"] = importlib.metadata.version("ale-bench")
        import zipfile

        from ale_bench.utils import get_cache_dir
        from huggingface_hub import try_to_load_from_cache

        result["dataset_archives"] = {}
        for task in TASKS:
            archive = try_to_load_from_cache(
                "SakanaAI/ALE-Bench",
                task + ".zip",
                repo_type="dataset",
                cache_dir=get_cache_dir(),
            )
            if not isinstance(archive, str):
                raise RuntimeError(f"Uncached ALE dataset archive: {task}")  # noqa: TRY004 - missing cached resource, not a caller type error
            with zipfile.ZipFile(archive) as zipped:
                metadata_file = next(
                    name for name in zipped.namelist() if name.endswith("/data.json")
                )
                metadata = json.loads(zipped.read(metadata_file))
            result["dataset_archives"][task] = {
                "path": archive,
                "private_lite_cases": len(metadata["seeds"]["private_lite"]),
                "public_private_seed_overlap": sorted(
                    set(range(50)) & set(metadata["seeds"]["private_lite"])
                ),
                "split_check": "seed-disjoint; case_gen uses supplied public seeds, private_eval uses generated private_lite seeds",
                "generated_input_identity_checked": False,
                "public_protocol": "generated seeds0..49, unchanged from prior ALE evaluator",
            }

        if any(
            row["public_private_seed_overlap"]
            for row in result["dataset_archives"].values()
        ):
            raise RuntimeError(
                "Public/private seed overlap detected; review protocol before launch"
            )
        client = docker.DockerClient(base_url=f"unix://{socket}")
        result["container_engine"] = client.version()
        result["container_images"] = [
            tag for image in client.images.list() for tag in image.tags
        ]
        client.close()
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/v1/models", timeout=10
        ) as response:
            result["model"] = next(
                m
                for m in json.load(response)["data"]
                if m["id"] == "Qwen/Qwen3.8-27B-FP8"
            )
        if result["model"].get("max_model_len", 0) < 65536:
            raise RuntimeError("Generation service context smaller than32768+32768")
    except Exception as error:  # noqa: BLE001 - persist exact read-only preflight blocker
        result["errors"].append(f"{type(error).__name__}: {error}")
    result["passed"] = not result["errors"]
    base.write(OUT / "preflight.json", result)
    base.write(
        OUT / "manifest.json",
        {
            "tasks": TASKS,
            "plans": PLANS,
            "checkpoints": CHECKPOINTS,
            "math50_barrier": True,
            "max_parallel": 6,
            "public_workers": 1,
            "private_workers": PRIVATE_WORKERS,
            "private_concurrency": 1,
            "public_cpu_pool_max": 12,
            "private_cpu_reservation": 4,
            "public_cases": 50,
            "language": "cpp",
            "eval_timeout": 600,
            "simplify_nudge": False,
            "diff_match": "exact",
            "input_tokens": 32768,
            "memory_tokens": 16384,
            "inspirations": 4,
            "output_tokens": 32768,
            "private_policy": "Exactcheckpointpublicbest only, no feedback/selection from private outcomes",
            "git_commit": result.get("git_commit"),
            "source_sha256": base.sha(Path(__file__).read_text()),
        },
    )
    return result


def concurrency():
    value = base.read(OUT / "control.json", {}).get("max_parallel", 6)
    if type(value) is not int or not 1 <= value <= 6:
        raise ValueError("ALE max_parallel must be1..6 to reserveCPU")
    return value


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not base.read(OUT / "launch_authorized.json", {}).get("prerequisites_passed"):
        raise RuntimeError("Coordinator authorization required")
    state = {
        "supervisor_pid": os.getpid(),
        "status": "waiting_math50",
        "started_unix": time.time(),
        "active_cells": {},
        "private_active": None,
        "completed_cells": [],
        "failed_cells": {},
        "private_failures": [],
    }
    report()  # Initial pending view only, avoiding concurrent rewrites during Math.
    while not math_barrier()["passed"]:
        barrier = math_barrier()
        state.update(
            status="blocked_math50"
            if barrier["math_status"]
            in {"failed", "partial_failure", "blocked_preflight"}
            else "waiting_math50",
            math_barrier=barrier,
            last_heartbeat_unix=time.time(),
        )
        base.write(OUT / "status.json", state)
        time.sleep(30)
    report()
    checked = preflight()
    if not checked["passed"]:
        state.update(status="blocked_preflight", error=checked["errors"])
        base.write(OUT / "status.json", state)
        return 1
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
            "Math semantic worker still present; refusing duplicate consumer"
        )
    available = sorted(os.sched_getaffinity(0))
    if len(available) >= 16:
        pool, private_cpus = available[-16:-4], available[-4:]
    else:
        count = min(12, max(1, len(available) // 2))
        pool = available[:count]
        private_cpus = available[count : count + PRIVATE_WORKERS] or pool[-1:]
    os.sched_setaffinity(0, sorted(set(pool + private_cpus)))
    state.update(
        public_cpu_pool=pool,
        private_cpu_pool=private_cpus,
        private_workers=min(PRIVATE_WORKERS, len(private_cpus)),
    )
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
            "TRIMEM_EMBEDDING_THREADS",
        ]
    }
    env.update(
        PYTHONUNBUFFERED="1",
        ALE_BENCH_DATA="/localhome/hza214/ale-bench-data",
        ALE_BENCH_NO_CONTAINER_USER="1",
        DOCKER_HOST=f"unix:///run/user/{os.getuid()}/podman/podman.sock",
    )
    env["TRIMEM_ALE_PRIVATE_WORKERS"] = str(min(PRIVATE_WORKERS, len(private_cpus)))
    active = {}
    private = None
    private_queue = []
    worker = None

    def poll_private():
        nonlocal private
        if private and private.poll() is not None:
            if private.returncode:
                state["private_failures"].append(state["private_active"])
            private = None
            state["private_active"] = None
        if private is None and private_queue:
            task, plan, iteration = private_queue.pop(0)
            target = private_path(task, plan, iteration)
            if target.exists():
                if base.read(target, {}).get("status") != "complete":
                    state["private_failures"].append(
                        {
                            "task": task,
                            "plan": plan,
                            "iteration": iteration,
                            "existing_result": str(target),
                        }
                    )
                return
            with (OUT / "private_eval.log").open("a") as log:
                private = subprocess.Popen(
                    [
                        "taskset",
                        "-c",
                        ",".join(map(str, private_cpus)),
                        str(base.ROOT / ".venv-e0/bin/python"),
                        "-m",
                        "trimem.experiments.ale_trajectory_run",
                        "--private",
                        "--task",
                        task,
                        "--plan",
                        plan,
                        "--iteration",
                        str(iteration),
                    ],
                    cwd=base.ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            state["private_active"] = {
                "pid": private.pid,
                "task": task,
                "plan": plan,
                "iteration": iteration,
            }

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
        base.write(
            OUT / "math50_barrier.json", dict(math_barrier(), timestamp=time.time())
        )
        for iteration in CHECKPOINTS:
            pending = []
            for task, plan, name in cells():
                path = stage_path(task, plan, iteration)
                if complete_stage(path, iteration):
                    private_queue.append((task, plan, iteration))
                    continue
                if path.exists() and any(path.iterdir()):
                    raise RuntimeError(
                        f"Incomplete existing output cannot be overwritten: {path}"
                    )
                if iteration > 10 and not complete_stage(
                    stage_path(task, plan, iteration - 10), iteration - 10
                ):
                    raise RuntimeError("Missing predecessor checkpoint")
                pending.append((task, plan, name))
            while pending or active:
                if worker.poll() is not None:
                    raise RuntimeError("Semantic worker exited")
                for name, process in list(active.items()):
                    if process.poll() is None:
                        continue
                    info = state["active_cells"][name]
                    task, plan = info["task"], info["plan"]
                    if process.returncode or not complete_stage(
                        stage_path(task, plan, iteration), iteration
                    ):
                        state["failed_cells"][f"{iteration}:{name}"] = base.read(
                            stage_path(task, plan, iteration) / "run_receipt.json", {}
                        )
                    else:
                        state["completed_cells"].append(f"{iteration}:{name}")
                        private_queue.append((task, plan, iteration))
                    del active[name]
                    del state["active_cells"][name]
                if state["failed_cells"] and not active:
                    raise RuntimeError("Publicstagefailed, laterstagesnotadmitted")
                if not state["failed_cells"]:
                    for task, plan, name in pending[
                        : max(0, concurrency() - len(active))
                    ]:
                        path = stage_path(task, plan, iteration)
                        path.mkdir(parents=True, exist_ok=True)
                        occupied = {v["slot"] for v in state["active_cells"].values()}
                        slot = next(i for i in range(6) if i not in occupied)
                        cpus = sorted(
                            {
                                pool[(2 * slot) % len(pool)],
                                pool[(2 * slot + 1) % len(pool)],
                            }
                        )
                        with (path / "run.log").open("a") as log:
                            process = subprocess.Popen(
                                [
                                    "taskset",
                                    "-c",
                                    ",".join(map(str, cpus)),
                                    *command(task, plan, iteration),
                                ],
                                cwd=base.ROOT,
                                env=env
                                | {
                                    "TRIMEM_ALE_CELL": "1",
                                    "TRIMEM_EXPERIMENT_CELL": str(path),
                                    "TRIMEM_EMBEDDING_DEVICE": f"cuda:{slot % 2}",
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
                            "slot": slot,
                            "iteration_target": iteration,
                            "cpus": cpus,
                            "output": str(path),
                        }
                        base.write(
                            path / "resource_config.json",
                            dict(
                                state["active_cells"][name],
                                execution_mode="parallel",
                                public_workers=1,
                            ),
                        )
                        pending.remove((task, plan, name))
                        base.write(OUT / "status.json", state)
                poll_private()
                state.update(
                    status="running_public",
                    stage=iteration,
                    last_heartbeat_unix=time.time(),
                    private_pending=len(private_queue),
                    pending_cells=[x[2] for x in pending],
                )
                base.write(OUT / "status.json", state)
                report()
                if pending or active:
                    time.sleep(15)
        while private_queue or private:
            poll_private()
            state.update(
                status="finishing_private",
                last_heartbeat_unix=time.time(),
                private_pending=len(private_queue),
            )
            base.write(OUT / "status.json", state)
            report()
            if private_queue or private:
                time.sleep(15)
        state["status"] = (
            "partial_private_failure" if state["private_failures"] else "complete"
        )
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        for process in active.values():
            base.stop_child(process)
        base.stop_child(private)
        raise
    finally:
        base.stop_child(worker)
        state["finished_unix"] = time.time()
        base.write(OUT / "status.json", state)
        report()
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--cell" in argv:
        argv = list(argv)
        argv.remove("--cell")
        return cell_main(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--plan", choices=PLANS)
    parser.add_argument("--iteration", type=int, choices=CHECKPOINTS)
    args = parser.parse_args(argv)
    os.chdir(base.ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.private:
        if not all([args.task, args.plan, args.iteration]):
            parser.error("--private requires task/plan/iteration")
        return private_score(args.task, args.plan, args.iteration)
    if args.preflight:
        result = preflight()
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    if args.report:
        report()
        return 0
    if args.run:
        return run()
    parser.error("Choose a mode")


# Multiprocessing spawn reimports this module; instrument worker prompt assembly too.
if os.getenv("TRIMEM_ALE_CELL") == "1":
    from trimem.experiments import trajectory_cell  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
