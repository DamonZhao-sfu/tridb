"""Bounded-parallel, resumable-by-completed-cell supervisor for the authorized Math experiment.

Run `--preflight`, then start `--run` in a new OS session only after prerequisites
pass and launch_authorized.json has been written by the coordinating Codex agent.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/trimem_v0.1.0/math_trajectory_v4_20261013"
HTML = ROOT / "triMEM_optimizer/it20_summary/trimem_it20_summary_en.html"
DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_traj"
TASKS = ["heilbronn_convex_13", "second_autocorr_ineq", "heilbronn_triangle"]
PLANS = ["VGR", "SemRankOnly", "P1", "P2"]
START = "<!-- trajectory-v4-math:start -->"
END = "<!-- trajectory-v4-math:end -->"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    temp.replace(path)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return default


def rows(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except ValueError:
            pass  # The only expected malformed line is the live writer's last line.
    return result


def command(task, plan):
    return [
        str(ROOT / ".venv-e0/bin/python"),
        "-m",
        "trimem.experiments.trajectory_cell",
        "--task",
        "math:" + task,
        "--arm",
        "trimem",
        "--task-source",
        "trimem",
        "--trajectory-store",
        "--trimem-dsn",
        DSN,
        "--trajectory-plan",
        plan,
        "--trajectory-memory-tokens",
        "16384",
        "--trajectory-prompt-tokens",
        "32768",
        "--inject-as",
        "code",
        "--num-diverse",
        "4",
        "--injection-policy",
        "fixed",
        "--injection-rate",
        "1.0",
        "--injection-frequency",
        "1.0",
        "--llm-base",
        "http://127.0.0.1:8000/v1",
        "--llm-model",
        "Qwen/Qwen3.8-27B-FP8",
        "--max-iterations",
        "20",
        "--max-tokens",
        "32768",
        "--seed",
        "42",
        "--thinking",
        "false",
        "--llm-telemetry",
        "--eval-timeout",
        "240",
        "--llm-timeout",
        "1800",
        "--checkpoint-interval",
        "10",
        "--out",
        str(OUT / f"{task}__{plan}__s42"),
    ]


def preflight():
    import psycopg
    from psycopg.rows import dict_row

    from bench.agent_memory.gem_oe.run_arm import resolve_task_trimem
    from trimem.trajectory_query import query

    result = {"timestamp": time.time(), "errors": [], "warnings": [], "tasks": {}}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as r:
            models = json.load(r)["data"]
        model = next(m for m in models if m["id"] == "Qwen/Qwen3.8-27B-FP8")
        result["model"] = model
        if model.get("max_model_len", 0) < 65536:
            result["errors"].append(
                "model context window below input32768 + output32768"
            )
    except Exception as e:  # noqa: BLE001 - persist precise preflight blockers
        result["errors"].append(f"model: {e}")
    try:
        with psycopg.connect(DSN, row_factory=dict_row, connect_timeout=5) as c:
            result["attempt_count"] = c.execute(
                "SELECT count(*) AS n FROM attempt"
            ).fetchone()["n"]
            result["load_meta"] = c.execute("SELECT * FROM load_meta").fetchone()
        if (
            result["load_meta"]["code_embedding"]["embedding_version"]
            != "qwen3-full-code-v1"
        ):
            result["errors"].append("wrong embedding contract")
        for task in TASKS:
            t = resolve_task_trimem(DSN, "math:" + task)
            result["tasks"][task] = {
                "seed_sha256": sha(t["seed_code"]),
                "system_sha256": sha(t["spec"]),
                "evaluator": t["evaluator"],
                "evaluator_sha256": hashlib.sha256(
                    Path(t["evaluator"]).read_bytes()
                ).hexdigest(),
            }
    except Exception as e:  # noqa: BLE001 - persist precise preflight blockers
        result["errors"].append(f"database/task: {e}")
    result["available_cpus"] = sorted(os.sched_getaffinity(0))
    result["cpu_affinity_for_experiment"] = result["available_cpus"][-2:]
    result["gpu_inventory"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    processes = subprocess.check_output(
        ["ps", "-eo", "pid,args"], text=True
    ).splitlines()
    result["other_experiments"] = [
        line
        for line in processes
        if "-m bench.agent_memory.gem_oe.run_arm " in line and "bash -" not in line
    ]
    result["workers"] = [
        line
        for line in processes
        if (
            "-m trimem.worker " in line
            or "-m trimem.experiments.trajectory_worker " in line
        )
        and "bash -" not in line
    ]
    result["warnings"].append(
        "Shared GPU model service has existing ALE workloads; latency includes contention."
    )
    result["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "bench/agent_memory/gem_oe/run_arm.py",
            ROOT / "bench/agent_memory/gem_oe/trajectory_prompt.py",
            ROOT / "bench/agent_memory/gem_oe/trajectory_retriever.py",
            ROOT / "trimem/trajectory_query.py",
            Path(__file__),
            ROOT / "trimem/experiments/trajectory_cell.py",
            ROOT / "trimem/experiments/trajectory_worker.py",
        ]
    }
    result["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result["sql"] = {plan: query(plan, "emb") for plan in PLANS}
    result["passed"] = not result["errors"]
    write(OUT / "preflight.json", result)
    write(
        OUT / "manifest.json",
        {
            "version": "math-trajectory-v4-32k-input",
            "dsn": DSN,
            "tasks": TASKS,
            "plans": PLANS,
            "seed": 42,
            "iterations": 20,
            "checkpoints": [10, 20],
            "inspiration_max": 4,
            "inspiration_tokens": 16384,
            "input_tokens": 32768,
            "output_tokens": 32768,
            "top_own_programs": 1,
            "semrank_contract": "score-only, no reranking",
            "construction": "fixed existing offline corpus, not charged to cells",
            "commands": [command(t, p) for t in TASKS for p in PLANS],
        },
    )
    return result


def quality(metrics):
    """Evaluator-success proxy when the Math evaluator exposes no explicit validity."""
    if not isinstance(metrics, dict):
        return None
    if (
        metrics.get("is_valid") is False
        or metrics.get("error")
        or metrics.get("error_signature")
    ):
        return None
    if str(metrics.get("judge_result", "")).lower() in {
        "wa",
        "re",
        "tle",
        "ce",
        "failed",
        "invalid",
    }:
        return None
    value = metrics.get("combined_score")
    return (
        float(value)
        if isinstance(value, (float, int)) and math.isfinite(value)
        else None
    )


def summarize_cell(task, plan, semantic, context_owners=None):
    cell = OUT / f"{task}__{plan}__s42"
    receipt = read(cell / "run_receipt.json", {})
    evolution = rows(cell / "evolution_trace.jsonl")
    retrieval = rows(cell / "trajectory_retrieval.jsonl")
    timing = rows(cell / "retrieval_audit.jsonl")
    generation = rows(cell / "llm_telemetry.jsonl")
    audits = rows(cell / "prompt_context.jsonl")
    dispatches = rows(cell / "generation_prompt_index.jsonl")
    attempts = sorted(
        {int(r["iteration"]) for r in dispatches if r.get("iteration") is not None}
    )
    retrieval_decisions = len({r.get("iteration") for r in timing})
    seed = quality(receipt.get("live_seed_metrics"))
    if seed is None and evolution:
        seed = quality(
            min(evolution, key=lambda r: r["iteration"]).get("parent_metrics")
        )
    curve, best = [], seed
    if seed is not None:
        curve.append({"iteration": 0, "quality": seed})
    by_iteration = {int(r["iteration"]): r for r in evolution}
    for i in attempts:
        if (
            i not in by_iteration
            and i >= max(by_iteration, default=0)
            and receipt.get("status") not in {"complete", "failed", "incomplete"}
        ):
            continue
        score = quality(by_iteration.get(i, {}).get("child_metrics"))
        if score is not None:
            best = score if best is None else max(best, score)
        curve.append({"iteration": i, "quality": best})
    hashes = {r.get("context_sha256") for r in retrieval}
    semantic_rows = [s for s in semantic if s.get("context_sha256") in hashes]

    # Unknown usage on failed attempts stays explicit. Values are observed sums only.
    def total(rs, key):
        vals = [r[key] for r in rs if isinstance(r.get(key), (float, int))]
        return sum(vals) if vals else None

    data = {
        "task": task,
        "plan": plan,
        "status": receipt.get("status", "pending"),
        "attempted_rounds": len(attempts),
        "retrieval_decisions": retrieval_decisions,
        "child_count": len(evolution),
        "seed_quality": seed,
        "curve": curve,
        "quality10": next((v["quality"] for v in curve if v["iteration"] == 10), None),
        "quality20": next((v["quality"] for v in curve if v["iteration"] == 20), None),
        "validity_policy": "finite combined_score, no recorded error/explicit invalid flag; Math evaluators may not expose independent validity",
        "generation_prompt_tokens": total(generation, "prompt_tokens"),
        "generation_completion_tokens": total(generation, "completion_tokens"),
        "generation_usage_unknown_calls": sum(
            r.get("prompt_tokens") is None for r in generation
        ),
        "generation_api_seconds": (total(generation, "llm_api_ms") or 0) / 1000
        if generation
        else None,
        "retrieval_seconds_inclusive": (total(timing, "retrieval_total_ms") or 0) / 1000
        if timing
        else None,
        "embedding_seconds": (total(timing, "embed_ms") or 0) / 1000
        if timing
        else None,
        "evaluation_seconds_children": total(
            [r.get("child_metrics", {}) for r in evolution], "eval_time"
        ),
        "generation_failed_or_invalid_calls": sum(
            r.get("status") != "success" or r.get("diff_valid") is False
            for r in generation
        ),
        "semantic_usage": {},
        "prompt_audits": audits,
        "retrieval_audits": retrieval,
        "full_prompts": rows(cell / "generation_prompt_index.jsonl"),
        "path": str(cell),
        "execution_mode": read(cell / "resource_config.json", {}).get(
            "execution_mode", "unrecorded"
        ),
    }
    for op in ["SemFilter", "SemScore"]:
        matching_rows = [r for r in semantic_rows if r.get("op") == op]
        ambiguous_rows = [
            r
            for r in matching_rows
            if len(
                (context_owners or {}).get((op, r.get("context_sha256")), {cell.name})
            )
            > 1
        ]
        op_rows = [r for r in matching_rows if r not in ambiguous_rows]
        relevant = (op == "SemFilter" and plan in {"P1", "P2"}) or (
            op == "SemScore" and plan == "SemRankOnly"
        )
        data["semantic_usage"][op] = {
            "observed_api_calls": len(op_rows),
            "ambiguous_api_calls_excluded": len(ambiguous_rows),
            "attribution": "partial: shared context across multiple cells, ambiguous calls excluded"
            if ambiguous_rows
            else "unique context SHA",
            "prompt_tokens": total(
                [r.get("usage") or {} for r in op_rows], "prompt_tokens"
            )
            if relevant
            else 0,
            "completion_tokens": total(
                [r.get("usage") or {} for r in op_rows], "completion_tokens"
            )
            if relevant
            else 0,
            "unknown_usage_calls": sum(not r.get("usage") for r in op_rows),
            "note": "exact API usage attributed by full current-context SHA; no API rows may mean cache hit or unavailable, not inferred zero"
            if relevant
            else "operator absent",
        }
    return data


def report():
    semantic = rows(OUT / "semantic_api.jsonl")
    context_owners = {}
    for task, plan, name in matrix_cells():
        op = (
            "SemFilter"
            if plan in {"P1", "P2"}
            else "SemScore"
            if plan == "SemRankOnly"
            else None
        )
        if op:
            for r in rows(OUT / name / "trajectory_retrieval.jsonl"):
                context_owners.setdefault((op, r.get("context_sha256")), set()).add(
                    name
                )
    data = [
        summarize_cell(t, p, semantic, context_owners) for t in TASKS for p in PLANS
    ]
    write(OUT / "summary.json", {"updated_unix": time.time(), "cells": data})
    fields = [
        "task",
        "plan",
        "status",
        "attempted_rounds",
        "child_count",
        "seed_quality",
        "quality10",
        "quality20",
        "generation_prompt_tokens",
        "generation_completion_tokens",
        "generation_api_seconds",
        "retrieval_seconds_inclusive",
        "embedding_seconds",
        "evaluation_seconds_children",
    ]
    with (OUT / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)
    from trimem.experiments.trajectory_dashboard import render

    render(data, HTML, OUT / "historical_best.json")
    return data


def stop_child(p):
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


class AdoptedProcess:
    """Monitor only a recorded, identity-checked session leader after supervisor recovery."""

    def __init__(self, pid, expected, receipt=None):
        self.pid, self.receipt, self.returncode = pid, receipt, None
        proc = Path(f"/proc/{pid}")
        command_line = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if expected not in command_line or os.getsid(pid) != pid:
            raise RuntimeError(f"PID {pid} identity/session check failed")
        self.start = (proc / "stat").read_text().split()[21]

    def poll(self):
        try:
            stat = Path(f"/proc/{self.pid}/stat").read_text().split()
            alive = stat[2] != "Z" and stat[21] == self.start
        except FileNotFoundError:
            alive = False
        if alive:
            return None
        status = read(self.receipt, {}) if self.receipt else {}
        self.returncode = 0 if status.get("status") == "complete" else 1
        return self.returncode

    def wait(self, timeout):
        end = time.monotonic() + timeout
        while self.poll() is None and time.monotonic() < end:
            time.sleep(0.2)
        if self.poll() is None:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return self.returncode


def matrix_cells():
    """Prioritize comparing all four plans for one task before the next task."""
    return [(task, plan, f"{task}__{plan}__s42") for task in TASKS for plan in PLANS]


def is_complete(receipt):
    return (
        receipt.get("status") == "complete"
        and receipt.get("iterations_traced", 0) >= 20
    )


def concurrency_limit(default=6):
    value = read(OUT / "control.json", {}).get("max_parallel", default)
    if type(value) is not int or not 1 <= value <= 12:
        raise ValueError("control.json max_parallel must be an integer from 1 to 12")
    return value


def admission(pending, active, completed, failed, limit):
    """Pure admission rule: no duplicates and never exceed the requested concurrency."""
    slots = max(0, limit - len(active))
    return [
        c
        for c in pending
        if c[2] not in active and c[2] not in completed and c[2] not in failed
    ][:slots]


def adoption_cells(adoption):
    if adoption.get("active_cells"):
        return dict(adoption["active_cells"])
    if adoption.get("cell") and adoption.get("cell_pid"):
        return {
            adoption["cell"]: {
                "pid": adoption["cell_pid"],
                "started_unix": adoption.get("cell_started_unix"),
            }
        }
    return {}


def run(max_parallel=6):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (read(OUT / "launch_authorized.json", {}) or {}).get("prerequisites_passed"):
        raise RuntimeError("Coordinator must confirm prerequisite tests before launch")
    adoption = read(OUT / "adopt.json", {})
    pre = preflight()
    if not pre["passed"]:
        raise RuntimeError(str(pre["errors"]))
    allowed = pre["available_cpus"]
    cpu_pool = allowed[-min(12, max(1, len(allowed) // 2)) :]
    os.sched_setaffinity(0, cpu_pool)
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
    env["PYTHONUNBUFFERED"] = "1"
    state = {
        "supervisor_pid": os.getpid(),
        "status": "starting",
        "started_unix": time.time(),
        "completed_cells": [],
        "failed_cells": {},
        "active_cells": {},
        "max_parallel": concurrency_limit(max_parallel),
        "cpu_pool": cpu_pool,
        "adopted_from_supervisor": adoption.get("supervisor_pid"),
        "scheduling": "task-major, compare plans within task first; existing cells preserved",
        "generation_gpu_routing": "shared vLLM data-parallel service; no per-cell generation binding",
    }
    active = {}
    worker = None
    adopted = adoption_cells(adoption)
    pending = []
    # Classify all cells before starting any process. Unknown leftovers are never overwritten.
    for task, plan, name in matrix_cells():
        cell = OUT / name
        receipt = read(cell / "run_receipt.json", {})
        if is_complete(receipt):
            state["completed_cells"].append(name)
            continue
        if name in adopted:
            info = dict(adopted[name])
            info.setdefault("slot", len(active))
            process = AdoptedProcess(
                info.get("pid", info.get("cell_pid")),
                str(cell),
                cell / "run_receipt.json",
            )
            active[name] = process
            state["active_cells"][name] = dict(
                info, pid=process.pid, task=task, plan=plan, adopted=True
            )
            continue
        if receipt or (cell.exists() and any(cell.iterdir())):
            raise RuntimeError(
                f"Existing incomplete cell {cell} must be explicitly adopted/recovered, never overwritten"
            )
        pending.append((task, plan, name))
    try:
        existing = [w for w in pre["workers"] if "55450" in w]
        if adoption.get("worker_pid"):
            worker = AdoptedProcess(
                adoption["worker_pid"], "-m trimem.experiments.trajectory_worker"
            )
            if any(int(w.split()[0]) != worker.pid for w in existing):
                raise RuntimeError("Unexpected additional worker on isolated DB55450")
        else:
            if existing:
                raise RuntimeError(
                    "Existing worker on DB55450 must be explicitly adopted"
                )
            with (OUT / "semantic_worker.log").open("a") as log:
                worker = subprocess.Popen(
                    [
                        str(ROOT / ".venv-e0/bin/python"),
                        "-m",
                        "trimem.experiments.trajectory_worker",
                        "--dsn",
                        DSN,
                        "--log",
                        str(OUT / "semantic_api.jsonl"),
                        "--concurrency",
                        "8",
                        "--max-take",
                        "8",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        state["worker_pid"] = worker.pid
        write(OUT / "status.json", state)
        if adoption:
            write(OUT / f"adoption_used_{os.getpid()}.json", adoption)
            (OUT / "adopt.json").unlink()
        write(
            OUT / "parallel_configuration.json",
            {
                "timestamp": time.time(),
                "max_parallel_initial": state["max_parallel"],
                "cpu_pool": cpu_pool,
                "queue": matrix_cells(),
                "embedding_device_env": "TRIMEM_EMBEDDING_DEVICE",
                "existing_cells_preserved": True,
            },
        )
        while pending or active:
            if worker.poll() is not None:
                raise RuntimeError("Dedicated semantic worker exited")
            for name, process in list(active.items()):
                if process.poll() is None:
                    continue
                receipt = read(OUT / name / "run_receipt.json", {})
                if process.returncode == 0 and is_complete(receipt):
                    state["completed_cells"].append(name)
                else:
                    state["failed_cells"][name] = {
                        "returncode": process.returncode,
                        "receipt_status": receipt.get("status"),
                        "iterations_traced": receipt.get("iterations_traced"),
                        "error": receipt.get("error"),
                    }
                del active[name]
                del state["active_cells"][name]
            limit = concurrency_limit(max_parallel)
            for task, plan, name in admission(
                pending, active, state["completed_cells"], state["failed_cells"], limit
            ):
                cell = OUT / name
                cell.mkdir(parents=True, exist_ok=True)
                occupied = {v.get("slot") for v in state["active_cells"].values()}
                slot = next(i for i in range(12) if i not in occupied)
                cpus = sorted(
                    {
                        cpu_pool[(slot * 2) % len(cpu_pool)],
                        cpu_pool[(slot * 2 + 1) % len(cpu_pool)],
                    }
                )
                device = f"cuda:{slot % 2}"
                cell_env = env | {
                    "TRIMEM_EXPERIMENT_CELL": str(cell),
                    "TRIMEM_EMBEDDING_DEVICE": device,
                }
                with (cell / "run.log").open("a") as log:
                    process = subprocess.Popen(
                        [
                            "taskset",
                            "-c",
                            ",".join(map(str, cpus)),
                            *command(task, plan),
                        ],
                        cwd=ROOT,
                        env=cell_env,
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
                    "cpus": cpus,
                    "embedding_device": device,
                    "started_unix": time.time(),
                    "adopted": False,
                    "execution_mode": "parallel",
                    "max_parallel_at_admission": limit,
                }
                write(cell / "resource_config.json", state["active_cells"][name])
                pending.remove((task, plan, name))
                # Record the new PID immediately so an unexpected supervisor exit is recoverable.
                write(OUT / "status.json", state)
            summary = report()
            by_name = {f"{r['task']}__{r['plan']}__s42": r for r in summary}
            for name, process in active.items():
                current = by_name[name]
                state["active_cells"][name].update(
                    attempted_rounds=current["attempted_rounds"],
                    child_count=current["child_count"],
                    retrieval_decisions=current["retrieval_decisions"],
                )
                for row in rows(OUT / name / "llm_telemetry.jsonl"):
                    if (row.get("prompt_tokens") or 0) > 32768 or row.get(
                        "thinking_enabled"
                    ) is True:
                        raise RuntimeError(
                            f"Live generation prompt/thinking contract violation: {name}"
                        )
            state.update(
                status="running",
                last_heartbeat_unix=time.time(),
                max_parallel=limit,
                pending_cells=[c[2] for c in pending],
            )
            write(OUT / "status.json", state)
            if pending or active:
                time.sleep(15)
        state["status"] = "partial_failure" if state["failed_cells"] else "complete"
    except BaseException as e:
        state.update(status="failed", error=f"{type(e).__name__}: {e}")
        for process in active.values():
            stop_child(process)
        raise
    finally:
        stop_child(worker)
        state["finished_unix"] = time.time()
        write(OUT / "status.json", state)
        report()


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--max-parallel", type=int, default=6)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.preflight:
        r = preflight()
        print(
            json.dumps(
                {"passed": r["passed"], "errors": r["errors"], "output": str(OUT)}
            )
        )
        return 0 if r["passed"] else 1
    if a.report:
        report()
        return 0
    if a.run:
        run(a.max_parallel)
        return 0
    p.error("Select --preflight, --report, or --run")


if __name__ == "__main__":
    sys.exit(main())
