"""Real AgentGym evaluator (ALFWorld / ScienceWorld / WebShop) and resource-gated TRACE PR queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import signal
import sqlite3
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from trimem.optimizer.joint_search_ledger import EpisodeNotStarted, RetryableEpisodeError
from trimem.optimizer.plan_schema import JointSpace

ROOT = Path(__file__).resolve().parents[1]
SLOTS = queue.Queue()
for _ in range(8):
    SLOTS.put(0)
    SLOTS.put(1)


def runner_config(config):
    from trimem.experiments.plan_execution_adapter import compile_config

    c = JointSpace.live().canonicalize(config)
    legacy = compile_config(c)
    return {
        "plan": legacy.plan,
        "edges": "+".join(legacy.edges),  # from graph_direction
        "memory_config": {
            "seed_count": legacy.seed_count,
            "depth": legacy.depth,
            "fanout": legacy.fanout,
            "top_k": legacy.top_k,
        },
        "v2": {
            "sim_gate": None,
            "max_failed": None,
            "memory_form": "raw",
            "memory_forms_version": "memory-forms-v1",
            "outcome_policy": "success_only"
            if c["trajectory_category"] == "Success"
            else "success_failure",
            # VGR+Rank (filter_order none) has no SemFilter, so no threshold.
            "semfilter_threshold": {
                "none": None,
                "before": c["sem_threshold_pre"],
                "after": c["sem_threshold_post"],
            }[c["filter_order"]],
            "pool_cap": 256,
            "expand_limit": 4096,
            # Engine extension v0.1.0: read by agentgym_session_eval.retrieve_v2.
            "graph_impl": c["graph_impl"],
            "beam_width": c["beam_width"],
            "rank_impl": c["rank_impl"],
            "retrieval_unit": c["retrieval_unit"],
            "injection_timing": c["injection_timing"],
            "retrieval_interval": c["retrieval_interval"],
        },
    }


def receipt_from_files(result, memory, episodes, calls, task):
    idx = int(task.rsplit("_", 1)[1])
    rows = [r for r in episodes if r.get("item") == idx]
    calls = [r for r in calls if r.get("item") == idx]
    if (
        result.get("item_id") != task
        or result.get("reuse")
        or len(rows) != 1
        or not calls
    ):
        raise ValueError("Incomplete, duplicated or reused episode evidence")
    tokens = 0
    for call in calls:
        if call.get("status", "ok") != "ok":
            raise ValueError("Failed agent API call")
        for key in ("prompt_tokens", "completion_tokens"):
            if type(call.get(key)) is not int or call[key] < 0:
                raise ValueError("Incomplete generation token evidence")
            tokens += call[key]
    if tokens != rows[0].get("generation_tokens"):
        raise ValueError("Generation token ledgers disagree")
    values = [
        result.get("reward"),
        memory.get("semantic_tokens"),
        memory.get("retrieval_ms"),
        rows[0].get("episode_ms"),
    ]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError("Nonfinite/missing raw score or cost")
    if any(v < 0 for v in values[1:]):
        raise ValueError("Negative measured cost")
    # episodes.jsonl episode_ms is timed by agentgym_official_eval.process_logged
    # around the whole item, i.e. retrieval + agent episode. It is the end-to-end
    # latency already; adding retrieval_ms again would count retrieval twice.
    if values[3] < values[2]:
        raise ValueError("Episode wall time shorter than its retrieval")
    return {
        "raw_score": float(values[0]),
        "total_tokens": tokens + values[1],
        "latency_seconds": values[3] / 1000,
        "retrieval_id": memory["rid"],
        "agent_tokens": tokens,
        "retrieval_tokens": values[1],
        "agent_latency_seconds": (values[3] - values[2]) / 1000,
        "retrieval_latency_seconds": values[2] / 1000,
        "agent_steps": rows[0].get("agent_steps"),
        "quality_source": "environment_raw_score",
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DATASETS = ("alfworld", "sciworld", "webshop")
MEMORY_ENV = {
    "alfworld": "TRIMEM_ALFWORLD_MEMORY_DB",
    "sciworld": "TRIMEM_SCIWORLD_MEMORY_DB",
    "webshop": "TRIMEM_WEBSHOP_MEMORY_DB",
}
WORKER_SOURCES = (
    "tools/trace_joint_episode_worker.py",
    "trimem/experiments/agentgym_session_eval.py",
    "trimem/experiments/agentgym_official_eval.py",
    "trimem/experiments/agentenv_task_adapter.py",
)


def workers_alive(out):
    """Processes whose command line names this episode directory (the worker
    and its forked children all carry --output_dir <out>)."""
    needle = str(out).encode()
    alive = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if needle in (proc / "cmdline").read_bytes():
                alive.append(int(proc.name))
        except OSError:
            continue
    return alive


def memory_writes(database):
    """Row writes recorded in a memory clone (non-temporary tables). A frozen
    clone must keep this value; any write or a statistics reset changes it."""
    import psycopg

    dsn = f"host=/tmp/trimem-traj port=55450 dbname={database}"
    with psycopg.connect(dsn) as conn:
        read_only = conn.execute("SHOW default_transaction_read_only").fetchone()[0]
        writes = conn.execute(
            "SELECT coalesce(sum(n_tup_ins+n_tup_upd+n_tup_del),0) FROM pg_stat_user_tables "
            "WHERE schemaname NOT LIKE 'pg_temp%%'"
        ).fetchone()[0]
    if read_only != "on":
        raise ValueError(f"Memory clone {database} is not read-only by default")
    return int(writes)


def verify_memory(manifest):
    fixed = manifest["fixed"]
    if "memory_writes" in fixed and memory_writes(fixed["memory_database"]) != fixed["memory_writes"]:
        raise ValueError(f"Frozen memory clone {fixed['memory_database']} was written")


def attempt_identity(manifest, key, task, split):
    space_fp = manifest["space"]["execution_fingerprint"]
    return {
        "task_id": task,
        "split": split,
        "protocol": manifest["protocol"],
        "config_key": key,
        "execution_fingerprint": space_fp,
        "runtime": runtime_identity(manifest),
    }


def runtime_identity(manifest):
    """What one episode actually runs with; bound into every receipt.

    The evaluator sources are hashed when the episode starts, so a code change
    after preparation cannot enter the archive under the frozen fingerprint.
    """
    live = manifest["live"]
    dataset = manifest["fixed"]["dataset"]
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    return {
        "dataset": dataset,
        "model": live["model"],
        "max_round": int(live.get("max_round", 30)),
        # Historical TRACE runs: ALFWorld function_calling, ScienceWorld/WebShop react;
        # agent max_tokens 4096 (controlled-c16 profile). Missing = old campaign default.
        "action_format": live.get("action_format", "function_calling"),
        "agent_max_tokens": int(live.get("agent_max_tokens", 8192)),
        "memory_database": live["environment"][MEMORY_ENV[dataset]],
        "evaluator_sources": {p: sha(ROOT / p) for p in WORKER_SOURCES},
    }


def verify_artifacts(manifest):
    for path, expected in manifest["live"]["artifacts"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen live artifact changed: {path}")


def evaluate(config, task, manifest):
    if available_memory() < 32 * 1024**3:
        raise EpisodeNotStarted(
            "Host memory admission floor reached; no new episode launched"
        )
    live = manifest["live"]
    space = JointSpace(
        manifest["space"]["domains"],
        execution_fingerprint=manifest["space"]["execution_fingerprint"],
    )
    key = space.key(config)
    split = manifest.get("evaluation_split", "TRAIN")
    if split not in ("TRAIN", "TEST") or task not in manifest["tasks"]:
        raise ValueError("Episode outside the frozen evaluation panel")
    runtime = runtime_identity(manifest)
    dataset = runtime["dataset"]
    if not task.startswith(dataset + "_"):
        raise ValueError(f"Task {task} is not a {dataset} item")
    root = Path(live["run_dir"])
    out = (root / "holdout" if split == "TEST" else root) / "episodes" / key / task
    if out.exists():
        # The ledger only re-dispatches a FAILED episode; keep its evidence.
        n = 1
        while out.with_name(f"{task}.attempt{n}").exists():
            n += 1
        out.rename(out.with_name(f"{task}.attempt{n}"))
    out.mkdir(parents=True)
    result_path = out / (task + ".json")
    # Launcher files live in a subdirectory: the worker's memory-policy guard
    # refuses an output directory that already holds top-level *.json files.
    meta = out / "attempt"
    meta.mkdir()
    (meta / "input.json").write_text(json.dumps([{"item_id": task}]))
    # Durable identity of this attempt, written before launch; recovery checks it.
    identity = attempt_identity(manifest, key, task, split)
    (meta / "attempt.json").write_text(json.dumps({**identity, "started_at": time.time()}))
    slot = SLOTS.get()
    try:
        cfg = runner_config(config)
        if live.get("whole_card_cap") == "none":  # amendment 2026-11-03: uncapped whole cards
            cfg["v2"]["whole_card_cap"] = "none"
        base = live["llm_bases"][slot]
        cmd = [
            str(ROOT / ".venv-agentgym/bin/python"),
            "-m",
            "tools.trace_joint_episode_worker",
            "--plan",
            cfg["plan"],
            "--edges",
            cfg["edges"],
            "--memory_config",
            json.dumps(cfg["memory_config"]),
            "--space_v2",
            json.dumps(cfg["v2"]),
            "--split",
            split.lower(),
            *(["--panel", live["panel_path"]] if split == "TRAIN" else []),
            "--baseline_dir",
            live["empty_baseline_dir"],
            "--experiment",
            manifest["protocol"],
            "--semantic_base",
            base.removesuffix("/v1"),
            "--rank_base",
            live["rank_bases"][slot],
            "--memory_policy",
            "success+failure-v1",
            "--protocol_extra",
            json.dumps(
                {
                    "panel_hash": live["panel_sha256"]
                    if split == "TRAIN"
                    else manifest["holdout"]["ids_sha256"],
                    "joint_config": config,
                    "joint_key": key,
                    "quality_source": "environment_raw_score",
                    "concurrency": 16,
                }
            ),
            "--action_format",
            runtime["action_format"],
            "--api_key",
            "EMPTY",
            "--model",
            live["model"],
            "--temperature",
            "0",
            "--top_p",
            "1",
            "--max_tokens",
            str(runtime["agent_max_tokens"]),
            "--max_round",
            str(runtime["max_round"]),
            "--task_name",
            dataset,
            "--processes",
            "1",
            "--data_len",
            "10000",
            "--base_url",
            base,
            "--env_server_base",
            live["environment_bases"][slot],
            "--inference_file",
            str(meta / "input.json"),
            "--output_dir",
            str(out),
        ]
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("TRIMEM_", "POOL_"))
        }
        env.update(live["environment"])
        env.update(
            PYTHONPATH=str(ROOT),
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with (out / "worker.log").open("w") as log:
            child = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                code = child.wait(timeout=900)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(15)
                except subprocess.TimeoutExpired:
                    pass
                code = None
            try:  # no member of the worker's group may outlive the attempt
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if timed_out or code != 0:
            what = "timeout" if timed_out else f"exit {code}"
            if not result_path.exists() and not workers_alive(out):
                raise RetryableEpisodeError(f"Episode {what} without result: {out}/worker.log")
            raise RuntimeError(f"Episode {what}; result evidence uncertain: {out}")

        def lines(path):
            return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

        receipt = receipt_from_files(
            json.loads(result_path.read_text()),
            json.loads(
                (out / "memory" / (task.rsplit("_", 1)[1] + ".json")).read_text()
            ),
            lines(out / "episodes.jsonl"),
            lines(out / "llm_calls.jsonl"),
            task,
        )
        return {
            **receipt,
            "task_id": task,
            "split": split,
            "protocol": manifest["protocol"],
            "config_key": key,
            "execution_fingerprint": space.execution_fingerprint,
            "result_file": str(result_path),
            "result_sha256": sha(result_path),
            "runtime": identity["runtime"],
            "synthetic": False,
        }
    finally:
        SLOTS.put(slot)


def recover_claims(directory, manifest):
    """Resolve claims left by a crashed driver before any new dispatch.

    - No attempt.json: the worker was never launched; release the claim.
    - A process still names the episode directory: refuse to continue.
    - The attempt identity differs from the frozen one: uncertain, failed.
    - Complete, valid evidence: attach it as the receipt.
    - Terminated without a result: failed, retryable. Otherwise uncertain.
    """
    from trimem.experiments.trace_joint_plan_search import check_receipt, get_space
    from trimem.optimizer.joint_search_ledger import SearchLedger

    directory = Path(directory)
    space = get_space(manifest)
    split = manifest.get("evaluation_split", "TRAIN")
    report = {"released": 0, "attached": 0, "retryable": 0, "uncertain": 0}
    with SearchLedger(directory / "ledger.sqlite", manifest) as ledger:
        for item in ledger.episodes("claimed"):
            task, key = item["task_id"], item["key"]
            out = directory / "episodes" / key / task
            attempt_path = out / "attempt" / "attempt.json"
            if not attempt_path.exists():
                ledger.release(key, task)
                report["released"] += 1
                continue
            if workers_alive(out):
                raise RuntimeError(f"Worker for {out} is still running; wait for it")
            attempt = json.loads(attempt_path.read_text())
            expected = attempt_identity(manifest, key, task, split)
            if {k: attempt.get(k) for k in expected} != expected:
                ledger.fail(key, task, "attempt identity differs from the frozen run")
                report["uncertain"] += 1
                continue
            result_path = out / (task + ".json")
            if not result_path.exists():
                ledger.fail(key, task, "interrupted; terminated without result", retryable=True)
                report["retryable"] += 1
                continue
            try:
                lines = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
                receipt = {
                    **receipt_from_files(
                        json.loads(result_path.read_text()),
                        json.loads((out / "memory" / (task.rsplit("_", 1)[1] + ".json")).read_text()),
                        lines(out / "episodes.jsonl"),
                        lines(out / "llm_calls.jsonl"),
                        task,
                    ),
                    **expected,
                    "result_file": str(result_path),
                    "result_sha256": sha(result_path),
                    "synthetic": False,
                    "recovered": True,
                }
                check_receipt(receipt, item, manifest, space)
            except (OSError, ValueError, KeyError) as exc:
                ledger.fail(key, task, f"uncertain evidence: {exc!r}")
                report["uncertain"] += 1
                continue
            ledger.finish(key, task, receipt)
            report["attached"] += 1
    return report


def eligible_resources(available, gpu_util):
    return available >= 32 * 1024**3 and len(gpu_util) == 2 and max(gpu_util) <= 20


def available_memory():
    mem = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    return mem["MemAvailable"]


def resource_snapshot():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {
        "available_memory_bytes": available_memory(),
        "gpu_utilization": [int(v) for v in raw.split()],
    }


def get_json(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as r:
        return json.load(r)


def service_preflight(manifest):
    live = manifest["live"]
    for base in live["llm_bases"]:
        models = get_json(base + "/models")
        if live["model"] not in [m["id"] for m in models["data"]]:
            raise ValueError("Actor model identity changed")
    for base in live["environment_bases"]:
        get_json(base + "/managed_stats")
    for base in live["rank_bases"]:
        get_json(base + "/health")
    embed = get_json("http://127.0.0.1:8021/health")
    if embed.get("model") != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError("Embedding identity mismatch")


def main():
    from trimem.experiments.trace_joint_plan_search import (
        atomic_json,
        run_round,
        validate_manifest,
    )
    from trimem.optimizer.joint_search_ledger import SearchLedger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    m = json.loads((args.run_dir / "manifest.json").read_text())
    validate_manifest(m)
    verify_artifacts(m)
    service_preflight(m)
    if args.preflight_only:
        print(json.dumps({"status": "passed", "resources": resource_snapshot()}))
        return
    status_path = args.run_dir / "status.json"
    stop = threading.Event()

    def progress():
        while not stop.wait(60):
            try:
                with sqlite3.connect(args.run_dir / "ledger.sqlite", timeout=5) as db:
                    counts = dict(
                        db.execute("SELECT state,count(*) FROM episodes GROUP BY state")
                    )
                    completed = db.execute(
                        "SELECT count(*) FROM configs c WHERE NOT EXISTS "
                        "(SELECT 1 FROM episodes e WHERE e.config=c.key AND e.state!='done')"
                    ).fetchone()[0]
                atomic_json(
                    args.run_dir / "progress.json",
                    {
                        "at": time.time(),
                        "episodes": counts,
                        "complete_configs": completed,
                        "total_config_budget": m["budget"],
                        "total_episode_budget": m["budget"] * len(m["tasks"]),
                    },
                )
            except (sqlite3.Error, OSError):
                pass

    threading.Thread(target=progress, daemon=True).start()
    try:
        with (
            (args.run_dir / "queue.lock").open("a") as own,
            (ROOT / "results/pool_evaluation_queue.lock").open("a") as lease,
        ):
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stable = 0
            while True:
                resources = resource_snapshot()
                try:
                    # Shared, like the other evaluation queues; the idle-GPU gate prevents co-running.
                    fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    acquired = False
                ready = acquired and eligible_resources(
                    resources["available_memory_bytes"], resources["gpu_utilization"]
                )
                stable = stable + 1 if ready else 0
                if stable >= 3:
                    break
                if acquired:
                    fcntl.flock(lease, fcntl.LOCK_UN)
                atomic_json(
                    status_path,
                    {
                        "status": "waiting_resources",
                        "at": time.time(),
                        **resources,
                        "evaluation_lease_available": acquired,
                        "initial_configs": m.get("initial_configs", 10),
                        "new_configs": m["new_configs"],
                    },
                )
                time.sleep(30)
            atomic_json(
                status_path,
                {
                    "status": "running",
                    "at": time.time(),
                    "initial_configs": m.get("initial_configs", 10),
                    "new_configs": m["new_configs"],
                },
            )
            while True:
                verify_artifacts(m)
                verify_memory(m)
                service_preflight(m)
                recover_claims(args.run_dir, m)
                result = run_round(args.run_dir, evaluator=evaluate)
                with SearchLedger(args.run_dir / "ledger.sqlite", m) as ledger:
                    complete = ledger.complete_configs()
                    pending = ledger.pending()
                atomic_json(
                    status_path,
                    {
                        "status": result["status"],
                        "at": time.time(),
                        "complete_configs": len(complete),
                        "pending_episodes": len(pending),
                        "initial_configs": m.get("initial_configs", 10),
                        "new_configs": m["new_configs"],
                    },
                )
                if result["status"] == "budget_complete":
                    if m.get("holdout"):
                        from tools.trace_joint_protocol import run_holdout

                        atomic_json(
                            status_path,
                            {"status": "testing_train_frontier", "at": time.time()},
                        )
                        test_result = run_holdout(args.run_dir, evaluator=evaluate)
                        atomic_json(
                            status_path,
                            {
                                "status": test_result["status"],
                                "at": time.time(),
                                "train_configs": len(complete),
                                "test_configs": test_result.get("configurations", 0),
                                "test_episodes": test_result.get("episodes", 0),
                            },
                        )
                    return
                if result["status"] == "pending":
                    raise RuntimeError(
                        "Episodes still pending after bounded retries; inspect, then use the retry action"
                    )
    except BaseException as exc:
        atomic_json(
            status_path, {"status": "stopped", "at": time.time(), "error": repr(exc)}
        )
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    main()
