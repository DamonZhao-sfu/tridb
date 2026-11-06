"""NoMemory episodes for TRACE v0.7.0: the historical official AgentGym path, same runtime as TRACE episodes.

``evaluate_nomemory(config, task, manifest)`` mirrors ``tools.trace_joint_live.evaluate`` (read-only import):
32 GiB admission floor, the same ``live.SLOTS`` (same LLM / environment bases), durable attempt identity written
before launch, 900 s timeout + killpg + ``RetryableEpisodeError``, the same environment (os.environ minus
TRIMEM_/POOL_, then ``live["environment"]`` incl. the memory-DB variables, PYTHONPATH, one BLAS thread), the same
agent model, ``--max_tokens`` (4096), ``--max_round`` and ``--action_format``. The command has no plan / memory
arguments: ``tools.trace_v070_nomemory_worker`` (official evaluator + strict patches).

Receipt: raw_score = reward, total_tokens = agent tokens, latency = episode_ms / 1000, retrieval 0, plus the
TRACE identity fields (config_key = the NoMemory baseline key) and ``baseline_worker_sha256``.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from tools import trace_joint_live as live
from trimem.optimizer.joint_search_ledger import (
    EpisodeNotStarted,
    RetryableEpisodeError,
)

ROOT = live.ROOT
WORKER = "tools/trace_v070_nomemory_worker.py"
TIMEOUT_S = 900


def nomemory_receipt_from_files(result, episodes, calls, task) -> dict:
    """``receipt_from_files`` without the memory file (no retrieval)."""
    idx = int(task.rsplit("_", 1)[1])
    rows = [r for r in episodes if r.get("item") == idx]
    calls = [r for r in calls if r.get("item") == idx]
    if result.get("item_id") != task or result.get("reuse") or len(rows) != 1 or not calls:
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
    values = [result.get("reward"), rows[0].get("episode_ms")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Nonfinite/missing raw score or cost")
    if values[1] < 0:
        raise ValueError("Negative measured cost")
    return {
        "raw_score": float(values[0]),
        "total_tokens": tokens,
        "latency_seconds": values[1] / 1000,
        "retrieval_id": None,
        "agent_tokens": tokens,
        "retrieval_tokens": 0,
        "agent_latency_seconds": values[1] / 1000,
        "retrieval_latency_seconds": 0.0,
        "agent_steps": rows[0].get("agent_steps"),
        "quality_source": "environment_raw_score",
        "memory_chars": 0,
        "baseline": "NoMemory",
    }


def nomemory_command(manifest, slot: int, meta: Path, out: Path) -> list[str]:
    lv = manifest["live"]
    rt = live.runtime_identity(manifest)
    return [
        str(ROOT / ".venv-agentgym/bin/python"), "-m", "tools.trace_v070_nomemory_worker",
        "--action_format", rt["action_format"], "--api_key", "EMPTY", "--model", lv["model"],
        "--temperature", "0", "--top_p", "1", "--max_tokens", str(rt["agent_max_tokens"]),
        "--max_round", str(rt["max_round"]), "--task_name", rt["dataset"], "--processes", "1",
        "--data_len", "10000", "--base_url", lv["llm_bases"][slot],
        "--env_server_base", lv["environment_bases"][slot],
        "--inference_file", str(meta / "input.json"), "--output_dir", str(out),
    ]


def nomemory_env(manifest) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TRIMEM_", "POOL_"))}
    env.update(manifest["live"]["environment"])
    env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def receipt_of(out: Path, task: str, identity: dict) -> dict:
    result_path = out / (task + ".json")
    receipt = nomemory_receipt_from_files(json.loads(result_path.read_text()), _lines(out / "episodes.jsonl"),
                                          _lines(out / "llm_calls.jsonl"), task)
    return {**receipt, **identity, "result_file": str(result_path), "result_sha256": live.sha(result_path),
            "baseline_worker_sha256": live.sha(ROOT / WORKER), "synthetic": False}


def episode_dir(manifest, key: str, task: str) -> Path:
    return Path(manifest["live"]["run_dir"]) / "episodes" / key / task


def evaluate_nomemory(config, task, manifest):
    if live.available_memory() < 32 * 1024**3:
        raise EpisodeNotStarted("Host memory admission floor reached; no new episode launched")
    if manifest.get("baseline") != "NoMemory" or dict(config) != {"baseline": "NoMemory"}:
        raise ValueError("evaluate_nomemory runs the NoMemory baseline ledger only")
    split = manifest.get("evaluation_split", "TRAIN")
    if split not in ("TRAIN", "TEST") or task not in manifest["tasks"]:
        raise ValueError("Episode outside the NoMemory evaluation panel")
    runtime = live.runtime_identity(manifest)
    if not task.startswith(runtime["dataset"] + "_"):
        raise ValueError(f"Task {task} is not a {runtime['dataset']} item")
    key = manifest["key"]
    out = episode_dir(manifest, key, task)
    if out.exists():  # the ledger only re-dispatches a FAILED episode; keep its evidence
        n = 1
        while out.with_name(f"{task}.attempt{n}").exists():
            n += 1
        out.rename(out.with_name(f"{task}.attempt{n}"))
    out.mkdir(parents=True)
    meta = out / "attempt"
    meta.mkdir()
    (meta / "input.json").write_text(json.dumps([{"item_id": task}]))
    identity = live.attempt_identity(manifest, key, task, split)
    (meta / "attempt.json").write_text(json.dumps({**identity, "started_at": time.time()}))
    slot = live.SLOTS.get()
    try:
        cmd = nomemory_command(manifest, slot, meta, out)
        with (out / "worker.log").open("w") as log:
            child = subprocess.Popen(cmd, cwd=ROOT, env=nomemory_env(manifest), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                code = child.wait(timeout=TIMEOUT_S)
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
            if not (out / (task + ".json")).exists() and not live.workers_alive(out):
                raise RetryableEpisodeError(f"NoMemory episode {what} without result: {out}/worker.log")
            raise RuntimeError(f"NoMemory episode {what}; result evidence uncertain: {out}")
        return receipt_of(out, task, identity)
    finally:
        live.SLOTS.put(slot)


def recover_nomemory_claims(directory, manifest) -> dict:
    """``tools.trace_joint_live.recover_claims`` for a NoMemory ledger (no memory file, no JointSpace)."""
    from trimem.experiments.trace_joint_plan_search import check_receipt
    from trimem.experiments.trace_v070_baseline import BaselineSpace
    from trimem.optimizer.joint_search_ledger import SearchLedger

    space = BaselineSpace(manifest["space"]["execution_fingerprint"])
    split = manifest.get("evaluation_split", "TRAIN")
    report = {"released": 0, "attached": 0, "retryable": 0, "uncertain": 0}
    with SearchLedger(Path(directory) / "ledger.sqlite", manifest) as ledger:
        for item in ledger.episodes("claimed"):
            task, key = item["task_id"], item["key"]
            out = episode_dir(manifest, key, task)
            attempt_path = out / "attempt" / "attempt.json"
            if not attempt_path.exists():
                ledger.release(key, task)
                report["released"] += 1
                continue
            if live.workers_alive(out):
                raise RuntimeError(f"Worker for {out} is still running; wait for it")
            expected = live.attempt_identity(manifest, key, task, split)
            attempt = json.loads(attempt_path.read_text())
            if {k: attempt.get(k) for k in expected} != expected:
                ledger.fail(key, task, "attempt identity differs from the frozen run")
                report["uncertain"] += 1
                continue
            if not (out / (task + ".json")).exists():
                ledger.fail(key, task, "interrupted; terminated without result", retryable=True)
                report["retryable"] += 1
                continue
            try:
                receipt = {**receipt_of(out, task, expected), "recovered": True}
                check_receipt(receipt, item, manifest, space)
            except (OSError, ValueError, KeyError) as exc:
                ledger.fail(key, task, f"uncertain evidence: {exc!r}")
                report["uncertain"] += 1
                continue
            ledger.finish(key, task, receipt)
            report["attached"] += 1
    return report
