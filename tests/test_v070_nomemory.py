"""T4: NoMemory receipts, evaluator (fake Popen), command / env parity with TRACE episodes, claim recovery."""

import json
import os
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from tools import trace_joint_live as live
from tools import trace_v070_nomemory as nm
from trimem.experiments.trace_joint_plan_search import check_receipt
from trimem.experiments.trace_v070_baseline import (
    BaselineSpace,
    baseline_dir,
    baseline_manifest,
    prepare_baseline,
)
from trimem.experiments.trace_v070_runtime import space_patches
from trimem.optimizer.joint_search_ledger import RetryableEpisodeError, SearchLedger
from trimem.optimizer.plan_schema_v070 import JointSpaceV070

TASKS = ["alfworld_3", "alfworld_5", "alfworld_8", "alfworld_13"]


def run_manifest(tmp: Path) -> dict:
    space = JointSpaceV070.live(execution_fingerprint="fp-nomemory-test")
    panel = tmp / "train_panel.json"
    panel.write_text(json.dumps({"items": [{"item_id": t, "task_key": f"k{t}"} for t in TASKS]}))
    lv = {"run_dir": str(tmp), "model": "Qwen/Qwen3.8-27B-FP8",
          "llm_bases": ["http://127.0.0.1:8030/v1", "http://127.0.0.1:8031/v1"],
          "rank_bases": ["http://127.0.0.1:8040", "http://127.0.0.1:8041"],
          "environment_bases": ["http://127.0.0.1:36001", "http://127.0.0.1:36011"],
          "environment": {"TRIMEM_ALFWORLD_MEMORY_DB": "trace_ablation_sci_x", "TRIMEM_SEMFILTER_V2": "1"},
          "max_round": 30, "action_format": "function_calling", "agent_max_tokens": 4096,
          "panel_path": str(panel), "panel_sha256": live.sha(panel), "empty_baseline_dir": str(tmp / "empty"),
          "artifacts": {}}
    m = {"protocol": "nm-test-v1", "space": space.manifest(), "live": lv, "fixed": {"dataset": "alfworld"},
         "tasks": TASKS, "holdout": {"ids_sha256": "h"}, "smoke": False, "task_concurrency": 2}
    m["fixed"]["runtime"] = live.runtime_identity(m)
    return m


def evidence(out: Path, task: str, *, tokens=(10, 5), reward=1.0, gen=None):
    idx = int(task.rsplit("_", 1)[1])
    (out / f"{task}.json").write_text(json.dumps({"item_id": task, "reward": reward, "conversations": []}))
    calls = [{"item": idx, "prompt_tokens": tokens[0], "completion_tokens": tokens[1], "status": "ok"}] * 2
    (out / "llm_calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    total = 2 * sum(tokens) if gen is None else gen
    (out / "episodes.jsonl").write_text(json.dumps({"item": idx, "episode_ms": 2500.0, "generation_tokens": total,
                                                    "agent_steps": 4}) + "\n")


def read(out: Path, task: str):
    lines = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return json.loads((out / f"{task}.json").read_text()), lines(out / "episodes.jsonl"), lines(
        out / "llm_calls.jsonl")


def test_receipt_builder(tmp_path):
    evidence(tmp_path, "alfworld_3")
    r = nm.nomemory_receipt_from_files(*read(tmp_path, "alfworld_3"), "alfworld_3")
    assert (r["raw_score"], r["total_tokens"], r["latency_seconds"]) == (1.0, 30, 2.5)
    assert r["retrieval_tokens"] == 0 and r["retrieval_latency_seconds"] == 0.0 and r["retrieval_id"] is None
    assert r["agent_steps"] == 4 and r["memory_chars"] == 0
    with pytest.raises(ValueError):
        nm.nomemory_receipt_from_files(*read(tmp_path, "alfworld_3"), "alfworld_5")
    evidence(tmp_path, "alfworld_3", gen=99)
    with pytest.raises(ValueError, match="disagree"):
        nm.nomemory_receipt_from_files(*read(tmp_path, "alfworld_3"), "alfworld_3")
    evidence(tmp_path, "alfworld_3", reward=float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        nm.nomemory_receipt_from_files(*read(tmp_path, "alfworld_3"), "alfworld_3")


class FakePopen:
    seen: ClassVar[list] = []
    write = True
    code = 0

    def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
        FakePopen.seen.append((cmd, env))
        self.pid = -12345
        out = Path(cmd[cmd.index("--output_dir") + 1])
        task = json.loads(Path(cmd[cmd.index("--inference_file") + 1]).read_text())[0]["item_id"]
        if FakePopen.write:
            evidence(out, task)

    def wait(self, timeout=None):
        return FakePopen.code


@pytest.fixture
def fake(monkeypatch):
    FakePopen.seen, FakePopen.write, FakePopen.code = [], True, 0
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    monkeypatch.setattr(os, "killpg", lambda *a: None)
    monkeypatch.setattr(live, "available_memory", lambda: 64 * 1024**3)
    monkeypatch.setenv("TRIMEM_LEAK", "x")
    monkeypatch.setenv("POOL_LEAK", "y")
    return FakePopen


def test_evaluate_nomemory_passes_check_receipt(tmp_path, fake):
    m = run_manifest(tmp_path)
    bm = prepare_baseline(tmp_path, m, split="TRAIN", tasks=TASKS)
    receipt = nm.evaluate_nomemory({"baseline": "NoMemory"}, "alfworld_5", bm)
    check_receipt(receipt, {"key": bm["key"], "task_id": "alfworld_5"}, bm, BaselineSpace("fp-nomemory-test"))
    assert receipt["config_key"] == bm["key"] and receipt["baseline_worker_sha256"] == live.sha(live.ROOT / nm.WORKER)
    assert Path(receipt["result_file"]).parent == baseline_dir(tmp_path) / "episodes" / bm["key"] / "alfworld_5"
    with pytest.raises(ValueError):
        nm.evaluate_nomemory({"baseline": "NoMemory"}, "alfworld_99", bm)


def test_command_and_env_parity_with_trace_episode(tmp_path, fake):
    m = run_manifest(tmp_path)
    bm = baseline_manifest(m, split="TRAIN", tasks=TASKS, run_dir=tmp_path)
    nm.evaluate_nomemory({"baseline": "NoMemory"}, "alfworld_3", bm)
    fake.write, fake.code = False, 1
    with space_patches(), pytest.raises(RetryableEpisodeError):
        live.evaluate(JointSpaceV070.live().default(rank_impl="none", filter_order="none"), "alfworld_3", m)
    (ncmd, nenv), (tcmd, tenv) = fake.seen
    assert ncmd[2] == "tools.trace_v070_nomemory_worker" and "--plan" not in ncmd and "--space_v2" not in ncmd
    assert tcmd[tcmd.index("--plan") + 1] == "VGR"
    for flag in ("--action_format", "--api_key", "--model", "--temperature", "--top_p", "--max_tokens",
                 "--max_round", "--task_name", "--processes", "--data_len"):
        assert ncmd[ncmd.index(flag) + 1] == tcmd[tcmd.index(flag) + 1], flag
    assert ncmd[ncmd.index("--max_tokens") + 1] == "4096"
    for cmd in (ncmd, tcmd):  # LLM base and env server of one slot
        slot = m["live"]["llm_bases"].index(cmd[cmd.index("--base_url") + 1])
        assert cmd[cmd.index("--env_server_base") + 1] == m["live"]["environment_bases"][slot]
    assert nenv == tenv and nenv["TRIMEM_ALFWORLD_MEMORY_DB"] == "trace_ablation_sci_x"
    assert "TRIMEM_LEAK" not in nenv and "POOL_LEAK" not in nenv and nenv["OMP_NUM_THREADS"] == "1"


def test_recovery_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "workers_alive", lambda out: [])
    m = run_manifest(tmp_path)
    bm = prepare_baseline(tmp_path, m, split="TRAIN", tasks=TASKS)
    key = bm["key"]
    with SearchLedger(baseline_dir(tmp_path) / "ledger.sqlite", bm) as led:
        for t in TASKS:
            assert led.claim(key, t)
    ident = live.attempt_identity(bm, key, "x", "TRAIN")
    for t, kind in zip(TASKS, ("none", "mismatch", "no-result", "complete")):
        out = nm.episode_dir(bm, key, t)
        if kind == "none":
            continue
        (out / "attempt").mkdir(parents=True)
        body = {**ident, "task_id": t} if kind != "mismatch" else {**ident, "task_id": t, "protocol": "other"}
        (out / "attempt" / "attempt.json").write_text(json.dumps(body))
        if kind == "complete":
            evidence(out, t)
    report = nm.recover_nomemory_claims(baseline_dir(tmp_path), bm)
    assert report == {"released": 1, "attached": 1, "retryable": 1, "uncertain": 1}
    with SearchLedger(baseline_dir(tmp_path) / "ledger.sqlite", bm) as led:
        states = dict(led.db.execute("SELECT task, state FROM episodes"))
        retry = dict(led.db.execute("SELECT task, retryable FROM episodes"))
    assert states == {"alfworld_3": "ready", "alfworld_5": "failed", "alfworld_8": "failed", "alfworld_13": "done"}
    assert retry["alfworld_8"] == 1 and retry["alfworld_5"] == 0
