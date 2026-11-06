from pathlib import Path
import pytest

from tools.trace_joint_live import eligible_resources, receipt_from_files, runner_config
from trimem.optimizer.plan_schema import JointSpace


def test_live_mapping_retains_selected_threshold_and_no_vector_knob():
    c = JointSpace.live().default(
        filter_order="after", sem_threshold_post=0.8, seed_count=128
    )
    mapped = runner_config(c)
    assert mapped["plan"] == "P2+Rank" and mapped["v2"]["semfilter_threshold"] == 0.8
    assert mapped["memory_config"]["seed_count"] == 128
    assert (
        mapped["v2"]["sim_gate"] is None
        and mapped["v2"]["outcome_policy"] == "success_failure"
    )


def test_receipt_requires_raw_reward_and_complete_cost_evidence():
    result = {"item_id": "sciworld_5", "reward": -100}
    memory = {"rid": "x", "semantic_tokens": 11, "retrieval_ms": 20}
    episodes = [{"item": 5, "generation_tokens": 31, "episode_ms": 40}]
    calls = [{"item": 5, "prompt_tokens": 21, "completion_tokens": 10}]
    receipt = receipt_from_files(result, memory, episodes, calls, "sciworld_5")
    assert receipt["raw_score"] == -100 and receipt["total_tokens"] == 42
    # episode_ms (40 ms) already contains retrieval (20 ms): no double count.
    assert receipt["latency_seconds"] == pytest.approx(0.04)
    with pytest.raises(ValueError):
        receipt_from_files(result, memory, episodes, [], "sciworld_5")
    with pytest.raises(ValueError):
        receipt_from_files(
            result,
            memory,
            episodes,
            [dict(calls[0], completion_tokens=None)],
            "sciworld_5",
        )


def test_queue_waits_for_gpu_and_host_memory():
    assert not eligible_resources(70 * 1024**3, [100, 100])
    assert not eligible_resources(20 * 1024**3, [0, 0])
    assert eligible_resources(70 * 1024**3, [10, 15])


@pytest.mark.parametrize("split", ["TRAIN", "TEST"])
def test_live_dispatch_keeps_test_ids_out_of_training_panel(
    tmp_path, monkeypatch, split
):
    from tools import trace_joint_live as live

    captured = []

    class Child:
        pid = 2**22 + 12345  # nonexistent process group

        def __init__(self, cmd, **kwargs):
            captured.append(cmd)

        def wait(self, timeout=None):
            return 1  # Stop before reading result files; no external services called.

    monkeypatch.setattr(live.subprocess, "Popen", Child)
    monkeypatch.setattr(live, "available_memory", lambda: 64 * 1024**3)
    space = JointSpace.live()
    m = {
        "space": space.manifest(),
        "tasks": ["sciworld_1"],
        "protocol": "fixture",
        "evaluation_split": split,
        "holdout": {"ids_sha256": "test-id-hash"},
        "fixed": {"dataset": "sciworld"},
        "live": {
            "run_dir": str(tmp_path),
            "panel_path": "train.json",
            "panel_sha256": "train-hash",
            "empty_baseline_dir": "empty",
            "model": "fixture",
            "environment": {"TRIMEM_SCIWORLD_MEMORY_DB": "trace_ablation_sci_0123abcd"},
            "llm_bases": ["http://fixture/v1"] * 2,
            "rank_bases": ["http://rank"] * 2,
            "environment_bases": ["http://env"] * 2,
        },
    }
    with pytest.raises(RuntimeError, match="without result"):
        live.evaluate(space.default(), "sciworld_1", m)
    cmd = captured[0]
    assert cmd[cmd.index("--split") + 1] == split.lower()
    assert ("--panel" in cmd) == (split == "TRAIN")
    assert ("/holdout/episodes/" in cmd[-1]) == (split == "TEST")


@pytest.mark.parametrize(
    "dataset,task,max_round",
    [("alfworld", "alfworld_7", 30), ("sciworld", "sciworld_7", 30), ("webshop", "webshop_7", 10)],
)
def test_live_dispatch_passes_dataset_runtime(tmp_path, monkeypatch, dataset, task, max_round):
    from tools import trace_joint_live as live

    captured = []

    class Child:
        pid = 2**22 + 12345

        def __init__(self, cmd, **kwargs):
            captured.append((cmd, kwargs["env"]))

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(live.subprocess, "Popen", Child)
    monkeypatch.setattr(live, "available_memory", lambda: 64 * 1024**3)
    space = JointSpace.live()
    db = "trace_ablation_web_0123abcd" if dataset == "webshop" else "trace_ablation_sci_0123abcd"
    m = {
        "space": space.manifest(),
        "tasks": [task],
        "protocol": "fixture",
        "fixed": {"dataset": dataset},
        "live": {
            "run_dir": str(tmp_path),
            "panel_path": "train.json",
            "panel_sha256": "train-hash",
            "empty_baseline_dir": "empty",
            "model": "fixture",
            "max_round": max_round,
            "action_format": "function_calling" if dataset == "alfworld" else "react",
            "agent_max_tokens": 4096,
            "environment": {live.MEMORY_ENV[dataset]: db},
            "llm_bases": ["http://fixture/v1"] * 2,
            "rank_bases": ["http://rank"] * 2,
            "environment_bases": ["http://env"] * 2,
        },
    }
    with pytest.raises(RuntimeError, match="without result"):
        live.evaluate(space.default(filter_order="none"), task, m)
    cmd, env = captured[0]
    assert cmd[cmd.index("--task_name") + 1] == dataset
    assert cmd[cmd.index("--max_round") + 1] == str(max_round)
    assert cmd[cmd.index("--plan") + 1] == "VGR+Rank"
    assert cmd[cmd.index("--action_format") + 1] == ("function_calling" if dataset == "alfworld" else "react")
    assert cmd[cmd.index("--max_tokens") + 1] == "4096"
    assert env[live.MEMORY_ENV[dataset]] == db
    # Regression: the worker's memory-policy guard must accept the launch layout.
    from trimem.experiments import memory_policy as mp

    mp.claim_dir(Path(cmd[cmd.index("--output_dir") + 1]), "success+failure-v1")
    # A retried episode archives the failed attempt instead of overwriting it.
    with pytest.raises(RuntimeError, match="without result"):
        live.evaluate(space.default(filter_order="none"), task, m)
    out = tmp_path / "episodes" / space.key(space.default(filter_order="none"))
    assert (out / f"{task}.attempt1").is_dir() and (out / task).is_dir()
    with pytest.raises(ValueError, match="not a"):
        live.evaluate(space.default(), "other_7", {**m, "tasks": ["other_7"]})


def test_recover_claims_classifies_by_evidence(tmp_path, monkeypatch):
    """Never launched -> released; terminated without result -> retryable;
    identity mismatch -> uncertain; a live worker -> refuse."""
    import json as _json

    from tools import trace_joint_live as live
    from trimem.experiments.trace_joint_plan_search import prepare_run, source_identity
    from trimem.optimizer.joint_search_ledger import SearchLedger

    space = JointSpace.live()
    m = {
        "space": space.manifest(), "tasks": ["sciworld_1", "sciworld_2", "sciworld_3"],
        "protocol": "recover", "smoke": True, "budget": 14, "task_concurrency": 1,
        "sources": source_identity(), "fixed": {"dataset": "sciworld"},
        "live": {"model": "fixture", "environment": {"TRIMEM_SCIWORLD_MEMORY_DB": "trace_ablation_sci_0123abcd"}},
    }
    frozen = prepare_run(tmp_path, m)
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        key = ledger.pending()[0]["key"]
        for t in m["tasks"]:
            assert ledger.claim(key, t)
    base = tmp_path / "episodes" / key
    good = live.attempt_identity(frozen, key, "sciworld_2", "TRAIN")
    (base / "sciworld_2" / "attempt").mkdir(parents=True)
    (base / "sciworld_2" / "attempt" / "attempt.json").write_text(_json.dumps(good))
    bad = live.attempt_identity(frozen, key, "sciworld_3", "TRAIN")
    bad["runtime"] = {**bad["runtime"], "model": "other"}
    (base / "sciworld_3" / "attempt").mkdir(parents=True)
    (base / "sciworld_3" / "attempt" / "attempt.json").write_text(_json.dumps(bad))
    monkeypatch.setattr(live, "workers_alive", lambda out: [])
    report = live.recover_claims(tmp_path, frozen)
    assert report == {"released": 1, "attached": 0, "retryable": 1, "uncertain": 1}
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        failed = {e["task_id"]: e["retryable"] for e in ledger.episodes("failed")}
    assert failed == {"sciworld_2": True, "sciworld_3": False}
    # A still-running worker blocks recovery entirely.
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        assert ledger.claim(key, "sciworld_1")
    (base / "sciworld_1" / "attempt").mkdir(parents=True)
    (base / "sciworld_1" / "attempt" / "attempt.json").write_text(_json.dumps(good))
    monkeypatch.setattr(live, "workers_alive", lambda out: [123])
    with pytest.raises(RuntimeError, match="still running"):
        live.recover_claims(tmp_path, frozen)


def test_latency_is_end_to_end_episode_time_without_double_count():
    from tools import trace_joint_live as live

    receipt = live.receipt_from_files(
        {"item_id": "sciworld_7", "reward": -100},
        {"semantic_tokens": 50, "retrieval_ms": 40_000, "rid": "r"},
        [{"item": 7, "generation_tokens": 30, "episode_ms": 100_000, "agent_steps": 5}],
        [{"item": 7, "prompt_tokens": 20, "completion_tokens": 10, "status": "ok"}],
        "sciworld_7",
    )
    assert receipt["latency_seconds"] == 100.0  # not 140: retrieval is inside episode_ms
    assert receipt["retrieval_latency_seconds"] == 40.0
    assert receipt["agent_latency_seconds"] == 60.0
    assert receipt["raw_score"] == -100.0
