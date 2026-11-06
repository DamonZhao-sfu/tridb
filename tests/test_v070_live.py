"""Launcher (tools.trace_v070_live) offline: refusal, preflight, NoMemory stage, init loop to test_deferred;
router adapter (tools.trace_v070_router): own-sample rows incl. NoMemory, honest matrix, TEST outcome union."""

import json
from unittest import mock

import numpy as np
import pytest

import tools.prepare_trace_v070_campaign as prep
from tests import v070_fixtures as fx
from tools import trace_joint_live as live
from tools import trace_v070_live as tl
from tools import trace_v070_nomemory as tn
from tools import trace_v070_router as tr
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_v070_baseline import baseline_entry, baseline_key


@pytest.fixture
def offline(monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(live, "ROOT", tmp_path)  # queue lease file lives under the temp root
    monkeypatch.setattr(live, "service_preflight", lambda m: None)
    monkeypatch.setattr(live, "verify_artifacts", lambda m: None)
    monkeypatch.setattr(live, "resource_snapshot", lambda: {"available_memory_bytes": 64 << 30,
                                                            "gpu_utilization": [0, 0]})
    monkeypatch.setattr(tl.tll, "check_no_vae_training", lambda: None)
    monkeypatch.setattr(driver, "initial_advice", fx.advice_stub())


def prepared(tmp_path):
    run = tmp_path / "init"
    m = prep.shared_prepare_run_v070(run, fx.shared_manifest(size=40))
    ev = fx.evaluator_for(fx.tiny_space_v070(), baseline_key(m))
    return run, m, ev


def test_refuses_non_v070(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"protocol": "old"}))
    with pytest.raises(SystemExit, match="not a TRACE v0.7.0"):
        tl.load(tmp_path)


def test_preflight_and_baseline_only(tmp_path, offline, monkeypatch):
    run, m, ev = prepared(tmp_path)
    rep = tl.preflight(run, m)
    assert rep["status"] == "passed" and rep["reserved_configs"] == 5 and rep["nomemory_train"]["episodes"] == 0
    assert isinstance(rep["vae_training_running"], bool)
    monkeypatch.setattr(tn, "evaluate_nomemory", ev)
    monkeypatch.setattr(tn, "recover_nomemory_claims", lambda d, bm: None)
    out = tl.run(run, baseline_only=True, sleep=lambda s: None)
    assert out["status"] == "nomemory_train_complete" and out["nomemory"]["episodes"] == 40
    assert json.loads((run / "status.json").read_text())["status"] == "nomemory_train_complete"
    assert baseline_entry(run, "TRAIN")["complete"]


def test_init_run_to_test_deferred(tmp_path, offline, monkeypatch):
    run, _, ev = prepared(tmp_path)
    monkeypatch.setattr(tn, "evaluate_nomemory", ev)
    monkeypatch.setattr(tn, "recover_nomemory_claims", lambda d, bm: None)
    monkeypatch.setattr(live, "evaluate", ev)
    assert tl.run(run, sleep=lambda s: None) == {"status": "test_deferred"}
    st = json.loads((run / "status.json").read_text())
    assert st["status"] == "test_deferred" and st["train_configs"] == 10
    samples = json.loads((run / "samples.json").read_text())
    assert len(samples["configs"]) == 10 and "NoMemory" in samples["baselines"]


def test_honest_matrix():
    keys = ["a", "b"]
    own = {"a": ["t1", "t2"], "b": ["t2", "t3"]}
    union = ["t1", "t2", "t3"]
    oof = {k: np.full((2, 3), 1.0) for k in keys}
    H = tr.honest_matrix(union, keys, own, oof, lambda k: np.full((3, 3), 2.0))
    assert H[:, 0, 0].tolist() == [1.0, 1.0, 2.0] and H[:, 1, 0].tolist() == [2.0, 1.0, 1.0]


def test_non_owner_never_writes_status(tmp_path, offline):
    import fcntl

    run, _m, _ev = prepared(tmp_path)
    with (run / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another launcher owns the run
        with pytest.raises(SystemExit, match="another launcher owns"):
            tl.run(run, sleep=lambda s: None)
    assert not (run / "status.json").exists()


def test_outcomes_union_nomemory_test(tmp_path):
    arm = tmp_path / "arm"
    calls = []

    def fake(ledger, keys=None):
        calls.append(str(ledger))
        return {"nm": {"t": {"score": 1}}} if "baselines" in str(ledger) else {"bo": {"t": {"score": 0}}}

    (arm / "holdout/baselines/nomemory").mkdir(parents=True)
    (arm / "holdout/baselines/nomemory/ledger.sqlite").write_text("")
    with mock.patch.object(tr.cr, "outcomes", fake):
        _, patched = tr.outcomes_with_nomemory(arm)
        assert set(patched(arm / "holdout/ledger.sqlite")) == {"bo", "nm"}
        assert set(patched(arm / "ledger.sqlite")) == {"bo"}


def test_failure_status_written_while_holding_the_lock(tmp_path, offline, monkeypatch):
    import fcntl

    run, _m, _ev = prepared(tmp_path)
    seen = []
    real = tl._status

    def status(r, s, **kw):
        if s == "stopped":
            probe = (r / "queue.lock").open("a")
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append("lock free")
            except BlockingIOError:
                seen.append("lock held")
            finally:
                probe.close()
        real(r, s, **kw)

    def boom(r, m):
        raise RuntimeError("baseline stage failed")

    monkeypatch.setattr(tl, "_status", status)
    monkeypatch.setattr(tl, "baseline_stage", boom)
    with pytest.raises(RuntimeError, match="baseline stage failed"):
        tl.run(run, sleep=lambda s: None)
    assert seen == ["lock held"] and json.loads((run / "status.json").read_text())["status"] == "stopped"


def test_round_loop_backfills_committed_decisions_first(tmp_path, offline, monkeypatch):
    shared = fx.make_shared_init(tmp_path, size=40)
    arm = fx.make_arm(tmp_path, shared, name="arm_backfill")
    m = json.loads((arm / "manifest.json").read_text())
    from trimem.experiments.trace_v070_ledger import SampledLedger

    with SampledLedger(arm / "ledger.sqlite", m) as led:  # a commit whose log line was lost in a crash
        key = m["initial_keys"][0]
        led.apply_stage_decisions(20, {key: {"stage": 20, "prune": False, "prune_rule": None}}, {})

    def stop(*a, **kw):
        assert (arm / "pruning_log.jsonl").exists(), "round started before the back-fill"
        raise RuntimeError("stop after the check")

    from trimem.experiments.trace_v070_runtime import v070_patches

    monkeypatch.setattr(driver, "run_round", stop)
    with v070_patches(m), pytest.raises(RuntimeError, match="stop after the check"):
        tl.round_loop(arm, m)  # as in tl.run: inside the patches
    lines = [json.loads(x) for x in (arm / "pruning_log.jsonl").read_text().splitlines()]
    assert [(d["key"], d["stage"]) for d in lines] == [(key, 20)]
