"""T8: TEST of a v0.7.0 arm: NoMemory candidate and always tested; pruned / partial excluded; refusals."""

import json

import pytest

from tests import v070_fixtures as fx
from tools import trace_v070_holdout as th
from trimem.experiments import trace_v070_baseline as tb
from trimem.experiments import trace_v070_candidates as tc
from trimem.experiments.trace_v070_baseline import baseline_entry, baseline_key
from trimem.experiments.trace_v070_ledger import SampledLedger
from trimem.experiments.trace_v070_runtime import v070_patches


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return fx.make_shared_init(tmp_path_factory.mktemp("holdout"), size=40)


def evaluator(m):
    return fx.evaluator_for(fx.tiny_space_v070(), baseline_key(m))


def test_shared_init_test_is_deferred(shared):
    m = json.loads((shared / "manifest.json").read_text())
    assert th.run_pruned_holdout_v070(shared, evaluator=evaluator(m)) == {"status": "test_deferred"}


def test_arm_test_with_nomemory(shared, tmp_path):
    arm = fx.make_arm(tmp_path, shared)
    m = json.loads((arm / "manifest.json").read_text())
    summary = th.run_pruned_holdout_v070(arm, evaluator=evaluator(m))
    sel = json.loads((arm / "holdout/selection.json").read_text())
    nm = sel["baselines"]["NoMemory"]
    assert nm["candidate"] and nm["in_pareto"] and nm["tested"] and nm["key"] == baseline_key(m)
    assert nm["key"] in sel["candidate_keys"] and nm["key"] not in sel["selected_keys"]
    assert len(sel["candidate_keys"]) == 11 and sel["excluded_pruned"] == []
    assert summary["status"] == "test_complete" and summary["baselines_tested"] == ["NoMemory"]
    rows = {r["key"]: r for r in summary["results"]}
    assert rows[nm["key"]]["baseline"] == "NoMemory" and len(rows) == len(sel["selected_keys"]) + 1
    e = baseline_entry(arm, "TEST")
    assert e["complete"] and len(e["receipts"]) == 200 and sorted(e["tasks"]) == sorted(m["holdout"]["tasks"])
    assert all(r["split"] == "TEST" for r in e["receipts"])
    assert summary["episodes"] == 200 * (len(sel["selected_keys"]) + 1)


def test_nomemory_forced_when_dominated(shared, tmp_path, monkeypatch):
    arm = fx.make_arm(tmp_path, shared, name="arm_forced")
    m = json.loads((arm / "manifest.json").read_text())
    real = tc.candidates

    def dominated(run, split="TRAIN", **kw):
        rows = real(run, split, **kw)
        for r in rows:
            if r["baseline"] == "NoMemory":
                r.update(raw_score=-1.0, total_tokens=1e9, latency_seconds=1e9)
        return rows

    monkeypatch.setattr(tc, "candidates", dominated)
    sel = th.run_pruned_holdout_v070(arm, evaluator=evaluator(m), select_only=True)["selection"]
    nm = sel["baselines"]["NoMemory"]
    assert nm["candidate"] and not nm["in_pareto"] and nm["tested"] and nm["test_rule"] == "always"
    # fix round 2: NoMemory stays a router / OneConfig candidate even when fuzzy pruning drops it
    assert not nm["in_fuzzy"] and nm["key"] not in sel["fuzzy_keys"] and nm["key"] in sel["router_candidates"]


def _extra_config(arm, m, *, done_half: bool, prune: bool):
    space = fx.tiny_space_v070()
    ev = evaluator(m)
    with v070_patches(m), SampledLedger(arm / "ledger.sqlite", m) as led:
        seen = {k for (k,) in led.db.execute("SELECT key FROM configs")}
        c = next(c for c in fx.enumerate_space(space) if space.key(c) not in seen)
        led.reserve(space, [c])
        key = space.key(c)
        tasks = led.sample_of(key)
        for t in tasks[: len(tasks) // 2] if done_half else []:
            led.claim(key, t)
            led.finish(key, t, ev(c, t, m))
        if prune:
            led.prune_many({key: "test prune"})
    return key


def test_pruned_partial_excluded(shared, tmp_path):
    arm = fx.make_arm(tmp_path, shared, name="arm_pruned")
    m = json.loads((arm / "manifest.json").read_text())
    key = _extra_config(arm, m, done_half=True, prune=True)
    sel = th.run_pruned_holdout_v070(arm, evaluator=evaluator(m), now=True, select_only=True)["selection"]
    assert key in sel["excluded_pruned"] and key not in sel["candidate_keys"] and key in sel["train_keys"]


def test_refusals(shared, tmp_path, monkeypatch):
    arm = fx.make_arm(tmp_path, shared, name="arm_refuse")
    m = json.loads((arm / "manifest.json").read_text())
    monkeypatch.setattr(tb, "baseline_complete", lambda run, split="TRAIN": False)
    with pytest.raises(ValueError, match="NoMemory TRAIN"):
        th.run_pruned_holdout_v070(arm, evaluator=evaluator(m))
    monkeypatch.undo()
    _extra_config(arm, m, done_half=False, prune=False)
    with pytest.raises(ValueError, match="pending"):
        th.run_pruned_holdout_v070(arm, evaluator=evaluator(m))


def _engineer(monkeypatch, arm):
    """Rewrite the TRAIN candidate means: P (Pareto), F (fuzzy-retained, 3-obj dominated by P), X (fuzzy-pruned)."""
    real = tc.candidates
    with SampledLedger(arm / "ledger.sqlite", json.loads((arm / "manifest.json").read_text())) as led:
        keys = [k for (k,) in led.db.execute("SELECT key FROM configs ORDER BY created, key")]
    roles = {keys[0]: (0.9, 1000.0, 10.0), keys[1]: (0.89, 1000.0, 10.5), keys[2]: (0.5, 5000.0, 50.0)}

    def engineered(run, split="TRAIN", **kw):
        rows = real(run, split, **kw)
        if split != "TRAIN":
            return rows
        for r in rows:
            q, tok, lat = (0.05, 800.0, 8.0) if r["baseline"] == "NoMemory" else roles.get(r["key"], (0.1, 9000.0, 90.0))
            r.update(raw_score=q, raw_score_pos=q, total_tokens=tok, latency_seconds=lat)
        return rows

    monkeypatch.setattr(tc, "candidates", engineered)
    return keys[0], keys[1], keys[2]


def test_fuzzy_retained_config_is_tested_and_router_candidate(shared, tmp_path, monkeypatch):
    from tools import trace_v070_router as tr

    arm = fx.make_arm(tmp_path, shared, name="arm_fuzzy")
    m = json.loads((arm / "manifest.json").read_text())
    p, f, x = _engineer(monkeypatch, arm)
    th.run_pruned_holdout_v070(arm, evaluator=evaluator(m))
    sel = json.loads((arm / "holdout/selection.json").read_text())
    nm = baseline_key(m)
    assert p in sel["pareto_keys"] and f not in sel["pareto_keys"] and f in sel["fuzzy_keys"]
    assert x not in sel["fuzzy_keys"] and x not in sel["pareto_keys"]
    assert f in sel["selected_keys"] and x not in sel["selected_keys"]  # fuzzy-retained non-Pareto -> TESTed
    assert set(sel["router_candidates"]) == set(sel["fuzzy_keys"]) | {nm} and nm in sel["fuzzy_keys"]
    assert set(tr.router_candidates(arm)) == set(sel["router_candidates"])  # every one TEST-complete
    keys, _own, _rows = tr.router_rows(arm)
    assert f in keys and x not in keys and nm in keys
    bad = {**sel, "router_candidates": [*sel["router_candidates"], x]}  # a candidate without TEST receipts
    (arm / "holdout/selection.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="without complete TEST receipts"):
        tr.router_candidates(arm)
    bad = {**sel, "router_candidates": [k for k in sel["router_candidates"] if k != nm]}
    (arm / "holdout/selection.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="NoMemory must be a router candidate"):
        tr.router_candidates(arm)


def test_selection_waits_for_the_holdout_lock(shared, tmp_path):
    import fcntl

    arm = fx.make_arm(tmp_path, shared, name="arm_lock")
    m = json.loads((arm / "manifest.json").read_text())
    (arm / "holdout").mkdir()
    with (arm / "holdout/driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            th.run_pruned_holdout_v070(arm, evaluator=evaluator(m), select_only=True)
    assert not (arm / "holdout/selection.json").exists()
