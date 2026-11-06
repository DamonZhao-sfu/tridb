import json
import subprocess
from types import SimpleNamespace

import pytest

from tools import trace_joint_live as live
from tools import trace_latent_live as tll
from trimem.experiments import trace_joint_plan_search as driver


def test_backend_patches_restore():
    original = driver._search_worker
    with tll.backend_patches({"search_backend": "latent-mobo-v1"}):
        assert driver._search_worker is not original
    assert driver._search_worker is original
    with pytest.raises(RuntimeError):
        with tll.backend_patches({"search_backend": "latent-mobo-v1"}):
            raise RuntimeError("boom")
    assert driver._search_worker is original
    for m in ({"search_backend": "explicit-pr"}, {"search_backend": "shared-init-v1"}, {}):
        with tll.backend_patches(m):
            assert driver._search_worker is original
        assert driver._search_worker is original


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        with tll.backend_patches({"search_backend": "nope"}):
            pass


def test_interleave_order_deterministic():
    arms = ["a", "b"]
    assert tll.interleave_order(7, 3, arms) == tll.interleave_order(7, 3, arms)
    orders = {tuple(tll.interleave_order(7, k, arms)) for k in range(20)}
    assert len(orders) >= 2
    assert all(sorted(o) == arms for o in orders)
    assert arms == ["a", "b"]


def test_refuses_during_vae_training(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    with pytest.raises(SystemExit, match="train_trace_seq_vae"):
        tll.check_no_vae_training()
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    tll.check_no_vae_training()


def _manifest(backend, tasks=("t1", "t2"), dataset="alf"):
    return {
        "search_backend": backend,
        "fixed": {"dataset": dataset},
        "tasks": list(tasks),
        "budget": 10,
        "new_configs": 4,
    }


def _make_run(tmp_path, name, manifest):
    d = tmp_path / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.setattr(tll, "check_no_vae_training", lambda: None)
    monkeypatch.setattr(driver, "validate_manifest", lambda m: None)
    for name in ("verify_artifacts", "verify_memory", "service_preflight"):
        monkeypatch.setattr(live, name, lambda m: None)
    monkeypatch.setattr(live, "recover_claims", lambda d, m: {})
    monkeypatch.setattr(
        live, "resource_snapshot",
        lambda: {"available_memory_bytes": 10**11, "gpu_utilization": [0, 0]},
    )
    monkeypatch.setattr(live, "eligible_resources", lambda a, g: True)
    monkeypatch.setattr(live, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(
        tll, "_ledger_counts", lambda d, m: {"complete_configs": 0, "pending_episodes": 0}
    )


def test_interleave_runs_until_both_complete(stubbed, tmp_path, monkeypatch):
    a = _make_run(tmp_path, "latent", _manifest("latent-mobo-v1"))
    b = _make_run(tmp_path, "explicit", _manifest("explicit-pr"))
    original = driver._search_worker
    limits = {str(a): 2, str(b): 3}
    calls = {str(a): 0, str(b): 0}
    seen = {}

    def fake_run_round(directory, *, evaluator=None, agent=()):
        key = str(directory)
        calls[key] += 1
        seen.setdefault(key, []).append(driver._search_worker is not original)
        return {"status": "budget_complete" if calls[key] == limits[key] else "round_complete"}

    monkeypatch.setattr(driver, "run_round", fake_run_round)
    tll.run_interleaved([a, b], order_seed=1, sleep=lambda s: None)
    assert calls == limits
    assert seen[str(a)] == [True, True]
    assert seen[str(b)] == [False, False, False]
    for d, n in ((a, 2), (b, 3)):
        recs = [json.loads(x) for x in (d / "interleave_log.jsonl").read_text().splitlines()]
        assert [r["phase"] for r in recs] == ["before", "after"] * n
        assert recs[-1]["status"] == "budget_complete"
        assert json.loads((d / "status.json").read_text())["status"] == "budget_complete"
    assert driver._search_worker is original


def test_interleave_rejects_mismatched_arms(stubbed, tmp_path):
    a = _make_run(tmp_path, "latent", _manifest("latent-mobo-v1"))
    b = _make_run(tmp_path, "explicit", _manifest("explicit-pr", tasks=("t1", "t9")))
    with pytest.raises(ValueError):
        tll.run_interleaved([a, b], order_seed=1, sleep=lambda s: None)
    c = _make_run(tmp_path, "other", _manifest("explicit-pr", dataset="sci"))
    with pytest.raises(ValueError):
        tll.run_interleaved([a, c], order_seed=1, sleep=lambda s: None)


def test_round_exception_marks_failed(stubbed, tmp_path, monkeypatch):
    a = _make_run(tmp_path, "latent", _manifest("latent-mobo-v1"))
    b = _make_run(tmp_path, "explicit", _manifest("explicit-pr"))

    def boom(directory, *, evaluator=None, agent=()):
        raise RuntimeError("kaput")

    monkeypatch.setattr(driver, "run_round", boom)
    with pytest.raises(RuntimeError):
        tll.run_interleaved([a, b], order_seed=1, sleep=lambda s: None)
    failed = [
        d for d in (a, b)
        if (d / "status.json").exists()
        and json.loads((d / "status.json").read_text())["status"] == "failed"
    ]
    assert len(failed) == 1


def _holdout_setup(stubbed, tmp_path, monkeypatch, holdout):
    from tools import trace_joint_protocol as proto

    ma, mb = _manifest("latent-mobo-v1"), _manifest("explicit-pr")
    if holdout:
        ma["holdout"] = {"selection": "train_pareto"}
    a = _make_run(tmp_path, "latent", ma)
    b = _make_run(tmp_path, "explicit", mb)
    events, writes = [], []
    calls = {str(a): 0, str(b): 0}

    def fake_round(directory, *, evaluator=None, agent=()):
        calls[str(directory)] += 1
        done = calls[str(directory)] >= (1 if directory == a else 3)
        events.append(("round", str(directory)))
        return {"status": "budget_complete" if done else "round_complete"}

    def fake_holdout(directory, *, evaluator=None):
        events.append(("holdout", str(directory), evaluator))
        return {"status": "test_complete", "configurations": 3, "episodes": 30}

    real_atomic = driver.atomic_json

    def spy(path, body):
        writes.append((str(path), body["status"]))
        real_atomic(path, body)

    monkeypatch.setattr(driver, "run_round", fake_round)
    monkeypatch.setattr(proto, "run_holdout", fake_holdout)
    monkeypatch.setattr(driver, "atomic_json", spy)
    monkeypatch.setattr(
        tll, "_ledger_counts", lambda d, m: {"complete_configs": 7, "pending_episodes": 0}
    )
    return a, b, events, writes, calls


def test_holdout_runs_after_budget_complete(stubbed, tmp_path, monkeypatch):
    a, b, events, writes, calls = _holdout_setup(stubbed, tmp_path, monkeypatch, True)
    tll.run_interleaved([a, b], order_seed=1, sleep=lambda s: None)
    holdouts = [e for e in events if e[0] == "holdout"]
    assert holdouts == [("holdout", str(a), live.evaluate)]
    assert calls[str(a)] == 1 and calls[str(b)] == 3
    seq = [s for p, s in writes if p == str(a / "status.json") and s != "waiting_resources"]
    assert seq == ["budget_complete", "testing_train_frontier", "test_complete"]
    final = json.loads((a / "status.json").read_text())
    assert final["status"] == "test_complete"
    assert (final["train_configs"], final["test_configs"], final["test_episodes"]) == (7, 3, 30)
    # arm A left active only after holdout returned; arm B kept running afterwards
    idx = events.index(holdouts[0])
    assert ("round", str(b)) in events[idx:] or calls[str(b)] == 3
    assert not any(e == ("round", str(a)) for e in events[idx:])


def test_no_holdout_key_skips_holdout(stubbed, tmp_path, monkeypatch):
    a, b, events, writes, calls = _holdout_setup(stubbed, tmp_path, monkeypatch, False)
    tll.run_interleaved([a, b], order_seed=1, sleep=lambda s: None)
    assert not [e for e in events if e[0] == "holdout"]
    assert json.loads((a / "status.json").read_text())["status"] == "budget_complete"
