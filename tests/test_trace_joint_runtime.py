from concurrent.futures import ThreadPoolExecutor

import pytest

from trimem.experiments.plan_execution_adapter import compile_config
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_schema import JointSpace


def test_execution_capabilities_and_mapping():
    s = JointSpace.live()
    for order, plan in [("before", "P1+Rank"), ("after", "P2+Rank")]:
        cfg = compile_config(s.default(filter_order=order))
        assert cfg.plan == plan and cfg.sim_gate is None
        assert cfg.edges == ("similar", "similar_up")
    with pytest.raises(ValueError):
        compile_config(JointSpace().default(graph_impl="evolutionary"))


def test_atomic_admission_crash_claim_and_manifest(tmp_path):
    path = tmp_path / "ledger.db"
    s = JointSpace.live()
    manifest = {
        "space": s.manifest(),
        "tasks": ["a", "b"],
        "budget": 2,
        "protocol": "test",
        "smoke": True,
    }

    def reserve(_):
        with SearchLedger(path, manifest) as ledger:
            return ledger.reserve(s, [s.default()])

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(len(r) for r in pool.map(reserve, range(4))) == 1
    with SearchLedger(path, manifest) as ledger:
        key = s.key(s.default())
        assert ledger.claim(key, "a")
        assert not ledger.claim(key, "a")
        assert ledger.complete_configs() == []
        # A crash after claim remains uncertain and is never automatically rerun.
        assert ledger.pending()[0]["key"] == key
        ledger.finish(key, "a", {"task_id": "a", "raw_score": -1})
        assert ledger.claim(key, "b")
        ledger.finish(key, "b", {"task_id": "b", "raw_score": 100})
        assert len(ledger.complete_configs()) == 1
        ledger.reserve(s, [s.default(seed_count=32)])
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve(s, [s.default(seed_count=64)])
    with pytest.raises(ValueError, match="manifest"):
        SearchLedger(path, {**manifest, "protocol": "other"})


def test_driver_excludes_concurrent_runner(tmp_path):
    import fcntl

    from trimem.experiments.trace_joint_plan_search import run_round

    with (tmp_path / "driver.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another driver"):
            run_round(tmp_path)


def test_fresh_initialization_then_resume_without_redispatch(tmp_path):
    from trimem.experiments.trace_joint_plan_search import (
        prepare_run,
        run_round,
        source_identity,
    )

    s = JointSpace.live()
    manifest = {
        "space": s.manifest(),
        "tasks": ["0", "1"],
        "protocol": "fresh-smoke",
        "smoke": True,
        "budget": 10,
        "q": 4,
        "task_concurrency": 2,
        "sources": source_identity(),
    }
    prepare_run(tmp_path, manifest)
    calls = []

    def evaluator(c, t, m):
        calls.append((s.key(c), t))
        return {
            "task_id": t,
            "split": "TRAIN",
            "protocol": m["protocol"],
            "synthetic": True,
            "raw_score": -1 if t == "0" else 100,
            "total_tokens": 10,
            "latency_seconds": 1,
        }

    result = run_round(tmp_path, evaluator=evaluator)
    assert result["complete_configs"] == 10 and len(calls) == 20
    assert run_round(tmp_path, evaluator=evaluator)["status"] == "budget_complete"
    assert len(calls) == 20
