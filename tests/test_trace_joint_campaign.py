from trimem.experiments.trace_joint_plan_search import (
    prepare_run,
    source_identity,
    validate_manifest,
)
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_schema import JointSpace


def test_twenty_initial_and_fifty_total_are_accepted(tmp_path):
    space = JointSpace.live()
    manifest = {
        "space": space.manifest(),
        "tasks": [str(i) for i in range(100)],
        "protocol": "scienceworld-pr-20-plus-30",
        "smoke": False,
        "initial_configs": 20,
        "new_configs": 30,
        "budget": 50,
        "q": 4,
        "task_concurrency": 16,
        "sources": source_identity(),
    }
    validate_manifest(manifest)
    frozen = prepare_run(tmp_path, manifest)
    assert len(frozen["initial_keys"]) == 20
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        assert len(ledger.pending()) == 2000


def test_final_batch_is_two():
    from trimem.experiments.trace_joint_plan_search import next_batch_size

    m = {"initial_configs": 20, "new_configs": 30, "budget": 50, "q": 4}
    assert [next_batch_size(m, n) for n in (20, 24, 44, 48, 50)] == [4, 4, 4, 2, 0]


def test_failed_episode_stops_further_admission(tmp_path):
    import pytest

    from trimem.experiments.trace_joint_plan_search import run_round

    s = JointSpace.live()
    m = {
        "space": s.manifest(),
        "tasks": ["0", "1"],
        "protocol": "failure-smoke",
        "smoke": True,
        "budget": 14,
        "task_concurrency": 1,
        "sources": source_identity(),
    }
    frozen = prepare_run(tmp_path, m)
    calls = []

    def fail(c, t, m):
        calls.append(t)
        raise RuntimeError("infrastructure unavailable")

    with pytest.raises(RuntimeError, match="infrastructure"):
        run_round(tmp_path, evaluator=fail)
    # Three consecutive failures trip the breaker; nothing stays claimed.
    assert len(calls) == 3
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        states = [p["state"] for p in ledger.pending()]
        assert "claimed" not in states and states.count("failed") == 3
        # A generic error is uncertain evidence: never retried automatically.
        assert all(not e["retryable"] for e in ledger.episodes("failed"))
        assert ledger.retry() == 0
        assert ledger.retry(include_uncertain=True) == 3


def _smoke_manifest(tmp_path, protocol):
    s = JointSpace.live()
    m = {
        "space": s.manifest(),
        "tasks": ["0", "1"],
        "protocol": protocol,
        "smoke": True,
        "budget": 14,
        "task_concurrency": 1,
        "sources": source_identity(),
    }
    return s, prepare_run(tmp_path, m)


def _receipt(t, m, **over):
    return {
        "task_id": t,
        "split": "TRAIN",
        "protocol": m["protocol"],
        "raw_score": 1.0,
        "total_tokens": 10,
        "latency_seconds": 1.0,
        "synthetic": True,
        **over,
    }


def test_transient_failure_is_retried_once(tmp_path):
    from trimem.experiments.trace_joint_plan_search import run_round

    s, frozen = _smoke_manifest(tmp_path, "retry-smoke")
    calls = []

    def flaky(c, t, m):
        calls.append(t)
        if calls.count(t) == 1 and t == "0":
            from trimem.optimizer.joint_search_ledger import RetryableEpisodeError

            raise RetryableEpisodeError("worker exited without a result")
        return _receipt(t, m)

    result = run_round(tmp_path, evaluator=flaky)
    assert result["status"] == "batch_complete"
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        assert not ledger.pending()


def test_invalid_receipt_is_never_marked_done(tmp_path):
    import pytest

    from trimem.experiments.trace_joint_plan_search import run_round

    s, frozen = _smoke_manifest(tmp_path, "validate-smoke")

    def bad(c, t, m):
        return _receipt(t, m, raw_score=float("nan"))

    with pytest.raises(RuntimeError, match="failed"):
        run_round(tmp_path, evaluator=bad)
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        assert not ledger.complete_configs()
        assert "nonfinite" in ledger.episodes("failed")[0]["error"]


def test_not_started_releases_claim_without_consuming_attempt(tmp_path):
    import pytest

    from trimem.experiments.trace_joint_plan_search import run_round
    from trimem.optimizer.joint_search_ledger import EpisodeNotStarted

    s, frozen = _smoke_manifest(tmp_path, "notstarted-smoke")

    def gate(c, t, m):
        raise EpisodeNotStarted("memory floor")

    with pytest.raises(RuntimeError, match="breaker=True"):
        run_round(tmp_path, evaluator=gate)
    with SearchLedger(tmp_path / "ledger.sqlite", frozen) as ledger:
        states = [p["state"] for p in ledger.pending()]
        assert set(states) == {"ready"}
        rows = ledger.db.execute("SELECT max(attempts) FROM episodes").fetchone()
        assert rows[0] == 0


def test_cli_actions_refuse_while_service_owns_run(tmp_path):
    import fcntl

    import pytest

    from trimem.experiments.trace_joint_plan_search import owned

    with (tmp_path / "queue.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="owns"):
            owned(tmp_path)
    with owned(tmp_path):
        pass
