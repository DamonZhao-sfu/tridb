import json

from trimem.experiments import math_trajectory_run as run


def test_matrix_pins_budgets_and_output():
    args = run.command(run.TASKS[0], "VGR")
    for flag, expected in [
        ("--trajectory-memory-tokens", "16384"),
        ("--trajectory-prompt-tokens", "32768"),
        ("--max-tokens", "32768"),
        ("--num-diverse", "4"),
        ("--thinking", "false"),
    ]:
        assert args[args.index(flag) + 1] == expected
    assert "--resume-from" not in args


def test_quality_accepts_zero_but_rejects_explicit_errors():
    assert run.quality({"combined_score": 0}) == 0
    assert run.quality({"combined_score": 1, "is_valid": False}) is None
    assert run.quality({"combined_score": 1, "error": "invalid arrangement"}) is None
    assert run.quality({"combined_score": float("nan")}) is None


def test_inflight_checkpoint_not_reported_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    cell = tmp_path / "test__VGR__s42"
    cell.mkdir()
    (cell / "run_receipt.json").write_text(json.dumps({"status": "running"}))
    (cell / "retrieval_audit.jsonl").write_text(json.dumps({"iteration": 10}) + "\n")
    row = run.summarize_cell("test", "VGR", [])
    assert row["attempted_rounds"] == 0
    assert row["retrieval_decisions"] == 1
    assert row["quality10"] is None
    assert row["generation_prompt_tokens"] is None


def test_semantic_usage_requires_matching_context(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    cell = tmp_path / "test__P1__s42"
    cell.mkdir()
    (cell / "trajectory_retrieval.jsonl").write_text(
        json.dumps({"context_sha256": "ours"}) + "\n"
    )
    semantic = [
        {
            "context_sha256": h,
            "op": "SemFilter",
            "usage": {"prompt_tokens": n, "completion_tokens": 5},
        }
        for h, n in [("ours", 12), ("other", 900)]
    ]
    row = run.summarize_cell("test", "P1", semantic)
    assert row["semantic_usage"]["SemFilter"]["prompt_tokens"] == 12
    assert row["semantic_usage"]["SemScore"]["prompt_tokens"] == 0


def test_prefetched_rounds_do_not_advance_dispatch_or_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    cell = tmp_path / "test__VGR__s42"
    cell.mkdir()
    (cell / "run_receipt.json").write_text(
        json.dumps({"status": "running", "live_seed_metrics": {"combined_score": 0.1}})
    )
    # OpenEvolve can prefetch many retrieval decisions while only one generation runs.
    (cell / "retrieval_audit.jsonl").write_text(
        "".join(json.dumps({"iteration": i}) + "\n" for i in range(1, 11))
    )
    (cell / "generation_prompt_index.jsonl").write_text(
        json.dumps({"iteration": 1}) + "\n"
    )
    row = run.summarize_cell("test", "VGR", [])
    assert row["retrieval_decisions"] == 10
    assert row["attempted_rounds"] == 1
    assert row["child_count"] == 0
    assert row["quality10"] is None
    assert row["curve"] == [{"iteration": 0, "quality": 0.1}]
    # Only actual completed child1 should advance the curve, not the prefetched frontier10.
    (cell / "evolution_trace.jsonl").write_text(
        json.dumps({"iteration": 1, "child_metrics": {"combined_score": 0.4}}) + "\n"
    )
    row = run.summarize_cell("test", "VGR", [])
    assert row["curve"] == [
        {"iteration": 0, "quality": 0.1},
        {"iteration": 1, "quality": 0.4},
    ]
    assert row["quality10"] is None


def test_parallel_queue_prioritizes_plans_within_task():
    queue = run.matrix_cells()
    assert [(t, p) for t, p, _ in queue[:4]] == [(run.TASKS[0], p) for p in run.PLANS]
    assert len(queue) == len({c[2] for c in queue}) == 12


def test_admission_caps_active_and_skips_completed_failed_or_adopted():
    queue = run.matrix_cells()
    active = {queue[1][2]: object(), queue[2][2]: object()}
    completed = [queue[0][2]]
    failed = {queue[3][2]: "failed"}
    selected = run.admission(queue, active, completed, failed, 6)
    assert len(selected) == 4
    assert not ({c[2] for c in selected} & (set(active) | set(completed) | set(failed)))
    # Lowering the cap blocks admission without requesting termination of running cells.
    assert run.admission(queue, active, completed, failed, 1) == []


def test_adoption_preserves_legacy_and_parallel_pids():
    legacy = run.adoption_cells({"cell": "c", "cell_pid": 123, "cell_started_unix": 42})
    assert legacy == {"c": {"pid": 123, "started_unix": 42}}
    parallel = {"c": {"pid": 123}, "d": {"pid": 456}}
    assert run.adoption_cells({"active_cells": parallel}) == parallel
    assert run.is_complete({"status": "complete", "iterations_traced": 20})
    assert not run.is_complete({"status": "complete", "iterations_traced": 19})


def test_dynamic_control_changes_admission_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    assert run.concurrency_limit(6) == 6
    (tmp_path / "control.json").write_text('{"max_parallel":12}')
    assert run.concurrency_limit(6) == 12
    (tmp_path / "control.json").write_text('{"max_parallel":4}')
    assert run.concurrency_limit(6) == 4


def test_parallel_shared_context_cost_is_not_counted_for_each_cell(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run, "OUT", tmp_path)
    name = "test__P1__s42"
    cell = tmp_path / name
    cell.mkdir()
    (cell / "trajectory_retrieval.jsonl").write_text(
        json.dumps({"context_sha256": "shared"}) + "\n"
    )
    semantic = [
        {
            "context_sha256": "shared",
            "op": "SemFilter",
            "usage": {"prompt_tokens": 99, "completion_tokens": 5},
        }
    ]
    row = run.summarize_cell(
        "test", "P1", semantic, {("SemFilter", "shared"): {name, "test__P2__s42"}}
    )
    usage = row["semantic_usage"]["SemFilter"]
    assert usage["prompt_tokens"] is None
    assert usage["observed_api_calls"] == 0
    assert usage["ambiguous_api_calls_excluded"] == 1


def test_existing_incomplete_cell_fails_before_any_launch(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(run, "OUT", tmp_path)
    (tmp_path / "launch_authorized.json").write_text('{"prerequisites_passed":true}')
    _task, _plan, name = run.matrix_cells()[0]
    cell = tmp_path / name
    cell.mkdir()
    receipt = {"status": "running", "iterations_traced": 7}
    (cell / "run_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(
        run, "preflight", lambda: {"passed": True, "available_cpus": [0, 1]}
    )
    monkeypatch.setattr(run.os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr(run.os, "getpriority", lambda *_: 10)
    monkeypatch.setattr(
        run.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("must not launch over an existing cell"),
    )
    with pytest.raises(RuntimeError, match="never overwritten"):
        run.run(6)
    assert json.loads((cell / "run_receipt.json").read_text()) == receipt
