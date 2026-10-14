import json

import pytest

from trimem.experiments import ale_trajectory_run as ale


def checkpoint(path, n, found=7, uid="winner", code="int answer = 1;", score=42):
    ck = path / f"openevolve/checkpoints/checkpoint_{n}"
    (ck / "programs").mkdir(parents=True)
    (ck / "metadata.json").write_text(json.dumps({"last_iteration": n}))
    (ck / "best_program_info.json").write_text(
        json.dumps(
            {"id": uid, "current_iteration": n, "metrics": {"combined_score": score}}
        )
    )
    (ck / "programs" / f"{uid}.json").write_text(
        json.dumps(
            {
                "id": uid,
                "iteration_found": found,
                "code": code,
                "metrics": {"combined_score": score, "judge_result": "ACCEPTED"},
            }
        )
    )
    return ck


def test_ale_command_pins_contract_and_resumes_only_previous_stage(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ale, "OUT", tmp_path)
    args = ale.command("ahc008", "P1", 30)
    for flag, value in [
        ("--task", "ale:ahc008"),
        ("--max-iterations", "30"),
        ("--max-tokens", "32768"),
        ("--trajectory-prompt-tokens", "32768"),
        ("--trajectory-memory-tokens", "16384"),
        ("--num-diverse", "4"),
        ("--language", "cpp"),
        ("--eval-timeout", "600"),
        ("--checkpoint-interval", "10"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--no-simplify-nudge" in args
    assert "--diff-ws-tolerant" not in args
    assert args[args.index("--resume-from") + 1] == str(
        tmp_path / "it20/ahc008__P1__s42/openevolve/checkpoints/checkpoint_20"
    )
    assert "--resume-from" not in ale.command("ahc008", "P1", 10)


def test_exact_checkpoint_selection_does_not_look_at_future(tmp_path):
    checkpoint(tmp_path, 10, found=7, uid="old", code="int old;", score=20)
    checkpoint(tmp_path, 20, found=19, uid="new", code="int future;", score=999)
    selected = ale.checkpoint_selection(tmp_path, 10)
    assert selected["source_code"] == "int old;"
    assert selected["program_id"] == "old"
    assert selected["public_quality"] == 20


def test_future_or_wrong_checkpoint_is_rejected(tmp_path):
    ck = checkpoint(tmp_path, 10, found=11)
    with pytest.raises(ValueError, match="future"):
        ale.checkpoint_selection(tmp_path, 10)
    (ck / "metadata.json").write_text('{"last_iteration":20}')
    with pytest.raises(ValueError, match="checkpoint identity"):
        ale.checkpoint_selection(tmp_path, 10)


def test_isolated_public_evaluator_changes_only_worker_count(tmp_path):
    source = tmp_path / "source.py"
    target = tmp_path / "target.py"
    text = "NUM_CASES = 50\nWORKERS = 8\nMAGIC = 123\n"
    source.write_text(text)
    receipt = ale.isolated_evaluator(source, target)
    assert source.read_text() == text
    assert target.read_text() == text.replace("WORKERS = 8", "WORKERS = 1")
    assert receipt["public_cases"] == 50
    assert receipt["workers"] == 1


def test_math_barrier_requires_all_cells_and_supervisor_finalization(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ale.math_run, "OUT", tmp_path)
    monkeypatch.setattr(
        ale.math_run, "cells", lambda: [("task", "VGR", "one"), ("task", "P1", "two")]
    )
    monkeypatch.setattr(
        ale.math_run, "complete_cell", lambda path, _: path.name == "one"
    )
    (tmp_path / "status.json").write_text('{"status":"complete"}')
    assert not ale.math_barrier()["passed"]
    monkeypatch.setattr(ale.math_run, "complete_cell", lambda *_: True)
    assert ale.math_barrier()["passed"]
    (tmp_path / "status.json").write_text('{"status":"running"}')
    assert not ale.math_barrier()["passed"]


def test_private_score_uses_public_selected_code_without_mutating_checkpoint(
    tmp_path, monkeypatch
):
    from trimem.experiments import ale_private

    monkeypatch.setattr(ale, "OUT", tmp_path)
    path = ale.stage_path("ahc008", "VGR", 10)
    ck = checkpoint(path, 10, code="int selected;")
    (path / "initial_program.py").write_text("int seed;")
    before = {str(p): p.read_bytes() for p in ck.rglob("*") if p.is_file()}
    calls = []

    def score(task, code, cache, workers):
        calls.append((task, code, workers))
        return {
            "private_abs": 10,
            "rank": 3,
            "performance": 200 if "selected" in code else 100,
            "judge": "ACCEPTED",
            "cases": 200,
            "eval_seconds": 1,
        }

    monkeypatch.setattr(ale_private, "score", score)
    assert ale.private_score("ahc008", "VGR", 10) == 0
    result = json.loads(ale.private_path("ahc008", "VGR", 10).read_text())
    assert result["delta_performance"] == 100
    assert result["private_feedback_to_agent"] is False
    assert calls == [("ahc008", "int seed;", 4), ("ahc008", "int selected;", 4)]
    assert before == {str(p): p.read_bytes() for p in ck.rglob("*") if p.is_file()}


def test_private_failure_is_explicit_not_zero_quality(tmp_path, monkeypatch):
    from trimem.experiments import ale_private

    monkeypatch.setattr(ale, "OUT", tmp_path)
    path = ale.stage_path("ahc008", "VGR", 10)
    checkpoint(path, 10)
    (path / "initial_program.py").write_text("int seed;")

    def fail(*_args, **_kwargs):
        raise RuntimeError("fixture evaluator error")

    monkeypatch.setattr(ale_private, "score", fail)
    assert ale.private_score("ahc008", "VGR", 10) == 1
    result = json.loads(ale.private_path("ahc008", "VGR", 10).read_text())
    assert result["status"] == "failed"
    assert "performance" not in result
    assert "fixture evaluator error" in result["error"]


def test_each_checkpoint_stage_contains_exactly_ten_new_rounds(tmp_path):
    checkpoint(tmp_path, 30, found=27)
    (tmp_path / "run_receipt.json").write_text(
        '{"status":"complete","iterations_traced":10,"resumed_at_iteration":20}'
    )
    (tmp_path / "generation_prompt_index.jsonl").write_text(
        "".join(json.dumps({"iteration": i}) + "\n" for i in range(21, 31))
    )
    assert ale.complete_stage(tmp_path, 30)
    (tmp_path / "generation_prompt_index.jsonl").write_text(
        "".join(json.dumps({"iteration": i}) + "\n" for i in range(1, 31))
    )
    assert not ale.complete_stage(tmp_path, 30)
