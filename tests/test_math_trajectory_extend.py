import json

import pytest

from trimem.experiments import math_trajectory_extend as extension


def complete(root, name, horizon):
    path = root / name
    path.mkdir(parents=True)
    checkpoint = path / f"openevolve/checkpoints/checkpoint_{horizon}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"last_iteration": horizon}))
    (path / "run_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "iterations_traced": 20 if horizon == 20 else 30,
                "resumed_at_iteration": 0 if horizon == 20 else 20,
            }
        )
    )
    start = 1 if horizon == 20 else 21
    (path / "generation_prompt_index.jsonl").write_text(
        "".join(json.dumps({"iteration": i}) + "\n" for i in range(start, horizon + 1))
    )
    return path


def test_resume_command_preserves_parameters_and_uses_fresh_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(extension, "STAGE20", tmp_path / "first")
    monkeypatch.setattr(extension, "STAGE50", tmp_path / "continuation")
    command = extension.cell_command("circle_packing", "P1", 50)
    assert command[command.index("--max-iterations") + 1] == "50"
    assert command[command.index("--checkpoint-interval") + 1] == "10"
    assert command[command.index("--max-tokens") + 1] == "32768"
    assert command[command.index("--trajectory-prompt-tokens") + 1] == "32768"
    assert command[command.index("--resume-from") + 1] == str(
        tmp_path / "first/circle_packing__P1__s42/openevolve/checkpoints/checkpoint_20"
    )
    assert command[command.index("--out") + 1] == str(
        tmp_path / "continuation/circle_packing__P1__s42"
    )
    assert "--resume-from" not in extension.cell_command("circle_packing", "P1", 20)


def test_all_four_tasks_must_complete20_before_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(extension, "STAGE20", tmp_path / "first")
    monkeypatch.setattr(extension, "STAGE50", tmp_path / "next")
    for task, _plan, name in extension.cells():
        if task != "circle_packing":
            complete(extension.STAGE20, name, 20)
    assert len(extension.barrier20()["completed_cells"]) == 12
    assert len(extension.stage_pending(20)) == 4
    assert all(t == "circle_packing" for t, _p, _n in extension.stage_pending(20))
    with pytest.raises(RuntimeError, match="All16 cells"):
        extension.stage_pending(50)
    for _task, _plan, name in extension.cells()[12:]:
        complete(extension.STAGE20, name, 20)
    assert extension.barrier20()["pending_cells"] == []
    assert extension.stage_pending(20) == []
    assert len(extension.stage_pending(50)) == 16


def test_continuation_completion_is30_new_rounds_not50(tmp_path):
    path = complete(tmp_path, "cell", 50)
    assert extension.complete_cell(path, 50)
    receipt = json.loads((path / "run_receipt.json").read_text())
    receipt["iterations_traced"] = 50
    (path / "run_receipt.json").write_text(json.dumps(receipt))
    assert not extension.complete_cell(path, 50)


def test_checkpoint_and_dispatch_indices_must_match_horizon(tmp_path):
    path = complete(tmp_path, "cell", 20)
    assert extension.complete_cell(path, 20)
    (path / "openevolve/checkpoints/checkpoint_20/metadata.json").write_text(
        '{"last_iteration":19}'
    )
    assert not extension.complete_cell(path, 20)
    (path / "openevolve/checkpoints/checkpoint_20/metadata.json").write_text(
        '{"last_iteration":20}'
    )
    (path / "generation_prompt_index.jsonl").write_text(
        "".join(json.dumps({"iteration": i}) + "\n" for i in range(2, 22))
    )
    assert not extension.complete_cell(path, 20)


def test_incomplete_original_output_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(extension, "STAGE20", tmp_path)
    name = extension.cells()[0][2]
    path = tmp_path / name
    path.mkdir()
    (path / "run_receipt.json").write_text('{"status":"running"}')
    with pytest.raises(RuntimeError, match="cannot be overwritten"):
        extension.stage_pending(20)
    assert (path / "run_receipt.json").read_text() == '{"status":"running"}'


def test_stitch_preserves_best20_and_counts_each_stage_once():
    first = {
        "status": "complete",
        "path": "/first",
        "curve": [
            {"iteration": 0, "quality": 0.1},
            {"iteration": 10, "quality": 0.5},
            {"iteration": 20, "quality": 0.9},
        ],
        "attempted_rounds": 20,
        "generation_prompt_tokens": 1000,
        "semantic_usage": {
            "SemFilter": {"prompt_tokens": 10},
            "SemScore": {"prompt_tokens": 0},
        },
    }
    later = {
        "status": "complete",
        "path": "/later",
        "curve": [
            {"iteration": 0, "quality": 0.1},
            {"iteration": 21, "quality": 0.3},
            {"iteration": 30, "quality": 0.6},
            {"iteration": 40, "quality": 1.0},
            {"iteration": 50, "quality": 1.1},
        ],
        "attempted_rounds": 30,
        "generation_prompt_tokens": 2000,
        "semantic_usage": {
            "SemFilter": {"prompt_tokens": 20},
            "SemScore": {"prompt_tokens": 0},
        },
    }
    combined = extension.merge_rows(first, later)
    assert combined["quality10"] == 0.5
    assert combined["quality20"] == combined["quality30"] == 0.9
    assert combined["quality40"] == 1.0
    assert combined["quality50"] == 1.1
    assert combined["attempted_rounds"] == 50
    assert combined["generation_prompt_tokens"] == 3000
    assert combined["semantic_usage"]["SemFilter"]["prompt_tokens"] == 30
    assert len([p for p in combined["curve"] if p["iteration"] == 0]) == 1


def test_default_branch_is_required_unless_exact_explicit_override():
    defaults = {"default_code_branch": "experiment-trajectory-retrieval-v4"}
    assert extension.branch_contract(defaults["default_code_branch"], defaults)[
        "passed"
    ]
    with pytest.raises(RuntimeError, match="branch mismatch"):
        extension.branch_contract("main", defaults)
    assert (
        extension.branch_contract("other-experiment", defaults, "other-experiment")[
            "override"
        ]
        == "other-experiment"
    )
    with pytest.raises(RuntimeError, match="branch mismatch"):
        extension.branch_contract("main", defaults, "other-experiment")


def test_missing_branch_contract_is_explicit_error():
    with pytest.raises(RuntimeError, match="default_code_branch"):
        extension.branch_contract("main", {})
