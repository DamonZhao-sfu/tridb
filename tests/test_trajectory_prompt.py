"""Verify the agent sees real history, complete code and unchanged task systems."""

import copy
import json
import pickle

import pytest
from openevolve.config import PromptConfig

from bench.agent_memory.gem_oe import trajectory_prompt as tp


def program(pid, iteration, fitness, parent_fitness=1.0):
    return {
        "id": pid,
        "iteration_found": iteration,
        "parent_id": "seed",
        "code": f"def candidate_{pid}():\n    return {iteration}\n",
        "language": "python",
        "metrics": {"combined_score": fitness, "eval_time": 99},
        "changes_description": "Changed initialization.",
        "metadata": {
            "parent_metrics": {"combined_score": parent_fitness, "eval_time": 1}
        },
    }


def sampler(**kwargs):
    cfg = PromptConfig()
    cfg.template_dir = str(tp.TEMPLATE_DIR)
    cfg.system_message = "Original task system\nDo not change this.\n"
    cfg.suggest_simplification_after_chars = None
    cfg.code_length_threshold = None
    return tp.TrajectoryPromptSampler(cfg, **kwargs)


def test_history_excludes_future_external_and_seed_and_uses_recency():
    rows = [program(str(i), i, 10 - i) for i in range(1, 7)]
    external = program("outside", 2, 100)
    external["metadata"]["provenance"] = {"memory": "trajectory"}
    previous, top = tp.own_context(rows + [external], "4", 6)
    assert [p["id"] for p in previous] == ["3", "4", "5"]
    assert [p["id"] for p in top] == ["1", "2", "3", "5"]


def test_outcome_uses_quality_not_increased_runtime_and_never_invents_parent():
    assert tp.outcome(program("a", 1, 2)) == "Improvement in combined_score"
    assert tp.outcome(program("a", 1, 1)) == "No change in combined_score"
    assert tp.outcome(program("a", 1, 0)) == "Decline in combined_score"
    missing = program("a", 1, 2)
    missing["metadata"] = {}
    assert "unavailable" in tp.outcome(missing)


def test_top_excludes_invalid_and_duplicate_parent_code():
    parent = program("parent", 1, 1)
    duplicate = program("duplicate", 2, 3)
    duplicate["code"] = parent["code"]
    invalid = program("invalid", 3, 5)
    invalid["metrics"]["judge_result"] = "TIME_LIMIT_EXCEEDED"
    valid = program("valid", 4, 2)
    _, top = tp.own_context([parent, duplicate, invalid, valid], "parent", 5)
    assert [p["id"] for p in top] == ["valid"]


def test_catalog_counts_programs_not_code_fences():
    from tools.trajectory_prompt_catalog import context_section, program_count

    user = "## Other Context Solutions\n### Program 1 (score: 1)\n```cpp\n# Current Solution\n```\n```cpp\nsecond block\n```\n### Program 2 (score: 2)\n```cpp\ncode\n```\n# Current Solution\nparent"
    section, codes = context_section(user)
    assert program_count(section) == 2
    assert len(codes) == 3
    assert "parent" not in section


def test_runner_accepts_code_mode_for_trajectory_retriever(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from bench.agent_memory.gem_oe import trajectory_retriever
    from bench.agent_memory.gem_oe.run_arm import build_retriever

    monkeypatch.setattr(trajectory_retriever, "TrajectoryRetriever", lambda **kw: kw)
    args = SimpleNamespace(
        arm="trimem",
        trajectory_store=True,
        inject_as="code",
        trimem_dsn="test",
        trajectory_plan="P2+Rank",
        out=tmp_path,
        trajectory_total_token_budget=None,
        trajectory_objective=None,
        trajectory_memory_tokens=8192,
        trajectory_include_code=True,
    )
    result = build_retriever(args)
    assert result["include_code"] is True


def test_real_prompt_includes_full_code_intent_and_previous_and_top():
    s = sampler(
        own_programs=[program(str(i), i, i) for i in range(1, 5)],
        parent_id="4",
        iteration=5,
    )
    inspiration = program("memory", 0, 8)
    inspiration["code"] = "# full source\n" * 1000 + "tail_must_survive = 42\n"
    original = copy.deepcopy(inspiration)
    result = s.build_prompt(
        current_program="parent = 42",
        program_metrics={"combined_score": 4},
        inspirations=[inspiration],
        language="python",
        evolution_round=5,
    )
    assert result["system"] == s.config.system_message
    assert "### Attempt 4" in result["user"]
    assert result["user"].index("### Attempt 4") < result["user"].index("### Attempt 2")
    assert "## Top Performing Programs\n\n### Program 1" in result["user"]
    section = (
        result["user"].split("## Inspiration Programs")[1].split("# Current Program")[0]
    )
    assert inspiration["code"] in section
    assert "Intent: Changed initialization." in section
    for field in ("source_uid", "attempt_id", "historical_status", "semantic_score"):
        assert field not in section
    assert inspiration == original


def test_top_budget_skips_whole_program_and_records_it(monkeypatch):
    from trimem.objective.store import count_tokens

    s = sampler()
    long = program("long", 1, 2)
    long["code"] = "x = 3\n" * 100
    short = program("short", 2, 1)
    monkeypatch.setattr(tp, "TOP_TOKEN_BUDGET", count_tokens(short["code"]) + 45)
    text = s._format_evolution_history([], [long, short], [], "python")
    assert long["code"] not in text
    assert short["code"] in text
    assert s.context_receipt["top_skipped_for_budget"] == 1
    assert s.context_receipt["truncated_records"] == 0


def test_worker_hook_installs_sampler_and_restores_it(monkeypatch):
    from types import SimpleNamespace

    import openevolve.process_parallel as parallel

    original = object()
    monkeypatch.setattr(parallel, "_worker_prompt_sampler", original, raising=False)
    monkeypatch.setattr(
        parallel,
        "_worker_config",
        SimpleNamespace(prompt=sampler().config),
        raising=False,
    )
    monkeypatch.setattr(parallel, "_lazy_init_worker_components", lambda: None)
    snapshot = {"programs": {"a": program("a", 1, 1)}, "islands": [["a"]]}

    def worker(*args):
        assert isinstance(parallel._worker_prompt_sampler, tp.TrajectoryPromptSampler)
        return parallel._worker_prompt_sampler.own_programs

    monkeypatch.setattr(tp, "_base_worker", worker)
    assert tp.trajectory_iteration_worker(2, snapshot, "a", []) == [
        snapshot["programs"]["a"]
    ]
    assert parallel._worker_prompt_sampler is original
    snapshot["islands"] = [[]]
    assert tp.trajectory_iteration_worker(2, snapshot, "a", []) == []
    assert (
        pickle.loads(pickle.dumps(tp.trajectory_iteration_worker))
        is tp.trajectory_iteration_worker
    )


def test_current_and_previous_feedback_are_separate_and_not_truncated():
    parent = program("parent", 1, 1)
    recent = program("recent", 2, 0)
    recent["metrics"]["error"] = "recent compile failure"
    parent["metadata"]["evaluation_feedback"] = "current metadata diagnostic"
    current_error = "runtime diagnostic\n" * 2000 + "CURRENT_ERROR_TAIL"
    artifacts = {
        "parent": {"stderr": current_error},
        "recent": {"stderr": "RECENT_ERROR_TAIL"},
    }
    s = sampler(
        own_programs=[parent, recent],
        parent_id="parent",
        iteration=3,
        artifacts_by_program=artifacts,
    )
    rendered = s.build_prompt(
        current_program=parent["code"],
        program_metrics=parent["metrics"],
        inspirations=[],
        language="python",
    )
    user = rendered["user"]
    current = user.split("## Current Evaluation Feedback\n")[1].split(
        "# Program Evolution History"
    )[0]
    previous = user.split("### Attempt 2\n")[1].split("### Attempt 1")[0]
    assert "CURRENT_ERROR_TAIL" in current
    assert "current metadata diagnostic" in current
    assert "RECENT_ERROR_TAIL" not in current
    assert "RECENT_ERROR_TAIL" in previous and "recent compile failure" in previous
    assert "CURRENT_ERROR_TAIL" not in previous
    assert "... (truncated)" not in current
    assert rendered["system"] == s.config.system_message


def test_missing_feedback_is_explicit():
    assert (
        tp.evaluation_feedback({})
        == "No additional evaluation/error feedback recorded."
    )
    assert "No additional" in tp.evaluation_feedback(
        program("p", 1, 1), include_metrics=False
    )


def test_snapshot_loads_selected_and_recent_artifacts_past_base_cap(monkeypatch):
    from types import SimpleNamespace

    programs = {str(i): program(str(i), i, i) for i in range(1, 105)}
    base = {"programs": programs, "islands": [list(programs)], "artifacts": {}}
    monkeypatch.setattr(tp, "_base_snapshot", lambda controller: copy.deepcopy(base))
    db = SimpleNamespace(
        injection_trace=[{"parent_id": "1"}],
        get_artifacts=lambda pid: {"stderr": "feedback-" + pid},
    )
    snapshot = tp.trajectory_database_snapshot(SimpleNamespace(database=db))
    assert set(snapshot["artifacts"]) == {"1", "102", "103", "104"}


def test_chat_template_count_includes_assistant_prefix():
    from trimem.objective.store import count_tokens

    prompt = {"system": "s", "user": "u"}
    rendered = tp.generation_tokenizer().apply_chat_template(
        [{"role": key, "content": value} for key, value in prompt.items()],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "<|im_start|>assistant" in rendered
    assert tp.generation_prompt_tokens(prompt) == count_tokens(rendered)
    assert tp.generation_prompt_tokens(prompt) > sum(
        count_tokens(v) for v in prompt.values()
    )


def test_inspiration_cap_and_section_budget_preserve_whole_records():
    from trimem.objective.store import count_tokens

    memories = [program(f"memory{i}", i, 1) for i in range(6)]
    s = sampler()
    rendered = s.build_prompt(current_program="current = 1", inspirations=memories)
    assert s.context_receipt["inspiration_count"] == 4
    assert [x["id"] for x in s.context_receipt["removed_records"]] == [
        "memory4",
        "memory5",
    ]
    assert all(p["code"] in rendered["user"] for p in memories[:4])
    assert all(p["code"] not in rendered["user"] for p in memories[4:])
    memories[0]["code"] = "large = 123\n" * 100
    limit = count_tokens(s._format_inspirations_section([memories[1]], "python"))
    s = sampler(inspiration_token_budget=limit)
    rendered = s.build_prompt(current_program="current = 1", inspirations=memories[:2])
    assert s.context_receipt["inspiration_ids"] == ["memory1"]
    assert memories[0]["code"] not in rendered["user"]
    assert memories[1]["code"] in rendered["user"]
    assert s.context_receipt["truncated_records"] == 0


def test_full_budget_drops_inspirations_then_top_then_oldest_feedback(
    monkeypatch, tmp_path
):
    recent = [program(str(i), i, 1) for i in range(1, 4)]
    for p in recent:
        p["metrics"]["error"] = f"original error {p['id']}\n" * 100
    top = program("top", 0, 3)
    memory = program("memory", 0, 2)
    base_args = {
        "current_program": "current_code_tail = 1",
        "program_metrics": {"error": "CURRENT_FAILURE_TAIL"},
        "evolution_round": 4,
    }
    baseline = sampler().build_prompt(**base_args, previous_programs=recent[1:])
    limit = tp.generation_prompt_tokens(baseline)
    path = tmp_path / "tokens.jsonl"
    monkeypatch.setenv("TRIMEM_TRAJECTORY_PROMPT_LOG", str(path))
    s = sampler(prompt_token_budget=limit)
    rendered = s.build_prompt(
        **base_args, previous_programs=recent, top_programs=[top], inspirations=[memory]
    )
    assert rendered == baseline
    assert [x["section"] for x in s.context_receipt["removed_records"]] == [
        "inspirations",
        "top_programs",
        "previous_programs",
    ]
    assert s.context_receipt["previous_attempt_ids"] == ["2", "3"]
    assert "CURRENT_FAILURE_TAIL" in rendered["user"]
    assert "current_code_tail = 1" in rendered["user"]
    assert s.context_receipt["prompt_input_tokens"] <= limit
    assert s.context_receipt["prompt_input_tokens_original"] > limit
    assert json.loads(path.read_text())["removed_record_count"] == 3
    assert s.context_receipt["truncated_records"] == 0


def test_mandatory_prompt_overflow_is_logged_and_fails_without_clipping(
    monkeypatch, tmp_path
):
    path = tmp_path / "overflow.jsonl"
    monkeypatch.setenv("TRIMEM_TRAJECTORY_PROMPT_LOG", str(path))
    s = sampler(prompt_token_budget=20)
    with pytest.raises(ValueError, match="Mandatory trajectory prompt"):
        s.build_prompt(
            current_program="whole_code_tail = 1",
            program_metrics={"error": "whole_error_tail"},
        )
    receipt = json.loads(path.read_text())
    assert receipt["prompt_budget_error"] is True
    assert receipt["prompt_input_tokens"] > 20
    assert receipt["truncated_records"] == 0


def test_worker_budget_configuration_comes_from_environment(monkeypatch):
    monkeypatch.setenv("TRIMEM_TRAJECTORY_PROMPT_TOKEN_BUDGET", "32001")
    monkeypatch.setenv("TRIMEM_TRAJECTORY_MEMORY_TOKEN_BUDGET", "16001")
    monkeypatch.setenv("TRIMEM_TRAJECTORY_ENABLE_THINKING", "true")
    s = sampler()
    assert s.prompt_token_budget == 32001
    assert s.inspiration_token_budget == 16001
    assert s.enable_thinking is True
