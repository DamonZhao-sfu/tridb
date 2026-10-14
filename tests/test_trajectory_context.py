"""Dynamic judge context must survive every semantic placement and change with state."""

import hashlib
import json
from dataclasses import replace

import pytest

from trimem.trajectory_query import (
    CurrentAgentContext,
    bind_context,
    check_request_size,
    query,
    render_judge_input,
)


def context():
    return CurrentAgentContext(
        task_id="math:test",
        problem="Place three points inside the triangle.",
        objective="Repair the out-of-bounds point without reducing the minimum area.",
        program_code="points = [(2, 2)]",
        program_status="invalid",
        failure_reason="Point (2, 2) lies outside the triangle.",
        constraints="All points must be inside.",
        provenance="agent state at round 4",
        fitness=0.0,
        iteration=4,
    )


def params(c):
    return bind_context(
        {
            "task": c.task_id,
            "predicate": "Is this edit applicable?",
            "score_predicate": "Rate expected usefulness.",
        },
        c,
    )


@pytest.mark.parametrize(
    "field",
    [
        "problem",
        "objective",
        "program_code",
        "program_status",
        "failure_reason",
        "constraints",
        "provenance",
    ],
)
def test_context_rejects_missing_fields(field):
    with pytest.raises(ValueError, match=field):
        params(replace(context(), **{field: ""}))


def test_context_rejects_task_name_as_problem_and_cross_task():
    c = context()
    with pytest.raises(ValueError, match="problem statement"):
        params(replace(c, problem=c.task_id))
    with pytest.raises(ValueError, match="task differ"):
        bind_context({"task": "other"}, c)
    with pytest.raises(TypeError, match="requires CurrentAgentContext"):
        bind_context({"task": c.task_id}, None)


@pytest.mark.parametrize(
    "field", ["problem", "objective", "program_code", "failure_reason", "fitness"]
)
def test_context_digest_changes_with_agent_state(field):
    c = context()
    changed = replace(
        c, **{field: 0.25 if field == "fitness" else "new observed state"}
    )
    assert (
        params(c)["current_context_sha256"] != params(changed)["current_context_sha256"]
    )
    assert render_judge_input(
        params(c)["current_context"], "same candidate"
    ) != render_judge_input(params(changed)["current_context"], "same candidate")


def test_full_code_and_feedback_are_preserved_and_oversize_rejected():
    raw = '当前程序出错。🙂\\"\n' * 1000
    c = replace(
        context(), program_code=raw, problem=raw, objective=raw, failure_reason=raw
    )
    text = c.to_text()
    obj = json.loads(text)
    assert obj["program_code"] == raw
    assert obj["field_coverage"]["program_code"]["omitted_chars"] == 0
    assert (
        obj["field_coverage"]["program_code"]["source_sha256"]
        == hashlib.sha256(raw.encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="nothing was truncated"):
        c.to_text(max_bytes=4096)
    with pytest.raises(ValueError, match="limit"):
        check_request_size("predicate", "🙂" * 9000, max_input_bytes=32768)


def test_unknown_failure_is_explicit_not_imputed():
    c = replace(
        context(),
        failure_reason="No diagnosis recorded; failure state unknown.",
        fitness=None,
    )
    obj = json.loads(params(c)["current_context"])
    assert obj["fitness"] is None and "unknown" in obj["failure_reason"]


@pytest.mark.parametrize(
    "plan", ["P1", "P2", "SemRankOnly", "P1+Rank", "P2+Rank", "SemJoin/lowered"]
)
def test_every_semantic_operator_gets_dynamic_context(plan):
    sql = query(plan, "emb")
    n = sql.count("trimem.sem_filter(") + sql.count("trimem.sem_score(")
    assert sql.count("%(current_context)s") == n
    assert n > 0
    assert "CURRENT_AGENT_CONTEXT=" in sql and "CANDIDATE_EVIDENCE=" in sql
    assert (
        "CURRENT INTENT=" not in sql
    )  # no stale DB intent posing as current live state


def test_nonsemantic_vgr_has_no_new_context_dependency():
    assert "%(current_context)s" not in query("VGR", "emb")


@pytest.mark.parametrize("plan", ["P1", "P2", "P1+Rank", "P2+Rank", "SemRankOnly"])
def test_candidate_complete_code_is_bound_at_every_semantic_placement(plan):
    sql = query(plan, "emb")
    assert "candidate_code=" in sql
    assert "coalesce(a.code,'')" in sql
    assert "left(a.code" not in sql and "left(d.code" not in sql
    if plan != "SemRankOnly":
        assert "coalesce(d.code,'')" in sql
