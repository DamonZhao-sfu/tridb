"""Full parent state and deterministic cards must reach the OpenEvolve boundary."""

import json
from types import SimpleNamespace

import pytest

from bench.agent_memory.gem_oe.trajectory_retriever import (
    evidence_card,
    parent_context,
    streaming_plan_nodes,
)
from trimem.code_embedding import code_text


def test_code_tail_is_retained_in_embedding_and_semantic_context():
    code = "# prefix\n" * 1000 + "critical_tail_fix = 123\n"
    parent = SimpleNamespace(
        id="p",
        code=code,
        metrics={"combined_score": 0, "error": "index out of bounds"},
        metadata={},
    )
    context = parent_context(
        "math:x", "Place points inside a triangle.", parent, 7, 5000
    )
    assert code_text(code) == code
    actual = json.loads(context.to_text())
    assert actual["program_code"] == code
    assert "index out of bounds" in actual["failure_reason"]
    assert actual["remaining_token_budget"] == 5000
    assert actual["program_status"] == "failed"
    assert "Repair" in actual["objective"]


def test_feedback_and_budget_are_not_inferred_from_candidate():
    parent = SimpleNamespace(id="p", code="x=1", metrics={}, metadata={})
    c = parent_context("math:x", "An actual problem.", parent, 1)
    assert c.fitness is None and c.remaining_token_budget is None
    assert "unknown" in c.failure_reason
    assert "validity not explicitly recorded" == c.program_status


def test_card_is_only_stored_evidence_and_optional_complete_code():
    row = {
        "uid": "source",
        "id": 1,
        "fitness": 2.0,
        "parent_fitness": 1.0,
        "intent": "try a grid",
        "intent_source": "response_prose",
        "framework_changes": "changed initialization",
        "code": "a" * 4000 + "tail",
        "sem_score": 0.75,
    }
    assert evidence_card(row) == "try a grid"
    assert evidence_card(row, True) == "try a grid\n\n" + row["code"]
    assert (
        evidence_card({"framework_changes": "changed initialization"})
        == "changed initialization"
    )
    assert evidence_card({}) == ""


def test_repeat_retrieval_refreshes_card_in_openevolve():
    from openevolve.config import DatabaseConfig

    from bench.agent_memory.gem_oe.memory_database import (
        GemMemoryDatabase,
        RetrievedProgram,
    )

    db = GemMemoryDatabase(
        DatabaseConfig(), retriever=None, task_uid="math:x", render_mode="changes"
    )
    first = RetrievedProgram(
        uid="historical",
        code="full source",
        metrics={"combined_score": 1.0},
        changes_description='{"semantic_score":0.2}',
        provenance={"memory": "trajectory"},
    )
    p = db._register(first)
    second = RetrievedProgram(
        uid="historical",
        code="full source",
        metrics={"combined_score": 1.0},
        changes_description='{"semantic_score":0.9}',
        provenance={"memory": "trajectory"},
    )
    updated = db._register(second)
    assert p.id == updated.id
    assert json.loads(updated.code)["semantic_score"] == 0.9


def test_negative_evaluator_verdict_changes_repair_objective():
    parent = SimpleNamespace(
        id="p",
        code="x=1",
        metrics={"combined_score": 0, "judge_result": "CE"},
        metadata={},
    )
    c = parent_context("ale:test", "Compile a valid solution.", parent, 2)
    assert c.program_status == "failed"
    assert "Repair" in c.objective


@pytest.mark.parametrize("kind", ["Sort", "Hash Join", "Materialize"])
def test_streaming_guard_rejects_planner_regression(kind):
    with pytest.raises(ValueError, match="streaming contract"):
        streaming_plan_nodes(
            {
                "Node Type": kind,
                "Plans": [{"Node Type": "Index Scan", "Index Name": "attempt_hnsw"}],
            }
        )


def test_streaming_guard_requires_code_index():
    with pytest.raises(ValueError, match="HNSW"):
        streaming_plan_nodes({"Node Type": "Index Scan", "Index Name": "attempt_pkey"})


def test_retrieval_hard_cap_closes_cursor_after_four_records():
    from contextlib import nullcontext

    from bench.agent_memory.gem_oe.trajectory_retriever import TrajectoryRetriever

    class Cursor:
        fetched = 0
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def execute(self, *args):
            pass

        def __iter__(self):
            for i in range(10):
                self.fetched += 1
                yield {
                    "id": i,
                    "uid": f"memory-{i}",
                    "fitness": 1.0,
                    "code": f"x = {i}",
                    "intent": "Change x.",
                }

    cursor = Cursor()

    def execute(sql, *args):
        row = {"specification": "Improve the actual task."}
        if sql.startswith("EXPLAIN"):
            row = {
                "QUERY PLAN": [
                    {"Plan": {"Node Type": "Index Scan", "Index Name": "attempt_hnsw"}}
                ]
            }
        return SimpleNamespace(fetchone=lambda: row)

    retriever = TrajectoryRetriever.__new__(TrajectoryRetriever)
    retriever.conn = SimpleNamespace(
        execute=execute, transaction=nullcontext, cursor=lambda **kw: cursor
    )
    retriever.database = None
    retriever.embedder = SimpleNamespace(embed=lambda code: [1.0], last_receipt={})
    retriever.objective = None
    retriever.total_token_budget = None
    retriever.max_bytes = 524288
    retriever.out = None
    retriever.plan = "VGR"
    retriever.memory_tokens = 16384
    retriever.telemetry = []
    parent = SimpleNamespace(id="parent", code="x = 1", metrics={}, metadata={})
    result = retriever.retrieve(task_uid="math:x", parent=parent, k=10, iteration=1)
    assert len(result) == cursor.fetched == 4
    assert cursor.closed
    assert retriever.telemetry[-1]["requested_k"] == 10
    assert retriever.telemetry[-1]["effective_k"] == 4
