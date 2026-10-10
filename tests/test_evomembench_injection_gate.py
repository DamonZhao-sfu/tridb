from __future__ import annotations

import json

import pytest

from bench.agent_memory.evomembench.injection_gate import (
    INJECTION_RATE_ARMS,
    SCHEMA_VERSION,
    gate_decision,
    injection_rate_for_arm,
    load_gate_manifest,
)
from bench.agent_memory.evomembench.run_know_system import (
    _is_evolving_arm,
    _physical_plan,
)


def _manifest(tasks: int = 107) -> dict:
    rows = []
    for index in range(tasks):
        rows.append(
            {
                "task_id": f"task-{index}",
                "context_id": "context-1",
                "ordinal": index + 1,
                "gates": {
                    "p0.1": index < 11,
                    "p0.5": index < 54,
                    "p1.0": True,
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "task_count": tasks + 1,
        "context_count": 1,
        "reuse_decision_count": tasks,
        "tasks": rows,
    }


def test_injection_rate_arms_use_vector_seeded_graph_expand() -> None:
    assert set(INJECTION_RATE_ARMS) == {
        "evolving_graph_expand_p0.1",
        "evolving_graph_expand_p0.5",
        "evolving_graph_expand_p1.0",
    }
    for arm in INJECTION_RATE_ARMS:
        assert _is_evolving_arm(arm)
        assert _physical_plan(arm) == "graph_expand"
    assert injection_rate_for_arm("evolving_graph_expand_p0.1") == 0.1
    assert injection_rate_for_arm("memory_off") is None


def test_gate_manifest_loads_nested_exact_cardinality(tmp_path) -> None:
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(_manifest()))
    payload = load_gate_manifest(path)
    by_id = payload["_tasks_by_id"]
    assert len(by_id) == 107
    assert sum(row["gates"]["p0.1"] for row in by_id.values()) == 11
    assert sum(row["gates"]["p0.5"] for row in by_id.values()) == 54
    assert sum(row["gates"]["p1.0"] for row in by_id.values()) == 107


def test_gate_decision_distinguishes_no_history_off_and_on(tmp_path) -> None:
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(_manifest()))
    payload = load_gate_manifest(path)

    assert gate_decision(
        payload,
        arm="evolving_graph_expand_p0.1",
        task_id="ordinal-zero-is-not-in-manifest",
        context_id="context-1",
        ordinal=0,
    ) == (False, "no_prior_history", 0.1)
    assert gate_decision(
        payload,
        arm="evolving_graph_expand_p0.1",
        task_id="task-50",
        context_id="context-1",
        ordinal=51,
    ) == (False, "injection_gate_off", 0.1)
    assert gate_decision(
        payload,
        arm="evolving_graph_expand_p0.5",
        task_id="task-50",
        context_id="context-1",
        ordinal=51,
    ) == (True, "injection_gate_on", 0.5)


def test_gate_manifest_rejects_non_nested_rows(tmp_path) -> None:
    payload = _manifest(tasks=1)
    payload["tasks"][0]["gates"] = {
        "p0.1": True,
        "p0.5": False,
        "p1.0": True,
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-nested"):
        load_gate_manifest(path)
