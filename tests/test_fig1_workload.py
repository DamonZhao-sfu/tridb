"""Figure 1 EvoTrace workload: replay order, plans, and the live eager/deferred invariants.

The live test runs a tiny replay (one task, 3 sessions x 20 nodes) against a PG16 with
vector + graph_store_am + tjs_pg 0.3.0 + join_order when TRIDB_FIG1_DSN is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DATA = Path("data/evotrace/normalized")
DSN = os.environ.get("TRIDB_FIG1_DSN")
needs_data = pytest.mark.skipif(
    not (DATA / "nodes.jsonl").exists() or not (DATA / "vectors.npz").exists(),
    reason="EvoTrace normalized corpus not present",
)


def test_plans_validate() -> None:
    from bench.agent_memory.fig1.memory import IngestPlan, RetrievePlan

    with pytest.raises(ValueError):
        IngestPlan(mode="lazy")
    with pytest.raises(ValueError):
        RetrievePlan(plan="whatever")
    with pytest.raises(ValueError):
        RetrievePlan(consistency="eventual")


def test_mock_responder_is_content_sensitive() -> None:
    import json

    from bench.agent_memory.fig1.mock import make_responder
    from bench.agent_memory.semantic.ops import SPECS

    respond = make_responder(keep_rate=0.5)
    sysf = SPECS["sem_filter"].system
    verdicts = {
        json.loads(respond(sysf, json.dumps({"condition": "c", "item": i})))["keep"]
        for i in range(64)
    }
    assert verdicts == {True, False}
    sysx = SPECS["sem_extract"].system
    item = {"language": "cpp", "change": "x", "judge": "ACCEPTED", "fitness": 1.0}
    out = json.loads(respond(sysx, json.dumps({"instruction": "i", "item": item})))
    assert "ACCEPTED" in out["text"]


@needs_data
def test_replay_timeline_is_parent_first_and_interleaved() -> None:
    from bench.agent_memory.fig1.corpus import load_corpus

    c = load_corpus(
        DATA, tasks=["ale:ahc008"], max_sessions_per_task=3, max_nodes_per_session=25
    )
    seen: set[str] = set()
    for uid in c.timeline:
        parent = c.nodes[uid].parent_uid
        assert parent is None or parent in seen
        seen.add(uid)
    assert len(c.timeline) == len(c.nodes) == 75
    first_three = {c.nodes[u].session_uid for u in c.timeline[:3]}
    assert len(first_three) == 3  # round-robin across sessions
    assert all(v.shape == (1024,) for v in c.vectors.values())


@needs_data
def test_live_eager_and_deferred_materialize_equal_state(tmp_path: Path) -> None:
    if not DSN:
        pytest.skip("TRIDB_FIG1_DSN not set")
    from bench.agent_memory.fig1 import run as fig1_run

    tag = "pytest" + os.urandom(3).hex()
    rc = fig1_run.main(
        [
            "--dsn",
            DSN,
            "--tasks",
            "ale:ahc008",
            "--max-sessions-per-task",
            "3",
            "--max-nodes-per-session",
            "20",
            "--arms",
            "eager,onread,onread_stale,batch",
            "--batch-every",
            "16",
            "--tag",
            tag,
            "--replay-check",
            "--progress",
            "0",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    import json

    comp = json.loads((tmp_path / "comparison.json").read_text())
    for arm, verdict in comp["mat_equal_to_eager"].items():
        assert verdict["lessons_equal"] and verdict["fixed_by_equal"], arm
        assert verdict["fixed_by_native_equal"] and verdict["native_equals_mirror"], arm
    assert comp["offline_replay"] == {
        "lessons_equal": True,
        "fixed_by_equal": True,
        "fixed_by_native_equal": True,
        "llm_calls": 0,
    }
    agree = comp["output_agreement_vs_eager_planned_mat"]
    # Same state, same plan order: streaming and materialized plans return the same set.
    assert agree["eager/planned"]["exact_ordered"] == 1.0
    # Same logical query over the same candidate universe: predicate order changes cost,
    # never the answer (deterministic LLM, identical state).
    assert agree["eager/naive"]["exact_ordered"] == 1.0
    assert agree["eager/naive"]["same_text"] == 1.0
    eager = json.loads((tmp_path / "eager.json").read_text())
    onread = json.loads((tmp_path / "onread.json").read_text())
    stale = json.loads((tmp_path / "onread_stale.json").read_text())
    # Deferred ingest takes the LLM off the ingest path; stale reads never pay for it.
    assert (
        onread["ingest_llm_during_replay"]["llm_calls"]
        < eager["ingest_llm"]["llm_calls"]
    )
    assert stale["ingest_llm_during_replay"]["llm_calls"] == 0
    # The planner never evaluates more semantic predicates than the naive order.
    r = eager["retrieve"]
    assert r["planned"]["llm"]["n_sem"] <= r["naive"]["llm"]["n_sem"]
    assert r["planned"]["llm"]["n_sem"] <= r["planned_mat"]["llm"]["n_sem"]
