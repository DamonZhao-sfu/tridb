"""Semantic-operator layer (plan P1/P4, ADR-0026): cache, ops, planner, stream.

Unit tests run without a database (an in-memory cache with the same API). The live half
runs against a PG with vector + graph_store_am + tjs_pg + join_order when
TRIDB_FIG1_DSN is set (e.g. postgresql://$USER@127.0.0.1:55440/fig1).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pytest

from bench.agent_memory.semantic.cache import SemMeter, input_hash
from bench.agent_memory.semantic.llm import LLMReply, MockLLM, parse_json_reply
from bench.agent_memory.semantic.ops import CacheMiss, SemOps
from bench.agent_memory.semantic.planner import PredicateLeg, SemPlanner

DSN = os.environ.get("TRIDB_FIG1_DSN")


class MemCache:
    """In-memory stand-in for SemCache (same get_many / put_many contract)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, str], Any] = {}

    def get_many(self, op: str, model: str, pv: str, hashes: Any) -> dict[str, Any]:
        return {
            h: self.rows[(op, model, pv, h)]
            for h in hashes
            if (op, model, pv, h) in self.rows
        }

    def put_many(self, op: str, model: str, pv: str, rows: Any) -> int:
        n = 0
        for h, out, *_ in rows:
            self.rows.setdefault((op, model, pv, h), out)
            n += 1
        return n


def keep_if_even(system: str, user: str) -> str:
    item = json.loads(user)["item"]
    return json.dumps({"keep": item % 2 == 0})


def test_cache_hit_avoids_llm_and_counts_n_sem() -> None:
    llm = MockLLM(responder=keep_if_even)
    ops = SemOps(llm, MemCache(), SemMeter())
    assert ops.sem_filter("even", [1, 2, 3, 4]) == [False, True, False, True]
    assert llm.calls == 4
    assert ops.sem_filter("even", [2, 3, 6]) == [True, False, True]
    assert llm.calls == 5  # only 6 was new
    snap = ops.meter.snapshot()
    assert snap["n_sem"] == 7 and snap["llm_calls"] == 5 and snap["cache_hits"] == 2


def test_duplicate_inputs_in_one_batch_cost_one_call() -> None:
    llm = MockLLM(responder=keep_if_even)
    ops = SemOps(llm, MemCache())
    assert ops.sem_filter("even", [2, 2, 2]) == [True, True, True]
    assert llm.calls == 1
    assert ops.meter.calls == 1 and ops.meter.hits == 2


def test_cache_key_covers_model_and_input() -> None:
    cache = MemCache()
    a = SemOps(MockLLM(model="m-a", responder=keep_if_even), cache)
    b = SemOps(MockLLM(model="m-b", responder=keep_if_even), cache)
    a.sem_filter("even", [2])
    b.sem_filter("even", [2])
    assert b.llm.calls == 1  # different model -> different partition
    assert input_hash({"x": 1, "y": 2}) == input_hash({"y": 2, "x": 1})


def test_offline_replay_raises_on_miss_and_serves_hits() -> None:
    cache = MemCache()
    SemOps(MockLLM(responder=keep_if_even), cache).sem_filter("even", [2, 3])
    offline = SemOps(MockLLM(responder=keep_if_even), cache, offline=True)
    assert offline.sem_filter("even", [3, 2]) == [False, True]
    assert offline.llm.calls == 0
    with pytest.raises(CacheMiss):
        offline.sem_filter("even", [5])


def test_rank_and_topk_are_stable_and_bounded() -> None:
    def score(system: str, user: str) -> str:
        return json.dumps({"score": {"a": 3, "b": 9, "c": 3}[json.loads(user)["item"]]})

    ops = SemOps(MockLLM(responder=score), MemCache())
    assert ops.sem_rank("q", ["a", "b", "c"]) == [1, 0, 2]  # tie a/c keeps engine order
    assert ops.sem_topk("q", ["a", "b", "c"], 2) == [1, 0]


def test_bool_parse_fails_closed() -> None:
    ops = SemOps(MockLLM(responder=lambda s, u: "I cannot decide"), MemCache())
    assert ops.sem_filter("x", [1]) == [False]
    assert parse_json_reply('```json\n{"keep": true}\n```') == {"keep": True}


def test_mock_complete_many_charges_waves() -> None:
    slept: list[float] = []
    llm = MockLLM(latency_s=0.5, concurrency=4, sleep=slept.append)
    replies = llm.complete_many([("s", str(i)) for i in range(9)])
    assert len(replies) == 9 and isinstance(replies[0], LLMReply)
    assert slept == [1.5]  # ceil(9 / 4) waves


def test_planner_puts_cheap_selective_predicate_first() -> None:
    planner = SemPlanner(None)
    legs = [
        PredicateLeg("theta_sem", 0.4, 0.5, 0.0),
        PredicateLeg("theta_r", 0.2, 2e-6, 0.0, pushable=True),
    ]
    placement = planner.place(legs, n_in=48)
    assert placement.order == ("theta_r", "theta_sem")
    assert placement.source == "python-ref"
    naive = planner.cost_of(legs, ("theta_sem", "theta_r"), 48)
    assert placement.expected_cost < naive


def test_planner_cache_hits_can_move_semantic_first() -> None:
    planner = SemPlanner(None, cache_cost=0.0)
    legs = [
        PredicateLeg("theta_r", 0.9, 1e-3, 0.0),
        PredicateLeg("theta_sem", 0.1, 0.5, 1.0),  # fully cached -> free and selective
    ]
    assert planner.place(legs, n_in=10).order == ("theta_sem", "theta_r")


def test_batch_size_adapts_to_selectivity() -> None:
    planner = SemPlanner(None)
    assert planner.batch_size(5, 0.5, 4, 64) == 10
    assert planner.batch_size(5, 0.01, 4, 64) == 64
    assert planner.batch_size(1, 1.0, 4, 64) == 4
    assert planner.batch_size(0, 0.5, 4, 64) == 0


# --------------------------------------------------------------------------- live


@pytest.fixture()
def live_conn() -> Any:
    if not DSN:
        pytest.skip("TRIDB_FIG1_DSN not set")
    psycopg = pytest.importorskip("psycopg")
    conn = psycopg.connect(DSN, autocommit=False)
    from bench.agent_memory.semantic.cache import ensure_schema

    ensure_schema(conn)
    conn.commit()
    yield conn
    conn.rollback()
    conn.close()


def test_live_pg_cache_roundtrip(live_conn: Any) -> None:
    from bench.agent_memory.semantic.cache import SemCache

    model = "test-" + uuid.uuid4().hex[:8]
    llm = MockLLM(model=model, responder=keep_if_even)
    ops = SemOps(llm, SemCache(live_conn))
    assert ops.sem_filter("even", [10, 11]) == [True, False]
    live_conn.commit()
    again = SemOps(
        MockLLM(model=model, responder=keep_if_even), SemCache(live_conn), offline=True
    )
    assert again.sem_filter("even", [11, 10]) == [False, True]
    live_conn.execute("DELETE FROM gem_sem_cache WHERE model = %s", (model,))
    live_conn.commit()


def test_live_pending_queue_fifo_and_resolve(live_conn: Any) -> None:
    from bench.agent_memory.semantic.pending import PendingQueue

    q = PendingQueue(live_conn)
    scope = "test:" + uuid.uuid4().hex[:8]
    a = q.enqueue(scope, "lesson", "u2", trigger="on_read", snapshot_seq=20)
    b = q.enqueue(scope, "lesson", "u1", trigger="on_read", snapshot_seq=10)
    assert q.enqueue(scope, "lesson", "u1", trigger="on_read", snapshot_seq=11) is None
    items = q.open_items(scope)
    assert [i.target_ref for i in items] == ["u1", "u2"]
    assert q.count_open(scope, targets=["u1"]) == 1
    assert q.resolve([b], by="on_read") == 1
    assert q.count_open(scope) == 1
    live_conn.execute("DELETE FROM gem_pending WHERE scope_id = %s", (scope,))
    live_conn.commit()
    assert a is not None


def test_live_planner_engine_matches_python_ref(live_conn: Any) -> None:
    planner = SemPlanner(live_conn)
    if not planner.engine_available():
        pytest.skip("join_order 0.2.0 (tridb_place_sem) not installed")
    legs = [
        PredicateLeg("theta_sem", 0.4, 0.5, 0.25),
        PredicateLeg("theta_r", 0.2, 2e-6, 0.0),
        PredicateLeg("theta_g", 0.7, 1e-4, 0.0),
    ]
    engine = planner.place(legs, n_in=48)
    ref = SemPlanner(None).place(legs, n_in=48)
    assert engine.source == "engine" and ref.source == "python-ref"
    assert engine.order == ref.order
    assert engine.expected_cost == ref.expected_cost
    assert planner.batch_size(5, 0.3, 4, 64) == SemPlanner(None).batch_size(
        5, 0.3, 4, 64
    )
