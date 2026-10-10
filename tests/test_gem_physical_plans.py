from __future__ import annotations

import time
import types
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.agent_memory.gem_eg.corpus import Corpus
from bench.agent_memory.gem_eg.oracle import (
    Oracle,
    Predicate,
    QuerySpec,
    logical_spec_hash,
)
from bench.agent_memory.gem_eg.physical import PhysicalExecutor, PhysicalPlan
from bench.agent_memory.gem_eg.query import EngineResult
from tools.evotrace.embed import node_query_text, node_text

pytestmark = pytest.mark.unit


def _node(uid: str, fitness: float, *, session: str = "historical") -> dict:
    return {
        "_kind": "node",
        "node_uid": uid,
        "task_uid": "math:t",
        "session_uid": session,
        "is_valid": True,
        "fitness": fitness,
        "failure_class": "clean",
    }


def _corpus() -> Corpus:
    corpus = Corpus()
    corpus.tasks["math:t"] = {"task_uid": "math:t", "domain": "math"}
    corpus.nodes = {
        "seed-near": _node("seed-near", 0.0),
        "seed-far": _node("seed-far", 0.0),
        "near-low": _node("near-low", 1.0),
        "shared": _node("shared", 5.0),
        "far-high": _node("far-high", 10.0),
    }
    corpus.children = defaultdict(list)
    corpus.children["seed-near"] = ["near-low", "shared"]
    corpus.children["seed-far"] = ["far-high", "shared"]
    corpus.vectors = np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    corpus.vector_row = {"seed-near": 0, "seed-far": 1}
    return corpus


def _spec(**overrides) -> QuerySpec:
    values = {
        "query_id": "q",
        "decision_point": "d",
        "seed_uid": "external-parent",
        "relation": "lineage",
        "hops": 3,
        "predicate": Predicate(
            require_valid=True,
            require_fitness=True,
            require_finite_fitness=True,
            fitness_gt=0.5,
            include_tasks=frozenset({"math:t"}),
        ),
        "k": 3,
        "rank_by": "seed_fitness",
        "query_vector": (1.0, 0.0),
        "ann_entry_kind": "node",
        "ann_m_seeds": 2,
    }
    values.update(overrides)
    return QuerySpec(**values)


def test_oracle_uses_best_reaching_seed_then_fitness() -> None:
    result = Oracle(_corpus()).run(_spec())
    assert result.entries == ("seed-near", "seed-far")
    # shared is reached by both seeds and inherits the nearer seed.  Therefore it and
    # near-low precede far-high even though far-high has the largest fitness.
    assert result.topk == ("shared", "near-low", "far-high")
    assert result.distances[0] == result.distances[1] == pytest.approx(0.0)
    assert result.distances[2] == pytest.approx(0.2)


def test_strict_fitness_and_finite_contract() -> None:
    pred = Predicate(fitness_gt=1.0, require_fitness=True, require_finite_fitness=True)
    assert not pred.accepts(_node("equal", 1.0))
    assert pred.accepts(_node("higher", 1.0001))
    assert not pred.accepts(_node("inf", float("inf")))
    sql = pred.to_sql()
    assert "fitness > 1.0" in sql
    assert "Infinity" in sql


def test_logical_hash_excludes_metadata_and_physical_plan() -> None:
    a = _spec(meta={"physical_plan": "vfwd", "note": "one"})
    b = _spec(meta={"physical_plan": "rrev", "note": "two"})
    assert logical_spec_hash(a) == logical_spec_hash(b)
    assert logical_spec_hash(a) != logical_spec_hash(_spec(k=2))


def test_live_and_offline_node_renderers_are_byte_identical() -> None:
    row = (7, "node", "python", "valid", "changed loop", None, "print(1)")
    assert node_text(row) == node_query_text(
        language="python",
        status="valid",
        changes="changed loop",
        error=None,
        payload="print(1)",
    )


def test_forward_outer_seed_order_stops_without_draining_later_seed() -> None:
    executor = object.__new__(PhysicalExecutor)
    executor._started = time.perf_counter()
    pulled: list[int] = []

    def fake_forward(self, spec, seed_vid, result, seen, remaining):
        pulled.append(seed_vid)
        yield from ({1: [("a", 2.0), ("b", 1.0)], 2: [("c", 3.0)]}[seed_vid])

    executor._forward_seed = types.MethodType(fake_forward, executor)
    result = EngineResult()
    spec = _spec(k=2)
    executor._run_forward_entries(
        spec,
        iter([(1, "seed-near", 0.0), (2, "seed-far", 0.2)]),
        result,
        100,
    )
    assert result.ids == ["a", "b"]
    assert pulled == [1]
    assert result.entries == ["seed-near"]
    assert result.termination_reason == "top_k"


def test_early_top_k_explicitly_closes_graph_iterator() -> None:
    executor = object.__new__(PhysicalExecutor)
    executor._started = time.perf_counter()
    closed: list[int] = []

    def fake_forward(self, spec, seed_vid, result, seen, remaining):
        try:
            yield "a", 3.0
            yield "unpulled", 2.0
        finally:
            closed.append(seed_vid)

    executor._forward_seed = types.MethodType(fake_forward, executor)
    result = EngineResult()
    executor._run_forward_entries(
        _spec(k=1), iter([(1, "seed-near", 0.0)]), result, 100
    )
    assert result.ids == ["a"]
    assert closed == [1]


@pytest.mark.parametrize("value", ["vfwd", "rrev", "aivg"])
def test_physical_plan_names_are_stable(value: str) -> None:
    assert PhysicalPlan.parse(value).value == value
