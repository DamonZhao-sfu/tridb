"""Arm C's invariants: the same question, a total order, and no silent emptiness.

These need the live Milvus + Neo4j + pgvector stack, so they skip when it is absent
rather than fail -- but when the stack IS up, the parity claim arm C rests on has to
be checkable by running the suite, not by rerunning an ad-hoc script.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.integration

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("pymilvus")
pytest.importorskip("neo4j")

GEM_DSN = "postgresql://127.0.0.1:55432/evotrace_eg"
SCOPE = "evotrace:349117b0"


def _stack_is_up() -> bool:
    import socket

    for port in (19530, 7688, 5434, 55432):
        sock = socket.socket()
        sock.settimeout(2)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False
        finally:
            sock.close()
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _stack_is_up(),
        reason="needs Milvus :19530, Neo4j :7688, pgvector :5434 and GEM :55432",
    ),
]


class _Parent:
    """Just enough of `openevolve.Program` for `GemRetriever._spec`."""

    def __init__(self, fitness: float) -> None:
        self.metrics = {"combined_score": fitness}
        self.id = "stub"


@pytest.fixture(scope="module")
def pair():
    from bench.agent_memory.gem_eg.store import EgStore
    from bench.agent_memory.gem_oe.polyglot_backend import PolyglotBackend
    from bench.agent_memory.gem_oe.retrievers import GemRetriever

    store = EgStore.connect(GEM_DSN)
    store.bootstrap_edge_types()
    gem = GemRetriever(store=store, scope_id=SCOPE)
    poly = PolyglotBackend()
    yield gem, poly
    poly.close()


def _query(gem, poly, task: str, fitness: float, k: int = 5):
    spec = gem._spec(task, _Parent(fitness), k, 1)
    expected = list(gem._engine.run(spec, gem.knobs).ids)
    literal = gem._engine._vectors.get(spec.seed_uid)
    vector = np.asarray(json.loads(literal), dtype=np.float32)
    vector /= np.linalg.norm(vector) or 1.0
    observed, telemetry = poly.reuse_query(spec, vector)
    return expected, observed, telemetry


TASKS = ["math:circle_packing", "math:heilbronn_triangle", "math:heilbronn_convex_13"]


def test_never_returns_an_empty_answer_where_gem_finds_rows(pair) -> None:
    """The E0 retraction in one assertion.

    1,010 of 1,010 cells once returned empty `result_ids` and every one was scored as
    a zero. An empty result from a three-system pipeline is an outage far more often
    than it is a genuinely empty eligible set.
    """
    gem, poly = pair
    for task in TASKS:
        for fitness in (0.0, 0.4, 0.8):
            expected, observed, _ = _query(gem, poly, task, fitness)
            if expected:
                assert observed, f"{task} @ {fitness}: gem found {len(expected)}, polyglot none"


def test_answers_the_same_question_as_gem(pair) -> None:
    """Recall against GEM, judged per artifact.

    Exact uid equality is the wrong bar: 939 of 10,672 nodes duplicate another node's
    code, so two systems can return different uids for one tied slot and still have
    given the same answer.
    """
    gem, poly = pair
    conn = psycopg.connect(GEM_DSN)
    artifact = {
        row[0]: row[1]
        for row in conn.execute("SELECT node_uid, artifact_uid FROM gem_eg_node").fetchall()
    }
    recalls = []
    for task in TASKS:
        for fitness in (0.0, 0.4, 0.8):
            expected, observed, _ = _query(gem, poly, task, fitness)
            if not expected:
                continue
            want = {artifact.get(u, u) for u in expected}
            got = {artifact.get(u, u) for u in observed}
            recalls.append(len(want & got) / len(want))
    assert recalls, "no query produced a comparable answer"
    assert sum(recalls) / len(recalls) >= 0.90


def test_ranking_is_a_total_order(pair) -> None:
    """Identical input must give identical output, ties included.

    Ranking by distance alone left the k-th slot to insertion order, which cost one
    result in five against GEM and read as a retrieval difference.
    """
    gem, poly = pair
    first, second = None, None
    for _ in range(2):
        _, observed, _ = _query(gem, poly, "math:circle_packing", 0.0)
        first, second = second, observed
    assert first == second


def test_first_row_equals_total_because_the_stages_are_serial(pair) -> None:
    """Not a tautology -- it is the measurement.

    Nothing can be returned until stage 3 finishes, so polyglot pays its whole cost
    before its first row. A fused operator does not, and that gap is what arm C is
    for.
    """
    gem, poly = pair
    _, _, telemetry = _query(gem, poly, "math:circle_packing", 0.0)
    assert telemetry["first_row_ms"] == telemetry["total_ms"]
    assert telemetry["ann_ms"] + telemetry["traverse_ms"] <= telemetry["total_ms"]
