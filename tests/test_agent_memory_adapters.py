"""Unit tests for the LongMemEval and LoCoMo TriDB adapters."""

from __future__ import annotations

from bench.agent_memory.tridbBackend.backend import SearchHit, TriDBMemoryBackend
from bench.agent_memory.tridbBackend.locomo_adapter import (
    adapt_sample,
    build_prompt,
    build_units as build_locomo_units,
    iter_sessions,
)
from bench.agent_memory.tridbBackend.longmemeval_adapter import (
    adapt_entry,
    build_units as build_longmemeval_units,
    retrieval_metrics,
)


def _hit(
    external_id: str,
    content: str = "memory",
    *,
    scope_id: str = "scope",
) -> SearchHit:
    return SearchHit(
        id=1,
        scope_id=scope_id,
        external_id=external_id,
        session_id="S1",
        kind="turn",
        role="A",
        content=content,
        event_time="yesterday",
        event_order=1,
        metadata={},
        score=0.75,
    )


class FakeBackend:
    table = "fake_units"

    def __init__(self, hits):
        self.hits = hits
        self.replacements = []
        self.searches = []

    def replace_scope(self, scope_id, units, *, isolated=True):
        self.replacements.append((scope_id, list(units), isolated))
        return len(units)

    def search(self, scope_id, *, query_text=None, query_embedding=None, k=10):
        self.searches.append((scope_id, query_text, query_embedding, k))
        return self.hits[:k]


def _longmemeval_entry():
    return {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What color was mentioned?",
        "answer": "blue",
        "question_date": "2025-01-03",
        "haystack_session_ids": ["answer_s1", "filler_s2"],
        "haystack_dates": ["2025-01-01", "2025-01-02"],
        "answer_session_ids": ["answer_s1"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "The color was blue.",
                    "has_answer": True,
                },
                {
                    "role": "assistant",
                    "content": "I will remember that.",
                    "has_answer": False,
                },
            ],
            [
                {"role": "user", "content": "Unrelated user text."},
                {"role": "assistant", "content": "Unrelated assistant text."},
            ],
        ],
    }


def test_longmemeval_session_default_matches_user_only_flat_index():
    units = build_longmemeval_units(
        _longmemeval_entry(),
        granularity="session",
    )
    assert [unit.external_id for unit in units] == ["answer_s1", "filler_s2"]
    assert units[0].content == "The color was blue."
    assert "assistant" not in units[0].content.lower()
    assert "has_answer" not in units[0].metadata


def test_longmemeval_role_preserving_session_and_turn_ids():
    entry = _longmemeval_entry()
    sessions = build_longmemeval_units(
        entry,
        granularity="session",
        include_assistant=True,
    )
    assert "assistant: I will remember that." in sessions[0].content

    turns = build_longmemeval_units(
        entry,
        granularity="turn",
        include_assistant=True,
    )
    assert [unit.external_id for unit in turns[:2]] == [
        "answer_s1_1",
        "noans_s1_2",
    ]
    assert turns[1].role == "assistant"


def test_longmemeval_turn_ids_use_original_turn_offsets():
    entry = _longmemeval_entry()
    entry["haystack_sessions"][0].insert(
        1,
        {
            "role": "assistant",
            "content": "interleaved",
            "has_answer": False,
        },
    )
    entry["haystack_sessions"][0].append(
        {
            "role": "user",
            "content": "later user turn",
            "has_answer": False,
        }
    )
    units = build_longmemeval_units(entry, granularity="turn")
    assert [unit.external_id for unit in units[:2]] == [
        "answer_s1_1",
        "noans_s1_4",
    ]


def test_longmemeval_metrics_and_output_shape():
    metrics = retrieval_metrics(
        ["filler_s2", "answer_s1"],
        ["answer_s1", "filler_s2"],
        granularity="session",
    )
    assert metrics["session"]["recall_any@1"] == 0.0
    assert metrics["session"]["recall_any@3"] == 1.0

    backend = FakeBackend(
        [
            _hit("answer_s1", "The color was blue.", scope_id="q1"),
            _hit("filler_s2", "Other.", scope_id="q1"),
        ]
    )
    output = adapt_entry(
        _longmemeval_entry(),
        backend,
        granularity="session",
    )
    assert output["retrieval_results"]["ranked_items"][0]["corpus_id"] == ("answer_s1")
    assert output["retrieval_results"]["backend"]["scope_id"] == "q1"
    assert backend.replacements[0][0] == "q1"
    assert backend.searches == [("q1", "What color was mentioned?", None, 50)]


def test_longmemeval_turn_to_session_ndcg_counts_relevant_turns():
    metrics = retrieval_metrics(
        ["answer_s1_1", "answer_s1_3", "filler_s2_1"],
        ["answer_s1_1", "filler_s2_1", "answer_s1_3"],
        granularity="turn",
    )
    assert metrics["session"]["ndcg_any@3"] == 1.0


def _locomo_sample():
    return {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_2_date_time": "Tuesday",
            "session_2": [
                {"speaker": "B", "dia_id": "D2:1", "text": "Second session."}
            ],
            "session_1_date_time": "Monday",
            "session_1": [
                {
                    "speaker": "A",
                    "dia_id": "D1:1",
                    "text": "I chose blue.",
                    "blip_caption": "a blue square",
                }
            ],
        },
        "qa": [
            {
                "question": "What color did A choose?",
                "answer": "blue",
                "evidence": ["D1:1"],
                "category": 1,
            }
        ],
        "event_summary": {},
        "observation": {},
        "session_summary": {},
    }


def test_locomo_sessions_are_numeric_and_not_hard_coded():
    sessions = list(iter_sessions(_locomo_sample()["conversation"]))
    assert [session[0] for session in sessions] == [1, 2]


def test_locomo_units_preserve_ids_but_do_not_ingest_evidence():
    sample = _locomo_sample()
    units = build_locomo_units(sample)
    assert [unit.external_id for unit in units] == ["D1:1", "D2:1"]
    assert units[0].session_id == "S1"
    assert "(Monday) A said" in units[0].content
    assert "a blue square" in units[0].content
    serialized = " ".join(unit.content + repr(dict(unit.metadata)) for unit in units)
    assert "evidence" not in serialized


def test_locomo_output_matches_official_context_key_contract():
    sample = _locomo_sample()
    backend = FakeBackend(
        [_hit("D1:1", '(Monday) A said, "I chose blue."', scope_id="conv-1")]
    )
    output = adapt_sample(sample, backend, top_k=5)
    qa = output["qa"][0]
    assert qa["tridb_prediction_context"] == ["D1:1"]
    assert qa["tridb_prediction_retrieval"][0]["dia_id"] == "D1:1"
    assert "What color did A choose?" in qa["tridb_prediction_prompt"]
    assert backend.replacements[0][0] == "conv-1"
    assert len(backend.searches) == 1


def test_locomo_prompt_has_explicit_abstention_instruction():
    prompt = build_prompt("Where?", [_hit("D1:1", "Known context")])
    assert "not available" in prompt
    assert "Known context" in prompt


class FakeCursor:
    rowcount = 0

    def __init__(self, rows=()):
        self.rows = list(rows)

    def fetchall(self):
        return list(self.rows)


class FakeSearchConnection:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if "SELECT id, scope_id" in sql:
            return FakeCursor(
                [
                    (
                        7,
                        "scope-a",
                        "D1:1",
                        "S1",
                        "turn",
                        "A",
                        "content",
                        "Monday",
                        1,
                        {},
                        0.25,
                    )
                ]
            )
        return FakeCursor()


def test_backend_search_pushes_scope_predicate_into_sql():
    conn = FakeSearchConnection()
    backend = TriDBMemoryBackend(conn, dim=2)
    hits = backend.search(
        "scope-a",
        query_embedding=[1.0, 0.0],
        k=3,
    )
    search_sql, params = conn.calls[-1]
    assert "WHERE scope_id = %s" in search_sql
    assert params[1] == "scope-a"
    assert params[-1] == 3
    assert hits[0].external_id == "D1:1"
    assert hits[0].score == 0.75


# ---------------------------------------------------------------------------
# Native-graph surface (opt-in graph=True). These cover the wiring off-Postgres;
# the id == vid contract, the graph lift, tenant isolation and scope-filter
# quoting are additionally exercised live against the engine.
# ---------------------------------------------------------------------------


class GraphCursor(FakeCursor):
    """FakeCursor plus the fetchone() the graph helpers use."""

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeGraphConnection:
    """Minimal stand-in that answers the graph helper queries."""

    def __init__(self):
        self.calls = []
        self.allocated = 0

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if "gph_allocated_vids" in sql:
            nid = self.allocated
            self.allocated += 1
            return GraphCursor([(nid,)])
        if "gph_upsert_vertex" in sql:
            return GraphCursor([(params[0],)])  # identity: no drift
        if "register_edge_type" in sql:
            return GraphCursor([(4,)])
        if "SELECT id FROM graph_store.edge_type" in sql:
            return GraphCursor([(4,)])
        if "gph_traverse_typed" in sql:
            return GraphCursor([(2,)])
        if "tjs_open(" in sql:
            return GraphCursor([(2,), (1,)])
        if "tjs_open_candidates_examined" in sql:
            return GraphCursor([(11, 3, False, "term_cond", None, 1)])
        if "SELECT id, scope_id" in sql:
            return GraphCursor(
                [
                    (
                        2,
                        "scope-a",
                        "e:berlin",
                        None,
                        "entity",
                        None,
                        "Berlin",
                        None,
                        0,
                        {},
                    )
                ]
            )
        return GraphCursor()

    def transaction(self):
        connection = self

        class _Txn:
            def __enter__(self):
                connection.calls.append(("BEGIN", None))
                return connection

            def __exit__(self, *exc):
                connection.calls.append(("COMMIT", None))
                return False

        return _Txn()

    # psycopg's sql.Literal(...).as_string(conn) needs these on the adapted conn
    @property
    def info(self):
        raise AttributeError


def test_graph_methods_refuse_when_graph_mode_is_off():
    backend = TriDBMemoryBackend(FakeGraphConnection(), dim=2)
    for call in (
        lambda: backend.link(1, 2, "moved_to"),
        lambda: backend.neighbors(1),
        lambda: backend.graph_stats(),
        lambda: backend.search_fused("s", query_embedding=[1.0, 0.0]),
        lambda: backend.add_units("s", []),
    ):
        try:
            call()
        except RuntimeError as exc:
            assert "graph mode is off" in str(exc)
        else:  # pragma: no cover - the assertion below reports the failure
            raise AssertionError("graph method did not refuse with graph=False")


def test_add_units_allocates_ids_from_the_graph_not_the_identity_sequence():
    conn = FakeGraphConnection()
    backend = TriDBMemoryBackend(conn, dim=2, graph=True)
    from bench.agent_memory.tridbBackend.backend import MemoryUnit

    units = [
        MemoryUnit(scope_id="s", external_id="a", content="a", embedding=[1.0, 0.0]),
        MemoryUnit(scope_id="s", external_id="b", content="b", embedding=[0.0, 1.0]),
    ]
    ids = backend.add_units("s", units)
    assert ids == [0, 1]  # dense vids, so id == vid holds for tjs_open
    inserts = [sql for sql, _ in conn.calls if sql.startswith("INSERT INTO")]
    assert len(inserts) == 2
    assert " id, scope_id," in inserts[0]  # id is written explicitly


def test_add_units_aborts_on_vid_drift():
    class DriftingConnection(FakeGraphConnection):
        def execute(self, sql, params=None):
            if "gph_upsert_vertex" in sql:
                return GraphCursor(
                    [(params[0] + 1,)]
                )  # engine handed back a different vid
            return super().execute(sql, params)

    from bench.agent_memory.tridbBackend.backend import MemoryUnit

    backend = TriDBMemoryBackend(DriftingConnection(), dim=2, graph=True)
    unit = MemoryUnit(scope_id="s", external_id="a", content="a", embedding=[1.0, 0.0])
    try:
        backend.add_units("s", [unit])
    except RuntimeError as exc:
        assert "dense-id drift" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a drifting vid must abort the insert")


def test_neighbors_uses_outgoing_direction_and_projectset_position():
    conn = FakeGraphConnection()
    backend = TriDBMemoryBackend(conn, dim=2, graph=True)
    backend.neighbors(1, rel="moved_to")
    traversal = [sql for sql, _ in conn.calls if "gph_traverse_typed" in sql][-1]
    # (src, type_id, direction=0 out, source_id=-1 unscoped); in/both RAISE today
    assert "%s, %s, 0, -1" in traversal
    assert traversal.startswith("SELECT (e).dst FROM (SELECT")
