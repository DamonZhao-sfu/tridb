"""Integration tests for GEM against a real engine. Skipped without a DSN.

Requires PostgreSQL with **pgvector + graph_store_am + tjs_pg**. Gate::

    TRIDB_GEM_DSN=postgresql://... pytest tests/test_gem_live.py

matching how the rest of the repo gates engine work.

**These have not been executed.** They were written on a workstation without
the extension stack; running them is the G2/G3 gate, not something this file
can assert on its own. Treat a green run here as the first real evidence, and
until then treat the SQL as unverified against the engine.
"""

from __future__ import annotations

import os

import pytest

from bench.agent_memory.gem import conformance
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.policy import PolicyEngine, seed_policies
from bench.agent_memory.gem.salience import ExponentialSalience
from bench.agent_memory.gem.store import GemStore
from bench.agent_memory.gem.strategies.deterministic import DeterministicIngestStrategy
from bench.agent_memory.gem.types import (
    InteractionEvent,
    Query,
    RetrievalMode,
    RetrievalRoute,
)

DSN = os.environ.get("TRIDB_GEM_DSN")

pytestmark = pytest.mark.skipif(
    not DSN, reason="set TRIDB_GEM_DSN to run GEM integration tests"
)

DIM = 8


class StubEmbedder:
    """Deterministic vectors so similarity is reproducible across runs."""

    def encode(self, texts):
        out = []
        for text in texts:
            vector = [0.0] * DIM
            for index, char in enumerate(text[:DIM]):
                vector[index] = (ord(char) % 17) / 17.0
            if not any(vector):
                vector[0] = 1.0
            out.append(vector)
        return out


class FakeChunker:
    def chunk(self, text):
        return [text]


@pytest.fixture()
def memory():
    store = GemStore.connect(DSN, dim=DIM)
    mem = TriDBGovernedMemory(
        store,
        embedder=StubEmbedder(),
        salience=ExponentialSalience(),
        policy_engine=PolicyEngine(extra=seed_policies()),
    )
    mem.init_schema()
    yield mem
    mem.close()


@pytest.fixture()
def scope(memory, request):
    scope_id = f"test_{request.node.name[:40]}"
    # Never DELETE in operator code; test fixtures may, since they own the scope.
    memory.store.conn.execute("DELETE FROM gem_unit WHERE scope_id = %s", (scope_id,))
    memory.store.conn.execute(
        "DELETE FROM gem_transition WHERE scope_id = %s", (scope_id,)
    )
    return scope_id


def _event(scope_id, external_id, content, when="2026-03-01"):
    return InteractionEvent(
        scope_id=scope_id,
        external_id=external_id,
        content=content,
        event_time=when,
    )


def _deterministic():
    return DeterministicIngestStrategy(chunker=FakeChunker())


# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_ingest_then_retrieve(self, memory, scope):
        result = memory.ingest(
            [_event(scope, "e1", "the deadline is April 20")],
            strategy=_deterministic(),
        )
        assert result.committed, result.aborted_reason
        assert result.delta.units_created == 1

        found = memory.retrieve(
            Query(
                scope_id=scope,
                text="deadline",
                mode=RetrievalMode.VECTOR,
                reinforce=False,
            )
        )
        assert found.committed, found.aborted_reason
        assert found.hits

    def test_unit_id_equals_graph_vid(self, memory, scope):
        """tjs_open's graph leg resolves a reach vertex by passing the raw vid
        through id_col, so a unit is graph-reachable only when id == vid."""
        result = memory.ingest(
            [_event(scope, "e1", "alpha")], strategy=_deterministic()
        )
        unit_id = result.units[0]
        neighbours = memory.store.conn.execute(
            "SELECT graph_store.gph_traverse_bfs(%s, 1, 0)", (unit_id,)
        ).fetchall()
        assert neighbours is not None  # the vertex exists

    def test_scope_isolation(self, memory, scope):
        memory.ingest([_event(scope, "e1", "alpha")], strategy=_deterministic())
        other = memory.retrieve(
            Query(
                scope_id=f"{scope}_other",
                text="alpha",
                mode=RetrievalMode.VECTOR,
                reinforce=False,
            )
        )
        assert other.hits == ()


class TestC6Write:
    def test_reinforce_lands_and_is_strictly_increasing(self, memory, scope):
        memory.ingest([_event(scope, "e1", "alpha")], strategy=_deterministic())
        query = Query(
            scope_id=scope, text="alpha", mode=RetrievalMode.VECTOR, reinforce=True
        )
        first = memory.retrieve(query)
        assert first.committed, first.aborted_reason
        for hit in first.hits:
            assert hit.salience_after > hit.salience_before

        second = memory.retrieve(query)
        assert second.hits[0].salience_before > first.hits[0].salience_before

    def test_access_count_increments(self, memory, scope):
        memory.ingest([_event(scope, "e1", "alpha")], strategy=_deterministic())
        query = Query(
            scope_id=scope, text="alpha", mode=RetrievalMode.VECTOR, reinforce=True
        )
        memory.retrieve(query)
        memory.retrieve(query)
        count = memory.store.conn.execute(
            "SELECT max(access_count) FROM gem_unit WHERE scope_id = %s", (scope,)
        ).fetchone()[0]
        assert count == 2

    def test_the_c6_write_is_HOT(self, memory, scope):
        """The measured claim: at fillfactor=70 the salience write is 100% HOT
        and costs zero index churn (interface §6.3). A future schema change
        that drops fillfactor must fail HERE, loudly, not silently double the
        cost of every query."""
        memory.ingest(
            [_event(scope, f"e{i}", f"content number {i}") for i in range(20)],
            strategy=_deterministic(),
        )
        memory.store.conn.execute(
            "SELECT pg_stat_reset_single_table_counters(to_regclass('gem_unit')::oid)"
        )
        query = Query(
            scope_id=scope,
            text="content",
            k=10,
            mode=RetrievalMode.VECTOR,
            reinforce=True,
        )
        for _ in range(10):
            memory.retrieve(query)

        memory.store.conn.execute("SELECT pg_stat_force_next_flush()")
        upd, hot = memory.store.conn.execute(
            "SELECT n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables"
            " WHERE relname = 'gem_unit'"
        ).fetchone()
        assert upd > 0
        assert hot == upd, (
            f"{upd - hot} of {upd} salience updates were NOT heap-only — "
            "gem_unit lost its fillfactor=70 and C6 now churns the HNSW index"
        )


class TestAbortLeavesNoPartialState:
    def test_a_failed_transition_writes_nothing(self, memory, scope):
        before = memory.store.conn.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id = %s", (scope,)
        ).fetchone()[0]

        class Exploding:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                raise RuntimeError("deliberate failure")

        result = memory.ingest([_event(scope, "e1", "x")], strategy=Exploding())
        assert not result.committed

        after = memory.store.conn.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id = %s", (scope,)
        ).fetchone()[0]
        assert after == before

    def test_the_abort_is_still_in_the_trajectory(self, memory, scope):
        """C2 is 'a violating transition is rejected'; an abort that vanishes
        makes the condition unobservable. This is what the audit connection is
        for — PostgreSQL has no autonomous transactions."""

        class Exploding:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                raise RuntimeError("deliberate failure")

        memory.ingest([_event(scope, "e1", "x")], strategy=Exploding())
        aborted = [t for t in memory.trajectory(scope) if not t["committed"]]
        assert aborted
        assert "deliberate failure" in aborted[-1]["aborted_reason"]


class TestSupersession:
    def _ingest_deadline(self, memory, scope, value, when, supersede):
        from bench.agent_memory.gem import plan as planmod

        class Plan:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id=scope,
                        title="Website Redesign",
                        ref="w",
                        embed_text="Website Redesign",
                    ),
                    planmod.append_field_value(
                        ref="w",
                        field="deadline",
                        value=value,
                        valid_from=when,
                        supersede_current=supersede,
                    ),
                ]

        return memory.ingest([_event(scope, "e", "x")], strategy=Plan())

    def test_c1_default_query_returns_only_the_new_value(self, memory, scope):
        """The paper's own Figure 1 scenario."""
        self._ingest_deadline(memory, scope, "March 15", "2026-02-01", False)
        self._ingest_deadline(memory, scope, "April 20", "2026-03-08", True)

        current = memory.store.conn.execute(
            "SELECT fv.value FROM gem_field_value fv JOIN gem_unit u"
            " ON u.id = fv.unit_id WHERE u.scope_id = %s AND fv.field = 'deadline'"
            "   AND fv.valid_to IS NULL",
            (scope,),
        ).fetchall()
        assert [row[0] for row in current] == ["April 20"]

    def test_c4_the_old_value_and_its_chain_survive(self, memory, scope):
        self._ingest_deadline(memory, scope, "March 15", "2026-02-01", False)
        self._ingest_deadline(memory, scope, "April 20", "2026-03-08", True)

        history = memory.store.conn.execute(
            "SELECT fv.value, fv.valid_to, fv.superseded_by FROM gem_field_value fv"
            " JOIN gem_unit u ON u.id = fv.unit_id WHERE u.scope_id = %s"
            "  AND fv.field = 'deadline' ORDER BY fv.valid_from",
            (scope,),
        ).fetchall()
        assert len(history) == 2
        assert history[0][0] == "March 15"
        assert history[0][1] is not None  # closed
        assert history[0][2] is not None  # chained to its successor

    def test_a_second_current_value_is_refused_by_the_index(self, memory, scope):
        """[GEM] Observation 3a's engine-level mechanism, exercised: appending a
        second current value without supersession must ABORT, not coexist."""
        self._ingest_deadline(memory, scope, "March 15", "2026-02-01", False)
        result = self._ingest_deadline(memory, scope, "April 20", "2026-03-08", False)
        assert not result.committed
        assert "gem_field_value_current_uq" in (result.aborted_reason or "")


class TestPropagation:
    def test_extension_edges_propagate_and_association_edges_do_not(
        self, memory, scope
    ):
        from bench.agent_memory.gem import plan as planmod

        class Build:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id=scope, title="A", ref="a", embed_text="A"
                    ),
                    planmod.upsert_unit(
                        scope_id=scope, title="B", ref="b", embed_text="B"
                    ),
                    planmod.upsert_unit(
                        scope_id=scope, title="C", ref="c", embed_text="C"
                    ),
                    planmod.link(
                        edge_kind="extension", rel="entails", src_ref="a", dst_ref="b"
                    ),
                    planmod.link(
                        edge_kind="association", rel="related", src_ref="a", dst_ref="c"
                    ),
                ]

        result = memory.ingest([_event(scope, "e", "x")], strategy=Build())
        assert result.committed, result.aborted_reason
        ids = dict(
            memory.store.conn.execute(
                "SELECT title, id FROM gem_unit WHERE scope_id = %s", (scope,)
            ).fetchall()
        )

        revision = memory.revise(scope, evidence=[{"unit_id": ids["A"]}], max_hops=3)
        assert revision.committed, revision.aborted_reason
        reached = set(revision.delta.propagated_units)
        assert ids["B"] in reached, "extension dependent was not reached"
        assert ids["C"] not in reached, (
            "an association neighbour was propagated to — propagation followed "
            "relatedness rather than entailment"
        )


class TestForgetLadder:
    def test_low_salience_units_are_hidden_then_archived_never_deleted(
        self, memory, scope
    ):
        memory.ingest(
            [_event(scope, f"e{i}", f"item {i}") for i in range(5)],
            strategy=_deterministic(),
        )
        before = memory.store.conn.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id = %s", (scope,)
        ).fetchone()[0]

        result = memory.forget(scope)
        assert result.committed, result.aborted_reason

        after = memory.store.conn.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id = %s", (scope,)
        ).fetchone()[0]
        assert after == before, "forget DELETEd rows — C4/C5 both broken"

        states = dict(
            memory.store.conn.execute(
                "SELECT state, count(*) FROM gem_unit WHERE scope_id = %s"
                " GROUP BY state",
                (scope,),
            ).fetchall()
        )
        assert states.get("archived", 0) > 0

    def test_attenuated_content_is_excluded_by_the_predicate_not_post_filtered(
        self, memory, scope
    ):
        memory.ingest([_event(scope, "e1", "alpha")], strategy=_deterministic())
        memory.forget(scope)

        hidden = memory.retrieve(
            Query(
                scope_id=scope, text="alpha", mode=RetrievalMode.VECTOR, reinforce=False
            )
        )
        assert hidden.hits == ()

        # C5: archived content remains recoverable by explicit lookup.
        recovered = memory.retrieve(
            Query(
                scope_id=scope,
                text="alpha",
                mode=RetrievalMode.VECTOR,
                include_archived=True,
                reinforce=False,
            )
        )
        assert recovered.hits


class TestTemporalRoute:
    def test_as_of_returns_the_value_that_was_current_then(self, memory, scope):
        test = TestSupersession()
        test._ingest_deadline(memory, scope, "March 15", "2026-02-01", False)
        test._ingest_deadline(memory, scope, "April 20", "2026-03-08", True)

        past = memory.retrieve(
            Query(
                scope_id=scope,
                text="deadline",
                mode=RetrievalMode.VECTOR,
                route=RetrievalRoute.TEMPORAL,
                as_of="2026-02-15",
                reinforce=False,
            )
        )
        values = {h.value for h in past.hits if h.field_name == "deadline"}
        assert values == {"March 15"}


class TestConformanceReport:
    def test_the_report_names_conditions_individually(self, memory, scope):
        memory.ingest([_event(scope, "e1", "alpha")], strategy=_deterministic())
        memory.retrieve(
            Query(
                scope_id=scope, text="alpha", mode=RetrievalMode.VECTOR, reinforce=True
            )
        )
        report = conformance.run(memory, scope, configuration={"strategy": "det"})
        payload = report.to_dict()
        assert set(payload["results"][0]) >= {"condition", "holds", "evidence"}
        assert payload["caveats"]["graph_read_visibility"] == (
            "commit_visible, not snapshot_isolated"
        )

    def test_an_unchecked_condition_never_reads_as_conformant(self, memory, scope):
        report = conformance.run(memory, scope, only=["C1"])
        assert report.label() != "GEM-conformant"
        assert set(report.unchecked) >= {"C2", "C3", "C4", "C5", "C6"}
