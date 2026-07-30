"""Unit tests for the GEM operators — no PostgreSQL, runs in CI.

Extends the fake-connection pattern of ``test_agent_memory_adapters.py``. What
these can and cannot show is worth stating plainly, because the distinction is
the difference between a test suite and a claim:

* **CAN** cover: strategy planning, the ``validate_plan`` gate matrix, salience
  monotonicity as a property, the policy registry, the edge-orientation
  invariant, ladder arithmetic, the transition envelope's commit/abort/audit
  behaviour, and the SQL each operator emits.
* **CANNOT** cover: that the SQL is correct against a real engine. The partial
  unique index, HOT update behaviour, ``tjs_open`` fusion, and graph visibility
  are all engine properties. Those live in ``test_gem_live.py`` and
  ``test_gem_conformance.py``, both skip-gated on ``TRIDB_GEM_DSN``.
"""

from __future__ import annotations

import math
import re

import psycopg.adapt as psycopg_adapt
import pytest
from psycopg import adapters as psycopg_adapters

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.ingest import IngestOperator, validate_plan
from bench.agent_memory.gem.policy import (
    ACTIONS,
    CONDITIONS,
    PolicyEngine,
    seed_policies,
)
from bench.agent_memory.gem.retrieve import (
    FUSED_ITERATIVE_SCAN,
    VECTOR_ITERATIVE_SCAN,
    RetrieveOperator,
    build_prompt_block,
)
from bench.agent_memory.gem.salience import ExponentialSalience
from bench.agent_memory.gem.store import (
    NATIVE_EDGE_KINDS,
    GemStore,
    PolicyViolation,
    SingleWriterViolation,
    vec_literal,
)
from bench.agent_memory.gem.strategies.deterministic import (
    EMBEDDING_SOURCE,
    DeterministicIngestStrategy,
)
from bench.agent_memory.gem.types import (
    EdgeKind,
    InteractionEvent,
    Policy,
    PolicyEvent,
    Query,
    RetrievalMode,
    RetrievalRoute,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeTransaction:
    """Mimics psycopg's ``conn.transaction()`` block, including rollback."""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.events.append(("BEGIN", None))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.conn.events.append(("ROLLBACK" if exc_type else "COMMIT", None))
        if exc_type:
            self.conn.rolled_back = True
        return False  # never swallow


class FakeConn:
    """Records every statement and answers from a pattern->rows table.

    Matching is on the first regex whose pattern is found in the statement, so
    tests only describe the queries whose answers they care about.

    Carries psycopg's global adapter registry so ``sql.Composable.as_string``
    renders against this fake exactly as it does against a real connection —
    which is what lets the predicate-quoting tests below be meaningful rather
    than testing a different code path from production.
    """

    connection = None
    adapters = psycopg_adapt.AdaptersMap(psycopg_adapters)

    def __init__(self, responses=None):
        self.statements: list[tuple[str, tuple]] = []
        self.events: list[tuple[str, object]] = []
        self.responses = list(responses or [])
        self.rolled_back = False
        self.closed = False

    def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), tuple(params or ())))
        for pattern, rows in self.responses:
            if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
                return FakeCursor(rows() if callable(rows) else rows)
        return FakeCursor([])

    def transaction(self):
        return FakeTransaction(self)

    def close(self):
        self.closed = True

    # -- assertions helpers
    def sql_matching(self, pattern):
        return [s for s, _ in self.statements if re.search(pattern, s, re.I | re.S)]

    def ran(self, pattern) -> bool:
        return bool(self.sql_matching(pattern))


class FakeEmbedder:
    """Deterministic 4-dim vectors; counts calls so batching can be asserted."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t) % 7), 1.0, 0.0, 0.5] for t in texts]


class FakeChunker:
    def __init__(self, chunks_per_text=2):
        self.chunks_per_text = chunks_per_text

    def chunk(self, text):
        return [f"{text}::chunk{i}" for i in range(self.chunks_per_text)]


def _store(responses=None, *, audit=None, hook=None) -> GemStore:
    return GemStore(
        FakeConn(responses), audit_conn=audit, dim=4, postcondition_hook=hook
    )


def _event(external_id="e1", content="hello world", scope="s1") -> InteractionEvent:
    return InteractionEvent(
        scope_id=scope,
        external_id=external_id,
        content=content,
        session_id="sess",
        role="user",
        event_time="2026-03-08",
    )


# ---------------------------------------------------------------------------
# salience — the C6 property
# ---------------------------------------------------------------------------


class TestSalience:
    def test_reinforce_is_strictly_increasing_for_every_rank(self):
        """C6: repeated retrieval STRICTLY reduces eligibility for attenuation.

        The last-ranked hit is the case a missing floor would break, so it is
        swept explicitly rather than sampled.
        """
        policy = ExponentialSalience()
        for k in (1, 2, 5, 20):
            for rank in range(k):
                for current in (0.0, 0.01, 1.0, 12.5, 1e3):
                    assert policy.reinforce(current, rank=rank, k=k) > current

    def test_zero_floor_is_rejected(self):
        with pytest.raises(ValueError, match="STRICTLY increasing"):
            ExponentialSalience(floor=0.0)

    def test_repeated_retrieval_outranks_a_single_retrieval(self):
        """The C5/C6 coupling: retention is relevance-driven, not age-driven."""
        policy = ExponentialSalience()
        often = single = 0.0
        for _ in range(5):
            often = policy.reinforce(often, rank=0, k=10)
        single = policy.reinforce(single, rank=0, k=10)
        assert often > single

    def test_decay_is_monotone_and_bounded(self):
        policy = ExponentialSalience()
        assert policy.decay(1.0, seconds_idle=0.0) == pytest.approx(1.0)
        assert policy.decay(1.0, seconds_idle=86_400) == pytest.approx(0.5, abs=1e-9)
        assert policy.decay(1.0, seconds_idle=1e9) < 1e-6

    def test_ladder_must_be_ordered(self):
        with pytest.raises(ValueError, match="ordered"):
            ExponentialSalience(theta_archive=0.9, theta_remove=0.2, theta_summary=0.5)

    @pytest.mark.parametrize(
        "salience,expected",
        [(0.9, None), (0.4, "compressed"), (0.1, "hidden"), (0.01, "archived")],
    )
    def test_rung_selects_the_most_severe_applicable(self, salience, expected):
        assert ExponentialSalience().rung(salience) == expected

    def test_rank_out_of_range_is_rejected(self):
        with pytest.raises(ValueError):
            ExponentialSalience().reinforce(0.0, rank=5, k=5)


# ---------------------------------------------------------------------------
# validate_plan — one case per gate
# ---------------------------------------------------------------------------


class TestValidateGates:
    def test_a_well_formed_plan_has_no_rejections(self):
        ops = [
            planmod.upsert_unit(scope_id="s", title="T", ref="a", embed_text="x"),
            planmod.append_field_value(
                ref="a", field="f", value="v", valid_from="2026-01-01"
            ),
            planmod.link(edge_kind="extension", rel="r", src_ref="a", dst=7),
        ]
        assert validate_plan(ops) == []

    def test_unknown_kind_is_a_schema_rejection(self):
        rejects = validate_plan([{"kind": "teleport"}])
        assert [r["gate"] for r in rejects] == ["schema"]

    def test_unit_without_a_vector_source_is_rejected(self):
        op = planmod.upsert_unit(scope_id="s", title="T", ref="a")
        assert validate_plan([op])[0]["gate"] == "schema"

    def test_field_with_no_target_is_a_referential_rejection(self):
        op = planmod.append_field_value(field="f", value="v", valid_from="t")
        assert validate_plan([op])[0]["gate"] == "referential"

    def test_ref_not_created_in_this_plan_is_a_dangling_vertex(self):
        """Gate 4: every referenced unit must have a vid after apply."""
        op = planmod.append_field_value(
            ref="ghost", field="f", value="v", valid_from="t"
        )
        rejects = validate_plan([op])
        assert rejects[0]["gate"] == "dangling_vertex"
        assert "ghost" in rejects[0]["detail"]

    def test_edge_kind_outside_the_two_native_kinds_is_an_enum_rejection(self):
        op = planmod.link(edge_kind="mentions", rel="r", src=1, dst=2)
        assert validate_plan([op])[0]["gate"] == "enum"

    def test_link_to_an_undefined_ref_is_rejected(self):
        op = planmod.link(edge_kind="association", rel="r", src=1, dst_ref="nope")
        assert validate_plan([op])[0]["gate"] == "dangling_vertex"

    def test_rejection_targets_the_op_not_the_run(self):
        """A single bad extraction must not discard the whole batch."""
        good = planmod.upsert_unit(scope_id="s", title="T", ref="a", embed_text="x")
        bad = planmod.append_field_value(
            ref="ghost", field="f", value="v", valid_from="t"
        )
        rejects = validate_plan([good, bad])
        assert len(rejects) == 1 and rejects[0]["index"] == 1

    def test_split_topic_needs_a_unit_and_fields(self):
        gates = {
            r["gate"]
            for r in validate_plan(
                [{"kind": planmod.SPLIT_TOPIC, "unit_id": None, "fields": []}]
            )
        }
        assert gates == {"referential", "schema"}

    def test_plan_helper_rejects_an_ambiguous_embedding_source(self):
        with pytest.raises(ValueError, match="not both"):
            planmod.upsert_unit(
                scope_id="s", title="T", embedding=[0.0], embed_text="x"
            )


# ---------------------------------------------------------------------------
# DeterministicIngest — the G2 regression gate's strategy
# ---------------------------------------------------------------------------


class TestDeterministicStrategy:
    def test_emits_one_unit_and_one_content_field_per_chunk(self):
        strategy = DeterministicIngestStrategy(chunker=FakeChunker(3))
        ops = strategy.plan([_event()], view=None)
        units = [o for o in ops if o["kind"] == planmod.UPSERT_UNIT]
        fields = [o for o in ops if o["kind"] == planmod.APPEND_FIELD_VALUE]
        assert len(units) == 3 and len(fields) == 3
        assert {f["field"] for f in fields} == {"content"}

    def test_creates_no_edges_and_calls_no_model(self):
        """Paradigm II is chunk/index only — edges or an LLM would be a
        different construction form and a different [AM] row."""
        strategy = DeterministicIngestStrategy(chunker=FakeChunker())
        ops = strategy.plan([_event()], view=None)
        assert not [o for o in ops if o["kind"] == planmod.LINK]
        assert strategy.cost == {}

    def test_records_embedding_source_so_runs_are_never_pooled_silently(self):
        strategy = DeterministicIngestStrategy(chunker=FakeChunker(1))
        unit = strategy.plan([_event()], view=None)[0]
        assert unit["metadata"]["embedding_source"] == EMBEDDING_SOURCE
        assert unit["embed_text"] and unit["embedding"] is None

    def test_plan_is_valid_by_construction(self):
        strategy = DeterministicIngestStrategy(chunker=FakeChunker(4))
        ops = strategy.plan([_event("a"), _event("b")], view=None)
        assert validate_plan(ops) == []

    def test_content_field_never_supersedes(self):
        """Deterministic construction is blind to existing state ([GEM] Failure
        ②) — appending beside an outdated value is the faithful behaviour."""
        strategy = DeterministicIngestStrategy(chunker=FakeChunker(2))
        ops = strategy.plan([_event()], view=None)
        fields = [o for o in ops if o["kind"] == planmod.APPEND_FIELD_VALUE]
        assert all(f["supersede_current"] is False for f in fields)


# ---------------------------------------------------------------------------
# store — the transition envelope
# ---------------------------------------------------------------------------


class TestTransitionEnvelope:
    def test_commit_logs_on_the_main_connection_inside_the_transaction(self):
        store = _store([(r"^SELECT nextval", [(42,)]), (r"RETURNING t", [(42,)])])
        with store.transition("ingest", "s1", "construction") as tx:
            tx.execute("SELECT 1")
        assert tx.transition_id == 42
        order = [s for s, _ in store.conn.events]
        assert order == ["BEGIN", "COMMIT"]
        # logged BEFORE the commit, so the log is atomic with the state
        assert store.conn.ran(r"INSERT INTO gem_transition")

    def test_abort_is_logged_on_the_audit_connection_after_rollback(self):
        """C2 is 'a violating transition is rejected' — an abort that vanishes
        from the trajectory would make the condition unobservable."""
        audit = FakeConn([(r"RETURNING t", [(7,)])])
        store = _store([(r"^SELECT nextval", [(7,)])], audit=audit)
        with pytest.raises(RuntimeError, match="boom"):
            with store.transition("ingest", "s1", "construction"):
                raise RuntimeError("boom")

        assert store.conn.rolled_back
        assert not store.conn.ran(r"INSERT INTO gem_transition")
        logged = audit.sql_matching(r"INSERT INTO gem_transition")
        assert len(logged) == 1
        # column order: t, scope_id, operator, phase, committed, aborted_reason
        params = audit.statements[0][1]
        assert params[0] == 7  # the id reserved before the body ran
        assert params[4] is False  # committed
        assert "boom" in params[5]  # aborted_reason

    def test_audit_logging_never_masks_the_original_error(self):
        class ExplodingConn(FakeConn):
            def execute(self, sql, params=None):
                raise OSError("audit connection is down")

        store = _store(audit=ExplodingConn())
        with pytest.raises(ValueError, match="the real failure"):
            with store.transition("ingest", "s1", "construction"):
                raise ValueError("the real failure")

    def test_postcondition_failure_aborts_and_is_logged_as_a_policy_violation(self):
        def hook(tx):
            raise PolicyViolation("bound-active-state", "too many active units")

        audit = FakeConn([(r"RETURNING t", [(1,)])])
        store = _store([(r"^SELECT nextval", [(1,)])], audit=audit, hook=hook)
        with pytest.raises(PolicyViolation):
            with store.transition("ingest", "s1", "construction") as tx:
                tx.execute("INSERT INTO gem_unit DEFAULT VALUES")
        assert store.conn.rolled_back
        assert "bound-active-state" in audit.statements[0][1][5]

    def test_c5_active_counts_are_sampled_on_every_transition(self):
        store = _store(
            [
                (r"count\(\*\) FROM gem_unit", [(11,)]),
                (r"count\(\*\) FROM gem_field_value", [(37,)]),
                (r"RETURNING t", [(1,)]),
            ]
        )
        with store.transition("retrieve", "s1", "retrieval") as tx:
            pass
        assert (tx.delta.active_units, tx.delta.active_fields) == (11, 37)

    def test_db_statements_are_counted_structurally(self):
        store = _store([(r"RETURNING t", [(1,)])])
        with store.transition("ingest", "s1", "construction") as tx:
            tx.execute("SELECT 1")
            tx.execute("SELECT 2")
        # 1 id reservation + 2 explicit + 2 from the C5 sample
        assert tx.meter.db_statements == 5


class TestWriterLockAndAllocator:
    def test_writer_lock_is_taken_once_and_is_transaction_scoped(self):
        store = _store([(r"RETURNING t", [(1,)])])
        with store.transition("ingest", "s1", "construction") as tx:
            tx.lock_writer()
            tx.lock_writer()
        locks = store.conn.sql_matching(r"pg_advisory_xact_lock")
        assert len(locks) == 1

    def test_allocate_vertex_takes_the_lock_and_returns_the_vid(self):
        store = _store(
            [
                (r"gph_allocated_vids", [(5,)]),
                (r"gph_upsert_vertex", [(5,)]),
                (r"RETURNING t", [(1,)]),
            ]
        )
        with store.transition("ingest", "s1", "construction") as tx:
            assert tx.allocate_vertex() == 5
        assert store.conn.ran(r"pg_advisory_xact_lock")

    def test_null_vid_raises_a_named_contract_violation_not_a_typeerror(self):
        """Under REPEATABLE READ the re-SELECT can miss a concurrent winner."""
        store = _store(
            [(r"gph_allocated_vids", [(5,)]), (r"gph_upsert_vertex", [(None,)])]
        )
        with pytest.raises(SingleWriterViolation, match="returned NULL"):
            with store.transition("ingest", "s1", "construction") as tx:
                tx.allocate_vertex()

    def test_vid_drift_raises(self):
        store = _store(
            [(r"gph_allocated_vids", [(5,)]), (r"gph_upsert_vertex", [(9,)])]
        )
        with pytest.raises(SingleWriterViolation, match="dense-id drift"):
            with store.transition("ingest", "s1", "construction") as tx:
                tx.allocate_vertex()


class TestEdgeKinds:
    def test_exactly_two_native_kinds_are_registered(self):
        """Registering a native type per relation name would make 'all
        extension edges regardless of rel' inexpressible against the engine's
        single-id equality filter (interface §6.1)."""
        assert NATIVE_EDGE_KINDS == ("extension", "association")

    def test_an_unregistered_kind_is_refused(self):
        store = _store()
        with pytest.raises(ValueError, match="exactly"):
            store.edge_type_id("mentions")

    def test_link_writes_topology_to_the_am_and_metadata_to_gem_edge(self):
        store = _store([(r"register_edge_type", [(1,)]), (r"RETURNING t", [(1,)])])
        with store.transition("ingest", "s1", "construction") as tx:
            edge = store.link(tx, 1, 2, kind=EdgeKind.EXTENSION, rel="schedules")
        assert store.conn.ran(r"gph_insert_edge")
        assert store.conn.ran(r"INSERT INTO gem_edge")
        assert edge.kind is EdgeKind.EXTENSION and edge.rel == "schedules"
        assert tx.delta.edges_created == 1

    def test_extension_orientation_is_src_entails_dst(self):
        """A --extension--> B means 'a change in A entails re-evaluating B',
        because revision can only walk OUT-edges (ADR-0016 defers reverse
        adjacency). Getting this backwards makes C3 propagation find nothing."""
        store = _store([(r"register_edge_type", [(1,)]), (r"RETURNING t", [(1,)])])
        with store.transition("ingest", "s1", "construction") as tx:
            store.link(tx, 10, 20, kind="extension", rel="schedules")
        insert = store.conn.sql_matching(r"gph_insert_edge")[0]
        params = next(p for s, p in store.conn.statements if "gph_insert_edge" in s)
        assert params[:2] == (10, 20), "src must be the entailing unit"
        assert "gph_insert_edge" in insert


# ---------------------------------------------------------------------------
# ingest operator
# ---------------------------------------------------------------------------


class TestIngestOperator:
    def _responses(self, next_vid=1):
        counter = {"n": next_vid}

        def allocate(_=None):
            value = counter["n"]
            counter["n"] += 1
            return [(value,)]

        return [
            (r"gph_allocated_vids", allocate),
            (r"gph_upsert_vertex", lambda: [(counter["n"] - 1,)]),
            (r"SELECT id FROM gem_unit WHERE scope_id", []),
            (r"UPDATE gem_field_value SET valid_to", [(99,)]),
            (r"INSERT INTO gem_field_value", [(100,)]),
            (r"register_edge_type", [(1,)]),
            (r"RETURNING t", [(1,)]),
        ]

    def test_embeddings_are_resolved_in_ONE_batched_call(self):
        """Re-embedding is 0% HOT and grows the relation ~369 B per update
        (measured, §6.3) — per-unit embedding would make ingest the expensive
        path for no reason."""
        embedder = FakeEmbedder()
        store = _store(self._responses())
        operator = IngestOperator(store, embedder=embedder)
        strategy = DeterministicIngestStrategy(chunker=FakeChunker(5))

        result = operator.ingest([_event()], strategy=strategy)

        assert result.committed, result.aborted_reason
        assert len(embedder.calls) == 1
        assert len(embedder.calls[0]) == 5

    def test_supersession_writes_the_full_c1_c4_sequence(self):
        store = _store(self._responses())
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class Supersedes:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id="s1", title="Website", ref="w", embed_text="x"
                    ),
                    planmod.append_field_value(
                        ref="w",
                        field="deadline",
                        value="April 20",
                        valid_from="2026-03-08",
                        supersede_current=True,
                    ),
                ]

        result = operator.ingest([_event()], strategy=Supersedes())
        assert result.committed, result.aborted_reason
        # close the old value, insert the new, then CHAIN them (C4)
        assert store.conn.ran(r"UPDATE gem_field_value SET valid_to")
        assert store.conn.ran(r"INSERT INTO gem_field_value")
        assert store.conn.ran(r"SET superseded_by")
        assert result.delta.values_superseded == 1

    def test_no_supersession_means_no_valid_to_update(self):
        """The partial unique index is the enforcement mechanism; ingest must
        not add an application-level duplicate check that hides it."""
        store = _store(self._responses())
        operator = IngestOperator(store, embedder=FakeEmbedder())
        operator.ingest(
            [_event()], strategy=DeterministicIngestStrategy(chunker=FakeChunker(1))
        )
        assert not store.conn.ran(r"UPDATE gem_field_value SET valid_to")

    def test_rejections_are_reported_and_the_rest_still_applies(self):
        store = _store(self._responses())
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class PartlyBad:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id="s1", title="Good", ref="g", embed_text="x"
                    ),
                    planmod.append_field_value(
                        ref="ghost", field="f", value="v", valid_from="t"
                    ),
                ]

        result = operator.ingest([_event()], strategy=PartlyBad())
        assert result.committed
        assert len(result.rejected) == 1
        assert result.rejected[0]["gate"] == "dangling_vertex"
        assert result.delta.units_created == 1

    def test_exceeding_the_capability_floor_aborts_the_whole_run(self):
        """[AM] §4.4: a structural failure is a FAILED CONFIGURATION, not an
        accuracy datapoint."""
        store = _store(self._responses(), audit=FakeConn([(r"RETURNING t", [(1,)])]))
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class MostlyBad:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.append_field_value(
                        ref="ghost", field="f", value="v", valid_from="t"
                    ),
                    planmod.upsert_unit(
                        scope_id="s1", title="Ok", ref="a", embed_text="x"
                    ),
                ]

        result = operator.ingest(
            [_event()], strategy=MostlyBad(), max_rejection_rate=0.1
        )
        assert not result.committed
        assert "FAILED CONFIGURATION" in result.aborted_reason
        assert store.conn.rolled_back

    def test_extension_links_flag_dependents_for_revision(self):
        """C3 hook ([GEM] Alg. 1 line 4). Ingest FLAGS; it never propagates —
        propagation is a revise decision."""
        responses = self._responses() + [
            (r"SELECT DISTINCT dst FROM gem_edge", [(20,), (21,)])
        ]
        store = _store(responses)
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class Links:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id="s1", title="A", ref="a", embed_text="x"
                    ),
                    planmod.upsert_unit(
                        scope_id="s1", title="B", ref="b", embed_text="y"
                    ),
                    planmod.link(
                        edge_kind="extension", rel="schedules", src_ref="a", dst_ref="b"
                    ),
                ]

        result = operator.ingest([_event()], strategy=Links())
        assert result.committed, result.aborted_reason
        assert store.conn.ran(r"needs_revision")
        assert set(result.delta.propagated_units) == {20, 21}

    def test_association_links_do_not_flag_anything(self):
        """Association edges expand retrieval context and never propagate."""
        store = _store(self._responses())
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class Assoc:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id="s1", title="A", ref="a", embed_text="x"
                    ),
                    planmod.upsert_unit(
                        scope_id="s1", title="B", ref="b", embed_text="y"
                    ),
                    planmod.link(
                        edge_kind="association",
                        rel="related",
                        src_ref="a",
                        dst_ref="b",
                    ),
                ]

        result = operator.ingest([_event()], strategy=Assoc())
        assert result.committed
        assert result.delta.propagated_units == ()

    def test_the_result_carries_the_transition_id_and_the_c5_counts(self):
        """Regression: transition() samples C5, evaluates P_t and writes the log
        during __exit__. Capturing the result INSIDE the with body returned
        transition_id=None, policies_evaluated=(), and active_units=None on
        every operator, while the gem_transition row itself was correct — so
        any caller reading C5 or the policy list off the result was misled."""
        responses = self._responses() + [
            (r"^SELECT nextval", [(77,)]),
            (r"count\(\*\) FROM gem_unit", [(5,)]),
            (r"count\(\*\) FROM gem_field_value", [(9,)]),
        ]
        store = _store(responses)
        operator = IngestOperator(store, embedder=FakeEmbedder())
        result = operator.ingest(
            [_event()], strategy=DeterministicIngestStrategy(chunker=FakeChunker(1))
        )
        assert result.committed, result.aborted_reason
        assert result.transition_id == 77
        assert result.delta.active_units == 5
        assert result.delta.active_fields == 9

    def test_provenance_cites_the_reserved_transition_id(self):
        """C4's gem_field_value.transition_id is documented as "the transition
        that committed it". The id is reserved at the top of the envelope so a
        value written mid-body can actually cite it."""
        responses = self._responses() + [(r"^SELECT nextval", [(77,)])]
        store = _store(responses)
        operator = IngestOperator(store, embedder=FakeEmbedder())
        operator.ingest(
            [_event()], strategy=DeterministicIngestStrategy(chunker=FakeChunker(1))
        )
        params = next(
            p
            for sql, p in store.conn.statements
            if "INSERT INTO gem_field_value" in sql
        )
        assert params[-1] == 77, "the field value did not record its transition"

    def test_a_rejected_upsert_also_rejects_its_dependents(self):
        """validate_plan promises per-OP rejection. If a rejected upsert still
        counted as defining its ref, the dependent op would pass the gate,
        reach apply, fail to resolve, and roll the WHOLE batch back."""
        ops = [
            planmod.upsert_unit(scope_id="s1", title="", ref="a", embed_text="x"),
            planmod.append_field_value(ref="a", field="f", value="v", valid_from="t"),
        ]
        gates = {r["gate"] for r in validate_plan(ops)}
        assert gates == {"schema", "dangling_vertex"}

    def test_that_batch_still_commits_rather_than_aborting(self):
        store = _store(self._responses())
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class BadUpsert:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id="s1", title="", ref="a", embed_text="x"
                    ),
                    planmod.append_field_value(
                        ref="a", field="f", value="v", valid_from="t"
                    ),
                    planmod.upsert_unit(
                        scope_id="s1", title="Good", ref="g", embed_text="y"
                    ),
                ]

        result = operator.ingest([_event()], strategy=BadUpsert())
        assert result.committed, result.aborted_reason
        assert len(result.rejected) == 2
        assert result.delta.units_created == 1

    def test_split_topic_ops_are_applied_not_silently_dropped(self):
        """The agentic split_topic tool emits these. Without an apply branch the
        tool reported success to the model and did nothing."""
        responses = self._responses() + [(r"^SELECT nextval", [(1,)])]
        store = _store(responses)
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class Splits:
            name = "test"
            cost: dict = {}

            def plan(self, events, view):
                return [
                    planmod.split_topic(
                        unit_id=3, fields=["deadline"], new_title="Deadlines"
                    )
                ]

        result = operator.ingest([_event()], strategy=Splits())
        assert result.committed, result.aborted_reason
        assert store.conn.ran(r"INSERT INTO gem_unit")
        assert store.conn.ran(r"UPDATE gem_field_value SET unit_id")
        # parent -> child, so revision walks out-edges
        params = next(p for sql, p in store.conn.statements if "gph_insert_edge" in sql)
        assert params[0] == 3, "the split edge must be oriented parent -> child"

    def test_a_multi_scope_batch_is_refused(self):
        operator = IngestOperator(_store(), embedder=FakeEmbedder())
        with pytest.raises(ValueError, match="exactly one scope"):
            operator.ingest(
                [_event(scope="a"), _event(scope="b")],
                strategy=DeterministicIngestStrategy(chunker=FakeChunker()),
            )

    def test_strategy_llm_cost_is_folded_into_the_transition_meter(self):
        store = _store(self._responses())
        operator = IngestOperator(store, embedder=FakeEmbedder())

        class Costly:
            name = "llm_mediated"
            cost = {"llm_calls": 3, "prompt_tokens": 120, "completion_tokens": 45}

            def plan(self, events, view):
                return [
                    planmod.upsert_unit(
                        scope_id="s1", title="T", ref="a", embed_text="x"
                    )
                ]

        result = operator.ingest([_event()], strategy=Costly())
        assert result.cost.llm_calls == 3
        assert result.cost.prompt_tokens == 120


# ---------------------------------------------------------------------------
# retrieve operator
# ---------------------------------------------------------------------------


class TestRetrieveOperator:
    def _responses(self):
        return [
            (
                r"SELECT id, title, state, salience FROM gem_unit",
                [(1, "T1", "active", 0.0)],
            ),
            (r"SELECT unit_id, field, value", [(1, "content", "hello")]),
            (r"SELECT id, salience FROM gem_unit", [(1, 0.0)]),
            (r"SELECT id FROM gem_unit", [(1,)]),
            (r"RETURNING t", [(1,)]),
        ]

    def _operator(self, responses=None):
        store = _store(responses if responses is not None else self._responses())
        return RetrieveOperator(store, embedder=FakeEmbedder()), store

    def test_reinforce_false_performs_no_write(self):
        """The ablation switch: a read-only retrieval is NOT GEM-conformant, but
        Paradigm I/II reproductions must stay faithful to their originals."""
        operator, store = self._operator()
        result = operator.retrieve(
            Query(
                scope_id="s1",
                text="q",
                mode=RetrievalMode.VECTOR,
                reinforce=False,
            )
        )
        assert result.committed, result.aborted_reason
        assert not store.conn.ran(r"UPDATE gem_unit SET salience")
        assert result.delta.salience_updates == 0

    def test_reinforce_true_commits_the_c6_write_in_the_same_transaction(self):
        operator, store = self._operator()
        result = operator.retrieve(
            Query(scope_id="s1", text="q", mode=RetrievalMode.VECTOR, reinforce=True)
        )
        assert result.committed, result.aborted_reason
        assert store.conn.ran(r"UPDATE gem_unit SET salience")
        assert store.conn.ran(r"access_count = access_count \+ 1")
        assert store.conn.ran(r"UPDATE gem_field_value SET salience")
        # inside the transaction, before COMMIT
        events = [e for e, _ in store.conn.events]
        assert events == ["BEGIN", "COMMIT"]
        assert result.delta.salience_updates == 1

    def test_the_active_state_filter_is_pushed_into_the_predicate(self):
        """Attenuated content must cost nothing, not be fetched and discarded."""
        operator, _ = self._operator()
        text = operator.predicate(Query(scope_id="s1")).as_string(operator.store.conn)
        assert "state = 'active'" in text
        assert "'s1'" in text

    def test_include_archived_drops_the_active_filter(self):
        """C5: archived content remains recoverable by explicit lookup."""
        operator, _ = self._operator()
        text = operator.predicate(
            Query(scope_id="s1", include_archived=True)
        ).as_string(operator.store.conn)
        assert "state = 'active'" not in text

    def test_scope_is_literal_quoted(self):
        operator, _ = self._operator()
        text = operator.predicate(Query(scope_id="o'brien")).as_string(
            operator.store.conn
        )
        assert "'o''brien'" in text

    def test_fused_and_vector_use_different_iterative_scan_settings(self):
        """The engine refuses strict_order on the fused path outright, so these
        are different operating points and must never be pooled (§6.1a)."""
        assert FUSED_ITERATIVE_SCAN == "relaxed_order"
        assert VECTOR_ITERATIVE_SCAN == "strict_order"

        operator, store = self._operator()
        operator.retrieve(
            Query(scope_id="s1", text="q", mode=RetrievalMode.VECTOR, reinforce=False)
        )
        assert store.conn.ran(r"SET hnsw.iterative_scan = strict_order")

    def test_fused_records_the_engine_probes_verbatim(self):
        """Censoring travels: a censored run is a different operating point,
        not a faster exact one."""
        responses = self._responses() + [
            (r"SELECT t FROM tjs_open", [(1,)]),
            (
                r"tjs_open_candidates_examined",
                [(120, 8, True, "budget", True, 3)],
            ),
        ]
        operator, store = self._operator(responses)
        result = operator.retrieve(
            Query(scope_id="s1", text="q", mode=RetrievalMode.FUSED, reinforce=False)
        )
        assert result.probes["graph_censored"] is True
        assert result.probes["termination_reason"] == "budget"
        assert result.probes["budget_capped"] is True
        assert result.probes["hnsw_iterative_scan"] == "relaxed_order"

    def test_as_of_selects_the_value_current_then_not_now(self):
        """C1: prior values appear only when q explicitly requests history."""
        operator, store = self._operator()
        operator.retrieve(
            Query(
                scope_id="s1",
                text="q",
                mode=RetrievalMode.VECTOR,
                as_of="2026-03-01",
                reinforce=False,
            )
        )
        history = store.conn.sql_matching(r"SELECT unit_id, field, value")[0]
        assert "valid_from <= %s" in history and "valid_to > %s" in history

    def test_default_query_returns_only_current_values(self):
        operator, store = self._operator()
        operator.retrieve(
            Query(scope_id="s1", text="q", mode=RetrievalMode.VECTOR, reinforce=False)
        )
        history = store.conn.sql_matching(r"SELECT unit_id, field, value")[0]
        assert "valid_to IS NULL" in history

    def test_temporal_binds_the_ranking_vector_to_its_own_placeholder(self):
        """Regression: the ranking vector sits in the SELECT list, so it is the
        FIRST placeholder in the statement. Appending it to the parameter list
        last bound the scope id to ``%s::vector`` and every other argument one
        position off — a mismatch a placeholder COUNT check cannot see."""
        operator, store = self._operator()
        operator.retrieve(
            Query(
                scope_id="s1",
                embedding=[1.0, 0.0, 0.0, 0.5],
                route=RetrievalRoute.TEMPORAL,
                as_of="2026-03-01",
                reinforce=False,
            )
        )
        sql, params = next(
            (sql, params)
            for sql, params in store.conn.statements
            if "SELECT DISTINCT u.id" in sql
        )
        assert sql.count("%s") == len(params)
        offsets = [m.start() for m in re.finditer(r"%s", sql)]
        assert len(offsets) == len(params)
        for index, (offset, value) in enumerate(zip(offsets, params, strict=True)):
            # Anchor on the text IMMEDIATELY around this placeholder — a
            # substring search over the whole statement would match an earlier
            # occurrence and silently pass.
            following = sql[offset + 2 :]
            preceding = sql[:offset]
            if following.startswith("::vector"):
                assert value.startswith("["), (
                    f"argument {index} binds to ::vector but is {value!r}"
                )
            elif preceding.endswith("scope_id = "):
                assert value == "s1", f"argument {index} binds to scope_id"
            elif preceding.endswith("valid_from <= ") or preceding.endswith(
                "valid_to > "
            ):
                assert value == "2026-03-01", f"argument {index} binds to as_of"

    def test_temporal_without_a_vector_still_binds_correctly(self):
        operator, store = self._operator()
        operator.retrieve(
            Query(
                scope_id="s1",
                mode=RetrievalMode.RELATIONAL,
                route=RetrievalRoute.TEMPORAL,
                reinforce=False,
            )
        )
        sql, params = next(
            (sql, params)
            for sql, params in store.conn.statements
            if "SELECT DISTINCT u.id" in sql
        )
        assert sql.count("%s") == len(params) == 1
        assert params[0] == "s1"

    def test_graph_mode_needs_no_query_vector(self):
        """Regression: GRAPH is a pure traversal from an explicit anchor and
        never ranks by similarity, but the vector check treated it like the
        vector modes — which made the whole mode unreachable without a vector."""
        responses = self._responses() + [
            (r"gph_traverse_bfs", [(2,), (3,)]),
            (r"^SELECT nextval", [(1,)]),
        ]
        operator, store = self._operator(responses)
        result = operator.retrieve(
            Query(
                scope_id="s1",
                mode=RetrievalMode.GRAPH,
                anchor_id=5,
                hops=3,
                reinforce=False,
            )
        )
        assert result.committed, result.aborted_reason
        assert store.conn.ran(r"gph_traverse_bfs")
        assert result.probes["mode"] == "graph"

    def test_graph_mode_one_hop_uses_the_typed_traversal(self):
        responses = self._responses() + [
            (r"gph_traverse_typed", [(2,)]),
            (r"^SELECT nextval", [(1,)]),
        ]
        operator, store = self._operator(responses)
        result = operator.retrieve(
            Query(scope_id="s1", mode=RetrievalMode.GRAPH, anchor_id=5, reinforce=False)
        )
        assert result.committed, result.aborted_reason
        # target-list (ProjectSet) position: a FROM-clause FunctionScan loses
        # early termination under LIMIT (TR-1)
        assert store.conn.ran(r"SELECT \(e\).dst FROM \(SELECT")

    def test_structural_route_requires_an_anchor(self):
        operator, _ = self._operator()
        result = operator.retrieve(
            Query(scope_id="s1", text="q", route=RetrievalRoute.STRUCTURAL)
        )
        assert not result.committed
        assert "anchor_id" in result.aborted_reason

    def test_a_graph_vid_with_no_visible_row_is_skipped_not_fabricated(self):
        """Graph reads are commit-visible, not snapshot-isolated (§6.5), so the
        graph leg can return a vid whose row this snapshot cannot see."""
        responses = [
            (r"SELECT id FROM gem_unit", [(1,), (99,)]),
            (
                r"SELECT id, title, state, salience FROM gem_unit",
                [(1, "T1", "active", 0.0)],
            ),
            (r"SELECT unit_id, field, value", [(1, "content", "hello")]),
            (r"RETURNING t", [(1,)]),
        ]
        operator, _ = self._operator(responses)
        result = operator.retrieve(
            Query(scope_id="s1", text="q", mode=RetrievalMode.VECTOR, reinforce=False)
        )
        assert {h.unit_id for h in result.hits} == {1}

    def test_association_edges_are_strengthened_but_never_promoted(self):
        """Promotion grants propagation rights and would silently widen the C3
        frontier — that stays a revise decision."""
        responses = [
            (r"SELECT id FROM gem_unit", [(1,), (2,)]),
            (
                r"SELECT id, title, state, salience FROM gem_unit",
                [(1, "T1", "active", 0.0), (2, "T2", "active", 0.0)],
            ),
            (
                r"SELECT unit_id, field, value",
                [(1, "content", "a"), (2, "content", "b")],
            ),
            (r"SELECT id, salience FROM gem_unit", [(1, 0.0), (2, 0.0)]),
            (r"RETURNING t", [(1,)]),
        ]
        operator, store = self._operator(responses)
        operator.retrieve(
            Query(scope_id="s1", text="q", mode=RetrievalMode.VECTOR, reinforce=True)
        )
        assert store.conn.ran(r"co_access_count = co_access_count \+ 1")
        assert store.conn.ran(r"extension_candidate")
        # nothing turns an association edge into an extension edge
        assert not store.conn.ran(r"SET kind = 'extension'")

    def test_salience_after_is_strictly_greater_than_before(self):
        responses = [
            (r"SELECT id FROM gem_unit", [(1,)]),
            (
                r"SELECT id, title, state, salience FROM gem_unit",
                [(1, "T1", "active", 0.25)],
            ),
            (r"SELECT unit_id, field, value", [(1, "content", "hello")]),
            (r"SELECT id, salience FROM gem_unit", [(1, 1.3)]),
            (r"RETURNING t", [(1,)]),
        ]
        operator, _ = self._operator(responses)
        result = operator.retrieve(
            Query(scope_id="s1", text="q", mode=RetrievalMode.VECTOR, reinforce=True)
        )
        for hit in result.hits:
            assert hit.salience_after > hit.salience_before

    def test_prompt_block_attributes_every_line_to_a_unit(self):
        from bench.agent_memory.gem.types import Hit, UnitState

        block = build_prompt_block(
            [
                Hit(
                    unit_id=7,
                    title="Website",
                    field_name="deadline",
                    value="April 20",
                    score=1.0,
                    state=UnitState.ACTIVE,
                )
            ]
        )
        assert "[unit 7]" in block and "deadline: April 20" in block


# ---------------------------------------------------------------------------
# policy registry
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    def test_the_registry_is_closed_and_named(self):
        """A general policy LANGUAGE is [GEM]'s own research direction, not
        ours — the registry is the scope fence."""
        assert {
            "no_duplicate_current",
            "dependents_flagged",
            "active_below_bound",
            "salience_monotone",
        } <= set(CONDITIONS)
        assert {"flag_for_revision", "promote_edge_candidate", "attenuate"} <= set(
            ACTIONS
        )

    def test_an_unknown_condition_rejects_the_transition(self):
        engine = PolicyEngine(
            extra=[
                Policy(
                    name="bogus",
                    event=PolicyEvent.UNIT_INGESTED,
                    condition={"name": "does_not_exist"},
                    action={},
                )
            ]
        )
        store = _store(audit=FakeConn([(r"RETURNING t", [(1,)])]), hook=engine.evaluate)
        with pytest.raises(PolicyViolation, match="not in the registry"):
            with store.transition("ingest", "s1", "construction"):
                pass

    def test_a_failing_condition_aborts_and_names_the_policy(self):
        engine = PolicyEngine(
            extra=[
                Policy(
                    name="bound-active-state",
                    event=PolicyEvent.UNIT_INGESTED,
                    condition={
                        "name": "active_below_bound",
                        "params": {"max_active_units": 1},
                    },
                    action={},
                )
            ]
        )
        store = _store(
            [
                (r"count\(\*\) FROM gem_unit", [(99,)]),
                (r"count\(\*\) FROM gem_field_value", [(0,)]),
            ],
            audit=FakeConn([(r"RETURNING t", [(1,)])]),
            hook=engine.evaluate,
        )
        with pytest.raises(PolicyViolation, match="bound-active-state"):
            with store.transition("ingest", "s1", "construction"):
                pass
        assert store.conn.rolled_back

    def test_a_policy_only_fires_for_its_own_event(self):
        engine = PolicyEngine(
            extra=[
                Policy(
                    name="retrieval-only",
                    event=PolicyEvent.RETRIEVAL,
                    condition={"name": "salience_monotone"},
                    action={},
                )
            ]
        )
        store = _store([(r"RETURNING t", [(1,)])], hook=engine.evaluate)
        with store.transition("ingest", "s1", "construction") as tx:
            pass
        assert tx.policies_evaluated == []

    def test_dependents_flagged_checks_out_from_what_CHANGED(self):
        """Regression: the condition read delta.propagated_units — the units it
        just FLAGGED — and demanded their out-neighbours be flagged too. That is
        one hop further than ingest ever flags, so the default
        propagate-on-change policy aborted every ingest on any A->B->C extension
        chain. It must read tx.changed_units instead."""
        engine = PolicyEngine(extra=seed_policies())
        store = _store(
            [
                # No unflagged dependent of the CHANGED unit.
                (r"count\(\*\) FROM gem_unit u JOIN gem_edge", [(0,)]),
                (r"^SELECT nextval", [(1,)]),
                (r"RETURNING t", [(1,)]),
            ],
            hook=engine.evaluate,
        )
        with store.transition("ingest", "s1", "construction") as tx:
            tx.changed_units.append(1)  # A changed
            tx.delta.propagated_units.append(2)  # B was flagged
        assert "propagate-on-change" in tx.policies_evaluated
        # the condition must be asked about the CHANGED unit, not the flagged one
        params = next(
            p
            for sql, p in store.conn.statements
            if "JOIN gem_edge e ON e.dst = u.id" in sql
        )
        assert params[0] == [1], "the C3 condition looked at the wrong units"

    def test_dependents_flagged_still_rejects_a_genuinely_unflagged_dependent(self):
        engine = PolicyEngine(extra=seed_policies())
        store = _store(
            [
                (r"count\(\*\) FROM gem_unit u JOIN gem_edge", [(1,)]),
                (r"^SELECT nextval", [(1,)]),
            ],
            audit=FakeConn([(r"RETURNING t", [(1,)])]),
            hook=engine.evaluate,
        )
        with pytest.raises(PolicyViolation, match="propagate-on-change"):
            with store.transition("ingest", "s1", "construction") as tx:
                tx.changed_units.append(1)

    def test_seed_policies_cover_the_listing_1_rules(self):
        names = {p.name for p in seed_policies(max_active_units=100)}
        assert names == {
            "propagate-on-change",
            "reinforce-on-read",
            "bound-active-state",
        }

    def test_attenuate_refuses_a_state_off_the_ladder(self):
        store = _store()
        with store.transition("forget", "s1", "construction") as tx:
            with pytest.raises(ValueError, match="ladder rung"):
                ACTIONS["attenuate"](tx, {"state": "deleted"})

    def test_no_action_ever_deletes(self):
        """C4/C5: never destructive. Guarding the registry itself is cheaper
        than auditing every call site later."""
        import inspect

        from bench.agent_memory.gem import forget as forget_module
        from bench.agent_memory.gem import policy as policy_module

        for module in (policy_module, forget_module):
            source = inspect.getsource(module)
            assert not re.search(r"\bDELETE\s+FROM\b", source, re.I)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def test_vec_literal_round_trips_floats():
    assert vec_literal([1, 2.5, -0.25]) == "[1.0,2.5,-0.25]"


def test_lambda_default_is_a_24h_half_life():
    assert ExponentialSalience().lam == pytest.approx(math.log(2) / 86_400)
