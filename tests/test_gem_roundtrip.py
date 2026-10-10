"""End-to-end operator wiring against a fake that MODELS the C1 index.

``test_gem_unit.py`` checks the statements each operator emits. That catches a
wrong query but not a wrong *sequence* — an operator can emit three individually
correct statements in an order that produces the wrong state.

So this module runs the real operators against a small relational fake that
keeps actual rows and, crucially, **enforces the partial unique index**::

    CREATE UNIQUE INDEX gem_field_value_current_uq ON gem_field_value
        (unit_id, field) WHERE valid_to IS NULL AND state <> 'archived'

That index is [GEM] Observation 3a's "engine-level mechanism" — the thing
append-only stores lack — so a fake that ignores it would let a broken
supersession path pass. Here, appending a second current value raises exactly
as PostgreSQL would, and the operator must abort.

This is still not a substitute for ``test_gem_live.py``: the fake models one
index, not an engine. It cannot speak to HNSW, tjs_open fusion, HOT updates, or
graph visibility.
"""

from __future__ import annotations

import contextlib

import psycopg
import psycopg.adapt
import pytest

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.ingest import IngestOperator
from bench.agent_memory.gem.retrieve import RetrieveOperator
from bench.agent_memory.gem.store import GemStore
from bench.agent_memory.gem.types import (
    InteractionEvent,
    Query,
    RetrievalMode,
)


class MiniPG:
    """A tiny relational fake: real rows, and the C1 index actually enforced."""

    connection = None
    adapters = psycopg.adapt.AdaptersMap(psycopg.adapters)

    def __init__(self) -> None:
        self.units: dict[int, dict] = {}
        self.values: dict[int, dict] = {}
        self.transitions: list[tuple] = []
        self.next_vid = 0
        self.next_value_id = 1

    def execute(self, sql, params=None):
        query = " ".join(sql.split())
        args = tuple(params or ())
        rows = self._dispatch(query, args)

        class Cursor:
            def __init__(self, r):
                self.rows = list(r or [])

            def fetchone(self):
                return self.rows[0] if self.rows else None

            def fetchall(self):
                return list(self.rows)

        return Cursor(rows)

    def transaction(self):
        return contextlib.nullcontext()

    def _dispatch(self, q: str, p: tuple):
        if "gph_allocated_vids" in q:
            return [(self.next_vid,)]
        if "gph_upsert_vertex" in q:
            self.next_vid = max(self.next_vid, int(p[0]) + 1)
            return [(int(p[0]),)]

        if q.startswith("SELECT id FROM gem_unit WHERE scope_id") and "title" in q:
            match = [
                u
                for u in self.units.values()
                if u["scope_id"] == p[0] and u["title"] == p[1]
            ]
            return [(match[0]["id"],)] if match else []

        if q.startswith("INSERT INTO gem_unit"):
            self.units[p[0]] = {
                "id": p[0],
                "scope_id": p[1],
                "title": p[2],
                "summary": p[3],
                "state": "active",
                "salience": 0.0,
                "access_count": 0,
            }
            return []

        if q.startswith("UPDATE gem_field_value SET valid_to") and "RETURNING" in q:
            live = [
                v
                for v in self.values.values()
                if v["unit_id"] == p[1] and v["field"] == p[2] and v["valid_to"] is None
            ]
            if not live:
                return []
            live[0]["valid_to"] = p[0]
            return [(live[0]["id"],)]

        if q.startswith("INSERT INTO gem_field_value"):
            # *** the partial unique index ***
            duplicate = [
                v
                for v in self.values.values()
                if v["unit_id"] == p[0]
                and v["field"] == p[1]
                and v["valid_to"] is None
                and v["state"] != "archived"
            ]
            if duplicate:
                raise psycopg.errors.UniqueViolation(
                    "duplicate key value violates unique constraint "
                    '"gem_field_value_current_uq"'
                )
            value_id = self.next_value_id
            self.next_value_id += 1
            self.values[value_id] = {
                "id": value_id,
                "unit_id": p[0],
                "field": p[1],
                "value": p[2],
                "valid_from": p[3],
                "valid_to": None,
                "superseded_by": None,
                "state": "active",
                "salience": 0.0,
            }
            return [(value_id,)]

        if q.startswith("UPDATE gem_field_value SET superseded_by"):
            self.values[p[1]]["superseded_by"] = p[0]
            return []

        if "count(*) FROM gem_unit" in q:
            return [(len([u for u in self.units.values() if u["state"] == "active"]),)]
        if "count(*) FROM gem_field_value" in q:
            return [(len(self.values),)]
        if q.startswith("SELECT nextval"):
            return [(len(self.transitions) + 1,)]
        if "INSERT INTO gem_transition" in q:
            self.transitions.append(p)
            return [(p[0],)]

        if q.startswith("SELECT id FROM gem_unit WHERE"):
            return [(u["id"],) for u in self.units.values() if u["state"] == "active"]
        if q.startswith("SELECT id, title, state, salience FROM gem_unit"):
            return [
                (u["id"], u["title"], u["state"], u["salience"])
                for u in self.units.values()
                if u["id"] in p[0]
            ]
        if q.startswith("SELECT unit_id, field, value"):
            return [
                (v["unit_id"], v["field"], v["value"])
                for v in self.values.values()
                if v["unit_id"] in p[0] and v["valid_to"] is None
            ]
        if q.startswith("SELECT id, salience FROM gem_unit"):
            return [
                (u["id"], u["salience"]) for u in self.units.values() if u["id"] in p[0]
            ]
        if q.startswith("UPDATE gem_unit SET salience = v.salience"):
            for unit_id, salience in zip(p[0], p[1], strict=True):
                self.units[unit_id]["salience"] = salience
                self.units[unit_id]["access_count"] += 1
            return []
        return []

    # -- convenience
    def current(self, field="deadline"):
        return [
            v["value"]
            for v in self.values.values()
            if v["field"] == field and v["valid_to"] is None
        ]

    def history(self, field="deadline"):
        return sorted(
            (v["value"], v["valid_to"], v["superseded_by"])
            for v in self.values.values()
            if v["field"] == field
        )


class Embedder:
    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.5] for _ in texts]


def _strategy(ops):
    class Fixed:
        name = "test"
        cost: dict = {}

        def plan(self, events, view):
            return ops

    return Fixed()


def _events(scope="S"):
    return [
        InteractionEvent(
            scope_id=scope, external_id="e", content="c", event_time="2026-02-01"
        )
    ]


def _deadline(value, when, supersede):
    return [
        planmod.upsert_unit(
            scope_id="S", title="Website Redesign", ref="w", embed_text="Website"
        ),
        planmod.append_field_value(
            ref="w",
            field="deadline",
            value=value,
            valid_from=when,
            supersede_current=supersede,
        ),
    ]


@pytest.fixture()
def stack():
    pg = MiniPG()
    store = GemStore(pg, dim=4)
    return pg, store, IngestOperator(store, embedder=Embedder())


class TestFigureOneScenario:
    """[GEM] Figure 1: the deadline moves from March 15 to April 20."""

    def test_first_ingest_creates_the_unit_and_the_fact(self, stack):
        pg, _, ingest = stack
        result = ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        assert result.committed, result.aborted_reason
        assert result.delta.units_created == 1
        assert pg.current() == ["March 15"]

    def test_appending_a_second_current_value_ABORTS(self, stack):
        """Failure ②: without supersession the two values would coexist with
        equal status. The index is the engine-level mechanism that forbids it,
        and the operator must let it fire rather than work around it."""
        pg, _, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        result = ingest.ingest(
            _events(), strategy=_strategy(_deadline("April 20", "2026-03-08", False))
        )
        assert not result.committed
        assert "gem_field_value_current_uq" in result.aborted_reason
        assert pg.current() == ["March 15"], "the aborted write left state behind"

    def test_c1_after_supersession_only_the_new_value_is_current(self, stack):
        pg, _, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        result = ingest.ingest(
            _events(), strategy=_strategy(_deadline("April 20", "2026-03-08", True))
        )
        assert result.committed, result.aborted_reason
        assert pg.current() == ["April 20"]

    def test_c4_the_prior_value_survives_and_is_chained(self, stack):
        pg, _, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("April 20", "2026-03-08", True))
        )
        history = pg.history()
        assert len(history) == 2
        old = next(h for h in history if h[0] == "March 15")
        new = next(h for h in history if h[0] == "April 20")
        assert old[1] is not None, "the superseded value was not closed"
        assert old[2] is not None, "the superseded value was not chained (C4)"
        assert new[1] is None and new[2] is None

    def test_the_unit_is_updated_not_duplicated(self, stack):
        """A unit is a TOPIC. Two rows for one topic is the entity-grain
        scatter [GEM] §4.1 rejects."""
        pg, _, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("April 20", "2026-03-08", True))
        )
        assert len(pg.units) == 1


class TestRetrievalAfterSupersession:
    def test_retrieval_returns_only_the_current_value(self, stack):
        pg, store, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("April 20", "2026-03-08", True))
        )
        result = RetrieveOperator(store, embedder=Embedder()).retrieve(
            Query(
                scope_id="S",
                embedding=[1.0, 0.0, 0.0, 0.5],
                mode=RetrievalMode.VECTOR,
                reinforce=False,
            )
        )
        assert result.committed, result.aborted_reason
        assert [h.value for h in result.hits] == ["April 20"]

    def test_c6_salience_rises_and_access_is_recorded(self, stack):
        pg, store, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        operator = RetrieveOperator(store, embedder=Embedder())
        query = Query(
            scope_id="S",
            embedding=[1.0, 0.0, 0.0, 0.5],
            mode=RetrievalMode.VECTOR,
            reinforce=True,
        )
        first = operator.retrieve(query)
        assert first.committed, first.aborted_reason
        assert all(h.salience_after > h.salience_before for h in first.hits)

        second = operator.retrieve(query)
        assert second.hits[0].salience_before > first.hits[0].salience_before
        assert [u["access_count"] for u in pg.units.values()] == [2]

    def test_reinforce_false_leaves_salience_untouched(self, stack):
        pg, store, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        operator = RetrieveOperator(store, embedder=Embedder())
        for _ in range(3):
            operator.retrieve(
                Query(
                    scope_id="S",
                    embedding=[1.0, 0.0, 0.0, 0.5],
                    mode=RetrievalMode.VECTOR,
                    reinforce=False,
                )
            )
        assert [u["salience"] for u in pg.units.values()] == [0.0]
        assert [u["access_count"] for u in pg.units.values()] == [0]


class TestTrajectory:
    def test_every_transition_is_logged_with_its_committed_flag(self, stack):
        pg, _, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        assert len(pg.transitions) == 1
        # column order: t, scope_id, operator, phase, committed, ...
        assert pg.transitions[0][4] is True

    def test_an_aborted_transition_is_not_logged_on_the_main_connection(self, stack):
        """It rolls back with the transaction — which is exactly why the store
        keeps a separate audit connection for the aborted path."""
        pg, _, ingest = stack
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("March 15", "2026-02-01", False))
        )
        before = len(pg.transitions)
        ingest.ingest(
            _events(), strategy=_strategy(_deadline("April 20", "2026-03-08", False))
        )
        assert len(pg.transitions) == before
