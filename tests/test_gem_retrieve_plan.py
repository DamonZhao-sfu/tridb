"""Plan P3: retrieve as a plan — consistency, theta_pol, theta_sem, Assemble_B.

Fake-connection tests pin the SQL and receipt mechanics; live tests
(``TRIDB_GEM_DSN``) pin the engine behaviour: a strong read sees the
materialized pending set while a stale read reports staleness, an ACL'd unit
never reaches another principal on any route, and theta_sem is counted and
cache-suppressed.
"""

from __future__ import annotations

import json
import os
import re
import uuid

import psycopg.adapt as psycopg_adapt
import pytest
from psycopg import adapters as psycopg_adapters

from bench.agent_memory.gem.ingest_plan import DEFERRED, IngestPlan, PlannedIngest
from bench.agent_memory.gem.policy import PolicyEngine, seed_policies
from bench.agent_memory.gem.retrieve import (
    SEM_OVERFETCH,
    RetrieveOperator,
    _approx_tokens,
    build_prompt_block,
)
from bench.agent_memory.gem.store import GemStore
from bench.agent_memory.gem.strategies.deterministic import DeterministicIngestStrategy
from bench.agent_memory.gem.types import (
    Hit,
    InteractionEvent,
    Query,
    RetrievalMode,
    RetrievalRoute,
    UnitState,
)
from bench.agent_memory.semantic.cache import SemMeter
from bench.agent_memory.semantic.llm import MockLLM
from bench.agent_memory.semantic.ops import SemOps

DIM = 8


class StubEmbedder:
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


class OneChunk:
    def chunk(self, text):
        return [text]


class DictCache:
    def __init__(self):
        self.rows = {}

    def get_many(self, op, model, prompt_version, hashes):
        return {
            h: self.rows[(op, model, prompt_version, h)]
            for h in hashes
            if (op, model, prompt_version, h) in self.rows
        }

    def put_many(self, op, model, prompt_version, rows):
        for h, out, *_ in rows:
            self.rows.setdefault((op, model, prompt_version, h), out)
        return len(rows)


def keep_if_marked(system: str, user: str) -> str:
    """sem_filter oracle: keep items whose title contains 'keep'."""
    item = json.loads(user)["item"]
    return json.dumps({"keep": "keep" in item["title"]})


# ---------------------------------------------------------------------------
# Fake connection
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.events.append("BEGIN")

    def __exit__(self, exc_type, exc, tb):
        self.conn.events.append("ROLLBACK" if exc_type else "COMMIT")
        return False


class FakeConn:
    connection = None
    adapters = psycopg_adapt.AdaptersMap(psycopg_adapters)

    def __init__(self, units, *, open_pending=0):
        self.units = units  # id -> (title, {field: value})
        self.open_pending = open_pending
        self.statements = []
        self.events = []

    def transaction(self):
        return FakeTransaction(self)

    def close(self):
        pass

    def execute(self, sql, params=None):
        text = " ".join(str(sql).split())
        params = tuple(params or ())
        self.statements.append((text, params))
        self.events.append(text)
        if text.startswith("SELECT count(*) FROM gem_pending"):
            return FakeCursor([(self.open_pending,)])
        if "ORDER BY embedding <=>" in text:
            limit = int(params[-1])
            return FakeCursor([(i,) for i in sorted(self.units)][:limit])
        if text.startswith("SELECT id, title, state, salience FROM gem_unit"):
            return FakeCursor(
                [
                    (i, self.units[i][0], "active", 0.0)
                    for i in params[0]
                    if i in self.units
                ]
            )
        if text.startswith("SELECT unit_id, field, value FROM gem_field_value"):
            return FakeCursor(
                [
                    (i, f, v)
                    for i in params[0]
                    if i in self.units
                    for f, v in sorted(self.units[i][1].items())
                ]
            )
        return FakeCursor([])

    def ran(self, pattern):
        return [s for s, _ in self.statements if re.search(pattern, s, re.I)]


def _op(units, **kwargs):
    conn = FakeConn(units, open_pending=kwargs.pop("open_pending", 0))
    store = GemStore(conn, dim=4)
    return conn, RetrieveOperator(store, **kwargs)


def _vq(**kwargs):
    return Query(
        scope_id="s1",
        embedding=[1.0, 0.0, 0.0, 0.0],
        mode=RetrievalMode.VECTOR,
        reinforce=False,
        **kwargs,
    )


UNITS = {
    i: (f"{'keep' if i % 2 else 'skip'}-{i}", {"f": f"v{i}"}) for i in range(1, 21)
}


class TestConsistencyFake:
    def test_strong_resolves_before_the_read_transaction(self):
        calls = []
        conn, op = _op(
            UNITS, resolver=lambda scope: calls.append((scope, list(conn.events))) or 3
        )
        result = op.retrieve(_vq(consistency="strong"))
        assert result.committed, result.aborted_reason
        assert result.pending_resolved == 3 and result.staleness == 0
        assert calls and calls[0][0] == "s1" and calls[0][1] == [], (
            "resolver ran before BEGIN"
        )
        assert result.probes["consistency"] == "strong"

    def test_stale_reports_open_items_without_resolving(self):
        calls = []
        _, op = _op(
            UNITS, resolver=lambda scope: calls.append(scope) or 0, open_pending=4
        )
        result = op.retrieve(_vq())
        assert (
            result.committed and result.staleness == 4 and result.pending_resolved == 0
        )
        assert calls == []

    def test_legacy_operator_issues_no_pending_statement(self):
        conn, op = _op(UNITS)
        assert op.retrieve(_vq()).committed
        assert not conn.ran("gem_pending")

    def test_strong_without_a_resolver_refuses_to_read_stale_state(self):
        _, op = _op(UNITS, open_pending=2)
        result = op.retrieve(_vq(consistency="strong"))
        assert not result.committed and "strong read" in result.aborted_reason


class TestPrincipalPredicateFake:
    def test_acl_clause_is_pushed_with_scope_and_quoted(self):
        _, op = _op(UNITS)
        rendered = op.predicate(_vq(principal="alice")).as_string(op.store.conn)
        assert "scope_id = 's1'" in rendered
        assert "(NOT (metadata ? 'acl') OR metadata->'acl' ? 'alice')" in rendered
        hostile = op.predicate(_vq(principal="x' OR true --")).as_string(op.store.conn)
        assert "? 'x'' OR true --'" in hostile, "the principal is a quoted literal"
        assert "acl" not in op.predicate(_vq()).as_string(op.store.conn)

    def test_percent_in_a_principal_does_not_break_parameter_binding(self):
        conn, op = _op(UNITS)
        result = op.retrieve(_vq(principal="50%off"))
        assert result.committed, result.aborted_reason
        sql = conn.ran("ORDER BY embedding <=>")[0]
        assert "'50%%off'" in sql
        assert result.probes["principal_predicate"] is True


class TestSemFilterFake:
    def test_overfetch_is_bounded_counted_and_survivors_keep_engine_order(self):
        llm = MockLLM(responder=keep_if_marked)
        ops = SemOps(llm, DictCache(), SemMeter())
        conn, op = _op(UNITS, semops=ops, sem_cap=12)
        result = op.retrieve(_vq(k=3, sem_predicate="is marked keep"))
        assert result.committed, result.aborted_reason
        k_prime = min(SEM_OVERFETCH * 3, 12)
        limit = conn.statements[
            [s for s, _ in conn.statements].index(conn.ran("ORDER BY embedding")[0])
        ][1][-1]
        assert limit == k_prime
        assert [h.unit_id for h in result.hits] == [1, 3, 5]
        assert result.n_sem == k_prime and result.sem_llm_calls == k_prime
        probe = result.probes["sem_filter"]
        assert probe["k_prime"] == k_prime and probe["candidates"] == k_prime
        assert (
            probe["survivors"] == 3
            and probe["placement"] == "post_engine_bounded_overfetch"
        )
        again = op.retrieve(_vq(k=3, sem_predicate="is marked keep"))
        assert again.n_sem == k_prime and again.sem_llm_calls == 0, (
            "second read is all cache hits"
        )

    def test_sem_predicate_without_semops_aborts(self):
        _, op = _op(UNITS)
        result = op.retrieve(_vq(sem_predicate="anything"))
        assert not result.committed and "SemOps" in result.aborted_reason


class TestAssembleBudget:
    HITS = [
        Hit(
            unit_id=i,
            title=f"title {i}",
            field_name="f",
            value="x" * 30,
            score=1.0,
            state=UnitState.ACTIVE,
        )
        for i in range(10)
    ]

    @pytest.mark.parametrize("budget", [0, 5, 13, 40, 1000])
    def test_block_is_an_order_preserving_prefix_within_budget(self, budget):
        full = build_prompt_block(self.HITS).split("\n")
        block = build_prompt_block(self.HITS, budget_tokens=budget)
        assert _approx_tokens(block) <= budget
        lines = block.split("\n") if block else []
        assert lines == full[: len(lines)]
        if len(lines) < len(full):
            assert _approx_tokens("\n".join(full[: len(lines) + 1])) > budget

    def test_retrieve_applies_the_budget_and_discloses_it(self):
        _, op = _op(UNITS)
        result = op.retrieve(_vq(k=10, budget_tokens=12))
        assert _approx_tokens(result.prompt_block) <= 12
        assert result.probes["assemble"]["lines_total"] == 10
        assert result.probes["assemble"]["lines_kept"] < 10


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

DSN = os.environ.get("TRIDB_GEM_DSN")
live = pytest.mark.skipif(
    not DSN, reason="set TRIDB_GEM_DSN to run GEM integration tests"
)


@pytest.fixture()
def live_memory():
    from bench.agent_memory.gem.memory import TriDBGovernedMemory

    store = GemStore.connect(DSN, dim=DIM)
    mem = TriDBGovernedMemory(
        store,
        embedder=StubEmbedder(),
        policy_engine=PolicyEngine(extra=seed_policies()),
    )
    mem.init_schema()
    yield mem
    mem.close()


def _clean(conn, scope):
    conn.execute("DELETE FROM gem_unit WHERE scope_id = %s", (scope,))
    conn.execute("DELETE FROM gem_transition WHERE scope_id = %s", (scope,))
    conn.execute("DELETE FROM gem_pending WHERE scope_id = %s", (scope,))


def _event(scope, ext, content, **metadata):
    return InteractionEvent(
        scope_id=scope,
        external_id=ext,
        content=content,
        event_time="2026-03-01",
        metadata=metadata,
    )


@live
class TestRetrievePlanLive:
    def test_strong_sees_materialized_state_while_stale_reports_staleness(
        self, live_memory
    ):
        scope = "rplan_consistency"
        conn = live_memory.store.conn
        _clean(conn, scope)
        planned = PlannedIngest(
            live_memory,
            DeterministicIngestStrategy(chunker=OneChunk()),
            IngestPlan(mode=DEFERRED, trigger="on_read"),
        )
        receipt = planned.ingest(
            [_event(scope, f"e{i}", f"fact number {i}") for i in range(3)]
        )
        assert len(receipt.pending_enqueued) == 3
        op = RetrieveOperator(
            live_memory.store, embedder=StubEmbedder(), resolver=planned.resolver()
        )
        query = dict(
            scope_id=scope, mode=RetrievalMode.RELATIONAL, k=10, reinforce=False
        )
        stale = op.retrieve(Query(**query))
        assert stale.committed, stale.aborted_reason
        assert stale.hits == () and stale.staleness == 3 and stale.pending_resolved == 0
        strong = op.retrieve(Query(**query, consistency="strong"))
        assert strong.committed, strong.aborted_reason
        assert strong.pending_resolved == 3 and strong.staleness == 0
        assert {h.value for h in strong.hits} == {f"fact number {i}" for i in range(3)}
        after = op.retrieve(Query(**query))
        assert after.staleness == 0 and len(after.hits) == 3

    def test_acl_never_leaks_across_principals_on_any_route(self, live_memory):
        scope = "rplan_acl"
        conn = live_memory.store.conn
        _clean(conn, scope)
        strategy = DeterministicIngestStrategy(chunker=OneChunk())
        owned = {}
        for ext, acl in (
            ("pub", None),
            ("alice", ["alice"]),
            ("bob", ["bob"]),
            ("both", ["alice", "bob"]),
        ):
            meta = {} if acl is None else {"acl": acl}
            result = live_memory.ingest(
                [_event(scope, ext, f"note {ext}", **meta)], strategy=strategy
            )
            assert result.committed, result.aborted_reason
            owned[ext] = result.units[0]
        visible = {
            "alice": {owned["pub"], owned["alice"], owned["both"]},
            "bob": {owned["pub"], owned["bob"], owned["both"]},
            "mallory' OR true --%": {owned["pub"]},
        }
        routes = [
            dict(mode=RetrievalMode.RELATIONAL),
            dict(mode=RetrievalMode.VECTOR, text="note"),
            dict(
                mode=RetrievalMode.FUSED, text="note", hops=1, m_seeds=2, term_cond=64
            ),
            dict(route=RetrievalRoute.TEMPORAL, text="note"),
        ]
        for principal, expected in visible.items():
            for route in routes:
                result = live_memory.retrieve(
                    Query(
                        scope_id=scope,
                        k=10,
                        reinforce=False,
                        principal=principal,
                        **route,
                    )
                )
                assert result.committed, (principal, route, result.aborted_reason)
                got = {h.unit_id for h in result.hits}
                assert got <= expected, (principal, route, got - expected)
                if route.get("mode") is RetrievalMode.RELATIONAL:
                    assert got == expected

    def test_budget_and_sem_filter_counts(self, live_memory):
        from bench.agent_memory.semantic.cache import SemCache

        scope = "rplan_sem"
        conn = live_memory.store.conn
        _clean(conn, scope)
        strategy = DeterministicIngestStrategy(chunker=OneChunk())
        for i in range(8):
            ext = f"{'keep' if i % 2 else 'skip'}{i}"
            assert live_memory.ingest(
                [_event(scope, ext, f"body {i}")], strategy=strategy
            ).committed
        llm = MockLLM(responder=keep_if_marked)
        op = RetrieveOperator(
            live_memory.store,
            embedder=StubEmbedder(),
            semops=SemOps(llm, SemCache(conn)),
        )
        query = Query(
            scope_id=scope,
            mode=RetrievalMode.RELATIONAL,
            k=2,
            reinforce=False,
            # Run-unique predicate: gem_sem_cache persists across runs by design.
            sem_predicate=f"title marked keep ({uuid.uuid4().hex})",
            budget_tokens=20,
        )
        first = op.retrieve(query)
        assert first.committed, first.aborted_reason
        assert first.n_sem == 8 and first.sem_llm_calls == 8
        assert len({h.unit_id for h in first.hits}) == 2
        assert all(h.title.startswith("keep") for h in first.hits)
        assert _approx_tokens(first.prompt_block) <= 20
        second = op.retrieve(query)
        assert second.n_sem == 8 and second.sem_llm_calls == 0
        assert [h.unit_id for h in second.hits] == [h.unit_id for h in first.hits]
