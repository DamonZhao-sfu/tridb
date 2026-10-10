"""Plan P2: ingest as a plan — eager vs deferred derivation over Q (gem_pending).

Two layers, as everywhere in the GEM suites:

* **Fake connection** (always runs): a small stateful fake that answers the
  gem_pending / allocator / transition statements. It pins the MECHANICS — the
  deferred enqueue makes no LLM call and takes no writer lock, materialize
  marks an item resolved inside the same transaction as its derivation, model
  calls during materialize happen with no transaction open, and the write trace
  of eager-per-event equals deferred+materialize. Replay from cache is pinned
  here too (deterministic fake vids make prompts identical across runs).
* **Live** (``TRIDB_GEM_DSN``): the equivalence claim Mat(S_eager) ==
  Mat(S_deferred) on real gem_unit / gem_field_value / gem_edge rows, with an
  order-dependent LLM (it picks host topics from the CURRENT slate, D2).
"""

from __future__ import annotations

import json
import os
import re

import pytest

from bench.agent_memory.gem.ingest import IngestOperator
from bench.agent_memory.gem.ingest_plan import (
    DEFERRED,
    EAGER,
    IngestPlan,
    PlannedIngest,
    event_payload,
    payload_event,
    target_ref_for,
)
from bench.agent_memory.gem.policy import PolicyEngine, seed_policies
from bench.agent_memory.gem.semops import CachedCompletionClient
from bench.agent_memory.gem.store import WRITER_LOCK_KEY, GemStore, Tx
from bench.agent_memory.gem.strategies.deterministic import DeterministicIngestStrategy
from bench.agent_memory.gem.strategies.llm_mediated import (
    MODE_BATCH,
    MODE_SEQUENTIAL,
    PROMPT_VERSION,
    LLMMediatedIngestStrategy,
)
from bench.agent_memory.gem.types import InteractionEvent, PolicyEvent
from bench.agent_memory.semantic.llm import MockLLM
from bench.agent_memory.semantic.ops import CacheMiss

DIM = 8

# ---------------------------------------------------------------------------
# Shared deterministic pieces
# ---------------------------------------------------------------------------


class StubEmbedder:
    """Same deterministic vectors as tests/test_gem_live.py."""

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


def responder(system: str, user: str) -> str:
    """A valid llm_mediated reply that is ORDER-DEPENDENT (plan D2).

    The host is the slate candidate whose title matches the excerpt's topic if
    one is offered, else a new topic — so the answer depends on what earlier
    derivations committed, exactly like the real extractor.
    """
    payload = json.loads(user)
    fields = dict(part.split("=", 1) for part in payload["excerpt"].split(";"))
    host = next(
        (
            {"unit_id": c["unit_id"]}
            for c in payload["candidate_topics"]
            if c.get("title") == fields["topic"]
        ),
        {"new_title": fields["topic"], "summary": f"about {fields['topic']}"},
    )
    out = {
        "host": host,
        "facts": [
            {
                "field": fields["field"],
                "value": fields["value"],
                "valid_from": fields["date"],
                "confidence": 0.9,
            }
        ],
        "edges": [],
    }
    if fields.get("link"):
        out["edges"].append(
            {"dst_title": fields["link"], "kind": "association", "rel": "related"}
        )
    return json.dumps(out)


CORPUS = [
    ("Alpha", "owner", "ann", "2026-01-01", "Beta"),
    ("Beta", "city", "paris", "2026-01-02", ""),
    ("Alpha", "owner", "bob", "2026-01-03", ""),
    ("Gamma", "deadline", "2026-02-01", "2026-01-04", "Alpha"),
    ("Beta", "city", "rome", "2026-01-05", ""),
    ("Alpha", "budget", "10", "2026-01-06", ""),
    ("Delta", "owner", "cy", "2026-01-07", "Gamma"),
    ("Gamma", "deadline", "2026-03-01", "2026-01-08", ""),
]


def corpus_events(scope_id: str, corpus=CORPUS) -> list[InteractionEvent]:
    events = []
    for index, (topic, field, value, date, link) in enumerate(corpus):
        content = f"topic={topic};field={field};value={value};date={date}"
        if link:
            content += f";link={link}"
        events.append(
            InteractionEvent(
                scope_id=scope_id,
                external_id=f"ev{index}",
                content=content,
                event_time=date,
                event_order=index,
            )
        )
    return events


class DictCache:
    """In-memory stand-in with the SemCache get_many / put_many API."""

    def __init__(self):
        self.rows: dict[tuple[str, str, str, str], object] = {}

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


# ---------------------------------------------------------------------------
# Stateful fake connection
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, rows, rowcount=0):
        self._rows = list(rows)
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.depth += 1
        self.conn.events.append("BEGIN")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.conn.depth -= 1
        self.conn.writer_locked = False
        self.conn.events.append("ROLLBACK" if exc_type else "COMMIT")
        if exc_type:
            self.conn.pending.update(self.conn._pending_snapshot)
        return False


class StatefulFakeConn:
    """Answers the statements the ingest plan path issues; records the rest."""

    def __init__(self):
        self.statements: list[tuple[str, tuple]] = []
        self.events: list[str] = []
        self.depth = 0
        self.writer_locked = False
        self.pending: dict[int, dict] = {}
        self._pending_snapshot: dict[int, dict] = {}
        self.next_pending = 1
        self.next_vid = 100
        self.next_fv = 1000
        self.epoch = 0
        self.epoch_bump_on_read: int | None = None  # simulate a concurrent commit

    def transaction(self):
        self._pending_snapshot = {k: dict(v) for k, v in self.pending.items()}
        return FakeTransaction(self)

    def close(self):
        pass

    def execute(self, sql, params=None):
        text = " ".join(str(sql).split())
        params = tuple(params or ())
        self.statements.append((text, params))
        self.events.append(text)
        return self._answer(text, params)

    def _open(self, scope):
        return sorted(
            (
                p
                for p in self.pending.values()
                if p["scope"] == scope and p["status"] == "pending"
            ),
            key=lambda p: (p["seq"], p["id"]),
        )

    def _answer(self, text, params):  # noqa: C901 — a dispatch table, kept flat
        if "pg_advisory_xact_lock" in text and params and params[0] == WRITER_LOCK_KEY:
            self.writer_locked = True
            return FakeCursor([])
        if text.startswith("INSERT INTO gem_pending"):
            scope, derivation, ref, trigger, seq, payload = params
            if any(
                p["scope"] == scope
                and p["derivation"] == derivation
                and p["ref"] == ref
                for p in self.pending.values()
            ):
                return FakeCursor([])
            pid = self.next_pending
            self.next_pending += 1
            self.pending[pid] = {
                "id": pid,
                "scope": scope,
                "derivation": derivation,
                "ref": ref,
                "trigger": trigger,
                "seq": seq,
                "payload": payload.obj,
                "status": "pending",
            }
            return FakeCursor([(pid,)])
        if "max(snapshot_seq)" in text:
            seqs = [p["seq"] for p in self.pending.values() if p["scope"] == params[0]]
            return FakeCursor([(max(seqs, default=0),)])
        if (
            text.startswith("SELECT id, scope_id, derivation")
            and "FROM gem_pending" in text
        ):
            items = self._open(params[0])
            if "derivation = %s" in text:
                items = [p for p in items if p["derivation"] == params[1]]
            if "LIMIT %s" in text:
                items = items[: int(params[-1])]
            return FakeCursor(
                [
                    (
                        p["id"],
                        p["scope"],
                        p["derivation"],
                        p["ref"],
                        p["trigger"],
                        p["seq"],
                        p["payload"],
                    )
                    for p in items
                ]
            )
        if text.startswith("SELECT count(*) FROM gem_pending"):
            return FakeCursor([(len(self._open(params[0])),)])
        if text.startswith("SELECT status FROM gem_pending"):
            item = self.pending.get(int(params[0]))
            return FakeCursor([] if item is None else [(item["status"],)])
        if text.startswith("UPDATE gem_pending SET status = 'resolved'"):
            by, ids = params
            n = 0
            for pid in ids:
                if self.pending[pid]["status"] == "pending":
                    self.pending[pid].update(status="resolved", by=by)
                    n += 1
            return FakeCursor([], rowcount=n)
        if text.startswith("UPDATE gem_pending SET status = 'failed'"):
            self.pending[int(params[1])]["status"] = "failed"
            return FakeCursor([])
        if "count(*) FROM gem_transition" in text:
            value = self.epoch
            if self.epoch_bump_on_read is not None:
                self.epoch_bump_on_read -= 1
                if self.epoch_bump_on_read <= 0:
                    self.epoch += 1
                    self.epoch_bump_on_read = None
            return FakeCursor([(value,)])
        if "gph_allocated_vids" in text:
            return FakeCursor([(self.next_vid,)])
        if "gph_upsert_vertex" in text:
            self.next_vid += 1
            return FakeCursor([(params[0],)])
        if text.startswith("INSERT INTO gem_field_value"):
            self.next_fv += 1
            return FakeCursor([(self.next_fv,)])
        return FakeCursor([])

    # helpers
    def ran(self, pattern):
        return [s for s, _ in self.statements if re.search(pattern, s, re.I)]


_WRITE = re.compile(
    r"^(INSERT INTO gem_(unit|field_value|edge)|UPDATE gem_(unit|field_value|edge)"
    r"|SELECT graph_store\.gph_(insert_edge|upsert_vertex|tombstone_edge))"
)


def write_trace(conn: StatefulFakeConn) -> list[tuple[str, tuple]]:
    return [(s, p) for s, p in conn.statements if _WRITE.match(s)]


def _llm_strategy(client, *, mode=MODE_SEQUENTIAL, model="mock-llm-v1"):
    return LLMMediatedIngestStrategy(
        client=client,
        model=model,
        mode=mode,
        chunker=OneChunk(),
        embedder=StubEmbedder(),
    )


def _fake_stack(*, plan=None, llm=None, cache=None, offline=False):
    conn = StatefulFakeConn()
    store = GemStore(conn, dim=DIM)
    store._edge_types = {"extension": 1, "association": 2}
    llm = llm or MockLLM(responder=responder)
    client = CachedCompletionClient(
        llm,
        cache if cache is not None else DictCache(),
        prompt_version=PROMPT_VERSION,
        offline=offline,
    )
    planned = PlannedIngest(
        IngestOperator(store, embedder=StubEmbedder()),
        _llm_strategy(client),
        plan or IngestPlan(mode=DEFERRED),
        cache_client=client,
    )
    return conn, store, llm, client, planned


# ---------------------------------------------------------------------------
# Fake-connection tests
# ---------------------------------------------------------------------------


class TestIngestPlanValidation:
    def test_bad_mode_and_trigger_are_rejected(self):
        with pytest.raises(ValueError, match="mode"):
            IngestPlan(mode="lazy")
        with pytest.raises(ValueError, match="trigger"):
            IngestPlan(mode=DEFERRED, trigger="cron")

    def test_payload_round_trips_and_target_ref_is_stable(self):
        event = corpus_events("s")[0]
        assert payload_event(json.loads(json.dumps(event_payload(event)))) == event
        anonymous = InteractionEvent(scope_id="s", external_id="", content="x")
        assert target_ref_for(anonymous) == target_ref_for(anonymous)
        assert target_ref_for(anonymous).startswith("sha256:")


class TestDeferredEnqueue:
    def test_enqueue_is_one_txn_no_llm_no_writer_lock_no_state_write(self):
        conn, _, llm, _, planned = _fake_stack(
            plan=IngestPlan(mode=DEFERRED, trigger="background")
        )
        result = planned.ingest(corpus_events("s1")[:3])
        assert result.committed
        assert result.pending_enqueued == (1, 2, 3)
        assert llm.calls == 0, "deferred enqueue must not call the LLM"
        assert not conn.ran(WRITER_LOCK_KEY) and not [
            p for _, p in conn.statements if WRITER_LOCK_KEY in p
        ], "deferred enqueue must not take the writer lock"
        assert write_trace(conn) == [], "no gem_* state changes until materialized"
        assert conn.events[0] == "BEGIN" and conn.events[-1] == "COMMIT"
        assert conn.events.count("BEGIN") == 1
        items = sorted(conn.pending.values(), key=lambda p: p["id"])
        assert [p["seq"] for p in items] == [1, 2, 3]
        assert {p["trigger"] for p in items} == {"background"}
        assert {p["derivation"] for p in items} == {"llm_ingest"}
        assert [p["ref"] for p in items] == ["ev0", "ev1", "ev2"]

    def test_snapshot_seq_keeps_increasing_and_duplicates_are_reported(self):
        conn, _, _, _, planned = _fake_stack()
        planned.ingest(corpus_events("s1")[:2])
        again = planned.ingest(corpus_events("s1")[:3])
        assert again.pending_enqueued == (3,)
        assert [r["gate"] for r in again.rejected] == ["pending_duplicate"] * 2
        assert conn.pending[3]["seq"] == 5  # monotone across calls, never reused


class TestMaterialize:
    def test_resolution_commits_atomically_with_the_derivation(self):
        conn, _, llm, _, planned = _fake_stack()
        planned.ingest(corpus_events("s1")[:2])
        receipt = planned.materialize("s1")
        assert receipt.resolved == (1, 2) and receipt.failed == ()
        assert all(
            p["status"] == "resolved" and p["by"] == "batch"
            for p in conn.pending.values()
        )
        # Each resolution sits inside the SAME transaction (BEGIN..COMMIT) that
        # takes the writer lock and inserts the derived unit.
        windows, current = [], None
        for event in conn.events:
            if event == "BEGIN":
                current = []
            elif event in ("COMMIT", "ROLLBACK"):
                windows.append((event, current))
                current = None
            elif current is not None:
                current.append(event)
        resolving = [
            (end, w)
            for end, w in windows
            if any(
                e.startswith("UPDATE gem_pending SET status = 'resolved'") for e in w
            )
        ]
        assert len(resolving) == 2
        for end, window in resolving:
            assert end == "COMMIT"
            assert any("pg_advisory_xact_lock" in e for e in window)
            assert any(e.startswith("INSERT INTO gem_unit") for e in window)
        assert llm.calls == 2

    def test_no_llm_call_while_a_transaction_or_writer_lock_is_open(self):
        observed = []
        conn_box = {}

        def guarded(system, user):
            conn = conn_box["conn"]
            observed.append((conn.depth, conn.writer_locked))
            return responder(system, user)

        conn, _, llm, _, planned = _fake_stack(llm=MockLLM(responder=guarded))
        conn_box["conn"] = conn
        planned.ingest(corpus_events("s1"))
        assert observed == [], "deferred enqueue calls no model"
        receipt = planned.materialize("s1")
        assert receipt.resolved_count == len(CORPUS) and receipt.replanned == 0
        assert observed and all(o == (0, False) for o in observed)

    def test_stale_preplan_is_replanned_under_the_lock(self):
        conn, _, llm, _, planned = _fake_stack()
        planned.ingest(corpus_events("s1")[:1])
        conn.epoch_bump_on_read = 1  # a commit lands between plan and lock
        receipt = planned.materialize("s1")
        assert receipt.resolved == (1,) and receipt.replanned == 1
        # The in-lock re-plan issues the same prompt (nothing changed in this
        # fake), so it is served from gem_sem_cache: one real call, one hit.
        assert llm.calls == 1 and receipt.cache_misses == 1 and receipt.cache_hits == 1

    def test_until_bounds_the_fifo_prefix(self):
        conn, _, _, _, planned = _fake_stack()
        planned.ingest(corpus_events("s1")[:4])
        first = planned.materialize("s1", until=2)
        assert first.resolved == (1, 2)
        assert planned.pending_open("s1") == 2
        assert planned.materialize("s1", by="on_read").resolved == (3, 4)

    def test_a_failed_derivation_is_marked_and_the_walk_continues(self):
        def flaky(system, user):
            if "Beta" in json.loads(user)["excerpt"].split(";")[0]:
                return "not json at all"
            return responder(system, user)

        conn, _, _, _, planned = _fake_stack(llm=MockLLM(responder=flaky))
        planned.max_rejection_rate = 0.0  # an unparseable extraction aborts
        planned.ingest(corpus_events("s1")[:3])
        receipt = planned.materialize("s1")
        assert receipt.resolved == (1, 3) and receipt.failed == (2,)
        assert conn.pending[2]["status"] == "failed"

    def test_eager_ingest_first_materializes_an_open_prefix(self):
        conn, store, llm, client, deferred = _fake_stack()
        deferred.ingest(corpus_events("s1")[:2])
        eager = PlannedIngest(
            deferred.operator, deferred.strategy, IngestPlan(mode=EAGER)
        )
        result = eager.ingest(corpus_events("s1")[2:3])
        assert result.committed
        assert all(p["status"] == "resolved" for p in conn.pending.values())


class TestEquivalenceFake:
    def test_eager_per_event_and_deferred_materialize_write_the_same_trace(self):
        events = corpus_events("s1")
        eager_conn, _, _, _, eager = _fake_stack(plan=IngestPlan(mode=EAGER))
        for event in events:
            assert eager.ingest([event]).committed
        deferred_conn, _, _, _, deferred = _fake_stack()
        assert deferred.ingest(events).pending_enqueued
        assert write_trace(deferred_conn) == []
        deferred.materialize("s1")
        assert write_trace(eager_conn) == write_trace(deferred_conn)
        assert write_trace(eager_conn), "the trace must not be vacuous"


class TestReplayFromCache:
    def test_offline_replay_reproduces_the_trace_and_misses_raise(self):
        cache = DictCache()
        online_conn, _, llm, client, online = _fake_stack(cache=cache)
        online.ingest(corpus_events("s1"))
        online_receipt = online.materialize("s1")
        assert online_receipt.cache_misses == len(CORPUS) and llm.calls == len(CORPUS)

        offline_llm = MockLLM(responder=responder)
        replay_conn, _, _, _, replay = _fake_stack(
            cache=cache, llm=offline_llm, offline=True
        )
        replay.ingest(corpus_events("s1"))
        receipt = replay.materialize("s1")
        assert offline_llm.calls == 0, "replay must not touch the network"
        assert receipt.cache_hits == len(CORPUS) and receipt.cache_misses == 0
        assert write_trace(replay_conn) == write_trace(online_conn)
        # strategy meter counts only real requests
        assert all(r.cost.llm_calls == 0 for r in receipt.results)

        novel = InteractionEvent(
            scope_id="s1",
            external_id="new",
            content="topic=Omega;field=x;value=y;date=2026-05-01",
        )
        replay.ingest([novel])
        with pytest.raises(CacheMiss):
            replay.materialize("s1")
        assert not replay_conn.ran("status = 'failed'"), (
            "a replay miss is not a failure"
        )


class TestCachedCompletionClient:
    def test_key_covers_model_and_prompt_version_and_usage_marks_hits(self):
        cache = DictCache()
        llm = MockLLM(responder=lambda s, u: "{}")
        a = CachedCompletionClient(llm, cache, prompt_version="v1")
        assert a.complete("sys", "u")[1]["cache_hit"] is False
        text, usage = a.complete("sys", "u")
        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit": True}
        assert llm.calls == 1 and a.stats.snapshot() == (1, 1)
        CachedCompletionClient(llm, cache, prompt_version="v2").complete("sys", "u")
        CachedCompletionClient(llm, cache, prompt_version="v1", model="other").complete(
            "sys", "u"
        )
        assert llm.calls == 3
        with pytest.raises(CacheMiss):
            CachedCompletionClient(
                llm, cache, prompt_version="v1", offline=True
            ).complete("s", "new")

    def test_strategy_style_tuple_clients_are_supported(self):
        class TupleClient:
            model = "tuple-model"

            def complete(self, system, user):
                return "hello", {"prompt_tokens": 3, "completion_tokens": 2}

        client = CachedCompletionClient(TupleClient(), DictCache(), prompt_version="v1")
        assert client.complete("a", "b") == (
            "hello",
            {"prompt_tokens": 3, "completion_tokens": 2, "cache_hit": False},
        )
        assert client.stats.prompt_tokens == 3


class TestPolicyTriggerD6:
    def _tx(self, operator="ingest", fields=1):
        conn = StatefulFakeConn()
        store = GemStore(conn, dim=DIM)
        tx = Tx(store, operator, "s1", "construction")
        tx.delta.fields_appended = fields
        tx.delta.propagated_units = [7]
        return conn, tx

    def test_default_engine_fires_no_action(self):
        conn, tx = self._tx()
        PolicyEngine(extra=seed_policies()).evaluate(tx)
        assert not conn.ran("needs_revision")
        assert tx.actions_fired == []

    def test_opt_in_engine_fires_matching_actions_before_conditions(self):
        conn, tx = self._tx()
        PolicyEngine(extra=seed_policies(), fire_actions=True).evaluate(tx)
        assert tx.actions_fired == ["propagate-on-change"]
        flag = conn.ran("jsonb_set\\(metadata, '\\{needs_revision\\}'")
        assert flag, "flag_for_revision must have run"
        assert PolicyEngine.events_for(tx) == [PolicyEvent.FIELD_UPDATED]

    def test_retrieval_action_is_scoped(self):
        conn, tx = self._tx(operator="retrieve", fields=0)
        PolicyEngine(extra=seed_policies(), fire_actions=True).evaluate(tx)
        assert tx.actions_fired == ["reinforce-on-read"]
        sql, params = next(
            (s, p) for s, p in conn.statements if "extension_candidate" in s
        )
        assert "scope_id = %s" in sql and params[0] == "s1"

    def test_planned_ingest_runs_maintenance_after_a_flagging_commit(self):
        class Reviser:
            calls = []

            def revise(self, scope_id, **kwargs):
                self.calls.append((scope_id, kwargs))
                return "revised"

        conn = StatefulFakeConn()
        store = GemStore(conn, dim=DIM)
        store._edge_types = {"extension": 1, "association": 2}
        reviser = Reviser()

        class Linking(DeterministicIngestStrategy):
            def plan(self, events, view):
                ops = super().plan(events, view)
                ops.append(
                    {
                        "kind": "link",
                        "edge_kind": "extension",
                        "rel": "r",
                        "src": 1,
                        "dst": 2,
                    }
                )
                return ops

        conn._answer_orig = conn._answer

        def answer(text, params):
            if text.startswith("SELECT DISTINCT dst FROM gem_edge"):
                return FakeCursor([(2,)])
            return conn._answer_orig(text, params)

        conn._answer = answer
        planned = PlannedIngest(
            IngestOperator(store, embedder=StubEmbedder()),
            Linking(chunker=OneChunk()),
            IngestPlan(mode=EAGER),
            maintenance=True,
            reviser=reviser,
        )
        result = planned.ingest(corpus_events("s1")[:1])
        assert result.committed and result.delta.propagated_units == (2,)
        assert reviser.calls == [("s1", {"kinds": ("propagate",)})]
        off = PlannedIngest(
            planned.operator, planned.strategy, IngestPlan(mode=EAGER), reviser=reviser
        )
        off.ingest(corpus_events("s1")[1:2])
        assert len(reviser.calls) == 1, "maintenance is opt-in"


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
    from bench.agent_memory.gem.salience import ExponentialSalience

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


def _clean(conn, *scopes):
    for scope in scopes:
        conn.execute("DELETE FROM gem_unit WHERE scope_id = %s", (scope,))
        conn.execute("DELETE FROM gem_transition WHERE scope_id = %s", (scope,))
        conn.execute("DELETE FROM gem_pending WHERE scope_id = %s", (scope,))


def materialized_state(conn, scope_id):
    """Mat(S) for a scope with ids and timestamps projected away (ids -> titles)."""
    units = conn.execute(
        "SELECT title, summary, state, metadata FROM gem_unit WHERE scope_id = %s ORDER BY title",
        (scope_id,),
    ).fetchall()
    fields = conn.execute(
        "SELECT u.title, fv.field, fv.value, fv.valid_from::text, fv.valid_to::text,"
        " fv.state, fv.extractor_model, fv.prompt_version, fv.source_external_ids,"
        " (SELECT n.value FROM gem_field_value n WHERE n.id = fv.superseded_by)"
        " FROM gem_field_value fv JOIN gem_unit u ON u.id = fv.unit_id"
        " WHERE u.scope_id = %s ORDER BY 1, 2, 4, 3",
        (scope_id,),
    ).fetchall()
    edges = conn.execute(
        "SELECT s.title, d.title, e.kind, e.rel, e.weight, e.tombstoned_at IS NULL"
        " FROM gem_edge e JOIN gem_unit s ON s.id = e.src JOIN gem_unit d ON d.id = e.dst"
        " WHERE s.scope_id = %s ORDER BY 1, 2, 3",
        (scope_id,),
    ).fetchall()
    return {"units": units, "fields": fields, "edges": edges}


def writer_lock_free(dsn):
    import psycopg

    side = psycopg.connect(dsn, autocommit=True)

    def check() -> bool:
        # In autocommit an xact lock is released at statement end, so this only
        # PROBES: true iff nobody else holds the writer lock right now.
        return bool(
            side.execute(
                "SELECT pg_try_advisory_xact_lock(hashtext(%s))", (WRITER_LOCK_KEY,)
            ).fetchone()[0]
        )

    return side, check


@live
class TestEquivalenceLive:
    def test_mat_eager_equals_mat_deferred_and_no_llm_under_the_writer_lock(
        self, live_memory
    ):
        from bench.agent_memory.semantic.cache import SemCache

        conn = live_memory.store.conn
        eager_scope, deferred_scope = "plan_eq_eager", "plan_eq_deferred"
        _clean(conn, eager_scope, deferred_scope)
        side, lock_free = writer_lock_free(DSN)
        seen: list[bool] = []

        def watched(system, user):
            seen.append(lock_free())
            return responder(system, user)

        try:
            llm = MockLLM(responder=watched)
            client = CachedCompletionClient(
                llm, SemCache(conn), prompt_version=PROMPT_VERSION
            )
            strategy = _llm_strategy(client)
            eager = PlannedIngest(
                live_memory, strategy, IngestPlan(mode=EAGER), cache_client=client
            )
            for event in corpus_events(eager_scope):
                result = eager.ingest([event])
                assert result.committed, result.aborted_reason
            assert seen and not any(seen), "eager (legacy shape) plans under the lock"

            seen.clear()
            deferred = PlannedIngest(
                live_memory,
                strategy,
                IngestPlan(mode=DEFERRED, trigger="batch"),
                cache_client=client,
            )
            receipt = deferred.ingest(corpus_events(deferred_scope))
            assert len(receipt.pending_enqueued) == len(CORPUS)
            assert seen == [], "deferred enqueue made an LLM call"
            assert materialized_state(conn, deferred_scope)["units"] == []
            done = deferred.materialize(deferred_scope)
            assert done.resolved == receipt.pending_enqueued and done.failed == ()
            # Every REAL call (cache misses; the first event's empty-slate prompt
            # is shared with the eager scope and hits) saw the writer lock free.
            assert len(seen) == done.cache_misses and all(seen), (
                "LLM called under the writer lock"
            )
            assert done.cache_hits + done.cache_misses == len(CORPUS)
            assert deferred.pending_open(deferred_scope) == 0
        finally:
            side.close()

        eager_state = materialized_state(conn, eager_scope)
        deferred_state = materialized_state(conn, deferred_scope)
        assert eager_state["units"] and eager_state["edges"]
        assert any(row[9] is not None for row in eager_state["fields"]), (
            "supersession exercised"
        )
        assert eager_state == deferred_state

    def test_offline_replay_from_gem_sem_cache_reproduces_state(self, live_memory):
        """P1: replay with the network disabled. Batch mode, no embedder, so the
        prompts carry no unit ids and two scopes issue identical prompts."""
        from bench.agent_memory.semantic.cache import SemCache

        conn = live_memory.store.conn
        first, second = "plan_replay_a", "plan_replay_b"
        _clean(conn, first, second)
        corpus = [
            ("Kappa", "owner", "kim", "2026-04-01", ""),
            ("Lambda", "city", "oslo", "2026-04-02", "Kappa"),
            ("Mu", "owner", "max", "2026-04-03", ""),
        ]
        online_llm = MockLLM(responder=responder)
        online = CachedCompletionClient(
            online_llm, SemCache(conn), prompt_version="replay-test-v1"
        )
        strategy = LLMMediatedIngestStrategy(
            client=online, model=online_llm.model, mode=MODE_BATCH, chunker=OneChunk()
        )
        planned = PlannedIngest(
            live_memory, strategy, IngestPlan(mode=DEFERRED), cache_client=online
        )
        planned.ingest(corpus_events(first, corpus))
        assert planned.materialize(first).resolved_count == 3

        offline_llm = MockLLM(responder=responder)
        offline = CachedCompletionClient(
            offline_llm,
            SemCache(conn),
            prompt_version="replay-test-v1",
            offline=True,
            model=online_llm.model,
        )
        replay_strategy = LLMMediatedIngestStrategy(
            client=offline, model=online_llm.model, mode=MODE_BATCH, chunker=OneChunk()
        )
        replay = PlannedIngest(
            live_memory,
            replay_strategy,
            IngestPlan(mode=DEFERRED),
            cache_client=offline,
        )
        replay.ingest(corpus_events(second, corpus))
        receipt = replay.materialize(second)
        assert offline_llm.calls == 0 and receipt.cache_hits == 3
        assert materialized_state(conn, first) == materialized_state(conn, second)
