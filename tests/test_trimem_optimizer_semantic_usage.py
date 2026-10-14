"""Physical request accounting and private-mailbox ownership; mock HTTP only."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import httpx
import pytest

from trimem.optimizer.clients import OpenAICompatibleTransport
from trimem.optimizer.compiler import QueryContext, compile_candidate
from trimem.optimizer.ledger import MissingUsageError, TokenLedger
from trimem.optimizer.schema import CandidateSpec
from trimem.optimizer.semantic_usage import (
    ExclusiveClusterLease,
    ExclusiveSemanticUsageCollector,
    MeteredSemanticJudge,
    SemanticBudgetExhausted,
    SemanticUsageError,
)


def transport(responses, *, retries=0):
    pending = iter(responses)
    calls = []

    def handle(request):
        calls.append(request)
        status, body = next(pending)
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    api = OpenAICompatibleTransport(
        model="test-model",
        base_url="https://fixture.invalid/v1",
        prompt_token_count=lambda _: 10,
        model_context_window=100,
        tokenizer_id="mock-chat-v1",
        max_retries=retries,
        http_client=client,
        temperature=0,
    )
    return api, calls, client


def reply(content='{"verdict":true}', *, usage=True):
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage:
        body["usage"] = {"prompt_tokens": 10, "completion_tokens": 2}
    return body


def test_metered_judge_preserves_failed_request_retry_and_invalid_reply_costs():
    api, calls, http = transport(
        [(503, reply()), (200, reply()), (200, reply("invalid"))], retries=1
    )
    ledger = TokenLedger(200)
    judge = MeteredSemanticJudge(api, ledger)
    try:
        verdicts = judge.judge_many([("p", "one"), ("p", "two")])
    finally:
        http.close()
    assert [result.value for result in verdicts] == [True, None]
    assert len(calls) == 3 and ledger.total_tokens == 36
    assert [event.status for event in ledger.events] == [
        "failure",
        "success",
        "success",
    ]
    assert [event.category for event in ledger.events] == [
        "semantic_filter",
        "retry",
        "semantic_filter",
    ]


def test_missing_usage_is_never_zero_and_score_uses_its_own_category():
    api, _, http = transport([(200, reply('{"score":8}')), (200, reply(usage=False))])
    ledger = TokenLedger(200)
    judge = MeteredSemanticJudge(api, ledger)
    try:
        assert judge.score_many([("p", "one")])[0].value == 0.8
        with pytest.raises(MissingUsageError):
            judge.judge_many([("p", "two")])
    finally:
        http.close()
    assert ledger.events[0].category == "semantic_rank"
    assert ledger.events[1].input_tokens is None
    with pytest.raises(MissingUsageError):
        _ = ledger.total_tokens


def test_uninstrumented_transport_cannot_masquerade_as_zero_cost():
    class Unmetered:
        model = "test-model"

        def complete(self, *args, **kwargs):
            return SimpleNamespace(
                content='{"verdict":true}',
                input_tokens=10,
                output_tokens=2,
                request_ids=("not-recorded",),
            )

    ledger = TokenLedger(100)
    with pytest.raises(MissingUsageError, match="every physical"):
        MeteredSemanticJudge(Unmetered(), ledger).judge_many([("p", "text")])
    assert not ledger.objective_usage_complete


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0]

    def fetchall(self):
        return self.rows


class PrivateConnection:
    autocommit = True

    def __init__(self, *, extras=(), occupied=0):
        self.extras, self.occupied, self.closed = extras, occupied, False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def execute(self, sql):
        if "pg_control_system" in sql:
            return FakeResult([("private-system", 202, "private-db")])
        if "pg_stat_activity" in sql:
            return FakeResult([(101, "private-db"), (202, "private-db"), *self.extras])
        if "trimem.status" in sql:
            return FakeResult(
                [("free", 100), ("posted", self.occupied), ("taken", 0), ("done", 0)]
            )
        raise AssertionError(sql)


class OneCallWorker:
    max_take, take_ms = 1, 10

    def __init__(self, judge, done, *, invoke=True):
        self.judge, self.done, self.invoke = judge, done, invoke
        self._stopped = threading.Event()
        self._called = False

    def step(self, connection):
        if not self._called:
            self._called = True
            try:
                if self.invoke:
                    self.judge.judge_many([("p", "text")])
            finally:
                self.done.set()
        self._stopped.wait(0.01)

    def stop(self):
        self._stopped.set()


def lease():
    return ExclusiveClusterLease(
        "private-system",
        "private-db",
        101,
        "owned temporary cluster, no other clients/workers",
        True,
        True,
    )


def collector(api, *, connection=None, invoke=True, worker_factory=None, timeout=1):
    done = threading.Event()
    connection = connection or PrivateConnection()
    instance = ExclusiveSemanticUsageCollector(
        lease=lease(),
        connection_factory=lambda: connection,
        judge_factory=lambda ledger: MeteredSemanticJudge(api, ledger),
        worker_factory=worker_factory
        or (lambda judge: OneCallWorker(judge, done, invoke=invoke)),
        join_timeout=timeout,
    )
    return instance, done, connection


def plan():
    return compile_candidate(
        CandidateSpec("vgr_score_v1"), QueryContext("fixture", (1.0,))
    )


def test_collector_owns_worker_and_returns_actual_execution_tagged_events():
    api, calls, http = transport([(200, reply())])
    instance, done, conn = collector(api)
    instance.set_token_budget(100)
    try:
        handle = instance.start("execution-a", plan())
        assert done.wait(1)
        events = instance.finish(handle)
    finally:
        http.close()
    assert conn.closed and len(calls) == 1 and len(events) == 1
    assert events[0]["execution_id"] == "execution-a"
    assert events[0]["input_tokens"] + events[0]["output_tokens"] == 12
    assert "private-system" not in instance._active_clusters


@pytest.mark.parametrize(
    "connection",
    [PrivateConnection(extras=[(303, "other-db")]), PrivateConnection(occupied=1)],
)
def test_nonexclusive_or_dirty_mailbox_rejected_before_provider(connection):
    api, calls, http = transport([])
    instance, _, conn = collector(api, connection=connection)
    try:
        with pytest.raises(SemanticUsageError):
            instance.start("blocked", plan())
    finally:
        http.close()
    assert conn.closed and calls == []
    assert "private-system" not in instance._active_clusters


def test_another_collector_cannot_steal_mailbox_and_cache_only_is_zero_requests():
    api, calls, http = transport([])
    first, done, _ = collector(api, invoke=False)
    second, _, _ = collector(api, invoke=False)
    try:
        handle = first.start("first", plan())
        assert done.wait(1)
        with pytest.raises(SemanticUsageError, match="another collector"):
            second.start("second", plan())
        assert first.finish(handle) == []
    finally:
        http.close()
    assert not calls


def test_budget_rejection_dispatches_nothing_and_is_not_missing_usage():
    api, calls, http = transport([])
    instance, done, conn = collector(api)
    instance.set_token_budget(10)  # exact prompt needs 10, no generated output fits
    try:
        handle = instance.start("budget", plan())
        assert done.wait(1)
        with pytest.raises(SemanticBudgetExhausted) as raised:
            instance.finish(handle)
    finally:
        http.close()
    assert raised.value.usage_complete is True and raised.value.usage_events == []
    assert calls == [] and conn.closed


def test_worker_failure_includes_known_and_unknown_request_costs():
    api, calls, http = transport([(200, reply(usage=False))])
    instance, done, conn = collector(api)
    try:
        handle = instance.start("unknown", plan())
        assert done.wait(1)
        with pytest.raises(SemanticUsageError) as raised:
            instance.finish(handle)
    finally:
        http.close()
    assert conn.closed and len(calls) == 1
    assert raised.value.usage_events[0]["input_tokens"] is None
    assert raised.value.usage_events[0]["execution_id"] == "unknown"


def test_timeout_poison_keeps_lease_until_owned_thread_really_stops():
    entered, unblock = threading.Event(), threading.Event()

    class StuckWorker(OneCallWorker):
        def step(self, connection):
            entered.set()
            unblock.wait(2)

    api, _, http = transport([])
    instance, _, conn = collector(
        api, worker_factory=lambda judge: StuckWorker(judge, entered), timeout=0.01
    )
    try:
        handle = instance.start("stuck", plan())
        assert entered.wait(1)
        with pytest.raises(SemanticUsageError, match="poisoned"):
            instance.finish(handle)
        assert instance._active_clusters["private-system"] is instance
    finally:
        unblock.set()
        instance._run.thread.join(1)
        http.close()
    assert conn.closed and "private-system" not in instance._active_clusters
    with pytest.raises(RuntimeError, match="unpoisoned"):
        instance.start("again", plan())


def test_lease_requires_cluster_level_exclusivity_not_a_schema_name():
    with pytest.raises(ValueError, match="exclusive"):
        ExclusiveClusterLease("system", "schema-copy", 1, "shared cluster", False, True)
