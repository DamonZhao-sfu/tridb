"""Actual semantic-request accounting for an exclusive private PG cluster.

The existing mailbox is cluster-wide: it carries neither database identity nor
execution identity. A different database/schema inside a shared cluster is NOT
an attribution boundary. This adapter therefore requires a caller-owned private
cluster lease, a single querying backend, and no other worker. It does not start
a database, mutate the existing worker, or contact a model at import time.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Protocol

from trimem.llm import (
    EXTRACT_PROMPT,
    SCORE_PROMPT,
    SYSTEM_PROMPT,
    Verdict,
    extract_prompt,
    parse_label,
    parse_score,
    parse_verdict,
    user_prompt,
)

from .compiler import NativePlanSpec
from .ledger import MissingUsageError, TokenBudgetExhausted, TokenLedger


class SemanticUsageError(MissingUsageError):
    """Unusable attribution, retaining all observed physical-request costs."""

    def __init__(self, message: str, usage_events: Sequence[Mapping[str, Any]] = ()):
        super().__init__(message)
        self.usage_events = [dict(event) for event in usage_events]


class SemanticBudgetExhausted(TokenBudgetExhausted):
    usage_complete = True
    stop_reason = "semantic_token_budget"

    def __init__(self, message: str, usage_events: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.usage_events = [dict(event) for event in usage_events]


class ChatTransport(Protocol):
    """The instrumented clients.OpenAICompatibleTransport contract.

    Each physical request/retry must use the supplied ledger and perform serial
    request admission. Returning only final-response tokens is insufficient.
    """

    model: str

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        ledger: TokenLedger,
        category: str,
        seed: int,
        max_output_tokens: int,
    ) -> Any: ...


class MeteredSemanticJudge:
    """Worker-compatible judge with no hidden retries or parallel requests.

    Transport owns request-level accounting and exact prompt admission. The
    existing OpenAIJudge is deliberately unsupported: it hides retries and turns
    missing provider usage into zero. Invalid semantic JSON is still a paid call.
    """

    def __init__(
        self,
        transport: ChatTransport,
        ledger: TokenLedger,
        *,
        max_output_tokens: int = 16,
    ):
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.transport, self.ledger = transport, ledger
        self.model = transport.model
        self.max_output_tokens = max_output_tokens

    def _many(
        self, pairs, *, system, parser, category, renderer=user_prompt, output_cap=None
    ):
        answers = []
        for pair in pairs:
            before = {event.request_id for event in self.ledger.events}
            started = time.monotonic()
            try:
                response = self.transport.complete(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": renderer(*pair)},
                    ],
                    ledger=self.ledger,
                    category=category,
                    seed=0,
                    max_output_tokens=output_cap or self.max_output_tokens,
                )
            except TokenBudgetExhausted:
                # Admission can stop before dispatch; no fictional paid request.
                raise
            except BaseException:
                if not any(
                    event.request_id not in before for event in self.ledger.events
                ):
                    self.ledger.record(
                        "unattributed-" + uuid.uuid4().hex,
                        category,
                        None,
                        None,
                        "failure",
                    )
                raise
            fresh = [
                event for event in self.ledger.events if event.request_id not in before
            ]
            reported_ids = tuple(getattr(response, "request_ids", ()))
            if (
                not fresh
                or not reported_ids
                or set(reported_ids) != {event.request_id for event in fresh}
                or any(
                    event.category not in {category, "retry"}
                    or not event.usage_complete
                    for event in fresh
                )
            ):
                if not fresh or all(event.usage_complete for event in fresh):
                    self.ledger.record(
                        "unattributed-" + uuid.uuid4().hex,
                        category,
                        None,
                        None,
                        "failure",
                    )
                raise MissingUsageError(
                    "semantic transport did not account for every physical request"
                )
            answers.append(
                Verdict(
                    parser(response.content),
                    response.input_tokens,
                    response.output_tokens,
                    time.monotonic() - started,
                )
            )
        return answers

    def judge_many(self, pairs):
        return self._many(
            pairs,
            system=SYSTEM_PROMPT,
            parser=parse_verdict,
            category="semantic_filter",
        )

    def score_many(self, pairs):
        return self._many(
            pairs, system=SCORE_PROMPT, parser=parse_score, category="semantic_rank"
        )

    def extract_many(self, pairs):
        return self._many(
            pairs,
            system=EXTRACT_PROMPT,
            parser=parse_label,
            category="retrieval",
            renderer=extract_prompt,
            output_cap=max(48, self.max_output_tokens),
        )


@dataclass(frozen=True)
class ExclusiveClusterLease:
    system_identifier: str
    database_name: str
    query_backend_pid: int
    isolation_evidence: str
    exclusive_worker: bool
    exclusive_semantic_queries: bool

    def __post_init__(self):
        if (
            not self.system_identifier
            or not self.database_name
            or not self.isolation_evidence.strip()
        ):
            raise ValueError(
                "private cluster identity and isolation evidence are required"
            )
        if type(self.query_backend_pid) is not int or self.query_backend_pid <= 0:
            raise ValueError(
                "query_backend_pid must identify the selected-query backend"
            )
        if (
            self.exclusive_worker is not True
            or self.exclusive_semantic_queries is not True
        ):
            raise ValueError(
                "cluster-wide exclusive worker and semantic query ownership are required"
            )


@dataclass
class _Execution:
    execution_id: str
    ledger: TokenLedger
    connection: Any
    worker: Any
    stack: ExitStack
    stop: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    error: BaseException | None = None


class _TakenBatchConnection:
    """Remember bounded mailbox slot/seq pairs so a failed judge wakes its query.

    The legacy Worker.step does not catch transport/admission exceptions. Its
    normal _fail protocol completes slots with NULL, which safely raises in the
    waiting backend. No graph rows or retrieval candidates are buffered here.
    """

    def __init__(self, connection):
        self.connection = connection
        self.taken: list[tuple[int, int]] = []

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def execute(self, sql, *args, **kwargs):
        cursor = self.connection.execute(sql, *args, **kwargs)
        if "FROM trimem.take(" not in sql:
            return cursor
        owner = self

        class TakenResult:
            def fetchall(self):
                rows = cursor.fetchall()
                owner.taken = [(row[0], row[1]) for row in rows]
                return rows

        return TakenResult()

    def fail_taken(self):
        if self.taken:
            self.connection.execute(
                "SELECT trimem.complete(%s::int[], %s::bigint[], %s::boolean[])",
                (
                    [slot for slot, _ in self.taken],
                    [seq for _, seq in self.taken],
                    [None] * len(self.taken),
                ),
            )


def _default_worker(judge):
    # Worker.run would open its own DSN. We only use step with the connection
    # factory that was checked against the private-cluster lease.
    from trimem.worker import Worker

    return Worker("", judge, max_take=1, take_ms=50)


class ExclusiveSemanticUsageCollector:
    """One owned worker during one selected execution, with actual token receipts.

    ``connection_factory`` returns a fresh autocommit connection context manager
    to the same private DB as the query backend. ``judge_factory(ledger)`` creates
    MeteredSemanticJudge using the instrumented client. set_token_budget snapshots
    the main ledger's remaining budget; no other agent request may run until
    finish and receipt import. This is serial admission, not a provider billing
    or timeout guarantee. Explicit lease claims remain caller responsibilities.
    """

    _guard = threading.Lock()
    _active_clusters: ClassVar[dict[str, ExclusiveSemanticUsageCollector]] = {}

    def __init__(
        self,
        *,
        lease: ExclusiveClusterLease,
        connection_factory: Callable[[], Any],
        judge_factory: Callable[[TokenLedger], MeteredSemanticJudge],
        worker_factory: Callable[[MeteredSemanticJudge], Any] = _default_worker,
        join_timeout: float = 30.0,
    ):
        if not 0 < join_timeout <= 60:
            raise ValueError("join_timeout must be in (0,60]")
        self.lease, self.connection_factory = lease, connection_factory
        self.judge_factory, self.worker_factory = judge_factory, worker_factory
        self.join_timeout = join_timeout
        self._budget: int | None = None
        self._run: _Execution | None = None
        self._poisoned = False

    def set_token_budget(self, remaining_tokens: int | None) -> None:
        if remaining_tokens is not None and (
            type(remaining_tokens) is not int or remaining_tokens < 0
        ):
            raise ValueError("remaining_tokens must be nonnegative or None")
        if self._run is not None:
            raise RuntimeError("cannot change a running execution's token budget")
        self._budget = remaining_tokens

    def _check_private_connection(self, connection):
        if connection.autocommit is not True:
            raise ValueError("worker connection must use autocommit")
        identity, own_pid, database = connection.execute(
            "SELECT system_identifier::text, pg_backend_pid(), current_database() FROM pg_control_system()"
        ).fetchone()
        if (
            identity != self.lease.system_identifier
            or database != self.lease.database_name
        ):
            raise SemanticUsageError(
                "worker connection does not match private cluster/database lease"
            )
        clients = connection.execute(
            "SELECT pid, datname FROM pg_stat_activity WHERE backend_type='client backend'"
        ).fetchall()
        expected = {own_pid, self.lease.query_backend_pid}
        if {row[0] for row in clients} != expected or any(
            row[1] != database for row in clients
        ):
            raise SemanticUsageError(
                "private cluster contains unexpected client backends"
            )
        occupancy = dict(
            connection.execute("SELECT state, slots FROM trimem.status()").fetchall()
        )
        if set(occupancy) != {"free", "posted", "taken", "done"} or any(
            occupancy[name] for name in ("posted", "taken", "done")
        ):
            raise SemanticUsageError("mailbox is not empty before selected execution")

    def _release_cluster(self):
        with self._guard:
            if self._active_clusters.get(self.lease.system_identifier) is self:
                del self._active_clusters[self.lease.system_identifier]

    def start(self, execution_id: str, plan: NativePlanSpec):
        if not execution_id or self._run is not None or self._poisoned:
            raise RuntimeError(
                "collector requires a new execution and an unpoisoned idle state"
            )
        with self._guard:
            if self.lease.system_identifier in self._active_clusters:
                raise SemanticUsageError("another collector owns this cluster mailbox")
            self._active_clusters[self.lease.system_identifier] = self
        stack = ExitStack()
        try:
            connection = stack.enter_context(self.connection_factory())
            self._check_private_connection(connection)
            ledger = TokenLedger(self._budget)
            judge = self.judge_factory(ledger)
            if (
                not isinstance(judge, MeteredSemanticJudge)
                or judge.ledger is not ledger
            ):
                raise TypeError(
                    "judge_factory must bind MeteredSemanticJudge to the execution ledger"
                )
            worker = self.worker_factory(judge)
            if (
                worker.judge is not judge
                or worker.max_take != 1
                or not 0 < worker.take_ms <= 100
            ):
                raise ValueError(
                    "owned worker must use this judge, max_take=1 and take_ms<=100"
                )
            run = _Execution(execution_id, ledger, connection, worker, stack)
            self._run = run
            run.thread = threading.Thread(
                target=self._serve,
                args=(run,),
                name="trimem-sem-" + execution_id,
                daemon=True,
            )
            run.thread.start()
            return execution_id
        except BaseException:
            stack.close()
            self._run = None
            self._release_cluster()
            raise

    def _serve(self, run):
        connection = _TakenBatchConnection(run.connection)
        try:
            while not run.stop.is_set():
                connection.taken = []
                run.worker.step(connection)
        except BaseException as exc:  # noqa: BLE001 -- propagate thread failures through finish
            run.error = exc
            run.stop.set()
            try:
                connection.fail_taken()
            except Exception as completion_error:  # noqa: BLE001 -- preserve original error
                # The query may already have cancelled/closed; native seq checks
                # prevent late completion from affecting a later execution.
                run.error.add_note(
                    f"mailbox error completion failed: {type(completion_error).__name__}"
                )
        finally:
            try:
                run.stack.close()
            except BaseException as exc:  # noqa: BLE001 -- preserve connection-close failure
                run.error = run.error or exc
            finally:
                run.finished.set()
                if self._poisoned:
                    self._release_cluster()

    @staticmethod
    def _events(run):
        return [
            dict(asdict(event), execution_id=run.execution_id)
            for event in run.ledger.events
        ]

    def finish(self, handle):
        run = self._run
        if run is None or handle != run.execution_id:
            raise RuntimeError("unknown or already finished semantic execution")
        run.stop.set()
        run.worker.stop()
        run.thread.join(self.join_timeout)
        if run.thread.is_alive():
            self._poisoned = True
            if run.finished.is_set():
                self._release_cluster()
            # Keep ownership while the transport may still be spending tokens.
            raise SemanticUsageError(
                "semantic worker did not stop; collector is poisoned", self._events(run)
            )
        self._run = None
        self._release_cluster()
        events = self._events(run)
        if isinstance(run.error, TokenBudgetExhausted):
            if not run.ledger.objective_usage_complete:
                raise SemanticUsageError(
                    "semantic budget stopped with incomplete usage", events
                )
            raise SemanticBudgetExhausted(str(run.error), events)
        if run.error is not None:
            raise SemanticUsageError(
                f"semantic worker failed: {type(run.error).__name__}: {run.error}",
                events,
            )
        if not run.ledger.objective_usage_complete:
            raise SemanticUsageError("semantic provider usage is incomplete", events)
        return events
