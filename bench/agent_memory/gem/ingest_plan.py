"""Ingest as a plan: eager vs deferred derivation over the pending set Q (plan P2).

    Ingest_pi(x, S) -> <S', r>      pi in {eager, deferred(trigger)}

``IngestOperator.ingest`` is the EAGER physical shape: T, C and delta fused and
committed in one transaction. :class:`PlannedIngest` adds the DEFERRED shape:

* **deferred** — in ONE short transaction, enqueue each raw ``InteractionEvent``
  into ``gem_pending`` (derivation ``'llm_ingest'``, ``target_ref`` = the event's
  external id or a stable content hash, ``snapshot_seq`` = a per-scope logical
  sequence persisted in ``gem_pending`` itself). No model call, no writer lock,
  no ``gem_*`` state change. The receipt lists the pending ids.
* :meth:`PlannedIngest.materialize` resolves pending items in FIFO
  ``snapshot_seq`` order, running the real ingest for each and marking the item
  resolved IN THE SAME TRANSACTION as the derivation it stands for.
  ``trigger`` (``background`` | ``batch`` | ``on_read``) records who is expected
  to resolve an item; it does not change the order — see D2 below.

**D2 — why on-read resolution materializes the whole FIFO prefix.** GEM's
LLM-mediated derivation is NOT a function of ``x`` alone: its candidate slate
``C`` is read from the CURRENT state (``find_similar`` + title lookups,
``strategies/llm_mediated.py``), and that state includes every derivation
committed before it. The paper's claim ``Mat(S_eager) == Mat(S_deferred)``
therefore holds only if each deferred derivation sees exactly the state an eager
plan would have shown it — i.e. derivations run in arrival order with nothing
skipped. Resolving only the items a read "touches" would let a later item run
against a state missing an earlier one and change its prompt, its cache key and
its result. So on-read (strong) resolution materializes the entire FIFO prefix
of the scope, never a subset; and an EAGER ingest into a scope with open items
first materializes that prefix for the same reason.

Granularity matters too: one deferred item is one event, so the equivalent
eager plan ingests one event per transition. An eager BATCH of N events is a
different derivation (within one ``plan()`` no event sees another's writes).

**No model call under the writer lock.** Deferred enqueue never calls a model.
``materialize`` plans each item OUTSIDE any transaction
(:meth:`IngestOperator.plan_outside`) and applies the ops under the writer lock
only if no transition committed in the scope meanwhile (TOCTOU witness
:func:`~bench.agent_memory.gem.ingest.scope_epoch`); otherwise it re-plans
under the lock, which is correct but slower, and is counted as ``replanned``.

**D6 maintenance.** With ``maintenance=True`` a committed derivation that
flagged extension dependents (``delta.propagated_units``) is followed by the
policy-triggered ``revise(kinds=('propagate',))`` the interface doc promises.
Off by default.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from bench.agent_memory.gem.ingest import IngestOperator
from bench.agent_memory.gem.types import (
    IngestResult,
    InteractionEvent,
    PhaseCost,
    RevisionResult,
    StateDelta,
)
from bench.agent_memory.semantic.cache import input_hash
from bench.agent_memory.semantic.ops import CacheMiss
from bench.agent_memory.semantic.pending import TRIGGERS, PendingQueue

EAGER = "eager"
DEFERRED = "deferred"
MODES = (EAGER, DEFERRED)

#: The gem_pending derivation name for a whole deferred ingest of one event.
DERIVATION = "llm_ingest"

#: Advisory-lock namespace serialising snapshot_seq allocation per scope. It is
#: NOT the writer lock: enqueue must never wait behind a writer holding an LLM.
_SEQ_LOCK_PREFIX = "gem_pending_seq:"


@dataclass(frozen=True)
class IngestPlan:
    """pi: when the derivation runs. ``trigger`` applies to deferred plans only."""

    mode: str = EAGER
    trigger: str = "batch"
    derivation: str = DERIVATION
    #: materialize() plans outside the writer txn (R11). False = legacy in-lock.
    plan_outside_lock: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}, got {self.trigger!r}")


@dataclass(frozen=True)
class MaterializeReceipt:
    scope_id: str
    by: str
    resolved: tuple[int, ...] = ()
    failed: tuple[int, ...] = ()
    results: tuple[IngestResult, ...] = ()
    replanned: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    seconds: float = 0.0
    maintenance: tuple[RevisionResult, ...] = ()

    @property
    def resolved_count(self) -> int:
        return len(self.resolved)


class _AlreadyResolved(RuntimeError):
    """Another resolver committed this item first; skip it."""


def event_payload(event: InteractionEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["metadata"] = dict(event.metadata or {})
    return payload


def payload_event(payload: Mapping[str, Any]) -> InteractionEvent:
    return InteractionEvent(
        scope_id=str(payload["scope_id"]),
        external_id=str(payload["external_id"]),
        content=str(payload["content"]),
        session_id=payload.get("session_id"),
        role=payload.get("role"),
        kind=str(payload.get("kind") or "turn"),
        event_time=payload.get("event_time"),
        event_order=int(payload.get("event_order") or 0),
        metadata=dict(payload.get("metadata") or {}),
    )


def target_ref_for(event: InteractionEvent) -> str:
    """The event's external id, or a stable hash of its content when absent."""
    if event.external_id:
        return str(event.external_id)
    return "sha256:" + input_hash(event_payload(event))


def _single_scope(events: Sequence[InteractionEvent], scope_id: str | None) -> str:
    if scope_id is not None:
        return scope_id
    scopes = {event.scope_id for event in events}
    if len(scopes) != 1:
        raise ValueError(
            f"ingest needs exactly one scope; events span {sorted(scopes)}"
        )
    return scopes.pop()


class PlannedIngest:
    """An ``IngestPlan`` bound to one ingest operator and one strategy.

    ``target`` is a ``TriDBGovernedMemory`` (its ingest/revise operators and
    rejection threshold are used) or a bare :class:`IngestOperator`.
    ``cache_client`` (optional) is the strategy's
    :class:`~bench.agent_memory.gem.semops.CachedCompletionClient`; its hit /
    miss counters feed the receipts.
    """

    def __init__(
        self,
        target: Any,
        strategy: Any,
        plan: IngestPlan | None = None,
        *,
        max_rejection_rate: float | None = None,
        cache_client: Any | None = None,
        maintenance: bool = False,
        reviser: Any | None = None,
    ) -> None:
        if isinstance(target, IngestOperator):
            self.operator = target
        else:
            self.operator = target._ingest
            reviser = (
                reviser if reviser is not None else getattr(target, "_revise", None)
            )
            if max_rejection_rate is None:
                max_rejection_rate = getattr(target, "max_rejection_rate", None)
        self.store = self.operator.store
        self.strategy = strategy
        self.plan = plan or IngestPlan()
        self.max_rejection_rate = max_rejection_rate
        self.cache_client = cache_client
        self.maintenance = maintenance
        self.reviser = reviser
        self.maintenance_log: list[RevisionResult] = []

    # -- helpers ---------------------------------------------------------

    def _cache_snapshot(self) -> tuple[int, int]:
        stats = getattr(self.cache_client, "stats", None)
        return (0, 0) if stats is None else (int(stats.hits), int(stats.misses))

    def _with_cache(
        self, result: IngestResult, before: tuple[int, int]
    ) -> IngestResult:
        hits, misses = self._cache_snapshot()
        return replace(
            result, cache_hits=hits - before[0], cache_misses=misses - before[1]
        )

    def _maintain(self, scope_id: str, result: IngestResult) -> RevisionResult | None:
        if not (self.maintenance and self.reviser is not None and result.committed):
            return None
        if not result.delta.propagated_units:
            return None
        revision = self.reviser.revise(scope_id, kinds=("propagate",))
        self.maintenance_log.append(revision)
        return revision

    def pending_open(self, scope_id: str) -> int:
        return PendingQueue(self.store.conn).count_open(scope_id)

    # -- Ingest_pi -------------------------------------------------------

    def ingest(
        self, events: Sequence[InteractionEvent], *, scope_id: str | None = None
    ) -> IngestResult:
        scope = _single_scope(events, scope_id)
        if self.plan.mode == DEFERRED:
            return self._enqueue(events, scope)
        # D2: an eager derivation must see the state the FIFO prefix produces.
        if self.pending_open(scope):
            self.materialize(scope, by="batch")
        before = self._cache_snapshot()
        result = self.operator.ingest(
            events,
            strategy=self.strategy,
            scope_id=scope,
            max_rejection_rate=self.max_rejection_rate,
        )
        self._maintain(scope, result)
        return self._with_cache(result, before)

    def _enqueue(self, events: Sequence[InteractionEvent], scope: str) -> IngestResult:
        """One short transaction: allocate snapshot_seq, enqueue, commit. No LLM."""
        started = time.perf_counter()
        conn = self.store.conn
        queue = PendingQueue(conn)
        enqueued: list[int] = []
        duplicates: list[dict[str, Any]] = []
        statements = 0
        with conn.transaction():
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (_SEQ_LOCK_PREFIX + scope,),
            )
            row = conn.execute(
                "SELECT COALESCE(max(snapshot_seq), 0) FROM gem_pending WHERE scope_id = %s",
                (scope,),
            ).fetchone()
            statements += 2
            seq = 0 if row is None or row[0] is None else int(row[0])
            for index, event in enumerate(events):
                seq += 1
                ref = target_ref_for(event)
                pending_id = queue.enqueue(
                    scope,
                    self.plan.derivation,
                    ref,
                    trigger=self.plan.trigger,
                    snapshot_seq=seq,
                    payload=event_payload(event),
                )
                statements += 1
                if pending_id is None:
                    duplicates.append(
                        {
                            "index": index,
                            "gate": "pending_duplicate",
                            "detail": f"{self.plan.derivation}:{ref} already queued or resolved",
                        }
                    )
                else:
                    enqueued.append(pending_id)
        return IngestResult(
            operator="ingest",
            committed=True,
            delta=StateDelta(),
            cost=PhaseCost(
                phase="construction",
                seconds=time.perf_counter() - started,
                db_statements=statements,
            ),
            rejected=tuple(duplicates),
            pending_enqueued=tuple(enqueued),
        )

    # -- Mat(S) ------------------------------------------------------------

    def materialize(
        self, scope_id: str, until: int | None = None, by: str = "batch"
    ) -> MaterializeReceipt:
        """Resolve open items of ``scope_id`` in FIFO ``snapshot_seq`` order.

        ``until`` bounds the prefix by ``snapshot_seq`` (inclusive). Each item
        is one ingest transition that also marks the item resolved; a failed
        derivation is marked ``failed`` and the walk continues (an aborted
        eager ingest also leaves the state unchanged and moves on). A
        :class:`CacheMiss` (offline replay) propagates without marking anything.
        """
        if by not in TRIGGERS:
            raise ValueError(f"by must be one of {TRIGGERS}, got {by!r}")
        started = time.perf_counter()
        queue = PendingQueue(self.store.conn)
        cache_before = self._cache_snapshot()
        resolved: list[int] = []
        failed: list[int] = []
        results: list[IngestResult] = []
        maintenance: list[RevisionResult] = []
        replanned = 0
        skipped: set[int] = set()
        while True:
            items = [
                item
                for item in queue.open_items(
                    scope_id,
                    derivation=self.plan.derivation,
                    limit=len(skipped) + 1,
                    lock=False,
                )
                if item.id not in skipped
            ]
            if not items:
                break
            item = items[0]
            if until is not None and item.snapshot_seq > until:
                break
            outcome, result, used = self._resolve_one(scope_id, item, by)
            if outcome == "skipped":
                skipped.add(item.id)
                continue
            results.append(result)
            if outcome == "resolved":
                resolved.append(item.id)
                if used is False:
                    replanned += 1
                revision = self._maintain(scope_id, result)
                if revision is not None:
                    maintenance.append(revision)
            else:
                failed.append(item.id)
        hits, misses = self._cache_snapshot()
        return MaterializeReceipt(
            scope_id=scope_id,
            by=by,
            resolved=tuple(resolved),
            failed=tuple(failed),
            results=tuple(results),
            replanned=replanned,
            cache_hits=hits - cache_before[0],
            cache_misses=misses - cache_before[1],
            seconds=time.perf_counter() - started,
            maintenance=tuple(maintenance),
        )

    def _resolve_one(
        self, scope_id: str, item: Any, by: str
    ) -> tuple[str, IngestResult | None, bool | None]:
        event = payload_event(item.payload)
        before = self._cache_snapshot()
        preplan = None
        if self.plan.plan_outside_lock:
            try:
                preplan = self.operator.plan_outside(
                    [event], strategy=self.strategy, scope_id=scope_id
                )
            except CacheMiss:
                raise
            except Exception as exc:  # noqa: BLE001 — recorded as a failed item
                reason = f"{type(exc).__name__}: {exc}"
                with self.store.conn.transaction():
                    PendingQueue(self.store.conn).fail(item.id, reason)
                failed = IngestResult(
                    operator="ingest",
                    committed=False,
                    delta=StateDelta(),
                    cost=PhaseCost(phase="construction", seconds=0.0),
                    aborted_reason=reason,
                )
                return "failed", failed, None

        def mark_resolved(tx: Any) -> None:
            row = tx.execute(
                "SELECT status FROM gem_pending WHERE id = %s FOR UPDATE", (item.id,)
            ).fetchone()
            if row is None or row[0] != "pending":
                raise _AlreadyResolved(f"gem_pending {item.id} is no longer pending")
            PendingQueue(tx).resolve([item.id], by=by)

        result = self.operator.ingest(
            [event],
            strategy=self.strategy,
            scope_id=scope_id,
            max_rejection_rate=self.max_rejection_rate,
            preplan=preplan,
            in_txn=mark_resolved,
        )
        result = self._with_cache(result, before)
        used = None if preplan is None else preplan.used
        if result.committed:
            return "resolved", result, used
        reason = result.aborted_reason or "aborted"
        if reason.startswith(_AlreadyResolved.__name__):
            return "skipped", None, None
        if reason.startswith(CacheMiss.__name__):
            raise CacheMiss(reason)
        with self.store.conn.transaction():
            PendingQueue(self.store.conn).fail(item.id, reason)
        return "failed", result, used

    # -- hooks -------------------------------------------------------------

    def resolver(self) -> Callable[[str], int]:
        """The strong-read hook for ``RetrieveOperator(resolver=...)``."""

        def resolve(scope_id: str) -> int:
            return self.materialize(scope_id, by="on_read").resolved_count

        return resolve


__all__ = [
    "DEFERRED",
    "DERIVATION",
    "EAGER",
    "IngestPlan",
    "MaterializeReceipt",
    "PlannedIngest",
    "event_payload",
    "payload_event",
    "target_ref_for",
]
