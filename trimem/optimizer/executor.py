"""Single selected-query Open/Next/Close with a fail-closed EXPLAIN gate.

No EXPLAIN ANALYZE is run. A named PostgreSQL cursor fetches one result at a
time; closing it stops the upstream native iterator. Actual semantic usage
must come from a request-correlated collector, never global cache deltas.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Protocol, Self

from .compiler import NativePlanSpec, expand_node_count


class CapabilityError(ValueError):
    """The planned query cannot satisfy the supported streaming contract."""


class UsageCollector(Protocol):
    """Adapter to worker/provider telemetry correlated to this exact execution.

    ``finish`` includes failed/retried requests and must raise when attribution
    or token usage is incomplete. A cache-only query may return zero events.
    """

    def start(self, execution_id: str, plan: NativePlanSpec) -> Any: ...

    def finish(self, handle: Any) -> Sequence[Mapping[str, Any]]: ...


def _nodes(root: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield root
    for child in root.get("Plans", []):
        yield from _nodes(child)


def check_streaming_plan(explain: Any, plan: NativePlanSpec) -> dict[str, Any]:
    """Reject blocking/unknown operators, missing native nodes and lost ANN.

    This checks a concrete non-ANALYZE plan, not live executor correctness.
    Engine integration remains a separate target-specific validation gate.
    """
    if isinstance(explain, str):
        explain = json.loads(explain)
    root = (
        explain[0]["Plan"]
        if isinstance(explain, list)
        else explain.get("Plan", explain)
    )
    # Append is a streaming UNION ALL: a union hop's per-type Expand nodes, or
    # frontier="all" (keep hop-i endpoint, then descend; same Expand count).
    allowed = {
        "Append",
        "Limit",
        "Result",
        "Subquery Scan",
        "Nested Loop",
        "Index Scan",
        "Index Only Scan",
        "Seq Scan",
    }
    expected = Counter(plan.expected_custom_nodes)
    actual: Counter[str] = Counter()
    found_ann = False
    for node in _nodes(root):
        kind = node.get("Node Type")
        if kind == "Custom Scan":
            provider = node.get("Custom Plan Provider", "")
            if provider not in {"TriMEM Expand", "TriMEM SemFilter", "TriMEM SemScore"}:
                raise CapabilityError(f"unverified custom provider: {provider!r}")
            actual[provider] += 1
            if provider in {"TriMEM SemFilter", "TriMEM SemScore"}:
                batch = str(node.get("Batch Size", "")).split()
                if not batch or batch[0] != "1":
                    raise CapabilityError("semantic nodes must use batch size 1")
        elif kind not in allowed:
            raise CapabilityError(f"blocking or unverified plan node: {kind!r}")
        if node.get("Parallel Aware", False):
            raise CapabilityError("parallel retrieval is outside the v1 capability")
        if kind in {"Index Scan", "Index Only Scan"} and "<=>" in str(
            node.get("Order By", "")
        ):
            found_ann = True
        if kind == "Nested Loop":
            children = node.get("Plans", [])
            if len(children) != 2:
                raise CapabilityError("unverified nested-loop child layout")
            outer, inner = children
            if any(
                n.get("Custom Plan Provider") == "TriMEM Expand" for n in _nodes(inner)
            ):
                outer_is_frontier = any(
                    n.get("Custom Plan Provider") == "TriMEM Expand"
                    or "<=>" in str(n.get("Order By", ""))
                    for n in _nodes(outer)
                )
                if not outer_is_frontier:
                    raise CapabilityError(
                        "graph moved inside a non-frontier rescan loop"
                    )
    if actual != expected:
        raise CapabilityError(
            f"native operator mismatch: wanted {expected}, got {actual}"
        )
    if plan.sql is not None and not found_ann:
        raise CapabilityError("vector seed order is not supplied by an ordered index")

    # Native SQL barriers should keep final scoring above all graph nodes, and
    # row filtering above graph/telemetry. Reject plans that lost those barriers.
    expand_nodes = expand_node_count(plan.action)
    for node in _nodes(root):
        provider = node.get("Custom Plan Provider")
        descendants = list(_nodes(node))
        count_expand = sum(
            n.get("Custom Plan Provider") == "TriMEM Expand" for n in descendants
        )
        if provider == "TriMEM SemScore" and count_expand != expand_nodes:
            raise CapabilityError("StreamScore moved below graph expansion")
        if (
            provider == "TriMEM SemScore"
            and plan.action.filter_target
            and not any(
                n.get("Custom Plan Provider") == "TriMEM SemFilter" for n in descendants
            )
        ):
            raise CapabilityError("StreamScore moved before SemFilter")
        if provider == "TriMEM SemFilter":
            wanted = expand_nodes if plan.action.filter_target == "row" else 0
            if count_expand != wanted:
                raise CapabilityError(
                    "SemFilter predicate target/order differs from the selected plan"
                )
            if plan.action.filter_target == "seed" and not any(
                "<=>" in str(n.get("Order By", "")) for n in descendants
            ):
                raise CapabilityError("seed filter is not downstream of vector seeds")
    return dict(root)


@dataclass
class ExecutionReceipt:
    execution_id: str
    action_hash: str
    sql_hash: str
    plan_hash: str | None = None
    rows_fetched: int = 0
    score_rejected: int = 0
    rows_emitted: int = 0
    actual_executions: int = 0
    emitted_edge_upper_bound: int = 0
    edge_visits_measured: int | None = None
    stop_reason: str | None = None
    closed: bool = False
    usage_complete: bool = False
    usage_events: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0  # telemetry only; never an objective/constraint
    # ordering="gain": rows read into the bounded pool / distinct attempt ids.
    ranked_by: str = "stream"
    pool_rows: int | None = None
    pool_distinct: int | None = None


class NativeExecutor:
    def __init__(self, connection: Any, usage_collector: UsageCollector | None = None):
        self.connection = connection
        self.usage_collector = usage_collector

    def open_selected(self, plan: NativePlanSpec) -> SelectedCursor:
        return SelectedCursor(self.connection, plan, self.usage_collector)


class SelectedCursor(Iterator[dict[str, Any]]):
    def __init__(
        self, connection: Any, plan: NativePlanSpec, collector: UsageCollector | None
    ):
        self.connection, self.plan, self.collector = connection, plan, collector
        self.receipt = ExecutionReceipt(
            uuid.uuid4().hex,
            plan.action.action_hash,
            plan.sql_hash,
            emitted_edge_upper_bound=plan.emitted_edge_upper_bound,
        )
        self._stack = ExitStack()
        self._cursor: Any = None
        self._started: float | None = None
        self._usage_handle: Any = None
        self._usage_started = False

    def __enter__(self) -> Self:
        if self._started is not None:
            raise RuntimeError("a selected cursor can only be opened once")
        self._started = time.monotonic()
        if self.plan.sql is None:
            self.receipt.usage_complete = True
            return self
        if (
            self.plan.action.filter_target or self.plan.action.uses_score
        ) and self.collector is None:
            self.close("missing_usage_collector")
            raise ValueError(
                "semantic execution requires request-correlated UsageCollector"
            )
        try:
            # Retrieval is read-only. Roll back this transaction/savepoint after
            # Close to restore all LOCAL settings, including in an outer txn.
            self._stack.enter_context(self.connection.transaction(force_rollback=True))
            with self.connection.cursor() as preflight:
                version = getattr(
                    getattr(self.connection, "info", None), "server_version", 160000
                )
                for name, value in self.plan.settings:
                    if name == "enable_memoize" and version < 140000:
                        continue
                    preflight.execute("SELECT set_config(%s, %s, true)", (name, value))
                preflight.execute(
                    "EXPLAIN (FORMAT JSON) " + self.plan.sql, self.plan.params
                )
                root = check_streaming_plan(preflight.fetchone()[0], self.plan)
                encoded = json.dumps(root, sort_keys=True).encode()
                self.receipt.plan_hash = hashlib.sha256(encoded).hexdigest()
            if self.collector is not None:
                self._usage_handle = self.collector.start(
                    self.receipt.execution_id, self.plan
                )
                self._usage_started = True
            else:
                self.receipt.usage_complete = True
            self._cursor = self._stack.enter_context(
                self.connection.cursor(name="trimem_joint_" + self.receipt.execution_id)
            )
            self._cursor.itersize = 1
            self._cursor.arraysize = 1
            self.receipt.actual_executions = 1
            self._cursor.execute(self.plan.sql, self.plan.params)
            return self
        except BaseException:
            self.close("open_error")
            raise

    def __iter__(self) -> SelectedCursor:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._started is None:
            raise RuntimeError("use 'with executor.open_selected(plan)' before Next")
        if self.receipt.closed:
            raise StopIteration
        if self.plan.sql is None:
            self.close("no_memory")
            raise StopIteration
        try:
            while self.receipt.rows_fetched < self.plan.action.candidate_visit_cap:
                row = self._cursor.fetchone()
                if row is None:
                    self.close("source_exhausted")
                    raise StopIteration
                self.receipt.rows_fetched += 1
                if isinstance(row, Mapping):
                    record = dict(row)
                else:
                    names = [column.name for column in self._cursor.description]
                    record = dict(zip(names, row, strict=True))
                if self.plan.action.uses_score:
                    score = record.get("sem_score")
                    if (
                        score is None
                        or not math.isfinite(float(score))
                        or not 0 <= float(score) <= 1
                    ):
                        self.receipt.score_rejected += 1
                        continue
                    if float(score) < self.plan.action.score_admit_min:
                        self.receipt.score_rejected += 1
                        continue
                self.receipt.rows_emitted += 1
                return record
            self.close("candidate_visit_cap")
            raise StopIteration
        except StopIteration:
            raise
        except BaseException:
            self.close("fetch_error")
            raise

    def close(self, reason: str = "caller_closed") -> None:
        if self.receipt.closed:
            return
        self.receipt.stop_reason = self.receipt.stop_reason or reason
        self.receipt.closed = True
        try:
            self._stack.close()
        finally:
            try:
                if self._usage_started:
                    events = self.collector.finish(self._usage_handle)
                    self.receipt.usage_events = [dict(event) for event in events]
                    self.receipt.usage_complete = True
            except BaseException as exc:
                # A collector may fail after some physical requests completed.
                # Preserve those costs; the adapter also marks attribution incomplete.
                self.receipt.usage_events = [
                    dict(event) for event in getattr(exc, "usage_events", ())
                ]
                self.receipt.usage_complete = (
                    getattr(exc, "usage_complete", False) is True
                )
                self.receipt.stop_reason = getattr(
                    exc, "stop_reason", "usage_collection_error"
                )
                raise
            finally:
                if self._started is not None:
                    self.receipt.elapsed_seconds = time.monotonic() - self._started

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close("consumer_error" if exc_type else "caller_closed")
