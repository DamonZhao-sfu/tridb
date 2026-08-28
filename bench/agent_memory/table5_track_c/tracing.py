"""Lossless per-request stage tracing for the Track C load harness.

The tracer is deliberately adapter-local and opt-in.  It records client-side
boundaries without pretending that a compound native API can always be split
into independent vector/graph/LLM timings.  Such calls use ``fusion`` or
``framework_other`` and carry an explicit ``includes`` attribute.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .scheduler import percentile

STAGE_CATEGORIES = frozenset(
    {
        "request",
        "embedding",
        "vector",
        "graph",
        "relational",
        "llm",
        "persistence",
        "fusion",
        "framework_other",
    }
)

OBSERVABLE_CALL_KINDS = (
    "http_client",
    "database_client",
    "opaque_native_api",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TraceContext:
    recorder: "SpanRecorder"
    trace_id: str
    root_span_id: str
    system: str
    build_id: str
    phase: str
    request_index: int
    parent_span_id: str


_CURRENT: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "track_c_trace_context", default=None
)


class SpanRecorder:
    """Thread-safe JSONL span recorder shared by warmup and formal runners."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._observable_calls: dict[str, Counter[str]] = {}

    @staticmethod
    def request_ids(
        system: str, build_id: str, phase: str, request_index: int
    ) -> tuple[str, str]:
        seed = f"{system}\0{build_id}\0{phase}\0{request_index}"
        trace_id = hashlib.sha256(seed.encode()).hexdigest()[:32]
        root_span_id = hashlib.sha256(f"root\0{seed}".encode()).hexdigest()[:16]
        return trace_id, root_span_id

    def write(self, record: Mapping[str, Any]) -> None:
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            attributes = record.get("attributes") or {}
            kind = attributes.get("observable_call_kind")
            count = attributes.get("observable_call_count", 1)
            if (
                record.get("category") != "request"
                and kind in OBSERVABLE_CALL_KINDS
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count > 0
            ):
                trace_id = str(record.get("trace_id") or "")
                self._observable_calls.setdefault(trace_id, Counter())[kind] += count
            self._handle.write(encoded + "\n")
            self._handle.flush()

    def observable_call_summary(
        self, trace_id: str, *, timed_out: bool
    ) -> dict[str, Any]:
        """Return only client-visible call boundaries; never invent internals."""
        with self._lock:
            observed = self._observable_calls.get(trace_id, Counter()).copy()
        counts = {kind: int(observed.get(kind, 0)) for kind in OBSERVABLE_CALL_KINDS}
        return {
            **counts,
            "total": sum(counts.values()),
            "source": "stage_span_attributes",
            "scope": (
                "partial_client_boundaries_before_timeout"
                if timed_out
                else "complete_client_boundaries"
            ),
            "exact_internal_http_round_trips": None,
            "exact_internal_database_round_trips": None,
            "internal_round_trip_policy": (
                "unavailable inside compound native APIs; never inferred from one "
                "native API call"
            ),
        }

    @contextmanager
    def bind_request(
        self,
        *,
        trace_id: str,
        root_span_id: str,
        system: str,
        build_id: str,
        phase: str,
        request_index: int,
    ) -> Iterator[None]:
        context = TraceContext(
            recorder=self,
            trace_id=trace_id,
            root_span_id=root_span_id,
            system=system,
            build_id=build_id,
            phase=phase,
            request_index=request_index,
            parent_span_id=root_span_id,
        )
        token = _CURRENT.set(context)
        try:
            yield
        finally:
            _CURRENT.reset(token)

    def record_request(
        self,
        *,
        trace_id: str,
        root_span_id: str,
        system: str,
        build_id: str,
        phase: str,
        request_index: int,
        started_ns: int,
        completed_ns: int,
        success: bool,
        timed_out: bool,
        error: str | None,
    ) -> None:
        self.write(
            {
                "schema_version": "table5_track_c_span_v0.1.0",
                "trace_id": trace_id,
                "span_id": root_span_id,
                "parent_span_id": None,
                "system": system,
                "build_id": build_id,
                "phase": phase,
                "request_index": request_index,
                "category": "request",
                "operation": "harness.service_window",
                "backend": "track_c_harness",
                "started_at_ns": started_ns,
                "completed_at_ns": completed_ns,
                "started_at": None,
                "completed_at": _utc_now(),
                "duration_ms": (completed_ns - started_ns) / 1_000_000,
                "success": success,
                "timeout": timed_out,
                "error": error,
                "attributes": {"clock": "perf_counter_ns"},
            }
        )

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "SpanRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@contextmanager
def stage_span(
    category: str,
    operation: str,
    *,
    backend: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Record a nested stage when a request trace is active; otherwise no-op."""
    if category not in STAGE_CATEGORIES - {"request"}:
        raise ValueError(f"unknown Track C stage category: {category}")
    context = _CURRENT.get()
    if context is None:
        yield
        return

    span_id = uuid.uuid4().hex[:16]
    started_ns = time.perf_counter_ns()
    started_at = _utc_now()
    child = TraceContext(
        recorder=context.recorder,
        trace_id=context.trace_id,
        root_span_id=context.root_span_id,
        system=context.system,
        build_id=context.build_id,
        phase=context.phase,
        request_index=context.request_index,
        parent_span_id=span_id,
    )
    token = _CURRENT.set(child)
    success = False
    error: str | None = None
    try:
        yield
        success = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        completed_ns = time.perf_counter_ns()
        _CURRENT.reset(token)
        context.recorder.write(
            {
                "schema_version": "table5_track_c_span_v0.1.0",
                "trace_id": context.trace_id,
                "span_id": span_id,
                "parent_span_id": context.parent_span_id,
                "system": context.system,
                "build_id": context.build_id,
                "phase": context.phase,
                "request_index": context.request_index,
                "category": category,
                "operation": operation,
                "backend": backend,
                "started_at_ns": started_ns,
                "completed_at_ns": completed_ns,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "duration_ms": (completed_ns - started_ns) / 1_000_000,
                "success": success,
                "timeout": False,
                "error": error,
                "attributes": dict(attributes or {}),
            }
        )


def _union_ms(intervals: Sequence[tuple[int, int]]) -> float:
    if not intervals:
        return 0.0
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged) / 1_000_000


def summarize_spans(
    path: str | Path, *, phase_prefix: str = "formal_"
) -> dict[str, Any]:
    """Summarize formal stage work while preserving overlap semantics."""
    path = Path(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    records = [
        row for row in records if str(row.get("phase", "")).startswith(phase_prefix)
    ]
    roots = [row for row in records if row.get("category") == "request"]
    children = [row for row in records if row.get("category") != "request"]
    root_by_trace = {str(row["trace_id"]): row for row in roots}
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in children:
        by_category.setdefault(str(row["category"]), []).append(row)

    stages: dict[str, Any] = {}
    for category in sorted(by_category):
        rows = by_category[category]
        raw_durations = [float(row["duration_ms"]) for row in rows]
        by_trace: dict[str, list[tuple[int, int]]] = {}
        client_visible_stage_work_ms = 0.0
        for row in rows:
            trace_id = str(row["trace_id"])
            root = root_by_trace.get(trace_id)
            if root is None:
                continue
            visible_start = max(int(row["started_at_ns"]), int(root["started_at_ns"]))
            visible_end = min(int(row["completed_at_ns"]), int(root["completed_at_ns"]))
            if visible_end <= visible_start:
                continue
            by_trace.setdefault(trace_id, []).append((visible_start, visible_end))
            client_visible_stage_work_ms += (visible_end - visible_start) / 1_000_000
        unions = [_union_ms(value) for value in by_trace.values()]
        raw_stage_work_ms = sum(raw_durations)
        stages[category] = {
            "span_count": len(rows),
            "request_count": len(by_trace),
            "raw_request_count": len({str(row["trace_id"]) for row in rows}),
            "successful_spans": sum(row.get("success") is True for row in rows),
            # Compatibility field now has an explicit client-visible meaning.
            "stage_work_ms": client_visible_stage_work_ms,
            "client_visible_stage_work_ms": client_visible_stage_work_ms,
            "raw_stage_work_ms": raw_stage_work_ms,
            "post_request_work_ms": max(
                0.0, raw_stage_work_ms - client_visible_stage_work_ms
            ),
            "per_request_union_ms": {
                "p50": percentile(unions, 50),
                "p90": percentile(unions, 90),
                "p99": percentile(unions, 99),
                "mean": sum(unions) / len(unions) if unions else None,
            },
        }

    child_intervals: dict[str, list[tuple[int, int]]] = {}
    for row in children:
        trace_id = str(row["trace_id"])
        root = root_by_trace.get(trace_id)
        if root is None:
            continue
        visible_start = max(int(row["started_at_ns"]), int(root["started_at_ns"]))
        visible_end = min(int(row["completed_at_ns"]), int(root["completed_at_ns"]))
        if visible_end > visible_start:
            child_intervals.setdefault(trace_id, []).append(
                (visible_start, visible_end)
            )
    unattributed = []
    for trace_id, root in root_by_trace.items():
        service_ms = float(root["duration_ms"])
        covered_ms = _union_ms(child_intervals.get(trace_id, []))
        unattributed.append(max(0.0, service_ms - covered_ms))
    return {
        "schema_version": "table5_track_c_breakdown_v0.2.0",
        "trace_file": str(path.resolve()),
        "formal_requests": len(roots),
        "formal_requests_with_stage_spans": len(child_intervals),
        "stages": stages,
        "unattributed_per_request_ms": {
            "p50": percentile(unattributed, 50),
            "p90": percentile(unattributed, 90),
            "p99": percentile(unattributed, 99),
            "mean": sum(unattributed) / len(unattributed) if unattributed else None,
        },
        "post_request_stage_work_ms": sum(
            float(stage["post_request_work_ms"]) for stage in stages.values()
        ),
        "overlap_note": (
            "stage_work_ms and per_request_union_ms are clipped to each client-visible "
            "root request window; raw_stage_work_ms retains drain work after timeout. "
            "Concurrent stage_work_ms may exceed request latency, while per-request "
            "unions merge overlap. Compound native APIs are not assigned fake substage "
            "time."
        ),
    }


def observable_call_counts_match_spans(
    requests: Sequence[Mapping[str, Any]], spans: Sequence[Mapping[str, Any]]
) -> bool:
    """Recompute client-boundary counters from raw child spans.

    A timed-out worker cannot be force-cancelled and may finish a child span
    after its client-visible request record is written.  Its recorded counters
    are therefore a prefix and must be no greater than the final span-derived
    counts.  All non-timeout requests must match exactly.
    """
    derived: dict[str, Counter[str]] = {}
    for span in spans:
        if span.get("category") == "request":
            continue
        attributes = span.get("attributes") or {}
        kind = attributes.get("observable_call_kind")
        count = attributes.get("observable_call_count", 1)
        if (
            kind not in OBSERVABLE_CALL_KINDS
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
        ):
            continue
        trace_id = str(span.get("trace_id") or "")
        derived.setdefault(trace_id, Counter())[kind] += count
    for request in requests:
        trace_id = str(request.get("trace_id") or "")
        observed = request.get("observable_call_counts")
        if not isinstance(observed, Mapping):
            return False
        expected = derived.get(trace_id, Counter())
        for kind in OBSERVABLE_CALL_KINDS:
            value = observed.get(kind)
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            if request.get("timeout") is True:
                if value > expected.get(kind, 0):
                    return False
            elif value != expected.get(kind, 0):
                return False
    return True
