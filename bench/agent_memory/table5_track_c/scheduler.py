"""No-drift open-loop request admission and lossless JSONL recording."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from .tracing import SpanRecorder

RequestCall = Callable[[Any], Mapping[str, Any] | Any]
PartialReceipt = Callable[[Any], Mapping[str, Any] | Any]
DEFAULT_MAX_IN_FLIGHT = 768
DEFAULT_ADMISSION_QPS_RELATIVE_TOLERANCE = 0.01
DEFAULT_ADMISSION_LAG_P99_MAX_MS = 100.0


class RecordedRequestFailure(RuntimeError):
    """A measured failure whose partial native receipt must remain auditable."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Linear interpolation, equivalent to numpy.percentile's default."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"value": repr(value)}


class JsonlSink:
    """Append one complete line at a time, including failures and warmups."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._handle.write(encoded + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class OpenLoopRunner:
    """Admit calls on an absolute schedule; never accumulate sleep drift."""

    def __init__(
        self,
        *,
        qps: float = 10.0,
        timeout_seconds: float = 60.0,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        if qps <= 0:
            raise ValueError("qps must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self.qps = float(qps)
        self.timeout_seconds = float(timeout_seconds)
        self.max_in_flight = int(max_in_flight)
        self._waiting = 0
        self._active = 0
        self.max_observed_queue_depth = 0
        self.max_observed_active = 0

    async def run(
        self,
        requests: Sequence[Any],
        call: RequestCall,
        *,
        sink: JsonlSink,
        phase: str,
        system: str,
        build_id: str,
        request_metadata: Callable[[Any], Mapping[str, Any]] | None = None,
        partial_receipt: PartialReceipt | None = None,
        span_recorder: SpanRecorder | None = None,
        late_sink: JsonlSink | None = None,
    ) -> list[dict[str, Any]]:
        base_ns = time.perf_counter_ns()
        semaphore = asyncio.Semaphore(self.max_in_flight)
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        # The Python default executor is capped near 32 threads.  The frozen
        # 768 capacity covers a full 10 QPS x 60-second timeout window (600
        # calls) with 28% headroom, so the harness does not throttle an
        # otherwise admissible request before its timeout.  Context shutdown
        # waits for calls whose client-visible timeout elapsed but whose thread
        # cannot be force-cancelled, preventing cross-phase interference.
        with ThreadPoolExecutor(
            max_workers=self.max_in_flight, thread_name_prefix="track-c-request"
        ) as executor:
            for index, request in enumerate(requests):
                scheduled_ns = base_ns + round(index * 1_000_000_000 / self.qps)
                delay = (scheduled_ns - time.perf_counter_ns()) / 1_000_000_000
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(
                    asyncio.create_task(
                        self._run_one(
                            request,
                            call,
                            sink=sink,
                            semaphore=semaphore,
                            phase=phase,
                            system=system,
                            build_id=build_id,
                            request_index=index,
                            scheduled_ns=scheduled_ns,
                            metadata=(
                                dict(request_metadata(request))
                                if request_metadata
                                else {}
                            ),
                            partial_receipt=partial_receipt,
                            executor=executor,
                            span_recorder=span_recorder,
                            late_sink=late_sink,
                        )
                    )
                )
            if not tasks:
                return []
            return list(await asyncio.gather(*tasks))

    async def _run_one(
        self,
        request: Any,
        call: RequestCall,
        *,
        sink: JsonlSink,
        semaphore: asyncio.Semaphore,
        phase: str,
        system: str,
        build_id: str,
        request_index: int,
        scheduled_ns: int,
        metadata: dict[str, Any],
        partial_receipt: PartialReceipt | None = None,
        executor: Executor | None = None,
        span_recorder: SpanRecorder | None = None,
        late_sink: JsonlSink | None = None,
    ) -> dict[str, Any]:
        admitted_ns = time.perf_counter_ns()
        self._waiting += 1
        queue_depth_at_admission = self._waiting
        self.max_observed_queue_depth = max(
            self.max_observed_queue_depth, self._waiting
        )
        async with semaphore:
            self._waiting -= 1
            self._active += 1
            active_at_start = self._active
            self.max_observed_active = max(self.max_observed_active, self._active)
            started_ns = time.perf_counter_ns()
            started_wall = _utc_now()
            success = False
            timed_out = False
            error: str | None = None
            payload: Any = {}
            client_completed_ns: int | None = None
            trace_id = None
            root_span_id = None
            if span_recorder is not None:
                trace_id, root_span_id = span_recorder.request_ids(
                    system, build_id, phase, request_index
                )

            def invoke() -> Any:
                if span_recorder is None:
                    return call(request)
                assert trace_id is not None and root_span_id is not None
                with span_recorder.bind_request(
                    trace_id=trace_id,
                    root_span_id=root_span_id,
                    system=system,
                    build_id=build_id,
                    phase=phase,
                    request_index=request_index,
                ):
                    return call(request)

            def read_partial_receipt() -> Any:
                if partial_receipt is None:
                    return {}
                try:
                    return partial_receipt(request)
                except Exception as exc:  # noqa: BLE001 - preserve primary failure
                    return {"partial_receipt_error": f"{type(exc).__name__}: {exc}"}

            try:
                if executor is None:
                    raise RuntimeError("request executor is required")
                concurrent_future = executor.submit(invoke)
                payload = await asyncio.wait_for(
                    asyncio.wrap_future(concurrent_future),
                    timeout=self.timeout_seconds,
                )
                success = True
            except TimeoutError:
                timed_out = True
                error = f"TimeoutError: exceeded {self.timeout_seconds:.3f}s"
                payload = read_partial_receipt()
                client_completed_ns = time.perf_counter_ns()
                if late_sink is not None:

                    def record_late_outcome(done: Any) -> None:
                        worker_completed_ns = time.perf_counter_ns()
                        late_success = False
                        late_error: str | None = None
                        late_payload: Any = {}
                        try:
                            late_payload = done.result()
                            late_success = True
                        except RecordedRequestFailure as exc:
                            late_payload = exc.receipt
                            late_error = f"{type(exc).__name__}: {exc}"
                        except Exception as exc:  # noqa: BLE001 - measured outcome
                            late_error = f"{type(exc).__name__}: {exc}"
                            late_payload = read_partial_receipt()
                        late_sink.write(
                            {
                                "schema_version": (
                                    "table5_track_c_late_outcome_v0.1.0"
                                ),
                                "system": system,
                                "build_id": build_id,
                                "phase": phase,
                                "request_index": request_index,
                                "trace_id": trace_id,
                                "root_span_id": root_span_id,
                                "client_timed_out_at_ns": client_completed_ns,
                                "worker_completed_at_ns": worker_completed_ns,
                                "post_timeout_work_ms": (
                                    worker_completed_ns - client_completed_ns
                                )
                                / 1_000_000,
                                "final_success": late_success,
                                "final_error": late_error,
                                **metadata,
                                "receipt": _jsonable(late_payload),
                            }
                        )

                    concurrent_future.add_done_callback(record_late_outcome)
            except RecordedRequestFailure as exc:
                payload = exc.receipt
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - benchmark failures are data
                error = f"{type(exc).__name__}: {exc}"
                payload = read_partial_receipt()
            completed_ns = client_completed_ns or time.perf_counter_ns()
            try:
                if span_recorder is not None:
                    assert trace_id is not None and root_span_id is not None
                    span_recorder.record_request(
                        trace_id=trace_id,
                        root_span_id=root_span_id,
                        system=system,
                        build_id=build_id,
                        phase=phase,
                        request_index=request_index,
                        started_ns=started_ns,
                        completed_ns=completed_ns,
                        success=success,
                        timed_out=timed_out,
                        error=error,
                    )
                observable_calls = (
                    span_recorder.observable_call_summary(
                        str(trace_id), timed_out=timed_out
                    )
                    if span_recorder is not None
                    else None
                )
                record = {
                    "schema_version": "table5_track_c_request_v0.3.0",
                    "system": system,
                    "build_id": build_id,
                    "phase": phase,
                    "request_index": request_index,
                    "scheduled_at_ns": scheduled_ns,
                    "admitted_at_ns": admitted_ns,
                    "started_at_ns": started_ns,
                    "completed_at_ns": completed_ns,
                    "started_at": started_wall,
                    "completed_at": _utc_now(),
                    "admission_lag_ms": (admitted_ns - scheduled_ns) / 1_000_000,
                    "queue_latency_ms": (started_ns - scheduled_ns) / 1_000_000,
                    "service_latency_ms": (completed_ns - started_ns) / 1_000_000,
                    "user_visible_latency_ms": (completed_ns - scheduled_ns)
                    / 1_000_000,
                    "queue_depth_at_admission": queue_depth_at_admission,
                    "active_at_start": active_at_start,
                    "success": success,
                    "timeout": timed_out,
                    "error": error,
                    "harness_retries": 0,
                    "system_internal_retries": None,
                    "system_internal_retry_observability": (
                        "native counter unavailable; any internal retry remains "
                        "inside service latency"
                    ),
                    "observable_call_counts": observable_calls,
                    **({"trace_id": trace_id} if trace_id is not None else {}),
                    **metadata,
                    "receipt": _jsonable(payload),
                }
                sink.write(record)
                return record
            finally:
                self._active -= 1


async def run_sequential_warmup(
    requests: Sequence[Any],
    call: RequestCall,
    *,
    sink: JsonlSink,
    phase: str,
    system: str,
    build_id: str,
    timeout_seconds: float = 60.0,
    request_metadata: Callable[[Any], Mapping[str, Any]] | None = None,
    partial_receipt: PartialReceipt | None = None,
    span_recorder: SpanRecorder | None = None,
    late_sink: JsonlSink | None = None,
) -> list[dict[str, Any]]:
    """Warmups are serialized, including drain work after a client timeout."""
    records = []
    runner = OpenLoopRunner(qps=1.0, timeout_seconds=timeout_seconds)
    for index, request in enumerate(requests):
        # A timeout fixes the client-visible result but does not stop a Python
        # worker that is already inside a native call.  Close one executor per
        # warmup so that worker drains before the next warmup starts; otherwise
        # later warmups would queue behind hidden post-timeout work and the
        # purportedly sequential gate would no longer be sequential.
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="track-c-warmup"
        ) as executor:
            now = time.perf_counter_ns()
            records.append(
                await runner._run_one(
                    request,
                    call,
                    sink=sink,
                    semaphore=asyncio.Semaphore(1),
                    phase=phase,
                    system=system,
                    build_id=build_id,
                    request_index=index,
                    scheduled_ns=now,
                    metadata=(
                        dict(request_metadata(request)) if request_metadata else {}
                    ),
                    partial_receipt=partial_receipt,
                    executor=executor,
                    span_recorder=span_recorder,
                    late_sink=late_sink,
                )
            )
    return records
