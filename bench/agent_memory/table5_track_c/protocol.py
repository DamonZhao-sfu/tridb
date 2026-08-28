"""Executable Track C Search/Add protocol shared by every native adapter."""

from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .dataset import (
    EXPECTED_FORMAL_QUESTION_COUNT,
    EventItem,
    LoCoMoCorpus,
    QueryItem,
    WARMUP_SAMPLE_ID,
    representative_warmup_indices,
)
from .scheduler import (
    DEFAULT_ADMISSION_LAG_P99_MAX_MS,
    DEFAULT_ADMISSION_QPS_RELATIVE_TOLERANCE,
    DEFAULT_MAX_IN_FLIGHT,
    JsonlSink,
    OpenLoopRunner,
    RecordedRequestFailure,
    run_sequential_warmup,
)
from .stats import summarize
from .tracing import SpanRecorder, summarize_spans


class Adapter(Protocol):
    name: str

    def init_schema(self) -> dict[str, Any]: ...

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]: ...

    def finalize_build(self) -> dict[str, Any]: ...

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]: ...

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]: ...

    def stats(self) -> dict[str, Any]: ...

    def snapshot_fingerprint(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    """Atomically replace machine-readable receipts after every phase."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _late_outcome_summary(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phases = Counter(str(record.get("phase")) for record in records)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": "table5_track_c_late_outcome_summary_v0.1.0",
        "path": str(path.resolve()),
        "sha256": digest,
        "records": len(records),
        "by_phase": dict(sorted(phases.items())),
        "final_success": sum(record.get("final_success") is True for record in records),
        "final_failed": sum(record.get("final_success") is False for record in records),
        "post_timeout_work_ms": {
            "total": sum(float(record["post_timeout_work_ms"]) for record in records),
            "max": max(
                (float(record["post_timeout_work_ms"]) for record in records),
                default=None,
            ),
        },
    }


def benchmark_code_sha256() -> str:
    """Hash every in-repo Python module reachable by the memory benchmark."""
    repository = Path(__file__).resolve().parents[3]
    files = sorted(repository.glob("bench/agent_memory/**/*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_empty_run_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"run directory is not empty (refusing duplicate append): {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def _gate_search_warmup(records: Sequence[dict[str, Any]]) -> None:
    """Abort before formal Search on an all-infrastructure-error warmup.

    An all-timeout warmup remains a real overload observation. An all-failed
    warmup with non-timeout exceptions instead indicates broken wiring and
    must not be presented as a completed latency point.
    """
    if not records or any(bool(record.get("success")) for record in records):
        return
    if any(bool(record.get("timeout")) for record in records):
        return
    errors = Counter(str(record.get("error") or "unknown") for record in records)
    rendered = "; ".join(f"{error} ({count})" for error, count in errors.most_common(3))
    raise RuntimeError(
        f"all Search warmups failed with non-timeout infrastructure errors: {rendered}"
    )


class HostResourceSampler(AbstractContextManager["HostResourceSampler"]):
    """Sample host/GPU totals and runner RSS without claiming attribution."""

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []

    @staticmethod
    def _sample() -> dict[str, Any]:
        memory: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.strip().split()[0]) * 1024
        process_rss = None
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                process_rss = int(line.split()[1]) * 1024
                break
        gpu_used: list[int] | None = None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            gpu_used = [
                int(line.strip()) * 1024 * 1024
                for line in result.stdout.splitlines()
                if line.strip()
            ]
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        total = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        return {
            "at": _now(),
            "runner_process_rss_bytes": process_rss,
            "host_memory_used_bytes": (
                total - available
                if total is not None and available is not None
                else None
            ),
            "gpu_memory_used_bytes": gpu_used,
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.samples.append(self._sample())
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "HostResourceSampler":
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
        self.samples.append(self._sample())

    def summary(self) -> dict[str, Any]:
        def maximum(key: str) -> int | None:
            values = [
                sample[key] for sample in self.samples if sample.get(key) is not None
            ]
            return max(values) if values else None

        gpu_count = max(
            (len(sample.get("gpu_memory_used_bytes") or []) for sample in self.samples),
            default=0,
        )
        gpu_peaks = [
            max(
                (
                    sample["gpu_memory_used_bytes"][index]
                    for sample in self.samples
                    if sample.get("gpu_memory_used_bytes")
                    and index < len(sample["gpu_memory_used_bytes"])
                ),
                default=None,
            )
            for index in range(gpu_count)
        ]
        return {
            "sample_count": len(self.samples),
            "peak_runner_process_rss_bytes": maximum("runner_process_rss_bytes"),
            "peak_host_memory_used_bytes": maximum("host_memory_used_bytes"),
            "peak_gpu_memory_used_bytes": gpu_peaks,
            "scope_note": (
                "host and GPU totals include the shared model servers and are not "
                "attributable to the adapter alone"
            ),
        }


def _query_metadata(item: QueryItem) -> dict[str, Any]:
    return {
        "sample_id": item.sample_id,
        "question_id": item.question_id,
        "question": item.question,
        "answer": item.answer,
        "category": item.category,
        "evidence_ids": list(item.evidence_ids),
    }


def _retrieval_query(item: QueryItem) -> QueryItem:
    """Remove evaluation-only labels before a native adapter sees a query."""
    return replace(item, answer=None, category=None, evidence_ids=())


def _event_metadata(item: EventItem) -> dict[str, Any]:
    return {
        "sample_id": item.sample_id,
        "event_id": item.event_id,
        "session_id": item.session_id,
        "event_ordinal": item.ordinal,
    }


def _warmup_queries(corpus: LoCoMoCorpus) -> list[QueryItem]:
    result = list(corpus.queries_by_sample[WARMUP_SAMPLE_ID])
    for sample_id in corpus.formal_sample_ids:
        queries = corpus.queries_by_sample[sample_id]
        result.extend(
            queries[index] for index in representative_warmup_indices(queries)
        )
    return result


def evidence_quality(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Evidence-only gate; answer/judge quality is a separate offline phase."""
    evidenced = 0
    recall_sum = 0.0
    any_hit = 0
    result_counts: list[int] = []
    for record in records:
        evidence = set(str(value) for value in record.get("evidence_ids") or [])
        if evidence:
            evidenced += 1
        if record.get("success") is not True:
            # Retrieval failures remain in the quality denominator and
            # contribute zero overlap.  They must never improve Recall@20.
            continue
        receipt = record.get("receipt") or {}
        hits = [str(value) for value in receipt.get("hit_ids") or []]
        result_counts.append(int(receipt.get("result_count") or len(hits)))
        if not evidence:
            continue
        overlap = evidence.intersection(hits[:20])
        recall_sum += len(overlap) / len(evidence)
        any_hit += int(bool(overlap))
    return {
        "successful_queries": sum(record.get("success") is True for record in records),
        "evidenced_queries": evidenced,
        "evidence_recall_at_20": recall_sum / evidenced if evidenced else None,
        "evidence_any_hit_at_20": any_hit / evidenced if evidenced else None,
        "mean_result_count": sum(result_counts) / len(result_counts)
        if result_counts
        else None,
        "answer_judge_status": "pending_offline_quality_phase",
    }


def search_receipt_contract(records: Sequence[dict[str, Any]]) -> bool:
    """Validate comparable retrieval payloads for every successful Search."""
    for record in records:
        if record.get("success") is not True:
            continue
        observed = record.get("receipt")
        if not isinstance(observed, dict):
            return False
        result_count = observed.get("result_count")
        hit_ids = observed.get("hit_ids")
        contexts = observed.get("contexts")
        if (
            not isinstance(result_count, int)
            or isinstance(result_count, bool)
            or result_count < 0
            or not isinstance(hit_ids, list)
            or not all(isinstance(value, str) for value in hit_ids)
            or not isinstance(contexts, list)
            or not all(isinstance(value, str) for value in contexts)
            or not isinstance(observed.get("empty"), bool)
            or observed["empty"] is not (not contexts)
            or observed.get("context_tokens") is not None
        ):
            return False
    return True


def request_observability_contract(
    records: Sequence[dict[str, Any]], *, stage_profiling: bool
) -> bool:
    """Require honest retry and client-boundary call evidence per request."""
    required_call_fields = {
        "http_client",
        "database_client",
        "opaque_native_api",
        "total",
        "source",
        "scope",
        "exact_internal_http_round_trips",
        "exact_internal_database_round_trips",
        "internal_round_trip_policy",
    }
    for record in records:
        if record.get("harness_retries") != 0:
            return False
        if record.get("system_internal_retries") is not None:
            return False
        retry_scope = record.get("system_internal_retry_observability")
        if not isinstance(retry_scope, str) or not retry_scope.strip():
            return False
        observed = record.get("observable_call_counts")
        if not stage_profiling:
            if observed is not None:
                return False
            continue
        if not isinstance(observed, dict) or set(observed) != required_call_fields:
            return False
        counts = [
            observed.get("http_client"),
            observed.get("database_client"),
            observed.get("opaque_native_api"),
        ]
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            return False
        if observed.get("total") != sum(counts):
            return False
        expected_scope = (
            "partial_client_boundaries_before_timeout"
            if record.get("timeout") is True
            else "complete_client_boundaries"
        )
        if (
            observed.get("source") != "stage_span_attributes"
            or observed.get("scope") != expected_scope
            or observed.get("exact_internal_http_round_trips") is not None
            or observed.get("exact_internal_database_round_trips") is not None
            or observed.get("internal_round_trip_policy")
            != "unavailable inside compound native APIs; never inferred from one "
            "native API call"
        ):
            return False
    return True


_ADD_CREATION_COUNT_FIELDS = (
    "created_memory_count",
    "created_node_count",
    "created_edge_count",
)


def add_receipt_contract(records: Sequence[dict[str, Any]]) -> bool:
    """Validate honest, comparable creation-count evidence for successful Add.

    Native APIs do not all expose per-call storage deltas.  ``None`` is therefore
    a valid value, but it must be explicit and paired with a source describing
    why the count is or is not available.  This prevents an unavailable count
    from being silently presented as zero.
    """
    for record in records:
        if record.get("success") is not True:
            continue
        observed = record.get("receipt")
        if not isinstance(observed, dict):
            return False
        if not all(field in observed for field in _ADD_CREATION_COUNT_FIELDS):
            return False
        counts = [observed[field] for field in _ADD_CREATION_COUNT_FIELDS]
        if any(
            value is not None
            and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
            for value in counts
        ):
            return False
        available = observed.get("creation_counts_available")
        if not isinstance(available, bool) or available is not any(
            value is not None for value in counts
        ):
            return False
        source = observed.get("creation_count_source")
        if not isinstance(source, str) or not source.strip():
            return False
    return True


def _base_receipt(
    *,
    corpus: LoCoMoCorpus,
    adapter: Adapter,
    build_id: str,
    phase: str,
    timeout_seconds: float = 60.0,
    qps: float = 10.0,
    profile_stages: bool = False,
) -> dict[str, Any]:
    if qps <= 0:
        raise ValueError("qps must be positive")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def redact(value: str) -> str:
        return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1<redacted>\2", value)

    fresh_state_launcher = None
    if os.environ.get("TRACKC_FRESH_STATE_PREEXISTED_CHECK_PASSED") is not None:
        fresh_state_launcher = {
            "schema_version": "table5_track_c_fresh_state_launcher_v0.1.0",
            "policy": os.environ.get("TRACKC_FRESH_STATE_POLICY"),
            "build_id": os.environ.get("TRACKC_FRESH_STATE_BUILD_ID"),
            "kind": os.environ.get("TRACKC_FRESH_STATE_KIND"),
            "primary_identity": os.environ.get("TRACKC_FRESH_STATE_PRIMARY_ID"),
            "secondary_identity": os.environ.get("TRACKC_FRESH_STATE_SECONDARY_ID"),
            "preexisting_check_passed": (
                os.environ.get("TRACKC_FRESH_STATE_PREEXISTED_CHECK_PASSED") == "true"
            ),
        }

    return {
        "schema_version": "table5_track_c_run_v0.3.0",
        "status": "running",
        "started_at": _now(),
        "system": adapter.name,
        "build_id": build_id,
        "phase": phase,
        "execution": {
            "argv": [redact(value) for value in sys.argv],
            "python": sys.executable,
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
            "platform": platform.platform(),
            "git_branch": branch,
            "git_commit": commit,
            "git_status_sha256": hashlib.sha256(status).hexdigest(),
            "benchmark_code_sha256": benchmark_code_sha256(),
            "launcher_script_sha256": os.environ.get("TRACKC_LAUNCHER_SHA256"),
            "formal_schedule_sha256": os.environ.get("TRACKC_FORMAL_SCHEDULE_SHA256"),
            "protocol_receipt_sha256": os.environ.get("TRACKC_PROTOCOL_RECEIPT_SHA256"),
            "dirty_worktree_preserved": bool(status),
            "fresh_state_launcher": fresh_state_launcher,
        },
        "dataset": {
            "path": str(corpus.path),
            "sha256": corpus.sha256,
            "events": len(corpus.all_events),
            "questions_total": sum(
                len(value) for value in corpus.queries_by_sample.values()
            ),
            "formal_questions": len(corpus.formal_queries),
        },
        "protocol": {
            "qps": float(qps),
            "timeout_seconds": float(timeout_seconds),
            "max_in_flight": DEFAULT_MAX_IN_FLIGHT,
            "top_k": 35,
            "retries": 0,
            "harness_retries": 0,
            "system_internal_retry_policy": "native and included in service latency",
            "scheduler": "absolute no-drift open-loop",
            "scheduled_interval_ns": round(1_000_000_000 / float(qps)),
            "admission_qps_relative_tolerance": (
                DEFAULT_ADMISSION_QPS_RELATIVE_TOLERANCE
            ),
            "admission_lag_p99_max_ms": DEFAULT_ADMISSION_LAG_P99_MAX_MS,
            "evaluation_fields_visible_to_adapter": False,
            "stage_profiling": bool(profile_stages),
            "stage_trace_schema": (
                "table5_track_c_span_v0.1.0" if profile_stages else None
            ),
            "per_request_call_observability": bool(profile_stages),
            "call_observation_fields": [
                "http_client",
                "database_client",
                "opaque_native_api",
                "total",
                "source",
                "scope",
                "exact_internal_http_round_trips",
                "exact_internal_database_round_trips",
                "internal_round_trip_policy",
            ],
            "internal_round_trip_policy": (
                "client-boundary counts only; exact HTTP/database round trips inside "
                "compound native APIs are unavailable and never inferred"
            ),
            "late_timeout_outcome_policy": (
                "one sidecar record after every timed-out worker drains; the "
                "client-visible timeout record is immutable"
            ),
            "late_outcome_schema": "table5_track_c_late_outcome_v0.1.0",
            "fresh_state_policy": (
                "hashed launcher refuses every pre-existing per-point database, "
                "volume, or namespace before creation"
            ),
            "fresh_state_receipt_schema": (
                "table5_track_c_fresh_state_launcher_v0.1.0"
            ),
        },
    }


def _snapshot_fingerprint(adapter: Adapter) -> dict[str, Any]:
    method = getattr(adapter, "snapshot_fingerprint", None)
    if method is None:
        raise RuntimeError(
            f"{adapter.name} does not implement snapshot_fingerprint; refusing reuse"
        )
    result = method()
    if not isinstance(result, dict) or not result:
        raise RuntimeError(f"{adapter.name} returned an empty snapshot fingerprint")
    return result


def _load_reused_build(
    path: str | Path, *, adapter: Adapter, corpus: LoCoMoCorpus
) -> tuple[dict[str, Any], str]:
    path = Path(path)
    encoded = path.read_bytes()
    build = json.loads(encoded)
    checks = {
        "schema": build.get("schema_version") == "table5_track_c_search_build_v0.1.0",
        "complete": build.get("status") == "complete",
        "system": build.get("system") == adapter.name,
        "dataset_sha256": build.get("dataset", {}).get("sha256") == corpus.sha256,
        "fingerprint": isinstance(build.get("snapshot_fingerprint"), dict)
        and bool(build["snapshot_fingerprint"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"reused build receipt failed {failed}: {path.resolve()}")
    return build, hashlib.sha256(encoded).hexdigest()


def run_build_search(
    *,
    adapter: Adapter,
    corpus: LoCoMoCorpus,
    build_id: str,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Build one immutable Search corpus without issuing warmup/formal queries."""
    run_dir = Path(run_dir)
    _ensure_empty_run_dir(run_dir)
    receipt_path = run_dir / "build_receipt.json"
    receipt = _base_receipt(
        corpus=corpus,
        adapter=adapter,
        build_id=build_id,
        phase="build:search",
        timeout_seconds=60.0,
        qps=10.0,
        profile_stages=False,
    )
    receipt["schema_version"] = "table5_track_c_search_build_v0.1.0"
    receipt["protocol"].update(
        {
            "qps": None,
            "formal_admissions": 0,
            "warmup_admissions": 0,
            "queries_executed": False,
            "snapshot_must_remain_immutable": True,
        }
    )
    _write_json(receipt_path, receipt)
    try:
        receipt["schema_gate"] = adapter.init_schema()
        started_ns = time.perf_counter_ns()
        receipt["build_started_at"] = _now()
        with HostResourceSampler() as resources:
            receipt["build"] = adapter.ingest_history(corpus.all_events)
            receipt["build_finalize"] = adapter.finalize_build()
        receipt["build_wall_seconds"] = (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000_000
        receipt["build_resources"] = resources.summary()
        receipt["build_stats"] = adapter.stats()
        receipt["snapshot_fingerprint"] = _snapshot_fingerprint(adapter)
        receipt["build_completed_at"] = _now()
        receipt["status"] = "complete"
        receipt["completed_at"] = _now()
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = _now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        _write_json(receipt_path, receipt)
        raise
    finally:
        adapter.close()


def run_search(
    *,
    adapter: Adapter,
    corpus: LoCoMoCorpus,
    build_id: str,
    run_dir: str | Path,
    timeout_seconds: float = 60.0,
    qps: float = 10.0,
    profile_stages: bool = False,
    reuse_build_receipt: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    _ensure_empty_run_dir(run_dir)
    receipt_path = run_dir / "run_receipt.json"
    receipt = _base_receipt(
        corpus=corpus,
        adapter=adapter,
        build_id=build_id,
        phase="search",
        timeout_seconds=timeout_seconds,
        qps=qps,
        profile_stages=profile_stages,
    )
    _write_json(receipt_path, receipt)
    try:
        if profile_stages and hasattr(adapter, "enable_stage_tracing"):
            adapter.enable_stage_tracing()  # type: ignore[attr-defined]
        reused_build = None
        reused_sha256 = None
        if reuse_build_receipt is not None:
            reused_build, reused_sha256 = _load_reused_build(
                reuse_build_receipt, adapter=adapter, corpus=corpus
            )
        receipt["schema_gate"] = adapter.init_schema()
        if hasattr(adapter, "search_semantics"):
            receipt["search_semantics"] = adapter.search_semantics()  # type: ignore[attr-defined]
            receipt["semantic_boundary"] = receipt["search_semantics"].get(
                "semantic_boundary"
            )
        _write_json(receipt_path, receipt)
        if reused_build is None:
            build_started_ns = time.perf_counter_ns()
            receipt["build_started_at"] = _now()
            with HostResourceSampler() as build_resources:
                receipt["build"] = adapter.ingest_history(corpus.all_events)
                receipt["build_finalize"] = adapter.finalize_build()
            receipt["build_wall_seconds"] = (
                time.perf_counter_ns() - build_started_ns
            ) / 1_000_000_000
            receipt["build_resources"] = build_resources.summary()
            receipt["build_stats"] = adapter.stats()
            receipt["build_completed_at"] = _now()
            receipt["build_mode"] = "fresh_ingest"
        else:
            prepare = getattr(adapter, "prepare_reused_build", None)
            if prepare is not None:
                prepare(corpus)
            observed = _snapshot_fingerprint(adapter)
            expected = reused_build["snapshot_fingerprint"]
            if observed != expected:
                raise RuntimeError(
                    "snapshot fingerprint mismatch: "
                    f"expected={expected!r}, observed={observed!r}"
                )
            receipt["build_mode"] = "cloned_snapshot"
            receipt["build_wall_seconds"] = 0.0
            receipt["build_stats"] = adapter.stats()
            receipt["build_completed_at"] = _now()
            receipt["build_reuse"] = {
                "receipt_path": str(Path(reuse_build_receipt).resolve()),
                "receipt_sha256": reused_sha256,
                "source_build_id": reused_build["build_id"],
                "expected_fingerprint": expected,
                "observed_fingerprint": observed,
                "dataset_sha256_match": True,
            }
        _write_json(receipt_path, receipt)

        warmups = _warmup_queries(corpus)
        if len(corpus.formal_queries) != EXPECTED_FORMAL_QUESTION_COUNT:
            raise RuntimeError("formal Search cardinality changed after dataset gate")

        def search_call(item: QueryItem) -> dict[str, Any]:
            result = adapter.search(_retrieval_query(item), top_k=35)
            semantics = receipt.get("search_semantics") or {}
            if semantics.get("generation_llm_policy") == "forbidden_fail_closed":
                if result.get("llm_call_count") != 0:
                    raise RecordedRequestFailure(
                        f"LLM-free Search contract failed for {item.question_id}: "
                        f"llm_call_count={result.get('llm_call_count')!r}",
                        result,
                    )
                if result.get("cross_encoder_call_count") != 0:
                    raise RecordedRequestFailure(
                        f"model-free reranker contract failed for {item.question_id}: "
                        "cross_encoder_call_count="
                        f"{result.get('cross_encoder_call_count')!r}",
                        result,
                    )
                if result.get("semantic_boundary") != semantics.get(
                    "semantic_boundary"
                ):
                    raise RecordedRequestFailure(
                        f"Search semantic boundary mismatch for {item.question_id}",
                        result,
                    )
            return result

        trace_context = (
            SpanRecorder(run_dir / "spans.jsonl")
            if profile_stages
            else nullcontext(None)
        )
        late_path = run_dir / "late_outcomes.jsonl"
        with JsonlSink(late_path) as late_sink:
            with trace_context as span_recorder:
                with HostResourceSampler() as search_resources:
                    with JsonlSink(run_dir / "warmup.jsonl") as sink:
                        warmup_records = asyncio.run(
                            run_sequential_warmup(
                                warmups,
                                search_call,
                                sink=sink,
                                phase="warmup_search",
                                system=adapter.name,
                                build_id=build_id,
                                timeout_seconds=timeout_seconds,
                                request_metadata=_query_metadata,
                                span_recorder=span_recorder,
                                late_sink=late_sink,
                            )
                        )
                    receipt["warmup_summary"] = summarize(warmup_records)
                    _write_json(receipt_path, receipt)
                    _gate_search_warmup(warmup_records)
                    runner = OpenLoopRunner(qps=qps, timeout_seconds=timeout_seconds)
                    with JsonlSink(run_dir / "formal.jsonl") as sink:
                        formal_records = asyncio.run(
                            runner.run(
                                corpus.formal_queries,
                                search_call,
                                sink=sink,
                                phase="formal_search",
                                system=adapter.name,
                                build_id=build_id,
                                request_metadata=_query_metadata,
                                span_recorder=span_recorder,
                                late_sink=late_sink,
                            )
                        )
        receipt["formal_summary"] = summarize(formal_records)
        if (receipt.get("search_semantics") or {}).get(
            "generation_llm_policy"
        ) == "forbidden_fail_closed":
            successful = [
                record
                for record in (*warmup_records, *formal_records)
                if record.get("success") is True
            ]
            receipt["llm_conformance"] = {
                "policy": "forbidden_fail_closed",
                "successful_requests_checked": len(successful),
                "llm_call_count": sum(
                    int((record.get("receipt") or {}).get("llm_call_count") or 0)
                    for record in successful
                ),
                "cross_encoder_call_count": sum(
                    int(
                        (record.get("receipt") or {}).get(
                            "cross_encoder_call_count"
                        )
                        or 0
                    )
                    for record in successful
                ),
                "passed": all(
                    (record.get("receipt") or {}).get("llm_call_count") == 0
                    and (record.get("receipt") or {}).get(
                        "cross_encoder_call_count"
                    )
                    == 0
                    for record in successful
                ),
            }
            if not receipt["llm_conformance"]["passed"]:
                raise RuntimeError("LLM-free Search conformance failed")
        receipt["quality"] = evidence_quality(formal_records)
        receipt["search_resources"] = search_resources.summary()
        receipt["late_outcome_summary"] = _late_outcome_summary(late_path)
        receipt["scheduler_observed"] = {
            "max_queue_depth": runner.max_observed_queue_depth,
            "max_active": runner.max_observed_active,
        }
        if profile_stages:
            receipt["time_breakdown"] = summarize_spans(run_dir / "spans.jsonl")
        receipt["final_stats"] = adapter.stats()
        receipt["status"] = "complete"
        receipt["completed_at"] = _now()
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = _now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        _write_json(receipt_path, receipt)
        raise
    finally:
        adapter.close()


def _add_items(
    corpus: LoCoMoCorpus, definition: str
) -> tuple[list[EventItem], list[EventItem]]:
    source = corpus.all_events[:2_010]
    if len(source) != 2_010:
        raise RuntimeError(f"need 2,010 LoCoMo events for Add, found {len(source)}")
    items = [
        replace(
            item,
            event_id=f"trackc_{definition}_{index:04d}_{item.event_id}",
            ordinal=100_000_000 + index,
            metadata={**item.metadata, "track_c_add_index": index},
        )
        for index, item in enumerate(source)
    ]
    return items[:10], items[10:]


def run_add(
    *,
    adapter: Adapter,
    corpus: LoCoMoCorpus,
    build_id: str,
    definition: str,
    run_dir: str | Path,
    timeout_seconds: float = 60.0,
    qps: float = 10.0,
    profile_stages: bool = False,
) -> dict[str, Any]:
    if definition not in {"native", "source_to_searchable"}:
        raise ValueError(f"unknown Add definition: {definition}")
    run_dir = Path(run_dir)
    _ensure_empty_run_dir(run_dir)
    receipt_path = run_dir / "run_receipt.json"
    receipt = _base_receipt(
        corpus=corpus,
        adapter=adapter,
        build_id=build_id,
        phase=f"add:{definition}",
        timeout_seconds=timeout_seconds,
        qps=qps,
        profile_stages=profile_stages,
    )
    receipt["add_definition"] = definition
    receipt["state_mode"] = "fresh_empty"
    receipt["protocol"].update({"warmup_admissions": 10, "formal_admissions": 2_000})
    _write_json(receipt_path, receipt)
    visibility = definition == "source_to_searchable"
    progress_lock = threading.Lock()
    request_progress: dict[str, dict[str, Any]] = {}

    def initial_progress() -> dict[str, Any]:
        return {
            "commit_observed": False,
            "visibility_requested": visibility,
            "visibility_probe_started": False,
            "committed_at_ns": None,
            "searchable_at_ns": None,
            "visibility_probe": None,
            "visibility_error": None,
        }

    def partial_receipt(item: EventItem) -> dict[str, Any]:
        with progress_lock:
            return dict(request_progress.get(item.event_id, initial_progress()))

    try:
        if profile_stages and hasattr(adapter, "enable_stage_tracing"):
            adapter.enable_stage_tracing()  # type: ignore[attr-defined]
        receipt["schema_gate"] = adapter.init_schema()
        if hasattr(adapter, "add_semantics"):
            receipt["add_semantics"] = adapter.add_semantics()  # type: ignore[attr-defined]
            receipt["semantic_boundary"] = receipt["add_semantics"].get(
                "semantic_boundary"
            )
        warmups, formal = _add_items(corpus, definition)

        def add_call(item: EventItem) -> dict[str, Any]:
            with progress_lock:
                request_progress[item.event_id] = initial_progress()

            def publish(update: Mapping[str, Any]) -> None:
                with progress_lock:
                    request_progress[item.event_id].update(dict(update))

            result = adapter.add(item, visibility=visibility, progress=publish)
            semantics = receipt.get("add_semantics") or {}
            if semantics.get("construction_llm_policy") == "forbidden_fail_closed":
                if result.get("llm_call_count") != 0:
                    raise RecordedRequestFailure(
                        f"LLM-free Add contract failed for {item.event_id}: "
                        f"llm_call_count={result.get('llm_call_count')!r}",
                        result,
                    )
                if result.get("semantic_boundary") != semantics.get(
                    "semantic_boundary"
                ):
                    raise RecordedRequestFailure(
                        f"prepared-item semantic boundary mismatch for {item.event_id}",
                        result,
                    )
            merged = partial_receipt(item)
            merged.update(result)
            merged["commit_observed"] = merged.get("committed_at_ns") is not None
            if visibility:
                merged["visibility_probe_started"] = True
            publish(merged)
            if visibility and merged.get("visibility_probe") is not True:
                detail = merged.get("visibility_error") or "probe returned false"
                raise RecordedRequestFailure(
                    f"visibility probe failed for {item.event_id}: {detail}", merged
                )
            return merged

        trace_context = (
            SpanRecorder(run_dir / "spans.jsonl")
            if profile_stages
            else nullcontext(None)
        )
        late_path = run_dir / "late_outcomes.jsonl"
        with JsonlSink(late_path) as late_sink:
            with trace_context as span_recorder:
                with HostResourceSampler() as resources:
                    with JsonlSink(run_dir / "warmup.jsonl") as sink:
                        warmup_records = asyncio.run(
                            run_sequential_warmup(
                                warmups,
                                add_call,
                                sink=sink,
                                phase=f"warmup_add_{definition}",
                                system=adapter.name,
                                build_id=build_id,
                                timeout_seconds=timeout_seconds,
                                request_metadata=_event_metadata,
                                partial_receipt=partial_receipt,
                                span_recorder=span_recorder,
                                late_sink=late_sink,
                            )
                        )
                    runner = OpenLoopRunner(qps=qps, timeout_seconds=timeout_seconds)
                    with JsonlSink(run_dir / "formal.jsonl") as sink:
                        formal_records = asyncio.run(
                            runner.run(
                                formal,
                                add_call,
                                sink=sink,
                                phase=f"formal_add_{definition}",
                                system=adapter.name,
                                build_id=build_id,
                                request_metadata=_event_metadata,
                                partial_receipt=partial_receipt,
                                span_recorder=span_recorder,
                                late_sink=late_sink,
                            )
                        )
        receipt["warmup_summary"] = summarize(warmup_records)
        receipt["formal_summary"] = summarize(formal_records)
        if (receipt.get("add_semantics") or {}).get(
            "construction_llm_policy"
        ) == "forbidden_fail_closed":
            successful = [
                record
                for record in (*warmup_records, *formal_records)
                if record.get("success") is True
            ]
            receipt["llm_conformance"] = {
                "policy": "forbidden_fail_closed",
                "successful_requests_checked": len(successful),
                "llm_call_count": sum(
                    int((record.get("receipt") or {}).get("llm_call_count") or 0)
                    for record in successful
                ),
                "passed": all(
                    (record.get("receipt") or {}).get("llm_call_count") == 0
                    for record in successful
                ),
            }
            if not receipt["llm_conformance"]["passed"]:
                raise RuntimeError("LLM-free Add conformance failed")
        receipt["resources"] = resources.summary()
        receipt["late_outcome_summary"] = _late_outcome_summary(late_path)
        receipt["scheduler_observed"] = {
            "max_queue_depth": runner.max_observed_queue_depth,
            "max_active": runner.max_observed_active,
        }
        if profile_stages:
            receipt["time_breakdown"] = summarize_spans(run_dir / "spans.jsonl")
        receipt["final_stats"] = adapter.stats()
        receipt["status"] = "complete"
        receipt["completed_at"] = _now()
        _write_json(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = _now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        _write_json(receipt_path, receipt)
        raise
    finally:
        adapter.close()
