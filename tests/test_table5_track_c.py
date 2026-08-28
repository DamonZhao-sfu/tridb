from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from bench.agent_memory.table5_track_c.dataset import (
    EXPECTED_FORMAL_QUESTION_COUNT,
    EXPECTED_QUESTION_COUNT,
    representative_warmup_indices,
)
from bench.agent_memory.table5_track_c.dataset import EventItem, QueryItem
from bench.agent_memory.table5_track_c.adapters.memos import MemosAdapter, MemosConfig
from bench.agent_memory.table5_track_c.aggregate import (
    _clusters,
    _decomposition_row,
    _metric_row,
    aggregate_answer_quality,
    aggregate_decomposition,
)
from bench.agent_memory.table5_track_c.conformance import (
    SCOPE_CONTROL_TEXT,
    VISIBILITY_TARGET_TEXT,
)
from bench.agent_memory.table5_track_c.scheduler import (
    DEFAULT_MAX_IN_FLIGHT,
    JsonlSink,
    OpenLoopRunner,
    RecordedRequestFailure,
    percentile,
    run_sequential_warmup,
)
from bench.agent_memory.table5_track_c.protocol import (
    _gate_search_warmup,
    _retrieval_query,
    benchmark_code_sha256,
    evidence_quality,
    request_observability_contract,
    run_build_search,
    run_search,
)
from bench.agent_memory.table5_track_c.quality import (
    JUDGE_PROMPT,
    _judge_label,
    _summary,
    verify_quality_tree,
)
from bench.agent_memory.table5_track_c.verify import (
    _add_progress_evidence,
    _admission_schedule_checks,
    _in_request_order,
    _indices_exact,
    _late_outcome_checks,
    _timings_consistent,
)
from bench.agent_memory.table5_track_c.figures import (
    plot_add_ecdf,
    plot_add_p99,
    plot_latency_quality_frontier,
    plot_search_box,
    plot_search_ecdf,
    plot_search_p99,
    plot_success_rate,
)
from bench.agent_memory.table5_track_c.stats import summarize
from bench.agent_memory.table5_track_c.sweep import validate_manifest
from bench.agent_memory.table5_track_c.tracing import (
    SpanRecorder,
    observable_call_counts_match_spans,
    stage_span,
    summarize_spans,
)


def _query(index: int, length: int = 1) -> QueryItem:
    return QueryItem("sample", str(index), "q" * length, "a", 1, (), index)


def test_frozen_question_counts_are_not_conflated():
    assert EXPECTED_QUESTION_COUNT == 1_986
    assert EXPECTED_FORMAL_QUESTION_COUNT == 1_787


def test_conformance_visibility_controls_are_explicit_durable_memories():
    for text in (VISIBILITY_TARGET_TEXT, SCOPE_CONTROL_TEXT):
        assert text.startswith("Please remember this for future conversations:")
        assert "access code" in text
        assert "conformance phrase" not in text.lower()


def test_representative_warmup_matches_head_quantile_longest_shape():
    queries = [_query(index, 1) for index in range(100)]
    queries[70] = _query(70, 1_000)
    queries[71] = _query(71, 999)
    selected = representative_warmup_indices(queries)
    assert len(selected) == 12
    assert selected[:3] == (0, 1, 2)
    assert 70 in selected
    assert 71 in selected
    assert len(set(selected)) == len(selected)


def test_percentile_uses_linear_interpolation():
    assert percentile([0, 10], 90) == pytest.approx(9.0)
    assert percentile([], 90) is None


def test_search_warmup_gate_rejects_all_non_timeout_infrastructure_failures():
    records = [
        {"success": False, "timeout": False, "error": "DatabaseError: sealed"},
        {"success": False, "timeout": False, "error": "DatabaseError: sealed"},
    ]
    with pytest.raises(RuntimeError, match="all Search warmups failed"):
        _gate_search_warmup(records)


def test_search_warmup_gate_retains_true_timeout_overload():
    _gate_search_warmup(
        [
            {
                "success": False,
                "timeout": True,
                "error": "TimeoutError: exceeded 60.000s",
            }
        ]
    )


def test_open_loop_retains_failure_and_absolute_schedule(tmp_path):
    output = tmp_path / "requests.jsonl"

    def call(value):
        if value == 1:
            raise RuntimeError("expected")
        time.sleep(0.002)
        return {"value": value}

    async def execute():
        with JsonlSink(output) as sink:
            return await OpenLoopRunner(qps=100.0).run(
                [0, 1, 2],
                call,
                sink=sink,
                phase="formal_search",
                system="fake",
                build_id="b1",
            )

    records = asyncio.run(execute())
    persisted = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == len(persisted) == 3
    assert sum(record["success"] for record in records) == 2
    assert "RuntimeError: expected" in records[1]["error"]
    assert records[0]["schema_version"] == "table5_track_c_request_v0.3.0"
    assert records[0]["harness_retries"] == 0
    assert records[0]["system_internal_retries"] is None
    assert records[0]["system_internal_retry_observability"].startswith(
        "native counter unavailable"
    )
    assert records[0]["observable_call_counts"] is None
    assert records[0]["queue_depth_at_admission"] >= 1
    assert records[0]["active_at_start"] >= 1
    interval = records[2]["scheduled_at_ns"] - records[1]["scheduled_at_ns"]
    assert interval == 10_000_000
    summary = summarize(records)
    assert summary["total"] == 3
    assert summary["successful"] == 2
    assert summary["service_latency_ms"]["count"] == 2
    assert summary["scheduled_qps"] == pytest.approx(100.0)
    assert summary["admission_qps"] > 50.0
    assert summary["completion_throughput_qps"] > 0
    assert summary["admission_lag_ms"]["count"] == 3


def test_summary_distinguishes_scheduled_from_actual_admission_qps():
    records = [
        {
            "success": True,
            "scheduled_at_ns": index * 100_000_000,
            "admitted_at_ns": index * 200_000_000,
            "completed_at_ns": index * 200_000_000 + 10_000_000,
            "service_latency_ms": 10.0,
            "admission_lag_ms": index * 100.0,
            "queue_latency_ms": index * 100.0,
            "user_visible_latency_ms": index * 100.0 + 10.0,
        }
        for index in range(3)
    ]

    summary = summarize(records)

    assert summary["scheduled_qps"] == pytest.approx(10.0)
    assert summary["admission_qps"] == pytest.approx(5.0)


def test_summary_excludes_inter_build_idle_time_from_pooled_qps():
    records = []
    for build_id, origin in (("b1", 0), ("b2", 100_000_000_000)):
        for index in range(3):
            scheduled = origin + index * 100_000_000
            records.append(
                {
                    "build_id": build_id,
                    "success": True,
                    "scheduled_at_ns": scheduled,
                    "admitted_at_ns": scheduled,
                    "completed_at_ns": scheduled + 10_000_000,
                    "service_latency_ms": 10.0,
                    "admission_lag_ms": 0.0,
                    "queue_latency_ms": 0.0,
                    "user_visible_latency_ms": 10.0,
                }
            )

    summary = summarize(records)

    assert summary["scheduled_qps"] == pytest.approx(10.0)
    assert summary["admission_qps"] == pytest.approx(10.0)


def test_metric_row_separates_success_service_from_all_admission_latency():
    records = [
        {
            "build_id": "b1",
            "sample_id": "sample-a",
            "success": True,
            "scheduled_at_ns": 0,
            "completed_at_ns": 10_000_000,
            "service_latency_ms": 8.0,
            "user_visible_latency_ms": 10.0,
            "admission_lag_ms": 1.0,
            "queue_latency_ms": 1.0,
        },
        {
            "build_id": "b1",
            "sample_id": "sample-b",
            "success": False,
            "timeout": True,
            "error": "TimeoutError: exceeded 60.000s",
            "scheduled_at_ns": 100_000_000,
            "completed_at_ns": 60_105_000_000,
            "service_latency_ms": 60_000.0,
            "user_visible_latency_ms": 60_005.0,
            "admission_lag_ms": 5.0,
            "queue_latency_ms": 0.0,
        },
    ]

    row = _metric_row("fake", "search", "b1", records)

    assert row["formal_requests"] == 2
    assert row["successful"] == 1
    assert row["failed"] == 1
    assert row["service_mean_ms"] == 8.0
    assert row["service_p99_ms"] == 8.0
    assert row["user_visible_mean_ms"] == pytest.approx(30_007.5)
    assert row["user_visible_p90_ms"] == pytest.approx(54_005.5)
    assert row["user_visible_p99_ms"] == pytest.approx(59_405.05)
    assert row["user_visible_max_ms"] == 60_005.0
    assert row["user_visible_mean_ci_low_ms"] is not None
    assert row["user_visible_mean_ci_high_ms"] is not None


def test_bootstrap_clusters_same_conversation_across_fresh_builds():
    records = [
        {"build_id": build, "sample_id": sample, "success": True}
        for build in ("b1", "b2", "b3")
        for sample in ("conv-a", "conv-b")
    ]

    clusters = _clusters(records)

    assert len(clusters) == 2
    assert sorted(len(cluster) for cluster in clusters) == [3, 3]
    assert all(len({row["sample_id"] for row in cluster}) == 1 for cluster in clusters)


def test_bootstrap_refuses_missing_conversation_identity():
    with pytest.raises(ValueError, match="sample_id"):
        _clusters([{"build_id": "b1", "success": True}])


def test_runner_retains_partial_receipt_for_measured_visibility_failure(tmp_path):
    output = tmp_path / "visibility.jsonl"

    def call(_):
        raise RecordedRequestFailure(
            "visibility probe failed",
            {
                "committed_at_ns": 123,
                "searchable_at_ns": None,
                "visibility_probe": False,
                "visibility_error": "probe returned false",
            },
        )

    async def execute():
        with JsonlSink(output) as sink:
            return await OpenLoopRunner(qps=10.0).run(
                [0],
                call,
                sink=sink,
                phase="formal_add_source_to_searchable",
                system="fake",
                build_id="b1",
            )

    record = asyncio.run(execute())[0]
    assert record["success"] is False
    assert "visibility probe failed" in record["error"]
    assert record["receipt"]["committed_at_ns"] == 123
    assert record["receipt"]["visibility_probe"] is False


def test_runner_retains_progress_receipt_at_client_visible_timeout(tmp_path):
    output = tmp_path / "timeout-progress.jsonl"
    late_output = tmp_path / "late-outcomes.jsonl"
    progress = {
        "commit_observed": True,
        "committed_at_ns": 123,
        "visibility_requested": True,
        "visibility_probe_started": True,
        "visibility_probe": None,
        "visibility_error": None,
        "searchable_at_ns": None,
    }

    def call(_):
        time.sleep(0.03)
        return {**progress, "visibility_probe": True, "searchable_at_ns": 456}

    async def execute():
        with JsonlSink(output) as sink, JsonlSink(late_output) as late_sink:
            return await OpenLoopRunner(qps=10.0, timeout_seconds=0.01).run(
                [0],
                call,
                sink=sink,
                phase="formal_add_source_to_searchable",
                system="fake",
                build_id="b1",
                partial_receipt=lambda _: progress,
                late_sink=late_sink,
            )

    record = asyncio.run(execute())[0]
    assert record["success"] is False
    assert record["timeout"] is True
    assert record["receipt"] == progress
    assert record["receipt"]["commit_observed"] is True
    assert record["receipt"]["visibility_probe_started"] is True
    assert record["receipt"]["visibility_probe"] is None
    late = json.loads(late_output.read_text(encoding="utf-8"))
    assert late["schema_version"] == "table5_track_c_late_outcome_v0.1.0"
    assert late["system"] == "fake"
    assert late["build_id"] == "b1"
    assert late["phase"] == "formal_add_source_to_searchable"
    assert late["request_index"] == 0
    assert late["client_timed_out_at_ns"] == record["completed_at_ns"]
    assert late["worker_completed_at_ns"] >= late["client_timed_out_at_ns"]
    assert late["post_timeout_work_ms"] == pytest.approx(
        (late["worker_completed_at_ns"] - late["client_timed_out_at_ns"]) / 1_000_000
    )
    assert late["final_success"] is True
    assert late["final_error"] is None
    assert late["receipt"]["visibility_probe"] is True
    assert late["receipt"]["searchable_at_ns"] == 456

    receipt = {
        "system": "fake",
        "build_id": "b1",
        "late_outcome_summary": {
            "schema_version": "table5_track_c_late_outcome_summary_v0.1.0",
            "path": str(late_output.resolve()),
            "sha256": hashlib.sha256(late_output.read_bytes()).hexdigest(),
            "records": 1,
            "by_phase": {"formal_add_source_to_searchable": 1},
            "final_success": 1,
            "final_failed": 0,
            "post_timeout_work_ms": {
                "total": late["post_timeout_work_ms"],
                "max": late["post_timeout_work_ms"],
            },
        },
    }
    assert all(
        _late_outcome_checks(
            late_path=late_output,
            late_records=[late],
            request_records=[record],
            receipt=receipt,
        ).values()
    )


def test_sequential_warmup_drains_timed_out_worker_before_next_request(tmp_path):
    output = tmp_path / "warmup.jsonl"
    late_output = tmp_path / "late.jsonl"
    worker_times: dict[int, tuple[int, int]] = {}

    def call(value: int) -> dict[str, int]:
        started = time.perf_counter_ns()
        time.sleep(0.03)
        completed = time.perf_counter_ns()
        worker_times[value] = (started, completed)
        return {"value": value}

    async def execute():
        with JsonlSink(output) as sink, JsonlSink(late_output) as late_sink:
            return await run_sequential_warmup(
                [0, 1],
                call,
                sink=sink,
                phase="warmup_search",
                system="fake",
                build_id="b1",
                timeout_seconds=0.01,
                late_sink=late_sink,
            )

    records = asyncio.run(execute())

    assert all(record["timeout"] is True for record in records)
    assert worker_times[1][0] >= worker_times[0][1]
    assert len(late_output.read_text(encoding="utf-8").splitlines()) == 2


def test_partial_receipt_error_cannot_hide_primary_request_failure(tmp_path):
    output = tmp_path / "partial-error.jsonl"

    def call(_):
        raise RuntimeError("primary failure")

    def broken_partial(_):
        raise ValueError("progress reader failed")

    async def execute():
        with JsonlSink(output) as sink:
            return await OpenLoopRunner(qps=10.0).run(
                [0],
                call,
                sink=sink,
                phase="formal_add_source_to_searchable",
                system="fake",
                build_id="b1",
                partial_receipt=broken_partial,
            )

    record = asyncio.run(execute())[0]
    assert record["error"] == "RuntimeError: primary failure"
    assert record["receipt"] == {
        "partial_receipt_error": "ValueError: progress reader failed"
    }


def test_add_progress_gate_accepts_explicit_timeout_state_and_rejects_omission():
    timeout = {
        "success": False,
        "timeout": True,
        "error": "TimeoutError: exceeded 60.000s",
        "receipt": {
            "commit_observed": True,
            "committed_at_ns": 123,
            "visibility_requested": True,
            "visibility_probe_started": True,
            "visibility_probe": None,
            "visibility_error": None,
            "searchable_at_ns": None,
        },
    }
    assert _add_progress_evidence([timeout], visibility=True)
    del timeout["receipt"]["visibility_probe_started"]
    assert not _add_progress_evidence([timeout], visibility=True)


def test_add_receipt_records_fresh_empty_state(tmp_path, monkeypatch):
    from bench.agent_memory.table5_track_c import protocol

    class Adapter:
        name = "fake"

        def init_schema(self):
            return {"ok": True}

        def add(self, item, *, visibility=False):
            raise AssertionError("empty test corpus must not add")

        def stats(self):
            return {"rows": 0}

        def close(self):
            return None

    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def summary(self):
            return {"sample_count": 0}

    corpus = SimpleNamespace(
        path=tmp_path / "locomo.json",
        sha256="frozen",
        all_events=(),
        queries_by_sample={},
        formal_queries=(),
    )
    monkeypatch.setattr(protocol, "_add_items", lambda *_: ([], []))
    monkeypatch.setattr(protocol, "HostResourceSampler", Sampler)

    receipt = protocol.run_add(
        adapter=Adapter(),
        corpus=corpus,
        build_id="fresh-add",
        definition="native",
        run_dir=tmp_path / "run",
    )

    assert receipt["state_mode"] == "fresh_empty"
    assert receipt["protocol"]["warmup_admissions"] == 10
    assert receipt["protocol"]["formal_admissions"] == 2_000


def test_run_add_persists_commit_and_probe_progress_when_request_times_out(
    tmp_path, monkeypatch
):
    from bench.agent_memory.table5_track_c import protocol

    class Adapter:
        name = "fake"

        def init_schema(self):
            return {"ok": True}

        def add(self, item, *, visibility=False, progress=None):
            committed_at_ns = time.perf_counter_ns()
            progress({"commit_observed": True, "committed_at_ns": committed_at_ns})
            progress({"visibility_probe_started": True})
            time.sleep(0.03)
            return {
                "committed_at_ns": committed_at_ns,
                "searchable_at_ns": time.perf_counter_ns(),
                "visibility_probe": True,
                "visibility_error": None,
            }

        def stats(self):
            return {"rows": 2}

        def close(self):
            return None

    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def summary(self):
            return {"sample_count": 0}

    def event(event_id):
        return EventItem("sample", event_id, "S1", "now", "user", "remember", 1, {})

    corpus = SimpleNamespace(
        path=tmp_path / "locomo.json",
        sha256="frozen",
        all_events=(),
        queries_by_sample={},
        formal_queries=(),
    )
    monkeypatch.setattr(
        protocol, "_add_items", lambda *_: ([event("warmup")], [event("formal")])
    )
    monkeypatch.setattr(protocol, "HostResourceSampler", Sampler)

    protocol.run_add(
        adapter=Adapter(),
        corpus=corpus,
        build_id="timeout-progress",
        definition="source_to_searchable",
        run_dir=tmp_path / "run",
        timeout_seconds=0.01,
    )

    record = json.loads((tmp_path / "run/formal.jsonl").read_text())
    assert record["timeout"] is True
    assert record["receipt"]["commit_observed"] is True
    assert record["receipt"]["visibility_requested"] is True
    assert record["receipt"]["visibility_probe_started"] is True
    assert record["receipt"]["visibility_probe"] is None
    assert record["receipt"]["searchable_at_ns"] is None


def test_verify_tree_requires_exact_five_system_45_point_completion_tree(
    tmp_path, monkeypatch
):
    from bench.agent_memory.table5_track_c import verify

    systems = (
        "tridb_gem",
        "mem0",
        "memos",
        "cognee",
        "graphiti_zep_oss_proxy",
    )
    phases = {
        "search": "search",
        "add:native": "add_native",
        "add:source_to_searchable": "add_source_to_searchable",
    }
    code_hash = "a" * 64
    launcher_hash = "b" * 64
    schedule_hash = "c" * 64
    protocol_hash = "d" * 64
    identities = {}
    for system in systems:
        for phase, stem in phases.items():
            for round_name in ("b1", "b2", "b3"):
                build_id = f"tc5v2_{system}_{stem}_{round_name}"
                path = tmp_path / system / stem / round_name / "run_receipt.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "execution": {
                                "benchmark_code_sha256": code_hash,
                                "launcher_script_sha256": launcher_hash,
                                "formal_schedule_sha256": schedule_hash,
                                "protocol_receipt_sha256": protocol_hash,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                identities[path.resolve()] = (system, phase, build_id)

    def fake_verify_run(path):
        system, phase, build_id = identities[path.resolve()]
        return {
            "receipt": str(path.resolve()),
            "system": system,
            "phase": phase,
            "build_id": build_id,
            "checks": {"receipt_complete": True},
            "passed": True,
        }

    monkeypatch.setattr(verify, "verify_run", fake_verify_run)
    result = verify.verify_tree(
        tmp_path,
        expected_code_hash=code_hash,
        expected_launcher_hash=launcher_hash,
        expected_schedule_hash=schedule_hash,
        expected_protocol_hash=protocol_hash,
    )

    assert result["schema_version"] == "table5_track_c_verification_v0.3.0"
    assert result["runs_found"] == 45
    assert result["formal_tree_exact"] is True
    assert result["coverage_complete_for_five_systems"] is True
    assert result["graphiti_proxy_expected_in_track_c"] is True
    assert result["production_zep_expected_in_track_c"] is False


def test_runner_writes_one_trace_and_stage_breakdown_per_request(tmp_path):
    requests = tmp_path / "requests.jsonl"
    spans = tmp_path / "spans.jsonl"

    def call(value):
        with stage_span(
            "embedding",
            "fake.embed",
            backend="fake",
            attributes={"observable_call_kind": "http_client"},
        ):
            time.sleep(0.001)
        return {"value": value}

    async def execute():
        with JsonlSink(requests) as sink, SpanRecorder(spans) as recorder:
            return await OpenLoopRunner(qps=200.0).run(
                [0, 1, 2],
                call,
                sink=sink,
                phase="formal_search",
                system="fake",
                build_id="qps5",
                span_recorder=recorder,
            )

    records = asyncio.run(execute())
    persisted_spans = [json.loads(line) for line in spans.read_text().splitlines()]
    assert len({record["trace_id"] for record in records}) == 3
    assert sum(row["category"] == "request" for row in persisted_spans) == 3
    assert sum(row["category"] == "embedding" for row in persisted_spans) == 3
    assert all(
        row["parent_span_id"] is not None
        for row in persisted_spans
        if row["category"] == "embedding"
    )
    breakdown = summarize_spans(spans)
    assert breakdown["formal_requests"] == 3
    assert breakdown["formal_requests_with_stage_spans"] == 3
    assert breakdown["stages"]["embedding"]["span_count"] == 3
    assert breakdown["stages"]["embedding"]["stage_work_ms"] > 0
    assert all(
        record["observable_call_counts"]["http_client"] == 1 for record in records
    )
    assert all(record["observable_call_counts"]["total"] == 1 for record in records)
    assert request_observability_contract(records, stage_profiling=True)
    records[0]["observable_call_counts"]["exact_internal_http_round_trips"] = 1
    assert not request_observability_contract(records, stage_profiling=True)


def test_stage_breakdown_clips_late_timeout_drain_work(tmp_path):
    spans = tmp_path / "spans.jsonl"
    rows = [
        {
            "trace_id": "trace",
            "category": "request",
            "phase": "formal_search",
            "started_at_ns": 0,
            "completed_at_ns": 10_000_000,
            "duration_ms": 10.0,
        },
        {
            "trace_id": "trace",
            "category": "fusion",
            "phase": "formal_search",
            "started_at_ns": 1_000_000,
            "completed_at_ns": 30_000_000,
            "duration_ms": 29.0,
            "success": True,
        },
    ]
    spans.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    breakdown = summarize_spans(spans)
    fusion = breakdown["stages"]["fusion"]

    assert breakdown["schema_version"] == "table5_track_c_breakdown_v0.2.0"
    assert fusion["raw_stage_work_ms"] == pytest.approx(29.0)
    assert fusion["client_visible_stage_work_ms"] == pytest.approx(9.0)
    assert fusion["stage_work_ms"] == pytest.approx(9.0)
    assert fusion["post_request_work_ms"] == pytest.approx(20.0)
    assert fusion["per_request_union_ms"]["mean"] == pytest.approx(9.0)
    assert breakdown["post_request_stage_work_ms"] == pytest.approx(20.0)
    assert breakdown["unattributed_per_request_ms"]["mean"] == pytest.approx(1.0)


def test_observable_call_counts_recompute_exact_and_timeout_prefix():
    requests = [
        {
            "trace_id": "complete",
            "timeout": False,
            "observable_call_counts": {
                "http_client": 1,
                "database_client": 0,
                "opaque_native_api": 1,
            },
        },
        {
            "trace_id": "timeout",
            "timeout": True,
            "observable_call_counts": {
                "http_client": 1,
                "database_client": 0,
                "opaque_native_api": 0,
            },
        },
    ]
    spans = [
        {
            "trace_id": "complete",
            "category": "embedding",
            "attributes": {"observable_call_kind": "http_client"},
        },
        {
            "trace_id": "complete",
            "category": "fusion",
            "attributes": {"observable_call_kind": "opaque_native_api"},
        },
        {
            "trace_id": "timeout",
            "category": "embedding",
            "attributes": {
                "observable_call_kind": "http_client",
                "observable_call_count": 2,
            },
        },
    ]

    assert observable_call_counts_match_spans(requests, spans)
    requests[0]["observable_call_counts"]["http_client"] = 0
    assert not observable_call_counts_match_spans(requests, spans)
    requests[0]["observable_call_counts"]["http_client"] = 1
    requests[1]["observable_call_counts"]["http_client"] = 3
    assert not observable_call_counts_match_spans(requests, spans)


def test_sweep_manifest_requires_exact_1_5_10_coverage(tmp_path):
    points = []
    for qps in (1, 5, 10):
        points.append(
            {
                "system": "fake",
                "workload": "search",
                "qps": qps,
                "run_dir": str(tmp_path / f"qps{qps}"),
                "command": [
                    "python",
                    "-m",
                    "bench.agent_memory.table5_track_c.cli",
                    "search",
                    "--qps",
                    str(qps),
                    "--profile-stages",
                ],
            }
        )
    assert len(validate_manifest({"points": points})) == 3
    with pytest.raises(ValueError, match="incomplete QPS coverage"):
        validate_manifest({"points": points[:-1]})
    subset = validate_manifest({"qps": [1, 5], "points": points[:-1]})
    assert [point["qps"] for point in subset] == [1, 5]
    matrix = validate_manifest(
        {
            "qps": [1, 5],
            "matrix": {
                "systems": ["fake"],
                "workloads": ["search"],
                "point_runner": "runner.sh",
                "output_root": str(tmp_path),
            },
        }
    )
    assert len(matrix) == 2
    assert matrix[1]["run_dir"].endswith("fake/search/qps_5")


def test_evidence_quality_scores_ranked_top_20_and_counts_failures_as_zero():
    records = [
        {
            "success": True,
            "evidence_ids": ["a", "b"],
            "receipt": {"hit_ids": ["a", *[f"x{i}" for i in range(30)]]},
        },
        {
            "success": True,
            "evidence_ids": ["z"],
            "receipt": {"hit_ids": [*[f"x{i}" for i in range(20)], "z"]},
        },
        {"success": False, "evidence_ids": ["a"], "receipt": {"hit_ids": ["a"]}},
    ]
    quality = evidence_quality(records)
    assert quality["successful_queries"] == 2
    assert quality["evidenced_queries"] == 3
    assert quality["evidence_recall_at_20"] == pytest.approx(1 / 6)
    assert quality["evidence_any_hit_at_20"] == pytest.approx(1 / 3)


def test_retrieval_adapter_cannot_see_evaluation_labels():
    query = QueryItem(
        "sample",
        "q1",
        "Where?",
        "gold answer",
        4,
        ("dia-1", "dia-2"),
        0,
    )
    retrieval = _retrieval_query(query)
    assert retrieval.sample_id == query.sample_id
    assert retrieval.question_id == query.question_id
    assert retrieval.question == query.question
    assert retrieval.answer is None
    assert retrieval.category is None
    assert retrieval.evidence_ids == ()


def test_quality_judge_prompt_and_parser_are_deterministic():
    prompt = JUDGE_PROMPT.format(question="q", gold_answer="g", generated_answer="a")
    assert '{"label": "CORRECT"}' in prompt
    assert _judge_label('{"label": "CORRECT"}') == "CORRECT"
    assert _judge_label('```json\n{"label": "CORRECT"}\n```') == "CORRECT"
    assert _judge_label("CORRECT") == "CORRECT"
    assert _judge_label("WRONG") == "WRONG"
    assert _judge_label("CORRECT and WRONG") == "WRONG"
    assert _judge_label("INCORRECT") == "WRONG"
    assert _judge_label("not correct") == "WRONG"


def test_runner_executor_is_not_capped_at_python_default(tmp_path):
    output = tmp_path / "concurrency.jsonl"

    def call(value):
        time.sleep(0.1)
        return {"value": value}

    async def execute():
        runner = OpenLoopRunner(qps=1_000.0, max_in_flight=40)
        with JsonlSink(output) as sink:
            await runner.run(
                list(range(40)),
                call,
                sink=sink,
                phase="concurrency_test",
                system="fake",
                build_id="b1",
            )
        return runner

    runner = asyncio.run(execute())
    assert runner.max_observed_active > 32
    assert len(benchmark_code_sha256()) == 64


def test_default_capacity_covers_the_frozen_timeout_window():
    assert DEFAULT_MAX_IN_FLIGHT >= 10 * 60


def test_verifier_recomputes_every_timing_field():
    record = {
        "scheduled_at_ns": 1_000_000,
        "admitted_at_ns": 2_000_000,
        "started_at_ns": 3_000_000,
        "completed_at_ns": 8_000_000,
        "admission_lag_ms": 1.0,
        "queue_latency_ms": 2.0,
        "service_latency_ms": 5.0,
        "user_visible_latency_ms": 7.0,
    }
    assert _timings_consistent([record])
    record["service_latency_ms"] = 4.9
    assert not _timings_consistent([record])


def test_verifier_rejects_synthetic_10qps_when_real_admissions_are_5qps():
    records = [
        {
            "request_index": index,
            "admitted_at_ns": index * 200_000_000,
            "admission_lag_ms": index * 100.0,
        }
        for index in range(10)
    ]

    checks = _admission_schedule_checks(
        records,
        target_qps=10.0,
        relative_tolerance=0.01,
        lag_p99_max_ms=100.0,
    )

    assert checks["admission_order_exact"] is True
    assert checks["actual_admission_qps_within_tolerance"] is False
    assert checks["admission_lag_p99_within_limit"] is False


def test_verifier_reconstructs_admission_order_from_completion_order():
    records = [
        {"request_index": 2},
        {"request_index": 0},
        {"request_index": 1},
    ]
    assert _indices_exact(records, 3)
    assert [record["request_index"] for record in _in_request_order(records)] == [
        0,
        1,
        2,
    ]
    assert not _indices_exact([*records, {"request_index": 2}], 3)


def test_quality_verifier_accepts_lossless_completion_order(tmp_path, monkeypatch):
    from bench.agent_memory.table5_track_c import quality

    monkeypatch.setattr(quality, "EXPECTED_FORMAL_QUESTION_COUNT", 3)
    run = tmp_path / "tridb_gem" / "b1"
    run.mkdir(parents=True)
    answers = [
        {
            "schema_version": "table5_track_c_answer_v0.2.0",
            "request_index": index,
            "system": "tridb_gem",
            "build_id": "tridb_gem_search_b1",
            "sample_id": "sample-1",
            "question_id": str(index),
            "category": "1",
            "retrieval_success": True,
            "success": True,
            "generated_answer": "answer",
            "context_count": 1,
            "context_tokens": 10,
            "error": None,
        }
        for index in (2, 0, 1)
    ]
    judges = [
        {
            "schema_version": "table5_track_c_judge_v0.2.0",
            "request_index": index,
            "system": "tridb_gem",
            "build_id": "tridb_gem_search_b1",
            "sample_id": "sample-1",
            "question_id": str(index),
            "category": "1",
            "answer_success": True,
            "success": True,
            "correct": index != 2,
            "label": "WRONG" if index == 2 else "CORRECT",
            "error": None,
        }
        for index in (1, 2, 0)
    ]
    answers_path = run / "answers.jsonl"
    judges_path = run / "judges.jsonl"
    answers_path.write_text("".join(json.dumps(row) + "\n" for row in answers))
    judges_path.write_text("".join(json.dumps(row) + "\n" for row in judges))
    summary = quality._summary_evidence(answers, judges)
    summary.update({"system": "tridb_gem", "build_id": "tridb_gem_search_b1"})
    summary_path = run / "quality_summary.json"
    summary_path.write_text(json.dumps(summary))
    formal_path = run / "formal.jsonl"
    formal_path.write_text("formal\n")
    receipt = {
        "schema_version": "table5_track_c_quality_run_v0.2.0",
        "status": "complete",
        "system": "tridb_gem",
        "build_id": "tridb_gem_search_b1",
        "benchmark_code_sha256": "code",
        "formal_records": 3,
        "model": "Qwen/Qwen3-32B",
        "endpoint": "http://127.0.0.1:8000/v1",
        "workers": 32,
        "answer_prompt_sha256": quality.hashlib.sha256(
            quality.ANSWER_PROMPT.encode()
        ).hexdigest(),
        "judge_prompt_sha256": quality.hashlib.sha256(
            quality.JUDGE_PROMPT.encode()
        ).hexdigest(),
        "answers_sha256": quality._sha256(answers_path),
        "judges_sha256": quality._sha256(judges_path),
        "summary_sha256": quality._sha256(summary_path),
        "formal_path": str(formal_path),
        "formal_sha256": quality._sha256(formal_path),
        "overall": summary["overall"],
    }
    (run / "quality_receipt.json").write_text(json.dumps(receipt))

    result = verify_quality_tree(
        tmp_path,
        expected_runs={("tridb_gem", "tridb_gem_search_b1")},
        expected_code_hash="code",
    )
    assert result["coverage_exact"]
    assert result["all_found_runs_pass"]

    summary["overall"].update({"correct": 3, "score": 1.0, "conditional_score": 1.0})
    summary_path.write_text(json.dumps(summary))
    receipt["summary_sha256"] = quality._sha256(summary_path)
    receipt["overall"] = summary["overall"]
    (run / "quality_receipt.json").write_text(json.dumps(receipt))
    tampered = verify_quality_tree(
        tmp_path,
        expected_runs={("tridb_gem", "tridb_gem_search_b1")},
        expected_code_hash="code",
    )
    assert tampered["coverage_exact"]
    assert tampered["all_found_runs_pass"] is False
    assert tampered["runs"][0]["checks"]["summary_evidence_recomputed"] is False
    assert tampered["runs"][0]["checks"]["receipt_overall_matches_summary"] is True


def test_decomposition_preserves_fused_and_visibility_telemetry():
    records = [
        {
            "success": True,
            "started_at_ns": 1_000_000,
            "error": None,
            "receipt": {
                "result_count": 35,
                "cost": {"seconds": 0.010},
                "model_calls": [{"elapsed_seconds": 0.004}],
                "committed_at_ns": 6_000_000,
                "searchable_at_ns": 9_000_000,
                "visibility_probe": True,
                "commit_observed": True,
                "visibility_requested": True,
                "visibility_probe_started": True,
                "created_memory_count": 1,
                "created_node_count": 2,
                "created_edge_count": 3,
                "creation_counts_available": True,
                "creation_count_source": "native_delta",
                "probes": {
                    "candidates_examined": 20,
                    "graph_examined": 4,
                    "bridges_injected": 2,
                    "termination_reason": "term_cond",
                },
            },
        },
        {
            "success": False,
            "timeout": True,
            "started_at_ns": 10_000_000,
            "error": "TimeoutError: exceeded 60.000s",
            "receipt": {
                "commit_observed": True,
                "committed_at_ns": 12_000_000,
                "visibility_requested": True,
                "visibility_probe_started": True,
                "visibility_probe": False,
                "visibility_error": "probe failed before timeout",
                "searchable_at_ns": None,
            },
        },
        {
            "success": False,
            "timeout": True,
            "started_at_ns": 20_000_000,
            "error": "TimeoutError: exceeded 60.000s",
            "receipt": {
                "commit_observed": True,
                "committed_at_ns": 22_000_000,
                "visibility_requested": True,
                "visibility_probe_started": True,
                "visibility_probe": None,
                "visibility_error": None,
                "searchable_at_ns": None,
            },
        },
    ]
    late_outcomes = [
        {
            "final_success": True,
            "post_timeout_work_ms": 12.0,
            "receipt": {
                "committed_at_ns": 12_000_000,
                "visibility_probe": False,
            },
        },
        {
            "final_success": False,
            "post_timeout_work_ms": 28.0,
            "receipt": {
                "committed_at_ns": 22_000_000,
                "visibility_probe": True,
            },
        },
    ]
    row = _decomposition_row(
        "tridb_gem",
        "add:source_to_searchable",
        "b1",
        records,
        late_outcomes,
    )
    assert row["internal_cost_mean_ms"] == pytest.approx(10)
    assert row["model_call_elapsed_mean_ms"] == pytest.approx(4)
    assert row["commit_mean_ms"] == pytest.approx(5)
    assert row["searchable_mean_ms"] == pytest.approx(8)
    assert row["visibility_probe_pass_rate"] == 0.5
    assert row["visibility_probe_false_count"] == 1
    assert row["commit_observed_count"] == 3
    assert row["failed_after_commit_count"] == 2
    assert row["visibility_probe_started_count"] == 3
    assert row["timeout_after_commit_count"] == 2
    assert row["timeout_during_visibility_probe_count"] == 1
    assert row["client_timeout_count"] == 2
    assert row["late_worker_outcome_count"] == 2
    assert row["late_worker_final_success_count"] == 1
    assert row["late_worker_final_failed_count"] == 1
    assert row["late_worker_commit_observed_count"] == 2
    assert row["late_worker_visibility_pass_count"] == 1
    assert row["post_timeout_work_mean_ms"] == 20.0
    assert row["post_timeout_work_max_ms"] == 28.0
    assert row["termination_reasons"] == '{"term_cond": 1}'
    assert row["creation_counts_available_rate"] == 1.0
    assert row["created_memory_count_coverage"] == 1.0
    assert row["created_memory_total"] == 1.0
    assert row["created_node_mean"] == 2.0
    assert row["created_edge_total"] == 3.0
    assert row["creation_count_sources"] == '{"native_delta": 1}'


def test_decomposition_excludes_warmup_late_outcomes(tmp_path):
    run_dir = tmp_path / "cognee" / "add_native" / "b1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_receipt.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "system": "cognee",
                "phase": "add:native",
                "build_id": "b1",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "formal.jsonl").write_text(
        json.dumps(
            {
                "success": False,
                "timeout": True,
                "error": "TimeoutError",
                "receipt": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "late_outcomes.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "phase": phase,
                    "final_success": True,
                    "post_timeout_work_ms": duration,
                    "receipt": {},
                }
            )
            + "\n"
            for phase, duration in (
                ("warmup_add_native", 99.0),
                ("formal_add_native", 5.0),
            )
        ),
        encoding="utf-8",
    )

    rows = aggregate_decomposition(tmp_path)

    assert len(rows) == 2
    assert all(row["late_worker_outcome_count"] == 1 for row in rows)
    assert all(row["post_timeout_work_mean_ms"] == 5.0 for row in rows)


def test_search_finalizes_build_before_measurement(tmp_path, monkeypatch):
    from bench.agent_memory.table5_track_c import protocol

    calls = []

    class Adapter:
        name = "fake"

        def init_schema(self):
            calls.append("init")
            return {"ok": True}

        def ingest_history(self, events):
            calls.append("ingest")
            return {"events": len(events)}

        def finalize_build(self):
            calls.append("finalize")
            return {"ready": True}

        def search(self, item, *, top_k=35):
            raise AssertionError("empty formal corpus must not search")

        def stats(self):
            calls.append("stats")
            return {"ok": True}

        def close(self):
            calls.append("close")

    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def summary(self):
            return {"sample_count": 0}

    corpus = SimpleNamespace(
        path=tmp_path / "locomo.json",
        sha256="frozen",
        all_events=(),
        queries_by_sample={},
        formal_queries=(),
    )
    monkeypatch.setattr(protocol, "EXPECTED_FORMAL_QUESTION_COUNT", 0)
    monkeypatch.setattr(protocol, "_warmup_queries", lambda _: [])
    monkeypatch.setattr(protocol, "HostResourceSampler", Sampler)
    fresh_policy = (
        "hashed launcher refuses every pre-existing per-point database, volume, "
        "or namespace before creation"
    )
    for key, value in {
        "TRACKC_FRESH_STATE_POLICY": fresh_policy,
        "TRACKC_FRESH_STATE_BUILD_ID": "b1",
        "TRACKC_FRESH_STATE_KIND": "postgres_database",
        "TRACKC_FRESH_STATE_PRIMARY_ID": "b1",
        "TRACKC_FRESH_STATE_SECONDARY_ID": "database_owner:hza214",
        "TRACKC_FRESH_STATE_PREEXISTED_CHECK_PASSED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    receipt = run_search(
        adapter=Adapter(),
        corpus=corpus,
        build_id="b1",
        run_dir=tmp_path / "run",
        timeout_seconds=300.0,
        qps=5.0,
        profile_stages=True,
    )
    assert calls[:4] == ["init", "ingest", "finalize", "stats"]
    assert receipt["build_finalize"] == {"ready": True}
    assert receipt["build_wall_seconds"] >= 0
    assert receipt["protocol"]["timeout_seconds"] == 300.0
    assert receipt["protocol"]["qps"] == 5.0
    assert receipt["protocol"]["stage_profiling"] is True
    assert receipt["protocol"]["fresh_state_policy"] == fresh_policy
    assert receipt["execution"]["fresh_state_launcher"] == {
        "schema_version": "table5_track_c_fresh_state_launcher_v0.1.0",
        "policy": fresh_policy,
        "build_id": "b1",
        "kind": "postgres_database",
        "primary_identity": "b1",
        "secondary_identity": "database_owner:hza214",
        "preexisting_check_passed": True,
    }
    assert receipt["time_breakdown"]["formal_requests"] == 0
    assert (tmp_path / "run" / "spans.jsonl").exists()


def test_search_can_reuse_verified_build_without_reingest(tmp_path, monkeypatch):
    from bench.agent_memory.table5_track_c import protocol

    class Adapter:
        name = "fake"

        def __init__(self, fingerprint):
            self.fingerprint = fingerprint
            self.ingests = 0

        def init_schema(self):
            return {"ok": True}

        def ingest_history(self, events):
            self.ingests += 1
            return {"events": len(events)}

        def finalize_build(self):
            return {"ready": True}

        def snapshot_fingerprint(self):
            return dict(self.fingerprint)

        def stats(self):
            return {"fingerprint": self.fingerprint}

        def search(self, item, *, top_k=35):
            raise AssertionError("empty formal corpus must not search")

        def close(self):
            return None

    class Sampler:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def summary(self):
            return {"sample_count": 0}

    corpus = SimpleNamespace(
        path=tmp_path / "locomo.json",
        sha256="frozen",
        all_events=(),
        queries_by_sample={},
        formal_queries=(),
    )
    monkeypatch.setattr(protocol, "EXPECTED_FORMAL_QUESTION_COUNT", 0)
    monkeypatch.setattr(protocol, "_warmup_queries", lambda _: [])
    monkeypatch.setattr(protocol, "HostResourceSampler", Sampler)
    builder = Adapter({"rows": 42})
    build = run_build_search(
        adapter=builder,
        corpus=corpus,
        build_id="canonical",
        run_dir=tmp_path / "build",
    )
    assert builder.ingests == 1
    assert build["snapshot_fingerprint"] == {"rows": 42}

    clone = Adapter({"rows": 42})
    receipt = run_search(
        adapter=clone,
        corpus=corpus,
        build_id="qps5",
        run_dir=tmp_path / "search",
        qps=5,
        reuse_build_receipt=tmp_path / "build" / "build_receipt.json",
    )
    assert clone.ingests == 0
    assert receipt["build_mode"] == "cloned_snapshot"
    assert receipt["build_reuse"]["expected_fingerprint"] == {"rows": 42}
    assert receipt["build_reuse"]["observed_fingerprint"] == {"rows": 42}


def test_memos_scope_is_constructed_once_under_concurrent_add_admission(
    tmp_path, monkeypatch
):
    adapter = MemosAdapter(
        MemosConfig(namespace="test", user_db_dir=str(tmp_path / "users"))
    )
    created = object()
    calls = 0

    def fake_new_scope(sample_id):
        nonlocal calls
        calls += 1
        time.sleep(0.02)
        return created

    monkeypatch.setattr(adapter, "_new_scope", fake_new_scope)
    with ThreadPoolExecutor(max_workers=20) as executor:
        scopes = list(executor.map(adapter._scope, ["conv-new"] * 20))
    assert calls == 1
    assert all(scope is created for scope in scopes)


def test_figures_render_from_verified_shapes(tmp_path):
    run = tmp_path / "runs" / "tridb_gem" / "search" / "b1"
    run.mkdir(parents=True)
    (run / "run_receipt.json").write_text(
        json.dumps({"status": "complete", "phase": "search", "system": "tridb_gem"})
    )
    records = [
        {"success": True, "service_latency_ms": 1.0},
        {"success": True, "service_latency_ms": 2.0},
    ]
    (run / "formal.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    metric_rows = []
    for phase in ("search", "add:native", "add:source_to_searchable"):
        metric_rows.append(
            {
                "system": "tridb_gem",
                "phase": phase,
                "build_id": "pooled",
                "service_p99_ms": "2.0",
                "service_p99_ci_low_ms": "1.5",
                "service_p99_ci_high_ms": "2.5",
                "success_rate": "1.0",
            }
        )
    output = tmp_path / "figures"
    plot_search_ecdf(tmp_path / "runs", output)
    plot_add_ecdf(tmp_path / "runs", output)
    plot_search_box(tmp_path / "runs", output)
    plot_search_p99(metric_rows, output)
    plot_add_p99(metric_rows, output)
    plot_success_rate(metric_rows, output)
    assert (output / "search_latency_ecdf.pdf").stat().st_size > 0
    assert (output / "search_p99_bar.png").stat().st_size > 0
    assert (output / "add_p99_bar.pdf").stat().st_size > 0
    assert (output / "formal_success_rate.png").stat().st_size > 0
    assert (output / "add_latency_ecdf.pdf").stat().st_size > 0
    assert (output / "search_latency_box.png").stat().st_size > 0

    quality = tmp_path / "quality" / "tridb_gem" / "b1"
    quality.mkdir(parents=True)
    (quality / "quality_summary.json").write_text(
        json.dumps(
            {
                "system": "tridb_gem",
                "overall": {"total": 10, "correct": 8},
            }
        )
    )
    assert plot_latency_quality_frontier(metric_rows, tmp_path / "quality", output)
    assert (output / "latency_quality_frontier.pdf").stat().st_size > 0


def test_answer_quality_pooling_uses_counts_and_emits_noninferiority(tmp_path):
    for system, build_id, total, correct in (
        ("tridb_gem", "b1", 4, 3),
        ("tridb_gem", "b2", 6, 3),
        ("mem0", "b1", 4, 3),
        ("mem0", "b2", 6, 4),
    ):
        output = tmp_path / system / build_id
        output.mkdir(parents=True)
        (output / "quality_summary.json").write_text(
            json.dumps(
                {
                    "system": system,
                    "build_id": build_id,
                    "overall": {
                        "total": total,
                        "evaluated": total,
                        "correct": correct,
                    },
                    "by_category": {},
                    "context_tokens_total": 10 * total,
                    "context_token_records": total,
                    "context_token_coverage": 1.0,
                    "mean_context_tokens": 10,
                }
            )
        )
        (output / "judges.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "request_index": index,
                        "sample_id": f"sample-{index % 2}",
                        "question_id": f"q-{index}",
                        "success": True,
                        "correct": index < correct,
                    }
                )
                + "\n"
                for index in range(total)
            )
        )
    rows, gates = aggregate_answer_quality(
        tmp_path, bootstrap_iterations=100, bootstrap_seed=7
    )
    pooled = next(
        row
        for row in rows
        if row["system"] == "tridb_gem"
        and row["build_id"] == "pooled"
        and row["category"] == "overall"
    )
    assert pooled["score"] == pytest.approx(6 / 10)
    assert pooled["mean_context_tokens"] == pytest.approx(10)
    assert pooled["context_token_coverage"] == pytest.approx(1.0)
    assert len(gates) == 1
    gate = gates[0]
    assert gate["candidate"] == "tridb_gem"
    assert gate["reference"] == "mem0"
    assert gate["candidate_score"] == pytest.approx(6 / 10)
    assert gate["reference_score"] == pytest.approx(7 / 10)
    assert gate["delta"] == pytest.approx(-0.1)
    assert gate["method"] == "paired_cluster_bootstrap"
    assert gate["matched_records"] == 10
    assert gate["clusters"] == 2
    assert gate["bootstrap_iterations"] == 100
    assert gate["bootstrap_seed"] == 7
    assert gate["one_sided_lower_bound"] <= gate["delta"]
    assert gate["passes_noninferiority"] is False


def test_quality_summary_preserves_context_token_coverage() -> None:
    answers = [
        {"success": True, "context_tokens": 10},
        {"success": True, "context_tokens": 20},
        {"success": False, "context_tokens": None},
    ]
    judges = [
        {"success": True, "correct": True, "category": "1"},
        {"success": True, "correct": False, "category": "1"},
        {"success": False, "correct": False, "category": "2"},
    ]

    result = _summary(answers, judges)

    assert result["context_tokens_total"] == 30
    assert result["context_token_records"] == 2
    assert result["context_token_coverage"] == pytest.approx(2 / 3)
    assert result["mean_context_tokens"] == pytest.approx(15)
