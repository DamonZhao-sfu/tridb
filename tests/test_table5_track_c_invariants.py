from __future__ import annotations

from pathlib import Path

from bench.agent_memory.table5_track_c.tridb_invariants import (
    INVARIANT_CHECK_NAMES,
    _conformance_checks,
    _static_checks,
)


def _conformance() -> dict[str, object]:
    probes = {
        "mode": "fused",
        "route": "topic",
        "hnsw_iterative_scan": "relaxed_order",
        "candidates_examined": 61,
        "graph_examined": 7,
        "termination_reason": "term_cond",
    }
    stats = {
        "units": 60,
        "edge_metadata_rows": 112,
        "native_vertices": 60,
        "native_visible_edges": 112,
    }
    return {
        "status": "passed",
        "system": "tridb_gem",
        "execution": {
            "benchmark_code_sha256": "a" * 64,
            "formal_schedule_sha256": "b" * 64,
            "protocol_receipt_sha256": "c" * 64,
        },
        "build": {
            "events": 58,
            "units_created": 58,
            "edges_created": 110,
            "rejected": [],
        },
        "build_finalize": {"units": 58, "visible_edges": 110},
        "stats_before_restart": dict(stats),
        "stats_after_restart": dict(stats),
        "restart_visible": True,
        "scope_isolated": True,
        "searches": [{"probes": dict(probes)} for _ in range(5)],
    }


def test_tridb_static_invariants_cover_native_streaming_path() -> None:
    repository = Path(__file__).resolve().parents[1]
    checks, hashes = _static_checks(repository)

    assert checks
    assert all(checks.values())
    assert len(hashes) == 6
    assert all(isinstance(value, str) and len(value) == 64 for value in hashes.values())
    assert set(checks).issubset(INVARIANT_CHECK_NAMES)


def test_conformance_invariants_require_fused_work_and_frozen_hashes() -> None:
    schedule = {
        "benchmark_code_sha256": "a" * 64,
        "protocol_receipt_sha256": "c" * 64,
    }
    receipt = _conformance()

    assert all(_conformance_checks(receipt, schedule, "b" * 64).values())

    receipt["searches"][0]["probes"]["mode"] = "vector"
    checks = _conformance_checks(receipt, schedule, "b" * 64)
    assert checks["conformance_five_fused_topic_searches"] is False


def test_conformance_invariants_reject_hidden_or_unobserved_termination() -> None:
    schedule = {
        "benchmark_code_sha256": "a" * 64,
        "protocol_receipt_sha256": "c" * 64,
    }
    receipt = _conformance()
    receipt["searches"][2]["probes"]["termination_reason"] = None
    receipt["searches"][3]["probes"]["candidates_examined"] = 0

    checks = _conformance_checks(receipt, schedule, "b" * 64)

    assert checks["conformance_termination_disclosed"] is False
    assert checks["conformance_stream_work_observed"] is False
