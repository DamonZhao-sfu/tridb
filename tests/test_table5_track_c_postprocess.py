from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter

import tools.table5_track_c_completion_v2_postprocess as postprocess
from bench.agent_memory.table5_track_c import report as legacy_report


SYSTEMS = (
    "tridb_gem",
    "mem0",
    "memos",
    "cognee",
    "graphiti_zep_oss_proxy",
)


def test_legacy_report_keeps_proxy_as_fifth_system() -> None:
    rows = [
        {"system": system, "phase": "search", "build_id": "pooled"}
        for system in SYSTEMS
    ]

    assert [row["system"] for row in legacy_report._pooled(rows, "search")] == list(
        SYSTEMS
    )
    assert (
        legacy_report.SYSTEM_LABELS["graphiti_zep_oss_proxy"]
        == "Graphiti (Zep OSS proxy)"
    )


def test_raw_artifact_manifest_scope_covers_full_five_system_matrix() -> None:
    repository = Path(postprocess.__file__).resolve().parents[1]
    protocol_path = (
        repository
        / "bench/agent_memory/table5_track_c/manifests/completion_v2_protocol.json"
    )
    schedule_path = (
        repository
        / "bench/agent_memory/table5_track_c/manifests/completion_v2_schedule.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = repository / protocol["result_root"]

    specs = postprocess._raw_artifact_specs(
        root=root,
        protocol=protocol,
        schedule_path=schedule_path,
        protocol_path=protocol_path,
    )
    counts = Counter(role for role, _ in specs)

    assert len(specs) == 364
    assert counts["formal_request_jsonl"] == 45
    assert counts["warmup_request_jsonl"] == 45
    assert counts["late_outcome_jsonl"] == 45
    assert counts["stage_span_jsonl"] == 45
    assert counts["run_receipt"] == 45
    assert counts["formal_driver_log"] == 45
    assert counts["quality_answers"] == 15
    assert counts["quality_judges"] == 15
    assert counts["quality_summary"] == 15
    assert counts["quality_receipt"] == 15
    assert counts["quality_driver_log"] == 15
    assert counts["conformance_receipt"] == 5
    assert counts["source_receipt"] == 5
    assert counts["model_service_log"] == 2


def test_final_coverage_checks_refuse_partial_delivery() -> None:
    protocol = {
        "systems": list(SYSTEMS),
        "workloads": ["search", "add_native", "add_source_to_searchable"],
        "rounds": ["b1", "b2", "b3"],
        "protocol": {"search_formal": 1_787, "add_formal": 2_000},
        "paper_reference": {"values_ms": {}},
    }

    checks = postprocess._final_coverage_checks(
        protocol,
        metrics=[],
        retrieval_quality=[],
        decomposition=[],
        resources=[],
        time_breakdown=[],
        observable_calls=[],
        operational=[],
        visibility=[],
        answer_quality=[],
        noninferiority=[],
        paper_reference=[],
        source_provenance=[],
    )

    assert set(checks) == set(postprocess.POSTPROCESS_COVERAGE_CHECK_NAMES)
    assert checks["metric_rows"] is False
    assert checks["answer_quality_rows"] is False
    assert not all(checks.values())


def test_observable_call_rows_aggregate_client_boundaries_honestly(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "tridb_gem" / "search" / "b1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_receipt.json").write_text(
        json.dumps(
            {
                "system": "tridb_gem",
                "phase": "search",
                "build_id": "tc5v2_tridb_gem_search_b1",
            }
        ),
        encoding="utf-8",
    )
    observations = [
        {
            "http_client": 1,
            "database_client": 2,
            "opaque_native_api": 0,
            "total": 3,
            "scope": "complete_client_boundaries",
        },
        {
            "http_client": 1,
            "database_client": 0,
            "opaque_native_api": 1,
            "total": 2,
            "scope": "partial_client_boundaries_before_timeout",
        },
    ]
    (run_dir / "formal.jsonl").write_text(
        "".join(
            json.dumps({"observable_call_counts": observation}) + "\n"
            for observation in observations
        ),
        encoding="utf-8",
    )

    rows = postprocess._observable_call_rows(tmp_path / "runs")

    assert rows == [
        {
            "system": "tridb_gem",
            "phase": "search",
            "build_id": "tc5v2_tridb_gem_search_b1",
            "formal_requests": 2,
            "requests_with_call_observation": 2,
            "observation_coverage": 1.0,
            "http_client_boundaries_total": 2,
            "http_client_boundaries_mean_per_request": 1.0,
            "database_client_boundaries_total": 2,
            "database_client_boundaries_mean_per_request": 1.0,
            "opaque_native_api_boundaries_total": 1,
            "opaque_native_api_boundaries_mean_per_request": 0.5,
            "all_observed_boundaries_total": 5,
            "all_observed_boundaries_mean_per_request": 2.5,
            "partial_timeout_observations": 1,
            "exact_internal_http_round_trips": None,
            "exact_internal_database_round_trips": None,
            "interpretation": (
                "client-boundary counts only; compound native internals are unavailable "
                "and are not inferred"
            ),
        }
    ]


def test_postprocess_emits_five_system_delivery_tree_and_context_table(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    root = repository / "results"
    root.mkdir(parents=True)
    goal_audit = repository / "tools/table5_track_c_goal_audit.py"
    goal_audit.parent.mkdir(parents=True)
    goal_audit.write_text("# frozen goal auditor\n", encoding="utf-8")
    required = [
        "hardware_receipt.json",
        "model_receipt.json",
        "tridb_invariant_receipt.json",
        "raw_artifact_manifest.json",
        "aggregates/search_latency.csv",
        "aggregates/add_native_latency.csv",
        "aggregates/add_source_to_searchable_latency.csv",
        "aggregates/quality_gate.csv",
        "aggregates/answer_quality.csv",
        "aggregates/quality_noninferiority.csv",
        "aggregates/time_breakdown.csv",
        "aggregates/observable_call_counts.csv",
        "aggregates/resource_and_build_cost.csv",
        "aggregates/operational_integrity.csv",
        "aggregates/visibility_integrity.csv",
        "aggregates/failures.csv",
        "aggregates/paper_reference_delta.csv",
        "aggregates/source_provenance.csv",
        "figures/search_latency_ecdf.pdf",
        "figures/search_p99_bar.pdf",
        "figures/add_latency_ecdf.pdf",
        "figures/add_p99_bar.pdf",
        "figures/formal_success_rate.pdf",
        "figures/latency_quality_frontier.pdf",
        "REPORT.md",
    ]
    for name in (
        "hardware_receipt.json",
        "model_receipt.json",
        "tridb_invariant_receipt.json",
    ):
        (root / name).write_text("{}\n", encoding="utf-8")
    protocol = {
        "result_root": "results",
        "plan": "haikaidocs/table5_mem0_zep_memos_cognee_tridb_gem_reproduction_plan_2026-08-19.md",
        "systems": list(SYSTEMS),
        "workloads": ["search", "add_native", "add_source_to_searchable"],
        "rounds": ["b1", "b2", "b3"],
        "protocol": {"search_formal": 1_787, "add_formal": 2_000},
        "dataset": {"sha256": "dataset-sha", "conversations": 10},
        "models": {
            "answer_revision": "answer-revision",
            "embedding_revision": "embedding-revision",
            "runtime_versions": {"vllm": "0.15.0"},
        },
        "zep_naming": {
            "decision": "user_approved_local_proxy_2026_08_21",
            "result_label": "Graphiti (Zep OSS proxy)",
        },
        "quality": {
            "non_inferiority_margin_percentage_points": 2.0,
            "non_inferiority_method": "paired_cluster_bootstrap",
            "pairing_keys": ["round", "sample_id", "question_id"],
            "bootstrap_cluster": "sample_id",
            "bootstrap_iterations": 2_000,
            "bootstrap_seed": 20_260_819,
            "one_sided_confidence_level": 0.95,
        },
        "paper_reference": {
            "comparison_boundary": "context only",
            "values_ms": {
                "mem0": {
                    "search": {"mean": 1.0, "p90": 2.0, "p99": 3.0},
                    "add:native": {"mean": 1.0, "p90": 2.0, "p99": 3.0},
                },
                "memos": {
                    "search": {"mean": 1.0, "p90": 2.0, "p99": 3.0},
                    "add:native": {"mean": 1.0, "p90": 2.0, "p99": 3.0},
                },
            },
        },
        "required_deliverables": required,
    }
    protocol_path = repository / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    schedule = {
        "status": "frozen",
        "protocol_receipt": "protocol.json",
        "benchmark_code_sha256": "code-sha",
        "postprocess_script_sha256": postprocess._sha256(
            Path(postprocess.__file__).resolve()
        ),
        "goal_audit_script_sha256": postprocess._sha256(goal_audit),
    }
    schedule_path = repository / "schedule.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

    metrics: list[dict[str, object]] = []
    for system_index, system in enumerate(SYSTEMS, start=1):
        for workload, phase, formal_count in (
            ("search", "search", 1_787),
            ("add_native", "add:native", 2_000),
            (
                "add_source_to_searchable",
                "add:source_to_searchable",
                2_000,
            ),
        ):
            for build_id, row_count in [
                *[
                    (f"tc5v2_{system}_{workload}_{round_name}", formal_count)
                    for round_name in ("b1", "b2", "b3")
                ],
                ("pooled", formal_count * 3),
            ]:
                metrics.append(
                    {
                        "system": system,
                        "phase": phase,
                        "build_id": build_id,
                        "service_mean_ms": 10.0 * system_index,
                        "service_mean_ci_low_ms": 9.0 * system_index,
                        "service_mean_ci_high_ms": 11.0 * system_index,
                        "service_p90_ms": 20.0 * system_index,
                        "service_p90_ci_low_ms": 19.0 * system_index,
                        "service_p90_ci_high_ms": 21.0 * system_index,
                        "service_p99_ms": 30.0 * system_index,
                        "service_p99_ci_low_ms": 29.0 * system_index,
                        "service_p99_ci_high_ms": 31.0 * system_index,
                        "formal_requests": row_count,
                        "successful": row_count - system_index,
                        "failed": system_index,
                        "user_visible_mean_ms": 40.0 * system_index,
                        "user_visible_mean_ci_low_ms": 39.0 * system_index,
                        "user_visible_mean_ci_high_ms": 41.0 * system_index,
                        "user_visible_p90_ms": 50.0 * system_index,
                        "user_visible_p90_ci_low_ms": 49.0 * system_index,
                        "user_visible_p90_ci_high_ms": 51.0 * system_index,
                        "user_visible_p99_ms": 60.0 * system_index,
                        "user_visible_p99_ci_low_ms": 59.0 * system_index,
                        "user_visible_p99_ci_high_ms": 61.0 * system_index,
                        "success_rate": 1.0,
                        "scheduled_qps": 10.0,
                        "admission_qps": 10.0,
                        "actual_qps": 10.0,
                        "admission_lag_mean_ms": 0.1,
                        "admission_lag_p99_ms": 0.2,
                        "queue_p99_ms": 0.3,
                        "error_classes": "{}",
                    }
                )
    answer_quality = []
    for system in SYSTEMS:
        for build_id in [
            *(
                f"tc5v2_{system}_search_{round_name}"
                for round_name in ("b1", "b2", "b3")
            ),
            "pooled",
        ]:
            total = 5_361 if build_id == "pooled" else 1_787
            correct = 4_000 if build_id == "pooled" else 1_333
            answer_quality.append(
                {
                    "system": system,
                    "build_id": build_id,
                    "category": "overall",
                    "total": total,
                    "evaluated": total,
                    "correct": correct,
                    "score": correct / total,
                    "conditional_score": correct / total,
                    "coverage": 1.0,
                    "context_tokens_total": total * 1_000,
                    "context_token_records": total,
                    "context_token_coverage": 1.0,
                    "mean_context_tokens": 1_000.0,
                }
            )
    retrieval_quality = [
        {"system": system, "build_id": build_id, "quality": 1.0}
        for system in SYSTEMS
        for build_id in [
            *(
                f"tc5v2_{system}_search_{round_name}"
                for round_name in ("b1", "b2", "b3")
            ),
            "pooled",
        ]
    ]
    decomposition = [
        {
            "system": row["system"],
            "phase": row["phase"],
            "build_id": row["build_id"],
            "total_requests": row["formal_requests"],
            "successful_requests": row["successful"],
            "failed_requests": row["failed"],
            "commit_observed_count": row["formal_requests"],
            "failed_after_commit_count": row["failed"],
            "visibility_requested_count": row["formal_requests"],
            "visibility_probe_started_count": row["formal_requests"],
            "visibility_probe_count": row["formal_requests"],
            "visibility_probe_pass_rate": (
                (row["formal_requests"] - row["failed"]) / row["formal_requests"]
                if row["phase"] == "add:source_to_searchable"
                else 1.0
            ),
            "visibility_probe_false_count": (
                row["failed"] if row["phase"] == "add:source_to_searchable" else 0
            ),
            "timeout_after_commit_count": 0,
            "timeout_during_visibility_probe_count": 0,
            "visibility_failures": 0,
        }
        for row in metrics
    ]
    resources = [
        {
            "system": row["system"],
            "phase": row["phase"],
            "build_id": row["build_id"],
            "bytes": 1,
        }
        for row in metrics
        if row["build_id"] != "pooled"
    ]
    time_breakdown = [
        {
            "system": row["system"],
            "phase": row["phase"],
            "build_id": row["build_id"],
            "stage": "unattributed",
        }
        for row in metrics
        if row["build_id"] != "pooled"
    ]
    observable_calls = [
        {
            "system": row["system"],
            "phase": row["phase"],
            "build_id": row["build_id"],
            "formal_requests": row["formal_requests"],
            "requests_with_call_observation": row["formal_requests"],
            "observation_coverage": 1.0,
            "http_client_boundaries_total": row["formal_requests"],
            "database_client_boundaries_total": row["formal_requests"],
            "opaque_native_api_boundaries_total": 0,
            "all_observed_boundaries_total": 2 * int(row["formal_requests"]),
            "partial_timeout_observations": 0,
            "exact_internal_http_round_trips": None,
            "exact_internal_database_round_trips": None,
            "interpretation": "client-boundary counts only",
        }
        for row in metrics
        if row["build_id"] != "pooled"
    ]
    monkeypatch.setattr(postprocess, "REPOSITORY", repository)
    monkeypatch.setattr(postprocess, "benchmark_code_sha256", lambda: "code-sha")

    def write_raw_manifest(**kwargs):
        manifest = {
            "schema_version": "table5_track_c_raw_artifact_manifest_v0.1.0",
            "status": "complete",
            "artifact_count": 1,
            "combined_sha256": "b" * 64,
            "artifacts": [{"path": "synthetic", "sha256": "a" * 64}],
        }
        postprocess._write_json(root / "raw_artifact_manifest.json", manifest)
        return manifest

    monkeypatch.setattr(postprocess, "_write_raw_artifact_manifest", write_raw_manifest)
    monkeypatch.setattr(
        postprocess,
        "aggregate",
        lambda _: (metrics, retrieval_quality),
    )
    monkeypatch.setattr(
        postprocess,
        "aggregate_decomposition",
        lambda _: decomposition,
    )
    monkeypatch.setattr(
        postprocess,
        "aggregate_resources",
        lambda _: resources,
    )
    monkeypatch.setattr(
        postprocess,
        "aggregate_answer_quality",
        lambda _, **kwargs: (
            answer_quality,
            [
                {
                    "candidate": "tridb_gem",
                    "reference": reference,
                    "margin": 0.02,
                    "method": "paired_cluster_bootstrap",
                    "pairing_keys": "round,sample_id,question_id",
                    "cluster": "sample_id",
                    "matched_records": 5_361,
                    "clusters": 10,
                    "bootstrap_iterations": 2_000,
                    "bootstrap_seed": 20_260_819,
                    "one_sided_confidence_level": 0.95,
                    "one_sided_lower_bound": 0.0,
                    "passes_noninferiority": True,
                }
                for reference in SYSTEMS
                if reference != "tridb_gem"
            ],
        ),
    )
    monkeypatch.setattr(
        postprocess,
        "_time_breakdown",
        lambda _: time_breakdown,
    )
    monkeypatch.setattr(
        postprocess,
        "_observable_call_rows",
        lambda _: observable_calls,
    )
    monkeypatch.setattr(
        postprocess,
        "_failure_rows",
        lambda _: [
            {
                "system": "tridb_gem",
                "phase": "search",
                "build_id": "b1",
                "timeout": False,
                "error": "example",
                "count": 1,
            }
        ],
    )

    def figure(stem: str):
        def render(*args):
            output = Path(args[-1])
            output.mkdir(parents=True, exist_ok=True)
            (output / f"{stem}.pdf").write_bytes(b"%PDF-synthetic\n")

        return render

    monkeypatch.setattr(postprocess, "plot_search_ecdf", figure("search_latency_ecdf"))
    monkeypatch.setattr(postprocess, "plot_add_ecdf", figure("add_latency_ecdf"))
    monkeypatch.setattr(postprocess, "plot_search_box", figure("search_latency_box"))
    monkeypatch.setattr(postprocess, "plot_search_p99", figure("search_p99_bar"))
    monkeypatch.setattr(postprocess, "plot_add_p99", figure("add_p99_bar"))
    monkeypatch.setattr(postprocess, "plot_success_rate", figure("formal_success_rate"))

    def frontier(*args):
        figure("latency_quality_frontier")(*args)
        return True

    monkeypatch.setattr(postprocess, "plot_latency_quality_frontier", frontier)
    audit_calls = 0

    def audit(_):
        nonlocal audit_calls
        audit_calls += 1
        return {
            "complete_valid_runs": 45,
            "plan_sha256": "plan-sha",
            "prerequisite_complete": True,
            "quality_complete": True,
            "goal_complete": audit_calls == 2,
            "tridb_invariant_gate": {"valid": True},
            "source_receipts": {
                system: {
                    "valid": True,
                    "status": "installed",
                    "path": f"/receipts/{system}.json",
                    "sha256": system + "-sha",
                    "expected_sha256": system + "-sha",
                    "sha256_matches": True,
                    "identity_matches": True,
                    "runtime": {
                        "valid": True,
                        "checks": {"source": True, "environment": True},
                        "observed": {
                            "head": system + "-commit",
                            "tracked_dirty_entries": 0,
                            "package_version": "test-version",
                        },
                        "errors": [],
                    },
                }
                for system in SYSTEMS
            },
        }

    monkeypatch.setattr(postprocess, "audit", audit)

    receipt = postprocess.postprocess(schedule_path)

    assert receipt["status"] == "complete"
    assert set(receipt["coverage_checks"]) == set(
        postprocess.POSTPROCESS_COVERAGE_CHECK_NAMES
    )
    assert all(receipt["coverage_checks"].values())
    assert audit_calls == 2
    assert all((root / relative).is_file() for relative in required)
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    assert "Graphiti (Zep OSS proxy)" in report
    assert "user_approved_local_proxy_2026_08_21" in report
    assert "answer-revision" in report
    assert "embedding-revision" in report
    assert "Mean context tokens" in report
    assert "1000.0" in report
    assert "not an H800 numerical reproduction" in report
    assert "does **not** enforce or claim a benchmark-specific CPU affinity" in report
    assert "timeout-during-probe" in report
    assert "including probes that returned false" in report
    assert "All-admission user-visible latency" in report
    assert "client-visible timeouts" in report
    assert "Operational integrity and failures" in report
    assert "Source-to-searchable visibility outcomes" in report
    assert "aggregates/failures.csv" in report
    assert "explicit `unattributed` row" in report
    assert "Paper Table 5 reference context" in report
    assert "Graphiti is deliberately not paired with the paper Zep row" in report
    assert "Frozen source and live runtime provenance" in report
    assert "TriDB architecture invariant" in report
    assert "bounded Open/Next/Close traversal" in report
    assert "aggregates/observable_call_counts.csv" in report
    assert "one native API call is never converted" in report
    assert "aggregates/source_provenance.csv" in report
    assert "raw_artifact_manifest.json" in report
    assert "Raw evidence manifest: 1 artifacts" in report
    with (root / "aggregates/source_provenance.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        provenance_rows = list(csv.DictReader(source))
    assert len(provenance_rows) == 5
    assert all(row["valid"] == "True" for row in provenance_rows)
    assert {row["system_display_name"] for row in provenance_rows} == set(
        postprocess.LABELS.values()
    )
    with (root / "aggregates/paper_reference_delta.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        reference_rows = list(csv.DictReader(source))
    mem0_search = next(
        row
        for row in reference_rows
        if row["system"] == "mem0" and row["phase"] == "search"
    )
    assert float(mem0_search["paper_mean_ms"]) == 1.0
    assert float(mem0_search["local_mean_ms"]) == 20.0
    assert float(mem0_search["relative_delta_mean"]) == 19.0
    assert mem0_search["system_display_name"] == "Mem0 2.0.18"
    with (root / "aggregates/search_latency.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        search_rows = list(csv.DictReader(source))
    graphiti_rows = [
        row for row in search_rows if row["system"] == "graphiti_zep_oss_proxy"
    ]
    assert graphiti_rows
    assert all(
        row["system_display_name"] == "Graphiti (Zep OSS proxy)"
        for row in graphiti_rows
    )
    with (root / "aggregates/quality_noninferiority.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        noninferiority_rows = list(csv.DictReader(source))
    graphiti_reference = next(
        row
        for row in noninferiority_rows
        if row["reference"] == "graphiti_zep_oss_proxy"
    )
    assert graphiti_reference["reference_display_name"] == ("Graphiti (Zep OSS proxy)")
    assert (
        "| TriDB/GEM | Search | 10.000 | 10.000 | 10.000 | 10.000 | 0.200 | 0.300 | 100.00% | 1 |"
        in report
    )
    assert "| TriDB/GEM | 6000 | 6000 | 6000 | 99.98% | 1 | 1 | 0 |" in report
    assert (
        "| TriDB/GEM | 40.000 [39.000, 41.000] | "
        "50.000 [49.000, 51.000] | 60.000 [59.000, 61.000] | 5361 | 1 |" in report
    )
    assert "clustered by LoCoMo conversation across fresh builds" in report


def test_service_table_renders_zero_success_without_crashing() -> None:
    row = {
        "system": "tridb_gem",
        "phase": "search",
        "build_id": "pooled",
        "service_mean_ms": None,
        "service_p90_ms": None,
        "service_p99_ms": None,
        "success_rate": 0.0,
        "actual_qps": 9.5,
    }

    rendered = postprocess._service_table([row], "search")

    assert "| TriDB/GEM | — | — | — | 0.00% | 9.500 |" in rendered


def test_failure_export_retains_exact_errors_timeouts_and_counts(
    tmp_path: Path,
) -> None:
    run = tmp_path / "tridb_gem" / "search" / "b1"
    run.mkdir(parents=True)
    (run / "run_receipt.json").write_text(
        json.dumps(
            {
                "system": "tridb_gem",
                "phase": "search",
                "build_id": "tc5v2_tridb_gem_search_b1",
            }
        ),
        encoding="utf-8",
    )
    records = [
        {"success": True, "timeout": False, "error": None},
        {"success": False, "timeout": True, "error": "TimeoutError: 60s"},
        {"success": False, "timeout": True, "error": "TimeoutError: 60s"},
        {"success": False, "timeout": False, "error": "visibility failed"},
    ]
    (run / "formal.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    rows = postprocess._failure_rows(tmp_path)

    assert rows == [
        {
            "system": "tridb_gem",
            "phase": "search",
            "build_id": "tc5v2_tridb_gem_search_b1",
            "timeout": True,
            "error": "TimeoutError: 60s",
            "count": 2,
        },
        {
            "system": "tridb_gem",
            "phase": "search",
            "build_id": "tc5v2_tridb_gem_search_b1",
            "timeout": False,
            "error": "visibility failed",
            "count": 1,
        },
    ]
