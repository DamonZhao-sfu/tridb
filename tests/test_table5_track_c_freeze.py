from __future__ import annotations

import hashlib
from pathlib import Path

import tools.table5_track_c_freeze_completion_v2 as freeze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mandol_v7_retry_preserves_partial_and_gates_qps_on_canonical() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (
        repository / "scripts/table5_track_c_mandol_search_completion_gpu1_v7.sh"
    ).read_text(encoding="utf-8")
    manifest = (
        repository
        / "bench/agent_memory/table5_track_c/manifests/search_mandol_qps_1_5_10_v7.json"
    ).read_text(encoding="utf-8")

    assert "sw15v2/mandol/canonical" in driver
    assert "sw15v3/mandol/canonical" in driver
    assert "preserve partial; fresh retry; never append or overwrite" in driver
    assert "CUDA_VISIBLE_DEVICES=1" in driver
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_memory" in driver
    assert driver.count("HF_HUB_OFFLINE=1") == 4
    for revision in (
        "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    ):
        assert revision in driver
    build_at = driver.index('table5_track_c_build_search_snapshot.sh" mandol')
    complete_gate_at = driver.index(
        "canonical snapshot did not pass the completion gate"
    )
    sweep_at = driver.index("bench.agent_memory.table5_track_c.sweep")
    assert build_at < complete_gate_at < sweep_at
    assert '"qps": [1, 5, 10]' in manifest
    assert '"systems": ["mandol"]' in manifest
    assert '"workloads": ["search"]' in manifest


def test_execution_driver_sets_repo_pythonpath_before_python_tools() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (repository / "scripts/table5_track_c_completion_v2_execute.sh").read_text(
        encoding="utf-8"
    )

    export_at = driver.index('export PYTHONPATH="$repo"')
    postprocess_at = driver.index(
        '"$repo/tools/table5_track_c_completion_v2_postprocess.py"'
    )

    assert export_at < postprocess_at


def test_execution_driver_audits_full_protocol_before_model_launch() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (repository / "scripts/table5_track_c_completion_v2_execute.sh").read_text(
        encoding="utf-8"
    )

    audit_hash_at = driver.index("goal auditor does not match its frozen schedule hash")
    protocol_gate_at = driver.index("--protocol >/dev/null")
    source_gate_at = driver.index('--source "$system_name"')
    model_launch_at = driver.index(
        'systemd-run --user --unit="${answer_unit%.service}"'
    )
    assert audit_hash_at < protocol_gate_at < source_gate_at < model_launch_at
    assert "protocol/code gate failed before model launch" in driver
    assert "live source/install identity failed before model launch" in driver


def test_execution_driver_records_resource_control_boundary() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (repository / "scripts/table5_track_c_completion_v2_execute.sh").read_text(
        encoding="utf-8"
    )

    assert "table5_track_c_completion_hardware_receipt_v0.3.0" in driver
    assert "runner_cpu_affinity" in driver
    assert "cpuset.cpus.effective" in driver
    assert "memory.max" in driver
    assert "inherited_value" in driver
    assert driver.count("CUDA_VISIBLE_DEVICES=0") >= 3
    assert '"cpu_affinity_fixed": False' in driver
    assert '"ram_cap_fixed": False' in driver
    assert '"gpu_exclusive_mode": False' in driver
    gpu_gate_at = driver.index("physical GPU 0 already has compute processes")
    model_launch_at = driver.index(
        'systemd-run --user --unit="${answer_unit%.service}"'
    )
    assert gpu_gate_at < model_launch_at
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_memory" in driver
    assert "never terminate external processes" in driver


def test_all_measurement_phases_bind_frozen_offline_model_revisions() -> None:
    repository = Path(__file__).resolve().parents[1]
    execute = (
        repository / "scripts/table5_track_c_completion_v2_execute.sh"
    ).read_text(encoding="utf-8")
    conformance = (
        repository / "scripts/table5_track_c_completion_v2_conformance.sh"
    ).read_text(encoding="utf-8")
    run_one = (
        repository / "scripts/table5_track_c_completion_v2_run_one.sh"
    ).read_text(encoding="utf-8")
    quality = (
        repository / "scripts/table5_track_c_completion_v2_quality_driver.sh"
    ).read_text(encoding="utf-8")

    assert '--revision "$answer_revision"' in execute
    assert '--tokenizer-revision "$answer_revision"' in execute
    assert '--revision "$embedding_revision"' in execute
    assert '--tokenizer-revision "$embedding_revision"' in execute
    assert execute.count("HF_HUB_OFFLINE=1") >= 3
    assert "table5_track_c_completion_model_receipt_v0.2.0" in execute
    assert 'model_runtime_json=$("$vllm_python"' in execute
    assert 'export TRACKC_MODEL_RUNTIME_JSON="$model_runtime_json"' in execute
    assert 'json.loads(os.environ["TRACKC_MODEL_RUNTIME_JSON"])' in execute
    for phase in (conformance, run_one):
        assert "models.answer_revision" in phase
        assert "models.embedding_revision" in phase
        assert "HF_HUB_OFFLINE=1" in phase
        assert "--tokenizer-revision" in phase
    assert "models.answer_revision" in quality
    assert "frozen offline revision" in quality


def test_execution_driver_gates_formal_runs_on_tridb_architecture_receipt() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (repository / "scripts/table5_track_c_completion_v2_execute.sh").read_text(
        encoding="utf-8"
    )

    conformance_at = driver.index(
        '"$repo/scripts/table5_track_c_completion_v2_conformance.sh"'
    )
    invariant_at = driver.index("bench.agent_memory.table5_track_c.tridb_invariants")
    formal_at = driver.index(
        '"$repo/scripts/table5_track_c_completion_v2_formal_driver.sh"'
    )

    assert conformance_at < invariant_at < formal_at
    assert 'tridb_invariant_receipt.json"' in driver


def test_execution_driver_stops_models_before_raw_artifact_hashing() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (repository / "scripts/table5_track_c_completion_v2_execute.sh").read_text(
        encoding="utf-8"
    )

    quality_at = driver.index(
        '"$repo/scripts/table5_track_c_completion_v2_quality_driver.sh"'
    )
    blocking_stop_at = driver.index(
        'systemctl --user stop "$embedding_unit" "$answer_unit"'
    )
    inactive_gate_at = driver.index(
        'systemctl --user --quiet is-active "$embedding_unit"', blocking_stop_at
    )
    postprocess_at = driver.index(
        '"$repo/tools/table5_track_c_completion_v2_postprocess.py"'
    )

    assert quality_at < blocking_stop_at < inactive_gate_at < postprocess_at
    assert "controlled-model service remained active before postprocess" in driver


def test_quality_driver_preserves_per_point_stdout_and_stderr_logs() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (
        repository / "scripts/table5_track_c_completion_v2_quality_driver.sh"
    ).read_text(encoding="utf-8")

    assert 'log="$result_root/logs/quality_${round_name}_${system_name}.log"' in driver
    assert '>>"$log" 2>&1' in driver
    assert "starting quality $system_name $round_name" in driver
    assert "completed quality $system_name $round_name" in driver


def test_formal_point_fail_closes_on_live_source_environment_drift() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (repository / "scripts/table5_track_c_completion_v2_run_one.sh").read_text(
        encoding="utf-8"
    )

    source_receipt_hash_at = driver.index(
        'if [[ $source_receipt_sha != "$expected_source_receipt_sha" ]]'
    )
    live_source_at = driver.index('--source "$system_name"')
    conformance_at = driver.index('conformance="$result_root/conformance/')

    assert source_receipt_hash_at < live_source_at < conformance_at
    assert "live source/dependency/runtime identity drifted" in driver


def test_final_conformance_fail_closes_on_live_source_environment_drift() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (
        repository / "scripts/table5_track_c_completion_v2_conformance.sh"
    ).read_text(encoding="utf-8")

    assert '--source "$system_name"' in driver
    assert "live source/dependency/runtime identity drifted" in driver
    assert "memos_backend_script_sha256:$memos_backend_sha" in driver
    assert "graphiti_backend_script_sha256:$graphiti_backend_sha" in driver


def test_formal_graphiti_paths_bind_and_use_the_per_group_fifo_guard() -> None:
    repository = Path(__file__).resolve().parents[1]
    conformance = (
        repository / "scripts/table5_track_c_completion_v2_conformance.sh"
    ).read_text(encoding="utf-8")
    run_one = (
        repository / "scripts/table5_track_c_completion_v2_run_one.sh"
    ).read_text(encoding="utf-8")
    standalone = (
        repository / "scripts/table5_track_c_graphiti_conformance_gpu0.sh"
    ).read_text(encoding="utf-8")

    assert "graphiti_track_c_formal.hashing" in conformance
    assert "graphiti_track_c_formal.hashing" in run_one
    assert conformance.count("experiments.graphiti_track_c_formal.conformance") == 2
    assert standalone.count("experiments.graphiti_track_c_formal.conformance") == 2
    assert "experiments.graphiti_track_c_formal.cli" in run_one
    assert "experiments.graphiti_track_c.cli" not in run_one


def test_formal_point_binds_backend_launchers_to_frozen_schedule() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (repository / "scripts/table5_track_c_completion_v2_run_one.sh").read_text(
        encoding="utf-8"
    )

    assert "assert_schedule_hash memos_backend_script_sha256" in driver
    assert "assert_schedule_hash graphiti_backend_script_sha256" in driver
    assert 'scripts/table5_track_c_memos_neo4j.sh" | awk' in driver
    assert 'scripts/table5_track_c_graphiti_neo4j.sh" | awk' in driver


def test_final_conformance_is_semantically_recomputed_before_formal() -> None:
    repository = Path(__file__).resolve().parents[1]
    conformance = (
        repository / "scripts/table5_track_c_completion_v2_conformance.sh"
    ).read_text(encoding="utf-8")
    run_one = (
        repository / "scripts/table5_track_c_completion_v2_run_one.sh"
    ).read_text(encoding="utf-8")

    assert '--conformance "$system_name"' in conformance
    assert "conformance did not produce a frozen valid receipt" in conformance
    assert "trap cleanup_backend EXIT" in conformance
    assert '--conformance "$system_name"' in run_one
    assert "failed independent semantic audit" in run_one


def test_formal_and_quality_drivers_recheck_frozen_schedule_semantics() -> None:
    repository = Path(__file__).resolve().parents[1]
    formal = (
        repository / "scripts/table5_track_c_completion_v2_formal_driver.sh"
    ).read_text(encoding="utf-8")
    quality = (
        repository / "scripts/table5_track_c_completion_v2_quality_driver.sh"
    ).read_text(encoding="utf-8")

    for driver in (formal, quality):
        assert "--protocol >/dev/null" in driver
        assert "invalid frozen protocol/schedule" in driver


def test_after_preflight_driver_is_fail_closed_and_waits_for_every_queue() -> None:
    repository = Path(__file__).resolve().parents[1]
    driver = (
        repository / "scripts/table5_track_c_completion_v2_after_preflight.sh"
    ).read_text(encoding="utf-8")

    for unit in (
        "tridb-table5-mandol-search-completion-gpu1-v6.service",
        "tridb-table5-graphiti-install-after-current.service",
        "tridb-table5-graphiti-conformance-after-install.service",
        "tridb-table5-graphiti-build-after-conformance.service",
        "tridb-table5-graphiti-search-after-canonical.service",
        "tridb-table5-graphiti-add-after-canonical.service",
    ):
        assert unit in driver
    freeze_at = driver.index('"$freeze_tool" --schedule "$schedule" --apply')
    credential_at = driver.index("credential file is absent or not mode 0600")
    execute_at = driver.index('exec "$execute_driver"')
    assert credential_at < freeze_at < execute_at
    assert "!= frozen" in driver
    assert "rm " not in driver


def test_graphiti_preflight_proves_a_distinct_persistent_runtime() -> None:
    adapter_sha = freeze._graphiti_adapter_sha256()
    schema_gate = {
        "adapter_sha256": adapter_sha,
        "graphiti_source": {"commit": freeze.GRAPHITI_COMMIT},
    }
    counts = {"nodes": 4, "relationships": 2}
    groups = {"a": {"nodes": 2, "relationships": 1}}
    search_results = [
        {
            "sample_id": "a",
            "result": {
                "result_count": 1,
                "hit_ids": ["a-1"],
                "contexts": ["Asteria works with Cedar Labs"],
                "context_tokens": None,
                "empty": False,
            },
        },
        {
            "sample_id": "b",
            "result": {
                "result_count": 1,
                "hit_ids": ["b-1"],
                "contexts": ["Borealis works with Maple Works"],
                "context_tokens": None,
                "empty": False,
            },
        },
    ]
    additions = [
        {
            "created_memory_count": 1,
            "created_node_count": None,
            "created_edge_count": None,
            "creation_counts_available": True,
            "creation_count_source": "graphiti_episode_created_entities_resolved",
            "write_concurrency_policy": "per_group_fifo_cross_group_parallel",
            "write_group_id": "conformance_a" if index < 2 else "conformance_b",
            "write_group_fifo_ticket": index % 2,
            "write_group_fifo_wait_ms": 0.0,
            "write_concurrency_provenance": "official MCP QueueService",
        }
        for index in range(4)
    ]
    receipt = {
        "schema_version": "graphiti_track_c_conformance_v0.3.0",
        "status": "complete",
        "display_label": "Graphiti (Zep OSS proxy)",
        "interpretation_boundary": "not production Zep",
        "write_phase": {
            "schema_gate": schema_gate,
            "counts": counts,
            "group_counts": groups,
            "events": 4,
            "additions": additions,
            "search_results": search_results,
            "neo4j_runtime": {
                "systemd_invocation_id": "a" * 32,
                "main_pid": 101,
            },
        },
        "read_phase": {
            "schema_gate": schema_gate,
            "counts": counts,
            "group_counts": groups,
            "search_results": search_results,
            "neo4j_runtime": {
                "systemd_invocation_id": "b" * 32,
                "main_pid": 202,
            },
        },
    }

    checks = freeze._graphiti_preflight_checks(receipt)

    assert all(checks.values())
    receipt["read_phase"]["neo4j_runtime"]["systemd_invocation_id"] = "a" * 32
    assert freeze._graphiti_preflight_checks(receipt)["runtime_restart"] is False


def test_frozen_payloads_bind_protocol_code_scripts_and_source_receipts(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    for relative in freeze.HASHED_SCRIPTS.values():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    plan_relative = (
        "haikaidocs/"
        "table5_mem0_zep_memos_cognee_tridb_gem_reproduction_plan_2026-08-19.md"
    )
    plan_path = repository / plan_relative
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# frozen plan\n", encoding="utf-8")
    source_receipts = {
        system: f"receipts/{system}.json"
        for system in (
            "tridb_gem",
            "mem0",
            "memos",
            "cognee",
            "graphiti_zep_oss_proxy",
        )
    }
    for system, relative in source_receipts.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"system":"{system}"}}\n', encoding="utf-8")

    monkeypatch.setattr(freeze, "REPOSITORY", repository)
    monkeypatch.setattr(freeze, "benchmark_code_sha256", lambda: "a" * 64)
    monkeypatch.setattr(freeze, "_graphiti_adapter_sha256", lambda: "b" * 64)
    schedule = {
        "status": "draft_pending_graphiti_conformance_and_code_freeze",
        "protocol_receipt_sha256": None,
        "benchmark_code_sha256": None,
        "graphiti_adapter_sha256": None,
        "plan_sha256": None,
        **{field: None for field in freeze.HASHED_SCRIPTS},
        "source_receipt_sha256": {system: None for system in source_receipts},
    }
    protocol = {
        "status": "draft_pending_graphiti_conformance_and_code_freeze",
        "plan": plan_relative,
        "source_receipts": source_receipts,
    }

    frozen_schedule, frozen_protocol = freeze._frozen_payloads(
        schedule, protocol, "2026-08-21T17:30:00+00:00"
    )

    assert frozen_schedule["status"] == "frozen"
    assert frozen_protocol["status"] == "frozen"
    assert frozen_schedule["benchmark_code_sha256"] == "a" * 64
    assert frozen_schedule["graphiti_adapter_sha256"] == "b" * 64
    assert frozen_schedule["plan_sha256"] == _sha256(plan_path)
    assert frozen_schedule["protocol_receipt_sha256"] == freeze._bytes_sha256(
        freeze._json_bytes(frozen_protocol)
    )
    assert frozen_schedule["source_receipt_sha256"] == {
        system: _sha256(repository / relative)
        for system, relative in source_receipts.items()
    }
    assert all(
        frozen_schedule[field] == _sha256(repository / relative)
        for field, relative in freeze.HASHED_SCRIPTS.items()
    )
