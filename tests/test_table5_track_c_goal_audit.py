from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tools.table5_track_c_goal_audit as goal_audit
from bench.agent_memory.table5_track_c.protocol import (
    add_receipt_contract,
    search_receipt_contract,
)
from bench.agent_memory.table5_track_c.tridb_invariants import INVARIANT_CHECK_NAMES
from tools.table5_track_c_goal_audit import (
    _add_workload_semantics,
    _completion_hardware_receipt_valid,
    _completion_model_receipt_valid,
    _dataset_manifest_observation,
    _generic_conformance_checks,
    _graphiti_conformance_checks,
    _fresh_state_launcher_checks,
    _input_keys_exact,
    _late_outcome_checks,
    _postprocess_gate,
    _quality_point,
    _raw_artifact_manifest_gate,
    _request_checks,
    _path_entry_is_within,
    _resolved_path_is_within,
    _source_identity_matches,
    _tridb_invariant_gate,
    audit_conformance_receipt,
    audit_source_receipt,
    audit_run,
)


def test_resolved_path_containment_accepts_equivalent_symlink_spelling(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    executable = real_root / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("python\n", encoding="utf-8")
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    assert _resolved_path_is_within(alias_root / "venv/bin/python", real_root / "venv")
    assert _resolved_path_is_within(executable, alias_root / "venv")
    assert not _resolved_path_is_within(executable, real_root / "source")


def test_path_entry_containment_does_not_follow_venv_python_leaf_symlink(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    venv = real_root / "venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    system_python = tmp_path / "system/python3"
    system_python.parent.mkdir()
    system_python.write_text("python\n", encoding="utf-8")
    (venv_bin / "python").symlink_to(system_python)
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    assert _path_entry_is_within(alias_root / "venv/bin/python", venv)
    assert not _resolved_path_is_within(alias_root / "venv/bin/python", venv)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _observable_calls(*, timed_out: bool = False) -> dict[str, object]:
    return {
        "http_client": 0,
        "database_client": 0,
        "opaque_native_api": 1,
        "total": 1,
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


def _schedule() -> dict[str, object]:
    return {
        "status": "frozen",
        "benchmark_code_sha256": "a" * 64,
        "graphiti_adapter_sha256": "b" * 64,
        "run_one_script_sha256": "c" * 64,
        "formal_driver_script_sha256": "d" * 64,
        "conformance_driver_script_sha256": "e" * 64,
        "quality_driver_script_sha256": "2" * 64,
        "postprocess_script_sha256": "3" * 64,
        "execution_driver_script_sha256": "4" * 64,
        "goal_audit_script_sha256": "6" * 64,
        "protocol_receipt_sha256": "f" * 64,
        "_schedule_sha256": "1" * 64,
    }


def _protocol() -> dict[str, object]:
    return {
        "status": "frozen",
        "dataset": {"sha256": "dataset-sha"},
        "protocol": {
            "qps": 10.0,
            "scheduled_interval_ns": 100_000_000,
            "admission_qps_relative_tolerance": 0.01,
            "admission_lag_p99_max_ms": 100.0,
            "timeout_seconds": 60.0,
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


def _valid_model_receipt() -> dict[str, Any]:
    answer_revision = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
    embedding_revision = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    offline_environment = (
        "CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "HF_HOME=/localhome/hza214/.cache/huggingface"
    )
    return {
        "schema_version": "table5_track_c_completion_model_receipt_v0.2.0",
        "status": "passed",
        "schedule_sha256": "1" * 64,
        "protocol_sha256": "f" * 64,
        "answer_endpoint": "http://127.0.0.1:8000/v1",
        "embedding_endpoint": "http://127.0.0.1:8001/v1",
        "answer_thinking": False,
        "answer_revision": answer_revision,
        "embedding_revision": embedding_revision,
        "runtime_versions": {
            "vllm": "0.15.0",
            "transformers": "4.57.0",
            "huggingface_hub": "0.36.2",
            "torch": "2.9.1",
        },
        "local_snapshots": {
            "answer": {"revision": answer_revision, "exists": True},
            "embedding": {"revision": embedding_revision, "exists": True},
        },
        "model_service_commands": {
            "answer": f"--revision {answer_revision} --tokenizer-revision {answer_revision}",
            "embedding": f"--revision {embedding_revision} --tokenizer-revision {embedding_revision}",
        },
        "model_service_environments": {
            "answer": offline_environment,
            "embedding": offline_environment,
        },
        "answer": {
            "data": [
                {
                    "id": "Qwen/Qwen3-32B",
                    "root": "Qwen/Qwen3-32B-FP8",
                    "max_model_len": 32768,
                }
            ]
        },
        "embedding": {
            "data": [
                {
                    "id": "Qwen/Qwen3-Embedding-0.6B",
                    "root": "Qwen/Qwen3-Embedding-0.6B",
                    "max_model_len": 8192,
                }
            ]
        },
        "embedding_dimension": 1024,
        "embedding_vector_sha256": "9" * 64,
        "chat_content": "TOKEN_OK",
        "chat_reasoning_content": None,
        "checks": {
            "answer_identity_exact": True,
            "embedding_identity_exact": True,
            "embedding_dimension_1024": True,
            "thinking_disabled": True,
            "deterministic_chat_probe": True,
            "answer_revision_exact": True,
            "embedding_revision_exact": True,
            "runtime_versions_exact": True,
            "offline_launch_exact": True,
            "local_snapshots_present": True,
        },
    }


def test_completion_model_receipt_requires_observed_probe_evidence() -> None:
    schedule = _schedule()
    receipt = _valid_model_receipt()
    assert _completion_model_receipt_valid(receipt, schedule)

    receipt["embedding_dimension"] = 768
    assert not _completion_model_receipt_valid(receipt, schedule)
    receipt["embedding_dimension"] = 1024

    receipt["answer_revision"] = "moving-main"
    assert not _completion_model_receipt_valid(receipt, schedule)
    receipt["answer_revision"] = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"

    del receipt["checks"]["thinking_disabled"]
    assert not _completion_model_receipt_valid(receipt, schedule)


def test_completion_hardware_receipt_requires_bound_observed_inventory() -> None:
    schedule = _schedule()
    receipt = {
        "schema_version": "table5_track_c_completion_hardware_receipt_v0.3.0",
        "status": "complete",
        "schedule_sha256": "1" * 64,
        "protocol_sha256": "f" * 64,
        "platform": "Linux-test",
        "machine": "x86_64",
        "cpu": '{"lscpu": "observed"}',
        "memory": "MemTotal: 1 kB",
        "gpus": "0, GPU, uuid, memory, driver",
        "runner_cpu_affinity": [0, 1],
        "runner_cgroup": {
            "membership": "0::/user.slice/test.service",
            "path": "/sys/fs/cgroup/user.slice/test.service",
            "cpuset_cpus_effective": "0-39",
            "cpuset_cpus_effective_source": "/sys/fs/cgroup/cpuset.cpus.effective",
            "memory_max": "max",
            "memory_max_source": ("/sys/fs/cgroup/user.slice/test.service/memory.max"),
        },
        "gpu_launch_gate": {
            "target_physical_index": 0,
            "target_uuid": "GPU-test",
            "checked_at": "2026-08-21T00:00:00Z",
            "prelaunch_compute_processes": [],
            "prelaunch_empty": True,
            "policy": (
                "refuse model launch when physical GPU 0 already has any compute "
                "process; never terminate external processes"
            ),
        },
        "model_service_bindings": {
            "answer_unit": "tridb-table5-completion-v2-answer.service",
            "answer_environment": (
                "CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                "HF_HOME=/localhome/hza214/.cache/huggingface"
            ),
            "answer_command": (
                "--revision aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df "
                "--tokenizer-revision aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            ),
            "embedding_unit": "tridb-table5-completion-v2-embedding.service",
            "embedding_environment": (
                "CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                "HF_HOME=/localhome/hza214/.cache/huggingface"
            ),
            "embedding_command": (
                "--revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 "
                "--tokenizer-revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
            ),
        },
        "resource_control_claim": {
            "cpu_affinity_fixed": False,
            "ram_cap_fixed": False,
            "model_gpu_binding_fixed": True,
            "gpu_exclusive_mode": False,
            "interpretation": "observed but not benchmark-capped",
        },
        "paper_h800_match": False,
        "gx10_signoff": False,
        "claim": "same-host controlled comparison only",
    }
    assert _completion_hardware_receipt_valid(receipt, schedule)

    receipt["gpus"] = ""
    assert not _completion_hardware_receipt_valid(receipt, schedule)
    receipt["gpus"] = "0, GPU, uuid, memory, driver"
    receipt["paper_h800_match"] = True
    assert not _completion_hardware_receipt_valid(receipt, schedule)
    receipt["paper_h800_match"] = False
    receipt["resource_control_claim"]["ram_cap_fixed"] = True
    assert not _completion_hardware_receipt_valid(receipt, schedule)
    receipt["resource_control_claim"]["ram_cap_fixed"] = False
    receipt["gpu_launch_gate"]["prelaunch_compute_processes"] = [
        {"pid": 123, "process_name": "external"}
    ]
    assert not _completion_hardware_receipt_valid(receipt, schedule)


def test_tridb_invariant_gate_recomputes_all_bound_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    root = repository / "results"
    conformance = root / "conformance/tridb_gem.json"
    conformance.parent.mkdir(parents=True)
    conformance.write_text('{"status":"passed"}\n', encoding="utf-8")
    source_file = repository / "src/native.c"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("native\n", encoding="utf-8")
    source_manifest = repository / "receipts/tridb.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text('{"system":"tridb_gem"}\n', encoding="utf-8")
    protocol = {
        "source_receipts": {"tridb_gem": "receipts/tridb.json"},
    }
    protocol_path = repository / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    schedule_path = repository / "schedule.json"
    schedule_path.write_text('{"status":"frozen"}\n', encoding="utf-8")
    schedule = {
        "benchmark_code_sha256": "a" * 64,
        "source_receipt_sha256": {"tridb_gem": goal_audit._sha256(source_manifest)},
    }
    receipt = {
        "schema_version": "table5_track_c_tridb_invariant_v0.1.0",
        "status": "passed",
        "schedule": str(schedule_path.resolve()),
        "schedule_sha256": goal_audit._sha256(schedule_path),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": goal_audit._sha256(protocol_path),
        "benchmark_code_sha256": "a" * 64,
        "conformance_receipt": str(conformance.resolve()),
        "conformance_receipt_sha256": goal_audit._sha256(conformance),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": goal_audit._sha256(source_manifest),
        "source_files_sha256": {"src/native.c": goal_audit._sha256(source_file)},
        "source_runtime_audit": {"valid": True, "returncode": 0},
        "checks": {name: True for name in INVARIANT_CHECK_NAMES},
        "claims": {
            "tr1_open_next_close": True,
            "native_graph": True,
            "same_postgres_process": True,
            "one_postgres_wal": True,
            "full_intermediate_materialization": False,
            "paper_h800_match": False,
            "gx10_signoff": False,
        },
        "interpretation": (
            "This is not an H800 numerical reproduction and not a GX10 build/sign-off."
        ),
    }
    invariant = root / "tridb_invariant_receipt.json"
    invariant.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(goal_audit, "REPOSITORY", repository)
    monkeypatch.setattr(goal_audit, "benchmark_code_sha256", lambda: "a" * 64)

    result = _tridb_invariant_gate(root, schedule_path, schedule, protocol_path)

    assert result["valid"] is True
    source_file.write_text("drift\n", encoding="utf-8")
    assert not _tridb_invariant_gate(root, schedule_path, schedule, protocol_path)[
        "valid"
    ]


def test_source_receipt_requires_live_runtime_identity(
    tmp_path: Path, monkeypatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = {
        "schema_version": "table5_track_c_system_manifest_v1",
        "system": "mem0",
        "status": "installed_and_conformant",
        "package_version": "2.0.18",
        "source_commit": "a" * 40,
    }
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    monkeypatch.setattr(goal_audit, "REPOSITORY", tmp_path)
    schedule = {"source_receipt_sha256": {"mem0": goal_audit._sha256(receipt_path)}}
    protocol = {
        "source_receipts": {"mem0": "receipt.json"},
        "source_identity": {
            "mem0": {
                "schema_version": "table5_track_c_system_manifest_v1",
                "status": "installed_and_conformant",
                "package_version": "2.0.18",
                "source_commit": "a" * 40,
            }
        },
    }
    monkeypatch.setattr(
        goal_audit,
        "_source_runtime_observation",
        lambda _receipt, _system: {
            "checks": {"environment_freeze_exact": False},
            "observed": {},
            "errors": [],
            "valid": False,
        },
    )
    assert audit_source_receipt(schedule, protocol, "mem0")["valid"] is False

    monkeypatch.setattr(
        goal_audit,
        "_source_runtime_observation",
        lambda _receipt, _system: {
            "checks": {"environment_freeze_exact": True},
            "observed": {},
            "errors": [],
            "valid": True,
        },
    )
    assert audit_source_receipt(schedule, protocol, "mem0")["valid"] is True


def test_missing_goal_point_is_explicit(tmp_path: Path) -> None:
    result = audit_run(_schedule(), _protocol(), tmp_path, "tridb_gem", "search", "b1")
    assert result["status"] == "MISSING"
    assert result["valid"] is False


def test_fresh_state_launcher_receipt_binds_each_system_resource_identity() -> None:
    policy = _protocol()["protocol"]["fresh_state_policy"]
    workload = "add_source_to_searchable"
    round_name = "b1"
    cases = {
        "tridb_gem": (
            "postgres_database",
            "tc5v2_tridb_gem_source_b1",
            "database_owner:hza214",
        ),
        "mem0": (
            "postgres_database_and_filesystem",
            "tc5v2_mem0_source_b1",
            "/localhome/hza214/agent-memory-table5/volumes/completion_v2/"
            "mem0/tc5v2_mem0_add_source_to_searchable_b1",
        ),
        "cognee": (
            "postgres_database_and_dataset_namespace",
            "tc5v2_cognee_source_b1",
            "dataset_prefix:tc5v2_cognee_add_source_to_searchable_b1",
        ),
        "memos": (
            "filesystem_and_neo4j_state",
            "/localhome/hza214/agent-memory-table5/volumes/completion_v2/"
            "memos/tc5v2_memos_add_source_to_searchable_b1",
            "/localhome/hza214/agent-memory-table5/volumes/memos_neo4j/"
            "tc5v2_memos_add_source_to_searchable_b1",
        ),
        "graphiti_zep_oss_proxy": (
            "neo4j_state_directory",
            "/localhome/hza214/agent-memory-table5/volumes/completion_v2/"
            "graphiti_neo4j/tc5v2_graphiti_zep_oss_proxy_"
            "add_source_to_searchable_b1",
            "systemd_unit:tridb-table5-graphiti-completion-v2-neo4j.service",
        ),
    }
    for system, (kind, primary, secondary) in cases.items():
        build_id = f"tc5v2_{system}_{workload}_{round_name}"
        receipt = {
            "execution": {
                "fresh_state_launcher": {
                    "schema_version": ("table5_track_c_fresh_state_launcher_v0.1.0"),
                    "policy": policy,
                    "build_id": build_id,
                    "kind": kind,
                    "primary_identity": primary,
                    "secondary_identity": secondary,
                    "preexisting_check_passed": True,
                }
            }
        }
        checks = _fresh_state_launcher_checks(
            receipt,
            _protocol(),
            system=system,
            workload=workload,
            round_name=round_name,
        )
        assert all(checks.values()), (system, checks)
        receipt["execution"]["fresh_state_launcher"]["primary_identity"] += "-old"
        assert not _fresh_state_launcher_checks(
            receipt,
            _protocol(),
            system=system,
            workload=workload,
            round_name=round_name,
        )["primary_identity"]


def test_goal_audit_requires_explicit_add_progress_even_on_timeout() -> None:
    record = {
        "success": False,
        "timeout": True,
        "error": "TimeoutError: exceeded 60.000s",
        "receipt": {
            "commit_observed": False,
            "committed_at_ns": None,
            "visibility_requested": True,
            "visibility_probe_started": False,
            "visibility_probe": None,
            "visibility_error": None,
            "searchable_at_ns": None,
            "created_memory_count": None,
            "created_node_count": None,
            "created_edge_count": None,
            "creation_counts_available": False,
            "creation_count_source": "native_api_unavailable",
        },
    }
    assert _add_workload_semantics("add_source_to_searchable", [record])
    del record["receipt"]["commit_observed"]
    assert not _add_workload_semantics("add_source_to_searchable", [record])


def test_goal_audit_binds_each_timeout_to_one_late_worker_outcome(
    tmp_path: Path,
) -> None:
    path = tmp_path / "late_outcomes.jsonl"
    request = {
        "system": "cognee",
        "build_id": "b1",
        "phase": "formal_add_native",
        "request_index": 7,
        "trace_id": "a" * 32,
        "sample_id": "sample-1",
        "event_id": "event-7",
        "completed_at_ns": 10_000,
        "timeout": True,
    }
    late = {
        "schema_version": "table5_track_c_late_outcome_v0.1.0",
        "system": "cognee",
        "build_id": "b1",
        "phase": "formal_add_native",
        "request_index": 7,
        "trace_id": "a" * 32,
        "root_span_id": "b" * 16,
        "sample_id": "sample-1",
        "event_id": "event-7",
        "client_timed_out_at_ns": 10_000,
        "worker_completed_at_ns": 20_000,
        "post_timeout_work_ms": 0.01,
        "final_success": True,
        "final_error": None,
        "receipt": {"committed_at_ns": 15_000},
    }
    _write_jsonl(path, [late])
    receipt = {
        "late_outcome_summary": {
            "schema_version": "table5_track_c_late_outcome_summary_v0.1.0",
            "path": str(path.resolve()),
            "sha256": goal_audit._sha256(path),
            "records": 1,
            "by_phase": {"formal_add_native": 1},
            "final_success": 1,
            "final_failed": 0,
            "post_timeout_work_ms": {"total": 0.01, "max": 0.01},
        }
    }

    checks = _late_outcome_checks(
        goal_audit._jsonl(path),
        path=path,
        request_records=[request],
        receipt=receipt,
        expected_system="cognee",
        expected_build="b1",
    )

    assert all(checks.values())
    late["request_index"] = 8
    _write_jsonl(path, [late])
    assert not _late_outcome_checks(
        goal_audit._jsonl(path),
        path=path,
        request_records=[request],
        receipt=receipt,
        expected_system="cognee",
        expected_build="b1",
    )["one_per_timeout"]


def test_raw_artifact_manifest_gate_rederives_every_file_hash(
    tmp_path: Path, monkeypatch
) -> None:
    import tools.table5_track_c_completion_v2_postprocess as postprocess

    repository = tmp_path / "repo"
    root = repository / "results"
    schedule_path = repository / "schedule.json"
    protocol_path = repository / "protocol.json"
    plan = repository / "plan.md"
    dataset = repository / "locomo.json"
    source = repository / "source.json"
    for path, content in (
        (schedule_path, "schedule\n"),
        (protocol_path, "protocol\n"),
        (plan, "plan\n"),
        (dataset, "dataset\n"),
        (source, "source\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    protocol = {
        "systems": ["fake"],
        "workloads": ["search"],
        "rounds": ["b1"],
        "source_receipts": {"fake": "source.json"},
        "plan": "plan.md",
        "dataset": {"path": str(dataset)},
    }
    run = root / "runs/fake/search/b1"
    quality = root / "quality/fake/b1"
    for directory, names in (
        (
            run,
            (
                "formal.jsonl",
                "warmup.jsonl",
                "late_outcomes.jsonl",
                "spans.jsonl",
                "run_receipt.json",
            ),
        ),
        (
            quality,
            (
                "answers.jsonl",
                "judges.jsonl",
                "quality_summary.json",
                "quality_receipt.json",
            ),
        ),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / name).write_text(name + "\n", encoding="utf-8")
    for path in (
        root / "conformance/fake.json",
        root / "hardware_receipt.json",
        root / "model_receipt.json",
        root / "tridb_invariant_receipt.json",
        root / "logs/search_b1_fake.log",
        root / "logs/quality_b1_fake.log",
        root / "logs/answer_model.log",
        root / "logs/embedding_model.log",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name + "\n", encoding="utf-8")
    monkeypatch.setattr(goal_audit, "REPOSITORY", repository)
    monkeypatch.setattr(postprocess, "REPOSITORY", repository)

    manifest = postprocess._write_raw_artifact_manifest(
        root=root,
        protocol=protocol,
        schedule_path=schedule_path,
        protocol_path=protocol_path,
    )
    gate = _raw_artifact_manifest_gate(root, schedule_path, protocol_path, protocol)

    assert manifest["artifact_count"] == 22
    assert gate["valid"] is True
    assert all(gate["checks"].values())
    (run / "late_outcomes.jsonl").write_text("tampered\n", encoding="utf-8")
    tampered = _raw_artifact_manifest_gate(root, schedule_path, protocol_path, protocol)
    assert tampered["valid"] is False
    assert tampered["checks"]["entries_exact"] is False


def test_successful_search_receipt_contract_requires_comparable_payload() -> None:
    record = {
        "success": True,
        "receipt": {
            "result_count": 1,
            "hit_ids": ["event-1"],
            "contexts": ["memory context"],
            "context_tokens": None,
            "empty": False,
        },
    }
    assert search_receipt_contract([record])

    del record["receipt"]["contexts"]
    assert not search_receipt_contract([record])


def test_successful_add_receipt_contract_distinguishes_unavailable_from_zero() -> None:
    record = {
        "success": True,
        "receipt": {
            "created_memory_count": 0,
            "created_node_count": None,
            "created_edge_count": None,
            "creation_counts_available": True,
            "creation_count_source": "native_result",
        },
    }
    assert add_receipt_contract([record])

    record["receipt"]["creation_counts_available"] = False
    assert not add_receipt_contract([record])
    record["receipt"]["creation_counts_available"] = True
    record["receipt"]["created_memory_count"] = None
    record["receipt"]["creation_counts_available"] = False
    assert add_receipt_contract([record])
    record["receipt"]["creation_count_source"] = ""
    assert not add_receipt_contract([record])


def _valid_add_receipt() -> dict[str, Any]:
    return {
        "visibility_probe": True,
        "created_memory_count": 1,
        "created_node_count": None,
        "created_edge_count": None,
        "creation_counts_available": True,
        "creation_count_source": "native_result",
    }


def _valid_search_receipt(context: str = "memory") -> dict[str, Any]:
    return {
        "result_count": 1,
        "hit_ids": ["event-1"],
        "contexts": [context],
        "context_tokens": None,
        "empty": False,
    }


def test_generic_conformance_is_recomputed_from_receipt_payloads() -> None:
    receipt = {
        "status": "passed",
        "build": {"events": 58},
        "searches": [_valid_search_receipt() for _ in range(5)],
        "target_add": _valid_add_receipt(),
        "scope_control_add": _valid_add_receipt(),
        "restart_schema_gate": {"ok": True},
        "restart_visible": True,
        "scope_isolated": True,
    }
    assert all(_generic_conformance_checks(receipt).values())
    receipt["target_add"]["created_memory_count"] = -1
    assert not _generic_conformance_checks(receipt)["add_creation_receipt_contract"]


def test_single_conformance_audit_rejects_semantic_drift_before_formal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    path = root / "conformance" / "mem0.json"
    path.parent.mkdir(parents=True)
    schedule = {
        "benchmark_code_sha256": "a" * 64,
        "_schedule_sha256": "b" * 64,
        "protocol_receipt_sha256": "c" * 64,
    }
    protocol = {"systems": ["mem0"]}
    receipt = {
        "status": "passed",
        "execution": {
            "benchmark_code_sha256": "a" * 64,
            "formal_schedule_sha256": "b" * 64,
            "protocol_receipt_sha256": "c" * 64,
        },
        "build": {"events": 58},
        "searches": [_valid_search_receipt() for _ in range(5)],
        "target_add": _valid_add_receipt(),
        "scope_control_add": _valid_add_receipt(),
        "restart_schema_gate": {"ok": True},
        "restart_visible": True,
        "scope_isolated": True,
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")

    assert audit_conformance_receipt(schedule, protocol, root, "mem0")["passed"]
    receipt["scope_isolated"] = False
    path.write_text(json.dumps(receipt), encoding="utf-8")
    result = audit_conformance_receipt(schedule, protocol, root, "mem0")
    assert result["passed"] is False
    assert result["checks"]["scope_isolated"] is False


def test_graphiti_conformance_recomputes_add_search_and_isolation() -> None:
    adapter_sha = "a" * 64
    source_commit = "b" * 40
    schema = {
        "adapter_sha256": adapter_sha,
        "graphiti_source": {"commit": source_commit},
    }
    additions = []
    for index in range(4):
        addition = _valid_add_receipt()
        addition.update(
            {
                "write_concurrency_policy": "per_group_fifo_cross_group_parallel",
                "write_group_id": ("conformance_a" if index < 2 else "conformance_b"),
                "write_group_fifo_ticket": index % 2,
                "write_group_fifo_wait_ms": 0.0,
                "write_concurrency_provenance": "official MCP QueueService",
            }
        )
        additions.append(addition)
    searches = [
        {"result": _valid_search_receipt("Asteria Cedar Labs")},
        {"result": _valid_search_receipt("Borealis Maple Works")},
    ]
    receipt = {
        "status": "complete",
        "schema_version": "graphiti_track_c_conformance_v0.3.0",
        "display_label": "Graphiti (Zep OSS proxy)",
        "interpretation_boundary": "not production Zep",
        "write_phase": {
            "schema_gate": schema,
            "events": 4,
            "additions": additions,
            "search_results": searches,
            "counts": {"nodes": 1},
            "group_counts": {"a": {"nodes": 1}},
            "neo4j_runtime": {"systemd_invocation_id": "c" * 32, "main_pid": 1},
        },
        "read_phase": {
            "schema_gate": schema,
            "search_results": searches,
            "counts": {"nodes": 1},
            "group_counts": {"a": {"nodes": 1}},
            "neo4j_runtime": {"systemd_invocation_id": "d" * 32, "main_pid": 2},
        },
    }
    checks = _graphiti_conformance_checks(
        receipt, expected_adapter_sha=adapter_sha, expected_commit=source_commit
    )
    assert all(checks.values())
    receipt["read_phase"]["search_results"][1]["result"]["contexts"] = [
        "Borealis Maple Works Asteria"
    ]
    assert not _graphiti_conformance_checks(
        receipt, expected_adapter_sha=adapter_sha, expected_commit=source_commit
    )["scope_isolation_after_restart"]


def test_complete_fresh_search_point_passes_all_structural_gates(
    tmp_path: Path, monkeypatch
) -> None:
    system = "tridb_gem"
    workload = "search"
    round_name = "b1"
    build_id = "tc5v2_tridb_gem_search_b1"
    run_dir = tmp_path / "runs" / system / workload / round_name
    formal = [
        {
            "request_index": index,
            "trace_id": f"{index:032x}",
            "phase": "formal_search",
            "system": system,
            "build_id": build_id,
            "schema_version": "table5_track_c_request_v0.3.0",
            "sample_id": "sample",
            "question_id": f"formal-question-{index}",
            "success": index % 17 != 0,
            "timeout": False,
            "error": None if index % 17 != 0 else "RuntimeError: measured failure",
            "harness_retries": 0,
            "system_internal_retries": None,
            "system_internal_retry_observability": "native counter unavailable",
            "observable_call_counts": _observable_calls(),
            "scheduled_at_ns": index * 100_000_000,
            "admitted_at_ns": index * 100_000_000,
            "started_at_ns": index * 100_000_000 + 1_000_000,
            "completed_at_ns": index * 100_000_000 + 2_000_000,
            "admission_lag_ms": 0.0,
            "queue_latency_ms": 1.0,
            "service_latency_ms": 1.0,
            "user_visible_latency_ms": 2.0,
            "receipt": {
                "result_count": 1,
                "hit_ids": ["event-1"],
                "contexts": ["context"],
                "context_tokens": None,
                "empty": False,
            },
        }
        for index in range(1_787)
    ]
    warmup = [
        {
            "request_index": index,
            "trace_id": f"{index + 100_000:032x}",
            "phase": "warmup_search",
            "system": system,
            "build_id": build_id,
            "schema_version": "table5_track_c_request_v0.3.0",
            "sample_id": "sample",
            "question_id": f"warmup-question-{index}",
            "success": True,
            "timeout": False,
            "error": None,
            "harness_retries": 0,
            "system_internal_retries": None,
            "system_internal_retry_observability": "native counter unavailable",
            "observable_call_counts": _observable_calls(),
            "scheduled_at_ns": index * 10_000_000,
            "admitted_at_ns": index * 10_000_000,
            "started_at_ns": index * 10_000_000 + 1_000_000,
            "completed_at_ns": index * 10_000_000 + 2_000_000,
            "admission_lag_ms": 0.0,
            "queue_latency_ms": 1.0,
            "service_latency_ms": 1.0,
            "user_visible_latency_ms": 2.0,
            "receipt": {
                "result_count": 1,
                "hit_ids": ["event-1"],
                "contexts": ["context"],
                "context_tokens": None,
                "empty": False,
            },
        }
        for index in range(307)
    ]
    spans = []
    for record in [*formal, *warmup]:
        root_span_id = f"{record['request_index']:016x}"[-16:]
        child_span_id = f"{record['request_index'] + 10_000:016x}"[-16:]
        common = {
            "schema_version": "table5_track_c_span_v0.1.0",
            "trace_id": record["trace_id"],
            "system": system,
            "build_id": build_id,
            "phase": record["phase"],
            "request_index": record["request_index"],
            "success": record["success"],
            "timeout": record["timeout"],
            "error": record["error"],
        }
        spans.extend(
            [
                {
                    **common,
                    "span_id": root_span_id,
                    "category": "request",
                    "operation": "harness.service_window",
                    "parent_span_id": None,
                    "started_at_ns": record["started_at_ns"],
                    "completed_at_ns": record["completed_at_ns"],
                    "duration_ms": record["service_latency_ms"],
                },
                {
                    **common,
                    "span_id": child_span_id,
                    "category": "fusion",
                    "operation": "test.compound_call",
                    "parent_span_id": root_span_id,
                    "started_at_ns": record["started_at_ns"],
                    "completed_at_ns": record["completed_at_ns"],
                    "duration_ms": record["service_latency_ms"],
                    "attributes": {"observable_call_kind": "opaque_native_api"},
                },
            ]
        )
    _write_jsonl(run_dir / "formal.jsonl", formal)
    _write_jsonl(run_dir / "warmup.jsonl", warmup)
    _write_jsonl(run_dir / "spans.jsonl", spans)
    _write_jsonl(run_dir / "late_outcomes.jsonl", [])
    receipt = {
        "schema_version": "table5_track_c_run_v0.3.0",
        "status": "complete",
        "system": system,
        "phase": "search",
        "build_id": build_id,
        "dataset": {"sha256": "dataset-sha"},
        "protocol": {
            "qps": 10.0,
            "timeout_seconds": 60.0,
            "harness_retries": 0,
            "max_in_flight": 768,
            "scheduled_interval_ns": 100_000_000,
            "admission_qps_relative_tolerance": 0.01,
            "admission_lag_p99_max_ms": 100.0,
            "top_k": 35,
            "evaluation_fields_visible_to_adapter": False,
            "stage_profiling": True,
            "per_request_call_observability": True,
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
                "client-boundary counts only; exact HTTP/database round trips "
                "inside compound native APIs are unavailable and never inferred"
            ),
            "late_timeout_outcome_policy": (
                "one sidecar record after every timed-out worker drains; the "
                "client-visible timeout record is immutable"
            ),
            "late_outcome_schema": "table5_track_c_late_outcome_v0.1.0",
        },
        "execution": {
            "benchmark_code_sha256": "a" * 64,
            "launcher_script_sha256": "c" * 64,
            "formal_schedule_sha256": "1" * 64,
            "protocol_receipt_sha256": "f" * 64,
            "git_branch": "hza214/table5-track-c-20260819",
            "fresh_state_launcher": {
                "schema_version": "table5_track_c_fresh_state_launcher_v0.1.0",
                "policy": (
                    "hashed launcher refuses every pre-existing per-point database, "
                    "volume, or namespace before creation"
                ),
                "build_id": build_id,
                "kind": "postgres_database",
                "primary_identity": build_id,
                "secondary_identity": "database_owner:hza214",
                "preexisting_check_passed": True,
            },
        },
        "formal_summary": {
            "total": 1_787,
            "successful": 1_681,
            "failed": 106,
            "service_latency_ms": {"mean": 1.0, "p90": 2.0, "p99": 3.0},
        },
        "time_breakdown": goal_audit.summarize_spans(run_dir / "spans.jsonl"),
        "build_mode": "fresh_ingest",
        "build_finalize": {},
        "build_wall_seconds": 1.0,
        "late_outcome_summary": {
            "schema_version": "table5_track_c_late_outcome_summary_v0.1.0",
            "path": str((run_dir / "late_outcomes.jsonl").resolve()),
            "sha256": goal_audit._sha256(run_dir / "late_outcomes.jsonl"),
            "records": 0,
            "by_phase": {},
            "final_success": 0,
            "final_failed": 0,
            "post_timeout_work_ms": {"total": 0, "max": None},
        },
    }
    (run_dir / "run_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        goal_audit,
        "_canonical_input_keys",
        lambda protocol, workload: (
            [("sample", f"warmup-question-{index}") for index in range(307)],
            [("sample", f"formal-question-{index}") for index in range(1_787)],
        ),
    )

    result = audit_run(_schedule(), _protocol(), tmp_path, system, workload, round_name)

    assert result["status"] == "COMPLETE_VALID"
    assert result["valid"] is True
    assert all(result["checks"].values())
    assert all(result["span_checks"].values())


def test_exact_input_gate_rejects_duplicate_or_reordered_identity() -> None:
    records = [
        {"request_index": 1, "sample_id": "b", "question_id": "q2"},
        {"request_index": 0, "sample_id": "a", "question_id": "q1"},
    ]
    expected = [("a", "q1"), ("b", "q2")]
    assert _input_keys_exact(records, expected, "question_id")

    records[0]["question_id"] = "q1"
    assert not _input_keys_exact(records, expected, "question_id")


def test_request_gate_checks_real_admission_rate_and_lag() -> None:
    records = []
    for index in range(10):
        scheduled = index * 100_000_000
        admitted = scheduled + 1_000_000
        records.append(
            {
                "request_index": index,
                "trace_id": f"{index:032x}",
                "phase": "formal_search",
                "system": "fake",
                "build_id": "b1",
                "schema_version": "table5_track_c_request_v0.3.0",
                "scheduled_at_ns": scheduled,
                "admitted_at_ns": admitted,
                "started_at_ns": admitted + 1_000_000,
                "completed_at_ns": admitted + 2_000_000,
                "admission_lag_ms": 1.0,
                "queue_latency_ms": 2.0,
                "service_latency_ms": 1.0,
                "user_visible_latency_ms": 3.0,
                "success": True,
                "timeout": False,
                "error": None,
                "harness_retries": 0,
                "system_internal_retries": None,
            }
        )
    scan = {"exists": True, "errors": 0, "records": records}
    kwargs = {
        "expected_count": 10,
        "expected_phase": "formal_search",
        "expected_system": "fake",
        "expected_build": "b1",
        "expected_interval_ns": 100_000_000,
        "expected_qps": 10.0,
        "admission_qps_relative_tolerance": 0.01,
        "admission_lag_p99_max_ms": 100.0,
    }

    checks = _request_checks(scan, **kwargs)

    assert checks["admission_order_exact"] is True
    assert checks["actual_admission_qps_within_tolerance"] is True
    assert checks["admission_lag_p99_within_limit"] is True

    for index, record in enumerate(records):
        admitted = index * 200_000_000
        record["admitted_at_ns"] = admitted
        record["started_at_ns"] = admitted + 1_000_000
        record["completed_at_ns"] = admitted + 2_000_000
        record["admission_lag_ms"] = index * 100.0
        record["queue_latency_ms"] = index * 100.0 + 1.0
        record["user_visible_latency_ms"] = index * 100.0 + 2.0

    checks = _request_checks(scan, **kwargs)

    assert checks["actual_admission_qps_within_tolerance"] is False
    assert checks["admission_lag_p99_within_limit"] is False


def test_dataset_manifest_observation_recomputes_every_scale_field(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "locomo10.json"
    dataset.write_text("[]\n", encoding="utf-8")
    corpus = SimpleNamespace(
        path=dataset.resolve(),
        sha256="dataset-sha",
        sample_ids=("a", "b"),
        all_events=tuple(range(4)),
        queries_by_sample={"a": (1, 2), "b": (3,)},
        formal_queries=(2, 3),
    )
    monkeypatch.setattr(goal_audit, "load_locomo", lambda _: corpus)
    protocol = {
        "dataset": {
            "path": str(dataset.resolve()),
            "sha256": "dataset-sha",
            "conversations": 2,
            "events": 4,
            "questions_total": 3,
            "formal_search": 2,
        }
    }

    observed = _dataset_manifest_observation(protocol)

    assert observed["valid"] is True
    assert observed["resolved_path"] == str(dataset.resolve())
    protocol["dataset"]["events"] = 5
    assert _dataset_manifest_observation(protocol)["valid"] is False


def test_draft_schedule_cannot_validate_a_complete_point(tmp_path: Path) -> None:
    schedule = _schedule()
    schedule["status"] = "draft"
    result = audit_run(schedule, _protocol(), tmp_path, "tridb_gem", "search", "b1")
    assert result["status"] == "MISSING"
    assert result["valid"] is False


def test_quality_receipt_file_alone_is_not_a_quality_gate(tmp_path: Path) -> None:
    output = tmp_path / "quality" / "tridb_gem" / "b1"
    output.mkdir(parents=True)
    (output / "quality_receipt.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )

    result = _quality_point(tmp_path, _schedule(), "tridb_gem", "b1")

    assert result["status"] == "COMPLETE_INVALID"
    assert result["valid"] is False


def test_quality_gate_recomputes_summary_from_raw_evidence(tmp_path: Path) -> None:
    from bench.agent_memory.table5_track_c import quality

    output = tmp_path / "quality" / "tridb_gem" / "b1"
    output.mkdir(parents=True)
    build_id = "tc5v2_tridb_gem_search_b1"
    answers = [
        {
            "schema_version": "table5_track_c_answer_v0.2.0",
            "request_index": index,
            "system": "tridb_gem",
            "build_id": build_id,
            "sample_id": f"sample-{index % 10}",
            "question_id": str(index),
            "category": "1",
            "retrieval_success": True,
            "success": True,
            "generated_answer": "answer",
            "context_count": 1,
            "context_tokens": 10,
            "error": None,
        }
        for index in range(1_787)
    ]
    judges = [
        {
            "schema_version": "table5_track_c_judge_v0.2.0",
            "request_index": index,
            "system": "tridb_gem",
            "build_id": build_id,
            "sample_id": f"sample-{index % 10}",
            "question_id": str(index),
            "category": "1",
            "answer_success": True,
            "success": True,
            "correct": index % 2 == 0,
            "label": "CORRECT" if index % 2 == 0 else "WRONG",
            "error": None,
        }
        for index in range(1_787)
    ]
    answers_path = output / "answers.jsonl"
    judges_path = output / "judges.jsonl"
    _write_jsonl(answers_path, answers)
    _write_jsonl(judges_path, judges)
    summary = quality._summary_evidence(answers, judges)
    summary.update({"system": "tridb_gem", "build_id": build_id})
    summary_path = output / "quality_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    formal_path = tmp_path / "runs" / "tridb_gem" / "search" / "b1" / "formal.jsonl"
    _write_jsonl(formal_path, [{"request_index": index} for index in range(1_787)])
    receipt = {
        "schema_version": "table5_track_c_quality_run_v0.2.0",
        "status": "complete",
        "system": "tridb_gem",
        "build_id": build_id,
        "formal_records": 1_787,
        "model": "Qwen/Qwen3-32B",
        "endpoint": "http://127.0.0.1:8000/v1",
        "workers": 32,
        "answer_prompt_sha256": goal_audit.hashlib.sha256(
            quality.ANSWER_PROMPT.encode()
        ).hexdigest(),
        "judge_prompt_sha256": goal_audit.hashlib.sha256(
            quality.JUDGE_PROMPT.encode()
        ).hexdigest(),
        "benchmark_code_sha256": "a" * 64,
        "formal_sha256": goal_audit._sha256(formal_path),
        "answers_sha256": goal_audit._sha256(answers_path),
        "judges_sha256": goal_audit._sha256(judges_path),
        "summary_sha256": goal_audit._sha256(summary_path),
        "overall": summary["overall"],
    }
    receipt_path = output / "quality_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert _quality_point(tmp_path, _schedule(), "tridb_gem", "b1")["valid"]

    summary["overall"].update(
        {"correct": 1_787, "score": 1.0, "conditional_score": 1.0}
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt["summary_sha256"] = goal_audit._sha256(summary_path)
    receipt["overall"] = summary["overall"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    tampered = _quality_point(tmp_path, _schedule(), "tridb_gem", "b1")
    assert tampered["valid"] is False
    assert tampered["checks"]["summary_evidence_recomputed"] is False
    assert tampered["checks"]["receipt_overall_matches_summary"] is True


def test_protocol_gate_binds_current_code_scripts_dataset_and_claims(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    dataset = tmp_path / "locomo10.json"
    dataset.write_text("[]\n", encoding="utf-8")
    schedule_path = repository / "schedule.json"
    protocol_path = repository / "protocol.json"
    schedule_path.write_text("{}\n", encoding="utf-8")
    protocol_path.write_text("{}\n", encoding="utf-8")
    graphiti = repository / "experiments" / "graphiti_track_c"
    graphiti.mkdir(parents=True)
    (graphiti / "adapter.py").write_text("VERSION = 1\n", encoding="utf-8")
    plan_relative = (
        "haikaidocs/"
        "table5_mem0_zep_memos_cognee_tridb_gem_reproduction_plan_2026-08-19.md"
    )
    plan_path = repository / plan_relative
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# frozen plan\n", encoding="utf-8")
    script_paths = dict(goal_audit.HASHED_SCRIPTS)
    for relative in script_paths.values():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")

    monkeypatch.setattr(goal_audit, "REPOSITORY", repository)
    monkeypatch.setattr(goal_audit, "benchmark_code_sha256", lambda: "a" * 64)
    monkeypatch.setattr(
        goal_audit,
        "_dataset_manifest_observation",
        lambda _: {"valid": True},
    )
    monkeypatch.setattr(
        goal_audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="branch-name\n"),
    )
    systems = [
        "tridb_gem",
        "mem0",
        "memos",
        "cognee",
        "graphiti_zep_oss_proxy",
    ]
    workloads = ["search", "add_native", "add_source_to_searchable"]
    rounds = ["b1", "b2", "b3"]
    protocol = {
        "schema_version": "table5_track_c_completion_protocol_v0.1.0",
        "status": "frozen",
        "result_root": "results",
        "branch": "branch-name",
        "plan": plan_relative,
        "dataset": {
            "path": str(dataset),
            "sha256": goal_audit._sha256(dataset),
        },
        "systems": systems,
        "workloads": workloads,
        "rounds": rounds,
        "expected_formal_runs": 45,
        "source_receipts": {system: f"receipts/{system}.json" for system in systems},
        "source_identity": {system: {"status": "installed"} for system in systems},
        "models": {
            "answer_artifact": "Qwen/Qwen3-32B-FP8",
            "answer_revision": "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df",
            "answer_served_name": "Qwen/Qwen3-32B",
            "answer_endpoint": "http://127.0.0.1:8000/v1",
            "answer_thinking": False,
            "answer_max_model_len": 32768,
            "embedding_artifact": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            "embedding_served_name": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_endpoint": "http://127.0.0.1:8001/v1",
            "embedding_dimensions": 1024,
            "embedding_max_model_len": 8192,
            "offline_cache_required": True,
            "runtime_versions": {
                "vllm": "0.15.0",
                "transformers": "4.57.0",
                "huggingface_hub": "0.36.2",
                "torch": "2.9.1",
            },
        },
        "protocol": {
            "qps": 10.0,
            "timeout_seconds": 60.0,
            "search_formal": 1_787,
            "add_formal": 2_000,
            "search_warmup": 307,
            "add_warmup": 10,
            "scheduler": "absolute no-drift open-loop",
            "scheduled_interval_ns": 100_000_000,
            "admission_qps_relative_tolerance": 0.01,
            "admission_lag_p99_max_ms": 100.0,
            "max_in_flight": 768,
            "search_top_k": 35,
            "stage_profiling": True,
            "per_request_call_observability": True,
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
                "client-boundary counts only; exact HTTP/database round trips "
                "inside compound native APIs are unavailable and never inferred"
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
            "add_progress_receipt_on_timeout": True,
            "add_progress_fields": [
                "commit_observed",
                "visibility_requested",
                "visibility_probe_started",
                "committed_at_ns",
                "searchable_at_ns",
                "visibility_probe",
                "visibility_error",
            ],
            "add_creation_fields": [
                "created_memory_count",
                "created_node_count",
                "created_edge_count",
                "creation_counts_available",
                "creation_count_source",
            ],
            "unavailable_creation_count_policy": (
                "explicit null with non-empty native provenance; never coerce to zero"
            ),
            "fresh_state_per_system_workload_round": True,
            "one_measured_system_at_a_time": True,
            "automatic_failed_run_retry": False,
            "graphiti_add_concurrency_policy": (
                "same group_id FIFO with max one active native add_episode; different "
                "group_ids may run concurrently; FIFO wait remains inside measured "
                "service latency; post-commit visibility probe is outside the write lock"
            ),
        },
        "display_names": {"graphiti_zep_oss_proxy": "Graphiti (Zep OSS proxy)"},
        "zep_naming": {
            "decision": "user_approved_local_proxy_2026_08_21",
            "result_label": "Graphiti (Zep OSS proxy)",
            "production_zep_claim_allowed": False,
            "table5_zep_reproduction_claim_allowed": False,
        },
        "hardware_claim": {
            "h800_numerical_reproduction": False,
            "gx10_signoff": False,
        },
        "resource_policy": {
            "measured_system_concurrency": (
                "one formal system/workload point at a time"
            ),
            "model_gpu_binding": (
                "answer and embedding vLLM services use CUDA_VISIBLE_DEVICES=0"
            ),
            "model_gpu_prelaunch": (
                "physical GPU0 must have zero compute processes before "
                "controlled-model launch; never terminate external processes "
                "and do not claim exclusive mode"
            ),
            "cpu_affinity": ("observed inherited affinity; no benchmark-specific pin"),
            "ram_cap": ("observed inherited cgroup limit; no benchmark-specific cap"),
            "cache_policy": (
                "warm model and warm process/index; no host page-cache drop"
            ),
            "protocol_deviation": (
                "Pre-registration fairness rule 10.5 requested fixed CPU affinity "
                "and RAM cap. Shared PostgreSQL and separately managed backends "
                "are not placed in one benchmark-only cgroup, so this run records "
                "but does not claim those two controls."
            ),
        },
        "statistics": {
            "percentile_method": "linear interpolation",
            "bootstrap_confidence_level": 0.95,
            "bootstrap_iterations": 2_000,
            "bootstrap_seed": 20_260_819,
            "bootstrap_cluster": (
                "sample_id (LoCoMo conversation; pooled across fresh builds)"
            ),
            "outlier_policy": "retain all formal admissions",
            "service_latency_population": "successful completions",
            "user_visible_latency_population": (
                "all admissions including failures and timeouts"
            ),
        },
        "quality": {
            "answer_and_judge_outside_latency_clock": True,
            "non_inferiority_margin_percentage_points": 2.0,
            "non_inferiority_method": "paired_cluster_bootstrap",
            "pairing_keys": ["round", "sample_id", "question_id"],
            "bootstrap_cluster": "sample_id",
            "bootstrap_iterations": 2_000,
            "bootstrap_seed": 20_260_819,
            "one_sided_confidence_level": 0.95,
            "failure_scoring": (
                "retrieval, answer, or judge failure scores incorrect (0)"
            ),
            "receipt": "quality/<system>/<round>/quality_receipt.json",
        },
        "paper_reference": {
            "source": "Mandol arXiv:2606.29778 Table 5",
            "comparison_boundary": (
                "context only; Track C uses different controlled models and "
                "non-H800 hardware"
            ),
            "values_ms": {
                "mem0": {
                    "search": {"mean": 1089.0, "p90": 1397.0, "p99": 4637.0},
                    "add:native": {
                        "mean": 888.0,
                        "p90": 1650.0,
                        "p99": 2841.0,
                    },
                },
                "memos": {
                    "search": {"mean": 440.5, "p90": 528.4, "p99": 777.1},
                    "add:native": {
                        "mean": 191.9,
                        "p90": 211.6,
                        "p99": 376.4,
                    },
                },
            },
            "excluded": {
                "graphiti_zep_oss_proxy": (
                    "not production Zep and not comparable to the paper Zep row"
                ),
                "cognee": "no Cognee row in paper Table 5",
                "tridb_gem": "no TriDB/GEM row in paper Table 5",
                "add:source_to_searchable": (
                    "paper Table 5 does not report this endpoint definition"
                ),
            },
        },
    }
    schedule = {
        "schema_version": "table5_track_c_completion_schedule_v0.1.0",
        "status": "frozen",
        "result_root": "results",
        "systems": systems,
        "workloads": workloads,
        "rounds": rounds,
        "expected_formal_runs": 45,
        "round_order": {
            "b1": [
                "tridb_gem",
                "mem0",
                "memos",
                "cognee",
                "graphiti_zep_oss_proxy",
            ],
            "b2": [
                "mem0",
                "memos",
                "cognee",
                "graphiti_zep_oss_proxy",
                "tridb_gem",
            ],
            "b3": [
                "memos",
                "cognee",
                "graphiti_zep_oss_proxy",
                "tridb_gem",
                "mem0",
            ],
        },
        "failed_run_retry": "never_automatic",
        "historical_qps_sweep_reused_as_formal_build": False,
        "historical_qps_sweep_exclusion_reason": (
            "Search build and measurement receipts have different benchmark hashes "
            "and are not bound to one frozen schedule/protocol receipt."
        ),
        "benchmark_code_sha256": "a" * 64,
        "graphiti_adapter_sha256": goal_audit._graphiti_adapter_sha256(),
        "plan_sha256": goal_audit._sha256(plan_path),
        "protocol_receipt_sha256": goal_audit._sha256(protocol_path),
        "_schedule_sha256": goal_audit._sha256(schedule_path),
        "source_receipt_sha256": {system: "5" * 64 for system in systems},
    }
    schedule.update(
        {
            field: goal_audit._sha256(repository / relative)
            for field, relative in script_paths.items()
        }
    )

    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is True
    assert all(result["checks"].values())

    plan_path.write_text("# drifted plan\n", encoding="utf-8")
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["plan"] is False
    plan_path.write_text("# frozen plan\n", encoding="utf-8")

    schedule["round_order"]["b2"][-1] = "mem0"
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["round_order"] is False
    schedule["round_order"]["b2"][-1] = "tridb_gem"

    protocol["zep_naming"]["decision"] = "unapproved_proxy"
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["graphiti_user_decision"] is False
    protocol["zep_naming"]["decision"] = "user_approved_local_proxy_2026_08_21"

    protocol["zep_naming"]["result_label"] = "Zep"
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["graphiti_result_label"] is False
    protocol["zep_naming"]["result_label"] = "Graphiti (Zep OSS proxy)"

    protocol["models"]["answer_revision"] = "moving-main"
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["models"] is False
    protocol["models"]["answer_revision"] = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"

    protocol["protocol"]["admission_qps_relative_tolerance"] = 0.5
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["execution_policy"] is False
    protocol["protocol"]["admission_qps_relative_tolerance"] = 0.01

    (repository / script_paths["run_one_script_sha256"]).write_text(
        "mutated\n", encoding="utf-8"
    )
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["run_one_script_sha256"] is False

    (repository / script_paths["run_one_script_sha256"]).write_text(
        f"{script_paths['run_one_script_sha256']}\n", encoding="utf-8"
    )
    (repository / script_paths["memos_backend_script_sha256"]).write_text(
        "mutated backend\n", encoding="utf-8"
    )
    result = goal_audit._protocol_gate(schedule_path, schedule, protocol_path, protocol)
    assert result["valid"] is False
    assert result["checks"]["memos_backend_script_sha256"] is False


def test_source_identity_normalizes_graphiti_nested_fields() -> None:
    expected = {
        "schema_version": "table5_track_c_install_manifest_v0.2.0",
        "status": "installed",
        "installed": True,
        "display_label": "Graphiti (Zep OSS proxy)",
        "interpretation_boundary": "not production Zep",
        "source_commit": "commit",
        "package_version": "0.29.3",
        "backend": "Neo4j Community 5.26.6",
        "neo4j_version": "5.26.6",
        "cypher_shell_version": "Cypher-Shell 5.26.6",
    }
    receipt = {
        "schema_version": expected["schema_version"],
        "system": "graphiti_zep_oss_proxy",
        "status": "installed",
        "installed": True,
        "display_label": expected["display_label"],
        "interpretation_boundary": expected["interpretation_boundary"],
        "source": {"commit": "commit"},
        "environment": {"graphiti_core_version": "0.29.3"},
        "backend": expected["backend"],
        "backend_identity": {
            "neo4j_version": "5.26.6",
            "cypher_shell_version": "Cypher-Shell 5.26.6",
        },
    }
    assert _source_identity_matches(receipt, "graphiti_zep_oss_proxy", expected)
    receipt["environment"]["graphiti_core_version"] = "0.29.4"
    assert not _source_identity_matches(receipt, "graphiti_zep_oss_proxy", expected)
    receipt["environment"]["graphiti_core_version"] = "0.29.3"
    receipt["backend_identity"]["neo4j_version"] = "5.26.7"
    assert not _source_identity_matches(receipt, "graphiti_zep_oss_proxy", expected)


def test_postprocess_gate_rejects_output_modified_after_receipt(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    schedule_path = tmp_path / "schedule.json"
    protocol_path = tmp_path / "protocol.json"
    schedule_path.write_text("{}\n", encoding="utf-8")
    protocol_path.write_text("{}\n", encoding="utf-8")
    required = ["REPORT.md", "aggregates/search_latency.csv"]
    expected = set(required) | {
        "aggregates/request_metrics.csv",
        "aggregates/latency_decomposition.csv",
        "figures/search_latency_box.pdf",
    }
    for relative in expected:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    outputs = {relative: goal_audit._sha256(root / relative) for relative in expected}
    receipt = {
        "schema_version": "table5_track_c_completion_postprocess_v0.2.0",
        "status": "complete",
        "schedule": str(schedule_path.resolve()),
        "schedule_sha256": goal_audit._sha256(schedule_path),
        "protocol_sha256": goal_audit._sha256(protocol_path),
        "input_goal_audit": {
            "complete_valid_runs": 45,
            "quality_complete": True,
            "prerequisite_complete": True,
        },
        "coverage_checks": {
            name: True for name in goal_audit.POSTPROCESS_COVERAGE_CHECK_NAMES
        },
        "outputs": outputs,
    }
    (root / "postprocess_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    result = _postprocess_gate(
        root, schedule_path, protocol_path, {"required_deliverables": required}
    )
    assert result["valid"] is True

    receipt["coverage_checks"]["metric_rows"] = False
    (root / "postprocess_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    result = _postprocess_gate(
        root, schedule_path, protocol_path, {"required_deliverables": required}
    )
    assert result["valid"] is False
    assert result["checks"]["coverage"] is False
    receipt["coverage_checks"]["metric_rows"] = True
    (root / "postprocess_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    (root / "REPORT.md").write_text("tampered", encoding="utf-8")
    result = _postprocess_gate(
        root, schedule_path, protocol_path, {"required_deliverables": required}
    )
    assert result["valid"] is False
    assert result["checks"]["outputs"] is False


def test_claim_boundary_gate_rejects_proxy_relabeling(tmp_path: Path) -> None:
    root = tmp_path / "results"
    aggregates = root / "aggregates"
    aggregates.mkdir(parents=True)
    (root / "REPORT.md").write_text(
        "\n".join(
            (
                "`Graphiti (Zep OSS proxy)` is not production Zep and is not a "
                "reproduced Zep Table 5 row",
                "Graphiti is deliberately not paired with the paper Zep row",
                "not an H800 numerical reproduction and not a GX10 sign-off",
            )
        ),
        encoding="utf-8",
    )
    csv_path = aggregates / "request_metrics.csv"
    fields = (
        "system",
        "system_display_name",
        "reference",
        "reference_display_name",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "system": "graphiti_zep_oss_proxy",
                "system_display_name": "Graphiti (Zep OSS proxy)",
                "reference": "graphiti_zep_oss_proxy",
                "reference_display_name": "Graphiti (Zep OSS proxy)",
            }
        )
    protocol = {
        "display_names": {
            key: value
            for key, value in goal_audit.EXPECTED_DISPLAY_NAMES.items()
            if key != "all"
        },
        "zep_naming": {
            "production_zep_claim_allowed": False,
            "table5_zep_reproduction_claim_allowed": False,
        },
    }

    result = goal_audit._claim_boundary_gate(root, protocol)
    assert result["valid"] is True

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    rows[0]["system_display_name"] = "Zep"
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = goal_audit._claim_boundary_gate(root, protocol)
    assert result["valid"] is False
    assert result["checks"]["csv_display_names"] is False
