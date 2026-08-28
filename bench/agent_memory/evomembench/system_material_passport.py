"""Material Passport for the complete three-arm EvoMemBench systems run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bench.agent_memory.evomembench.material_passport import (
    _implementation_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _dual_gpu_gate(root: Path, *, required: bool) -> dict[str, Any]:
    admission_log = root / "gpu_layout_admission.log"
    admission_samples = (
        sum(
            "two-GPU ownership gate passed" in line
            for line in admission_log.read_text().splitlines()
        )
        if admission_log.is_file()
        else 0
    )
    continuous_log = root / "gpu_layout_continuous.log"
    continuous_samples = (
        sum("exit_status=0" in line for line in continuous_log.read_text().splitlines())
        if continuous_log.is_file()
        else 0
    )
    violation = root / "gpu_layout_violation.txt"
    violation_detected = violation.is_file()
    passed = not required or (
        admission_samples > 0 and continuous_samples > 0 and not violation_detected
    )
    return {
        "required": required,
        "passed": passed,
        "admission_samples": admission_samples,
        "continuous_successful_samples": continuous_samples,
        "violation_detected": violation_detected,
        "sampling_interval_seconds": 10 if required else None,
    }


def _database_concurrency_gate(root: Path, *, required: bool) -> dict[str, Any]:
    continuous_log = root / "database_concurrency_continuous.log"
    continuous_samples = (
        sum("exit_status=0" in line for line in continuous_log.read_text().splitlines())
        if continuous_log.is_file()
        else 0
    )
    violation = root / "database_concurrency_violation.txt"
    violation_detected = violation.is_file()
    passed = not required or continuous_samples > 0 and not violation_detected
    return {
        "required": required,
        "passed": passed,
        "continuous_successful_samples": continuous_samples,
        "violation_detected": violation_detected,
        "sampling_interval_seconds": 10 if required else None,
    }


def _answer_server_gate(root: Path, *, required: bool) -> dict[str, Any]:
    path = root / "answer_server_admission.json"
    if not path.is_file():
        return {
            "required": required,
            "passed": not required,
            "error": "missing answer-server admission receipt" if required else None,
        }
    receipt = json.loads(path.read_text())
    protocol_path = root / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.is_file() else {}
    models = protocol.get("models") or {}
    observed = receipt.get("observed") or {}
    generation = observed.get("generation_config") or {}
    checks = receipt.get("checks") or {}
    passed = (
        receipt.get("status") == "complete"
        and receipt.get("passed") is True
        and checks
        and all(value is True for value in checks.values())
        and generation.get("max_new_tokens") == 4096
        and generation.get("max_new_tokens") == models.get("judge_max_tokens")
        and observed.get("cuda_visible_devices") == "0"
        and observed.get("model") == models.get("answer_artifact")
        and observed.get("served_model") == models.get("answer")
        and observed.get("context_tokens") == str(models.get("context_window_tokens"))
    )
    return {
        "required": required,
        "passed": passed if required else True,
        "unit": receipt.get("unit"),
        "main_pid": receipt.get("main_pid"),
        "command_sha256": receipt.get("command_sha256"),
        "checks": checks,
        "observed": observed,
    }


def _embedding_server_gate(root: Path, *, required: bool) -> dict[str, Any]:
    path = root / "embedding_server_admission.json"
    if not path.is_file():
        return {
            "required": required,
            "passed": not required,
            "error": "missing embedding-server admission receipt" if required else None,
        }
    receipt = json.loads(path.read_text())
    protocol_path = root / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.is_file() else {}
    models = protocol.get("models") or {}
    observed = receipt.get("observed") or {}
    checks = receipt.get("checks") or {}
    passed = (
        receipt.get("status") == "complete"
        and receipt.get("passed") is True
        and checks
        and all(value is True for value in checks.values())
        and observed.get("cuda_visible_devices") == "1"
        and observed.get("context_tokens") == "32768"
        and observed.get("context_tokens")
        == str(models.get("embedding_context_tokens"))
        and observed.get("runner") == "pooling"
        and observed.get("model") == models.get("embedding")
        and observed.get("served_model") == models.get("embedding")
        and observed.get("revision") == models.get("embedding_revision")
        and observed.get("tokenizer_revision") == models.get("embedding_revision")
    )
    return {
        "required": required,
        "passed": passed if required else True,
        "unit": receipt.get("unit"),
        "main_pid": receipt.get("main_pid"),
        "command_sha256": receipt.get("command_sha256"),
        "checks": checks,
        "observed": observed,
    }


def _database_preparation_gate(
    root: Path, *, local_library_required: bool
) -> dict[str, Any]:
    path = root / "database_preparation.json"
    if not path.is_file():
        return {"passed": False, "error": "missing database preparation receipt"}
    receipt = json.loads(path.read_text())
    expected_functions = {
        "tjs_open",
        "tjs_open_candidates_examined",
        "tjs_open_relational_examined",
        "tjs_open_relational_passed",
        "tjs_open_graph_examined",
        "tjs_open_graph_reached",
        "tjs_open_graph_censored",
        "tjs_open_termination_reason",
        "tjs_open_budget_capped",
        "tjs_open_bridges_injected",
    }
    databases = receipt.get("databases", [])
    database_checks = {
        str(row.get("label")): row.get("gem_units") == 0
        and set(row.get("tjs_functions", {})) == expected_functions
        for row in databases
    }
    library = receipt.get("tjs_library")
    library_valid = not local_library_required or (
        isinstance(library, dict)
        and isinstance(library.get("path"), str)
        and len(str(library.get("sha256", ""))) == 64
    )
    passed = (
        receipt.get("status") == "complete"
        and set(database_checks) == {"native", "scale"}
        and all(database_checks.values())
        and library_valid
    )
    return {
        "passed": passed,
        "local_library_required": local_library_required,
        "local_library": library,
        "database_checks": database_checks,
    }


def _know_grading_gate(root: Path) -> dict[str, Any]:
    arms = ("memory_off", "long_context", "full_gem")
    receipts: dict[str, dict[str, Any]] = {}
    for arm in arms:
        path = root / "know" / "graded" / f"{arm}.receipt.json"
        receipts[arm] = json.loads(path.read_text()) if path.is_file() else {}
    passed = all(
        receipt.get("status") == "complete"
        and receipt.get("predictions") == 884
        and receipt.get("graded") == 884
        and receipt.get("denominator_preserved") is True
        for receipt in receipts.values()
    )
    return {
        "passed": passed,
        "expected_per_arm": 884,
        "arms": {
            arm: {
                "status": receipt.get("status"),
                "predictions": receipt.get("predictions"),
                "graded": receipt.get("graded"),
                "denominator_preserved": receipt.get("denominator_preserved"),
                "judge_failures_counted_as_score_zero": receipt.get(
                    "judge_failures_counted_as_score_zero"
                ),
            }
            for arm, receipt in receipts.items()
        },
    }


def _formal_count_gate(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def cleaned(loads: list[dict[str, Any]]) -> bool:
        return bool(loads) and all(
            row.get("cleanup", {}).get("scope") == "exact_run_owned_namespace"
            and row.get("cleanup", {}).get("collection_dropped") is True
            and row.get("cleanup", {}).get("neo4j_nodes_deleted") == row.get("units")
            and row.get("cleanup", {}).get("postgresql_rows_deleted")
            == row.get("units")
            for row in loads
        )

    know_loads = receipts["know_multi_system"].get("load_metrics", [])
    tool_loads = receipts["tool_multi_system"].get("load_metrics", [])
    scale_loads = [
        point.get("multi_system_load", {})
        for point in receipts["scale"].get("points", [])
    ]
    checks = {
        "know_predictions": receipts["know"].get("predictions") == 884 * 3,
        "know_reuse_queries_per_arm": receipts["know"].get(
            "cross_episode_decisions_per_arm"
        )
        == 764,
        "know_multi_queries": receipts["know_multi_system"].get("queries") == 764,
        "know_multi_snapshot_loads": len(know_loads) == 120,
        "know_multi_namespace_cleanup": cleaned(know_loads),
        "know_native_footprint": bool(receipts["know"].get("database_footprint_after")),
        "tool_source_tasks_per_arm": receipts["tool"].get(
            "source_building_evaluations_per_memory_arm"
        )
        == 200,
        "tool_target_queries_per_arm": receipts["tool"].get(
            "target_evaluations_per_arm"
        )
        == 600,
        "tool_transfer_cells": receipts["tool"].get("directed_transfer_pairs") == 12,
        "tool_multi_queries": receipts["tool_multi_system"].get("queries") == 600,
        "tool_multi_snapshot_loads": len(tool_loads) == 4,
        "tool_multi_namespace_cleanup": cleaned(tool_loads),
        "tool_native_footprints": bool(
            receipts["tool"].get("database_footprint_before")
        )
        and bool(receipts["tool"].get("database_footprint_after")),
        "scale_history_sizes": receipts["scale"].get("history_sizes")
        == [1000, 10000, 100000, 1000000],
        "scale_queries": receipts["scale"].get("queries") == 100,
        "scale_warmups": receipts["scale"].get("warmups") == 10,
        "scale_repetitions": receipts["scale"].get("measured_repetitions") == 30,
        "scale_points": len(receipts["scale"].get("points", [])) == 4,
        "scale_multi_load_metrics": all(
            bool(point.get("multi_system_load"))
            for point in receipts["scale"].get("points", [])
        ),
        "scale_multi_namespace_cleanup": cleaned(scale_loads),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _tool_canonical_source_gate(
    root: Path, tool_receipt: dict[str, Any]
) -> dict[str, Any]:
    path = root / "tool" / "canonical_source_trajectories.json"
    if not path.is_file():
        return {"passed": False, "error": "missing canonical source manifest"}
    payload = json.loads(path.read_text())
    observed_digest = payload.pop("digest", None)
    recomputed = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    environments = payload.get("environments", {})
    source_errors = tool_receipt.get("source_building_errors", {})
    update_checks: dict[str, dict[str, Any]] = {}
    for arm in ("long_context", "gem_fused"):
        rows = _jsonl(root / "tool" / "updates" / f"{arm}.jsonl")
        sha_matches = 0
        for row in rows:
            expected = environments.get(row.get("source_environment"), {}).get(
                row.get("source_id")
            )
            sha_matches += int(
                bool(expected) and row.get("canonical_trajectory_sha256") == expected
            )
        update_checks[arm] = {
            "rows": len(rows),
            "committed_rows": sum(row.get("status") == "committed" for row in rows),
            "canonical_sha_matches": sha_matches,
            "passed": len(rows) == 200
            and all(row.get("status") == "committed" for row in rows)
            and sha_matches == 200,
        }
    passed = (
        observed_digest == recomputed
        and tool_receipt.get("canonical_source_trajectory_digest") == observed_digest
        and len(environments) == 4
        and all(len(rows) == 50 for rows in environments.values())
        and all(int(value) == 0 for value in source_errors.values())
        and len(source_errors) == 12
        and all(check["passed"] for check in update_checks.values())
    )
    return {
        "passed": passed,
        "digest": observed_digest,
        "recomputed_digest": recomputed,
        "environments": {name: len(rows) for name, rows in environments.items()},
        "source_building_errors": source_errors,
        "construction_update_checks": update_checks,
    }


def _update_trace_gate(root: Path) -> dict[str, Any]:
    counts = {
        "know_long_context": len(_jsonl(root / "know/updates/long_context.jsonl")),
        "know_full_gem": len(_jsonl(root / "know/updates/full_gem.jsonl")),
        "tool_long_context": len(_jsonl(root / "tool/updates/long_context.jsonl")),
        "tool_gem_fused": len(_jsonl(root / "tool/updates/gem_fused.jsonl")),
        "scale": len(_jsonl(root / "scale/per_update.jsonl")),
    }
    checks = {
        "know_long_context": counts["know_long_context"] == 884,
        "know_full_gem": counts["know_full_gem"] == 884,
        "tool_long_context": counts["tool_long_context"] == 200,
        "tool_gem_fused": counts["tool_gem_fused"] == 200,
        "scale": counts["scale"] > 0,
    }
    return {"passed": all(checks.values()), "counts": counts, "checks": checks}


def _empirical_claim_gates(summary: dict[str, Any]) -> dict[str, Any]:
    claims = summary.get("claim_gates") or {}
    latency = claims.get("5_latency", {})
    native_latency = latency.get("native", {})
    scale_latency = latency.get("systems_scale", [])
    intermediate = claims.get("6_peak_application_materialized_ids", {})
    tokens = claims.get("7_answer_prompt_token_reduction_with_construction", {})
    quality = claims.get("4_full_gem_long_context_quality_noninferiority", {})
    checks = {
        "retrieval_parity": claims.get("1_full_gem_multi_system_parity_100pct") is True,
        "quality_noninferiority": bool(quality)
        and all(value is True for value in quality.values()),
        "native_latency": bool(native_latency)
        and all(row.get("passed") is True for row in native_latency.values()),
        "scale_latency": len(scale_latency) == 4
        and all(row.get("passed") is True for row in scale_latency),
        "peak_materialization": bool(intermediate)
        and all(row.get("passed") is True for row in intermediate.values()),
        "answer_token_reduction": bool(tokens)
        and all(row.get("passed") is True for row in tokens.values()),
    }
    know_utility = claims.get("2_know_memory_utility") or {}
    tool_utility = claims.get("3_tool_memory_utility") or {}
    return {
        "systems_headline_passed": all(checks.values()),
        "systems_checks": checks,
        "know_memory_utility_claim_passed": know_utility.get("memory_utility_pass")
        is True,
        "tool_memory_utility_claim_passed": tool_utility.get("memory_utility_pass")
        is True,
    }


def build(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "system_summary.json").read_text())
    preflight = json.loads((root / "preflight.json").read_text())
    run_receipts = {
        name: json.loads((root / name / "run_receipt.json").read_text())
        for name in (
            "know",
            "know_multi_system",
            "tool",
            "tool_multi_system",
            "scale",
        )
    }
    executions_valid = all(
        receipt.get("status") == "complete" and receipt.get("formal") is True
        for receipt in run_receipts.values()
    )
    formal_count_gate = _formal_count_gate(run_receipts)
    tool_source_gate = _tool_canonical_source_gate(root, run_receipts["tool"])
    update_trace_gate = _update_trace_gate(root)
    native_parity_valid = all(
        track["parity"]["all_passed"] for track in summary["tracks"].values()
    )
    scale = summary.get("systems_scale")
    scale_parity_valid = (
        bool(scale)
        and all(point["parity"]["all_passed"] for point in scale.get("points", []))
        and len(scale.get("points", [])) == 4
    )
    parity_valid = native_parity_valid and scale_parity_valid
    dual_gpu_required = "dual_gpu_host" in preflight.get("required_checks", [])
    dual_gpu_gate = _dual_gpu_gate(root, required=dual_gpu_required)
    gpu_layout_valid = dual_gpu_gate["passed"]
    answer_server_gate = _answer_server_gate(root, required=dual_gpu_required)
    embedding_server_gate = _embedding_server_gate(root, required=dual_gpu_required)
    database_required = "database_concurrency" in preflight.get("required_checks", [])
    database_gate = _database_concurrency_gate(root, required=database_required)
    database_preparation_gate = _database_preparation_gate(
        root, local_library_required=dual_gpu_required
    )
    grading_gate = _know_grading_gate(root)
    artifact_receipt_path = root / "canonical_artifact_receipt.json"
    artifact_receipt = (
        json.loads(artifact_receipt_path.read_text())
        if artifact_receipt_path.is_file()
        else {}
    )
    canonical_artifacts_valid = (
        artifact_receipt.get("status") == "complete"
        and artifact_receipt.get("parity_all_passed") is True
        and all(
            (root / path).exists()
            for path in (
                "hardware.json",
                "per_query.jsonl",
                "per_update.jsonl",
                "parity.json",
                "native_summary.json",
                "scale_summary.json",
                "figures",
            )
        )
    )
    protocol_valid = (
        bool(preflight.get("passed"))
        and executions_valid
        and parity_valid
        and gpu_layout_valid
        and answer_server_gate["passed"]
        and embedding_server_gate["passed"]
        and database_gate["passed"]
        and database_preparation_gate["passed"]
        and grading_gate["passed"]
        and formal_count_gate["passed"]
        and tool_source_gate["passed"]
        and update_trace_gate["passed"]
        and canonical_artifacts_valid
    )
    gx10_valid = bool(
        preflight.get("checks", {}).get("gx10_hardware", {}).get("passed")
    )
    empirical_claim_gates = _empirical_claim_gates(summary)
    headline_claim_valid = (
        protocol_valid
        and gx10_valid
        and empirical_claim_gates["systems_headline_passed"]
    )
    excluded = {"MATERIAL_PASSPORT.json", "MATERIAL_PASSPORT.md", "SHA256SUMS"}
    artefacts = {
        str(path.relative_to(root)): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    implementation, implementation_digest = _implementation_manifest()
    return {
        "schema_version": "evomembench_system_material_passport_v0.1.0",
        "verification_status": (
            "executed_protocol_invalid"
            if not protocol_valid
            else (
                "executed_protocol_valid_off_target"
                if not gx10_valid
                else (
                    "executed_protocol_valid_gx10_claim_eligible"
                    if headline_claim_valid
                    else "executed_protocol_valid_gx10_claim_gates_failed"
                )
            )
        ),
        "protocol_valid": protocol_valid,
        "gx10_hardware_gate_passed": gx10_valid,
        "headline_claim_valid": headline_claim_valid,
        "empirical_claim_gates": empirical_claim_gates,
        "preflight_passed": bool(preflight.get("passed")),
        "dual_gpu_layout_gate": dual_gpu_gate,
        "answer_server_admission_gate": answer_server_gate,
        "embedding_server_admission_gate": embedding_server_gate,
        "database_concurrency_gate": database_gate,
        "database_preparation_gate": database_preparation_gate,
        "know_grading_denominator_gate": grading_gate,
        "formal_workload_count_gate": formal_count_gate,
        "tool_canonical_source_gate": tool_source_gate,
        "update_trace_gate": update_trace_gate,
        "canonical_artifacts_gate": {
            "passed": canonical_artifacts_valid,
            "receipt": artifact_receipt,
        },
        "formal_receipts": {
            name: {
                "status": receipt.get("status"),
                "formal": receipt.get("formal"),
                "run_id": receipt.get("run_id"),
            }
            for name, receipt in run_receipts.items()
        },
        "parity": {
            "native": {
                name: track["parity"] for name, track in summary["tracks"].items()
            },
            "systems_scale": (
                [
                    {
                        "history_size": point["history_size"],
                        **point["parity"],
                    }
                    for point in scale["points"]
                ]
                if scale
                else None
            ),
        },
        "implementation_manifest_sha256": implementation_digest,
        "implementation_files": implementation,
        "artefacts": artefacts,
        "known_limitations": [
            "EvoMemBench has no independent source-experience relevance qrels",
            "multi-system end-to-end latency is parity-conditioned reconstruction",
            "application-visible transfer bytes are serialized-payload accounting, not packet capture",
            *(
                []
                if gx10_valid
                else [
                    "off-target execution cannot sign off PG13.4 ARM64/CUDA/GX10 headline claims"
                ]
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    json_path = args.run_root / "MATERIAL_PASSPORT.json"
    markdown_path = args.run_root / "MATERIAL_PASSPORT.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing existing system Material Passport")
    passport = build(args.run_root)
    json_path.write_text(json.dumps(passport, ensure_ascii=False, indent=2) + "\n")
    markdown_path.write_text(
        "\n".join(
            [
                "# Material Passport",
                "",
                f"- Verification status: `{passport['verification_status']}`",
                f"- Protocol valid: `{passport['protocol_valid']}`",
                f"- Content-addressed artefacts: {len(passport['artefacts'])}",
                "",
                "## Known limitations",
                "",
                *[f"- {item}" for item in passport["known_limitations"]],
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
