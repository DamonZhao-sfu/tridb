#!/usr/bin/env python3
"""Validate and atomically freeze the Track C completion-v2 protocol.

The default is a read-only preflight. ``--apply`` is deliberately unavailable
until every historical workload unit has exited and the Graphiti live
install/conformance receipt exists.  Formal measurements remain disabled if
any identity, checksum, source receipt, code hash, or validation command fails.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.table5_track_c.protocol import benchmark_code_sha256
from tools.table5_track_c_goal_audit import (
    DEFAULT_SCHEDULE,
    HASHED_SCRIPTS,
    REPOSITORY,
    _graphiti_adapter_sha256,
    _graphiti_conformance_checks,
    _dataset_manifest_observation,
    _protocol_gate,
    _sha256,
    _source_identity_matches,
    _source_runtime_observation,
)

GRAPHITI_PREFLIGHT = (
    REPOSITORY
    / "bench/out/table5_reproduction_2026_08_19/systems/graphiti/conformance.json"
)
GRAPHITI_COMMIT = "993e081a6d7948a0d8851c12a5fbdbeb49fed862"
WAIT_UNITS = (
    "tridb-table5-add-native-qps10-gpu0-v7.service",
    "tridb-table5-mandol-search-completion-gpu1-v6.service",
    "tridb-table5-graphiti-install-after-current.service",
    "tridb-table5-graphiti-conformance-after-install.service",
    "tridb-table5-graphiti-build-after-conformance.service",
    "tridb-table5-graphiti-search-after-canonical.service",
    "tridb-table5-graphiti-add-after-canonical.service",
    "tridb-table5-completion-v2-execute.service",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _active_units() -> list[str]:
    result = []
    for unit in WAIT_UNITS:
        status = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit], check=False
        )
        if status.returncode == 0:
            result.append(unit)
    return result


def _graphiti_preflight_checks(receipt: dict[str, Any]) -> dict[str, bool]:
    return _graphiti_conformance_checks(
        receipt,
        expected_adapter_sha=_graphiti_adapter_sha256(),
        expected_commit=GRAPHITI_COMMIT,
    )


def preflight(
    schedule_path: Path,
    schedule: dict[str, Any],
    protocol_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    systems = protocol.get("systems") or []
    source_results: dict[str, dict[str, Any]] = {}
    for system in systems:
        relative = (protocol.get("source_receipts") or {}).get(system)
        path = REPOSITORY / str(relative)
        expected = (protocol.get("source_identity") or {}).get(system) or {}
        result: dict[str, Any] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "identity": False,
            "runtime": {"valid": False},
            "sha256": _sha256(path),
        }
        if path.is_file():
            try:
                receipt = _load(path)
                result["identity"] = _source_identity_matches(receipt, system, expected)
                result["runtime"] = _source_runtime_observation(receipt, system)
            except (OSError, json.JSONDecodeError):
                pass
        result["valid"] = (
            result["exists"] and result["identity"] and result["runtime"]["valid"]
        )
        source_results[system] = result

    graphiti_checks: dict[str, bool] = {"exists": GRAPHITI_PREFLIGHT.is_file()}
    if GRAPHITI_PREFLIGHT.is_file():
        try:
            graphiti_checks.update(
                _graphiti_preflight_checks(_load(GRAPHITI_PREFLIGHT))
            )
        except (OSError, json.JSONDecodeError):
            graphiti_checks["readable"] = False

    dataset = Path(str((protocol.get("dataset") or {}).get("path", "")))
    dataset_observation = _dataset_manifest_observation(protocol)
    active = _active_units()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    hash_fields = [
        *HASHED_SCRIPTS,
        "benchmark_code_sha256",
        "graphiti_adapter_sha256",
        "plan_sha256",
        "protocol_receipt_sha256",
    ]
    plan_path = REPOSITORY / str(protocol.get("plan") or "")
    checks = {
        "schedule_draft": schedule.get("status")
        == "draft_pending_graphiti_conformance_and_code_freeze",
        "protocol_draft": protocol.get("status")
        == "draft_pending_graphiti_conformance_and_code_freeze",
        "branch": branch == protocol.get("branch"),
        "dataset": dataset.is_file()
        and _sha256(dataset) == (protocol.get("dataset") or {}).get("sha256")
        and dataset_observation["valid"],
        "plan": protocol.get("plan")
        == "haikaidocs/table5_mem0_zep_memos_cognee_tridb_gem_reproduction_plan_2026-08-19.md"
        and plan_path.is_file(),
        "sources": bool(source_results)
        and all(value["valid"] for value in source_results.values()),
        "graphiti_preflight": bool(graphiti_checks) and all(graphiti_checks.values()),
        "historical_units_inactive": not active,
        "hash_fields_empty": all(schedule.get(field) is None for field in hash_fields),
        "source_hashes_empty": all(
            value is None
            for value in (schedule.get("source_receipt_sha256") or {}).values()
        ),
        "result_tree_has_no_formal_runs": not (
            REPOSITORY / str(protocol.get("result_root")) / "runs"
        ).exists(),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "active_units": active,
        "dataset_observation": dataset_observation,
        "sources": source_results,
        "graphiti_preflight": {
            "path": str(GRAPHITI_PREFLIGHT.resolve()),
            "checks": graphiti_checks,
        },
        "schedule": str(schedule_path.resolve()),
        "protocol": str(protocol_path.resolve()),
    }


def _run_validation() -> list[dict[str, Any]]:
    commands: Sequence[Sequence[str]] = (
        (
            "bash",
            "-n",
            *(
                str(path)
                for path in sorted(
                    (REPOSITORY / "scripts").glob("table5_track_c_completion_v2_*.sh")
                )
            ),
            str(REPOSITORY / HASHED_SCRIPTS["memos_backend_script_sha256"]),
            str(REPOSITORY / HASHED_SCRIPTS["graphiti_backend_script_sha256"]),
        ),
        (
            str(REPOSITORY / ".venv/bin/ruff"),
            "format",
            "--check",
            "bench/agent_memory/table5_track_c",
            "experiments/graphiti_track_c",
            "experiments/graphiti_track_c_formal",
            "tools/table5_track_c_goal_audit.py",
            "tools/table5_track_c_completion_v2_postprocess.py",
            "tools/table5_track_c_freeze_completion_v2.py",
            "tests/test_table5_track_c.py",
            "tests/test_table5_track_c_freeze.py",
            "tests/test_table5_track_c_goal_audit.py",
            "tests/test_table5_track_c_invariants.py",
            "tests/test_graphiti_track_c_adapter.py",
            "tests/test_graphiti_track_c_formal.py",
            "tests/test_table5_track_c_postprocess.py",
        ),
        (
            str(REPOSITORY / ".venv/bin/ruff"),
            "check",
            "bench/agent_memory/table5_track_c",
            "experiments/graphiti_track_c",
            "experiments/graphiti_track_c_formal",
            "tools/table5_track_c_goal_audit.py",
            "tools/table5_track_c_completion_v2_postprocess.py",
            "tools/table5_track_c_freeze_completion_v2.py",
            "tests/test_table5_track_c.py",
            "tests/test_table5_track_c_freeze.py",
            "tests/test_table5_track_c_goal_audit.py",
            "tests/test_table5_track_c_invariants.py",
            "tests/test_graphiti_track_c_adapter.py",
            "tests/test_graphiti_track_c_formal.py",
            "tests/test_table5_track_c_postprocess.py",
        ),
        (str(REPOSITORY / ".venv/bin/python"), "-m", "pytest", "-q", "tests/"),
        ("git", "diff", "--check"),
    )
    results = []
    for command in commands:
        completed = subprocess.run(
            list(command),
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "argv": list(command),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4_000:],
                "stderr_tail": completed.stderr[-4_000:],
            }
        )
        if completed.returncode != 0:
            break
    return results


def _frozen_payloads(
    schedule: dict[str, Any], protocol: dict[str, Any], frozen_at: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_protocol = copy.deepcopy(protocol)
    frozen_protocol["status"] = "frozen"
    frozen_protocol["frozen_at"] = frozen_at
    protocol_sha = _bytes_sha256(_json_bytes(frozen_protocol))

    frozen_schedule = copy.deepcopy(schedule)
    frozen_schedule["status"] = "frozen"
    frozen_schedule["frozen_at"] = frozen_at
    frozen_schedule["protocol_receipt_sha256"] = protocol_sha
    frozen_schedule["benchmark_code_sha256"] = benchmark_code_sha256()
    frozen_schedule["graphiti_adapter_sha256"] = _graphiti_adapter_sha256()
    frozen_schedule["plan_sha256"] = _sha256(REPOSITORY / str(frozen_protocol["plan"]))
    for field, relative in HASHED_SCRIPTS.items():
        frozen_schedule[field] = _sha256(REPOSITORY / relative)
    frozen_schedule["source_receipt_sha256"] = {
        system: _sha256(REPOSITORY / relative)
        for system, relative in frozen_protocol["source_receipts"].items()
    }
    return frozen_schedule, frozen_protocol


def freeze(schedule_path: Path) -> dict[str, Any]:
    schedule_path = schedule_path.resolve()
    schedule = _load(schedule_path)
    protocol_path = REPOSITORY / str(schedule["protocol_receipt"])
    protocol = _load(protocol_path)
    result = preflight(schedule_path, schedule, protocol_path, protocol)
    if not result["valid"]:
        return {"status": "refused", "preflight": result}

    validations = _run_validation()
    if not validations or any(item["returncode"] != 0 for item in validations):
        return {
            "status": "refused_validation",
            "preflight": result,
            "validations": validations,
        }

    frozen_at = datetime.now(timezone.utc).isoformat()
    frozen_schedule, frozen_protocol = _frozen_payloads(schedule, protocol, frozen_at)
    original_schedule = schedule_path.read_bytes()
    original_protocol = protocol_path.read_bytes()
    try:
        _atomic_write(protocol_path, _json_bytes(frozen_protocol))
        _atomic_write(schedule_path, _json_bytes(frozen_schedule))
        observed_schedule = _load(schedule_path)
        observed_schedule["_schedule_sha256"] = _sha256(schedule_path)
        observed_protocol = _load(protocol_path)
        gate = _protocol_gate(
            schedule_path,
            observed_schedule,
            protocol_path,
            observed_protocol,
        )
        if not gate["valid"]:
            raise RuntimeError(f"post-freeze protocol gate failed: {gate['checks']}")
    except BaseException:
        _atomic_write(protocol_path, original_protocol)
        _atomic_write(schedule_path, original_schedule)
        raise
    return {
        "status": "frozen",
        "frozen_at": frozen_at,
        "schedule_sha256": _sha256(schedule_path),
        "protocol_sha256": _sha256(protocol_path),
        "preflight": result,
        "validations": validations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="run validation and atomically freeze; default is read-only preflight",
    )
    args = parser.parse_args()
    schedule_path = args.schedule.resolve()
    if args.apply:
        result = freeze(schedule_path)
    else:
        schedule = _load(schedule_path)
        protocol_path = REPOSITORY / str(schedule["protocol_receipt"])
        result = preflight(schedule_path, schedule, protocol_path, _load(protocol_path))
        result = {"status": "ready" if result["valid"] else "not_ready", **result}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"ready", "frozen"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
