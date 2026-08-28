"""Create a fail-closed TriDB architecture receipt for Track C.

The latency harness alone cannot prove that the measured TriDB/GEM point used
the architecture claimed by the system: a native adjacency store, the fused
``tjs_open`` path, and an Open/Next/Close iterator with early termination.  This
module binds structural source checks, live source/runtime identity, and the
fresh conformance probes to the frozen completion-v2 schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import benchmark_code_sha256


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_SCHEDULE = (
    REPOSITORY
    / "bench/agent_memory/table5_track_c/manifests/completion_v2_schedule.json"
)

INVARIANT_CHECK_NAMES = (
    "schedule_frozen",
    "protocol_frozen",
    "schedule_protocol_hash_bound",
    "benchmark_code_hash_bound",
    "source_runtime_identity_valid",
    "source_manifest_hash_bound",
    "conformance_status_passed",
    "conformance_execution_hashes_bound",
    "conformance_build_uses_native_topology",
    "conformance_restart_native_topology_persistent",
    "conformance_five_fused_topic_searches",
    "conformance_relaxed_order",
    "conformance_stream_work_observed",
    "conformance_termination_disclosed",
    "adapter_single_postgres_dsn",
    "adapter_forces_fused_topic",
    "adapter_native_graph_backend",
    "adapter_declares_gem_edge_metadata_only",
    "retrieve_calls_canonical_tjs_open",
    "retrieve_records_open_next_close_and_early_termination",
    "retrieve_collects_only_bounded_output",
    "retrieve_reads_operator_work_probes",
    "tjs_operator_bans_materializing_bfs",
    "tjs_operator_owns_early_termination",
    "graph_iterator_open_next_close",
    "graph_iterator_one_edge_per_next",
    "graph_iterator_bounded_to_one_adjacency_page",
    "graph_topology_written_through_native_function",
    "graph_metadata_not_used_as_topology",
    "graph_uses_postgres_generic_wal",
    "stock_pg_not_gx10_claim",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_files(repository: Path) -> dict[str, Path]:
    return {
        "adapter": repository
        / "bench/agent_memory/table5_track_c/adapters/tridb_gem.py",
        "retrieve": repository / "bench/agent_memory/gem/retrieve.py",
        "store": repository / "bench/agent_memory/gem/store.py",
        "graph_header": repository / "src/graph_store/graphstore.h",
        "graph_am": repository / "src/graph_store/graph_am.c",
        "tjs_pg": repository / "src/tjs_pg/tjs_pg.c",
    }


def _static_checks(repository: Path) -> tuple[dict[str, bool], dict[str, str | None]]:
    paths = _source_files(repository)
    text = {
        name: path.read_text(encoding="utf-8") if path.is_file() else ""
        for name, path in paths.items()
    }
    tjs_c_files = sorted((repository / "src/tjs_pg").glob("*.c"))
    tjs_text = "\n".join(path.read_text(encoding="utf-8") for path in tjs_c_files)
    adapter = text["adapter"]
    retrieve = text["retrieve"]
    store = text["store"]
    graph_header = text["graph_header"]
    graph_am = text["graph_am"]
    tjs_pg = text["tjs_pg"]
    checks = {
        "adapter_single_postgres_dsn": (
            "dsn: str" in adapter
            and "TriDBGovernedMemory.connect(\n            config.dsn" in adapter
            and "neo4j" not in adapter.lower()
            and "graph_base_url" not in adapter
        ),
        "adapter_forces_fused_topic": (
            "mode=RetrievalMode.FUSED" in adapter
            and "route=RetrievalRoute.TOPIC" in adapter
        ),
        "adapter_native_graph_backend": (
            'backend="graph_store_am"' in adapter
            and "_link_session_chains" in adapter
            and "_link_new_unit" in adapter
        ),
        "adapter_declares_gem_edge_metadata_only": (
            "``gem_edge`` contains only\nedge metadata" in adapter
        ),
        "retrieve_calls_canonical_tjs_open": (
            "SELECT t FROM tjs_open(" in retrieve
            and '"fusion",\n            "tridb.tjs_open"' in retrieve
        ),
        "retrieve_records_open_next_close_and_early_termination": (
            '"iterator_contract": "Open/Next/Close"' in retrieve
            and '"early_termination": True' in retrieve
        ),
        "retrieve_collects_only_bounded_output": (
            "Only its bounded returned top-k ids are collected" in retrieve
            and "no full candidate or reach intermediate is" in retrieve
        ),
        "retrieve_reads_operator_work_probes": all(
            value in retrieve
            for value in (
                "tjs_open_candidates_examined()",
                "tjs_open_graph_examined()",
                "tjs_open_graph_censored()",
                "tjs_open_termination_reason()",
                "tjs_open_budget_capped()",
                "tjs_open_bridges_injected()",
            )
        ),
        "tjs_operator_bans_materializing_bfs": bool(tjs_c_files)
        and "gph_traverse_bfs" not in tjs_text,
        "tjs_operator_owns_early_termination": (
            "break;\t\t/* TR-1 early termination: we own the loop, just stop */"
            in tjs_pg
            and 'strcmp(iter, "relaxed_order")' in tjs_pg
        ),
        "graph_iterator_open_next_close": all(
            value in graph_header and value in graph_am
            for value in ("gs_open", "gs_getnext", "gs_close")
        ),
        "graph_iterator_one_edge_per_next": (
            "one EDGE per call" in graph_header
            and "advance the scan by ONE visible" in graph_am
        ),
        "graph_iterator_bounded_to_one_adjacency_page": (
            "exactly one page is ever\n\t * buffered (streaming)" in graph_am
            and "stops before later chain pages are ever read" in graph_am
        ),
        "graph_topology_written_through_native_function": (
            "SELECT graph_store.gph_insert_edge(%s, %s, %s)" in store
        ),
        "graph_metadata_not_used_as_topology": (
            "Write one typed edge: topology in the AM, metadata in ``gem_edge``"
            in store
            and "INSERT INTO gem_edge" in store
        ),
        "graph_uses_postgres_generic_wal": (
            "GenericXLogStart(rel)" in graph_am
            and "GenericXLogRegisterBuffer" in graph_am
            and "GenericXLogFinish" in graph_am
            and "No private buffer pool, no second WAL" in graph_am
        ),
    }
    hashes = {
        str(path.relative_to(repository)): _sha256(path) for path in paths.values()
    }
    return checks, hashes


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _conformance_checks(
    receipt: dict[str, Any], schedule: dict[str, Any], schedule_sha: str | None
) -> dict[str, bool]:
    execution = receipt.get("execution") or {}
    build = receipt.get("build") or {}
    finalized = receipt.get("build_finalize") or {}
    before = receipt.get("stats_before_restart") or {}
    after = receipt.get("stats_after_restart") or {}
    searches = receipt.get("searches") or []
    probes = [value.get("probes") or {} for value in searches]
    allowed_termination = {"term_cond", "scan_budget", "stream_end_unknown"}
    return {
        "conformance_status_passed": receipt.get("status") == "passed"
        and receipt.get("system") == "tridb_gem",
        "conformance_execution_hashes_bound": (
            execution.get("benchmark_code_sha256")
            == schedule.get("benchmark_code_sha256")
            and execution.get("formal_schedule_sha256") == schedule_sha
            and execution.get("protocol_receipt_sha256")
            == schedule.get("protocol_receipt_sha256")
        ),
        "conformance_build_uses_native_topology": (
            build.get("events") == 58
            and build.get("units_created") == 58
            and _nonnegative_int(build.get("edges_created"))
            and build["edges_created"] > 0
            and build.get("rejected") == []
            and finalized.get("units") == 58
            and finalized.get("visible_edges") == build.get("edges_created")
            and before.get("native_vertices") == before.get("units")
            and before.get("native_visible_edges") == before.get("edge_metadata_rows")
        ),
        "conformance_restart_native_topology_persistent": (
            receipt.get("restart_visible") is True
            and receipt.get("scope_isolated") is True
            and after.get("native_vertices") == before.get("native_vertices")
            and after.get("native_visible_edges") == before.get("native_visible_edges")
            and after.get("native_visible_edges") == after.get("edge_metadata_rows")
        ),
        "conformance_five_fused_topic_searches": len(searches) == 5
        and all(
            probe.get("mode") == "fused" and probe.get("route") == "topic"
            for probe in probes
        ),
        "conformance_relaxed_order": len(probes) == 5
        and all(
            probe.get("hnsw_iterative_scan") == "relaxed_order" for probe in probes
        ),
        "conformance_stream_work_observed": len(probes) == 5
        and all(
            _nonnegative_int(probe.get("candidates_examined"))
            and probe["candidates_examined"] > 0
            and _nonnegative_int(probe.get("graph_examined"))
            for probe in probes
        ),
        "conformance_termination_disclosed": len(probes) == 5
        and all(
            probe.get("termination_reason") in allowed_termination for probe in probes
        ),
    }


def _live_source_audit(schedule_path: Path) -> dict[str, Any]:
    command = [
        str(REPOSITORY / ".venv/bin/python"),
        str(REPOSITORY / "tools/table5_track_c_goal_audit.py"),
        "--schedule",
        str(schedule_path),
        "--source",
        "tridb_gem",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {
            "valid": False,
            "error": "source auditor did not emit JSON",
            "stdout_tail": completed.stdout[-1_000:],
        }
    result["command"] = command
    result["returncode"] = completed.returncode
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-1_000:]
    return result


def build_receipt(
    schedule_path: Path, *, source_audit: dict[str, Any] | None = None
) -> dict[str, Any]:
    schedule_path = schedule_path.resolve()
    schedule = _load(schedule_path)
    protocol_path = REPOSITORY / str(schedule.get("protocol_receipt") or "")
    protocol = _load(protocol_path)
    root = REPOSITORY / str(protocol.get("result_root") or "")
    conformance_path = root / "conformance/tridb_gem.json"
    conformance = _load(conformance_path) if conformance_path.is_file() else {}
    schedule_sha = _sha256(schedule_path)
    protocol_sha = _sha256(protocol_path)
    static_checks, source_hashes = _static_checks(REPOSITORY)
    runtime = (
        source_audit if source_audit is not None else _live_source_audit(schedule_path)
    )
    source_manifest = REPOSITORY / str(
        (protocol.get("source_receipts") or {}).get("tridb_gem") or ""
    )
    checks: dict[str, bool] = {
        "schedule_frozen": schedule.get("status") == "frozen",
        "protocol_frozen": protocol.get("status") == "frozen",
        "schedule_protocol_hash_bound": (
            schedule.get("protocol_receipt_sha256") == protocol_sha
        ),
        "benchmark_code_hash_bound": (
            schedule.get("benchmark_code_sha256") == benchmark_code_sha256()
        ),
        "source_runtime_identity_valid": runtime.get("valid") is True
        and runtime.get("returncode", 0) == 0,
        "source_manifest_hash_bound": (
            _sha256(source_manifest)
            == (schedule.get("source_receipt_sha256") or {}).get("tridb_gem")
        ),
        **_conformance_checks(conformance, schedule, schedule_sha),
        **static_checks,
        "stock_pg_not_gx10_claim": (
            (protocol.get("hardware_claim") or {}).get("gx10_signoff") is False
            and (protocol.get("hardware_claim") or {}).get(
                "h800_numerical_reproduction"
            )
            is False
            and (protocol.get("source_identity") or {})
            .get("tridb_gem", {})
            .get("postgres")
            == "16.14"
        ),
    }
    if set(checks) != set(INVARIANT_CHECK_NAMES):
        raise RuntimeError(
            "TriDB invariant implementation/check-name drift: "
            f"missing={sorted(set(INVARIANT_CHECK_NAMES) - set(checks))}, "
            f"extra={sorted(set(checks) - set(INVARIANT_CHECK_NAMES))}"
        )
    claims = {
        "tr1_open_next_close": all(
            checks[name]
            for name in (
                "retrieve_records_open_next_close_and_early_termination",
                "retrieve_collects_only_bounded_output",
                "tjs_operator_bans_materializing_bfs",
                "tjs_operator_owns_early_termination",
                "graph_iterator_open_next_close",
                "graph_iterator_one_edge_per_next",
                "graph_iterator_bounded_to_one_adjacency_page",
                "conformance_stream_work_observed",
                "conformance_termination_disclosed",
            )
        ),
        "native_graph": all(
            checks[name]
            for name in (
                "adapter_native_graph_backend",
                "graph_topology_written_through_native_function",
                "graph_metadata_not_used_as_topology",
                "conformance_build_uses_native_topology",
                "conformance_restart_native_topology_persistent",
            )
        ),
        "same_postgres_process": all(
            checks[name]
            for name in (
                "adapter_single_postgres_dsn",
                "source_runtime_identity_valid",
                "retrieve_calls_canonical_tjs_open",
                "graph_topology_written_through_native_function",
            )
        ),
        "one_postgres_wal": all(
            checks[name]
            for name in (
                "adapter_single_postgres_dsn",
                "source_runtime_identity_valid",
                "graph_uses_postgres_generic_wal",
            )
        ),
        "full_intermediate_materialization": False,
        "paper_h800_match": False,
        "gx10_signoff": False,
    }
    passed = all(checks.values()) and all(
        value is True
        for name, value in claims.items()
        if name
        not in {
            "full_intermediate_materialization",
            "paper_h800_match",
            "gx10_signoff",
        }
    )
    return {
        "schema_version": "table5_track_c_tridb_invariant_v0.1.0",
        "status": "passed" if passed else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schedule": str(schedule_path),
        "schedule_sha256": schedule_sha,
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "benchmark_code_sha256": benchmark_code_sha256(),
        "conformance_receipt": str(conformance_path.resolve()),
        "conformance_receipt_sha256": _sha256(conformance_path),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "source_files_sha256": source_hashes,
        "source_runtime_audit": runtime,
        "checks": checks,
        "claims": claims,
        "interpretation": (
            "This receipt proves the frozen stock-PostgreSQL Track C path and its "
            "fresh conformance behavior. It is not an H800 numerical reproduction "
            "or a GX10 build/sign-off."
        ),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = build_receipt(args.schedule)
    schedule = _load(args.schedule.resolve())
    protocol = _load(REPOSITORY / schedule["protocol_receipt"])
    output = args.output or (
        REPOSITORY / protocol["result_root"] / "tridb_invariant_receipt.json"
    )
    _atomic_write(output.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
