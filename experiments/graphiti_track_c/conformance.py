"""Two-phase live conformance for Graphiti model, isolation, and persistence."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.agent_memory.table5_track_c.dataset import EventItem, QueryItem, load_locomo
from bench.agent_memory.table5_track_c.protocol import (
    _base_receipt,
    add_receipt_contract,
    search_receipt_contract,
)

from .adapter import GraphitiTrackCAdapter, GraphitiTrackCConfig

DEFAULT_DATASET = (
    "/localhome/hza214/Mandol/experimental/self_host_benchmarks/locomo/"
    "data/locomo10.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _event(
    sample_id: str, event_id: str, turn: int, speaker: str, statement: str
) -> EventItem:
    return EventItem(
        sample_id=sample_id,
        event_id=event_id,
        session_id="S1",
        timestamp="9:00 am on 1 January, 2026",
        role=speaker,
        text=f'(9:00 am on 1 January, 2026) {speaker} said, "{statement}"',
        ordinal=1_000_000 + turn,
        metadata={"source": "synthetic_conformance", "turn_number": turn},
    )


EVENTS = (
    _event(
        "conformance_a",
        "A:1",
        1,
        "Asteria Northwind",
        "Asteria Northwind mentors Celeste Eastwind.",
    ),
    _event(
        "conformance_a",
        "A:2",
        2,
        "Celeste Eastwind",
        "Celeste Eastwind reviews database designs with Asteria Northwind.",
    ),
    _event(
        "conformance_b",
        "B:1",
        1,
        "Borealis Southwind",
        "Borealis Southwind mentors Dorian Westwind.",
    ),
    _event(
        "conformance_b",
        "B:2",
        2,
        "Dorian Westwind",
        "Dorian Westwind reviews interface designs with Borealis Southwind.",
    ),
)

QUERIES = (
    QueryItem(
        sample_id="conformance_a",
        question_id="A:q1",
        question="Who does Asteria Northwind mentor?",
        answer=None,
        category=None,
        evidence_ids=(),
        ordinal=0,
    ),
    QueryItem(
        sample_id="conformance_b",
        question_id="B:q1",
        question="Who does Borealis Southwind mentor?",
        answer=None,
        category=None,
        evidence_ids=(),
        ordinal=0,
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("write", "read"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:27687")
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument(
        "--embedding-model", default="Qwen/Qwen3-Embedding-0.6B"
    )
    parser.add_argument(
        "--require-answer-endpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--neo4j-invocation-id", required=True)
    parser.add_argument("--neo4j-main-pid", required=True, type=int)
    return parser


def _adapter(args: argparse.Namespace) -> GraphitiTrackCAdapter:
    return GraphitiTrackCAdapter(
        GraphitiTrackCConfig(
            dataset_path=args.dataset,
            neo4j_state_dir=args.state_dir,
            neo4j_uri=args.neo4j_uri,
            answer_base_url=args.answer_base_url,
            answer_model=args.answer_model,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            require_answer_endpoint=args.require_answer_endpoint,
        )
    )


def _search_and_gate(adapter: GraphitiTrackCAdapter) -> list[dict[str, Any]]:
    results = [adapter.search(query, top_k=10) for query in QUERIES]
    checks = {
        "search_receipt_contract": search_receipt_contract(
            [{"success": True, "receipt": result} for result in results]
        ),
        "group_a_nonempty": results[0]["result_count"] > 0,
        "group_b_nonempty": results[1]["result_count"] > 0,
        "group_a_excludes_b": "Borealis" not in " ".join(results[0]["contexts"])
        and "Dorian" not in " ".join(results[0]["contexts"]),
        "group_b_excludes_a": "Asteria" not in " ".join(results[1]["contexts"])
        and "Celeste" not in " ".join(results[1]["contexts"]),
        "generation_llm_calls_zero": all(
            result.get("llm_call_count") == 0 for result in results
        ),
        "cross_encoder_calls_zero": all(
            result.get("cross_encoder_call_count") == 0 for result in results
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Graphiti Search/isolation checks failed: {failed}")
    return [
        {
            "sample_id": query.sample_id,
            "question": query.question,
            "result": result,
        }
        for query, result in zip(QUERIES, results, strict=True)
    ]


def _gate_group_counts(counts: dict[str, dict[str, int]]) -> None:
    expected = {"conformance_a", "conformance_b"}
    observed = set(counts)
    if observed != expected:
        raise RuntimeError(
            f"Graphiti group isolation mismatch: expected={expected}, observed={observed}"
        )
    for group_id in expected:
        if counts[group_id]["nodes"] <= 0 or counts[group_id]["relationships"] <= 0:
            raise RuntimeError(
                f"Graphiti group is empty: {group_id}={counts[group_id]}"
            )


def _runtime_identity(invocation_id: str, main_pid: int) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None:
        raise RuntimeError(f"invalid Neo4j systemd InvocationID: {invocation_id!r}")
    if main_pid <= 0:
        raise RuntimeError(f"invalid Neo4j MainPID: {main_pid}")
    return {"systemd_invocation_id": invocation_id, "main_pid": main_pid}


def _gate_runtime_restart(before: dict[str, Any], after: dict[str, Any]) -> None:
    if before.get("systemd_invocation_id") == after.get("systemd_invocation_id"):
        raise RuntimeError("Neo4j systemd InvocationID did not change across restart")


def main() -> int:
    args = _parser().parse_args()
    receipt_path = Path(args.receipt).resolve()
    if args.phase == "write" and receipt_path.exists():
        raise FileExistsError(f"refusing existing conformance receipt: {receipt_path}")
    if args.phase == "read" and not receipt_path.is_file():
        raise FileNotFoundError(f"write-phase receipt is absent: {receipt_path}")
    adapter = _adapter(args)
    runtime_identity = _runtime_identity(args.neo4j_invocation_id, args.neo4j_main_pid)
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path.is_file()
        else {
            **_base_receipt(
                corpus=load_locomo(args.dataset, verify=True),
                adapter=adapter,
                build_id="graphiti_zep_oss_proxy_conformance",
                phase="conformance",
            ),
            "schema_version": "graphiti_track_c_conformance_v0.3.0",
            "display_label": "Graphiti (Zep OSS proxy)",
            "interpretation_boundary": "not production Zep",
            "state_dir": str(Path(args.state_dir).resolve()),
        }
    )
    try:
        schema_gate = adapter.init_schema()
        if args.phase == "write":
            if schema_gate["live_counts"]["nodes"] != 0:
                raise RuntimeError("Graphiti conformance write state is not empty")
            additions = [adapter.add(item, visibility=False) for item in EVENTS]
            if not add_receipt_contract(
                [{"success": True, "receipt": result} for result in additions]
            ):
                raise RuntimeError("Graphiti Add creation receipt contract failed")
            if not all(result.get("llm_call_count") == 0 for result in additions):
                raise RuntimeError("Graphiti prepared Add invoked a forbidden LLM")
            search_results = _search_and_gate(adapter)
            group_counts = adapter.group_counts()
            _gate_group_counts(group_counts)
            receipt["write_phase"] = {
                "completed_at": _now(),
                "schema_gate": schema_gate,
                "events": len(EVENTS),
                "additions": additions,
                "search_results": search_results,
                "counts": adapter.stats()["live_counts"],
                "group_counts": group_counts,
                "neo4j_runtime": runtime_identity,
                "llm_conformance": {
                    "policy": "forbidden_fail_closed",
                    "add_requests_checked": len(additions),
                    "search_requests_checked": len(search_results),
                    "llm_call_count": 0,
                    "cross_encoder_call_count": 0,
                    "passed": True,
                },
            }
            receipt["status"] = "write_complete_restart_required"
        else:
            if receipt.get("status") != "write_complete_restart_required":
                raise RuntimeError("conformance receipt is not ready for read phase")
            _gate_runtime_restart(
                receipt["write_phase"]["neo4j_runtime"], runtime_identity
            )
            search_results = _search_and_gate(adapter)
            observed_counts = adapter.stats()["live_counts"]
            expected_counts = receipt["write_phase"]["counts"]
            if observed_counts != expected_counts:
                raise RuntimeError(
                    "Graphiti persistence count mismatch: "
                    f"expected={expected_counts}, observed={observed_counts}"
                )
            group_counts = adapter.group_counts()
            _gate_group_counts(group_counts)
            if group_counts != receipt["write_phase"]["group_counts"]:
                raise RuntimeError("Graphiti per-group counts changed after restart")
            receipt["read_phase"] = {
                "completed_at": _now(),
                "schema_gate": schema_gate,
                "search_results": search_results,
                "counts": observed_counts,
                "group_counts": group_counts,
                "neo4j_runtime": runtime_identity,
                "llm_conformance": {
                    "policy": "forbidden_fail_closed",
                    "search_requests_checked": len(search_results),
                    "llm_call_count": 0,
                    "cross_encoder_call_count": 0,
                    "passed": True,
                },
            }
            receipt["status"] = "complete"
            receipt["passed"] = True
            receipt["completed_at"] = _now()
        _write(receipt_path, receipt)
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["passed"] = False
        receipt["failed_at"] = _now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        _write(receipt_path, receipt)
        raise
    finally:
        adapter.close()
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
