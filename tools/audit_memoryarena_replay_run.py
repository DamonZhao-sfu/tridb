"""Independently audit a completed gold-response MemoryArena live replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from bench.agent_memory.memoryarena.dataset import load_export
from bench.agent_memory.memoryarena.oracle import Arm
from bench.agent_memory.memoryarena.receipts import (
    SCHEMA_VERSION as RECEIPT_SCHEMA_VERSION,
    verify_serialized_receipt,
)
from tools.audit_memoryarena_agent_run import _distribution, _file_sha256, _ndcg


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-full", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def _receipt_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["arm"]),
        str(row["task_uid"]),
        str(row["target_session_uid"]),
    )


def _update_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["arm"]), str(row["task_uid"]), str(row["session_uid"])


def _index(
    rows: Sequence[Mapping[str, Any]], key: Any, label: str
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result = {}
    for row in rows:
        row_key = key(row)
        if row_key in result:
            raise ValueError(f"duplicate {label}: {row_key}")
        result[row_key] = row
    return result


def _word_budget_selection(
    ranked_uids: Sequence[str],
    sessions: Mapping[str, Any],
    *,
    top_k: int,
    budget: int,
) -> list[str]:
    accepted: list[str] = []
    for uid in ranked_uids[:top_k]:
        trial_uids = [*accepted, uid]
        trial = "\n\n".join(
            f"Question: {sessions[item].question}\nResponse: {sessions[item].gold_answer}"
            for item in trial_uids
        )
        if len(trial.split()) <= budget:
            accepted.append(uid)
    return accepted


def audit(export_dir: Path, run_dir: Path, *, require_full: bool) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    corpus = load_export(export_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks_run = int(summary["dataset"]["tasks_run"])
    if require_full and tasks_run != len(corpus.tasks):
        raise ValueError(f"not full corpus: {tasks_run} != {len(corpus.tasks)}")
    if summary["dataset"]["manifest_sha256"] != corpus.manifest_sha256:
        raise ValueError("dataset manifest mismatch")
    if summary.get("execution_schedule") != "task_outer_cyclic_arm_rotation":
        raise ValueError("run lacks task-level cyclic arm rotation")
    arms = tuple(arm.value for arm in Arm)
    if tuple(row["arm"] for row in summary["arms"]) != arms:
        raise ValueError("run is not the canonical six-arm order")

    tasks = corpus.tasks[:tasks_run]
    sessions = {
        session.session_uid: session for task in tasks for session in task.sessions
    }
    task_sessions = {
        task.task_uid: {session.session_uid for session in task.sessions}
        for task in tasks
    }
    receipts_path = run_dir / "receipts.jsonl"
    updates_path = run_dir / "updates.jsonl"
    receipts = _read_jsonl(receipts_path)
    updates = _read_jsonl(updates_path)
    receipt_by_key = _index(receipts, _receipt_key, "receipt")
    update_by_key = _index(updates, _update_key, "update")
    session_count = sum(len(task.sessions) for task in tasks)
    decision_count = sum(len(task.sessions) - 1 for task in tasks)
    if len(receipts) != decision_count * len(arms):
        raise ValueError("receipt completeness mismatch")
    if len(updates) != session_count * len(arms):
        raise ValueError("update completeness mismatch")

    violations: defaultdict[str, int] = defaultdict(int)
    metrics: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    update_metrics: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    oracle_queries = 0
    for task in tasks:
        for arm in arms:
            for session in task.sessions:
                key = (arm, task.task_uid, session.session_uid)
                update = update_by_key.get(key)
                if update is None:
                    raise ValueError(f"missing update: {key}")
                if not bool(update["committed"]):
                    violations["uncommitted_update"] += 1
                if arm == Arm.MEMORY_OFF.value and not (
                    update["delta"] is None
                    and update["cost"] is None
                    and int(update["linked_edges"]) == 0
                    and int(update["wal_bytes_interval"]) == 0
                    and float(update["update_ms"]) == 0.0
                ):
                    violations["memory_off_write"] += 1
                update_metrics[arm].append(update)
                if session.ordinal == 0:
                    if key in receipt_by_key:
                        violations["initial_receipt"] += 1
                    continue
                receipt = receipt_by_key.get(key)
                if receipt is None:
                    raise ValueError(f"missing receipt: {key}")
                if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION:
                    violations["receipt_schema"] += 1
                if not verify_serialized_receipt(receipt):
                    violations["receipt_digest"] += 1
                if receipt["dataset_manifest_sha256"] != corpus.manifest_sha256:
                    violations["manifest"] += 1
                if int(receipt["cutoff_ordinal"]) != session.ordinal:
                    violations["cutoff"] += 1
                if (
                    receipt["query_sha256"]
                    != hashlib.sha256(session.question.encode()).hexdigest()
                ):
                    violations["query_hash"] += 1
                if int(receipt["injection_tokens"]) > int(
                    receipt["injection_token_budget"]
                ):
                    violations["token_budget"] += 1
                selected = [str(uid) for uid in receipt["selected_session_uids"]]
                if len(selected) > int(receipt["top_k"]):
                    violations["top_k"] += 1
                candidates = {
                    str(candidate["session_uid"]): candidate
                    for candidate in receipt["candidates"]
                }
                if len(candidates) != len(receipt["candidates"]):
                    violations["duplicate_candidate"] += 1
                if set(candidates) != task_sessions[task.task_uid]:
                    violations["candidate_universe"] += 1
                eligible = frozenset(
                    uid
                    for uid, candidate in candidates.items()
                    if str(candidate["task_uid"]) == task.task_uid
                    and uid != session.session_uid
                    and int(candidate["ordinal"]) < session.ordinal
                    and bool(candidate["scope_allowed"])
                )
                expected_eligible = frozenset(session.protocol_dependencies)
                if eligible != expected_eligible:
                    violations["exact_eligibility_parity"] += 1
                if any(uid not in expected_eligible for uid in selected):
                    violations["future_target_cross_task_scope_leakage"] += 1
                if len(selected) != len(set(selected)):
                    violations["duplicate_selected"] += 1
                for rank, uid in enumerate(selected, 1):
                    if candidates[uid]["selected_rank"] != rank:
                        violations["rank"] += 1

                injection = "\n\n".join(
                    f"Question: {sessions[uid].question}\nResponse: {sessions[uid].gold_answer}"
                    for uid in selected
                )
                if (
                    receipt["injection_sha256"]
                    != hashlib.sha256(injection.encode()).hexdigest()
                ):
                    violations["injection_hash"] += 1
                if arm == Arm.MEMORY_OFF.value and selected:
                    violations["memory_off_selected"] += 1
                if arm == Arm.RECENT_FIFO.value:
                    ranked = sorted(
                        expected_eligible,
                        key=lambda uid: (-sessions[uid].ordinal, uid),
                    )
                    expected = _word_budget_selection(
                        ranked,
                        sessions,
                        top_k=int(receipt["top_k"]),
                        budget=int(receipt["injection_token_budget"]),
                    )
                    if selected != expected:
                        violations["recent_parity"] += 1
                if arm == Arm.ORACLE.value:
                    ranked = sorted(
                        expected_eligible, key=lambda uid: sessions[uid].ordinal
                    )
                    expected = _word_budget_selection(
                        ranked,
                        sessions,
                        top_k=int(receipt["top_k"]),
                        budget=int(receipt["injection_token_budget"]),
                    )
                    if selected != expected:
                        violations["oracle_parity"] += 1
                    oracle_queries += 1
                actual_scores = [
                    candidate
                    for candidate in candidates.values()
                    if candidate["semantic_score_source"]
                    == "pgvector_cosine_similarity_selected_top_k"
                ]
                if (
                    arm not in (Arm.VECTOR_ONLY.value, Arm.GEM_FUSED.value)
                    and actual_scores
                ):
                    violations["score_provenance_wrong_arm"] += len(actual_scores)
                if any(
                    not -1.000001 <= float(candidate["semantic_score"]) <= 1.000001
                    for candidate in actual_scores
                ):
                    violations["score_bounds"] += 1
                relevant = expected_eligible
                valid = [
                    bool(candidates[uid]["is_valid"])
                    and not bool(candidates[uid]["is_stale"])
                    and bool(candidates[uid]["scope_allowed"])
                    for uid in selected
                ]
                metrics[arm].append(
                    {
                        "recall": len(set(selected[:10]) & relevant) / len(relevant),
                        "ndcg": _ndcg(selected, relevant),
                        "constraint_valid": statistics.fmean(valid) if valid else 1.0,
                        "first_row_ms": receipt["first_row_ms"],
                        "time_to_k_ms": float(receipt["time_to_k_ms"]),
                        "candidates_examined": receipt["candidates_examined"],
                        "visited_nodes": receipt["visited_nodes"],
                        "visited_edges": receipt["visited_edges"],
                        "returned": len(selected),
                    }
                )

    nonzero = {name: count for name, count in sorted(violations.items()) if count}
    if nonzero:
        raise ValueError("audit violations: " + json.dumps(nonzero, sort_keys=True))
    arm_reports = []
    for arm in arms:
        rows = metrics[arm]
        writes = update_metrics[arm]
        update_ms = [
            float(row["update_ms"]) for row in writes if float(row["update_ms"]) > 0
        ]
        returned_over_examined = [
            float(row["returned"]) / float(row["candidates_examined"])
            for row in rows
            if row["candidates_examined"] not in (None, 0)
        ]
        arm_reports.append(
            {
                "arm": arm,
                "decisions": len(rows),
                "dependency_recall_at_10": _distribution(
                    [float(row["recall"]) for row in rows]
                ),
                "ndcg_at_10": _distribution([float(row["ndcg"]) for row in rows]),
                "harmful_at_10": None,
                "stale_at_10": None,
                "annotation_note": "N/A: public release has no harmful/stale labels",
                "constraint_valid_fraction": _distribution(
                    [float(row["constraint_valid"]) for row in rows]
                ),
                "time_to_first_row_ms": _distribution(
                    [
                        float(row["first_row_ms"])
                        for row in rows
                        if row["first_row_ms"] is not None
                    ]
                ),
                "time_to_k_ms": _distribution(
                    [float(row["time_to_k_ms"]) for row in rows]
                ),
                "candidates_examined": _distribution(
                    [
                        float(row["candidates_examined"])
                        for row in rows
                        if row["candidates_examined"] is not None
                    ]
                ),
                "returned_over_candidates_examined": _distribution(
                    returned_over_examined
                ),
                "candidate_contraction_fraction": _distribution(
                    [1.0 - ratio for ratio in returned_over_examined]
                ),
                "visited_nodes": _distribution(
                    [
                        float(row["visited_nodes"])
                        for row in rows
                        if row["visited_nodes"] is not None
                    ]
                ),
                "visited_nodes_note": "N/A for fused: no distinct-node probe",
                "visited_edges": _distribution(
                    [
                        float(row["visited_edges"])
                        for row in rows
                        if row["visited_edges"] is not None
                    ]
                ),
                "visited_edges_note": "tjs_open_graph_examined edge-step counter",
                "update_ms": _distribution(update_ms),
                "update_throughput_per_second": len(update_ms) * 1000.0 / sum(update_ms)
                if update_ms
                else None,
                "wal_bytes_interval": _distribution(
                    [
                        float(row["wal_bytes_interval"])
                        for row in writes
                        if arm != Arm.MEMORY_OFF.value
                    ]
                ),
            }
        )
    report = {
        "schema_version": "memoryarena_replay_independent_audit_v0.1.0",
        "status": "pass",
        "run_id": summary["run_id"],
        "dataset": {
            "config": corpus.config,
            "manifest_sha256": corpus.manifest_sha256,
            "tasks": tasks_run,
            "sessions_per_arm": session_count,
            "decisions_per_arm": decision_count,
        },
        "gates": {
            "receipt_digest_violations": 0,
            "exact_eligibility_parity_queries": len(receipts),
            "exact_eligibility_parity_violations": 0,
            "exact_oracle_selection_parity_queries": oracle_queries,
            "exact_oracle_selection_parity_violations": 0,
            "future_target_cross_task_scope_leakage": 0,
            "memory_off_write_violations": 0,
            "token_or_top_k_violations": 0,
        },
        "retrieval_and_system_metrics": arm_reports,
        "claim_boundary": "gold-response database-path replay; no generated agent outcome",
        "implementation": {
            str(path.relative_to(repository)): _file_sha256(path)
            for path in (
                repository / "tools/pilot_memoryarena_tridb.py",
                repository / "bench/agent_memory/memoryarena/tridb_backend.py",
                repository / "bench/agent_memory/memoryarena/cross_session.py",
                repository / "bench/agent_memory/memoryarena/receipts.py",
                Path(__file__).resolve(),
            )
        },
        "artifacts": {
            "summary": {
                "path": summary_path.name,
                "sha256": _file_sha256(summary_path),
            },
            "receipts": {
                "path": receipts_path.name,
                "sha256": _file_sha256(receipts_path),
            },
            "updates": {
                "path": updates_path.name,
                "sha256": _file_sha256(updates_path),
            },
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit(args.export_dir, args.run_dir, require_full=args.require_full)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(
        f"PASS tasks={report['dataset']['tasks']} "
        f"eligibility={report['gates']['exact_eligibility_parity_queries']} leakage=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
