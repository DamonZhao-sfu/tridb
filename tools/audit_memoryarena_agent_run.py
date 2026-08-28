"""Independently audit and summarize a completed MemoryArena agent run.

The runner is intentionally not imported: this verifier rebuilds expected task/session
keys from the pinned export and checks the JSONL evidence without trusting its summary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import random
from typing import Any, Iterable, Mapping, Sequence

import psycopg

from bench.agent_memory.memoryarena.dataset import load_export
from bench.agent_memory.memoryarena.oracle import Arm
from bench.agent_memory.memoryarena.receipts import (
    SCHEMA_VERSION as RECEIPT_SCHEMA_VERSION,
    verify_serialized_receipt,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--answer-revision", required=True)
    parser.add_argument("--embedding-revision", required=True)
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[float]) -> dict[str, int | float | None]:
    materialized = list(values)
    return {
        "n": len(materialized),
        "mean": statistics.fmean(materialized) if materialized else None,
        "p50": _percentile(materialized, 0.50),
        "p95": _percentile(materialized, 0.95),
        "p99": _percentile(materialized, 0.99),
    }


def _ndcg(selected: Sequence[str], relevant: frozenset[str], k: int = 10) -> float:
    gains = [
        1.0 / math.log2(rank + 2)
        for rank, uid in enumerate(selected[:k])
        if uid in relevant
    ]
    ideal_count = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return sum(gains) / ideal if ideal else 0.0


def _task_outcome_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    ordered = sorted(
        (row for row in rows if bool(row["evaluated"])),
        key=lambda row: int(row["ordinal"]),
    )
    if not ordered:
        raise ValueError("task has no evaluated agent outcomes")
    correct = [float(bool(row["exact_match"])) for row in ordered]
    cumulative = [
        statistics.fmean(correct[: index + 1]) for index in range(len(correct))
    ]
    return {
        "cross_session_accuracy": statistics.fmean(correct),
        "final_session_success": correct[-1],
        "all_sessions_success": float(all(correct)),
        "progress_aulc": statistics.fmean(cumulative),
        "prompt_tokens": float(
            sum(int(row["usage"].get("prompt_tokens", 0)) for row in ordered)
        ),
        "completion_tokens": float(
            sum(int(row["usage"].get("completion_tokens", 0)) for row in ordered)
        ),
        "steps": float(len(ordered)),
        "pre_update_time_to_target_ms": float(
            sum(float(row["effective_total_ms"]) for row in ordered)
        ),
        "time_to_target_ms": float(
            sum(
                float(row["effective_total_ms"]) + float(row["_update_ms"])
                for row in ordered
            )
        ),
    }


def _paired_bootstrap(
    values: Mapping[str, Mapping[str, float]],
    baseline: Mapping[str, Mapping[str, float]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    task_ids = sorted(set(values) & set(baseline))
    deltas = [values[task][metric] - baseline[task][metric] for task in task_ids]
    generator = random.Random(seed)
    samples = [
        statistics.fmean(deltas[generator.randrange(len(deltas))] for _ in deltas)
        for _ in range(iterations)
    ]
    return {
        "n_tasks": len(task_ids),
        "mean_delta": statistics.fmean(deltas),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "positive_transfer_fraction": sum(delta > 0 for delta in deltas) / len(deltas),
        "negative_transfer_fraction": sum(delta < 0 for delta in deltas) / len(deltas),
        "zero_transfer_fraction": sum(delta == 0 for delta in deltas) / len(deltas),
        "method": "paired_task_cluster_percentile_bootstrap",
        "iterations": iterations,
        "seed": seed,
    }


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    session_uid = (
        row["session_uid"] if "session_uid" in row else row["target_session_uid"]
    )
    return str(row["arm"]), str(row["task_uid"]), str(session_uid)


def _unique(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"duplicate {label} key: {key}")
        indexed[key] = row
    return indexed


def _assert_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise ValueError(f"{label}: {observed!r} != {expected!r}")


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}: {observed!r} != {expected!r}")


def audit(
    export_dir: Path,
    run_dir: Path,
    *,
    dsn: str,
    require_full: bool,
    answer_revision: str,
    embedding_revision: str,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    corpus = load_export(export_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks_run = int(summary["dataset"]["tasks_run"])
    if require_full:
        _assert_equal(tasks_run, len(corpus.tasks), "full-run task count")
    tasks = corpus.tasks[:tasks_run]
    session_by_uid = {
        session.session_uid: session for task in tasks for session in task.sessions
    }
    arms = tuple(arm.value for arm in Arm)
    _assert_equal(tuple(summary["protocol"]["arms"]), arms, "six-arm order")
    _assert_equal(
        summary["dataset"]["manifest_sha256"],
        corpus.manifest_sha256,
        "dataset manifest",
    )
    if not bool(summary["protocol"].get("seed_first_session_with_gold")):
        raise ValueError("expected fixed gold source-session admission protocol")

    paths = {
        "outcomes": run_dir / "outcomes.jsonl",
        "receipts": run_dir / "receipts.jsonl",
        "updates": run_dir / "updates.jsonl",
        "preflight": run_dir / "provenance_preflight.json",
    }
    outcomes = _read_jsonl(paths["outcomes"])
    receipts = _read_jsonl(paths["receipts"])
    updates = _read_jsonl(paths["updates"])
    outcome_by_key = _unique(outcomes, "outcome")
    receipt_by_key = _unique(receipts, "receipt")
    update_by_key = _unique(updates, "update")

    sessions_per_arm = sum(len(task.sessions) for task in tasks)
    decisions_per_arm = sum(len(task.sessions) - 1 for task in tasks)
    _assert_equal(len(outcomes), sessions_per_arm * len(arms), "outcome rows")
    _assert_equal(len(updates), sessions_per_arm * len(arms), "update rows")
    _assert_equal(len(receipts), decisions_per_arm * len(arms), "receipt rows")

    expected_nonoff_scopes = {
        f"memoryarena_agent:{summary['run_id']}:{arm}:{task.task_uid}": (arm, task)
        for task in tasks
        for arm in arms
        if arm != Arm.MEMORY_OFF.value
    }
    expected_off_scopes = [
        f"memoryarena_agent:{summary['run_id']}:{Arm.MEMORY_OFF.value}:{task.task_uid}"
        for task in tasks
    ]
    with psycopg.connect(dsn, autocommit=True) as database:
        # Load both extension libraries before reading their custom GUC defaults.
        database.execute(
            "SELECT tjs_open_candidates_examined(),'[0]'::vector"
        ).fetchone()
        server = database.execute("SELECT version(),current_database()").fetchone()
        extensions = database.execute(
            "SELECT extname,extversion FROM pg_extension"
            " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY 1"
        ).fetchall()
        settings = {
            name: database.execute(
                "SELECT current_setting(%s,true)", (name,)
            ).fetchone()[0]
            for name in (
                "hnsw.ef_search",
                "hnsw.max_scan_tuples",
                "hnsw.scan_mem_multiplier",
                "tjs.graph_work_budget",
                "tjs.vector_scan_budget",
                "tjs.filter_probe",
            )
        }
        database_rows = database.execute(
            "SELECT scope_id,metadata->>'session_id',"
            " metadata->>'retrieval_receipt_sha256' FROM gem_unit"
            " WHERE scope_id=ANY(%s) ORDER BY scope_id,id",
            (list(expected_nonoff_scopes),),
        ).fetchall()
        memory_off_units = int(
            database.execute(
                "SELECT count(*) FROM gem_unit WHERE scope_id=ANY(%s)",
                (expected_off_scopes,),
            ).fetchone()[0]
        )
        global_counts = database.execute(
            "SELECT count(*),graph_store.gph_vertex_count(),"
            " graph_store.gph_visible_edge_count() FROM gem_unit"
        ).fetchone()
    database_by_scope: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    if memory_off_units:
        raise ValueError(f"memory-off database units present: {memory_off_units}")
    for scope_id, session_uid, receipt_sha256 in database_rows:
        database_by_scope[str(scope_id)].append((str(session_uid), str(receipt_sha256)))
    for scope_id, (arm, task) in expected_nonoff_scopes.items():
        observed = database_by_scope[scope_id]
        if len(observed) != len(task.sessions):
            raise ValueError(
                f"database scope completeness {scope_id}: "
                f"{len(observed)} != {len(task.sessions)}"
            )
        if {uid for uid, _ in observed} != {
            session.session_uid for session in task.sessions
        }:
            raise ValueError(f"database session identity mismatch: {scope_id}")
        for session_uid, receipt_sha256 in observed:
            if (
                outcome_by_key[(arm, task.task_uid, session_uid)][
                    "retrieval_receipt_sha256"
                ]
                != receipt_sha256
            ):
                raise ValueError(f"database receipt linkage mismatch: {scope_id}")

    violations: defaultdict[str, int] = defaultdict(int)
    metric_rows: dict[str, list[dict[str, float | int | None]]] = defaultdict(list)
    update_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    exact_oracle_queries = 0
    semantic_provenance_rows = 0

    for task_index, task in enumerate(tasks):
        expected_candidate_uids = {session.session_uid for session in task.sessions}
        for arm_index, arm in enumerate(arms):
            expected_position = (arm_index - task_index) % len(arms)
            for session in task.sessions:
                key = (arm, task.task_uid, session.session_uid)
                if key not in outcome_by_key:
                    raise ValueError(f"missing outcome: {key}")
                if key not in update_by_key:
                    raise ValueError(f"missing update: {key}")
                outcome = outcome_by_key[key]
                update = update_by_key[key]
                _assert_equal(
                    int(outcome["ordinal"]), session.ordinal, f"ordinal {key}"
                )
                _assert_equal(
                    outcome["question_sha256"],
                    _text_sha256(session.question),
                    f"question {key}",
                )
                _assert_equal(
                    outcome["response_sha256"],
                    _text_sha256(str(outcome["response"])),
                    f"response {key}",
                )
                _assert_equal(
                    int(outcome["arm_execution_position"]),
                    expected_position,
                    f"rotation {key}",
                )
                seeded = session.ordinal == 0
                _assert_equal(
                    bool(outcome["source_session_seeded_with_gold"]),
                    seeded,
                    f"seed {key}",
                )
                _assert_equal(
                    bool(outcome["evaluated"]), not seeded, f"evaluated {key}"
                )
                _assert_equal(bool(update["committed"]), True, f"commit {key}")
                if arm == Arm.MEMORY_OFF.value:
                    if not (
                        update["delta"] is None
                        and update["cost"] is None
                        and int(update["linked_edges"]) == 0
                        and int(update["wal_bytes_interval"]) == 0
                        and float(update["update_ms"]) == 0.0
                    ):
                        violations["memory_off_write"] += 1
                else:
                    if update["delta"] is None or update["cost"] is None:
                        violations["memory_arm_missing_write_evidence"] += 1
                update_rows[arm].append(update)

                if seeded:
                    if key in receipt_by_key:
                        violations["source_session_receipt_present"] += 1
                    continue
                if key not in receipt_by_key:
                    raise ValueError(f"missing decision receipt: {key}")
                receipt = receipt_by_key[key]
                if not verify_serialized_receipt(receipt):
                    violations["receipt_digest"] += 1
                _assert_equal(
                    receipt["schema_version"], RECEIPT_SCHEMA_VERSION, f"schema {key}"
                )
                _assert_equal(
                    receipt["dataset_manifest_sha256"],
                    corpus.manifest_sha256,
                    f"manifest {key}",
                )
                _assert_equal(
                    receipt["target_session_uid"], session.session_uid, f"target {key}"
                )
                _assert_equal(
                    int(receipt["cutoff_ordinal"]), session.ordinal, f"cutoff {key}"
                )
                _assert_equal(
                    receipt["query_sha256"],
                    _text_sha256(session.question),
                    f"query {key}",
                )
                _assert_equal(
                    outcome["retrieval_receipt_sha256"],
                    receipt["receipt_sha256"],
                    f"receipt link {key}",
                )
                if not str(receipt["snapshot_id"]).startswith("sha256:"):
                    violations["snapshot_not_content_addressed"] += 1
                if int(receipt["injection_tokens"]) > int(
                    receipt["injection_token_budget"]
                ):
                    violations["token_budget"] += 1
                selected = [str(uid) for uid in receipt["selected_session_uids"]]
                if len(selected) > int(receipt["top_k"]):
                    violations["top_k"] += 1
                if len(selected) != len(set(selected)):
                    violations["duplicate_selected"] += 1

                candidates = receipt["candidates"]
                candidate_uids = [
                    str(candidate["session_uid"]) for candidate in candidates
                ]
                if len(candidate_uids) != len(set(candidate_uids)):
                    violations["duplicate_candidates"] += 1
                if set(candidate_uids) != expected_candidate_uids:
                    violations["candidate_universe_parity"] += 1
                candidate_by_uid = {
                    str(candidate["session_uid"]): candidate for candidate in candidates
                }
                expected_eligible = frozenset(session.protocol_dependencies)
                observed_eligible = frozenset(
                    uid
                    for uid, candidate in candidate_by_uid.items()
                    if str(candidate["task_uid"]) == task.task_uid
                    and uid != session.session_uid
                    and int(candidate["ordinal"]) < session.ordinal
                    and bool(candidate["scope_allowed"])
                )
                if observed_eligible != expected_eligible:
                    violations["exact_eligibility_parity"] += 1
                for rank, uid in enumerate(selected, 1):
                    candidate = candidate_by_uid.get(uid)
                    if candidate is None:
                        violations["selected_absent_candidate"] += 1
                        continue
                    if uid not in expected_eligible:
                        violations["future_target_or_cross_task_leakage"] += 1
                    if int(candidate["selected_rank"]) != rank:
                        violations["selected_rank"] += 1
                for uid, candidate in candidate_by_uid.items():
                    expected_rank = selected.index(uid) + 1 if uid in selected else None
                    if candidate["selected_rank"] != expected_rank:
                        violations["candidate_rank_parity"] += 1

                prior_outcomes = outcome_by_key
                rendered = "\n\n".join(
                    "Question: "
                    + session_by_uid[uid].question
                    + "\nResponse: "
                    + str(prior_outcomes[(arm, task.task_uid, uid)]["response"])
                    for uid in selected
                )
                if receipt["injection_sha256"] != _text_sha256(rendered):
                    violations["injection_content_hash"] += 1

                if arm == Arm.MEMORY_OFF.value and selected:
                    violations["memory_off_selected"] += 1
                if arm == Arm.RECENT_FIFO.value:
                    expected = sorted(
                        expected_eligible,
                        key=lambda uid: (-session_by_uid[uid].ordinal, uid),
                    )
                    if selected != expected[: len(selected)]:
                        violations["recent_rank_parity"] += 1
                if arm == Arm.ORACLE.value:
                    expected = sorted(
                        expected_eligible, key=lambda uid: session_by_uid[uid].ordinal
                    )
                    if selected != expected[: min(10, len(expected))]:
                        violations["exact_oracle_selection_parity"] += 1
                    exact_oracle_queries += 1

                actual_score_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate["semantic_score_source"]
                    == "pgvector_cosine_similarity_selected_top_k"
                ]
                if arm in (Arm.VECTOR_ONLY.value, Arm.GEM_FUSED.value):
                    for candidate in actual_score_candidates:
                        score = float(candidate["semantic_score"])
                        if not -1.000001 <= score <= 1.000001:
                            violations["semantic_score_bounds"] += 1
                    semantic_provenance_rows += len(actual_score_candidates)
                elif actual_score_candidates:
                    violations["semantic_score_wrong_arm"] += len(
                        actual_score_candidates
                    )

                relevant = frozenset(session.protocol_dependencies)
                recall = (
                    len(set(selected[:10]) & relevant) / len(relevant)
                    if relevant
                    else 0.0
                )
                valid = [
                    bool(candidate_by_uid[uid]["is_valid"])
                    and not bool(candidate_by_uid[uid]["is_stale"])
                    and bool(candidate_by_uid[uid]["scope_allowed"])
                    for uid in selected
                ]
                metric_rows[arm].append(
                    {
                        "dependency_recall_at_10": recall,
                        "ndcg_at_10": _ndcg(selected, relevant),
                        "constraint_valid_fraction": statistics.fmean(valid)
                        if valid
                        else 1.0,
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

    arm_metrics = []
    for arm in arms:
        rows = metric_rows[arm]
        writes = update_rows[arm]
        update_ms = [
            float(row["update_ms"]) for row in writes if float(row["update_ms"]) > 0
        ]
        examined = [
            float(row["candidates_examined"])
            for row in rows
            if row["candidates_examined"] is not None
        ]
        returned_for_examined = [
            float(row["returned"]) / float(row["candidates_examined"])
            for row in rows
            if row["candidates_examined"] not in (None, 0)
        ]
        arm_metrics.append(
            {
                "arm": arm,
                "decisions": len(rows),
                "dependency_recall_at_10": _distribution(
                    float(row["dependency_recall_at_10"]) for row in rows
                ),
                "ndcg_at_10": _distribution(float(row["ndcg_at_10"]) for row in rows),
                "harmful_at_10": None,
                "harmful_annotation": "N/A: public Progressive Search release has no harmful labels",
                "stale_at_10": None,
                "stale_annotation": "N/A: public Progressive Search release has no stale labels",
                "constraint_valid_fraction": _distribution(
                    float(row["constraint_valid_fraction"]) for row in rows
                ),
                "time_to_first_row_ms": _distribution(
                    float(row["first_row_ms"])
                    for row in rows
                    if row["first_row_ms"] is not None
                ),
                "time_to_k_ms": _distribution(
                    float(row["time_to_k_ms"]) for row in rows
                ),
                "candidates_examined": _distribution(examined),
                "visited_nodes": _distribution(
                    float(row["visited_nodes"])
                    for row in rows
                    if row["visited_nodes"] is not None
                ),
                "visited_nodes_note": (
                    "N/A for live fused queries: tjs_pg exposes edge-steps, not a "
                    "distinct-vertex counter"
                ),
                "visited_edges": _distribution(
                    float(row["visited_edges"])
                    for row in rows
                    if row["visited_edges"] is not None
                ),
                "visited_edges_note": "tjs_open_graph_examined edge-step counter",
                "returned_over_candidates_examined": _distribution(
                    returned_for_examined
                ),
                "candidate_contraction_fraction": _distribution(
                    [1.0 - ratio for ratio in returned_for_examined]
                ),
                "update_ms": _distribution(update_ms),
                "update_throughput_per_second": len(update_ms) * 1000.0 / sum(update_ms)
                if update_ms
                else None,
                "wal_bytes_interval": _distribution(
                    float(row["wal_bytes_interval"])
                    for row in writes
                    if arm != Arm.MEMORY_OFF.value
                ),
            }
        )

    outcome_groups: defaultdict[str, defaultdict[str, list[Mapping[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in outcomes:
        enriched = {
            **row,
            "_update_ms": float(update_by_key[_key(row)]["update_ms"]),
        }
        outcome_groups[str(row["arm"])][str(row["task_uid"])].append(enriched)
    task_outcomes = {
        arm: {
            task_uid: _task_outcome_metrics(rows)
            for task_uid, rows in outcome_groups[arm].items()
        }
        for arm in arms
    }
    runner_arms = {str(row["arm"]): row for row in summary["arms"]}
    independent_agent_arms = []
    for arm in arms:
        task_values = task_outcomes[arm]
        evaluated = [row for row in outcomes if row["arm"] == arm and row["evaluated"]]
        observed_accuracy = statistics.fmean(
            float(bool(row["exact_match"])) for row in evaluated
        )
        _assert_close(
            observed_accuracy,
            float(runner_arms[arm]["cross_session_exact_match"]),
            f"runner outcome summary {arm}",
        )
        _assert_equal(
            len(task_values), int(runner_arms[arm]["tasks"]), f"runner tasks {arm}"
        )
        total_end_to_end_ms = sum(
            float(row["effective_total_ms"])
            + float(update_by_key[_key(row)]["update_ms"])
            for row in evaluated
        )
        total_inference_tokens = sum(
            int(row["usage"].get("prompt_tokens", 0))
            + int(row["usage"].get("completion_tokens", 0))
            for row in evaluated
        )
        successes = sum(bool(row["exact_match"]) for row in evaluated)
        independent_agent_arms.append(
            {
                "arm": arm,
                "tasks": len(task_values),
                "evaluated_sessions": len(evaluated),
                "cross_session_exact_match": observed_accuracy,
                "total_steps": len(evaluated),
                "total_inference_tokens": total_inference_tokens,
                "total_injection_tokens": sum(
                    int(row["injection_tokens"]) for row in evaluated
                ),
                "successful_episodes_per_second": (
                    successes * 1000.0 / total_end_to_end_ms
                    if total_end_to_end_ms > 0
                    else None
                ),
                "successes_per_1m_inference_tokens": (
                    successes * 1_000_000.0 / total_inference_tokens
                    if total_inference_tokens > 0
                    else None
                ),
                "task_metrics": {
                    metric: _distribution(
                        [values[metric] for values in task_values.values()]
                    )
                    for metric in next(iter(task_values.values()))
                },
            }
        )
    paired_example = next(
        iter(next(iter(summary["paired_vs_memory_off"].values())).values())
    )
    bootstrap_iterations = int(paired_example["iterations"])
    baseline_outcomes = task_outcomes[Arm.MEMORY_OFF.value]
    independent_paired = {
        arm: {
            metric: _paired_bootstrap(
                task_outcomes[arm],
                baseline_outcomes,
                metric,
                iterations=bootstrap_iterations,
                seed=int(summary["protocol"]["seed"]),
            )
            for metric in next(iter(task_outcomes[arm].values()))
        }
        for arm in arms
        if arm != Arm.MEMORY_OFF.value
    }

    missing_predictions = sum(
        1
        for row in outcomes
        if bool(row["evaluated"]) and row["predicted_exact_answer"] is None
    )
    model_calls = summary["model_calls"]
    _assert_equal(int(model_calls["failed_calls"]), 0, "model call failures")
    _assert_equal(
        int(model_calls["by_kind"]["answer_generation"]),
        len(receipts),
        "one answer generation per evaluated decision",
    )
    audit_payload = {
        "schema_version": "memoryarena_agent_independent_audit_v0.1.0",
        "status": "pass",
        "claim_boundary": summary["claim_boundary"],
        "run_id": summary["run_id"],
        "dataset": {
            "config": corpus.config,
            "source_revision": corpus.source_revision,
            "manifest_sha256": corpus.manifest_sha256,
            "tasks": tasks_run,
            "sessions_per_arm": sessions_per_arm,
            "decisions_per_arm": decisions_per_arm,
        },
        "model_provenance": {
            "answer_model": summary["protocol"]["answer_model"],
            "answer_revision": answer_revision,
            "embedding_model": summary["protocol"]["embedding_model"],
            "embedding_revision": embedding_revision,
            "temperature": summary["protocol"]["temperature"],
            "seed": summary["protocol"]["seed"],
            "injection_token_budget": summary["protocol"]["injection_token_budget"],
        },
        "database_post_state": {
            "version": server[0],
            "database": server[1],
            "extensions": {name: version for name, version in extensions},
            "default_settings_observed_at_audit": settings,
            "query_overrides": {
                "vector_only.hnsw.iterative_scan": "strict_order",
                "gem_fused.hnsw.iterative_scan": "relaxed_order",
            },
            "nonoff_scopes": len(expected_nonoff_scopes),
            "units_linked_to_receipts": len(database_rows),
            "memory_off_units": memory_off_units,
            "global_counts_at_audit": list(global_counts),
            "scope_and_receipt_linkage_gate": "pass",
        },
        "implementation": {
            str(path.relative_to(repository)): _file_sha256(path)
            for path in (
                repository / "tools/run_memoryarena_agent_outcomes.py",
                repository / "bench/agent_memory/memoryarena/tridb_backend.py",
                repository / "bench/agent_memory/memoryarena/cross_session.py",
                repository / "bench/agent_memory/memoryarena/oracle.py",
                repository / "bench/agent_memory/memoryarena/receipts.py",
                repository / "bench/agent_memory/memoryarena/dataset.py",
                Path(__file__).resolve(),
            )
        },
        "completeness": {
            "outcomes": len(outcomes),
            "receipts": len(receipts),
            "updates": len(updates),
            "expected_outcomes": sessions_per_arm * len(arms),
            "expected_receipts": decisions_per_arm * len(arms),
            "expected_updates": sessions_per_arm * len(arms),
            "missing_predictions": missing_predictions,
            "model_call_failures": int(model_calls["failed_calls"]),
            "answer_generation_calls": int(model_calls["by_kind"]["answer_generation"]),
        },
        "gates": {
            "receipt_digest_violations": 0,
            "exact_eligibility_parity_queries": len(receipts),
            "exact_eligibility_parity_violations": 0,
            "exact_oracle_selection_parity_queries": exact_oracle_queries,
            "exact_oracle_selection_parity_violations": 0,
            "future_target_cross_task_scope_leakage": 0,
            "token_budget_violations": 0,
            "top_k_violations": 0,
            "memory_off_write_violations": 0,
            "injection_content_hash_violations": 0,
            "semantic_score_provenance_rows": semantic_provenance_rows,
        },
        "retrieval_and_system_metrics": arm_metrics,
        "agent_outcome": {
            "arms": independent_agent_arms,
            "paired_vs_memory_off": independent_paired,
            "confidence_interval_unit": "task-level paired cluster bootstrap",
            "steps": "one fixed direct-answer generation call per evaluated decision",
            "time_to_target_definition": (
                "sum(retrieval + answer generation + post-answer memory update) over "
                "evaluated sessions"
            ),
            "runner_summary_cross_check": "pass",
        },
        "artifacts": {
            name: {"path": path.name, "sha256": _file_sha256(path)}
            for name, path in {**paths, "summary": summary_path}.items()
        },
    }
    return audit_payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit(
        args.export_dir,
        args.run_dir,
        dsn=args.dsn,
        require_full=args.require_full,
        answer_revision=args.answer_revision,
        embedding_revision=args.embedding_revision,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(
        f"PASS tasks={report['dataset']['tasks']} receipts={report['completeness']['receipts']} "
        "leakage=0 oracle_parity=0_violations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
