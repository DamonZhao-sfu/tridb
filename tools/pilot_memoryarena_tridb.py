"""Run a gold-response MemoryArena protocol pilot against live stock-PG TriDB."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any, Sequence

from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.store import GemStore
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)
from bench.agent_memory.memoryarena.cross_session import CrossSessionDriver
from bench.agent_memory.memoryarena.dataset import load_export
from bench.agent_memory.memoryarena.metrics import retrieval_metrics
from bench.agent_memory.memoryarena.oracle import Arm, Candidate, DecisionPoint
from bench.agent_memory.memoryarena.tridb_backend import TriDBMemoryArenaBackend


class HashEmbedder:
    """Deterministic smoke-test embedder; never label its ranking as model quality."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        result = []
        for text in texts:
            vector = [0.0] * self.dim
            for index, byte in enumerate(text.encode()):
                vector[index % self.dim] += (byte % 31) / 31.0
            if not any(vector):
                vector[0] = 1.0
            result.append(vector)
        return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-tasks", type=int, default=12)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--injection-word-budget", type=int, default=4096)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[arm.value for arm in Arm],
        default=[a.value for a in Arm],
    )
    return parser.parse_args(argv)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _candidate_from_receipt(item: Any, task_uid: str) -> Candidate:
    return Candidate(
        session_uid=item.session_uid,
        task_uid=item.task_uid or task_uid,
        ordinal=item.ordinal,
        semantic_score=item.semantic_score,
        semantic_score_source=item.semantic_score_source,
        graph_distance=item.graph_distance,
        scope_allowed=item.scope_allowed,
        is_valid=item.is_valid,
        is_stale=item.is_stale,
        is_harmful=item.is_harmful,
    )


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_tasks < 1:
        raise ValueError("max-tasks must be positive")
    if args.embedding_base_url and not args.embedding_revision:
        raise ValueError("real embedding runs require --embedding-revision")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_export(args.export_dir)
    ledger: CallLedger | None = None
    if args.embedding_base_url:
        ledger = CallLedger()
        client = OpenAIEmbeddingClient(
            args.embedding_base_url,
            "EMPTY",
            args.embedding_model,
            batch_size=args.embedding_batch_size,
            timeout=args.embedding_timeout_seconds,
            ledger=ledger,
        )
        advertised = client.discover_models()
        if advertised != [args.embedding_model]:
            raise RuntimeError(
                f"embedding endpoint identity mismatch: {advertised!r} != "
                f"{[args.embedding_model]!r}"
            )
        embedder: Any = PhasedEmbedder(client)
        probe = embedder.encode(["dimension probe"])[0]
        if len(probe) != args.dim:
            raise RuntimeError(
                f"embedding dimension mismatch: {len(probe)} != {args.dim}"
            )
        embedding_manifest = {
            "model": args.embedding_model,
            "revision": args.embedding_revision,
            "base_url": args.embedding_base_url,
            "dim": args.dim,
            "advertised_models": advertised,
            "quality_claim": "real pinned embedding model; gold-response replay",
        }
    else:
        embedder = HashEmbedder(args.dim)
        embedding_manifest = {
            "model": "deterministic_hash_smoke_only",
            "dim": args.dim,
            "quality_claim": "none",
        }
    memory = TriDBGovernedMemory(
        GemStore.connect(args.dsn, dim=args.dim), embedder=embedder
    )
    memory.init_schema()
    server = memory.store.conn.execute(
        "SELECT version(), current_database()"
    ).fetchone()
    extensions = memory.store.conn.execute(
        "SELECT extname, extversion FROM pg_extension"
        " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY 1"
    ).fetchall()
    receipt_path = args.output_dir / "receipts.jsonl"
    update_path = args.output_dir / "updates.jsonl"
    arm_values = [Arm(name) for name in args.arms]
    stats: dict[str, dict[str, Any]] = {
        arm.value: {
            "latencies": [],
            "first_row_latencies": [],
            "updates_ms": [],
            "wal_bytes": [],
            "dependency_recall": [],
            "ndcg": [],
            "selected_total": 0,
            "empty_results": 0,
        }
        for arm in arm_values
    }
    try:
        with ExitStack() as stack:
            receipt_file = stack.enter_context(receipt_path.open("x", encoding="utf-8"))
            update_file = stack.enter_context(update_path.open("x", encoding="utf-8"))
            for task_index, task in enumerate(corpus.tasks[: args.max_tasks]):
                rotation = task_index % len(arm_values)
                task_arm_order = arm_values[rotation:] + arm_values[:rotation]
                for arm in task_arm_order:
                    arm_stats = stats[arm.value]
                    namespace = f"memoryarena_pilot:{args.run_id}:{arm.value}"
                    backend = TriDBMemoryArenaBackend(
                        memory,
                        task,
                        namespace=namespace,
                        reinforce=False,
                        write_enabled=arm is not Arm.MEMORY_OFF,
                    )
                    existing = memory.store.conn.execute(
                        "SELECT count(*) FROM gem_unit WHERE scope_id=%s",
                        (backend.scope_id,),
                    ).fetchone()[0]
                    if int(existing):
                        raise RuntimeError(
                            f"scope is not empty; choose a new run-id: {backend.scope_id}"
                        )
                    driver = CrossSessionDriver(
                        backend,
                        run_id=f"{args.run_id}:{arm.value}:{task.task_uid}",
                        dataset_manifest_sha256=corpus.manifest_sha256,
                        arm=arm,
                        top_k=args.top_k,
                        injection_token_budget=args.injection_word_budget,
                        count_tokens=lambda value: len(value.split()),
                    )
                    for session in task.sessions:
                        opened = driver.open_session(session)
                        if session.ordinal > 0:
                            serialized = opened.receipt.as_dict()
                            receipt_file.write(
                                json.dumps(serialized, sort_keys=True) + "\n"
                            )
                            arm_stats["latencies"].append(opened.receipt.time_to_k_ms)
                            if opened.receipt.first_row_ms is not None:
                                arm_stats["first_row_latencies"].append(
                                    opened.receipt.first_row_ms
                                )
                            if not opened.receipt.selected_session_uids:
                                arm_stats["empty_results"] += 1
                            arm_stats["selected_total"] += len(
                                opened.receipt.selected_session_uids
                            )
                            by_uid = {
                                item.session_uid: _candidate_from_receipt(
                                    item, task.task_uid
                                )
                                for item in opened.receipt.candidates
                            }
                            selected = [
                                by_uid[uid]
                                for uid in opened.receipt.selected_session_uids
                            ]
                            metrics = retrieval_metrics(
                                selected, DecisionPoint.from_session(session)
                            )
                            arm_stats["dependency_recall"].append(
                                float(metrics["dependency_recall_at_10"])
                            )
                            arm_stats["ndcg"].append(float(metrics["ndcg_at_10"]))
                        update = dict(
                            driver.complete_session(
                                session, response=session.gold_answer
                            )
                        )
                        update_file.write(
                            json.dumps(
                                {
                                    "arm": arm.value,
                                    "task_uid": task.task_uid,
                                    "session_uid": session.session_uid,
                                    **update,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        arm_stats["updates_ms"].append(float(update["update_ms"]))
                        arm_stats["wal_bytes"].append(int(update["wal_bytes_interval"]))
    finally:
        memory.close()
    arm_summaries = []
    for arm in arm_values:
        arm_stats = stats[arm.value]
        latencies = arm_stats["latencies"]
        first_row_latencies = arm_stats["first_row_latencies"]
        updates_ms = arm_stats["updates_ms"]
        arm_summaries.append(
            {
                "arm": arm.value,
                "decision_points": len(latencies),
                "selected_total": arm_stats["selected_total"],
                "time_to_k_ms": _summary(latencies),
                "time_to_first_row_ms": _summary(first_row_latencies),
                "time_to_first_row_missing": len(latencies) - len(first_row_latencies),
                "empty_result_count": arm_stats["empty_results"],
                "update_ms": _summary(updates_ms),
                "update_throughput_per_second": (
                    1000.0 * len(updates_ms) / sum(updates_ms)
                    if updates_ms and sum(updates_ms) > 0
                    else None
                ),
                "wal_bytes_interval": _summary(arm_stats["wal_bytes"]),
                "dependency_recall_at_10": _summary(arm_stats["dependency_recall"]),
                "ndcg_at_10": _summary(arm_stats["ndcg"]),
                "agent_outcome": None,
                "agent_outcome_reason": "gold-response protocol replay",
            }
        )
    summary = {
        "schema_version": "memoryarena_tridb_pilot_v0.1.0",
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "protocol_and_live_adapter_pilot_only",
        "dataset": {
            "config": corpus.config,
            "source_revision": corpus.source_revision,
            "source_sha256": corpus.source_file_sha256,
            "manifest_sha256": corpus.manifest_sha256,
            "tasks_run": min(args.max_tasks, len(corpus.tasks)),
        },
        "engine": {
            "version": server[0],
            "database": server[1],
            "extensions": {name: version for name, version in extensions},
            "hardware_claim": "x86_64 stock-PG only; no GX10 sign-off",
        },
        "embedding": embedding_manifest,
        "embedding_calls": ledger.summary() if ledger is not None else None,
        "embedding_tokens": ledger.tokens() if ledger is not None else None,
        "token_counter": "whitespace_words_protocol_only",
        "execution_schedule": "task_outer_cyclic_arm_rotation",
        "arms": arm_summaries,
        "artifacts": {
            "receipts": receipt_path.name,
            "updates": update_path.name,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS tasks={summary['dataset']['tasks_run']} arms={len(arm_summaries)} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
