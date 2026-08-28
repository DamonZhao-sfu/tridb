"""Run paired Progressive Search agent outcomes over the live GEM six-arm path.

This is deliberately a direct-answer track, not MemoryArena's web-search-agent
reproduction.  It isolates whether bounded cross-session Q/A memory changes the
same pinned local model's answer.  Scores are deterministic normalized exact match;
they must not be reported as the upstream LLM-judge metric.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence
import unicodedata

import requests

from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.store import GemStore
from bench.agent_memory.memoryarena.cross_session import CrossSessionDriver
from bench.agent_memory.memoryarena.dataset import extract_exact_answer, load_export
from bench.agent_memory.memoryarena.oracle import Arm
from bench.agent_memory.memoryarena.tridb_backend import TriDBMemoryArenaBackend
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)

SYSTEM_PROMPT = (
    "You answer a sequence of progressively constrained entity-identification "
    "questions. Retrieved memory contains prior questions and your prior answers "
    "from this task only. Use it when relevant, but resolve conflicts in favor of "
    "the current question. Return exactly two lines:\n"
    "Exact Answer: <the shortest unambiguous answer>\n"
    "Explanation: <one sentence of at most 30 words>"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-tasks", type=int, default=3)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="qwen3.8")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--injection-token-budget", type=int, default=2048)
    parser.add_argument("--max-answer-tokens", type=int, default=128)
    parser.add_argument(
        "--seed-first-session-with-gold",
        action="store_true",
        help=(
            "admit the released first-session response as fixed source experience "
            "and evaluate only later cross-session decisions"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=[arm.value for arm in Arm],
        default=[arm.value for arm in Arm],
    )
    return parser.parse_args(argv)


def normalize_exact_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("**", "")
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    normalized = " ".join(normalized.split())
    return normalized or None


class EndpointTokenCounter:
    """Exact serving-tokenizer counts with a per-run immutable text cache."""

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.url = f"{root}/tokenize"
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.cache: dict[str, int] = {"": 0}
        self.requests = 0

    def __call__(self, text: str) -> int:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        response = self.session.post(
            self.url,
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        count = int(response.json()["count"])
        self.cache[text] = count
        self.requests += 1
        return count


def _messages(question: str, injection: str) -> list[dict[str, str]]:
    memory = injection if injection else "<none>"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Retrieved prior-session memory:\n{memory}\n\nQuestion:\n{question}",
        },
    ]


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _task_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    ordered = sorted(
        (row for row in rows if bool(row["evaluated"])),
        key=lambda row: int(row["ordinal"]),
    )
    if not ordered:
        raise ValueError("task has no evaluated sessions")
    correct = [float(bool(row["exact_match"])) for row in ordered]
    cumulative = [statistics.fmean(correct[: index + 1]) for index in range(len(correct))]
    return {
        "all_session_accuracy": statistics.fmean(correct),
        "cross_session_accuracy": statistics.fmean(correct),
        "final_session_success": correct[-1],
        "all_sessions_success": float(all(correct)),
        "progress_aulc": statistics.fmean(cumulative),
        "prompt_tokens": float(sum(int(row["usage"].get("prompt_tokens", 0)) for row in ordered)),
        "completion_tokens": float(
            sum(int(row["usage"].get("completion_tokens", 0)) for row in ordered)
        ),
        "time_to_target_ms": float(sum(float(row["effective_total_ms"]) for row in ordered)),
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
    if not deltas:
        return {"n_tasks": 0, "mean_delta": None, "ci95": [None, None]}
    generator = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(
            statistics.fmean(deltas[generator.randrange(len(deltas))] for _ in deltas)
        )
    return {
        "n_tasks": len(task_ids),
        "mean_delta": statistics.fmean(deltas),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "method": "paired_task_cluster_percentile_bootstrap",
        "iterations": iterations,
        "seed": seed,
    }


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_tasks < 1 or args.bootstrap_iterations < 1:
        raise ValueError("max-tasks and bootstrap-iterations must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = load_export(args.export_dir)
    if corpus.config != "progressive_search":
        raise ValueError("this deterministic exact-answer track supports progressive_search only")

    ledger = CallLedger()
    embedding_client = OpenAIEmbeddingClient(
        args.embedding_base_url,
        "EMPTY",
        args.embedding_model,
        batch_size=64,
        timeout=args.timeout_seconds,
        ledger=ledger,
    )
    if embedding_client.discover_models() != [args.embedding_model]:
        raise RuntimeError("embedding endpoint identity mismatch")
    embedder = PhasedEmbedder(embedding_client)
    if len(embedder.encode(["dimension probe"])[0]) != args.dim:
        raise RuntimeError("embedding dimension mismatch")

    answer_client = OpenAIChatClient(
        args.answer_base_url,
        "EMPTY",
        timeout=args.timeout_seconds,
        ledger=ledger,
    )
    advertised_answers = answer_client.discover_models()
    if advertised_answers != [args.answer_model]:
        raise RuntimeError(
            f"answer endpoint identity mismatch: {advertised_answers!r} != {[args.answer_model]!r}"
        )
    token_counter = EndpointTokenCounter(
        args.answer_base_url, args.answer_model, args.timeout_seconds
    )
    memory = TriDBGovernedMemory(GemStore.connect(args.dsn, dim=args.dim), embedder=embedder)
    memory.init_schema()
    server = memory.store.conn.execute("SELECT version(), current_database()").fetchone()
    extensions = memory.store.conn.execute(
        "SELECT extname, extversion FROM pg_extension"
        " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY 1"
    ).fetchall()

    outcomes_path = args.output_dir / "outcomes.jsonl"
    receipts_path = args.output_dir / "receipts.jsonl"
    updates_path = args.output_dir / "updates.jsonl"
    by_arm_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    try:
        with ExitStack() as stack:
            outcomes_file = stack.enter_context(outcomes_path.open("x", encoding="utf-8"))
            receipts_file = stack.enter_context(receipts_path.open("x", encoding="utf-8"))
            updates_file = stack.enter_context(updates_path.open("x", encoding="utf-8"))
            arm_values = [Arm(name) for name in args.arms]
            for task_index, task in enumerate(corpus.tasks[: args.max_tasks]):
                rotation = task_index % len(arm_values)
                task_arm_order = arm_values[rotation:] + arm_values[:rotation]
                for arm_position, arm in enumerate(task_arm_order):
                    namespace = f"memoryarena_agent:{args.run_id}:{arm.value}"
                    backend = TriDBMemoryArenaBackend(
                        memory,
                        task,
                        namespace=namespace,
                        write_enabled=arm is not Arm.MEMORY_OFF,
                    )
                    existing = memory.store.conn.execute(
                        "SELECT count(*) FROM gem_unit WHERE scope_id=%s", (backend.scope_id,)
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
                        injection_token_budget=args.injection_token_budget,
                        count_tokens=token_counter,
                    )
                    for session in task.sessions:
                        admitted = time.perf_counter()
                        opened = driver.open_session(session)
                        seeded = args.seed_first_session_with_gold and session.ordinal == 0
                        if seeded:
                            response = session.gold_answer
                            predicted = session.gold_exact_answer
                            exact_match = True
                            usage: Mapping[str, Any] = {}
                            effective_ttft_ms = 0.0
                            generation_ms = 0.0
                            effective_total_ms = (time.perf_counter() - admitted) * 1000.0
                        else:
                            result = answer_client.stream_chat(
                                model=args.answer_model,
                                messages=_messages(session.question, opened.injection_text),
                                max_tokens=args.max_answer_tokens,
                                temperature=0.0,
                                seed=args.seed,
                            )
                            response = result.text
                            usage = result.usage
                            predicted = extract_exact_answer(response)
                            exact_match = normalize_exact_answer(
                                predicted
                            ) == normalize_exact_answer(session.gold_exact_answer)
                            effective_ttft_ms = (
                                result.first_token_at - admitted
                            ) * 1000.0
                            generation_ms = (
                                result.completed_at - result.request_started
                            ) * 1000.0
                            effective_total_ms = (
                                result.completed_at - admitted
                            ) * 1000.0
                        row = {
                            "arm": arm.value,
                            "task_execution_index": task_index,
                            "arm_execution_position": arm_position,
                            "task_uid": task.task_uid,
                            "session_uid": session.session_uid,
                            "ordinal": session.ordinal,
                            "question_sha256": hashlib.sha256(session.question.encode()).hexdigest(),
                            "gold_exact_answer": session.gold_exact_answer,
                            "predicted_exact_answer": predicted,
                            "evaluated": not seeded,
                            "source_session_seeded_with_gold": seeded,
                            "normalized_exact_match_v1": exact_match,
                            "exact_match": exact_match,
                            "response": response,
                            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                            "injection_tokens": opened.receipt.injection_tokens,
                            "usage": usage,
                            "effective_ttft_ms": effective_ttft_ms,
                            "generation_ms": generation_ms,
                            "effective_total_ms": effective_total_ms,
                            "retrieval_receipt_sha256": opened.receipt.as_dict()[
                                "receipt_sha256"
                            ],
                        }
                        outcomes_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        outcomes_file.flush()
                        by_arm_task[arm.value][task.task_uid].append(row)
                        if session.ordinal > 0:
                            receipts_file.write(
                                json.dumps(opened.receipt.as_dict(), sort_keys=True) + "\n"
                            )
                            receipts_file.flush()
                        update = driver.complete_session(session, response=response)
                        updates_file.write(
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
                        updates_file.flush()
    finally:
        memory.close()

    task_metrics = {
        arm: {task: _task_metrics(rows) for task, rows in tasks.items()}
        for arm, tasks in by_arm_task.items()
    }
    metric_names = tuple(next(iter(next(iter(task_metrics.values())).values())))
    arm_summaries = []
    for arm in args.arms:
        rows = [row for task_rows in by_arm_task[arm].values() for row in task_rows]
        evaluated = [row for row in rows if bool(row["evaluated"])]
        cross = [row for row in evaluated if int(row["ordinal"]) > 0]
        metrics = task_metrics[arm]
        arm_summaries.append(
            {
                "arm": arm,
                "tasks": len(metrics),
                "sessions": len(rows),
                "evaluated_sessions": len(evaluated),
                "decision_sessions": len(cross),
                "session_exact_match": statistics.fmean(
                    float(bool(row["exact_match"])) for row in evaluated
                ),
                "cross_session_exact_match": statistics.fmean(
                    float(bool(row["exact_match"])) for row in cross
                ),
                "task_metrics": {
                    name: _summary([values[name] for values in metrics.values()])
                    for name in metric_names
                },
                "effective_ttft_ms": _summary(
                    [float(row["effective_ttft_ms"]) for row in evaluated]
                ),
                "effective_total_ms": _summary(
                    [float(row["effective_total_ms"]) for row in evaluated]
                ),
                "injection_tokens": _summary(
                    [float(row["injection_tokens"]) for row in cross]
                ),
            }
        )

    baseline = task_metrics.get(Arm.MEMORY_OFF.value)
    paired: dict[str, Any] = {}
    if baseline is not None:
        for arm, values in task_metrics.items():
            if arm == Arm.MEMORY_OFF.value:
                continue
            paired[arm] = {
                metric: _paired_bootstrap(
                    values,
                    baseline,
                    metric,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed,
                )
                for metric in metric_names
            }

    protocol = {
        "dataset_manifest_sha256": corpus.manifest_sha256,
        "arms": args.arms,
        "top_k": args.top_k,
        "injection_token_budget": args.injection_token_budget,
        "injection_tokenizer": f"vllm:/tokenize:{args.answer_model}",
        "answer_model": args.answer_model,
        "answer_base_url": args.answer_base_url,
        "embedding_model": args.embedding_model,
        "embedding_base_url": args.embedding_base_url,
        "temperature": 0.0,
        "seed": args.seed,
        "max_answer_tokens": args.max_answer_tokens,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "seed_first_session_with_gold": args.seed_first_session_with_gold,
        "execution_schedule": "task_outer_cyclic_arm_rotation",
        "score": "normalized_exact_match_v1; not upstream LLM judge",
        "response_admission": "generated response is written only after retrieval and answer",
    }
    summary = {
        "schema_version": "memoryarena_agent_outcome_v0.3.0",
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "direct_answer_agent_outcome_track",
        "claim_boundary": (
            "paired local-model direct-answer evidence; not the upstream web-search-agent "
            "or LLM-judge reproduction"
        ),
        "dataset": {
            "config": corpus.config,
            "source_revision": corpus.source_revision,
            "source_sha256": corpus.source_file_sha256,
            "manifest_sha256": corpus.manifest_sha256,
            "tasks_run": min(args.max_tasks, len(corpus.tasks)),
        },
        "protocol": protocol,
        "protocol_sha256": _manifest_hash(protocol),
        "engine": {
            "version": server[0],
            "database": server[1],
            "extensions": {name: version for name, version in extensions},
            "hardware_claim": "x86_64 stock-PG only; no GX10 sign-off",
        },
        "git": _git_state(),
        "model_calls": ledger.summary(),
        "model_tokens": ledger.tokens(),
        "tokenizer_requests": token_counter.requests,
        "arms": arm_summaries,
        "paired_vs_memory_off": paired,
        "artifacts": {
            "outcomes": outcomes_path.name,
            "receipts": receipts_path.name,
            "updates": updates_path.name,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS tasks={summary['dataset']['tasks_run']} arms={len(args.arms)} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
