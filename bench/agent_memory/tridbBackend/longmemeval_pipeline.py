"""Run the MemoryAgentBench LongMemEval workload end to end on TriDB.

The paper-compatible workload shape is five independent long histories with
60 questions per history.  Each history is chunked and inserted into TriDB
once, then reused for all of its questions.  Answer generation is streamed so
effective TTFT includes query embedding, retrieval, prompt assembly, vLLM
queueing, and prefill.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.tridbBackend.backend import (
    DEFAULT_DSN,
    MemoryUnit,
    TriDBMemoryBackend,
)

# Moved to bench/agent_memory/tridbBackend/chunking.py so the GEM
# DeterministicIngest strategy chunks through the SAME code path as this
# pipeline — the G2 regression gate compares the two and is only meaningful if
# they cannot drift.
from bench.agent_memory.tridbBackend.chunking import TiktokenSentenceChunker

# Same reason, one level up: the GEM arm (bench/agent_memory/gem_bench) and this
# embedRAG arm must load the workload, stream generation, time TTFT, and grade
# through identical code, or [AM] Fig. 2 would compare two harnesses rather than
# two memory systems.
from bench.agent_memory.serving import (
    DEFAULT_ANSWER_BASE_URL,
    DEFAULT_ANSWER_MODEL,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SOURCE,
    QUERY_TEMPLATE,
    SYSTEM_MESSAGE,
    CallLedger,
    LongMemEvalQuestion,
    LongMemEvalWorkload,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    StreamingChatResult,
    _atomic_write_json,
    _git_state,
    _sha256,
    _utc_now,
    _validate_single_model,
    _wilson_interval,
    _write_jsonl,
    build_answer_messages,
    build_judge_prompt,
    extract_retrieval_query,
    fit_answer_prompt,
    judge_protocol_label,
    latency_summary,
    load_workloads,
    parse_judge_yes_no,
)

__all__ = [
    "DEFAULT_ANSWER_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "LongMemEvalQuestion",
    "LongMemEvalWorkload",
    "QUERY_TEMPLATE",
    "SYSTEM_MESSAGE",
    "StreamingChatResult",
    "build_answer_messages",
    "build_judge_prompt",
    "build_summary",
    "extract_retrieval_query",
    "fit_answer_prompt",
    "latency_summary",
    "load_workloads",
    "parse_judge_yes_no",
    "run_pipeline",
]


def build_summary(
    *,
    manifest: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
    judge_results: Sequence[dict[str, Any]],
    ledger: CallLedger,
    construction_records: Sequence[dict[str, Any]],
    lifecycle_seconds: float,
) -> dict[str, Any]:
    judged = [result for result in judge_results if "correct" in result]
    correct = sum(bool(result["correct"]) for result in judged)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for result in judged:
        by_type[str(result["question_type"])].append(bool(result["correct"]))

    timings = [prediction["timing"] for prediction in predictions]
    return {
        "schema_version": "tridb_longmemeval_pipeline_v0.1.0",
        "status": (
            "completed"
            if len(predictions) == manifest["workload"]["selected_questions"]
            else "partial"
        ),
        "paper_reference": {
            "configuration": "embedRAG",
            "accuracy": 0.398,
            "lifecycle_wall_seconds": 14.4 * 60,
            "model_calls": 610,
            "ttft_p50_seconds": 1.96,
            "total_time_p50_seconds": 2.78,
        },
        "accuracy": {
            "judge_model": manifest["models"].get("judge"),
            "judge_protocol": manifest["evaluation"]["judge_protocol"],
            "judged": len(judged),
            "correct": correct,
            "accuracy": None if not judged else correct / len(judged),
            "wilson_95": _wilson_interval(correct, len(judged)),
            "by_question_type": {
                question_type: {
                    "count": len(values),
                    "correct": sum(values),
                    "accuracy": sum(values) / len(values),
                }
                for question_type, values in sorted(by_type.items())
            },
        },
        "walltime": {
            "construction_plus_qa_seconds": lifecycle_seconds,
            "construction_seconds_sum": sum(
                float(record["construction_seconds"]) for record in construction_records
            ),
            "qa_seconds_sum": sum(float(timing["total_seconds"]) for timing in timings),
            "judge_seconds_excluded": sum(
                float(result.get("judge_seconds", 0)) for result in judge_results
            ),
            "by_history": list(construction_records),
        },
        "latency": {
            "effective_ttft_seconds": latency_summary(
                timing["effective_ttft_seconds"] for timing in timings
            ),
            "total_time_seconds": latency_summary(
                timing["total_seconds"] for timing in timings
            ),
            "query_embedding_seconds": latency_summary(
                timing["query_embedding_seconds"] for timing in timings
            ),
            "tridb_retrieval_seconds": latency_summary(
                timing["tridb_retrieval_seconds"] for timing in timings
            ),
            "prompt_assembly_seconds": latency_summary(
                timing["prompt_assembly_seconds"] for timing in timings
            ),
            "vllm_queue_prefill_seconds": latency_summary(
                timing["vllm_queue_prefill_seconds"] for timing in timings
            ),
            "decode_seconds": latency_summary(
                timing["decode_seconds"] for timing in timings
            ),
        },
        "calls": ledger.summary(),
        "tokens": {
            "prompt_tokens": sum(
                int(prediction.get("usage", {}).get("prompt_tokens", 0))
                for prediction in predictions
            ),
            "completion_tokens": sum(
                int(prediction.get("usage", {}).get("completion_tokens", 0))
                for prediction in predictions
            ),
        },
        "comparability": {
            "answer_model_match": (
                manifest["models"]["answer"] == DEFAULT_ANSWER_MODEL
            ),
            "embedding_model_match": (
                manifest["models"]["embedding"] == DEFAULT_EMBEDDING_MODEL
            ),
            "paper_hardware_match": False,
            "judge_protocol_match": (
                manifest["evaluation"]["judge_protocol"] == "memoryagentbench_gpt4o"
            ),
            "note": (
                "Current-hardware TriDB reproduction. Absolute walltime and latency "
                "must not be presented as an H100 hardware replication."
            ),
        },
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    workloads = load_workloads(
        args.input,
        source=args.source,
        strict_shape=not args.allow_nonstandard_shape,
    )
    if args.limit_samples is not None:
        workloads = workloads[: args.limit_samples]
    if args.limit_questions is not None:
        workloads = [
            LongMemEvalWorkload(
                scope_id=workload.scope_id,
                source=workload.source,
                context=workload.context,
                questions=workload.questions[: args.limit_questions],
            )
            for workload in workloads
        ]
    selected_questions = sum(len(workload.questions) for workload in workloads)
    if selected_questions == 0:
        raise ValueError("no questions selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = CallLedger()
    answer_client = OpenAIChatClient(
        args.answer_base_url,
        args.answer_api_key,
        timeout=args.request_timeout,
        ledger=ledger,
    )
    embedding_client = OpenAIEmbeddingClient(
        args.embedding_base_url,
        args.embedding_api_key,
        args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.request_timeout,
        ledger=ledger,
    )
    advertised_answer_models = answer_client.discover_models()
    advertised_embedding_models = embedding_client.discover_models()
    _validate_single_model(
        advertised_answer_models,
        args.answer_model,
        endpoint_name="answer endpoint",
    )
    _validate_single_model(
        advertised_embedding_models,
        args.embedding_model,
        endpoint_name="embedding endpoint",
    )

    backend = TriDBMemoryBackend.connect(
        args.dsn,
        dim=args.embedding_dim,
        table=args.table,
    )
    backend.init_schema()
    chunker = TiktokenSentenceChunker(
        chunk_size=args.chunk_size,
        tokenizer_model=args.chunk_tokenizer,
    )

    judge_enabled = not args.skip_judge
    judge_client = (
        OpenAIChatClient(
            args.judge_base_url,
            args.judge_api_key,
            timeout=args.request_timeout,
            ledger=ledger,
        )
        if judge_enabled
        else None
    )
    judge_protocol = judge_protocol_label(
        enabled=judge_enabled,
        model=args.judge_model,
        base_url=args.judge_base_url,
    )
    manifest = {
        "schema_version": "tridb_longmemeval_manifest_v0.1.0",
        "started_at": _utc_now(),
        "input": {
            "path": str(args.input.resolve()),
            "sha256": _sha256(args.input),
            "source": args.source,
        },
        "workload": {
            "histories": len(workloads),
            "questions_per_history": [
                len(workload.questions) for workload in workloads
            ],
            "selected_questions": selected_questions,
            "chunk_size_tokens": args.chunk_size,
            "chunk_tokenizer": args.chunk_tokenizer,
            "top_k": args.top_k,
            "max_prompt_memories": args.max_prompt_memories,
            "prompt_token_budget": args.prompt_token_budget,
            "embedding_batch_size": args.embedding_batch_size,
            "serial_execution": True,
        },
        "models": {
            "answer": args.answer_model,
            "answer_base_url": args.answer_base_url,
            "answer_endpoint_models": getattr(
                answer_client,
                "model_records",
                [{"id": model} for model in advertised_answer_models],
            ),
            "embedding": args.embedding_model,
            "embedding_base_url": args.embedding_base_url,
            "embedding_endpoint_models": getattr(
                embedding_client,
                "model_records",
                [{"id": model} for model in advertised_embedding_models],
            ),
            "embedding_dim": args.embedding_dim,
            "judge": args.judge_model if judge_enabled else None,
            "judge_base_url": args.judge_base_url if judge_enabled else None,
            "thinking_enabled": False,
        },
        "generation": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.answer_max_tokens,
            "stream": True,
        },
        "evaluation": {
            "judge_enabled": judge_enabled,
            "judge_protocol": judge_protocol,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_calls_excluded_from_serving_metrics": True,
        },
        "tridb": {
            "dsn_redacted": re.sub(r"://[^@]+@", "://***@", args.dsn),
            "table": args.table,
            "mode": "vector_with_relational_scope",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_state(),
        },
    }
    _atomic_write_json(args.output_dir / "run_manifest.json", manifest)

    predictions: list[dict[str, Any]] = []
    construction_records: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    lifecycle_started = time.perf_counter()
    try:
        for history_index, workload in enumerate(workloads):
            construction_started = time.perf_counter()
            chunk_started = time.perf_counter()
            chunks = chunker.chunk(workload.context)
            chunk_seconds = time.perf_counter() - chunk_started
            if not chunks:
                raise RuntimeError(f"{workload.scope_id} produced no chunks")

            embedding_started = time.perf_counter()
            vectors = embedding_client.encode(chunks, phase="construction")
            embedding_seconds = time.perf_counter() - embedding_started
            if any(len(vector) != args.embedding_dim for vector in vectors):
                dimensions = sorted({len(vector) for vector in vectors})
                raise RuntimeError(
                    f"embedding endpoint returned dimensions {dimensions}, "
                    f"expected {args.embedding_dim}"
                )
            units = [
                MemoryUnit(
                    scope_id=workload.scope_id,
                    external_id=f"{workload.scope_id}_chunk_{index:04d}",
                    session_id=workload.scope_id,
                    kind="chunk",
                    content=chunk,
                    event_order=index,
                    metadata={
                        "source": workload.source,
                        "history_index": history_index,
                        "chunk_index": index,
                    },
                    embedding=vector,
                )
                for index, (chunk, vector) in enumerate(
                    zip(chunks, vectors, strict=True)
                )
            ]
            insert_started = time.perf_counter()
            insert_status = "ok"
            try:
                inserted = backend.replace_scope(
                    workload.scope_id,
                    units,
                    isolated=True,
                )
            except BaseException:
                insert_status = "error"
                raise
            finally:
                insert_seconds = time.perf_counter() - insert_started
                ledger.record(
                    kind="tridb_insert",
                    phase="construction",
                    target=args.table,
                    items=len(units),
                    elapsed_seconds=insert_seconds,
                    status=insert_status,
                )
            if inserted != len(units):
                raise RuntimeError(f"TriDB inserted {inserted} of {len(units)} chunks")
            construction_seconds = time.perf_counter() - construction_started
            construction_record = {
                "history_index": history_index,
                "scope_id": workload.scope_id,
                "chunks": len(chunks),
                "questions": len(workload.questions),
                "construction_start_offset_seconds": (
                    construction_started - lifecycle_started
                ),
                "construction_end_offset_seconds": (
                    time.perf_counter() - lifecycle_started
                ),
                "chunking_seconds": chunk_seconds,
                "construction_embedding_seconds": embedding_seconds,
                "tridb_insert_and_index_seconds": insert_seconds,
                "construction_seconds": construction_seconds,
            }
            construction_records.append(construction_record)
            event_rows.append(
                {
                    "event": "construction_complete",
                    "history_index": history_index,
                    **construction_record,
                }
            )

            for question_index, question in enumerate(workload.questions):
                admitted = time.perf_counter()
                retrieval_query = extract_retrieval_query(question.question)

                query_embedding_started = time.perf_counter()
                query_vector = embedding_client.encode(
                    [retrieval_query],
                    phase="query",
                )[0]
                query_embedding_seconds = time.perf_counter() - query_embedding_started
                query_embedding_ended = time.perf_counter()
                if len(query_vector) != args.embedding_dim:
                    raise RuntimeError(
                        f"query embedding has {len(query_vector)} dimensions; "
                        f"expected {args.embedding_dim}"
                    )

                retrieval_started = time.perf_counter()
                retrieval_status = "ok"
                try:
                    hits = backend.search(
                        workload.scope_id,
                        query_embedding=query_vector,
                        k=args.top_k,
                    )
                except BaseException:
                    retrieval_status = "error"
                    raise
                finally:
                    retrieval_seconds = time.perf_counter() - retrieval_started
                    retrieval_ended = time.perf_counter()
                    ledger.record(
                        kind="tridb_retrieval",
                        phase="qa",
                        target=args.table,
                        items=1,
                        elapsed_seconds=retrieval_seconds,
                        status=retrieval_status,
                    )

                prompt_started = time.perf_counter()
                messages, prompt_hit_count, estimated_prompt_tokens = fit_answer_prompt(
                    question.question,
                    [hit.content for hit in hits[: args.max_prompt_memories]],
                    token_counter=chunker.count,
                    token_budget=args.prompt_token_budget,
                )
                prompt_seconds = time.perf_counter() - prompt_started
                prompt_ended = time.perf_counter()
                generation = answer_client.stream_chat(
                    model=args.answer_model,
                    messages=messages,
                    max_tokens=args.answer_max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                completed = generation.completed_at
                timing = {
                    "query_admitted_offset_seconds": admitted - lifecycle_started,
                    "query_embedding_start_offset_seconds": (
                        query_embedding_started - lifecycle_started
                    ),
                    "query_embedding_end_offset_seconds": (
                        query_embedding_ended - lifecycle_started
                    ),
                    "retrieval_start_offset_seconds": (
                        retrieval_started - lifecycle_started
                    ),
                    "retrieval_end_offset_seconds": (
                        retrieval_ended - lifecycle_started
                    ),
                    "prompt_assembly_start_offset_seconds": (
                        prompt_started - lifecycle_started
                    ),
                    "prompt_assembly_end_offset_seconds": (
                        prompt_ended - lifecycle_started
                    ),
                    "generation_request_sent_offset_seconds": (
                        generation.request_started - lifecycle_started
                    ),
                    "generation_first_token_offset_seconds": (
                        generation.first_token_at - lifecycle_started
                    ),
                    "generation_last_token_offset_seconds": (
                        generation.completed_at - lifecycle_started
                    ),
                    "query_embedding_seconds": query_embedding_seconds,
                    "tridb_retrieval_seconds": retrieval_seconds,
                    "prompt_assembly_seconds": prompt_seconds,
                    "vllm_queue_prefill_seconds": (
                        generation.first_token_at - generation.request_started
                    ),
                    "effective_ttft_seconds": (generation.first_token_at - admitted),
                    "decode_seconds": (
                        generation.completed_at - generation.first_token_at
                    ),
                    "total_seconds": completed - admitted,
                }
                prediction = {
                    "history_index": history_index,
                    "question_index": question_index,
                    "scope_id": workload.scope_id,
                    "question_id": question.question_id,
                    "qa_pair_id": question.qa_pair_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "answer": question.answer,
                    "prediction": generation.text,
                    "retrieval_query": retrieval_query,
                    "retrieved": [
                        {
                            "rank": rank,
                            "external_id": hit.external_id,
                            "score": hit.score,
                            "content": hit.content,
                        }
                        for rank, hit in enumerate(hits, start=1)
                    ],
                    "retrieved_count": len(hits),
                    "prompt_memory_count": prompt_hit_count,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "usage": generation.usage,
                    "timing": timing,
                }
                predictions.append(prediction)
                event_rows.append(
                    {
                        "event": "query_complete",
                        "history_index": history_index,
                        "question_index": question_index,
                        "question_id": question.question_id,
                        **timing,
                    }
                )
                _write_jsonl(
                    args.output_dir / "predictions.jsonl",
                    predictions,
                )
                _write_jsonl(args.output_dir / "events.jsonl", event_rows)
                print(
                    f"[longmemeval] {len(predictions)}/{selected_questions} "
                    f"{question.question_id} ttft={timing['effective_ttft_seconds']:.3f}s "
                    f"total={timing['total_seconds']:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        backend.close()
    serving_completed = time.perf_counter()
    lifecycle_seconds = serving_completed - lifecycle_started

    judge_results: list[dict[str, Any]] = []
    if judge_client is not None:
        for index, prediction in enumerate(predictions, start=1):
            prompt = build_judge_prompt(
                prediction["question_type"],
                prediction["question"],
                prediction["answer"],
                prediction["prediction"],
                abstention="_abs" in prediction["question_id"],
            )
            raw, usage, judge_seconds = judge_client.chat(
                model=args.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.judge_max_tokens,
                temperature=0.0,
                seed=args.seed,
                ledger_kind="judge",
            )
            result = {
                "question_id": prediction["question_id"],
                "question_type": prediction["question_type"],
                "judge_model": args.judge_model,
                "correct": parse_judge_yes_no(raw),
                "raw": raw,
                "usage": usage,
                "judge_seconds": judge_seconds,
            }
            judge_results.append(result)
            _write_jsonl(
                args.output_dir / "judge_results.jsonl",
                judge_results,
            )
            print(
                f"[longmemeval-judge] {index}/{len(predictions)} "
                f"{prediction['question_id']} correct={result['correct']}",
                file=sys.stderr,
                flush=True,
            )

    _write_jsonl(args.output_dir / "call_ledger.jsonl", ledger.records)
    summary = build_summary(
        manifest=manifest,
        predictions=predictions,
        judge_results=judge_results,
        ledger=ledger,
        construction_records=construction_records,
        lifecycle_seconds=lifecycle_seconds,
    )
    summary["completed_at"] = _utc_now()
    _atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--dsn", default=os.environ.get("TRIDB_DSN", DEFAULT_DSN))
    parser.add_argument("--table", default="longmemeval_mab_units")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-prompt-memories",
        type=int,
        default=5,
        help=(
            "maximum retrieved chunks assembled for the local answer model; "
            "the paper uses five when ten 4,096-token chunks overflow context"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--chunk-tokenizer", default="gpt-4o-mini")
    parser.add_argument(
        "--prompt-token-budget",
        type=int,
        default=36_000,
        help=(
            "estimated input-token budget; lower-ranked retrieved chunks are "
            "dropped before generation when they do not fit"
        ),
    )
    parser.add_argument(
        "--answer-base-url",
        default=os.environ.get("VLLM_BASE_URL", DEFAULT_ANSWER_BASE_URL),
    )
    parser.add_argument(
        "--answer-api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get(
            "EMBEDDING_BASE_URL",
            DEFAULT_EMBEDDING_BASE_URL,
        ),
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.environ.get("EMBEDDING_API_KEY", "EMPTY"),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help=(
            "texts per embedding HTTP request; 64 yields about two construction "
            "calls per 360K-token history and matches the paper's call accounting"
        ),
    )
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--allow-nonstandard-shape", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get(
            "JUDGE_BASE_URL",
            "https://api.openai.com/v1",
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--judge-max-tokens", type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_prompt_memories <= 0:
        parser.error("--max-prompt-memories must be positive")
    if args.max_prompt_memories > args.top_k:
        parser.error("--max-prompt-memories cannot exceed --top-k")
    if args.prompt_token_budget <= 0:
        parser.error("--prompt-token-budget must be positive")
    if args.limit_samples is not None and args.limit_samples <= 0:
        parser.error("--limit-samples must be positive")
    if args.limit_questions is not None and args.limit_questions <= 0:
        parser.error("--limit-questions must be positive")
    if not args.skip_judge and not args.judge_api_key:
        parser.error(
            "--judge-api-key/OPENAI_API_KEY is required unless --skip-judge is used"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
