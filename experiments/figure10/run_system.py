"""Build one frozen Figure-10 memory snapshot and measure serial TTFT:Total."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import requests

from bench.agent_memory.serving import build_answer_messages
from bench.agent_memory.table5_track_c.dataset import EventItem, QueryItem
from bench.agent_memory.table5_track_c.tracing import SpanRecorder


SYSTEM_INTERPRETER = {
    "tridb_gem": "/local-scratch/localhome/hza214/tridb/.venv/bin/python",
    "mem0": "/localhome/hza214/agent-memory-table5/venv/mem0/bin/python",
    "memos": "/localhome/hza214/agent-memory-table5/venv/memos/bin/python",
    "cognee": "/localhome/hza214/agent-memory-table5/venv/cognee/bin/python",
    "mandol": "/localhome/hza214/agent-memory-table5/venv/mandol-main/bin/python",
    "graphiti": "/localhome/hza214/agent-memory-table5/venv/graphiti/bin/python",
    "evermemos": "/localhome/hza214/agent-memory-table5/venv/evermemos/bin/python",
}

ANSWER_MODEL = "qwen3.8"
ANSWER_ENDPOINT = "http://127.0.0.1:8000/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_ENDPOINT = "http://127.0.0.1:8011/v1"
FROZEN_TIMESTAMP = "12:00 AM on 01 January, 2024"
SECRET_PREFIX = "env:"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, ensure_ascii=False, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def resolve_secrets(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, str) and value.startswith(SECRET_PREFIX):
            variable = value[len(SECRET_PREFIX) :]
            secret = os.environ.get(variable)
            if not secret:
                raise RuntimeError(f"configuration needs unset environment {variable}")
            resolved[key] = secret
        else:
            resolved[key] = value
    return resolved


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if any(
                    token in str(key).lower()
                    for token in ("password", "secret", "api_key")
                )
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1<redacted>\2", value)
    return value


def factory(system: str, config: dict[str, Any]) -> Any:
    if system == "tridb_gem":
        from bench.agent_memory.table5_track_c.adapters import (
            TriDBGEMAdapter,
            TriDBGEMConfig,
        )

        return TriDBGEMAdapter(TriDBGEMConfig(**config))
    if system == "mem0":
        from bench.agent_memory.table5_track_c.adapters import Mem0Adapter, Mem0Config

        return Mem0Adapter(Mem0Config(**config))
    if system == "memos":
        from bench.agent_memory.table5_track_c.adapters import MemosAdapter, MemosConfig

        return MemosAdapter(MemosConfig(**config))
    if system == "cognee":
        from bench.agent_memory.table5_track_c.adapters import (
            CogneeAdapter,
            CogneeConfig,
        )

        return CogneeAdapter(CogneeConfig(**config))
    if system == "mandol":
        from experiments.mandol_track_c.adapter import MandolAdapter, MandolConfig

        return MandolAdapter(MandolConfig(**config))
    if system == "graphiti":
        from experiments.graphiti_track_c.adapter import (
            GraphitiTrackCAdapter,
            GraphitiTrackCConfig,
        )

        return GraphitiTrackCAdapter(GraphitiTrackCConfig(**config))
    if system == "evermemos":
        from experiments.evermemos_track_c.adapter import (
            EverMemOSTrackCAdapter,
            EverMemOSTrackCConfig,
        )

        return EverMemOSTrackCAdapter(EverMemOSTrackCConfig(**config))
    raise ValueError(f"unknown system {system!r}")


class CaptureSpanRecorder(SpanRecorder):
    """Retain just enough in memory to place t1 and derive a call ledger."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.by_trace: dict[str, list[dict[str, Any]]] = {}

    def write(self, record: Mapping[str, Any]) -> None:
        row = dict(record)
        super().write(row)
        trace_id = str(row.get("trace_id") or "")
        if trace_id:
            self.by_trace.setdefault(trace_id, []).append(row)

    def embedding_completion(self, trace_id: str, t0: int, t2: int) -> tuple[int, str]:
        ends = [
            int(row["completed_at_ns"])
            for row in self.by_trace.get(trace_id, [])
            if row.get("category") == "embedding"
            and row.get("completed_at_ns") is not None
        ]
        if ends:
            return max(t0, min(max(ends), t2)), "observed_embedding_span"
        # No work is removed from TTFT: for an opaque native API, t2 is a
        # conservative upper-bound marker for the hidden embedding completion.
        return t2, "opaque_upper_bound_at_context_ready"


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * quantile
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def distribution(values: Iterable[float]) -> dict[str, Any]:
    rows = list(float(value) for value in values)
    return {
        "count": len(rows),
        "mean": None if not rows else sum(rows) / len(rows),
        "p50": percentile(rows, 0.50),
        "p90": percentile(rows, 0.90),
        "p95": percentile(rows, 0.95),
        "p99": percentile(rows, 0.99),
        "max": None if not rows else max(rows),
    }


def endpoint_models(endpoint: str) -> list[str]:
    response = requests.get(f"{endpoint.rstrip('/')}/models", timeout=30)
    response.raise_for_status()
    return [str(row["id"]) for row in response.json().get("data", [])]


def token_counter() -> Callable[[str], int]:
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return lambda text: len(encoding.encode(text))


def fit_prompt(
    question: str,
    contexts: list[str],
    count_tokens: Callable[[str], int],
    budget: int,
) -> tuple[list[dict[str, str]], int, int, bool]:
    def render(rows: list[str]) -> tuple[list[dict[str, str]], int]:
        messages = build_answer_messages(question, rows)
        return messages, sum(count_tokens(row["content"]) for row in messages)

    top_ten = contexts[:10]
    messages, tokens = render(top_ten)
    if tokens <= budget:
        return messages, len(top_ten), tokens, False
    top_five = contexts[:5]
    messages, tokens = render(top_five)
    if tokens > budget:
        raise RuntimeError(
            f"top-5 answer prompt has {tokens} tokens, over frozen budget {budget}"
        )
    return messages, len(top_five), tokens, True


def stream_answer(
    session: requests.Session,
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    seed: int,
) -> dict[str, Any]:
    t3 = time.perf_counter_ns()
    response = session.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    t4: int | None = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    models: set[str] = set()
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        payload = json.loads(data)
        if payload.get("model"):
            models.add(str(payload["model"]))
        if payload.get("usage"):
            usage = dict(payload["usage"])
        for choice in payload.get("choices", []):
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str) and content:
                if t4 is None:
                    t4 = time.perf_counter_ns()
                pieces.append(content)
    t5 = time.perf_counter_ns()
    if t4 is None:
        raise RuntimeError("stream completed without a non-empty answer token")
    if models and models != {model}:
        raise RuntimeError(f"answer model switched: {sorted(models)!r}")
    return {
        "t3_ns": t3,
        "t4_ns": t4,
        "t5_ns": t5,
        "text": "".join(pieces).strip(),
        "usage": usage,
        "response_models": sorted(models),
    }


def load_artifact(path: Path, mode: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    shape = payload.get("shape") or {}
    if shape.get("histories") != 5 or shape.get("questions") != 300:
        raise RuntimeError(f"invalid frozen workload shape: {shape!r}")
    histories = list(payload["histories"])
    if mode == "smoke":
        history = dict(histories[0])
        history["chunks"] = list(history["chunks"][:3])
        history["questions"] = list(history["questions"][:3])
        return [history]
    return histories


def events(history: Mapping[str, Any]) -> list[EventItem]:
    result = []
    for index, row in enumerate(history["chunks"]):
        metadata = dict(row.get("metadata") or {})
        metadata.update({"session_number": 1, "turn_number": index + 1})
        result.append(
            EventItem(
                sample_id=str(history["scope_id"]),
                event_id=str(row["event_id"]),
                session_id="S1",
                timestamp=FROZEN_TIMESTAMP,
                role=str(row.get("role") or "memory_history"),
                text=str(row["text"]),
                ordinal=index + 1,
                metadata=metadata,
            )
        )
    return result


def query_item(history: Mapping[str, Any], row: Mapping[str, Any]) -> QueryItem:
    return QueryItem(
        sample_id=str(history["scope_id"]),
        question_id=str(row["question_id"]),
        question=str(row["retrieval_query"]),
        answer=row.get("answer"),
        category=row.get("question_type"),
        evidence_ids=(),
        ordinal=int(row["question_index"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=sorted(SYSTEM_INTERPRETER))
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--answer-endpoint", default=ANSWER_ENDPOINT)
    parser.add_argument("--answer-model", default=ANSWER_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prompt-token-budget", type=int, default=30_000)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--query-timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    expected = Path(SYSTEM_INTERPRETER[args.system]).resolve()
    if Path(sys.executable).resolve() != expected:
        raise RuntimeError(f"{args.system} requires {expected}, got {sys.executable}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    raw_config = json.loads(args.system_config.read_text(encoding="utf-8"))
    config = resolve_secrets(raw_config)
    histories = load_artifact(args.workload, args.mode)
    models = endpoint_models(args.answer_endpoint)
    if models != [args.answer_model]:
        raise RuntimeError(f"answer endpoint mismatch: {models!r}")
    if endpoint_models(EMBEDDING_ENDPOINT) != [EMBEDDING_MODEL]:
        raise RuntimeError("shared embedding endpoint identity mismatch")
    conformance = (
        args.output_dir.parents[2]
        / "conformance"
        / "shared_answer_path"
        / "conformance_receipt.json"
    )
    if (
        not conformance.exists()
        or json.loads(conformance.read_text()).get("status") != "passed"
    ):
        raise RuntimeError(f"B1 streaming conformance is not passed: {conformance}")

    receipt_path = args.output_dir / "run_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": "figure10_system_run_v0.1.0",
        "status": "running",
        "claim": "PILOT same-host controlled protocol extension",
        "system": args.system,
        "build_id": args.build_id,
        "mode": args.mode,
        "started_at": utc_now(),
        "workload": {
            "path": str(args.workload.resolve()),
            "sha256": sha256(args.workload),
            "histories": len(histories),
            "chunks": sum(len(row["chunks"]) for row in histories),
            "questions": sum(len(row["questions"]) for row in histories),
        },
        "config": redact(raw_config),
        "answer": {
            "endpoint": args.answer_endpoint,
            "model": args.answer_model,
            "stream": True,
            "thinking": False,
            "temperature": 0.0,
            "seed": args.seed,
            "max_tokens": args.answer_max_tokens,
        },
        "query_contract": {
            "serial": True,
            "top_k": args.top_k,
            "overflow_fallback_k": 5,
            "prompt_token_budget": args.prompt_token_budget,
            "timeout_seconds": args.query_timeout,
            "retry_count": 0,
            "answer_writeback": False,
        },
    }
    atomic_json(receipt_path, receipt)

    adapter = factory(args.system, config)
    if hasattr(adapter, "enable_stage_tracing"):
        adapter.enable_stage_tracing()
    predictions_path = args.output_dir / "predictions.jsonl"
    construction_path = args.output_dir / "construction.jsonl"
    ledger_path = args.output_dir / "call_ledger.jsonl"
    spans_path = args.output_dir / "stage_spans.jsonl"
    session = requests.Session()
    count_tokens = token_counter()
    failures = 0
    rows: list[dict[str, Any]] = []
    try:
        receipt["schema_gate"] = adapter.init_schema()
        atomic_json(receipt_path, receipt)
        with construction_path.open("x", encoding="utf-8") as construction_sink:
            for history in histories:
                began = time.perf_counter_ns()
                build = adapter.ingest_history(events(history))
                ended = time.perf_counter_ns()
                record = {
                    "history_index": history["history_index"],
                    "scope_id": history["scope_id"],
                    "chunks": len(history["chunks"]),
                    "started_ns": began,
                    "completed_ns": ended,
                    "wall_seconds": (ended - began) / 1e9,
                    "receipt": build,
                }
                construction_sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                construction_sink.flush()
                print(
                    f"[{args.system}] built history {history['history_index']} "
                    f"({len(history['chunks'])} chunks)",
                    flush=True,
                )
        receipt["build_finalize"] = adapter.finalize_build()
        receipt["build_completed_at"] = utc_now()
        atomic_json(receipt_path, receipt)

        request_index = 0
        with (
            CaptureSpanRecorder(spans_path) as recorder,
            predictions_path.open("x", encoding="utf-8") as prediction_sink,
            ledger_path.open("x", encoding="utf-8") as ledger_sink,
        ):
            for history in histories:
                for question in history["questions"]:
                    item = query_item(history, question)
                    trace_id, root_span_id = recorder.request_ids(
                        args.system, args.build_id, "formal_query", request_index
                    )
                    row: dict[str, Any] = {
                        "schema_version": "figure10_prediction_v0.1.0",
                        "system": args.system,
                        "build_id": args.build_id,
                        "request_index": request_index,
                        "history_index": history["history_index"],
                        "scope_id": history["scope_id"],
                        "question_index": question["question_index"],
                        "question_id": question["question_id"],
                        "question_type": question["question_type"],
                        "question": question["question"],
                        "retrieval_query": question["retrieval_query"],
                        "answer": question.get("answer"),
                        "trace_id": trace_id,
                        "success": False,
                        "error": None,
                    }
                    t0 = time.perf_counter_ns()
                    try:
                        with recorder.bind_request(
                            trace_id=trace_id,
                            root_span_id=root_span_id,
                            system=args.system,
                            build_id=args.build_id,
                            phase="formal_query",
                            request_index=request_index,
                        ):
                            search_receipt = adapter.search(item, top_k=args.top_k)
                        t2 = time.perf_counter_ns()
                        t1, t1_observation = recorder.embedding_completion(
                            trace_id, t0, t2
                        )
                        contexts = [
                            str(value)
                            for value in (search_receipt.get("contexts") or [])
                            if str(value).strip()
                        ]
                        messages, context_count, prompt_tokens, fell_back = fit_prompt(
                            str(question["question"]),
                            contexts,
                            count_tokens,
                            args.prompt_token_budget,
                        )
                        generation = stream_answer(
                            session,
                            endpoint=args.answer_endpoint,
                            model=args.answer_model,
                            messages=messages,
                            max_tokens=args.answer_max_tokens,
                            timeout=args.query_timeout,
                            seed=args.seed,
                        )
                        t3, t4, t5 = (
                            generation["t3_ns"],
                            generation["t4_ns"],
                            generation["t5_ns"],
                        )
                        row.update(
                            {
                                "success": True,
                                "search": search_receipt,
                                "prediction": generation["text"],
                                "usage": generation["usage"],
                                "response_models": generation["response_models"],
                                "prompt_context_count": context_count,
                                "estimated_prompt_tokens": prompt_tokens,
                                "overflow_fallback_to_top5": fell_back,
                                "timing": {
                                    "t0_ns": t0,
                                    "t1_ns": t1,
                                    "t2_ns": t2,
                                    "t3_ns": t3,
                                    "t4_ns": t4,
                                    "t5_ns": t5,
                                    "t1_observation": t1_observation,
                                    "query_embedding_seconds": (t1 - t0) / 1e9,
                                    "post_embedding_retrieval_seconds": (t2 - t1) / 1e9,
                                    "context_ready_seconds": (t2 - t0) / 1e9,
                                    "prompt_assembly_seconds": (t3 - t2) / 1e9,
                                    "answer_queue_prefill_seconds": (t4 - t3) / 1e9,
                                    "effective_ttft_seconds": (t4 - t0) / 1e9,
                                    "decode_seconds": (t5 - t4) / 1e9,
                                    "total_seconds": (t5 - t0) / 1e9,
                                },
                            }
                        )
                        rows.append(row)
                    except Exception as exc:  # noqa: BLE001 - failure is evidence
                        failures += 1
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        row["traceback"] = traceback.format_exc()
                    completed = time.perf_counter_ns()
                    recorder.record_request(
                        trace_id=trace_id,
                        root_span_id=root_span_id,
                        system=args.system,
                        build_id=args.build_id,
                        phase="formal_query",
                        request_index=request_index,
                        started_ns=t0,
                        completed_ns=completed,
                        success=bool(row["success"]),
                        timed_out=False,
                        error=row["error"],
                    )
                    prediction_sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    prediction_sink.flush()
                    trace_spans = recorder.by_trace.get(trace_id, [])
                    for span in trace_spans:
                        if span.get("category") == "llm":
                            ledger_sink.write(
                                json.dumps(
                                    {
                                        "request_index": request_index,
                                        "trace_id": trace_id,
                                        "phase": "memory_retrieval",
                                        "span": span,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    if row["success"]:
                        ledger_sink.write(
                            json.dumps(
                                {
                                    "request_index": request_index,
                                    "trace_id": trace_id,
                                    "phase": "answer_generation",
                                    "model": args.answer_model,
                                    "usage": row["usage"],
                                    "t3_ns": row["timing"]["t3_ns"],
                                    "t5_ns": row["timing"]["t5_ns"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    ledger_sink.flush()
                    request_index += 1
                    if request_index % 25 == 0 or args.mode == "smoke":
                        print(
                            f"[{args.system}] {request_index}/"
                            f"{sum(len(h['questions']) for h in histories)} "
                            f"failures={failures}",
                            flush=True,
                        )

        receipt["final_stats"] = adapter.stats()
        receipt["serving"] = {
            "questions": request_index,
            "successes": len(rows),
            "failures": failures,
            "effective_ttft_seconds": distribution(
                row["timing"]["effective_ttft_seconds"] for row in rows
            ),
            "total_seconds": distribution(
                row["timing"]["total_seconds"] for row in rows
            ),
            "context_ready_seconds": distribution(
                row["timing"]["context_ready_seconds"] for row in rows
            ),
        }
        receipt["status"] = "complete" if failures == 0 else "complete_with_failures"
        receipt["completed_at"] = utc_now()
        atomic_json(receipt_path, receipt)
        return 0 if failures == 0 else 2
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["failed_at"] = utc_now()
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()
        atomic_json(receipt_path, receipt)
        raise
    finally:
        adapter.close()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
