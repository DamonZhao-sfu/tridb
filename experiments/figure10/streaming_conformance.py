"""Validate the shared Figure-10 streaming answer path.

The gate deliberately measures the first non-empty SSE content delta rather
than HTTP headers, keep-alives, or empty deltas.  It also checks the server's
prompt-token usage against the frozen local tokenizer for every request.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


SYSTEM_MESSAGE = "You are a concise assistant."
USER_MESSAGE = "Reply with exactly one word: ready"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def distribution(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": None if not rows else sum(rows) / len(rows),
        "p50": percentile(rows, 0.50),
        "p90": percentile(rows, 0.90),
        "p95": percentile(rows, 0.95),
        "p99": percentile(rows, 0.99),
        "max": None if not rows else max(rows),
    }


def prompt_token_count(tokenized_prompt: Any) -> int:
    """Count token ids from list or transformers BatchEncoding output."""
    input_ids = (
        tokenized_prompt["input_ids"]
        if hasattr(tokenized_prompt, "keys") and "input_ids" in tokenized_prompt
        else tokenized_prompt
    )
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise RuntimeError("conformance prompt unexpectedly produced a batch")
        input_ids = input_ids[0]
    return len(input_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.8")
    parser.add_argument("--model-artifact", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.repetitions != 300:
        raise ValueError("Figure-10 B1 requires exactly 300 conformance requests")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "streaming_requests.jsonl"
    receipt_path = output / "conformance_receipt.json"
    if records_path.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing existing conformance output: {output}")

    session = requests.Session()
    models_response = session.get(
        f"{args.endpoint.rstrip('/')}/models", timeout=args.request_timeout
    )
    models_response.raise_for_status()
    model_records = list(models_response.json().get("data", []))
    advertised = [str(row.get("id")) for row in model_records if row.get("id")]
    if advertised != [args.model]:
        raise RuntimeError(
            f"answer endpoint must expose only {args.model!r}; got {advertised!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_artifact,
        revision=args.revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_MESSAGE},
    ]
    tokenized_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # transformers 5.x returns BatchEncoding for this Qwen3.8 tokenizer;
    # len(BatchEncoding) is the number of mapping keys, not the token count.
    local_prompt_tokens = prompt_token_count(tokenized_prompt)
    receipt: dict[str, Any] = {
        "schema_version": "figure10_streaming_conformance_v0.1.0",
        "status": "running",
        "started_at": utc_now(),
        "endpoint": args.endpoint,
        "advertised_models": advertised,
        "model": args.model,
        "model_artifact": args.model_artifact,
        "model_revision": args.revision,
        "thinking": "disabled",
        "stream": True,
        "first_token_definition": "first non-empty SSE delta.content",
        "repetitions": args.repetitions,
        "request_timeout_seconds": args.request_timeout,
        "local_prompt_tokens": local_prompt_tokens,
        "records_path": str(records_path.resolve()),
        "completed_requests": 0,
    }
    atomic_json(receipt_path, receipt)

    records: list[dict[str, Any]] = []
    with records_path.open("x", encoding="utf-8") as sink:
        for index in range(args.repetitions):
            t3 = time.perf_counter_ns()
            response = session.post(
                f"{args.endpoint.rstrip('/')}/chat/completions",
                json={
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": args.max_tokens,
                    "seed": args.seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                stream=True,
                timeout=args.request_timeout,
            )
            headers_at = time.perf_counter_ns()
            response.raise_for_status()
            first_token_at: int | None = None
            last_event_at = headers_at
            empty_delta_count = 0
            keepalive_count = 0
            response_models: set[str] = set()
            pieces: list[str] = []
            usage: dict[str, Any] = {}
            for line in response.iter_lines(decode_unicode=True):
                last_event_at = time.perf_counter_ns()
                if not line:
                    keepalive_count += 1
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                if payload.get("model"):
                    response_models.add(str(payload["model"]))
                if payload.get("usage"):
                    usage = dict(payload["usage"])
                for choice in payload.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str) and content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter_ns()
                        pieces.append(content)
                    else:
                        empty_delta_count += 1
            t5 = time.perf_counter_ns()
            if first_token_at is None:
                raise RuntimeError(f"request {index} produced no non-empty token")
            prompt_tokens = usage.get("prompt_tokens")
            if prompt_tokens != local_prompt_tokens:
                raise RuntimeError(
                    f"request {index} prompt token mismatch: "
                    f"server={prompt_tokens!r}, local={local_prompt_tokens}"
                )
            if response_models and response_models != {args.model}:
                raise RuntimeError(
                    f"request {index} model switched: {sorted(response_models)}"
                )
            record = {
                "request_index": index,
                "t3_ns": t3,
                "headers_at_ns": headers_at,
                "t4_ns": first_token_at,
                "last_sse_event_at_ns": last_event_at,
                "t5_ns": t5,
                "header_seconds": (headers_at - t3) / 1e9,
                "queue_prefill_seconds": (first_token_at - t3) / 1e9,
                "decode_seconds": (t5 - first_token_at) / 1e9,
                "total_seconds": (t5 - t3) / 1e9,
                "empty_delta_count": empty_delta_count,
                "keepalive_count": keepalive_count,
                "first_token_after_headers": first_token_at >= headers_at,
                "text": "".join(pieces).strip(),
                "usage": usage,
                "response_models": sorted(response_models),
            }
            sink.write(json.dumps(record, sort_keys=True) + "\n")
            sink.flush()
            records.append(record)
            if (index + 1) % 25 == 0:
                receipt["completed_requests"] = index + 1
                atomic_json(receipt_path, receipt)
                print(f"[figure10-b1] {index + 1}/{args.repetitions}", flush=True)

    receipt.update(
        {
            "status": "passed",
            "completed_at": utc_now(),
            "completed_requests": len(records),
            "failures": 0,
            "model_switches": 0,
            "prompt_token_mismatches": 0,
            "stream_contract": {
                "all_first_tokens_non_empty": True,
                "all_first_tokens_after_headers": all(
                    row["first_token_after_headers"] for row in records
                ),
                "empty_deltas_observed": sum(
                    row["empty_delta_count"] for row in records
                ),
                "keepalives_observed": sum(row["keepalive_count"] for row in records),
            },
            "latency_seconds": {
                "queue_prefill": distribution(
                    row["queue_prefill_seconds"] for row in records
                ),
                "decode": distribution(row["decode_seconds"] for row in records),
                "total": distribution(row["total_seconds"] for row in records),
            },
        }
    )
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except BaseException as exc:
        receipt_path = args.output_dir / "conformance_receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.update(
                {
                    "status": "failed",
                    "failed_at": utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            atomic_json(receipt_path, receipt)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
