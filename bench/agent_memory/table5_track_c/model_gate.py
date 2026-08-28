"""Freeze and verify the two local controlled-model endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .protocol import _write_json, benchmark_code_sha256

ANSWER_ENDPOINT = "http://127.0.0.1:8000/v1"
EMBEDDING_ENDPOINT = "http://127.0.0.1:8001/v1"
ANSWER_IDENTITY = [
    {
        "id": "Qwen/Qwen3-32B",
        "root": "Qwen/Qwen3-32B-FP8",
        "max_model_len": 32768,
    }
]
EMBEDDING_IDENTITY = [
    {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "root": "Qwen/Qwen3-Embedding-0.6B",
        "max_model_len": 8192,
    }
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "root": item.get("root"),
            "max_model_len": item.get("max_model_len"),
        }
        for item in payload.get("data") or []
    ]


def run_gate(*, output: Path, expected_code_hash: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model gate receipt: {output}")
    current_hash = benchmark_code_sha256()
    if current_hash != expected_code_hash:
        raise RuntimeError(
            f"model gate code hash mismatch: {current_hash} != {expected_code_hash}"
        )
    receipt: dict[str, Any] = {
        "schema_version": "table5_track_c_model_gate_v1",
        "status": "running",
        "started_at": _now(),
        "benchmark_code_sha256": current_hash,
        "answer_endpoint": ANSWER_ENDPOINT,
        "embedding_endpoint": EMBEDDING_ENDPOINT,
    }
    _write_json(output, receipt)
    try:
        with httpx.Client(timeout=60.0) as client:
            answer_models = _identity(
                client.get(f"{ANSWER_ENDPOINT}/models").raise_for_status().json()
            )
            embedding_models = _identity(
                client.get(f"{EMBEDDING_ENDPOINT}/models").raise_for_status().json()
            )
            embedding_payload = (
                client.post(
                    f"{EMBEDDING_ENDPOINT}/embeddings",
                    json={
                        "model": "Qwen/Qwen3-Embedding-0.6B",
                        "input": ["Track C frozen dimension probe"],
                    },
                )
                .raise_for_status()
                .json()
            )
            vector = embedding_payload["data"][0]["embedding"]
            chat_payload = (
                client.post(
                    f"{ANSWER_ENDPOINT}/chat/completions",
                    json={
                        "model": "Qwen/Qwen3-32B",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Reply with exactly TOKEN_OK and nothing else.",
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 32,
                    },
                )
                .raise_for_status()
                .json()
            )
        message = chat_payload["choices"][0]["message"]
        content = str(message["content"])
        reasoning_content = message.get("reasoning_content")
        checks = {
            "answer_identity_exact": answer_models == ANSWER_IDENTITY,
            "embedding_identity_exact": embedding_models == EMBEDDING_IDENTITY,
            "embedding_dimension_1024": len(vector) == 1024,
            "thinking_disabled_without_request_override": "<think>" not in content
            and "</think>" not in content
            and reasoning_content in (None, ""),
            "deterministic_probe_answer": content.strip() == "TOKEN_OK",
        }
        receipt.update(
            {
                "answer_identity": answer_models,
                "embedding_identity": embedding_models,
                "embedding_dimension": len(vector),
                "embedding_vector_sha256": hashlib.sha256(
                    json.dumps(vector, separators=(",", ":")).encode()
                ).hexdigest(),
                "chat_content": content,
                "chat_reasoning_content": reasoning_content,
                "chat_usage": chat_payload.get("usage"),
                "checks": checks,
                "status": "passed" if all(checks.values()) else "failed",
                "completed_at": _now(),
            }
        )
        _write_json(output, receipt)
        if receipt["status"] != "passed":
            raise RuntimeError(f"model gate failed: {checks}")
        return receipt
    except BaseException as exc:
        receipt.update(
            {
                "status": "failed",
                "failed_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(output, receipt)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    args = parser.parse_args(argv)
    receipt = run_gate(
        output=Path(args.output), expected_code_hash=args.expected_code_hash
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
