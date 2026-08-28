"""Pinned CPU OpenAI-compatible embedding service for EvoMemBench.

The two physical GPUs are reserved for independent Qwen3.8 answer replicas.
This service loads the exact pinned Qwen3-Embedding-0.6B checkpoint on CPU,
uses the model's declared last-token pooling, and L2-normalizes every vector.
It is deliberately small and fail-closed: overlength inputs are rejected, not
silently truncated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, Sequence


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_DIMENSION = 1024
MAX_MODEL_TOKENS = 32768


def _validate_inputs(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise TypeError("embedding input must be a string or sequence of strings")
    if not items:
        raise ValueError("embedding input must not be empty")
    if len(items) > 128:
        raise ValueError("embedding batch exceeds the fixed 128-item limit")
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError("every embedding input must be a non-empty string")
    return items


def _last_token_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Return each row's final non-padding token for either padding direction."""
    import torch

    if bool((attention_mask[:, -1] == 1).all()):
        return last_hidden_state[:, -1]
    indices = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        indices,
    ]


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    prompt_tokens: int


class CpuEmbeddingEngine:
    def __init__(self, model_path: str, *, threads: int) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads < 1:
            raise ValueError("CPU embedding threads must be positive")
        torch.set_num_threads(threads)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="right",
        )
        self._model = (
            AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
            )
            .to("cpu")
            .eval()
        )
        hidden_size = int(getattr(self._model.config, "hidden_size", -1))
        if hidden_size != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"embedding dimension mismatch: {hidden_size} != {EMBEDDING_DIMENSION}"
            )
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> EmbeddingBatch:
        torch = self._torch
        with self._lock, torch.inference_mode():
            inputs = self._tokenizer(
                list(texts),
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            longest = int(lengths.max().item())
            if longest > MAX_MODEL_TOKENS:
                raise ValueError(
                    f"embedding input has {longest} tokens; maximum is "
                    f"{MAX_MODEL_TOKENS} and truncation is forbidden"
                )
            hidden = self._model(**inputs).last_hidden_state.float()
            pooled = _last_token_pool(hidden, inputs["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return EmbeddingBatch(
                vectors=normalized.cpu().tolist(),
                prompt_tokens=int(lengths.sum().item()),
            )


def create_app(engine: CpuEmbeddingEngine) -> Any:
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="EvoMemBench pinned CPU embedding service")
    created = int(time.time())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": "cpu",
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": created,
                    "owned_by": "local-pinned-cpu",
                    "revision": MODEL_REVISION,
                    "max_model_len": MAX_MODEL_TOKENS,
                    "embedding_dimension": EMBEDDING_DIMENSION,
                }
            ],
        }

    @app.post("/v1/embeddings")
    def embeddings(request: dict[str, Any]) -> dict[str, Any]:
        if request.get("model") != MODEL_ID:
            raise HTTPException(status_code=404, detail="unknown embedding model")
        if request.get("encoding_format") not in (None, "float"):
            raise HTTPException(
                status_code=400,
                detail="only float embedding encoding is supported",
            )
        try:
            texts = _validate_inputs(request.get("input"))
            batch = engine.encode(texts)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "object": "list",
            "model": MODEL_ID,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(batch.vectors)
            ],
            "usage": {
                "prompt_tokens": batch.prompt_tokens,
                "total_tokens": batch.prompt_tokens,
                "completion_tokens": 0,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("EVOMEMBENCH_CPU_EMBED_THREADS", "20")),
    )
    args = parser.parse_args()
    model_path = Path(args.model_path).resolve()
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"pinned embedding model is incomplete: {model_path}")
    engine = CpuEmbeddingEngine(str(model_path), threads=args.threads)
    app = create_app(engine)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
