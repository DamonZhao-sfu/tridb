"""GEM's bridge to the Figure 1 semantic-operator layer (plan P1 wiring).

The ingest strategies (``strategies/llm_mediated.py``, ``strategies/agentic.py``)
call any object with ``complete(system, user) -> (text, usage)``. This module
gives them one that routes through the PG-resident LLM result cache
``gem_sem_cache`` (ADR-0026), keyed by

    (op='gem_ingest_llm', model, prompt_version, sha256(canonical {system, user}))

so that:

* a hit is a table read — the LLM is not called and the usage reports zero
  tokens with ``cache_hit=True``;
* a miss calls the wrapped client and writes the reply back on the cache's
  connection (inside the caller's transaction when one is open);
* ``offline=True`` turns a miss into :class:`CacheMiss`. That is the replay mode:
  a run replayed with the network disabled either reproduces the same state
  from cache alone or fails loudly — it never silently re-derives.

The wrapped client may be a ``semantic.llm`` client (returns ``LLMReply``) or a
strategy-style client such as ``OpenAIChatExtractor`` (returns ``(text, usage)``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from bench.agent_memory.semantic.cache import SemCache, input_hash
from bench.agent_memory.semantic.ops import CacheMiss

#: The cache ``op`` under which ingest-time completions are stored.
INGEST_OP = "gem_ingest_llm"

__all__ = ["INGEST_OP", "CacheMiss", "CacheStats", "CachedCompletionClient"]


class _CacheLike(Protocol):
    def get_many(
        self, op: str, model: str, prompt_version: str, hashes: Any
    ) -> dict[str, Any]: ...

    def put_many(self, op: str, model: str, prompt_version: str, rows: Any) -> int: ...


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0

    def snapshot(self) -> tuple[int, int]:
        return self.hits, self.misses


def _model_of(client: Any) -> str:
    model = getattr(client, "model", None)
    if not model:
        raise ValueError(
            "wrapped client must expose .model (it is part of the cache key)"
        )
    return str(model)


class CachedCompletionClient:
    """``complete(system, user) -> (text, usage)`` behind ``gem_sem_cache``."""

    def __init__(
        self,
        client: Any,
        cache: SemCache | _CacheLike,
        *,
        prompt_version: str,
        op: str = INGEST_OP,
        offline: bool = False,
        model: str | None = None,
    ) -> None:
        if not prompt_version:
            raise ValueError("prompt_version is part of the cache key and must be set")
        self.client = client
        self.cache = cache
        self.prompt_version = str(prompt_version)
        self.op = op
        self.offline = offline
        self.model = str(model) if model else _model_of(client)
        self.stats = CacheStats()

    @staticmethod
    def key(system: str, user: str) -> str:
        return input_hash({"system": system, "user": user})

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        digest = self.key(system, user)
        found = self.cache.get_many(self.op, self.model, self.prompt_version, [digest])
        if digest in found:
            self.stats.hits += 1
            output = found[digest]
            text = output.get("text", "") if isinstance(output, dict) else str(output)
            return str(text), {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cache_hit": True,
            }
        if self.offline:
            raise CacheMiss(f"{self.op}: completion not in cache (offline replay)")
        started = time.perf_counter()
        reply = self.client.complete(system, user)
        seconds = time.perf_counter() - started
        if isinstance(reply, tuple):
            text, usage = str(reply[0] or ""), dict(reply[1] or {})
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
        else:  # semantic.llm.LLMReply
            text = str(reply.text or "")
            prompt_tokens, completion_tokens = (
                int(reply.prompt_tokens),
                int(reply.completion_tokens),
            )
            seconds = float(reply.seconds or seconds)
        self.cache.put_many(
            self.op,
            self.model,
            self.prompt_version,
            [(digest, {"text": text}, prompt_tokens, completion_tokens, seconds)],
        )
        self.stats.misses += 1
        self.stats.prompt_tokens += prompt_tokens
        self.stats.completion_tokens += completion_tokens
        self.stats.llm_seconds += seconds
        return text, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit": False,
        }
