"""LLM clients for semantic operators: a real OpenAI-compatible client and a mock.

The LLM is an external pure function (ADR-0026): nothing here touches the database.
Both clients expose the same two calls:

    complete(system, user)            -> LLMReply
    complete_many([(system, user)])   -> list[LLMReply]   (order-preserving)

``complete_many`` is how a semantic operator batches its cache misses; the real client
issues them concurrently against vLLM, which batches them on the GPU.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> LLMReply: ...

    def complete_many(self, prompts: Sequence[tuple[str, str]]) -> list[LLMReply]: ...


class OpenAIChatClient:
    """Pinned (temperature 0, seed 0, thinking off) chat client with bounded concurrency."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        *,
        model: str | None = None,
        api_key: str = "EMPTY",
        max_tokens: int = 256,
        timeout: float = 180.0,
        concurrency: int = 32,
        retries: int = 6,
    ) -> None:
        import requests

        self.retries = max(0, retries)

        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.concurrency = max(1, concurrency)
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.concurrency, pool_maxsize=self.concurrency
        )
        self._session.mount("http://", adapter)
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        if model is None:
            listing = self._session.get(f"{self.base_url}/models", timeout=10).json()
            model = str(listing["data"][0]["id"])
        self.model = model

    def complete(self, system: str, user: str) -> LLMReply:
        """One pinned call; transient transport/5xx failures retry with backoff."""
        import requests

        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return self._complete_once(system, user)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.retries:
                    raise
                last = exc
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status < 500 or attempt == self.retries:
                    raise
                last = exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable") from last

    def _complete_once(self, system: str, user: str) -> LLMReply:
        started = time.perf_counter()
        response = self._session.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        return LLMReply(
            text=str(payload["choices"][0]["message"]["content"] or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            seconds=time.perf_counter() - started,
        )

    def complete_many(self, prompts: Sequence[tuple[str, str]]) -> list[LLMReply]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.complete(*prompts[0])]
        workers = min(self.concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda p: self.complete(*p), prompts))


@dataclass
class MockLLM:
    """Deterministic stand-in for the LLM, used for tests and scale runs.

    The answer is a pure function of (system, user), so cache equivalence and
    eager/deferred equivalence can be checked exactly. ``latency_s`` simulates a
    per-call service time; ``batch_latency_s`` is charged once per ``complete_many``
    call on top of the per-call cost divided by ``concurrency`` (vLLM-like batching).
    ``responder`` decides the text; the default emits a hash-derived verdict.
    """

    model: str = "mock-llm-v1"
    latency_s: float = 0.0
    concurrency: int = 32
    responder: Callable[[str, str], str] | None = None
    calls: int = field(default=0, init=False)
    sleep: Callable[[float], None] = time.sleep

    def _answer(self, system: str, user: str) -> str:
        if self.responder is not None:
            return self.responder(system, user)
        digest = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()
        return json.dumps({"verdict": int(digest[:2], 16) % 2 == 0, "h": digest[:12]})

    def _reply(self, system: str, user: str, seconds: float) -> LLMReply:
        self.calls += 1
        text = self._answer(system, user)
        return LLMReply(
            text=text,
            prompt_tokens=(len(system) + len(user)) // 4,
            completion_tokens=max(1, len(text) // 4),
            seconds=seconds,
        )

    def complete(self, system: str, user: str) -> LLMReply:
        if self.latency_s:
            self.sleep(self.latency_s)
        return self._reply(system, user, self.latency_s)

    def complete_many(self, prompts: Sequence[tuple[str, str]]) -> list[LLMReply]:
        if not prompts:
            return []
        waves = -(-len(prompts) // max(1, self.concurrency))
        wall = waves * self.latency_s
        if wall:
            self.sleep(wall)
        return [self._reply(s, u, self.latency_s) for s, u in prompts]


def parse_json_reply(text: str) -> Any:
    """Parse a model reply as JSON, tolerating markdown fences only."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
