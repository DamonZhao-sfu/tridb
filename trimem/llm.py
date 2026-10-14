"""LLM clients that judge / score / label ``(predicate, text)`` pairs for the TriMEM worker."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import httpx

from trimem import DEFAULT_MODEL

SYSTEM_PROMPT = (
    "You are a strict semantic filter. Decide whether the TEXT satisfies the PREDICATE. "
    'Answer with JSON only: {"verdict": true} or {"verdict": false}.'
)


@dataclass(frozen=True)
class Verdict:
    # verdict (filter), score in [0, 1] (score) or label (extract); None = invalid (not cached)
    value: bool | float | str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


class Judge(Protocol):
    model: str

    def judge_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]: ...

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]: ...

    def extract_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]: ...


SCORE_PROMPT = (
    "You rate how well the TEXT satisfies the PREDICATE. Answer with JSON only: "
    '{"score": n} where n is an integer from 0 (not at all) to 10 (fully).'
)


def parse_score(text: str) -> float | None:
    """Strict: a JSON object with a numeric ``score`` in [0, 10], returned scaled to [0, 1]."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    v = obj.get("score") if isinstance(obj, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 10:
        return None
    return float(v) / 10.0


EXTRACT_PROMPT = (
    "Answer the QUESTION about the TEXT with a short label only: one word or a short phrase, "
    "lower-case. If the QUESTION lists the allowed answers, use exactly one of them. "
    'Answer with JSON only: {"label": "..."}.'
)
LABEL_MAX_CHARS = 64


def parse_label(text: str) -> str | None:
    """Strict: a JSON object with a non-empty string ``label`` of at most LABEL_MAX_CHARS
    characters after normalisation (lower-case, collapsed whitespace, no surrounding punctuation)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    v = obj.get("label") if isinstance(obj, dict) else None
    if not isinstance(v, str):
        return None
    label = " ".join(v.split()).lower().strip(" .,;:!?\"'")
    if not label or len(label) > LABEL_MAX_CHARS:
        return None
    return label


def parse_verdict(text: str) -> bool | None:
    """Strict: a JSON object with a boolean ``verdict``; anything else is invalid."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    value = obj.get("verdict") if isinstance(obj, dict) else None
    return value if isinstance(value, bool) else None


def user_prompt(predicate: str, text: str) -> str:
    return f"PREDICATE: {predicate}\nTEXT: {text}"


def extract_prompt(question: str, text: str) -> str:
    return f"QUESTION: {question}\nTEXT: {text}"


class OpenAIJudge:
    """OpenAI-compatible chat endpoint (vLLM). Thinking off, temperature 0, seed 0."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = DEFAULT_MODEL,
        *,
        concurrency: int = 32,
        timeout: float = 120.0,
        retries: int = 3,
        max_tokens: int = 16,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.concurrency = concurrency
        self.retries = retries
        self.max_tokens = max_tokens
        self._http = httpx.Client(timeout=timeout)

    def _one(
        self,
        pair: tuple[str, str],
        system: str = SYSTEM_PROMPT,
        parse=None,
        *,
        render: Callable[[str, str], str] = user_prompt,
        max_tokens: int | None = None,
    ) -> Verdict:
        parse = parse or parse_verdict
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": render(*pair)},
            ],
            "temperature": 0,
            "seed": 0,
            "max_tokens": max_tokens or self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        delay = 1.0
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=body)
                resp.raise_for_status()
            except httpx.HTTPError:
                if attempt == self.retries:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            data = resp.json()
            usage = data.get("usage") or {}
            text = data["choices"][0]["message"].get("content") or ""
            return Verdict(
                parse(text),
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                time.perf_counter() - started,
            )
        raise AssertionError("unreachable")

    def judge_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]:
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(pairs))) as pool:
            return list(pool.map(self._one, pairs))

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]:
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(pairs))) as pool:
            return list(
                pool.map(lambda p: self._one(p, SCORE_PROMPT, parse_score), pairs)
            )

    def extract_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]:
        """Labels for ``(question, text)`` pairs; a label needs more tokens than a verdict."""
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(pairs))) as pool:
            return list(
                pool.map(
                    lambda p: self._one(
                        p,
                        EXTRACT_PROMPT,
                        parse_label,
                        render=extract_prompt,
                        max_tokens=max(self.max_tokens, 48),
                    ),
                    pairs,
                )
            )


class MockJudge:
    """Deterministic judge for tests: ``rule(predicate, text) -> bool``.

    ``extract_rule(question, text) -> label | None`` defaults to ``"yes"`` / ``"no"`` from
    ``rule``; a ``None`` label behaves like an unparsable LLM reply.
    """

    def __init__(
        self,
        rule: Callable[[str, str], bool],
        model: str = DEFAULT_MODEL,
        score_rule: Callable[[str, str], float] | None = None,
        extract_rule: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.rule = rule
        self.score_rule = score_rule or (lambda p, t: 1.0 if rule(p, t) else 0.0)
        self.extract_rule = extract_rule or (lambda q, t: "yes" if rule(q, t) else "no")
        self.model = model
        self.calls = 0

    def judge_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]:
        self.calls += len(pairs)
        return [Verdict(bool(self.rule(p, t))) for p, t in pairs]

    def score_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]:
        self.calls += len(pairs)
        return [Verdict(float(self.score_rule(p, t))) for p, t in pairs]

    def extract_many(self, pairs: Sequence[tuple[str, str]]) -> list[Verdict]:
        self.calls += len(pairs)
        return [Verdict(self.extract_rule(q, t)) for q, t in pairs]
