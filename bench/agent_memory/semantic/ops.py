"""Semantic operators behind the PG-resident cache (plan P1).

Every operator goes through :meth:`SemOps.run`:

1. hash each logical input (canonical JSON) and look the batch up in ``gem_sem_cache``;
2. send only the distinct misses to the LLM in one ``complete_many`` batch;
3. parse, write the new verdicts back to the cache in the caller's transaction;
4. charge the meter: ``n_sem`` logical evaluations, ``llm_calls`` real requests.

With ``offline=True`` a miss raises :class:`CacheMiss`; that is the replay mode used to
prove that a run is reproducible from the cache alone.

Operators: sem_extract, sem_filter, sem_match, sem_conflict, sem_rank, sem_topk,
sem_summarize (paper §3). sem_rank / sem_topk re-rank a *bounded* prefix after the
engine stream (ADR-0007: the vector leg stays the single rank authority).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from bench.agent_memory.semantic.cache import SemCache, SemMeter, canonical, input_hash
from bench.agent_memory.semantic.llm import LLMClient, parse_json_reply


class CacheMiss(RuntimeError):
    """Raised in offline (replay) mode when an evaluation is not in the cache."""


class InvalidReply(ValueError):
    """The model's reply does not satisfy the operator's output contract."""


@dataclass(frozen=True)
class OpSpec:
    name: str
    prompt_version: str
    system: str
    parse: Callable[
        [Any, str], Any
    ]  # (parsed_json_or_None, raw) -> output | InvalidReply
    fallback: Any  # fail-closed value for an invalid reply (never cached)


_TRUE = {"true", "yes"}
_FALSE = {"false", "no"}


def _bool_field(*keys: str) -> Callable[[Any, str], Any]:
    """Strict: a JSON bool (or the strings true/false/yes/no) under one of ``keys``."""

    def parse(obj: Any, raw: str) -> Any:
        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    value = obj[key]
                    if isinstance(value, bool):
                        return value
                    if (
                        isinstance(value, str)
                        and value.strip().lower() in _TRUE | _FALSE
                    ):
                        return value.strip().lower() in _TRUE
                    raise InvalidReply(f"{key}={value!r} is not a boolean")
        raise InvalidReply(f"no boolean field {keys} in reply {raw[:80]!r}")

    return parse


def _text_field(*keys: str, limit: int = 600) -> Callable[[Any, str], Any]:
    def parse(obj: Any, raw: str) -> Any:
        if isinstance(obj, dict):
            for key in keys:
                if isinstance(obj.get(key), str) and obj[key].strip():
                    return obj[key][:limit]
            raise InvalidReply(f"no text field {keys} in reply {raw[:80]!r}")
        text = (raw or "").strip()
        if not text:
            raise InvalidReply("empty reply")
        return text[:limit]

    return parse


def _score(obj: Any, raw: str) -> float:
    if isinstance(obj, dict):
        for key in ("score", "relevance"):
            if (
                key in obj
                and isinstance(obj[key], (int, float))
                and not isinstance(obj[key], bool)
            ):
                return float(obj[key])
        if isinstance(obj.get("h"), str):  # MockLLM default reply
            return float(int(obj["h"][:2], 16) % 11)
    raise InvalidReply(f"no numeric score in reply {raw[:80]!r}")


SPECS: dict[str, OpSpec] = {
    "sem_extract": OpSpec(
        "sem_extract",
        "v1",
        "You extract information. Follow the instruction and reply with JSON only: "
        '{"text": "<at most 60 words>"}.',
        _text_field("text", "lesson", "h"),
        "",
    ),
    "sem_filter": OpSpec(
        "sem_filter",
        "v1",
        "You decide whether an item satisfies a condition. Reply with JSON only: "
        '{"keep": true|false}.',
        _bool_field("keep", "verdict"),
        False,
    ),
    "sem_match": OpSpec(
        "sem_match",
        "v1",
        "You decide whether a relation holds between LEFT and RIGHT. Reply with JSON only: "
        '{"match": true|false}.',
        _bool_field("match", "verdict"),
        False,
    ),
    "sem_conflict": OpSpec(
        "sem_conflict",
        "v1",
        "You decide whether statement A and statement B contradict each other. Reply with "
        'JSON only: {"conflict": true|false}.',
        _bool_field("conflict", "verdict"),
        False,
    ),
    "sem_rank": OpSpec(
        "sem_rank",
        "v1",
        "You score how useful an item is for a query on a 0-10 scale. Reply with JSON only: "
        '{"score": <integer 0-10>}.',
        _score,
        0.0,
    ),
    "sem_summarize": OpSpec(
        "sem_summarize",
        "v1",
        "You summarize items under a word budget. Reply with JSON only: "
        '{"text": "<summary>"}.',
        _text_field("text", "summary", "h", limit=4000),
        "",
    ),
}


class SemOps:
    """The semantic operator set over one LLM client and one cache."""

    def __init__(
        self,
        llm: LLMClient,
        cache: SemCache,
        meter: SemMeter | None = None,
        *,
        offline: bool = False,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.meter = meter if meter is not None else SemMeter()
        self.offline = offline

    @property
    def model(self) -> str:
        return str(self.llm.model)

    # -- the one execution path -------------------------------------------

    def run(self, op: str, inputs: Sequence[Any]) -> list[Any]:
        spec = SPECS[op]
        if not inputs:
            return []
        hashes = [input_hash(x) for x in inputs]
        found = self.cache.get_many(op, self.model, spec.prompt_version, hashes)
        miss_order: list[str] = []
        miss_input: dict[str, Any] = {}
        for h, x in zip(hashes, inputs, strict=True):
            if h not in found and h not in miss_input:
                miss_input[h] = x
                miss_order.append(h)
        if miss_order and self.offline:
            raise CacheMiss(f"{op}: {len(miss_order)} evaluation(s) not in cache")
        invalid: dict[str, Any] = {}
        if miss_order:
            prompts = [(spec.system, canonical(miss_input[h])) for h in miss_order]
            started = time.perf_counter()
            replies = self.llm.complete_many(prompts)
            self.meter.wall_seconds += time.perf_counter() - started
            self.meter.batches += 1
            new_rows = []
            for h, reply in zip(miss_order, replies, strict=True):
                self.meter.prompt_tokens += reply.prompt_tokens
                self.meter.completion_tokens += reply.completion_tokens
                self.meter.llm_seconds += reply.seconds
                try:
                    output = spec.parse(parse_json_reply(reply.text), reply.text)
                except InvalidReply:
                    # Fail closed for this call, and do NOT cache: a formatting failure
                    # must not become a permanent semantic decision.
                    invalid[h] = spec.fallback
                    self.meter.invalid += 1
                    continue
                new_rows.append(
                    (
                        h,
                        output,
                        reply.prompt_tokens,
                        reply.completion_tokens,
                        reply.seconds,
                    )
                )
            self.cache.put_many(op, self.model, spec.prompt_version, new_rows)
            # The persisted row is authoritative: if a concurrent writer inserted the same
            # key first (ON CONFLICT DO NOTHING), use ITS answer, so derived state is
            # always reproducible from the cache.
            persisted = self.cache.get_many(
                op, self.model, spec.prompt_version, [r[0] for r in new_rows]
            )
            for h, output, *_ in new_rows:
                found[h] = persisted.get(h, output)
        calls = len(miss_order)
        self.meter.calls += calls
        self.meter.hits += len(inputs) - calls
        self.meter.by_op[op] = self.meter.by_op.get(op, 0) + len(inputs)
        return [found[h] if h in found else invalid[h] for h in hashes]

    # -- operators ----------------------------------------------------------

    def sem_extract(self, instruction: str, items: Sequence[Any]) -> list[str]:
        return self.run(
            "sem_extract", [{"instruction": instruction, "item": x} for x in items]
        )

    def sem_filter(self, predicate: str, items: Sequence[Any]) -> list[bool]:
        return self.run(
            "sem_filter", [{"condition": predicate, "item": x} for x in items]
        )

    def sem_match(self, relation: str, pairs: Sequence[tuple[Any, Any]]) -> list[bool]:
        return self.run(
            "sem_match",
            [{"relation": relation, "left": a, "right": b} for a, b in pairs],
        )

    def sem_conflict(self, pairs: Sequence[tuple[Any, Any]]) -> list[bool]:
        return self.run("sem_conflict", [{"a": a, "b": b} for a, b in pairs])

    def sem_rank(self, query: Any, items: Sequence[Any]) -> list[int]:
        """Indices of ``items`` by descending score; ties keep engine order (stable)."""
        scores = self.run("sem_rank", [{"query": query, "item": x} for x in items])
        return sorted(range(len(items)), key=lambda i: (-float(scores[i]), i))

    def sem_topk(self, query: Any, items: Sequence[Any], k: int) -> list[int]:
        """Top-k of a bounded prefix; the caller bounds ``items`` (ADR-0007)."""
        if k <= 0 or not items:
            return []
        return self.sem_rank(query, items)[:k]

    def sem_summarize(self, items: Sequence[Any], budget_words: int) -> str:
        if not items:
            return ""
        out = self.run(
            "sem_summarize", [{"items": list(items), "budget_words": budget_words}]
        )
        return str(out[0])
