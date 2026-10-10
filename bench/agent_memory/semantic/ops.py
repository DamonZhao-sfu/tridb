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


@dataclass(frozen=True)
class OpSpec:
    name: str
    prompt_version: str
    system: str
    parse: Callable[[Any, str], Any]  # (parsed_json_or_None, raw_text) -> output


def _bool_field(*keys: str) -> Callable[[Any, str], Any]:
    def parse(obj: Any, raw: str) -> Any:
        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    value = obj[key]
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.strip().lower() in {"true", "yes", "1"}
                    return bool(value)
        low = (raw or "").strip().lower()
        return low.startswith("yes") or low.startswith("true")

    return parse


def _text_field(*keys: str, limit: int = 600) -> Callable[[Any, str], Any]:
    def parse(obj: Any, raw: str) -> Any:
        if isinstance(obj, dict):
            for key in keys:
                if isinstance(obj.get(key), str):
                    return obj[key][:limit]
        return (raw or "").strip()[:limit]

    return parse


def _score(obj: Any, raw: str) -> float:
    if isinstance(obj, dict):
        for key in ("score", "relevance"):
            if key in obj:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    break
        if "h" in obj:  # MockLLM default reply
            return float(int(str(obj["h"])[:2], 16) % 11)
    return 0.0


SPECS: dict[str, OpSpec] = {
    "sem_extract": OpSpec(
        "sem_extract",
        "v1",
        "You extract information. Follow the instruction and reply with JSON only: "
        '{"text": "<at most 60 words>"}.',
        _text_field("text", "lesson", "h"),
    ),
    "sem_filter": OpSpec(
        "sem_filter",
        "v1",
        "You decide whether an item satisfies a condition. Reply with JSON only: "
        '{"keep": true|false}.',
        _bool_field("keep", "verdict"),
    ),
    "sem_match": OpSpec(
        "sem_match",
        "v1",
        "You decide whether a relation holds between LEFT and RIGHT. Reply with JSON only: "
        '{"match": true|false}.',
        _bool_field("match", "verdict"),
    ),
    "sem_conflict": OpSpec(
        "sem_conflict",
        "v1",
        "You decide whether statement A and statement B contradict each other. Reply with "
        'JSON only: {"conflict": true|false}.',
        _bool_field("conflict", "verdict"),
    ),
    "sem_rank": OpSpec(
        "sem_rank",
        "v1",
        "You score how useful an item is for a query on a 0-10 scale. Reply with JSON only: "
        '{"score": <integer 0-10>}.',
        _score,
    ),
    "sem_summarize": OpSpec(
        "sem_summarize",
        "v1",
        "You summarize items under a word budget. Reply with JSON only: "
        '{"text": "<summary>"}.',
        _text_field("text", "summary", "h", limit=4000),
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
        if miss_order:
            prompts = [(spec.system, canonical(miss_input[h])) for h in miss_order]
            started = time.perf_counter()
            replies = self.llm.complete_many(prompts)
            self.meter.wall_seconds += time.perf_counter() - started
            self.meter.batches += 1
            new_rows = []
            for h, reply in zip(miss_order, replies, strict=True):
                output = spec.parse(parse_json_reply(reply.text), reply.text)
                found[h] = output
                new_rows.append(
                    (
                        h,
                        output,
                        reply.prompt_tokens,
                        reply.completion_tokens,
                        reply.seconds,
                    )
                )
                self.meter.prompt_tokens += reply.prompt_tokens
                self.meter.completion_tokens += reply.completion_tokens
                self.meter.llm_seconds += reply.seconds
            self.cache.put_many(op, self.model, spec.prompt_version, new_rows)
        calls = len(miss_order)
        self.meter.calls += calls
        self.meter.hits += len(inputs) - calls
        self.meter.by_op[op] = self.meter.by_op.get(op, 0) + len(inputs)
        return [found[h] for h in hashes]

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
        return self.sem_rank(query, items)[: max(0, k)]

    def sem_summarize(self, items: Sequence[Any], budget_words: int) -> str:
        if not items:
            return ""
        out = self.run(
            "sem_summarize", [{"items": list(items), "budget_words": budget_words}]
        )
        return str(out[0])
