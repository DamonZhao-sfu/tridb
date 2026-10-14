"""Bounded context packing with the agent's actual tokenizer and early Close.

Only the final context (at most top_k records) is retained. The packer never
consumes the unneeded tail to estimate hypothetical truncation statistics.
Prompt counts describe the supplied renderer; adapters must include their
actual chat template in that renderer when enforcing a model context window.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode(
        self, text: str, *, add_special_tokens: bool = False
    ) -> Sequence[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class PackingEvent:
    record_id: str
    original_tokens: int
    retained_tokens: int
    disposition: str


@dataclass
class PackedContext:
    text: str = ""
    memory_tokens: int = 0
    full_prompt_tokens: int | None = None
    retained_ids: list[str] = field(default_factory=list)
    events: list[PackingEvent] = field(default_factory=list)
    records_seen: int = 0
    truncated_records: int = 0
    skipped_records: int = 0
    duplicate_records: int = 0
    stop_reason: str = "source_exhausted"
    tokenizer_id: str = ""

    @property
    def prompt_was_truncated(self) -> bool:
        return self.truncated_records > 0


def format_memory(row: Mapping[str, Any]) -> str:
    """Unabridged payload; truncation is tokenizer-based, never code_chars."""
    return (
        f"[memory id={row['id']}]\n"
        f"fitness={row.get('fitness')} parent_fitness={row.get('parent_fitness')}\n"
        f"changes:\n{row.get('changes') or ''}\n"
        f"code:\n{row.get('code') or ''}"
    )


def gain_key(row: Mapping[str, Any]) -> tuple[float, int]:
    """Sort key: larger fitness - parent_fitness first, then smaller id."""
    fitness, parent = row.get("fitness"), row.get("parent_fitness")
    gain = (
        float(fitness) - float(parent)
        if fitness is not None and parent is not None
        else float("-inf")
    )
    return (-gain, int(row["id"]))


class GainRankedPool:
    """ordering="gain": drain a bounded cursor, then yield rows ranked by gain.

    The selected cursor itself stops at ``candidate_visit_cap`` fetched rows and
    closes (or is exhausted), so the pool is bounded and the SQL stays a plain
    streaming query; ranking is client-side over at most that many rows.
    Duplicate attempt ids (union hops / all-depth frontiers) keep one row.
    Pool statistics are written to the cursor's ExecutionReceipt.
    """

    def __init__(self, cursor: Any):
        self.cursor = cursor

    def __iter__(self):
        rows = list(self.cursor)
        ranked: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            ranked.setdefault(str(row["id"]), row)
        receipt = self.cursor.receipt
        receipt.ranked_by = "gain"
        receipt.pool_rows = len(rows)
        receipt.pool_distinct = len(ranked)
        return iter(sorted(ranked.values(), key=gain_key))

    def close(self, reason: str = "caller_closed") -> None:
        self.cursor.close(reason)


def pack_stream(
    cursor: Any,
    tokenizer: Tokenizer,
    *,
    top_k: int,
    memory_token_cap: int = 8192,
    tokenizer_id: str,
    model_context_window: int | None = None,
    render_prompt: Callable[[str], str] | None = None,
    max_output_tokens: int = 0,
    record_formatter: Callable[[Mapping[str, Any]], str] = format_memory,
) -> PackedContext:
    """Consume an opened pull cursor and always Close on stop or exception.

    ``render_prompt`` includes system/question/current trajectory and the chat
    template. The output reserve is additional to its exact encoded count.
    Token-prefix search is conservative (token counts need not be monotonic
    under concatenation); the final assembled memory and prompt are rechecked.
    No estimate based on characters is ever used as a token count.
    """
    result = PackedContext(tokenizer_id=tokenizer_id)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def prompt_count(text: str) -> int | None:
        return count(render_prompt(text)) if render_prompt is not None else None

    def fits(text: str) -> bool:
        if count(text) > memory_token_cap:
            return False
        return model_context_window is None or (
            prompt_count(text) + max_output_tokens <= model_context_window
        )

    try:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if type(memory_token_cap) is not int or not 0 <= memory_token_cap <= 8192:
            raise ValueError("v1 memory_token_cap must be between 0 and 8192")
        if not tokenizer_id:
            raise ValueError("tokenizer_id must identify the actual agent tokenizer")
        if type(max_output_tokens) is not int or max_output_tokens < 0:
            raise ValueError("max_output_tokens must be nonnegative")
        if model_context_window is not None and (
            type(model_context_window) is not int
            or model_context_window < 0
            or render_prompt is None
        ):
            raise ValueError("a context window requires an exact render_prompt")
        if not fits(""):
            result.stop_reason = "base_prompt_exceeds_window"
            raise ValueError(
                "base prompt plus output reserve exceeds model context window"
            )
        if top_k == 0 or memory_token_cap == 0:
            result.stop_reason = "top_k" if top_k == 0 else "memory_token_cap"
            return result

        retained: set[str] = set()
        for row in cursor:
            result.records_seen += 1
            record_id = str(row["id"])
            if record_id in retained:
                result.duplicate_records += 1
                continue
            record = record_formatter(row)
            token_ids = tokenizer.encode(record, add_special_tokens=False)
            original_count = len(token_ids)
            separator = "\n\n" if result.text else ""

            def append(piece: str, separator: str = separator) -> str:
                return result.text + separator + piece

            selected = record
            disposition = "accepted"
            if not fits(append(selected)):
                # Prefixes may yield fewer than the maximum possible tokens;
                # every returned prefix nevertheless passes the exact checks.
                low, high, best = 1, len(token_ids) - 1, ""
                while low <= high:
                    mid = (low + high) // 2
                    prefix = tokenizer.decode(token_ids[:mid])
                    candidate = prefix + "\n[truncated]"
                    if prefix and fits(append(candidate)):
                        best, low = candidate, mid + 1
                    else:
                        high = mid - 1
                if not best:
                    result.skipped_records += 1
                    result.events.append(
                        PackingEvent(record_id, original_count, 0, "skipped_budget")
                    )
                    continue
                selected, disposition = best, "truncated"
                result.truncated_records += 1

            result.text = append(selected)
            # This invariant does not rely on a sum of separately tokenized rows.
            if not fits(result.text):
                raise RuntimeError("tokenizer/renderer changed during context packing")
            result.memory_tokens = count(result.text)
            result.full_prompt_tokens = prompt_count(result.text)
            result.retained_ids.append(record_id)
            retained.add(record_id)
            result.events.append(
                PackingEvent(record_id, original_count, count(selected), disposition)
            )
            if len(result.retained_ids) >= top_k:
                result.stop_reason = "top_k"
                break
            if result.memory_tokens >= memory_token_cap:
                result.stop_reason = "memory_token_cap"
                break
            if model_context_window is not None and (
                result.full_prompt_tokens + max_output_tokens >= model_context_window
            ):
                result.stop_reason = "model_context_window"
                break
        return result
    except BaseException:
        result.stop_reason = "packing_error"
        raise
    finally:
        try:
            result.memory_tokens = count(result.text)
            result.full_prompt_tokens = prompt_count(result.text)
        finally:
            cursor.close(result.stop_reason)
