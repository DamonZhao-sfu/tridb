"""PG-resident LLM result cache and the semantic-operator cost meter.

Cache rows live in ``gem_sem_cache`` in the same database as the stores (ADR-0026).
The cache key is ``(op, model, prompt_version, sha256(canonical input))``. A hit is a
table read, so a replay with the LLM disabled reproduces the same outputs byte for byte.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def ensure_schema(conn: Any) -> None:
    """Apply the cache + pending DDL idempotently on ``conn`` (caller commits)."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def input_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass
class SemMeter:
    """Counts what the semantic layer spent. n_sem = hits + misses (logical evaluations)."""

    calls: int = 0  # LLM requests actually sent (= cache misses)
    hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_seconds: float = 0.0  # sum of per-call service times
    wall_seconds: float = 0.0  # wall time spent inside LLM batches
    batches: int = 0
    by_op: dict[str, int] = field(default_factory=dict)

    @property
    def n_sem(self) -> int:
        return self.calls + self.hits

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_sem": self.n_sem,
            "llm_calls": self.calls,
            "cache_hits": self.hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_seconds": round(self.llm_seconds, 6),
            "llm_wall_seconds": round(self.wall_seconds, 6),
            "llm_batches": self.batches,
            "by_op": dict(self.by_op),
        }

    def diff(self, before: dict[str, Any]) -> dict[str, Any]:
        now = self.snapshot()
        out: dict[str, Any] = {}
        for key, value in now.items():
            if key == "by_op":
                prev = before.get("by_op", {})
                out[key] = {
                    k: v - prev.get(k, 0)
                    for k, v in value.items()
                    if v - prev.get(k, 0)
                }
            else:
                out[key] = value - before.get(key, 0)
        return out


class SemCache:
    """Batch get/put over ``gem_sem_cache``. Uses the caller's connection and transaction."""

    def __init__(self, conn: Any, *, enabled: bool = True) -> None:
        self.conn = conn
        self.enabled = enabled

    def get_many(
        self, op: str, model: str, prompt_version: str, hashes: Sequence[str]
    ) -> dict[str, Any]:
        if not self.enabled or not hashes:
            return {}
        rows = self.conn.execute(
            "SELECT input_hash, output FROM gem_sem_cache"
            " WHERE op = %s AND model = %s AND prompt_version = %s"
            "   AND input_hash = ANY(%s)",
            (op, model, prompt_version, list(set(hashes))),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def put_many(
        self,
        op: str,
        model: str,
        prompt_version: str,
        rows: Iterable[tuple[str, Any, int, int, float]],
    ) -> int:
        """rows: (input_hash, output, prompt_tokens, completion_tokens, seconds)."""
        if not self.enabled:
            return 0
        from psycopg.types.json import Jsonb

        payload = [
            (op, model, prompt_version, h, Jsonb(out), pt, ct, sec)
            for h, out, pt, ct, sec in rows
        ]
        if not payload:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO gem_sem_cache"
                " (op, model, prompt_version, input_hash, output,"
                "  prompt_tokens, completion_tokens, llm_seconds)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (op, model, prompt_version, input_hash) DO NOTHING",
                payload,
            )
        return len(payload)

    def hit_rate(
        self, op: str, model: str, prompt_version: str, hashes: Sequence[str]
    ) -> float:
        if not hashes:
            return 0.0
        found = self.get_many(op, model, prompt_version, hashes)
        return sum(1 for h in hashes if h in found) / len(hashes)
