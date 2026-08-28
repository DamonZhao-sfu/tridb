"""Formal Graphiti adapter with a per-group FIFO prepared-write contract.

The LLM-free Add boundary writes one prepared fact through Graphiti's public
node/edge namespace APIs. The shared Track C runner remains open-loop; this
guard preserves event order within one ``group_id`` while allowing independent
LoCoMo conversations to progress concurrently.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from bench.agent_memory.table5_track_c.dataset import EventItem
from bench.agent_memory.table5_track_c.tracing import stage_span
from experiments.graphiti_track_c.adapter import GraphitiTrackCAdapter

from .hashing import graphiti_adapter_sha256


class AsyncGroupFifoGate:
    """A ticketed FIFO per group, with independent groups allowed in parallel."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._next_ticket: dict[str, int] = defaultdict(int)

    async def acquire(self, group_id: str) -> tuple[int, int]:
        queued_at_ns = time.perf_counter_ns()
        ticket = self._next_ticket[group_id]
        self._next_ticket[group_id] += 1
        lock = self._locks.setdefault(group_id, asyncio.Lock())
        await lock.acquire()
        acquired_at_ns = time.perf_counter_ns()
        return ticket, acquired_at_ns - queued_at_ns

    def release(self, group_id: str) -> None:
        self._locks[group_id].release()


class FormalGraphitiTrackCAdapter(GraphitiTrackCAdapter):
    """Bind formal Graphiti writes to its documented per-group FIFO semantics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._formal_add_gate = AsyncGroupFifoGate()
        self._formal_add_receipts: dict[str, dict[str, Any]] = {}
        self._formal_add_receipts_lock = threading.Lock()

    def _adapter_sha256(self) -> str:
        return graphiti_adapter_sha256(Path(__file__).resolve().parents[2])

    async def _add_async(self, item: EventItem) -> Any:
        with stage_span(
            "framework_other",
            "graphiti.group_fifo_wait",
            backend="graphiti-core-0.29.3 QueueService semantics",
            attributes={
                "group_id": item.sample_id,
                "boundary": "native per-group admission queue",
            },
        ):
            ticket, wait_ns = await self._formal_add_gate.acquire(item.sample_id)
        try:
            result = await super()._add_async(item)
        finally:
            self._formal_add_gate.release(item.sample_id)
        with self._formal_add_receipts_lock:
            self._formal_add_receipts[item.event_id] = {
                "write_group_id": item.sample_id,
                "write_group_fifo_ticket": ticket,
                "write_group_fifo_wait_ms": wait_ns / 1_000_000,
            }
        return result

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        receipt = super().add(item, visibility=visibility, progress=progress)
        with self._formal_add_receipts_lock:
            fifo_receipt = self._formal_add_receipts.pop(item.event_id, None)
        if fifo_receipt is None:
            raise RuntimeError("formal Graphiti add completed without a FIFO receipt")
        return {
            **receipt,
            "write_concurrency_policy": "per_group_fifo_cross_group_parallel",
            **fifo_receipt,
            "write_concurrency_provenance": (
                "benchmark-enforced per-group FIFO for prepared Graphiti "
                "node/edge namespace writes"
            ),
        }

    def stats(self) -> dict[str, Any]:
        return {
            **super().stats(),
            "formal_write_concurrency": {
                "policy": "per_group_fifo_cross_group_parallel",
                "queue_boundary": (
                    "prepared namespace writes only; inside measured service latency; "
                    "post-commit visibility probe is outside this lock"
                ),
                "same_group_max_active": 1,
                "cross_group_parallel": True,
                "provenance": (
                    "Track C prepared-memory ordering contract; this is not the "
                    "high-level Graphiti.add_episode API"
                ),
            },
        }
