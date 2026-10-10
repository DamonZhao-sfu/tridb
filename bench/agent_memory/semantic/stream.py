"""Streaming semantic filter over a server-side portal (ADR-0026 §4).

The engine operator (e.g. the ``tjs_vsgt`` CustomScan) stays a pure Open/Next/Close
stream. This consumer pulls bounded batches with ``FETCH n`` from a NO SCROLL cursor,
evaluates the semantic predicate on each batch through the cache, and closes the portal as
soon as ``k`` survivors exist. It never materializes the full candidate set (TR-1): the
buffer is at most ``max_batch`` rows, and the batch size adapts to the observed
selectivity via ``tridb_sem_batch``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from bench.agent_memory.semantic.ops import SemOps
from bench.agent_memory.semantic.planner import SemPlanner


@dataclass
class StreamReceipt:
    rows_pulled: int = 0
    fetches: int = 0
    sem_evaluated: int = 0
    survivors: int = 0
    closed_early: bool = False
    exhausted: bool = False
    batch_sizes: list[int] = field(default_factory=list)
    seconds: float = 0.0
    engine_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_pulled": self.rows_pulled,
            "fetches": self.fetches,
            "sem_evaluated": self.sem_evaluated,
            "survivors": self.survivors,
            "closed_early": self.closed_early,
            "exhausted": self.exhausted,
            "batch_sizes": list(self.batch_sizes),
            "seconds": round(self.seconds, 6),
            "engine_seconds": round(self.engine_seconds, 6),
        }


def sem_filter_stream(
    conn: Any,
    sql: str,
    params: Sequence[Any],
    ops: SemOps,
    predicate: str,
    item_of: Callable[[Any], Any],
    *,
    k: int,
    planner: SemPlanner,
    sel_hat: float = 0.5,
    min_batch: int = 4,
    max_batch: int = 64,
    pre_filter: Callable[[Any], bool] | None = None,
    prepare: Callable[[list[Any]], list[Any]] | None = None,
    map_row: Callable[[Any], Any] | None = None,
) -> tuple[list[Any], StreamReceipt]:
    """Return up to ``k`` rows of ``sql`` that pass ``predicate``, in stream order.

    ``pre_filter`` is an optional cheap predicate evaluated in Python before the semantic
    one (used only by baselines; the planned path pushes cheap predicates into the engine).
    ``sel_hat`` is the prior selectivity; it is updated from the observed pass rate.
    ``prepare`` runs once per fetched batch before the semantic predicate (strong-mode
    reads use it to resolve pending derivations of exactly the rows in the batch).
    """
    receipt = StreamReceipt()
    started = time.perf_counter()
    kept: list[Any] = []
    passed = 0
    evaluated = 0
    with conn.cursor(name="semstream_" + uuid4().hex) as cur:
        t0 = time.perf_counter()
        cur.execute(sql, params)
        receipt.engine_seconds += time.perf_counter() - t0
        while len(kept) < k:
            observed = (passed + 1) / (evaluated + 2) if evaluated else sel_hat
            batch = planner.batch_size(k - len(kept), observed, min_batch, max_batch)
            t0 = time.perf_counter()
            rows = cur.fetchmany(batch)
            receipt.engine_seconds += time.perf_counter() - t0
            receipt.fetches += 1
            receipt.batch_sizes.append(batch)
            receipt.rows_pulled += len(rows)
            short_fetch = len(rows) < batch
            if map_row is not None:
                rows = [map_row(r) for r in rows]
            if pre_filter is not None:
                rows = [r for r in rows if pre_filter(r)]
            if rows and prepare is not None:
                rows = prepare(rows)
            if rows:
                verdicts = ops.sem_filter(predicate, [item_of(r) for r in rows])
                evaluated += len(rows)
                for row, keep in zip(rows, verdicts, strict=True):
                    if keep:
                        passed += 1
                        if len(kept) < k:
                            kept.append(row)
            if short_fetch:
                receipt.exhausted = True
                break
        receipt.closed_early = not receipt.exhausted
    receipt.sem_evaluated = evaluated
    receipt.survivors = len(kept)
    receipt.seconds = time.perf_counter() - started
    return kept, receipt
