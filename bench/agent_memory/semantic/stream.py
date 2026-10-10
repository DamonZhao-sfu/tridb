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
    stop_reason: str = ""
    universe: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_pulled": self.rows_pulled,
            "fetches": self.fetches,
            "sem_evaluated": self.sem_evaluated,
            "survivors": self.survivors,
            "closed_early": self.closed_early,
            "exhausted": self.exhausted,
            "batch_sizes": list(self.batch_sizes),
            "stop_reason": self.stop_reason,
            "universe": self.universe,
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
    post_filter: Callable[[Any], bool] | None = None,
    universe_limit: int | None = None,
    early_close: bool = True,
    prepare: Callable[[list[Any]], list[Any]] | None = None,
    map_row: Callable[[Any], Any] | None = None,
) -> tuple[list[Any], StreamReceipt]:
    """Streaming theta_sem over a server-side portal.

    Logical result (identical for every physical order): let U be the first
    ``universe_limit`` stream rows that satisfy the cheap predicate theta_R (every row
    when theta_R is pushed into the engine); return the first ``k`` rows of U, in stream
    order, that satisfy ``predicate``.

    Physical order:
      * ``pre_filter``  — theta_R evaluated before the semantic predicate (planner order);
      * ``post_filter`` — theta_R after it (semantic-first: every raw row is judged);
      * neither         — theta_R was pushed into the engine operator.
    ``early_close`` closes the portal at k survivors; otherwise the scan runs to the
    universe boundary (the "materialized" baselines) — still in bounded batches, never a
    full fetch. ``prepare`` runs per batch before the semantic predicate (strong reads
    resolve pending derivations of exactly those rows). ``sel_hat`` seeds the adaptive
    batch size, which follows the observed pass rate via ``tridb_sem_batch``.
    """
    if pre_filter is not None and post_filter is not None:
        raise ValueError("theta_R is either before (pre_filter) or after (post_filter)")
    receipt = StreamReceipt()
    started = time.perf_counter()
    kept: list[Any] = []
    passed = 0
    evaluated = 0
    universe = 0
    cap = universe_limit if universe_limit is not None else 1 << 62
    stop = ""
    with conn.cursor(name="semstream_" + uuid4().hex) as cur:
        t0 = time.perf_counter()
        cur.execute(sql, params)
        receipt.engine_seconds += time.perf_counter() - t0
        while True:
            if early_close and len(kept) >= k:
                stop = "k"
                break
            if universe >= cap:
                stop = "universe"
                break
            if early_close:
                observed = (passed + 1) / (evaluated + 2) if evaluated else sel_hat
                need = k - len(kept)
                batch = planner.batch_size(need, observed, min_batch, max_batch)
            else:
                batch = max_batch
            t0 = time.perf_counter()
            rows = cur.fetchmany(batch)
            receipt.engine_seconds += time.perf_counter() - t0
            receipt.fetches += 1
            receipt.batch_sizes.append(batch)
            receipt.rows_pulled += len(rows)
            short_fetch = len(rows) < batch
            if map_row is not None:
                rows = [map_row(r) for r in rows]
            if post_filter is None:
                # theta_R already applied (engine or pre_filter): rows are universe members.
                if pre_filter is not None:
                    rows = [r for r in rows if pre_filter(r)]
                rows = rows[: cap - universe]
                if rows and prepare is not None:
                    rows = prepare(rows)
                if rows:
                    verdicts = ops.sem_filter(predicate, [item_of(r) for r in rows])
                    evaluated += len(rows)
                    universe += len(rows)
                    for row, keep in zip(rows, verdicts, strict=True):
                        if keep:
                            passed += 1
                            kept.append(row)
            elif rows:
                # Semantic-first: judge every raw row, then apply theta_R in order.
                if prepare is not None:
                    rows = prepare(rows)
                verdicts = ops.sem_filter(predicate, [item_of(r) for r in rows])
                evaluated += len(rows)
                for row, keep in zip(rows, verdicts, strict=True):
                    if not post_filter(row):
                        continue
                    if universe >= cap:
                        break
                    universe += 1
                    if keep:
                        passed += 1
                        kept.append(row)
            if short_fetch:
                receipt.exhausted = True
                stop = stop or "exhausted"
                break
        receipt.closed_early = stop in ("k", "universe") and not receipt.exhausted
    receipt.stop_reason = stop
    receipt.universe = universe
    receipt.sem_evaluated = evaluated
    kept = kept[:k]
    receipt.survivors = len(kept)
    receipt.seconds = time.perf_counter() - started
    return kept, receipt
