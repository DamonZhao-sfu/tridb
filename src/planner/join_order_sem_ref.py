"""Semantic-operator placement: executable spec for src/planner/join_order_sem.c.

Figure-1 plan P4 / ADR-0026. Three pure functions, bit-identical to the C port:

- ``place_sem``     — expensive-predicate (Hellerstein) rank ordering of predicate legs.
- ``sem_plan_cost`` — expected cost of applying predicates in a given order.
- ``sem_batch``     — FETCH size for a streaming semantic filter.

Bit-identity contract: every float is produced by the same IEEE-754 binary64 operations, in
the same order, as the C code (one operation per statement there; Python never fuses a*b+c):

    c_eff = cost * (1.0 - hit) + cache_cost * hit
    rank  = (1.0 - sel) / c_eff            (c_eff == 0 -> +inf: free predicates go first)
    acc = 0.0; surv = 1.0
    for j in ord: acc = acc + c_eff[j] * surv; surv = surv * sel[j]
    cost  = n_in * acc
    batch = clamp(ceil(k / max(sel_hat, 1e-6)), min_batch, max_batch); k <= 0 -> 0

Invalid input raises ``ValueError`` where the C port raises ``ERROR``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MAX_PREDS = 64
SEL_FLOOR = 1e-6


def _validate(
    sels: Sequence[float],
    costs: Sequence[float],
    hits: Sequence[float],
    cache_cost: float,
) -> None:
    if not (len(sels) == len(costs) == len(hits)):
        raise ValueError("sels, costs and hits must have equal lengths")
    if len(sels) > MAX_PREDS:
        raise ValueError("at most 64 predicates are supported")
    if any(v is None for v in (*sels, *costs, *hits)) or cache_cost is None:
        raise ValueError("inputs must not contain None")
    if not (cache_cost >= 0.0) or math.isinf(cache_cost):
        raise ValueError("cache_cost must be finite and >= 0")
    for sel, cost, hit in zip(sels, costs, hits):
        if not (0.0 <= sel <= 1.0):
            raise ValueError("every sel must be in [0, 1]")
        if not (cost >= 0.0) or math.isinf(cost):
            raise ValueError("every cost must be finite and >= 0")
        if not (0.0 <= hit <= 1.0):
            raise ValueError("every hit must be in [0, 1]")


def c_eff(cost: float, hit: float, cache_cost: float) -> float:
    """Effective per-row cost: cost*(1-hit) + cache_cost*hit (left to right, no fusion)."""
    miss = 1.0 - hit
    llm = cost * miss
    cached = cache_cost * hit
    return llm + cached


def rank(sel: float, ceff: float) -> float:
    """Hellerstein rank (1-sel)/c_eff; a free predicate (c_eff == 0) ranks +inf."""
    if ceff == 0.0:
        return math.inf
    return (1.0 - sel) / ceff


def place_sem(
    sels: Sequence[float],
    costs: Sequence[float],
    hits: Sequence[float],
    cache_cost: float = 0.0,
) -> list[int]:
    """0-based predicate order: descending rank, ties by ascending input index."""
    _validate(sels, costs, hits, cache_cost)
    ranks = [rank(s, c_eff(c, h, cache_cost)) for s, c, h in zip(sels, costs, hits)]
    return sorted(range(len(ranks)), key=lambda i: (-ranks[i], i))


def sem_plan_cost(
    sels: Sequence[float],
    costs: Sequence[float],
    hits: Sequence[float],
    ord_: Sequence[int],
    n_in: float,
    cache_cost: float = 0.0,
) -> float:
    """n_in * sum_pos c_eff[ord[pos]] * prod_{q<pos} sels[ord[q]]."""
    _validate(sels, costs, hits, cache_cost)
    if n_in is None or not (n_in >= 0.0) or math.isinf(n_in):
        raise ValueError("n_in must be finite and >= 0")
    if len(ord_) != len(sels) or sorted(ord_) != list(range(len(sels))):
        raise ValueError("ord must be a permutation of 0..n-1")
    acc = 0.0
    surv = 1.0
    for j in ord_:
        step = c_eff(costs[j], hits[j], cache_cost) * surv
        acc = acc + step
        surv = surv * sels[j]
    return n_in * acc


def sem_batch(k_remaining: int, sel_hat: float, min_batch: int, max_batch: int) -> int:
    """clamp(ceil(k / max(sel_hat, 1e-6)), min_batch, max_batch); k <= 0 -> 0."""
    if not (0.0 <= sel_hat <= 1.0):
        raise ValueError("sel_hat must be in [0, 1]")
    if min_batch < 1 or max_batch < min_batch:
        raise ValueError("require 1 <= min_batch <= max_batch")
    if k_remaining <= 0:
        return 0
    s = sel_hat if sel_hat > SEL_FLOOR else SEL_FLOOR
    v = math.ceil(float(k_remaining) / s)
    if v < min_batch:
        return min_batch
    if v > max_batch:
        return max_batch
    return v
