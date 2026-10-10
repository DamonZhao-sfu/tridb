"""Planner glue for semantic predicates (plan P4, Python side).

The ordering decision itself is the additive engine function ``tridb_place_sem``
(src/planner/join_order_sem.c, ADR-0026). When the extension function is installed in the
connected database the planner calls it; otherwise it falls back to the bit-identical
Python spec ``src/planner/join_order_sem_ref.py``. Either way the decision and its source
go into the receipt.

A predicate leg is (name, selectivity, per-tuple cost, cache hit rate). Cheap legs that
the engine can push down (relational / graph) have cost ~1e-6 s; a semantic leg costs one
LLM evaluation per tuple, discounted by the cache hit rate.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REF_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "planner" / "join_order_sem_ref.py"
)


def _load_ref() -> Any:
    name = "join_order_sem_ref"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REF_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load planner reference at {_REF_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class PredicateLeg:
    name: str
    selectivity: float
    cost: float  # seconds per evaluated tuple
    hit_rate: float = 0.0
    pushable: bool = False  # can run inside the engine operator (relational / graph)


@dataclass(frozen=True)
class Placement:
    order: tuple[str, ...]
    expected_cost: float
    source: str  # 'engine' (tridb_place_sem) | 'python-ref'


class SemPlanner:
    def __init__(self, conn: Any | None = None, *, cache_cost: float = 0.0) -> None:
        self.conn = conn
        self.cache_cost = cache_cost
        self._engine: bool | None = None

    def engine_available(self) -> bool:
        if self._engine is None:
            self._engine = False
            if self.conn is not None:
                row = self.conn.execute(
                    "SELECT to_regprocedure("
                    "'tridb_place_sem(double precision[],double precision[],"
                    "double precision[],double precision)') IS NOT NULL"
                ).fetchone()
                self._engine = bool(row and row[0])
        return self._engine

    def place(self, legs: Sequence[PredicateLeg], n_in: float) -> Placement:
        sels = [float(leg.selectivity) for leg in legs]
        costs = [float(leg.cost) for leg in legs]
        hits = [float(leg.hit_rate) for leg in legs]
        if self.engine_available():
            order = self.conn.execute(
                "SELECT tridb_place_sem(%s::float8[], %s::float8[], %s::float8[], %s)",
                (sels, costs, hits, self.cache_cost),
            ).fetchone()[0]
            cost = self.conn.execute(
                "SELECT tridb_sem_plan_cost(%s::float8[], %s::float8[], %s::float8[],"
                " %s::int[], %s, %s)",
                (sels, costs, hits, list(order), float(n_in), self.cache_cost),
            ).fetchone()[0]
            source = "engine"
        else:
            ref = _load_ref()
            order = ref.place_sem(sels, costs, hits, self.cache_cost)
            cost = ref.sem_plan_cost(
                sels, costs, hits, list(order), float(n_in), self.cache_cost
            )
            source = "python-ref"
        return Placement(tuple(legs[i].name for i in order), float(cost), source)

    def cost_of(
        self, legs: Sequence[PredicateLeg], order_names: Sequence[str], n_in: float
    ) -> float:
        index = {leg.name: i for i, leg in enumerate(legs)}
        ref = _load_ref()
        return float(
            ref.sem_plan_cost(
                [leg.selectivity for leg in legs],
                [leg.cost for leg in legs],
                [leg.hit_rate for leg in legs],
                [index[n] for n in order_names],
                float(n_in),
                self.cache_cost,
            )
        )

    def batch_size(
        self, k_remaining: int, sel_hat: float, min_batch: int, max_batch: int
    ) -> int:
        if self.engine_available():
            return int(
                self.conn.execute(
                    "SELECT tridb_sem_batch(%s, %s, %s, %s)",
                    (int(k_remaining), float(sel_hat), int(min_batch), int(max_batch)),
                ).fetchone()[0]
            )
        return int(_load_ref().sem_batch(k_remaining, sel_hat, min_batch, max_batch))
