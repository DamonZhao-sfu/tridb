"""Contract tests for the retrieval-leg ablation arms (A2 / B1 / B2).

No engine, no Postgres: these pin the things that make the ablation *mean* something —
the guard floor that every arm keeps, that the beat-parent bound is the ONLY thing
`require_beat_parent` moves, and that the five arms are exactly
`native_fused_plan x require_beat_parent` with no off-operator path.

Plan: docs/evotrace_leg_ablation_ale3_math4_p050_iter20_plan_v0.2.0.md
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.agent_memory.gem_oe.retrievers import (  # noqa: E402
    ABLATION_ARMS,
    NATIVE_FUSED_PLANS,
    node_predicate,
)


@dataclass
class FakeProgram:
    code: str = "print(1)\n"
    metrics: dict[str, float] = field(default_factory=lambda: {"combined_score": 10.0})
    metadata: dict[str, Any] = field(
        default_factory=lambda: {"session_uid": "sess:target"}
    )


class FakeConn:
    """Records every statement; returns no duplicate nodes and no rows."""

    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> "FakeConn":
        self.statements.append((sql, params))
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.rows[0] if self.rows else None


class FakeStore:
    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn

    def edge_type_id(self, name: str) -> int:
        assert name == "eg_lineage"
        return 5


# -- the guard floor ---------------------------------------------------------


def _predicate(require_beat_parent: bool, *, split: str = "same_task"):
    return node_predicate(
        FakeStore(FakeConn()),
        task_uid="ale:ahc008",
        parent=FakeProgram(),
        split=split,
        seed="node",
        require_beat_parent=require_beat_parent,
    )


def test_hard_guards_survive_the_filter_ablation() -> None:
    """Ablating the relational leg must not weaken scope, validity or the split.

    If it did, the ablated arms could see the run's own session and their quality
    numbers would be leakage rather than a leg effect.
    """
    off = _predicate(False)
    assert off.kind == "node"
    assert off.require_valid is True
    assert off.exclude_sessions == frozenset({"sess:target"})
    assert off.include_tasks == frozenset({"ale:ahc008"})
    assert off.require_finite_fitness is True


def test_beat_parent_is_the_only_thing_the_filter_ablation_moves() -> None:
    on, off = _predicate(True), _predicate(False)
    assert on.fitness_gt == 10.0 and on.require_fitness is True
    assert off.fitness_gt is None and off.require_fitness is False
    for shared in ("kind", "require_valid", "exclude_sessions", "include_tasks"):
        assert getattr(on, shared) == getattr(off, shared)


def test_filter_ablation_drops_only_the_beat_parent_clause_from_the_sql() -> None:
    assert "fitness > 10.0" in _predicate(True).to_sql()
    off_sql = _predicate(False).to_sql()
    assert "fitness > 10.0" not in off_sql
    # The guards are still rendered -- including the finite-fitness bound, which is a
    # guard (81 of 10,672 nodes have no orderable reward), not the relational leg.
    assert "is_valid" in off_sql and "task_uid IN" in off_sql
    assert "fitness IS NOT NULL" in off_sql
    assert "fitness > '-Infinity'::double precision" in off_sql


def test_cross_task_split_still_excludes_the_task_when_filter_is_ablated() -> None:
    p = _predicate(False, split="cross_task")
    assert p.exclude_tasks == frozenset({"ale:ahc008"}) and p.include_tasks is None


# -- arm identity ------------------------------------------------------------


def test_every_arm_is_one_tjs_open_plan_times_one_predicate_setting() -> None:
    """No arm may live outside the operator.

    An earlier draft walked the graph with a recursive CTE over `gem_eg_edge`; that both
    violates CLAUDE.md golden rule 3 (topology is an access method, never a join table)
    and measures Postgres's CTE machinery instead of the graph AM. The operator's
    filter-first branch already generates candidates with
    `graph_store.gph_traverse_bounded`, so the graph-only arm is a predicate change, not
    a new code path.
    """
    assert ABLATION_ARMS == {
        ("graph_expand", True): "A0_vgr_full",
        ("source_anchored", True): "A1_gr_anchored",
        ("source_anchored", False): "B1_g_only",
        ("vector_only", False): "A2_v_only",
        ("vector_only", True): "B2_r_only",
    }
    assert {plan for plan, _ in ABLATION_ARMS} == NATIVE_FUSED_PLANS


def test_the_three_new_arms_reuse_the_two_shipped_plans() -> None:
    new = {"A2_v_only", "B1_g_only", "B2_r_only"}
    plans = {plan for plan, _ in ABLATION_ARMS if ABLATION_ARMS[(plan, _)] in new}
    assert plans == {"vector_only", "source_anchored"}


def test_arms_are_distinguished_by_the_predicate_not_only_the_plan() -> None:
    """A2 and B2 share a plan; B1 and A1 share a plan. The bound is what differs."""
    assert ABLATION_ARMS[("vector_only", False)] != ABLATION_ARMS[("vector_only", True)]
    assert (
        ABLATION_ARMS[("source_anchored", False)]
        != ABLATION_ARMS[("source_anchored", True)]
    )


# -- CLI wiring --------------------------------------------------------------


def _args(**kw: Any):
    from argparse import Namespace

    base = dict(native_fused_plan=None, require_beat_parent=None)
    base.update(kw)
    return Namespace(**base)


def test_beat_parent_defaults_on_and_an_explicit_flag_wins() -> None:
    from bench.agent_memory.gem_oe.run_arm import _require_beat_parent

    assert _require_beat_parent(_args()) is True
    assert _require_beat_parent(_args(native_fused_plan="vector_only")) is True
    assert (
        _require_beat_parent(
            _args(native_fused_plan="vector_only", require_beat_parent="false")
        )
        is False
    )
