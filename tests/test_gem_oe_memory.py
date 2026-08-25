"""Unit tests for the OpenEvolve memory injection.

These cover the invariants that, if broken, silently invalidate every agent number:
replacement (not addition) keeps the prompt budget equal across arms; an injected
program can never become a parent; and a retrieval failure kills the cell instead of
degrading arm B into arm A.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

openevolve = pytest.importorskip(
    "openevolve", reason="openevolve 0.3.2 lives in .venv-e0; run these there"
)

from openevolve.config import DatabaseConfig  # noqa: E402
from openevolve.database import Program  # noqa: E402

from bench.agent_memory.gem_oe.memory_database import (  # noqa: E402
    EXTERNAL_FLAG,
    EXTERNAL_PREFIX,
    GemMemoryDatabase,
    NullRetriever,
    RetrievedProgram,
    is_external,
)


class FakeRetriever:
    name = "fake"

    def __init__(self, count: int = 3, fail: bool = False) -> None:
        self.count = count
        self.fail = fail
        self.calls: list[dict] = []

    def retrieve(self, *, task_uid, parent, k, iteration):
        self.calls.append({"task_uid": task_uid, "k": k, "iteration": iteration})
        if self.fail:
            raise ConnectionError("memory backend unreachable")
        return [
            RetrievedProgram(
                uid=f"hist{i}",
                code=f"# historical program {i}\n",
                metrics={"combined_score": 0.5 + i / 100},
                provenance={"session_uid": f"s{i}"},
            )
            for i in range(min(self.count, k))
        ]


def _db(retriever, *, islands: int = 2, **kwargs) -> GemMemoryDatabase:
    config = DatabaseConfig()
    config.num_islands = islands
    config.in_memory = True
    config.random_seed = 42
    return GemMemoryDatabase(
        config, retriever=retriever, task_uid="math:circle_packing", **kwargs
    )


def _seed(db: GemMemoryDatabase, n: int = 6) -> None:
    for i in range(n):
        db.add(
            Program(
                id=f"own{i}",
                code=f"# own {i}\n",
                metrics={"combined_score": 0.1 * i},
            )
        )


def test_injection_replaces_rather_than_adds() -> None:
    """Arm A and arm B must render the same NUMBER of inspirations.

    If memory were appended, the prompt would grow and any outcome difference could be
    explained by context length instead of memory quality.
    """
    retriever = FakeRetriever(count=3)
    db = _db(retriever)
    _seed(db)
    _, inspirations = db.sample_from_island(0, num_inspirations=5)

    # Same seed, same island, same count -- arm A is the reference for how many
    # inspirations the prompt renders, because the base class returns fewer than
    # `num_inspirations` whenever the island is small.
    baseline = _db(NullRetriever())
    _seed(baseline)
    _, own_only = baseline.sample_from_island(0, num_inspirations=5)
    assert len(inspirations) == len(own_only)

    injected = [p for p in inspirations if is_external(p.id)]
    assert injected, "nothing was injected, so the count assertion proves nothing"
    assert len(injected) == min(3, len(own_only))
    assert all(p.metadata[EXTERNAL_FLAG] for p in injected)


def test_injected_program_is_never_a_parent() -> None:
    """Continuing evolution from another session's code is seeding, not memory."""
    db = _db(FakeRetriever(count=2))
    _seed(db)
    for _ in range(30):
        parent, _ = db.sample_from_island(0, num_inspirations=4)
        assert not is_external(parent.id)


def test_injected_programs_stay_out_of_islands_and_archive() -> None:
    """They must be resolvable from the snapshot and invisible everywhere else."""
    db = _db(FakeRetriever(count=3))
    _seed(db)
    db.sample_from_island(0, num_inspirations=5)

    external = db.external_ids
    assert external, "nothing was injected, so the assertion proves nothing"
    assert all(pid in db.programs for pid in external), "snapshot would drop them"
    for island in db.islands:
        assert not (set(island) & external)
    assert not (set(db.archive) & external)
    assert db.best_program_id not in external
    assert not (set(db.own_programs()) & external)


def test_random_parent_fallback_excludes_externals() -> None:
    """`_sample_random_parent` picks from all of `self.programs`; narrow it."""
    db = _db(FakeRetriever(count=2))
    _seed(db, n=2)
    db.sample_from_island(0, num_inspirations=4)
    for _ in range(50):
        assert not is_external(db._sample_random_parent().id)


def test_retrieval_failure_kills_the_cell() -> None:
    """A silent degrade to arm A is the worst outcome; it must raise instead."""
    db = _db(FakeRetriever(fail=True))
    _seed(db)
    with pytest.raises(ConnectionError):
        db.sample_from_island(0, num_inspirations=5)


def test_trace_records_every_injected_id() -> None:
    """The prompt assertion (Phase 0 gate 6) reads this trace; it must be complete."""
    db = _db(FakeRetriever(count=2))
    _seed(db)
    for _ in range(4):
        db.sample_from_island(0, num_inspirations=3)

    assert len(db.injection_trace) == 4
    for row in db.injection_trace:
        assert row["arm"] == "fake"
        assert len(row["injected_ids"]) == 2
        assert all(pid.startswith(EXTERNAL_PREFIX) for pid in row["injected_ids"])
        assert row["error"] is None
        assert not is_external(row["parent_id"])


def test_max_injected_caps_the_replacement() -> None:
    """The injection-rate knob must bound how much of the budget memory takes."""
    db = _db(FakeRetriever(count=5), max_injected=2)
    _seed(db)
    _, inspirations = db.sample_from_island(0, num_inspirations=5)

    baseline = _db(NullRetriever())
    _seed(baseline)
    _, own_only = baseline.sample_from_island(0, num_inspirations=5)
    assert len(inspirations) == len(own_only)
    assert sum(1 for p in inspirations if is_external(p.id)) == min(2, len(own_only))


def test_null_retriever_is_stock_behaviour() -> None:
    db = _db(NullRetriever())
    _seed(db)
    _, inspirations = db.sample_from_island(0, num_inspirations=5)
    assert not any(is_external(p.id) for p in inspirations)
    assert db.external_ids == set()
