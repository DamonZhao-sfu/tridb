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
    """Seed with programs of DIFFERENT lengths.

    MAP-Elites keys a cell on (complexity, diversity), and complexity is code length,
    so identical-length programs collapse into one cell and `add()` keeps only the
    winner: seeding six same-length programs leaves two in the database. Varying the
    length is what makes the island actually hold what the test says it holds.
    """
    for i in range(n):
        db.add(
            Program(
                id=f"own{i}",
                code=f"# own {i}\n" * (2 ** i),
                metrics={"combined_score": 0.1 * i},
            )
        )


def test_match_baseline_renders_the_same_count_as_arm_a() -> None:
    """Under `match_baseline`, arm A and arm B render the same NUMBER of inspirations.

    If memory were appended, the prompt would grow and any outcome difference could be
    explained by context length instead of memory quality. This is the conservative
    policy; see `test_fixed_policy_injects_even_when_the_island_is_empty` for the cost.
    """
    # One island so every seeded program lands in it: with the default 5, arm A itself
    # renders nothing and the comparison would be vacuous (the guard below catches it).
    retriever = FakeRetriever(count=3)
    db = _db(retriever, islands=1, policy="match_baseline")
    _seed(db, n=10)
    _, inspirations = db.sample_from_island(0, num_inspirations=5)

    # Same seed, same island, same count -- arm A is the reference for how many
    # inspirations the prompt renders, because the base class returns fewer than
    # `num_inspirations` whenever the island is small.
    baseline = _db(NullRetriever(), islands=1)
    _seed(baseline, n=10)
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
    db = _db(FakeRetriever(count=5), islands=1, max_injected=2, policy="match_baseline")
    _seed(db, n=10)
    _, inspirations = db.sample_from_island(0, num_inspirations=5)

    baseline = _db(NullRetriever(), islands=1)
    _seed(baseline, n=10)
    _, own_only = baseline.sample_from_island(0, num_inspirations=5)
    assert len(inspirations) == len(own_only)
    assert sum(1 for p in inspirations if is_external(p.id)) == min(2, len(own_only))


def test_null_retriever_is_stock_behaviour() -> None:
    db = _db(NullRetriever())
    _seed(db)
    _, inspirations = db.sample_from_island(0, num_inspirations=5)
    assert not any(is_external(p.id) for p in inspirations)
    assert db.external_ids == set()


def test_fixed_policy_injects_even_when_the_island_is_empty() -> None:
    """The measured failure of `match_baseline`, pinned as a test.

    With 5 islands and a nearly empty database the base class returns ZERO own
    inspirations, so `match_baseline` injects nothing and arm B is byte-identical to
    arm A -- observed for three straight iterations in the first live smoke run.
    `fixed` is what makes the treatment exist during those iterations.
    """
    strict = _db(FakeRetriever(count=3), islands=5, policy="match_baseline")
    strict.add(Program(id="only", code="# seed\n", metrics={"combined_score": 0.1}))
    _, none_rendered = strict.sample_from_island(0, num_inspirations=5)
    assert not [p for p in none_rendered if is_external(p.id)]

    loose = _db(FakeRetriever(count=3), islands=5, policy="fixed")
    loose.add(Program(id="only", code="# seed\n", metrics={"combined_score": 0.1}))
    _, rendered = loose.sample_from_island(0, num_inspirations=5)
    assert len([p for p in rendered if is_external(p.id)]) == 3


def test_trace_records_what_was_rendered_not_what_was_retrieved() -> None:
    """`injected_ids` must mean "reached the prompt".

    The first implementation recorded the retrieved count, which reported
    `injected=5, rendered=0` while nothing at all was being injected.
    """
    db = _db(FakeRetriever(count=3), islands=5, policy="match_baseline")
    db.add(Program(id="only", code="# seed\n", metrics={"combined_score": 0.1}))
    db.sample_from_island(0, num_inspirations=5)

    row = db.injection_trace[-1]
    assert row["retrieved"] == 3
    assert row["injected_ids"] == []
    assert row["rendered"] == len(row["own_ids"])


def test_unknown_policy_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown injection policy"):
        _db(NullRetriever(), policy="whatever")


def test_fingerprint_is_always_a_substring_of_the_code() -> None:
    """The standing proof that the gate cannot raise a false alarm.

    The first implementation filtered uninteresting lines and joined the rest, which
    welded `def f():` onto a docstring body with the quote line removed -- a string
    that appears nowhere in the original. The injection gate then reported 100% of
    programs absent from a prompt that visibly contained every one of them.
    """
    from bench.agent_memory.gem_oe.memory_database import code_fingerprint

    samples = [
        '# banner\n"""doc"""\nimport numpy as np\n\n\ndef f():\n    """\n    body\n    """\n    return 1\n',
        "import os\nfrom pathlib import Path\n\n\nclass C:\n    x = 1\n",
        "x = 1\ny = 2\nz = 3\n",
        "",
        "\n\n\n",
        "# only comments\n# and more\n",
    ]
    for code in samples:
        fingerprint = code_fingerprint(code)
        assert fingerprint in code, f"not a substring of {code!r}: {fingerprint!r}"


def test_fingerprint_skips_the_shared_preamble() -> None:
    """Anchoring on `import numpy as np` would match essentially any program here."""
    from bench.agent_memory.gem_oe.memory_database import code_fingerprint

    fingerprint = code_fingerprint(
        '# EVOLVE-BLOCK-START\n"""banner"""\nimport numpy as np\n\n\ndef pack():\n    return 26\n'
    )
    assert fingerprint.startswith("def pack():")
