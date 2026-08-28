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
                changes_description=f"historical change {i}",
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
                code=f"# own {i}\n" * (2**i),
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
    #
    # The global RNG is reseeded before each sample because `ProgramDatabase`'s parent
    # selection draws from `random`, module-wide. Without this the two arms read the
    # global sequence at different offsets, pick different parents, and end up with
    # different own-inspiration counts -- so the test passed or failed depending on
    # which tests ran before it.
    #
    # seed 1 specifically: MAP-Elites keys a cell on (complexity, diversity) and
    # collapses these ten programs into three, so the base class yields at most ONE
    # own inspiration and does so for only about half of all seeds. On a seed that
    # yields zero, `match_baseline` injects nothing and the comparison is vacuous --
    # which the guard below turns into a failure rather than a false pass.
    import random as _random

    # The base class seeds the global RNG itself (`database.py:176`), so reseeding
    # before construction is overwritten; and its parent selection reaches
    # `random.choice(list(...))` over a set, whose iteration order moves with the
    # process's PYTHONHASHSEED. The result was 1 failure in 8 identical full-suite
    # runs -- always "nothing was injected", i.e. the base class happened to return
    # zero own inspirations and `match_baseline` then had nothing to replace.
    #
    # So the RNG seed is searched rather than assumed: the first seed that makes
    # arm A non-empty is the one both arms are measured at. The comparison is still
    # apples-to-apples (same seed, same island, same count on both sides) and no
    # longer depends on which tests ran first or on the interpreter's hash seed.
    inspirations = own_only = None
    for rng_seed in range(50):
        retriever = FakeRetriever(count=3)
        db = _db(retriever, islands=1, policy="match_baseline")
        _seed(db, n=10)
        _random.seed(rng_seed)
        _, inspirations = db.sample_from_island(0, num_inspirations=5)

        baseline = _db(NullRetriever(), islands=1)
        _seed(baseline, n=10)
        _random.seed(rng_seed)
        _, own_only = baseline.sample_from_island(0, num_inspirations=5)
        if own_only:
            break
    assert own_only, "no RNG seed produced a non-empty arm A; the test is vacuous"

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


def test_changes_render_mode_injects_method_text_without_two_sided_diff_mode() -> None:
    db = _db(FakeRetriever(count=1), render_mode="changes", policy="fixed")
    _seed(db)
    _, inspirations = db.sample_from_island(0, num_inspirations=2)
    external = next(program for program in inspirations if is_external(program.id))
    assert external.code == "historical change 0"
    assert external.language == "text"
    assert external.metadata["render_mode"] == "changes"
    assert external.metadata["source_code_sha256"]


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
    import random as _random

    db = _db(FakeRetriever(count=5), islands=1, max_injected=2, policy="match_baseline")
    _seed(db, n=10)
    _random.seed(1)
    _, inspirations = db.sample_from_island(0, num_inspirations=5)

    baseline = _db(NullRetriever(), islands=1)
    _seed(baseline, n=10)
    _random.seed(1)
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


def test_injection_frequency_gates_whole_iterations() -> None:
    """arXiv:2606.29823's `p`: the per-step probability that memory fires at all.

    Distinct from how many programs one injection carries. At p=0.5 roughly half the
    iterations should see nothing, and those iterations must degrade to stock
    behaviour rather than to an error.
    """
    db = _db(FakeRetriever(count=3), islands=1, frequency=0.5, frequency_seed=7)
    _seed(db, n=10)
    for _ in range(200):
        db.sample_from_island(0, num_inspirations=5)

    opened = sum(1 for r in db.injection_trace if r["gate_open"])
    assert 0.35 < opened / 200 < 0.65, f"gate fired {opened}/200 at p=0.5"
    assert all(r["frequency"] == 0.5 for r in db.injection_trace)
    # A closed gate means no retrieval happened at all, not a failed one.
    closed = [r for r in db.injection_trace if not r["gate_open"]]
    assert closed and all(r["retrieved"] == 0 and not r["injected_ids"] for r in closed)


def test_injection_frequency_one_always_fires() -> None:
    db = _db(FakeRetriever(count=3), islands=1, frequency=1.0)
    _seed(db, n=10)
    for _ in range(30):
        db.sample_from_island(0, num_inspirations=5)
    assert all(r["gate_open"] for r in db.injection_trace)


def test_injection_frequency_zero_never_fires() -> None:
    """p=0 must be indistinguishable from the no-memory arm, not a broken memory arm."""
    db = _db(FakeRetriever(count=3), islands=1, frequency=0.0)
    _seed(db, n=10)
    for _ in range(30):
        db.sample_from_island(0, num_inspirations=5)
    assert not any(r["gate_open"] for r in db.injection_trace)
    assert db.external_ids == set()


def test_invalid_frequency_is_rejected() -> None:
    with pytest.raises(ValueError, match="injection frequency"):
        _db(NullRetriever(), frequency=1.5)


def test_closed_gate_empties_every_prompt_section():
    """A closed gate must zero all THREE program sections, not just inspirations.

    The regression this pins: `sample_from_island` feeds only `inspirations`, while
    `previous_programs` and `top_programs` are sliced from the snapshot's island
    lists inside the worker (process_parallel.py:154-169). A gate that closed only
    the first left the run's own programs in the prompt, so p measured
    `none` -> memory instead of `nocontext` -> memory.
    """
    import types

    from openevolve.process_parallel import ProcessParallelController

    from bench.agent_memory.gem_oe.run_arm import _install_zero_context_gate

    original = ProcessParallelController._create_database_snapshot
    base = {"islands": [["a", "b"], ["c"]], "programs": {}, "current_island": 0}
    try:
        ProcessParallelController._create_database_snapshot = lambda self: dict(base)
        _install_zero_context_gate()

        closed = types.SimpleNamespace(
            database=types.SimpleNamespace(last_gate_open=False)
        )
        snap = ProcessParallelController._create_database_snapshot(closed)
        assert snap["islands"] == [[], []], (
            "closed gate left own programs in the prompt"
        )

        opened = types.SimpleNamespace(
            database=types.SimpleNamespace(last_gate_open=True)
        )
        assert ProcessParallelController._create_database_snapshot(opened)[
            "islands"
        ] == [
            ["a", "b"],
            ["c"],
        ], "open gate must not disturb the snapshot"
    finally:
        ProcessParallelController._create_database_snapshot = original


@pytest.mark.parametrize("frequency, expected", [(0.0, False), (1.0, True)])
def test_gate_draw_is_recorded_for_the_snapshot_hook(frequency, expected):
    """last_gate_open must track the draw; the hook has no other way to see it."""
    db = _db(FakeRetriever(count=3), frequency=frequency)
    _seed(db)
    db.sample_from_island(island_id=0, num_inspirations=5)
    assert db.last_gate_open is expected


@pytest.mark.parametrize("mode, expect_empty", [("empty", True), ("own", False)])
def test_closed_gate_drops_own_inspirations_under_empty(mode, expect_empty):
    """Emptying the island snapshot is not enough on its own.

    `inspirations` reach the worker as an ID list resolved against `programs`, not
    `islands`, so the snapshot hook cannot reach them. A live 8-iteration run caught
    this: iteration 6 drew a closed gate and still rendered one own program.
    """
    db = _db(FakeRetriever(count=3), frequency=0.0, gate_closed=mode)
    _seed(db)
    _, inspirations = db.sample_from_island(island_id=0, num_inspirations=5)
    assert db.last_gate_open is False
    if expect_empty:
        assert inspirations == [], "closed gate leaked own programs into the prompt"
    else:
        assert all(not is_external(p.id) for p in inspirations)
