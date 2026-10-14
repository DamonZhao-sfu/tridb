"""Pure-Python tests for the TriMEM replay harness (no database, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from trimem.replay.analyze import cell_summary, identity, main
from trimem.replay.run import build_prompt, extract_code
from trimem.replay.tasks import Task


def _task() -> Task:
    return Task(
        "math:x", "x", Path("/dev/null"), "Maximise f.", "def f(): return 1\n", 0.5
    )


def test_extract_code_takes_the_longest_python_block() -> None:
    text = "intro\n```python\nshort\n```\nmore\n```python\nlonger block\nline\n```"
    assert extract_code(text) == "longer block\nline\n"
    assert extract_code("no code here") is None


def test_prompt_contains_memory_only_when_retrieved() -> None:
    bare = build_prompt(_task(), "code", 0.5, [])
    assert "# Memory" not in bare and "combined_score = 0.500000" in bare
    item = {
        "parent_fitness": 0.4,
        "fitness": 0.6,
        "changes": "use annealing",
        "code": "x = 1",
    }
    rich = build_prompt(_task(), "code", 0.5, [item])
    assert "# Memory" in rich and "0.4000 -> 0.6000" in rich and "use annealing" in rich


def _rec(it: int, fit: float, best: float, ids: list[int]) -> dict:
    return {
        "task": "math:x",
        "arm": "A2-trimem",
        "iteration": it,
        "parent_fitness": 0.5,
        "best_fitness": best,
        "retrieved_ids": ids,
        "evaluation": {"fitness": fit, "ms": 10.0},
        "generation": {"ms": 100.0, "prompt_tokens": 7, "completion_tokens": 3},
        "retrieval": {
            "ms": 5.0,
            "sem_prompt_tokens": 2,
            "sem_completion_tokens": 1,
            "deferred": {
                "ms": 1.0,
                "prompt_tokens": 4,
                "completion_tokens": 1,
                "calls": 1,
            },
        },
        "construction": {
            "embed": {"ms": 2.0, "prompt_tokens": 9},
            "derive": {
                "ms": 3.0,
                "prompt_tokens": 6,
                "completion_tokens": 2,
                "calls": 1,
            },
            "insert_ms": 1.0,
        },
    }


def test_cell_summary_adds_up_phases() -> None:
    recs = [_rec(1, 0.6, 0.6, [1, 2]), _rec(2, 0.0, 0.6, [3])]
    s = cell_summary(recs, 0.5)
    assert s["gain"] == 0.6 - 0.5 and s["valid"] == 1
    assert s["construction_ms"] == 2 * (2 + 3 + 1)
    assert s["retrieval_ms"] == 2 * (5 + 1) and s["retrieval_deferred_in"] == 8
    assert s["generation_in"] == 14 and s["curve"] == [0.5, 0.6, 0.6]


def test_identity_counts_equal_contexts_and_first_divergence() -> None:
    a = cell_summary([_rec(1, 0.6, 0.6, [1]), _rec(2, 0.7, 0.7, [2])], 0.5)
    b = cell_summary([_rec(1, 0.6, 0.6, [1]), _rec(2, 0.65, 0.65, [9])], 0.5)
    assert identity(a, b) == {
        "same_context_iterations": 1,
        "first_fitness_divergence": 2,
    }


def test_main_aggregates_a_run_directory(tmp_path: Path) -> None:
    for arm in ("A2-trimem", "A3-trimem-alt"):
        lines = [_rec(1, 0.6, 0.6, [1]) | {"arm": arm}]
        (tmp_path / f"c__{arm}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in lines)
        )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "cells": [
                    {"task": "math:x", "arm": "A2-trimem", "seed_fitness": 0.5},
                    {"task": "math:x", "arm": "A3-trimem-alt", "seed_fitness": 0.5},
                ]
            }
        )
    )
    out = main(str(tmp_path))
    assert set(out["arms"]) == {"A2-trimem", "A3-trimem-alt"}
    assert (
        out["identity"]["math:x|A2-trimem~A3-trimem-alt"]["same_context_iterations"]
        == 1
    )
    assert sum(a["wins"] for a in out["arms"].values()) == 1
