"""Structural gates for the formal native-fused EvoTrace Math experiment."""

from __future__ import annotations

import inspect
import json
from argparse import Namespace
from pathlib import Path

from bench.agent_memory.gem_eg.query import W1Engine
from bench.agent_memory.gem_oe.run_arm import _locked_generation_policy
from experiments.e2 import gem_math4_native_fused_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]


def test_native_adapter_has_one_production_tjs_open_call() -> None:
    source = inspect.getsource(W1Engine.run_native_seedless)
    assert source.count("tjs_open(") == 1
    assert "tjs_e0_open" not in source
    assert "NULL" in source
    assert "seedless_plan=native_plan" in source


def test_native_vector_only_adapter_has_one_production_tjs_open_call() -> None:
    source = inspect.getsource(W1Engine.run_native_vector_only)
    assert source.count("tjs_open(") == 1
    assert "tjs_e0_open" not in source
    assert "NULL, 0" in source
    assert 'seedless_plan="vector_stream"' in source
    assert "result.calls = 1" in source


def test_source_anchored_adapter_has_one_production_tjs_open_call() -> None:
    source = inspect.getsource(W1Engine.run_native_source_anchored)
    assert source.count("tjs_open(") == 1
    assert "tjs_e0_open" not in source
    assert "source_vid" in source
    assert "query_vec,\n                    source_vid," in source


def test_formal_retriever_does_not_import_python_plan_executor() -> None:
    for relative in (
        "bench/agent_memory/gem_oe/retrievers.py",
        "bench/agent_memory/gem_oe/run_arm.py",
        "experiments/e2/gem_math4_native_fused_matrix.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "bench.agent_memory.gem_eg.physical" not in source


def test_matrix_is_exactly_24_treatments_plus_four_controls(tmp_path: Path) -> None:
    legacy_tasks = tuple(
        task for task in matrix.SCOREABLE_TASKS if "second_" not in task
    )
    treatments, controls = matrix._cells(legacy_tasks, tmp_path, 20)
    assert len(treatments) == 24
    assert len(controls) == 4
    assert {row["native_fused_plan"] for row in treatments} == {
        "source_anchored",
        "graph_expand",
    }
    assert {row["injection_probability"] for row in treatments} == {0.1, 0.5, 1.0}
    assert all(
        row["top_m"] == (0 if row["native_fused_plan"] == "source_anchored" else 3)
        for row in treatments
    )
    phases = matrix._execution_phases(treatments, controls, legacy_tasks)
    assert phases[0][0] == "no_external_memory_controls"
    assert len(phases[0][1]) == 4
    assert all(row["arm"] == "none" for row in phases[0][1])
    assert [name for name, _ in phases[1:]] == [f"task:{task}" for task in legacy_tasks]
    assert all(
        {row["task_uid"] for row in rows} == {task}
        for task, (_, rows) in zip(legacy_tasks, phases[1:], strict=True)
    )


def test_vector_first_missing_cell_selection_omits_other_branch_and_controls(
    tmp_path: Path,
) -> None:
    treatments, controls = matrix._cells(
        ("math:second_autocorr_ineq",),
        tmp_path,
        10,
        native_plans=("graph_expand",),
        injection_probabilities=(0.1, 1.0),
        include_controls=False,
    )
    assert controls == []
    assert len(treatments) == 2
    assert {row["native_fused_plan"] for row in treatments} == {"graph_expand"}
    assert {row["injection_probability"] for row in treatments} == {0.1, 1.0}
    assert all(row["top_m"] == 3 for row in treatments)


def test_formal_command_pins_same_task_node_seed_and_medium_thinking(
    tmp_path: Path,
) -> None:
    treatments, _ = matrix._cells(("math:circle_packing",), tmp_path, 20)
    args = Namespace(
        python="python",
        dsn="postgresql://example/db",
        scope="evotrace:test",
        llm_bases=("http://localhost:8003/v1",),
        embedding_endpoints=("http://localhost:8012/v1/embeddings",),
        llm_model="qwen3.8-262k",
        llm_generation_policy="baseline",
        llm_primary_max_tokens=8192,
        llm_primary_reasoning_effort="low",
        llm_rescue_max_tokens=16384,
        llm_rescue_reasoning_effort="medium",
        llm_rescue_seed_offset=1,
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        eval_timeout=240,
        dry_run=False,
    )
    command = matrix._command(args, treatments[0], 0, 20)

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--split") == "same_task"
    assert value("--seed-layer") == "node"
    assert value("--native-fused-plan") in {
        "source_anchored",
        "graph_expand",
    }
    assert value("--m-seeds") == (
        "0" if value("--native-fused-plan") == "source_anchored" else "3"
    )
    assert value("--llm-thinking") == "on"
    assert value("--llm-reasoning-effort") == "medium"
    assert value("--llm-generation-policy") == "baseline"
    assert value("--llm-primary-max-tokens") == "8192"
    assert value("--llm-primary-reasoning-effort") == "low"
    assert value("--llm-rescue-max-tokens") == "16384"
    assert value("--llm-rescue-reasoning-effort") == "medium"
    assert value("--llm-rescue-seed-offset") == "1"
    assert value("--max-iterations") == "20"
    assert "--physical-plan" not in command


def test_reduced_resume_accepts_a_complete_longer_run(tmp_path: Path) -> None:
    treatments, _ = matrix._cells(("math:circle_packing",), tmp_path, 10)
    row = treatments[0]
    cell = Path(row["path"])
    cell.mkdir(parents=True)
    receipt = {
        "status": "complete",
        "task_uid": row["task_uid"],
        "resolved_seed": 42,
        "injection_slots": 2,
        "own_top_program_slots": 3,
        "own_diverse_program_slots": 2,
        "inject_as": "code",
        "gate_closed": "own",
        "injection_gate_seed": row["injection_gate_seed"],
        "physical_plan": None,
        "native_fused_plan": row["native_fused_plan"],
        "native_fused_only": True,
        "top_m": row["top_m"],
        "src_is_null": row["native_fused_plan"] != "source_anchored",
        "injection_frequency": row["injection_probability"],
        "injection_rate_requested": 1.0,
        "max_injected": 2,
        "external_inspiration_slots": 2,
        "iterations_expected": 20,
        "iterations_traced": 20,
        "live_seed_fitness": 0.1,
        "llm": {"thinking": "on", "reasoning_effort": "medium"},
    }
    (cell / "run_receipt.json").write_text(json.dumps(receipt))
    assert matrix._is_complete(row, dry_run=False, iterations=10)
    assert not matrix._is_complete(
        row,
        dry_run=False,
        iterations=10,
        generation_policy="strict-rescue",
    )


def test_new_matrix_defaults_to_strict_rescue(tmp_path: Path) -> None:
    args = matrix.build_parser().parse_args(["--out", str(tmp_path)])
    assert args.llm_generation_policy == "strict-rescue"
    assert args.llm_primary_max_tokens == 8192
    assert args.llm_primary_reasoning_effort == "low"
    assert args.llm_rescue_max_tokens == 16384
    assert args.llm_rescue_reasoning_effort == "medium"
    assert args.llm_rescue_seed_offset == 1


def test_running_matrix_directory_lock_preserves_baseline(tmp_path: Path) -> None:
    cell = tmp_path / "cells" / "cell"
    cell.mkdir(parents=True)
    lock = tmp_path / "llm_generation_policy.json"
    lock.write_text(json.dumps({"policy": "baseline"}))

    policy, resolved = _locked_generation_policy(cell)

    assert policy == "baseline"
    assert resolved == lock.resolve()
