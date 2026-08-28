from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.e2.gem_math_ale_plan_agent_matrix import (
    PLANS,
    TASKS,
    _cells,
    _command,
    _complete,
    _dry_complete,
)
from experiments.e2.gem_math_ale_plan_report import (
    _fields,
    _injection,
    _merge_private,
    _quality,
)
from experiments.e2.gem_math_plan_replay import _summary

pytestmark = pytest.mark.unit


def test_matrix_expands_17_tasks_by_3_plans_with_rotation(tmp_path: Path) -> None:
    cells = _cells(TASKS, PLANS, tmp_path)
    assert len(cells) == 51
    assert {(row["task_uid"], row["physical_plan"]) for row in cells} == {
        (task, plan) for task in TASKS for plan in PLANS
    }
    assert [row["physical_plan"] for row in cells[:3]] == ["vfwd", "rrev", "aivg"]
    assert [row["physical_plan"] for row in cells[3:6]] == ["rrev", "aivg", "vfwd"]


def test_formal_command_freezes_approved_agent_settings(tmp_path: Path) -> None:
    args = argparse.Namespace(
        python="python",
        dsn="dsn",
        scope="scope",
        llm_base="llm",
        llm_model="model",
        embedding_endpoint="embedding",
        embedding_model="embedding-model",
        max_iterations=None,
        dry_run=False,
        ale_eval_timeout=600,
        math_eval_timeout=240,
    )
    row = _cells(("ale:ahc008",), PLANS, tmp_path)[0]
    command = _command(args, row, tmp_path / "cell")
    joined = " ".join(command)
    assert "--seed 42" in joined
    assert "--injection-frequency 0.5" in joined
    assert "--injection-rate 0.5" in joined
    assert "--num-diverse 10" in joined
    assert "--inject-as changes" in joined
    assert "--eval-timeout 600" in joined
    assert "--language cpp" in joined


def test_complete_receipt_requires_exact_task_plan_and_iterations(
    tmp_path: Path,
) -> None:
    receipt = {
        "status": "complete",
        "task_uid": "math:circle_packing",
        "language": "python",
        "physical_plan": "vfwd",
        "resolved_seed": 42,
        "iterations_expected": 40,
        "iterations_traced": 40,
        "live_seed_fitness": 1.0,
    }
    (tmp_path / "run_receipt.json").write_text(json.dumps(receipt))
    assert _complete(
        tmp_path,
        expected_iterations=40,
        expected_task="math:circle_packing",
        expected_plan="vfwd",
    )
    assert not _complete(
        tmp_path,
        expected_iterations=40,
        expected_task="math:circle_packing",
        expected_plan="rrev",
    )


def test_dry_complete_accepts_resolved_receipt_without_iterations(
    tmp_path: Path,
) -> None:
    receipt = {
        "status": "dry_run",
        "task_uid": "ale:ahc008",
        "language": "cpp",
        "physical_plan": "aivg",
        "resolved_seed": 42,
        "injection_frequency": 0.5,
        "injection_rate_requested": 0.5,
        "injection_slots": 10,
        "max_injected": 5,
        "inject_as": "changes",
    }
    (tmp_path / "run_receipt.json").write_text(json.dumps(receipt))
    assert _dry_complete(tmp_path, expected_task="ale:ahc008", expected_plan="aivg")


def test_replay_summary_emits_domain_and_combined_rollups() -> None:
    template = {
        "exact_order": True,
        "recall_at_10": 1.0,
        "query_uid": "q",
        "embed_ms": 0.0,
        "ann_ms": 1.0,
        "graph_ms": 2.0,
        "predicate_ms": 3.0,
        "dedup_rank_ms": 4.0,
        "hydrate_ms": 5.0,
        "executor_overhead_ms": 6.0,
        "retriever_total_ms": 21.0,
    }
    rows = [
        {**template, "task_uid": "math:t", "physical_plan": "vfwd"},
        {**template, "task_uid": "ale:a", "physical_plan": "vfwd"},
    ]
    tasks = {row["task_uid"] for row in _summary(rows)}
    assert {"__all_math__", "__all_ale__", "__all__"} <= tasks


def test_quality_curve_starts_from_live_not_stored_seed(tmp_path: Path) -> None:
    (tmp_path / "evolution_trace.jsonl").write_text(
        json.dumps({"iteration": 2, "child_metrics": {"combined_score": 12.0}}) + "\n"
    )
    receipt = {
        "task_uid": "ale:ahc008",
        "physical_plan": "vfwd",
        "live_seed_fitness": 10.0,
        "iterations_traced": 40,
    }
    quality, trajectory = _quality(
        tmp_path, receipt, {"seed": 8.0, "corpus_best": 20.0, "gap": 12.0}
    )
    assert quality["seed_score"] == 10.0
    assert quality["stored_seed_score"] == 8.0
    assert trajectory[0]["raw_best_so_far"] == 10.0
    assert trajectory[1]["raw_best_so_far"] == 12.0


def test_injection_gate_fraction_uses_actual_gate_draw(tmp_path: Path) -> None:
    rows = [
        {"gate_open": False, "budget": 5, "rendered": 0},
        {"gate_open": True, "budget": 5, "rendered": 3},
    ]
    (tmp_path / "injection_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    metrics = _injection(tmp_path)
    assert metrics["injection_gate_open_fraction"] == 0.5
    assert metrics["injection_render_rate"] == 0.6


def test_private_merge_and_csv_fields_preserve_ale_only_columns(
    tmp_path: Path,
) -> None:
    qualities = [
        {"domain": "math", "task_uid": "math:t", "physical_plan": "vfwd"},
        {"domain": "ale", "task_uid": "ale:ahc008", "physical_plan": "rrev"},
    ]
    private = [
        {
            "problem": "ahc008",
            "physical_plan": "rrev",
            "final_private_performance": 1234,
            "seed_private_performance": 1200,
            "private_performance_delta_from_seed": 34,
            "private_performance_delta_vs_nocontext": 20,
            "generalization_label": "aligned",
            "private_case_count": 200,
        }
    ]
    path = tmp_path / "private.json"
    path.write_text(json.dumps(private))
    merged, missing = _merge_private(qualities, path)
    assert merged == 1 and not missing
    assert qualities[1]["final_private_performance"] == 1234
    assert "final_private_performance" in _fields(qualities, ())
