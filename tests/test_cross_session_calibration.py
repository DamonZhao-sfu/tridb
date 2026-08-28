from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from experiments.cross_session_reuse.calibration.circle_c18_evaluator import (
    validate_packing,
)
from experiments.cross_session_reuse.calibration.runner import (
    aggregate,
    session_specs,
    summarize_session,
)


def test_c18_initial_program_passes_independent_validator():
    path = Path("experiments/cross_session_reuse/calibration/circle_c18_initial.py")
    spec = importlib.util.spec_from_file_location("circle_c18_initial", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    centers, radii, reported_sum = module.run_packing()
    metrics, error = validate_packing(centers, radii, reported_sum)
    assert error == ""
    assert metrics["validity"] == 1.0
    assert metrics["combined_score"] == np.sum(radii)


def test_c18_validator_rejects_overlap_and_wrong_shape():
    centers = np.full((18, 2), 0.5)
    radii = np.full(18, 0.01)
    metrics, error = validate_packing(centers, radii, float(np.sum(radii)))
    assert error == "circle_overlap"
    assert metrics["combined_score"] == 0.0

    metrics, error = validate_packing(centers[:17], radii[:17], 0.17)
    assert error.startswith("wrong_shape")
    assert metrics["validity"] == 0.0


def test_calibration_matrix_has_six_sessions_and_sixty_child_slots():
    specs = session_specs()
    assert len(specs) == 6
    assert {spec.task_id for spec in specs} == {"function_f1", "circle_c18"}
    assert len({spec.session_id for spec in specs}) == 6
    assert len(specs) * 10 == 60


def test_summary_accounts_for_trace_and_generation_failure(tmp_path):
    spec = session_specs()[0]
    session_dir = tmp_path / spec.session_id
    logs = session_dir / "logs"
    logs.mkdir(parents=True)
    ledger = [
        {"program_role": "initial", "status": "valid"},
        {"program_role": "child", "status": "valid"},
    ]
    (session_dir / "attempt_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
    )
    (session_dir / "evolution_trace.jsonl").write_text("{}\n", encoding="utf-8")
    (session_dir / "driver.log").write_text(
        "Iteration 2 error: No valid diffs found in response\n", encoding="utf-8"
    )
    summary = summarize_session(
        spec, session_dir, iterations=2, returncode=0, wall_seconds=1.0
    )
    assert summary["audit_complete"] is True
    assert summary["calibration_complete"] is False
    assert summary["evaluated_child_nodes"] == 1
    assert len(summary["generation_failures"]) == 1

    manifest = aggregate([summary], iterations=2)
    assert manifest["actual_program_nodes"] == 2
    assert manifest["configured_generated_attempt_slots"] == 2
    assert manifest["calibration_complete"] is False
