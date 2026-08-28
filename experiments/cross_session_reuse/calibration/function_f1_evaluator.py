"""Audited wrapper around the existing deterministic F1 evaluator."""

from __future__ import annotations

import time

from experiments.cross_session_reuse.calibration.ledger import append_attempt
from experiments.e0.openevolve_function_minimization.evaluator import (
    evaluate as evaluate_f1,
)


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    error_message = ""
    try:
        base = evaluate_f1(program_path)
        reliability = float(base.get("reliability_score", 0.0))
        metrics: dict[str, float | str] = {
            **base,
            "validity": reliability,
            "is_buggy": float(reliability < 1.0),
            "eval_seconds": time.monotonic() - started,
            "status": "valid" if reliability == 1.0 else "invalid",
        }
    except Exception as exc:  # pragma: no cover - defensive audit boundary
        error_message = f"{type(exc).__name__}: {exc}"
        metrics = {
            "value_score": 0.0,
            "distance_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
            "validity": 0.0,
            "is_buggy": 1.0,
            "eval_seconds": time.monotonic() - started,
            "status": "error",
            "error_message": error_message,
        }
    append_attempt(
        program_path,
        metrics={
            key: value for key, value in metrics.items() if key != "error_message"
        },
        status=str(metrics["status"]),
        error_message=error_message,
    )
    return metrics
