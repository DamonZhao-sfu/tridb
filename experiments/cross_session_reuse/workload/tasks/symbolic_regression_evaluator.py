"""Deterministic held-out evaluator for symbolic regression."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np

from experiments.cross_session_reuse.calibration.ledger import append_attempt


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    status = "valid"
    error_message = ""
    score = 0.0
    mse = 1e9
    try:
        spec = importlib.util.spec_from_file_location("csr_symbolic_candidate", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load candidate: {program_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        x = np.linspace(-2.93, 2.87, 257)
        expected = np.sin(x) + 0.5 * x * x - 0.25 * x
        actual = np.asarray(module.predict(x), dtype=float)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError("prediction has wrong shape or non-finite values")
        mse = float(np.mean((actual - expected) ** 2))
        score = 1.0 / (1.0 + mse)
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
    output: dict[str, float | str] = {
        "combined_score": score,
        "mse": mse,
        "validity": float(status == "valid"),
        "is_buggy": float(status != "valid"),
        "eval_seconds": time.monotonic() - started,
        "status": status,
        "error_message": error_message,
    }
    append_attempt(
        program_path,
        metrics={key: value for key, value in output.items() if key != "error_message"},
        status=status,
        error_message=error_message,
    )
    return output


if __name__ == "__main__":
    print(evaluate(str(Path(__file__).with_name("symbolic_regression_initial.py"))))

