"""Parameterized, independently recomputed unit-square circle evaluator."""

from __future__ import annotations

import importlib.util
import math
import os
import time
from pathlib import Path

import numpy as np

from experiments.cross_session_reuse.calibration.ledger import append_attempt

TOLERANCE = 1e-8


def _invalid() -> dict[str, float]:
    return {
        "combined_score": 0.0,
        "sum_radii": 0.0,
        "validity": 0.0,
        "is_buggy": 1.0,
        "boundary_slack": -1.0,
        "overlap_slack": -1.0,
    }


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    status = "valid"
    error_message = ""
    metrics = _invalid()
    try:
        n = int(os.environ.get("CSR_CIRCLE_N", "18"))
        spec = importlib.util.spec_from_file_location("csr_circle_candidate", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load candidate: {program_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        centers_value, radii_value, reported_value = module.run_packing()
        centers = np.asarray(centers_value, dtype=float)
        radii = np.asarray(radii_value, dtype=float)
        reported = float(reported_value)
        if centers.shape != (n, 2) or radii.shape != (n,):
            raise ValueError(f"wrong shape: centers={centers.shape}, radii={radii.shape}")
        if not np.isfinite(centers).all() or not np.isfinite(radii).all():
            raise ValueError("non-finite geometry")
        if np.any(radii <= TOLERANCE):
            raise ValueError("non-positive radius")
        boundary_slack = float(
            np.min(
                np.column_stack(
                    (
                        centers[:, 0] - radii,
                        centers[:, 1] - radii,
                        1.0 - centers[:, 0] - radii,
                        1.0 - centers[:, 1] - radii,
                    )
                )
            )
        )
        overlap_slack = math.inf
        for left in range(n):
            for right in range(left + 1, n):
                distance = float(np.linalg.norm(centers[left] - centers[right]))
                overlap_slack = min(
                    overlap_slack, distance - float(radii[left] + radii[right])
                )
        sum_radii = float(np.sum(radii))
        reported_match = abs(sum_radii - reported) <= TOLERANCE
        valid = (
            boundary_slack >= -TOLERANCE
            and overlap_slack >= -TOLERANCE
            and reported_match
        )
        metrics = {
            "combined_score": sum_radii if valid else 0.0,
            "sum_radii": sum_radii if valid else 0.0,
            "validity": float(valid),
            "is_buggy": float(not valid),
            "boundary_slack": boundary_slack,
            "overlap_slack": overlap_slack,
            "reported_sum_match": float(reported_match),
        }
        if not valid:
            status = "invalid"
            error_message = "invalid geometry or reported sum"
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
    metrics["eval_seconds"] = time.monotonic() - started
    output: dict[str, float | str] = {
        **metrics,
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
    print(evaluate(str(Path(__file__).with_name("circle_packing_initial.py"))))

