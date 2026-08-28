"""Strict, independently recomputed evaluator for the C18 packing task."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.cross_session_reuse.calibration.ledger import append_attempt

N_CIRCLES = 18
GEOMETRY_TOLERANCE = 1e-8
CANDIDATE_TIMEOUT_SECONDS = 20

_CANDIDATE_DRIVER = r"""
import contextlib
import importlib.util
import io
import json
import sys
import traceback

import numpy as np

path = sys.argv[1]
stdout = io.StringIO()
stderr = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        spec = importlib.util.spec_from_file_location("csr_candidate", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        centers, radii, reported_sum = module.run_packing()
    result = {
        "ok": True,
        "centers": np.asarray(centers).tolist(),
        "radii": np.asarray(radii).tolist(),
        "reported_sum": float(reported_sum),
        "candidate_stdout": stdout.getvalue()[-4000:],
        "candidate_stderr": stderr.getvalue()[-4000:],
    }
except Exception as exc:
    result = {
        "ok": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc()[-8000:],
        "candidate_stdout": stdout.getvalue()[-4000:],
        "candidate_stderr": stderr.getvalue()[-4000:],
    }
print(json.dumps(result, allow_nan=True))
"""


def _run_candidate(program_path: str) -> dict[str, Any]:
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", _CANDIDATE_DRIVER, program_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=CANDIDATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise TimeoutError(f"candidate exceeded {CANDIDATE_TIMEOUT_SECONDS}s") from exc
    if process.returncode != 0:
        raise RuntimeError(
            f"candidate driver exited {process.returncode}: {stderr[-2000:]}"
        )
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"candidate returned invalid JSON: {stdout[-2000:]}"
        ) from exc


def validate_packing(
    centers_value: Any, radii_value: Any, reported_sum: Any
) -> tuple[dict[str, float], str]:
    """Return independently recomputed metrics and a stable failure reason."""
    try:
        centers = np.asarray(centers_value, dtype=float)
        radii = np.asarray(radii_value, dtype=float)
        reported = float(reported_sum)
    except (TypeError, ValueError) as exc:
        return _invalid_metrics(), f"non_numeric_output: {exc}"

    if centers.shape != (N_CIRCLES, 2) or radii.shape != (N_CIRCLES,):
        return (
            _invalid_metrics(),
            f"wrong_shape: centers={centers.shape}, radii={radii.shape}",
        )
    if not np.isfinite(centers).all() or not np.isfinite(radii).all():
        return _invalid_metrics(), "non_finite_geometry"
    if not math.isfinite(reported):
        return _invalid_metrics(), "non_finite_reported_sum"
    if np.any(radii <= GEOMETRY_TOLERANCE):
        return _invalid_metrics(), "non_positive_radius"

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
    for left in range(N_CIRCLES):
        for right in range(left + 1, N_CIRCLES):
            distance = float(np.linalg.norm(centers[left] - centers[right]))
            overlap_slack = min(
                overlap_slack, distance - float(radii[left] + radii[right])
            )
    sum_radii = float(np.sum(radii))
    reported_match = float(abs(sum_radii - reported) <= 1e-8)
    valid = (
        boundary_slack >= -GEOMETRY_TOLERANCE and overlap_slack >= -GEOMETRY_TOLERANCE
    )
    metrics = {
        "sum_radii": sum_radii if valid else 0.0,
        "validity": float(valid),
        "is_buggy": float(not valid),
        "boundary_slack": boundary_slack,
        "overlap_slack": overlap_slack,
        "reported_sum_match": reported_match,
        "combined_score": sum_radii if valid else 0.0,
    }
    if not valid:
        reason = "boundary_violation" if boundary_slack < 0 else "circle_overlap"
        return metrics, reason
    if not reported_match:
        return metrics, "reported_sum_mismatch"
    return metrics, ""


def _invalid_metrics() -> dict[str, float]:
    return {
        "sum_radii": 0.0,
        "validity": 0.0,
        "is_buggy": 1.0,
        "boundary_slack": -1.0,
        "overlap_slack": -1.0,
        "reported_sum_match": 0.0,
        "combined_score": 0.0,
    }


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    error_message = ""
    status = "valid"
    try:
        candidate = _run_candidate(program_path)
        if not candidate.get("ok"):
            error_message = (
                f"{candidate.get('error_type', 'CandidateError')}: "
                f"{candidate.get('error_message', '')}"
            )
            metrics = _invalid_metrics()
            status = "error"
        else:
            metrics, error_message = validate_packing(
                candidate.get("centers"),
                candidate.get("radii"),
                candidate.get("reported_sum"),
            )
            status = "valid" if metrics["validity"] == 1.0 else "invalid"
    except TimeoutError as exc:
        metrics = _invalid_metrics()
        metrics["timeout"] = 1.0
        error_message = str(exc)
        status = "timeout"
    except Exception as exc:  # pragma: no cover - defensive audit boundary
        metrics = _invalid_metrics()
        error_message = f"{type(exc).__name__}: {exc}"
        status = "error"

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
    candidate_path = Path(__file__).with_name("circle_c18_initial.py")
    print(json.dumps(evaluate(str(candidate_path)), indent=2, sort_keys=True))
