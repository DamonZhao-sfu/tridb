"""Fork-safe evaluator for a fused bias + SiLU GPU kernel."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.cross_session_reuse.calibration.ledger import append_attempt

WORKER = Path(__file__).with_name("gpu_fused_silu_worker.py")
WORKER_TIMEOUT_SECONDS = 180


def _run_worker(program_path: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            program_path,
            os.environ.get("CSR_GPU_ROWS", "1024"),
            os.environ.get("CSR_GPU_COLUMNS", "1024"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"GPU worker exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"GPU worker returned invalid JSON: {completed.stdout[-2000:]}"
        ) from exc


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    status = "valid"
    error_message = ""
    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "validity": 0.0,
        "is_buggy": 1.0,
        "speedup": 0.0,
        "max_abs_error": -1.0,
    }
    try:
        result = _run_worker(program_path)
        if not result.get("ok"):
            raise RuntimeError(
                f"{result.get('error_type', 'GPUWorkerError')}: "
                f"{result.get('error_message', '')}"
            )
        metrics = {key: float(value) for key, value in result["metrics"].items()}
    except subprocess.TimeoutExpired:
        status = "timeout"
        error_message = f"GPU worker exceeded {WORKER_TIMEOUT_SECONDS}s"
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
    print(evaluate(str(Path(__file__).with_name("gpu_fused_silu_initial.py"))))

