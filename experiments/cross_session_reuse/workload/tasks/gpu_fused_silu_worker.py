"""One-process CUDA correctness and timing worker; emits one JSON object."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import traceback

import torch


def _load(path: str):
    spec = importlib.util.spec_from_file_location("csr_gpu_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(function, x: torch.Tensor, bias: torch.Tensor) -> float:
    for _ in range(5):
        function(x, bias)
    torch.cuda.synchronize()
    samples = []
    for _ in range(15):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function(x, bias)
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def run(program_path: str, rows: int, columns: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    candidate = _load(program_path).fused_silu
    device = torch.device("cuda:0")
    max_error = 0.0
    for seed, shape in ((1701, (rows, columns)), (1702, (37, columns))):
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        bias = torch.randn(
            (shape[1],), device=device, dtype=torch.float32, generator=generator
        )
        expected = torch.nn.functional.silu(x + bias)
        actual = candidate(x, bias)
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
            raise ValueError("candidate returned the wrong type or shape")
        max_error = max(max_error, float((actual - expected).abs().max().item()))
    if not torch.isfinite(actual).all() or max_error > 2e-5:
        raise ValueError(f"correctness failure: max_abs_error={max_error}")
    generator = torch.Generator(device=device).manual_seed(1703)
    x = torch.randn(
        (rows, columns), device=device, dtype=torch.float32, generator=generator
    )
    bias = torch.randn(
        (columns,), device=device, dtype=torch.float32, generator=generator
    )
    baseline_ms = _measure(lambda a, b: torch.nn.functional.silu(a + b), x, bias)
    candidate_ms = _measure(candidate, x, bias)
    speedup = baseline_ms / max(candidate_ms, 1e-6)
    return {
        "ok": True,
        "metrics": {
            "combined_score": min(speedup, 10.0),
            "validity": 1.0,
            "is_buggy": 0.0,
            "speedup": speedup,
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
            "max_abs_error": max_error,
        },
    }


def main() -> int:
    try:
        result = run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    except Exception as exc:
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc()[-4000:],
        }
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

