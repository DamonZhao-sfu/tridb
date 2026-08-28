"""Hidden-test evaluator for the code-repair family."""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

from experiments.cross_session_reuse.calibration.ledger import append_attempt


def _load(path: str):
    spec = importlib.util.spec_from_file_location("csr_repair_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cases(module, variant: str) -> list[bool]:
    if variant == "chunk":
        return [
            module.chunked([], 3) == [],
            module.chunked([1], 3) == [[1]],
            module.chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]],
            module.chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]],
        ]
    if variant == "path":
        return [
            module.normalize_path("/a/./b") == "/a/b",
            module.normalize_path("/a/x/../b") == "/a/b",
            module.normalize_path("a//b") == "a/b",
            module.normalize_path("/") == "/",
        ]
    if variant == "record":
        return [
            module.parse_record("a=1,b=2") == {"a": "1", "b": "2"},
            module.parse_record("name=Ada Lovelace, role = engineer ")
            == {"name": "Ada Lovelace", "role": "engineer"},
            module.parse_record("token=a=b=c") == {"token": "a=b=c"},
            module.parse_record("") == {},
        ]
    raise ValueError(f"unknown repair variant: {variant}")


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    variant = os.environ.get("CSR_REPAIR_VARIANT", "chunk")
    status = "valid"
    error_message = ""
    passed = 0
    total = 4
    try:
        results = _cases(_load(program_path), variant)
        passed = sum(bool(result) for result in results)
        if passed != total:
            status = "invalid"
            error_message = f"{total - passed} hidden tests failed"
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
    pass_rate = passed / total
    output: dict[str, float | str] = {
        "combined_score": pass_rate,
        "test_pass_rate": pass_rate,
        "tests_passed": float(passed),
        "tests_total": float(total),
        "validity": float(passed == total),
        "is_buggy": float(passed != total),
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
    print(evaluate(str(Path(__file__).with_name("code_repair_initial.py"))))

