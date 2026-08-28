"""Fail-closed admission for the live embedding vLLM process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bench.agent_memory.evomembench.answer_server_admission import (
    _main_pid,
    _option,
)


MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def inspect(
    unit: str,
    *,
    expected_model: str = MODEL,
    expected_revision: str = REVISION,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    pid = _main_pid(unit)
    proc = proc_root / str(pid)
    arguments = [
        value.decode(errors="replace")
        for value in (proc / "cmdline").read_bytes().split(b"\0")
        if value
    ]
    environment = {
        key.decode(errors="replace"): value.decode(errors="replace")
        for item in (proc / "environ").read_bytes().split(b"\0")
        if item and b"=" in item
        for key, value in [item.split(b"=", 1)]
    }
    cgroup = (proc / "cgroup").read_text()
    checks = {
        "unit_cgroup": f"/{unit}" in cgroup,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES") == "1",
        "model": expected_model in arguments,
        "revision": _option(arguments, "--revision") == expected_revision,
        "tokenizer_revision": _option(arguments, "--tokenizer-revision")
        == expected_revision,
        "served_model": _option(arguments, "--served-model-name") == expected_model,
        "runner": _option(arguments, "--runner") == "pooling",
        "port": _option(arguments, "--port") == "8011",
        "tensor_parallel_size": _option(arguments, "--tensor-parallel-size") == "1",
        "dtype": _option(arguments, "--dtype") == "bfloat16",
        "context_tokens": _option(arguments, "--max-model-len") == "32768",
        "gpu_memory_utilization": _option(arguments, "--gpu-memory-utilization")
        == "0.15",
    }
    observed = {
        "model": expected_model if expected_model in arguments else None,
        "revision": _option(arguments, "--revision"),
        "tokenizer_revision": _option(arguments, "--tokenizer-revision"),
        "served_model": _option(arguments, "--served-model-name"),
        "runner": _option(arguments, "--runner"),
        "port": _option(arguments, "--port"),
        "tensor_parallel_size": _option(arguments, "--tensor-parallel-size"),
        "dtype": _option(arguments, "--dtype"),
        "context_tokens": _option(arguments, "--max-model-len"),
        "gpu_memory_utilization": _option(arguments, "--gpu-memory-utilization"),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
    }
    return {
        "schema_version": "evomembench_embedding_server_admission_v0.1.0",
        "status": "complete" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "unit": unit,
        "main_pid": pid,
        "command_sha256": hashlib.sha256(
            b"\0".join(value.encode() for value in arguments)
        ).hexdigest(),
        "observed": observed,
        "expected": {
            "model": expected_model,
            "revision": expected_revision,
            "tokenizer_revision": expected_revision,
            "served_model": expected_model,
            "runner": "pooling",
            "port": 8011,
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "context_tokens": 32768,
            "gpu_memory_utilization": 0.15,
            "cuda_visible_devices": "1",
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    report = inspect(args.unit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(69)


if __name__ == "__main__":
    main()
