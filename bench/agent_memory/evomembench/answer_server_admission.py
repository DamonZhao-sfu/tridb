"""Fail-closed admission for the live answer/judge vLLM process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence


def _option(arguments: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(arguments):
        if value == name:
            return arguments[index + 1] if index + 1 < len(arguments) else None
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _main_pid(unit: str) -> int:
    observed = subprocess.run(
        ["systemctl", "--user", "show", "--property=MainPID", "--value", unit],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    pid = int(observed)
    if pid <= 0:
        raise RuntimeError(f"answer unit has no live MainPID: {unit}")
    return pid


def inspect(
    unit: str,
    *,
    expected_model: str = "Qwen/Qwen3.8-27B-FP8",
    expected_served_model: str = "qwen3.8",
    expected_context_tokens: int = 262144,
    expected_max_new_tokens: int = 4096,
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
    generation_raw = _option(arguments, "--override-generation-config")
    thinking_raw = _option(arguments, "--default-chat-template-kwargs")
    try:
        generation = json.loads(generation_raw or "null")
    except json.JSONDecodeError:
        generation = None
    try:
        thinking = json.loads(thinking_raw or "null")
    except json.JSONDecodeError:
        thinking = None
    checks = {
        "unit_cgroup": f"/{unit}" in cgroup,
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES") == "0",
        "model": expected_model in arguments,
        "served_model": _option(arguments, "--served-model-name")
        == expected_served_model,
        "port": _option(arguments, "--port") == "8000",
        "context_tokens": _option(arguments, "--max-model-len")
        == str(expected_context_tokens),
        "prefix_cache": "--enable-prefix-caching" in arguments,
        "prompt_token_details": "--enable-prompt-tokens-details" in arguments,
        "thinking_disabled": thinking == {"enable_thinking": False},
        "temperature": isinstance(generation, dict)
        and float(generation.get("temperature", -1)) == 0.0,
        "top_p": isinstance(generation, dict)
        and float(generation.get("top_p", -1)) == 1.0,
        "top_k": isinstance(generation, dict) and int(generation.get("top_k", 0)) == -1,
        "max_new_tokens": isinstance(generation, dict)
        and int(generation.get("max_new_tokens", -1)) == expected_max_new_tokens,
    }
    return {
        "schema_version": "evomembench_answer_server_admission_v0.1.0",
        "status": "complete" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "unit": unit,
        "main_pid": pid,
        "command_sha256": hashlib.sha256(
            b"\0".join(a.encode() for a in arguments)
        ).hexdigest(),
        "observed": {
            "model": expected_model if expected_model in arguments else None,
            "served_model": _option(arguments, "--served-model-name"),
            "port": _option(arguments, "--port"),
            "context_tokens": _option(arguments, "--max-model-len"),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "thinking": thinking,
            "generation_config": generation,
        },
        "expected": {
            "model": expected_model,
            "served_model": expected_served_model,
            "port": 8000,
            "context_tokens": expected_context_tokens,
            "cuda_visible_devices": "0",
            "thinking": {"enable_thinking": False},
            "generation_config": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": expected_max_new_tokens,
            },
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--expected-max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    report = inspect(args.unit, expected_max_new_tokens=args.expected_max_new_tokens)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(69)


if __name__ == "__main__":
    main()
