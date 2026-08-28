"""Fail-closed admission for an uncontended two-GPU vLLM startup window."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


def vllm_frontends(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Return vLLM launchers even before their CUDA engine appears in nvidia-smi."""
    rows: list[dict[str, Any]] = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            arguments = [
                item.decode(errors="replace")
                for item in (proc / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            cgroup = (proc / "cgroup").read_text()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        launcher = False
        for index, argument in enumerate(arguments):
            following = arguments[index + 1] if index + 1 < len(arguments) else ""
            if Path(argument).name == "vllm" and following == "serve":
                launcher = True
                break
            if argument == "-m" and following.startswith("vllm."):
                launcher = True
                break
            if "/vllm/entrypoints/" in argument and argument.endswith(".py"):
                launcher = True
                break
        if launcher:
            unit = next(
                (
                    part
                    for part in reversed(cgroup.strip().split("/"))
                    if part.endswith(".service") or part.endswith(".scope")
                ),
                None,
            )
            rows.append({"pid": int(proc.name), "cgroup_unit": unit})
    return sorted(rows, key=lambda row: row["pid"])


def inspect(proc_root: Path = Path("/proc")) -> dict[str, Any]:
    errors: list[str] = []
    try:
        gpu_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        gpu_rows = []
        errors.append(f"gpu_inventory: {type(exc).__name__}")
    try:
        compute_rows = [
            line
            for line in subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError) as exc:
        compute_rows = []
        errors.append(f"compute_inventory: {type(exc).__name__}")
    frontends = vllm_frontends(proc_root)
    return {
        "schema_version": "evomembench_gpu_startup_admission_v0.1.0",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "passed": not errors
        and len(gpu_rows) == 2
        and not compute_rows
        and not frontends,
        "gpu_count": len(gpu_rows),
        "compute_processes": compute_rows,
        "vllm_frontends": frontends,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = inspect()
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    if not report["passed"]:
        raise SystemExit(69)


if __name__ == "__main__":
    main()
