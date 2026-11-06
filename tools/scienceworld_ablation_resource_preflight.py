"""Read-only capacity check for the ScienceWorld ablation; never starts jobs."""

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments.scienceworld_trace_ablation import validate_resource_preflight


def collect(cgroup, gpu_ids):
    cgroup = Path(cgroup)
    files = ("cgroup.procs", "cpu.max", "memory.max")
    delegated = all(
        (cgroup / name).is_file() and os.access(cgroup / name, os.W_OK)
        for name in files
    )
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, tail = line.partition(":")
        if key == "MemAvailable":
            values[key] = int(tail.strip().split()[0]) * 1024
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    ).stdout
    gpu = {}
    for line in output.splitlines():
        ident, usage = line.split(",", 1)
        gpu[ident.strip()] = float(usage.strip())
    return {
        "cgroup_delegated": delegated,
        "cgroup_path": str(cgroup.resolve()),
        "available_memory_bytes": values["MemAvailable"],
        "gpu_ids": gpu_ids,
        "gpu_utilization_percent": gpu,
        "observed_at": datetime.now(UTC).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limits", type=Path, required=True, help="JSON with explicit hard limits"
    )
    parser.add_argument(
        "--cgroup", type=Path, required=True, help="already delegated dedicated cgroup"
    )
    parser.add_argument("--gpus", nargs="+", required=True)
    args = parser.parse_args()
    limits = json.loads(args.limits.read_text())
    state = collect(args.cgroup, args.gpus)
    try:
        validate_resource_preflight(state, limits)
    except ValueError as exc:
        print(
            json.dumps(
                {"ready": False, "reason": str(exc), "observation": state}, indent=2
            )
        )
        return 1
    print(
        json.dumps(
            {
                "ready": True,
                "observation": state,
                "note": "Preflight only; hard limits must still be applied by the launcher.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
