"""Sample host and GPU resources while a systemd experiment unit is active."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    return parser.parse_args()


def command(*parts: str) -> tuple[int, str]:
    result = subprocess.run(
        parts,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.returncode, result.stdout.strip()


def sample(unit: str) -> dict[str, object]:
    state_code, state = command(
        "systemctl", "--user", "show", unit, "--property=ActiveState", "--value"
    )
    props_code, props = command(
        "systemctl",
        "--user",
        "show",
        unit,
        "--property=MainPID,MemoryCurrent,CPUUsageNSec,TasksCurrent",
    )
    gpu_code, gpu = command(
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits",
    )
    return {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "monotonic_ns": time.perf_counter_ns(),
        "unit": unit,
        "active_state": state if state_code == 0 else "unknown",
        "systemd_properties": props if props_code == 0 else None,
        "gpu_csv": gpu if gpu_code == 0 else None,
        "errors": {
            "state_returncode": state_code,
            "properties_returncode": props_code,
            "gpu_returncode": gpu_code,
        },
    }


def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing existing resource trace: {args.output}")
    seen_active = False
    with args.output.open("x", encoding="utf-8") as sink:
        while True:
            row = sample(args.unit)
            sink.write(json.dumps(row, sort_keys=True) + "\n")
            sink.flush()
            state = row["active_state"]
            if state in {"active", "activating"}:
                seen_active = True
            elif seen_active:
                break
            elif state in {"failed", "inactive"}:
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
