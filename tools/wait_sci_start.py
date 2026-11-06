"""Wait for the frozen resource limits before entering initialization."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tools.scienceworld_ablation_resource_preflight import collect
from tools.scienceworld_ablation_runtime import verify_current_cgroup
from trimem.experiments import scienceworld_trace_ablation as ab


def wait_for_admission(limits, *, check_quota, observe, sleep, cgroup=None):
    cgroup = cgroup or verify_current_cgroup(limits)["cgroup"]
    while True:
        check_quota()
        try:
            ab.validate_resource_preflight(observe(cgroup, ["0", "1"]), limits)
        except ValueError as exc:
            print(f"waiting for resource preflight: {exc}", flush=True)
            sleep(30)
            continue
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--identity-pid", type=int, required=True)
    parser.add_argument("--identity-start-ticks", type=int, required=True)
    args = parser.parse_args()
    path = Path(os.environ["TRIMEM_ABLATION_SYSTEM_LIMITS"])
    limits = json.loads(path.read_text())
    wait_for_admission(
        limits, check_quota=ab.check_global_quota, observe=collect, sleep=time.sleep
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "tools.start_scienceworld_ablation_initialization",
            "--campaign-root",
            str(args.campaign_root),
            "--identity-pid",
            str(args.identity_pid),
            "--identity-start-ticks",
            str(args.identity_start_ticks),
        ],
    )


if __name__ == "__main__":
    main()
