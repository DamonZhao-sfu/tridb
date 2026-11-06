"""Unattended queue for the three-dataset TRACE-PR campaign (plan 2026-11-03).

Order: TEST-panel preflight -> c16 real smoke (all datasets) -> ScienceWorld ->
ALFWorld -> WebShop campaigns (TRAIN 5+5+40, then TRAIN-Pareto on TEST200).
Every step waits for the shared evaluation lease and idle GPUs (inside the
smoke / tools.trace_joint_live). The queue stops at the first failed step and
records why; it never starts a campaign after a failed smoke.

Usage: python -m tools.trace_pr_three_dataset_queue --queue <queue.json>
queue.json: {"smoke_prepared": dir, "smoke_out": dir, "runs": [[dataset, run_dir], ...]}
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-optimizer/bin/python")


def write(path, body):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**body, "at": time.time()}, indent=2))
    tmp.replace(path)


def step(status, name, cmd):
    write(status, {"status": "running", "step": name})
    code = subprocess.call(cmd, cwd=ROOT)
    if code != 0:
        write(status, {"status": "stopped", "step": name, "exit_code": code})
        sys.exit(code)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    a = p.parse_args()
    q = json.loads(a.queue.read_text())
    status = a.queue.with_name("queue_status.json")
    step(status, "fixed_test_panel_guard", [PY, "-m", "tools.fixed_test_panel_guard"])
    smoke_out = Path(q["smoke_out"])
    if not (smoke_out / "smoke_report.json").exists():
        step(status, "live_smoke", [PY, "-m", "tools.trace_joint_live_smoke",
                                     "--prepared", q["smoke_prepared"], "--out", str(smoke_out)])
    report = json.loads((smoke_out / "smoke_report.json").read_text())
    if not report.get("all_passed"):
        write(status, {"status": "stopped", "step": "live_smoke", "reason": "smoke failed",
                       "report": str(smoke_out / "smoke_report.json")})
        sys.exit(2)
    for dataset, run_dir in q["runs"]:
        run_status = Path(run_dir) / "status.json"
        if run_status.exists() and json.loads(run_status.read_text()).get("status") == "test_complete":
            continue
        step(status, f"campaign:{dataset}", [PY, "-m", "tools.trace_joint_live", "--run-dir", run_dir])
        final = json.loads(run_status.read_text()).get("status")
        if final != "test_complete":
            write(status, {"status": "stopped", "step": f"campaign:{dataset}", "run_status": final})
            sys.exit(3)
    write(status, {"status": "complete"})


if __name__ == "__main__":
    main()
