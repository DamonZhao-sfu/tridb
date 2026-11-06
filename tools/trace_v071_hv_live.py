"""Live TRAIN HV of the TRACE v0.7.1 arms in the page's "Search efficiency" section (user 2026-11-05).

Read-only on the run ledgers. Adds or refreshes one series per dataset in ``hcp-data`` of qa-trajectory.html, in the
page's fixed normalization (predefined-v1), next to the historical methods. Same rules as the current TRACE-latent
series (results/atlas_cleanup_hv_20261105/update_page.py):
* archive = the 10 init configurations + every complete, never-pruned exploration configuration; NoMemory excluded,
  because the other series have no NoMemory point;
* x = wall hours since the first exploration reservation, from result-file mtimes; hour 0 = the init archive;
* a checkpoint per completed configuration; pruned and in-progress configurations do not enter HV.
Plot data: <out>/hv_all_series.csv and one sheet in triMEM_optimizer/trimem_experiments.xlsx.

    python -m tools.trace_v071_hv_live [--interval 600] [--unit trace-v071-e2e-20261105]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import fcntl
import json
import math
import re
import sqlite3
import statistics
import subprocess
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
OUT = ROOT / "results/trace_v070_e2e_20261105/hv_live"
SERIES_ID = "trace-v071-latent-20261105"
COLOR = "#d95f02"
DATASETS = ("alfworld", "sciworld", "webshop")
FIELDS = ("raw_score", "total_tokens", "latency_seconds")


def hv(points) -> float:
    """Exact union of origin-anchored boxes (same recursion as the page's current TRACE-latent series)."""
    points = [tuple(p) for p in points if all(v > 0 for v in p)]
    if not points:
        return 0.0
    if len(points[0]) == 1:
        return max(p[0] for p in points)
    total, previous = 0.0, 0.0
    for bound in sorted({p[0] for p in points}):
        total += (bound - previous) * hv([p[1:] for p in points if p[0] >= bound])
        previous = bound
    return total


def load_configs(run: Path) -> dict:
    db = sqlite3.connect(f"file:{run}/ledger.sqlite?mode=ro", uri=True, timeout=30)
    try:
        cfgs = {k: {"key": k, "created": c, "states": {}, "receipts": []}
                for k, c in db.execute("SELECT key, created FROM configs")}
        for key, state, receipt in db.execute("SELECT config, state, receipt FROM episodes"):
            cfgs[key]["states"][state] = cfgs[key]["states"].get(state, 0) + 1
            if state == "done":
                cfgs[key]["receipts"].append(json.loads(receipt))
    finally:
        db.close()
    return cfgs


def build(dataset: str, run: Path, norm: dict, now: str) -> dict | None:
    manifest = json.loads((run / "manifest.json").read_text())
    size, initial = int(manifest["v070"]["sampling"]["size"]), set(manifest["initial_keys"])
    cfgs = load_configs(run)
    lower, upper = norm["lower"], norm["upper"]

    def point(c):
        q, tokens, latency = (statistics.mean(r[f] for r in c["receipts"]) for f in FIELDS)
        values = (q / (100 if dataset == "sciworld" else 1), -math.log(tokens), -math.log(latency * 1000))
        return [(v - lo) / (hi - lo) for v, lo, hi in zip(values, lower, upper)]

    def complete(c):
        return c["states"] == {"done": size}

    if not all(complete(cfgs[k]) for k in initial):
        return None
    explore = [c for k, c in cfgs.items() if k not in initial]
    start = min((c["created"] for c in explore), default=None)
    events = []
    for c in explore:
        stamps = [Path(r["result_file"]).stat().st_mtime for r in c["receipts"]]
        c["done_at"] = max(stamps, default=None)
        events += [((t - start) / 3600, r["total_tokens"]) for t, r in zip(stamps, c["receipts"])]
    events.sort()
    times = [e[0] for e in events]
    token_sums = [0]
    for _, tokens in events:
        token_sums.append(token_sums[-1] + tokens)
    archive = [point(cfgs[k]) for k in sorted(initial)]
    full = sorted((c for c in explore if complete(c)), key=lambda c: c["done_at"])
    checkpoints: list[dict] = []
    for index, c in enumerate([None, *full]):
        if c:
            archive.append(point(c))
        hours = (c["done_at"] - start) / 3600 if c else 0.0
        n = bisect.bisect_right(times, hours)
        metrics = {"hv3": hv(archive), "hv_tokens": hv([p[:2] for p in archive]),
                   "hv_latency": hv([[p[0], p[2]] for p in archive])}
        cp = {"round": index, "source_round": index, "completion_checkpoint": index, "configs": len(archive),
              "hours": hours, "source_hours": hours, "episodes": n, "source_episodes": n + len(initial) * size,
              "tokens": None, "source_tokens": None, "recorded_episode_tokens": token_sums[n],
              "token_status": "unavailable full search accounting; episode subtotal exported separately",
              "time_status": "reconstructed wall time from config reservation / result-file mtimes",
              "episode_status": "recorded completed TRAIN episodes, including pruned prefixes",
              "seed": manifest["seed"], "key": c["key"] if c else "shared-init-10", **metrics}
        for m, v in metrics.items():
            cp["delta_" + m] = v - (checkpoints[0][m] if checkpoints else v)
        checkpoints.append(cp)
    pruned = sum(1 for c in explore if c["states"].get("pruned"))
    return {"id": SERIES_ID, "name": "TRACE-latent v0.7.1 (live, 1 seed)", "color": COLOR, "historical": False,
            "checkpoints": checkpoints, "source": str(run.relative_to(ROOT) / "ledger.sqlite"),
            "note": "Live TRAIN snapshot. Space v0.7.0 (rank optional, no LLM rank, direction Both, unit whole), each "
                    "configuration on its own stratified 100 tasks (TEST type mix), SemFilter v2c in exploration. "
                    f"{len(full)} complete and {pruned} pruned exploration configurations so far. Task sample, space "
                    "and SemFilter differ from the other curves: descriptive comparison only.",
            "budget_scope": "Shared 10-config initialization excluded; hour 0 = first exploration reservation. "
                            "Pruned and in-progress configurations are not in HV. NoMemory excluded. No TEST data.",
            "timing_kind": "reconstructed_wall", "snapshot_at": now}


def export(data: dict) -> list[dict]:
    rows = [{"dataset": ds["key"], "series": r["name"], "current_v071": r["id"] == SERIES_ID,
             **{k: cp.get(k) for k in ("hours", "configs", "episodes", "hv3", "hv_tokens", "hv_latency",
                                       "delta_hv3")}}
            for ds in data["datasets"] for r in ds["runs"] for cp in r["checkpoints"]]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "hv_all_series.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    subprocess.run([str(ROOT / ".venv/bin/python"), "-c", """
import csv, sys
from openpyxl import load_workbook
x, name = 'triMEM_optimizer/trimem_experiments.xlsx', 'hv_live_v071'
wb = load_workbook(x)
if name in wb.sheetnames:
    del wb[name]
ws = wb.create_sheet(name)
for row in csv.reader(open(sys.argv[1])):
    ws.append(row)
wb.save(x)
""", str(OUT / "hv_all_series.csv")], cwd=ROOT, check=True)
    return rows


def update_once() -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with ExitStack() as stack:
        for name in ("qa-trajectory.lock", "qa-trajectory.publish.lock"):
            fcntl.flock(stack.enter_context(PAGE.with_name(name).open("a")), fcntl.LOCK_EX)
        page = PAGE.read_text()
        m = re.search(r'(<script id="hcp-data"[^>]*>)(.*?)(</script>)', page, re.S)
        data = json.loads(m[2])
        norm = data["normalization"]
        added = []
        for ds in data["datasets"]:
            run = ROOT / f"results/trace_v071_latent_{ds['key']}_20261105"
            if not ((run / "manifest.json").exists() and (run / "ledger.sqlite").exists()):
                continue
            series = build(ds["key"], run, norm, now)
            if series is None:
                continue
            ds["runs"] = [r for r in ds["runs"] if r["id"] != SERIES_ID] + [series]
            added.append({"dataset": ds["key"], "hours": series["checkpoints"][-1]["hours"],
                          "configs": series["checkpoints"][-1]["configs"], "hv3": series["checkpoints"][-1]["hv3"]})
        if not added:
            return []
        data["generated"] = now
        OUT.mkdir(parents=True, exist_ok=True)
        backup = OUT / "qa-trajectory.before.html"
        if not backup.exists():
            backup.write_text(page)
        page = page[: m.start(2)] + json.dumps(data).replace("</", "<\\/") + page[m.end(2):]
        tmp = PAGE.with_suffix(".hvlive.tmp")
        tmp.write_text(page)
        tmp.replace(PAGE)
    export(data)
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=0.0, help="seconds between updates; 0 = once")
    ap.add_argument("--unit", default="trace-v071-e2e-20261105", help="stop after this unit ends (final update)")
    a = ap.parse_args()
    while True:
        print(json.dumps({"at": time.time(), "updated": update_once()}), flush=True)
        if not a.interval:
            return
        active = subprocess.run(["systemctl", "--user", "is-active", "-q", a.unit]).returncode == 0
        if not active:
            print(json.dumps({"at": time.time(), "final": update_once()}), flush=True)
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
