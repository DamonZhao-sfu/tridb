"""OFFLINE / dry-run latency-cap reporter for TRACE v0.7.0 runs (fix round 1: never prunes).

The live latency cap runs inside ``trace_v070_pruning.staged_evaluate_v070`` at each stage barrier on the exact
own prefix (``pruning.latency_cap``). This tool only REPORTS what the stock rule (``tools.latent_latency_pruner``
``_rows`` / ``decide``, read-only import) would decide on the current ledger. The live front excludes NoMemory
(user 2026-11-05); this reporter can add NoMemory to the front as a what-if (default on; ``--no-baseline-front``
gives the live front). It appends to ``latency_prune_report.jsonl`` and refuses any non-dry use. NoMemory is
never judged (asserted).

Usage: python -m tools.latent_latency_pruner_v070 --run-dir R [--factor F] [--step 20] [--once] [--no-baseline-front]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path

from tools import latent_latency_pruner as stock
from trimem.experiments.trace_v070_pruning import FACTORS

PROTOCOL = "latency-cap-v070-nomemory-front-report-v1"
NOMEMORY_ROW = "NoMemory"


def nomemory_row(run: Path) -> dict | None:
    """The NoMemory TRAIN row in ``_rows`` format (complete=True only when its whole sample is done)."""
    from trimem.experiments.trace_v070_baseline import baseline_entry

    e = baseline_entry(run, "TRAIN")
    if e is None or not e["receipts"]:
        return None
    done = e["receipts"]
    return {"states": {"done": len(done)}, "n": len(done),
            "q": statistics.mean(x["raw_score"] for x in done),
            "tok": statistics.mean(x["total_tokens"] for x in done),
            "lat": statistics.mean(x["latency_seconds"] for x in done),
            "complete": e["complete"], "key": e["key"]}


def run_once(run: Path, judged: dict, *, factor: float, step: int, dry: bool = True, baseline_front: bool = True):
    from trimem.experiments.trace_v070_runtime import is_v070

    if not dry:
        raise SystemExit("latent_latency_pruner_v070 is an offline dry-run reporter; the live latency cap runs "
                         "inside staged_evaluate_v070 (fix round 1)")

    manifest = json.loads((run / "manifest.json").read_text())
    if not is_v070(manifest):
        raise SystemExit(f"{run}: not a v0.7.0 manifest (use tools.latent_latency_pruner)")
    if not manifest.get("pruning"):
        raise SystemExit(f"{run}: manifest has no pruning block (partial panels would not enter the GP)")
    init = set(manifest.get("initial_keys", []))
    db = sqlite3.connect(f"file:{run / 'ledger.sqlite'}?mode=ro", uri=True, timeout=30)
    try:
        rows = stock._rows(db, int(manifest["v070"]["sampling"]["size"]))
    finally:
        db.close()
    use_front = baseline_front  # what-if only: the live cap excludes NoMemory (latency_front false)
    nm = nomemory_row(run) if use_front else None
    if nm is not None:
        if nm["key"] in rows:
            raise AssertionError("NoMemory must never be in the main ledger")
        rows[NOMEMORY_ROW] = nm
        init = init | {NOMEMORY_ROW}
    decisions, fastest, front = stock.decide(rows, init, judged, factor=factor, step=step)
    if any(d["key"] == NOMEMORY_ROW for d in decisions):
        raise AssertionError("NoMemory was judged by the latency pruner")
    with (run / "latency_prune_report.jsonl").open("a") as log:
        for d in decisions:
            log.write(json.dumps({"at": time.time(), "protocol": PROTOCOL, "factor": factor,
                                  "fastest_front_latency": fastest, "front": front,
                                  "nomemory_in_front_reference": nm is not None, "dry_run": dry, **d}) + "\n")
    return decisions


def default_factor(run: Path) -> float:
    return FACTORS[json.loads((run / "manifest.json").read_text())["fixed"]["dataset"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", type=Path, required=True)
    ap.add_argument("--factor", type=float, default=None, help="default: per dataset (ALF 2, Sci 3, WebShop 2)")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="accepted for compatibility; always dry")
    ap.add_argument("--no-baseline-front", action="store_true")
    a = ap.parse_args(argv)
    judged = {str(r): {} for r in a.run_dir}
    while True:
        for run in a.run_dir:
            if (run / "manifest.json").exists() and (run / "ledger.sqlite").exists():
                factor = a.factor if a.factor is not None else default_factor(run)
                for d in run_once(run, judged[str(run)], factor=factor, step=a.step, dry=True,
                                  baseline_front=not a.no_baseline_front):
                    print(json.dumps({"run": str(run), **d}), flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
