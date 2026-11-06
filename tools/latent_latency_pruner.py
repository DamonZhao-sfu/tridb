"""Latency-cap pruning for the TRACE-latent arms (user 2026-11-05).

Rule (user): at every 20-task checkpoint of a BO configuration, prune it when its mean episode latency so far is
more than 2x the latency of the FASTEST configuration on the current TRAIN Pareto front.

* Current Pareto front = sample-mean Pareto set (max quality, min tokens, min latency) of the configurations that
  have a complete 100-task panel (never-pruned, all episodes done), initialization included.
* Only BO configurations are judged (initialization is protected) and only when their done count has reached a
  new multiple of 20 (20, 40, 60, 80) since the last decision.
* Pruning uses the ledger's own prune(): ready/failed episodes -> pruned; claimed episodes finish normally; the
  finished episodes enter the GP as a partial panel (manifest pruning block, allow_partial).
* No hashed file is touched; this runs as a separate process next to the live driver. Every decision is appended
  to <run>/latency_prune_log.jsonl.

Usage: python -m tools.latent_latency_pruner --run-dir R [--run-dir R2 ...] [--factor 2.0] [--step 20]
       [--interval 30] [--once] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path

PROTOCOL = "latency-cap-v1"


def _rows(db, tasks: int):
    out = {}
    for key, body in db.execute("SELECT key, body FROM configs"):
        states = dict(db.execute("SELECT state, count(*) FROM episodes WHERE config=? GROUP BY state", (key,)))
        done = [json.loads(r) for (r,) in db.execute(
            "SELECT receipt FROM episodes WHERE config=? AND state='done' AND receipt IS NOT NULL", (key,))]
        out[key] = {
            "states": states,
            "n": len(done),
            "q": statistics.mean(x["raw_score"] for x in done) if done else None,
            "tok": statistics.mean(x["total_tokens"] for x in done) if done else None,
            "lat": statistics.mean(x["latency_seconds"] for x in done) if done else None,
            "complete": len(done) == tasks and not states.get("pruned"),
        }
    return out


def front_fastest(rows: dict) -> tuple[float | None, list[str]]:
    full = {k: r for k, r in rows.items() if r["complete"]}
    front = []
    for k, a in full.items():
        dominated = any(
            b["q"] >= a["q"] and b["tok"] <= a["tok"] and b["lat"] <= a["lat"]
            and (b["q"] > a["q"] or b["tok"] < a["tok"] or b["lat"] < a["lat"])
            for j, b in full.items() if j != k
        )
        if not dominated:
            front.append(k)
    if not front:
        return None, []
    return min(full[k]["lat"] for k in front), front


def decide(rows: dict, init: set[str], judged: dict[str, int], *, factor: float, step: int):
    fastest, front = front_fastest(rows)
    decisions = []
    if fastest is None:
        return decisions, fastest, front
    for k, r in rows.items():
        if k in init or r["complete"] or r["states"].get("pruned") or r["n"] < step:
            continue
        checkpoint = (r["n"] // step) * step
        if checkpoint <= judged.get(k, 0):
            continue
        judged[k] = checkpoint
        decisions.append({"key": k, "checkpoint": checkpoint, "n": r["n"], "latency": r["lat"],
                          "cap": factor * fastest, "prune": r["lat"] > factor * fastest})
    return decisions, fastest, front


def run_once(run: Path, judged: dict, *, factor: float, step: int, dry: bool) -> list[dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    if not manifest.get("pruning"):
        raise SystemExit(f"{run}: manifest has no pruning block (partial panels would not enter the GP)")
    init = set(manifest.get("initial_keys", []))
    db = sqlite3.connect(f"file:{run / 'ledger.sqlite'}?mode=ro", uri=True, timeout=30)
    try:
        rows = _rows(db, len(manifest["tasks"]))
    finally:
        db.close()
    decisions, fastest, front = decide(rows, init, judged, factor=factor, step=step)
    to_prune = {d["key"]: d for d in decisions if d["prune"]}
    if to_prune and not dry:
        from trimem.optimizer.joint_search_ledger import SearchLedger

        with SearchLedger(run / "ledger.sqlite", manifest) as led:
            skipped = led.prune_many({
                k: f"{PROTOCOL} (user 2026-11-05): mean latency {d['latency']:.1f}s > {factor}x fastest "
                   f"Pareto-front config {fastest:.1f}s at {d['checkpoint']} tasks"
                for k, d in to_prune.items()
            })
        for k, n in skipped.items():
            to_prune[k]["skipped_episodes"] = n
    with (run / "latency_prune_log.jsonl").open("a") as log:
        for d in decisions:
            log.write(json.dumps({"at": time.time(), "protocol": PROTOCOL, "factor": factor,
                                  "fastest_front_latency": fastest, "front": front, "dry_run": dry, **d}) + "\n")
    return decisions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", type=Path, required=True)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    judged = {str(r): {} for r in a.run_dir}
    while True:
        for run in a.run_dir:
            if (run / "manifest.json").exists() and (run / "ledger.sqlite").exists():
                for d in run_once(run, judged[str(run)], factor=a.factor, step=a.step, dry=a.dry_run):
                    print(json.dumps({"run": str(run), **d}), flush=True)
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
