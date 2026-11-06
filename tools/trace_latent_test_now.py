"""Stop a TRACE-latent TRAIN search early and run TEST200 on its current TRAIN Pareto set (user 2026-11-05).

User 2026-11-05 (ScienceWorld): pause TRAIN now, TEST the current TRAIN sample-mean Pareto configurations.
Same selection as the automatic ``tools.trace_latent_live.run_pruned_holdout`` (candidates = never-pruned
configurations with the full TRAIN panel; sample-mean Pareto on (quality, -tokens, -latency) over them), except
that TRAIN need not be complete: in-progress configurations (partial panel, not pruned) are not candidates and are
recorded under ``train_truncated``. The TRAIN ledger and every hashed live file are not modified; TEST outcomes go
to the separate holdout ledger with the stock evaluation path (resource gate, lease, recover_claims, evaluate).

Usage: python -m tools.trace_latent_test_now --run-dir results/trace_latent_sciworld_20261104 [--select-only]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import time
from pathlib import Path

from tools.trace_pr_test_now import wait_for_resources


def select(run: Path, m: dict):
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.optimizer.joint_search_ledger import SearchLedger
    from trimem.optimizer.plan_surrogate import aggregate_panel

    space = driver.get_space(m)
    tasks = {str(t) for t in m["tasks"]}
    with SearchLedger(run / "ledger.sqlite", m) as train:
        entries = train.complete_configs()
        pending = len(train.pending())
    with sqlite3.connect(f"file:{run / 'ledger.sqlite'}?mode=ro", uri=True) as db:
        reserved = [k for (k,) in db.execute("SELECT key FROM configs")]
    full = [e for e in entries if not e["pruned"] and {str(r["task_id"]) for r in e["receipts"]} == tasks]
    obs = [aggregate_panel(space, e["config"], e["receipts"], task_ids=m["tasks"], protocol=m["protocol"],
                           allow_synthetic=False, allow_partial=False) for e in full]
    points = [(o.mean[0], -o.mean[1], -o.mean[2]) for o in obs]
    idx = [i for i, p in enumerate(points)
           if not any(all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p)) for q in points)]
    done_keys = {e["key"] for e in entries}
    selection = {
        "policy": m["holdout"]["selection"],
        "train_protocol": m["protocol"],
        "train_keys": [e["key"] for e in entries],
        "candidate_keys": [e["key"] for e in full],
        "excluded_pruned": [e["key"] for e in entries if e not in full],
        "candidate_rule": "never pruned and full TRAIN panel (user 2026-11-04)",
        "train_truncated": {
            "reason": "user 2026-11-05: pause ScienceWorld TRAIN, TEST the current TRAIN Pareto set",
            "complete_candidates": len(full),
            "reserved_configs": len(reserved),
            "in_progress_configs": [k for k in reserved if k not in done_keys],
            "pending_episodes_left": pending,
        },
        "configs": [obs[i].config for i in idx],
        "train_means": [obs[i].mean for i in idx],
    }
    return space, selection


def main() -> None:
    from tools.fixed_test_panel_guard import validate_ids
    from tools.trace_joint_live import (
        ROOT,
        evaluate,
        recover_claims,
        service_preflight,
        verify_artifacts,
        verify_memory,
    )
    from trimem.experiments.trace_joint_plan_search import atomic_json, evaluate_pending
    from trimem.optimizer.joint_search_ledger import SearchLedger

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    run = a.run_dir.resolve()
    m = json.loads((run / "manifest.json").read_text())
    h = m["holdout"]
    validate_ids(m["fixed"]["dataset"], h["tasks"])
    space, selection = select(run, m)
    selected = selection["configs"]
    keys = [space.key(c) for c in selected]
    print(json.dumps({"selected_keys": keys, "train_means": selection["train_means"],
                      "train_truncated": selection["train_truncated"]}, indent=1, default=str))
    if a.select_only:
        return
    out = run / "holdout"
    out.mkdir(exist_ok=True)
    status_path = run / "status.json"
    test_manifest = {**m, "tasks": h["tasks"], "budget": len(selected), "protocol": m["protocol"] + "-test",
                     "evaluation_split": "TEST", "selected_keys": keys,
                     "train_truncated": selection["train_truncated"]}
    verify_artifacts(m)
    verify_memory(m)
    service_preflight(m)
    with (
        (run / "queue.lock").open("a") as own,
        (ROOT / "results/pool_evaluation_queue.lock").open("a") as lease,
        (out / "driver.lock").open("a") as lock,
    ):
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait_for_resources(run, lease, status_path)
        atomic_json(status_path, {"status": "testing_train_frontier", "at": time.time(),
                                  "train_truncated": True, "test_configs": len(selected)})
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            test.reserve(space, selected)
        atomic_json(out / "selection.json", {**selection, "selected_keys": keys})
        atomic_json(out / "manifest.json", test_manifest)
        recover_claims(out, test_manifest)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            pending = test.pending()
        evaluate_pending(out, test_manifest, space, pending, evaluate)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            if test.pending():
                raise RuntimeError("Uncertain TEST claims require evidence recovery")
            complete = test.complete_configs()
        n = len(h["tasks"])
        metrics = ("raw_score", "total_tokens", "latency_seconds", "agent_steps")
        summary = {
            "status": "test_complete", "configurations": len(complete), "tasks_per_configuration": n,
            "episodes": n * len(complete), "train_truncated": selection["train_truncated"],
            "results": [{"key": e["key"], "config": e["config"],
                         **{f: sum(r[f] for r in e["receipts"]) / n for f in metrics
                            if all(r.get(f) is not None for r in e["receipts"])}} for e in complete],
        }
        atomic_json(out / "summary.json", summary)
        atomic_json(status_path, {"status": "test_complete", "at": time.time(), "train_truncated": True,
                                  "test_configs": len(complete), "test_episodes": n * len(complete)})


if __name__ == "__main__":
    main()
