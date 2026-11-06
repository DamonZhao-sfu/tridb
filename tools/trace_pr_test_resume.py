"""Resume a TRACE-PR TEST200 run from its frozen holdout manifest, with a priority order.

User 2026-11-04 (ALFWorld v0.6.0c): run the cross-encoder configurations before the
LLM-rank ones. The TEST selection is NOT recomputed: the frozen ``holdout/manifest.json``
(written by ``trace_pr_test_now``) fixes the configurations. Pending episodes are dispatched
in two passes: configurations whose ``rank_impl`` equals ``--first`` first, then the rest.
Claims left by a stopped driver are recovered first (``recover_claims``). Writes the same
``holdout/summary.json`` and ``status.json`` as ``trace_pr_test_now``.

Usage: python -m tools.trace_pr_test_resume --run-dir <run> --first cross-encoder
"""

import argparse
import fcntl
import json
import time
from pathlib import Path


def main():
    from tools.trace_joint_live import ROOT, evaluate, recover_claims, service_preflight, verify_artifacts, verify_memory
    from tools.trace_pr_test_now import wait_for_resources
    from trimem.experiments.trace_joint_plan_search import atomic_json, evaluate_pending, get_space
    from trimem.optimizer.joint_search_ledger import SearchLedger

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--first", default="cross-encoder")
    a = ap.parse_args()
    run = a.run_dir.resolve()
    m = json.loads((run / "manifest.json").read_text())
    out = run / "holdout"
    tm = json.loads((out / "manifest.json").read_text())
    space = get_space(m)
    status_path = run / "status.json"
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
                                  "resumed": True, "priority": a.first})
        print(json.dumps({"recover_claims": recover_claims(out, tm)}))
        with SearchLedger(out / "ledger.sqlite", tm) as test:
            pending = test.pending()
        first = [p for p in pending if p["config"].get("rank_impl") == a.first]
        rest = [p for p in pending if p["config"].get("rank_impl") != a.first]
        print(json.dumps({"pending": len(pending), "first": len(first), "rest": len(rest)}))
        for part in (first, rest):
            if part:
                evaluate_pending(out, tm, space, part, evaluate)
        with SearchLedger(out / "ledger.sqlite", tm) as test:
            if test.pending():
                raise RuntimeError("Uncertain TEST claims require evidence recovery")
            complete = test.complete_configs()
        metrics = ("raw_score", "total_tokens", "latency_seconds", "agent_steps")
        summary = {
            "status": "test_complete",
            "configurations": len(complete),
            "tasks_per_configuration": 200,
            "episodes": 200 * len(complete),
            "train_truncated": tm.get("train_truncated"),
            "results": [
                {"key": e["key"], "config": e["config"],
                 **{f: sum(r[f] for r in e["receipts"]) / 200 for f in metrics
                    if all(r.get(f) is not None for r in e["receipts"])}}
                for e in complete
            ],
        }
        atomic_json(out / "summary.json", summary)
        atomic_json(status_path, {"status": "test_complete", "at": time.time(),
                                  "train_truncated": tm.get("train_truncated") is not None,
                                  "test_configs": len(complete), "test_episodes": 200 * len(complete)})


if __name__ == "__main__":
    main()
