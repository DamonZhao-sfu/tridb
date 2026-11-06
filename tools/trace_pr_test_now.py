"""Stop a TRACE-PR TRAIN search early and run TEST200 on its complete configurations.

User 2026-11-04 (ALFWorld v0.6.0c): pause TRAIN at the current configurations and start
TEST. ``run_holdout`` requires the full TRAIN budget, so this tool repeats its steps on the
complete configurations only (partial configurations keep their receipts and are not
candidates). TEST candidates (user 2026-11-04, fix 1):
  TRAIN sample-mean Pareto  U  GP posterior-mean Pareto  U  {GP posterior-mean OneConfig}
All choices use TRAIN only; TEST outcomes are written to the separate holdout ledger.
The TRAIN ledger and the hashed live files are not modified.

Usage: python -m tools.trace_pr_test_now --run-dir results/trace_pr_alfworld_v060_20261103c
"""

import argparse
import fcntl
import json
import time
from pathlib import Path

from tools.trace_pr_final_select import fit_posterior, pareto


def wait_for_resources(run_dir, lease, status_path):
    from tools.trace_joint_live import eligible_resources, resource_snapshot
    from trimem.experiments.trace_joint_plan_search import atomic_json

    stable = 0
    while True:
        resources = resource_snapshot()
        try:
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        ready = acquired and eligible_resources(resources["available_memory_bytes"], resources["gpu_utilization"])
        stable = stable + 1 if ready else 0
        if stable >= 3:
            return
        if acquired:
            fcntl.flock(lease, fcntl.LOCK_UN)
        atomic_json(status_path, {"status": "waiting_resources_test_now", "at": time.time(), **resources})
        time.sleep(30)


def select(run_dir):
    rows, meta = fit_posterior(run_dir, allow_partial=True)
    full = [i for i, r in enumerate(rows) if not r["pruned"]]  # pruned configs are never TEST candidates
    sample = [full[j] for j in pareto(
        [(rows[i]["train_mean"]["quality"], -rows[i]["train_mean"]["tokens"], -rows[i]["train_mean"]["latency"]) for i in full])]
    gp = [full[j] for j in pareto(
        [(rows[i]["posterior_mean"]["quality"], -rows[i]["posterior_mean"]["tokens"], -rows[i]["posterior_mean"]["latency"]) for i in full])]
    one_gp = max(full, key=lambda i: rows[i]["posterior_mean"]["quality"])
    one_raw = max(full, key=lambda i: rows[i]["train_mean"]["quality"])
    chosen = sorted(set(sample) | set(gp) | {one_gp})
    return rows, meta, chosen, {
        "sample_mean_pareto": [rows[i]["key"] for i in sample],
        "gp_pareto": [rows[i]["key"] for i in gp],
        "one_config_gp": rows[one_gp]["key"],
        "one_config_sample_mean": rows[one_raw]["key"],
    }


def main():
    from tools.trace_joint_live import (
        ROOT,
        evaluate,
        recover_claims,
        service_preflight,
        verify_artifacts,
        verify_memory,
    )
    from tools.fixed_test_panel_guard import validate_ids
    from trimem.experiments.trace_joint_plan_search import atomic_json, evaluate_pending, get_space
    from trimem.optimizer.joint_search_ledger import SearchLedger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    m = json.loads((run / "manifest.json").read_text())
    h = m["holdout"]
    validate_ids(m["fixed"]["dataset"], h["tasks"])
    space = get_space(m)
    with SearchLedger(run / "ledger.sqlite", m) as train:
        n_complete = len(train.complete_configs())
        n_pending = len(train.pending())
    import sqlite3

    with sqlite3.connect(f"file:{run / 'ledger.sqlite'}?mode=ro", uri=True) as db:
        n_reserved = db.execute("SELECT count(*) FROM configs").fetchone()[0]
    rows, meta, chosen, sel = select(run)
    selected = [rows[i]["config"] for i in chosen]
    selection = {
        "policy": "train_pareto_union_gp_posterior (user 2026-11-04)",
        "train_protocol": m["protocol"],
        "train_truncated": None
        if n_complete == m["budget"] and n_pending == 0
        else {
            "reason": "user 2026-11-04: pause TRAIN, run TEST on the current configurations",
            "complete_configs": n_complete,
            "budget": m["budget"],
            "pending_episodes_left": n_pending,
            "reserved_configs": n_reserved,
        },
        "train_keys": [r["key"] for r in rows],
        **sel,
        "selected_keys": [rows[i]["key"] for i in chosen],
        "configs": selected,
        "train_means": [rows[i]["train_mean"] for i in chosen],
        "posterior_means": [rows[i]["posterior_mean"] for i in chosen],
        "gp": meta,
    }
    print(json.dumps({k: selection[k] for k in ("train_truncated", "selected_keys", "one_config_gp",
                                                  "one_config_sample_mean")}, indent=1))
    if args.select_only:
        return
    out = run / "holdout"
    out.mkdir(exist_ok=True)
    status_path = run / "status.json"
    test_manifest = {
        **m,
        "tasks": h["tasks"],
        "budget": len(selected),
        "protocol": m["protocol"] + "-test",
        "evaluation_split": "TEST",
        "selected_keys": selection["selected_keys"],
        "train_truncated": selection["train_truncated"],
    }
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
                                  "train_configs": n_complete,
                                  "train_truncated": selection["train_truncated"] is not None})
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            test.reserve(space, selected)
        atomic_json(out / "selection.json", selection)
        atomic_json(out / "manifest.json", test_manifest)
        recover_claims(out, test_manifest)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            pending = test.pending()
        evaluate_pending(out, test_manifest, space, pending, evaluate)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            if test.pending():
                raise RuntimeError("Uncertain TEST claims require evidence recovery")
            complete = test.complete_configs()
        metrics = ("raw_score", "total_tokens", "latency_seconds", "agent_steps")
        summary = {
            "status": "test_complete",
            "configurations": len(complete),
            "tasks_per_configuration": 200,
            "episodes": 200 * len(complete),
            "train_truncated": selection["train_truncated"],
            "results": [
                {"key": e["key"], "config": e["config"],
                 **{f: sum(r[f] for r in e["receipts"]) / 200 for f in metrics
                    if all(r.get(f) is not None for r in e["receipts"])}}
                for e in complete
            ],
        }
        atomic_json(out / "summary.json", summary)
        atomic_json(status_path, {"status": "test_complete", "at": time.time(), "train_configs": n_complete,
                                  "train_truncated": selection["train_truncated"] is not None, "test_configs": len(complete),
                                  "test_episodes": 200 * len(complete)})


if __name__ == "__main__":
    main()
