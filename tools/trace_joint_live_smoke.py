"""Phase-0 real-environment smoke: 2 configs x 8 TRAIN tasks per dataset at c16.

Runs the real evaluator (``tools.trace_joint_live.evaluate``) against the real
services, outside any campaign ledger, and validates every receipt with the same
``check_receipt`` used before the ledger marks an episode done. Covers both
plan families: VGR+Rank (filter_order none) and P2+Rank (filter_order after).

It waits for the shared evaluation lease and the resource gate (>= 32 GiB free,
both GPUs <= 20% for three checks) so it never runs beside another experiment.
Outputs: <out>/<dataset>/report.json and <out>/smoke_report.json.

Usage: scripts/in_eval_slice.sh .venv-optimizer/bin/python -m tools.trace_joint_live_smoke \\
           --prepared results/trace_pr_three_dataset_prepare/live_smoke --out <dir>
"""

import argparse
import fcntl
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools import trace_joint_live as live
from trimem.experiments.trace_joint_plan_search import atomic_json, check_receipt, get_space

DATASETS = ("sciworld", "alfworld", "webshop")


def wait_for_lease(lease, status_path):
    stable = 0
    while True:
        snapshot = live.resource_snapshot()
        try:
            # Shared, like the other evaluation queues; the idle-GPU gate prevents co-running.
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            acquired = False
        ready = acquired and live.eligible_resources(
            snapshot["available_memory_bytes"], snapshot["gpu_utilization"]
        )
        stable = stable + 1 if ready else 0
        if stable >= 3:
            return
        if acquired:
            fcntl.flock(lease, fcntl.LOCK_UN)
        atomic_json(status_path, {"status": "waiting_resources", "at": time.time(),
                                  "lease": acquired, **snapshot})
        time.sleep(30)


def smoke_dataset(run_dir, out):
    m = json.loads((run_dir / "manifest.json").read_text())
    live.verify_artifacts(m)
    live.verify_memory(m)
    live.service_preflight(m)
    space = get_space(m)
    # Engine extension v0.1.0: four configurations that together cover every new operator.
    configs = [
        space.default(filter_order="none", graph_impl="MCTS", rank_impl="cross-encoder",
                      retrieval_unit="window", graph_direction="Both", seed_count=16, fanout=8, returned_k=5),
        space.default(filter_order="after", graph_impl="beam", beam_width=2, rank_impl="LLM",
                      retrieval_unit="actions", graph_direction="Backward", seed_count=16, fanout=8, returned_k=5),
        space.default(filter_order="before", graph_impl="best-first", rank_impl="ColBERT",
                      injection_timing="periodic", retrieval_interval=5, graph_direction="Forward",
                      seed_count=8, fanout=8, returned_k=5),
        space.default(filter_order="none", graph_impl="BFS", injection_timing="periodic",
                      retrieval_interval=10, retrieval_unit="window", seed_count=16, fanout=8, returned_k=5),
    ]
    jobs = [(c, t) for c in configs for t in m["tasks"][:4]]  # 16 episodes at c16
    writes_before = live.memory_writes(m["fixed"]["memory_database"])
    began = time.monotonic()

    def run(job):
        config, task = job
        item = {"key": space.key(config), "task_id": task}
        try:
            receipt = live.evaluate(config, task, m)
            check_receipt(receipt, item, m, space)
            return {"ok": True, "plan": space.plan(config).describe(), **{
                k: receipt[k] for k in ("task_id", "raw_score", "total_tokens",
                                        "latency_seconds", "agent_steps")}}
        except Exception as exc:  # report every failure; the smoke must not hide one
            return {"ok": False, "task_id": task, "plan": space.plan(config).describe(),
                    "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(run, jobs))
    ok = [r for r in rows if r["ok"]]
    stats = {}
    for base in m["live"]["environment_bases"]:
        stats[base] = live.get_json(base + "/managed_stats")
    report = {
        "dataset": m["fixed"]["dataset"],
        "episodes": len(rows),
        "passed": len(ok),
        "wall_seconds": time.monotonic() - began,
        "mean_latency_seconds": statistics.mean(r["latency_seconds"] for r in ok) if ok else None,
        "mean_agent_steps": statistics.mean(r["agent_steps"] for r in ok) if ok else None,
        "rows": rows,
        "env_after": stats,
        "memory_writes": [writes_before, live.memory_writes(m["fixed"]["memory_database"])],
    }
    if report["memory_writes"][0] != report["memory_writes"][1]:
        report["passed"] = 0  # a frozen clone must never be written
    atomic_json(out / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Use a new smoke output directory")
    a.out.mkdir(parents=True)
    status = a.out / "status.json"
    with (live.ROOT / "results/pool_evaluation_queue.lock").open("a") as lease:
        wait_for_lease(lease, status)
        atomic_json(status, {"status": "running", "at": time.time()})
        reports = {}
        for ds in a.datasets:
            (a.out / ds).mkdir()
            reports[ds] = smoke_dataset(a.prepared / ds, a.out / ds)
            summary = {k: v for k, v in reports[ds].items() if k not in ("rows", "env_after")}
            atomic_json(status, {"status": "running", "at": time.time(), "done": {ds: summary}})
    final = {ds: {k: v for k, v in r.items() if k not in ("rows", "env_after")}
             for ds, r in reports.items()}
    final["all_passed"] = all(r["passed"] == r["episodes"] for r in reports.values())
    atomic_json(a.out / "smoke_report.json", final)
    atomic_json(status, {"status": "complete", "at": time.time(), "all_passed": final["all_passed"]})


if __name__ == "__main__":
    main()
