"""Paired runtime calibration: replay a historical c16 configuration on its own tasks.

For one dataset at a time: take N episodes of one historical configuration from
opt.session_retrieval, rebuild the same joint configuration, run the same tasks
with the current live evaluator at c16 (16 concurrent), and compare end-to-end
latency task by task. Criterion (user rule 2026-10-21): mean ratio <= 1.05 and
p90 ratio <= 1.10. Calibration episodes go to their own directory and protocol;
they never enter any campaign ledger.

Usage: python -m tools.trace_joint_calibration --dataset alfworld \\
    --prepared results/trace_pr_three_dataset_20261103e/smoke_prepared/alfworld \\
    --experiment 'alfworld:e2holdout:b29f00a4af6d1721:validation%' --config-hash <hash> --out <dir>
"""

import argparse
import fcntl
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

from tools import trace_joint_live as live
from tools.trace_joint_live_smoke import wait_for_lease
from trimem.experiments.trace_joint_plan_search import atomic_json
from trimem.optimizer.plan_schema import JointSpace

OPT = "host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa options='-c default_transaction_read_only=on'"
PLAN_TO_FILTER = {"VGR+Rank": "none", "P1+Rank": "before", "P2+Rank": "after"}


def reference(dataset, experiment, config_hash, n):
    with psycopg.connect(OPT) as conn:
        rows = conn.execute(
            "SELECT query_id, e2e_ms, retrieval_ms, total_tokens, quality, "
            "(generation->>'agent_steps')::float, config, protocol->'session_v2' "
            "FROM opt.session_retrieval WHERE dataset=%s AND experiment_id LIKE %s AND config_hash=%s "
            "AND status='evaluated' AND e2e_ms IS NOT NULL ORDER BY query_id",
            (dataset, experiment, config_hash),
        ).fetchall()
    seen, out = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    if len(out) < n:
        raise ValueError(f"Only {len(out)} reference episodes")
    return out[:n]


def joint_config(space, config, v2):
    c = {
        "filter_order": PLAN_TO_FILTER[config["plan"]],
        "seed_count": config["seed_count"],
        "depth": config["depth"],
        "fanout": config["fanout"],
        "returned_k": config["top_k"],
        "trajectory_category": "Success" if v2["outcome_policy"] == "success_only" else "All",
    }
    if c["filter_order"] == "before":
        c["sem_threshold_pre"] = v2["semfilter_threshold"]
    if c["filter_order"] == "after":
        c["sem_threshold_post"] = v2["semfilter_threshold"]
    if (config.get("sim_gate"), v2.get("memory_form"), v2.get("pool_cap"), v2.get("expand_limit"), tuple(config["edges"])) != (
        None, "raw", 256, 4096, ("similar", "similar_up")
    ):
        raise ValueError("Reference configuration is outside the replayable space")
    return space.canonicalize(c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--config-hash", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Use a new calibration directory")
    a.out.mkdir(parents=True)
    base = json.loads((a.prepared / "manifest.json").read_text())
    if base["fixed"]["dataset"] != a.dataset:
        raise SystemExit("Prepared run is for another dataset")
    space = JointSpace(base["space"]["domains"], execution_fingerprint=base["space"]["execution_fingerprint"])
    refs = reference(a.dataset, a.experiment, a.config_hash, a.n)
    config = joint_config(space, refs[0][6], refs[0][7])
    pool = json.loads((live.ROOT / json.loads((a.prepared / "preparation.json").read_text())["pool"]).read_text())
    items = {r["item_id"]: r for r in pool["items"]}
    panel_items = [items[r[0]] for r in refs]
    panel = {"dataset": a.dataset, "splits": sorted({r["split"] for r in panel_items}),
             "selection": "calibration replay of historical episodes", "items": panel_items}
    atomic_json(a.out / "calibration_panel.json", panel)
    m = json.loads(json.dumps(base))
    m["tasks"] = [r[0] for r in refs]
    m["protocol"] = a.out.name + "-calibration"
    m["live"].update(run_dir=str(a.out.resolve()), panel_path=str((a.out / "calibration_panel.json").resolve()),
                     panel_sha256=live.sha(a.out / "calibration_panel.json"))
    status = a.out / "status.json"
    with (live.ROOT / "results/pool_evaluation_queue.lock").open("a") as lease:
        wait_for_lease(lease, status)
        live.verify_memory(m)
        live.service_preflight(m)
        atomic_json(status, {"status": "running", "at": time.time(), "episodes": len(refs)})
        began = time.monotonic()

        def run(ref):
            try:
                r = live.evaluate(config, ref[0], m)
                return {"task": ref[0], "ok": True, "latency": r["latency_seconds"],
                        "retrieval": r["retrieval_latency_seconds"], "steps": r["agent_steps"],
                        "tokens": r["total_tokens"], "score": r["raw_score"]}
            except Exception as exc:
                return {"task": ref[0], "ok": False, "error": repr(exc)}

        with ThreadPoolExecutor(max_workers=16) as pool_:
            rows = list(pool_.map(run, refs))
        wall = time.monotonic() - began
    paired = []
    for ref, row in zip(refs, rows):
        if row["ok"]:
            hist = {"latency": ref[1] / 1000, "retrieval": ref[2] / 1000, "tokens": ref[3], "steps": ref[5]}
            paired.append({**row, "hist": hist, "ratio": row["latency"] / hist["latency"],
                           "per_step_ratio": (row["latency"] / max(row["steps"], 1)) / (hist["latency"] / max(hist["steps"] or 1, 1))})
    ratios = sorted(x["ratio"] for x in paired)
    mean = lambda xs: statistics.mean(xs) if xs else None
    report = {
        "dataset": a.dataset, "reference": {"experiment": a.experiment, "config_hash": a.config_hash},
        "config": config, "episodes": len(rows), "ok": len(paired), "wall_seconds": wall,
        "mean_latency_now": mean([x["latency"] for x in paired]),
        "mean_latency_hist": mean([x["hist"]["latency"] for x in paired]),
        "mean_ratio_of_means": (mean([x["latency"] for x in paired]) / mean([x["hist"]["latency"] for x in paired])) if paired else None,
        "p90_now": statistics.quantiles([x["latency"] for x in paired], n=10)[-1] if len(paired) > 2 else None,
        "p90_hist": statistics.quantiles([x["hist"]["latency"] for x in paired], n=10)[-1] if len(paired) > 2 else None,
        "median_pair_ratio": statistics.median(ratios) if ratios else None,
        "mean_steps_now": mean([x["steps"] for x in paired]), "mean_steps_hist": mean([x["hist"]["steps"] for x in paired]),
        "mean_retrieval_now": mean([x["retrieval"] for x in paired]), "mean_retrieval_hist": mean([x["hist"]["retrieval"] for x in paired]),
        "mean_per_step_ratio": mean([x["per_step_ratio"] for x in paired]),
        "rows": rows,
    }
    if report["p90_now"] and report["p90_hist"]:
        report["p90_ratio"] = report["p90_now"] / report["p90_hist"]
        report["passes_rule"] = report["mean_ratio_of_means"] <= 1.05 and report["p90_ratio"] <= 1.10
    atomic_json(a.out / "calibration_report.json", report)
    atomic_json(status, {"status": "complete", "at": time.time()})
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
