"""TRAIN / TEST candidates of a TRACE v0.7.0 run: BO configurations U NoMemory, each on its OWN tasks.

``candidates(run, split)`` -> one row per candidate with its own task list, per-candidate sample means and SEs
(unpaired; Amendment A.4). ``full_only`` keeps configurations with their full own sample and never pruned;
NoMemory is a candidate when ``include_baselines`` and its ledger is complete. ``pareto(points)`` = indices of the
sample-mean Pareto set (quality max, tokens min, latency min; weak domination with one strict). Used by the
holdout selection, the latency pruner front, the router candidates and ``export`` (HV / plot input).

    python -m trimem.experiments.trace_v070_candidates RUN [--split TRAIN|TEST] [--out points.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from trimem.optimizer.joint_search_ledger import SearchLedger

METRICS = ("raw_score", "total_tokens", "latency_seconds", "agent_steps")


def row_of(key, config, receipts, tasks, *, pruned=False, source="bo", baseline=None) -> dict:
    by = {str(r["task_id"]): r for r in receipts}
    done = [by[t] for t in tasks if t in by]
    out = {"key": key, "config": config, "source": source, "baseline": baseline, "tasks": list(tasks),
           "n": len(done), "full": len(done) == len(tasks) and not pruned, "pruned": bool(pruned)}
    for f in METRICS:
        vals = [float(r[f]) for r in done if r.get(f) is not None]
        if vals and len(vals) == len(done):
            out[f] = statistics.mean(vals)
            out[f + "_se"] = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    if done:
        out["raw_score_pos"] = statistics.mean(max(float(r["raw_score"]), 0.0) for r in done)
    return out


def brane_acc_cost(row, dataset: str) -> tuple[float, float]:
    """BRANE accuracy / cost of one candidate on its own sample (user 2026-11-05): acc = mean quality
    (ScienceWorld max(score, 0) / 100; ALFWorld / WebShop score); cost = 0.5 ln(mean tokens) + 0.5 ln(mean
    latency ms) (``brane.config_cost``)."""
    acc = row["raw_score_pos"] / 100.0 if dataset == "sciworld" else row["raw_score"]
    return acc, 0.5 * math.log(row["total_tokens"]) + 0.5 * math.log(row["latency_seconds"] * 1000.0)


def fuzzy_set(rows, dataset: str) -> list[int]:
    """BRANE fuzzy Pareto set (TAU_ACC 0.02, TAU_COST 0.10) over the given candidates (indices into rows)."""
    import numpy as np

    from trimem.optimizer.query_router.brane import TAU_ACC, TAU_COST, fuzzy_pareto

    if not rows:
        return []
    ac = np.array([brane_acc_cost(r, dataset) for r in rows])
    return fuzzy_pareto(ac[:, 0], ac[:, 1], tau_acc=TAU_ACC, tau_cost=TAU_COST)


def _bo_rows(run: Path, split: str) -> list[dict]:
    path = run / ("holdout" if split == "TEST" else "") / "ledger.sqlite"
    mpath = run / ("holdout" if split == "TEST" else "") / "manifest.json"
    if not path.exists():
        return []
    manifest = json.loads(mpath.read_text())
    rows = []
    with SearchLedger(path, manifest) as led:
        samples = {}
        if split == "TRAIN":
            from .trace_v070_ledger import own_tasks

            samples = {k: own_tasks(manifest, k) for (k,) in led.db.execute("SELECT key FROM configs")}
        pruned = {k for (k,) in led.db.execute("SELECT DISTINCT config FROM episodes WHERE state='pruned'")}
        for key, body in led.db.execute("SELECT key, body FROM configs ORDER BY created, key").fetchall():
            tasks = samples.get(key) or [str(t) for t in manifest["tasks"]]
            rows.append(row_of(key, json.loads(body), led.receipts(key), tasks, pruned=key in pruned))
    return rows


def candidates(run, split: str = "TRAIN", *, include_baselines: bool = True, full_only: bool = True) -> list[dict]:
    from .trace_v070_baseline import baseline_entry

    run = Path(run)
    rows = _bo_rows(run, split)
    if include_baselines:
        e = baseline_entry(run, split)
        if e is not None:
            rows.append(row_of(e["key"], e["config"], e["receipts"], e["tasks"], source="baseline",
                               baseline="NoMemory"))
    if full_only:
        rows = [r for r in rows if r["full"]]
    return rows


def point(row) -> tuple[float, float, float]:
    return (row["raw_score"], row["total_tokens"], row["latency_seconds"])


def pareto(points) -> list[int]:
    """Indices of the non-dominated points (quality max, tokens min, latency min)."""
    pts = [tuple(map(float, p)) for p in points]

    def dominates(a, b):
        return a[0] >= b[0] and a[1] <= b[1] and a[2] <= b[2] and a != b

    return [i for i, p in enumerate(pts) if not any(dominates(q, p) for j, q in enumerate(pts) if j != i)]


def export(run, split: str = "TRAIN") -> dict:
    """HV / plot input: per-candidate own-task means (unpaired) and the Pareto flags."""
    rows = candidates(run, split)
    dataset = json.loads((Path(run) / "manifest.json").read_text())["fixed"]["dataset"]
    front, fuzzy = set(pareto([point(r) for r in rows])), set(fuzzy_set(rows, dataset))
    return {"run": str(run), "split": split, "rule": "per-candidate own-task sample means (unpaired)",
            "points": [{k: v for k, v in r.items() if k != "tasks"} | {"pareto": i in front, "brane_fuzzy": i in fuzzy}
                       for i, r in enumerate(rows)]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--split", choices=("TRAIN", "TEST"), default="TRAIN")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    body = export(a.run, a.split)
    text = json.dumps(body, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text if not a.out else json.dumps({"points": len(body["points"]), "out": str(a.out)}))


if __name__ == "__main__":
    main()
