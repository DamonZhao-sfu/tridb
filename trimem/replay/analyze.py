"""Aggregate a replay run: quality and per-phase cost per (task, arm), plus context identity.

PYTHONPATH=. python -m trimem.replay.analyze results/trimem_v0.1.0/replay/run1
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def _g(d: dict, *path, default=0.0):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d if d is not None else default


def load(run: Path) -> dict[tuple[str, str], list[dict]]:
    cells: dict[tuple[str, str], list[dict]] = {}
    for f in sorted(run.glob("*.jsonl")):
        recs = [json.loads(line) for line in f.open() if line.strip()]
        if recs:
            cells[(recs[0]["task"], recs[0]["arm"])] = recs
    return cells


def cell_summary(recs: list[dict], seed: float) -> dict:
    s = defaultdict(float)
    for r in recs:
        s["construction_ms"] += (
            _g(r, "construction", "embed", "ms")
            + _g(r, "construction", "derive", "ms")
            + _g(r, "construction", "insert_ms")
        )
        s["construction_embed_tokens"] += _g(
            r, "construction", "embed", "prompt_tokens"
        )
        s["construction_llm_in"] += _g(r, "construction", "derive", "prompt_tokens")
        s["construction_llm_out"] += _g(
            r, "construction", "derive", "completion_tokens"
        )
        s["construction_llm_calls"] += _g(r, "construction", "derive", "calls")
        s["retrieval_ms"] += _g(r, "retrieval", "ms") + _g(
            r, "retrieval", "deferred", "ms"
        )
        s["retrieval_sem_in"] += _g(r, "retrieval", "sem_prompt_tokens")
        s["retrieval_sem_out"] += _g(r, "retrieval", "sem_completion_tokens")
        s["retrieval_deferred_in"] += _g(r, "retrieval", "deferred", "prompt_tokens")
        s["retrieval_deferred_out"] += _g(
            r, "retrieval", "deferred", "completion_tokens"
        )
        s["retrieval_deferred_calls"] += _g(r, "retrieval", "deferred", "calls")
        s["retrieval_rows_judged"] += _g(r, "retrieval", "sem_rows_judged")
        s["retrieval_worker_requests"] += _g(r, "retrieval", "sem_worker_requests")
        s["retrieval_edges"] += _g(r, "retrieval", "edges")
        s["retrieval_sem_wait_ms"] += _g(r, "retrieval", "sem_llm_wait_ms")
        s["generation_ms"] += _g(r, "generation", "ms")
        s["generation_in"] += _g(r, "generation", "prompt_tokens")
        s["generation_out"] += _g(r, "generation", "completion_tokens")
        s["evaluation_ms"] += _g(r, "evaluation", "ms")
        s["valid"] += 1 if _g(r, "evaluation", "fitness") > 0 else 0
    best = recs[-1]["best_fitness"]
    out = dict(s)
    out.update(
        {
            "iterations": len(recs),
            "seed": seed,
            "best": best,
            "gain": best - seed,
            "curve": [recs[0]["parent_fitness"]] + [r["best_fitness"] for r in recs],
            "fitness": [_g(r, "evaluation", "fitness") for r in recs],
            "placements": sorted(
                {str(_g(r, "retrieval", "placement", default="")) for r in recs} - {""}
            ),
            "decisions": [r.get("retrieval", {}).get("decision") for r in recs],
            "retrieved": [r.get("retrieved_ids", []) for r in recs],
        }
    )
    return out


def identity(a: dict, b: dict) -> dict:
    """Iterations with the same parent and retrieved context, until the trajectories diverge."""
    same_ctx = sum(
        1 for x, y in zip(a["retrieved"], b["retrieved"], strict=False) if x == y
    )
    first_div = next(
        (
            i + 1
            for i, (x, y) in enumerate(zip(a["fitness"], b["fitness"], strict=False))
            if abs(x - y) > 1e-12
        ),
        None,
    )
    return {"same_context_iterations": same_ctx, "first_fitness_divergence": first_div}


def main(run_dir: str) -> dict:
    run = Path(run_dir)
    seeds = {}
    for f in sorted(run.glob("summary*.json")):  # one per process sharing the directory
        for c in json.loads(f.read_text()).get("cells", []):
            if "seed_fitness" in c:
                seeds[(c["task"], c["arm"])] = c["seed_fitness"]
    cells = load(run)
    out = {"cells": {}, "arms": {}, "identity": {}}
    for (task, arm), recs in cells.items():
        out["cells"][f"{task}|{arm}"] = cell_summary(
            recs, seeds.get((task, arm), recs[0]["parent_fitness"])
        )
    arms = sorted({k.split("|")[1] for k in out["cells"]})
    tasks = sorted({k.split("|")[0] for k in out["cells"]})
    for arm in arms:
        cs = [out["cells"][f"{t}|{arm}"] for t in tasks if f"{t}|{arm}" in out["cells"]]
        agg = defaultdict(float)
        for c in cs:
            for k, v in c.items():
                if isinstance(v, (int, float)):
                    agg[k] += v
        agg["mean_gain"] = agg["gain"] / max(len(cs), 1)
        agg["wins"] = 0
        out["arms"][arm] = dict(agg)
    for t in tasks:
        best_arm = max(
            (a for a in arms if f"{t}|{a}" in out["cells"]),
            key=lambda a: out["cells"][f"{t}|{a}"]["best"],
        )
        out["arms"][best_arm]["wins"] += 1
        for a, b in (
            ("A2-trimem", "A3-trimem-alt"),
            ("A2-trimem", "A4-trimem-deferred"),
        ):
            if f"{t}|{a}" in out["cells"] and f"{t}|{b}" in out["cells"]:
                out["identity"][f"{t}|{a}~{b}"] = identity(
                    out["cells"][f"{t}|{a}"], out["cells"][f"{t}|{b}"]
                )
    (run / "analysis.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main(sys.argv[1])
    for arm, a in res["arms"].items():
        print(
            f"{arm:20s} gain {a['mean_gain']:+.4f} wins {a['wins']} | constr {a['construction_ms'] / 1000:6.1f}s "
            f"llm {a['construction_llm_in']:.0f}+{a['construction_llm_out']:.0f} | retr {a['retrieval_ms'] / 1000:6.1f}s "
            f"sem {a['retrieval_sem_in']:.0f}+{a['retrieval_sem_out']:.0f} def {a['retrieval_deferred_in']:.0f} | "
            f"gen {a['generation_ms'] / 1000:6.0f}s {a['generation_in']:.0f}+{a['generation_out']:.0f}"
        )
    for k, v in res["identity"].items():
        print(k, v)


def main_multi(run_dirs: list[str], out: str) -> dict:
    """Average several replicate runs: cells keyed task|arm, numeric fields averaged over reps,
    curves averaged element-wise, per-rep best scores kept in `reps`."""
    per = [main(d) for d in run_dirs]
    keys = sorted({k for p in per for k in p["cells"]})
    cells = {}
    for k in keys:
        cs = [p["cells"][k] for p in per if k in p["cells"]]
        m = {
            f: sum(c[f] for c in cs) / len(cs)
            for f in cs[0]
            if isinstance(cs[0][f], (int, float))
        }
        n = min(len(c["curve"]) for c in cs)
        m["curve"] = [sum(c["curve"][i] for c in cs) / len(cs) for i in range(n)]
        m["reps"] = [c["best"] for c in cs]
        m["placements"] = sorted({p for c in cs for p in c["placements"]})
        cells[k] = m
    arms = sorted({k.split("|")[1] for k in keys})
    tasks = sorted({k.split("|")[0] for k in keys})
    arm_out = {}
    for a in arms:
        cs = [cells[f"{t}|{a}"] for t in tasks if f"{t}|{a}" in cells]
        agg = {
            f: sum(c[f] for c in cs)
            for f in cs[0]
            if isinstance(cs[0][f], (int, float))
        }
        agg["mean_gain"] = agg["gain"] / len(cs)
        agg["wins"] = 0
        arm_out[a] = agg
    for t in tasks:
        best = max(
            (a for a in arms if f"{t}|{a}" in cells),
            key=lambda a: cells[f"{t}|{a}"]["best"],
        )
        arm_out[best]["wins"] += 1
    res = {
        "cells": cells,
        "arms": arm_out,
        "identity": {},
        "replicates": len(run_dirs),
        "run_dirs": run_dirs,
    }
    Path(out).write_text(json.dumps(res, indent=1))
    return res
