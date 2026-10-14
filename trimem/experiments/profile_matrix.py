"""Offline profiling (BRANE-style, arXiv 2605.27361): every search state × every configuration.

Configurations (2026-10-12): none | rank × {semantic-first, traversal-first} × mixed/{down, up}, k = 5.

A *state* is an OpenEvolve checkpoint of a finished fixed-plan run plus a sampling seed: resuming it
for exactly one iteration with seed s draws the same parent, the same own inspirations and the same
LLM sampling seed under every configuration, so the configurations differ only in what TriMEM
injects. Each (state, configuration) is one `run_arm --policy fixed` cell; its decision features and
outcome land in `trimem_outcome` (session `prof__...`), the generation cost in the cell's telemetry.

    PYTHONPATH=. python -m trimem.experiments.profile_matrix --domain math --states-per-task 40
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from trimem.experiments.oe_matrix import cpu_util

R = Path("results/trimem_v0.1.0")
# 2026-10-12 profiling space: no memory, or rank (SemScore-weighted) with k = 5 on the 2-hop mixed path,
# semantic operator placed first (d=base) or after the traversal (d=top), direction down / up.
CONFIGS = {
    "none": ["--arm", "none"],
    **{f"rank-{order}-mixed-{d}": ["--arm", "trimem", "--trimem-mode", "rank", "--trimem-force", force,
                                   "--trimem-graph", f"mixed/{d}"]
       for order, force in (("semfirst", "d=base"), ("travfirst", "d=top")) for d in ("down", "up")},
}
MATH = ["circle_packing", "heilbronn_convex_13", "second_autocorr_ineq", "third_autocorr_ineq", "heilbronn_triangle"]
PLANS = ["nocontext", "none", "trimodal", "rank", "hard", "hard-semfirst", "hard-travfirst"]
CHECKPOINTS = [("it20", 10), ("it20", 20), ("it50", 30), ("it50", 40)]


def states(task: str, n: int) -> list[tuple[Path, int, int]]:
    """(checkpoint dir, iteration, seed): every (run, checkpoint) with seed 1001, then again with 1002, ..."""
    base = [(R / "openevolve" / ph / f"{task}__{plan}__s42" / "openevolve" / "checkpoints" / f"checkpoint_{t}", t)
            for plan in PLANS for ph, t in CHECKPOINTS]
    base = [(p, t) for p, t in base if p.exists()]
    out, rep = [], 0
    while len(out) < n and base:
        out += [(p, t, 1001 + rep) for p, t in base][: n - len(out)]
        rep += 1
    return out


def done(cell: Path) -> bool:
    r = cell / "run_receipt.json"
    return r.exists() and json.loads(r.read_text()).get("status") == "complete"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="math", choices=["math"])
    ap.add_argument("--states-per-task", type=int, default=50)
    ap.add_argument("--max-parallel", type=int, default=12)
    ap.add_argument("--cpu-target", type=float, default=0.8)
    ap.add_argument("--cores", default="0-27")
    ap.add_argument("--dsn", default=f"postgresql://{os.environ.get('USER')}@127.0.0.1:55441/trimem_evo")
    args = ap.parse_args(argv)
    out = R / "profiling" / args.domain
    out.mkdir(parents=True, exist_ok=True)
    queue = []
    for task in MATH:
        for ck, t, seed in states(task, args.states_per_task):
            src = ck.parents[2].name.split("__")[1]  # plan of the source run
            for cfg in CONFIGS:
                name = f"prof__{task}__{cfg}__{src}__ck{t}__s{seed}"
                if not done(out / name):
                    queue.append((task, cfg, ck, t, seed, out / name))
    (out / "plan.json").write_text(json.dumps({"configs": list(CONFIGS), "states_per_task": args.states_per_task,
                                               "cells": len(queue)}, indent=1))
    print(f"{len(queue)} profiling cells to run", flush=True)
    running: list[tuple[subprocess.Popen, Path]] = []
    while queue or running:
        for proc, cell in list(running):
            if proc.poll() is not None:
                running.remove((proc, cell))
                print(f"finished {cell.name} rc={proc.returncode} complete={done(cell)}", flush=True)
        if queue and len(running) < args.max_parallel and (not running or cpu_util() < args.cpu_target):
            task, cfg, ck, t, seed, cell = queue.pop(0)
            cell.mkdir(parents=True, exist_ok=True)
            cmd = ["nice", "-n", "10", "taskset", "-c", args.cores, ".venv-e0/bin/python", "-m",
                   "bench.agent_memory.gem_oe.run_arm", "--task", f"math:{task}", *CONFIGS[cfg],
                   "--task-source", "trimem", "--trimem-dsn", args.dsn,
                   "--trimem-prompt-version", f"prof-{task}",  # one semantic cache per task
                   "--resume-from", str(ck), "--max-iterations", str(t + 1), "--seed", str(seed),
                   "--out", str(cell), "--llm-base", "http://127.0.0.1:8000/v1",
                   "--llm-model", "Qwen/Qwen3.8-27B-FP8",
                   "--embedding-endpoint", "http://127.0.0.1:8011/v1/embeddings",
                   "--injection-rate", "0.5", "--injection-frequency", "1.0",
                   "--llm-telemetry", "--thinking", "false"]
            env = os.environ | {"PYTHONPATH": ".", "TRIMEM_EVAL_THREADS": "1", "OMP_NUM_THREADS": "1"}
            log = (cell / "run.log").open("w")
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), cell))
        else:
            time.sleep(5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
