"""Run OpenEvolve x TriMEM cells (task x plan x seed) with bounded concurrency and a CPU cap.

    PYTHONPATH=. python -m trimem.experiments.oe_matrix --out results/trimem_v0.1.0/openevolve/it20 \
        --iterations 20 --seeds 42,43 --max-parallel 12

Each cell is one `bench.agent_memory.gem_oe.run_arm` process (OpenEvolve 0.3.2) pinned to the
experiment cores at low priority. A new cell starts only while whole-machine CPU utilisation is
below --cpu-target. Cells already finished (run_receipt.json status complete) are skipped, so the
command can be re-run to resume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PLANS = {
    "nocontext": ["--arm", "nocontext"],  # no reference programs at all
    "none": ["--arm", "none"],  # OpenEvolve's own population only
    "trimodal": ["--arm", "trimem", "--trimem-mode", "trimodal"],
    "hard": ["--arm", "trimem", "--trimem-mode", "hard"],
    "hard-semfirst": ["--arm", "trimem", "--trimem-mode", "hard", "--trimem-force", "d=base"],
    "hard-travfirst": ["--arm", "trimem", "--trimem-mode", "hard", "--trimem-force", "d=top"],
    "rank": ["--arm", "trimem", "--trimem-mode", "rank"],
    "rank-semfirst": ["--arm", "trimem", "--trimem-mode", "rank", "--trimem-force", "d=base"],
    "rank-travfirst": ["--arm", "trimem", "--trimem-mode", "rank", "--trimem-force", "d=top"],
    # SemFilter (search-algorithm change) + SemRank (transferable idea), placements chosen jointly by the optimizer
    "semfilter-rank": ["--arm", "trimem", "--trimem-mode", "filterrank"],
    # docs/trimem_math_semfilter_plans_v0.1.0.md: seed-SemFilter plans, k = 10 summaries, lineage+similar, mean-depth hops
    **{f"msem-{p}": ["--arm", "trimem", "--trimem-mode", f"seed-{p}"] for p in ("vgr", "rank", "p1", "p2", "opt")},
    # downstream-aware choice of (family, k) every iteration (plan v0.2.0); 8 of 10 slots available
    "adaptive-random": ["--arm", "trimem-adaptive", "--policy", "random"],
    **{f"adaptive-{o}-{m}": ["--arm", "trimem-adaptive", "--policy", o, "--policy-mode", mode]
       for o in ("o1", "o2") for m, mode in (("op", "offline"), ("onc", "online-cold"), ("onw", "online-warm"))},
    # plan v0.3.0: graph configuration x family x k, target = improvement over the run's best,
    # separate 'inject nothing' estimator, O3 = cheapest candidate predicted >= rank:5
    "adaptive-v2-random": ["--arm", "trimem-adaptive", "--space", "v2", "--policy", "random"],
    **{f"adaptive-v2-o3-{m}": ["--arm", "trimem-adaptive", "--space", "v2", "--policy", "o3", "--policy-mode", mode]
       for m, mode in (("op", "offline"), ("onc", "online-cold"), ("onw", "online-warm"))},
    # plan v0.4.0: + mixed path (similar -> lineage), O4 = ĝ/ĝ_ref − 0.1·L̂/L_ref (expected inference
    # seconds per valid child; input tokens not penalised), online prior scaled to Var Δ_best
    "adaptive-v3-random": ["--arm", "trimem-adaptive", "--space", "v3", "--policy", "random"],
    # O5 (BRANE form): argmax P̂(new best | c) − λ·E[output tokens | c]/1000, λ by dual ascent on budget B
    **{f"adaptive-v3-o5-{m}": ["--arm", "trimem-adaptive", "--space", "v3", "--policy", "o5", "--policy-mode", mode]
       for m, mode in (("op", "offline"), ("onc", "online-cold"), ("onw", "online-warm"))},
    **{f"adaptive-v3-o4-{m}": ["--arm", "trimem-adaptive", "--space", "v3", "--policy", "o4", "--policy-mode", mode]
       for m, mode in (("op", "offline"), ("onc", "online-cold"), ("onw", "online-warm"))},
}
ADAPTIVE_RATE = "0.8"
ALE_TASKS = ["ale:ahc015", "ale:ahc025", "ale:ahc024", "ale:ahc039", "ale:ahc046"]
TASKS = ["math:circle_packing", "math:heilbronn_convex_13", "math:second_autocorr_ineq",
         "math:third_autocorr_ineq", "math:heilbronn_triangle"]


def cpu_util(interval: float = 2.0) -> float:
    def read():
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]

    t0, i0 = read()
    time.sleep(interval)
    t1, i1 = read()
    return 1.0 - (i1 - i0) / max(t1 - t0, 1)


def running_cells() -> set[str]:
    """Cells whose run_arm process is alive (e.g. left running by a previous scheduler)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return {line.split("--out ", 1)[1].split()[0].rstrip("/")
            for line in out.splitlines() if "gem_oe.run_arm" in line and "--out " in line}


def done(cell: Path) -> bool:
    receipt = cell / "run_receipt.json"
    return receipt.exists() and json.loads(receipt.read_text()).get("status") == "complete"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--plans", default=",".join(PLANS))
    ap.add_argument("--max-parallel", type=int, default=12)
    ap.add_argument("--cpu-target", type=float, default=0.8)
    ap.add_argument("--cores", default="0-27")
    ap.add_argument("--priors", default="results/trimem_v0.1.0/objective/priors")
    ap.add_argument("--priors-v2", default="results/trimem_v0.1.0/objective/priors_v2")
    ap.add_argument("--priors-v3", default="results/trimem_v0.1.0/objective/priors_v3")
    ap.add_argument("--priors-v3-ale", default="results/trimem_v0.1.0/ale/priors_v3")
    ap.add_argument("--resume-dir", default=None,
                    help="continue the finished cells of this directory from --resume-at to --iterations")
    ap.add_argument("--resume-at", type=int, default=20)
    ap.add_argument("--extra", default="", help="extra run_arm arguments appended to every cell (space separated)")
    ap.add_argument("--dsn", default=f"postgresql://{os.environ.get('USER')}@127.0.0.1:55441/trimem_evo")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queue = []
    for seed in args.seeds.split(","):  # seed-major: replicate 1 of everything finishes first
        for task in args.tasks.split(","):
            for plan in args.plans.split(","):
                cell = out / f"{task.split(':')[1]}__{plan}__s{seed}"
                if not done(cell):
                    queue.append((task, plan, seed, cell))
    running: list[tuple[subprocess.Popen, Path]] = []
    adopted = {c for c in running_cells() if Path(c).parent.resolve() == out.resolve()}
    queue = [q for q in queue if str(q[3]) not in adopted and str(q[3].resolve()) not in adopted]
    print(f"{len(queue)} cells to run, {len(adopted)} already running (adopted)", flush=True)
    while queue or running or adopted:
        adopted &= running_cells()
        for proc, cell in list(running):
            if proc.poll() is not None:
                running.remove((proc, cell))
                print(f"finished {cell.name} rc={proc.returncode} complete={done(cell)}", flush=True)
        # every live cell in this directory counts, whichever scheduler started it
        busy = len({c for c in running_cells() if Path(c).parent.resolve() == out.resolve()})
        if queue and busy < args.max_parallel and (busy == 0 or cpu_util() < args.cpu_target):
            task, plan, seed, cell = queue.pop(0)
            cell.mkdir(parents=True, exist_ok=True)
            cmd = ["nice", "-n", "10", "taskset", "-c", args.cores, ".venv-e0/bin/python", "-m",
                   "bench.agent_memory.gem_oe.run_arm", "--task", task, *PLANS[plan],
                   "--task-source", "trimem", "--trimem-dsn", args.dsn,
                   # msem arms of one task share one semantic cache: identical SemFilter verdicts for P1/P2/opt
                   "--trimem-prompt-version", (f"msem-{task.split(':')[1]}" if plan.startswith("msem-") else
                                               f"oe-{Path(args.resume_dir).name if args.resume_dir else out.name}-{cell.name}"),
                   "--out", str(cell), "--llm-base", "http://127.0.0.1:8000/v1",
                   "--llm-model", "Qwen/Qwen3.8-27B-FP8",
                   "--embedding-endpoint", "http://127.0.0.1:8011/v1/embeddings",
                   "--max-iterations", str(args.iterations), "--seed", seed,
                   "--injection-rate", ADAPTIVE_RATE if plan.startswith("adaptive") else "0.5",
                   "--injection-frequency", "1.0",
                   "--llm-telemetry", "--thinking", "false"]
            if task.startswith("ale:"):  # C++; retrieved programs enter the prompt as change summaries
                # 2026-10-12 ALE setting: 10 slots all filled from memory (10 change summaries), traversal
                # over lineage + similar edges up to 3 hops; 262,144-token window − 32,768 output − margin (×4 output, ×2 context)
                cmd += ["--language", "cpp", "--inject-as", "changes", "--eval-timeout", "600",
                        "--num-diverse", "10", "--prompt-budget-tokens", "220000", "--max-tokens", "32768", "--injection-rate", "1.0",
                        "--no-simplify-nudge"]  # 2026-10-12: the nudge caused most compile errors
                if "--arm" in cmd and cmd[cmd.index("--arm") + 1] == "trimem":
                    cmd += ["--trimem-graph", "lineage+similar/3"]
            if args.resume_dir:  # same cache namespace as the source cell (set above)
                cmd += ["--resume-from", str(Path(args.resume_dir) / cell.name / "openevolve" / "checkpoints"
                                             / f"checkpoint_{args.resume_at}")]
            if plan.startswith(("adaptive-o", "adaptive-v2-o", "adaptive-v3-o")):  # leave-one-task-out prior
                priors = {"adaptive-v2": args.priors_v2,
                          "adaptive-v3": args.priors_v3_ale if task.startswith("ale:") else args.priors_v3
                          }.get(plan[:11], args.priors)
                cmd += ["--policy-prior", str(Path(priors) / f"prior_{task.replace(':', '_')}.pkl")]
            cmd += args.extra.split()
            env = os.environ | {"PYTHONPATH": ".", "TRIMEM_EVAL_THREADS": "1", "OMP_NUM_THREADS": "1",
                                "ALE_BENCH_DATA": "/localhome/hza214/ale-bench-data",
                                "DOCKER_HOST": f"unix:///run/user/{os.getuid()}/podman/podman.sock"}
            log = (cell / "run.log").open("w")
            running.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), cell))
            print(f"started {cell.name} ({len(running)} running, {len(queue)} queued)", flush=True)
        else:
            time.sleep(10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
