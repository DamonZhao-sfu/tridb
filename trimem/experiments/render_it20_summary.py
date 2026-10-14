"""Summary page: fixed plans (openevolve/it20 -> it50) + optimizer arms (objective/it20 -> it50), one seed.
Iterations 1-20 and the resumed 21-50 live in separate cell directories and are stitched here.

The optimizer arms enter as additional plan cells of every task, next to the fixed plans; a plan
is shown once it has a finished cell for every task.

    PYTHONPATH=. python -m trimem.experiments.render_it20_summary --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from trimem.experiments.oe_analyze import _checkpoint_children, cell

R = Path("results/trimem_v0.1.0")
FIXED = ["nocontext", "none", "trimodal", "rank", "rank-semfirst", "rank-travfirst", "hard", "hard-semfirst",
         "hard-travfirst", "semfilter-rank"]
ADAPTIVE = ["adaptive-random", "adaptive-o1-op", "adaptive-o1-onc", "adaptive-o1-onw",
            "adaptive-o2-op", "adaptive-o2-onc", "adaptive-o2-onw",
            "adaptive-v2-random", "adaptive-v2-o3-op", "adaptive-v2-o3-onc", "adaptive-v2-o3-onw",
            "adaptive-v3-random", "adaptive-v3-o4-op", "adaptive-v3-o4-onc", "adaptive-v3-o4-onw",
            "adaptive-v3-o5-op", "adaptive-v3-o5-onw"]
TASKS = ["circle_packing", "heilbronn_convex_13", "second_autocorr_ineq", "third_autocorr_ineq",
         "heilbronn_triangle"]
TEMPLATE = Path(__file__).with_name("it20_summary_template.html")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def construction() -> dict[str, dict]:
    out = {}
    for name in ("experience_graph_build.json", "experience_graph_build_heilbronn_triangle.json"):
        d = json.loads((R / "offline" / name).read_text())
        total = sum(v["derived"] for v in d["per_task"].values())
        for task, v in d["per_task"].items():
            out[task.split(":")[1]] = {"wall_ms": 1000 * d["wall_s"] * v["derived"] / total,
                                       "tokens": v["prompt_tokens"] + v["completion_tokens"]}
    return out


def extend(curve: list[float], cont: Path, until: int) -> list[float]:
    """Append the resumed iterations 21..until (openevolve/it50/<cell>) to a cell's 0..20 curve.
    A finished continuation is read from its trace; a running one from its latest checkpoint."""
    if not cont.exists():
        return curve
    receipt = cont / "run_receipt.json"
    finished = receipt.exists() and json.loads(receipt.read_text()).get("status") == "complete"
    by_iter: dict[int, list[float]] = {}
    trace = _jsonl(cont / "evolution_trace.jsonl")
    for r in trace:
        by_iter.setdefault(int(r["iteration"]), []).append(float((r.get("child_metrics") or {}).get("combined_score") or 0.0))
    # children the tracer dropped (parent evicted) are in the checkpoint database
    extra, ck_done = _checkpoint_children(cont, skip={r.get("child_id") for r in trace})
    for it, score in extra:
        by_iter.setdefault(it, []).append(score)
    done = until if finished else ck_done
    out, best = list(curve), curve[-1]
    for it in range(len(curve), min(done, until) + 1):
        best = max([best, *by_iter.get(it, [])])
        out.append(best)
    return out


def row(d: Path, task: str, plan: str, cons: dict, until: int = 50) -> dict | None:
    receipt = d / "run_receipt.json"
    if not receipt.exists() or json.loads(receipt.read_text()).get("status") != "complete":
        return None
    c = cell(d, 20)
    if c is None:
        return None
    # a complete cell whose last iterations produced no child keeps its best: pad to iteration 20
    c["curve"] = c["curve"] + [c["curve"][-1]] * (21 - len(c["curve"]))
    ret, sem = c["retrieval_ms"], c["retrieval_sem_in"] + c["retrieval_sem_out"]
    if plan.startswith("adaptive"):  # decision rows carry pool + semantic-pass cost; observe rows carry none
        dec = [r for r in _jsonl(d / "retrieval_telemetry.jsonl") if "chosen" in r]
        ret, sem = sum(r["total_ms"] for r in dec), sum(r.get("sem_tokens", 0) for r in dec)
    memory = plan not in ("nocontext", "none")
    cont = R / ("objective" if plan.startswith("adaptive") else "openevolve") / "it50" / d.name
    curve = extend(c["curve"][:21], cont, until)
    cost50 = None  # cumulative costs over iterations 1..50, once the continuation is complete
    if len(curve) > 50 and (cont / "run_receipt.json").exists() and \
            json.loads((cont / "run_receipt.json").read_text()).get("status") == "complete":
        k = cell(cont, until)
        if plan.startswith("adaptive"):
            dec = [r for r in _jsonl(cont / "retrieval_telemetry.jsonl") if "chosen" in r]
            k["retrieval_ms"], k["retrieval_sem_in"], k["retrieval_sem_out"] = (
                sum(r["total_ms"] for r in dec), sum(r.get("sem_tokens", 0) for r in dec), 0)
        cost50 = {"valid": c["valid"] + k["valid"], "gen_in": c["generation_in"] + k["generation_in"],
                  "gen_out": c["generation_out"] + k["generation_out"], "gen_ms": c["generation_ms"] + k["generation_ms"],
                  "ret_ms": ret + k["retrieval_ms"], "sem_tok": sem + k["retrieval_sem_in"] + k["retrieval_sem_out"],
                  "eval_ms": c["evaluation_ms"] + k["evaluation_ms"]}
    return {"cost50": cost50, "task": task, "plan": plan, "seed": c["seed"], "curve": curve, "best": curve[-1],
            "valid": c["valid"], "gen_in": c["generation_in"], "gen_out": c["generation_out"],
            "gen_ms": c["generation_ms"], "ret_ms": ret, "sem_tok": sem, "eval_ms": c["evaluation_ms"],
            "cons_wall_ms": cons[task]["wall_ms"] if memory else 0, "cons_tok": cons[task]["tokens"] if memory else 0}


def optimizer_state(seed: str) -> dict:
    """Progress of every optimizer cell (finished or not) and every logged decision of the arms."""
    import os
    import re

    import psycopg

    status = {}
    for task in TASKS:
        for plan in ADAPTIVE:
            name = f"{task}__{plan}__s{seed}"
            st = {"phase": "queued", "iteration": 0}
            for phase, base in (("it50", R / "objective" / "it50"), ("it20", R / "objective" / "it20")):
                d = base / name
                if not (d / "run_receipt.json").exists():
                    continue
                done = json.loads((d / "run_receipt.json").read_text()).get("status") == "complete"
                its = re.findall(r"Iteration (\d+)", (d / "run.log").read_text()) if (d / "run.log").exists() else []
                st = {"phase": phase, "done": done, "iteration": int(its[-1]) if its else (20 if phase == "it50" else 0)}
                break
            status[name] = st
    dsn = f"postgresql://{os.environ.get('USER')}@127.0.0.1:55441/trimem_evo"
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT session, iteration, candidate, propensity, delta, valid, delta_best FROM trimem_outcome "
            "WHERE session LIKE %s ORDER BY session, iteration", (f"%__adaptive-%__s{seed}",)).fetchall()
    decisions = [{"cell": r[0], "it": r[1], "cand": r[2], "p": r[3], "delta": r[4], "valid": r[5], "delta_best": r[6]}
                 for r in rows]
    return {"status": status, "decisions": decisions}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="42")
    ap.add_argument("--out", default="triMEM_optimizer/it20_summary/trimem_it20_summary.html")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--hide", default="adaptive-random,adaptive-v2-random,adaptive-v3-random,adaptive-o1-op,adaptive-o1-onc,adaptive-o1-onw,"
                    "adaptive-o2-op,adaptive-o2-onc,adaptive-o2-onw",
                    help="plans left off the page (default: random exploration and the v1 arms A1-A6)")
    args = ap.parse_args(argv)
    cons = construction()
    cells = []
    for task in TASKS:
        for plan, base in [(p, R / "openevolve" / "it20") for p in FIXED] + [(p, R / "objective" / "it20") for p in ADAPTIVE]:
            r = row(base / f"{task}__{plan}__s{args.seed}", task, plan, cons)
            if r:
                cells.append(r)
    hidden = {h for h in args.hide.split(",") if h}
    shown = [p for p in FIXED + ADAPTIVE if p not in hidden]
    opt = optimizer_state(args.seed)
    opt = {"status": {k: v for k, v in opt["status"].items() if k.split("__")[1] in shown},
           "decisions": [d for d in opt["decisions"] if d["cell"].split("__")[1] in shown]}
    data = {"plans": shown, "tasks": TASKS, "cells": [c for c in cells if c["plan"] in shown], "seed": args.seed,
            "optimizer": opt}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "data.json").write_text(json.dumps(data))
    out.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(data)))
    if args.pdf:
        subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=8000",
                        f"--print-to-pdf={out.with_suffix('.pdf').resolve()}", "--no-pdf-header-footer",
                        f"file://{out.resolve()}"], check=False, capture_output=True, timeout=120)
    plans = sorted({c["plan"] for c in data["cells"]}, key=data["plans"].index)
    print(out, f"{len(data['cells'])} cells", plans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
