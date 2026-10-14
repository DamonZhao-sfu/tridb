"""TriMEM replay: evolve EvoTrace math tasks with memory retrieved under different plans.

    PYTHONPATH=. python -m trimem.replay.run --dsn postgresql://$USER@127.0.0.1:55441/trimem_evo \
        --out results/trimem_v0.1.0/replay/run1 --iterations 10

Every (task, arm) cell runs concurrently in its own thread. One SemFilter worker serves all cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import psycopg

from trimem import DEFAULT_MODEL
from trimem.llm import OpenAIJudge
from trimem.replay.memory import Chat, Embedder, Memory, Phase, ensure_schema
from trimem.replay.tasks import Task, evaluate, resolve
from trimem.worker import Worker

ARMS = {
    "A0-none": {
        "memory": False,
        "semantic": False,
        "deferred": False,
        "placement": None,
    },
    "A1-trimodal": {
        "memory": True,
        "semantic": False,
        "deferred": False,
        "placement": None,
    },
    "A2-trimem": {
        "memory": True,
        "semantic": True,
        "deferred": False,
        "placement": "auto",
    },
    "A3-trimem-alt": {
        "memory": True,
        "semantic": True,
        "deferred": False,
        "placement": "alt",
    },
    "A4-trimem-deferred": {
        "memory": True,
        "semantic": True,
        "deferred": True,
        "placement": "auto",
    },
    # 0.2.0 soft semantic operators (not in the default arm set of run1-run3)
    "A5-trimem-keep-unknown": {
        "memory": True,
        "semantic": "keep_unknown",
        "deferred": False,
        "placement": "auto",
    },
    "A6-trimem-rank": {
        "memory": True,
        "semantic": "score",
        "deferred": False,
        "placement": None,  # optimizer chooses where SemScore runs
    },
    "A7-rank-semantic-first": {
        "memory": True,
        "semantic": "score",
        "deferred": False,
        "placement": None,
        "force": "d=base",  # score every candidate before the traversal joins it
    },
    "A8-rank-traverse-first": {
        "memory": True,
        "semantic": "score",
        "deferred": False,
        "placement": None,
        "force": "d=top",  # traverse + telemetry first, score the survivors
    },
}
DEFAULT_ARMS = [
    "A0-none",
    "A1-trimodal",
    "A2-trimem",
    "A3-trimem-alt",
    "A4-trimem-deferred",
]
TASKS = [
    "math:circle_packing",
    "math:heilbronn_convex_13",
    "math:second_autocorr_ineq",
    "math:third_autocorr_ineq",
    "math:heilbronn_triangle",
]

SYSTEM = (
    "You are an expert in algorithm design and numerical optimisation. You improve a Python "
    "program for a scored task. Reply with the COMPLETE improved program in a single "
    "```python code block, keeping every function and signature the evaluator calls."
)


def build_prompt(task: Task, code: str, fitness: float, items: list[dict]) -> str:
    parts = [
        f"# Task\n{task.spec[:6000]}",
        f"# Current program (combined_score = {fitness:.6f}, higher is better)\n```python\n{code}\n```",
    ]
    if items:
        mem = ["# Memory: changes that improved other solutions to this task"]
        for i, it in enumerate(items, 1):
            mem.append(
                f"## Idea {i}: score {it['parent_fitness']:.4f} -> {it['fitness']:.4f}\n"
                f"Change: {it.get('changes') or '(no summary)'}\n"
                f"Code excerpt:\n```python\n{it['code']}\n```"
            )
        parts.append("\n\n".join(mem))
    parts.append(
        "# Instruction\nImprove the current program so its combined_score increases. "
        "Return the full program."
    )
    return "\n\n".join(parts)


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None


def run_cell(
    dsn: str,
    task_uid: str,
    arm: str,
    iterations: int,
    out: Path,
    tag: str,
    llm_url: str,
    emb_url: str,
    context_k: int = 5,
) -> dict:
    cfg = ARMS[arm]
    cell = f"{tag}:{task_uid.split(':')[1]}:{arm}"
    with psycopg.connect(dsn) as c:
        task = resolve(c, task_uid)
    chat, embedder = Chat(llm_url), Embedder(emb_url)
    mem = Memory(
        dsn,
        task_uid,
        cell,
        embedder=embedder,
        chat=chat,
        deferred=cfg["deferred"],
        semantic=cfg["semantic"],
        placement=cfg["placement"],
        derive=cfg["memory"],
        force=cfg.get("force"),
        k=context_k,
    )
    path = out / f"{cell.replace(':', '__')}.jsonl"
    done = [json.loads(x) for x in path.open()] if path.exists() else []
    if done:
        # Resume: the cell's attempts are already in the store; continue from its best program.
        with psycopg.connect(dsn) as c:
            sid, sfit = c.execute(
                "SELECT id, fitness FROM attempt WHERE cell = %s AND iteration = 0",
                (cell,),
            ).fetchone()
            bid, bcode, bfit = c.execute(
                "SELECT id, code, fitness FROM attempt WHERE cell = %s "
                "ORDER BY fitness DESC, iteration LIMIT 1",
                (cell,),
            ).fetchone()
        seed_eval = {"combined_score": sfit}
        stel = {"embed": Phase()}
        best = {"id": bid, "code": bcode, "fitness": bfit}
        history = [r["best_fitness"] for r in done]
        start = done[-1]["iteration"] + 1
        log = path.open("a")
    else:
        log = path.open("w")
        seed_eval = evaluate(task, task.seed_code)
        sid, stel = mem.ingest(
            task.seed_code, seed_eval["combined_score"], True, None, 0, None
        )
        best = {
            "id": sid,
            "code": task.seed_code,
            "fitness": seed_eval["combined_score"],
        }
        history = [best["fitness"]]
        start = 1
    for it in range(start, iterations + 1):
        rec: dict = {
            "cell": cell,
            "task": task_uid,
            "arm": arm,
            "iteration": it,
            "parent_id": best["id"],
            "parent_fitness": best["fitness"],
        }
        items, rtel = [], None
        if cfg["memory"]:
            rtel = mem.retrieve(best["id"], best["code"], best["fitness"])
            items = rtel.items
            rec["retrieval"] = {
                k: v for k, v in asdict(rtel).items() if k not in ("items", "deferred")
            }
            rec["retrieval"]["deferred"] = asdict(rtel.deferred)
            rec["retrieved_ids"] = [i["id"] for i in items]
        prompt = build_prompt(task, best["code"], best["fitness"], items)
        # iteration in the seed: a retry after a failed attempt must not regenerate the same program
        seed = int(hashlib.sha256(f"{it}\x1f{prompt}".encode()).hexdigest()[:8], 16)
        text, pt, ct, gms = chat.complete(
            SYSTEM, prompt, temperature=0.7, top_p=0.95, max_tokens=6144, seed=seed
        )
        rec["generation"] = {"ms": gms, "prompt_tokens": pt, "completion_tokens": ct}
        code = extract_code(text)
        ev = (
            evaluate(task, code)
            if code
            else {"combined_score": 0.0, "error": "no code block", "eval_seconds": 0.0}
        )
        fit = ev["combined_score"]
        rec["evaluation"] = {
            "ms": ev["eval_seconds"] * 1000,
            "fitness": fit,
            "error": str(ev.get("error", ""))[:300],
        }
        if code:
            aid, ctel = mem.ingest(code, fit, fit > 0, best["id"], it, best["code"])
            rec["construction"] = {
                "embed": asdict(ctel["embed"]),
                "derive": asdict(ctel["derive"]),
                "insert_ms": ctel["insert_ms"],
            }
            if fit > best["fitness"]:
                best = {"id": aid, "code": code, "fitness": fit}
        history.append(best["fitness"])
        rec["best_fitness"] = best["fitness"]
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(
            f"[{cell}] it {it}: fit {fit:.4f} best {best['fitness']:.4f} gen {gms / 1000:.0f}s "
            f"tok {pt}+{ct} ctx {len(items)}",
            flush=True,
        )
    log.close()
    return {
        "cell": cell,
        "task": task_uid,
        "arm": arm,
        "seed_fitness": seed_eval["combined_score"],
        "best_fitness": best["fitness"],
        "history": history,
        "seed_construction": asdict(stel["embed"]),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--llm", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--emb", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--tag", default=f"r{int(time.time()) % 100000}")
    ap.add_argument(
        "--summary-name",
        default="summary.json",
        help="lets several processes share one output directory (different arms)",
    )
    ap.add_argument(
        "--context-k",
        type=int,
        default=5,
        help="retrieved items per iteration (SQL LIMIT)",
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ensure_schema(args.dsn)
    worker = Worker(
        args.dsn,
        OpenAIJudge(args.llm, DEFAULT_MODEL, concurrency=64),
        max_take=64,
        take_ms=100,
    )
    wt = threading.Thread(target=worker.run, kwargs={"flush_every": 2.0}, daemon=True)
    wt.start()
    results, threads = [], []

    def go(t: str, a: str) -> None:
        try:
            results.append(
                run_cell(
                    args.dsn,
                    t,
                    a,
                    args.iterations,
                    out,
                    args.tag,
                    args.llm,
                    args.emb,
                    args.context_k,
                )
            )
        except Exception:  # keep other cells running; record the failure
            results.append({"task": t, "arm": a, "error": traceback.format_exc()})
            print(traceback.format_exc(), flush=True)

    for t in args.tasks.split(","):
        for a in args.arms.split(","):
            th = threading.Thread(target=go, args=(t, a))
            th.start()
            threads.append(th)
    for th in threads:
        th.join()
    worker.stop()
    wt.join(timeout=10)
    (out / args.summary_name).write_text(
        json.dumps(
            {
                "tag": args.tag,
                "model": DEFAULT_MODEL,
                "iterations": args.iterations,
                "cells": results,
                "worker": asdict(worker.stats),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
