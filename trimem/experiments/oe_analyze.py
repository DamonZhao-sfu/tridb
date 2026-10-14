"""Aggregate OpenEvolve x TriMEM cells into the replay-analysis schema (cells keyed task|arm).

Per cell (one seed): best-so-far curve over iterations (seed score at index 0, OpenEvolve child
scores folded in iteration order), generation tokens/latency from llm_telemetry.jsonl,
retrieval latency/semantic-operator cost from retrieval_telemetry.jsonl, evaluation time from
the children's eval_time. Construction cost is zero during the run: the experience graph is
built offline (see experience_graph_build*.json). Seeds are averaged per (task, plan).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.open() if x.strip()] if path.exists() else []


def _checkpoint_scores(cell_dir: Path) -> tuple[dict[int, list[float]], int]:
    """Own (non-injected) programs of the latest OpenEvolve checkpoint, by iteration found.
    Lets a still-running cell be read at a checkpoint (checkpoints every 10 iterations)."""
    cks = sorted((cell_dir / "openevolve" / "checkpoints").glob("checkpoint_*"),
                 key=lambda p: int(p.name.split("_")[1]))
    out: dict[int, list[float]] = defaultdict(list)
    if not cks:
        return out, 0
    for f in (cks[-1] / "programs").glob("*.json"):
        prog = json.loads(f.read_text())
        if str(prog.get("id", "")).startswith("gemext:") or (prog.get("metadata") or {}).get("tridb_external"):
            continue
        out[int(prog.get("iteration_found") or 0)].append(float((prog.get("metrics") or {}).get("combined_score") or 0.0))
    return out, int(cks[-1].name.split("_")[1])


def _checkpoint_children(cell_dir: Path, skip: set) -> tuple[list[tuple[int, float]], int]:
    """(iteration_found, score) of the latest checkpoint's own children not in `skip`, and its iteration."""
    cks = sorted((cell_dir / "openevolve" / "checkpoints").glob("checkpoint_*"),
                 key=lambda p: int(p.name.split("_")[1]))
    if not cks:
        return [], 0
    out = []
    for f in (cks[-1] / "programs").glob("*.json"):
        prog = json.loads(f.read_text())
        pid = str(prog.get("id", ""))
        if pid.startswith("gemext:") or (prog.get("metadata") or {}).get("tridb_external") or pid in skip:
            continue
        if not prog.get("parent_id"):
            continue  # the run's root (seed program)
        out.append((int(prog.get("iteration_found") or 0), float((prog.get("metrics") or {}).get("combined_score") or 0.0)))
    return out, int(cks[-1].name.split("_")[1])


def cell(cell_dir: Path, iterations: int) -> dict | None:
    receipt_path = cell_dir / "run_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    trace = sorted(_jsonl(cell_dir / "evolution_trace.jsonl"), key=lambda r: r.get("iteration", 0))
    seed = receipt.get("live_seed_fitness")
    if seed is None:
        seed = receipt.get("seed_fitness") or 0.0
    best, curve, by_iter = seed, [seed], defaultdict(list)
    for r in trace:
        by_iter[int(r.get("iteration", 0))].append(float((r.get("child_metrics") or {}).get("combined_score") or 0.0))
    done = max(by_iter) if by_iter else 0
    # OpenEvolve's tracer drops a child whose parent was evicted before the child returned (3.5% of
    # children measured); the checkpoint database holds every program, so its children are merged in.
    traced = {r.get("child_id") for r in trace}
    extra, ck_done = _checkpoint_children(cell_dir, skip=traced)
    resumed = int(receipt.get("resumed_at_iteration") or 0)  # a resumed cell's checkpoint holds the source's programs
    extra = [(it, sc) for it, sc in extra if it > resumed]
    for it, score in extra:
        by_iter[it].append(score)
    if not trace:  # still running: read the latest checkpoint, exactly up to its iteration
        done = ck_done
    else:
        done = max(done, min(ck_done, max(by_iter) if by_iter else 0))
    valid_extra = sum(1 for it, sc in extra if sc > 0 and it <= iterations)
    for it in range(1, min(done, iterations) + 1):
        best = max([best, *by_iter.get(it, [])])
        curve.append(best)
    llm = [r for r in _jsonl(cell_dir / "llm_telemetry.jsonl") if r.get("status") == "success"]
    ret = _jsonl(cell_dir / "retrieval_telemetry.jsonl")
    placements = sorted({r["placement"] for r in ret if r.get("placement")})
    return {
        "seed": seed, "best": curve[-1], "gain": curve[-1] - seed, "curve": curve,
        "iterations": len(curve) - 1, "status": receipt.get("status"),
        "valid": sum(1 for r in trace if float((r.get("child_metrics") or {}).get("combined_score") or 0) > 0) + valid_extra,
        "generation_ms": sum(r.get("llm_api_ms") or 0 for r in llm),
        "generation_in": sum(r.get("prompt_tokens") or 0 for r in llm),
        "generation_out": sum(r.get("completion_tokens") or 0 for r in llm),
        "retrieval_ms": sum(r.get("total_ms") or 0 for r in ret),
        "retrieval_sem_in": sum(r.get("sem_prompt_tokens") or 0 for r in ret),
        "retrieval_sem_out": sum(r.get("sem_completion_tokens") or 0 for r in ret),
        "retrieval_deferred_in": 0, "retrieval_deferred_out": 0,
        "retrieval_rows_judged": sum(r.get("sem_rows") or 0 for r in ret),
        "retrieval_worker_requests": sum(r.get("sem_worker_requests") or 0 for r in ret),
        "retrieval_edges": sum(r.get("edges") or 0 for r in ret),
        "evaluation_ms": sum(1000 * float((r.get("child_metrics") or {}).get("eval_time") or 0) for r in trace),
        "construction_ms": 0.0, "construction_llm_in": 0, "construction_llm_out": 0,
        "placements": placements,
    }


def main(run_dir: str, iterations: int = 20, prefix: str = "OE-") -> dict:
    run = Path(run_dir)
    per = defaultdict(list)
    for d in sorted(p for p in run.iterdir() if p.is_dir() and "__" in p.name):
        task_key, plan, _seed = d.name.split("__")
        c = cell(d, iterations)
        if c is not None and c["iterations"] > 0:
            per[(f"math:{task_key}", f"{prefix}{plan}")].append(c)
    cells = {}
    for (task, arm), cs in per.items():
        num = [k for k, v in cs[0].items() if isinstance(v, (int, float))]
        m = {k: sum(c[k] for c in cs) / len(cs) for k in num}
        n = min(len(c["curve"]) for c in cs)
        m["curve"] = [sum(c["curve"][i] for c in cs) / len(cs) for i in range(n)]
        m["reps"] = [c["best"] for c in cs]
        m["placements"] = sorted({p for c in cs for p in c["placements"]})
        cells[f"{task}|{arm}"] = m
    return {"cells": cells}
