"""Reduce a (task x arm) matrix into the two metrics the plan reports, plus the
diagnostics that say whether those metrics mean anything.

    python3 -m bench.agent_memory.gem_oe.analyze --matrix bench/out/oe/phase2_...

METRICS (both are AlphaEvolve's own; see benchmarkAnalysis.md appendix D)
------------------------------------------------------------------------
* best-so-far against compute budget -- the shape of its Figure 8. Emitted as a
  per-iteration series per (task, arm), plus the cross-task mean.
* per-task best score, aggregated as CLASSIFICATION RATES rather than a mean.
  `combined_score` is normalised against each task's own benchmark constant, so
  averaging it across tasks ranks two different units against each other. Every task
  is placed in one of: beat the corpus's best recorded score / matched it within the
  evaluator's own reproduction tolerance / below it.

DIAGNOSTICS (not decoration -- they decide whether the metrics are readable)
---------------------------------------------------------------------------
`copy_rate` is the fraction of iterations whose generated program appears VERBATIM in
its own prompt. On the abandoned T=0 run this was 10/30 for the memory arm and 0/30
for the no-memory arm, and the memory arm's "worse final score" was entirely that: it
copied a good program early and then stopped searching. A best-so-far curve drawn
without this number beside it is unreadable, so it is reported per arm, always.

`distinct_programs` is the same fact from the other side: 15 unique programs in 30
iterations versus 30.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import psycopg

DEFAULT_DSN = "postgresql://127.0.0.1:55432/evotrace_eg"


#: Fractions of a task's improvable headroom at which "reached it" is declared.
THRESHOLDS = (0.25, 0.50, 0.75, 0.90)


def headroom(dsn: str) -> dict[str, dict[str, float]]:
    """Per task: where the run starts, where the corpus got to, and the gap.

    The gap is the denominator of every threshold. Absolute scores cannot be used:
    seeds range from 0.0 (heilbronn_triangle) to 0.9912 (first_autocorr_ineq), so a
    fixed target would be trivial on one task and unreachable on another.
    """
    conn = psycopg.connect(dsn)
    rows = conn.execute(
        "SELECT b.task_uid, r.seed_fit, b.corpus_best FROM"
        " (SELECT task_uid, max(fitness) corpus_best FROM gem_eg_node"
        "  WHERE is_valid AND fitness IS NOT NULL GROUP BY 1) b"
        " JOIN (SELECT task_uid, max(fitness) seed_fit FROM gem_eg_node"
        "       WHERE parent_node_uid IS NULL AND is_valid AND fitness IS NOT NULL"
        "       GROUP BY 1) r USING (task_uid)"
    ).fetchall()
    return {
        r[0]: {
            "seed": float(r[1]),
            "corpus_best": float(r[2]),
            "headroom": float(r[2]) - float(r[1]),
        }
        for r in rows
    }


def iterations_to_thresholds(
    best_so_far: list[float], ref: dict[str, float] | None
) -> dict[str, int | None]:
    """First iteration reaching seed + alpha * headroom, per alpha.

    `None` means the threshold was never reached inside the budget -- RIGHT CENSORED,
    not "the budget". Recording the budget instead would merge "missed it by one
    iteration" with "never came close", and the aggregate would then depend on how
    long the run happened to be allowed to go.
    """
    out: dict[str, int | None] = {}
    for alpha in THRESHOLDS:
        key = f"iters_to_{int(alpha * 100)}"
        if not ref or ref["headroom"] <= 0:
            out[key] = None
            continue
        target = ref["seed"] + alpha * ref["headroom"]
        hit = next((i for i, v in enumerate(best_so_far) if v >= target), None)
        out[key] = hit
    return out


def corpus_best(dsn: str) -> dict[str, float]:
    """The best score each task ever reached in the recorded corpus."""
    conn = psycopg.connect(dsn)
    return {
        row[0]: float(row[1])
        for row in conn.execute(
            "SELECT task_uid, max(fitness) FROM gem_eg_node"
            " WHERE is_valid AND fitness IS NOT NULL GROUP BY task_uid"
        ).fetchall()
    }


def injection_rates(cell: Path) -> dict[str, Any]:
    """How much of the agent's context actually came from memory.

    Three different questions get called "injection rate", and they answer different
    things, so all three are reported rather than one being picked:

    `coverage`  -- fraction of ITERATIONS that received any injection at all. Under
        the `match_baseline` policy this can be far below 1.0 with nothing wrong:
        that policy renders only as many inspirations as the no-memory arm would, and
        early on, with sparse islands, that is zero. Measured at 0% for three straight
        iterations on the first live run, which is what made the policy switch.

    `slot_share` -- of the inspiration slots the prompt actually rendered, the
        fraction filled from memory. This is the knob the plan's Phase 3 sweeps via
        `num_diverse_programs`, and the one to quote when someone asks "how much
        memory did it get".

    `retrieval_yield` -- of what the memory system RETURNED, the fraction that reached
        the prompt. Below 1.0 means retrieval did work that was then discarded (the
        `match_baseline` cap, or the sampler's dedup against the top/diverse sections),
        which is cost without benefit and should be visible.
    """
    path = cell / "injection_trace.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.open()]
    if not rows:
        return {}
    iterations = len(rows)
    with_injection = sum(1 for r in rows if r.get("injected_ids"))
    injected = sum(len(r.get("injected_ids") or []) for r in rows)
    slots = sum(r.get("rendered", 0) for r in rows)
    retrieved = sum(r.get("retrieved", 0) for r in rows)
    return {
        "coverage": round(with_injection / iterations, 4),
        "slot_share": round(injected / slots, 4) if slots else None,
        "retrieval_yield": round(injected / retrieved, 4) if retrieved else None,
        "injected_total": injected,
        "slots_total": slots,
        "retrieved_total": retrieved,
    }


def _injected_hashes(cell: Path) -> set[str]:
    """Every distinct program this run was handed, by content hash.

    Empty for runs from before `injected_sha256` was recorded, and for arm `none`,
    which is handed nothing. An empty set means the exact-reproduction rate is not
    measurable for that cell, which is reported as None rather than as zero.
    """
    path = cell / "injection_trace.jsonl"
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.open():
        out.update(json.loads(line).get("injected_sha256") or [])
    return out


def read_cell(cell: Path) -> dict[str, Any] | None:
    trace = cell / "evolution_trace.jsonl"
    receipt = cell / "run_receipt.json"
    if not trace.is_file() or not receipt.is_file():
        return None

    meta = json.loads(receipt.read_text())
    injected_hashes = _injected_hashes(cell)
    reproduced_injected = 0
    scores: list[float | None] = []
    best_so_far: list[float] = []
    running = float("-inf")
    copied = 0
    hashes: set[str] = set()
    total = 0

    for line in trace.open():
        row = json.loads(line)
        total += 1
        code = (row.get("child_code") or "").strip()
        if code:
            hashes.add(hashlib.sha1(code.encode(), usedforsecurity=False).hexdigest())
            if hashlib.sha256(code.encode("utf-8")).hexdigest() in injected_hashes:
                reproduced_injected += 1
        prompt = row.get("prompt") or {}
        text = (
            f"{prompt.get('system', '')}\n{prompt.get('user', '')}"
            if isinstance(prompt, dict)
            else str(prompt)
        )
        # 200 chars: below that a "verbatim match" is just boilerplate.
        if code and len(code) > 200 and code in text:
            copied += 1
        metrics = row.get("child_metrics") or {}
        score = metrics.get("combined_score")
        scores.append(float(score) if score is not None else None)
        if score is not None and float(score) > running:
            running = float(score)
        best_so_far.append(running if running > float("-inf") else 0.0)

    valid = [s for s in scores if s is not None]
    # The injection rate is an experimental axis, not a detail: three cells of the
    # same task all carry `arm: gem` and are indistinguishable without it.
    rate = meta.get("injection_rate_requested")
    label = meta["arm"] if rate is None else f"{meta['arm']}@{rate:.0%}"
    return {
        "task_uid": meta["task_uid"],
        "arm": label,
        "base_arm": meta["arm"],
        "injection_rate": rate,
        "inject_as": meta.get("inject_as"),
        "retriever": meta.get("retriever"),
        "policy": meta.get("injection_policy"),
        "status": meta.get("status"),
        "iterations": total,
        "seed_fitness": meta.get("seed_fitness"),
        "best": max(valid) if valid else None,
        "best_so_far": best_so_far,
        "scored": len(valid),
        "zero_scored": sum(1 for s in valid if s == 0.0),
        "distinct_programs": len(hashes),
        "copied_verbatim": copied,
        "copy_rate": round(copied / total, 4) if total else None,
        # None, not 0.0, when nothing was injected or the run predates the field:
        # "no injections to reproduce" and "injections that were never reproduced"
        # are different facts and must not average together.
        "reproduced_injected": reproduced_injected if injected_hashes else None,
        "reproduced_injected_rate": (
            round(reproduced_injected / total, 4) if injected_hashes and total else None
        ),
        "injected_programs": meta.get("injected_programs"),
        **{f"injection_{k}": v for k, v in injection_rates(cell).items()},
    }


def classify(best: float | None, reference: float | None, tolerance: float) -> str:
    if best is None or reference is None:
        return "unscored"
    if best > reference * (1 + tolerance):
        return "beat_corpus_best"
    if best >= reference * (1 - tolerance):
        return "matched_corpus_best"
    return "below_corpus_best"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument(
        "--tolerance", type=float, default=1e-2,
        help="band around the corpus best that counts as 'matched'. Defaults to the "
             "same 1e-2 the evaluator gate uses, because the optimiser-based "
             "evaluators drift up to 4.6e-3 between library versions and a tighter "
             "band would report that drift as a difference in agent quality.",
    )
    ap.add_argument(
        "--min-iterations", type=int, default=90,
        help="cells below this are excluded as wedged rather than averaged in",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cells = [c for c in sorted(args.matrix.glob("math_*")) if c.is_dir()]
    rows = [r for r in (read_cell(c) for c in cells) if r]
    # A short cell is not a data point. Six cells once reported `status: complete`
    # after 5-16 of 100 iterations because an evaluator timeout stops OpenEvolve
    # without raising; averaging those into a per-arm rate would let a wedge look
    # like a result.
    short = [r for r in rows if r["iterations"] < args.min_iterations]
    if short:
        print(f"excluded {len(short)} cell(s) with < {args.min_iterations} iterations:")
        for row in short:
            print(f"  {row['task_uid']:<28}{row['arm']:<12}{row['iterations']:>5} iters")
        print()
        rows = [r for r in rows if r["iterations"] >= args.min_iterations]
    if not rows:
        raise SystemExit("no cell reached the minimum iteration count")
    if not rows:
        raise SystemExit(f"no readable cells under {args.matrix}")
    reference = corpus_best(args.dsn)
    refs = headroom(args.dsn)
    for row in rows:
        row.update(iterations_to_thresholds(row["best_so_far"], refs.get(row["task_uid"])))
        row["headroom"] = (refs.get(row["task_uid"]) or {}).get("headroom")

    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a.split("@")[0], a))
    print(f"{'task':<28}{'arm':<12}{'iters':>6}{'best':>9}{'corpus':>9}"
          f"{'verdict':<22}{'cover%':>8}{'slot%':>7}{'copy%':>7}")
    for row in rows:
        ref = reference.get(row["task_uid"])
        row["reference"] = ref
        row["verdict"] = classify(row["best"], ref, args.tolerance)
        # Formatted outside the f-string: nesting same-type quotes inside one is a
        # syntax error before Python 3.12, and this module has to import under the
        # 3.11 interpreter in .venv-e0 as well as the repo's 3.13.
        best = f"{row['best']:.4f}" if row["best"] is not None else "--"
        corpus = f"{ref:.4f}" if ref else "--"
        copy_pct = (row["copy_rate"] or 0.0) * 100
        cover = row.get("injection_coverage")
        share = row.get("injection_slot_share")
        cover_s = f"{cover * 100:.1f}%" if cover is not None else "--"
        share_s = f"{share * 100:.1f}%" if share is not None else "--"
        print(
            f"{row['task_uid']:<28}{row['arm']:<12}{row['iterations']:>6}"
            f"{best:>9}{corpus:>9}  {row['verdict']:<20}"
            f"{cover_s:>8}{share_s:>7}{copy_pct:>6.1f}%"
        )

    print("\n-- iterations to threshold (PRIMARY). `--` = never reached: right "
          "censored, not the budget --")
    print(f"{'task':<28}{'arm':<12}{'headroom':>9}"
          + "".join(f"{f'@{int(a * 100)}%':>7}" for a in THRESHOLDS))
    for row in rows:
        cells_ = "".join(
            f"{(row.get(f'iters_to_{int(a * 100)}')):>7}"
            if row.get(f"iters_to_{int(a * 100)}") is not None else f"{'--':>7}"
            for a in THRESHOLDS
        )
        hr = f"{row['headroom']:.3f}" if row.get("headroom") is not None else "--"
        print(f"{row['task_uid']:<28}{row['arm']:<12}{hr:>9}{cells_}")

    print("\n-- per-task best, aggregated as classification rates (never a mean) --")
    summary: dict[str, Any] = {"arms": {}, "tolerance": args.tolerance}
    for arm in arms:
        subset = [r for r in rows if r["arm"] == arm]
        counts: dict[str, int] = {}
        for row in subset:
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        copies = [r["copy_rate"] for r in subset if r["copy_rate"] is not None]
        distinct = [r["distinct_programs"] / r["iterations"] for r in subset if r["iterations"]]
        covers = [r["injection_coverage"] for r in subset
                  if r.get("injection_coverage") is not None]
        shares = [r["injection_slot_share"] for r in subset
                  if r.get("injection_slot_share") is not None]
        yields = [r["injection_retrieval_yield"] for r in subset
                  if r.get("injection_retrieval_yield") is not None]
        reached = {}
        for alpha in THRESHOLDS:
            key = f"iters_to_{int(alpha * 100)}"
            hits = [r[key] for r in subset if r.get(key) is not None]
            reached[key] = {
                "reached": len(hits),
                "of": len(subset),
                # Median of those that reached it. Reporting a mean over all cells
                # would require inventing a value for the censored ones.
                "median_iters": statistics.median(hits) if hits else None,
            }
        entry = {
            "cells": len(subset),
            "thresholds": reached,
            "verdicts": counts,
            "mean_copy_rate": round(statistics.mean(copies), 4) if copies else None,
            "mean_distinct_fraction": round(statistics.mean(distinct), 4) if distinct else None,
            "mean_injection_coverage": round(statistics.mean(covers), 4) if covers else None,
            "mean_injection_slot_share": round(statistics.mean(shares), 4) if shares else None,
            "mean_retrieval_yield": round(statistics.mean(yields), 4) if yields else None,
        }
        summary["arms"][arm] = entry
        print(f"  {arm:<12} {counts}")
        for key, hit in entry["thresholds"].items():
            if hit["reached"]:
                print(f"           {key}: {hit['reached']}/{hit['of']} cells, "
                      f"median {hit['median_iters']} iters")
        print(f"           injection: coverage={entry['mean_injection_coverage']} "
              f"slot_share={entry['mean_injection_slot_share']} "
              f"retrieval_yield={entry['mean_retrieval_yield']}")
        print(f"           copy_rate={entry['mean_copy_rate']}  "
              f"distinct/iter={entry['mean_distinct_fraction']}")

    worst = max(
        (a for a in summary["arms"].values() if a["mean_copy_rate"] is not None),
        key=lambda a: a["mean_copy_rate"],
        default=None,
    )
    if worst and worst["mean_copy_rate"] > 0.05:
        print(
            f"\n  WARNING: an arm reproduced its prompt verbatim in "
            f"{worst['mean_copy_rate']:.1%} of iterations. Its best-so-far curve "
            "reflects copying, not search, and must not be read as agent quality."
        )

    summary["cells"] = rows
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
