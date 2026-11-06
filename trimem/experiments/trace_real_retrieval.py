"""Reconstruct one logged trajectory retrieval exactly, including every SemFilter verdict.

For a cell + iteration this re-creates the retrieval inputs byte for byte and checks them
against the logged receipt:
- the current-agent context (``retrieval_contexts/NNNN.json`` -> CurrentAgentContext.to_text,
  sha256 must equal the receipt's ``context_sha256``);
- the query vector (FullCodeEmbedder on the parent program; sha256 of the embedded text must
  equal the receipt's ``embedding.sha256``).
It then streams the plan's rows in executor order and looks every judge input up in
``trimem.sem_cache`` (key = trimem.key(op, model, prompt_version, predicate, input)) to recover
the verdict the run actually received. The replayed selection must equal the logged
``selected_attempt_ids``. Read-only on trimem_traj; no LLM call.

    CUDA_VISIBLE_DEVICES=1 .venv-e0/bin/python -m trimem.experiments.trace_real_retrieval CELL_DIR [--iteration 1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from trimem import trajectory_query as tq
from trimem.code_embedding import FullCodeEmbedder
from trimem.experiments import trajectory_v7 as v7

DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_traj"
MODEL, PROMPT_VERSION = "Qwen/Qwen3.8-27B-FP8", "trajectory-context-v4-candidate-code"
SEEDS_START = "SELECT a.id AS seed_id"


def seeds_sql(base: dict) -> str:
    """The plan's seed subquery (vector seeds + relational guard), cut from the canonical SQL."""
    sql = v7.v7_query("VGR", "emb", base, "L+Cu+S")
    i = sql.index(SEEDS_START)
    j = sql.index(") s\nCROSS JOIN LATERAL")
    return sql[i:j]


def verdict(
    conn: psycopg.Connection, op: str, predicate: str, context: str, evidence: str
) -> bool | None:
    text = tq.render_judge_input(context, evidence)
    row = conn.execute(
        "SELECT c.verdict FROM trimem.sem_cache c WHERE c.key = trimem.key(%s, %s, %s, %s, %s)",
        (op, MODEL, PROMPT_VERSION, predicate, text),
    ).fetchone()
    return None if row is None else row["verdict"]


def trace(cell: Path, iteration: int, device: str = "cuda:0") -> dict:
    rec = [json.loads(x) for x in open(cell / "trajectory_retrieval.jsonl")]
    rec = next(r for r in rec if r["iteration"] == iteration)
    audit = next(
        json.loads(x)
        for x in open(cell / "retrieval_audit.jsonl")
        if json.loads(x)["iteration"] == iteration
    )
    ctx = tq.CurrentAgentContext(
        **json.load(open(cell / f"retrieval_contexts/{iteration:04d}.json"))
    )
    emb = FullCodeEmbedder(device=device)
    vec = [float(x) for x in emb.embed(ctx.program_code)]
    if emb.last_receipt["sha256"] != rec["embedding"]["sha256"]:
        raise SystemExit("embedded text differs from the logged query")
    params = tq.bind_context(
        {
            "task": ctx.task_id,
            "exclude_session": None,
            "q": "[" + ",".join(map(str, vec)) + "]",
            "predicate": tq.FILTER_CRITERION,
            "score_predicate": tq.SCORE_CRITERION,
        },
        ctx,
    )
    if params["current_context_sha256"] != rec["context_sha256"]:
        raise SystemExit("rebuilt context differs from the logged context")
    plan, base, edges = rec["plan"], rec["base_knobs"], rec["v7_knobs"]["edges"]
    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        conn.read_only = True
        seeds = conn.execute(seeds_sql(base), params).fetchall()
        for s in seeds:
            s["cos"] = round(
                1
                - float(
                    conn.execute(
                        "SELECT emb <=> %s::vector AS d FROM attempt WHERE id=%s",
                        (params["q"], s["id"]),
                    ).fetchone()["d"]
                ),
                4,
            )
            if plan == "P1":
                s["verdict"] = verdict(
                    conn,
                    "sem_filter",
                    params["predicate"],
                    params["current_context"],
                    s["evidence"],
                )
        # Rows of the graph stage: VGR for P2; for P1 the same expansion but only from passing seeds.
        rows = conn.execute(v7.v7_query("VGR", "emb", base, edges), params).fetchall()
        if plan == "P1":
            passing = {s["id"] for s in seeds if s["verdict"]}
            rows = [r for r in rows if r["seed_id"] in passing]
        judged, selected, seen = [], [], set()
        for pos, r in enumerate(rows):
            item = {
                "pos": pos + 1,
                "id": r["id"],
                "seed_id": r["seed_id"],
                "session": r["session"],
                "backend": r["backend"],
                "fitness": r["fitness"],
                "parent_fitness": r["parent_fitness"],
                "intent": (r["intent"] or r["framework_changes"] or "")[:600],
                "code_chars": len(r["code"] or ""),
            }
            if (
                plan == "P2"
            ):  # every cached verdict, also for rows after the stream stopped
                item["verdict"] = verdict(
                    conn,
                    "sem_filter",
                    params["predicate"],
                    params["current_context"],
                    r["evidence"],
                )
            judged.append(item)
            if len(selected) >= rec["effective_k"] or (
                plan == "P2" and item["verdict"] is not True
            ):
                continue
            if r["id"] in seen:
                item["pack"] = "duplicate (UNION ALL reached it again)"
            elif not r["code"]:
                item["pack"] = "no code"
            else:
                seen.add(r["id"])
                selected.append(r["id"])
                item["pack"] = f"selected as Inspiration {len(selected)}"
    ok = sorted(selected) == sorted(
        rec["selected_attempt_ids"]
    )  # SemFilter emits in completion order
    return {
        "cell": str(cell),
        "iteration": iteration,
        "plan": plan,
        "knobs": {**base, **rec["v7_knobs"]},
        "context": {
            k: getattr(ctx, k)
            for k in (
                "task_id",
                "objective",
                "program_status",
                "failure_reason",
                "fitness",
                "iteration",
            )
        },
        "context_sha256": rec["context_sha256"],
        "embedding": rec["embedding"],
        "seeds": [
            {
                k: s[k]
                for k in (
                    "id",
                    "session",
                    "backend",
                    "fitness",
                    "parent_fitness",
                    "cos",
                )
            }
            | {
                "intent": (s["intent"] or s["framework_changes"] or "")[:400],
                "verdict": s.get("verdict"),
            }
            for s in seeds
        ],
        "graph_rows": len(rows),
        "judged": judged,
        "selected": selected,
        "logged_selected": rec["selected_attempt_ids"],
        "logged_rows_seen": rec["rows_seen"],
        "replay_matches_log": ok,
        "memory_tokens": rec["memory_tokens"],
        "audit_selected": [s["provenance"]["attempt_id"] for s in audit["selected"]],
        "physical_plan": [
            n.get("Custom Plan Provider") or n.get("Index Name") or n["Node Type"]
            for n in rec["physical_plan_nodes"]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path)
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    t = trace(a.cell, a.iteration)
    if a.out:
        a.out.write_text(json.dumps(t, ensure_ascii=False, indent=1, default=str))
    print(
        json.dumps(
            {
                k: t[k]
                for k in (
                    "plan",
                    "knobs",
                    "graph_rows",
                    "selected",
                    "logged_selected",
                    "logged_rows_seen",
                    "replay_matches_log",
                )
            },
            default=str,
        )
    )
    print(
        "verdicts:",
        [
            (j["pos"], j["id"], j.get("verdict"), j.get("pack"))
            for j in t["judged"]
            if j.get("verdict") is not None or j.get("pack")
        ],
    )
    if t["plan"] == "P1":
        print("seed verdicts:", [(s["id"], s["verdict"]) for s in t["seeds"]])
    print("sha", hashlib.sha256(json.dumps(t, default=str).encode()).hexdigest()[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
