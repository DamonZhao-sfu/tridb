"""Turn the fixed-plan OpenEvolve cells into `trimem_outcome` rows (offline training data, §5).

A fixed cell injects one candidate per iteration — (none, 0) for `none`/`nocontext`, (family, 5)
for the TriMEM plans — with propensity 1. Its features are recomputed exactly as the adaptive arm
computes them: the injected rows from the experience graph, cosine similarity to the parent's code
embedding, and the semantic score/verdict of every row (one shared cache namespace, so the passes
are paid once). Iterations that produced no child are Δ = 0, not valid.

    PYTHONPATH=. python -m trimem.objective.backfill --cells results/trimem_v0.1.0/openevolve/it20
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import psycopg

from trimem.objective.pool import FILTER_SQL, SCORE_SQL, Candidate, PoolRow, featurize
from trimem.objective.store import OUTCOME_DDL, count_tokens
from trimem.replay.memory import Embedder

FAMILY = {
    "trimodal": "trimodal",
    "rank": "rank",
    "hard": "hard",
    "hard-semfirst": "hard",
    "hard-travfirst": "hard",
    "rank-semfirst": "rank",
    "rank-travfirst": "rank",
    "none": "none",
    "nocontext": "nocontext",
}
ROWS_SQL = """
SELECT id, fitness, changes, code, coalesce(tokens, 0), 1 - (emb <=> %(q)s::vector), session
FROM attempt WHERE id = ANY(%(ids)s)
"""


def _jsonl(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if path.exists()
        else []
    )


def _score(metrics: object) -> float:
    m = ast.literal_eval(metrics) if isinstance(metrics, str) else (metrics or {})
    return float(m.get("combined_score") or 0.0)


def backfill_cell(
    conn: psycopg.Connection, emb: Embedder, cell: Path, horizon: int
) -> int:
    task_part, plan, _seed = cell.name.split("__")
    family = FAMILY.get(plan)
    if family is None:
        return 0
    receipt = json.loads((cell / "run_receipt.json").read_text())
    task = receipt["task_uid"]
    trace = {int(r["iteration"]): r for r in _jsonl(cell / "evolution_trace.jsonl")}
    ret = {int(r["iteration"]): r for r in _jsonl(cell / "retrieval_telemetry.jsonl")}
    inj = {int(r["iteration"]): r for r in _jsonl(cell / "injection_trace.jsonl")}
    known: dict[
        str, tuple[str, float]
    ] = {}  # program id -> (code, fitness), from every trace row
    for r in trace.values():
        known[r["parent_id"]] = (r["parent_code"], _score(r["parent_metrics"]))
        known[r["child_id"]] = (r["child_code"], _score(r["child_metrics"]))
    # OpenEvolve's tracer drops a child whose parent was evicted before it returned: recover those
    # iterations from the checkpoint database (it holds every surviving program).
    ck_dir = cell / "openevolve" / "checkpoints" / f"checkpoint_{horizon}"
    ck = {}
    if (ck_dir / "programs").exists():
        for fp in (ck_dir / "programs").glob("*.json"):
            prog = json.loads(fp.read_text())
            ck[prog["id"]] = prog
    for prog in ck.values():
        known.setdefault(prog["id"], (prog.get("code"), _score(prog.get("metrics") or {})))
    for prog in ck.values():
        it = int(prog.get("iteration_found") or 0)
        if prog.get("parent_id") and it not in trace and 0 < it <= horizon and prog["parent_id"] in known:
            pc, pf = known[prog["parent_id"]]
            trace[it] = {"parent_code": pc, "parent_metrics": {"combined_score": pf},
                         "child_code": prog.get("code"), "child_metrics": prog.get("metrics") or {},
                         "child_id": prog["id"], "parent_id": prog["parent_id"]}
    n = 0
    for it in sorted(inj):
        if it > horizon:
            break
        t = trace.get(it)
        if t is not None:
            parent_code, parent_fit, child_fit = (
                t["parent_code"],
                _score(t["parent_metrics"]),
                _score(t["child_metrics"]),
            )
        else:  # generation failed: Δ = 0, invalid; the parent is resolved from other trace rows
            parent_code, parent_fit = known.get(inj[it]["parent_id"], (None, None))
            child_fit = 0.0
        ids = (
            ret.get(it, {}).get("ids", [])
            if family not in ("none", "nocontext")
            else []
        )
        prov = {
            p["attempt_id"]: p.get("parent_fitness")
            for p in inj[it].get("provenance") or []
            if isinstance(p, dict) and "attempt_id" in p
        }
        rows: list[PoolRow] = []
        if ids and parent_code is not None:
            q, _, _ = emb.embed(parent_code)
            qs = "[" + ",".join(f"{x:.6g}" for x in q) + "]"
            got = {
                r[0]: r
                for r in conn.execute(ROWS_SQL, {"q": qs, "ids": ids}).fetchall()
            }
            scores = dict(conn.execute(SCORE_SQL, {"ids": ids}).fetchall())
            passed = {r[0] for r in conn.execute(FILTER_SQL, {"ids": ids}).fetchall()}
            for i in ids:
                r = got[i]
                rows.append(
                    PoolRow(
                        id=i,
                        fitness=r[1],
                        parent_fitness=float(prov.get(i) or r[1]),
                        changes=r[2],
                        code=r[3],
                        tokens=r[4],
                        sim=float(r[5]),
                        session=r[6],
                        score=scores.get(i),
                        verdict=i in passed,
                    )
                )
        if parent_fit is None:
            continue  # parent never seen in the trace: no Δ reference
        fam = "none" if family in ("none", "nocontext") else family
        cand = Candidate(fam, len(rows), rows)
        f = featurize(
            cand,
            parent_fitness=parent_fit,
            iteration=it,
            horizon=horizon,
            parent_tokens=count_tokens(parent_code),
            parent_session=None,
        )
        f["nocontext"] = float(family == "nocontext")
        delta = max(0.0, child_fit - parent_fit)
        conn.execute(
            "INSERT INTO trimem_outcome (session, task, iteration, policy, candidate, propensity, features,"
            " parent_fitness, child_fitness, delta, valid, tok_injected, tok_child, observed_at)"
            " VALUES (%s,%s,%s,%s,%s,1.0,%s,%s,%s,%s,%s,%s,%s, now())"
            " ON CONFLICT (session, iteration) DO UPDATE SET child_fitness = EXCLUDED.child_fitness,"
            " delta = EXCLUDED.delta, valid = EXCLUDED.valid, tok_child = EXCLUDED.tok_child",
            (
                f"fixed:{cell.name}",
                task,
                it,
                f"fixed-{plan}",
                f"{fam}:{len(rows)}",
                json.dumps(f),
                parent_fit,
                child_fit if t else None,
                delta,
                child_fit > 0,
                int(f["tokens_injected"]),
                count_tokens(t["child_code"]) if t else None,
            ),
        )
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument(
        "--dsn",
        default=f"postgresql://{os.environ.get('USER')}@127.0.0.1:55441/trimem_evo",
    )
    ap.add_argument("--prompt-version", default="oe-objective")
    args = ap.parse_args(argv)
    emb = Embedder()
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute(OUTCOME_DDL)
        for stmt in (
            "SET jit = off",
            "SET trimem.timeout_ms = 900000",
            f"SET trimem.prompt_version = '{args.prompt_version}'",
        ):
            conn.execute(stmt)
        total = 0
        for cell in sorted(Path(args.cells).iterdir()):
            if (cell / "run_receipt.json").exists() and json.loads(
                (cell / "run_receipt.json").read_text()
            ).get("status") == "complete":
                k = backfill_cell(conn, emb, cell, args.horizon)
                total += k
                print(f"{cell.name}: {k}", flush=True)
    print(f"total {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
