"""Write the experiments' own run telemetry back into the TriMEM relational store.

Each OpenEvolve cell (iterations 1-20 in it20/<cell>, resumed 21-50 in it50/<cell>) becomes one *run*:

* every program the run produced is inserted into `attempt` (code, score, judge, run time, tokens) with
  `cell = 'run:<run>'`, a graph vertex and a native lineage edge to its parent -- the corpus filter
  `cell IS NULL` keeps them out of every existing plan; a query opts in by widening that filter;
* every turn (iteration) becomes one `run_turn` row: generation tokens / latency / TTFT / finish
  reason (joined from llm_telemetry by prompt hash), retrieval latency / semantic tokens / edges /
  injected memory ids / placement, evaluation time, parent and child score -- keyed to the child's
  `attempt.id`, so turns join the tri-modal store.

Idempotent per run: a run already ingested is replaced.

    PYTHONPATH=. .venv/bin/python -m trimem.data.ingest_runs --roots openevolve objective ale
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

import psycopg

from trimem.objective.costs import iteration_costs
from trimem.objective.store import count_tokens

R = Path("results/trimem_v0.1.0")
DDL = """
CREATE TABLE IF NOT EXISTS run_turn (
    id                bigserial PRIMARY KEY,
    run               text NOT NULL,             -- <cell name>, e.g. ahc015__rank__s42
    domain            text NOT NULL,             -- openevolve (math fixed) | objective (math optimizer) | ale
    task              text NOT NULL,
    plan              text NOT NULL,
    iteration         integer NOT NULL,          -- OpenEvolve iteration (turn)
    attempt_id        integer,                   -- child in attempt (NULL: the turn produced no child)
    parent_attempt_id integer,
    parent_score      double precision,
    child_score       double precision,
    delta             double precision,          -- child - parent
    valid             boolean,
    judge             text,                      -- ALE judge result
    prompt_tokens     integer,
    completion_tokens integer,
    llm_latency_ms    double precision,
    ttft_ms           double precision,          -- time to first token (streamed runs only)
    finish_reason     text,
    retrieval_ms      double precision,
    retrieval_sem_tokens integer,
    retrieval_edges   integer,
    injected_attempt_ids integer[],
    placement         text,
    eval_ms           double precision,
    UNIQUE (run, iteration, attempt_id)
);
CREATE INDEX IF NOT EXISTS run_turn_task_idx ON run_turn(task, plan);
CREATE INDEX IF NOT EXISTS run_turn_attempt_idx ON run_turn(attempt_id);
"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _metrics(m: object) -> dict:
    return ast.literal_eval(m) if isinstance(m, str) else (m or {})


def _programs(dirs: list[Path]) -> dict[str, dict]:
    """Own programs of the run (latest checkpoint wins; it holds every surviving program)."""
    progs: dict[str, dict] = {}
    for d in dirs:
        cks = sorted((d / "openevolve" / "checkpoints").glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[1]))
        if cks:
            for f in (cks[-1] / "programs").glob("*.json"):
                p = json.loads(f.read_text())
                if not str(p["id"]).startswith("gemext:"):
                    progs[p["id"]] = p
    return progs


def ingest_run(conn: psycopg.Connection, domain: str, name: str) -> tuple[int, int]:
    dirs = [d for d in (R / domain / "it20" / name, R / domain / "it50" / name) if (d / "run_receipt.json").exists()]
    if not dirs:
        return 0, 0
    receipt = json.loads((dirs[0] / "run_receipt.json").read_text())
    task, plan = receipt["task_uid"], name.split("__")[1]
    cell = f"run:{name}"
    progs = _programs(dirs)
    # replace a previous ingest of this run
    conn.execute("DELETE FROM run_turn WHERE run = %s", (name,))
    conn.execute("DELETE FROM attempt WHERE cell = %s", (cell,))
    next_id = conn.execute("SELECT coalesce(max(id), 0) + 1 FROM attempt").fetchone()[0]
    ids: dict[str, int] = {}
    order = sorted(progs.values(), key=lambda p: (int(p.get("iteration_found") or 0), p.get("generation") or 0))
    for p in order:
        ids[p["id"]] = next_id
        next_id += 1
    lang = receipt.get("language") or ("cpp" if task.startswith("ale:") else "python")
    for p in order:
        m = p.get("metrics") or {}
        score = m.get("combined_score")
        valid = score is not None and score > -1e15 and (score > 0 if task.startswith("math:") else
                                                          m.get("judge_result") == "ACCEPTED")
        conn.execute(
            "INSERT INTO attempt (id, uid, task, session, iteration, is_valid, fitness, status, judge, exec_time, "
            "memory_mib, language, code, cell, parent_id, tokens, changes) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (ids[p["id"]], f"{cell}:{p['id']}", task, name, int(p.get("iteration_found") or 0), bool(valid),
             score if score is not None and score > -1e15 else None, "accepted" if valid else "rejected",
             m.get("judge_result"), m.get("max_execution_time_sec") or m.get("eval_time"), m.get("max_memory_usage_mib"),
             lang, p.get("code"), cell, ids.get(p.get("parent_id")), count_tokens(p.get("code")),
             p.get("changes_description") or None))
    conn.execute("UPDATE attempt SET vid = graph_store.gph_upsert_vertex(id) WHERE cell = %s", (cell,))
    conn.execute(
        "SELECT count(graph_store.gph_insert_edge(p.vid, c.vid, graph_store.register_edge_type('lineage'))) "
        "FROM attempt c JOIN attempt p ON p.id = c.parent_id WHERE c.cell = %s", (cell,))
    # turns
    costs, ret, inj = {}, {}, {}
    for d in dirs:
        costs |= iteration_costs(d)
        for r in _jsonl(d / "retrieval_telemetry.jsonl"):
            it = r.get("oe_iteration") or r.get("iteration")
            if it is not None and "observed" not in r:
                ret[int(it)] = r
        for r in _jsonl(d / "injection_trace.jsonl"):
            inj[int(r["iteration"])] = r
    by_iter: dict[int, list[dict]] = {}
    for p in progs.values():
        if p.get("parent_id"):
            by_iter.setdefault(int(p.get("iteration_found") or 0), []).append(p)
    turns = sorted(set(by_iter) | set(inj))
    n = 0
    for it in turns:
        children = by_iter.get(it) or [None]
        c = costs.get(it, {})
        r = ret.get(it, {})
        ij = inj.get(it, {})
        injected = [x.get("attempt_id") for x in (ij.get("provenance") or []) if isinstance(x, dict) and x.get("attempt_id")]
        for ch in children:
            m = (ch or {}).get("metrics") or {}
            parent = progs.get((ch or {}).get("parent_id")) or progs.get(ij.get("parent_id"))
            pscore = (parent or {}).get("metrics", {}).get("combined_score")
            cscore = m.get("combined_score") if ch else None
            if cscore is not None and cscore <= -1e15:
                cscore = None
            conn.execute(
                "INSERT INTO run_turn (run, domain, task, plan, iteration, attempt_id, parent_attempt_id, parent_score, "
                "child_score, delta, valid, judge, prompt_tokens, completion_tokens, llm_latency_ms, ttft_ms, "
                "finish_reason, retrieval_ms, retrieval_sem_tokens, retrieval_edges, injected_attempt_ids, placement, "
                "eval_ms) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING",
                (name, domain, task, plan, it, ids.get((ch or {}).get("id")), ids.get((parent or {}).get("id")),
                 pscore, cscore, (cscore - pscore) if cscore is not None and pscore is not None else None,
                 bool(ch) and cscore is not None and (cscore > 0 if task.startswith("math:") else
                                                       m.get("judge_result") == "ACCEPTED"),
                 m.get("judge_result"), c.get("prompt_tokens"), c.get("completion_tokens"), c.get("llm_ms"),
                 c.get("ttft_ms"), c.get("finish_reason"),
                 r.get("total_ms") or r.get("sql_ms"), (r.get("sem_prompt_tokens") or 0) + (r.get("sem_completion_tokens") or 0)
                 + (r.get("sem_tokens") or 0) or None, r.get("edges"), injected or None, r.get("placement"),
                 1000 * float(m.get("eval_time") or 0) if ch else None))
            n += 1
    return len(progs), n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["openevolve", "objective", "ale"])
    ap.add_argument("--dsn", default=f"postgresql://{os.environ.get('USER')}@127.0.0.1:55441/trimem_evo")
    ap.add_argument("--complete-only", action="store_true", default=True)
    args = ap.parse_args(argv)
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute(DDL)
        total_a = total_t = 0
        for domain in args.roots:
            names = sorted({p.name for ph in ("it20", "it50") for p in (R / domain / ph).glob("*__*__s*")})
            for name in names:
                receipts = [R / domain / ph / name / "run_receipt.json" for ph in ("it20", "it50")]
                done = [json.loads(r.read_text()).get("status") == "complete" for r in receipts if r.exists()]
                if args.complete_only and (not done or not all(done)):
                    continue
                with conn.transaction():
                    a, t = ingest_run(conn, domain, name)
                total_a += a
                total_t += t
                print(f"{domain}/{name}: {a} programs, {t} turns", flush=True)
        conn.execute("SELECT trimem.analyze_graph()")
    print(f"total {total_a} programs, {total_t} turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
