"""EvoTrace plan-choice experiment (TrajectoryDB's example): telemetry-first vs semantic-first.

    PYTHONPATH=. python -m trimem.experiments.evotrace_plans \
        --dsn postgresql://$USER@127.0.0.1:55441/trimem_evo --out results/trimem_v0.1.0/evotrace_plans.json

"Pushing a cheap telemetry filter before a semantic judge can reduce LLM invocations, whereas an
early, highly selective semantic filter may avoid an expensive traversal. Choosing between these
plans requires estimating both the cost and selectivity of relational and semantic operators."

Scenario S1 (cold) / S2 (warm): an agent's attempt on ahc015 timed out; retrieve how earlier
timeouts were repaired by their lineage descendants, restricted to timeouts whose change tried to
do more search work (the usual cause of the timeout). S3: from the ANN neighbourhood of the agent's attempt, keep attempts whose
change restructured the solution and return the best attempts within two hops of the
cross-session similarity graph.

For each scenario every placement candidate is forced in turn (trimem.force_placement), with its
own prompt_version for cold runs, and the planner's own estimate for every candidate is read from
the auto plan's "Placement Decision".
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import psycopg

from trimem import DEFAULT_MODEL
from trimem.llm import OpenAIJudge
from trimem.worker import Worker

P_SPEED = "The change makes the program do more search work per run (for example more samples, a deeper lookahead, or more iterations), which can cause a time-limit failure."
P_ALGO = "The change restructures the solution (for example a full rewrite or new search logic) rather than only tuning numeric parameters or constants."

Q_REPAIR = """
SELECT d.id, round(d.fitness::numeric, 0) AS fitness, s.id AS timed_out_attempt
FROM attempt s,
     LATERAL trimem.expand(s.vid, 'lineage', 32) e
     JOIN attempt d ON d.vid = e.dst
WHERE s.task = 'ale:ahc015' AND s.judge = 'TIME_LIMIT_EXCEEDED'
  AND d.is_valid AND d.judge = 'ACCEPTED' AND d.exec_time < 2.0
  {sem}
ORDER BY d.fitness DESC LIMIT 5
""".strip()
SEM_REPAIR = (
    f"AND trimem.sem_filter('{P_SPEED.replace(chr(39), chr(39) * 2)}', s.changes)"
)

Q_NEIGH = """
SELECT DISTINCT n.id, round(n.fitness::numeric, 0) AS fitness
FROM (SELECT id, vid, changes FROM attempt
      WHERE task = 'ale:ahc015' AND id <> 500
      ORDER BY emb <=> (SELECT emb FROM attempt WHERE id = 500) LIMIT 100) s,
     LATERAL trimem.expand(s.vid, 'similar', 16) e1,
     LATERAL trimem.expand(e1.dst, 'similar', 16) e2
     JOIN attempt n ON n.vid = e2.dst
WHERE n.is_valid AND n.judge = 'ACCEPTED' AND n.fitness > 0
  {sem}
ORDER BY fitness DESC, n.id LIMIT 5
""".strip()
SEM_NEIGH = f"AND trimem.sem_filter('{P_ALGO}', s.changes)"

SCENARIOS = [
    {
        "id": "S1",
        "title": "Timeout repair, cold cache",
        "sql": Q_REPAIR,
        "sem": SEM_REPAIR,
        "predicate": P_SPEED,
        "warm_from": None,
        "story": "381 timed-out ahc015 attempts are candidates, but only 74 have lineage children. "
        "Traversal plus the telemetry filter on the children removes most seeds before any LLM call.",
    },
    {
        "id": "S2",
        "title": "Timeout repair, warm cache",
        "sql": Q_REPAIR,
        "sem": SEM_REPAIR,
        "predicate": P_SPEED,
        "warm_from": ("S1", "s=base"),
        "story": "Same query after earlier agents judged every timed-out attempt: the semantic judge is now a "
        "cache lookup, so filtering seeds first skips traversal for the rejected ones.",
    },
    {
        "id": "S3",
        "title": "Strategy neighbourhood, cold cache",
        "sql": Q_NEIGH,
        "sem": SEM_NEIGH,
        "predicate": P_ALGO,
        "warm_from": None,
        "story": "100 ANN seeds, each expanding to about 256 attempts over two hops of the similarity graph. "
        "Judging the 100 seeds first avoids traversing the neighbourhoods of rejected seeds.",
    },
]
CANDIDATES = [
    ("s=base", "SemFilter on the seeds, before the traversal"),
    ("s=top", "SemFilter above the join, after traversal + telemetry"),
]


def walk(p: dict):
    yield p
    for c in p.get("Plans", []):
        yield from walk(c)


def session(conn: psycopg.Connection, pv: str, force: str | None) -> None:
    conn.execute("RESET ALL")
    for stmt in (
        "SET jit = off",
        "SET trimem.timeout_ms = 900000",
        "SET hnsw.iterative_scan = relaxed_order",
        "SET hnsw.ef_search = 200",
        f"SET trimem.prompt_version = '{pv}'",
    ):
        conn.execute(stmt)
    if force:
        conn.execute(f"SET trimem.force_placement = '{force}'")


def run(conn, worker, sql, pv, force, label, note) -> dict:
    session(conn, pv, force)
    calls0, inv0 = worker.stats.llm_calls, worker.stats.invalid
    tok0 = conn.execute(
        "SELECT count(*), coalesce(sum(prompt_tokens),0), coalesce(sum(completion_tokens),0) "
        "FROM trimem.sem_cache WHERE prompt_version = %s",
        (pv,),
    ).fetchone()
    t0 = time.perf_counter()
    explain = conn.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}").fetchone()[
        0
    ][0]
    cold = time.perf_counter() - t0
    tok1 = conn.execute(
        "SELECT count(*), coalesce(sum(prompt_tokens),0), coalesce(sum(completion_tokens),0) "
        "FROM trimem.sem_cache WHERE prompt_version = %s",
        (pv,),
    ).fetchone()
    t0 = time.perf_counter()
    rows = conn.execute(sql).fetchall()
    warm = time.perf_counter() - t0
    sem = [
        n
        for n in walk(explain["Plan"])
        if n.get("Custom Plan Provider") == "TriMEM SemFilter"
    ]
    exp = [
        n
        for n in walk(explain["Plan"])
        if n.get("Custom Plan Provider") == "TriMEM Expand"
    ]
    return {
        "candidate": label,
        "note": note,
        "prompt_version": pv,
        "force": force,
        "explain": explain,
        "planner_total_cost": explain["Plan"]["Total Cost"],
        "execution_ms": explain["Execution Time"],
        "planning_ms": explain["Planning Time"],
        "cold_wall_s": cold,
        "warm_wall_s": warm,
        "llm_wait_ms": sum(n.get("LLM Wait", 0.0) for n in sem),
        "rows_judged": sum(n.get("Rows Judged", 0) for n in sem),
        "worker_requests": sum(n.get("Worker Requests", 0) for n in sem),
        "cache_hits": sum(n.get("Cache Hits", 0) for n in sem),
        "edges_traversed": sum(n.get("Edges Emitted", 0) for n in exp),
        "expand_calls": sum(n.get("Seeds", 0) for n in exp),
        "llm_calls": worker.stats.llm_calls - calls0,
        "invalid_replies": worker.stats.invalid - inv0,
        "prompt_tokens": int(tok1[1] - tok0[1]),
        "completion_tokens": int(tok1[2] - tok0[2]),
        "result": [
            [float(x) if not isinstance(x, int) else x for x in r] for r in rows
        ],
    }


def decision(conn, sql, pv) -> str | None:
    session(conn, pv, None)
    plan = conn.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchone()[0][0]["Plan"]
    return next(
        (
            n.get("Placement Decision")
            for n in walk(plan)
            if n.get("Placement Decision")
        ),
        None,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tag", default=f"evo-{int(time.time())}")
    args = ap.parse_args(argv)
    worker = Worker(
        args.dsn,
        OpenAIJudge(args.base_url, DEFAULT_MODEL, concurrency=64),
        max_take=64,
        take_ms=100,
    )
    th = threading.Thread(target=worker.run, kwargs={"flush_every": 1.0}, daemon=True)
    th.start()
    out = {"model": DEFAULT_MODEL, "tag": args.tag, "scenarios": []}
    pvs: dict[tuple[str, str], str] = {}
    try:
        with psycopg.connect(args.dsn, autocommit=True) as conn:
            for sc in SCENARIOS:
                sql = sc["sql"].format(sem=sc["sem"])
                res = {k: sc[k] for k in ("id", "title", "story", "predicate")} | {
                    "sql": sql,
                    "runs": [],
                }
                base = run(
                    conn,
                    worker,
                    sc["sql"].format(sem=""),
                    f"{args.tag}-{sc['id']}-none",
                    None,
                    "no semantic",
                    "Tri-modal only: V/G/R operators, no LLM",
                )
                res["baseline"] = base
                if sc["warm_from"]:
                    pv = pvs[sc["warm_from"]]
                    res["auto_decision"] = decision(conn, sql, pv)
                    for force, note in CANDIDATES:
                        res["runs"].append(
                            run(conn, worker, sql, pv, force, force, note)
                        )
                else:
                    res["auto_decision"] = decision(
                        conn, sql, f"{args.tag}-{sc['id']}-auto"
                    )
                    for force, note in CANDIDATES:
                        pv = f"{args.tag}-{sc['id']}-{force}"
                        pvs[(sc["id"], force)] = pv
                        res["runs"].append(
                            run(conn, worker, sql, pv, force, force, note)
                        )
                out["scenarios"].append(res)
                print(sc["id"], res["auto_decision"], flush=True)
                for r in [base] + res["runs"]:
                    print(
                        f"   {r['candidate']:12s} exec {r['execution_ms']:9.0f} ms  llm {r['llm_calls']:4d}  "
                        f"req {r['worker_requests']:4d}  tok {r['prompt_tokens']}+{r['completion_tokens']}  "
                        f"edges {r['edges_traversed']:6d}  ids {[x[0] for x in r['result']]}",
                        flush=True,
                    )
    finally:
        worker.stop()
        th.join(timeout=10)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
