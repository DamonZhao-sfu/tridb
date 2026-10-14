"""Run the Fig. 1(b) retrieval query under every candidate plan and record cost and results.

    PYTHONPATH=. python -m trimem.experiments.fig1b_plans \
        --dsn postgresql://$USER@127.0.0.1:55441/trimem_dev --out results/trimem_v0.1.0/fig1b_plans.json

Each plan uses its own prompt_version, so its first run is cache-cold and its LLM calls/tokens
are attributable. The cold run is an EXPLAIN ANALYZE (plan tree + per-node counters + latency);
the second run is the plain query (returned rows, warm-cache latency). The predicate is decidable
("the attempt number is divisible by 3") so every LLM verdict is checked against id % 3 = 0.
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

PRED = "the attempt number is divisible by 3"
SEM = f"trimem.sem_filter('{PRED}', a.outcome)"

QUERY = """
WITH seeds AS (
  (SELECT vid FROM attempt ORDER BY emb <=> (SELECT emb FROM attempt WHERE id = 42) LIMIT 8)
  UNION ALL
  (SELECT vid FROM attempt WHERE id = ANY('{{7,11}}'))
)
SELECT a.id, a.fitness FROM seeds s,
  LATERAL trimem.expand(s.vid, 'lineage', 16) e
  JOIN attempt a ON a.vid = e.dst
WHERE a.valid AND a.fitness > 0.2 {sem}
ORDER BY a.fitness DESC LIMIT 5
"""

PLANS = [
    {
        "id": "P0",
        "name": "Tri-modal only (no semantic operator)",
        "role": "V/G/R baseline: no LLM, answers a different (weaker) question",
        "settings": {},
        "semantic": False,
    },
    {
        "id": "P1",
        "name": "TriMEM planned: SemFilter above the join",
        "role": "cost-based choice (trimem.placement = auto)",
        "settings": {"trimem.placement": "auto"},
        "semantic": True,
    },
    {
        "id": "P2",
        "name": "SemFilter pushed to the base scan (nested loop)",
        "role": "classic predicate push-down: judged per inner rescan",
        "settings": {"trimem.placement": "base"},
        "semantic": True,
    },
    {
        "id": "P3",
        "name": "Semantic-first: SemFilter on the base table, hash join",
        "role": "naive order: judge every candidate row before the graph narrows them",
        "settings": {
            "trimem.placement": "base",
            "enable_nestloop": "off",
            "enable_mergejoin": "off",
        },
        "semantic": True,
    },
    {
        "id": "P4",
        "name": "SemFilter above the join, hash join",
        "role": "same semantic placement as P1, different relational join method",
        "settings": {
            "trimem.placement": "top",
            "enable_nestloop": "off",
            "enable_mergejoin": "off",
        },
        "semantic": True,
    },
]


def apply(conn: psycopg.Connection, settings: dict[str, str]) -> None:
    conn.execute("RESET ALL")
    conn.execute("SET jit = off")
    conn.execute("SET trimem.timeout_ms = 600000")
    for k, v in settings.items():
        conn.execute(f"SET {k} = '{v}'")


def walk(p: dict):
    yield p
    for c in p.get("Plans", []):
        yield from walk(c)


def run_plan(conn: psycopg.Connection, plan: dict, tag: str, worker: Worker) -> dict:
    pv = f"{tag}-{plan['id']}"
    apply(conn, plan["settings"])
    conn.execute(f"SET trimem.prompt_version = '{pv}'")
    sql = QUERY.format(sem=f"AND {SEM}" if plan["semantic"] else "")
    calls0, inv0 = worker.stats.llm_calls, worker.stats.invalid
    t0 = time.perf_counter()
    explain = conn.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}").fetchone()[
        0
    ][0]
    cold_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = conn.execute(sql).fetchall()
    warm_wall = time.perf_counter() - t0
    oracle_sql = QUERY.format(sem="AND a.id % 3 = 0" if plan["semantic"] else "")
    oracle = conn.execute(oracle_sql).fetchall()
    tok = conn.execute(
        "SELECT count(*), coalesce(sum(prompt_tokens),0), coalesce(sum(completion_tokens),0), "
        "coalesce(sum(seconds),0) FROM trimem.sem_cache WHERE prompt_version = %s",
        (pv,),
    ).fetchone()
    judged = conn.execute(
        "SELECT predicate, verdict, key FROM trimem.sem_cache WHERE prompt_version = %s",
        (pv,),
    ).fetchall()
    sem_nodes = [
        n
        for n in walk(explain["Plan"])
        if n.get("Custom Plan Provider") == "TriMEM SemFilter"
    ]
    # verdict accuracy against the decidable oracle (inputs are "attempt N outcome text")
    acc = None
    if plan["semantic"]:
        inputs = conn.execute(
            "SELECT trimem.key('sem_filter', %s, %s, %s, outcome), id FROM attempt",
            (DEFAULT_MODEL, pv, PRED),
        ).fetchall()
        id_of = dict(inputs)
        correct = sum(
            1 for _p, v, k in judged if k in id_of and v == (id_of[k] % 3 == 0)
        )
        acc = {"judged": len(judged), "correct": correct}
    return {
        **plan,
        "sql": sql.strip(),
        "prompt_version": pv,
        "explain": explain,
        "planner_total_cost": explain["Plan"]["Total Cost"],
        "execution_ms": explain["Execution Time"],
        "planning_ms": explain["Planning Time"],
        "cold_wall_s": cold_wall,
        "warm_wall_s": warm_wall,
        "llm_wait_ms": sum(n.get("LLM Wait", 0.0) for n in sem_nodes),
        "rows_judged": sum(n.get("Rows Judged", 0) for n in sem_nodes),
        "llm_calls": worker.stats.llm_calls - calls0,
        "invalid_replies": worker.stats.invalid - inv0,
        "cached_verdicts": tok[0],
        "prompt_tokens": int(tok[1]),
        "completion_tokens": int(tok[2]),
        "llm_seconds_sum": float(tok[3]),
        "result": [{"id": r[0], "fitness": float(r[1])} for r in rows],
        "oracle_result": [{"id": r[0], "fitness": float(r[1])} for r in oracle],
        "matches_oracle": [r[0] for r in rows] == [r[0] for r in oracle],
        "verdict_accuracy": acc,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tag", default=f"fig1b-{int(time.time())}")
    ap.add_argument("--plans", default="P0,P1,P2,P3,P4")
    args = ap.parse_args(argv)

    judge = OpenAIJudge(args.base_url, DEFAULT_MODEL, concurrency=64)
    worker = Worker(args.dsn, judge, max_take=64, take_ms=100)
    th = threading.Thread(target=worker.run, kwargs={"flush_every": 1.0}, daemon=True)
    th.start()
    out = {"model": DEFAULT_MODEL, "predicate": PRED, "tag": args.tag, "plans": []}
    try:
        with psycopg.connect(args.dsn, autocommit=True) as conn:
            for plan in PLANS:
                if plan["id"] not in args.plans.split(","):
                    continue
                res = run_plan(conn, plan, args.tag, worker)
                out["plans"].append(res)
                print(
                    f"{res['id']}: exec {res['execution_ms']:.0f} ms, judged {res['rows_judged']}, "
                    f"llm {res['llm_calls']}, tokens {res['prompt_tokens']}+{res['completion_tokens']}, "
                    f"warm {res['warm_wall_s'] * 1000:.0f} ms, result {[r['id'] for r in res['result']]}, "
                    f"oracle_match {res['matches_oracle']}",
                    flush=True,
                )
    finally:
        worker.stop()
        th.join(timeout=10)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
