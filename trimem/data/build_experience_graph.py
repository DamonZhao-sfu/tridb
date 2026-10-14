"""Offline construction of the EvoTrace experience graph for the replay tasks.

The loader already holds attempts (relational telemetry + code), code embeddings (HNSW),
native lineage and cross-session similarity edges. This step completes the semantic layer:
every corpus attempt of the given tasks that has a parent but no change description gets one,
derived by the LLM from the parent->child diff (the same derivation the online ingest uses), so
the semantic operators can read every node. Cost is recorded as offline construction cost.

    PYTHONPATH=. python -m trimem.data.build_experience_graph --dsn ... --out results/.../build.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

from trimem.replay.memory import DERIVE_SYSTEM, Chat, _diff, ensure_schema

TASKS = ["math:circle_packing", "math:heilbronn_convex_13", "math:second_autocorr_ineq",
         "math:third_autocorr_ineq"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--retrievable", action="store_true",
                    help="(re)derive every retrievable row (valid, better than its parent), replacing corpus "
                         "summaries that are only structural diff stubs; the original is kept in changes_raw")
    ap.add_argument("--diff-limit", type=int, default=6000, help="characters of unified diff given to the summariser")
    ap.add_argument("--stubs", action="store_true",
                    help="(re)derive every row whose corpus summary is only a stub ('Change N: ...' or 'Full rewrite')")
    args = ap.parse_args(argv)
    tasks = args.tasks.split(",")
    ensure_schema(args.dsn)
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        conn.execute("ALTER TABLE attempt ADD COLUMN IF NOT EXISTS changes_raw text")
        which = ("(c.changes ~ '^Change [0-9]+:' OR c.changes = 'Full rewrite') AND c.changes_raw IS NULL" if args.stubs
                 else "c.is_valid AND c.fitness > p.fitness AND c.changes_raw IS NULL" if args.retrievable
                 else "(c.changes IS NULL OR c.changes = '')")
        rows = conn.execute(
            "SELECT c.id, c.task, p.code, c.code FROM attempt c JOIN attempt p ON p.id = c.parent_id "
            f"WHERE c.task = ANY(%s) AND c.cell IS NULL AND {which} "
            "AND c.code IS NOT NULL AND p.code IS NOT NULL", (tasks,)).fetchall()
    chat = Chat()
    started = time.perf_counter()

    def derive(r):
        text, pt, ct, ms = chat.complete(DERIVE_SYSTEM, f"PARENT/CHILD unified diff:\n{_diff(r[2], r[3], args.diff_limit)}",
                                         temperature=0.0, max_tokens=160)
        return r[0], r[1], " ".join(text.split())[:600], pt, ct, ms

    with ThreadPoolExecutor(args.concurrency) as pool:
        out = list(pool.map(derive, rows))
    wall = time.perf_counter() - started
    per_task: dict[str, dict] = {}
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        with conn.transaction():
            for aid, task, summary, pt, ct, ms in out:
                conn.execute("UPDATE attempt SET changes_raw = coalesce(changes_raw, changes, ''), changes = %s "
                             "WHERE id = %s", (summary, aid))
                t = per_task.setdefault(task, {"derived": 0, "prompt_tokens": 0, "completion_tokens": 0, "llm_ms": 0.0})
                t["derived"] += 1
                t["prompt_tokens"] += pt
                t["completion_tokens"] += ct
                t["llm_ms"] += ms
        conn.execute("ANALYZE attempt")
        conn.execute("SELECT trimem.analyze_graph()")
        census = conn.execute(
            "SELECT task, count(*), count(changes), count(DISTINCT session) FROM attempt "
            "WHERE task = ANY(%s) AND cell IS NULL GROUP BY 1", (tasks,)).fetchall()
    report = {"wall_s": wall, "derived": len(out), "per_task": per_task,
              "census": {t: {"attempts": n, "with_changes": c, "sessions": s} for t, n, c, s in census}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
