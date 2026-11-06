"""Create a frozen read-only clone of a memory database (same PostgreSQL instance).

The source is never modified. The clone name is derived from the source content
digest, so the same content always maps to the same name. Refuses to run while
the source has other connections (TEMPLATE copy needs an idle source).

Usage: python -m tools.trace_joint_clone_memory --source trimem_webshop --env webshop --prefix web [--commit]
"""

import argparse
import json

import psycopg
from psycopg import sql

from tools.prepare_trace_joint_campaign import memory_identity

ADMIN = "host=/tmp/trimem-traj port=55450 dbname=postgres"


def source_digest(source, env):
    import hashlib

    dsn = f"host=/tmp/trimem-traj port=55450 dbname={source} options='-c default_transaction_read_only=on'"
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT s.key, s.success, s.final_score FROM public.session s "
            "JOIN public.task t ON t.id=s.task_id WHERE t.env=%s ORDER BY s.key",
            (env,),
        ).fetchall()
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(list(row), default=str).encode() + b"\n")
    return h.hexdigest(), len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--prefix", required=True, choices=["web", "sci"])
    p.add_argument("--commit", action="store_true")
    a = p.parse_args()
    digest, sessions = source_digest(a.source, a.env)
    target = f"trace_ablation_{a.prefix}_{digest[:16]}"
    with psycopg.connect(ADMIN, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target,)).fetchone()
        others = conn.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE datname=%s", (a.source,)
        ).fetchone()[0]
        plan = {"source": a.source, "target": target, "sessions": sessions,
                "source_digest": digest, "exists": bool(exists), "source_connections": others}
        if not a.commit or exists:
            print(json.dumps(plan, indent=2))
            if exists:
                print(json.dumps(memory_identity(target, a.env), indent=2))
            return
        if others:
            raise SystemExit(f"Source {a.source} has {others} connections; retry when idle")
        conn.execute(sql.SQL("CREATE DATABASE {} TEMPLATE {} STRATEGY WAL_LOG").format(
            sql.Identifier(target), sql.Identifier(a.source)))
        conn.execute(sql.SQL("ALTER DATABASE {} SET default_transaction_read_only = on").format(
            sql.Identifier(target)))
    identity = memory_identity(target, a.env)
    if identity["memory_snapshot"] != digest:
        raise SystemExit("Clone content digest differs from the source")
    print(json.dumps({**plan, "created": True, "identity": identity}, indent=2))


if __name__ == "__main__":
    main()
