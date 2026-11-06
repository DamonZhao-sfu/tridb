"""Read-only SWE-QA session adapter. No gold answers enter this module."""

from __future__ import annotations

import hashlib
import re
import time

import psycopg
from psycopg.rows import dict_row

DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa"


def repo_family(name):
    # SFT has basenames, not verified owner/repo identities. Conservative L1:
    # exclude matching normalized basenames, including possible name collisions.
    return re.sub(r"[-_.]", "", str(name).rstrip("/").split("/")[-1].lower())


class SWEQASessionAdapter:
    def __init__(self, vector, repository, task_id=None, dsn=DSN):
        self.vector, self.family, self.task_id = (
            vector,
            repo_family(repository),
            task_id,
        )
        self.conn = psycopg.connect(dsn, row_factory=dict_row)
        self.conn.execute("SET TRANSACTION READ ONLY")
        for name in (
            "enable_hashjoin",
            "enable_mergejoin",
            "enable_material",
            "enable_memoize",
            "enable_seqscan",
            "enable_sort",
            "enable_bitmapscan",
        ):
            self.conn.execute(f"SET LOCAL {name}=off")
        self.conn.execute("SET LOCAL max_parallel_workers_per_gather=0")
        self.conn.execute("SET LOCAL statement_timeout='60s'")
        self.conn.execute("SET LOCAL hnsw.ef_search=1000")
        self.sql_receipts = []
        self.membership_paths = []

    def close(self):
        self.conn.close()

    def _read(self, sql, params=()):
        start = time.monotonic()
        event = {
            "sql": sql,
            "parameters": [
                {"sha256": hashlib.sha256(p.encode()).hexdigest(), "characters": len(p)}
                if isinstance(p, str) and len(p) > 2000
                else p
                for p in params
            ],
        }

        def walk(node):
            yield node
            for child in node.get("Plans", []):
                yield from walk(child)

        blocked = {
            "Sort",
            "Incremental Sort",
            "Materialize",
            "Hash",
            "Hash Join",
            "Aggregate",
            "Function Scan",
        }
        try:
            plan = self.conn.execute("EXPLAIN (FORMAT JSON) " + sql, params).fetchone()[
                "QUERY PLAN"
            ]
            event["plan"] = plan
            if any(n["Node Type"] in blocked for n in walk(plan[0]["Plan"])):
                raise RuntimeError(
                    "Non-streaming database operator in session retrieval"
                )
            rows = self.conn.execute(sql, params).fetchall()
            event["rows"] = len(rows)
            return rows
        except Exception as exc:
            event["error"] = repr(exc)
            raise
        finally:
            event["wall_ms"] = (time.monotonic() - start) * 1000
            self.sql_receipts.append(event)

    def _session(self, sid=None, vid=None):
        key, value = ("id", sid) if sid is not None else ("vid", vid)
        rows = self._read(
            f"SELECT s.id,s.vid,s.task_id,t.repo_label,t.question,"
            f"1-(s.emb <=> %s::vector) cosine FROM session s "
            f"JOIN task t ON t.id=s.task_id WHERE s.{key}=%s",
            (self.vector, value),
        )
        return rows[0] if rows else None

    def allowed(self, session):
        return (
            session["task_id"] != self.task_id
            and repo_family(session["repo_label"]) != self.family
        )

    def seeds(self, count):
        # Adaptive ANN over-fetch after conservative L1 filtering. Receipt exposes
        # underfill if the ANN index exhausts its candidates. Never weaken L1.
        limit = max(32, count * 2)
        while True:
            rows = self._read(
                "SELECT id FROM session ORDER BY emb <=> %s::vector LIMIT %s",
                (self.vector, limit),
            )
            found = []
            for row in rows:
                s = self._session(sid=row["id"])
                if self.allowed(s):
                    found.append(s)
                if len(found) == count:
                    return found
            if len(rows) < limit or limit >= 1024:
                return found
            limit *= 2

    def neighbors(self, session, edge, fanout):
        if edge not in {"similar", "similar_up"}:
            raise ValueError("SWE-QA graph does not have the requested session edge")
        rows = self._read(
            "SELECT dst FROM trimem.expand(%s,%s,%s)", (session["vid"], edge, fanout)
        )
        return [s for row in rows if (s := self._session(vid=row["dst"])) is not None]

    def hydrate(self, session):
        stored = self._read(
            "SELECT final_answer,n_steps,model FROM session WHERE id=%s",
            (session["id"],),
        )[0]
        # One extra edge probes stale-low metadata instead of silently certifying
        # a truncated session. The frozen corpus must match its stored degree.
        nodes = self._read(
            "SELECT dst FROM trimem.expand(%s,'has_step',%s)",
            (session["vid"], stored["n_steps"] + 1),
        )
        if len(nodes) != stored["n_steps"] or len({n["dst"] for n in nodes}) != len(
            nodes
        ):
            raise ValueError("Incomplete native session membership")
        steps = []
        for node in nodes:
            row = self._read(
                "SELECT s.id,s.session_id,s.ordinal,s.kind,s.intent,tc.name tool,"
                "tc.arguments,o.content observation FROM step s "
                "LEFT JOIN tool_call tc ON tc.step_id=s.id "
                "LEFT JOIN tool_observation o ON o.step_id=s.id WHERE s.vid=%s",
                (node["dst"],),
            )[0]
            if row["session_id"] != session["id"]:
                raise ValueError("Membership ownership mismatch")
            self.membership_paths.append(
                {"src": session["vid"], "dst": node["dst"], "edge": "has_step"}
            )
            steps.append(row)
        if len(steps) != stored["n_steps"]:
            raise ValueError("Incomplete native session membership")
        steps.sort(key=lambda s: s["ordinal"])
        return {
            **session,
            **stored,
            "steps": steps,
            "quality_label": "ungraded teacher trajectory, not verified gold",
        }
