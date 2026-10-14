"""Native graph publication and nonblocking retrieval; no relational edge tables."""

from __future__ import annotations

import json
import time

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from trimem.experiments.exg_pilot import MODEL, QUERY, inspect_plan
from trimem.exg.runtime import read, save


class Graph:
    def __init__(self, admin_dsn, name, arm, hint_token_limit=8192):
        self.arm = arm
        self.hint_token_limit = hint_token_limit
        self.native = arm != "V"
        self.semantic = arm.startswith("S")
        with psycopg.connect(admin_dsn, autocommit=True) as c:
            if not c.execute(
                "SELECT 1 FROM pg_database WHERE datname=%s", (name,)
            ).fetchone():
                c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        self.dsn = admin_dsn + f" dbname={name}"
        self.c = psycopg.connect(self.dsn, autocommit=True)
        # Prevent two resumed processes from mutating the same cell.
        if not self.c.execute("SELECT pg_try_advisory_lock(260517721)").fetchone()[0]:
            raise RuntimeError(f"Another writer owns {name}")
        for ext in ("vector", "graph_store_am", "tjs_pg", "trimem_pg"):
            self.c.execute(
                sql.SQL("CREATE EXTENSION IF NOT EXISTS {}").format(sql.Identifier(ext))
            )
        self.c.execute("CREATE SEQUENCE IF NOT EXISTS vector_case_id")
        self.c.execute(
            "CREATE TABLE IF NOT EXISTS anchors(task_id text PRIMARY KEY, vid bigint NOT NULL)"
        )
        self.c.execute(
            "CREATE TABLE IF NOT EXISTS cases(vid bigint PRIMARY KEY, uid text UNIQUE NOT NULL, task_id text NOT NULL, sequence int NOT NULL, attempt int NOT NULL, passed bool NOT NULL, hint text NOT NULL, embedding vector(1024) NOT NULL, receipt jsonb NOT NULL, UNIQUE(task_id,attempt))"
        )
        self.c.execute(
            "CREATE INDEX IF NOT EXISTS cases_ann ON cases USING hnsw(embedding vector_cosine_ops)"
        )
        self.types = {
            t: self.c.execute(
                "SELECT graph_store.register_edge_type(%s)", (t,)
            ).fetchone()[0]
            for t in ("contain", "similar_to", "fixed_by")
        }
        self.settings = {
            "enable_seqscan": "off",
            "enable_hashjoin": "off",
            "enable_mergejoin": "off",
            "enable_material": "off",
            "jit": "off",
            "hnsw.ef_search": "100",
            "max_parallel_workers_per_gather": "0",
            "work_mem": "16MB",
            "trimem.placement": {"S0": "base", "S1": "top"}.get(arm, "auto"),
            "trimem.min_batch": "1",
            "trimem.max_batch": "1",
            "trimem.timeout_ms": "1800000",
            "statement_timeout": "3600000",
            "trimem.prompt_version": name,
            "trimem.model": MODEL,
        }
        for setting, value in self.settings.items():
            self.c.execute("SELECT set_config(%s,%s,false)", (setting, value))

    def retrieve(self, vector, sequence, question, calls, path, worker=None):
        if path.exists():
            return read(path)
        if self.arm == "V":
            query = "SELECT vid,hint FROM cases WHERE sequence <= %s ORDER BY embedding <=> %s::vector LIMIT 30"
            params = [sequence, vector]
        else:
            query = QUERY.replace("c.sequence < %s", "c.sequence <= %s").format(
                semantic="AND trimem.sem_filter(%s,c.hint)" if self.semantic else ""
            )
            params = [vector, sequence]
            if self.semantic:
                params.append(
                    "This past attempt provides useful transferable experience for solving: "
                    + question
                )
        plan = self.c.execute("EXPLAIN (FORMAT JSON) " + query, params).fetchone()[0][0]
        inspect_plan(plan["Plan"])
        save(
            path.with_suffix(".sql.json"),
            dict(query=query, parameters=params, settings=self.settings, plan=plan),
        )
        w_before = worker.stats.requests if worker else 0
        hints, vids, seen, rows = [], [], set(), 0
        accepted_hint_tokens = 0
        skipped_for_token_budget = []
        skipped_token_details = []
        start = time.perf_counter()
        token_seconds = 0.0
        with self.c.transaction(), self.c.cursor(name="hint_stream") as cur:
            cur.execute(query, params)
            while len(hints) < 5:
                row = cur.fetchone()
                if row is None:
                    break
                rows += 1
                vid, hint = row
                if vid in seen:
                    continue
                seen.add(vid)
                ts = time.perf_counter()
                hint_tokens = calls.tokens(
                    text=json.dumps(hints + [hint], ensure_ascii=False)
                )
                token_seconds += time.perf_counter() - ts
                if hint_tokens > self.hint_token_limit:
                    skipped_for_token_budget.append(vid)
                    skipped_token_details.append(
                        dict(
                            vid=vid,
                            combined_hint_tokens=hint_tokens,
                            excess_tokens=hint_tokens - self.hint_token_limit,
                        )
                    )
                    continue
                hints.append(hint)
                vids.append(vid)
                accepted_hint_tokens = hint_tokens
        receipt = dict(
            hints=hints,
            vids=vids,
            returned_rows=rows,
            unique_returned=len(seen),
            hint_token_limit=self.hint_token_limit,
            accepted_hint_tokens=accepted_hint_tokens,
            skipped_for_token_budget=skipped_for_token_budget,
            skipped_for_token_budget_count=len(skipped_for_token_budget),
            skipped_token_details=skipped_token_details,
            seconds=time.perf_counter() - start,
            tokenization_seconds=token_seconds,
            worker_requests=(worker.stats.requests - w_before) if worker else 0,
            plan=plan,
            stage="retrieval",
            edge_work="not instrumented by plain EXPLAIN",
        )
        save(path, receipt)
        return receipt

    def publish(self, case, vector, lesson=None, source_neighbors=None):
        """Atomic and idempotent across both PG commit and receipt-file crash windows."""
        uid = f"{case['task_id']}:{case['attempt']}"
        old = self.c.execute(
            "SELECT receipt FROM cases WHERE uid=%s", (uid,)
        ).fetchone()
        if old:
            assert old[0]["case"] == case
            return old[0]
        start = time.perf_counter()
        with self.c.transaction():
            anchor = None
            if self.native:
                row = self.c.execute(
                    "SELECT vid FROM anchors WHERE task_id=%s", (case["task_id"],)
                ).fetchone()
                if row:
                    anchor = row[0]
                else:
                    anchor = self.c.execute(
                        "SELECT graph_store.gph_insert_vertex()"
                    ).fetchone()[0]
                    self.c.execute(
                        "INSERT INTO anchors VALUES(%s,%s)", (case["task_id"], anchor)
                    )
            vid = self.c.execute(
                "SELECT graph_store.gph_insert_vertex()"
                if self.native
                else "SELECT nextval('vector_case_id')"
            ).fetchone()[0]
            if source_neighbors is None and self.native:
                neighbors = self.c.execute(
                    "SELECT uid FROM cases ORDER BY embedding <=> %s::vector LIMIT 10",
                    (vector,),
                ).fetchall()
                source_neighbors = [r[0] for r in neighbors]
            source_neighbors = source_neighbors or []
            payload = dict(
                question=case["question"], answer=case["answer"], correct=case["passed"]
            )
            if lesson is not None:
                payload["lesson"] = lesson
            hint = json.dumps(payload, ensure_ascii=False)
            # Metadata is in the same transaction as native graph topology.
            self.c.execute(
                "INSERT INTO cases VALUES(%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)",
                (
                    vid,
                    uid,
                    case["task_id"],
                    case["sequence"],
                    case["attempt"],
                    case["passed"],
                    hint,
                    vector,
                    Jsonb({}),
                ),
            )
            edges = []
            if self.native:
                edges.append((anchor, vid, "contain"))
                for neighbor_uid in source_neighbors:
                    other = self.c.execute(
                        "SELECT vid FROM cases WHERE uid=%s", (neighbor_uid,)
                    ).fetchone()
                    if other is None:
                        raise RuntimeError(f"Missing source neighbor {neighbor_uid}")
                    edges.extend(
                        [(vid, other[0], "similar_to"), (other[0], vid, "similar_to")]
                    )
                if case["attempt"] == 2 and case["passed"]:
                    prior = self.c.execute(
                        "SELECT vid,passed FROM cases WHERE task_id=%s AND attempt=1",
                        (case["task_id"],),
                    ).fetchone()
                    assert prior and not prior[1]
                    edges.append((prior[0], vid, "fixed_by"))
                for src, dst, kind in edges:
                    self.c.execute(
                        "SELECT graph_store.gph_insert_edge(%s,%s,%s)",
                        (src, dst, self.types[kind]),
                    )
            receipt = dict(
                case=case,
                uid=uid,
                vid=vid,
                anchor=anchor,
                vector=vector,
                lesson=lesson,
                neighbors=source_neighbors,
                edges=edges,
                seconds=time.perf_counter() - start,
                commit_timing="SQL work through receipt update; transaction commit measured by caller",
            )
            self.c.execute(
                "UPDATE cases SET receipt=%s WHERE vid=%s", (Jsonb(receipt), vid)
            )
        return receipt

    def fingerprint(self):
        # Audit outside the measured retrieval path, with bounded server cursor memory.
        import hashlib

        sha = hashlib.sha256()
        count = 0
        with self.c.transaction(), self.c.cursor(name="snapshot") as cur:
            cur.execute(
                "SELECT uid,hint,embedding::text,receipt FROM cases ORDER BY vid"
            )
            for row in cur:
                sha.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
                count += 1
        return dict(
            cases=count,
            sha256=sha.hexdigest(),
            native_vertices=self.c.execute(
                "SELECT graph_store.gph_vertex_count()"
            ).fetchone()[0]
            if self.native
            else 0,
            physical_edges=self.c.execute(
                "SELECT graph_store.gph_edge_count()"
            ).fetchone()[0]
            if self.native
            else 0,
        )
