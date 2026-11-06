"""Durable per-retrieval opt events, independent of the frozen memory corpus.

Test and smoke events are archived but never appear in the BO training view.
Quality stays NULL until downstream generation and all judge rounds complete.
"""

import hashlib
import json
import time
import uuid
from dataclasses import asdict

import psycopg
from psycopg.types.json import Jsonb

from trimem.experiments.session_retrieval import retrieve

DDL = """
CREATE SCHEMA IF NOT EXISTS opt;
CREATE TABLE IF NOT EXISTS opt.session_retrieval (
 retrieval_id uuid PRIMARY KEY,
 experiment_id text NOT NULL,
 dataset text NOT NULL,
 split text NOT NULL CHECK (split IN ('train','test','smoke')),
 query_id text NOT NULL,
 query_context jsonb NOT NULL,
 config jsonb NOT NULL,
 config_hash text NOT NULL,
 protocol jsonb NOT NULL,
 status text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now(),
 retrieval_ms double precision,
 receipt jsonb,
 error text,
 generation jsonb,
 judge jsonb,
 quality double precision,
 paired_gain double precision,
 total_tokens bigint,
 e2e_ms double precision
);
CREATE INDEX IF NOT EXISTS session_retrieval_experiment_idx
 ON opt.session_retrieval(experiment_id,split,query_id);
CREATE TABLE IF NOT EXISTS opt.session_judge_round (
 retrieval_id uuid NOT NULL REFERENCES opt.session_retrieval,
 round int NOT NULL CHECK (round BETWEEN 1 AND 3),
 result jsonb NOT NULL,
 recorded_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY (retrieval_id,round)
);
CREATE OR REPLACE VIEW opt.session_profile_training AS
 SELECT * FROM opt.session_retrieval WHERE split='train' AND status='evaluated'
 AND quality IS NOT NULL AND paired_gain IS NOT NULL
 AND total_tokens IS NOT NULL AND e2e_ms IS NOT NULL;
"""


class SessionProfileStore:
    def __init__(self, dsn):
        self.dsn = dsn
        with psycopg.connect(dsn) as conn:
            if conn.info.port in {55440, 55441}:
                raise ValueError("Refusing optimizer writes to shared service cluster")
            # Two arm supervisors may initialize simultaneously. PostgreSQL
            # IF NOT EXISTS alone does not serialize system-catalog mutations.
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('trimem.opt.session_schema_v1'))"
            )
            conn.execute(DDL)

    def start(
        self,
        *,
        experiment_id,
        dataset,
        split,
        query_id,
        query_context,
        config,
        protocol,
    ):
        rid = str(uuid.uuid4())
        cfg = asdict(config)
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        with psycopg.connect(self.dsn) as conn:
            conn.execute(
                "INSERT INTO opt.session_retrieval "
                "(retrieval_id,experiment_id,dataset,split,query_id,query_context,config,"
                "config_hash,protocol,status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'retrieving')",
                (
                    rid,
                    experiment_id,
                    dataset,
                    split,
                    str(query_id),
                    Jsonb(query_context),
                    Jsonb(cfg),
                    digest,
                    Jsonb(protocol),
                ),
            )
        return rid

    def patch(self, rid, *, status, receipt=None, error=None, retrieval_ms=None):
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "UPDATE opt.session_retrieval SET status=%s,receipt=COALESCE(%s,receipt),error=%s,"
                "retrieval_ms=COALESCE(%s,retrieval_ms),updated_at=now() WHERE retrieval_id=%s RETURNING retrieval_id",
                (
                    status,
                    Jsonb(receipt) if receipt is not None else None,
                    error,
                    retrieval_ms,
                    rid,
                ),
            ).fetchone()
            if row is None:
                raise ValueError("Unknown retrieval ID")

    def generation(self, rid, result, total_tokens, e2e_ms):
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "UPDATE opt.session_retrieval SET status='generated_pending_judge',"
                "generation=%s,total_tokens=%s,e2e_ms=%s,updated_at=now() "
                "WHERE retrieval_id=%s RETURNING retrieval_id",
                (Jsonb(result), total_tokens, e2e_ms, rid),
            ).fetchone()
            if not row:
                raise ValueError("Missing retrieval row for generation")

    def judge_round(self, rid, result):
        with psycopg.connect(self.dsn) as conn:
            old = conn.execute(
                "SELECT result FROM opt.session_judge_round "
                "WHERE retrieval_id=%s AND round=%s",
                (rid, result["round"]),
            ).fetchone()
            if old:
                if old[0] != result:
                    raise ValueError("Cannot overwrite a persisted judge round")
                return
            conn.execute(
                "INSERT INTO opt.session_judge_round (retrieval_id,round,result) "
                "VALUES (%s,%s,%s)",
                (rid, result["round"], Jsonb(result)),
            )

    def verify_resume(self, rid, experiment_id, query_id, config):
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "SELECT experiment_id,query_id,config FROM opt.session_retrieval "
                "WHERE retrieval_id=%s",
                (rid,),
            ).fetchone()
            expected = (
                experiment_id,
                str(query_id),
                json.loads(json.dumps(asdict(config))),
            )
            if row is None or row != expected:
                raise ValueError(
                    "Filesystem receipt does not match the durable opt identity"
                )

    def evaluate(self, rid, judgments, baseline_quality=None):
        if len(judgments) != 3 or any(j.get("status") != "scored" for j in judgments):
            raise ValueError("Three successful independent judge rounds are required")
        if {j["round"] for j in judgments} != {1, 2, 3}:
            raise ValueError("Judge rounds must be distinct")
        values = [j["result"]["total_score"] / 50 for j in judgments]
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError("Invalid normalized quality")
        quality = sum(values) / 3
        gain = None if baseline_quality is None else quality - baseline_quality
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute(
                "UPDATE opt.session_retrieval SET status='evaluated',judge=%s,"
                "quality=%s,paired_gain=%s,updated_at=now() WHERE retrieval_id=%s "
                "AND generation IS NOT NULL RETURNING retrieval_id",
                (Jsonb(judgments), quality, gain, rid),
            ).fetchone()
            if not row:
                raise ValueError("Cannot score a missing or unfinished generation")


def recorded_retrieve(adapter, config, store, metadata, *, retrieval_id=None, **kwargs):
    rid = retrieval_id or store.start(config=config, **metadata)
    start = time.monotonic()
    try:
        receipt = retrieve(adapter, config, **kwargs)
        if hasattr(adapter, "enrich_receipt"):
            receipt = adapter.enrich_receipt(receipt)
        receipt["retrieval_id"] = rid
        receipt["sql_receipts"] = adapter.sql_receipts
        receipt["membership_paths"] = adapter.membership_paths
        elapsed = (time.monotonic() - start) * 1000
        receipt["retrieval_ms"] = elapsed
        store.patch(
            rid,
            status="retrieved_pending_evaluation",
            receipt=receipt,
            retrieval_ms=elapsed,
        )
        return receipt
    except Exception as exc:
        partial = {
            "sql_receipts": adapter.sql_receipts,
            "membership_paths": adapter.membership_paths,
            "task_membership_paths": getattr(adapter, "task_membership_paths", []),
            "seed_tasks": getattr(adapter, "seed_tasks", []),
            "semantic_calls": (
                getattr(kwargs.get("semfilter"), "calls", [])
                + getattr(kwargs.get("semrank"), "calls", [])
            ),
            "partial": True,
        }
        store.patch(
            rid,
            status="retrieval_failed",
            receipt=partial,
            error=repr(exc),
            retrieval_ms=(time.monotonic() - start) * 1000,
        )
        raise
