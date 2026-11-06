"""Atomic finite-budget admission and at-most-once episode dispatch.

States: ready -> claimed -> done, or claimed -> failed -> ready (retry).
Staged pruning (manifests with ``pruning``): ready -> pruned. A pruned episode is never
dispatched; a configuration whose episodes are all done or pruned is complete, and its
observation uses only its done episodes (a partial panel).
A claimed episode is never retried implicitly: the driver either attaches its
durable external receipt through finish(), or marks it failed with fail() after
checking that no worker can still write it. Only failed episodes are retried,
and every claim increments ``attempts``.
"""

import json
import sqlite3
import time


class EpisodeNotStarted(RuntimeError):
    """Raised before any worker was launched: the claim returns to ready."""


class RetryableEpisodeError(RuntimeError):
    """The worker is confirmed terminated and left no result: safe to rerun."""

from .plan_schema import digest


class SearchLedger:
    def __init__(self, path, manifest):
        self.manifest = manifest
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(id INTEGER PRIMARY KEY, hash TEXT, body TEXT);
            CREATE TABLE IF NOT EXISTS configs(key TEXT PRIMARY KEY, body TEXT, created REAL);
            CREATE TABLE IF NOT EXISTS episodes(config TEXT, task TEXT, state TEXT, receipt TEXT,
                PRIMARY KEY(config,task), FOREIGN KEY(config) REFERENCES configs(key));
        """)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(episodes)")}
        for name, ddl in (
            ("attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("error", "TEXT"),
            ("retryable", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns:
                self.db.execute(f"ALTER TABLE episodes ADD COLUMN {name} {ddl}")
        try:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute("SELECT hash FROM metadata WHERE id=1").fetchone()
            if old and old[0] != digest(manifest):
                raise ValueError("Frozen manifest mismatch; use a new run directory")
            if not old:
                self.db.execute(
                    "INSERT INTO metadata VALUES(1,?,?)",
                    (digest(manifest), json.dumps(manifest)),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def reserve(self, space, configs):
        if digest(space.manifest()) != digest(self.manifest["space"]):
            raise ValueError("Capability/execution manifest mismatch")
        incoming = {space.key(c): space.canonicalize(c) for c in configs}
        try:
            self.db.execute("BEGIN IMMEDIATE")
            existing = {r[0] for r in self.db.execute("SELECT key FROM configs")}
            new = {k: c for k, c in incoming.items() if k not in existing}
            if len(existing) + len(new) > self.manifest["budget"]:
                raise ValueError("Configuration budget exhausted")
            for key, c in new.items():
                self.db.execute(
                    "INSERT INTO configs VALUES(?,?,?)",
                    (key, json.dumps(c), time.time()),
                )
                self.db.executemany(
                    "INSERT INTO episodes(config,task,state,receipt) VALUES(?,?,?,NULL)",
                    [(key, str(t), "ready") for t in self.manifest["tasks"]],
                )
            self.db.commit()
            return list(new)
        except Exception:
            self.db.rollback()
            raise

    def claim(self, key, task):
        result = self.db.execute(
            "UPDATE episodes SET state='claimed',attempts=attempts+1 "
            "WHERE config=? AND task=? AND state='ready'",
            (key, str(task)),
        )
        return result.rowcount == 1

    def finish(self, key, task, receipt):
        if str(receipt.get("task_id")) != str(task):
            raise ValueError("Receipt task identity mismatch")
        body = json.dumps(receipt, allow_nan=False, sort_keys=True)
        result = self.db.execute(
            "UPDATE episodes SET state='done',receipt=? WHERE config=? AND task=? AND state='claimed'",
            (body, key, str(task)),
        )
        if result.rowcount != 1:
            old = self.db.execute(
                "SELECT receipt FROM episodes WHERE config=? AND task=?",
                (key, str(task)),
            ).fetchone()
            if not old or old[0] != body:
                raise ValueError("Unclaimed episode or conflicting durable receipt")

    def fail(self, key, task, error, *, retryable=False):
        """claimed -> failed. ``retryable`` only when the attempt is confirmed
        terminated without a result; uncertain or completed attempts never are."""
        result = self.db.execute(
            "UPDATE episodes SET state='failed',error=?,retryable=? "
            "WHERE config=? AND task=? AND state='claimed'",
            (str(error)[:4000], int(bool(retryable)), key, str(task)),
        )
        if result.rowcount != 1:
            raise ValueError("Only a claimed episode can fail")

    def release(self, key, task):
        """claimed -> ready for an attempt that provably never launched."""
        result = self.db.execute(
            "UPDATE episodes SET state='ready',attempts=max(attempts-1,0) "
            "WHERE config=? AND task=? AND state='claimed'",
            (key, str(task)),
        )
        if result.rowcount != 1:
            raise ValueError("Only a claimed episode can be released")

    def retry(self, *, max_attempts=None, include_uncertain=False):
        """failed -> ready. Automatic retry: retryable rows below max_attempts.
        ``include_uncertain`` is the operator's explicit decision for the rest."""
        sql = "UPDATE episodes SET state='ready',retryable=0 WHERE state='failed'"
        args = []
        if not include_uncertain:
            sql += " AND retryable=1"
        if max_attempts is not None:
            sql += " AND attempts < ?"
            args.append(int(max_attempts))
        return self.db.execute(sql, args).rowcount

    def episodes(self, state):
        return [
            {"key": k, "config": json.loads(c), "task_id": t, "attempts": a,
             "error": e, "retryable": bool(r)}
            for k, c, t, a, e, r in self.db.execute(
                "SELECT c.key,c.body,e.task,e.attempts,e.error,e.retryable FROM configs c "
                "JOIN episodes e ON e.config=c.key WHERE e.state=? ORDER BY c.created,e.task",
                (state,),
            )
        ]

    def complete_configs(self):
        rows = self.db.execute(
            "SELECT c.key,c.body FROM configs c WHERE NOT EXISTS "
            "(SELECT 1 FROM episodes e WHERE e.config=c.key AND e.state NOT IN ('done','pruned')) "
            "ORDER BY c.created,c.key"
        )
        return [
            {
                "key": k,
                "config": json.loads(c),
                "receipts": self.receipts(k),
                "pruned": self.db.execute(
                    "SELECT count(*) FROM episodes WHERE config=? AND state='pruned'", (k,)
                ).fetchone()[0] > 0,
            }
            for k, c in rows
        ]

    def receipts(self, key):
        """Done receipts of one configuration (any completion state)."""
        return [
            json.loads(r[0])
            for r in self.db.execute(
                "SELECT receipt FROM episodes WHERE config=? AND state='done' ORDER BY task",
                (key,),
            )
        ]

    def prune(self, key, reason):
        """ready/failed -> pruned for one configuration (claimed episodes are never touched; the
        caller prunes only after every episode of the checkpoint prefix is done)."""
        return self.db.execute(
            "UPDATE episodes SET state='pruned',error=? WHERE config=? AND state IN ('ready','failed')",
            (str(reason)[:4000], key),
        ).rowcount

    def prune_many(self, decisions):
        """Apply one checkpoint's prune decisions atomically: {key: reason} -> skipped episodes."""
        try:
            self.db.execute("BEGIN IMMEDIATE")
            out = {k: self.prune(k, r) for k, r in decisions.items()}
            self.db.commit()
            return out
        except Exception:
            self.db.rollback()
            raise

    def pending(self):
        return [
            {"key": k, "config": json.loads(c), "task_id": t, "state": state}
            for k, c, t, state in self.db.execute(
                "SELECT c.key,c.body,e.task,e.state FROM configs c JOIN episodes e ON e.config=c.key "
                "WHERE e.state NOT IN ('done','pruned') ORDER BY c.created,e.task"
            )
        ]
