"""TRACE v0.7.0 TRAIN ledger: every configuration reserves its OWN stratified sample (Amendment A.4).

``SampledLedger`` = the hashed ``SearchLedger`` with one change: ``reserve`` on a v0.7.0 TRAIN manifest inserts the
configuration's own sample (``trace_v070_samples.config_sample``) instead of one frozen panel, and records it in
a ``samples`` table (key, ordered task list, sha256). Every other state transition (claim / finish / fail / retry
/ prune / pending / complete_configs) is the stock code. On a TEST manifest, or a manifest without a v0.7.0
sampling block, ``reserve`` is the stock one (TEST is one fixed 200-task panel for every configuration).

``samples.json`` (written by ``write_samples_json``) mirrors the table plus the NoMemory baseline's sample.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_schema import digest

from .trace_v070_samples import config_sample, sample_sha

SAMPLES_DDL = "CREATE TABLE IF NOT EXISTS samples(key TEXT PRIMARY KEY, tasks TEXT, sha256 TEXT, created REAL)"
DECISIONS_DDL = ("CREATE TABLE IF NOT EXISTS stage_decisions(key TEXT, stage INTEGER, body TEXT, created REAL, "
                 "PRIMARY KEY(key, stage))")


def sampling_of(manifest) -> dict | None:
    """The v0.7.0 sampling block of a TRAIN manifest, else None."""
    if manifest.get("evaluation_split", "TRAIN") != "TRAIN":
        return None
    return (manifest.get("v070") or {}).get("sampling")


def own_tasks(manifest, key: str) -> list[str]:
    block = sampling_of(manifest)
    if block is None:
        raise ValueError("own_tasks needs a v0.7.0 TRAIN manifest")
    return config_sample(block, key)


class SampledLedger(SearchLedger):
    def reserve(self, space, configs):
        block = sampling_of(self.manifest)
        if block is None:
            return super().reserve(space, configs)
        if digest(space.manifest()) != digest(self.manifest["space"]):
            raise ValueError("Capability/execution manifest mismatch")
        pool = {str(t) for t in self.manifest["tasks"]}
        incoming = {space.key(c): space.canonicalize(c) for c in configs}
        self.db.execute(SAMPLES_DDL)
        try:
            self.db.execute("BEGIN IMMEDIATE")
            existing = {r[0] for r in self.db.execute("SELECT key FROM configs")}
            new = {k: c for k, c in incoming.items() if k not in existing}
            if len(existing) + len(new) > self.manifest["budget"]:
                raise ValueError("Configuration budget exhausted")
            for key, c in new.items():
                tasks = config_sample(block, key)
                if not set(tasks) <= pool or len(tasks) != int(block["size"]):
                    raise ValueError("Configuration sample outside the frozen TRAIN pool")
                self.db.execute("INSERT INTO configs VALUES(?,?,?)", (key, json.dumps(c), time.time()))
                self.db.executemany(
                    "INSERT INTO episodes(config,task,state,receipt) VALUES(?,?,?,NULL)",
                    [(key, t, "ready") for t in tasks],
                )
                self.db.execute("INSERT INTO samples VALUES(?,?,?,?)",
                                (key, json.dumps(tasks), sample_sha(tasks), time.time()))
            self.db.commit()
            return list(new)
        except Exception:
            self.db.rollback()
            raise

    def import_reserved(self, space, config, receipts) -> str:
        """Reserve one configuration and attach its imported done receipts (arm import).

        The receipts must cover exactly the configuration's own sample in this run."""
        key = space.key(config)
        self.reserve(space, [config])
        tasks = self.sample_of(key)
        if sorted(str(r["task_id"]) for r in receipts) != sorted(tasks):
            raise ValueError(f"Imported receipts of {key[:12]} do not cover its own sample")
        for r in receipts:
            self.claim(key, r["task_id"])
            self.finish(key, r["task_id"], r)
        return key

    def apply_stage_decisions(self, stage: int, decisions: dict, prune: dict) -> dict:
        """One checkpoint, one transaction: record every decision of ``stage`` (key -> JSON-able dict) and prune the
        ``prune`` keys (key -> reason). Fix round 1: the ledger is the durable record (commit first, log after);
        a (key, stage) decision already recorded is never recomputed or rewritten."""
        self.db.execute(DECISIONS_DDL)
        try:
            self.db.execute("BEGIN IMMEDIATE")
            for key, body in decisions.items():
                self.db.execute("INSERT OR IGNORE INTO stage_decisions VALUES(?,?,?,?)",
                                (key, int(stage), json.dumps(body, default=list, sort_keys=True), time.time()))
            out = {k: self.prune(k, r) for k, r in prune.items()}
            self.db.commit()
            return out
        except Exception:
            self.db.rollback()
            raise

    def stage_decisions(self, stage: int | None = None) -> dict:
        """{(key, stage): decision} recorded so far (optionally one stage)."""
        self.db.execute(DECISIONS_DDL)
        sql, args = "SELECT key, stage, body FROM stage_decisions", ()
        if stage is not None:
            sql, args = sql + " WHERE stage=?", (int(stage),)
        return {(k, s): json.loads(b) for k, s, b in self.db.execute(sql, args)}

    def sample_of(self, key: str) -> list[str] | None:
        self.db.execute(SAMPLES_DDL)
        row = self.db.execute("SELECT tasks FROM samples WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def samples(self) -> dict[str, list[str]]:
        self.db.execute(SAMPLES_DDL)
        return {k: json.loads(t) for k, t in self.db.execute("SELECT key, tasks FROM samples ORDER BY created, key")}


def read_samples(ledger_path) -> dict[str, list[str]]:
    """Read-only view of a ledger's samples table (empty when the table does not exist)."""
    db = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True, timeout=30)
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='samples'").fetchone():
            return {}
        return {k: json.loads(t) for k, t in db.execute("SELECT key, tasks FROM samples ORDER BY created, key")}
    finally:
        db.close()


def write_samples_json(run_dir, manifest) -> Path:
    """<run>/samples.json: every configuration's own sample (ledger table) + the NoMemory sample."""
    from .trace_v070_baseline import baseline_dir, baseline_key

    run_dir = Path(run_dir)
    configs = read_samples(run_dir / "ledger.sqlite") if (run_dir / "ledger.sqlite").exists() else {}
    block = sampling_of(manifest)
    for key, tasks in configs.items():
        if block is not None and tasks != config_sample(block, key):
            raise ValueError(f"Ledger sample of {key[:12]} differs from its deterministic draw")
    body = {
        "policy": block["policy"] if block else None,
        "seed": block["seed"] if block else None,
        "size": block["size"] if block else None,
        "quota": block["quota"] if block else None,
        "configs": {k: {"tasks": t, "sha256": sample_sha(t)} for k, t in configs.items()},
        "baselines": {},
    }
    bm = baseline_dir(run_dir) / "manifest.json"
    if bm.exists():
        tasks = json.loads(bm.read_text())["tasks"]
        body["baselines"]["NoMemory"] = {"key": baseline_key(manifest), "tasks": tasks, "sha256": sample_sha(tasks)}
    path = run_dir / "samples.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=1) + "\n")
    os.replace(tmp, path)
    return path
