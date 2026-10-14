"""Load EvoTrace into a TriMEM trajectory store (v0.2, docs/trimem_trajectory_storage_v0.2.0.md).

Relational = properties / text / time; native graph = topology (golden rule 3):
  task, session, attempt (the single vector-bearing vertex table, id -> vid), prompt,
  prompt_system, llm_call, search_membership / search_scalar / search_checkpoint, edge_prop
  (property mirror + idempotency guard, never traversed), load_meta.
  Graph edge types: lineage, merge_parent, context, similar, each with an `_up` reverse.

    PYTHONPATH=. taskset -c 20-27 nice -n 10 python -m trimem.data.evotrace \
        --dsn "host=/tmp/trimem-traj port=55450 dbname=trimem_traj" --receipt results/traj_load.json

The production clusters (55440 / 55441) and the system cluster (5432) are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import psycopg
from psycopg.conninfo import conninfo_to_dict

from trimem.data import evotrace_graph as gr
from trimem.data import evotrace_text as tx

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "dataset/evotrace/normalized"
RAW = REPO / "dataset/evotrace/raw"
FORBIDDEN_PORTS = frozenset({5432, 55440, 55441})
FORBIDDEN_DATA_DIRS = (".tridb-pgdata-trimem",)
RELATIONAL = (
    "task",
    "session",
    "attempt",
    "prompt_system",
    "prompt",
    "llm_call",
    "search_membership",
    "search_scalar",
    "search_checkpoint",
)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS graph_store_am;
CREATE EXTENSION IF NOT EXISTS trimem_pg;
CREATE TABLE IF NOT EXISTS load_meta (
    only_row boolean PRIMARY KEY DEFAULT true CHECK (only_row),
    fingerprint text NOT NULL, revision text, graph_policy jsonb);
ALTER TABLE load_meta ADD COLUMN IF NOT EXISTS graph_policy jsonb;
CREATE TABLE IF NOT EXISTS task (
    task text PRIMARY KEY, domain text, family text, task_key text, specification text,
    emb vector(1024));                                   -- task_spec track (18)
CREATE TABLE IF NOT EXISTS session (
    session text PRIMARY KEY, task text NOT NULL REFERENCES task, backend text NOT NULL,
    domain text, model text, mode text, temperature real, search_algorithm text,
    configured_iterations integer, run_rel text NOT NULL, prompt_complete_ratio real,
    has_evaluator boolean, best_attempt integer);
CREATE TABLE IF NOT EXISTS attempt (
    id integer PRIMARY KEY, uid text UNIQUE NOT NULL, vid bigint,
    task text NOT NULL, session text NOT NULL REFERENCES session, backend text NOT NULL,
    program_id text NOT NULL, iteration integer NOT NULL, generation integer,
    is_valid boolean NOT NULL, fitness double precision, status text NOT NULL,
    judge text, exec_time double precision, memory_mib double precision, language text,
    error text, framework_changes text,   -- the dataset's own nodes.changes (diff summary)
    intent text,                          -- the LLM's own stated change intent
    intent_source text,                   -- response_prose | shinka_patch_description
    artifact_sha text, code text, cell text, parent_id integer, tokens integer,
    emb vector(1024),                     -- node_artifact track (code, first 2000 chars)
    intent_emb vector(1024));
CREATE TABLE IF NOT EXISTS prompt_system (system_sha text PRIMARY KEY, text text NOT NULL);
CREATE TABLE IF NOT EXISTS prompt (
    attempt_id integer PRIMARY KEY REFERENCES attempt, template text,
    system_sha text REFERENCES prompt_system, user_text text, response_text text,
    response_sha1 text, prompts_sha256 text, source text NOT NULL);
CREATE TABLE IF NOT EXISTS llm_call (
    session text NOT NULL REFERENCES session, seq integer NOT NULL,
    attempt_id integer REFERENCES attempt, link_method text, link_candidates integer[], purpose text NOT NULL,
    model text, temperature real, finish_reason text, prompt_tokens integer,
    completion_tokens integer, reasoning_tokens integer, error text, content text,
    reasoning text, content_sha1 text,
    system_sha text, user_text text,      -- shinka logs carry the prompt messages
    PRIMARY KEY (session, seq));
ALTER TABLE llm_call ADD COLUMN IF NOT EXISTS link_candidates integer[];
CREATE TABLE IF NOT EXISTS search_checkpoint (
    session text NOT NULL, iteration integer NOT NULL, PRIMARY KEY (session, iteration));
CREATE TABLE IF NOT EXISTS search_membership (
    session text NOT NULL, role text NOT NULL, slot_key text, attempt_id integer,
    from_iter integer NOT NULL, to_iter integer NOT NULL, value jsonb);
CREATE TABLE IF NOT EXISTS search_scalar (
    session text NOT NULL, iteration integer, key text NOT NULL, value jsonb,
    attempt_id integer);
CREATE TABLE IF NOT EXISTS edge_prop (     -- audit/property mirror; NEVER traversed
    src integer NOT NULL, dst integer NOT NULL, edge_type integer NOT NULL,
    label text, ordinal integer, weight double precision, provenance text,
    PRIMARY KEY (src, dst, edge_type));
"""

INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS attempt_vid_idx ON attempt(vid);
CREATE INDEX IF NOT EXISTS attempt_task_judge_idx ON attempt(task, judge);
CREATE INDEX IF NOT EXISTS attempt_session_iter_idx ON attempt(session, iteration);
CREATE INDEX IF NOT EXISTS attempt_cell_idx ON attempt(task, cell);
CREATE INDEX IF NOT EXISTS llm_call_attempt_idx ON llm_call(attempt_id);
CREATE INDEX IF NOT EXISTS search_membership_asof_idx
    ON search_membership(session, from_iter, to_iter);
CREATE INDEX IF NOT EXISTS search_scalar_idx ON search_scalar(session, iteration);
"""

HNSW = """
CREATE INDEX IF NOT EXISTS attempt_hnsw ON attempt USING hnsw (emb vector_cosine_ops);
CREATE INDEX IF NOT EXISTS attempt_intent_hnsw ON attempt USING hnsw (intent_emb vector_cosine_ops);
CREATE INDEX IF NOT EXISTS task_hnsw ON task USING hnsw (emb vector_cosine_ops);
"""

# Reconstructed "considered but not selected" set for one attempt: the pool at the session's
# previous checkpoint minus the attempt itself, its parent and its context / merge sources (read
# from the NATIVE graph). The selection policy is not recorded; this is a reconstruction.
NOT_SELECTED = """
CREATE OR REPLACE FUNCTION not_selected(p_attempt integer)
RETURNS TABLE (candidate_id integer, role text, slot_key text, checkpoint integer)
LANGUAGE sql STABLE AS $$
    WITH a AS (SELECT id, vid, session, iteration, parent_id FROM attempt WHERE id = p_attempt),
    ck AS (SELECT max(c.iteration) AS it FROM search_checkpoint c, a
           WHERE c.session = a.session AND c.iteration < a.iteration),
    src AS (SELECT (graph_store.gph_traverse_typed(a.vid, t.id, 0, -1)).dst AS vid
            FROM a, graph_store.edge_type t WHERE t.name IN ('context_up', 'merge_parent_up'))
    SELECT DISTINCT m.attempt_id, m.role, m.slot_key, ck.it
    FROM a, ck, search_membership m
    WHERE m.session = a.session AND ck.it BETWEEN m.from_iter AND m.to_iter
      AND m.attempt_id IS NOT NULL AND m.attempt_id <> a.id
      AND m.attempt_id IS DISTINCT FROM a.parent_id
      AND NOT EXISTS (SELECT 1 FROM src JOIN attempt c ON c.vid = src.vid
                      WHERE c.id = m.attempt_id)
$$;
"""


# --------------------------------------------------------------------------- guards / io


def guard_dsn(dsn: str) -> dict:
    """Refuse production / system clusters before connecting. Port defaults follow libpq."""
    import os

    params = conninfo_to_dict(dsn)
    port = int(params.get("port") or os.environ.get("PGPORT") or 5432)
    if port in FORBIDDEN_PORTS:
        raise SystemExit(
            f"refusing to load into port {port}: production/system cluster"
        )
    return params


def connect(dsn: str) -> psycopg.Connection:
    guard_dsn(dsn)
    conn = psycopg.connect(dsn, autocommit=True)
    port = int(conn.execute("SELECT current_setting('port')").fetchone()[0])
    data_dir = conn.execute("SELECT current_setting('data_directory')").fetchone()[0]
    if port in FORBIDDEN_PORTS or any(d in data_dir for d in FORBIDDEN_DATA_DIRS):
        conn.close()
        raise SystemExit(f"refusing to load into {data_dir}:{port}: production cluster")
    return conn


def clean(x: object) -> object:
    """Postgres text cannot hold NUL; the dataset has a few."""
    return x.replace("\x00", "") if isinstance(x, str) else x


def _f(x: object) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError):
        return None
    return v if v is None or np.isfinite(v) else None


def vec(v: np.ndarray | None) -> str | None:
    return None if v is None else "[" + ",".join(map(repr, v.tolist())) + "]"


def jsonb(v: object) -> str | None:
    return None if v is None else json.dumps(v)


class Dataset:
    """The normalized files that are small enough to hold, plus per-session raw access."""

    def __init__(self, src: Path, raw: Path) -> None:
        self.src, self.raw = src, raw
        self.tasks = list(tx.read_jsonl(src / "tasks.jsonl"))
        self.sessions = list(tx.read_jsonl(src / "sessions.jsonl"))
        self.nodes = list(tx.read_jsonl(src / "nodes.jsonl"))
        self.ids = {n["node_uid"]: i + 1 for i, n in enumerate(self.nodes)}
        self.by_session: dict[str, list[dict]] = defaultdict(list)
        for n in self.nodes:
            self.by_session[n["session_uid"]].append(n)
        self.run_rel = {s["session_uid"]: s["run_rel"] for s in self.sessions}
        self.backend = {s["session_uid"]: s["backend"] for s in self.sessions}
        self.prompt_rel = {
            p["node_uid"]: p["blob_rel"] for p in tx.read_jsonl(src / "prompts.jsonl")
        }
        self.artifacts = {
            a["artifact_uid"]: a for a in tx.read_jsonl(src / "artifacts.jsonl")
        }
        z = np.load(src / "vectors.npz", allow_pickle=True)
        self.vectors = {
            (str(k), str(u)): v
            for u, k, v in zip(z["uids"], z["kinds"], z["vectors"], strict=True)
        }
        self.fingerprint = hashlib.sha256(
            (src / "nodes.jsonl").read_bytes()
        ).hexdigest()

    def run_dir(self, session: str) -> Path:
        return self.raw / self.run_rel[session]

    def raw_programs(self, session: str) -> dict[str, dict]:
        path = self.run_dir(session) / "programs.jsonl"
        return {r["id"]: r for r in tx.read_jsonl(path)} if path.is_file() else {}

    def code(self, node: dict) -> str | None:
        art = self.artifacts.get(node.get("artifact_uid") or "")
        if not art:
            return None
        candidates = [node["session_uid"], *art.get("sessions", [])]
        for s in candidates:
            p = self.run_dir(s) / art["blob_rel"] if s in self.run_rel else None
            if p is not None and p.is_file():
                return p.read_text(errors="replace")
        return None

    def texts(self) -> Iterator[tuple[dict, tx.NodeText | None]]:
        """(node, resolved prompt/response) in session order, one raw programs file at a time."""
        for s in self.run_rel:
            progs = self.raw_programs(s) if self.backend[s] == "shinkaevolve" else {}
            for n in self.by_session.get(s, []):
                yield (
                    n,
                    tx.node_text(
                        n,
                        self.run_dir(s),
                        progs.get(n["program_id"]),
                        self.prompt_rel.get(n["node_uid"]),
                    ),
                )


# --------------------------------------------------------------------------- relational core


def load_tasks_sessions(conn: psycopg.Connection, ds: Dataset) -> None:
    with conn.cursor().copy(
        "COPY task (task, domain, family, task_key, specification, emb) FROM STDIN"
    ) as cp:
        for t in ds.tasks:
            cp.write_row(
                (
                    t["task_uid"],
                    t.get("domain"),
                    t.get("task_family"),
                    t.get("task_key"),
                    clean(t.get("specification")),
                    vec(ds.vectors.get(("task", t["task_uid"]))),
                )
            )
    with conn.cursor().copy(
        "COPY session (session, task, backend, domain, model, mode, temperature, "
        "search_algorithm, configured_iterations, run_rel, prompt_complete_ratio, "
        "has_evaluator, best_attempt) FROM STDIN"
    ) as cp:
        for s in ds.sessions:
            files = s.get("files_present") or {}
            cp.write_row(
                (
                    s["session_uid"],
                    s["task_uid"],
                    s["backend"],
                    s.get("domain"),
                    s.get("model"),
                    s.get("mode"),
                    _f(s.get("temperature")),
                    s.get("search_algorithm"),
                    s.get("configured_iterations"),
                    s["run_rel"],
                    _f(s.get("prompt_complete_ratio")),
                    bool(files.get("evaluate.py")),
                    ds.ids.get(s.get("best_node_uid") or ""),
                )
            )


def load_attempts(
    conn: psycopg.Connection, ds: Dataset, count_tokens: Callable | None
) -> tuple[dict[str, dict[str, list[int]]], dict[str, str], int]:
    """Pass 1: one row per node. Returns per-session {response sha1: [attempt ids]} for the
    llm_call link, the distinct system prompts, and the number of nodes without code."""
    parent = {
        e["dst_node_uid"]: ds.ids[e["src_node_uid"]]
        for e in tx.read_jsonl(ds.src / "lineage_edges.jsonl")
    }
    node_sha: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    systems: dict[str, str] = {}
    no_code = 0
    cols = (
        "id, uid, task, session, backend, program_id, iteration, generation, is_valid, "
        "fitness, status, judge, exec_time, memory_mib, language, error, "
        "framework_changes, intent, intent_source, artifact_sha, code, parent_id, tokens, emb"
    )
    with conn.cursor().copy(f"COPY attempt ({cols}) FROM STDIN") as cp:
        for n, t in ds.texts():
            aid, s = ds.ids[n["node_uid"]], n["session_uid"]
            if t and t.response:
                node_sha[s][tx.sha1(clean(t.response))].append(aid)
            if t and t.system:
                systems[tx.sha1(clean(t.system))] = clean(t.system)
            code = clean(ds.code(n))
            no_code += code is None
            m = n.get("metrics") or {}
            cp.write_row(
                (
                    aid,
                    n["node_uid"],
                    n["task_uid"],
                    s,
                    ds.backend[s],
                    n["program_id"],
                    int(n["iteration"]),
                    n.get("generation"),
                    bool(n["is_valid"]),
                    _f(n.get("fitness")),
                    n["status"],
                    m.get("judge_result"),
                    _f(m.get("max_execution_time_sec")),
                    _f(m.get("max_memory_usage_mib")),
                    n.get("language"),
                    clean(n.get("error_signature")),
                    clean(n.get("changes")),
                    clean(t.intent) if t else None,
                    t.intent_source if t else None,
                    n.get("solution_sha256"),
                    code,
                    parent.get(n["node_uid"]),
                    count_tokens(code) if count_tokens and code else None,
                    vec(ds.vectors.get(("node", n["node_uid"]))),
                )
            )
    # vids in id order (dense, deterministic); gph_upsert_vertex reuses the map on a re-run.
    conn.execute(
        "UPDATE attempt a SET vid = m.vid FROM (SELECT id, graph_store.gph_upsert_vertex(id)"
        " AS vid FROM (SELECT id FROM attempt ORDER BY id) o) m WHERE a.id = m.id"
    )
    for aids in (x for d in node_sha.values() for x in d.values()):
        aids.sort(key=lambda a: (int(ds.nodes[a - 1]["iteration"]), a))
    return node_sha, systems, no_code


def load_prompts(
    conn: psycopg.Connection, ds: Dataset, systems: dict[str, str]
) -> None:
    """Pass 2 (separate so attempt rows exist for the FK): prompt text per attempt."""
    with conn.cursor().copy("COPY prompt_system (system_sha, text) FROM STDIN") as cp:
        for sha, text in systems.items():
            cp.write_row((sha, text))
    with conn.cursor().copy(
        "COPY prompt (attempt_id, template, system_sha, user_text, response_text, "
        "response_sha1, prompts_sha256, source) FROM STDIN"
    ) as cp:
        for n, t in ds.texts():
            if t is None or (t.response is None and t.user is None):
                continue  # shinka nodes with only a patch_description: intent, no prompt
            resp = clean(t.response)
            cp.write_row(
                (
                    ds.ids[n["node_uid"]],
                    t.template,
                    tx.sha1(clean(t.system)) if t.system else None,
                    clean(t.user),
                    resp,
                    tx.sha1(resp) if resp else None,
                    n.get("prompts_sha256"),
                    t.source,
                )
            )


def load_llm_calls(
    conn: psycopg.Connection, ds: Dataset, node_sha: dict[str, dict[str, list[int]]]
) -> None:
    """Stream each session's raw LLM log beside its normalized rows (same seq order)."""
    norm: dict[str, list[dict]] = defaultdict(list)
    for r in tx.read_jsonl(ds.src / "llm_calls.jsonl"):
        norm[r["session_uid"]].append(r)
    systems: dict[str, str] = {}
    cols = (
        "session, seq, attempt_id, link_method, link_candidates, purpose, model, temperature, finish_reason, "
        "prompt_tokens, completion_tokens, reasoning_tokens, error, content, reasoning, "
        "content_sha1, system_sha, user_text"
    )
    with conn.cursor().copy(f"COPY llm_call ({cols}) FROM STDIN") as cp:
        for s in ds.run_rel:
            rows = sorted(norm.get(s, []), key=lambda r: r["seq"])
            if not rows:
                continue
            raw = list(tx.read_jsonl(ds.run_dir(s) / "logs/llm_calls.jsonl"))
            if len(raw) != len(rows):
                raise RuntimeError(
                    f"{s}: {len(raw)} raw vs {len(rows)} normalized llm calls"
                )
            shas = [
                tx.sha1(clean(x["content"])) if x.get("content") else None for x in raw
            ]
            links = tx.pair_links(shas, node_sha.get(s, {}))
            for r, x, h, (aid, method, candidates) in zip(
                rows, raw, shas, links, strict=True
            ):
                system = clean(x.get("system_msg"))
                if system:
                    systems[tx.sha1(system)] = system
                cp.write_row(
                    (
                        s,
                        r["seq"],
                        aid,
                        method,
                        list(candidates),
                        tx.classify_call(x, aid is not None, ds.backend[s]),
                        r.get("model"),
                        _f(r.get("temperature")),
                        r.get("finish_reason"),
                        r.get("prompt_tokens"),
                        r.get("completion_tokens"),
                        r.get("reasoning_tokens"),
                        clean(r.get("error")),
                        clean(x.get("content")),
                        clean(x.get("reasoning_content")),
                        h,
                        tx.sha1(system) if system else None,
                        clean(x.get("user_msg")),
                    )
                )
    known = {r[0] for r in conn.execute("SELECT system_sha FROM prompt_system")}
    with conn.cursor().copy("COPY prompt_system (system_sha, text) FROM STDIN") as cp:
        for sha, text in systems.items():
            if sha not in known:
                cp.write_row((sha, text))


def load_state(conn: psycopg.Connection, ds: Dataset) -> int:
    """Stream state_events: scalars straight to COPY, membership run-length encoded."""
    rle = tx.MembershipRLE()
    scalars = 0
    with conn.cursor().copy(
        "COPY search_scalar (session, iteration, key, value, attempt_id) FROM STDIN"
    ) as cp:
        for e in tx.read_jsonl(ds.src / "state_events.jsonl"):
            if e["entity"] == "membership":
                rle.add(e)
            else:
                scalars += 1
                cp.write_row(
                    (
                        e["session_uid"],
                        e["iteration"],
                        e["role"],
                        jsonb(e["value"]),
                        ds.ids.get(e.get("node_uid") or ""),
                    )
                )
    with conn.cursor().copy(
        "COPY search_checkpoint (session, iteration) FROM STDIN"
    ) as cp:
        for row in rle.checkpoint_rows():
            cp.write_row(row)
    with conn.cursor().copy(
        "COPY search_membership (session, role, slot_key, attempt_id, from_iter, to_iter, value) "
        "FROM STDIN"
    ) as cp:
        for s, role, slot, node, lo, hi, value in rle.intervals():
            cp.write_row((s, role, slot, ds.ids.get(node or ""), lo, hi, value))
    return rle.events + scalars


# --------------------------------------------------------------------------- orchestration


def _check_fingerprint(
    conn: psycopg.Connection, ds: Dataset, graph_policy: dict
) -> None:
    row = conn.execute("SELECT fingerprint, graph_policy FROM load_meta").fetchone()
    edges = conn.execute("SELECT count(*) FROM edge_prop").fetchone()[0]
    if row and row[0] != ds.fingerprint and edges:
        raise SystemExit(
            "database already holds a load of a DIFFERENT dataset; use a fresh db"
        )
    if edges and (not row or row[1] != graph_policy):
        raise SystemExit(
            "existing graph policy differs or is unknown; use a fresh database"
        )
    conn.execute(
        "INSERT INTO load_meta (fingerprint, graph_policy) VALUES (%s, %s::jsonb) "
        "ON CONFLICT (only_row) DO UPDATE SET fingerprint = EXCLUDED.fingerprint, "
        "graph_policy = EXCLUDED.graph_policy",
        (ds.fingerprint, json.dumps(graph_policy)),
    )


def load(
    dsn: str,
    src: Path = SRC,
    raw: Path = RAW,
    *,
    similar_k: int = 16,
    merge_as_context: bool = False,
    count_tokens: Callable | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict:
    """Full (idempotent) load. Relational tables are rebuilt; graph edges are only added when
    their edge_prop mirror row is new, and vertices are reused through gph_vid_map."""
    ds = Dataset(src, raw)
    with connect(dsn) as conn:
        conn.execute(SCHEMA)
        graph_policy = {"similar_k": similar_k, "merge_as_context": merge_as_context}
        _check_fingerprint(conn, ds, graph_policy)
        conn.execute(f"TRUNCATE {', '.join(RELATIONAL)} CASCADE")
        load_tasks_sessions(conn, ds)
        node_sha, systems, no_code = load_attempts(conn, ds, count_tokens)
        load_prompts(conn, ds, systems)
        load_llm_calls(conn, ds, node_sha)
        state_events = load_state(conn, ds)
        inserted: dict[str, int] = {}
        for edges in gr.all_edges(ds, similar_k, merge_as_context=merge_as_context):
            for k, v in gr.insert_edges(conn, edges).items():
                inserted[k] = inserted.get(k, 0) + v
        conn.execute(INDEXES)
        conn.execute(NOT_SELECTED)
        embedded = embed_intents(conn, embed) if embed else 0
        conn.execute(HNSW)
        for t in (*RELATIONAL, "edge_prop"):
            conn.execute(f"ANALYZE {t}")
        conn.execute("SELECT trimem.analyze_graph()")
        return receipt(
            conn,
            ds,
            no_code=no_code,
            state_events=state_events,
            inserted=inserted,
            embedded=embedded,
            graph_policy=graph_policy,
        )


def embed_intents(
    conn: psycopg.Connection,
    embed: Callable[[list[str]], list[list[float]]],
    batch: int = 32,
) -> int:
    from trimem.data.embed_intent import fill_intent_emb

    return fill_intent_emb(conn, embed, batch=batch)


def _lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def receipt(conn: psycopg.Connection, ds: Dataset, **extra: object) -> dict:
    q = lambda sql: conn.execute(sql).fetchone()[0]
    edges = dict(
        conn.execute(
            "SELECT t.name, count(*) FROM edge_prop e JOIN graph_store.edge_type t "
            "ON t.id = e.edge_type GROUP BY 1"
        ).fetchall()
    )
    native = dict(
        conn.execute("SELECT edge_type, n_edges FROM trimem.graph_stats").fetchall()
    )
    r = {
        "attempt": q("SELECT count(*) FROM attempt"),
        "attempt_with_code": q("SELECT count(code) FROM attempt"),
        "attempt_with_vid": q("SELECT count(vid) FROM attempt"),
        "attempt_emb": q("SELECT count(emb) FROM attempt"),
        "intent": q("SELECT count(intent) FROM attempt"),
        "intent_emb": q("SELECT count(intent_emb) FROM attempt"),
        "task": q("SELECT count(*) FROM task"),
        "task_emb": q("SELECT count(emb) FROM task"),
        "session": q("SELECT count(*) FROM session"),
        "prompt": q("SELECT count(*) FROM prompt"),
        "prompt_system": q("SELECT count(*) FROM prompt_system"),
        "llm_call": q("SELECT count(*) FROM llm_call"),
        "llm_call_linked": q("SELECT count(attempt_id) FROM llm_call"),
        "llm_call_link_methods": dict(
            conn.execute(
                "SELECT link_method, count(*) FROM llm_call WHERE attempt_id IS NOT NULL "
                "GROUP BY 1"
            ).fetchall()
        ),
        "llm_call_purpose": dict(
            conn.execute("SELECT purpose, count(*) FROM llm_call GROUP BY 1").fetchall()
        ),
        "attempts_with_llm_call": q("SELECT count(DISTINCT attempt_id) FROM llm_call"),
        "search_membership": q("SELECT count(*) FROM search_membership"),
        "search_scalar": q("SELECT count(*) FROM search_scalar"),
        "search_checkpoint": q("SELECT count(*) FROM search_checkpoint"),
        "edges_mirrored": edges,
        "edges_native": native,
        **extra,
    }
    fwd = {
        t: edges.get(t, 0) for t in ("lineage", "context", "merge_parent", "similar")
    }
    checks = {
        "attempt == nodes": r["attempt"] == len(ds.nodes) == r["attempt_with_vid"],
        "task_emb == tasks": r["task_emb"] == r["task"] == len(ds.tasks),
        "session == sessions": r["session"] == len(ds.sessions),
        "lineage == file": fwd["lineage"] == _lines(ds.src / "lineage_edges.jsonl"),
        "context + merge == file": fwd["context"] + fwd["merge_parent"]
        == _lines(ds.src / "context_edges.jsonl"),
        "llm_call == file": r["llm_call"] == _lines(ds.src / "llm_calls.jsonl"),
        "_up mirrors forward": all(
            edges.get(t + "_up", 0) == n for t, n in fwd.items()
        ),
        "native == mirrored": all(native.get(t, 0) == n for t, n in edges.items()),
        "intent_emb == intent or 0": r["intent_emb"] in (0, r["intent"]),
    }
    r["checks"] = checks
    r["ok"] = all(checks.values())
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--similar-k", type=int, default=16)
    ap.add_argument(
        "--merge-as-context",
        action="store_true",
        help="keep Merge Parent B inside context, preserving its label",
    )
    ap.add_argument(
        "--no-tokens", action="store_true", help="skip Qwen3.8 code token counts"
    )
    ap.add_argument("--embed-endpoint", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--no-embed", action="store_true", help="leave intent_emb NULL")
    ap.add_argument("--receipt", type=Path)
    args = ap.parse_args(argv)
    guard_dsn(args.dsn)
    count_tokens = None
    if not args.no_tokens:
        from trimem.objective.store import count_tokens
    embed = None
    if not args.no_embed:
        from trimem.data.embed_intent import HttpEmbedder

        embed = HttpEmbedder(args.embed_endpoint).embed
    r = load(
        args.dsn,
        args.src,
        args.raw,
        similar_k=args.similar_k,
        merge_as_context=args.merge_as_context,
        count_tokens=count_tokens,
        embed=embed,
    )
    text = json.dumps(r, indent=2, default=str)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text)
    print(text)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
