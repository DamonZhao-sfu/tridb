"""Mem0 as a W1.a retriever, driven with the SAME vectors TriDB holds.

Two choices make this a fair comparison rather than a plausible-looking one:

**The vectors are not re-embedded.** `PGVector.insert(vectors, payloads, ids)` takes
precomputed vectors, so both systems index bit-identical embeddings of bit-identical
text. Re-embedding through Mem0's own path would introduce a difference we would then
have to argue was negligible; this removes the question. It also means ingestion makes
zero LLM or embedding calls.

**The query vector is passed in, not derived.** `PGVector.search(query, vectors, ...)`
accepts the vector directly, so the 10,479 queries never touch the embedding endpoint
either. Same seed vector on both sides, and the sweep is unaffected by GPU contention.

WHAT MEM0 CAN AND CANNOT EXPRESS HERE
W1.a's traversal (Task -> Session -> Node over `eg_hier`, 2 hops) is provably
equivalent to the metadata predicate `task_uid == T` on this corpus — 18/18 tasks,
zero discrepancy — so Mem0 expresses the *whole* query exactly: an ANN entry over Task
vectors, then a filtered ANN over Nodes. Nothing is emulated and nothing is
approximated away.

The three lineage queries (W1.b / W1.b-fail / W1.d) are a different matter: arbitrary
depth parent-child chains have no metadata encoding, and this Mem0 configuration has no
graph. They are reported `unsupported`, not degraded.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

DEFAULT_DSN = "postgresql://hza214@127.0.0.1:55432/w1x_mem0"
DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
EMBED_DIM = 1024
BATCH = 500


@dataclass
class Mem0W1Config:
    dsn: str = DEFAULT_DSN
    collection: str = "w1a_full"
    normalized: Path = DEFAULT_NORMALIZED
    embedding_dim: int = EMBED_DIM


@dataclass
class Mem0Result:
    """One W1.a answer, in the same shape the TriDB side reports."""

    ids: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0
    total_ms: float = 0.0
    # Mem0 returns a materialised list; there is no streaming boundary to time.
    first_row_ms: float | None = None
    calls: int = 0


class Mem0W1a:
    def __init__(self, config: Mem0W1Config | None = None) -> None:
        self.config = config or Mem0W1Config()
        from mem0.vector_stores.pgvector import PGVector

        self.store = PGVector(
            dbname=_dbname(self.config.dsn),
            collection_name=self.config.collection,
            embedding_model_dims=self.config.embedding_dim,
            user=_user(self.config.dsn),
            password=_password(self.config.dsn),
            host=_host(self.config.dsn),
            port=_port(self.config.dsn),
            diskann=False,
            hnsw=True,
        )

    # -- ingest ----------------------------------------------------------

    def ingest(self, *, limit: int | None = None) -> dict[str, Any]:
        """Load Tasks and Nodes with the exported vectors and a W1 payload."""
        blob = np.load(self.config.normalized / "vectors.npz", allow_pickle=False)
        uids = [str(u) for u in blob["uids"]]
        vectors = blob["vectors"].astype(np.float32, copy=False)
        row_of = {uid: i for i, uid in enumerate(uids)}

        tasks = {t["task_uid"]: t for t in _read(self.config.normalized / "tasks.jsonl")}
        nodes = list(_read(self.config.normalized / "nodes.jsonl"))
        if limit is not None:
            nodes = nodes[:limit]

        payloads: list[dict[str, Any]] = []
        rows: list[int] = []
        ids: list[str] = []

        for task_uid, task in tasks.items():
            row = row_of.get(task_uid)
            if row is None:
                continue
            rows.append(row)
            ids.append(_uuid(task_uid))
            payloads.append(
                {
                    "data": f"{task['domain']} task {task['task_key']}",
                    "kind": "task",
                    "node_uid": task_uid,
                    "task_uid": task_uid,
                    "domain": task["domain"],
                }
            )

        for node in nodes:
            row = row_of.get(node["node_uid"])
            if row is None:
                continue
            rows.append(row)
            ids.append(_uuid(node["node_uid"]))
            payloads.append(
                {
                    "data": node["node_uid"],
                    "kind": "node",
                    "node_uid": node["node_uid"],
                    "task_uid": node["task_uid"],
                    "session_uid": node["session_uid"],
                    # The W1 predicate is `is_valid AND fitness IS NOT NULL`. Folding
                    # both into one boolean keeps the Mem0 filter a single equality and
                    # therefore exactly the predicate TriDB pushes down — not a looser
                    # one that would quietly widen Mem0's candidate set.
                    "scorable": bool(node["is_valid"] and node.get("fitness") is not None),
                    "is_valid": bool(node["is_valid"]),
                }
            )

        self.store.create_col()
        started = time.perf_counter()
        for start in range(0, len(rows), BATCH):
            chunk = slice(start, start + BATCH)
            self.store.insert(
                vectors=[vectors[r].tolist() for r in rows[chunk]],
                payloads=payloads[chunk],
                ids=ids[chunk],
            )
        seconds = time.perf_counter() - started
        return {
            "collection": self.config.collection,
            "tasks": sum(1 for p in payloads if p["kind"] == "task"),
            "nodes": sum(1 for p in payloads if p["kind"] == "node"),
            "scorable_nodes": sum(1 for p in payloads if p.get("scorable")),
            "seconds": round(seconds, 2),
            "ms_per_row": round(seconds * 1000 / max(1, len(rows)), 2),
            "embedding_calls": 0,
            "llm_calls": 0,
        }

    # -- the query -------------------------------------------------------

    def search(
        self,
        seed_vector: Sequence[float],
        *,
        target_session: str,
        k: int = 10,
        m_seeds: int = 4,
    ) -> Mem0Result:
        """W1.a: ANN over Task vectors, then a filtered ANN over Nodes.

        Deliberately the same two stages and the same predicate as the TriDB side. The
        only difference is where the work happens: TriDB pushes the whole thing into one
        operator over a native adjacency; Mem0 runs two pgvector searches with a
        metadata filter.
        """
        out = Mem0Result()
        vector = list(seed_vector)

        t0 = time.perf_counter()
        hits = self.store.search("", vector, top_k=m_seeds, filters={"kind": "task"})
        out.stage1_ms = (time.perf_counter() - t0) * 1000.0
        out.calls += 1
        entries = [_payload(h).get("task_uid") for h in hits]
        entries = [e for e in entries if e]
        out.entries = entries
        if not entries:
            out.total_ms = out.stage1_ms
            return out

        t1 = time.perf_counter()
        rows = self.store.search(
            "",
            vector,
            top_k=k,
            filters={
                "kind": "node",
                "task_uid": {"in": entries},
                "session_uid": {"ne": target_session},
                "scorable": True,
            },
        )
        out.stage2_ms = (time.perf_counter() - t1) * 1000.0
        out.calls += 1
        out.ids = [str(_payload(r).get("node_uid")) for r in rows]
        out.total_ms = out.stage1_ms + out.stage2_ms
        return out

    def close(self) -> None:
        conn = getattr(self.store, "conn", None)
        if conn is not None:
            conn.close()


# -- helpers -------------------------------------------------------------


def _payload(hit: Any) -> dict[str, Any]:
    return getattr(hit, "payload", None) or {}


def _read(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _uuid(uid: str) -> str:
    """Mem0's pgvector store keys on a UUID; ours is derived so a reload is stable."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, uid))


def _parse(dsn: str) -> Any:
    from urllib.parse import urlparse

    return urlparse(dsn)


def _dbname(dsn: str) -> str:
    return _parse(dsn).path.lstrip("/") or "postgres"


def _user(dsn: str) -> str:
    return _parse(dsn).username or "postgres"


def _password(dsn: str) -> str:
    return _parse(dsn).password or ""


def _host(dsn: str) -> str:
    return _parse(dsn).hostname or "127.0.0.1"


def _port(dsn: str) -> int:
    return int(_parse(dsn).port or 5432)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--collection", default="w1a_full")
    parser.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    adapter = Mem0W1a(
        Mem0W1Config(dsn=args.dsn, collection=args.collection, normalized=args.normalized)
    )
    try:
        print(json.dumps(adapter.ingest(limit=args.limit), indent=2))
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
