"""P0 gate for Mem0: can it express the W1 query family at all?

Run inside the mem0 venv:

    /localhome/hza214/agent-memory-table5/venv/mem0/bin/python \
        experiments/e2/p0_gate_mem0.py

Standalone on purpose — no tridb imports — so a dependency clash in one venv cannot
take the probe down. It answers the three questions that decide whether the
cross-system comparison is possible at all, before any hours go into ingestion:

  G1  node_uid survives a round trip. Mem0's add() runs an LLM that rewrites and
      merges facts; if the identifier does not come back verbatim the returned rows
      cannot be scored against a ground truth keyed on node_uid, and the quality
      comparison is dead on arrival.
  G2  infer=False stores text verbatim and skips the LLM. Worth ~3,200 LLM calls on
      the Tier-1 corpus, and it is the difference between "Mem0 stores our nodes" and
      "Mem0 stores its own paraphrase of our nodes".
  G3  metadata filtering can express the W1 predicate: task_uid IN (...) AND
      session_uid != target AND is_valid. The session exclusion is the whole point of
      cross-session reuse — without it the benchmark leaks the answer.
"""

from __future__ import annotations

import json
import os
import time
import uuid

os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

DSN = "postgresql://hza214@127.0.0.1:55432/w1x_mem0"
COLLECTION = f"p0probe_{uuid.uuid4().hex[:8]}"
EMBED_DIM = 1024

RESULT: dict[str, object] = {"system": "mem0", "collection": COLLECTION}


def config() -> dict:
    return {
        "version": "v1.1",
        "history_db_path": f"/tmp/{COLLECTION}_history.db",
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "collection_name": COLLECTION,
                "embedding_model_dims": EMBED_DIM,
                "connection_string": DSN,
                "hnsw": True,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "Qwen/Qwen3-32B",
                "api_key": "EMPTY",
                "openai_base_url": "http://127.0.0.1:8000/v1",
                "temperature": 0.0,
                "max_tokens": 512,
                "is_reasoning_model": False,
            },
        },
        # No embedding_dims: Qwen3-Embedding is fixed 1024-d and rejects OpenAI's
        # `dimensions` parameter outright.
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "api_key": "EMPTY",
                "openai_base_url": "http://127.0.0.1:8001/v1",
            },
        },
    }


NODES = [
    {
        "node_uid": f"349117b0:evox/t{t}_run_{s}#prog-{i}",
        "task_uid": f"math:task_{t}",
        "session_uid": f"349117b0:evox/t{t}_run_{s}",
        "is_valid": i % 3 != 0,
        "fitness": float(100 - i),
        "text": (
            f"Language: python\nOutcome: {'accepted' if i % 3 else 'rejected'}\n"
            f"Edit: rewrite variant {i}\nCode:\n"
            f"def solve_{t}_{i}(x):\n    return sum(x) * {i} / {t + 1}\n"
        ),
    }
    for t in range(3)
    for s in range(2)
    for i in range(4)
]


def main() -> int:
    from mem0 import Memory

    memory = Memory.from_config(config())

    # -- endpoint gate ---------------------------------------------------
    vector = memory.embedding_model.embed("dimension probe", "search")
    RESULT["embedding_dim"] = len(vector)
    RESULT["G0_embedding_endpoint"] = len(vector) == EMBED_DIM
    memory.vector_store._ensure_collection()

    # -- G2: does infer=False skip the LLM and store text verbatim? -------
    started = time.perf_counter()
    infer_false_ok = True
    infer_false_error = None
    try:
        for node in NODES:
            memory.add(
                node["text"],
                user_id="w1probe",
                metadata={
                    "node_uid": node["node_uid"],
                    "task_uid": node["task_uid"],
                    "session_uid": node["session_uid"],
                    "is_valid": node["is_valid"],
                    "fitness": node["fitness"],
                },
                infer=False,
            )
    except Exception as exc:  # noqa: BLE001 - the probe's job is to report it
        infer_false_ok = False
        infer_false_error = f"{type(exc).__name__}: {exc}"
    RESULT["G2_infer_false"] = infer_false_ok
    RESULT["G2_error"] = infer_false_error
    RESULT["ingest_seconds_no_llm"] = round(time.perf_counter() - started, 2)
    RESULT["ingest_ms_per_node"] = round(
        (time.perf_counter() - started) * 1000 / max(1, len(NODES)), 1
    )

    # -- G1: node_uid round trip -----------------------------------------
    hit = memory.search(
        "python solve rewrite", top_k=10, filters={"user_id": "w1probe"}, rerank=False
    )
    rows = list(hit.get("results") or [])
    RESULT["search_rows"] = len(rows)
    uids = [str((r.get("metadata") or {}).get("node_uid")) for r in rows]
    RESULT["G1_node_uid_roundtrip"] = bool(rows) and all(
        u in {n["node_uid"] for n in NODES} for u in uids
    )
    RESULT["sample_uids"] = uids[:3]
    if rows:
        stored = str(rows[0].get("memory") or "")
        original = next(
            (n["text"] for n in NODES if n["node_uid"] == uids[0]), ""
        )
        RESULT["G2b_text_verbatim"] = stored.strip() == original.strip()
        RESULT["stored_prefix"] = stored[:80]

    # -- G3: can the W1 predicate be pushed into the filter? --------------
    target_session = NODES[0]["session_uid"]
    checks: dict[str, object] = {}
    for label, filters in (
        ("equality_task", {"user_id": "w1probe", "task_uid": "math:task_0"}),
        ("in_task", {"user_id": "w1probe", "task_uid": {"in": ["math:task_0", "math:task_1"]}}),
        ("ne_session", {"user_id": "w1probe", "session_uid": {"ne": target_session}}),
        ("bool_valid", {"user_id": "w1probe", "is_valid": True}),
        (
            "combined",
            {
                "user_id": "w1probe",
                "task_uid": {"in": ["math:task_0", "math:task_1"]},
                "session_uid": {"ne": target_session},
                "is_valid": True,
            },
        ),
    ):
        try:
            got = memory.search("solve", top_k=50, filters=filters, rerank=False)
            got_rows = list(got.get("results") or [])
            meta = [(r.get("metadata") or {}) for r in got_rows]
            checks[label] = {
                "ok": True,
                "rows": len(got_rows),
                "distinct_task": sorted({str(m.get("task_uid")) for m in meta}),
                "target_session_leaked": any(
                    str(m.get("session_uid")) == target_session for m in meta
                ),
            }
        except Exception as exc:  # noqa: BLE001
            checks[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    RESULT["G3_filters"] = checks

    # -- search latency, warm -------------------------------------------
    for _ in range(3):
        memory.search("solve", top_k=10, filters={"user_id": "w1probe"}, rerank=False)
    times = []
    for _ in range(10):
        t = time.perf_counter()
        memory.search("solve", top_k=10, filters={"user_id": "w1probe"}, rerank=False)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    RESULT["search_ms_p50"] = round(times[len(times) // 2], 2)

    print(json.dumps(RESULT, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
