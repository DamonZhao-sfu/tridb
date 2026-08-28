"""P0 gate for Cognee: can it express the W1 query family, and at what ingest cost?

Run inside the cognee venv:

    /localhome/hza214/agent-memory-table5/venv/cognee/bin/python \
        experiments/e2/p0_gate_cognee.py

Cognee differs from Mem0 in a way that decides the whole design: its public add API
exposes no per-passage metadata, so an identifier can only travel inside the text and
be recovered by regex, and the ONLY scoping mechanism is the dataset. That forces one
design decision up front — **one dataset per session** — because the W1 predicate's
core constraint is `session_uid != target`, and a dataset list is the only place Cognee
can express it.

Gates:
  G1  node_uid survives ingestion inside the text and is recoverable from the returned
      context.
  G2  ingest cost per node. cognify() is LLM-driven entity extraction; this is the
      number that decides whether the corpus must be subsampled and by how much.
  G3  scoping: can `datasets=[...]` express "these tasks, excluding the target
      session"? And can the reward predicate (is_valid, fitness) be pushed down at
      all — expected NO, which would make it a client-side post-filter and a reported
      limitation rather than a silent difference.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

DB_NAME = "w1x_cognee"
EMBED_DIM = 1024
MARKER = re.compile(r"\[node_uid=([^\]]+)\]")

RESULT: dict[str, object] = {"system": "cognee"}

os.environ.update(
    {
        "USE_UNIFIED_PROVIDER": "",
        "ENABLE_BACKEND_ACCESS_CONTROL": "true",
        "CACHING": "false",
        "DB_PROVIDER": "postgres",
        "DB_HOST": "127.0.0.1",
        "DB_PORT": "55432",
        "DB_USERNAME": "hza214",
        "DB_PASSWORD": os.environ.get("COGNEE_DB_PASSWORD", "w1xprobe"),
        "DB_NAME": DB_NAME,
        "GRAPH_DATABASE_PROVIDER": "postgres",
        "GRAPH_DATASET_DATABASE_HANDLER": "postgres_graph_shared",
        "VECTOR_DB_PROVIDER": "pgvector",
        "VECTOR_DATASET_DATABASE_HANDLER": "pgvector_shared",
        "CACHE_BACKEND": "postgres",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "openai/Qwen/Qwen3-32B",
        "LLM_ENDPOINT": "http://127.0.0.1:8000/v1",
        "LLM_API_KEY": "EMPTY",
        "OPENAI_API_KEY": "EMPTY",
        "LLM_TEMPERATURE": "0.0",
        "LLM_MAX_COMPLETION_TOKENS": "512",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "LLM_RATE_LIMIT_ENABLED": "false",
        "AUTO_RATE_LIMIT": "false",
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
        "EMBEDDING_ENDPOINT": "http://127.0.0.1:8001/v1",
        "EMBEDDING_API_KEY": "EMPTY",
        "EMBEDDING_DIMENSIONS": str(EMBED_DIM),
        "EMBEDDING_BATCH_SIZE": "64",
        "HUGGINGFACE_TOKENIZER": "Qwen/Qwen3-Embedding-0.6B",
        "LOG_LEVEL": "ERROR",
        "TELEMETRY_DISABLED": "true",
        "COGNEE_TELEMETRY_DISABLED": "true",
    }
)

# Deliberately tiny: cognify() is LLM-driven, and the point of this gate is to learn
# the per-node cost before committing to a corpus size, not to build one.
SESSIONS = ["s_t0_a", "s_t0_b", "s_t1_a"]
NODES = [
    {
        "node_uid": f"349117b0:evox/{s}#prog-{i}",
        "task_uid": "math:task_0" if s.startswith("s_t0") else "math:task_1",
        "session_uid": f"349117b0:evox/{s}",
        "dataset": s,
        "is_valid": i % 3 != 0,
        "fitness": float(100 - i),
        "text": (
            f"Language: python. Outcome: {'accepted' if i % 3 else 'rejected'}. "
            f"Edit: rewrite variant {i}. "
            f"Code: def solve_{i}(x): return sum(x) * {i}"
        ),
    }
    for s in SESSIONS
    for i in range(3)
]


def document(node: dict) -> str:
    """The identifier has to live in the text: Cognee exposes no passage metadata."""
    return f"[node_uid={node['node_uid']}] {node['text']}"


async def run() -> None:
    import cognee
    from cognee.modules.search.types import SearchType

    RESULT["search_types"] = sorted(
        t for t in dir(SearchType) if t.isupper() and not t.startswith("_")
    )

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # -- G2: ingest cost -------------------------------------------------
    t_add = time.perf_counter()
    for node in NODES:
        await cognee.add(document(node), dataset_name=node["dataset"])
    add_seconds = time.perf_counter() - t_add

    t_cog = time.perf_counter()
    cognify_error = None
    try:
        await cognee.cognify(datasets=SESSIONS)
    except Exception as exc:  # noqa: BLE001 - the probe reports rather than raises
        cognify_error = f"{type(exc).__name__}: {exc}"
    cognify_seconds = time.perf_counter() - t_cog

    RESULT["G2_add_seconds"] = round(add_seconds, 2)
    RESULT["G2_cognify_seconds"] = round(cognify_seconds, 2)
    RESULT["G2_cognify_error"] = cognify_error
    RESULT["G2_total_ms_per_node"] = round(
        (add_seconds + cognify_seconds) * 1000 / len(NODES), 1
    )
    # The number that sizes the whole experiment.
    RESULT["G2_projected_hours_3200_nodes"] = round(
        (add_seconds + cognify_seconds) / len(NODES) * 3200 / 3600, 2
    )

    # -- G1 + G3: scoping and identifier recovery ------------------------
    known = {n["node_uid"] for n in NODES}
    target_session = "349117b0:evox/s_t0_a"
    probes: dict[str, object] = {}
    for label, datasets, query_type in (
        ("all_sessions", SESSIONS, "HYBRID_COMPLETION"),
        # The W1 scoping: task_0's sessions, minus the target. This is the ONLY way
        # Cognee can express `session_uid != target`.
        ("task0_excluding_target", ["s_t0_b"], "HYBRID_COMPLETION"),
        ("chunks_no_llm", SESSIONS, "CHUNKS"),
        # If CYPHER really executes a user-supplied traversal, the lineage
        # queries (W1.b/W1.d) are NOT automatically unsupported here.
        ("cypher_probe", SESSIONS, "CYPHER"),
    ):
        try:
            started = time.perf_counter()
            got = await cognee.search(
                query_text="python solve rewrite variant",
                query_type=getattr(SearchType, query_type),
                only_context=True,
                top_k=10,
                datasets=datasets,
            )
            elapsed = (time.perf_counter() - started) * 1000
            contexts = [str(v) for v in (got or [])]
            found = list(dict.fromkeys(MARKER.findall("\n".join(contexts))))
            probes[label] = {
                "ok": True,
                "ms": round(elapsed, 1),
                "contexts": len(contexts),
                "node_uids_recovered": len(found),
                "all_known": all(u in known for u in found),
                "target_session_leaked": any(target_session in u for u in found),
                "sample": found[:3],
            }
        except Exception as exc:  # noqa: BLE001
            probes[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    RESULT["G1_G3_probes"] = probes

    # -- G3b: is a reward predicate expressible at all? ------------------
    # Cognee's search signature is the evidence: if it takes no filter argument, the
    # predicate can only be applied client-side, after the top-k has already been cut.
    import inspect

    RESULT["G3b_search_signature"] = str(inspect.signature(cognee.search))


def main() -> int:
    try:
        asyncio.run(run())
    except Exception as exc:  # noqa: BLE001
        RESULT["fatal"] = f"{type(exc).__name__}: {exc}"[:600]
    print(json.dumps(RESULT, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
