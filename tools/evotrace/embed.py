"""P3 — populate the Experience Graph's two vector tracks.

    python3 tools/evotrace/embed.py --dsn ... --track task
    python3 tools/evotrace/embed.py --dsn ... --track node

TWO TRACKS, NEVER POOLED
------------------------
`task_spec`
    One vector per Task, over the problem specification. This is the entry point
    arXiv:2606.29823 actually specifies: "cross-session reuse is vector-seeded graph
    retrieval", seeded from a *task description*. There are only 16 of them, so this
    track measures plan correctness, not ANN scale, and any latency reported for it
    must say so.

`node_artifact`
    One vector per Node, over its code plus its edit description and failure signature.
    Not in the paper's critical path — it exists because tjs_open drops candidates
    whose vector is NULL (tjs_pg.c, both the filter-first and PPR bridge legs do
    `if (vnull) continue;`), so a query that must RETURN Nodes needs Node vectors.
    That makes it a real requirement of this engine and a separate experiment from the
    Task ANN.

Sessions and Prompts stay NULL on purpose: they are hops on the path to an answer, and
inventing a vector for them would invent a ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.gem_eg.store import DEFAULT_DIM, DEFAULT_DSN, EgStore, vec_literal

DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1/embeddings"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_MANIFEST = Path("data/evotrace/normalized/embedding_manifest.json")

#: Code is truncated before embedding. Stated here rather than buried: an embedding of
#: the first N characters is not an embedding of the program, and the manifest records
#: the limit so a later run at a different limit is not silently comparable.
CODE_CHARS = 2000
BATCH = 64


@dataclass(frozen=True)
class Track:
    name: str
    kind: str
    sql: str


TASK_TRACK = Track(
    name="task_spec",
    kind="task",
    sql=(
        "SELECT v.id, t.task_uid, t.domain, t.task_key, t.specification,"
        "       t.success_metric, t.target_environment"
        " FROM gem_eg_vertex v JOIN gem_eg_task t ON t.id = v.id"
        " WHERE v.scope_id = %s AND (v.embedding IS NULL OR %s)"
        " ORDER BY v.id"
    ),
)

NODE_TRACK = Track(
    name="node_artifact",
    kind="node",
    sql=(
        "SELECT v.id, n.node_uid, n.language, n.status, n.changes, n.error_signature,"
        "       a.payload"
        " FROM gem_eg_vertex v"
        " JOIN gem_eg_node n ON n.id = v.id"
        " LEFT JOIN gem_eg_artifact a ON a.artifact_uid = n.artifact_uid"
        " WHERE v.scope_id = %s AND (v.embedding IS NULL OR %s)"
        " ORDER BY v.id"
    ),
)

TRACKS = {"task": TASK_TRACK, "node": NODE_TRACK}


def task_text(row: Sequence[Any]) -> str:
    _, task_uid, domain, task_key, spec, metric, environment = row
    env = environment if isinstance(environment, dict) else json.loads(environment or "{}")
    languages = ", ".join(env.get("languages") or []) or "unspecified"
    return (
        f"Task: {task_key}\n"
        f"Domain: {domain}\n"
        f"Identifier: {task_uid}\n"
        f"Implementation language: {languages}\n"
        f"Success metric: {metric or 'combined_score'}\n"
        f"Specification: {spec}"
    )


def node_text(row: Sequence[Any]) -> str:
    _, node_uid, language, status, changes, error, payload = row
    return node_query_text(
        language=language,
        status=status,
        changes=changes,
        error=error,
        payload=payload,
    )


def node_query_text(
    *,
    language: str | None,
    status: str | None,
    changes: str | None,
    error: str | None,
    payload: str | None,
) -> str:
    """Canonical node-artifact renderer shared by offline and live paths.

    Keeping the generated-parent path here prevents a subtle second embedding space:
    the online query must have byte-identical field labels and truncation to the
    stored corpus embeddings.
    """

    code = (payload or "")[:CODE_CHARS]
    parts = [
        f"Language: {language or 'unknown'}",
        f"Outcome: {status or 'unknown'}",
    ]
    if changes:
        parts.append(f"Edit: {changes}")
    if error:
        # The failure signature is deliberately part of the vector: the Repair pattern
        # searches for attempts that failed the same way.
        parts.append(f"Failure: {error}")
    parts.append(f"Code:\n{code}" if code else "Code: not held (blob missing)")
    return "\n".join(parts)


def embed_batch(
    texts: list[str], *, endpoint: str, model: str, retries: int = 3
) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
            return [item["embedding"] for item in payload["data"]]
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"embedding endpoint failed after {retries} attempts: {last}")


def run_track(
    store: EgStore,
    track: Track,
    *,
    scope_id: str,
    endpoint: str,
    model: str,
    refresh: bool,
    limit: int | None,
) -> dict[str, Any]:
    rows = store.conn.execute(track.sql, (scope_id, refresh)).fetchall()
    if limit is not None:
        rows = rows[:limit]
    renderer = task_text if track.kind == "task" else node_text

    embedded = 0
    text_hash = hashlib.sha256()
    started = time.perf_counter()
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        texts = [renderer(row) for row in batch]
        for text in texts:
            text_hash.update(text.encode("utf-8"))
        vectors = embed_batch(texts, endpoint=endpoint, model=model)
        if vectors and len(vectors[0]) != store.dim:
            raise SystemExit(
                f"endpoint returned dimension {len(vectors[0])}, schema is "
                f"vector({store.dim}) — refusing to write a mixed embedding space"
            )
        for row, vector in zip(batch, vectors):
            store.conn.execute(
                "UPDATE gem_eg_vertex"
                " SET embedding = %s::vector, embedding_source = %s, embedding_model = %s"
                " WHERE id = %s",
                (vec_literal(vector), track.name, model, int(row[0])),
            )
            embedded += 1
        store.conn.commit()
        print(f"  {track.name}: {embedded}/{len(rows)}", end="\r", flush=True)
    print()
    return {
        "track": track.name,
        "vertex_kind": track.kind,
        "model": model,
        "endpoint": endpoint,
        "dimension": store.dim,
        "metric": "cosine",
        "code_truncation_chars": CODE_CHARS if track.kind == "node" else None,
        "rows_embedded": embedded,
        "canonical_text_sha256": text_hash.hexdigest(),
        "seconds": round(time.perf_counter() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--track", choices=sorted(TRACKS) + ["all"], default="all")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--refresh", action="store_true", help="re-embed rows that already have one")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test cap per track")
    args = parser.parse_args(argv)

    names = sorted(TRACKS) if args.track == "all" else [args.track]
    store = EgStore.connect(args.dsn, dim=args.dim)
    results = []
    try:
        for name in names:
            results.append(
                run_track(
                    store,
                    TRACKS[name],
                    scope_id=args.scope,
                    endpoint=args.endpoint,
                    model=args.model,
                    refresh=args.refresh,
                    limit=args.limit,
                )
            )
    finally:
        store.close()

    manifest = {
        "scope_id": args.scope,
        "tracks": results,
        "note": (
            "task_spec and node_artifact are SEPARATE experiments. The paper's "
            "cross-session reuse is seeded from task_spec (16 vectors: correctness, "
            "not ANN scale). node_artifact exists because tjs_open skips candidates "
            "with NULL vectors, so returning Nodes requires Node vectors."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2))
    for row in results:
        print(f"{row['track']:16} {row['rows_embedded']:>7,} rows in {row['seconds']}s")
    print(f"manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
