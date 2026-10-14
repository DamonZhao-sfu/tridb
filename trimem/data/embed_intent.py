"""Embed `attempt.intent` into `attempt.intent_emb` (Qwen3-Embedding-0.6B on the 8011 server).

Same model / dimension / metric as the node_artifact and task_spec tracks, but a separate track:
intent vectors are never ranked together with code vectors. Resumable: only rows with
`intent IS NOT NULL AND intent_emb IS NULL` are embedded.

    PYTHONPATH=. python -m trimem.data.embed_intent --dsn "host=/tmp/trimem-traj port=55450 dbname=trimem_traj"
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx
import psycopg

MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_CHARS = 6000  # p99 intent is far below this; the server's limit is 8192 tokens


class HttpEmbedder:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8011/v1",
        model: str = MODEL,
        retries: int = 6,
    ) -> None:
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.retries = model, retries
        self.http = httpx.Client(timeout=120)

    def embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.retries):
            try:
                r = self.http.post(
                    self.url,
                    json={"model": self.model, "input": [t[:MAX_CHARS] for t in texts]},
                )
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            except httpx.HTTPError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(60, 2**attempt * 2))  # the shared server can restart
        raise AssertionError("unreachable")


def fill_intent_emb(
    conn: psycopg.Connection,
    embed: Callable[[list[str]], list[list[float]]],
    *,
    batch: int = 32,
    workers: int = 2,
) -> int:
    rows = conn.execute(
        "SELECT id, intent FROM attempt WHERE intent IS NOT NULL "
        "AND intent_emb IS NULL ORDER BY id"
    ).fetchall()
    chunks = [rows[i : i + batch] for i in range(0, len(rows), batch)]
    n = 0
    with ThreadPoolExecutor(workers) as pool:
        for chunk, vecs in zip(
            chunks, pool.map(lambda c: embed([t for _, t in c]), chunks), strict=True
        ):
            if len(vecs) != len(chunk):
                raise RuntimeError(
                    f"embedder returned {len(vecs)} vectors for {len(chunk)} texts"
                )
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(
                        "UPDATE attempt SET intent_emb = %s::vector WHERE id = %s",
                        [
                            ("[" + ",".join(map(repr, v)) + "]", i)
                            for (i, _), v in zip(chunk, vecs, strict=True)
                        ],
                    )
            n += len(chunk)
    return n


def main(argv: list[str] | None = None) -> int:
    from trimem.data.evotrace import connect

    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)
    with connect(args.dsn) as conn:
        n = fill_intent_emb(conn, HttpEmbedder(args.endpoint).embed, batch=args.batch)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS attempt_intent_hnsw ON attempt "
            "USING hnsw (intent_emb vector_cosine_ops)"
        )
    print(f"intent_emb filled: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
