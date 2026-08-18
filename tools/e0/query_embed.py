"""Embed E0 JSONL query text with the pinned local embedding endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.e0 import QWEN3_EMBEDDING_REVISION
from tools.e0.common import (
    artifact_record,
    endpoint_models,
    environment_record,
    hf_snapshot_revisions,
    write_json,
)
from tools.e0.embed import _embed_batch


def embed_queries(
    queries_path: Path,
    output_path: Path,
    *,
    base_url: str,
    model: str,
    batch_size: int,
    timeout: int,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output_path.exists() and not force:
        manifest_path = output_path.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise FileExistsError(
                f"{output_path} exists without a manifest; pass --force only after audit"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_input = artifact_record(queries_path)
        current_output = artifact_record(output_path)
        if (
            existing.get("input", {}).get("sha256") == current_input["sha256"]
            and existing.get("output", {}).get("sha256") == current_output["sha256"]
            and existing.get("model") == model
        ):
            existing["reused"] = True
            return existing
        raise FileExistsError(
            f"stale or mismatched {output_path}; pass --force to replace it"
        )
    rows = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_ids = [str(row["query_id"]) for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_id values must be unique")

    model_info = endpoint_models(base_url, timeout)
    advertised = {str(row["id"]) for row in model_info}
    if advertised != {model}:
        raise RuntimeError(f"expected only {model!r}, got endpoint models {advertised}")
    artifact = str(model_info[0].get("root", model))
    revisions = hf_snapshot_revisions(artifact)
    if (
        model == "Qwen/Qwen3-Embedding-0.6B"
        and QWEN3_EMBEDDING_REVISION not in revisions
    ):
        raise RuntimeError(
            f"expected revision {QWEN3_EMBEDDING_REVISION} is not locally visible"
        )

    vectors: list[list[float]] = []
    texts = [str(row["query_text"]) for row in rows]
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        vectors.extend(
            _embed_batch(
                texts[start:stop],
                base_url=base_url,
                model=model,
                timeout=timeout,
                retries=retries,
            )
        )
        print(f"embedded queries {stop}/{len(texts)}", flush=True)
    dimensions = sorted({len(vector) for vector in vectors})
    if len(vectors) != len(rows) or len(dimensions) != 1:
        raise RuntimeError(
            f"invalid embedding response rows={len(vectors)} dimensions={dimensions}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "query_id": query_ids,
            "embedding": vectors,
            "embedding_model": [model] * len(rows),
            "query_text_sha256": [
                hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
            ],
        }
    )
    pq.write_table(table, output_path, compression="zstd")
    manifest = {
        "schema_version": "e0-query-embeddings-v0.1.0",
        "environment": environment_record(),
        "input": artifact_record(queries_path),
        "output": artifact_record(output_path),
        "endpoint": base_url,
        "model": model,
        "artifact": artifact,
        "artifact_revision": (
            QWEN3_EMBEDDING_REVISION if model == "Qwen/Qwen3-Embedding-0.6B" else None
        ),
        "locally_visible_revisions": revisions,
        "rows": len(rows),
        "dimensions": dimensions,
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = embed_queries(
        args.queries,
        args.output,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        timeout=args.timeout,
        retries=args.retries,
        force=args.force,
    )
    print(json.dumps({"rows": manifest["rows"], "dimensions": manifest["dimensions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
