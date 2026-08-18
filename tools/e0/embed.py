"""Checkpointed embedding generation through a pinned OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
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


def _request(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer local", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _embed_batch(
    texts: list[str], *, base_url: str, model: str, timeout: int, retries: int
) -> list[list[float]]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            payload = _request(
                f"{base_url.rstrip('/')}/embeddings",
                {"model": model, "input": texts, "encoding_format": "float"},
                timeout,
            )
            ordered = sorted(payload["data"], key=lambda row: int(row["index"]))
            return [row["embedding"] for row in ordered]
        except (OSError, KeyError, ValueError, urllib.error.HTTPError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"embedding request failed after {retries + 1} attempts: {last}")


def embed_parquet(
    input_path: Path,
    output_path: Path,
    *,
    id_column: str,
    text_column: str,
    base_url: str,
    model: str,
    batch_size: int,
    max_chars: int,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    model_info = endpoint_models(base_url, timeout)
    advertised = {str(row["id"]) for row in model_info}
    if advertised != {model}:
        raise RuntimeError(
            f"endpoint model mismatch: expected only {model!r}, got {advertised}"
        )
    artifact = str(model_info[0].get("root", model))
    local_revisions = hf_snapshot_revisions(artifact)
    if (
        model == "Qwen/Qwen3-Embedding-0.6B"
        and QWEN3_EMBEDDING_REVISION not in local_revisions
    ):
        raise RuntimeError(
            f"expected embedding revision {QWEN3_EMBEDDING_REVISION} is not visible: "
            f"{local_revisions}"
        )

    table = pq.read_table(input_path, columns=[id_column, text_column])
    ids = table[id_column].to_pylist()
    texts = table[text_column].to_pylist()
    parts_dir = output_path.parent / f".{output_path.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    dimensions: set[int] = set()
    truncated_count = 0

    for start in range(0, len(ids), batch_size):
        stop = min(start + batch_size, len(ids))
        part = parts_dir / f"part-{start:09d}-{stop:09d}.parquet"
        if part.exists():
            existing = pq.read_table(part)
            if existing.num_rows == stop - start:
                dimensions.update(
                    len(value) for value in existing["embedding"].to_pylist()
                )
                truncated_count += sum(existing["truncated"].to_pylist())
                continue
            raise RuntimeError(f"invalid checkpoint part: {part}")

        batch_texts: list[str] = []
        truncated: list[bool] = []
        text_hashes: list[str] = []
        original_chars: list[int] = []
        for value in texts[start:stop]:
            text = "" if value is None else str(value)
            original_chars.append(len(text))
            was_truncated = len(text) > max_chars
            text = text[:max_chars]
            batch_texts.append(text)
            truncated.append(was_truncated)
            text_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
        vectors = _embed_batch(
            batch_texts,
            base_url=base_url,
            model=model,
            timeout=timeout,
            retries=retries,
        )
        if len(vectors) != len(batch_texts):
            raise RuntimeError(
                f"endpoint returned {len(vectors)} vectors for {len(batch_texts)} inputs"
            )
        dimensions.update(len(value) for value in vectors)
        if len(dimensions) != 1:
            raise RuntimeError(f"inconsistent embedding dimensions: {dimensions}")
        rows = pa.Table.from_pydict(
            {
                "node_id": ids[start:stop],
                "embedding": vectors,
                "embedding_model": [model] * len(vectors),
                "source_text_sha256": text_hashes,
                "original_chars": original_chars,
                "truncated": truncated,
            }
        )
        pq.write_table(rows, part, compression="zstd")
        truncated_count += sum(truncated)
        print(f"embedded {stop}/{len(ids)}", flush=True)

    part_paths = sorted(parts_dir.glob("part-*.parquet"))
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for part in part_paths:
            part_table = pq.read_table(part)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(
                    output_path, part_table.schema, compression="zstd"
                )
            writer.write_table(part_table)
            total += part_table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if total != len(ids):
        raise RuntimeError(
            f"combined embedding row count {total} != input row count {len(ids)}"
        )

    manifest = {
        "schema_version": "e0-embeddings-v0.1.0",
        "environment": environment_record(),
        "input": artifact_record(input_path),
        "output": artifact_record(output_path),
        "endpoint": base_url,
        "model": model,
        "advertised_model": model_info,
        "artifact": artifact,
        "artifact_revision": (
            QWEN3_EMBEDDING_REVISION if model == "Qwen/Qwen3-Embedding-0.6B" else None
        ),
        "locally_visible_revisions": local_revisions,
        "dimensions": sorted(dimensions),
        "rows": total,
        "batch_size": batch_size,
        "max_chars": max_chars,
        "truncated_rows": truncated_count,
        "checkpoint_parts": len(part_paths),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", default="node_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=16_000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    manifest = embed_parquet(
        args.input,
        args.output,
        id_column=args.id_column,
        text_column=args.text_column,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps({"rows": manifest["rows"], "dimensions": manifest["dimensions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
