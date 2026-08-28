"""Convert the official STaRK-MAG ada-002 tensors into E1 Parquet artifacts.

STaRK embeds retrieval candidates only (MAG papers), while author, institution, and topic
nodes remain graph anchors.  The output therefore intentionally has fewer embedding rows
than graph nodes; manifests make that coverage boundary explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.e0.common import (
    artifact_record,
    environment_record,
    sha256_file,
    write_json,
)

MODEL = "text-embedding-ada-002"
DIMENSION = 1536


def _write_vectors(
    path: Path,
    values: dict[int, Any],
    *,
    id_name: str,
    selected_ids: list[int],
) -> int:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for start in range(0, len(selected_ids), 2048):
            ids = selected_ids[start : start + 2048]
            vectors = np.stack(
                [values[node_id].reshape(-1).numpy() for node_id in ids]
            ).astype(np.float32, copy=False)
            if vectors.shape[1] != DIMENSION:
                raise ValueError(f"unexpected vector shape: {vectors.shape}")
            flat = pa.array(vectors.reshape(-1), type=pa.float32())
            table = pa.table(
                {
                    id_name: pa.array(ids, type=pa.int64()),
                    "embedding": pa.FixedSizeListArray.from_arrays(flat, DIMENSION),
                    "embedding_model": [MODEL] * len(ids),
                }
            )
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(ids)
    finally:
        if writer is not None:
            writer.close()
    return count


def convert(
    normalized_dir: Path,
    node_source: Path,
    query_source: Path,
    queries_path: Path,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import torch

    nodes = pq.read_table(
        normalized_dir / "nodes.parquet", columns=["node_id", "entity_type"]
    )
    paper_ids = set(
        pc.filter(nodes["node_id"], pc.equal(nodes["entity_type"], "paper")).to_pylist()
    )
    node_vectors = torch.load(node_source, map_location="cpu", weights_only=True)
    candidate_ids = sorted(int(value) for value in node_vectors)
    if set(candidate_ids) != paper_ids:
        missing = sorted(paper_ids - set(candidate_ids))[:10]
        extra = sorted(set(candidate_ids) - paper_ids)[:10]
        raise ValueError(
            "official candidate embeddings do not exactly cover MAG papers: "
            f"missing={missing}, extra={extra}"
        )

    embeddings_path = normalized_dir / "embeddings.parquet"
    if embeddings_path.exists():
        existing = pq.ParquetFile(embeddings_path)
        if existing.metadata.num_rows != len(candidate_ids):
            raise ValueError(
                "existing candidate embedding artifact is incomplete: "
                f"{existing.metadata.num_rows} != {len(candidate_ids)}"
            )
        existing_ids = pq.read_table(embeddings_path, columns=["node_id"])[
            "node_id"
        ].to_pylist()
        if existing_ids != candidate_ids:
            raise ValueError("existing candidate embedding IDs do not match source")
        node_rows = existing.metadata.num_rows
    else:
        node_rows = _write_vectors(
            embeddings_path,
            node_vectors,
            id_name="node_id",
            selected_ids=candidate_ids,
        )
    del node_vectors

    queries = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    official_query_vectors = torch.load(
        query_source, map_location="cpu", weights_only=True
    )
    query_vectors = {
        ordinal: official_query_vectors[int(row["source_query_id"])]
        for ordinal, row in enumerate(queries)
    }
    query_ids = list(range(len(queries)))
    temporary_path = normalized_dir / ".query_embeddings_by_ordinal.parquet"
    _write_vectors(
        temporary_path,
        query_vectors,
        id_name="query_ordinal",
        selected_ids=query_ids,
    )
    temporary = pq.read_table(temporary_path)
    query_table = temporary.rename_columns(
        ["query_id", "embedding", "embedding_model"]
    ).set_column(
        0,
        "query_id",
        pa.array([str(row["query_id"]) for row in queries], type=pa.string()),
    )
    query_embeddings_path = normalized_dir / "query_embeddings_v0.2.parquet"
    pq.write_table(query_table, query_embeddings_path, compression="zstd")
    temporary_path.unlink()

    manifest = {
        "schema_version": "e0-stark-mag-official-embeddings-v0.1.0",
        "dataset": "stark_mag",
        "model": MODEL,
        "dimension": DIMENSION,
        "environment": environment_record(),
        "coverage": {
            "policy": "official_stark_candidate_types_only",
            "candidate_entity_type": "paper",
            "graph_nodes": nodes.num_rows,
            "embedded_nodes": node_rows,
            "queries": len(queries),
            "all_papers_embedded": node_rows == len(paper_ids),
            "nonpaper_graph_anchors_intentionally_unembedded": True,
        },
        "sources": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (node_source, query_source)
        ],
        "artifacts": [
            artifact_record(path, relative_to=normalized_dir)
            for path in (embeddings_path, query_embeddings_path)
        ],
    }
    manifest["ready"] = bool(
        manifest["coverage"]["all_papers_embedded"]
        and len(queries) == query_table.num_rows
    )
    write_json(normalized_dir / "embeddings.manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("data/e0/stark_mag/normalized"),
    )
    parser.add_argument(
        "--node-source",
        type=Path,
        default=Path(
            "data/e0/stark_mag/official_embeddings/text-embedding-ada-002/"
            "doc/candidate_emb_dict.pt"
        ),
    )
    parser.add_argument(
        "--query-source",
        type=Path,
        default=Path(
            "data/e0/stark_mag/official_embeddings/text-embedding-ada-002/"
            "query/query_emb_dict.pt"
        ),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/e0/stark_mag/normalized/queries_v0.2.jsonl"),
    )
    args = parser.parse_args(argv)
    result = convert(
        args.normalized_dir, args.node_source, args.query_source, args.queries
    )
    print(json.dumps(result["coverage"], indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
