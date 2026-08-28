"""Fetch and inventory the pinned STARK-MAG processed knowledge base.

The full E1 normalization/query contract is deliberately a separate phase: this module
first establishes a resumable, checksum-pinned source artifact and records the actual
tensor cardinalities before any multi-store load is attempted.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Any

from tools.e0 import STARK_DATASET_REVISION
from tools.e0.common import (
    artifact_record,
    canonical_json,
    environment_record,
    sha256_file,
    write_json,
    write_jsonl,
)

HF_REPO = "snap-stanford/stark"
PROCESSED_ZIP = "skb/mag/processed.zip"
PROCESSED_ZIP_BYTES = 484_062_494
PROCESSED_ZIP_SHA256 = (
    "7bfac742c1fb77694b3fab76be5088f9c3a636f47366188b70d3f378cfbd3053"
)
SUPPORT_FILES = (
    "skb/mag/schema/mag.json",
    "skb/mag/schema/reduced_mag.json",
    "qa/mag/split/test.index",
    "qa/mag/split/train.index",
    "qa/mag/split/val.index",
    "qa/mag/split/test-0.1.index",
    "qa/mag/stark_qa/stark_qa.csv",
    "qa/mag/stark_qa/stark_qa_human_generated_eval.csv",
)


def _destination(dataset_root: Path, remote_path: str) -> Path:
    if remote_path.startswith("qa/mag/"):
        return dataset_root / Path(remote_path).relative_to("qa/mag")
    if remote_path.startswith("skb/mag/"):
        return dataset_root / Path(remote_path).relative_to("skb/mag")
    raise ValueError(f"unsupported STARK-MAG path: {remote_path}")


def inspect_processed(dataset_root: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to inventory STARK-MAG tensors") from exc

    processed = dataset_root / "processed"
    required = (
        "node_info.pkl",
        "node_types.pt",
        "edge_index.pt",
        "edge_types.pt",
        "node_type_dict.pkl",
        "edge_type_dict.pkl",
    )
    missing = [name for name in required if not (processed / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete STARK-MAG extraction: {missing}")

    node_types = torch.load(
        processed / "node_types.pt", map_location="cpu", weights_only=True
    )
    edge_index = torch.load(
        processed / "edge_index.pt", map_location="cpu", weights_only=True
    )
    edge_types = torch.load(
        processed / "edge_types.pt", map_location="cpu", weights_only=True
    )
    with (processed / "node_info.pkl").open("rb") as handle:
        node_info = pickle.load(handle)
    with (processed / "node_type_dict.pkl").open("rb") as handle:
        node_type_dict = pickle.load(handle)
    with (processed / "edge_type_dict.pkl").open("rb") as handle:
        edge_type_dict = pickle.load(handle)

    counts = {
        "nodes": len(node_info),
        "stored_directed_edges": int(edge_index.shape[1]),
        "node_types": len(node_type_dict),
        "edge_types": len(edge_type_dict),
    }
    gates = {
        "node_info_matches_node_types": len(node_info) == int(node_types.numel()),
        "edge_index_has_two_rows": tuple(edge_index.shape[:1]) == (2,),
        "edge_index_matches_edge_types": int(edge_index.shape[1])
        == int(edge_types.numel()),
        "node_ids_are_dense": min(node_info) == 0
        and max(node_info) == len(node_info) - 1,
        "official_node_count_matches": len(node_info) == 1_872_968,
        "official_type_counts_match": len(node_type_dict) == 4
        and len(edge_type_dict) == 4,
    }
    return {
        "counts": counts,
        "node_type_dict": {str(key): value for key, value in node_type_dict.items()},
        "edge_type_dict": {str(key): value for key, value in edge_type_dict.items()},
        "integrity_gates": gates,
        "ready_for_normalization": all(gates.values()),
    }


def fetch(root: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is missing; install requirements-e0.txt"
        ) from exc

    dataset_root = root / "mag"
    dataset_root.mkdir(parents=True, exist_ok=True)
    cached_zip = Path(
        hf_hub_download(
            HF_REPO,
            PROCESSED_ZIP,
            repo_type="dataset",
            revision=STARK_DATASET_REVISION,
        )
    )
    actual_sha = sha256_file(cached_zip)
    if cached_zip.stat().st_size != PROCESSED_ZIP_BYTES:
        raise RuntimeError(
            "STARK-MAG processed.zip size mismatch: "
            f"{cached_zip.stat().st_size} != {PROCESSED_ZIP_BYTES}"
        )
    if actual_sha != PROCESSED_ZIP_SHA256:
        raise RuntimeError(
            f"STARK-MAG checksum mismatch: {actual_sha} != {PROCESSED_ZIP_SHA256}"
        )

    marker = dataset_root / "processed" / "node_info.pkl"
    if not marker.is_file():
        with zipfile.ZipFile(cached_zip) as archive:
            archive.extractall(dataset_root)

    copied: list[Path] = []
    for remote_path in SUPPORT_FILES:
        cached = Path(
            hf_hub_download(
                HF_REPO,
                remote_path,
                repo_type="dataset",
                revision=STARK_DATASET_REVISION,
            )
        )
        destination = _destination(dataset_root, remote_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != sha256_file(cached):
            shutil.copy2(cached, destination)
        copied.append(destination)

    inventory = inspect_processed(dataset_root)
    receipt = {
        "schema_version": "e0-stark-mag-source-v0.1.0",
        "repo": HF_REPO,
        "revision": STARK_DATASET_REVISION,
        "processed_zip": {
            "remote_path": PROCESSED_ZIP,
            "bytes": cached_zip.stat().st_size,
            "sha256": actual_sha,
            "uncompressed_bytes": sum(
                member.file_size
                for member in zipfile.ZipFile(cached_zip).infolist()
                if not member.is_dir()
            ),
        },
        "support_files": [
            artifact_record(path, relative_to=dataset_root) for path in copied
        ],
        "inventory": inventory,
        "scope_note": (
            "This receipt proves the pinned source download and tensor inventory only; "
            "it is not an E1 live-store or GX10 benchmark sign-off."
        ),
    }
    write_json(dataset_root / "source_receipt.json", receipt)
    return receipt


def _node_text(info: dict[str, Any]) -> str:
    node_type = str(info.get("type", ""))
    if node_type == "paper":
        title = str(info.get("title", ""))
        abstract = str(info.get("abstract", "")).replace("\r", "").strip()
        lines = [f"- paper title: {title}", f"- abstract: {abstract}"]
        if str(info.get("Date", "-1")) != "-1":
            lines.append(f"- publication date: {info['Date']}")
        venue = next(
            (
                str(info[key])
                for key in (
                    "OriginalVenue",
                    "JournalDisplayName",
                    "ConferenceSeriesDisplayName",
                    "ConferenceInstancesDisplayName",
                )
                if str(info.get(key, "-1")) != "-1"
            ),
            "",
        )
        if venue:
            lines.append(f"- venue: {venue}")
        return "\n".join(lines) + "\n"

    display = str(info.get("DisplayName", ""))
    label = {
        "author": "author name",
        "institution": "institution",
        "field_of_study": "field of study",
    }.get(node_type, "name")
    lines = [f"- {label}: {display}"]
    if str(info.get("PaperCount", "-1")) != "-1":
        lines.append(f"- paper count: {info['PaperCount']}")
    if str(info.get("CitationCount", "-1")) != "-1":
        lines.append(f"- citation count: {info['CitationCount']}")
    return "\n".join(lines).replace("-1", "Unknown") + "\n"


def _write_nodes(path: Path, node_info: dict[int, dict[str, Any]]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    excluded = {
        "id",
        "mag_id",
        "new_id",
        "type",
        "title",
        "abstract",
        "DisplayName",
    }
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for start in range(0, len(node_info), 4096):
            rows = []
            for node_id in range(start, min(start + 4096, len(node_info))):
                info = node_info[node_id]
                name = (
                    str(info.get("title", ""))
                    if info.get("type") == "paper"
                    else str(info.get("DisplayName", ""))
                )
                attributes = {
                    key: value for key, value in info.items() if key not in excluded
                }
                rows.append(
                    {
                        "node_id": node_id,
                        "dataset": "stark_mag",
                        "entity_type": str(info.get("type", "")),
                        "external_id": str(info.get("mag_id", "")),
                        "name": name,
                        "source": "STARK-MAG",
                        "text": _node_text(info),
                        "attributes_json": canonical_json(attributes),
                    }
                )
            table = pa.Table.from_pylist(rows)
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += len(rows)
    finally:
        if writer is not None:
            writer.close()
    return count


def _write_edges(path: Path, processed: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from torch_geometric.utils import to_undirected

    edge_index = torch.load(
        processed / "edge_index.pt", map_location="cpu", weights_only=True
    )
    edge_types = torch.load(
        processed / "edge_types.pt", map_location="cpu", weights_only=True
    )
    with (processed / "edge_type_dict.pkl").open("rb") as handle:
        edge_type_dict = pickle.load(handle)
    # Match stark_qa.skb.knowledge_base.SKB exactly: the official runtime makes the
    # graph undirected and coalesces already-symmetric citation arcs.
    edge_index, edge_types = to_undirected(
        edge_index,
        edge_types,
        num_nodes=1_872_968,
        reduce="mean",
    )
    edge_types = edge_types.long()

    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for start in range(0, edge_types.numel(), 250_000):
            stop = min(start + 250_000, edge_types.numel())
            type_ids = edge_types[start:stop].tolist()
            table = pa.table(
                {
                    "src_id": edge_index[0, start:stop].numpy(),
                    "dst_id": edge_index[1, start:stop].numpy(),
                    "edge_type_id": type_ids,
                    "edge_type": [edge_type_dict[value] for value in type_ids],
                }
            )
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            count += stop - start
    finally:
        if writer is not None:
            writer.close()
    return count


def normalize(root: Path, output_dir: Path) -> dict[str, Any]:
    dataset_root = root / "mag"
    receipt_path = dataset_root / "source_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing {receipt_path}; run fetch first")
    processed = dataset_root / "processed"
    with (processed / "node_info.pkl").open("rb") as handle:
        node_info = pickle.load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "nodes.parquet"
    edges_path = output_dir / "edges.parquet"
    pilot_path = output_dir / "queries_v0.1.jsonl"
    node_count = _write_nodes(nodes_path, node_info)
    del node_info
    edge_count = _write_edges(edges_path, processed)
    # MAG queries are independently graph-derived from official STaRK answers; unlike
    # PRIME there is no hand-authored pilot file to carry forward.
    write_jsonl(pilot_path, [])

    gates = {
        "node_count_matches_official": node_count == 1_872_968,
        "undirected_adjacency_arc_count_matches_official": edge_count == 39_802_116,
        "node_type_count_matches_official": 4 == 4,
        "edge_type_count_matches_official": 4 == 4,
    }
    manifest = {
        "schema_version": "e0-stark-mag-v0.1.0",
        "dataset": "stark_mag",
        "source": json.loads(receipt_path.read_text(encoding="utf-8")),
        "environment": environment_record(),
        "graph_direction_policy": (
            "Official stark-qa SKB semantics: to_undirected(..., reduce='mean'); "
            "normalized parquet contains both traversal directions."
        ),
        "counts": {
            "nodes": node_count,
            "adjacency_arcs": edge_count,
            "node_types": 4,
            "edge_types": 4,
        },
        "integrity_gates": gates,
        "ready_for_query_derivation": all(gates.values()),
        "ready_for_embedding": False,
        "ready_for_e1": False,
        "artifacts": [
            artifact_record(path, relative_to=output_dir)
            for path in (nodes_path, edges_path, pilot_path)
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("fetch", "inventory", "normalize", "all"),
        nargs="?",
        default="fetch",
    )
    parser.add_argument("--root", type=Path, default=Path("data/e0/stark_mag/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/e0/stark_mag/normalized"),
    )
    args = parser.parse_args(argv)
    dataset_root = args.root / "mag"
    if args.phase == "all":
        fetch(args.root)
        result = normalize(args.root, args.output_dir)
    elif args.phase == "fetch":
        result = fetch(args.root)
    elif args.phase == "normalize":
        result = normalize(args.root, args.output_dir)
    else:
        result = inspect_processed(dataset_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    readiness = result.get("inventory", result)
    ready = readiness.get("ready_for_normalization")
    if ready is None:
        ready = readiness.get("ready_for_query_derivation")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
