"""Fetch and normalize the pinned STARK-PRIME graph and E0 pilot queries."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

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
PROCESSED_ZIP = "skb/prime/processed.zip"
PROCESSED_ZIP_SHA256 = (
    "f6f265f60b761784fb7c2f052359f9cd3bf9ea20c4a307865f62ce173b294dfc"
)
QA_FILES = (
    "qa/prime/split/test.index",
    "qa/prime/split/train.index",
    "qa/prime/split/val.index",
    "qa/prime/split/test-0.1.index",
    "qa/prime/stark_qa/stark_qa.csv",
    "qa/prime/stark_qa/stark_qa_human_generated_eval.csv",
)


def fetch(root: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "stark-qa dependencies are missing; install requirements-e0.txt"
        ) from exc

    dataset_root = root / "prime"
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
    if actual_sha != PROCESSED_ZIP_SHA256:
        raise RuntimeError(
            f"STARK processed.zip checksum mismatch: {actual_sha} != {PROCESSED_ZIP_SHA256}"
        )
    processed_marker = dataset_root / "processed" / "node_info.pkl"
    if not processed_marker.exists():
        with zipfile.ZipFile(cached_zip) as archive:
            archive.extractall(dataset_root)

    copied: list[Path] = []
    for remote_path in QA_FILES:
        cached = Path(
            hf_hub_download(
                HF_REPO,
                remote_path,
                repo_type="dataset",
                revision=STARK_DATASET_REVISION,
            )
        )
        relative = Path(remote_path).relative_to("qa/prime")
        destination = dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or sha256_file(destination) != sha256_file(cached):
            shutil.copy2(cached, destination)
        copied.append(destination)

    receipt = {
        "repo": HF_REPO,
        "revision": STARK_DATASET_REVISION,
        "processed_zip": {
            "remote_path": PROCESSED_ZIP,
            "bytes": cached_zip.stat().st_size,
            "sha256": actual_sha,
        },
        "qa_files": [
            artifact_record(path, relative_to=dataset_root) for path in copied
        ],
    }
    write_json(dataset_root / "source_receipt.json", receipt)
    return receipt


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            if "source_query_id" not in row:
                raise ValueError(f"{path}:{line_number} missing source_query_id")
            rows.append(row)
    return rows


def _read_qa(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "stark_qa" / "stark_qa.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            qid = int(row["id"])
            rows[qid] = {
                "query_text": row["query"],
                "answer_ids": [
                    int(value) for value in ast.literal_eval(row["answer_ids"])
                ],
            }
    return rows


def _name_index(node_info: dict[int, dict[str, Any]]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for node_id, info in node_info.items():
        name = str(info.get("name", "")).strip().casefold()
        if name:
            index.setdefault(name, []).append(int(node_id))
    return index


def _resolve_annotations(
    annotations: list[dict[str, Any]],
    qa_rows: dict[int, dict[str, Any]],
    skb: Any,
) -> list[dict[str, Any]]:
    node_info = skb.node_info
    names = _name_index(node_info)
    queries: list[dict[str, Any]] = []
    for ordinal, annotation in enumerate(annotations):
        qid = int(annotation["source_query_id"])
        if qid not in qa_rows:
            raise ValueError(f"annotated STARK query {qid} does not exist")
        anchor_ids: list[int] = []
        for anchor_name in annotation["anchor_names"]:
            matches = names.get(str(anchor_name).casefold(), [])
            if len(matches) != 1:
                raise ValueError(
                    f"query {qid} anchor {anchor_name!r} resolved to {len(matches)} nodes"
                )
            anchor_ids.append(matches[0])
        row = qa_rows[qid]
        answer_types = {node_info[node_id]["type"] for node_id in row["answer_ids"]}
        if answer_types != {annotation["target_entity_type"]}:
            raise ValueError(
                f"query {qid} answer types {answer_types} do not match annotation"
            )
        required_anchors = (
            anchor_ids
            if annotation["structured_predicate"].get("combine_anchors")
            == "intersection"
            else anchor_ids[:1]
        )
        reachable_by_anchor = [
            _typed_reachable(
                skb,
                anchor_id,
                annotation["edge_types"],
                int(annotation["hop_limit"]),
            )
            for anchor_id in required_anchors
        ]
        all_answers_reachable = all(
            all(answer_id in reachable for reachable in reachable_by_anchor)
            for answer_id in row["answer_ids"]
        )
        if not all_answers_reachable:
            raise ValueError(
                f"query {qid} has official answers outside its audited typed-hop envelope"
            )
        queries.append(
            {
                "query_id": f"stark-prime-{ordinal:03d}",
                "dataset": "stark_prime",
                "source_query_id": qid,
                "query_text": row["query_text"],
                "answer_ids": row["answer_ids"],
                "anchor_ids": anchor_ids,
                "anchor_names": annotation["anchor_names"],
                "target_entity_type": annotation["target_entity_type"],
                "edge_types": annotation["edge_types"],
                "hop_limit": int(annotation["hop_limit"]),
                "structured_predicate": annotation["structured_predicate"],
                "template": annotation["template"],
                "annotation_status": "manually_audited_v0.1",
                "oracle_method": "official_answer_ids_plus_exact_graph_path_audit",
                "path_audit": {
                    "all_official_answers_reachable": True,
                    "required_from_each_anchor": len(required_anchors) > 1,
                },
            }
        )
    return queries


def _typed_reachable(
    skb: Any, anchor_id: int, edge_types: list[str], hop_limit: int
) -> set[int]:
    seen = {anchor_id}
    frontier = {anchor_id}
    reachable: set[int] = set()
    for _ in range(hop_limit):
        following = {
            int(neighbor)
            for node_id in frontier
            for edge_type in edge_types
            for neighbor in skb.get_neighbor_nodes(node_id, edge_type)
        }
        frontier = following - seen
        seen.update(frontier)
        reachable.update(frontier)
    return reachable


def _write_node_parquet(path: Path, skb: Any, batch_size: int = 4096) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for start in range(0, skb.num_nodes(), batch_size):
            rows = []
            for node_id in range(start, min(start + batch_size, skb.num_nodes())):
                info = dict(skb.node_info[node_id])
                details = info.pop("details", {})
                text = skb.get_doc_info(node_id, add_rel=False, compact=False)
                rows.append(
                    {
                        "node_id": node_id,
                        "dataset": "stark_prime",
                        "entity_type": str(info.get("type", "")),
                        "external_id": str(info.get("id", "")),
                        "name": str(info.get("name", "")),
                        "source": str(info.get("source", "")),
                        "text": text,
                        "attributes_json": canonical_json(details),
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


def _edge_batches(skb: Any, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    edge_index = skb.edge_index
    edge_types = skb.edge_types
    total = skb.num_edges()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        src = edge_index[0, start:stop].tolist()
        dst = edge_index[1, start:stop].tolist()
        rel = edge_types[start:stop].tolist()
        yield [
            {
                "src_id": int(source),
                "dst_id": int(target),
                "edge_type_id": int(rel_id),
                "edge_type": str(skb.edge_type_dict[int(rel_id)]),
            }
            for source, target, rel_id in zip(src, dst, rel, strict=True)
        ]


def _write_edge_parquet(path: Path, skb: Any, batch_size: int = 250_000) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for rows in _edge_batches(skb, batch_size):
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


def normalize(root: Path, output_dir: Path, annotations_path: Path) -> dict[str, Any]:
    try:
        from stark_qa.skb.prime import PrimeSKB
    except ImportError as exc:
        raise RuntimeError("stark-qa is required; install requirements-e0.txt") from exc

    dataset_root = root / "prime"
    receipt_path = dataset_root / "source_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"missing {receipt_path}; run the fetch phase first")
    skb = PrimeSKB(root=str(dataset_root), download_processed=False)
    qa_rows = _read_qa(dataset_root)
    annotations = _load_annotations(annotations_path)
    queries = _resolve_annotations(annotations, qa_rows, skb)

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "nodes.parquet"
    edges_path = output_dir / "edges.parquet"
    queries_path = output_dir / "queries_v0.1.jsonl"
    node_count = _write_node_parquet(nodes_path, skb)
    edge_count = _write_edge_parquet(edges_path, skb)
    write_jsonl(queries_path, queries)

    gates = {
        "node_count_matches_official": node_count == 129_375,
        "adjacency_arc_count_matches_official": edge_count == 8_100_498,
        "node_type_count_matches_official": len(skb.node_type_dict) == 10,
        "edge_type_count_matches_official": len(skb.edge_type_dict) == 18,
        "pilot_query_count_is_10": len(queries) == 10,
        "all_queries_manually_audited": all(
            row["annotation_status"] == "manually_audited_v0.1" for row in queries
        ),
        "all_official_answers_in_typed_hop_envelope": all(
            row["path_audit"]["all_official_answers_reachable"] for row in queries
        ),
    }
    manifest = {
        "schema_version": "e0-stark-prime-v0.1.0",
        "dataset": "stark_prime",
        "source": json.loads(receipt_path.read_text(encoding="utf-8")),
        "environment": environment_record(),
        "graph_direction_policy": (
            "Official stark-qa default: each PrimeKG relation is represented as two "
            "directed adjacency arcs for bidirectional traversal."
        ),
        "counts": {
            "nodes": node_count,
            "adjacency_arcs": edge_count,
            "node_types": len(skb.node_type_dict),
            "edge_types": len(skb.edge_type_dict),
            "qa_rows": len(qa_rows),
            "pilot_queries": len(queries),
        },
        "integrity_gates": gates,
        "ready_for_embedding": all(gates.values()),
        "ready_for_e0": False,
        "ready_for_e0_note": "Set true only after the pinned embeddings are complete.",
        "artifacts": [
            artifact_record(path, relative_to=output_dir)
            for path in (nodes_path, edges_path, queries_path)
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("fetch", "normalize", "all"), nargs="?", default="all"
    )
    parser.add_argument("--root", type=Path, default=Path("data/e0/stark_prime/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/e0/stark_prime/normalized"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("configs/e0/stark_prime_pilot_annotations.jsonl"),
    )
    args = parser.parse_args(argv)
    if args.phase in {"fetch", "all"}:
        receipt = fetch(args.root)
        print("fetched_revision=", receipt["revision"])
    if args.phase in {"normalize", "all"}:
        manifest = normalize(args.root, args.output_dir, args.annotations)
        print(json.dumps(manifest["counts"], sort_keys=True))
        if not manifest["ready_for_embedding"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
