"""Normalize an OpenEvolve 0.3.2 JSONL evolution trace for E0.

The raw trace remains immutable. This module creates a node table, a lineage-edge
table, ten deterministic vector-seeded graph query templates, and a manifest with
integrity-gate results. It deliberately does not infer a tree: island migration and
selection make the safe abstraction a directed lineage graph.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from tools.e0 import OPENEVOLVE_VERSION
from tools.e0.common import (
    artifact_record,
    canonical_json,
    environment_record,
    sha256_file,
    write_json,
    write_jsonl,
)


def _numeric_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                out[str(key)] = number
    return out


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            for required in ("parent_id", "child_id", "iteration"):
                if required not in row:
                    raise ValueError(f"{path}:{line_number} missing {required}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty evolution trace: {path}")
    return rows


def normalize_trace(
    trace_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def merge_node(
        node_id: str,
        *,
        code: str | None,
        metrics: dict[str, Any] | None,
        generation: int | None,
        island_id: int | None,
        timestamp: float | None,
        iteration: int | None,
        changes_description: str | None,
    ) -> None:
        numeric = _numeric_metrics(metrics)
        primary = numeric.get("combined_score")
        current = nodes.setdefault(
            node_id,
            {
                "node_id": node_id,
                "dataset": "openevolve_function_minimization",
                "entity_type": "program",
                "generation": generation,
                "island_id": island_id,
                "timestamp": timestamp,
                "iteration_found": iteration,
                "language": "python",
                "code": code or "",
                "changes_description": changes_description or "",
                "metrics_json": canonical_json(metrics or {}),
                "combined_score": primary,
                "complexity": numeric.get("complexity"),
                "metadata_json": "{}",
            },
        )
        if code and not current["code"]:
            current["code"] = code
        if numeric and current["metrics_json"] == "{}":
            current["metrics_json"] = canonical_json(metrics or {})
            current["combined_score"] = primary
            current["complexity"] = numeric.get("complexity")
        for key, value in (
            ("generation", generation),
            ("island_id", island_id),
            ("timestamp", timestamp),
            ("iteration_found", iteration),
        ):
            if current[key] is None and value is not None:
                current[key] = value

    for trace in trace_rows:
        parent_id = str(trace["parent_id"])
        child_id = str(trace["child_id"])
        iteration = int(trace["iteration"])
        timestamp = float(trace.get("timestamp", 0.0))
        island_id = trace.get("island_id")
        generation = trace.get("generation")
        merge_node(
            parent_id,
            code=trace.get("parent_code"),
            metrics=trace.get("parent_metrics"),
            generation=None if generation is None else max(0, int(generation) - 1),
            island_id=island_id,
            timestamp=None,
            iteration=None,
            changes_description=trace.get("parent_changes_description"),
        )
        merge_node(
            child_id,
            code=trace.get("child_code"),
            metrics=trace.get("child_metrics"),
            generation=None if generation is None else int(generation),
            island_id=island_id,
            timestamp=timestamp,
            iteration=iteration,
            changes_description=trace.get("child_changes_description"),
        )
        edges.append(
            {
                "src_id": parent_id,
                "dst_id": child_id,
                "edge_type": "evolved_to",
                "iteration": iteration,
                "timestamp": timestamp,
                "island_id": island_id,
                "generation": generation,
                "improvement_delta_json": canonical_json(
                    trace.get("improvement_delta") or {}
                ),
                "code_diff": trace.get("code_diff") or "",
            }
        )

    for node in nodes.values():
        node["code_sha256"] = sha256_text(node["code"])
    return sorted(nodes.values(), key=lambda row: row["node_id"]), edges


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def graph_properties(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    node_ids = {row["node_id"] for row in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    duplicate_edges = 0
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        pair = (edge["src_id"], edge["dst_id"])
        if pair in seen_edges:
            duplicate_edges += 1
        seen_edges.add(pair)
        children[pair[0]].append(pair[1])
        if pair[1] in indegree:
            indegree[pair[1]] += 1

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    depth = {node_id: 0 for node_id in queue}
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for child in children.get(node_id, []):
            depth[child] = max(depth.get(child, 0), depth[node_id] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return {
        "all_edge_endpoints_resolve": all(
            edge["src_id"] in node_ids and edge["dst_id"] in node_ids for edge in edges
        ),
        "acyclic": visited == len(node_ids),
        "duplicate_edges": duplicate_edges,
        "root_count": sum(
            1 for row in nodes if row["node_id"] in depth and depth[row["node_id"]] == 0
        ),
        "branching_node_count": sum(
            1 for values in children.values() if len(set(values)) >= 2
        ),
        "max_depth": max(depth.values(), default=0),
        "nodes_with_code": sum(bool(row["code"].strip()) for row in nodes),
        "nodes_with_numeric_metrics": sum(row["metrics_json"] != "{}" for row in nodes),
    }


def _ancestors(node_id: str, parents: dict[str, list[str]], hops: int) -> set[str]:
    frontier = {node_id}
    out: set[str] = set()
    for _ in range(hops):
        frontier = {p for node in frontier for p in parents.get(node, [])} - out
        out.update(frontier)
    return out


def _descendants(node_id: str, children: dict[str, list[str]], hops: int) -> set[str]:
    frontier = {node_id}
    out: set[str] = set()
    for _ in range(hops):
        frontier = {c for node in frontier for c in children.get(node, [])} - out
        out.update(frontier)
    return out


def generate_queries(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        parents[edge["dst_id"]].append(edge["src_id"])
        children[edge["src_id"]].append(edge["dst_id"])

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates = sorted(
        nodes,
        key=lambda row: (
            -(row["generation"] if row["generation"] is not None else -1),
            row["node_id"],
        ),
    )
    templates = ("ancestor_reuse", "sibling_comparison", "descendant_search")
    for template in templates:
        for node in candidates:
            anchor = node["node_id"]
            hop_limit = 2
            if template == "ancestor_reuse":
                answers = _ancestors(anchor, parents, hop_limit)
                edge_types = ["evolved_to:reverse"]
                predicate = {"generation_lte": node.get("generation")}
            elif template == "sibling_comparison":
                parent_ids = parents.get(anchor, [])
                answers = {
                    sibling
                    for parent in parent_ids
                    for sibling in children.get(parent, [])
                    if sibling != anchor
                }
                edge_types = ["evolved_to:reverse", "evolved_to"]
                predicate = {"same_parent": True}
            else:
                answers = _descendants(anchor, children, hop_limit)
                edge_types = ["evolved_to"]
                predicate = {"generation_gte": node.get("generation")}
            if not answers:
                continue
            code_preview = " ".join(node["code"].split())[:600]
            buckets[template].append(
                {
                    "query_id": "",
                    "dataset": "openevolve_function_minimization",
                    "source_query_id": None,
                    "query_text": f"Find evolution versions related to: {code_preview}",
                    "answer_ids": sorted(answers),
                    "anchor_ids": [anchor],
                    "target_entity_type": "program",
                    "edge_types": edge_types,
                    "hop_limit": hop_limit,
                    "structured_predicate": predicate,
                    "template": template,
                    "annotation_status": "generated_exact_oracle",
                    "oracle_method": "exact_lineage_expansion",
                }
            )

    queries: list[dict[str, Any]] = []
    offset = 0
    while len(queries) < limit:
        added = False
        for template in templates:
            if offset < len(buckets[template]):
                query = buckets[template][offset]
                query["query_id"] = f"oe-{len(queries):03d}"
                queries.append(query)
                added = True
                if len(queries) == limit:
                    break
        if not added:
            break
        offset += 1
    return queries


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install requirements-e0.txt") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def prepare(trace: Path, output_dir: Path, query_count: int = 10) -> dict[str, Any]:
    trace_rows = read_trace(trace)
    nodes, edges = normalize_trace(trace_rows)
    properties = graph_properties(nodes, edges)
    queries = generate_queries(nodes, edges, query_count)
    gates = {
        "all_parent_references_resolve": properties["all_edge_endpoints_resolve"],
        "lineage_is_acyclic": properties["acyclic"],
        "all_nodes_have_code": properties["nodes_with_code"] == len(nodes),
        "all_nodes_have_numeric_metrics": properties["nodes_with_numeric_metrics"]
        == len(nodes),
        "has_branching": properties["branching_node_count"] >= 1,
        "max_depth_at_least_3": properties["max_depth"] >= 3,
        "query_count_reached": len(queries) == query_count,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    node_path = output_dir / "nodes.parquet"
    edge_path = output_dir / "edges.parquet"
    query_path = output_dir / "queries.jsonl"
    _write_parquet(node_path, nodes)
    _write_parquet(edge_path, edges)
    write_jsonl(query_path, queries)
    manifest = {
        "schema_version": "e0-openevolve-v0.1.0",
        "dataset": "openevolve_function_minimization",
        "source": {
            "package": "openevolve",
            "version": OPENEVOLVE_VERSION,
            "example": "function_minimization (adapted from official v0.3.2 example)",
            "raw_trace": str(trace),
            "raw_trace_sha256": sha256_file(trace),
            "model_receipt": (
                json.loads(
                    (trace.parent / "model_receipt.json").read_text(encoding="utf-8")
                )
                if (trace.parent / "model_receipt.json").exists()
                else None
            ),
            "run_config": (
                {
                    "path": str(trace.parent / "run_config.yaml"),
                    "sha256": sha256_file(trace.parent / "run_config.yaml"),
                }
                if (trace.parent / "run_config.yaml").exists()
                else None
            ),
        },
        "environment": environment_record(),
        "counts": {
            "trace_rows": len(trace_rows),
            "nodes": len(nodes),
            "edges": len(edges),
            "queries": len(queries),
        },
        "graph_properties": properties,
        "integrity_gates": gates,
        "ready_for_e0": all(gates.values()),
        "artifacts": [
            artifact_record(path, relative_to=output_dir)
            for path in (node_path, edge_path, query_path)
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=10)
    args = parser.parse_args(argv)
    manifest = prepare(args.trace, args.output_dir, args.query_count)
    print(json.dumps(manifest["counts"], sort_keys=True))
    print("ready_for_e0=", manifest["ready_for_e0"])
    return 0 if manifest["ready_for_e0"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
