from __future__ import annotations

import json

import pytest

from tools.e0.openevolve_prepare import (
    generate_queries,
    graph_properties,
    normalize_trace,
)


def _trace(parent, child, generation, score):
    return {
        "iteration": generation,
        "timestamp": float(generation),
        "parent_id": parent,
        "child_id": child,
        "parent_metrics": {"combined_score": score - 0.1},
        "child_metrics": {"combined_score": score},
        "parent_code": f"code-{parent}",
        "child_code": f"code-{child}",
        "improvement_delta": {"combined_score": 0.1},
        "island_id": 0,
        "generation": generation,
    }


def test_openevolve_trace_normalization_and_graph_gates():
    trace = [
        _trace("root", "a", 1, 0.2),
        _trace("root", "b", 1, 0.3),
        _trace("a", "c", 2, 0.4),
        _trace("c", "d", 3, 0.5),
    ]
    nodes, edges = normalize_trace(trace)
    props = graph_properties(nodes, edges)
    assert len(nodes) == 5
    assert props["acyclic"] is True
    assert props["branching_node_count"] == 1
    assert props["max_depth"] == 3
    assert all(len(row["code_sha256"]) == 64 for row in nodes)


def test_openevolve_query_answers_are_exact_graph_neighbors():
    trace = [
        _trace("root", "a", 1, 0.2),
        _trace("root", "b", 1, 0.3),
        _trace("a", "c", 2, 0.4),
        _trace("c", "d", 3, 0.5),
    ]
    nodes, edges = normalize_trace(trace)
    queries = generate_queries(nodes, edges, limit=6)
    assert queries
    assert all(row["answer_ids"] for row in queries)
    sibling = next(row for row in queries if row["template"] == "sibling_comparison")
    assert {sibling["anchor_ids"][0], sibling["answer_ids"][0]} == {"a", "b"}


def test_stark_annotation_file_has_ten_unique_audited_queries():
    path = "configs/e0/stark_prime_pilot_annotations.jsonl"
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 10
    assert len({row["source_query_id"] for row in rows}) == 10
    assert all(row["anchor_names"] for row in rows)
    assert all(row["edge_types"] for row in rows)


def test_parquet_dependency_is_isolated():
    pytest.importorskip("pyarrow")
