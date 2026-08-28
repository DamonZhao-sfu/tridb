"""Parquet correctness backend for E0.

This backend executes the real query semantics over the prepared vectors, attributes, and
adjacency arcs. Its timing is Python timing and is therefore never valid as system latency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .model import PlanSpec, QuerySpec, quality_metrics


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _json_bytes(values: list[Any]) -> int:
    return len(json.dumps(values, ensure_ascii=False, default=str).encode("utf-8"))


class ReferenceDataset:
    """Exact cosine + native adjacency CSR + relational attributes."""

    backend_name = "parquet_reference"
    valid_for_system_latency_claims = False

    def __init__(self, name: str, spec: dict[str, Any]) -> None:
        import pyarrow.parquet as pq

        self.name = name
        nodes = pq.read_table(Path(spec["nodes"]))
        self.node_ids = np.asarray(nodes["node_id"].to_pylist(), dtype=object)
        self.id_to_idx = {value: idx for idx, value in enumerate(self.node_ids)}
        self.entity_type = np.asarray(nodes["entity_type"].to_pylist(), dtype=object)
        self.generation = (
            np.asarray(nodes["generation"].to_pylist(), dtype=np.int64)
            if "generation" in nodes.column_names
            else np.full(len(self.node_ids), -1, dtype=np.int64)
        )

        embedding_table = pq.read_table(
            Path(spec["embeddings"]), columns=["node_id", "embedding"]
        )
        vector_ids = embedding_table["node_id"].to_pylist()
        unknown = [node_id for node_id in vector_ids if node_id not in self.id_to_idx]
        if unknown:
            raise ValueError(f"{name}: embeddings contain unknown IDs: {unknown[:5]}")
        self.vector_node_indices = np.fromiter(
            (self.id_to_idx[node_id] for node_id in vector_ids), dtype=np.int64
        )
        self.vectors = np.asarray(
            embedding_table["embedding"].to_pylist(), dtype=np.float32
        )
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors /= norms

        query_table = pq.read_table(
            Path(spec["query_embeddings"]), columns=["query_id", "embedding"]
        )
        self.query_vectors = {
            str(query_id): np.asarray(vector, dtype=np.float32)
            for query_id, vector in zip(
                query_table["query_id"].to_pylist(),
                query_table["embedding"].to_pylist(),
            )
        }
        for query_id, vector in self.query_vectors.items():
            norm = float(np.linalg.norm(vector))
            if norm == 0:
                raise ValueError(f"{name}/{query_id}: zero query vector")
            self.query_vectors[query_id] = vector / norm

        self._load_graph(
            Path(spec["edges"]), bool(spec["directed_edges_already_include_reverse"])
        )
        self._order_cache: dict[str, np.ndarray] = {}
        self._rank_cache: dict[str, np.ndarray] = {}
        self._reach_cache: dict[tuple[str, int], set[int]] = {}
        self._predicate_cache: dict[str, np.ndarray] = {}

    def _load_graph(self, path: Path, reverse_already_present: bool) -> None:
        import pyarrow.parquet as pq

        edges = pq.read_table(path, columns=["src_id", "dst_id", "edge_type"])
        raw_src = edges["src_id"].to_pylist()
        raw_dst = edges["dst_id"].to_pylist()
        raw_types = [str(value) for value in edges["edge_type"].to_pylist()]
        src = np.fromiter((self.id_to_idx[value] for value in raw_src), dtype=np.int64)
        dst = np.fromiter((self.id_to_idx[value] for value in raw_dst), dtype=np.int64)

        if not reverse_already_present:
            original_src = src
            original_dst = dst
            src = np.concatenate([original_src, original_dst])
            dst = np.concatenate([original_dst, original_src])
            raw_types = raw_types + [f"{value}:reverse" for value in raw_types]

        self.type_names = sorted(set(raw_types))
        self.type_id = {value: idx for idx, value in enumerate(self.type_names)}
        edge_type = np.fromiter(
            (self.type_id[value] for value in raw_types), dtype=np.int16
        )
        order = np.argsort(src, kind="stable")
        src = src[order]
        self.graph_dst = dst[order]
        self.graph_type = edge_type[order]
        self.graph_indptr = np.searchsorted(
            src, np.arange(len(self.node_ids) + 1), side="left"
        )

    def load_queries(self, path: Path) -> list[QuerySpec]:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        queries = [QuerySpec.from_mapping(row) for row in rows]
        missing = [
            query.query_id
            for query in queries
            if query.query_id not in self.query_vectors
        ]
        if missing:
            raise ValueError(f"{self.name}: missing query embeddings: {missing[:5]}")
        return queries

    def prepare_query(self, query: QuerySpec, hops: set[int]) -> None:
        self._similarity_order(query)
        self._predicate_mask(query)
        for value in hops:
            self._reachable(query, value)

    def _similarity_order(self, query: QuerySpec) -> np.ndarray:
        if query.query_id not in self._order_cache:
            scores = self.vectors @ self.query_vectors[query.query_id]
            local_order = np.argsort(-scores, kind="stable")
            order = self.vector_node_indices[local_order]
            rank = np.full(len(self.node_ids), len(self.node_ids), dtype=np.int64)
            rank[order] = np.arange(len(order), dtype=np.int64)
            self._order_cache[query.query_id] = order
            self._rank_cache[query.query_id] = rank
        return self._order_cache[query.query_id]

    def _neighbors(self, node: int, allowed: np.ndarray) -> np.ndarray:
        lo, hi = self.graph_indptr[node], self.graph_indptr[node + 1]
        return self.graph_dst[lo:hi][np.isin(self.graph_type[lo:hi], allowed)]

    def _reachable_from(
        self, anchor: Any, edge_types: tuple[str, ...], hops: int
    ) -> set[int]:
        try:
            allowed = np.asarray(
                [self.type_id[value] for value in edge_types], dtype=np.int16
            )
        except KeyError as exc:
            raise ValueError(f"{self.name}: unknown edge type {exc.args[0]!r}") from exc
        anchor_idx = self.id_to_idx[anchor]
        seen = {anchor_idx}
        frontier = {anchor_idx}
        result: set[int] = set()
        for _ in range(hops):
            following: set[int] = set()
            for node in frontier:
                following.update(self._neighbors(node, allowed).tolist())
            frontier = following - seen
            seen.update(frontier)
            result.update(frontier)
        return result

    def _reachable(self, query: QuerySpec, hops: int) -> set[int]:
        key = (query.query_id, hops)
        if key not in self._reach_cache:
            sets = [
                self._reachable_from(anchor, query.edge_types, hops)
                for anchor in query.anchor_ids
            ]
            if query.require_each_anchor and sets:
                result = set.intersection(*sets)
            else:
                result = set().union(*sets)
            # Anchors are graph constraints, never answer candidates.  Remove every
            # anchor globally, including one reached from another anchor through a cycle.
            result.difference_update(
                self.id_to_idx[value] for value in query.anchor_ids
            )
            self._reach_cache[key] = result
        return self._reach_cache[key]

    def _predicate_mask(self, query: QuerySpec) -> np.ndarray:
        if query.query_id in self._predicate_cache:
            return self._predicate_cache[query.query_id]
        mask = np.ones(len(self.node_ids), dtype=bool)
        if query.target_entity_type:
            mask &= self.entity_type == query.target_entity_type
        predicate = query.structured_predicate
        if "generation_lte" in predicate:
            mask &= self.generation <= int(predicate["generation_lte"])
        if "generation_gte" in predicate:
            mask &= self.generation >= int(predicate["generation_gte"])
        if predicate.get("same_parent"):
            parent_type = self.type_id.get("evolved_to:reverse")
            if parent_type is None:
                mask &= False
            else:
                allowed = np.asarray([parent_type], dtype=np.int16)
                anchor_parents: set[int] = set()
                for anchor in query.anchor_ids:
                    anchor_parents.update(
                        self._neighbors(self.id_to_idx[anchor], allowed).tolist()
                    )
                sibling_mask = np.zeros(len(self.node_ids), dtype=bool)
                if anchor_parents:
                    forward = np.asarray([self.type_id["evolved_to"]], dtype=np.int16)
                    for parent in anchor_parents:
                        sibling_mask[self._neighbors(parent, forward)] = True
                for anchor in query.anchor_ids:
                    sibling_mask[self.id_to_idx[anchor]] = False
                mask &= sibling_mask
        self._predicate_cache[query.query_id] = mask
        return mask

    def valid_result_ids(self, query: QuerySpec, hops: int) -> set[Any]:
        """Return every ID satisfying the graph and relational constraints."""
        predicate = self._predicate_mask(query)
        return {
            self.node_ids[index]
            for index in self._reachable(query, hops)
            if predicate[index]
        }

    def execute(
        self, query: QuerySpec, plan: PlanSpec, *, top_n: int
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        stages = {"ann_ms": 0.0, "traverse_ms": 0.0, "filter_ms": 0.0, "merge_ms": 0.0}
        cardinality = {
            "seeds": 0,
            "reached": 0,
            "predicate_matches": 0,
            "candidates": 0,
        }
        order = self._similarity_order(query)
        rank = self._rank_cache[query.query_id]
        predicate = self._predicate_mask(query)
        cardinality["predicate_matches"] = int(predicate.sum())
        bytes_shipped = 0

        if plan.shape == "vector_first":
            stage = time.perf_counter_ns()
            seeds = order[: plan.k].tolist()
            cardinality["seeds"] = len(seeds)
            stages["ann_ms"] = _elapsed_ms(stage)
            bytes_shipped += _json_bytes([self.node_ids[idx] for idx in seeds])

            stage = time.perf_counter_ns()
            reachable = self._reachable(query, plan.hops)
            cardinality["reached"] = len(reachable)
            candidates = [idx for idx in seeds if idx in reachable]
            if plan.predicate_placement == "during":
                candidates = [idx for idx in candidates if predicate[idx]]
            stages["traverse_ms"] = _elapsed_ms(stage)
            if plan.predicate_placement == "post":
                stage = time.perf_counter_ns()
                candidates = [idx for idx in candidates if predicate[idx]]
                stages["filter_ms"] = _elapsed_ms(stage)

        elif plan.shape == "filter_first":
            stage = time.perf_counter_ns()
            filtered = np.flatnonzero(predicate)
            stages["filter_ms"] = _elapsed_ms(stage)
            bytes_shipped += _json_bytes([self.node_ids[idx] for idx in filtered])
            stage = time.perf_counter_ns()
            filtered_order = sorted(filtered.tolist(), key=rank.__getitem__)
            seeds = filtered_order[: plan.k]
            cardinality["seeds"] = len(seeds)
            stages["ann_ms"] = _elapsed_ms(stage)
            stage = time.perf_counter_ns()
            reachable = self._reachable(query, plan.hops)
            cardinality["reached"] = len(reachable)
            candidates = [idx for idx in seeds if idx in reachable]
            stages["traverse_ms"] = _elapsed_ms(stage)

        elif plan.shape == "traverse_first":
            stage = time.perf_counter_ns()
            reachable = self._reachable(query, plan.hops)
            cardinality["reached"] = len(reachable)
            candidates = list(reachable)
            if plan.predicate_placement == "during":
                candidates = [idx for idx in candidates if predicate[idx]]
            stages["traverse_ms"] = _elapsed_ms(stage)
            bytes_shipped += _json_bytes([self.node_ids[idx] for idx in candidates])
            if plan.predicate_placement == "post":
                stage = time.perf_counter_ns()
                candidates = [idx for idx in candidates if predicate[idx]]
                stages["filter_ms"] = _elapsed_ms(stage)
            stage = time.perf_counter_ns()
            candidates.sort(key=rank.__getitem__)
            candidates = candidates[: plan.k]
            cardinality["seeds"] = len(candidates)
            stages["ann_ms"] = _elapsed_ms(stage)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown plan shape: {plan.shape}")

        stage = time.perf_counter_ns()
        candidates.sort(key=rank.__getitem__)
        result_ids = [self.node_ids[idx] for idx in candidates[:top_n]]
        stages["merge_ms"] = _elapsed_ms(stage)
        cardinality["candidates"] = len(candidates)
        latency_ms = _elapsed_ms(started)
        return {
            "status": "ok",
            "backend": self.backend_name,
            "valid_for_system_latency_claims": self.valid_for_system_latency_claims,
            "latency_ms": latency_ms,
            "stage_latency_ms": stages,
            "intermediate_cardinality": cardinality,
            "round_trips": 0,
            "bytes_shipped": bytes_shipped,
            "serialization_fraction": 0.0,
            "quality": quality_metrics(result_ids, query.answer_ids),
            "result_ids": [
                value.item() if hasattr(value, "item") else value
                for value in result_ids
            ],
        }
