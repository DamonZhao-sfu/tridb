"""Polyglot-Tuned live adapter: Milvus + Neo4j + pgvector on one host."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .model import PlanSpec, QuerySpec, quality_metrics


def _ms(start: int) -> float:
    return (time.perf_counter_ns() - start) / 1_000_000.0


def _bytes(values: list[Any]) -> int:
    return len(json.dumps(values, ensure_ascii=False, default=str).encode())


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"unsafe database identifier: {value!r}")
    return value


def _relationship(value: str) -> str:
    return _identifier(re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower())


def _empty_stores(*, milvus: int, neo4j: int, postgres: int) -> list[str]:
    """Which of the three Polyglot legs hold zero rows for this dataset.

    Pure decision logic, deliberately split out of `PolyglotLiveDataset.__init__`
    so it can be unit-tested with plain integers, with no live store required.
    An empty leg here is indistinguishable, from the caller's side, from a
    slow/cold index — the E0 v0.2 OpenEvolve run measured against exactly this
    state (every leg legitimately empty, because its polyglot loader ran after
    the measurement) and it produced 1,010 well-formed, silently empty
    observations instead of an error.
    """
    empty = []
    if milvus == 0:
        empty.append("milvus")
    if neo4j == 0:
        empty.append("neo4j")
    if postgres == 0:
        empty.append("postgres")
    return empty


# The runnable command that (re)populates each dataset's Polyglot stores. Keyed
# by dataset name rather than hardcoded into the guard's error message, so
# adding a fourth E0 dataset means adding one entry here, not editing the
# raise. See _loader_command below for the fallback when a name isn't listed.
_POLYGLOT_LOADERS: dict[str, str] = {
    "openevolve": "make e0-openevolve-polyglot-load",
    "stark_prime": "python -m tools.e0.load_polyglot all",
}


def _loader_command(name: str) -> str:
    """The remedy to print alongside an empty-store error: a command an
    operator can actually run, not just a diagnosis. Falls back to listing
    every known loader if `name` isn't in `_POLYGLOT_LOADERS` yet, so a new
    dataset never regresses to a silent dead end.
    """
    if name in _POLYGLOT_LOADERS:
        return _POLYGLOT_LOADERS[name]
    return "; ".join(f"{n}: {cmd}" for n, cmd in _POLYGLOT_LOADERS.items())


class PolyglotLiveDataset:
    backend_name = "polyglot_live"
    valid_for_system_latency_claims = True

    def __init__(self, name: str, spec: dict[str, Any]) -> None:
        import psycopg
        import pyarrow.parquet as pq
        from neo4j import GraphDatabase
        from pymilvus import Collection, connections

        live = spec["live"]
        self.name = name
        self.label = _identifier(str(live["neo4j_label"]))
        self.table = _identifier(str(live["postgres_table"]))
        self.id_type = str(live["id_type"])
        connections.connect(alias=f"e0_{name}", host="127.0.0.1", port="19530")
        self.collection = Collection(str(live["milvus_collection"]), using=f"e0_{name}")
        self.collection.load()
        milvus_count = self.collection.num_entities

        self.neo_driver = GraphDatabase.driver(
            "bolt://127.0.0.1:7688", auth=("neo4j", "testpassword")
        )
        with self.neo_driver.session() as session:
            neo4j_count = session.run(
                f"MATCH (n:{self.label}) RETURN count(n) AS n"
            ).single()["n"]

        self.pg = psycopg.connect(
            host="127.0.0.1",
            port=5434,
            dbname="tridb_wiki",
            user="postgres",
            password="postgres",
            autocommit=True,
        )
        with self.pg.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {self.table}")
            postgres_count = cursor.fetchone()[0]

        empty = _empty_stores(
            milvus=milvus_count, neo4j=neo4j_count, postgres=postgres_count
        )
        if empty:
            self.pg.close()
            self.neo_driver.close()
            raise RuntimeError(
                f"{name}: {', '.join(empty)} store(s) hold 0 rows — run "
                f"`{_loader_command(name)}` before constructing this backend"
            )
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
            norm = float(np.linalg.norm(vector)) or 1.0
            self.query_vectors[query_id] = vector / norm

        self._parent_cache: dict[tuple[Any, ...], list[Any]] = {}

    def close(self) -> None:
        self.pg.close()
        self.neo_driver.close()

    def load_queries(self, path: Path) -> list[QuerySpec]:
        return [
            QuerySpec.from_mapping(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def prepare_query(self, query: QuerySpec, hops: set[int]) -> None:
        if query.query_id not in self.query_vectors:
            raise ValueError(f"{self.name}: missing query vector {query.query_id}")

    def _vector_literal(self, query: QuerySpec) -> str:
        return (
            "["
            + ",".join(f"{value:.8g}" for value in self.query_vectors[query.query_id])
            + "]"
        )

    def _milvus(self, query: QuerySpec, k: int) -> list[Any]:
        result = self.collection.search(
            data=[self.query_vectors[query.query_id].tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": max(64, k)}},
            limit=k,
        )
        return [hit.id for hit in result[0]]

    def _parents_of(self, anchor_ids: tuple[Any, ...]) -> list[Any]:
        key = tuple(anchor_ids)
        cached = self._parent_cache.get(key)
        if cached is not None:
            return cached
        with self.pg.cursor() as cursor:
            cursor.execute(
                f"SELECT DISTINCT parent_id FROM {self.table} "
                "WHERE node_id = ANY(%s) AND parent_id IS NOT NULL",
                (list(anchor_ids),),
            )
            parents = [row[0] for row in cursor.fetchall()]
        self._parent_cache[key] = parents
        return parents

    def _predicate_sql(self, query: QuerySpec) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if query.target_entity_type:
            clauses.append("entity_type = %s")
            params.append(query.target_entity_type)
        predicate = query.structured_predicate
        if "generation_lte" in predicate:
            clauses.append("generation <= %s")
            params.append(int(predicate["generation_lte"]))
        if "generation_gte" in predicate:
            clauses.append("generation >= %s")
            params.append(int(predicate["generation_gte"]))
        if predicate.get("same_parent"):
            parents = self._parents_of(query.anchor_ids)
            if not parents:
                clauses.append("FALSE")
            else:
                clauses.append("parent_id = ANY(%s)")
                params.append(list(parents))
                clauses.append("NOT (node_id = ANY(%s))")
                params.append(list(query.anchor_ids))
        return (" AND ".join(clauses) if clauses else "TRUE"), params

    def _pg_filter(self, query: QuerySpec, ids: list[Any]) -> list[Any]:
        if not ids:
            return []
        predicate, params = self._predicate_sql(query)
        sql = (
            f"SELECT node_id FROM {self.table} WHERE node_id = ANY(%s) AND {predicate}"
        )
        with self.pg.cursor() as cursor:
            cursor.execute(sql, [ids, *params])
            keep = {row[0] for row in cursor.fetchall()}
        return [node_id for node_id in ids if node_id in keep]

    def _pg_rank(
        self, query: QuerySpec, k: int, ids: list[Any] | None = None
    ) -> list[Any]:
        predicate, params = self._predicate_sql(query)
        clauses = [predicate]
        values: list[Any] = list(params)
        if ids is not None:
            if not ids:
                return []
            clauses.insert(0, "node_id = ANY(%s)")
            values.insert(0, ids)
        values.extend([self._vector_literal(query), k])
        sql = (
            f"SELECT node_id FROM {self.table} WHERE {' AND '.join(clauses)} "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        with self.pg.cursor() as cursor:
            cursor.execute(sql, values)
            return [row[0] for row in cursor.fetchall()]

    def _neo_predicate(self, query: QuerySpec) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if query.target_entity_type:
            clauses.append("b.entity_type = $target_type")
            params["target_type"] = query.target_entity_type
        predicate = query.structured_predicate
        if "generation_lte" in predicate:
            clauses.append("b.generation <= $generation_lte")
            params["generation_lte"] = int(predicate["generation_lte"])
        if "generation_gte" in predicate:
            clauses.append("b.generation >= $generation_gte")
            params["generation_gte"] = int(predicate["generation_gte"])
        if predicate.get("same_parent"):
            parents = self._parents_of(query.anchor_ids)
            if not parents:
                clauses.append("FALSE")
            else:
                clauses.append("b.parent_id IN $same_parents")
                params["same_parents"] = list(parents)
                clauses.append("NOT b.node_id IN $anchors")
        return (" AND ".join(clauses) if clauses else "TRUE"), params

    def _neo_reach(
        self,
        query: QuerySpec,
        hops: int,
        *,
        restrict_ids: list[Any] | None,
        apply_predicate: bool,
    ) -> list[Any]:
        relationships = "|".join(_relationship(value) for value in query.edge_types)
        where = ["TRUE"]
        params: dict[str, Any] = {
            "anchors": list(query.anchor_ids),
            "required": len(query.anchor_ids) if query.require_each_anchor else 1,
        }
        if restrict_ids is not None:
            if not restrict_ids:
                return []
            where.append("b.node_id IN $restrict_ids")
            params["restrict_ids"] = restrict_ids
        where.append("NOT b.node_id IN $anchors")
        if apply_predicate:
            predicate, predicate_params = self._neo_predicate(query)
            where.append(predicate)
            params.update(predicate_params)
        cypher = (
            f"MATCH (a:{self.label}) WHERE a.node_id IN $anchors "
            f"MATCH (a)-[:{relationships}*1..{int(hops)}]->(b:{self.label}) "
            f"WHERE {' AND '.join(where)} "
            "WITH b, count(DISTINCT a) AS anchor_hits "
            "WHERE anchor_hits >= $required RETURN DISTINCT b.node_id AS node_id"
        )
        with self.neo_driver.session() as session:
            return [row["node_id"] for row in session.run(cypher, **params)]

    def execute(
        self, query: QuerySpec, plan: PlanSpec, *, top_n: int
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        stages = {"ann_ms": 0.0, "traverse_ms": 0.0, "filter_ms": 0.0, "merge_ms": 0.0}
        cardinality = {
            "seeds": 0,
            "reached": 0,
            "predicate_matches": None,
            "candidates": 0,
        }
        round_trips = 0
        bytes_shipped = 0
        serialization_ms = 0.0

        def ship(values: list[Any]) -> None:
            nonlocal bytes_shipped, serialization_ms
            serialization_start = time.perf_counter_ns()
            bytes_shipped += _bytes(values)
            serialization_ms += _ms(serialization_start)

        if plan.shape == "vector_first":
            stage = time.perf_counter_ns()
            seeds = self._milvus(query, plan.k)
            stages["ann_ms"] = _ms(stage)
            round_trips += 1
            cardinality["seeds"] = len(seeds)
            ship(seeds)
            if plan.predicate_placement == "during":
                stage = time.perf_counter_ns()
                seeds = self._pg_filter(query, seeds)
                stages["filter_ms"] = _ms(stage)
                round_trips += 1
                ship(seeds)
            stage = time.perf_counter_ns()
            candidates = self._neo_reach(
                query, plan.hops, restrict_ids=seeds, apply_predicate=False
            )
            stages["traverse_ms"] = _ms(stage)
            round_trips += 1
            cardinality["reached"] = len(candidates)
            ship(candidates)
            if plan.predicate_placement == "post":
                stage = time.perf_counter_ns()
                candidates = self._pg_filter(query, candidates)
                stages["filter_ms"] = _ms(stage)
                round_trips += 1
                ship(candidates)
            rank = {node_id: idx for idx, node_id in enumerate(seeds)}
            candidates.sort(key=lambda node_id: rank.get(node_id, len(rank)))

        elif plan.shape == "filter_first":
            stage = time.perf_counter_ns()
            seeds = self._pg_rank(query, plan.k)
            stages["filter_ms"] = _ms(stage)
            stages["ann_ms"] = stages["filter_ms"]
            round_trips += 1
            cardinality["seeds"] = len(seeds)
            ship(seeds)
            stage = time.perf_counter_ns()
            candidates = self._neo_reach(
                query, plan.hops, restrict_ids=seeds, apply_predicate=False
            )
            stages["traverse_ms"] = _ms(stage)
            round_trips += 1
            cardinality["reached"] = len(candidates)
            ship(candidates)
            rank = {node_id: idx for idx, node_id in enumerate(seeds)}
            candidates.sort(key=lambda node_id: rank.get(node_id, len(rank)))

        elif plan.shape == "traverse_first":
            stage = time.perf_counter_ns()
            reached = self._neo_reach(
                query,
                plan.hops,
                restrict_ids=None,
                apply_predicate=plan.predicate_placement == "during",
            )
            stages["traverse_ms"] = _ms(stage)
            round_trips += 1
            cardinality["reached"] = len(reached)
            ship(reached)
            stage = time.perf_counter_ns()
            candidates = self._pg_rank(query, plan.k, reached)
            stages["ann_ms"] = _ms(stage)
            round_trips += 1
            ship(candidates)
        else:  # pragma: no cover
            raise ValueError(f"unknown shape {plan.shape}")

        stage = time.perf_counter_ns()
        result_ids = candidates[:top_n]
        stages["merge_ms"] = _ms(stage)
        cardinality["candidates"] = len(candidates)
        latency_ms = _ms(started)
        return {
            "status": "ok",
            "backend": self.backend_name,
            "valid_for_system_latency_claims": True,
            "latency_ms": latency_ms,
            "stage_latency_ms": stages,
            "intermediate_cardinality": cardinality,
            "round_trips": round_trips,
            "bytes_shipped": bytes_shipped,
            "serialization_ms": serialization_ms,
            "serialization_fraction": (
                0.0 if latency_ms == 0 else serialization_ms / latency_ms
            ),
            "quality": quality_metrics(result_ids, query.answer_ids),
            "result_ids": result_ids,
        }
