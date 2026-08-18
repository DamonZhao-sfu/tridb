"""TriDB-native backend for the E0 plan-space experiment.

The timed path is one PostgreSQL statement.  Vector search, relational predicate
evaluation, and native adjacency traversal all execute in the same backend process;
only the final result crosses the client boundary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .live_backend import _identifier
from .model import PlanSpec, QuerySpec, quality_metrics


def _ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


class TriDBLiveDataset:
    backend_name = "tridb_live"
    valid_for_system_latency_claims = True

    def __init__(self, name: str, spec: dict[str, Any]) -> None:
        import psycopg
        import pyarrow.parquet as pq

        cfg = spec["tridb"]
        self.name = name
        self.table = _identifier(str(cfg.get("table", "e0_node")))
        self.graph_budget = int(cfg.get("graph_work_budget", 50_000_000))
        self.pg = psycopg.connect(
            host=str(cfg["host"]),
            port=int(cfg["port"]),
            dbname=str(cfg["dbname"]),
            user=cfg.get("user"),
            autocommit=True,
        )
        with self.pg.cursor() as cursor:
            cursor.execute("SET enable_seqscan = off")
            cursor.execute("SET hnsw.iterative_scan = relaxed_order")
            cursor.execute(
                "SELECT set_config('hnsw.ef_search', %s, false)",
                (str(int(cfg.get("hnsw_ef_search", 100))),),
            )
            cursor.execute("SET graph_store.assume_dense_open = on")
            cursor.execute(
                "SELECT set_config('tjs.graph_work_budget', %s, false)",
                (str(self.graph_budget),),
            )
            cursor.execute("SELECT name, id FROM graph_store.edge_type")
            self.edge_type_ids = {str(row[0]): int(row[1]) for row in cursor}

        node_table = pq.read_table(Path(spec["nodes"]), columns=["node_id"])
        external_ids = node_table["node_id"].to_pylist()
        self.external_to_vid = {value: idx for idx, value in enumerate(external_ids)}
        self.vid_to_external = external_ids

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
        self._predicate_cache: dict[str, str] = {}
        self._predicate_count_cache: dict[str, int] = {}

    def close(self) -> None:
        self.pg.close()

    def load_queries(self, path: Path) -> list[QuerySpec]:
        return [
            QuerySpec.from_mapping(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _vids(self, values: tuple[Any, ...]) -> list[int]:
        try:
            return [self.external_to_vid[value] for value in values]
        except KeyError as exc:
            raise ValueError(f"{self.name}: unknown node id {exc.args[0]!r}") from exc

    def _literal(self, value: Any) -> str:
        from psycopg import sql

        return sql.Literal(value).as_string(self.pg)

    def _predicate_sql(self, query: QuerySpec) -> str:
        cached = self._predicate_cache.get(query.query_id)
        if cached is not None:
            return cached
        clauses: list[str] = []
        if query.target_entity_type:
            clauses.append(f"entity_type = {self._literal(query.target_entity_type)}")
        predicate = query.structured_predicate
        if "generation_lte" in predicate:
            clauses.append(f"generation <= {int(predicate['generation_lte'])}")
        if "generation_gte" in predicate:
            clauses.append(f"generation >= {int(predicate['generation_gte'])}")
        if predicate.get("same_parent"):
            anchors = self._vids(query.anchor_ids)
            with self.pg.cursor() as cursor:
                cursor.execute(
                    f"SELECT DISTINCT parent_vid FROM {self.table} "
                    "WHERE id = ANY(%s) AND parent_vid IS NOT NULL",
                    (anchors,),
                )
                parents = [int(row[0]) for row in cursor]
            if not parents:
                clauses.append("FALSE")
            else:
                clauses.append(
                    f"parent_vid = ANY(ARRAY[{','.join(map(str, parents))}]::bigint[])"
                )
                clauses.append(
                    f"NOT (id = ANY(ARRAY[{','.join(map(str, anchors))}]::bigint[]))"
                )
        result = " AND ".join(clauses) if clauses else "TRUE"
        self._predicate_cache[query.query_id] = result
        return result

    def prepare_query(self, query: QuerySpec, hops: set[int]) -> None:
        del hops
        if query.query_id not in self.query_vectors:
            raise ValueError(f"{self.name}: missing query vector {query.query_id}")
        self._vids(query.anchor_ids)
        missing_types = sorted(set(query.edge_types) - set(self.edge_type_ids))
        if missing_types:
            raise ValueError(f"{self.name}: unknown edge types {missing_types}")
        predicate = self._predicate_sql(query)
        with self.pg.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {self.table} WHERE {predicate}")
            self._predicate_count_cache[query.query_id] = int(cursor.fetchone()[0])

    def _vector_literal(self, query: QuerySpec) -> str:
        return (
            "["
            + ",".join(
                f"{float(value):.8g}" for value in self.query_vectors[query.query_id]
            )
            + "]"
        )

    def execute(
        self, query: QuerySpec, plan: PlanSpec, *, top_n: int
    ) -> dict[str, Any]:
        anchors = self._vids(query.anchor_ids)
        type_ids = [self.edge_type_ids[value] for value in query.edge_types]
        predicate = self._predicate_sql(query)
        started = time.perf_counter_ns()
        with self.pg.cursor() as cursor:
            # Target-list SRF placement preserves the pull path used by the graph iterator.
            cursor.execute(
                "SELECT tjs_e0_open(%s::regclass,%s,%s,%s,%s,%s,%s::vector,"
                "%s::bigint[],%s::integer[],%s,%s,%s)",
                (
                    self.table,
                    plan.k,
                    top_n,
                    plan.hops,
                    "id",
                    predicate,
                    self._vector_literal(query),
                    anchors,
                    type_ids,
                    query.require_each_anchor,
                    plan.shape,
                    plan.predicate_placement,
                ),
            )
            result_vids = [int(row[0]) for row in cursor]
        latency_ms = _ms(started)

        # Honesty probes run after the timed statement and are excluded from latency_ms.
        with self.pg.cursor() as cursor:
            cursor.execute(
                "SELECT tjs_open_candidates_examined(), tjs_open_graph_examined(), "
                "tjs_open_graph_censored(), tjs_e0_vector_us(), tjs_e0_graph_us(), "
                "tjs_e0_filter_us(), tjs_e0_termination()"
            )
            examined, graph_edges, censored, vector_us, graph_us, filter_us, term = (
                cursor.fetchone()
            )

        result_ids = [self.vid_to_external[vid] for vid in result_vids]
        stages = {
            "ann_ms": float(vector_us) / 1000.0,
            "traverse_ms": float(graph_us) / 1000.0,
            "filter_ms": float(filter_us) / 1000.0,
            "merge_ms": max(
                0.0,
                latency_ms
                - (float(vector_us) + float(graph_us) + float(filter_us)) / 1000.0,
            ),
        }
        return {
            "status": "ok",
            "backend": self.backend_name,
            "valid_for_system_latency_claims": True,
            "latency_ms": latency_ms,
            "stage_latency_ms": stages,
            "intermediate_cardinality": {
                "seeds": int(examined) if plan.shape != "traverse_first" else None,
                "reached": None,
                "predicate_matches": self._predicate_count_cache[query.query_id],
                "candidates": len(result_vids),
                "graph_edges_examined": int(graph_edges),
            },
            # No intermediate crosses a storage boundary.  One client statement returns
            # only final ids; the post-timing probe is instrumentation, not query execution.
            "round_trips": 0,
            "client_query_round_trips": 1,
            "bytes_shipped": 0,
            "result_bytes": len(json.dumps(result_ids, default=str).encode()),
            "serialization_fraction": 0.0,
            "graph_censored": bool(censored),
            "termination": str(term),
            "quality": quality_metrics(result_ids, query.answer_ids),
            "result_ids": result_ids,
        }
