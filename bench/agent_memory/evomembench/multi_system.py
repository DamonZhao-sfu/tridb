"""Live Milvus + Neo4j + PostgreSQL Experience Graph baseline.

This is intentionally an out-of-process materialize-transfer-prune path.  It
loads a content-addressed snapshot exported from GEM, executes the same
vector-seed/association-traversal/eligibility semantics, and records every
application-visible intermediate.  It is not the same-PostgreSQL staged
diagnostic in :mod:`run_systems`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import time
from typing import Any, Callable, Sequence

from bench.agent_memory.evomembench.injection import fit_injection
from bench.agent_memory.evomembench.run_systems import _merge_membership_candidates
from bench.agent_memory.evomembench.system_snapshot import ExperienceSnapshot


_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{2,62}$")


def _payload_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _vec_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def _milvus_cosine_distance(similarity: float) -> float:
    """Translate Milvus COSINE (larger is better) to pgvector <=> distance."""
    return 1.0 - float(similarity)


@dataclass(frozen=True)
class MultiSystemConfig:
    namespace: str
    milvus_host: str = os.environ.get("MILVUS_HOST", "127.0.0.1")
    milvus_port: str = os.environ.get("MILVUS_PORT", "19530")
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    neo4j_user: str = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "testpassword")
    pg_dsn: str = os.environ.get(
        "EVOMEMBENCH_BASELINE_DSN",
        "postgresql://postgres:postgres@127.0.0.1:5432/tridb_baseline",
    )
    milvus_index: str = "HNSW"
    milvus_hnsw_m: int = 16
    milvus_ef_construction: int = 200
    milvus_ef: int = 128
    ann_overfetch: int = 8

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.namespace):
            raise ValueError(
                "namespace must be 3..63 lowercase alphanumeric/underscore characters"
            )
        if self.milvus_index != "HNSW":
            raise ValueError("formal baseline currently locks Milvus to HNSW")
        if min(self.milvus_hnsw_m, self.milvus_ef_construction, self.milvus_ef) < 1:
            raise ValueError("Milvus HNSW parameters must be positive")
        if self.ann_overfetch < 1:
            raise ValueError("ann_overfetch must be positive")

    @property
    def collection(self) -> str:
        return f"evomem_{self.namespace}"

    @property
    def milvus_alias(self) -> str:
        return f"evomem_{self.namespace}"


@dataclass(frozen=True)
class MultiSystemQuery:
    query_id: str
    scope_id: str
    embedding: tuple[float, ...]
    cutoff_ordinal: int
    k: int
    m_seeds: int
    hops: int
    graph_work_budget: int
    token_budget: int
    validity_states: tuple[str, ...] = ("active",)
    source_phases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query_id or not self.scope_id or not self.embedding:
            raise ValueError("multi-system query is missing identity or embedding")
        if (
            min(
                self.cutoff_ordinal,
                self.k,
                self.m_seeds,
                self.hops,
                self.graph_work_budget,
                self.token_budget,
            )
            < 0
        ):
            raise ValueError("multi-system query values must be non-negative")
        if self.k < 1 or self.m_seeds < 1 or self.graph_work_budget < 1:
            raise ValueError("k, seed count, and graph-work budget must be positive")
        if self.hops not in {1, 2, 3}:
            raise ValueError("formal baseline supports one to three graph hops")


@dataclass(frozen=True)
class MultiSystemResult:
    selected_ids: tuple[str, ...]
    selected_ordinals: tuple[int, ...]
    injection: str
    injection_tokens: int
    latency_ms: dict[str, float]
    intermediate: dict[str, int | bool | None]
    probes: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _BoundedGraphResult:
    candidates: tuple[tuple[str, int], ...]
    edge_rows_transferred: int
    round_trips: int
    request_bytes: int
    response_bytes: int
    peak_frontier_ids: int
    peak_edge_rows: int
    work_budget_reached: bool


def _bounded_graph_bfs(
    session: Any,
    *,
    namespace: str,
    seeds: Sequence[str],
    hops: int,
    edge_budget: int,
) -> _BoundedGraphResult:
    """Pull a level-at-a-time BFS with one shared transferred-edge budget.

    Neo4j does not expose a cheap per-query equivalent of TriDB's native
    ``gph_visits()`` counter.  Consequently the comparable, enforceable unit
    for this out-of-process baseline is an EVO_ASSOCIATION edge row crossing
    Neo4j's client boundary.  ``LIMIT`` sits on the streaming one-hop query;
    there is no variable-length path enumeration or post-hoc result limit.
    """
    frontier = list(dict.fromkeys(str(uid) for uid in seeds))
    visited = set(frontier)
    # TriDB's seedless reach hash contains the selected seeds themselves.  They
    # therefore participate in the bridge-cap finalize (and are de-duplicated
    # against vector winners); the physical baseline must preserve that detail.
    candidates: list[tuple[str, int]] = [(uid, 0) for uid in frontier]
    remaining = int(edge_budget)
    round_trips = 0
    request_bytes = 0
    response_bytes = 0
    edge_rows_transferred = 0
    peak_frontier_ids = len(frontier)
    peak_edge_rows = 0
    cypher = (
        "UNWIND $frontier AS uid "
        "MATCH (src:EvoMemUnit {namespace:$namespace, uid:uid})"
        "-[:EVO_ASSOCIATION]->"
        "(dst:EvoMemUnit {namespace:$namespace}) "
        "RETURN src.uid AS src_uid, dst.uid AS dst_uid LIMIT $remaining"
    )
    for hop in range(1, hops + 1):
        if not frontier or remaining <= 0:
            break
        params = {
            "frontier": frontier,
            "namespace": namespace,
            "remaining": remaining,
        }
        request_bytes += _payload_bytes(params)
        rows = [
            (str(row["src_uid"]), str(row["dst_uid"]))
            for row in session.run(cypher, **params)
        ]
        round_trips += 1
        response_bytes += _payload_bytes(rows)
        peak_edge_rows = max(peak_edge_rows, len(rows))
        edge_rows_transferred += len(rows)
        remaining -= len(rows)
        next_frontier: list[str] = []
        for _src_uid, dst_uid in rows:
            if dst_uid in visited:
                continue
            visited.add(dst_uid)
            next_frontier.append(dst_uid)
            candidates.append((dst_uid, hop))
        frontier = next_frontier
        peak_frontier_ids = max(peak_frontier_ids, len(frontier))
    return _BoundedGraphResult(
        candidates=tuple(candidates),
        edge_rows_transferred=edge_rows_transferred,
        round_trips=round_trips,
        request_bytes=request_bytes,
        response_bytes=response_bytes,
        peak_frontier_ids=peak_frontier_ids,
        peak_edge_rows=peak_edge_rows,
        # This is deliberately named "reached", not "censored": without an
        # extra edge probe Neo4j cannot distinguish exact exhaustion at N from
        # a truncated N+1 edge.  The formal parity gate must fail closed if it
        # needs an exact uncensored claim at this boundary.
        work_budget_reached=remaining == 0,
    )


class LiveMultiSystemExperienceStore:
    """One live connection set. Connections and index load are outside timing."""

    def __init__(self, config: MultiSystemConfig) -> None:
        self.config = config
        self._milvus_collection: Any = None
        self._neo4j: Any = None
        self._pg: Any = None
        self._unit_count: int | None = None

    def connect(self) -> None:
        if any(
            value is not None
            for value in (self._milvus_collection, self._neo4j, self._pg)
        ):
            raise RuntimeError("multi-system store is already connected")
        from neo4j import GraphDatabase
        from psycopg import connect
        from pymilvus import connections

        connections.connect(
            alias=self.config.milvus_alias,
            host=self.config.milvus_host,
            port=self.config.milvus_port,
        )
        self._neo4j = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )
        self._pg = connect(self.config.pg_dsn)
        self._neo4j.verify_connectivity()
        with self._pg.cursor() as cursor:
            cursor.execute("SELECT 1")

    def close(self) -> None:
        try:
            if self._neo4j is not None:
                self._neo4j.close()
        finally:
            try:
                if self._pg is not None:
                    self._pg.close()
            finally:
                if self._milvus_collection is not None:
                    self._milvus_collection = None
                try:
                    from pymilvus import connections

                    connections.disconnect(self.config.milvus_alias)
                except Exception:  # pragma: no cover - best-effort close
                    pass

    def __enter__(self) -> "LiveMultiSystemExperienceStore":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_connected(self) -> None:
        if self._neo4j is None or self._pg is None:
            raise RuntimeError("multi-system store is not connected")

    def preflight(self) -> dict[str, Any]:
        self._require_connected()
        from pymilvus import utility

        with self._neo4j.session() as session:
            neo = int(session.run("RETURN 1 AS ok").single()["ok"])
        with self._pg.cursor() as cursor:
            cursor.execute("SELECT current_setting('server_version')")
            pg_version = str(cursor.fetchone()[0])
        return {
            "milvus_connected": True,
            "milvus_collection_exists": utility.has_collection(
                self.config.collection, using=self.config.milvus_alias
            ),
            "neo4j_connected": neo == 1,
            "postgres_version": pg_version,
        }

    def load_snapshot(self, snapshot: ExperienceSnapshot) -> dict[str, Any]:
        """Load one new namespace. Existing state is a hard failure, never cleared."""
        self._require_connected()
        snapshot.verify()
        declared_dimension = getattr(snapshot, "embedding_dim", None)
        if declared_dimension is None:
            dimensions = {len(unit.embedding) for unit in snapshot.units}
            if len(dimensions) != 1:
                raise ValueError("snapshot embeddings have inconsistent dimensions")
            dimension = dimensions.pop()
        else:
            dimension = int(declared_dimension)
            if dimension < 1:
                raise ValueError("snapshot embedding dimension must be positive")
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        if utility.has_collection(
            self.config.collection, using=self.config.milvus_alias
        ):
            raise FileExistsError(
                f"refusing existing Milvus collection {self.config.collection}"
            )
        with self._pg.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS evomem_system_unit ("
                " namespace text NOT NULL, uid text NOT NULL, scope_id text NOT NULL,"
                " node_kind text NOT NULL, ordinal integer, state text NOT NULL,"
                " validity_state text NOT NULL, source_phase text, payload text NOT NULL,"
                " embedding vector NOT NULL, metadata jsonb NOT NULL,"
                " PRIMARY KEY(namespace, uid))"
            )
            cursor.execute(
                "SELECT count(*) FROM evomem_system_unit WHERE namespace=%s",
                (self.config.namespace,),
            )
            if int(cursor.fetchone()[0]) != 0:
                raise FileExistsError(
                    f"refusing existing PostgreSQL namespace {self.config.namespace}"
                )
        with self._neo4j.session() as session:
            existing = int(
                session.run(
                    "MATCH (n:EvoMemUnit {namespace:$namespace}) RETURN count(n) AS n",
                    namespace=self.config.namespace,
                ).single()["n"]
            )
            if existing:
                raise FileExistsError(
                    f"refusing existing Neo4j namespace {self.config.namespace}"
                )

        began = time.perf_counter()
        schema = CollectionSchema(
            [
                FieldSchema("uid", DataType.VARCHAR, is_primary=True, max_length=512),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dimension),
            ]
        )
        collection = Collection(
            self.config.collection,
            schema,
            using=self.config.milvus_alias,
        )
        batch_size = 1000
        for start in range(0, len(snapshot.units), batch_size):
            batch = snapshot.units[start : start + batch_size]
            collection.insert(
                [
                    [unit.uid for unit in batch],
                    [list(unit.embedding) for unit in batch],
                ]
            )
        collection.flush()
        collection.create_index(
            "embedding",
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": self.config.milvus_hnsw_m,
                    "efConstruction": self.config.milvus_ef_construction,
                },
            },
        )
        collection.load()
        self._milvus_collection = collection
        self._unit_count = len(snapshot.units)

        with self._neo4j.session() as session:
            session.run(
                "CREATE INDEX evomem_namespace_uid IF NOT EXISTS"
                " FOR (n:EvoMemUnit) ON (n.namespace, n.uid)"
            )
            for start in range(0, len(snapshot.units), 5000):
                rows = [
                    {
                        "uid": unit.uid,
                        "scope_id": unit.scope_id,
                        "node_kind": unit.node_kind,
                    }
                    for unit in snapshot.units[start : start + 5000]
                ]
                session.run(
                    "UNWIND $rows AS row CREATE (:EvoMemUnit {"
                    "namespace:$namespace, uid:row.uid, scope_id:row.scope_id,"
                    "node_kind:row.node_kind})",
                    namespace=self.config.namespace,
                    rows=rows,
                ).consume()
            for start in range(0, len(snapshot.edges), 5000):
                rows = [asdict(edge) for edge in snapshot.edges[start : start + 5000]]
                session.run(
                    "UNWIND $rows AS row "
                    "MATCH (src:EvoMemUnit {namespace:$namespace, uid:row.src_uid}),"
                    " (dst:EvoMemUnit {namespace:$namespace, uid:row.dst_uid}) "
                    "CREATE (src)-[:EVO_ASSOCIATION {namespace:$namespace,"
                    " rel:row.rel, weight:row.weight}]->(dst)",
                    namespace=self.config.namespace,
                    rows=rows,
                ).consume()

        with self._pg.cursor() as cursor:
            for start in range(0, len(snapshot.units), batch_size):
                batch = snapshot.units[start : start + batch_size]
                cursor.executemany(
                    "INSERT INTO evomem_system_unit"
                    " (namespace,uid,scope_id,node_kind,ordinal,state,validity_state,"
                    " source_phase,payload,embedding,metadata)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s::jsonb)",
                    [
                        (
                            self.config.namespace,
                            unit.uid,
                            unit.scope_id,
                            unit.node_kind,
                            unit.ordinal,
                            unit.state,
                            str(unit.metadata.get("validity_state", "active")),
                            unit.metadata.get("source_phase"),
                            unit.payload,
                            _vec_literal(unit.embedding),
                            json.dumps(unit.metadata, ensure_ascii=False),
                        )
                        for unit in batch
                    ],
                )
        self._pg.commit()
        return {
            "snapshot_digest": snapshot.digest,
            "units": len(snapshot.units),
            "edges": len(snapshot.edges),
            "embedding_dim": dimension,
            "load_seconds": time.perf_counter() - began,
            "collection": self.config.collection,
            "namespace": self.config.namespace,
        }

    def cleanup_owned_namespace(self) -> dict[str, Any]:
        """Remove only this run-owned namespace after a successful measurement.

        Failed measurements deliberately never call this method, preserving their
        backend state for diagnosis. Cleanup latency is reported and excluded from
        load/read latency.
        """
        self._require_connected()
        from pymilvus import utility

        began = time.perf_counter()
        collection_dropped = False
        if utility.has_collection(
            self.config.collection, using=self.config.milvus_alias
        ):
            if self._milvus_collection is not None:
                self._milvus_collection.release()
            utility.drop_collection(
                self.config.collection, using=self.config.milvus_alias
            )
            collection_dropped = True
            self._milvus_collection = None
        with self._neo4j.session() as session:
            neo4j_nodes = int(
                session.run(
                    "MATCH (n:EvoMemUnit {namespace:$namespace}) RETURN count(n) AS n",
                    namespace=self.config.namespace,
                ).single()["n"]
            )
            session.run(
                "MATCH (n:EvoMemUnit {namespace:$namespace}) DETACH DELETE n",
                namespace=self.config.namespace,
            ).consume()
        with self._pg.cursor() as cursor:
            cursor.execute(
                "DELETE FROM evomem_system_unit WHERE namespace=%s",
                (self.config.namespace,),
            )
            postgresql_rows = int(cursor.rowcount)
        self._pg.commit()
        self._unit_count = 0
        return {
            "scope": "exact_run_owned_namespace",
            "collection_dropped": collection_dropped,
            "neo4j_nodes_deleted": neo4j_nodes,
            "postgresql_rows_deleted": postgresql_rows,
            "cleanup_seconds_excluded": time.perf_counter() - began,
        }

    def _eligible(
        self, uids: Sequence[str], query: MultiSystemQuery
    ) -> dict[str, tuple[int, str, float]]:
        if not uids:
            return {}
        clauses = [
            "namespace=%s",
            "uid=ANY(%s)",
            "scope_id=%s",
            "state='active'",
            "node_kind='experience'",
            "ordinal < %s",
            "validity_state=ANY(%s)",
        ]
        params: list[Any] = [
            self.config.namespace,
            list(uids),
            query.scope_id,
            query.cutoff_ordinal,
            list(query.validity_states),
        ]
        if query.source_phases:
            clauses.append("source_phase=ANY(%s)")
            params.append(list(query.source_phases))
        params.append(_vec_literal(query.embedding))
        with self._pg.cursor() as cursor:
            cursor.execute(
                "SELECT uid, ordinal, payload, embedding <=> %s::vector AS distance"
                " FROM evomem_system_unit WHERE " + " AND ".join(clauses),
                [params[-1], *params[:-1]],
            )
            return {
                str(row[0]): (int(row[1]), str(row[2]), float(row[3]))
                for row in cursor.fetchall()
            }

    def query(
        self,
        query: MultiSystemQuery,
        *,
        count_tokens: Callable[[str], int],
    ) -> MultiSystemResult:
        self._require_connected()
        if self._milvus_collection is None:
            from pymilvus import Collection

            self._milvus_collection = Collection(
                self.config.collection, using=self.config.milvus_alias
            )
            self._milvus_collection.load()
        total_started = time.perf_counter()
        seed_window = max(query.k, query.m_seeds * 8, query.m_seeds + 32)
        vector_limit = min(
            seed_window * self.config.ann_overfetch,
            self._unit_count or int(self._milvus_collection.num_entities),
        )

        began = time.perf_counter()
        response = self._milvus_collection.search(
            data=[list(query.embedding)],
            anns_field="embedding",
            param={
                "metric_type": "COSINE",
                "params": {"ef": self.config.milvus_ef},
            },
            limit=vector_limit,
            output_fields=["uid"],
        )
        vector_rows = [
            (str(hit.id), _milvus_cosine_distance(hit.distance)) for hit in response[0]
        ]
        vector_ms = (time.perf_counter() - began) * 1000

        began = time.perf_counter()
        eligible_vector = self._eligible([uid for uid, _ in vector_rows], query)
        seed_filter_ms = (time.perf_counter() - began) * 1000
        ranked_vector = [
            (uid, distance) for uid, distance in vector_rows if uid in eligible_vector
        ]
        seeds = [uid for uid, _ in ranked_vector[: query.m_seeds]]

        began = time.perf_counter()
        with self._neo4j.session() as session:
            graph = _bounded_graph_bfs(
                session,
                namespace=self.config.namespace,
                seeds=seeds,
                hops=query.hops,
                edge_budget=query.graph_work_budget,
            )
        graph_rows = list(graph.candidates)
        graph_ms = (time.perf_counter() - began) * 1000

        graph_uids = list(dict.fromkeys(uid for uid, _hop in graph_rows))
        began = time.perf_counter()
        eligible_graph = self._eligible(graph_uids, query)
        graph_filter_ms = (time.perf_counter() - began) * 1000
        graph_ranked = sorted(
            ((uid, row[2]) for uid, row in eligible_graph.items()),
            key=lambda item: (item[1], item[0]),
        )

        began = time.perf_counter()
        selected = _merge_membership_candidates(
            ranked_vector,
            graph_ranked,
            k=query.k,
        )
        selected_rows = {
            **eligible_vector,
            **eligible_graph,
        }
        items = [
            {
                "unit_id": uid,
                "episode_uid": uid,
                "ordinal": selected_rows[uid][0],
                "text": selected_rows[uid][1],
            }
            for uid in selected
        ]
        accepted, injection, injection_tokens = fit_injection(
            items,
            max_items=query.k,
            token_budget=query.token_budget,
            count_tokens=count_tokens,
        )
        merge_ms = (time.perf_counter() - began) * 1000
        selected_ids = tuple(str(item["episode_uid"]) for item in accepted)
        selected_ordinals = tuple(int(item["ordinal"]) for item in accepted)

        vector_payload_bytes = _payload_bytes(vector_rows)
        seed_payload_bytes = _payload_bytes(seeds)
        graph_payload_bytes = graph.response_bytes
        relational_request_bytes = _payload_bytes(
            [uid for uid, _ in vector_rows]
        ) + _payload_bytes(graph_uids)
        relational_response_bytes = _payload_bytes(
            {uid: [row[0], row[1]] for uid, row in selected_rows.items()}
        )
        total_ms = (time.perf_counter() - total_started) * 1000
        return MultiSystemResult(
            selected_ids=selected_ids,
            selected_ordinals=selected_ordinals,
            injection=injection,
            injection_tokens=injection_tokens,
            latency_ms={
                "total": total_ms,
                "vector": vector_ms,
                "seed_relational_filter": seed_filter_ms,
                "graph": graph_ms,
                "graph_relational_filter": graph_filter_ms,
                "merge_and_prompt": merge_ms,
            },
            intermediate={
                "vector_candidates_materialized": len(vector_rows),
                "eligible_vector_candidates": len(ranked_vector),
                "vector_seeds_transferred": len(seeds),
                "graph_candidates_materialized": len(graph_rows),
                "graph_edge_rows_transferred": graph.edge_rows_transferred,
                "graph_round_trips": graph.round_trips,
                "graph_frontier_request_bytes": graph.request_bytes,
                "graph_peak_frontier_ids": graph.peak_frontier_ids,
                "graph_peak_edge_rows": graph.peak_edge_rows,
                "eligible_graph_candidates": len(eligible_graph),
                "app_candidates_materialized": len(selected_rows),
                "final_results": len(selected_ids),
                "peak_materialized_ids": max(
                    len(vector_rows), len(graph_rows), len(selected_rows)
                ),
                "vector_response_bytes": vector_payload_bytes,
                "seed_transfer_bytes": seed_payload_bytes,
                "graph_response_bytes": graph_payload_bytes,
                "relational_request_bytes": relational_request_bytes,
                "relational_response_bytes": relational_response_bytes,
                "application_visible_transfer_bytes": (
                    vector_payload_bytes
                    + seed_payload_bytes
                    + graph_payload_bytes
                    + relational_request_bytes
                    + relational_response_bytes
                ),
                "graph_work_budget_reached": graph.work_budget_reached,
                "graph_work_censored": None,
                "graph_work_censoring_exact": False,
            },
            probes={
                "mode": "live_milvus_neo4j_postgresql_app_merge",
                "namespace": self.config.namespace,
                "collection": self.config.collection,
                "ann_overfetch": self.config.ann_overfetch,
                "milvus_index": self.config.milvus_index,
                "milvus_score_transform": "cosine_distance=1-cosine_similarity",
                "hops": query.hops,
                "graph_work_budget": query.graph_work_budget,
                "graph_work_unit": "neo4j_edge_rows_transferred",
                "materializes_intermediates": True,
                "single_transaction": False,
            },
        )
