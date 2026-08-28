"""Stream STARK-MAG into the E1 Polyglot-Tuned stores.

MAG has 1.87M graph nodes but only 700,244 official retrieval candidates
(papers).  Milvus and pgvector therefore receive exactly those candidate
vectors; Neo4j receives the complete 39.8M-arc graph.  The loader never builds
an in-memory node->vector dictionary and refuses to start when the combined E1
capacity gate fails unless explicitly forced.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Iterator

from tools.e0.common import environment_record, write_json
from tools.e0.stark_mag_capacity import GIB, estimate as estimate_capacity

NORMALIZED = Path("data/e0/stark_mag/normalized")
DIMENSION = 1536
COLLECTION = "e0_stark_mag"
NEO4J_LABEL = "MAGNode"
PG_TABLE = "e0_stark_mag_node"

DEFAULTS: dict[str, Any] = {
    "milvus_host": "127.0.0.1",
    "milvus_port": "19530",
    "neo4j_uri": "bolt://127.0.0.1:7688",
    "neo4j_user": "neo4j",
    "neo4j_password": "testpassword",
    "pg_host": "127.0.0.1",
    "pg_port": 5434,
    "pg_db": "tridb_wiki",
    "pg_user": "postgres",
    "pg_password": "postgres",
}


def relationship_type(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    if not result or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result):
        raise ValueError(f"unsafe Neo4j relationship type derived from {value!r}")
    return result


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in values) + "]"


def _embedding_rows() -> Iterator[tuple[int, list[float]]]:
    import pyarrow.parquet as pq

    source = pq.ParquetFile(NORMALIZED / "embeddings.parquet")
    previous: int | None = None
    for batch in source.iter_batches(batch_size=512, columns=["node_id", "embedding"]):
        for node_id, vector in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist()
        ):
            node_id = int(node_id)
            if previous is not None and node_id <= previous:
                raise ValueError(
                    "MAG candidate embeddings are not strictly node-id sorted"
                )
            previous = node_id
            yield node_id, vector


def load_milvus(cfg: dict[str, Any], *, drop: bool) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    alias = "e0_mag_load"
    connections.connect(alias=alias, host=cfg["milvus_host"], port=cfg["milvus_port"])
    exists = utility.has_collection(COLLECTION, using=alias)
    if exists and not drop:
        raise RuntimeError(f"Milvus collection {COLLECTION!r} already exists")
    if exists:
        utility.drop_collection(COLLECTION, using=alias)
    collection = Collection(
        COLLECTION,
        CollectionSchema(
            [
                FieldSchema("node_id", DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIMENSION),
            ]
        ),
        using=alias,
    )
    source = pq.ParquetFile(NORMALIZED / "embeddings.parquet")
    started = time.time()
    inserted = 0
    for batch in source.iter_batches(batch_size=1024, columns=["node_id", "embedding"]):
        ids = [int(value) for value in batch.column(0).to_pylist()]
        vectors = np.asarray(batch.column(1).to_pylist(), dtype=np.float32)
        if vectors.shape != (len(ids), DIMENSION):
            raise ValueError(f"unexpected MAG vector batch shape {vectors.shape}")
        collection.insert([ids, vectors.tolist()])
        inserted += len(ids)
    collection.flush()
    if inserted != source.metadata.num_rows:
        raise RuntimeError(
            f"Milvus inserted {inserted}, expected {source.metadata.num_rows}"
        )
    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    collection.load()
    observed = int(collection.num_entities)
    return {
        "rows": observed,
        "expected": inserted,
        "seconds": round(time.time() - started, 3),
        "ready": observed == inserted,
    }


def load_neo4j(
    cfg: dict[str, Any], *, drop: bool, batch_size: int = 10_000
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"])
    )
    nodes = pq.ParquetFile(NORMALIZED / "nodes.parquet")
    edges = pq.ParquetFile(NORMALIZED / "edges.parquet")
    started = time.time()
    with driver.session() as session:
        existing = int(
            session.run(f"MATCH (n:{NEO4J_LABEL}) RETURN count(n) AS c").single()["c"]
        )
        if existing and not drop:
            raise RuntimeError(f"Neo4j already has {existing} {NEO4J_LABEL} nodes")
        if existing:
            while True:
                deleted = session.run(
                    f"MATCH (n:{NEO4J_LABEL}) WITH n LIMIT 20000 "
                    "DETACH DELETE n RETURN count(*) AS c"
                ).single()["c"]
                if not deleted:
                    break
        session.run(
            "CREATE CONSTRAINT e0_stark_mag_node_id IF NOT EXISTS "
            f"FOR (n:{NEO4J_LABEL}) REQUIRE n.node_id IS UNIQUE"
        ).consume()
        session.run(
            "CREATE INDEX e0_stark_mag_node_type IF NOT EXISTS "
            f"FOR (n:{NEO4J_LABEL}) ON (n.entity_type)"
        ).consume()

        inserted_nodes = 0
        for batch in nodes.iter_batches(
            batch_size=batch_size, columns=["node_id", "entity_type"]
        ):
            rows = [
                {"node_id": int(node_id), "entity_type": entity_type}
                for node_id, entity_type in zip(
                    batch.column(0).to_pylist(),
                    batch.column(1).to_pylist(),
                )
            ]
            session.run(
                f"UNWIND $rows AS row CREATE (n:{NEO4J_LABEL} {{"
                "node_id: row.node_id, entity_type: row.entity_type})",
                rows=rows,
            ).consume()
            inserted_nodes += len(rows)
        node_seconds = time.time() - started

        inserted_edges = 0
        edge_types: set[str] = set()
        for batch in edges.iter_batches(
            batch_size=batch_size, columns=["src_id", "dst_id", "edge_type"]
        ):
            grouped: dict[str, list[dict[str, int]]] = {}
            for src, dst, edge_type in zip(
                batch.column(0).to_pylist(),
                batch.column(1).to_pylist(),
                batch.column(2).to_pylist(),
            ):
                edge_type = str(edge_type)
                grouped.setdefault(edge_type, []).append(
                    {"src": int(src), "dst": int(dst)}
                )
            for edge_type, rows in grouped.items():
                rel = relationship_type(edge_type)
                session.run(
                    f"UNWIND $rows AS row "
                    f"MATCH (a:{NEO4J_LABEL} {{node_id: row.src}}), "
                    f"(b:{NEO4J_LABEL} {{node_id: row.dst}}) "
                    f"CREATE (a)-[:{rel}]->(b)",
                    rows=rows,
                ).consume()
                inserted_edges += len(rows)
                edge_types.add(edge_type)

        observed_nodes = int(
            session.run(f"MATCH (n:{NEO4J_LABEL}) RETURN count(n) AS c").single()["c"]
        )
        observed_edges = int(
            session.run(
                f"MATCH (:{NEO4J_LABEL})-[r]->(:{NEO4J_LABEL}) RETURN count(r) AS c"
            ).single()["c"]
        )
    driver.close()
    ready = (
        inserted_nodes == observed_nodes == nodes.metadata.num_rows
        and inserted_edges == observed_edges == edges.metadata.num_rows
    )
    return {
        "nodes": observed_nodes,
        "relationships": observed_edges,
        "edge_types": sorted(edge_types),
        "node_seconds": round(node_seconds, 3),
        "seconds": round(time.time() - started, 3),
        "ready": ready,
    }


def load_postgres(cfg: dict[str, Any], *, drop: bool) -> dict[str, Any]:
    import psycopg
    import pyarrow.parquet as pq

    connection = psycopg.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        dbname=cfg["pg_db"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
        autocommit=True,
    )
    nodes = pq.ParquetFile(NORMALIZED / "nodes.parquet")
    expected_vectors = pq.ParquetFile(
        NORMALIZED / "embeddings.parquet"
    ).metadata.num_rows
    started = time.time()
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(f"SELECT to_regclass('{PG_TABLE}')")
        exists = cursor.fetchone()[0] is not None
        if exists and not drop:
            raise RuntimeError(f"Postgres table {PG_TABLE!r} already exists")
        if exists:
            cursor.execute(f"DROP TABLE {PG_TABLE}")
        cursor.execute(
            f"CREATE TABLE {PG_TABLE} ("
            "node_id bigint PRIMARY KEY, entity_type text NOT NULL, "
            "generation integer, parent_id bigint, "
            f"embedding vector({DIMENSION}))"
        )

        embeddings = _embedding_rows()
        current = next(embeddings, None)
        copied_nodes = 0
        copied_vectors = 0
        with cursor.copy(
            f"COPY {PG_TABLE} (node_id,entity_type,embedding) FROM STDIN"
        ) as copy:
            for batch in nodes.iter_batches(
                batch_size=2048,
                columns=["node_id", "entity_type"],
            ):
                for node_id, entity_type in zip(
                    batch.column(0).to_pylist(),
                    batch.column(1).to_pylist(),
                ):
                    node_id = int(node_id)
                    vector = None
                    if current is not None:
                        vector_id, vector_value = current
                        if vector_id == node_id:
                            vector = _vector_literal(vector_value)
                            copied_vectors += 1
                            current = next(embeddings, None)
                        elif vector_id < node_id:
                            raise ValueError(
                                f"candidate vector {vector_id} is not aligned with nodes"
                            )
                    copy.write_row((node_id, entity_type, vector))
                    copied_nodes += 1
        if current is not None:
            raise ValueError(f"unknown trailing candidate vector {current[0]}")
        if copied_vectors != expected_vectors:
            raise RuntimeError(
                f"Postgres copied {copied_vectors} vectors, expected {expected_vectors}"
            )
        cursor.execute(f"CREATE INDEX {PG_TABLE}_type ON {PG_TABLE}(entity_type)")
        cursor.execute(
            f"CREATE INDEX {PG_TABLE}_hnsw ON {PG_TABLE} USING hnsw "
            "(embedding vector_cosine_ops) WITH (m=16, ef_construction=200)"
        )
        cursor.execute(f"ANALYZE {PG_TABLE}")
        cursor.execute(f"SELECT count(*),count(embedding) FROM {PG_TABLE}")
        observed_nodes, observed_vectors = map(int, cursor.fetchone())
    connection.close()
    return {
        "rows": observed_nodes,
        "vectors": observed_vectors,
        "expected_rows": nodes.metadata.num_rows,
        "expected_vectors": expected_vectors,
        "seconds": round(time.time() - started, 3),
        "ready": (
            observed_nodes == copied_nodes == nodes.metadata.num_rows
            and observed_vectors == copied_vectors == expected_vectors
        ),
    }


def verify(cfg: dict[str, Any]) -> dict[str, Any]:
    import psycopg
    import pyarrow.parquet as pq
    from neo4j import GraphDatabase
    from pymilvus import Collection, connections

    expected_nodes = pq.ParquetFile(NORMALIZED / "nodes.parquet").metadata.num_rows
    expected_edges = pq.ParquetFile(NORMALIZED / "edges.parquet").metadata.num_rows
    expected_vectors = pq.ParquetFile(
        NORMALIZED / "embeddings.parquet"
    ).metadata.num_rows

    connections.connect(
        alias="e0_mag_verify", host=cfg["milvus_host"], port=cfg["milvus_port"]
    )
    milvus_vectors = int(Collection(COLLECTION, using="e0_mag_verify").num_entities)

    with psycopg.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        dbname=cfg["pg_db"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT count(*),count(embedding) FROM {PG_TABLE}")
            pg_nodes, pg_vectors = map(int, cursor.fetchone())

    driver = GraphDatabase.driver(
        cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"])
    )
    queries = [
        json.loads(line)
        for line in (NORMALIZED / "queries_v0.2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    unreachable: list[dict[str, Any]] = []
    with driver.session() as session:
        neo_nodes = int(
            session.run(f"MATCH (n:{NEO4J_LABEL}) RETURN count(n) AS c").single()["c"]
        )
        neo_edges = int(
            session.run(
                f"MATCH (:{NEO4J_LABEL})-[r]->(:{NEO4J_LABEL}) RETURN count(r) AS c"
            ).single()["c"]
        )
        for query in queries:
            relationships = "|".join(
                relationship_type(value) for value in query["edge_types"]
            )
            anchors = [int(value) for value in query["anchor_ids"]]
            path_audit = query.get("path_audit") or query.get("audit") or {}
            required = (
                len(anchors) if path_audit.get("required_from_each_anchor") else 1
            )
            rows = session.run(
                f"MATCH (a:{NEO4J_LABEL}) WHERE a.node_id IN $anchors "
                f"MATCH (a)-[:{relationships}*1..{int(query['hop_limit'])}]->"
                f"(b:{NEO4J_LABEL}) WHERE NOT b.node_id IN $anchors "
                "WITH b,count(DISTINCT a) AS hits WHERE hits >= $required "
                "RETURN DISTINCT b.node_id AS node_id",
                anchors=anchors,
                required=required,
            )
            reach = {int(row["node_id"]) for row in rows}
            missing = sorted(set(query["answer_ids"]) - reach)
            if missing:
                unreachable.append({"query_id": query["query_id"], "missing": missing})
    driver.close()
    checks = {
        "milvus_candidate_vectors_match": milvus_vectors == expected_vectors,
        "neo4j_nodes_match": neo_nodes == expected_nodes,
        "neo4j_relationships_match": neo_edges == expected_edges,
        "postgres_nodes_match": pg_nodes == expected_nodes,
        "postgres_candidate_vectors_match": pg_vectors == expected_vectors,
        "all_official_answers_reachable": not unreachable,
    }
    return {
        "expected": {
            "nodes": expected_nodes,
            "directed_edges": expected_edges,
            "candidate_vectors": expected_vectors,
        },
        "observed": {
            "milvus_candidate_vectors": milvus_vectors,
            "neo4j_nodes": neo_nodes,
            "neo4j_relationships": neo_edges,
            "postgres_nodes": pg_nodes,
            "postgres_candidate_vectors": pg_vectors,
        },
        "unreachable_queries": unreachable,
        "checks": checks,
        "ready": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["capacity", "all", "milvus", "neo4j", "postgres", "verify"]
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--force-capacity",
        action="store_true",
        help="run despite the simultaneous-deployment capacity gate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/e0/stark_mag/polyglot_load.json"),
    )
    args = parser.parse_args(argv)
    capacity = estimate_capacity()
    report: dict[str, Any] = {
        "schema_version": "e0-stark-mag-polyglot-load-v0.1.0",
        "environment": environment_record(),
        "capacity": capacity,
        "phases": {},
        "platform_claim": "x86_64 co-located Polyglot-Tuned; no GX10 sign-off",
    }
    if args.phase == "capacity":
        report["ready"] = capacity["ready"]
        write_json(args.out, report)
        print(
            f"ready={capacity['ready']} free={capacity['observed_free_bytes'] / GIB:.1f} GiB "
            f"required={capacity['required_free_bytes'] / GIB:.1f} GiB"
        )
        return 0 if capacity["ready"] else 2
    polyglot_capacity = capacity["modes"]["polyglot_isolated"]
    if not polyglot_capacity["ready"] and not args.force_capacity:
        report["ready"] = False
        report["failure"] = (
            "polyglot-isolated capacity gate failed; do not start the load"
        )
        write_json(args.out, report)
        print(report["failure"])
        return 2

    cfg = dict(DEFAULTS)
    drop = not args.keep
    phase_functions = {
        "milvus": load_milvus,
        "neo4j": load_neo4j,
        "postgres": load_postgres,
    }
    selected = list(phase_functions) if args.phase == "all" else [args.phase]
    for phase in selected:
        if phase == "verify":
            report["phases"][phase] = verify(cfg)
        else:
            report["phases"][phase] = phase_functions[phase](cfg, drop=drop)
        write_json(args.out, report)
        print(f"[stark-mag-polyglot] {phase}: {report['phases'][phase]}", flush=True)
    if args.phase == "all":
        report["phases"]["verify"] = verify(cfg)
    report["ready"] = all(
        bool(value.get("ready")) for value in report["phases"].values()
    )
    report["failure"] = None if report["ready"] else "one or more phases failed"
    write_json(args.out, report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
