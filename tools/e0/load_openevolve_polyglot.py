"""Load the small OpenEvolve E0 corpus into Milvus, Neo4j, and pgvector."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.e0.common import environment_record, write_json

ROOT = Path("data/e0/openevolve/normalized/seed_42")
COLLECTION = "e0_openevolve"
LABEL = "E0OpenEvolveNode"
TABLE = "e0_openevolve_node"
DIM = 1024


def _vectors() -> tuple[list[str], np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(ROOT / "embeddings.parquet", columns=["node_id", "embedding"])
    ids = [str(value) for value in table["node_id"].to_pylist()]
    vectors = np.asarray(table["embedding"].to_pylist(), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, vectors / norms


def load_milvus(*, drop: bool) -> dict[str, Any]:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    connections.connect(alias="e0oe", host="127.0.0.1", port="19530")
    if drop and utility.has_collection(COLLECTION, using="e0oe"):
        utility.drop_collection(COLLECTION, using="e0oe")
    collection = Collection(
        COLLECTION,
        CollectionSchema(
            [
                FieldSchema(
                    "node_id", DataType.VARCHAR, max_length=64, is_primary=True
                ),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
            ]
        ),
        using="e0oe",
    )
    ids, vectors = _vectors()
    started = time.time()
    collection.insert([ids, vectors.tolist()])
    collection.flush()
    collection.create_index(
        "embedding",
        {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    collection.load()
    return {"rows": int(collection.num_entities), "seconds": time.time() - started}


def load_neo4j(*, drop: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        "bolt://127.0.0.1:7688", auth=("neo4j", "testpassword")
    )
    nodes = pq.read_table(
        ROOT / "nodes.parquet", columns=["node_id", "entity_type", "generation"]
    ).to_pylist()
    edges = pq.read_table(
        ROOT / "edges.parquet", columns=["src_id", "dst_id"]
    ).to_pylist()
    started = time.time()
    with driver.session() as session:
        if drop:
            session.run(f"MATCH (n:{LABEL}) DETACH DELETE n").consume()
        session.run(
            f"CREATE CONSTRAINT e0_oe_node_id IF NOT EXISTS "
            f"FOR (n:{LABEL}) REQUIRE n.node_id IS UNIQUE"
        ).consume()
        session.run(
            f"UNWIND $rows AS r MERGE (n:{LABEL} {{node_id: r.node_id}}) "
            "SET n.entity_type=r.entity_type, n.generation=r.generation",
            rows=nodes,
        ).consume()
        session.run(
            f"UNWIND $rows AS r MATCH (a:{LABEL} {{node_id:r.src_id}}), "
            f"(b:{LABEL} {{node_id:r.dst_id}}) "
            "CREATE (a)-[:evolved_to]->(b), (b)-[:evolved_to_reverse]->(a)",
            rows=edges,
        ).consume()
        record = session.run(
            f"MATCH (n:{LABEL}) WITH count(n) AS nodes "
            f"MATCH (:{LABEL})-[r]->(:{LABEL}) RETURN nodes, count(r) AS rels"
        ).single()
    driver.close()
    return {
        "nodes": record["nodes"],
        "relationships": record["rels"],
        "seconds": time.time() - started,
    }


def load_postgres(*, drop: bool) -> dict[str, Any]:
    import psycopg
    import pyarrow.parquet as pq

    ids, vectors = _vectors()
    vector_by_id = dict(zip(ids, vectors))
    nodes = pq.read_table(
        ROOT / "nodes.parquet", columns=["node_id", "entity_type", "generation"]
    ).to_pylist()
    connection = psycopg.connect(
        host="127.0.0.1",
        port=5434,
        dbname="tridb_wiki",
        user="postgres",
        password="postgres",
        autocommit=True,
    )
    started = time.time()
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if drop:
            cursor.execute(f"DROP TABLE IF EXISTS {TABLE}")
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE} ("
            "node_id text PRIMARY KEY, entity_type text NOT NULL, generation integer, "
            f"embedding vector({DIM}) NOT NULL)"
        )
        with cursor.copy(
            f"COPY {TABLE} (node_id, entity_type, generation, embedding) FROM STDIN"
        ) as copy:
            for row in nodes:
                vector = vector_by_id[str(row["node_id"])]
                literal = "[" + ",".join(f"{value:.8g}" for value in vector) + "]"
                copy.write_row(
                    (
                        str(row["node_id"]),
                        row["entity_type"],
                        row["generation"],
                        literal,
                    )
                )
        cursor.execute(f"CREATE INDEX {TABLE}_generation ON {TABLE}(generation)")
        cursor.execute(
            f"CREATE INDEX {TABLE}_hnsw ON {TABLE} USING hnsw "
            "(embedding vector_cosine_ops) WITH (m=16, ef_construction=200)"
        )
        cursor.execute(f"ANALYZE {TABLE}")
        cursor.execute(f"SELECT count(*) FROM {TABLE}")
        count = cursor.fetchone()[0]
    connection.close()
    return {"rows": count, "seconds": time.time() - started}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["all", "milvus", "neo4j", "postgres"])
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("data/e0/openevolve_polyglot_load.json")
    )
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"environment": environment_record()}
    if args.phase in ("all", "milvus"):
        report["milvus"] = load_milvus(drop=not args.keep)
    if args.phase in ("all", "neo4j"):
        report["neo4j"] = load_neo4j(drop=not args.keep)
    if args.phase in ("all", "postgres"):
        report["postgres"] = load_postgres(drop=not args.keep)
    report["ready"] = (
        all(
            [
                report.get("milvus", {}).get("rows") == 31,
                report.get("neo4j", {}).get("nodes") == 31,
                report.get("neo4j", {}).get("relationships") == 60,
                report.get("postgres", {}).get("rows") == 31,
            ]
        )
        if args.phase == "all"
        else None
    )
    write_json(args.out, report)
    print(report)
    return 0 if report.get("ready", True) is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
