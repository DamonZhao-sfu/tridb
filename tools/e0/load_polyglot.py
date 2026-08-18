"""Load normalized STARK-PRIME into the Polyglot-Tuned baseline (Milvus + Neo4j + pgvector).

WHAT THIS IS
------------
Phase B of docs/e0_plan_space_execution_v0.1.0.md. The three stores are the *system under
measurement* for E0 -- the plan space being enumerated is the one an application author has
when these three engines are all they have. So the load has to be faithful to how each store
is actually used, not merely convenient:

  Milvus   : node embeddings, HNSW/COSINE, dim 1024 (Qwen3-Embedding-0.6B) -- the ANN leg
  Neo4j    : typed adjacency -- the traversal leg
  pgvector : node relational attributes + the same vectors -- the filter/rerank leg

BOTH ARCS ARE LOADED INTO NEO4J. STARK stores each PrimeKG relation twice, once per
direction with the same edge_type (verified over the full 8.1M-arc file: 50000/50000 sampled
arcs have their same-type reverse, zero self-loops). Collapsing them to one directed
relationship would silently halve traversal fan-out and break official-answer reachability;
loading them as they are keeps `typed_reachable` in Cypher identical to the audit semantics.

EDGE TYPE REPRESENTATION. `edge_type` is stored BOTH as the Neo4j relationship type
(`-[:target]->`) and as a property. Relationship-type matching is what Neo4j optimizes for,
and pinning it lets the executor emit `-[:target|interacts_with*1..2]->`; the property is
carried so a query can filter on it without the label-explosion of 18 types. Which one the
executor uses is a measured choice, not an assumed one -- see `benchmark` in this module.

NOTHING HERE MEASURES TriDB. E0's subject is the polyglot baseline.

CLI:
    python -m tools.e0.load_polyglot all
    python -m tools.e0.load_polyglot milvus|neo4j|postgres|verify|benchmark
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.e0.common import environment_record, write_json

NORMALIZED = Path("data/e0/stark_prime/normalized")
DIM = 1024
MILVUS_COLLECTION = "e0_stark_prime"
NEO4J_LABEL = "Node"
PG_TABLE = "e0_stark_prime_node"

DEFAULTS = {
    "milvus_host": "localhost",
    "milvus_port": "19530",
    "neo4j_uri": "bolt://localhost:7688",
    "neo4j_user": "neo4j",
    "neo4j_password": "testpassword",
    "pg_host": "127.0.0.1",
    "pg_port": 5434,
    "pg_db": "tridb_wiki",
    "pg_user": "postgres",
    "pg_password": "postgres",
}


def rel_type(edge_type: str) -> str:
    """Neo4j relationship types cannot contain spaces or punctuation."""
    return re.sub(r"[^A-Za-z0-9]+", "_", edge_type).strip("_").lower()


# ----------------------------------------------------------------------------------
# Milvus -- the ANN leg
# ----------------------------------------------------------------------------------


def load_milvus(cfg: dict[str, Any], *, drop: bool) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    connections.connect(alias="e0", host=cfg["milvus_host"], port=cfg["milvus_port"])
    if drop and utility.has_collection(MILVUS_COLLECTION, using="e0"):
        utility.drop_collection(MILVUS_COLLECTION, using="e0")

    collection = Collection(
        MILVUS_COLLECTION,
        CollectionSchema(
            [
                FieldSchema("node_id", DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
            ]
        ),
        using="e0",
    )

    started = time.time()
    table = pq.read_table(
        NORMALIZED / "embeddings.parquet", columns=["node_id", "embedding"]
    )
    node_ids = table["node_id"].to_pylist()
    vectors = np.asarray(table["embedding"].to_pylist(), dtype=np.float32)
    # Cosine metric on Milvus does not require unit vectors, but pgvector's `<=>` rerank leg
    # and the exact oracle must agree with it, so normalize once here and use the same
    # matrix for BOTH stores.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    batch = 4096
    for start in range(0, len(node_ids), batch):
        stop = min(start + batch, len(node_ids))
        collection.insert([node_ids[start:stop], vectors[start:stop].tolist()])
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
    return {
        "rows": int(collection.num_entities),
        "expected": len(node_ids),
        "seconds": round(time.time() - started, 1),
    }


# ----------------------------------------------------------------------------------
# Neo4j -- the traversal leg
# ----------------------------------------------------------------------------------


def load_neo4j(
    cfg: dict[str, Any], *, drop: bool, batch: int = 50_000
) -> dict[str, Any]:
    import pyarrow.parquet as pq
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"])
    )
    started = time.time()
    with driver.session() as session:
        if drop:
            # Delete in chunks: a single DETACH DELETE over 8.1M relationships blows the
            # transaction heap.
            while True:
                summary = session.run(
                    f"MATCH (n:{NEO4J_LABEL}) WITH n LIMIT 100000 "
                    "DETACH DELETE n RETURN count(*) AS c"
                ).single()
                if not summary or summary["c"] == 0:
                    break
        session.run(
            f"CREATE CONSTRAINT e0_node_id IF NOT EXISTS "
            f"FOR (n:{NEO4J_LABEL}) REQUIRE n.node_id IS UNIQUE"
        )
        session.run(
            f"CREATE INDEX e0_node_type IF NOT EXISTS "
            f"FOR (n:{NEO4J_LABEL}) ON (n.entity_type)"
        )

        nodes = pq.read_table(
            NORMALIZED / "nodes.parquet", columns=["node_id", "entity_type", "name"]
        ).to_pylist()
        for start in range(0, len(nodes), batch):
            session.run(
                f"UNWIND $rows AS r MERGE (n:{NEO4J_LABEL} {{node_id: r.node_id}}) "
                "SET n.entity_type = r.entity_type, n.name = r.name",
                rows=nodes[start : start + batch],
            )
        node_seconds = time.time() - started

        edges = pq.read_table(
            NORMALIZED / "edges.parquet", columns=["src_id", "dst_id", "edge_type"]
        )
        src = edges["src_id"].to_pylist()
        dst = edges["dst_id"].to_pylist()
        ets = edges["edge_type"].to_pylist()
        by_type: dict[str, list[dict[str, int]]] = {}
        for s, d, t in zip(src, dst, ets):
            by_type.setdefault(t, []).append({"s": s, "d": d})

        # One statement per edge type: the relationship type must be a literal in Cypher,
        # so a single parameterized UNWIND cannot cover all 18.
        for edge_type, rows in by_type.items():
            label = rel_type(edge_type)
            for start in range(0, len(rows), batch):
                session.run(
                    f"UNWIND $rows AS r "
                    f"MATCH (a:{NEO4J_LABEL} {{node_id: r.s}}), "
                    f"      (b:{NEO4J_LABEL} {{node_id: r.d}}) "
                    f"CREATE (a)-[:{label} {{edge_type: $t}}]->(b)",
                    rows=rows[start : start + batch],
                    t=edge_type,
                )

        counts = session.run(
            f"MATCH (n:{NEO4J_LABEL}) WITH count(n) AS nodes "
            "MATCH ()-[r]->() RETURN nodes, count(r) AS rels"
        ).single()
    driver.close()
    return {
        "nodes": counts["nodes"],
        "relationships": counts["rels"],
        "edge_types": len(by_type),
        "node_seconds": round(node_seconds, 1),
        "seconds": round(time.time() - started, 1),
    }


# ----------------------------------------------------------------------------------
# pgvector -- the relational filter / exact rerank leg
# ----------------------------------------------------------------------------------


def load_postgres(cfg: dict[str, Any], *, drop: bool) -> dict[str, Any]:
    import psycopg
    import pyarrow.parquet as pq

    conn = psycopg.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        dbname=cfg["pg_db"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if drop:
        cur.execute(f"DROP TABLE IF EXISTS {PG_TABLE}")
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {PG_TABLE} ("
        " node_id bigint PRIMARY KEY,"
        " entity_type text NOT NULL,"
        " name text,"
        " attributes jsonb,"
        f" embedding vector({DIM})"
        ")"
    )

    started = time.time()
    nodes = pq.read_table(
        NORMALIZED / "nodes.parquet",
        columns=["node_id", "entity_type", "name", "attributes_json"],
    )
    emb = pq.read_table(
        NORMALIZED / "embeddings.parquet", columns=["node_id", "embedding"]
    )
    vec_by_id = dict(zip(emb["node_id"].to_pylist(), emb["embedding"].to_pylist()))

    ids = nodes["node_id"].to_pylist()
    types = nodes["entity_type"].to_pylist()
    names = nodes["name"].to_pylist()
    attrs = nodes["attributes_json"].to_pylist()

    with cur.copy(
        f"COPY {PG_TABLE} (node_id, entity_type, name, attributes, embedding) FROM STDIN"
    ) as copy:
        for node_id, entity_type, name, attr in zip(ids, types, names, attrs):
            vector = vec_by_id.get(node_id)
            if vector is None:
                literal = None
            else:
                arr = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(arr)) or 1.0
                arr = arr / norm
                literal = "[" + ",".join(f"{x:.6f}" for x in arr) + "]"
            copy.write_row(
                (node_id, entity_type, name, attr if attr else None, literal)
            )

    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {PG_TABLE}_type ON {PG_TABLE} (entity_type)"
    )
    cur.execute(
        f"CREATE INDEX IF NOT EXISTS {PG_TABLE}_hnsw ON {PG_TABLE} "
        f"USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=200)"
    )
    cur.execute(f"ANALYZE {PG_TABLE}")
    cur.execute(f"SELECT count(*) FROM {PG_TABLE}")
    rows = cur.fetchone()[0]
    cur.close()
    conn.close()
    return {
        "rows": rows,
        "expected": len(ids),
        "seconds": round(time.time() - started, 1),
    }


# ----------------------------------------------------------------------------------
# Reconciliation
# ----------------------------------------------------------------------------------


def verify(cfg: dict[str, Any]) -> dict[str, Any]:
    """Counts across all three stores, plus oracle reachability re-checked IN NEO4J.

    The reachability re-check matters more than the counts: it proves the graph the
    executor will traverse is the same graph the query set was audited against. A load that
    drops one direction of the arcs passes every count check and still breaks every query.
    """
    import pyarrow.parquet as pq
    import psycopg
    from neo4j import GraphDatabase
    from pymilvus import Collection, connections

    expected_nodes = pq.ParquetFile(NORMALIZED / "nodes.parquet").metadata.num_rows
    expected_edges = pq.ParquetFile(NORMALIZED / "edges.parquet").metadata.num_rows

    connections.connect(alias="e0v", host=cfg["milvus_host"], port=cfg["milvus_port"])
    milvus_rows = int(Collection(MILVUS_COLLECTION, using="e0v").num_entities)

    conn = psycopg.connect(
        host=cfg["pg_host"],
        port=cfg["pg_port"],
        dbname=cfg["pg_db"],
        user=cfg["pg_user"],
        password=cfg["pg_password"],
    )
    cur = conn.cursor()
    cur.execute(f"SELECT count(*), count(embedding) FROM {PG_TABLE}")
    pg_rows, pg_vectors = cur.fetchone()
    cur.close()
    conn.close()

    driver = GraphDatabase.driver(
        cfg["neo4j_uri"], auth=(cfg["neo4j_user"], cfg["neo4j_password"])
    )
    queries = [
        json.loads(line)
        for line in (NORMALIZED / "queries_v0.2.jsonl").read_text().splitlines()
        if line.strip()
    ]
    unreachable = []
    with driver.session() as session:
        rec = session.run(f"MATCH (n:{NEO4J_LABEL}) RETURN count(n) AS c").single()
        neo_nodes = rec["c"]
        # SCOPED to the E0 subgraph on purpose. A bare `MATCH ()-[r]->()` counts every
        # relationship in the database, so anything another experiment left behind lands in
        # this reconciliation as phantom edges -- observed: a 99-relationship smoke-test
        # residue made the count read 8,100,597 against an expected 8,100,498.
        rec = session.run(
            f"MATCH (:{NEO4J_LABEL})-[r]->(:{NEO4J_LABEL}) RETURN count(r) AS c"
        ).single()
        neo_rels = rec["c"]
        foreign = session.run(
            f"MATCH (n) WHERE NOT n:{NEO4J_LABEL} RETURN count(n) AS c"
        ).single()["c"]

        for query in queries:
            types = "|".join(rel_type(t) for t in query["edge_types"])
            hops = int(query["hop_limit"])
            rows = session.run(
                f"MATCH (a:{NEO4J_LABEL} {{node_id: $anchor}})"
                f"-[:{types}*1..{hops}]->(b:{NEO4J_LABEL}) "
                "RETURN DISTINCT b.node_id AS id",
                anchor=int(query["anchor_ids"][0]),
            )
            reach = {r["id"] for r in rows}
            missing = [a for a in query["answer_ids"] if a not in reach]
            if missing:
                unreachable.append({"query_id": query["query_id"], "missing": missing})
    driver.close()

    checks = {
        "milvus_rows_match": milvus_rows == expected_nodes,
        "neo4j_nodes_match": neo_nodes == expected_nodes,
        "neo4j_relationships_match": neo_rels == expected_edges,
        "postgres_rows_match": pg_rows == expected_nodes,
        "postgres_vectors_complete": pg_vectors == expected_nodes,
        "all_queries_reachable_in_neo4j": not unreachable,
    }
    return {
        "schema_version": "e0-polyglot-load-v0.1.0",
        "environment": environment_record(),
        "expected": {"nodes": expected_nodes, "edges": expected_edges},
        "observed": {
            "milvus_rows": milvus_rows,
            "neo4j_nodes": neo_nodes,
            "neo4j_relationships": neo_rels,
            "postgres_rows": pg_rows,
            "postgres_vectors": pg_vectors,
            # Not a failure: reported so a foreign graph sharing the instance is VISIBLE
            # rather than silently folded into the counts above.
            "neo4j_foreign_nodes": foreign,
        },
        "unreachable_queries": unreachable,
        "checks": checks,
        "ready": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "phase",
        choices=["all", "milvus", "neo4j", "postgres", "verify"],
    )
    parser.add_argument("--keep", action="store_true", help="do not drop existing data")
    parser.add_argument("--out", type=Path, default=Path("data/e0/polyglot_load.json"))
    args = parser.parse_args(argv)

    cfg = dict(DEFAULTS)
    drop = not args.keep
    report: dict[str, Any] = {}

    if args.phase in ("all", "milvus"):
        report["milvus"] = load_milvus(cfg, drop=drop)
        print(f"[load] milvus  {report['milvus']}")
    if args.phase in ("all", "neo4j"):
        report["neo4j"] = load_neo4j(cfg, drop=drop)
        print(f"[load] neo4j   {report['neo4j']}")
    if args.phase in ("all", "postgres"):
        report["postgres"] = load_postgres(cfg, drop=drop)
        print(f"[load] postgres {report['postgres']}")
    if args.phase in ("all", "verify"):
        result = verify(cfg)
        report["verify"] = result
        for name, ok in result["checks"].items():
            print(f"[load] {'PASS' if ok else 'FAIL'} {name}")
        if result["unreachable_queries"]:
            for row in result["unreachable_queries"][:5]:
                print(f"       {row['query_id']} missing {row['missing']}")
        print(f"[load] ready={result['ready']}")

    write_json(args.out, {"environment": environment_record(), **report})
    if "verify" in report and not report["verify"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
