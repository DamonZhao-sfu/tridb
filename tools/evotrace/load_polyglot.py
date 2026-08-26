"""Mirror the Experience Graph into Milvus + Neo4j + pgvector for arm C.

Arm C compares two SYSTEMS answering the same question. That only means something if
both hold the same data, so this loader is written against the same normalized corpus
the GEM loader reads, not against a re-export of GEM's tables.

    python3 -m tools.evotrace.load_polyglot --receipt bench/out/polyglot/load_receipt.json

Store split, matching how a polyglot stack is actually assembled:

    Milvus    vectors + ANN            (the `tjs_open` seedless leg)
    Neo4j     typed edges + traversal  (the bounded expansion)
    pgvector  vertex attributes        (the pushed-down predicate)

The receipt is not bookkeeping. The E0 polyglot numbers were retracted on 2026-08-18
because 1,010 of 1,010 cells returned empty result sets -- the loader finished two
minutes AFTER the measurement began -- and every empty answer was silently scored as
zero. `gate_polyglot_parity` refuses to measure without a receipt naming row counts and
a completion time, and this is what writes it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_NORMALIZED = Path("data/evotrace/normalized")
DEFAULT_SCOPE = "evotrace:349117b0"
COLLECTION = "evotrace_eg"
LABEL = "EgVertex"
PG_TABLE = "eg_vertex_polyglot"

MILVUS = {"host": "127.0.0.1", "port": "19530"}
NEO4J = {"uri": "bolt://127.0.0.1:7688", "auth": ("neo4j", "testpassword")}
# The GEM database itself is on 55432; the polyglot leg gets its own pgvector so the
# comparison is between two stacks, not between one stack and a view of the other.
PG = {"host": "127.0.0.1", "port": 5434, "dbname": "tridb_wiki",
      "user": "postgres", "password": "postgres"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


def load_milvus(
    uids: list[str], vectors: np.ndarray, dim: int, kinds: list[str]
) -> int:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

    connections.connect(alias="evo", **MILVUS)
    if utility.has_collection(COLLECTION, using="evo"):
        utility.drop_collection(COLLECTION, using="evo")
    # `kind` is carried IN Milvus, not just in pgvector. Without it the entry filter
    # can only run after the ANN returns, and 18 of the 10,690 vectors are tasks:
    # the top-32 by cosine are all nodes, so a task-entry query filters down to zero.
    # Measured -- parity against GEM was 42.9% with every circle_packing query empty.
    # Denormalising one column into the vector store is what a competent polyglot
    # deployment does, and withholding it would make arm C lose on a strawman.
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("uid", DataType.VARCHAR, max_length=256),
        FieldSchema("kind", DataType.VARCHAR, max_length=16),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dim),
    ])
    collection = Collection(COLLECTION, schema, using="evo")
    batch = 2000
    for start in range(0, len(uids), batch):
        stop = min(start + batch, len(uids))
        collection.insert([
            list(range(start, stop)),
            uids[start:stop],
            kinds[start:stop],
            vectors[start:stop].tolist(),
        ])
    collection.flush()
    collection.create_index(
        "embedding",
        # Same metric as GEM's HNSW (`vector_cosine_ops`). A different metric would
        # make the two arms answer different questions.
        {"index_type": "HNSW", "metric_type": "COSINE",
         "params": {"M": 16, "efConstruction": 64}},
    )
    collection.load()
    return collection.num_entities


def load_neo4j(vertices: list[dict[str, Any]], edges: list[tuple[str, str, str]]) -> int:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J["uri"], auth=NEO4J["auth"])
    with driver.session() as session:
        session.run(f"MATCH (n:{LABEL}) DETACH DELETE n")
        session.run(
            f"CREATE INDEX eg_uid IF NOT EXISTS FOR (n:{LABEL}) ON (n.uid)"
        )
        for start in range(0, len(vertices), 5000):
            session.run(
                f"UNWIND $rows AS r CREATE (n:{LABEL}) SET n = r",
                rows=vertices[start:start + 5000],
            )
        for start in range(0, len(edges), 5000):
            session.run(
                f"UNWIND $rows AS r"
                f" MATCH (a:{LABEL} {{uid: r.src}}), (b:{LABEL} {{uid: r.dst}})"
                f" CREATE (a)-[:EG {{relation: r.relation}}]->(b)",
                rows=[{"src": s, "dst": d, "relation": rel}
                      for s, d, rel in edges[start:start + 5000]],
            )
        count = session.run(f"MATCH (n:{LABEL}) RETURN count(n) AS n").single()["n"]
    driver.close()
    return int(count)


def load_pgvector(vertices: list[dict[str, Any]]) -> int:
    import psycopg

    conn = psycopg.connect(**PG, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {PG_TABLE}")
        cur.execute(
            f"CREATE TABLE {PG_TABLE} ("
            " uid text PRIMARY KEY, kind text, task_uid text, session_uid text,"
            " backend text, domain text, language text, status text,"
            " is_valid boolean, fitness double precision, iteration integer)"
        )
        with cur.copy(
            f"COPY {PG_TABLE} (uid, kind, task_uid, session_uid, backend, domain,"
            " language, status, is_valid, fitness, iteration) FROM STDIN"
        ) as copy:
            for v in vertices:
                copy.write_row([
                    v["uid"], v["kind"], v.get("task_uid"), v.get("session_uid"),
                    v.get("backend"), v.get("domain"), v.get("language"),
                    v.get("status"), v.get("is_valid"), v.get("fitness"),
                    v.get("iteration"),
                ])
        # The predicate arm B pushes down is `kind AND is_valid AND fitness >= x`,
        # so the polyglot leg gets the index that predicate needs. Denying it one
        # would make the comparison about indexing rather than about architecture.
        cur.execute(
            f"CREATE INDEX ON {PG_TABLE} (task_uid, fitness DESC)"
            " WHERE kind = 'node' AND is_valid"
        )
        cur.execute(f"SELECT count(*) FROM {PG_TABLE}")
        count = cur.fetchone()[0]
    conn.close()
    return int(count)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normalized", type=Path, default=DEFAULT_NORMALIZED)
    ap.add_argument("--scope", default=DEFAULT_SCOPE)
    ap.add_argument("--receipt", type=Path,
                    default=Path("bench/out/polyglot/load_receipt.json"))
    args = ap.parse_args(argv)

    started = time.time()
    tasks = read_jsonl(args.normalized / "tasks.jsonl")
    sessions = read_jsonl(args.normalized / "sessions.jsonl")
    nodes = read_jsonl(args.normalized / "nodes.jsonl")
    lineage = read_jsonl(args.normalized / "lineage_edges.jsonl")

    by_session = {s["session_uid"]: s for s in sessions}
    vertices: list[dict[str, Any]] = []
    for t in tasks:
        vertices.append({"uid": t["task_uid"], "kind": "task", "task_uid": t["task_uid"],
                         "domain": t["domain"], "session_uid": None, "backend": None,
                         "language": None, "status": None, "is_valid": None,
                         "fitness": None, "iteration": None})
    for s in sessions:
        vertices.append({"uid": s["session_uid"], "kind": "session",
                         "task_uid": s["task_uid"], "session_uid": s["session_uid"],
                         "backend": s["backend"], "domain": s["domain"],
                         "language": None, "status": None, "is_valid": None,
                         "fitness": None, "iteration": None})
    for n in nodes:
        owner = by_session.get(n["session_uid"], {})
        vertices.append({"uid": n["node_uid"], "kind": "node",
                         "task_uid": n["task_uid"], "session_uid": n["session_uid"],
                         # Denormalised from the session, exactly as the GEM loader
                         # does -- these are the columns the predicate filters on.
                         "backend": owner.get("backend"), "domain": owner.get("domain"),
                         "language": n.get("language"), "status": n["status"],
                         "is_valid": n["is_valid"], "fitness": n.get("fitness"),
                         "iteration": n.get("iteration")})

    edges: list[tuple[str, str, str]] = []
    for s in sessions:
        edges.append((s["task_uid"], s["session_uid"], "has_session"))
    for n in nodes:
        edges.append((n["session_uid"], n["node_uid"], "has_node"))
    for e in lineage:
        edges.append((e["src_node_uid"], e["dst_node_uid"], "has_child"))

    blob = np.load(args.normalized / "vectors.npz", allow_pickle=False)
    uids = [str(u) for u in blob["uids"]]
    matrix = blob["vectors"].astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    print(f"loading {len(vertices):,} vertices, {len(edges):,} edges, "
          f"{len(uids):,} vectors")
    kind_by_uid = {v["uid"]: v["kind"] for v in vertices}
    kinds = [kind_by_uid.get(u, "node") for u in uids]
    milvus_rows = load_milvus(uids, matrix, matrix.shape[1], kinds)
    print(f"  milvus   {milvus_rows:,}")
    neo_rows = load_neo4j(vertices, edges)
    print(f"  neo4j    {neo_rows:,}")
    pg_rows = load_pgvector(vertices)
    print(f"  pgvector {pg_rows:,}")

    receipt = {
        "scope": args.scope,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - started, 1),
        "rows_loaded": {"milvus": milvus_rows, "neo4j": neo_rows, "pgvector": pg_rows,
                        "edges": len(edges)},
        "collection": COLLECTION, "label": LABEL, "table": PG_TABLE,
    }
    if not all(receipt["rows_loaded"].values()):
        raise SystemExit(f"a store loaded zero rows: {receipt['rows_loaded']}")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"receipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
