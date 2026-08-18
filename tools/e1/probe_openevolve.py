"""Probe each Polyglot leg for oe-000 with the exact parameters the E0 run used.

Run:  .venv/bin/python -m tools.e1.probe_openevolve
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("data/e0/openevolve/normalized/seed_42")
LABEL = "E0OpenEvolveNode"
TABLE = "e0_openevolve_node"
COLLECTION = "e0_openevolve"


def main() -> int:
    from neo4j import GraphDatabase
    from pymilvus import Collection, connections
    import psycopg

    query = json.loads(ROOT.joinpath("queries.jsonl").read_text().splitlines()[0])
    anchors = list(query["anchor_ids"])
    print("query_id", query["query_id"], "anchors", anchors)
    print("edge_types", query["edge_types"], "target", query["target_entity_type"])

    qemb = pq.read_table(ROOT / "query_embeddings.parquet")
    vec_by_id = dict(
        zip(
            [str(v) for v in qemb["query_id"].to_pylist()],
            qemb["embedding"].to_pylist(),
        )
    )
    vector = np.asarray(vec_by_id[query["query_id"]], dtype=np.float32)

    connections.connect(alias="probe", host="127.0.0.1", port="19530")
    collection = Collection(COLLECTION, using="probe")
    collection.load()
    hits = collection.search(
        data=[vector.tolist()],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
    )
    milvus_ids = [hit.id for hit in hits[0]]
    print("LEG milvus:", len(milvus_ids), milvus_ids[:3])

    driver = GraphDatabase.driver(
        "bolt://127.0.0.1:7688", auth=("neo4j", "testpassword")
    )
    with driver.session() as session:
        present = session.run(
            f"MATCH (a:{LABEL}) WHERE a.node_id IN $anchors RETURN count(a) AS n",
            anchors=anchors,
        ).single()["n"]
        print("LEG neo4j anchors present:", present)
        rels = session.run(
            f"MATCH (:{LABEL})-[r]->(:{LABEL}) "
            "RETURN type(r) AS t, count(*) AS n ORDER BY t"
        ).data()
        print("LEG neo4j relationship types:", rels)
        reached = session.run(
            f"MATCH (a:{LABEL}) WHERE a.node_id IN $anchors "
            f"MATCH (a)-[:evolved_to_reverse*1..2]->(b:{LABEL}) "
            "RETURN DISTINCT b.node_id AS node_id",
            anchors=anchors,
        ).data()
        print("LEG neo4j reached:", len(reached), [r["node_id"] for r in reached][:3])
    driver.close()

    with psycopg.connect(
        host="127.0.0.1",
        port=5434,
        dbname="tridb_wiki",
        user="postgres",
        password="postgres",
    ) as pg:
        with pg.cursor() as cursor:
            cursor.execute(f"SELECT count(*) FROM {TABLE}")
            print("LEG pg rows:", cursor.fetchone()[0])
            cursor.execute(
                f"SELECT count(*) FROM {TABLE} WHERE entity_type = %s",
                (query["target_entity_type"],),
            )
            print("LEG pg predicate matches:", cursor.fetchone()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
