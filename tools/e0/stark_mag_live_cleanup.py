"""Remove only the generated STARK-MAG live replicas between isolated E1 phases.

This tool never touches PRIME/OpenEvolve data.  It requires --execute and uses
exact database, collection, label, and table names so system-isolated runs can
reclaim capacity without broad globs or manual filesystem deletion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.e0.common import environment_record, write_json

TRIDB_DATABASE = "tridb_e0_stark_mag"
MILVUS_COLLECTION = "e0_stark_mag"
NEO4J_LABEL = "MAGNode"
POSTGRES_TABLE = "e0_stark_mag_node"


def cleanup_tridb() -> dict[str, Any]:
    import psycopg
    from psycopg import sql

    with psycopg.connect(
        host="/localhome/hza214/tridb/.tridb-pgdata",
        port=55432,
        dbname="postgres",
        autocommit=True,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_database_size(datname) FROM pg_database WHERE datname=%s",
                (TRIDB_DATABASE,),
            )
            row = cursor.fetchone()
            before = 0 if row is None else int(row[0])
            if row is not None:
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname=%s AND pid<>pg_backend_pid()",
                    (TRIDB_DATABASE,),
                )
                cursor.execute(
                    sql.SQL("DROP DATABASE {}").format(sql.Identifier(TRIDB_DATABASE))
                )
    return {
        "target": TRIDB_DATABASE,
        "bytes_before": before,
        "removed": row is not None,
    }


def cleanup_polyglot() -> dict[str, Any]:
    import psycopg
    from neo4j import GraphDatabase
    from pymilvus import connections, utility

    connections.connect(alias="e0_mag_cleanup", host="127.0.0.1", port="19530")
    collection_existed = utility.has_collection(
        MILVUS_COLLECTION, using="e0_mag_cleanup"
    )
    if collection_existed:
        utility.drop_collection(MILVUS_COLLECTION, using="e0_mag_cleanup")

    driver = GraphDatabase.driver(
        "bolt://127.0.0.1:7688", auth=("neo4j", "testpassword")
    )
    deleted_nodes = 0
    with driver.session() as session:
        while True:
            deleted = int(
                session.run(
                    f"MATCH (n:{NEO4J_LABEL}) WITH n LIMIT 20000 "
                    "DETACH DELETE n RETURN count(*) AS c"
                ).single()["c"]
            )
            deleted_nodes += deleted
            if deleted == 0:
                break
    driver.close()

    with psycopg.connect(
        host="127.0.0.1",
        port=5434,
        dbname="tridb_wiki",
        user="postgres",
        password="postgres",
        autocommit=True,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (POSTGRES_TABLE,))
            table_existed = cursor.fetchone()[0] is not None
            if table_existed:
                from psycopg import sql

                cursor.execute(
                    sql.SQL("DROP TABLE {}").format(sql.Identifier(POSTGRES_TABLE))
                )
    return {
        "milvus_collection": MILVUS_COLLECTION,
        "milvus_removed": collection_existed,
        "neo4j_label": NEO4J_LABEL,
        "neo4j_nodes_removed": deleted_nodes,
        "postgres_table": POSTGRES_TABLE,
        "postgres_removed": table_existed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["tridb", "polyglot"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("data/e0/stark_mag/cleanup.json")
    )
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "would_remove": (
                        [TRIDB_DATABASE]
                        if args.system == "tridb"
                        else [MILVUS_COLLECTION, NEO4J_LABEL, POSTGRES_TABLE]
                    ),
                    "execute": False,
                },
                indent=2,
            )
        )
        return 0
    result = cleanup_tridb() if args.system == "tridb" else cleanup_polyglot()
    report = {
        "schema_version": "e0-stark-mag-live-cleanup-v0.1.0",
        "environment": environment_record(),
        "system": args.system,
        "result": result,
        "recoverable_by": (
            "python -m tools.e0.load_tridb --staged-config "
            "configs/e1/staged_mag_v0.3.yaml --dataset stark_mag --reset"
            if args.system == "tridb"
            else "python -m tools.e0.load_stark_mag_polyglot all"
        ),
    }
    write_json(args.out, report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
