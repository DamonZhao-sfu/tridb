"""Load and verify the two E0 corpora in TriDB's native three stores.

Each corpus gets its own PostgreSQL database because graph_store_am owns one native
adjacency container per database.  Edge staging is temporary and is dropped after
batched insertion; query-time topology exists only in the native graph access method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.e0.plan_spread.config import load_config
from experiments.e0.plan_spread.model import QuerySpec
from tools.e0.common import artifact_record, environment_record, write_json

ROOT = Path(__file__).resolve().parents[2]
GRAPH_SO = ROOT / "src/graph_store/graph_store_am.so"
TJS_SO = ROOT / "src/tjs_pg/tjs_pg.so"
BENCH_SCHEMA = ROOT / "experiments/e0/tridb_schema.sql"


def _conn_args(cfg: dict[str, Any], dbname: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": str(cfg["host"]),
        "port": int(cfg["port"]),
        "dbname": dbname,
    }
    if cfg.get("user"):
        result["user"] = str(cfg["user"])
    return result


def _reset_database(cfg: dict[str, Any], *, reset: bool) -> None:
    import psycopg
    from psycopg import sql

    dbname = str(cfg["dbname"])
    if not dbname.startswith("tridb_e0_"):
        raise ValueError(f"refusing to manage non-E0 database {dbname!r}")
    with psycopg.connect(**_conn_args(cfg, "postgres"), autocommit=True) as admin:
        with admin.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            exists = cursor.fetchone() is not None
            if exists and not reset:
                raise RuntimeError(
                    f"database {dbname} exists; pass --reset to replace it"
                )
            if exists:
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = %s AND pid <> pg_backend_pid()",
                    (dbname,),
                )
                cursor.execute(
                    sql.SQL("DROP DATABASE {}").format(sql.Identifier(dbname))
                )
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


def _create_extensions(connection: Any) -> None:
    # CREATE FUNCTION validates MODULE_PATHNAME by dlopening the released image.  Run
    # CREATE EXTENSION in a short-lived backend, then rebind the benchmark entry points
    # from a fresh backend so the old and new graph images never coexist in one process.
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION vector")
        cursor.execute("CREATE EXTENSION graph_store_am")
        cursor.execute("CREATE EXTENSION tjs_pg")


def _install_schema(connection: Any) -> None:
    for path in (GRAPH_SO, TJS_SO, BENCH_SCHEMA):
        if not path.exists():
            raise FileNotFoundError(path)
    with connection.cursor() as cursor:
        schema = BENCH_SCHEMA.read_text(encoding="utf-8")
        schema = schema.replace("@GRAPH_LIB@", str(GRAPH_SO))
        schema = schema.replace("@TJS_LIB@", str(TJS_SO))
        cursor.execute(schema)


def _vec_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in values) + "]"


def _dataset_rows(spec: dict[str, Any]) -> tuple[list[Any], dict[Any, dict[str, Any]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(Path(spec["nodes"]))
    rows = table.to_pylist()
    ids = [row["node_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("node ids are not unique")
    return ids, {row["node_id"]: row for row in rows}


def _parent_map(spec: dict[str, Any], id_to_vid: dict[Any, int]) -> dict[int, int]:
    import pyarrow.parquet as pq

    if spec["directed_edges_already_include_reverse"]:
        return {}
    edges = pq.read_table(Path(spec["edges"]), columns=["src_id", "dst_id"])
    result: dict[int, int] = {}
    for src, dst in zip(edges["src_id"].to_pylist(), edges["dst_id"].to_pylist()):
        child, parent = id_to_vid[dst], id_to_vid[src]
        previous = result.setdefault(child, parent)
        if previous != parent:
            raise ValueError(f"node {dst!r} has multiple parents")
    return result


def _load_relational(
    connection: Any,
    spec: dict[str, Any],
    ids: list[Any],
    node_by_id: dict[Any, dict[str, Any]],
    parent_by_vid: dict[int, int],
) -> tuple[int, int]:
    import pyarrow.parquet as pq

    id_to_vid = {value: idx for idx, value in enumerate(ids)}
    embeddings = pq.ParquetFile(Path(spec["embeddings"]))
    dimension: int | None = None
    copied = 0
    with connection.cursor() as cursor:
        first = next(embeddings.iter_batches(batch_size=1, columns=["embedding"]))
        dimension = len(first.column(0)[0].as_py())
        cursor.execute(
            "CREATE TABLE e0_node ("
            "id bigint PRIMARY KEY, external_id text NOT NULL UNIQUE, "
            "entity_type text NOT NULL, generation integer, parent_vid bigint, "
            f"embedding vector({dimension}) NOT NULL)"
        )
        with cursor.copy(
            "COPY e0_node (id,external_id,entity_type,generation,parent_vid,embedding) "
            "FROM STDIN"
        ) as copy:
            for batch in embeddings.iter_batches(
                batch_size=512, columns=["node_id", "embedding"]
            ):
                for external_id, vector in zip(
                    batch.column(0).to_pylist(), batch.column(1).to_pylist()
                ):
                    if external_id not in id_to_vid:
                        raise ValueError(f"embedding has unknown id {external_id!r}")
                    vid = id_to_vid[external_id]
                    node = node_by_id[external_id]
                    copy.write_row(
                        (
                            vid,
                            str(external_id),
                            str(node["entity_type"]),
                            node.get("generation"),
                            parent_by_vid.get(vid),
                            _vec_literal(vector),
                        )
                    )
                    copied += 1
        if copied != len(ids):
            raise ValueError(f"embedding rows {copied} != nodes {len(ids)}")
        cursor.execute("CREATE INDEX e0_node_entity_type ON e0_node(entity_type)")
        cursor.execute("CREATE INDEX e0_node_generation ON e0_node(generation)")
        cursor.execute("CREATE INDEX e0_node_parent ON e0_node(parent_vid)")
        cursor.execute(
            "CREATE INDEX e0_node_embedding_hnsw ON e0_node USING hnsw "
            "(embedding vector_cosine_ops) WITH (m=16, ef_construction=200)"
        )
        cursor.execute("ANALYZE e0_node")
    return copied, int(dimension)


def _load_graph(
    connection: Any, spec: dict[str, Any], ids: list[Any]
) -> tuple[int, dict[str, int]]:
    import pyarrow.parquet as pq

    id_to_vid = {value: idx for idx, value in enumerate(ids)}
    edge_file = pq.ParquetFile(Path(spec["edges"]))
    type_names: set[str] = set()
    for batch in edge_file.iter_batches(batch_size=65536, columns=["edge_type"]):
        type_names.update(str(value) for value in batch.column(0).to_pylist())
    if not spec["directed_edges_already_include_reverse"]:
        type_names.update(f"{value}:reverse" for value in list(type_names))

    with connection.cursor() as cursor:
        cursor.execute("SET LOCAL work_mem = '1GB'")
        cursor.execute("SET LOCAL graph_store.assume_dense_open = on")
        cursor.execute(
            "SELECT count(graph_store.gph_insert_vertex()) FROM generate_series(1,%s)",
            (len(ids),),
        )
        inserted_vertices = int(cursor.fetchone()[0])
        if inserted_vertices != len(ids):
            raise RuntimeError("native vertex count drift during load")
        type_ids: dict[str, int] = {}
        for type_name in sorted(type_names):
            cursor.execute("SELECT graph_store.register_edge_type(%s)", (type_name,))
            type_ids[type_name] = int(cursor.fetchone()[0])

        cursor.execute(
            "CREATE TEMP TABLE e0_edge_stage "
            "(src bigint, dst bigint, type_id integer, ordinal bigint)"
        )
        copied = 0
        with cursor.copy(
            "COPY e0_edge_stage (src,dst,type_id,ordinal) FROM STDIN"
        ) as copy:
            for batch in edge_file.iter_batches(
                batch_size=65536, columns=["src_id", "dst_id", "edge_type"]
            ):
                for src, dst, raw_type in zip(
                    batch.column(0).to_pylist(),
                    batch.column(1).to_pylist(),
                    batch.column(2).to_pylist(),
                ):
                    edge_type = str(raw_type)
                    src_vid, dst_vid = id_to_vid[src], id_to_vid[dst]
                    copy.write_row((src_vid, dst_vid, type_ids[edge_type], copied))
                    copied += 1
                    if not spec["directed_edges_already_include_reverse"]:
                        reverse = f"{edge_type}:reverse"
                        copy.write_row((dst_vid, src_vid, type_ids[reverse], copied))
                        copied += 1
        cursor.execute(
            "SELECT coalesce(sum(graph_store.gph_insert_edges(src,dsts,type_id)),0) "
            "FROM (SELECT src,type_id,array_agg(dst ORDER BY ordinal) AS dsts "
            "FROM e0_edge_stage GROUP BY src,type_id ORDER BY src,type_id) grouped"
        )
        inserted_edges = int(cursor.fetchone()[0])
        if inserted_edges != copied:
            raise RuntimeError(f"native edge count {inserted_edges} != staged {copied}")
        cursor.execute("DROP TABLE e0_edge_stage")
        cursor.execute("SELECT graph_store.gph_set_identity_mode(true)")
    return copied, type_ids


def _verify(
    connection: Any,
    spec: dict[str, Any],
    ids: list[Any],
    expected_edges: int,
    type_ids: dict[str, int],
) -> dict[str, Any]:
    id_to_vid = {value: idx for idx, value in enumerate(ids)}
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*),count(embedding),graph_store.gph_vertex_count(),"
            "graph_store.gph_edge_count(),graph_store.gph_visible_edge_count() "
            "FROM e0_node"
        )
        rows, vectors, vertices, raw_edges, visible_edges = map(int, cursor.fetchone())
        cursor.execute("SET LOCAL graph_store.assume_dense_open = on")
        queries = [
            QuerySpec.from_mapping(json.loads(line))
            for line in Path(spec["queries"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        unreachable: list[dict[str, Any]] = []
        censored_queries: list[str] = []
        for query in queries:
            anchors = [id_to_vid[value] for value in query.anchor_ids]
            answers = {id_to_vid[value] for value in query.answer_ids}
            edge_types = [type_ids[value] for value in query.edge_types]
            cursor.execute(
                "SELECT graph_store.gph_traverse_bounded_multi(%s,%s,%s,%s,%s)",
                (
                    anchors,
                    query.hop_limit,
                    edge_types,
                    query.require_each_anchor,
                    50_000_000,
                ),
            )
            reach = {int(row[0]) for row in cursor}
            missing = sorted(answers - reach)
            cursor.execute("SELECT graph_store.gph_traverse_bounded_censored()")
            if bool(cursor.fetchone()[0]):
                censored_queries.append(query.query_id)
            if missing:
                unreachable.append(
                    {
                        "query_id": query.query_id,
                        "missing": [str(ids[value]) for value in missing],
                    }
                )
    checks = {
        "relational_rows_match": rows == len(ids),
        "vectors_complete": vectors == len(ids),
        "native_vertices_match": vertices == len(ids),
        "native_raw_edges_match": raw_edges == expected_edges,
        "native_visible_edges_match": visible_edges == expected_edges,
        "all_query_answers_reachable": not unreachable,
        "verification_uncensored": not censored_queries,
    }
    return {
        "expected": {"nodes": len(ids), "directed_edges": expected_edges},
        "observed": {
            "relational_rows": rows,
            "vectors": vectors,
            "native_vertices": vertices,
            "native_raw_edges": raw_edges,
            "native_visible_edges": visible_edges,
        },
        "unreachable_queries": unreachable,
        "censored_queries": censored_queries,
        "checks": checks,
        "ready": all(checks.values()),
    }


def load_dataset(name: str, spec: dict[str, Any], *, reset: bool) -> dict[str, Any]:
    import psycopg

    started = time.time()
    cfg = spec["tridb"]
    _reset_database(cfg, reset=reset)
    ids, node_by_id = _dataset_rows(spec)
    id_to_vid = {value: idx for idx, value in enumerate(ids)}
    parent_by_vid = _parent_map(spec, id_to_vid)
    with psycopg.connect(**_conn_args(cfg, str(cfg["dbname"]))) as extension_conn:
        _create_extensions(extension_conn)
    with psycopg.connect(**_conn_args(cfg, str(cfg["dbname"]))) as connection:
        _install_schema(connection)
        relational_started = time.time()
        rows, dimension = _load_relational(
            connection, spec, ids, node_by_id, parent_by_vid
        )
        connection.commit()
        relational_seconds = time.time() - relational_started
        graph_started = time.time()
        edges, type_ids = _load_graph(connection, spec, ids)
        connection.commit()
        graph_seconds = time.time() - graph_started
        verification = _verify(connection, spec, ids, edges, type_ids)
        connection.commit()
    return {
        "database": str(cfg["dbname"]),
        "rows": rows,
        "dimension": dimension,
        "directed_edges": edges,
        "edge_types": type_ids,
        "relational_seconds": round(relational_seconds, 3),
        "graph_seconds": round(graph_seconds, 3),
        "seconds": round(time.time() - started, 3),
        "verification": verification,
        "ready": verification["ready"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e0/plan_space_v0.3.yaml")
    )
    parser.add_argument(
        "--dataset", action="append", choices=["stark_prime", "openevolve"]
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("data/e0/tridb_load_v0.3.json")
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    selected = args.dataset or list(config["datasets"])
    report: dict[str, Any] = {
        "schema_version": "e0-tridb-load-v0.3.0",
        "environment": environment_record(),
        "config": artifact_record(args.config),
        "build": {
            "graph_store_am": artifact_record(GRAPH_SO),
            "tjs_pg": artifact_record(TJS_SO),
        },
        "platform_claim": "stock PostgreSQL x86_64 only; no GX10/fork sign-off",
        "datasets": {},
    }
    for name in selected:
        print(f"[tridb-e0] loading {name}", flush=True)
        report["datasets"][name] = load_dataset(
            name, config["datasets"][name], reset=args.reset
        )
        write_json(args.out, report)
        print(f"[tridb-e0] {name}: {report['datasets'][name]}", flush=True)
    report["ready"] = all(value["ready"] for value in report["datasets"].values())
    write_json(args.out, report)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
