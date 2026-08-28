"""Preload one frozen EvoMemBench Experience Graph into three live systems.

The loader deliberately imports *completed* CrossEp-Know experiences from a
previous GEM run.  It never turns benchmark questions, rubrics, or reference
answers into fake experiences.  One content-addressed snapshot is shared by:

* GEM/TriDB (PostgreSQL + pgvector + native graph AM, one transaction),
* the live Polyglot baseline (Milvus + Neo4j + PostgreSQL), and
* Cognee's PostgreSQL graph and PGVector storage adapters.

Cognee receives the already-modelled nodes, edges, and Qwen embeddings.  This
is a ``prebuilt_graph_import`` and does not call ``cognify()`` or an LLM.  It is
therefore suitable for storage/retrieval parity experiments, but must not be
reported as Cognee's native graph-construction quality or construction cost.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Sequence
from urllib.parse import unquote, urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from bench.agent_memory.evomembench.leakage import assert_online_payload
from bench.agent_memory.evomembench.multi_system import (
    LiveMultiSystemExperienceStore,
    MultiSystemConfig,
)
from bench.agent_memory.evomembench.system_snapshot import (
    ExperienceSnapshot,
    SnapshotEdge,
    SnapshotUnit,
    export_gem_snapshot,
)
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.store import vec_literal


RECEIPT_SCHEMA_VERSION = "evomembench_three_system_preload_v0.1.0"
DEFAULT_BUNDLE_SCOPE = "evomembench:crossep_know:pilot_b2:completed"
COGNEE_IMPORT_KIND = "prebuilt_graph_import"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_from_dsn(dsn: str) -> str:
    parsed = urlparse(dsn)
    name = unquote(parsed.path.lstrip("/")) if parsed.scheme else ""
    if not name:
        raise ValueError("target DSN must declare dbname")
    return str(name)


def create_fresh_database(*, admin_dsn: str, target_dsn: str, owner: str) -> None:
    """Create exactly one named database; existing state is a hard failure."""
    from psycopg import connect, sql

    name = _database_from_dsn(target_dsn)
    with connect(admin_dsn, autocommit=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname=%s", (name,)
        ).fetchone()
        if exists is not None:
            raise FileExistsError(f"refusing existing PostgreSQL database {name}")
        connection.execute(
            sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(name), sql.Identifier(owner)
            )
        )


def _context_uid(source_scope: str) -> str:
    marker = ":context:"
    if marker not in source_scope:
        raise ValueError(f"scope lacks CrossEp-Know context marker: {source_scope!r}")
    value = source_scope.rsplit(marker, 1)[1]
    if not value:
        raise ValueError(f"scope has empty context id: {source_scope!r}")
    return value


def export_completed_bundle(
    *, source_dsn: str, source_scope_pattern: str, bundle_scope: str
) -> ExperienceSnapshot:
    """Merge session-local GEM scopes into one collision-free snapshot bundle."""
    memory = TriDBGovernedMemory.connect(source_dsn, dim=1024)
    try:
        scope_rows = memory.store.conn.execute(
            "SELECT DISTINCT scope_id FROM gem_unit WHERE scope_id LIKE %s ORDER BY 1",
            (source_scope_pattern,),
        ).fetchall()
        source_scopes = [str(row[0]) for row in scope_rows]
        if not source_scopes:
            raise ValueError(
                f"no completed GEM scopes matched {source_scope_pattern!r}"
            )

        units: list[SnapshotUnit] = []
        edges: list[SnapshotEdge] = []
        for source_scope in source_scopes:
            session = export_gem_snapshot(memory, scope_id=source_scope)
            context_uid = _context_uid(source_scope)
            target_scope = f"evomembench:crossep_know:context:{context_uid}"
            uid_map = {
                unit.uid: f"{context_uid}::{unit.uid}" for unit in session.units
            }
            for unit in session.units:
                metadata = {
                    **dict(unit.metadata),
                    "snapshot_source_scope": source_scope,
                    "snapshot_context_uid": context_uid,
                }
                candidate = SnapshotUnit(
                    uid=uid_map[unit.uid],
                    scope_id=target_scope,
                    node_kind=unit.node_kind,
                    ordinal=unit.ordinal,
                    state=unit.state,
                    summary=unit.summary,
                    payload=unit.payload,
                    embedding=unit.embedding,
                    metadata=metadata,
                )
                assert_online_payload(
                    {
                        "summary": candidate.summary,
                        "payload": candidate.payload,
                        "metadata": candidate.metadata,
                    },
                    location=f"snapshot[{candidate.uid}]",
                )
                units.append(candidate)
            edges.extend(
                SnapshotEdge(
                    src_uid=uid_map[edge.src_uid],
                    dst_uid=uid_map[edge.dst_uid],
                    rel=edge.rel,
                    weight=edge.weight,
                )
                for edge in session.edges
            )
    finally:
        memory.close()

    snapshot = ExperienceSnapshot.build(
        scope_id=bundle_scope, units=units, edges=edges
    )
    kinds: dict[str, int] = {}
    for unit in snapshot.units:
        kinds[unit.node_kind] = kinds.get(unit.node_kind, 0) + 1
    if kinds.get("experience", 0) == 0:
        raise ValueError("bundle contains no completed experience nodes")
    return snapshot


def _snapshot_shape(snapshot: ExperienceSnapshot) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    scopes = set()
    dimensions = set()
    for unit in snapshot.units:
        kinds[unit.node_kind] = kinds.get(unit.node_kind, 0) + 1
        scopes.add(unit.scope_id)
        dimensions.add(len(unit.embedding))
    return {
        "digest": snapshot.digest,
        "units": len(snapshot.units),
        "edges": len(snapshot.edges),
        "node_kinds": dict(sorted(kinds.items())),
        "session_scopes": len(scopes),
        "embedding_dimensions": sorted(dimensions),
    }


def load_gem_snapshot(
    *, snapshot: ExperienceSnapshot, dsn: str, admin_dsn: str, owner: str
) -> dict[str, Any]:
    """Load the frozen snapshot as one GEM transition and one PostgreSQL xid."""
    create_fresh_database(admin_dsn=admin_dsn, target_dsn=dsn, owner=owner)
    memory = TriDBGovernedMemory.connect(dsn, dim=len(snapshot.units[0].embedding))
    started = time.perf_counter()
    try:
        schema_receipt = memory.init_schema()
        with memory.store.transition("ingest", snapshot.scope_id, "construction") as tx:
            tx.lock_writer()
            uid_to_id: dict[str, int] = {}
            for unit in snapshot.units:
                unit_id = tx.allocate_vertex()
                uid_to_id[unit.uid] = unit_id
                tx.execute(
                    "INSERT INTO gem_unit"
                    " (id,scope_id,title,summary,embedding,state,metadata)"
                    " VALUES (%s,%s,%s,%s,%s::vector,%s,%s::jsonb)",
                    (
                        unit_id,
                        unit.scope_id,
                        unit.uid,
                        unit.summary,
                        vec_literal(unit.embedding),
                        unit.state,
                        json.dumps(dict(unit.metadata), ensure_ascii=False),
                    ),
                )
                tx.delta.units_created += 1
                tx.execute(
                    "INSERT INTO gem_field_value"
                    " (unit_id,field,value,valid_from,operator,transition_id)"
                    " VALUES (%s,'memory_payload',%s,%s,%s,%s)",
                    (
                        unit_id,
                        unit.payload,
                        "2026-08-25T00:00:00+00:00",
                        "evomembench_snapshot_import",
                        tx.transition_id,
                    ),
                )
                tx.delta.fields_appended += 1
            for edge in snapshot.edges:
                memory.store.link(
                    tx,
                    uid_to_id[edge.src_uid],
                    uid_to_id[edge.dst_uid],
                    kind="association",
                    rel=edge.rel,
                    weight=edge.weight,
                )

        row = memory.store.conn.execute(
            "SELECT count(*), count(DISTINCT scope_id),"
            " min(vector_dims(embedding)), max(vector_dims(embedding)) FROM gem_unit"
        ).fetchone()
        edge_count = int(
            memory.store.conn.execute(
                "SELECT count(*) FROM gem_edge WHERE tombstoned_at IS NULL"
            ).fetchone()[0]
        )
        native_visible_arcs = int(
            memory.store.conn.execute(
                "SELECT graph_store.gph_visible_edge_count()"
            ).fetchone()[0]
        )
        native_physical_arcs = int(
            memory.store.conn.execute("SELECT graph_store.gph_edge_count()").fetchone()[0]
        )
        return {
            "system": "gem",
            "storage_shape": "PostgreSQL + pgvector + native graph AM",
            "database": _database_from_dsn(dsn),
            "snapshot_digest": snapshot.digest,
            "units": int(row[0]),
            "session_scopes": int(row[1]),
            "embedding_dim_min": int(row[2]),
            "embedding_dim_max": int(row[3]),
            "relational_edge_metadata_rows": edge_count,
            "native_graph_arcs": native_visible_arcs,
            "native_graph_physical_arcs": native_physical_arcs,
            "schema": schema_receipt,
            "load_seconds": time.perf_counter() - started,
            "gpu_calls": 0,
            "llm_calls": 0,
        }
    finally:
        memory.close()


def load_polyglot_snapshot(
    *, snapshot: ExperienceSnapshot, config: MultiSystemConfig
) -> dict[str, Any]:
    started = time.perf_counter()
    with LiveMultiSystemExperienceStore(config) as store:
        preflight = store.preflight()
        if preflight["milvus_collection_exists"]:
            raise FileExistsError(
                f"refusing existing Milvus collection {config.collection}"
            )
        loaded = store.load_snapshot(snapshot)
        from neo4j import GraphDatabase
        from psycopg import connect as pg_connect
        from pymilvus import Collection

        collection = Collection(config.collection, using=config.milvus_alias)
        collection.flush()
        milvus_entities = int(collection.num_entities)
        with GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        ) as driver:
            with driver.session() as session:
                neo4j_nodes = int(
                    session.run(
                        "MATCH (n:EvoMemUnit {namespace:$namespace})"
                        " RETURN count(n) AS n",
                        namespace=config.namespace,
                    ).single()["n"]
                )
                neo4j_edges = int(
                    session.run(
                        "MATCH (:EvoMemUnit {namespace:$namespace})"
                        "-[e:EVO_ASSOCIATION]->"
                        "(:EvoMemUnit {namespace:$namespace}) RETURN count(e) AS n",
                        namespace=config.namespace,
                    ).single()["n"]
                )
        with pg_connect(config.pg_dsn) as connection:
            pg_rows = int(
                connection.execute(
                    "SELECT count(*) FROM evomem_system_unit WHERE namespace=%s",
                    (config.namespace,),
                ).fetchone()[0]
            )
    return {
        "system": "polyglot",
        "storage_shape": "Milvus HNSW + Neo4j adjacency + PostgreSQL payload/filter",
        **loaded,
        "preflight": preflight,
        "verified": {
            "milvus_entities": milvus_entities,
            "neo4j_nodes": neo4j_nodes,
            "neo4j_edges": neo4j_edges,
            "postgresql_rows": pg_rows,
        },
        "wall_seconds": time.perf_counter() - started,
        "gpu_calls": 0,
        "llm_calls": 0,
    }


def _configure_cognee_database(
    *, host: str, port: int, user: str, password: str, database: str, dim: int
) -> None:
    """Apply the local Cognee PostgreSQL configuration after its .env import."""
    os.environ.update(
        {
            "USE_UNIFIED_PROVIDER": "",
            "ENABLE_BACKEND_ACCESS_CONTROL": "false",
            "CACHING": "false",
            "DB_PROVIDER": "postgres",
            "DB_HOST": host,
            "DB_PORT": str(port),
            "DB_USERNAME": user,
            "DB_PASSWORD": password,
            "DB_NAME": database,
            "GRAPH_DATABASE_PROVIDER": "postgres",
            "GRAPH_DATABASE_HOST": host,
            "GRAPH_DATABASE_PORT": str(port),
            "GRAPH_DATABASE_USERNAME": user,
            "GRAPH_DATABASE_PASSWORD": password,
            "GRAPH_DATABASE_NAME": database,
            "VECTOR_DB_PROVIDER": "pgvector",
            "VECTOR_DB_HOST": host,
            "VECTOR_DB_PORT": str(port),
            "VECTOR_DB_USERNAME": user,
            "VECTOR_DB_PASSWORD": password,
            "VECTOR_DB_NAME": database,
            "EMBEDDING_DIMENSIONS": str(dim),
            "TELEMETRY_DISABLED": "true",
            "COGNEE_TELEMETRY_DISABLED": "true",
            "LOG_LEVEL": "ERROR",
        }
    )
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.relational.config import get_relational_config
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.config import get_vectordb_config

    for cached in (get_graph_config, get_relational_config, get_vectordb_config):
        cached.cache_clear()
    create_relational_engine.cache_clear()


class _PrecomputedEmbeddingEngine:
    """Loader-only engine; it returns the exact vectors frozen in the snapshot."""

    def __init__(self, vectors: dict[str, Sequence[float]], dim: int) -> None:
        self._vectors = vectors
        self._dim = dim

    async def embed_text(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._vectors]
        if missing:
            raise KeyError(f"precomputed embedding missing for {missing[0][:120]!r}")
        return [list(self._vectors[text]) for text in texts]

    def get_vector_size(self) -> int:
        return self._dim

    def get_batch_size(self) -> int:
        return 128


async def _load_cognee_async(
    *,
    snapshot: ExperienceSnapshot,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> dict[str, Any]:
    import asyncpg

    bootstrap = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )
    try:
        await bootstrap.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await bootstrap.close()

    # Import Cognee first because it loads its repository .env with override;
    # apply the isolated target configuration immediately afterwards.
    import cognee

    del cognee  # import side effects are the only thing needed here
    dim = len(snapshot.units[0].embedding)
    _configure_cognee_database(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        dim=dim,
    )

    from pydantic import Field
    from cognee.infrastructure.databases.graph.get_graph_engine import (
        create_graph_engine,
    )
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
        PGVectorAdapter,
    )
    from cognee.low_level import DataPoint

    class EvoMemUnit(DataPoint):
        name: str
        uid: str
        scope_id: str
        node_kind: str
        ordinal: int | None = None
        state: str
        summary: str
        payload: str
        source_metadata: dict[str, Any] = Field(default_factory=dict)
        text: str
        metadata: dict[str, Any] = {
            "index_fields": ["text"],
            "identity_fields": ["uid"],
        }

    graph = create_graph_engine(
        graph_database_provider="postgres",
        graph_file_path="",
        graph_database_name=database,
        graph_database_username=user,
        graph_database_password=password,
        graph_database_host=host,
        graph_database_port=str(port),
    )
    await graph.initialize()

    texts: dict[str, Sequence[float]] = {}
    ids: dict[str, UUID] = {}
    data_points: list[EvoMemUnit] = []
    for unit in snapshot.units:
        point_id = uuid5(NAMESPACE_URL, f"cognee:evomembench:{unit.uid}")
        ids[unit.uid] = point_id
        # Cognee's IndexSchema calls DataPoint.get_embeddable_data(), which
        # strips the text before invoking the embedding engine. Normalize the
        # loader lookup key identically so source summaries with trailing space
        # still resolve to their frozen vector.
        text = f"[evomembench_uid={unit.uid}] {unit.summary}".strip()
        texts[text] = unit.embedding
        data_points.append(
            EvoMemUnit(
                id=point_id,
                name=unit.uid,
                uid=unit.uid,
                scope_id=unit.scope_id,
                node_kind=unit.node_kind,
                ordinal=unit.ordinal,
                state=unit.state,
                summary=unit.summary,
                payload=unit.payload,
                source_metadata=dict(unit.metadata),
                text=text,
            )
        )

    await graph.add_nodes(data_points)
    graph_edges = [
        (
            str(ids[edge.src_uid]),
            str(ids[edge.dst_uid]),
            "EVO_ASSOCIATION",
            {"rel": edge.rel, "weight": edge.weight},
        )
        for edge in snapshot.edges
    ]
    await graph.add_edges(graph_edges)

    vector = PGVectorAdapter(
        connection_string=(
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        ),
        api_key=None,
        embedding_engine=_PrecomputedEmbeddingEngine(texts, dim),
    )
    await vector.index_data_points("EvoMemUnit", "text", data_points)
    probe = await vector.search("EvoMemUnit_text", data_points[0].text, limit=1)
    probe_ids = [str(getattr(item, "id", "")) for item in probe]

    # Verify through SQL rather than relying on adapter-private counters.
    verification = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )
    try:
        graph_nodes = int(await verification.fetchval("SELECT count(*) FROM graph_node"))
        graph_edge_count = int(
            await verification.fetchval("SELECT count(*) FROM graph_edge")
        )
        vector_rows = int(
            await verification.fetchval('SELECT count(*) FROM "EvoMemUnit_text"')
        )
        vector_dims = await verification.fetchrow(
            'SELECT min(vector_dims(vector)) AS min_dim,'
            ' max(vector_dims(vector)) AS max_dim FROM "EvoMemUnit_text"'
        )
    finally:
        await verification.close()
    await vector.close()
    return {
        "system": "cognee",
        "version": "1.5.0-local",
        "import_kind": COGNEE_IMPORT_KIND,
        "construction_quality_comparable": False,
        "construction_cost_comparable": False,
        "retrieval_storage_parity_eligible": True,
        "storage_shape": "Cognee PostgreSQL graph adapter + Cognee PGVector adapter",
        "database": database,
        "snapshot_digest": snapshot.digest,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edge_count,
        "vector_rows": vector_rows,
        "embedding_dim_min": int(vector_dims["min_dim"]),
        "embedding_dim_max": int(vector_dims["max_dim"]),
        "self_vector_probe_first_id": probe_ids[0] if probe_ids else None,
        "self_vector_probe_expected_id": str(data_points[0].id),
        "gpu_calls": 0,
        "llm_calls": 0,
    }


def load_cognee_snapshot(
    *,
    snapshot: ExperienceSnapshot,
    admin_dsn: str,
    dsn: str,
    owner: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database_precreated: bool = False,
) -> dict[str, Any]:
    if not database_precreated:
        create_fresh_database(admin_dsn=admin_dsn, target_dsn=dsn, owner=owner)
    database = _database_from_dsn(dsn)
    started = time.perf_counter()
    result = asyncio.run(
        _load_cognee_async(
            snapshot=snapshot,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
    )
    result["load_seconds"] = time.perf_counter() - started
    return result


def _require_equal_counts(
    snapshot: ExperienceSnapshot, system_receipts: Iterable[dict[str, Any]]
) -> None:
    expected_units = len(snapshot.units)
    expected_edges = len(snapshot.edges)
    for receipt in system_receipts:
        if receipt["system"] == "gem":
            observed_units = receipt["units"]
            observed_edges = receipt["native_graph_arcs"]
        elif receipt["system"] == "polyglot":
            observed_units = receipt["verified"]["milvus_entities"]
            observed_edges = receipt["verified"]["neo4j_edges"]
        elif receipt["system"] == "cognee":
            observed_units = receipt["graph_nodes"]
            observed_edges = receipt["graph_edges"]
        else:  # pragma: no cover - caller contract
            raise ValueError(f"unknown receipt system {receipt['system']!r}")
        if observed_units != expected_units or observed_edges != expected_edges:
            raise RuntimeError(
                f"{receipt['system']} parity failure: "
                f"units={observed_units}/{expected_units}, "
                f"edges={observed_edges}/{expected_edges}"
            )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _load_command(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = ExperienceSnapshot.read(args.snapshot)
    system_receipts: list[dict[str, Any]] = []
    if "gem" in args.systems:
        system_receipts.append(
            load_gem_snapshot(
                snapshot=snapshot,
                dsn=args.gem_dsn,
                admin_dsn=args.native_admin_dsn,
                owner=args.native_owner,
            )
        )
    if "polyglot" in args.systems:
        system_receipts.append(
            load_polyglot_snapshot(
                snapshot=snapshot,
                config=MultiSystemConfig(
                    namespace=args.polyglot_namespace,
                    milvus_host=args.milvus_host,
                    milvus_port=args.milvus_port,
                    neo4j_uri=args.neo4j_uri,
                    neo4j_user=args.neo4j_user,
                    neo4j_password=args.neo4j_password,
                    pg_dsn=args.polyglot_pg_dsn,
                ),
            )
        )
    if "cognee" in args.systems:
        system_receipts.append(
            load_cognee_snapshot(
                snapshot=snapshot,
                admin_dsn=args.native_admin_dsn,
                dsn=args.cognee_dsn,
                owner=args.native_owner,
                host=args.cognee_host,
                port=args.cognee_port,
                user=args.cognee_user,
                password=args.cognee_password,
                database_precreated=args.cognee_database_precreated,
            )
        )
    _require_equal_counts(snapshot, system_receipts)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": _snapshot_shape(snapshot),
        "snapshot_path": str(args.snapshot.resolve()),
        "snapshot_file_sha256": _sha256(args.snapshot),
        "systems": system_receipts,
        "parity": {
            "same_snapshot_digest": all(
                item["snapshot_digest"] == snapshot.digest
                for item in system_receipts
            ),
            "unit_count_equal": True,
            "edge_count_equal": True,
            "gpu_calls": 0,
            "llm_calls": 0,
        },
        "claim_boundary": {
            "dataset_slice": "CrossEp-Know completed-experience pilot B2",
            "not_full_120_session_dataset": True,
            "cognee_import": COGNEE_IMPORT_KIND,
            "cognee_native_cognify_measured": False,
        },
    }
    _write_json(args.receipt, receipt)
    return receipt


def _merge_command(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [json.loads(path.read_text()) for path in args.receipts]
    if any(item.get("status") != "complete" for item in inputs):
        raise RuntimeError("cannot merge an incomplete preload receipt")
    digests = {item["snapshot"]["digest"] for item in inputs}
    file_hashes = {item["snapshot_file_sha256"] for item in inputs}
    if len(digests) != 1 or len(file_hashes) != 1:
        raise RuntimeError("preload receipts do not reference the same snapshot")
    systems = [system for item in inputs for system in item["systems"]]
    names = [system["system"] for system in systems]
    if sorted(names) != ["cognee", "gem", "polyglot"]:
        raise RuntimeError(f"expected one receipt per system, observed {names!r}")
    snapshot = ExperienceSnapshot.read(inputs[0]["snapshot_path"])
    _require_equal_counts(snapshot, systems)
    merged = {
        **inputs[0],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "systems": systems,
        "source_receipts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in args.receipts
        ],
        "parity": {
            "same_snapshot_digest": all(
                item["snapshot_digest"] == snapshot.digest for item in systems
            ),
            "unit_count_equal": True,
            "edge_count_equal": True,
            "embedding_dim_equal": True,
            "gpu_calls": sum(int(item["gpu_calls"]) for item in systems),
            "llm_calls": sum(int(item["llm_calls"]) for item in systems),
        },
    }
    _write_json(args.output, merged)
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--source-dsn", required=True)
    export_parser.add_argument(
        "--source-scope-pattern", default="%:gem_fused:evomembench:crossep_know:context:%"
    )
    export_parser.add_argument("--bundle-scope", default=DEFAULT_BUNDLE_SCOPE)
    export_parser.add_argument("--snapshot", type=Path, required=True)

    load_parser = subparsers.add_parser("load")
    load_parser.add_argument("--snapshot", type=Path, required=True)
    load_parser.add_argument("--receipt", type=Path, required=True)
    load_parser.add_argument(
        "--systems",
        nargs="+",
        choices=("gem", "polyglot", "cognee"),
        default=["gem", "polyglot", "cognee"],
    )
    load_parser.add_argument("--native-admin-dsn", required=True)
    load_parser.add_argument("--native-owner", default="hza214")
    load_parser.add_argument("--gem-dsn", required=True)
    load_parser.add_argument("--cognee-dsn", required=True)
    load_parser.add_argument("--cognee-host", default="127.0.0.1")
    load_parser.add_argument("--cognee-port", type=int, default=55432)
    load_parser.add_argument("--cognee-user", default="hza214")
    load_parser.add_argument("--cognee-password", default="unused")
    load_parser.add_argument("--cognee-database-precreated", action="store_true")
    load_parser.add_argument("--polyglot-namespace", required=True)
    load_parser.add_argument("--milvus-host", default="127.0.0.1")
    load_parser.add_argument("--milvus-port", default="19530")
    load_parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:7688")
    load_parser.add_argument("--neo4j-user", default="neo4j")
    load_parser.add_argument("--neo4j-password", default="testpassword")
    load_parser.add_argument("--polyglot-pg-dsn", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "export":
        if args.snapshot.exists():
            raise FileExistsError(f"refusing existing snapshot {args.snapshot}")
        snapshot = export_completed_bundle(
            source_dsn=args.source_dsn,
            source_scope_pattern=args.source_scope_pattern,
            bundle_scope=args.bundle_scope,
        )
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write(args.snapshot)
        print(json.dumps(_snapshot_shape(snapshot), ensure_ascii=False, indent=2))
        return 0
    if args.command == "merge":
        if args.output.exists():
            raise FileExistsError(f"refusing existing receipt {args.output}")
        receipt = _merge_command(args)
    else:
        receipt = _load_command(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
