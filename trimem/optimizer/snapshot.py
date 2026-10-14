"""Immutable input artifact for constructing native graph branches.

This JSON file is experimental input, not a graph sidecar or query-time edge
store. All traversal after import uses the in-process native adjacency AM.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .compiler import QueryContext


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenSnapshot:
    path: Path
    sha256: str
    _document_json: bytes

    @property
    def document(self):
        # Each reader gets an independent view; mutating it cannot change the
        # graph constructed under an already frozen snapshot identity.
        return json.loads(self._document_json)

    @classmethod
    def read(cls, path: str | Path, expected_sha256: str | None = None):
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ValueError("frozen graph snapshot content hash changed")
        doc = json.loads(raw)
        if doc.get("schema_version") != 1:
            raise ValueError("unsupported frozen snapshot schema")
        dimension = doc.get("embedding_dimension")
        if type(dimension) is not int or not 1 <= dimension <= 2000:
            raise ValueError("snapshot needs a supported HNSW vector dimension")
        ids = set()
        for row in doc["attempts"]:
            if type(row["id"]) is not int or row["id"] <= 0 or row["id"] in ids:
                raise ValueError(
                    "snapshot attempt IDs must be unique positive integers"
                )
            ids.add(row["id"])
            if not row["task"] or type(row["is_valid"]) is not bool:
                raise ValueError("snapshot requires explicit task and validity")
            if row.get("fitness") is not None and not math.isfinite(row["fitness"]):
                raise ValueError("snapshot fitness must be finite")
            if row["is_valid"] and row.get("fitness") is None:
                raise ValueError("valid snapshot attempt needs fitness")
            vector = row.get("emb")
            if vector is not None and (
                len(vector) != dimension or not all(math.isfinite(v) for v in vector)
            ):
                raise ValueError("snapshot vector has invalid dimension or coordinates")
        if not ids:
            raise ValueError("frozen retrieval snapshot must contain attempts")
        for row in doc["attempts"]:
            if row.get("parent_id") is not None and row["parent_id"] not in ids:
                raise ValueError("snapshot parent is missing")
        edge_keys = set()
        for edge in doc["edges"]:
            if edge["src"] not in ids or edge["dst"] not in ids:
                raise ValueError("snapshot edge references a missing vertex")
            if edge["type"] not in {"lineage", "lineage_up", "similar", "similar_up"}:
                raise ValueError("unsupported snapshot edge type")
            key = (edge["src"], edge["dst"], edge["type"])
            if key in edge_keys:
                raise ValueError("snapshot contains duplicate edges")
            edge_keys.add(key)
        for query in doc.get("queries", {}).values():
            context = QueryContext(**query)
            if len(context.query_vector) != dimension:
                raise ValueError("query vector and snapshot dimension differ")
        if not doc.get("embedding_provenance"):
            raise ValueError("snapshot must identify its embedding source")
        return cls(path, digest, raw)

    @property
    def snapshot_id(self):
        return "sha256:" + self.sha256

    def query(self, checkpoint_id: str) -> QueryContext:
        query = self.document["queries"][checkpoint_id]
        query["query_vector"] = tuple(query["query_vector"])
        return QueryContext(**query)

    def load_native(self, connection):
        """Construct a new, empty private database. Never touches a shared DB."""
        doc = self.document
        dim = doc["embedding_dimension"]
        for extension in ("vector", "graph_store_am", "tjs_pg", "trimem_pg"):
            connection.execute(f"CREATE EXTENSION {extension}")
        connection.execute(f"""CREATE TABLE attempt (
            id bigint PRIMARY KEY, vid bigint UNIQUE, task text NOT NULL,
            cell text, parent_id bigint, fitness double precision,
            is_valid boolean NOT NULL, changes text, code text, emb vector({dim}))""")
        with connection.cursor().copy(
            "COPY attempt(id,task,cell,parent_id,fitness,is_valid,changes,code,emb) FROM STDIN"
        ) as copy:
            for row in doc["attempts"]:
                vector = row.get("emb")
                encoded = (
                    None
                    if vector is None
                    else "[" + ",".join(format(v, ".17g") for v in vector) + "]"
                )
                copy.write_row(
                    (
                        row["id"],
                        row["task"],
                        row.get("cell"),
                        row.get("parent_id"),
                        row.get("fitness"),
                        row["is_valid"],
                        row.get("changes"),
                        row.get("code"),
                        encoded,
                    )
                )
        connection.execute("UPDATE attempt SET vid=graph_store.gph_upsert_vertex(id)")
        vertices = dict(connection.execute("SELECT id,vid FROM attempt").fetchall())
        # Edge input exists only while loading. There is no relational edge table.
        types = {
            name: connection.execute(
                "SELECT graph_store.register_edge_type(%s)",
                (name,),
            ).fetchone()[0]
            for name in ("lineage", "lineage_up", "similar", "similar_up")
        }
        with connection.cursor() as cursor:
            cursor.executemany(
                "SELECT graph_store.gph_insert_edge(%s,%s,%s)",
                (
                    (vertices[e["src"]], vertices[e["dst"]], types[e["type"]])
                    for e in doc["edges"]
                ),
            )
        connection.execute(
            "CREATE INDEX attempt_emb_hnsw ON attempt USING hnsw (emb vector_cosine_ops)"
        )
        connection.execute("ANALYZE attempt")
        connection.execute("SELECT trimem.analyze_graph()")
