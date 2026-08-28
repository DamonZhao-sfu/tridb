"""Bounded-memory, content-addressed snapshots for the 1M systems track."""

from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Generic, Iterator, Mapping, Sequence, TypeVar, overload

from bench.agent_memory.evomembench.system_snapshot import SnapshotEdge, SnapshotUnit


SCALE_SNAPSHOT_SCHEMA_VERSION = "evomembench_scale_snapshot_v0.1.0"
_T = TypeVar("_T")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        stripped = value.strip().removeprefix("[").removesuffix("]")
        return tuple(float(item) for item in stripped.split(",") if item.strip())
    return tuple(float(item) for item in value)


def _float32_blob(values: Sequence[float]) -> bytes:
    packed = array("f", (float(value) for value in values))
    if sys.byteorder != "little":  # pragma: no cover - target hosts are little-endian
        packed.byteswap()
    return packed.tobytes()


def _blob_vector(value: bytes) -> tuple[float, ...]:
    unpacked = array("f")
    unpacked.frombytes(value)
    if sys.byteorder != "little":  # pragma: no cover - target hosts are little-endian
        unpacked.byteswap()
    return tuple(float(item) for item in unpacked)


def _canonical_line(kind: str, payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            {"kind": kind, **dict(payload)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


class _SQLiteSequence(Generic[_T]):
    def __init__(
        self,
        database: Path,
        *,
        table: str,
        count: int,
        decode: Any,
    ) -> None:
        self.database = database
        self.table = table
        self.count = count
        self.decode = decode

    def __len__(self) -> int:
        return self.count

    @overload
    def __getitem__(self, key: int) -> _T: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[_T, ...]: ...

    def __getitem__(self, key: int | slice) -> _T | tuple[_T, ...]:
        if isinstance(key, int):
            index = key + self.count if key < 0 else key
            if index < 0 or index >= self.count:
                raise IndexError(index)
            start, stop = index, index + 1
            scalar = True
        else:
            start, stop, step = key.indices(self.count)
            if step != 1:
                raise ValueError("scale snapshot slices require step=1")
            scalar = False
        if stop <= start:
            return ()
        with sqlite3.connect(f"file:{self.database}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                f"SELECT * FROM {self.table} WHERE rowid>? AND rowid<=? ORDER BY rowid",
                (start, stop),
            ).fetchall()
        decoded = tuple(self.decode(row) for row in rows)
        return decoded[0] if scalar else decoded

    def __iter__(self) -> Iterator[_T]:
        batch_size = 4096
        for start in range(0, self.count, batch_size):
            yield from self[start : start + batch_size]


def _decode_unit(row: Sequence[Any]) -> SnapshotUnit:
    return SnapshotUnit(
        uid=str(row[0]),
        scope_id=str(row[1]),
        node_kind=str(row[2]),
        ordinal=None if row[3] is None else int(row[3]),
        state=str(row[4]),
        summary=str(row[5]),
        payload=str(row[6]),
        embedding=_blob_vector(bytes(row[7])),
        metadata=json.loads(str(row[8])),
    )


def _decode_edge(row: Sequence[Any]) -> SnapshotEdge:
    return SnapshotEdge(
        src_uid=str(row[0]),
        dst_uid=str(row[1]),
        rel=str(row[2]),
        weight=float(row[3]),
    )


@dataclass(frozen=True)
class SQLiteExperienceSnapshot:
    manifest_path: Path
    database_path: Path
    scope_id: str
    digest: str
    database_sha256: str
    unit_count: int
    edge_count: int
    embedding_dim: int
    schema_version: str = SCALE_SNAPSHOT_SCHEMA_VERSION

    @property
    def units(self) -> _SQLiteSequence[SnapshotUnit]:
        return _SQLiteSequence(
            self.database_path,
            table="unit",
            count=self.unit_count,
            decode=_decode_unit,
        )

    @property
    def edges(self) -> _SQLiteSequence[SnapshotEdge]:
        return _SQLiteSequence(
            self.database_path,
            table="edge",
            count=self.edge_count,
            decode=_decode_edge,
        )

    def verify(self) -> None:
        if self.schema_version != SCALE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported scale snapshot schema")
        if _sha256_file(self.database_path) != self.database_sha256:
            raise ValueError("scale snapshot database digest mismatch")
        with sqlite3.connect(
            f"file:{self.database_path}?mode=ro", uri=True
        ) as connection:
            units = int(connection.execute("SELECT count(*) FROM unit").fetchone()[0])
            edges = int(connection.execute("SELECT count(*) FROM edge").fetchone()[0])
            dimensions = {
                int(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT length(embedding)/4 FROM unit"
                )
            }
        if units != self.unit_count or edges != self.edge_count:
            raise ValueError("scale snapshot row count mismatch")
        if dimensions != {self.embedding_dim}:
            raise ValueError("scale snapshot embedding dimension mismatch")

    @classmethod
    def read(cls, path: str | Path) -> "SQLiteExperienceSnapshot":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text())
        snapshot = cls(
            manifest_path=manifest_path,
            database_path=(manifest_path.parent / payload["database"]).resolve(),
            scope_id=str(payload["scope_id"]),
            digest=str(payload["digest"]),
            database_sha256=str(payload["database_sha256"]),
            unit_count=int(payload["unit_count"]),
            edge_count=int(payload["edge_count"]),
            embedding_dim=int(payload["embedding_dim"]),
            schema_version=str(payload["schema_version"]),
        )
        snapshot.verify()
        return snapshot


def export_gem_scale_snapshot(
    memory: Any,
    *,
    scope_root: str,
    output_dir: str | Path,
    fetch_size: int = 4096,
) -> SQLiteExperienceSnapshot:
    """Stream primary and cross-scope decoy state without a Python full copy."""
    if fetch_size < 1:
        raise ValueError("snapshot fetch_size must be positive")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"refusing existing scale snapshot: {root}")
    root.mkdir(parents=True)
    database_path = root / "snapshot.sqlite3"
    manifest_path = root / "manifest.json"
    sqlite = sqlite3.connect(database_path)
    sqlite.execute("PRAGMA journal_mode=DELETE")
    sqlite.execute("PRAGMA synchronous=FULL")
    sqlite.execute(
        "CREATE TABLE unit (uid TEXT PRIMARY KEY, scope_id TEXT NOT NULL,"
        " node_kind TEXT NOT NULL, ordinal INTEGER, state TEXT NOT NULL,"
        " summary TEXT NOT NULL, payload TEXT NOT NULL, embedding BLOB NOT NULL,"
        " metadata TEXT NOT NULL)"
    )
    sqlite.execute(
        "CREATE TABLE edge (src_uid TEXT NOT NULL, dst_uid TEXT NOT NULL,"
        " rel TEXT NOT NULL, weight REAL NOT NULL)"
    )
    semantic = hashlib.sha256()
    unit_count = 0
    edge_count = 0
    embedding_dim: int | None = None
    conn = memory.store.conn
    conn.commit()
    selected = "(u.scope_id=%s OR u.scope_id LIKE %s)"
    scope_pattern = scope_root + ":negative:%"
    try:
        with conn.transaction():
            with conn.cursor(name="evomem_scale_units") as cursor:
                cursor.execute(
                    "SELECT u.title,u.scope_id,u.metadata->>'node_kind',"
                    " (u.metadata->>'experience_ordinal')::integer,u.state,"
                    " coalesce(u.summary,''),coalesce(fv.value,u.summary,''),"
                    " u.embedding::text,u.metadata::text"
                    " FROM gem_unit u LEFT JOIN gem_field_value fv"
                    " ON fv.unit_id=u.id AND fv.field='memory_payload'"
                    " AND fv.valid_to IS NULL WHERE " + selected + " ORDER BY u.title",
                    (scope_root, scope_pattern),
                )
                while rows := cursor.fetchmany(fetch_size):
                    inserts = []
                    for row in rows:
                        embedding = _vector(row[7])
                        if embedding_dim is None:
                            embedding_dim = len(embedding)
                        elif len(embedding) != embedding_dim:
                            raise ValueError("inconsistent scale snapshot embeddings")
                        metadata = json.loads(str(row[8]))
                        unit = SnapshotUnit(
                            uid=str(row[0]),
                            scope_id=str(row[1]),
                            node_kind=str(row[2]),
                            ordinal=None if row[3] is None else int(row[3]),
                            state=str(row[4]),
                            summary=str(row[5]),
                            payload=str(row[6]),
                            embedding=embedding,
                            metadata=metadata,
                        )
                        blob = _float32_blob(embedding)
                        inserts.append(
                            (
                                unit.uid,
                                unit.scope_id,
                                unit.node_kind,
                                unit.ordinal,
                                unit.state,
                                unit.summary,
                                unit.payload,
                                blob,
                                json.dumps(
                                    metadata, ensure_ascii=False, sort_keys=True
                                ),
                            )
                        )
                        semantic.update(
                            _canonical_line(
                                "unit",
                                {
                                    **asdict(unit),
                                    "embedding": hashlib.sha256(blob).hexdigest(),
                                },
                            )
                        )
                    sqlite.executemany(
                        "INSERT INTO unit(uid,scope_id,node_kind,ordinal,state,summary,"
                        "payload,embedding,metadata) VALUES (?,?,?,?,?,?,?,?,?)",
                        inserts,
                    )
                    sqlite.commit()
                    unit_count += len(inserts)
            association_type = memory.store.edge_type_id("association")
            with conn.cursor(name="evomem_scale_edges") as cursor:
                cursor.execute(
                    "SELECT su.title,du.title,e.rel,e.weight FROM gem_edge e"
                    " JOIN gem_unit su ON su.id=e.src JOIN gem_unit du ON du.id=e.dst"
                    " WHERE e.tombstoned_at IS NULL AND e.edge_type=%s"
                    " AND (su.scope_id=%s OR su.scope_id LIKE %s)"
                    " AND (du.scope_id=%s OR du.scope_id LIKE %s)"
                    " ORDER BY su.title,du.title,e.rel",
                    (
                        association_type,
                        scope_root,
                        scope_pattern,
                        scope_root,
                        scope_pattern,
                    ),
                )
                while rows := cursor.fetchmany(fetch_size):
                    inserts = []
                    for row in rows:
                        edge = SnapshotEdge(
                            src_uid=str(row[0]),
                            dst_uid=str(row[1]),
                            rel=str(row[2]),
                            weight=float(row[3]),
                        )
                        inserts.append(
                            (edge.src_uid, edge.dst_uid, edge.rel, edge.weight)
                        )
                        semantic.update(_canonical_line("edge", asdict(edge)))
                    sqlite.executemany(
                        "INSERT INTO edge(src_uid,dst_uid,rel,weight) VALUES (?,?,?,?)",
                        inserts,
                    )
                    sqlite.commit()
                    edge_count += len(inserts)
    finally:
        sqlite.execute("CREATE INDEX edge_src ON edge(src_uid)")
        sqlite.commit()
        sqlite.close()
    if unit_count == 0 or embedding_dim is None:
        raise ValueError("cannot export an empty scale snapshot")
    database_sha256 = _sha256_file(database_path)
    payload = {
        "schema_version": SCALE_SNAPSHOT_SCHEMA_VERSION,
        "scope_id": scope_root,
        "database": database_path.name,
        "database_sha256": database_sha256,
        "digest": semantic.hexdigest(),
        "unit_count": unit_count,
        "edge_count": edge_count,
        "embedding_dim": embedding_dim,
        "scope_contract": "primary_scope_plus_cross_scope_negative_prefix",
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return SQLiteExperienceSnapshot.read(manifest_path)
