"""Canonical, content-addressed Experience Graph snapshots for system parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.agent_memory.evomembench.system_protocol import canonical_digest


SNAPSHOT_SCHEMA_VERSION = "evomembench_experience_snapshot_v0.1.0"


def _vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        stripped = value.strip().removeprefix("[").removesuffix("]")
        return tuple(float(item) for item in stripped.split(",") if item.strip())
    return tuple(float(item) for item in value)


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    if not isinstance(parsed, Mapping):
        raise ValueError("unit metadata must be an object")
    return dict(parsed)


@dataclass(frozen=True)
class SnapshotUnit:
    uid: str
    scope_id: str
    node_kind: str
    ordinal: int | None
    state: str
    summary: str
    payload: str
    embedding: tuple[float, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.uid or not self.scope_id or not self.node_kind:
            raise ValueError("snapshot unit is missing identity fields")
        if not self.embedding:
            raise ValueError("snapshot unit requires its stored embedding")


@dataclass(frozen=True)
class SnapshotEdge:
    src_uid: str
    dst_uid: str
    rel: str
    weight: float


@dataclass(frozen=True)
class ExperienceSnapshot:
    scope_id: str
    units: tuple[SnapshotUnit, ...]
    edges: tuple[SnapshotEdge, ...]
    digest: str
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        scope_id: str,
        units: Sequence[SnapshotUnit],
        edges: Sequence[SnapshotEdge],
    ) -> "ExperienceSnapshot":
        ordered_units = tuple(sorted(units, key=lambda item: item.uid))
        ordered_edges = tuple(
            sorted(edges, key=lambda item: (item.src_uid, item.dst_uid, item.rel))
        )
        if len({item.uid for item in ordered_units}) != len(ordered_units):
            raise ValueError("snapshot contains duplicate unit uids")
        known = {item.uid for item in ordered_units}
        if any(
            edge.src_uid not in known or edge.dst_uid not in known
            for edge in ordered_edges
        ):
            raise ValueError("snapshot edge escapes its unit set")
        unsigned = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "scope_id": scope_id,
            "units": [asdict(item) for item in ordered_units],
            "edges": [asdict(item) for item in ordered_edges],
        }
        return cls(
            scope_id=scope_id,
            units=ordered_units,
            edges=ordered_edges,
            digest=canonical_digest(unsigned),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "units": [asdict(item) for item in self.units],
            "edges": [asdict(item) for item in self.edges],
            "digest": self.digest,
        }

    def verify(self) -> None:
        rebuilt = self.build(scope_id=self.scope_id, units=self.units, edges=self.edges)
        if rebuilt.digest != self.digest:
            raise ValueError("experience snapshot digest mismatch")

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n"
        )

    @classmethod
    def read(cls, path: str | Path) -> "ExperienceSnapshot":
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported experience snapshot schema")
        snapshot = cls(
            scope_id=str(payload["scope_id"]),
            units=tuple(
                SnapshotUnit(
                    **{
                        **row,
                        "embedding": tuple(row["embedding"]),
                    }
                )
                for row in payload["units"]
            ),
            edges=tuple(SnapshotEdge(**row) for row in payload["edges"]),
            digest=str(payload["digest"]),
        )
        snapshot.verify()
        return snapshot


def export_gem_snapshot(memory: Any, *, scope_id: str) -> ExperienceSnapshot:
    """Export the exact committed GEM graph used by the fused query.

    This is an untimed setup operation.  Evaluation labels are absent because
    GEM's online payload gate never admits them into these tables.
    """
    rows = memory.store.conn.execute(
        "SELECT id, title, summary, embedding::text, metadata::text, state"
        " FROM gem_unit WHERE scope_id=%s ORDER BY id",
        (scope_id,),
    ).fetchall()
    if not rows:
        raise ValueError(f"cannot export empty GEM scope {scope_id!r}")
    id_to_uid = {int(row[0]): str(row[1]) for row in rows}
    ids = list(id_to_uid)
    payload_rows = memory.store.conn.execute(
        "SELECT unit_id, value FROM gem_field_value"
        " WHERE unit_id=ANY(%s) AND field='memory_payload' AND valid_to IS NULL",
        (ids,),
    ).fetchall()
    payload_by_id = {int(row[0]): str(row[1]) for row in payload_rows}
    units: list[SnapshotUnit] = []
    for row in rows:
        unit_id = int(row[0])
        metadata = _metadata(row[4])
        raw_ordinal = metadata.get("experience_ordinal")
        units.append(
            SnapshotUnit(
                uid=str(row[1]),
                scope_id=scope_id,
                node_kind=str(metadata.get("node_kind", "unknown")),
                ordinal=None if raw_ordinal is None else int(raw_ordinal),
                state=str(row[5]),
                summary=str(row[2] or ""),
                payload=payload_by_id.get(unit_id, str(row[2] or "")),
                embedding=_vector(row[3]),
                metadata=metadata,
            )
        )
    association_type = memory.store.edge_type_id("association")
    edge_rows = memory.store.conn.execute(
        "SELECT src, dst, rel, weight FROM gem_edge"
        " WHERE tombstoned_at IS NULL AND edge_type=%s"
        " AND src=ANY(%s) AND dst=ANY(%s)"
        " ORDER BY src, dst, rel",
        (association_type, ids, ids),
    ).fetchall()
    edges = [
        SnapshotEdge(
            src_uid=id_to_uid[int(row[0])],
            dst_uid=id_to_uid[int(row[1])],
            rel=str(row[2]),
            weight=float(row[3]),
        )
        for row in edge_rows
    ]
    return ExperienceSnapshot.build(scope_id=scope_id, units=units, edges=edges)
