"""Connection, schema and the idempotent Experience Graph loader.

Follows :mod:`bench.agent_memory.gem.store`'s contracts because they are the engine's,
not GEM's: vids come from ``graph_store.gph_allocated_vids()`` under the single-writer
advisory lock, and ``gem_eg_vertex.id`` must equal the vid or tjs_open cannot resolve a
graph-reached candidate.

The loader is deliberately boring in one respect: **topology goes into the access
method, and the relational `gem_eg_edge` row is the audit mirror, never the query path**
(CLAUDE.md rule 3). The mirror doubles as the idempotence guard — an AM edge is inserted
only when the mirror insert actually inserted, so re-running a load adds nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

DEFAULT_DSN = "postgresql:///postgres?host=127.0.0.1&port=55432"

#: Qwen3-Embedding-0.6B, the model bench/agent_memory already serves on :8001.
DEFAULT_DIM = 1024

#: Serialises vid allocation. Distinct from GEM's key so an EG load and a GEM ingest
#: do not block each other for no reason — they allocate from the SAME global vid
#: space, so they must NOT run concurrently; see :meth:`EgStore.allocate_vertices`.
WRITER_LOCK_KEY = "gem_eg_vid_alloc"

#: Native edge types, registered with graph_store.register_edge_type().
#:
#: GEM registers exactly two types because it needs "every extension edge regardless of
#: relation" to be one equality filter. The Experience Graph has the opposite need: its
#: relations ARE the query surface ("expand lineage, not context"), so each gets a type.
#: Verified live — register_edge_type accepts arbitrary names and gph_traverse_bounded
#: filters on a single type id.
#:
#: The `eg_hier*` rollups exist because gph_traverse_bounded takes ONE type id, so
#: Task -> Session -> Node in a single bounded BFS needs the two hierarchy relations to
#: share a type. Every hierarchy edge is therefore written twice: under its specific
#: type and under the rollup. The rollup copy is a physical duplicate, not a second
#: logical edge, and the load receipt counts them apart.
EG_EDGE_TYPES: tuple[str, ...] = (
    "eg_has_session",   # task    -> session
    "eg_has_node",      # session -> node
    "eg_hier",          # rollup of the two above, for one-call depth-2 BFS
    "eg_lineage",       # parent  -> child   (the causal edit)
    "eg_context",       # context -> consumer (inspiration; NOT causal parenthood)
    "eg_has_prompt",    # node    -> prompt
    # Inverses. graph_store supports OUTGOING traversal only — direction=in raises
    # "only GRAPH_SCAN_OUTGOING is supported" — so ancestors and "which session owns
    # this node" are reachable ONLY through edges we materialize ourselves. Flagged
    # derived_inverse=true and excluded from logical edge counts.
    "eg_child_of",      # child   -> parent
    "eg_hier_inv",      # node    -> session -> task
    # Cross-modal path statistic for the optimizer experiment.  For each node seed,
    # one-hop adjacency contains its depth-3 lineage reach ordered by
    # (fitness DESC, uid).  It is a native graph synopsis, never a relational join.
    "eg_lineage_h3_fit_v1",
)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

NORMALIZER_VERSION = "evotrace-normalize-v1"


class EgError(RuntimeError):
    """A load-time invariant was violated. Always fail closed."""


def vec_literal(embedding: Sequence[float]) -> str:
    """pgvector's text input form."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


@dataclass
class LoadReceipt:
    """What a load actually did. `added_*` being zero is how idempotence is proven."""

    scope_id: str
    dataset_revision: str
    counts: dict[str, int] = field(default_factory=dict)
    rejects: dict[str, int] = field(default_factory=dict)
    load_id: int | None = None

    def add(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def reject(self, key: str, n: int = 1) -> None:
        self.rejects[key] = self.rejects.get(key, 0) + n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class EgStore:
    """Owns the connection, the schema, the edge-type registry and the vid allocator."""

    def __init__(self, conn: Any, *, dim: int = DEFAULT_DIM) -> None:
        self.conn = conn
        self.dim = dim
        self._edge_types: dict[str, int] = {}

    @classmethod
    def connect(cls, dsn: str = DEFAULT_DSN, *, dim: int = DEFAULT_DIM) -> EgStore:
        import psycopg

        return cls(psycopg.connect(dsn, autocommit=False), dim=dim)

    def close(self) -> None:
        self.conn.close()

    # -- schema ----------------------------------------------------------

    def init_schema(self) -> dict[str, Any]:
        """Apply ``schema.sql`` idempotently and register the EG edge types."""
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS graph_store_am")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS tjs_pg")

        row = self.conn.execute(
            "SELECT atttypmod FROM pg_attribute"
            " WHERE attrelid = to_regclass('gem_eg_vertex') AND attname = 'embedding'"
        ).fetchone()
        if row is not None and row[0] not in (-1, self.dim):
            raise EgError(
                f"existing gem_eg_vertex.embedding is vector({row[0]}), configured "
                f"dimension is {self.dim}; use a separate database rather than "
                "silently mixing embedding spaces"
            )

        ddl = _SCHEMA_PATH.read_text(encoding="utf-8").replace(":dim", str(self.dim))
        self.conn.execute(ddl)
        types = self.bootstrap_edge_types()
        self.conn.commit()
        return {"ok": True, "dim": self.dim, "edge_types": types}

    def bootstrap_edge_types(self) -> dict[str, int]:
        for name in EG_EDGE_TYPES:
            row = self.conn.execute(
                "SELECT graph_store.register_edge_type(%s)", (name,)
            ).fetchone()
            self._edge_types[name] = int(row[0])
        return dict(self._edge_types)

    def edge_type_id(self, name: str) -> int:
        if name not in EG_EDGE_TYPES:
            raise EgError(f"unregistered EG edge type {name!r}; expected one of {EG_EDGE_TYPES}")
        if name not in self._edge_types:
            self.bootstrap_edge_types()
        return self._edge_types[name]

    # -- vid allocation --------------------------------------------------

    def lock_writer(self) -> None:
        self.conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (WRITER_LOCK_KEY,))

    def allocate_vertices(self, count: int) -> list[int]:
        """Reserve ``count`` dense vids in one locked span.

        Allocating in a span rather than per row is the difference between one
        advisory-lock round trip and 10,000 of them, and the vids stay dense because
        the lock is held across the whole span.
        """
        if count <= 0:
            return []
        self.lock_writer()
        start = int(
            self.conn.execute("SELECT graph_store.gph_allocated_vids()").fetchone()[0]
        )
        vids: list[int] = []
        for offset in range(count):
            want = start + offset
            row = self.conn.execute(
                "SELECT graph_store.gph_upsert_vertex(%s)", (want,)
            ).fetchone()
            got = None if row is None else row[0]
            if got is None or int(got) != want:
                raise EgError(
                    f"dense-id drift: reserved {want}, graph returned {got} — a "
                    "concurrent writer violated the single-writer contract"
                )
            vids.append(want)
        return vids

    # -- reads used by the loader and the query layer ---------------------

    def vertex_ids(self, scope_id: str) -> dict[str, int]:
        """uid -> vid for one scope. The idempotence map."""
        rows = self.conn.execute(
            "SELECT uid, id FROM gem_eg_vertex WHERE scope_id = %s", (scope_id,)
        ).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    # -- the graph write --------------------------------------------------

    def link(
        self,
        src: int,
        dst: int,
        *,
        relation: str,
        edge_type: str,
        derived_inverse: bool = False,
        rollup_of: str | None = None,
        provenance: str = "",
        source_row: int | None = None,
        dataset_revision: str | None = None,
    ) -> bool:
        """Write one typed edge. Returns True if it was new.

        The relational mirror is written FIRST and its ``ON CONFLICT DO NOTHING``
        decides whether the AM insert happens. That ordering is what makes a re-load
        add zero adjacency entries: without it the AM would accumulate duplicates the
        mirror could not see.
        """
        type_id = self.edge_type_id(edge_type)
        inserted = self.conn.execute(
            "INSERT INTO gem_eg_edge"
            " (src, dst, edge_type, relation, derived_inverse, rollup_of,"
            "  provenance, source_row, dataset_revision)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (src, dst, edge_type) DO NOTHING"
            " RETURNING 1",
            (
                src,
                dst,
                type_id,
                relation,
                derived_inverse,
                rollup_of,
                provenance,
                source_row,
                dataset_revision,
            ),
        ).fetchone()
        if inserted is None:
            return False
        self.conn.execute(
            "SELECT graph_store.gph_insert_edge(%s, %s, %s)", (src, dst, type_id)
        )
        return True

    def record_load(self, receipt: LoadReceipt) -> int:
        row = self.conn.execute(
            "INSERT INTO gem_eg_load"
            " (scope_id, dataset_revision, normalizer_version, counts, rejects)"
            " VALUES (%s,%s,%s,%s,%s) RETURNING id",
            (
                receipt.scope_id,
                receipt.dataset_revision,
                NORMALIZER_VERSION,
                json.dumps(receipt.counts),
                json.dumps(receipt.rejects),
            ),
        ).fetchone()
        receipt.load_id = int(row[0])
        return receipt.load_id

    # -- convenience ------------------------------------------------------

    def counts(self, scope_id: str) -> dict[str, int]:
        """The numbers a gate compares against the normalizer's integrity report."""
        out: dict[str, int] = {}
        for kind in ("task", "session", "node", "prompt"):
            out[kind] = int(
                self.conn.execute(
                    "SELECT count(*) FROM gem_eg_vertex WHERE scope_id=%s AND kind=%s",
                    (scope_id, kind),
                ).fetchone()[0]
            )
        rows = self.conn.execute(
            "SELECT relation, derived_inverse, count(*) FROM gem_eg_edge e"
            " JOIN gem_eg_vertex v ON v.id = e.src AND v.scope_id = %s"
            " GROUP BY 1,2",
            (scope_id,),
        ).fetchall()
        for relation, inverse, n in rows:
            out[f"edge:{relation}{':inv' if inverse else ''}"] = int(n)
        out["artifacts"] = int(
            self.conn.execute("SELECT count(*) FROM gem_eg_artifact").fetchone()[0]
        )
        out["state_events"] = int(
            self.conn.execute(
                "SELECT count(*) FROM gem_eg_state_event WHERE scope_id=%s", (scope_id,)
            ).fetchone()[0]
        )
        out["am_visible_edges"] = int(
            self.conn.execute("SELECT graph_store.gph_visible_edge_count()").fetchone()[0]
        )
        return out


def batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
