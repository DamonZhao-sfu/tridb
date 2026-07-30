"""Shared TriDB storage backend for long-context memory benchmarks.

This is deliberately a vector-only first adapter. It uses stock TriDB's
PostgreSQL + pgvector surface while preserving benchmark scope, source ids,
timestamps, roles, and metadata. Graph construction is left to a later
adapter so the first LongMemEval/LoCoMo results isolate the storage and
retrieval path from LLM-generated memory semantics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

DEFAULT_DSN = "postgresql://postgres:tridb@localhost:5432/postgres"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384
DEFAULT_TABLE = "agent_memory_units"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _checked_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return value


def _vec_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Batched fastembed wrapper with no GPU requirement."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 64,
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.embed(list(texts), batch_size=self.batch_size)
        return [[float(value) for value in vector] for vector in vectors]


@dataclass(frozen=True)
class MemoryUnit:
    scope_id: str
    external_id: str
    content: str
    session_id: str | None = None
    kind: str = "turn"
    role: str | None = None
    event_time: str | None = None
    event_order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    embedding: Sequence[float] | None = None


@dataclass(frozen=True)
class SearchHit:
    id: int
    scope_id: str
    external_id: str
    session_id: str | None
    kind: str
    role: str | None
    content: str
    event_time: str | None
    event_order: int
    metadata: Mapping[str, Any]
    score: float


class TriDBMemoryBackend:
    """Scoped memory units over a single TriDB/pgvector table."""

    def __init__(
        self,
        conn: Any,
        *,
        dim: int = DEFAULT_DIM,
        table: str = DEFAULT_TABLE,
        embedder: EmbeddingProvider | None = None,
        graph: bool = False,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.conn = conn
        self.dim = dim
        self.table = _checked_identifier(table)
        self.embedder = embedder
        # Graph mode is OPT-IN. With graph=False every method below behaves
        # exactly as before (Paradigm II embedRAG); nothing touches the native
        # AM. With graph=True, ids are allocated from the graph so that
        # memory.id == graph vid, which is what tjs_open's graph leg requires
        # (it fetches reach vids straight back through id_col — src/tjs_pg
        # /tjs_pg.c bridge fetch).
        self.graph = graph

    @classmethod
    def connect(
        cls,
        dsn: str = DEFAULT_DSN,
        *,
        dim: int = DEFAULT_DIM,
        table: str = DEFAULT_TABLE,
        embedder: EmbeddingProvider | None = None,
        graph: bool = False,
    ) -> TriDBMemoryBackend:
        import psycopg

        conn = psycopg.connect(dsn, autocommit=True)
        return cls(conn, dim=dim, table=table, embedder=embedder, graph=graph)

    @property
    def _quoted_table(self) -> str:
        return f'"{self.table}"'

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> dict[str, Any]:
        """Create the benchmark table and cosine HNSW index idempotently."""
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if self.graph:
            # Dependency order matters: tjs_pg's graph leg links against the AM.
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS graph_store_am")
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS tjs_pg")
        row = self.conn.execute(
            "SELECT atttypmod FROM pg_attribute"
            " WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
            (self.table,),
        ).fetchone()
        if row is not None and row[0] != self.dim:
            raise RuntimeError(
                f"existing {self.table}.embedding is vector({row[0]}), configured "
                f"dimension is {self.dim}"
            )

        table = self._quoted_table
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            " id bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            " scope_id text NOT NULL,"
            " external_id text NOT NULL,"
            " session_id text,"
            " kind text NOT NULL,"
            " role text,"
            " content text NOT NULL,"
            " event_time text,"
            " event_order bigint NOT NULL DEFAULT 0,"
            " metadata jsonb NOT NULL DEFAULT '{}'::jsonb,"
            f" embedding vector({self.dim}) NOT NULL,"
            " created_at timestamptz NOT NULL DEFAULT now(),"
            " UNIQUE (scope_id, external_id))"
        )
        self.conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{self.table}_scope_idx" ON {table} (scope_id)'
        )
        self.conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{self.table}_session_idx"'
            f" ON {table} (scope_id, session_id)"
        )
        self.conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{self.table}_embedding_hnsw"'
            f" ON {table} USING hnsw (embedding vector_cosine_ops)"
        )
        return {"ok": True, "table": self.table, "dim": self.dim, "graph": self.graph}

    # ------------------------------------------------------------------
    # Native graph surface (opt-in; requires graph=True)
    #
    # The contract this implements, verified against the engine: tjs_open's
    # graph leg resolves a reach vertex by running
    #   SELECT <vec> FROM <tbl> WHERE <id_col> = $1   with $1 = the raw vid
    # so a memory row is only reachable through the graph when its id EQUALS
    # its graph vid. Ids are therefore allocated from the graph's own dense
    # counter rather than from the identity sequence.
    # ------------------------------------------------------------------

    def _require_graph(self) -> None:
        if not self.graph:
            raise RuntimeError(
                "graph mode is off; construct the backend with graph=True"
            )

    def _allocate_vertex(self) -> int:
        """Reserve the next dense vid and return it as the memory id.

        Uses gph_allocated_vids() (a monotonic allocation counter) rather than
        max(id)+1 over the table: TRUNCATE resets the table but NOT the graph,
        and reusing a vid would silently adopt the previous occupant's edges.
        """
        next_id = self.conn.execute(
            "SELECT graph_store.gph_allocated_vids()"
        ).fetchone()[0]
        vid = self.conn.execute(
            "SELECT graph_store.gph_upsert_vertex(%s)", (next_id,)
        ).fetchone()[0]
        if vid != next_id:
            # Another writer allocated concurrently. The single-writer contract
            # (docs/mcp_agent_memory_v0.1.0.md) is violated; a mis-addressed
            # graph leg is worse than a failed insert.
            raise RuntimeError(
                f"dense-id drift: reserved {next_id}, graph returned {vid} — "
                "single-writer contract violated"
            )
        return int(vid)

    def add_units(
        self,
        scope_id: str,
        units: Sequence[MemoryUnit],
    ) -> list[int]:
        """Append memories, creating one graph vertex per row in ONE txn.

        This is the incremental write path (``replace_scope`` replaces a whole
        corpus). Returns the assigned ids, which are also the graph vids.
        """
        self._require_graph()
        for unit in units:
            if unit.scope_id != scope_id:
                raise ValueError(
                    f"unit {unit.external_id!r} belongs to scope "
                    f"{unit.scope_id!r}, expected {scope_id!r}"
                )
        if not units:
            return []
        vectors = self._resolve_unit_vectors(units)
        table = self._quoted_table
        ids: list[int] = []
        with self.conn.transaction():
            for unit, vector in zip(units, vectors, strict=True):
                memory_id = self._allocate_vertex()
                self.conn.execute(
                    f"INSERT INTO {table} ("
                    " id, scope_id, external_id, session_id, kind, role, content,"
                    " event_time, event_order, metadata, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,"
                    "         %s::vector)",
                    (
                        memory_id,
                        unit.scope_id,
                        unit.external_id,
                        unit.session_id,
                        unit.kind,
                        unit.role,
                        unit.content,
                        unit.event_time,
                        unit.event_order,
                        json.dumps(dict(unit.metadata), sort_keys=True),
                        _vec_literal(vector),
                    ),
                )
                ids.append(memory_id)
        return ids

    def link(self, src_id: int, dst_id: int, rel: str) -> dict[str, Any]:
        """Typed directed edge between two memories. Relation names
        auto-register (graph_store.register_edge_type)."""
        self._require_graph()
        with self.conn.transaction():
            edge_type = self.conn.execute(
                "SELECT graph_store.register_edge_type(%s)", (rel,)
            ).fetchone()[0]
            self.conn.execute(
                "SELECT graph_store.gph_insert_edge(%s, %s, %s)",
                (int(src_id), int(dst_id), int(edge_type)),
            )
        return {"src": int(src_id), "dst": int(dst_id), "rel": rel, "type": edge_type}

    def neighbors(
        self,
        memory_id: int,
        *,
        rel: str | None = None,
        hops: int = 1,
    ) -> list[SearchHit]:
        """Direct native-graph read: typed 1-hop out-neighbours, or the
        multi-hop reach set."""
        self._require_graph()
        if hops < 1:
            raise ValueError("hops must be >= 1")
        if rel is None:
            type_id = 0  # GPH_EDGE_TYPE_ANY
        else:
            row = self.conn.execute(
                "SELECT id FROM graph_store.edge_type WHERE name = %s", (rel,)
            ).fetchone()
            if row is None:
                return []
            type_id = int(row[0])
        if hops == 1:
            # Target-list (ProjectSet) position, per the AM's contract: a
            # FROM-clause FunctionScan loses early termination under LIMIT.
            # Args are (src, type_id, direction=0 out, source_id=-1 unscoped);
            # direction in/both RAISEs until reverse adjacency lands (ADR-0016).
            rows = self.conn.execute(
                "SELECT (e).dst FROM (SELECT graph_store.gph_traverse_typed("
                "%s, %s, 0, -1) AS e) s",
                (int(memory_id), type_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT graph_store.gph_traverse_bfs(%s, %s, %s)",
                (int(memory_id), int(hops), type_id),  # (seed, max_depth, type_id)
            ).fetchall()
        ids = [int(row[0]) for row in rows if row[0] is not None]
        return self._hits_by_id(ids)

    def _hits_by_id(self, ids: Sequence[int]) -> list[SearchHit]:
        if not ids:
            return []
        rows = self.conn.execute(
            "SELECT id, scope_id, external_id, session_id, kind, role, content,"
            f" event_time, event_order, metadata FROM {self._quoted_table}"
            " WHERE id = ANY(%s)",
            (list(ids),),
        ).fetchall()
        by_id = {int(row[0]): row for row in rows}
        hits: list[SearchHit] = []
        for memory_id in ids:
            row = by_id.get(int(memory_id))
            if row is None:
                continue  # a vertex with no surviving memory row
            metadata = row[9]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            hits.append(
                SearchHit(
                    id=row[0],
                    scope_id=row[1],
                    external_id=row[2],
                    session_id=row[3],
                    kind=row[4],
                    role=row[5],
                    content=row[6],
                    event_time=row[7],
                    event_order=row[8],
                    metadata=metadata,
                    score=float("nan"),
                )
            )
        return hits

    def search_fused(
        self,
        scope_id: str,
        *,
        query_text: str | None = None,
        query_embedding: Sequence[float] | None = None,
        k: int = 10,
        m_seeds: int = 4,
        hops: int = 1,
        term_cond: int = 32,
        anchor_id: int | None = None,
        extra_filter: str | None = None,
    ) -> dict[str, Any]:
        """Fused vector + native-graph + relational retrieval via tjs_open.

        The relational predicate (scope, and anything in ``extra_filter``) is
        pushed INTO the operator, not applied afterwards, so the graph leg and
        the vector leg both see it. Returns the hits plus the engine's honesty
        probes — a censored or right-censored run is a different operating
        point, not a faster exact one, so the caller must record them.
        """
        self._require_graph()
        if k <= 0:
            raise ValueError("k must be positive")
        vector = self._resolve_query_vector(query_text, query_embedding)

        from psycopg import sql as _sql

        predicate = _sql.SQL("scope_id = {}").format(_sql.Literal(scope_id))
        if extra_filter:
            predicate = _sql.SQL("({}) AND ({})").format(
                predicate, _sql.SQL(extra_filter)
            )
        filter_text = predicate.as_string(self.conn)

        # The vector-first fused path REQUIRES relaxed_order (the engine
        # refuses strict_order outright). This is a different operating point
        # from search()'s strict_order scan and must be reported as such.
        self.conn.execute("SET hnsw.iterative_scan = relaxed_order")
        rows = self.conn.execute(
            f"SELECT t FROM tjs_open('{self.table}', %s, %s, %s, %s, 'id', %s,"
            " %s::vector, %s) AS t",
            (
                int(k),
                int(term_cond),
                int(m_seeds),
                int(hops),
                filter_text,
                _vec_literal(vector),
                None if anchor_id is None else int(anchor_id),
            ),
        ).fetchall()
        ids = [int(row[0]) for row in rows]
        # Probes describe the LAST call and must be read on THIS connection,
        # before anything else runs on it.
        probes = self.conn.execute(
            "SELECT tjs_open_candidates_examined(), tjs_open_graph_examined(),"
            " tjs_open_graph_censored(), tjs_open_termination_reason(),"
            " tjs_open_budget_capped(), tjs_open_bridges_injected()"
        ).fetchone()
        return {
            "ids": ids,
            "hits": self._hits_by_id(ids),
            "probes": {
                "candidates_examined": probes[0],
                "graph_examined": probes[1],
                "graph_censored": probes[2],
                "termination_reason": probes[3],
                "budget_capped": probes[4],
                "bridges_injected": probes[5],
            },
            "params": {
                "k": k,
                "m_seeds": m_seeds,
                "hops": hops,
                "term_cond": term_cond,
                "anchor_id": anchor_id,
                "filter": filter_text,
                "hnsw_iterative_scan": "relaxed_order",
            },
        }

    def graph_stats(self) -> dict[str, Any]:
        self._require_graph()
        row = self.conn.execute(
            "SELECT graph_store.gph_vertex_count(),"
            " graph_store.gph_visible_edge_count(),"
            " graph_store.gph_visits(), graph_store.gph_page_reads()"
        ).fetchone()
        types = self.conn.execute(
            "SELECT name FROM graph_store.edge_type ORDER BY id"
        ).fetchall()
        return {
            "vertices": row[0],
            "visible_edges": row[1],
            "visits": row[2],
            "page_reads": row[3],
            "edge_types": [t[0] for t in types],
        }

    def _validate_vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self.dim:
            raise ValueError(
                f"embedding has {len(values)} dimensions, expected {self.dim}"
            )
        return values

    def _resolve_query_vector(
        self,
        query_text: str | None,
        query_embedding: Sequence[float] | None,
    ) -> list[float]:
        if query_embedding is not None:
            return self._validate_vector(query_embedding)
        if query_text is None:
            raise ValueError("query_text or query_embedding is required")
        if self.embedder is None:
            raise RuntimeError("no embedder configured for query_text")
        encoded = self.embedder.encode([query_text])
        if len(encoded) != 1:
            raise RuntimeError("embedder did not return exactly one query vector")
        return self._validate_vector(encoded[0])

    def _resolve_unit_vectors(self, units: Sequence[MemoryUnit]) -> list[list[float]]:
        vectors: list[list[float] | None] = []
        missing_positions: list[int] = []
        missing_texts: list[str] = []
        for position, unit in enumerate(units):
            if unit.embedding is None:
                vectors.append(None)
                missing_positions.append(position)
                missing_texts.append(unit.content)
            else:
                vectors.append(self._validate_vector(unit.embedding))

        if missing_positions:
            if self.embedder is None:
                raise RuntimeError(
                    "no embedder configured; provide MemoryUnit.embedding values"
                )
            generated = self.embedder.encode(missing_texts)
            if len(generated) != len(missing_positions):
                raise RuntimeError(
                    "embedder returned a different number of vectors than inputs"
                )
            for position, vector in zip(missing_positions, generated, strict=True):
                vectors[position] = self._validate_vector(vector)

        return [vector for vector in vectors if vector is not None]

    def replace_scope(
        self,
        scope_id: str,
        units: Sequence[MemoryUnit],
        *,
        isolated: bool = True,
    ) -> int:
        """Atomically replace one corpus.

        ``isolated=True`` truncates this adapter's dedicated benchmark table.
        That matches LongMemEval's per-question corpus and LoCoMo's
        per-conversation corpus without HNSW tombstone accumulation.
        ``isolated=False`` preserves other scopes to exercise filtered ANN.

        In graph mode the rows are written through the vertex-allocating path
        so that id == vid holds for every memory. TRUNCATE does not reset the
        graph, so the replaced corpus gets FRESH vids and never inherits the
        previous occupant's edges; the orphaned vertices remain allocated and
        keep gph_vertex_count() above the live row count (expected, and why
        graph runs should report visible_edges alongside it).
        """
        for unit in units:
            if unit.scope_id != scope_id:
                raise ValueError(
                    f"unit {unit.external_id!r} belongs to scope "
                    f"{unit.scope_id!r}, expected {scope_id!r}"
                )
        if self.graph:
            with self.conn.transaction():
                if isolated:
                    self.conn.execute(f"TRUNCATE TABLE {self._quoted_table}")
                else:
                    self.conn.execute(
                        f"DELETE FROM {self._quoted_table} WHERE scope_id = %s",
                        (scope_id,),
                    )
                return len(self.add_units(scope_id, units))
        vectors = self._resolve_unit_vectors(units)
        table = self._quoted_table
        rows = [
            (
                unit.scope_id,
                unit.external_id,
                unit.session_id,
                unit.kind,
                unit.role,
                unit.content,
                unit.event_time,
                unit.event_order,
                json.dumps(dict(unit.metadata), sort_keys=True),
                _vec_literal(vector),
            )
            for unit, vector in zip(units, vectors, strict=True)
        ]

        with self.conn.transaction():
            if isolated:
                self.conn.execute(f"TRUNCATE TABLE {table}")
            else:
                self.conn.execute(
                    f"DELETE FROM {table} WHERE scope_id = %s", (scope_id,)
                )
            if rows:
                with self.conn.cursor() as cursor:
                    cursor.executemany(
                        f"INSERT INTO {table} ("
                        " scope_id, external_id, session_id, kind, role, content,"
                        " event_time, event_order, metadata, embedding)"
                        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s,"
                        "         %s::jsonb, %s::vector)"
                        " ON CONFLICT (scope_id, external_id) DO UPDATE SET"
                        " session_id = EXCLUDED.session_id,"
                        " kind = EXCLUDED.kind,"
                        " role = EXCLUDED.role,"
                        " content = EXCLUDED.content,"
                        " event_time = EXCLUDED.event_time,"
                        " event_order = EXCLUDED.event_order,"
                        " metadata = EXCLUDED.metadata,"
                        " embedding = EXCLUDED.embedding",
                        rows,
                    )
        return len(rows)

    def delete_scope(self, scope_id: str) -> int:
        cursor = self.conn.execute(
            f"DELETE FROM {self._quoted_table} WHERE scope_id = %s",
            (scope_id,),
        )
        return cursor.rowcount

    def search(
        self,
        scope_id: str,
        *,
        query_text: str | None = None,
        query_embedding: Sequence[float] | None = None,
        k: int = 10,
    ) -> list[SearchHit]:
        if k <= 0:
            raise ValueError("k must be positive")
        vector = self._resolve_query_vector(query_text, query_embedding)
        literal = _vec_literal(vector)

        # Scope is a correctness predicate. Iterative scan prevents selective
        # post-filtering from silently returning fewer than k rows.
        self.conn.execute("SET hnsw.iterative_scan = strict_order")
        rows = self.conn.execute(
            "SELECT id, scope_id, external_id, session_id, kind, role, content,"
            " event_time, event_order, metadata,"
            f" embedding <=> %s::vector AS distance FROM {self._quoted_table}"
            " WHERE scope_id = %s"
            " ORDER BY embedding <=> %s::vector LIMIT %s",
            (literal, scope_id, literal, k),
        ).fetchall()
        hits = []
        for row in rows:
            metadata = row[9]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            hits.append(
                SearchHit(
                    id=row[0],
                    scope_id=row[1],
                    external_id=row[2],
                    session_id=row[3],
                    kind=row[4],
                    role=row[5],
                    content=row[6],
                    event_time=row[7],
                    event_order=row[8],
                    metadata=metadata,
                    score=1.0 - float(row[10]),
                )
            )
        return hits

    def count_scope(self, scope_id: str) -> int:
        return self.conn.execute(
            f"SELECT count(*) FROM {self._quoted_table} WHERE scope_id = %s",
            (scope_id,),
        ).fetchone()[0]
