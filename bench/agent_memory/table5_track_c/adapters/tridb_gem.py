"""TriDB/GEM Track C adapter.

Construction keeps one provenance-addressable semantic unit per LoCoMo turn
and adds chronological association edges between adjacent turns in a session.
Those edges are stored through the native graph AM; ``gem_edge`` contains only
edge metadata.  Retrieval always uses GEM's canonical FUSED ``tjs_open`` path.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.strategies.deterministic import (
    DeterministicIngestStrategy,
)
from bench.agent_memory.gem.types import (
    EdgeKind,
    InteractionEvent,
    Query,
    RetrievalMode,
    RetrievalRoute,
)
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)

from ..dataset import EventItem, QueryItem
from ..tracing import stage_span


@dataclass(frozen=True)
class TriDBGEMConfig:
    dsn: str
    embedding_base_url: str = "http://127.0.0.1:8001/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    embedding_batch_size: int = 64
    model_timeout_seconds: float = 120.0
    reinforce: bool = True
    m_seeds: int = 4
    hops: int = 1
    term_cond: int = 128


def _event(item: EventItem) -> InteractionEvent:
    return InteractionEvent(
        scope_id=item.sample_id,
        external_id=item.event_id,
        content=item.text,
        session_id=item.session_id,
        role=item.role,
        kind="turn",
        event_time=item.timestamp or None,
        event_order=item.ordinal,
        metadata=item.metadata,
    )


class _ThreadContext:
    def __init__(self, config: TriDBGEMConfig) -> None:
        self.ledger = CallLedger()
        client = OpenAIEmbeddingClient(
            config.embedding_base_url,
            "EMPTY",
            config.embedding_model,
            batch_size=config.embedding_batch_size,
            timeout=config.model_timeout_seconds,
            ledger=self.ledger,
        )
        self.embedder = PhasedEmbedder(client)
        self.memory = TriDBGovernedMemory.connect(
            config.dsn, dim=config.embedding_dim, embedder=self.embedder
        )

    def close(self) -> None:
        self.memory.close()


class TriDBGEMAdapter:
    """Native graph + vector + relational GEM adapter with per-thread DB sessions."""

    name = "tridb_gem"

    def __init__(self, config: TriDBGEMConfig) -> None:
        self.config = config
        self._local = threading.local()
        self._contexts: list[_ThreadContext] = []
        self._contexts_lock = threading.Lock()
        # v1 graph allocation is single-writer.  Waiting on this lock remains
        # inside adapter service time, so an overloaded Add path is not hidden.
        self._writer_lock = threading.Lock()

    def _context(self) -> _ThreadContext:
        context = getattr(self._local, "context", None)
        if context is None:
            context = _ThreadContext(self.config)
            self._local.context = context
            with self._contexts_lock:
                self._contexts.append(context)
        return context

    def init_schema(self) -> dict[str, Any]:
        result = self._context().memory.init_schema()
        advertised = self._context().embedder.client.discover_models()
        if advertised != [self.config.embedding_model]:
            raise RuntimeError(
                "embedding endpoint identity mismatch: "
                f"expected {[self.config.embedding_model]!r}, got {advertised!r}"
            )
        vector = self._context().embedder.encode(["dimension probe"])[0]
        if len(vector) != self.config.embedding_dim:
            raise RuntimeError(
                f"embedding dimension mismatch: {len(vector)} != "
                f"{self.config.embedding_dim}"
            )
        return result

    def ingest_history(self, events: Sequence[EventItem]) -> dict[str, Any]:
        """Build all scopes, then add native chronological association edges."""
        started = time.perf_counter()
        by_scope: dict[str, list[EventItem]] = {}
        for item in events:
            by_scope.setdefault(item.sample_id, []).append(item)

        created = 0
        edge_count = 0
        rejected: list[dict[str, Any]] = []
        context = self._context()
        strategy = DeterministicIngestStrategy(chunk_tokens=4096, batch_size=64)
        with self._writer_lock, context.embedder.phase("construction"):
            for scope_id, scope_events in by_scope.items():
                result = context.memory.ingest(
                    [_event(item) for item in scope_events],
                    strategy=strategy,
                    scope_id=scope_id,
                )
                if not result.committed:
                    raise RuntimeError(
                        f"TriDB/GEM build failed for {scope_id}: {result.aborted_reason}"
                    )
                created += result.delta.units_created
                rejected.extend(dict(item) for item in result.rejected)
                edge_count += self._link_session_chains(context, scope_id)
        return {
            "wall_seconds": time.perf_counter() - started,
            "events": len(events),
            "units_created": created,
            "edges_created": edge_count,
            "rejected": rejected,
            "model_calls": context.ledger.summary(),
            "tokens": context.ledger.tokens(),
        }

    def _link_session_chains(self, context: _ThreadContext, scope_id: str) -> int:
        rows = context.memory.store.conn.execute(
            "SELECT id, metadata->>'session_id',"
            " COALESCE((metadata->>'event_order')::bigint, 0)"
            " FROM gem_unit WHERE scope_id = %s ORDER BY 2, 3, id",
            (scope_id,),
        ).fetchall()
        # event_order is a top-level metadata field from DeterministicIngest.
        # Older rows without it retain id order as the deterministic fallback.
        by_session: dict[str, list[int]] = {}
        for unit_id, session_id, _ in rows:
            by_session.setdefault(str(session_id), []).append(int(unit_id))
        pairs = [
            (left, right)
            for ids in by_session.values()
            for left, right in zip(ids, ids[1:])
        ]
        if not pairs:
            return 0
        with context.memory.store.transition("ingest", scope_id, "construction") as tx:
            for left, right in pairs:
                context.memory.store.link(
                    tx,
                    left,
                    right,
                    kind=EdgeKind.ASSOCIATION,
                    rel="next_turn",
                )
                context.memory.store.link(
                    tx,
                    right,
                    left,
                    kind=EdgeKind.ASSOCIATION,
                    rel="previous_turn",
                )
        return len(pairs) * 2

    def finalize_build(self) -> dict[str, Any]:
        context = self._context()
        context.memory.store.conn.execute("ANALYZE gem_unit")
        row = context.memory.store.conn.execute(
            "SELECT count(*), pg_total_relation_size('gem_unit'),"
            " pg_database_size(current_database()) FROM gem_unit"
        ).fetchone()
        edge_row = context.memory.store.conn.execute(
            "SELECT count(*) FROM gem_edge WHERE tombstoned_at IS NULL"
        ).fetchone()
        return {
            "units": int(row[0]),
            "gem_unit_bytes": int(row[1]),
            "database_bytes": int(row[2]),
            "visible_edges": int(edge_row[0]),
        }

    def search(self, item: QueryItem, *, top_k: int = 35) -> dict[str, Any]:
        context = self._context()
        before = len(context.ledger.records)
        with stage_span(
            "framework_other", "tridb_gem.retrieve", backend="tridb_postgresql"
        ):
            result = context.memory.retrieve(
                Query(
                    scope_id=item.sample_id,
                    text=item.question,
                    k=top_k,
                    mode=RetrievalMode.FUSED,
                    route=RetrievalRoute.TOPIC,
                    reinforce=self.config.reinforce,
                    m_seeds=self.config.m_seeds,
                    hops=self.config.hops,
                    term_cond=self.config.term_cond,
                )
            )
        if not result.committed:
            raise RuntimeError(result.aborted_reason or "TriDB/GEM retrieve aborted")
        unit_ids = list(dict.fromkeys(hit.unit_id for hit in result.hits))
        external_ids = self._external_ids(context, unit_ids)
        ledger_records = context.ledger.records[before:]
        return {
            "result_count": len(unit_ids),
            "hit_ids": external_ids,
            "unit_ids": unit_ids,
            "contexts": [hit.value or hit.title for hit in result.hits],
            "context_tokens": None,
            "empty": not unit_ids,
            "cost": asdict(result.cost),
            "probes": dict(result.probes),
            "model_calls": ledger_records,
        }

    @staticmethod
    def _external_ids(context: _ThreadContext, unit_ids: Sequence[int]) -> list[str]:
        if not unit_ids:
            return []
        with stage_span(
            "relational",
            "tridb.external_id_lookup",
            backend="postgresql",
            attributes={"observable_call_kind": "database_client"},
        ):
            rows = context.memory.store.conn.execute(
                "SELECT id, metadata->>'external_id' FROM gem_unit WHERE id = ANY(%s)",
                (list(unit_ids),),
            ).fetchall()
        by_id = {int(row[0]): str(row[1]) for row in rows if row[1] is not None}
        return [by_id[unit_id] for unit_id in unit_ids if unit_id in by_id]

    def add(
        self,
        item: EventItem,
        *,
        visibility: bool,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        context = self._context()
        strategy = DeterministicIngestStrategy(chunk_tokens=4096, batch_size=64)
        with self._writer_lock, context.embedder.phase("construction"):
            result = context.memory.ingest(
                [_event(item)], strategy=strategy, scope_id=item.sample_id
            )
            if not result.committed:
                raise RuntimeError(result.aborted_reason or "TriDB/GEM ingest aborted")
            unit_ids = [int(value) for value in result.units]
            with stage_span(
                "graph", "tridb.link_admission_edges", backend="graph_store_am"
            ):
                linked_edges = self._link_new_unit(context, item.sample_id, unit_ids)
        committed_at_ns = time.perf_counter_ns()
        if progress is not None:
            progress({"commit_observed": True, "committed_at_ns": committed_at_ns})
        visible = None
        visibility_error = None
        searchable_at_ns = None
        if visibility:
            if progress is not None:
                progress({"visibility_probe_started": True})
            try:
                visible = self.visibility_probe(item)
            except Exception as exc:  # noqa: BLE001 - retained as measured evidence
                visible = False
                visibility_error = f"{type(exc).__name__}: {exc}"
            searchable_at_ns = time.perf_counter_ns() if visible else None
        receipt = {
            "committed_at_ns": committed_at_ns,
            "searchable_at_ns": searchable_at_ns,
            "visibility_probe": visible,
            "visibility_error": visibility_error,
            "units_created": result.delta.units_created,
            "edges_created": result.delta.edges_created + linked_edges,
            "created_memory_count": result.delta.units_created,
            "created_node_count": result.delta.units_created,
            "created_edge_count": result.delta.edges_created + linked_edges,
            "creation_counts_available": True,
            "creation_count_source": "tridb_ingest_delta_and_admission_links",
            "cost": asdict(result.cost),
            "rejected": [dict(item) for item in result.rejected],
        }
        if progress is not None:
            progress(receipt)
        return receipt

    def _link_new_unit(
        self, context: _ThreadContext, scope_id: str, unit_ids: Sequence[int]
    ) -> int:
        if not unit_ids:
            return 0
        unit_id = int(unit_ids[-1])
        with stage_span(
            "relational",
            "tridb.previous_admission_lookup",
            backend="postgresql",
            attributes={"observable_call_kind": "database_client"},
        ):
            row = context.memory.store.conn.execute(
                "SELECT id FROM gem_unit WHERE scope_id = %s AND id < %s"
                " ORDER BY id DESC LIMIT 1",
                (scope_id, unit_id),
            ).fetchone()
        if row is None:
            return 0
        previous = int(row[0])
        with context.memory.store.transition("ingest", scope_id, "construction") as tx:
            context.memory.store.link(
                tx,
                previous,
                unit_id,
                kind=EdgeKind.ASSOCIATION,
                rel="prior_admission",
            )
            context.memory.store.link(
                tx,
                unit_id,
                previous,
                kind=EdgeKind.ASSOCIATION,
                rel="next_admission",
            )
        return 2

    def visibility_probe(self, item: EventItem) -> bool:
        probe = QueryItem(
            sample_id=item.sample_id,
            question_id=f"probe:{item.event_id}",
            question=item.text,
            answer="",
            category="probe",
            evidence_ids=(item.event_id,),
            ordinal=0,
        )
        result = self.search(probe, top_k=35)
        return item.event_id in result["hit_ids"]

    def stats(self) -> dict[str, Any]:
        context = self._context()
        row = context.memory.store.conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE state='active') FROM gem_unit"
        ).fetchone()
        edges = context.memory.store.conn.execute(
            "SELECT count(*) FROM gem_edge WHERE tombstoned_at IS NULL"
        ).fetchone()[0]
        graph = context.memory.store.conn.execute(
            "SELECT graph_store.gph_vertex_count(),"
            " graph_store.gph_visible_edge_count()"
        ).fetchone()
        return {
            "units": int(row[0]),
            "active_units": int(row[1]),
            "edge_metadata_rows": int(edges),
            "native_vertices": int(graph[0]),
            "native_visible_edges": int(graph[1]),
            "manifest": context.memory.manifest(
                strategy_name="deterministic_turn_graph",
                mode="FUSED",
                route="TOPIC",
                reinforce=self.config.reinforce,
                revise_enabled=False,
                forget_enabled=False,
                embedding_model=self.config.embedding_model,
            ),
            "ledger": context.ledger.summary(),
            "tokens": context.ledger.tokens(),
        }

    def snapshot_fingerprint(self) -> dict[str, Any]:
        stats = self.stats()
        return {
            key: stats[key]
            for key in (
                "units",
                "active_units",
                "edge_metadata_rows",
                "native_vertices",
                "native_visible_edges",
            )
        }

    def close(self) -> None:
        with self._contexts_lock:
            contexts = list(self._contexts)
            self._contexts.clear()
        for context in contexts:
            context.close()

    def dump_config(self) -> str:
        return json.dumps(asdict(self.config), sort_keys=True)
