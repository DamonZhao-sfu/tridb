"""Live MemoryArena backend over GEM's single-process TriDB operators."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.strategies.deterministic import (
    DeterministicIngestStrategy,
)
from bench.agent_memory.gem.store import vec_literal
from bench.agent_memory.gem.types import EdgeKind, InteractionEvent
from bench.agent_memory.memoryarena.cross_session import BackendSearch, RetrievedItem
from bench.agent_memory.memoryarena.dataset import MemoryArenaSession, MemoryArenaTask
from bench.agent_memory.memoryarena.oracle import (
    Arm,
    Candidate,
    DecisionPoint,
    candidates_for_task,
)
from bench.agent_memory.memoryarena.receipts import RetrievalReceipt


class WholeEventChunker:
    """One admission unit per session for session-level benchmark receipts."""

    @staticmethod
    def chunk(text: str) -> list[str]:
        return [text]


class TriDBMemoryArenaBackend:
    """Map six benchmark arms onto GEM while preserving one DB/WAL boundary.

    Read-only benchmark retrieval uses a named server cursor with ``itersize=1`` and
    closes as soon as top-k is reached.  This preserves tjs_open's target-list
    Open/Next/Close path and makes ``first_row_ms`` a real client-observed TTFR.
    """

    def __init__(
        self,
        memory: TriDBGovernedMemory,
        task: MemoryArenaTask,
        *,
        namespace: str,
        reinforce: bool = False,
        write_enabled: bool = True,
        m_seeds: int = 4,
        hops: int = 1,
        term_cond: int = 128,
    ) -> None:
        if reinforce:
            raise ValueError(
                "streaming benchmark backend currently requires reinforce=False; "
                "C6 reinforcement needs its own transactional operating point"
            )
        self.memory = memory
        self.task = task
        self.scope_id = f"{namespace}:{task.task_uid}"
        self.m_seeds = m_seeds
        self.hops = hops
        self.term_cond = term_cond
        self.write_enabled = write_enabled
        self.strategy = DeterministicIngestStrategy(chunker=WholeEventChunker())

    def _snapshot_id(self) -> str:
        rows = self.memory.store.conn.execute(
            "SELECT id, title, state, access_count, metadata::text"
            " FROM gem_unit WHERE scope_id=%s ORDER BY id",
            (self.scope_id,),
        ).fetchall()
        payload = json.dumps(rows, default=str, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def _all_candidates(
        self,
        decision: DecisionPoint,
        *,
        semantic_scores: Mapping[str, float] | None = None,
        semantic_score_source: str = "not_computed_for_arm",
    ) -> tuple[Candidate, ...]:
        semantic_scores = semantic_scores or {}
        return candidates_for_task(
            self.task,
            semantic_scores=semantic_scores,
            semantic_score_sources={
                session_uid: semantic_score_source for session_uid in semantic_scores
            },
            graph_distances={
                session.session_uid: max(0, decision.cutoff_ordinal - session.ordinal)
                for session in self.task.sessions
            },
        )

    def _cutoff_filter(self, decision: DecisionPoint) -> str:
        from psycopg import sql

        # Both values are rendered by psycopg's literal adapter.  The predicate is
        # pushed inside tjs_open so vector and graph candidates see the same cutoff.
        predicate = sql.SQL(
            "scope_id={} AND state='active' AND "
            "(metadata->>'session_ordinal')::integer < {}"
        ).format(sql.Literal(self.scope_id), sql.Literal(decision.cutoff_ordinal))
        return predicate.as_string(self.memory.store.conn)

    def _direct_ids(self, decision: DecisionPoint, *, recent: bool) -> list[int]:
        order = "DESC" if recent else "ASC"
        rows = self.memory.store.conn.execute(
            "SELECT id FROM gem_unit WHERE scope_id=%s AND state='active'"
            " AND (metadata->>'session_ordinal')::integer < %s"
            f" ORDER BY (metadata->>'session_ordinal')::integer {order}, id {order}"
            " LIMIT %s",
            (self.scope_id, decision.cutoff_ordinal, 10_000),
        ).fetchall()
        return [int(row[0]) for row in rows]

    def _anchor_id(self, decision: DecisionPoint) -> int | None:
        ids = self._direct_ids(decision, recent=True)
        return ids[0] if ids else None

    def _stream_ids(
        self,
        sql_text: str,
        params: Sequence[Any],
        *,
        top_k: int,
        started: float,
        setup: Sequence[tuple[str, str]] = (),
        read_tjs_probes: bool = False,
    ) -> tuple[list[int], float | None, dict[str, Any]]:
        ids: list[int] = []
        first_row_ms: float | None = None
        probes: dict[str, Any] = {}
        conn = self.memory.store.conn
        with conn.transaction():
            for name, value in setup:
                conn.execute("SELECT set_config(%s, %s, true)", (name, value))
            cursor_name = f"memoryarena_{time.perf_counter_ns()}"
            with conn.cursor(name=cursor_name) as cursor:
                cursor.itersize = 1
                cursor.execute(sql_text, tuple(params))
                for row in cursor:
                    if row[0] is None:
                        continue
                    if first_row_ms is None:
                        first_row_ms = (time.perf_counter() - started) * 1000.0
                    ids.append(int(row[0]))
                    if len(ids) >= top_k:
                        break
            if read_tjs_probes:
                row = conn.execute(
                    "SELECT tjs_open_candidates_examined(),"
                    " tjs_open_graph_examined(), tjs_open_graph_censored(),"
                    " tjs_open_termination_reason(), tjs_open_budget_capped(),"
                    " tjs_open_bridges_injected()"
                ).fetchone()
                probes = {
                    "candidates_examined": row[0],
                    "graph_examined": row[1],
                    "graph_censored": row[2],
                    "termination_reason": row[3],
                    "budget_capped": row[4],
                    "bridges_injected": row[5],
                }
        return ids, first_row_ms, probes

    def _retrieve_ids(
        self,
        arm: Arm,
        decision: DecisionPoint,
        top_k: int,
        *,
        started: float,
    ) -> tuple[list[int], Mapping[str, Any], float | None, Sequence[float] | None]:
        if arm is Arm.MEMORY_OFF:
            return [], {"mode": "off", "termination_reason": "memory_off"}, None, None

        if arm in (Arm.RECENT_FIFO, Arm.ORACLE):
            order = "DESC" if arm is Arm.RECENT_FIFO else "ASC"
            ids, first, _ = self._stream_ids(
                "SELECT id FROM gem_unit WHERE scope_id=%s AND state='active'"
                " AND (metadata->>'session_ordinal')::integer < %s"
                f" ORDER BY (metadata->>'session_ordinal')::integer {order},"
                f" id {order} LIMIT %s",
                (self.scope_id, decision.cutoff_ordinal, top_k),
                top_k=top_k,
                started=started,
            )
            mode = "recent_fifo" if arm is Arm.RECENT_FIFO else "protocol_prefix_oracle"
            return ids, {"mode": mode, "termination_reason": "limit"}, first, None

        if arm is Arm.GRAPH_RELATIONAL:
            anchor = self._anchor_id(decision)
            if anchor is None:
                return (
                    [],
                    {
                        "mode": "graph_relational",
                        "termination_reason": "empty_snapshot",
                    },
                    None,
                    None,
                )
            type_id = self.memory.store.edge_type_id(EdgeKind.ASSOCIATION)
            # Topology comes only from the native AM.  The join resolves reached vids
            # to vertex properties for the hard scope/cutoff filter; it is not an edge
            # join and remains bounded by LIMIT/early cursor close.
            ids, first, _ = self._stream_ids(
                "SELECT reached.dst FROM ("
                " SELECT (e).dst::bigint AS dst FROM ("
                "  SELECT graph_store.gph_traverse_typed(%s,%s,0,-1) AS e"
                " ) traversed"
                ") reached JOIN gem_unit u ON u.id=reached.dst"
                " WHERE u.scope_id=%s AND u.state='active'"
                " AND (u.metadata->>'session_ordinal')::integer < %s LIMIT %s",
                (
                    anchor,
                    type_id,
                    self.scope_id,
                    decision.cutoff_ordinal,
                    top_k,
                ),
                top_k=top_k,
                started=started,
            )
            return (
                ids,
                {"mode": "graph_relational", "termination_reason": "limit"},
                first,
                None,
            )

        vector = self.memory.embedder.encode([decision.query])[0]
        if arm is Arm.VECTOR_ONLY:
            ids, first, _ = self._stream_ids(
                "SELECT id FROM gem_unit WHERE scope_id=%s AND state='active'"
                " AND (metadata->>'session_ordinal')::integer < %s"
                " ORDER BY embedding <=> %s::vector LIMIT %s",
                (
                    self.scope_id,
                    decision.cutoff_ordinal,
                    vec_literal(vector),
                    top_k,
                ),
                top_k=top_k,
                started=started,
                setup=(("hnsw.iterative_scan", "strict_order"),),
            )
            return (
                ids,
                {"mode": "vector", "termination_reason": "limit"},
                first,
                vector,
            )

        ids, first, probes = self._stream_ids(
            "SELECT tjs_open('gem_unit'::regclass,%s,%s,%s,%s,'id',%s,"
            " %s::vector,NULL,0)",
            (
                top_k,
                self.term_cond,
                self.m_seeds,
                self.hops,
                self._cutoff_filter(decision),
                vec_literal(vector),
            ),
            top_k=top_k,
            started=started,
            setup=(("hnsw.iterative_scan", "relaxed_order"),),
            read_tjs_probes=True,
        )
        probes["mode"] = "fused"
        return ids, probes, first, vector

    def _selected_semantic_scores(
        self, unit_ids: Sequence[int], vector: Sequence[float] | None
    ) -> dict[str, float]:
        """Read scores only for the bounded returned set; never drain the corpus."""
        if not unit_ids or vector is None:
            return {}
        rows = self.memory.store.conn.execute(
            "SELECT metadata->>'session_id', 1.0 - (embedding <=> %s::vector)"
            " FROM gem_unit WHERE id=ANY(%s)",
            (vec_literal(vector), list(unit_ids)),
        ).fetchall()
        return {str(session_uid): float(score) for session_uid, score in rows}

    def _render_items(
        self, unit_ids: Sequence[int], candidates: Sequence[Candidate]
    ) -> tuple[RetrievedItem, ...]:
        if not unit_ids:
            return ()
        rows = self.memory.store.conn.execute(
            "SELECT u.id, u.metadata->>'session_id', fv.value"
            " FROM gem_unit u JOIN gem_field_value fv ON fv.unit_id=u.id"
            " WHERE u.id=ANY(%s) AND fv.valid_to IS NULL"
            " ORDER BY array_position(%s::bigint[], u.id), fv.id",
            (list(unit_ids), list(unit_ids)),
        ).fetchall()
        candidate_by_uid = {item.session_uid: item for item in candidates}
        contents: dict[str, list[str]] = {}
        order: list[str] = []
        for _, session_uid, value in rows:
            uid = str(session_uid)
            if uid not in contents:
                contents[uid] = []
                order.append(uid)
            contents[uid].append(str(value))
        return tuple(
            RetrievedItem(candidate_by_uid[uid], "\n".join(contents[uid]))
            for uid in order
            if uid in candidate_by_uid
        )

    def retrieve(
        self, arm: Arm, decision: DecisionPoint, *, top_k: int
    ) -> BackendSearch:
        snapshot_id = self._snapshot_id()
        started = time.perf_counter()
        unit_ids, probes, first_row_ms, vector = self._retrieve_ids(
            arm, decision, top_k, started=started
        )
        semantic_scores = self._selected_semantic_scores(unit_ids, vector)
        candidates = self._all_candidates(
            decision,
            semantic_scores=semantic_scores,
            semantic_score_source="pgvector_cosine_similarity_selected_top_k",
        )
        ranked_items = self._render_items(unit_ids, candidates)[:top_k]
        total_ms = (time.perf_counter() - started) * 1000.0
        candidates_examined = probes.get("candidates_examined")
        graph_examined = probes.get("graph_examined")
        return BackendSearch(
            snapshot_id=snapshot_id,
            all_candidates=candidates,
            ranked_items=ranked_items,
            first_row_ms=first_row_ms,
            time_to_k_ms=total_ms,
            candidates_examined=(
                int(candidates_examined) if candidates_examined is not None else None
            ),
            # tjs_open_graph_examined() is defined by tjs_pg as graph-leg
            # edge-steps.  The extension does not expose a distinct-vertex count.
            visited_nodes=None,
            visited_edges=(int(graph_examined) if graph_examined is not None else None),
            termination_reason=str(
                probes.get("termination_reason") or probes.get("mode") or "complete"
            ),
        )

    @staticmethod
    def _event_time(ordinal: int) -> str:
        instant = datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ordinal)
        return instant.isoformat()

    def update(
        self,
        session: MemoryArenaSession,
        *,
        response: str,
        retrieval_receipt: RetrievalReceipt,
    ) -> Mapping[str, Any]:
        if not self.write_enabled:
            return {
                "committed": True,
                "snapshot_id": self._snapshot_id(),
                "update_ms": 0.0,
                "delta": None,
                "linked_edges": 0,
                "cost": None,
                "wal_bytes_interval": 0,
                "wal_scope_note": "write disabled for memory-off arm",
            }
        previous = self._direct_ids(DecisionPoint.from_session(session), recent=True)[
            :1
        ]
        content = f"Question: {session.question}\nResponse: {response}"
        started = time.perf_counter()
        wal_before = self.memory.store.conn.execute(
            "SELECT pg_current_wal_lsn()"
        ).fetchone()[0]
        phase = getattr(self.memory.embedder, "phase", None)
        phase_context = phase("construction") if phase is not None else nullcontext()
        with phase_context:
            result = self.memory.ingest(
                [
                    InteractionEvent(
                        scope_id=self.scope_id,
                        external_id=session.session_uid,
                        content=content,
                        session_id=session.session_uid,
                        role="agent_trajectory",
                        kind="episode",
                        event_time=self._event_time(session.ordinal),
                        event_order=session.ordinal,
                        metadata={
                            "dataset": "memoryarena",
                            "task_uid": session.task_uid,
                            "session_ordinal": session.ordinal,
                            "cutoff_ordinal": session.cutoff_ordinal,
                            "retrieval_receipt_sha256": retrieval_receipt.as_dict()[
                                "receipt_sha256"
                            ],
                        },
                    )
                ],
                strategy=self.strategy,
                scope_id=self.scope_id,
            )
        if not result.committed:
            raise RuntimeError(result.aborted_reason or "TriDB ingest aborted")
        new_ids = list(dict.fromkeys(int(unit_id) for unit_id in result.units))
        linked = 0
        if previous and new_ids:
            with self.memory.store.transition(
                "ingest", self.scope_id, "construction"
            ) as tx:
                self.memory.store.link(
                    tx,
                    previous[0],
                    new_ids[0],
                    kind=EdgeKind.ASSOCIATION,
                    rel="precedes_session",
                )
                self.memory.store.link(
                    tx,
                    new_ids[0],
                    previous[0],
                    kind=EdgeKind.ASSOCIATION,
                    rel="follows_session",
                )
                linked = 2
        wal_after = self.memory.store.conn.execute(
            "SELECT pg_current_wal_lsn()"
        ).fetchone()[0]
        wal_bytes = self.memory.store.conn.execute(
            "SELECT pg_wal_lsn_diff(%s, %s)", (wal_after, wal_before)
        ).fetchone()[0]
        return {
            "committed": True,
            "snapshot_id": self._snapshot_id(),
            "update_ms": (time.perf_counter() - started) * 1000.0,
            "delta": asdict(result.delta),
            "linked_edges": linked,
            "cost": asdict(result.cost),
            "wal_bytes_interval": int(wal_bytes),
            "wal_scope_note": (
                "global WAL delta over this update interval; may include concurrent "
                "database activity"
            ),
        }
