"""The ``retrieve`` operator — an output AND a state transition.

    R(M_t, q) -> o    AND    M_{t+1} = U(M_t, {}, R_q)

[GEM] Observation 1 is that this cannot be patched from outside: "Caches,
materialized views, and post-retrieval triggers can record that an access
occurred, but they cannot lift the operator out of being a pure function,
because the state-modifying step is decoupled from the query." So the salience
write lives INSIDE this operator's transaction. TriDB commits the read and the
write together — one transaction manager, one WAL.

Shape::

    admit(q) -> resolve query vector
      -> route: TOPIC | TEMPORAL | STRUCTURAL
      -> mode:  VECTOR | GRAPH | RELATIONAL | FUSED
      -> fetch CURRENT field values for the returned unit ids   (C1)
      -> build hits + prompt block
      -> if q.reinforce: the C6 write   (SAME transaction)
      -> log gem_transition -> COMMIT

**TR-1.** ``tjs_open``'s ids are materialised FIRST and the operator closes
before any UPDATE runs. Retrieval must honour Open/Next/Close and early
termination; C6 must not turn it into a blocking operator.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from bench.agent_memory.gem.salience import ExponentialSalience
from bench.agent_memory.gem.store import GemStore, Tx, vec_literal
from bench.agent_memory.gem.types import (
    EdgeKind,
    Hit,
    PhaseCost,
    Query,
    RetrievalMode,
    RetrievalResult,
    RetrievalRoute,
    StateDelta,
    UnitState,
)

#: The fused vector-first path REQUIRES relaxed_order — the engine refuses
#: strict_order outright. Fused and vector-only are therefore different
#: operating points and must never be pooled in one results table
#: (interface doc §6.1a). Recorded in every result's ``probes``.
FUSED_ITERATIVE_SCAN = "relaxed_order"
VECTOR_ITERATIVE_SCAN = "strict_order"

#: An association edge whose co_access_count crosses this is FLAGGED as an
#: extension candidate — flagged only. Promotion is a `revise` decision, never
#: a retrieval side effect, because an extension edge grants propagation rights
#: and retrieval must not silently widen the C3 frontier.
DEFAULT_PROMOTION_THRESHOLD = 5


class RetrieveOperator:
    """Executes one query and, unless ablated, commits the C6 write."""

    def __init__(
        self,
        store: GemStore,
        *,
        embedder: Any | None = None,
        salience: ExponentialSalience | None = None,
        table: str = "gem_unit",
        promotion_threshold: int = DEFAULT_PROMOTION_THRESHOLD,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.salience = salience or ExponentialSalience()
        self.table = table
        self.promotion_threshold = promotion_threshold

    # -- the operator ----------------------------------------------------

    def retrieve(self, query: Query) -> RetrievalResult:
        started = time.perf_counter()
        try:
            with self.store.transition("retrieve", query.scope_id, "retrieval") as tx:
                vector = self._resolve_vector(tx, query)
                ids, probes = self._route(tx, query, vector)
                hits = self._materialise(tx, query, ids)

                if query.reinforce:
                    # C6, in the same transaction as the read.
                    hits = self._reinforce(tx, query, hits)

                delta = tx.delta.freeze()
                cost = tx.meter.freeze(time.perf_counter() - started)
                transition_id = tx.transition_id
                policies = tuple(tx.policies_evaluated)
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            return RetrievalResult(
                operator="retrieve",
                committed=False,
                delta=StateDelta(),
                cost=PhaseCost(
                    phase="retrieval", seconds=time.perf_counter() - started
                ),
                aborted_reason=f"{type(exc).__name__}: {exc}",
            )

        return RetrievalResult(
            operator="retrieve",
            committed=True,
            delta=delta,
            cost=cost,
            transition_id=transition_id,
            policies_evaluated=policies,
            hits=tuple(hits),
            probes=probes,
            prompt_block=build_prompt_block(hits),
        )

    # -- query admission -------------------------------------------------

    def _resolve_vector(self, tx: Tx, query: Query) -> list[float] | None:
        if query.embedding is not None:
            return [float(value) for value in query.embedding]
        if query.text is None:
            if query.mode is RetrievalMode.RELATIONAL:
                return None  # a pure relational/temporal lookup needs no vector
            raise ValueError("query needs text or embedding for a vector leg")
        if self.embedder is None:
            raise RuntimeError("no embedder configured for query text")
        encoded = self.embedder.encode([query.text])
        tx.meter.embed_calls += 1
        tx.meter.embed_sequences += 1
        return [float(value) for value in encoded[0]]

    def predicate(self, query: Query) -> Any:
        """The relational predicate, pushed INTO the operator.

        The active-state filter is part of it, always. Attenuated content then
        costs nothing instead of being fetched and discarded — that is why
        ``forget`` is cheap at retrieval time rather than a post-filter.
        ``sql.Literal`` quoting is verified injection-safe.
        """
        from psycopg import sql

        clauses = [sql.SQL("scope_id = {}").format(sql.Literal(query.scope_id))]
        if not query.include_archived:
            clauses.append(sql.SQL("state = 'active'"))
        if query.extra_filter:
            clauses.append(sql.SQL("({})").format(sql.SQL(query.extra_filter)))
        return sql.SQL(" AND ").join(clauses)

    # -- routing ---------------------------------------------------------

    def _route(
        self, tx: Tx, query: Query, vector: Sequence[float] | None
    ) -> tuple[list[int], dict[str, Any]]:
        if query.route is RetrievalRoute.TEMPORAL:
            return self._temporal(tx, query, vector)
        if query.route is RetrievalRoute.STRUCTURAL:
            return self._structural(tx, query, vector)
        return self._topic(tx, query, vector)

    def _topic(
        self, tx: Tx, query: Query, vector: Sequence[float] | None
    ) -> tuple[list[int], dict[str, Any]]:
        if query.mode is RetrievalMode.FUSED:
            return self._tjs_open(tx, query, vector, anchor_id=query.anchor_id)
        if query.mode is RetrievalMode.GRAPH:
            return self._graph_only(tx, query)
        if query.mode is RetrievalMode.RELATIONAL:
            return self._relational_only(tx, query)
        return self._vector_only(tx, query, vector)

    def _structural(
        self, tx: Tx, query: Query, vector: Sequence[float] | None
    ) -> tuple[list[int], dict[str, Any]]:
        """Anchored retrieval: the graph leg starts from an explicit unit."""
        if query.anchor_id is None:
            raise ValueError("STRUCTURAL route requires anchor_id")
        if query.mode is RetrievalMode.GRAPH:
            return self._graph_only(tx, query)
        return self._tjs_open(tx, query, vector, anchor_id=query.anchor_id)

    def _temporal(
        self, tx: Tx, query: Query, vector: Sequence[float] | None
    ) -> tuple[list[int], dict[str, Any]]:
        """History-predicate retrieval, optionally re-ranked by the vector.

        C1: prior values appear only when ``q`` explicitly requests historical
        context, which on this route is ``as_of``.
        """
        # Parameters are positional, so they are built in the order their
        # placeholders APPEAR in the statement — the ranking vector sits in the
        # select list and therefore binds first, ahead of the WHERE clause.
        params: list[Any] = []
        if vector is not None:
            projection = "u.embedding <=> %s::vector"
            params.append(vec_literal(vector))
        else:
            projection = "u.id"

        params.append(query.scope_id)
        clauses = ["u.scope_id = %s"]
        if not query.include_archived:
            clauses.append("u.state = 'active'")
        if query.as_of is not None:
            # The value that was current AT as_of, not the value current now.
            clauses.append(
                "fv.valid_from <= %s AND (fv.valid_to IS NULL OR fv.valid_to > %s)"
            )
            params.extend([query.as_of, query.as_of])
        else:
            clauses.append("fv.valid_to IS NULL")

        sql_text = (
            f"SELECT DISTINCT u.id, {projection} AS ord"
            " FROM gem_unit u JOIN gem_field_value fv ON fv.unit_id = u.id"
            " WHERE " + " AND ".join(clauses) + f" ORDER BY ord LIMIT {int(query.k)}"
        )
        rows = tx.execute(sql_text, tuple(params)).fetchall()
        return [int(row[0]) for row in rows], {
            "route": "temporal",
            "as_of": query.as_of,
            "hnsw_iterative_scan": None,
        }

    def _vector_only(
        self, tx: Tx, query: Query, vector: Sequence[float] | None
    ) -> tuple[list[int], dict[str, Any]]:
        if vector is None:
            raise ValueError("VECTOR mode needs a query vector")
        literal = vec_literal(vector)
        predicate = self.predicate(query).as_string(self.store.conn)
        tx.execute(f"SET hnsw.iterative_scan = {VECTOR_ITERATIVE_SCAN}")
        rows = tx.execute(
            f"SELECT id FROM {self.table} WHERE {predicate}"
            " ORDER BY embedding <=> %s::vector LIMIT %s",
            (literal, int(query.k)),
        ).fetchall()
        return [int(row[0]) for row in rows], {
            "route": query.route.value,
            "mode": "vector",
            "hnsw_iterative_scan": VECTOR_ITERATIVE_SCAN,
            "filter": predicate,
        }

    def _relational_only(
        self, tx: Tx, query: Query
    ) -> tuple[list[int], dict[str, Any]]:
        predicate = self.predicate(query).as_string(self.store.conn)
        rows = tx.execute(
            f"SELECT id FROM {self.table} WHERE {predicate} ORDER BY id LIMIT %s",
            (int(query.k),),
        ).fetchall()
        return [int(row[0]) for row in rows], {
            "route": query.route.value,
            "mode": "relational",
            "filter": predicate,
            "hnsw_iterative_scan": None,
        }

    def _graph_only(self, tx: Tx, query: Query) -> tuple[list[int], dict[str, Any]]:
        """Pure traversal from an anchor, over ONE edge kind or ANY.

        ``edge_type`` is a single id, not a set (0 = ANY) — which is fine only
        because GEM registers exactly two native kinds and keeps the relation
        name relationally (interface doc §6.1).
        """
        if query.anchor_id is None:
            raise ValueError("GRAPH mode requires anchor_id")
        type_id = 0
        if query.extra_filter in (EdgeKind.EXTENSION.value, EdgeKind.ASSOCIATION.value):
            type_id = self.store.edge_type_id(query.extra_filter)
        if query.hops <= 1:
            # Target-list (ProjectSet) position, per the AM's contract: a
            # FROM-clause FunctionScan loses early termination under LIMIT.
            rows = tx.execute(
                "SELECT (e).dst FROM (SELECT graph_store.gph_traverse_typed("
                "%s, %s, 0, -1) AS e) s",
                (int(query.anchor_id), int(type_id)),
            ).fetchall()
        else:
            rows = tx.execute(
                "SELECT graph_store.gph_traverse_bfs(%s, %s, %s)",
                (int(query.anchor_id), int(query.hops), int(type_id)),
            ).fetchall()
        ids = [int(row[0]) for row in rows if row[0] is not None][: query.k]
        return ids, {
            "route": query.route.value,
            "mode": "graph",
            "edge_type": type_id,
            "hnsw_iterative_scan": None,
        }

    def _tjs_open(
        self,
        tx: Tx,
        query: Query,
        vector: Sequence[float] | None,
        *,
        anchor_id: int | None,
    ) -> tuple[list[int], dict[str, Any]]:
        """The tri-modal path: vector leg + native graph leg + pushed predicate."""
        if vector is None:
            raise ValueError("FUSED mode needs a query vector")
        predicate = self.predicate(query).as_string(self.store.conn)
        tx.execute(f"SET hnsw.iterative_scan = {FUSED_ITERATIVE_SCAN}")
        rows = tx.execute(
            f"SELECT t FROM tjs_open('{self.table}', %s, %s, %s, %s, 'id', %s,"
            " %s::vector, %s) AS t",
            (
                int(query.k),
                int(query.term_cond),
                int(query.m_seeds),
                int(query.hops),
                predicate,
                vec_literal(vector),
                None if anchor_id is None else int(anchor_id),
            ),
        ).fetchall()
        ids = [int(row[0]) for row in rows]
        # Probes describe the LAST call and must be read on THIS connection,
        # before anything else runs on it. Censoring travels.
        probe_row = tx.execute(
            "SELECT tjs_open_candidates_examined(), tjs_open_graph_examined(),"
            " tjs_open_graph_censored(), tjs_open_termination_reason(),"
            " tjs_open_budget_capped(), tjs_open_bridges_injected()"
        ).fetchone()
        return ids, {
            "route": query.route.value,
            "mode": "fused",
            "candidates_examined": probe_row[0],
            "graph_examined": probe_row[1],
            "graph_censored": probe_row[2],
            "termination_reason": probe_row[3],
            "budget_capped": probe_row[4],
            "bridges_injected": probe_row[5],
            "hnsw_iterative_scan": FUSED_ITERATIVE_SCAN,
            "filter": predicate,
            "anchor_id": anchor_id,
        }

    # -- materialisation -------------------------------------------------

    def _materialise(self, tx: Tx, query: Query, ids: Sequence[int]) -> list[Hit]:
        """Fetch the CURRENT field values for the returned units (C1).

        With ``as_of`` set, fetch the values that were current then instead —
        prior values surface only when the query explicitly asks for history.
        """
        if not ids:
            return []
        id_list = [int(i) for i in ids]
        units = {
            int(row[0]): row
            for row in tx.execute(
                "SELECT id, title, state, salience FROM gem_unit WHERE id = ANY(%s)",
                (id_list,),
            ).fetchall()
        }

        if query.as_of is not None:
            value_rows = tx.execute(
                "SELECT unit_id, field, value FROM gem_field_value"
                " WHERE unit_id = ANY(%s) AND state <> 'archived'"
                "   AND valid_from <= %s AND (valid_to IS NULL OR valid_to > %s)"
                " ORDER BY unit_id, field",
                (id_list, query.as_of, query.as_of),
            ).fetchall()
        elif query.include_history:
            value_rows = tx.execute(
                "SELECT unit_id, field, value FROM gem_field_value"
                " WHERE unit_id = ANY(%s) AND state <> 'archived'"
                " ORDER BY unit_id, field, valid_from",
                (id_list,),
            ).fetchall()
        else:
            value_rows = tx.execute(
                "SELECT unit_id, field, value FROM gem_field_value"
                " WHERE unit_id = ANY(%s) AND valid_to IS NULL"
                "   AND state <> 'archived' ORDER BY unit_id, field",
                (id_list,),
            ).fetchall()

        by_unit: dict[int, list[tuple[str, str]]] = {}
        for row in value_rows:
            by_unit.setdefault(int(row[0]), []).append((row[1], row[2]))

        via = "vector" if query.mode is not RetrievalMode.GRAPH else "graph"
        hits: list[Hit] = []
        for rank, unit_id in enumerate(id_list):
            unit_row = units.get(unit_id)
            if unit_row is None:
                # A vertex the graph leg returned whose row this snapshot
                # cannot see. Commit-visible graph reads make this possible
                # (interface doc §6.5) — skip rather than fabricate a hit.
                continue
            values = by_unit.get(unit_id) or [(None, None)]
            for field_name, value in values:
                hits.append(
                    Hit(
                        unit_id=unit_id,
                        title=unit_row[1],
                        field_name=field_name,
                        value=value,
                        # Rank-derived: the engine returns an ordered id list,
                        # not distances, so a fabricated similarity would be
                        # worse than an honest positional score.
                        score=1.0 - (rank / max(len(id_list), 1)),
                        state=UnitState(unit_row[2]),
                        salience_before=float(unit_row[3]),
                        via=via,
                    )
                )
        return hits

    # -- C6 ---------------------------------------------------------------

    def _reinforce(self, tx: Tx, query: Query, hits: Sequence[Hit]) -> list[Hit]:
        """The C6 write: salience up, access recorded, structure updated.

        None of these columns is indexed and both tables are ``fillfactor=70``,
        so these are 100% HOT updates costing zero index churn (measured,
        interface §6.3). That is what makes "every retrieval is a write"
        affordable.
        """
        if not hits:
            return list(hits)

        ranked: dict[int, int] = {}
        for rank, hit in enumerate(hits):
            ranked.setdefault(hit.unit_id, rank)
        unit_ids = list(ranked)
        k = max(len(unit_ids), 1)

        current = {
            int(row[0]): float(row[1])
            for row in tx.execute(
                "SELECT id, salience FROM gem_unit WHERE id = ANY(%s)", (unit_ids,)
            ).fetchall()
        }
        updated = {
            unit_id: self.salience.reinforce(
                current.get(unit_id, 0.0), rank=min(rank, k - 1), k=k
            )
            for unit_id, rank in ranked.items()
        }

        tx.execute(
            "UPDATE gem_unit SET salience = v.salience, access_count = access_count + 1,"
            " last_access = now() FROM (SELECT * FROM unnest(%s::bigint[],"
            " %s::double precision[]) AS s(id, salience)) v WHERE gem_unit.id = v.id",
            (list(updated), [updated[i] for i in updated]),
        )
        tx.delta.salience_updates += len(updated)

        # Per-field salience: [GEM] §3.2's sub-unit granularity — part of a unit
        # may be attenuated while the rest stays current.
        fields = sorted({hit.field_name for hit in hits if hit.field_name})
        if fields:
            tx.execute(
                "UPDATE gem_field_value SET salience = salience + %s"
                " WHERE unit_id = ANY(%s) AND field = ANY(%s) AND valid_to IS NULL",
                (self.salience.floor, unit_ids, fields),
            )

        self._update_structure(tx, query, unit_ids)

        after = {
            int(row[0]): float(row[1])
            for row in tx.execute(
                "SELECT id, salience FROM gem_unit WHERE id = ANY(%s)", (unit_ids,)
            ).fetchall()
        }
        return [
            Hit(
                unit_id=hit.unit_id,
                title=hit.title,
                field_name=hit.field_name,
                value=hit.value,
                score=hit.score,
                state=hit.state,
                salience_before=hit.salience_before,
                salience_after=after.get(hit.unit_id),
                via=hit.via,
            )
            for hit in hits
        ]

    def _update_structure(self, tx: Tx, query: Query, unit_ids: Sequence[int]) -> None:
        """ "Retrieval updates memory structure" — the second half of C6.

        Co-retrieved units strengthen their association edge; co-retrieved units
        with no edge get one created. An association edge whose co_access_count
        crosses the threshold is FLAGGED as an extension candidate and nothing
        more: promoting it would grant propagation rights and silently widen the
        C3 frontier, which is a ``revise`` decision.
        """
        if len(unit_ids) < 2:
            return
        ids = [int(i) for i in unit_ids]
        tx.execute(
            "UPDATE gem_edge SET co_access_count = co_access_count + 1"
            " WHERE src = ANY(%s) AND dst = ANY(%s) AND kind = 'association'"
            "   AND tombstoned_at IS NULL",
            (ids, ids),
        )
        # Flag, never promote.
        tx.execute(
            "UPDATE gem_unit SET metadata = jsonb_set(metadata,"
            " '{extension_candidate}', 'true'::jsonb) WHERE id IN ("
            "  SELECT src FROM gem_edge WHERE kind = 'association'"
            "   AND tombstoned_at IS NULL AND co_access_count >= %s"
            "   AND src = ANY(%s))",
            (int(self.promotion_threshold), ids),
        )


def build_prompt_block(hits: Sequence[Hit]) -> str:
    """Render hits into the block a generator consumes.

    Deterministic ordering and an explicit unit id per line, so a downstream
    judge can attribute an answer to a unit rather than to the corpus.
    """
    lines: list[str] = []
    for hit in hits:
        label = f"[unit {hit.unit_id}] {hit.title}"
        if hit.field_name:
            lines.append(f"{label} — {hit.field_name}: {hit.value}")
        else:
            lines.append(label)
    return "\n".join(lines)
