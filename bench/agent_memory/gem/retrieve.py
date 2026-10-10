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

The default TOPIC/FUSED query uses seedless ``tjs_open`` with PPR fusion.
Its final ranking materializes a bounded candidate pool before returning rows;
it is NOT a streaming final-result operator. This is the explicitly accepted
GEM reranking exception to TR-1. Candidate work remains budgeted. Reinforcement
is applied only to returned hits, in the same transaction.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Callable, Sequence

from bench.agent_memory.gem.salience import ExponentialSalience
from bench.agent_memory.gem.store import GemStore, Tx, vec_literal
from bench.agent_memory.gem.types import (
    Hit,
    PhaseCost,
    Query,
    RetrievalMode,
    RetrievalResult,
    RetrievalRoute,
    StateDelta,
    UnitState,
)
from bench.agent_memory.table5_track_c.tracing import stage_span

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

#: theta_sem interim (plan P3): the engine result is over-fetched to
#: k' = min(SEM_OVERFETCH * k, sem_cap) units and sem_filter runs over that
#: bounded prefix. Disclosed in ``probes['sem_filter']``.
SEM_OVERFETCH = 4
DEFAULT_SEM_CAP = 64


def _approx_tokens(text: str) -> int:
    """Assemble_B's token estimate: ceil(len(text) / 4)."""
    return math.ceil(len(text) / 4)


def _splice(predicate: str) -> str:
    """A rendered predicate placed into SQL text that ALSO carries %s params.

    Literals are already quoted by ``sql.Literal``; a literal ``%`` (e.g. in a
    principal or scope name) would otherwise be read as a placeholder.
    """
    return predicate.replace("%", "%%")


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
        resolver: Callable[[str], int] | None = None,
        semops: Any | None = None,
        sem_cap: int = DEFAULT_SEM_CAP,
        track_pending: bool | None = None,
    ) -> None:
        """``resolver`` (plan P3): called with the scope before a STRONG read;
        it materializes the pending set Q (``PlannedIngest.resolver()``) and
        returns how many items it resolved. ``semops`` is a
        ``semantic.SemOps`` used for ``Query.sem_predicate``. ``track_pending``
        makes stale reads count open Q items (default: on iff a resolver is
        configured, so the legacy read path issues no extra statement).
        """
        self.store = store
        self.embedder = embedder
        self.salience = salience or ExponentialSalience()
        self.table = table
        self.promotion_threshold = promotion_threshold
        self.resolver = resolver
        self.semops = semops
        self.sem_cap = max(1, int(sem_cap))
        self.track_pending = (
            resolver is not None if track_pending is None else track_pending
        )
        self._tjs_probe_capabilities: tuple[bool, bool, bool] | None = None

    def warm_tjs_probe_capabilities(self) -> tuple[bool, bool, bool]:
        """Discover the read-only probe surface outside measured retrievals."""
        self._tjs_probe_capabilities = self._detect_tjs_probe_capabilities(
            self.store.conn
        )
        return self._tjs_probe_capabilities

    def _detect_tjs_probe_capabilities(self, executor: Any) -> tuple[bool, bool, bool]:
        row = executor.execute(
            "SELECT to_regprocedure('tjs_open_graph_reached()') IS NOT NULL,"
            " to_regprocedure('tjs_open_relational_examined()') IS NOT NULL,"
            " to_regprocedure('tjs_open_relational_passed()') IS NOT NULL"
        ).fetchone()
        return bool(row[0]), bool(row[1]), bool(row[2])

    # -- the operator ----------------------------------------------------

    def retrieve(self, query: Query) -> RetrievalResult:
        started = time.perf_counter()
        pending_resolved = 0
        staleness = 0
        n_sem = 0
        sem_llm_calls = 0
        try:
            if query.consistency == "strong" and self.resolver is not None:
                # Resolve Q BEFORE the read transaction: materialization runs
                # its own ingest transitions (the FIFO prefix, plan D2), and
                # this read's snapshot is taken after they commit, so it sees
                # Mat(S) for every item enqueued before the call.
                pending_resolved = int(self.resolver(query.scope_id))
            with self.store.transition("retrieve", query.scope_id, "retrieval") as tx:
                if query.consistency == "strong" or self.track_pending:
                    staleness = self._open_pending(tx, query.scope_id)
                    if query.consistency == "strong" and staleness:
                        raise RuntimeError(
                            f"strong read: {staleness} pending derivation(s) remain "
                            "open in scope and no resolver materialized them"
                        )
                vector = self._resolve_vector(tx, query)
                route_query = query
                if query.sem_predicate:
                    if self.semops is None:
                        raise ValueError(
                            "sem_predicate needs a SemOps (RetrieveOperator(semops=...))"
                        )
                    route_query = replace(
                        query, k=min(SEM_OVERFETCH * int(query.k), self.sem_cap)
                    )
                ids, probes = self._route(tx, route_query, vector)
                probes = dict(probes)
                if query.principal is not None:
                    probes["principal_predicate"] = True
                hits = self._materialise(tx, query, ids)
                if query.sem_predicate:
                    hits, sem_probe = self._sem_filter(query, ids, hits, route_query.k)
                    probes["sem_filter"] = sem_probe
                    n_sem = sem_probe["n_sem"]
                    sem_llm_calls = sem_probe["llm_calls"]

                if query.reinforce:
                    # C6, in the same transaction as the read.
                    hits = self._reinforce(tx, query, hits)

            # Captured AFTER the envelope closes: transition() samples the C5
            # counts, evaluates P_t, and writes the log during __exit__, so
            # anything read inside the body is stale.
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

        prompt_block = build_prompt_block(hits, budget_tokens=query.budget_tokens)
        probes["consistency"] = query.consistency
        if query.budget_tokens is not None:
            probes["assemble"] = {
                "budget_tokens": int(query.budget_tokens),
                "approx_tokens": _approx_tokens(prompt_block),
                "lines_kept": len(prompt_block.splitlines()) if prompt_block else 0,
                "lines_total": len(hits),
                "token_estimate": "ceil(len/4)",
            }
        return RetrievalResult(
            operator="retrieve",
            committed=True,
            delta=delta,
            cost=cost,
            transition_id=transition_id,
            policies_evaluated=policies,
            hits=tuple(hits),
            probes=probes,
            prompt_block=prompt_block,
            pending_resolved=pending_resolved,
            staleness=0 if query.consistency == "strong" else staleness,
            n_sem=n_sem,
            sem_llm_calls=sem_llm_calls,
        )

    # -- plan P3 stages ----------------------------------------------------

    @staticmethod
    def _open_pending(tx: Tx, scope_id: str) -> int:
        row = tx.execute(
            "SELECT count(*) FROM gem_pending WHERE scope_id = %s AND status = 'pending'",
            (scope_id,),
        ).fetchone()
        return 0 if row is None or row[0] is None else int(row[0])

    def _sem_filter(
        self, query: Query, ids: Sequence[int], hits: Sequence[Hit], k_prime: int
    ) -> tuple[list[Hit], dict[str, Any]]:
        """theta_sem over the bounded engine prefix (interim, plan P3).

        One logical evaluation per candidate UNIT, in engine order; the item is
        the unit's title and current field values (no ids, so the cache key is
        stable across replays). Survivors keep engine order (ADR-0007: the
        engine stays the rank authority) and are cut to ``query.k`` units.
        """
        by_unit: dict[int, list[Hit]] = {}
        order: list[int] = []
        for hit in hits:
            if hit.unit_id not in by_unit:
                order.append(hit.unit_id)
            by_unit.setdefault(hit.unit_id, []).append(hit)
        items = [
            {
                "title": by_unit[unit_id][0].title,
                "fields": {
                    h.field_name: h.value for h in by_unit[unit_id] if h.field_name
                },
            }
            for unit_id in order
        ]
        before = self.semops.meter.snapshot()
        verdicts = (
            self.semops.sem_filter(str(query.sem_predicate), items) if items else []
        )
        spent = self.semops.meter.diff(before)
        kept_units = [u for u, keep in zip(order, verdicts, strict=True) if keep][
            : int(query.k)
        ]
        kept = set(kept_units)
        survivors = [hit for hit in hits if hit.unit_id in kept]
        return survivors, {
            "predicate": query.sem_predicate,
            "k": int(query.k),
            "k_prime": int(k_prime),
            "cap": self.sem_cap,
            "overfetch": SEM_OVERFETCH,
            "candidates": len(order),
            "engine_returned": len(ids),
            "survivors": len(kept_units),
            "exhausted": len(ids) < int(k_prime),
            "n_sem": int(spent.get("n_sem", 0)),
            "llm_calls": int(spent.get("llm_calls", 0)),
            "cache_hits": int(spent.get("cache_hits", 0)),
            "placement": "post_engine_bounded_overfetch",
        }

    # -- query admission -------------------------------------------------

    def _resolve_vector(self, tx: Tx, query: Query) -> list[float] | None:
        if query.embedding is not None:
            return [float(value) for value in query.embedding]
        if query.text is None:
            # Two modes have no vector leg at all: a pure relational lookup, and
            # a pure GRAPH traversal, which starts from an explicit anchor and
            # never ranks by similarity. Demanding a vector from them made the
            # whole GRAPH mode unreachable without one.
            if query.mode in (RetrievalMode.RELATIONAL, RetrievalMode.GRAPH):
                return None
            raise ValueError("query needs text or embedding for a vector leg")
        if self.embedder is None:
            raise RuntimeError("no embedder configured for query text")
        with stage_span(
            "embedding",
            "tridb.query_embedding",
            backend="qwen3_embedding_0.6b",
            attributes={"observable_call_kind": "http_client"},
        ):
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
        if query.principal is not None:
            # theta_pol (plan P3, D5): a unit that carries an 'acl' array is
            # visible only to principals listed in it; a unit without one is
            # public. Pushed with scope_id so it runs INSIDE the operator.
            # tjs_open takes the predicate as text, so the value is bound via
            # sql.Literal (server-side quoting), never by string formatting.
            clauses.append(
                sql.SQL("(NOT (metadata ? 'acl') OR metadata->'acl' ? {})").format(
                    sql.Literal(str(query.principal))
                )
            )
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
        if query.principal is not None:
            clauses.append("(NOT (u.metadata ? 'acl') OR u.metadata->'acl' ? %s)")
            params.append(str(query.principal))
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
            f"SELECT id FROM {self.table} WHERE {_splice(predicate)}"
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
            f"SELECT id FROM {self.table} WHERE {_splice(predicate)} ORDER BY id LIMIT %s",
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
        type_id = (
            0
            if query.graph_edge_kind is None
            else self.store.edge_type_id(query.graph_edge_kind)
        )
        predicate = self.predicate(query).as_string(self.store.conn)
        if query.hops <= 1:
            # Target-list (ProjectSet) position, per the AM's contract: a
            # FROM-clause FunctionScan loses early termination under LIMIT.
            rows = tx.execute(
                f"SELECT reached.dst FROM (SELECT (e).dst::bigint AS dst FROM ("
                " SELECT graph_store.gph_traverse_typed(%s, %s, 0, -1) AS e"
                f") traversed) reached JOIN {self.table} u ON u.id = reached.dst"
                f" WHERE {_splice(predicate)} LIMIT %s",
                (int(query.anchor_id), int(type_id), int(query.k)),
            ).fetchall()
        else:
            rows = tx.execute(
                f"SELECT reached.dst FROM (SELECT graph_store.gph_traverse_bfs("
                f"%s, %s, %s)::bigint AS dst) reached JOIN {self.table} u"
                f" ON u.id = reached.dst WHERE {_splice(predicate)} LIMIT %s",
                (
                    int(query.anchor_id),
                    int(query.hops),
                    int(type_id),
                    int(query.k),
                ),
            ).fetchall()
        ids = [int(row[0]) for row in rows if row[0] is not None]
        return ids, {
            "route": query.route.value,
            "mode": "graph",
            "edge_type": type_id,
            "filter": predicate,
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
        if query.graph_scoring is None:
            scoring = tx.execute("SHOW tjs.graph_scoring").fetchone()[0]
        else:
            scoring = query.graph_scoring
        if scoring not in ("ppr", "membership"):
            raise ValueError(f"unsupported tjs.graph_scoring: {scoring!r}")
        # Check the applied value. LOCAL avoids leaking this query's choice
        # into a later transaction on the same connection. The installed
        # tjs_pg extension must support graph_scoring (see the live PPR test).
        observed = tx.execute(
            "SELECT set_config('tjs.graph_scoring', %s, true)", (scoring,)
        ).fetchone()
        if not observed or observed[0] != scoring:
            raise RuntimeError("failed to set transaction-local tjs.graph_scoring")
        ppr_enabled = (
            scoring == "ppr"
            and anchor_id is None
            and query.m_seeds > 0
            and query.hops > 0
        )
        predicate = self.predicate(query).as_string(self.store.conn)
        edge_type = (
            0
            if query.graph_edge_kind is None
            else self.store.edge_type_id(query.graph_edge_kind)
        )
        if self._tjs_probe_capabilities is None:
            self._tjs_probe_capabilities = self._detect_tjs_probe_capabilities(tx)
        (
            graph_reached_available,
            relational_examined_available,
            relational_passed_available,
        ) = self._tjs_probe_capabilities
        has_graph_reached_probe = graph_reached_available
        has_relational_probes = (
            relational_examined_available and relational_passed_available
        )
        with stage_span(
            "fusion",
            "tridb.tjs_open",
            backend="postgresql_tjs_pg",
            attributes={
                "includes": ["vector", "graph", "relational"],
                "iterator_contract": "materialized_final_ranking",
                "early_termination": True,
                "final_results_streaming": False,
                "graph_scoring": scoring,
                "ppr_enabled": ppr_enabled,
                "attribution": "single_fused_operator",
            },
        ):
            tx.execute(f"SET LOCAL hnsw.iterative_scan = {FUSED_ITERATIVE_SCAN}")
            rows = tx.execute(
                f"SELECT t FROM tjs_open('{self.table}', %s, %s, %s, %s, 'id', %s,"
                " %s::vector, %s, %s) AS t",
                (
                    int(query.k),
                    int(query.term_cond),
                    int(query.m_seeds),
                    int(query.hops),
                    predicate,
                    vec_literal(vector),
                    None if anchor_id is None else int(anchor_id),
                    int(edge_type),
                ),
            ).fetchall()
            ids = [int(row[0]) for row in rows]
            # Probes describe the LAST call and must be read on THIS connection,
            # before anything else runs on it. Censoring travels.
            probe_sql = (
                "SELECT tjs_open_candidates_examined(), tjs_open_graph_examined(),"
                " tjs_open_graph_censored(), tjs_open_termination_reason(),"
                " tjs_open_budget_capped(), tjs_open_bridges_injected()"
            )
            if has_graph_reached_probe:
                probe_sql += ", tjs_open_graph_reached()"
            if has_relational_probes:
                probe_sql += (
                    ", tjs_open_relational_examined(), tjs_open_relational_passed()"
                )
            probe_started = time.perf_counter()
            probe_row = tx.execute(probe_sql).fetchone()
            probe_read_ms = (time.perf_counter() - probe_started) * 1000
        next_probe = 6
        graph_reached = None
        if has_graph_reached_probe:
            graph_reached = probe_row[next_probe]
            next_probe += 1
        relational_examined = None
        relational_passed = None
        if has_relational_probes:
            relational_examined = probe_row[next_probe]
            relational_passed = probe_row[next_probe + 1]
        return ids, {
            "route": query.route.value,
            "mode": "fused",
            "retrieval_backend": "tjs_open",
            "graph_scoring": scoring,
            "ppr_enabled": ppr_enabled,
            "m_seeds": query.m_seeds,
            "hops": query.hops,
            "final_results_streaming": False,
            "hit_score_kind": "rank_position",
            "candidates_examined": probe_row[0],
            "graph_examined": probe_row[1],
            "graph_censored": probe_row[2],
            "termination_reason": probe_row[3],
            "budget_capped": probe_row[4],
            "bridges_injected": probe_row[5],
            "graph_reached": graph_reached,
            "graph_reached_available": has_graph_reached_probe,
            "relational_candidates_examined": relational_examined,
            "relational_candidates_passed": relational_passed,
            "relational_candidate_probes_available": has_relational_probes,
            "instrumentation_probe_read_ms": probe_read_ms,
            "hnsw_iterative_scan": FUSED_ITERATIVE_SCAN,
            "filter": predicate,
            "anchor_id": anchor_id,
            "edge_type": edge_type,
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

        Co-retrieved units strengthen the association edges that ALREADY exist
        between them (``co_access_count`` + 1 in ``gem_edge``). No new edge is
        created here: a new edge would have to go through the native graph AM
        (``GemStore.link``) and is left to ingest / revise (plan D9). An
        association edge whose co_access_count
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


def build_prompt_block(hits: Sequence[Hit], *, budget_tokens: int | None = None) -> str:
    """Render hits into the block a generator consumes.

    Deterministic ordering and an explicit unit id per line, so a downstream
    judge can attribute an answer to a unit rather than to the corpus.

    ``budget_tokens`` (Assemble_B, plan P3): keep the longest PREFIX of lines,
    in hit order, whose rendered block stays within the budget under the
    estimate ``ceil(len(text) / 4)``. Lines are never reordered or cut midway.
    """
    lines: list[str] = []
    for hit in hits:
        label = f"[unit {hit.unit_id}] {hit.title}"
        if hit.field_name:
            line = f"{label} — {hit.field_name}: {hit.value}"
        else:
            line = label
        if (
            budget_tokens is not None
            and _approx_tokens("\n".join([*lines, line])) > budget_tokens
        ):
            break
        lines.append(line)
    return "\n".join(lines)
