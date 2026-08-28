"""Storage foundation for the four GEM operators.

Three primitives live here, and every operator inherits their behaviour:

1. **The transition envelope** (:meth:`GemStore.transition`). Orogat & Mansour
   (arXiv:2605.26252v1) Algorithm 1 line 12 makes each operator ONE transaction
   that either commits the proposed ``M_{t+1}`` or aborts. On TriDB that
   transaction spans the relational row, the vector, and the native graph edges
   together — one transaction manager, one WAL.

2. **The vid allocator** (:meth:`GemStore.allocate_vertex`), moved out of
   ``backend.py`` because ``gem_unit.id`` MUST equal the native graph vid:
   tjs_open's graph leg resolves a reach vertex with
   ``SELECT <vec> FROM <tbl> WHERE <id_col> = $1`` passing the raw vid.

3. **The read-only view** (:class:`TriDBMemoryView`), so an ingest strategy can
   see ``M_t`` without writing.

**The abort-logging problem.** A ``gem_transition`` row written inside the
transaction is rolled back with it, so failed transitions would vanish — but
GEM correctness is a property of the TRAJECTORY, and an abort is part of it
(C2 is precisely "a violating transition is rejected"). PostgreSQL has no
autonomous transactions, so this module holds a second, dedicated **audit
connection** in autocommit: committed transitions are logged on the main
connection inside the transaction (so the log commits atomically with the state
it describes), aborted ones on the audit connection after rollback. Both paths
set ``committed`` explicitly.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from bench.agent_memory.gem.types import (
    Edge,
    EdgeKind,
    FieldValue,
    PhaseCost,
    Policy,
    PolicyEvent,
    Provenance,
    SemanticUnit,
    StateDelta,
    UnitState,
)
from bench.agent_memory.table5_track_c.tracing import stage_span

DEFAULT_DSN = "postgresql://postgres:tridb@localhost:5432/postgres"
DEFAULT_DIM = 384

#: The advisory-lock key that serialises writers. See :meth:`Tx.lock_writer`.
WRITER_LOCK_KEY = "gem_vid_alloc"

#: Exactly TWO native edge types are registered, ever. tjs_open and
#: gph_traverse_typed take a SINGLE type id (0 = ANY), not a set, so a native
#: type per relation name would make "all extension edges regardless of rel"
#: inexpressible against an equality filter. The relation name lives in
#: ``gem_edge.rel`` instead. Verified live; interface doc §6.1.
NATIVE_EDGE_KINDS: tuple[str, ...] = (
    EdgeKind.EXTENSION.value,
    EdgeKind.ASSOCIATION.value,
)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class GemError(RuntimeError):
    """Base for every error this package raises deliberately."""


class SingleWriterViolation(GemError):
    """The engine's v1 single-writer contract was broken.

    ``gph_upsert_vertex`` documents two hazards: a lost allocation race returns
    the winner's vid leaving an orphan vertex, and under REPEATABLE READ the
    re-SELECT can miss a concurrent winner and return NULL. Both are excluded
    by the single-writer contract, which :meth:`Tx.lock_writer` honours
    explicitly rather than hoping for. This error exists so the NULL case
    surfaces as a named contract violation instead of a ``TypeError`` deep
    inside ``int(vid)``.
    """


class PolicyViolation(GemError):
    """A proposed ``M_{t+1}`` failed a ``P_t`` postcondition (C2).

    Raised inside the transaction so the rollback is the enforcement mechanism.
    """

    def __init__(self, policy_name: str, detail: str) -> None:
        super().__init__(f"policy {policy_name!r} rejected the transition: {detail}")
        self.policy_name = policy_name
        self.detail = detail


def vec_literal(embedding: Sequence[float]) -> str:
    """pgvector text literal. Mirrors ``backend._vec_literal``."""
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


# ---------------------------------------------------------------------------
# Mutable accumulators — the frozen dataclasses in types.py are the OUTPUT
# ---------------------------------------------------------------------------


@dataclass
class DeltaAccumulator:
    """Mutable counterpart of :class:`~bench.agent_memory.gem.types.StateDelta`.

    Operators mutate this as they work; :meth:`freeze` produces the immutable
    delta that travels in the result and into ``gem_transition.delta``.
    """

    units_created: int = 0
    units_updated: int = 0
    units_archived: int = 0
    fields_appended: int = 0
    values_superseded: int = 0
    edges_created: int = 0
    edges_tombstoned: int = 0
    salience_updates: int = 0
    propagated_units: list[int] = field(default_factory=list)
    active_units: int | None = None
    active_fields: int | None = None

    def freeze(self) -> StateDelta:
        return StateDelta(
            units_created=self.units_created,
            units_updated=self.units_updated,
            units_archived=self.units_archived,
            fields_appended=self.fields_appended,
            values_superseded=self.values_superseded,
            edges_created=self.edges_created,
            edges_tombstoned=self.edges_tombstoned,
            salience_updates=self.salience_updates,
            propagated_units=tuple(self.propagated_units),
            active_units=self.active_units,
            active_fields=self.active_fields,
        )


@dataclass
class CostMeter:
    """Mutable counterpart of :class:`~bench.agent_memory.gem.types.PhaseCost`.

    Omri et al. (arXiv:2606.06448v1) §3.3 attribute cost to construction /
    retrieval / generation on one monotonic timeline. ``db_statements`` is
    incremented by :meth:`Tx.execute` so the count is structural rather than
    something each operator remembers to maintain.
    """

    phase: str
    llm_calls: int = 0
    embed_calls: int = 0
    embed_sequences: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_input_tokens: int = 0
    db_statements: int = 0
    gpu_joules: float | None = None  # stays None until the NVML sampler lands (M2)

    def freeze(self, seconds: float) -> PhaseCost:
        return PhaseCost(
            phase=self.phase,
            seconds=seconds,
            llm_calls=self.llm_calls,
            embed_calls=self.embed_calls,
            embed_sequences=self.embed_sequences,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            embed_input_tokens=self.embed_input_tokens,
            db_statements=self.db_statements,
            gpu_joules=self.gpu_joules,
        )


# ---------------------------------------------------------------------------
# The transaction handle
# ---------------------------------------------------------------------------


class Tx:
    """One GEM state transition in flight.

    Accumulates the delta and the phase cost, and is the only thing operators
    execute SQL through, so ``db_statements`` cannot drift from reality.
    """

    def __init__(
        self, store: GemStore, operator: str, scope_id: str, phase: str
    ) -> None:
        self.store = store
        self.operator = operator
        self.scope_id = scope_id
        self.delta = DeltaAccumulator()
        self.meter = CostMeter(phase=phase)
        self.transition_id: int | None = None
        self.policies_evaluated: list[str] = []
        # The units whose own content this transition CHANGED, as distinct from
        # `delta.propagated_units`, which holds the dependents that were flagged
        # because of them. C3 is a statement about the former's out-edges; a
        # condition that confuses the two demands that the flagged units' OWN
        # dependents be flagged, which is one hop further than anyone flags.
        self.changed_units: list[int] = []
        self._writer_locked = False

    @property
    def conn(self) -> Any:
        return self.store.conn

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        self.meter.db_statements += 1
        with stage_span(
            "relational",
            "tridb.postgresql.execute",
            backend="postgresql",
            attributes={"observable_call_kind": "database_client"},
        ):
            return self.store.conn.execute(sql, params)

    def lock_writer(self) -> None:
        """Serialise writers on the allocator; readers are unaffected.

        Taken at the top of every WRITING operator (``ingest``, ``revise``,
        ``forget``) and released automatically at commit. This is the engine's
        documented v1 single-writer contract made explicit — see
        :class:`SingleWriterViolation` for the two hazards it excludes. It is
        also the ingest-throughput ceiling noted in the plan's risk table;
        DIRECTION-04 is the unblock.
        """
        if self._writer_locked:
            return
        self.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (WRITER_LOCK_KEY,))
        self._writer_locked = True

    def allocate_vertex(self) -> int:
        """Reserve the next dense vid, which becomes ``gem_unit.id``.

        Requires the writer lock — allocation without it is exactly the race
        the single-writer contract excludes.
        """
        self.lock_writer()
        next_id = self.execute("SELECT graph_store.gph_allocated_vids()").fetchone()[0]
        row = self.execute(
            "SELECT graph_store.gph_upsert_vertex(%s)", (int(next_id),)
        ).fetchone()
        vid = None if row is None else row[0]
        if vid is None:
            # Under REPEATABLE READ gph_upsert_vertex's re-SELECT can miss a
            # concurrent winner and return NULL. Name it rather than let
            # int(None) raise TypeError three frames away.
            raise SingleWriterViolation(
                "gph_upsert_vertex returned NULL — a concurrent writer won the "
                "allocation race; the v1 single-writer contract is violated"
            )
        if int(vid) != int(next_id):
            raise SingleWriterViolation(
                f"dense-id drift: reserved {next_id}, graph returned {vid} — "
                "the v1 single-writer contract is violated"
            )
        return int(vid)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class GemStore:
    """Connection, schema, transition envelope, and the vid allocator.

    ``postcondition_hook`` is the seam ``policy.py`` (G4) plugs into: it is
    called with the in-flight :class:`Tx` just before commit and must raise
    :class:`PolicyViolation` to reject the transition. Keeping it a hook means
    G1 lands without a policy engine and G4 lands without touching this file.
    """

    def __init__(
        self,
        conn: Any,
        *,
        audit_conn: Any | None = None,
        dim: int = DEFAULT_DIM,
        postcondition_hook: Callable[[Tx], None] | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.conn = conn
        self.audit_conn = audit_conn
        self.dim = dim
        self.postcondition_hook = postcondition_hook
        self._edge_types: dict[str, int] = {}

    @classmethod
    def connect(
        cls,
        dsn: str = DEFAULT_DSN,
        *,
        dim: int = DEFAULT_DIM,
        postcondition_hook: Callable[[Tx], None] | None = None,
    ) -> GemStore:
        """Open BOTH connections: the main one and the audit one.

        The audit connection exists solely so an aborted transition still
        appears in the trajectory (see the module docstring).
        """
        import psycopg

        conn = psycopg.connect(dsn, autocommit=True)
        audit_conn = psycopg.connect(dsn, autocommit=True)
        return cls(
            conn, audit_conn=audit_conn, dim=dim, postcondition_hook=postcondition_hook
        )

    def close(self) -> None:
        self.conn.close()
        if self.audit_conn is not None:
            self.audit_conn.close()

    # -- schema ---------------------------------------------------------

    def init_schema(self) -> dict[str, Any]:
        """Apply ``schema.sql`` idempotently and register the two edge kinds."""
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS graph_store_am")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS tjs_pg")

        row = self.conn.execute(
            "SELECT atttypmod FROM pg_attribute"
            " WHERE attrelid = to_regclass('gem_unit') AND attname = 'embedding'"
        ).fetchone()
        if row is not None and row[0] != self.dim:
            raise GemError(
                f"existing gem_unit.embedding is vector({row[0]}), configured "
                f"dimension is {self.dim}"
            )

        ddl = _SCHEMA_PATH.read_text(encoding="utf-8").replace(":dim", str(self.dim))
        self.conn.execute(ddl)
        edge_types = self.bootstrap_edge_types()
        return {"ok": True, "dim": self.dim, "edge_types": edge_types}

    def bootstrap_edge_types(self) -> dict[str, int]:
        """Register exactly the two native edge kinds and cache their ids.

        **Orientation invariant, enforced in code by :meth:`link`**:
        ``A --extension--> B`` means "a change in A entails re-evaluating B".
        Revision can only walk OUT-edges — ``gph_traverse_typed`` with
        direction=in/both RAISEs until reverse adjacency lands (ADR-0016) — so
        extension edges must be written in the propagation direction or C3
        propagation silently finds nothing. Its honest limitation: a
        bidirectional entailment needs two edges, and "what does B depend on?"
        is not answerable on this engine.
        """
        for kind in NATIVE_EDGE_KINDS:
            row = self.conn.execute(
                "SELECT graph_store.register_edge_type(%s)", (kind,)
            ).fetchone()
            self._edge_types[kind] = int(row[0])
        return dict(self._edge_types)

    def edge_type_id(self, kind: EdgeKind | str) -> int:
        """Native type id for an edge kind, registering on first use."""
        name = kind.value if isinstance(kind, EdgeKind) else str(kind)
        if name not in NATIVE_EDGE_KINDS:
            raise ValueError(
                f"unknown edge kind {name!r}; GEM registers exactly "
                f"{NATIVE_EDGE_KINDS} natively and keeps the relation name in "
                "gem_edge.rel (interface doc §6.1)"
            )
        if name not in self._edge_types:
            self.bootstrap_edge_types()
        return self._edge_types[name]

    # -- the transition envelope ----------------------------------------

    @contextmanager
    def transition(self, operator: str, scope_id: str, phase: str) -> Iterator[Tx]:
        """One GEM state transition.

        Yields a :class:`Tx` that accumulates the delta and the phase cost;
        evaluates ``P_t`` as a postcondition; commits or aborts. On abort the
        transition is still logged, on the audit connection, because the
        trajectory is the correctness object and a rejected transition is part
        of it (C2).
        """
        tx = Tx(self, operator, scope_id, phase)
        started = time.perf_counter()
        try:
            with self.conn.transaction():
                # Reserve the transition id BEFORE the body runs. The log row
                # itself cannot be written until the end (it carries the delta
                # and the cost), but provenance written during the body has to
                # cite the transition that committed it — C4's
                # `gem_field_value.transition_id` is otherwise always NULL.
                # `t` is GENERATED BY DEFAULT, so an explicit value is accepted.
                tx.transition_id = self._reserve_transition_id(tx)
                yield tx
                # C5 accounting: |D_t^active| sampled after every transition.
                self._sample_active(tx)
                # C2: the postcondition runs against the PROPOSED M_{t+1},
                # which inside this transaction is simply what it can see.
                if self.postcondition_hook is not None:
                    self.postcondition_hook(tx)
                self._log(
                    self.conn,
                    tx,
                    committed=True,
                    seconds=time.perf_counter() - started,
                )
        except Exception as exc:  # noqa: BLE001 — re-raised after logging
            self._log_aborted(tx, exc, seconds=time.perf_counter() - started)
            raise

    def _reserve_transition_id(self, tx: Tx) -> int | None:
        """Take the next ``gem_transition.t`` without inserting the row yet.

        A sequence advance is not rolled back, so an aborted transition simply
        burns an id. That is the right trade: gaps in the trajectory's numbering
        are harmless, whereas provenance pointing at the wrong transition is not.
        """
        row = tx.execute(
            "SELECT nextval(pg_get_serial_sequence('gem_transition', 't'))"
        ).fetchone()
        return None if row is None or row[0] is None else int(row[0])

    def _sample_active(self, tx: Tx) -> None:
        """C5's ``|D_t^active|``, and the [GEM] agenda's third ground-truth level."""
        units = tx.execute(
            "SELECT count(*) FROM gem_unit WHERE scope_id = %s AND state = 'active'",
            (tx.scope_id,),
        ).fetchone()
        fields = tx.execute(
            "SELECT count(*) FROM gem_field_value fv JOIN gem_unit u"
            " ON u.id = fv.unit_id"
            " WHERE u.scope_id = %s AND fv.state = 'active' AND fv.valid_to IS NULL",
            (tx.scope_id,),
        ).fetchone()
        tx.delta.active_units = None if units is None else int(units[0])
        tx.delta.active_fields = None if fields is None else int(fields[0])

    def _log(
        self,
        conn: Any,
        tx: Tx,
        *,
        committed: bool,
        seconds: float,
        aborted_reason: str | None = None,
    ) -> int | None:
        meter = tx.meter
        row = conn.execute(
            "INSERT INTO gem_transition ("
            " t, scope_id, operator, phase, committed, aborted_reason,"
            " policies_evaluated, delta, active_units, active_fields, seconds,"
            " llm_calls, embed_calls, embed_sequences, prompt_tokens,"
            " completion_tokens, embed_input_tokens, db_statements, gpu_joules)"
            " VALUES (COALESCE(%s, nextval(pg_get_serial_sequence("
            "         'gem_transition', 't'))),"
            "         %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s,"
            "         %s, %s, %s, %s, %s, %s, %s, %s) RETURNING t",
            (
                tx.transition_id,
                tx.scope_id,
                tx.operator,
                meter.phase,
                committed,
                aborted_reason,
                list(tx.policies_evaluated),
                json.dumps(_delta_json(tx.delta), sort_keys=True),
                tx.delta.active_units,
                tx.delta.active_fields,
                seconds,
                meter.llm_calls,
                meter.embed_calls,
                meter.embed_sequences,
                meter.prompt_tokens,
                meter.completion_tokens,
                meter.embed_input_tokens,
                meter.db_statements,
                meter.gpu_joules,
            ),
        ).fetchone()
        return None if row is None else int(row[0])

    def _log_aborted(self, tx: Tx, exc: BaseException, *, seconds: float) -> None:
        """Log the rejected transition on the AUDIT connection, after rollback.

        Best-effort by construction: if the audit connection is unavailable the
        original exception must still reach the caller, so a logging failure is
        swallowed rather than masking the real error.
        """
        if self.audit_conn is None:
            return
        reason = f"{type(exc).__name__}: {exc}"
        try:
            self._log(
                self.audit_conn,
                tx,
                committed=False,
                seconds=seconds,
                aborted_reason=reason[:2000],
            )
        except Exception:  # noqa: BLE001 — never mask the real failure
            pass

    # -- writes shared by more than one operator -------------------------

    def link(
        self,
        tx: Tx,
        src: int,
        dst: int,
        *,
        kind: EdgeKind | str,
        rel: str,
        weight: float = 1.0,
    ) -> Edge:
        """Write one typed edge: topology in the AM, metadata in ``gem_edge``.

        Topology lives in the native graph access method, never in a relational
        join table (CLAUDE.md rule 3). ``gem_edge`` carries only what the AM
        cannot: the propagation right, the relation name, and the co-access
        counter retrieval strengthens.

        Remember the orientation invariant for ``extension``: src entails
        re-evaluating dst. See :meth:`bootstrap_edge_types`.
        """
        edge_kind = EdgeKind(kind) if not isinstance(kind, EdgeKind) else kind
        type_id = self.edge_type_id(edge_kind)
        tx.lock_writer()
        tx.execute(
            "SELECT graph_store.gph_insert_edge(%s, %s, %s)",
            (int(src), int(dst), int(type_id)),
        )
        tx.execute(
            "INSERT INTO gem_edge (src, dst, edge_type, kind, rel, weight)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (src, dst, edge_type) DO UPDATE SET"
            "  rel = EXCLUDED.rel, weight = EXCLUDED.weight, tombstoned_at = NULL",
            (int(src), int(dst), int(type_id), edge_kind.value, rel, float(weight)),
        )
        tx.delta.edges_created += 1
        return Edge(
            src=int(src),
            dst=int(dst),
            kind=edge_kind,
            rel=rel,
            weight=float(weight),
            edge_type=int(type_id),
        )

    def put_policy(self, policy: Policy) -> None:
        self.conn.execute(
            "INSERT INTO gem_policy (name, scope_id, event, condition, action,"
            " enabled, version) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)"
            " ON CONFLICT (name) DO UPDATE SET"
            "  scope_id = EXCLUDED.scope_id, event = EXCLUDED.event,"
            "  condition = EXCLUDED.condition, action = EXCLUDED.action,"
            "  enabled = EXCLUDED.enabled, version = EXCLUDED.version",
            (
                policy.name,
                policy.scope_id,
                policy.event.value,
                json.dumps(dict(policy.condition), sort_keys=True),
                json.dumps(dict(policy.action), sort_keys=True),
                policy.enabled,
                policy.version,
            ),
        )

    def trajectory(self, scope_id: str, *, since: int = 0) -> list[dict[str, Any]]:
        """The trajectory ``{M_t}``. Part of the interface, not a debug aid."""
        rows = self.conn.execute(
            "SELECT t, operator, phase, committed, aborted_reason,"
            " policies_evaluated, delta, active_units, active_fields, seconds,"
            " llm_calls, embed_calls, embed_sequences, prompt_tokens,"
            " completion_tokens, embed_input_tokens, db_statements, gpu_joules, at"
            " FROM gem_transition WHERE scope_id = %s AND t > %s ORDER BY t",
            (scope_id, int(since)),
        ).fetchall()
        keys = (
            "t",
            "operator",
            "phase",
            "committed",
            "aborted_reason",
            "policies_evaluated",
            "delta",
            "active_units",
            "active_fields",
            "seconds",
            "llm_calls",
            "embed_calls",
            "embed_sequences",
            "prompt_tokens",
            "completion_tokens",
            "embed_input_tokens",
            "db_statements",
            "gpu_joules",
            "at",
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            record = dict(zip(keys, row, strict=True))
            if isinstance(record["delta"], str):
                record["delta"] = json.loads(record["delta"])
            out.append(record)
        return out


def _delta_json(delta: DeltaAccumulator) -> dict[str, Any]:
    return {
        "units_created": delta.units_created,
        "units_updated": delta.units_updated,
        "units_archived": delta.units_archived,
        "fields_appended": delta.fields_appended,
        "values_superseded": delta.values_superseded,
        "edges_created": delta.edges_created,
        "edges_tombstoned": delta.edges_tombstoned,
        "salience_updates": delta.salience_updates,
        "propagated_units": list(delta.propagated_units),
    }


# ---------------------------------------------------------------------------
# The read-only view handed to strategies
# ---------------------------------------------------------------------------


class TriDBMemoryView:
    """Read-only window on ``M_t`` (the ``MemoryView`` Protocol).

    Backed by the SAME connection as the operator that created it, so a
    strategy running inside an operator's transaction sees that transaction's
    own uncommitted writes. That is not incidental — it is exactly what makes
    :class:`~bench.agent_memory.gem.strategies.agentic.AgenticIngestStrategy`
    coherent: the agent's later reads observe its own earlier writes without
    per-round commits, which would break atomicity.
    """

    def __init__(self, store: GemStore, *, tx: Tx | None = None) -> None:
        self.store = store
        self.tx = tx

    def _execute(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        if self.tx is not None:
            return self.tx.execute(sql, params)
        return self.store.conn.execute(sql, params)

    _UNIT_COLUMNS = (
        "id, scope_id, title, summary, state, salience, access_count,"
        " last_access, metadata"
    )

    def _unit_from_row(self, row: Sequence[Any], *, with_fields: bool) -> SemanticUnit:
        metadata = row[8]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        unit = SemanticUnit(
            id=int(row[0]),
            scope_id=row[1],
            title=row[2],
            summary=row[3],
            state=UnitState(row[4]),
            salience=float(row[5]),
            access_count=int(row[6]),
            last_access=None if row[7] is None else str(row[7]),
            metadata=metadata or {},
        )
        if with_fields:
            unit.fields = self.field_history(unit.id or 0)
        return unit

    def units(self, scope_id: str, *, limit: int = 100) -> list[SemanticUnit]:
        rows = self._execute(
            f"SELECT {self._UNIT_COLUMNS} FROM gem_unit WHERE scope_id = %s"
            " ORDER BY id LIMIT %s",
            (scope_id, int(limit)),
        ).fetchall()
        return [self._unit_from_row(row, with_fields=False) for row in rows]

    def unit(self, unit_id: int) -> SemanticUnit | None:
        row = self._execute(
            f"SELECT {self._UNIT_COLUMNS} FROM gem_unit WHERE id = %s", (int(unit_id),)
        ).fetchone()
        return None if row is None else self._unit_from_row(row, with_fields=True)

    def unit_by_title(self, scope_id: str, title: str) -> SemanticUnit | None:
        row = self._execute(
            f"SELECT {self._UNIT_COLUMNS} FROM gem_unit"
            " WHERE scope_id = %s AND title = %s",
            (scope_id, title),
        ).fetchone()
        return None if row is None else self._unit_from_row(row, with_fields=False)

    def latest_experience_before(
        self, scope_id: str, ordinal: int
    ) -> SemanticUnit | None:
        """Newest eligible Experience in a scope, using the cutoff index."""
        row = self._execute(
            f"SELECT {self._UNIT_COLUMNS} FROM gem_unit"
            " WHERE scope_id = %s AND state = 'active'"
            " AND metadata->>'node_kind' = 'experience'"
            " AND (metadata->>'experience_ordinal')::integer < %s"
            " ORDER BY (metadata->>'experience_ordinal')::integer DESC, id DESC"
            " LIMIT 1",
            (scope_id, int(ordinal)),
        ).fetchone()
        return None if row is None else self._unit_from_row(row, with_fields=False)

    def find_similar(
        self, scope_id: str, embedding: Sequence[float], *, k: int = 10
    ) -> list[SemanticUnit]:
        """The host-topic slate an LLM-mediated strategy chooses from."""
        literal = vec_literal(embedding)
        rows = self._execute(
            f"SELECT {self._UNIT_COLUMNS} FROM gem_unit"
            " WHERE scope_id = %s AND state = 'active'"
            " ORDER BY embedding <=> %s::vector LIMIT %s",
            (scope_id, literal, int(k)),
        ).fetchall()
        return [self._unit_from_row(row, with_fields=False) for row in rows]

    def field_history(self, unit_id: int) -> dict[str, list[FieldValue]]:
        """Full ``H = <(v, t, pi)>`` per field, oldest first.

        Provenance travels with every entry and is never dropped on
        supersession — that is C4.
        """
        rows = self._execute(
            "SELECT id, field, value, valid_from, valid_to, superseded_by,"
            " source_external_ids, source_unit_ids, extractor_model,"
            " prompt_version, confidence, operator, transition_id, salience, state"
            " FROM gem_field_value WHERE unit_id = %s"
            " ORDER BY field, valid_from, id",
            (int(unit_id),),
        ).fetchall()
        history: dict[str, list[FieldValue]] = {}
        for row in rows:
            history.setdefault(row[1], []).append(
                FieldValue(
                    value_id=int(row[0]),
                    value=row[2],
                    valid_from=str(row[3]),
                    valid_to=None if row[4] is None else str(row[4]),
                    superseded_by=None if row[5] is None else int(row[5]),
                    provenance=Provenance(
                        source_external_ids=tuple(row[6] or ()),
                        source_unit_ids=tuple(int(v) for v in (row[7] or ())),
                        extractor_model=row[8],
                        prompt_version=row[9],
                        confidence=None if row[10] is None else float(row[10]),
                        operator=row[11],
                        transition_id=None if row[12] is None else int(row[12]),
                    ),
                    salience=float(row[13]),
                    state=UnitState(row[14]),
                )
            )
        return history

    def edges(self, unit_id: int) -> list[Edge]:
        rows = self._execute(
            "SELECT src, dst, edge_type, kind, rel, weight, co_access_count"
            " FROM gem_edge WHERE src = %s AND tombstoned_at IS NULL"
            " ORDER BY dst",
            (int(unit_id),),
        ).fetchall()
        return [
            Edge(
                src=int(row[0]),
                dst=int(row[1]),
                edge_type=int(row[2]),
                kind=EdgeKind(row[3]),
                rel=row[4],
                weight=float(row[5]),
                co_access_count=int(row[6]),
            )
            for row in rows
        ]

    def policies(self, scope_id: str | None = None) -> list[Policy]:
        rows = self._execute(
            "SELECT name, scope_id, event, condition, action, enabled, version"
            " FROM gem_policy WHERE enabled AND (scope_id IS NULL OR scope_id = %s)"
            " ORDER BY name",
            (scope_id,),
        ).fetchall()
        out: list[Policy] = []
        for row in rows:
            condition = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            action = json.loads(row[4]) if isinstance(row[4], str) else row[4]
            out.append(
                Policy(
                    name=row[0],
                    scope_id=row[1],
                    event=PolicyEvent(row[2]),
                    condition=condition or {},
                    action=action or {},
                    enabled=bool(row[5]),
                    version=int(row[6]),
                )
            )
        return out
