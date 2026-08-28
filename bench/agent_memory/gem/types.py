"""State model for Governed Evolving Memory (GEM) on TriDB.

Implements the state tuple of Orogat & Mansour, "Is Agent Memory a Database?"
(arXiv:2605.26252v1) Definition 1:

    M_t = (D_t, S_t, P_t)

    D_t  stored content, organised as SEMANTIC UNITS that hold all related data
         elements, each field carrying a value history H = <(v, t, pi)>
    S_t  the structural organisation over that content (typed edges)
    P_t  policies governing access, ingestion, revision and forgetting

Correctness is a property of the trajectory {M_t}, not of any single record
(Definition 4, C1-C6). The types here exist to make each condition
representable; enforcement lives in the operators and in the schema
(``schema.sql``).

Nothing in this module performs I/O. It is the vocabulary shared by the four
operators, the ingest strategies, and the profiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# D_t — content
# ---------------------------------------------------------------------------


class UnitState(str, Enum):
    """The graded attenuation ladder (paper §4.2 "Forgetting").

    Forgetting is graded, never destructive: below theta_summary a history is
    compressed, below theta_remove a field is hidden from active retrieval,
    below theta_archive the topic is archived but remains recoverable through
    explicit lookup. ARCHIVED content still satisfies C5's "archived content
    remains recoverable".
    """

    ACTIVE = "active"
    COMPRESSED = "compressed"
    HIDDEN = "hidden"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class Provenance:
    """pi — the provenance component of a value-history entry.

    C4 (provenance preservation) requires that forgetting and revision preserve
    the provenance chain of any unit that remains reachable, so this travels
    with every value and is never dropped on supersession.
    """

    source_external_ids: tuple[str, ...] = ()
    source_unit_ids: tuple[int, ...] = ()
    extractor_model: str | None = None
    prompt_version: str | None = None
    confidence: float | None = None
    operator: str | None = None  # which GEM operator wrote it
    transition_id: int | None = None  # gem_transition.t that committed it


@dataclass(frozen=True)
class FieldValue:
    """One entry of a field's value history H_{i,j} = <(v, t, pi)>.

    Updates APPEND rather than overwrite. ``valid_to is None`` marks the
    current value; the schema carries a partial unique index so that at most
    one current value per (unit, field) can exist, which is the engine-level
    mechanism Observation 3a says append-only stores lack.
    """

    value: str
    valid_from: str
    provenance: Provenance
    valid_to: str | None = None
    superseded_by: int | None = None
    salience: float = 0.0
    state: UnitState = UnitState.ACTIVE
    value_id: int | None = None


@dataclass
class SemanticUnit:
    """The atom of D_t — MemState's "topic", self-contained by design.

    A unit groups every field of one concept in a single record. Entity-grain
    designs scatter attributes across nodes and need multiple accesses to
    reconstruct one concept (paper §4.1); the whole point of the topic grain is
    that the C3 propagation frontier stays small.

    ``id`` MUST equal the native graph vid. tjs_open's graph leg resolves a
    reach vertex with ``SELECT <vec> FROM <tbl> WHERE <id_col> = $1`` passing
    the raw vid, so a unit is reachable through the graph only when this holds.
    """

    scope_id: str
    title: str
    summary: str
    id: int | None = None
    embedding: Sequence[float] | None = None
    fields: Mapping[str, list[FieldValue]] = field(default_factory=dict)
    state: UnitState = UnitState.ACTIVE
    salience: float = 0.0
    access_count: int = 0
    last_access: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def current(self, field_name: str) -> FieldValue | None:
        """The value C1 calls "current": the most recent non-archived entry."""
        history = self.fields.get(field_name) or []
        live = [
            value
            for value in history
            if value.valid_to is None and value.state is not UnitState.ARCHIVED
        ]
        return live[-1] if live else None


# ---------------------------------------------------------------------------
# S_t — structure
# ---------------------------------------------------------------------------


class EdgeKind(str, Enum):
    """The two edge kinds of paper §4.1, distinguished by PROPAGATION rights.

    The distinction is load-bearing for C3, not cosmetic: "C3 propagation must
    follow entailment, not relatedness, so revision traverses extension edges
    only. Association edges support retrieval context expansion without
    propagation."
    """

    EXTENSION = "extension"
    ASSOCIATION = "association"


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    kind: EdgeKind
    rel: str
    weight: float = 1.0
    co_access_count: int = 0
    edge_type: int | None = None  # graph_store.register_edge_type() id


# ---------------------------------------------------------------------------
# P_t — policies
# ---------------------------------------------------------------------------


class PolicyEvent(str, Enum):
    FIELD_UPDATED = "field_updated"
    UNIT_INGESTED = "unit_ingested"
    RETRIEVAL = "retrieval"
    TICK = "tick"  # scheduled maintenance


@dataclass(frozen=True)
class Policy:
    """A typed rule <event, condition, action> (Definition 3).

    Policies are declarative and live INSIDE M_t: their conditions reference
    the state directly, and the postcondition on a proposed M_{t+1} is
    evaluated before commit. A violating transition is rejected, which is what
    lifts C2 to a data-model-level guarantee.
    """

    name: str
    event: PolicyEvent
    condition: Mapping[str, Any]
    action: Mapping[str, Any]
    scope_id: str | None = None  # None = global
    enabled: bool = True
    version: int = 1


# ---------------------------------------------------------------------------
# Operator inputs and outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionEvent:
    """One item of the ingestion stream I_t.

    The paper's ingestion stage receives "user-assistant dialogue, tool calls,
    documents, execution traces, and environmental feedback" (Omri §2.1); the
    unit of processing is a strategy decision, not an input property.
    """

    scope_id: str
    external_id: str
    content: str
    session_id: str | None = None
    role: str | None = None
    kind: str = "turn"
    event_time: str | None = None
    event_order: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RetrievalMode(str, Enum):
    """Which stores participate. FUSED is the tri-modal path (tjs_open)."""

    VECTOR = "vector"
    GRAPH = "graph"
    RELATIONAL = "relational"
    FUSED = "fused"


class RetrievalRoute(str, Enum):
    """MemState's three routing modes (paper §4.2 "Retrieval")."""

    TOPIC = "topic"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class Query:
    scope_id: str
    text: str | None = None
    embedding: Sequence[float] | None = None
    k: int = 10
    mode: RetrievalMode = RetrievalMode.FUSED
    route: RetrievalRoute = RetrievalRoute.TOPIC
    as_of: str | None = None  # C1: historical values only when explicitly asked
    include_history: bool = False
    include_archived: bool = False
    # C6 is ON by default. Setting it False yields a READ-ONLY retrieval, which
    # is not GEM-conformant — it exists so the cost of C6 can be ablated and
    # so Paradigm I/II reproductions stay faithful to their originals.
    reinforce: bool = True
    hops: int = 1
    m_seeds: int = 4
    term_cond: int = 32
    anchor_id: int | None = None
    # Retrieval expands context over association edges.  ``None`` is the
    # explicit diagnostic opt-in to ANY edge type; it must never be the
    # benchmark default because extension edges carry propagation rights.
    graph_edge_kind: EdgeKind | None = EdgeKind.ASSOCIATION
    extra_filter: str | None = None


@dataclass(frozen=True)
class Hit:
    unit_id: int
    title: str
    field_name: str | None
    value: str | None
    score: float
    state: UnitState
    salience_before: float | None = None
    salience_after: float | None = None
    via: str = "vector"  # vector | graph | relational — which leg surfaced it


@dataclass(frozen=True)
class StateDelta:
    """What a transition changed. The trajectory {M_t} is the correctness
    object, so every operator reports its delta rather than only its output."""

    units_created: int = 0
    units_updated: int = 0
    units_archived: int = 0
    fields_appended: int = 0
    values_superseded: int = 0
    edges_created: int = 0
    edges_tombstoned: int = 0
    salience_updates: int = 0
    propagated_units: tuple[int, ...] = ()
    active_units: int | None = None  # C5 |D_t^active| after the transition
    active_fields: int | None = None


@dataclass(frozen=True)
class PhaseCost:
    """Omri et al. §3.3 phase-aware telemetry, per operator invocation.

    Attributes cost to construction / retrieval / generation on one monotonic
    timeline so a GEM run is directly comparable with the paper's Table 3.
    ``gpu_joules`` stays None until the NVML sampler lands.
    """

    phase: str
    seconds: float
    llm_calls: int = 0
    embed_calls: int = 0
    embed_sequences: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_input_tokens: int = 0
    db_statements: int = 0
    gpu_joules: float | None = None


@dataclass(frozen=True)
class TransitionResult:
    """Common envelope: output (if any), the state delta, and the cost."""

    operator: str
    committed: bool
    delta: StateDelta
    cost: PhaseCost
    transition_id: int | None = None
    policies_evaluated: tuple[str, ...] = ()
    aborted_reason: str | None = None


@dataclass(frozen=True)
class IngestResult(TransitionResult):
    units: tuple[int, ...] = ()
    rejected: tuple[Mapping[str, Any], ...] = ()  # schema-gate failures (Omri §4.4)
    # Omri Recommendation 10: LLM-bounded phases need external iteration caps,
    # and a capped run is a RECORDED operating point, never a silent
    # truncation. Set when an agentic strategy exhausted max_rounds or
    # max_tool_calls; the operator still commits what exists.
    capped: bool = False


@dataclass(frozen=True)
class RetrievalResult(TransitionResult):
    """Retrieval returns an output AND induces a transition (Observation 1).

    ``probes`` carries the engine's honesty counters verbatim; a censored or
    right-censored run is a different operating point, not a faster exact one.
    """

    hits: tuple[Hit, ...] = ()
    probes: Mapping[str, Any] = field(default_factory=dict)
    prompt_block: str | None = None


@dataclass(frozen=True)
class RevisionResult(TransitionResult):
    repairs: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ForgetResult(TransitionResult):
    demoted: tuple[Mapping[str, Any], ...] = ()
