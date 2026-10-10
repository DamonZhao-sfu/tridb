"""The four GEM operators and the pluggable strategies behind them.

Orogat & Mansour (arXiv:2605.26252v1) §3.2 replaces record-level CRUD with
four STATE-LEVEL operators:

    ingestion   M_{t+1} = U(M_t, I_t, {})        integrate, do not append
    revision    M_{t+1} = U(M_t, {}, rev(delta)) reconcile + propagate
    forgetting  M_{t+1} = U(M_t, {}, F)          graded, relevance-driven
    retrieval   R(M_t, q) -> o  AND  M_{t+1} = U(M_t, {}, R_q)

The last one is the structural claim: a pure-function retrieval CANNOT satisfy
C6, so ``retrieve`` here returns an output *and* commits a state transition.
Ablating that (``Query.reinforce = False``) yields a non-conformant read-only
mode, kept only so Paradigm I/II reproductions stay faithful.

Ingestion is deliberately factored into a STRATEGY because Omri et al.
(arXiv:2606.06448v1) §2.1 name four construction forms — absent, deterministic,
LLM-mediated, agentic — and their whole systems characterisation is the claim
that the choice moves order-of-magnitude cost between the write and read paths.
One memory, swappable construction, is what makes that measurable on TriDB.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from bench.agent_memory.gem.types import (
    Edge,
    ForgetResult,
    IngestResult,
    InteractionEvent,
    Policy,
    Query,
    RetrievalResult,
    RevisionResult,
    SemanticUnit,
)

# ---------------------------------------------------------------------------
# Read-side view handed to strategies (they plan; the operator commits)
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryView(Protocol):
    """A read-only window on M_t.

    Strategies receive this rather than the store itself: an ingest strategy
    must be able to SEE the current state (MemState's LLM "reads topic titles
    and summaries to select a host topic") but must not write. All writes go
    through the operator so that P_t is evaluated once, at commit.
    """

    def units(self, scope_id: str, *, limit: int = 100) -> Sequence[SemanticUnit]: ...

    def unit(self, unit_id: int) -> SemanticUnit | None: ...

    def unit_by_title(self, scope_id: str, title: str) -> SemanticUnit | None: ...

    def latest_experience_before(
        self, scope_id: str, ordinal: int
    ) -> SemanticUnit | None: ...

    def find_similar(
        self, scope_id: str, embedding: Sequence[float], *, k: int = 10
    ) -> Sequence[SemanticUnit]: ...

    def edges(self, unit_id: int) -> Sequence[Edge]: ...

    def policies(self, scope_id: str | None = None) -> Sequence[Policy]: ...


# ---------------------------------------------------------------------------
# Write plans — what a strategy proposes, never applies
# ---------------------------------------------------------------------------


class WriteOp(Protocol):
    """Marker for the plan vocabulary below."""

    kind: str


class UpsertUnit(Protocol):
    kind: str  # "upsert_unit"
    scope_id: str
    title: str
    summary: str
    unit_id: int | None


class AppendFieldValue(Protocol):
    kind: str  # "append_field_value"
    unit_id: int | None
    field: str
    value: str
    valid_from: str
    supersede_current: bool  # True = UPDATE semantics; the old value is kept


class LinkUnits(Protocol):
    kind: str  # "link"
    src: int
    dst: int
    edge_kind: str  # extension | association
    rel: str


class SplitTopic(Protocol):
    """Promote a subset of fields into a standalone unit (paper Figure 3).

    "Alice splits out of Website Redesign once enough interactions reference
    her directly." Available to revision and to agentic ingest.
    """

    kind: str  # "split_topic"
    unit_id: int
    fields: Sequence[str]
    new_title: str


# ---------------------------------------------------------------------------
# Ingest strategies — Omri's construction forms
# ---------------------------------------------------------------------------


@runtime_checkable
class IngestStrategy(Protocol):
    """Turn raw interaction events into a proposed write plan.

    ``name`` is recorded in the run manifest and is the taxonomy label, so it
    must be one of the paper's construction forms.
    """

    name: str  # deterministic | llm_mediated | agentic

    def plan(
        self, events: Sequence[InteractionEvent], view: MemoryView
    ) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class DeterministicIngest(IngestStrategy, Protocol):
    """Construction WITHOUT an LLM: chunk / index only (Omri Table 1, embedRAG).

    One unit per chunk, a single ``content`` field, one batched embedding call
    per ``batch_size`` chunks. No extraction, no consolidation, no edges. This
    is TriDB's current behaviour and reproduces the paper's embedRAG row
    unchanged; keeping it behind the same interface is what makes the
    construction-cost comparison apples-to-apples.
    """

    chunk_tokens: int  # 4096 = the MemoryAgentBench streaming protocol
    batch_size: int


@runtime_checkable
class LLMMediatedIngest(IngestStrategy, Protocol):
    """LLM as a FIXED EXTRACTOR at predefined points (Omri Paradigm III).

    The LLM reads unit titles and summaries, picks a host unit, and emits facts
    against a PINNED prompt and a VERSIONED output schema. Two sub-modes are
    required because the paper's §4.3 finding is that embedding traffic is
    bimodal by paradigm and the two regimes stress a serving stack differently:

      batch (III.a, GraphRAG/HippoRAG-like)  large offline embedding batches,
                                             append-only, no consolidation
      sequential (III.b, Mem0/SimpleMem-like) embed each extracted fact before
                                             its similarity search resolves the
                                             ADD/UPDATE/DELETE decision, giving
                                             a 1:1 call-to-sequence ratio

    ``validate`` is not optional. Omri §4.4 shows that below an
    algorithm-specific capability floor a weak construction model does not
    merely lower accuracy, it CORRUPTS the store (MIRIX fails outright at
    Qwen3-1.7B). A unit that fails schema or referential validation must be
    rejected and rolled back, and counted in ``IngestResult.rejected`` — a
    structural failure is a failed configuration, not an accuracy datapoint.
    """

    mode: str  # batch | sequential
    model: str
    prompt_version: str
    schema_version: str

    def validate(self, extracted: Mapping[str, Any]) -> tuple[bool, str | None]: ...


@runtime_checkable
class AgenticIngest(IngestStrategy, Protocol):
    """LLM-CONTROLLED writes: the model decides when and what to write.

    Paradigm IV. The model is handed memory tools and loops until it stops.
    Because the specification permits arbitrary depth, ``max_rounds`` and
    ``max_tool_calls`` are mandatory: Omri Recommendation 10 is that
    LLM-bounded phases need external iteration caps, and their tails
    (p95/p50 up to 5.9x) are the evidence.
    """

    model: str
    max_rounds: int
    max_tool_calls: int
    tools: Sequence[str]  # search_memory, read_unit, write_field, link, split_topic


# ---------------------------------------------------------------------------
# Salience — the C5/C6 coupling
# ---------------------------------------------------------------------------


@runtime_checkable
class SaliencePolicy(Protocol):
    """Rises on access, decays on disuse, at SUB-UNIT granularity.

    Sub-unit matters: "part of a unit may be attenuated while the rest stays
    current", so salience is carried per field, not only per unit.

    C6 requires that repeated retrieval STRICTLY reduces a unit's eligibility
    for attenuation — ``reinforce`` must be strictly increasing in hits, and
    the forgetting thresholds must be read from the same signal.
    """

    def reinforce(self, current: float, *, rank: int, k: int) -> float: ...

    def decay(self, current: float, *, seconds_idle: float) -> float: ...

    theta_summary: float  # below -> compress the history
    theta_remove: float  # below -> hide from active retrieval
    theta_archive: float  # below -> archive, still recoverable


# ---------------------------------------------------------------------------
# The four operators
# ---------------------------------------------------------------------------


@runtime_checkable
class GovernedMemory(Protocol):
    """M_t = (D_t, S_t, P_t) with four state-level operators.

    Every operator is ONE transaction: the proposed M_{t+1} is checked against
    P_t and either commits atomically or aborts. On TriDB that transaction
    spans the relational row, the vector, and the native graph edges together —
    one transaction manager, one WAL — which is precisely the commit protocol
    the abstraction needs and the property a multi-store stack cannot offer.
    """

    # -- ingestion: integrate I_t into the existing state, do not append blindly
    def ingest(
        self,
        events: Sequence[InteractionEvent],
        *,
        strategy: IngestStrategy,
    ) -> IngestResult: ...

    # -- revision: reconcile overlapping units, propagate along EXTENSION edges
    #    only, preserve superseded values with provenance (C2, C3, C4)
    def revise(
        self,
        scope_id: str,
        *,
        evidence: Sequence[Mapping[str, Any]] | None = None,
        max_hops: int = 3,
    ) -> RevisionResult: ...

    # -- forgetting: graded attenuation by RELEVANCE, never destructive (C5)
    def forget(self, scope_id: str, *, now: str | None = None) -> ForgetResult: ...

    # -- retrieval: output + state transition (C1, C6)
    def retrieve(self, query: Query) -> RetrievalResult: ...

    # -- policy administration; P_t lives in the state
    def put_policy(self, policy: Policy) -> None: ...

    # -- trajectory access: correctness is a property of {M_t}, so the log is
    #    part of the interface, not a debugging aid
    def trajectory(
        self, scope_id: str, *, since: int = 0
    ) -> Sequence[Mapping[str, Any]]: ...
