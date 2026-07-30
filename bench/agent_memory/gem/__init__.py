"""Governed Evolving Memory (GEM) interfaces for TriDB.

Two papers meet here:

* Orogat & Mansour, *Is Agent Memory a Database?* (arXiv:2605.26252v1) supplies
  the ABSTRACTION — state M_t = (D_t, S_t, P_t), four state-level operators
  replacing record-level CRUD, and six correctness conditions over the state
  trajectory {M_t}.
* Omri et al., *Agent Memory: Characterization and System Implications of
  Stateful Long-Horizon Workloads* (arXiv:2606.06448v1) supplies the WORKLOAD —
  four construction forms, a phase-aware cost model, and the experiment suite
  this package is meant to make reproducible on TriDB.

The two fit together cleanly: GEM's ``ingest`` strategy slot is exactly Omri's
construction-form axis, so one memory implementation with swappable strategies
yields the paper's taxonomy as CONFIGURATIONS rather than as separate systems.

This package currently defines types and protocols only — no behaviour. See
``docs/agent_memory_gem_interface_v0.1.0.md`` for the implementation plan, the
configuration matrix, and what each milestone unblocks.
"""

from bench.agent_memory.gem.protocols import (
    AgenticIngest,
    DeterministicIngest,
    GovernedMemory,
    IngestStrategy,
    LLMMediatedIngest,
    MemoryView,
    SaliencePolicy,
)
from bench.agent_memory.gem.types import (
    Edge,
    EdgeKind,
    FieldValue,
    ForgetResult,
    Hit,
    IngestResult,
    InteractionEvent,
    PhaseCost,
    Policy,
    PolicyEvent,
    Provenance,
    Query,
    RetrievalMode,
    RetrievalResult,
    RetrievalRoute,
    RevisionResult,
    SemanticUnit,
    StateDelta,
    TransitionResult,
    UnitState,
)

__all__ = [
    "AgenticIngest",
    "DeterministicIngest",
    "Edge",
    "EdgeKind",
    "FieldValue",
    "ForgetResult",
    "GovernedMemory",
    "Hit",
    "IngestResult",
    "IngestStrategy",
    "InteractionEvent",
    "LLMMediatedIngest",
    "MemoryView",
    "PhaseCost",
    "Policy",
    "PolicyEvent",
    "Provenance",
    "Query",
    "RetrievalMode",
    "RetrievalResult",
    "RetrievalRoute",
    "RevisionResult",
    "SaliencePolicy",
    "SemanticUnit",
    "StateDelta",
    "TransitionResult",
    "UnitState",
]
