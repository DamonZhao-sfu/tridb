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

See ``docs/agent_memory_gem_interface_v0.1.0.md`` for the configuration matrix
and what each milestone unblocks, and
``docs/agent_memory_gem_implementation_plan_v0.1.0.md`` for the build order.

**What holds today is per-condition, never wholesale.** Use
:mod:`~bench.agent_memory.gem.conformance` to produce the C1-C6 report; a
configuration is labelled ``TriDB-vector`` (Paradigm II embedRAG) until every
condition actually holds. ``reinforce`` and ``forget`` belong in every run
manifest — a run with either on is not comparable to a paper row that had
neither.
"""

from bench.agent_memory.gem.conformance import ConformanceReport
from bench.agent_memory.gem.forget import ForgetOperator
from bench.agent_memory.gem.ingest import IngestOperator, validate_plan
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.policy import PolicyEngine, seed_policies
from bench.agent_memory.gem.protocols import (
    AgenticIngest,
    DeterministicIngest,
    GovernedMemory,
    IngestStrategy,
    LLMMediatedIngest,
    MemoryView,
    SaliencePolicy,
)
from bench.agent_memory.gem.retrieve import RetrieveOperator
from bench.agent_memory.gem.revise import ReviseOperator
from bench.agent_memory.gem.salience import ExponentialSalience
from bench.agent_memory.gem.store import GemStore, TriDBMemoryView
from bench.agent_memory.gem.strategies import (
    AgenticIngestStrategy,
    DeterministicIngestStrategy,
    LLMMediatedIngestStrategy,
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
    "AgenticIngestStrategy",
    "ConformanceReport",
    "DeterministicIngestStrategy",
    "ExponentialSalience",
    "ForgetOperator",
    "GemStore",
    "IngestOperator",
    "LLMMediatedIngestStrategy",
    "PolicyEngine",
    "RetrieveOperator",
    "ReviseOperator",
    "TriDBGovernedMemory",
    "TriDBMemoryView",
    "seed_policies",
    "validate_plan",
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
