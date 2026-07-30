"""``TriDBGovernedMemory`` — the four operators wired to one store.

Implements the ``GovernedMemory`` Protocol. Everything interesting lives in the
operator modules; this is the assembly point and the place where the
configuration matrix (interface doc §8) becomes real settings:

===========================  ========  ======  ======  ======  =========
[AM] system / paradigm       ingest    mode    revise  forget  reinforce
===========================  ========  ======  ======  ======  =========
II embedRAG (today)          determ.   VECTOR  off     off     off
III.a GraphRAG-like          llm/batch FUSED   off     off     off
III.b Mem0-like              llm/seq   VECTOR  conflct off     off
IV agentic                   agentic   FUSED   on      on      off
**GEM-conformant**           any       FUSED   **on**  **on**  **on**
===========================  ========  ======  ======  ======  =========

The last row is the position no system in [AM]'s Table 1 occupies and no
paradigm in [GEM]'s Table 1 covers.

**Honesty gate.** ``reinforce`` and ``forget`` are recorded in every run
manifest by :meth:`TriDBGovernedMemory.manifest`. A run with either on is not
comparable to a paper row that had neither, and until C1–C6 all hold the label
is ``TriDB-vector`` / Paradigm II embedRAG — never "GEM-conformant".
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from bench.agent_memory.gem.forget import ForgetOperator
from bench.agent_memory.gem.ingest import IngestOperator
from bench.agent_memory.gem.policy import PolicyEngine
from bench.agent_memory.gem.retrieve import RetrieveOperator
from bench.agent_memory.gem.revise import ReviseOperator
from bench.agent_memory.gem.salience import ExponentialSalience
from bench.agent_memory.gem.store import DEFAULT_DIM, DEFAULT_DSN, GemStore
from bench.agent_memory.gem.types import (
    ForgetResult,
    IngestResult,
    InteractionEvent,
    Policy,
    Query,
    RetrievalResult,
    RevisionResult,
)


class TriDBGovernedMemory:
    """``M_t = (D_t, S_t, P_t)`` with four state-level operators on TriDB."""

    def __init__(
        self,
        store: GemStore,
        *,
        embedder: Any | None = None,
        salience: ExponentialSalience | None = None,
        policy_engine: PolicyEngine | None = None,
        max_rejection_rate: float | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.salience = salience or ExponentialSalience()
        self.policy_engine = policy_engine or PolicyEngine()
        self.max_rejection_rate = max_rejection_rate

        # The policy engine becomes the store's commit postcondition. This is
        # what lifts C2 from convention to a data-model guarantee: a violating
        # transition raises inside the transaction, so rollback IS the
        # enforcement mechanism.
        store.postcondition_hook = self.policy_engine.evaluate

        self._ingest = IngestOperator(store, embedder=embedder)
        self._retrieve = RetrieveOperator(
            store, embedder=embedder, salience=self.salience
        )
        self._revise = ReviseOperator(store, embedder=embedder)
        self._forget = ForgetOperator(store, salience=self.salience)

    @classmethod
    def connect(
        cls,
        dsn: str = DEFAULT_DSN,
        *,
        dim: int = DEFAULT_DIM,
        embedder: Any | None = None,
        salience: ExponentialSalience | None = None,
        policy_engine: PolicyEngine | None = None,
        max_rejection_rate: float | None = None,
    ) -> TriDBGovernedMemory:
        store = GemStore.connect(dsn, dim=dim)
        return cls(
            store,
            embedder=embedder,
            salience=salience,
            policy_engine=policy_engine,
            max_rejection_rate=max_rejection_rate,
        )

    def init_schema(self) -> dict[str, Any]:
        return self.store.init_schema()

    def close(self) -> None:
        self.store.close()

    # -- the four operators ----------------------------------------------

    def ingest(
        self,
        events: Sequence[InteractionEvent],
        *,
        strategy: Any,
        scope_id: str | None = None,
    ) -> IngestResult:
        return self._ingest.ingest(
            events,
            strategy=strategy,
            scope_id=scope_id,
            max_rejection_rate=self.max_rejection_rate,
        )

    def revise(
        self,
        scope_id: str,
        *,
        evidence: Sequence[Mapping[str, Any]] | None = None,
        max_hops: int = 3,
    ) -> RevisionResult:
        return self._revise.revise(scope_id, evidence=evidence, max_hops=max_hops)

    def forget(self, scope_id: str, *, now: str | None = None) -> ForgetResult:
        return self._forget.forget(scope_id, now=now)

    def retrieve(self, query: Query) -> RetrievalResult:
        return self._retrieve.retrieve(query)

    # -- P_t administration and the trajectory ----------------------------

    def put_policy(self, policy: Policy) -> None:
        self.store.put_policy(policy)

    def policies(self, scope_id: str | None = None) -> list[Policy]:
        from bench.agent_memory.gem.store import TriDBMemoryView

        return TriDBMemoryView(self.store).policies(scope_id)

    def trajectory(self, scope_id: str, *, since: int = 0) -> list[dict[str, Any]]:
        return self.store.trajectory(scope_id, since=since)

    # -- the honesty gate --------------------------------------------------

    def manifest(
        self,
        *,
        strategy_name: str,
        mode: str,
        route: str,
        reinforce: bool,
        revise_enabled: bool,
        forget_enabled: bool,
        embedding_model: str,
    ) -> dict[str, Any]:
        """The configuration record that must accompany every reported run.

        Names the operating point rather than letting a reader assume one. The
        two entries that matter most are ``reinforce`` and ``forget``: a run
        with either on is a different operating point from the paper rows,
        and ``conformance`` is deliberately per-condition, never a wholesale
        "GEM-conformant" claim.
        """
        return {
            "ingest_strategy": strategy_name,
            "retrieval_mode": mode,
            "retrieval_route": route,
            "reinforce": reinforce,
            "revise": revise_enabled,
            "forget": forget_enabled,
            "embedding_model": embedding_model,
            "salience": {
                "gain": self.salience.gain,
                "floor": self.salience.floor,
                "lambda": self.salience.lam,
                "theta_summary": self.salience.theta_summary,
                "theta_remove": self.salience.theta_remove,
                "theta_archive": self.salience.theta_archive,
                # Decay is lazy: salience is only current as of the last tick.
                "decay": "lazy, computed at forget tick time",
            },
            # Graph reads are commit-visible, not snapshot-isolated
            # (interface doc §6.5). A conformance claim that depends on
            # repeatable-read topology cannot be made on this engine today.
            "graph_read_visibility": "commit_visible",
            "paper_hardware_match": False,
        }
