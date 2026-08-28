"""Fail-closed retrieve → inject → answer → update state machine.

The driver is deliberately agent- and database-agnostic.  Live TriDB, polyglot, and
exact-reference backends must all pass through the same cutoff, budget, and receipt
logic, so a backend cannot gain quality by quietly injecting more or newer memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from bench.agent_memory.memoryarena.dataset import MemoryArenaSession
from bench.agent_memory.memoryarena.oracle import (
    Arm,
    Candidate,
    DecisionPoint,
    assert_no_leakage,
)
from bench.agent_memory.memoryarena.receipts import RetrievalReceipt, build_receipt


@dataclass(frozen=True)
class RetrievedItem:
    candidate: Candidate
    rendered_text: str


@dataclass(frozen=True)
class BackendSearch:
    snapshot_id: str
    all_candidates: tuple[Candidate, ...]
    ranked_items: tuple[RetrievedItem, ...]
    first_row_ms: float | None
    time_to_k_ms: float
    candidates_examined: int | None
    visited_nodes: int | None
    visited_edges: int | None
    termination_reason: str


class CrossSessionBackend(Protocol):
    def retrieve(
        self,
        arm: Arm,
        decision: DecisionPoint,
        *,
        top_k: int,
    ) -> BackendSearch: ...

    def update(
        self,
        session: MemoryArenaSession,
        *,
        response: str,
        retrieval_receipt: RetrievalReceipt,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class OpenedSession:
    session: MemoryArenaSession
    decision: DecisionPoint
    injection_text: str
    receipt: RetrievalReceipt


class CrossSessionDriver:
    """Enforce serial session replay and post-answer memory admission."""

    def __init__(
        self,
        backend: CrossSessionBackend,
        *,
        run_id: str,
        dataset_manifest_sha256: str,
        arm: Arm,
        top_k: int = 10,
        injection_token_budget: int = 4096,
        count_tokens: Callable[[str], int],
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if injection_token_budget < 0:
            raise ValueError("injection_token_budget must be non-negative")
        self.backend = backend
        self.run_id = run_id
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.arm = arm
        self.top_k = top_k
        self.injection_token_budget = injection_token_budget
        self.count_tokens = count_tokens
        self._next_ordinal: dict[str, int] = {}
        self._pending: OpenedSession | None = None

    def open_session(self, session: MemoryArenaSession) -> OpenedSession:
        if self._pending is not None:
            raise RuntimeError(
                f"session {self._pending.session.session_uid} must complete before "
                f"opening {session.session_uid}"
            )
        expected = self._next_ordinal.get(session.task_uid, 0)
        if session.ordinal != expected:
            raise ValueError(
                f"non-serial session for {session.task_uid}: got {session.ordinal}, "
                f"expected {expected}"
            )
        decision = DecisionPoint.from_session(session)
        search = self.backend.retrieve(self.arm, decision, top_k=self.top_k)
        ranked = search.ranked_items[: self.top_k]
        assert_no_leakage(decision, [item.candidate for item in ranked])

        accepted: list[RetrievedItem] = []
        injection_text = ""
        for item in ranked:
            trial = "\n\n".join(
                [*(entry.rendered_text for entry in accepted), item.rendered_text]
            )
            if self.count_tokens(trial) <= self.injection_token_budget:
                accepted.append(item)
                injection_text = trial

        injection_tokens = self.count_tokens(injection_text)
        receipt = build_receipt(
            run_id=self.run_id,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            snapshot_id=search.snapshot_id,
            arm=self.arm,
            decision=decision,
            all_candidates=search.all_candidates,
            selected=[item.candidate for item in accepted],
            top_k=self.top_k,
            injection_token_budget=self.injection_token_budget,
            injection_tokens=injection_tokens,
            injection_text=injection_text,
            first_row_ms=search.first_row_ms,
            time_to_k_ms=search.time_to_k_ms,
            candidates_examined=search.candidates_examined,
            visited_nodes=search.visited_nodes,
            visited_edges=search.visited_edges,
            termination_reason=search.termination_reason,
        )
        opened = OpenedSession(session, decision, injection_text, receipt)
        self._pending = opened
        return opened

    def complete_session(
        self, session: MemoryArenaSession, *, response: str
    ) -> Mapping[str, Any]:
        opened = self._pending
        if opened is None:
            raise RuntimeError("cannot update memory before retrieval/open_session")
        if opened.session.session_uid != session.session_uid:
            raise ValueError(
                f"pending session is {opened.session.session_uid}, not "
                f"{session.session_uid}"
            )
        result = self.backend.update(
            session,
            response=response,
            retrieval_receipt=opened.receipt,
        )
        self._next_ordinal[session.task_uid] = session.ordinal + 1
        self._pending = None
        return result
