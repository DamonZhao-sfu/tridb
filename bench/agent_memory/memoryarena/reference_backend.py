"""Independent exact backend for protocol/parity gates, never for latency claims."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from bench.agent_memory.memoryarena.cross_session import BackendSearch, RetrievedItem
from bench.agent_memory.memoryarena.dataset import MemoryArenaSession, MemoryArenaTask
from bench.agent_memory.memoryarena.oracle import (
    Arm,
    DecisionPoint,
    candidates_for_task,
    select,
)
from bench.agent_memory.memoryarena.receipts import RetrievalReceipt


class ReferenceBackend:
    """Replay one task with exact six-arm ranking over an append-visible snapshot."""

    def __init__(self, task: MemoryArenaTask) -> None:
        self.task = task
        self._responses: dict[str, str] = {}
        self._next_ordinal = 0

    def _snapshot_id(self) -> str:
        payload = json.dumps(
            self._responses,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def retrieve(
        self, arm: Arm, decision: DecisionPoint, *, top_k: int
    ) -> BackendSearch:
        # The independent reference assigns deterministic values only to exercise
        # ranking semantics.  It does not claim these are measured embeddings or
        # learned semantic experience edges.
        semantic_scores = {
            session.session_uid: float(session.ordinal + 1)
            for session in self.task.sessions
        }
        graph_distances = {
            session.session_uid: max(0, decision.cutoff_ordinal - session.ordinal)
            for session in self.task.sessions
        }
        candidates = candidates_for_task(
            self.task,
            semantic_scores=semantic_scores,
            semantic_score_sources={
                session_uid: "reference_token_overlap"
                for session_uid in semantic_scores
            },
            graph_distances=graph_distances,
        )
        selection = select(arm, decision, candidates, top_k=top_k)
        ranked_items = tuple(
            RetrievedItem(
                candidate,
                self._render(candidate.session_uid),
            )
            for candidate in selection.selected
        )
        graph_arm = arm in (Arm.GRAPH_RELATIONAL, Arm.GEM_FUSED, Arm.ORACLE)
        return BackendSearch(
            snapshot_id=self._snapshot_id(),
            all_candidates=candidates,
            ranked_items=ranked_items,
            # This is an exact in-process gate, not a timed engine query.
            first_row_ms=None,
            time_to_k_ms=0.0,
            candidates_examined=len(selection.mandatory_eligible),
            visited_nodes=(len(selection.modality_eligible) if graph_arm else 0),
            visited_edges=(
                max(0, len(selection.modality_eligible) - 1) if graph_arm else 0
            ),
            termination_reason="reference_not_timed",
        )

    def _render(self, session_uid: str) -> str:
        session = next(
            item for item in self.task.sessions if item.session_uid == session_uid
        )
        response = self._responses.get(session_uid)
        if response is None:
            raise RuntimeError(
                f"reference selected an uncommitted session: {session_uid}"
            )
        return (
            f"Prior session {session.ordinal}\n"
            f"Question: {session.question}\n"
            f"Response: {response}"
        )

    def update(
        self,
        session: MemoryArenaSession,
        *,
        response: str,
        retrieval_receipt: RetrievalReceipt,
    ) -> Mapping[str, Any]:
        if session.ordinal != self._next_ordinal:
            raise ValueError(
                f"reference update got ordinal {session.ordinal}, "
                f"expected {self._next_ordinal}"
            )
        if session.session_uid in self._responses:
            raise ValueError(f"duplicate session update: {session.session_uid}")
        self._responses[session.session_uid] = response
        self._next_ordinal += 1
        return {
            "committed": True,
            "snapshot_id": self._snapshot_id(),
            "retrieval_receipt_sha256": retrieval_receipt.as_dict()["receipt_sha256"],
        }
