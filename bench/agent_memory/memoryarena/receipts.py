"""Content-addressed retrieval/injection receipts and leakage validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from bench.agent_memory.memoryarena.oracle import (
    Arm,
    Candidate,
    DecisionPoint,
    leakage_violations,
)

# v0.3 corrects the live tjs_pg probe mapping: tjs_open_graph_examined() is an
# edge-step counter, not a distinct-vertex counter.  Existing v0.2 receipts remain
# content-addressed evidence but their ``visited_nodes`` field must not be interpreted
# as nodes visited.
SCHEMA_VERSION = "memoryarena_retrieval_receipt_v0.3.0"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CandidateReceipt:
    session_uid: str
    task_uid: str
    ordinal: int
    semantic_score: float
    semantic_score_source: str
    graph_distance: int | None
    scope_allowed: bool
    is_valid: bool
    is_stale: bool
    is_harmful: bool
    selected_rank: int | None
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalReceipt:
    run_id: str
    dataset_manifest_sha256: str
    snapshot_id: str
    arm: str
    task_uid: str
    target_session_uid: str
    cutoff_ordinal: int
    query_sha256: str
    injection_sha256: str
    top_k: int
    injection_token_budget: int
    injection_tokens: int
    selected_session_uids: tuple[str, ...]
    candidates: tuple[CandidateReceipt, ...]
    first_row_ms: float | None
    time_to_k_ms: float
    candidates_examined: int | None
    visited_nodes: int | None
    visited_edges: int | None
    termination_reason: str

    def unsigned_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def as_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        return {**payload, "receipt_sha256": _digest(payload)}


def build_receipt(
    *,
    run_id: str,
    dataset_manifest_sha256: str,
    snapshot_id: str,
    arm: Arm,
    decision: DecisionPoint,
    all_candidates: Sequence[Candidate],
    selected: Sequence[Candidate],
    top_k: int,
    injection_token_budget: int,
    injection_tokens: int,
    injection_text: str,
    first_row_ms: float | None,
    time_to_k_ms: float,
    candidates_examined: int | None,
    visited_nodes: int | None,
    visited_edges: int | None,
    termination_reason: str,
) -> RetrievalReceipt:
    selected_rank = {item.session_uid: rank for rank, item in enumerate(selected, 1)}
    candidate_receipts: list[CandidateReceipt] = []
    for item in all_candidates:
        reasons: list[str] = []
        if item.ordinal >= decision.cutoff_ordinal:
            reasons.append("at_or_after_cutoff")
        if item.session_uid == decision.target_session_uid:
            reasons.append("target_session")
        if not item.scope_allowed:
            reasons.append("scope_denied")
        if not item.is_valid:
            reasons.append("invalid")
        if item.is_stale:
            reasons.append("stale")
        if item.graph_distance is None:
            reasons.append("not_graph_reachable")
        candidate_receipts.append(
            CandidateReceipt(
                session_uid=item.session_uid,
                task_uid=item.task_uid,
                ordinal=item.ordinal,
                semantic_score=item.semantic_score,
                semantic_score_source=item.semantic_score_source,
                graph_distance=item.graph_distance,
                scope_allowed=item.scope_allowed,
                is_valid=item.is_valid,
                is_stale=item.is_stale,
                is_harmful=item.is_harmful,
                selected_rank=selected_rank.get(item.session_uid),
                rejection_reasons=tuple(reasons),
            )
        )
    receipt = RetrievalReceipt(
        run_id=run_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        snapshot_id=snapshot_id,
        arm=arm.value,
        task_uid=decision.task_uid,
        target_session_uid=decision.target_session_uid,
        cutoff_ordinal=decision.cutoff_ordinal,
        query_sha256=_sha256_text(decision.query),
        injection_sha256=_sha256_text(injection_text),
        top_k=top_k,
        injection_token_budget=injection_token_budget,
        injection_tokens=injection_tokens,
        selected_session_uids=tuple(item.session_uid for item in selected),
        candidates=tuple(candidate_receipts),
        first_row_ms=first_row_ms,
        time_to_k_ms=time_to_k_ms,
        candidates_examined=candidates_examined,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        termination_reason=termination_reason,
    )
    validate_receipt(receipt, decision)
    return receipt


def validate_receipt(receipt: RetrievalReceipt, decision: DecisionPoint) -> None:
    if receipt.query_sha256 != _sha256_text(decision.query):
        raise ValueError("receipt query hash does not match decision point")
    if receipt.target_session_uid != decision.target_session_uid:
        raise ValueError("receipt target session does not match decision point")
    if receipt.cutoff_ordinal != decision.cutoff_ordinal:
        raise ValueError("receipt cutoff does not match decision point")
    if receipt.injection_tokens > receipt.injection_token_budget:
        raise ValueError("receipt exceeds injection token budget")
    if len(receipt.selected_session_uids) > receipt.top_k:
        raise ValueError("receipt selects more candidates than top_k")
    candidate_uids = [item.session_uid for item in receipt.candidates]
    if len(candidate_uids) != len(set(candidate_uids)):
        raise ValueError("receipt contains duplicate candidate ids")
    if len(receipt.selected_session_uids) != len(set(receipt.selected_session_uids)):
        raise ValueError("receipt contains duplicate selected ids")
    by_uid = {
        item.session_uid: Candidate(
            session_uid=item.session_uid,
            task_uid=item.task_uid,
            ordinal=item.ordinal,
            semantic_score=item.semantic_score,
            semantic_score_source=item.semantic_score_source,
            graph_distance=item.graph_distance,
            scope_allowed=item.scope_allowed,
            is_valid=item.is_valid,
            is_stale=item.is_stale,
            is_harmful=item.is_harmful,
        )
        for item in receipt.candidates
    }
    try:
        selected = [by_uid[uid] for uid in receipt.selected_session_uids]
    except KeyError as exc:
        raise ValueError(
            f"selected candidate absent from receipt: {exc.args[0]}"
        ) from exc
    violations = leakage_violations(decision, selected)
    if violations:
        raise ValueError("receipt contains leakage: " + ", ".join(violations))
    expected_ranks = {
        uid: rank for rank, uid in enumerate(receipt.selected_session_uids, 1)
    }
    for item in receipt.candidates:
        if item.selected_rank != expected_ranks.get(item.session_uid):
            raise ValueError("receipt candidate rank is inconsistent")


def verify_serialized_receipt(payload: Mapping[str, Any]) -> bool:
    observed = payload.get("receipt_sha256")
    if not isinstance(observed, str):
        return False
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return _digest(unsigned) == observed
