"""Independent six-arm reference semantics and mandatory leakage gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, Sequence

from bench.agent_memory.memoryarena.dataset import MemoryArenaSession, MemoryArenaTask


class Arm(StrEnum):
    MEMORY_OFF = "memory_off"
    RECENT_FIFO = "recent_fifo"
    VECTOR_ONLY = "vector_only"
    GRAPH_RELATIONAL = "graph_relational"
    GEM_FUSED = "gem_fused"
    ORACLE = "oracle"


@dataclass(frozen=True)
class DecisionPoint:
    task_uid: str
    target_session_uid: str
    cutoff_ordinal: int
    query: str
    relevant_session_uids: frozenset[str]
    dependency_basis: str
    relevance_available: bool = True

    @classmethod
    def from_session(cls, session: MemoryArenaSession) -> "DecisionPoint":
        return cls(
            task_uid=session.task_uid,
            target_session_uid=session.session_uid,
            cutoff_ordinal=session.cutoff_ordinal,
            query=session.question,
            relevant_session_uids=frozenset(session.protocol_dependencies),
            dependency_basis=session.dependency_basis,
        )


@dataclass(frozen=True)
class Candidate:
    session_uid: str
    task_uid: str
    ordinal: int
    semantic_score: float = 0.0
    semantic_score_source: str = "not_observed"
    graph_distance: int | None = None
    scope_allowed: bool = True
    is_valid: bool = True
    is_stale: bool = False
    is_harmful: bool = False


@dataclass(frozen=True)
class Selection:
    arm: Arm
    selected: tuple[Candidate, ...]
    mandatory_eligible: tuple[Candidate, ...]
    modality_eligible: tuple[Candidate, ...]


def candidates_for_task(
    task: MemoryArenaTask,
    *,
    semantic_scores: Mapping[str, float] | None = None,
    semantic_score_sources: Mapping[str, str] | None = None,
    graph_distances: Mapping[str, int] | None = None,
    stale: frozenset[str] = frozenset(),
    harmful: frozenset[str] = frozenset(),
    invalid: frozenset[str] = frozenset(),
) -> tuple[Candidate, ...]:
    semantic_scores = semantic_scores or {}
    semantic_score_sources = semantic_score_sources or {}
    graph_distances = graph_distances or {}
    return tuple(
        Candidate(
            session_uid=session.session_uid,
            task_uid=session.task_uid,
            ordinal=session.ordinal,
            semantic_score=float(semantic_scores.get(session.session_uid, 0.0)),
            semantic_score_source=str(
                semantic_score_sources.get(session.session_uid, "not_observed")
            ),
            graph_distance=graph_distances.get(session.session_uid),
            is_valid=session.session_uid not in invalid,
            is_stale=session.session_uid in stale,
            is_harmful=session.session_uid in harmful,
        )
        for session in task.sessions
    )


def mandatory_eligible(
    decision: DecisionPoint, candidates: Sequence[Candidate]
) -> tuple[Candidate, ...]:
    """Governance/cutoff constraints that no ablation is allowed to remove."""
    return tuple(
        candidate
        for candidate in candidates
        if candidate.task_uid == decision.task_uid
        and candidate.session_uid != decision.target_session_uid
        and candidate.ordinal < decision.cutoff_ordinal
        and candidate.scope_allowed
    )


def select(
    arm: Arm,
    decision: DecisionPoint,
    candidates: Sequence[Candidate],
    *,
    top_k: int = 10,
) -> Selection:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    base = mandatory_eligible(decision, candidates)

    if arm is Arm.MEMORY_OFF:
        modality = ()
        selected = ()
    elif arm is Arm.RECENT_FIFO:
        modality = base
        selected = tuple(
            sorted(base, key=lambda item: (-item.ordinal, item.session_uid))
        )
    elif arm is Arm.VECTOR_ONLY:
        # Cutoff/scope remain mandatory. Validity and freshness are the relational
        # modality under test, so this arm intentionally does not filter them.
        modality = base
        selected = tuple(
            sorted(base, key=lambda item: (-item.semantic_score, item.session_uid))
        )
    elif arm is Arm.GRAPH_RELATIONAL:
        modality = tuple(
            item
            for item in base
            if item.graph_distance is not None and item.is_valid and not item.is_stale
        )
        selected = tuple(
            sorted(
                modality,
                key=lambda item: (
                    item.graph_distance,
                    -item.ordinal,
                    item.session_uid,
                ),
            )
        )
    elif arm is Arm.GEM_FUSED:
        modality = tuple(
            item
            for item in base
            if item.graph_distance is not None and item.is_valid and not item.is_stale
        )
        selected = tuple(
            sorted(
                modality,
                key=lambda item: (
                    -item.semantic_score,
                    item.graph_distance,
                    item.session_uid,
                ),
            )
        )
    elif arm is Arm.ORACLE:
        if not decision.relevance_available:
            raise ValueError(
                "oracle arm requires released or independently annotated relevance"
            )
        modality = tuple(
            item for item in base if item.session_uid in decision.relevant_session_uids
        )
        selected = tuple(sorted(modality, key=lambda item: item.ordinal))
    else:  # pragma: no cover - StrEnum exhaustiveness guard
        raise ValueError(f"unknown arm: {arm}")

    return Selection(
        arm=arm,
        selected=selected[:top_k],
        mandatory_eligible=base,
        modality_eligible=modality,
    )


def leakage_violations(
    decision: DecisionPoint, selected: Sequence[Candidate]
) -> tuple[str, ...]:
    violations: list[str] = []
    for item in selected:
        if item.session_uid == decision.target_session_uid:
            violations.append(f"target_session:{item.session_uid}")
        if item.ordinal >= decision.cutoff_ordinal:
            violations.append(f"future_session:{item.session_uid}")
        if item.task_uid != decision.task_uid:
            violations.append(f"cross_task:{item.session_uid}")
        if not item.scope_allowed:
            violations.append(f"scope_denied:{item.session_uid}")
    return tuple(violations)


def assert_no_leakage(decision: DecisionPoint, selected: Sequence[Candidate]) -> None:
    violations = leakage_violations(decision, selected)
    if violations:
        raise ValueError("retrieval leakage: " + ", ".join(violations))
