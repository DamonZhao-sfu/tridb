"""MemoryArena retrieval metrics, kept separate from agent outcome metrics."""

from __future__ import annotations

import math
from typing import Sequence

from bench.agent_memory.memoryarena.oracle import Candidate, DecisionPoint


def dependency_recall_at_k(
    selected: Sequence[Candidate], decision: DecisionPoint, k: int = 10
) -> float | None:
    if not decision.relevance_available:
        return None
    relevant = decision.relevant_session_uids
    if not relevant:
        return 0.0
    observed = {item.session_uid for item in selected[:k]}
    return len(observed & relevant) / len(relevant)


def ndcg_at_k(
    selected: Sequence[Candidate], decision: DecisionPoint, k: int = 10
) -> float | None:
    if not decision.relevance_available:
        return None
    relevant = decision.relevant_session_uids
    if not relevant:
        return 0.0
    gains = [
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(selected[:k])
        if item.session_uid in relevant
    ]
    ideal_count = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return sum(gains) / ideal if ideal else 0.0


def harmful_at_k(selected: Sequence[Candidate], k: int = 10) -> float:
    window = selected[:k]
    return sum(item.is_harmful for item in window) / len(window) if window else 0.0


def stale_at_k(selected: Sequence[Candidate], k: int = 10) -> float:
    window = selected[:k]
    return sum(item.is_stale for item in window) / len(window) if window else 0.0


def constraint_valid_fraction(selected: Sequence[Candidate]) -> float:
    if not selected:
        return 1.0
    return sum(item.is_valid and not item.is_stale for item in selected) / len(selected)


def retrieval_metrics(
    selected: Sequence[Candidate], decision: DecisionPoint, k: int = 10
) -> dict[str, float | int | None]:
    return {
        "returned": len(selected),
        "dependency_recall_at_10": dependency_recall_at_k(selected, decision, k),
        "ndcg_at_10": ndcg_at_k(selected, decision, k),
        "harmful_at_10": harmful_at_k(selected, k),
        "stale_at_10": stale_at_k(selected, k),
        "constraint_valid_fraction": constraint_valid_fraction(selected),
    }
