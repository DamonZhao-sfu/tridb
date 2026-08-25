"""Retrieval quality metrics over reward-graded relevance.

Small and separate because these are the numbers a paper reports, and a reader must be
able to check the definition in one screen without reading a harness.

nDCG uses the standard exponential gain, `2**g - 1`, over grades {3,2,1,0,-1}. The
negative grade is deliberate and load-bearing: a store that confidently returns known
dead ends is worse than one that returns nothing, and a gain function floored at zero
cannot express that. `harmful_at_k` reports the same fact directly, because a single
nDCG number hides whether a low score came from missing good results or from surfacing
bad ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def gain(grade: int) -> float:
    """2**g - 1, extended to negative grades so harm actually subtracts."""
    return float(2**grade - 1) if grade >= 0 else -(2 ** abs(grade)) + 1


def dcg(grades: list[int]) -> float:
    return sum(gain(g) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(retrieved_grades: list[int], ideal_grades: list[int], k: int) -> float:
    """DCG of what was returned, over DCG of the best possible k from the same pool.

    Returns 0.0 when the ideal DCG is non-positive — that is, when the eligible pool
    holds nothing worth retrieving. Reporting 1.0 there ("we retrieved everything good,
    which was nothing") would flatter the system on exactly the queries it cannot help
    with, so those decision points are counted separately by the harness instead.
    """
    ideal = dcg(sorted(ideal_grades, reverse=True)[:k])
    if ideal <= 0:
        return 0.0
    return dcg(retrieved_grades[:k]) / ideal


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def harmful_at_k(grades: list[int], k: int) -> float:
    """Fraction of the top-k that is a known dead end."""
    window = grades[:k]
    return sum(1 for g in window if g < 0) / len(window) if window else 0.0


#: Two candidates at the same distance are equally good answers, and which one an
#: implementation returns is arbitrary. This corpus makes that structural rather than
#: rare: 10,672 nodes share only 9,733 distinct artifacts, so identical code yields
#: identical vectors and exact ties at the k-th position are common.
#:
#: Set from the two scales that bracket it, not by taste. Vectors are stored as float32,
#: so a distance is only meaningful to ~2**-24 = 6e-8, and the engine and the oracle were
#: measured to disagree by exactly that much on genuinely tied pairs. The nearest
#: *distinct* distances in this corpus differ by ~1e-3. 1e-6 therefore sits three orders
#: of magnitude above the noise and three below the signal. A tighter value (1e-9 was
#: tried) reports float32 rounding as a ranking disagreement.
TIE_EPS = 1e-6


def parity(
    observed: list[str],
    expected: tuple[str, ...],
    *,
    observed_distances: list[float] | None = None,
    expected_distances: list[float] | None = None,
) -> tuple[bool, float, bool]:
    """(exact id match, set recall, tie-equivalent) against the exact oracle.

    Three numbers because they fail differently:

    * **exact** — same ids in the same order. The strictest gate.
    * **recall** — found the right candidates, possibly in a different order.
    * **tie_equivalent** — returned an answer that is *equally good*: the sorted
      distance vectors agree to within `TIE_EPS`. This is the gate that matters when
      ties are structural, because a strict-id comparison penalises an implementation
      for breaking a tie differently, which is not an error.

    Reported side by side. Collapsing them into one number would either flatter the
    engine (tie-tolerant only) or understate it (exact only).
    """
    exact = list(observed) == list(expected)
    recall = 1.0 if not expected else len(set(observed) & set(expected)) / len(expected)
    tie_equivalent = exact
    if not exact and observed_distances is not None and expected_distances is not None:
        a, b = sorted(observed_distances), sorted(expected_distances)
        tie_equivalent = len(a) == len(b) and all(abs(x - y) <= TIE_EPS for x, y in zip(a, b))
    return exact, recall, tie_equivalent


@dataclass
class Accumulator:
    """Running totals for one (query, operating point) cell."""

    n: int = 0
    n_scored: int = 0
    n_empty_ideal: int = 0
    ceiling_sum: float = 0.0
    parity_exact: int = 0
    parity_tie_equivalent: int = 0
    recall_sum: float = 0.0
    ndcg_sum: float = 0.0
    harmful_sum: float = 0.0
    trivial_hits: int = 0
    first_row_ms: list[float] = None  # type: ignore[assignment]
    total_ms: list[float] = None  # type: ignore[assignment]
    examined: list[int] = None  # type: ignore[assignment]
    stage1_ms: list[float] = None  # type: ignore[assignment]
    stage2_ms: list[float] = None  # type: ignore[assignment]
    merge_ms: list[float] = None  # type: ignore[assignment]
    first_candidate_ms: list[float] = None  # type: ignore[assignment]
    graph_examined: list[int] = None  # type: ignore[assignment]
    censored: int = 0

    def __post_init__(self) -> None:
        self.first_row_ms = []
        self.total_ms = []
        self.examined = []
        self.stage1_ms = []
        self.stage2_ms = []
        self.merge_ms = []
        self.first_candidate_ms = []
        self.graph_examined = []

    def summary(self) -> dict[str, float | int | None]:
        def pct(values: list[float], q: float) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
            return round(ordered[idx], 3)

        scored = self.n_scored or 1
        return {
            "n": self.n,
            "n_scored": self.n_scored,
            # Decision points where nothing in the pool was worth retrieving. Excluded
            # from nDCG rather than scored as 0 or 1, and reported so the denominator
            # of every quality number is visible.
            "n_empty_ideal": self.n_empty_ideal,
            "parity_exact_rate": round(self.parity_exact / (self.n or 1), 4),
            "parity_tie_equivalent_rate": round(self.parity_tie_equivalent / (self.n or 1), 4),
            "oracle_recall": round(self.recall_sum / (self.n or 1), 4),
            "ndcg": round(self.ndcg_sum / scored, 4),
            # How much of the arm-independent ideal this formulation could reach AT
            # ALL, if it ranked its own candidate pool perfectly. Separates "cannot
            # see the good nodes" from "sees them but ranks them badly".
            "ndcg_ceiling": round(self.ceiling_sum / scored, 4),
            "rank_efficiency": (
                round(self.ndcg_sum / self.ceiling_sum, 4) if self.ceiling_sum > 0 else None
            ),
            "harmful": round(self.harmful_sum / scored, 4),
            "trivial_hit_rate": round(self.trivial_hits / (self.n or 1), 4),
            "first_row_ms_p50": pct(self.first_row_ms, 0.50),
            "first_row_ms_p95": pct(self.first_row_ms, 0.95),
            "total_ms_p50": pct(self.total_ms, 0.50),
            "total_ms_p95": pct(self.total_ms, 0.95),
            "examined_p50": pct([float(x) for x in self.examined], 0.50),
            "graph_examined_p50": pct([float(x) for x in self.graph_examined], 0.50),
            # Coarse stage split. `stage1` is the ANN entry (vector leg), `stage2` the
            # fused traversal+filter+rank, `merge` the cross-entry re-rank. It does NOT
            # decompose stage2 into graph/relational/vector -- tjs_open interleaves
            # those inside one C call. That decomposition is w1_latency.py.
            "stage1_ms_p50": pct(self.stage1_ms, 0.50),
            "stage2_ms_p50": pct(self.stage2_ms, 0.50),
            "merge_ms_p50": pct(self.merge_ms, 0.50),
            "first_candidate_ms_p50": pct(self.first_candidate_ms, 0.50),
            "censored_rate": round(self.censored / (self.n or 1), 4),
        }
