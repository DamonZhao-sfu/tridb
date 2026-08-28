"""Pure analysis primitives for E1 quality/latency comparisons."""

from __future__ import annotations

from typing import Any, Iterable


def _number(point: dict[str, Any], key: str) -> float:
    value = point.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def pareto_frontier(
    points: Iterable[dict[str, Any]],
    *,
    quality_key: str = "recall_at_20",
    latency_key: str = "latency_p50_ms",
) -> list[dict[str, Any]]:
    """Return points not dominated by both lower latency and higher quality."""
    rows = list(points)
    for row in rows:
        quality = _number(row, quality_key)
        latency = _number(row, latency_key)
        if not 0 <= quality <= 1:
            raise ValueError(f"{quality_key} must be in [0, 1]")
        if latency < 0:
            raise ValueError(f"{latency_key} must be non-negative")

    frontier = []
    for candidate in rows:
        candidate_quality = _number(candidate, quality_key)
        candidate_latency = _number(candidate, latency_key)
        dominated = any(
            _number(other, latency_key) <= candidate_latency
            and _number(other, quality_key) >= candidate_quality
            and (
                _number(other, latency_key) < candidate_latency
                or _number(other, quality_key) > candidate_quality
            )
            for other in rows
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda row: (_number(row, latency_key), -_number(row, quality_key)),
    )


def matched_quality_pairs(
    tridb_points: Iterable[dict[str, Any]],
    polyglot_points: Iterable[dict[str, Any]],
    *,
    mrr_epsilon: float,
    recall_epsilon: float,
    require_positive: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate quality-equivalent cross-system points with latency speedups."""
    if mrr_epsilon <= 0 or recall_epsilon <= 0:
        raise ValueError("quality epsilons must be positive")
    pairs = []
    for tridb in tridb_points:
        for polyglot in polyglot_points:
            mrr_delta = abs(_number(tridb, "mrr") - _number(polyglot, "mrr"))
            recall_delta = abs(
                _number(tridb, "recall_at_20") - _number(polyglot, "recall_at_20")
            )
            if mrr_delta > mrr_epsilon or recall_delta > recall_epsilon:
                continue
            tridb_mrr = _number(tridb, "mrr")
            tridb_recall = _number(tridb, "recall_at_20")
            polyglot_mrr = _number(polyglot, "mrr")
            polyglot_recall = _number(polyglot, "recall_at_20")
            if (
                require_positive
                and max(tridb_mrr, tridb_recall, polyglot_mrr, polyglot_recall) <= 0
            ):
                continue
            tridb_latency = _number(tridb, "latency_p50_ms")
            polyglot_latency = _number(polyglot, "latency_p50_ms")
            if tridb_latency <= 0:
                raise ValueError("TriDB latency must be positive for a speedup ratio")
            pairs.append(
                {
                    "tridb_point_id": tridb["point_id"],
                    "polyglot_point_id": polyglot["point_id"],
                    "mrr_delta": mrr_delta,
                    "recall_at_20_delta": recall_delta,
                    "polyglot_over_tridb_p50": polyglot_latency / tridb_latency,
                    "tridb_mrr": tridb_mrr,
                    "polyglot_mrr": polyglot_mrr,
                    "tridb_recall_at_20": tridb_recall,
                    "polyglot_recall_at_20": polyglot_recall,
                    "tridb_latency_p50_ms": tridb_latency,
                    "polyglot_latency_p50_ms": polyglot_latency,
                }
            )
    return sorted(
        pairs,
        key=lambda row: (
            row["mrr_delta"] + row["recall_at_20_delta"],
            row["tridb_point_id"],
            row["polyglot_point_id"],
        ),
    )
