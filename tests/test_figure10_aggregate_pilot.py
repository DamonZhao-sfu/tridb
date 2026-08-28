import statistics

import pytest

from experiments.figure10.aggregate_pilot import (
    cluster_bootstrap_ci,
    distribution,
    percentile,
)


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_distribution_keeps_count_and_tail() -> None:
    result = distribution([1.0, 2.0, 5.0])
    assert result["count"] == 3
    assert result["mean"] == pytest.approx(8 / 3)
    assert result["p50"] == 2.0
    assert result["max"] == 5.0


def test_bootstrap_resamples_histories_not_individual_queries() -> None:
    rows = [
        {"history_index": 0, "value": 0.0},
        {"history_index": 0, "value": 0.0},
        {"history_index": 1, "value": 10.0},
        {"history_index": 1, "value": 10.0},
    ]
    first = cluster_bootstrap_ci(
        rows,
        lambda row: row["value"],
        statistics.fmean,
        iterations=500,
        seed=7,
    )
    second = cluster_bootstrap_ci(
        rows,
        lambda row: row["value"],
        statistics.fmean,
        iterations=500,
        seed=7,
    )
    assert first == second
    assert first == (0.0, 10.0)
