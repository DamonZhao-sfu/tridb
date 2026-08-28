from tools.benchmark_memoryarena_fused_vs_staged import (
    _distribution,
    _quality,
    _task_bootstrap,
)


def test_distribution_interpolates_percentiles() -> None:
    result = _distribution([1.0, 2.0, 3.0, 4.0])
    assert result["n"] == 4
    assert result["mean"] == 2.5
    assert result["p50"] == 2.5
    assert result["p95"] == 3.8499999999999996


def test_latency_bootstrap_clusters_on_tasks() -> None:
    result = _task_bootstrap(
        {"task-a": [1.0, 3.0], "task-b": [-2.0, 0.0]},
        iterations=200,
        seed=7,
    )
    assert result["n_tasks"] == 2
    assert result["mean_delta_ms_staged_minus_fused"] == 0.5
    assert result["ci95"][0] <= 0.5 <= result["ci95"][1]


def test_quality_grades_released_all_prior_dependency_protocol() -> None:
    result = _quality([4, 2, 99], [1, 2, 3, 4], k=10)
    assert result["dependency_recall_at_10"] == 0.5
    assert 0.0 < result["ndcg_at_10"] < 1.0
    assert result["returned"] == 3
    assert result["constraint_violations"] == 1


def test_quality_is_identity_invariant_when_all_returned_are_relevant() -> None:
    left = _quality([1, 2], [1, 2, 3], k=10)
    right = _quality([3, 1], [1, 2, 3], k=10)
    assert left["dependency_recall_at_10"] == right["dependency_recall_at_10"]
    assert left["ndcg_at_10"] == right["ndcg_at_10"]
    assert 0.0 < left["ndcg_at_10"] < 1.0
