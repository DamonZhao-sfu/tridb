from tools.run_memoryarena_agent_outcomes import (
    _paired_bootstrap,
    normalize_exact_answer,
)


def test_normalized_exact_answer_is_format_insensitive_not_substring_match() -> None:
    assert normalize_exact_answer("**Ada  Lovelace!**") == "ada lovelace"
    assert normalize_exact_answer(None) is None
    assert normalize_exact_answer("Ada Lovelace") != normalize_exact_answer(
        "Ada Lovelace Byron"
    )


def test_paired_bootstrap_clusters_on_tasks() -> None:
    memory = {"a": {"score": 1.0}, "b": {"score": 0.0}}
    baseline = {"a": {"score": 0.0}, "b": {"score": 0.0}}
    result = _paired_bootstrap(
        memory, baseline, "score", iterations=100, seed=7
    )
    assert result["n_tasks"] == 2
    assert result["mean_delta"] == 0.5
    assert result["ci95"][0] <= 0.5 <= result["ci95"][1]
