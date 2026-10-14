from trimem.experiments.trajectory_quality_cost import checkpoint_cost


def test_checkpoint_excludes_future_usage_but_counts_retries():
    records = {
        "dispatch": [
            {"sha256": "a", "iteration": 10},
            {"sha256": "b", "iteration": 11},
        ],
        "generation": [
            {"prompt_sha256": "a", "prompt_tokens": 100, "completion_tokens": 20},
            {"prompt_sha256": "a", "prompt_tokens": 100, "completion_tokens": 30},
            {"prompt_sha256": "b", "prompt_tokens": 999, "completion_tokens": 999},
        ],
        "retrieval": [],
    }
    result = checkpoint_cost(("task", "VGR"), 10, records, [], {})
    assert result["total_tokens_observed"] == 250
    assert result["generation_calls"] == 2
    assert result["usage_complete"]


def test_shared_semantic_context_is_not_charged_twice():
    key = ("task", "P1")
    records = {
        "dispatch": [{"sha256": "a", "iteration": 1}],
        "generation": [
            {"prompt_sha256": "a", "prompt_tokens": 100, "completion_tokens": 20}
        ],
        "retrieval": [{"context_sha256": "ctx", "iteration": 1}],
    }
    rows = [
        {
            "op": "SemFilter",
            "context_sha256": "ctx",
            "usage": {"prompt_tokens": 400, "completion_tokens": 3},
        }
    ]
    owners = {("SemFilter", "ctx"): {key, ("task", "P2")}}
    result = checkpoint_cost(key, 10, records, rows, owners)
    assert result["total_tokens_observed"] == 120
    assert result["ambiguous_calls_excluded"] == 1
    assert not result["usage_complete"]
    owners[("SemFilter", "ctx")] = {key}
    result = checkpoint_cost(key, 10, records, rows, owners)
    assert result["total_tokens_observed"] == 523
    assert result["usage_complete"]


def test_reused_context_across_checkpoint_is_ambiguous():
    key = ("task", "P2")
    records = {
        "dispatch": [
            {"sha256": "a", "iteration": 10},
            {"sha256": "a", "iteration": 11},
        ],
        "generation": [
            {"prompt_sha256": "a", "prompt_tokens": 100, "completion_tokens": 20}
        ],
        "retrieval": [
            {"context_sha256": "ctx", "iteration": 10},
            {"context_sha256": "ctx", "iteration": 11},
        ],
    }
    rows = [
        {
            "op": "SemFilter",
            "context_sha256": "ctx",
            "usage": {"prompt_tokens": 400, "completion_tokens": 3},
        }
    ]
    result = checkpoint_cost(key, 10, records, rows, {("SemFilter", "ctx"): {key}})
    assert result["ambiguous_calls_excluded"] == 2
    assert result["total_tokens_observed"] is None
    assert not result["usage_complete"]


def test_missing_usage_is_partial_not_zero_cost():
    records = {
        "dispatch": [{"sha256": "a", "iteration": 1}],
        "generation": [
            {"prompt_sha256": "a", "prompt_tokens": 100, "completion_tokens": None}
        ],
        "retrieval": [],
    }
    result = checkpoint_cost(("task", "VGR"), 10, records, [], {})
    assert result["total_tokens_observed"] == 100
    assert result["unknown_usage_calls"] == 1
    assert not result["usage_complete"]


def test_dispatched_prompt_without_any_telemetry_makes_cost_partial():
    records = {
        "dispatch": [
            {"sha256": "a", "iteration": 1},
            {"sha256": "missing", "iteration": 2},
        ],
        "generation": [
            {"prompt_sha256": "a", "prompt_tokens": 100, "completion_tokens": 20}
        ],
        "retrieval": [],
    }
    result = checkpoint_cost(("task", "VGR"), 10, records, [], {})
    assert result["dispatches_without_telemetry"] == 1
    assert not result["usage_complete"]
