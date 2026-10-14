from trimem.exg.budget import record_budget, summarize
from trimem.exg.runtime import read


def test_budget_summary_includes_rejected_prompts(tmp_path):
    context = dict(dataset="musique", mode="online", rep=0, arm="T", task_id="q")
    base = tmp_path / "cells/musique/online/r0/T/tasks/q"
    accepted = record_budget(
        base / "1/prompt_budget.json",
        context | dict(attempt=1),
        24576,
        dict(hints=["a"], accepted_hint_tokens=8192, hint_token_limit=8192),
    )
    assert not accepted["context_limit_exceeded"]
    rejected_path = base / "2/prompt_budget.json"
    rejected = record_budget(
        rejected_path,
        context | dict(attempt=2),
        24577,
        dict(hints=["a"], hint_token_limit=8192, skipped_for_token_budget=[3, 7]),
    )
    # Audit exists even though no generation response or completed episode exists.
    assert read(rejected_path)["context_excess_tokens"] == 1
    assert rejected["context_limit_exceeded"]
    overall = summarize(tmp_path)["overall"]
    assert overall["prompts_checked"] == 2
    assert overall["prompts_with_memory_omission"] == 1
    assert overall["skipped_hint_occurrences"] == 2
    assert overall["memory_omission_rate"] == 0.5
    assert overall["prompts_exceeding_context"] == 1
    assert overall["prompts_actually_truncated"] == 0
    assert overall["prompt_tokens_actually_removed"] == 0
