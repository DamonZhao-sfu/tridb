"""Protocol invariants independent of a GPU or running PostgreSQL."""

import json

import pytest

from trimem.experiments.exg_pilot import inspect_plan, run_arm, save


@pytest.mark.parametrize(
    "answers,expected_calls,pass1,pass2",
    [
        (["secret-gold"], 1, True, True),
        (["wrong", "secret-gold"], 2, False, True),
        (["wrong", "wrong-again"], 2, False, False),
    ],
)
def test_feedback_retry_budget_and_no_gold_injection(
    tmp_path, answers, expected_calls, pass1, pass2
):
    save(
        tmp_path / "musique/inputs.json",
        {"task": {"id": "task", "question": "A question", "passages": []}},
    )
    save(
        tmp_path / "musique/evaluator_labels.json",
        {"task": {"answer": "secret-gold", "aliases": []}},
    )

    class Calls:
        def __init__(self):
            self.records = []
            self.context = {}
            self.messages = []

        def tokens(self, **kwargs):
            return 10

        def chat(self, stage, messages):
            self.messages.append(messages)
            answer = answers[len(self.records)]  # A third call fails the test.
            self.records.append(
                dict(
                    index=len(self.records),
                    stage=stage,
                    seconds=1,
                    usage=dict(prompt_tokens=10, completion_tokens=2),
                )
            )
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"answer": answer})},
                    }
                ]
            }

    calls = Calls()
    run_arm(tmp_path, "musique", {"pilot": ["task"]}, "N", calls, "unused", "unused")
    result = json.loads((tmp_path / "musique/N/0000.json").read_text())
    assert result["llm_calls"] == expected_calls
    assert (result["pass1"], result["pass2"]) == (pass1, pass2)
    for messages in calls.messages:
        # Gold can only have appeared as a generated answer; these retries follow a wrong one.
        assert "secret-gold" not in json.dumps(messages)
        assert "aliases" not in json.dumps(messages)
    if expected_calls == 2:
        assert calls.messages[1][-1]["content"].startswith(
            "Evaluator feedback: incorrect."
        )
    assert not (tmp_path / "musique/N/0001.json").exists()


@pytest.mark.parametrize(
    "node", ["Sort", "Hash Join", "Hash", "Aggregate", "Materialize", "Function Scan"]
)
def test_reject_blocking_retrieval(node):
    with pytest.raises(RuntimeError, match="TR-1"):
        inspect_plan({"Node Type": "Limit", "Plans": [{"Node Type": node}]})
