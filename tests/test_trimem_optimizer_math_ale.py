"""Real adapters exercised only with mocked HTTP and tiny local evaluator fixtures."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import pytest

from trimem.optimizer.clients import (
    ChatResponse,
    LLMRequestError,
    OpenAICompatibleTransport,
)
from trimem.optimizer.evaluator import (
    CohortEvaluator,
    EvalContext,
    EvaluationProtocol,
    EvalUnit,
    IsolatedBranch,
    validate_observation,
)
from trimem.optimizer.ledger import MissingUsageError, TokenBudgetExhausted, TokenLedger
from trimem.optimizer.math_ale import (
    CheckpointSpec,
    MathAleContinuation,
    ProgramEvaluation,
    SubprocessScorer,
    classify_metrics,
    extract_program,
)


class MockResponse:
    def __init__(self, status_code, *, json=None, text=None):
        self.status_code, self.payload, self.text = status_code, json, text

    def json(self):
        if self.text is not None:
            raise ValueError("mock invalid JSON")
        return self.payload


class MockHTTP:
    def __init__(self, handler):
        self.handler = handler

    def post(self, url, *, json, headers):
        return self.handler(
            SimpleNamespace(content=__import__("json").dumps(json).encode())
        )


def transport(handler, **kwargs):
    client = MockHTTP(handler)
    return OpenAICompatibleTransport(
        model="test-model",
        base_url="https://unit.test/v1",
        prompt_token_count=lambda messages: 7,
        model_context_window=1000,
        tokenizer_id="test-tokenizer",
        http_client=client,
        **kwargs,
    )


def response_data(content="```python\nprint('ok')\n```", incoming=7, outgoing=3):
    return {
        "usage": {"prompt_tokens": incoming, "completion_tokens": outgoing},
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }


def test_transport_thinking_off_full_messages_and_capped_admission():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return MockResponse(200, json=response_data())

    client = transport(handler)
    ledger = TokenLedger(10)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "complete prompt"},
    ]
    response = client.complete(messages, ledger=ledger, seed=42)
    assert requests[0]["messages"] == messages
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert requests[0]["max_tokens"] == 3
    assert requests[0]["seed"] == 42
    assert response.input_tokens == 7 and response.output_tokens == 3
    assert response.prompt_tokens == 7 and response.completion_tokens == 3
    assert ledger.total_tokens == 10
    with pytest.raises(TokenBudgetExhausted):
        client.complete(messages, ledger=ledger)
    assert len(requests) == 1


def test_transport_explicit_known_usage_retry_counts_both_physical_requests():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return MockResponse(
                503, json={"usage": {"prompt_tokens": 2, "completion_tokens": 1}}
            )
        return MockResponse(200, json=response_data())

    ledger = TokenLedger(100)
    response = transport(handler, max_retries=1).complete([], ledger=ledger)
    assert ledger.total_tokens == 13
    assert [event.status for event in ledger.events] == ["failure", "success"]
    assert [event.category for event in ledger.events] == ["generation", "retry"]
    assert len(response.request_ids) == 2


@pytest.mark.parametrize(
    "failure", ["http", "transport", "missing_output", "invalid_json"]
)
def test_unknown_usage_fails_closed_without_hidden_retry(failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "transport":
            raise TimeoutError("mocked timeout")
        if failure == "http":
            return MockResponse(503, json={"error": "unavailable"})
        if failure == "invalid_json":
            return MockResponse(200, text="not JSON")
        data = response_data()
        del data["usage"]["completion_tokens"]
        return MockResponse(200, json=data)

    ledger = TokenLedger(100)
    with pytest.raises(MissingUsageError):
        transport(handler, max_retries=3).complete([], ledger=ledger)
    assert len(calls) == 1
    assert not ledger.objective_usage_complete


def test_default_transport_has_no_implicit_retry_and_checks_exact_prompt_count():
    calls = []

    def handler(request):
        calls.append(request)
        return MockResponse(
            503, json={"usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        )

    with pytest.raises(LLMRequestError):
        transport(handler).complete([], ledger=TokenLedger(100))
    assert len(calls) == 1
    ledger = TokenLedger(100)
    with pytest.raises(LLMRequestError, match="chat-template"):
        transport(
            lambda request: MockResponse(200, json=response_data(incoming=8))
        ).complete([], ledger=ledger)
    assert ledger.total_tokens == 11


def checkpoint(tmp_path, **kwargs):
    code = tmp_path / "seed.py"
    spec = tmp_path / "spec.txt"
    evaluator = tmp_path / "evaluate.py"
    code.write_text("# valid seed\nprint(-10)\n")
    spec.write_text("full specification " + "x" * 9000)
    evaluator.write_text(
        "def evaluate(program_path):\n    return {'combined_score': -8, 'validity': 1, 'extra': {'original': True}}\n"
    )
    return CheckpointSpec("math:test", "cp-1", code, spec, evaluator, -10, **kwargs)


def protocol(**kwargs):
    return EvaluationProtocol(
        "math-test",
        "test-model",
        "full-source-v1",
        "test-tokenizer",
        **kwargs,
    )


def context(source, p):
    unit = EvalUnit(
        source.task_id, source.checkpoint_id, "graph-1", 42, source.initial_fitness
    )
    return EvalContext(
        p,
        unit,
        TokenLedger(p.token_budget),
        IsolatedBranch("test", "graph-1", {}, "unit test clone"),
    )


def branches():
    ids = count()

    @contextmanager
    def factory(unit, action_hash):
        yield IsolatedBranch(
            str(next(ids)), unit.graph_snapshot, {}, "fresh test branch"
        )

    return factory


class FakeClient:
    model_version = "test-model"
    tokenizer_id = "test-tokenizer"

    def __init__(self, programs):
        self.programs = iter(programs)
        self.prompts = []

    def complete(
        self, messages, *, ledger, category="generation", seed=0, max_output_tokens=None
    ):
        ledger.request_allowance(7, 3)
        self.prompts.append(messages)
        request_id = f"request-{len(self.prompts)}"
        ledger.record(request_id, category, 7, 3)
        return ChatResponse(next(self.programs), 7, 3, (request_id,), "stop")


def test_fixed_round_continuation_keeps_best_valid_negative_program_and_full_memory(
    tmp_path,
):
    source = checkpoint(tmp_path)
    p = protocol(token_budget=100, horizon_kind="fixed_steps", max_steps=3)
    client = FakeClient(
        [
            "```python\n# good -8\n```",
            "```python\n# invalid zero\n```",
            "```python\n# worse -9\n```",
        ]
    )
    values = iter(
        [
            ProgramEvaluation(-8, True),
            ProgramEvaluation(0, False, error="invalid"),
            ProgramEvaluation(-9, True),
        ]
    )
    continuation = MathAleContinuation(
        {source.task_id: source},
        client=client,
        scorer=lambda *args: next(values),
        render_chat=lambda messages: json.dumps(messages),
    )
    retrieved = []

    def retrieve(candidate, ctx, state):
        retrieved.append(candidate.action_hash)
        return SimpleNamespace(text="all retrieved memory")

    unit = context(source, p).unit
    result = CohortEvaluator(
        p, [unit], continuation.runner(retrieve), branches()
    ).evaluate(SimpleNamespace(action_hash="p1"))
    assert result.complete_for_model
    assert result.quality_mean == -8
    assert result.total_tokens_mean == 30
    assert result.units[0].rollout.steps == 3
    assert result.units[0].rollout.valid_programs == 2
    assert retrieved == ["p1"]
    assert "all retrieved memory" in client.prompts[2][1]["content"]
    assert "x" * 9000 in client.prompts[0][1]["content"]
    assert "# good -8" in client.prompts[2][1]["content"]
    assert "# invalid zero" not in client.prompts[2][1]["content"]
    assert result.units[0].rollout.telemetry["best_code"] == "# good -8"
    assert validate_observation(result, p, [unit]) == result


@pytest.mark.parametrize(
    "horizon,complete", [("fixed_steps", False), ("token_budget", True)]
)
def test_token_budget_exhaustion_has_protocol_specific_completion(
    tmp_path, horizon, complete
):
    source = checkpoint(tmp_path)
    p = protocol(token_budget=10, horizon_kind=horizon, max_steps=3)
    continuation = MathAleContinuation(
        {source.task_id: source},
        client=FakeClient(["```python\n# update\n```"]),
        scorer=lambda *args: ProgramEvaluation(-8, True),
        render_chat=json.dumps,
    )
    unit = context(source, p).unit
    result = CohortEvaluator(
        p, [unit], continuation.runner(lambda *args: ""), branches()
    ).evaluate(SimpleNamespace(action_hash="none"))
    assert result.complete_for_model is complete
    assert result.units[0].rollout.steps == 1
    assert result.units[0].rollout.stop_reason == "token_budget"
    assert validate_observation(result, p, [unit]) == result


def test_checkpoint_hash_version_and_minimize_validity_are_explicit(tmp_path):
    source = checkpoint(tmp_path)
    digest = hashlib.sha256(source.code_path.read_bytes()).hexdigest()
    assert replace(source, code_sha256=digest).load().digests["code"] == digest
    with pytest.raises(ValueError, match="digest"):
        replace(source, code_sha256="wrong").load()
    client = FakeClient(["```python\n# better -12\n```"])
    runner = MathAleContinuation(
        {source.task_id: source},
        client=client,
        scorer=lambda *args: ProgramEvaluation(-12, True),
        render_chat=json.dumps,
    )
    p = protocol(
        token_budget=100,
        horizon_kind="fixed_steps",
        max_steps=1,
        quality_direction="minimize",
    )
    unit = context(source, p).unit
    result = CohortEvaluator(
        p, [unit], runner.runner(lambda *args: ""), branches()
    ).evaluate(SimpleNamespace(action_hash="none"))
    assert result.quality_mean == 12
    with pytest.raises(ValueError, match="model"):
        runner.initialize(context(source, replace(p, model_version="other")))


def test_evaluator_contract_handles_ale_negative_scores_and_ambiguous_errors(tmp_path):
    loaded = checkpoint(tmp_path).load()
    accepted = classify_metrics(
        loaded, {"combined_score": -100, "judge_result": "ACCEPTED"}
    )
    assert accepted.valid and accepted.fitness == -100
    invalid = classify_metrics(
        loaded, {"combined_score": 0, "judge_result": "COMPILATION_ERROR"}
    )
    assert not invalid.valid and invalid.infrastructure_error is None
    ambiguous = classify_metrics(
        loaded, {"combined_score": 0, "error": "service unavailable"}
    )
    assert not ambiguous.valid and ambiguous.infrastructure_error
    program = classify_metrics(
        replace(loaded, source=replace(loaded.source, error_policy="program")),
        {"combined_score": 0, "error": "bad program"},
    )
    assert not program.valid and not program.infrastructure_error


def test_subprocess_scorer_retains_original_metrics_and_enforces_cpu_threads(tmp_path):
    source = checkpoint(tmp_path)
    source.evaluator_path.write_text(
        "import os\ndef evaluate(program_path):\n    return {'combined_score': -8, 'validity': 1, 'cpu': os.environ['OMP_NUM_THREADS'], 'extra': {'unchanged': True}}\n"
    )
    loaded = source.load()
    result = SubprocessScorer(timeout=3, cpu_threads=1)(
        loaded, "print('ok')", context(source, protocol(token_budget=100))
    )
    assert result.valid and result.fitness == -8
    assert result.metrics["extra"] == {"unchanged": True}
    assert result.metrics["cpu"] == "1"
    assert result.seconds >= 0


def test_subprocess_timeout_and_import_errors_are_infrastructure_failures(tmp_path):
    source = checkpoint(tmp_path)
    source.evaluator_path.write_text(
        "import time\ndef evaluate(program_path):\n    time.sleep(10)\n"
    )
    ctx = context(source, protocol(token_budget=100))
    result = SubprocessScorer(timeout=0.1)(source.load(), "print(1)", ctx)
    assert not result.valid and "timeout" in result.infrastructure_error
    source.evaluator_path.write_text(
        "import module_that_does_not_exist_optimizer_test\n"
    )
    result = SubprocessScorer(timeout=3)(source.load(), "print(1)", ctx)
    assert not result.valid and "exit=" in result.infrastructure_error


def test_invalid_output_keeps_baseline_and_missing_transport_receipt_fails_closed(
    tmp_path,
):
    source = checkpoint(tmp_path)
    p = protocol(token_budget=100, horizon_kind="fixed_steps", max_steps=1)
    continuation = MathAleContinuation(
        {source.task_id: source},
        client=FakeClient(["no code block"]),
        scorer=lambda *args: pytest.fail("invalid response must not execute evaluator"),
        render_chat=json.dumps,
    )
    unit = context(source, p).unit
    evaluator = CohortEvaluator(
        p, [unit], continuation.runner(lambda *args: ""), branches()
    )
    result = evaluator.evaluate(SimpleNamespace(action_hash="none"))
    assert result.complete_for_model and result.quality_mean == -10
    assert result.validity_rate == 0
    continuation.client.complete = lambda *args, **kwargs: ChatResponse(
        "no code", 7, 3, ()
    )
    result = evaluator.evaluate(SimpleNamespace(action_hash="broken"))
    assert not result.complete_for_model
    assert result.units[0].rollout.status == "missing_usage"


def test_cpp_extraction_preserves_original_program_text():
    assert (
        extract_program("```cpp\nint main() { return 0; }\n```", "cpp20")
        == "int main() { return 0; }"
    )
    assert extract_program("```python\nprint(1)\n```", "cpp20") is None
