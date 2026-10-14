from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
pytest.importorskip("openevolve")

from bench.agent_memory.gem_oe.llm_telemetry import (  # noqa: E402
    GENERATION_POLICY_ENV,
    PRIMARY_MAX_TOKENS_ENV,
    PRIMARY_REASONING_EFFORT_ENV,
    REASONING_EFFORT_ENV,
    RESCUE_MAX_TOKENS_ENV,
    RESCUE_REASONING_EFFORT_ENV,
    RESCUE_SEED_OFFSET_ENV,
    STRICT_RESCUE_POLICY,
    THINKING_ENABLED_ENV,
    has_complete_search_replace_diff,
    install,
    invalid_diff_reason,
    prompt_sha256,
)

from openevolve.llm.openai import OpenAILLM  # noqa: E402


_ORIGINAL_CALL_API = OpenAILLM._call_api


@pytest.fixture(autouse=True)
def _reset_llm_hook(monkeypatch):
    OpenAILLM._call_api = _ORIGINAL_CALL_API
    if hasattr(OpenAILLM, "_tridb_telemetry_path"):
        delattr(OpenAILLM, "_tridb_telemetry_path")
    for name in (
        GENERATION_POLICY_ENV,
        PRIMARY_MAX_TOKENS_ENV,
        PRIMARY_REASONING_EFFORT_ENV,
        RESCUE_MAX_TOKENS_ENV,
        RESCUE_REASONING_EFFORT_ENV,
        RESCUE_SEED_OFFSET_ENV,
        REASONING_EFFORT_ENV,
        THINKING_ENABLED_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    OpenAILLM._call_api = _ORIGINAL_CALL_API
    if hasattr(OpenAILLM, "_tridb_telemetry_path"):
        delattr(OpenAILLM, "_tridb_telemetry_path")


def _usage(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details={"cached_tokens": 3},
        completion_tokens_details=None,
    )
    usage.model_dump = lambda mode="json": {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_tokens_details": usage.prompt_tokens_details,
        "completion_tokens_details": usage.completion_tokens_details,
    }
    return usage


def _response(
    content: str,
    *,
    finish_reason: str = "stop",
    reasoning: str = "",
    completion_tokens: int = 7,
) -> SimpleNamespace:
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, reasoning_content=reasoning),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        id=f"server-{finish_reason}",
        choices=[choice],
        usage=_usage(13, completion_tokens),
    )


class _Completions:
    def create(self, **params):
        assert params["model"] == "fake-model"
        assert params["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }
        return _response("answer", reasoning="think")


class _SequencedCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return self.responses.pop(0)


def test_llm_telemetry_records_usage_hash_and_medium_thinking(
    tmp_path, monkeypatch
) -> None:
    from openevolve.llm.openai import OpenAILLM

    path = tmp_path / "llm.jsonl"
    monkeypatch.setenv(THINKING_ENABLED_ENV, "true")
    monkeypatch.setenv(REASONING_EFFORT_ENV, "medium")
    install(path)
    llm = object.__new__(OpenAILLM)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    llm.api_base = "http://example/v1"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    result = asyncio.run(
        llm._call_api({"model": "fake-model", "messages": messages, "max_tokens": 8})
    )
    row = json.loads(path.read_text())
    assert result == "answer"
    assert row["status"] == "success"
    assert row["prompt_sha256"] == prompt_sha256(messages)
    assert row["prompt_tokens"] == 13
    assert row["completion_tokens"] == 7
    assert row["total_tokens"] == 20
    assert row["input_chars"] == 14
    assert row["output_chars"] == 6
    assert row["reasoning_chars"] == 5
    assert row["thinking_enabled"] is True
    assert row["reasoning_effort"] == "medium"
    assert row["generation_policy"] == "baseline"
    assert row["attempt_role"] == "single"
    assert row["returned_to_openevolve"] is True
    assert row["llm_api_ms"] >= 0


@pytest.mark.parametrize(
    ("content", "valid", "reason"),
    [
        (
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
            True,
            None,
        ),
        ("", False, "empty_content"),
        (
            "<<<<<<< SEARCH\nold\n=======\nnew",
            False,
            "missing_or_incomplete_search_replace",
        ),
        (
            "<<<<<<< SEARCH\n\n=======\nnew\n>>>>>>> REPLACE",
            False,
            "missing_or_incomplete_search_replace",
        ),
    ],
)
def test_search_replace_validation(content, valid, reason) -> None:
    assert has_complete_search_replace_diff(content) is valid
    assert invalid_diff_reason(content) == reason


def test_strict_rescue_retries_invalid_reasoning_once_with_deterministic_seed(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "llm.jsonl"
    monkeypatch.setenv(THINKING_ENABLED_ENV, "true")
    monkeypatch.setenv(REASONING_EFFORT_ENV, "medium")
    monkeypatch.setenv(GENERATION_POLICY_ENV, STRICT_RESCUE_POLICY)
    monkeypatch.setenv(PRIMARY_MAX_TOKENS_ENV, "8192")
    monkeypatch.setenv(PRIMARY_REASONING_EFFORT_ENV, "low")
    monkeypatch.setenv(RESCUE_MAX_TOKENS_ENV, "16384")
    monkeypatch.setenv(RESCUE_REASONING_EFFORT_ENV, "medium")
    monkeypatch.setenv(RESCUE_SEED_OFFSET_ENV, "1")
    completions = _SequencedCompletions(
        [
            _response(
                "",
                finish_reason="length",
                reasoning="hidden reasoning",
                completion_tokens=8192,
            ),
            _response(
                "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
                reasoning="short",
            ),
        ]
    )
    install(path)
    llm = object.__new__(OpenAILLM)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm.api_base = "http://example/v1"

    result = asyncio.run(
        llm._call_api(
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "question"}],
                "max_tokens": 16384,
                "seed": 42,
            }
        )
    )

    assert result.endswith(">>>>>>> REPLACE")
    assert len(completions.calls) == 2
    primary, rescue = completions.calls
    assert primary["max_tokens"] == 8192
    assert primary["seed"] == 42
    assert primary["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "low"
    assert rescue["max_tokens"] == 16384
    assert rescue["seed"] == 43
    assert rescue["extra_body"]["chat_template_kwargs"]["reasoning_effort"] == "medium"

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["logical_request_id"] == rows[1]["logical_request_id"]
    assert rows[0]["attempt_role"] == "primary"
    assert rows[0]["invalid_reason"] == "empty_content"
    assert rows[0]["returned_to_openevolve"] is False
    assert rows[0]["reasoning_chars"] == len("hidden reasoning")
    assert rows[1]["attempt_role"] == "rescue"
    assert rows[1]["rescue_trigger"] == "empty_content"
    assert rows[1]["diff_valid"] is True
    assert rows[1]["returned_to_openevolve"] is True


def test_strict_rescue_accepts_complete_primary_without_second_request(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "llm.jsonl"
    monkeypatch.setenv(THINKING_ENABLED_ENV, "true")
    monkeypatch.setenv(GENERATION_POLICY_ENV, STRICT_RESCUE_POLICY)
    completions = _SequencedCompletions(
        [
            _response(
                "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
                finish_reason="length",
            )
        ]
    )
    install(path)
    llm = object.__new__(OpenAILLM)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm.api_base = "http://example/v1"

    result = asyncio.run(
        llm._call_api(
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "question"}],
                "max_tokens": 16384,
                "seed": 42,
            }
        )
    )

    assert result.endswith(">>>>>>> REPLACE")
    assert len(completions.calls) == 1
    row = json.loads(path.read_text())
    assert row["attempt_role"] == "primary"
    assert row["finish_reason"] == "length"
    assert row["diff_valid"] is True
    assert row["returned_to_openevolve"] is True
