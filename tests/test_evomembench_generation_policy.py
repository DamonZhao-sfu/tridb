from __future__ import annotations

from dataclasses import replace

import pytest

from bench.agent_memory.evomembench.generation_policy import (
    GenerationPolicyError,
    generate_strict_rescue,
)
from bench.agent_memory.serving import StreamingChatResult


def _result(
    text: str, finish_reason: str, prompt_tokens: int = 10
) -> StreamingChatResult:
    return StreamingChatResult(
        text=text,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 5,
            "total_tokens": prompt_tokens + 5,
        },
        request_started=1.0,
        first_token_at=1.1,
        completed_at=1.5,
        finish_reason=finish_reason,
    )


class _Chat:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[dict[str, object]] = []

    def stream_chat(self, **kwargs: object) -> StreamingChatResult:
        self.calls.append(kwargs)
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, StreamingChatResult)
        return value


def _count(_messages: object, template: object) -> int:
    assert isinstance(template, dict)
    return 10 if template["reasoning_effort"] == "low" else 11


def test_strict_rescue_accepts_natural_primary_without_rescue() -> None:
    chat = _Chat([_result("answer", "stop")])
    result = generate_strict_rescue(
        chat=chat,
        count_chat=_count,
        model="model",
        messages=[{"role": "user", "content": "question"}],
        context_window_tokens=262144,
        context_safety_tokens=1024,
        seed=42,
    )
    assert result.selected_attempt == "primary"
    assert len(result.attempts) == 1
    assert chat.calls[0]["max_tokens"] == 8192
    assert chat.calls[0]["seed"] == 42
    assert chat.calls[0]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "low",
    }


@pytest.mark.parametrize(
    "primary",
    [
        _result("partial", "length"),
        RuntimeError("transport"),
    ],
)
def test_strict_rescue_uses_one_medium_rescue(primary: object) -> None:
    rescue = replace(_result("complete", "stop", 11), request_started=2.0)
    chat = _Chat([primary, rescue])
    result = generate_strict_rescue(
        chat=chat,
        count_chat=_count,
        model="model",
        messages=[{"role": "user", "content": "question"}],
        context_window_tokens=262144,
        context_safety_tokens=1024,
        seed=42,
    )
    assert result.selected_attempt == "rescue"
    assert len(result.attempts) == 2
    assert chat.calls[1]["max_tokens"] == 32768
    assert chat.calls[1]["seed"] == 43
    assert chat.calls[1]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


def test_strict_rescue_fails_after_exactly_two_attempts() -> None:
    chat = _Chat([_result("partial", "length"), _result("again", "length")])
    with pytest.raises(GenerationPolicyError) as raised:
        generate_strict_rescue(
            chat=chat,
            count_chat=_count,
            model="model",
            messages=[{"role": "user", "content": "question"}],
            context_window_tokens=262144,
            context_safety_tokens=1024,
            seed=42,
        )
    assert len(raised.value.attempts) == 2
    assert len(chat.calls) == 2
