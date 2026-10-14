"""Explicit, metered OpenAI-compatible calls; construction performs no request."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .ledger import MissingUsageError, TokenLedger


@dataclass(frozen=True)
class ChatResponse:
    content: str
    input_tokens: int
    output_tokens: int
    request_ids: tuple[str, ...]
    finish_reason: str | None = None

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens


class LLMRequestError(RuntimeError):
    """A provider request failed; its known/unknown usage remains in the ledger."""


class OpenAICompatibleTransport:
    """Serial transport with actual usage and explicitly configured retries.

    ``prompt_token_count`` must use the deployed model's chat template, including
    the generation prefix and ``enable_thinking=False``. It must count the complete
    messages, not just their content. Provider usage is authoritative and checked
    against this count. No generation is dispatched if its prompt/output allowance
    does not fit. Unknown failed-request usage stops further calls.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        prompt_token_count: Callable[[Sequence[Mapping[str, str]]], int],
        model_context_window: int,
        tokenizer_id: str,
        model_version: str | None = None,
        max_output_tokens: int = 8192,
        max_retries: int = 0,
        timeout: float = 120.0,
        http_client: Any = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        for name, value in (
            ("model_context_window", model_context_window),
            ("max_output_tokens", max_output_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if not model or not tokenizer_id or not base_url:
            raise ValueError("model, tokenizer_id, and base_url are required")
        self.model = model
        self.model_version = model_version or model
        self.tokenizer_id = tokenizer_id
        self.prompt_token_count = prompt_token_count
        self.model_context_window = model_context_window
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.temperature, self.top_p = temperature, top_p
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._owns_client = http_client is None
        if http_client is None:
            import httpx

            http_client = httpx.Client(timeout=timeout)
        self._client = http_client

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        ledger: TokenLedger,
        category: str = "generation",
        seed: int = 0,
        max_output_tokens: int | None = None,
    ) -> ChatResponse:
        prompt_tokens = self.prompt_token_count(messages)
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError(
                "exact chat-template token counter must return a nonnegative integer"
            )
        requested_output = (
            self.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        if type(requested_output) is not int or requested_output <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        output_space = self.model_context_window - prompt_tokens
        if output_space <= 0:
            raise ValueError(
                "complete prompt leaves no output space in the model context window"
            )
        request_ids = []
        for attempt in range(self.max_retries + 1):
            allowance = ledger.request_allowance(
                prompt_tokens, min(requested_output, output_space)
            )
            request_id = str(uuid4())
            request_ids.append(request_id)
            ledger.start(request_id, category if attempt == 0 else "retry")
            body = {
                "model": self.model,
                "messages": [dict(message) for message in messages],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": seed + attempt,
                "max_tokens": allowance,
                "n": 1,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            try:
                response = self._client.post(
                    self.endpoint, json=body, headers=self.headers
                )
            except Exception as exc:
                ledger.finish(
                    request_id, input_tokens=None, output_tokens=None, status="failure"
                )
                raise MissingUsageError(
                    f"request {request_id} failed without provider usage ({type(exc).__name__})"
                ) from exc
            try:
                data = response.json()
            except (ValueError, TypeError) as exc:
                ledger.finish(
                    request_id, input_tokens=None, output_tokens=None, status="failure"
                )
                raise MissingUsageError(
                    f"request {request_id} returned no parseable usage"
                ) from exc
            usage = data.get("usage") if isinstance(data, dict) else None
            incoming = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            outgoing = (
                usage.get("completion_tokens") if isinstance(usage, dict) else None
            )
            incoming = incoming if type(incoming) is int and incoming >= 0 else None
            outgoing = outgoing if type(outgoing) is int and outgoing >= 0 else None
            success = 200 <= response.status_code < 300
            content: str | None = None
            finish_reason: str | None = None
            if success:
                try:
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")
                    success = isinstance(content, str)
                except (KeyError, IndexError, TypeError):
                    success = False
            ledger.finish(
                request_id,
                input_tokens=incoming,
                output_tokens=outgoing,
                status="success" if success else "failure",
            )
            if incoming is None or outgoing is None:
                raise MissingUsageError(
                    f"request {request_id} has incomplete actual usage"
                )
            if success and incoming != prompt_tokens:
                raise LLMRequestError(
                    "provider input usage differs from the deployed chat-template token count"
                )
            if outgoing > allowance:
                raise LLMRequestError(
                    "provider output usage exceeded its admitted token cap"
                )
            if success:
                return ChatResponse(
                    content, incoming, outgoing, tuple(request_ids), finish_reason
                )
            retryable = response.status_code in {408, 429, 500, 502, 503, 504}
            if not retryable or attempt == self.max_retries:
                raise LLMRequestError(
                    f"provider request failed (HTTP {response.status_code})"
                )
        raise AssertionError("unreachable")
