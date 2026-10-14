"""Process-safe OpenEvolve LLM request telemetry for experiment cells.

OpenEvolve 0.3.2 returns only the response text from its OpenAI-compatible
adapter, discarding the server's usage counters and request duration.  The ALE
physical-plan experiment needs those values to separate retrieval, generation,
evaluation, and orchestration time without reconstructing LLM time as a residual.

The patch is opt-in through ``TRIDB_LLM_TELEMETRY_PATH``.  ``run_arm`` sets that
variable before OpenEvolve starts its process pool.  Calling ``install_from_env``
at module import time also installs the hook in workers created with ``spawn``.
No prompt or response body is persisted here; a stable prompt hash joins this
record to the already-sealed evolution trace.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

TELEMETRY_PATH_ENV = "TRIDB_LLM_TELEMETRY_PATH"
THINKING_ENABLED_ENV = "TRIDB_LLM_THINKING_ENABLED"
REASONING_EFFORT_ENV = "TRIDB_LLM_REASONING_EFFORT"
GENERATION_POLICY_ENV = "TRIDB_LLM_GENERATION_POLICY"
#: "1" = stream the completion to measure time-to-first-token (TTFT); the returned text is identical.
STREAM_ENV = "TRIDB_LLM_STREAM"
PRIMARY_MAX_TOKENS_ENV = "TRIDB_LLM_PRIMARY_MAX_TOKENS"
PRIMARY_REASONING_EFFORT_ENV = "TRIDB_LLM_PRIMARY_REASONING_EFFORT"
RESCUE_MAX_TOKENS_ENV = "TRIDB_LLM_RESCUE_MAX_TOKENS"
RESCUE_REASONING_EFFORT_ENV = "TRIDB_LLM_RESCUE_REASONING_EFFORT"
RESCUE_SEED_OFFSET_ENV = "TRIDB_LLM_RESCUE_SEED_OFFSET"
BASELINE_POLICY = "baseline"
STRICT_RESCUE_POLICY = "strict-rescue"
DEFAULT_PRIMARY_MAX_TOKENS = 8192
DEFAULT_RESCUE_MAX_TOKENS = 16384
DEFAULT_RESCUE_SEED_OFFSET = 1
TELEMETRY_SCHEMA = "tridb_openevolve_llm_telemetry_v1"

_DIFF_BLOCK = re.compile(
    r"<<<<<<< SEARCH[ \t]*\r?\n"
    r"(?P<search>.*?)\r?\n"
    r"=======[ \t]*\r?\n"
    r"(?P<replacement>.*?)\r?\n"
    r">>>>>>> REPLACE",
    re.DOTALL,
)


def prompt_sha256(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def has_complete_search_replace_diff(content: str) -> bool:
    """Return whether OpenEvolve can receive at least one complete, non-empty diff."""
    return any(match.group("search").strip() for match in _DIFF_BLOCK.finditer(content))


def invalid_diff_reason(content: str) -> str | None:
    """Classify semantic failures that justify the opt-in rescue request."""
    if not content.strip():
        return "empty_content"
    if not has_complete_search_replace_diff(content):
        return "missing_or_incomplete_search_replace"
    return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _request_params(
    params: dict[str, Any],
    *,
    max_tokens: int | None,
    reasoning_effort: str | None,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Copy request params and apply one deterministic generation-attempt contract."""
    updated = dict(params)
    if max_tokens is not None:
        key = (
            "max_completion_tokens"
            if "max_completion_tokens" in updated and "max_tokens" not in updated
            else "max_tokens"
        )
        updated[key] = max_tokens

    thinking_value = os.environ.get(THINKING_ENABLED_ENV)
    if thinking_value is not None:
        enabled = thinking_value.lower() == "true"
        extra_body = dict(updated.get("extra_body") or {})
        template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = enabled
        if enabled and reasoning_effort:
            template_kwargs["reasoning_effort"] = reasoning_effort
        elif not enabled:
            template_kwargs.pop("reasoning_effort", None)
        extra_body["chat_template_kwargs"] = template_kwargs
        updated["extra_body"] = extra_body
        if "reasoning_effort" in updated and reasoning_effort:
            updated["reasoning_effort"] = reasoning_effort

    seed = updated.get("seed")
    if seed_offset and isinstance(seed, int):
        updated["seed"] = seed + seed_offset
    return updated


def _model_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_")
        and key
        in {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
        }
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def install(path: Path) -> bool:
    """Install the opt-in adapter hook once in the current process."""
    from openevolve.llm.openai import OpenAILLM

    resolved = path.expanduser().resolve()
    current = getattr(OpenAILLM, "_tridb_telemetry_path", None)
    if current == str(resolved):
        return False
    if current is not None and current != str(resolved):
        raise RuntimeError(f"LLM telemetry already targets {current}, not {resolved}")

    async def call_api(self: Any, params: dict[str, Any]) -> str:
        if self.client is None:
            raise RuntimeError(
                "OpenAI client is not initialized (manual_mode enabled?)"
            )

        policy = os.environ.get(GENERATION_POLICY_ENV, BASELINE_POLICY)
        if policy not in {BASELINE_POLICY, STRICT_RESCUE_POLICY}:
            raise RuntimeError(f"unsupported LLM generation policy: {policy}")
        logical_request_id = str(uuid.uuid4())
        logical_started = time.perf_counter()

        async def perform_attempt(
            attempt_params: dict[str, Any],
            *,
            attempt_index: int,
            attempt_role: str,
            rescue_trigger: str | None = None,
        ) -> tuple[str, str | None]:
            request_id = str(uuid.uuid4())
            messages = list(attempt_params.get("messages") or [])
            template_kwargs = (attempt_params.get("extra_body") or {}).get(
                "chat_template_kwargs", {}
            )
            started_wall = time.time()
            started = time.perf_counter()
            base = {
                "schema_version": TELEMETRY_SCHEMA,
                "logical_request_id": logical_request_id,
                "request_id": request_id,
                "generation_policy": policy,
                "attempt_index": attempt_index,
                "attempt_role": attempt_role,
                "rescue_trigger": rescue_trigger,
                "pid": os.getpid(),
                "started_at_unix": started_wall,
                "model": attempt_params.get("model"),
                "api_base": str(getattr(self, "api_base", "")),
                "prompt_sha256": prompt_sha256(messages),
                "message_count": len(messages),
                "input_chars": sum(
                    len(str(item.get("content") or "")) for item in messages
                ),
                "max_tokens_requested": attempt_params.get(
                    "max_tokens", attempt_params.get("max_completion_tokens")
                ),
                "thinking_enabled": template_kwargs.get("enable_thinking"),
                "reasoning_effort": template_kwargs.get("reasoning_effort"),
                "seed": attempt_params.get("seed"),
            }
            loop = asyncio.get_event_loop()
            stream = os.environ.get(STREAM_ENV) == "1"
            ttft_ms = None
            try:
                if stream:
                    response, ttft_ms = await loop.run_in_executor(
                        None, lambda: _streamed(self.client, attempt_params))
                else:
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(**attempt_params),
                    )
            except BaseException as exc:
                _append(
                    resolved,
                    {
                        **base,
                        "status": "error",
                        "llm_api_ms": (time.perf_counter() - started) * 1000.0,
                        "logical_request_elapsed_ms": (
                            time.perf_counter() - logical_started
                        )
                        * 1000.0,
                        "returned_to_openevolve": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise

            choice = response.choices[0]
            message = choice.message
            content = message.content or ""
            reasoning = getattr(message, "reasoning_content", None) or ""
            invalid_reason = invalid_diff_reason(content)
            returned = (
                policy == BASELINE_POLICY
                or invalid_reason is None
                or attempt_role == "rescue"
            )
            usage = _model_dump(getattr(response, "usage", None)) or {}
            _append(
                resolved,
                {
                    **base,
                    "status": "success",
                    "llm_api_ms": (time.perf_counter() - started) * 1000.0,
                    "logical_request_elapsed_ms": (
                        time.perf_counter() - logical_started
                    )
                    * 1000.0,
                    "server_request_id": getattr(response, "id", None),
                    "streamed": stream,
                    "ttft_ms": ttft_ms,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "output_chars": len(content),
                    "reasoning_chars": len(str(reasoning)),
                    "diff_valid": invalid_reason is None,
                    "invalid_reason": invalid_reason,
                    "returned_to_openevolve": returned,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "prompt_tokens_details": usage.get("prompt_tokens_details"),
                    "completion_tokens_details": usage.get("completion_tokens_details"),
                },
            )
            logger = logging.getLogger("openevolve.llm.openai")
            logger.debug("API parameters: %s", attempt_params)
            logger.debug("API response: %s", content)
            return content, invalid_reason

        if policy == BASELINE_POLICY:
            baseline_params = _request_params(
                params,
                max_tokens=None,
                reasoning_effort=os.environ.get(REASONING_EFFORT_ENV),
            )
            content, _ = await perform_attempt(
                baseline_params, attempt_index=0, attempt_role="single"
            )
            return content

        primary_params = _request_params(
            params,
            max_tokens=_env_int(PRIMARY_MAX_TOKENS_ENV, DEFAULT_PRIMARY_MAX_TOKENS),
            reasoning_effort=os.environ.get(PRIMARY_REASONING_EFFORT_ENV, "low"),
        )
        content, invalid_reason = await perform_attempt(
            primary_params, attempt_index=0, attempt_role="primary"
        )
        if invalid_reason is None:
            return content

        rescue_params = _request_params(
            params,
            max_tokens=_env_int(RESCUE_MAX_TOKENS_ENV, DEFAULT_RESCUE_MAX_TOKENS),
            reasoning_effort=os.environ.get(RESCUE_REASONING_EFFORT_ENV, "medium"),
            seed_offset=_env_int(RESCUE_SEED_OFFSET_ENV, DEFAULT_RESCUE_SEED_OFFSET),
        )
        rescued, _ = await perform_attempt(
            rescue_params,
            attempt_index=1,
            attempt_role="rescue",
            rescue_trigger=invalid_reason,
        )
        return rescued

    OpenAILLM._call_api = call_api
    OpenAILLM._tridb_telemetry_path = str(resolved)
    return True


def _streamed(client: Any, params: dict[str, Any]) -> tuple[Any, float | None]:
    """Run one chat completion with stream=True and rebuild a response-like object.

    Returns (response, ttft_ms) where ttft_ms is the time to the first content or reasoning delta.
    """
    from types import SimpleNamespace

    started = time.perf_counter()
    ttft = None
    content, reasoning, finish, usage, rid = [], [], None, None, None
    stream = client.chat.completions.create(**params, stream=True, stream_options={"include_usage": True})
    for chunk in stream:
        rid = rid or getattr(chunk, "id", None)
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        for ch in chunk.choices or []:
            delta = ch.delta
            piece = getattr(delta, "content", None)
            rpiece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if (piece or rpiece) and ttft is None:
                ttft = (time.perf_counter() - started) * 1000.0
            if piece:
                content.append(piece)
            if rpiece:
                reasoning.append(rpiece)
            if ch.finish_reason:
                finish = ch.finish_reason
    message = SimpleNamespace(content="".join(content), reasoning_content="".join(reasoning))
    choice = SimpleNamespace(message=message, finish_reason=finish)
    return SimpleNamespace(id=rid, choices=[choice], usage=usage), ttft


def install_from_env() -> bool:
    value = os.environ.get(TELEMETRY_PATH_ENV)
    return False if not value else install(Path(value))
