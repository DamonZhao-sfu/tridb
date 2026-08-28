"""Leakage-free vector query text for cross-episode experience retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib

from bench.agent_memory.evomembench.dataset import EvoEpisode, Message
from bench.agent_memory.evomembench.leakage import assert_online_payload


def _user_text(messages: Iterable[Message | Mapping[str, object]]) -> list[str]:
    out: list[str] = []
    for message in messages:
        if isinstance(message, Message):
            role, content = message.role, message.content
        else:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
        if role == "user" and content.strip():
            out.append(content.strip())
    return out


def _bounded(value: str, *, max_chars: int = 12_000) -> str:
    """Deterministic head/tail bound; never delegate silent truncation upstream."""
    if len(value) <= max_chars:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()
    head = max_chars // 3
    tail = max_chars - head
    return (
        value[:head]
        + f"\n[bounded-input sha256={digest} original_chars={len(value)}]\n"
        + value[-tail:]
    )


def knowledge_task_signature(episode: EvoEpisode) -> str:
    """Represent the current task without rubric or historical answers."""
    payload = {
        "benchmark": "CrossEp-Know",
        "category": episode.category,
        "subcategory": episode.subcategory,
        "user_turns": [_bounded(text) for text in _user_text(episode.messages)],
    }
    assert_online_payload(payload)
    return "\n".join(
        [
            "benchmark: CrossEp-Know",
            f"category: {episode.category}",
            f"subcategory: {episode.subcategory}",
            *(f"user: {text}" for text in payload["user_turns"]),
        ]
    )


def tool_task_signature(
    *,
    question: Sequence[Sequence[Mapping[str, object]]],
    involved_classes: Sequence[str],
    allowed_functions: Sequence[str] = (),
) -> str:
    """Represent a Tool task from prompts and available schema, never its path."""
    # At episode admission only the first user request is observable.  Later
    # turns are future information and therefore cannot seed retrieval.
    visible_user_turns = _user_text(question[0] if question else ())[:1]
    payload = {
        "benchmark": "CrossEp-Tool",
        "involved_classes": list(involved_classes),
        "allowed_functions": list(allowed_functions),
        "user_turns": [_bounded(text) for text in visible_user_turns],
    }
    assert_online_payload(payload)
    return "\n".join(
        [
            "benchmark: CrossEp-Tool",
            "environments: " + ", ".join(involved_classes),
            "available tools: " + ", ".join(allowed_functions),
            *(f"user: {text}" for text in payload["user_turns"]),
        ]
    )
