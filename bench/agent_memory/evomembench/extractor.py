"""Rubric-free, bounded extraction of completed agent trajectories.

The extractor deliberately consumes only the task input and observations that
were visible to the agent while it ran.  Official evaluator artefacts are
rejected before serialization.  Typed concepts are supplied from the pinned
prompt/schema adapter; they are not inferred from withheld answers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from bench.agent_memory.evomembench.leakage import assert_online_payload

EXTRACTOR_SCHEMA = "evomembench_observed_trajectory_v0.1.0"
CONCEPT_KINDS = frozenset(
    {
        "category",
        "skill",
        "environment",
        "tool",
        "function",
        "constraint",
        "failure_mode",
    }
)


@dataclass(frozen=True)
class ExtractedConcept:
    kind: str
    value: str
    provenance: str


@dataclass(frozen=True)
class ExtractedExperience:
    memory_payload: str
    input_sha256: str
    payload_sha256: str
    concepts: tuple[ExtractedConcept, ...]
    source_external_ids: tuple[str, ...]
    observed_messages: int
    original_chars: int
    retained_chars: int
    capped: bool
    extractor_schema: str = EXTRACTOR_SCHEMA
    llm_calls: int = 0


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _observable_messages(
    trajectory: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for index, message in enumerate(trajectory):
        if not isinstance(message, Mapping):
            raise ValueError(f"trajectory[{index}] must be an object")
        assert_online_payload(message, location=f"trajectory[{index}]")
        role = str(message.get("role", "unknown")).strip() or "unknown"
        content = _content(message.get("content", "")).strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def _bound(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars < 256:
        raise ValueError("max_chars must be at least 256")
    if len(text) <= max_chars:
        return text, False
    digest = hashlib.sha256(text.encode()).hexdigest()
    marker = f"\n[bounded trajectory sha256={digest} original_chars={len(text)}]\n"
    available = max_chars - len(marker)
    head = available // 2
    return text[:head] + marker + text[-(available - head) :], True


def _observed_failure_mode(messages: Sequence[Mapping[str, str]]) -> str | None:
    tool_text = "\n".join(
        message["content"]
        for message in messages
        if message["role"].casefold() in {"tool", "function"}
    ).casefold()
    if not tool_text:
        return None
    patterns = (
        ("permission_denied", ("permission denied", "unauthorized", "forbidden")),
        ("resource_not_found", ("not found", "does not exist", "unknown resource")),
        ("invalid_argument", ("invalid argument", "validation error", "bad request")),
        ("timeout", ("timeout", "timed out")),
        ("tool_error", ("error", "failed", "exception")),
    )
    for label, needles in patterns:
        if any(needle in tool_text for needle in needles):
            return label
    return None


def extract_observed_trajectory(
    *,
    task_signature: str,
    trajectory: Sequence[Mapping[str, Any]],
    concept_hints: Sequence[tuple[str, str]],
    source_external_ids: Sequence[str],
    max_chars: int = 16_000,
) -> ExtractedExperience:
    """Produce one deterministic ExperienceUnit payload from a completed run."""
    request = {
        "task_signature": task_signature,
        "trajectory": list(trajectory),
        "concept_hints": list(concept_hints),
        "source_external_ids": list(source_external_ids),
    }
    assert_online_payload(request, location="extractor_input")
    messages = _observable_messages(trajectory)
    rendered = "\n".join(
        ["Prior task signature:", task_signature, "Prior observed trajectory:"]
        + [f"{item['role']}: {item['content']}" for item in messages]
    ).strip()
    bounded, capped = _bound(rendered, max_chars)

    concepts: list[ExtractedConcept] = []
    seen: set[tuple[str, str]] = set()
    for kind, value in concept_hints:
        normalized_kind = str(kind).strip().casefold()
        normalized_value = str(value).strip()
        if normalized_kind not in CONCEPT_KINDS:
            raise ValueError(f"unsupported concept kind: {kind!r}")
        if not normalized_value or (normalized_kind, normalized_value) in seen:
            continue
        seen.add((normalized_kind, normalized_value))
        concepts.append(
            ExtractedConcept(
                kind=normalized_kind,
                value=normalized_value,
                provenance="agent_visible_prompt_or_schema",
            )
        )
    failure_mode = _observed_failure_mode(messages)
    if failure_mode is not None and ("failure_mode", failure_mode) not in seen:
        concepts.append(
            ExtractedConcept(
                kind="failure_mode",
                value=failure_mode,
                provenance="agent_visible_tool_feedback",
            )
        )

    canonical_input = json.dumps(
        request, ensure_ascii=False, sort_keys=True, default=str
    )
    return ExtractedExperience(
        memory_payload=bounded,
        input_sha256=hashlib.sha256(canonical_input.encode()).hexdigest(),
        payload_sha256=hashlib.sha256(bounded.encode()).hexdigest(),
        concepts=tuple(concepts),
        source_external_ids=tuple(str(item) for item in source_external_ids),
        observed_messages=len(messages),
        original_chars=len(rendered),
        retained_chars=len(bounded),
        capped=capped,
    )
