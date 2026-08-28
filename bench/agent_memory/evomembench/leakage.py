"""Fail-closed separation between agent-visible inputs and evaluator labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# ``path`` is the expected tool trajectory in CrossEp-Tool, not a harmless URI.
EVALUATION_ONLY_KEYS = frozenset(
    {
        "rubric",
        "rubrics",
        "ground_truth",
        "possible_answer",
        "possible_answers",
        "reference_answer",
        "expected_answer",
        "path",
        "score",
        "success",
    }
)


class EvaluationLeakageError(ValueError):
    """An evaluator-only field was about to cross the online boundary."""


def assert_online_payload(value: Any, *, location: str = "payload") -> None:
    """Recursively reject evaluator-only keys before query or memory admission."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in EVALUATION_ONLY_KEYS:
                raise EvaluationLeakageError(
                    f"evaluation-only field at {location}.{key}"
                )
            assert_online_payload(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_online_payload(child, location=f"{location}[{index}]")
