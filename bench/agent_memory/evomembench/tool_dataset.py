"""Pinned CrossEp-Tool prompt loader; ground truth is loaded elsewhere."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.agent_memory.evomembench.dataset import Message, PINNED_REVISION
from bench.agent_memory.evomembench.leakage import assert_online_payload

TOOL_SOURCE_RELATIVE_PATH = (
    "Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/"
    "bfcl_eval/data/BFCL_v4_multi_turn_ours.json"
)
PINNED_TOOL_SOURCE_SHA256 = (
    "08fd17fa87a2b915a45aa0787668f8196d034dcfea4e6bab55d41abd4b13ea2d"
)
PINNED_TOOL_ROW_COUNT = 200
TOOL_CATEGORIES = ("gorilla_fs", "vehicle_control", "trading_bot", "travel_api")


@dataclass(frozen=True)
class ToolEpisode:
    episode_uid: str
    source_id: str
    ordinal: int
    category: str
    question: tuple[tuple[Message, ...], ...]
    initial_config: Mapping[str, Any]
    involved_classes: tuple[str, ...]
    excluded_functions: tuple[str, ...]

    def online_payload(self) -> dict[str, Any]:
        payload = {
            "id": self.source_id,
            "question": [
                [{"role": item.role, "content": item.content} for item in turn]
                for turn in self.question
            ],
            "initial_config": self.initial_config,
            "involved_classes": self.involved_classes,
            "excluded_function": self.excluded_functions,
        }
        assert_online_payload(payload)
        return payload


@dataclass(frozen=True)
class ToolCorpus:
    source_revision: str
    source_sha256: str
    episodes: tuple[ToolEpisode, ...]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    return rows


def load_tool_category_ids(ids_dir: str | Path) -> dict[str, str]:
    """Load the four pinned 50-id protocol partitions and reject overlap."""
    root = Path(ids_dir)
    mapping: dict[str, str] = {}
    for category in TOOL_CATEGORIES:
        payload = json.loads((root / f"ids_{category}.json").read_text())
        ids = payload.get("multi_turn_ours")
        if (
            not isinstance(ids, list)
            or len(ids) != 50
            or not all(isinstance(item, str) for item in ids)
        ):
            raise ValueError(f"{category} must contain exactly 50 string ids")
        for source_id in ids:
            if source_id in mapping:
                raise ValueError(f"tool id appears in multiple partitions: {source_id}")
            mapping[source_id] = category
    if len(mapping) != PINNED_TOOL_ROW_COUNT:
        raise ValueError(f"tool partition union has {len(mapping)} ids, expected 200")
    return mapping


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(value)


def _question(value: Any) -> tuple[tuple[Message, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("question must be a non-empty turn list")
    turns: list[tuple[Message, ...]] = []
    for turn in value:
        if not isinstance(turn, list) or not turn:
            raise ValueError("question turn must be a non-empty message list")
        messages: list[Message] = []
        for item in turn:
            if not isinstance(item, Mapping):
                raise ValueError("question message must be an object")
            role, content = item.get("role"), item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("question message needs string role/content")
            messages.append(Message(role, content))
        turns.append(tuple(messages))
    return tuple(turns)


def normalize_crossep_tool(
    rows: Sequence[Mapping[str, Any]],
    *,
    category_by_id: Mapping[str, str] | None = None,
    source_revision: str = "unknown",
    source_sha256: str = "unknown",
) -> ToolCorpus:
    category_by_id = category_by_id or {}
    episodes: list[ToolEpisode] = []
    seen: set[str] = set()
    for ordinal, row in enumerate(rows):
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("tool row needs a non-empty id")
        if source_id in seen:
            raise ValueError(f"duplicate tool id: {source_id}")
        seen.add(source_id)
        initial_config = row.get("initial_config")
        if not isinstance(initial_config, Mapping):
            raise ValueError("initial_config must be an object")
        episode = ToolEpisode(
            episode_uid=f"evomembench:crossep_tool:episode:{source_id}",
            source_id=source_id,
            ordinal=ordinal,
            category=category_by_id.get(source_id, "uncategorized"),
            question=_question(row.get("question")),
            initial_config=dict(initial_config),
            involved_classes=_strings(row.get("involved_classes"), "involved_classes"),
            excluded_functions=_strings(
                row.get("excluded_function", []), "excluded_function"
            ),
        )
        episode.online_payload()  # fail before an unsafe corpus can be returned
        episodes.append(episode)
    return ToolCorpus(source_revision, source_sha256, tuple(episodes))


def load_crossep_tool(
    path: str | Path,
    *,
    category_by_id: Mapping[str, str] | None = None,
    source_revision: str = PINNED_REVISION,
    expected_sha256: str | None = PINNED_TOOL_SOURCE_SHA256,
) -> ToolCorpus:
    source = Path(path)
    payload = source.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            f"CrossEp-Tool checksum mismatch: {observed} != {expected_sha256}"
        )
    corpus = normalize_crossep_tool(
        _jsonl(source),
        category_by_id=category_by_id,
        source_revision=source_revision,
        source_sha256=observed,
    )
    if (
        expected_sha256 == PINNED_TOOL_SOURCE_SHA256
        and len(corpus.episodes) != PINNED_TOOL_ROW_COUNT
    ):
        raise ValueError(
            f"pinned CrossEp-Tool row count is {len(corpus.episodes)}, "
            f"expected {PINNED_TOOL_ROW_COUNT}"
        )
    return corpus
