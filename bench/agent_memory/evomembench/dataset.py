"""Normalize the EvoMemBench CrossEp-Know stream without inventing relevance labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from bench.agent_memory.memoryarena.oracle import DecisionPoint

SOURCE_REPOSITORY = "https://github.com/DSAIL-Memory/EvoMemBench"
PINNED_REVISION = "aa4cea8fd936b76b2d3591d3ef897030617dc43a"
SOURCE_RELATIVE_PATH = "Cross-Episode-Knowledge/CROSSEP-KNOW/CL-bench_context_ge5.jsonl"
PINNED_SOURCE_SHA256 = (
    "f4652ddcf954dd33653f91c9b40ed6138617b92e0f80d470cd1c92bb50890ce9"
)
SCHEMA_VERSION = "evomembench_crossep_know_v0.1.0"


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class EvoEpisode:
    context_uid: str
    episode_uid: str
    source_task_id: str
    ordinal: int
    category: str
    subcategory: str
    messages: tuple[Message, ...]
    rubrics: tuple[str, ...]
    prior_episode_uids: tuple[str, ...]

    @property
    def cutoff_ordinal(self) -> int:
        return self.ordinal

    @property
    def retrieval_query(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        raise ValueError(f"episode {self.episode_uid} has no user message")

    def decision_point(self) -> DecisionPoint:
        """Create a cutoff-bearing decision with explicitly unavailable relevance."""
        return DecisionPoint(
            task_uid=self.context_uid,
            target_session_uid=self.episode_uid,
            cutoff_ordinal=self.cutoff_ordinal,
            query=self.retrieval_query,
            relevant_session_uids=frozenset(),
            dependency_basis="not_released_by_evomembench",
            relevance_available=False,
        )


@dataclass(frozen=True)
class EvoContext:
    context_uid: str
    source_context_id: str
    category: str
    episodes: tuple[EvoEpisode, ...]

    def decision_episodes(self) -> tuple[EvoEpisode, ...]:
        return self.episodes[1:]


@dataclass(frozen=True)
class EvoCorpus:
    source_repository: str
    source_revision: str
    source_sha256: str
    contexts: tuple[EvoContext, ...]
    relevance_annotation: str

    @property
    def episode_count(self) -> int:
        return sum(len(context.episodes) for context in self.contexts)

    @property
    def decision_count(self) -> int:
        return sum(max(0, len(context.episodes) - 1) for context in self.contexts)

    def iter_decision_episodes(self) -> Iterator[EvoEpisode]:
        for context in self.contexts:
            yield from context.decision_episodes()


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _messages(value: Any) -> tuple[Message, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty list")
    messages: list[Message] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("message must be an object")
        messages.append(
            Message(
                role=_string(item.get("role"), "message.role"),
                content=_string(item.get("content"), "message.content"),
            )
        )
    if not any(message.role == "user" for message in messages):
        raise ValueError("messages must contain a user turn")
    return tuple(messages)


def _rubrics(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("rubrics must be a non-empty list")
    return tuple(_string(item, "rubric") for item in value)


def normalize_crossep_know(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_revision: str = "unknown",
    source_sha256: str = "unknown",
) -> EvoCorpus:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    context_order: list[str] = []
    seen_task_ids: set[str] = set()
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        context_id = _string(metadata.get("context_id"), "metadata.context_id")
        task_id = _string(metadata.get("task_id"), "metadata.task_id")
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen_task_ids.add(task_id)
        if context_id not in grouped:
            context_order.append(context_id)
        grouped[context_id].append(row)

    contexts: list[EvoContext] = []
    for context_id in context_order:
        context_uid = f"evomembench:crossep_know:context:{context_id}"
        episodes: list[EvoEpisode] = []
        category: str | None = None
        for ordinal, row in enumerate(grouped[context_id]):
            metadata = row["metadata"]
            row_category = _string(
                metadata.get("context_category"), "metadata.context_category"
            )
            if category is None:
                category = row_category
            elif category != row_category:
                raise ValueError(f"context {context_id} changes category")
            task_id = str(metadata["task_id"])
            episode_uid = f"evomembench:crossep_know:episode:{task_id}"
            episodes.append(
                EvoEpisode(
                    context_uid=context_uid,
                    episode_uid=episode_uid,
                    source_task_id=task_id,
                    ordinal=ordinal,
                    category=row_category,
                    subcategory=_string(
                        metadata.get("sub_category"), "metadata.sub_category"
                    ),
                    messages=_messages(row.get("messages")),
                    rubrics=_rubrics(row.get("rubrics")),
                    prior_episode_uids=tuple(
                        episode.episode_uid for episode in episodes
                    ),
                )
            )
        if not episodes or category is None:
            raise ValueError(f"context {context_id} has no episodes")
        contexts.append(
            EvoContext(
                context_uid=context_uid,
                source_context_id=context_id,
                category=category,
                episodes=tuple(episodes),
            )
        )
    return EvoCorpus(
        source_repository=SOURCE_REPOSITORY,
        source_revision=source_revision,
        source_sha256=source_sha256,
        contexts=tuple(contexts),
        relevance_annotation=(
            "no source-episode relevance labels released; retrieval recall and oracle "
            "are unavailable until an independent annotation artifact is pinned"
        ),
    )


def load_crossep_know(
    path: str | Path,
    *,
    source_revision: str = PINNED_REVISION,
    expected_sha256: str | None = PINNED_SOURCE_SHA256,
) -> EvoCorpus:
    source = Path(path)
    payload = source.read_bytes()
    observed_sha = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise ValueError(
            f"EvoMemBench source checksum mismatch: {observed_sha} != {expected_sha256}"
        )
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(payload.decode().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_no}: expected object")
        rows.append(row)
    return normalize_crossep_know(
        rows,
        source_revision=source_revision,
        source_sha256=observed_sha,
    )
