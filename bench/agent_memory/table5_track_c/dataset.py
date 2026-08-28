"""Frozen LoCoMo parsing rules for the Track C reproduction.

The benchmark never reads QA evidence while building memory.  Evidence is
carried only on query records so the offline quality gate can score returned
dialogue ids after latency measurement has completed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

EXPECTED_LOCOMO_SHA256 = (
    "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
)
EXPECTED_SAMPLE_COUNT = 10
EXPECTED_QUESTION_COUNT = 1_986
EXPECTED_FORMAL_QUESTION_COUNT = 1_787
WARMUP_SAMPLE_ID = "conv-26"

_SESSION_KEY = re.compile(r"^session_(\d+)$")


@dataclass(frozen=True)
class EventItem:
    sample_id: str
    event_id: str
    session_id: str
    timestamp: str
    role: str
    text: str
    ordinal: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QueryItem:
    sample_id: str
    question_id: str
    question: str
    answer: Any
    category: Any
    evidence_ids: tuple[str, ...]
    ordinal: int


@dataclass(frozen=True)
class LoCoMoCorpus:
    path: Path
    sha256: str
    events_by_sample: dict[str, tuple[EventItem, ...]]
    queries_by_sample: dict[str, tuple[QueryItem, ...]]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.queries_by_sample)

    @property
    def formal_sample_ids(self) -> tuple[str, ...]:
        return tuple(key for key in self.sample_ids if key != WARMUP_SAMPLE_ID)

    @property
    def all_events(self) -> tuple[EventItem, ...]:
        return tuple(
            event
            for sample_id in self.sample_ids
            for event in self.events_by_sample[sample_id]
        )

    @property
    def formal_queries(self) -> tuple[QueryItem, ...]:
        return tuple(
            query
            for sample_id in self.formal_sample_ids
            for query in self.queries_by_sample[sample_id]
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_evidence(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for key in ("dia_id", "dialogue_id", "id"):
            if value.get(key) is not None:
                yield str(value[key])
                return
        for nested in value.values():
            yield from _flatten_evidence(nested)
        return
    if isinstance(value, Sequence):
        for nested in value:
            yield from _flatten_evidence(nested)
        return
    yield str(value)


def _render_turn(turn: dict[str, Any], timestamp: str) -> str:
    speaker = str(turn.get("speaker", "Unknown"))
    body = str(
        turn.get("compressed_text", turn.get("clean_text", turn.get("text", "")))
    )
    rendered = f'({timestamp}) {speaker} said, "{body}"'
    if turn.get("blip_caption"):
        rendered += f"\n[{speaker} shares {turn['blip_caption']}]"
    return rendered


def _events(sample: dict[str, Any]) -> tuple[EventItem, ...]:
    sample_id = str(sample["sample_id"])
    conversation = sample["conversation"]
    sessions: list[tuple[int, list[dict[str, Any]]]] = []
    for key, turns in conversation.items():
        match = _SESSION_KEY.fullmatch(key)
        if match and isinstance(turns, list):
            sessions.append((int(match.group(1)), turns))

    result: list[EventItem] = []
    for session_number, turns in sorted(sessions):
        timestamp = str(conversation.get(f"session_{session_number}_date_time", ""))
        for turn_number, turn in enumerate(turns, start=1):
            event_id = turn.get("dia_id")
            if not event_id:
                raise ValueError(
                    f"{sample_id} session {session_number} turn {turn_number} has no dia_id"
                )
            result.append(
                EventItem(
                    sample_id=sample_id,
                    event_id=str(event_id),
                    session_id=f"S{session_number}",
                    timestamp=timestamp,
                    role=str(turn.get("speaker", "Unknown")),
                    text=_render_turn(turn, timestamp),
                    ordinal=session_number * 1_000_000 + turn_number,
                    metadata={
                        "source": "locomo",
                        "session_number": session_number,
                        "turn_number": turn_number,
                    },
                )
            )
    return tuple(result)


def _queries(sample: dict[str, Any]) -> tuple[QueryItem, ...]:
    sample_id = str(sample["sample_id"])
    result = []
    for index, qa in enumerate(sample["qa"]):
        evidence_ids = tuple(dict.fromkeys(_flatten_evidence(qa.get("evidence"))))
        result.append(
            QueryItem(
                sample_id=sample_id,
                question_id=str(
                    qa.get("question_id") or qa.get("id") or f"{sample_id}_q{index + 1}"
                ),
                question=str(qa["question"]),
                answer=qa.get("answer"),
                category=qa.get("category"),
                evidence_ids=evidence_ids,
                ordinal=index,
            )
        )
    return tuple(result)


def load_locomo(path: str | Path, *, verify: bool = True) -> LoCoMoCorpus:
    path = Path(path).resolve()
    checksum = _sha256(path)
    if verify and checksum != EXPECTED_LOCOMO_SHA256:
        raise ValueError(
            f"LoCoMo checksum mismatch: got {checksum}, expected {EXPECTED_LOCOMO_SHA256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LoCoMo root must be a JSON array")

    events_by_sample: dict[str, tuple[EventItem, ...]] = {}
    queries_by_sample: dict[str, tuple[QueryItem, ...]] = {}
    for sample in payload:
        sample_id = str(sample["sample_id"])
        if sample_id in queries_by_sample:
            raise ValueError(f"duplicate LoCoMo sample_id: {sample_id}")
        events_by_sample[sample_id] = _events(sample)
        queries_by_sample[sample_id] = _queries(sample)

    corpus = LoCoMoCorpus(
        path=path,
        sha256=checksum,
        events_by_sample=events_by_sample,
        queries_by_sample=queries_by_sample,
    )
    if verify:
        total = sum(len(queries) for queries in queries_by_sample.values())
        formal = len(corpus.formal_queries)
        if len(payload) != EXPECTED_SAMPLE_COUNT:
            raise ValueError(f"expected 10 samples, found {len(payload)}")
        if total != EXPECTED_QUESTION_COUNT:
            raise ValueError(f"expected 1,986 questions, found {total}")
        if formal != EXPECTED_FORMAL_QUESTION_COUNT:
            raise ValueError(f"expected 1,787 formal questions, found {formal}")
        if WARMUP_SAMPLE_ID not in queries_by_sample:
            raise ValueError(f"warmup sample {WARMUP_SAMPLE_ID} is absent")
    return corpus


def representative_warmup_indices(
    queries: Sequence[QueryItem], target_count: int = 12
) -> tuple[int, ...]:
    """Match Mandol v0.1.0-paper-repro's head_quantile_longest selector."""
    target_count = min(max(0, target_count), len(queries))
    if target_count == 0:
        return ()

    selected: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if len(selected) >= target_count:
            return
        if 0 <= index < len(queries) and index not in seen:
            seen.add(index)
            selected.append(index)

    head_count = min(3, target_count)
    long_reserve = 2 if target_count >= 8 else 1 if target_count >= 5 else 0
    quantile_limit = max(head_count, target_count - long_reserve)
    for index in range(head_count):
        add(index)
    for fraction in (0.1, 0.25, 0.35, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0):
        if len(selected) >= quantile_limit:
            break
        add(round((len(queries) - 1) * fraction))
    for index in sorted(
        range(len(queries)), key=lambda item: len(queries[item].question), reverse=True
    ):
        add(index)
    for index in range(len(queries)):
        add(index)
    return tuple(selected)
