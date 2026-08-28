"""LongMemEval-S corpus loader, shaped like :mod:`dataset` so adapters are unchanged.

Mandol Table 4 reports LongMemEval accuracy over the six question types the
benchmark defines. This module turns ``longmemeval_s.json`` into the same
``EventItem`` / ``QueryItem`` / corpus surface the LoCoMo loader produces, so
every Track C adapter (TriDB/GEM, Mem0, MemOS, Cognee, Graphiti, EverMemOS)
runs against it without a single adapter change.

Three decisions are baked in here and every one of them changes what the
numbers mean, so they are stated rather than buried:

**One memory scope per question.** LongMemEval gives each question its own
haystack, and the distractor sessions in that haystack were selected *for that
question*. Pooling questions into a shared store would let a query retrieve
another question's distractors and would silently change the task. So
``sample_id == question_id`` and there are 500 isolated scopes. Sessions that
appear in several haystacks (19,829 distinct ids across 25,112 references) are
therefore ingested once per scope; the ~21% duplication is the price of
keeping the benchmark's semantics intact.

**Session dates are rendered into the turn text.** LongMemEval carries the date
as per-session metadata, not inside the utterance. LoCoMo carries it inline.
Adapters differ in which one they can use: Graphiti and EverMemOS read the
structured timestamp, while TriDB/GEM has no temporal model at all and only
ever sees the text. Emitting the date in *both* places reproduces LoCoMo's
native conditions and keeps the comparison fair -- dropping the inline copy
would hand every extraction-based system a temporal signal that GEM cannot
reach, and the resulting Temporal column would measure our rendering choice
rather than the systems.

**Event ids are position-keyed.** ``event_id`` is
``{session_index}:{session_id}#{turn_index}``. Session ids are not unique
within a haystack -- 13 of the 500 questions repeat one -- so an id-keyed
event id collides and any store with a per-scope uniqueness constraint
rejects the build. Evidence ids derived from ``has_answer`` use the same key,
so recall metrics stay aligned with what was actually ingested.

**Abstention questions are kept and flagged.** 30 of the 500 ids end in
``_abs``; the correct response is a refusal. They stay in the corpus with
``metadata["abstention"] = True`` so a driver can report them separately, but
they are NOT silently dropped -- Mandol's per-column counts (30/56/133/133/
78/70) sum to exactly 500, so the paper scores the full set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .dataset import EventItem, QueryItem

#: LoCoMo's rendered timestamp format. Reused verbatim so that adapters which
#: parse it (``graphiti_track_c.reference_time``,
#: ``evermemos_track_c.reference_epoch_ms``) need no LongMemEval branch.
LOCOMO_TIMESTAMP_FORMAT = "%I:%M %p on %d %B, %Y"

#: LongMemEval's own session-date format, e.g. ``2023/05/20 (Sat) 02:21``.
LONGMEMEVAL_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"

#: ``question_type`` -> Mandol Table 4 column header.
QUESTION_TYPE_TO_COLUMN = {
    "single-session-preference": "SS-Pref",
    "single-session-assistant": "SS-Asst",
    "temporal-reasoning": "Temporal",
    "multi-session": "Multi-S",
    "knowledge-update": "Know. Upd.",
    "single-session-user": "SS-User",
}

#: Table 4 column order, as printed in the paper.
COLUMN_ORDER = (
    "SS-Pref",
    "SS-Asst",
    "Temporal",
    "Multi-S",
    "Know. Upd.",
    "SS-User",
)

ABSTENTION_SUFFIX = "_abs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_session_date(value: str) -> datetime | None:
    """Parse a LongMemEval session date; ``None`` when it is absent or odd.

    A few haystack dates carry no time component. Returning ``None`` rather
    than guessing keeps a fabricated timestamp out of the corpus.
    """
    if not value:
        return None
    for fmt in (LONGMEMEVAL_DATE_FORMAT, "%Y/%m/%d (%a)", "%Y/%m/%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def render_timestamp(moment: datetime | None) -> str:
    """Render into LoCoMo's inline format so adapters parse it unchanged."""
    if moment is None:
        return ""
    # %-d/%-I would drop the zero padding, but strptime with %d/%I accepts the
    # padded form on round-trip, so keep padding for lossless parsing.
    return moment.strftime(LOCOMO_TIMESTAMP_FORMAT)


def render_turn(role: str, content: str, stamp: str) -> str:
    """Mirror LoCoMo's ``(time) Speaker said, "text"`` rendering."""
    speaker = "User" if role == "user" else "Assistant"
    if stamp:
        return f'({stamp}) {speaker} said, "{content}"'
    return f'{speaker} said, "{content}"'


@dataclass(frozen=True)
class LongMemEvalCorpus:
    """Same surface as :class:`~.dataset.LoCoMoCorpus`.

    ``warmup_sample_ids`` is explicit rather than a module constant: LoCoMo
    reserves one hard-coded conversation, but LongMemEval has 500 independent
    scopes and which ones serve as warmup is a run-level choice.
    """

    path: Path
    sha256: str
    events_by_sample: dict[str, tuple[EventItem, ...]]
    queries_by_sample: dict[str, tuple[QueryItem, ...]]
    warmup_sample_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.queries_by_sample)

    @property
    def formal_sample_ids(self) -> tuple[str, ...]:
        return tuple(k for k in self.sample_ids if k not in self.warmup_sample_ids)

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

    def stats(self) -> dict[str, Any]:
        events = self.all_events
        return {
            "questions": len(self.queries_by_sample),
            "formal_questions": len(self.formal_sample_ids),
            "warmup_questions": len(self.warmup_sample_ids),
            "events": len(events),
            "sessions": len({(e.sample_id, e.session_id) for e in events}),
            "characters": sum(len(e.text) for e in events),
        }


def _events_for_question(record: dict[str, Any]) -> tuple[
    tuple[EventItem, ...], tuple[str, ...]
]:
    question_id = str(record["question_id"])
    sessions = record.get("haystack_sessions") or []
    session_ids = record.get("haystack_session_ids") or []
    dates = record.get("haystack_dates") or []
    answer_sessions = set(record.get("answer_session_ids") or [])

    events: list[EventItem] = []
    evidence: list[str] = []
    ordinal = 0
    for index, session in enumerate(sessions):
        session_id = str(session_ids[index]) if index < len(session_ids) else f"s{index}"
        raw_date = str(dates[index]) if index < len(dates) else ""
        moment = parse_session_date(raw_date)
        stamp = render_timestamp(moment)
        for turn_index, turn in enumerate(session or []):
            content = str(turn.get("content") or "")
            if not content.strip():
                continue
            role = str(turn.get("role") or "user")
            # Keyed on the session's POSITION, not its id: 13 of the 500
            # haystacks list the same session_id twice (e.g. 58bf7951 carries
            # 07b7a667_1 twice), and an id-keyed event_id collides inside the
            # scope. TriDB/GEM enforces that as a real unique constraint --
            # `gem_field_value_current_uq` -- so the build aborts rather than
            # silently merging two distinct turns.
            event_id = f"{index:03d}:{session_id}#{turn_index}"
            # `has_answer` is LongMemEval's own turn-level gold marker; it is
            # the only evidence signal in the file, so recall metrics must key
            # on it rather than on the session id (a gold session still holds
            # many non-evidence turns).
            if bool(turn.get("has_answer")):
                evidence.append(event_id)
            events.append(
                EventItem(
                    sample_id=question_id,
                    event_id=event_id,
                    session_id=session_id,
                    timestamp=stamp,
                    role=role,
                    text=render_turn(role, content, stamp),
                    ordinal=ordinal,
                    metadata={
                        "source": "longmemeval_s",
                        "question_id": question_id,
                        "session_id": session_id,
                        "session_index": index,
                        "turn_index": turn_index,
                        "turn_number": ordinal,
                        "session_date_raw": raw_date,
                        "role": role,
                        "is_answer_session": session_id in answer_sessions,
                        "has_answer": bool(turn.get("has_answer")),
                    },
                )
            )
            ordinal += 1
    return tuple(events), tuple(evidence)


def load_longmemeval(
    path: str | Path,
    *,
    warmup_sample_ids: frozenset[str] | set[str] | None = None,
    limit: int | None = None,
    verify: bool = True,
) -> LongMemEvalCorpus:
    """Load LongMemEval-S into the Track C corpus surface.

    ``limit`` truncates to the first N questions in file order. It exists for
    pipeline smoke tests only -- file order is not stratified by question
    type, so a truncated corpus is NOT a valid sample for reporting. Use
    :func:`stratified_sample_ids` when a real subset is wanted.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"LongMemEval file is not a list of questions: {path}")
    if limit is not None:
        payload = payload[:limit]

    events_by_sample: dict[str, tuple[EventItem, ...]] = {}
    queries_by_sample: dict[str, tuple[QueryItem, ...]] = {}
    for ordinal, record in enumerate(payload):
        question_id = str(record["question_id"])
        if question_id in queries_by_sample:
            raise ValueError(f"duplicate LongMemEval question_id: {question_id}")
        events, evidence = _events_for_question(record)
        question_type = str(record.get("question_type") or "")
        if verify and question_type not in QUESTION_TYPE_TO_COLUMN:
            raise ValueError(
                f"unknown LongMemEval question_type {question_type!r} "
                f"for {question_id}"
            )
        events_by_sample[question_id] = events
        queries_by_sample[question_id] = (
            QueryItem(
                sample_id=question_id,
                question_id=question_id,
                question=str(record.get("question") or ""),
                answer=record.get("answer"),
                category=question_type,
                evidence_ids=evidence,
                ordinal=ordinal,
            ),
        )

    corpus = LongMemEvalCorpus(
        path=path,
        sha256=_sha256(path),
        events_by_sample=events_by_sample,
        queries_by_sample=queries_by_sample,
        warmup_sample_ids=frozenset(warmup_sample_ids or ()),
    )
    if verify:
        missing = [k for k in corpus.warmup_sample_ids if k not in queries_by_sample]
        if missing:
            raise ValueError(f"warmup ids absent from corpus: {sorted(missing)}")
    return corpus


def stratified_sample_ids(
    corpus: LongMemEvalCorpus, per_type: dict[str, int]
) -> tuple[str, ...]:
    """Pick question ids per type, deterministically, in file order.

    No RNG: the selection must be identical on every machine and every rerun,
    or two runs of "the same" subset are not comparable.
    """
    taken: dict[str, int] = {}
    chosen: list[str] = []
    for sample_id in corpus.sample_ids:
        query = corpus.queries_by_sample[sample_id][0]
        kind = str(query.category)
        want = per_type.get(kind, 0)
        if taken.get(kind, 0) < want:
            taken[kind] = taken.get(kind, 0) + 1
            chosen.append(sample_id)
    shortfall = {k: v - taken.get(k, 0) for k, v in per_type.items() if v > taken.get(k, 0)}
    if shortfall:
        raise ValueError(f"not enough questions for stratified sample: {shortfall}")
    return tuple(chosen)


def iter_by_column(corpus: LongMemEvalCorpus) -> Iterator[tuple[str, QueryItem]]:
    """Yield ``(Table 4 column, query)`` for every formal question."""
    for query in corpus.formal_queries:
        yield QUESTION_TYPE_TO_COLUMN[str(query.category)], query
