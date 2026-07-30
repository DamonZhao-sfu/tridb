"""Ingest strategies for the wiki demo. Both are LLM-free by construction.

TAXONOMY HONESTY
----------------
``IngestStrategy.name`` is the taxonomy label that lands in the run manifest, so
it must be one of [AM]'s four construction forms. Both strategies here extract
structure without a model, which is ``deterministic`` — but neither is
interchangeable with :class:`DeterministicIngestStrategy`, which is chunk-grain
with a single ``content`` field and no edges. The difference is recorded as
``variant`` in the manifest and as ``strategy_variant`` on every unit, and a run
of one must never be compared against a run of the other without saying so:
their embedding sources differ (article lead vs chunk text), which alone makes
the retrieval numbers incomparable.

WHY NO LLM HERE
---------------
Wikidata already ships the extraction an LLM would otherwise guess at — typed
class membership, subsumption, language-scoped labels. Spending a model on it
would add cost and error to a step whose ground truth is published. The
LLM-mediated and agentic strategies are G4's job, on the same slice, so that
comparison measures construction form rather than data.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from bench.agent_memory.demo import adapter
from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.protocols import MemoryView
from bench.agent_memory.gem.types import InteractionEvent


class WikiSliceIngestStrategy:
    """Act 1: the whole slice as one plan — units, fields and typed edges.

    ``plan`` reads nothing from the view and calls no model: it is a pure
    function of the event stream, so the construction cost the profiler records
    is exactly chunk-free extraction plus the operator's one batched embed.
    """

    name = "deterministic"
    variant = "wiki_article"

    def __init__(self, *, valid_from: str, include_classes: bool = True) -> None:
        self.valid_from = valid_from
        self.include_classes = include_classes

    def plan(
        self, events: Sequence[InteractionEvent], view: MemoryView
    ) -> Sequence[Mapping[str, Any]]:
        if not events:
            return []
        scope_id = events[0].scope_id
        wiki = adapter.slice_from_events(events)
        return adapter.slice_plan_ops(
            wiki,
            scope_id=scope_id,
            valid_from=self.valid_from,
            include_classes=self.include_classes,
        )


class WikidataRevisionStrategy:
    """Act 3: replay real Wikidata edits as field-level supersessions.

    Each event is one parsed edit. ``supersede_current=True`` is UPDATE
    semantics — the prior current value is closed with ``valid_to`` and chained
    through ``superseded_by``, never overwritten, which is what C1 and C4 are
    checked against afterwards.

    Unlike act 1 this strategy DOES read the view: an edit names an entity
    (``Q7251``), and only the store knows which unit id that entity became. An
    edit whose entity has no unit in this scope is dropped and counted — the
    revision histories cover entities that may not all have made it into the
    slice, and inventing a unit for one would be fabricating state.
    """

    name = "deterministic"
    variant = "wikidata_revision"

    def __init__(self, *, view_limit: int = 20000) -> None:
        self.view_limit = view_limit
        #: Populated by ``plan``; the scenario reports it rather than letting a
        #: silent drop look like a clean replay.
        self.unresolved: list[str] = []

    def plan(
        self, events: Sequence[InteractionEvent], view: MemoryView
    ) -> Sequence[Mapping[str, Any]]:
        if not events:
            return []
        scope_id = events[0].scope_id
        unit_ids_by_qid: dict[str, list[int]] = {}
        for unit in view.units(scope_id, limit=self.view_limit):
            qid = (unit.metadata or {}).get("qid")
            if qid and unit.id is not None:
                unit_ids_by_qid.setdefault(str(qid), []).append(int(unit.id))

        self.unresolved = []
        ops: list[dict[str, Any]] = []
        for event in events:
            meta = event.metadata or {}
            qid = str(meta.get("qid", ""))
            unit_ids = unit_ids_by_qid.get(qid)
            if not unit_ids:
                self.unresolved.append(qid)
                continue
            # One entity can appear as both an article and a class unit. Both
            # representations receive the same pinned entity revision.
            for unit_id in unit_ids:
                ops.append(
                    planmod.append_field_value(
                        unit_id=unit_id,
                        field=str(meta["field"]),
                        value=event.content,
                        valid_from=str(meta.get("timestamp") or event.event_time or ""),
                        # UPDATE semantics: the old value survives as history.
                        supersede_current=True,
                        provenance={
                            "source_external_ids": [
                                f"wikidata:{qid}@{meta.get('revid')}"
                            ],
                            "operator": "ingest",
                            # A claim value trimmed of its tool summary is flagged,
                            # so a reader can tell an exact value from a recovered one.
                            "confidence": (
                                1.0 if meta.get("confidence") == "exact" else 0.5
                            ),
                        },
                    )
                )
        return ops


def revision_events(
    edits: Sequence[adapter.ParsedEdit], *, scope_id: str
) -> list[InteractionEvent]:
    """Parsed Wikidata edits as an ingestion stream, oldest first.

    Order is load-bearing: applying a newer value before an older one would
    leave the older one current and invert C1. :func:`adapter.parse_revisions`
    already sorts, and this preserves that order in ``event_order``.
    """
    return [
        InteractionEvent(
            scope_id=scope_id,
            external_id=f"wikidata:{edit.qid}@{edit.revid}",
            content=edit.value,
            kind="revision",
            event_time=edit.timestamp,
            event_order=index,
            metadata={
                "qid": edit.qid,
                "revid": edit.revid,
                "field": edit.field,
                "action": edit.action,
                "confidence": edit.confidence,
                "timestamp": edit.timestamp,
                "user": edit.user,
            },
        )
        for index, edit in enumerate(edits)
    ]
