"""The write-plan vocabulary strategies emit and operators apply.

A strategy PROPOSES; the operator APPLIES. That split is what lets ``P_t`` be
evaluated exactly once, at commit, no matter which of Omri's four construction
forms produced the plan.

Plans are plain mappings (the ``IngestStrategy`` Protocol returns
``Sequence[Mapping[str, Any]]``) so they are trivially serialisable into
``gem_transition.delta`` and into a test fixture. The constructors here exist so
the key names have exactly one spelling.

**Plan-local references.** A plan routinely creates a unit and then attaches
fields and edges to it, before any vid exists. Such ops carry a ``ref`` — an
arbitrary plan-local handle — instead of a ``unit_id``. The operator resolves
every ref to a real vid during apply, and a ref that never resolves is a
referential-validity failure, not a silent no-op.
"""

from __future__ import annotations

from typing import Any, Mapping

UPSERT_UNIT = "upsert_unit"
APPEND_FIELD_VALUE = "append_field_value"
LINK = "link"
SPLIT_TOPIC = "split_topic"

KINDS = (UPSERT_UNIT, APPEND_FIELD_VALUE, LINK, SPLIT_TOPIC)


def upsert_unit(
    *,
    scope_id: str,
    title: str,
    summary: str = "",
    ref: str | None = None,
    unit_id: int | None = None,
    embedding: list[float] | None = None,
    embed_text: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or update one semantic unit.

    Supply EITHER ``embedding`` (already computed) or ``embed_text`` (the
    operator batches every such text into ONE embedding call). Never both —
    an ambiguous embedding source is exactly the thing that makes two runs
    incomparable, so it is rejected rather than silently resolved.
    """
    if embedding is not None and embed_text is not None:
        raise ValueError("supply embedding or embed_text, not both")
    return {
        "kind": UPSERT_UNIT,
        "scope_id": scope_id,
        "title": title,
        "summary": summary,
        "ref": ref,
        "unit_id": unit_id,
        "embedding": embedding,
        "embed_text": embed_text,
        "metadata": dict(metadata or {}),
    }


def append_field_value(
    *,
    field: str,
    value: str,
    valid_from: str,
    ref: str | None = None,
    unit_id: int | None = None,
    supersede_current: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one entry to a field's value history ``H = <(v, t, pi)>``.

    ``supersede_current=True`` is UPDATE semantics: the prior current value is
    closed with ``valid_to`` and chained through ``superseded_by`` — retained as
    historical evidence, never overwritten (C4). With it False the write is an
    insert, and the partial unique index aborts the transaction if that would
    leave two current values for one ``(unit, field)`` (C1).
    """
    return {
        "kind": APPEND_FIELD_VALUE,
        "field": field,
        "value": value,
        "valid_from": valid_from,
        "ref": ref,
        "unit_id": unit_id,
        "supersede_current": bool(supersede_current),
        "provenance": dict(provenance or {}),
    }


def link(
    *,
    edge_kind: str,
    rel: str,
    src_ref: str | None = None,
    src: int | None = None,
    dst_ref: str | None = None,
    dst: int | None = None,
    weight: float = 1.0,
) -> dict[str, Any]:
    """A typed edge.

    For ``extension`` the orientation is load-bearing: ``src --extension--> dst``
    means "a change in src entails re-evaluating dst", because revision can only
    walk out-edges (interface doc §6.4).
    """
    return {
        "kind": LINK,
        "edge_kind": edge_kind,
        "rel": rel,
        "src_ref": src_ref,
        "src": src,
        "dst_ref": dst_ref,
        "dst": dst,
        "weight": float(weight),
    }


def split_topic(
    *, unit_id: int, fields: list[str], new_title: str, ref: str | None = None
) -> dict[str, Any]:
    """Promote a field subset into a standalone unit ([GEM] Figure 3, "Alice")."""
    return {
        "kind": SPLIT_TOPIC,
        "unit_id": unit_id,
        "fields": list(fields),
        "new_title": new_title,
        "ref": ref,
    }
