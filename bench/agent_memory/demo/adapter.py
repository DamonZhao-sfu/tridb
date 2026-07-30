"""Translate a fetched wiki slice into GEM's vocabulary. No I/O, no network.

This module is where the demo's substantive modelling decisions live, so it is
the part with real unit tests.

EDGE TYPING — the decision the whole demo rests on
--------------------------------------------------
[GEM] §4.1 distinguishes the two edge kinds by PROPAGATION RIGHTS, not by
strength: revision follows entailment, not relatedness. Wikipedia gives us both
kinds honestly, and they must not be conflated:

    article --hyperlink--> article        ASSOCIATION
        A links to B because B is worth reading. Nothing about a change to A
        entails re-evaluating B. Retrieval may expand along it; revision may
        not.

    class --member_of--> article          EXTENSION   (from P31 / P106)
    class --subclass_of--> class          EXTENSION   (from P279)
        If what "cryptographer" denotes changes, every unit asserted to BE one
        must be re-evaluated. That is entailment, and it is why the orientation
        is class -> member: `revise` walks OUT-edges, and the change originates
        at the class.

Getting this backwards would make C3 vacuous, so the orientation is asserted in
one place (:func:`slice_plan_ops`) and tested directly.

FIELD-LEVEL SUPERSESSION — where act 3's evidence comes from
------------------------------------------------------------
Wikidata edit comments are structured, e.g.::

    /* wbsetdescription-set:1|ca */ informàtic anglès (1912–1954)
    /* wbsetdescription-set:1|ca */ pioner de la informàtica anglès (1912–1954)

Two real edits, same entity, same field, one minute apart: the second value
supersedes the first. That is a genuine C1 case observed in the wild, not a
staged one. :func:`parse_wikidata_comment` recovers ``(field, value)`` from such
a comment and is explicit about how confident the value extraction is —
description/label values are exact (everything after the marker), claim values
are heuristic (the API appends a human edit summary after the value with no
delimiter that cannot also occur inside a value).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.types import InteractionEvent

#: Plan-local reference prefixes. Articles and classes share one id space in
#: the store but are distinct kinds of unit, so their refs must not collide.
ARTICLE_REF = "a:"
CLASS_REF = "c:"

REL_HYPERLINK = "hyperlink"
REL_MEMBER_OF = "member_of"
REL_SUBCLASS_OF = "subclass_of"


# ---------------------------------------------------------------------------
# Loading a slice written by wiki_source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WikiArticle:
    title: str
    pageid: int
    qid: str | None
    extract: str
    label: str = ""
    description: str = ""
    classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WikiClass:
    qid: str
    label: str
    parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class WikiRevision:
    qid: str
    revid: int
    timestamp: str
    user: str
    comment: str


@dataclass
class WikiSlice:
    articles: tuple[WikiArticle, ...] = ()
    classes: tuple[WikiClass, ...] = ()
    links: tuple[tuple[str, str], ...] = ()
    revisions: tuple[WikiRevision, ...] = ()
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def by_qid(self) -> dict[str, WikiArticle]:
        return {a.qid: a for a in self.articles if a.qid}

    @property
    def class_by_qid(self) -> dict[str, WikiClass]:
        return {c.qid: c for c in self.classes}

    @classmethod
    def load(cls, directory: Path | str) -> WikiSlice:
        base = Path(directory)
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        articles = tuple(
            WikiArticle(
                title=row["title"],
                pageid=int(row["pageid"]),
                qid=row.get("qid"),
                extract=row.get("extract", ""),
                label=row.get("label", ""),
                description=row.get("description", ""),
                classes=tuple(row.get("classes", ())),
            )
            for row in _read_jsonl(base / "articles.jsonl")
        )
        classes = tuple(
            WikiClass(
                qid=row["qid"],
                label=row.get("label", ""),
                parents=tuple(row.get("parents", ())),
            )
            for row in _read_jsonl(base / "classes.jsonl")
        )
        links = tuple(
            (parts[0], parts[1])
            for parts in (
                line.split("\t")
                for line in (base / "links.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
            if len(parts) == 2
        )
        revisions = tuple(
            WikiRevision(
                qid=row["qid"],
                revid=int(row["revid"]),
                timestamp=row["timestamp"],
                user=row.get("user", ""),
                comment=row.get("comment", ""),
            )
            for row in _read_jsonl(base / "revisions.jsonl")
        )
        return cls(
            articles=articles,
            classes=classes,
            links=links,
            revisions=revisions,
            manifest=manifest,
        )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Slice -> write plan
# ---------------------------------------------------------------------------


def summarize(extract: str, *, max_chars: int = 320) -> str:
    """The unit summary: the lead sentence(s), bounded, never cut mid-word.

    A unit's ``summary`` is what an LLM-mediated strategy reads to choose a host
    topic and what the prompt block shows, so a truncated word is a real defect
    rather than a cosmetic one. The rule is deliberately simple and predictable:
    cut at the last sentence end within the budget; failing that, at the last
    word boundary. A short summary is fine — the full lead is kept verbatim in
    the ``abstract`` field, so bounding this loses nothing.
    """
    text = " ".join(extract.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sentence_end = cut.rfind(". ")
    if sentence_end > 0:
        return cut[: sentence_end + 1].strip()
    word_end = cut.rfind(" ")
    return (cut[:word_end] if word_end > 0 else cut).strip()


def slice_plan_ops(
    wiki: WikiSlice,
    *,
    scope_id: str,
    valid_from: str,
    include_classes: bool = True,
) -> list[dict[str, Any]]:
    """The whole slice as ONE write plan: units, fields, and typed edges.

    One plan means one transaction: the relational rows, the vectors and the
    native graph edges for the entire slice commit together or not at all. That
    is the property the demo exists to show, so the ingest is deliberately not
    chunked into per-article transactions.

    Only relations whose BOTH endpoints are in the slice are emitted. A dangling
    edge would be a referential-validity failure at apply time, and dropping it
    here (counted in the manifest) is honest where emitting it is not.
    """
    ops: list[dict[str, Any]] = []
    article_titles = {article.title for article in wiki.articles}
    class_by_qid = wiki.class_by_qid

    for article in wiki.articles:
        ref = ARTICLE_REF + article.title
        ops.append(
            planmod.upsert_unit(
                scope_id=scope_id,
                title=article.title,
                summary=summarize(article.extract),
                ref=ref,
                embed_text=article.extract or article.title,
                metadata={
                    "kind": "article",
                    "qid": article.qid,
                    "pageid": article.pageid,
                    "source": "enwiki",
                    # Records that these vectors are lead-paragraph vectors, not
                    # the chunk vectors DeterministicIngestStrategy produces.
                    # Two runs with different embedding sources are different
                    # operating points and must never be compared silently.
                    "embedding_source": "article_lead",
                    "strategy_variant": "wiki_article",
                },
            )
        )
        provenance = {
            "source_external_ids": [f"enwiki:{article.pageid}"],
            "operator": "ingest",
        }
        for field_name, value in (
            ("abstract", article.extract),
            ("label", article.label),
            ("description", article.description),
        ):
            if value:
                ops.append(
                    planmod.append_field_value(
                        ref=ref,
                        field=field_name,
                        value=value,
                        valid_from=valid_from,
                        provenance=provenance,
                    )
                )

    if include_classes:
        used_classes = _used_classes(wiki)
        for qid in sorted(used_classes):
            node = class_by_qid.get(qid)
            if node is None or not node.label:
                continue
            ops.append(
                planmod.upsert_unit(
                    scope_id=scope_id,
                    # The durable store also has UNIQUE(scope_id, title). Real
                    # slices contain class labels that are article titles, so a
                    # bare label merged 14 distinct a:/c: units on the G2 run.
                    title=f"Wikidata class {node.qid}: {node.label}",
                    summary=f"Wikidata class {node.qid}: {node.label}",
                    ref=CLASS_REF + node.qid,
                    embed_text=node.label,
                    metadata={
                        "kind": "class",
                        "qid": node.qid,
                        "label": node.label,
                        "source": "wikidata",
                        "embedding_source": "class_label",
                        "strategy_variant": "wiki_article",
                    },
                )
            )

    # ASSOCIATION: hyperlinks. No propagation rights.
    for src, dst in wiki.links:
        if src in article_titles and dst in article_titles and src != dst:
            ops.append(
                planmod.link(
                    edge_kind="association",
                    rel=REL_HYPERLINK,
                    src_ref=ARTICLE_REF + src,
                    dst_ref=ARTICLE_REF + dst,
                )
            )

    if include_classes:
        used_classes = _used_classes(wiki)
        # EXTENSION: class -> member. A change to the class entails
        # re-evaluating everything asserted to be one.
        for article in wiki.articles:
            for qid in article.classes:
                if qid in used_classes:
                    ops.append(
                        planmod.link(
                            edge_kind="extension",
                            rel=REL_MEMBER_OF,
                            src_ref=CLASS_REF + qid,
                            dst_ref=ARTICLE_REF + article.title,
                        )
                    )
        # EXTENSION: parent class -> child class, same argument one level up.
        for qid in sorted(used_classes):
            node = class_by_qid.get(qid)
            if node is None:
                continue
            for parent in node.parents:
                if parent in used_classes:
                    ops.append(
                        planmod.link(
                            edge_kind="extension",
                            rel=REL_SUBCLASS_OF,
                            src_ref=CLASS_REF + parent,
                            dst_ref=CLASS_REF + qid,
                        )
                    )

    return ops


def slice_events(wiki: WikiSlice, *, scope_id: str) -> list[InteractionEvent]:
    """The slice as an ingestion stream ``I_t``.

    A strategy must be a pure function of the events it is handed — if it also
    held the slice, ``plan()`` would depend on hidden state and the construction
    cost would stop being attributable to the stream. So every row travels as an
    event carrying its own payload, and :func:`slice_from_events` is the exact
    inverse. Class membership and subsumption ride on the article and class rows
    they came from; only hyperlinks need events of their own.
    """
    events: list[InteractionEvent] = []
    order = 0
    for article in wiki.articles:
        events.append(
            InteractionEvent(
                scope_id=scope_id,
                external_id=f"enwiki:{article.pageid}",
                content=article.extract,
                kind="article",
                event_order=order,
                metadata={
                    "title": article.title,
                    "pageid": article.pageid,
                    "qid": article.qid,
                    "label": article.label,
                    "description": article.description,
                    "classes": list(article.classes),
                },
            )
        )
        order += 1
    for node in wiki.classes:
        events.append(
            InteractionEvent(
                scope_id=scope_id,
                external_id=f"wikidata:{node.qid}",
                content=node.label,
                kind="class",
                event_order=order,
                metadata={
                    "qid": node.qid,
                    "label": node.label,
                    "parents": list(node.parents),
                },
            )
        )
        order += 1
    for src, dst in wiki.links:
        events.append(
            InteractionEvent(
                scope_id=scope_id,
                external_id=f"link:{src}->{dst}",
                content="",
                kind="link",
                event_order=order,
                metadata={"src": src, "dst": dst},
            )
        )
        order += 1
    return events


def slice_from_events(events: Sequence[InteractionEvent]) -> WikiSlice:
    """Rebuild the slice a strategy was handed. Inverse of :func:`slice_events`."""
    articles: list[WikiArticle] = []
    classes: list[WikiClass] = []
    links: list[tuple[str, str]] = []
    for event in events:
        meta = event.metadata or {}
        if event.kind == "article":
            articles.append(
                WikiArticle(
                    title=str(meta.get("title", "")),
                    pageid=int(meta.get("pageid", 0)),
                    qid=meta.get("qid"),
                    extract=event.content,
                    label=str(meta.get("label", "")),
                    description=str(meta.get("description", "")),
                    classes=tuple(meta.get("classes", ())),
                )
            )
        elif event.kind == "class":
            classes.append(
                WikiClass(
                    qid=str(meta.get("qid", "")),
                    label=str(meta.get("label", "")),
                    parents=tuple(meta.get("parents", ())),
                )
            )
        elif event.kind == "link":
            links.append((str(meta.get("src", "")), str(meta.get("dst", ""))))
    return WikiSlice(
        articles=tuple(articles), classes=tuple(classes), links=tuple(links)
    )


def _used_classes(wiki: WikiSlice) -> set[str]:
    """Classes an article actually claims, plus their in-slice ancestors.

    A class nobody in the slice belongs to would be an isolated unit that only
    inflates ``|D_t|`` and the C5 bound, so it is left out.
    """
    class_by_qid = wiki.class_by_qid
    used: set[str] = set()
    frontier = [qid for article in wiki.articles for qid in article.classes]
    while frontier:
        qid = frontier.pop()
        if qid in used or qid not in class_by_qid:
            continue
        used.add(qid)
        frontier.extend(class_by_qid[qid].parents)
    return used


# ---------------------------------------------------------------------------
# Wikidata edit comments -> field-level updates
# ---------------------------------------------------------------------------

_COMMENT = re.compile(
    r"^/\*\s*(?P<action>[a-zA-Z-]+):(?P<rest>[^*]*?)\s*\*/\s*(?P<tail>.*)$", re.S
)
_PROPERTY = re.compile(r"^\[\[Property:(?P<pid>P\d+)\]\]:\s*(?P<value>.*)$", re.S)

#: Suffixes MediaWiki tools append to a claim edit summary after the value.
#: Cutting at these is a heuristic — a value could in principle contain one —
#: so any value trimmed this way is marked ``heuristic``.
_SUMMARY_MARKERS = (
    ", Matched to [[",
    ", Added or updated using [[",
    ", #quickstatements",
    ", [[:toollabs:",
    ", [[User:",
)

#: Actions that carry a language-tagged text value; everything after the marker
#: IS the value, so extraction is exact.
_TEXT_ACTIONS = {
    "wbsetlabel-add": "label",
    "wbsetlabel-set": "label",
    "wbsetdescription-add": "description",
    "wbsetdescription-set": "description",
    "wbsetaliases-add": "aliases",
    "wbsetaliases-set": "aliases",
}

_CLAIM_ACTIONS = {
    "wbsetclaim-create",
    "wbsetclaim-update",
    "wbcreateclaim-create",
    "wbsetclaim-remove",
}


@dataclass(frozen=True)
class ParsedEdit:
    """One Wikidata edit reduced to a field-level assignment."""

    qid: str
    revid: int
    timestamp: str
    field: str
    value: str
    action: str
    confidence: str  # exact | heuristic
    user: str = ""
    raw_comment: str = ""


def parse_wikidata_comment(revision: WikiRevision) -> ParsedEdit | None:
    """Recover ``(field, value)`` from a Wikidata edit comment, or ``None``.

    ``None`` is a first-class outcome, not a failure: ``wbeditentity-update``
    batch edits and bare tool summaries carry no field-level assignment at all.
    The demo reports how many revisions parsed rather than pretending the
    unparsed ones do not exist.
    """
    match = _COMMENT.match(revision.comment.strip())
    if not match:
        return None
    action = match.group("action")
    rest = match.group("rest")
    tail = match.group("tail").strip()

    if action in _TEXT_ACTIONS and tail:
        # rest looks like "1|ca" — the language is what makes the field unique.
        language = rest.split("|")[-1].strip()
        if not language:
            return None
        return ParsedEdit(
            qid=revision.qid,
            revid=revision.revid,
            timestamp=revision.timestamp,
            field=f"{_TEXT_ACTIONS[action]}@{language}",
            value=tail,
            action=action,
            confidence="exact",
            user=revision.user,
            raw_comment=revision.comment,
        )

    if action in _CLAIM_ACTIONS:
        property_match = _PROPERTY.match(tail)
        if not property_match:
            return None
        value = property_match.group("value").strip()
        confidence = "exact"
        for marker in _SUMMARY_MARKERS:
            index = value.find(marker)
            if index >= 0:
                value = value[:index].strip()
                confidence = "heuristic"
                break
        if not value:
            return None
        return ParsedEdit(
            qid=revision.qid,
            revid=revision.revid,
            timestamp=revision.timestamp,
            field=property_match.group("pid"),
            value=value,
            action=action,
            confidence=confidence,
            user=revision.user,
            raw_comment=revision.comment,
        )

    return None


def parse_revisions(revisions: Sequence[WikiRevision]) -> list[ParsedEdit]:
    """Parse in revision order, oldest first — the order they must be applied.

    Applying out of order would make the newest value lose to an older one and
    silently invert C1, so the sort is part of the contract rather than a
    convenience.
    """
    parsed = [
        edit
        for edit in (parse_wikidata_comment(revision) for revision in revisions)
        if edit is not None
    ]
    return sorted(parsed, key=lambda edit: (edit.qid, edit.revid))


def supersessions(edits: Sequence[ParsedEdit]) -> list[tuple[str, str, int]]:
    """``(qid, field, times_written)`` for fields written more than once.

    These are the observed C1 cases: the same entity's same field really did
    take a new value. Act 3 quotes this count, so it is computed from the data
    rather than asserted.
    """
    counts: dict[tuple[str, str], int] = {}
    for edit in edits:
        counts[(edit.qid, edit.field)] = counts.get((edit.qid, edit.field), 0) + 1
    return sorted(
        ((qid, field_name, n) for (qid, field_name), n in counts.items() if n > 1),
        key=lambda row: (-row[2], row[0], row[1]),
    )
