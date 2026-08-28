"""Paradigm III construction: the LLM as a FIXED EXTRACTOR at predefined points.

The model reads unit titles and summaries, picks a host unit ([GEM] §4.2's
recipe), and emits ``(field, value, timestamp, provenance)`` against a **pinned
prompt** and a **versioned output schema**, both recorded in every
``Provenance``. Changing either without bumping its version makes two runs
silently incomparable, which is why they are module constants rather than
constructor defaults a caller can drift.

**Two modes**, differing only in call structure, because [AM] §4.3 found
embedding traffic is bimodal by paradigm and the two regimes stress a serving
stack differently:

============  ===============================  ==============================
              ``batch`` (III.a)                ``sequential`` (III.b)
============  ===============================  ==============================
embedding     one call per N units at the end  one call per extracted fact
conflict      append-only                      similarity search -> ADD/UPDATE
target        large sequences-per-call         1:1 call-to-sequence
example       GraphRAG ~2,300 seq/call         Mem0 / SimpleMem
============  ===============================  ==============================

The mode is recorded in the manifest; ``calls.items_by_kind`` already yields the
sequences-per-call ratio [AM] §4.3 plots.

**``validate()`` is not optional.** [AM] §4.4 shows that below an
algorithm-specific capability floor a weak construction model does not merely
lower accuracy — it *corrupts the store* (MIRIX fails outright at Qwen3-1.7B).
A unit failing a gate is rejected and counted, and a run whose rejection rate
exceeds the configured threshold is a **failed configuration**.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.protocols import MemoryView
from bench.agent_memory.gem.types import EdgeKind, InteractionEvent

#: Bump BOTH of these together with any wording change. They travel into every
#: gem_field_value row's provenance so a stored fact can always be traced to
#: the exact extractor that produced it (C4).
PROMPT_VERSION = "v1"
SCHEMA_VERSION = "v1"

MODE_BATCH = "batch"
MODE_SEQUENTIAL = "sequential"
MODES = (MODE_BATCH, MODE_SEQUENTIAL)

#: How many existing units are offered as host candidates. [GEM] §4.2 has the
#: model choose among titles + summaries rather than raw content, which is what
#: keeps the prompt bounded as the store grows.
DEFAULT_CANDIDATES = 8

SYSTEM_PROMPT = """\
You extract durable facts from a conversation excerpt into a topic-grained \
memory store.

A TOPIC is a self-contained concept holding every field of one thing (a \
project, a person, a trip). Prefer attaching to an existing topic over \
creating a new one.

Return ONLY a JSON object of this exact shape:

{"host": {"unit_id": <int>} | {"new_title": "<str>", "summary": "<str>"},
 "facts": [{"field": "<str>", "value": "<str>", "valid_from": "<ISO date>",
            "confidence": <float 0..1>}],
 "edges": [{"dst_title": "<str>", "kind": "extension"|"association",
            "rel": "<str>"}]}

Rules:
- "extension" means a change in the HOST entails re-evaluating the target. \
Use it only for genuine entailment.
- "association" means merely related. Use it for everything else.
- Emit a fact only if the excerpt states it. Never infer.
- "field" is a short snake_case attribute name, e.g. deadline, owner, city.
- No prose, no markdown fences, no explanation. JSON only.\
"""


class ExtractionRejected(Exception):
    """A gate refused one extraction. Carries the gate name for accounting."""

    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


class LLMMediatedIngestStrategy:
    """Implements the ``LLMMediatedIngest`` Protocol.

    ``client`` is any object with ``complete(system, user) -> (text, usage)``.
    Keeping it that narrow means the same strategy runs against the co-located
    vLLM answer endpoint (the default, so [AM] §4.3's construction/generation
    interference stays measurable) or against a separate one, with no code
    change.
    """

    name = "llm_mediated"
    prompt_version = PROMPT_VERSION
    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        mode: str = MODE_BATCH,
        chunker: Any | None = None,
        embedder: Any | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        chunk_tokens: int = 4096,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == MODE_SEQUENTIAL and embedder is None:
            # III.b's defining property is embedding each fact BEFORE its
            # similarity search resolves ADD/UPDATE/DELETE. Without an embedder
            # it would silently degrade into III.a and mislabel the run.
            raise ValueError(
                "sequential mode requires an embedder — the 1:1 "
                "call-to-sequence ratio IS the paradigm signature ([AM] §4.3)"
            )
        self.client = client
        self.model = model
        self.mode = mode
        self.embedder = embedder
        self.candidates = candidates
        self.chunk_tokens = chunk_tokens
        self._chunker = chunker
        self.cost: dict[str, int] = {}
        self.rejections: list[dict[str, Any]] = []

    @property
    def chunker(self) -> Any:
        if self._chunker is None:
            from bench.agent_memory.tridbBackend.chunking import TiktokenSentenceChunker

            self._chunker = TiktokenSentenceChunker(chunk_size=self.chunk_tokens)
        return self._chunker

    # -- the output contract ---------------------------------------------

    def validate(
        self,
        extracted: Mapping[str, Any],
        *,
        allowed_host_ids: set[int] | None = None,
    ) -> tuple[bool, str | None]:
        """The gates, in order. Any failure rejects the UNIT, not the run.

        1. JSON parses (done by the caller before this point);
        2. conforms to the pinned schema — required keys, types, enum for
           ``kind``;
        3. referential validity — ``host.unit_id`` exists in scope; every
           ``edges[].dst_title`` resolves or is created in the same plan;
        4. no dangling vertex — checked by ``validate_plan`` once the ops are
           emitted, so both layers agree;
        5. duplicate/conflict policy conformance — a fact that would create a
           second current value must carry supersession intent.
        """
        if not isinstance(extracted, Mapping):
            return False, "schema: extraction is not an object"

        host = extracted.get("host")
        if not isinstance(host, Mapping):
            return False, "schema: missing host object"
        if "unit_id" in host:
            if not isinstance(host["unit_id"], int):
                return False, "schema: host.unit_id is not an integer"
            if (
                allowed_host_ids is not None
                and int(host["unit_id"]) not in allowed_host_ids
            ):
                return (
                    False,
                    "referential: host.unit_id was not offered in candidate_topics",
                )
        elif not host.get("new_title"):
            return False, "schema: host has neither unit_id nor new_title"

        facts = extracted.get("facts", [])
        if not isinstance(facts, list):
            return False, "schema: facts is not a list"
        seen: set[str] = set()
        for fact in facts:
            if not isinstance(fact, Mapping):
                return False, "schema: fact is not an object"
            for key in ("field", "value", "valid_from"):
                if not fact.get(key):
                    return False, f"schema: fact missing {key}"
            confidence = fact.get("confidence")
            if confidence is not None and not isinstance(confidence, (int, float)):
                return False, "schema: confidence is not numeric"
            field_name = str(fact["field"])
            if field_name in seen and not fact.get("supersede"):
                # Two values for one field in one extraction would leave two
                # current values, which the partial unique index would abort.
                # Catching it here names the cause instead of surfacing an
                # integrity error.
                return (
                    False,
                    f"conflict: two values for field {field_name!r} without "
                    "supersession intent",
                )
            seen.add(field_name)

        edges = extracted.get("edges", [])
        if not isinstance(edges, list):
            return False, "schema: edges is not a list"
        for edge in edges:
            if not isinstance(edge, Mapping):
                return False, "schema: edge is not an object"
            if not edge.get("dst_title"):
                return False, "referential: edge missing dst_title"
            if edge.get("kind") not in (
                EdgeKind.EXTENSION.value,
                EdgeKind.ASSOCIATION.value,
            ):
                return False, f"enum: edge kind {edge.get('kind')!r} is not valid"

        return True, None

    # -- planning ---------------------------------------------------------

    def plan(
        self, events: Sequence[InteractionEvent], view: MemoryView
    ) -> list[Mapping[str, Any]]:
        self.cost = {}
        self.rejections = []
        ops: list[Mapping[str, Any]] = []
        # Titles created in this plan, so an edge can target a unit that does
        # not exist yet without becoming a dangling vertex.
        titles_in_plan: dict[str, str] = {}
        pending_embeds: list[tuple[str, str]] = []
        planned_fields: set[tuple[str, str]] = set()

        for event in events:
            for position, chunk in enumerate(self.chunker.chunk(event.content)):
                slate = self._host_slate(view, event.scope_id, chunk)
                raw = self._extract(chunk, slate)
                if raw is None:
                    continue
                allowed_host_ids = {
                    int(candidate["unit_id"])
                    for candidate in slate
                    if isinstance(candidate.get("unit_id"), int)
                }
                ok, reason = self.validate(
                    raw,
                    allowed_host_ids=allowed_host_ids,
                )
                if ok and self.mode == MODE_BATCH:
                    reason = self._batch_plan_conflict(
                        raw,
                        event,
                        position,
                        titles_in_plan,
                        planned_fields,
                        view,
                    )
                    ok = reason is None
                if not ok:
                    self.rejections.append(
                        {
                            "external_id": event.external_id,
                            "chunk_index": position,
                            "gate": (reason or "").split(":", 1)[0],
                            "detail": reason,
                        }
                    )
                    continue
                ops += self._emit(
                    raw,
                    event,
                    position,
                    titles_in_plan,
                    pending_embeds,
                    planned_fields,
                    view,
                )

        self._resolve_unit_embeddings(ops, pending_embeds)
        return ops

    def _host_slate(
        self, view: MemoryView, scope_id: str, chunk: str
    ) -> list[Mapping[str, Any]]:
        """Candidate host topics: titles + summaries only, never raw content."""
        if view is None or self.embedder is None:
            return []
        vector = self.embedder.encode([chunk])[0]
        self._bump("embed_calls", 1)
        self._bump("embed_sequences", 1)
        units = view.find_similar(scope_id, vector, k=self.candidates)
        return [
            {"unit_id": unit.id, "title": unit.title, "summary": unit.summary}
            for unit in units
        ]

    def _extract(
        self, chunk: str, slate: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any] | None:
        user = json.dumps(
            {"candidate_topics": list(slate), "excerpt": chunk},
            ensure_ascii=False,
            sort_keys=True,
        )
        text, usage = self.client.complete(SYSTEM_PROMPT, user)
        self._bump("llm_calls", 1)
        self._bump("prompt_tokens", int(usage.get("prompt_tokens", 0)))
        self._bump("completion_tokens", int(usage.get("completion_tokens", 0)))
        try:
            return json.loads(_strip_fences(text))
        except (ValueError, TypeError) as exc:
            self.rejections.append(
                {"gate": "json", "detail": f"extraction did not parse: {exc}"}
            )
            return None

    def _emit(
        self,
        extraction: Mapping[str, Any],
        event: InteractionEvent,
        position: int,
        titles_in_plan: dict[str, str],
        pending_embeds: list[tuple[str, str]],
        planned_fields: set[tuple[str, str]],
        view: MemoryView,
    ) -> list[Mapping[str, Any]]:
        ops: list[Mapping[str, Any]] = []
        host = extraction["host"]
        provenance = {
            "source_external_ids": [event.external_id],
            "extractor_model": self.model,
            "prompt_version": self.prompt_version,
            "operator": "ingest",
        }

        if "unit_id" in host:
            host_ref = None
            host_id = int(host["unit_id"])
        else:
            title = str(host["new_title"])
            host_ref = titles_in_plan.get(title) or f"{event.external_id}#{position}"
            host_id = None
            if title not in titles_in_plan:
                titles_in_plan[title] = host_ref
                ops.append(
                    planmod.upsert_unit(
                        scope_id=event.scope_id,
                        title=title,
                        summary=str(host.get("summary") or ""),
                        ref=host_ref,
                        metadata={
                            # Unlike DeterministicIngest this vector is the
                            # TITLE+SUMMARY, not the chunk text. Never compare
                            # retrieval quality across the two without saying so.
                            "embedding_source": "title_summary",
                            "strategy": self.name,
                            "mode": self.mode,
                            "prompt_version": self.prompt_version,
                            "schema_version": self.schema_version,
                        },
                    )
                )
                pending_embeds.append(
                    (host_ref, f"{title}\n{host.get('summary') or ''}".strip())
                )

        for fact in extraction.get("facts", []):
            host_key = f"id:{host_id}" if host_id is not None else f"ref:{host_ref}"
            field_key = (host_key, str(fact["field"]))
            supersede = bool(fact.get("supersede"))
            if self.mode == MODE_SEQUENTIAL and host_id is not None:
                # III.b: embed the fact, then let the similarity search decide
                # ADD vs UPDATE. This is the 1:1 call-to-sequence write loop.
                supersede = supersede or self._resolves_to_update(
                    view, host_id, str(fact["field"])
                )
            if self.mode == MODE_SEQUENTIAL and field_key in planned_fields:
                supersede = True
            ops.append(
                planmod.append_field_value(
                    ref=host_ref,
                    unit_id=host_id,
                    field=str(fact["field"]),
                    value=str(fact["value"]),
                    valid_from=str(fact["valid_from"]),
                    supersede_current=supersede,
                    provenance={**provenance, "confidence": fact.get("confidence")},
                )
            )
            planned_fields.add(field_key)

        for edge in extraction.get("edges", []):
            title = str(edge["dst_title"])
            dst_ref = titles_in_plan.get(title)
            dst_id = None
            if dst_ref is None:
                dst_id = self._resolve_title(view, event.scope_id, title)
                if dst_id is None:
                    # Create the target in this plan so the edge is never a
                    # dangling vertex.
                    dst_ref = f"edge:{title}"
                    titles_in_plan[title] = dst_ref
                    ops.append(
                        planmod.upsert_unit(
                            scope_id=event.scope_id,
                            title=title,
                            summary="",
                            ref=dst_ref,
                            metadata={
                                "embedding_source": "title_summary",
                                "strategy": self.name,
                                "created_by": "edge_target",
                            },
                        )
                    )
                    pending_embeds.append((dst_ref, title))
            ops.append(
                planmod.link(
                    edge_kind=str(edge["kind"]),
                    rel=str(edge.get("rel") or edge["kind"]),
                    src_ref=host_ref,
                    src=host_id,
                    dst_ref=dst_ref,
                    dst=dst_id,
                )
            )
        return ops

    def _batch_plan_conflict(
        self,
        extraction: Mapping[str, Any],
        event: InteractionEvent,
        position: int,
        titles_in_plan: Mapping[str, str],
        planned_fields: set[tuple[str, str]],
        view: MemoryView,
    ) -> str | None:
        """Reject an append-only extraction that would create two currents."""
        host = extraction["host"]
        if "unit_id" in host:
            unit_id = int(host["unit_id"])
            host_key = f"id:{unit_id}"
            existing = view.unit(unit_id) if view is not None else None
        else:
            title = str(host["new_title"])
            host_ref = titles_in_plan.get(title) or f"{event.external_id}#{position}"
            host_key = f"ref:{host_ref}"
            existing = None

        for fact in extraction.get("facts", []):
            field = str(fact["field"])
            if (host_key, field) in planned_fields:
                return (
                    "conflict: append-only plan already has a current value for "
                    f"field {field!r} on this host"
                )
            if existing is not None and existing.current(field) is not None:
                return (
                    "conflict: append-only host already has a current value for "
                    f"field {field!r}"
                )
        return None

    def _resolves_to_update(self, view: MemoryView, unit_id: int, field: str) -> bool:
        """III.b's ADD/UPDATE decision: does this field already have a value?"""
        if view is None:
            return False
        unit = view.unit(unit_id)
        return bool(unit and unit.current(field) is not None)

    def _resolve_title(self, view: MemoryView, scope_id: str, title: str) -> int | None:
        if view is None:
            return None
        for unit in view.units(scope_id, limit=1000):
            if unit.title == title:
                return unit.id
        return None

    def _resolve_unit_embeddings(
        self, ops: Sequence[Mapping[str, Any]], pending: list[tuple[str, str]]
    ) -> None:
        """Attach unit vectors according to the mode's embedding regime.

        ``batch`` leaves ``embed_text`` set so the OPERATOR issues one call for
        the whole plan — that is the large-sequences-per-call signature.
        ``sequential`` resolves them here, per fact, which is the 1:1 signature.
        Same store, different traffic shape, which is exactly what [AM] §4.3
        measures.
        """
        by_ref = {ref: text for ref, text in pending}
        for op in ops:
            if op.get("kind") == planmod.UPSERT_UNIT and op.get("ref") in by_ref:
                op["embed_text"] = by_ref[str(op["ref"])]  # type: ignore[index]

        if self.mode != MODE_SEQUENTIAL or not pending or self.embedder is None:
            return
        texts = [text for _, text in pending]
        vectors = self.embedder.encode(texts)
        self._bump("embed_calls", len(texts))  # one call per item, by design
        self._bump("embed_sequences", len(texts))
        resolved = dict(zip([ref for ref, _ in pending], vectors, strict=True))
        for op in ops:
            ref = op.get("ref")
            if op.get("kind") == planmod.UPSERT_UNIT and ref in resolved:
                op["embedding"] = [float(v) for v in resolved[str(ref)]]  # type: ignore[index]
                op["embed_text"] = None  # type: ignore[index]

    def _bump(self, key: str, amount: int) -> None:
        self.cost[key] = self.cost.get(key, 0) + amount


def _strip_fences(text: str) -> str:
    """Tolerate markdown fences the prompt forbids but weak models still emit.

    Deliberately the ONLY leniency: a model that cannot follow the output
    contract at all is the capability-floor finding [AM] §4.4 is about, and
    repairing its output would hide exactly the signal being measured.
    """
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


class OpenAIChatExtractor:
    """Minimal ``complete()`` adapter over an OpenAI-compatible endpoint.

    Defaults to the SAME endpoint as generation. That co-location is the point:
    [AM] §4.3's construction/generation interference finding is only observable
    when both share a serving stack. Point ``base_url`` elsewhere to run the
    separation as an explicit experiment.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        *,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 1024,
    ) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        return text, dict(payload.get("usage") or {})
