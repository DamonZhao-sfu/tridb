"""The ``ingest`` operator — integrate ``I_t``, do not append blindly.

    M_{t+1} = U(M_t, I_t, {})

Strategy-agnostic by construction: this module validates and APPLIES a write
plan, and knows nothing about how the plan was produced. That is what makes
Omri et al.'s four construction forms configurations of one memory rather than
four systems.

**Transaction shape** — one transaction for the whole call::

    advisory lock
      -> strategy.plan(events, view)        # may call LLM/embeddings; NO writes
      -> validate plan (referential + schema gates)
      -> apply writes (allocate vid -> row -> vertex -> fields -> edges)
      -> refresh embeddings for touched units   (ONE batched embed call)
      -> flag extension-linked units for revision   (C3 hook, [GEM] Alg. 1 line 4)
      -> evaluate P_t postconditions
      -> log gem_transition -> COMMIT

**Model calls happen in ``plan()``, inside the transaction.** Deliberate: it
keeps atomicity, and it makes LLM latency visible in the transaction duration,
which is what [AM] §4.6's freshness argument needs to measure. The alternative
— plan outside, apply inside — is a small move but opens a TOCTOU window
against ``M_t`` that would then need revalidating. Plan-inside first; revisit
only if measured.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.store import GemStore, Tx, TriDBMemoryView, vec_literal
from bench.agent_memory.gem.types import (
    EdgeKind,
    IngestResult,
    InteractionEvent,
    PhaseCost,
    StateDelta,
)

_EMPTY_DELTA = StateDelta()


def _zero_cost(phase: str, seconds: float) -> PhaseCost:
    return PhaseCost(phase=phase, seconds=seconds)


class PlanValidationError(Exception):
    """A plan failed a gate. Carries the reason so it can be counted.

    [AM] §4.4: below an algorithm-specific capability floor a weak construction
    model does not merely lower accuracy, it CORRUPTS the store. So a validation
    failure is recorded as a rejection with its reason, and a run whose rejection
    rate exceeds the configured threshold is reported as a **failed
    configuration**, not as a low-accuracy datapoint.
    """

    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def validate_plan(ops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The gates, in order. Returns the rejections; never raises.

    Each gate rejects the offending OP, not the run — a single malformed
    extraction must not discard a whole ingest batch.

    1. schema — known ``kind``, required keys present and correctly typed;
    2. referential validity — every op names a unit by ``ref`` or ``unit_id``;
    3. no dangling vertex — every ``ref`` used is defined by some ``upsert_unit``
       in the SAME plan, so it will have a vid after apply;
    4. enum conformance — ``edge_kind`` is one of the two native kinds.
    """
    rejections: list[dict[str, Any]] = []
    defined_refs = {
        op.get("ref")
        for op in ops
        if op.get("kind") == planmod.UPSERT_UNIT and op.get("ref")
    }

    def reject(index: int, op: Mapping[str, Any], gate: str, detail: str) -> None:
        rejections.append(
            {"index": index, "kind": op.get("kind"), "gate": gate, "detail": detail}
        )

    for index, op in enumerate(ops):
        kind = op.get("kind")
        if kind not in planmod.KINDS:
            reject(index, op, "schema", f"unknown op kind {kind!r}")
            continue

        if kind == planmod.UPSERT_UNIT:
            if not op.get("scope_id"):
                reject(index, op, "schema", "upsert_unit needs scope_id")
            if not op.get("title"):
                reject(index, op, "schema", "upsert_unit needs a non-empty title")
            if op.get("embedding") is None and not op.get("embed_text"):
                reject(index, op, "schema", "upsert_unit needs embedding or embed_text")

        elif kind == planmod.APPEND_FIELD_VALUE:
            if not op.get("field"):
                reject(index, op, "schema", "append_field_value needs a field")
            if op.get("value") is None:
                reject(index, op, "schema", "append_field_value needs a value")
            if not op.get("valid_from"):
                reject(index, op, "schema", "append_field_value needs valid_from")
            target_ref, target_id = op.get("ref"), op.get("unit_id")
            if target_ref is None and target_id is None:
                reject(index, op, "referential", "no ref and no unit_id")
            elif target_ref is not None and target_ref not in defined_refs:
                reject(
                    index,
                    op,
                    "dangling_vertex",
                    f"ref {target_ref!r} is not created by any upsert_unit in this plan",
                )

        elif kind == planmod.LINK:
            edge_kind = op.get("edge_kind")
            if edge_kind not in (EdgeKind.EXTENSION.value, EdgeKind.ASSOCIATION.value):
                reject(
                    index,
                    op,
                    "enum",
                    f"edge_kind {edge_kind!r} is not extension|association",
                )
            for side in ("src", "dst"):
                ref, ident = op.get(f"{side}_ref"), op.get(side)
                if ref is None and ident is None:
                    reject(index, op, "referential", f"link has no {side}")
                elif ref is not None and ref not in defined_refs:
                    reject(
                        index,
                        op,
                        "dangling_vertex",
                        f"{side}_ref {ref!r} is not created by any upsert_unit",
                    )

        elif kind == planmod.SPLIT_TOPIC:
            if op.get("unit_id") is None:
                reject(index, op, "referential", "split_topic needs unit_id")
            if not op.get("fields"):
                reject(index, op, "schema", "split_topic needs a non-empty field list")

    return rejections


class IngestOperator:
    """Applies a validated plan inside one transition.

    Held separately from ``TriDBGovernedMemory`` so ``revise`` can reuse the
    same apply path — a repair and an ingest write the same rows and must not
    drift into two spellings of supersession.
    """

    def __init__(self, store: GemStore, *, embedder: Any | None = None) -> None:
        self.store = store
        self.embedder = embedder

    # -- apply ----------------------------------------------------------

    def apply(
        self, tx: Tx, ops: Sequence[Mapping[str, Any]], *, scope_id: str
    ) -> dict[str, Any]:
        """Apply the ops. Returns the ref->unit_id map and the touched units."""
        tx.lock_writer()
        resolved: dict[str, int] = {}
        touched: list[int] = []

        self._resolve_embeddings(tx, ops)

        for op in ops:
            if op.get("kind") != planmod.UPSERT_UNIT:
                continue
            unit_id = self._apply_upsert(tx, op, scope_id=scope_id)
            if op.get("ref"):
                resolved[str(op["ref"])] = unit_id
            touched.append(unit_id)

        for op in ops:
            if op.get("kind") != planmod.APPEND_FIELD_VALUE:
                continue
            unit_id = self._target(op, resolved, side=None)
            self._apply_field(tx, op, unit_id)
            if unit_id not in touched:
                touched.append(unit_id)

        linked_sources: list[int] = []
        for op in ops:
            if op.get("kind") != planmod.LINK:
                continue
            src = self._target(op, resolved, side="src")
            dst = self._target(op, resolved, side="dst")
            self.store.link(
                tx,
                src,
                dst,
                kind=op["edge_kind"],
                rel=op.get("rel") or op["edge_kind"],
                weight=float(op.get("weight", 1.0)),
            )
            if op["edge_kind"] == EdgeKind.EXTENSION.value:
                linked_sources.append(src)

        # C3 hook ([GEM] Alg. 1 line 4): a unit that just gained an extension
        # out-edge has dependents that may now need re-evaluating. Flag them for
        # `revise`; ingest never propagates itself, because propagation is a
        # revision decision and must not become an ingest side effect.
        flagged = self._flag_for_revision(tx, linked_sources)

        return {"resolved": resolved, "touched": touched, "flagged": flagged}

    @staticmethod
    def _target(
        op: Mapping[str, Any], resolved: Mapping[str, int], *, side: str | None
    ) -> int:
        """Resolve one op's unit reference to a real vid.

        ``side`` is ``"src"``/``"dst"`` for links and None for field ops. A ref
        that survived validation but does not resolve here means apply ran out
        of order — a bug, not bad input, so it raises rather than rejects.
        """
        ref_key = "ref" if side is None else f"{side}_ref"
        id_key = "unit_id" if side is None else side
        identifier = op.get(id_key)
        if identifier is not None:
            return int(identifier)
        ref = op.get(ref_key)
        if ref is not None and str(ref) in resolved:
            return resolved[str(ref)]
        raise PlanValidationError(
            "referential", f"{ref_key}={ref!r} did not resolve to a unit id"
        )

    def _resolve_embeddings(self, tx: Tx, ops: Sequence[Mapping[str, Any]]) -> None:
        """ONE batched embedding call for every op that needs a vector.

        Re-embedding is the expensive write (0% HOT, ~369 B of relation growth
        per update — measured, interface §6.3), so it is batched here rather
        than issued per unit.
        """
        pending = [
            op
            for op in ops
            if op.get("kind") == planmod.UPSERT_UNIT
            and op.get("embedding") is None
            and op.get("embed_text")
        ]
        if not pending:
            return
        if self.embedder is None:
            raise PlanValidationError(
                "schema", "plan needs embed_text resolved but no embedder configured"
            )
        texts = [str(op["embed_text"]) for op in pending]
        vectors = self.embedder.encode(texts)
        if len(vectors) != len(pending):
            raise PlanValidationError(
                "schema",
                f"embedder returned {len(vectors)} vectors for {len(pending)} inputs",
            )
        tx.meter.embed_calls += 1
        tx.meter.embed_sequences += len(texts)
        for op, vector in zip(pending, vectors, strict=True):
            # ops are plain dicts by contract; mutate in place so apply below
            # sees the resolved vector.
            op["embedding"] = [float(value) for value in vector]  # type: ignore[index]

    def _apply_upsert(self, tx: Tx, op: Mapping[str, Any], *, scope_id: str) -> int:
        embedding = op.get("embedding")
        if embedding is None:
            raise PlanValidationError(
                "schema", "upsert_unit reached apply with no vector"
            )
        literal = vec_literal(embedding)
        metadata = json.dumps(dict(op.get("metadata") or {}), sort_keys=True)

        unit_id = op.get("unit_id")
        if unit_id is None:
            # A title collision is an UPDATE, not a second unit: the unit is the
            # topic, and two rows for one topic is the entity-grain scatter
            # [GEM] §4.1 rejects.
            row = tx.execute(
                "SELECT id FROM gem_unit WHERE scope_id = %s AND title = %s",
                (scope_id, op["title"]),
            ).fetchone()
            unit_id = None if row is None else int(row[0])

        if unit_id is None:
            unit_id = tx.allocate_vertex()
            tx.execute(
                "INSERT INTO gem_unit (id, scope_id, title, summary, embedding,"
                " metadata) VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)",
                (
                    unit_id,
                    scope_id,
                    op["title"],
                    op.get("summary") or "",
                    literal,
                    metadata,
                ),
            )
            tx.delta.units_created += 1
        else:
            tx.execute(
                "UPDATE gem_unit SET title = %s, summary = %s,"
                " embedding = %s::vector, metadata = %s::jsonb WHERE id = %s",
                (op["title"], op.get("summary") or "", literal, metadata, unit_id),
            )
            tx.delta.units_updated += 1
        return int(unit_id)

    def _apply_field(self, tx: Tx, op: Mapping[str, Any], unit_id: int) -> int:
        """Append a value, superseding the current one when asked.

        The three statements below are the C1/C4 core. The partial unique index
        makes a plan that would leave two current values abort at commit, so no
        application-level check is needed — and none is written here on purpose.
        """
        provenance = dict(op.get("provenance") or {})
        old_id: int | None = None

        if op.get("supersede_current"):
            row = tx.execute(
                "UPDATE gem_field_value SET valid_to = %s"
                " WHERE unit_id = %s AND field = %s AND valid_to IS NULL"
                " RETURNING id",
                (op["valid_from"], unit_id, op["field"]),
            ).fetchone()
            old_id = None if row is None else int(row[0])

        row = tx.execute(
            "INSERT INTO gem_field_value (unit_id, field, value, valid_from,"
            " source_external_ids, source_unit_ids, extractor_model,"
            " prompt_version, confidence, operator, transition_id)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                unit_id,
                op["field"],
                op["value"],
                op["valid_from"],
                list(provenance.get("source_external_ids") or []) or None,
                list(provenance.get("source_unit_ids") or []) or None,
                provenance.get("extractor_model"),
                provenance.get("prompt_version"),
                provenance.get("confidence"),
                provenance.get("operator") or tx.operator,
                provenance.get("transition_id"),
            ),
        ).fetchone()
        new_id = int(row[0])
        tx.delta.fields_appended += 1

        if old_id is not None:
            # C4: the chain, not just the closed interval. A superseded value
            # must stay reachable FROM its replacement.
            tx.execute(
                "UPDATE gem_field_value SET superseded_by = %s WHERE id = %s",
                (new_id, old_id),
            )
            tx.delta.values_superseded += 1
        return new_id

    def _flag_for_revision(self, tx: Tx, sources: Sequence[int]) -> list[int]:
        """Mark extension-reachable dependents as needing re-evaluation (C3).

        One-hop and materialised: the graph leg is commit-visible rather than
        snapshot-isolated (proven, interface §6.5), so anything that
        re-traverses can see edges appear mid-walk. ``revise`` does the
        multi-hop walk; ingest only records the frontier it created.
        """
        if not sources:
            return []
        rows = tx.execute(
            "SELECT DISTINCT dst FROM gem_edge WHERE src = ANY(%s)"
            " AND kind = 'extension' AND tombstoned_at IS NULL",
            (list({int(s) for s in sources}),),
        ).fetchall()
        flagged = sorted(int(row[0]) for row in rows)
        if flagged:
            tx.execute(
                "UPDATE gem_unit SET metadata = jsonb_set(metadata,"
                " '{needs_revision}', 'true'::jsonb) WHERE id = ANY(%s)",
                (flagged,),
            )
            tx.delta.propagated_units.extend(flagged)
        return flagged

    # -- the operator ----------------------------------------------------

    def ingest(
        self,
        events: Sequence[InteractionEvent],
        *,
        strategy: Any,
        scope_id: str | None = None,
        max_rejection_rate: float | None = None,
    ) -> IngestResult:
        """One transition: plan, validate, apply, evaluate, commit.

        ``max_rejection_rate`` implements [AM] §4.4's capability floor: above
        it, the run is a FAILED CONFIGURATION and the transition aborts rather
        than committing a partially corrupted store.
        """
        if scope_id is None:
            scopes = {event.scope_id for event in events}
            if len(scopes) != 1:
                raise ValueError(
                    f"ingest needs exactly one scope; events span {sorted(scopes)}"
                )
            scope_id = scopes.pop()

        started = time.perf_counter()
        units: tuple[int, ...] = ()
        rejections: list[dict[str, Any]] = []
        try:
            with self.store.transition("ingest", scope_id, "construction") as tx:
                tx.lock_writer()
                view = TriDBMemoryView(self.store, tx=tx)
                ops = list(strategy.plan(events, view))
                _absorb_strategy_cost(tx, strategy)
                # A strategy's own gate failures (a malformed LLM extraction,
                # an unknown tool) are rejections too, and must land in the
                # same accounting as the plan gates — otherwise a run's
                # rejection rate understates the capability floor it hit.
                strategy_rejections = [
                    dict(item) for item in getattr(strategy, "rejections", []) or []
                ]
                capped = bool(getattr(strategy, "capped", False))

                rejections = validate_plan(ops)
                rejected_indices = {item["index"] for item in rejections}
                accepted = [
                    op for index, op in enumerate(ops) if index not in rejected_indices
                ]

                if max_rejection_rate is not None and (ops or strategy_rejections):
                    attempted = len(ops) + len(strategy_rejections)
                    rate = (len(rejections) + len(strategy_rejections)) / attempted
                    if rate > max_rejection_rate:
                        raise PlanValidationError(
                            "capability_floor",
                            f"rejection rate {rate:.3f} exceeds "
                            f"{max_rejection_rate:.3f} — FAILED CONFIGURATION, "
                            "not an accuracy datapoint ([AM] §4.4)",
                        )

                applied = self.apply(tx, accepted, scope_id=scope_id)
                units = tuple(applied["touched"])
                rejections = [*strategy_rejections, *rejections]
                delta = tx.delta.freeze()
                cost = tx.meter.freeze(time.perf_counter() - started)
                transition_id = tx.transition_id
                policies = tuple(tx.policies_evaluated)
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            return IngestResult(
                operator="ingest",
                committed=False,
                delta=_EMPTY_DELTA,
                cost=_zero_cost("construction", time.perf_counter() - started),
                aborted_reason=f"{type(exc).__name__}: {exc}",
                rejected=tuple(rejections),
            )

        return IngestResult(
            operator="ingest",
            committed=True,
            delta=delta,
            cost=cost,
            transition_id=transition_id,
            policies_evaluated=policies,
            units=units,
            rejected=tuple(rejections),
            capped=capped,
        )


def _absorb_strategy_cost(tx: Tx, strategy: Any) -> None:
    """Fold a strategy's own LLM/embedding counters into the transition meter.

    Strategies that call a model expose a ``cost`` mapping; those that do not
    (deterministic) expose nothing and contribute zero.
    """
    cost = getattr(strategy, "cost", None)
    if not cost:
        return
    for attribute in (
        "llm_calls",
        "embed_calls",
        "embed_sequences",
        "prompt_tokens",
        "completion_tokens",
        "embed_input_tokens",
    ):
        setattr(
            tx.meter,
            attribute,
            getattr(tx.meter, attribute) + int(cost.get(attribute, 0)),
        )
