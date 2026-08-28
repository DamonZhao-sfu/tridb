"""Independent synthetic history expansion for the systems-only track."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterator, Mapping, Sequence

from bench.agent_memory.evomembench.modeling import (
    ExperienceFeature,
    ExperienceUnit,
    experience_plan,
    reuse_existing_feature_units,
)
from bench.agent_memory.gem.types import InteractionEvent

SCALE_KINDS = (
    "cross_scope_hard_negative",
    "same_domain_semantic_negative",
    "graph_decoy_branch",
    "stale_superseded_version",
)


@dataclass(frozen=True)
class ScaleRecord:
    index: int
    kind: str
    unit: ExperienceUnit
    content_sha256: str


def iter_scaled_experiences(
    *,
    count: int,
    primary_scope: str,
    seed: int,
    start_index: int = 0,
    forbidden_content_hashes: Sequence[str] = (),
) -> Iterator[ScaleRecord]:
    """Yield unique decoys; never replicate a native target or positive record."""
    if count < 0:
        raise ValueError("scaled history count must be non-negative")
    if start_index < 0:
        raise ValueError("scaled history start index must be non-negative")
    forbidden = set(forbidden_content_hashes)
    observed: set[str] = set()
    for index in range(start_index, start_index + count):
        kind = SCALE_KINDS[index % len(SCALE_KINDS)]
        nonce = int.from_bytes(
            hashlib.sha256(
                f"evomembench-systems-scale:{seed}:{index}".encode()
            ).digest()[:16],
            "big",
        )
        task = (
            "benchmark: CrossEp systems-only decoy\n"
            f"kind: {kind}\nindex: {index}\nnonce: {nonce:032x}\n"
            "user: solve a related but independently generated agent task"
        )
        payload = (
            f"Synthetic systems-track experience {index}; class={kind}; "
            f"nonce={nonce:032x}. This is not an answer to any native target."
        )
        content_hash = hashlib.sha256(f"{task}\0{payload}".encode()).hexdigest()
        if content_hash in forbidden or content_hash in observed:
            raise RuntimeError(
                "synthetic expansion produced a duplicate/forbidden content hash"
            )
        observed.add(content_hash)
        scope = (
            f"{primary_scope}:negative:{index % 64}"
            if kind == "cross_scope_hard_negative"
            else primary_scope
        )
        feature_value = (
            f"decoy-branch-{index % 32}"
            if kind == "graph_decoy_branch"
            else f"systems-{kind}"
        )
        validity_state = "stale" if kind == "stale_superseded_version" else "active"
        unit = ExperienceUnit(
            uid=f"evomembench:systems:decoy:{seed}:{index}",
            scope_id=scope,
            ordinal=index,
            task_signature=task,
            memory_payload=payload,
            source_external_ids=(f"systems-decoy-{seed}-{index}",),
            features=(ExperienceFeature("skill", feature_value, "USES_SKILL"),),
            metadata={
                "benchmark": "EvoMemBench-scaled-systems",
                "scale_kind": kind,
                "validity_state": validity_state,
                "content_sha256": content_hash,
                "systems_only": True,
            },
        )
        yield ScaleRecord(
            index=index, kind=kind, unit=unit, content_sha256=content_hash
        )


class BatchExperienceStrategy:
    """Bounded batched writer; one batch is one PostgreSQL transaction."""

    name = "evomembench_scaled_batch_v0.1.0"

    def __init__(self, units: Sequence[ExperienceUnit], *, valid_from: str) -> None:
        if not units:
            raise ValueError("scaled batch cannot be empty")
        scopes = {unit.scope_id for unit in units}
        if len(scopes) != 1:
            raise ValueError("one scaled batch may write only one scope")
        self.units = tuple(units)
        self.valid_from = valid_from

    def plan(
        self, events: Sequence[InteractionEvent], view: Any
    ) -> list[Mapping[str, Any]]:
        expected = {
            external_id
            for unit in self.units
            for external_id in unit.source_external_ids
        }
        observed = {event.external_id for event in events}
        if expected != observed:
            raise ValueError("scaled batch events do not match prepared units")
        ops = [
            op
            for unit in self.units
            for op in experience_plan(unit, valid_from=self.valid_from)
        ]
        return reuse_existing_feature_units(
            ops, scope_id=self.units[0].scope_id, view=view
        )
