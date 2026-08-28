"""Atomic admission of an extracted EvoMemBench experience into GEM."""

from __future__ import annotations

from dataclasses import dataclass

from bench.agent_memory.evomembench.extractor import ExtractedExperience
from bench.agent_memory.evomembench.modeling import (
    ExperienceFeature,
    ExperienceUnit,
    PreparedExperienceStrategy,
)
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.gem.types import IngestResult, InteractionEvent


@dataclass(frozen=True)
class Admission:
    unit: ExperienceUnit
    result: IngestResult


def admit_extracted_experience(
    memory: TriDBGovernedMemory,
    *,
    uid: str,
    scope_id: str,
    ordinal: int,
    task_signature: str,
    extracted: ExtractedExperience,
    valid_from: str,
    metadata: dict[str, object] | None = None,
) -> Admission:
    """Write row, vector, fields, concept vertices and native edges in one xid."""
    unit = ExperienceUnit(
        uid=uid,
        scope_id=scope_id,
        ordinal=ordinal,
        task_signature=task_signature,
        memory_payload=extracted.memory_payload,
        source_external_ids=extracted.source_external_ids,
        features=tuple(
            ExperienceFeature(item.kind, item.value, f"HAS_{item.kind.upper()}")
            for item in extracted.concepts
        ),
        metadata={
            **(metadata or {}),
            "extractor_schema": extracted.extractor_schema,
            "extractor_input_sha256": extracted.input_sha256,
            "memory_payload_sha256": extracted.payload_sha256,
            "extractor_capped": extracted.capped,
        },
    )
    event = InteractionEvent(
        scope_id=scope_id,
        external_id=extracted.source_external_ids[0],
        content=extracted.memory_payload,
        session_id=uid,
        role="agent_trajectory",
        kind="experience",
        event_time=valid_from,
        event_order=ordinal,
    )
    result = memory.ingest(
        [event],
        strategy=PreparedExperienceStrategy(unit, valid_from=valid_from),
        scope_id=scope_id,
    )
    return Admission(unit=unit, result=result)
