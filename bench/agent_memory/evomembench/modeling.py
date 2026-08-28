"""Map completed EvoMemBench episodes into GEM ExperienceUnits and associations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Mapping, Sequence

from bench.agent_memory.evomembench.dataset import EvoEpisode
from bench.agent_memory.evomembench.leakage import assert_online_payload
from bench.agent_memory.evomembench.task_signature import (
    knowledge_task_signature,
    tool_task_signature,
)
from bench.agent_memory.evomembench.tool_dataset import ToolEpisode
from bench.agent_memory.gem import plan as planmod
from bench.agent_memory.gem.types import (
    EdgeKind,
    InteractionEvent,
    Query,
    RetrievalMode,
)


POSTGRES_NUL_REPLACEMENT = "\ufffd"


def sanitize_postgres_text(value: str) -> str:
    """Map PostgreSQL-forbidden NUL characters to an auditable Unicode marker."""
    return value.replace("\x00", POSTGRES_NUL_REPLACEMENT)


def _assert_postgres_text_safe(value: Any, *, path: str) -> None:
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(f"PostgreSQL text contains NUL at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_postgres_text_safe(key, path=f"{path}.<key>")
            _assert_postgres_text_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_postgres_text_safe(item, path=f"{path}[{index}]")


def _nul_count(*values: str) -> int:
    return sum(value.count("\x00") for value in values)


@dataclass(frozen=True)
class ExperienceFeature:
    kind: str
    value: str
    relation: str


@dataclass(frozen=True)
class ExperienceUnit:
    """The semantic atom: vector signature + injectible experience + metadata."""

    uid: str
    scope_id: str
    ordinal: int
    task_signature: str
    memory_payload: str
    source_external_ids: tuple[str, ...]
    features: tuple[ExperienceFeature, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("experience ordinal must be non-negative")
        _assert_postgres_text_safe(
            {
                "uid": self.uid,
                "scope_id": self.scope_id,
                "task_signature": self.task_signature,
                "memory_payload": self.memory_payload,
                "source_external_ids": self.source_external_ids,
                "features": tuple(
                    (feature.kind, feature.value, feature.relation)
                    for feature in self.features
                ),
                "metadata": self.metadata,
            },
            path="experience_unit",
        )
        assert_online_payload(
            {
                "task_signature": self.task_signature,
                "memory_payload": self.memory_payload,
                "metadata": self.metadata,
            }
        )


def knowledge_experience(
    episode: EvoEpisode, *, response: str, scope_id: str
) -> ExperienceUnit:
    user_turns = [item.content for item in episode.messages if item.role == "user"]
    raw_signature = knowledge_task_signature(episode)
    raw_memory_payload = "\n".join(
        [
            f"Prior task ({episode.subcategory}): " + "\n".join(user_turns),
            "Prior response:",
            response.strip(),
        ]
    )
    raw_feature_values = (episode.category, episode.subcategory)
    replacements = _nul_count(
        episode.episode_uid,
        scope_id,
        raw_signature,
        raw_memory_payload,
        episode.source_task_id,
        *raw_feature_values,
    )
    return ExperienceUnit(
        uid=sanitize_postgres_text(episode.episode_uid),
        scope_id=sanitize_postgres_text(scope_id),
        ordinal=episode.ordinal,
        task_signature=sanitize_postgres_text(raw_signature),
        memory_payload=sanitize_postgres_text(raw_memory_payload),
        source_external_ids=(sanitize_postgres_text(episode.source_task_id),),
        features=(
            ExperienceFeature(
                "category", sanitize_postgres_text(episode.category), "HAS_CATEGORY"
            ),
            ExperienceFeature(
                "skill", sanitize_postgres_text(episode.subcategory), "USES_SKILL"
            ),
        ),
        metadata={
            "benchmark": "CrossEp-Know",
            "context_uid": sanitize_postgres_text(episode.context_uid),
            "episode_uid": sanitize_postgres_text(episode.episode_uid),
            "postgres_nul_replacements": replacements,
        },
    )


def tool_experience(
    episode: ToolEpisode,
    *,
    response: str,
    scope_id: str,
    allowed_functions: Sequence[str] = (),
) -> ExperienceUnit:
    question = [
        [{"role": item.role, "content": item.content} for item in turn]
        for turn in episode.question
    ]
    raw_signature = tool_task_signature(
        question=question,
        involved_classes=episode.involved_classes,
        allowed_functions=allowed_functions,
    )
    user_turns = [
        item.content
        for turn in episode.question
        for item in turn
        if item.role == "user"
    ]
    raw_memory_payload = "\n".join(
        ["Prior tool task: " + "\n".join(user_turns), "Prior trajectory:", response]
    )
    raw_feature_values = (
        episode.category,
        *episode.involved_classes,
        *allowed_functions,
    )
    replacements = _nul_count(
        episode.episode_uid,
        scope_id,
        raw_signature,
        raw_memory_payload,
        episode.source_id,
        *raw_feature_values,
    )
    features = [
        ExperienceFeature(
            "environment", sanitize_postgres_text(episode.category), "HAS_ENVIRONMENT"
        )
    ]
    features.extend(
        ExperienceFeature("tool", sanitize_postgres_text(name), "USES_TOOL")
        for name in episode.involved_classes
    )
    features.extend(
        ExperienceFeature("function", sanitize_postgres_text(name), "USES_FUNCTION")
        for name in allowed_functions
    )
    return ExperienceUnit(
        uid=sanitize_postgres_text(episode.episode_uid),
        scope_id=sanitize_postgres_text(scope_id),
        ordinal=episode.ordinal,
        task_signature=sanitize_postgres_text(raw_signature),
        memory_payload=sanitize_postgres_text(raw_memory_payload),
        source_external_ids=(sanitize_postgres_text(episode.source_id),),
        features=tuple(features),
        metadata={
            "benchmark": "CrossEp-Tool",
            "episode_uid": sanitize_postgres_text(episode.episode_uid),
            "environment": sanitize_postgres_text(episode.category),
            "involved_classes": [
                sanitize_postgres_text(value) for value in episode.involved_classes
            ],
            "postgres_nul_replacements": replacements,
        },
    )


def _feature_ref(feature: ExperienceFeature) -> str:
    digest = hashlib.sha256(f"{feature.kind}\0{feature.value}".encode()).hexdigest()
    return f"feature:{digest}"


def experience_plan(unit: ExperienceUnit, *, valid_from: str) -> list[dict[str, Any]]:
    """Produce bounded plan ops; topology is written only through the native AM."""
    experience_ref = f"experience:{unit.uid}"
    metadata = {
        **dict(unit.metadata),
        "node_kind": "experience",
        "experience_ordinal": unit.ordinal,
        "task_signature_schema": "evomembench_task_signature_v0.1.0",
        "evaluation_clean": True,
    }
    ops: list[dict[str, Any]] = [
        planmod.upsert_unit(
            scope_id=unit.scope_id,
            title=unit.uid,
            summary=unit.task_signature[:1000],
            ref=experience_ref,
            embed_text=unit.task_signature,
            metadata=metadata,
        ),
        planmod.append_field_value(
            ref=experience_ref,
            field="memory_payload",
            value=unit.memory_payload,
            valid_from=valid_from,
            supersede_current=True,
            provenance={
                "source_external_ids": list(unit.source_external_ids),
                "operator": "evomembench_experience_ingest",
                "prompt_version": "evaluation_clean_v0.1.0",
            },
        ),
    ]
    seen: set[tuple[str, str]] = set()
    for feature in unit.features:
        key = (feature.kind, feature.value)
        if key in seen:
            continue
        seen.add(key)
        ref = _feature_ref(feature)
        ops.append(
            planmod.upsert_unit(
                scope_id=unit.scope_id,
                title=ref,
                summary=f"{feature.kind}: {feature.value}",
                ref=ref,
                embed_text=f"{feature.kind}: {feature.value}",
                metadata={
                    "node_kind": "feature",
                    "feature_kind": feature.kind,
                    "feature_value": feature.value,
                    "evaluation_clean": True,
                },
            )
        )
        # Retrieval associations are symmetric.  Both arcs are native-AM edges;
        # gem_edge keeps only relation/provenance metadata.
        ops.extend(
            [
                planmod.link(
                    src_ref=experience_ref,
                    dst_ref=ref,
                    edge_kind=EdgeKind.ASSOCIATION.value,
                    rel=feature.relation,
                ),
                planmod.link(
                    src_ref=ref,
                    dst_ref=experience_ref,
                    edge_kind=EdgeKind.ASSOCIATION.value,
                    rel=f"{feature.relation}_INV",
                ),
            ]
        )
    return ops


def reuse_existing_feature_units(
    ops: Sequence[Mapping[str, Any]], *, scope_id: str, view: Any
) -> list[dict[str, Any]]:
    """Avoid re-embedding/updating shared feature hubs on every episode."""
    ref_to_existing_id: dict[str, int] = {}
    declared_feature_refs: set[str] = set()
    retained: list[dict[str, Any]] = []
    for raw in ops:
        op = dict(raw)
        is_feature = (
            op.get("kind") == planmod.UPSERT_UNIT
            and (op.get("metadata") or {}).get("node_kind") == "feature"
        )
        if not is_feature:
            retained.append(op)
            continue
        ref = str(op["ref"])
        existing = view.unit_by_title(scope_id, str(op["title"]))
        if existing is not None:
            ref_to_existing_id[ref] = int(existing.id)
            continue
        if ref in declared_feature_refs:
            continue
        declared_feature_refs.add(ref)
        retained.append(op)

    for op in retained:
        if op.get("kind") != planmod.LINK:
            continue
        for side in ("src", "dst"):
            ref_key = f"{side}_ref"
            ref = op.get(ref_key)
            if ref in ref_to_existing_id:
                op[ref_key] = None
                op[side] = ref_to_existing_id[str(ref)]
    return retained


class PreparedExperienceStrategy:
    """A no-LLM strategy that admits one already constructed ExperienceUnit."""

    name = "deterministic_experience_graph"

    def __init__(self, unit: ExperienceUnit, *, valid_from: str) -> None:
        self.unit = unit
        self.valid_from = valid_from

    def plan(
        self, events: Sequence[InteractionEvent], view: Any
    ) -> list[Mapping[str, Any]]:
        if (
            len(events) != 1
            or events[0].external_id not in self.unit.source_external_ids
        ):
            raise ValueError("prepared experience does not match its ingestion event")
        ops = reuse_existing_feature_units(
            experience_plan(self.unit, valid_from=self.valid_from),
            scope_id=self.unit.scope_id,
            view=view,
        )
        previous = view.latest_experience_before(self.unit.scope_id, self.unit.ordinal)
        if previous is not None:
            current_ref = f"experience:{self.unit.uid}"
            # Native association topology carries both navigable directions;
            # PRECEDES/FOLLOWS remain relation metadata, not relational joins.
            ops.extend(
                [
                    planmod.link(
                        src=previous.id,
                        dst_ref=current_ref,
                        edge_kind=EdgeKind.ASSOCIATION.value,
                        rel="PRECEDES",
                    ),
                    planmod.link(
                        src_ref=current_ref,
                        dst=previous.id,
                        edge_kind=EdgeKind.ASSOCIATION.value,
                        rel="FOLLOWS",
                    ),
                ]
            )
        return ops


def experience_query(
    *,
    scope_id: str,
    task_signature: str,
    cutoff_ordinal: int,
    mode: RetrievalMode = RetrievalMode.FUSED,
    top_k: int = 10,
    hops: int = 2,
    m_seeds: int = 4,
    term_cond: int = 32,
    reinforce: bool = False,
    validity_states: Sequence[str] = (),
    source_phases: Sequence[str] = (),
) -> Query:
    """Construct the canonical vector + association graph + cutoff filter query."""
    if cutoff_ordinal < 0:
        raise ValueError("cutoff_ordinal must be non-negative")
    assert_online_payload({"task_signature": task_signature})
    return Query(
        scope_id=scope_id,
        text=task_signature,
        k=top_k,
        mode=mode,
        reinforce=reinforce,
        hops=hops,
        m_seeds=m_seeds,
        term_cond=term_cond,
        graph_edge_kind=EdgeKind.ASSOCIATION,
        extra_filter=build_experience_filter(
            cutoff_ordinal=cutoff_ordinal,
            validity_states=validity_states,
            source_phases=source_phases,
        ),
    )


_FILTER_VALUE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _literal_list(values: Sequence[str], *, field: str) -> str:
    normalized = sorted(set(str(value) for value in values))
    if not normalized or any(
        not _FILTER_VALUE.fullmatch(value) for value in normalized
    ):
        raise ValueError(f"unsafe or empty {field} eligibility value")
    return ", ".join("'" + value.replace("'", "''") + "'" for value in normalized)


def build_experience_filter(
    *,
    cutoff_ordinal: int,
    validity_states: Sequence[str] = (),
    source_phases: Sequence[str] = (),
) -> str:
    """Build only whitelisted metadata predicates; no dataset text is SQL."""
    if cutoff_ordinal < 0:
        raise ValueError("cutoff_ordinal must be non-negative")
    clauses = [
        "metadata->>'node_kind' = 'experience'",
        f"(metadata->>'experience_ordinal')::integer < {int(cutoff_ordinal)}",
    ]
    if validity_states:
        clauses.append(
            "metadata->>'validity_state' IN ("
            + _literal_list(validity_states, field="validity state")
            + ")"
        )
    if source_phases:
        clauses.append(
            "metadata->>'source_phase' IN ("
            + _literal_list(source_phases, field="source phase")
            + ")"
        )
    return " AND ".join(clauses)
