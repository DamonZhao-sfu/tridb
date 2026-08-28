"""Stable data contracts for E0 plan-space observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class PlanSpec:
    shape: str
    k: int
    hops: int
    predicate_placement: str
    is_default: bool = False

    @property
    def plan_id(self) -> str:
        payload = {
            "hops": self.hops,
            "k": self.k,
            "predicate_placement": self.predicate_placement,
            "shape": self.shape,
        }
        digest = hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:12]
        return f"{self.shape}-{digest}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "shape": self.shape,
            "k": self.k,
            "hops": self.hops,
            "predicate_placement": self.predicate_placement,
            "is_default": self.is_default,
        }


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    dataset: str
    query_text: str
    anchor_ids: tuple[Any, ...]
    answer_ids: tuple[Any, ...]
    edge_types: tuple[str, ...]
    hop_limit: int
    structured_predicate: dict[str, Any]
    target_entity_type: str
    template: str
    annotation_status: str
    require_each_anchor: bool

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "QuerySpec":
        audit = row.get("path_audit") or row.get("audit") or {}
        return cls(
            query_id=str(row["query_id"]),
            dataset=str(row.get("dataset", "unknown")),
            query_text=str(row["query_text"]),
            anchor_ids=tuple(row["anchor_ids"]),
            answer_ids=tuple(row["answer_ids"]),
            edge_types=tuple(str(value) for value in row["edge_types"]),
            hop_limit=int(row["hop_limit"]),
            structured_predicate=dict(row.get("structured_predicate") or {}),
            target_entity_type=str(row.get("target_entity_type", "")),
            template=str(row.get("template", "unknown")),
            annotation_status=str(row.get("annotation_status", "unknown")),
            require_each_anchor=bool(audit.get("required_from_each_anchor", False)),
        )


def quality_metrics(
    result_ids: list[Any], answer_ids: tuple[Any, ...]
) -> dict[str, float]:
    answers = set(answer_ids)
    first_rank = next(
        (
            rank
            for rank, node_id in enumerate(result_ids, start=1)
            if node_id in answers
        ),
        None,
    )
    top20 = result_ids[:20]
    return {
        "hit_at_1": float(bool(result_ids and result_ids[0] in answers)),
        "hit_at_5": float(any(node_id in answers for node_id in result_ids[:5])),
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "recall_at_20": (
            0.0
            if not answers
            else len(answers.intersection(top20)) / float(len(answers))
        ),
    }
