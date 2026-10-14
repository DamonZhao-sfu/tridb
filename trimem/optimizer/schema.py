"""Versioned metadata-only retrieval actions; no executed rows in the search space."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any

PLAN_IDS = (
    "no_memory_v1",
    "vgr_v1",
    "vgr_score_v1",
    "seed_sem_expand_v1",
    "expand_row_sem_v1",
    # SemFilter plans without the trailing StreamScore (no score configuration).
    "seed_sem_expand_noscore_v1",
    "expand_row_sem_noscore_v1",
)
EDGE_TYPES = frozenset(("lineage", "lineage_up", "similar", "similar_up"))
# A union hop expands several native edge types from the same frontier (UNION
# ALL of per-type ``trimem.expand``; no dedup operator). One string token keeps
# hop_types hashable/encodable exactly like a single-edge hop.
UNION_HOP_TYPES = frozenset(("lineage|similar",))
HOP_TYPES = EDGE_TYPES | UNION_HOP_TYPES
# Which depths a graph plan returns: "final" = endpoints at exactly
# len(hop_types) hops (v1); "all" = endpoints at every depth 1..len(hop_types).
FRONTIERS = ("final", "all")
# How the retained memories are chosen from the stream: "stream" = first top_k
# in stream order (v1); "gain" = read a bounded pool of candidate_visit_cap rows,
# rank by fitness - parent_fitness (desc, id asc), dedup, then pack top_k.
ORDERINGS = ("stream", "gain")
DEFAULT_PREDICATE = (
    "The described change is relevant and reusable for solving the current task."
)


def hop_edges(hop: str) -> tuple[str, ...]:
    """Native edge types expanded by one hop token (``a|b`` is a union hop)."""
    return tuple(hop.split("|"))


def expansion_bound(seed_count: int, fanout: int, hop_types: tuple[str, ...]) -> int:
    """Worst-case emitted edges: seeds * sum_d prod_{hops<=d}(fanout * |edges|)."""
    total, frontier = 0, 1
    for hop in hop_types:
        frontier *= fanout * len(hop_edges(hop))
        total += frontier
    return seed_count * total


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CandidateSpec:
    """An action, not a materialized candidate pool.

    By default (frontier="final") graph plans return endpoints at the explicit
    path depth; frontier="all" returns the union of every depth 1..len(hop_types)
    (no dedup, no global order). Neither claims globally sorted SemRank semantics.
    Inactive fields are absent from identity and restored to defaults on load;
    frontier is identity-active only when it is "all" with two or more hops (with
    one hop "all" == "final"), so pre-v5 actions keep their hashes. Likewise
    ordering is in identity only when "gain", and min_gain only when > 0
    (seed/endpoint rule ``fitness > parent.fitness + min_gain``; 0 is strict >).
    """

    plan_id: str
    seed_count: int = 24
    hop_types: tuple[str, ...] = ("lineage",)
    fanout: int = 8
    top_k: int = 5
    memory_token_cap: int = 8192
    score_admit_min: float = 0.5
    candidate_visit_cap: int = 64
    edge_emit_cap: int = 65536
    filter_predicate: str = DEFAULT_PREDICATE
    score_predicate: str = DEFAULT_PREDICATE
    predicate_version: str = "optimizer-v1"
    unknown_policy: str = "drop"
    schema_version: int = 1
    frontier: str = "final"
    ordering: str = "stream"
    min_gain: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.plan_id not in PLAN_IDS
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("unknown plan or schema version")
        if self.plan_id == "no_memory_v1":
            return
        for key in (
            "seed_count",
            "fanout",
            "top_k",
            "memory_token_cap",
            "candidate_visit_cap",
            "edge_emit_cap",
        ):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 2**31 - 1:
                raise ValueError(f"{key} must be a positive bounded integer")
        if not isinstance(self.hop_types, tuple) or not 1 <= len(self.hop_types) <= 4:
            raise ValueError("hop_types must be a tuple with one to four edges")
        if any(edge not in HOP_TYPES for edge in self.hop_types):
            raise ValueError("unsupported native edge type")
        if self.frontier not in FRONTIERS:
            raise ValueError("frontier must be final or all")
        if self.ordering not in ORDERINGS:
            raise ValueError("ordering must be stream or gain")
        if (
            isinstance(self.min_gain, bool)
            or not isinstance(self.min_gain, (int, float))
            or not math.isfinite(self.min_gain)
            or not 0 <= self.min_gain < 1
        ):
            raise ValueError("min_gain must be finite in [0, 1)")
        if self.memory_token_cap > 8192:
            raise ValueError("memory_token_cap exceeds first-edition limit 8192")
        if self.top_k > self.candidate_visit_cap:
            raise ValueError("top_k exceeds candidate_visit_cap")
        if self.expansion_bound > self.edge_emit_cap:
            raise ValueError("worst-case emitted-edge expansion exceeds edge_emit_cap")
        if self.uses_score:
            if (
                isinstance(self.score_admit_min, bool)
                or not isinstance(self.score_admit_min, (int, float))
                or not math.isfinite(self.score_admit_min)
                or not 0 <= self.score_admit_min <= 1
            ):
                raise ValueError("score_admit_min must be finite in [0, 1]")
            if not self.score_predicate.strip():
                raise ValueError("score predicate must be nonempty")
        if self.filter_target:
            if self.unknown_policy not in ("drop", "keep"):
                raise ValueError("unknown_policy must be drop or keep")
            if not self.filter_predicate.strip():
                raise ValueError("filter predicate must be nonempty")
        if (self.filter_target or self.uses_score) and not self.predicate_version:
            raise ValueError("predicate_version must be nonempty")

    @property
    def total_hops(self) -> int:
        return 0 if self.plan_id == "no_memory_v1" else len(self.hop_types)

    @property
    def expansion_bound(self) -> int:
        if self.plan_id == "no_memory_v1":
            return 0
        return expansion_bound(self.seed_count, self.fanout, self.hop_types)

    @property
    def returns_all_depths(self) -> bool:
        return (
            self.plan_id != "no_memory_v1"
            and self.frontier == "all"
            and len(self.hop_types) > 1
        )

    @property
    def uses_score(self) -> bool:
        return self.plan_id in (
            "vgr_score_v1",
            "seed_sem_expand_v1",
            "expand_row_sem_v1",
        )

    @property
    def filter_target(self) -> str | None:
        return {
            "seed_sem_expand_v1": "seed",
            "expand_row_sem_v1": "row",
            "seed_sem_expand_noscore_v1": "seed",
            "expand_row_sem_noscore_v1": "row",
        }.get(self.plan_id)

    def to_dict(self) -> dict:
        out = {"schema_version": self.schema_version, "plan_id": self.plan_id}
        if self.plan_id == "no_memory_v1":
            return out
        for key in (
            "seed_count",
            "fanout",
            "top_k",
            "memory_token_cap",
            "candidate_visit_cap",
            "edge_emit_cap",
        ):
            out[key] = getattr(self, key)
        out["hop_types"] = list(self.hop_types)
        if self.returns_all_depths:
            out["frontier"] = self.frontier
        if self.ordering != "stream":
            out["ordering"] = self.ordering
        if self.min_gain > 0:
            out["min_gain"] = float(self.min_gain)
        if self.uses_score:
            out.update(
                score_admit_min=float(self.score_admit_min),
                score_predicate=self.score_predicate,
            )
        if self.filter_target:
            out.update(
                filter_predicate=self.filter_predicate,
                unknown_policy=self.unknown_policy,
            )
        if self.filter_target or self.uses_score:
            out["predicate_version"] = self.predicate_version
        return out

    @property
    def action_hash(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict) -> CandidateSpec:
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown action fields: {sorted(unknown)}")
        params = dict(value)
        if "hop_types" in params:
            if not isinstance(params["hop_types"], (tuple, list)):
                raise ValueError("hop_types must be an array")
            params["hop_types"] = tuple(params["hop_types"])
        return cls(**params)
