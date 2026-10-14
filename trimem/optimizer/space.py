"""Finite, conditional *metadata* space for trajectory retrieval policies.

This module never retrieves rows. Different plans are allowed to change context;
the downstream evaluator determines whether those changes help the agent.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

from .schema import PLAN_IDS, CandidateSpec, expansion_bound


@dataclass(frozen=True)
class SpaceGrid:
    """Explicit grids, frozen for a search run; these are not tuned optima."""

    seed_counts: tuple[int, ...] = (8, 24, 48)
    hop_paths: tuple[tuple[str, ...], ...] = (
        ("lineage",),
        ("lineage", "lineage"),
        ("lineage",) * 4,
        ("similar", "lineage"),
    )
    fanouts: tuple[int, ...] = (4, 8, 16)
    top_ks: tuple[int, ...] = (1, 3, 5, 8)
    unknown_policies: tuple[str, ...] = ("drop", "keep")
    memory_token_cap: int = 8192
    candidate_visit_cap: int = 64
    # Depths returned by graph plans (schema.FRONTIERS). "all" on a one-hop
    # path is identical to "final" and deduplicates by action hash.
    frontiers: tuple[str, ...] = ("final",)
    # Optional visit-cap grid; None keeps the single fixed cap above (and thus
    # the exact pre-v5 candidate set/hashes for the default grid).
    candidate_visit_caps: tuple[int, ...] | None = None
    # Fixed (not searched) result ordering and improving-edit epsilon; the
    # defaults ("stream", 0.0) are absent from identity.
    ordering: str = "stream"
    min_gain: float = 0.0
    # Conservative bound on emitted graph edges, not inspected adjacency slots.
    edge_emit_cap: int = 65536
    # Admission controls for streaming score; selectivity is measured, not set.
    score_admit_mins: tuple[float, ...] = (0.3, 0.5, 0.7)
    # Backwards-compatible explicit fixed-threshold override of the tuple grid.
    score_admit_min: float | None = None


# v5 "medium" grid (RAG-Stack-style declared space). Fixed: memory cap 8192,
# unknown_policy drop. Restrict plans with ``allowed_plan_ids`` (V5_PLAN_IDS).
MEDIUM_GRID = SpaceGrid(
    seed_counts=(8, 16, 24, 48),
    # msem-style traversal: every hop expands lineage AND similar.
    hop_paths=tuple(("lineage|similar",) * depth for depth in (1, 2, 3)),
    frontiers=("final", "all"),
    fanouts=(4, 8),
    top_ks=(3, 5, 8),
    unknown_policies=("drop",),
    memory_token_cap=8192,
    # With gain ranking the cap is the size of the ranked pool, a real knob.
    candidate_visit_caps=(64, 256, 1024),
    # msem-style: rank the bounded pool by fitness gain; ignore float-noise
    # "improvements" (seen at +4.4e-16) as seeds and endpoints.
    ordering="gain",
    min_gain=1e-9,
    edge_emit_cap=65536,
)
V5_PLAN_IDS = (
    "no_memory_v1",
    "vgr_v1",
    "seed_sem_expand_noscore_v1",
    "expand_row_sem_noscore_v1",
)


class SearchSpace:
    """Canonical candidates with a deterministic conditional feature encoding.

    Numeric controls have grid-frozen min/max normalization. Categorical controls
    use integer labels consumed by a categorical GP kernel, never ordinal distance.
    Each control also has an active mask. Values for absent controls have no policy
    meaning and cannot create duplicate actions.
    """

    _numeric_fields = (
        "seed_count",
        "fanout",
        "top_k",
        "memory_token_cap",
        "score_admit_min",
        "candidate_visit_cap",
        "edge_emit_cap",
    )
    _categorical_fields = (
        "plan_id",
        "hop_types",
        "filter_predicate",
        "score_predicate",
        "predicate_version",
        "unknown_policy",
    )
    # v5 graph-plan fields whose default is absent from identity. They are
    # encoded (defaults filled) only when they vary across the space, so the
    # feature vectors of spaces where they are constant are unchanged.
    _graph_defaults = MappingProxyType(
        {"frontier": "final", "ordering": "stream", "min_gain": 0.0}
    )

    def __init__(self, candidates: Iterable[CandidateSpec]) -> None:
        unique = {action.action_hash: action for action in candidates}
        if not unique:
            raise ValueError("search space must contain at least one legal action")
        self.candidates = tuple(
            sorted(
                unique.values(), key=lambda action: (action.plan_id, action.action_hash)
            )
        )
        self._by_hash = {action.action_hash: action for action in self.candidates}
        self.plan_ids = tuple(
            plan_id
            for plan_id in PLAN_IDS
            if any(action.plan_id == plan_id for action in self.candidates)
        )
        if set(self.plan_ids) != {action.plan_id for action in self.candidates}:
            raise ValueError("search space contains an unregistered plan")
        rows = [self._row(action) for action in self.candidates]
        varying = {
            key
            for key in self._graph_defaults
            if len({row[key] for row in rows if key in row}) > 1
        }
        self._numeric_fields = self._numeric_fields + tuple(
            key for key in ("min_gain",) if key in varying
        )
        self._categorical_fields = self._categorical_fields + tuple(
            key for key in ("frontier", "ordering") if key in varying
        )
        self.feature_names: tuple[str, ...] = ()
        names: list[str] = []
        cat_dims: list[int] = []
        bounds: dict[str, tuple[float, float]] = {}
        categories: dict[str, tuple[str, ...]] = {}
        for key in self._numeric_fields:
            values = [float(row[key]) for row in rows if key in row]
            bounds[key] = (min(values), max(values)) if values else (0.0, 0.0)
            names.extend((key, f"{key}__active"))
        for key in self._categorical_fields:
            values = tuple(sorted({self._category(row, key) for row in rows}))
            categories[key] = values
            if len(values) > 1:
                cat_dims.append(len(names))
            names.extend((key, f"{key}__active"))
        self.feature_names = tuple(names)
        self.categorical_dims = tuple(cat_dims)
        self._bounds = bounds
        self._categories = categories

    @classmethod
    def _row(cls, action: CandidateSpec) -> dict:
        """Identity dict with absent v5 graph defaults filled for graph plans."""
        row = action.to_dict()
        if "hop_types" in row:
            row = {**cls._graph_defaults, **row}
        return row

    @staticmethod
    def _category(row: dict, key: str) -> str:
        if key not in row:
            return "<inactive>"
        value = row[key]
        if isinstance(value, (tuple, list)):
            return "/".join(value)
        return str(value)

    def __len__(self) -> int:
        return len(self.candidates)

    def contains(self, action: CandidateSpec) -> bool:
        return action.action_hash in self._by_hash

    def subset(self, allowed_plan_ids: Iterable[str]) -> SearchSpace:
        allowed = frozenset(allowed_plan_ids)
        if not allowed or not allowed.issubset(self.plan_ids):
            raise ValueError("allowed_plan_ids must be a nonempty subset of this space")
        return SearchSpace(a for a in self.candidates if a.plan_id in allowed)

    def encode(self, action: CandidateSpec) -> tuple[float, ...]:
        if not self.contains(action):
            raise ValueError("cannot encode an action outside this frozen search space")
        row = self._row(action)
        out: list[float] = []
        for key in self._numeric_fields:
            low, high = self._bounds[key]
            active = key in row
            value = (
                (float(row[key]) - low) / (high - low) if active and high > low else 0.0
            )
            out.extend((value, float(active)))
        for key in self._categorical_fields:
            value = self._category(row, key)
            active = key in row
            out.extend((float(self._categories[key].index(value)), float(active)))
        return tuple(out)


def make_space(
    grid: SpaceGrid | None = None,
    allowed_plan_ids: Iterable[str] | None = None,
) -> SearchSpace:
    """Enumerate legal templates/configurations, with no execution side effects."""
    grid = grid or SpaceGrid()
    allowed = frozenset(PLAN_IDS if allowed_plan_ids is None else allowed_plan_ids)
    if not allowed or not allowed.issubset(PLAN_IDS):
        raise ValueError("allowed_plan_ids contains no plans or an unknown plan")
    actions = []
    for plan_id in PLAN_IDS:
        if plan_id not in allowed:
            continue
        if plan_id == "no_memory_v1":
            actions.append(CandidateSpec(plan_id=plan_id))
            continue
        policies = (
            grid.unknown_policies
            if CandidateSpec(plan_id=plan_id).filter_target
            else ("drop",)
        )
        thresholds = (
            (
                (grid.score_admit_min,)
                if grid.score_admit_min is not None
                else grid.score_admit_mins
            )
            if CandidateSpec(plan_id=plan_id).uses_score
            else (0.5,)
        )
        visit_caps = (
            grid.candidate_visit_caps
            if grid.candidate_visit_caps is not None
            else (grid.candidate_visit_cap,)
        )
        for (
            seeds,
            path,
            fanout,
            top_k,
            visit_cap,
            frontier,
            unknown,
            threshold,
        ) in itertools.product(
            grid.seed_counts,
            grid.hop_paths,
            grid.fanouts,
            grid.top_ks,
            visit_caps,
            grid.frontiers,
            policies,
            thresholds,
        ):
            if (
                expansion_bound(seeds, fanout, path) > grid.edge_emit_cap
                or top_k > visit_cap
                or (frontier != "final" and len(path) < 2)
            ):
                continue
            actions.append(
                CandidateSpec(
                    plan_id=plan_id,
                    seed_count=seeds,
                    hop_types=path,
                    fanout=fanout,
                    top_k=top_k,
                    frontier=frontier,
                    memory_token_cap=grid.memory_token_cap,
                    candidate_visit_cap=visit_cap,
                    ordering=grid.ordering,
                    min_gain=grid.min_gain,
                    edge_emit_cap=grid.edge_emit_cap,
                    score_admit_min=threshold,
                    unknown_policy=unknown,
                )
            )
    missing_plans = allowed - {action.plan_id for action in actions}
    if missing_plans:
        raise ValueError(
            "grid/budgets leave no legal configurations for requested plans: "
            + ", ".join(sorted(missing_plans))
        )
    return SearchSpace(actions)
