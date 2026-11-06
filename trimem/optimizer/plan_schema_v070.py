"""TRACE configuration space v0.7.0 (user 2026-11-05).

Additive to ``plan_schema`` (which stays byte-identical: live runs hash it):

* ``rank_impl`` gains ``"none"`` (appended, so earlier categorical codes are unchanged): no Rank operator; the
  (filtered) pool is ordered by query cosine (task-vector cosine to the query, ties by session id) and the top
  ``returned_k`` are injected -- the historical VGR / P1 / P2 plans of ``agentgym_session_eval``.
* ``rank_impl = "LLM"`` is removed from the live domain.
* ``graph_direction`` fixed to ``Both`` and ``retrieval_unit`` fixed to ``whole`` (user 2026-11-05).
* NoMemory is a fixed protected baseline handled outside the configuration space (not a value here).

The stock ``JointSpace`` refuses these domains (``rank_impl`` "none" is not in its FIELDS), so a stock launcher
pointed at a v0.7.0 run fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping

from trimem.optimizer import plan_schema as base

SCHEMA_VERSION = "trace-pr-v070-rank-optional-v1"
RANK_NONE = "none"
FIELDS = tuple((n, k, v + (RANK_NONE,)) if n == "rank_impl" else (n, k, v) for n, k, v in base.FIELDS)
FIELD_NAMES = base.FIELD_NAMES
LIVE_DOMAINS = {
    **base.LIVE_DOMAINS,
    "rank_impl": ("ColBERT", "cross-encoder", RANK_NONE),
    "retrieval_unit": ("whole",),
    "graph_direction": ("Both",),
}
SEMANTICS = {
    "rank_impl": {
        RANK_NONE: "no Rank operator: the (filtered) pool is ordered by query cosine (task-vector cosine to the "
        "query, ties by session id) and the top returned_k are injected (historical VGR / P1 / P2)"
    },
    "baselines": "NoMemory is a fixed protected baseline, not a configuration of this space",
}
EXECUTOR_PLAN = {
    ("none", False): "VGR",
    ("before", False): "P1",
    ("after", False): "P2",
    ("none", True): "VGR+Rank",
    ("before", True): "P1+Rank",
    ("after", True): "P2+Rank",
}
ORDERING = {RANK_NONE: "query_cosine", "ColBERT": "colbert_score", "cross-encoder": "cross_encoder_score"}


class JointSpaceV070(base.JointSpace):
    """``JointSpace`` validated against the v0.7.0 FIELDS (``rank_impl`` may be ``"none"``)."""

    def __init__(self, capabilities: Mapping | None = None, *, execution_fingerprint: str = "representation-only"):
        caps = dict(capabilities or {})
        if set(caps) - set(FIELD_NAMES):
            raise ValueError("Unknown capability fields")
        self.domains = {}
        for name, _, full in FIELDS:
            requested = tuple(caps.get(name, full))
            if not requested or len(set(requested)) != len(requested):
                raise ValueError(f"Empty or repeated capability for {name}")
            if any(v not in full for v in requested):
                raise ValueError(f"Invalid capability for {name}")
            self.domains[name] = tuple(v for v in full if v in requested)
        self.execution_fingerprint = execution_fingerprint
        self.identity = base.digest(self.manifest())

    @classmethod
    def live(cls, *, execution_fingerprint: str = "legacy-task-ann-v070"):
        return cls(LIVE_DOMAINS, execution_fingerprint=execution_fingerprint)

    def manifest(self) -> dict:
        return {**super().manifest(), "schema": SCHEMA_VERSION, "semantics": SEMANTICS}

    def plan(self, config: Mapping) -> base.PlanSpec:
        spec = super().plan(config)
        c = self.canonicalize(config)
        if c["rank_impl"] != RANK_NONE:
            return spec
        nodes = list(spec.nodes)
        last = nodes[-1]
        if last.role != "Rank":
            raise AssertionError("v0.7.0 plan must end with the Rank slot")
        nodes[-1] = base.PlanNode("TopK", "query_cosine", last.inputs,
                                  {"returned_k": c["returned_k"], "order": "query_cosine"})
        return base.PlanSpec(tuple(nodes), spec.injection)


def has_rank(config: Mapping) -> bool:
    return config["rank_impl"] != RANK_NONE


def executor_plan(config: Mapping) -> str:
    return EXECUTOR_PLAN[(config["filter_order"], has_rank(config))]


def ordering(config: Mapping) -> str:
    return ORDERING[config["rank_impl"]]
