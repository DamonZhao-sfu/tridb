"""TRACE v0.7.0 configuration -> the historical session executor (plan 2026-11-05 section 4).

Stock ``plan_execution_adapter`` / ``tools.trace_joint_live.runner_config`` stay byte-identical (live runs hash
them); this module differs in two places only:

* ``plan`` = ``plan_schema_v070.EXECUTOR_PLAN[(filter_order, has_rank)]``: with ``rank_impl = "none"`` the plan is
  the historical VGR / P1 / P2 (no Rank operator; ``agentgym_session_eval.retrieve_v2`` builds no ranker for a plan
  outside ``RANK_PLANS`` and ``session_retrieval.retrieve`` keeps the pool in (-query cosine, id) order before
  ``select_top_k``); otherwise VGR+Rank / P1+Rank / P2+Rank as before.
* ``v2["rank_impl"]`` is the configuration's value (may be ``"none"``) and ``v2["ordering"]`` records the final
  ordering signal (audit only; the executor ignores unknown v2 keys).

``retrieval_unit`` is fixed to ``whole`` in the v0.7.0 live space (the ``actions`` unit's own ColBERT step ranker
therefore never runs).
"""

from __future__ import annotations

from trimem.optimizer.plan_schema import DIRECTION_EDGES
from trimem.optimizer.plan_schema_v070 import JointSpaceV070, executor_plan, ordering

from .session_retrieval import SessionConfigV2


def _canonical(config) -> dict:
    return JointSpaceV070.live().canonicalize(config)


def compile_config(config) -> SessionConfigV2:
    c = _canonical(config)
    return SessionConfigV2(
        plan=executor_plan(c),
        seed_count=c["seed_count"],
        depth=c["depth"],
        fanout=c["fanout"],
        top_k=c["returned_k"],
        edges=DIRECTION_EDGES[c["graph_direction"]],
        sim_gate=None,
        max_failed=None,
        graph_impl=c["graph_impl"],
        beam_width=c["beam_width"],
    )


def runner_config(config) -> dict:
    """``tools.trace_joint_live.runner_config`` for the v0.7.0 space (same keys; plan from EXECUTOR_PLAN)."""
    c = _canonical(config)
    legacy = compile_config(c)
    return {
        "plan": legacy.plan,
        "edges": "+".join(legacy.edges),
        "memory_config": {
            "seed_count": legacy.seed_count,
            "depth": legacy.depth,
            "fanout": legacy.fanout,
            "top_k": legacy.top_k,
        },
        "v2": {
            "sim_gate": None,
            "max_failed": None,
            "memory_form": "raw",
            "memory_forms_version": "memory-forms-v1",
            "outcome_policy": "success_only" if c["trajectory_category"] == "Success" else "success_failure",
            "semfilter_threshold": {
                "none": None,
                "before": c["sem_threshold_pre"],
                "after": c["sem_threshold_post"],
            }[c["filter_order"]],
            "pool_cap": 256,
            "expand_limit": 4096,
            "graph_impl": c["graph_impl"],
            "beam_width": c["beam_width"],
            "rank_impl": c["rank_impl"],
            "retrieval_unit": c["retrieval_unit"],
            "injection_timing": c["injection_timing"],
            "retrieval_interval": c["retrieval_interval"],
            "ordering": ordering(c),
        },
    }
