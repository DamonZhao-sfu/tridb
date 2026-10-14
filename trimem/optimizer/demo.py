"""Synthetic executable wiring example; never an agent/Math/ALE result."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from pathlib import Path

from .controller import JointOptimizer
from .evaluator import (
    CohortEvaluator,
    EvaluationProtocol,
    EvalUnit,
    IsolatedBranch,
    RolloutResult,
)
from .schema import CandidateSpec
from .search import SearchSettings
from .space import SearchSpace


def demo_optimizer(directory: str | Path, *, fixed_plan: str | None = None):
    actions = [CandidateSpec("no_memory_v1")]
    for plan in ("vgr_v1", "vgr_score_v1", "seed_sem_expand_v1", "expand_row_sem_v1"):
        for seeds, hops, k in ((8, 1, 1), (24, 2, 3), (48, 1, 5)):
            actions.append(
                CandidateSpec(
                    plan,
                    seed_count=seeds,
                    hop_types=("lineage",) * hops,
                    top_k=k,
                )
            )
    space = SearchSpace(actions)
    if fixed_plan is not None:
        space = space.subset([fixed_plan])
    protocol = EvaluationProtocol(
        "synthetic-smoke-only-v1",
        "synthetic-model",
        "synthetic-prompt",
        "synthetic-tokenizer",
        20000,
        horizon_kind="fixed_steps",
        max_steps=1,
    )
    units = [
        EvalUnit("synthetic:task", "synthetic-checkpoint", "synthetic-graph", seed, 0.2)
        for seed in (1, 2)
    ]

    @contextmanager
    def branch(unit, action_hash):
        yield IsolatedBranch(
            uuid.uuid4().hex,
            unit.graph_snapshot,
            {},
            "independent in-memory synthetic fixture; no real DB/model",
        )

    def runner(action, context):
        if action.plan_id == "no_memory_v1":
            quality, tokens = 0.25, 500
        else:
            bonus = {
                "vgr_v1": 0.0,
                "vgr_score_v1": 0.07,
                "seed_sem_expand_v1": 0.13,
                "expand_row_sem_v1": 0.11,
            }[action.plan_id]
            quality = 0.35 + bonus + 0.025 * action.top_k + 0.015 * action.total_hops
            tokens = (
                800 + 180 * action.top_k + 12 * action.seed_count + int(bonus * 2000)
            )
        quality += (context.unit.seed - 1.5) * 0.01
        context.ledger.record("synthetic-request", "generation", tokens - 100, 100)
        return RolloutResult(
            quality,
            True,
            steps=1,
            valid_programs=1,
            attempted_programs=1,
            telemetry={"synthetic": True},
        )

    evaluator = CohortEvaluator(protocol, units, runner, branch)
    return JointOptimizer.create(
        directory,
        space,
        SearchSettings(
            0.0,
            20000.0,
            initial_per_plan=1,
            max_fit_iterations=8,
            mc_samples=16,
            pool_size=32,
        ),
        evaluator,
        provenance={"synthetic": True, "no_database_or_model_calls": True},
    )
