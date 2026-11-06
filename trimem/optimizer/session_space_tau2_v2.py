"""τ²-bench session space for the pool-profiling round (user decision 2026-10-19): the subset of
``session_space_agentenv_v2`` that the τ² episode runner (``trimem.experiments.tau2.runner_pool``) executes.

Identical to agent-env v2 (same plans, knob grids, canonicalization, protocol constants, GP encoding and
configuration keys) EXCEPT the fixed memory_form: -v5 (user decision 2026-10-20) memory_form is not searched;
τ² fixes it to ``compact`` (agent-env ALFWorld / ScienceWorld: ``raw``). Reason: on τ² compact cuts the
injected memory ~80% (6,365 -> 1,261 Qwen tokens for the VGR default). The shared router code
(``query_router.sparse``) validates with this space after ``sparse.use_space`` (``pool_profile`` does it).

-v4 (user 2026-10-20): memory_form = {raw, compact, distilled} as in agent-env -v4; τ² ``distilled`` = the
offline customer-service distiller (``memory_distill.TAU2_SYSTEM``, once per stored τ² session at ingest).
-v3 restricted memory_form to {raw, success_only_raw} (no τ² distiller then).

sim_gate (user decision 2026-10-20): the agent-env space -v6 FIXES sim_gate = none; this τ² -v5 space keeps it a
SEARCHED dimension (``gate_searched=True`` everywhere: grid, validation, keys, GP feature), so the gated round-0
configurations of the live run stay valid and modelled. Every configuration PROPOSED from round 1 on has
sim_gate = none: ``PROPOSE_FIXED`` is a proposal-side constraint honoured by ``session_pool.suggest`` (Sobol,
crossover, DLS neighbours, forced gap, agent rows) and by the LLM analyst (``proposal_view``: sim_gate shown as
FIXED = none, candidates naming it rejected); protocol ``PROPOSAL_PROTOCOL``, recorded in state.json.

Plan extension ``plan-p1rank-v1`` (user decision 2026-10-20; ``PLAN_EXTENSION``): the plan ``P1+Rank`` (display
``V+SemFilter+G+Rank``: SemFilter on the vector seeds BEFORE graph expansion, then expansion, pool cap, ColBERT
rank, top-k) is a level of this space (validation, keys, GP encoding: category index 4, appended) WITHOUT a version
bump: the live run keeps its experiment id. It is PROPOSABLE only from BO round ``from_round`` = 7
(``proposal_plans``); rounds 0-6 proposed over ``BASE_PLANS`` (the 4 -v6 plans, unchanged). The live run records
the extension once in state.json ``space_extension`` (``pool_profile.extension_block``). Knobs and defaults as
P2+Rank (semfilter_threshold active; ordering colbert_score).

Everything else of v2 is executed by the τ² runner: edges fixed ``similar+similar_up`` (user correction
2026-10-20; the earlier ``session-space-tau2-v2`` fixed ``similar_up`` only and is void); seed_count / depth / fanout /
top_k grids; sim_gate; outcome_policy (success_only -> memory policy success-only-v0, unlabelled cards;
success_failure[_capped] -> success+failure-v1, ``max_failed`` = floor(top_k / 2) for capped); SemFilter
threshold (P2 / P2+Rank); candidate pool cap 256 / expansion guard 4,096. Note the τ² store is small
(train: airline 30 / retail 74 / telecom 74 tasks, 4 sessions each), so seed_count >= 32 saturates airline
(every admissible airline task is a seed) and the 256-session pool cap can bind only for retail / telecom.
"""

import itertools
import json

from trimem.optimizer import session_space_agentenv_v2 as v2

VERSION = "session-space-tau2-v5"  # v2 = edges similar_up (void); v3 = memory_form {raw, success_only_raw};
# v4 = memory_form searched {raw, compact, distilled}; v5 = memory_form fixed compact (user 2026-10-20)
BASE_VERSION = v2.VERSION
# pinned (2026-10-22): the agent-env space added the gated plan P1 (plan-p1-v1); τ² keeps its five plans
PLANS = (
    *v2.BASE_PLANS,
    "P1+Rank",
)  # every level of the space (incl. P1+Rank, plan-p1rank-v1)
BASE_PLANS = v2.BASE_PLANS  # proposable in rounds < PLAN_EXTENSION["from_round"]
ADDED_PLANS = (
    "P1+Rank",
)  # session_pool: DLS start at the plan default while unmeasured
PLAN_EXTENSION = {
    "protocol": v2.PLAN_EXTENSION_PROTOCOL,
    "from_round": 7,  # the τ² -v5 live run's next round at the decision (user 2026-10-20)
    "added_plans": list(ADDED_PLANS),
}
SEMFILTER_PLANS = ("P2", "P2+Rank", "P1+Rank")
EDGES = v2.EDGES
POOL_CAP = v2.POOL_CAP
EXPAND_LIMIT = v2.EXPAND_LIMIT
SIM_GATES = v2.SIM_GATES_SEARCHED  # searched in this space (existing round-0 rows)
GATE = {"gate_searched": True}
# proposal-side constraint (user 2026-10-20): every configuration proposed from round 1 on has no gate
PROPOSE_FIXED = {"sim_gate": "none"}
PROPOSAL_PROTOCOL = "tau2-propose-simgate-none-v1"
MEMORY_FORMS = v2.MEMORY_FORMS
MEMORY_FORM = v2.FIXED_MEMORY_FORM["tau2"]  # "compact" (-v5: fixed, not searched)
TAU2_MEMORY_FORMS = (MEMORY_FORM,)
UNSUPPORTED: dict = {}
OUTCOME_POLICIES = v2.OUTCOME_POLICIES
SEMFILTER_THRESHOLDS = v2.SEMFILTER_THRESHOLDS
FIXED = {
    k: v for k, v in {**v2.FIXED, "memory_form": MEMORY_FORM}.items() if k != "sim_gate"
}
NUMERIC = v2.NUMERIC
CAT_DIMS = v2.CAT_DIMS
NOMEM = v2.NOMEM
DISPLAY = v2.DISPLAY

plan_defaults = v2.plan_defaults
orderings = v2.orderings
fallbacks = v2.fallbacks
canonical_memory = v2.canonical_memory
horizon = v2.horizon
max_failed = v2.max_failed


def grid():
    return v2.grid(**GATE)


def v7_defaults(plan):
    return v2.v7_defaults(plan, MEMORY_FORM, **GATE)


def complete(cfg):
    return v2.complete(cfg, MEMORY_FORM, **GATE)


def domain(plan):
    if plan == "NoMemory":
        return {}
    if plan not in PLANS:
        raise ValueError("Unknown session plan")
    g = grid()
    if plan not in SEMFILTER_PLANS:
        g.pop("semfilter_threshold")
    return g


def validate(cfg):
    """v2 validation with the τ² fixed form: any other memory_form raises (``Unsupported session capability``)."""
    return v2.validate(cfg, MEMORY_FORM, **GATE)


def key(cfg):
    return json.dumps(validate(cfg), sort_keys=True, separators=(",", ":"))


def features(cfg, task=None, start_norm=0, iter_start=0, h=1):
    return v2.features(
        validate(cfg), task, start_norm, iter_start, h, MEMORY_FORM, **GATE
    )


def features_unchecked(cfg):
    """Auxiliary-observation encoding (``session_space_agentenv_v2.features_unchecked``; off-grid levels allowed)."""
    return v2.features_unchecked(cfg, MEMORY_FORM, **GATE)


def space():
    """Every distinct (canonical) configuration, NoMemory first."""
    yield dict(NOMEM)
    seen = set()
    for plan in PLANS:
        d = domain(plan)
        for values in itertools.product(*d.values()):
            cfg = validate({"plan": plan, **dict(zip(d, values, strict=True))})
            k = key(cfg)
            if k not in seen:
                seen.add(k)
                yield cfg


def size() -> dict:
    per = {p: sum(1 for c in space() if c["plan"] == p) for p in PLANS}
    return {**per, "total": sum(per.values())}


def runner_args(cfg) -> dict:
    """Runner form (``plan``, ``edges``, ``memory_config``, ``v2`` options) = v2's for a τ² configuration."""
    return v2.runner_args(validate(cfg), MEMORY_FORM, **GATE)


def proposal_plans(r: int, extension: dict | None = None) -> tuple[str, ...]:
    """Plans proposable in BO round ``r``: ``BASE_PLANS``, plus the extension's added plans from its ``from_round``
    (``extension`` = the block recorded in state.json; default ``PLAN_EXTENSION``)."""
    ext = PLAN_EXTENSION if extension is None else extension
    added = tuple(ext["added_plans"]) if r >= ext["from_round"] else ()
    if any(p not in PLANS for p in added):
        raise ValueError(f"extension plans {added} outside the space")
    return (*BASE_PLANS, *added)


def proposal_ok(cfg) -> bool:
    """True when a configuration satisfies the proposal-side constraint (NoMemory always)."""
    cfg = validate(cfg)
    return cfg["plan"] == "NoMemory" or all(
        cfg[k] == v for k, v in PROPOSE_FIXED.items()
    )


class _ProposalView:
    """This space as the LLM analyst must see it from round 1 on: ``PROPOSE_FIXED`` knobs are FIXED (shown as such,
    not searchable: a candidate naming one is rejected) and absent from ``domain``; everything else delegates."""

    FIXED = {**FIXED, **PROPOSE_FIXED}
    PROPOSE_FIXED = PROPOSE_FIXED

    def __getattr__(self, name):
        return globals()[name]

    @staticmethod
    def domain(plan):
        return {k: v for k, v in domain(plan).items() if k not in PROPOSE_FIXED}


def proposal_view():
    return _ProposalView()
