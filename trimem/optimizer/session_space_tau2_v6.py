"""τ²-bench session space of the isolated re-profile ``pool-v6`` (user decision 2026-10-23;
``trimem.experiments.tau2.design`` selects it with ``TAU2_POOL_DESIGN=explore-v6``; the -v5 run and every other caller
keep ``session_space_tau2_v2`` = ``session-space-tau2-v5``).

Identical to ``session_space_tau2_v2`` (-v5: agent-env v2 knob grids / canonicalization / GP encoding, memory_form
FIXED ``compact``, edges similar+similar_up, sim_gate a GP feature with the proposal-side constraint sim_gate = none)
EXCEPT the plans:

* SIX plans, all proposable from round 0 (no plan extension, no gated plan): VGR, VGR+Rank, P2 (VGR+SemFilter),
  P2+Rank (VGR+SemFilter+Rank), P1+Rank (V+SemFilter+G+Rank) and P1 (V+SemFilter+G = SemFilter on the vector seeds
  before graph expansion, no Rank; as agent-env plan-p1-v1). ``SEMFILTER_PLANS`` includes P1, so
  ``tau2.runner_pool`` builds the SemFilter judge for it (semfilter-v2 with the recorded knobs); the retrieval itself
  is ``session_retrieval`` plan P1 (= P1+Rank without the ColBERT rank), as executed for ALFWorld / ScienceWorld.
* GP plan category = agent-env v2 ``PLANS`` index (VGR 0, VGR+Rank 1, P2 2, P2+Rank 3, P1+Rank 4, P1 5; NoMemory 6),
  i.e. the same encoding as -v5 for the five shared plans.

``proposal_view`` / ``PROPOSE_FIXED`` / ``PROPOSAL_PROTOCOL`` as -v5 (every configuration of this run, round 0
included, has sim_gate none).
"""

import itertools
import json

from trimem.optimizer import session_space_agentenv_v2 as v2
from trimem.optimizer import session_space_tau2_v2 as v5

VERSION = "session-space-tau2-v6"  # = -v5 + plan P1, all six plans proposable from round 0 (pool-v6)
BASE_VERSION = v2.VERSION
PLANS = (*v2.BASE_PLANS, "P1+Rank", "P1")
BASE_PLANS = PLANS  # proposable from round 0
ADDED_PLANS: tuple[
    str, ...
] = ()  # no plan extension (session_pool: no dls-newplan starts)
PLAN_EXTENSION = {
    "protocol": "tau2-plans-v6-all-from-round0",
    "from_round": 0,
    "added_plans": [],
}
SEMFILTER_PLANS = ("P2", "P2+Rank", "P1+Rank", "P1")
EDGES = v5.EDGES
POOL_CAP = v5.POOL_CAP
EXPAND_LIMIT = v5.EXPAND_LIMIT
SIM_GATES = v5.SIM_GATES
GATE = dict(v5.GATE)
PROPOSE_FIXED = dict(v5.PROPOSE_FIXED)
PROPOSAL_PROTOCOL = v5.PROPOSAL_PROTOCOL
MEMORY_FORMS = v5.MEMORY_FORMS
MEMORY_FORM = v5.MEMORY_FORM
TAU2_MEMORY_FORMS = v5.TAU2_MEMORY_FORMS
UNSUPPORTED: dict = {}
OUTCOME_POLICIES = v5.OUTCOME_POLICIES
SEMFILTER_THRESHOLDS = v5.SEMFILTER_THRESHOLDS
FIXED = dict(v5.FIXED)
NUMERIC = v5.NUMERIC
CAT_DIMS = v5.CAT_DIMS
NOMEM = v5.NOMEM
DISPLAY = v2.DISPLAY

plan_defaults = v2.plan_defaults
orderings = v2.orderings
fallbacks = v2.fallbacks
canonical_memory = v2.canonical_memory
horizon = v2.horizon
max_failed = v2.max_failed

if any(p not in v2.PLANS for p in PLANS) or any(
    p not in v2.SEMFILTER_PLANS for p in SEMFILTER_PLANS
):
    raise ImportError("session_space_tau2_v6: plan outside the agent-env v2 space")


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
    """v2 validation with the τ² fixed form; plans outside ``PLANS`` raise."""
    out = v2.validate(cfg, MEMORY_FORM, **GATE)
    if out["plan"] != "NoMemory" and out["plan"] not in PLANS:
        raise ValueError("Unknown session plan")
    return out


def key(cfg):
    return json.dumps(validate(cfg), sort_keys=True, separators=(",", ":"))


def features(cfg, task=None, start_norm=0, iter_start=0, h=1):
    return v2.features(
        validate(cfg), task, start_norm, iter_start, h, MEMORY_FORM, **GATE
    )


def features_unchecked(cfg):
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
    return v2.runner_args(validate(cfg), MEMORY_FORM, **GATE)


def proposal_plans(r: int, extension: dict | None = None) -> tuple[str, ...]:
    """Every plan from round 0 (``extension`` must be this space's PLAN_EXTENSION or None)."""
    ext = PLAN_EXTENSION if extension is None else extension
    if ext.get("added_plans"):
        raise ValueError(f"{VERSION} has no plan extension: {ext}")
    return BASE_PLANS


def proposal_ok(cfg) -> bool:
    cfg = validate(cfg)
    return cfg["plan"] == "NoMemory" or all(
        cfg[k] == v for k, v in PROPOSE_FIXED.items()
    )


class _ProposalView:
    """This space as the LLM analyst sees it: ``PROPOSE_FIXED`` knobs FIXED (as -v5)."""

    FIXED = {**FIXED, **PROPOSE_FIXED}
    PROPOSE_FIXED = PROPOSE_FIXED

    def __getattr__(self, name):
        return globals()[name]

    @staticmethod
    def domain(plan):
        return {k: v for k, v in domain(plan).items() if k not in PROPOSE_FIXED}


def proposal_view():
    return _ProposalView()
