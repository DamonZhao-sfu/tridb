"""ALFWorld / ScienceWorld session space v2 (pool profiling, user decisions 2026-10-19).

Same interface as ``session_space_agentenv`` (v1, unchanged and reproducible). Changes:

* ``edges`` FIXED to ``similar+similar_up`` (both session-kNN directions: every hop expands each frontier
  session over ``similar`` AND ``similar_up``, ``fanout`` per edge type; recorded in every configuration, not
  searched). User correction 2026-10-20: the first v2 (``session-space-agentenv-v2``) fixed ``similar_up`` only;
  that space and its episodes are void, the version string is bumped to ``-v3`` so no row of it can mix in.
  -v4 (user 2026-10-20): memory_form = {raw, compact, distilled} (``trimem.experiments.memory_forms``), see below.
  -v5 (user decision 2026-10-20): memory_form is NOT searched any more; it is FIXED per dataset
  (``FIXED_MEMORY_FORM``: ALFWorld / ScienceWorld ``raw``, τ² ``compact`` in ``session_space_tau2_v2``) and
  recorded in every configuration like ``edges``. Reasons (docs/trimem_pool_profiling_plan_v0.1.0.md addendum
  2026-10-20): on ALF/Sci compact ~= raw in injected tokens and distilled is longer; on τ² compact cuts the
  injected memory ~80%. -v4 rows / episodes never mix in (new version string -> new experiment ids / dirs).
  -v6 (user decision 2026-10-20, before ALFWorld / ScienceWorld started): ``sim_gate`` is FIXED to ``none`` (no
  gate), round 0 included -- recorded in every configuration, enforced by ``v7_defaults``, not searched (not in
  ``grid()`` / ``domain()``; the GP feature stays, constant 0 = "no gate"). τ² keeps its -v5 space, where
  sim_gate is a searched dimension of the existing round-0 configurations (``session_space_tau2_v2`` passes
  ``gate_searched=True``) and is fixed only on the PROPOSAL side from round 1 (``PROPOSE_FIXED``).
  -v7 (user decision 2026-10-20, protocol ``plan-p1rank-v1``, before ALFWorld / ScienceWorld started): a fifth plan
  ``P1+Rank`` (display ``V+SemFilter+G+Rank``): SemFilter on the vector SEED sessions BEFORE graph expansion
  (descendants are not re-filtered, as for P1), then expansion, relational isolation / pool cap, ColBERT rank,
  top-k (``session_retrieval.retrieve``). Same knobs as P2+Rank (seed_count, depth, fanout, top_k, outcome_policy,
  semfilter_threshold; ordering colbert_score), same plan defaults; sim_gate / edges / memory_form fixed as in -v6.
  Round 0 = the 5 plan defaults + 5 Sobol (10 configurations, as before). The plan category is appended (GP
  index 4; NoMemory moves to 5 -- categorical, so only equality matters). τ² keeps its -v5 space identity and gets
  the plan from BO round 7 (``session_space_tau2_v2.PLAN_EXTENSION``); ``BASE_PLANS`` = the -v6 plans.
* plan-p1-v1 (user decision 2026-10-22): a sixth plan ``P1`` (display ``V+SemFilter+G``: SemFilter on the vector
  SEED sessions, then graph expansion, relational isolation / pool cap, top-k by query cosine; no Rank) is appended
  to ``ADDED_PLANS`` (GP category index 5; every earlier plan keeps its index; NoMemory moves to 6 -- categorical,
  only equality matters). It is GATED (``GATED_PLANS``): never proposable unless the run records the extension in
  state.json ``plan_extension_p1`` (``session_bo_agentenv.plan_p1_block``, from its ``from_round``); without it the
  proposal plans are exactly the -v7 five, so earlier runs / replays are unchanged. VERSION (the run identity) is
  unchanged. τ² (``session_space_tau2_v2``) does not get P1.
* wider knobs: seed_count {4, 8, 16, 32, 64, 128}, depth {2, 3, 4}, fanout {4, 8, 16, 32, 64},
  top_k {3, 5, 8, 10}.
* plans (-v7): VGR, VGR+Rank, P2 (VGR+SemFilter), P2+Rank (VGR+SemFilter+Rank), P1+Rank (V+SemFilter+G+Rank).
* new dimensions (searched unless noted):
  - ``sim_gate`` {none, 0.5, 0.6, 0.7} (-v6: FIXED none here; searched only in the τ² -v5 space): inject nothing
    when the best pool session's query cosine is below it;
  - ``memory_form`` (-v5: FIXED per dataset, not searched; vocabulary {raw, compact, distilled},
    ``memory_forms``, protocol ``memory-forms-v1``): raw = the
    agent-env card; compact = rule-based card (goal, collapsed action sequence, outcome + label, <= 600 Qwen
    tokens); distilled = the session's OFFLINE LLM card (``memory_distill``, once per stored session at ingest,
    never per episode). (-v3 had {raw, distilled = query-time SemExtract, success_only_raw}; success_only_raw
    is raw x success_only, query-time SemExtract is no longer a level.)
  - ``outcome_policy`` {success_only, success_failure, success_failure_capped}: which outcomes are
    retrievable (success_only = memory policy success-only-v0, unlabelled cards; success_failure =
    success+failure-v1, labelled; _capped = the same with at most floor(top_k / 2) FAILED cards in top-k);
  - ``semfilter_threshold`` {0.4, 0.6, 0.8}: only for SemFilter plans (P2 / P2+Rank / P1+Rank).
* protocol constants (not searched): candidate pool cap ``POOL_CAP`` = 256 sessions (truncated by query
  cosine; SemFilter judges at most the capped pool), expansion guard ``EXPAND_LIMIT`` = 4,096.

Conditional / redundant combinations are canonicalized (``complete``), so each distinct behaviour has
exactly one key and BO never spends an evaluation on a duplicate:
  - memory_form is fixed (-v5), so the searched memory dimension is outcome_policy alone (3 levels);
  - semfilter_threshold is inactive for VGR / VGR+Rank: fixed to its default 0.6 (not in their domain).
GP encoding (``features``): numeric knobs normalized to [0, 1] (sim_gate none -> 0.0, i.e. "no gate";
inactive semfilter_threshold -> its constant default); plan and outcome_policy categorical (``CAT_DIMS``; the
fixed memory_form is not a feature since -v5). Hooks: ``FIXED`` lists dimensions held at a constant (edges,
memory_form); they are enforced by ``v7_defaults`` (``validate`` rejects any other value).
"""

import itertools
import json

VERSION = "session-space-agentenv-v7"  # v2 = edges similar_up (void); v3 = old memory_form levels;
# v4 = memory_form searched {raw, compact, distilled}; v5 = memory_form fixed per dataset (user 2026-10-20);
# v6 = sim_gate fixed none (user 2026-10-20); v7 = + plan P1+Rank (plan-p1rank-v1, user 2026-10-20)
BASE_PLANS = ("VGR", "VGR+Rank", "P2", "P2+Rank")  # the -v6 plans (τ² -v5 rounds 0-6)
PLAN_EXTENSION_PROTOCOL = "plan-p1rank-v1"
ADDED_PLANS = (
    "P1+Rank",
    "P1",  # plan-p1-v1 (user 2026-10-22), gated: GATED_PLANS
)  # appended: the GP category index of every earlier plan is unchanged
# plans proposable only from a recorded extension (state.json; session_pool.use_space leaves them out by default)
PLAN_P1_PROTOCOL = "plan-p1-v1"
GATED_PLANS = {"P1": PLAN_P1_PROTOCOL}
PLANS = (*BASE_PLANS, *ADDED_PLANS)
SEMFILTER_PLANS = ("P2", "P2+Rank", "P1+Rank", "P1")
EDGES = ("similar+similar_up",)
RENDERS = ("agentenv_card",)
POOL_CAP = 256
EXPAND_LIMIT = 4096
SIM_GATES_SEARCHED = ("none", 0.5, 0.6, 0.7)  # the τ² -v5 space (gate_searched=True)
SIM_GATES = ("none",)  # -v6: fixed (user 2026-10-20)
FIXED_SIM_GATE = "none"
MEMORY_FORMS = (
    "raw",
    "compact",
    "distilled",
)  # vocabulary (-v5: one fixed level per dataset)
FIXED_MEMORY_FORM = {
    "alfworld": "raw",
    "sciworld": "raw",
    "tau2": "compact",
}  # user decision 2026-10-20
MEMORY_FORM = (
    "raw"  # this space (ALFWorld / ScienceWorld); τ²: session_space_tau2_v2.MEMORY_FORM
)
MEMORY_FORMS_VERSION = "memory-forms-v1"  # trimem.experiments.memory_forms.VERSION
OUTCOME_POLICIES = ("success_only", "success_failure", "success_failure_capped")
SEMFILTER_THRESHOLDS = (0.4, 0.6, 0.8)
FIXED = {
    "edges": "similar+similar_up",
    "memory_form": MEMORY_FORM,
    "sim_gate": FIXED_SIM_GATE,
}
NUMERIC = ("seed_count", "depth", "fanout", "top_k", "sim_gate", "semfilter_threshold")
CAT_DIMS = [
    6,
    7,
]  # plan, outcome_policy (feature positions; -v5: memory_form fixed, not a feature)
NOMEM = {"plan": "NoMemory", "seed_count": 0, "depth": 0, "fanout": 0, "top_k": 0}
DISPLAY = {
    "VGR": "VGR",
    "VGR+Rank": "VGR+Rank",
    "P2": "VGR+SemFilter",
    "P2+Rank": "VGR+SemFilter+Rank",
    "P1+Rank": "V+SemFilter+G+Rank",  # SemFilter on the vector seeds, then graph expansion, then ColBERT rank
    "P1": "V+SemFilter+G",  # plan-p1-v1: SemFilter on the vector seeds, then graph expansion (no Rank)
}
_MAX = {
    "seed_count": 128,
    "depth": 4,
    "fanout": 64,
    "top_k": 10,
    "sim_gate": 0.7,
    "semfilter_threshold": 0.8,
}


def grid(gate_searched: bool = False):
    """Searched levels. -v6: sim_gate is not searched (fixed in ``v7_defaults``); ``gate_searched=True`` = the τ²
    -v5 grid (same key order as before, so τ² behaviour is unchanged)."""
    g = {
        "seed_count": (4, 8, 16, 32, 64, 128),
        "depth": (2, 3, 4),
        "fanout": (4, 8, 16, 32, 64),
        "top_k": (3, 5, 8, 10),
        "sim_gate": SIM_GATES_SEARCHED,
        "outcome_policy": OUTCOME_POLICIES,
        "semfilter_threshold": SEMFILTER_THRESHOLDS,
    }
    if not gate_searched:
        g.pop("sim_gate")
    return g


def plan_defaults():
    """Round-0 plan defaults inside the v2 grid (the v1 defaults seed 8 / depth 1 / fanout 4 / top_k 3 are
    off-grid): seed 8, depth 2, fanout 4, top_k 5, edges similar+similar_up, no gate, raw cards, success+failure
    (the current memory policy), SemFilter threshold 0.6 (= the v1 FILTER_THRESHOLD). The fixed memory_form is
    supplied by ``v7_defaults`` (-v5)."""
    return {
        "seed_count": 8,
        "depth": 2,
        "fanout": 4,
        "top_k": 5,
        "sim_gate": "none",
        "outcome_policy": "success_failure",
        "semfilter_threshold": 0.6,
    }


def orderings(plan):
    return ("colbert_score",) if plan.endswith("+Rank") else ("query_cosine",)


def fallbacks(plan):
    return ("none",)


# semfilter-v2b (user 2026-10-23, WebShop from round 1): ``sem_fallback`` "fill-topk" is a VALID level for
# SemFilter plans (``session_retrieval.retrieve``: a SemFilter stage passing fewer than top_k candidates is filled
# with the non-passed ones in the plan's order). It is never searched and never a default: ``v7_defaults`` keeps
# "none", so every ALFWorld / ScienceWorld / τ² configuration, key, runner argument and GP feature is unchanged.
# Only a run that records semfilter-v2b sets it on PROPOSED SemFilter configurations
# (``session_pool.use_proposal_rule``) and on its test / fixed-plan SemFilter arms. Not a GP feature (the
# semfilter-regime categorical separates the regimes).
SEM_FALLBACK_LEVELS = ("none", "fill-topk")


def v7_defaults(plan, memory_form: str = MEMORY_FORM, gate_searched: bool = False):
    """Fixed capabilities (interface name kept for the shared engine) + the fixed edges and memory_form (+ the
    fixed sim_gate, -v6, unless ``gate_searched``)."""
    gate = {} if gate_searched else {"sim_gate": FIXED_SIM_GATE}
    return {
        **gate,
        "render": "agentenv_card",
        "ordering": orderings(plan)[0],
        "dedup": "id",
        "cadence": 1,
        "sem_fallback": "none",
        "edges": FIXED["edges"],
        "memory_form": memory_form,
    }


def domain(plan, gate_searched: bool = False):
    if plan == "NoMemory":
        return {}
    if plan not in PLANS:
        raise ValueError("Unknown session plan")
    g = grid(gate_searched)
    if plan not in SEMFILTER_PLANS:
        g.pop("semfilter_threshold")
    return g


def canonical_memory(form: str, outcome: str) -> tuple[str, str]:
    """-v4: memory_form and outcome_policy are orthogonal (identity; kept for the shared interface)."""
    return form, outcome


def complete(cfg, memory_form: str = MEMORY_FORM, gate_searched: bool = False):
    if cfg["plan"] == "NoMemory":
        return dict(NOMEM)
    out = {
        "plan": cfg["plan"],
        **plan_defaults(),
        **v7_defaults(cfg["plan"], memory_form, gate_searched),
        **cfg,
    }
    out["memory_form"], out["outcome_policy"] = canonical_memory(
        out["memory_form"], out["outcome_policy"]
    )
    if out["plan"] not in SEMFILTER_PLANS:  # inactive: held at its default
        out["semfilter_threshold"] = plan_defaults()["semfilter_threshold"]
    return out


def validate(cfg, memory_form: str = MEMORY_FORM, gate_searched: bool = False):
    """Canonical configuration; ``memory_form`` = the space's fixed form (τ² passes ``compact``);
    ``gate_searched`` = sim_gate is a searched dimension (τ² -v5), else fixed none (-v6)."""
    if memory_form not in MEMORY_FORMS:
        raise ValueError(f"Unknown memory_form {memory_form}")
    out = complete(cfg, memory_form, gate_searched)
    if out["plan"] == "NoMemory":
        return out
    g = grid(gate_searched)
    for k, levels in g.items():
        if out[k] not in levels or isinstance(out[k], bool):
            raise ValueError(f"Invalid session configuration {k}={out[k]}")
    fixed = v7_defaults(out["plan"], memory_form, gate_searched)
    for k, value in fixed.items():
        if (
            k == "sem_fallback"
            and out["plan"] in SEMFILTER_PLANS
            and out[k] in SEM_FALLBACK_LEVELS
        ):
            continue  # semfilter-v2b: fill-topk is valid for SemFilter plans (see SEM_FALLBACK_LEVELS)
        if out[k] != value:
            raise ValueError(f"Unsupported session capability {k}={out[k]}")
    if set(out) != {"plan", *g, *fixed}:
        raise ValueError("Unknown configuration fields")
    return out


def key(cfg, memory_form: str = MEMORY_FORM, gate_searched: bool = False):
    return json.dumps(
        validate(cfg, memory_form, gate_searched), sort_keys=True, separators=(",", ":")
    )


def horizon(cfg):
    return 1


def _num(cfg, k):
    v = cfg[k]
    return 0.0 if v == "none" else float(v) / _MAX[k]


def features(
    cfg,
    task=None,
    start_norm=0,
    iter_start=0,
    h=1,
    memory_form: str = MEMORY_FORM,
    gate_searched: bool = False,
):
    cfg = validate(cfg, memory_form, gate_searched)
    if cfg["plan"] == "NoMemory":
        return [0.0] * len(NUMERIC) + [
            float(len(PLANS)),
            float(len(OUTCOME_POLICIES)),
        ]
    return [_num(cfg, k) for k in NUMERIC] + [
        float(PLANS.index(cfg["plan"])),
        float(OUTCOME_POLICIES.index(cfg["outcome_policy"])),
    ]


def features_unchecked(
    cfg, memory_form: str = MEMORY_FORM, gate_searched: bool = False
) -> list[float]:
    """GP input of an AUXILIARY observation (aux-obs-v1, user 2026-10-20): the same encoding as ``features`` but
    without the grid check, so earlier profiling runs with off-grid levels (depth 1, fanout 2, top_k 1) can inform
    the GP. Numeric dimensions keep their ``_MAX`` scaling: every aux level lies in (0, _MAX], so the union of grid
    and aux values is normalised by the same constants and current points keep their encoding. Plan / outcome
    policy must be levels of this space; a value above ``_MAX`` raises. The PROPOSAL space is unchanged."""
    out = complete(cfg, memory_form, gate_searched)
    if out["plan"] == "NoMemory":
        return features(out, memory_form=memory_form, gate_searched=gate_searched)
    if out["plan"] not in PLANS or out["outcome_policy"] not in OUTCOME_POLICIES:
        raise ValueError(f"aux configuration outside the space's categories: {cfg}")
    for k in NUMERIC:
        v = out[k]
        if v != "none" and (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not 0 < v <= _MAX[k]
        ):
            raise ValueError(f"aux configuration {k}={v} outside (0, {_MAX[k]}]")
    return [_num(out, k) for k in NUMERIC] + [
        float(PLANS.index(out["plan"])),
        float(OUTCOME_POLICIES.index(out["outcome_policy"])),
    ]


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


def max_failed(cfg) -> int | None:
    return (
        cfg["top_k"] // 2 if cfg["outcome_policy"] == "success_failure_capped" else None
    )


def runner_args(
    cfg, memory_form: str = MEMORY_FORM, gate_searched: bool = False
) -> dict:
    """agentgym_session_eval arguments for a configuration (``v2`` = the v2-only options)."""
    cfg = validate(cfg, memory_form, gate_searched)
    return {
        "plan": cfg["plan"],
        "edges": cfg["edges"],
        "memory_config": {
            k: cfg[k] for k in ("seed_count", "depth", "fanout", "top_k")
        },
        "v2": {
            "sim_gate": None if cfg["sim_gate"] == "none" else cfg["sim_gate"],
            "memory_form": cfg["memory_form"],
            "outcome_policy": cfg["outcome_policy"],
            "semfilter_threshold": cfg["semfilter_threshold"],
            "pool_cap": POOL_CAP,
            "expand_limit": EXPAND_LIMIT,
            "max_failed": max_failed(cfg),
            "memory_forms_version": MEMORY_FORMS_VERSION,
            # semfilter-v2b only (absent for "none": runner arguments of every earlier configuration unchanged)
            **(
                {"sem_fallback": cfg["sem_fallback"]}
                if cfg["sem_fallback"] != "none"
                else {}
            ),
        },
    }


def runner_cli(cfg) -> list[str]:
    """Extra CLI arguments of ``agentgym_session_eval`` for a v2 configuration."""
    return ["--space_v2", json.dumps(runner_args(cfg)["v2"], sort_keys=True)]


def executed(session_config: dict, protocol: dict) -> dict:
    """What an opt row actually ran, in ``runner_args`` form (SessionConfigV2 fields + the recorded v2
    options), for the declared-vs-executed check."""
    edges = session_config["edges"]
    v2 = dict(protocol.get("session_v2") or {})
    for k in ("sim_gate", "pool_cap", "expand_limit", "max_failed"):
        v2[k] = session_config.get(k)
    return {
        "plan": session_config["plan"],
        "edges": "+".join(edges) if isinstance(edges, (list, tuple)) else edges,
        "memory_config": {
            k: session_config[k] for k in ("seed_count", "depth", "fanout", "top_k")
        },
        "v2": {
            k: v2.get(k)
            for k in (
                "sim_gate",
                "memory_form",
                "outcome_policy",
                "semfilter_threshold",
                "pool_cap",
                "expand_limit",
                "max_failed",
                "memory_forms_version",
            )
        }
        | (
            {"sem_fallback": v2["sem_fallback"]}
            if v2.get("sem_fallback", "none") != "none"
            else {}
        ),
    }
