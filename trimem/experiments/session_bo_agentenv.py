"""Three-objective BO for ALFWorld / ScienceWorld session retrieval (training panels only).

Reuses the shared session-pool engine (``trimem.optimizer.session_pool``: three GPs on
[paired gain, -ln tokens, -ln latency], qLogNEHVI, Sobol / DLS / crossover candidate channels)
with the agent-env space (``session_space_agentenv``). This module adds what the agent-env run
needs: loading training rows from ``opt``, the round-0 design, batches of 5 (kriging believer:
each pick is added at its posterior mean before the next pick), hypervolume / Pareto summaries
and the frozen Q-best / Knee selection. It never reads split=test rows.

Run with ``.venv-optimizer`` (botorch).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import replace

from trimem.optimizer import session_observations as hb
from trimem.optimizer import session_pool as pool
from trimem.optimizer import session_space_agentenv as space

pool.use_space(space)

ROUND0_SOBOL = 6  # with 4 plans; round 0 is ROUND0_TOTAL configurations whatever the number of plans
ROUND0_TOTAL = 10  # plan defaults + Sobol fill (4 plans: 4 + 6; agent-env -v7, 5 plans: 5 + 5; plan-p1rank-v1)
BATCH = 5
STALL_ROUNDS = 1  # tau_s: rounds (of BATCH evaluations) without a new relaxed-front point before forcing
RELAX_Z = 1.0  # user decision 2026-10-16: a point is dominated only if beaten by > 1 pooled SE in some objective


BATCH_TRAIN500 = (
    10  # user 2026-10-17: train500 profiling = 10 rounds of 10 (round 0 + 9 BO rounds)
)
# Experiment version per training panel: the finished 100-panel run keeps its id; the 500-panel run is v2.
# train200 (user 2026-10-18): WebShop's product-isolated 200-query panel, a standalone panel (it does not
# start with the train100 items), experiment <ds>:train200:bo3obj-v2.
PANEL_VERSIONS = {
    "train100": "bo3obj-v1",
    "train500": "bo3obj-v2",
    "train200": "bo3obj-v2",
}


def experiment_id(dataset: str, panel_name: str = "train100") -> str:
    """BO experiment id of a training panel (default: the finished 100-panel experiment), under the
    current memory policy (``memory_policy.tag``: success-only ids are unchanged)."""
    from trimem.experiments import memory_policy as mp

    if panel_name not in PANEL_VERSIONS:
        raise ValueError(f"unknown training panel {panel_name!r}")
    return mp.tag(f"{dataset}:{panel_name}:{PANEL_VERSIONS[panel_name]}")


def expected_configs(rounds: int, batch: int = BATCH) -> int:
    """Configurations of a finished profiling run: round 0 (plan defaults + Sobol fill = ``ROUND0_TOTAL``) + batch
    per BO round."""
    return ROUND0_TOTAL + batch * rounds


def task_of(dataset: str, panel_hash: str) -> str:
    return f"{dataset}:{panel_hash}"


def context(dataset: str, panel_hash: str) -> pool.Context:
    return pool.Context(task_of(dataset, panel_hash), 0.0, 0, 1)


def load_points(
    rows: list[dict],
    *,
    panel_ids: list[str],
    dataset: str,
    panel_hash: str,
    experiment: str | None = None,
) -> list[hb.Point]:
    """Rows of opt.session_retrieval (split=train, this BO experiment) -> one point per complete
    configuration. The optimizer's configuration is the space form stored in protocol.space_config.
    ``experiment`` defaults to the 100-panel experiment id."""
    exp = experiment or experiment_id(dataset)
    mapped = []
    for r in rows:
        if r["split"] != "train":
            raise ValueError("BO input must be training rows only")
        if r["experiment_id"] != exp:
            continue
        if r["protocol"].get("space_version") != space.VERSION:
            raise ValueError("Row profiled under a different space version")
        declared = space.validate(r["protocol"]["space_config"])
        if executed_of(r) != space.runner_args(declared):
            raise ValueError(
                f"Executed configuration {r['config']} differs from the declared {declared}"
            )
        mapped.append({**r, "config": declared})
    return hb.aggregate(
        mapped,
        panel_ids=panel_ids,
        dataset=dataset,
        panel_hash=panel_hash,
        experiment_id=exp,
    )


def use_space(module) -> None:
    """Switch this module (and the shared BO engine) to another agent-env space, e.g.
    ``session_space_agentenv_v2`` for the pool-profiling round. Default: v1 (unchanged)."""
    global space
    space = module
    pool.use_space(module)


def use_plans(plans) -> None:
    """Restrict the proposable plans of the active space (``session_pool.use_plans``; plan-p1rank-v1)."""
    pool.use_plans(plans)


def use_latency_regime_for(state: dict, r: int) -> bool:
    """latency-regime-v1 (bo_pool_driver.latency_regime_block): from its from_round the configurations of earlier
    rounds are the old latency regime (session_pool.use_latency_regime); off before / without it."""
    blk = state.get("latency_regime")
    if not blk or r < blk["from_round"]:
        pool.use_latency_regime(None)
        return False
    pool.use_latency_regime(
        {
            space.key(c["config"])
            for rr in state["rounds"]
            if rr["round"] < blk["from_round"]
            for c in rr["configs"]
        }
    )
    return True


def use_semfilter_regime_for(state: dict, r: int) -> bool:
    """semfilter-regime-v1 (bo_pool_driver.semfilter_block): from its from_round the SemFilter-plan configurations
    of earlier rounds are the old judge (session_pool.use_semfilter_regime); off before / without it."""
    blk = state.get("semfilter")
    if not blk or r < blk["from_round"]:
        pool.use_semfilter_regime(None)
        return False
    # semfilter-v2b (WebShop, user 2026-10-23): from its from_round the SemFilter configurations of EARLIER rounds
    # (semfilter-v2: rubric v3, no fallback) are the old regime (= 1); predictions at v2b (= 0)
    v2b = blk.get(SEMFILTER_V2B_KEY)
    r0 = v2b["from_round"] if v2b and r >= v2b["from_round"] else blk["from_round"]
    pool.use_semfilter_regime(
        {
            space.key(c["config"])
            for rr in state["rounds"]
            if rr["round"] < r0
            for c in rr["configs"]
            if c["config"].get("plan") in pool.SEMFILTER_REGIME_PLANS
        }
    )
    return True


SEMFILTER_V2B_KEY = (
    "v2b"  # state.json semfilter.v2b (bo_pool_driver.semfilter_v2b_block)
)


def semfilter_v2b_rule(v2b: dict) -> dict:
    """The proposal rule of a recorded semfilter-v2b block (``session_pool.use_proposal_rule``). Pure."""
    return {
        "plans": list(pool.SEMFILTER_REGIME_PLANS),
        "levels": {"semfilter_threshold": list(v2b["thresholds"])},
        "set": {"sem_fallback": v2b["sem_fallback"]},
    }


def use_semfilter_v2b_for(state: dict, r: int) -> bool:
    """semfilter-v2b: from its from_round every PROPOSED SemFilter configuration carries sem_fallback fill-topk and
    a threshold in the recorded set (0.8 not proposable); off before / without it (proposals unchanged)."""
    v2b = (state.get("semfilter") or {}).get(SEMFILTER_V2B_KEY)
    if not v2b or r < v2b["from_round"]:
        pool.use_proposal_rule(None)
        return False
    pool.use_proposal_rule(semfilter_v2b_rule(v2b))
    return True


PLAN_P1_KEY = "plan_extension_p1"  # state.json block (plan-p1-v1, user 2026-10-22)


def plan_p1_block(state: dict, now: str, want: bool) -> dict | None:
    """plan-p1-v1 (user 2026-10-22): the gated plan P1 becomes proposable from the next round to be proposed,
    recorded once in state.json ``plan_extension_p1`` (``want`` = the opt-in of this start; a recorded block stays
    without it). The space identity (VERSION) is unchanged; earlier rounds keep their proposal plans."""
    blk = state.get(PLAN_P1_KEY)
    gated = getattr(space, "GATED_PLANS", {}) or {}
    if blk is not None:
        if blk.get("protocol") != gated.get("P1"):
            raise SystemExit(
                f"state.json {PLAN_P1_KEY} {blk.get('protocol')!r}; refusing to mix"
            )
        return blk
    if not want:
        return None
    if "P1" not in gated:
        raise SystemExit(f"the active space {space.VERSION} has no gated plan P1")
    state[PLAN_P1_KEY] = {
        "protocol": gated["P1"],
        "from_round": max(1, len(state["rounds"])),
        "added_plans": ["P1"],
        "display": space.DISPLAY["P1"],
        "plan_default": space.validate({"plan": "P1", **space.plan_defaults()}),
        "rule": "P1 joins the proposable plans (Sobol breadth, DLS plan switches, analyst view, snapping) from "
        "from_round; while unmeasured, its plan default is an extra DLS start (dls-newplan, "
        "session_pool.unmeasured_plans); GP category appended (earlier plans keep their index)",
        "space_version": space.VERSION,
        "recorded_at": now,
        "note": "user 2026-10-22: add P1 (V->SemFilter->G, no Rank) to the plan search space",
    }
    return state[PLAN_P1_KEY]


def proposal_plans(state: dict, r: int) -> tuple[str, ...]:
    """Plans proposable in round ``r``: the space's ungated plans, plus gated plans whose recorded extension
    applies from a round <= ``r`` (plan-p1-v1)."""
    gated = getattr(space, "GATED_PLANS", {}) or {}
    blk = state.get(PLAN_P1_KEY)
    on = set(blk["added_plans"]) if blk and r >= blk["from_round"] else set()
    return tuple(p for p in space.PLANS if p not in gated or p in on)


def executed_of(row: dict) -> dict:
    """Executed configuration of an opt row in the active space's ``runner_args`` form."""
    if hasattr(space, "executed"):
        return space.executed(row["config"], row["protocol"])
    return executed(row["config"])


def executed(session_config: dict) -> dict:
    """The SessionConfig stored in an opt row, in ``space.runner_args`` form."""
    edges = session_config["edges"]
    return {
        "plan": session_config["plan"],
        "edges": "+".join(edges) if isinstance(edges, (list, tuple)) else edges,
        "memory_config": {
            k: session_config[k] for k in ("seed_count", "depth", "fanout", "top_k")
        },
    }


def defaults() -> list[dict]:
    """Plan defaults of the proposable plans (``session_pool.RETRIEVAL_PLANS``: every plan of the space unless
    ``pool.use_plans`` restricted them), in the space's plan order."""
    return [
        space.validate({"plan": p, **space.plan_defaults()})
        for p in pool.RETRIEVAL_PLANS
    ]


def round0(seed: int = 0, *, sampler: str = "sobol") -> list[dict]:
    """Plan defaults + coverage up to ROUND0_TOTAL; preserve legacy Sobol draws.

    LHS draws the remaining allowance directly, with bounded deterministic
    top-ups for NoMemory, canonical duplicates and collisions with defaults.
    Consequently the final discrete batch need not be an exact Latin hypercube.
    """
    out = defaults()
    seen = {space.key(c) for c in out}
    if sampler == "lhs":
        for attempt in range(64):
            if len(out) >= ROUND0_TOTAL:
                return out
            for c in pool.coverage_breadth(
                ROUND0_TOTAL - len(out), seed + 7 + attempt, sampler="lhs"
            ):
                key = space.key(c)
                if c["plan"] != "NoMemory" and key not in seen:
                    out.append(c)
                    seen.add(key)
        raise ValueError("LHS initialization could not fill the distinct batch")
    if sampler != "sobol":
        raise ValueError(f"unknown initialization sampler {sampler!r}")
    for c in pool.sobol_breadth(256, seed + 7):
        if c["plan"] == "NoMemory" or space.key(c) in seen:
            continue
        out.append(c)
        seen.add(space.key(c))
        if len(out) == ROUND0_TOTAL:
            break
    return out


def _se(p: hb.Point) -> tuple[float, float, float]:
    return tuple(math.sqrt(v) for v in p.noise)


def relaxed_dominates(q: hb.Point, p: hb.Point, z: float = RELAX_Z) -> bool:
    """q statistically dominates p: at least as good on every objective and better than p by more than
    z pooled standard errors on at least one (gain; -ln tokens; -ln latency)."""
    a, b, sa, sb = objectives(q), objectives(p), _se(q), _se(p)
    return all(x >= y for x, y in zip(a, b, strict=True)) and any(
        a[i] > b[i] + z * math.sqrt(sa[i] ** 2 + sb[i] ** 2) for i in range(3)
    )


def relaxed_pareto(points: list[hb.Point], z: float = RELAX_Z) -> list[hb.Point]:
    """Noise-aware front: points not statistically dominated (z = 0 gives the strict front)."""
    return [
        p
        for p in points
        if not any(q is not p and relaxed_dominates(q, p, z) for q in points)
    ]


def cost(p: hb.Point) -> float:
    """The engine's 2-D cost axis (maximised): mean of -ln tokens and -ln latency."""
    return 0.5 * (-math.log(p.tokens) - math.log(p.latency_ms))


def realized_front_2d(points: list[hb.Point]) -> list[tuple[float, float]]:
    """Non-dominated (gain, cost) pairs of the measured points (both maximised)."""
    xy = [(p.quality, cost(p)) for p in relaxed_pareto(points)]
    return sorted(
        {a for a in xy if not any(b[0] >= a[0] and b[1] >= a[1] and b != a for b in xy)}
    )


def stalled(rounds: list[list[hb.Point]], tau: int = STALL_ROUNDS) -> bool:
    """Stagnation (RAG-PE forced channels, noise-aware): none of the last ``tau`` rounds' points is on
    the relaxed front of everything measured so far."""
    if len(rounds) < tau + 1:
        return False
    front = {id(p) for p in relaxed_pareto([p for r in rounds for p in r])}
    return not any(id(p) in front for r in rounds[-tau:] for p in r)


def propose_batch(
    points: list[hb.Point],
    ctx: pool.Context,
    *,
    k: int = BATCH,
    seed: int = 0,
    exclude: set[str] | None = None,
    forced: bool = False,
    agent_rows: list[dict] | None = None,
    realized_front: list[tuple[float, float]] | None = None,
    qfirst: dict | None = None,
    slots: dict | None = None,
    aux: list[hb.Point] | None = None,
    search_options: dict | None = None,
    stall_recovery: bool = False,
) -> list[dict]:
    """k distinct qLogNEHVI decisions. Kriging believer: after each pick the chosen configuration is
    added as a pseudo-observation at its posterior mean (with the smallest measured noise), so the
    next pick accounts for it. NoMemory is never proposed (its paired gain is 0 by definition).

    ``qfirst`` (``qfirst_settings``; user 2026-10-20, BO rounds >= ``QFIRST_FROM_ROUND``): the FIRST slot is a
    quality-first pick (``acq_mode="qfirst"``: UCB_beta(quality) x P(tokens and latency within budget)) from the
    same candidate pool, with the measured quality champion as an extra DLS start; the believer conditions on it
    and the remaining k - 1 slots are the usual hv picks (seeds ``seed + j`` as without the slot).

    ``slots`` (``channel_slots_settings``; channel-slots-v1, user 2026-10-20): ordinary (non-forced) rounds fill the
    batch in ``slot_plan`` order -- qfirst, then the crossover slot (argmax qLogNEHVI over crossover_v2 rows and
    dls-crossover results), the Sobol exploration slot (largest range-normalised posterior uncertainty among the
    Sobol breadth rows, hierarchy-aware novelty tie-break), then hv picks over the whole pool; each pick conditions
    on the earlier ones (kriging believer). Forced rounds keep the forced logic (qfirst + forced picks) with the
    richer crossover rows in the pool.

    channel-slots-v2 (user 2026-10-21; ``slots["protocol"]``): ordinary rounds fill qfirst, llm-slot, xover-slot,
    dls-slot, sobol-slot (``slot_plan``; believer in that order). Forced rounds keep the forced picks but, for a
    batch of k >= ``FORCED_COVER_MIN_BATCH``, reserve the trailing slots so that crossover, DLS and Sobol each
    contribute at least one non-qfirst pick (``_forced_v2_picks``).

    channel-slots-v4 (user 2026-10-22): as v3 (layout ``SLOT_LAYOUT_V4``: sobol x4, llm x2, xover x2, dls x2) and
    ``qfirst`` is ignored in every v4 round (ordinary and forced).

    ``stall_recovery`` (front-gap-v1): allow stalled v5/v6 batches to use the existing gap-probability
    forced picks and xover/DLS/coverage fills. Ordinary layouts stay unchanged. Explicit sobol-only rounds
    still suppress forcing. False preserves historical runs; drivers read the frozen search_design setting.

    ``aux`` (aux-obs-v1, ``aux_points``): auxiliary observations of earlier runs; they enter every GP fit of the
    batch and nothing else (believer ranges / noise, the forced front and the qfirst champion use ``points``)."""
    exclude = set(exclude or ()) | {space.key(space.NOMEM)}
    believed = list(points)
    picks = []
    # Believer values are clamped to the measured range so they cannot move the engine's
    # reference point (computed from the target minima) during the batch.
    q_lo, q_hi = min(p.quality for p in points), max(p.quality for p in points)
    t_lo, t_hi = min(p.tokens for p in points), max(p.tokens for p in points)
    l_lo, l_hi = min(p.latency_ms for p in points), max(p.latency_ms for p in points)
    noise = tuple(min(p.noise[i] for p in points) for i in range(3))

    def believe(j, cfg, gain, tokens, latency_ms):
        believed.append(
            replace(
                points[0],
                point_id=f"believer-{j}",
                cfg=cfg,
                quality=min(max(gain, q_lo), q_hi),
                tokens=min(max(tokens, t_lo), t_hi),
                latency_ms=min(max(latency_ms, l_lo), l_hi),
                noise=noise,
            )
        )

    aux = list(aux or ())
    if slots and slots["protocol"] in (
        CHANNEL_SLOTS_V4,
        *ONE_PER_CHANNEL,
        CHANNEL_SLOTS_SOBOL,
    ):
        qfirst = None  # channel-slots-v4 / -v5 / sobol-only-v1: no qfirst pick (ordinary and forced rounds)
    if slots and (
        slots["protocol"] == CHANNEL_SLOTS_SOBOL
        or (slots["protocol"] in ONE_PER_CHANNEL and not stall_recovery)
    ):
        forced = False  # Explicit random rounds and historical v5/v6 keep their recorded behavior.
    if qfirst and k > 0:
        picks.append(
            _qfirst_pick(
                points, ctx, qfirst, seed, exclude, agent_rows, aux, search_options
            )
        )
        q = picks[0]
        exclude.add(space.key(q["config"]))
        believe(
            "qfirst",
            q["config"],
            q["pred_gain"],
            q["pred_tokens"],
            q["pred_latency_ms"],
        )
    if (
        slots
        and forced
        and slots["protocol"]
        in (
            CHANNEL_SLOTS_V2,
            CHANNEL_SLOTS_V3,
            CHANNEL_SLOTS_V4,
            *(ONE_PER_CHANNEL if stall_recovery else ()),
        )
    ):
        return _named(
            slots,
            picks
            + _forced_v2_picks(
                points,
                believed,
                ctx,
                slots,
                k,
                k - len(picks),
                seed,
                exclude,
                agent_rows,
                realized_front,
                believe,
                aux,
                search_options,
            ),
        )
    if slots and not forced:
        return _named(
            slots,
            picks
            + _slot_picks(
                believed,
                ctx,
                slots,
                k - len(picks),
                seed,
                exclude,
                agent_rows,
                believe,
                aux,
                search_options,
            ),
        )
    xs = slots["xover"] if slots else None
    for j in range(k - len(picks)):
        # Forced (stagnation) decisions target the largest gap of the realized (gain, cost) front:
        # crossover candidates ranked by their probability of landing in that gap (engine's forced mode).
        # Analyzer rows (RAG-PE channel R^LLM) enter the pool and seed DLS; arbitration is unchanged.
        res = pool.suggest(
            believed,
            ctx,
            seed=seed + j,
            exclude=exclude,
            acq_mode="hv",
            forced=forced,
            agent_rows=agent_rows,
            realized_front=(realized_front or realized_front_2d(points))
            if forced
            else None,
            xover=xs,
            dls_crossover=bool(xs),
            aux=aux,
            **(search_options or {}),
        )
        ch = res["chosen"]
        cfg = space.validate(ch["config"])
        if not pool.proposable(
            cfg
        ):  # proposal-side constraint of the space (PROPOSE_FIXED)
            raise AssertionError(f"proposal violates {pool.propose_fixed()}: {cfg}")
        picks.append(
            {
                "config": cfg,
                "channel": ch["channel"],
                "acq": ch["acq"],
                "mode": res["mode"],
                "pred_gain": ch["pred_quality_per_iter"],
                "pred_tokens": ch["pred_tokens_per_iter"],
                "pred_latency_ms": ch["pred_latency_ms_per_iter"],
                "acquisition": res["acquisition"],
                "pool_size": res["pool_size"],
                "plan_counts": res.get("plan_counts"),
                **(
                    {"search_audit": res["search_audit"]}
                    if "search_audit" in res
                    else {}
                ),
                **(
                    {"channel_counts": res["channel_counts"], "slot": "forced"}
                    if xs
                    else {}
                ),
            }
        )
        exclude.add(space.key(cfg))
        believe(
            j,
            cfg,
            ch["pred_quality_per_iter"],
            ch["pred_tokens_per_iter"],
            ch["pred_latency_ms_per_iter"],
        )
    return picks


# ------------------------------------------------------------------ channel slots (channel-slots-v1, user 2026-10-20)

CHANNEL_SLOTS_PROTOCOL = (
    "channel-slots-v1"  # default of channel_slots_block (τ², replay tools): unchanged
)
CHANNEL_SLOTS_V2 = "channel-slots-v2"  # user 2026-10-21: ALF/Sci fresh runs (bo_pool_driver), from round 1
CHANNEL_SLOTS_V3 = "channel-slots-v3"  # user 2026-10-22: opt-in (POOL_CHANNEL_SLOTS=v3), batch 10, from the next round
# channel-slots-v4 (user 2026-10-22): v3 with the qfirst / token-first / latency-first quotas replaced by Sobol slots
CHANNEL_SLOTS_V4 = "channel-slots-v4"
# sobol-only-v1 (user 2026-10-22, ScienceWorld): random-sampling rounds -- every slot is a sobol-slot pick (max
# posterior-SD product over the Sobol breadth rows, kriging believer); recorded as a channel-slot block for a round
# range [from_round, the v4 block's from_round) in the ``before`` chain of channel-slots-v4 (``sobol_until``)
CHANNEL_SLOTS_SOBOL = "sobol-only-v1"
# channel-slots-v5 (user 2026-10-22, ALFWorld v7 from round 7): batch 4 = one pick per channel, layout (believer
# order) llm, dls, xover, sobol -- each by its v4 rule, an empty family back-filled with the HV-best over the union
# (recorded); never stall-forced; smaller (budget-tail) batches keep the first k of the layout
CHANNEL_SLOTS_V5 = "channel-slots-v5"
SLOT_LAYOUT_V5 = ("llm", "dls", "xover", "sobol")
# channel-slots-v6 (user 2026-10-22, ScienceWorld from round 3): batch 8 = the v5 per-channel picks (llm, dls, xover,
# sobol; same rules, back-fill and believer) followed by 4 hv picks = the top remaining qLogNEHVI candidates over the
# UNION of every channel's candidates (sequential, kriging believer, measured / picked excluded); never stall-forced;
# smaller (budget-tail) batches keep the first k of the layout (k = 2: llm, dls); hv beyond 8
CHANNEL_SLOTS_V6 = "channel-slots-v6"
SLOT_LAYOUT_V6 = (*SLOT_LAYOUT_V5, "hv", "hv", "hv", "hv")
CHANNEL_SLOTS_PROTOCOLS = (
    CHANNEL_SLOTS_PROTOCOL,
    CHANNEL_SLOTS_V2,
    CHANNEL_SLOTS_V3,
    CHANNEL_SLOTS_V4,
    CHANNEL_SLOTS_SOBOL,
    CHANNEL_SLOTS_V5,
    CHANNEL_SLOTS_V6,
)
# opt-in upgrades, in rank order (an upgrade only ever goes to a later entry; downgrades are refused)
CHANNEL_SLOTS_TARGETS = (CHANNEL_SLOTS_V4, CHANNEL_SLOTS_V5, CHANNEL_SLOTS_V6)
ONE_PER_CHANNEL = (
    CHANNEL_SLOTS_V5,
    CHANNEL_SLOTS_V6,
)  # no qfirst, never stall-forced, analyst once per round
CHANNEL_SLOTS_NAMED = (
    CHANNEL_SLOTS_V3,
    CHANNEL_SLOTS_V4,
    CHANNEL_SLOTS_SOBOL,
    CHANNEL_SLOTS_V5,
    CHANNEL_SLOTS_V6,
)  # per-pick slot_name / slot_rule, backfill, coverage
CHANNEL_SLOTS_FROM_ROUND = 1  # = QFIRST_FROM_ROUND for a fresh run (user 2026-10-21: from round 1); a live run: the next round to be proposed
SLOT_PRIORITY = ("qfirst", "xover", "sobol")  # then hv for every remaining slot
SLOT_PRIORITY_V2 = (
    "qfirst",
    "llm",
    "xover",
    "dls",
    "sobol",
)  # then hv for every remaining slot
# channel-slots-v3 (user 2026-10-22): the 10-slot layout (believer order) and the priority of smaller batches
SLOT_LAYOUT_V3 = (
    "qfirst",
    "tokfirst",
    "latfirst",
    "llm",
    "llm",
    "xover",
    "xover",
    "dls",
    "dls",
    "sobol",
)
SLOT_PRIORITY_V3 = (
    "qfirst",
    "tokfirst",
    "latfirst",
    "llm",
    "xover",
    "dls",
    "sobol",
    "llm",
    "xover",
    "dls",
)  # then hv for every remaining slot
# channel-slots-v4 (user 2026-10-22): sobol x4, llm x2, xover x2, dls x2 (believer order); no qfirst / cost-first.
# Budget-tail truncation keeps the first k of SLOT_PRIORITY_V4 (one of each channel, then the Sobol pair, again),
# laid out in SLOT_LAYOUT_V4 order: k = 4 -> sobol, llm, xover, dls; k = 5 -> sobol x2, llm, xover, dls.
SLOT_LAYOUT_V4 = (
    "sobol",
    "sobol",
    "sobol",
    "sobol",
    "llm",
    "llm",
    "xover",
    "xover",
    "dls",
    "dls",
)
SLOT_PRIORITY_V4 = (
    "llm",
    "xover",
    "dls",
    "sobol",
    "sobol",
    "llm",
    "xover",
    "dls",
    "sobol",
    "sobol",
)  # then hv for every remaining slot
SLOT_LABELS = {
    "tokfirst": "token-first",
    "latfirst": "latency-first",
    "xover": "xover-slot",
    "sobol": "sobol-slot",
    "llm": "llm-slot",
    "dls": "dls-slot",
    "hv": "hv",
}
SLOT_SEED_OFFSETS = {
    "tokfirst": 300,
    "latfirst": 350,
    "xover": 600,
    "sobol": 700,
    "llm": 800,
    "dls": 900,
}  # hv slots keep seed + j (j = 0, 1, ...); channel-slots-v3: the i-th repeat of a kind uses offset + i
SLOT_RULES_V3 = {
    "qfirst": "UCB_1.5(gain) x P(cost within the guard budget) (qfirst-slot-v1, unchanged)",
    "tokfirst": "token-first: lowest posterior ln tokens among the union candidates whose posterior gain >= quality "
    "champion - 1 pooled posterior SD (session_pool.COST_FIRST_RULE)",
    "latfirst": "latency-first: lowest posterior ln latency among the union candidates whose posterior gain >= "
    "quality champion - 1 pooled posterior SD (session_pool.COST_FIRST_RULE)",
    "llm": "argmax qLogNEHVI over the analyst rows + dls-agent endpoints",
    "xover": "argmax qLogNEHVI over crossover rows + dls-crossover endpoints",
    "dls": "argmax qLogNEHVI over dls-incumbent / -random / -pool / -newplan endpoints",
    "sobol": "max product of range-normalised posterior SDs over the Sobol breadth rows",
    "hv": "argmax qLogNEHVI over the whole pool",
}
SLOT_RULES_V4 = {
    k: v
    for k, v in SLOT_RULES_V3.items()
    if k not in ("qfirst", "tokfirst", "latfirst")
} | {
    "sobol": "max product of range-normalised posterior SDs over the Sobol breadth rows (hierarchy-aware novelty "
    "tie-break); kriging believer + exclusion of measured / already-picked configurations make the repeats differ",
}
SLOT_RULES_SOBOL = {k: SLOT_RULES_V4[k] for k in ("sobol", "hv")}
SLOT_RULES = {
    CHANNEL_SLOTS_V3: SLOT_RULES_V3,
    CHANNEL_SLOTS_V4: SLOT_RULES_V4,
    CHANNEL_SLOTS_SOBOL: SLOT_RULES_SOBOL,
    CHANNEL_SLOTS_V5: SLOT_RULES_V4,
    CHANNEL_SLOTS_V6: SLOT_RULES_V4
    | {
        "hv": "argmax qLogNEHVI over the union of every channel's candidates (Sobol breadth, analyst rows, "
        "crossover rows, every DLS start's endpoint); measured / already-picked configurations excluded, kriging "
        "believer over the earlier picks of the round"
    },
}
# channel-slots-v2 forced rounds: at batch size >= 4 crossover, DLS and Sobol each contribute >= 1 non-qfirst pick
FORCED_COVER = ("xover", "dls", "sobol")
FORCED_COVER_MIN_BATCH = 4
XOVER_SETTINGS = {
    "top": pool.XOVER_TOP,
    "sizes": list(pool.XOVER_SIZES),
    "max_total": pool.XOVER_MAX,
    "z": pool.XOVER_Z,
}


def slot_plan(
    k: int, qfirst: bool, protocol: str = CHANNEL_SLOTS_PROTOCOL
) -> list[str]:
    """Slot kinds of a batch of ``k``. channel-slots-v1: qfirst (when active), xover, sobol, then hv; smaller
    batches keep this priority (k = 2 with qfirst: qfirst + xover; k = 3: + sobol; k = 5: qfirst, xover, sobol,
    hv, hv). channel-slots-v2: qfirst, llm, xover, dls, sobol (k = 5: one each), hv beyond 5; same priority for
    smaller batches (k = 3: qfirst, llm, xover)."""
    if protocol not in CHANNEL_SLOTS_PROTOCOLS:
        raise ValueError(f"unknown channel-slot protocol {protocol!r}")
    if protocol == CHANNEL_SLOTS_SOBOL:
        return ["sobol"] * k  # sobol-only-v1: random-sampling round (no qfirst)
    if protocol == CHANNEL_SLOTS_V5:
        # channel-slots-v5: one per channel in layout order; k < 4 keeps the first k; hv beyond 4
        return list(SLOT_LAYOUT_V5[:k]) + ["hv"] * max(0, k - len(SLOT_LAYOUT_V5))
    if protocol == CHANNEL_SLOTS_V6:
        # channel-slots-v6: llm, dls, xover, sobol, hv x 4; k < 8 keeps the first k; hv beyond 8
        return list(SLOT_LAYOUT_V6[:k]) + ["hv"] * max(0, k - len(SLOT_LAYOUT_V6))
    if protocol in CHANNEL_SLOTS_NAMED:
        # channel-slots-v3 / -v4: the first k of the priority (hv beyond 10), laid out in layout order; v4 never
        # has a qfirst slot (``qfirst`` ignored)
        v4 = protocol == CHANNEL_SLOTS_V4
        prio, layout = (
            (SLOT_PRIORITY_V4, SLOT_LAYOUT_V4)
            if v4
            else (SLOT_PRIORITY_V3, SLOT_LAYOUT_V3)
        )
        head = [s for s in prio if s != "qfirst" or qfirst][:k]
        rank = {kind: i for i, kind in enumerate(layout)}
        return sorted(head, key=lambda kind: rank[kind]) + ["hv"] * max(
            0, k - len(head)
        )
    prio = SLOT_PRIORITY_V2 if protocol == CHANNEL_SLOTS_V2 else SLOT_PRIORITY
    head = [s for s in prio if s != "qfirst" or qfirst]
    return (head + ["hv"] * max(0, k - len(head)))[:k]


def channel_slots_block(
    state: dict,
    now: str,
    protocol: str = CHANNEL_SLOTS_PROTOCOL,
    sobol_until: int | None = None,
) -> dict:
    """The channel-slot protocol, recorded once in state.json (``state["channel_slots"]``): from_round =
    max(CHANNEL_SLOTS_FROM_ROUND, rounds already proposed) -- never retroactive. Not part of the run identity.
    ``protocol`` applies only when the block is recorded: a state that already records a known protocol keeps it
    (ALFWorld / τ² keep channel-slots-v1); an unknown recorded protocol is refused."""
    if protocol not in CHANNEL_SLOTS_PROTOCOLS or protocol == CHANNEL_SLOTS_SOBOL:
        raise ValueError(f"unknown channel-slot protocol {protocol!r}")
    rec = state.get("channel_slots", {}).get("protocol")
    if sobol_until is not None:
        return _sobol_then_v4(state, now, protocol, rec, sobol_until)
    if rec in CHANNEL_SLOTS_TARGETS and protocol == CHANNEL_SLOTS_V3:
        raise SystemExit(
            f"state.json records {rec}; channel-slots-v3 requested; refusing to downgrade"
        )
    if (
        protocol in CHANNEL_SLOTS_TARGETS
        and rec in CHANNEL_SLOTS_TARGETS
        and CHANNEL_SLOTS_TARGETS.index(protocol) < CHANNEL_SLOTS_TARGETS.index(rec)
    ):
        raise SystemExit(
            f"state.json records {rec}; {protocol} requested; refusing to downgrade"
        )
    if protocol in CHANNEL_SLOTS_TARGETS and rec != protocol:
        # opt-in upgrade (user 2026-10-22; usually v3 -> v4): from the next round to be proposed; earlier rounds
        # keep the recorded protocol (``before``)
        before = state.get("channel_slots")
        if before is not None and before["protocol"] not in CHANNEL_SLOTS_PROTOCOLS:
            raise SystemExit(
                "state.json has another channel-slot protocol; refusing to mix"
            )
        state["channel_slots"] = _target_block(protocol, state, now, before)
        return state["channel_slots"]
    if protocol == CHANNEL_SLOTS_V3 and rec != CHANNEL_SLOTS_V3:
        # opt-in upgrade (user 2026-10-22): from the next round to be proposed; earlier rounds keep the recorded
        # protocol (``before``, used by channel_slots_settings for r < from_round)
        before = state.get("channel_slots")
        if before is not None and before["protocol"] not in CHANNEL_SLOTS_PROTOCOLS:
            raise SystemExit(
                "state.json has another channel-slot protocol; refusing to mix"
            )
        state["channel_slots"] = _channel_slots_v3_block(state, now, before)
        return state["channel_slots"]
    if "channel_slots" not in state and protocol == CHANNEL_SLOTS_V2:
        state["channel_slots"] = _channel_slots_v2_block(state, now)
    elif "channel_slots" not in state:
        state["channel_slots"] = {
            "protocol": CHANNEL_SLOTS_PROTOCOL,
            "from_round": max(CHANNEL_SLOTS_FROM_ROUND, len(state["rounds"])),
            "crossover": {
                **XOVER_SETTINGS,
                "anchors": [
                    "champ-q / champ-c: posterior quality and cost champions (each grafts the other's knobs)",
                    (
                        "adj-q / adj-c: every adjacent pair of the noise-aware relaxed posterior (quality, cost) "
                        f"front (z = {pool.XOVER_Z}), grafted in both directions with the same tau score"
                    ),
                ],
                "graft_sizes": list(pool.XOVER_SIZES),
                "tau": "RAG-Stack §4: tau = n_other(graft) - n_other(star) - [n_own(star) - n_own(graft)]+, "
                "min-max normalised over the star and its grafts",
                "rule": "top per direction by tau (measured / excluded skipped), deduped by key (highest tau), "
                "hierarchy-valid and PROPOSE_FIXED only, at most max_total rows (champion rows first)",
                "label": "crossover:<anchor-kind>:<knob[+knob]>",
            },
            "dls_crossover": "no agent row in the round (analyst disabled or failed): the crossover rows are the "
            "DLS seed set in place of the agent seeds (label dls-crossover); other DLS starts unchanged",
            "slot_order": ["qfirst", "xover-slot", "sobol-slot", "hv", "hv"],
            "small_batch_priority": ["qfirst", "xover-slot", "sobol-slot", "hv"],
            "slots": {
                "qfirst": "unchanged (qfirst-slot-v1)",
                "xover-slot": "argmax qLogNEHVI over {crossover rows} U {dls-crossover results}; source_channel "
                "recorded",
                "sobol-slot": "Coverage breadth rows only (sampler in search_design; legacy Sobol): "
                "max product of the range-normalised posterior SDs of "
                "the three objectives; tie-break hierarchy-aware novelty (RAG-Stack)",
                "hv": "qLogNEHVI over the whole pool (all channels)",
            },
            "believer": "kriging believer in slot order qfirst -> xover -> sobol -> hv",
            "seeds": {
                "qfirst": "round seed + 500",
                "xover": "round seed + 600",
                "sobol": "round seed + 700",
                "hv": "round seed + j",
            },
            "forced": "stall-triggered rounds keep the existing forced logic (qfirst + forced picks ranked by "
            "gap-landing probability; RAG-Stack rule), with the crossover_v2 rows in the pool",
            "acquisition_tag": "bo-acq-v3 (crossover_v2 pool); bo-acq-v2 before from_round",
            "recorded_at": now,
            "note": "user 2026-10-20; run identity unchanged (as budget_extension / analyst); "
            "docs/trimem_pool_profiling_plan_v0.1.0.md addendum channel-slots-v1",
        }
    elif state["channel_slots"]["protocol"] not in CHANNEL_SLOTS_PROTOCOLS:
        raise SystemExit(
            "state.json has another channel-slot protocol; refusing to mix"
        )
    return state["channel_slots"]


def _channel_slots_v2_block(state: dict, now: str) -> dict:
    """The state.json record of channel-slots-v2 (user 2026-10-21)."""
    return {
        "protocol": CHANNEL_SLOTS_V2,
        "from_round": max(CHANNEL_SLOTS_FROM_ROUND, len(state["rounds"])),
        "crossover": {
            **XOVER_SETTINGS,
            "graft_sizes": list(pool.XOVER_SIZES),
            "rule": "unchanged from channel-slots-v1 (session_pool.crossover_v2)",
        },
        "dls_crossover": "unchanged from channel-slots-v1: no agent row in the round -> the crossover rows are "
        "the DLS seed set in place of the agent seeds (label dls-crossover)",
        "slot_order": ["qfirst", "llm-slot", "xover-slot", "dls-slot", "sobol-slot"],
        "small_batch_priority": [
            "qfirst",
            "llm-slot",
            "xover-slot",
            "dls-slot",
            "sobol-slot",
            "hv",
        ],
        "slots": {
            "qfirst": "unchanged (qfirst-slot-v1): UCB(gain) x P(cost within the guard budget)",
            "llm-slot": "argmax qLogNEHVI over the LLM analyst's candidates (agent / agent-gap rows) and the DLS "
            "endpoints started from them (dls-agent; DLS restricted to those starts). Fallback when the analyst "
            "yields nothing valid (or every such row is excluded): the remaining DLS starts run too and the slot "
            "takes the best hv candidate of the whole pool; recorded as channel hv + slot_info.slot_fallback "
            "(analyst_rows, full_pool_dls) and logged",
            "xover-slot": "unchanged: argmax qLogNEHVI over {crossover rows} U {dls-crossover results}",
            "dls-slot": "argmax qLogNEHVI over the endpoints of the DLS starts from incumbents (posterior quality / "
            "cost champions and front spread), Sobol random restarts, the top-8 pool rows and unmeasured plan "
            "defaults (dls-incumbent / dls-random / dls-pool / dls-newplan); never agent or crossover starts",
            "sobol-slot": "unchanged: Sobol breadth rows only, max product of the range-normalised posterior SDs "
            "of the three objectives, hierarchy-aware novelty tie-break",
            "hv": "only for batches > 5: qLogNEHVI over the whole pool",
            "fallback": "any slot with an empty family -> best hv candidate of the whole pool (channel hv)",
        },
        "believer": "kriging believer in slot order qfirst -> llm -> xover -> dls -> sobol",
        "seeds": {
            "qfirst": "round seed + 500",
            "llm": "round seed + 800",
            "xover": "round seed + 600",
            "dls": "round seed + 900",
            "sobol": "round seed + 700",
            "hv": "round seed + j",
            "forced": "round seed + j (j = 0, 1, ... over the forced picks)",
        },
        "forced": (
            "stall-triggered rounds: qfirst first, then the existing forced picks (gap-landing probability over "
            "the forced set; RAG-Stack rule). Coverage rule for a batch of k >= "
            f"{FORCED_COVER_MIN_BATCH}: let M = the kinds of {list(FORCED_COVER)} not yet represented among the "
            "round's non-qfirst picks (a pick represents xover if one of its producing channels is crossover:* or "
            "dls-crossover, dls if one is dls-incumbent/-random/-pool/-newplan, sobol if one is sobol/lhs) and "
            "not yet tried as a fill slot. While the remaining slots exceed |M| a forced pick is made; otherwise "
            "the next kind of M (order xover -> dls -> sobol) is filled by its ordinary slot pick (forced_fill, "
            "same family / fallback rules). k = 5 with qfirst: qfirst + 1 to 4 forced picks + the missing fills; "
            "k = 4 with qfirst: qfirst + the xover, dls and sobol fills (no forced pick). k < 4: forced logic "
            "unchanged. No llm slot in forced rounds (agent-gap rows stay in the forced set)"
        ),
        "acquisition_tag": "bo-acq-v3 (crossover_v2 pool)",
        "recorded_at": now,
        "note": "user 2026-10-21; run identity unchanged (as budget_extension / analyst); every BO round >= 1 "
        "interleaves crossover, DLS and Sobol with the LLM analyst applied; "
        "docs/trimem_pool_profiling_plan_v0.1.0.md addendum channel-slots-v2",
    }


def _channel_slots_v3_block(state: dict, now: str, before: dict | None) -> dict:
    """The state.json record of channel-slots-v3 (user 2026-10-22); ``before`` = the block it replaces (kept
    verbatim: rounds < from_round were proposed under it)."""
    return {
        "protocol": CHANNEL_SLOTS_V3,
        "from_round": max(CHANNEL_SLOTS_FROM_ROUND, len(state["rounds"])),
        "crossover": {
            **XOVER_SETTINGS,
            "graft_sizes": list(pool.XOVER_SIZES),
            "rule": "unchanged from channel-slots-v1 (session_pool.crossover_v2)",
        },
        "dls_crossover": "unchanged from channel-slots-v1",
        "slot_order": [SLOT_LABELS.get(k, k) for k in SLOT_LAYOUT_V3],
        "small_batch_priority": [SLOT_LABELS.get(k, k) for k in SLOT_PRIORITY_V3]
        + ["hv"],
        "slots": {SLOT_LABELS.get(k, k): v for k, v in SLOT_RULES_V3.items()},
        "cost_first": {
            "rule": pool.COST_FIRST_RULE,
            "z": pool.COST_FIRST_Z,
            "tie_break_beta": pool.COST_FIRST_BETA,
            "candidates": "the whole pool of the slot's suggest call (Sobol breadth, analyst rows, crossover rows, "
            "every DLS start's endpoint), measured / already-picked configurations excluded",
        },
        "backfill": "a channel slot whose family is empty (e.g. analyst off / error, every row excluded) takes the "
        "HV-best remaining candidate of the union pool (session_pool.suggest slot_fallback); the pick records "
        "channel hv + backfill {slot, reason} and is logged",
        "believer": "kriging believer in slot order " + " -> ".join(SLOT_LAYOUT_V3),
        "seeds": {
            "qfirst": "round seed + 500",
            **{
                k: f"round seed + {SLOT_SEED_OFFSETS[k]} (+ i for the i-th repeat)"
                for k in ("tokfirst", "latfirst", "llm", "xover", "dls", "sobol")
            },
            "hv": "round seed + j",
            "forced": "round seed + j (j = 0, 1, ... over the forced picks)",
        },
        "forced": "stall-triggered rounds: unchanged channel-slots-v2 forced rule (qfirst, forced picks by "
        "gap-landing probability = RAG-Stack rule, xover / dls / sobol coverage fills); no cost-first slots",
        "coverage_log": "per pick: predicted delta vs the measured front's per-axis best (gain; ln tokens; ln "
        "latency; positive = improves) in units of the measured per-axis range; flagged when an objective has no "
        "improving pick in the round (round record coverage)",
        "acquisition_tag": "bo-acq-v3 (crossover_v2 pool)",
        "before": before,
        "opt_in": "POOL_CHANNEL_SLOTS=v3 (or --channel_slots v3)",
        "recorded_at": now,
        "note": "user 2026-10-22; run identity unchanged; HV reference = whichever rule state.json hv_ref selects",
    }


def recorded_sobol_range(state: dict) -> tuple[int, int] | None:
    """(first, last) sobol-only-v1 round of the recorded channel-slot chain (None without one)."""
    blk = state.get("channel_slots")
    while blk:
        b = blk.get("before")
        if b and b["protocol"] == CHANNEL_SLOTS_SOBOL:
            return b["from_round"], blk["from_round"] - 1
        blk = b
    return None


def recorded_sobol_until(state: dict) -> int | None:
    """The last sobol-only-v1 round of the recorded channel-slot chain (None without one)."""
    blk = state.get("channel_slots")
    while blk:
        if blk.get("before") and blk["before"]["protocol"] == CHANNEL_SLOTS_SOBOL:
            return blk["from_round"] - 1
        blk = blk.get("before")
    return None


def _sobol_then_v4(
    state: dict, now: str, protocol: str, rec: str | None, until: int
) -> dict:
    """sobol-only-v1 for rounds [next round, ``until``], channel-slots-v4 from ``until`` + 1 (user 2026-10-22,
    ScienceWorld: round 1 random sampling, v4 from round 2). Recorded once; a later start must ask for the same
    schedule (or nothing: the recorded chain stays)."""
    if protocol not in CHANNEL_SLOTS_TARGETS:
        raise SystemExit(
            "random-sampling rounds (sobol_until) are only defined with channel-slots-v4 / -v5"
        )
    if rec == protocol:
        rng = recorded_sobol_range(state)
        if (
            rng is not None
            and rng[1] != until
            and rng[0] >= max(CHANNEL_SLOTS_FROM_ROUND, len(state["rounds"]))
        ):
            # unused chain (user 2026-10-22, ScienceWorld: 1 -> 2 random rounds before round 1 was proposed): no
            # round was proposed under the recorded sobol-only / target blocks, so re-record them from the block
            # they replaced; the unused chain is kept in ``replaced_unused`` (as batch_schedule)
            cur = state["channel_slots"]
            sob = cur["before"]
            state["channel_slots"] = sob["before"]
            try:
                new = _sobol_then_v4(
                    state,
                    now,
                    protocol,
                    (sob["before"] or {}).get("protocol"),
                    until,
                )
            except BaseException:
                state["channel_slots"] = cur
                raise
            unused = {k: x for k, x in cur.items() if k != "replaced_unused"}
            new["replaced_unused"] = [*cur.get("replaced_unused", []), unused]
            return new
        if recorded_sobol_until(state) != until:
            raise SystemExit(
                f"state.json records {rec} with sobol-only rounds until {recorded_sobol_until(state)}; "
                f"{until} requested; refusing to mix"
            )
        return state["channel_slots"]
    if (
        rec in CHANNEL_SLOTS_TARGETS
        and recorded_sobol_until(state) == until
        and until < max(CHANNEL_SLOTS_FROM_ROUND, len(state["rounds"]))
    ):
        # the recorded random rounds are in the past (user 2026-10-22, ScienceWorld v5 -> v6 from round 3): an
        # explicit upgrade along CHANNEL_SLOTS_TARGETS from the next round to be proposed; the recorded chain
        # (random rounds, the replaced target block) stays as ``before``; downgrades refused
        if CHANNEL_SLOTS_TARGETS.index(protocol) < CHANNEL_SLOTS_TARGETS.index(rec):
            raise SystemExit(
                f"state.json records {rec}; {protocol} requested; refusing to downgrade"
            )
        state["channel_slots"] = _target_block(
            protocol, state, now, state["channel_slots"]
        )
        return state["channel_slots"]
    nxt = max(CHANNEL_SLOTS_FROM_ROUND, len(state["rounds"]))
    if until < nxt:
        raise SystemExit(
            f"sobol-only rounds until {until} requested, but the next round to be proposed is {nxt}; refusing"
        )
    before = state.get("channel_slots")
    if before is not None and before["protocol"] not in CHANNEL_SLOTS_PROTOCOLS:
        raise SystemExit(
            "state.json has another channel-slot protocol; refusing to mix"
        )
    sob = {
        "protocol": CHANNEL_SLOTS_SOBOL,
        "from_round": nxt,
        "until_round": until,
        "crossover": {
            **XOVER_SETTINGS,
            "graft_sizes": list(pool.XOVER_SIZES),
            "rule": "unused (no xover slot); recorded for the settings shape",
        },
        "slot_order": "sobol-slot x batch",
        "slots": {
            "sobol-slot": SLOT_RULES_SOBOL["sobol"],
            "hv": SLOT_RULES_SOBOL["hv"],
        },
        "believer": "kriging believer over the successive sobol picks (+ exclusion of measured / picked "
        "configurations): the picks differ",
        "seeds": {
            "sobol": f"round seed + {SLOT_SEED_OFFSETS['sobol']} + i for the i-th pick"
        },
        "qfirst": "not used",
        "forced": "never forced (random-sampling round)",
        "backfill": "an empty Sobol family takes the HV-best remaining candidate of the union pool (recorded)",
        "before": before,
        "recorded_at": now,
        "note": "user 2026-10-22 (ScienceWorld): random sampling only for these rounds, then channel-slots-v4",
    }
    v4 = _target_block(protocol, state, now, sob)
    v4["from_round"] = until + 1
    v4["opt_in"] += (
        f"; sobol-only-v1 rounds {nxt}..{until} first (POOL_RANDOM_ROUNDS_UNTIL={until})"
    )
    state["channel_slots"] = v4
    return v4


def _target_block(protocol: str, state: dict, now: str, before: dict | None) -> dict:
    if protocol == CHANNEL_SLOTS_V6:
        return _channel_slots_v6_block(state, now, before)
    if protocol == CHANNEL_SLOTS_V5:
        return _channel_slots_v5_block(state, now, before)
    return _channel_slots_v4_block(state, now, before)


def _channel_slots_v5_block(state: dict, now: str, before: dict | None) -> dict:
    """The state.json record of channel-slots-v5 (user 2026-10-22, ALFWorld v7 from round 7, batch 4)."""
    blk = _channel_slots_v4_block(state, now, before)
    return {
        **blk,
        "protocol": CHANNEL_SLOTS_V5,
        "slot_order": [SLOT_LABELS[k] for k in SLOT_LAYOUT_V5],
        "small_batch_priority": [SLOT_LABELS[k] for k in SLOT_LAYOUT_V5] + ["hv"],
        "slots": {SLOT_LABELS[k]: SLOT_RULES_V4[k] for k in (*SLOT_LAYOUT_V5, "hv")},
        "believer": "kriging believer in slot order " + " -> ".join(SLOT_LAYOUT_V5),
        "seeds": {
            **{k: f"round seed + {SLOT_SEED_OFFSETS[k]}" for k in SLOT_LAYOUT_V5},
            "hv": "round seed + j",
        },
        "budget_tail": "a round capped below 4 by the budget keeps the first k of slot_order",
        "forced": "never stall-forced (one pick per channel every round)",
        "analyst": "called once per round (the llm slot's rows)",
        "opt_in": "POOL_CHANNEL_SLOTS=v5 (or --channel_slots v5); usually with POOL_BATCH_STEPS (batch 4)",
        "note": "user 2026-10-22: from round 7 four configurations per round, one per channel (llm, dls, xover, "
        "sobol); run identity unchanged",
    }


def _channel_slots_v6_block(state: dict, now: str, before: dict | None) -> dict:
    """The state.json record of channel-slots-v6 (user 2026-10-22, ScienceWorld from round 3, batch 8)."""
    blk = _channel_slots_v5_block(state, now, before)
    layout = [SLOT_LABELS[k] for k in SLOT_LAYOUT_V6]
    rules = SLOT_RULES[CHANNEL_SLOTS_V6]
    return {
        **blk,
        "protocol": CHANNEL_SLOTS_V6,
        "slot_order": layout,
        "small_batch_priority": layout + ["hv"],
        "slots": {SLOT_LABELS[k]: rules[k] for k in (*SLOT_LAYOUT_V5, "hv")},
        "believer": "kriging believer in slot order " + " -> ".join(SLOT_LAYOUT_V6),
        "seeds": {
            **{k: f"round seed + {SLOT_SEED_OFFSETS[k]}" for k in SLOT_LAYOUT_V5},
            "hv": "round seed + j (j = 0..3 over the round's hv picks)",
        },
        "budget_tail": "a round capped below 8 by the budget / schedule keeps the first k of slot_order "
        "(k = 2: llm, dls)",
        "forced": "never stall-forced (one pick per channel + 4 union picks every round)",
        "analyst": "called once per round (the llm slot's rows; also in the union of the hv picks)",
        "opt_in": "POOL_CHANNEL_SLOTS=v6 (or --channel_slots v6); usually with POOL_BATCH_STEPS (batch 8); an "
        "upgrade from v5 applies from the next round to be proposed (v5 kept as ``before``); downgrades refused",
        "note": "user 2026-10-22 (ScienceWorld): from round 3 eight configurations per round = the best of each "
        "channel's pool (llm, dls, xover, sobol; v5 rules) + the top-4 remaining by qLogNEHVI over the union of all "
        "channels' candidates; run identity unchanged",
    }


def _channel_slots_v4_block(state: dict, now: str, before: dict | None) -> dict:
    """The state.json record of channel-slots-v4 (user 2026-10-22): channel-slots-v3 with the qfirst / token-first
    / latency-first quotas replaced by Sobol slots; ``before`` = the block it replaces (kept verbatim)."""
    blk = _channel_slots_v3_block(state, now, before)
    return {
        **blk,
        "protocol": CHANNEL_SLOTS_V4,
        "slot_order": [SLOT_LABELS.get(k, k) for k in SLOT_LAYOUT_V4],
        "small_batch_priority": [SLOT_LABELS.get(k, k) for k in SLOT_PRIORITY_V4]
        + ["hv"],
        "slots": {SLOT_LABELS.get(k, k): v for k, v in SLOT_RULES_V4.items()},
        "cost_first": None,
        "qfirst": "not used: no qfirst pick in channel-slots-v4 rounds (ordinary or forced), although state.json "
        "qfirst_slot stays recorded for the earlier rounds",
        "believer": "kriging believer in slot order " + " -> ".join(SLOT_LAYOUT_V4),
        "seeds": {
            **{
                k: f"round seed + {SLOT_SEED_OFFSETS[k]} (+ i for the i-th repeat)"
                for k in ("sobol", "llm", "xover", "dls")
            },
            "hv": "round seed + j",
            "forced": "round seed + j (j = 0, 1, ... over the forced picks)",
        },
        "budget_tail": "a round capped below 10 by the budget keeps the first k of small_batch_priority "
        "(llm, xover, dls, sobol, sobol, llm, xover, dls, sobol, sobol), laid out in slot_order order",
        "forced": "stall-triggered rounds: unchanged channel-slots-v2/-v3 forced rule (forced picks by gap-landing "
        "probability = RAG-Stack rule, xover / dls / sobol coverage fills) WITHOUT the qfirst pick; no cost-first "
        "slots",
        "opt_in": "POOL_CHANNEL_SLOTS=v4 (or --channel_slots v4); downgrade to v3 / v2 refused",
        "note": "user 2026-10-22; channel-slots-v3 minus qfirst / token-first / latency-first, plus 3 Sobol slots; "
        "run identity unchanged; HV reference = whichever rule state.json hv_ref selects",
    }


def channel_slots_settings(state: dict, r: int) -> dict | None:
    """Settings of round ``r``'s channel slots (None before ``from_round`` or without the block). A v3 block
    answers rounds before its ``from_round`` from the block it replaced (``before``)."""
    blk = state.get("channel_slots")
    while blk and r < blk["from_round"] and blk.get("before"):
        blk = blk["before"]
    if not blk or r < blk["from_round"]:
        return None
    x = blk["crossover"]
    return {
        "protocol": blk["protocol"],
        "xover": {
            "top": x["top"],
            "sizes": tuple(x["sizes"]),
            "max_total": x["max_total"],
            "z": x["z"],
        },
    }


def _slot_pick(
    believed, ctx, slots, kind, sd, exclude, agent_rows, aux=None, search_options=None
) -> dict:
    """One ordinary slot pick of ``kind`` (xover / sobol / llm / dls / tokfirst / latfirst, or hv = the whole
    pool)."""
    requested_kind = kind
    policy = (search_options or {}).get("ablation_policy")
    if policy is not None:
        from trimem.experiments.scienceworld_trace_ablation import effective_slot

        kind = effective_slot(kind, policy)
    res = pool.suggest(
        believed,
        ctx,
        seed=sd,
        exclude=exclude,
        acq_mode="hv",
        agent_rows=agent_rows,
        xover=slots["xover"],
        dls_crossover=True,
        slot=None if kind == "hv" else kind,
        aux=aux,
        **(search_options or {}),
    )
    ch = res["chosen"]
    cfg = space.validate(ch["config"])
    if not pool.proposable(cfg):
        raise AssertionError(f"proposal violates {pool.propose_fixed()}: {cfg}")
    info = res["slot_info"]
    fallback = bool(info.get("slot_fallback"))
    v3 = slots["protocol"] in CHANNEL_SLOTS_NAMED
    rules = SLOT_RULES.get(slots["protocol"], SLOT_RULES_V3)
    return {
        "config": cfg,
        "channel": "hv" if fallback else SLOT_LABELS[kind],
        **(
            {
                "slot_name": SLOT_LABELS[kind],
                "slot_rule": rules["hv" if fallback else kind],
            }
            if v3
            else {}
        ),
        **(
            {
                "backfill": {
                    "slot": SLOT_LABELS[kind],
                    "reason": info["slot_fallback"],
                    "rule": "HV-best remaining candidate over the union pool",
                }
            }
            if v3 and fallback
            else {}
        ),
        **(
            {"backfill": {"slot": SLOT_LABELS[requested_kind],
                          "reason": "disabled ablation channel",
                          "rule": "global coverage (sobol slot)"}}
            if requested_kind != kind else {}
        ),
        "source_channel": info.get("source_channel", ch["channel"]),
        "slot": kind,
        "sources": res["chosen_sources"],
        "acq": ch["acq"],
        "mode": res["mode"],
        "pred_gain": ch["pred_quality_per_iter"],
        "pred_tokens": ch["pred_tokens_per_iter"],
        "pred_latency_ms": ch["pred_latency_ms_per_iter"],
        "pred_sd": ch["sd"],
        "acquisition": res["acquisition"],
        "pool_size": res["pool_size"],
        "channel_counts": res["channel_counts"],
        "source_counts": res["source_counts"],
        "plan_counts": res.get("plan_counts"),
        **({"search_audit": res["search_audit"]} if "search_audit" in res else {}),
        "slot_info": info,
        "seed": sd,
        "protocol": slots["protocol"],
        **({"crossover_rows": len(res["crossover"])} if kind == "xover" else {}),
    }


def _slot_picks(
    believed,
    ctx,
    slots,
    n,
    seed,
    exclude,
    agent_rows,
    believe,
    aux=None,
    search_options=None,
) -> list[dict]:
    """The non-qfirst slots of an ordinary channel-slots round (see ``propose_batch``); ``believe`` appends the
    pseudo-observation of each pick to ``believed`` (kriging believer)."""
    out, hv_j, rep = [], 0, {}
    # the qfirst slot (if any) is already picked
    for kind in slot_plan(n, False, slots["protocol"]):
        if kind == "hv":
            sd = seed + hv_j
            hv_j += 1
        else:  # channel-slots-v3 repeats a kind: the i-th repeat uses offset + i (v1 / v2: i = 0 always)
            sd = seed + SLOT_SEED_OFFSETS[kind] + rep.get(kind, 0)
            rep[kind] = rep.get(kind, 0) + 1
        p = _slot_pick(
            believed, ctx, slots, kind, sd, exclude, agent_rows, aux, search_options
        )
        out.append(p)
        exclude.add(space.key(p["config"]))
        believe(
            f"{kind}-{len(out)}",
            p["config"],
            p["pred_gain"],
            p["pred_tokens"],
            p["pred_latency_ms"],
        )
    return out


def _named(slots: dict, picks: list[dict]) -> list[dict]:
    """channel-slots-v3 / -v4: every pick records its slot name and rule (qfirst / forced picks included)."""
    if slots["protocol"] not in CHANNEL_SLOTS_NAMED:
        return picks
    for p in picks:
        if p.get("channel") == "qfirst":
            p.setdefault("slot_name", "qfirst")
            p.setdefault("slot_rule", SLOT_RULES_V3["qfirst"])
        elif p.get("slot") == "forced":
            p.setdefault("slot_name", "forced")
            p.setdefault(
                "slot_rule",
                "stall round: gap-landing probability over the forced set (RAG-Stack rule, unchanged)",
            )
        elif p.get("forced_fill"):
            p["slot_name"] = f"forced-fill:{p['slot_name']}"
    return picks


def pick_kinds(sources: list[str]) -> set[str]:
    """The FORCED_COVER kinds a pick represents, from its producing channels (multi-label sources)."""
    out = set()
    if any(s.startswith("crossover") or s == "dls-crossover" for s in sources):
        out.add("xover")
    if any(s in pool.DLS_SLOT_LABELS for s in sources):
        out.add("dls")
    if any(s in ("sobol", "lhs") for s in sources):
        out.add("sobol")
    return out


def _forced_v2_picks(
    points,
    believed,
    ctx,
    slots,
    k,
    n,
    seed,
    exclude,
    agent_rows,
    realized_front,
    believe,
    aux=None,
    search_options=None,
) -> list[dict]:
    """The non-qfirst picks of a forced (stall) channel-slots-v2 round of total batch ``k`` (``n`` slots left):
    forced picks as before, except that for k >= FORCED_COVER_MIN_BATCH the trailing slots are reserved so that
    xover, dls and sobol each contribute >= 1 pick (rule: state.json channel_slots.forced)."""
    need = list(FORCED_COVER) if k >= FORCED_COVER_MIN_BATCH else []
    front = realized_front or realized_front_2d(points)
    out, tried, j = [], set(), 0
    while len(out) < n:
        covered = set().union(*(pick_kinds(p.get("sources", [])) for p in out))
        missing = [x for x in need if x not in covered and x not in tried]
        if n - len(out) > len(missing):
            res = pool.suggest(
                believed,
                ctx,
                seed=seed + j,
                exclude=exclude,
                acq_mode="hv",
                forced=True,
                agent_rows=agent_rows,
                realized_front=front,
                xover=slots["xover"],
                dls_crossover=True,
                aux=aux,
                **(search_options or {}),
            )
            j += 1
            ch = res["chosen"]
            cfg = space.validate(ch["config"])
            if not pool.proposable(cfg):
                raise AssertionError(f"proposal violates {pool.propose_fixed()}: {cfg}")
            p = {
                "config": cfg,
                "channel": ch["channel"],
                "source_channel": ch["channel"],
                "slot": "forced",
                "sources": res["chosen_sources"],
                "acq": ch["acq"],
                "mode": res["mode"],
                "pred_gain": ch["pred_quality_per_iter"],
                "pred_tokens": ch["pred_tokens_per_iter"],
                "pred_latency_ms": ch["pred_latency_ms_per_iter"],
                "acquisition": res["acquisition"],
                "pool_size": res["pool_size"],
                "channel_counts": res["channel_counts"],
                "source_counts": res["source_counts"],
                "plan_counts": res.get("plan_counts"),
                **(
                    {"search_audit": res["search_audit"]}
                    if "search_audit" in res
                    else {}
                ),
                "seed": seed + j - 1,
                "protocol": slots["protocol"],
            }
        else:
            kind = missing[0]
            tried.add(kind)
            p = _slot_pick(
                believed,
                ctx,
                slots,
                kind,
                seed + SLOT_SEED_OFFSETS[kind],
                exclude,
                agent_rows,
                aux,
                search_options,
            )
            p["forced_fill"] = True
        out.append(p)
        exclude.add(space.key(p["config"]))
        believe(
            f"forced-{len(out)}",
            p["config"],
            p["pred_gain"],
            p["pred_tokens"],
            p["pred_latency_ms"],
        )
    return out


def slot_log_lines(r: int, picks: list[dict]) -> list[str]:
    """Human-readable lines of a channel-slots round: every pick's channel / source / predictions, and the pool
    composition (count per channel) of the first whole-pool slot."""
    out = []
    for p in picks:
        if "slot" not in p or p.get("channel") == "qfirst":
            continue
        out.append(
            f"round {r} {p.get('slot_name', p['channel'])}"
            + (f" [channel {p['channel']}]" if "slot_name" in p else "")
            + f" (source {p.get('source_channel')}, {p['mode']}): {space.key(p['config'])} "
            f"pred gain {p['pred_gain']:+.3f}, {p['pred_tokens']:,.0f} tok, {p['pred_latency_ms'] / 1000:.1f} s, "
            f"acq {p['acq']:.4g}"
            + (
                f", {p['slot_info'].get('candidates')} slot candidates"
                if p.get("slot_info")
                else ""
            )
            + (" [forced fill]" if p.get("forced_fill") else "")
            + (
                f" [BACKFILL {p['backfill']['slot']}: {p['backfill']['rule']}]"
                if p.get("backfill")
                else ""
            )
            + (
                f" [rule {p['slot_info']['rule']}, {p['slot_info']['tolerant']}/"
                f"{p['slot_info']['candidates']} tolerant, gain >= {p['slot_info']['threshold_gain']:+.3f}]"
                if (p.get("slot_info") or {}).get("threshold_gain") is not None
                else ""
            )
            + (
                f" [FALLBACK {p['slot']}-slot -> {p['slot_info']['slot_fallback']}"
                + (
                    f"; analyst rows {p['slot_info']['analyst_rows']}"
                    if "analyst_rows" in p["slot_info"]
                    else ""
                )
                + "]"
                if (p.get("slot_info") or {}).get("slot_fallback")
                else ""
            )
        )
    full = next((p for p in picks if p.get("slot") in ("hv", "forced")), None)
    full = full or next((p for p in picks if "channel_counts" in p), None)
    if full:
        out.append(
            f"round {r} pool composition ({full['pool_size']} rows, first label): "
            + json.dumps(full["channel_counts"], sort_keys=True)
            + (
                "; by source (multi-label): "
                + json.dumps(full["source_counts"], sort_keys=True)
                if "source_counts" in full
                else ""
            )
            + (
                "; by plan: " + json.dumps(full["plan_counts"], sort_keys=True)
                if full.get("plan_counts")
                else ""
            )
        )
    return out


COVERAGE_AXES = ("gain", "ln_tokens", "ln_latency")


def round_coverage(picks: list[dict], points: list[hb.Point]) -> dict:
    """channel-slots-v3 coverage audit of a proposed round: each pick's predicted delta vs the current measured
    front on every objective, in units of that objective's measured range (spread). The front's per-axis best =
    max measured gain / min measured ln tokens / min measured ln latency (each axis's extreme is a front point);
    delta > 0 = the pick is predicted to improve the front on that axis. ``flag_no_improvement`` lists the
    objectives that got zero improving picks. Attaches ``coverage`` (the three deltas) to every pick."""
    real = [p for p in points if not str(p.point_id).startswith("believer-")]
    vals = [(p.quality, math.log(p.tokens), math.log(p.latency_ms)) for p in real]
    best = (
        max(v[0] for v in vals),
        min(v[1] for v in vals),
        min(v[2] for v in vals),
    )
    spread = tuple(
        max(max(v[i] for v in vals) - min(v[i] for v in vals), 1e-12) for i in range(3)
    )
    rows = []
    for p in picks:
        pred = (
            p["pred_gain"],
            math.log(p["pred_tokens"]),
            math.log(p["pred_latency_ms"]),
        )
        d = (
            (pred[0] - best[0]) / spread[0],
            (best[1] - pred[1]) / spread[1],
            (best[2] - pred[2]) / spread[2],
        )
        p["coverage"] = dict(zip(COVERAGE_AXES, d, strict=True))
        rows.append(
            {
                "slot": p.get("slot_name", p.get("channel")),
                "key": space.key(p["config"]),
                **p["coverage"],
            }
        )
    improving = {a: sum(r[a] > 0 for r in rows) for a in COVERAGE_AXES}
    return {
        "rule": "delta vs the measured front's per-axis best, in units of the measured per-axis range; > 0 = "
        "improves",
        "front_best": dict(zip(COVERAGE_AXES, best, strict=True)),
        "spread": dict(zip(COVERAGE_AXES, spread, strict=True)),
        "picks": rows,
        "improving_picks": improving,
        "flag_no_improvement": [a for a in COVERAGE_AXES if improving[a] == 0],
    }


def coverage_summary(picks: list[dict]) -> dict:
    """Round-level record of the per-pick ``coverage`` deltas (``round_coverage``): improving picks per objective
    and the objectives without any (state.json round ``coverage``)."""
    rows = [p["coverage"] for p in picks if "coverage" in p]
    improving = {a: sum(r[a] > 0 for r in rows) for a in COVERAGE_AXES}
    return {
        "rule": "per-pick coverage = delta vs the measured front's per-axis best, in units of the measured "
        "per-axis range; > 0 = improves",
        "improving_picks": improving,
        "flag_no_improvement": [a for a in COVERAGE_AXES if improving[a] == 0],
    }


def coverage_log_lines(r: int, cov: dict) -> list[str]:
    """The per-round coverage line(s) of ``round_coverage``."""
    parts = [
        f"[{x['slot']}] dgain {x['gain']:+.2f} dlntok {x['ln_tokens']:+.2f} dlnlat {x['ln_latency']:+.2f}"
        for x in cov["picks"]
    ]
    out = [
        f"round {r} coverage (delta vs front best, x axis range; + = improves): "
        + "; ".join(parts)
    ]
    out.append(
        f"round {r} coverage improving picks: "
        + json.dumps(cov["improving_picks"], sort_keys=True)
        + (
            f" -- FLAG: no improving pick on {cov['flag_no_improvement']}"
            if cov["flag_no_improvement"]
            else ""
        )
    )
    return out


# ------------------------------------------------------------------ auxiliary observations (aux-obs-v1, 2026-10-20)

AUX_PROTOCOL = "aux-obs-v1"
AUX_COST_SD_MIN = 0.5  # ln-scale SD floor of auxiliary token / latency observations (incomparable protocols)


def aux_block(state: dict, now: str, build, min_round: int = 1) -> dict | None:
    """Auxiliary observations, recorded once in state.json (``state["aux_observations"]``; ``build()`` returns
    the block body, see ``trimem.experiments.aux_observations``): from_round = max(min_round, rounds already
    proposed) -- never retroactive. Not part of the run identity. Never fatal: a failing ``build`` records nothing
    (returns None; the run proceeds without auxiliary points and retries at the next start)."""
    if "aux_observations" not in state:
        try:
            body = build()
        except (Exception, SystemExit) as e:  # noqa: BLE001 - optional GP input
            print(
                f"aux observations unavailable ({type(e).__name__}: {e}); none recorded",
                flush=True,
            )
            return None
        state["aux_observations"] = {
            "protocol": AUX_PROTOCOL,
            "from_round": max(min_round, len(state["rounds"])),
            **body,
            "recorded_at": now,
        }
    elif state["aux_observations"]["protocol"] != AUX_PROTOCOL:
        raise SystemExit(
            "state.json has another aux-observation protocol; refusing to mix"
        )
    return state["aux_observations"]


def aux_points(state: dict, r: int, task: str) -> list[hb.Point]:
    """Round ``r``'s auxiliary GP points (empty before ``from_round`` or without the block); also switches the
    engine's source dimension on / off for the round (``session_pool.use_aux``)."""
    blk = state.get("aux_observations")
    on = bool(blk) and r >= blk["from_round"] and bool(blk["points"])
    pool.use_aux(on)
    if not on:
        return []
    out = []
    for i, a in enumerate(blk["points"]):
        out.append(
            hb.Point(
                f"{pool.AUX_PREFIX}{a['experiment']}:{i}",
                a["experiment"],
                task,
                a["config"],
                0,
                0,
                1,
                a["gain"],
                a["tokens"],
                a["latency_ms"],
                tuple(a["noise"]),
            )
        )
    return out


# ------------------------------------------------------------------ quality-first slot (user 2026-10-20)

QFIRST_PROTOCOL = "qfirst-slot-v1"
QFIRST_FROM_ROUND = (
    1  # user 2026-10-21: every BO round >= 1 reserves the qfirst slot (was 3)
)
QFIRST_BETA = 1.5  # = session_pool.suggest default
QFIRST_TARGET = "gain"  # coordinator 2026-10-20: "improve" is degenerate on panel-aggregated pool points
QFIRST_BUDGET_SLACK = 1.1  # budget = max observed (tokens, latency) x 1.1
QFIRST_SEED_OFFSET = 500  # qfirst seed = round seed + 500 (hv slots keep seed + j)
QFIRST_BUDGET_RULE = (
    "max observed per-configuration mean tokens and latency over all BO observations x 1.1 (a feasibility "
    "guard only: the slot is effectively UCB(quality) and may explore around the quality champion)"
)


def qfirst_budget(
    points: list[hb.Point], slack: float = QFIRST_BUDGET_SLACK
) -> tuple[float, float]:
    """Loose cost budget (tokens, latency_ms) of the quality-first slot: max observed x ``slack``."""
    return (
        slack * max(p.tokens for p in points),
        slack * max(p.latency_ms for p in points),
    )


def quality_champion(points: list[hb.Point]) -> hb.Point:
    """Measured point with the highest quality (ties: lowest key)."""
    return min(points, key=lambda p: (-p.quality, space.key(p.cfg)))


def qfirst_block(state: dict, now: str) -> dict:
    """The quality-first slot protocol, recorded once in state.json (``state["qfirst_slot"]``):
    from_round = max(QFIRST_FROM_ROUND, rounds already proposed) -- never retroactive."""
    if "qfirst_slot" not in state:
        state["qfirst_slot"] = {
            "protocol": QFIRST_PROTOCOL,
            "from_round": max(QFIRST_FROM_ROUND, len(state["rounds"])),
            "slots": 1,
            "acq_mode": "qfirst",
            "beta": QFIRST_BETA,
            "quality_target": QFIRST_TARGET,
            "budget_rule": QFIRST_BUDGET_RULE,
            "budget_slack": QFIRST_BUDGET_SLACK,
            "seed": f"round seed + {QFIRST_SEED_OFFSET}",
            "recorded_at": now,
            "note": "user 2026-10-20: 1 of the batch slots = UCB_beta(quality) x P(tokens, latency within "
            "budget), picked first from the same candidate pool (quality champion added as a DLS start), "
            "proposal-side constraint honoured; the kriging believer conditions on it before the hv picks",
        }
    elif state["qfirst_slot"]["protocol"] != QFIRST_PROTOCOL:
        raise SystemExit(
            "state.json has another quality-first slot protocol; refusing to mix"
        )
    return state["qfirst_slot"]


def qfirst_settings(state: dict, r: int, points: list[hb.Point]) -> dict | None:
    """Settings of round ``r``'s quality-first slot (None before ``from_round`` or without the block)."""
    blk = state.get("qfirst_slot")
    if not blk or r < blk["from_round"]:
        return None
    return {
        "beta": blk["beta"],
        "budget": qfirst_budget(points, blk["budget_slack"]),
        "quality_target": blk["quality_target"],
        "protocol": blk["protocol"],
    }


def _qfirst_pick(
    points, ctx, qf, seed, exclude, agent_rows, aux=(), search_options=None
) -> dict:
    champ = quality_champion(points)
    budget = tuple(qf["budget"])
    res = pool.suggest(
        points,
        ctx,
        seed=seed + QFIRST_SEED_OFFSET,
        exclude=exclude,
        acq_mode="qfirst",
        beta=qf["beta"],
        budget=budget,
        fitted=pool.fit(points + list(aux), quality_target=qf["quality_target"]),
        agent_rows=agent_rows,
        dls_extra=[(champ.cfg, "dls-qchamp")],
        aux=list(aux),
        **(search_options or {}),
    )
    ch = res["chosen"]
    cfg = space.validate(ch["config"])
    if not pool.proposable(cfg):
        raise AssertionError(f"proposal violates {pool.propose_fixed()}: {cfg}")
    if qf["quality_target"] == "gain":
        gain = ch["pred_quality_per_iter"]
    else:  # the believer needs the gain-scale posterior mean
        m, _ = pool.panel_posterior(pool.fit(points + list(aux))[0], ctx, [cfg])
        gain = float(m[0, 0])
    return {
        "config": cfg,
        "channel": "qfirst",
        "source_channel": ch["channel"],
        "acq": ch["acq"],
        "mode": res["mode"],
        "pred_gain": gain,
        "pred_tokens": ch["pred_tokens_per_iter"],
        "pred_latency_ms": ch["pred_latency_ms_per_iter"],
        "acquisition": res["acquisition"],
        "pool_size": res["pool_size"],
        "plan_counts": res.get("plan_counts"),
        **({"search_audit": res["search_audit"]} if "search_audit" in res else {}),
        "qfirst": {
            "protocol": qf["protocol"],
            "beta": qf["beta"],
            "budget_tokens": budget[0],
            "budget_latency_ms": budget[1],
            "quality_target": qf["quality_target"],
            "champion": champ.cfg,
            "champion_gain": champ.quality,
            "seed": seed + QFIRST_SEED_OFFSET,
            "top": [
                {
                    "config": t["config"],
                    "channel": t["channel"],
                    "acq": t["acq"],
                    "pred_gain": t["pred_quality_per_iter"],
                    "pred_tokens": t["pred_tokens_per_iter"],
                    "pred_latency_ms": t["pred_latency_ms_per_iter"],
                }
                for t in res["top"][:5]
            ],
        },
    }


# ------------------------------------------------------------------ GP protocol (gp-fix-v1, 2026-10-20)


def gp_block(state: dict, now: str, noise: str) -> dict:
    """The GP protocol (``session_pool.GP_PROTOCOL``), recorded once in state.json (``state["gp"]``):
    from_round = the next round to be proposed -- never retroactive (earlier rounds stay as proposed under
    gp-v0). ``noise`` describes the driver's noise model."""
    if "gp" not in state:
        state["gp"] = {
            "protocol": pool.GP_PROTOCOL,
            "from_round": len(state["rounds"]),
            "before": pool.GP_LEGACY,
            "outputscale_prior": {
                "family": "Gamma(concentration, rate) on every ScaleKernel (standardized outcomes)",
                "value": list(pool.OUTPUTSCALE_PRIOR),
            },
            "noise": noise,
            "degenerate_ratio": pool.DEGENERATE_RATIO,
            "recorded_at": now,
            "note": "2026-10-20: the τ² -v5 quality GP collapsed to a constant under gp-v0 (outputscale ~0, every "
            "prediction +0.0354, SD ~0); docs/trimem_pool_profiling_plan_v0.1.0.md addendum gp-fix-v1",
        }
    elif state["gp"]["protocol"] != pool.GP_PROTOCOL:
        raise SystemExit("state.json has another GP protocol; refusing to mix")
    return state["gp"]


def use_gp_for(state: dict, r: int) -> bool:
    """Select round ``r``'s GP protocol (process-wide ``session_pool.use_gp``): gp-fix-v1 from the recorded
    ``from_round`` on, gp-v0 before it or without the block. Returns True when gp-fix-v1 is on."""
    blk = state.get("gp")
    on = bool(blk) and r >= blk["from_round"]
    pool.use_gp(pool.GP_PROTOCOL if on else pool.GP_LEGACY)
    return on


# ------------------------------------------------------------------ HV reference point (hv-ref-anchored-v1, 2026-10-22)

HV_REF_ENV = "POOL_HV_REF"  # opt-in at (re)start: POOL_HV_REF=anchored-v1
HV_REF_ENV_VALUES = {
    "anchored-v1": pool.HV_REF_ANCHORED,
    pool.HV_REF_ANCHORED: pool.HV_REF_ANCHORED,
}


def hv_ref_block(state: dict, now: str) -> dict | None:
    """The acquisition's HV reference-point rule (``session_pool.HV_REF_ANCHORED``), recorded once in state.json
    (``state["hv_ref"]``) when the run is (re)started with ``POOL_HV_REF=anchored-v1``: from_round = the next round
    to be proposed -- never retroactive. Once recorded it stays on (the env var may then be unset); a conflicting
    value (another protocol, or "0" to switch it off) is refused. Without the env var and the block: None (legacy
    rule)."""
    env = os.environ.get(HV_REF_ENV, "").strip()
    blk = state.get("hv_ref")
    if blk is not None:
        if blk.get("protocol") != pool.HV_REF_ANCHORED:
            raise SystemExit(
                f"state.json hv_ref {blk.get('protocol')!r} != {pool.HV_REF_ANCHORED}; refusing to mix"
            )
        if env and HV_REF_ENV_VALUES.get(env) != blk["protocol"]:
            raise SystemExit(
                f"{HV_REF_ENV}={env!r} conflicts with the recorded hv_ref {blk['protocol']} "
                f"(from round {blk['from_round']}); refusing to mix"
            )
        return blk
    if env in ("", "0"):
        return None
    if env not in HV_REF_ENV_VALUES:
        raise SystemExit(
            f"{HV_REF_ENV}={env!r}: unknown value (expected one of {sorted(HV_REF_ENV_VALUES)})"
        )
    state["hv_ref"] = {
        "protocol": pool.HV_REF_ANCHORED,
        "from_round": len(state["rounds"]),
        "before": pool.HV_REF_LEGACY,
        "opt_in": f"{HV_REF_ENV}={env}",
        "recorded_at": now,
        "rule": "acquisition HV reference point (qLogNEHVI / qLogEHVI, every hv / slot / forced pick): quality = "
        "NoMemory baseline (0 in gain units) - 1 SE of the NoMemory quality estimate; each cost (-ln tokens, "
        "-ln latency) = nadir of the non-dominated set of the posterior means of the measured configurations "
        f"(believer pseudo-points excluded) - {pool.HV_REF_MARGIN} x the front's range on that axis (zero range: "
        "ln 1.1); then strictly below every front point on every axis",
        "before_rule": "quality min(0, worst observed) - 1e-3; costs worst observed - ln 1.1",
        "reporting": "unchanged: reported HV (render_cost_saving_hv, report curves, fixed_reference) keeps its "
        "fixed definition",
        "reason": "user 2026-10-22: a meaningful quality / cost exchange rate (dominated cost outliers no longer "
        "dilute cost improvements)",
    }
    return state["hv_ref"]


def use_hv_ref_for(state: dict, r: int, baseline: tuple[float, float] | None) -> bool:
    """Select round ``r``'s HV reference rule (process-wide ``session_pool.use_hv_ref``): anchored from the recorded
    ``from_round`` on with ``baseline`` = (NoMemory quality in objective units, its SE), legacy before it or without
    the block. Returns True when the anchored rule is on."""
    blk = state.get("hv_ref")
    on = bool(blk) and r >= blk["from_round"]
    pool.use_hv_ref(
        pool.HV_REF_ANCHORED if on else pool.HV_REF_LEGACY, baseline if on else None
    )
    return on


# ------------------------------------------------------------------ reporting / selection


def objectives(p: hb.Point) -> tuple[float, float, float]:
    return (p.quality, -math.log(p.tokens), -math.log(p.latency_ms))


def pareto(points: list[hb.Point]) -> list[hb.Point]:
    ys = [objectives(p) for p in points]

    def dominated(i):
        return any(
            all(b >= a for a, b in zip(ys[i], ys[j], strict=True)) and ys[j] != ys[i]
            for j in range(len(ys))
            if j != i
        )

    return [p for i, p in enumerate(points) if not dominated(i)]


def hypervolume(points: list[hb.Point], ref: tuple[float, float, float]) -> float:
    """Exact 3-D hypervolume (maximisation) of the measured points above ``ref``."""
    import torch
    from botorch.utils.multi_objective.box_decompositions.dominated import (
        DominatedPartitioning,
    )

    ys = [
        objectives(p)
        for p in points
        if all(a > r for a, r in zip(objectives(p), ref, strict=True))
    ]
    if not ys:
        return 0.0
    part = DominatedPartitioning(
        ref_point=torch.tensor(ref, dtype=torch.double),
        Y=torch.tensor(ys, dtype=torch.double),
    )
    return float(part.compute_hypervolume())


def fixed_reference(points: list[hb.Point]) -> tuple[float, float, float]:
    """One reference point for every round's hypervolume (fixed after all profiling)."""
    ys = [objectives(p) for p in points]
    return (
        min(0.0, min(y[0] for y in ys)) - 1e-3,
        min(y[1] for y in ys) - math.log(1.1),
        min(y[2] for y in ys) - math.log(1.1),
    )


def select(points: list[hb.Point], noise_floor: float) -> dict:
    """Frozen choice from the TRAINING measurements: Q-best = highest measured mean gain;
    Knee = among configurations within ``noise_floor`` of Q-best's gain, the lowest
    ln tokens + ln latency."""
    cand = [p for p in points if p.cfg["plan"] != "NoMemory"]
    q = max(cand, key=lambda p: p.quality)
    near = [p for p in cand if p.quality >= q.quality - noise_floor]
    knee = min(near, key=lambda p: math.log(p.tokens) + math.log(p.latency_ms))
    return {
        "q_best": q,
        "knee": knee,
        "noise_floor": noise_floor,
        "near_q_best": len(near),
    }


def noise_floor(rep_a: list[float], rep_b: list[float]) -> float:
    """Noise floor of a panel-mean gain: the larger of |mean difference between two identical
    runs| and the standard error of that difference."""
    if len(rep_a) < 2 or len(rep_a) != len(rep_b):
        raise ValueError("noise floor needs two complete, aligned panel runs")
    d = [a - b for a, b in zip(rep_a, rep_b, strict=True)]
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
    return max(abs(statistics.mean(d)), se)


# ------------------------------------------------------------------ pool profiling: unpaired estimator

# User design 2026-10-19 (docs/trimem_pool_profiling_plan_v0.1.0.md): every configuration runs on its OWN
# stratified 200-sample of the profiling pool, so paired per-query gains do not exist. ``unpaired``
# (default of the pool driver): BO quality = stratified mean quality - NoMemory stratified mean (an unpaired
# gain; the NoMemory term is a common constant, so its SE is NOT part of the GP noise, which is the
# configuration's own stratified SE^2); costs = stratified means with delta-method log noise.
# ``paired`` = the old behaviour (``load_points``: one shared panel, paired gains), still selectable.
ESTIMATORS = ("unpaired", "paired")


_UNIFORM_SAMPLES: set[str] = (
    set()
)  # sample-size-v1: config keys whose sample is uniform (simple estimator)


def use_uniform_samples(keys) -> None:
    """sample-size-v1 (user 2026-10-22): configurations whose own sample was drawn uniformly from the pool get the
    simple-random-sample estimator (``unpaired_stats.simple``) instead of the stratified one."""
    _UNIFORM_SAMPLES.clear()
    _UNIFORM_SAMPLES.update(keys or ())


def config_estimates(
    rows: list[dict],
    *,
    samples: dict[str, list[str]],
    strata: dict[str, str],
    weights: dict[str, float],
    sizes: dict[str, int] | None,
    experiment: str,
) -> dict[str, dict]:
    """Rows of one pool-profiling experiment -> per configuration key (complete samples only):
    stratified estimates of quality, tokens and latency plus the per-query values. Every row must belong
    to its configuration's own recorded sample (anything else is a protocol error, never silently used)."""
    from trimem.optimizer import unpaired_stats as us

    groups: dict[str, dict[str, dict]] = {}
    quality_domains = {
        r.get("quality_domain", "legacy") for r in rows
        if r.get("experiment_id") == experiment and r.get("split") == "train"
    }
    if len(quality_domains) > 1 or not quality_domains <= {"legacy", "raw_environment_score"}:
        raise ValueError("inconsistent or unsupported quality domain")
    raw_quality = quality_domains == {"raw_environment_score"}
    for r in rows:
        if r["split"] != "train":
            raise ValueError("BO input must be training rows only")
        if r["experiment_id"] != experiment:
            continue
        if r["protocol"].get("space_version") != space.VERSION:
            raise ValueError("Row profiled under a different space version")
        declared = space.validate(r["protocol"]["space_config"])
        if executed_of(r) != space.runner_args(declared):
            raise ValueError(
                f"Executed configuration {r['config']} differs from the declared {declared}"
            )
        k = space.key(declared)
        if k not in samples:
            raise ValueError(f"row of configuration {k} without a recorded sample")
        q = str(r["query_id"])
        if q not in set(samples[k]):
            raise ValueError(f"query {q} outside the recorded sample of {k}")
        g = groups.setdefault(k, {})
        if q in g:
            raise ValueError(f"duplicate evaluated row for {q} / {k}")
        for f in ("quality", "total_tokens", "e2e_ms"):
            if r[f] is None or not math.isfinite(float(r[f])):
                raise ValueError(f"missing {f} for {q}")
        if (
            (not raw_quality and not 0 <= float(r["quality"]) <= 1)
            or float(r["total_tokens"]) <= 0
            or float(r["e2e_ms"]) <= 0
        ):
            raise ValueError(f"objective outside its valid domain for {q}")
        g[q] = r
    out = {}
    for k, g in groups.items():
        ids = list(samples[k])
        if set(g) != set(ids):
            continue  # an incomplete sample is not an observation
        h = [strata[q] for q in ids]
        vals = {
            f: [float(g[q][f]) for q in ids]
            for f in ("quality", "total_tokens", "e2e_ms")
        }
        if (
            k in _UNIFORM_SAMPLES
        ):  # sample-size-v1: uniform sample -> simple estimator (FPC = pool size)
            pop = sum(sizes.values()) if sizes else None

            def est(v, _pop=pop):
                return us.simple(v, _pop)

        else:

            def est(v, _h=h):
                return us.stratified(v, _h, weights, sizes)

        out[k] = {
            "cfg": json.loads(k),
            "ids": ids,
            "estimator": "simple" if k in _UNIFORM_SAMPLES else "stratified",
            "quality": est(vals["quality"]),
            "tokens": est(vals["total_tokens"]),
            "latency_ms": est(vals["e2e_ms"]),
            "per_query": {f: dict(zip(ids, v, strict=True)) for f, v in vals.items()},
        }
    return out


def load_points_unpaired(
    rows: list[dict],
    *,
    samples: dict[str, list[str]],
    strata: dict[str, str],
    weights: dict[str, float],
    sizes: dict[str, int] | None,
    dataset: str,
    pool_hash: str,
    experiment: str,
    baseline_quality: float,
) -> list[hb.Point]:
    """One BO point per configuration with a complete own sample (see ESTIMATORS)."""
    from trimem.optimizer import unpaired_stats as us

    pts = []
    for k, e in config_estimates(
        rows,
        samples=samples,
        strata=strata,
        weights=weights,
        sizes=sizes,
        experiment=experiment,
    ).items():
        noise = (
            max(e["quality"].se ** 2, 1e-8),
            max(us.log_mean_var(e["tokens"]), 1e-8),
            max(us.log_mean_var(e["latency_ms"]), 1e-8),
        )
        pts.append(
            hb.Point(
                hashlib.sha256(k.encode()).hexdigest(),
                experiment,
                task_of(dataset, pool_hash),
                e["cfg"],
                0,
                0,
                1,
                e["quality"].mean - baseline_quality,
                e["tokens"].mean,
                e["latency_ms"].mean,
                noise,
            )
        )
    return pts
