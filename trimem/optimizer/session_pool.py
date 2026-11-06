"""Latest harness candidate-pool engine adapted to shared session observations.

Derived from tridb_v6_stage/trimem/experiments/harness_pool.py. Provenance and
intentional differences are recorded beside this module. Sobol, analyzer input,
DLS, crossover and arbitration retain their source implementation. Use one
instance per dataset with a frozen train-panel identity.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from typing import Any

from trimem.optimizer import session_observations as hb
from trimem.optimizer import session_space as sv

KNOB_ORDER = (
    "seed_count",
    "depth",
    "fanout",
    "top_k",
    "edges",
    "ordering",
    "dedup",
    "render",
    "cadence",
    "sem_fallback",
)
RETRIEVAL_PLANS = tuple(
    p for p in sv.PLANS if p not in (getattr(sv, "GATED_PLANS", {}) or {})
)
NOMEM_SHARE = 0.04  # share of Sobol breadth draws that are NoMemory (one configuration)


def _key(cfg: dict[str, Any]) -> str:
    return sv.key(cfg)


def use_space(module) -> None:
    """Select the session space (default: SWE-QA ``session_space``; agent-env datasets use
    ``session_space_agentenv``). Also switches the observation aggregator to the same space."""
    global sv, RETRIEVAL_PLANS
    sv = module
    # gated plans (e.g. plan-p1-v1 P1) are proposable only after use_plans names them (recorded extension)
    gated = getattr(module, "GATED_PLANS", {}) or {}
    RETRIEVAL_PLANS = tuple(p for p in module.PLANS if p not in gated)
    hb.space = module


def use_plans(plans) -> None:
    """Restrict the PROPOSABLE plans (Sobol breadth, plan switches in DLS neighbours, snapping of agent rows) to
    ``plans`` -- a subset of the active space's ``PLANS`` in their order (plan-p1rank-v1: the τ² run proposes
    P1+Rank only from its extension round). Validation, keys and the GP encoding keep every level of the space.
    ``use_space`` resets to all plans."""
    global RETRIEVAL_PLANS
    plans = tuple(plans)
    if not plans or any(p not in sv.PLANS for p in plans):
        raise ValueError(f"plans {plans} not a subset of the space's {sv.PLANS}")
    RETRIEVAL_PLANS = tuple(p for p in sv.PLANS if p in plans)


def plan_default(plan: str) -> dict[str, Any]:
    """The plan's round-0 default (``plan_defaults`` + fixed capabilities), proposable form."""
    return canon({"plan": plan, **sv.plan_defaults(), **sv.v7_defaults(plan)})


def unmeasured_plans(measured: list[dict[str, Any]]) -> list[str]:
    """Proposable plans added by a plan extension (the space's ``ADDED_PLANS``, plan-p1rank-v1) that have no
    measured configuration yet. Spaces without ``ADDED_PLANS`` never get such starts (legacy behaviour)."""
    seen = {c["plan"] for c in measured}
    added = tuple(getattr(sv, "ADDED_PLANS", ()))
    return [p for p in RETRIEVAL_PLANS if p in added and p not in seen]


def propose_fixed() -> dict[str, Any]:
    """Proposal-side constraint of the active space (``PROPOSE_FIXED``, e.g. τ² sim_gate = none from round 1,
    user 2026-10-20): knobs every PROPOSED configuration must hold, while measured configurations with other values
    stay valid observations. Empty for spaces without it."""
    return dict(getattr(sv, "PROPOSE_FIXED", {}) or {})


# Plan-conditional proposal rule (semfilter-v2b, user 2026-10-23, WebShop from round 1; None = off = unchanged):
# {"plans": SemFilter plans, "levels": {knob: allowed levels}, "set": {knob: value}}. For a configuration of one of
# ``plans``: the PROPOSAL domain keeps only ``levels`` (semfilter_threshold {0.4, 0.6}: 0.8 no longer proposable),
# ``canon`` sets the ``set`` knobs (sem_fallback fill-topk) and snaps a disallowed numeric level to the nearest
# allowed one (0.8 -> 0.6), and ``proposable`` requires both. Measured configurations with other values stay valid
# observations (validation, keys and the GP encoding keep every level). Set by the driver per round
# (``session_bo_agentenv.use_semfilter_v2b_for``).
_proposal_rule: dict[str, Any] = {"rule": None}


def use_proposal_rule(rule: dict[str, Any] | None) -> None:
    if rule is not None:
        rule = {
            "plans": frozenset(rule["plans"]),
            "levels": {k: tuple(v) for k, v in rule.get("levels", {}).items()},
            "set": dict(rule.get("set", {})),
        }
    _proposal_rule["rule"] = rule


def proposal_rule() -> dict[str, Any] | None:
    return _proposal_rule["rule"]


def _rule_for(plan: str) -> dict[str, Any] | None:
    rule = _proposal_rule["rule"]
    return rule if rule is not None and plan in rule["plans"] else None


def proposal_domain(plan: str) -> dict:
    """The plan's searched knobs minus the proposal-fixed ones (and restricted by the proposal rule, if any)."""
    fixed = propose_fixed()
    out = {k: v for k, v in sv.domain(plan).items() if k not in fixed}
    rule = _rule_for(plan)
    if rule is not None:
        for k, levels in rule["levels"].items():
            if k in out:
                out[k] = tuple(x for x in out[k] if x in levels)
    return out


def _apply_rule(cfg: dict[str, Any]) -> dict[str, Any]:
    rule = _rule_for(cfg["plan"])
    if rule is None:
        return cfg
    out = {**sv.complete(cfg), **rule["set"]}
    for k, levels in rule["levels"].items():
        v = out.get(k)
        if v not in levels:
            out[k] = min(levels, key=lambda x: (abs(x - v), x))
    return out


def canon(cfg: dict[str, Any]) -> dict[str, Any]:
    """Canonical proposable form: the proposal-fixed knobs set to their fixed values (identity without them);
    with a proposal rule, its knobs set / snapped for the rule's plans."""
    fixed = propose_fixed()
    if cfg["plan"] in ("NoMemory", "SemRankOnly"):
        return sv.validate(cfg) if cfg["plan"] != "SemRankOnly" else cfg
    if not fixed:
        return sv.validate(_apply_rule(sv.validate(cfg)))
    return sv.validate(_apply_rule({**sv.complete(cfg), **fixed}))


def proposable(cfg: dict[str, Any]) -> bool:
    fixed = propose_fixed()
    if cfg["plan"] == "NoMemory":
        return True
    if not all(cfg.get(k) == v for k, v in fixed.items()):
        return False
    rule = _rule_for(cfg["plan"])
    if rule is None:
        return True
    return all(cfg.get(k) == v for k, v in rule["set"].items()) and all(
        cfg.get(k) in levels for k, levels in rule["levels"].items()
    )


IMPROVE_EPS = 1e-12  # a window "improved" if its normalized gain exceeds this
IMPROVE_NOISE = 0.25  # Bernoulli variance bound p(1-p) <= 1/4, per window


def targets(points: list[hb.Point], quality_target: str = "gain"):
    """Training targets [quality, -ln tokens/iter, -ln latency/iter] and fixed noise.

    quality_target="gain": normalized best-fitness gain per iteration (noise sigma_q^2 / h).
    quality_target="improve": 1 if the window improved the best at all, else 0 -- the
    probability of improvement within the window's horizon (h is a feature), which separates
    configurations even where most single-iteration gains are exactly zero."""
    y, yvar = hb._ys(points)
    if quality_target == "improve":
        for row, var, p in zip(y, yvar, points, strict=True):
            row[0] = 1.0 if p.quality > IMPROVE_EPS else 0.0
            var[0] = IMPROVE_NOISE
    elif quality_target != "gain":
        raise ValueError(f"unknown quality target {quality_target!r}")
    return y, yvar


def p2_budget(points: list[hb.Point]) -> tuple[float, float]:
    """Per-iteration (tokens, latency_ms) budget = mean of the measured P2 windows."""
    import statistics

    p2 = [p for p in points if p.cfg.get("plan") == "P2"]
    if not p2:
        raise ValueError("no P2 windows to derive the budget from")
    return statistics.mean(p.tokens for p in p2), statistics.mean(
        p.latency_ms for p in p2
    )


def quality_first_scores(
    model, ctxs, cfgs, budget: tuple[float, float] | None, beta: float
) -> list[float]:
    """Quality-first acquisition: UCB_beta of the quality posterior x P(cost within budget).

    ``budget=None``: no cost constraint at all -- the score is UCB_beta(quality) alone.

    Feasibility is the probability that both ln tokens/iter and ln latency/iter are at most the
    budget's, from the cost GPs (independent Gaussians; outputs are -ln cost). Averaged over
    the panel contexts. Cheaper configurations get no credit beyond feasibility."""
    import torch

    norm = torch.distributions.Normal(0.0, 1.0)
    total = torch.zeros(len(cfgs), dtype=torch.double)
    with torch.no_grad():
        for c in ctxs:
            post = model.posterior(c.x(cfgs))
            m, sd = post.mean, post.variance.clamp_min(1e-12).sqrt()
            ucb = m[:, 0] + beta * sd[:, 0]
            if budget is None:
                total += ucb
                continue
            ln_t, ln_l = math.log(budget[0]), math.log(budget[1])
            feas = norm.cdf((m[:, 1] + ln_t) / sd[:, 1]) * norm.cdf(
                (m[:, 2] + ln_l) / sd[:, 2]
            )
            total += ucb.clamp_min(1e-6) * feas
    return (total / len(ctxs)).tolist()


# ------------------------------------------------------------------ GP protocol (gp-fix-v1, 2026-10-20)
# gp-v0: the measured SE^2 as fixed noise, no outputscale prior. On the τ² -v5 run (round 3, 20 points) ML-II drove
# the quality outputscale to ~0: half of each paired SE^2 is common-mode (shared NoMemory arm, shared task set),
# so the fixed noise exceeded the observed between-configuration spread (Cochran Q 12.8 on 19 df) and the
# constant-mean model won. gp-fix-v1 = Gamma outputscale priors on every ScaleKernel (no collapse to a constant)
# + a degeneracy diagnostic after every fit; the common-mode part of the noise is removed by the caller that
# knows the sampling design (``unpaired_stats.common_mode_cov``; τ²: ``pool_profile.observed_points``).
GP_LEGACY = "gp-v0"
GP_PROTOCOL = "gp-fix-v1"
OUTPUTSCALE_PRIOR = (
    2.0,
    1.0,
)  # Gamma(concentration, rate) on standardized outcomes: mode 1, density 0 at 0
DEGENERATE_RATIO = 0.05  # posterior-mean SD over the training points < 5% of the observed SD = degenerate
_gp = {"protocol": GP_PROTOCOL}


def use_gp(protocol: str) -> None:
    """Select the GP protocol of ``fit`` (process-wide, like ``use_space``): gp-fix-v1 (default) or gp-v0."""
    if protocol not in (GP_LEGACY, GP_PROTOCOL):
        raise ValueError(f"unknown GP protocol {protocol!r}")
    _gp["protocol"] = protocol


def gp_protocol() -> str:
    return _gp["protocol"]


# ------------------------------------------------------------------ auxiliary observations (aux-obs-v1, 2026-10-20)
# Earlier profiling runs (other protocols) enter the GP ONLY, as points whose ``point_id`` starts with AUX_PREFIX.
# While ``use_aux(True)`` (set per round by the driver), every GP input gets one extra CATEGORICAL dimension
# "source" (0 = current run, 1 = auxiliary); candidates and current points are encoded at source 0. Auxiliary
# configurations may be off the proposal grid (``space.features_unchecked``); they never become candidates,
# baselines, anchors, champions or front points (``suggest(aux=...)`` fits on points + aux and uses ``points``
# for everything else).
AUX_PREFIX = "aux:"
_aux = {"on": False}


def use_aux(on: bool) -> None:
    """Process-wide switch of the source dimension (like ``use_gp``)."""
    _aux["on"] = bool(on)


def aux_on() -> bool:
    return _aux["on"]


def is_aux(p: hb.Point) -> bool:
    return p.point_id.startswith(AUX_PREFIX)


# ------------------------------------------------------------------ latency regime (latency-regime-v1, 2026-10-22)
# A run whose runtime (concurrency) changed mid-run (ALFWorld v7: c128 rounds 0-5 -> c16 from round 6) must not mix
# the two regimes' LATENCY. While ``use_latency_regime(old_keys)`` is on, every GP input gets one more CATEGORICAL
# column "latency regime": in the LATENCY model's training inputs it is 1 for observations of the old regime
# (configurations in ``old_keys``, and auxiliary observations) and 0 for the current regime; in the quality and
# token models, in every candidate / posterior input it is 0. So quality and tokens are shared across regimes
# unchanged (a constant column), and the latency GP learns how far the old regime's latency transfers (the
# categorical kernel's correlation) while every prediction is made at the current regime.
_lat_regime: dict[str, Any] = {"on": False, "old": frozenset()}


def use_latency_regime(old_keys) -> None:
    """Process-wide switch (like ``use_aux``): ``old_keys`` = space keys of old-regime configurations; None = off."""
    _lat_regime["on"] = old_keys is not None
    _lat_regime["old"] = frozenset(old_keys or ())


def latency_regime_on() -> bool:
    return _lat_regime["on"]


# ------------------------------------------------------------------ SemFilter regime (semfilter-regime-v1, 2026-10-22)
# A run whose SemFilter implementation changed mid-run (ALFWorld v7: v1 judge rounds 0-5 -> semfilter-v2 from round 6:
# top-32 cosine cutoff, batched score-only judge, rubric v3) must not pool the two judges' observations as one: v2
# changes the SemFilter plans' semantic tokens and latency (and possibly quality). While
# ``use_semfilter_regime(old_keys)`` is on, every GP input gets one more CATEGORICAL column "semfilter regime" (after
# the latency-regime column): in the training inputs of ALL THREE models it is 1 for SemFilter-plan observations
# of the old judge (configurations in ``old_keys``; auxiliary SemFilter-plan observations) and 0 otherwise; every
# candidate / posterior input is 0 (predictions at v2). Non-SemFilter plans are always 0, i.e. unaffected.
SEMFILTER_REGIME_PLANS = frozenset({"P1", "P1+Rank", "P2", "P2+Rank"})
_sf_regime: dict[str, Any] = {"on": False, "old": frozenset()}


def use_semfilter_regime(old_keys) -> None:
    """Process-wide switch: ``old_keys`` = space keys of the old-judge SemFilter configurations; None = off."""
    _sf_regime["on"] = old_keys is not None
    _sf_regime["old"] = frozenset(old_keys or ())


def semfilter_regime_on() -> bool:
    return _sf_regime["on"]


def _extra_cols() -> list[float]:
    return (
        ([0.0] if _aux["on"] else [])
        + ([0.0] if _lat_regime["on"] else [])
        + ([0.0] if _sf_regime["on"] else [])
    )


def _lat_col() -> int:
    return _n_base() + (1 if _aux["on"] else 0)


def _sf_col() -> int:
    return _lat_col() + (1 if _lat_regime["on"] else 0)


def _old_semfilter(p) -> bool:
    if p.cfg.get("plan") not in SEMFILTER_REGIME_PLANS:
        return False
    return is_aux(p) or _key(p.cfg) in _sf_regime["old"]


# ------------------------------------------------------------------ HV reference point (hv-ref-anchored-v1, 2026-10-22)
# The acquisition's hypervolume is computed in raw objective space (quality, -ln tokens, -ln latency), so the
# reference point sets the implicit exchange rate between the objectives (the GP's Standardize does not).
# hv-ref-worst-v0 (legacy): quality min(0, worst observed) - 1e-3, costs worst observed - ln 1.1 -- a dominated
#   outlier (e.g. 322k tokens) pushes the cost refs out and dilutes every cost improvement.
# hv-ref-anchored-v1 (user 2026-10-22): quality = the NoMemory baseline minus one SE of its estimate (objective
#   units: the pool drivers' quality is the gain vs NoMemory, so the baseline is 0); each cost = the nadir of the
#   current 3-D non-dominated set of posterior means minus 0.1 x the front's range on that axis (BoTorch
#   infer_reference_point style; zero range -> ln 1.1 slack). Dominated points never move it. The ref is then
#   made strictly worse than every front point on every axis (``clamped`` lists the axes that needed it).
#   Without a NoMemory baseline the quality ref falls back to the legacy rule (logged, ``quality_rule``).
# Acquisition only: reported hypervolumes (render_cost_saving_hv, optimizer_replay, session_bo_agentenv
# ``hypervolume`` / ``fixed_reference``) keep their own fixed definitions.
HV_REF_LEGACY = "hv-ref-worst-v0"
HV_REF_ANCHORED = "hv-ref-anchored-v1"
HV_REF_MARGIN = 0.1  # x the front's range beyond its nadir (cost axes)
HV_REF_SLACK = math.log(1.1)  # cost slack of a zero-range front (= the legacy slack)
HV_REF_EPS = 1e-3  # strictness margin (= the legacy quality margin)
_hv_ref: dict[str, Any] = {
    "protocol": HV_REF_LEGACY,
    "baseline": None,
    "margin_floor": 0.0,
    "cost_anchor": None,
    "spread": None,
}


def use_hv_ref(
    protocol: str,
    baseline: tuple[float, float] | None = None,
    *,
    margin_floor: float = 0.0,
    cost_anchor: tuple[float, float] | None = None,
    spread: tuple[float, float, float] | None = None,
) -> None:
    """Process-wide reference-point rule of the HV acquisition (like ``use_gp``). ``baseline`` = (NoMemory quality in
    the objective's units, its SE); only the anchored rule uses it (None -> legacy quality rule, logged).
    ``margin_floor`` / ``cost_anchor``: offline-study variants of the anchored cost refs (``anchored_reference``);
    the defaults (0, None) are hv-ref-anchored-v1 as specified."""
    if protocol not in (HV_REF_LEGACY, HV_REF_ANCHORED):
        raise ValueError(f"unknown HV reference protocol {protocol!r}")
    if baseline is not None:
        q, se = (float(x) for x in baseline)
        if not (math.isfinite(q) and math.isfinite(se) and se >= 0):
            raise ValueError(f"invalid NoMemory baseline {baseline!r}")
        baseline = (q, se)
    _hv_ref.update(
        protocol=protocol,
        baseline=baseline if protocol == HV_REF_ANCHORED else None,
        margin_floor=float(margin_floor),
        cost_anchor=tuple(cost_anchor) if cost_anchor is not None else None,
        spread=tuple(map(float, spread)) if spread is not None else None,
    )


def hv_ref_protocol() -> str:
    return _hv_ref["protocol"]


def legacy_reference(ys: list[list[float]]) -> list[float]:
    """hv-ref-worst-v0 over the training targets ``ys`` (rows [quality, -ln tok, -ln lat])."""
    worst = [min(y[i] for y in ys) for i in range(3)]
    return [
        min(0.0, worst[0]) - 1e-3,
        worst[1] - math.log(1.1),
        worst[2] - math.log(1.1),
    ]


def nondominated(pts: list[list[float]]) -> list[list[float]]:
    """Strict 3-D non-dominated subset (maximisation; identical vectors all kept)."""

    def dom(a, b):
        return all(x >= y for x, y in zip(a, b, strict=True)) and any(
            x > y for x, y in zip(a, b, strict=True)
        )

    return [p for p in pts if not any(dom(o, p) for o in pts if o is not p)]


def anchored_reference(
    means: list[list[float]],
    baseline: tuple[float, float] | None,
    ys: list[list[float]] | None = None,
    *,
    margin_floor: float = 0.0,
    cost_anchor: tuple[float, float] | None = None,
    spread: tuple[float, float, float] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """hv-ref-anchored-v1 from the posterior means ``means`` of the measured configurations (the function takes the
    non-dominated subset itself). ``ys`` = training targets for the legacy quality fallback (default ``means``).

    Offline-study variants of the cost refs (defaults = the specified rule): ``margin_floor`` -> margin =
    max(0.1 x range, margin_floor) (-ln units, e.g. ln 1.1); ``cost_anchor`` = NoMemory's (-ln tokens, -ln latency)
    -> ref = min(front nadir, NoMemory) - ln 1.1 (the worse of the two costs, then 10 % slack). ``spread`` = per-axis
    scales s ("normalized HV": y~ = (y - anchor) / s, ref -1 on every axis) -> ref = (NoMemory baseline, front
    nadir, front nadir) - s; the HV in y~ is the raw HV / prod(s), so the acquisition is unchanged by the scaling."""
    if not means:
        raise ValueError("anchored reference needs at least one measured configuration")
    front = nondominated([list(map(float, m)) for m in means])
    lo = [min(p[i] for p in front) for i in range(3)]
    hi = [max(p[i] for p in front) for i in range(3)]
    ref = [0.0, 0.0, 0.0]
    if baseline is not None:
        ref[0] = baseline[0] - (baseline[1] if spread is None else float(spread[0]))
        q_rule = "nomemory-minus-se" if spread is None else "nomemory-minus-spread"
    else:
        ref[0] = legacy_reference(ys if ys is not None else means)[0]
        q_rule = "fallback-legacy (no NoMemory baseline)"
        logging.getLogger(__name__).warning(
            "%s: no NoMemory baseline in this context; quality ref falls back to the legacy rule",
            HV_REF_ANCHORED,
        )
    for i in (1, 2):
        rng = hi[i] - lo[i]
        if spread is not None:
            ref[i] = lo[i] - float(spread[i])
        elif cost_anchor is not None:
            ref[i] = min(lo[i], float(cost_anchor[i - 1])) - HV_REF_SLACK
        else:
            margin = HV_REF_MARGIN * rng if rng > 1e-12 else HV_REF_SLACK
            ref[i] = lo[i] - max(margin, margin_floor)
    clamped = []
    for i, name in enumerate(("quality", "tokens", "latency")):
        if ref[i] >= lo[i]:
            ref[i] = lo[i] - HV_REF_EPS
            clamped.append(name)
    return ref, {
        "protocol": HV_REF_ANCHORED,
        "quality_rule": q_rule,
        "baseline": list(baseline) if baseline is not None else None,
        "front_size": len(front),
        "front_nadir": lo,
        "front_ideal": hi,
        "clamped": clamped,
        "margin_floor": margin_floor,
        "cost_anchor": list(cost_anchor) if cost_anchor is not None else None,
        "spread": list(spread) if spread is not None else None,
        "ref": ref,
    }


def _n_base() -> int:
    return len(sv.features(dict(sv.NOMEM)))


def cat_dims() -> list[int]:
    n = _n_base()
    out = list(sv.CAT_DIMS) + ([n] if _aux["on"] else [])
    if _lat_regime["on"]:
        out.append(n + (1 if _aux["on"] else 0))
    if _sf_regime["on"]:
        out.append(_sf_col())
    return out


def _old_regime(p: hb.Point) -> bool:
    return is_aux(p) or _key(p.cfg) in _lat_regime["old"]


def point_features(p: hb.Point) -> list[float]:
    """GP input of an observation: the space's encoding (auxiliary ones without the grid check) + the source
    dimension when ``use_aux`` is on."""
    if is_aux(p):
        if not _aux["on"]:
            raise ValueError("auxiliary observation while use_aux is off")
        return (
            list(sv.features_unchecked(p.cfg))
            + [1.0]
            + ([0.0] if _lat_regime["on"] else [])
            + ([1.0 if _old_semfilter(p) else 0.0] if _sf_regime["on"] else [])
        )
    base = list(sv.features(p.cfg, p.task, p.start_norm, p.iter_start, p.h))
    return base + _extra_cols()


def _outputscale_priors(m, prior: tuple[float, float] = OUTPUTSCALE_PRIOR) -> None:
    from gpytorch.kernels import ScaleKernel
    from gpytorch.priors import GammaPrior

    for k in m.covar_module.modules():
        if isinstance(k, ScaleKernel):
            k.register_prior(
                "outputscale_prior",
                GammaPrior(*prior),
                lambda s: s.outputscale,
                lambda s, v: s._set_outputscale(v),
            )


def diagnostics(model, x, y, yvar) -> list[dict[str, Any]]:
    """Per output: outputscales and signal variance (standardized units), mean noise, the SD of the posterior
    mean over the training points vs the observed SD, and ``degenerate`` (a model that cannot rank)."""
    import torch
    from gpytorch.kernels import ScaleKernel

    out = []
    with torch.no_grad():
        for i, m in enumerate(model.models):
            xi = x[i] if isinstance(x, (list, tuple)) else x
            std = float(m.outcome_transform.stdvs.reshape(-1)[0])
            kxx = float(m.covar_module(m.train_inputs[0]).diagonal().mean())
            mu = m.posterior(xi).mean[:, 0]
            obs_sd = float(y[:, i].std())
            mean_sd = float(mu.std())
            out.append(
                {
                    "outputscales": [
                        float(k.outputscale)
                        for k in m.covar_module.modules()
                        if isinstance(k, ScaleKernel)
                    ],
                    "signal_var_std": kxx,
                    "noise_var_std": float(yvar[:, i].mean()) / std**2,
                    "obs_sd": obs_sd,
                    "post_mean_sd": mean_sd,
                    "explained_ratio": mean_sd / obs_sd if obs_sd > 0 else 0.0,
                    "degenerate": obs_sd > 0 and mean_sd < DEGENERATE_RATIO * obs_sd,
                }
            )
    return out


def fit(
    points: list[hb.Point], quality_target: str = "gain", protocol: str | None = None
):
    """Three MixedSingleTaskGPs on [quality, -ln tokens, -ln latency] with the points' noise fixed
    (heteroscedastic; ``Standardize`` rescales it to the model's scale). ``protocol`` (default: ``use_gp``):
    gp-fix-v1 adds Gamma outputscale priors. ``model.gp_diagnostics`` = ``diagnostics`` of the fit."""
    from functools import partial

    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.models.utils.gpytorch_modules import (
        get_covar_module_with_dim_scaled_prior,
    )
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    args = {"dtype": torch.double}
    x = torch.tensor([point_features(p) for p in points], **args)
    if _sf_regime[
        "on"
    ]:  # semfilter-regime-v1: all three models see the old-judge column
        x[:, _sf_col()] = torch.tensor(
            [1.0 if _old_semfilter(p) else 0.0 for p in points], **args
        )
    xs = [x, x, x]
    if _lat_regime[
        "on"
    ]:  # latency-regime-v1: only the latency model sees the regime column
        x_lat = x.clone()
        x_lat[:, _lat_col()] = torch.tensor(
            [1.0 if _old_regime(p) else 0.0 for p in points], **args
        )
        xs[2] = x_lat
    y, yvar = targets(points, quality_target)
    y, yvar = torch.tensor(y, **args), torch.tensor(yvar, **args)
    models = [
        MixedSingleTaskGP(
            train_X=xs[i],
            train_Y=y[:, i : i + 1],
            train_Yvar=yvar[:, i : i + 1],
            cat_dims=cat_dims(),
            outcome_transform=Standardize(m=1),
            # Retain upstream dimension-scaled priors and the
            # 0.025 lengthscale floor. An unconstrained custom
            # kernel collapsed below 1e-8 and failed line search.
            cont_kernel_factory=partial(
                get_covar_module_with_dim_scaled_prior, use_rbf_kernel=False
            ),
        )
        for i in range(3)
    ]
    protocol = protocol or gp_protocol()
    if protocol == GP_PROTOCOL:
        for m in models:
            _outputscale_priors(m)
    elif protocol != GP_LEGACY:
        raise ValueError(f"unknown GP protocol {protocol!r}")
    model = ModelListGP(*models)
    if protocol == GP_PROTOCOL:
        # Deterministic: on an optimizer failure botorch retries from initial values sampled from the priors
        # (global torch RNG); a fixed local seed keeps every fit of the same points identical.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    else:
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    model.gp_protocol = protocol
    model.gp_diagnostics = diagnostics(model, xs, y, yvar)
    return model, y


class Context:
    def __init__(self, task: str, start_norm: float, iter_start: int, horizon: int = 1):
        self.task, self.start_norm, self.iter_start, self.horizon = (
            task,
            start_norm,
            iter_start,
            horizon,
        )

    def x(self, cfgs: list[dict[str, Any]]):
        import torch

        src = (
            _extra_cols()
        )  # candidates: current run (source 0), current latency regime (0)
        return torch.tensor(
            [
                list(
                    sv.features(
                        c,
                        self.task,
                        self.start_norm,
                        self.iter_start,
                        sv.horizon(c) if self.horizon is None else self.horizon,
                    )
                )
                + src
                for c in cfgs
            ],
            dtype=torch.double,
        )

    def as_dict(self):
        return {
            "task": self.task,
            "start_best_norm": self.start_norm,
            "iter_start": self.iter_start,
            "horizon": self.horizon,
        }


# ------------------------------------------------------------------ channels


def sobol_breadth(n: int, seed: int) -> list[dict[str, Any]]:
    from torch.quasirandom import SobolEngine

    eng = SobolEngine(
        dimension=1 + len(KNOB_ORDER), scramble=True, seed=seed
    )  # max knob count
    out, seen = [], set()
    for u in eng.draw(n).tolist():
        if u[0] < NOMEM_SHARE:
            cfg = dict(sv.NOMEM)
        else:
            plan = RETRIEVAL_PLANS[
                min(
                    len(RETRIEVAL_PLANS) - 1,
                    int(
                        (u[0] - NOMEM_SHARE) / (1 - NOMEM_SHARE) * len(RETRIEVAL_PLANS)
                    ),
                )
            ]
            dom = proposal_domain(plan)
            cfg = {"plan": plan}
            for knob, v in zip(
                dom, u[1:], strict=False
            ):  # this plan's active knobs only
                levels = dom[knob]
                cfg[knob] = levels[min(len(levels) - 1, int(v * len(levels)))]
            cfg = canon(cfg)
        k = _key(cfg)
        if k not in seen:
            seen.add(k)
            out.append(sv.validate(cfg))
    return out


def switch_plan(cfg: dict[str, Any], plan: str) -> dict[str, Any]:
    """Same knobs where they exist on ``plan``; plan-specific knobs at that plan's defaults."""
    if plan == "NoMemory":
        return dict(sv.NOMEM)
    grid, defaults = sv.grid(), sv.plan_defaults()
    base = (
        {k: cfg[k] for k in grid}
        if cfg["plan"] not in ("NoMemory", "SemRankOnly")
        else {
            k: v for k, v in defaults.items() if k in grid
        }  # fixed knobs come from v7_defaults
    )
    base = {k: (v if v in grid[k] else defaults[k]) for k, v in base.items()}
    out = {"plan": plan, **base, **sv.v7_defaults(plan)}
    if cfg["plan"] not in ("NoMemory", "SemRankOnly"):
        for k in ("edges", "dedup", "render", "cadence"):
            if k in sv.domain(plan):  # only knobs searched in the active space
                out[k] = cfg.get(k, out[k])
        if cfg.get("ordering") in sv.orderings(plan):
            out["ordering"] = cfg["ordering"]
        if cfg.get("sem_fallback") in sv.fallbacks(plan):
            out["sem_fallback"] = cfg["sem_fallback"]
    return sv.validate(out)


def coverage_breadth(n: int, seed: int, sampler: str = "sobol") -> list[dict]:
    """Coverage proposals; LHS stratifies each active plan's conditional domain.

    Same raw draw allowance and NoMemory share as Sobol; canonical duplicates
    are removed, not replaced with extra draws. Shuffle the final rows so a
    consumer taking a prefix does not systematically favor the first plan.
    """
    if n <= 0:
        return []
    if sampler == "sobol":
        return sobol_breadth(n, seed)
    if sampler != "lhs":
        raise ValueError(f"unknown coverage sampler {sampler!r}")
    import random

    rng = random.Random(seed)
    n_nomem = sum(rng.random() < NOMEM_SHARE for _ in range(n))
    counts = [0] * len(RETRIEVAL_PLANS)
    order = list(range(len(counts)))
    rng.shuffle(order)
    for i in range(n - n_nomem):
        counts[order[i % len(order)]] += 1
    rows = [dict(sv.NOMEM)] if n_nomem else []
    for plan, count in zip(RETRIEVAL_PLANS, counts, strict=True):
        if not count:
            continue
        dom = proposal_domain(plan)
        cols = {}
        for knob, levels in dom.items():
            strata = list(range(count))
            rng.shuffle(strata)
            cols[knob] = [
                levels[
                    min(len(levels) - 1, int((j + rng.random()) / count * len(levels)))
                ]
                for j in strata
            ]
        rows.extend(
            canon({"plan": plan, **{k: col[i] for k, col in cols.items()}})
            for i in range(count)
        )
    unique = list({_key(c): c for c in rows}.values())
    rng.shuffle(unique)
    return unique


def neighbors(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = sv.complete(cfg)
    out = [
        canon(switch_plan(cfg, p))
        for p in ("NoMemory", *RETRIEVAL_PLANS)
        if p != cfg["plan"]
    ]
    for knob, levels in proposal_domain(cfg["plan"]).items():
        for level in levels:
            if level != cfg.get(knob):
                out.append(canon(cfg | {knob: level}))
    # Spaces that canonicalize redundant combinations (session_space_agentenv_v2) can map a one-knob move
    # back onto the configuration itself or onto another neighbour: keep each distinct neighbour once
    # (a no-op for spaces without canonicalization).
    own, seen, uniq = _key(cfg), set(), []
    for c in out:
        k = _key(c)
        if k != own and k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def snap(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Nearest valid grid point of a (possibly partial / off-grid) configuration, or None."""
    plan = cfg.get("plan")
    if plan not in ("NoMemory", *RETRIEVAL_PLANS):
        return None
    if plan == "NoMemory":
        return dict(sv.NOMEM)
    out = {"plan": plan, **sv.plan_defaults(), **sv.v7_defaults(plan)}
    for knob, levels in proposal_domain(plan).items():
        if knob not in cfg:
            continue
        v = cfg[knob]
        if v in levels:
            out[knob] = v
        elif (
            all(isinstance(level, (int, float)) for level in levels)
            and isinstance(v, (int, float))
            and not isinstance(v, bool)
        ):
            out[knob] = min(levels, key=lambda level: (abs(level - v), level))
        else:
            return None
    return canon(out)


# ------------------------------------------------------------------ arbitration


def _acq(
    model,
    y,
    ctx: Context,
    here: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    seed: int,
    mc_samples: int = 128,
    front_cfgs: list[dict[str, Any]] | None = None,
    ref_info: dict | None = None,
):
    """qLogNEHVI with the configurations measured AT THIS CONTEXT as the noisy baseline.

    Configurations measured only elsewhere are not baseline points (a candidate identical to a
    baseline point gets ~zero improvement, which would forbid re-measuring a known
    configuration at a new state). Without any same-context measurement there is no noisy
    incumbent, so the v6 rule applies: qLogEHVI against the non-dominated posterior means at
    this context of every configuration measured anywhere.

    Reference point: ``use_hv_ref`` (hv-ref-worst-v0 by default). Under hv-ref-anchored-v1 the front is the
    non-dominated set of the posterior means at this context of ``front_cfgs`` (default ``measured``; ``suggest``
    passes the real measurements only, so kriging-believer pseudo-points never move the ref); ``ref_info`` (if
    given) is filled with the rule's audit record.
    """
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogExpectedHypervolumeImprovement,
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )
    from botorch.utils.multi_objective.pareto import is_non_dominated

    tag = ""
    if _hv_ref["protocol"] == HV_REF_ANCHORED:
        with torch.no_grad():
            fm = model.posterior(ctx.x(front_cfgs or measured)).mean
        ref_l, info = anchored_reference(
            fm.tolist(),
            _hv_ref["baseline"],
            y.tolist(),
            margin_floor=_hv_ref["margin_floor"],
            cost_anchor=_hv_ref["cost_anchor"],
            spread=_hv_ref["spread"],
        )
        tag = f" [{HV_REF_ANCHORED}]"
    else:
        ref_l = legacy_reference(y.tolist())
        info = {"protocol": HV_REF_LEGACY, "ref": ref_l}
    if ref_info is not None:
        ref_info.update(info)
    ref = torch.tensor(ref_l, dtype=torch.double)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=seed)
    if here:
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref.tolist(),
            X_baseline=ctx.x(here),
            prune_baseline=True,
            sampler=sampler,
            cache_root=False,
        )
        return acq, ref, "qLogNEHVI(baseline: same-context measurements)" + tag
    with torch.no_grad():
        mean = model.posterior(ctx.x(measured)).mean
    inc = mean[(mean > ref).all(dim=-1)]
    front = inc[is_non_dominated(inc)] if len(inc) else inc
    acq = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref.tolist(),
        partitioning=FastNondominatedPartitioning(ref_point=ref, Y=front),
        sampler=sampler,
    )
    return (
        acq,
        ref,
        "qLogEHVI(front: posterior means of measured configs at this context)" + tag,
    )


def score(
    acq, ctx: Context, cfgs: list[dict[str, Any]], batch: int = 256
) -> list[float]:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(cfgs), batch):
            out += acq(ctx.x(cfgs[i : i + batch]).unsqueeze(1)).reshape(-1).tolist()
    return out


def dls(
    start: dict[str, Any],
    f,
    avoid: set[str],
    max_steps: int = 30,
    priority_knobs: list[str] | None = None,
) -> tuple[dict[str, Any], float, int]:
    """Greedy one-coordinate ascent of f from start (start itself may be measured)."""
    cur, cur_v = start, f([start])[0]
    if _key(cur) in avoid:
        cur_v = -math.inf
    steps = 0
    for steps in range(1, max_steps + 1):
        nb = [c for c in neighbors(cur) if _key(c) not in avoid]
        if not nb:
            break
        # Priorities bias the path, never remove the fallback neighborhood. Each
        # priority is one acquisition-scored coordinate group, in declared order.
        groups, used = [], set()
        for knob in priority_knobs or []:
            group = [
                c
                for c in nb
                if _key(c) not in used
                and (
                    c["plan"] != cur["plan"]
                    if knob == "plan"
                    else c["plan"] == cur["plan"] and c.get(knob) != cur.get(knob)
                )
            ]
            groups.append(group)
            used.update(_key(c) for c in group)
        groups.append([c for c in nb if _key(c) not in used])
        moved = False
        for group in groups:
            if not group:
                continue
            vals = f(group)
            i = max(range(len(group)), key=lambda j: vals[j])
            if vals[i] > cur_v:
                cur, cur_v, moved = group[i], vals[i], True
                break
        if not moved:
            break
    return cur, cur_v, steps


def panel_posterior(model, ctxs, cfgs):
    """Mean (over the panel contexts) of the posterior mean and variance at each configuration."""
    import torch

    ctxs = ctxs if isinstance(ctxs, list) else [ctxs]
    with torch.no_grad():
        posts = [model.posterior(c.x(cfgs)) for c in ctxs]
        mean = sum(p.mean for p in posts) / len(posts)
        var = (
            sum(p.variance for p in posts) / len(posts) ** 2
        )  # variance of a mean of independent contexts
    return mean, var


def _cost(mean_row) -> float:
    return 0.5 * (float(mean_row[1]) + float(mean_row[2]))  # both maximized (-log cost)


def crossover(
    model, ctx: Context, measured: list[dict[str, Any]], top: int = 2
) -> list[dict[str, Any]]:
    """Pareto-tension grafts between the quality champion and the cost champion."""

    mean, _ = panel_posterior(model, ctx, measured)
    q = mean[:, 0].tolist()
    c = [_cost(m) for m in mean]
    champ = {
        "quality": measured[max(range(len(q)), key=q.__getitem__)],
        "cost": measured[max(range(len(c)), key=c.__getitem__)],
    }
    out = []
    for o, other in (("quality", "cost"), ("cost", "quality")):
        star, donor = sv.complete(champ[o]), sv.complete(champ[other])
        grafts = []
        if star["plan"] != donor["plan"]:
            grafts.append(("plan", switch_plan(star, donor["plan"])))
        if star["plan"] not in ("NoMemory",) and donor["plan"] == star["plan"]:
            for knob in sv.domain(star["plan"]):
                if star.get(knob) != donor.get(knob):
                    try:
                        grafts.append((knob, sv.validate(star | {knob: donor[knob]})))
                    except ValueError:
                        continue
        if not grafts:
            continue
        gm, _ = panel_posterior(model, ctx, [star] + [g for _, g in grafts])
        gq = gm[:, 0].tolist()
        gc = [_cost(m) for m in gm]

        def norm(v):
            lo, hi = min(v), max(v)
            return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in v]

        nq, nc = norm(gq), norm(gc)
        own, oth = (nq, nc) if o == "quality" else (nc, nq)
        tau = [oth[j] - oth[0] - max(0.0, own[0] - own[j]) for j in range(1, len(gq))]
        for j in sorted(range(len(tau)), key=lambda j: -tau[j])[:top]:
            out.append(
                {
                    **grafts[j][1],
                    "_label": f"crossover:{o}:{grafts[j][0]}",
                    "_tau": tau[j],
                }
            )
    return out


# ------------------------------------------------------------------ crossover v2 (channel-slots-v1, user 2026-10-20)
# RAG-Stack §4 Pareto-tension crossover, extended: top 4 per direction; anchors = the posterior quality / cost
# champions PLUS every pair of ADJACENT points on the noise-aware relaxed posterior front (each pair grafts in both
# directions, same tau score); single-knob AND two-knob grafts from the donor; ranked by tau, deduped, proposable
# (PROPOSE_FIXED) and hierarchy-valid only; at most XOVER_MAX rows (champion rows always kept).
XOVER_TOP = 4
XOVER_SIZES = (1, 2)
XOVER_MAX = 20
XOVER_Z = 1.0  # relaxed front: dominated only if beaten by > z pooled posterior SDs (= session_bo_agentenv.RELAX_Z)


def _minmax(v: list[float]) -> list[float]:
    lo, hi = min(v), max(v)
    return [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in v]


def relaxed_front_posterior(mean, var, z: float = XOVER_Z) -> list[int]:
    """Indices of the noise-aware relaxed front in the (quality, cost) plane of posterior means: i is dropped only
    if some j is at least as good on both axes and better by more than ``z`` pooled posterior SDs on one. Sorted by
    posterior quality (ascending), so consecutive entries are ADJACENT front points."""
    q = [float(m[0]) for m in mean]
    c = [_cost(m) for m in mean]
    sq = [math.sqrt(max(float(v[0]), 0.0)) for v in var]
    sc = [0.5 * math.sqrt(max(float(v[1]) + float(v[2]), 0.0)) for v in var]

    def dom(j, i):
        if not (q[j] >= q[i] and c[j] >= c[i]):
            return False
        return q[j] > q[i] + z * math.hypot(sq[i], sq[j]) or c[j] > c[
            i
        ] + z * math.hypot(sc[i], sc[j])

    n = len(q)
    front = [i for i in range(n) if not any(j != i and dom(j, i) for j in range(n))]
    return sorted(front, key=lambda i: (q[i], -c[i], i))


def grafts(
    star: dict[str, Any], donor: dict[str, Any], sizes: tuple[int, ...] = XOVER_SIZES
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Hierarchy-valid grafts of 1 (and 2) donor knobs into ``star``. Across plans the plan itself is a knob:
    ``plan`` = switch_plan(star, donor plan); a two-knob graft is the plan switch plus one donor knob active on the
    donor plan. Every graft is canonical (PROPOSE_FIXED) and differs from the star."""
    if star["plan"] == "NoMemory" or donor["plan"] == "NoMemory":
        return []
    star, donor = sv.complete(star), sv.complete(donor)
    raw: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    if star["plan"] != donor["plan"]:
        base = switch_plan(star, donor["plan"])
        raw.append((("plan",), base))
        if 2 in sizes:
            for knob in proposal_domain(donor["plan"]):
                if base.get(knob) != donor.get(knob):
                    raw.append((("plan", knob), base | {knob: donor[knob]}))
    else:
        diff = [k for k in proposal_domain(star["plan"]) if star.get(k) != donor.get(k)]
        for n in sizes:
            for ks in itertools.combinations(diff, n):
                raw.append((ks, star | {k: donor[k] for k in ks}))
    own, out, seen = _key(canon(star)), [], set()
    for ks, g in raw:
        try:
            g = canon(g)
        except ValueError:
            continue  # hierarchy-invalid combination
        k = _key(g)
        if k != own and k not in seen and proposable(g):
            seen.add(k)
            out.append((ks, g))
    return out


def _tension(model, ctx, star, gs, own_axis: str) -> list[float]:
    """RAG-Stack tau of each graft: normalized gain on the OTHER axis minus the normalized loss on the star's own
    axis (min-max over the star and its grafts)."""
    gm, _ = panel_posterior(model, ctx, [star] + [g for _, g in gs])
    nq, nc = _minmax(gm[:, 0].tolist()), _minmax([_cost(m) for m in gm])
    own, oth = (nq, nc) if own_axis == "quality" else (nc, nq)
    return [oth[j] - oth[0] - max(0.0, own[0] - own[j]) for j in range(1, len(gm))]


def crossover_v2(
    model,
    ctx,
    measured: list[dict[str, Any]],
    *,
    top: int = XOVER_TOP,
    sizes: tuple[int, ...] = XOVER_SIZES,
    max_total: int = XOVER_MAX,
    z: float = XOVER_Z,
    avoid: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Richer Pareto-tension crossover (channel-slots-v1). Directions: champion quality <- cost donor and
    champion cost <- quality donor, plus for every adjacent pair (a, b) of the relaxed posterior front (b the higher
    posterior quality): b <- a (own axis quality, ``adj-q``) and a <- b (own axis cost, ``adj-c``). Each direction
    keeps its ``top`` grafts by tau (configurations in ``avoid`` skipped); rows are deduped by key (highest tau kept)
    and capped at ``max_total`` (champion rows first). Labels ``crossover:<anchor-kind>:<knob[+knob]>``."""
    avoid = set(avoid or ())
    cand = [m for m in measured if m["plan"] != "NoMemory"]
    if len(cand) < 2:
        return []
    mean, var = panel_posterior(model, ctx, cand)
    q = mean[:, 0].tolist()
    c = [_cost(m) for m in mean]
    qi = max(range(len(cand)), key=lambda i: (q[i], -i))
    ci = max(range(len(cand)), key=lambda i: (c[i], -i))
    dirs = [("champ-q", qi, ci, "quality"), ("champ-c", ci, qi, "cost")]
    front = relaxed_front_posterior(mean, var, z)
    for a, b in itertools.pairwise(front):
        dirs += [("adj-q", b, a, "quality"), ("adj-c", a, b, "cost")]
    rows: dict[str, dict[str, Any]] = {}
    for kind, si, di, axis in dirs:
        if si == di:
            continue
        gs = [
            (ks, g)
            for ks, g in grafts(cand[si], cand[di], sizes)
            if _key(g) not in avoid
        ]
        if not gs:
            continue
        tau = _tension(model, ctx, cand[si], gs, axis)
        for j in sorted(range(len(gs)), key=lambda j: (-tau[j], _key(gs[j][1])))[:top]:
            ks, g = gs[j]
            k = _key(g)
            row = {
                **g,
                "_label": f"crossover:{kind}:{'+'.join(ks)}",
                "_tau": tau[j],
                "_anchor": kind,
                "_star": _key(cand[si]),
                "_donor": _key(cand[di]),
            }
            if k not in rows or tau[j] > rows[k]["_tau"]:  # dedupe: highest tau kept
                rows[k] = row
    champ = sorted(
        (r for r in rows.values() if r["_anchor"].startswith("champ")),
        key=lambda r: (-r["_tau"], _key(_strip(r))),
    )
    adj = sorted(
        (r for r in rows.values() if r["_anchor"].startswith("adj")),
        key=lambda r: (-r["_tau"], _key(_strip(r))),
    )
    keep = champ[:max_total] + adj[: max(0, max_total - len(champ))]
    return sorted(keep, key=lambda r: (-r["_tau"], _key(_strip(r))))


def _strip(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def novelty(cfg: dict[str, Any], measured: list[dict[str, Any]]) -> int:
    """Hierarchy-aware novelty: the smallest distance to a measured configuration, where a plan difference counts 1
    plus every child knob active on only one of the two plans, and shared active knobs count 1 each if they differ
    (proposal-fixed knobs ignored)."""
    if cfg["plan"] == "NoMemory":
        return 0
    a = sv.complete(cfg)
    da = set(proposal_domain(a["plan"]))
    best = None
    for m in measured:
        if m["plan"] == "NoMemory":
            d = 1 + len(da)
        else:
            b = sv.complete(m)
            db = set(proposal_domain(b["plan"]))
            d = int(a["plan"] != b["plan"]) + len(da ^ db)
            d += sum(a.get(k) != b.get(k) for k in da & db)
        best = d if best is None else min(best, d)
    return 0 if best is None else best


def gap_rectangle(
    front: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Largest empty rectangle between adjacent points of a 2-D (quality, cost) front, both
    maximized and min-max normalized. Returns ((q_lo, q_hi), (c_lo, c_hi)) in raw units."""
    if len(front) < 2:
        return None
    qs, cs = [p[0] for p in front], [p[1] for p in front]
    qr, cr = (max(qs) - min(qs)) or 1.0, (max(cs) - min(cs)) or 1.0
    pts = sorted(front, key=lambda p: p[0])
    best, area = None, -1.0
    for a, b in itertools.pairwise(pts):
        width = (b[0] - a[0]) / qr
        height = (a[1] - b[1]) / cr
        if width * height > area:
            area, best = width * height, ((a[0], b[0]), (b[1], a[1]))
    return best


def landing_probability(model, ctx: Context, cfgs, gap) -> list[float]:

    (q_lo, q_hi), (c_lo, c_hi) = gap
    mean, var = panel_posterior(model, ctx, cfgs)
    out = []
    for m, v in zip(mean.tolist(), var.tolist(), strict=True):
        cm, cv = 0.5 * (m[1] + m[2]), 0.25 * (v[1] + v[2])

        def p_in(mu, var_, lo, hi):
            sd = math.sqrt(max(var_, 1e-12))

            def phi(z):
                return 0.5 * (1 + math.erf(z / math.sqrt(2)))

            return phi((hi - mu) / sd) - phi((lo - mu) / sd)

        out.append(p_in(m[0], v[0], q_lo, q_hi) * p_in(cm, cv, c_lo, c_hi))
    return out


# channel-slots-v2 (user 2026-10-21): slot families over a candidate's producing channels (multi-label sources)
# task-complementary-v1 adds "dls-specialist" (a measured configuration with exclusive per-query HV contribution)
DLS_SLOT_LABELS = (
    "dls-incumbent",
    "dls-random",
    "dls-pool",
    "dls-newplan",
    "dls-specialist",
)
LLM_SLOT_LABELS = ("agent", "agent-gap", "dls-agent")
# channel-slots-v3 (user 2026-10-22): cost-first slots over the UNION of every channel's candidates; the value is the
# model output index of the cost axis (outputs [gain, -ln tokens, -ln latency])
COST_FIRST_SLOTS = {"tokfirst": 1, "latfirst": 2}
COST_FIRST_Z = 1.0  # quality tolerance = champion posterior mean - z x pooled posterior SD (= RELAX_Z / XOVER_Z)
COST_FIRST_BETA = 1.5  # tie-break UCB_beta(gain) (= QFIRST_BETA)
COST_FIRST_RULE = (
    "champion = measured configuration (real observations, believers excluded) with the highest posterior mean "
    "gain; pooled SD s = sqrt(mean posterior gain variance over those measured configurations); tolerant = "
    "candidates with posterior mean gain >= champion mean - z s (z = 1; the candidate's own SD does not widen the "
    "tolerance, so poorly-known cheap rows do not qualify by being uncertain); pick = lowest posterior mean ln cost "
    "(tokens: tokfirst, latency: latfirst) among the tolerant, tie-break UCB_1.5(gain), then key. None tolerant: "
    "argmax P(gain >= champion mean - z s) x max(0, ln cost(champion) - ln cost(candidate)) if positive "
    "(rule 'p-tolerant x cost-improvement'), else the lowest-cost candidate with posterior mean gain >= 0 "
    "('lowest-cost gain>=0'), else the HV-best candidate ('hv')"
)


def cost_first_order(
    mean: list[list[float]],
    var: list[list[float]],
    champ_mean: list[float],
    pooled_sd: float,
    axis: int,
    keys: list[str],
    hv: list[float] | None = None,
    z: float = COST_FIRST_Z,
    beta: float = COST_FIRST_BETA,
) -> tuple[list[int], dict[str, Any]]:
    """channel-slots-v3 token-/latency-first ranking (rule: ``COST_FIRST_RULE``) of candidates with posterior means
    ``mean`` / variances ``var`` (rows [gain, -ln tokens, -ln latency]). ``axis`` = 1 (tokens) or 2 (latency).
    Returns (candidate indices, best first; audit record). Pure (no model)."""
    if axis not in (1, 2):
        raise ValueError(f"cost axis must be 1 or 2, not {axis!r}")
    n = len(mean)
    thr = float(champ_mean[0]) - z * float(pooled_sd)
    sd = [math.sqrt(max(float(v[0]), 0.0)) for v in var]
    lncost = [-float(m[axis]) for m in mean]  # ln cost (lower is better)
    ucb = [float(m[0]) + beta * s for m, s in zip(mean, sd, strict=True)]
    tol = [i for i in range(n) if float(mean[i][0]) >= thr]
    info: dict[str, Any] = {
        "axis": "tokens" if axis == 1 else "latency",
        "champion_mean": [float(x) for x in champ_mean],
        "pooled_sd": float(pooled_sd),
        "z": z,
        "threshold_gain": thr,
        "tolerant": len(tol),
        "candidates": n,
    }
    if tol:
        order = sorted(tol, key=lambda i: (lncost[i], -ucb[i], keys[i]))
        tset = set(tol)
        rest = sorted(
            (i for i in range(n) if i not in tset),
            key=lambda i: (lncost[i], -ucb[i], keys[i]),
        )
        info["rule"] = "tolerant lowest-cost"
        return order + rest, info
    champ_cost = -float(champ_mean[axis])
    score = [
        _norm_cdf((float(mean[i][0]) - thr) / max(sd[i], 1e-12))
        * max(0.0, champ_cost - lncost[i])
        for i in range(n)
    ]
    if max(score, default=0.0) > 0:
        info["rule"] = "p-tolerant x cost-improvement"
        info["score_top"] = max(score)
        return sorted(range(n), key=lambda i: (-score[i], lncost[i], keys[i])), info
    nonneg = [i for i in range(n) if float(mean[i][0]) >= 0.0]
    if nonneg:
        info["rule"] = "lowest-cost gain>=0"
        order = sorted(nonneg, key=lambda i: (lncost[i], -ucb[i], keys[i]))
        nset = set(nonneg)
        return order + [i for i in range(n) if i not in nset], info
    info["rule"] = "hv"
    h = hv if hv is not None else [0.0] * n
    return sorted(range(n), key=lambda i: (-h[i], keys[i])), info


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no torch)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _slot_family(slot: str | None, src: list[str]) -> list[str]:
    """The labels of ``src`` that make a candidate a member of ``slot``'s arbitration set."""
    if slot in COST_FIRST_SLOTS:  # channel-slots-v3: union of all channels
        return list(src)
    if slot == "xover":
        return [
            lab for lab in src if lab.startswith("crossover") or lab == "dls-crossover"
        ]
    if slot == "llm":
        return [lab for lab in src if lab in LLM_SLOT_LABELS]
    if slot == "dls":
        return [lab for lab in src if lab in DLS_SLOT_LABELS]
    return [lab for lab in src if lab in ("sobol", "lhs")]


def _tc_starts(
    tc,
    believers: list[dict[str, Any]],
    f,
    *,
    base: list[tuple[dict[str, Any], str]],
    pool_rows: list[dict[str, Any]],
    specialists: int = 2,
    pool_top: int = 8,
) -> list[tuple[dict[str, Any], str]]:
    """task-complementary-v1 DLS starts = the ordinary dls-slot starts (incumbents, extras, new-plan defaults,
    random draws) with ``dls-pool`` re-ranked by A_TC instead of qLogNEHVI, plus ``specialists`` measured
    configurations with the largest exclusive per-query contribution (``dls-specialist``). First key wins."""
    out: dict[str, tuple[dict[str, Any], str]] = {}

    def put(cfg, label):
        out.setdefault(_key(cfg), (cfg, label))

    for c in tc.specialists(believers, specialists):
        put(c, "dls-specialist")
    for c, label in base:
        if label in DLS_SLOT_LABELS and label != "dls-pool":
            put(c, label)
    if pool_rows:
        vals = f(pool_rows)
        ranked = sorted(
            (i for i in range(len(pool_rows)) if math.isfinite(vals[i])),
            key=lambda i: (-vals[i], _key(pool_rows[i])),
        )
        for i in ranked[:pool_top]:
            put(pool_rows[i], "dls-pool")
    return list(out.values())


def suggest(
    points: list[hb.Point],
    ctx: Context,
    *,
    seed: int = 0,
    agent_rows: list[dict] | None = None,
    forced: bool = False,
    realized_front: list[tuple[float, float]] | None = None,
    exclude: set[str] | None = None,
    n_sobol: int = 256,
    dls_restarts: int = 8,
    oracle: bool = False,
    fitted=None,
    acq_mode: str = "hv",
    beta: float = 1.5,
    budget: tuple[float, float] | None = None,
    dls_extra: list[tuple[dict[str, Any], str]] | None = None,
    xover: dict[str, Any] | None = None,
    slot: str | None = None,
    dls_crossover: bool = False,
    aux: list[hb.Point] | None = None,
    three_channel: bool = False,
    coverage_sampler: str = "sobol",
    tc=None,
    ablation_policy: dict | None = None,
) -> dict[str, Any]:
    """One decision at ``ctx`` (a Context, or a list of Contexts = a panel of agent states).

    For a panel, every acquisition value is the mean over the panel contexts (one configuration
    is run on every panel state), posterior summaries are panel means, and a configuration
    measured at any panel state is not re-proposed. Returns the chosen configuration with its
    channel provenance.

    acq_mode="hv": three-objective (log)NEHVI/EHVI (default). acq_mode="qfirst": quality-first,
    cost as a constraint -- UCB_beta(quality) x P(tokens and latency within ``budget``); use it
    with a model fit on quality_target="improve".

    ``dls_extra``: additional (configuration, label) DLS starts, e.g. the measured quality champion for the
    quality-first slot (whose model's own incumbent is the P(improve) argmax, not the gain champion).

    channel-slots-v1 (user 2026-10-20): ``xover`` = settings of ``crossover_v2`` (top / sizes / max_total / z;
    None = the legacy two-champion crossover). ``dls_crossover``: when no agent row survives (analyst off or
    failed), the crossover rows replace the agent rows as DLS starts (label ``dls-crossover``). ``slot`` restricts
    the ARBITRATION set (the pool is built as usual): ``"xover"`` = argmax of the acquisition over rows produced by
    the crossover or dls-crossover channels (DLS restricted to the crossover starts); ``"sobol"`` = Sobol breadth
    rows only, ranked by the product of the range-normalised posterior SDs of the three objectives, ties broken by
    the hierarchy-aware ``novelty`` (no DLS); None = the whole pool. An empty slot set falls back to the whole pool
    (``slot_fallback`` in the result). Forced decisions ignore ``slot``.

    channel-slots-v2 (user 2026-10-21) adds ``"llm"`` = argmax of the acquisition over the analyst's rows (agent /
    agent-gap) and the DLS endpoints started from them (dls-agent; DLS restricted to those starts; when that set is
    empty the remaining DLS starts run too and the slot falls back to the whole pool, ``analyst_rows`` /
    ``full_pool_dls`` in ``slot_info``), and ``"dls"`` = argmax over the endpoints of the incumbent / random /
    top-pool / new-plan DLS starts (``DLS_SLOT_LABELS``; never agent or crossover starts).

    ``aux`` (aux-obs-v1): auxiliary observations that enter the GP fit only (``use_aux`` must be on); the
    measured set, the same-context baseline, the reference point, crossover anchors and incumbents come from
    ``points`` alone.

    ``tc`` (task-complementary-v1, a ``task_complementary.TCScorer``): only in the ``"dls"`` slot, an active scorer
    replaces the DLS walk score AND the slot arbitration by A_TC (expected per-query HV improvement over the
    completed configurations plus this batch's believers); starts = ``_tc_starts``. An inactive scorer (cold start
    / degraded gate) or a slot without a supported endpoint keeps the qLogNEHVI behaviour (recorded in
    ``slot_info["task_complementary"]``). Every other slot ignores ``tc``."""
    import torch

    if ablation_policy is not None:
        required = ("analyzer", "recombination", "local_refinement")
        if any(type(ablation_policy.get(k)) is not bool for k in required):
            raise ValueError("ablation policy must define boolean channel switches")
        if not ablation_policy["analyzer"]:
            agent_rows = []
    torch.set_num_threads(1)
    exclude = set(exclude or ())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        aux = list(aux or ())
        if aux and not (_aux["on"] and all(is_aux(p) for p in aux)):
            raise ValueError(
                "aux points need use_aux(True) and the aux: point_id prefix"
            )
        if any(is_aux(p) for p in points):
            raise ValueError(
                "auxiliary observations must be passed as aux=, never as points"
            )
        model, y = fitted if fitted is not None else fit(points + aux)
        y = (
            y[: len(points)] if aux else y
        )  # current-run targets (reference point, ranges)
        measured, seen = [], set()
        for p in points:
            k = _key(p.cfg)
            if k not in seen:
                seen.add(k)
                measured.append(
                    sv.complete(p.cfg)
                    if p.cfg["plan"] not in ("NoMemory", "SemRankOnly")
                    else p.cfg
                )
        # Same-context evaluated configurations are never re-proposed.
        ctxs = ctx if isinstance(ctx, list) else [ctx]

        def at(p, c):
            return (
                p.task == c.task
                and p.iter_start == c.iter_start
                and abs(p.start_norm - c.start_norm) < 1e-9
            )

        avoid = {_key(p.cfg) for p in points if any(at(p, c) for c in ctxs)} | exclude
        # hv-ref-anchored-v1: the reference front comes from real measurements only (no kriging believers)
        real = {
            _key(p.cfg): sv.complete(p.cfg)
            if p.cfg["plan"] not in ("NoMemory", "SemRankOnly")
            else p.cfg
            for p in points
            if not str(p.point_id).startswith("believer-")
        }
        front_cfgs = list(real.values()) or None
        acqs, acq_names, n_here, ref_infos = [], [], 0, []
        for j, c in enumerate(ctxs if acq_mode == "hv" else []):
            here, hseen = [], set()
            for p in points:
                if at(p, c) and _key(p.cfg) not in hseen:
                    hseen.add(_key(p.cfg))
                    here.append(p.cfg)
            n_here += len(here)
            info: dict = {}
            a, ref, name = _acq(
                model,
                y,
                c,
                here,
                measured,
                seed + j,
                front_cfgs=front_cfgs,
                ref_info=info,
            )
            ref_infos.append(info)
            acqs.append((a, c))
            acq_names.append(name)
        acq_name = "; ".join(sorted(set(acq_names))) + (
            f" averaged over {len(ctxs)} panel contexts" if len(ctxs) > 1 else ""
        )
        if acq_mode == "qfirst":
            constraint = (
                "no cost constraint"
                if budget is None
                else f"x P(tokens<={budget[0]:.0f}, latency<={budget[1]:.0f}ms per iteration)"
            )
            acq_name = f"quality-first: UCB(beta={beta}) {constraint}" + (
                f" averaged over {len(ctxs)} contexts" if len(ctxs) > 1 else ""
            )
            ref = torch.tensor([float("nan")] * 3)
        elif acq_mode != "hv":
            raise ValueError(f"unknown acq_mode {acq_mode!r}")
        here = [None] * n_here
        # Proposable versions of the measured configurations (legacy SemRankOnly dropped,
        # off-grid v4 defaults such as top_k=4 snapped to the grid).
        prop, pseen = [], set()
        for c in measured:
            s = snap(c) if c["plan"] != "SemRankOnly" else None
            if s is not None and _key(s) not in pseen:
                pseen.add(_key(s))
                prop.append(s)

        cache: dict[str, float] = {}

        def f(cfgs):
            todo = [c for c in cfgs if _key(c) not in cache]
            if todo and acq_mode == "qfirst":
                for cfg, v in zip(
                    todo,
                    quality_first_scores(model, ctxs, todo, budget, beta),
                    strict=True,
                ):
                    cache[_key(cfg)] = v
            elif todo:
                vals = [score(a, c, todo) for a, c in acqs]
                for i, cfg in enumerate(todo):
                    cache[_key(cfg)] = sum(v[i] for v in vals) / len(vals)
            return [cache[_key(c)] for c in cfgs]

        pool: dict[str, tuple[dict[str, Any], str]] = {}
        sources: dict[
            str, list[str]
        ] = {}  # every channel that produced a row (the pool keeps the first label)

        def add(cfg, label):
            cfg = canon(
                cfg
            )  # proposal-side constraint (PROPOSE_FIXED): every pool row is proposable
            k = _key(cfg)
            if k in avoid:
                return
            if k not in pool:
                pool[k] = (
                    sv.complete(cfg) if cfg["plan"] != "NoMemory" else dict(sv.NOMEM),
                    label,
                )
            if label not in sources.setdefault(k, []):
                sources[k].append(label)

        if not three_channel and coverage_sampler != "sobol":
            raise ValueError("LHS requires three_channel=True")
        breadth = coverage_breadth(n_sobol, seed + 100000, coverage_sampler)
        for c in breadth:
            add(c, coverage_sampler)
        agent = []
        guidance = {}
        fixed = propose_fixed()
        for row in agent_rows or []:
            if any(k in row and row[k] != v for k, v in fixed.items()):
                continue  # an agent row violating the proposal-side constraint is rejected, never snapped
            s = snap({k: v for k, v in row.items() if not k.startswith("_")})
            if s is not None:
                agent.append(s)
                if three_channel:
                    guidance[_key(s)] = {
                        k: row[k]
                        for k in ("_priority_knobs", "_guidance_id", "_hypothesis_id")
                        if k in row
                    }
                else:
                    add(s, row.get("_label", "agent"))
        cross = (
            [] if ablation_policy is not None and not ablation_policy["recombination"]
            else crossover_v2(model, ctx, prop, avoid=avoid, **xover)
            if xover is not None else crossover(model, ctx, prop)
        )
        for c in cross:
            add({k: v for k, v in c.items() if not k.startswith("_")}, c["_label"])
        gap = gap_rectangle(realized_front) if forced and realized_front else None
        gap_rows = [
            k
            for k, (_, lab) in pool.items()
            if lab.startswith(("agent-gap", "crossover"))
        ]
        skip_dls = bool(
            gap and gap_rows and not three_channel
        )  # forced decisions rank only gap/crossover rows
        # DLS sources: agent, incumbents (quality champion, cost champion, up to 3 front points),
        # Sobol restarts, top-8 breadth rows.
        mm, _ = panel_posterior(model, ctx, prop)
        qi = max(range(len(prop)), key=lambda i: float(mm[i, 0]))
        ci = max(range(len(prop)), key=lambda i: _cost(mm[i]))
        from botorch.utils.multi_objective.pareto import is_non_dominated

        nd = [i for i, flag in enumerate(is_non_dominated(mm).tolist()) if flag]
        front_idx = sorted(nd, key=lambda i: float(mm[i, 0]))
        spread = (
            [front_idx[int(j * (len(front_idx) - 1) / 2)] for j in range(3)]
            if len(front_idx) >= 3
            else front_idx
        )
        # channel-slots-v1: no agent row this round -> the crossover rows are the DLS seed set instead
        xstarts = (
            [(_strip(c), "dls-crossover") for c in cross]
            if dls_crossover and not agent
            else []
        )
        starts = [(c, "dls-agent") for c in agent] + xstarts
        starts += [(prop[i], "dls-incumbent") for i in dict.fromkeys([qi, ci, *spread])]
        starts += [(c, label) for c, label in dls_extra or []]
        # plan-p1rank-v1: a proposable plan without any observation (added mid-run) gets its plan default as a DLS
        # start (no-op when every plan is measured, e.g. after a round 0 over all plans)
        starts += [(plan_default(p), "dls-newplan") for p in unmeasured_plans(measured)]
        starts += [
            (c, "dls-random")
            for c in coverage_breadth(dls_restarts, seed + 200000, coverage_sampler)
        ]
        bk = list(pool)
        bscore = f([pool[k][0] for k in bk])
        starts += [
            (pool[bk[i]][0], "dls-pool")
            for i in sorted(range(len(bk)), key=lambda i: -bscore[i])[:8]
        ]
        deferred: list[tuple[dict[str, Any], str]] = []
        tc_f, tc_info = None, None
        if slot == "dls" and tc is not None:
            tc_info = {"status": tc.status, "protocol": "task-complementary-v1"}
            if tc.active:
                believers = [
                    p.cfg for p in points if str(p.point_id).startswith("believer-")
                ]
                tc_f, tc_audit = tc.score_fn(believers)
                tc_starts = _tc_starts(
                    tc,
                    believers,
                    tc_f,
                    base=starts,
                    pool_rows=[pool[k][0] for k in bk],
                    specialists=tc.info.get("specialist_starts", 2),
                    pool_top=tc.info.get("pool_starts", 8),
                )
                tc_info.update(
                    audit=tc_audit,
                    believers=[_key(c) for c in believers],
                    starts=[[label, _key(c)] for c, label in tc_starts],
                )
            else:
                tc_info["fallback"] = "qLogNEHVI DLS (response model not active)"
        if slot == "xover":
            starts = xstarts  # the slot ranks crossover / dls-crossover rows only
        elif slot == "sobol":
            starts = []  # the slot ranks Sobol breadth rows only
        elif slot == "llm":
            # channel-slots-v2: DLS from the analyst's rows only; the other starts run only if the analyst family
            # turns out empty (fallback = the best hv candidate of the full pool, as an ordinary hv pick)
            if agent:
                deferred = [s for s in starts if s[1] != "dls-agent"]
                starts = [s for s in starts if s[1] == "dls-agent"]
        elif slot == "dls":
            # channel-slots-v2: incumbent / random / top-pool (and new-plan) starts; never agent or crossover
            starts = (
                tc_starts
                if tc_f is not None
                else [s for s in starts if s[1] in DLS_SLOT_LABELS]
            )
        elif slot in COST_FIRST_SLOTS:
            pass  # channel-slots-v3: every DLS start runs (the union of all channels' candidates)
        elif slot is not None:
            raise ValueError(f"unknown slot {slot!r}")
        dls_log = []

        def run_dls(todo):
            for start, label in todo:
                guide = guidance.get(_key(start), {}) if label == "dls-agent" else {}
                args = (
                    {"priority_knobs": guide.get("_priority_knobs", [])}
                    if three_channel
                    else {}
                )
                end, val, steps = dls(canon(start), tc_f or f, avoid, **args)
                dls_log.append(
                    {
                        "label": label,
                        "steps": steps,
                        "acq": val,
                        "start_key": _key(start),
                        "end_key": _key(end),
                        **guide,
                    }
                )
                add(end, label)

        no_local = ablation_policy is not None and not ablation_policy["local_refinement"]
        run_dls([] if skip_dls or no_local else starts)
        llm_full_dls = False
        if (
            slot == "llm"
            and deferred
            and not skip_dls and not no_local
            and not any(_slot_family("llm", sources[k]) for k in pool)
        ):
            run_dls(deferred)
            llm_full_dls = True
        keys = list(pool)
        cfgs = [pool[k][0] for k in keys]
        labels = [pool[k][1] for k in keys]
        if not cfgs:
            raise RuntimeError("empty candidate pool (every configuration excluded)")
        if not all(proposable(c) for c in cfgs):
            raise AssertionError(
                f"candidate pool violates the proposal-side constraint {fixed}"
            )
        values = f(cfgs)
        mode = "ordinary"
        slot_info: dict[str, Any] = {}

        def family(i):
            return _slot_family(slot, sources.get(keys[i], [labels[i]]))

        members = (
            [i for i in range(len(cfgs)) if family(i)]
            if slot is not None and not (forced and gap)
            else []
        )
        tcv: dict[int, float] = {}
        if tc_f is not None and not (forced and gap):
            # Arbitration is never budget-limited: every DLS endpoint gets its A_TC.
            tcv = dict(zip(members, tc_f([cfgs[i] for i in members], bounded=False)))
            members = [i for i in members if math.isfinite(tcv[i])]
            if not members:
                # No supported fresh DLS endpoint: still arbitrate by A_TC, over the whole candidate pool.
                tcv = dict(enumerate(tc_f(cfgs, bounded=False)))
                members = [i for i in range(len(cfgs)) if math.isfinite(tcv[i])]
                tc_info["pool_arbitration"] = True
                if not members:
                    tc_info["fallback"] = "no supported candidate: whole-pool qLogNEHVI"
        if forced and gap:
            mode = "forced"
            forced_idx = [
                i
                for i, lab in enumerate(labels)
                if (
                    any(s.startswith(("dls-", "crossover")) for s in sources[keys[i]])
                    if three_channel
                    else lab.startswith(("agent-gap", "crossover"))
                )
            ]
            if not forced_idx:
                forced_idx = list(range(len(cfgs)))
            land = landing_probability(model, ctx, [cfgs[i] for i in forced_idx], gap)
            order = [
                forced_idx[j]
                for j in sorted(
                    range(len(forced_idx)),
                    key=lambda j: (-land[j], keys[forced_idx[j]]),
                )
            ]
        elif members:
            mode = f"slot-{slot}"
            if slot in COST_FIRST_SLOTS:  # channel-slots-v3 token- / latency-first
                mcfg = list(real.values()) or measured
                rm, rv = panel_posterior(model, ctx, mcfg)
                ci_ = max(
                    range(len(mcfg)), key=lambda i: (float(rm[i, 0]), _key(mcfg[i]))
                )
                pooled = math.sqrt(float(rv[:, 0].clamp_min(0).mean()))
                cm, cv = panel_posterior(model, ctx, [cfgs[i] for i in members])
                sub, cinfo = cost_first_order(
                    cm.tolist(),
                    cv.tolist(),
                    rm[ci_].tolist(),
                    pooled,
                    COST_FIRST_SLOTS[slot],
                    [keys[i] for i in members],
                    hv=[values[i] for i in members],
                )
                order = [members[j] for j in sub]
                slot_info.update(
                    cinfo, champion=_key(mcfg[ci_]), rule_text=COST_FIRST_RULE
                )
            elif slot == "dls" and tc_f is not None:
                # task-complementary-v1: argmax A_TC over the supported DLS-family endpoints
                order = sorted(members, key=lambda i: (-tcv[i], keys[i]))
                tc_info["top"] = [[keys[i], tcv[i]] for i in order[:10]]
                tc_info["chosen"] = tc.explain(cfgs[order[0]], believers)
            elif slot in (
                "xover",
                "llm",
                "dls",
            ):  # argmax of the acquisition (qLogNEHVI) over the slot family
                order = sorted(members, key=lambda i: (-values[i], keys[i]))
            else:  # sobol: largest range-normalised posterior uncertainty, hierarchy-aware novelty tie-break
                _, mv = panel_posterior(model, ctx, [cfgs[i] for i in members])
                rng = (y.max(dim=0).values - y.min(dim=0).values).clamp_min(1e-12)
                unc = (mv.clamp_min(0).sqrt() / rng).prod(dim=-1).tolist()
                top_u = max(unc) or 1.0
                nov = [novelty(cfgs[i], measured) for i in members]
                rank = sorted(
                    range(len(members)),
                    key=lambda j: (
                        -round(unc[j] / top_u, 9),
                        -nov[j],
                        keys[members[j]],
                    ),
                )
                order = [members[j] for j in rank]
                slot_info["uncertainty"] = [unc[j] for j in rank[:10]]
                slot_info["novelty"] = [nov[j] for j in rank[:10]]
            slot_info["candidates"] = len(members)
            slot_info["source_channel"] = (family(order[0]) or [labels[order[0]]])[0]
        else:
            if slot is not None:
                slot_info = {
                    "candidates": 0,
                    "slot_fallback": "whole pool (empty slot set)",
                }
                if slot == "llm":
                    slot_info["analyst_rows"] = len(agent)
                    slot_info["full_pool_dls"] = llm_full_dls or not agent
            order = sorted(range(len(cfgs)), key=lambda i: (-values[i], keys[i]))
        if not all(math.isfinite(v) or v == -math.inf for v in values):
            raise FloatingPointError("non-finite acquisition")
        pmean, pvar = panel_posterior(model, ctx, [cfgs[i] for i in order[:10]])

        def row(rank, i):
            m, v = pmean[rank].tolist(), pvar[rank].tolist()
            return {
                "config": cfgs[i],
                "channel": labels[i],
                "acq": values[i],
                "pred_quality_per_iter": m[0],
                "pred_tokens_per_iter": math.exp(-m[1]),
                "pred_latency_ms_per_iter": math.exp(-m[2]),
                "sd": [x**0.5 for x in v],
            }

        if tc_info is not None:
            if tc_f is not None and forced and gap:
                # front-gap-v1 stall rounds keep the baseline's gap-landing rule for every slot (spec 5.4)
                tc_info["fallback"] = "stall-forced decision: gap-landing order"
            tc_info["mechanism_active"] = tc_f is not None and "fallback" not in tc_info
            slot_info["task_complementary"] = tc_info
        result = {
            "mode": mode,
            "chosen": row(0, order[0]),
            "top": [row(r, i) for r, i in enumerate(order[:10])],
            "pool_size": len(cfgs),
            "channel_counts": {lab: labels.count(lab) for lab in sorted(set(labels))},
            # candidate pool per plan (plan-p1rank-v1 audit: a newly added plan must reach the pool)
            "plan_counts": {
                p: sum(c["plan"] == p for c in cfgs)
                for p in sorted({c["plan"] for c in cfgs})
            },
            "source_counts": {
                lab: sum(lab in v for v in sources.values())
                for lab in sorted({x for v in sources.values() for x in v})
            },
            "chosen_sources": sources.get(keys[order[0]], [labels[order[0]]]),
            "slot": slot,
            "slot_info": slot_info,
            "crossover": [
                {"key": _key(_strip(c)), "label": c["_label"], "tau": c.get("_tau")}
                for c in cross
            ],
            "n_points": len(points),
            "measured_configs": len(measured),
            "ref_point": ref.tolist(),
            "hv_ref": ref_infos[0] if len(ref_infos) == 1 else ref_infos,
            "dls": dls_log,
            "gap": gap,
            "context": [c.as_dict() for c in ctxs],
            "space": sv.VERSION,
            "acquisition": acq_name
            + (" bo-acq-v3" if xover is not None else " bo-acq-v2"),
            "same_context_measured": len(here),
        }
        if three_channel:

            def channel_family(label):
                if label in ("sobol", "lhs"):
                    return "global-coverage"
                if label.startswith("dls-"):
                    return "local-refinement"
                if label.startswith("crossover"):
                    return "frontier-recombination"
                raise ValueError(f"unexpected three-channel provenance: {label}")

            result["provenance_counts"] = result["channel_counts"]
            families = [channel_family(lab) for lab in labels]
            result["channel_counts"] = {
                fam: families.count(fam)
                for fam in (
                    "global-coverage",
                    "local-refinement",
                    "frontier-recombination",
                )
            }
            result["search_audit"] = {
                "protocol": "three-channel-v1",
                "coverage_sampler": coverage_sampler,
                "first_producer_family": channel_family(labels[order[0]]),
                "attribution": "channel_counts count first producers; family_source_counts include overlaps",
                "family_source_counts": {
                    fam: sum(
                        any(channel_family(s) == fam for s in labs)
                        for labs in sources.values()
                    )
                    for fam in (
                        "global-coverage",
                        "local-refinement",
                        "frontier-recombination",
                    )
                },
                "source_families": sorted(
                    {channel_family(s) for s in sources[keys[order[0]]]}
                ),
                "guided_endpoints": [
                    {**x, "acq": x["acq"] if math.isfinite(x["acq"]) else None}
                    for x in dls_log
                    if "_guidance_id" in x
                ],
            }
        if oracle:
            best_v, best_c = -math.inf, None
            space = [c for c in sv.space() if _key(c) not in avoid and proposable(c)]
            if (
                proposal_rule() is not None
            ):  # semfilter-v2b: the rule's plans in their proposable form
                space = [
                    c
                    for c in {_key(canon(c)): canon(c) for c in sv.space()}.values()
                    if _key(c) not in avoid and proposable(c)
                ]
            for i in range(0, len(space), 2048):
                chunk = space[i : i + 2048]
                for c, v in zip(chunk, f(chunk), strict=True):
                    if v > best_v:
                        best_v, best_c = v, c
            result["oracle"] = {
                "best_acq": best_v,
                "best_config": best_c,
                "space_size": len(space),
                "pool_regret": best_v - values[order[0]]
                if mode == "ordinary"
                else None,
            }
    return result


def summarize(result: dict[str, Any]) -> str:
    return json.dumps(
        {k: result[k] for k in ("mode", "chosen", "pool_size", "channel_counts")},
        default=str,
    )
