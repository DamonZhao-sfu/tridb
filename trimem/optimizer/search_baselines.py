"""Channel-free profiling search; observations and execution stay with the drivers.

Each round is rebuilt deterministically from real TRAIN aggregates. The drivers
persist the complete proposal batch before executing it; pending choices never
become observations. This module performs no profiling or database IO; the only
LLM-guided method, agentic-autorag-v1 (``agentic_autorag``), replays its recorded
per-round calls instead of re-querying.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version

from trimem.optimizer import session_pool as pool

METHODS = {
    "gp-lognehvi-v1": "gp_lognehvi",
    "smac-parego-v1": "smac_parego",
    "mo-tpe-v1": "mo_tpe",
    "lhs-v1": "lhs",
    "agentic-autorag-v1": "agentic_autorag",  # user 2026-10-27: exploration + whole-pipeline ablation
}
LLM_METHODS = frozenset({"agentic_autorag"})


@dataclass(frozen=True)
class Options:
    seed: int = 20261022
    raw_samples: int = 256
    restarts: int = 8
    max_steps: int = 30
    max_scored: int = 4096
    mc_samples: int = 128
    tpe_candidates: int = 24
    tpe_max_attempts: int = 128

    def __post_init__(self):
        if not 0 <= self.seed < 2**31:
            raise ValueError("search seed must be in [0, 2**31)")
        for key, value in asdict(self).items():
            if key != "seed" and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be a positive integer")


def method_options(opts, method):
    values = asdict(opts)
    if method == "smac_parego":
        return {k: values[k] for k in ("seed", "restarts", "max_scored")}
    if method == "mo_tpe":
        return {k: values[k] for k in ("seed", "tpe_candidates", "tpe_max_attempts")}
    if method in ("lhs", "agentic_autorag"):
        return {"seed": values["seed"]}
    if opts.restarts > opts.raw_samples or opts.raw_samples > opts.max_scored:
        raise ValueError("require restarts <= raw_samples <= max_scored")
    return {k: v for k, v in values.items() if not k.startswith("tpe_")}


def active(settings):
    return settings.get("protocol") in METHODS


def add_arguments(parser):
    for name in asdict(Options()):
        parser.add_argument("--baseline_" + name, type=int, default=None)


def freeze(settings, args):
    """Freeze acquisition controls, package versions and method on run creation."""
    requested = {
        name: getattr(args, "baseline_" + name, None) for name in asdict(Options())
    }
    if not active(settings):
        if any(v is not None for v in requested.values()):
            raise ValueError("baseline options require a baseline search_design")
        return
    if getattr(args, "analyzer_effort", None) is not None:
        raise ValueError("baseline has no analyzer")
    if getattr(args, "plan_p1", False) or os.environ.get(
        "POOL_PLAN_P1", ""
    ).strip() in ("1", "true", "yes", "on"):
        raise ValueError("baseline cannot use the delayed --plan_p1 extension")
    if any(
        getattr(args, name, None)
        for name in ("channel_slots", "random_rounds", "random_rounds_until")
    ) or any(
        os.environ.get(name)
        for name in (
            "POOL_CHANNEL_SLOTS",
            "POOL_RANDOM_ROUNDS",
            "POOL_RANDOM_ROUNDS_UNTIL",
        )
    ):
        raise ValueError(
            "baseline cannot use TRACE channel slots/random-round overrides"
        )
    if settings["initialization_sampler"] != "lhs":
        raise ValueError("baseline common initialization requires randomized LHS")
    old = settings.get("baseline")
    opts = Options(
        **{
            key: value
            if value is not None
            else (old or {}).get("options", {}).get(key, default)
            for key, default in asdict(Options()).items()
            for value in [requested[key]]
        }
    )
    method = METHODS[settings["protocol"]]
    used_options = method_options(opts, method)
    if any(
        requested[name] is not None for name in requested.keys() - used_options.keys()
    ):
        raise ValueError(
            "backend-inapplicable baseline options (including GP-only controls)"
        )
    # the LLM-guided method fits no surrogate: it records only its YAML parser
    packages = ["PyYAML"] if method in LLM_METHODS else ["botorch", "torch"]
    if settings["protocol"] == "smac-parego-v1":
        packages += ["smac", "ConfigSpace", "scikit-learn"]
    if settings["protocol"] == "mo-tpe-v1":
        packages += ["optuna"]
        if version("optuna") != "4.2.1":
            raise ValueError("MO-TPE requires Optuna 4.2.1 in an isolated environment")
    want = {
        "method": method,
        "options": used_options,
        "versions": {name: version(name) for name in packages},
        "objectives": ["gain_vs_nomemory", "-log_mean_tokens", "-log_mean_latency_ms"],
        "analyzer": False,
        **(
            {
                "llm": {
                    "client": "codex-cli",
                    "model": _analyzer_model(),
                    "effort": "low",
                }
            }
            if method in LLM_METHODS
            else {}
        ),
        "channel_pool": False,
        "auxiliary_observations": False,
        "resume": "persisted driver batch; deterministic reconstruction at round boundaries",
    }
    if old is not None and old != want:
        raise ValueError("baseline settings/dependencies changed; use a fresh run")
    settings["baseline"] = want


def _analyzer_model():
    from trimem.optimizer.search_analyzer import MODEL

    return MODEL


def initialize(bo, settings):
    bind_space(settings)
    seed = settings["baseline"]["options"]["seed"]
    return [
        {
            "config": pool.canon(c),
            "channel": "default" if i < len(bo.defaults()) else "lhs",
            "source_channel": "init_global_search",
            "admission_mode": "init",
            "method": METHODS[settings["protocol"]],
        }
        for i, c in enumerate(bo.round0(seed, sampler="lhs"))
    ]


def bind_space(settings):
    """Refuse search-space drift even when a legacy space version was not bumped."""
    spec = json.loads(
        json.dumps(
            {
                "version": pool.sv.VERSION,
                "plans": list(pool.RETRIEVAL_PLANS),
                "domains": {p: pool.proposal_domain(p) for p in pool.RETRIEVAL_PLANS},
                "defaults": [pool.plan_default(p) for p in pool.RETRIEVAL_PLANS],
                "fixed": pool.propose_fixed(),
            },
            sort_keys=True,
        )
    )
    if settings.get("baseline_space", spec) != spec:
        raise ValueError("baseline search space changed; use a fresh run")
    settings["baseline_space"] = spec


def _observations(points, ctx):
    pts = [p for p in points if p.cfg["plan"] != "NoMemory"]
    if len(pts) < 2:
        raise ValueError(
            "baseline requires at least two real same-context observations"
        )
    seen = set()
    for p in pts:
        key = pool.sv.key(pool.sv.validate(p.cfg))
        if key in seen or p.point_id.startswith(("believer-", "aux:", "pending")):
            raise ValueError("duplicate or synthetic observation")
        seen.add(key)
        if (p.task, p.start_norm, p.iter_start, p.h) != (
            ctx.task,
            ctx.start_norm,
            ctx.iter_start,
            ctx.horizon,
        ):
            raise ValueError("baseline requires a single frozen TRAIN context")
        # The split occupies the second identity field; later fields are free
        # run tags (e.g. "test-smac" is a valid TRAIN run tag).
        if re.match(r"^[^:]+:(?:test|heldout)(?:[:0-9-]|$)", p.experiment_id.lower()):
            raise ValueError("TEST observations are forbidden")
        if not all(
            math.isfinite(x) for x in (p.quality, p.tokens, p.latency_ms, *p.noise)
        ):
            raise ValueError("non-finite measurement")
        if p.tokens <= 0 or p.latency_ms <= 0 or any(v < 0 for v in p.noise):
            raise ValueError("invalid measurement/noise variance")
        if not pool.proposable(p.cfg) or p.cfg["plan"] not in pool.RETRIEVAL_PLANS:
            raise ValueError("observation outside the frozen baseline search space")
    if len({p.experiment_id for p in pts}) != 1:
        raise ValueError("mixed experiments")
    return sorted(pts, key=lambda p: pool.sv.key(p.cfg))


def maximize(acq, ctx, avoid, opts, seed):
    """Bounded multistart coordinate ascent on the legal mixed discrete space.

    LHS supplies starts, not a competing TRACE channel. Every legal one-coordinate
    move and plan switch is scored. Half the starts are acquisition-ranked and
    half randomly chosen; all scored candidates remain eligible for the maximum.
    """
    method_options(opts, "gp_lognehvi")
    candidates = {}
    for c in pool.coverage_breadth(opts.raw_samples, seed, sampler="lhs"):
        if c["plan"] != "NoMemory" and pool.sv.key(c) not in avoid:
            candidates[pool.sv.key(c)] = pool.canon(c)
    if not candidates:
        raise RuntimeError("no unmeasured acquisition starts; no random fallback")
    scores = {}

    def evaluate(cfgs):
        fresh = {pool.sv.key(c): c for c in cfgs if pool.sv.key(c) not in scores}
        if len(scores) + len(fresh) > opts.max_scored:
            return False
        values = pool.score(acq, ctx, list(fresh.values()), batch=64) if fresh else []
        for (key, cfg), val in zip(fresh.items(), values, strict=True):
            if not math.isfinite(val):
                raise RuntimeError("non-finite acquisition; no fallback")
            scores[key] = val
            candidates[key] = cfg
        return True

    evaluate(list(candidates.values()))
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    n_top = (opts.restarts + 1) // 2
    starts = ranked[:n_top]
    rest = ranked[n_top:]
    starts += random.Random(seed).sample(
        rest, min(len(rest), opts.restarts - len(starts))
    )
    endings = []
    for start in starts:
        cur, reason, steps = start, "step_limit", 0
        for steps in range(opts.max_steps):
            neighbors = [
                c
                for c in pool.neighbors(candidates[cur])
                if c["plan"] != "NoMemory" and pool.sv.key(c) not in avoid
            ]
            if not evaluate(neighbors):
                reason = "evaluation_limit"
                break
            keys = [cur, *(pool.sv.key(c) for c in neighbors)]
            best = min(keys, key=lambda key: (-scores[key], key))
            if scores[best] <= scores[cur] + 1e-12:
                reason = "local_optimum"
                break
            cur = best
        endings.append({"start": start, "end": cur, "steps": steps + 1, "stop": reason})
    best = min(scores, key=lambda key: (-scores[key], key))
    return (
        candidates[best],
        scores[best],
        {
            "scored": len(scores),
            "restarts": endings,
            "optimizer": "multistart_discrete_coordinate_ascent",
            "fallback": None,
        },
    )


LHS_MAX_ATTEMPTS = 64


def lhs_batch(points, k, seed):
    """Model-free baseline: fresh randomized LHS draws over the same conditional space.

    Observations only exclude already-planned configurations (pruned ones included);
    no objective value influences the choice. Same per-plan stratified sampler and
    bounded deterministic top-ups as the shared LHS initialization.
    """
    seen = {pool.sv.key(p.cfg) for p in points}
    out = []
    for attempt in range(LHS_MAX_ATTEMPTS):
        for cfg in pool.coverage_breadth(k - len(out), seed + attempt, sampler="lhs"):
            key = pool.sv.key(cfg)
            if cfg["plan"] == "NoMemory" or key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "config": pool.canon(cfg),
                    "channel": "lhs",
                    "mode": "lhs",
                    "acquisition": "randomized Latin hypercube; no surrogate",
                    "search_audit": {
                        "pending_count": len(out),
                        "lhs_attempt": attempt,
                        "lhs_seed": seed + attempt,
                        "fallback": None,
                    },
                }
            )
            if len(out) == k:
                return out
    raise RuntimeError("LHS could not fill a distinct batch; no random fallback")


def gp_batch(points, ctx, k, opts, seed):
    import torch

    started = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, y = pool.fit(points)
        fit_seconds = time.perf_counter() - started
        configs = [p.cfg for p in points]
        ref_info = {}
        acq, ref, name = pool._acq(
            model,
            y,
            ctx,
            configs,
            configs,
            seed,
            mc_samples=opts.mc_samples,
            ref_info=ref_info,
        )
        if not name.startswith("qLogNEHVI"):
            raise RuntimeError("GP baseline requires actual qLogNEHVI")
        avoid, pending, out = {pool.sv.key(c) for c in configs}, [], []
        for i in range(k):
            start = time.perf_counter()
            cfg, score, audit = maximize(acq, ctx, avoid, opts, seed + i)
            audit.update(
                fit_seconds=fit_seconds if i == 0 else 0.0,
                acquisition_seconds=time.perf_counter() - start,
                reference=ref_info,
                ref_point=ref.tolist(),
                batch="sequential_X_pending",
                pending_count=i,
                gp_diagnostics=model.gp_diagnostics,
            )
            with torch.no_grad():
                mu = model.posterior(ctx.x([cfg])).mean[0].tolist()
            out.append(
                {
                    "config": cfg,
                    "channel": "gp_lognehvi",
                    "acq": score,
                    "acquisition": name,
                    "mode": "hv",
                    "pred_gain": mu[0],
                    "pred_tokens": math.exp(mu[1] * -1),
                    "pred_latency": math.exp(mu[2] * -1),
                    "search_audit": audit,
                }
            )
            avoid.add(pool.sv.key(cfg))
            pending.append(cfg)
            acq.set_X_pending(ctx.x(pending))
    return out


def propose(points, ctx, k, settings, round_index, **evidence):
    """``evidence`` (run_dir, recent, trials_remaining, chat) is used by the LLM-guided method only."""
    if not active(settings) or k < 1 or round_index < 1:
        raise ValueError("invalid baseline request")
    pts = _observations(points, ctx)
    bind_space(settings)
    opts = Options(**settings["baseline"]["options"])
    seed = (opts.seed + 1000 * round_index) % (2**31)
    method = METHODS[settings["protocol"]]
    start = time.perf_counter()
    if method == "gp_lognehvi":
        out = gp_batch(pts, ctx, k, opts, seed)
    elif method == "lhs":
        out = lhs_batch(pts, k, seed)
    elif method == "agentic_autorag":
        from trimem.optimizer.agentic_autorag import propose_batch

        out = propose_batch(pts, k, opts, seed, round_index=round_index, **evidence)
    elif method == "smac_parego":
        from trimem.optimizer.smac_parego import propose_batch

        out = propose_batch(pts, k, opts, seed)
    else:
        from trimem.optimizer.mo_tpe import propose_batch

        out = propose_batch(pts, k, opts, seed)
    digest = hashlib.sha256(
        json.dumps(
            [
                [
                    p.point_id,
                    pool.sv.key(p.cfg),
                    p.quality,
                    p.tokens,
                    p.latency_ms,
                    p.noise,
                ]
                for p in pts
            ],
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    elapsed = time.perf_counter() - start
    for i, row in enumerate(out):
        row.update(
            method=method,
            source_channel=method,
            admission_mode=row.get("admission_mode", "acquisition"),
            forced=False,
        )
        row["search_audit"].update(
            method=method,
            seed=seed,
            observations=len(pts),
            observation_hash=digest,
            options=method_options(opts, method),
            llm_calls=row["search_audit"].get("llm_calls", 0)
            if method in LLM_METHODS
            else 0,
            search_llm_tokens=row["search_audit"].get("search_llm_tokens", 0)
            if method in LLM_METHODS
            else 0,
            batch_search_seconds=elapsed if i == 0 else 0.0,
        )
    keys = [pool.sv.key(row["config"]) for row in out]
    if (
        len(keys) != k
        or len(set(keys)) != k
        or set(keys) & {pool.sv.key(p.cfg) for p in pts}
    ):
        raise RuntimeError("baseline returned duplicate or measured configurations")
    return out
