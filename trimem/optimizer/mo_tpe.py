"""Optuna 4.2.1 MO-TPE adapter using syftr's multivariate/constant-liar choices.

Common driver initialization replaces upstream random startup. Each round rebuilds
an in-memory study from real TRAIN aggregates; RUNNING batch members have no
invented outcomes. Conditional finite domains are categorical Optuna parameters.
"""

from __future__ import annotations

import math
import time

from trimem.optimizer import session_pool as pool


def parameters(cfg):
    cfg = pool.canon(cfg)
    return {
        "plan": cfg["plan"],
        **{k: cfg[k] for k in sorted(pool.proposal_domain(cfg["plan"]))},
    }


def distributions(plan):
    from optuna.distributions import CategoricalDistribution

    return {
        "plan": CategoricalDistribution(list(pool.RETRIEVAL_PLANS)),
        **{
            k: CategoricalDistribution(list(v))
            for k, v in sorted(pool.proposal_domain(plan).items())
        },
    }


def propose_batch(points, k, opts, seed):
    import optuna

    if optuna.__version__ != "4.2.1":
        raise RuntimeError(
            "MO-TPE protocol requires Optuna 4.2.1; use the isolated baseline environment"
        )
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=0,
        n_ei_candidates=opts.tpe_candidates,
        multivariate=True,
        group=False,
        constant_liar=True,
    )
    study = optuna.create_study(directions=["minimize"] * 3, sampler=sampler)
    for p in points:
        params = parameters(p.cfg)
        if pool.sv.key(pool.canon(params)) != pool.sv.key(p.cfg):
            raise ValueError("MO-TPE encoding changed a measured configuration")
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions(p.cfg["plan"]),
                values=[-p.quality, math.log(p.tokens), math.log(p.latency_ms)],
            )
        )
    seen = {pool.sv.key(p.cfg) for p in points}
    out = []
    for i in range(k):
        start = time.perf_counter()
        for attempt in range(1, opts.tpe_max_attempts + 1):
            trial = study.ask()
            plan = trial.suggest_categorical("plan", list(pool.RETRIEVAL_PLANS))
            cfg = pool.canon(
                {
                    "plan": plan,
                    **{
                        key: trial.suggest_categorical(key, list(values))
                        for key, values in sorted(pool.proposal_domain(plan).items())
                    },
                }
            )
            key = pool.sv.key(cfg)
            if key not in seen:
                break
            # A rejected duplicate is not a measured performance observation.
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            raise RuntimeError(
                "MO-TPE exhausted duplicate rejection attempts; no random fallback"
            )
        seen.add(key)
        out.append(
            {
                "config": cfg,
                "channel": "mo_tpe",
                "mode": "mo_tpe",
                "acquisition": "Optuna MO-TPE density ratio; Pareto/HV split and weights",
                "search_audit": {
                    "pending_count": i,
                    "ask_seconds": time.perf_counter() - start,
                    "attempts": attempt,
                    "duplicates_rejected": attempt - 1,
                    "multivariate": True,
                    "group": False,
                    "constant_liar": True,
                    "startup_trials": 0,
                    "n_ei_candidates": opts.tpe_candidates,
                    "encoding": "conditional categorical finite domains",
                    "noise": "aggregate objectives; no explicit heteroscedastic Yvar",
                    "batch": "RUNNING pending; no fabricated observations",
                    "fallback": None,
                },
            }
        )
    return out
