"""Official SMAC3 RF + ParEGO + EI, adapted to the profiling configuration space.

The driver owns initialization, staged evaluation, budget and persistence. SMAC
only sees completed aggregates; pending batch choices stay RUNNING, never told
with invented outcomes. Rebuilding from the archive each round is intentional.
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

from trimem.optimizer import session_pool as pool


def configspace(seed):
    from ConfigSpace import Categorical, ConfigurationSpace, InCondition
    from ConfigSpace.hyperparameters import OrdinalHyperparameter

    cs = ConfigurationSpace(seed=seed)
    plans = list(pool.RETRIEVAL_PLANS)
    plan = Categorical("plan", plans)
    cs.add(plan)
    domains = {p: pool.proposal_domain(p) for p in plans}
    knobs = sorted({knob for d in domains.values() for knob in d})
    for knob in knobs:
        active = [p for p in plans if knob in domains[p]]
        levels = domains[active[0]][knob]
        if any(tuple(domains[p][knob]) != tuple(levels) for p in active):
            raise ValueError(f"plan-dependent levels need an explicit encoding: {knob}")
        hp = (
            OrdinalHyperparameter(knob, sequence=list(levels))
            if all(isinstance(v, (int, float)) for v in levels)
            else Categorical(knob, list(levels))
        )
        cs.add(hp)
        if len(active) < len(plans):
            cs.add(InCondition(hp, plan, active))
    return cs


def encode(cfg, cs):
    from ConfigSpace import Configuration

    cfg = pool.canon(cfg)
    active = {"plan", *pool.proposal_domain(cfg["plan"])}
    return Configuration(cs, values={k: cfg[k] for k in active})


def decode(cfg):
    # ConfigSpace ordinal levels may be NumPy scalars; protocol JSON needs Python types.
    return pool.canon(
        {
            key: value.item() if hasattr(value, "item") else value
            for key, value in dict(cfg).items()
        }
    )


def selection_path(origin):
    """Preserve SMAC's native random design and exhaustion fallback provenance."""
    if origin == "Random Search (max retries, no candidates)":
        return "random_fallback", "SMAC selector exhausted acquisition challengers"
    if origin == "Random Search":
        return "random_design", None
    return "acquisition", None


def acquisition_maximizer(cs, opts, seed):
    from smac.acquisition.maximizer.local_and_random_search import (
        LocalAndSortedRandomSearch,
    )
    from smac.acquisition.maximizer.local_search import LocalSearch

    class StableLocalSearch(LocalSearch):
        def _get_initial_points(
            self, previous_configs, n_points, additional_start_points
        ):
            # SMAC 2.4.1 deduplicates with a set. Stabilize its output before
            # neighborhood RNG seeds are assigned, including across processes.
            configs = super()._get_initial_points(
                previous_configs, n_points, additional_start_points
            )
            return sorted(configs, key=lambda c: pool.sv.key(decode(c)))

    maximizer = LocalAndSortedRandomSearch(
        cs,
        challengers=opts.max_scored,
        local_search_iterations=opts.restarts,
        seed=seed,
    )
    # Pinned SMAC private composition seam; all search logic stays upstream.
    maximizer._local_search = StableLocalSearch(cs, seed=seed)
    return maximizer


def propose_batch(points, k, opts, seed):
    from smac import HyperparameterOptimizationFacade as HPO, Scenario
    from smac.acquisition.function.expected_improvement import EI
    from smac.model.random_forest.random_forest import RandomForest
    from smac.multi_objective.parego import ParEGO
    from smac.runhistory.dataclasses import TrialInfo, TrialValue
    from smac.runhistory.encoder.encoder import RunHistoryEncoder

    cs = configspace(seed)

    # No real target function can be executed from the search backend.
    def forbidden_target(config, seed):
        raise RuntimeError("SMAC baseline is ask/tell only; use the shared profiler")

    with tempfile.TemporaryDirectory(prefix="tridb-smac-") as directory:
        scenario = Scenario(
            cs,
            objectives=["negative_gain", "log_tokens", "log_latency"],
            deterministic=True,
            n_trials=len(points) + k + 100,
            seed=seed,
            output_directory=Path(directory),
        )
        parego = ParEGO(scenario, rho=0.05, seed=seed)
        model = RandomForest(configspace=cs, log_y=False, n_trees=10, seed=seed)
        smac = HPO(
            scenario,
            forbidden_target,
            model=model,
            multi_objective_algorithm=parego,
            runhistory_encoder=RunHistoryEncoder(scenario),
            acquisition_function=EI(log=False),
            acquisition_maximizer=acquisition_maximizer(cs, opts, seed),
            # SMAC caches the RF while the number of observations is unchanged.
            # Keep one scalarization for the entire unevaluated batch, so weights
            # and the fitted RF cannot disagree between consecutive ask() calls.
            config_selector=HPO.get_config_selector(scenario, retrain_after=k),
            initial_design=HPO.get_initial_design(scenario, n_configs=0),
            intensifier=HPO.get_intensifier(scenario, max_config_calls=1),
            overwrite=True,
            logging_level=False,
        )
        # deterministic=True makes SMAC's trial seed 0, independent of optimizer seed.
        for p in points:
            cfg = encode(p.cfg, cs)
            if pool.sv.key(decode(cfg)) != pool.sv.key(p.cfg):
                raise ValueError(
                    "ConfigSpace encoding changed a measured configuration"
                )
            smac.tell(
                TrialInfo(config=cfg, seed=0),
                TrialValue(
                    cost=[-p.quality, math.log(p.tokens), math.log(p.latency_ms)],
                    time=0.0,
                ),
                save=False,
            )
        seen, out = {pool.sv.key(p.cfg) for p in points}, []
        for i in range(k):
            start = time.perf_counter()
            trial = smac.ask()
            cfg = decode(trial.config)
            key = pool.sv.key(cfg)
            if key in seen:
                raise RuntimeError(
                    "SMAC re-proposed measured/pending configuration; no fallback"
                )
            seen.add(key)
            admission, fallback = selection_path(trial.config.origin)
            out.append(
                {
                    "config": cfg,
                    "channel": "smac_parego",
                    "mode": "parego",
                    "admission_mode": admission,
                    "acquisition": "SMAC3 RandomForest + ParEGO + EI",
                    "search_audit": {
                        "ask_seconds": time.perf_counter() - start,
                        "batch": "one ParEGO scalarization per batch; RUNNING pending, no fantasy outcomes",
                        "pending_count": i,
                        "origin": trial.config.origin,
                        "trial_seed": trial.seed,
                        "parego_rho": 0.05,
                        "parego_weights": None
                        if parego._theta is None
                        else parego._theta.tolist(),
                        "surrogate": type(model).__name__,
                        "trees": 10,
                        "normalization": "SMAC observed min/max per objective before ParEGO",
                        "noise": "RF ensemble uncertainty; no explicit heteroscedastic Yvar",
                        "random_design": "SMAC default probability 0.2",
                        "challengers": opts.max_scored,
                        "local_search_iterations": opts.restarts,
                        "fallback": fallback,
                        "local_start_order": "canonical configuration key; hash-seed independent",
                    },
                }
            )
    return out
