"""Opt-in Hydra Local Refinement integration with the existing pool profiler.

The original router fit/choice, staged measurements, and crossover generator are
called unchanged. Only TRAIN feature extraction, proposal search and arbitration
are added. Files under the run directory are reproducible profiling artifacts.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from trimem.optimizer import session_pool as pool
from trimem.optimizer.profiling_local import (
    BehaviorArchive,
    LocalObjective,
    behavior_regions,
    group_responses,
    local_search,
    transfer_groups,
)
from trimem.optimizer.profiling_response import (
    ResponseModel,
    TrainingData,
    config_key,
    digest,
    validation_gate,
)
from trimem.optimizer.query_router import features, runtime, sparse

PROTOCOL = "hydra-local-v1"
OPTIONS = {
    "members": 32,
    "trees": 64,
    "panel_size": 256,
    "max_groups": 4,
    "min_group_size": 25,
    "coverage_size": 256,
    "restarts": 8,
    "max_steps": 30,
    "epoch_rounds": 3,
    "max_evaluations_per_restart": 256,
    "prediction_cache_limit": 256,
    "coverage_draw_batches": 16,
    "seed": 20261023,
}


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def config_features(config):
    """Existing numeric/plan encoding plus explicit active-knob mask."""
    encoded = list(pool.sv.features(config))
    knobs = sorted({k for p in pool.RETRIEVAL_PLANS for k in pool.proposal_domain(p)})
    active = (
        set()
        if config["plan"] == "NoMemory"
        else set(pool.proposal_domain(config["plan"]))
    )
    return encoded + [float(k in active) for k in knobs]


def preference_grid():
    weights = {(1.0, 0.0, 0.0)}
    for s in sparse.SLICE_S:
        weights.update([(1 - s, s, 0.0), (1 - s, 0.0, s), (1 - s, s / 2, s / 2)])
    return np.array(sorted(weights))


def profile_rows(profile):
    return [
        {
            "config": c,
            "query_id": q,
            "split": "train",
            "quality": profile.quality[i][q],
            "tokens": profile.tokens[i][q],
            "latency_ms": profile.latency_ms[i][q],
        }
        for i, c in enumerate(profile.configs)
        for q in profile.samples[i]
    ]


def choose_panel(ids, strata, stratum_weights, size, seed):
    """Fixed stratified TRAIN reference panel; no performance-based selection."""
    rng = np.random.default_rng(seed)
    buckets = {
        s: sorted(q for q in ids if strata[q] == s) for s in sorted(stratum_weights)
    }
    buckets = {s: rng.permutation(qs).tolist() for s, qs in buckets.items() if qs}
    chosen = {s: [] for s in buckets}
    target = min(size, len(ids))
    while sum(map(len, chosen.values())) < target:
        available = [s for s in buckets if buckets[s]]
        s = max(available, key=lambda s: (stratum_weights[s] / (len(chosen[s]) + 1), s))
        chosen[s].append(buckets[s].pop())
    panel = sorted(q for qs in chosen.values() for q in qs)
    mass = sum(stratum_weights[s] for s, qs in chosen.items() if qs)
    p = [stratum_weights[strata[q]] / len(chosen[strata[q]]) / mass for q in panel]
    return panel, np.array(p)


class PredictorCache:
    """Bound retained raw ensembles; rejected neighbors retain only scalar scores."""

    def __init__(self, model, qids, limit=256):
        self.model, self.qids, self.limit = model, qids, limit
        self.cache = OrderedDict()

    def __call__(self, configs):
        result = []
        for offset in range(0, len(configs), 16):
            chunk = configs[offset : offset + 16]
            missing = {
                config_key(c): c for c in chunk if config_key(c) not in self.cache
            }
            if missing:
                pred = self.model.predict(list(missing.values()), self.qids)
                for i, key in enumerate(missing):
                    self.cache[key] = pred[:, i].copy()
            result.extend(self.cache[config_key(c)] for c in chunk)
            for c in chunk:
                self.cache.move_to_end(config_key(c))
            while len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        return np.stack(result, axis=1)


def replenish_coverage(
    coverage, measured, pending, k, *, seed=0, sampler="sobol", batches=16
):
    avoid = set(pending) | {pool._key(c) for c in measured} | {pool._key(pool.sv.NOMEM)}
    fresh = {pool._key(c): pool.canon(c) for c in coverage if pool._key(c) not in avoid}
    for attempt in range(batches):
        if len(fresh) >= k:
            break
        for c in pool.coverage_breadth(
            max(256, 4 * k), seed + 7919 * (attempt + 1), sampler
        ):
            if pool._key(c) not in avoid:
                fresh.setdefault(pool._key(c), pool.canon(c))
    return list(fresh.values())


def model_supported(config, observed):
    """Trees cannot identify effects of an unseen plan or unseen active knob range."""
    same_plan = [c for c in observed if c["plan"] == config["plan"]]
    if not same_plan:
        return False
    for knob, value in config.items():
        values = [c[knob] for c in same_plan if knob in c]
        if not values:
            return False
        if isinstance(value, (int, float)) and all(
            isinstance(v, (int, float)) for v in values
        ):
            if not min(values) <= value <= max(values):
                return False
        elif value not in values:
            return False
    return True


def anchor_indices(configs, measured, router, reference_index, limit=8):
    selected = [reference_index]
    keys = {config_key(c): i for i, c in enumerate(configs)}
    representatives = []
    seen_plans = set()
    for i, c in enumerate(configs):
        if c["plan"] not in seen_plans:
            seen_plans.add(c["plan"])
            representatives.append(i)
    representatives += [keys[config_key(c)] for c in router["configs"]]
    for i in representatives:
        if i not in selected and len(selected) < limit:
            selected.append(i)
    flat = measured.mean(axis=0).reshape(len(configs), -1)
    flat = (flat - flat.mean(axis=0)) / np.maximum(flat.std(axis=0), 0.1)
    while len(selected) < min(limit, len(configs)):
        distance = ((flat[:, None] - flat[selected]) ** 2).mean(axis=-1).min(axis=1)
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    return selected


def propose_candidates(
    model,
    data,
    qids,
    p,
    router,
    epoch,
    *,
    round_id,
    k,
    measured_configs,
    pending,
    coverage,
    cross,
    options=None,
    agent_rows=(),
    forced=False,
    router_X=None,
):
    """Pure proposal assembly; epoch is an explicitly persisted mutable snapshot."""
    opts = {**OPTIONS, **(options or {})}
    predicted = PredictorCache(model, qids, opts["prediction_cache_limit"])
    measured = predicted(data.configs)
    indices = {config_key(c): i for i, c in enumerate(data.configs)}
    choices = np.array(
        [
            [
                indices[
                    config_key(
                        runtime.choose_utility(
                            router, (router_X or data.X)[q], tuple(w)
                        )["config"]
                    )
                ]
                for w in preference_grid()
            ]
            for q in qids
        ]
    )
    reference = indices[epoch["reference_key"]]
    norm = router["cost_norm"]
    obj = LocalObjective(
        measured,
        choices,
        p,
        preference_grid(),
        norm,
        reference_index=reference,
        epoch_norm=epoch["norm"],
    )
    if "groups" not in epoch:
        anchors = anchor_indices(data.configs, measured, router, reference)
        responses = (
            obj.utility(measured[:, anchors], epoch=True) - obj.reference[:, None]
        )
        signatures = responses.transpose(0, 2, 1, 3).reshape(
            len(responses), len(qids), -1
        )
        support = [len(np.unique(data.config_ids[data.query_ids == q])) for q in qids]
        groups = group_responses(
            signatures,
            p,
            max_groups=opts["max_groups"],
            min_size=opts["min_group_size"],
            support=support,
        )
        epoch["groups"] = transfer_groups(
            groups,
            obj.utility(measured, epoch=True),
            p,
            min_size=opts["min_group_size"],
        )
        epoch["anchor_keys"] = [config_key(data.configs[i]) for i in anchors]
    obj.labels = np.array(epoch["groups"]["labels"])
    archive = BehaviorArchive(obj, measured, tau=epoch.get("tau"))
    epoch.setdefault("tau", archive.tau)
    candidate_pool, sources = {}, {}
    avoid = (
        set(pending)
        | {pool._key(c) for c in measured_configs}
        | {pool._key(pool.sv.NOMEM)}
    )

    def add(c, source):
        c = pool.canon(c)
        key = pool._key(c)
        if key in avoid or (
            source != "coverage" and not model_supported(c, data.configs)
        ):
            return
        candidate_pool[key] = c
        sources.setdefault(key, [])
        if source not in sources[key]:
            sources[key].append(source)

    coverage = replenish_coverage(
        coverage,
        measured_configs,
        pending,
        k,
        seed=opts["seed"] + 1000 * round_id,
        batches=opts["coverage_draw_batches"],
    )
    for c in coverage:
        add(c, "coverage")
    for c in cross:
        add(pool._strip(c), "xover")
    # No raw analyzer row bypasses local search.
    valid_agent = []
    for row in agent_rows:
        try:
            c = pool.canon(pool._strip(row))
            if pool.proposable(c):
                valid_agent.append(c)
        except ValueError:
            continue
    basis = list(candidate_pool.values())
    local_logs = []
    cache_scores = {}

    def scoring(kind, group=None):
        def score(configs):
            missing = {
                pool._key(c): c
                for c in configs
                if (kind, group, pool._key(c)) not in cache_scores
            }
            items = list(missing.items())
            for offset in range(0, len(items), 16):
                chunk = items[offset : offset + 16]
                raw = predicted([c for _, c in chunk])
                scores = (
                    archive.score(raw)
                    if kind == "archive"
                    else obj.hydra(raw, group=group)
                )
                for (key, _), value in zip(chunk, scores, strict=True):
                    cache_scores[kind, group, key] = float(value)
            return [cache_scores[kind, group, pool._key(c)] for c in configs]

        return score

    groups = list(np.unique(obj.labels))
    archive_enabled = len(groups) > 1
    archive_count = (6 if forced else opts["restarts"] // 2) if archive_enabled else 0
    archive_count = min(archive_count, opts["restarts"])
    group_count = opts["restarts"] - archive_count
    starts = []
    for n in range(group_count):
        g = int(groups[n % len(groups)])
        f = scoring("group", g)
        candidates = [
            c
            for c in data.configs + valid_agent + basis
            if model_supported(c, data.configs)
        ]
        scores = f(candidates)
        # Different passes use different high-score seeds under a fixed total budget.
        ranked = sorted(
            range(len(candidates)), key=lambda i: (-scores[i], pool._key(candidates[i]))
        )
        start = candidates[ranked[(n // len(groups)) % len(ranked)]]
        starts.append((start, f, "group", g))
    if archive_enabled:
        elites = [data.configs[archive.elites[r]] for r in sorted(archive.elites)]
        f = scoring("archive")
        f(basis)
        # Fill with promising breadth points from distinct predicted niches.
        niche_points = {}
        for c in basis:
            if not model_supported(c, data.configs):
                continue
            region = int(
                np.bincount(
                    behavior_regions(obj.descriptors(predicted([c])), archive.tau)[:, 0]
                ).argmax()
            )
            if region not in niche_points or f([c])[0] > f([niche_points[region]])[0]:
                niche_points[region] = c
        seeds = [niche_points[r] for r in sorted(niche_points)] + elites
        if not seeds:
            seeds = data.configs + valid_agent
        for n in range(archive_count):
            starts.append((seeds[n % len(seeds)], f, "archive", None))
    for start, f, kind, group in starts:
        end, trace = local_search(
            start,
            f,
            lambda c: [
                n for n in pool.neighbors(c) if model_supported(n, data.configs)
            ],
            pool._key,
            avoid,
            max_steps=opts["max_steps"],
            max_evaluations=opts["max_evaluations_per_restart"],
        )
        add(end, kind)
        local_logs.append(
            dict(
                source=kind,
                group=group,
                start=pool._key(start),
                end=pool._key(end),
                **trace,
            )
        )
    if len(candidate_pool) < k:
        raise ValueError(
            "bounded proposal budget exhausted; domain exhaustion is unproven"
        )
    keys = list(candidate_pool)
    supported = [model_supported(candidate_pool[key], data.configs) for key in keys]
    raw = predicted([candidate_pool[key] for key in keys])
    batch = obj.baseline.copy()
    chosen, remaining = [], set(range(len(keys)))
    history = list(measured_configs)
    for slot in range(k):
        requested = ("group", "archive", "xover", "coverage")[slot % 4]
        marginal = obj.hydra(raw, baseline=batch)
        eligible = [i for i in sorted(remaining) if requested in sources[keys[i]]]
        fallback = None
        if not eligible:
            eligible = [i for i in sorted(remaining) if supported[i]]
            fallback = "union-hydra"
            if not eligible:
                eligible = sorted(remaining)
                fallback = "coverage-novelty"
        if not eligible:
            raise ValueError("candidate space exhausted before batch completed")
        if (
            requested == "coverage" and fallback is None
        ) or fallback == "coverage-novelty":
            winner = min(
                eligible,
                key=lambda i: (-pool.novelty(candidate_pool[keys[i]], history), i),
            )
        else:
            winner = min(eligible, key=lambda i: (-marginal[i], keys[i]))
        c = candidate_pool[keys[winner]]
        mean = np.einsum("mcqj,q->mcj", raw[:, winner : winner + 1], p).mean(
            axis=(0, 1)
        )
        chosen.append(
            {
                "config": c,
                "channel": "hydra-" + requested,
                "source_channel": sources[keys[winner]][0],
                "acquisition": "expected-positive-query-utility",
                "acq": float(marginal[winner]) if supported[winner] else None,
                "model_supported": supported[winner],
                "pred_quality": float(mean[0]) if supported[winner] else None,
                "pred_tokens": float(mean[1]) if supported[winner] else None,
                "pred_latency_ms": float(mean[2]) if supported[winner] else None,
                "sources": sources[keys[winner]],
                "fallback": fallback,
                "forced": forced,
            }
        )
        remaining.remove(winner)
        history.append(c)
        if supported[winner]:
            batch = obj.include(batch, raw[:, winner : winner + 1])
    audit = {
        "protocol": PROTOCOL,
        "round": round_id,
        "groups": epoch["groups"],
        "archive": archive.audit(),
        "local_search": local_logs,
        "pool_size": len(keys),
        "prediction_cache_size": len(predicted.cache),
        "unsupported_coverage_candidates": sum(
            not model_supported(c, data.configs) for c in coverage
        ),
        "source_counts": {
            s: sum(s in ss for ss in sources.values())
            for s in ("group", "archive", "xover", "coverage")
        },
        "actual_router_choices": choices.tolist(),
        "options": opts,
        "norm_round": norm,
        "norm_epoch": epoch["norm"],
        "gate": epoch["gate"],
    }
    return chosen, audit


def coverage_batch(coverage, measured, pending, k):
    history, choices = list(measured), []
    avoid = set(pending) | {pool._key(c) for c in history} | {pool._key(pool.sv.NOMEM)}
    coverage = replenish_coverage(coverage, measured, pending, k)
    remaining = {
        pool._key(c): pool.canon(c) for c in coverage if pool._key(c) not in avoid
    }
    for _ in range(k):
        if not remaining:
            raise ValueError(
                "bounded coverage draw budget exhausted; domain exhaustion is unproven"
            )
        key = max(remaining, key=lambda key: pool.novelty(remaining[key], history))
        c = remaining.pop(key)
        history.append(c)
        choices.append(
            {
                "config": c,
                "channel": "hydra-coverage",
                "source_channel": "coverage",
                "acquisition": "hierarchical-novelty",
                "forced": False,
            }
        )
    return choices


def propose(
    driver,
    ds,
    task_pool,
    state,
    round_id,
    nm,
    args,
    points,
    k,
    *,
    agent_rows=(),
    forced=False,
):
    """Entry from bo_pool_driver after normal space/observation preparation."""
    directory = driver.run_dir(ds, args.sampling) / "hydra_local"
    directory.mkdir(exist_ok=True, parents=True)
    opts = state["search_design"]["hydra_options"]
    full = driver.full_keys(state)
    estimates = {
        key: e
        for key, e in driver.estimates(
            ds, task_pool, state, state["experiment"]
        ).items()
        if key in full
    }
    if args.nomemory_candidate:
        nomemory = driver.nomemory_candidate(ds, args, state)
        if nomemory is not None:
            estimates[driver.space.key(driver.space.NOMEM)] = nomemory
    measured = [point.cfg for point in points]
    pending = set(driver.samples_of(state))
    coverage = pool.coverage_breadth(
        opts["coverage_size"],
        opts["seed"] + 1000 * round_id,
        state["search_design"]["coverage_sampler"],
    )
    if not estimates:
        fingerprint = digest(
            {
                "empty": True,
                "pending": sorted(pending),
                "measured": [asdict(point) for point in points],
                "agent_rows": list(agent_rows),
                "coverage_sampler": state["search_design"]["coverage_sampler"],
                "options": opts,
                "k": k,
            }
        )
        path = directory / f"round_{round_id:03}.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("input_fingerprint") != fingerprint:
                raise ValueError("same-round profiling inputs changed; refusing replay")
            return prior["picks"]
        picks = coverage_batch(coverage, measured, pending, k)
        write(
            path,
            {
                "status": "cold_start",
                "reason": "no complete profiles",
                "picks": picks,
                "mechanism_active": False,
                "input_fingerprint": fingerprint,
            },
        )
        return picks
    strata, stratum_weights, sizes = driver.pool_meta(task_pool)
    profile = sparse.from_estimates(estimates, strata, stratum_weights, sizes)
    stored = state.setdefault("hydra_local", {})
    if "panel" not in stored:
        panel, p = choose_panel(
            sorted(strata), strata, stratum_weights, opts["panel_size"], opts["seed"]
        )
        stored.update(panel=panel, query_weights=p.tolist())
    qids, p = stored["panel"], np.array(stored["query_weights"])
    all_ids = sorted(set(profile.queries) | set(qids))
    records = driver.feature_records(
        ds, all_ids, task_pool, False, directory / "features_train.json"
    )
    if "feature_spec" not in stored:
        stored["feature_spec"] = features.fit_spec(
            records, tuple(args.groups.split(","))
        )
    X = {r["query_id"]: runtime.transform(stored["feature_spec"], r) for r in records}
    data = TrainingData.from_rows(profile_rows(profile), X, config_features)
    fingerprint = digest(
        {
            "data": data.identity,
            "points": [asdict(point) for point in points],
            "agent_rows": list(agent_rows),
            "coverage_sampler": state["search_design"]["coverage_sampler"],
            "panel": qids,
            "weights": p.tolist(),
            "panel_features": {q: X[q].tolist() for q in qids},
            "pending": sorted(pending),
            "options": opts,
            "router_options": [args.groups, args.trials, args.jobs, args.seed],
            "forced": forced,
            "k": k,
        }
    )
    round_path = directory / f"round_{round_id:03}.json"
    if round_path.exists():
        prior = json.loads(round_path.read_text())
        if prior.get("input_fingerprint") != fingerprint:
            raise ValueError("same-round profiling inputs changed; refusing replay")
        state["hydra_local"] = deepcopy(prior["state_snapshot"])
        return prior["picks"]

    input_path = directory / f"round_{round_id:03}_inputs.json"
    if input_path.exists():
        frozen = json.loads(input_path.read_text())
        if frozen["fingerprint"] != fingerprint:
            raise ValueError("same-round profiling inputs changed before commit")
        stored.clear()
        stored.update(deepcopy(frozen["state_initial"]))
    else:
        write(
            input_path, {"fingerprint": fingerprint, "state_initial": deepcopy(stored)}
        )

    def commit(audit, picks):
        stored["last_round"] = round_id
        audit.update(
            input_fingerprint=fingerprint, picks=picks, state_snapshot=deepcopy(stored)
        )
        write(round_path, audit)
        write(directory / "state_snapshot.json", stored)
        return picks

    if len(data.configs) < 6 or len(np.unique(data.query_ids)) < 60:
        picks = coverage_batch(coverage, measured, pending, k)
        return commit(
            {
                "status": "cold_start",
                "data_hash": data.identity,
                "mechanism_active": False,
            },
            picks,
        )
    # Exactly the current router fitting rule and options; no alternate selector.
    router_records = [r for r in records if r["query_id"] in set(profile.queries)]
    router_spec = features.fit_spec(router_records, tuple(args.groups.split(",")))
    router_X = {r["query_id"]: runtime.transform(router_spec, r) for r in records}
    model, _ = sparse.fit(
        {q: router_X[q] for q in profile.queries},
        profile,
        trials=args.trials,
        n_jobs=args.jobs,
        seed=args.seed,
    )
    write(directory / f"router_{round_id:03}.json", model)
    stored.setdefault("norm_initial", model["cost_norm"])
    ref = stored.get("reference_key")
    if ref is None:
        ref_config = next(
            (c for c in data.configs if c["plan"] == "NoMemory"), data.configs[0]
        )
        ref = stored["reference_key"] = config_key(ref_config)
    epoch = stored.get("epoch")
    refresh = epoch is None or round_id - epoch["start_round"] >= opts["epoch_rounds"]
    recovery_refresh = forced and (
        epoch is None or round_id - epoch["start_round"] >= 2
    )
    if refresh or recovery_refresh:
        gate = validation_gate(
            data,
            preference_grid(),
            stored["norm_initial"],
            seed=opts["seed"],
            members=opts["members"],
            trees=opts["trees"],
        )
        epoch = {
            "start_round": round_id,
            "norm": model["cost_norm"],
            "reference_key": ref,
            "gate": gate,
        }
        stored["epoch"] = epoch
    if epoch["gate"]["status"] != "ready":
        picks = coverage_batch(coverage, measured, pending, k)
        audit = {
            "protocol": PROTOCOL,
            "round": round_id,
            "gate": epoch["gate"],
            "picks": picks,
            "mechanism_active": False,
            "data_hash": data.identity,
            "epoch": epoch,
        }
    else:
        response = ResponseModel.fit(
            data,
            seed=opts["seed"] + round_id,
            members=opts["members"],
            trees=opts["trees"],
        )
        aggregate, _ = pool.fit(points)
        context = driver.bo_module().context(ds, task_pool["pool_hash"])
        # Unchanged existing parent selection, legal grafts and tension filtering.
        cross = pool.crossover_v2(aggregate, context, measured, avoid=pending)
        picks, audit = propose_candidates(
            response,
            data,
            qids,
            p,
            model,
            epoch,
            round_id=round_id,
            k=k,
            measured_configs=measured,
            pending=pending,
            coverage=coverage,
            cross=cross,
            options=opts,
            agent_rows=agent_rows,
            forced=forced,
            router_X=router_X,
        )
        audit.update(
            mechanism_active=True,
            data_hash=data.identity,
            router_hash=digest(model),
            picks=picks,
        )
    return commit(audit, picks)
