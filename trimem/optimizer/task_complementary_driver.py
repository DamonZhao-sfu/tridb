"""task-complementary-v1 driver glue: build the round's TCScorer from real TRAIN rows (bo_pool_driver passes itself
as ``driver``). Training rows = completed configurations + pre-pruning rows of pruned ones + NoMemory; S = completed
configurations + NoMemory. TEST rows are rejected by TrainingData. State lives in ``state["task_complementary"]``.
"""

from __future__ import annotations

import time

import numpy as np

from trimem.optimizer import session_pool as pool
from trimem.optimizer import task_complementary as tc
from trimem.optimizer.profiling_driver import (
    choose_panel,
    config_features,
    profile_rows,
    write,
)
from trimem.optimizer.profiling_response import (
    ResponseModel,
    TrainingData,
    validation_gate,
)
from trimem.optimizer.query_router import features, runtime, sparse

MIN_COMPLETE, MIN_QUERIES = (
    6,
    60,
)  # same cold-start floor as hydra-local-v1 (NoMemory counts as a configuration)


def levels(plan: str, knob: str) -> list:
    return list(pool.proposal_domain(plan).get(knob, ()))


def _features(driver, ds, ids, task_pool, stored, args, directory) -> dict:
    """Router feature records of TRAIN pool items -> frozen-spec vectors (spec fit once, at the first round)."""
    records = driver.feature_records(
        ds, ids, task_pool, False, directory / "features_train.json"
    )
    if "feature_spec" not in stored:
        stored["feature_spec"] = features.fit_spec(
            records, tuple(args.groups.split(","))
        )
    return {
        r["query_id"]: runtime.transform(stored["feature_spec"], r) for r in records
    }


def _estimates(driver, ds, task_pool, state, r) -> dict:
    """Every configuration on its OBSERVED sample prefix (pruned ones included, as the GP sees them); the driver's
    plain ``estimates`` keeps complete samples only."""
    obs = driver.observed_samples(state)
    rows = driver.opt_rows(ds, state["experiment"], state, r)
    rows = driver.restrict_rows(rows, obs, driver.samples_of(state))
    return driver.estimates_on(rows, obs, task_pool, state["experiment"])


def build_scorer(
    driver,
    ds: str,
    task_pool: dict,
    state: dict,
    r: int,
    args,
    *,
    extra: dict | None = None,
    audit_name: str | None = None,
) -> tc.TCScorer:
    started = time.monotonic()
    opts = state["search_design"]["tc_options"]
    directory = driver.run_dir(ds, args.sampling) / "task_complementary"
    directory.mkdir(parents=True, exist_ok=True)
    stored = state.setdefault("task_complementary", {})
    full = driver.full_keys(state)
    est = dict(_estimates(driver, ds, task_pool, state, r))
    est.update(extra or {})  # e.g. a pruning candidate on its newest observed prefix
    # NoMemory is always a member of S (independent of the router's --nomemory_candidate option).
    nm = driver.nomemory_candidate(ds, args, state)
    nm_key = pool._key(pool.sv.NOMEM)
    if nm is not None:
        est[nm_key] = nm
    est = {k: e for k, e in est.items() if e["ids"]}
    strata, weights, sizes = driver.pool_meta(task_pool)
    complete = sorted(k for k in est if k in full or k == nm_key)
    audit = {
        "protocol": tc.PROTOCOL,
        "round": r,
        "complete": len(complete),
        "partial": len(est) - len(complete),
        "nomemory": nm is not None,
    }
    if not est:
        return _finish(
            directory, r, audit, started, None, "cold_start", stored, audit_name
        )
    if nm is None or nm_key not in est:
        audit["reason"] = "NoMemory estimate unavailable (required in S)"
        return _finish(
            directory, r, audit, started, None, "cold_start", stored, audit_name
        )
    profile = sparse.from_estimates(est, strata, weights, sizes)
    rows = profile_rows(profile)
    if "scales" not in stored:
        # Frozen once, explicitly from round-0 configurations + NoMemory (whatever round first builds the scorer).
        round0 = {pool._key(c["config"]) for c in state["rounds"][0]["configs"]} | {
            nm_key
        }
        base = [row for row in rows if pool._key(row["config"]) in round0]
        stored["scales"] = tc.Scales.freeze(base).to_dict()
        stored["scales_rows"] = len(base)
        stored["scales_round"] = r
    scales = tc.Scales.from_dict(stored["scales"])
    if "panel" not in stored:
        panel, p = choose_panel(
            sorted(strata), strata, weights, opts["panel_size"], opts["seed"]
        )
        stored.update(panel=panel, query_weights=p.tolist())
    qids, p = stored["panel"], np.array(stored["query_weights"])
    X = _features(
        driver,
        ds,
        sorted(set(profile.queries) | set(qids)),
        task_pool,
        stored,
        args,
        directory,
    )
    data = TrainingData.from_rows(rows, X, config_features)
    audit.update(
        rows=len(rows),
        queries=int(len(np.unique(data.query_ids))),
        data_hash=data.identity,
        scales=stored["scales"],
    )
    if len(complete) < MIN_COMPLETE or audit["queries"] < MIN_QUERIES:
        return _finish(
            directory, r, audit, started, None, "cold_start", stored, audit_name
        )
    epoch = stored.get("epoch")
    if epoch is None or r - epoch["start_round"] >= opts["epoch_rounds"]:
        gate = validation_gate(
            data,
            np.eye(
                3
            ),  # unit preferences on (Q, L, T) + gate_norm scales = squared error in the frozen y space
            scales.gate_norm(),
            seed=opts["seed"],
            members=opts["members"],
            trees=opts["trees"],
        )
        epoch = stored["epoch"] = {"start_round": r, "gate": gate}
    audit["gate"] = epoch["gate"]
    status = "active" if epoch["gate"]["status"] == "ready" else epoch["gate"]["status"]
    if status != "active":
        return _finish(directory, r, audit, started, None, status, stored, audit_name)
    model = ResponseModel.fit(
        data, seed=opts["seed"] + r, members=opts["members"], trees=opts["trees"]
    )
    s_configs = [est[k]["cfg"] for k in complete]
    scorer = tc.TCScorer(
        model=model,
        qids=list(qids),
        p=p,
        scales=scales,
        s_configs=s_configs,
        observed=list(data.configs),
        levels=levels,
        status="active",
        key=pool._key,
        strata={q: strata[q] for q in qids},
        budget=opts["budget"],
        info={k: opts[k] for k in ("specialist_starts", "pool_starts")},
    )
    audit["s_keys"] = [pool._key(c) for c in s_configs]
    return _finish(directory, r, audit, started, scorer, "active", stored, audit_name)


def is_tc_pick(entry: dict) -> bool:
    """A dls pick whose proposal used an active task-complementary scorer."""
    info = (entry.get("slot_info") or {}).get("task_complementary") or {}
    return entry.get("slot") == "dls" and bool(info.get("mechanism_active"))


def prune_decision(
    driver, ds, task_pool, state, r, args, entry, prefix_ids, rows, stage
):
    """tc-complementary-v1 verdict for a TC pick at checkpoint ``stage``; None = scorer inactive (caller keeps the
    ordinary rule). The model is refit with the pick's observed prefix; S excludes the (incomplete) pick."""
    key = driver.space.key(entry["config"])
    extra = driver.estimates_on(
        rows, {key: list(prefix_ids)}, task_pool, state["experiment"]
    )
    if key not in extra:
        return None
    scorer = build_scorer(
        driver,
        ds,
        task_pool,
        state,
        r,
        args,
        extra=extra,
        audit_name=f"round_{r:03}_prune_{driver.cfg_name(entry['config'])}_{stage:03}_scorer.json",
    )
    if not scorer.active:
        return None
    gains = scorer.objective([]).member_gain(scorer._predict([entry["config"]]))[:, 0]
    return {**tc.prune_verdict(gains), "stage": stage, "prefix": len(prefix_ids)}


def _finish(directory, r, audit, started, scorer, status, stored, audit_name=None):
    audit.update(status=status, seconds=time.monotonic() - started)
    write(directory / (audit_name or f"round_{r:03}_scorer.json"), audit)
    if scorer is not None:
        return scorer
    return tc.TCScorer(
        model=None,
        qids=[],
        p=np.ones(1),
        scales=tc.Scales.from_dict(stored["scales"])
        if "scales" in stored
        else tc.Scales(0.0, 1.0, 0.0, 1.0),
        s_configs=[],
        observed=[],
        levels=levels,
        status=status,
        key=pool._key,
    )
