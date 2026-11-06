"""Staged pruning for TRACE v0.7.0: per-configuration own prefixes, unpaired (plan section 7, Amendment A.4).

The stock ``trace_joint_plan_search.staged_evaluate`` runs every BO configuration on prefixes of ONE frozen
order. Here every configuration runs on prefixes of its OWN stratified sample (``config_sample`` is a stratified
interleave, so each 20-task prefix follows the TEST mix). At each checkpoint n (before the last), behind the same
stage barrier, two rules decide on the configuration's exact own n-prefix:

1. Pareto rule (``staged_pruning.decide``, unchanged): its optimistic corner (``driver._stage_obj``: mean +- z SE,
   Agresti-Coull quality floor) vs every fully evaluated, never-pruned configuration's POINT estimate over its own
   100 tasks and every other configuration's PESSIMISTIC corner on ITS own n-prefix. Plan guard
   (``staged_pruning.guard_plans``) keyed by the six executor plans (``plan_schema_v070.executor_plan``; Q5).
   Optional NoMemory comparator (``baselines.NoMemory.prune_comparator``; default off, Q3).
2. Latency cap (fix round 1; replaces the asynchronous pruner process): prune when the mean latency of the own
   n-prefix exceeds ``pruning.latency_cap.factor`` x the latency of the fastest configuration on the sample-mean
   Pareto front of the complete never-pruned BO (retrieval) configurations; NoMemory is EXCLUDED from this front
   (user 2026-11-05). Factor per dataset (ALFWorld 2.0, ScienceWorld 3.0, WebShop 2.0). Executor-plan guard
   (user 2026-11-05): it never prunes the last remaining never-pruned configuration of an executor plan
   (``guarded = "plan-last"``).

Initialization configurations are protected; NoMemory is never pruned (it is not in the main ledger).
Durability (fix round 1): a checkpoint's decisions and prunes are committed in ONE ledger transaction
(``SampledLedger.apply_stage_decisions``); ``pruning_log.jsonl`` is appended afterwards (missing lines are
back-filled from the ledger). On resume a (key, stage) decision already recorded is never recomputed; decisions are
a pure function of the ledger state at the barrier.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from trimem.optimizer import staged_pruning as sp
from trimem.optimizer.plan_schema_v070 import executor_plan

NOMEMORY_TAG = "NoMemory"
FACTORS = {"alfworld": 2.0, "sciworld": 3.0, "webshop": 2.0}
LATENCY_PROTOCOL = "latency-cap-v070-stage-v1"
SMOKE_FACTOR = 2.0


def latency_cap_block(dataset: str) -> dict:
    return {"protocol": LATENCY_PROTOCOL, "factor": FACTORS.get(dataset, SMOKE_FACTOR),
            "front": "sample-mean Pareto front of complete never-pruned BO (retrieval) configurations; NoMemory excluded",
            "plan_guard": "never prunes the last remaining never-pruned configuration of an executor plan",
            "statistic": "mean latency of the configuration's exact own prefix at the stage checkpoint",
            "protect": "initialization configurations; NoMemory is outside the ledger"}


def _own(manifest, key):
    from .trace_v070_ledger import own_tasks

    return own_tasks(manifest, key)


def _nomemory_obj(directory, n, qs, z):
    """(full Obj, n-prefix Obj) of the complete NoMemory TRAIN ledger, or (None, None)."""
    from trimem.experiments import trace_joint_plan_search as driver

    from .trace_v070_baseline import baseline_entry

    e = baseline_entry(directory, "TRAIN")
    if e is None or not e["complete"]:
        return None, None
    tasks = list(e["tasks"])
    return driver._stage_obj(e["receipts"], tasks, qs, z), driver._stage_obj(e["receipts"], tasks[:n], qs, z)


def latency_front(full: dict, nomemory) -> tuple[float | None, list[str]]:
    cands = dict(full)
    if nomemory is not None:
        cands[NOMEMORY_TAG] = nomemory
    front = sp.pareto_front(cands)
    return (min(cands[k].latency for k in front) if front else None), sorted(front)


def decide_checkpoint(directory, manifest, ledger, undecided, n, protected, z, qs) -> dict:
    """{key: record} for ``undecided`` (live configurations whose done set is exactly their own n-prefix)."""
    from trimem.experiments import trace_joint_plan_search as driver

    keys = [k for (k,) in ledger.db.execute("SELECT key FROM configs")]
    bodies = {k: json.loads(b) for k, b in ledger.db.execute("SELECT key, body FROM configs").fetchall()}
    receipts = {k: ledger.receipts(k) for k in keys}
    pruned = {k for (k,) in ledger.db.execute("SELECT DISTINCT config FROM episodes WHERE state='pruned'")}
    own = {k: _own(manifest, k) for k in keys}
    got = {k: {str(x["task_id"]) for x in r} for k, r in receipts.items()}
    full = {k: driver._stage_obj(receipts[k], own[k], qs, z) for k in keys
            if k not in pruned and got[k] == set(own[k])}
    stage_ok = [k for k in keys if set(own[k][:n]) <= got[k]]
    nm_full, nm_stage = _nomemory_obj(directory, n, qs, z)
    comparator = bool(((manifest.get("baselines") or {}).get("NoMemory") or {}).get("prune_comparator"))
    cmp_full = {NOMEMORY_TAG: nm_full} if comparator and nm_full else {}
    cmp_stage = {NOMEMORY_TAG: nm_stage} if comparator and nm_stage else {}
    decisions, objs = {}, {}
    for key in undecided:
        objs[key] = driver._stage_obj(receipts[key], own[key][:n], qs, z)
        partial = {k: driver._stage_obj(receipts[k], own[k][:n], qs, z) for k in stage_ok if k != key}
        decisions[key] = sp.decide(
            objs[key], {**{k: o for k, o in full.items() if k != key}, **cmp_full},
            stage=n, last=False, protected=key in protected, z=z, partial={**partial, **cmp_stage},
        )
    plan_of = {k: executor_plan(bodies[k]) for k in decisions}
    covered = {executor_plan(bodies[k]) for k in full}
    decisions = sp.guard_plans(decisions, objs, plan_of, covered, z)
    cap = (manifest["pruning"].get("latency_cap") or {})
    # user 2026-11-05 (fix round 2): the front reference EXCLUDES NoMemory (BO retrieval configurations only)
    fastest, front = latency_front(full, None) if cap else (None, [])
    lat = {}
    for key, d in decisions.items():
        if cap and fastest is not None and not d.prune:
            limit = float(cap["factor"]) * fastest
            lat[key] = {"protocol": cap["protocol"], "factor": float(cap["factor"]), "latency": objs[key].latency,
                        "fastest_front_latency": fastest, "cap": limit, "front": front, "guarded": None,
                        "prune": key not in protected and objs[key].latency > limit}
    _latency_plan_guard(lat, {k for k, d in decisions.items() if d.prune}, keys, pruned, bodies)
    records = {}
    for key, d in decisions.items():
        lc = lat.get(key)
        reason = "pareto" if d.prune else ("latency-cap" if lc and lc["prune"] else None)
        records[key] = {**d.as_dict(), "pareto_prune": d.prune, "prune": reason is not None,
                        "prune_rule": reason, "executor_plan": plan_of[key], "plan_guard": "executor-plan-v1",
                        "latency_cap": lc}
        if lc and lc["guarded"]:
            records[key]["guarded"] = lc["guarded"]
    return records


def _latency_plan_guard(lat: dict, pareto_pruned: set, keys, pruned: set, bodies: dict) -> None:
    """User 2026-11-05 (fix round 2): the latency cap never prunes the last remaining (never-pruned) configuration
    of an executor plan. If every surviving configuration of a plan would be pruned at this checkpoint (Pareto or
    latency), the latency-capped one with the lowest prefix latency (then key) is kept: ``guarded = "plan-last"``."""
    leaving = pareto_pruned | {k for k, v in lat.items() if v["prune"]}
    by_plan: dict[str, list[str]] = {}
    for k in keys:
        if k not in pruned:
            by_plan.setdefault(executor_plan(bodies[k]), []).append(k)
    for members in by_plan.values():
        capped = [k for k in members if k in lat and lat[k]["prune"]]
        if capped and all(k in leaving for k in members):
            keep = min(capped, key=lambda k: (lat[k]["latency"], k))
            lat[keep].update(prune=False, guarded="plan-last")


def _backfill_log(directory: Path, recorded: dict) -> int:
    path = directory / "pruning_log.jsonl"
    have = set()
    torn = False
    if path.exists():
        text = path.read_text()
        torn = bool(text) and not text.endswith("\n")
        for line in text.splitlines():
            if line.strip():
                try:  # a crash mid-append can leave a torn last line (review round 2, item 10)
                    d = json.loads(line)
                except ValueError:
                    continue
                have.add((d["key"], int(d["stage"])))
    missing = sorted(k for k in recorded if k not in have)
    with path.open("a") as log:
        if torn:
            log.write("\n")
        for key, stage in missing:
            log.write(json.dumps({"at": time.time(), "key": key, **recorded[(key, stage)], "stage": stage},
                                 default=list) + "\n")
    return len(missing)


def backfill_from_ledger(directory, manifest) -> int:
    """Fix round 2: back-fill ``pruning_log.jsonl`` from the committed ``stage_decisions`` (launcher startup and
    before every round), so a crash between the ledger commit and the log write never loses an audit line even when
    the resumed batch has nothing left to evaluate."""
    from .trace_v070_ledger import SampledLedger

    directory = Path(directory)
    if not (directory / "ledger.sqlite").exists():
        return 0
    with SampledLedger(directory / "ledger.sqlite", manifest) as ledger:
        recorded = ledger.stage_decisions()
    return _backfill_log(directory, recorded) if recorded else 0


def staged_evaluate_v070(directory, manifest, space, pending, evaluator):
    """Drop-in for driver.staged_evaluate on a v0.7.0 manifest (own prefixes; Pareto + latency cap)."""
    from trimem.experiments import trace_joint_plan_search as driver

    from .trace_v070_ledger import SampledLedger

    directory = Path(directory)
    p = manifest["pruning"]
    z, sizes, qs = float(p["z"]), [int(x) for x in p["sizes"]], p.get("quality_scale")
    size = int(manifest["v070"]["sampling"]["size"])
    if sizes[-1] != size:
        raise ValueError(f"last stage {sizes[-1]} must be the sample size {size}")
    batch = sorted({x["key"] for x in pending})
    own = {k: _own(manifest, k) for k in batch}
    protected = set(driver.initialization_keys(directory, manifest, space))
    for i, n in enumerate(sizes):
        with SampledLedger(directory / "ledger.sqlite", manifest) as ledger:
            current = [x for x in ledger.pending()
                       if x["key"] in batch and str(x["task_id"]) in set(own[x["key"]][:n])]
        if current:
            driver.evaluate_pending(directory, manifest, space, current, evaluator)
        with SampledLedger(directory / "ledger.sqlite", manifest) as ledger:
            live = [k for k in batch if ledger.db.execute(
                "SELECT count(*) FROM episodes WHERE config=? AND state='pruned'", (k,)).fetchone()[0] == 0]
            done = {k: {str(r["task_id"]) for r in ledger.receipts(k)} for k in live}
            missing = {k: len(set(own[k][:n]) - d) for k, d in done.items() if not set(own[k][:n]) <= d}
            if missing:
                raise RuntimeError(f"Stage {n} incomplete for {missing}; recover claims first")
        if i == len(sizes) - 1:
            return
        with SampledLedger(directory / "ledger.sqlite", manifest) as ledger:
            recorded = ledger.stage_decisions(n)
            undecided = [k for k in live if done[k] == set(own[k][:n]) and (k, n) not in recorded]
            if undecided:
                records = decide_checkpoint(directory, manifest, ledger, undecided, n, protected, z, qs)
                ledger.apply_stage_decisions(n, records, {
                    k: json.dumps({"rule": r["prune_rule"], "pareto": {f: r[f] for f in ("reason", "dominated_by")},
                                   "latency_cap": r["latency_cap"]}, default=list)
                    for k, r in records.items() if r["prune"]})
            recorded = ledger.stage_decisions()
        _backfill_log(directory, recorded)
