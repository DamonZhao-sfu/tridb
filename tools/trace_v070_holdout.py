"""TEST of a TRACE v0.7.0 run: (TRAIN 3-objective Pareto U BRANE fuzzy Pareto) of the BO configurations, U NoMemory.

``run_pruned_holdout_v070`` replaces ``tools.trace_joint_protocol.run_holdout`` while ``v070_patches`` is active
(also callable directly / from the CLI):

1. ``holdout/driver.lock`` is taken FIRST (fix round 1): nothing in ``holdout/`` is written without it.
2. TRAIN completion = the driver's own rule under pruning (nothing pending, ``staged_batch_size == 0``; without
   pruning: ``budget`` complete configurations); the NoMemory TRAIN ledger must be complete.
   ``--now`` skips the budget rule (nothing pending is still required).
3. Candidates (``trace_v070_candidates``): BO configurations with their FULL own sample and never pruned, plus
   NoMemory; each with its own-task sample means (unpaired).
4. TEST set (user 2026-11-05): sample-mean 3-objective Pareto set (quality max, tokens min, latency min) U BRANE
   fuzzy Pareto set (``brane.fuzzy_pareto``, TAU_ACC 0.02, TAU_COST 0.10; acc = own-sample mean quality,
   ScienceWorld max(score, 0) / 100; cost = 0.5 ln mean tokens + 0.5 ln mean latency) U NoMemory (always, Q4).
   Router / OneConfig candidates = fuzzy set U {NoMemory} (``router_candidates``; NoMemory always, user
   2026-11-05), so every one is TEST-measured.
5. BO configurations -> ``holdout/ledger.sqlite`` (200 fixed TEST tasks); NoMemory -> ``holdout/baselines/nomemory``.
6. ``holdout/selection.json`` records each component separately; ``holdout/summary.json`` has a NoMemory row.
``holdout.selection == "deferred"`` (the shared init) returns ``test_deferred`` without selecting anything.

    python -m tools.trace_v070_holdout --run-dir R [--select-only] [--now]
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

METRICS = ("raw_score", "total_tokens", "latency_seconds", "agent_steps")


def _check_train(directory: Path, m: dict, *, now: bool) -> None:
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments import trace_v070_baseline as tb
    from trimem.experiments.trace_v070_ledger import SampledLedger

    with SampledLedger(directory / "ledger.sqlite", m) as train:
        if train.pending():
            raise ValueError("Require complete TRAIN search before TEST selection (episodes pending)")
        complete = len(train.complete_configs())
        if not now:
            done = (driver.staged_batch_size(train, m) == 0 if m.get("pruning")
                    else complete >= m["budget"])
            if not done:
                raise ValueError("Require complete TRAIN search before TEST selection (budget not reached)")
    if not tb.baseline_complete(directory, "TRAIN"):
        raise ValueError("Require the complete NoMemory TRAIN ledger before TEST selection")


def select(directory, m: dict) -> dict:
    from trimem.experiments import trace_v070_candidates as tc
    from trimem.experiments.trace_v070_ledger import SampledLedger

    directory = Path(directory)
    with SampledLedger(directory / "ledger.sqlite", m) as train:
        entries = train.complete_configs()
    rows = tc.candidates(directory, "TRAIN", include_baselines=True, full_only=True)
    front = tc.pareto([tc.point(r) for r in rows])
    fuzzy = tc.fuzzy_set(rows, m["fixed"]["dataset"])
    b = m["baselines"]["NoMemory"]
    nm = next((r for r in rows if r["baseline"] == "NoMemory"), None)
    tested = sorted(set(front) | set(fuzzy), key=lambda i: i)
    bo = [rows[i] for i in tested if rows[i]["source"] == "bo"]
    full_keys = {r["key"] for r in rows if r["source"] == "bo"}
    keys = [r["key"] for r in rows]
    router = [keys[i] for i in fuzzy] + ([nm["key"]] if nm is not None and nm["key"] not in
                                         [keys[i] for i in fuzzy] else [])
    return {
        "policy": m["holdout"]["selection"],
        "train_protocol": m["protocol"],
        "train_keys": [e["key"] for e in entries],
        "candidate_keys": keys,
        "excluded_pruned": [e["key"] for e in entries if e["key"] not in full_keys],
        "candidate_rule": "never pruned and full OWN TRAIN sample, plus NoMemory; per-candidate own-task "
                          "sample means (unpaired, Amendment A.4)",
        "test_rule": "3-objective sample-mean Pareto U BRANE fuzzy Pareto (TAU_ACC 0.02, TAU_COST 0.10) U NoMemory "
                     "(user 2026-11-05)",
        "pareto_keys": [keys[i] for i in front],
        "fuzzy_keys": [keys[i] for i in fuzzy],
        "router_candidates": router,
        "router_candidate_keys": router,
        "router_rule": "Opt-PerQuery router and OneConfig candidates = BRANE fuzzy Pareto set U {NoMemory} "
                       "(user 2026-11-05: NoMemory is always a router candidate)",
        "brane": {keys[i]: dict(zip(("acc", "cost"), tc.brane_acc_cost(rows[i], m["fixed"]["dataset"])))
                  for i in range(len(rows))},
        "configs": [r["config"] for r in bo],
        "selected_keys": [r["key"] for r in bo],
        "train_means": [list(tc.point(r)) for r in bo],
        "baselines": {"NoMemory": {
            "key": b["key"], "candidate": nm is not None,
            "in_pareto": nm is not None and nm["key"] in [keys[i] for i in front],
            "in_fuzzy": nm is not None and nm["key"] in [keys[i] for i in fuzzy],
            "tested": b["test"] == "always" or (nm is not None and nm["key"] in [keys[i] for i in tested]),
            "test_rule": b["test"], "train_mean": list(tc.point(nm)) if nm else None,
        }},
    }


def _result(e, n) -> dict:
    return {f: sum(r[f] for r in e["receipts"]) / n for f in METRICS
            if all(r.get(f) is not None for r in e["receipts"])}


def _test_bo(out, m, space, selection, evaluator):
    from tools import trace_joint_live as live
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.optimizer.joint_search_ledger import SearchLedger

    h = m["holdout"]
    test_manifest = {**m, "tasks": h["tasks"], "budget": len(selection["configs"]),
                     "protocol": m["protocol"] + "-test", "evaluation_split": "TEST",
                     "selected_keys": selection["selected_keys"]}
    with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
        test.reserve(space, selection["configs"])
    driver.atomic_json(out / "manifest.json", test_manifest)
    if not m.get("smoke"):
        live.recover_claims(out, test_manifest)
    with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
        pending = test.pending()
    driver.evaluate_pending(out, test_manifest, space, pending, evaluator)
    with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
        if test.pending():
            raise RuntimeError("Uncertain TEST claims require evidence recovery")
        return test.complete_configs()


def run_pruned_holdout_v070(directory, *, evaluator, nomemory_evaluator=None, now=False, select_only=False):
    from tools.trace_v070_nomemory import evaluate_nomemory, recover_nomemory_claims
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments.trace_v070_baseline import (
        baseline_entry,
        prepare_baseline,
        run_baseline,
    )
    from trimem.experiments.trace_v070_runtime import v070_patches

    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    with v070_patches(m):
        driver.validate_manifest(m)
        h = m["holdout"]
        if h["selection"] == "deferred":
            return {"status": "test_deferred"}
        if h["selection"] != "train_pareto":
            raise ValueError("v0.7.0 TEST selection is train_pareto (or deferred)")
        if not m.get("smoke"):
            from tools.fixed_test_panel_guard import validate_ids

            validate_ids(m["fixed"]["dataset"], h["tasks"])
        out = directory / "holdout"
        out.mkdir(exist_ok=True)
        with (out / "driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # before any selection / holdout write
            _check_train(directory, m, now=now)
            space = driver.get_space(m)
            selection = select(directory, m)
            driver.atomic_json(out / "selection.json", selection)
            if select_only:
                return {"status": "selected", "selection": selection}
            complete = _test_bo(out, m, space, selection, evaluator)
            nm_row = None
            if selection["baselines"]["NoMemory"]["tested"]:
                nm_eval = nomemory_evaluator or (evaluator if m.get("smoke") else evaluate_nomemory)
                prepare_baseline(directory, m, split="TEST", tasks=h["tasks"])
                run_baseline(directory, m, "TEST", nm_eval,
                             recover=None if m.get("smoke") else recover_nomemory_claims)
                e = baseline_entry(directory, "TEST")
                nm_row = {"key": e["key"], "config": e["config"], "baseline": "NoMemory",
                          **_result(e, len(h["tasks"]))}
            n = len(h["tasks"])
            results = [{"key": e["key"], "config": e["config"], **_result(e, n)} for e in complete]
            summary = {
                "status": "test_complete",
                "configurations": len(complete),
                "baselines_tested": ["NoMemory"] if nm_row else [],
                "tasks_per_configuration": n,
                "episodes": n * (len(complete) + (1 if nm_row else 0)),
                "router_candidate_keys": selection["router_candidate_keys"],
                "results": results + ([nm_row] if nm_row else []),
            }
            driver.atomic_json(out / "summary.json", summary)
            return summary


def main(argv=None) -> None:
    from tools import trace_joint_live as live
    from trimem.experiments.trace_joint_plan_search import owned

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--now", action="store_true", help="skip the budget-complete rule (nothing pending required)")
    a = ap.parse_args(argv)
    with owned(a.run_dir):  # the live service holds queue.lock for its whole life
        res = run_pruned_holdout_v070(a.run_dir, evaluator=live.evaluate, now=a.now, select_only=a.select_only)
    print(json.dumps({k: v for k, v in res.items() if k != "selection"} | (
        {"selection": {k: res["selection"][k] for k in ("selected_keys", "router_candidate_keys", "baselines")}}
        if "selection" in res else {})))


if __name__ == "__main__":
    main()
