"""Frozen random TRAIN100 and canonical TEST200 protocol helpers."""

import fcntl
import json
import random
from pathlib import Path


def make_panels(pool, test_ids, *, seed, dataset="sciworld"):
    if len(test_ids) != 200 or len(set(test_ids)) != 200:
        raise ValueError("Expected the fixed 200 distinct TEST identities")
    if pool.get("dataset") not in (None, dataset):
        raise ValueError(f"Pool dataset {pool.get('dataset')} is not {dataset}")
    excluded = set(test_ids) | set(pool.get("validation_ids", []))
    eligible = sorted(
        (
            r
            for r in pool["items"]
            if r["split"] == "train" and r["item_id"] not in excluded
        ),
        key=lambda r: r["item_id"],
    )
    if len({r["item_id"] for r in eligible}) != len(eligible) or len(eligible) < 100:
        raise ValueError("Need at least 100 unique eligible source-TRAIN tasks")
    return (
        {
            "dataset": dataset,
            "splits": ["train"],
            "seed": seed,
            "selection": "uniform random without replacement; frozen across configurations",
            "eligible_count": len(eligible),
            "items": random.Random(seed).sample(eligible, 100),
        },
        {
            "dataset": dataset,
            "split": "TEST",
            "tasks": list(test_ids),
            "selection": "canonical fixed_test_v1; never used for search",
        },
    )


def run_holdout(directory, *, evaluator):
    """Select from completed TRAIN evidence, then evaluate a separate TEST ledger."""
    from trimem.experiments.trace_joint_plan_search import (
        atomic_json,
        evaluate_pending,
        get_space,
        observations,
        validate_manifest,
    )
    from trimem.optimizer.joint_search_ledger import SearchLedger

    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    validate_manifest(m)
    h = m["holdout"]
    if not m.get("smoke"):
        from tools.fixed_test_panel_guard import validate_ids

        validate_ids(m["fixed"]["dataset"], h["tasks"])
    if h["selection"] == "deferred":
        return {"status": "test_deferred"}
    space = get_space(m)
    with SearchLedger(directory / "ledger.sqlite", m) as train:
        if train.pending() or len(train.complete_configs()) != m["budget"]:
            raise ValueError("Require complete TRAIN search before TEST selection")
        obs = observations(train, space, m)
    # Costs are minimized, raw quality maximized. No TEST outcome is available here.
    points = [(o.mean[0], -o.mean[1], -o.mean[2]) for o in obs]
    indices = [
        i
        for i, p in enumerate(points)
        if h["selection"] == "all"
        or not any(
            all(a >= b for a, b in zip(q, p)) and any(a > b for a, b in zip(q, p))
            for q in points
        )
    ]
    selected = [obs[i].config for i in indices]
    selection = {
        "policy": h["selection"],
        "train_protocol": m["protocol"],
        "train_keys": [space.key(o.config) for o in obs],
        "configs": selected,
        "train_means": [obs[i].mean for i in indices],
    }
    out = directory / "holdout"
    out.mkdir(exist_ok=True)
    # This manifest is used only by the independent receipt ledger, never by BO.
    test_manifest = {
        **m,
        "tasks": h["tasks"],
        "budget": len(selected),
        "protocol": m["protocol"] + "-test",
        "evaluation_split": "TEST",
        "selected_keys": [space.key(c) for c in selected],
    }
    with (out / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            test.reserve(space, selected)
            pending = test.pending()
        atomic_json(out / "selection.json", selection)
        atomic_json(out / "manifest.json", test_manifest)
        if not m.get("smoke"):
            from tools.trace_joint_live import recover_claims

            recover_claims(out, test_manifest)
            with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
                pending = test.pending()
        evaluate_pending(out, test_manifest, space, pending, evaluator)
        with SearchLedger(out / "ledger.sqlite", test_manifest) as test:
            if test.pending():
                raise RuntimeError("Uncertain TEST claims require evidence recovery")
            complete = test.complete_configs()
        summary = {
            "status": "test_complete",
            "configurations": len(complete),
            "tasks_per_configuration": 200,
            "episodes": 200 * len(complete),
            "results": [
                {
                    "key": e["key"],
                    "config": e["config"],
                    **{
                        metric: sum(r[metric] for r in e["receipts"]) / 200
                        for metric in (
                            "raw_score",
                            "total_tokens",
                            "latency_seconds",
                            "agent_steps",
                        )
                        if all(r.get(metric) is not None for r in e["receipts"])
                    },
                }
                for e in complete
            ],
        }
        atomic_json(out / "summary.json", summary)
        return summary
