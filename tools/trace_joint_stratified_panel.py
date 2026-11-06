"""TRAIN panel stratified to the TEST task-type mix (user 2026-11-04).

``make_panels`` (tools/trace_joint_protocol.py) draws 100 TRAIN tasks uniformly from the
eligible pool, so the TRAIN type mix follows the pool, not TEST. Here each task type gets
a quota proportional to its TEST count (largest-remainder rounding to 100), and tasks are
drawn uniformly without replacement inside each type with a fixed seed.

Only TEST *task types* are read (from ``test_types``: TEST id -> task_type, read once from
the frozen read-only memory clone); no TEST outcome is used. TEST types with no eligible
TRAIN task cannot be covered; their TEST mass is recorded and the quotas are renormalized
over the covered types. If a type has fewer eligible tasks than its quota, the shortfall is
redistributed by the same rule over types with spare tasks.

This module is new (2026-11-04) so the live runs' hashed ``trace_joint_protocol.py`` stays
unchanged; ``prepare_trace_joint_campaign_stratified`` swaps it in for new runs only.
"""

from __future__ import annotations

import random
from collections import Counter

PANEL_SIZE = 100


def largest_remainder(weights: dict[str, float], total: int, caps: dict[str, int]) -> dict[str, int]:
    """Integer quotas proportional to ``weights`` summing to ``total``, each <= its cap."""
    quota = {k: 0 for k in weights}
    active = {k for k, w in weights.items() if w > 0 and caps[k] > 0}
    left = total
    while left > 0 and active:
        mass = sum(weights[k] for k in active)
        share = {k: left * weights[k] / mass for k in active}
        add = {k: min(int(share[k]), caps[k] - quota[k]) for k in active}
        if sum(add.values()) == 0:  # hand out single remainders, largest first
            k = max(sorted(active), key=lambda x: share[x] - int(share[x]))
            add = {k: 1}
        for k, n in add.items():
            quota[k] += n
            left -= n
        active = {k for k in active if quota[k] < caps[k]}
    if left:
        raise ValueError(f"Pool cannot fill {total} TRAIN tasks over the covered types")
    # Every covered TEST type gets at least one TRAIN task; the slot comes from the type
    # whose quota most exceeds its exact proportional share.
    mass = sum(weights.values())
    for k in sorted(k for k, w in weights.items() if w > 0 and caps[k] > 0 and quota[k] == 0):
        donor = max(
            (d for d in sorted(quota) if quota[d] > 1),
            key=lambda d: quota[d] - total * weights[d] / mass,
        )
        quota[donor] -= 1
        quota[k] = 1
    return quota


def make_stratified_panels(pool, test_ids, test_types, *, seed, dataset):
    if len(test_ids) != 200 or len(set(test_ids)) != 200:
        raise ValueError("Expected the fixed 200 distinct TEST identities")
    if set(test_types) != set(test_ids):
        raise ValueError("A task type is required for every TEST id")
    if pool.get("dataset") not in (None, dataset):
        raise ValueError(f"Pool dataset {pool.get('dataset')} is not {dataset}")
    excluded = set(test_ids) | set(pool.get("validation_ids", []))
    eligible = sorted(
        (r for r in pool["items"] if r["split"] == "train" and r["item_id"] not in excluded),
        key=lambda r: r["item_id"],
    )
    if len({r["item_id"] for r in eligible}) != len(eligible):
        raise ValueError("Duplicate eligible TRAIN ids")
    by_type: dict[str, list[dict]] = {}
    for r in eligible:
        by_type.setdefault(r["stratum"], []).append(r)
    test_mix = Counter(test_types[t] for t in test_ids)
    covered = {k: v for k, v in test_mix.items() if k in by_type}
    uncovered = {k: v for k, v in test_mix.items() if k not in by_type}
    quota = largest_remainder(
        {k: float(v) for k, v in covered.items()},
        PANEL_SIZE,
        {k: len(by_type[k]) for k in covered},
    )
    rng = random.Random(seed)
    items = []
    for k in sorted(quota):  # fixed order -> reproducible draw
        items += rng.sample(by_type[k], quota[k])
    rng.shuffle(items)
    train = {
        "dataset": dataset,
        "splits": ["train"],
        "seed": seed,
        "selection": "stratified by TEST task-type mix; uniform without replacement within type; "
        "frozen across configurations",
        "eligible_count": len(eligible),
        "test_type_mix": dict(sorted(test_mix.items())),
        "quota": dict(sorted(quota.items())),
        "uncovered_test_types": dict(sorted(uncovered.items())),
        "covered_test_share": sum(covered.values()) / 200,
        "items": items,
    }
    test = {
        "dataset": dataset,
        "split": "TEST",
        "tasks": list(test_ids),
        "selection": "canonical fixed_test_v1; never used for search",
    }
    return train, test
