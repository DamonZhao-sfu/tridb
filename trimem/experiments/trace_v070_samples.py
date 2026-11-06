"""Per-configuration stratified TRAIN samples for TRACE v0.7.0 (plan 2026-11-05, Amendment A.4).

Every configuration (each init configuration, each BO configuration and the NoMemory baseline) draws its OWN
``size`` (100) TRAIN tasks from the dataset's eligible TRAIN pool:

* Eligible pool = the stratified prepare's pool rule (``trace_joint_stratified_panel.make_stratified_panels``):
  source split ``train``, minus the 200 fixed TEST ids and the pool's validation ids.
* Quotas = the TEST task-type mix with the largest-remainder rule of ``trace_joint_stratified_panel``
  (``largest_remainder``), computed ONCE per run and frozen in the manifest; uncovered TEST types are recorded.
* Draw = ``random.Random(int(digest([seed, key])[:16], 16))``, uniform without replacement inside each type
  (types in sorted order, ids sorted), so a sample is a pure function of (run seed, configuration key, pool).
* Order = stratified interleave (``staged_pruning.stage_order`` with seed ``digest([seed, key, "interleave"])``):
  every prefix (every 20-task staged-pruning stage) follows the quota mix within about one task per type.

Only TEST task *types* are read (never a TEST outcome). Comparisons between configurations are unpaired.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping

from trimem.optimizer import staged_pruning as sp
from trimem.optimizer.plan_schema import digest

POLICY = "per-config-stratified-test-mix-v1"
ORDER_RULE = "stratified-interleave-v1 (staged_pruning.stage_order, seed digest([seed, key, 'interleave']))"
SEED_RULE = "random.Random(int(digest([seed, config_key])[:16], 16)); types sorted; ids sorted"
SAMPLE_SIZE = 100


def eligible_items(pool: Mapping, test_ids, *, dataset: str) -> list[dict]:
    """Eligible TRAIN items, exactly the stratified prepare's rule, sorted by item id."""
    if pool.get("dataset") not in (None, dataset):
        raise ValueError(f"Pool dataset {pool.get('dataset')} is not {dataset}")
    excluded = set(test_ids) | set(pool.get("validation_ids", []))
    items = sorted(
        (r for r in pool["items"] if r["split"] == "train" and r["item_id"] not in excluded),
        key=lambda r: r["item_id"],
    )
    if len({r["item_id"] for r in items}) != len(items):
        raise ValueError("Duplicate eligible TRAIN ids")
    if any(not r.get("stratum") for r in items):
        raise ValueError("Every eligible TRAIN item needs a stratum")
    return items


def sampling_block(pool: Mapping, test_ids, test_types: Mapping, *, seed: int, dataset: str,
                   size: int = SAMPLE_SIZE) -> tuple[dict, dict]:
    """(manifest ``v070.sampling`` block, TRAIN panel file = the whole eligible pool with strata)."""
    from tools.trace_joint_stratified_panel import largest_remainder

    if len(test_ids) != len(set(test_ids)):
        raise ValueError("TEST ids must be distinct")
    if set(test_types) != set(test_ids):
        raise ValueError("A task type is required for every TEST id")
    items = eligible_items(pool, test_ids, dataset=dataset)
    by_type = Counter(r["stratum"] for r in items)
    test_mix = Counter(test_types[t] for t in test_ids)
    covered = {k: v for k, v in test_mix.items() if k in by_type}
    uncovered = {k: v for k, v in test_mix.items() if k not in by_type}
    quota = largest_remainder({k: float(v) for k, v in covered.items()}, size, {k: by_type[k] for k in covered})
    quota = {k: n for k, n in sorted(quota.items()) if n > 0}
    block = {
        "policy": POLICY,
        "seed": int(seed),
        "size": int(size),
        "seed_rule": SEED_RULE,
        "order": ORDER_RULE,
        "eligible_count": len(items),
        "test_type_mix": dict(sorted(test_mix.items())),
        "quota": quota,
        "uncovered_test_types": dict(sorted(uncovered.items())),
        "covered_test_share": sum(covered.values()) / len(test_ids),
        "strata": {r["item_id"]: r["stratum"] for r in items},
    }
    panel = {
        "dataset": dataset,
        "splits": ["train"],
        "seed": int(seed),
        "selection": "eligible TRAIN pool; each configuration draws its own stratified sample "
        f"({POLICY}); never one panel shared by all configurations",
        "eligible_count": len(items),
        "test_type_mix": block["test_type_mix"],
        "quota": quota,
        "uncovered_test_types": block["uncovered_test_types"],
        "covered_test_share": block["covered_test_share"],
        "items": items,
    }
    return block, panel


def validate_sampling(block: Mapping, tasks) -> None:
    if not isinstance(block, Mapping) or block.get("policy") != POLICY:
        raise ValueError(f"v070.sampling.policy must be {POLICY!r}")
    tasks = [str(t) for t in tasks]
    strata = block["strata"]
    if len(set(tasks)) != len(tasks) or set(strata) != set(tasks):
        raise ValueError("v070.sampling.strata must cover exactly the manifest's TRAIN pool (tasks)")
    size, quota = int(block["size"]), dict(block["quota"])
    have = Counter(strata.values())
    if sum(quota.values()) != size or size < 2 or size > len(tasks):
        raise ValueError("v070.sampling quotas must sum to size (2 <= size <= pool)")
    if any(n < 1 or n > have.get(h, 0) for h, n in quota.items()):
        raise ValueError("v070.sampling quota exceeds a stratum of the pool")


def config_sample(block: Mapping, key: str) -> list[str]:
    """The configuration's own ordered TRAIN sample (deterministic in (seed, key, pool))."""
    seed = int(block["seed"])
    by_type: dict[str, list[str]] = {}
    for item, h in block["strata"].items():
        by_type.setdefault(h, []).append(str(item))
    rng = random.Random(int(digest([seed, key])[:16], 16))
    picked: list[str] = []
    for h in sorted(block["quota"]):
        picked += rng.sample(sorted(by_type[h]), int(block["quota"][h]))
    return sp.stage_order(picked, block["strata"], digest([seed, key, "interleave"]))


def sample_sha(tasks) -> str:
    return digest([str(t) for t in tasks])


def prefix_mix(tasks, strata: Mapping, n: int) -> dict[str, int]:
    return dict(sorted(Counter(strata[t] for t in list(tasks)[:n]).items()))


def max_prefix_deviation(block: Mapping, tasks, sizes) -> float:
    """max over prefixes and types of |count - n * quota / size| (stratified-prefix diagnostic)."""
    size, quota = int(block["size"]), block["quota"]
    worst = 0.0
    for n in sizes:
        mix = prefix_mix(tasks, block["strata"], n)
        worst = max(worst, *(abs(mix.get(h, 0) - n * q / size) for h, q in quota.items()))
    return worst
