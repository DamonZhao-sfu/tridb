"""Tiny-space fixtures for the TRACE v0.7.0 tests (CPU only, synthetic receipts, no Postgres)."""

from __future__ import annotations

import itertools
from pathlib import Path
from unittest import mock

from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_v070_samples import sampling_block
from trimem.optimizer.plan_schema import FIELD_NAMES, digest
from trimem.optimizer.plan_schema_v070 import JointSpaceV070

TINY_V070 = {
    "filter_order": ("none", "before", "after"),
    "graph_impl": ("BFS", "beam"),
    "filter_pre_impl": ("LLM",),
    "filter_post_impl": ("LLM",),
    "rank_impl": ("ColBERT", "none"),
    "retrieval_unit": ("whole",),
    "injection_timing": ("start", "periodic"),
    "seed_count": (8, 32),
    "depth": (2, 3),
    "fanout": (8,),
    "edge_threshold": ("off",),
    "beam_width": (2, 4),
    "sem_threshold_pre": (0.6,),
    "sem_threshold_post": (0.6,),
    "retrieval_interval": (5,),
    "returned_k": (3, 5),
    "trajectory_category": ("All",),
    "graph_direction": ("Both",),
}
STRATA = ("A", "B", "C", "D")
HOLDOUT = [f"tiny_{1000 + i}" for i in range(200)]


def tiny_space_v070(fp: str = "tiny-v070") -> JointSpaceV070:
    return JointSpaceV070(TINY_V070, execution_fingerprint=fp)


def enumerate_space(space) -> list[dict]:
    seen: dict[str, dict] = {}
    for combo in itertools.product(*(space.domains[n] for n in FIELD_NAMES)):
        c = space.canonicalize(dict(zip(FIELD_NAMES, combo)))
        seen.setdefault(space.key(c), c)
    return list(seen.values())


def tiny_pool(n: int = 300) -> dict:
    """Uneven strata (A 40%, B 30%, C 20%, D 10%); ids tiny_0..; a validation id and a TEST id are excluded."""
    cut = (0.4, 0.7, 0.9)
    items = [{"item_id": f"tiny_{i}", "task_key": f"tiny:{i}", "split": "train", "server": "official",
              "stratum": STRATA[sum(i / n >= c for c in cut)]} for i in range(n)]
    items.append({"item_id": HOLDOUT[0], "task_key": "tiny:test", "split": "train", "server": "official",
                  "stratum": "A"})
    return {"dataset": "tiny", "items": items, "validation_ids": ["tiny_7"]}


def tiny_test_types() -> dict:
    """TEST mix A 20%, B 20%, C 40%, D 20% (differs from the pool mix on purpose)."""
    return {t: ("A", "B", "C", "C", "D")[i % 5] for i, t in enumerate(HOLDOUT)}


def tiny_sampling(size: int = 40, seed: int = 3) -> tuple[dict, dict]:
    return sampling_block(tiny_pool(), HOLDOUT, tiny_test_types(), seed=seed, dataset="tiny", size=size)


def synthetic_receipt(config: dict, task: str, manifest: dict) -> dict:
    """Quality up with seed_count 32 and with a Rank; tokens up with depth and Rank; latency up with beam.
    NoMemory: low quality, few tokens, fast."""
    jitter = int(digest([task, "q"])[:2], 16) % 7
    base = {"task_id": task, "split": manifest.get("evaluation_split", "TRAIN"), "protocol": manifest["protocol"],
            "synthetic": True, "agent_steps": 5}
    if config.get("baseline") == "NoMemory":
        return {**base, "raw_score": 5.0 + jitter, "total_tokens": 800.0 + jitter, "latency_seconds": 8.0 + jitter % 3}
    rank = config["rank_impl"] != "none"
    beam = (config.get("beam_width") or 1) if config["graph_impl"] == "beam" else 1
    return {**base,
            "raw_score": 30.0 * (config["seed_count"] == 32) + 25.0 * rank + 10.0 + jitter,
            "total_tokens": 1000.0 * config["depth"] + 600.0 * rank + 100.0 * (config["seed_count"] // 8) + jitter,
            "latency_seconds": 10.0 + 6.0 * beam + 2.0 * config["depth"] + 3.0 * rank + jitter % 3}


def evaluator_for(space, nomemory_key=None):
    def evaluate(config, task, manifest):
        key = nomemory_key if config.get("baseline") == "NoMemory" else space.key(config)
        return {**synthetic_receipt(config, task, manifest), "config_key": key}

    return evaluate


def shared_manifest(space=None, *, size: int = 40, protocol: str = "v070-shared-fixture", seed: int = 3) -> dict:
    space = space or tiny_space_v070()
    block, panel = tiny_sampling(size, seed)
    return {
        "space": space.manifest(), "tasks": [r["item_id"] for r in panel["items"]], "protocol": protocol,
        "smoke": True, "fixed": {"dataset": "tiny", "synthetic": True}, "live": {"artifacts": {}},
        "initial_configs": 10, "new_configs": 0, "budget": 10,
        "initialization": {"method": "sobol_then_llm", "sobol": 5, "llm": 5}, "q": 4, "seed": seed,
        "task_concurrency": 4, "sources": driver.source_identity(),
        "holdout": {"tasks": list(HOLDOUT), "selection": "deferred"}, "v070": {"sampling": block},
    }


def advice_stub(seed: int = 3):
    def advice(sp, obs, settings):
        seen = {sp.key(o.config) for o in obs}
        return sp.sample(5, seed=seed + 555, avoid=seen), {"status": "synthetic-stand-in"}

    return advice


def make_shared_init(tmp_path, *, size: int = 40) -> Path:
    """Prepared + completed v0.7.0 smoke shared init: 10 configs on own samples + NoMemory TRAIN."""
    import tools.prepare_trace_v070_campaign as prep
    from trimem.experiments.trace_v070_baseline import baseline_key, run_baseline
    from trimem.experiments.trace_v070_runtime import v070_patches

    space = tiny_space_v070()
    run = Path(tmp_path) / "shared_init"
    m = prep.shared_prepare_run_v070(run, shared_manifest(space, size=size))
    evaluate = evaluator_for(space, baseline_key(m))
    with v070_patches(m), mock.patch.object(driver, "initial_advice", advice_stub()):
        run_baseline(run, m, "TRAIN", evaluate)
        for _ in range(4):
            if driver.run_round(run, evaluator=evaluate)["status"] == "budget_complete":
                break
        else:
            raise RuntimeError("shared init did not reach budget_complete")
    return run


def child_get_space(manifest, out) -> None:
    """Spawn target WITHOUT the v0.7.0 patches (a fresh child does not inherit the parent's patches)."""
    try:
        space = driver.get_space(manifest)
        Path(out).write_text("ok " + type(space).__name__)
    except Exception as exc:  # noqa: BLE001 - recorded for the parent
        Path(out).write_text("error " + repr(exc))


def child_patched_get_space(manifest, out) -> None:
    """Spawn target that re-enters the patches first (as latent_search_worker_v070 does)."""
    from trimem.experiments.trace_v070_runtime import v070_patches

    with v070_patches(manifest):
        child_get_space(manifest, out)


FAKE_LATENT = {"checkpoint": "smoke.pt", "checkpoint_sha256": "0", "affine": "smoke.json", "affine_sha256": "0",
               "d": 4, "decode": "masked-greedy-v1", "proposal_budget": {"seconds": 300, "calls": 10000},
               "anchors": 16, "h0": 0.4}


def make_arm(tmp_path, shared: Path, *, budget: int = 10, name: str = "arm", latent=None) -> Path:
    """v0.7.0 arm imported from ``shared``. budget=10 -> the pruning episode budget is spent by the import
    (TRAIN complete right away: staged_batch_size == 0)."""
    import json

    import tools.prepare_trace_v070_campaign as prep

    old = json.loads((shared / "manifest.json").read_text())
    arm = Path(tmp_path) / name
    m = {k: v for k, v in old.items() if k not in ("initial_keys", "objective", "imported_observations")}
    m.update(protocol=f"{old['protocol']}-{name}", initial_configs=10, new_configs=budget - 10, budget=budget,
             initialization=None, live={**old["live"], "run_dir": str(arm)})
    saved = prep.STATE["latent"]
    prep.STATE["latent"] = latent or dict(FAKE_LATENT)
    try:
        prep.arm_import_v070(arm, m, shared)
    finally:
        prep.STATE["latent"] = saved
    return arm
