"""Shared tiny-space fixtures for the TRACE latent-MOBO tests (CPU only, fast)."""

from __future__ import annotations

import functools
import itertools

import torch

from trimem.optimizer.latent_seq_vae import train_seq_vae
from trimem.optimizer.plan_schema import FIELD_NAMES, JointSpace

torch.set_num_threads(4)

TINY_DOMAINS = {
    "filter_order": ("none", "before", "after"),
    "graph_impl": ("BFS", "beam"),
    "filter_pre_impl": ("LLM",),
    "filter_post_impl": ("LLM",),
    "rank_impl": ("ColBERT", "LLM"),
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
    "graph_direction": ("Forward", "Both"),
}


def tiny_space() -> JointSpace:
    return JointSpace(TINY_DOMAINS, execution_fingerprint="tiny-test")


def enumerate_space(space: JointSpace) -> list[dict]:
    """All canonical configs of the space, deduped by JointSpace.key."""
    seen: dict[str, dict] = {}
    for combo in itertools.product(*(space.domains[n] for n in FIELD_NAMES)):
        c = space.canonicalize(dict(zip(FIELD_NAMES, combo)))
        seen.setdefault(space.key(c), c)
    return list(seen.values())


@functools.cache
def tiny_model(d: int = 4):
    space = tiny_space()
    model, _ = train_seq_vae(
        space, d, n_samples=192, holdout=16, epochs=3, seed=0, layers=1, batch=64, log=lambda *_: None
    )
    model.eval()
    return model


def synthetic_receipt(config: dict, task: str, manifest: dict) -> dict:
    """Deterministic synthetic receipt: quality up with seed_count=32 and LLM rank,
    tokens up with depth, latency up with beam."""
    good = config["seed_count"] == 32
    llm = config["rank_impl"] == "LLM"
    n = int(task[1:]) % 2 if task[1:].isdigit() else 0
    beam = (config.get("beam_width") or 1) if config["graph_impl"] == "beam" else 1
    return {
        "task_id": task,
        "split": manifest.get("evaluation_split", "TRAIN"),
        "protocol": manifest["protocol"],
        "synthetic": True,
        "raw_score": 40 * good + 40 * llm + 10 + n,
        "total_tokens": 1000 * config["depth"] + 100 * (config["seed_count"] // 8),
        "latency_seconds": 10 + 5 * beam + 2 * config["depth"],
        "agent_steps": 3 + config["depth"],
    }


SMOKE_TASKS = [f"t{i}" for i in range(10)]
SMOKE_HOLDOUT = [f"h{i}" for i in range(200)]


def tiny_shared_manifest(space: JointSpace | None = None, *, protocol: str = "shared-fixture") -> dict:
    """Smoke manifest of a 10-config shared-init run (budget 10, 5 Sobol + 5 LLM) on the tiny space."""
    from trimem.experiments import trace_joint_plan_search as driver

    space = space or tiny_space()
    return {
        "space": space.manifest(),
        "tasks": list(SMOKE_TASKS),
        "protocol": protocol,
        "smoke": True,
        "fixed": {"dataset": "tiny"},
        "live": {"artifacts": {}},
        "initial_configs": 10,
        "new_configs": 0,
        "budget": 10,
        "initialization": {"method": "sobol_then_llm", "sobol": 5, "llm": 5},
        "q": 4,
        "seed": 3,
        "task_concurrency": 2,
        "sources": driver.source_identity(),
        "holdout": {"tasks": list(SMOKE_HOLDOUT), "selection": "deferred"},
    }


def make_shared_init(tmp_path, *, forward: bool = False):
    """Smoke shared-init run (10 complete configs + objective.json) via the normal driver path.

    No Postgres. ``driver.initial_advice`` is patched (unittest.mock) to return 5 valid tiny-space
    configs; ``forward=True`` makes them Forward-direction configs (the v0.5.0 label that the
    stock importer would relabel to Both)."""
    from pathlib import Path
    from unittest import mock

    from trimem.experiments import trace_joint_plan_search as driver

    space = tiny_space()
    run = Path(tmp_path) / "shared_init"
    manifest = driver.prepare_run(run, tiny_shared_manifest(space))
    sobol = {n for n in manifest["initial_keys"]}

    def advice(sp, obs, settings):
        seen = sobol | {sp.key(o.config) for o in obs}
        pool = [c for c in enumerate_space(sp) if sp.key(c) not in seen]
        if forward:
            fwd = [c for c in pool if c["graph_direction"] == "Forward"]
            rest = [c for c in pool if c["graph_direction"] != "Forward"]
            picks = fwd[:2] + rest[:3]
        else:
            picks = [c for c in pool if c["graph_direction"] == "Both"][:5]
        return picks, {"status": "ok"}

    def evaluator(config, task, m):  # real receipts carry the config key (the importer restates it)
        return {**synthetic_receipt(config, task, m), "config_key": space.key(config)}

    with mock.patch.object(driver, "initial_advice", advice):
        driver.run_round(run, evaluator=evaluator)
        status = driver.run_round(run, evaluator=evaluator)
        assert status["complete_configs"] == 10, status
        assert driver.run_round(run, evaluator=evaluator)["status"] == "budget_complete"
    assert (run / "objective.json").exists()
    return run
