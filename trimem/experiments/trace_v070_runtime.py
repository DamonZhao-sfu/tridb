"""Runtime patches that run a TRACE v0.7.0 manifest through the hashed driver (plan 2026-11-05 section 5).

No hashed file is edited. While ``v070_patches(manifest)`` is active, these module attributes are replaced and
restored on exit:

* ``trace_joint_plan_search.JointSpace`` and ``tools.trace_joint_live.JointSpace`` -> ``JointSpaceV070``;
  ``tools.trace_joint_live.runner_config`` -> ``plan_execution_adapter_v070.runner_config`` (``space_patches``).
* driver ``validate_manifest`` -> ``validate_manifest_v070`` (the stock checks on a 100-task view + the v0.7.0
  block; the manifest's ``tasks`` is the eligible TRAIN pool, so the stock validator alone fails closed).
* driver ``aggregate_panel`` -> per-configuration wrapper: each observation is the mean / SE over the
  configuration's OWN sample; all observations share one per-config panel identity (one GP).
* driver ``SearchLedger`` -> ``SampledLedger`` (own sample per reserved configuration).
* driver ``pruning_budget`` -> full cost = sample size (100), not the pool size.
* driver ``staged_evaluate`` -> ``trace_v070_pruning.staged_evaluate_v070`` (own prefixes, executor-plan guard).
* driver ``_search_worker`` -> ``latent_search_worker_v070`` (latent backend; any other backend refuses);
  ``guarded_search`` spawns it, and the child re-enters these patches before the stock latent worker runs.
* ``tools.trace_joint_protocol.run_holdout`` -> ``tools.trace_v070_holdout.run_pruned_holdout_v070``.

One v0.7.0 run per process (re-entry with the same protocol nests; another protocol is refused).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time
from collections.abc import Iterator
from pathlib import Path

from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_joint_plan_search import (
    validate_manifest as _stock_validate,
)
from trimem.optimizer.plan_schema import digest
from trimem.optimizer.plan_schema_v070 import SCHEMA_VERSION, JointSpaceV070

from .trace_v070_samples import POLICY, config_sample, validate_sampling

VERSION = "0.7.0"
PLAN_GUARD = "executor-plan-v1"
LATENT_BACKEND = "latent-mobo-v1"
SHARED_BACKEND = "shared-init-v1"
REQUIRED_ARTIFACTS = (
    "trimem/optimizer/plan_schema_v070.py",
    "trimem/experiments/plan_execution_adapter_v070.py",
    "trimem/experiments/trace_v070_runtime.py",
    "trimem/experiments/trace_v070_samples.py",
    "trimem/experiments/trace_v070_ledger.py",
    "trimem/experiments/trace_v070_baseline.py",
    "trimem/experiments/trace_v070_pruning.py",
    "trimem/experiments/trace_v070_candidates.py",
    "trimem/experiments/trace_v070_marker.py",
    "tools/trace_v070_nomemory.py",
    "tools/trace_v070_nomemory_worker.py",
    "tools/trace_v070_holdout.py",
    "tools/trace_v070_live.py",
    "tools/prepare_trace_v070_campaign.py",
)
_ACTIVE = {"protocol": None, "depth": 0}


def is_v070(manifest) -> bool:
    return (manifest.get("v070") or {}).get("version") == VERSION


def validate_v070_manifest(m) -> None:
    """The v0.7.0 blocks: version / plan guard / sampling, space schema and domains, NoMemory, artifacts."""
    from .trace_v070_baseline import PROTOCOL as NOMEMORY_PROTOCOL
    from .trace_v070_baseline import baseline_key

    v = m.get("v070") or {}
    if v.get("version") != VERSION or v.get("plan_guard") != PLAN_GUARD:
        raise ValueError(f"not a TRACE v{VERSION} manifest (v070.version / plan_guard {PLAN_GUARD})")
    validate_sampling(v.get("sampling"), m["tasks"])
    space = m["space"]
    if space.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"space.schema must be {SCHEMA_VERSION!r}")
    if "LLM" in space["domains"]["rank_impl"]:
        raise ValueError("rank_impl 'LLM' is not in the v0.7.0 space")
    smoke = bool(m.get("smoke"))
    if not smoke and digest(space["domains"]) != digest(JointSpaceV070.live().domains):
        raise ValueError("live v0.7.0 runs require the v0.7.0 LIVE_DOMAINS")
    b = (m.get("baselines") or {}).get("NoMemory") or {}
    if (b.get("protocol") != NOMEMORY_PROTOCOL or b.get("key") != baseline_key(m)
            or b.get("dir") != "baselines/nomemory" or b.get("test") != "always"
            or not isinstance(b.get("prune_comparator"), bool) or not isinstance(b.get("latency_front"), bool)):
        raise ValueError("manifest baselines.NoMemory block is missing or differs from the v0.7.0 protocol")
    if b.get("latency_front") is not False:
        raise ValueError("baselines.NoMemory.latency_front must be false (user 2026-11-05: the latency-cap front "
                         "excludes NoMemory)")
    validate_parity(m)
    if m.get("search_backend") not in (LATENT_BACKEND, SHARED_BACKEND):
        raise ValueError(f"v0.7.0 search_backend must be {LATENT_BACKEND} or {SHARED_BACKEND}")
    if not smoke:
        root = Path(driver.__file__).resolve().parents[2]
        arts = {str(Path(p).resolve()) for p in m["live"]["artifacts"]}
        missing = [p for p in REQUIRED_ARTIFACTS if str((root / p).resolve()) not in arts]
        if missing:
            raise ValueError(f"v0.7.0 modules missing from live.artifacts: {missing}")


CONCURRENCY, AGENT_MAX_TOKENS = 16, 4096


def validate_parity(m) -> None:
    """Fix round 1 (finding 6): the protocol constants shared by TRACE and NoMemory episodes, enforced at the
    manifest boundary (live runs): c16, agent max_tokens 4096, the latency-cap factor of the dataset."""
    from .trace_v070_pruning import FACTORS, LATENCY_PROTOCOL

    cap = (m.get("pruning") or {}).get("latency_cap")
    if m.get("pruning") is not None and (not cap or cap.get("protocol") != LATENCY_PROTOCOL):
        raise ValueError(f"pruning.latency_cap ({LATENCY_PROTOCOL}) is required with staged pruning")
    if m.get("smoke"):
        return
    lv, fx = m["live"], m["fixed"]
    if (m.get("task_concurrency") != CONCURRENCY or fx.get("concurrency") != CONCURRENCY
            or int(lv.get("agent_max_tokens", 0)) != AGENT_MAX_TOKENS
            or int(fx.get("agent_max_tokens", 0)) != AGENT_MAX_TOKENS
            or fx["runtime"]["agent_max_tokens"] != AGENT_MAX_TOKENS):
        raise ValueError(f"v0.7.0 live runs require task_concurrency {CONCURRENCY} and agent max_tokens "
                         f"{AGENT_MAX_TOKENS} (TRACE and NoMemory alike)")
    if len(lv["llm_bases"]) != len(lv["environment_bases"]) or not lv["llm_bases"]:
        raise ValueError("live.llm_bases and live.environment_bases must pair one slot each")
    if cap and float(cap["factor"]) != FACTORS[fx["dataset"]]:
        raise ValueError(f"latency_cap.factor must be {FACTORS[fx['dataset']]} for {fx['dataset']}")


def _view(m) -> dict:
    ref = config_sample(m["v070"]["sampling"], "stock-validation-view")
    view = {**m, "tasks": ref}
    if m.get("pruning"):
        view["pruning"] = {**m["pruning"], "order": ref}
    return view


def validate_manifest_v070(m) -> None:
    validate_v070_manifest(m)
    if set(map(str, m["tasks"])) & set(map(str, (m.get("holdout") or {}).get("tasks", []))):
        raise ValueError("TRAIN pool overlaps the fixed TEST panel")
    _stock_validate(_view(m))


def per_config_aggregate(block, base):
    """driver.aggregate_panel over each configuration's own sample; one shared panel identity."""
    pool_id = digest(sorted(block["strata"]))
    shared = [POLICY, block["seed"], block["size"], digest(block["quota"]), pool_id]

    def aggregate(space, config, rows, *, task_ids, protocol, allow_synthetic=False, allow_partial=False,
                  quality_scale=None):
        own = config_sample(block, space.key(config))
        if not set(own) <= {str(t) for t in task_ids}:
            raise ValueError("Configuration sample outside the caller's TRAIN pool")
        obs = base(space, config, rows, task_ids=own, protocol=protocol, allow_synthetic=allow_synthetic,
                   allow_partial=allow_partial, quality_scale=quality_scale)
        return dataclasses.replace(obs, panel_identity=digest([protocol, *shared, obs.synthetic]))

    aggregate.v070 = True
    return aggregate


def pruning_budget_v070(manifest):
    from trimem.optimizer import staged_pruning as sp

    p = manifest["pruning"]
    return sp.Budget(int(p["budget_episodes"]), int(manifest["v070"]["sampling"]["size"]), int(manifest["budget"]))


def _refuse_worker(*args, **kwargs):
    raise RuntimeError("v0.7.0 runs only search with the latent backend (latent_search_worker_v070)")


@contextlib.contextmanager
def _setattrs(pairs) -> Iterator[None]:
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in pairs]
    try:
        for obj, name, value in pairs:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


@contextlib.contextmanager
def space_patches() -> Iterator[None]:
    from tools import trace_joint_live as live

    from .plan_execution_adapter_v070 import runner_config

    with _setattrs([(driver, "JointSpace", JointSpaceV070), (live, "JointSpace", JointSpaceV070),
                    (live, "runner_config", runner_config)]):
        yield


@contextlib.contextmanager
def v070_patches(manifest) -> Iterator[None]:
    if not is_v070(manifest):
        raise ValueError("v070_patches refuses a non-v0.7.0 manifest")
    validate_v070_manifest(manifest)
    if _ACTIVE["depth"] and _ACTIVE["protocol"] != manifest["protocol"]:
        raise RuntimeError(f"another v0.7.0 run ({_ACTIVE['protocol']}) is active in this process")
    if _ACTIVE["depth"]:
        _ACTIVE["depth"] += 1
        try:
            yield
        finally:
            _ACTIVE["depth"] -= 1
        return
    from tools import trace_joint_protocol as proto
    from tools.trace_v070_holdout import run_pruned_holdout_v070

    from .trace_v070_ledger import SampledLedger
    from .trace_v070_pruning import staged_evaluate_v070

    worker = latent_search_worker_v070 if manifest.get("search_backend") == LATENT_BACKEND else _refuse_worker
    pairs = [
        (driver, "validate_manifest", validate_manifest_v070),
        (driver, "aggregate_panel", per_config_aggregate(manifest["v070"]["sampling"], driver.aggregate_panel)),
        (driver, "SearchLedger", SampledLedger),
        (driver, "pruning_budget", pruning_budget_v070),
        (driver, "staged_evaluate", staged_evaluate_v070),
        (driver, "_search_worker", worker),
        (proto, "run_holdout", run_pruned_holdout_v070),
    ]
    with space_patches(), _setattrs(pairs):
        _ACTIVE.update(protocol=manifest["protocol"], depth=1)
        try:
            yield
        finally:
            _ACTIVE.update(protocol=None, depth=0)


def patched_names() -> dict:
    """Which driver attributes are v0.7.0 now (audit marker of the search worker)."""
    from tools import trace_joint_live as live

    return {"JointSpace": driver.JointSpace is JointSpaceV070, "live.JointSpace": live.JointSpace is JointSpaceV070,
            "aggregate_panel": getattr(driver.aggregate_panel, "v070", False),
            "SearchLedger": driver.SearchLedger.__name__, "staged_evaluate": driver.staged_evaluate.__name__,
            "validate_manifest": driver.validate_manifest.__name__}


def latent_search_worker_v070(directory, manifest, entries, agent, seed, round_file):
    """Spawn target: re-enter the v0.7.0 patches in the child, then the stock latent worker."""
    from trimem.experiments import trace_latent_search as tls

    with v070_patches(manifest):
        driver.atomic_json(Path(directory) / f"{round_file}.v070.json",
                           {"worker": "latent_search_worker_v070", "pid": os.getpid(), "at": time.time(),
                            "patched": patched_names()})
        tls.latent_search_worker(directory, manifest, entries, agent, seed, round_file)
