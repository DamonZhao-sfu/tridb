"""Frozen protocol and auditing primitives for ScienceWorld TRACE ablation.

No experiment is launched on import. Only verified, completed measurements
are admissible; this module never modifies a historical campaign.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import time
import uuid
from collections import Counter
from itertools import pairwise
from pathlib import Path

ARMS = (
    "full",
    "without_llm",
    "without_recombination",
    "without_local",
    "without_pruning",
)
CHECKPOINTS = (20, 40, 60, 80, 100)
FROZEN_ARTIFACTS = frozenset(
    {
        "source",
        "prompt",
        "model",
        "task_pool",
        "candidate_space",
        "memory",
        "environment",
    }
)
RUNTIME_IDENTITY_FIELDS = (
    "actor",
    "model_revision",
    "memory_snapshot",
    "environment_version",
    "serving_profile",
    "episode_budget",
    "sampling",
)


def runtime_identities_match(source, target):
    return all(
        source.get(k) not in (None, "", [], {})
        and target.get(k) not in (None, "", [], {})
        and source[k] == target[k]
        for k in RUNTIME_IDENTITY_FIELDS
    )


def verify_identity_evidence(snapshot, evidence):
    """Check explicit recorded identity evidence; live-service provenance is separate."""
    if set(evidence) != {"source", "target"}:
        raise ValueError("source and target identity evidence are required")
    for side in ("source", "target"):
        entry = evidence[side]
        raw = Path(entry["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
            raise ValueError(f"{side} identity evidence hash mismatch")
        data = json.loads(raw)
        expected = snapshot[f"{side}_identity"]
        if any(
            data.get(k) != expected.get(k) or expected.get(k) in (None, "", [], {})
            for k in RUNTIME_IDENTITY_FIELDS
        ):
            raise ValueError(f"{side} identity evidence fields mismatch")
    return True


def validate_artifact_hashes(hashes):
    if set(hashes) != FROZEN_ARTIFACTS or any(
        not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
        for v in hashes.values()
    ):
        raise ValueError("all required artifact SHA-256 identities must be present")
    return True


def check_global_quota(snapshot=None):
    """Use the 30% guard unless this resumed campaign has explicit opt-in."""
    from tools.autorag_quota_guard import check_snapshot, read_snapshot

    resumed = os.environ.get("TRIMEM_ABLATION_RESUME_AFTER_30", "")
    if resumed not in ("", "1"):
        raise ValueError("invalid ablation quota resume opt-in")
    return check_snapshot(
        read_snapshot() if snapshot is None else snapshot,
        10080,
        100 if resumed == "1" else 30,
    )


def protocol():
    initial, new, panel = 10, 20, 100
    return {
        "dataset": "sciworld",
        "seed": 20261022,
        "initial_configs": initial,
        "new_configs_per_arm": new,
        "max_configs": initial + new,
        "batch_size": 8,
        "full_panel": panel,
        "checkpoints": list(CHECKPOINTS),
        "arms": list(ARMS),
        "one_seed_per_arm": True,
        "validation_tasks": 50,
        "test_tasks": 200,
        "unique_candidate_episode_upper_bound": (initial + len(ARMS) * new) * panel,
        "independent_candidate_episode_upper_bound": len(ARMS)
        * (initial + new)
        * panel,
        "quality": "raw_environment_score",
        "memory_policy": "success+failure-v1",
        "edges": "similar+similar_up",
        "concurrency": 16,
        "new_candidate_reuse": False,
        "global_quota": {"window_minutes": 10080, "pause_at_percent": 30},
    }


def arm_policy(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown ablation arm {arm!r}")
    return {
        "analyzer": arm != "without_llm",
        "recombination": arm != "without_recombination",
        "local_refinement": arm != "without_local",
        "checkpoints": [100] if arm == "without_pruning" else list(CHECKPOINTS),
        "backfill": "global-coverage",
        "max_configs": 30,
        "analyzer_interaction": "diagnostic call retained; its local endpoints are disabled"
        if arm == "without_local"
        else "normal",
    }


def arm_run_tags(prefix):
    if not re.fullmatch(r"[a-z0-9][a-z0-9.]{0,13}", prefix):
        raise ValueError("invalid isolated run-tag prefix")
    return {arm: f"{prefix}.{i}" for i, arm in enumerate(ARMS)}


def effective_slot(slot, policy):
    disabled = {"llm": "analyzer", "xover": "recombination", "dls": "local_refinement"}
    if slot == "llm" and not policy["local_refinement"]:
        return "sobol"
    if slot in disabled and not policy[disabled[slot]]:
        return "sobol"
    return slot


def validate_isolated_memory_database(name, env):
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"trace_ablation_sci_[a-f0-9]{8,32}", name)
        or env.get("TRIMEM_SCIWORLD_MEMORY_DB") != name
    ):
        raise ValueError(
            "isolated ScienceWorld memory database differs from the frozen manifest"
        )
    return True


def verify_isolated_memory_evidence(snapshot, evidence, name):
    """Require the identity file to name the measured isolated memory clone."""
    validate_isolated_memory_database(name, {"TRIMEM_SCIWORLD_MEMORY_DB": name})
    entry = evidence["target"]
    raw = Path(entry["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("isolated memory evidence hash differs")
    data = json.loads(raw)
    proof = data.get("_verification") or {}
    if (
        proof.get("memory_database") != name
        or proof.get("memory_verified") is not True
        or proof.get("memory_fingerprint")
        != snapshot["target_identity"]["memory_snapshot"]
    ):
        raise ValueError("isolated memory evidence does not match the frozen clone")
    return True


def verify_postrun_identity(snapshot, before_path, after_path, memory_database):
    """Require before/after evidence for the same memory, model and live services."""
    validate_isolated_memory_database(
        memory_database, {"TRIMEM_SCIWORLD_MEMORY_DB": memory_database}
    )
    identity = snapshot["target_identity"]
    before, after = (
        json.loads(Path(path).read_text()) for path in (before_path, after_path)
    )
    for data in (before, after):
        if any(
            data.get(key) != identity.get(key) or not identity.get(key)
            for key in RUNTIME_IDENTITY_FIELDS
        ):
            raise ValueError(
                "postrun runtime identity differs from the frozen identity"
            )
        proof = data.get("_verification") or {}
        if (
            proof.get("memory_verified") is not True
            or proof.get("memory_database") != memory_database
            or proof.get("memory_fingerprint") != identity["memory_snapshot"]
        ):
            raise ValueError("postrun memory fingerprint or clone differs")
        if not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("model_fingerprint", ""))):
            raise ValueError("model file fingerprint is missing")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(proof.get("clone_receipt_sha256", ""))
        ):
            raise ValueError("clone receipt fingerprint is missing")
        models, environments, env_proofs = (
            proof.get("models") or [],
            proof.get("environments") or [],
            proof.get("environment_proofs") or [],
        )
        if (
            {x.get("port") for x in models} != {8030, 8031}
            or len(models) != 2
            or {x.get("port") for x in environments} != {36002, 36012}
            or len(environments) != 2
            or len(env_proofs) != 2
            or any(
                x.get("model") != identity["actor"]
                or x.get("revision") != identity["model_revision"]
                for x in models
            )
            or any(
                x.get("environment_fingerprint") != identity["environment_version"]
                for x in env_proofs
            )
        ):
            raise ValueError("model or environment service evidence is incomplete")
    left, right = before["_verification"], after["_verification"]
    for key in (
        "memory_fingerprint",
        "memory_database",
        "clone_receipt_sha256",
        "model_fingerprint",
        "models",
        "environments",
        "environment_proofs",
    ):
        if left.get(key) != right.get(key):
            raise ValueError(f"postrun {key} differs from initial evidence")
    return True


def validate_ablation_environment(env):
    """Reject inherited overrides that would change the frozen protocol."""
    allowed = {
        "POOL_RUN_TAG": None,
        "POOL_RUNTIME_PROFILE": "controlled-c16-v1",
        "POOL_CONCURRENCY_SCIWORLD": "16",
        "POOL_TEST_CONCURRENCY_SCIWORLD": "16",
        "POOL_NO_LIMEQO": "1",
        "POOL_SEMFILTER": "v2",
    }
    for name, value in env.items():
        if (
            name.startswith("POOL_")
            and str(value).strip()
            and (
                name not in allowed
                or (allowed[name] is not None and str(value).strip() != allowed[name])
            )
        ):
            raise ValueError(f"unfrozen ablation environment override: {name}")
    forbidden = (
        "TRIMEM_REUSE_SINCE",
        "TRIMEM_SEMANTIC_ROUTING",
        "TRIMEM_SEMANTIC_BASES",
        "TRIMEM_SEMANTIC_WORKERS",
        "TRIMEM_RETRIEVAL_SLOTS",
        "TRIMEM_SEMFILTER_SLOTS",
    )
    if any(str(env.get(name, "")).strip() for name in forbidden):
        raise ValueError("unfrozen ablation reuse or semantic runtime override")
    if env.get("TRIMEM_MEMORY_POLICY", "success+failure-v1") != "success+failure-v1":
        raise ValueError("ablation memory policy differs from frozen protocol")
    if str(env.get("MAX_ENVS", "32")).strip() != "32":
        raise ValueError("ablation environment server limit differs")
    return True


def verified_semfilter_v2(source, manifest):
    """Bind initialization and search to the exact historical v2 regime."""
    expected = {
        "protocol": "semfilter-v2",
        "from_round": 0,
        "top_n": 32,
        "batch": 8,
        "max_tokens": 96,
        "judge_input": "raw",
        "rubric": "semantic-rubric-v3",
        "guided": True,
    }
    block = source.get("semfilter")
    frozen = manifest.get("semfilter")
    if (
        not isinstance(block, dict)
        or block != frozen
        or any(block.get(k) != v for k, v in expected.items())
    ):
        raise ValueError("SemFilter v2 settings do not match the frozen Exp2 source")
    return dict(block)


def validate_driver_arguments(args):
    policy = arm_policy(args.ablation_arm)
    if (
        args.dataset != "sciworld"
        or not args.run_tag
        or not re.fullmatch(r"[a-z0-9][a-z0-9.]{0,15}", args.run_tag)
    ):
        raise ValueError("ablation requires ScienceWorld and a separate run tag")
    if (
        args.max_configs != 30
        or args.full_panel_size != 100
        or args.initial_budget_episodes != 3000
    ):
        raise ValueError("ablation requires a 30-config, 100-task, 3000-episode budget")
    if args.batch != 8 or not args.task_sample_namespace:
        raise ValueError("ablation requires batch size 8 and a shared sample namespace")
    if args.seed != protocol()["seed"]:
        raise ValueError("ablation requires the frozen search seed")
    if args.reuse:
        raise ValueError(
            "ablation new candidates require --no_reuse for physical accounting"
        )
    expected_staging = "none" if args.ablation_arm == "without_pruning" else "pareto"
    if args.staging != expected_staging:
        raise ValueError(f"ablation staging must be {expected_staging}")
    if args.stage_sizes != ",".join(map(str, CHECKPOINTS)):
        raise ValueError("ablation requires the frozen 20,40,60,80,100 stage schedule")
    if policy["analyzer"] and not args.analyst:
        raise ValueError("only without_llm may disable the analyzer")


def validate_search_design(settings):
    if settings.get("protocol") != "three-channel-v1":
        raise ValueError("ablation requires the three-channel-v1 search design")
    if settings.get("analyzer_workflow") != "diagnose-propose-v2":
        raise ValueError(
            "ablation requires the frozen diagnose-propose-v2 analyzer workflow"
        )


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def freeze_manifest(
    spec,
    source_sha256,
    pool_sha256,
    source_config_keys,
    validation_ids=(),
    *,
    training_pool=None,
):
    if (
        spec != protocol()
        or len(source_config_keys) != 10
        or len(source_config_keys) != len(set(source_config_keys))
    ):
        raise ValueError("invalid protocol or duplicate source configurations")
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("duplicate validation task IDs")
    payload = {
        "version": 1,
        "protocol": spec,
        "source_sha256": source_sha256,
        "pool_sha256": pool_sha256,
        "source_config_keys": list(source_config_keys),
        "validation_ids": list(validation_ids),
        "training_pool_sha256": _digest(training_pool)
        if training_pool is not None
        else None,
        "runtime_identity_attested": False,
        "stage": "code_only",
    }
    return {**payload, "manifest_sha256": _digest(payload)}


def verify_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _digest(payload):
        raise ValueError("manifest hash mismatch")
    keys = manifest.get("source_config_keys", [])
    validation = manifest.get("validation_ids", [])
    if (
        manifest.get("protocol") != protocol()
        or len(keys) != 10
        or len(set(keys)) != 10
        or len(set(validation)) != len(validation)
    ):
        raise ValueError("manifest protocol or identities invalid")
    if manifest.get("stage") == "seed_prepared":
        final_only = {
            "manifest_sha256",
            "recorded_identity_evidence_verified",
            "live_runtime_identity_verified",
            "shared_initialization_sha256",
            "resource_ledger_binding",
            "test_ids_sha256",
            "artifact_sha256",
            "resource_limits",
            "system_limits",
            "identity_evidence",
            "memory_database",
        }
        draft = {key: value for key, value in manifest.items() if key not in final_only}
        draft.update({"runtime_identity_attested": False, "stage": "code_only"})
        if manifest.get("resource_ledger_binding") != _digest(draft):
            raise ValueError("resource ledger binding differs from the draft manifest")
    return True


def select_validation(items, initial_ids, test_ids, *, size=50, seed=20261022):
    from trimem.experiments import build_agentenv_pool as bp

    candidates = [
        r
        for r in items
        if r["item_id"] not in initial_ids and r["item_id"] not in test_ids
    ]
    if len({r["item_id"] for r in items}) != len(items) or not 0 < size <= len(
        candidates
    ):
        raise ValueError("invalid validation population or size")
    selected = bp.draw_sample(candidates, size, seed)
    return [r["item_id"] for r in selected]


def training_pool_without_validation(pool, validation_ids, initial_ids):
    validation_ids, initial_ids = set(validation_ids), set(initial_ids)
    if validation_ids & initial_ids:
        raise ValueError("validation overlaps initial configuration samples")
    original = pool["items"]
    ids = {r["item_id"] for r in original}
    if len(ids) != len(original) or not validation_ids <= ids or not initial_ids <= ids:
        raise ValueError("pool identities are missing or duplicated")
    items = [r for r in original if r["item_id"] not in validation_ids]
    new_hash = hashlib.sha256(
        json.dumps([r["item_id"] for r in items]).encode()
    ).hexdigest()[:16]
    return {
        **pool,
        "source_pool_hash": pool["pool_hash"],
        "pool_hash": new_hash,
        "size": len(items),
        "strata": dict(Counter(r["stratum"] for r in items)),
        "items": items,
        "validation_ids": sorted(validation_ids),
    }


def validate_isolated_pool(pool, manifest):
    verify_manifest(manifest)
    ids = [r["item_id"] for r in pool["items"]]
    validation = set(manifest["validation_ids"])
    if validation & set(ids) or validation != set(pool["validation_ids"]):
        raise ValueError("validation IDs not isolated from training pool")
    if (
        pool["dataset"] != "sciworld"
        or len(ids) != len(set(ids))
        or pool["size"] != len(ids)
    ):
        raise ValueError("invalid training pool identities")
    if pool["strata"] != dict(Counter(r["stratum"] for r in pool["items"])):
        raise ValueError("training pool strata mismatch")
    if pool["pool_hash"] != hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]:
        raise ValueError("training pool hash mismatch")
    if (
        not manifest.get("training_pool_sha256")
        or _digest(pool) != manifest["training_pool_sha256"]
    ):
        raise ValueError("training pool identity does not match manifest")
    return True


def panel_task_weights(sample_ids, pool):
    """Match unpaired_stats.stratified's population weighting on sampled strata."""
    items = pool["items"]
    strata = {r["item_id"]: r["stratum"] for r in items}
    if (
        len(strata) != len(items)
        or len(sample_ids) != len(set(sample_ids))
        or not set(sample_ids) <= set(strata)
        or not sample_ids
    ):
        raise ValueError("invalid sample or pool identities")
    population = Counter(strata.values())
    sampled = Counter(strata[q] for q in sample_ids)
    included = sum(population[h] for h in sampled)
    return {
        q: population[strata[q]] / included / sampled[strata[q]] for q in sample_ids
    }


def aggregate_hv_panel(config_key, rows, *, sample_ids, completion_s, task_weights):
    """Build one full TRAIN panel from observed rows and frozen estimator weights."""
    ids = [r["query_id"] for r in rows]
    if (
        len(sample_ids) != 100
        or len(set(sample_ids)) != 100
        or len(rows) != 100
        or len(set(ids)) != 100
        or set(ids) != set(sample_ids)
    ):
        raise ValueError("full panel task identities mismatch")
    if any(
        r.get("config_key") != config_key
        or r.get("split") != "train"
        or r.get("status") != "evaluated"
        for r in rows
    ):
        raise ValueError(
            "full panel requires evaluated TRAIN measurements for one config"
        )
    if (
        set(task_weights) != set(sample_ids)
        or any(not math.isfinite(float(w)) or w < 0 for w in task_weights.values())
        or not math.isclose(math.fsum(task_weights.values()), 1, abs_tol=1e-9)
    ):
        raise ValueError("invalid frozen task weights")
    sums = {}
    for metric in ("quality", "tokens", "latency_ms"):
        vals = [float(r[metric]) for r in rows]
        if any(not math.isfinite(v) or (metric != "quality" and v <= 0) for v in vals):
            raise ValueError("invalid panel measurement")
        sums[metric] = math.fsum(
            float(r[metric]) * task_weights[r["query_id"]] for r in rows
        )
    if not math.isfinite(float(completion_s)) or completion_s < 0:
        raise ValueError("invalid completion time")
    return {
        "config_key": config_key,
        "split": "train",
        "complete": True,
        "tasks": 100,
        "completion_s": float(completion_s),
        **sums,
    }


def full_training_panels(state, rows, pool, ledger, segments, *, key_of):
    """Join completed panels to their actual DB and ledger receipts for HV."""
    elapsed_segments(segments)
    if any(
        r.get("split") != "train" or r.get("quality_domain") != "raw_environment_score"
        for r in rows
    ):
        raise ValueError("HV input must use raw-score TRAIN receipts")
    indexed = {}
    for r in rows:
        if (
            r.get("experiment_id") != state["experiment"]
            or r.get("status") != "evaluated"
        ):
            raise ValueError("HV input has foreign or incomplete receipts")
        pair = (r["config_key"], r["query_id"])
        if pair in indexed:
            raise ValueError("duplicate HV receipt")
        indexed[pair] = r
    recorded = ledger._read()["pairs"]
    result, seen = [], set()
    for rnd in state["rounds"]:
        for c in rnd["configs"]:
            key, staging = key_of(c["config"]), c.get("staging") or {}
            if key in seen:
                raise ValueError("duplicate configuration in search state")
            seen.add(key)
            if (
                staging.get("stage_reached") != 100
                or staging.get("pruned_at") is not None
            ):
                continue
            ids = c["sample"]["ids"]
            selected = []
            for query in ids:
                row = indexed.get((key, query))
                logical = recorded.get(
                    json.dumps([key, query], separators=(",", ":")), {}
                )
                if (
                    not row
                    or logical.get("status") != "complete"
                    or logical.get("retrieval_id") != str(row.get("retrieval_id"))
                    or any(
                        not math.isclose(float(logical[a]), float(row[b]), abs_tol=1e-8)
                        for a, b in (
                            ("reward", "quality"),
                            ("tokens", "total_tokens"),
                            ("latency_ms", "e2e_ms"),
                        )
                    )
                ):
                    raise ValueError("HV DB and ledger receipt mismatch")
                selected.append(
                    {**row, "tokens": row["total_tokens"], "latency_ms": row["e2e_ms"]}
                )
            if rnd["round"] == 0:
                completion = 0.0  # shared initialization is present when search begins
            else:
                events = [
                    e
                    for e in staging.get("events", [])
                    if e.get("stage") == 100 and e.get("status") == "complete"
                ]
                if len(events) != 1:
                    raise ValueError("full candidate lacks a measured completion event")
                completion = event_elapsed(events[0], segments)
            result.append(
                aggregate_hv_panel(
                    key,
                    selected,
                    sample_ids=ids,
                    completion_s=completion,
                    task_weights=panel_task_weights(ids, pool),
                )
            )
    return result


def candidate_work_curve(curve, state, rows, ledger, segments, *, key_of):
    """Count committed candidate episodes and summed latency at each HV event."""
    elapsed_segments(segments)
    initial = {key_of(c["config"]) for c in state["rounds"][0]["configs"]}
    recorded = ledger._read()["pairs"]
    start, end = float(segments[0]["start_wall_s"]), float(segments[-1]["end_wall_s"])
    work, seen = [], set()
    for r in rows:
        key, query = r["config_key"], r["query_id"]
        pair = (key, query)
        if (
            pair in seen
            or r.get("experiment_id") != state["experiment"]
            or r.get("split") != "train"
            or r.get("status") != "evaluated"
            or r.get("quality_domain") != "raw_environment_score"
        ):
            raise ValueError("invalid or duplicate candidate work receipt")
        seen.add(pair)
        logical = recorded.get(json.dumps([key, query], separators=(",", ":")), {})
        if (
            logical.get("status") != "complete"
            or logical.get("retrieval_id") != str(r.get("retrieval_id"))
            or any(
                not math.isclose(float(logical[a]), float(r[b]), abs_tol=1e-8)
                for a, b in (
                    ("reward", "quality"),
                    ("tokens", "total_tokens"),
                    ("latency_ms", "e2e_ms"),
                )
            )
        ):
            raise ValueError("candidate work receipt does not match ledger")
        timestamp = r.get("updated_at")
        if hasattr(timestamp, "timestamp"):
            if timestamp.tzinfo is None:
                raise ValueError("candidate timestamp must include timezone")
            timestamp = timestamp.timestamp()
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (
            key not in initial and not start <= timestamp <= end
        ):
            raise ValueError("candidate completion timestamp outside measured search")
        work.append((key in initial, timestamp, float(r["e2e_ms"]) / 1000))
    result = []
    for point in curve:
        t = float(point["completion_s"])
        if not 0 <= t <= end - start:
            raise ValueError("HV event outside measured search")
        included = [x for x in work if x[0] or x[1] <= start + t]
        result.append(
            {
                **point,
                "episodes": len(included),
                "summed_episode_latency_s": math.fsum(x[2] for x in included),
            }
        )
    return result


def freeze_hv_reference(initial_rows):
    """Freeze clipped normalization from complete TRAIN initialization only."""
    if len(initial_rows) != 10 or len({r["config_key"] for r in initial_rows}) != 10:
        raise ValueError(
            "HV normalization needs ten unique initialization configurations"
        )
    vectors = []
    for row in initial_rows:
        if (
            row.get("split") != "train"
            or row.get("complete") is not True
            or row.get("tasks") != 100
        ):
            raise ValueError("HV reference requires complete TRAIN panels")
        q, t, lat = (float(row[k]) for k in ("quality", "tokens", "latency_ms"))
        if not all(math.isfinite(v) for v in (q, t, lat)) or min(t, lat) <= 0:
            raise ValueError("invalid initialization objectives")
        vectors.append((q, -math.log(t), -math.log(lat)))
    reference, scale = [], []
    for i in range(3):
        low, high = min(v[i] for v in vectors), max(v[i] for v in vectors)
        width = max(high - low, 1e-9)
        reference.append(low - 0.1 * width)
        scale.append(1.1 * width)
    payload = {
        "reference": reference,
        "scale": scale,
        "clip": [0, 1],
        "source": "shared_initialization_train",
        "margin_fraction": 0.1,
        "config_keys": sorted(r["config_key"] for r in initial_rows),
    }
    return {**payload, "sha256": _digest(payload)}


def hv_curve(records, *, reference, scale):
    """TRAIN-only completion steps, with one frozen reference and clip to [0,1]."""
    import numpy as np

    from trimem.optimizer.task_complementary import hv3d

    if any(r.get("split") != "train" for r in records):
        raise ValueError("HV curve may only use TRAIN observations")
    ref, bounds = np.asarray(reference, dtype=float), np.asarray(scale, dtype=float)
    if (
        ref.shape != (3,)
        or bounds.shape != (3,)
        or not np.isfinite(ref).all()
        or not np.isfinite(bounds).all()
        or np.any(bounds <= 0)
    ):
        raise ValueError(
            "HV reference and scale must be finite 3-vectors with positive scales"
        )
    eligible = [
        r for r in records if r.get("complete") is True and r.get("tasks") == 100
    ]
    eligible.sort(key=lambda r: (float(r["completion_s"]), r["config_key"]))
    frontier, result, seen = [], [], set()
    for r in eligible:
        if r["config_key"] in seen:
            raise ValueError("duplicate completed configuration")
        seen.add(r["config_key"])
        t, latency = float(r["tokens"]), float(r["latency_ms"])
        vals = np.asarray([float(r["quality"]), -math.log(t), -math.log(latency)])
        if (
            not np.isfinite(vals).all()
            or not math.isfinite(float(r["completion_s"]))
            or float(r["completion_s"]) < 0
        ):
            raise ValueError("invalid completion or objective")
        frontier.append(np.clip((vals - ref) / bounds, 0, 1))
        result.append(
            {
                "config_key": r["config_key"],
                "completion_s": float(r["completion_s"]),
                "nhv": hv3d(np.asarray(frontier)),
            }
        )
    return result


def elapsed_segments(segments):
    """Report actual wall span and sum of monotonic active segments; fail on gaps in evidence."""
    if not segments:
        raise ValueError("incomplete timing: no segments")
    active, previous = 0.0, None
    for segment in segments:
        try:
            start, end = float(segment["start_wall_s"]), float(segment["end_wall_s"])
            m0, m1 = (
                int(segment["start_monotonic_ns"]),
                int(segment["end_monotonic_ns"]),
            )
            if (
                not segment["boot_id"]
                or not all(math.isfinite(v) for v in (start, end))
                or end < start
                or m1 < m0
                or (previous is not None and start < previous)
            ):
                raise ValueError("invalid or overlapping timing segment")
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete timing segment") from exc
        active += (m1 - m0) / 1e9
        previous = end
    wall = float(segments[-1]["end_wall_s"]) - float(segments[0]["start_wall_s"])
    gap = wall - active
    if gap < -1:
        raise ValueError("wall and monotonic timing disagree")
    return {"wall_elapsed_s": wall, "active_elapsed_s": active, "gap_s": max(0.0, gap)}


def event_elapsed(event, segments):
    """Time since post-initialization search start, including observed pauses."""
    elapsed_segments(segments)
    wall, monotonic = (
        float(event["observed_wall_s"]),
        int(event["observed_monotonic_ns"]),
    )
    matches = [
        s
        for s in segments
        if s["boot_id"] == event["boot_id"]
        and s["start_wall_s"] <= wall <= s["end_wall_s"]
        and s["start_monotonic_ns"] <= monotonic <= s["end_monotonic_ns"]
    ]
    if len(matches) != 1:
        raise ValueError("event is not within exactly one measured search segment")
    segment = matches[0]
    if (
        abs(
            (wall - segment["start_wall_s"])
            - (monotonic - segment["start_monotonic_ns"]) / 1e9
        )
        > 1
    ):
        raise ValueError("event clock disagrees with its segment")
    return wall - segments[0]["start_wall_s"]


def common_time_hv(curves):
    """Compare step functions only through each arm's measured search end."""
    if not curves:
        raise ValueError("missing curves")
    for curve in curves.values():
        points, end = curve["points"], float(curve["end_s"])
        times = [float(p["completion_s"]) for p in points]
        values = [float(p["nhv"]) for p in points]
        if (
            not points
            or not math.isfinite(end)
            or any(not math.isfinite(t) or t < 0 for t in times)
            or times != sorted(times)
            or times[-1] > end
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or any(b < a - 1e-12 for a, b in pairwise(values))
        ):
            raise ValueError("invalid observed HV curve")
    start = max(float(c["points"][0]["completion_s"]) for c in curves.values())
    end = min(float(c["end_s"]) for c in curves.values())
    if start > end:
        raise ValueError("curves have no common observed interval")
    values = {
        arm: [p for p in curve["points"] if float(p["completion_s"]) <= end][-1]["nhv"]
        for arm, curve in curves.items()
    }
    return {"common_start_s": start, "common_end_s": end, "values": values}


class SearchClock:
    """Persistent wall/monotonic segments. An unclosed crash is never estimated."""

    def __init__(self, path, manifest_hash):
        self.path = Path(path)
        self.manifest_hash = manifest_hash

    def _read(self):
        if not self.path.exists():
            return {"manifest_hash": self.manifest_hash, "segments": []}
        data = json.loads(self.path.read_text())
        if data.get("manifest_hash") != self.manifest_hash:
            raise ValueError("clock identity mismatch")
        return data

    def _update(self, fn):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self._read()
            fn(data)
            fd, name = tempfile.mkstemp(
                prefix=self.path.name + ".", dir=self.path.parent
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(name, self.path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)

    def start(self, *, wall_s, monotonic_ns, boot_id):
        def update(data):
            segments = data["segments"]
            if segments and "end_wall_s" not in segments[-1]:
                raise ValueError("open segment requires crash reconciliation")
            if not math.isfinite(float(wall_s)) or monotonic_ns < 0 or not boot_id:
                raise ValueError("invalid clock start")
            if segments and wall_s < segments[-1]["end_wall_s"]:
                raise ValueError("clock start precedes previous segment")
            segments.append(
                {
                    "start_wall_s": float(wall_s),
                    "start_monotonic_ns": int(monotonic_ns),
                    "boot_id": boot_id,
                }
            )

        self._update(update)

    def end(self, *, wall_s, monotonic_ns, boot_id):
        def update(data):
            if not data["segments"] or "end_wall_s" in data["segments"][-1]:
                raise ValueError("no open segment")
            segment = data["segments"][-1]
            if segment["boot_id"] != boot_id:
                raise ValueError("clock boot changed within segment")
            segment.update(end_wall_s=float(wall_s), end_monotonic_ns=int(monotonic_ns))
            elapsed_segments(data["segments"])

        self._update(update)

    def summary(self):
        """Return the measured wall span and active segments."""
        return elapsed_segments(self._read()["segments"])


def record_stage_event(staging, *, stage, status, wall_s, monotonic_ns, boot_id):
    if stage not in CHECKPOINTS or status not in ("complete", "pruned"):
        raise ValueError("invalid stage or status")
    if not math.isfinite(float(wall_s)) or monotonic_ns < 0 or not boot_id:
        raise ValueError("invalid stage event clock")
    event = {
        "stage": stage,
        "status": status,
        "observed_wall_s": float(wall_s),
        "observed_monotonic_ns": int(monotonic_ns),
        "boot_id": boot_id,
    }
    events = staging.setdefault("events", [])
    old = [x for x in events if x["stage"] == stage]
    if old:
        # Resume observes the already-completed stage at a later time. Preserve
        # the original timestamp; only a changed decision is a conflict.
        if len(old) != 1 or old[0]["status"] != status:
            raise ValueError("conflicting stage event")
        return False
    if events and stage <= events[-1]["stage"]:
        raise ValueError("stage events out of order")
    events.append(event)
    return True


def measured_router_overhead(raw, prediction_ms):
    overhead = raw.get("overhead") or {}
    keys = ("embedding_ms", "probe_ms", "llm_ms", "llm_tokens")
    if any(overhead.get(k) is None for k in keys):
        raise ValueError("router input-generation overhead is missing")
    values = [float(overhead[k]) for k in keys]
    if any(not math.isfinite(v) or v < 0 for v in (*values, prediction_ms)):
        raise ValueError("invalid router input-generation overhead")
    return {
        "router_overhead_ms": sum(values[:3]) + prediction_ms,
        "router_overhead_tokens": values[3],
        "prediction_ms": prediction_ms,
        "input_overhead": dict(overhead),
    }


def build_validation_routes(
    model,
    feature_spec,
    routing,
    raw_queries,
    validation_ids,
    catalog,
    train_ids,
    *,
    key_of,
    record_overhead=False,
):
    """Route frozen validation tasks with the already fitted joint-selected model."""
    from trimem.optimizer.query_router import runtime

    if (
        model.get("kind") != "joint-selected"
        or model.get("quality_kind") != "raw_score"
    ):
        raise ValueError("ScienceWorld requires the raw-score joint-selected router")
    if (
        len(validation_ids) != 50
        or len(set(validation_ids)) != 50
        or set(validation_ids) & set(train_ids)
    ):
        raise ValueError("validation identities invalid or overlap TRAIN")
    query_ids = [r["query_id"] for r in raw_queries]
    if (
        len(query_ids) != 50
        or len(set(query_ids)) != 50
        or set(query_ids) != set(validation_ids)
    ):
        raise ValueError("validation query identities mismatch")
    model_keys = [key_of(c) for c in model["configs"]]
    if len(model_keys) != len(set(model_keys)) or set(model_keys) != set(catalog):
        raise ValueError("router and OneConfig must share the frozen catalog")
    routes = []
    for raw in raw_queries:
        started = time.perf_counter_ns()
        decision = runtime.choose_routing(
            model, runtime.transform(feature_spec, raw), routing
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        key = key_of(decision["config"])
        pred = decision["pred"]
        if (
            key not in catalog
            or not all(
                math.isfinite(float(pred[k]))
                for k in ("quality", "tokens", "latency_ms")
            )
            or min(pred["tokens"], pred["latency_ms"]) <= 0
        ):
            raise ValueError("invalid router decision")
        row = {"query_id": raw["query_id"], "config_key": key, "prediction": pred}
        if record_overhead:
            row.update(measured_router_overhead(raw, elapsed_ms))
        routes.append(row)
    return routes


def aggregate_validation_episodes(policy_id, kind, records, validation_ids, catalog):
    """Aggregate observed validation episodes for a policy on a frozen catalog."""
    if kind not in ("oneconfig", "router") or not policy_id:
        raise ValueError("invalid policy")
    if (
        len(validation_ids) != 50
        or len(set(validation_ids)) != 50
        or len(records) != 50
    ):
        raise ValueError("validation identities must contain 50 unique tasks")
    ids = [r["query_id"] for r in records]
    if len(set(ids)) != 50 or set(ids) != set(validation_ids):
        raise ValueError("validation episode identities do not match")
    if any(
        r.get("split") != "validation" or r.get("status") != "evaluated"
        for r in records
    ):
        raise ValueError("validation requires completed validation measurements")
    keys = {r["config_key"] for r in records}
    if not keys <= set(catalog) or (kind == "oneconfig" and len(keys) != 1):
        raise ValueError("policy must use the frozen catalog")
    totals = {k: [] for k in ("quality", "tokens", "latency_ms")}
    for r in records:
        for k, values in totals.items():
            value = float(r[k])
            if not math.isfinite(value) or (k != "quality" and value <= 0):
                raise ValueError("invalid validation measurement")
            values.append(value)
    return {
        "policy_id": policy_id,
        "kind": kind,
        "split": "validation",
        "covered": 50,
        "task_ids": sorted(ids),
        "config_keys": sorted(keys),
        **{k: sum(values) / 50 for k, values in totals.items()},
    }


def select_validation_policies(rows, *, validation_ids):
    """Choose fixed and router policies from validation values, never TEST."""
    if not rows or any(r["kind"] not in ("oneconfig", "router") for r in rows):
        raise ValueError("invalid validation policies")
    if len(validation_ids) != 50 or len(set(validation_ids)) != 50:
        raise ValueError("validation task identities must contain 50 unique tasks")
    ids = [r["policy_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate policy identity")
    for r in rows:
        if r.get("split") != "validation" or r.get("covered") != 50:
            raise ValueError("validation policy requires all 50 validation tasks")
        if len(r.get("task_ids", ())) != 50 or set(r["task_ids"]) != set(
            validation_ids
        ):
            raise ValueError("validation task identities differ between policies")
        vals = [float(r[k]) for k in ("quality", "tokens", "latency_ms")]
        if not all(math.isfinite(v) for v in vals) or min(vals[1:]) <= 0:
            raise ValueError("invalid validation measurement")
    fixed = [r for r in rows if r["kind"] == "oneconfig"]
    if not fixed:
        raise ValueError("missing OneConfig reference")
    reference = max(float(r["quality"]) for r in fixed)
    if reference <= 0:
        raise ValueError("reference quality must be positive")
    targets = {}
    for pct in (100, 95, 90):
        floor = reference * pct / 100
        group = {"quality_target": floor}
        for kind in ("oneconfig", "router"):
            eligible = [
                r for r in rows if r["kind"] == kind and float(r["quality"]) >= floor
            ]
            group[kind] = (
                min(
                    eligible,
                    key=lambda r: (
                        float(r["tokens"]),
                        float(r["latency_ms"]),
                        r["policy_id"],
                    ),
                )
                if eligible
                else None
            )
        targets[str(pct)] = group
    payload = {
        "reference_quality": reference,
        "targets": targets,
        "selection_source": "validation",
        "validation_ids_sha256": _digest(sorted(validation_ids)),
    }
    return {**payload, "sha256": _digest(payload)}


def freeze_router_validation(
    policies,
    model,
    feature_spec,
    raw_queries,
    validation_ids,
    *,
    train_ids,
    test_ids,
    catalog,
    training_snapshot,
    full_experiment,
    key_of,
):
    """Freeze policy targets from measured validation, with TRAIN-only model inputs."""
    if set(validation_ids) & (set(train_ids) | set(test_ids)) or set(train_ids) & set(
        test_ids
    ):
        raise ValueError("TRAIN, validation and TEST identities overlap")
    recorded_train = [r["query_id"] for r in training_snapshot.get("records", [])]
    if (
        training_snapshot.get("split") != "train"
        or training_snapshot.get("dataset") != "sciworld"
        or training_snapshot.get("source", {}).get("experiment") != full_experiment
        or len(recorded_train) != len(set(recorded_train))
        or set(recorded_train) != set(train_ids)
    ):
        raise ValueError(
            "router training snapshot does not match Full TRACE TRAIN identities"
        )
    if not policies or not catalog:
        raise ValueError("missing validation policies or frozen catalog")
    identifiers = [p["policy_id"] for p in policies]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate validation policy")
    results, definitions, fixed_configs = [], {}, set()
    for policy in policies:
        records = policy["records"]
        kind = policy["kind"]
        if kind == "router":
            routes = build_validation_routes(
                model,
                feature_spec,
                policy["routing"],
                raw_queries,
                validation_ids,
                catalog,
                train_ids,
                key_of=key_of,
            )
            expected = {r["query_id"]: r["config_key"] for r in routes}
            if any(r["config_key"] != expected.get(r["query_id"]) for r in records):
                raise ValueError("measured validation routes differ from frozen router")
            receipts = policy.get("overhead_receipts") or []
            receipt_by_id = {r["query_id"]: r for r in receipts}
            query_by_id = {r["query_id"]: r for r in raw_queries}
            if len(receipts) != len(validation_ids) or set(receipt_by_id) != set(
                validation_ids
            ):
                raise ValueError("validation router overhead receipts are incomplete")
            for record in records:
                receipt = receipt_by_id[record["query_id"]]
                if receipt.get("config_key") != expected[record["query_id"]]:
                    raise ValueError("validation overhead route differs")
                calculated = measured_router_overhead(
                    query_by_id[record["query_id"]], float(receipt["prediction_ms"])
                )
                if any(
                    float(record[k]) != float(receipt[k])
                    or float(receipt[k]) != calculated[k]
                    for k in ("router_overhead_ms", "router_overhead_tokens")
                ):
                    raise ValueError(
                        "validation router overhead differs from measured receipts"
                    )
            definitions[policy["policy_id"]] = {
                "kind": kind,
                "routing": policy["routing"],
                "overhead_receipts_sha256": _digest(
                    sorted(receipts, key=lambda r: r["query_id"])
                ),
            }
        elif kind == "oneconfig":
            keys = {r["config_key"] for r in records}
            if len(keys) != 1:
                raise ValueError("OneConfig policy must use one configuration")
            key = next(iter(keys))
            if key in fixed_configs:
                raise ValueError("duplicate OneConfig policy")
            fixed_configs.add(key)
            definitions[policy["policy_id"]] = {"kind": kind, "config_key": key}
        else:
            raise ValueError("invalid validation policy kind")
        measured = []
        for r in records:
            overhead_ms, overhead_tokens = (
                float(r["router_overhead_ms"]),
                float(r["router_overhead_tokens"]),
            )
            if (
                not math.isfinite(overhead_ms)
                or not math.isfinite(overhead_tokens)
                or min(overhead_ms, overhead_tokens) < 0
            ):
                raise ValueError("invalid measured router overhead")
            if kind == "oneconfig" and (overhead_ms != 0 or overhead_tokens != 0):
                raise ValueError("OneConfig has no router overhead")
            measured.append(
                {
                    **r,
                    "tokens": float(r["tokens"]) + overhead_tokens,
                    "latency_ms": float(r["latency_ms"]) + overhead_ms,
                }
            )
        results.append(
            aggregate_validation_episodes(
                policy["policy_id"], kind, measured, validation_ids, catalog
            )
        )
        definitions[policy["policy_id"]]["validation_receipts_sha256"] = _digest(
            sorted(records, key=lambda r: r["query_id"])
        )
    if fixed_configs != set(catalog):
        raise ValueError("OneConfig validation must cover the entire frozen catalog")
    selection = select_validation_policies(results, validation_ids=validation_ids)
    payload = {
        "model_sha256": _digest(model),
        "feature_spec_sha256": _digest(feature_spec),
        "training_snapshot_sha256": _digest(training_snapshot),
        "full_experiment": full_experiment,
        "catalog_sha256": _digest(sorted(catalog)),
        "definitions": definitions,
        "selection": selection,
        "selection_split": "validation",
    }
    return {**payload, "sha256": _digest(payload)}


def verify_frozen_router_policy(frozen):
    payload = {k: v for k, v in frozen.items() if k != "sha256"}
    if frozen.get("sha256") != _digest(payload):
        raise ValueError("frozen router policy hash mismatch")
    if frozen.get("selection_split") != "validation" or set(
        frozen.get("selection", {}).get("targets", {})
    ) != {"100", "95", "90"}:
        raise ValueError("invalid frozen validation targets")
    return True


def build_frozen_test_routes(
    frozen,
    model,
    feature_spec,
    raw_queries,
    test_ids,
    *,
    train_ids,
    validation_ids,
    catalog,
    target,
    key_of,
    record_overhead=False,
):
    """Route TEST only with policies selected and frozen on validation."""
    from trimem.optimizer.query_router import runtime

    verify_frozen_router_policy(frozen)
    if (
        frozen["model_sha256"] != _digest(model)
        or frozen["feature_spec_sha256"] != _digest(feature_spec)
        or frozen["catalog_sha256"] != _digest(sorted(catalog))
    ):
        raise ValueError(
            "TEST router model, features or catalog differ from frozen policy"
        )
    ids = [q["query_id"] for q in raw_queries]
    if (
        len(test_ids) != 200
        or len(set(test_ids)) != 200
        or len(ids) != 200
        or len(set(ids)) != 200
        or set(ids) != set(test_ids)
        or set(ids) & (set(train_ids) | set(validation_ids))
    ):
        raise ValueError("TEST task identities invalid or overlap TRAIN/validation")
    model_keys = [key_of(c) for c in model["configs"]]
    if (
        model.get("kind") != "joint-selected"
        or model.get("quality_kind") != "raw_score"
        or len(model_keys) != len(set(model_keys))
        or set(model_keys) != set(catalog)
    ):
        raise ValueError("TEST router differs from frozen raw-score catalog")
    if target not in ("100", "95", "90"):
        raise ValueError("invalid frozen quality target")
    selected = frozen["selection"]["targets"][target]
    fixed = selected["oneconfig"]
    router = selected["router"]
    if fixed is None:
        return {
            "quality_target": selected["quality_target"],
            "oneconfig": None,
            "router": None,
            "frozen_sha256": frozen["sha256"],
        }
    fixed_def = frozen["definitions"][fixed["policy_id"]]
    if (
        fixed_def.get("kind") != "oneconfig"
        or fixed_def.get("config_key") not in catalog
    ):
        raise ValueError("invalid frozen OneConfig definition")
    fixed_routes = {q: fixed_def["config_key"] for q in test_ids}
    router_routes = None
    overheads = None
    if router is not None:
        definition = frozen["definitions"][router["policy_id"]]
        if definition.get("kind") != "router":
            raise ValueError("invalid frozen router definition")
        router_routes = {}
        overheads = {} if record_overhead else None
        for raw in raw_queries:
            started = time.perf_counter_ns()
            decision = runtime.choose_routing(
                model, runtime.transform(feature_spec, raw), definition["routing"]
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            key = key_of(decision["config"])
            if key not in catalog:
                raise ValueError("TEST router selected outside frozen catalog")
            router_routes[raw["query_id"]] = key
            if record_overhead:
                overheads[raw["query_id"]] = measured_router_overhead(raw, elapsed_ms)
    return {
        "quality_target": selected["quality_target"],
        "oneconfig": fixed_routes,
        "router": router_routes,
        "router_overhead": overheads,
        "frozen_sha256": frozen["sha256"],
    }


def make_routing_sample(phase, records, task_ids, *, train_ids, test_ids):
    size = {"validation": 50, "test": 200}.get(phase)
    ids = [r["item_id"] for r in records]
    if (
        size is None
        or len(task_ids) != size
        or len(set(task_ids)) != size
        or len(ids) != size
        or len(set(ids)) != size
        or set(ids) != set(task_ids)
    ):
        raise ValueError("routing sample identities mismatch")
    if set(ids) & set(train_ids) or (
        phase == "validation" and set(ids) & set(test_ids)
    ):
        raise ValueError("routing sample overlaps another split")
    if phase == "test" and set(ids) != set(test_ids):
        raise ValueError("TEST sample identities mismatch")
    by_id = {r["item_id"]: r for r in records}
    if any(not by_id[q].get("task_key") for q in task_ids):
        raise ValueError("routing sample lacks task metadata")
    sample_hash = hashlib.sha256(json.dumps(task_ids).encode()).hexdigest()[:16]
    panel_hash = _digest([phase, task_ids])
    return {
        "dataset": "sciworld",
        "kind": "pool-sample",
        "label": f"ablation-{phase}",
        "seed": protocol()["seed"],
        "size": size,
        "pool_hash": panel_hash,
        "panel_hash": panel_hash,
        "sampling": "frozen",
        "sample_hash": sample_hash,
        "items": [copy.deepcopy(by_id[q]) for q in task_ids],
    }


def routing_evaluation_plan(
    phase,
    catalog,
    task_ids,
    *,
    max_episodes,
    frozen_hash=None,
    routes_by_target=None,
    sample_sha256=None,
):
    """Freeze and deduplicate physical (configuration, task) evaluation pairs."""
    size = {"validation": 50, "test": 200}.get(phase)
    if (
        size is None
        or len(task_ids) != size
        or len(set(task_ids)) != size
        or not catalog
    ):
        raise ValueError("invalid routing phase, task identities or catalog")
    if type(max_episodes) is not int or max_episodes < 0:
        raise ValueError("invalid routing episode budget")
    if sample_sha256 is not None and (
        not isinstance(sample_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sample_sha256)
    ):
        raise ValueError("invalid phase sample file hash")
    pairs = set()
    if phase == "validation":
        pairs = {(key, q) for key in catalog for q in task_ids}
        if frozen_hash is not None or routes_by_target is not None:
            raise ValueError("validation plan cannot depend on TEST routes")
    else:
        if (
            not frozen_hash
            or not routes_by_target
            or set(routes_by_target) != {"100", "95", "90"}
        ):
            raise ValueError("TEST plan requires all frozen validation targets")
        for route in routes_by_target.values():
            if route.get("frozen_sha256") != frozen_hash:
                raise ValueError("TEST route policy identity mismatch")
            for mode in ("oneconfig", "router"):
                mapping = route.get(mode)
                if mapping is None:
                    continue
                if set(mapping) != set(task_ids) or not set(mapping.values()) <= set(
                    catalog
                ):
                    raise ValueError(
                        "TEST route task or configuration identities mismatch"
                    )
                if mode == "oneconfig" and len(set(mapping.values())) != 1:
                    raise ValueError("OneConfig route uses multiple configurations")
                pairs.update((key, q) for q, key in mapping.items())
    if len(pairs) > max_episodes:
        raise ValueError("routing episode budget exhausted")
    payload = {
        "phase": phase,
        "catalog_sha256": _digest(sorted(catalog)),
        "task_ids_sha256": _digest(sorted(task_ids)),
        "frozen_policy_sha256": frozen_hash,
        "sample_sha256": sample_sha256,
        "max_episodes": max_episodes,
        "episode_count": len(pairs),
        "pairs": [{"config_key": k, "query_id": q} for k, q in sorted(pairs)],
    }
    return {**payload, "sha256": _digest(payload)}


def routing_job_id(phase, config_key, query_id):
    if phase not in ("validation", "test") or not config_key or not query_id:
        raise ValueError("invalid routing job identity")
    return f"routing:{_digest([phase, config_key, query_id])}"


def verify_routing_plan(plan):
    payload = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("sha256") != _digest(payload):
        raise ValueError("routing plan hash mismatch")
    pairs = plan.get("pairs", [])
    if (
        plan.get("phase") not in ("validation", "test")
        or len(pairs) != plan.get("episode_count")
        or len(pairs) > plan.get("max_episodes", -1)
    ):
        raise ValueError("invalid routing evaluation plan")
    ids = [routing_job_id(plan["phase"], p["config_key"], p["query_id"]) for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate routing evaluation pair")
    return ids


def reserve_routing_plan(resources, plan):
    """Reserve all distinct physical validation/TEST pairs atomically."""
    ids = verify_routing_plan(plan)
    return resources.reserve_many(plan["phase"], ids) if ids else 0


def paired_test_summary(
    fixed_rows,
    router_rows,
    test_ids,
    *,
    fixed_routes,
    router_routes,
    quality_target,
    seed,
    bootstrap_samples=2000,
):
    """Report paired measured TEST results; bootstrap uncertainty is over tasks, not seeds."""
    import numpy as np

    if (
        len(test_ids) != 200
        or len(set(test_ids)) != 200
        or not math.isfinite(float(quality_target))
        or bootstrap_samples < 100
    ):
        raise ValueError("invalid frozen TEST identities or report parameters")

    def ordered(rows, routes):
        ids = [r["query_id"] for r in rows]
        if len(ids) != 200 or len(set(ids)) != 200 or set(ids) != set(test_ids):
            raise ValueError("paired TEST task identities mismatch")
        if any(
            r.get("split") != "test" or r.get("status") != "evaluated" for r in rows
        ):
            raise ValueError("paired TEST requires completed TEST episodes")
        if set(routes) != set(test_ids) or any(
            r.get("config_key") != routes[r["query_id"]] for r in rows
        ):
            raise ValueError("TEST measurements do not match frozen routes")
        by_id = {r["query_id"]: r for r in rows}
        data = []
        for q in test_ids:
            r = by_id[q]
            vals = [
                float(r[k])
                for k in (
                    "quality",
                    "tokens",
                    "latency_ms",
                    "router_overhead_ms",
                    "router_overhead_tokens",
                    "agent_steps",
                )
            ]
            if (
                not all(math.isfinite(v) for v in vals)
                or min(vals[1:3]) <= 0
                or min(vals[3:]) < 0
            ):
                raise ValueError("invalid TEST measurement")
            quality, tokens, latency, overhead, overhead_tokens, steps = vals
            data.append(
                [quality, tokens + overhead_tokens, latency + overhead, steps, overhead]
            )
        return np.asarray(data)

    if len(set(fixed_routes.values())) != 1:
        raise ValueError("OneConfig must use one configuration")
    one, router = ordered(fixed_rows, fixed_routes), ordered(router_rows, router_routes)
    names = (
        "quality",
        "tokens",
        "online_latency_ms",
        "agent_steps",
        "router_overhead_ms",
    )

    def summary(a):
        mean = dict(zip(names, np.mean(a, axis=0).tolist()))
        mean["target_attained"] = mean["quality"] >= quality_target
        mean["coverage"] = len(a)
        return mean

    delta = router - one
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 200, size=(bootstrap_samples, 200))
    samples = delta[indices].mean(axis=1)
    comparison = {
        name: {
            "estimate": float(delta[:, i].mean()),
            "ci95": [float(x) for x in np.quantile(samples[:, i], [0.025, 0.975])],
        }
        for i, name in enumerate(names)
    }
    return {
        "oneconfig": summary(one),
        "router": summary(router),
        "paired_difference": comparison,
        "quality_target": float(quality_target),
        "uncertainty": "paired task bootstrap; single search seed",
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
    }


def report_frozen_test(
    frozen,
    routes,
    fixed_rows,
    router_rows,
    test_ids,
    *,
    target,
    seed,
    bootstrap_samples=2000,
):
    """Summarize only the validation-selected operating point on paired TEST."""
    verify_frozen_router_policy(frozen)
    if (
        target not in ("100", "95", "90")
        or routes.get("frozen_sha256") != frozen["sha256"]
    ):
        raise ValueError("TEST routes do not match the frozen policy")
    expected = frozen["selection"]["targets"][target]
    if not math.isclose(
        float(routes["quality_target"]),
        float(expected["quality_target"]),
        abs_tol=1e-10,
    ):
        raise ValueError("TEST quality target changed after validation")
    if expected["router"] is None or routes.get("router") is None:
        return {
            "target": target,
            "status": "infeasible_on_validation",
            "frozen_sha256": frozen["sha256"],
        }
    overheads = routes.get("router_overhead")
    if not isinstance(overheads, dict) or set(overheads) != set(test_ids):
        raise ValueError("frozen TEST router overhead is missing")
    for row in fixed_rows:
        if any(
            float(row[k]) != 0 for k in ("router_overhead_ms", "router_overhead_tokens")
        ):
            raise ValueError("OneConfig cannot include router overhead")
    for row in router_rows:
        recorded = overheads.get(row["query_id"])
        if recorded is None or any(
            float(row[k]) != float(recorded[k])
            for k in ("router_overhead_ms", "router_overhead_tokens")
        ):
            raise ValueError("TEST router overhead differs from frozen measurements")
    result = paired_test_summary(
        fixed_rows,
        router_rows,
        test_ids,
        fixed_routes=routes["oneconfig"],
        router_routes=routes["router"],
        quality_target=expected["quality_target"],
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "target": target,
        "status": "complete",
        "frozen_sha256": frozen["sha256"],
        **result,
    }


def checked_raw_score(raw_score, legacy_quality):
    if raw_score is None or legacy_quality is None:
        raise ValueError("missing original score or legacy quality")
    raw, legacy = float(raw_score), float(legacy_quality)
    if (
        not math.isfinite(raw)
        or not math.isfinite(legacy)
        or not math.isclose(max(raw, 0) / 100, legacy, abs_tol=1e-8, rel_tol=0)
    ):
        raise ValueError("original score and legacy quality disagree")
    return raw


def raw_search_rows(rows):
    """Copy completed optimizer rows while recovering their original score."""
    return [
        {
            **r,
            "quality": checked_raw_score(
                (r.get("generation") or {}).get("reward"), r.get("quality")
            ),
            "quality_domain": "raw_environment_score",
        }
        for r in rows
    ]


def checked_semantic_tokens(calls):
    total = 0
    for call in calls:
        usage = call.get("usage") or {}
        value = usage.get("total_tokens")
        if type(value) is not int or value < 0:
            raise ValueError("semantic request usage is missing or invalid")
        total += value
    return total


def validate_historical_episode(row, result, sample_ids):
    """Check one completed receipt; caller still must verify runtime provenance."""
    if row.get("status") != "evaluated":
        raise ValueError("episode not evaluated")
    if row.get("query_id") not in sample_ids:
        raise ValueError("query outside frozen sample")
    if result.get("item_id") != row["query_id"]:
        raise ValueError("episode identity mismatch")
    score = checked_raw_score(
        (row.get("generation") or {}).get("reward"), row.get("quality")
    )
    if result.get("reward") is None or not math.isclose(
        float(result["reward"]), score, abs_tol=1e-8
    ):
        raise ValueError("episode reward does not match DB")
    for name in ("total_tokens", "e2e_ms"):
        if (
            row.get(name) is None
            or not math.isfinite(float(row[name]))
            or float(row[name]) <= 0
        ):
            raise ValueError(f"invalid episode {name}")
    generation = row.get("generation") or {}
    try:
        token_parts = float(generation["semantic_tokens"]) + float(
            generation["generation_tokens"]
        )
        latency_parts = float(generation["retrieval_ms"]) + float(
            generation["episode_ms"]
        )
        if not math.isclose(
            float(row["total_tokens"]), token_parts, abs_tol=1e-8, rel_tol=0
        ) or not math.isclose(
            float(row["e2e_ms"]), latency_parts, abs_tol=1e-6, rel_tol=1e-9
        ):
            raise ValueError("episode cost components do not match totals")
    except (KeyError, TypeError) as exc:
        raise ValueError("episode cost components missing") from exc
    return score


def checked_routing_measurement(
    row, result, *, phase, expected_experiment, manifest_hash, catalog, key_of
):
    """Recover an original validation/TEST score from a matching measured receipt."""
    if phase not in ("validation", "test"):
        raise ValueError("invalid routing measurement phase")
    protocol = row.get("protocol") or {}
    expected_split = "train" if phase == "validation" else "test"
    if (
        row.get("experiment_id") != expected_experiment
        or row.get("split") != expected_split
        or protocol.get("ablation_phase") != phase
        or protocol.get("ablation_manifest_sha256") != manifest_hash
    ):
        raise ValueError("routing measurement phase or manifest mismatch")
    key = key_of(protocol["space_config"])
    if key not in catalog:
        raise ValueError("routing measurement is outside the frozen catalog")
    score = validate_historical_episode(row, result, {row["query_id"]})
    steps = (row.get("generation") or {}).get("agent_steps")
    if steps is None or not math.isfinite(float(steps)) or float(steps) < 0:
        raise ValueError("routing measurement is missing agent steps")
    return {
        "query_id": row["query_id"],
        "config_key": key,
        "retrieval_id": str(row["retrieval_id"]),
        "split": phase,
        "status": "evaluated",
        "quality": score,
        "tokens": float(row["total_tokens"]),
        "latency_ms": float(row["e2e_ms"]),
        "agent_steps": float(steps),
    }


def remaining_admissions(current_configs, requested, *, maximum=30):
    if not 0 <= current_configs <= maximum or requested < 0:
        raise ValueError("invalid configuration count or admission request")
    return min(requested, maximum - current_configs)


def physical_episode_job_id(config_key, query_id, *, arm, shared_initialization):
    if (
        arm not in ARMS
        or not config_key
        or not query_id
        or type(shared_initialization) is not bool
    ):
        raise ValueError("invalid physical episode identity")
    scope = "shared_initialization" if shared_initialization else arm
    return f"candidate:{_digest([scope, config_key, query_id])}"


def validate_system_limits(limits):
    required = (
        "episode_concurrency",
        "cpu_quota_cores",
        "memory_max_bytes",
        "min_available_memory_bytes",
        "max_gpu_utilization_percent",
        "max_cpu_utilization_percent",
    )
    if any(k not in limits for k in required):
        raise ValueError("resource limits are incomplete")
    concurrency, cpu, memory, minimum, gpu, host_cpu = (limits[k] for k in required)
    if (
        type(concurrency) is not int
        or concurrency != 16
        or type(memory) is not int
        or type(minimum) is not int
        or not all(
            math.isfinite(float(v)) and float(v) > 0 for v in (cpu, memory, minimum)
        )
        or not 0 <= float(gpu) <= 100
        or not math.isfinite(float(host_cpu))
        or not 0 < float(host_cpu) <= 100
        or minimum < memory
    ):
        raise ValueError("invalid resource hard limits")
    return True


def validate_resource_preflight(state, limits):
    """Check measured capacity and explicit hard-limit evidence; no side effects."""
    validate_system_limits(limits)
    minimum, gpu = (
        limits["min_available_memory_bytes"],
        limits["max_gpu_utilization_percent"],
    )
    if state.get("cgroup_delegated") is not True:
        raise ValueError(
            "delegated cgroup is required to enforce CPU and memory limits"
        )
    if state.get("available_memory_bytes", -1) < minimum:
        raise ValueError("insufficient available memory")
    devices, utilization = (
        state.get("gpu_ids", []),
        state.get("gpu_utilization_percent", {}),
    )
    if (
        not devices
        or len(devices) != len(set(devices))
        or any(
            device not in utilization
            or not math.isfinite(float(utilization[device]))
            or not 0 <= utilization[device] <= gpu
            for device in devices
        )
    ):
        raise ValueError("GPU utilization is busy or unavailable")
    return True


def read_host_resources():
    """Read Linux host memory and CPU tick counters without touching services."""
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, tail = line.partition(":")
        if key == "MemAvailable":
            values["available_bytes"] = int(tail.split()[0]) * 1024
    fields = Path("/proc/stat").read_text().splitlines()[0].split()
    if fields[0] != "cpu" or len(fields) < 6 or "available_bytes" not in values:
        raise ValueError("host resource counters unavailable")
    ticks = [int(x) for x in fields[1:9]]
    idle = ticks[3] + ticks[4]
    total = sum(ticks)
    return values, (total - idle, total)


class RuntimeResourceMonitor:
    """Stop dispatch and drain owned workers under host memory or CPU pressure."""

    def __init__(self, limits, *, read=read_host_resources):
        validate_system_limits(limits)
        threshold = float(limits["max_cpu_utilization_percent"])
        if not math.isfinite(threshold) or not 0 < threshold <= 100:
            raise ValueError("invalid host CPU pressure threshold")
        self.minimum = limits["min_available_memory_bytes"]
        self.cpu_limit = threshold
        self.read = read
        self.previous = None

    def check(self):
        try:
            memory, (busy, total) = self.read()
            if memory["available_bytes"] < self.minimum:
                raise SystemExit("host memory pressure: pause and drain ablation")
            if self.previous is not None:
                old_busy, old_total = self.previous
                delta, busy_delta = total - old_total, busy - old_busy
                if delta <= 0 or busy_delta < 0 or busy_delta > delta:
                    raise ValueError("host CPU counters are invalid")
                if 100 * busy_delta / delta >= self.cpu_limit:
                    raise SystemExit("host CPU pressure: pause and drain ablation")
            self.previous = (busy, total)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"host resource monitoring failed: {exc}") from exc


def check_initialization(rows, test_ids):
    seen = set()
    counts = Counter()
    for r in rows:
        key = (r["config_key"], r["query_id"])
        if key[1] in test_ids:
            raise ValueError("TRAIN/TEST task ID overlap")
        if key in seen:
            raise ValueError("duplicate initialization pair")
        seen.add(key)
        counts[key[0]] += 1
    return {
        "complete": len(counts) == 10 and all(n == 100 for n in counts.values()),
        "configs": len(counts),
        "episodes": len(rows),
        "per_config": dict(counts),
    }


def initialization_recovery_plan(
    configs, records, test_ids, *, source_identity, target_identity
):
    """Classify frozen initialization pairs without mutating historical results."""
    if len(configs) != 10 or len({c["config_key"] for c in configs}) != 10:
        raise ValueError("initialization requires ten unique configs")
    expected = set()
    for c in configs:
        ids = c["sample_ids"]
        if len(ids) != 100 or len(set(ids)) != 100 or set(ids) & set(test_ids):
            raise ValueError("invalid initialization sample")
        expected.update((c["config_key"], q) for q in ids)
    seen, reused, pending = set(), set(), set()
    identity_ok = runtime_identities_match(source_identity, target_identity)
    for r in records:
        key = (r["config_key"], r["query_id"])
        if key not in expected:
            raise ValueError("receipt outside frozen initialization")
        if key in seen:
            raise ValueError("duplicate initialization receipt")
        seen.add(key)
        if (
            r["status"] == "evaluated"
            and r.get("local_verified") is True
            and identity_ok
        ):
            reused.add(key)
        elif r["status"] in ("retrieving", "retrieved_pending_evaluation"):
            pending.add(key)
    missing = expected - reused - pending

    def pairs(keys):
        return [{"config_key": c, "query_id": q} for c, q in sorted(keys)]

    payload = {
        "identity_matches": identity_ok,
        "reusable": pairs(reused),
        "blocked_pending": pairs(pending),
        "missing": pairs(missing),
        "reusable_count": len(reused),
        "blocked_pending_count": len(pending),
        "missing_count": len(missing),
    }
    return {**payload, "sha256": _digest(payload)}


def initialization_recovery_jobs(configs, plan):
    """Validate the complete frozen plan and return its unique budget IDs."""
    payload = {k: v for k, v in plan.items() if k != "sha256"}
    if plan.get("sha256") != _digest(payload):
        raise ValueError("initialization recovery plan hash mismatch")
    if plan.get("blocked_pending") or plan.get("blocked_pending_count") != 0:
        raise ValueError("historical pending episodes must be resolved before recovery")
    if len(configs) != 10 or len({c["config_key"] for c in configs}) != 10:
        raise ValueError("initialization requires ten distinct configurations")
    expected = set()
    for c in configs:
        if len(c["sample_ids"]) != 100 or len(set(c["sample_ids"])) != 100:
            raise ValueError("invalid initialization sample")
        expected.update((c["config_key"], q) for q in c["sample_ids"])
    pairs = []
    for category in ("reusable", "missing"):
        values = plan.get(category, [])
        if len(values) != plan.get(f"{category}_count"):
            raise ValueError("initialization plan count mismatch")
        pairs.extend((item["config_key"], item["query_id"]) for item in values)
    if (
        len(pairs) != 1000
        or len(set(pairs)) != 1000
        or set(pairs) != expected
        or (plan.get("reusable") and not plan.get("identity_matches"))
    ):
        raise ValueError("initialization plan does not cover the frozen sample")
    jobs = [
        physical_episode_job_id(c, q, arm="full", shared_initialization=True)
        for c, q in sorted(pairs)
    ]
    return jobs


def reserve_initialization_recovery(ledger, configs, plan):
    """Reserve the entire shared sample before any recovery episode starts."""
    return ledger.reserve_many("candidate", initialization_recovery_jobs(configs, plan))


def shared_initialization_snapshot(
    configs, receipts, test_ids, *, source_identity, target_identity
):
    """Freeze exactly the 10 x 100 completed sample; never infer absent outcomes."""
    if not runtime_identities_match(source_identity, target_identity):
        raise ValueError(
            "runtime identity must be complete and match before initialization reuse"
        )
    if len(configs) != 10 or len({c["config_key"] for c in configs}) != 10:
        raise ValueError("shared initialization requires ten distinct configurations")
    expected = set()
    for c in configs:
        ids = c["sample_ids"]
        if len(ids) != 100 or len(set(ids)) != 100 or set(ids) & set(test_ids):
            raise ValueError("invalid or TEST-overlapping initialization sample")
        expected.update((c["config_key"], q) for q in ids)
    if not check_initialization(receipts, set(test_ids))["complete"]:
        raise ValueError("initialization has incomplete measurements")
    if {(r["config_key"], r["query_id"]) for r in receipts} != expected:
        raise ValueError("initialization receipts do not match frozen samples")
    if any(not r.get("retrieval_id") for r in receipts) or len(
        {str(r["retrieval_id"]) for r in receipts}
    ) != len(receipts):
        raise ValueError("duplicate source retrieval IDs")
    for r in receipts:
        vals = [float(r[k]) for k in ("reward", "tokens", "latency_ms")]
        if not all(math.isfinite(v) for v in vals) or min(vals[1:]) <= 0:
            raise ValueError("invalid initialization measurement")
    payload = {
        "version": 1,
        "configs": configs,
        "receipts": receipts,
        "test_ids": sorted(test_ids),
        "source_identity": source_identity,
        "target_identity": target_identity,
        "episode_count": len(receipts),
    }
    return {**payload, "sha256": _digest(payload)}


def verify_shared_initialization(snapshot):
    payload = {k: v for k, v in snapshot.items() if k != "sha256"}
    if snapshot.get("sha256") != _digest(payload):
        raise ValueError("shared initialization snapshot hash mismatch")
    if snapshot.get("episode_count") != 1000 or not runtime_identities_match(
        snapshot.get("source_identity", {}), snapshot.get("target_identity", {})
    ):
        raise ValueError("shared initialization is not fully validated")
    reconstructed = shared_initialization_snapshot(
        snapshot["configs"],
        snapshot["receipts"],
        snapshot["test_ids"],
        source_identity=snapshot["source_identity"],
        target_identity=snapshot["target_identity"],
    )
    if reconstructed != snapshot:
        raise ValueError("shared initialization structure is invalid")
    return True


def finalize_manifest(
    draft,
    snapshot,
    training_pool,
    *,
    test_ids,
    artifact_hashes,
    resource_limits,
    system_limits,
    identity_evidence,
    memory_database=None,
):
    """Freeze a complete, isolated seed; this does not enable execution."""
    verify_manifest(draft)
    verify_shared_initialization(snapshot)
    verify_identity_evidence(snapshot, identity_evidence)
    if memory_database is not None:
        verify_isolated_memory_evidence(snapshot, identity_evidence, memory_database)
    validate_isolated_pool(training_pool, draft)
    validate_artifact_hashes(artifact_hashes)
    validate_system_limits(system_limits)
    ResourceLedger("unused", draft["manifest_sha256"], resource_limits)
    validation = set(draft["validation_ids"])
    train_ids = {item["item_id"] for item in training_pool["items"]}
    initial_ids = {q for config in snapshot["configs"] for q in config["sample_ids"]}
    if len(test_ids) != 200 or len(set(test_ids)) != 200 or len(validation) != 50:
        raise ValueError("invalid frozen validation/TEST identities")
    if (set(test_ids) & (train_ids | validation | initial_ids)) or validation & (
        train_ids | initial_ids
    ):
        raise ValueError("TRAIN, validation and TEST identities overlap")
    if set(snapshot["test_ids"]) != set(test_ids):
        raise ValueError("snapshot TEST identities do not match")
    if (
        set(draft["source_config_keys"])
        != {c["config_key"] for c in snapshot["configs"]}
        or not initial_ids <= train_ids
    ):
        raise ValueError("initialization does not match frozen TRAIN catalog")
    for receipt in snapshot["receipts"]:
        if not receipt.get("source_experiment") or not re.fullmatch(
            r"[0-9a-f]{64}", str(receipt.get("result_sha256", ""))
        ):
            raise ValueError("initialization receipt lacks source provenance")
        raw = Path(receipt["result_file"]).read_bytes()
        result = json.loads(raw)
        if (
            hashlib.sha256(raw).hexdigest() != receipt["result_sha256"]
            or result.get("item_id") != receipt["query_id"]
            or not math.isclose(
                float(result["reward"]),
                float(receipt["reward"]),
                abs_tol=1e-8,
                rel_tol=0,
            )
        ):
            raise ValueError("initialization source result does not match receipt")
    payload = {
        **{k: v for k, v in draft.items() if k != "manifest_sha256"},
        "stage": "seed_prepared",
        "runtime_identity_attested": False,
        "recorded_identity_evidence_verified": True,
        "live_runtime_identity_verified": False,
        "shared_initialization_sha256": snapshot["sha256"],
        # The draft exists before initialization; its hash keeps one ledger
        # identity across initialization and the finalized search manifest.
        "resource_ledger_binding": draft["manifest_sha256"],
        "test_ids_sha256": _digest(sorted(test_ids)),
        "artifact_sha256": dict(artifact_hashes),
        "resource_limits": dict(resource_limits),
        "system_limits": dict(system_limits),
        "identity_evidence": copy.deepcopy(identity_evidence),
        "memory_database": memory_database,
    }
    return {**payload, "manifest_sha256": _digest(payload)}


def seed_initial_round(source_round, snapshot, *, key_of, sample_path):
    """Create isolated round metadata only after every shared receipt is verified.

    Files and DB rows still need independent copying and validation before the
    resulting state can be used. No historical timestamps are manufactured.
    """
    verify_shared_initialization(snapshot)
    if source_round.get("round") != 0 or len(source_round.get("configs", ())) != 10:
        raise ValueError("source must contain exactly ten round-zero configs")
    expected = {c["config_key"]: c["sample_ids"] for c in snapshot["configs"]}
    result = copy.deepcopy(source_round)
    seen, paths = set(), set()
    for c in result["configs"]:
        key = key_of(c["config"])
        if key in seen or key not in expected or c["sample"]["ids"] != expected[key]:
            raise ValueError("source configuration/sample does not match snapshot")
        seen.add(key)
        destination = str(sample_path(key))
        if (
            not destination
            or destination in paths
            or destination == c["sample"].get("file")
        ):
            raise ValueError("sample destination must be unique and isolated")
        paths.add(destination)
        c["sample"]["file"] = destination
        historical = c.get("staging") or {}
        c["historical_staging"] = copy.deepcopy(historical)
        c["staging"] = {
            **historical,
            "stage_reached": 100,
            "episodes": 100,
            "pruned_at": None,
            "decisions": [],
            "events": [],
        }
        c["shared_initialization_sha256"] = snapshot["sha256"]
    if seen != set(expected):
        raise ValueError("source configurations differ from snapshot")
    result["shared_initialization_sha256"] = snapshot["sha256"]
    result["completion_time"] = None
    result["seeded_round_sha256"] = _digest(result)
    return result


def install_shared_round(fresh_state, seeded_round, snapshot, *, key_of):
    """Return independent arm state metadata; caller must first materialize receipts."""
    verify_shared_initialization(snapshot)
    if (
        fresh_state.get("dataset") != "sciworld"
        or fresh_state.get("ablation_arm") not in ARMS
        or fresh_state.get("rounds") != []
    ):
        raise ValueError("shared initialization requires fresh ScienceWorld arm state")
    if (
        seeded_round.get("round") != 0
        or seeded_round.get("shared_initialization_sha256") != snapshot["sha256"]
    ):
        raise ValueError("seeded round identity mismatch")
    if seeded_round.get("seeded_round_sha256") != _digest(
        {k: v for k, v in seeded_round.items() if k != "seeded_round_sha256"}
    ):
        raise ValueError("seeded round hash mismatch")
    configs = seeded_round.get("configs", [])
    if len(configs) != 10 or any(
        c.get("shared_initialization_sha256") != snapshot["sha256"]
        or c.get("staging", {}).get("stage_reached") != 100
        or c.get("staging", {}).get("pruned_at") is not None
        for c in configs
    ):
        raise ValueError("seeded round is not fully completed")
    expected = {c["config_key"]: c["sample_ids"] for c in snapshot["configs"]}
    actual = {key_of(c["config"]): c["sample"]["ids"] for c in configs}
    if len(actual) != 10 or actual != expected:
        raise ValueError("seeded round sample identities mismatch")
    state = copy.deepcopy(fresh_state)
    state["rounds"] = [copy.deepcopy(seeded_round)]
    state["shared_initialization_sha256"] = snapshot["sha256"]
    state["shared_initialization_wall_s"] = None
    return state


def copy_initial_episode(receipt, target):
    """Copy one hash-verified original episode without changing either copy on resume."""
    source, target = Path(receipt["result_file"]), Path(target)
    if target.is_symlink() or source.resolve() == target.resolve():
        raise ValueError("source and destination must be distinct")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != receipt["result_sha256"]:
        raise ValueError("source result hash mismatch")
    result = json.loads(raw)
    if result.get("item_id") != receipt["query_id"] or not math.isclose(
        float(result["reward"]), float(receipt["reward"]), abs_tol=1e-8, rel_tol=0
    ):
        raise ValueError("source result identity or reward mismatch")
    return _install_verified_bytes(raw, target, digest)


def _install_verified_bytes(raw, target, digest):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(name, target)
            return True
        except FileExistsError:
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("destination conflicts with verified source") from None
            return False
    finally:
        os.unlink(name)


def checked_initial_sample(source, sample_ids, sample_hash, *, seed, pool_hash):
    """Read and verify the historical sample without writing it."""
    source = Path(source)
    if source.is_symlink():
        raise ValueError("source sample must not be a symlink")
    raw = source.read_bytes()
    sample = json.loads(raw)
    ids = [item["item_id"] for item in sample.get("items", [])]
    if (
        len(sample_ids) != 100
        or len(set(sample_ids)) != 100
        or ids != sample_ids
        or sample.get("dataset") != "sciworld"
        or sample.get("sample_hash") != sample_hash
        or sample.get("seed") != seed
        or sample.get("pool_hash") != pool_hash
        or hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16] != sample_hash
    ):
        raise ValueError("source sample identity mismatch")
    return raw


def copy_initial_sample(source, target, sample_ids, sample_hash, *, seed, pool_hash):
    """Copy a source sample with the exact frozen task order and hash."""
    source, target = Path(source), Path(target)
    if target.is_symlink() or source.resolve() == target.resolve():
        raise ValueError("source and destination must be distinct")
    raw = checked_initial_sample(
        source, sample_ids, sample_hash, seed=seed, pool_hash=pool_hash
    )
    return _install_verified_bytes(raw, target, hashlib.sha256(raw).hexdigest())


def copy_optimizer_row(row, receipt, destination, target_experiment, *, key_of):
    """Prepare one independently identified DB row after copying its verified file."""
    if (
        str(row["retrieval_id"]) != str(receipt["retrieval_id"])
        or row["experiment_id"] != receipt["source_experiment"]
        or row["query_id"] != receipt["query_id"]
        or key_of(row["protocol"]["space_config"]) != receipt["config_key"]
    ):
        raise ValueError("source optimizer row identity mismatch")
    if not target_experiment or target_experiment == row["experiment_id"]:
        raise ValueError("destination experiment must be isolated")
    destination = Path(destination)
    raw = destination.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["result_sha256"]:
        raise ValueError("copied destination hash mismatch")
    result = json.loads(raw)
    score = validate_historical_episode(row, result, {receipt["query_id"]})
    if (
        not math.isclose(score, float(receipt["reward"]), abs_tol=1e-8)
        or not math.isclose(
            float(row["total_tokens"]), float(receipt["tokens"]), abs_tol=1e-8
        )
        or not math.isclose(
            float(row["e2e_ms"]), float(receipt["latency_ms"]), abs_tol=1e-6
        )
    ):
        raise ValueError("source cost or score differs from frozen receipt")
    out = copy.deepcopy(row)
    out["retrieval_id"] = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"trace-ablation:{target_experiment}:{receipt['retrieval_id']}",
    )
    out["experiment_id"] = target_experiment
    out["protocol"]["branched_from"] = {
        "experiment": receipt["source_experiment"],
        "retrieval_id": str(receipt["retrieval_id"]),
        "result_sha256": receipt["result_sha256"],
    }
    out["protocol"]["bo_experiment"] = target_experiment
    out["generation"]["source_result_file"] = row["generation"]["result_file"]
    out["generation"]["result_file"] = str(destination.resolve())
    return out


def insert_copied_initial_rows(connection, snapshot, rows, target_experiment):
    """Insert a complete copied initialization transactionally and idempotently.

    Every row must have been prepared by ``copy_optimizer_row`` after file
    verification. Connection must use psycopg's dict_row factory. Callers must
    use a new experiment ID, never a historical ID.
    """
    from psycopg.types.json import Jsonb

    from trimem.experiments.pool_branch import JSON_COLUMNS, ROW_COLUMNS
    from trimem.optimizer import session_space_agentenv_v2 as space

    verify_shared_initialization(snapshot)
    receipts = {str(r["retrieval_id"]): r for r in snapshot["receipts"]}
    if len(rows) != 1000 or not target_experiment:
        raise ValueError("initial row copy requires 1000 rows and a destination")
    seen_sources, seen_destinations, seen_paths = set(), set(), set()
    for row in rows:
        source = (row.get("protocol") or {}).get("branched_from") or {}
        source_id = str(source.get("retrieval_id", ""))
        receipt = receipts.get(source_id)
        if (
            not receipt
            or source_id in seen_sources
            or row.get("experiment_id") != target_experiment
            or not receipt.get("source_experiment")
            or not receipt.get("result_sha256")
            or receipt.get("source_experiment") == target_experiment
            or source.get("experiment") != receipt.get("source_experiment")
            or source.get("result_sha256") != receipt.get("result_sha256")
            or space.key(row["protocol"]["space_config"]) != receipt["config_key"]
            or row.get("query_id") != receipt["query_id"]
            or row.get("status") != "evaluated"
            or row.get("split") != "train"
            or not math.isclose(
                float(row["total_tokens"]), float(receipt["tokens"]), abs_tol=1e-8
            )
            or not math.isclose(
                float(row["e2e_ms"]), float(receipt["latency_ms"]), abs_tol=1e-6
            )
            or not math.isclose(
                float((row.get("generation") or {})["reward"]),
                float(receipt["reward"]),
                abs_tol=1e-8,
            )
        ):
            raise ValueError("copied row conflicts with shared initialization")
        expected_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"trace-ablation:{target_experiment}:{source_id}"
        )
        if (
            str(row.get("retrieval_id")) != str(expected_id)
            or str(expected_id) in seen_destinations
        ):
            raise ValueError("invalid or duplicate destination retrieval ID")
        checked_raw_score(receipt["reward"], row["quality"])
        destination = Path(row["generation"]["result_file"])
        resolved = destination.resolve()
        if (
            destination.is_symlink()
            or resolved == Path(receipt["result_file"]).resolve()
            or resolved in seen_paths
        ):
            raise ValueError("destination result paths must be distinct and isolated")
        data = destination.read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt["result_sha256"]:
            raise ValueError("copied result file hash mismatch")
        validate_historical_episode(row, json.loads(data), {receipt["query_id"]})
        seen_destinations.add(str(expected_id))
        seen_sources.add(source_id)
        seen_paths.add(resolved)
    expected_destination_ids = {str(r["retrieval_id"]) for r in rows}
    with connection.transaction():
        existing = connection.execute(
            "SELECT retrieval_id FROM opt.session_retrieval WHERE experiment_id=%s AND split='train'",
            (target_experiment,),
        ).fetchall()
        if any(
            str(r["retrieval_id"]) not in expected_destination_ids for r in existing
        ):
            raise ValueError("destination experiment contains unrelated training rows")
        for row in rows:
            vals = [
                Jsonb(row[k]) if k in JSON_COLUMNS and row[k] is not None else row[k]
                for k in ROW_COLUMNS
            ]
            connection.execute(
                f"INSERT INTO opt.session_retrieval ({', '.join(ROW_COLUMNS)}) VALUES ({', '.join(['%s'] * len(ROW_COLUMNS))}) ON CONFLICT (retrieval_id) DO NOTHING",
                vals,
            )
            stored = connection.execute(
                "SELECT experiment_id, query_id, status, quality, total_tokens, e2e_ms, protocol, generation, config FROM opt.session_retrieval WHERE retrieval_id=%s",
                (row["retrieval_id"],),
            ).fetchone()
            fields = (
                "experiment_id",
                "query_id",
                "status",
                "quality",
                "total_tokens",
                "e2e_ms",
                "protocol",
                "generation",
                "config",
            )
            if not stored or any(stored[k] != row[k] for k in fields):
                raise ValueError(
                    "existing destination row conflicts with copied measurement"
                )
    return len(rows)


def seed_shared_arm(
    fresh_state,
    source_round,
    snapshot,
    manifest,
    source_rows,
    *,
    campaign_root,
    target_dir,
    source_pool_hash,
    connection,
):
    """Materialize verified shared initialization; caller persists state last.

    This routine does not run any episodes. On interruption, matching files and
    destination rows are verified and reused. The caller must separately attach
    ledgers and enforce resources before enabling the experiment driver.
    """
    from trimem.optimizer import session_space_agentenv_v2 as space

    verify_manifest(manifest)
    verify_shared_initialization(snapshot)
    verify_identity_evidence(snapshot, manifest["identity_evidence"])
    if (
        manifest.get("stage") != "seed_prepared"
        or manifest.get("shared_initialization_sha256") != snapshot["sha256"]
    ):
        raise ValueError("manifest is not bound to the completed shared initialization")
    if (
        fresh_state.get("dataset") != "sciworld"
        or fresh_state.get("ablation_arm") not in ARMS
        or fresh_state.get("rounds") != []
    ):
        raise ValueError("target must be a fresh isolated ScienceWorld arm")
    if not fresh_state.get("experiment"):
        raise ValueError("target experiment identity missing")
    by_id = {str(r["retrieval_id"]): r for r in source_rows}
    expected_ids = {str(r["retrieval_id"]) for r in snapshot["receipts"]}
    if len(by_id) != 1000 or set(by_id) != expected_ids:
        raise ValueError("source DB rows do not match the frozen initialization")
    campaign_root, target_dir = (
        Path(campaign_root).resolve(),
        Path(target_dir).resolve(),
    )
    if (
        not campaign_root.name.startswith("scienceworld_trace_ablation_")
        or target_dir == campaign_root
        or not target_dir.is_relative_to(campaign_root)
    ):
        raise ValueError(
            "target directory must be isolated within a new ablation campaign"
        )

    def cfg_dir(key):
        return "cfg_" + hashlib.sha256(key.encode()).hexdigest()[:12]

    def sample_path(key):
        return target_dir / "samples" / (cfg_dir(key) + ".json")

    seeded = seed_initial_round(
        source_round, snapshot, key_of=space.key, sample_path=sample_path
    )
    for c in source_round["configs"]:
        key, sample = space.key(c["config"]), c["sample"]
        copy_initial_sample(
            sample["file"],
            sample_path(key),
            sample["ids"],
            sample["sample_hash"],
            seed=sample["seed"],
            pool_hash=source_pool_hash,
        )
    copied = []
    for receipt in snapshot["receipts"]:
        q = receipt["query_id"]
        if not re.fullmatch(r"sciworld_\d+", q):
            raise ValueError("invalid ScienceWorld query identity")
        destination = (
            target_dir / "configs" / cfg_dir(receipt["config_key"]) / (q + ".json")
        )
        copy_initial_episode(receipt, destination)
        copied.append(
            copy_optimizer_row(
                by_id[str(receipt["retrieval_id"])],
                receipt,
                destination,
                fresh_state["experiment"],
                key_of=space.key,
            )
        )
    insert_copied_initial_rows(connection, snapshot, copied, fresh_state["experiment"])
    ledger_path = target_dir / "episode_ledger.json"
    seed_arm_episode_ledger(
        ledger_path,
        snapshot,
        arm=fresh_state["ablation_arm"],
        target_experiment=fresh_state["experiment"],
        manifest_hash=manifest["manifest_sha256"],
    )
    resource_ledger_path = campaign_root / "resource_ledger.sqlite"
    seed_global_resource_ledger(resource_ledger_path, snapshot, manifest)
    state = install_shared_round(fresh_state, seeded, snapshot, key_of=space.key)
    state["ablation_manifest_sha256"] = manifest["manifest_sha256"]
    state["ablation_policy"] = arm_policy(state["ablation_arm"])
    state["ablation_episode_ledger"] = str(ledger_path)
    state["ablation_resource_ledger"] = str(resource_ledger_path)
    state["ablation_resource_ledger_binding"] = manifest["resource_ledger_binding"]
    state["ablation_resource_limits"] = dict(manifest["resource_limits"])
    state["ablation_system_limits"] = dict(manifest["system_limits"])
    return state


def save_seeded_arm_state(state, path, *, campaign_root):
    """Publish the initial state last; never overwrite an existing run."""
    path, root = Path(path).resolve(), Path(campaign_root).resolve()
    if (
        not root.name.startswith("scienceworld_trace_ablation_")
        or not path.is_relative_to(root)
        or path.name != "state.json"
    ):
        raise ValueError("state destination is not an isolated ablation run")
    if (
        state.get("dataset") != "sciworld"
        or state.get("ablation_arm") not in ARMS
        or not state.get("shared_initialization_sha256")
        or not state.get("ablation_manifest_sha256")
        or len(state.get("rounds", [])) != 1
    ):
        raise ValueError("state is not a fully seeded ablation arm")
    raw = (json.dumps(state, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return _install_verified_bytes(raw, path, hashlib.sha256(raw).hexdigest())


def verify_seeded_arm(state, snapshot, manifest, *, connection):
    """Recheck shared rows, files and both ledgers on resume without writing."""
    from trimem.optimizer import session_space_agentenv_v2 as space

    verify_manifest(manifest)
    verify_shared_initialization(snapshot)
    verify_identity_evidence(snapshot, manifest["identity_evidence"])
    if (
        state.get("ablation_arm") not in ARMS
        or state.get("dataset") != "sciworld"
        or state.get("ablation_policy") != arm_policy(state.get("ablation_arm"))
        or state.get("ablation_manifest_sha256") != manifest["manifest_sha256"]
        or state.get("shared_initialization_sha256") != snapshot["sha256"]
        or manifest.get("shared_initialization_sha256") != snapshot["sha256"]
        or state.get("ablation_resource_ledger_binding")
        != manifest.get("resource_ledger_binding")
        or state.get("ablation_system_limits") != manifest.get("system_limits")
    ):
        raise ValueError("seeded state identity mismatch")
    configs = state["rounds"][0]["configs"]
    actual = {space.key(c["config"]): c["sample"]["ids"] for c in configs}
    expected = {c["config_key"]: c["sample_ids"] for c in snapshot["configs"]}
    if len(actual) != 10 or actual != expected:
        raise ValueError("seeded state sample identities mismatch")
    ledger = EpisodeLedger.open(
        state["ablation_episode_ledger"], manifest["manifest_sha256"]
    )
    ledger_data = ledger._read()
    if ledger_data["arm"] != state["ablation_arm"]:
        raise ValueError("seeded episode ledger arm mismatch")
    resource = ResourceLedger.open(
        state["ablation_resource_ledger"],
        manifest["resource_ledger_binding"],
        manifest["resource_limits"],
    )
    with sqlite3.connect(resource.path) as db:
        resource_rows = {
            job_id: (status, metadata)
            for job_id, status, metadata in db.execute(
                "SELECT job_id,status,metadata FROM completions"
            )
        }
    for receipt in snapshot["receipts"]:
        source_id = str(receipt["retrieval_id"])
        target_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"trace-ablation:{state['experiment']}:{source_id}"
        )
        row = connection.execute(
            "SELECT experiment_id,query_id,status,quality,total_tokens,e2e_ms,protocol,generation,config FROM opt.session_retrieval WHERE retrieval_id=%s",
            (target_id,),
        ).fetchone()
        if (
            not row
            or row["experiment_id"] != state["experiment"]
            or row["query_id"] != receipt["query_id"]
            or row["status"] != "evaluated"
            or row["protocol"].get("branched_from", {}).get("retrieval_id") != source_id
            or space.key(row["protocol"]["space_config"]) != receipt["config_key"]
            or not math.isclose(
                float(row["total_tokens"]), float(receipt["tokens"]), abs_tol=1e-8
            )
            or not math.isclose(
                float(row["e2e_ms"]), float(receipt["latency_ms"]), abs_tol=1e-6
            )
        ):
            raise ValueError("seeded optimizer row mismatch")
        data = Path(row["generation"]["result_file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt["result_sha256"]:
            raise ValueError("seeded result file hash mismatch")
        validate_historical_episode(row, json.loads(data), {receipt["query_id"]})
        key = json.dumps(
            [receipt["config_key"], receipt["query_id"]], separators=(",", ":")
        )
        logical = ledger_data["pairs"].get(key, {})
        if (
            logical.get("status") != "complete"
            or logical.get("retrieval_id") != str(target_id)
            or any(
                logical.get(k) != float(receipt[k])
                for k in ("reward", "tokens", "latency_ms")
            )
        ):
            raise ValueError("seeded episode ledger incomplete")
        job_id = physical_episode_job_id(
            receipt["config_key"],
            receipt["query_id"],
            arm=state["ablation_arm"],
            shared_initialization=True,
        )
        status, metadata = resource_rows.get(job_id, (None, "{}"))
        if (
            status != "complete"
            or json.loads(metadata).get("source_retrieval_id") != source_id
        ):
            raise ValueError("shared resource ledger incomplete")
    return True


class EpisodeLedger:
    """Per-arm durable admission ledger; a reserved pair consumes its quota."""

    def __init__(self, path, manifest_hash):
        self.path = Path(path)
        self.manifest_hash = manifest_hash

    @classmethod
    def create(cls, path, arm, manifest_hash):
        arm_policy(arm)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "arm": arm, "manifest_hash": manifest_hash, "pairs": {}}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        return cls(path, manifest_hash)

    @classmethod
    def open(cls, path, manifest_hash):
        ledger = cls(path, manifest_hash)
        ledger._read()
        return ledger

    def _read(self):
        data = json.loads(self.path.read_text())
        if data.get("manifest_hash") != self.manifest_hash or data.get("version") != 1:
            raise ValueError("ledger manifest/version mismatch")
        arm_policy(data["arm"])
        return data

    def _update(self, fn):
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = self._read()
            changed, result = fn(data)
            if changed:
                fd, name = tempfile.mkstemp(
                    prefix=self.path.name + ".", dir=self.path.parent
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(data, f, sort_keys=True)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(name, self.path)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)
            return result

    def admit(self, config_key, query_id):
        return bool(self.admit_many(config_key, [query_id]))

    def admit_many(self, config_key, query_ids):
        if (
            not config_key
            or not query_ids
            or len(query_ids) != len(set(query_ids))
            or any(not q for q in query_ids)
        ):
            raise ValueError("invalid episode admission batch")

        def update(data):
            pairs = data["pairs"]
            additions = [
                (json.dumps([config_key, q], separators=(",", ":")), q)
                for q in query_ids
            ]
            additions = [(key, q) for key, q in additions if key not in pairs]
            if not additions:
                return False, 0
            configs = {v["config_key"] for v in pairs.values()}
            if len(pairs) + len(additions) > 3000 or (
                config_key not in configs and len(configs) >= 30
            ):
                raise ValueError("candidate budget exhausted")
            if (
                sum(v["config_key"] == config_key for v in pairs.values())
                + len(additions)
                > 100
            ):
                raise ValueError("configuration panel exhausted")
            for key, query_id in additions:
                pairs[key] = {
                    "config_key": config_key,
                    "query_id": query_id,
                    "status": "reserved",
                }
            return True, len(additions)

        return self._update(update)

    def complete(
        self, config_key, query_id, *, retrieval_id, reward, tokens, latency_ms
    ):
        values = [float(reward), float(tokens), float(latency_ms)]
        if not all(math.isfinite(v) for v in values) or min(values[1:]) <= 0:
            raise ValueError("invalid completed measurement")

        def update(data):
            key = json.dumps([config_key, query_id], separators=(",", ":"))
            if key not in data["pairs"]:
                raise ValueError("cannot complete an unreserved episode")
            row = data["pairs"][key]
            if any(
                other_key != key and other.get("retrieval_id") == str(retrieval_id)
                for other_key, other in data["pairs"].items()
            ):
                raise ValueError("duplicate retrieval identity")
            completed = {
                **row,
                "status": "complete",
                "retrieval_id": str(retrieval_id),
                "reward": values[0],
                "tokens": values[1],
                "latency_ms": values[2],
            }
            if row["status"] == "complete":
                if row != completed:
                    raise ValueError("conflicting completion receipt")
                return False, False
            data["pairs"][key] = completed
            return True, True

        return self._update(update)

    def counts(self):
        counts = Counter(row["status"] for row in self._read()["pairs"].values())
        return {"reserved": counts["reserved"], "complete": counts["complete"]}

    def complete_many(self, receipts):
        """Reconcile a batch atomically, including conflicts and duplicate IDs."""
        normalized = []
        for r in receipts:
            values = [float(r[k]) for k in ("reward", "tokens", "latency_ms")]
            if (
                not r.get("retrieval_id")
                or not all(math.isfinite(v) for v in values)
                or min(values[1:]) <= 0
            ):
                raise ValueError("invalid completed measurement")
            key = json.dumps([r["config_key"], r["query_id"]], separators=(",", ":"))
            normalized.append((key, str(r["retrieval_id"]), values))
        if len({key for key, _, _ in normalized}) != len(normalized) or len(
            {rid for _, rid, _ in normalized}
        ) != len(normalized):
            raise ValueError("duplicate batch identity")

        def update(data):
            pairs = data["pairs"]
            old_ids = {
                r["retrieval_id"]: key
                for key, r in pairs.items()
                if r.get("retrieval_id")
            }
            changes = []
            for key, rid, values in normalized:
                if key not in pairs:
                    raise ValueError("cannot complete an unreserved episode")
                if rid in old_ids and old_ids[rid] != key:
                    raise ValueError("duplicate retrieval identity")
                row = pairs[key]
                completed = {
                    **row,
                    "status": "complete",
                    "retrieval_id": rid,
                    "reward": values[0],
                    "tokens": values[1],
                    "latency_ms": values[2],
                }
                if row["status"] == "complete":
                    if row != completed:
                        raise ValueError("conflicting completion receipt")
                else:
                    changes.append((key, completed))
            for key, completed in changes:
                pairs[key] = completed
            return bool(changes), len(changes)

        return self._update(update)


class ResourceLedger:
    """Atomic, restart-safe reservations for physical and auxiliary work."""

    CATEGORIES = frozenset(
        {"candidate", "validation", "test", "retry", "analyzer", "nomemory", "noise"}
    )

    def __init__(self, path, manifest_hash, limits):
        self.path = Path(path)
        self.identity = _digest({"manifest_hash": manifest_hash, "limits": limits})
        if set(limits) != self.CATEGORIES or any(
            type(v) is not int or v < 0 for v in limits.values()
        ):
            raise ValueError("all resource hard limits must be nonnegative integers")
        if limits["candidate"] > protocol()["unique_candidate_episode_upper_bound"]:
            raise ValueError("candidate physical budget exceeds protocol")
        self.limits = dict(limits)

    @classmethod
    def create(cls, path, manifest_hash, limits):
        ledger = cls(path, manifest_hash, limits)
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(ledger.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        with sqlite3.connect(ledger.path) as db:
            db.execute("CREATE TABLE meta (identity TEXT NOT NULL)")
            db.execute("INSERT INTO meta VALUES (?)", (ledger.identity,))
            db.execute(
                "CREATE TABLE reservations (job_id TEXT PRIMARY KEY, category TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE completions (job_id TEXT PRIMARY KEY, status TEXT NOT NULL, tokens REAL NOT NULL, elapsed_s REAL NOT NULL, metadata TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE dispatches (job_id TEXT PRIMARY KEY, attempts INTEGER NOT NULL)"
            )
        return ledger

    @classmethod
    def open(cls, path, manifest_hash, limits):
        ledger = cls(path, manifest_hash, limits)
        if not ledger.path.is_file():
            raise ValueError("resource ledger missing")
        with sqlite3.connect(ledger.path) as db:
            if db.execute("SELECT identity FROM meta").fetchone() != (ledger.identity,):
                raise ValueError("resource ledger identity mismatch")
        return ledger

    def reserve(self, category, job_id):
        return bool(self.reserve_many(category, [job_id]))

    def reserve_many(self, category, job_ids):
        if (
            category not in self.CATEGORIES
            or not job_ids
            or len(job_ids) != len(set(job_ids))
            or any(not j for j in job_ids)
        ):
            raise ValueError("invalid resource reservation")
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT identity FROM meta").fetchone() != (self.identity,):
                raise ValueError("resource ledger identity mismatch")
            additions = []
            for job_id in job_ids:
                old = db.execute(
                    "SELECT category FROM reservations WHERE job_id = ?", (job_id,)
                ).fetchone()
                if old and old[0] != category:
                    raise ValueError("conflicting reservation identity")
                if not old:
                    additions.append(job_id)
            count = db.execute(
                "SELECT COUNT(*) FROM reservations WHERE category = ?", (category,)
            ).fetchone()[0]
            if count + len(additions) > self.limits[category]:
                raise ValueError(f"{category} budget exhausted")
            db.executemany(
                "INSERT INTO reservations VALUES (?, ?)",
                [(job_id, category) for job_id in additions],
            )
            return len(additions)

    def complete(self, job_id, *, tokens, elapsed_s, status, metadata):
        if (
            status not in ("complete", "failed")
            or not all(
                math.isfinite(float(v)) and float(v) >= 0 for v in (tokens, elapsed_s)
            )
            or not isinstance(metadata, dict)
        ):
            raise ValueError("invalid resource completion")
        data = (
            status,
            float(tokens),
            float(elapsed_s),
            json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT identity FROM meta").fetchone() != (self.identity,):
                raise ValueError("resource ledger identity mismatch")
            if not db.execute(
                "SELECT 1 FROM reservations WHERE job_id = ?", (job_id,)
            ).fetchone():
                raise ValueError("resource job is not reserved")
            old = db.execute(
                "SELECT status,tokens,elapsed_s,metadata FROM completions WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if old:
                if old != data:
                    raise ValueError("conflicting resource completion")
                return False
            db.execute(
                "INSERT INTO completions VALUES (?, ?, ?, ?, ?)", (job_id, *data)
            )
            return True

    def mark_dispatch(self, job_id):
        """Conservatively charge each launch after the first as a retry."""
        return self.mark_dispatch_many([job_id])[job_id]

    def mark_dispatch_many(self, job_ids):
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("duplicate dispatch identity")
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT identity FROM meta").fetchone() != (self.identity,):
                raise ValueError("resource ledger identity mismatch")
            counts = {}
            for job_id in job_ids:
                if not db.execute(
                    "SELECT 1 FROM reservations WHERE job_id = ?", (job_id,)
                ).fetchone():
                    raise ValueError("resource job is not reserved")
                if db.execute(
                    "SELECT 1 FROM completions WHERE job_id = ? AND status='complete'",
                    (job_id,),
                ).fetchone():
                    counts[job_id] = None
                else:
                    old = db.execute(
                        "SELECT attempts FROM dispatches WHERE job_id = ?", (job_id,)
                    ).fetchone()
                    counts[job_id] = old[0] if old else 0
            used = db.execute(
                "SELECT COUNT(*) FROM reservations WHERE category='retry'"
            ).fetchone()[0]
            if (
                used + sum(value is not None and value > 0 for value in counts.values())
                > self.limits["retry"]
            ):
                raise ValueError("retry budget exhausted")
            result = {}
            for job_id, count in counts.items():
                if count is None:
                    result[job_id] = 0
                    continue
                if count:
                    retry_id = f"dispatch-retry:{_digest([job_id, count + 1])}"
                    db.execute(
                        "INSERT INTO reservations VALUES (?, 'retry')", (retry_id,)
                    )
                db.execute(
                    "INSERT INTO dispatches VALUES (?, ?) ON CONFLICT(job_id) DO UPDATE SET attempts=excluded.attempts",
                    (job_id, count + 1),
                )
                result[job_id] = count + 1
            return result

    def counts(self):
        with sqlite3.connect(self.path) as db:
            if db.execute("SELECT identity FROM meta").fetchone() != (self.identity,):
                raise ValueError("resource ledger identity mismatch")
            counts = dict(
                db.execute(
                    "SELECT category, COUNT(*) FROM reservations GROUP BY category"
                )
            )
        return {category: counts.get(category, 0) for category in self.CATEGORIES}

    def summary(self):
        """Sum job durations; concurrent durations are not wall-clock time."""
        with sqlite3.connect(self.path) as db:
            if db.execute("SELECT identity FROM meta").fetchone() != (self.identity,):
                raise ValueError("resource ledger identity mismatch")
            rows = db.execute(
                "SELECT r.category, COUNT(*), SUM(CASE WHEN c.status='complete' THEN 1 ELSE 0 END), SUM(CASE WHEN c.status='failed' THEN 1 ELSE 0 END), COALESCE(SUM(c.tokens),0), COALESCE(SUM(c.elapsed_s),0) FROM reservations r LEFT JOIN completions c USING(job_id) GROUP BY r.category"
            ).fetchall()
        result = {
            category: {
                "reserved": 0,
                "complete": 0,
                "failed": 0,
                "pending": 0,
                "tokens": 0,
                "elapsed_s": 0,
            }
            for category in self.CATEGORIES
        }
        for category, reserved, complete, failed, tokens, elapsed_s in rows:
            result[category] = {
                "reserved": reserved,
                "complete": complete,
                "failed": failed,
                "pending": reserved - complete - failed,
                "tokens": tokens,
                "elapsed_s": elapsed_s,
            }
        return result


def seed_arm_episode_ledger(path, snapshot, *, arm, target_experiment, manifest_hash):
    """Account the common initialization once in an arm's own logical budget."""
    verify_shared_initialization(snapshot)
    arm_policy(arm)
    if not target_experiment or not manifest_hash:
        raise ValueError("missing destination or manifest identity")
    path = Path(path)
    if path.exists():
        ledger = EpisodeLedger.open(path, manifest_hash)
        if ledger._read()["arm"] != arm:
            raise ValueError("episode ledger arm mismatch")
    else:
        ledger = EpisodeLedger.create(path, arm, manifest_hash)
    for config in snapshot["configs"]:
        ledger.admit_many(config["config_key"], config["sample_ids"])
    completions = [
        {
            **r,
            "retrieval_id": str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"trace-ablation:{target_experiment}:{r['retrieval_id']}",
                )
            ),
        }
        for r in snapshot["receipts"]
    ]
    ledger.complete_many(completions)
    return ledger


def seed_global_resource_ledger(path, snapshot, manifest):
    """Record shared physical episodes once across all five logical arms."""
    verify_shared_initialization(snapshot)
    verify_manifest(manifest)
    if manifest.get("shared_initialization_sha256") != snapshot["sha256"]:
        raise ValueError("resource ledger snapshot mismatch")
    path = Path(path)
    limits = manifest["resource_limits"]
    ledger = (
        ResourceLedger.open(path, manifest["resource_ledger_binding"], limits)
        if path.exists()
        else ResourceLedger.create(path, manifest["resource_ledger_binding"], limits)
    )
    for r in snapshot["receipts"]:
        job_id = physical_episode_job_id(
            r["config_key"], r["query_id"], arm="full", shared_initialization=True
        )
        ledger.reserve("candidate", job_id)
        ledger.complete(
            job_id,
            tokens=r["tokens"],
            elapsed_s=float(r["latency_ms"]) / 1000,
            status="complete",
            metadata={
                "source_retrieval_id": str(r["retrieval_id"]),
                "source_experiment": r["source_experiment"],
                "elapsed_kind": "episode_latency_not_wall_time",
            },
        )
    return ledger


def reconcile_candidate_ledger(ledger, rows):
    """Commit evaluated raw-score rows; incomplete rows remain reserved."""
    evaluated = [r for r in rows if r.get("status") == "evaluated"]
    keys = [(r["config_key"], r["query_id"]) for r in evaluated]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate candidate measurements")
    if any(r.get("quality_domain") != "raw_environment_score" for r in evaluated):
        raise ValueError("candidate quality must be raw environment score")
    receipts = [
        {
            "config_key": r["config_key"],
            "query_id": r["query_id"],
            "retrieval_id": r["retrieval_id"],
            "reward": r["quality"],
            "tokens": r["total_tokens"],
            "latency_ms": r["e2e_ms"],
        }
        for r in evaluated
    ]
    return ledger.complete_many(receipts)


def reserve_candidate_prefix(episodes, resources, arm, config_key, query_ids):
    """Reserve the logical arm quota and global physical quota before dispatch."""
    arm_policy(arm)
    if episodes._read()["arm"] != arm:
        raise ValueError("candidate ledger arm mismatch")
    episodes.admit_many(config_key, query_ids)
    for query_id in query_ids:
        job_id = physical_episode_job_id(
            config_key, query_id, arm=arm, shared_initialization=False
        )
        resources.reserve("candidate", job_id)


def reconcile_dispatched_candidates(episodes, resources, arm, rows):
    """Account completed physical work idempotently after its observed DB rows."""
    arm_policy(arm)
    if episodes._read()["arm"] != arm:
        raise ValueError("candidate ledger arm mismatch")
    changed = reconcile_candidate_ledger(episodes, rows)
    for r in rows:
        if r.get("status") != "evaluated":
            continue
        job_id = physical_episode_job_id(
            r["config_key"], r["query_id"], arm=arm, shared_initialization=False
        )
        resources.complete(
            job_id,
            tokens=r["total_tokens"],
            elapsed_s=float(r["e2e_ms"]) / 1000,
            status="complete",
            metadata={
                "retrieval_id": str(r["retrieval_id"]),
                "elapsed_kind": "episode_latency_not_wall_time",
            },
        )
    return changed


class BudgetedAnalyzer:
    """Reserve and account each actual model call, including failed calls."""

    def __init__(self, client, resources, arm, round_number, *, quota_check=None):
        arm_policy(arm)
        self.client, self.resources, self.arm, self.round_number = (
            client,
            resources,
            arm,
            round_number,
        )
        self.quota_check = quota_check or check_global_quota

    def chat(self, *args, **kwargs):
        self.quota_check()
        job_id = f"analyzer:{self.arm}:{self.round_number}:{uuid.uuid4().hex}"
        try:
            self.resources.reserve("analyzer", job_id)
        except ValueError as exc:
            raise SystemExit(f"analyzer budget or ledger rejected call: {exc}") from exc
        started = time.monotonic()
        result = self.client.chat(*args, **kwargs)
        if (
            result.input_tokens is None
            or result.output_tokens is None
            or result.latency_ms is None
        ):
            raise SystemExit(
                "analyzer usage unavailable; call remains reserved for audit"
            )
        try:
            self.resources.complete(
                job_id,
                tokens=result.input_tokens + result.output_tokens,
                elapsed_s=float(result.latency_ms) / 1000,
                status="failed" if result.error else "complete",
                metadata={
                    "arm": self.arm,
                    "round": self.round_number,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "call_wall_s": time.monotonic() - started,
                },
            )
        except ValueError as exc:
            raise SystemExit(f"analyzer usage could not be recorded: {exc}") from exc
        return result


def retry_budget_from_environment(env=None):
    """Load a worker retry ledger only when a complete ablation context is set."""
    env = os.environ if env is None else env
    names = (
        "TRIMEM_ABLATION_RESOURCE_LEDGER",
        "TRIMEM_ABLATION_MANIFEST_HASH",
        "TRIMEM_ABLATION_RESOURCE_LIMITS",
    )
    values = [env.get(name) for name in names]
    if not any(values):
        return None
    if not all(values):
        raise SystemExit("incomplete ablation retry-ledger environment")
    try:
        return ResourceLedger.open(values[0], values[1], json.loads(values[2]))
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"cannot open ablation retry ledger: {exc}") from exc


def worker_retry_environment(state):
    path, manifest_hash, limits = (
        state["ablation_resource_ledger"],
        state.get(
            "ablation_resource_ledger_binding", state["ablation_manifest_sha256"]
        ),
        state["ablation_resource_limits"],
    )
    ResourceLedger(path, manifest_hash, limits)
    result = {
        "TRIMEM_ABLATION_RESOURCE_LEDGER": str(path),
        "TRIMEM_ABLATION_MANIFEST_HASH": manifest_hash,
        "TRIMEM_ABLATION_RESOURCE_LIMITS": json.dumps(
            limits, sort_keys=True, separators=(",", ":")
        ),
    }
    if "ablation_system_limits" in state:
        validate_system_limits(state["ablation_system_limits"])
        result["TRIMEM_ABLATION_SYSTEM_LIMITS"] = json.dumps(
            state["ablation_system_limits"], sort_keys=True, separators=(",", ":")
        )
    return result


def reserve_api_retry(ledger):
    """Charge a retry before issuing it; unknown failed-call costs remain pending."""
    try:
        ledger.reserve("retry", f"api-retry:{uuid.uuid4().hex}")
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"retry budget or ledger rejected call: {exc}") from exc
