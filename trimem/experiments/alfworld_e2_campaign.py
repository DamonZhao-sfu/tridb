"""E2 (ALFWorld / ScienceWorld / WebShop): prepare/dry-run, bounded discovery, resume, and OneConfig only.

Prepare reads TRAIN evidence but performs no episode, generation or optimizer
proposal calls. Execute requires a separate user-confirmation receipt bound to
the manifest. A free queue lock is never interpreted as authorization.

The module name is historical: it is dataset-generic (``prepare --dataset``, default
alfworld; later modes read the frozen manifest's dataset). Per-dataset values:
``alfworld_e2_evidence.DATASETS``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import alfworld_e2_evidence as ev

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/exp2_alfworld_prepare_20261022"
SOURCE = ROOT / "results/exp2_oneconfig_prepare_20261022/provenance/alfworld_state.json"
HISTORICAL = ROOT / "results/bo_agentenv_20261016/alfworld_pool-strat200-v7__mem-sf1"
METHODS = ("mo-tpe-v1", "gp-lognehvi-v1")
SEEDS = (20261022, 20261023, 20261024)
# Run-tag letter per method (e2{letter}{seed%100000}...); original pair keeps "t"/"g".
SHORT = {"mo-tpe-v1": "tpe", "gp-lognehvi-v1": "gp", "lhs-v1": "lhs"}
MODEL = "Qwen/Qwen3.8-27B-FP8"


def now():
    return datetime.now(UTC).isoformat()


def driver():
    from trimem.experiments import bo_pool_driver as d

    return d


def isolated_env(dataset="alfworld", evaluation=False):
    """The campaign's complete knob set; ``evaluation``: a process that never runs ``profile()``
    (holdout / OneConfig TEST) and must export a recorded SemFilter regime itself (WebShop v2b)."""
    s = ev.spec(dataset)
    # Inherited knobs belong to other running experiments, not this campaign.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("POOL_", "TRIMEM_", "TAU2_"))
    }
    upper = dataset.upper()
    env.update(
        PYTHONPATH=str(ROOT),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TRIMEM_MEMORY_POLICY="success+failure-v1",
        POOL_RUNTIME_PROFILE="controlled-c16-v1",
        # controlled-c16-v1 requires profiling = test = 16 (bo_pool_driver.runtime_profile_want)
        **{f"POOL_CONCURRENCY_{upper}": "16", f"POOL_TEST_CONCURRENCY_{upper}": "16"},
        POOL_SEMFILTER="v2",
        POOL_NO_LIMEQO="1",
        MAX_ENVS="32",
        TRIMEM_SEMFILTER_V2="1",
        TRIMEM_SEMFILTER_TOPN="32",
        TRIMEM_SEMFILTER_BATCH="8",
        TRIMEM_SEMFILTER_MAX_TOKENS="96",
        TRIMEM_SEMFILTER_JUDGE_INPUT="raw",
        TRIMEM_SEMFILTER_RUBRIC="semantic-rubric-v3",
        TRIMEM_SEMFILTER_GUIDED="1",
    )
    if s["semfilter_v2b"]:
        # The v2 block records the default rubric (v3, as TRACE); v2b then exports rubric v4. A resume checks the
        # caller's TRIMEM_SEMFILTER_* against the effective (v2b) env, so discovery must not pin rubric v3.
        del env["TRIMEM_SEMFILTER_RUBRIC"]
    env.update(s["env"])
    if evaluation:
        env.update(s["evaluation_env"])
    return env


@contextlib.contextmanager
def scoped_environ(env):
    """Run driver hooks that read/export ``os.environ`` under ``env``; restore the caller's environment."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@contextlib.contextmanager
def proposal_rule(d, dataset, a, settings):
    """Apply the round-0 proposal rule exactly as ``plan_round`` does at runtime.

    ``profile()`` records the SemFilter blocks (``semfilter_block``, ``semfilter_v2b_block``) on a fresh state
    before ``plan_round(r=0)`` applies ``use_semfilter_v2b_for`` and only then calls ``baselines.initialize``.
    Freezing the initialization without that rule would differ from the runtime picks. ALFWorld / ScienceWorld:
    no rule (unchanged).
    """
    if not ev.spec(dataset)["semfilter_v2b"]:
        yield None
        return
    probe = {
        "dataset": dataset,
        "rounds": [],
        "search_design": copy.deepcopy(settings),
        **d.fresh_panel_settings(a),
    }
    with scoped_environ(isolated_env(dataset)):
        d.semfilter_block(dataset, probe, Path(os.devnull), now())
        v2b = d.semfilter_v2b_block(dataset, probe, Path(os.devnull), now())
    bo = d.bo_module()
    try:
        if (
            v2b is None
            or v2b["from_round"] != 0
            or not bo.use_semfilter_v2b_for(probe, 0)
        ):
            raise ValueError("semfilter-v2b must apply from round 0")
        yield v2b
    finally:
        bo.pool.use_proposal_rule(None)


def frozen_initialization(d, dataset, a):
    """(backend settings, round-0 picks) as the runtime's ``plan_round(r=0)`` computes them."""
    from trimem.optimizer import search_analyzer as sa

    settings = sa.settings_for({}, a)
    with proposal_rule(d, dataset, a, settings):
        init = sa.baselines.initialize(d.bo_module(), settings)
    return settings, init


def source_hashes():
    # Freeze the complete in-repository Python execution surface, including
    # deferred imports and the evaluator launched in a different interpreter.
    files = sorted((ROOT / "trimem").rglob("*.py"))
    files += sorted((ROOT / "vendor/AgentGym/agentenv").rglob("*.py"))
    return {str(p.relative_to(ROOT)): ev.file_hash(p) for p in files}


def read_rows(experiment, split="train"):
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(driver().drv.OPT_DSN, row_factory=dict_row) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        rows = conn.execute(
            "SELECT experiment_id,dataset,split,query_id,status,config,protocol,quality,paired_gain,"
            "total_tokens,e2e_ms,created_at,updated_at,"
            "receipt->'semfilter'->'judge_total_tokens' AS judge_tokens FROM opt.session_retrieval "
            "WHERE experiment_id=%s AND split=%s AND status=%s",
            (experiment, split, "evaluated"),
        ).fetchall()
    # Receipts contain raw memory text. Retain only measured judge usage.
    for r in rows:
        r["total_tokens"] = (
            float(r["total_tokens"]) if r["total_tokens"] is not None else None
        )
        for k in ("created_at", "updated_at"):
            r[k] = r[k].isoformat()
    return rows


def snapshot_services(dataset="alfworld"):
    d = driver()
    ports = ev.spec(dataset)["env_ports"]
    result = {"captured_at": now(), "services": {}}
    if dataset != "alfworld":
        result["env_ports"] = [str(p) for p in ports]
    for port, path in [
        (8030, "v1/models"),
        (8031, "v1/models"),
        *((p, "managed_stats") for p in ports),
    ]:
        try:
            result["services"][str(port)] = d.drv.get(
                f"http://127.0.0.1:{port}/{path}", timeout=3
            )
        except (OSError, ValueError) as exc:
            result["services"][str(port)] = {"unavailable": str(exc)}
    # Process names/IDs only: avoid copying other sessions' complete command lines.
    ps = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    result["evaluation_processes"] = [
        {
            "pid": line.split()[0],
            "ppid": line.split()[1],
            "module": next(
                (s for s in line.split() if s.startswith("trimem.experiments.")),
                "unknown",
            ),
        }
        for line in ps.splitlines()
        if " -m trimem.experiments." in line
        and any(
            x in line
            for x in (
                "agentgym_session_eval",
                "agentgym_official_eval",
                "bo_pool_driver",
                "tau2.pool_profile",
                "alf_test",
                # the other datasets' own evaluators share the same vLLM servers
                *(() if dataset == "alfworld" else ("sci_test", "webshop_test")),
            )
        )
    ]
    result["model_revision"] = (
        "not exposed by /v1/models; must attest exact serving revision before launch"
    )
    result["serving_load"] = {}
    for port in (8030, 8031):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=3
            ) as response:
                metrics = response.read().decode()
            values = [
                float(line.rsplit(" ", 1)[1])
                for line in metrics.splitlines()
                if line.startswith(
                    ("vllm:num_requests_running", "vllm:num_requests_waiting")
                )
            ]
            result["serving_load"][str(port)] = sum(values) if values else None
        except (OSError, ValueError):
            result["serving_load"][str(port)] = None
    return result


def require_idle(snapshot):
    if snapshot["evaluation_processes"]:
        raise ValueError("shared evaluator workload is active; leave it untouched")
    for port in ("8030", "8031"):
        models = [x["id"] for x in snapshot["services"][port].get("data", [])]
        if MODEL not in models:
            raise ValueError(f"model/service unavailable at {port}")
        if snapshot.get("serving_load", {}).get(port) != 0:
            raise ValueError(f"serving {port} has active/unknown load")
    for port in snapshot.get("env_ports", ("36001", "36011")):
        s = snapshot["services"][port]
        if "unavailable" in s or s.get("live_envs", 0) or s.get("jvms", 0):
            raise ValueError(f"env {port} unavailable or occupied")


def freeze(path, value):
    if path.exists():
        if ev.read_json(path) != value:
            raise ValueError(
                f"frozen artifact differs: {path}; use a new prepare directory"
            )
    else:
        ev.write_json(path, value)


def reference_baseline(trace, pool, destination=None, dataset="alfworld"):
    """Freeze the historical c16 TRAIN NoMemory reference; never buy a new sample.

    ALFWorld: the TRACE latency-regime rerun (200). ScienceWorld: the c16 NoMemory arm (200). WebShop: the
    recorded nomemory-prefix-v1 100-task prefix of the TRACE NoMemory sample (its remaining tasks never ran).
    """
    s = ev.spec(dataset)
    sample = trace["nomemory"]["sample"]
    directory = Path(
        trace["latency_regime"]["nomemory"]["dir"]
        if s["nomemory_dir"] is None
        else ROOT / s["nomemory_dir"]
    )
    args_file = directory / "run_args.json"
    captured = {}

    def read(path):
        path = Path(path)
        captured[path] = path.read_bytes()
        return json.loads(captured[path])

    args = read(args_file)
    # Exactly what the historical run_args.json records (processes / format differ per dataset).
    if any(args.get(k) != v for k, v in s["nomemory_run_args"].items()):
        raise ValueError("historical NoMemory runtime does not match the E2 reference")
    panel = read(sample["file"])
    ids = sample["ids"]
    if (
        len(ids) != 200
        or len(ids) != len(set(ids))
        or ids != [x["item_id"] for x in panel["items"]]
        or not set(ids) <= {x["item_id"] for x in pool["items"]}
    ):
        raise ValueError("historical NoMemory sample is not a unique TRAIN subset")
    if s["nomemory_prefix"] is not None:
        used = trace["nomemory"].get("sample_used") or {}
        prefix = used.get("ids") or []
        if (
            used.get("protocol") != s["nomemory_prefix"]
            or len(prefix) != s["nomemory_size"]
            or prefix != trace["nomemory"]["order"][: len(prefix)]
            or not set(prefix) <= set(ids)
        ):
            raise ValueError(
                "historical NoMemory prefix is not the recorded stage prefix"
            )
        ids = list(prefix)
        sample = {
            **sample,
            "ids": ids,
            "used": {
                "protocol": s["nomemory_prefix"],
                "size": len(ids),
                "of_sample_hash": sample.get("sample_hash"),
                "rule": "TRACE state.json nomemory.sample_used (= nomemory.order prefix); the panel file is the "
                "full TRACE NoMemory sample, ``ids`` is the reused prefix",
            },
        }
    elif len(ids) != s["nomemory_size"]:
        raise ValueError(
            "historical NoMemory sample size differs from the E2 reference"
        )
    paths = [Path(sample["file"]), args_file]
    for q in ids:
        path = directory / f"{q}.json"
        row = read(path)
        reward = row.get("reward")
        if (
            row.get("item_id") != q
            or not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(reward)
        ):
            raise ValueError(f"invalid historical NoMemory result: {q}")
        paths.append(path)
    paths += [
        directory / n
        for n in ("episodes.jsonl", "llm_calls.jsonl")
        if (directory / n).exists()
    ]
    for path in paths:
        if path not in captured:
            captured[path] = path.read_bytes()
    source_hashes = {str(p): hashlib.sha256(captured[p]).hexdigest() for p in paths}
    source_directory = str(directory)
    if destination is not None:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if len({p.name for p in paths}) != len(paths):
            raise ValueError("historical reference filenames collide")
        for path in paths:
            target = destination / path.name
            if target.exists():
                if ev.file_hash(target) != source_hashes[str(path)]:
                    raise ValueError(f"frozen reference snapshot differs: {target}")
            else:
                target.write_bytes(captured[path])
            if ev.file_hash(target) != source_hashes[str(path)]:
                raise ValueError(f"reference changed while snapshotting: {path}")
            target.chmod(0o444)
        sample = {**sample, "file": str(destination / Path(sample["file"]).name)}
        paths = [destination / p.name for p in paths]
        directory = destination
        directory.chmod(0o555)
    extra = {} if dataset == "alfworld" else {"dataset": dataset}
    return {
        **extra,
        "protocol": "historical-nomemory-reuse-v1",
        "directory": str(directory),
        "source_directory": source_directory,
        "source_files_sha256": source_hashes,
        "sample": sample,
        "reused_tasks": len(ids),
        "new_episodes": 0,
        "files_sha256": {str(p): ev.file_hash(p) for p in paths},
        "identity_limit": "Historical model ID and runtime settings checked; exact serving revision and evaluator hash were not recorded by this legacy reference. User-authorized reuse, not exact-identity TEST reuse.",
    }


def install_reference(d, reference):
    """Expose a read-only historical baseline while preserving candidate sampling."""
    original = d.write_sample
    original_directory = d.nomemory_dir
    dataset = ev.dataset_of(reference)

    def directory(ds, sampling="per-config", *, regime=True):
        # load_state resets _NM_REGIME for fresh runs and resumes without a
        # latency migration. The campaign's frozen reference is independent of it.
        if ds == dataset:
            return Path(reference["directory"])
        return original_directory(ds, sampling, regime=regime)

    d.nomemory_dir = directory

    def sample(ds, pool, base, name, label, exp, size=200, uniform=False):
        if name != "NoMemory":
            return original(ds, pool, base, name, label, exp, size, uniform)
        if ds != dataset:
            raise ValueError(f"E2 reference only applies to {dataset}")
        source = reference["sample"]
        path = Path(base) / "samples/NoMemory.json"
        freeze(path, ev.read_json(source["file"]))
        return {**source, "file": str(path)}

    d.write_sample = sample


def run_argv(dataset, method, seed, out, pause_after_round=None):
    """(run tag, bo_pool_driver argv) of one discovery run; identical protocol for every dataset."""
    tag = f"e2{SHORT[method][0]}{seed % 100000}{ev.digest(str(out))[:6]}"
    argv = [
        "profile",
        "--dataset",
        dataset,
        "--run_tag",
        tag,
        "--search_design",
        method,
        "--baseline_seed",
        str(seed),
        "--full_panel_size",
        "100",
        "--initial_budget_episodes",
        "5000",
        "--task_sample_namespace",
        f"{dataset}-e2-{seed}",
        "--batch",
        "5",
        "--max_configs",
        "80",
        "--rounds",
        "8",
        "--pruning_policy",
        "syftr-style",
        "--no_analyst",
        "--no_limeqo",
        "--no_reuse",
    ] + (
        []
        if pause_after_round is None
        else ["--pause_after_round", str(pause_after_round)]
    )
    return tag, argv


def prepare(
    out, methods=METHODS, seeds=SEEDS, pause_after_round=None, dataset="alfworld"
):
    """``pause_after_round`` N freezes an early stop: profiling halts before round N+1 is proposed."""
    from trimem.optimizer import session_space_agentenv_v2 as space

    ds, s = dataset, ev.spec(dataset)
    d = driver()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        if ev.dataset_of(ev.read_json(out / "manifest.json")) != ds:
            raise ValueError("frozen prepare directory belongs to another dataset")
        return dry_run(
            out
        )  # preparation is idempotent, never rewrites a frozen campaign
    env = isolated_env(ds)
    os.environ.clear()
    os.environ.update(env)
    pool = d.load_pool(ds)
    test_files = [
        ROOT / f"results/eval_controlled_20261015/shards/{ds}_{x}.json"
        for x in ("a", "b")
    ]
    test_ids = [x["item_id"] for p in test_files for x in ev.read_json(p)]
    train_ids = [x["item_id"] for x in pool["items"]]
    if len(test_ids) != len(set(test_ids)) or set(test_ids) & set(train_ids):
        raise ValueError("TRAIN/TEST overlap or duplicate TEST task")
    (out / "provenance").mkdir(exist_ok=True)
    source = SOURCE if ds == "alfworld" else ROOT / s["trace_state"]
    historical = HISTORICAL if ds == "alfworld" else ROOT / s["trace_dir"]
    trace = ev.read_json(source)
    if ev.dataset_of(trace) != ds:
        raise ValueError(f"TRACE state {source} is not {ds}")
    rows = read_rows(trace["experiment"])
    trace_audit_error = None
    try:
        catalog, audit = ev.audited_catalog(
            trace, rows, historical, pool, historical=True
        )
    except Exception as exc:  # noqa: BLE001
        if s["trace_required"]:
            raise
        # TRACE is provenance only for this dataset (direct TEST runs --no-trace): record, never block.
        trace_audit_error = repr(exc)
        catalog = []
        audit = [{"eligible": False, "reason": f"TRACE audit failed: {exc!r}"}]
    ev.write_json(out / f"provenance/{ds}_state.json", trace)
    ev.write_json(out / "provenance/trace_train_rows.json", rows)
    ev.write_json(out / "trace_catalog.json", catalog)
    ev.write_csv(
        out / "trace_catalog_audit.csv",
        audit,
        [
            "key",
            "source_round",
            "display_round",
            "expected",
            "completed_opt",
            "completed_files",
            "sample_hash",
            "runtime",
            "eligible",
            "reason",
            "path",
        ],
    )
    # Prespecified, common bounds; no TEST or per-method tuning. Natural logs.
    normalization = {
        "source": "predefined-v1",
        "objectives": ["quality", "-ln(tokens)", "-ln(latency_ms)"],
        "lower": [0, -math.log(1000000), -math.log(3600000)],
        "upper": [1, -math.log(1000), -math.log(1000)],
        "reference": [0, 0, 0],
        "clip": False,
    }
    grid = {
        "source": "predefined-v1",
        "latency": [10000 + 24500 * i for i in range(21)],
        "tokens": [10000 + 14500 * i for i in range(21)],
        "fixed_latency_ms": 500000,
        "fixed_tokens": 300000,
    }
    campaign = ROOT / "results" / out.name.replace("prepare", "campaign")
    if campaign == out:
        campaign = out / "campaign"
    if campaign.exists():
        raise ValueError(
            "campaign output already exists; resume its original frozen manifest"
        )
    d.OUT = campaign / "runs"
    runs = []
    methods, seeds = tuple(methods), tuple(seeds)
    if not methods or len(set(methods)) != len(methods) or set(methods) - SHORT.keys():
        raise ValueError(f"methods must be distinct members of {sorted(SHORT)}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    for seed in seeds:
        init_keys = []
        for method in methods:
            short = SHORT[method]
            tag, argv = run_argv(ds, method, seed, out, pause_after_round)
            a = d.build_parser().parse_args(argv)
            os.environ[d.RUN_TAG_ENV] = tag
            settings, init = frozen_initialization(d, ds, a)
            init_keys.append([space.key(x["config"]) for x in init])
            state = d.new_state(ds, pool, a)
            state["staging"] = d.staging_block(a)
            prepared = []
            for pick in init:
                cfg = pick["config"]
                sample = d.write_sample(
                    ds,
                    pool,
                    out / "initialization" / str(seed),
                    d.cfg_name(cfg),
                    space.key(cfg),
                    state["task_sample_namespace"],
                    size=100,
                )
                c = {"config": cfg, "sample": sample}
                order = d.config_order(
                    state, c, {x["item_id"]: x["stratum"] for x in pool["items"]}
                )
                prepared.append({"config": cfg, "sample": sample, "stage_order": order})
            runs.append(
                {
                    "id": f"{short}-{seed}",
                    "method": method,
                    "seed": seed,
                    "tag": tag,
                    "argv": argv,
                    "experiment": state["experiment"],
                    "directory": str(d.run_dir(ds)),
                    "settings": settings,
                    "staging": state["staging"],
                    "initialization": prepared,
                }
            )
        if any(keys != init_keys[0] for keys in init_keys):
            raise ValueError("methods have different initialization")
    identity = {
        "model": MODEL,
        "model_revision": "pending attestation",
        "trial": 0,
        "agent_trial_seed": None,
        "agent_trial_rule": "one temperature=0 trial; existing evaluator exposes no seed option",
        "memory_policy": "success+failure-v1",
        "runtime": "controlled-c16-v1",
        "semfilter": s["semfilter"]["protocol"],
        **s["identity_extra"],
        "evaluator_sha256": ev.file_hash(
            ROOT / "trimem/experiments/agentgym_session_eval.py"
        ),
        "memory_snapshot": "pending attestation",
        "semfilter_code_sha256": ev.file_hash(
            ROOT / "trimem/experiments/semfilter_v2.py"
        ),
    }
    reference = reference_baseline(trace, pool, out / "provenance/nomemory", ds)
    limits = {
        "candidate_evaluations_per_run": 5000,
        "candidate_evaluations_total": 5000 * len(runs),
        "max_unique_configs": 80,
        "batch": 5,
        "shared_nomemory": 0,
        "shared_nomemory_reused": reference["reused_tasks"],
        "noise_repeat_per_run": 100,
        "pilot_new_episodes": 0,
        "canary": "each round-0 stage-50 prefix verified independently, included in candidate quota",
        "retry_attempts_per_run": 250,
        "automatic_retries": 0,
        "max_attempts_per_pair": 2,
        "discovery_wall_seconds_per_run": 48 * 3600,
        "discovery_wall_seconds_total": len(runs) * 48 * 3600,
        "test_policy_requests_upper_bound": len(test_ids) * 42 * 7,
        "test_episode_cap": None,
        "test_retry_cap": 0,
        "test_wall_seconds": None,
    }
    manifest = {
        "protocol": f"{ds}-search-e2-v1",
        "dataset": ds,
        "status": "prepared",
        "source_prompt": "paper/sigmod/exp2_alfworld_search_ablation_prompt_v0.1.0.md",
        "campaign_dir": str(campaign),
        "methods": list(methods),
        "seeds": list(seeds),
        "pause_after_round": pause_after_round,
        "runs": runs,
        "pool_file": str(ROOT / f"data/external/panels/{ds}_pool.json"),
        "pool_sha256": ev.file_hash(ROOT / f"data/external/panels/{ds}_pool.json"),
        "test_files": {str(p): ev.file_hash(p) for p in test_files},
        "test_ids": test_ids,
        "train_weights": pool["strata"],
        "space_version": space.VERSION,
        "plans": list(space.BASE_PLANS) + ["P1+Rank"],
        "grid": space.grid(),
        "space_sha256": ev.file_hash(
            ROOT / "trimem/optimizer/session_space_agentenv_v2.py"
        ),
        "normalization": normalization,
        "slo_grid": grid,
        "limits": limits,
        "reference_baseline": reference,
        "identity": identity,
        "fixed_memory": {
            "edges": "similar+similar_up",
            "memory_policy": "success+failure-v1",
            "memory_form": "raw",
            "sim_gate": "none",
        },
        "semfilter": trace["semfilter"],
        "code_sha256": source_hashes(),
        "trace_state_sha256": ev.file_hash(source),
        "historical_discovery_new_episodes": 0,
        "trace_catalog_sha256": ev.file_hash(out / "trace_catalog.json"),
        "trace_inputs_sha256": {
            name: ev.file_hash(out / "provenance" / name)
            for name in (f"{ds}_state.json", "trace_train_rows.json")
        },
        "episode_python": str(ROOT / ".venv-agentgym/bin/python"),
        "search_python": str(ROOT / ".venv-search-baselines/bin/python"),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("optuna", "botorch", "torch", "numpy", "scipy")
        },
        "authorization": "pending; separate receipt bound to manifest SHA256",
        "search_time": "sum of active search segments (candidate profiling + optimizer + stage/noise waits); common NoMemory warmup and manual downtime excluded",
        "search_tokens": "candidate profiling agent+judge tokens + optimizer LLM tokens; shared NoMemory, noise repeats and TEST separately accounted",
        "staging_mode": "async",
        "scheduling": "per-configuration asynchronous checkpoints; retained configurations continue as capacity permits; total concurrency 16; automatic drain/rebalance disabled",
        "phase_order": "first seed pair, then remaining paired seeds; stop on any failure",
    }
    if ds != "alfworld":
        manifest.update(
            trace_from_round=s["trace_from_round"],
            trace_audit_error=trace_audit_error,
            max_round=s["max_round"],
            semfilter_identity=s["semfilter"],
        )
    ev.write_json(out / "manifest.json", manifest)
    environment = snapshot_services(ds)
    environment["python"] = sys.version
    environment["packages"] = manifest["packages"]
    environment["episode_runner_note"] = (
        "Existing half_cmd uses .venv-agentgym/bin/python, not the prompt shorthand .venv/bin/python; preserved and frozen."
    )
    environment["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    ev.write_json(out / "environment.json", environment)
    # Save exact rubric + judge implementation, not only version names.
    for name in ("semantic_rubric.py", "semfilter_v2.py"):
        (out / "provenance" / name).write_bytes(
            (ROOT / "trimem/experiments" / name).read_bytes()
        )
    differences = [
        {
            "field": "sampler",
            "new": "randomized LHS + actual backend",
            "TRACE": "legacy + Sobol",
        },
        {
            "field": "pruner",
            "new": "syftr-style z=1.645 full comparators",
            "TRACE": "historical staged z=1.96 + stage comparators / guard",
        },
        {
            "field": "start",
            "new": "cold ten-point init",
            "TRACE": "source round 6 shown as 0, rounds 0-5 influenced search"
            if ds == "alfworld"
            else f"source round {s['trace_from_round']} shown as 0; TRACE is provenance only "
            "(direct TEST runs --no-trace)",
        },
        {
            "field": "plans",
            "new": "five plans from round zero",
            "TRACE": "historical P1 extension also present",
        },
        {
            "field": "repeats",
            "new": "three paired seeds",
            "TRACE": "one historical archive",
        },
        {
            "field": "cost coverage",
            "new": "attempt ledger + checkpoint snapshots",
            "TRACE": "discarded/superseded/analyzer cost incomplete; no fabricated zeros",
        },
    ]
    ev.write_csv(
        out / "protocol_differences.csv", differences, ["field", "new", "TRACE"]
    )
    mean_latency = (
        sum(c["latency_ms"] for c in catalog) / len(catalog) / 1000 if catalog else None
    )
    mean_tokens = sum(c["tokens"] for c in catalog) / len(catalog) if catalog else None
    ev.write_json(
        out / "budget_estimate.json",
        {
            "limits": limits,
            "candidate_plus_references": 30600,
            "max_discovery_attempts_including_retry_allowance": 32100,
            "historical_eligible_configs": len(catalog),
            "historical_mean_episode_seconds": mean_latency,
            "rough_discovery_hours_at_16_slots": 30600 * mean_latency / 16 / 3600
            if mean_latency
            else None,
            "rough_discovery_agent_tokens": 30600 * mean_tokens
            if mean_tokens
            else None,
            "estimate_scope": "historical c16 full catalogs; throughput idealization, excludes optimizer/queue/retry overhead and judge tokens",
            "test": "exact dedup list frozen only after all six catalogs finish; requires separate confirmation; no TEST launched by execute",
        },
    )
    confirmation = {
        "manifest_sha256": ev.file_hash(out / "manifest.json"),
        "confirmed_by_user": False,
        "scope": "discovery",
        "fixed_knobs_and_limits_confirmed": False,
        "services_reserved": False,
        "model_revision": None,
        "memory_snapshot": None,
        "note": "Fill only after explicit user confirmation and runtime/memory attestation",
    }
    ev.write_json(out / "confirmation.template.json", confirmation)
    cmd = f"{shlex.quote(str(ROOT / '.venv-search-baselines/bin/python'))} -m trimem.experiments.alfworld_e2_campaign"
    opts = f" --out {shlex.quote(str(out))}"
    (out / "commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n"
        "source /local-scratch/localhome/hza214/miniconda3/etc/profile.d/conda.sh\nconda activate tridb\n"
        f"{cmd} dry-run{opts}\n"
        f"# After recorded user confirmation: {cmd} execute{opts} --confirmation {out}/confirmation.json --phase first\n"
        f"# After first pair completes: {cmd} execute{opts} --confirmation {out}/confirmation.json --phase remaining\n"
        f"# Resume same identities: {cmd} resume{opts} --confirmation {out}/confirmation.json --phase first\n"
        f"# Refresh evidence: {cmd} export{opts}\n"
        f"# TEST requires separately approved frozen test_plan.json: {cmd} test{opts} --confirmation {out}/test_confirmation.json\n"
    )
    export(out)
    return dry_run(out)


def dry_run(out):
    m = ev.read_json(out / "manifest.json")
    if source_hashes() != m["code_sha256"]:
        raise ValueError(
            "implementation changed since prepare; refresh into a new directory"
        )
    for path, sha in m.get("reference_baseline", {}).get("files_sha256", {}).items():
        if ev.file_hash(path) != sha:
            raise ValueError(f"historical NoMemory reference changed: {path}")
    if ev.file_hash(m["pool_file"]) != m["pool_sha256"] or any(
        ev.file_hash(p) != h for p, h in m["test_files"].items()
    ):
        raise ValueError("TRAIN/TEST source changed")
    if ev.file_hash(out / "trace_catalog.json") != m["trace_catalog_sha256"]:
        raise ValueError("frozen TRACE catalog changed")
    for name, sha in m["trace_inputs_sha256"].items():
        if ev.file_hash(out / "provenance" / name) != sha:
            raise ValueError(f"frozen TRACE input changed: {name}")
    d = driver()
    if {p: importlib.metadata.version(p) for p in m["packages"]} != m["packages"]:
        raise ValueError("resolved search dependencies changed since prepare")
    ds = ev.dataset_of(m)
    for run in m["runs"]:
        a = d.build_parser().parse_args(run["argv"])
        if a.dataset != ds:
            raise ValueError("run dataset differs from manifest")
        if d.staging_block(a) != run["staging"]:
            raise ValueError("staging differs from manifest")
        # Same round-0 rule as prepare and the runtime plan_round (WebShop: semfilter-v2b).
        settings, initialization = frozen_initialization(d, ds, a)
        if settings != run["settings"]:
            raise ValueError("backend settings/space differ from frozen manifest")
        if [p["config"] for p in initialization] != [
            p["config"] for p in run["initialization"]
        ]:
            raise ValueError("paired initialization differs from frozen manifest")
    report = {
        "status": "prepared",
        "manifest_sha256": ev.file_hash(out / "manifest.json"),
        "runs": len(m["runs"]),
        "candidate_evaluations": m["limits"]["candidate_evaluations_total"],
        "episodes_launched": 0,
    }
    ev.write_json(out / "dry_run.json", report)
    return report


def authorize(out, receipt, scope):
    dry_run(out)
    r = ev.read_json(receipt)
    if (
        r.get("manifest_sha256") != ev.file_hash(out / "manifest.json")
        or r.get("scope") != scope
        or not all(
            r.get(k) is True
            for k in (
                "confirmed_by_user",
                "fixed_knobs_and_limits_confirmed",
                "services_reserved",
            )
        )
    ):
        raise ValueError("explicit user confirmation bound to the manifest is required")
    if not r.get("model_revision") or not r.get("memory_snapshot"):
        raise ValueError(
            "exact model revision and memory snapshot attestation required"
        )
    return r


def historical_checkpoints(out, manifest):
    """Reconstruct only contemporaneously available full configurations.

    Surviving evaluated rows are a lower bound on historical task expenditure:
    discarded attempts and superseded trials are not silently assigned zero.
    Keep the lower bound out of the main comparable task axis.
    """
    ds = ev.dataset_of(manifest)
    start = ev.spec(ds)["trace_from_round"]
    state = ev.read_json(out / f"provenance/{ds}_state.json")
    rows = ev.read_json(out / "provenance/trace_train_rows.json")
    catalog = ev.read_json(out / "trace_catalog.json")
    checkpoints = []
    d = driver()
    for i, rr in enumerate(state["rounds"]):
        if rr["round"] < start:
            continue
        cutoff = (
            state["rounds"][i + 1]["proposed_at"]
            if i + 1 < len(state["rounds"])
            else max((r["updated_at"] for r in rows), default="")
        )
        prefix = [r for r in rows if r["updated_at"] <= cutoff]
        complete = []
        for c in catalog:
            observations = [r for r in prefix if d.row_key(r) == c["key"]]
            if c["source_round"] <= rr["round"] and len(observations) == c["expected"]:
                complete.append(c)
        cp = {
            "round": rr["round"] - start,
            "catalog": complete,
            "candidate_evaluations": None,
            "search_seconds": None,
            "search_tokens": None,
            "coverage": "historical axes incomplete: surviving evaluated tasks are a lower bound; warm start retained",
        }
        row = ev.checkpoint_row("TRACE-historical", cp, manifest["normalization"])
        row.update(
            source_round=rr["round"],
            historical_recorded_candidate_evaluations=len(prefix),
        )
        checkpoints.append(row)
    return checkpoints


def export(out):
    dry_run(out)
    m = ev.read_json(out / "manifest.json")
    campaign = Path(m["campaign_dir"])
    archives = {
        "TRACE-historical": ev.policies(
            ev.read_json(out / "trace_catalog.json"), m["slo_grid"]
        )
    }
    checkpoints = historical_checkpoints(out, m)
    ledger = []
    pending = []
    for run in m["runs"]:
        base = Path(run["directory"])
        catalog_path = base / "e2_catalog.json"
        if catalog_path.exists() and (base / "profile_complete.json").exists():
            archives[run["id"]] = ev.policies(ev.read_json(catalog_path), m["slo_grid"])
        else:
            pending.append(run["id"])
        for p in sorted((base / "e2_checkpoints").glob("*.json")):
            c = ev.read_json(p)
            checkpoints.append(ev.checkpoint_row(run["id"], c, m["normalization"]))
        lp = base / "e2_resources.json"
        if lp.exists():
            ledger.append({"archive": run["id"], **ev.read_json(lp)})
    policy_file = {
        "manifest_sha256": ev.file_hash(out / "manifest.json"),
        "selection_split": "train",
        "grid": m["slo_grid"],
        "archives": archives,
        "pending_archives": pending,
    }
    ev.write_json(out / "oneconfig_policies.json", policy_file)
    test_identity = (
        ev.read_json(out / "test_plan.json")["identity"]
        if (out / "test_plan.json").exists()
        else m["identity"]
    )
    pairs = ev.test_plan(archives, m["test_ids"], test_identity)
    ev.write_json(
        out / "test_plan.preview.json",
        {
            "policies_sha256": ev.digest(policy_file),
            "pairs": pairs,
            "episodes": len(pairs),
            "anchors": 200,
            "status": "preview until all catalogs and serving/memory identity frozen",
        },
    )
    # Historical TEST reuse is conservatively refused: exact model revision and memory snapshot are absent.
    coverage = [
        {
            "pair_id": p["pair_id"],
            "key": p["key"],
            "task": p["task"],
            "status": "pending",
            "reason": "no full-identity matched measurement; historical TEST not assumed reusable",
        }
        for p in pairs
    ]
    ev.write_csv(
        out / "test_coverage.csv",
        coverage,
        ["pair_id", "key", "task", "status", "reason"],
    )
    measurements = (
        ev.read_json(campaign / "test_measurements.json")
        if (campaign / "test_measurements.json").exists()
        else []
    )
    measured_ids = ev.validate_test_rows(pairs, measurements)
    for row in coverage:
        if row["pair_id"] in measured_ids:
            row.update(status="measured", reason="full frozen identity matched")
    ev.write_csv(
        out / "test_coverage.csv",
        coverage,
        ["pair_id", "key", "task", "status", "reason"],
    )
    results = ev.summarize_test(archives, pairs, measurements)
    ev.write_csv(
        out / "test_results.csv",
        results,
        [
            "archive",
            "policy_id",
            "axis",
            "index",
            "key",
            "expected",
            "completed",
            "coverage",
            "measurement_status",
            "quality",
            "latency_ms",
            "tokens",
            "latency_slo_violation",
            "token_slo_violation",
        ],
    )
    # Legacy wall/token chronology is not reconstructable from final means. Keep the gap explicit.
    ledger.append(
        {
            "archive": "TRACE-historical",
            "coverage": "missing superseded attempts and complete analyzer time/tokens",
            "candidate_evaluations": None,
            "search_seconds": None,
            "search_tokens": None,
        }
    )
    ev.write_csv(
        out / "search_checkpoints.csv",
        checkpoints,
        [
            "archive",
            "round",
            "nhv",
            "candidate_evaluations",
            "historical_recorded_candidate_evaluations",
            "source_round",
            "search_seconds",
            "search_tokens",
            "coverage",
            "full_configs",
            "checkpoint_sha256",
        ],
    )
    ev.write_csv(
        out / "resource_ledger.csv",
        ledger,
        [
            "archive",
            "candidate_evaluations",
            "attempts",
            "retry_attempts",
            "cache_hits",
            "search_seconds",
            "optimizer_seconds",
            "profiling_tokens",
            "noise_repeat_tokens",
            "shared_nomemory_tokens",
            "attempts_without_completed_episode",
            "noise_token_coverage",
            "reference_token_coverage",
            "optimizer_llm_tokens",
            "search_tokens",
            "coverage",
        ],
    )
    ev.write_json(
        out / "plot_data.json", {"checkpoints": checkpoints, "test_results": results}
    )
    figures = []
    if checkpoints or any(r["measurement_status"] == "complete" for r in results):
        # Plot with the existing matplotlib environment; leave search dependencies untouched.
        completed = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "trimem.experiments.alfworld_e2_evidence",
                "--data",
                str(out / "plot_data.json"),
                "--out",
                str(out / "figures"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        figures = json.loads(completed.stdout)
    (out / "report.md").write_text(
        f"# {ev.spec(ev.dataset_of(m))['name']} E2 prepare\n\n## Material Passport\n"
        "- Origin: academic-research-suite / experiment-agent; nc-manuscript-craft.\n"
        f"- Stage: prepared; completed new discovery archives {6 - len(pending)}/6; measured TEST pairs {len(measurements)}.\n"
        "- Question: equal-budget MO-TPE vs GP+LogNEHVI search, with historical TRACE reference.\n\n"
        f"Six isolated runs, 30,000 candidate evaluations; {len(pending)} archives pending. "
        f"TRACE eligible catalog: {len(ev.read_json(out / 'trace_catalog.json'))} full configurations. "
        f"Current deduplicated TEST preview: {len(pairs)} episodes plus 200 NoMemory anchors.\n\n"
        "TRACE retains legacy/Sobol/staged/warm-start identity, including historical P1 space expansion. "
        "No historical search-time/token curves are fabricated; complete chronology and superseded costs are missing. "
        "Only complete measured rows support comparisons; pending coverage is never promoted to a full result.\n\n"
        "Fixed SLO grid: latency 10–500 seconds; tokens 10k–300k, 21 indices each. "
        "Fixed opposite caps: 500 seconds / 300k tokens. These are prespecified proposals awaiting confirmation.\n\n"
        "Actual runner is .venv-agentgym/bin/python. One temperature-zero trial; evaluator has no explicit agent seed. "
        "Exact serving model revision and memory/index snapshot require attestation before launch. "
        "Queue occupancy is recorded in environment.json. Shared services will not be restarted.\n\n"
        f"Generated measured figures: {len(figures)}. Empty files contain schemas/pending coverage, not synthetic results.\n"
    )
    return {
        "pending_archives": pending,
        "test_preview_episodes": len(pairs),
        "figures": figures,
    }


@contextlib.contextmanager
def queue_lock():
    with (ROOT / "results/pool_evaluation_queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "evaluation queue occupied; no experiment started"
            ) from exc
        yield lock


def reserve_attempts(ledger, pairs, limits):
    """Durable admission before launch; unknown/crashed attempts stay charged."""
    counts = dict(ledger.get("pair_attempts", {}))
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1
        if counts[pair] > limits["max_attempts_per_pair"]:
            raise ValueError("per-pair attempt cap exhausted")
    retries = sum(n - 1 for n in counts.values())
    if retries > limits["retry_attempts_per_run"]:
        raise ValueError("retry attempt cap exhausted")
    return {
        **ledger,
        "pair_attempts": counts,
        "attempts": sum(counts.values()),
        "retry_attempts": retries,
    }


def install_admission_guards(d, manifest, attempt_path):
    """Both transports use the same durable budget and reuse-only guard."""
    original_attempt, original_launch = d._attempt, d._launch

    def admit(ds, arms):
        reference = manifest.get("reference_baseline")
        if reference and any(
            arm.name == "NoMemory"
            or arm.out.resolve() == Path(reference["directory"]).resolve()
            for arm in arms
        ):
            raise RuntimeError(
                "NoMemory is reuse-only; missing reference results must not trigger evaluation"
            )
        pairs = []
        for arm in arms:
            completed = arm.completed(ds)
            pairs.extend(str(arm.out) + ":" + q for q in arm.ids if q not in completed)
        admitted = reserve_attempts(
            ev.read_json(attempt_path), pairs, manifest["limits"]
        )
        ev.write_json(attempt_path, admitted)

    def attempt(ds, arms, per_half, concurrency):
        admit(ds, arms)
        return original_attempt(ds, arms, per_half, concurrency)

    def launch(ds, arm, halves):
        admit(ds, [arm])
        return original_launch(ds, arm, halves)

    d._attempt, d._launch = attempt, launch


def record_async_canary(base, state):
    """Reconcile evidence from the committed state, including after a resume."""
    if not state.get("staging", {}).get("async"):
        return
    configurations = {}
    for rnd in state.get("rounds", []):
        if rnd["round"] != 0:
            continue
        for j, config in enumerate(rnd["configs"]):
            # A fresh async run commits round 0 before its staging blocks exist: no evidence yet.
            stage = config.get("staging") or {}
            decision = next(
                (x for x in stage.get("decisions", []) if x.get("stage") == 50), None
            )
            if stage.get("stage_reached", 0) >= 50 and decision is not None:
                configurations[f"r0c{j}"] = {
                    "at": decision.get("decided_at"),
                    "completed": 50,
                    "decision": decision,
                }
    ev.write_json(
        base / "canary.json",
        {
            "status": "verified" if configurations else "pending",
            "mode": "async",
            "scope": "individual round-0 stage-50 prefixes",
            "candidate_budget": "included",
            "configurations": configurations,
        },
    )


def collect_tokens(base, rows, nm_dir=None):
    roots = [base]
    if nm_dir is not None:
        roots.append(nm_dir)
    calls = []
    for root in roots:
        for path in Path(root).rglob("llm_calls.jsonl"):
            if any(s in path.relative_to(root).parts for s in ("test", "_orphans")):
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    try:
                        calls.append(json.loads(line))
                    except json.JSONDecodeError:
                        return None, "incomplete call log"
    if not calls:
        return None, "no call logs"
    if any(
        c.get("prompt_tokens") is None or c.get("completion_tokens") is None
        for c in calls
    ):
        return None, "missing usage in call logs"
    # Agent usage logs include failed calls when usage is available. Semantic usage
    # lives in evaluated retrieval receipts; missing receipts remain explicit gaps.
    if any(
        r["protocol"].get("semfilter") and r.get("judge_tokens") is None for r in rows
    ):
        return None, "missing semantic judge usage"
    return sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls) + sum(
        r.get("judge_tokens") or 0 for r in rows
    ), "complete for logged agent calls and evaluated judge receipts"


def token_scopes(base, rows, noise_rows, nm, reference_executed):
    candidate, coverage = collect_tokens(base / "configs", rows)
    noise, noise_coverage = collect_tokens(base / "noise_repeat_VGR", noise_rows)
    reference, reference_coverage = (
        collect_tokens(nm, [])
        if reference_executed
        else (0, "not executed by this run")
    )
    return {
        "profiling_tokens": candidate,
        "search_tokens": candidate,
        "coverage": coverage,
        "noise_repeat_tokens": noise,
        "noise_token_coverage": noise_coverage,
        "shared_nomemory_tokens": reference,
        "reference_token_coverage": reference_coverage,
    }


def install_runtime(m):
    d = driver()
    if m.get("reference_baseline"):
        install_reference(d, m["reference_baseline"])
    d.OUT = Path(m["campaign_dir"]) / "runs"
    ds = ev.dataset_of(m)
    d.CONCURRENCY[ds] = d.TEST_CONCURRENCY[ds] = 16
    d.drv.ATTEMPTS = 1
    d.rebalance_due = lambda *args, **kwargs: False
    d.ASYNC_REBALANCE = False

    # The shared runner normally restarts servers. E2 never restarts shared services.
    def health_only(ds):
        for port in d.drv.DS[ds]["env"]:
            stats = d.drv.get(f"http://127.0.0.1:{port}/managed_stats", timeout=10)
            if stats.get("live_envs", 0) or stats.get("jvms", 0):
                raise RuntimeError(f"env {port} did not drain; stop without restart")

    d.drv.env_check_and_restart = health_only

    def stopped(signum, frame):
        raise SystemExit(
            f"worker interrupted by signal {signum}; admitted attempts remain charged"
        )

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    return d


def worker(out, run_id, confirmation):
    receipt = authorize(out, confirmation, "discovery")
    queue_fd = int(os.environ.get("E2_QUEUE_FD", "-1"))
    if (
        queue_fd < 0
        or os.fstat(queue_fd).st_ino
        != (ROOT / "results/pool_evaluation_queue.lock").stat().st_ino
    ):
        raise ValueError("internal worker requires the supervisor-held queue lock")
    fcntl.flock(queue_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    m = ev.read_json(out / "manifest.json")
    ds = ev.dataset_of(m)
    run = next(r for r in m["runs"] if r["id"] == run_id)
    env = isolated_env_from_manifest(m, run)
    os.environ.clear()
    os.environ.update(env)
    d = install_runtime(m)
    a = d.build_parser().parse_args(run["argv"])
    base = Path(run["directory"])
    base.mkdir(parents=True, exist_ok=True)
    if (base / "profile_complete.json").exists():
        return {"status": "already complete"}
    lease = (base / "worker.lock").open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if ev.read_json(Path(m["campaign_dir"]) / "attestation.json") != receipt:
        raise ValueError("campaign attestation changed")

    def deadline(signum, frame):
        raise TimeoutError("worker-owned discovery deadline reached")

    signal.signal(signal.SIGALRM, deadline)
    remaining = float(os.environ.get("E2_REMAINING_WALL_SECONDS", "0"))
    if remaining <= 0:
        raise ValueError("supervisor must provide remaining cumulative wall budget")
    signal.alarm(math.ceil(remaining))
    expected = {
        "manifest_sha256": ev.file_hash(out / "manifest.json"),
        "attestation": receipt,
    }
    freeze(base / "e2_identity.json", expected)
    old = (
        ev.read_json(base / "e2_resources.json")
        if (base / "e2_resources.json").exists()
        else {}
    )
    started = time.monotonic()
    prior_seconds = old.get("search_seconds", 0)
    ledger = {**old, "optimizer_llm_tokens": 0, "cache_hits": 0}
    attempt_path = base / "e2_attempts.json"
    if not attempt_path.exists():
        ev.write_json(
            attempt_path, {"pair_attempts": {}, "attempts": 0, "retry_attempts": 0}
        )
    install_admission_guards(d, m, attempt_path)
    original_plan = d.plan_round

    def propose(*args, **kwargs):
        start = time.monotonic()
        try:
            picks = original_plan(*args, **kwargs)
            keys = [d.space.key(p["config"]) for p in picks]
            planned = d.samples_of(args[2])
            if args[3] == 0 and [p["config"] for p in picks] != [
                p["config"] for p in run["initialization"]
            ]:
                raise ValueError("initialization differs from paired frozen manifest")
            if len(set(keys)) != len(keys) or set(keys) & set(planned):
                raise ValueError("duplicate proposal rejected before episode admission")
            return picks
        finally:
            ledger["optimizer_seconds"] = (
                ledger.get("optimizer_seconds", 0) + time.monotonic() - start
            )

    d.plan_round = propose
    original_round = d.run_staged_round

    def checkpoint(ds, pool, state, state_p, r, args):
        original_round(ds, pool, state, state_p, r, args)
        rows = read_rows(state["experiment"])
        catalog, audit = ev.audited_catalog(state, rows, base, pool)
        resources(state, rows)
        ev.write_json(base / "e2_catalog.json", catalog)
        ev.write_json(base / "e2_audit.json", audit)
        snap = {
            "round": r,
            "catalog": catalog,
            "candidate_evaluations": d.episodes_used(state),
            "search_seconds": ledger["search_seconds"],
            "search_tokens": ledger["search_tokens"],
            "coverage": ledger["coverage"],
        }
        path = base / "e2_checkpoints" / f"{r:03}.json"
        # A replayed completed round must retain its original timing/observations.
        if not path.exists():
            ev.write_json(path, snap)

    d.run_staged_round = checkpoint
    original_run = d._run_or_stop

    def canary(ds, arms, what):
        original_run(ds, arms, what)
        if what == "round 0 stage 50" and not (base / "canary.json").exists():
            if any(len(arm.completed(ds)) != len(arm.ids) for arm in arms):
                raise ValueError("canary profiler/storage completion mismatch")
            ev.write_json(
                base / "canary.json",
                {
                    "status": "verified",
                    "at": now(),
                    "scope": what,
                    "candidate_budget": "included",
                    "arms": {x.name: len(x.ids) for x in arms},
                },
            )

    d._run_or_stop = canary
    original_save = d.save

    def save(path, value):
        original_save(path, value)
        if Path(path) == base / "state.json":
            record_async_canary(base, ev.read_json(path))

    d.save = save

    def resources(state, rows):
        admitted = ev.read_json(attempt_path)
        nm = d.nomemory_dir(ds)
        noise_rows = read_rows(state["experiment"] + ":noise-repeat")
        # Shared NoMemory is billed only to the run that actually admitted it.
        reference_attempts = sum(
            n
            for p, n in admitted["pair_attempts"].items()
            if p.rsplit(":", 1)[0] == str(nm)
        )
        ledger.update(
            token_scopes(base, rows, noise_rows, nm, bool(reference_attempts))
        )
        ledger.update({k: admitted[k] for k in ("attempts", "retry_attempts")})
        ledger.update(
            candidate_evaluations=d.episodes_used(state),
            search_seconds=prior_seconds + time.monotonic() - started,
            attempts_semantics="reserved before launch, includes unfinished/crashed attempts",
            successful_candidate_episodes=len(rows),
            reference_attempts=reference_attempts,
            noise_repeat_episodes=len(noise_rows),
            updated_at=now(),
        )
        completed_files = {}
        for pair in admitted["pair_attempts"]:
            directory = pair.rsplit(":", 1)[0]
            if directory not in completed_files:
                completed_files[directory] = d.drv.item_files(Path(directory), ds)
        successful_pairs = sum(
            task in completed_files[directory]
            for directory, task in (
                pair.rsplit(":", 1) for pair in admitted["pair_attempts"]
            )
        )
        ledger["attempts_without_completed_episode"] = (
            admitted["attempts"] - successful_pairs
        )
        ledger["cache_hits"] = (
            len(
                set(state.get("nomemory", {}).get("sample", {}).get("ids", []))
                & set(d.drv.item_files(nm, ds))
            )
            if not reference_attempts
            else 0
        )
        if ledger["attempts_without_completed_episode"]:
            ledger.update(
                search_tokens=None,
                coverage="unfinished/retry judge token coverage incomplete; logged token subtotal retained",
            )
        ev.write_json(base / "e2_resources.json", ledger)

    try:
        # Reuse the frozen historical reference before either search clock starts.
        pool = d.load_pool(ds)
        sample = d.write_sample(
            ds,
            pool,
            base,
            "NoMemory",
            d.sample_label("NoMemory", None, a.sampling),
            d.nomemory_sample_experiment(ds, a.sampling),
            size=100,
        )
        anchor = d.Arm("NoMemory", d.nomemory_dir(ds), sample["ids"], None)
        if len(anchor.completed(ds)) < len(anchor.ids):
            reference_start = time.monotonic()
            d._run_or_stop(ds, [anchor], "shared NoMemory warmup")
            ledger["reference_seconds"] = (
                ledger.get("reference_seconds", 0) + time.monotonic() - reference_start
            )
        started = time.monotonic()
        d.profile(ds, a)
    finally:
        signal.alarm(0)
        if (base / "state.json").exists():
            state = ev.read_json(base / "state.json")
            resources(state, read_rows(state["experiment"]))
    return {"status": "complete"}


def isolated_env_from_manifest(m, run=None):
    """Discovery worker env (``run``), else an evaluation process's env (holdout / OneConfig TEST)."""
    env = isolated_env(ev.dataset_of(m), evaluation=run is None)
    mode = m.get("staging_mode", "sync")
    if mode not in ("sync", "async"):
        raise ValueError(f"unknown E2 staging mode: {mode}")
    if mode == "async":
        env["POOL_ASYNC_STAGING"] = "1"
    if run:
        env["POOL_RUN_TAG"] = run["tag"]
    return env


def execute(out, confirmation, phase, resume=False):
    receipt = authorize(out, confirmation, "discovery")
    m = ev.read_json(out / "manifest.json")
    ds = ev.dataset_of(m)
    campaign = Path(m["campaign_dir"])
    campaign.mkdir(parents=True, exist_ok=True)
    first_seed = m["runs"][0]["seed"]
    first = [r for r in m["runs"] if r["seed"] == first_seed]
    runs = first if phase == "first" else [r for r in m["runs"] if r not in first]
    if phase == "remaining" and any(
        not (Path(r["directory"]) / "profile_complete.json").exists() for r in first
    ):
        raise ValueError("complete the first paired seed before remaining seeds")
    with queue_lock() as held_queue:
        freeze(campaign / "attestation.json", receipt)
        env = snapshot_services(ds)
        require_idle(env)
        for run in runs:
            base = Path(run["directory"])
            base.mkdir(parents=True, exist_ok=True)
            if (base / "profile_complete.json").exists():
                continue
            if (base / "state.json").exists() and not resume:
                raise ValueError(f"{run['id']} already started; use resume")
            prior = (
                ev.read_json(base / "supervisor.json")
                if (base / "supervisor.json").exists()
                else {}
            )
            remaining = m["limits"]["discovery_wall_seconds_per_run"] - prior.get(
                "active_seconds", 0
            )
            if remaining <= 0:
                raise ValueError("cumulative wall-time cap exhausted")
            env = snapshot_services(ds)
            require_idle(env)
            if ev.read_json(confirmation) != receipt:
                raise ValueError("confirmation changed between paired runs")
            ev.write_json(base / f"launch_environment_{time.time_ns()}.json", env)
            command = [
                m["search_python"],
                "-m",
                "trimem.experiments.alfworld_e2_campaign",
                "_worker",
                "--out",
                str(out),
                "--run-id",
                run["id"],
                "--confirmation",
                str(confirmation.resolve()),
            ]
            with (base / "supervisor.log").open("a") as log:
                p = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={
                        **isolated_env_from_manifest(m, run),
                        "E2_QUEUE_FD": str(held_queue.fileno()),
                        "E2_REMAINING_WALL_SECONDS": str(remaining),
                    },
                    pass_fds=(held_queue.fileno(),),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                start = time.monotonic()
                status = {
                    **prior,
                    "pid": p.pid,
                    "supervisor_pid": os.getpid(),
                    "command": command,
                    "status": "running",
                    "started_at": now(),
                }
                try:
                    while p.poll() is None:
                        elapsed = time.monotonic() - start
                        status.update(
                            heartbeat=now(),
                            active_seconds=prior.get("active_seconds", 0) + elapsed,
                        )
                        ev.write_json(base / "supervisor.json", status)
                        if elapsed >= remaining:
                            raise TimeoutError(
                                "frozen cumulative wall-time limit reached"
                            )
                        time.sleep(5)
                    if p.returncode:
                        raise RuntimeError(
                            f"{run['id']} failed with exit {p.returncode}; no automatic retry"
                        )
                    status["status"] = "complete"
                except BaseException as exc:
                    status.update(status="stopped", failure=repr(exc))
                    raise
                finally:
                    if p.poll() is None:
                        p.terminate()  # worker SIGTERM unwinds runner finally blocks and reaps its children
                        try:
                            p.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            p.kill()
                            p.wait()
                            status["failure"] = (
                                "worker failed to stop; inspect child processes before resuming"
                            )
                    status.update(
                        exit_code=p.returncode,
                        finished_at=now(),
                        active_seconds=prior.get("active_seconds", 0)
                        + time.monotonic()
                        - start,
                    )
                    ev.write_json(base / "supervisor.json", status)
    return export(out)


def freeze_test(out, receipt_path):
    """Build exact deduplicated TEST work after TRAIN is final; still no episodes."""
    receipt = authorize(out, receipt_path, "discovery")
    export(out)
    m = ev.read_json(out / "manifest.json")
    p = ev.read_json(out / "oneconfig_policies.json")
    if p["pending_archives"]:
        raise ValueError("all TRAIN catalogs must finish before final TEST plan")
    identity = {
        **m["identity"],
        "model_revision": receipt["model_revision"],
        "memory_snapshot": receipt["memory_snapshot"],
    }
    pairs = ev.test_plan(p["archives"], m["test_ids"], identity)
    plan = {
        "policies_sha256": ev.digest(p),
        "manifest_sha256": ev.file_hash(out / "manifest.json"),
        "identity": identity,
        "pairs": pairs,
        "test_ids": m["test_ids"],
        "episode_cap": len(pairs) + len(m["test_ids"]),
        "retry_cap": 0,
        "wall_seconds": max(3600, (len(pairs) + len(m["test_ids"])) * 2400),
        "anchors": len(m["test_ids"]),
    }
    freeze(out / "test_plan.json", plan)
    ev.write_json(
        out / "test_confirmation.template.json",
        {
            "manifest_sha256": ev.file_hash(out / "manifest.json"),
            "test_plan_sha256": ev.file_hash(out / "test_plan.json"),
            "scope": "test",
            "confirmed_by_user": False,
            "fixed_knobs_and_limits_confirmed": False,
            "services_reserved": False,
            "model_revision": receipt["model_revision"],
            "memory_snapshot": receipt["memory_snapshot"],
        },
    )
    return {
        "test_plan_sha256": ev.file_hash(out / "test_plan.json"),
        "episode_cap": plan["episode_cap"],
    }


def test_worker(out, confirmation):
    receipt = authorize(out, confirmation, "test")
    m = ev.read_json(out / "manifest.json")
    plan = ev.read_json(out / "test_plan.json")
    if receipt.get("test_plan_sha256") != ev.file_hash(out / "test_plan.json"):
        raise ValueError("confirm the exact frozen deduplicated TEST plan")
    policies = ev.read_json(out / "oneconfig_policies.json")
    if ev.digest(policies) != plan["policies_sha256"]:
        raise ValueError("TRAIN policies changed after TEST plan freeze")
    if any(
        receipt[k] != plan["identity"][k] for k in ("model_revision", "memory_snapshot")
    ):
        raise ValueError("TEST attestation differs from frozen identity")
    ds = ev.dataset_of(m)
    with queue_lock():
        require_idle(snapshot_services(ds))
        env = isolated_env_from_manifest(m)
        os.environ.clear()
        os.environ.update(env)
        d = install_runtime(m)
        base = Path(m["campaign_dir"]) / "test"
        base.mkdir(parents=True, exist_ok=True)

        # A single SIGALRM bounds the whole TEST process including profiler waits.
        def timeout(signum, frame):
            raise TimeoutError("TEST wall-time limit reached")

        signal.signal(signal.SIGALRM, timeout)
        status_path = base / "supervisor.json"
        prior = ev.read_json(status_path) if status_path.exists() else {}
        prior_seconds = prior.get("active_seconds", 0)
        remaining = plan["wall_seconds"] - prior_seconds
        if remaining <= 0:
            raise ValueError("cumulative TEST wall-time limit exhausted")
        signal.alarm(math.ceil(remaining))
        attempt_path = base / "attempts.json"
        if not attempt_path.exists():
            ev.write_json(attempt_path, {"pair_attempts": {}})
        original_attempt = d._attempt

        def attempt(ds, arms, per_half, concurrency):
            pairs = [
                str(a.out) + ":" + q
                for a in arms
                for q in a.ids
                if q not in a.completed(ds)
            ]
            charged = reserve_attempts(
                ev.read_json(attempt_path),
                pairs,
                {"max_attempts_per_pair": 1, "retry_attempts_per_run": 0},
            )
            if charged["attempts"] > plan["episode_cap"]:
                raise ValueError("TEST episode cap exhausted")
            ev.write_json(attempt_path, charged)
            return original_attempt(ds, arms, per_half, concurrency)

        d._attempt = attempt
        nm = base / "NoMemory"
        grouped = {}
        for p in plan["pairs"]:
            grouped.setdefault(p["key"], []).append(p)
        started = time.monotonic()
        stop_heartbeat = threading.Event()
        status = {
            "pid": os.getpid(),
            "status": "running",
            "started_at": now(),
            "test_plan_sha256": ev.file_hash(out / "test_plan.json"),
        }

        def heartbeat():
            while not stop_heartbeat.is_set():
                ev.write_json(
                    status_path,
                    {
                        **status,
                        "heartbeat": now(),
                        "active_seconds": prior_seconds + time.monotonic() - started,
                    },
                )
                stop_heartbeat.wait(5)

        beat = threading.Thread(target=heartbeat, daemon=True)
        beat.start()
        try:
            d._run_or_stop(
                ds,
                [d.Arm("TEST NoMemory", nm, plan["test_ids"], None)],
                "TEST anchor",
            )
            for pairs in grouped.values():
                cfg = pairs[0]["config"]
                exp = f"{ds}:e2:test:{ev.file_hash(out / 'manifest.json')[:16]}"
                session = {
                    **d.space.runner_args(cfg),
                    "runner_extra": d.space.runner_cli(cfg),
                    "split": "test",
                    "experiment": exp,
                    "baseline_dir": nm,
                    "panel": None,
                    "reuse": False,
                    "reuse_from": [],
                    "reuse_from_dirs": [],
                    "protocol_extra": {
                        "panel_hash": ev.digest(plan["test_ids"]),
                        "space_config": cfg,
                        "space_version": d.space.VERSION,
                        "e2_identity": plan["identity"],
                        "e2_policies_sha256": plan["policies_sha256"],
                    },
                }
                arm = d.Arm(
                    "OneConfig " + d.cfg_name(cfg),
                    base / "pairs" / d.cfg_name(cfg),
                    [p["task"] for p in pairs],
                    session,
                )
                d._run_or_stop(ds, [arm], "TEST OneConfig")
                rows = read_rows(exp, "test")
                by = {(d.row_key(r), r["query_id"]): r for r in rows}
                measured = []
                for p in plan["pairs"]:
                    r = by.get((p["key"], p["task"]))
                    if r:
                        if r["protocol"].get("e2_identity") != plan["identity"]:
                            raise ValueError("stored TEST identity mismatch")
                        measured.append(
                            {
                                **p,
                                "split": "test",
                                "quality": r["quality"],
                                "tokens": r["total_tokens"],
                                "latency_ms": r["e2e_ms"],
                                "status": "measured",
                            }
                        )
                ev.validate_test_rows(plan["pairs"], measured)
                ev.write_json(
                    Path(m["campaign_dir"]) / "test_measurements.json", measured
                )
                tokens, coverage = collect_tokens(base, rows)
                ev.write_json(
                    base / "resources.json",
                    {
                        "attempts": ev.read_json(attempt_path).get("attempts", 0),
                        "completed_memory_episodes": len(measured),
                        "search_tokens": tokens,
                        "coverage": coverage,
                        "optimizer_llm_tokens": 0,
                        "updated_at": now(),
                    },
                )
            status["status"] = "complete"
        except BaseException as exc:
            status.update(status="stopped", failure=repr(exc))
            raise
        finally:
            signal.alarm(0)
            stop_heartbeat.set()
            beat.join()
            ev.write_json(
                status_path,
                {
                    **status,
                    "finished_at": now(),
                    "active_seconds": prior_seconds + time.monotonic() - started,
                },
            )
    return export(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "dry-run",
            "execute",
            "resume",
            "export",
            "freeze-test",
            "test",
            "_worker",
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"prepare directory (default {DEFAULT_OUT}; required for --dataset other than alfworld)",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(ev.DATASETS),
        default=None,
        help="prepare only (default alfworld); later modes use the frozen manifest's dataset",
    )
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--phase", choices=("first", "remaining"), default="first")
    parser.add_argument("--run-id")
    parser.add_argument("--methods", nargs="+", choices=sorted(SHORT), default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--pause-after-round", type=int, default=None)
    args = parser.parse_args(argv)
    if args.out is None:
        if args.mode == "prepare" and args.dataset not in (None, "alfworld"):
            parser.error("--out is required for a non-ALFWorld prepare")
        args.out = DEFAULT_OUT
    args.out = args.out.resolve()
    if args.mode != "prepare" and args.dataset is not None:
        frozen = ev.dataset_of(ev.read_json(args.out / "manifest.json"))
        if frozen != args.dataset:
            parser.error(
                f"--dataset {args.dataset} != frozen manifest dataset {frozen}"
            )
    if (
        args.mode in ("execute", "resume", "_worker", "freeze-test", "test")
        and args.confirmation is None
    ):
        parser.error(
            "--confirmation receipt is required; prepare never launches episodes"
        )
    if args.mode == "prepare":
        result = prepare(
            args.out,
            args.methods,
            args.seeds,
            args.pause_after_round,
            args.dataset or "alfworld",
        )
    elif args.mode == "dry-run":
        result = dry_run(args.out)
    elif args.mode == "export":
        result = export(args.out)
    elif args.mode in ("execute", "resume"):
        result = execute(args.out, args.confirmation, args.phase, args.mode == "resume")
    elif args.mode == "_worker":
        result = worker(args.out, args.run_id, args.confirmation)
    elif args.mode == "freeze-test":
        result = freeze_test(args.out, args.confirmation)
    else:
        result = test_worker(args.out, args.confirmation)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
