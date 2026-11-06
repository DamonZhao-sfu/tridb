"""ALFWorld trace-tc discovery: task-complementary-v1, rounds 0-5 (6 rounds x 8 configurations), no TEST.

Paired with the hydra comparison's trace-hv arm: identical 8 round-0 configurations and per-configuration TRAIN
samples, the same frozen shared NoMemory reference (reuse-only), the same staging/pruning, runtime profile and
propose_batch seed offset. Executes from an isolated source workspace under the shared evaluation queue lease.

  prepare   freeze workspace hashes + manifest + attestation (no episodes)
  run       supervisor: wait for an idle evaluator + the queue lease, then run the discovery worker
  worker    internal (inherits the lease)
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/trace_tc_alfworld_20261027"
WS = OUT / "workspace"
PREP = OUT / "prepared"
HYDRA = ROOT / "results/alfworld_hydra_local_20261027"
PYTHON = ROOT / ".venv-search-baselines/bin/python"
SEED = 20261023  # the hydra comparison's paired seed (trace-hv)
TAG = "tc27tc"
ROUNDS = 5  # rounds 0..5 = 6 rounds
WALL = 6 * 86400


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bootstrap():
    for name in list(os.environ):
        if name.startswith(("POOL_", "TRIMEM_", "TAU2_")):
            del os.environ[name]
    os.environ.update(PYTHONPATH=str(WS), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    sys.path.insert(0, str(WS))
    from trimem.experiments import alfworld_e2_campaign as campaign

    require(Path(campaign.__file__).is_relative_to(WS), "Wrong execution workspace")
    sys.path.insert(0, str(ROOT / "paper/tools"))
    import unified_search24_runtime as rt

    return campaign, rt


def install(c, manifest, run):
    """Same runtime adjustments as the hydra comparison's trace-hv arm."""
    d = c.install_runtime(manifest)
    d.N_ROUND0 = 8
    d.STAGE_SIZES = (20, 40, 60, 80, 100)
    os.environ["POOL_FROM_ROUND0"] = "1"

    def no_historical_aux(state, now, build, min_round=1):
        if state.get("aux_observations", {}).get("points"):
            raise ValueError("Historical auxiliary observations are forbidden")
        state["aux_observations"] = {
            "protocol": d.bo_module().AUX_PROTOCOL,
            "from_round": min_round,
            "points": [],
            "sources": [],
            "recorded_at": now,
            "reason": "paired fresh-data comparison",
        }
        return None

    d.bo_module().aux_block = no_historical_aux
    original = d.plan_round

    def paired(ds, task_pool, state, r, nm, args, k=None):
        if r == 0:
            original(ds, task_pool, state, r, nm, args, k=k)
            configs = [p["config"] for p in run["initialization"]]
            require(
                all(d.bo_module().pool.proposable(x) for x in configs),
                "Frozen initialization is not legal under current proposal rules",
            )
            return [
                {"config": x, "channel": "paired-frozen-initialization"}
                for x in configs
            ]
        return original(ds, task_pool, state, r, nm, args, k=k)

    d.plan_round = paired
    original_batch = d.bo_module().propose_batch

    def seeded(*args, **kwargs):
        kwargs["seed"] = SEED + kwargs.get("seed", 0)
        return original_batch(*args, **kwargs)

    d.bo_module().propose_batch = seeded
    return d


def prepare(c, rt):
    d = c.driver()
    d.N_ROUND0, d.STAGE_SIZES = 8, (20, 40, 60, 80, 100)
    env = c.isolated_env("alfworld")
    env["POOL_ASYNC_STAGING"] = "1"
    os.environ.clear()
    os.environ.update(env)
    d.OUT = OUT / "campaign/runs"
    hydra = read(HYDRA / "prepared/manifest.json")
    source = next(r for r in hydra["runs"] if r["id"] == "trace-hv")
    initial = source["initialization"]
    argv = [
        "profile",
        "--dataset",
        "alfworld",
        "--run_tag",
        TAG,
        "--search_design",
        "task-complementary-v1",
        "--coverage_sampler",
        "sobol",
        "--full_panel_size",
        "100",
        "--initial_budget_episodes",
        str(100 * 8 * (ROUNDS + 1)),
        "--task_sample_namespace",
        source["argv"][source["argv"].index("--task_sample_namespace") + 1],
        "--batch",
        "8",
        "--max_configs",
        str(8 * (ROUNDS + 1)),
        "--rounds",
        str(ROUNDS),
        "--pause_after_round",
        str(ROUNDS),
        "--pruning_policy",
        "syftr-style",
        "--no_limeqo",
        "--no_reuse",
        "--channel_slots",
        "v6",
    ]
    args = d.build_parser().parse_args(argv)
    os.environ["POOL_RUN_TAG"] = TAG
    settings = d.sa.settings_for({}, args)
    require(settings["protocol"] == "task-complementary-v1", "Wrong search design")
    run = {
        "id": "trace-tc",
        "dataset": "alfworld",
        "method": "trace-tc",
        "seed": SEED,
        "tag": TAG,
        "argv": argv,
        "initialization": initial,
        "directory": str(d.run_dir("alfworld")),
        "settings": settings,
        "staging": d.staging_block(args),
    }
    from trimem.experiments.alfworld_holdout_eval import serving_identity

    reference_frozen = read(HYDRA / "reference_frozen.json")
    frozen = [ROOT / "data/external/panels/alfworld_pool.json"]
    frozen += [Path(x["sample"]["file"]) for x in initial]
    frozen += [Path(hydra["reference_baseline"]["sample"]["file"])]
    m = {
        "protocol": "trace-tc-alfworld-v1",
        "dataset": "alfworld",
        "seed": SEED,
        "campaign_dir": str(OUT / "campaign"),
        "runs": [run],
        "reference_baseline": hydra["reference_baseline"],
        "reference_frozen": reference_frozen,
        "paired_with": {
            "manifest": str(HYDRA / "prepared/manifest.json"),
            "manifest_sha256": sha(HYDRA / "prepared/manifest.json"),
            "run": "trace-hv",
            "scope": "round-0 configurations, TRAIN samples, NoMemory reference, staging, seed offset",
        },
        "fixed": hydra["fixed"],
        "memory": rt.memory_identity("alfworld"),
        "serving_identity": serving_identity(c.snapshot_services()),
        "code_sha256": c.source_hashes(),
        "controller_sha256": sha(__file__),
        "runtime_helper_sha256": sha(rt.__file__),
        "staging_mode": "async",
        "frozen_files": {str(p): sha(p) for p in frozen},
        "limits": {
            "max_attempts_per_pair": 1,
            "retry_attempts_per_run": 0,
            "candidate_evaluations_per_run": 100 * 8 * (ROUNDS + 1),
            "automatic_retries": 0,
            "discovery_wall_seconds_per_run": WALL,
        },
        "accounting": "48 proposed configurations (rounds 0-5 x 8); pruning may stop a configuration before 100 "
        "TRAIN tasks. NoMemory reused from the hydra comparison (no new episodes); the driver's 100-episode "
        "noise-repeat arm is charged separately. No TEST in this campaign.",
    }
    c.freeze(PREP / "manifest.json", m)
    attestation = {
        "manifest_sha256": sha(PREP / "manifest.json"),
        "scope": "discovery",
        "confirmed_by_user": True,
        "fixed_knobs_and_limits_confirmed": True,
        "services_reserved": True,
        "authorization": "User 2026-10-27: implement trace-tc, run ALFWorld profiling for 6 rounds in the background; "
        "pause the running ScienceWorld TEST and run this task now.",
        "model_revision": m["fixed"]["model"],
        "memory_snapshot": m["memory"],
    }
    c.freeze(OUT / "campaign/attestation.json", attestation)
    print(json.dumps({"prepared": str(PREP / "manifest.json"), "episodes_launched": 0}))


def verify(c, rt):
    m = read(PREP / "manifest.json")
    require(m["protocol"] == "trace-tc-alfworld-v1", "Wrong campaign")
    require(c.source_hashes() == m["code_sha256"], "Frozen execution code changed")
    require(sha(__file__) == m["controller_sha256"], "Frozen controller changed")
    require(sha(rt.__file__) == m["runtime_helper_sha256"], "Frozen helper changed")
    for path, digest in {**m["frozen_files"], **m["reference_frozen"]}.items():
        require(sha(path) == digest, "Frozen file changed: " + path)
    require(
        rt.memory_identity("alfworld")["id"] == m["memory"]["id"],
        "Memory identity changed",
    )
    r = read(OUT / "campaign/attestation.json")
    require(
        r["manifest_sha256"] == sha(PREP / "manifest.json") and r["confirmed_by_user"],
        "Missing/stale confirmed manifest",
    )
    return m


def worker(c, rt):
    m = verify(c, rt)
    fd = int(os.environ.get("E2_QUEUE_FD", "-1"))
    require(
        fd >= 0
        and os.fstat(fd).st_ino
        == (ROOT / "results/pool_evaluation_queue.lock").stat().st_ino,
        "Worker needs inherited supervisor lease",
    )
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = m["runs"][0]
    env = c.isolated_env_from_manifest(m, run)
    env["PYTHONPATH"] = str(WS)
    os.environ.clear()
    os.environ.update(env)
    original_runtime = c.install_runtime

    def runtime(manifest):
        c.install_runtime = original_runtime
        return install(c, manifest, run)

    c.install_runtime = runtime

    def authorize(directory, confirmation, scope):
        require(
            Path(directory) == PREP and scope == "discovery",
            "Wrong worker authorization",
        )
        verify(c, rt)
        return read(OUT / "campaign/attestation.json")

    c.authorize = authorize
    c.worker(PREP, run["id"], OUT / "campaign/attestation.json")
    base = Path(run["directory"])
    state = read(base / "state.json")
    write(
        OUT / "discovery_summary.json",
        {
            "rounds": [r["round"] for r in state["rounds"]],
            "configs_per_round": [len(r["configs"]) for r in state["rounds"]],
            "state_sha256": sha(base / "state.json"),
            "profile_complete": (base / "profile_complete.json").exists(),
        },
    )


def wait_for_idle(c, poll=120):
    """Block (no preemption) until the shared evaluator is idle and the queue lease is free."""
    while True:
        reason = None
        try:
            c.require_idle(c.snapshot_services())
        except Exception as exc:  # noqa: BLE001 - any not-idle reason means wait
            reason = str(exc)
        if reason is None:
            lock = (ROOT / "results/pool_evaluation_queue.lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock
            except BlockingIOError:
                lock.close()
                reason = "evaluation queue lease held"
        write(
            OUT / "status.json",
            {
                "status": "waiting",
                "reason": reason,
                "pid": os.getpid(),
                "at": time.time(),
            },
        )
        time.sleep(poll)


def supervise(c, rt):
    verify(c, rt)
    lease = wait_for_idle(c)

    def stop(signum, frame):
        raise SystemExit(f"Supervisor interrupted {signum}; attempts preserved")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    verify(c, rt)
    clock = OUT / "discovery.clock.json"
    if not clock.exists():
        write(clock, {"deadline_epoch": time.time() + WALL})
    remaining = read(clock)["deadline_epoch"] - time.time()
    require(remaining > 0, "Discovery wall budget exhausted")
    env = dict(
        os.environ,
        E2_QUEUE_FD=str(lease.fileno()),
        E2_REMAINING_WALL_SECONDS=str(remaining),
    )
    process = None
    try:
        with (OUT / "discovery.log").open("ab") as log:
            process = subprocess.Popen(
                [str(PYTHON), "-u", str(Path(__file__).resolve()), "worker"],
                cwd=WS,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lease.fileno(),),
            )
            started = time.monotonic()
            while process.poll() is None:
                write(
                    OUT / "status.json",
                    {
                        "status": "running",
                        "phase": "discovery",
                        "pid": os.getpid(),
                        "worker_pid": process.pid,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                time.sleep(30)
            require(
                process.returncode == 0,
                "discovery failed; inspect discovery.log, no automatic retry",
            )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    write(
        OUT / "status.json",
        {"status": "complete", "pid": os.getpid(), "at": time.time()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "verify", "run", "worker"])
    args = parser.parse_args()
    c, rt = bootstrap()
    if args.mode == "prepare":
        prepare(c, rt)
    elif args.mode == "verify":
        verify(c, rt)
        print("verified; no episodes launched")
    elif args.mode == "worker":
        worker(c, rt)
    else:
        try:
            supervise(c, rt)
        except BaseException as exc:
            write(
                OUT / "status.json",
                {
                    "status": "stopped_for_diagnosis",
                    "error": str(exc),
                    "pid": os.getpid(),
                },
            )
            raise


if __name__ == "__main__":
    main()
