"""τ²-bench pool profiling round (user decisions 2026-10-19; plan docs/trimem_pool_profiling_plan_v0.1.0.md,
τ² addendum). Queue script ``results/pool_v5_20261020/run_queue.sh`` (-v5; -v4 queue: results/tau2_pool_20261020).

  POOL       the 178 official τ² TRAIN tasks (airline 30 / retail 74 / telecom 74; panel ``tau2_train178``).
             EVERY profiled configuration runs ALL 178 tasks (pool < 200: no subsampling), so configurations are
             PAIRED on (task, trial seed).
  TRIALS     PROFILING 1 per task (user change 2026-10-20): trial 0 with the official seed-300 derivation (= the t0
             episode of a 4-trial run); TEST 1 per task as well (user 2026-10-21, test-1trial-v1; was 4), same trial 0.
  QUALITY    profiling: pass^1 per task (= the single trial's success; pass^4 is not computable from 1 trial);
             test: pass^1 per task (test-1trial-v1, user 2026-10-21; was pass^4 over 4 trials) -- the reused 4-trial
             reference arms are compared on their trial-0 episodes (their 4-trial pass^1 reported as context);
             pass^1..4, pass@4 and the mean reward are recorded too. Tokens = agent in + out + SemFilter per
             episode (the opt row's total_tokens; user simulator excluded), latency = episode wall time
             (retrieval + episode); per task = mean over its trials.
  MEMORY     all 712 train sessions (store trimem_tau2), run memory policy success+failure-v1, no per-task cap;
             retrieval isolation L1 (own task, same family, shared entity, other domain, held out).
  SPACE      ``session-space-tau2-v5`` = agent-env space (v5) with memory_form FIXED ``compact`` (user 2026-10-20; -v4 searched it);
             edges fixed similar+similar_up (user correction 2026-10-20; the similar_up-only v2 run is void).
  PROFILING  STAGED (protocol tau2-pool-staged-v2, ``staged_pruning`` v2, user 2026-10-20): round 0 = 4 plan defaults
             + 6 Sobol, then BO rounds of 5 (qLogNEHVI, kriging believer; rounds >= 1 add the LLM analyst channel
             ``analyzer_pool`` (local Qwen, temperature 0; labels agent / agent-gap; OFF from the round recorded in state.json
             ``analyst``, user 2026-10-20, live -v5 run: round 5); ``session_bo_agentenv`` switched to the
             τ² space). EVERY round (round 0 included; only NoMemory is protected) runs nested 50 -> 100 -> 178 tasks
             (one domain-stratified order, seed in state.json; NoMemory fills the same prefixes alongside), and a
             configuration is pruned at 50 / 100 when its optimistic corner (gain + 1.96 SE, tokens - 1.96 SE,
             latency - 1.96 SE) is weakly dominated by the point of a fully evaluated configuration or by the
             PESSIMISTIC corner (gain - 1.96 SE, tokens + 1.96 SE, latency + 1.96 SE) of NoMemory / any configuration
             evaluated at >= that stage, all on the SAME stage subset (paired: gains vs NoMemory on those tasks).
             Plan guard: the last live configuration of a plan without a full one is never pruned. Budget =
             30 x 178 x 1 = 5,340 episodes (the unpruned plan, round 0 included); saved episodes buy more
             configurations (cap 60). Pruned configurations stay GP observations (partial-n means, larger SE);
             selection and the router use fully evaluated configurations only. BO quality = paired pass^1 gain vs the
             train NoMemory arm (178 x 1, round 0), GP noise = its paired SE^2; costs = -ln mean tokens / latency.
             gp-fix-v1 (2026-10-20, from the round recorded in state.json ``gp``; the live -v5 run: round 4): the
             quality noise is the paired SE^2 minus the common-mode covariance shared by every point (same 178
             tasks, same NoMemory arm; ``unpaired_stats.common_mode_cov`` over the fully evaluated configurations,
             floor 25% of SE^2) and the GPs carry Gamma outputscale priors (``session_pool.GP_PROTOCOL``); every
             proposal logs the posterior spread. Doc: docs/trimem_pool_profiling_plan_v0.1.0.md addendum gp-fix-v1.
             channel-slots-v1 (user 2026-10-20, from the round recorded in state.json ``channel_slots``; the live -v5
             run: round 6): richer crossover (champions + adjacent relaxed-front pairs, 1- / 2-knob grafts, top 4 per
             direction), DLS seeded from the crossover rows while the analyst is off, and ordinary batches filled
             qfirst / xover-slot / sobol-slot / hv / hv (``session_bo_agentenv.slot_plan``); forced rounds unchanged.
  ROUTER     Opt-PerQuery = sparse BRANE (``query_router.sparse``; the profile is dense here) with the three-weight
             objective (perquery-utility3-v1); NoMemory is a router candidate (its profile = the train NoMemory
             arm). Per-query label = the train pass^1 indicator per task (mean reward stored alongside). Features =
             structured (domain) + probe + embedding of the store task (customer's first message), no LLM.
  SELECTION  SLO-1/2/3 (``slo_select_bo``, τ² SLO-3 threshold = median configuration train pass^1, 2 decimals) on
             train for Opt-OneConfig and Opt-PerQuery (weights chosen on the OOF train replay), frozen in
             selection.json before any test episode.
  TEST       official 100 test tasks x 1 trial (t0; user 2026-10-21, was x 4), 40 concurrent episodes (= the reused arms): Opt-OneConfig-SLO-1/2/3 and Opt-PerQuery
             at exactly 4 weight vectors (w_Q, w_L, w_T) = (1, 0, 0), (0.6, 0.2, 0.2), (0.6, 0.4, 0), (0.6, 0, 0.4)
             (no weight slices; the Opt-PerQuery SLO choices are computed on train and recorded, not tested),
             routed offline from frozen artifacts; each distinct (configuration, task, trial) runs ONCE. NoMemory pairs are reused from the existing 4-trial test arm
             (``results/tau2_uncapped_4trials_20261019/test/NoMemory``, also the paired baseline); the existing
             NoMemory / VGR / VGR_SemFilter test arms are reported as references (never rerun).

Stages (``.venv-optimizer``; episodes run in ``.venv-tau2`` via ``trimem.experiments.tau2.runner_pool``):
    profile | router | select | test | report | all
    PYTHONPATH=. .venv-optimizer/bin/python -m trimem.experiments.tau2.pool_profile all

Resumable everywhere (result files per (task, trial); proposals stored once in state.json).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import aux_observations as aux_obs
from trimem.experiments import bo_newenv_datasets as dsets
from trimem.experiments import memory_policy as mp
from trimem.experiments import quality_reuse as qr
from trimem.experiments.tau2 import controlled_compare as cc
from trimem.experiments.tau2 import design as tau2_design
from trimem.experiments.tau2 import runner as R
from trimem.experiments.tau2 import runner_pool as rp
from trimem.optimizer import pruning_policy as pp_policy
from trimem.optimizer import staged_pruning as sp
from trimem.optimizer import unpaired_stats as us

ROOT = Path(__file__).resolve().parents[3]
# Session space of the selected design (``tau2.design``; TAU2_POOL_DESIGN): session-space-tau2-v5
# (``session_space_tau2_v2``, the -v5 run, default) or session-space-tau2-v6 (explore-v6 = pool-v6, user 2026-10-23)
space = tau2_design.space_module()
DESIGN = tau2_design.design()
# results/tau2_pool_20261019 = the void edges-similar_up run (session-space-tau2-v2, kept as evidence); the
# corrected space (edges similar+similar_up, session-space-tau2-v3; user 2026-10-20) ran in results/tau2_pool_20261020
# (-v4, memory_form searched; stopped, kept as evidence). -v5 (memory_form fixed compact, user 2026-10-20) runs here.
OUT = Path(os.environ.get("TAU2_POOL_OUT", ROOT / "results/tau2_pool_v5_20261020"))
PANEL_PATH = ROOT / "data/external/panels/tau2_train178.json"
REUSE = (
    ROOT / "results/tau2_uncapped_4trials_20261019/test"
)  # existing 4-trial test arms (never rerun)
REFERENCE_ARMS = ("NoMemory", "VGR", "VGR_SemFilter")
DS = "tau2"
SEED = 300
# User change 2026-10-20 (before the -v5 launch): PROFILING runs ONE trial per train task (trial 0, official seed
# derivation of seed 300 = the t0 episode of a 4-trial run; legacy single-trial file names); profiling quality =
# that trial's success (pass^1; pass^4 is not computable from one trial). TEST stays 4 trials, pass^4 primary,
# comparable to the reused 4-trial test arms. The experiment id carries ``train178-1trial`` so 4-trial rows
# (``train178-4trials``) never mix.
PROFILE_TRIALS = 1
PROFILE_K = 1  # profiling quality = pass^1 (single-trial success indicator)
# User change 2026-10-21 (test-1trial-v1): TEST runs ONE trial per test task too (trial 0, official seed-300
# derivation = the t0 episode of a 4-trial run; legacy single-trial file names), test quality = pass^1. The reused
# 4-trial reference arms (REUSE) are compared on their trial-0 episodes (REF_TRIALS = their protocol); their 4-trial
# pass^1 is reported as context only. TEST_VERSION / test_experiment carry the change (new test reuse index).
TEST_TRIALS = 1
TEST_K = 1  # test quality = pass^1
REF_TRIALS = 4  # trials per task of the reused reference arms (results/tau2_uncapped_4trials_20261019/test)
NUM_TRIALS = TEST_TRIALS  # legacy name (test protocol)
K = TEST_K
PANEL = "train178-1trial"
VERSION = "bo3obj-v1"
ANALYST_TAG = (
    "analyst-pool-v1"  # = analyzer_pool.PROTOCOL (LLM analyst channel, BO rounds >= 1)
)
# Analyst OFF switch (user 2026-10-20, live -v5 run: from round 5). NOT part of the run identity (experiment_id keeps
# ANALYST_TAG): recorded once in state.json ``analyst`` (``analyst_block``) when the τ² queue sets this env var to
# the first round proposed without the analyst; from that round on the analyst is never called (no agent /
# agent-gap rows, no DLS-from-agent starts). Every other channel is unchanged. ALF / Sci (bo_pool_driver) unaffected.
ANALYST_OFF_ENV = "TAU2_ANALYST_OFF_FROM_ROUND"
ANALYST_OFF_PROTOCOL = "analyst-off-v1"
ANALYST_OFF_REASON = (
    "user 2026-10-20: LLM analyst candidate channel off for the live τ² pool BO"
)
# Channel slots (user 2026-10-20; ``session_bo_agentenv.CHANNEL_SLOTS_PROTOCOL``): NOT part of the run identity
# (experiment id unchanged, as budget_extension / analyst); recorded once in state.json ``channel_slots`` from the
# next round to be proposed (live -v5 run: round 6). Proposals carry acquisition tag bo-acq-v3 from then on.
CHANNEL_SLOTS_TAG = "channel-slots-v1"
TEST_VERSION = "pool-v3-1trial"  # v3 (user 2026-10-21): 1 test trial, pass^1; v2 (user 2026-10-20): Opt-PerQuery test weights default / quality / tokens / latency
# Opt-PerQuery test arms (user 2026-10-20, v2): exactly these (w_Q, w_L, w_T); no weight slices, no SLO arms on test.
TEST_WEIGHTS = {
    "default": (0.6, 0.2, 0.2),
    "quality": (0.8, 0.1, 0.1),
    "tokens": (0.1, 0.1, 0.8),
    "latency": (0.1, 0.8, 0.1),
}
BATCH = 5
# Profiling pause (profiling-pause-v1, user 2026-10-21: pause after round 6 and test with the current points).
# ``profile --pause_after_round N`` (or env TAU2_PAUSE_AFTER_ROUND) stops BEFORE proposing round N+1 and writes
# profile_complete.json with ``paused_by_user`` (the router / select / test stages then run on rounds 0..N).
# RESUME: rerun ``profile`` / ``all`` WITHOUT the flag (or with a later round): the paused marker and every artifact
# derived from it (PAUSED_ARTIFACTS) are moved to OUT/paused_after_round<N>_<ts>/ and BO continues at round N+1
# (budget / state unchanged); test pairs (test/pairs) are kept and reused (each (config, task, trial) runs once).
PAUSE_ENV = "TAU2_PAUSE_AFTER_ROUND"
PAUSE_PROTOCOL = "profiling-pause-v1"
PAUSED_ARTIFACTS = (
    "profile_complete.json",
    "profile.json",
    "selection.json",
    "router",
    "test/routing.json",
    "test_report.json",
    "test_report.md",
    "report.md",
)
# Staged profiling with 3-D Pareto pruning (user 2026-10-19, ``staged_pruning``): nested 50 -> 100 -> 178 tasks
# (1 trial each, user 2026-10-20) for BO configurations; budget = the unpruned plan's 30 x 178 x 1 profiling episodes, saved
# episodes buy more BO configurations (at most MAX_CONFIGS).
# v2 (user 2026-10-20): round 0 staged too (only NoMemory protected), corner-vs-pessimistic-corner rule, plan guard.
# The live -v5 state.json was migrated from v1 during round 0 (``load_state``; ``staging.migrated_from_v1``).
STAGED_PROTOCOL = "tau2-pool-staged-v2"
STAGED_PROTOCOL_V1 = "tau2-pool-staged-v1"
STAGE_SIZES = (50, 100, 178)
BUDGET_CONFIGS = 30
MAX_CONFIGS = 60
SLO_NAMES = ("SLO-1", "SLO-2", "SLO-3")
PY_TAU2 = ROOT / ".venv-tau2/bin/python"
API_BASES = ("http://127.0.0.1:8030/v1", "http://127.0.0.1:8031/v1")
# Total concurrent episodes over all parallel arms (split evenly over the unfinished arms). Profiling: 64 (vLLM
# 8030 / 8031 at --max-num-seqs 64 = 128 sequence slots; one agent-or-user call in flight per episode + SemFilter
# judge calls). Test: 40 = the 4-trial run (lb64) whose NoMemory / VGR / VGR_SemFilter arms are reused, so the
# Opt arms' latency is measured under the same load (user 2026-10-19: keep 40 for comparability).
CONCURRENCY = int(os.environ.get("TAU2_POOL_CONCURRENCY", "64"))
TEST_CONCURRENCY = int(os.environ.get("TAU2_TEST_CONCURRENCY", "40"))
RETRIEVAL_SLOTS = int(
    os.environ.get("TAU2_RETRIEVAL_SLOTS", "12")
)  # host-wide (PG 55450: 32 connections)
SEMANTIC_WORKERS = int(
    os.environ.get("TAU2_SEMANTIC_WORKERS", "4")
)  # SemFilter judge calls per episode
# Quality reuse (quality-reuse-v1, user 2026-10-20) and the load regime (concurrency-v1): both recorded once in
# state.json (``quality_reuse`` / ``concurrency``) from the round given by the env var (never retroactive for
# episodes already run; the run id is unchanged).
REUSE_FROM_ENV = "TAU2_REUSE_FROM_ROUND"
CONCURRENCY_FROM_ENV = "TAU2_CONCURRENCY_FROM_ROUND"
REUSE_DIR = OUT / "quality_reuse"
HIT_LOG_S = 600  # per-arm reuse hit-rate log interval while arms run
STALL_S = 3600  # no new result file for this long -> stop the arm, retry (resumes)
ATTEMPTS = 5
RUN_ENV = {**dsets.RUN_ENV, mp.ENV_VAR: mp.SUCCESS_FAILURE}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}", flush=True)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    tmp.replace(path)


# ------------------------------------------------------------------ identities / panel


def experiment_id() -> str:
    """Run identity: 1-trial profiling panel (train178-1trial), space -v5 and the BO protocol incl. the LLM
    analyst channel of rounds >= 1 (``ANALYST_TAG``, user 2026-10-20)."""
    tag = os.environ.get("TAU2_POOL_RUN_TAG", "")
    if tag and (
        len(tag) > 32
        or not all(c.isascii() and (c.isalnum() or c in ".-") for c in tag)
    ):
        raise ValueError("invalid TAU2_POOL_RUN_TAG")
    suffix = f":{tag}" if tag else ""
    if DESIGN is not None:
        check_design_run()
    return mp.tag(
        f"{DS}:{PANEL}:pool-{space.VERSION}:{VERSION}+{ANALYST_TAG}{suffix}",
        mp.SUCCESS_FAILURE,
    )


def check_design_run() -> None:
    """explore-v6 (pool-v6) runs only isolated: its own TAU2_POOL_RUN_TAG and TAU2_POOL_OUT (never the -v5 dir)."""
    if not os.environ.get("TAU2_POOL_RUN_TAG") or not os.environ.get("TAU2_POOL_OUT"):
        raise SystemExit(
            f"{tau2_design.DESIGN_ENV}={DESIGN} needs its own TAU2_POOL_RUN_TAG and TAU2_POOL_OUT"
        )
    if OUT.resolve() == (ROOT / "results/tau2_pool_v5_20261020").resolve():
        raise SystemExit(f"{tau2_design.DESIGN_ENV}={DESIGN}: refusing the -v5 run dir")


def test_experiment() -> str:
    n = f"{TEST_TRIALS}trial" + ("s" if TEST_TRIALS > 1 else "")
    # explore-v6 (pool-v6): its own test opt experiment (the run tag), never mixed with the -v5 deployed test rows
    suffix = f":{os.environ['TAU2_POOL_RUN_TAG']}" if DESIGN is not None else ""
    return mp.tag(f"{DS}:test100-{n}:{TEST_VERSION}:pairs{suffix}", mp.SUCCESS_FAILURE)


def load_panel() -> dict:
    p = json.loads(PANEL_PATH.read_text())
    if p["size"] != 178 or len(p["items"]) != 178:
        raise SystemExit(f"{PANEL_PATH}: expected the 178 official τ² train tasks")
    return p


def train_keys(panel: dict) -> list[str]:
    return [x["item_id"] for x in panel["items"]]


def test_keys() -> list[str]:
    return [
        dsets.tau2_key(d, t)
        for d in dsets.TAU2_DOMAINS
        for t in dsets.tau2_split_ids(d, "test")
    ]


def domain_of(key: str) -> str:
    return key.split(":", 1)[0]


def strata_weights(keys: list[str]) -> dict[str, float]:
    n = len(keys)
    out: dict[str, float] = {}
    for k in keys:
        out[domain_of(k)] = out.get(domain_of(k), 0.0) + 1 / n
    return out


def cfg_name(cfg: dict) -> str:
    return "cfg_" + hashlib.sha256(space.key(cfg).encode()).hexdigest()[:12]


def bo_module():
    """``session_bo_agentenv`` switched to the τ² space (process-local; botorch imported by the stages that fit)."""
    from trimem.experiments import session_bo_agentenv as bo

    bo.use_space(space)
    bo.use_plans(
        space.BASE_PLANS
    )  # plan-p1rank-v1: added plans only via plan_round from their round
    return bo


# ------------------------------------------------------------------ episodes -> per-task metrics (pure)


def trials_of(split: str) -> int:
    """Trials per task: profiling (train) 1, test 4 (user 2026-10-20)."""
    return PROFILE_TRIALS if split == "train" else TEST_TRIALS


def k_of(split: str) -> int:
    """Quality pass^k of a split: profiling pass^1, test pass^4."""
    return PROFILE_K if split == "train" else TEST_K


def file_trial(split: str, trial: int) -> int | None:
    """Result-file trial index: a single-trial run writes legacy names (``runner.result_name(trial=None)``)."""
    return None if trials_of(split) == 1 else trial


def task_summary(
    rewards: list[float],
    tokens: list[float],
    latency_ms: list[float],
    k: int = K,
    steps: list[int | None] | None = None,
) -> dict:
    """One task's trials: official pass^j (j = 1..n), pass@n, the pass^k indicator (quality), mean reward,
    mean tokens / latency; agent steps (user 2026-10-21: agent calls per episode) = mean over the trials with a
    step value (``steps``, None if none: a missing value is never 0) and over the successful trials only
    (``steps_success``)."""
    n = len(rewards)
    if n < k or not (len(tokens) == len(latency_ms) == n):
        raise ValueError(f"task needs >= {k} aligned trials, got {n}")
    c = sum(cc.is_successful(r) for r in rewards)
    return {
        "trials": n,
        "successes": c,
        "rewards": list(rewards),
        "pass_hat": {j: cc.pass_hat_k(n, c, j) for j in range(1, n + 1)},
        "pass_at_n": cc.pass_at_k(n, c, n),
        "quality": cc.pass_hat_k(n, c, k),
        "mean_reward": sum(rewards) / n,
        "tokens": sum(tokens) / n,
        "latency_ms": sum(latency_ms) / n,
        **_steps_of(rewards, steps),
    }


def _steps_of(rewards: list[float], steps: list[int | None] | None) -> dict:
    st = [x for x in (steps or []) if x is not None]
    ok = [
        x
        for r, x in zip(rewards, steps or [None] * len(rewards), strict=True)
        if x is not None and cc.is_successful(r)
    ]
    return {
        "steps": sum(st) / len(st) if st else None,
        "steps_success": sum(ok) / len(ok) if ok else None,
    }


def episode_steps(r: dict) -> int | None:
    """Agent steps of one τ² episode: ``agent_calls`` (agent LLM calls), else the recorded / recomputed
    ``agent_steps`` (assistant turns, ``agent_step_metrics.tau2_steps``); None when unavailable."""
    if r.get("agent_calls") is not None:
        return int(r["agent_calls"])
    if r.get("agent_steps") is not None:
        return int(r["agent_steps"])
    if isinstance(r.get("simulation"), dict):
        from trimem.experiments import agent_step_metrics as sm

        return sm.tau2_steps(r["simulation"])["agent_steps"]
    return None


def episode_record(r: dict) -> dict:
    sem = (r.get("memory") or {}).get("semantic_tokens", 0) or 0
    return {
        "reward": float(r["reward"]),
        "tokens": float(r["agent_tokens"] + sem),
        "latency_ms": float(r["wall_ms"]),
        "steps": episode_steps(r),
        "reuse": bool(r.get("reuse")),
        "regime": rp.source_regime(
            r
        ),  # load regime of the run that produced the episode (latency)
    }


def check_episode(
    r: dict,
    *,
    key: str,
    trial: int,
    split: str,
    cfg: dict | None,
    num_trials: int | None = None,
) -> None:
    """The file is the expected (task, trial) episode of this arm (official seed; executed configuration).
    ``num_trials``: the producing run's trials per task (default: this split's; REF_TRIALS for reused arms)."""
    if (
        f"{r['domain']}:{r['task_id']}" != key
        or r.get("trial") != trial
        or r.get("num_trials") != (num_trials or trials_of(split))
    ):
        raise ValueError(f"{key}#t{trial}: result belongs to another episode")
    if (
        r.get("seed") != SEED
        or r.get("trial_seed") != R.trial_seed(SEED, trial)
        or r.get("split") != split
    ):
        raise ValueError(f"{key}#t{trial}: seed / split differ from the protocol")
    if not math.isfinite(float(r["reward"])) or not 0 <= float(r["reward"]) <= 1:
        raise ValueError(f"{key}#t{trial}: invalid official reward")
    if cfg is None:
        if r.get("mode") != "nomemory":
            raise ValueError(f"{key}#t{trial}: expected a NoMemory episode")
        return
    want = json.loads(json.dumps(asdict(rp.session_config(cfg))))
    if r.get("mode") != "memory" or r.get("config") != want:
        raise ValueError(
            f"{key}#t{trial}: executed configuration {r.get('config')} differs from {want}"
        )
    mem = r.get("memory")
    if mem is not None and mem.get("space_config_key") != space.key(cfg):
        raise ValueError(f"{key}#t{trial}: memory receipt of another configuration")


_EPISODES: dict[
    tuple, dict | None
] = {}  # (path, mtime_ns, size, split, config key) -> validated record


def read_episode(
    path: Path,
    *,
    key: str,
    trial: int,
    split: str,
    cfg: dict | None,
    num_trials: int | None = None,
) -> dict | None:
    """Validated episode record of one result file (None = absent / unfinished); memoized on (path, mtime, size):
    result files are ~100 KB (full transcript) and are written once (tmp + rename)."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    ck = (
        str(path),
        st.st_mtime_ns,
        st.st_size,
        split,
        None if cfg is None else space.key(cfg),
        num_trials,
    )
    if ck not in _EPISODES:
        r = R.read_result(path)
        if r is not None:
            check_episode(
                r, key=key, trial=trial, split=split, cfg=cfg, num_trials=num_trials
            )
        _EPISODES[ck] = None if r is None else episode_record(r)
    return _EPISODES[ck]


def arm_episodes(
    out: Path, keys: list[str], *, split: str, cfg: dict | None
) -> dict[str, dict[int, dict]]:
    """key -> trial -> episode record, for the (key, trial) result files present (validated)."""
    got: dict[str, dict[int, dict]] = {}
    for key in keys:
        d, t = key.split(":", 1)
        for k in range(trials_of(split)):
            e = read_episode(
                R.result_path(out, d, t, file_trial(split, k)),
                key=key,
                trial=k,
                split=split,
                cfg=cfg,
            )
            if e is not None:
                got.setdefault(key, {})[k] = e
    return got


def ref_episodes(
    out: Path, keys: list[str], trials: int = TEST_TRIALS
) -> dict[str, dict[int, dict]]:
    """NoMemory episodes of a reused REF_TRIALS-trial test arm (``__t<k>`` file names), trials 0..trials-1
    (test-1trial-v1: trial 0 = the same official seed as our single test trial)."""
    got: dict[str, dict[int, dict]] = {}
    for key in keys:
        d, t = key.split(":", 1)
        for k in range(trials):
            e = read_episode(
                R.result_path(out, d, t, k),
                key=key,
                trial=k,
                split="test",
                cfg=None,
                num_trials=REF_TRIALS,
            )
            if e is not None:
                got.setdefault(key, {})[k] = e
    return got


def files_present(out: Path, keys: list[str], split: str) -> int:
    """Cheap progress count: (key, trial) result files present (written last, by rename)."""
    return sum(
        R.result_path(out, *key.split(":", 1), file_trial(split, k)).exists()
        for key in keys
        for k in range(trials_of(split))
    )


def per_task(
    eps: dict[str, dict[int, dict]], keys: list[str], split: str
) -> dict[str, dict] | None:
    """Task summaries when EVERY key has all of the split's trials, else None (quality = pass^k_of(split))."""
    n = trials_of(split)
    if any(len(eps.get(k, {})) < n for k in keys):
        return None
    out = {}
    for k in keys:
        tr = [eps[k][j] for j in range(n)]
        out[k] = task_summary(
            [e["reward"] for e in tr],
            [e["tokens"] for e in tr],
            [e["latency_ms"] for e in tr],
            k_of(split),
            [e.get("steps") for e in tr],
        )
    return out


def estimate(
    tasks: dict[str, dict], keys: list[str], weights: dict[str, float], fld: str
) -> us.Estimate:
    """Domain-stratified mean over ALL keys (no finite-population correction: the 178 tasks stand for the task
    distribution; trial noise is inside each task's value)."""
    return us.stratified(
        [float(tasks[k][fld]) for k in keys],
        [domain_of(k) for k in keys],
        weights,
        None,
    )


def paired_gain(
    a: dict[str, dict], b: dict[str, dict], keys: list[str], weights: dict[str, float]
) -> us.Estimate:
    return us.stratified(
        [a[k]["quality"] - b[k]["quality"] for k in keys],
        [domain_of(k) for k in keys],
        weights,
        None,
    )


def summary_of(
    tasks: dict[str, dict], keys: list[str], weights: dict[str, float]
) -> dict:
    q, tok, lat = (
        estimate(tasks, keys, weights, f) for f in ("quality", "tokens", "latency_ms")
    )
    return {
        "q": q.mean,
        "q_se": q.se,
        "tok": tok.mean,
        "tok_se": tok.se,
        "lat": lat.mean / 1000,
        "lat_se": lat.se / 1000,
        "mean_reward": estimate(tasks, keys, weights, "mean_reward").mean,
        "pass_hat": {
            j: sum(tasks[k]["pass_hat"][j] for k in keys) / len(keys)
            for j in range(1, min(tasks[k]["trials"] for k in keys) + 1)
        },
        "pass_at_n": sum(tasks[k]["pass_at_n"] for k in keys) / len(keys),
        "n": len(keys),
        **_task_steps(tasks, keys),
    }


def _task_steps(tasks: dict[str, dict], keys: list[str]) -> dict:
    """Unweighted mean per-task agent steps (+ SE across tasks, + over successful trials); tasks without a
    step value are left out (never 0)."""
    from statistics import mean, stdev

    st = [tasks[k]["steps"] for k in keys if tasks[k].get("steps") is not None]
    ok = [
        tasks[k]["steps_success"]
        for k in keys
        if tasks[k].get("steps_success") is not None
    ]
    return {
        "steps": mean(st) if st else None,
        "steps_se": stdev(st) / len(st) ** 0.5 if len(st) > 1 else None,
        "steps_n": len(st),
        "steps_success": mean(ok) if ok else None,
    }


def bo_point(
    cfg: dict,
    tasks: dict[str, dict],
    nm: dict[str, dict],
    keys: list[str],
    weights: dict[str, float],
    experiment: str,
    panel_hash: str,
):
    """BO observation: paired pass^1 (profiling, 1 trial) gain vs NoMemory (noise = its paired SE^2) and mean tokens / latency
    (delta-method log noise), same layout as ``session_bo_agentenv.load_points_unpaired``."""
    from trimem.optimizer import session_observations as hb

    g = paired_gain(tasks, nm, keys, weights)
    tok, lat = (
        estimate(tasks, keys, weights, "tokens"),
        estimate(tasks, keys, weights, "latency_ms"),
    )
    k = space.key(cfg)
    return hb.Point(
        hashlib.sha256(k.encode()).hexdigest(),
        experiment,
        f"{DS}:{panel_hash}",
        space.validate(cfg),
        0,
        0,
        1,
        g.mean,
        tok.mean,
        lat.mean,
        (
            max(g.se**2, 1e-8),
            max(us.log_mean_var(tok), 1e-8),
            max(us.log_mean_var(lat), 1e-8),
        ),
    )


# ------------------------------------------------------------------ arms (parallel episode processes)


@dataclass
class Arm:
    name: str
    out: Path
    split: str
    keys: list[str]
    cfg: dict | None  # None = NoMemory
    experiment: str | None = None
    baseline: Path | None = None
    protocol_extra: dict = field(default_factory=dict)
    full_split: bool = False  # keys = the whole split (no --task_keys)
    reuse_index: Path | None = None  # quality reuse on (memory arms only)
    regime: str | None = None  # load regime label recorded in every result file

    @property
    def expected(self) -> int:
        return len(self.keys) * trials_of(self.split)

    def done(self) -> int:
        """Result files present (cheap; the stages validate every file when they aggregate)."""
        return files_present(self.out, self.keys, self.split)

    def validated(self) -> int:
        return sum(
            len(v)
            for v in arm_episodes(
                self.out, self.keys, split=self.split, cfg=self.cfg
            ).values()
        )


def runner_cmd(a: Arm, workers: int, n_arms: int = 0, total: int = 0) -> list[str]:
    cmd = [
        str(PY_TAU2),
        "-u",
        "-m",
        "trimem.experiments.tau2.runner_pool",
        "--mode",
        "nomemory" if a.cfg is None else "memory",
        "--split",
        a.split,
        "--out_dir",
        str(a.out),
        "--seed",
        str(SEED),
        "--num_trials",
        str(trials_of(a.split)),
        "--api_bases",
        ",".join(API_BASES),
        "--max_concurrency",
        str(workers),
        "--llm_http_connections",
        str(max(16, 2 * workers)),
        "--semantic_workers",
        str(SEMANTIC_WORKERS),
        "--retrieval_slots",
        str(RETRIEVAL_SLOTS),
        "--max_retries",
        "1",
        "--max_rss_gb",
        "8" if workers > 48 else "6" if workers > 16 else "4",
    ]
    if not a.full_split:
        if any("," in k for k in a.keys):
            raise ValueError("a τ² task key contains ','; --task_keys cannot carry it")
        cmd += ["--task_keys", ",".join(a.keys)]
    if a.regime:
        cmd += [
            "--load_regime",
            a.regime,
            "--total_concurrency",
            str(total),
            "--parallel_arms",
            str(n_arms),
        ]
    if a.cfg is not None and a.reuse_index is not None:
        cmd += ["--reuse_index", str(a.reuse_index)]
    if a.cfg is not None:
        cmd += [
            "--space_config",
            json.dumps(space.validate(a.cfg), sort_keys=True),
            "--experiment",
            a.experiment,
            "--protocol_extra",
            json.dumps(a.protocol_extra, sort_keys=True),
        ]
        if a.baseline is not None:
            cmd += ["--baseline_dir", str(a.baseline)]
    return cmd


def kill_group(p: subprocess.Popen) -> None:
    """Stop one of OUR episode processes and its children (its own session / process group)."""
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def drain_path(a: Arm) -> Path:
    return a.out / ".drain"


def rebalance_due(per: int, running: int, concurrency: int) -> bool:
    """Drain the still-running arms (no lost work: in-flight episodes finish) when relaunching them would at least
    double each arm's workers -- otherwise the slow (SemFilter) arms of a round run their tail on 1/n of the load."""
    return running > 0 and concurrency // running >= 2 * per


def _attempt(todo: list[Arm], per: int, concurrency: int) -> bool:
    """One attempt: every unfinished arm as one runner process (``per`` episodes each), per-arm stall watchdog.
    Returns True when the attempt ended by a rebalancing drain (not counted as a failed attempt)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), **RUN_ENV}
    running = {}
    for a in todo:
        a.out.mkdir(parents=True, exist_ok=True)
        drain_path(a).unlink(missing_ok=True)
        cmd = [
            *runner_cmd(a, per, len(todo), concurrency),
            "--drain_file",
            str(drain_path(a)),
        ]
        (a.out / ".command").write_text(json.dumps(cmd))
        logf = open(a.out / "runner.log", "a")
        running[a.name] = (
            a,
            subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ),
            logf,
        )
    progress = {n: (a.done(), time.time()) for n, (a, _, _) in running.items()}
    offsets = {n: log_offset(a) for n, (a, _, _) in running.items()}
    hit_t = time.time()
    drained = False
    try:
        while any(p.poll() is None for _, p, _ in running.values()):
            time.sleep(30)
            alive = [(n, a, p) for n, (a, p, _) in running.items() if p.poll() is None]
            if (
                not drained
                and len(alive) < len(running)
                and rebalance_due(per, len(alive), concurrency)
            ):
                log(
                    f"rebalance: {len(alive)} arms left at {per} workers; draining them (in-flight episodes "
                    f"finish) to relaunch at {concurrency // len(alive)}"
                )
                for _, a, _ in alive:
                    drain_path(a).touch()
                drained = True
            if time.time() - hit_t >= HIT_LOG_S:
                hit_t = time.time()
                log_hits(running, offsets, "progress")
            for n, a, p in alive:
                d = a.done()
                if d != progress[n][0]:
                    progress[n] = (d, time.time())
                elif time.time() - progress[n][1] > STALL_S:
                    log(
                        f"stall: {n} no new episode for {STALL_S}s at {d}/{a.expected}; stopping that arm"
                    )
                    kill_group(p)
    finally:
        for _, p, logf in running.values():
            kill_group(p)
            p.wait()
            logf.close()
        for n, (a, p, _) in running.items():
            drain_path(a).unlink(missing_ok=True)
            log(f"  {n}: exit {p.returncode}")
        log_hits(running, offsets, "attempt")
    return drained


def log_offset(a: Arm) -> int:
    try:
        return (a.out / "runner.log").stat().st_size
    except FileNotFoundError:
        return 0


def hits_since(a: Arm, offset: int) -> tuple[int, int]:
    """(reused, finished) episodes this attempt (runner.log lines after ``offset``; resumed ones excluded)."""
    try:
        with open(a.out / "runner.log", "rb") as f:
            f.seek(offset)
            lines = f.read().decode(errors="replace").splitlines()
    except FileNotFoundError:
        return 0, 0
    done = [x for x in lines if " reward=" in x and "(resumed)" not in x]
    return sum("(reused)" in x for x in done), len(done)


def log_hits(running: dict, offsets: dict, what: str) -> None:
    rows = []
    th = tt = 0
    for n, (a, _, _) in running.items():
        if a.reuse_index is None:
            continue
        h, t = hits_since(a, offsets[n])
        th, tt = th + h, tt + t
        rows.append(f"{n} {h}/{t}")
    if rows:
        log(
            f"quality reuse ({what}): hits/finished {th}/{tt}"
            + (f" ({th / tt:.0%})" if tt else "")
            + " -- "
            + ", ".join(rows)
        )


def run_parallel(arms: list[Arm], concurrency: int) -> dict[str, int]:
    """Run independent arms CONCURRENTLY, ``concurrency`` episodes in total split evenly over the unfinished arms
    (each arm's process pins every episode to the least-loaded of 8030 / 8031); as arms finish, the others are
    drained and relaunched with more workers (``rebalance_due``; drains are not failed attempts). Resumable."""
    failures = 0
    while failures < ATTEMPTS:
        todo = [a for a in arms if a.done() < a.expected]
        if not todo:
            break
        per = max(1, concurrency // len(todo))
        log(
            f"parallel attempt (failed so far {failures}): {len(todo)} arms x {per} episodes "
            f"({per * len(todo)} concurrent): " + ", ".join(a.name for a in todo)
        )
        if not _attempt(todo, per, concurrency):
            failures += 1
    res = {a.name: a.validated() for a in arms}
    log(
        "parallel done: "
        + ", ".join(f"{a.name} {res[a.name]}/{a.expected}" for a in arms)
    )
    return res


# ------------------------------------------------------------------ profiling


def new_state(panel: dict, a, previous=None) -> dict:
    keys = train_keys(panel)
    seed = sp.seed_of(experiment_id(), "stages")
    full_cost = len(keys) * PROFILE_TRIALS
    budget = sp.Budget.unpruned(a.budget_configs, full_cost, a.max_configs)
    state = {
        "dataset": DS,
        "experiment": experiment_id(),
        "panel_hash": panel["panel_hash"],
        "pool_size": panel["size"],
        "num_trials": PROFILE_TRIALS,
        "seed": SEED,
        "quality": f"pass^{PROFILE_K}",
        "memory_policy": mp.SUCCESS_FAILURE,
        "space_version": space.VERSION,
        "batch": a.batch,
        "staging": {
            "protocol": STAGED_PROTOCOL,
            "rule": sp.PROTOCOL,
            "z": sp.Z,
            "sizes": list(STAGE_SIZES),
            "order_seed": seed,
            "order": sp.stage_order(keys, domain_of, seed),
            "budget_episodes": budget.total_episodes,
            "budget_configs_unpruned": a.budget_configs,
            "full_cost": full_cost,
            "max_configs": budget.max_configs,
            "round0_staged": True,
            "comparators": "full configurations: point estimate (all 178 tasks); NoMemory and configurations "
            "evaluated at >= the stage: pessimistic corner on the same stage subset (paired gain vs NoMemory on "
            "those tasks; NoMemory gain 0, SE 0); z = 1.96; fixed checkpoints",
            "plan_guard": True,
            "budget_counts": "episodes actually run per configuration (result files on all 178 tasks), "
            "round 0 included",
        },
        "created_at": datetime.now(UTC).isoformat(),
        "rounds": [],
    }
    if (
        DESIGN is not None
    ):  # explore-v6 (pool-v6): recorded at creation, compared on every start
        state["design"] = tau2_design.block(space.VERSION, state["created_at"])
    state["staging"] = pp_policy.configure(state["staging"], a, previous)
    return state


def same_design(a: dict | None, b: dict | None) -> bool:
    """Design blocks equal up to their record time. Pure."""
    strip = lambda d: (
        None if d is None else {k: v for k, v in d.items() if k != "recorded_at"}
    )  # noqa: E731
    return strip(a) == strip(b)


def load_state(panel: dict, a) -> tuple[dict, Path]:
    from trimem.optimizer import search_analyzer as sa

    p = OUT / "state.json"
    st = json.loads(p.read_text()) if p.exists() else None
    want = new_state(panel, a, st.get("staging", {}) if st is not None else None)
    st = want if st is None else st
    sa.settings_for(st, a, existing=p.exists())
    if sa.baselines.active(st["search_design"]) or pp_policy.requires_isolation(
        want["staging"], st["search_design"]
    ):
        if (
            not os.environ.get("TAU2_POOL_RUN_TAG")
            or OUT.resolve() == (ROOT / "results/tau2_pool_v5_20261020").resolve()
        ):
            raise ValueError(
                "baseline/alternative pruning requires its own TAU2_POOL_RUN_TAG and TAU2_POOL_OUT"
            )
    for k in (
        "experiment",
        "panel_hash",
        "num_trials",
        "seed",
        "quality",
        "memory_policy",
        "space_version",
        "batch",
    ):
        if st[k] != want[k]:
            raise SystemExit(
                f"state.json has {k}={st[k]!r}, this run {want[k]!r}; refusing to mix"
            )
    if not same_design(st.get("design"), want.get("design")):
        raise SystemExit(
            f"state.json design {(st.get('design') or {}).get('protocol')!r} != this run "
            f"{(want.get('design') or {}).get('protocol')!r} ({tau2_design.DESIGN_ENV}); refusing to mix"
        )
    if st.get("staging", {}).get("protocol") == STAGED_PROTOCOL_V1:
        migrate_v1(st, want["staging"])
    if {
        k: v for k, v in st.get("staging", {}).items() if k not in (MIGRATED, ASYNC_KEY)
    } != want["staging"]:
        raise SystemExit(
            "state.json has another staging protocol / order / budget; refusing to mix"
        )
    return st, p


MIGRATED = "migrated_from_v1"


def migrate_v1(st: dict, want: dict) -> None:
    """v1 -> v2 (user 2026-10-20): allowed only while nothing was decided under v1 (round 0 proposed, no BO round,
    no decision) and with the same order / sizes / budget. Round 0 is staged RETROACTIVELY from now on: its
    completed episodes are kept (runners resume from result files); the time is recorded."""
    old = st["staging"]
    same = all(
        old.get(k) == want[k]
        for k in ("z", "sizes", "order_seed", "order", "budget_episodes", "full_cost")
    )
    decided = any(
        staging_of(c)["decisions"] or staging_of(c)["pruned_at"]
        for rr in st["rounds"]
        for c in rr["configs"]
    )
    if not same or len(st["rounds"]) > 1 or decided:
        raise SystemExit(
            "state.json is staged-v1 with BO rounds / decisions or another order; refusing to migrate"
        )
    for rr in st["rounds"]:
        for c in rr["configs"]:
            staging_of(c).pop("protected", None)
    st["staging"] = {
        **want,
        MIGRATED: {
            "at": datetime.now(UTC).isoformat(),
            "protocol": old["protocol"],
            "rule": old.get("rule"),
            "rounds_at_migration": len(st["rounds"]),
            "note": "round 0 staged retroactively from this time (user 2026-10-20); episodes completed before "
            "are kept; each decision uses the nested stage subset only",
        },
    }


# Budget extension (user 2026-10-20, live -v5 run: "total 50 configs"). NOT part of the run identity (``staging``
# keeps the original 5,340-episode plan, so ``load_state`` still matches): recorded once in state.json
# ``budget_extension`` while the driver is stopped (``extend_budget``). With it, the episode budget is ``to`` and the
# configuration cap counts EVALUATED BO configurations only (fully evaluated + pruned; NoMemory and the
# configurations stopped by the user excluded), capped at ``target_evaluated_configs``.
BUDGET_EXTENSION_PROTOCOL = "budget-extension-v1"


def budget_of(state: dict) -> sp.Budget:
    g = state["staging"]
    if ext := state.get("budget_extension"):
        return sp.Budget(ext["to"], g["full_cost"], ext["target_evaluated_configs"])
    return sp.Budget(g["budget_episodes"], g["full_cost"], g["max_configs"])


def stopped_by_user(c: dict) -> bool:
    return bool(staging_of(c).get("stopped_by_user"))


def counted_configs(state: dict) -> int:
    """Configurations counted against ``Budget.max_configs``: every proposed one (the original plan), or -- with a
    ``budget_extension`` -- the evaluated BO configurations only (proposed minus stopped by the user; running ones
    count, they end full or pruned)."""
    cs = [c for rr in state["rounds"] for c in rr["configs"]]
    if state.get("budget_extension"):
        return sum(not stopped_by_user(c) for c in cs)
    return len(cs)


def extend_budget(state: dict, to: int, target: int, reason: str, at: str) -> dict:
    """Record the budget extension once (idempotent for the same ``to`` / ``target``; refuses another one, a
    decrease, or a target below the configurations already counted)."""
    g = state["staging"]
    blk = {
        "protocol": BUDGET_EXTENSION_PROTOCOL,
        "from": g["budget_episodes"],
        "to": to,
        "reason": reason,
        "at": at,
        "target_evaluated_configs": target,
        "counts": "evaluated BO configurations = fully evaluated + pruned; NoMemory and the configurations "
        "stopped by the user (sim_gate) excluded",
        "max_configs_before": g["max_configs"],
        "episodes_used_at": episodes_used(state),
        "rounds_at": len(state["rounds"]),
        "note": "not part of the run identity (experiment id / staging unchanged); savings from pruning buy "
        "further configurations, but proposals stop at target_evaluated_configs",
    }
    if old := state.get("budget_extension"):
        if (old["to"], old["target_evaluated_configs"]) != (to, target):
            raise SystemExit(
                f"state.json budget_extension {old} differs; refusing to mix"
            )
        return old
    if to < g["budget_episodes"] or to < blk["episodes_used_at"]:
        raise ValueError(f"budget extension to {to} below the plan / episodes used")
    state["budget_extension"] = blk
    if counted_configs(state) > target:
        del state["budget_extension"]
        raise ValueError(f"target {target} below the configurations already evaluated")
    return blk


def stages_of(state: dict) -> list[list[str]]:
    g = state["staging"]
    return sp.stage_subsets(g["order"], g["sizes"])


def nomemory_dir() -> Path:
    return OUT / "NoMemory"


def config_dir(cfg: dict) -> Path:
    return OUT / "configs" / cfg_name(cfg)


def proposal_block(state: dict) -> dict:
    """Proposal-side constraint (user 2026-10-20): every configuration proposed from ``from_round`` on has
    ``space.PROPOSE_FIXED`` (sim_gate none). Recorded once in state.json: from the next round to be proposed (the
    live -v5 run: round 1; its round-0 configurations, gated ones included, are kept)."""
    if "proposal" not in state:
        state["proposal"] = {
            "protocol": space.PROPOSAL_PROTOCOL,
            "fixed": dict(space.PROPOSE_FIXED),
            "from_round": len(state["rounds"]),
            "recorded_at": datetime.now(UTC).isoformat(),
            "note": "proposal-side only: measured configurations with a gate stay valid GP observations "
            "(sim_gate stays a GP feature) and may be selected / routed when fully evaluated",
        }
    elif (
        state["proposal"]["fixed"] != space.PROPOSE_FIXED
        or state["proposal"]["protocol"] != space.PROPOSAL_PROTOCOL
    ):
        raise SystemExit(
            "state.json has another proposal-side constraint; refusing to mix"
        )
    return state["proposal"]


EXTENSION_REASON = (
    "user 2026-10-20: the search space must contain V -> SemFilter -> G -> Rank (P1+Rank) next to "
    "V -> G -> R -> SemFilter -> Rank (P2+Rank, unchanged)"
)


def extension_block(state: dict) -> dict:
    """Plan extension (``space.PLAN_EXTENSION``, plan-p1rank-v1; user 2026-10-20), recorded once in state.json
    ``space_extension``: the added plans are proposable from ``from_round`` on (rounds before stay proposed over
    ``space.BASE_PLANS``). NOT part of the run identity (experiment id / space_version unchanged). Never
    retroactive: refuses when round ``from_round`` was already proposed without the block."""
    ext = space.PLAN_EXTENSION
    blk = state.get("space_extension")
    if blk is None:
        if len(state["rounds"]) > ext["from_round"]:
            raise SystemExit(
                f"space extension {ext['protocol']} from round {ext['from_round']}: rounds 0.."
                f"{len(state['rounds']) - 1} already proposed without it; never retroactive"
            )
        state["space_extension"] = blk = {
            "protocol": ext["protocol"],
            "from_round": ext["from_round"],
            "added_plans": list(ext["added_plans"]),
            "display": {p: space.DISPLAY[p] for p in ext["added_plans"]},
            "base_plans": list(space.BASE_PLANS),
            "reason": EXTENSION_REASON,
            "recorded_at": datetime.now(UTC).isoformat(),
            "note": "not part of the run identity (experiment id and space_version unchanged); P1+Rank = SemFilter "
            "on the vector seeds before graph expansion, then expansion, pool cap, ColBERT rank, top-k; knobs and "
            "defaults as P2+Rank; GP plan category appended (index 4, NoMemory 5); while unmeasured, the plan "
            "default is an extra DLS start (dls-newplan) and the Sobol breadth draws over 5 plans",
        }
    elif (
        blk.get("protocol") != ext["protocol"]
        or blk.get("from_round") != ext["from_round"]
        or blk.get("added_plans") != list(ext["added_plans"])
    ):
        raise SystemExit(f"state.json space_extension {blk} != {ext}; refusing to mix")
    return blk


def proposal_plans(state: dict, r: int) -> tuple[str, ...]:
    """Plans proposable in round ``r`` (``space.proposal_plans`` with the recorded extension block)."""
    return space.proposal_plans(r, state.get("space_extension") or space.PLAN_EXTENSION)


def analyst_block(state: dict) -> dict | None:
    """Analyst OFF switch (``ANALYST_OFF_ENV``; user 2026-10-20), recorded once in state.json ``analyst``; never
    retroactive (refuses a round already proposed with the analyst) and never silently re-enabled (a recorded
    block persists without the env var). Returns the block, or None (analyst on: the default)."""
    raw = os.environ.get(ANALYST_OFF_ENV, "").strip()
    blk = state.get("analyst")
    if not raw:
        return blk
    off = int(raw)
    if blk is None:
        if off < len(state["rounds"]):
            raise SystemExit(
                f"{ANALYST_OFF_ENV}={off}: rounds 0..{len(state['rounds']) - 1} already proposed (with the analyst); "
                "the switch is never retroactive"
            )
        state["analyst"] = blk = {
            "protocol": ANALYST_OFF_PROTOCOL,
            "channel": ANALYST_TAG,
            "enabled": False,
            "off_from_round": off,
            "reason": ANALYST_OFF_REASON,
            "recorded_at": datetime.now(UTC).isoformat(),
            "note": "not part of the run identity (experiment id unchanged; episodes / opt rows kept); from "
            "off_from_round on no analyst call, no agent / agent-gap rows, no dls-agent starts; Sobol, crossover, "
            "DLS (incumbents / random / pool / qfirst champion), forced gap (rectangle computed without the LLM), "
            "the qfirst slot and gp-fix-v1 unchanged",
        }
    elif blk.get("off_from_round") != off or blk.get("enabled", True):
        raise SystemExit(
            f"state.json analyst block {blk} != {ANALYST_OFF_ENV}={off}; refusing to mix"
        )
    return blk


def analyst_off(state: dict, r: int, a) -> dict | None:
    """Why round ``r`` runs without the analyst (None = analyst on)."""
    if not getattr(a, "analyst", True):
        return {"by": "--no_analyst"}
    blk = state.get("analyst")
    if blk and not blk.get("enabled", True) and r >= blk["off_from_round"]:
        return {
            "by": "state.json analyst",
            "off_from_round": blk["off_from_round"],
            "reason": blk["reason"],
        }
    return None


def concurrency_block(state: dict, a) -> dict | None:
    """Load regime from ``CONCURRENCY_FROM_ENV`` (user 2026-10-20: ~100 concurrent profiling episodes from round 6),
    recorded once; never retroactive for a round already finished."""
    raw = os.environ.get(CONCURRENCY_FROM_ENV, "").strip()
    blk = state.get("concurrency")
    if not raw:
        return blk
    frm = int(raw)
    want = {
        "protocol": "concurrency-v1",
        "from_round": frm,
        "profiling_total": a.concurrency,
        "profiling_regime": f"c{a.concurrency}",
        "test_total": a.test_concurrency,
        "test_regime": f"c{a.test_concurrency}",
        "before": {"profiling_total": 64, "regime": rp.LEGACY_REGIME},
    }
    if blk is None:
        if frm < len(state["rounds"]) - 1:
            raise SystemExit(
                f"{CONCURRENCY_FROM_ENV}={frm}: rounds before the current one are done"
            )
        state["concurrency"] = blk = {
            **want,
            "recorded_at": datetime.now(UTC).isoformat(),
            "reason": "user 2026-10-20: raise τ² profiling concurrency (~60 -> ~100 episodes, 5 arms x 20); test "
            "stays 40",
            "latency_caveat": "latency from round from_round on is measured under higher load than rounds before "
            "it (BO latency objective comparability); every result file records load.regime / arm_workers / "
            "total_concurrency / parallel_arms, and staging.load per configuration counts episodes per regime, so "
            "latency can be load-adjusted later",
        }
    elif {k: blk.get(k) for k in want} != want:
        raise SystemExit(
            f"state.json concurrency block {blk} != this run {want}; refusing to mix"
        )
    return blk


def regime_of(state: dict, r: int | None) -> str:
    blk = state.get("concurrency")
    if blk and r is not None and r >= blk["from_round"]:
        return blk["profiling_regime"]
    return rp.LEGACY_REGIME


def reuse_block(state: dict) -> dict | None:
    """Quality reuse from ``REUSE_FROM_ENV`` (quality-reuse-v1, user 2026-10-20), recorded once (+ the index
    backfill from the existing episodes' receipts)."""
    raw = os.environ.get(REUSE_FROM_ENV, "").strip()
    blk = state.get("quality_reuse")
    if not raw:
        return blk
    frm = int(raw)
    if blk is None:
        if frm < len(state["rounds"]) - 1:
            raise SystemExit(
                f"{REUSE_FROM_ENV}={frm}: rounds before the current one are done"
            )
        idx = qr.ReuseIndex(qr.index_path(REUSE_DIR, state["experiment"], "train"))
        n = backfill(
            idx, state["experiment"], "train", sorted((OUT / "configs").glob("cfg_*"))
        )
        state["quality_reuse"] = blk = {
            "protocol": qr.PROTOCOL,
            "from_round": frm,
            "recorded_at": datetime.now(UTC).isoformat(),
            "key": "sha256(experiment, split, task key, trial, trial seed, run memory policy, model, memory_key); "
            "memory_key = sha256(memory_form, memory_forms render version, exact injected <memory> text)",
            "regime_in_key": False,
            "regime_rule": "coordinator 2026-10-20 (b): the load regime is NOT in the key -- c64 episodes (rounds "
            "0-5) are reusable from round from_round on; each reuse record carries source_regime and staging.reuse "
            "counts hits per source regime",
            "copied": "reward / quality, the source's agent tokens and episode latency (source wall - source "
            "retrieval ms); THIS configuration's own retrieval receipt, semantic tokens and retrieval ms are "
            "recorded (tokens = source agent tokens + own semantic tokens; latency = source episode ms + own "
            "retrieval ms)",
            "scope": "profiling (train) from from_round; the test stage (test index, own file; test regime). Never "
            "across train/test, experiments, tasks, trials or models; never from NoMemory",
            "empty_memory": qr.EMPTY_MEMORY_RULE,
            "in_flight": "a same-key episode in flight (claim held) is waited for, then reused",
            "budget": "reuse hits cost 0 budget episodes (staging.reused is subtracted in episodes_used): the "
            "budget buys configurations; proposals still stop at budget_extension.target_evaluated_configs (50)",
            "evaluated": "a reuse record is a result file: it counts as an evaluated episode for staging, pruning, "
            "the GP and the router",
            "index": str(qr.index_path(REUSE_DIR, state["experiment"], "train")),
            "backfill": {
                "source": "result files of configs/cfg_* (memory_text + memory.memory_form receipt of every "
                "original episode: the exact injected text is recorded, so keys are exact, not reconstructed)",
                "episodes": n,
                "regime": rp.LEGACY_REGIME,
            },
            "se_caveat": "reuse fixes the user-simulator / environment randomness for identical inputs: configurations "
            "that share reused episodes are positively correlated, so SEs of differences between them are understated",
        }
    elif blk.get("from_round") != frm or blk.get("protocol") != qr.PROTOCOL:
        raise SystemExit(
            f"state.json quality_reuse block {blk} != {REUSE_FROM_ENV}={frm}; refusing to mix"
        )
    return blk


def backfill(idx: qr.ReuseIndex, experiment: str, split: str, dirs: list[Path]) -> int:
    """Index every original memory episode of ``dirs`` (idempotent: sources already indexed are skipped)."""
    idx._refresh()
    have = {e["source"] for es in idx._entries.values() for e in es}
    n = 0
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            if f.name == rp.ARM_FILE:
                continue
            r = R.read_result(f)
            if (
                r is None
                or r.get("reuse")
                or r.get("mode") != "memory"
                or r.get("split") != split
            ):
                continue
            if r.get("experiment") != experiment:
                continue
            ek = rp.result_episode_key(r, experiment)
            src = str(f.resolve())
            if ek is None or src in have:
                continue
            idx.append(
                {
                    "key": ek,
                    "kind": "backfill",
                    "source": src,
                    "config_key": (r.get("memory") or {}).get("space_config_key"),
                    "memory_key": rp.result_memory_key(r),
                    "task": f"{r['domain']}:{r['task_id']}",
                    "trial": r.get("trial") or 0,
                    "split": split,
                    "regime": rp.source_regime(r),
                    "at": qr.now(),
                }
            )
            have.add(src)
            n += 1
    return n


def reuse_index_for(state: dict, r: int | None) -> Path | None:
    blk = state.get("quality_reuse")
    if blk and r is not None and r >= blk["from_round"]:
        return qr.index_path(REUSE_DIR, state["experiment"], "train")
    return None


def reuse_stats(cfg: dict, keys: list[str]) -> dict:
    """Per configuration: reuse hits (by source regime) and own load regimes over ``keys`` (validated files)."""
    eps = arm_episodes(config_dir(cfg), keys, split="train", cfg=cfg)
    recs = [e for v in eps.values() for e in v.values()]
    by_src: dict[str, int] = {}
    by_own: dict[str, int] = {}
    for e in recs:
        if e["reuse"]:
            by_src[e["regime"]] = by_src.get(e["regime"], 0) + 1
        else:
            by_own[e["regime"]] = by_own.get(e["regime"], 0) + 1
    hits = sum(by_src.values())
    return {
        "episodes": len(recs),
        "hits": hits,
        "hit_rate": hits / len(recs) if recs else 0.0,
        "latency_source_regime": {
            **by_own,
            **{f"reused:{k}": v for k, v in by_src.items()},
        },
    }


def config_arm(
    cfg: dict,
    label: str,
    keys: list[str],
    panel: dict,
    state: dict,
    rnd: int | None = None,
) -> Arm:
    prop = state.get("proposal")
    extra = (
        {"proposal": prop["protocol"]}
        if prop and rnd is not None and rnd >= prop["from_round"]
        else {}
    )
    return Arm(
        label,
        config_dir(cfg),
        "train",
        keys,
        space.validate(cfg),
        state["experiment"],
        nomemory_dir(),
        {
            "panel_hash": panel["panel_hash"],
            "bo_label": label,
            "bo_experiment": state["experiment"],
            "quality": f"pass^{PROFILE_K}",
            "staging": STAGED_PROTOCOL,
            "stage_n": len(keys),
            **extra,
        },
        full_split=len(keys) == panel["size"],
        reuse_index=reuse_index_for(state, rnd),
        regime=regime_of(state, rnd),
    )


def profiled(state: dict) -> list[tuple[dict, str, int, int]]:
    return [
        (c["config"], c.get("channel"), rr["round"], j)
        for rr in state["rounds"]
        for j, c in enumerate(rr["configs"])
    ]


def nomemory_tasks(keys: list[str]) -> dict[str, dict]:
    nm = per_task(
        arm_episodes(nomemory_dir(), keys, split="train", cfg=None), keys, "train"
    )
    if nm is None:
        raise SystemExit("train NoMemory arm incomplete")
    return nm


def config_tasks(state: dict, keys: list[str]) -> dict[str, dict]:
    """config key -> per-task summaries on ALL ``keys`` (fully evaluated configurations only)."""
    out = {}
    pruned = {
        space.key(c["config"])
        for rr in state["rounds"]
        for c in rr["configs"]
        if inactive(c)
    }
    for cfg, _, _, _ in profiled(state):
        if space.key(cfg) in pruned:
            continue
        t = per_task(
            arm_episodes(config_dir(cfg), keys, split="train", cfg=cfg), keys, "train"
        )
        if t is not None:
            out[space.key(cfg)] = t
    return out


def obj_of(
    tasks: dict[str, dict], nm: dict[str, dict], keys: list[str], w: dict[str, float]
) -> sp.Obj:
    """Pruning objectives on ``keys``: paired pass^1 gain vs NoMemory (= the BO quality), mean tokens, latency."""
    g = paired_gain(tasks, nm, keys, w)
    tok, lat = (
        estimate(tasks, keys, w, "tokens"),
        estimate(tasks, keys, w, "latency_ms"),
    )
    return sp.Obj(g.mean, tok.mean, lat.mean, g.se, tok.se, lat.se, len(keys))


def inactive(c: dict) -> bool:
    """Pruned at a checkpoint or stopped by the user: never run further, never selected / routed; stays a GP
    observation with its partial-n noise."""
    g = staging_of(c)
    return bool(g["pruned_at"] or g.get("stopped_by_user"))


STOP_REASON = "sim_gate fixed none (user decision 2026-10-20)"


def stop_violating(state: dict, keys: list[str]) -> list[str]:
    """User decision 2026-10-20: configurations proposed BEFORE the proposal-side constraint (``proposal_block``)
    that violate it (round-0 configurations with a sim_gate) and are not fully evaluated are STOPPED once:
    ``staging.stopped_by_user`` records the reason, the time, the nested stage reached (largest complete prefix),
    the episodes run and the tasks they are observed on (every completed task: their GP observation, partial-n
    noise). Their unused episodes return to the BO budget (``episodes_used`` counts what ran)."""
    prop, out = state["proposal"], []
    sizes, order = state["staging"]["sizes"], state["staging"]["order"]
    for rr in state["rounds"]:
        if rr["round"] >= prop["from_round"]:
            continue
        for c in rr["configs"]:
            cfg, g = c["config"], staging_of(c)
            if inactive(c) or all(cfg.get(k) == v for k, v in prop["fixed"].items()):
                continue
            eps = arm_episodes(config_dir(cfg), keys, split="train", cfg=cfg)
            done = [k for k in keys if len(eps.get(k, {})) >= PROFILE_TRIALS]
            if len(done) == len(keys):
                continue  # fully evaluated: kept (selectable)
            stage = max([n for n in sizes if set(order[:n]) <= set(done)], default=0)
            g.update(
                stage_reached=stage,
                episodes=len(done) * PROFILE_TRIALS,
                stopped_by_user={
                    "reason": STOP_REASON,
                    "at": datetime.now(UTC).isoformat(),
                    "stage_reached": stage,
                    "episodes": len(done) * PROFILE_TRIALS,
                    "observed_keys": done,
                },
            )
            out.append(space.key(cfg))
    return out


def observed_keys(state: dict, c: dict, keys: list[str]) -> list[str]:
    """Tasks a configuration is OBSERVED on (GP input): all when fully evaluated, the last stage prefix when
    pruned, every completed task when stopped by the user (not a nested prefix; the stratified estimator
    renormalizes over the domains present)."""
    g = staging_of(c)
    if g.get("stopped_by_user"):
        return list(g["stopped_by_user"]["observed_keys"])
    n = g["stage_reached"]
    return state["staging"]["order"][:n] if n < len(keys) else list(keys)


def staging_of(c: dict) -> dict:
    return c.setdefault(
        "staging",
        {"stage_reached": 0, "pruned_at": None, "decisions": [], "episodes": 0},
    )


def episodes_used(state: dict) -> int:
    """Profiling episodes of the configurations, round 0 included (NoMemory excluded, as in the unpruned plan): the
    episodes actually run (``staging.episodes``: result files on all tasks, so round-0 episodes run before the v2
    migration outside the stage prefix count too), at least the stage reached."""
    return sum(
        max(staging_of(c)["episodes"], staging_of(c)["stage_reached"] * PROFILE_TRIALS)
        - staging_of(c).get(
            "reused", 0
        )  # quality reuse: hits cost 0 budget episodes (quality-reuse-v1)
        for rr in state["rounds"]
        for c in rr["configs"]
    )


def full_objectives(
    state: dict, keys: list[str], w: dict[str, float]
) -> dict[str, sp.Obj]:
    """Fully evaluated configurations -> point estimates on all tasks (paired gain vs the full NoMemory arm)."""
    tasks = config_tasks(state, keys)
    if not tasks:
        return {}
    nm = nomemory_tasks(keys)
    return {k: obj_of(t, nm, keys, w) for k, t in tasks.items()}


def stage_comparators(
    state: dict, sub: list[str], nm_sub: dict[str, dict], w: dict[str, float], full: set
) -> dict[str, sp.Obj]:
    """Stage comparators on the subset ``sub``: NoMemory (paired gain 0, SE 0; its tokens / latency on ``sub``) and
    every configuration evaluated at >= this stage that is not fully evaluated (pruned ones included), each on
    ``sub`` with its paired gain vs NoMemory on the same tasks."""
    nmo = obj_of(nm_sub, nm_sub, sub, w)
    out = {
        space.key(space.NOMEM): sp.Obj(
            0.0, nmo.tokens, nmo.latency, 0.0, nmo.tokens_se, nmo.latency_se, len(sub)
        )
    }
    for rr in state["rounds"]:
        for c in rr["configs"]:
            k = space.key(c["config"])
            if k in full or staging_of(c)["stage_reached"] < len(sub):
                continue
            t = per_task(
                arm_episodes(
                    config_dir(c["config"]), sub, split="train", cfg=c["config"]
                ),
                sub,
                "train",
            )
            if t is None:
                raise SystemExit(f"{k}: stage {len(sub)} recorded but incomplete")
            out[k] = obj_of(t, nm_sub, sub, w)
    return out


def common_mode_quality(
    state: dict, keys: list[str], nm: dict[str, dict], w: dict[str, float]
) -> tuple[float, int]:
    """gp-fix-v1: common-mode covariance of the paired-gain estimators = the mean pairwise stratified covariance of
    the per-task paired differences (config - NoMemory) of the fully evaluated configurations (all on the same 178
    tasks against the same NoMemory arm). Returns (kappa, number of configurations it is estimated from)."""
    series = [
        [t[k]["quality"] - nm[k]["quality"] for k in keys]
        for t in config_tasks(state, keys).values()
    ]
    return us.common_mode_cov(series, [domain_of(k) for k in keys], w), len(series)


def observed_points(
    state: dict, keys: list[str], panel: dict, common_mode: bool = False
) -> dict:
    """BO observations of every profiled configuration: fully evaluated ones on all tasks, pruned ones on the tasks
    of the last stage they ran (partial-n means; larger SE = larger GP noise). ``common_mode`` (gp-fix-v1): the
    quality noise is the paired SE^2 minus ``common_mode_quality`` (floor ``us.IDIO_FLOOR`` x SE^2)."""
    w = strata_weights(keys)
    nm = nomemory_tasks(keys)
    kappa = common_mode_quality(state, keys, nm, w)[0] if common_mode else 0.0
    pts = {}
    for rr in state["rounds"]:
        for c in rr["configs"]:
            sub = observed_keys(state, c, keys)
            n = len(sub)
            if not sub:
                continue  # stopped before any episode: no observation
            t = per_task(
                arm_episodes(
                    config_dir(c["config"]), sub, split="train", cfg=c["config"]
                ),
                sub,
                "train",
            )
            if t is None:
                raise SystemExit(f"{space.key(c['config'])}: stage {n} incomplete")
            p = bo_point(
                c["config"], t, nm, sub, w, state["experiment"], panel["panel_hash"]
            )
            if common_mode:
                p = replace(
                    p, noise=(us.idiosyncratic_var(p.noise[0], kappa), *p.noise[1:])
                )
            pts[space.key(c["config"])] = p
    return pts


GP_NOISE = (
    "quality: paired SE^2 minus the common-mode covariance shared by every point (same tasks, same NoMemory "
    "arm; unpaired_stats.common_mode_cov over the fully evaluated configurations), floor 25% of SE^2; costs: "
    "delta-method log SE^2 (unchanged)"
)


def log_gp(r: int, pts: dict, ctx, aux: list | None = None) -> dict:
    """Fit once on the round's points and log what the proposals see: per output the posterior-mean spread over
    the measured configurations, the posterior SD range, the outputscales and the degeneracy flag."""
    from trimem.optimizer import session_pool as sp_pool

    model, _ = sp_pool.fit(list(pts.values()) + list(aux or ()))
    cfgs = [p.cfg for p in pts.values()]  # current configurations only (source 0)
    mean, var = sp_pool.panel_posterior(model, ctx, cfgs)
    out = {"protocol": model.gp_protocol, "aux_points": len(aux or ()), "outputs": []}
    for i, (name, d) in enumerate(
        zip(("gain", "-ln tokens", "-ln latency"), model.gp_diagnostics, strict=True)
    ):
        mu, sd = mean[:, i], var[:, i].clamp_min(0).sqrt()
        row = {
            "output": name,
            **d,
            "mean_min": float(mu.min()),
            "mean_max": float(mu.max()),
            "sd_min": float(sd.min()),
            "sd_max": float(sd.max()),
        }
        out["outputs"].append(row)
        log(
            f"round {r} GP {model.gp_protocol} {name}: posterior mean [{row['mean_min']:+.4f}, "
            f"{row['mean_max']:+.4f}] (SD over configs {d['post_mean_sd']:.4f} vs observed {d['obs_sd']:.4f}), "
            f"posterior SD [{row['sd_min']:.4f}, {row['sd_max']:.4f}], outputscales "
            f"{[round(x, 4) for x in d['outputscales']]}"
            + (" DEGENERATE" if d["degenerate"] else "")
        )
    return out


def plan_round(
    state: dict, r: int, keys: list[str], panel: dict, a, k: int
) -> list[dict]:
    bo = bo_module()
    from trimem.optimizer import search_analyzer as sa

    search_settings = sa.settings_for(state, a)
    plans = proposal_plans(state, r)
    if sa.baselines.active(search_settings):
        plans = tuple(
            space.PLANS
        )  # fresh baselines use a fixed five-plan space from initialization
    if r == 0:
        bo.use_plans(plans)
        if sa.baselines.active(search_settings):
            return sa.baselines.initialize(bo, search_settings)
        if DESIGN is not None:  # explore-v6: 10 Sobol, no plan defaults
            return round0_sobol_only(bo, k)
        sampler = sa.initialization_sampler(search_settings)
        return [
            {"config": c, "channel": "default" if i < len(plans) else sampler}
            for i, c in enumerate(
                bo.round0() if sampler == "sobol" else bo.round0(sampler=sampler)
            )
        ]
    on = bo.use_gp_for(state, r)
    # hv-ref-anchored-v1 from the recorded round on: BO quality = paired gain vs NoMemory (baseline 0 in objective
    # units); its SE = the NoMemory arm's domain-stratified quality SE over the train tasks
    if state.get("hv_ref"):
        nm_se = estimate(nomemory_tasks(keys), keys, strata_weights(keys), "quality").se
        if bo.use_hv_ref_for(state, r, (0.0, nm_se)):
            log(
                f"round {r} {bo.pool.HV_REF_ANCHORED}: quality ref = 0 - NoMemory SE {nm_se:.4f}"
            )
    pts = observed_points(state, keys, panel, common_mode=on)
    if sa.baselines.active(search_settings):
        bo.use_plans(plans)
        picks = sa.baselines.propose(
            list(pts.values()),
            bo.context(DS, panel["panel_hash"]),
            k,
            search_settings,
            r,
        )
        log(
            f"round {r}: {search_settings['protocol']} selected {len(picks)} configurations (no channel pool)"
        )
        return picks
    # aux-obs-v1 (user 2026-10-20): GP-only observations of the earlier τ² train178 run from the recorded round
    aux = bo.aux_points(state, r, bo.task_of(DS, panel["panel_hash"]))
    if aux:
        log(
            f"round {r} {bo.AUX_PROTOCOL}: {len(aux)} auxiliary GP observations (source dimension on)"
        )
    gp_report = log_gp(r, pts, bo.context(DS, panel["panel_hash"]), aux) if on else None
    forced = bo.stalled(
        [[pts[space.key(c["config"])] for c in rr["configs"]] for rr in state["rounds"]]
    )
    slots = bo.channel_slots_settings(state, r)
    recovery = sa.stall_recovery_enabled(search_settings)
    # Modern layouts omit qfirst. Historical v5/v6 suppress stalls unless the
    # run records front-gap-v1; explicit sobol-only rounds always skip the analyst.
    modern = bool(slots) and slots["protocol"] in (
        bo.CHANNEL_SLOTS_V4,
        bo.CHANNEL_SLOTS_SOBOL,
        *bo.ONE_PER_CHANNEL,
    )
    sobol_only = bool(slots) and slots["protocol"] == bo.CHANNEL_SLOTS_SOBOL
    if sobol_only or (
        modern and slots["protocol"] in bo.ONE_PER_CHANNEL and not recovery
    ):
        forced = False
    if sobol_only:
        agent, gap_front = [], None
        advice = {
            "protocol": bo.CHANNEL_SLOTS_SOBOL,
            "status": "skipped: sobol-only round",
        }
    else:
        agent, gap_front, advice = analyst(state, pts, forced, r, keys, panel, a)
    qf = None if modern else bo.qfirst_settings(state, r, list(pts.values()))
    bo.use_plans(
        plans
    )  # plan-p1rank-v1: right before proposing (bo_module() resets to every plan)
    log(f"round {r} proposable plans: {[space.DISPLAY[p] for p in plans]}")
    picks = bo.propose_batch(
        list(pts.values()),
        bo.context(DS, panel["panel_hash"]),
        k=k,
        seed=1000 * r,
        forced=forced,
        agent_rows=agent or None,
        realized_front=gap_front,
        qfirst=qf,
        slots=slots,
        aux=aux,
        search_options=sa.engine_options(search_settings),
        stall_recovery=recovery,
    )
    log_qfirst(r, picks)
    added = [p for p in plans if p not in space.BASE_PLANS]
    if added:
        hit = [
            f"{p['channel']}:{space.DISPLAY[p['config']['plan']]}"
            for p in picks
            if p["config"]["plan"] in added
        ]
        log(
            f"round {r} {space.PLAN_EXTENSION['protocol']}: picks with an added plan {added}: {hit or 'none'}; "
            f"picks by plan {[space.DISPLAY[p['config']['plan']] for p in picks]}"
        )
    if slots:
        log(
            f"round {r} {slots['protocol']}: {'FORCED' if forced else 'ordinary'}, "
            f"stall_recovery={search_settings.get('stall_recovery', sa.LEGACY)}, "
            f"selected slots {[p.get('slot', p['channel']) for p in picks]}"
        )
        for line in bo.slot_log_lines(r, picks):
            log(line)
        if slots["protocol"] in bo.CHANNEL_SLOTS_NAMED:
            for line in bo.coverage_log_lines(
                r, bo.round_coverage(picks, list(pts.values()))
            ):
                log(line)
    if advice.get("status") == "disabled" and any(
        "agent" in str(p.get(f, ""))
        for p in picks
        for f in ("channel", "source_channel")
    ):
        raise AssertionError(
            f"round {r}: analyst off but a pick carries an agent channel"
        )
    out = [
        {
            **p,
            "forced": forced,
            "analyst": advice,
            **({"stall_recovery": sa.STALL_RECOVERY} if recovery else {}),
        }
        for p in picks
    ]
    if gp_report:
        out[0]["gp"] = (
            gp_report  # once per round (the first pick), as recorded in the queue log
        )
    return out


def round0_sobol_only(bo, k: int) -> list[dict]:
    """explore-v6 round 0: the first ``k`` distinct Sobol breadth draws over the proposable plans (``session_pool
    .sobol_breadth``, seed 7 as ``round0``) that are memory configurations satisfying the proposal-side constraint
    (sim_gate none); NO plan defaults. Deterministic."""
    out, seen = [], set()
    for n in (256, 1024, 4096):
        for c in bo.pool.sobol_breadth(n, 7):
            if c["plan"] == "NoMemory" or not space.proposal_ok(c):
                continue
            key = space.key(c)
            if key not in seen:
                seen.add(key)
                out.append({"config": space.validate(c), "channel": "sobol"})
            if len(out) == k:
                return out
        out, seen = [], set()
    raise SystemExit(f"round 0: fewer than {k} distinct proposable Sobol draws")


def design_blocks(state: dict) -> None:
    """explore-v6 (pool-v6): every protocol block the design fixes must be recorded from round 0 -- hv-ref-anchored-v1,
    gp-fix-v1, quality reuse, concurrency, async staging (all opt-in env of run_tau2_v6.sh) -- and semfilter-v2
    (recorded here, ``semfilter`` with from_round 0, knobs ``tau2_design.SEMFILTER_V2_ENV``). Refuses otherwise."""
    from trimem.experiments.tau2 import sfv2_rerun

    if state.get("semfilter") is None:
        cfg = sfv2_rerun.want_settings()
        blk = sfv2_rerun.semfilter_block(cfg, [], "", datetime.now(UTC).isoformat())
        blk.pop("superseded")
        blk.update(
            reason="user 2026-10-23: τ² re-profile pool-v6 with semfilter-v2 (rubric v3 with the τ² CONFLICT clause) "
            "from round 0",
            before=None,
            scope="SemFilter plans (P1 / P1+Rank / P2 / P2+Rank) from round 0; no v1-judge episode in this run",
        )
        if blk["env"] != tau2_design.SEMFILTER_V2_ENV:
            raise SystemExit(
                f"explore-v6: semfilter-v2 knobs {blk['env']} != {tau2_design.SEMFILTER_V2_ENV}"
            )
        state["semfilter"] = blk
    want = {
        "semfilter": ("protocol", "semfilter-v2"),
        "hv_ref": ("protocol", "hv-ref-anchored-v1"),
        "gp": ("protocol", "gp-fix-v1"),
        "quality_reuse": ("protocol", qr.PROTOCOL),
        "concurrency": ("protocol", "concurrency-v1"),
    }
    bad = [
        f"{k} {(state.get(k) or {}).get(f)!r} (from round {(state.get(k) or {}).get('from_round')})"
        for k, (f, v) in want.items()
        if (state.get(k) or {}).get(f) != v
        or (state.get(k) or {}).get("from_round") != 0
    ]
    asy = state["staging"].get(ASYNC_KEY) or {}
    if asy.get("protocol") != ASYNC_PROTOCOL or asy.get("from_round") != 0:
        bad.append(f"staging.async {asy.get('protocol')!r}")
    if state.get("semfilter", {}).get("env") != tau2_design.SEMFILTER_V2_ENV:
        bad.append("semfilter env")
    if state.get("analyst"):
        bad.append("analyst block (the llm channel needs the analyst on)")
    if bad:
        raise SystemExit(
            f"explore-v6 needs these blocks from round 0: {bad} (use run_tau2_v6.sh)"
        )


def log_qfirst(r: int, picks: list[dict]) -> None:
    for p in picks:
        if p["channel"] == "qfirst":
            q = p["qfirst"]
            log(
                f"round {r} qfirst slot ({q['protocol']}, beta {q['beta']}, target {q['quality_target']}, budget "
                f"{q['budget_tokens']:,.0f} tok / {q['budget_latency_ms'] / 1000:.1f} s; via {p['source_channel']}): "
                f"{space.key(p['config'])} pred gain {p['pred_gain']:+.3f}, {p['pred_tokens']:,.0f} tok, "
                f"{p['pred_latency_ms'] / 1000:.1f} s"
            )


def analyst(
    state: dict, pts: dict, forced: bool, r: int, keys: list[str], panel: dict, a
) -> tuple[list, list | None, dict]:
    """LLM analyst channel (``analyzer_pool``; user 2026-10-20): TRAIN profiling evaluations (paired pass^1 gain,
    tokens, latency, staged n) -> at most 4 in-space edits labelled agent / agent-gap entering the candidate
    pool. Never fatal; ``--no_analyst`` (whole run) or the state.json ``analyst`` block (``ANALYST_OFF_ENV``, from
    a round on; user 2026-10-20: live -v5 run from round 5) disables it -- neither changes the run identity."""
    from trimem.optimizer import analyzer_pool as ap

    off = analyst_off(state, r, a)
    status = {
        space.key(c["config"]): {
            "n": staging_of(c)["stage_reached"],
            "n_full": len(keys),
            "pruned": inactive(c),
        }
        for rr in state["rounds"]
        for c in rr["configs"]
    }
    if off is not None:
        front = gap_front_without_llm(ap, pts, state, status, r) if forced else None
        log(
            f"analyst round {r}: OFF ({off['by']}"
            + (
                f", from round {off['off_from_round']}: {off['reason']}"
                if "off_from_round" in off
                else ""
            )
            + f") -> other channels only{'; forced gap rectangle computed without the LLM' if front else ''}"
        )
        return (
            [],
            front,
            {
                **off,
                "protocol": state.get("search_design", {}).get("protocol", ap.PROTOCOL),
                "status": "disabled",
            },
        )
    try:
        nm = nomemory_tasks(keys)
        nm_q = sum(t["quality"] for t in nm.values()) / len(nm)
    except SystemExit:
        nm_q = float("nan")
    ctx = {
        "dataset": "tau2-bench (airline / retail / telecom customer service)",
        "panel": f"{len(keys)} official TRAIN tasks, {PROFILE_TRIALS} trial each; staged {list(STAGE_SIZES)}",
        "quality_text": f"paired pass^{PROFILE_K} gain per task vs NoMemory on the same tasks (NoMemory pass^1 "
        f"{nm_q:.3f}), domain-stratified mean +- SE",
    }
    from trimem.optimizer import search_analyzer as sa

    if state.get("search_design", {}).get("protocol") == sa.PROTOCOL:
        rows, front, summary = sa.advise(
            space.proposal_view(),
            list(pts.values()),
            state,
            status=status,
            forced=forced,
            rnd=r,
            ctx=ctx,
            out_dir=OUT,
            settings=state["search_design"],
            plans=proposal_plans(state, r),
        )
        log(
            f"analyzer round {r}: {summary['status']}, {len(rows)} local starts ({summary['log']})"
        )
        return rows, front, summary
    conf = None
    try:
        from trimem.optimizer import analyzer_session as az
        from trimem.optimizer import session_pool as sp_pool

        conf = az.confidence_text(sp_pool.fit(list(pts.values()))[0])
    except Exception as e:  # noqa: BLE001 - optional card
        log(f"analyst round {r}: no confidence card ({type(e).__name__}: {e})")
    rows, front, summary = ap.advise(
        space.proposal_view(),  # sim_gate FIXED = none for proposals (user 2026-10-20)
        list(pts.values()),
        state,
        status=status,
        forced=forced,
        rnd=r,
        ctx=ctx,
        out_dir=OUT,
        confidence=conf,
    )
    log(
        f"analyst round {r} ({summary['label']}): {summary['status']}, {summary['candidates']} candidates, "
        f"{summary['rejects']} rejects{'' if rows else ' -> other channels only'} ({summary['log']})"
    )
    return rows, front, summary


def gap_front_without_llm(
    ap, pts: dict, state: dict, status: dict, r: int
) -> list | None:
    """Analyst off + forced (stalled) round: the same max-HV gap rectangle the analyst would use (``analyzer_pool
    .advise``: quality arm on odd rounds, cost on even), so the forced-gap channel (crossover rows ranked by
    landing probability) stays operative without agent rows."""
    from trimem.optimizer import analyzer_session as az

    try:
        rect = az.max_hv_rectangle(
            ap.evals_of(space.proposal_view(), list(pts.values()), state, status),
            "quality" if r % 2 else "cost",
        )
    except Exception as e:  # noqa: BLE001 - never fatal: ordinary round instead
        log(f"round {r}: no gap rectangle ({type(e).__name__}: {e})")
        return None
    return [tuple(e) for e in rect["endpoints"]] if rect else None


def run_round(
    state: dict, state_p: Path, r: int, keys: list[str], panel: dict, a
) -> None:
    """Every round, round 0 included (v2; only NoMemory is protected): synchronous nested stages -- every live
    configuration of the batch runs stage s (NoMemory fills the same prefix alongside until it is complete), then
    each is decided at the checkpoint (``decide_stage``); decisions are stored in state.json and reused on resume.
    Episodes already run outside a stage prefix (round 0 before the v2 migration) are kept; decisions use the
    prefix only."""
    blk = state["staging"].get(ASYNC_KEY)
    if blk and r >= blk["from_round"]:
        return run_round_async(state, state_p, r, keys, panel, a)
    rnd = state["rounds"][r]
    label = lambda j, c: f"r{r}c{j}:{space.DISPLAY[c['config']['plan']]}"  # noqa: E731
    subsets = stages_of(state)
    w = strata_weights(keys)
    for si, sub in enumerate(subsets):
        last = si == len(subsets) - 1
        live = [(j, c) for j, c in enumerate(rnd["configs"]) if not inactive(c)]
        todo = [(j, c) for j, c in live if staging_of(c)["stage_reached"] < len(sub)]
        nm_arm = Arm(
            "NoMemory",
            nomemory_dir(),
            "train",
            sub,
            None,
            full_split=len(sub) == len(keys),
            regime=regime_of(state, r),
        )
        arms = [] if nm_arm.done() >= nm_arm.expected else [nm_arm]
        arms += [
            config_arm(c["config"], label(j, c), sub, panel, state, r) for j, c in todo
        ]
        if not arms:
            continue
        log(
            f"round {r} stage {len(sub)}: {len(todo)} live configurations"
            + (" + NoMemory" if arms[0] is nm_arm else "")
            + f" ({', '.join(f'{x.name} {x.done()}/{x.expected}' for x in arms)})"
        )
        _run_or_stop(arms, a, f"round {r} stage {len(sub)}")
        for j, c in todo:
            rs = reuse_stats(c["config"], keys)
            staging_of(c).update(
                stage_reached=len(sub),
                episodes=files_present(config_dir(c["config"]), keys, "train")
                * PROFILE_TRIALS,
                reused=rs["hits"],
                reuse=rs,
            )
            if state.get("quality_reuse"):
                log(
                    f"quality reuse {label(j, c)}: {rs['hits']}/{rs['episodes']} reused "
                    f"(latency by regime {rs['latency_source_regime']})"
                )
        if todo and not last:
            decide_stage(state, r, todo, sub, keys, w, label)
        save(state_p, state)


def decide_stage(
    state: dict,
    r: int,
    todo: list[tuple[int, dict]],
    sub: list[str],
    keys: list[str],
    w: dict[str, float],
    label,
    mode: str = "sync",
) -> None:
    """Apply the frozen policy to configurations that completed stage ``sub``.

    Quality is paired gain versus NoMemory. Only the default staged policy uses
    partial-stage pessimistic comparators and the plan guard.
    """
    nm_sub = per_task(
        arm_episodes(nomemory_dir(), sub, split="train", cfg=None), sub, "train"
    )
    if nm_sub is None:
        raise SystemExit(f"NoMemory stage {len(sub)} incomplete")
    full = full_objectives(state, keys, w)
    stage = stage_comparators(state, sub, nm_sub, w, set(full))
    policy = pp_policy.selected(state["staging"])
    if policy == "none":
        full = {}
    objs, decs = {}, {}
    for _, c in todo:
        k = space.key(c["config"])
        objs[k] = stage[k]
        decs[k] = pp_policy.decide(
            objs[k],
            full,
            policy=policy,
            stage=len(sub),
            last=False,
            partial={j: o for j, o in stage.items() if j != k}
            if policy == "staged"
            else {},
        )
    now = {space.key(c["config"]) for _, c in todo}
    plan_of = {
        space.key(c["config"]): c["config"]["plan"]
        for rr in state["rounds"]
        for c in rr["configs"]
    }
    covered = {
        plan_of[space.key(c["config"])]
        for rr in state["rounds"]
        for c in rr["configs"]
        if space.key(c["config"]) not in now and not inactive(c)
    }
    decs = pp_policy.guard(decs, objs, plan_of, covered, policy=policy)
    if policy != "staged":
        stage = {k: stage[k] for k in now}
    short = lambda dk: (  # noqa: E731
        "NoMemory"
        if dk not in plan_of
        else f"{space.DISPLAY[plan_of[dk]]} {cfg_short(dk)}"
    )
    for j, c in todo:
        k = space.key(c["config"])
        d = decs[k]
        staging_of(c)["decisions"].append(
            {
                **d.as_dict(),
                "pruning_policy": policy,
                "n_full": len(full),
                "n_stage_comparators": len(stage) - 1 if policy == "staged" else 0,
                "mode": mode,
                # the comparator set of THIS decision (async: whatever exists at that moment)
                "comparators": {
                    "full": sorted(cfg_short(dk) for dk in full),
                    "stage": sorted(
                        short(dk)
                        for dk in stage
                        if dk != space.key(c["config"]) and policy == "staged"
                    ),
                },
                "objectives": {
                    "gain": objs[k].quality,
                    "gain_se": objs[k].quality_se,
                    "tokens": objs[k].tokens,
                    "tokens_se": objs[k].tokens_se,
                    "latency_ms": objs[k].latency,
                    "latency_ms_se": objs[k].latency_se,
                    "n": objs[k].n,
                },
                "decided_at": datetime.now(UTC).isoformat(),
            }
        )
        fmt = lambda t: f"({t[0]:+.3f}, {t[1]:,.0f} tok, {t[2] / 1000:.1f} s)"  # noqa: E731
        doms = "; ".join(
            ("NoMemory" if dk not in plan_of else f"{plan_of[dk]} {cfg_short(dk)}")
            + f" [{kind}] {fmt(dc)}"
            for dk, kind, dc in d.dominators
        )
        stage_names = (
            sorted(short(dk) for dk in stage if dk != k) if policy == "staged" else []
        )
        log(
            f"round {r} [{mode}] comparators of {label(j, c)} at {len(sub)}: {len(full)} full + "
            f"{len(stage_names)} stage ({', '.join(stage_names)})"
        )
        if d.prune:
            staging_of(c).update(
                pruned_at=len(sub), reason=d.reason, dominated_by=d.dominated_by
            )
            log(
                f"round {r}: PRUNED {label(j, c)} at {len(sub)} tasks: corner {fmt(d.corner)} dominated by {doms}"
            )
        else:
            log(
                f"round {r}: KEEP {label(j, c)} at {len(sub)} tasks: corner {fmt(d.corner)} -- {d.reason}"
                + (f" (dominators: {doms})" if d.dominators else "")
            )


# ------------------------------------------------------------------ asynchronous staging (user 2026-10-20)

# staged-async-v1: each live configuration advances through its own nested stages independently; as soon as its
# stage subset is complete it is decided (``decide_stage``, the same rule: optimistic corner vs the pessimistic
# corners of NoMemory and of the configurations evaluated at >= that stage, plus the point estimates of the fully
# evaluated ones; z = 1.96; plan guard) against WHATEVER comparators exist at that moment (logged per decision), and
# a kept configuration immediately starts its next stage. ~``concurrency`` episodes in total, split evenly over the
# running arms; arms whose share would at least double are drained (in-flight episodes finish) and relaunched. A
# round ends when every configuration is pruned or complete (rounds never overlap). Opt-in via ``ASYNC_ENV``,
# recorded once in state.json ``staging.async`` (NOT part of the run identity; ``load_state`` ignores it); decisions
# made before stay as they are (round 6 of the live -v5 run: stage 50 synchronous, stages 100 / 178 asynchronous).
ASYNC_KEY = "async"
ASYNC_PROTOCOL = "staged-async-v1"
ASYNC_ENV = "TAU2_ASYNC_STAGING"
ASYNC_POLL_S = 30


def round_open(state: dict, rr: dict, keys: list[str]) -> bool:
    """A configuration of the round is live (not pruned / stopped) and not yet fully evaluated."""
    return any(
        not inactive(c) and staging_of(c)["stage_reached"] < len(keys)
        for c in rr["configs"]
    )


def async_block(state: dict, keys: list[str]) -> dict | None:
    """Record ``staging.async`` once when ``ASYNC_ENV`` is set (from the round currently running, else the next
    one); a recorded block persists (never silently back to synchronous)."""
    g = state["staging"]
    blk = g.get(ASYNC_KEY)
    if blk is not None:
        if blk.get("protocol") != ASYNC_PROTOCOL:
            raise SystemExit(
                f"state.json staging.async {blk} != {ASYNC_PROTOCOL}; refusing to mix"
            )
        return blk
    if os.environ.get(ASYNC_ENV, "").strip() in ("", "0"):
        return None
    frm = next(
        (rr["round"] for rr in state["rounds"] if round_open(state, rr, keys)),
        len(state["rounds"]),
    )
    g[ASYNC_KEY] = blk = {
        "protocol": ASYNC_PROTOCOL,
        "from_round": frm,
        "at": datetime.now(UTC).isoformat(),
        "reason": "user 2026-10-20: asynchronous staged pruning, effective immediately",
        "rule": "unchanged (staged-pareto-prune-v2, z = 1.96, plan guard); each configuration is decided as soon "
        "as its own stage subset completes, against the comparators that exist at that moment (logged in each "
        "decision: decisions[].comparators, mode = async)",
        "concurrency": "~profiling concurrency in total, split evenly over the running arms; drained and "
        "relaunched when an arm's share would at least double (in-flight episodes finish)",
        "note": "decisions made before stay as they are (mixed sync / async within from_round is expected); "
        "budget accounting unchanged; rounds never overlap",
    }
    if pp_policy.selected(g) != "staged":
        blk["reason"] = "isolated run with frozen alternative pruning policy"
        blk["rule"] = (
            g["pruning_policy"]["description"]
            + "; each configuration is decided when its own checkpoint completes"
        )
    return blk


def next_stage(state: dict, c: dict) -> list[str] | None:
    """The configuration's next stage subset (None = fully evaluated)."""
    reached = staging_of(c)["stage_reached"]
    return next((sub for sub in stages_of(state) if len(sub) > reached), None)


def record_stage(
    state: dict, c: dict, sub: list[str], keys: list[str], name: str
) -> None:
    rs = reuse_stats(c["config"], keys)
    staging_of(c).update(
        stage_reached=len(sub),
        episodes=files_present(config_dir(c["config"]), keys, "train") * PROFILE_TRIALS,
        reused=rs["hits"],
        reuse=rs,
    )
    if state.get("quality_reuse"):
        log(
            f"quality reuse {name}: {rs['hits']}/{rs['episodes']} reused "
            f"(latency by regime {rs['latency_source_regime']})"
        )


def advance_async(
    state: dict, r: int, keys: list[str], panel: dict, w: dict, label, running: set[str]
) -> list[str]:
    """Record + decide every configuration of round ``r`` whose current stage subset is complete on disk (and
    whose NoMemory prefix is complete); repeats until nothing changes (resume: several stages may be complete).
    Returns the names whose stage was recorded."""
    rnd, done, changed = state["rounds"][r], [], True
    while changed:
        changed = False
        for j, c in enumerate(rnd["configs"]):
            if inactive(c) or label(j, c) in running:
                continue
            sub = next_stage(state, c)
            if sub is None:
                continue
            arm = config_arm(c["config"], label(j, c), sub, panel, state, r)
            if arm.done() < arm.expected:
                continue
            last = len(sub) == len(keys)
            if (
                not last
                and nomemory_arm(state, r, sub, keys).done() < len(sub) * PROFILE_TRIALS
            ):
                continue  # decided once NoMemory has the prefix (its arm runs)
            record_stage(state, c, sub, keys, label(j, c))
            if last:
                log(f"round {r} [async]: COMPLETE {label(j, c)} at {len(sub)} tasks")
            else:
                decide_stage(state, r, [(j, c)], sub, keys, w, label, mode="async")
            done.append(label(j, c))
            changed = True
    return done


def nomemory_arm(state: dict, r: int, sub: list[str], keys: list[str]) -> Arm:
    return Arm(
        "NoMemory",
        nomemory_dir(),
        "train",
        sub,
        None,
        full_split=len(sub) == len(keys),
        regime=regime_of(state, r),
    )


def wanted_arms(
    state: dict, r: int, keys: list[str], panel: dict, label, running: set[str]
) -> list:
    """Arms that should run now and do not: every live configuration's current stage (incomplete on disk) and
    NoMemory on the largest stage subset any live configuration is at (when incomplete)."""
    out, need = [], None
    for j, c in enumerate(state["rounds"][r]["configs"]):
        if inactive(c):
            continue
        sub = next_stage(state, c)
        if sub is None:
            continue
        if need is None or len(sub) > len(need):
            need = sub
        arm = config_arm(c["config"], label(j, c), sub, panel, state, r)
        if arm.name not in running and arm.done() < arm.expected:
            out.append(arm)
    if need is not None:
        nm = nomemory_arm(state, r, need, keys)
        if "NoMemory" not in running and nm.done() < nm.expected:
            out.insert(0, nm)
    return out


def launch(arm: Arm, workers: int, n_arms: int, total: int) -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT), **RUN_ENV}
    arm.out.mkdir(parents=True, exist_ok=True)
    drain_path(arm).unlink(missing_ok=True)
    cmd = [
        *runner_cmd(arm, workers, n_arms, total),
        "--drain_file",
        str(drain_path(arm)),
    ]
    (arm.out / ".command").write_text(json.dumps(cmd))
    logf = open(arm.out / "runner.log", "a")
    p = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"arm": arm, "p": p, "logf": logf, "workers": workers, "drained": False,
            "progress": (arm.done(), time.time()), "offset": log_offset(arm)}  # fmt: skip


def run_round_async(
    state: dict, state_p: Path, r: int, keys: list[str], panel: dict, a
) -> None:
    """staged-async-v1 event loop for round ``r`` (see ``ASYNC_PROTOCOL``). Resumable: stage completion and
    decisions are derived from the result files + state.json; a restart relaunches each arm on its current stage."""
    label = lambda j, c: f"r{r}c{j}:{space.DISPLAY[c['config']['plan']]}"  # noqa: E731
    w = strata_weights(keys)
    total = a.concurrency
    procs: dict[str, dict] = {}
    fails: dict[str, int] = {}
    hit_t = time.time()
    log(
        f"round {r} [async {ASYNC_PROTOCOL}]: ~{total} episodes in total over the running arms"
    )
    try:
        while True:
            for name in list(procs):  # reap
                pr = procs[name]
                if pr["p"].poll() is None:
                    continue
                pr["logf"].close()
                drain_path(pr["arm"]).unlink(missing_ok=True)
                arm = pr["arm"]
                log(
                    f"  {name}: exit {pr['p'].returncode} ({arm.done()}/{arm.expected} of stage {len(arm.keys)})"
                )
                del procs[name]
                if arm.done() < arm.expected and not pr["drained"]:
                    fails[name] = fails.get(name, 0) + 1
                    if fails[name] >= ATTEMPTS:
                        raise SystemExit(
                            f"round {r} [async]: {name} incomplete after {ATTEMPTS} failed attempts; resume to retry"
                        )
            if advance_async(state, r, keys, panel, w, label, set(procs)):
                save(state_p, state)
            want = wanted_arms(state, r, keys, panel, label, set(procs))
            n = len(procs) + len(want)
            if n == 0:
                break
            per = max(1, total // n)
            for arm in want:
                procs[arm.name] = launch(arm, per, n, total)
                log(
                    f"round {r} [async]: start {arm.name} stage {len(arm.keys)} at {arm.done()}/{arm.expected} "
                    f"with {per} workers ({n} arms)"
                )
            for (
                name,
                pr,
            ) in (
                procs.items()
            ):  # rebalance: freed concurrency goes to the arms still running
                if not pr["drained"] and per >= 2 * pr["workers"]:
                    log(
                        f"round {r} [async] rebalance: draining {name} ({pr['workers']} workers; in-flight episodes "
                        f"finish) to relaunch at {per}"
                    )
                    drain_path(pr["arm"]).touch()
                    pr["drained"] = True
            now = time.time()
            for name, pr in procs.items():  # stall watchdog
                d = pr["arm"].done()
                if d != pr["progress"][0]:
                    pr["progress"] = (d, now)
                elif now - pr["progress"][1] > STALL_S and pr["p"].poll() is None:
                    log(
                        f"stall: {name} no new episode for {STALL_S}s at {d}/{pr['arm'].expected}; stopping that arm"
                    )
                    kill_group(pr["p"])
            if now - hit_t >= HIT_LOG_S:
                hit_t = now
                log_hits(
                    {k: (v["arm"], v["p"], v["logf"]) for k, v in procs.items()},
                    {k: v["offset"] for k, v in procs.items()},
                    "progress",
                )
            time.sleep(ASYNC_POLL_S)
    finally:
        for name, pr in procs.items():
            kill_group(pr["p"])
            pr["p"].wait()
            pr["logf"].close()
            drain_path(pr["arm"]).unlink(missing_ok=True)
            log(f"  {name}: exit {pr['p'].returncode} (stopped with the driver)")
        save(state_p, state)
    log(f"round {r} [async]: every configuration pruned or complete")


def cfg_short(key: str) -> str:
    return "cfg_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _run_or_stop(arms: list[Arm], a, what: str) -> None:
    res = run_parallel(arms, a.concurrency)
    bad = [x.name for x in arms if res[x.name] < x.expected]
    if bad:
        raise SystemExit(
            f"{what}: {bad} incomplete after {ATTEMPTS} failed attempts; resume to retry"
        )


def pause_after_round(a) -> int | None:
    v = getattr(a, "pause_after_round", None)
    if v is None and os.environ.get(PAUSE_ENV, "").strip():
        v = int(os.environ[PAUSE_ENV])
    return v


def resume_from_pause(state: dict, pause: int | None) -> dict | None:
    """profiling-pause-v1: when profiling continues past a user pause (no pause flag, or a later round), archive the
    paused marker and its derived artifacts so the final router / select / test stages recompute; record it."""
    p = OUT / "profile_complete.json"
    if not p.exists():
        return None
    done = json.loads(p.read_text())
    if not done.get("paused_by_user") or (
        pause is not None and pause <= done["last_round"]
    ):
        return None
    at = datetime.now(UTC).isoformat()
    dest = OUT / f"paused_after_round{done['last_round']}_{at[:19].replace(':', '')}"
    moved = []
    for rel in PAUSED_ARTIFACTS:
        src = OUT / rel
        if src.exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest / rel)
            moved.append(rel)
    blk = state.setdefault("profiling_pause", {"protocol": PAUSE_PROTOCOL})
    blk.setdefault("history", []).append(
        {
            "paused_after_round": done["last_round"],
            "resumed_at": at,
            "archive": str(dest),
            "moved": moved,
        }
    )
    blk.update(status="resumed", resumed_at=at, archive=str(dest))
    return blk


def profile(a) -> None:
    """Round 0 (4 plan defaults + 6 Sobol; NoMemory alongside, never pruned), then BO rounds while the unpruned
    plan's episode budget allows (``staged_pruning.next_batch_size``; at most ``max_configs``); every round staged
    (v2), so round-0 savings go into the BO budget."""
    panel = load_panel()
    keys = train_keys(panel)
    state, state_p = load_state(panel, a)
    from trimem.optimizer import search_baselines

    baseline = search_baselines.active(state["search_design"])
    proposal_block(state)
    if not baseline:
        ext = extension_block(state)
        log(
            f"space extension {ext['protocol']} from round {ext['from_round']}: + {ext['added_plans']} "
            f"({', '.join(ext['display'].values())}); base plans {ext['base_plans']}"
        )
    if not baseline:
        bo_module().qfirst_block(state, datetime.now(UTC).isoformat())
    bo_module().gp_block(state, datetime.now(UTC).isoformat(), GP_NOISE)
    if not baseline and DESIGN is not None:
        # explore-v6 (pool-v6, user 2026-10-23): sobol-only-v1 round 1, channel-slots-v6 from round 2
        bo = bo_module()
        cs = bo.channel_slots_block(
            state,
            datetime.now(UTC).isoformat(),
            protocol=bo.CHANNEL_SLOTS_V6,
            sobol_until=tau2_design.RANDOM_UNTIL_ROUND,
        )
        if cs["protocol"] != bo.CHANNEL_SLOTS_V6 or bo.recorded_sobol_range(state) != (
            1,
            tau2_design.RANDOM_UNTIL_ROUND,
        ):
            raise SystemExit(
                f"explore-v6: unexpected channel-slot chain {cs['protocol']}"
            )
        log(
            f"channel slots: round 0 {tau2_design.ROUND0_SOBOL} Sobol (no plan defaults), sobol-only-v1 round "
            f"1..{tau2_design.RANDOM_UNTIL_ROUND}, {cs['protocol']} from round {cs['from_round']}: {cs['slot_order']}"
        )
    elif not baseline:
        cs = bo_module().channel_slots_block(state, datetime.now(UTC).isoformat())
        assert cs["protocol"] == CHANNEL_SLOTS_TAG
        log(
            f"channel slots {cs['protocol']} from round {cs['from_round']}: {cs['slot_order']} (crossover top "
            f"{cs['crossover']['top']}, grafts {cs['crossover']['graft_sizes']}, <= {cs['crossover']['max_total']} rows)"
        )
    ax = (
        None
        if baseline
        else bo_module().aux_block(
            state,
            datetime.now(UTC).isoformat(),
            aux_obs.build(DS, space),
            min_round=1,
        )
    )
    if ax:
        log(
            f"auxiliary GP observations {ax['protocol']} from round {ax['from_round']}: {len(ax['points'])} points "
            f"from {[s['experiment'] for s in ax['sources']]} ({ax['noise_model']})"
        )
    # HV reference point hv-ref-anchored-v1 (user 2026-10-22): opt-in POOL_HV_REF=anchored-v1, from the next round
    if hv := bo_module().hv_ref_block(state, datetime.now(UTC).isoformat()):
        log(
            f"HV reference {hv['protocol']} from round {hv['from_round']} (state.json hv_ref; before: "
            f"{hv['before']})"
        )
    if blk := analyst_block(state):
        log(
            f"analyst channel OFF from round {blk['off_from_round']} ({blk['reason']}; state.json analyst)"
        )
    if blk := concurrency_block(state, a):
        log(
            f"concurrency {blk['protocol']} from round {blk['from_round']}: profiling {blk['profiling_total']} "
            f"({blk['profiling_regime']}), test {blk['test_total']}; latency caveat recorded in state.json"
        )
    if blk := reuse_block(state):
        log(
            f"quality reuse {blk['protocol']} from round {blk['from_round']} (regime in key: "
            f"{blk['regime_in_key']}); index {blk['index']} (backfilled {blk['backfill']['episodes']} episodes)"
        )
    if blk := async_block(state, keys):
        log(
            f"staging {blk['protocol']} from round {blk['from_round']} (recorded {blk['at']}; state.json "
            "staging.async)"
        )
    stopped = stop_violating(state, keys)
    if stopped:
        log(
            f"stopped by the user ({STOP_REASON}): {len(stopped)} configurations: {stopped}"
        )
    if DESIGN is not None:
        design_blocks(state)
    save(state_p, state)
    if DESIGN is not None:
        # semfilter-v2 from round 0: export the recorded knobs before any runner starts (main() could not: the
        # state.json of a fresh run did not exist yet)
        from trimem.experiments.tau2 import sfv2_rerun

        if not sfv2_rerun.apply_semfilter(OUT):
            raise SystemExit("explore-v6: semfilter-v2 env not exported")
    budget = budget_of(state)
    if ext := state.get("budget_extension"):
        log(
            f"budget extension ({ext['reason']}): {ext['from']} -> {ext['to']} episodes, at most "
            f"{ext['target_evaluated_configs']} evaluated configurations (now {counted_configs(state)})"
        )
    pause = pause_after_round(a)
    if blk := resume_from_pause(state, pause):
        save(state_p, state)
        log(
            f"profiling RESUMED after the user pause ({PAUSE_PROTOCOL}); paused artifacts archived in {blk['archive']}"
        )
    r = 0
    while True:
        if pause is not None and r > pause:
            if len(state["rounds"]) > r:
                raise SystemExit(
                    f"pause after round {pause}: round {r} is already in state.json; remove it first"
                )
            log(
                f"profiling PAUSED by the user after round {pause} ({PAUSE_PROTOCOL}); round {r} not proposed"
            )
            break
        if len(state["rounds"]) <= r:
            k = (
                10
                if r == 0
                else sp.next_batch_size(
                    budget,
                    episodes_used(state),
                    counted_configs(state),
                    a.batch if DESIGN is None else tau2_design.batch_for(r, a.batch),
                )
            )
            if k == 0:
                break
            picks = plan_round(state, r, keys, panel, a, k)
            state["rounds"].append(
                {
                    "round": r,
                    "proposed_at": datetime.now(UTC).isoformat(),
                    "configs": picks,
                    "budget_used_before": episodes_used(state),
                }
            )
            save(state_p, state)
            off = r > 0 and picks[0].get("analyst", {}).get("status") == "disabled"
            log(
                f"round {r} ({k} configurations; budget used {episodes_used(state)}/{budget.total_episodes}"
                + ("; analyst OFF" if off else "")
                + "): "
                + "; ".join(f"[{p['channel']}] {space.key(p['config'])}" for p in picks)
            )
        run_round(state, state_p, r, keys, panel, a)
        r += 1
    full = config_tasks(state, keys)
    n_cfg = sum(len(rr["configs"]) for rr in state["rounds"])
    pruned = [  # pruned + stopped by the user
        space.key(c["config"])
        for rr in state["rounds"]
        for c in rr["configs"]
        if inactive(c)
    ]
    if len(full) + len(pruned) != n_cfg:
        raise SystemExit(
            f"profiling incomplete: {len(full)} full + {len(pruned)} pruned of {n_cfg}"
        )
    used = episodes_used(state)
    paused = {}
    if pause is not None:
        at = datetime.now(UTC).isoformat()
        reason = getattr(a, "pause_reason", None) or "user pause"
        paused = {
            "paused_by_user": True,
            "pause_protocol": PAUSE_PROTOCOL,
            "pause_reason": reason,
            "last_round": pause,
            "rounds": list(range(len(state["rounds"]))),
            "budget_used": used,
            "resume": "rerun `pool_profile profile` (or `all`) without --pause_after_round / TAU2_PAUSE_AFTER_ROUND: "
            "this marker and its derived artifacts are archived and BO continues at round "
            f"{pause + 1}",
        }
        blk = state.setdefault("profiling_pause", {"protocol": PAUSE_PROTOCOL})
        blk.update(
            status="paused",
            paused_at=at,
            paused_after_round=pause,
            reason=reason,
            rounds=paused["rounds"],
            configs=n_cfg,
            full_configs=len(full),
            pruned_configs=len(pruned),
            budget_used=used,
            budget_episodes=budget.total_episodes,
            resume=paused["resume"],
        )
        save(state_p, state)
    save(
        OUT / "profile_complete.json",
        {
            **paused,
            "completed_at": datetime.now(UTC).isoformat(),
            "experiment": state["experiment"],
            "panel_hash": panel["panel_hash"],
            "space_version": space.VERSION,
            "staging": STAGED_PROTOCOL,
            "configs": n_cfg,
            "full_configs": len(full),
            "pruned_configs": len(pruned),  # incl. stopped by the user
            "stopped_by_user": sum(
                bool(staging_of(c).get("stopped_by_user"))
                for rr in state["rounds"]
                for c in rr["configs"]
            ),
            "episodes_profiling": used,
            "episodes_nomemory": len(keys) * PROFILE_TRIALS,
            "evaluated_configs": n_cfg
            - sum(stopped_by_user(c) for rr in state["rounds"] for c in rr["configs"]),
            "budget_episodes": budget.total_episodes,
            "budget_extension": state.get("budget_extension"),
            "channel_slots": state.get("channel_slots"),
            "aux_observations": {
                k: v
                for k, v in (state.get("aux_observations") or {}).items()
                if k != "points"
            },
            "budget_unused": budget.total_episodes - used,
            "per_config": [
                {"key": space.key(c["config"]), "round": rr["round"], **staging_of(c)}
                for rr in state["rounds"]
                for c in rr["configs"]
            ],
        },
    )
    log(
        ("PROFILING PAUSED" if paused else "PROFILING DONE")
        + f" τ²: {n_cfg} configurations ({len(full)} full, {len(pruned)} pruned), {used} / "
        f"{budget.total_episodes} budget episodes + NoMemory {len(keys) * PROFILE_TRIALS}"
    )


def profile_table(a) -> dict:
    """Per FULLY evaluated configuration (pruned ones are excluded from selection and the router): pass^4 ± SE,
    paired gain vs NoMemory, tokens, latency (s), pass^1..4, pass@4."""
    panel = load_panel()
    keys = train_keys(panel)
    state, _ = load_state(panel, a)
    if not (OUT / "profile_complete.json").exists():
        raise SystemExit("profiling is not complete; no selection")
    w = strata_weights(keys)
    nm = nomemory_tasks(keys)
    tasks = config_tasks(state, keys)
    pts = []
    for cfg, ch, r, j in profiled(state):
        k = space.key(cfg)
        if k not in tasks:  # pruned
            continue
        g = paired_gain(tasks[k], nm, keys, w)
        pts.append(
            {
                "label": f"r{r}c{j}:{space.DISPLAY[cfg['plan']]}",
                "key": k,
                "cfg": space.validate(cfg),
                "round": r,
                "channel": ch,
                **summary_of(tasks[k], keys, w),
                "gain": g.mean,
                "gain_se": g.se,
                "fixed": k == space.key({"plan": cfg["plan"], **space.plan_defaults()}),
            }
        )
    T = {
        "panel": panel,
        "keys": keys,
        "weights": w,
        "state": state,
        "points": pts,
        "tasks": tasks,
        "nomemory_tasks": nm,
        "nomemory": summary_of(nm, keys, w),
    }
    save(
        OUT / "profile.json",
        {
            "generated": datetime.now(UTC).isoformat(),
            "experiment": state["experiment"],
            "quality": f"pass^{PROFILE_K} ({PROFILE_TRIALS} trial per task)",
            "staging": STAGED_PROTOCOL,
            "nomemory": T["nomemory"],
            "points": pts,
            "pruned": [
                {"key": space.key(c["config"]), "round": rr["round"], **staging_of(c)}
                for rr in state["rounds"]
                for c in rr["configs"]
                if inactive(c)
            ],
        },
    )
    return T


# ------------------------------------------------------------------ router (Opt-PerQuery)


def alias(keys: list[str]) -> dict[str, str]:
    """τ² key -> ``tau2_<i>`` (``sparse.SparseProfile.queries`` orders ids as ``<prefix>_<int>``)."""
    return {k: f"tau2_{i}" for i, k in enumerate(keys)}


def sparse_estimates(T: dict, nm_candidate: bool) -> dict[str, dict]:
    """``config_estimates``-style input of ``sparse.from_estimates`` (aliased ids): every configuration on all
    178 tasks, quality = the pass^4 indicator, cost = per-task mean tokens / latency; + NoMemory candidate."""
    al = alias(T["keys"])
    ids = [al[k] for k in T["keys"]]

    def entry(cfg: dict, tasks: dict) -> dict:
        return {
            "cfg": space.validate(cfg),
            "ids": ids,
            "per_query": {
                "quality": {al[k]: tasks[k]["quality"] for k in T["keys"]},
                "total_tokens": {al[k]: tasks[k]["tokens"] for k in T["keys"]},
                "e2e_ms": {al[k]: tasks[k]["latency_ms"] for k in T["keys"]},
                "mean_reward": {al[k]: tasks[k]["mean_reward"] for k in T["keys"]},
            },
        }

    est = {p["key"]: entry(p["cfg"], T["tasks"][p["key"]]) for p in T["points"]}
    if nm_candidate:
        est[space.key(space.NOMEM)] = entry(space.NOMEM, T["nomemory_tasks"])
    return est


def feature_records(keys: list[str], cache: Path) -> list[dict]:
    """Raw router records of τ² store tasks (structured + probe + embedding; no LLM), cached by key."""
    from trimem.experiments import query_router_newenv as qrn

    have = (
        {r["query_id"]: r for r in json.loads(cache.read_text())["records"]}
        if cache.exists()
        else {}
    )
    todo = [k for k in keys if k not in have]
    if todo:
        rows = qrn.task_rows(DS, todo)
        for r in qrn.records_for(DS, todo, rows, use_probe=True):
            have[r["query_id"]] = r
        save(
            cache,
            {
                "records": list(have.values()),
                "source": "query_router_newenv.records_for (features_newenv)",
            },
        )
    return [have[k] for k in keys]


def router_stage(a) -> None:
    import numpy as np

    from trimem.experiments import slo_select_bo as slo
    from trimem.optimizer.query_router import features, runtime, sparse

    T = profile_table(a)
    out = OUT / "router"
    out.mkdir(parents=True, exist_ok=True)
    al = alias(T["keys"])
    strata = {al[k]: domain_of(k) for k in T["keys"]}
    est = sparse_estimates(T, a.nomemory_candidate)
    prof = sparse.from_estimates(
        est, strata, T["weights"], {}
    )  # {} = no finite-population correction
    recs = [
        {**r, "query_id": al[r["query_id"]], "store_key": r["query_id"]}
        for r in feature_records(T["keys"], out / "features_train.json")
    ]
    from trimem.experiments.shared_router import save_snapshot

    save_snapshot(
        out / "training_snapshot.json",
        prof,
        recs,
        dataset=DS,
        source={
            "experiment": T["state"]["experiment"],
            "panel_hash": T["panel"]["panel_hash"],
        },
    )
    feature_records(
        test_keys(), out / "features_test.json"
    )  # deployment check + cache for the test routing
    groups = tuple(a.groups.split(","))
    spec = features.fit_spec(recs, groups)
    t0 = time.perf_counter()
    X = {r["query_id"]: runtime.transform(spec, r) for r in recs}
    tr_ms = (time.perf_counter() - t0) * 1000 / len(recs)
    log(
        f"router: {len(prof.configs)} configurations (NoMemory candidate {a.nomemory_candidate}), "
        f"{len(prof.queries)} tasks, features {spec['names'][:4]}..."
    )
    model, oof = sparse.fit(X, prof, trials=a.trials, n_jobs=a.jobs, seed=a.seed)
    log(
        f"router: {len(model['configs'])} candidates, predictors {[s.get('family') for s in model['selection']]}"
    )
    ovh = {
        r["query_id"]: (r["overhead"]["probe_ms"] if "probe" in groups else 0.0) + tr_ms
        for r in recs
    }
    B = slo.bounds(DS, T["points"])
    one = {
        n: slo.choose(T["points"], B[n], slo.SLOS[n]["objective"]) for n in SLO_NAMES
    }
    best_q = max(T["points"], key=lambda p: (p["q"], -p["lat"]))
    trained_on = {
        "experiment": T["state"]["experiment"],
        "panel_hash": T["panel"]["panel_hash"],
        "queries": len(al),
        "configs_profiled": len(prof.configs),
        "memory_policy": mp.SUCCESS_FAILURE,
        "groups": list(groups),
        "label": f"pass^{PROFILE_K} indicator per task ({PROFILE_TRIALS} train trial)",
        "query_alias": "tau2_<panel index>",
    }
    grid = sparse.weight_grid()
    sl = sparse.slices()
    extra = [w for ws in sl.values() for w in ws if w not in set(grid)]
    rows_all = sparse.weight_curve(
        prof, model, X, oof, grid + list(dict.fromkeys(extra)), overhead_ms=ovh
    )
    by_w = {sparse.weights_of(r): r for r in rows_all}
    rows = [by_w[w] for w in grid]
    curves = {
        "weight_grid": rows,
        "slices": {k: [by_w[w] for w in ws] for k, ws in sl.items()},
    }
    arts = {}
    for n in SLO_NAMES:
        obj = slo.SLOS[n]["objective"]
        pick = sparse.choose_weights(rows, B[n], obj)
        if pick is None:
            arts[n] = None
            near = sparse.nearest_infeasible(rows, B[n])
            log(
                f"router {n}: no weight vector satisfies the SLO on the replay estimate (nearest "
                f"{sparse.weights_of(near) if near else None})"
            )
            continue
        robust = sparse.choose_weights_robust(rows, B[n], obj)
        wts = sparse.weights_of(pick)
        art = sparse.artifact_weights(
            dataset=DS,
            spec=spec,
            model=model,
            weights=wts,
            fallback=(one[n] or best_q)["cfg"],
            trained_on={**trained_on, "trained_at": datetime.now(UTC).isoformat()},
            report={
                "slo": n,
                "bound": B[n],
                "share": pick["share"],
                "replay": {k: v for k, v in pick.items() if k != "share"},
            },
        )
        sha = runtime.artifact_sha(art)
        p = out / f"router_{sha}.json"
        if not p.exists():
            runtime.save(p, art)
        arts[n] = {
            "path": str(p),
            "sha": sha,
            "weights": list(wts),
            "replay_q": pick["q"],
            "replay_q_se": pick["quality_se"],
            "replay_lat_s": pick["lat"],
            "replay_tok": pick["tok"],
            "direct_q": pick["direct_quality"],
            "slack": sparse.slack(pick, B[n]),
            "n_feasible": sum(sparse.feasible(r, B[n]) for r in rows),
            "robust_1se": None
            if robust is None
            else {
                "weights": list(sparse.weights_of(robust)),
                "replay_q": robust["q"],
                "replay_lat_s": robust["lat"],
                "replay_tok": robust["tok"],
                "slack": sparse.slack(robust, B[n]),
            },
        }
        log(
            f"router {n}: weights {wts} -> replay pass^{PROFILE_K} {pick['q']:.3f} ± {pick['quality_se']:.3f}, "
            f"lat {pick['lat']:.1f}s, tok {pick['tok']:,.0f} ({sha})"
        )
    cart = sparse.artifact_weights(
        dataset=DS,
        spec=spec,
        model=model,
        weights=sparse.DEFAULT_WEIGHTS,
        fallback=best_q["cfg"],
        trained_on=trained_on,
        report={"slices": {k: [list(w) for w in ws] for k, ws in sl.items()}},
    )
    csha = runtime.artifact_sha(cart)
    cpath = out / f"router_{csha}.json"
    if not cpath.exists():
        runtime.save(cpath, cart)
    slice_router = {
        "path": str(cpath),
        "sha": csha,
        "default_weights": list(sparse.DEFAULT_WEIGHTS),
        "slices": {k: [list(w) for w in ws] for k, ws in sl.items()},
    }
    save(
        out / "router_report.json",
        {
            "generated": datetime.now(UTC).isoformat(),
            "experiment": T["state"]["experiment"],
            "protocol": sparse.PROTOCOL,
            "objective": f"w_Q pass^{PROFILE_K} (train, {PROFILE_TRIALS} trial); "
            + sparse.objective_text(model["cost_norm"]),
            "cost_norm": model["cost_norm"],
            "grid": {
                "step": sparse.GRID_STEP,
                "w_q_min": sparse.GRID_MIN_WQ,
                "size": len(grid),
            },
            "candidates": model["configs"],
            "selection": model["selection"],
            "nomemory_candidate": a.nomemory_candidate
            and any(c["plan"] == "NoMemory" for c in model["configs"]),
            "train_means": model["train_means"],
            "bounds": B,
            "routers": arts,
            "slice_router": slice_router,
            "curves": curves,
            "transform_ms": tr_ms,
            "alias": al,
            "per_query_mean_reward": {
                k: v["per_query"]["mean_reward"] for k, v in est.items()
            },
            "oof": {str(k): v for k, v in oof.items()},
        },
    )
    np.save(out / "union_queries.npy", np.array(prof.queries))


# ------------------------------------------------------------------ selection (frozen)


def select_stage(a) -> dict:
    from trimem.experiments import slo_select_bo as slo
    from trimem.optimizer.query_router import sparse

    sel_p = OUT / "selection.json"
    if sel_p.exists():
        return json.loads(sel_p.read_text())
    T = profile_table(a)
    rr = OUT / "router" / "router_report.json"
    if not rr.exists():
        raise SystemExit("run the router stage before freezing the selection")
    Rr = json.loads(rr.read_text())
    if Rr.get("protocol") != sparse.PROTOCOL:
        raise SystemExit(
            f"router report protocol {Rr.get('protocol')!r} != {sparse.PROTOCOL!r}; rerun the router stage"
        )
    B = slo.bounds(DS, T["points"])
    if json.loads(json.dumps(B)) != Rr["bounds"]:
        raise SystemExit(
            "router bounds differ from the profile's; rerun the router stage"
        )
    sels = {}
    for n in SLO_NAMES:
        p = slo.choose(T["points"], B[n], slo.SLOS[n]["objective"])
        sels[n] = {
            "opt_oneconfig": None
            if p is None
            else {
                k: p[k]
                for k in (
                    "label",
                    "key",
                    "cfg",
                    "q",
                    "q_se",
                    "gain",
                    "gain_se",
                    "tok",
                    "lat",
                    "fixed",
                )
            }
            | {"n_feasible": sum(slo.feasible(q, B[n]) for q in T["points"])},
            "opt_perquery": Rr["routers"].get(n),
        }
    sel = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "experiment": T["state"]["experiment"],
        "panel_hash": T["panel"]["panel_hash"],
        "memory_policy": mp.SUCCESS_FAILURE,
        "space_version": space.VERSION,
        "quality": f"pass^{PROFILE_K} (train profiling, {PROFILE_TRIALS} trial)",
        "n_configs": len(T["points"]),
        "bounds": B,
        "slo_text": {n: slo.SLOS[n]["text"] for n in SLO_NAMES},
        "nomemory_train": T["nomemory"],
        "selections": sels,
        "router_protocol": Rr["protocol"],
        "weight_slices": Rr["slice_router"],
        "points": T["points"],
    }
    save(sel_p, sel)
    log(f"selection frozen: {sel_p}")
    return sel


# ------------------------------------------------------------------ test


def routing_table(sel: dict) -> dict:
    """Frozen per-task routing of every test arm (written once, before any test episode): arm -> task key ->
    {config, overhead_ms}. All 4 trials of a task follow its route (the router sees the task, not the trial)."""
    from trimem.optimizer.query_router import runtime, sparse

    p = OUT / "test" / "routing.json"
    if p.exists():
        tab = json.loads(p.read_text())
        if tab.get("test_trials", 4) != TEST_TRIALS:
            raise SystemExit(
                f"{p} was frozen for {tab.get('test_trials', 4)} test trials, the protocol is {TEST_TRIALS}"
            )
        return tab
    if sel.get("router_protocol") != sparse.PROTOCOL:
        raise SystemExit(
            f"selection.json router protocol {sel.get('router_protocol')!r} != {sparse.PROTOCOL!r}"
        )
    ids = test_keys()
    recs = {
        r["query_id"]: r
        for r in json.loads((OUT / "router" / "features_test.json").read_text())[
            "records"
        ]
    }
    if set(ids) - set(recs):
        raise SystemExit(
            f"test feature records missing for {sorted(set(ids) - set(recs))[:3]}"
        )
    rr = json.loads((OUT / "router" / "router_report.json").read_text())
    ovh = {q: recs[q]["overhead"]["probe_ms"] + rr["transform_ms"] for q in ids}
    arms: dict[str, dict] = {}
    for n in SLO_NAMES:
        s = sel["selections"][n]
        if s["opt_oneconfig"]:
            cfg = space.validate(s["opt_oneconfig"]["cfg"])
            arms[f"Opt-OneConfig-{n}"] = {
                "kind": "oneconfig",
                "slo": n,
                "routes": {q: {"config": cfg, "overhead_ms": 0.0} for q in ids},
            }
    # Opt-PerQuery on test (user 2026-10-19): exactly TEST_WEIGHTS with the frozen weight-slice artifact (same
    # model as every SLO artifact; ``route_test_weights`` applies each weight vector with the deployed rule).
    # The train SLO choices of Opt-PerQuery stay in selection.json / router_report.json but are NOT test arms.
    c = sel["weight_slices"]
    art = json.loads(Path(c["path"]).read_text())
    if runtime.artifact_sha(art) != c["sha"]:
        raise SystemExit(
            "weight-slice router artifact changed since the selection was frozen"
        )
    names = [(f"Opt-PerQuery-{label}", w) for label, w in TEST_WEIGHTS.items()]
    for (name, w), routes in zip(
        names,
        sparse.route_test_weights(art, [recs[q] for q in ids], names, ovh).values(),
        strict=True,
    ):
        arms[name] = {
            "kind": "perquery",
            "weights": list(w),
            "weights_name": sparse.arm_name(w),
            "sha": c["sha"],
            "routes": routes,
        }
    tab = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "experiment": test_experiment(),
        "router_protocol": sel["router_protocol"],
        "num_trials": NUM_TRIALS,
        "test_trials": TEST_TRIALS,
        "test_quality": f"pass^{TEST_K}",
        "test_version": TEST_VERSION,
        "arms": arms,
        "perquery_slo_train_only": {
            n: sel["selections"][n]["opt_perquery"] for n in SLO_NAMES
        },
        "pair_budget": pair_budget(arms),
    }
    save(p, tab)
    log(f"routing frozen: {len(arms)} arms; pair budget {tab['pair_budget']}")
    return tab


def pair_budget(arms: dict[str, dict]) -> dict[str, int]:
    """Offline counts: distinct (configuration, task) pairs of the Opt-OneConfig arms / of all arms, the extra pairs
    the Opt-PerQuery arms add, the NoMemory pairs (reused, never run) and the NEW episodes (non-NoMemory pairs x
    trials)."""

    def pairs(kinds: tuple[str, ...] | None) -> set:
        return {
            (space.key(r["config"]), q)
            for arm in arms.values()
            if kinds is None or arm["kind"] in kinds
            for q, r in arm["routes"].items()
        }

    one_p, all_p = pairs(("oneconfig",)), pairs(None)
    nm = {x for x in all_p if json.loads(x[0])["plan"] == "NoMemory"}
    return {
        "oneconfig_pairs": len(one_p),
        "all_pairs": len(all_p),
        "perquery_extra_pairs": len(all_p - one_p),
        "nomemory_pairs_reused": len(nm),
        "new_episodes": NUM_TRIALS * len(all_p - nm),
    }


def pairs_of(tab: dict) -> dict[str, list[str]]:
    """Config-reuse principle: config key -> the test tasks any arm routes to it (each (config, task, trial) once)."""
    out: dict[str, set] = {}
    for arm in tab["arms"].values():
        for q, r in arm["routes"].items():
            out.setdefault(space.key(r["config"]), set()).add(q)
    order = {k: i for i, k in enumerate(test_keys())}
    return {k: sorted(v, key=order.__getitem__) for k, v in sorted(out.items())}


# ------------------------------------------------------------------ outer 5-fold CV (user 2026-10-21)


def cv_routing_path() -> Path:
    return OUT / "test" / "routing_cv.json"


def load_cv_routing() -> dict | None:
    p = cv_routing_path()
    return json.loads(p.read_text()) if p.exists() else None


def observed_test_reuse_rate() -> float | None:
    """Quality-reuse hit rate of the deployed test stage (reuse records / result files of its pairs dirs): the
    planning discount of the CV pair budget (actual reuse depends on the injected memory, known only at run time)."""
    hits = files = 0
    for f in (OUT / "test" / "pairs").glob("cfg_*/*.json"):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(r, dict) or "reward" not in r:
            continue
        files += 1
        hits += r.get("reuse") is True
    return hits / files if files else None


def completed_pairs(arms: dict[str, dict]) -> set[tuple[str, str]]:
    """(config key, task) pairs of ``arms`` whose test episode(s) are already on disk (any earlier test run)."""
    out = set()
    for k, qs in pairs_of({"arms": arms}).items():
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            continue
        eps = arm_episodes(
            OUT / "test" / "pairs" / cfg_name(cfg), qs, split="test", cfg=cfg
        )
        out |= {(k, q) for q in qs if len(eps.get(q, {})) == NUM_TRIALS}
    return out


def cv_stage(a) -> dict:
    """Outer 5-fold CV of Opt-OneConfig / Opt-PerQuery on the 178 training tasks (folds stratified by domain),
    then the frozen CV test routing (``test/routing_cv.json``). Offline: no episode runs here."""
    from trimem.experiments import router_outer_cv as rcv
    from trimem.experiments import slo_select_bo as slo
    from trimem.optimizer.query_router import sparse

    if not (OUT / "selection.json").exists():
        raise SystemExit("freeze the selection (select stage) before the CV stage")
    T = profile_table(a)
    al = alias(T["keys"])
    strata = {al[k]: domain_of(k) for k in T["keys"]}
    est = sparse_estimates(T, a.nomemory_candidate)
    prof = sparse.from_estimates(est, strata, T["weights"], {})
    recs = {
        al[r["query_id"]]: {
            **r,
            "query_id": al[r["query_id"]],
            "store_key": r["query_id"],
        }
        for r in feature_records(T["keys"], OUT / "router" / "features_train.json")
    }
    tids = test_keys()
    trecs = feature_records(tids, OUT / "router" / "features_test.json")
    keys = {p["key"]: p["label"] for p in T["points"]}
    cols = [i for i, c in enumerate(prof.configs) if space.key(c) in keys]
    labels = {i: keys[space.key(prof.configs[i])] for i in cols}
    nm = {al[k]: T["nomemory_tasks"][k]["quality"] for k in T["keys"]}
    best_q = max(T["points"], key=lambda p: (p["q"], -p["lat"]))
    rep = rcv.run_folds(
        OUT,
        dataset=DS,
        prof=prof,
        records=recs,
        test_records=trecs,
        groups=tuple(a.groups.split(",")),
        oneconfig_cols=cols,
        labels=labels,
        nomemory=nm,
        bounds_fn=lambda pts: slo.bounds(DS, pts),
        choose_fn=slo.choose,
        objectives={n: slo.SLOS[n]["objective"] for n in SLO_NAMES},
        test_weights=TEST_WEIGHTS,
        fit_kwargs={"trials": a.trials, "n_jobs": a.jobs, "seed": a.seed},
        fallback=best_q["cfg"],
        trained_on={
            "experiment": T["state"]["experiment"],
            "panel_hash": T["panel"]["panel_hash"],
            "memory_policy": mp.SUCCESS_FAILURE,
            "query_alias": "tau2_<panel index>",
        },
        log=log,
    )
    sel = json.loads((OUT / "selection.json").read_text())
    tab = rcv.freeze_routing(
        cv_routing_path(),
        rep=rep,
        test_ids=tids,
        validate=space.validate,
        deployed=routing_table(sel),
        key=space.key,
        nomemory_key=space.key(space.NOMEM),
        completed=completed_pairs,
        reuse_rate=observed_test_reuse_rate(),
        extra={
            "experiment": test_experiment(),
            "test_trials": TEST_TRIALS,
            "test_quality": f"pass^{TEST_K}",
            "test_version": TEST_VERSION,
            "num_trials": NUM_TRIALS,
        },
        log=log,
    )
    return tab


def test_reuse_index(experiment: str) -> Path | None:
    """Test-stage quality reuse (its own index: never across train / test) when the profiling state enabled it.
    semfilter-v2 recorded (tau2.sfv2_rerun, user 2026-10-22): a NEW index (``<experiment>:sfv2``) so no v1-judge
    test episode is ever reused."""
    st = OUT / "state.json"
    if st.exists() and (s := json.loads(st.read_text())).get("quality_reuse"):
        tag = ":sfv2" if s.get("semfilter") else ""
        return qr.index_path(REUSE_DIR, experiment + tag, "test")
    return None


def test_stage(a) -> None:
    if not (OUT / "selection.json").exists():
        raise SystemExit(
            "selection.json missing: freeze the selection (select stage) before any test episode"
        )
    tab = routing_table(json.loads((OUT / "selection.json").read_text()))
    cv = load_cv_routing()  # outer-CV fold arms (cv stage): their pairs run in the same pairs dirs, once each
    if cv is not None:
        tab = {**tab, "arms": {**tab["arms"], **cv["arms"]}}
        log(
            f"test: + {len(cv['arms'])} outer-CV arms (pair budget {cv['pair_budget']})"
        )
    pairs = pairs_of(tab)
    nm_key = space.key(space.NOMEM)
    arms = []
    for k, qs in pairs.items():
        cfg = json.loads(k)
        if (
            cfg["plan"] == "NoMemory"
        ):  # reused from the existing 4-trial NoMemory test arm
            eps = ref_episodes(REUSE / "NoMemory", qs)
            if (
                per_task(eps, qs, "test") is None
            ):  # (never expected) missing episodes are run in our own pairs dir
                arms.append(
                    Arm(
                        "test NoMemory pairs",
                        OUT / "test" / "pairs" / cfg_name(cfg),
                        "test",
                        qs,
                        None,
                    )
                )
            continue
        arms.append(
            Arm(
                f"test {cfg_name(cfg)}",
                OUT / "test" / "pairs" / cfg_name(cfg),
                "test",
                qs,
                cfg,
                tab["experiment"],
                REUSE / "NoMemory",
                {
                    "bo_label": "test-pairs",
                    "bo_experiment": experiment_id(),
                    "quality": f"pass^{K}",
                },
                reuse_index=test_reuse_index(tab["experiment"]),
                regime=f"c{a.test_concurrency}",
            )
        )
    log(
        f"test: {len(tab['arms'])} arms -> {sum(len(v) for k, v in pairs.items() if k != nm_key)} (config, task) pairs "
        f"incl. already-completed ones (resumed, not rerun) x {NUM_TRIALS} trials over {len(arms)} configurations; NoMemory pairs reused "
        f"{len(pairs.get(nm_key, []))} (vs {len(tab['arms']) * len(test_keys())} pairs without reuse)"
    )
    if arms:
        res = run_parallel(arms, a.test_concurrency)
        bad = [x.name for x in arms if res[x.name] < x.expected]
        if bad:
            raise SystemExit(f"test pairs incomplete: {bad}")
    log("TEST DONE τ²")


def test_rows(tab: dict) -> dict[tuple[str, str], dict]:
    """(config key, task) -> task summary (+ ``e2e_ms`` alias of latency) of every executed / reused test pair."""
    rows = {}
    for k, qs in pairs_of(tab).items():
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            eps = ref_episodes(REUSE / "NoMemory", qs)
            if per_task(eps, qs, "test") is None:
                eps = arm_episodes(
                    OUT / "test" / "pairs" / cfg_name(cfg), qs, split="test", cfg=None
                )
        else:
            eps = arm_episodes(
                OUT / "test" / "pairs" / cfg_name(cfg), qs, split="test", cfg=cfg
            )
        for q, t in (
            per_task(eps, [q for q in qs if len(eps.get(q, {})) == NUM_TRIALS], "test")
            or {}
        ).items():
            rows[(k, q)] = {**t, "e2e_ms": t["latency_ms"]}
    return rows


def arm_metrics(arm: dict, rows: dict) -> dict:
    """pass^1..4 / pass@4 / mean reward of a routed arm (per task from its routed pair)."""
    got = [
        rows[(space.key(r["config"]), q)]
        for q, r in arm["routes"].items()
        if (space.key(r["config"]), q) in rows
    ]
    if not got:
        return {}
    return {
        "pass_hat": {
            j: sum(t["pass_hat"][j] for t in got) / len(got)
            for j in range(1, NUM_TRIALS + 1)
        },
        "pass_at_n": sum(t["pass_at_n"] for t in got) / len(got),
        "mean_reward": sum(t["mean_reward"] for t in got) / len(got),
    }


def reference_arms() -> dict[str, dict]:
    """The existing REF_TRIALS-trial test arms (reused, not rerun) on the same 100 tasks: summary on their first
    TEST_TRIALS trials (test-1trial-v1: trial 0 = the same official seed as the Opt arms; primary) and, as
    context, on all REF_TRIALS trials (``all_trials``)."""
    keys, out = test_keys(), {}
    w = strata_weights(keys)
    for name in REFERENCE_ARMS:
        eps: dict[str, dict[int, dict]] = {}
        for key in keys:
            d, t = key.split(":", 1)
            for k in range(REF_TRIALS):
                r = R.read_result(R.result_path(REUSE / name, d, t, k))
                if r is not None:
                    eps.setdefault(key, {})[k] = episode_record(r)
        if any(len(eps.get(k, {})) < REF_TRIALS for k in keys):
            out[name] = None
            continue
        prim = per_task(
            {k: {j: v[j] for j in range(TEST_TRIALS)} for k, v in eps.items()},
            keys,
            "test",
        )
        full = {
            k: task_summary(
                [v[j]["reward"] for j in range(REF_TRIALS)],
                [v[j]["tokens"] for j in range(REF_TRIALS)],
                [v[j]["latency_ms"] for j in range(REF_TRIALS)],
                1,
            )
            for k, v in eps.items()
            if k in keys
        }
        out[name] = {
            **summary_of(prim, keys, w),
            "trials_used": list(range(TEST_TRIALS)),
            "all_trials": {"trials": REF_TRIALS, **summary_of(full, keys, w)},
            "dir": str(REUSE / name),
            "note": f"existing {REF_TRIALS}-trial arm (v1 config seed 8 / depth 1 / fanout 4 / top_k 3, edges "
            f"similar), 40 workers; primary = its trial(s) {list(range(TEST_TRIALS))}",
        }
    return out


def report_stage(a) -> dict:
    from trimem.experiments import bo_pool_driver as bpd

    tab = json.loads((OUT / "test" / "routing.json").read_text())
    if tab.get("test_trials", 4) != TEST_TRIALS:
        raise SystemExit("routing.json was frozen for another test-trial protocol")
    cv = load_cv_routing()
    rows = test_rows({**tab, "arms": {**tab["arms"], **(cv or {}).get("arms", {})}})
    keys = test_keys()
    nm_tasks = per_task(ref_episodes(REUSE / "NoMemory", keys), keys, "test")
    if nm_tasks is None:
        raise SystemExit(
            f"{REUSE / 'NoMemory'} lacks trials 0..{TEST_TRIALS - 1} of the 100 tasks"
        )
    nm = {q: t["quality"] for q, t in nm_tasks.items()}
    arms = bpd.assemble(
        tab, rows, nm, key=space.key
    )  # quality = pass^TEST_K per task; paired gain vs NoMemory (same trials)
    for name, arm in tab["arms"].items():
        arms[name].update(arm_metrics(arm, rows))
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "dataset": DS,
        "experiment": tab["experiment"],
        "test_trials": TEST_TRIALS,
        "test_version": TEST_VERSION,
        "quality": f"pass^{K} per task (official C(c,{K})/C({NUM_TRIALS},{K}) over {NUM_TRIALS} trial(s)); "
        "latency = episode wall time; tokens = agent + SemFilter",
        "baseline": {
            "dir": str(REUSE / "NoMemory"),
            "trials_used": list(range(TEST_TRIALS)),
            "quality": sum(nm.values()) / len(nm),
        },
        "references": reference_arms(),
        "notes": [
            "each distinct (configuration, task, trial) ran ONCE and is shared by every arm routing to it: "
            "arms are correlated, fine for a curve, not independent replicates",
            f"NoMemory pairs and the reference arms are the existing {REF_TRIALS}-trial test arms (40 workers), "
            f"compared on trial(s) {list(range(TEST_TRIALS))} (same official seed); the Opt arms ran at the same "
            "total concurrency (40)",
        ],
        "test_concurrency": a.test_concurrency,
        "pairs": len(rows),
        "arms": arms,
    }
    cv_md: list[str] = []
    if (
        cv is not None
    ):  # outer 5-fold CV (user 2026-10-21): primary numbers, mean ± std across folds
        from trimem.experiments import router_outer_cv as rcv

        def fold_metrics(t: dict) -> dict:
            out = bpd.assemble(t, rows, nm, key=space.key)
            return {
                k: v for k, v in out.items() if v.get("covered") == v.get("n")
            }  # complete arms only

        ts = rcv.test_summary(cv, fold_metrics)
        rep["outer_cv"] = {
            "protocol": cv["protocol"],
            "folds": cv["folds"],
            "pair_budget": cv["pair_budget"],
            "evaluation": "test set only (user 2026-10-21); folds fitted on 4/5 of the training tasks",
            "test": ts,
            "incomplete_test_arms": sorted(
                n
                for n, v in bpd.assemble(cv, rows, nm, key=space.key).items()
                if v.get("covered") != v.get("n")
            ),
            "labels": {
                "mean_std": "mean ± sample std (ddof 1) across the 5 outer folds",
                "se_tasks_mean": "mean within-fold SE across tasks (task-sampling noise), NOT across folds",
                "test_concurrency_note": "CV-only test pairs ran later than the deployed pairs (see pair load "
                "regime in each result file); deployed-arm pairs are reused as is",
            },
        }
        cv_md = rcv.markdown(ts, f"pass^{K}", len(keys)) + [
            "",
            "**Latency caveat.** Pairs first needed by the CV arms ran later than the deployed pairs, under "
            "concurrent ALFWorld profiling load on the same vLLM servers (each result file records its `load`). "
            "Their episode wall time is inflated, so CV latency is NOT comparable with the deployed rows; "
            "pass^1, tokens and gain are unaffected by load.",
            "",
        ]
    save(OUT / "test_report.json", rep)
    lines = [
        f"# τ² pool profiling — test (100 tasks x {NUM_TRIALS} trial(s))",
        "",
        *cv_md,
        "## Deployed router (single final fit on all 178 training tasks; continuity)",
        "",
        f"Quality = pass^{K}; NoMemory pass^{K} {rep['baseline']['quality']:.3f} (trial(s) "
        f"{rep['baseline']['trials_used']} of the {REF_TRIALS}-trial arm).",
        "",
        f"| arm | pass^{K} | ± SE | pass@{NUM_TRIALS} | tokens | latency s | configs | Δ pass^{K} vs NoMemory |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, v in arms.items():
        if v.get("quality") is None:
            continue
        lines.append(
            f"| {k} | {v['quality']:.3f} | {v['quality_se']:.3f} | "
            f"{v['pass_at_n']:.3f} | {v['tokens']:,.0f} | {v['latency_s']:.1f} | {v['distinct_configs']} | "
            f"{v['paired_gain_vs_nomemory']:+.3f} |"
        )
    for k, v in rep["references"].items():
        if v:
            lines.append(
                f"| ref {k} | {v['q']:.3f} | {v['q_se']:.3f} | {v['pass_at_n']:.3f} | "
                f"{v['tok']:,.0f} | {v['lat']:.1f} | 1 | |"
            )
    lines += [
        "",
        f"Reference arms on all {REF_TRIALS} trials (context): "
        + "; ".join(
            f"{k} pass^1 {v['all_trials']['pass_hat'][1]:.3f}"
            for k, v in rep["references"].items()
            if v
        ),
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    log(f"report: {OUT / 'test_report.json'}")
    return rep


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    from trimem.optimizer import search_analyzer as sa

    sa.add_arguments(ap)
    pp_policy.add_arguments(ap)
    ap.add_argument(
        "stage", choices=("profile", "router", "select", "cv", "test", "report", "all")
    )
    ap.add_argument(
        "--budget_configs",
        type=int,
        default=BUDGET_CONFIGS,
        help="profiling episode budget = this many configurations x 178 tasks x 1 trial (unpruned plan; 30 -> 5,340 episodes)",
    )
    ap.add_argument(
        "--max_configs",
        type=int,
        default=MAX_CONFIGS,
        help="cap on profiled configurations",
    )
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument(
        "--pause_after_round",
        type=int,
        default=None,
        help=f"profiling-pause-v1: stop before proposing the next round (env {PAUSE_ENV}); resume = rerun without it",
    )
    ap.add_argument(
        "--pause_reason",
        default=None,
        help="recorded in profile_complete.json / state.json",
    )
    ap.add_argument(
        "--no_analyst",
        dest="analyst",
        action="store_false",
        help="disable LLM analysis (new runs: local-search guidance; historical runs: legacy analyst)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        help="profiling: total concurrent episodes",
    )
    ap.add_argument("--test_concurrency", type=int, default=TEST_CONCURRENCY)
    ap.add_argument(
        "--no_nomemory_candidate", dest="nomemory_candidate", action="store_false"
    )
    ap.add_argument(
        "--groups",
        default="structured,probe,emb",
        help="router feature groups (no LLM)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=30,
        help="router: Optuna trials per candidate (paper: 30)",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    return ap


STAGES = {
    "profile": profile,
    "router": router_stage,
    "select": select_stage,
    "cv": cv_stage,
    "test": test_stage,
    "report": report_stage,
}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another driver holds {OUT / 'driver.lock'}") from None

    def stop(signum, frame):
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # semfilter-v2 (tau2.sfv2_rerun, user 2026-10-22): export the knobs recorded in state.json ``semfilter`` to every
    # runner (refused without TAU2_SEMFILTER=v2); nothing recorded = unchanged (v1 judge)
    from trimem.experiments.tau2 import sfv2_rerun

    if sf_env := sfv2_rerun.apply_semfilter(OUT):
        log(f"semfilter-v2 recorded: exporting {sorted(sf_env)}")
    log(
        f"tau2 pool_profile {a.stage}: space {space.VERSION}, profiling {PROFILE_TRIALS} trial / pass^{PROFILE_K}, "
        f"test {TEST_TRIALS} trials / pass^{TEST_K}, "
        f"concurrency {a.concurrency} / test {a.test_concurrency}, out {OUT} ({sys.executable})"
    )
    for s in (
        ("profile", "router", "select", "cv", "test", "report")
        if a.stage == "all"
        else (a.stage,)
    ):
        STAGES[s](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
