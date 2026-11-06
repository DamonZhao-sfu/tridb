"""Pool profiling round for ALFWorld / ScienceWorld (user design 2026-10-19; plan
docs/trimem_pool_profiling_plan_v0.1.0.md).

  TEST       the unchanged 200 AgentEval items per dataset (held out of memory and of profiling).
  MEMORY     the full store (every official fold except the test items), L1 isolation, memory policy
             ``$TRIMEM_MEMORY_POLICY`` (recorded in state.json; a resume under another policy is refused).
  POOL       every non-test task that the AgentGym env server can address (``build_agentenv_pool``).
  PROFILING  round 0 = the 4 plan defaults + 6 Sobol, then BO rounds of 5 (qLogNEHVI, kriging believer). EACH
             configuration runs on its OWN stratified random 200-sample of the pool (strata = task_type; seed =
             sha256(experiment|config key)), and so do NoMemory and the noise repeat of the default VGR. The BO
             objective uses per-configuration stratified means and SEs (``session_bo_agentenv.load_points_unpaired``);
             ``--sampling shared`` restores the old design (one common sample, paired gains; ``--staging none``).
  STAGING    (default ``--staging pareto``, protocol ``staged_pruning.PROTOCOL`` = staged-pareto-prune-v2, user
             2026-10-20) EVERY round, round 0 included (only NoMemory and the noise repeat are protected): each
             configuration runs nested prefixes 50 -> 100 -> 200 of its own sample (``staged_pruning.stage_order``,
             strata = task type), synchronously per batch, NoMemory the same prefixes of its own sample alongside
             (the noise repeat runs in full with the last stage), and is pruned at 50 / 100 when its optimistic
             corner (quality + 1.96 SE, tokens - 1.96 SE, latency - 1.96 SE) is weakly dominated by the point of a
             fully evaluated configuration or by the PESSIMISTIC corner (quality - 1.96 SE, tokens + 1.96 SE,
             latency + 1.96 SE) of NoMemory / any configuration evaluated at >= that stage, each on its own stage
             prefix (unpaired: own samples). Plan guard: the last live configuration of a plan without a full one is
             never pruned. Pruned configurations stay GP observations (partial-n means, their SE as noise);
             selection and the router use fully evaluated configurations only. Budget = the unpruned plan (50 x 200
             configuration episodes, round 0 included); savings buy more configurations (<= 80 in total).
  CHANNELS   Sobol / LLM analyst / DLS / crossover candidate pool, qLogNEHVI arbitration; qfirst-slot-v1 from BO
             round 3; channel-slots-v1 (user 2026-10-20, from the round recorded in state.json ``channel_slots``, a
             fresh run: round 3): richer crossover (champions + adjacent relaxed-front pairs, 1- / 2-knob grafts),
             DLS seeded from crossover rows when the analyst yields no rows, ordinary batches qfirst / xover-slot /
             sobol-slot / hv / hv; forced rounds unchanged. channel-slots-v2 (user 2026-10-21; states recorded from
             then on, fresh runs from round 1): qfirst / llm-slot / xover-slot / dls-slot / sobol-slot; forced rounds
             keep the forced picks but reserve fills so crossover, DLS and Sobol each contribute (batch >= 4).
  SELECTION  frozen in selection.json before any test episode: for each SLO of ``slo_select_bo`` (percentile
             thresholds recomputed on this profile) Opt-OneConfig = best profiled configuration satisfying it,
             Opt-PerQuery = sparse BRANE router (per-configuration predictors on each configuration's own
             sample) scoring U = w_Q q - w_L lat~ - w_T tok~ (one weight per Opt-OneConfig objective; raw-scale
             min-max costs over the candidates; user 2026-10-19), with (w_Q, w_L, w_T) chosen by the same SLO
             over a simplex grid on the out-of-fold TRAIN replay estimate (``sparse.PROTOCOL``).
  TEST       Opt-OneConfig SLO-1/2/3 + Opt-PerQuery at the 4 weight vectors of ``TEST_WEIGHTS`` (user 2026-10-19);
             each distinct (configuration, query) pair runs once; optional NoMemory + the 4 plan defaults
             (``--test_fixed``). Test concurrency = ``TEST_CONCURRENCY`` (40, user 2026-10-20; the reused controlled
             arms, whose NoMemory is the paired baseline, ran at 16 = 2 x 8: latencies are at a different load).

Stages (``.venv-optimizer``, inside trimem-eval.slice; the env servers of the dataset must be up):
    profile | router | select | cv | test | report | all  (profile --pause_after_round N: profiling-pause-v1)
    PYTHONPATH=. .venv-optimizer/bin/python -m trimem.experiments.bo_pool_driver all --dataset alfworld

Resumable everywhere: the sample of every arm is a pure function of (pool, experiment, label); proposals
are stored once in state.json; finished items are skipped (result file + exactly one evaluated opt row).
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import aux_observations as aux_obs
from trimem.experiments import bo_agentenv_driver as drv
from trimem.experiments import build_agentenv_pool as bp
from trimem.experiments import episode_budget as eb
from trimem.experiments import memory_policy as mp
from trimem.experiments import quality_reuse as qr
from trimem.optimizer import search_analyzer as sa
from trimem.optimizer import pruning_policy as pp_policy
from trimem.optimizer import session_space_agentenv_v2 as space
from trimem.optimizer import staged_pruning as sp
from trimem.optimizer import unpaired_stats as us

ROOT = drv.ROOT
OUT = drv.OUT
PANEL = "pool-strat200"
VERSION = "bo3obj-v3"
# BO rounds >= 1 add the LLM analyst candidate channel (user 2026-10-20; ``analyzer_pool.PROTOCOL``): recorded in
# the run identity (experiment id), NOT in the space-independent NoMemory sample namespace (reused episodes).
ANALYST_TAG = "analyst-pool-v1"
# Channel slots (user 2026-10-20; ``session_bo_agentenv.CHANNEL_SLOTS_PROTOCOL``): NOT part of the run identity;
# recorded once in state.json ``channel_slots`` (from max(1, next round); user 2026-10-21, was 3). Acquisition tag bo-acq-v3 from then on.
# channel-slots-v2 (user 2026-10-21) for states recorded from now on (fresh runs: from round 1): qfirst / llm-slot /
# xover-slot / dls-slot / sobol-slot; states that already record channel-slots-v1 (ALFWorld) keep it.
CHANNEL_SLOTS_TAG = "channel-slots-v2"
# channel-slots-v3 (user 2026-10-22): opt-in at (re)start with POOL_CHANNEL_SLOTS=v3 (or --channel_slots v3); recorded
# in state.json channel_slots from the next round to be proposed (the replaced block kept as ``before``). Usually
# together with POOL_BATCH=10 (or --round_batch 10): the per-round batch from the next round (state.json
# batch_schedule; the run identity ``batch`` and the staged budget -- 50 configurations x 200 -- are unchanged).
CHANNEL_SLOTS_ENV = "POOL_CHANNEL_SLOTS"
# channel-slots-v4 (user 2026-10-22): v3 with qfirst / token-first / latency-first replaced by Sobol slots (sobol x4,
# llm x2, xover x2, dls x2; no qfirst pick); opt-in POOL_CHANNEL_SLOTS=v4, an explicit v3 -> v4 upgrade applies from
# the next round to be proposed (v3 kept as ``before``); a v4 state refuses v3 / v2.
# sobol-only-v1 (user 2026-10-22, ALFWorld v7 resume / ScienceWorld): with v4, POOL_RANDOM_ROUNDS=N (or
# --random_rounds N) makes the next N proposed rounds random-sampling rounds (every slot a sobol-slot pick; recorded
# in state.json channel_slots as a sobol-only-v1 block with from_round / until_round) and v4 starts after them;
# POOL_RANDOM_ROUNDS_UNTIL=R (or --random_rounds_until R) gives the last random round directly.
RANDOM_ROUNDS_ENV = "POOL_RANDOM_ROUNDS_UNTIL"
RANDOM_ROUNDS_N_ENV = "POOL_RANDOM_ROUNDS"
CHANNEL_SLOTS_CHOICES = {
    "v2": "channel-slots-v2",
    "v3": "channel-slots-v3",
    "v4": "channel-slots-v4",
    "v5": "channel-slots-v5",  # user 2026-10-22 (ALF v7 from round 7: llm, dls, xover, sobol; batch 4)
    "v6": "channel-slots-v6",  # user 2026-10-22 (Sci from round 3: v5's 4 channel picks + 4 union hv; batch 8)
}
BATCH_ENV = "POOL_BATCH"
BATCH_SCHEDULE_PROTOCOL = "batch-schedule-v1"
# Router backends (user 2026-10-21): BRANE always; the LimeQO backend (fit, CV folds and Opt-PerQuery-LimeQO-* arms)
# can be switched off per run (--no_limeqo or POOL_NO_LIMEQO=1; ScienceWorld). Recorded once in state.json
# ``router_backends`` by the profile stage; later stages refuse a mismatching flag. Not part of the run identity.
ROUTER_BACKENDS_PROTOCOL = "router-backends-v1"
NO_LIMEQO_ENV = "POOL_NO_LIMEQO"
N_SAMPLE = 200
ROUNDS, BATCH = (
    8,
    5,
)  # the unpruned plan: 10 + 8 x 5 = 50 configurations (= the staged budget)
N_ROUND0 = 10  # plan defaults + Sobol fill (session_bo_agentenv.round0; -v7: 5 plan defaults + 5 Sobol)
# Staged profiling with 3-D Pareto pruning (user 2026-10-19; trimem.optimizer.staged_pruning): BO configurations
# run on nested prefixes 50 -> 100 -> 200 of their own stratified sample; savings buy more configurations.
STAGE_SIZES = (50, 100, N_SAMPLE)
MAX_CONFIGS = 80
STAGING_OFF = "none"
# v2 (2026-10-19): three-weight Opt-PerQuery arms (sparse.PROTOCOL); v3 (2026-10-19, user): test arms = Opt-OneConfig
# SLO-1/2/3 + Opt-PerQuery at exactly TEST_WEIGHTS (no weight slices, no Opt-PerQuery SLO arms on test).
# v4 (2026-10-20, user): Opt-PerQuery test weights = default (0.6, 0.2, 0.2), quality (0.8, 0.1, 0.1), tokens
# (0.1, 0.1, 0.8), latency (0.1, 0.8, 0.1).
TEST_VERSION = "pool-strat200-v4"
TEST_WEIGHTS = {  # (w_Q, w_L, w_T) of the Opt-PerQuery test arms (the frozen weight-slice artifact, deployed rule)
    "default": (0.6, 0.2, 0.2),
    "quality": (0.8, 0.1, 0.1),
    "tokens": (0.1, 0.1, 0.8),
    "latency": (0.1, 0.8, 0.1),
}
SLO_NAMES = ("SLO-1", "SLO-2", "SLO-3")
# Profiling pause (profiling-pause-v1, as tau2.pool_profile; user 2026-10-21: pause ALFWorld after round 5 and test
# with the current points). ``profile --pause_after_round N`` (or env POOL_PAUSE_AFTER_ROUND) stops BEFORE
# proposing round N+1 and writes profile_complete.json with ``paused_by_user`` (+ state.json ``profiling_pause``);
# router / select / cv / test / report then run on rounds 0..N. RESUME: rerun ``profile`` / ``all`` WITHOUT the
# flag (or with a later round): the paused marker (profile_complete.json) MOVES to <run>/paused_after_round<N>_<ts>/
# and every artifact derived from it (PAUSED_ARTIFACTS) is COPIED there and LEFT IN PLACE (user 2026-10-22, ALF v7
# resume: the published page / c16 test reports read them at their original paths; the router / select / cv / test
# stages overwrite them when rerun); BO continues at round N+1 (budget / state unchanged); test/pairs is kept (each
# (config, query) test pair runs once).
PAUSE_MARKER = "profile_complete.json"
PAUSE_ENV = "POOL_PAUSE_AFTER_ROUND"
PAUSE_PROTOCOL = "profiling-pause-v1"
PAUSED_ARTIFACTS = (
    "profile_complete.json",
    "selection.json",
    "router",
    "router_cv",
    "router_cv_limeqo",
    "test/routing.json",
    "test/routing_cv.json",
    "test_report.json",
    "report.md",
    "report_cv.md",
)
MAX_JAVA = 80  # host-wide java processes after an arm (2 servers x 32-env cap + slack); more = a leak
log = drv.log
save = drv.save


# ------------------------------------------------------------------ identities / paths

# Space tag in the run identity (user correction 2026-10-20): session space v2 fixed edges ``similar_up``; the
# corrected space (``session-space-agentenv-v3``, edges ``similar+similar_up``) runs under its own experiment ids
# and directories (``<ds>_pool-strat200-v3``), so no row / episode of the void similar_up run can mix in.
SPACE_TAG = space.VERSION.rsplit(
    "-", 1
)[
    1
]  # "v5" (-v4 = memory_form searched raw / compact / distilled; -v5 = memory_form fixed raw, 2026-10-20)


# Run tag (user 2026-10-22, ALFWorld branch "v8"): POOL_RUN_TAG=<tag> (or --run_tag) gives a run its own experiment id
# and directories (``<ds>_pool-strat200-v7-<tag>``) WITHOUT changing anything for untagged runs. The NoMemory arm and
# its sample namespace stay untagged (``nomemory_dir`` / ``nomemory_sample_experiment``): a tagged run points to the
# untagged run's NoMemory baseline instead of rerunning it.
RUN_TAG_ENV = "POOL_RUN_TAG"
RUN_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9.]{0,15}$")


def run_tag() -> str:
    """The process's run tag ('' = untagged); validated."""
    t = os.environ.get(RUN_TAG_ENV, "").strip()
    if t and not RUN_TAG_RE.match(t):
        raise SystemExit(f"{RUN_TAG_ENV}={t!r}: expected {RUN_TAG_RE.pattern}")
    return t


def _tagged(s: str) -> str:
    t = run_tag()
    return f"{s}-{t}" if t else s


def _panel_id(sampling: str) -> str:
    return f"{PANEL}{'' if sampling == 'per-config' else '-shared'}"


def experiment_id(ds: str, sampling: str = "per-config") -> str:
    return mp.tag(
        f"{ds}:{_tagged(f'{_panel_id(sampling)}-{SPACE_TAG}')}:{VERSION}+{ANALYST_TAG}"
    )


def nomemory_sample_experiment(ds: str, sampling: str = "per-config") -> str:
    """Seed namespace of the NoMemory sample. NoMemory does not depend on the session space, so its sample is
    space-INDEPENDENT (= the pre-v3 experiment id): a space correction keeps the NoMemory sample and its episodes
    (user 2026-10-20). Under ``--sampling shared`` every arm shares ONE sample, which must then follow the run's
    experiment id."""
    return (
        mp.tag(f"{ds}:{_panel_id(sampling)}:{VERSION}")
        if sampling == "per-config"
        else experiment_id(ds, sampling)
    )


def base_dir(ds: str, sampling: str = "per-config") -> Path:
    return OUT / _tagged(f"{ds}_{_panel_id(sampling)}-{SPACE_TAG}")


_NM_REGIME: dict[
    str, Path
] = {}  # ds -> NoMemory dir of the run's current latency regime (latency-regime-v1)


def nomemory_dir(ds: str, sampling: str = "per-config", *, regime: bool = True) -> Path:
    """The profiling NoMemory arm: always the UNTAGGED base directory (a tagged run reuses that baseline).
    latency-regime-v1: once the loaded state records a runtime regime change, the NoMemory rerun of the new
    regime (``state.json latency_regime.nomemory.dir``; same sample) replaces it everywhere (``regime=False``:
    the original arm)."""
    if _ABLATION_POOL is not None:
        if ds != "sciworld" or not run_tag():
            raise ValueError("ablation NoMemory requires an isolated ScienceWorld run tag")
        return run_dir(ds, sampling) / "NoMemory"
    if regime and ds in _NM_REGIME:
        return _NM_REGIME[ds]
    return OUT / f"{ds}_{_panel_id(sampling)}-{SPACE_TAG}" / "NoMemory"


def run_dir(ds: str, sampling: str = "per-config") -> Path:
    """Memory-arm directory (memory-policy tagged); NoMemory lives untagged in ``base_dir``."""
    return mp.tag_dir(base_dir(ds, sampling))


def test_experiment(ds: str, arm: str) -> str:
    return mp.tag(f"{ds}:agenteval200:{TEST_VERSION}:{arm}")


def test_arm_dir(ds: str, arm: str, sampling: str = "per-config") -> Path:
    return (
        base_dir(ds, sampling) / "test" / "NoMemory"
        if arm == "NoMemory"
        else run_dir(ds, sampling) / "test" / arm
    )


_ABLATION_POOL: dict | None = None


def configure_ablation_pool(path, manifest):
    from trimem.experiments.scienceworld_trace_ablation import validate_isolated_pool

    global _ABLATION_POOL
    candidate = json.loads(Path(path).read_text())
    validate_isolated_pool(candidate, manifest)
    _ABLATION_POOL = candidate


def load_pool(ds: str) -> dict:
    if _ABLATION_POOL is not None and ds != "sciworld":
        raise ValueError("isolated ablation pool is only valid for ScienceWorld")
    p = _ABLATION_POOL if _ABLATION_POOL is not None else json.load(open(bp.pool_path(ds)))
    if p["size"] < N_SAMPLE:
        raise SystemExit(f"pool of {p['size']} < sample size {N_SAMPLE}")
    return p


def pool_meta(pool: dict) -> tuple[dict, dict, dict]:
    """(qid -> stratum, stratum -> weight, stratum -> size)."""
    strata = {x["item_id"]: x["stratum"] for x in pool["items"]}
    return strata, bp.strata_weights(pool["items"]), dict(pool["strata"])


# ------------------------------------------------------------------ samples


def sample_label(kind: str, cfg: dict | None, sampling: str) -> str:
    if sampling == "shared":
        return "shared"
    if kind == "config":
        return space.key(cfg)
    if kind == "noise":
        return "noise-repeat:" + space.key(cfg)
    return kind  # NoMemory


def write_sample(
    ds: str,
    pool: dict,
    base: Path,
    name: str,
    label: str,
    exp: str,
    size: int = N_SAMPLE,
    uniform: bool = False,
) -> dict:
    """The arm's stratified sample (pure function of pool + experiment + label), written once as a
    panel file for ``agentgym_session_eval --panel``; returns its record. sample-size-v1 (``uniform``): ``size``
    tasks drawn uniformly at random from the whole pool (no strata; seed = sample_seed(exp, label|uniform-<size>)),
    recorded with ``sampling: uniform``; a stratified sample of size != 200 uses seed sample_seed(exp,
    label|stratified-<size>) (user correction 2026-10-22: ALF v7 from round 6 = 100 STRATIFIED tasks)."""
    if uniform:
        seed = bp.sample_seed(exp, f"{label}|uniform-{size}")
        pool_items = sorted(
            pool["items"], key=lambda x: int(x["item_id"].rsplit("_", 1)[1])
        )
        if size > len(pool_items):
            raise SystemExit(
                f"uniform sample of {size} from a pool of {len(pool_items)}"
            )
        items = random.Random(seed).sample(pool_items, size)
        items.sort(key=lambda x: int(x["item_id"].rsplit("_", 1)[1]))
    else:
        # sample-size-v1: a smaller stratified sample gets its own seed (the 200-sample seed is unchanged)
        seed = bp.sample_seed(
            exp, label if size == N_SAMPLE else f"{label}|stratified-{size}"
        )
        items = bp.draw_sample(pool["items"], size, seed)
    ids = [x["item_id"] for x in items]
    rec = {
        "dataset": ds,
        "kind": "pool-sample",
        "label": label,
        "seed": seed,
        "size": len(ids),
        "pool_hash": pool["pool_hash"],
        "splits": pool["splits"],
        "stratified_by": None if uniform else pool["stratified_by"],
        "sampling": "uniform" if uniform else "stratified",
        "sample_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
        "panel_hash": pool["pool_hash"],
        "items": items,
    }
    p = base / "samples" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = json.load(open(p))
        if (
            old["sample_hash"] != rec["sample_hash"]
            or old["pool_hash"] != rec["pool_hash"]
        ):
            raise SystemExit(
                f"{p} differs from the redrawn sample (pool or seed changed)"
            )
    else:
        save(p, rec)
    return {
        **{k: rec[k] for k in ("label", "seed", "sample_hash")},
        **({"sampling": "uniform", "size": size} if uniform else {}),
        **(
            {"sampling": "stratified", "size": size}
            if not uniform and size != N_SAMPLE
            else {}
        ),
        "file": str(p),
        "ids": ids,
    }


# ------------------------------------------------------------------ sample size (sample-size-v1, user 2026-10-22)
# ALFWorld v7 from round 6 (ScienceWorld later from its round 1): every NEW configuration runs ``size`` tasks drawn
# uniformly at random from the whole train pool (independently per configuration; seed recorded), staged
# (STAGE_SIZES below ``size``) + ``size`` (100 -> 50 / 100); the pruning rule is unchanged. Estimates of a uniform
# sample: plain mean / SE (unpaired_stats.simple, FPC = pool size) -- see its docstring for why not post-
# stratification. Earlier configurations keep their 200 stratified samples and the stratified estimator; the GP
# uses per-configuration means and SEs, so mixed n is fine. Opt-in POOL_SAMPLE_SIZE=<n> + POOL_SAMPLE_UNIFORM=1.
SAMPLE_SIZE_PROTOCOL = "sample-size-v1"
SAMPLE_SCHEDULE_KEY = "sample_schedule"  # state.json key ("sample_size" is the run's original identity field)
SAMPLE_SIZE_ENV = "POOL_SAMPLE_SIZE"
SAMPLE_UNIFORM_ENV = "POOL_SAMPLE_UNIFORM"


def sample_spec(state: dict, r: int) -> tuple[int, bool]:
    """(size, uniform) of the samples of configurations proposed in round ``r``."""
    blk = state.get(SAMPLE_SCHEDULE_KEY)
    if blk and r >= blk["from_round"]:
        return blk["size"], blk["sampling"] == "uniform"
    return state.get("full_panel_size", N_SAMPLE), False


def cfg_sizes(state: dict, c: dict) -> list[int]:
    """Stage sizes of one configuration: the recorded stage sizes below its sample size, then its sample size."""
    n = len(c["sample"]["ids"])
    return [s for s in state["staging"]["sizes"] if s < n] + [n]


def apply_sample_kinds(state: dict) -> None:
    """Configurations with uniform samples use the simple estimator (session_bo_agentenv.use_uniform_samples)."""
    bo_module().use_uniform_samples(
        {
            space.key(c["config"])
            for rr in state.get("rounds", [])
            for c in rr["configs"]
            if c.get("sample", {}).get("sampling") == "uniform"
        }
    )


def sample_size_block(state: dict, a, now: str) -> dict | None:
    """Record sample-size-v1 once (POOL_SAMPLE_SIZE / --sample_size + POOL_SAMPLE_UNIFORM / --sample_uniform):
    from the last proposed round when its configurations already carry samples of this spec and have run no
    episode (re-sampled by pool_discard_round --resample), else from the next round to be proposed."""
    if state.get("full_panel_size") is not None:
        if (
            getattr(a, "sample_size", None) is not None
            or os.environ.get(SAMPLE_SIZE_ENV, "").strip()
            or os.environ.get(SAMPLE_UNIFORM_ENV, "").strip()
        ):
            raise ValueError(
                "fresh full-panel protocol cannot use sample-size-v1 overrides"
            )
        return None
    size = getattr(a, "sample_size", None)
    if size is None and os.environ.get(SAMPLE_SIZE_ENV, "").strip():
        size = int(os.environ[SAMPLE_SIZE_ENV])
    uniform = bool(getattr(a, "sample_uniform", False)) or os.environ.get(
        SAMPLE_UNIFORM_ENV, ""
    ).strip() in ("1", "true", "yes", "on")
    blk = state.get(SAMPLE_SCHEDULE_KEY)
    if blk is not None:
        if blk.get("protocol") != SAMPLE_SIZE_PROTOCOL:
            raise SystemExit(
                f"state.json sample_schedule {blk.get('protocol')!r}; refusing to mix"
            )
        if size is not None and (size, "uniform" if uniform else "stratified") != (
            blk["size"],
            blk["sampling"],
        ):
            raise SystemExit(
                f"sample size {size} ({'uniform' if uniform else 'stratified'}) requested; state.json records "
                f"{blk['size']} ({blk['sampling']}) from round {blk['from_round']}; refusing to mix"
            )
        return blk
    if size is None:
        return None
    if not 0 < size <= N_SAMPLE:
        raise SystemExit(f"sample size must be in (0, {N_SAMPLE}], not {size}")
    rounds = state["rounds"]
    last = rounds[-1]["configs"] if rounds else []
    fresh = bool(last) and all(
        staging_of(c)["stage_reached"] == 0
        and staging_of(c)["episodes"] == 0
        and len(c["sample"]["ids"]) == size
        and (c["sample"].get("sampling") == "uniform") == uniform
        for c in last
    )
    frm = len(rounds) - 1 if fresh and len(rounds) > 1 else max(1, len(rounds))
    if not rounds and round0_design_from_round0(state):
        frm = 0  # round0-design-v1 (WebShop, user 2026-10-23): a fresh run samples 100 tasks from round 0
    state[SAMPLE_SCHEDULE_KEY] = {
        "protocol": SAMPLE_SIZE_PROTOCOL,
        "size": size,
        "sampling": "uniform" if uniform else "stratified",
        "from_round": frm,
        "stage_sizes": [s for s in state["staging"]["sizes"] if s < size] + [size],
        "seed": f"sample_seed(experiment, config key|uniform-{size})"
        if uniform
        else f"sample_seed(experiment, config key|stratified-{size})"
        if size != N_SAMPLE
        else "sample_seed(experiment, config key)",
        "stage_order": "uniform: a seeded random permutation of the sample (no strata); stratified: unchanged",
        "estimator": "uniform: simple-random-sample mean / SE with FPC = pool size (unpaired_stats.simple); "
        "stratified samples keep the stratified estimator",
        "pruning": "unchanged rule (z = 1.96) at the stages below the sample size",
        "before": {"size": N_SAMPLE, "sampling": "stratified"},
        "opt_in": f"{SAMPLE_SIZE_ENV}={size}"
        + (f" {SAMPLE_UNIFORM_ENV}=1" if uniform else ""),
        "recorded_at": now,
        "note": "user 2026-10-22 (corrected): each configuration runs only 100 train tasks from this round, "
        "stratified by task type as before (uniform sampling is available but not used)",
    }
    return state[SAMPLE_SCHEDULE_KEY]


# round0-design-v1 (user 2026-10-23, WebShop pool run; opt-in, FRESH runs only, recorded once in state.json
# ``round0_design``; untouched states / unset env behave exactly as before):
#   POOL_ROUND0_SOBOL=1      round 0 = N_ROUND0 (10) Sobol draws over the proposable plans (no plan defaults;
#                            ``round0_sobol``: session_pool.sobol_breadth(256, seed 7), NoMemory / duplicates skipped)
#   POOL_FROM_ROUND0=1       sample-size-v1 (POOL_SAMPLE_SIZE) and plan-p1-v1 (POOL_PLAN_P1) apply from round 0
#                            instead of round 1 (P1 proposable in round 0; round-0 samples = POOL_SAMPLE_SIZE tasks)
ROUND0_DESIGN_KEY = "round0_design"
ROUND0_DESIGN_PROTOCOL = "round0-design-v1"
ROUND0_SOBOL_ENV = "POOL_ROUND0_SOBOL"
FROM_ROUND0_ENV = "POOL_FROM_ROUND0"


def _env_on(name: str, env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(name, "").strip() in ("1", "true", "yes", "on")


def round0_design_block(state: dict, now: str, env=None) -> dict | None:
    """Record (fresh state, opt-in) or check state.json ``round0_design``. A request on a state that already has
    rounds without the record, or a request that differs from the record, is refused. Pure on (state, env)."""
    want = {
        "sampler": "sobol-only" if _env_on(ROUND0_SOBOL_ENV, env) else "defaults+sobol",
        "from_round0": _env_on(FROM_ROUND0_ENV, env),
    }
    asked = want["sampler"] == "sobol-only" or want["from_round0"]
    blk = state.get(ROUND0_DESIGN_KEY)
    if blk is not None:
        if asked and {k: blk.get(k) for k in want} != want:
            raise SystemExit(
                f"state.json {ROUND0_DESIGN_KEY} {blk} != requested {want}; refusing to mix"
            )
        return blk
    if not asked:
        return None
    if state.get("rounds"):
        raise SystemExit(
            f"{ROUND0_SOBOL_ENV} / {FROM_ROUND0_ENV} need a fresh run (state.json already has rounds)"
        )
    state[ROUND0_DESIGN_KEY] = blk = {
        "protocol": ROUND0_DESIGN_PROTOCOL,
        **want,
        "round0_size": N_ROUND0,
        "rule": "round 0 = the first N_ROUND0 distinct non-NoMemory configurations of session_pool.sobol_breadth(256, "
        "seed 7) over the round-0 proposable plans (no plan defaults)"
        if want["sampler"] == "sobol-only"
        else "round 0 = plan defaults + Sobol fill (session_bo_agentenv.round0, unchanged)",
        "from_round0_rule": "sample-size-v1 and plan-p1-v1 recorded from round 0 (not round 1)"
        if want["from_round0"]
        else None,
        "recorded_at": now,
        "note": "user 2026-10-23 (WebShop): round 0 = 10 Sobol, P1 in the search space from round 0, 100 tasks per "
        "configuration from round 0",
    }
    return blk


def round0_design_from_round0(state: dict) -> bool:
    return bool((state.get(ROUND0_DESIGN_KEY) or {}).get("from_round0"))


def round0_sobol(bo, n: int = N_ROUND0, seed: int = 0) -> list[dict]:
    """Round 0 of round0-design-v1 sobol-only: ``n`` distinct non-NoMemory Sobol configurations over the currently
    proposable plans (``bo.pool.RETRIEVAL_PLANS``), in draw order. Deterministic."""
    out, seen = [], set()
    for c in bo.pool.sobol_breadth(256, seed + 7):
        if c["plan"] == "NoMemory" or space.key(c) in seen:
            continue
        out.append(c)
        seen.add(space.key(c))
        if len(out) == n:
            return out
    raise ValueError(f"Sobol breadth yielded only {len(out)} distinct configurations")


ROUND0_PREFIX_ENV = "POOL_ROUND0_PREFIX"


def round0_prefix_check(state: dict, env=None) -> dict | None:
    """round0-prefix-v1 (user 2026-10-22, ScienceWorld; recorded by ``pool_round0_prefix``): round-0 samples are the
    stage-``size`` prefix of their stratified 200-sample. A start asking for POOL_ROUND0_PREFIX=<n> needs the record
    (and the same n); a record must match the round-0 samples. Pure."""
    env = os.environ if env is None else env
    want = env.get(ROUND0_PREFIX_ENV, "").strip()
    blk = state.get("round0_prefix")
    if blk is None:
        if want:
            raise SystemExit(
                f"{ROUND0_PREFIX_ENV}={want} but state.json has no round0_prefix: run "
                f"`python -m trimem.experiments.pool_round0_prefix --dataset <ds> --size {want}` first"
            )
        return None
    if want and int(want) != blk["size"]:
        raise SystemExit(
            f"{ROUND0_PREFIX_ENV}={want} but state.json round0_prefix records {blk['size']}; refusing"
        )
    bad = [
        cfg_name(c["config"])
        for c in state["rounds"][0]["configs"]
        if len(c["sample"]["ids"]) != blk["size"]
    ]
    if bad:
        raise SystemExit(
            f"round-0 samples of {bad} are not the recorded {blk['size']}-task prefix"
        )
    return blk


def cfg_name(cfg: dict) -> str:
    """Directory / sample name of a configuration (v2 key hash; ``drv.cfg_dir`` validates with v1)."""
    return "cfg_" + hashlib.sha256(space.key(cfg).encode()).hexdigest()[:12]


_PLANS_NOW: list = []  # the proposable plans set by plan_round (plan-p1-v1); re-applied after every use_space


def bo_module():
    """``session_bo_agentenv`` switched to session space v2 (botorch import deferred to the stages that fit).
    ``use_space`` resets the proposable plans, so the current round's plans (``_PLANS_NOW``) are re-applied."""
    from trimem.experiments import session_bo_agentenv as bo

    bo.use_space(space)
    if _PLANS_NOW:
        bo.use_plans(_PLANS_NOW[-1])
    return bo


# ------------------------------------------------------------------ env-server guards


_GAMES_SEEN: dict[
    int, int
] = {}  # port -> n_games of the last successful /managed_stats read


def _managed_stats(port: int, tries: int = 5) -> dict:
    """/managed_stats with retries. A busy env server (many live envs) can time out on this probe;
    that must not kill the driver (2026-10-21: three attempts died on a 10 s TimeoutError). After
    ``tries`` failures, fall back to the last successful read of this port, else re-raise."""
    err: Exception | None = None
    for k in range(tries):
        try:
            st = drv.get(f"http://127.0.0.1:{port}/managed_stats", timeout=30)
            # WebShop's server reports its served goal list as ``goals`` (no n_games key)
            _GAMES_SEEN[port] = int(st.get("n_games", st.get("goals", 0)))
            return st
        except (OSError, ValueError) as e:  # TimeoutError / URLError / bad JSON
            err = e
            time.sleep(5 * (k + 1))
    if port in _GAMES_SEEN:
        log(
            f"env server {port}: /managed_stats unreachable ({err!r}); using last read n_games={_GAMES_SEEN[port]}"
        )
        return {"n_games": _GAMES_SEEN[port]}
    raise err  # type: ignore[misc]


def check_games(ds: str, ids: list[str]) -> None:
    """Both env servers must serve every item id of the arm (the pool's appended games included)."""
    need = 1 + max(int(i.rsplit("_", 1)[1]) for i in ids)
    for port in drv.DS[ds]["env"]:
        st = _managed_stats(port)
        if (
            ds == "webshop" and "n_games" not in st
        ):  # WebShop server: ``goals`` = served goals (6,910)
            st = {**st, "n_games": st.get("goals", 0)}
        if st.get("n_games", 0) < need:
            raise SystemExit(
                f"env server {port} serves {st.get('n_games')} games, the arm needs {need} "
                "(restart with scripts/env_servers.sh after build_agentenv_pool)"
            )


def java_guard() -> None:
    n = int(
        subprocess.run(
            ["pgrep", "-c", "-x", "java"], capture_output=True, text=True
        ).stdout.strip()
        or 0
    )
    if n > MAX_JAVA:
        raise SystemExit(
            f"{n} java processes after the arm (> {MAX_JAVA}): JVM leak, stopping"
        )


def run_arm(
    ds: str, name: str, out: Path, sample: dict, *, session: dict | None
) -> int:
    check_games(ds, sample["ids"])
    sa, sb = drv.shards(ds, sample["ids"], out.parent / "_shards" / out.name)
    n = drv.run_arm(ds, name, out, sa, sb, sample["ids"], session=session)
    java_guard()
    return n


# ------------------------------------------------------------------ parallel arms (user 2026-10-19: max concurrency)

# Total concurrent episodes over both GPU halves (vLLM 8030 / 8031). The queue raises vLLM --max-num-seqs to 64
# per server; the env servers' --max-envs to MAX_ENVS per port. ScienceWorld is lower: one JVM per live env.
CONCURRENCY = {
    "alfworld": int(os.environ.get("POOL_CONCURRENCY_ALFWORLD", "80")),
    "sciworld": int(os.environ.get("POOL_CONCURRENCY_SCIWORLD", "60")),
    # WebShop (user 2026-10-23): controlled-c16-v1 from round 0 (2 halves x 8, env 36003 / 36013)
    "webshop": int(os.environ.get("POOL_CONCURRENCY_WEBSHOP", "16")),
}


@dataclass
class Arm:
    name: str
    out: Path
    ids: list[str]
    session: dict | None
    cmd: object = (
        None  # (ds, half, out, shard, *, session) -> argv; default drv.half_cmd
    )
    done: object = None  # (ds, out, expected, session) -> set; default drv.completed
    # max-round-v1 (WebShop, user 2026-10-23): runner --max_round of this arm; None = COMMON's 30 (unchanged argv)
    max_round: int | None = None

    def completed(self, ds: str) -> set:
        return (self.done or drv.completed)(ds, self.out, self.ids, self.session)


def split_halves(
    budget: int, n_arms: int, base: int, load: tuple[int, int] = (0, 0)
) -> list[tuple[int, int]]:
    """Per-arm (half a, half b) processes for ``n_arms`` arms launched together (process-split-v2, user
    2026-10-21: ``total // (2 x arms)`` alone left 128 - 110 = 18 episodes unused at 11 arms). Every half gets
    ``base``; the rest of ``budget`` goes one process at a time to the endpoint half (8030 / 8031) with the lower
    ``load`` (= processes already running there) and, within it, the next arm without an extra -- at most +1 per
    half, so the two endpoints differ by at most one. Pure."""
    out = [[base, base] for _ in range(n_arms)]
    extra = budget - 2 * n_arms * base
    ld = [load[0] + n_arms * base, load[1] + n_arms * base]
    nxt = [0, 0]  # next arm (per half) to receive an extra process
    while extra > 0 and min(nxt) < n_arms:
        h = 0 if nxt[1] >= n_arms or (nxt[0] < n_arms and ld[0] <= ld[1]) else 1
        out[nxt[h]][h] += 1
        nxt[h] += 1
        ld[h] += 1
        extra -= 1
    return [(a, b) for a, b in out]


def _with_processes(cmd: list[str], n: int) -> list[str]:
    cmd = list(cmd)
    i = cmd.index("--processes")
    cmd[i + 1] = str(n)
    return cmd


DRAIN_ENV = "TRIMEM_DRAIN_FILE"  # agentgym_official_eval.DRAIN_ENV (that module needs .venv-agentgym to import)


def drain_path(a: Arm) -> Path:
    return a.out / ".drain"


def remaining(ds: str, a: Arm) -> int:
    """Queries of the arm without a result file (cheap progress measure; completion is ``Arm.completed``)."""
    return len(set(a.ids) - set(drv.item_files(a.out, ds)))


def rebalance_due(per_half: int, left: list[int], concurrency: int) -> bool:
    """Drain the still-running arms (in-flight episodes finish, nothing is lost) and relaunch them with more
    processes when (a) relaunching would at least double each arm's processes per half and (b) some arm has more
    unfinished queries than it has episodes in flight (otherwise its tail is already fully parallel). ``left`` =
    unfinished queries of each still-running arm. Pure."""
    n = len(left)
    if n == 0:
        return False
    return concurrency // (2 * n) >= 2 * per_half and any(
        x > 2 * per_half for x in left
    )


def _attempt(ds: str, todo: list[Arm], per_half: int, concurrency: int, *, allow_rebalance: bool = True) -> bool:
    """One attempt: both halves of every unfinished arm, ``per_half`` processes each; per-arm stall watchdog. When
    arms finish early (a stage of some configurations done, a round's fast arms done) the others are DRAINED and
    the attempt ends, so the caller relaunches them with the freed slots (``rebalance_due``). Returns True when the
    attempt ended by such a drain (not a failed attempt)."""
    quota_check = None
    resource_monitor = None
    source_check = None
    if not allow_rebalance:
        from trimem.experiments.scienceworld_trace_ablation import RuntimeResourceMonitor, check_global_quota

        quota_check = check_global_quota
        quota_check()
        raw_limits = os.environ.get("TRIMEM_ABLATION_SYSTEM_LIMITS")
        if not raw_limits:
            raise SystemExit("ablation system limits missing; refusing dispatch")
        resource_monitor = RuntimeResourceMonitor(json.loads(raw_limits))
        resource_monitor.check()
        raw_markers = os.environ.get("TRIMEM_ABLATION_SOURCE_PROCESS_MARKERS")
        if raw_markers:
            from tools.scienceworld_ablation_initialization_run import check_source_idle

            markers = json.loads(raw_markers)
            source_check = lambda: check_source_idle(markers)
            source_check()
    last_quota_check = time.monotonic()
    env = {**os.environ, "PYTHONPATH": str(ROOT), mp.ENV_VAR: mp.current()}
    env.pop("TRIMEM_ABLATION_SOURCE_PROCESS_MARKERS", None)
    running: dict[str, list] = {}
    halves = split_halves(concurrency, len(todo), per_half)
    started = []
    try:
        for a, hv in zip(todo, halves, strict=True):
            drain_path(a).unlink(missing_ok=True)
            sa, sb = drv.shards(ds, a.ids, a.out.parent / "_shards" / a.out.name)
            procs = []
            for half, shard in enumerate((sa, sb)):
                if not json.loads(shard.read_text()):
                    continue
                cmd = (a.cmd or drv.half_cmd)(ds, half, a.out, shard, session=a.session)
                cmd = [
                    "taskset", "-c", "16-31", "nice", "-n", "10",
                    *eb.with_max_round(_with_processes(cmd, hv[half]), a.max_round),
                ]
                with open(a.out / f"half_{'ab'[half]}.log", "a") as logf:
                    proc = subprocess.Popen(
                        cmd, cwd=ROOT, env={**env, DRAIN_ENV: str(drain_path(a))},
                        stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                started.append(proc)
                procs.append(proc)
            running[a.name] = procs
    except BaseException:
        for proc in started:
            drv.kill_group(proc)
        for proc in started:
            proc.wait()
        raise
    progress = {a.name: (len(drv.item_files(a.out, ds)), time.time()) for a in todo}
    drained = False
    quota_pause = None
    try:
        while any(p.poll() is None for ps in running.values() for p in ps):
            time.sleep(20)
            if quota_check is not None and quota_pause is None and time.monotonic() - last_quota_check >= 30:
                try:
                    quota_check()
                    resource_monitor.check()
                    if source_check is not None:
                        source_check()
                except SystemExit as exc:
                    quota_pause = exc
                    for arm in todo:
                        drain_path(arm).touch()
                    drained = True
                    log(f"ablation quota or resource pause: draining this run's workers: {exc}")
                last_quota_check = time.monotonic()
            alive = [a for a in todo if any(p.poll() is None for p in running[a.name])]
            if (
                allow_rebalance
                and not drained
                and len(alive) < len(todo)
                and rebalance_due(
                    per_half, [remaining(ds, a) for a in alive], concurrency
                )
            ):
                log(
                    f"rebalance: {len(alive)} arms left at {per_half} processes per half; draining them (in-flight "
                    f"episodes finish) to relaunch at {concurrency // (2 * len(alive))}"
                )
                for a in alive:
                    drain_path(a).touch()
                drained = True
            for a in alive:
                n = len(drv.item_files(a.out, ds))
                if n != progress[a.name][0]:
                    progress[a.name] = (n, time.time())
                elif time.time() - progress[a.name][1] > drv.STALL_S:
                    log(
                        f"stall: {a.name} no new item for {drv.STALL_S}s at {n}/{len(a.ids)}; stopping that arm"
                    )
                    for p in running[a.name]:
                        drv.kill_group(p)
    finally:
        for ps in running.values():
            for p in ps:
                drv.kill_group(p)
            for p in ps:
                p.wait()
        for a in todo:
            drain_path(a).unlink(missing_ok=True)
    if quota_pause is not None:
        raise quota_pause
    return drained


def run_parallel(
    ds: str, arms: list[Arm], concurrency: int | None = None, *, allow_restart: bool = True
) -> dict[str, int]:
    """Run independent arms CONCURRENTLY (both GPU halves each), ``concurrency`` episodes in total split
    evenly over the unfinished arms' halves; as arms finish, the others are drained and relaunched with the freed
    slots (drains are not failed attempts); per-arm stall watchdog (a stalled arm is stopped, the others go on);
    after every attempt all process groups are reaped and the env servers checked + restarted once.
    Resumable exactly like ``drv.run_arm`` (result file + one evaluated opt row)."""
    concurrency = concurrency or CONCURRENCY[ds]
    for a in arms:
        check_games(ds, a.ids)
        a.out.mkdir(parents=True, exist_ok=True)
    failures = drains = 0
    while failures < drv.ATTEMPTS:
        todo = [a for a in arms if len(a.completed(ds)) < len(a.ids)]
        if not todo:
            break
        per_half = max(1, concurrency // (2 * len(todo)))
        log(
            f"parallel attempt (failed so far {failures}, drains {drains}): {len(todo)} arms x 2 halves x {per_half} "
            f"processes ({2 * per_half * len(todo)} concurrent episodes): "
            + ", ".join(a.name for a in todo)
        )
        drained_attempt = (_attempt(ds, todo, per_half, concurrency) if allow_restart
                           else _attempt(ds, todo, per_half, concurrency, allow_rebalance=False))
        if drained_attempt:
            drains += 1
        else:
            failures += 1
        if allow_restart:
            drv.env_check_and_restart(ds)
        elif not env_servers_healthy(ds):
            raise SystemExit("environment unhealthy; ablation paused without restarting shared services")
        if not allow_restart and any(len(a.completed(ds)) < len(a.ids) for a in arms):
            raise SystemExit("ablation arm exited incomplete; resume after retry-budget checks")
        java_guard()
    out = {a.name: len(a.completed(ds)) for a in arms}
    log(
        "parallel done: "
        + ", ".join(
            f"{k} {v}/{len(a.ids)}" for (k, v), a in zip(out.items(), arms, strict=True)
        )
    )
    return out


# Test stage: total concurrent episodes. User 2026-10-20: 40 for ALL three datasets (τ² already 40). The reused
# controlled-v1 ALF/Sci test arms (results/eval_controlled_20261015: NoMemory = the paired baseline, VGR) ran at
# 16 (one arm at a time, 2 halves x 8 processes) -- recorded in the routing table / report as
# ``reused_arms_concurrency``. Capacity checked 2026-10-20: env servers 2 ports x MAX_ENVS (ALF 64, Sci 40) live
# envs >= 40; ALF profiling already ran 72 concurrent; Sci JVMs -Xmx1g (~0.4 GB RSS) x 40 ~= 17 GB within the two
# 24G server scopes and the 100G trimem-eval.slice.
TEST_CONCURRENCY = {
    "alfworld": int(os.environ.get("POOL_TEST_CONCURRENCY_ALFWORLD", "40")),
    "sciworld": int(os.environ.get("POOL_TEST_CONCURRENCY_SCIWORLD", "40")),
    "webshop": int(os.environ.get("POOL_TEST_CONCURRENCY_WEBSHOP", "16")),
}
REUSED_TEST_CONCURRENCY = (
    16  # the controlled-v1 test arms reused as NoMemory / VGR references
)
# concurrency-v1 (user 2026-10-21: ALF / Sci profiling AND test concurrency 80-100, set by the queue through
# POOL_CONCURRENCY_<DS> / POOL_TEST_CONCURRENCY_<DS>): recorded once in state.json ``concurrency`` (as τ²), with
# the endpoint routing and the latency caveat; a resume under another regime is refused.
CONCURRENCY_PROTOCOL = "concurrency-v1"
SEMANTIC_ROUTING_ENV = (
    "TRIMEM_SEMANTIC_ROUTING"  # agentgym_session_eval.SEMANTIC_ROUTING_ENV
)


def concurrency_block(ds: str, state: dict) -> dict:
    routing = (
        "request-load-v1"
        if os.environ.get(SEMANTIC_ROUTING_ENV, "").strip() == "request-load"
        else "episode"
    )
    want = {
        "protocol": CONCURRENCY_PROTOCOL,
        "profiling_total": CONCURRENCY[ds],
        "test_total": TEST_CONCURRENCY[ds],
        "reused_arms_concurrency": REUSED_TEST_CONCURRENCY,
        "semantic_routing": routing,
    }
    blk = state.get("concurrency")
    if blk is None:
        state["concurrency"] = blk = {
            **want,
            "from_round": len(state["rounds"]),
            "recorded_at": datetime.now(UTC).isoformat(),
            "reason": "user 2026-10-21 option A: runtime profile "
            + os.environ.get(RUNTIME_PROFILE_ENV, "").strip()
            if os.environ.get(RUNTIME_PROFILE_ENV, "").strip()
            else "user 2026-10-21: raise ALF/Sci profiling and test concurrency to 80-100",
            "episode_endpoints": "static per half: half a -> vLLM 8030 + env port 36001/36002, half b -> vLLM "
            "8031 + env 36011/36012; each arm's sample interleaved over the halves, equal processes per half",
            "semantic_routing_note": "request-load-v1 (as τ²): SemFilter judge calls go to the endpoint with the "
            "fewest running + waiting requests (vLLM metrics); agent calls stay on the episode endpoint"
            if routing != "episode"
            else "episode: SemFilter judge calls go to the episode's own endpoint",
            "latency_caveat": f"test latency at concurrency {TEST_CONCURRENCY[ds]} is NOT "
            f"comparable to the reused controlled-v1 reference arms (NoMemory / VGR), which ran at "
            f"{REUSED_TEST_CONCURRENCY}; profiling latency is measured under ~{CONCURRENCY[ds]} concurrent "
            "episodes (vLLM max-num-seqs 64 per GPU, so requests queue)"
            if TEST_CONCURRENCY[ds] != REUSED_TEST_CONCURRENCY
            or CONCURRENCY[ds] != REUSED_TEST_CONCURRENCY
            else f"profiling and test at {REUSED_TEST_CONCURRENCY} = the reused controlled-v1 reference arms' "
            "concurrency (2 halves x 8): latencies comparable (see runtime_profile)",
        }
    elif {k: blk.get(k) for k in want} != want:
        if not concurrency_change_allowed(blk, want):
            raise SystemExit(
                f"state.json concurrency block {blk} != this run {want}; refusing to mix (an explicit "
                f"profiling/test concurrency change needs {CONCURRENCY_CHANGE_ENV}=1 and "
                f"{CONCURRENCY_CHANGE_REASON_ENV}=<reason>)"
            )
        record_concurrency_change(blk, want, state)
    return blk


# Explicit, recorded concurrency regime change (user 2026-10-21: ALF 96 -> 128 after vLLM max-num-seqs 128). Only
# the profiling / test totals may change, only with POOL_CONCURRENCY_CHANGE=1 and a non-empty reason; every change
# is appended to the block's ``changes`` (from / to / at / round + stage progress / reason) and the block's totals
# become the new regime, so later resumes compare against it. Default (flag unset): refusal, as before.
CONCURRENCY_CHANGE_ENV = "POOL_CONCURRENCY_CHANGE"
CONCURRENCY_CHANGE_REASON_ENV = "POOL_CONCURRENCY_CHANGE_REASON"
# latency-regime-v1 (user 2026-10-22): the SemFilter routing may change together with the totals (c128
# request-load -> c16 episode routing)
CONCURRENCY_CHANGEABLE = ("profiling_total", "test_total", "semantic_routing")


def concurrency_change_allowed(blk: dict, want: dict) -> bool:
    """True when the explicit change flag + reason are set and only the changeable totals differ. Pure."""
    if os.environ.get(CONCURRENCY_CHANGE_ENV, "").strip() != "1":
        return False
    if not os.environ.get(CONCURRENCY_CHANGE_REASON_ENV, "").strip():
        return False
    return all(
        blk.get(k) == v for k, v in want.items() if k not in CONCURRENCY_CHANGEABLE
    )


def record_concurrency_change(blk: dict, want: dict, state: dict) -> dict:
    """Append the change entry to ``blk['changes']`` and switch the block's totals to ``want``. Returns the entry."""
    rounds = state.get("rounds", [])
    last = rounds[-1] if rounds else {}
    entry = {
        "from": {k: blk.get(k) for k in CONCURRENCY_CHANGEABLE},
        "to": {k: want[k] for k in CONCURRENCY_CHANGEABLE},
        "at": datetime.now(UTC).isoformat(),
        "from_round": max(0, len(rounds) - 1),
        "stage_progress": [
            {
                "config": j,
                "stage_reached": c.get("staging", {}).get("stage_reached", 0),
                "pruned_at": c.get("staging", {}).get("pruned_at"),
            }
            for j, c in enumerate(last.get("configs", []))
        ],
        "reason": os.environ[CONCURRENCY_CHANGE_REASON_ENV].strip(),
        "note": "episodes finished before 'at' ran under 'from'; the round's latency mixes both regimes",
    }
    blk.setdefault("changes", []).append(entry)
    blk.update(entry["to"])
    log(
        f"concurrency change recorded: {entry['from']} -> {entry['to']} at round {entry['from_round']} "
        f"({entry['reason']})"
    )
    return entry


# Runtime profile (user decision 2026-10-21, option A): ScienceWorld profiling AND test run under the SAME runtime
# as the controlled-v1 reference arms and the ALFWorld c16 test rerun (results/eval_controlled_20261015
# run_controlled.sh / run_args.json; results/pool_v5_20261020/run_alf_test_c16.sh). Opt-in via RUNTIME_PROFILE_ENV;
# recorded once in state.json ``runtime_profile`` and every later stage (profile resume, test) under another
# regime -- or without the env -- is refused. The runtime is checked against the process env (pure).
RUNTIME_PROFILE_ENV = "POOL_RUNTIME_PROFILE"
RUNTIME_PROFILE_C16 = "controlled-c16-v1"
RUNTIME_PROFILE_FORBIDDEN_ENV = (  # must be UNSET: the reference ran with the defaults
    "TRIMEM_SEMANTIC_ROUTING",
    "TRIMEM_SEMANTIC_BASES",
    "TRIMEM_SEMANTIC_WORKERS",
    "TRIMEM_RETRIEVAL_SLOTS",
    "TRIMEM_SEMFILTER_SLOTS",
)
RUNTIME_PROFILES = {
    RUNTIME_PROFILE_C16: {
        "total_concurrency": 16,
        "layout": "2 halves x 8 processes: half a -> vLLM 8030 + env port a, half b -> vLLM 8031 + env port b; "
        "<= 16 in-flight episodes over all arms (async profiling: arms beyond the free slots queue, "
        "launch_halves; sync stages / test: run_capped, <= 8 arms per group)",
        "semantic_routing": "episode (TRIMEM_SEMANTIC_ROUTING unset): SemFilter judge calls on the half's own GPU",
        "semantic_workers": "default (agentgym_session_eval, 4)",
        "retrieval_slots": "default (agentgym_session_eval, 16)",
        "max_envs_per_port": 32,
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 4096,
            "max_round": 30,
            "timeout": 2400,
            "thinking": "disabled (chat_template_kwargs.enable_thinking=false)",
        },
        "latency": "wall = episode wall + retrieval ms (episodes.jsonl / opt-row e2e_ms, used by the optimizer "
        "and the report); LLM span (first to last call, llm_calls.jsonl) recorded too",
        "vllm": "8030 / 8031 as running: max-num-seqs 128, max-model-len 262144 (not restarted)",
        "reference": "results/eval_controlled_20261015 (run_controlled.sh, run_args.json); ALFWorld c16 test "
        "rerun results/pool_v5_20261020/run_alf_test_c16.sh",
    }
}


def runtime_profile_want(ds: str) -> dict | None:
    """The runtime profile requested by ``$POOL_RUNTIME_PROFILE`` (None when unset), after checking that this
    process's runtime matches it (SystemExit otherwise). Pure on the env and module settings."""
    name = os.environ.get(RUNTIME_PROFILE_ENV, "").strip()
    if not name:
        return None
    if name not in RUNTIME_PROFILES:
        raise SystemExit(
            f"{RUNTIME_PROFILE_ENV}={name!r} unknown (known: {sorted(RUNTIME_PROFILES)})"
        )
    prof = RUNTIME_PROFILES[name]
    bad = [k for k in RUNTIME_PROFILE_FORBIDDEN_ENV if os.environ.get(k, "").strip()]
    n = prof["total_concurrency"]
    if CONCURRENCY[ds] != n or TEST_CONCURRENCY[ds] != n:
        bad.append(
            f"concurrency profiling {CONCURRENCY[ds]} / test {TEST_CONCURRENCY[ds]} != {n}"
        )
    envs = os.environ.get("MAX_ENVS", "").strip()
    if envs and int(envs) != prof["max_envs_per_port"]:
        bad.append(f"MAX_ENVS {envs} != {prof['max_envs_per_port']}")
    c = drv.COMMON
    s = prof["sampling"]
    if (
        c[c.index("--temperature") + 1] != str(s["temperature"])
        or c[c.index("--max_round") + 1] != str(s["max_round"])
        or any(f in c for f in ("--top_p", "--max_tokens", "--timeout"))
    ):
        bad.append(f"runner sampling args {c} differ from {s}")
    if bad:
        raise SystemExit(f"runtime profile {name}: this process does not match: {bad}")
    return {"name": name, **prof}


def runtime_profile_block(ds: str, state: dict, *, record: bool = True) -> dict | None:
    """Record (once, ``record``) or verify state.json ``runtime_profile`` against this run's profile. A state with a
    profile resumed without one (or with another), or a profile requested on a state that already has rounds
    without one, is refused."""
    want = runtime_profile_want(ds)
    blk = state.get("runtime_profile")
    if blk is None:
        if want is None:
            return None
        if state.get("rounds") or not record:
            raise SystemExit(
                f"runtime profile {want['name']} requested but state.json has rounds (or a later stage) without "
                "one; refusing to mix"
            )
        state["runtime_profile"] = blk = {
            **want,
            "recorded_at": datetime.now(UTC).isoformat(),
            "reason": "user 2026-10-21 option A: Sci profiling and test under the controlled-v1 / ALF c16 runtime",
        }
        return blk
    if want is None or {k: blk.get(k) for k in want} != want:
        raise SystemExit(
            f"state.json runtime_profile {blk.get('name')!r} != this run "
            f"{want and want['name']!r} (or its settings differ); refusing to mix"
        )
    return blk


def channel_slots_want(a) -> str:
    """The channel-slot protocol this (re)start asks for: --channel_slots, else POOL_CHANNEL_SLOTS, else the
    default (``CHANNEL_SLOTS_TAG``; a state that records a protocol keeps it)."""
    v = (
        getattr(a, "channel_slots", None)
        or os.environ.get(CHANNEL_SLOTS_ENV, "").strip()
    )
    if not v:
        return CHANNEL_SLOTS_TAG
    if v not in CHANNEL_SLOTS_CHOICES and v not in CHANNEL_SLOTS_CHOICES.values():
        raise SystemExit(
            f"{CHANNEL_SLOTS_ENV}={v!r}: unknown value (expected one of {sorted(CHANNEL_SLOTS_CHOICES)})"
        )
    return CHANNEL_SLOTS_CHOICES.get(v, v)


def channel_slots_record(state: dict, a, now: str) -> dict:
    """Record / check the channel-slot protocol (``session_bo_agentenv.channel_slots_block``). An explicit request
    for an older protocol than the recorded channel-slots-v3 / -v4 is refused (never downgraded silently); the one
    allowed change of a v3 state is the explicit upgrade to v4 (from the next round to be proposed)."""
    bo = bo_module()
    want = channel_slots_want(a)
    explicit = bool(
        getattr(a, "channel_slots", None)
        or os.environ.get(CHANNEL_SLOTS_ENV, "").strip()
    )
    rec = state.get("channel_slots", {}).get("protocol")
    # explicit upgrades only along v3 -> v4 -> single-pick-4ch-v1 (from older protocols to any target)
    rank = {
        bo.CHANNEL_SLOTS_V3: 0,
        bo.CHANNEL_SLOTS_V4: 1,
        bo.CHANNEL_SLOTS_V5: 2,
        bo.CHANNEL_SLOTS_V6: 3,
    }
    upgrade = rec in rank and want in rank and rank[want] > rank[rec]
    if explicit and rec in rank and want != rec and not upgrade:
        raise SystemExit(
            f"state.json records {rec} (from round {state['channel_slots']['from_round']}); {want} requested; "
            "refusing to mix"
        )
    return bo.channel_slots_block(
        state, now, protocol=want, sobol_until=random_rounds_until(a, state)
    )


def random_rounds_until(a, state: dict | None = None) -> int | None:
    """sobol-only-v1 opt-in -> the last random-sampling round: --random_rounds_until / POOL_RANDOM_ROUNDS_UNTIL
    (absolute), else --random_rounds / POOL_RANDOM_ROUNDS = N (the N rounds from the recorded sobol-only block's
    first round, or from the next round to be proposed when none is recorded yet), else None."""
    v = getattr(a, "random_rounds_until", None)
    if v is None and os.environ.get(RANDOM_ROUNDS_ENV, "").strip():
        v = int(os.environ[RANDOM_ROUNDS_ENV])
    n = getattr(a, "random_rounds", None)
    if n is None and os.environ.get(RANDOM_ROUNDS_N_ENV, "").strip():
        n = int(os.environ[RANDOM_ROUNDS_N_ENV])
    if n is None or state is None:
        return v
    if n < 1:
        raise SystemExit(f"random rounds must be >= 1, not {n}")
    rng = bo_module().recorded_sobol_range(state)
    first = rng[0] if rng else max(1, len(state["rounds"]))
    u = first + n - 1
    if v is not None and v != u:
        raise SystemExit(f"random rounds {n} (until {u}) conflicts with until {v}")
    return u


BATCH_STEPS_ENV = "POOL_BATCH_STEPS"  # e.g. "10@6,4@7": batch 10 from round 6, 4 from round 7 (user 2026-10-22)


def _batch_request(a, state: dict) -> list[tuple[int, int]] | None:
    """The requested batch steps [(from_round, size), ...]: --batch_steps / POOL_BATCH_STEPS ("size@round,..."),
    else --round_batch / POOL_BATCH (one step from the next round to be proposed), else None."""
    steps = (
        getattr(a, "batch_steps", None) or os.environ.get(BATCH_STEPS_ENV, "").strip()
    )
    if steps:
        out = []
        for part in steps.split(","):
            size, _, rnd = part.strip().partition("@")
            out.append((int(rnd), int(size)))
        if [r for r, _ in out] != sorted({r for r, _ in out}):
            raise SystemExit(
                f"batch steps {steps!r}: rounds must be strictly increasing"
            )
        if any(n < 1 for _, n in out):
            raise SystemExit(f"batch steps {steps!r}: sizes must be >= 1")
        return out
    v = getattr(a, "round_batch", None)
    if v is None and os.environ.get(BATCH_ENV, "").strip():
        v = int(os.environ[BATCH_ENV])
    if v is None:
        return None
    if v < 1:
        raise SystemExit(f"round batch must be >= 1, not {v}")
    return [(max(1, len(state["rounds"])), v)]


def _size_at(steps: list[tuple[int, int]], r: int, default: int) -> int:
    """Batch size of round ``r`` under ``steps`` [(from_round, size), ...] (``default`` before the first). Pure."""
    size = default
    for r0, n in steps:
        if r >= r0:
            size = n
    return size


def _recorded_steps(blk: dict) -> list[tuple[int, int]]:
    return [tuple(x) for x in blk.get("steps") or [(blk["from_round"], blk["size"])]]


def batch_schedule_block(state: dict, a, now: str) -> dict | None:
    """Per-round batch size from the next round to be proposed (state.json ``batch_schedule``; user 2026-10-22):
    recorded once when --round_batch / POOL_BATCH (one size) or --batch_steps / POOL_BATCH_STEPS (sizes per round
    range, e.g. 10@6,4@7) asks for it; afterwards the recorded schedule holds (the env vars may be unset). A
    conflicting request is refused, unless no round was proposed under the recorded schedule yet (e.g. after
    ``pool_discard_round``): then it is re-recorded and the unused one kept in ``replaced_unused``. The staged budget
    is unchanged: ``staged_pruning.next_batch_size`` caps every round at what the remaining budget can evaluate in
    full."""
    req = _batch_request(a, state)
    blk = state.get("batch_schedule")
    if blk is not None:
        if blk.get("protocol") != BATCH_SCHEDULE_PROTOCOL:
            raise SystemExit(
                f"state.json batch_schedule {blk.get('protocol')!r}; refusing to mix"
            )
        if (
            req is None
            or req == _recorded_steps(blk)
            or (len(req) == 1 and "steps" not in blk and req[0][1] == blk["size"])
        ):
            return blk
        if blk["from_round"] >= len(state["rounds"]):
            unused = {k: x for k, x in blk.items() if k != "replaced_unused"}
            state.pop("batch_schedule")
            new = batch_schedule_block(state, a, now)
            if new is not None:
                new["replaced_unused"] = [*blk.get("replaced_unused", []), unused]
            return new
        n = len(state["rounds"])
        if all(
            _size_at(req, r, state["batch"]) == round_batch(state, r, state["batch"])
            for r in range(n)
        ):
            # user 2026-10-22 (ScienceWorld v6 from round 3): a change that leaves every proposed round's batch as
            # it was applies from the next round to be proposed; the replaced record goes to ``history``
            hist = [
                *blk.get("history", []),
                {k: x for k, x in blk.items() if k != "history"},
            ]
            state["batch_schedule"] = {
                **{
                    k: x
                    for k, x in blk.items()
                    if k not in ("history", "replaced_unused")
                },
                "size": req[0][1],
                "from_round": req[0][0],
                "steps": [list(x) for x in req],
                "opt_in": f"{BATCH_STEPS_ENV}={','.join(f'{m}@{r}' for r, m in req)}",
                "recorded_at": now,
                "changed_from_round": n,
                "history": hist,
                **(
                    {"replaced_unused": blk["replaced_unused"]}
                    if "replaced_unused" in blk
                    else {}
                ),
            }
            return state["batch_schedule"]
        raise SystemExit(
            f"round batch {req} requested; state.json batch_schedule records {_recorded_steps(blk)}; "
            "refusing to mix (it would change the batch of an already proposed round)"
        )
    if req is None or (len(req) == 1 and req[0][1] == state["batch"]):
        return None
    if req[0][0] < len(state["rounds"]):
        raise SystemExit(
            f"batch steps start at round {req[0][0]}, but round {len(state['rounds']) - 1} is already proposed"
        )
    state["batch_schedule"] = {
        "protocol": BATCH_SCHEDULE_PROTOCOL,
        "size": req[0][1],
        "from_round": req[0][0],
        **({"steps": [list(x) for x in req]} if len(req) > 1 else {}),
        "before": state["batch"],
        "opt_in": f"{BATCH_STEPS_ENV}={','.join(f'{n}@{r}' for r, n in req)}"
        if len(req) > 1
        else f"{BATCH_ENV}={req[0][1]} / --round_batch {req[0][1]}",
        "budget": "unchanged (state.json staging / budget_extension); a round starts at most min(size, max_configs "
        "- configurations so far, remaining budget // full_cost) configurations",
        "recorded_at": now,
        "note": "user 2026-10-22; run identity ``batch`` unchanged",
    }
    return state["batch_schedule"]


BUDGET_EXT_PROTOCOL = "budget-extension-v1"
BUDGET_EPISODES_ENV = "POOL_BUDGET_EPISODES"
BUDGET_MAX_CONFIGS_ENV = "POOL_BUDGET_MAX_CONFIGS"
BUDGET_REASON_ENV = "POOL_BUDGET_REASON"


def _budget_request(a) -> tuple[int | None, int | None, str | None]:
    ep = getattr(a, "budget_episodes", None)
    if ep is None and os.environ.get(BUDGET_EPISODES_ENV, "").strip():
        ep = int(os.environ[BUDGET_EPISODES_ENV])
    mc = getattr(a, "budget_max_configs", None)
    if mc is None and os.environ.get(BUDGET_MAX_CONFIGS_ENV, "").strip():
        mc = int(os.environ[BUDGET_MAX_CONFIGS_ENV])
    why = (
        getattr(a, "budget_reason", None)
        or os.environ.get(BUDGET_REASON_ENV, "").strip()
    )
    return ep, mc, why or None


def effective_budget(state: dict) -> sp.Budget:
    """The profiling budget: the last budget-extension-v1 entry (state.json ``budget_extension.history``), else the
    recorded staging block. ``full_cost`` is always the staging block's."""
    g = state["staging"]
    hist = (state.get("budget_extension") or {}).get("history") or []
    if hist:
        return sp.Budget(
            hist[-1]["budget_episodes"], g["full_cost"], hist[-1]["max_configs"]
        )
    return sp.Budget(g["budget_episodes"], g["full_cost"], g["max_configs"])


def budget_extension_block(state: dict, a, now: str) -> dict | None:
    """budget-extension-v1 (user 2026-10-22): a gated, recorded increase of the profiling budget (episodes and / or
    max_configs) from the next round to be proposed. The staging block -- and its resume consistency check against
    the run arguments -- is unchanged; ``effective_budget`` applies the extension. A request equal to the current
    effective budget is a no-op (restarts); shrinking, or going below what is already used / proposed, is refused;
    a new entry needs a reason (--budget_reason / POOL_BUDGET_REASON)."""
    ep, mc, why = _budget_request(a)
    blk = state.get("budget_extension")
    if blk is not None and blk.get("protocol") != BUDGET_EXT_PROTOCOL:
        raise SystemExit(
            f"state.json budget_extension {blk.get('protocol')!r}; refusing to mix"
        )
    if ep is None and mc is None:
        return blk
    cur = effective_budget(state)
    ep = cur.total_episodes if ep is None else ep
    mc = cur.max_configs if mc is None else mc
    if (ep, mc) == (cur.total_episodes, cur.max_configs):
        return blk
    n_cfg = sum(len(rr["configs"]) for rr in state["rounds"])
    if ep < cur.total_episodes or mc < cur.max_configs:
        raise SystemExit(
            f"budget {ep} episodes / {mc} configurations requested < the current {cur.total_episodes} / "
            f"{cur.max_configs}; refusing to shrink"
        )
    if ep < episodes_used(state) or mc < n_cfg:
        raise SystemExit(
            "requested budget is below what is already used / proposed; refusing"
        )
    if not why:
        raise SystemExit(
            f"a budget extension needs a reason (--budget_reason / {BUDGET_REASON_ENV})"
        )
    blk = state.setdefault(
        "budget_extension", {"protocol": BUDGET_EXT_PROTOCOL, "history": []}
    )
    blk["history"].append(
        {
            "from_round": max(1, len(state["rounds"])),
            "budget_episodes": ep,
            "max_configs": mc,
            "before": {
                "budget_episodes": cur.total_episodes,
                "max_configs": cur.max_configs,
            },
            "episodes_used_at_record": episodes_used(state),
            "configs_at_record": n_cfg,
            "reason": why,
            "opt_in": f"{BUDGET_EPISODES_ENV}={ep} / {BUDGET_MAX_CONFIGS_ENV}={mc}",
            "rule": "staged_pruning.next_batch_size against the extended budget; full_cost and stage sizes "
            "unchanged; state.json staging (the run identity's unpruned plan) kept as recorded",
            "recorded_at": now,
        }
    )
    return blk


def round_batch(state: dict, r: int, default: int) -> int:
    """Batch size of BO round ``r``: the recorded batch_schedule size from its from_round on, else ``default``."""
    blk = state.get("batch_schedule")
    if not blk:
        return default
    size = default
    for r0, n in _recorded_steps(blk):
        if r >= r0:
            size = n
    return size


def launch_halves(
    total: int, load: tuple[int, int], n_want: int, per_half: int
) -> list[tuple[int, int]]:
    """(half a, half b) processes for the FIRST k of ``n_want`` waiting arms (staged-async loop): every arm needs
    >= 1 process per half, so at most (total - running) // 2 arms start now and the rest queue (they start as
    running arms finish; process-split-v2 spreads what is free). The sum never exceeds ``total`` (user 2026-10-21:
    at concurrency 16 a round has 11-12 arms, more than 16 // 2). Pure."""
    free = total - sum(load)
    k = min(n_want, max(0, free // 2))
    if k == 0:
        return []
    base = max(1, min(per_half, free // (2 * k)))
    return split_halves(free, k, base, load)


def run_capped(ds: str, arms: list[Arm], concurrency: int, *, allow_restart: bool = True) -> dict[str, int]:
    """``run_parallel`` with the TOTAL concurrency never above ``concurrency``: at most concurrency // 2 arms at a
    time (each arm needs >= 1 process per half); within a group finished arms' slots are rebalanced as usual."""
    k = max(1, concurrency // 2)
    out: dict[str, int] = {}
    for i in range(0, len(arms), k):
        if allow_restart:
            out |= run_parallel(ds, arms[i : i + k], concurrency)
        else:
            out |= run_parallel(ds, arms[i : i + k], concurrency, allow_restart=False)
    return out


# ------------------------------------------------------------------ profiling


def session_args(
    ds: str,
    cfg: dict,
    sample: dict,
    pool: dict,
    *,
    experiment: str,
    baseline_dir: Path,
    label: str,
    sampling: str,
    reuse: bool,
    extra: dict | None = None,
    reuse_from: tuple[list[str], list[str]] = ([], []),
) -> dict:
    """Runner arguments of a profiling arm; ``extra`` is merged into the opt rows' protocol (staged BO arms: the
    staging protocol and the stage size). The panel stays the configuration's whole own sample; a stage arm runs a
    prefix of it (its shards)."""
    return {
        **space.runner_args(cfg),
        "runner_extra": space.runner_cli(cfg),
        "split": "train",
        "experiment": experiment,
        "baseline_dir": baseline_dir,
        "reuse": reuse,
        # cross-experiment reuse (state.json quality_reuse.sources; ALF branch v8): ORIGINAL training episodes of
        # the source experiments with byte-identical memory (agentgym_session_eval --reuse_from)
        "reuse_from": list(reuse_from[0]),
        "reuse_from_dirs": list(reuse_from[1]),
        "panel": Path(sample["file"]),
        "protocol_extra": {
            "panel_hash": pool["pool_hash"],
            "pool_hash": pool["pool_hash"],
            "sample_hash": sample["sample_hash"],
            "sample_seed": sample["seed"],
            "sampling": sampling,
            "space_version": space.VERSION,
            "space_config": cfg,
            "bo_label": label,
            "bo_experiment": experiment_id(ds, sampling),
            **(extra or {}),
        },
    }


def quality_of(ds: str, reward: float) -> float:
    """``agentgym_session_eval.quality_of`` (same formula), local: importing that module pulls the vendored
    AgentGym evaluator (``jsonlines``), which ``.venv-optimizer`` -- the driver's venv -- does not have."""
    return max(reward, 0) / 100 if ds == "sciworld" else float(reward)


# nomemory-prefix-v1 (user 2026-10-23, WebShop): NoMemory does not need its full 200 -- the baseline (gain anchor,
# HV reference, prune comparator, router candidate) is its STAGE-100 stratified prefix (state.json nomemory.order
# [:100], already measured), and the remaining NoMemory tasks are never run. Opt-in POOL_NOMEMORY_PREFIX=<n> (+
# POOL_NOMEMORY_PREFIX_REASON) on a round0-design-v1 from-round-0 run (WebShop); recorded once in state.json
# ``nomemory.sample_used``. Without it (ALFWorld / ScienceWorld) everything uses the whole sample, unchanged.
NM_PREFIX_ENV = "POOL_NOMEMORY_PREFIX"
NM_PREFIX_REASON_ENV = "POOL_NOMEMORY_PREFIX_REASON"
NM_PREFIX_PROTOCOL = "nomemory-prefix-v1"


def nomemory_ids(state: dict) -> list[str]:
    """NoMemory's ids used for every estimate: the recorded prefix (nomemory-prefix-v1), else its whole sample."""
    used = state["nomemory"].get("sample_used")
    return list(used["ids"]) if used else list(state["nomemory"]["sample"]["ids"])


def nomemory_sample(state: dict) -> dict:
    """``state.nomemory.sample`` restricted to ``nomemory_ids`` (identity without a recorded prefix)."""
    s = state["nomemory"]["sample"]
    return {**s, "ids": nomemory_ids(state)} if s["ids"] != nomemory_ids(state) else s


def nomemory_prefix_block(
    ds: str, state: dict, pool: dict, nm_dir: Path, now: str, env=None
) -> dict | None:
    env = os.environ if env is None else env
    # Fresh profiles call this hook before creating their NoMemory sample.
    # A measured prefix can only be selected on a later resume.
    nm = state.get("nomemory")
    if nm is None:
        return None
    if nm.get("sample_used") is not None:
        return nm["sample_used"]
    want = env.get(NM_PREFIX_ENV, "").strip()
    if not want:
        return None
    reason = env.get(NM_PREFIX_REASON_ENV, "").strip()
    if not reason:
        raise SystemExit(f"{NM_PREFIX_ENV} needs {NM_PREFIX_REASON_ENV}")
    if not round0_design_from_round0(state):
        raise SystemExit(
            f"{NM_PREFIX_ENV}: only for round0-design-v1 from-round-0 runs (WebShop)"
        )
    if not state["rounds"]:
        return None  # a fresh run: recorded at the first resume after round 0 (the prefix must be measured)
    n = int(want)
    strata, _, _ = pool_meta(pool)
    order = nomemory_order(state, strata)
    sizes = list(state["staging"].get("sizes") or [])
    if n not in sizes or n >= len(order):
        raise SystemExit(
            f"{NM_PREFIX_ENV}={n}: not a NoMemory stage size below {len(order)} ({sizes})"
        )
    ids = order[:n]
    done = drv.item_files(nm_dir, ds)
    if missing := [i for i in ids if i not in done]:
        raise SystemExit(f"NoMemory {n}-prefix incomplete: {missing[:5]}...")
    nm["sample_used"] = {
        "protocol": NM_PREFIX_PROTOCOL,
        "size": n,
        "ids": ids,
        "rule": f"stage-{n} stratified prefix of NoMemory's own sample (state.json nomemory.order[:{n}]); the "
        "baseline for gains / HV reference / prune comparators / router candidate; the remaining "
        f"{len(order) - n} NoMemory tasks are not run",
        "episodes_present": len([i for i in order if i in done]),
        "reason": reason,
        "recorded_at": now,
    }
    return nm["sample_used"]


def nomemory_estimate(
    ds: str, pool: dict, sample: dict, nm_dir: Path, *, max_round: int | None = None, raw_score: bool = False
) -> tuple[us.Estimate, dict[str, float]]:
    """NoMemory's stratified quality on ``sample``; ``max_round`` (max-round-v1): the derived-truncation view
    (an episode longer than ``max_round`` turns has quality 0); None = as measured (unchanged)."""
    strata, w, sizes = pool_meta(pool)
    files = drv.item_files(nm_dir, ds)
    if set(sample["ids"]) - set(files):
        raise SystemExit("NoMemory sample incomplete")

    def q_of(f: Path) -> float:
        res = json.load(open(f))
        if max_round is not None and (eb.result_steps(res) or 0) > max_round:
            return 0.0  # not bought by turn max_round: reward 0 (episode_budget REWARD_RULE)
        return float(res["reward"]) if raw_score else quality_of(ds, res["reward"])

    q = {i: q_of(files[i]) for i in sample["ids"]}
    return us.stratified(
        [q[i] for i in sample["ids"]], [strata[i] for i in sample["ids"]], w, sizes
    ), q


def nomemory_rows(
    ds: str, nm_dir: Path, ids: list[str], *, max_round: int | None = None, raw_score: bool = False
) -> dict[str, dict]:
    """``max_round`` (max-round-v1): every episode longer than ``max_round`` turns in the derived-truncation view
    (``episode_budget.truncate``: quality 0, steps / tokens of the first ``max_round`` calls, wall minus the removed
    tail; span cost: first .. ``max_round``-th call); None = as measured (unchanged). Per-query NoMemory results of an official-evaluator arm: qid -> {quality, tokens, e2e_ms} (the opt-row
    fields of the memory arms). Cost = ``episodes.jsonl`` (episode wall time + this episode's LLM tokens,
    ``agentgym_official_eval.process_logged``; last record per item); for items run before that log existed,
    ``llm_calls.jsonl`` (tokens; latency = span of the item's last worker's calls, which omits env reset /
    the final step: approximate, counted in ``cost_source``). Items without any cost record are left out.
    ``steps`` / ``steps_success``: environment steps recomputed from the conversation (None if unparseable)."""
    from trimem.experiments import agent_step_metrics as sm

    files = drv.item_files(nm_dir, ds)
    ep: dict[str, dict] = {}
    if (nm_dir / "episodes.jsonl").exists():
        for x in map(json.loads, open(nm_dir / "episodes.jsonl")):
            ep[f"{ds}_{x['item']}"] = x
    calls: dict[str, list] = {}
    if (nm_dir / "llm_calls.jsonl").exists():
        for c in map(json.loads, open(nm_dir / "llm_calls.jsonl")):
            if c.get("item") is not None:
                calls.setdefault(f"{ds}_{c['item']}", []).append(c)
    out = {}
    for q in ids:
        if q not in files:
            continue
        res = json.load(open(files[q]))
        r = with_steps(
            ds,
            {"quality": float(res["reward"]) if raw_score else float(quality_of(ds, res["reward"]))},
            sm.agentgym_steps(res)["environment_steps"],
            raw_science=raw_score and ds == "sciworld",
        )
        if q in ep:
            r |= {
                "tokens": float(ep[q]["generation_tokens"]),
                "e2e_ms": float(ep[q]["episode_ms"]),
                "cost_source": "episodes",
            }
        elif calls.get(q):
            last = max(calls[q], key=lambda c: c["t"])["pid"]
            mine = [c for c in calls[q] if c["pid"] == last]
            r |= {
                "tokens": float(
                    sum(
                        (c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0)
                        for c in mine
                    )
                ),
                "e2e_ms": 1000
                * (
                    max(c["t"] + c["seconds"] for c in mine) - min(c["t"] for c in mine)
                ),
                "cost_source": "llm_calls-span",
            }
        else:
            continue
        steps = r["steps"]
        if max_round is not None and steps is not None and steps > max_round:
            r = _truncated_nomemory(ds, r, res, ep.get(q), calls.get(q, []), max_round)
        out[q] = r
    return out


def _truncated_nomemory(
    ds: str, r: dict, res: dict, ep: dict | None, calls: list, max_round: int
) -> dict:
    """``nomemory_rows``' row of an episode longer than ``max_round`` turns in the max-round-v1 view (pure)."""
    if r["cost_source"] == "episodes":
        mine = [c for c in calls if c["pid"] == ep["pid"]]
        t = eb.truncate(
            reward=res["reward"],
            steps=r["steps"],
            calls=mine,
            episode_ms=ep["episode_ms"],
            max_round=max_round,
        )
        tokens, e2e = float(t["generation_tokens"]), t["episode_ms"]
    else:  # LLM-call span: first .. max_round-th call of the item's last worker
        last = max(calls, key=lambda c: c["t"])["pid"]
        cs = sorted((c for c in calls if c["pid"] == last), key=lambda c: c["t"])
        cs = cs[:max_round]
        tokens = float(
            sum((c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in cs)
        )
        e2e = 1000 * (max(c["t"] + c["seconds"] for c in cs) - cs[0]["t"])
    return with_steps(ds, {"quality": 0.0}, max_round) | {
        "tokens": tokens,
        "e2e_ms": e2e,
        "cost_source": r[
            "cost_source"
        ],  # same keys as a measured row (consumers compare / strip them)
    }


def nomemory_candidate(ds: str, a, state: dict) -> dict | None:
    """NoMemory as an Opt-PerQuery router candidate (user 2026-10-19; ExpHarness idea: route a query to "no
    retrieval"): a ``config_estimates``-style entry {cfg, ids, per_query} built from the profiling NoMemory arm
    (its own stratified-200 sample, round 0; excluded from the BO / GP and from Opt-OneConfig). None (router
    without the NoMemory candidate, logged) when that arm's sample lacks results or costs. Dataset-agnostic:
    a dataset whose NoMemory arm records costs elsewhere overrides ``nomemory_rows``."""
    ids = nomemory_ids(state)
    rows = nomemory_rows(
        ds, nomemory_dir(ds, a.sampling), ids, max_round=eb.view_max_round(state),
        raw_score=bool(state.get("ablation_arm"))
    )
    if len(rows) < len(ids):
        log(
            f"router: NoMemory candidate dropped ({len(rows)}/{len(ids)} sample queries have a result + cost)"
        )
        return None
    src = sorted({r["cost_source"] for r in rows.values()})
    log(f"router: NoMemory candidate on its own {len(ids)}-sample (cost source {src})")
    return {
        "cfg": space.validate(space.NOMEM),
        "ids": list(ids),
        "cost_source": src,
        "per_query": {
            "quality": {q: r["quality"] for q, r in rows.items()},
            "total_tokens": {q: r["tokens"] for q, r in rows.items()},
            "e2e_ms": {q: r["e2e_ms"] for q, r in rows.items()},
        },
    }


def samples_of(state: dict) -> dict[str, list[str]]:
    return {
        space.key(c["config"]): c["sample"]["ids"]
        for rr in state["rounds"]
        for c in rr["configs"]
    }


def opt_rows(ds: str, experiment: str, state: dict, r: int | None = None) -> list[dict]:
    """``drv.train_rows`` in the max-round-v1 view of round ``r`` (None = after profiling): unchanged without a
    recorded ``episode_budget`` or before its from_round; else every longer episode derived-truncated."""
    mr = eb.view_max_round(state, r)
    rows = (
        (drv.train_rows(ds, experiment, include_retrieval_id=True) if state.get("ablation_arm")
         else drv.train_rows(ds, experiment))
        if mr is None else eb.train_rows(experiment, mr)
    )
    if state.get("ablation_arm"):
        if ds != "sciworld" or mr is not None:
            raise ValueError("raw-score ablation requires ScienceWorld without a derived episode-budget view")
        from trimem.experiments.scienceworld_trace_ablation import raw_search_rows

        rows = raw_search_rows(rows)
    return rows


def episode_budget_snapshot(
    ds: str, pool: dict, state: dict, base: Path, nm_dir: Path, r: int | None
) -> Path:
    """max-round-v1 audit file ``<run>/episode_budget/<derived-truncation-n>_round<r>.json`` (never over an original
    file): per opt row (run + noise repeat) and per NoMemory baseline task the measured and the derived values."""
    mr = eb.view_max_round(state, r)
    ids = nomemory_ids(state)
    meas = nomemory_rows(ds, nm_dir, ids)
    der = nomemory_rows(ds, nm_dir, ids, max_round=mr)
    out = (
        base / "episode_budget" / f"{eb.tag(mr)}_round{'post' if r is None else r}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save(
        out,
        {
            "protocol": eb.PROTOCOL,
            "derivation": eb.DERIVATION,
            "max_round": mr,
            "round": r,
            "written_at": datetime.now(UTC).isoformat(),
            "block": eb.block(state),
            "opt_rows": eb.derived_summary(state["experiment"], mr),
            "noise_repeat_rows": eb.derived_summary(
                state["experiment"] + ":noise-repeat", mr
            ),
            "nomemory": {
                q: {"measured": meas.get(q), "derived": der.get(q)} for q in ids
            },
        },
    )
    log(f"{eb.PROTOCOL}: derived view snapshot {out}")
    return out


def estimates(ds: str, pool: dict, state: dict, experiment: str) -> dict[str, dict]:
    bo = bo_module()

    strata, w, sizes = pool_meta(pool)
    return bo.config_estimates(
        opt_rows(ds, experiment, state),
        samples=samples_of(state),
        strata=strata,
        weights=w,
        sizes=sizes,
        experiment=experiment,
    )


def points(
    ds: str,
    pool: dict,
    state: dict,
    nm: us.Estimate,
    estimator: str,
    rows: list[dict] | None = None,
    r: int | None = None,
):
    """BO points of ``state``'s configurations; ``rows`` overrides the experiment's opt rows (pool_branch
    preview: the source run's rows of the branched rounds). ``r``: the round being proposed (max-round-v1 view of
    that round; None = after profiling)."""
    bo = bo_module()

    exp = state["experiment"]
    rows = opt_rows(ds, exp, state, r) if rows is None else rows
    if (
        estimator == "paired"
    ):  # old behaviour: one common sample, paired gains (needs --sampling shared)
        if state["staging"]["protocol"] != STAGING_OFF:
            raise SystemExit("the paired estimator runs unstaged (--staging none)")
        ids = state["nomemory"]["sample"]["ids"]
        if any(set(s) != set(ids) for s in samples_of(state).values()):
            raise SystemExit(
                "the paired estimator needs one shared sample (--sampling shared)"
            )
        return bo.load_points(
            rows,
            panel_ids=ids,
            dataset=ds,
            panel_hash=pool["pool_hash"],
            experiment=exp,
        )
    strata, w, sizes = pool_meta(pool)
    obs = observed_samples(state)
    return bo.load_points_unpaired(
        restrict_rows(rows, obs, samples_of(state)),
        samples=obs,
        strata=strata,
        weights=w,
        sizes=sizes,
        dataset=ds,
        pool_hash=pool["pool_hash"],
        experiment=exp,
        baseline_quality=nm.mean,
    )


def new_state(ds: str, pool: dict, a) -> dict:
    return {
        "dataset": ds,
        "experiment": experiment_id(ds, a.sampling),
        "pool_hash": pool["pool_hash"],
        "pool_size": pool["size"],
        "sample_size": getattr(a, "full_panel_size", None) or N_SAMPLE,
        **fresh_panel_settings(a),
        "sampling": a.sampling,
        "estimator": a.estimator,
        "memory_policy": mp.current(),
        "rounds_planned": a.rounds,
        "batch": a.batch,
        "reuse": a.reuse,
        "space_version": space.VERSION,
        **({"ablation_arm": a.ablation_arm} if getattr(a, "ablation_arm", None) else {}),
        "created_at": datetime.now(UTC).isoformat(),
        "rounds": [],
    }


def fresh_panel_settings(a) -> dict:
    """Explicit fresh-run protocol; absent flags preserve every legacy identity.

    Unlike sample-size-v1, this applies to initialization, references and budget
    admission from round zero. A common namespace pairs samples across methods.
    """
    n = getattr(a, "full_panel_size", None)
    quota = getattr(a, "initial_budget_episodes", None)
    namespace = getattr(a, "task_sample_namespace", None)
    if n is None and quota is None and namespace is None:
        return {}
    if n is None or quota is None or not namespace:
        raise ValueError(
            "fresh panel needs full_panel_size, initial_budget_episodes and task_sample_namespace"
        )
    if not 50 <= n <= N_SAMPLE or quota < N_ROUND0 * n:
        raise ValueError(
            "fresh panel must be 50..200 tasks and fund all ten initial configurations"
        )
    if a.max_configs < N_ROUND0 or a.sampling != "per-config":
        raise ValueError(
            "fresh panel needs per-config sampling and at least ten configs"
        )
    if getattr(a, "sample_size", None) is not None or getattr(
        a, "sample_uniform", False
    ):
        raise ValueError("fresh panel cannot combine with sample-size-v1")
    return {
        "full_panel_size": n,
        "initial_budget_episodes": quota,
        "task_sample_namespace": namespace,
    }


def sampling_namespace(state: dict) -> str:
    return state.get("task_sample_namespace", state["experiment"])


def staging_block(a, previous=None) -> dict:
    """The staging protocol of this run (state.json ``staging``; compared on every resume). Budget = the unpruned
    plan: (10 round-0 + rounds x batch) configurations x 200 episodes (NoMemory and the noise repeat excluded,
    they are outside the configuration budget in both designs). ``--staging none`` = one stage of 200 and
    max_configs = the unpruned plan: exactly the old fixed rounds."""
    unpruned = N_ROUND0 + a.rounds * a.batch
    staged = a.staging != STAGING_OFF
    fresh = fresh_panel_settings(a)
    full = fresh.get("full_panel_size", N_SAMPLE)
    requested_sizes = getattr(a, "stage_sizes", None)
    if requested_sizes is not None:
        if not fresh:
            raise ValueError("explicit stage sizes require a fresh-panel protocol")
        try:
            requested_sizes = [int(s) for s in requested_sizes.split(",")]
        except ValueError as exc:
            raise ValueError("stage sizes must be comma-separated integers") from exc
        if not requested_sizes or requested_sizes[-1] != full or any(
            s <= 0 or s > full for s in requested_sizes
        ) or requested_sizes != sorted(set(requested_sizes)):
            raise ValueError("stage sizes must be strictly increasing and end at full panel size")
    stages = requested_sizes or [s for s in STAGE_SIZES if s < full] + [full]
    b = (
        sp.Budget(fresh["initial_budget_episodes"], full, a.max_configs)
        if fresh
        else sp.Budget.unpruned(
            unpruned, N_SAMPLE, a.max_configs if staged else unpruned
        )
    )
    if not fresh and b.max_configs < unpruned:
        raise SystemExit(
            f"--max_configs {b.max_configs} < the unpruned plan's {unpruned} configurations"
        )
    block = {
        "protocol": sp.PROTOCOL if staged else STAGING_OFF,
        "z": sp.Z,
        "sizes": stages if staged else [full],
        "applies_from_round": 0,
        "order": "per configuration: staged_pruning.stage_order(its own stratified sample, strata = pool stratum "
        "(task type), seed = staged_pruning.seed_of(experiment, 'stages', config key)); stages = prefixes",
        "prune_objectives": "optimistic corner: quality (unpaired, stratified) + z SE, tokens - z SE, latency_ms - "
        "z SE on the configuration's stage prefix vs (a) the point estimates of fully evaluated configurations, "
        "(b) the pessimistic corners (- z SE / + z SE) of NoMemory (its own sample's stage prefix) and of every "
        "configuration evaluated at >= the stage (its own prefix of that size); plan guard",
        "nomemory_staged": "NoMemory runs the stage prefixes of its own sample (order seed_of(experiment, 'stages', "
        "'NoMemory')) alongside round 0; the noise repeat runs in full with the last stage",
        "budget_episodes": b.total_episodes,
        "budget_configs_unpruned": unpruned,
        "full_cost": b.full_cost,
        "max_configs": b.max_configs,
    }
    return pp_policy.configure(block, a, previous)


def load_state(ds: str, pool: dict, a) -> tuple[dict, Path]:
    """state.json of the run (new if absent). Resume compatibility: a state written before staging existed (round 0
    proposed, no BO round) gets the ``staging`` block now; its round 0 is staged retroactively (v2: completed
    episodes are kept and count in the budget; decisions use the stage prefixes). A state whose BO rounds were
    proposed without staging is refused."""
    p = run_dir(ds, a.sampling) / "state.json"
    st = json.load(open(p)) if p.exists() else new_state(ds, pool, a)
    sa.settings_for(st, a, existing=p.exists())
    if getattr(a, "ablation_arm", None):
        from trimem.experiments.scienceworld_trace_ablation import (
            arm_policy,
            validate_search_design,
        )

        validate_search_design(st["search_design"])
        if p.exists() and st.get("ablation_policy") != arm_policy(a.ablation_arm):
            raise ValueError("recorded ablation policy changed; refusing to resume")
    if sa.baselines.active(st["search_design"]) and not run_tag():
        raise ValueError("baseline requires its own --run_tag")
    want = new_state(ds, pool, a)
    if st.get("ablation_arm") != want.get("ablation_arm"):
        raise ValueError("state.json ablation arm differs; refusing to mix campaigns")
    if want.get("ablation_arm") and (ds != "sciworld" or not run_tag()):
        raise ValueError("ScienceWorld ablation requires its own run tag")
    for key in ("full_panel_size", "initial_budget_episodes", "task_sample_namespace"):
        if st.get(key) != want.get(key):
            raise ValueError(
                f"state.json {key} differs; fresh-panel protocol cannot migrate or change on resume"
            )
    for k in (
        "experiment",
        "pool_hash",
        "sampling",
        "estimator",
        "memory_policy",
        "batch",
        "space_version",
    ):
        if st[k] != want[k]:
            raise SystemExit(
                f"state.json has {k}={st[k]!r}, this run {want[k]!r}; refusing to mix"
            )
    blk = staging_block(a, st.get("staging", {}) if p.exists() else None)
    # WebShop v7 (user 2026-10-23): syftr-style pruning in its OWN fresh run dir (no legacy WebShop pool run to mix
    # with), so no run tag is needed there; ALF / Sci / Exp2 keep the isolation rule
    if (
        pp_policy.requires_isolation(blk, st["search_design"])
        and not run_tag()
        and ds != "webshop"
    ):
        raise ValueError("alternative pruning requires its own --run_tag")
    if "staging" not in st:
        if len(st["rounds"]) > 1:
            raise SystemExit(
                "state.json has BO rounds proposed without a staging protocol; refusing to mix"
            )
        st["staging"] = {
            **blk,
            "recorded_at": datetime.now(UTC).isoformat(),
            "recorded_after_rounds": len(st["rounds"]),
        }
    elif {
        k: v
        for k, v in st["staging"].items()
        if not k.startswith("recorded_") and k != ASYNC_KEY
    } != blk:
        raise SystemExit(
            "state.json has another staging protocol / stage sizes / budget; refusing to mix "
            f"(state {st['staging']['protocol']!r}, this run {blk['protocol']!r})"
        )
    apply_regime(ds, st)
    apply_sample_kinds(st)
    # semfilter-regime-v1: export the recorded env; refuse an unlabelled / mismatched resume
    apply_semfilter(st)
    return st, p


# ------------------------------------------------------------------ latency regime (latency-regime-v1, 2026-10-22)
# ALFWorld v7 (user 2026-10-22, "请改成c16"): profiling ran at c128 (concurrency-v1, request-load routing) for rounds
# 0-5; from round 6 it runs under the controlled-c16-v1 runtime. Latency measured under the two regimes must not be
# mixed: (1) the latency GP gets a categorical regime input (session_pool.use_latency_regime: old-regime
# configurations at 1, predictions at the current regime 0; quality / tokens shared unchanged); (2) NoMemory is
# rerun under the new regime on its same 200-task sample (``nomemory.dir``) and replaces the old arm for the gain
# anchor, the HV reference's quality SE and pruning from ``from_round`` (the old estimate is logged alongside);
# (3) quality reuse only reuses episodes created after the switch (agentgym_session_eval TRIMEM_REUSE_SINCE),
# because a reused episode carries its source's wall time. Opt-in: POOL_RUNTIME_PROFILE=<profile> together with
# POOL_CONCURRENCY_CHANGE=1 + POOL_CONCURRENCY_CHANGE_REASON on a state that has rounds but no runtime profile.
LATENCY_REGIME_PROTOCOL = "latency-regime-v1"
REUSE_SINCE_ENV = "TRIMEM_REUSE_SINCE"  # agentgym_session_eval.REUSE_SINCE_ENV


def apply_regime(ds: str, state: dict) -> None:
    """Process-wide effects of a recorded latency regime: the NoMemory dir and the reuse-source time bound."""
    blk = state.get("latency_regime")
    if not blk:
        _NM_REGIME.pop(ds, None)
        return
    if blk.get("protocol") != LATENCY_REGIME_PROTOCOL:
        raise SystemExit(
            f"state.json latency_regime {blk.get('protocol')!r}; refusing to mix"
        )
    _NM_REGIME[ds] = Path(blk["nomemory"]["dir"])
    os.environ[REUSE_SINCE_ENV] = blk["at"]


def latency_regime_block(ds: str, state: dict, a, now: str) -> dict | None:
    """Record a mid-run runtime regime change (see above) once; returns the block (None without one). From round
    = the last proposed round when none of its configurations has run an episode yet (it reruns entirely under the
    new regime), else the next round to be proposed."""
    blk = state.get("latency_regime")
    if blk is not None:
        apply_regime(ds, state)
        return blk
    want = runtime_profile_want(ds)
    rounds = state.get("rounds") or []
    if want is None or state.get("runtime_profile") is not None or not rounds:
        return None
    reason = os.environ.get(CONCURRENCY_CHANGE_REASON_ENV, "").strip()
    if os.environ.get(CONCURRENCY_CHANGE_ENV, "").strip() != "1" or not reason:
        return None  # runtime_profile_block refuses, as before
    fresh = all(
        staging_of(c)["stage_reached"] == 0 and staging_of(c)["episodes"] == 0
        for c in rounds[-1]["configs"]
    )
    r0 = len(rounds) - 1 if fresh and len(rounds) > 1 else len(rounds)
    conc = state.get("concurrency") or {}
    before = {k: conc.get(k) for k in CONCURRENCY_CHANGEABLE}
    nm_old = nomemory_dir(ds, a.sampling, regime=False)
    nm_new = nm_old.parent / f"NoMemory_{want['name']}"
    state["runtime_profile"] = {
        **want,
        "from_round": r0,
        "before": {"concurrency": before, "runtime_profile": None},
        "recorded_at": now,
        "reason": reason,
    }
    state["latency_regime"] = {
        "protocol": LATENCY_REGIME_PROTOCOL,
        "from_round": r0,
        "at": now,
        "old": {"rounds": f"0..{r0 - 1}", "concurrency": before},
        "new": {
            "runtime_profile": want["name"],
            "total_concurrency": want["total_concurrency"],
        },
        "gp": "latency GP: categorical regime input (old-regime configurations = 1, predictions at 0); quality and "
        "tokens GPs unchanged (session_pool.use_latency_regime)",
        "nomemory": {
            "dir": str(nm_new),
            "before_dir": str(nm_old),
            "sample": "state.json nomemory.sample (same 200 train tasks, same stage order); run in full before "
            f"round {r0}'s arms",
            "use": f"from round {r0}: the gain anchor (quality), the HV reference's quality SE, pruning and every "
            "later stage use this rerun; the old arm stays in place (archived, logged alongside)",
        },
        "reuse": f"quality reuse sources restricted to rows created at / after {now} ({REUSE_SINCE_ENV})",
        "pruning": "stage pruning compares against configurations of both regimes (their latency corners as "
        "measured): old-regime latencies are higher, so they dominate new-regime configurations less often -- "
        "conservative (fewer prunes), never a wrong prune because of the regime",
        "front_and_hv_ref": "posterior means at the current regime (regime input 0) -> c16-consistent front / "
        "anchored HV reference",
        "recorded_at": now,
        "reason": reason,
    }
    if fresh and r0 == len(rounds) - 1:
        state["latency_regime"]["rerun_round"] = (
            f"round {r0} was proposed but no episode ran under the old regime: it runs entirely under the new one"
        )
    apply_regime(ds, state)
    return state["latency_regime"]


def run_regime_nomemory(ds: str, pool: dict, state: dict, a) -> None:
    """latency-regime-v1: run the NoMemory rerun in full (its sample, stage order) under the new runtime when
    incomplete; log the old and new estimates."""
    blk = state.get("latency_regime")
    if not blk:
        return
    strata, _, _ = pool_meta(pool)
    nm_new = Path(blk["nomemory"]["dir"])
    arm = Arm("NoMemory", nm_new, nomemory_order(state, strata), None)
    if len(arm.completed(ds)) < len(arm.ids):
        log(
            f"{LATENCY_REGIME_PROTOCOL}: NoMemory rerun under {blk['new']['runtime_profile']} -> {nm_new} "
            f"({len(arm.ids)} episodes)"
        )
        res = run_capped(ds, [arm], CONCURRENCY[ds])
        if len(arm.completed(ds)) < len(arm.ids):
            raise SystemExit(f"NoMemory rerun incomplete: {res}")
    old, _ = nomemory_estimate(
        ds, pool, state["nomemory"]["sample"], Path(blk["nomemory"]["before_dir"])
    )
    new, _ = nomemory_estimate(ds, pool, state["nomemory"]["sample"], nm_new)
    lat = {}
    for tag, d_ in (("old", Path(blk["nomemory"]["before_dir"])), ("new", nm_new)):
        rows = nomemory_rows(ds, d_, state["nomemory"]["sample"]["ids"])
        ms = [r_["e2e_ms"] for r_ in rows.values() if r_.get("e2e_ms")]
        lat[tag] = sum(ms) / len(ms) / 1000 if ms else float("nan")
    log(
        f"{LATENCY_REGIME_PROTOCOL}: NoMemory quality old {old.mean:.3f} (SE {old.se:.4f}), mean latency "
        f"{lat['old']:.1f} s | new {new.mean:.3f} (SE {new.se:.4f}), {lat['new']:.1f} s -- the new one is used from "
        f"round {blk['from_round']}"
    )


# ------------------------------------------------------------------ SemFilter regime (semfilter-regime-v1, 2026-10-22)
# User 2026-10-22: rerun ALF v7 round 6 (same 10 configurations, same 100-task samples) with semfilter-v2
# (``semfilter_v2``: top-N cosine cutoff, batched score-only judge, rubric v3, guided decoding). Opt-in
# ``POOL_SEMFILTER=v2`` (+ ``POOL_SEMFILTER_FROM_ROUND``, default = the next round to be proposed; the TRIMEM_SEMFILTER_*
# knobs of the caller's env, else the semfilter_v2 defaults) is recorded ONCE in state.json ``semfilter``; from then on
# the driver exports the recorded TRIMEM_SEMFILTER_* to its own env, so every episode runner it starts (profiling
# arms, NoMemory, test) inherits them, and a resume without POOL_SEMFILTER=v2 -- or with different knobs -- is
# refused. Recording refuses while any round >= from_round already has episodes (supersede them first:
# pool_discard_round --keep_configs). The GP gets a categorical "semfilter regime" input for SemFilter plans
# (session_pool.use_semfilter_regime: rounds < from_round = old judge, predictions at v2); non-SemFilter plans and
# the latency-regime input are unchanged. Without the env and without a recorded block nothing changes.
SEMFILTER_ENV = "POOL_SEMFILTER"
SEMFILTER_FROM_ENV = "POOL_SEMFILTER_FROM_ROUND"
SEMFILTER_REGIME_PROTOCOL = "semfilter-regime-v1"


def semfilter_env(blk: dict) -> dict[str, str]:
    """The TRIMEM_SEMFILTER_* environment of a recorded block (pure)."""
    return {
        "TRIMEM_SEMFILTER_V2": "1",
        "TRIMEM_SEMFILTER_TOPN": str(blk["top_n"]),
        "TRIMEM_SEMFILTER_BATCH": str(blk["batch"]),
        "TRIMEM_SEMFILTER_MAX_TOKENS": str(blk["max_tokens"]),
        "TRIMEM_SEMFILTER_JUDGE_INPUT": blk["judge_input"],
        "TRIMEM_SEMFILTER_RUBRIC": blk["rubric"],
        "TRIMEM_SEMFILTER_GUIDED": "1" if blk["guided"] else "0",
    }


def effective_semfilter_env(blk: dict) -> dict[str, str]:
    """The env of a recorded semfilter block: semfilter-v2's, or -- once semfilter-v2b is recorded (WebShop) --
    v2b's (rubric v4). Pure. The v2b env applies process-wide from the resume that records it on; round-0 is
    complete by then (semfilter_v2b_block refuses otherwise), so no old-regime episode runs under it."""
    v2b = blk.get(SEMFILTER_V2B_KEY)
    return dict(v2b["env"]) if v2b else semfilter_env(blk)


def semfilter_label(state: dict) -> str | None:
    """The SemFilter label of NEW SemFilter episodes of this run (test / fixed-plan / sweep rows):
    semfilter-v2b once recorded, else the block's protocol (semfilter-v2), None without a block. Pure."""
    blk = state.get("semfilter")
    if not blk:
        return None
    return (
        blk[SEMFILTER_V2B_KEY]["protocol"]
        if blk.get(SEMFILTER_V2B_KEY)
        else blk["protocol"]
    )


def check_semfilter(state: dict, env=None) -> dict[str, str] | None:
    """The env to export for a recorded block (refusing a resume without POOL_SEMFILTER=v2 or with other knobs), or
    None when nothing is recorded (refusing TRIMEM_SEMFILTER_V2=1 in the env without the opt-in: unlabelled). Pure."""
    env = os.environ if env is None else env
    blk = state.get("semfilter")
    want = env.get(SEMFILTER_ENV, "").strip()
    if blk is None:
        if not want and env.get("TRIMEM_SEMFILTER_V2", "").strip() == "1":
            raise SystemExit(
                f"TRIMEM_SEMFILTER_V2=1 without {SEMFILTER_ENV}=v2: SemFilter v2 episodes would be unrecorded"
            )
        return None
    if want != "v2":
        raise SystemExit(
            f"state.json records {blk['protocol']} from round {blk['from_round']}; resume with {SEMFILTER_ENV}=v2"
        )
    out = effective_semfilter_env(blk)
    diff = {
        k: (env.get(k), v)
        for k, v in out.items()
        if env.get(k) is not None and env.get(k) != v
    }
    if diff:
        raise SystemExit(f"TRIMEM_SEMFILTER_* differ from the recorded block: {diff}")
    return out


def apply_semfilter(state: dict) -> dict | None:
    """Export the recorded SemFilter v2 env to this process (inherited by every runner); None without a block."""
    out = check_semfilter(state)
    if out is not None:
        os.environ.update(out)
    return state.get("semfilter")


def round_has_episodes(base: Path, ds: str, rnd: dict) -> list[str]:
    """Configurations of a round whose config dir holds any result file."""
    return [
        cfg_name(c["config"])
        for c in rnd["configs"]
        if drv.item_files(base / "configs" / cfg_name(c["config"]), ds)
    ]


def semfilter_block(
    ds: str, state: dict, base: Path, now: str, env=None
) -> dict | None:
    """Record semfilter-regime-v1 once (opt-in ``POOL_SEMFILTER=v2``) and export its env; see above."""
    env = os.environ if env is None else env
    if state.get("semfilter") is None:
        want = env.get(SEMFILTER_ENV, "").strip()
        if not want:
            return apply_semfilter(state)
        if want != "v2":
            raise SystemExit(f"{SEMFILTER_ENV} must be v2")
        from trimem.experiments import semfilter_v2 as sf2

        cfg = sf2.settings({**env, sf2.ENV_GATE: "1"})
        n = len(state["rounds"])
        r0 = int(env.get(SEMFILTER_FROM_ENV, "").strip() or n)
        # Explicit fresh-panel baselines also enable v2 from their first round.
        # Preserve the legacy rules for already proposed or measured rounds.
        fresh_baseline = sa.baselines.active(state.get("search_design", {})) and all(
            k in state
            for k in (
                "full_panel_size",
                "initial_budget_episodes",
                "task_sample_namespace",
            )
        )
        lo = 0 if n == 0 and (round0_design_from_round0(state) or fresh_baseline) else 1
        if not lo <= r0 <= n:
            raise SystemExit(f"{SEMFILTER_FROM_ENV}={r0} outside [{lo}, {n}]")
        busy = {
            rr["round"]: round_has_episodes(base, ds, rr)
            for rr in state["rounds"]
            if rr["round"] >= r0
        }
        busy = {k: v for k, v in busy.items() if v}
        if busy:
            raise SystemExit(
                f"rounds >= {r0} already have episodes {busy}: supersede them first "
                "(pool_discard_round --keep_configs)"
            )
        state["semfilter"] = {
            "protocol": sf2.PROTOCOL,
            "regime_protocol": SEMFILTER_REGIME_PROTOCOL,
            "from_round": r0,
            **cfg,
            "before": "semfilter-v1: every candidate judged, one call per candidate, long JSON verdict "
            "(evidence / transfer), rubric of the memory policy (semantic-rubric-v1 / -v2)",
            "gp": "categorical semfilter-regime input in all three GPs for SemFilter plans (rounds < from_round = "
            "old judge = 1, predictions at 0); non-SemFilter plans unaffected (session_pool.use_semfilter_regime)",
            "env": semfilter_env(cfg),
            "opt_in": f"{SEMFILTER_ENV}=v2",
            "reason": env.get("POOL_SEMFILTER_REASON", "")
            or "user 2026-10-22: rerun round 6 with semfilter-v2 and verify the latency drop",
            "recorded_at": now,
        }
    else:
        # a resume that names a from_round must match the record (user 2026-10-22, ScienceWorld: semfilter-v2 from
        # round 0 after the round-0 SemFilter configurations were superseded; the only way to lower a recorded
        # from_round is ``pool_discard_round --keep_configs --configs ... --semfilter_from_round R``)
        want_r = env.get(SEMFILTER_FROM_ENV, "").strip()
        rec_r = state["semfilter"]["from_round"]
        if want_r and int(want_r) != rec_r:
            raise SystemExit(
                f"{SEMFILTER_FROM_ENV}={want_r} but state.json semfilter is recorded from round {rec_r}; to lower "
                "it supersede the earlier SemFilter configurations first (pool_discard_round --keep_configs "
                "--configs ... --semfilter_from_round R)"
            )
    return apply_semfilter(state)


# ------------------------------------------------------------------ semfilter-v2b (WebShop, user 2026-10-23)
# WebShop v7 round 0 showed SemFilter-v2 passing ZERO cards for most queries (threshold 0.8 empty for 73-85% of
# queries; rubric v3's WebShop CONFLICTS -- different option / price limit -- fire on nearly every card because
# retrieval is product-isolated) and empty-memory episodes ~= NoMemory at 2x tokens. Opt-in ``POOL_SEMFILTER_V2B=1``
# (+ ``POOL_SEMFILTER_V2B_REASON``), WebShop only, on a run with a semfilter-v2 block; recorded ONCE in state.json
# ``semfilter.v2b`` from the next round to be proposed:
#   * rubric semantic-rubric-v4-webshop (``semantic_rubric``; v1-v3 unchanged) -> TRIMEM_SEMFILTER_RUBRIC;
#   * every SemFilter configuration PROPOSED from from_round carries sem_fallback "fill-topk"
#     (``session_retrieval.retrieve``) and semfilter_threshold in {0.4, 0.6} (0.8 no longer proposable;
#     ``session_bo_agentenv.use_semfilter_v2b_for`` -> ``session_pool.use_proposal_rule``);
#   * GP: the SemFilter configurations of rounds < from_round are the OLD semfilter regime (categorical = 1,
#     predictions at 0 = v2b; ``use_semfilter_regime_for``); non-SemFilter configurations unaffected;
#   * catalog (router / selection / test): old-regime SemFilter configurations are excluded (``profile_table``),
#     the test / fixed-plan / sweep SemFilter arms run v2b (fill-topk + rubric v4; label ``semfilter_label``).
SEMFILTER_V2B_ENV = "POOL_SEMFILTER_V2B"
SEMFILTER_V2B_REASON_ENV = "POOL_SEMFILTER_V2B_REASON"
SEMFILTER_V2B_KEY = "v2b"
SEMFILTER_V2B_DATASETS = ("webshop",)
SEMFILTER_V2B_THRESHOLDS = (0.4, 0.6)


def sf_regime_from(state: dict, r: int) -> int:
    """The round from which SemFilter configurations are the CURRENT regime at round ``r`` (v2b's from_round once
    active, else the semfilter block's). Pure."""
    blk = state["semfilter"]
    v2b = blk.get(SEMFILTER_V2B_KEY)
    return v2b["from_round"] if v2b and r >= v2b["from_round"] else blk["from_round"]


def semfilter_v2b_block(
    ds: str, state: dict, base: Path, now: str, env=None
) -> dict | None:
    """Record semfilter-v2b once (see above); returns the recorded sub-block (None when absent). Refuses a
    non-WebShop run, a run without a semfilter-v2 block, an incomplete earlier SemFilter configuration (it would
    finish under the new rubric) and rounds >= from_round that already have episodes."""
    env = os.environ if env is None else env
    blk = state.get("semfilter")
    want = env.get(SEMFILTER_V2B_ENV, "").strip()
    if blk is not None and blk.get(SEMFILTER_V2B_KEY) is not None:
        return blk[SEMFILTER_V2B_KEY]
    if not want:
        return None
    if want != "1":
        raise SystemExit(f"{SEMFILTER_V2B_ENV} must be 1")
    if ds not in SEMFILTER_V2B_DATASETS:
        raise SystemExit(f"semfilter-v2b is WebShop-only (rubric v4-webshop), not {ds}")
    if blk is None or blk.get("protocol") != "semfilter-v2":
        raise SystemExit("semfilter-v2b needs a recorded semfilter-v2 block")
    reason = env.get(SEMFILTER_V2B_REASON_ENV, "").strip()
    if not reason:
        raise SystemExit(f"{SEMFILTER_V2B_ENV} needs {SEMFILTER_V2B_REASON_ENV}")
    from trimem.experiments import semantic_rubric as sr
    from trimem.experiments import semfilter_v2 as sf2

    n = len(state["rounds"])
    busy = {
        rr["round"]: round_has_episodes(base, ds, rr)
        for rr in state["rounds"]
        if rr["round"] >= n
    }
    if any(busy.values()):
        raise SystemExit(f"rounds >= {n} already have episodes {busy}")
    open_sf = [
        cfg_name(c["config"])
        for rr in state["rounds"]
        for c in rr["configs"]
        if c["config"].get("plan") in space.SEMFILTER_PLANS
        and not staging_of(c)["pruned_at"]
        and staging_of(c)["stage_reached"] < len(c["sample"]["ids"])
    ]
    if open_sf:
        raise SystemExit(
            f"semfilter-v2b: earlier SemFilter configurations still incomplete {open_sf}; finish them first"
        )
    v2_env = semfilter_env(blk)
    v2b_env = {**v2_env, "TRIMEM_SEMFILTER_RUBRIC": sr.VERSION_V4_WEBSHOP}
    blk[SEMFILTER_V2B_KEY] = {
        "protocol": sf2.PROTOCOL_V2B,
        "from_round": n,
        "rubric": sr.VERSION_V4_WEBSHOP,
        "sem_fallback": "fill-topk",
        "fallback_rule": "a SemFilter stage that passes fewer than top_k candidates is filled with the non-passed "
        "candidates in the plan's order: P2 / P2+Rank = passed first, then the rest of the capped pool (query "
        "cosine; +Rank: ColBERT), select_top_k over that list (final selection filled to top_k when the pool "
        "allows; failed-card cap still applies); P1 / P1+Rank = the seeds filled to min(top_k, |seeds|) with "
        "non-passed seeds in seed (query cosine) order before expansion. Configuration knob sem_fallback (part of "
        "the key); receipt.semfilter_fallback + receipt.semfilter.{passed, fallback_filled}",
        "thresholds": list(SEMFILTER_V2B_THRESHOLDS),
        "space_change": "semfilter_threshold 0.8 is no longer PROPOSABLE from from_round (session_pool proposal "
        "rule); measured 0.8 configurations stay valid observations (validation / keys / GP encoding keep it)",
        "judge_input": "unchanged (raw card; NEW TASK = the WebShop instruction, which already states the product "
        "and the price limit -- no extra product-type / price fields added)",
        "gp": "SemFilter configurations of rounds < from_round = old semfilter regime (categorical 1); "
        "predictions at v2b (0); non-SemFilter configurations unaffected",
        "catalog": "router / selection / test exclude the old-regime SemFilter configurations; test, fixed-plan "
        "and sweep SemFilter arms run v2b (fixed plans: plan defaults + sem_fallback fill-topk, threshold 0.6)",
        "before": {
            "protocol": blk["protocol"],
            "rubric": blk["rubric"],
            "sem_fallback": "none",
            "thresholds": [0.4, 0.6, 0.8],
        },
        "env": v2b_env,
        "opt_in": f"{SEMFILTER_V2B_ENV}=1",
        "reason": reason,
        "recorded_at": now,
    }
    return blk[SEMFILTER_V2B_KEY]


# ------------------------------------------------------------------ staging (staged_pruning)


def staging_of(c: dict) -> dict:
    return c.setdefault(
        "staging",
        {"stage_reached": 0, "pruned_at": None, "decisions": [], "episodes": 0},
    )


def config_order(state: dict, c: dict, strata: dict[str, str]) -> list[str]:
    """The configuration's nested-stage order (stratified prefixes of its own sample); recorded in state.json and
    verified on resume."""
    key = space.key(c["config"])
    seed = sp.seed_of(sampling_namespace(state), "stages", key)
    if (
        c["sample"].get("sampling") == "uniform"
    ):  # sample-size-v1: uniform prefixes (no strata)
        order = random.Random(seed).sample(
            sorted(c["sample"]["ids"]), len(c["sample"]["ids"])
        )
    else:
        order = sp.stage_order(c["sample"]["ids"], strata, seed)
    g = staging_of(c)
    if "order" in g and (g["order"] != order or g["order_seed"] != seed):
        # a branched run (pool_branch) keeps the parent's recorded order: valid iff it re-derives from its own seed
        if state.get("branched_from") and g["order"] == sp.stage_order(
            c["sample"]["ids"], strata, g["order_seed"]
        ):
            return g["order"]
        raise SystemExit(
            f"recorded stage order of {key} differs from the recomputed one"
        )
    g.update(order_seed=seed, order=order)
    return order


def episodes_used(state: dict) -> int:
    """Configuration episodes spent (the budget's unit), round 0 included (v2): the episodes actually run on the
    configuration's own sample (``staging.episodes``; a resumed round 0 may hold episodes outside its stage prefix),
    at least the last stage reached (NoMemory and the noise repeat excluded, as in the unpruned plan)."""
    return sp.episodes_used(
        {
            f"{rr['round']}:{j}": max(
                staging_of(c)["stage_reached"], staging_of(c)["episodes"]
            )
            for rr in state["rounds"]
            for j, c in enumerate(rr["configs"])
        }
    )


def observed_samples(state: dict) -> dict[str, list[str]]:
    """config key -> the queries it is OBSERVED on (BO / GP input): the whole own sample when fully evaluated, the
    prefix of its last stage when pruned (partial-n mean, larger SE = larger GP noise)."""
    out = {}
    for rr in state["rounds"]:
        for c in rr["configs"]:
            ids = c["sample"]["ids"]
            n = staging_of(c)["stage_reached"]
            if n >= len(ids):
                out[space.key(c["config"])] = list(ids)
            elif n > 0:
                out[space.key(c["config"])] = list(staging_of(c)["order"][:n])
    return out


def full_keys(state: dict) -> set[str]:
    """Configurations evaluated on their whole own sample (selection / router candidates)."""
    return {
        space.key(c["config"])
        for rr in state["rounds"]
        for c in rr["configs"]
        if staging_of(c)["stage_reached"] >= len(c["sample"]["ids"])
    }


def pruned_entries(state: dict) -> list[dict]:
    return [
        {
            "key": space.key(c["config"]),
            "round": rr["round"],
            "position": j,
            **{k: v for k, v in staging_of(c).items() if k != "order"},
        }
        for rr in state["rounds"]
        for j, c in enumerate(rr["configs"])
        if staging_of(c)["pruned_at"]
    ]


def row_key(r: dict) -> str:
    return space.key(space.validate(r["protocol"]["space_config"]))


def restrict_rows(
    rows: list[dict], observed: dict[str, list[str]], recorded: dict[str, list[str]]
) -> list[dict]:
    """Rows of this round's configurations restricted to what each is observed on (rows of a configuration's later,
    not yet decided stage are left out); rows of unknown configurations are kept, so ``config_estimates`` still
    refuses them as protocol errors."""
    obs = {k: set(v) for k, v in observed.items()}
    return [
        r
        for r in rows
        if row_key(r) not in recorded or str(r["query_id"]) in obs.get(row_key(r), ())
    ]


def obj_of(e: dict) -> sp.Obj:
    """``config_estimates`` entry -> pruning objectives (quality, tokens, latency_ms with SEs)."""
    q, t, lat = e["quality"], e["tokens"], e["latency_ms"]
    return sp.Obj(q.mean, t.mean, lat.mean, q.se, t.se, lat.se, q.n)


def estimates_on(
    rows: list[dict], samples: dict[str, list[str]], pool: dict, experiment: str
) -> dict[str, dict]:
    """``config_estimates`` of the given configurations on the given query sets only."""
    bo = bo_module()
    strata, w, sizes = pool_meta(pool)
    sets = {k: set(v) for k, v in samples.items()}
    keep = [
        r
        for r in rows
        if r["experiment_id"] == experiment
        and str(r["query_id"]) in sets.get(row_key(r), ())
    ]
    return bo.config_estimates(
        keep,
        samples=samples,
        strata=strata,
        weights=w,
        sizes=sizes,
        experiment=experiment,
    )


def nomemory_order(state: dict, strata: dict[str, str]) -> list[str]:
    """NoMemory's nested-stage order (stratified prefixes of its own sample; v2 stages NoMemory alongside round 0),
    recorded in state.json and verified on resume."""
    nm = state["nomemory"]
    seed = sp.seed_of(state["experiment"], "stages", "NoMemory")
    order = sp.stage_order(nm["sample"]["ids"], strata, seed)
    if "order" in nm and (nm["order"] != order or nm["order_seed"] != seed):
        # branched run: keep the parent's recorded order iff it re-derives from its own seed (see config_order)
        if state.get("branched_from") and nm["order"] == sp.stage_order(
            nm["sample"]["ids"], strata, nm["order_seed"]
        ):
            return nm["order"]
        raise SystemExit(
            "recorded NoMemory stage order differs from the recomputed one"
        )
    nm.update(order_seed=seed, order=order)
    return order


def nomemory_obj(
    ds: str,
    pool: dict,
    state: dict,
    nm_dir: Path,
    ids: list[str] | None = None,
    *,
    max_round: int | None = None,
) -> sp.Obj | None:
    """NoMemory's stratified quality / tokens / latency over ``ids`` (default: its whole own sample; the prune rule
    passes its stage prefix). Cost from ``nomemory_rows``; items without a cost record are left out and logged.
    None when < 2 items have costs."""
    strata, w, sizes = pool_meta(pool)
    ids = list(nomemory_ids(state) if ids is None else ids)
    rows = nomemory_rows(ds, nm_dir, ids, max_round=max_round, raw_score=bool(state.get("ablation_arm")))
    have = [q for q in ids if q in rows]
    if len(have) < len(ids):
        log(f"prune comparators: NoMemory costs for {len(have)}/{len(ids)} queries")
    if len(have) < 2:
        return None
    h = [strata[q] for q in have]
    est = {
        f: us.stratified([rows[q][f] for q in have], h, w, sizes)
        for f in ("quality", "tokens", "e2e_ms")
    }
    return sp.Obj(
        est["quality"].mean,
        est["tokens"].mean,
        est["e2e_ms"].mean,
        est["quality"].se,
        est["tokens"].se,
        est["e2e_ms"].se,
        len(have),
    )


def prune_front(
    ds: str, pool: dict, state: dict, rows: list[dict], nm_dir: Path
) -> dict[str, sp.Obj]:
    """Point estimates of every fully evaluated configuration (the FULL comparators of the prune rule)."""
    obs = observed_samples(state)
    fk = full_keys(state)
    return {
        k: obj_of(e)
        for k, e in estimates_on(
            rows, {k: obs[k] for k in fk if k in obs}, pool, state["experiment"]
        ).items()
    }


def stage_comparators(
    ds: str,
    pool: dict,
    state: dict,
    rows: list[dict],
    nm_dir: Path,
    n: int,
    full: set[str],
    *,
    max_round: int | None = None,
) -> dict[str, sp.Obj]:
    """STAGE comparators at checkpoint n (pessimistic corners): NoMemory on the n-prefix of its own sample, and
    every configuration evaluated at >= n that is not fully evaluated (pruned ones included) on the n-prefix of its
    own order (samples are per configuration: unpaired)."""
    strata, _, _ = pool_meta(pool)
    out = {}
    nm = nomemory_obj(
        ds,
        pool,
        state,
        nm_dir,
        nomemory_order(state, strata)[:n],
        max_round=max_round,
    )
    if nm is not None:
        out[space.key(space.NOMEM)] = nm
    sub = {
        space.key(c["config"]): staging_of(c)["order"][:n]
        for rr in state["rounds"]
        for c in rr["configs"]
        if staging_of(c)["stage_reached"] >= n
        and space.key(c["config"]) not in full
        and "order" in staging_of(c)
    }
    est = estimates_on(rows, sub, pool, state["experiment"]) if sub else {}
    missing = set(sub) - set(est)
    if missing:
        raise SystemExit(
            f"stage {n}: {sorted(missing)} recorded but without a complete estimate"
        )
    return out | {k: obj_of(e) for k, e in est.items()}


GP_NOISE = (
    "quality: the configuration's own stratified SE^2 (per-configuration samples: no shared queries, the "
    "NoMemory term is a constant; unchanged); costs: delta-method log SE^2 (unchanged)"
)


def router_backends(state: dict, a, now: str | None = None) -> bool:
    """True when the LimeQO backend runs. ``now`` given (profile stage): record the choice in state.json
    ``router_backends`` if absent. A recorded choice must match this invocation's flag (refuse to mix)."""
    want = bool(getattr(a, "limeqo", True))
    blk = state.get("router_backends")
    if blk is None:
        if now is not None:
            backend = getattr(a, "router_backend", "brane-config")
            primary = "joint-selected" if backend == "joint-selected" else "BRANE"
            state["router_backends"] = {
                "protocol": ROUTER_BACKENDS_PROTOCOL,
                "backends": [primary, "LimeQO"] if want else [primary],
                "limeqo": want,
                "arms": f"Opt-OneConfig SLO-1/2/3 + Opt-PerQuery ({primary}) at TEST_WEIGHTS"
                + (" + Opt-PerQuery-LimeQO at TEST_WEIGHTS" if want else ""),
                "recorded_at": now,
                "note": "user 2026-10-21"
                + (
                    ""
                    if want
                    else f": no LimeQO for this run (--no_limeqo / {NO_LIMEQO_ENV}=1); the code is unchanged"
                ),
            }
        return want
    if blk["limeqo"] != want:
        raise SystemExit(
            f"state.json router_backends records limeqo={blk['limeqo']}, this invocation {want}; "
            f"refusing to mix (set/unset --no_limeqo / {NO_LIMEQO_ENV})"
        )
    return want


# Router-backend change after profiling (user 2026-10-22, ALFWorld v7 re-test: "No LimeQO"): the recorded choice
# may be changed once per re-selection by an explicit opt-in (POOL_ROUTER_BACKENDS_CHANGE=1 + _REASON) in the router
# stage; the old block is kept in ``history``. Without the opt-in a mismatch is still refused (router_backends).
ROUTER_BACKENDS_CHANGE_ENV = "POOL_ROUTER_BACKENDS_CHANGE"
ROUTER_BACKENDS_REASON_ENV = "POOL_ROUTER_BACKENDS_REASON"


def router_backends_change(state: dict, a, now: str, env=None) -> bool:
    """Record a requested backend change in ``state`` (True when changed). Pure on (state, a, env)."""
    env = os.environ if env is None else env
    want = bool(getattr(a, "limeqo", True))
    blk = state.get("router_backends")
    if blk is None or blk["limeqo"] == want:
        return False
    if env.get(ROUTER_BACKENDS_CHANGE_ENV) != "1":
        return False  # router_backends refuses the mismatch
    reason = (env.get(ROUTER_BACKENDS_REASON_ENV) or "").strip()
    if not reason:
        raise SystemExit(
            f"{ROUTER_BACKENDS_CHANGE_ENV}=1 needs {ROUTER_BACKENDS_REASON_ENV}"
        )
    hist = blk.get("history", [])
    state["router_backends"] = {
        "protocol": ROUTER_BACKENDS_PROTOCOL,
        "backends": ["BRANE", "LimeQO"] if want else ["BRANE"],
        "limeqo": want,
        "arms": "Opt-OneConfig SLO-1/2/3 + Opt-PerQuery (BRANE) at TEST_WEIGHTS"
        + (" + Opt-PerQuery-LimeQO at TEST_WEIGHTS" if want else ""),
        "recorded_at": now,
        "note": reason,
        "history": [*hist, {k: v for k, v in blk.items() if k != "history"}],
    }
    return True


# Catalog regime (catalog-regime-v1, user 2026-10-22, ALFWorld v7 re-test): router training, Opt-OneConfig selection,
# the SLO bounds and the outer CV use ONLY the configurations of rounds >= from_round (the rounds profiled under the
# current runtime / SemFilter regime: latency-regime-v1 c16 + semfilter-regime-v1 v2 from round 6); earlier rounds'
# latency (c128) and SemFilter v1 judge are not comparable. Fully evaluated configurations only (pruned partial
# samples never enter, unchanged). The NoMemory candidate / gain anchor is the latency regime's NoMemory rerun
# (apply_regime). Opt-in POOL_CATALOG_FROM_ROUND=<r> + POOL_CATALOG_REASON in the router stage records it in
# state.json once; afterwards every stage applies the recorded block (a different request is refused).
CATALOG_PROTOCOL = "catalog-regime-v1"
CATALOG_FROM_ENV = "POOL_CATALOG_FROM_ROUND"
CATALOG_REASON_ENV = "POOL_CATALOG_REASON"


def catalog_want(env=None) -> int | None:
    env = os.environ if env is None else env
    v = (env.get(CATALOG_FROM_ENV) or "").strip()
    return int(v) if v else None


def catalog_block(state: dict, now: str | None = None, env=None) -> dict | None:
    """The recorded catalog block (None: every round, the old behaviour). ``now`` given (router stage): record a
    requested block if absent. Refuses a request that differs from the recorded block, or a from_round below the
    recorded latency / SemFilter regime switches. Mutates ``state`` only when recording. Pure otherwise."""
    env = os.environ if env is None else env
    want = catalog_want(env)
    blk = state.get("catalog_regime")
    if blk is not None:
        if blk.get("protocol") != CATALOG_PROTOCOL:
            raise SystemExit(
                f"state.json catalog_regime {blk.get('protocol')!r}; refusing"
            )
        if want is not None and want != blk["from_round"]:
            raise SystemExit(
                f"state.json catalog_regime from round {blk['from_round']}, requested {want}; refusing to mix"
            )
        return blk
    if want is None:
        return None
    if now is None:
        raise SystemExit(
            f"{CATALOG_FROM_ENV}={want} but no catalog_regime recorded: run the router stage first"
        )
    floors = {
        name: (state.get(name) or {}).get("from_round")
        for name in ("latency_regime", "semfilter")
    }
    low = [
        f"{n} from round {r}" for n, r in floors.items() if r is not None and want < r
    ]
    if low:
        raise SystemExit(f"catalog from round {want} mixes regimes ({', '.join(low)})")
    reason = (env.get(CATALOG_REASON_ENV) or "").strip()
    if not reason:
        raise SystemExit(f"{CATALOG_FROM_ENV} needs {CATALOG_REASON_ENV}")
    rounds = [rr["round"] for rr in state["rounds"]]
    state["catalog_regime"] = {
        "protocol": CATALOG_PROTOCOL,
        "from_round": want,
        "included_rounds": [r for r in rounds if r >= want],
        "excluded_rounds": [r for r in rounds if r < want],
        "candidates": "fully evaluated configurations of the included rounds (pruned partial samples excluded)",
        "regimes": {
            "latency": (state.get("runtime_profile") or {}).get("name"),
            "semfilter": (state.get("semfilter") or {}).get("protocol"),
            "nomemory": (state.get("latency_regime") or {})
            .get("nomemory", {})
            .get("dir"),
            # max-round-v1 (WebShop): every included round compared in the derived max_round-n view
            **({"episode_budget": eb.runtime_label(state)} if eb.block(state) else {}),
        },
        "scope": "router training, Opt-OneConfig selection, SLO bounds, outer CV (train only; test untouched)",
        "reason": reason,
        "recorded_at": now,
    }
    return state["catalog_regime"]


def record_post_profile(ds: str, a) -> None:
    """Router stage: record a requested catalog regime / router-backend change in state.json (atomic save)."""
    p = run_dir(ds, a.sampling) / "state.json"
    if not p.exists():
        return
    st = json.loads(p.read_text())
    now = datetime.now(UTC).isoformat()
    had = st.get("catalog_regime")
    changed = router_backends_change(st, a, now)
    blk = catalog_block(st, now)
    if changed or (blk is not None and had is None):
        save(p, st)
        log(
            f"state.json: catalog_regime {blk and blk['from_round']}, router_backends "
            f"{st['router_backends']['backends']} (recorded)"
        )


def plan_round(
    ds: str,
    pool: dict,
    state: dict,
    r: int,
    nm: us.Estimate,
    a,
    k: int | None = None,
) -> list[dict]:
    """Round ``r``'s picks (channel-slots-v3: each pick carries its ``coverage`` deltas)."""
    bo = bo_module()
    search_settings = sa.settings_for(state, a)
    # proposable plans of round r (plan-p1-v1: the gated P1 only from the recorded extension round)
    _PLANS_NOW[:] = [bo.proposal_plans(state, r)]
    bo.use_plans(_PLANS_NOW[-1])
    # semfilter-v2b (WebShop): proposed SemFilter configurations carry fill-topk, thresholds {0.4, 0.6}
    if bo.use_semfilter_v2b_for(state, r):
        log(
            f"round {r} semfilter-v2b proposal rule: SemFilter plans sem_fallback fill-topk, thresholds "
            f"{state['semfilter'][SEMFILTER_V2B_KEY]['thresholds']}"
        )

    k = a.batch if k is None else k
    if r == 0:
        if sa.baselines.active(search_settings):
            return sa.baselines.initialize(bo, search_settings)
        sampler = sa.initialization_sampler(search_settings)
        if (state.get(ROUND0_DESIGN_KEY) or {}).get("sampler") == "sobol-only":
            return [{"config": c, "channel": "sobol"} for c in round0_sobol(bo, k)]
        return [
            {"config": c, "channel": "default" if i < len(bo.defaults()) else sampler}
            for i, c in enumerate(
                bo.round0() if sampler == "sobol" else bo.round0(sampler=sampler)
            )
        ]
    # gp-fix-v1 from the recorded round on (priors; the own-sample SE^2 noise is unchanged)
    bo.use_gp_for(state, r)
    # latency-regime-v1: old-regime latencies enter the latency GP with a regime input (never mixed as-is)
    if bo.use_latency_regime_for(state, r):
        log(
            f"round {r} {LATENCY_REGIME_PROTOCOL}: latency GP with a regime input (old regime = rounds < "
            f"{state['latency_regime']['from_round']})"
        )
    # semfilter-regime-v1: old-judge SemFilter observations get a regime input in all three GPs
    if bo.use_semfilter_regime_for(state, r):
        log(
            f"round {r} {SEMFILTER_REGIME_PROTOCOL}: GP semfilter-regime input (old judge = SemFilter configs of "
            f"rounds < {sf_regime_from(state, r)})"
        )
    # hv-ref-anchored-v1 from the recorded round on: BO quality = gain vs NoMemory, so the NoMemory baseline is 0
    # in objective units; its SE = the NoMemory arm's own stratified SE
    if bo.use_hv_ref_for(state, r, (0.0, nm.se)):
        log(
            f"round {r} {bo.pool.HV_REF_ANCHORED}: quality ref = 0 - NoMemory SE {nm.se:.4f} (NoMemory q {nm.mean:.3f})"
        )
    pts = points(ds, pool, state, nm, a.estimator, r=r)
    measured = {space.key(p.cfg) for p in pts}
    planned = set(samples_of(state))
    if planned - measured:
        raise SystemExit(
            f"round {r}: {len(planned - measured)} planned configurations lack a complete stage"
        )
    if sa.baselines.active(search_settings):
        picks = sa.baselines.propose(
            pts,
            bo.context(ds, pool["pool_hash"]),
            k,
            search_settings,
            r,
            # agentic-autorag-v1 only: per-task TRAIN records + the previous round's configurations
            **(
                {
                    "run_dir": run_dir(ds, getattr(a, "sampling", "per-config")),
                    "recent": [
                        c["config"]
                        for rr in state["rounds"][r - 1 : r]
                        for c in rr["configs"]
                    ],
                }
                if search_settings["protocol"] == "agentic-autorag-v1"
                else {}
            ),
        )
        log(
            f"round {r}: {search_settings['protocol']} selected {len(picks)} configurations (no channel pool)"
        )
        return picks
    by = {space.key(p.cfg): p for p in pts}
    forced = bo.stalled(
        [[by[space.key(c["config"])] for c in rr["configs"]] for rr in state["rounds"]]
    )
    if search_settings["protocol"] == sa.HYDRA_PROTOCOL:
        import sys
        from trimem.optimizer import profiling_driver

        agent, _, advice = analyst(ds, pool, state, pts, forced, r, nm, a)
        picks = profiling_driver.propose(
            sys.modules[__name__],
            ds,
            pool,
            state,
            r,
            nm,
            a,
            pts,
            k,
            agent_rows=agent or (),
            forced=forced,
        )
        log(f"round {r}: hydra-local-v1 selected {len(picks)} configurations")
        return [{**p, "analyst": advice} for p in picks]
    # aux-obs-v1 (user 2026-10-20): GP-only observations of the earlier train-panel runs from the recorded round
    aux = bo.aux_points(state, r, bo.task_of(ds, pool["pool_hash"]))
    if aux and eb.view_max_round(state, r) is not None:
        # max-round-v1: the aux points are 30-turn measurements of another protocol (not re-derived): off
        bo.pool.use_aux(False)
        aux = []
        log(f"round {r} {eb.PROTOCOL}: auxiliary GP observations off (30-turn panel)")
    if aux:
        log(
            f"round {r} {bo.AUX_PROTOCOL}: {len(aux)} auxiliary GP observations (source dimension on)"
        )
    slots = bo.channel_slots_settings(state, r)
    recovery = sa.stall_recovery_enabled(search_settings)
    sobol_only = bool(slots) and slots["protocol"] == bo.CHANNEL_SLOTS_SOBOL
    if bool(slots) and slots["protocol"] in bo.ONE_PER_CHANNEL and not recovery:
        forced = False  # channel-slots-v5: never stall-forced (the analyst is called once, ordinary mode)
    arm = getattr(a, "ablation_arm", None)
    no_ablation_llm = arm == "without_llm"
    if sobol_only or no_ablation_llm:
        # sobol-only-v1 (random-sampling round): never forced, no analyst call (no llm slot)
        if sobol_only:
            forced = False
        agent, gap_front = [], None
        advice = {
            "protocol": bo.CHANNEL_SLOTS_SOBOL,
            "status": "skipped: ablation disables analyzer" if no_ablation_llm else "skipped: sobol-only round",
        }
    else:
        agent, gap_front, advice = analyst(ds, pool, state, pts, forced, r, nm, a)
    # channel-slots-v4 / sobol-only-v1: no qfirst pick (propose_batch ignores qfirst in those rounds too)
    v4 = bool(slots) and slots["protocol"] in (
        bo.CHANNEL_SLOTS_V4,
        bo.CHANNEL_SLOTS_SOBOL,
        *bo.ONE_PER_CHANNEL,
    )
    qf = None if v4 else bo.qfirst_settings(state, r, pts)
    search_options = sa.engine_options(search_settings)
    if arm:
        from trimem.experiments.scienceworld_trace_ablation import arm_policy

        search_options["ablation_policy"] = arm_policy(arm)
    if search_settings["protocol"] == sa.TC_PROTOCOL:
        import sys

        from trimem.optimizer import task_complementary_driver as tcd

        # task-complementary-v1: only the dls slot uses this scorer (session_pool.suggest ``tc``)
        search_options["tc"] = tcd.build_scorer(
            sys.modules[__name__], ds, pool, state, r, a
        )
        log(
            f"round {r}: task-complementary-v1 dls scorer {search_options['tc'].status}"
        )
    picks = bo.propose_batch(
        pts,
        bo.context(ds, pool["pool_hash"]),
        k=k,
        seed=(a.seed + 1000 * r) if arm else 1000 * r,
        forced=forced,
        agent_rows=agent or None,
        realized_front=gap_front,
        qfirst=qf,
        slots=slots,
        aux=aux,
        search_options=search_options,
        stall_recovery=recovery,
    )
    if slots:
        log(
            f"round {r} {slots['protocol']}: "
            + (
                (
                    "FORCED (forced logic + xover/dls/sobol coverage fill)"
                    if slots["protocol"]
                    in (
                        bo.CHANNEL_SLOTS_V2,
                        bo.CHANNEL_SLOTS_V3,
                        bo.CHANNEL_SLOTS_V4,
                        *(bo.ONE_PER_CHANNEL if recovery else ()),
                    )
                    else "FORCED (forced logic unchanged)"
                )
                if forced
                else "ordinary"
            )
        )
        for line in bo.slot_log_lines(r, picks):
            log(line)
        if slots["protocol"] in bo.CHANNEL_SLOTS_NAMED:
            cov = bo.round_coverage(picks, pts)
            for line in bo.coverage_log_lines(r, cov):
                log(line)
    for p in picks:
        if p["channel"] == "qfirst":
            q = p["qfirst"]
            log(
                f"round {r} qfirst slot ({q['protocol']}, beta {q['beta']}, target {q['quality_target']}, budget "
                f"{q['budget_tokens']:,.0f} tok / {q['budget_latency_ms'] / 1000:.1f} s; via {p['source_channel']}): "
                f"{space.key(p['config'])} pred gain {p['pred_gain']:+.3f}"
            )
    return [
        {
            **p,
            "forced": forced,
            "analyst": advice,
            **({"stall_recovery": sa.STALL_RECOVERY} if recovery else {}),
        }
        for p in picks
    ]


def analyst_status(state: dict) -> dict[str, dict]:
    """config key -> staged status (n evaluated of the full sample, pruned) for the analyst's cards."""
    return {
        space.key(c["config"]): {
            "n": staging_of(c)["stage_reached"],
            "n_full": len(
                c["sample"]["ids"]
            ),  # sample-size-v1: 100 for uniform samples
            "pruned": staging_of(c)["pruned_at"] is not None,
        }
        for rr in state["rounds"]
        for c in rr["configs"]
    }


def analyst(
    ds: str,
    pool: dict,
    state: dict,
    pts: list,
    forced: bool,
    r: int,
    nm: us.Estimate,
    a,
) -> tuple[list, list | None, dict]:
    """LLM analyst channel (``analyzer_pool``; user 2026-10-20): TRAIN profiling evaluations -> at most 4
    in-space edits labelled agent / agent-gap that enter the candidate pool. Never fatal (no candidates on
    any failure: the round uses the other channels). ``--no_analyst`` disables it."""
    from trimem.optimizer import analyzer_pool as ap

    if not getattr(a, "analyst", True):
        return (
            [],
            None,
            {
                "protocol": state.get("search_design", {}).get("protocol", ap.PROTOCOL),
                "status": "disabled",
            },
        )
    ctx = {
        "dataset": ds,
        "panel": f"stratified train pool of {pool.get('size', '?')} tasks, per-configuration samples of "
        f"{state.get('full_panel_size', N_SAMPLE)} (staged {state.get('staging', {}).get('sizes', list(STAGE_SIZES))})",
        "quality_text": "the optimizer's quality objective = stratified mean success/score of the configuration "
        f"on its own sample minus the NoMemory mean ({nm.mean:.3f}); unpaired, mean +- SE",
    }
    if state.get("ablation_arm"):
        ctx["quality_text"] = ("ScienceWorld raw environment score (negative scores retained), "
                               f"stratified mean minus NoMemory mean ({nm.mean:.3f}); unpaired, mean +- SE")
    if state.get("search_design", {}).get("protocol") in (
        *sa.THREE_CHANNEL,
        sa.HYDRA_PROTOCOL,
    ):
        chat_options = {}
        if state.get("ablation_arm"):
            from trimem.experiments.scienceworld_trace_ablation import BudgetedAnalyzer
            from trimem.optimizer.analyzer_session import CodexCLI

            _, resources = ablation_ledgers(state)
            settings = state["search_design"]
            client = CodexCLI(model=settings["model"], effort=settings["effort"],
                              reasoning_summary=settings["reasoning_summary"])
            chat_options["chat"] = BudgetedAnalyzer(client, resources, state["ablation_arm"], r)
        rows, front, summary = sa.advise(
            space,
            pts,
            state,
            status=analyst_status(state),
            forced=forced,
            rnd=r,
            ctx=ctx,
            out_dir=run_dir(ds, a.sampling),
            settings=state["search_design"],
            **chat_options,
        )
        log(
            f"analyzer round {r}: {summary['status']}, {len(rows)} local starts ({summary['log']})"
        )
        return rows, front, summary
    conf = None
    try:
        from trimem.optimizer import analyzer_session as az
        from trimem.optimizer import session_pool as sp_pool

        conf = az.confidence_text(sp_pool.fit(pts)[0])
    except Exception as e:  # noqa: BLE001 - the confidence card is optional
        log(f"analyst round {r}: no confidence card ({type(e).__name__}: {e})")
    rows, front, summary = ap.advise(
        PlanView(space, bo_module().pool.RETRIEVAL_PLANS),
        pts,
        state,
        status=analyst_status(state),
        forced=forced,
        rnd=r,
        ctx=ctx,
        out_dir=run_dir(ds, a.sampling),
        confidence=conf,
    )
    log(
        f"analyst round {r} ({summary['label']}): {summary['status']}, {summary['candidates']} candidates, "
        f"{summary['rejects']} rejects{'' if rows else ' -> other channels only'} ({summary['log']})"
    )
    return rows, front, summary


class PlanView:
    """The active space as the analyst sees it in a round: ``PLANS`` = the round's proposable plans (plan-p1-v1:
    P1 hidden before its extension round); everything else delegates to the space module."""

    def __init__(self, module, plans):
        self._m, self.PLANS = module, tuple(plans)

    def __getattr__(self, name):
        return getattr(self._m, name)

    def domain(self, plan):
        """semfilter-v2b: the proposal domain (thresholds {0.4, 0.6}) while a proposal rule is active; otherwise
        the space's domain (unchanged)."""
        from trimem.optimizer import session_pool as sp_pool

        if sp_pool.proposal_rule() is None:
            return self._m.domain(plan)
        return sp_pool.proposal_domain(plan)


PLAN_P1_ENV = "POOL_PLAN_P1"


def plan_p1_want(a) -> bool:
    """plan-p1-v1 opt-in: --plan_p1 or POOL_PLAN_P1=1."""
    return bool(getattr(a, "plan_p1", False)) or os.environ.get(
        PLAN_P1_ENV, ""
    ).strip() in ("1", "true", "yes", "on")


def fixed_test_plans(ds: str, a) -> tuple[str, ...]:
    """Plans of the optional fixed-plan test arms: the ungated plans + the recorded plan extensions."""
    p = run_dir(ds, a.sampling) / "state.json"
    st = json.loads(p.read_text()) if p.exists() else {"rounds": []}
    return bo_module().proposal_plans(st, 10**6)


def round_label(r: int, j: int, c: dict) -> str:
    return f"r{r}c{j}:{space.DISPLAY[c['config']['plan']]}"


REUSE_SUMMARY_SQL = (
    "SELECT protocol->'space_config' AS cfg, count(*) AS n, count(generation->>'reused_from') AS hits "
    "FROM opt.session_retrieval WHERE experiment_id = %s AND split = %s AND status = 'evaluated' GROUP BY 1"
)


def log_reuse(experiment: str, split: str, what: str) -> None:
    """Quality-reuse hit rate per configuration (opt rows: ``generation.reused_from``); never fails the run."""
    try:
        import psycopg

        with psycopg.connect(drv.OPT_DSN) as c:
            rows = [
                tuple(r)
                for r in c.execute(REUSE_SUMMARY_SQL, (experiment, split)).fetchall()
            ]
        if not rows:
            return
        n, h = sum(r[1] for r in rows), sum(r[2] for r in rows)
        per = ", ".join(
            f"{cfg_name(r[0]) if r[0] else '?'} {r[2]}/{r[1]}" for r in rows
        )
    except Exception as exc:  # noqa: BLE001 -- logging only
        log(f"quality reuse summary unavailable: {exc!r}")
        return
    log(
        f"quality reuse {what} ({split}): hits/evaluated {h}/{n}"
        + (f" ({h / n:.0%})" if n else "")
        + f" -- {per}"
    )


def reuse_block(state: dict, a) -> dict | None:
    """quality-reuse-v1 (user 2026-10-20): ALFWorld / ScienceWorld from their start; recorded once in state.json."""
    if not a.reuse:
        return state.get("quality_reuse")
    if "quality_reuse" not in state:
        state["quality_reuse"] = {
            "protocol": qr.PROTOCOL,
            "from_round": len(state["rounds"]),
            "recorded_at": datetime.now(UTC).isoformat(),
            "key": "opt rows of the same experiment, split, query (item; 1 trial), panel_hash, memory policy, "
            "memory_sha256 (exact injected text) AND memory_key = sha256(memory_form, render version, text)",
            "copied": "reward / quality, generation tokens and episode ms of the source; own semantic tokens and "
            "retrieval ms (agentgym_session_eval.reuse)",
            "scope": "profiling and the test stage (test rows: panel_hash test:<TEST_VERSION>); never across "
            "train / test or experiments; NoMemory episodes have no opt row, so they are never a source",
            "empty_memory": qr.EMPTY_MEMORY_RULE,
            "in_flight": "not waited for (opt-row lookup): only completed episodes are sources",
            "budget": "reuse hits count as episodes in the ALF/Sci staging budget (unchanged rule)",
            "backfill": "none needed: the -v5 runs start with this protocol",
            "se_caveat": "reuse fixes the environment / agent randomness for identical inputs: SEs of differences "
            "between configurations that share reused episodes are understated",
        }
    return state["quality_reuse"]


def reuse_sources(state: dict, r: int) -> tuple[list[str], list[str]]:
    """(source experiments, their config directories) of cross-experiment quality reuse in round ``r``: state.json
    ``quality_reuse.sources`` (recorded by ``pool_branch``; each {experiment, configs_dir, from_round}). Empty for
    every state without sources (all untagged runs). Training arms only (never noise repeats or test arms)."""
    srcs = (state.get("quality_reuse") or {}).get("sources") or []
    on = [x for x in srcs if r >= x["from_round"]]
    return [x["experiment"] for x in on], [x["configs_dir"] for x in on]


def _run_or_stop(ds: str, arms: list[Arm], what: str, *, allow_restart: bool = True) -> None:
    # run_capped: never above CONCURRENCY[ds] in flight even with more arms than CONCURRENCY[ds] // 2
    res = (run_capped(ds, arms, CONCURRENCY[ds]) if allow_restart
           else run_capped(ds, arms, CONCURRENCY[ds], allow_restart=False))
    exp = next((x.session["experiment"] for x in arms if x.session), None)
    if exp:
        log_reuse(exp, "train", what)
    bad = [x.name for x in arms if res[x.name] < len(x.ids)]
    if bad:
        raise SystemExit(
            f"{what}: {bad} incomplete after {drv.ATTEMPTS} failed attempts; resume to retry"
        )


def noise_repeat_arm(ds: str, pool: dict, state: dict, a) -> Arm:
    base, exp = run_dir(ds, a.sampling), state["experiment"]
    nr, rep = state["noise_repeat"]["sample"], state["noise_repeat"]["config"]
    return Arm(
        "noise-repeat VGR",
        base / "noise_repeat_VGR",
        nr["ids"],
        session_args(
            ds,
            rep,
            nr,
            pool,
            experiment=exp + ":noise-repeat",
            baseline_dir=nomemory_dir(ds, a.sampling),
            label="noise-repeat:VGR",
            sampling=a.sampling,
            reuse=False,
        ),
    )


def spent(rows: list[dict], key: str, ids: list[str]) -> int:
    """Evaluated opt rows of a configuration on its own sample (episodes actually run, the budget's unit)."""
    want = set(ids)
    return len(
        {
            str(r["query_id"])
            for r in rows
            if r.get("status", "evaluated") == "evaluated"
            and row_key(r) == key
            and str(r["query_id"]) in want
        }
    )


def ablation_ledgers(state):
    from trimem.experiments import scienceworld_trace_ablation as ab

    manifest_hash = state["ablation_manifest_sha256"]
    return (
        ab.EpisodeLedger.open(state["ablation_episode_ledger"], manifest_hash),
        ab.ResourceLedger.open(state["ablation_resource_ledger"], state.get("ablation_resource_ledger_binding", manifest_hash), state["ablation_resource_limits"]),
    )


def reserve_ablation_candidate(state, cfg, ids):
    from trimem.experiments import scienceworld_trace_ablation as ab

    episodes, resources = ablation_ledgers(state)
    ab.reserve_candidate_prefix(episodes, resources, state["ablation_arm"], space.key(cfg), ids)


def reconcile_ablation_candidate(state, cfg, ids, rows):
    from trimem.experiments import scienceworld_trace_ablation as ab

    key, wanted = space.key(cfg), set(ids)
    selected = [{**row, "config_key": key} for row in rows if row_key(row) == key and str(row["query_id"]) in wanted]
    episodes, resources = ablation_ledgers(state)
    ab.reconcile_dispatched_candidates(episodes, resources, state["ablation_arm"], selected)


def ablation_auxiliary_job_id(state, category, query_id):
    from trimem.experiments import scienceworld_trace_ablation as ab

    return f"{category}:{ab._digest([state['ablation_arm'], category, query_id])}"


def reserve_ablation_auxiliary(state, arm):
    category = {"NoMemory": "nomemory", "noise-repeat VGR": "noise"}.get(arm.name)
    if category is None:
        raise ValueError("unknown ablation auxiliary arm")
    _, resources = ablation_ledgers(state)
    for q in arm.ids:
        resources.reserve(category, ablation_auxiliary_job_id(state, category, q))


def reconcile_ablation_auxiliary(ds, state, arm):
    category = {"NoMemory": "nomemory", "noise-repeat VGR": "noise"}.get(arm.name)
    if category is None:
        raise ValueError("unknown ablation auxiliary arm")
    completed = set(arm.completed(ds))
    _, resources = ablation_ledgers(state)
    if category == "nomemory":
        measurements = nomemory_rows(ds, arm.out, arm.ids, raw_score=True)
        for q in completed:
            row = measurements.get(q)
            if row is None or row.get("cost_source") != "episodes":
                raise SystemExit("NoMemory cost receipt missing or approximate; ablation paused")
            resources.complete(ablation_auxiliary_job_id(state, category, q), tokens=row["tokens"],
                               elapsed_s=float(row["e2e_ms"]) / 1000, status="complete",
                               metadata={"query_id": q, "cost_source": row["cost_source"], "elapsed_kind": "episode_latency_not_wall_time"})
    else:
        rows = drv.train_rows(ds, state["experiment"] + ":noise-repeat", include_retrieval_id=True)
        by_id = {str(row["query_id"]): row for row in rows if str(row["query_id"]) in set(arm.ids)}
        if len(by_id) != sum(str(row["query_id"]) in set(arm.ids) for row in rows):
            raise ValueError("duplicate noise-repeat measurements")
        for q in completed:
            row = by_id.get(q)
            if row is None:
                raise SystemExit("noise-repeat cost receipt missing; ablation paused")
            resources.complete(ablation_auxiliary_job_id(state, category, q), tokens=row["total_tokens"],
                               elapsed_s=float(row["e2e_ms"]) / 1000, status="complete",
                               metadata={"query_id": q, "retrieval_id": str(row["retrieval_id"]), "elapsed_kind": "episode_latency_not_wall_time"})


def mark_ablation_dispatch(ds, state, arms, round_configs, round_number):
    from trimem.experiments import scienceworld_trace_ablation as ab

    by_name = {round_label(round_number, j, c): c for j, c in enumerate(round_configs)}
    jobs = []
    for arm in arms:
        pending = set(arm.ids) - set(arm.completed(ds))
        if arm.name in by_name:
            key = space.key(by_name[arm.name]["config"])
            jobs.extend(ab.physical_episode_job_id(key, q, arm=state["ablation_arm"], shared_initialization=False) for q in sorted(pending))
        elif arm.name in ("NoMemory", "noise-repeat VGR"):
            category = "nomemory" if arm.name == "NoMemory" else "noise"
            jobs.extend(ablation_auxiliary_job_id(state, category, q) for q in sorted(pending))
        else:
            raise ValueError("unrecognized ablation dispatch arm")
    _, resources = ablation_ledgers(state)
    resources.mark_dispatch_many(jobs)


def run_staged_round(
    ds: str, pool: dict, state: dict, state_p: Path, r: int, a
) -> None:
    if not state.get("ablation_arm"):
        return _run_staged_round(ds, pool, state, state_p, r, a)
    from trimem.experiments.scienceworld_trace_ablation import worker_retry_environment

    changes = worker_retry_environment(state)
    previous = {key: os.environ.get(key) for key in changes}
    try:
        os.environ.update(changes)
        return _run_staged_round(ds, pool, state, state_p, r, a)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_staged_round(
    ds: str, pool: dict, state: dict, state_p: Path, r: int, a
) -> None:
    """Round r (round 0 included, v2): synchronous nested stages (50 -> 100 -> 200 of each configuration's own
    stratified order). Every live configuration of the batch runs stage s in parallel (freed slots rebalanced,
    ``run_parallel``) -- with NoMemory's prefix of the same size while it is incomplete and, at the last stage,
    the noise repeat in full; then each is decided at the checkpoint (``decide_stage``). Stage progress and
    decisions are stored in state.json and reused on resume; episodes already run outside a prefix are kept."""
    blk = state["staging"].get(ASYNC_KEY)
    if blk and r >= blk["from_round"]:
        return run_staged_round_async(ds, pool, state, state_p, r, a)
    base, exp = run_dir(ds, a.sampling), state["experiment"]
    nm_dir = nomemory_dir(ds, a.sampling)
    strata, _, _ = pool_meta(pool)
    rnd = state["rounds"][r]
    orders = [config_order(state, c, strata) for c in rnd["configs"]]
    nm_order = nomemory_order(state, strata)
    save(state_p, state)
    sizes = state["staging"]["sizes"]
    for si, n in enumerate(sizes):
        last = si == len(sizes) - 1
        todo = [
            (j, c)
            for j, c in enumerate(rnd["configs"])
            if staging_of(c)["pruned_at"] is None
            and staging_of(c)["stage_reached"] < n
            and n in cfg_sizes(state, c)
        ]
        mr = eb.max_round_for(state, r)  # max-round-v1: None (30) before its from_round
        extra = {
            "staging": state["staging"]["protocol"],
            "stage_n": n,
            **eb.protocol_extra(mr),
        }
        arms = [
            Arm(
                round_label(r, j, c),
                base / "configs" / cfg_name(c["config"]),
                orders[j][:n],
                session_args(
                    ds,
                    c["config"],
                    c["sample"],
                    pool,
                    experiment=exp,
                    baseline_dir=nm_dir,
                    label=round_label(r, j, c),
                    sampling=a.sampling,
                    reuse=a.reuse,
                    extra=extra,
                    reuse_from=reuse_sources(state, r) if a.reuse else ([], []),
                ),
                max_round=mr,
            )
            for j, c in todo
        ]
        side = [Arm("NoMemory", nm_dir, nm_order[:n], None)]
        if (
            last and r == 0
        ):  # the noise repeat belongs to round 0 (in full, with its last stage)
            side.append(noise_repeat_arm(ds, pool, state, a))
        for x in side:
            x.out.mkdir(parents=True, exist_ok=True)
        arms = [x for x in side if len(x.completed(ds)) < len(x.ids)] + arms
        if not arms:
            continue
        if state.get("ablation_arm"):
            existing_rows = opt_rows(ds, exp, state, r)
            for j, c in todo:
                reserve_ablation_candidate(state, c["config"], orders[j][:n])
                reconcile_ablation_candidate(state, c["config"], orders[j][:n], existing_rows)
            for x in arms:
                if x.name in ("NoMemory", "noise-repeat VGR"):
                    reserve_ablation_auxiliary(state, x)
                    reconcile_ablation_auxiliary(ds, state, x)
            mark_ablation_dispatch(ds, state, arms, rnd["configs"], r)
        if state.get("ablation_arm"):
            _run_or_stop(ds, arms, f"round {r} stage {n}", allow_restart=False)
        else:
            _run_or_stop(ds, arms, f"round {r} stage {n}")
        rows = opt_rows(ds, exp, state, r)
        if state.get("ablation_arm"):
            for j, c in todo:
                reconcile_ablation_candidate(state, c["config"], orders[j][:n], rows)
            for x in arms:
                if x.name in ("NoMemory", "noise-repeat VGR"):
                    reconcile_ablation_auxiliary(ds, state, x)
        for j, c in todo:
            staging_of(c).update(
                stage_reached=n,
                episodes=max(
                    n, spent(rows, space.key(c["config"]), c["sample"]["ids"])
                ),
            )
        dec = [(j, c) for j, c in todo if n < cfg_sizes(state, c)[-1]]
        if dec:
            decide_stage(ds, pool, state, r, dec, orders, n, rows, nm_dir, a=a)
        if state.get("ablation_arm"):
            from trimem.experiments.scienceworld_trace_ablation import (
                record_stage_event,
            )

            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            for _, c in todo:
                info = staging_of(c)
                record_stage_event(
                    info, stage=n,
                    status="pruned" if info["pruned_at"] == n else "complete",
                    wall_s=time.time(), monotonic_ns=time.monotonic_ns(), boot_id=boot_id,
                )
        save(state_p, state)
    log(
        f"round {r} done: "
        + ", ".join(
            f"{round_label(r, j, c)} {'pruned@' + str(staging_of(c)['pruned_at']) if staging_of(c)['pruned_at'] else 'full'}"
            for j, c in enumerate(rnd["configs"])
        )
    )


def decide_stage(
    ds: str,
    pool: dict,
    state: dict,
    r: int,
    todo: list,
    orders: list[list[str]],
    n: int,
    rows: list[dict],
    nm_dir: Path,
    mode: str = "sync",
    a=None,
) -> None:
    """Prune decisions at checkpoint n (``staged_pruning`` v2) + the plan guard; recorded per configuration
    (``mode`` async = staged-async-v1: one configuration, the comparators existing at that moment, logged).
    tc-complementary-v1 (``search_design.tc_prune``): a dls pick of an active task-complementary scorer is decided
    by its refit complementary gain instead (``task_complementary_driver.prune_decision``)."""
    front = prune_front(ds, pool, state, rows, nm_dir)
    stage = stage_comparators(
        ds,
        pool,
        state,
        rows,
        nm_dir,
        n,
        set(front),
        max_round=eb.view_max_round(
            state, r
        ),  # max-round-v1: rows are in the same view
    )
    policy = pp_policy.selected(state["staging"])
    if policy != "staged":
        stage = {}
    if policy == "none":
        front = {}
    now = {space.key(c["config"]) for _, c in todo}
    est = {
        k: obj_of(e)
        for k, e in estimates_on(
            rows,
            {space.key(c["config"]): orders[j][:n] for j, c in todo},
            pool,
            state["experiment"],
        ).items()
    }
    if now - set(est):
        raise SystemExit(
            f"round {r} stage {n}: {sorted(now - set(est))} have no complete estimate"
        )
    decs = {
        k: pp_policy.decide(
            est[k],
            front,
            policy=policy,
            stage=n,
            last=False,
            partial={j: o for j, o in stage.items() if j != k},
        )
        for k in now
    }
    plan_of = {
        space.key(c["config"]): c["config"]["plan"]
        for rr in state["rounds"]
        for c in rr["configs"]
    }
    covered = {
        plan_of[space.key(c["config"])]
        for rr in state["rounds"]
        for c in rr["configs"]
        if space.key(c["config"]) not in now and staging_of(c)["pruned_at"] is None
    }
    decs = pp_policy.guard(decs, est, plan_of, covered, policy=policy)
    tc_info = {}
    if a is not None and state.get("search_design", {}).get("tc_prune"):
        import sys
        from dataclasses import replace as _replace

        from trimem.optimizer import task_complementary_driver as tcd

        for j, c in todo:
            if not tcd.is_tc_pick(c):
                continue
            key = space.key(c["config"])
            info = tcd.prune_decision(
                sys.modules[__name__], ds, pool, state, r, a, c, orders[j][:n], rows, n
            )
            if info is None:
                continue
            tc_info[key] = info
            decs[key] = _replace(
                decs[key],
                prune=info["prune"],
                dominated_by=None,
                reason=info["reason"],
                dominators=(),
            )
    for j, c in todo:
        key = space.key(c["config"])
        d = decs[key]
        staging_of(c)["decisions"].append(
            {
                **d.as_dict(),
                **({"tc_prune": tc_info[key]} if key in tc_info else {}),
                "pruning_policy": state["search_design"]["tc_prune"]
                if key in tc_info
                else policy,
                "front_size": len(front),
                "n_stage_comparators": len(stage) - (key in stage),
                "mode": mode,
                "comparators": {
                    "full": sorted(_short(dk, plan_of) for dk in front),
                    "stage": sorted(_short(dk, plan_of) for dk in stage if dk != key),
                },
                "decided_at": datetime.now(UTC).isoformat(),
            }
        )
        log(
            f"round {r} [{mode}] comparators of {round_label(r, j, c)} at {n}: {len(front)} full + "
            f"{len(stage) - (key in stage)} stage ("
            + ", ".join(_short(dk, plan_of) for dk in stage if dk != key)
            + ")"
        )
        doms = "; ".join(
            f"{'NoMemory' if dk not in plan_of else plan_of[dk] + ' ' + cfg_name(space.validate(json.loads(dk)))}"
            f" [{kind}] {tuple(round(x, 3) for x in dc)}"
            for dk, kind, dc in d.dominators
        )
        if d.prune:
            staging_of(c).update(
                pruned_at=n, reason=d.reason, dominated_by=d.dominated_by
            )
            log(
                f"round {r}: PRUNED {round_label(r, j, c)} at {n}: corner "
                f"{tuple(round(x, 3) for x in d.corner)} dominated by {doms}"
            )
        else:
            log(
                f"round {r}: KEEP {round_label(r, j, c)} at {n}: {d.reason}"
                + (f" (dominators: {doms})" if d.dominators else "")
            )


def _short(k: str, plan_of: dict[str, str]) -> str:
    if k not in plan_of:
        return "NoMemory"
    return f"{space.DISPLAY[plan_of[k]]} {cfg_name(space.validate(json.loads(k)))}"


# ------------------------------------------------------------------ asynchronous staging (user 2026-10-20)

# staged-async-v1 (as ``tau2.pool_profile``): each live configuration runs its own stage prefixes (50 -> 100 -> 200
# of its own order) independently; as soon as a stage is complete it is decided (``decide_stage`` mode async: the
# same rule against whatever comparators exist then, logged) and a kept configuration starts its next stage at once.
# NoMemory runs the largest prefix any live configuration needs; the noise repeat (round 0) runs in full once a
# configuration reaches the last stage. ~CONCURRENCY[ds] episodes in total over the running arms (2 halves each);
# an arm whose processes per half would at least double is drained (in-flight episodes finish) and relaunched.
# Opt-in via ``ASYNC_ENV``; recorded once in state.json ``staging.async`` (ignored by the resume comparison).
ASYNC_KEY = "async"
ASYNC_PROTOCOL = "staged-async-v1"
ASYNC_ENV = "POOL_ASYNC_STAGING"
ASYNC_POLL_S = 20
ASYNC_REBALANCE = True


def async_block(state: dict) -> dict | None:
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
    sizes = g["sizes"]
    frm = next(
        (
            rr["round"]
            for rr in state["rounds"]
            if any(
                staging_of(c)["pruned_at"] is None
                and staging_of(c)["stage_reached"] < sizes[-1]
                for c in rr["configs"]
            )
        ),
        len(state["rounds"]),
    )
    g[ASYNC_KEY] = blk = {
        "protocol": ASYNC_PROTOCOL,
        "from_round": frm,
        "at": datetime.now(UTC).isoformat(),
        "reason": "user 2026-10-20: asynchronous staged pruning (as the τ² pool run)",
        "rule": "unchanged (staged_pruning v2, z = 1.96, plan guard); each configuration decided as soon as its "
        "own stage prefix completes, against the comparators existing at that moment (decisions[].comparators)",
        "note": "budget accounting unchanged; rounds never overlap; decisions made before stay as they are",
    }
    if pp_policy.selected(g) != "staged":
        blk["rule"] = (
            g["pruning_policy"]["description"] + "; decide at each completed prefix"
        )
    return blk


def _launch(ds: str, a: Arm, per_half: int | tuple[int, int]) -> list:
    halves = (per_half, per_half) if isinstance(per_half, int) else per_half
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        mp.ENV_VAR: mp.current(),
        DRAIN_ENV: str(drain_path(a)),
    }
    drain_path(a).unlink(missing_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    sa, sb = drv.shards(ds, a.ids, a.out.parent / "_shards" / a.out.name)
    procs = []
    logs = []
    try:
        for half, shard in enumerate((sa, sb)):
            if not json.loads(shard.read_text()):
                continue
            cmd = (a.cmd or drv.half_cmd)(ds, half, a.out, shard, session=a.session)
            cmd = [
                "taskset",
                "-c",
                "16-31",
                "nice",
                "-n",
                "10",
                *eb.with_max_round(_with_processes(cmd, halves[half]), a.max_round),
            ]
            logf = open(a.out / f"half_{'ab'[half]}.log", "a")  # noqa: SIM115
            logs.append(logf)
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append((proc, logf))
        return procs
    except BaseException:
        # A failed second spawn must not leave the first half evaluating.
        for proc, _ in procs:
            drv.kill_group(proc)
            proc.wait()
        for logf in logs:
            logf.close()
        raise


def _launch_registered(ds, arm, halves, entry):
    """Defer parent stop handlers until children are owned, without masking children."""
    stops = {signal.SIGTERM, signal.SIGINT, signal.SIGALRM}
    previous = {sig: signal.getsignal(sig) for sig in stops}
    pending = []
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, stops)
    try:
        for sig in stops:
            signal.signal(sig, lambda signum, frame: pending.append(signum))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)
    try:
        entry["procs"] = _launch(ds, arm, halves)
    finally:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, stops)
        try:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, mask)
        if pending:
            signal.raise_signal(pending[0])


def env_servers_healthy(ds: str) -> bool:
    """Every env server of ``ds`` answers /managed_stats."""
    for port in drv.DS[ds]["env"]:
        try:
            drv.get(f"http://127.0.0.1:{port}/managed_stats")
        except Exception as exc:  # noqa: BLE001 - any failure = unhealthy
            log(f"WARNING env {port} stats unavailable: {exc!r}")
            return False
    return True


def env_check_async(ds: str, others_running: int, *, allow_restart: bool = True) -> None:
    """After an arm exited incomplete in the async loop. Restarting the env servers while other arms run drops
    their live envs, and the official runner then retries the in-flight episodes forever (KeyError 'observation'
    loop; seen 2026-10-21 05:43, ~90 workers livelocked). So: restart only when no other arm is running or a
    server is down (then the running arms are broken anyway); otherwise defer (log only)."""
    if not allow_restart:
        if not env_servers_healthy(ds):
            raise SystemExit("environment unhealthy; ablation paused without restarting shared services")
        log("  ablation retry: shared environment left unchanged")
    elif others_running == 0 or not env_servers_healthy(ds):
        drv.env_check_and_restart(ds)
    else:
        log(
            f"  env restart deferred: {others_running} arms still running and env servers healthy"
        )


def run_staged_round_async(
    ds: str, pool: dict, state: dict, state_p: Path, r: int, a
) -> None:
    """staged-async-v1 event loop of round ``r`` (see ``ASYNC_PROTOCOL``); resumable from state.json + results."""
    base, exp = run_dir(ds, a.sampling), state["experiment"]
    nm_dir = nomemory_dir(ds, a.sampling)
    strata, _, _ = pool_meta(pool)
    rnd = state["rounds"][r]
    orders = [config_order(state, c, strata) for c in rnd["configs"]]
    nm_order = nomemory_order(state, strata)
    sizes = state["staging"]["sizes"]
    total = CONCURRENCY[ds]
    save(state_p, state)

    def next_n(c):
        return next(
            (n for n in cfg_sizes(state, c) if n > staging_of(c)["stage_reached"]), None
        )

    def config_arm(j, c, n):
        mr = eb.max_round_for(state, r)  # max-round-v1: None (30) before its from_round
        extra = {
            "staging": state["staging"]["protocol"],
            "stage_n": n,
            **eb.protocol_extra(mr),
        }
        return Arm(
            round_label(r, j, c),
            base / "configs" / cfg_name(c["config"]),
            orders[j][:n],
            session_args(
                ds,
                c["config"],
                c["sample"],
                pool,
                experiment=exp,
                baseline_dir=nm_dir,
                label=round_label(r, j, c),
                sampling=a.sampling,
                reuse=a.reuse,
                extra=extra,
                reuse_from=reuse_sources(state, r) if a.reuse else ([], []),
            ),
            max_round=mr,
        )

    def complete(x: Arm) -> bool:
        return len(x.completed(ds)) >= len(x.ids)

    procs: dict[str, dict] = {}
    fails: dict[str, int] = {}
    last_queued = 0
    last_quota_check = float("-inf")
    quota_pause = None
    resource_monitor = None
    if state.get("ablation_arm"):
        from trimem.experiments.scienceworld_trace_ablation import RuntimeResourceMonitor

        resource_monitor = RuntimeResourceMonitor(state["ablation_system_limits"])
    log(f"round {r} [async {ASYNC_PROTOCOL}]: ~{total} episodes in total")
    try:
        while True:
            if state.get("ablation_arm") and quota_pause is None and time.monotonic() - last_quota_check >= 30:
                from trimem.experiments.scienceworld_trace_ablation import check_global_quota

                try:
                    check_global_quota()
                    resource_monitor.check()
                except SystemExit as exc:
                    quota_pause = exc
                    for pr in procs.values():
                        drain_path(pr["arm"]).touch()
                        pr["drained"] = True
                    log(f"ablation quota or resource pause: draining this run's workers: {exc}")
                last_quota_check = time.monotonic()
            for name in list(procs):  # reap
                pr = procs[name]
                if any(p.poll() is None for p, _ in pr["procs"]):
                    continue
                for _, f in pr["procs"]:
                    f.close()
                drain_path(pr["arm"]).unlink(missing_ok=True)
                del procs[name]
                java_guard()
                if state.get("ablation_arm") and pr["arm"].name in ("NoMemory", "noise-repeat VGR"):
                    reconcile_ablation_auxiliary(ds, state, pr["arm"])
                if not complete(pr["arm"]) and not pr["drained"]:
                    fails[name] = fails.get(name, 0) + 1
                    log(f"  {name}: exited incomplete (failure {fails[name]})")
                    if state.get("ablation_arm"):
                        env_check_async(ds, len(procs), allow_restart=False)
                        raise SystemExit("ablation arm exited incomplete; resume after retry-budget checks")
                    else:
                        env_check_async(ds, len(procs))
                    if fails[name] >= drv.ATTEMPTS:
                        raise SystemExit(
                            f"round {r} [async]: {name} incomplete after {drv.ATTEMPTS} attempts; resume to retry"
                        )
            if quota_pause is not None:
                if not procs:
                    raise quota_pause
                time.sleep(ASYNC_POLL_S)
                continue
            changed, rows = True, None
            while (
                changed
            ):  # record + decide every configuration whose current stage is complete
                changed = False
                live_max = 0
                for j, c in enumerate(rnd["configs"]):
                    n = next_n(c) if staging_of(c)["pruned_at"] is None else None
                    if n is None:
                        continue
                    live_max = max(live_max, n)
                    x = config_arm(j, c, n)
                    if x.name in procs or not complete(x):
                        continue
                    nm = Arm("NoMemory", nm_dir, nm_order[:n], None)
                    if n != cfg_sizes(state, c)[-1] and not complete(nm):
                        continue
                    rows = opt_rows(ds, exp, state, r)
                    if state.get("ablation_arm"):
                        reconcile_ablation_candidate(state, c["config"], orders[j][:n], rows)
                    before = copy.deepcopy(staging_of(c))
                    try:
                        staging_of(c).update(
                            stage_reached=n,
                            episodes=max(
                                n,
                                spent(rows, space.key(c["config"]), c["sample"]["ids"]),
                            ),
                        )
                        if n == cfg_sizes(state, c)[-1]:
                            log(f"round {r} [async]: COMPLETE {x.name} at {n}")
                        else:
                            decide_stage(
                                ds,
                                pool,
                                state,
                                r,
                                [(j, c)],
                                orders,
                                n,
                                rows,
                                nm_dir,
                                mode="async",
                                a=a,
                            )
                        if state.get("ablation_arm"):
                            from trimem.experiments.scienceworld_trace_ablation import (
                                record_stage_event,
                            )

                            info = staging_of(c)
                            record_stage_event(
                                info, stage=n,
                                status="pruned" if info["pruned_at"] == n else "complete",
                                wall_s=time.time(), monotonic_ns=time.monotonic_ns(),
                                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                            )
                    except BaseException:
                        # Outer finally saves state: never persist an advanced prefix
                        # without its complete pruning decision.
                        c["staging"] = before
                        raise
                    save(state_p, state)
                    changed = True
            want = []
            for j, c in enumerate(rnd["configs"]):
                n = next_n(c) if staging_of(c)["pruned_at"] is None else None
                if n is None:
                    continue
                x = config_arm(j, c, n)
                if x.name not in procs and not complete(x):
                    want.append(x)
            if live_max:
                nm = Arm("NoMemory", nm_dir, nm_order[:live_max], None)
                if "NoMemory" not in procs and not complete(nm):
                    want.insert(0, nm)
            if r == 0 and (live_max == sizes[-1] or not live_max):
                nr = noise_repeat_arm(ds, pool, state, a)
                if nr.name not in procs and not complete(nr):
                    want.append(nr)
            # round0-design-v1 (WebShop, 2026-10-23 fix): round-0 configurations stop at their 100-task samples, so
            # NoMemory only reached its 100-prefix; it still runs its FULL 200-task sample (as ALF / Sci, whose
            # round-0 configurations ran 200) once every round-0 configuration is finished
            if (
                r == 0
                and not live_max
                and round0_design_from_round0(state)
                and state["nomemory"].get("sample_used")
                is None  # nomemory-prefix-v1: the prefix is the baseline
            ):
                nmf = Arm("NoMemory", nm_dir, nm_order, None)
                if "NoMemory" not in procs and not complete(nmf):
                    want.insert(0, nmf)
            n_arms = len(procs) + len(want)
            if n_arms == 0:
                break
            per_half = max(1, total // (2 * n_arms))
            # process-split-v2: the new arms share what the running ones leave of ``total`` (remainder spread)
            load = tuple(sum(pr["halves"][h] for pr in procs.values()) for h in (0, 1))
            # never above ``total`` in flight: arms that do not fit (>= 1 process per half) queue (launch_halves)
            halves = launch_halves(total, load, len(want), per_half)
            if (queued := len(want) - len(halves)) and queued != last_queued:
                log(
                    f"round {r} [async]: {queued} arms queued ({sum(load)}/{total} in flight)"
                )
            last_queued = queued
            for x, hv in zip(want, halves, strict=False):
                check_games(ds, x.ids)
                if state.get("ablation_arm"):
                    matches = [c for j, c in enumerate(rnd["configs"]) if round_label(r, j, c) == x.name]
                    if len(matches) == 1:
                        reserve_ablation_candidate(state, matches[0]["config"], x.ids)
                        reconcile_ablation_candidate(state, matches[0]["config"], x.ids, opt_rows(ds, exp, state, r))
                    elif not matches and x.name in ("NoMemory", "noise-repeat VGR"):
                        reserve_ablation_auxiliary(state, x)
                        reconcile_ablation_auxiliary(ds, state, x)
                    elif matches or x.name not in ("NoMemory", "noise-repeat VGR"):
                        raise ValueError("unrecognized ablation dispatch arm")
                    mark_ablation_dispatch(ds, state, [x], rnd["configs"], r)
                procs[x.name] = {
                    "arm": x,
                    "procs": [],
                    "per_half": per_half,
                    "halves": hv,
                    "drained": False,
                    "progress": (len(drv.item_files(x.out, ds)), time.time()),
                }
                _launch_registered(ds, x, hv, procs[x.name])
                log(
                    f"round {r} [async]: start {x.name} ({len(x.ids)} queries, {len(x.completed(ds))} done) with "
                    f"{hv[0]} + {hv[1]} processes ({n_arms} arms)"
                )
            now = time.time()
            for name, pr in procs.items():
                if (
                    ASYNC_REBALANCE
                    and not state.get("ablation_arm")
                    and not pr["drained"]
                    and per_half >= 2 * pr["per_half"]
                ):
                    log(
                        f"round {r} [async] rebalance: draining {name} (2 x {pr['per_half']}; in-flight episodes "
                        f"finish) to relaunch at 2 x {per_half}"
                    )
                    drain_path(pr["arm"]).touch()
                    pr["drained"] = True
                k = len(drv.item_files(pr["arm"].out, ds))
                if k != pr["progress"][0]:
                    pr["progress"] = (k, now)
                elif now - pr["progress"][1] > drv.STALL_S:
                    log(
                        f"stall: {name} no new item for {drv.STALL_S}s; stopping that arm"
                    )
                    for p, _ in pr["procs"]:
                        drv.kill_group(p)
            time.sleep(ASYNC_POLL_S)
    finally:
        for pr in procs.values():
            for p, f in pr["procs"]:
                drv.kill_group(p)
                p.wait()
                f.close()
            drain_path(pr["arm"]).unlink(missing_ok=True)
        save(state_p, state)
    log_reuse(exp, "train", f"round {r} [async]")
    log(
        f"round {r} done: "
        + ", ".join(
            f"{round_label(r, j, c)} {'pruned@' + str(staging_of(c)['pruned_at']) if staging_of(c)['pruned_at'] else 'full'}"
            for j, c in enumerate(rnd["configs"])
        )
    )


def pause_after_round(a) -> int | None:
    v = getattr(a, "pause_after_round", None)
    if v is None and os.environ.get(PAUSE_ENV, "").strip():
        v = int(os.environ[PAUSE_ENV])
    return v


def resume_from_pause(base: Path, state: dict, pause: int | None) -> dict | None:
    """profiling-pause-v1: when profiling continues past a user pause (no pause flag, or a later round), archive the
    paused marker and its derived artifacts; record it in state.json ``profiling_pause.history``. Only the marker
    (profile_complete.json) is moved (so the later stages wait for the new profile); the derived artifacts are
    COPIED (originals stay in place, never deleted -- user 2026-10-22). The marker moves last, so an interrupted
    archive is redone on the next start. None when there is nothing to resume."""
    p = base / PAUSE_MARKER
    if not p.exists():
        return None
    done = json.loads(p.read_text())
    if not done.get("paused_by_user") or (
        pause is not None and pause <= done["last_round"]
    ):
        return None
    at = datetime.now(UTC).isoformat()
    dest = base / f"paused_after_round{done['last_round']}_{at[:19].replace(':', '')}"
    moved, copied = [], []
    for rel in PAUSED_ARTIFACTS:
        src = base / rel
        if rel == PAUSE_MARKER or not src.exists():
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest / rel, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest / rel)
        copied.append(rel)
    dest.mkdir(parents=True, exist_ok=True)
    p.rename(dest / PAUSE_MARKER)
    moved.append(PAUSE_MARKER)
    blk = state.setdefault("profiling_pause", {"protocol": PAUSE_PROTOCOL})
    blk.setdefault("history", []).append(
        {
            "paused_after_round": done["last_round"],
            "resumed_at": at,
            "archive": str(dest),
            "moved": moved,
            "copied_left_in_place": copied,
        }
    )
    blk.update(status="resumed", resumed_at=at, archive=str(dest))
    return blk


def pause_record(state: dict, pause: int, reason: str, summary: dict) -> dict:
    """profile_complete.json fields of a user pause after round ``pause``; also sets state.json
    ``profiling_pause`` (status paused). ``summary``: configs / full_configs / pruned_configs / budget_used /
    budget_episodes. Pure except for the state update."""
    at = datetime.now(UTC).isoformat()
    rounds = [rr["round"] for rr in state["rounds"]]
    resume = (
        "rerun `bo_pool_driver profile` (or `all`) without --pause_after_round / "
        f"{PAUSE_ENV}: this marker and its derived artifacts are archived and BO continues at round {pause + 1}"
    )
    blk = state.setdefault("profiling_pause", {"protocol": PAUSE_PROTOCOL})
    blk.update(
        status="paused",
        paused_at=at,
        paused_after_round=pause,
        reason=reason,
        rounds=rounds,
        evaluated_configs=summary["configs"],
        **summary,
        resume=resume,
    )
    return {
        "paused_by_user": True,
        "pause_protocol": PAUSE_PROTOCOL,
        "pause_reason": reason,
        "paused_at": at,
        "last_round": pause,
        "rounds": rounds,
        "evaluated_configs": summary["configs"],
        "budget_used": summary["budget_used"],
        "resume": resume,
    }


def ensure_ablation_execution_ready(a):
    if not getattr(a, "ablation_arm", None):
        return
    if os.environ.get("TRIMEM_ABLATION_EXECUTION") != "1":
        raise SystemExit("ScienceWorld ablation integration is not complete; execution is disabled without an explicit guarded launch")
    from tools.scienceworld_ablation_runtime import verify_current_cgroup
    from trimem.experiments import scienceworld_trace_ablation as ab

    limits_path = os.environ.get("TRIMEM_ABLATION_SYSTEM_LIMITS")
    if not limits_path:
        raise SystemExit("ablation execution is disabled without frozen system limits")
    limits = json.loads(Path(limits_path).read_text())
    ab.validate_system_limits(limits)
    ab.validate_isolated_memory_database(os.environ.get("TRIMEM_SCIWORLD_MEMORY_DB"), os.environ)
    verify_current_cgroup(limits)
    ab.check_global_quota()


def profile(ds: str, a) -> None:
    ensure_ablation_execution_ready(a)
    if not getattr(a, "ablation_arm", None):
        return _profile_impl(ds, a)
    from trimem.experiments.scienceworld_trace_ablation import SearchClock

    base = run_dir(ds, a.sampling)
    state = json.loads((base / "state.json").read_text())
    clock = SearchClock(base / "search_clock.json", state["ablation_manifest_sha256"])
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    clock.start(wall_s=time.time(), monotonic_ns=time.monotonic_ns(), boot_id=boot_id)
    try:
        return _profile_impl(ds, a)
    finally:
        clock.end(wall_s=time.time(), monotonic_ns=time.monotonic_ns(), boot_id=boot_id)


def _profile_impl(ds: str, a) -> None:
    """Round 0 (the 10 round-0 configurations, staged like every round; NoMemory and the noise repeat alongside,
    never pruned), then staged BO rounds while the unpruned plan's episode budget allows (``staged_pruning.next_batch_size``: a batch starts only
    with as many configurations as can each still be evaluated in full; at most ``max_configs``)."""
    pool = load_pool(ds)
    state, state_p = load_state(ds, pool, a)
    baseline = sa.baselines.active(state["search_design"])
    # quality-first slot (user 2026-10-20): recorded once; BO rounds >= from_round reserve 1 slot
    if not baseline:
        bo_module().qfirst_block(state, datetime.now(UTC).isoformat())
    # GP protocol gp-fix-v1 (2026-10-20): recorded once, from the next round to be proposed
    bo_module().gp_block(state, datetime.now(UTC).isoformat(), GP_NOISE)
    # channel slots channel-slots-v1 (user 2026-10-20): recorded once, from max(1, the next round to be proposed; user 2026-10-21, was 3)
    # channel-slots-v3 (user 2026-10-22): opt-in POOL_CHANNEL_SLOTS=v3, from the next round to be proposed
    # channel-slots-v4 (user 2026-10-22): opt-in POOL_CHANNEL_SLOTS=v4 (v3 -> v4 upgrade from the next round)
    if not baseline:
        cs = channel_slots_record(state, a, datetime.now(UTC).isoformat())
        assert cs["protocol"] in bo_module().CHANNEL_SLOTS_PROTOCOLS
        log(
            f"channel slots {cs['protocol']} from round {cs['from_round']} (state.json channel_slots)"
        )
    # pool provenance (user 2026-10-23, WebShop option B): recorded once when the pool file states it, so the
    # train / test product overlap travels with the run (pools without the key -- ALF / Sci -- record nothing)
    if "train_test_share_products" in pool and "pool_provenance" not in state:
        state["pool_provenance"] = {
            k: pool.get(k)
            for k in (
                "scope",
                "size",
                "pool_hash",
                "train_test_share_products",
                "goals_on_test_products",
                "products_shared_with_test",
                "test_goal_overlap",
                "product_rule",
                "built_at",
            )
        }
    # round0-design-v1 (user 2026-10-23, WebShop): opt-in POOL_ROUND0_SOBOL / POOL_FROM_ROUND0 (fresh runs only)
    if r0d := round0_design_block(state, datetime.now(UTC).isoformat()):
        log(
            f"round-0 design {r0d['protocol']}: sampler {r0d['sampler']}, sample size / P1 from round 0 "
            f"{r0d['from_round0']} (state.json {ROUND0_DESIGN_KEY})"
        )
    # sample-size-v1 (user 2026-10-22): opt-in POOL_SAMPLE_SIZE=100 + POOL_SAMPLE_UNIFORM=1
    if ss := sample_size_block(state, a, datetime.now(UTC).isoformat()):
        log(
            f"sample size {ss['protocol']}: {ss['size']} {ss['sampling']} tasks per configuration from round "
            f"{ss['from_round']} (stages {ss['stage_sizes']}; state.json sample_schedule)"
        )
    # round0-prefix-v1 (user 2026-10-22, ScienceWorld): round-0 samples = their stage-100 prefix (recorded by
    # pool_round0_prefix; POOL_ROUND0_PREFIX must match)
    if r0p := round0_prefix_check(state):
        log(
            f"round-0 samples {r0p['protocol']}: the stage-{r0p['size']} prefix of each 200-task stratified sample "
            f"(state.json round0_prefix; NoMemory keeps 200)"
        )
    # plan-p1-v1 (user 2026-10-22): opt-in POOL_PLAN_P1=1 (or --plan_p1), gated plan P1 from the next round
    if p1 := bo_module().plan_p1_block(
        state, datetime.now(UTC).isoformat(), plan_p1_want(a)
    ):
        if not state["rounds"] and round0_design_from_round0(state):
            p1["from_round"] = (
                0  # round0-design-v1: P1 proposable from round 0 (fresh run)
            )
        log(
            f"plan extension {p1['protocol']}: {p1['added_plans']} proposable from round {p1['from_round']} "
            f"(state.json {bo_module().PLAN_P1_KEY})"
        )
    # budget-extension-v1 (user 2026-10-22): opt-in POOL_BUDGET_EPISODES / POOL_BUDGET_MAX_CONFIGS (+ reason),
    # recorded from the next round to be proposed; the staging block (and its resume check) stays as recorded
    if be := budget_extension_block(state, a, datetime.now(UTC).isoformat()):
        h = be["history"][-1]
        log(
            f"budget {be['protocol']}: {h['budget_episodes']} episodes / max {h['max_configs']} configurations from "
            f"round {h['from_round']} (before {h['before']}; {h['reason']})"
        )
    if bs := batch_schedule_block(state, a, datetime.now(UTC).isoformat()):
        log(
            f"round batch {_recorded_steps(bs)} (size@from_round; state.json batch_schedule; before "
            f"{bs['before']}; budget unchanged)"
        )
    # auxiliary GP observations aux-obs-v1 (user 2026-10-20): recorded once, from max(1, the next round)
    ax = (
        None
        if baseline
        else bo_module().aux_block(
            state, datetime.now(UTC).isoformat(), aux_obs.build(ds, space), min_round=1
        )
    )
    if ax:
        log(
            f"auxiliary GP observations {ax['protocol']} from round {ax['from_round']}: {len(ax['points'])} points "
            f"from {[s['experiment'] for s in ax['sources']]}"
        )
    # HV reference point hv-ref-anchored-v1 (user 2026-10-22): opt-in POOL_HV_REF=anchored-v1, recorded once,
    # from the next round to be proposed (safe on a mid-run resume; earlier rounds stay as proposed)
    if hv := bo_module().hv_ref_block(state, datetime.now(UTC).isoformat()):
        log(
            f"HV reference {hv['protocol']} from round {hv['from_round']} (state.json hv_ref; before: "
            f"{hv['before']})"
        )
    if blk := reuse_block(state, a):
        log(
            f"quality reuse {blk['protocol']} from round {blk['from_round']} (state.json quality_reuse)"
        )
    if blk := async_block(state):
        log(
            f"staging {blk['protocol']} from round {blk['from_round']} (state.json staging.async)"
        )
    # latency-regime-v1 (user 2026-10-22): a recorded mid-run runtime change (c128 -> controlled-c16-v1 for ALF v7)
    if sfb := semfilter_block(
        ds, state, run_dir(ds, a.sampling), datetime.now(UTC).isoformat()
    ):
        log(
            f"semfilter {sfb['protocol']} ({sfb['regime_protocol']}) from round {sfb['from_round']}: top_n "
            f"{sfb['top_n']}, batch {sfb['batch']}, rubric {sfb['rubric']}, guided {sfb['guided']}, judge input "
            f"{sfb['judge_input']}; env exported to every runner"
        )
    if v2b := semfilter_v2b_block(
        ds, state, run_dir(ds, a.sampling), datetime.now(UTC).isoformat()
    ):
        # the v2 env was exported (and the caller's env checked against it) just above; switch to v2b's
        os.environ.update(effective_semfilter_env(state["semfilter"]))
        log(
            f"semfilter {v2b['protocol']} from round {v2b['from_round']}: rubric {v2b['rubric']}, sem_fallback "
            f"{v2b['sem_fallback']}, proposable thresholds {v2b['thresholds']}; env exported to every runner"
        )
    if nmp := nomemory_prefix_block(
        ds, state, pool, nomemory_dir(ds, a.sampling), datetime.now(UTC).isoformat()
    ):
        log(
            f"NoMemory {nmp['protocol']}: baseline = its stage-{nmp['size']} prefix ({len(nmp['ids'])} tasks); "
            "remaining NoMemory tasks not run (state.json nomemory.sample_used)"
        )
    if lr := latency_regime_block(ds, state, a, datetime.now(UTC).isoformat()):
        log(
            f"latency regime {lr['protocol']} from round {lr['from_round']}: {lr['old']['concurrency']} -> "
            f"{lr['new']}; NoMemory {lr['nomemory']['dir']}; reuse since {lr['at']}"
        )
    blk = concurrency_block(ds, state)
    log(
        f"concurrency {blk['protocol']}: profiling {blk['profiling_total']}, test {blk['test_total']}, SemFilter "
        f"routing {blk['semantic_routing']} (state.json concurrency, from round {blk['from_round']})"
    )
    if rp := runtime_profile_block(ds, state):
        log(
            f"runtime profile {rp['name']} (state.json runtime_profile): {rp['layout']}"
        )
    # max-round-v1 (user 2026-10-23, WebShop): opt-in POOL_EPISODE_MAX_ROUND, recorded once from the next round
    # to be proposed; earlier rounds (one still running included) finish at 30 and are compared derived-truncated
    if ebk := eb.record_block(state, ds, datetime.now(UTC).isoformat()):
        log(
            f"episode budget {ebk['protocol']}: max_round {ebk['max_round']} from round {ebk['from_round']} "
            f"(runtime {eb.runtime_label(state)}); rounds < {ebk['from_round']} / NoMemory / noise repeat compared "
            f"{eb.tag(ebk['max_round'])} from round {ebk['from_round']} on ({ebk['reason']})"
        )
    base, exp = run_dir(ds, a.sampling), state["experiment"]
    base.mkdir(parents=True, exist_ok=True)
    nm_dir = nomemory_dir(ds, a.sampling)
    rep = space.validate({"plan": "VGR", **space.plan_defaults()})
    rb = router_backends(state, a, datetime.now(UTC).isoformat())
    log(
        f"router backends: {state.get('router_backends', {}).get('backends', 'BRANE + LimeQO (unrecorded)')} (limeqo {rb})"
    )
    if "nomemory" not in state:
        state["nomemory"] = {
            "sample": write_sample(
                ds,
                pool,
                base,
                "NoMemory",
                sample_label("NoMemory", None, a.sampling),
                nomemory_sample_experiment(ds, a.sampling),
                size=state.get("full_panel_size", N_SAMPLE),
            )
        }
    if "noise_repeat" not in state:
        state["noise_repeat"] = {
            "config": rep,
            "sample": write_sample(
                ds,
                pool,
                base,
                "noise_repeat_VGR",
                sample_label("noise", rep, a.sampling),
                sampling_namespace(state),
                size=state.get("full_panel_size", N_SAMPLE),
            ),
        }
    save(state_p, state)
    budget = effective_budget(state)
    pause = pause_after_round(a)
    if blk := resume_from_pause(base, state, pause):
        save(state_p, state)
        log(
            f"profiling RESUMED after the user pause ({PAUSE_PROTOCOL}); paused artifacts archived in {blk['archive']}"
        )
    run_regime_nomemory(ds, pool, state, a)
    nm, nm_mr = None, None
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
            n_cfg = sum(len(rr["configs"]) for rr in state["rounds"])
            k = (
                N_ROUND0
                if r == 0
                else sp.next_batch_size(
                    budget, episodes_used(state), n_cfg, round_batch(state, r, a.batch)
                )
            )
            if k == 0:
                break
            # max-round-v1: the NoMemory anchor in the view of the round being proposed
            if r > 0 and (nm is None or nm_mr != eb.view_max_round(state, r)):
                nm_mr = eb.view_max_round(state, r)
                nm, _ = nomemory_estimate(
                    ds, pool, nomemory_sample(state), nm_dir, max_round=nm_mr, raw_score=bool(state.get("ablation_arm"))
                )
            if (
                nm_mr is not None
            ):  # max-round-v1: audit snapshot of the view this proposal uses
                episode_budget_snapshot(ds, pool, state, base, nm_dir, r)
            picks = plan_round(ds, pool, state, r, nm, a, k)
            info = (
                {"coverage": bo_module().coverage_summary(picks)}
                if any("coverage" in p for p in picks)
                else {}
            )
            for p in picks:
                size_r, uniform_r = sample_spec(state, r)
                p["sample"] = write_sample(
                    ds,
                    pool,
                    base,
                    cfg_name(p["config"]),
                    sample_label("config", p["config"], a.sampling),
                    sampling_namespace(state),
                    size=size_r,
                    uniform=uniform_r,
                )
            state["rounds"].append(
                {
                    "round": r,
                    "proposed_at": datetime.now(UTC).isoformat(),
                    "configs": picks,
                    "budget_used_before": episodes_used(state),
                    **info,
                }
            )
            apply_sample_kinds(state)
            save(state_p, state)
            log(
                f"round {r} ({k} configurations; budget used {episodes_used(state)}/{budget.total_episodes}): "
                + "; ".join(f"[{p['channel']}] {space.key(p['config'])}" for p in picks)
            )
        run_staged_round(ds, pool, state, state_p, r, a)
        r += 1
    est = estimates(ds, pool, state, exp)
    planned = set(samples_of(state))
    fk, pruned = full_keys(state), {p["key"] for p in pruned_entries(state)}
    n_cfg = sum(len(rr["configs"]) for rr in state["rounds"])
    if len(planned) != n_cfg or fk & pruned or fk | pruned != planned or set(est) != fk:
        raise SystemExit(
            f"profiling incomplete: {len(set(est) & fk)} full + {len(pruned)} pruned of {n_cfg} planned"
        )
    used = episodes_used(state)
    paused = {}
    if pause is not None:
        paused = pause_record(
            state,
            pause,
            getattr(a, "pause_reason", None) or "user pause",
            {
                "configs": n_cfg,
                "full_configs": len(fk),
                "pruned_configs": len(pruned),
                "budget_used": used,
                "budget_episodes": budget.total_episodes,
            },
        )
        save(state_p, state)
    save(
        base / "profile_complete.json",
        {
            **paused,
            "completed_at": datetime.now(UTC).isoformat(),
            "configs": n_cfg,
            "full_configs": len(fk),
            "pruned_configs": len(pruned),
            "experiment": exp,
            "pool_hash": pool["pool_hash"],
            "space_version": space.VERSION,
            "memory_policy": state["memory_policy"],
            "staging": state["staging"]["protocol"],
            "episodes_profiling": used,
            "episodes_nomemory": len(nomemory_ids(state)),
            "episodes_noise_repeat": len(state["noise_repeat"]["sample"]["ids"]),
            "budget_episodes": budget.total_episodes,
            "budget_unused": budget.total_episodes - used,
            "per_config": [
                {
                    "key": space.key(c["config"]),
                    "round": rr["round"],
                    "position": j,
                    **{k: v for k, v in staging_of(c).items() if k != "order"},
                }
                for rr in state["rounds"]
                for j, c in enumerate(rr["configs"])
            ],
        },
    )
    log(
        ("PROFILING PAUSED " if paused else "PROFILING DONE ")
        + f"{ds}: {n_cfg} configurations ({len(fk)} full, {len(pruned)} pruned), {used} / "
        f"{budget.total_episodes} budget episodes + NoMemory {len(state['nomemory']['sample']['ids'])} "
        f"+ noise repeat {len(state['noise_repeat']['sample']['ids'])}"
    )


# ------------------------------------------------------------------ profile table (selection input)


def profile_table(ds: str, a) -> dict:
    """Per configuration: stratified mean quality / SE, unpaired gain vs NoMemory, tokens, latency (s), the
    NoMemory overlap diagnostic; plus the noise floor from the independent VGR repeat."""
    pool = load_pool(ds)
    state, _ = load_state(ds, pool, a)
    base, exp = run_dir(ds, a.sampling), state["experiment"]
    if not (base / "profile_complete.json").exists():
        raise SystemExit("profiling is not complete; no selection")
    nm_dir = nomemory_dir(ds, a.sampling)
    nm, nm_q = nomemory_estimate(
        ds, pool, nomemory_sample(state), nm_dir, max_round=eb.view_max_round(state), raw_score=bool(state.get("ablation_arm"))
    )
    fk = full_keys(
        state
    )  # pruned configurations (partial samples) never enter selection or the router
    origin = {
        space.key(c["config"]): (rr["round"], j, c.get("channel"))
        for rr in state["rounds"]
        for j, c in enumerate(rr["configs"])
    }
    cat = catalog_block(state)  # catalog-regime-v1: only rounds >= from_round
    excluded = set()
    if cat is not None:
        excluded = {k for k in fk if origin[k][0] < cat["from_round"]}
        fk = fk - excluded
    v2b = (state.get("semfilter") or {}).get(SEMFILTER_V2B_KEY)
    if (
        v2b is not None
    ):  # semfilter-v2b: old-regime SemFilter configurations never enter router / selection / test
        old_sf = {
            k
            for k in fk
            if origin[k][0] < v2b["from_round"]
            and json.loads(k)["plan"] in space.SEMFILTER_PLANS
        }
        excluded |= old_sf
        fk = fk - old_sf
    est = {k: e for k, e in estimates(ds, pool, state, exp).items() if k in fk}
    pts = []
    for k, e in est.items():
        g = us.diff(e["quality"], nm)
        r, j, ch = origin[k]
        pts.append(
            {
                "label": f"r{r}c{j}:{space.DISPLAY[e['cfg']['plan']]}",
                "key": k,
                "cfg": space.validate(e["cfg"]),
                "round": r,
                "channel": ch,
                "q": e["quality"].mean,
                "q_se": e["quality"].se,
                "gain": g.mean,
                "gain_se": g.se,
                "tok": e["tokens"].mean,
                "tok_se": e["tokens"].se,
                "lat": e["latency_ms"].mean / 1000,
                "lat_se": e["latency_ms"].se / 1000,
                "fixed": k
                == space.key({"plan": e["cfg"]["plan"], **space.plan_defaults()}),
                "nomemory_overlap": us.paired_overlap(e["per_query"]["quality"], nm_q),
            }
        )
    strata, w, sizes = pool_meta(pool)
    rep_rows = opt_rows(ds, exp + ":noise-repeat", state)
    bo = bo_module()

    rep_key = space.key(state["noise_repeat"]["config"])
    rep = bo.config_estimates(
        rep_rows,
        samples={rep_key: state["noise_repeat"]["sample"]["ids"]},
        strata=strata,
        weights=w,
        sizes=sizes,
        experiment=exp + ":noise-repeat",
    )
    floor = (
        us.noise_floor(est[rep_key]["quality"], rep[rep_key]["quality"])
        if rep_key in rep and rep_key in est
        else None
    )
    return {
        "pool": pool,
        "state": state,
        "points": sorted(pts, key=lambda p: (p["round"], p["label"])),
        "nomemory": {"q": nm.mean, "q_se": nm.se, "n": nm.n},
        **({"episode_budget": eb.block(state)} if eb.block(state) else {}),
        "noise_floor": floor,
        "estimates": est,
        "pruned": [
            e
            for e in pruned_entries(state)
            if cat is None or e["round"] >= cat["from_round"]
        ],
        "catalog": None
        if cat is None
        else {
            **cat,
            "full_configs": len(est),
            "excluded_full_configs": len(excluded),
            **(
                {"excluded_semfilter_old_regime": len(old_sf)}
                if v2b is not None
                else {}
            ),
        },
    }


# ------------------------------------------------------------------ router (Opt-PerQuery)


def router_raw_science_profile(profile, state: dict, sampling: str):
    """Recover original TRAIN episode scores for a ScienceWorld router.

    Historic optimizer rows store max(score, 0)/100; missing or mismatched
    underlying rewards are errors, not inferred zeros. Derived episode-budget
    views must first supply their own raw-score semantics.
    """
    import psycopg
    from psycopg.rows import dict_row
    from trimem.optimizer.query_router.joint_selected import restore_scienceworld_scores

    if eb.view_max_round(state) is not None:
        raise ValueError("raw-score router does not support derived episode-budget views")
    scores = {}
    with psycopg.connect(drv.OPT_DSN + " options='-c default_transaction_read_only=on'", row_factory=dict_row) as conn:
        for row in conn.execute(
            "SELECT query_id, protocol->'space_config' AS cfg, generation->>'reward' AS reward "
            "FROM opt.session_retrieval WHERE experiment_id=%s AND split='train' AND status='evaluated'",
            (state["experiment"],),
        ):
            if row["cfg"] is None or row["reward"] is None:
                continue
            key = (space.key(row["cfg"]), row["query_id"])
            value = float(row["reward"])
            if key in scores and scores[key] != value:
                raise ValueError(f"conflicting raw scores for {key[1]}")
            scores[key] = value
    if any(cfg.get("plan") == "NoMemory" for cfg in profile.configs):
        for q, file in drv.item_files(nomemory_dir("sciworld", sampling), "sciworld").items():
            scores[(space.key(space.NOMEM), q)] = float(json.loads(file.read_text())["reward"])
    return restore_scienceworld_scores(profile, scores)


def router_bounds(ds: str, points: list[dict], *, raw_science: bool) -> dict:
    from trimem.experiments import slo_select_bo as slo

    result = slo.bounds(ds, points)
    if raw_science:
        result["SLO-3"]["q"] *= 100
    return result


def feature_records(
    ds: str, ids: list[str], pool: dict, test: bool, cache: Path
) -> list[dict]:
    """Raw router records (structured + probe + embedding; no LLM) of pool items or test items; cached."""
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments.agentenv_task_adapter import dsn_for
    from trimem.experiments.query_router_agentenv import TASK_SQL
    from trimem.optimizer.query_router import features

    have = (
        {r["query_id"]: r for r in json.load(open(cache))["records"]}
        if cache.exists()
        else {}
    )
    todo = [i for i in ids if i not in have]
    if todo:
        key = {x["item_id"]: x["task_key"] for x in pool["items"]}
        with psycopg.connect(dsn_for(ds), row_factory=dict_row) as c:
            for n, i in enumerate(todo):
                t = (
                    c.execute(TASK_SQL + "agenteval_item = %s", (i,)).fetchone()
                    if test
                    else c.execute(
                        TASK_SQL
                        + "key = %s AND NOT held_out AND agenteval_item IS NULL",
                        (key[i],),
                    ).fetchone()
                )
                if t is None:
                    raise SystemExit(f"no task row for {i}")
                vec = t.pop("emb")
                have[i] = features.raw_record(i, t, vec, use_probe=True)
                if n % 500 == 499:
                    save(cache, {"records": list(have.values())})
        save(cache, {"records": list(have.values())})
    return [have[i] for i in ids]


def router_stage(ds: str, a) -> None:
    ensure_ablation_execution_ready(a)
    resolve_router_defaults(a)
    import numpy as np

    from trimem.experiments import slo_select_bo as slo
    from trimem.optimizer.query_router import features, runtime, sparse

    record_post_profile(
        ds, a
    )  # catalog-regime-v1 / router-backend change (opt-in, recorded once)
    T = profile_table(ds, a)
    pool, state = T["pool"], T["state"]
    out = run_dir(ds, a.sampling) / "router"
    out.mkdir(parents=True, exist_ok=True)
    strata, w, sizes = pool_meta(pool)
    est = dict(T["estimates"])
    nm_cand = nomemory_candidate(ds, a, state) if a.nomemory_candidate else None
    if nm_cand is not None:
        est[space.key(space.NOMEM)] = nm_cand
    prof = sparse.from_estimates(est, strata, w, sizes)
    raw_science = ds == "sciworld" and a.router_backend == "joint-selected"
    if raw_science:
        if router_backends(state, a):
            raise ValueError("joint-selected raw ScienceWorld routing currently requires --no_limeqo")
        if not state.get("ablation_arm"):
            prof = router_raw_science_profile(prof, state, a.sampling)
    union = prof.queries
    recs = feature_records(ds, union, pool, False, out / "features_train.json")
    from trimem.experiments.shared_router import save_snapshot

    save_snapshot(
        out / "training_snapshot.json",
        prof,
        recs,
        dataset=ds,
        source={"experiment": state["experiment"], "pool_hash": pool["pool_hash"]},
    )
    test_ids = [
        x["item_id"]
        for s in ("a", "b")
        for x in json.load(open(drv.CTRL / "shards" / f"{ds}_{s}.json"))
    ]
    if not state.get("ablation_arm"):
        feature_records(
            ds, test_ids, pool, True, out / "features_test.json"
        )  # deployment check: features exist
    groups = tuple(a.groups.split(","))
    spec = features.fit_spec(recs, groups, emb_dims=24 if a.router_backend == "joint-selected" else 4)
    t0 = time.perf_counter()
    X = {r["query_id"]: runtime.transform(spec, r) for r in recs}
    tr_ms = (time.perf_counter() - t0) * 1000 / len(recs)
    log(
        f"router: {len(prof.configs)} configurations, {len(union)} profiled queries, features {spec['names'][:4]}..."
    )
    model, oof = sparse.fit(X, prof, trials=a.trials, n_jobs=a.jobs, seed=a.seed,
                            backend=a.router_backend, quality_kind="raw_score" if raw_science else None)
    log(
        f"router: {len(model['configs'])} candidates, predictors {[s.get('family') for s in model['selection']]}"
    )
    ovh = (
        {r["query_id"]: r["overhead"]["probe_ms"] + tr_ms for r in recs}
        if "probe" in groups
        else {r["query_id"]: tr_ms for r in recs}
    )
    points = T["points"]
    if raw_science:
        by_key = {space.key(cfg): i for i, cfg in enumerate(prof.configs)}
        points = [{**pt, "q": prof.est(by_key[pt["key"]], "quality").mean,
                   "q_se": prof.est(by_key[pt["key"]], "quality").se} for pt in points]
    B = router_bounds(ds, points, raw_science=raw_science)
    one = {
        n: slo.choose(points, B[n], slo.SLOS[n]["objective"]) for n in SLO_NAMES
    }
    best_q = max(points, key=lambda p: (p["q"], -p["lat"]))
    trained_on = {
        "experiment": state["experiment"],
        "pool_hash": pool["pool_hash"],
        "queries": len(union),
        "configs_profiled": len(prof.configs),
        "memory_policy": state["memory_policy"],
        "groups": list(groups),
        **(
            {"catalog_from_round": T["catalog"]["from_round"]}
            if T.get("catalog")
            else {}
        ),
    }
    # Three-weight objective (user 2026-10-19): train OOF replay over the simplex grid (selection) and over the
    # test slices (curves only); the grid rows are the only ones the SLO choice sees.
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
                f"{sparse.weights_of(near) if near else None}, slack {sparse.slack(near, B[n]) if near else None})"
            )
            continue
        robust = sparse.choose_weights_robust(rows, B[n], obj)
        wts = sparse.weights_of(pick)
        fb = (one[n] or best_q)["cfg"]
        art = sparse.artifact_weights(
            dataset=ds,
            spec=spec,
            model=model,
            weights=wts,
            fallback=fb,
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
            else {  # reported only, never deployed
                "weights": list(sparse.weights_of(robust)),
                "replay_q": robust["q"],
                "replay_q_se": robust["quality_se"],
                "replay_lat_s": robust["lat"],
                "replay_tok": robust["tok"],
                "slack": sparse.slack(robust, B[n]),
            },
        }
        log(
            f"router {n}: weights (w_Q, w_L, w_T) {wts} -> replay q {pick['q']:.3f} ± {pick['quality_se']:.3f}, "
            f"lat {pick['lat']:.1f}s, tok {pick['tok']:,.0f} ({sha}); 1-SE variant "
            f"{arts[n]['robust_1se']['weights'] if robust else None}"
        )
    # Test slices: one frozen artifact (default weights in its routing block; every slice point is routed offline
    # from it with ``route_test_weights``).
    cart = sparse.artifact_weights(
        dataset=ds,
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
    log(f"router weight slices: {slice_router['slices']} ({csha})")
    # LimeQO (user 2026-10-21): second router backend on the SAME frozen training snapshot, same bounds / SLO rule /
    # utility; its test arms are Opt-PerQuery-LimeQO-<weights>. Off when state.json router_backends says so.
    lime = None
    if router_backends(state, a):
        from trimem.experiments.shared_router import deployed_limeqo

        lime = deployed_limeqo(
            out / "training_snapshot.json",
            out,
            groups=groups,
            bounds=B,
            objectives={n: slo.SLOS[n]["objective"] for n in SLO_NAMES},
            fallback=best_q["cfg"],
            trained_on={**trained_on, "dataset": ds},
            seed=a.seed,
        )
        log(
            f"router LimeQO: {len(lime['candidates'])} candidates; SLO weights "
            f"{ {n: v and v['weights'] for n, v in lime['routers'].items()} } ({lime['slice_router']['sha']})"
        )
    else:
        log("router LimeQO: disabled for this run (state.json router_backends)")
    save(
        out / "router_report.json",
        {
            "generated": datetime.now(UTC).isoformat(),
            "experiment": state["experiment"],
            "protocol": sparse.PROTOCOL,
            "replay_estimator": sparse.REPLAY_ESTIMATOR,
            "catalog": T.get("catalog"),
            "objective": sparse.objective_text(model["cost_norm"]),
            "cost_norm": model["cost_norm"],
            # user 2026-10-22: influence diagnostics of the frozen scale (variance shares, flip rates, corners)
            "utility_diagnostics": sparse.router_diagnostics(
                prof, model, X, oof, dict(TEST_WEIGHTS)
            ),
            "grid": {
                "step": sparse.GRID_STEP,
                "w_q_min": sparse.GRID_MIN_WQ,
                "size": len(grid),
            },
            "candidates": model["configs"],
            "selection": model["selection"],
            "nomemory_candidate": None
            if nm_cand is None
            else {
                "in_candidates": any(c["plan"] == "NoMemory" for c in model["configs"]),
                "cost_source": nm_cand["cost_source"],
                "dir": str(nomemory_dir(ds, a.sampling)),
            },
            "train_means": model["train_means"],
            "bounds": B,
            "routers": arts,
            "slice_router": slice_router,
            "backends": [a.router_backend, "LimeQO"] if lime else [a.router_backend],
            "limeqo": lime and {k: v for k, v in lime.items() if k != "weight_grid"},
            "limeqo_weight_grid": lime and lime["weight_grid"],
            "curves": curves,
            "transform_ms": tr_ms,
            "oof": {str(k): v for k, v in oof.items()},
        },
    )
    np.save(out / "union_queries.npy", np.array(union))


# ------------------------------------------------------------------ selection (frozen)


def select_stage(ds: str, a) -> dict:
    ensure_ablation_execution_ready(a)
    from trimem.experiments import slo_select_bo as slo
    from trimem.optimizer.query_router import sparse

    base = run_dir(ds, a.sampling)
    sel_p = base / "selection.json"
    if sel_p.exists():
        return json.load(open(sel_p))
    T = profile_table(ds, a)
    rr = base / "router" / "router_report.json"
    if not rr.exists():
        raise SystemExit("run the router stage before freezing the selection")
    R = json.load(open(rr))
    if R.get("protocol") != sparse.PROTOCOL:
        raise SystemExit(
            f"router report protocol {R.get('protocol')!r} != {sparse.PROTOCOL!r}; rerun the router stage"
        )
    if (R.get("catalog") or {}).get("from_round") != (T.get("catalog") or {}).get(
        "from_round"
    ):
        raise SystemExit(
            "router report catalog regime differs from state.json's; rerun the router stage"
        )
    raw_science = ds == "sciworld" and a.router_backend == "joint-selected"
    points = T["points"]
    if raw_science:
        strata, weights, sizes = pool_meta(T["pool"])
        prof = sparse.from_estimates(dict(T["estimates"]), strata, weights, sizes)
        if not T["state"].get("ablation_arm"):
            prof = router_raw_science_profile(prof, T["state"], a.sampling)
        by_key = {space.key(cfg): i for i, cfg in enumerate(prof.configs)}
        nm_ids = nomemory_sample(T["state"])["ids"]
        nm_files = drv.item_files(nomemory_dir(ds, a.sampling), ds)
        nm_values = {q: float(json.loads(nm_files[q].read_text())["reward"]) for q in nm_ids}
        nm_est = us.stratified([nm_values[q] for q in nm_ids], [strata[q] for q in nm_ids], weights, sizes)
        updated = []
        for pt in points:
            i = by_key[pt["key"]]
            q_est = prof.est(i, "quality")
            gain = us.diff(q_est, nm_est)
            updated.append({**pt, "q": q_est.mean, "q_se": q_est.se,
                            "gain": gain.mean, "gain_se": gain.se,
                            "nomemory_overlap": us.paired_overlap(prof.quality[i], nm_values)})
        points = updated
        T["nomemory"] = {"q": nm_est.mean, "q_se": nm_est.se, "n": nm_est.n}
    B = router_bounds(ds, points, raw_science=raw_science)
    if json.loads(json.dumps(B)) != R["bounds"]:
        raise SystemExit(
            "router bounds differ from the profile's; rerun the router stage"
        )
    sels = {}
    for n in SLO_NAMES:
        p = slo.choose(points, B[n], slo.SLOS[n]["objective"])
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
            | {"n_feasible": sum(slo.feasible(q, B[n]) for q in points)},
            "opt_perquery": R["routers"].get(n),
            "opt_perquery_limeqo": (R.get("limeqo") or {}).get("routers", {}).get(n),
        }
    sel = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "experiment": T["state"]["experiment"],
        "pool_hash": T["pool"]["pool_hash"],
        "memory_policy": T["state"]["memory_policy"],
        "space_version": space.VERSION,
        "n_configs": len(T["points"]),
        "bounds": B,
        "slo_text": {n: slo.SLOS[n]["text"] for n in SLO_NAMES},
        "nomemory_train": T["nomemory"],
        "noise_floor_unpaired": T["noise_floor"],
        "selections": sels,
        "staging": {
            "protocol": T["state"]["staging"]["protocol"],
            "full_configs": len(T["points"]),
            "pruned_configs": len(T["pruned"]),
            "candidates": "fully evaluated configurations only",
        },
        "router_protocol": R["protocol"],
        "replay_estimator": R.get("replay_estimator", "sum-v0"),
        "catalog": T.get("catalog"),
        "weight_slices": R[
            "slice_router"
        ],  # frozen: test slices + artifact of the Opt-PerQuery weight arms
        "weight_slices_limeqo": (R.get("limeqo") or {}).get(
            "slice_router"
        ),  # frozen LimeQO artifact (Opt-PerQuery-LimeQO-<weights> arms)
        "limeqo_transform_ms": (R.get("limeqo") or {}).get("transform_ms"),
        "points": [
            {k: v for k, v in p.items() if k != "nomemory_overlap"}
            | {"nomemory_overlap": p["nomemory_overlap"]}
            for p in points
        ],
    }
    save(sel_p, sel)
    log(f"selection frozen: {sel_p}")
    return sel


# ------------------------------------------------------------------ outer 5-fold CV (user 2026-10-21)

# CV test pairs run automatically in the test stage only up to this many NEW (config, query) pairs; above it the
# CV test arms are deferred (logged; rerun ``test --cv_test_max_pairs N``).
CV_TEST_MAX_PAIRS = 2000


def cv_routing_path(ds: str, a) -> Path:
    return run_dir(ds, a.sampling) / "test" / "routing_cv.json"


def load_cv_routing(ds: str, a) -> dict | None:
    p = cv_routing_path(ds, a)
    return json.load(open(p)) if p.exists() else None


def completed_test_pairs(
    ds: str, a, experiment: str, arms: dict[str, dict]
) -> set[tuple[str, str]]:
    """(config key, query) pairs of ``arms`` already evaluated on test (opt rows; NoMemory pairs: result files)."""
    import psycopg

    with psycopg.connect(drv.OPT_DSN) as c:
        raw = c.execute(
            "SELECT query_id, protocol->'space_config' FROM opt.session_retrieval WHERE experiment_id = %s "
            "AND split = 'test' AND status = 'evaluated'",
            (experiment,),
        ).fetchall()
    done = {(space.key(cfg), q) for q, cfg in raw if cfg}
    nm_dir = run_dir(ds, a.sampling) / "test" / "pairs" / cfg_name(space.NOMEM)
    done |= {(space.key(space.NOMEM), q) for q in drv.item_files(nm_dir, ds)}
    want = {
        (space.key(r["config"]), q)
        for x in arms.values()
        for q, r in x["routes"].items()
    }
    return done & want


def cv_stage(ds: str, a) -> dict:
    """Outer 5-fold CV of Opt-OneConfig / Opt-PerQuery on the profiled training queries (folds stratified by the
    pool stratum), then the frozen CV test routing (``test/routing_cv.json``). Offline: no episode runs here.
    Sparse samples: held-out values use measured outcomes only (``outer_cv.heldout_eval``)."""
    ensure_ablation_execution_ready(a)
    from trimem.experiments import router_outer_cv as rcv
    from trimem.experiments import slo_select_bo as slo
    from trimem.optimizer.query_router import sparse

    base = run_dir(ds, a.sampling)
    if not (base / "selection.json").exists():
        raise SystemExit("freeze the selection (select stage) before the CV stage")
    sel = json.load(open(base / "selection.json"))
    T = profile_table(ds, a)
    pool, state = T["pool"], T["state"]
    strata, w, sizes = pool_meta(pool)
    est = dict(T["estimates"])
    if (
        a.nomemory_candidate
        and (nm_cand := nomemory_candidate(ds, a, state)) is not None
    ):
        est[space.key(space.NOMEM)] = nm_cand
    prof = sparse.from_estimates(est, strata, w, sizes)
    raw_science = ds == "sciworld" and a.router_backend == "joint-selected"
    if raw_science:
        if router_backends(state, a):
            raise ValueError("joint-selected raw ScienceWorld routing currently requires --no_limeqo")
        if not state.get("ablation_arm"):
            prof = router_raw_science_profile(prof, state, a.sampling)
    out = base / "router"
    recs = {
        r["query_id"]: r
        for r in feature_records(
            ds, prof.queries, pool, False, out / "features_train.json"
        )
    }
    snap = out / "training_snapshot.json"
    if snap.exists():  # BRANE and LimeQO folds use the router stage's frozen TRAIN snapshot (checked identical)
        from dataclasses import asdict

        from trimem.experiments.shared_router import load_snapshot

        _, sp_prof, sp_recs, _ = load_snapshot(snap)
        if json.loads(json.dumps(asdict(prof))) != json.loads(
            json.dumps(asdict(sp_prof))
        ) or json.loads(json.dumps(recs)) != json.loads(json.dumps(sp_recs)):
            raise SystemExit(
                f"the CV training profile / features differ from the frozen {snap}; rerun the router stage"
            )
    tids = test_ids(ds)
    trecs = feature_records(ds, tids, pool, True, out / "features_test.json")
    keys = {p["key"]: p["label"] for p in T["points"]}
    cols = [i for i, c in enumerate(prof.configs) if space.key(c) in keys]
    labels = {i: keys[space.key(prof.configs[i])] for i in cols}
    _, nm = nomemory_estimate(
        ds,
        pool,
        nomemory_sample(state),
        nomemory_dir(ds, a.sampling),
        max_round=eb.view_max_round(state),
        raw_score=bool(state.get("ablation_arm")),
    )
    points = T["points"]
    if raw_science:
        files = drv.item_files(nomemory_dir(ds, a.sampling), ds)
        nm = {q: float(json.loads(files[q].read_text())["reward"]) for q in nm}
        by_key = {space.key(cfg): i for i, cfg in enumerate(prof.configs)}
        points = [{**pt, "q": prof.est(by_key[pt["key"]], "quality").mean} for pt in points]
    best_q = max(points, key=lambda p: (p["q"], -p["lat"]))
    common = dict(
        dataset=ds,
        prof=prof,
        records=recs,
        test_records=trecs,
        groups=tuple(a.groups.split(",")),
        oneconfig_cols=cols,
        labels=labels,
        nomemory=nm,
        bounds_fn=lambda pts: router_bounds(ds, pts, raw_science=raw_science),
        choose_fn=slo.choose,
        objectives={n: slo.SLOS[n]["objective"] for n in SLO_NAMES},
        fallback=best_q["cfg"],
        trained_on={
            "experiment": state["experiment"],
            "pool_hash": pool["pool_hash"],
            "memory_policy": state["memory_policy"],
        },
        log=log,
    )
    rep = rcv.run_folds(
        base,
        **common,
        test_weights=TEST_WEIGHTS,
        fit_kwargs={"trials": a.trials, "n_jobs": a.jobs, "seed": a.seed, "backend": a.router_backend,
                    **({"quality_kind": "raw_score"} if raw_science else {})},
        backend=a.router_backend,
    )
    # LimeQO backend (user 2026-10-21): the same folds, records, candidates rule, utility and test weights
    # (skipped when state.json router_backends disables it)
    if router_backends(state, a):
        lime = rcv.run_folds(
            base,
            **common,
            test_weights={rcv.LIMEQO_PREFIX + k: w for k, w in TEST_WEIGHTS.items()},
            fit_kwargs={
                "backend": "limeqo",
                "seed": a.seed,
                "shared_options": dict(rcv.LIMEQO_OPTIONS),
            },
            cv_dir=rcv.LIMEQO_CV_DIR,
            backend="limeqo",
        )
        rep = rcv.with_backend(rep, lime)
    else:
        log("CV LimeQO: disabled for this run (state.json router_backends)")
    dep = routing_table(
        ds, a, sel
    )  # the deployed arms (frozen here if the test stage has not run yet)
    tab = rcv.freeze_routing(
        cv_routing_path(ds, a),
        rep=rep,
        test_ids=tids,
        validate=space.validate,
        deployed=dep,
        key=space.key,
        nomemory_key="",  # agent-env NoMemory router pairs RUN in the pairs dir (never reused): counted as new
        completed=lambda arms: completed_test_pairs(ds, a, dep["experiment"], arms),
        extra={
            "experiment": dep["experiment"],
            "test_version": TEST_VERSION,
            "catalog": sel.get("catalog"),
            "replay_estimator": sparse.REPLAY_ESTIMATOR,
        },
        log=log,
    )
    return tab


# ------------------------------------------------------------------ test


def test_baseline(ds: str, a) -> tuple[Path, str]:
    """Paired-quality baseline of the test arms: this protocol's NoMemory if it was run (``--test_fixed``),
    else the earlier controlled NoMemory arm (official evaluator, same 200 items, success-only-v0 era serving
    setup): quality pairing is valid, latency / token comparisons with the old arms are NOT like-for-like."""
    own = test_arm_dir(ds, "NoMemory", a.sampling)
    if len(drv.item_files(own, ds)) >= 200:
        return own, "pool-strat200 NoMemory"
    return (
        drv.CTRL / f"{ds}_NoMemory",
        "earlier controlled-v1 NoMemory (not like-for-like on cost)",
    )


def test_max_round(ds: str, sampling: str = "per-config") -> int | None:
    """max-round-v1: runner max_round of every test-time arm of ``ds`` (deployed test, NoMemory / reference arms,
    fixed plans, PerQuery sweep) = the profiling run's recorded ``episode_budget`` (None: COMMON's 30, unchanged);
    also the view of the 30-turn references they are compared with."""
    p = run_dir(ds, sampling) / "state.json"
    return eb.view_max_round(json.loads(p.read_text())) if p.exists() else None


def test_ids(ds: str) -> list[str]:
    return [
        x["item_id"]
        for s_ in ("a", "b")
        for x in json.load(open(drv.CTRL / "shards" / f"{ds}_{s_}.json"))
    ]


def routing_table(ds: str, a, sel: dict) -> dict:
    """Frozen per-query routing of every test arm (written once, before any test episode): arm -> query ->
    {config, overhead_ms}. Opt-OneConfig-SLO-k: its configuration for every query; Opt-PerQuery-<name>: the frozen
    weight-slice artifact at each (w_Q, w_L, w_T) of ``TEST_WEIGHTS`` (user 2026-10-19: no slice arms, no
    Opt-PerQuery SLO arms on test; their train choices are recorded in ``perquery_slo_train_only``).
    Routing is offline with the frozen artifacts and the test feature records cached by the router stage
    (deterministic: same features and decision rule the live router would use); router overhead = the
    record's probe time + the measured transform / prediction time."""
    from trimem.optimizer.query_router import runtime, sparse

    base = run_dir(ds, a.sampling)
    p = base / "test" / "routing.json"
    if sel.get("router_protocol") != sparse.PROTOCOL:
        raise SystemExit(
            f"selection.json router protocol {sel.get('router_protocol')!r} != {sparse.PROTOCOL!r}"
        )
    ids = test_ids(ds)
    if (
        p.exists()
    ):  # frozen; the optional fixed-plan arms (router-independent) may be added later
        tab = json.load(open(p))
        if a.test_fixed and any(
            plan not in tab["arms"] for plan in fixed_test_plans(ds, a)
        ):
            for plan in fixed_test_plans(ds, a):
                cfg = space.validate({"plan": plan, **space.plan_defaults()})
                tab["arms"].setdefault(
                    plan,
                    {
                        "kind": "fixed",
                        "routes": {q: {"config": cfg, "overhead_ms": 0.0} for q in ids},
                    },
                )
            save(p, tab)
        return tab
    recs = {
        r["query_id"]: r
        for r in json.load(open(base / "router" / "features_test.json"))["records"]
    }
    missing = set(ids) - set(recs)
    if missing:
        raise SystemExit(f"test feature records missing for {sorted(missing)[:3]}")
    rr = json.load(open(base / "router" / "router_report.json"))
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
    # Opt-PerQuery on test (user 2026-10-19): exactly TEST_WEIGHTS, routed with the frozen weight-slice artifact
    # (same model as every SLO artifact; ``route_test_weights`` applies each weight vector with the deployed rule).
    # The train SLO choices of Opt-PerQuery stay in selection.json / router_report.json but are NOT test arms.
    c = sel["weight_slices"]
    art = json.load(open(c["path"]))
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
    if lc := sel.get(
        "weight_slices_limeqo"
    ):  # LimeQO backend (user 2026-10-21), same weights / rule
        lart = json.load(open(lc["path"]))
        if runtime.artifact_sha(lart) != lc["sha"]:
            raise SystemExit(
                "LimeQO router artifact changed since the selection was frozen"
            )
        lovh = {
            q: recs[q]["overhead"]["probe_ms"] + sel["limeqo_transform_ms"] for q in ids
        }
        lnames = [
            (f"Opt-PerQuery-LimeQO-{label}", w) for label, w in TEST_WEIGHTS.items()
        ]
        for (name, w), routes in zip(
            lnames,
            sparse.route_test_weights(
                lart, [recs[q] for q in ids], lnames, lovh
            ).values(),
            strict=True,
        ):
            arms[name] = {
                "kind": "perquery",
                "router": "limeqo",
                "weights": list(w),
                "weights_name": sparse.arm_name(w),
                "sha": lc["sha"],
                "routes": routes,
            }
    if a.test_fixed:
        for plan in fixed_test_plans(ds, a):
            cfg = space.validate({"plan": plan, **space.plan_defaults()})
            arms[plan] = {
                "kind": "fixed",
                "routes": {q: {"config": cfg, "overhead_ms": 0.0} for q in ids},
            }
    tab = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "experiment": test_experiment(ds, "pairs"),
        "router_protocol": sel["router_protocol"],
        "arms": arms,
        "perquery_slo_train_only": {
            n: sel["selections"][n]["opt_perquery"] for n in SLO_NAMES
        },
        "perquery_limeqo_slo_train_only": {
            n: sel["selections"][n].get("opt_perquery_limeqo") for n in SLO_NAMES
        },
        "test_concurrency": TEST_CONCURRENCY[ds],
        "reused_arms_concurrency": REUSED_TEST_CONCURRENCY,
        "pair_budget": pair_budget(arms),
        "catalog": sel.get("catalog"),
        "replay_estimator": sel.get("replay_estimator"),
    }
    log(f"routing frozen: {len(arms)} arms; pair budget {tab['pair_budget']}")
    p.parent.mkdir(parents=True, exist_ok=True)
    save(p, tab)
    return tab


def pair_budget(arms: dict[str, dict]) -> dict[str, int]:
    """Offline episode count of a routing table: distinct (config, query) pairs of the Opt-OneConfig arms, of all
    arms, and the extra pairs the Opt-PerQuery (and optional fixed-plan) arms add on top. Pure."""

    def pairs(kinds: tuple[str, ...] | None) -> set:
        return {
            (space.key(r["config"]), q)
            for arm in arms.values()
            if kinds is None or arm["kind"] in kinds
            for q, r in arm["routes"].items()
        }

    one_p, all_p = pairs(("oneconfig",)), pairs(None)
    return {
        "oneconfig_pairs": len(one_p),
        "all_pairs": len(all_p),
        "perquery_extra_pairs": len(all_p - one_p),
    }


def pairs_of(tab: dict) -> dict[str, list[str]]:
    """Config-reuse principle: config key -> the test queries any arm routes to it; each pair runs ONCE."""
    out: dict[str, set] = {}
    for arm in tab["arms"].values():
        for q, r in arm["routes"].items():
            out.setdefault(space.key(r["config"]), set()).add(q)
    return {
        k: sorted(v, key=lambda q: int(q.rsplit("_", 1)[1]))
        for k, v in sorted(out.items())
    }


def test_stage(ds: str, a) -> None:
    """Route every test query for every arm offline (frozen), then execute each distinct (configuration,
    query) pair once -- one subset arm per configuration, all in parallel -- and assemble the arms in
    ``report``. Shared executions make the weight arms' results correlated (same episode reused): fine for a
    curve, not independent replicates."""
    ensure_ablation_execution_ready(a)
    base = run_dir(ds, a.sampling)
    if not (base / "selection.json").exists():
        raise SystemExit(
            "selection.json missing: freeze the selection (select stage) before any test episode"
        )
    sel = json.load(open(base / "selection.json"))
    tab = routing_table(ds, a, sel)
    cv = load_cv_routing(
        ds, a
    )  # outer-CV fold arms (cv stage): pairs shared with the deployed arms run once
    if cv is not None:
        n_new = cv["pair_budget"]["new_pairs"]
        if n_new <= a.cv_test_max_pairs:
            tab = {**tab, "arms": {**tab["arms"], **cv["arms"]}}
            log(
                f"test: + {len(cv['arms'])} outer-CV arms (pair budget {cv['pair_budget']})"
            )
        else:
            log(
                f"test: outer-CV test arms DEFERRED: {n_new} new pairs > --cv_test_max_pairs "
                f"{a.cv_test_max_pairs} (rerun the test stage with a larger cap)"
            )
    st_p = base / "state.json"
    if st_p.exists() and (
        rp := runtime_profile_block(ds, json.loads(st_p.read_text()), record=False)
    ):
        log(f"test under runtime profile {rp['name']} (state.json runtime_profile)")
    elif not st_p.exists() and runtime_profile_want(ds) is not None:
        raise SystemExit("runtime profile requested but state.json missing; refusing")
    ids = test_ids(ds)
    if (
        a.test_fixed
    ):  # this protocol's NoMemory first: the paired baseline of every test row
        nm = test_arm_dir(ds, "NoMemory", a.sampling)
        if run_capped(ds, [Arm("test NoMemory", nm, ids, None)], TEST_CONCURRENCY[ds])[
            "test NoMemory"
        ] < len(ids):
            raise SystemExit("test NoMemory incomplete")
    baseline, _ = test_baseline(ds, a)
    pairs = pairs_of(tab)
    log(
        f"test: {len(tab['arms'])} arms -> {sum(len(v) for v in pairs.values())} distinct (config, query) pairs "
        f"over {len(pairs)} configurations (vs {len(tab['arms']) * len(ids)} without reuse)"
    )
    arms = []
    for k, qs in pairs.items():
        cfg = json.loads(k)
        if (
            cfg["plan"] == "NoMemory"
        ):  # router candidate: official evaluator, cost in episodes.jsonl
            arms.append(
                Arm(
                    "test NoMemory pairs",
                    base / "test" / "pairs" / cfg_name(cfg),
                    qs,
                    None,
                )
            )
            continue
        session = {
            **space.runner_args(cfg),
            "runner_extra": space.runner_cli(cfg),
            "split": "test",
            "experiment": tab["experiment"],
            "baseline_dir": baseline,
            "panel": None,
            "reuse": a.reuse,  # quality-reuse-v1 in the test stage too (test rows only)
            "reuse_from": [],
            "reuse_from_dirs": [],
            "protocol_extra": {
                "panel_hash": f"test:{TEST_VERSION}",
                "space_version": space.VERSION,
                "space_config": cfg,
                "bo_label": "test-pairs",
                "round": PANEL,
                "test_version": TEST_VERSION,
                "test_concurrency": TEST_CONCURRENCY[ds],
            },
        }
        arms.append(
            Arm(
                f"test {cfg_name(cfg)}",
                base / "test" / "pairs" / cfg_name(cfg),
                qs,
                session,
            )
        )
    log(
        f"test concurrency {TEST_CONCURRENCY[ds]} (user 2026-10-20; the reused controlled arms ran at "
        f"{REUSED_TEST_CONCURRENCY})"
    )
    res = run_capped(ds, arms, TEST_CONCURRENCY[ds])
    log_reuse(tab["experiment"], "test", "test pairs")
    bad = [
        k for k, v in res.items() if v < len(next(x.ids for x in arms if x.name == k))
    ]
    if bad:
        raise SystemExit(f"test pairs incomplete: {bad}")
    log(f"TEST DONE {ds}")


def _se(v: list[float]) -> float:
    m = sum(v) / len(v)
    return (
        math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1) / len(v))
        if len(v) > 1
        else 0.0
    )


def assemble(
    tab: dict, rows: dict[tuple[str, str], dict], nm: dict[str, float], key=None
) -> dict:
    """Per arm: per-query result = the single execution of its routed (config, query) pair (+ router
    overhead on latency); quality mean +- SE, mean tokens, mean latency (s), distinct configs, paired gain vs
    the NoMemory baseline. Pure (unit-tested). ``key`` = the configuration key of the rows' space (default this
    driver's agent-env space; τ² passes its own, -v5 fixes memory_form per dataset)."""
    key = key or space.key
    out = {}
    for name, arm in tab["arms"].items():
        got = [
            (q, r, rows.get((key(r["config"]), q))) for q, r in arm["routes"].items()
        ]
        have = [(q, r, x) for q, r, x in got if x is not None]
        res = {
            "kind": arm["kind"],
            "lambda": arm.get("lambda"),
            "weights": arm.get("weights"),
            "slices": arm.get("slices"),
            "cost_share": arm.get("cost_share"),
            "slo": arm.get("slo"),
            "n": len(got),
            "covered": len(have),
            "distinct_configs": len({key(r["config"]) for _, r, _ in got}),
        }
        if have:
            q = [x["quality"] for _, _, x in have]
            d = [x["quality"] - nm[qid] for qid, _, x in have if qid in nm]
            res.update(
                quality=sum(q) / len(q),
                quality_se=_se(q),
                tokens=sum(x["tokens"] for _, _, x in have) / len(have),
                latency_s=sum(x["e2e_ms"] + r["overhead_ms"] for _, r, x in have)
                / len(have)
                / 1000,
                paired_gain_vs_nomemory=sum(d) / len(d) if d else None,
                paired_gain_se=_se(d) if d else None,
                **step_metrics([x for _, _, x in have]),
            )
        out[name] = res
    return out


def step_metrics(rows: list[dict]) -> dict:
    """Agent steps of an arm's routed rows (user 2026-10-21): ``steps`` = mean per-task steps (ALFWorld /
    ScienceWorld: environment steps = agent LLM turns; τ²: agent calls), its SE across tasks, the number of tasks
    with a step value (a missing value is never counted as 0), and ``steps_success`` = mean over the tasks'
    successful episodes only. Rows without the fields (older callers) give None."""
    st = [x["steps"] for x in rows if x.get("steps") is not None]
    ss = [x["steps_success"] for x in rows if x.get("steps_success") is not None]
    return {
        "steps": sum(st) / len(st) if st else None,
        "steps_se": _se(st) if st else None,
        "steps_n": len(st),
        "steps_success": sum(ss) / len(ss) if ss else None,
        "steps_success_n": len(ss),
    }


def episode_steps(
    ds: str, generation: dict | None, result_file: Path | None
) -> int | None:
    """Environment steps of one agent-env episode: the opt row's recorded ``environment_steps``, else recomputed
    from the result file's conversation (``agent_step_metrics.agentgym_steps``: (len - 3) // 2); None when
    neither is available (never 0)."""
    from trimem.experiments import agent_step_metrics as sm

    v = (generation or {}).get("environment_steps")
    if v is not None:
        return int(v)
    f = result_file or (
        Path(generation["result_file"])
        if (generation or {}).get("result_file")
        else None
    )
    if f is None or not f.exists():
        return None
    return sm.agentgym_steps(json.load(open(f)))["environment_steps"]


def with_steps(ds: str, row: dict, steps: int | None, *, raw_science: bool = False) -> dict:
    """An episode row + ``steps`` and ``steps_success`` (steps if the episode succeeded, else None)."""
    return {
        **row,
        "steps": steps,
        "steps_success": steps if steps is not None and row["quality"] >= (100.0 if raw_science else 1.0) else None,
    }


def report_stage(ds: str, a) -> dict:
    ensure_ablation_execution_ready(a)
    import psycopg
    from psycopg.rows import dict_row

    base = run_dir(ds, a.sampling)
    tab = json.load(open(base / "test" / "routing.json"))
    cv = load_cv_routing(ds, a)
    all_arms = {**tab, "arms": {**tab["arms"], **(cv or {}).get("arms", {})}}
    nm_dir, nm_label = test_baseline(ds, a)
    raw_science = ds == "sciworld" and a.router_backend == "joint-selected"
    nm = {
        i: float(json.load(open(f))["reward"]) if raw_science else quality_of(ds, json.load(open(f))["reward"])
        for i, f in drv.item_files(nm_dir, ds).items()
    }
    with psycopg.connect(drv.OPT_DSN, row_factory=dict_row) as c:
        raw = c.execute(
            "SELECT query_id, protocol->'space_config' AS cfg, quality, total_tokens, e2e_ms, generation FROM "
            "opt.session_retrieval WHERE experiment_id = %s AND split = 'test' AND status = 'evaluated'",
            (tab["experiment"],),
        ).fetchall()
    rows = {
        (space.key(r["cfg"]), r["query_id"]): with_steps(
            ds,
            {
                "quality": float(r["generation"]["reward"]) if raw_science else float(r["quality"]),
                "tokens": float(r["total_tokens"]),
                "e2e_ms": float(r["e2e_ms"]),
            },
            episode_steps(ds, r.get("generation"), None),
            raw_science=raw_science,
        )
        for r in raw
    }
    nm_key = space.key(
        space.NOMEM
    )  # NoMemory router-candidate pairs: official evaluator files (no opt rows)
    nm_pairs = pairs_of(all_arms).get(nm_key, [])
    if nm_pairs:
        nm_dir_pairs = base / "test" / "pairs" / cfg_name(space.NOMEM)
        rows |= {
            (nm_key, q): {k: v for k, v in r.items() if k != "cost_source"}
            for q, r in nomemory_rows(ds, nm_dir_pairs, nm_pairs).items()
        }
        if raw_science:
            files = drv.item_files(nm_dir_pairs, ds)
            for q in nm_pairs:
                key = (nm_key, q)
                if key in rows:
                    rows[key]["quality"] = float(json.loads(files[q].read_text())["reward"])
                    rows[key]["steps_success"] = rows[key].get("steps") if rows[key]["quality"] >= 100 else None
    arms = assemble(tab, rows, nm)
    # Reporting only: the frozen routing table and its evaluations stay unchanged.
    from trimem.experiments import difficulty_report as difficulty

    task_types = difficulty.load_task_types(ds)
    expected = test_ids(ds)
    difficulty_arms = {}
    for name, arm in tab["arms"].items():
        episodes = {}
        for qid, route in arm["routes"].items():
            row = rows.get((space.key(route["config"]), qid))
            if row is not None:
                episodes[qid] = {
                    "quality": row["quality"],
                    "tokens": row["tokens"],
                    "latency_s": (row["e2e_ms"] + route["overhead_ms"]) / 1000,
                }
        difficulty_arms[name] = difficulty.summarize(
            ds, expected, task_types, episodes, nm, success_threshold=100.0 if raw_science else 1.0
        )
    breakdown = {
        "definition": difficulty.DEFINITION,
        "baseline": difficulty.summarize(
            ds, expected, task_types, {q: {"quality": v} for q, v in nm.items()}, nm,
            success_threshold=100.0 if raw_science else 1.0,
        ),
        "arms": difficulty_arms,
    }
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "dataset": ds,
        "experiment": tab["experiment"],
        "baseline": {
            "dir": str(nm_dir),
            "label": nm_label,
            "quality": sum(nm.values()) / max(len(nm), 1),
        },
        "notes": [
            "each distinct (configuration, query) pair ran ONCE and is shared by every arm routing to it: "
            "arms (weight points) are correlated, fine for a curve, not independent replicates",
            "comparisons with the earlier success-only-v0 NoMemory / fixed-plan test arms are not like-for-like "
            "(memory policy, session space, serving setup and load differ)",
        ],
        "pairs": len(rows),
        "arms": arms,
        "difficulty": breakdown,
    }
    if (
        cv is not None
    ):  # outer 5-fold CV (user 2026-10-21): primary numbers, mean ± std across folds
        from trimem.experiments import router_outer_cv as rcv

        def fold_metrics(t: dict) -> dict:
            got = assemble(t, rows, nm)
            return {k: v for k, v in got.items() if v.get("covered") == v.get("n")}

        ts = rcv.test_summary(cv, fold_metrics)
        rep["outer_cv"] = {
            "protocol": cv["protocol"],
            "folds": cv["folds"],
            "pair_budget": cv["pair_budget"],
            "evaluation": "test set only (user 2026-10-21); folds fitted on 4/5 of the training queries",
            "test": ts,
            "incomplete_test_arms": sorted(
                n
                for n, v in assemble(cv, rows, nm).items()
                if v.get("covered") != v.get("n")
            ),
            "labels": {
                "mean_std": "mean ± sample std (ddof 1) across the 5 outer folds",
                "se_tasks_mean": "mean within-fold SE across tasks (task-sampling noise), NOT across folds",
            },
        }
        md = rcv.markdown(ts, "raw score" if raw_science else "quality" if ds == "alfworld" else "score/100", len(expected))
        (base / "report_cv.md").write_text(
            f"# {ds} pool profiling — outer 5-fold CV\n\n" + "\n".join(md) + "\n"
        )
    rep["test_concurrency"] = tab.get("test_concurrency")
    rep["reused_arms_concurrency"] = tab.get("reused_arms_concurrency")
    rep["reference_arms"] = reference_arms(ds, expected, nm, raw_science=raw_science)
    write_report_md(ds, base, rep, len(expected))
    save(base / "test_report.json", rep)
    for k, v in arms.items():
        if v.get("quality") is not None:
            log(
                f"{k:26s} q {v['quality']:.3f} ± {v['quality_se']:.3f}  tok {v['tokens']:,.0f}  "
                f"lat {v['latency_s']:.1f}s  configs {v['distinct_configs']}  ({v['covered']}/{v['n']})"
            )
    return rep


REFERENCE_ARMS = ("NoMemory", "VGR")  # reused controlled-v1 test arms (user 2026-10-21)


def reference_arms(ds: str, ids: list[str], nm: dict[str, float], *, raw_science: bool = False) -> dict[str, dict]:
    """The reused controlled-v1 test arms as references (single run): NoMemory (official evaluator files; cost
    from its call logs, ``nomemory_rows``) and VGR (opt rows of its controlled-v1 experiment, success-only-v0
    era: untagged id). Same metric fields as ``assemble``."""
    import psycopg

    out = {}
    for name in REFERENCE_ARMS:
        if name == "NoMemory":
            rows = nomemory_rows(ds, drv.CTRL / f"{ds}_NoMemory", ids)
            if raw_science:
                files = drv.item_files(drv.CTRL / f"{ds}_NoMemory", ds)
                for q, row in rows.items():
                    row["quality"] = float(json.loads(files[q].read_text())["reward"])
                    row["steps_success"] = row.get("steps") if row["quality"] >= 100 else None
            src = "controlled-v1 NoMemory (official evaluator; cost from llm_calls / episodes logs)"
        else:
            exp = f"{ds}:agenteval200:controlled-v1:{name}"
            with psycopg.connect(drv.OPT_DSN) as c:
                raw = c.execute(
                    "SELECT query_id, quality, total_tokens, e2e_ms, generation FROM opt.session_retrieval "
                    "WHERE experiment_id = %s AND split = 'test' AND status = 'evaluated'",
                    (exp,),
                ).fetchall()
            files = drv.item_files(drv.CTRL / f"{ds}_{name}", ds)
            rows = {
                q: with_steps(
                    ds,
                    {"quality": float(g["reward"]) if raw_science else float(x), "tokens": float(t), "e2e_ms": float(e)},
                    episode_steps(ds, g, files.get(q)),
                    raw_science=raw_science,
                )
                for q, x, t, e, g in raw
                if q in set(ids)
            }
            src = (
                f"controlled-v1 {name} (opt rows {exp}; memory policy success-only-v0)"
            )
        have = [rows[q] for q in ids if q in rows and "tokens" in rows[q]]
        d = [rows[q]["quality"] - nm[q] for q in ids if q in rows and q in nm]
        out[name] = {
            "protocol": src,
            "n": len(ids),
            "covered": len(have),
            "quality": sum(r["quality"] for r in have) / len(have) if have else None,
            "quality_se": _se([r["quality"] for r in have]) if have else None,
            "tokens": sum(r["tokens"] for r in have) / len(have) if have else None,
            "latency_s": sum(r["e2e_ms"] for r in have) / len(have) / 1000
            if have
            else None,
            "paired_gain_vs_nomemory": sum(d) / len(d) if d else None,
            "paired_gain_se": _se(d) if d else None,
            **step_metrics(have),
        }
    return out


def write_report_md(ds: str, base: Path, rep: dict, n_test: int) -> None:
    """report.md (pool_cv_report): CV test table, references, deployed arms, per-fold selections / candidates and
    the BRANE vs LimeQO vs Opt-OneConfig comparison."""
    from trimem.experiments import pool_cv_report as pr
    from trimem.experiments import router_outer_cv as rcv

    sel = json.load(open(base / "selection.json"))
    names = {p["key"]: p["label"] for p in sel["points"]}

    def label_of(cfg: dict) -> str:
        if cfg.get("plan") == "NoMemory":
            return "NoMemory"
        k = space.key(cfg)
        return names.get(k, cfg_name(cfg))

    rr = json.load(open(base / "router" / "router_report.json"))
    dep = {"BRANE": rr["candidates"]}
    if rr.get("limeqo"):
        dep["LimeQO"] = rr["limeqo"]["candidates"]
    profile = json.load(open(base / "profile_complete.json"))
    brane, lime = (
        pr.load_folds(base, rcv.CV_DIR),
        pr.load_folds(base, rcv.LIMEQO_CV_DIR),
    )
    rep["cv_selections"] = {
        "oneconfig": {
            str(f["fold"]): {
                n: v
                and {"label": label_of(v["cfg"]), "cfg": v["cfg"], "train": v["train"]}
                for n, v in f["oneconfig"].items()
            }
            for f in brane
        },
        "perquery_slo_weights": {
            name: {
                str(f["fold"]): {
                    n: v and v["weights"] for n, v in f["perquery_slo"].items()
                }
                for f in recs
            }
            for name, recs in (("BRANE", brane), ("LimeQO", lime))
        },
        "router_candidates": {
            name: {
                str(f["fold"]): [label_of(c) for c in f["router"]["candidate_configs"]]
                for f in recs
            }
            for name, recs in (("BRANE", brane), ("LimeQO", lime))
        }
        | {"deployed": {k: [label_of(c) for c in v] for k, v in dep.items()}},
    }
    md = pr.render(
        ds=ds,
        rep=rep,
        profile=profile,
        brane_folds=brane,
        limeqo_folds=lime,
        deployed_candidates=dep,
        label_of=label_of,
        refs=rep["reference_arms"],
        weights=list(TEST_WEIGHTS),
        slos=list(SLO_NAMES),
        n_test=n_test,
    )
    (base / "report.md").write_text(md + "\n")


# ------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sa.add_arguments(ap)
    ap.add_argument("--full_panel_size", type=int, default=None)
    ap.add_argument("--initial_budget_episodes", type=int, default=None)
    ap.add_argument("--task_sample_namespace", default=None)
    ap.add_argument("--stage_sizes", default=None, help="fresh-panel checkpoint sizes as ascending comma-separated integers")
    ap.add_argument("--ablation_arm", choices=("full", "without_llm", "without_recombination", "without_local", "without_pruning"), default=None)
    ap.add_argument(
        "stage", choices=("profile", "router", "select", "cv", "test", "report", "all")
    )
    ap.add_argument(
        "--dataset", choices=("alfworld", "sciworld", "webshop"), required=True
    )
    ap.add_argument(
        "--cv_test_max_pairs",
        type=int,
        default=CV_TEST_MAX_PAIRS,
        help="test: run the outer-CV test arms only when they add at most this many new (config, query) pairs",
    )
    ap.add_argument(
        "--no_nomemory_candidate",
        dest="nomemory_candidate",
        action="store_false",
        help="router without the NoMemory candidate (default: NoMemory is a candidate)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="BO rounds of the UNPRUNED plan: the configuration budget is (10 + rounds x batch) x 200 "
        "episodes; with staging the number of rounds follows from the budget",
    )
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument(
        "--run_tag",
        default=None,
        help=f"run tag (env {RUN_TAG_ENV}): own experiment id and run directory (<ds>_pool-strat200-v7-<tag>); "
        "NoMemory stays the untagged run's; seed a branch with trimem.experiments.pool_branch",
    )
    ap.add_argument(
        "--channel_slots",
        choices=sorted(CHANNEL_SLOTS_CHOICES),
        default=None,
        help=f"channel-slot protocol from the next round (env {CHANNEL_SLOTS_ENV}); v3 = 10-slot layout "
        "(qfirst, token-first, latency-first, llm x2, xover x2, dls x2, sobol); v4 = 10-slot layout (sobol x4, "
        "llm x2, xover x2, dls x2; no qfirst); recorded in state.json; v3 -> v4 upgrade allowed, downgrades refused",
    )
    ap.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help=f"sample-size-v1 (env {SAMPLE_SIZE_ENV}): tasks per NEW configuration from the next round",
    )
    ap.add_argument(
        "--sample_uniform",
        action="store_true",
        help=f"sample-size-v1 (env {SAMPLE_UNIFORM_ENV}=1): draw those samples uniformly from the pool",
    )
    ap.add_argument(
        "--plan_p1",
        action="store_true",
        help=f"plan-p1-v1 (env {PLAN_P1_ENV}=1): add the gated plan P1 (V+SemFilter+G) to the proposable plans "
        "from the next round (recorded in state.json plan_extension_p1)",
    )
    ap.add_argument(
        "--budget_episodes",
        type=int,
        default=None,
        help=f"budget-extension-v1 (env {BUDGET_EPISODES_ENV}): raise the profiling episode budget from the next "
        "round (recorded in state.json budget_extension; needs --budget_reason)",
    )
    ap.add_argument(
        "--budget_max_configs",
        type=int,
        default=None,
        help=f"budget-extension-v1 (env {BUDGET_MAX_CONFIGS_ENV}): raise max_configs from the next round",
    )
    ap.add_argument(
        "--budget_reason",
        default=None,
        help=f"reason recorded with a budget extension (env {BUDGET_REASON_ENV})",
    )
    ap.add_argument(
        "--batch_steps",
        default=None,
        help=f"batch-schedule-v1 steps (env {BATCH_STEPS_ENV}), e.g. 10@6,4@7: batch 10 from round 6, 4 from 7",
    )
    ap.add_argument(
        "--random_rounds",
        type=int,
        default=None,
        help=f"with --channel_slots v4 (env {RANDOM_ROUNDS_N_ENV}): the next N proposed rounds are sobol-only-v1 "
        "random-sampling rounds; v4 after them",
    )
    ap.add_argument(
        "--random_rounds_until",
        type=int,
        default=None,
        help=f"with --channel_slots v4 (env {RANDOM_ROUNDS_ENV}): rounds from the next proposed one up to this "
        "round are sobol-only-v1 random-sampling rounds (all slots sobol-slot); v4 from the round after",
    )
    ap.add_argument(
        "--round_batch",
        type=int,
        default=None,
        help=f"per-round batch from the next round (env {BATCH_ENV}; state.json batch_schedule); unlike --batch "
        "it changes neither the run identity nor the staged budget",
    )
    ap.add_argument(
        "--pause_after_round",
        type=int,
        default=None,
        help=f"{PAUSE_PROTOCOL}: stop before proposing the next round (env {PAUSE_ENV}); resume = rerun without it",
    )
    ap.add_argument(
        "--pause_reason",
        default=None,
        help="recorded in profile_complete.json / state.json profiling_pause",
    )
    ap.add_argument(
        "--no_limeqo",
        dest="limeqo",
        action="store_false",
        default=None,
        help=f"router/cv/test without the LimeQO backend (BRANE + Opt-OneConfig only; env {NO_LIMEQO_ENV}=1); "
        "recorded in state.json router_backends",
    )
    ap.add_argument(
        "--limeqo", dest="limeqo", action="store_true",
        help="also run the optional LimeQO comparison (legacy backends only for raw ScienceWorld)",
    )
    ap.add_argument(
        "--no_analyst",
        dest="analyst",
        action="store_false",
        help="disable LLM analysis (new runs: local-search guidance; historical runs: legacy analyst)",
    )
    ap.add_argument(
        "--staging",
        default="pareto",
        choices=("pareto", STAGING_OFF),
        help=f"pareto (default): BO configurations staged {'/'.join(map(str, STAGE_SIZES))} with 3-D Pareto "
        "pruning (fresh TRACE defaults to syftr-style); none: every configuration in full, fixed rounds (old design)",
    )
    pp_policy.add_arguments(ap)
    ap.add_argument(
        "--max_configs",
        type=int,
        default=MAX_CONFIGS,
        help="staging: at most this many configurations in total (round 0 included)",
    )
    ap.add_argument(
        "--sampling",
        default="per-config",
        choices=("per-config", "shared"),
        help="per-config (default): every arm draws its own stratified sample; shared: one common "
        "sample for every arm (old design; use with --estimator paired)",
    )
    ap.add_argument(
        "--estimator",
        default="unpaired",
        choices=("unpaired", "paired"),
        help="BO noise model: per-configuration means and SEs (default) or paired gains (old)",
    )
    ap.add_argument(
        "--no_reuse",
        dest="reuse",
        action="store_false",
        help="disable quality reuse of identical-memory episodes within the experiment",
    )
    ap.add_argument(
        "--groups",
        default=None,
        help="router feature groups (no LLM)",
    )
    ap.add_argument(
        "--router_backend",
        choices=("brane-config", "joint", "brane", "joint-selected"),
        default="joint-selected",
        help="joint-selected (default): one cross-configuration predictor per metric with TRAIN-only model selection",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=30,
        help="router: Optuna trials per candidate (paper: 30)",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--test_fixed",
        action="store_true",
        help="test: also NoMemory + the 4 plan defaults under this protocol (optional later stage; "
        "added to an already frozen routing table, pairs already run are reused)",
    )
    return ap


def resolve_router_defaults(a):
    if a.groups is None:
        a.groups = "structured,emb" if a.router_backend == "joint-selected" else "emb"
    if a.limeqo is None:
        a.limeqo = a.router_backend != "joint-selected" and os.environ.get(NO_LIMEQO_ENV) != "1"
    return a


def main(argv: list[str] | None = None) -> int:
    a = resolve_router_defaults(build_parser().parse_args(argv))
    if getattr(a, "ablation_arm", None):
        from trimem.experiments.scienceworld_trace_ablation import (
            validate_driver_arguments,
        )

        validate_driver_arguments(a)
        ensure_ablation_execution_ready(a)
    if a.run_tag:
        os.environ[RUN_TAG_ENV] = a.run_tag
    run_tag()  # validate
    if (a.sampling == "shared") != (a.estimator == "paired"):
        raise SystemExit("--estimator paired requires --sampling shared and vice versa")
    (OUT / a.dataset).mkdir(parents=True, exist_ok=True)
    lock = open(OUT / a.dataset / "driver.lock", "w")
    try:  # one driver per dataset (state, opt rows, env servers)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"another driver holds {OUT / a.dataset / 'driver.lock'}"
        ) from None

    def stop(signum, frame):
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit(
            "the pool round runs under memory policy success+failure-v1 (each v2 configuration's "
            "outcome_policy restricts what it retrieves); set TRIMEM_MEMORY_POLICY accordingly"
        )
    if a.estimator == "paired" and a.staging != STAGING_OFF:
        raise SystemExit("--estimator paired runs unstaged: add --staging none")
    log(
        f"bo_pool_driver {a.stage} {a.dataset}{' [run tag ' + run_tag() + ']' if run_tag() else ''}: space {space.VERSION}, memory policy {mp.current()}, sampling {a.sampling}, "
        f"estimator {a.estimator}, staging {a.staging}"
    )
    stages = (
        ("profile", "router", "select", "cv", "test", "report")
        if a.stage == "all"
        else (a.stage,)
    )
    for s in stages:
        {
            "profile": profile,
            "router": router_stage,
            "select": select_stage,
            "cv": cv_stage,
            "test": test_stage,
            "report": report_stage,
        }[s](a.dataset, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
