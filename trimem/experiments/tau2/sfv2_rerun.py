"""τ² pool -v5: rerun the SemFilter-plan configurations with semfilter-v2, then a DEPLOYED-only test (user decision
2026-10-22; prepared, launched by ``results/pool_v5_20261020/run_tau2_sfv2.sh`` after ALFWorld / ScienceWorld finish).

Run: ``results/tau2_pool_v5_20261020`` (``pool_profile``; 38 configurations: VGR 25, VGR+Rank 4, P2 2, P2+Rank 7;
profiling paused after round 6; 1 trial per task on the train178 panel; test 100 tasks x 1 trial, pass^1).

* semfilter-v2 (``semfilter_v2``: top-32 query-cosine prefilter, batched K=8 score-only guided json_schema integer
  tenths, rubric semantic-rubric-v3 with the τ² CONFLICT clause = ``semantic_rubric`` KNOWLEDGE / CONFLICTS
  ``tau2``; judge input = ``tau2_card``) is recorded ONCE in state.json ``semfilter`` (protocol semfilter-v2,
  regime semfilter-regime-v1, from_round 0, the knobs, the env, opt-in ``TAU2_SEMFILTER=v2``) by ``supersede``, as
  ALFWorld / ScienceWorld record theirs (``bo_pool_driver.semfilter_block``). From then on every pool_profile
  stage exports the recorded TRIMEM_SEMFILTER_* env (``apply_semfilter``; refused without the opt-in or with
  conflicting knobs); ``tau2.runner_pool`` builds the v2 judge for SemFilter plans only (VGR / VGR+Rank unaffected)
  and labels every retrieval (opt-row protocol ``semfilter``, result-file ``memory.semfilter`` receipt).
* supersede (per configuration, ``state.json semfilter.superseded``): the 9 SemFilter-plan configurations (P2 /
  P2+Rank, all rounds) -- their v1 episode files (``configs/<cfg>``) MOVE to ``_superseded_sfv1_<ts>/`` (never
  deleted), their opt rows (train + test experiment, every status not yet superseded) get status
  ``superseded-sfv1`` (previous status in the backup's rows.json), the 7 active ones get staging reset to 0 (old
  staging kept in the record); the 2 user-stopped round-0 configurations (sim_gate) stay stopped (their partial v1
  episodes are superseded too, nothing reruns). The derived artifacts of the old selection (selection.json,
  profile.json, router/, router_cv/, test/routing.json, test/routing_cv.json, test_report.*, report.md, ...) and
  the SemFilter-plan TEST pair dirs (v1 judge) move to ``_archive_presfv2_<ts>/``; the router feature caches are
  copied back (task features do not depend on the configurations). VGR / VGR+Rank keep their train episodes and
  their test pairs (same c40 test runtime: resumed, not rerun).
* rerun: the active superseded configurations on ALL 178 train tasks x 1 trial (no staging / pruning: they were all
  fully evaluated), at the run's recorded profiling runtime (load regime of the last round, ``TAU2_POOL_CONCURRENCY``
  total episodes, retrieval slots / semantic workers of the τ² queue env), quality reuse restricted to a NEW
  index (``<experiment>:sfv2``: only semfilter-v2 episodes, never a v1 one); staging then records 178 / episodes /
  reuse hits and ``sfv2_rerun``.
* offline: pool_profile router (BRANE, zscore-log-v1 utility, selfnorm-v1 replay = the sparse defaults) -> select
  (Opt-OneConfig SLO-1/2/3) -> the frozen DEPLOYED routing table; NO cv stage (no outer 5-fold CV).
* test: pool_profile test on the deployed arms only (refused while test/routing_cv.json exists), the test quality
  reuse index also ``:sfv2``; report: pool_profile report (deployed) + the semfilter / deployed-only labels.

Stages: ``plan`` (dry run: counts, files, rows, time estimate; writes nothing), ``supersede [--apply]``, ``rerun``,
``offline``, ``test``, ``report``. ``TAU2_POOL_OUT`` points pool_profile (and this module) at a COPY for a dry run.

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 .venv-optimizer/bin/python -m \\
        trimem.experiments.tau2.sfv2_rerun {plan|supersede|rerun|offline|test|report} [--apply] [--reason ...]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import statistics as st
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import semfilter_v2 as sf
from trimem.experiments.tau2 import pool_profile as pp

SEMFILTER_ENV = "TAU2_SEMFILTER"  # opt-in (= POOL_SEMFILTER of bo_pool_driver)
REGIME_PROTOCOL = "semfilter-regime-v1"
ROW_STATUS = "superseded-sfv1"
TAG = ":sfv2"  # quality-reuse index namespace of semfilter-v2 episodes (train and test)
SEMFILTER_PLANS = ("P1", "P1+Rank", "P2", "P2+Rank")
ARCHIVE_ITEMS = (
    "selection.json",
    "profile.json",
    "router",
    "router_cv",
    "test/routing.json",
    "test/routing_cv.json",
    "test_report.json",
    "test_report.md",
    "test_analysis.json",
    "report.md",
    "rounds_summary.md",
)
FEATURE_CACHES = ("features_train.json", "features_test.json")
DEFAULT_REASON = (
    "user 2026-10-22: rerun the τ² SemFilter-plan configurations (P2 / P2+Rank) with semfilter-v2 (top-32 cosine "
    "prefilter, batched K=8 score-only guided integer tenths, rubric v3 with the τ² CONFLICT clause); v1 episodes "
    "superseded, VGR / VGR+Rank kept; router / selection retrained; deployed-only test"
)


def log(msg: str) -> None:
    pp.log(f"sfv2: {msg}")


# ------------------------------------------------------------------ the recorded block (pure helpers)


def want_settings(env=None) -> dict:
    """The semfilter-v2 knobs to record: ``semfilter_v2.settings`` of the defaults (+ any TRIMEM_SEMFILTER_* the
    caller set), with the gate on. Raw judge input only (τ² cards)."""
    env = dict(os.environ if env is None else env)
    cfg = sf.settings({**env, sf.ENV_GATE: "1"})
    if cfg["judge_input"] != "raw":
        raise SystemExit("τ² semfilter-v2: judge input raw (tau2_card) only")
    return cfg


def env_of(cfg: dict) -> dict[str, str]:
    """TRIMEM_SEMFILTER_* env of recorded knobs (= ``bo_pool_driver.semfilter_env``). Pure."""
    return {
        sf.ENV_GATE: "1",
        sf.ENV_TOP_N: str(cfg["top_n"]),
        sf.ENV_BATCH: str(cfg["batch"]),
        sf.ENV_MAX_TOKENS: str(cfg["max_tokens"]),
        sf.ENV_JUDGE_INPUT: cfg["judge_input"],
        sf.ENV_RUBRIC: cfg["rubric"],
        sf.ENV_GUIDED: "1" if cfg["guided"] else "0",
    }


def semfilter_configs(state: dict) -> list[dict]:
    """SemFilter-plan configurations of the run: round, position, cfg name, plan, active / stopped / pruned,
    staging. Pure."""
    out = []
    for rr in state["rounds"]:
        for j, c in enumerate(rr["configs"]):
            if c["config"]["plan"] not in SEMFILTER_PLANS:
                continue
            g = c.get("staging") or {}
            out.append(
                {
                    "round": rr["round"],
                    "position": j,
                    "config": pp.cfg_name(c["config"]),
                    "plan": c["config"]["plan"],
                    "active": not pp.inactive(c),
                    "stopped_by_user": bool(g.get("stopped_by_user")),
                    "pruned_at": g.get("pruned_at"),
                    "stage_reached": g.get("stage_reached"),
                    "episodes": g.get("episodes"),
                }
            )
    return out


def semfilter_block(cfg: dict, configs: list[dict], reason: str, now: str) -> dict:
    """state.json ``semfilter`` (semfilter-v2, from round 0), in the ALFWorld / ScienceWorld schema. Pure."""
    return {
        "protocol": sf.PROTOCOL,
        "regime_protocol": REGIME_PROTOCOL,
        "from_round": 0,
        "top_n": cfg["top_n"],
        "batch": cfg["batch"],
        "max_tokens": cfg["max_tokens"],
        "judge_input": cfg["judge_input"],
        "rubric": cfg["rubric"],
        "guided": cfg["guided"],
        "rubric_dataset": "tau2",
        "before": "semfilter-v1: every candidate judged, one call per candidate, long JSON verdict (evidence / "
        "transfer), rubric of the memory policy (semantic-rubric-v2) with the τ² knowledge text",
        "env": env_of(cfg),
        "opt_in": f"{SEMFILTER_ENV}=v2",
        "scope": "SemFilter plans (P1 / P1+Rank / P2 / P2+Rank); VGR / VGR+Rank unaffected (kept as measured)",
        "superseded": {
            "configs": [c["config"] for c in configs],
            "rerun": [c["config"] for c in configs if c["active"]],
            "kept_stopped": [c["config"] for c in configs if not c["active"]],
            "row_status": ROW_STATUS,
        },
        "reason": reason,
        "recorded_at": now,
    }


def check_semfilter(state: dict, env=None) -> dict[str, str] | None:
    """The env to export for a recorded block (refusing without ``TAU2_SEMFILTER=v2`` or with other knobs), or None
    when nothing is recorded (refusing ``TRIMEM_SEMFILTER_V2=1`` without the opt-in: unlabelled). Pure."""
    env = os.environ if env is None else env
    blk = state.get("semfilter")
    want = env.get(SEMFILTER_ENV, "").strip()
    if blk is None:
        if not want and env.get(sf.ENV_GATE, "").strip() == "1":
            raise SystemExit(
                f"{sf.ENV_GATE}=1 without {SEMFILTER_ENV}=v2: τ² SemFilter v2 episodes would be unrecorded"
            )
        return None
    if blk.get("protocol") != sf.PROTOCOL:
        raise SystemExit(f"state.json semfilter {blk.get('protocol')!r}; refusing")
    if want != "v2":
        raise SystemExit(
            f"state.json records {blk['protocol']} from round {blk['from_round']}; run with {SEMFILTER_ENV}=v2"
        )
    out = dict(blk["env"])
    diff = {
        k: (env.get(k), v)
        for k, v in out.items()
        if env.get(k) is not None and env.get(k) != v
    }
    if diff:
        raise SystemExit(f"TRIMEM_SEMFILTER_* differ from the recorded block: {diff}")
    return out


def apply_semfilter(out: Path | None = None) -> dict | None:
    """Export the recorded semfilter-v2 env of ``OUT/state.json`` (every pool_profile stage calls this)."""
    p = (out or pp.OUT) / "state.json"
    if not p.exists():
        return None
    env = check_semfilter(json.loads(p.read_text()))
    if env:
        os.environ.update(env)
    return env


# ------------------------------------------------------------------ supersede


def opt_rows(state: dict, names: set[str]) -> list[dict]:
    """Not-yet-superseded opt rows of these configurations in the train experiment and the test experiment."""
    import psycopg
    from psycopg.rows import dict_row

    exps = [state["experiment"], pp.test_experiment()]
    with psycopg.connect(opt_dsn(), row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT retrieval_id, experiment_id, query_id, split, status, created_at, "
            "protocol->'space_config' AS cfg FROM opt.session_retrieval WHERE experiment_id = ANY(%s) "
            "AND status NOT LIKE 'superseded%%'",
            (exps,),
        ).fetchall()
    return [
        x
        for x in rows
        if x["cfg"] is not None
        and x["cfg"].get("plan") in SEMFILTER_PLANS
        and pp.cfg_name(x["cfg"]) in names
    ]


def opt_dsn() -> str:
    from trimem.experiments.tau2 import runner as R

    return R.OPT_DSN  # the τ² runners' opt rows (shared opt schema)


def test_pair_dirs(out: Path) -> list[Path]:
    """SemFilter-plan TEST pair dirs (any configuration a v1 test routed to)."""
    d = out / "test" / "pairs"
    res = []
    for p in sorted(d.glob("cfg_*")) if d.exists() else []:
        f = p / "arm.json"
        cfg = None
        if f.exists():
            ck = (json.loads(f.read_text()) or {}).get("config_key")
            cfg = json.loads(ck) if ck else None
        if cfg is None:
            cmd = p / ".command"
            if cmd.exists():
                argv = json.loads(cmd.read_text())
                if "--space_config" in argv:
                    cfg = json.loads(argv[argv.index("--space_config") + 1])
        if cfg and cfg.get("plan") in SEMFILTER_PLANS:
            res.append(p)
    return res


def result_files(d: Path) -> int:
    return sum(1 for f in d.glob("*.json") if f.name != "arm.json") if d.exists() else 0


def supersede_plan(state: dict, *, rows: bool = True) -> dict:
    out = pp.OUT
    confs = semfilter_configs(state)
    names = {c["config"] for c in confs}
    train_dirs = [out / "configs" / n for n in sorted(names)]
    tdirs = test_pair_dirs(out)
    rr = opt_rows(state, names) if rows else []
    by: dict[str, int] = {}
    for x in rr:
        k = f"{x['split']}:{x['status']}"
        by[k] = by.get(k, 0) + 1
    return {
        "semfilter_configs": confs,
        "n_configs": len(confs),
        "n_rerun": sum(c["active"] for c in confs),
        "train_dirs": {
            str(p.relative_to(out)): result_files(p) for p in train_dirs if p.exists()
        },
        "test_pair_dirs": {str(p.relative_to(out)): result_files(p) for p in tdirs},
        "archive": [x for x in ARCHIVE_ITEMS if (out / x).exists()],
        "opt_rows": len(rr),
        "opt_rows_by_split_status": by,
        "row_status": ROW_STATUS,
        "already_recorded": state.get("semfilter") is not None,
    }


def driver_lock():
    pp.OUT.mkdir(parents=True, exist_ok=True)
    lock = open(pp.OUT / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another driver holds {pp.OUT / 'driver.lock'}") from None
    return lock


REAL_OUT = pp.ROOT / "results/tau2_pool_v5_20261020"


def supersede(reason: str, apply: bool, skip_db: bool = False) -> dict:
    """Dry run (default): the plan. ``apply``: record the block, move files, supersede rows, reset staging.
    ``skip_db`` (a dry-run COPY only, never the real run): no opt-row update."""
    out = pp.OUT
    if skip_db and out.resolve() == REAL_OUT.resolve():
        raise SystemExit(
            "--skip_db is for a dry-run copy (TAU2_POOL_OUT), not the real run"
        )
    state_p = out / "state.json"
    state = json.loads(state_p.read_text())
    if state.get("semfilter") is not None:
        raise SystemExit("state.json already records semfilter (supersede done)")
    cfg = want_settings()
    p = supersede_plan(state)
    now = datetime.now(UTC).isoformat()
    blk = semfilter_block(cfg, p["semfilter_configs"], reason, now)
    p["block"] = blk
    if not apply:
        return p
    _lock = driver_lock()
    ts = now[:19].replace(":", "")
    dest = out / f"_superseded_sfv1_{ts}"
    arch = out / f"_archive_presfv2_{ts}"
    dest.mkdir()
    arch.mkdir()
    rows = (
        []
        if skip_db
        else opt_rows(state, {c["config"] for c in p["semfilter_configs"]})
    )
    (dest / "state_before.json").write_text(json.dumps(state, indent=1))
    (dest / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    for rel in p["train_dirs"]:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).rename(dest / rel)
    for rel in p["test_pair_dirs"]:
        (arch / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).rename(arch / rel)
    for rel in p["archive"]:
        (arch / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).rename(arch / rel)
    if (arch / "router").exists():  # task features do not depend on the configurations
        (out / "router").mkdir(exist_ok=True)
        for f in FEATURE_CACHES:
            if (arch / "router" / f).exists():
                shutil.copy2(arch / "router" / f, out / "router" / f)
    if rows:
        import psycopg

        with psycopg.connect(opt_dsn()) as c:
            n = c.execute(
                "UPDATE opt.session_retrieval SET status = %s, updated_at = now(), "
                "error = coalesce(error || ' | ', '') || %s WHERE retrieval_id = ANY(%s)",
                (
                    ROW_STATUS,
                    f"{ROW_STATUS}: {reason}",
                    [x["retrieval_id"] for x in rows],
                ),
            ).rowcount
        if n != len(rows):
            raise SystemExit(f"superseded {n} rows, expected {len(rows)}")
    rerun = set(blk["superseded"]["rerun"])
    staging_before = {}
    for rr in state["rounds"]:
        for c in rr["configs"]:
            n = pp.cfg_name(c["config"])
            if n in rerun:
                staging_before[n] = c.get("staging")
                c["staging"] = {
                    "stage_reached": 0,
                    "pruned_at": None,
                    "decisions": [],
                    "episodes": 0,
                    "sfv2_superseded": {"at": now, "staging_before": c.get("staging")},
                }
    blk["superseded"].update(
        backup=str(dest),
        archive=str(arch),
        moved_train=p["train_dirs"],
        moved_test_pairs=p["test_pair_dirs"],
        archived=p["archive"],
        rows_superseded=len(rows),
        rows_skipped_dry_copy=skip_db,
        rows_by_split_status_before=p["opt_rows_by_split_status"],
    )
    state["semfilter"] = blk
    pp.save(state_p, state)
    log(
        f"superseded {len(p['semfilter_configs'])} configurations -> {dest}; archive {arch}"
    )
    return p


# ------------------------------------------------------------------ rerun / offline / test / report


def _args(extra: list[str] | None = None) -> argparse.Namespace:
    return pp.build_parser().parse_args(["router", *(extra or [])])


def rerun_arms(state: dict, panel: dict) -> list:
    """The superseded active configurations on all 178 train tasks (full split, sfv2 reuse index)."""
    from trimem.experiments import quality_reuse as qr

    keys = pp.train_keys(panel)
    names = set(state["semfilter"]["superseded"]["rerun"])
    last = state["rounds"][-1]["round"]
    arms = []
    for rr in state["rounds"]:
        for j, c in enumerate(rr["configs"]):
            if pp.cfg_name(c["config"]) not in names:
                continue
            a = pp.config_arm(
                c["config"], f"r{rr['round']}c{j}-sfv2", keys, panel, state, last
            )
            a.protocol_extra = {
                **a.protocol_extra,
                "sfv2_rerun": sf.PROTOCOL,  # (not "semfilter": the runner's protocol.semfilter block)
            }
            a.reuse_index = qr.index_path(
                pp.REUSE_DIR, state["experiment"] + TAG, "train"
            )
            arms.append(a)
    return arms


def v2_labelled(out: Path) -> tuple[int, int]:
    """(result files, files whose memory receipt is labelled semfilter-v2) of a configuration dir."""
    from trimem.experiments.tau2 import runner as R

    n = v = 0
    for f in out.glob("*.json"):
        if f.name == "arm.json":
            continue
        r = R.read_result(f)
        if r is None:
            continue
        n += 1
        v += ((r.get("memory") or {}).get("semfilter") or {}).get(
            "protocol"
        ) == sf.PROTOCOL
    return n, v


def rerun() -> dict:
    out = pp.OUT
    state_p = out / "state.json"
    state = json.loads(state_p.read_text())
    if not (state.get("semfilter") or {}).get("superseded"):
        raise SystemExit(
            "no semfilter supersede recorded: run `supersede --apply` first"
        )
    if not apply_semfilter():
        raise SystemExit("semfilter env not exported")
    _lock = driver_lock()
    panel = pp.load_panel()
    arms = rerun_arms(state, panel)
    total = int(os.environ.get("TAU2_POOL_CONCURRENCY", pp.CONCURRENCY))
    log(
        f"rerun {len(arms)} configurations x {len(pp.train_keys(panel))} tasks at {total} concurrent episodes "
        f"(regime {arms[0].regime if arms else '-'}; reuse index {TAG})"
    )
    res = pp.run_parallel(arms, total)
    bad = [a.name for a in arms if res[a.name] < a.expected]
    if bad:
        raise SystemExit(f"sfv2 rerun incomplete: {bad}")
    keys = pp.train_keys(panel)
    now = datetime.now(UTC).isoformat()
    state = json.loads(state_p.read_text())
    for a in arms:
        n, v = v2_labelled(a.out)
        if v != n:
            raise SystemExit(
                f"{a.name}: {n - v} of {n} episodes lack the semfilter-v2 receipt"
            )
    for rr in state["rounds"]:
        for c in rr["configs"]:
            if pp.cfg_name(c["config"]) not in set(
                state["semfilter"]["superseded"]["rerun"]
            ):
                continue
            stats = pp.reuse_stats(c["config"], keys)
            g = c["staging"]
            g.update(
                stage_reached=len(keys),
                episodes=stats["episodes"],
                reused=stats["hits"],
                reuse=stats,
                sfv2_rerun={"done_at": now, "tasks": len(keys), "unstaged": True},
            )
    state["semfilter"]["rerun_done_at"] = now
    pp.save(state_p, state)
    # the pause marker's budget follows the staging (render_cost_saving_hv cross-checks fresh episodes with it)
    done_p = out / "profile_complete.json"
    done = json.loads(done_p.read_text())
    if "budget_used_before_sfv2" not in done:
        done["budget_used_before_sfv2"] = done.get("budget_used")
    done["budget_used"] = pp.episodes_used(state)
    done["sfv2_rerun"] = {
        "at": now,
        "configs": sorted(state["semfilter"]["superseded"]["rerun"]),
        "episodes": sum(res.values()),
        "note": "budget_used = fresh profiling episodes of the CURRENT staging (the superseded v1 episodes of "
        "the rerun configurations replaced by their semfilter-v2 episodes; quality-reuse hits excluded)",
    }
    pp.save(done_p, done)
    log("rerun done")
    return {a.name: res[a.name] for a in arms}


def offline(jobs: int | None = None) -> dict:
    """router -> select -> the frozen deployed routing table (no outer CV)."""
    if not apply_semfilter():
        raise SystemExit("state.json has no semfilter-v2 record")
    if (pp.OUT / "test" / "routing_cv.json").exists():
        raise SystemExit(
            "test/routing_cv.json exists: this test is deployed-only (archive it)"
        )
    _lock = driver_lock()
    a = _args(["--jobs", str(jobs)] if jobs else None)
    if not (pp.OUT / "selection.json").exists():
        pp.router_stage(a)
        pp.select_stage(a)
    sel = json.loads((pp.OUT / "selection.json").read_text())
    return pp.routing_table(sel)


def test() -> None:
    if not apply_semfilter():
        raise SystemExit("state.json has no semfilter-v2 record")
    if (pp.OUT / "test" / "routing_cv.json").exists():
        raise SystemExit("test/routing_cv.json exists: this test is deployed-only")
    _lock = driver_lock()
    pp.test_stage(_args())


def report() -> dict:
    if not apply_semfilter():
        raise SystemExit("state.json has no semfilter-v2 record")
    _lock = driver_lock()
    rep = pp.report_stage(_args())
    state = json.loads((pp.OUT / "state.json").read_text())
    rep |= {
        "deployed_only": True,
        "outer_cv": None,
        "semfilter": {
            k: state["semfilter"][k]
            for k in ("protocol", "rubric", "top_n", "batch", "from_round")
        },
    }
    pp.save(pp.OUT / "test_report.json", rep)
    return rep


# ------------------------------------------------------------------ dry-run plan (counts + time estimate)


def latency_minutes(dirs: list[Path]) -> list[float]:
    """Episode wall minutes of the v1 result files of ``dirs``."""
    from trimem.experiments.tau2 import runner as R

    out = []
    for d in dirs:
        for f in d.glob("*.json") if d.exists() else []:
            if f.name == "arm.json":
                continue
            r = R.read_result(f)
            if r is not None and r.get("wall_ms") is not None:
                out.append(float(r["wall_ms"]) / 60000)
    return out


def plan() -> dict:
    """Everything the launch would do, with counts and a wall-time estimate; writes nothing."""
    out = pp.OUT
    state = json.loads((out / "state.json").read_text())
    p = supersede_plan(state)
    confs = p["semfilter_configs"]
    keys = pp.train_keys(pp.load_panel())
    conc = int(os.environ.get("TAU2_POOL_CONCURRENCY", pp.CONCURRENCY))
    tconc = int(os.environ.get("TAU2_TEST_CONCURRENCY", pp.TEST_CONCURRENCY))
    rerun_eps = sum(c["active"] for c in confs) * len(keys) * pp.PROFILE_TRIALS
    v1 = latency_minutes([out / "configs" / c["config"] for c in confs if c["active"]])
    m = st.mean(v1) if v1 else None
    tab_p = out / "test" / "routing.json"
    old_pairs = {}
    if tab_p.exists():
        old = pp.pairs_of(json.loads(tab_p.read_text()))
        for k, qs in old.items():
            plan_ = json.loads(k)["plan"]
            grp = (
                "nomemory"
                if plan_ == "NoMemory"
                else "semfilter"
                if plan_ in SEMFILTER_PLANS
                else "other"
            )
            old_pairs[grp] = old_pairs.get(grp, 0) + len(qs)
    tl = latency_minutes(list(test_pair_dirs(out)))
    tm = st.mean(tl) if tl else m
    sf_test = old_pairs.get("semfilter", 0)
    return {
        "out": str(out),
        "supersede": {k: v for k, v in p.items() if k != "semfilter_configs"},
        "semfilter_configs": confs,
        "rerun_episodes": rerun_eps,
        "rerun_concurrency": conc,
        "v1_episode_wall_min_mean": m,
        "rerun_hours_estimate": None
        if m is None
        else round(rerun_eps * m / conc / 60, 2),
        "estimate_note": "saturated-throughput estimate = episodes x mean v1 episode wall / concurrency (v1 judge: "
        "every candidate one call; semfilter-v2 judges <= 32 candidates in batches of 8, so the rerun should be "
        "faster); router / select ~ minutes (CPU)",
        "old_deployed_test_pairs_by_group": old_pairs,
        "test_pairs_semfilter_rerun_estimate": sf_test,
        "test_concurrency": tconc,
        "test_hours_estimate": None
        if tm is None
        else round(sf_test * tm / tconc / 60, 2),
        "test_note": "deployed arms only (7); the new selection decides the pairs: the old deployed routing's "
        "SemFilter-plan pairs are the estimate (SemFilter-free pairs of configurations routed before are resumed "
        "from test/pairs, new ones run)",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "stage", choices=("plan", "supersede", "rerun", "offline", "test", "report")
    )
    ap.add_argument(
        "--apply", action="store_true", help="supersede: do it (default: dry run)"
    )
    ap.add_argument(
        "--skip_db",
        action="store_true",
        help="supersede --apply on a dry-run COPY: no opt-row update",
    )
    ap.add_argument("--reason", default=DEFAULT_REASON)
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args(argv)
    if a.stage == "plan":
        print(json.dumps(plan(), indent=1, default=str))
    elif a.stage == "supersede":
        print(
            json.dumps(supersede(a.reason, a.apply, a.skip_db), indent=1, default=str)
        )
    elif a.stage == "rerun":
        print(json.dumps(rerun(), indent=1))
    elif a.stage == "offline":
        tab = offline(a.jobs)
        print(
            json.dumps(
                {
                    n: len(
                        {
                            json.dumps(r["config"], sort_keys=True)
                            for r in x["routes"].values()
                        }
                    )
                    for n, x in tab["arms"].items()
                },
                indent=1,
            )
        )
    elif a.stage == "test":
        test()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
