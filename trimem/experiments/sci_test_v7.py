"""ScienceWorld pool-strat200-v7 TEST after profiling, DEPLOYED arms only (user decisions 2026-10-22).

Profiling (``results/pool_v5_20261020/run_sci_c16.sh``) pauses after round 6 (50 configurations: round 0 = 5 plan
defaults + 5 Sobol on the stage-100 prefix of their 200-task samples (round0-prefix-v1), round 1 = 10 Sobol picks,
round 2 = channel-slots-v5 x 4, rounds 3-5 = channel-slots-v6 x 8, round 6 = 2; rounds >= 1 at 100 stratified
tasks), every round under controlled-c16-v1 and -- after the round-0 SemFilter supersede
(``pool_discard_round --semfilter_from_round 0``) -- semfilter-v2 for every SemFilter configuration.

* catalog (``bo_pool_driver`` catalog-regime-v1, recorded in state.json by the router stage with
  POOL_CATALOG_FROM_ROUND=0): every round, fully evaluated configurations only (pruned partial samples never
  enter); NoMemory = this run's own NoMemory (c16 from round 0). The router stage refuses while state.json
  semfilter is still recorded from round 1 (catalog_block floors).
* router: BRANE only (state.json router_backends, --no_limeqo), zscore-log-v1 utility, self-normalised replay
  (``sparse.REPLAY_ESTIMATOR`` selfnorm-v1): the defaults.
* test arms (user 2026-10-22): ONLY the DEPLOYED arms, i.e. selection / router fitted on the full train split:
  Opt-OneConfig SLO-1/2/3 + Opt-PerQuery BRANE at the 4 TEST_WEIGHTS (default 0.6/0.2/0.2, quality 0.8/0.1/0.1,
  tokens 0.1/0.1/0.8, latency 0.1/0.8/0.1). NO outer 5-fold CV (no ``bo_pool_driver cv`` stage, no
  ``test/routing_cv.json``); the deployed routing table ``test/routing.json`` is frozen by ``offline``
  (``bo_pool_driver.routing_table``) before any test episode.
* test set = the 200 AgentEval ScienceWorld test items (``bo_pool_driver.test_ids``: results/eval_controlled_20261015/
  shards/sciworld_a/b.json), runtime = ``alf_test_c16``'s controlled-c16-v1 layout (one configuration at a time,
  2 halves x 8 processes, small configurations share one final batch, both-halves monitor), quality reuse OFF, the
  state.json semfilter-v2 knobs exported to every runner (inert for SemFilter-free plans).
* no earlier ScienceWorld test pairs exist for this run: every pair runs (into ``test_v7c16/pairs`` / opt experiment
  ``...:pairs-c16-v7:...``).
* references: the controlled-v1 originals (results/eval_controlled_20261015/sciworld_NoMemory: LLM-call span latency
  only, no episodes.jsonl; sciworld_VGR: opt experiment ``sciworld:agenteval200:controlled-v1:VGR``, wall + span);
  paired gain vs the controlled-v1 NoMemory.

Stages: ``offline`` (router -> select -> freeze test/routing.json; CPU only, skips what is frozen), ``plan`` (pair
counts; nothing runs), ``run`` (``--dry``: plan.json + log only), ``report`` (test_v7c16_report.json +
report_v7c16.md: deployed table quality ± SE, gain ± SE, tokens, wall / span latency, steps ± SE, configurations
used, routing shares; references). ``dry_copy`` builds an offline COPY of the run (state truncated to
``--last_round``, a fake profile_complete.json, NoMemory symlinked) under ``--out`` for a dry run; ``--out`` points
every stage at such a copy (``bo_pool_driver.OUT`` redirected; never the real run dir).

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 POOL_SEMFILTER=v2 .venv-optimizer/bin/python -m \\
        trimem.experiments.sci_test_v7 {offline|plan|run|report} [--dry] [--out DIR]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import alf_test_c16 as c16
from trimem.experiments import alf_test_v7 as v7

DS = "sciworld"
CONCURRENCY = c16.CONCURRENCY  # 16 = 2 halves x 8 (controlled-c16-v1)
SUBDIR = v7.SUBDIR  # test_v7c16 (inside the ScienceWorld run dir)
ARM_TAG = v7.ARM_TAG  # pairs-c16-v7 -> opt experiment sciworld:...:pairs-c16-v7:...
REPORT = v7.REPORT
REPORT_MD = v7.REPORT_MD
PROTOCOL = "sci-v7-test-deployed-v1"
CATALOG_FROM_ROUND = 0
REF_VGR_EXPERIMENT = f"{DS}:agenteval200:controlled-v1:VGR"
DRY_MARK = "dry_run_copy"  # key of the fake profile_complete.json of a dry-run copy


def redirect(out: str | None) -> None:
    """Point ``bo_pool_driver`` at a dry-run copy root (``base_dir`` / ``nomemory_dir`` / the driver lock)."""
    if not out:
        return
    from trimem.experiments import bo_pool_driver as d

    root = Path(out).resolve()
    real = Path(d.drv.OUT).resolve()
    if root == real:
        raise SystemExit("--out is the real results root; a dry run needs a copy")
    d.OUT = root


def run_base() -> Path:
    from trimem.experiments import bo_pool_driver as d

    return d.run_dir(DS)


# ------------------------------------------------------------------ dry-run copy


def dry_copy(out: str, last_round: int) -> dict:
    """An offline copy of the ScienceWorld run under ``out`` for a dry run: state.json truncated to rounds <=
    ``last_round`` (pause marker faked), samples copied, configs / NoMemory symlinked READ-ONLY. Never writes to the
    real run directory."""
    from trimem.experiments import bo_pool_driver as d

    real = d.run_dir(DS)
    real_nm = d.nomemory_dir(DS, regime=False)
    redirect(out)
    dst = d.run_dir(DS)
    if dst.exists():
        raise SystemExit(f"{dst} exists; use a fresh --out")
    dst.mkdir(parents=True)
    s = json.loads((real / "state.json").read_text())
    have = [r["round"] for r in s["rounds"]]
    s["rounds"] = [r for r in s["rounds"] if r["round"] <= last_round]
    s.pop("catalog_regime", None)
    (dst / "state.json").write_text(json.dumps(s, indent=1))
    shutil.copytree(real / "samples", dst / "samples")
    (dst / "configs").symlink_to(real / "configs")
    nm = d.nomemory_dir(DS, regime=False)
    nm.parent.mkdir(parents=True, exist_ok=True)
    nm.symlink_to(real_nm)
    rec = {
        DRY_MARK: True,
        "source": str(real),
        "rounds_in_source": have,
        "last_round": last_round,
        "paused_by_user": True,
        "note": "fake pause marker of an offline dry-run copy; no episodes",
        "created_at": datetime.now(UTC).isoformat(),
    }
    (dst / "profile_complete.json").write_text(json.dumps(rec, indent=1))
    return {"copy": str(dst), "rounds": [r["round"] for r in s["rounds"]], **rec}


# ------------------------------------------------------------------ offline (router -> select -> deployed routing)


def driver_args(jobs: int | None = None) -> argparse.Namespace:
    from trimem.experiments import bo_pool_driver as d

    argv = ["router", "--dataset", DS, "--no_limeqo"]
    return d.build_parser().parse_args(argv + (["--jobs", str(jobs)] if jobs else []))


def offline(jobs: int | None = None) -> dict:
    """Router (BRANE, catalog rounds >= 0) -> select -> the frozen DEPLOYED routing table; no CV stage. Skips
    what is already frozen for this catalog. Holds the dataset's driver lock (as ``bo_pool_driver.main``)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import memory_policy as mp

    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    base = run_base()
    p = base / "test" / "routing.json"
    if p.exists():
        tab = routing(base)
        d.log(f"offline: {p} already frozen ({len(tab['arms'])} arms)")
        return tab
    (d.OUT / DS).mkdir(parents=True, exist_ok=True)
    lock = open(d.OUT / DS / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another driver holds {d.OUT / DS / 'driver.lock'}") from None
    a = driver_args(jobs)
    sel_p = base / "selection.json"
    if not sel_p.exists():
        d.log("offline: router stage (BRANE only, catalog rounds >= 0)")
        d.router_stage(DS, a)
        d.log("offline: select stage")
        d.select_stage(DS, a)
    sel = json.loads(sel_p.read_text())
    if (sel.get("catalog") or {}).get("from_round") != CATALOG_FROM_ROUND:
        raise SystemExit(
            f"selection.json catalog {sel.get('catalog')!r} is not the rounds >= {CATALOG_FROM_ROUND} catalog: "
            "archive it and rerun with POOL_CATALOG_FROM_ROUND=0"
        )
    d.log("offline: freezing the deployed routing table (no outer CV)")
    d.routing_table(DS, a, sel)
    return routing(base)


# ------------------------------------------------------------------ routing + pairs


def routing(base: Path) -> dict:
    """The catalog-regime (rounds >= 0) DEPLOYED routing table; refuses unlabelled / LimeQO / fixed-plan ones."""
    tab = json.loads((base / "test" / "routing.json").read_text())
    cat = tab.get("catalog") or {}
    if cat.get("from_round") != CATALOG_FROM_ROUND:
        raise SystemExit(
            f"test/routing.json is not the catalog-regime (rounds >= {CATALOG_FROM_ROUND}) table: rerun the "
            "offline stage with POOL_CATALOG_FROM_ROUND=0"
        )
    bad = [
        n
        for n, a in tab["arms"].items()
        if c16.is_excluded(n, a) or a.get("kind") not in ("oneconfig", "perquery")
    ]
    if bad:
        raise SystemExit(
            f"test/routing.json has non-deployed arms {bad}; this test is BRANE deployed only"
        )
    return tab


def done_pairs(base: Path, run: dict[str, list[str]]) -> int:
    """Pairs of ``run`` that already have a result file in this test's pair dirs (a resumed run)."""
    from trimem.experiments import bo_pool_driver as d

    n = 0
    for k, qs in run.items():
        have = d.drv.item_files(base / SUBDIR / "pairs" / d.cfg_name(json.loads(k)), DS)
        n += len(set(qs) & set(have))
    return n


def arm_summary(tab: dict) -> dict[str, dict]:
    """Per deployed arm: kind, SLO / weights, routed tasks and distinct configurations (pure)."""
    from trimem.experiments import bo_pool_driver as d

    return {
        n: {
            "kind": a["kind"],
            "slo": a.get("slo"),
            "weights": a.get("weights"),
            "tasks": len(a["routes"]),
            "distinct_configs": len(
                {d.space.key(r["config"]) for r in a["routes"].values()}
            ),
        }
        for n, a in tab["arms"].items()
    }


def plan() -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    base = run_base()
    tab = routing(base)
    pairs = d.pairs_of(tab)
    sf = {
        k: v
        for k, v in pairs.items()
        if json.loads(k).get("plan") in space.SEMFILTER_PLANS
    }
    total = sum(map(len, pairs.values()))
    done = done_pairs(base, pairs)
    return {
        "arms": len(tab["arms"]),
        "arm_names": sorted(tab["arms"]),
        "deployed_only": True,
        "outer_cv": False,
        "distinct_pairs": total,
        "configs": len(pairs),
        "reused_pairs": 0,
        "pairs_to_run": total,
        "pairs_to_run_semfilter": sum(map(len, sf.values())),
        "pairs_to_run_nomemory": len(pairs.get(space.key(space.NOMEM), [])),
        "already_done_here": done,
        "new_pairs_remaining": total - done,
        "per_arm": arm_summary(tab),
        "per_config_to_run": {
            f"{d.cfg_name(json.loads(k))} ({space.DISPLAY.get(json.loads(k)['plan'], 'NoMemory')})": len(
                v
            )
            for k, v in sorted(pairs.items(), key=lambda kv: -len(kv[1]))
        },
    }


def build_arms(base: Path, run: dict[str, list[str]], semfilter: dict | None) -> list:
    """One subset arm per configuration (largest first), quality reuse OFF (as ``alf_test_v7.build_arms``)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    baseline, _ = d.test_baseline(DS, argparse.Namespace(sampling="per-config"))
    exp = d.test_experiment(DS, ARM_TAG)
    out = base / SUBDIR / "pairs"
    arms = []
    for k, qs in sorted(run.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            arms.append(d.Arm("sci NoMemory pairs", out / d.cfg_name(cfg), qs, None))
            continue
        session = {
            **space.runner_args(cfg),
            "runner_extra": space.runner_cli(cfg),
            "split": "test",
            "experiment": exp,
            "baseline_dir": baseline,
            "panel": None,
            "reuse": False,  # quality reuse OFF: latency must be measured, never copied
            "reuse_from": [],
            "reuse_from_dirs": [],
            "protocol_extra": {
                "panel_hash": f"test:{d.TEST_VERSION}",
                "space_version": space.VERSION,
                "space_config": cfg,
                "bo_label": "test-pairs-c16-v7",
                "round": d.PANEL,
                "test_version": d.TEST_VERSION,
                "test_concurrency": CONCURRENCY,
                "retest": PROTOCOL,
                "catalog_from_round": CATALOG_FROM_ROUND,
                "semfilter": (semfilter or {}).get("protocol")
                if cfg["plan"] in space.SEMFILTER_PLANS
                else None,
            },
        }
        arms.append(d.Arm(f"sci {d.cfg_name(cfg)}", out / d.cfg_name(cfg), qs, session))
    return arms


def half_remaining(a, half: int) -> int:
    """``alf_test_c16.half_remaining`` for ScienceWorld result files."""
    from trimem.experiments import bo_pool_driver as d

    f = a.out.parent / "_shards" / a.out.name / f"{'ab'[half]}.json"
    if not f.exists():
        return 0
    want = {x["item_id"] for x in json.loads(f.read_text())}
    return len(want - set(d.drv.item_files(a.out, DS)))


class HalfMonitor(c16.HalfMonitor):
    """``alf_test_c16.HalfMonitor`` counting ScienceWorld result files (same starvation rule)."""

    def run(self) -> None:
        from trimem.experiments import bo_pool_driver as d

        names = ", ".join(a.name for a in self.arms)
        while not self.stop.wait(self.dt):
            try:
                run_ = self.running()
                rem = [sum(half_remaining(a, h) for a in self.arms) for h in (0, 1)]
            except Exception as exc:  # noqa: BLE001 -- the guard must never stop the run
                d.log(f"half monitor [{names}]: sample failed {exc!r}")
                continue
            d.log(
                f"half monitor [{names}]: running GPU0 {run_[0]:.0f} / GPU1 {run_[1]:.0f}; remaining half a "
                f"{rem[0]} / half b {rem[1]}"
            )
            bad = c16.starved_halves(run_, rem)
            self.strikes = self.strikes + 1 if bad else 0
            if self.strikes >= 2:
                d.log(
                    f"half monitor [{names}]: half {'ab'[bad[0]]} starved twice -> draining the batch so both "
                    "halves relaunch"
                )
                for a in self.arms:
                    d.drain_path(a).touch()
                self.strikes = 0


def run(dry: bool = False) -> None:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import memory_policy as mp

    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    for k in ("TRIMEM_SEMANTIC_ROUTING", "TRIMEM_SEMANTIC_BASES", "MAX_ENVS"):
        if os.environ.get(k):
            raise SystemExit(f"{k} is set: controlled-c16-v1 runs with the defaults")
    base = run_base()
    if not dry and (
        json.loads((base / "profile_complete.json").read_text()).get(DRY_MARK)
    ):
        raise SystemExit("this is a dry-run copy: run --dry only")
    state = json.loads((base / "state.json").read_text())
    if (state.get("semfilter") or {}).get("from_round") != 0:
        raise SystemExit(
            "state.json semfilter is not recorded from round 0 (round-0 SemFilter supersede missing)"
        )
    tab = routing(base)
    blk, env = v7.semfilter_env(base)
    os.environ.update(env)
    todo = d.pairs_of(tab)
    arms = build_arms(base, todo, blk)
    p = plan()
    (base / SUBDIR).mkdir(parents=True, exist_ok=True)
    rec = {
        "started": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL,
        "concurrency": CONCURRENCY,
        "layout": "one configuration at a time, 2 halves x 8 processes (reference layout); configurations "
        f"with < {c16.SMALL} pairs share one final batch (total <= {CONCURRENCY})",
        "runtime_profile": "controlled-c16-v1",
        "quality_reuse": False,
        "semfilter": {"protocol": blk["protocol"], "env": env},
        "retrieval_slots": os.environ.get("TRIMEM_RETRIEVAL_SLOTS") or "default (16)",
        "experiment": d.test_experiment(DS, ARM_TAG),
        "reuse_rule": "none (no earlier ScienceWorld test pairs in this run)",
        "dry": dry,
        **p,
    }
    v7.c16_save(base / SUBDIR / "plan.json", rec)
    with open(base / SUBDIR / "plan_history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    batches = c16.schedule(arms)
    d.log(
        f"sci v7 test (deployed only): {p['pairs_to_run']} pairs ({p['new_pairs_remaining']} not yet done) over "
        f"{len(arms)} configurations in {len(batches)} batches at concurrency {CONCURRENCY}, quality reuse off, "
        f"semfilter-v2 env -> {base / SUBDIR}"
    )
    if dry:
        return
    res: dict[str, int] = {}
    for b in batches:
        mon = HalfMonitor(b)
        mon.start()
        try:
            res |= d.run_parallel(DS, b, CONCURRENCY)
        finally:
            mon.stop.set()
            mon.join(timeout=5)
    bad = [a.name for a in arms if res.get(a.name, 0) < len(a.ids)]
    if bad:
        raise SystemExit(f"sci v7 test pairs incomplete: {bad}")
    d.log("V7 TEST DONE sciworld")


# ------------------------------------------------------------------ report


def memory_row(r: dict, spans: dict[str, float]) -> tuple[dict, dict | None]:
    """``alf_test_c16_report.memory_row`` with ScienceWorld steps. Pure."""
    from trimem.experiments import bo_pool_driver as d

    g = r.get("generation") or {}
    wall = d.with_steps(
        DS,
        {
            "quality": float(r["quality"]),
            "tokens": float(r["total_tokens"]),
            "e2e_ms": float(r["e2e_ms"]),
        },
        d.episode_steps(DS, g, None),
    )
    sp = spans.get(r["query_id"])
    return wall, c16.with_span(wall, sp, float(g.get("retrieval_ms") or 0.0))


def nomemory_dir_rows(
    pair_dir: Path, ids: list[str]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """``alf_test_c16_report.nomemory_dir_rows`` for ScienceWorld."""
    from trimem.experiments import bo_pool_driver as d

    rows = d.nomemory_rows(DS, pair_dir, ids)
    spans = c16.call_spans(pair_dir, DS)
    strip = lambda r: {k: v for k, v in r.items() if k != "cost_source"}  # noqa: E731
    wall = {q: strip(r) for q, r in rows.items() if r.get("cost_source") == "episodes"}
    span = {
        q: c16.with_span(strip(r), spans[q], 0.0) for q, r in rows.items() if q in spans
    }
    return wall, span


def pair_rows(base: Path, tab: dict) -> tuple[dict, dict, dict[str, int]]:
    """(config key, task) -> row in the wall and span definitions (this test's pairs only) + source counts."""
    from trimem.experiments import alf_test_c16_report as c16r
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    want = {(k, q) for k, qs in d.pairs_of(tab).items() for q in qs}
    pdir = base / SUBDIR / "pairs"
    wall: dict = {}
    span: dict = {}
    spans: dict[str, dict[str, float]] = {}
    for r in c16r._rows_of_opt(d.test_experiment(DS, ARM_TAG)):
        if not r["cfg"]:
            continue
        k = space.key(r["cfg"])
        if (k, r["query_id"]) not in want:
            continue
        name = d.cfg_name(r["cfg"])
        if name not in spans:
            spans[name] = c16.call_spans(pdir / name, DS)
        w, s = memory_row(r, spans[name])
        wall[(k, r["query_id"])] = w
        if s is not None:
            span[(k, r["query_id"])] = s
    nk = space.key(space.NOMEM)
    nm_q = [q for k, q in want if k == nk]
    if nm_q:
        w, s = nomemory_dir_rows(pdir / d.cfg_name(space.NOMEM), nm_q)
        wall |= {(nk, q): r for q, r in w.items()}
        span |= {(nk, q): r for q, r in s.items() if q in w}
    return wall, span, {"new": len(wall), "reused": 0}


def routing_shares(routes: dict[str, dict], rows: dict, key) -> list[dict]:
    """Per configuration of one deployed arm: routed tasks, share, plan, and the mean quality of the routed tasks
    that have a result (pure)."""
    from trimem.experiments import bo_pool_driver as d

    by: dict[str, list[str]] = {}
    for q, r in routes.items():
        by.setdefault(key(r["config"]), []).append(q)
    n = max(len(routes), 1)
    out = []
    for k, qs in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
        got = [rows[(k, q)]["quality"] for q in qs if (k, q) in rows]
        out.append(
            {
                "config": d.cfg_name(cfg),
                "plan": cfg["plan"],
                "n": len(qs),
                "share": len(qs) / n,
                "quality": sum(got) / len(got) if got else None,
                "covered": len(got),
            }
        )
    return out


def plan_shares(shares: list[dict]) -> dict[str, float]:
    """Routing share per plan (pure)."""
    out: dict[str, float] = {}
    for s in shares:
        out[s["plan"]] = out.get(s["plan"], 0.0) + s["share"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def references(ids: list[str], nm_old: dict[str, float]) -> dict:
    """Controlled-v1 originals: NoMemory (span latency only) and VGR (wall + span)."""
    from trimem.experiments import alf_test_c16_report as c16r
    from trimem.experiments import bo_pool_driver as d

    base_q = {"old": nm_old}
    _, nm_s = nomemory_dir_rows(d.drv.CTRL / f"{DS}_NoMemory", ids)
    vdir = d.drv.CTRL / f"{DS}_VGR"
    spans = c16.call_spans(vdir, DS)
    files = d.drv.item_files(vdir, DS)
    vw, vs = {}, {}
    for r in c16r._rows_of_opt(REF_VGR_EXPERIMENT):
        if r["query_id"] not in set(ids):
            continue
        g = r.get("generation") or {}
        if g and "environment_steps" not in g:
            f = files.get(r["query_id"])
            r = {**r, "generation": {**g, "result_file": str(f) if f else None}}
        a, b = memory_row(r, spans)
        vw[r["query_id"]] = a
        if b is not None:
            vs[r["query_id"]] = b
    return {
        "NoMemory": c16r.ref_block({}, nm_s, ids, base_q)
        | {
            "note": "controlled-v1 2026-10-15; no episodes.jsonl: LLM-call span latency only"
        },
        "VGR": c16r.ref_block(vw, vs, ids, base_q)
        | {"note": f"controlled-v1 2026-10-15 ({REF_VGR_EXPERIMENT})"},
    }


def report() -> dict:
    from trimem.experiments import bo_pool_driver as d

    base = run_base()
    tab = routing(base)
    ids = d.test_ids(DS)
    ctrl_nm = d.drv.CTRL / f"{DS}_NoMemory"
    nm = {
        i: d.quality_of(DS, json.loads(f.read_text())["reward"])
        for i, f in d.drv.item_files(ctrl_nm, DS).items()
    }
    wall, span, src = pair_rows(base, tab)
    arms_w, arms_s = d.assemble(tab, wall, nm), d.assemble(tab, span, nm)
    arms = {}
    for k, v in arms_w.items():
        sh = routing_shares(tab["arms"][k]["routes"], wall, d.space.key)
        arms[k] = {
            **v,
            "latency_span_s": arms_s[k].get("latency_s"),
            "routing": sh,
            "plan_shares": plan_shares(sh),
        }
    sel = json.loads((base / "selection.json").read_text())
    pl = base / SUBDIR / "plan.json"
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL,
        "dataset": DS,
        "experiment": d.test_experiment(DS, ARM_TAG),
        "plan": json.loads(pl.read_text()) if pl.exists() else None,
        "catalog": sel.get("catalog"),
        "replay_estimator": sel.get("replay_estimator"),
        "deployed_only": True,
        "outer_cv": None,
        "concurrency": CONCURRENCY,
        "runtime_profile": "controlled-c16-v1",
        "quality_reuse": False,
        "latency_definitions": {
            "latency_s": "wall: official episode wall time + retrieval ms + router overhead (opt e2e_ms)",
            "latency_span_s": "span: first .. last LLM call of the episode (llm_calls.jsonl) + retrieval ms + "
            "router overhead (= the controlled-v1 NoMemory reference's definition)",
        },
        "baseline": {
            "dir": str(ctrl_nm),
            "label": "controlled-v1 NoMemory (paired baseline of gain)",
            "quality": sum(nm.values()) / max(len(nm), 1),
        },
        "pairs": len(wall),
        "pairs_span": len(span),
        "pair_sources": src,
        "complete": all(v.get("covered") == v.get("n") for v in arms.values()),
        "arms": arms,
        "reference_arms": references(ids, nm),
        "selection": {
            n: {
                "oneconfig": (s.get("opt_oneconfig") or {}).get("label"),
                "perquery_slo_train_only": (s.get("opt_perquery") or {}).get("weights"),
            }
            for n, s in sel["selections"].items()
        },
    }
    v7.c16_save(base / REPORT, rep)
    (base / REPORT_MD).write_text("\n".join(markdown(rep)) + "\n")
    return rep


def _f(x, fmt: str) -> str:
    return "–" if x is None else format(x, fmt)


def _pm(x, se, fmt: str) -> str:
    return (
        "–"
        if x is None
        else format(x, fmt)
        + ("" if se is None else " ± " + format(se, fmt.lstrip("+")))
    )


def markdown(rep: dict) -> list[str]:
    """report_v7c16.md: deployed arms, references, routing shares (pure)."""
    cat = rep.get("catalog") or {}
    rounds = cat.get("included_rounds") or ["?"]
    L = [
        "# ScienceWorld v7 test — deployed arms only (catalog rounds >= 0, semfilter-v2), concurrency 16",
        "",
        (
            f"Generated {rep['generated']}. Catalog: rounds {rounds[0]}–{rounds[-1]} ({cat.get('full_configs')} "
            f"fully evaluated configurations). Replay estimator {rep.get('replay_estimator')}. Selection / router "
            f"fitted on the full train split; no outer CV. {rep['pairs']} (config, task) pairs, quality reuse OFF, "
            f"controlled-c16-v1 runtime. Complete: {rep['complete']}."
        ),
        "",
        "## Deployed arms (200 test tasks × 1 run; ± = SE across tasks)",
        "",
        (
            "| arm | quality ± SE | gain vs NoMemory ± SE | tokens | latency wall s | latency span s | steps ± SE | "
            "steps (succ.) | configs | plan shares |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for n, x in rep["arms"].items():
        shares = ", ".join(
            f"{p} {100 * s:.0f}%" for p, s in (x.get("plan_shares") or {}).items()
        )
        L.append(
            f"| {n} | {_pm(x.get('quality'), x.get('quality_se'), '.3f')} | "
            f"{_pm(x.get('paired_gain_vs_nomemory'), x.get('paired_gain_se'), '+.3f')} | "
            f"{_f(x.get('tokens'), ',.0f')} | {_f(x.get('latency_s'), '.1f')} | "
            f"{_f(x.get('latency_span_s'), '.1f')} | {_pm(x.get('steps'), x.get('steps_se'), '.2f')} | "
            f"{_f(x.get('steps_success'), '.2f')} | {x.get('distinct_configs')} | {shares} |"
        )
    L += [
        "",
        "## References (controlled-v1 originals, single runs, 200 tasks)",
        "",
        "| arm | quality ± SE | tokens | latency wall s | latency span s | steps | gain vs NoMemory |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n, r in rep["reference_arms"].items():
        w, s = r.get("wall"), r.get("span")
        m = w or s
        if m:
            L.append(
                f"| {n} | {_f(m['quality'], '.3f')} ± {_f(m['quality_se'], '.3f')} | {_f(m['tokens'], ',.0f')} | "
                f"{_f(w and w['latency_s'], '.1f')} | {_f(s and s['latency_s'], '.1f')} | {_f(m['steps'], '.2f')} | "
                f"{_f(m['paired_gain_vs_nomemory'], '+.3f')} |"
            )
    L += [
        "",
        "## Routing shares per deployed arm (configuration: tasks, share, mean quality of those tasks)",
        "",
    ]
    for n, x in rep["arms"].items():
        L.append(
            f"- **{n}**: "
            + "; ".join(
                f"{r['config']} ({r['plan']}) {r['n']} ({100 * r['share']:.0f}%, q {_f(r['quality'], '.3f')})"
                for r in x.get("routing") or []
            )
        )
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("dry_copy", "offline", "plan", "run", "report"))
    ap.add_argument(
        "--out",
        default=os.environ.get("SCI_V7_OUT") or None,
        help="results root of a dry-run COPY (env SCI_V7_OUT); default: the real results root",
    )
    ap.add_argument(
        "--last_round", type=int, default=None, help="dry_copy: keep rounds <= this"
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="offline: router fit jobs (driver default 8)",
    )
    ap.add_argument(
        "--dry", action="store_true", help="run: write plan.json and log, start nothing"
    )
    a = ap.parse_args(argv)
    if a.stage == "dry_copy":
        if not a.out or a.last_round is None:
            raise SystemExit("dry_copy needs --out and --last_round")
        print(json.dumps(dry_copy(a.out, a.last_round), indent=1))
        return 0
    redirect(a.out)
    if a.stage == "offline":
        tab = offline(a.jobs)
        print(json.dumps(arm_summary(tab), indent=1))
    elif a.stage == "plan":
        print(json.dumps(plan(), indent=1))
    elif a.stage == "run":
        run(a.dry)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
