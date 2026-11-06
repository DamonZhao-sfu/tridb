"""WebShop pool-strat200-v7 TEST after profiling: DEPLOYED arms only (user 2026-10-23, latest; ``OUTER_CV`` False).

User 2026-10-23 (latest): no 5-fold router for WebShop for now -- only the deployed arms (as sci_test_v7); the
outer-CV path described below stays implemented behind ``OUTER_CV`` (protocol id webshop-v7-test-deployed-cv-v1).

Profiling (``results/pool_v5_20261020/run_webshop_v7.sh`` -> ``bo_pool_driver profile --dataset webshop``) pauses
after round 5 (50 configurations: round 0 = 10 Sobol over the 6 plans incl. P1 (round0-design-v1), round 1 = 10
Sobol (sobol-only-v1), rounds 2-4 = channel-slots-v6 x 8, round 5 = 6; 100 stratified train tasks per
configuration from round 0, stages 50 -> 100; controlled-c16-v1 and semfilter-v2 from round 0).

* catalog (``bo_pool_driver`` catalog-regime-v1, POOL_CATALOG_FROM_ROUND=0): every round, fully evaluated
  configurations only; NoMemory candidate / train gain anchor = this run's own NoMemory (c16, 200 tasks).
* router: BRANE only (``--no_limeqo``), zscore-log-v1 utility, self-normalised replay (selfnorm-v1): the defaults.
* arms (user 2026-10-23, the ALF v7 style -- not Sci's deployed-only variant): the DEPLOYED arms (selection / router
  fitted on all training queries: Opt-OneConfig SLO-1/2/3 + Opt-PerQuery at the 4 TEST_WEIGHTS) AND the outer
  5-fold CV arms (``bo_pool_driver.cv_stage``: folds over the union of the profiled training queries, stratified
  by the pool stratum; per fold the per-config predictors train on own-sample rows of the other 4 folds, the
  held-out fold gives validation diagnostics), every fold's choices evaluated on the same 200 test tasks (mean ± std
  across folds). ``offline`` freezes ``test/routing.json`` + ``test/routing_cv.json`` before any test episode.
* test set = the 200 AgentEval WebShop items (``bo_pool_driver.test_ids``: results/eval_controlled_20261015/shards/
  webshop_a/b.json). Train pool = all 3,930 stored train goals: train and test SHARE products (user 2026-10-23
  option B; no test goal is in the pool); retrieval keeps product-level isolation per query, runtime = controlled-c16-v1 (``alf_test_c16`` layout: one
  configuration at a time, 2 halves x 8 processes on env 36003 / 36013, small configurations share one final
  batch, both-halves monitor), quality reuse OFF, the state.json semfilter-v2 knobs exported to every runner.
* pairs run into ``test_v7c16/pairs`` / opt experiment ``webshop:...:pairs-c16-v7:...`` (the names the fixed-plan
  arms and the PerQuery sweep reuse from: ``fixed_plans_test`` / ``perquery_sweep`` source ``v7_test``).
* references: the controlled-v1 originals (results/eval_controlled_20261015/webshop_NoMemory: same 200 items, 2 x 8
  processes = c16, LLM-call span latency only; webshop_VGR: opt experiment ``webshop:agenteval200:controlled-v1:VGR``,
  success-only-v0 era); paired gain vs the controlled-v1 NoMemory.

Stages: ``offline`` (router -> select -> cv -> frozen routing tables; CPU only, skips what is frozen), ``plan``
(pair counts deployed + CV; nothing runs), ``run`` (``--dry``: plan.json + log only), ``report``
(test_v7c16_report.json + report_v7c16.md).

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 POOL_SEMFILTER=v2 POOL_CATALOG_FROM_ROUND=0 \\
        .venv-optimizer/bin/python -m trimem.experiments.webshop_test_v7 {offline|plan|run|report} [--dry]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import alf_test_c16 as c16
from trimem.experiments import alf_test_v7 as v7

DS = "webshop"
CONCURRENCY = c16.CONCURRENCY  # 16 = 2 halves x 8 (controlled-c16-v1)
SUBDIR = v7.SUBDIR  # test_v7c16
ARM_TAG = v7.ARM_TAG  # pairs-c16-v7 -> opt experiment webshop:...:pairs-c16-v7:...
REPORT = v7.REPORT
REPORT_MD = v7.REPORT_MD
# user 2026-10-23 (latest): NO outer 5-fold CV for WebShop for now -- deployed arms only (as sci_test_v7). The CV
# path stays implemented behind OUTER_CV (its protocol id differs, so a later CV test never mixes with this one).
OUTER_CV = False
PROTOCOL = (
    "webshop-v7-test-deployed-cv-v1" if OUTER_CV else "webshop-v7-test-deployed-v1"
)
CATALOG_FROM_ROUND = 0
RUNTIME = "controlled-c16-v1"
REF_VGR_EXPERIMENT = f"{DS}:agenteval200:controlled-v1:VGR"


def run_base() -> Path:
    from trimem.experiments import bo_pool_driver as d

    return d.run_dir(DS)


def driver_args(stage: str = "router", jobs: int | None = None) -> argparse.Namespace:
    from trimem.experiments import bo_pool_driver as d

    argv = [stage, "--dataset", DS, "--no_limeqo"]
    return d.build_parser().parse_args(argv + (["--jobs", str(jobs)] if jobs else []))


# ------------------------------------------------------------------ offline (router -> select -> cv -> routing)


def offline(jobs: int | None = None) -> dict:
    """Router (BRANE, catalog rounds >= 0) -> select -> outer 5-fold CV (freezes test/routing.json AND
    test/routing_cv.json). Skips what is frozen. Holds the dataset's driver lock."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import memory_policy as mp

    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    base = run_base()
    if not (base / "profile_complete.json").exists():
        raise SystemExit(
            f"profiling not paused / complete ({base / 'profile_complete.json'} missing)"
        )
    if (base / "test" / "routing.json").exists() and (
        not OUTER_CV or (base / "test" / "routing_cv.json").exists()
    ):
        tab, cv = routing(base)
        d.log(
            f"offline: routing tables already frozen ({len(tab['arms'])} deployed + "
            f"{len(cv['arms']) if cv else 0} CV arms)"
        )
        return {"deployed": tab, "cv": cv}
    (d.OUT / DS).mkdir(parents=True, exist_ok=True)
    lock = open(d.OUT / DS / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another driver holds {d.OUT / DS / 'driver.lock'}") from None
    a = driver_args("router", jobs)
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
    if OUTER_CV:
        d.log(
            "offline: outer 5-fold CV (freezes the deployed routing table and the CV routing table)"
        )
        d.cv_stage(DS, a)
    else:
        d.log(
            "offline: freezing the deployed routing table (deployed arms only, no outer CV)"
        )
        d.routing_table(DS, a, sel)
    tab, cv = routing(base)
    return {"deployed": tab, "cv": cv}


# ------------------------------------------------------------------ routing + pairs


def routing(base: Path) -> tuple[dict, dict | None]:
    """The catalog-regime (rounds >= 0) deployed routing table (+ the CV table when OUTER_CV, else None); refuses
    unlabelled / LimeQO / non-deployed ones."""
    tab = json.loads((base / "test" / "routing.json").read_text())
    cv = (
        json.loads((base / "test" / "routing_cv.json").read_text())
        if OUTER_CV
        else None
    )
    if not OUTER_CV:
        bad = [
            n
            for n, a in tab["arms"].items()
            if a.get("kind") not in ("oneconfig", "perquery")
        ]
        if bad:
            raise SystemExit(f"test/routing.json has non-deployed arms {bad}")
    for name, t in (("routing.json", tab), ("routing_cv.json", cv)):
        if t is None:
            continue
        if (t.get("catalog") or {}).get("from_round") != CATALOG_FROM_ROUND:
            raise SystemExit(
                f"test/{name} is not the catalog-regime (rounds >= {CATALOG_FROM_ROUND}) table"
            )
        if any(c16.is_excluded(n, a) for n, a in t["arms"].items()):
            raise SystemExit(f"test/{name} has LimeQO arms; this test is BRANE only")
    return tab, cv


def pair_set(
    tab: dict, cv: dict | None, deployed: bool = True, folds: bool = True
) -> dict[str, list[str]]:
    from trimem.experiments import bo_pool_driver as d

    arms = {
        **(tab["arms"] if deployed else {}),
        **(cv["arms"] if folds and cv else {}),
    }
    return d.pairs_of({"arms": arms})


def done_pairs(base: Path, run: dict[str, list[str]]) -> int:
    from trimem.experiments import bo_pool_driver as d

    n = 0
    for k, qs in run.items():
        have = d.drv.item_files(base / SUBDIR / "pairs" / d.cfg_name(json.loads(k)), DS)
        n += len(set(qs) & set(have))
    return n


def counts(pairs: dict[str, list[str]], base: Path | None) -> dict:
    """Pair counts of one pair set (pure but for the done count on disk)."""
    from trimem.optimizer import session_space_agentenv_v2 as space

    total = sum(map(len, pairs.values()))
    done = done_pairs(base, pairs) if base is not None else 0
    return {
        "distinct_pairs": total,
        "configs": len(pairs),
        "pairs_semfilter": sum(
            len(v)
            for k, v in pairs.items()
            if json.loads(k).get("plan") in space.SEMFILTER_PLANS
        ),
        "pairs_nomemory": len(pairs.get(space.key(space.NOMEM), [])),
        "already_done_here": done,
        "new_pairs_remaining": total - done,
    }


def plan() -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    base = run_base()
    tab, cv = routing(base)
    allp = pair_set(tab, cv)
    return {
        "arms_deployed": len(tab["arms"]),
        "outer_cv": OUTER_CV,
        "arms_cv": len(cv["arms"]) if cv else 0,
        "folds": cv.get("folds") if cv else None,
        "with_deployed": counts(allp, base),
        "deployed_only": counts(pair_set(tab, cv, folds=False), None),
        "cv_only": counts(pair_set(tab, cv, deployed=False), None) if cv else None,
        "per_config_to_run": {
            f"{d.cfg_name(json.loads(k))} ({space.DISPLAY.get(json.loads(k)['plan'], 'NoMemory')})": len(
                v
            )
            for k, v in sorted(allp.items(), key=lambda kv: -len(kv[1]))
        },
    }


def build_arms(base: Path, run: dict[str, list[str]], semfilter: dict | None) -> list:
    """One subset arm per configuration (largest first), quality reuse OFF (as ``alf_test_v7.build_arms``)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    baseline, _ = d.test_baseline(DS, argparse.Namespace(sampling="per-config"))
    exp = d.test_experiment(DS, ARM_TAG)
    out = base / SUBDIR / "pairs"
    mr = d.test_max_round(
        DS
    )  # max-round-v1: the profiling run's recorded turn budget (None = 30)
    arms = []
    for k, qs in sorted(run.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            arms.append(
                d.Arm(
                    "ws NoMemory pairs", out / d.cfg_name(cfg), qs, None, max_round=mr
                )
            )
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
                **d.eb.protocol_extra(mr),
            },
        }
        arms.append(
            d.Arm(
                f"ws {d.cfg_name(cfg)}",
                out / d.cfg_name(cfg),
                qs,
                session,
                max_round=mr,
            )
        )
    return arms


def half_remaining(a, half: int) -> int:
    """``alf_test_c16.half_remaining`` for WebShop result files."""
    from trimem.experiments import bo_pool_driver as d

    f = a.out.parent / "_shards" / a.out.name / f"{'ab'[half]}.json"
    if not f.exists():
        return 0
    want = {x["item_id"] for x in json.loads(f.read_text())}
    return len(want - set(d.drv.item_files(a.out, DS)))


class HalfMonitor(c16.HalfMonitor):
    """``alf_test_c16.HalfMonitor`` counting WebShop result files (same starvation rule). Also used by
    ``fixed_plans_test.monitor_cls('webshop')`` (fixed-plan arms and the PerQuery sweep)."""

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
    state = json.loads((base / "state.json").read_text())
    if (state.get("semfilter") or {}).get("from_round") != 0:
        raise SystemExit("state.json semfilter is not recorded from round 0")
    tab, cv = routing(base)
    blk, env = v7.semfilter_env(base)
    os.environ.update(env)
    todo = pair_set(tab, cv)
    arms = build_arms(base, todo, blk)
    p = plan()
    (base / SUBDIR).mkdir(parents=True, exist_ok=True)
    rec = {
        "started": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL,
        "concurrency": CONCURRENCY,
        "layout": "one configuration at a time, 2 halves x 8 processes (reference layout); configurations "
        f"with < {c16.SMALL} pairs share one final batch (total <= {CONCURRENCY})",
        "runtime_profile": d.eb.runtime_label(state, RUNTIME) or RUNTIME,
        "episode_budget": d.eb.block(state),
        "quality_reuse": False,
        "semfilter": {"protocol": blk["protocol"], "env": env},
        "retrieval_slots": os.environ.get("TRIMEM_RETRIEVAL_SLOTS") or "default (16)",
        "experiment": d.test_experiment(DS, ARM_TAG),
        "reuse_rule": "none (no earlier WebShop test pairs of this run)",
        "deployed_and_cv": OUTER_CV,
        "dry": dry,
        **p,
    }
    v7.c16_save(base / SUBDIR / "plan.json", rec)
    with open(base / SUBDIR / "plan_history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    batches = c16.schedule(arms)
    w = p["with_deployed"]
    d.log(
        f"webshop v7 test ({'deployed + 5-fold CV' if OUTER_CV else 'deployed arms only'}): {w['distinct_pairs']} pairs ({w['new_pairs_remaining']} not yet "
        f"done) over {len(arms)} configurations in {len(batches)} batches at concurrency {CONCURRENCY}, quality "
        f"reuse off, semfilter-v2 env -> {base / SUBDIR}"
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
        raise SystemExit(f"webshop v7 test pairs incomplete: {bad}")
    d.log("V7 TEST DONE webshop")


# ------------------------------------------------------------------ report


def memory_row(r: dict, spans: dict[str, float]) -> tuple[dict, dict | None]:
    """``alf_test_c16_report.memory_row`` with WebShop steps. Pure."""
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
    """``alf_test_c16_report.nomemory_dir_rows`` for WebShop; max-round-v1: every episode longer than the run's turn
    budget in the derived-truncation view (rows and spans)."""
    from trimem.experiments import bo_pool_driver as d

    mr = d.test_max_round(DS)
    rows = d.nomemory_rows(DS, pair_dir, ids, max_round=mr)
    spans = c16.call_spans(pair_dir, DS)
    if (
        mr is not None
    ):  # span of a truncated episode = its first mr calls (its last worker)
        spans |= d.eb.truncated_spans(pair_dir, DS, mr)
    strip = lambda r: {k: v for k, v in r.items() if k != "cost_source"}  # noqa: E731
    wall = {q: strip(r) for q, r in rows.items() if r.get("cost_source") == "episodes"}
    span = {
        q: c16.with_span(strip(r), spans[q], 0.0) for q, r in rows.items() if q in spans
    }
    return wall, span


def pair_rows(base: Path, tab: dict, cv: dict) -> tuple[dict, dict, dict[str, int]]:
    from trimem.experiments import alf_test_c16_report as c16r
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    want = {(k, q) for k, qs in pair_set(tab, cv).items() for q in qs}
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


def references(ids: list[str], nm_old: dict[str, float]) -> dict:
    """Controlled-v1 originals: NoMemory (span latency only) and VGR (wall + span)."""
    from trimem.experiments import alf_test_c16_report as c16r
    from trimem.experiments import bo_pool_driver as d

    base_q = {"old": nm_old}
    _, nm_s = nomemory_dir_rows(d.drv.CTRL / f"{DS}_NoMemory", ids)
    vdir = d.drv.CTRL / f"{DS}_VGR"
    mr = d.test_max_round(
        DS
    )  # max-round-v1: the 30-turn references in the derived-truncation view
    spans = c16.call_spans(vdir, DS)
    if mr is not None:
        spans |= d.eb.truncated_spans(vdir, DS, mr)
    files = d.drv.item_files(vdir, DS)
    vw, vs = {}, {}
    for r in c16r._rows_of_opt(REF_VGR_EXPERIMENT):
        if r["query_id"] not in set(ids):
            continue
        g = r.get("generation") or {}
        if g and "environment_steps" not in g:
            f = files.get(r["query_id"])
            r = {**r, "generation": {**g, "result_file": str(f) if f else None}}
        if mr is not None:
            r = d.eb.derive_opt_row(r, mr)
        a, b = memory_row(r, spans)
        vw[r["query_id"]] = a
        if b is not None:
            vs[r["query_id"]] = b
    return {
        "NoMemory": c16r.ref_block({}, nm_s, ids, base_q)
        | {
            "note": "controlled-v1 2026-10-18 (2 x 8 = c16); no episodes.jsonl: LLM-call span latency only"
        },
        "VGR": c16r.ref_block(vw, vs, ids, base_q)
        | {
            "note": f"controlled-v1 2026-10-18 ({REF_VGR_EXPERIMENT}; success-only-v0, seed 8 / depth 1 / "
            "fanout 4 / top-k 3)"
        },
    }


def report() -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import router_outer_cv as rcv
    from trimem.experiments import sci_test_v7 as stv

    base = run_base()
    tab, cv = routing(base)
    ids = d.test_ids(DS)
    ctrl_nm = d.drv.CTRL / f"{DS}_NoMemory"
    mr = d.test_max_round(DS)
    nm = {
        i: d.eb.view_quality(DS, json.loads(f.read_text()), mr)
        for i, f in d.drv.item_files(ctrl_nm, DS).items()
    }
    wall, span, src = pair_rows(base, tab, cv)

    def fold_metrics(rows):
        def f(t: dict) -> dict:
            got = d.assemble(t, rows, nm)
            return {k: v for k, v in got.items() if v.get("covered") == v.get("n")}

        return f

    arms_w, arms_s = d.assemble(tab, wall, nm), d.assemble(tab, span, nm)
    arms = {}
    for k, v in arms_w.items():
        sh = stv.routing_shares(tab["arms"][k]["routes"], wall, d.space.key)
        arms[k] = {
            **v,
            "latency_span_s": arms_s[k].get("latency_s"),
            "routing": sh,
            "plan_shares": stv.plan_shares(sh),
        }
    outer = None
    incomplete: list[str] = []
    if cv is not None:
        ts_w = rcv.test_summary(cv, fold_metrics(wall))
        ts_s = rcv.test_summary(cv, fold_metrics(span))
        for k, v in ts_w["summary"].items():
            v["latency_span_s"] = ts_s["summary"].get(k, {}).get("latency_s")
        incomplete = sorted(
            n
            for n, v in d.assemble(cv, wall, nm).items()
            if v.get("covered") != v.get("n")
        )
        outer = {
            "protocol": cv["protocol"],
            "folds": cv["folds"],
            "test": {**ts_w, "per_fold_span": ts_s["per_fold"]},
            "incomplete_test_arms": incomplete,
            "labels": {
                "mean_std": "mean ± sample std (ddof 1) across the 5 outer folds",
                "se_tasks_mean": "mean within-fold SE across tasks (task-sampling noise), NOT across folds",
            },
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
        "deployed_only": cv is None,
        "concurrency": CONCURRENCY,
        "runtime_profile": d.eb.runtime_label(
            json.loads((base / "state.json").read_text()), RUNTIME
        )
        or RUNTIME,
        "episode_budget": d.eb.block(json.loads((base / "state.json").read_text())),
        "quality_reuse": False,
        "quality": "reward (0-1; success = reward 1)",
        "train_test_share_products": pool_provenance().get("train_test_share_products"),
        "pool_provenance": pool_provenance(),
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
        "complete": not incomplete
        and all(v.get("covered") == v.get("n") for v in arms.values()),
        "arms": arms,
        "outer_cv": outer,
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


def pool_provenance() -> dict:
    """state.json ``pool_provenance`` (bo_pool_driver; option B), else the pool file's fields."""
    from trimem.experiments import bo_pool_driver as d

    st = json.loads((run_base() / "state.json").read_text())
    if st.get("pool_provenance"):
        return st["pool_provenance"]
    p = d.load_pool(DS)
    return {
        k: p.get(k)
        for k in (
            "scope",
            "size",
            "pool_hash",
            "train_test_share_products",
            "product_rule",
        )
    }


def markdown(rep: dict) -> list[str]:
    """report_v7c16.md: deployed arms with routing shares, outer-CV test summary, references (pure)."""
    from trimem.experiments import sci_test_v7 as stv

    deployed = stv.markdown({**rep, "arms": rep["arms"]})
    with_cv = bool(rep.get("outer_cv"))
    deployed[0] = (
        "# WebShop v7 test — deployed + outer 5-fold CV (catalog rounds >= 0, semfilter-v2), concurrency 16"
        if with_cv
        else "# WebShop v7 test — deployed arms only (catalog rounds >= 0, semfilter-v2), concurrency 16"
    )
    if with_cv:
        deployed = [
            x.replace(
                "Selection / router fitted on the full train split; no outer CV.",
                "Deployed selection / router fitted on the full train split + outer 5-fold CV arms.",
            )
            for x in deployed
        ]
    share = rep.get("train_test_share_products")
    if share:
        deployed[3:3] = [
            f"Train / test share products: {share} (train pool = all stored train goals; no test goal is in the "
            "pool; retrieval excludes each query's own product).",
            "",
        ]
    if not with_cv:
        return deployed
    cv = v7.markdown(rep)
    i = cv.index(
        "## Outer 5-fold CV — test set (mean ± std across folds; SE_tasks = mean within-fold task SE)"
    )
    j = cv.index("## References (controlled-v1 originals, single runs, 200 tasks)")
    return deployed + [""] + cv[i:j]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("offline", "plan", "run", "report"))
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
    if a.stage == "offline":
        out = offline(a.jobs)
        print(
            json.dumps(
                {
                    "deployed": sorted(out["deployed"]["arms"]),
                    "cv": len(out["cv"]["arms"]) if out["cv"] else 0,
                },
                indent=1,
            )
        )
    elif a.stage == "plan":
        print(json.dumps(plan(), indent=1))
    elif a.stage == "run":
        run(a.dry)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
