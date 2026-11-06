"""ALFWorld pool-strat200-v7 RE-TEST on the latest profiled configurations (user decision 2026-10-22).

Profiling finished at round 21 (105 configurations). The previous test (``alf_test_c16``) selected from rounds 0-5
only, whose latency ran at c128 and whose SemFilter plans ran the v1 judge. This re-test:

* catalog (``bo_pool_driver`` catalog-regime-v1, recorded in state.json): rounds >= 6 only (controlled-c16-v1
  runtime, semfilter-v2 for SemFilter plans, 100 stratified train tasks per configuration), fully evaluated
  configurations only (pruned partial samples never enter); NoMemory router candidate / gain anchor = the c16
  NoMemory rerun (latency-regime-v1). Router: BRANE only (no LimeQO), zscore-log-v1 utility, self-normalised
  replay (``sparse.REPLAY_ESTIMATOR`` selfnorm-v1). Opt-OneConfig SLO-1/2/3 + Opt-PerQuery at the 4 TEST_WEIGHTS,
  deployed (full train) + outer 5-fold CV (train only; every fold's choices are evaluated on the same 200 test
  tasks).
* test runtime = ``alf_test_c16``'s (controlled-c16-v1: one configuration at a time, 2 halves x 8 processes,
  smaller configurations share one final batch, both-halves monitor), quality reuse OFF, the state.json
  semfilter-v2 knobs exported to every runner (``bo_pool_driver.check_semfilter``; inert for non-SemFilter plans,
  exactly as in profiling rounds >= 6).
* pair reuse: a (configuration, task) pair already measured in the c16 test rerun (``test_c16``, opt experiment
  ``...:pairs-c16:...``, evaluated rows; NoMemory pair files with an episodes.jsonl cost record) is reused ONLY
  when its plan uses no SemFilter (same runtime, no judge-version difference); SemFilter pairs of that run
  (semfilter-v1) are never reused. New pairs run into ``test_v7c16/pairs`` / ``...:pairs-c16-v7:...``.
* references (unchanged): the controlled-v1 originals (NoMemory: LLM-call span latency; VGR: wall + span);
  paired gain vs the controlled-v1 NoMemory.

Stages: ``archive`` (rename the rounds 0-5 selection / router / test artifacts to ``*_v7r0-5_<ts>``, never
delete; manifest ``archive_v7r0-5.json``), ``plan`` (pair counts; nothing runs), ``run``, ``report``.

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 POOL_SEMFILTER=v2 .venv-optimizer/bin/python -m \
        trimem.experiments.alf_test_v7 {archive|plan|run|report}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import alf_test_c16 as c16

DS = "alfworld"
CONCURRENCY = c16.CONCURRENCY
SUBDIR = "test_v7c16"
ARM_TAG = "pairs-c16-v7"
REPORT = "test_v7c16_report.json"
REPORT_MD = "report_v7c16.md"
PROTOCOL = "alf-v7-retest-v1"
CATALOG_FROM_ROUND = 6
ARCHIVE_TAG = "v7r0-5"
MANIFEST = c16.MANIFEST
# rounds 0-5 artifacts that the re-selection would overwrite or that describe the old selection (relative paths)
ARCHIVE_ITEMS = (
    "router",
    "router_cv",
    "router_cv_limeqo",
    "selection.json",
    "test/routing.json",
    "test/routing_cv.json",
    "test_report.json",
    "report.md",
    "report_cv.md",
    "test_c16_report.json",
    "report_c16.md",
)
FEATURE_CACHES = ("features_train.json", "features_test.json")


def run_base() -> Path:
    return c16.run_base()


# ------------------------------------------------------------------ archive


def archived_name(rel: str, ts: str) -> str:
    """``selection.json`` -> ``selection_v7r0-5_<ts>.json``; ``router`` -> ``router_v7r0-5_<ts>``. Pure."""
    p = Path(rel)
    tag = f"_{ARCHIVE_TAG}_{ts}"
    return str(p.with_name(p.stem + tag + p.suffix)) if p.suffix else rel + tag


def archive(base: Path, ts: str | None = None) -> dict:
    """Rename the ARCHIVE_ITEMS present to their archived names (same directory; never deleted) and seed the new
    ``router/`` with the archived feature caches (per-query records, identical inputs). Idempotent: an existing
    manifest is returned unchanged."""
    m = base / MANIFEST
    if m.exists():
        return json.loads(m.read_text())
    ts = ts or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    moved = {}
    for rel in ARCHIVE_ITEMS:
        src = base / rel
        if not src.exists():
            continue
        dst = base / archived_name(rel, ts)
        if dst.exists():
            raise SystemExit(f"archive target exists: {dst}")
        src.rename(dst)
        moved[rel] = archived_name(rel, ts)
    seeded = []
    if "router" in moved:
        (base / "router").mkdir()
        for f in FEATURE_CACHES:
            if (base / moved["router"] / f).exists():
                shutil.copy2(base / moved["router"] / f, base / "router" / f)
                seeded.append(f"router/{f}")
    # the files the published rounds 0-5 reports (test_report.json c128, test_c16_report.json) were built from:
    # the minmax-raw set (the in-place zscore-log re-route of 2026-10-21 20:44 was evaluated offline only)
    report_sources = {
        rel: alt
        for rel, alt in (
            ("test/routing.json", "test/routing_minmaxraw_20261022.json"),
            ("test/routing_cv.json", "test/routing_cv_minmaxraw_20261022.json"),
            ("selection.json", "selection_minmaxraw_20261022.json"),
            ("router", "router_minmaxraw_20261022"),
            ("router_cv", "router_cv_minmaxraw_20261022"),
            ("router_cv_limeqo", "router_cv_limeqo_minmaxraw_20261022"),
        )
        if (base / alt).exists()
    }
    man = {
        "protocol": PROTOCOL,
        "archived_at": datetime.now(UTC).isoformat(),
        "tag": ARCHIVE_TAG,
        "ts": ts,
        "reason": "user 2026-10-22: re-select / re-test ALFWorld v7 on the rounds >= 6 catalog; the rounds 0-5 "
        "selection, routers, routing tables and test reports are kept under these names (never deleted)",
        "moved": moved,
        "report_sources": report_sources,
        "copied_into_new_router": seeded,
        "kept_in_place": [
            "test/pairs (c128 pair results)",
            "test_c16/ (c16 pair results: reuse source)",
            "*_minmaxraw_20261022*",
        ],
    }
    c16_save(m, man)
    return man


def c16_save(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


# ------------------------------------------------------------------ routing + pairs


def routing(base: Path) -> tuple[dict, dict]:
    """The NEW (catalog-regime) deployed and CV routing tables; refuses the archived / unlabelled ones."""
    tab = json.loads((base / "test" / "routing.json").read_text())
    cv = json.loads((base / "test" / "routing_cv.json").read_text())
    for name, t in (("routing.json", tab), ("routing_cv.json", cv)):
        cat = t.get("catalog") or {}
        if cat.get("from_round") != CATALOG_FROM_ROUND:
            raise SystemExit(
                f"test/{name} is not the catalog-regime (rounds >= {CATALOG_FROM_ROUND}) table: rerun the router / "
                "select / cv stages after `archive`"
            )
        if any(c16.is_excluded(n, a) for n, a in t["arms"].items()):
            raise SystemExit(f"test/{name} has LimeQO arms; this re-test is BRANE only")
    return tab, cv


def pair_set(tab: dict, cv: dict, deployed: bool = True) -> dict[str, list[str]]:
    from trimem.experiments import bo_pool_driver as d

    arms = {**(tab["arms"] if deployed else {}), **cv["arms"]}
    return d.pairs_of({"arms": arms})


def reusable_plan(cfg: dict) -> bool:
    """Pairs of plans without SemFilter (and NoMemory) are runtime-identical between the c16 run and this one."""
    from trimem.optimizer import session_space_agentenv_v2 as space

    return cfg.get("plan") not in space.SEMFILTER_PLANS


def old_c16_pairs(base: Path) -> set[tuple[str, str]]:
    """(config key, task) measured in the earlier c16 test rerun: evaluated opt rows + NoMemory pair files with an
    episodes.jsonl cost record."""
    from trimem.experiments import alf_test_c16_report as c16r
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    out = {
        (space.key(r["cfg"]), r["query_id"])
        for r in c16r._rows_of_opt(d.test_experiment(DS, c16.ARM_TAG))
        if r["cfg"]
    }
    nm_dir = base / c16.SUBDIR / "pairs" / d.cfg_name(space.NOMEM)
    if nm_dir.exists():
        rows = d.nomemory_rows(DS, nm_dir, sorted(d.drv.item_files(nm_dir, DS)))
        nk = space.key(space.NOMEM)
        out |= {(nk, q) for q, r in rows.items() if r.get("cost_source") == "episodes"}
    return out


def split_pairs(
    pairs: dict[str, list[str]], old: set[tuple[str, str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(reused, to run) per config key: a pair is reused iff measured in the old c16 run AND its plan is
    SemFilter-free. Pure."""
    reuse: dict[str, list[str]] = {}
    run: dict[str, list[str]] = {}
    for k, qs in pairs.items():
        ok = reusable_plan(json.loads(k))
        for q in qs:
            (reuse if ok and (k, q) in old else run).setdefault(k, []).append(q)
    return reuse, run


def new_done(base: Path, run: dict[str, list[str]]) -> int:
    """Pairs of ``run`` that already have a result file in this re-test's pair dirs (a resumed run)."""
    from trimem.experiments import bo_pool_driver as d

    n = 0
    for k, qs in run.items():
        have = d.drv.item_files(base / SUBDIR / "pairs" / d.cfg_name(json.loads(k)), DS)
        n += len(set(qs) & set(have))
    return n


def plan(deployed: bool = True) -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    base = run_base()
    tab, cv = routing(base)
    old = old_c16_pairs(base)
    out = {}
    for label, dep in (("with_deployed", True), ("cv_only", False)):
        pairs = pair_set(tab, cv, dep)
        reuse, run = split_pairs(pairs, old)
        sf = {k: v for k, v in run.items() if not reusable_plan(json.loads(k))}
        out[label] = {
            "arms": (len(tab["arms"]) if dep else 0) + len(cv["arms"]),
            "distinct_pairs": sum(map(len, pairs.values())),
            "configs": len(pairs),
            "reused_c16_pairs": sum(map(len, reuse.values())),
            "pairs_to_run": sum(map(len, run.values())),
            "pairs_to_run_semfilter": sum(map(len, sf.values())),
            "pairs_to_run_nomemory": len(run.get(space.key(space.NOMEM), [])),
            "already_done_here": new_done(base, run),
            "new_pairs_remaining": sum(map(len, run.values())) - new_done(base, run),
            "configs_to_run": len(run),
        }
        if dep == deployed:
            out["per_config_to_run"] = {
                f"{d.cfg_name(json.loads(k))} ({space.DISPLAY.get(json.loads(k)['plan'], 'NoMemory')})": len(
                    v
                )
                for k, v in sorted(run.items(), key=lambda kv: -len(kv[1]))
            }
    out["selected"] = "with_deployed" if deployed else "cv_only"
    return out


def build_arms(base: Path, run: dict[str, list[str]], semfilter: dict | None) -> list:
    """One subset arm per configuration (largest first), quality reuse OFF, new experiment / dirs."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    baseline, _ = d.test_baseline(DS, argparse.Namespace(sampling="per-config"))
    exp = d.test_experiment(DS, ARM_TAG)
    out = base / SUBDIR / "pairs"
    arms = []
    for k, qs in sorted(run.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            arms.append(d.Arm("v7 NoMemory pairs", out / d.cfg_name(cfg), qs, None))
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
        arms.append(d.Arm(f"v7 {d.cfg_name(cfg)}", out / d.cfg_name(cfg), qs, session))
    return arms


def semfilter_env(base: Path) -> tuple[dict, dict[str, str]]:
    """(state.json semfilter block, the TRIMEM_SEMFILTER_* env to export); refuses without POOL_SEMFILTER=v2 or
    with conflicting knobs (``bo_pool_driver.check_semfilter``)."""
    from trimem.experiments import bo_pool_driver as d

    state = json.loads((base / "state.json").read_text())
    env = d.check_semfilter(state)
    if env is None or state["semfilter"]["protocol"] != "semfilter-v2":
        raise SystemExit("state.json has no semfilter-v2 block")
    v2b = state["semfilter"].get(d.SEMFILTER_V2B_KEY)
    if v2b:  # semfilter-v2b (WebShop): new SemFilter test rows are labelled / run as v2b (rubric v4 env)
        return {**state["semfilter"], **v2b}, env
    return state["semfilter"], env


def run(deployed: bool = True, dry: bool = False) -> None:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import memory_policy as mp

    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    for k in ("TRIMEM_SEMANTIC_ROUTING", "TRIMEM_SEMANTIC_BASES", "MAX_ENVS"):
        if os.environ.get(k):
            raise SystemExit(f"{k} is set: controlled-c16-v1 runs with the defaults")
    base = run_base()
    tab, cv = routing(base)
    blk, env = semfilter_env(base)
    os.environ.update(env)
    _, todo = split_pairs(pair_set(tab, cv, deployed), old_c16_pairs(base))
    arms = build_arms(base, todo, blk)
    p = plan(deployed)
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
        "reuse_source": d.test_experiment(DS, c16.ARM_TAG),
        "reuse_rule": "old c16 pair reused iff measured there and its plan has no SemFilter",
        "deployed_arms": deployed,
        **p,
    }
    c16_save(base / SUBDIR / "plan.json", rec)
    with open(base / SUBDIR / "plan_history.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    batches = c16.schedule(arms)
    sel = p[p["selected"]]
    d.log(
        f"v7 re-test: {sel['pairs_to_run']} pairs to run ({sel['new_pairs_remaining']} not yet done; "
        f"{sel['reused_c16_pairs']} reused from test_c16) over {len(arms)} configurations in {len(batches)} "
        f"batches at concurrency {CONCURRENCY}, quality reuse off, semfilter-v2 env -> {base / SUBDIR}"
    )
    if dry:
        return
    res: dict[str, int] = {}
    for b in batches:
        mon = c16.HalfMonitor(b)
        mon.start()
        try:
            res |= d.run_parallel(DS, b, CONCURRENCY)
        finally:
            mon.stop.set()
            mon.join(timeout=5)
    bad = [a.name for a in arms if res.get(a.name, 0) < len(a.ids)]
    if bad:
        raise SystemExit(f"v7 re-test pairs incomplete: {bad}")
    d.log("V7 TEST DONE alfworld")


# ------------------------------------------------------------------ report


def pair_rows(base: Path, tab: dict, cv: dict) -> tuple[dict, dict, dict[str, int]]:
    """(config key, task) -> row in the wall and span definitions: this re-test's rows first, then the reusable
    old c16 rows (SemFilter-free plans only) for the pairs not run here; + source counts."""
    from trimem.experiments import alf_test_c16_report as c16r
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    want = {(k, q) for k, qs in pair_set(tab, cv).items() for q in qs}
    wall: dict = {}
    span: dict = {}
    src = {"new": 0, "reused_c16": 0}
    for tag, sub, reuse_only in (
        (ARM_TAG, SUBDIR, False),
        (c16.ARM_TAG, c16.SUBDIR, True),
    ):
        pdir = base / sub / "pairs"
        spans: dict[str, dict[str, float]] = {}
        for r in c16r._rows_of_opt(d.test_experiment(DS, tag)):
            if not r["cfg"]:
                continue
            k = space.key(r["cfg"])
            if (k, r["query_id"]) not in want or (k, r["query_id"]) in wall:
                continue
            if reuse_only and not reusable_plan(r["cfg"]):
                continue
            name = d.cfg_name(r["cfg"])
            if name not in spans:
                spans[name] = c16.call_spans(pdir / name)
            w, s = c16r.memory_row(r, spans[name])
            wall[(k, r["query_id"])] = w
            if s is not None:
                span[(k, r["query_id"])] = s
            src["reused_c16" if reuse_only else "new"] += 1
    nk = space.key(space.NOMEM)
    nm_q = [q for k, q in want if k == nk]
    for sub, label in ((SUBDIR, "new"), (c16.SUBDIR, "reused_c16")):
        todo = [q for q in nm_q if (nk, q) not in wall]
        if not todo:
            break
        w, s = c16r.nomemory_dir_rows(
            base / sub / "pairs" / d.cfg_name(space.NOMEM), todo
        )
        wall |= {(nk, q): r for q, r in w.items()}
        span |= {(nk, q): r for q, r in s.items() if q in w}
        src[label] += len(w)
    return wall, span, src


def references(ids: list[str], nm_old: dict[str, float]) -> dict:
    """Controlled-v1 originals: NoMemory (span latency only) and VGR (wall + span), as alf_test_c16_report."""
    from trimem.experiments import alf_test_c16_report as c16r
    from trimem.experiments import bo_pool_driver as d

    base_q = {"old": nm_old}
    _, nm_s = c16r.nomemory_dir_rows(d.drv.CTRL / f"{DS}_NoMemory", ids)
    vdir = d.drv.CTRL / f"{DS}_VGR"
    spans = c16.call_spans(vdir)
    files = d.drv.item_files(vdir, DS)
    vw, vs = {}, {}
    for r in c16r._rows_of_opt(f"{DS}:agenteval200:controlled-v1:VGR"):
        if r["query_id"] not in set(ids):
            continue
        g = r.get("generation") or {}
        if g and "environment_steps" not in g:
            f = files.get(r["query_id"])
            r = {**r, "generation": {**g, "result_file": str(f) if f else None}}
        a, b = c16r.memory_row(r, spans)
        vw[r["query_id"]] = a
        if b is not None:
            vs[r["query_id"]] = b
    return {
        "NoMemory": c16r.ref_block({}, nm_s, ids, base_q)
        | {
            "note": "controlled-v1 2026-10-15; no episodes.jsonl: LLM-call span latency only"
        },
        "VGR": c16r.ref_block(vw, vs, ids, base_q)
        | {
            "note": "controlled-v1 2026-10-15 (success-only-v0, seed 8 / depth 1 / fanout 4 / top-k 3)"
        },
    }


def report() -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import router_outer_cv as rcv

    base = run_base()
    tab, cv = routing(base)
    ids = d.test_ids(DS)
    ctrl_nm = d.drv.CTRL / f"{DS}_NoMemory"
    nm = {
        i: d.quality_of(DS, json.loads(f.read_text())["reward"])
        for i, f in d.drv.item_files(ctrl_nm, DS).items()
    }
    wall, span, src = pair_rows(base, tab, cv)

    def fold_metrics(rows):
        def f(t: dict) -> dict:
            got = d.assemble(t, rows, nm)
            return {k: v for k, v in got.items() if v.get("covered") == v.get("n")}

        return f

    arms_w, arms_s = d.assemble(tab, wall, nm), d.assemble(tab, span, nm)
    arms = {
        k: {**v, "latency_span_s": arms_s[k].get("latency_s")}
        for k, v in arms_w.items()
    }
    ts_w = rcv.test_summary(cv, fold_metrics(wall))
    ts_s = rcv.test_summary(cv, fold_metrics(span))
    for k, v in ts_w["summary"].items():
        v["latency_span_s"] = ts_s["summary"].get(k, {}).get("latency_s")
    incomplete = sorted(
        n for n, v in d.assemble(cv, wall, nm).items() if v.get("covered") != v.get("n")
    )
    sel = json.loads((base / "selection.json").read_text())
    pl = base / SUBDIR / "plan.json"
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL,
        "dataset": DS,
        "experiment": d.test_experiment(DS, ARM_TAG),
        "reuse_source": d.test_experiment(DS, c16.ARM_TAG),
        "plan": json.loads(pl.read_text()) if pl.exists() else None,
        "catalog": sel.get("catalog"),
        "replay_estimator": sel.get("replay_estimator"),
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
        "complete": not incomplete
        and all(v.get("covered") == v.get("n") for v in arms.values()),
        "arms": arms,
        "outer_cv": {
            "protocol": cv["protocol"],
            "folds": cv["folds"],
            "test": {**ts_w, "per_fold_span": ts_s["per_fold"]},
            "incomplete_test_arms": incomplete,
            "labels": {
                "mean_std": "mean ± sample std (ddof 1) across the 5 outer folds",
                "se_tasks_mean": "mean within-fold SE across tasks (task-sampling noise), NOT across folds",
            },
        },
        "reference_arms": references(ids, nm),
        "selection": {
            n: {
                "oneconfig": (s.get("opt_oneconfig") or {}).get("label"),
                "perquery_slo_train_only": (s.get("opt_perquery") or {}).get("weights"),
            }
            for n, s in sel["selections"].items()
        },
    }
    c16_save(base / REPORT, rep)
    (base / REPORT_MD).write_text("\n".join(markdown(rep)) + "\n")
    return rep


def _f(x, fmt: str) -> str:
    return "–" if x is None else format(x, fmt)


def _ms(m: dict | None, fmt: str) -> str:
    if not m or m.get("mean") is None:
        return "–"
    return format(m["mean"], fmt) + (
        "" if m.get("std") is None else " ± " + format(m["std"], fmt)
    )


def markdown(rep: dict) -> list[str]:
    cat = rep.get("catalog") or {}
    L = [
        "# ALFWorld v7 re-test — catalog rounds >= 6, concurrency 16",
        "",
        (
            f"Generated {rep['generated']}. Catalog: rounds {cat.get('included_rounds', ['?'])[0]}–"
            f"{cat.get('included_rounds', ['?'])[-1]} ({cat.get('full_configs')} fully evaluated configurations; "
            f"{cat.get('excluded_full_configs')} rounds 0–5 configurations excluded). Replay estimator "
            f"{rep.get('replay_estimator')}. {rep['pairs']} (config, task) pairs ({rep['pair_sources']}), quality "
            f"reuse OFF, controlled-c16-v1 runtime. Complete: {rep['complete']}."
        ),
        "",
        "## Outer 5-fold CV — test set (mean ± std across folds; SE_tasks = mean within-fold task SE)",
        "",
        (
            "| arm | quality | SE_tasks q | gain vs NoMemory | tokens | latency wall s | latency span s | steps | "
            "steps (succ.) | SE_tasks steps |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, x in rep["outer_cv"]["test"]["summary"].items():
        se = x.get("se_tasks_mean", {})
        L.append(
            f"| {n} | {_ms(x['quality'], '.3f')} | {_f(se.get('quality'), '.3f')} | {_ms(x['gain'], '+.3f')} | "
            f"{_ms(x['tokens'], ',.0f')} | {_ms(x['latency_s'], '.1f')} | {_ms(x.get('latency_span_s'), '.1f')} | "
            f"{_ms(x['steps'], '.2f')} | {_ms(x['steps_success'], '.2f')} | {_f(se.get('steps'), '.2f')} |"
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
        "## Deployed arms (full-train selection, single routing, 200 tasks)",
        "",
        "| arm | quality ± SE | tokens | latency wall s | latency span s | steps | gain vs NoMemory | configs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, x in rep["arms"].items():
        if x.get("quality") is None:
            continue
        L.append(
            f"| {n} | {x['quality']:.3f} ± {x['quality_se']:.3f} | {x['tokens']:,.0f} | {x['latency_s']:.1f} | "
            f"{_f(x.get('latency_span_s'), '.1f')} | {_f(x.get('steps'), '.2f')} | "
            f"{_f(x.get('paired_gain_vs_nomemory'), '+.3f')} | {x['distinct_configs']} |"
        )
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("archive", "plan", "run", "report"))
    ap.add_argument(
        "--cv_only",
        action="store_true",
        help="plan / run: only the outer-CV arms' pairs (default: deployed + CV, as the c16 rerun)",
    )
    ap.add_argument(
        "--dry", action="store_true", help="run: write plan.json and log, start nothing"
    )
    a = ap.parse_args(argv)
    if a.stage == "archive":
        print(json.dumps(archive(run_base()), indent=1))
    elif a.stage == "plan":
        print(json.dumps(plan(not a.cv_only), indent=1))
    elif a.stage == "run":
        run(not a.cv_only, a.dry)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
