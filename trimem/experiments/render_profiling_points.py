"""BO profiling points (train) and Opt-OneConfig / Opt-PerQuery test points for qa-trajectory.html (user 2026-10-19).

Section <!-- profiling-points:start/end -->, placed right after <!-- four-dataset-results:end -->.

A. Offline profiling (TRAINING panels only, split = 'train'): every configuration of the BO experiment
   that covers the full panel (``matrix.build``: exactly one evaluated row per panel query), aggregated
   per configuration -- mean quality, mean tokens (total_tokens = agent in + out + SemFilter tokens),
   mean latency (e2e_ms = retrieval + episode). No-Memory comes from the panel's official-evaluator files
   (``matrix.nomemory_column``). The relaxed front is the BO's noise-aware rule
   (``session_bo_agentenv.relaxed_dominates``, z = 1 pooled SE on paired gain / ln cost), restricted to the
   two objectives of each plot; membership in the BO's 3-objective relaxed front is listed as well.
   Q-best / Knee are the frozen choices in results/bo_agentenv_20261016/<panel>/selection.json.
B. Held-out test (ALFWorld / ScienceWorld 200 queries, τ² 100): fixed arms, Opt-OneConfig (BO Q-best arm) and
   Opt-PerQuery (the relaxed BRANE router; render_atlas_results loaders: quality, tokens = in + out + SemFilter,
   latency = retrieval + generation); only the router's aggregate point is shown.
C. SLO picks (slo_select_bo v2, percentile bounds over all training configs) on train and test plots, and the
   Opt-PerQuery out-of-fold λ sweep on the training panel (router_train_sweep.py JSON cache).
Display names (user 2026-10-19): BO Q-best -> "Opt-OneConfig", relaxed BRANE router -> "Opt-PerQuery";
internal ids / paths unchanged.

Read-only on results and the opt DB; no episodes are run.

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.render_profiling_points
"""

from __future__ import annotations

import fcntl
import json
import math
import statistics as st
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from trimem.experiments import render_atlas_results as ra

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
BO = ROOT / "results/bo_agentenv_20261016"
PANELS = ROOT / "data/external/panels"
DSN = (
    "host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa options='-c default_transaction_read_only=on "
    "-c statement_timeout=120000'"
)
START, END = "<!-- profiling-points:start -->", "<!-- profiling-points:end -->"
AFTER = "<!-- four-dataset-results:end -->"
RELAX_Z = 1.0  # = session_bo_agentenv.RELAX_Z

DS_NAME = {
    "alfworld": "ALFWorld",
    "sciworld": "ScienceWorld",
    "tau2": "τ²-bench",
    "webshop": "WebShop",
}
# (dataset, panel, BO experiment, run dir) -- train500 / train200 panels shown only when rows exist.
# ALFWorld train = the pool-strat200-v7 run (alf_pool_page.train_panel; user 2026-10-21): replaces the earlier
# alfworld train100 / train500 / train200 opt-DB panels (their 21 GP-only aux points are shown, toggleable, off).
TRAIN = [
    ("sciworld", "train100", "sciworld:train100:bo3obj-v1", "sciworld"),
    ("sciworld", "train500", "sciworld:train500:bo3obj-v2", "sciworld_train500"),
    ("sciworld", "train200", "sciworld:train200:bo3obj-v1", "sciworld_train200"),
]
# τ² train = the pool -v5 run (results/tau2_pool_v5_20261020; render_tau2_pool_points), user 2026-10-20: replaces the
# earlier tau2:train178:bo3obj-v1 opt-DB panel (its 6 GP-only aux points are shown, toggleable, off by default).
TEST_ARMS = [
    ("NoMemory", "No-Memory"),
    ("VGR", "VGR"),
    ("VGR+Rank", "VGR+Rank"),
    ("P2", "VGR+SemFilter"),
    ("P2+Rank", "VGR+SemFilter+Rank"),
    ("P1+Rank", "V+SemFilter+G+Rank"),
    ("BO-Qbest", "Opt-OneConfig"),
    ("BO-Knee", "BO Knee"),
]
N_TEST = 200


def cfg_label(cfg: dict) -> str:
    edges = {"similar": "sim", "similar_up": "up", "similar+similar_up": "sim+up"}.get(
        cfg["edges"], cfg["edges"]
    )
    return (
        f"{ra.PLAN_NAME.get(cfg['plan'], cfg['plan'])} s{cfg['seed_count']} d{cfg['depth']} "
        f"f{cfg['fanout']} k{cfg['top_k']} {edges}"
    )


def _se(a: np.ndarray) -> float:
    return float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0


def relaxed_front(
    objs: list[tuple[float, ...]], ses: list[tuple[float, ...]], z: float = RELAX_Z
) -> list[bool]:
    """Not statistically dominated (session_bo_agentenv.relaxed_dominates; all objectives maximised)."""

    def dom(q: int, p: int) -> bool:
        a, b, sa, sb = objs[q], objs[p], ses[q], ses[p]
        return all(x >= y for x, y in zip(a, b, strict=True)) and any(
            a[i] > b[i] + z * math.hypot(sa[i], sb[i]) for i in range(len(a))
        )

    return [
        not any(q != p and dom(q, p) for q in range(len(objs)))
        for p in range(len(objs))
    ]


def strict_front(xy: list[tuple[float, float]]) -> list[bool]:
    """(quality up, cost down) non-dominated."""
    return [
        not any(
            j != i and b[0] >= a[0] and b[1] <= a[1] and b != a
            for j, b in enumerate(xy)
        )
        for i, a in enumerate(xy)
    ]


# ------------------------------------------------------------------ A. training panels


def tau2_nomemory(query_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    """τ² train NoMemory (BO run's bo-NoMemory arm), in the opt train metric of tau2/runner.py:
    quality = reward, tokens = agent tokens (user simulator excluded), latency = episode wall (e2e_ms)."""
    by = {}
    for f in (BO / "tau2/arms/bo-NoMemory").glob("*.json"):
        r = json.load(open(f))
        if r.get("status") == "done":
            by[f"{r['domain']}:{r['task_id']}"] = r
    rs = [by[q] for q in query_ids]
    return {
        "quality": np.array([float(r["reward"]) for r in rs]),
        "tokens": np.array([float(r["agent_tokens"]) for r in rs]),
        "latency_ms": np.array([float(r["wall_ms"]) for r in rs]),
    }


def train_panel(ds: str, panel: str, exp: str, run: str) -> dict | None:
    import psycopg
    from psycopg.rows import dict_row

    from trimem.optimizer import session_space_agentenv as space
    from trimem.optimizer.query_router import matrix

    pf = PANELS / f"{ds}_{panel}.json"
    if not pf.exists():
        return None
    pnl = json.load(open(pf))
    ids = [str(x["item_id"]) for x in pnl["items"]]
    with psycopg.connect(DSN, row_factory=dict_row) as c:
        rows = c.execute(matrix.ROW_SQL, (exp,)).fetchall()
        n_cfg_any = c.execute(
            "SELECT count(DISTINCT config_hash) FROM opt.session_retrieval WHERE experiment_id = %s "
            "AND split = 'train'",
            (exp,),
        ).fetchone()["count"]
    if not rows:
        return None
    if any(r["split"] != "train" for r in rows):
        raise SystemExit(f"{exp}: non-train row in the profiling query")
    m = matrix.build(rows, panel_ids=ids, experiment=exp)
    nm = (
        tau2_nomemory(m.query_ids)
        if ds == "tau2"
        else matrix.nomemory_column(BO / run / "NoMemory", m.query_ids, ds)
    )
    if np.max(np.abs(nm["quality"] - m.baseline)) > 1e-9:
        raise SystemExit(
            f"{exp}: NoMemory files disagree with the paired baseline in opt"
        )
    sel_p = BO / run / "selection.json"
    sel = json.load(open(sel_p)) if sel_p.exists() else None
    if sel and sel["experiment"] != exp:
        sel = None
    qb = space.key(sel["q_best"]["config"]) if sel else None
    kn = space.key(sel["knee"]["config"]) if sel else None
    gain = m.gain
    pts = []
    for j, cfg in enumerate(m.configs):
        t, lat = m.tokens[:, j], m.latency_ms[:, j]
        pts.append(
            {
                "label": cfg_label(cfg),
                "plan": cfg["plan"],
                "cfg": space.validate(cfg),
                "q": float(m.quality[:, j].mean()),
                "q_se": _se(m.quality[:, j]),
                "gain": float(gain[:, j].mean()),
                "gain_se": _se(gain[:, j]),
                "tok": float(t.mean()),
                "lat": float(lat.mean()) / 1000,
                "tok_se": _se(t) / float(t.mean()),
                "lat_se": _se(lat) / float(lat.mean()),
                "qbest": space.key(cfg) == qb,
                "knee": space.key(cfg) == kn,
                "fixed": space.key(cfg)
                == space.key({"plan": cfg["plan"], **space.plan_defaults()}),
            }
        )
    obj = lambda p, ks: tuple([p["gain"]] + [-math.log(p[k]) for k in ks])  # noqa: E731
    se = lambda p, ks: tuple([p["gain_se"]] + [p[k + "_se"] for k in ks])  # noqa: E731
    for name, ks in (
        ("front_tok", ["tok"]),
        ("front_lat", ["lat"]),
        ("front3", ["tok", "lat"]),
    ):
        for p, f in zip(
            pts,
            relaxed_front([obj(p, ks) for p in pts], [se(p, ks) for p in pts]),
            strict=True,
        ):
            p[name] = f
    for name, k in (("strict_tok", "tok"), ("strict_lat", "lat")):
        for p, f in zip(pts, strict_front([(p["q"], p[k]) for p in pts]), strict=True):
            p[name] = f
    return {
        "ds": ds,
        "name": DS_NAME[ds],
        "panel": panel,
        "experiment": exp,
        "n_queries": len(m.query_ids),
        "n_configs": len(pts),
        "n_configs_seen": int(n_cfg_any),
        "split": "train",
        "noise_floor": sel["noise_floor"] if sel else None,
        "nomem": {
            "q": float(nm["quality"].mean()),
            "tok": float(nm["tokens"].mean()),
            "lat": float(nm["latency_ms"].mean()) / 1000,
        },
        "selection": bool(sel),
        "points": sorted(pts, key=lambda p: -p["q"]),
    }


# ------------------------------------------------------------------ B. test set

# per dataset: (number of test queries, per-episode loader, test arms). τ² test = 100 tasks (results/tau2_uncapped_4trials_20261019/test).
TEST = {
    "alfworld": (N_TEST, lambda a: ra.agentenv("alfworld", a), TEST_ARMS),
    "sciworld": (N_TEST, lambda a: ra.agentenv("sciworld", a), TEST_ARMS),
    "tau2": (
        100,
        ra.tau2,
        # reused 4-trial reference arms; the pool -v5 Opt arms are appended by tau2_pool_test
        [("NoMemory", "No-Memory"), ("VGR", "VGR"), ("P2", "VGR+SemFilter")],
    ),
}


def _cost(e: dict) -> tuple[float, float]:
    return e["tin"] + e["tout"] + e["tsem"], e["ret"] + e["gen"]


def test_dataset(ds: str) -> dict:
    if (
        ds == "webshop"
    ):  # user 2026-10-23: only listed once the WebShop v7 profiling paused (collect)
        return with_sweep_test(with_fixed_test(ws_pool_test(N_TEST)))
    n_test, load, test_arms = TEST[ds]
    if ds == "tau2":
        return with_fixed_test(tau2_pool_test(n_test))
    if ds == "alfworld":
        return with_sweep_test(with_fixed_test(alf_pool_test(n_test)))
    if ds == "sciworld" and sci_active():
        return with_sweep_test(with_fixed_test(sci_pool_test(n_test)))
    router = ra.relaxed_router(ds)
    arms = []
    base = load("NoMemory")
    for aid, label in test_arms + [("ROUTER", "Opt-PerQuery")]:
        real = router if aid == "ROUTER" else aid
        if not real:
            continue
        ep = load(real)
        if len(ep) != n_test or set(ep) != set(base):
            continue
        r = ra.summarize(ep, base)
        arms.append(
            {
                "id": aid,
                "label": label,
                **r,
                "q": r["quality"],
                "q_se": r["quality_se"],
                "tok": r["tokens"],
                "tok_se": r["tokens_se"],
                "lat": r["latency"],
                "lat_se": r["latency_se"],
                **(
                    {
                        "pass1": r["pass_k"]["1"]["mean"],
                        "pass1_se": r["pass_k"]["1"]["se"],
                    }
                    if "pass_k" in r
                    else {}
                ),
            }
        )
    return {
        "ds": ds,
        "name": DS_NAME[ds],
        "n": n_test,
        "router": router,
        "split": "test",
        "arms": arms,
        "trials": ra.TAU_TRIALS if ds == "tau2" else 1,
        "slo": slo_points(ds),
        "slo_bounds": slo_bounds(ds),
        "lam": test_sweep(ds),
    }


def with_sweep_test(T: dict) -> dict:
    """Opt-PerQuery weight sweep (user 2026-10-23; ``perquery_sweep_page``) under the ALFWorld / ScienceWorld test
    plots: overlays = this panel's arms (q / tok / lat = the plotted span latency)."""
    from trimem.experiments import perquery_sweep_page as pqs

    return {**T, "pqs": pqs.attach(T["ds"], T.get("arms") or [], "q", "tok", "lat")}


def with_fixed_test(T: dict) -> dict:
    """Fixed-plan reference arms (user 2026-10-23; ``fixed_plans_page``) appended to a test panel as ``mk: plan``
    (one marker per plan, legend toggle, SE bars on ALFWorld / ScienceWorld); ``fixed.line`` = the status line
    (pending until the fixed-plan report exists; absent plans)."""
    from trimem.experiments import fixed_plans_page as fp

    t = fp.strip(fp.table(T["ds"]))
    # ALFWorld / ScienceWorld panels read ``cv``: an empty object = single routing, SE bars, no CV columns
    extra = [{**a, "cv": {}} if T["ds"] != "tau2" else dict(a) for a in t["arms"]]
    return {
        **T,
        "arms": T["arms"] + extra,
        "fixed": {"status": t["status"], "line": fp.status_line(t)},
    }


def alf_pool_test(n_test: int) -> dict:
    """ALFWorld test panel: the pool-strat200-v7 outer-CV arms (mean ± std across 5 folds) + the reused
    controlled-v1 No-Memory / VGR references (alf_pool_page.test_table; user 2026-10-21)."""
    from trimem.experiments import alf_pool_page as ap

    t = ap.strip(ap.test_table())
    # v7 re-test (user 2026-10-22): arms = the deployed arms (primary), each with its CV numbers in ``cv``
    arms = [{**a, "cv": a.get("cv") or True} for a in t["refs"] + t["arms"]]
    return {
        "ds": "alfworld",
        "name": DS_NAME["alfworld"],
        "n": n_test,
        "router": None,
        "split": "test",
        "arms": arms,
        "trials": 1,
        "quality": "success rate",
        "slo": [],
        "slo_bounds": {},
        "lam": None,
        "cv": {
            k: t.get(k)
            for k in (
                "dir",
                "status",
                "k",
                "concurrency",
                "ref_concurrency",
                "memory_pairs",
                "reuse_hits",
                "deployed",
                "experiment",
                "run",
                "c128_meta",
                "primary",
            )
        }
        | {
            "c128_only": t.get("c128_only", []),
            "cv_arms": [
                {k: v for k, v in a.items() if k != "items"}
                for a in t.get("cv_arms", [])
            ],
        },
    }


def sci_active() -> bool:
    from trimem.experiments import sci_pool_page as sp

    return sp.active()


def ws_active() -> bool:
    from trimem.experiments import webshop_pool_page as wp

    return wp.active()


def ws_pool_test(n_test: int) -> dict:
    """WebShop test panel (user 2026-10-23; webshop_pool_page.test_table): the pool-strat200-v7 deployed arms
    (deployed only, user 2026-10-23 latest; their outer 5-fold CV numbers in ``cv`` if a CV test is ever run) + the
    controlled-v1 No-Memory / VGR references. Pending: a status line, no arms."""
    from trimem.experiments import webshop_pool_page as wp

    t = wp.strip(wp.test_table())
    # references: True; deployed arms: their CV numbers, or (deployed-only test, user 2026-10-23 latest) an object
    # without folds = alfTip's deployed branch with SE bars (as the ScienceWorld panel)
    arms = [{**a, "cv": True} for a in t["refs"]] + [
        {**a, "cv": a.get("cv") or {"deployed_only": True}} for a in t["arms"]
    ]
    return {
        "ds": "webshop",
        "name": DS_NAME["webshop"],
        "n": n_test,
        "router": None,
        "split": "test",
        "arms": arms,
        "trials": 1,
        "quality": "reward",
        "slo": [],
        "slo_bounds": {},
        "lam": None,
        "pending_v7": None if t.get("status") == "complete" else t.get("status"),
        "protocol_note": wp.protocol_note("test"),
        "cv": {
            k: t.get(k)
            for k in (
                "dir",
                "status",
                "k",
                "concurrency",
                "ref_concurrency",
                "memory_pairs",
                "reuse_hits",
                "deployed",
                "experiment",
                "run",
                "primary",
                "catalog_rounds",
                "catalog_configs",
            )
        }
        | {
            "c128_only": [],
            "cv_arms": [
                {k: v for k, v in a.items() if k != "items"}
                for a in t.get("cv_arms", [])
            ],
        },
    }


def sci_pool_test(n_test: int) -> dict:
    """ScienceWorld test panel (user 2026-10-22; sci_pool_page.test_table): the pool-strat200-v7 DEPLOYED arms only
    (Opt-OneConfig SLO-1/2/3 + Opt-PerQuery BRANE × 4 weights, ± SE across tasks; no outer CV) + the controlled-v1
    No-Memory / VGR references. Pending (no report yet): a status line, no arms."""
    from trimem.experiments import sci_pool_page as sp

    t = sp.strip(sp.test_table())
    meta = {
        k: t.get(k)
        for k in (
            "dir",
            "status",
            "k",
            "concurrency",
            "ref_concurrency",
            "memory_pairs",
            "reuse_hits",
            "deployed",
            "experiment",
            "run",
            "primary",
            "catalog_rounds",
            "catalog_configs",
        )
    }
    # deployed arms: an object ``cv`` without folds (alfTip's deployed branch, SE bars); references: True
    arms = [{**a, "cv": True} for a in t["refs"]] + [
        {**a, "cv": {"deployed_only": True}} for a in t["arms"]
    ]
    return {
        "ds": "sciworld",
        "name": DS_NAME["sciworld"],
        "n": n_test,
        "router": None,
        "split": "test",
        "arms": arms,
        "trials": 1,
        "quality": "score/100",
        "slo": [],
        "slo_bounds": {},
        "lam": None,
        "pending_v7": None if t.get("status") == "complete" else t.get("status"),
        "cv": meta | {"c128_only": [], "cv_arms": []},
    }


def tau2_ref_arms(trials: int) -> list[dict]:
    """Reused 4-trial reference arms (results/tau2_uncapped_4trials_20261019/test) on the test protocol's trials.
    4 trials: pass^4 (``ra.tau2``). Fewer (test-1trial-v1): their trials 0..trials-1 only (= the official seeds of the
    Opt arms), quality = pass^trials per task (1 trial: the trial-0 success), paired gain vs NoMemory on the same
    trials; their 4-trial pass^4 / pass^1 / costs ride along as context (``all4``)."""
    out = []
    full_base = ra.tau2("NoMemory")
    base = full_base if trials >= ra.TAU_TRIALS else _first_trials("NoMemory", trials)
    for aid, label in TEST["tau2"][2]:
        full = ra.tau2(aid)
        ep = full if trials >= ra.TAU_TRIALS else _first_trials(aid, trials)
        if len(ep) != TEST["tau2"][0] or set(ep) != set(base):
            continue
        r = ra.summarize(ep, base)
        a = {
            "id": aid,
            "label": label,
            **r,
            "q": r["quality"],
            "q_se": r["quality_se"],
            "tok": r["tokens"],
            "tok_se": r["tokens_se"],
            "lat": r["latency"],
            "lat_se": r["latency_se"],
            **(
                {"pass1": r["pass_k"]["1"]["mean"], "pass1_se": r["pass_k"]["1"]["se"]}
                if "pass_k" in r
                else {}
            ),
        }
        if trials < ra.TAU_TRIALS and len(full) == len(ep):
            f = ra.summarize(full, full_base)
            a["all4"] = {
                "trials": ra.TAU_TRIALS,
                "pass4": f["quality"],
                "pass1": f["pass_k"]["1"]["mean"],
                "tok": f["tokens"],
                "lat": f["latency"],
            }
        out.append(a)
    return out


def _first_trials(arm: str, trials: int) -> dict[str, dict]:
    """Per task: the arm's trials 0..trials-1 (tasks missing one are dropped); quality = pass^trials (all of them
    succeed; 1 trial = its success), costs = mean over those trials."""
    by: dict[str, dict[int, dict]] = {}
    for (task, trial), row in ra.tau2_episodes(arm).items():
        if trial < trials:
            by.setdefault(task, {})[trial] = row
    return {
        task: {
            **{
                k: st.mean(r[k] for r in rows.values())
                for k in ("tin", "tout", "tsem", "ret", "gen")
            },
            "q": float(all(r["q"] == 1 for r in rows.values())),
            "n_trials": trials,
        }
        for task, rows in by.items()
        if set(rows) == set(range(trials))
    }


def tau2_pool_test(n_test: int) -> dict:
    """τ² test panel: the reused reference arms + the pool -v5 Opt-OneConfig SLO-1/2/3 and Opt-PerQuery arms once
    the run's test stage has them (a pending note until then), on the protocol of the frozen routing table
    (test_trials / test_quality; 1 trial -> pass^1, 4 trials -> pass^4)."""
    from trimem.experiments import render_tau2_pool_points as t2

    pool = t2.test_arms()
    return {
        "ds": "tau2",
        "name": DS_NAME["tau2"],
        "n": n_test,
        "router": None,
        "split": "test",
        "arms": tau2_ref_arms(pool["trials"]) + pool.pop("arms"),
        "trials": pool["trials"],
        "quality": pool["quality"],
        "slo": [],
        "slo_bounds": pool.get("bounds") or {},
        "lam": None,
        "pool": pool,
    }


# ------------------------------------------------------------------ C. SLO-selected BO configurations

SLO_SEL = ROOT / "results/slo_selection_20261019/selections_v2.json"
LAM_SWEEP = ROOT / "results/slo_selection_20261019/router_lambda_sweep_train.json"


def slo_bounds(ds: str) -> dict:
    """Per-dataset SLO bounds (percentiles over all training configurations; slo_select_bo v2)."""
    if not SLO_SEL.exists():
        return {}
    d = json.load(open(SLO_SEL))["datasets"].get(ds)
    return d["bounds"] if d else {}


def slo_points(ds: str) -> list[dict]:
    """Test points of the SLO-selected configurations (selections_v2.json, user 2026-10-19): selected among
    all training configurations; tested only when an identical configuration has a complete test arm."""
    import psycopg

    if not SLO_SEL.exists() or ds not in json.load(open(SLO_SEL))["datasets"]:
        return []
    n_test, load, _ = TEST[ds]
    b_all = slo_bounds(ds)
    out = []
    for name, s in json.load(open(SLO_SEL))["datasets"][ds]["selections"].items():
        if not s:
            out.append({"slo": name, "none": True})
            continue
        tr = {
            "q": s["train_quality"],
            "tok": s["train_tokens"],
            "lat": s["train_latency_s"],
        }
        ep = load(s["test_arm"]) if s["test_arm"] else {}
        if len(ep) != n_test:
            out.append(
                {
                    "slo": name,
                    "arm": s["test_arm"],
                    "cfg": s["label"],
                    "pending": True,
                    "train": tr,
                }
            )
            continue
        if (
            ds == "tau2"
        ):  # τ² test latency = episode wall = retrieval + generation already
            e2e = None
        else:
            with psycopg.connect(DSN) as c:
                e2e = c.execute(
                    "SELECT avg(e2e_ms) / 1000 FROM opt.session_retrieval WHERE experiment_id = %s "
                    "AND split = 'test' AND status = 'evaluated'",
                    (f"{ds}:agenteval200:controlled-v1:{s['test_arm']}",),
                ).fetchone()[0]
        c_ = [_cost(e) for e in ep.values()]
        p = {
            "slo": name,
            "arm": s["test_arm"],
            "cfg": s["label"],
            "reuse": True,
            "n": len(ep),
            "q": st.mean(e["q"] for e in ep.values()),
            "tok": st.mean(x[0] for x in c_),
            "lat": st.mean(x[1] for x in c_),
            "e2e": float(e2e) if e2e is not None else None,
            "train": tr,
        }
        b = b_all[name]
        p["met"] = all(p[k] >= v if k == "q" else p[k] <= v for k, v in b.items()) and (
            "lat" not in b or p["e2e"] is None or p["e2e"] <= b["lat"]
        )
        out.append(p)
    return out


def test_sweep(ds: str) -> dict | None:
    """Opt-PerQuery TEST λ curve (user 2026-10-19). Per λ of the router grid, every test query is routed by the
    live replay of the deployed router (router_lambda_test_coverage: logged per-candidate scores, constant cost
    models, exact at the deployed λ); the episode of the routed configuration is an EXISTING test result of an
    identical configuration (deployed router arm first, then any other controlled test arm) or a targeted
    lambda-fill episode (lambda_fill.py). Single run per (configuration, query). A λ point is drawn only when
    every query has its episode."""
    if ds == "tau2":
        return None  # The legacy lambda-fill episodes have only one trial.
    from trimem.experiments import lambda_fill as lf
    from trimem.experiments import router_lambda_test_coverage as cov
    from trimem.optimizer.query_router import evaluate, runtime

    router = ra.relaxed_router(ds)
    if router is None:
        return None
    art = next(
        json.load(open(p))
        for p in sorted((cov.QR / ds).glob("router_*.json"))
        if f"Router-brane_paper-{runtime.artifact_sha(json.load(open(p)))}" == router
    )
    n_test, load, _ = TEST[ds]
    difficulty_types = None
    if ds in ("alfworld", "sciworld"):
        from trimem.experiments import difficulty_report as difficulty

        difficulty_types = difficulty.load_task_types(ds)
        difficulty_ids = [
            r["item_id"]
            for part in ("a", "b")
            for r in json.load(open(ra.CTRL / "shards" / f"{ds}_{part}.json"))
        ]
        difficulty_base = {
            f"{ds}_{q}": r["q"]
            for q, r in ra.agentenv(ds, "NoMemory", tolerant=True).items()
        }
    have, _ = cov.existing(ds)
    live = cov.live_scores(ds, art, router)
    cache: dict[str, dict] = {}

    def episodes(arm: str) -> dict:
        if arm not in cache:
            try:
                cache[arm] = (
                    ra.agentenv(ds, arm, tolerant=True)
                    if difficulty_types is not None
                    else load(arm)
                )
            except (KeyError, ValueError, OSError):
                cache[arm] = {}
        return cache[arm]

    def episode(key: tuple, cfg: dict, q: str) -> tuple[dict | None, str | None]:
        arms = [a for a in have.get((key, q), []) if a != lf.CANARY]
        order = ([router] if router in arms else []) + sorted(
            a for a in arms if a not in (router, lf.FILL)
        )
        order += [f"{lf.FILL}/{lf.cfg_name(cfg)}"] if lf.FILL in arms else []
        for a in order:
            e = episodes(a).get(
                q if ds == "tau2" else q.rsplit("_", 1)[1]
            )  # agentenv loaders key by item index
            if e is not None:
                return e, a
        return None, None

    pts = []
    lam0 = float(art["routing"]["lambda"])
    for lam in evaluate.LAMBDAS:
        rows, src, difficulty_episodes = [], {"reused": 0, "fill": 0}, {}
        for r in cov.route_live(art, live, lam):
            e, a = episode(r["key"], space_validate(r["config"]), r["query_id"])
            if e is not None:
                rows.append(e)
                src["fill" if a.startswith(lf.FILL) else "reused"] += 1
                if difficulty_types is not None:
                    tok, lat = _cost(e)
                    difficulty_episodes[r["query_id"]] = {
                        "quality": e["q"],
                        "tokens": tok if e["cost_complete"] else None,
                        "latency_s": lat if e["cost_complete"] else None,
                    }
        c = [_cost(e) for e in rows]
        p = {
            "lambda": lam,
            "deployed": math.isclose(lam, lam0),
            "n": len(live),
            "covered": len(rows),
            "complete": len(rows) == len(live) == n_test,
            **src,
        }
        if rows:
            p.update(
                q=st.mean(e["q"] for e in rows),
                tok=st.mean(x[0] for x in c),
                lat=st.mean(x[1] for x in c),
                q_se=float(
                    np.std([e["q"] for e in rows], ddof=1) / math.sqrt(len(rows))
                )
                if len(rows) > 1
                else 0.0,
            )
        if difficulty_types is not None:
            p["difficulty"] = difficulty.summarize(
                ds,
                difficulty_ids,
                difficulty_types,
                difficulty_episodes,
                difficulty_base,
            )
        pts.append(p)
    return {"deployed_lambda": lam0, "router": router, "points": pts}


def space_validate(cfg: dict) -> dict:
    from trimem.optimizer import session_space_agentenv as space

    return space.validate(cfg)


def train_slo(ds: str, panel: str) -> dict | None:
    """SLO bounds + selected training configurations, attached to the panel the selection was made on."""
    if not SLO_SEL.exists():
        return None
    d = json.load(open(SLO_SEL))["datasets"].get(ds)
    if not d or d["panel"] != panel:
        return None
    return {
        "bounds": d["bounds"],
        "picks": [
            {
                "slo": n,
                "label": s["label"],
                "tested": (
                    len(ra.tau2(s["test_arm"])) == 100 if ds == "tau2" else s["tested"]
                ),
                "arm": s["test_arm"],
            }
            for n, s in d["selections"].items()
            if s
        ],
    }


def train_sweep(ds: str, exp: str) -> dict | None:
    """Opt-PerQuery out-of-fold λ sweep on the training panel (router_train_sweep.py cache)."""
    if not LAM_SWEEP.exists():
        return None
    d = json.load(open(LAM_SWEEP))["datasets"].get(ds)
    return d if d and d["experiment"] == exp else None


# ------------------------------------------------------------------ D. pool profiling round (pool-strat200)

# v3 = the corrected space (edges similar+similar_up, user 2026-10-20); ``*_pool-strat200__mem-sf1`` = void run.
POOL_DIR = {
    ds: BO / f"{ds}_pool-strat200-v3__mem-sf1" for ds in ("alfworld", "sciworld")
} | {
    "alfworld": BO / "alfworld_pool-strat200-v7__mem-sf1"
}  # user 2026-10-21: ALFWorld = the v7 run


def pool_round(ds: str) -> dict:
    """Opt-PerQuery lambda curves of the pool-strat200 round (bo_pool_driver; user 2026-10-19): the training
    out-of-fold replay curve (router_report.json, equal cost weights) next to the TEST curve (test_report.json:
    each lambda arm assembled from single executions per (configuration, query), shared across arms), plus the
    SLO-chosen Opt-OneConfig / Opt-PerQuery test points. Empty / pending until the round produced them."""
    base = POOL_DIR[ds]
    out = {"ds": ds, "name": DS_NAME[ds], "dir": str(base.relative_to(ROOT))}
    from trimem.experiments import alf_test_c16 as c16

    # ALFWorld v7 after the re-test archive (user 2026-10-22): the rounds 0-5 artifacts under their archived names
    rr = c16.archived(base, "router") / "router_report.json"
    tr = c16.archived(base, "test_report.json")
    if (
        ds == "alfworld"
    ):  # v7 re-test (user 2026-10-22): rounds 6-21 router + test_v7c16 (deployed arms)
        from trimem.experiments import alf_test_v7 as v7

        rr, tr = base / "router" / "router_report.json", base / v7.REPORT
    if not rr.exists():
        return {**out, "pending": "profiling / router stage not finished"}
    R = json.load(open(rr))
    out["train_slices"] = pool_train_slices(R)
    if not tr.exists():
        return {**out, "pending": "test stage not finished"}
    T = json.load(open(tr))
    out["nomem"] = T["baseline"]
    out["notes"] = T.get("notes", [])
    pts = [
        {
            "name": k,
            "kind": v["kind"],
            "slo": v.get("slo"),
            "q": v["quality"],
            "q_se": v["quality_se"],
            "tok": v["tokens"],
            "lat": v["latency_s"],
            "distinct": v["distinct_configs"],
            "steps": v.get("steps"),
            "steps_success": v.get("steps_success"),
            "n": v["n"],
            "covered": v["covered"],
            **_pool_point_pos(v),
        }
        for k, v in T["arms"].items()
        if v.get("quality") is not None
    ]
    out["test"] = sorted(
        [p for p in pts if p["kind"] in ("slice", "curve")], key=lambda p: p["s"]
    )
    out["slo"] = [p for p in pts if p["kind"] in ("oneconfig", "perquery")]
    return out


def _wlabel(w: list[float]) -> str:
    return "w=" + "/".join(f"{v:g}" for v in w)


def _pool_point_pos(v: dict) -> dict:
    """Curve position of a pool test arm: three-weight slice arms (cost share s = 1 - w_Q, slice names) or
    legacy lambda-curve arms (s = lambda, one 'λ' slice)."""
    if v.get("weights") is not None and v["kind"] in ("slice", "perquery"):
        return {
            "s": round(1 - v["weights"][0], 6),
            "slices": v.get("slices") or [],
            "label": _wlabel(v["weights"]),
        }
    if v.get("lambda") is not None:
        return {"s": v["lambda"], "slices": ["λ"], "label": f"λ={v['lambda']:g}"}
    return {"s": 0.0, "slices": [], "label": ""}


def pool_train_slices(R: dict) -> dict[str, list[dict]]:
    """Training OOF replay curves of router_report.json, slice name -> points along the slice (by cost share
    s = w_L + w_T). Three-weight reports (sparse.PROTOCOL): curves['slices']; legacy reports: the equal-weight
    lambda curve as one 'λ' slice (s = lambda)."""
    pt = lambda r, s, lab: {  # noqa: E731
        "s": s,
        "label": lab,
        "q": r["q"],
        "q_se": r["quality_se"],
        "tok": r["tok"],
        "lat": r["lat"],
        "share": r["share"],
    }
    cur = R.get("curves", {})
    if "slices" in cur:
        return {
            k: [
                pt(
                    r,
                    round(1 - r["w_quality"], 6),
                    _wlabel([r["w_quality"], r["w_latency"], r["w_tokens"]]),
                )
                for r in rows
            ]
            for k, rows in cur["slices"].items()
        }
    if "lambda_curve" in cur:
        return {
            "λ": [pt(r, r["lambda"], f"λ={r['lambda']:g}") for r in cur["lambda_curve"]]
        }
    return {}


# ------------------------------------------------------------------ page

SECTION = r"""<!-- profiling-points:start -->
<section class="panel section" id="prof-pts"><style>
#prof-pts .pp-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:8px 0 14px}@media (max-width:900px){#prof-pts .pp-grid{grid-template-columns:minmax(0,1fr)}}
#prof-pts .pp-card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--panel);min-width:0}
#prof-pts .pp-card h4{margin:0 0 4px;font-size:14px}#prof-pts svg{display:block;width:100%;height:auto}
#prof-pts svg text{fill:var(--muted);font-size:11px;font-family:inherit}#prof-pts svg .pp-v{fill:var(--fg);font-size:10.5px}
#prof-pts .pp-leg{display:flex;gap:6px 14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:4px 0 8px}
#prof-pts .pp-leg i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
#prof-pts .pp-leg label{display:inline-flex;align-items:center;gap:3px;cursor:pointer;white-space:nowrap}#prof-pts .pp-leg input{margin:0 2px 0 0;accent-color:var(--fg)}#prof-pts .pp-leg input:focus-visible{outline:2px solid var(--fg);outline-offset:1px}
#prof-pts .pp-tabs{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 4px}#prof-pts .pp-tabs button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
#prof-pts .pp-stat{font-size:12.5px;color:var(--muted);margin:2px 0 6px}
#prof-pts table{border-collapse:collapse;font-size:12px;width:100%}#prof-pts th,#prof-pts td{border-bottom:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}
#prof-pts th:first-child,#prof-pts td:first-child{text-align:left}#prof-pts .pp-wrap{overflow-x:auto;max-height:420px}
#prof-pts .pp-leg svg{display:inline-block;width:12px;height:12px;vertical-align:-1px;margin-right:3px;overflow:visible}#prof-pts .pp-note{font-size:12px;color:var(--muted);margin:2px 0 8px}
#prof-pts details{margin:4px 0 12px}#prof-pts summary{cursor:pointer;font-size:13px}
</style>
<h2>BO profiling 点：训练集逐配置，与测试集上的 Opt-OneConfig / Opt-PerQuery</h2>
<p class="small"><b>命名</b>：Opt-OneConfig = BO 在训练面板上选出一个配置，测试集每题都用它；Opt-PerQuery = 逐题路由器（BRANE 式），为每道题选一个配置。</p>
<p class="small"><b>上半部分全部是训练集点</b>（<code>opt.session_retrieval</code> 中 <code>split='train'</code>、<code>status='evaluated'</code>，BO 实验 <code>&lt;ds&gt;:train100:bo3obj-v1</code>、<code>tau2:train178:bo3obj-v1</code> 等），一个点 = 一个跑满整个训练面板的配置（每题恰一行），取面板均值：
质量 = 平均成功率 / score / τ² reward；Token / 题 = agent 输入 + 输出 + 检索侧 LLM（SemFilter）（τ² 不含用户模拟器）；延迟 / 题 = 检索 + episode（<code>e2e_ms</code>）。No-Memory 来自同一面板的官方评估器结果文件（ALFWorld / ScienceWorld 延迟 = 首个到末个 LLM 调用的跨度；τ² = episode 墙钟）。
颜色 = 检索计划；实心点 = BO 用的<b>放宽 Pareto 前沿</b>（z = 1：只有在配对增益或 ln 成本上领先超过 1 个合并标准误才算支配，按当前图的两个目标计算）；虚线 = 严格前沿；◆ = 冻结的 Opt-OneConfig，▲ = Knee（训练面板上 Opt-OneConfig 噪声地板内 ln Token + ln 延迟最小者）。横轴为对数轴。</p>
<p class="small"><b>Opt-PerQuery 的 λ 扫描（绿色连线）</b>：路由器对每题按 score = q̂ − λ·(0.5 ln Token + 0.5 ln 延迟) 选配置；λ = 0 → 取预测质量最高的配置，λ 越大越偏向便宜的配置；部署的 λ（★）由 OOF knee 规则选出（质量在最好值一个 SE 内的最便宜 λ）。每个 λ 一个点 = 训练面板上的<b>折外</b>结果（5 折，训练折上用与部署路由器相同的特征 / 设置重新拟合，留出折按该 λ 路由，读其实测质量 / Token / 延迟，含路由开销；3 个折种子取均值）。测试集上只有部署 λ 有结果（★ Opt-PerQuery）；其他 λ 的测试点需要新的运行。</p>
<p class="small"><b>SLO（v2，阈值 = 该数据集训练面板全部配置的分位数，候选 = 全部配置）</b>：SLO-1 平均延迟 ≤ 第 25 百分位 → 取最高质量；SLO-2 延迟 ≤ 中位数且 Token ≤ 中位数 → 取最高质量；SLO-3 质量 ≥ 阈值（ALFWorld 0.90、ScienceWorld 0.80、τ² 取其配置质量中位数四舍五入到两位 = 0.73）→ 取最低延迟。并列：先低延迟再低 Token（SLO-3：先高质量再低 Token）。□ / ＋ / ▽ = SLO-1 / 2 / 3 选中的配置（界值见各数据集统计行）。图例每一项都是开关，可勾选显示 / 隐藏对应系列（对所有图生效，切换面板时保留）。</p>
<div id="pp-train"></div>
<h3>测试集：固定计划、Opt-OneConfig / Knee、SLO 选择与 Opt-PerQuery</h3>
<p class="small">Test points are per-task held-out results (ALFWorld / ScienceWorld 200 tasks each: <code>results/eval_controlled_20261015</code>; τ² 100 tasks: reference arms <code>results/tau2_uncapped_4trials_20261019/test</code>, Opt arms of the pool -v5 run), measured as in the training section: tokens = input + output + retrieval-side LLM; latency = retrieval (incl. routing probes) + generation (τ²: episode wall time, incl. the user simulator). Each point = one arm's mean over all test tasks; costs = mean over the trials used; SE at task level. τ² quality follows the test protocol of the frozen routing table (trials and pass^k in the τ² panel title; pass^1 on 1 trial, pass^4 on 4 trials); the training profiling is single-trial pass^1. ★ = Opt-PerQuery overall.
SLO picks with no test result of an identical configuration are marked as test pending (training values only, no point).</p>
<div id="pp-test"></div>
<script id="pp-data" type="application/json">__DATA__</script>
<script>(()=>{const D=JSON.parse(document.getElementById('pp-data').textContent),$=id=>document.getElementById(id),
e=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const PC={VGR:'#2a6fb3','VGR+Rank':'#1f9aa6',P2:'#8f5bb5','P2+Rank':'#c0558a','P1+Rank':'#b8862b',P1:'#6b8e23'},PN={P1:'V+SemFilter+G',VGR:'VGR','VGR+Rank':'VGR+Rank',P2:'VGR+SemFilter','P2+Rank':'VGR+SemFilter+Rank','P1+Rank':'V+SemFilter+G+Rank'};
const AC={NoMemory:'#8a8f87',VGR:'#2a6fb3','VGR+Rank':'#1f9aa6',P2:'#8f5bb5','P2+Rank':'#c0558a','P1+Rank':'#b8862b','BO-Qbest':'#c26a1b','BO-Knee':'#9a7b2f',ROUTER:'#2f8f6b'};
__FPJS____PQSJS__
const AN={NoMemory:'No-Memory',VGR:'VGR','VGR+Rank':'VGR+Rank',P2:'VGR+SemFilter','P2+Rank':'VGR+SemFilter+Rank','P1+Rank':'V+SemFilter+G+Rank','BO-Qbest':'Opt-OneConfig','BO-Knee':'BO Knee'};
const W=440,H=290,m={l:46,r:14,t:24,b:38};
const k=v=>v>=1000?(v/1000).toFixed(v>=1e5?0:1)+'k':v.toFixed(0),sec=v=>v.toFixed(v<10?1:0),q3=v=>v.toFixed(3);
const AX={tok:{n:'Token / 题',f:k},lat:{n:'延迟 s / 题',f:sec}};
const bf=(ax,v)=>ax==='lat'?v.toFixed(1)+' s':k(v);
function ticks(lo,hi){const out=[];const p=Math.floor(Math.log10(lo)),q=Math.ceil(Math.log10(hi));
 for(let e10=p;e10<=q;e10++)for(const s of [1,2,3,5,7]){const v=s*10**e10;if(v>=lo&&v<=hi)out.push(v)}
 if(out.length>6){return out.filter((v,i)=>i%Math.ceil(out.length/5)===0)}return out.length>=2?out:[lo,hi]}
function frame(xs,ys,ax,ylab='质量'){let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const lx=(Math.log(x1)-Math.log(x0))*.08||.1;x0=Math.exp(Math.log(x0)-lx);x1=Math.exp(Math.log(x1)+lx);const py=(y1-y0)*.1||.02;y0-=py;y1+=py;
 const X=v=>m.l+(Math.log(v)-Math.log(x0))/(Math.log(x1)-Math.log(x0))*(W-m.l-m.r),Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;const yt=[y0+py,(y0+y1)/2,y1-py];
 yt.forEach(v=>{s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${m.l-5}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`});
 ticks(x0,x1).forEach(v=>{s+=`<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--line)" stroke-dasharray="2 3"/><text x="${X(v)}" y="${H-m.b+14}" text-anchor="middle">${AX[ax].f(v)}</text>`});
 s+=`<text x="${(W+m.l)/2}" y="${H-6}" text-anchor="middle">${AX[ax].n}（对数轴，越左越好）</text><text x="12" y="${(H-m.b+m.t)/2}" text-anchor="middle" transform="rotate(-90 12 ${(H-m.b+m.t)/2})">${ylab}</text>`;
 return {s,X,Y}}
function tw(t){let w=0;for(const ch of t)w+=ch.charCodeAt(0)>255?10.5:6;return w}
// Greedy collision-free labels: merge coincident points, try 14 candidate positions, avoid markers / other labels / plot edge.
function labels(items,marks){const g=[];items.forEach(it=>{const h=g.find(o=>Math.hypot(o.x-it.x,o.y-it.y)<8);if(h){if(!h.t.split(' / ').includes(it.t))h.t+=' / '+it.t;h.r=Math.max(h.r,it.r)}else g.push({...it})});
 g.forEach(o=>{const ns=[...o.t.matchAll(/SLO-(\d)/g)].map(x=>x[1]);if(ns.length>1)o.t=o.t.split(' / ').filter(t=>!/^SLO-\d$/.test(t)).concat('SLO-'+ns.join('/')).join(' / ')});
 const boxes=marks.map(q=>q.box||{x1:q.x-q.r,x2:q.x+q.r,y1:q.y-q.r,y2:q.y+q.r}),placed=marks.filter(q=>q.box).map(q=>q.box);let out='';
 const hit=(b,arr)=>arr.reduce((n,o)=>n+(b.x1<o.x2&&o.x1<b.x2&&b.y1<o.y2&&o.y1<b.y2?1:0),0);
 g.sort((a,b)=>a.y-b.y).forEach(l=>{const w=tw(l.t),d=l.r+4,c=[];
  [1,2.5,4.5].forEach(dm=>[-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6].forEach(sy=>[1,-1,0].forEach(sx=>{if(!sx&&!sy)return;
   const ax=sx>0?'start':sx<0?'end':'middle',x=sx>0?l.x+d*dm:sx<0?l.x-d*dm:l.x,y=l.y+4+sy*(sx?13:Math.max(l.r+9,13));
   const x1=ax==='start'?x:ax==='end'?x-w:x-w/2;c.push({ax,x,y,dist:Math.hypot(x-l.x,y-4-l.y),b:{x1,x2:x1+w,y1:y-11,y2:y+3}})})));
  const own=boxes.filter(o=>!(Math.abs((o.x1+o.x2)/2-l.x)<1&&Math.abs((o.y1+o.y2)/2-l.y)<1));
  let best=null,bs=1e9;c.forEach((p,i)=>{const out_=p.b.x1<m.l+2||p.b.x2>W-m.r||p.b.y1<m.t-2||p.b.y2>H-m.b-2;const sc=(out_?1e5:0)+2000*hit(p.b,placed)+40*hit(p.b,own)+0.05*p.dist+0.001*i;if(sc<bs){bs=sc;best=p}});
  boxes.push(best.b);placed.push(best.b);const lead=Math.abs(best.y-4-l.y)>14?`<line x1="${l.x}" y1="${l.y}" x2="${best.ax==='end'?best.b.x2:best.ax==='start'?best.b.x1:best.x}" y2="${best.y-4}" stroke="var(--muted)" stroke-width=".6"/>`:'';
  out+=lead+`<text class="pp-v" x="${best.x}" y="${best.y}" text-anchor="${best.ax}">${e(l.t)}</text>`});return out}
function star(cx,cy,r){let p=[];for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,rr=i%2?r*.45:r;p.push((cx+rr*Math.cos(a)).toFixed(1)+','+(cy+rr*Math.sin(a)).toFixed(1))}return p.join(' ')}
const SC={'SLO-1':'#d6336c','SLO-2':'#1c7ed6','SLO-3':'#2b8a3e'};
function sloMark(n,cx,cy,t){const c=SC[n];return n==='SLO-1'?`<rect x="${cx-7}" y="${cy-7}" width="14" height="14" fill="none" stroke="${c}" stroke-width="2"><title>${e(t)}</title></rect>`
 :n==='SLO-2'?`<path d="M${cx-9} ${cy}H${cx+9}M${cx} ${cy-9}V${cy+9}" stroke="${c}" stroke-width="2.4"><title>${e(t)}</title></path>`
 :`<path d="M${cx-10} ${cy-7}L${cx+10} ${cy-7}L${cx} ${cy+10}Z" fill="none" stroke="${c}" stroke-width="2"><title>${e(t)}</title></path>`}
// Series visibility (legend checkboxes): one global set of hidden keys, kept across tabs; every toggle re-renders
// the plots so labels are re-placed for the visible series only. Axes stay fixed (computed from all data).
const HID=new Set(['aux']),on=k=>!HID.has(k);
function trainPlot(P,ax){const pts=P.points,SW=P.sweep,sp=SW?SW.points:[];
 const xs=pts.map(p=>p[ax]).concat(P.nomem[ax],sp.map(p=>p[ax])),ys=pts.map(p=>p.q).concat(P.nomem.q,sp.map(p=>p.q));
 let {s,X,Y}=frame(xs,ys,ax);const fr='front_'+ax,vis=pts.filter(p=>on('p:'+p.plan)),rel=p=>p[fr]&&on('relaxed');
 const st=vis.filter(p=>!vis.some(q=>q!==p&&q.q>=p.q&&q[ax]<=p[ax]&&(q.q>p.q||q[ax]<p[ax]))).sort((a,b)=>a[ax]-b[ax]);
 if(on('strict')&&st.length>1)s+=`<polyline fill="none" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.3" points="${st.map(p=>X(p[ax])+','+Y(p.q)).join(' ')}"/>`;
 const picks={};((P.slo&&P.slo.picks)||[]).forEach(x=>{(picks[x.label]=picks[x.label]||[]).push(x)});
 const tip=p=>`${p.label}${p.fixed?'（计划默认配置）':''}\n质量 ${q3(p.q)} ± ${q3(p.q_se)} · 增益 ${p.gain>=0?'+':''}${q3(p.gain)}\nToken ${k(p.tok)} · 延迟 ${sec(p.lat)} s${p[fr]?'\n放宽前沿（本图两目标）':''}${p.front3?'\nBO 三目标放宽前沿':''}${p.qbest?'\nOpt-OneConfig':''}${p.knee?'\nBO Knee':''}${(picks[p.label]||[]).map(x=>`\n${x.slo} 选中（${x.tested?'测试 arm '+x.arm:'测试未完成'}）`).join('')}`;
 vis.slice().sort((a,b)=>rel(a)-rel(b)).forEach(p=>{const cx=X(p[ax]),cy=Y(p.q),c=PC[p.plan]||'#888',f=rel(p);
  s+=`<circle cx="${cx}" cy="${cy}" r="${f?5.5:4.2}" fill="${f?c:'var(--panel)'}" fill-opacity="${f?.9:1}" stroke="${c}" stroke-width="1.6"><title>${e(tip(p))}</title></circle>`});
 const lab=[],mk=vis.map(p=>({x:X(p[ax]),y:Y(p.q),r:4})),nm=P.nomem;
 if(on('nomem')){lab.push({x:X(nm[ax]),y:Y(nm.q),t:'No-Memory',r:7});
  s+=`<rect x="${X(nm[ax])-5}" y="${Y(nm.q)-5}" width="10" height="10" fill="#8a8f87" transform="rotate(45 ${X(nm[ax])} ${Y(nm.q)})"><title>${e(`No-Memory（训练面板）\n质量 ${q3(nm.q)} · Token ${k(nm.tok)} · 延迟 ${sec(nm.lat)} s`)}</title></rect>`}
 if(sp.length&&on('sweep')){const o=sp.slice().sort((a,b)=>a.lambda-b.lambda),dep=SW.deployed_lambda;
  s+=`<polyline fill="none" stroke="${AC.ROUTER}" stroke-width="1.6" points="${o.map(p=>X(p[ax])+','+Y(p.q)).join(' ')}"/>`;
  o.forEach((p,i)=>{const cx=X(p[ax]),cy=Y(p.q),t=`Opt-PerQuery λ = ${p.lambda}（训练折外${p.lambda===dep?'，部署 λ':''}）\n质量 ${q3(p.q)}（逐题 SE ${q3(p.q_se_query)}）· Token ${k(p.tok)} · 延迟 ${sec(p.lat)} s\n选中 ${p.distinct} 个不同配置`;mk.push({x:cx,y:cy,r:3});
   s+=p.lambda===dep?`<polygon points="${star(cx,cy,8)}" fill="${AC.ROUTER}" stroke="var(--panel)" stroke-width="1"><title>${e(t)}</title></polygon>`
    :`<circle cx="${cx}" cy="${cy}" r="3" fill="${AC.ROUTER}"><title>${e(t)}</title></circle>`;
   if(p.lambda===dep)lab.push({x:cx,y:cy,t:`Opt-PerQuery λ=${p.lambda}`,r:9});else if(i===0||i===o.length-1)lab.push({x:cx,y:cy,t:`λ=${p.lambda}`,r:4})})}
 pts.filter(p=>(p.qbest&&on('qbest'))||(p.knee&&on('knee'))).forEach(p=>{const cx=X(p[ax]),cy=Y(p.q),qb=p.qbest&&on('qbest'),kn=p.knee&&on('knee');
  if(qb)s+=`<path d="M${cx} ${cy-9}L${cx+9} ${cy}L${cx} ${cy+9}L${cx-9} ${cy}Z" fill="none" stroke="#c26a1b" stroke-width="2"><title>${e(tip(p))}</title></path>`;
  if(kn)s+=`<path d="M${cx} ${cy-11}L${cx+10} ${cy+7}L${cx-10} ${cy+7}Z" fill="none" stroke="var(--fg)" stroke-width="1.6"><title>${e(tip(p))}</title></path>`;
  lab.push({x:cx,y:cy,t:qb&&kn?'Opt-OneConfig = Knee':qb?'Opt-OneConfig':'Knee',r:10})});
 pts.filter(p=>picks[p.label]).forEach(p=>{const cx=X(p[ax]),cy=Y(p.q);picks[p.label].filter(x=>on(x.slo)).forEach(x=>{s+=sloMark(x.slo,cx,cy,tip(p));lab.push({x:cx,y:cy,t:x.slo,r:10})})});
 s+=labels(lab,mk.concat(lab.map(l=>({x:l.x,y:l.y,r:l.r-2}))));
 return s+'</svg>'}
// τ² pool -v5 panel (render_tau2_pool_points): y = paired pass^1 gain vs No-Memory (what the BO sees); colour = plan,
// shape = proposal channel; filled + outline = 3-objective relaxed front; hollow dashed = partial n (stopped / pruned / incomplete).
const CHG={default:'default',sobol:'Sobol',qfirst:'qfirst','xover-slot':'xover-slot','sobol-slot':'sobol-slot',hv:'hv',dls:'DLS (agent / incumbent)','agent-gap':'agent-gap (LLM analyst)',crossover:'crossover (front grafts)',other:'other'};
const nn=(f,v)=>v==null?'—':f(v),sg=v=>(v>=0?'+':'')+q3(v);
function shp(g,cx,cy,r,a,t){const tt=t?`<title>${e(t)}</title>`:'',P=ps=>`<polygon points="${ps.map(([x,y])=>(cx+x*r).toFixed(1)+','+(cy+y*r).toFixed(1)).join(' ')}" ${a}>${tt}</polygon>`,q=.38;
 if(g==='sobol')return P([[-.85,-.85],[.85,-.85],[.85,.85],[-.85,.85]]);
 if(g==='qfirst')return P([[0,-1.3],[1.1,0],[0,1.3],[-1.1,0]]);
 if(g==='xover-slot')return P([[0,-1.25],[1.1,.8],[-1.1,.8]]);
 if(g==='sobol-slot')return P([[0,1.25],[1.1,-.8],[-1.1,-.8]]);
 if(g==='hv')return P([[-q,-1.15],[q,-1.15],[q,-q],[1.15,-q],[1.15,q],[q,q],[q,1.15],[-q,1.15],[-q,q],[-1.15,q],[-1.15,-q],[-q,-q]]);
 if(g==='dls')return P([0,1,2,3,4,5].map(i=>[Math.cos(Math.PI/3*i+Math.PI/6),Math.sin(Math.PI/3*i+Math.PI/6)]));
 if(g==='agent-gap')return P([0,1,2,3,4].map(i=>[1.15*Math.cos(2*Math.PI/5*i-Math.PI/2),1.15*Math.sin(2*Math.PI/5*i-Math.PI/2)]));
 if(g==='crossover')return P([[-1,-.6],[-.6,-1],[0,-.4],[.6,-1],[1,-.6],[.4,0],[1,.6],[.6,1],[0,.4],[-.6,1],[-1,.6],[-.4,0]]);
 return `<circle cx="${cx}" cy="${cy}" r="${r}" ${a}>${tt}</circle>`}
const gly=(g,a)=>`<svg viewBox="-7 -7 14 14" aria-hidden="true">${shp(g,0,0,5,a,'')}</svg>`;
const ptip=(P,p)=>`${p.id} · round ${p.round} · ${p.channel}${p.source?` (${p.source})`:''}\n${p.knobs}\nn = ${p.n} / ${P.n_queries}${p.partial?` (${p.status}${p.status==='incomplete'?`; ${p.done} tasks done, BO observation on ${p.n}`:''}${p.pruned_at?` at stage ${p.pruned_at}`:''})`:''} · reuse ${p.reuse} / ${p.n}\n${P.qname||'pass^1'} gain vs No-Memory${P.yk==='q'?' (unpaired, own samples)':''} ${sg(p.gain)} ± ${q3(p.gain_se)} SE\n${P.qname||'pass^1'} ${q3(p.q)} ± ${q3(p.q_se)}\nTokens / episode ${k(p.tok)} ± ${k(p.tok_se)} · latency ${sec(p.lat)} ± ${sec(p.lat_se)} s / episode${p.steps!=null?`\nsteps / episode ${p.steps.toFixed(2)}${p.steps_success!=null?` · on successes ${p.steps_success.toFixed(2)}`:''}`:''}${Object.keys(p.regimes).length?`\nload: ${Object.entries(p.regimes).map(([a,b])=>a+' '+b).join(', ')}`:''}${p.front3?'\n3-objective relaxed Pareto front':''}${p.strict3?' (strict front too)':''}${p.slo.length?'\nOpt-OneConfig '+p.slo.join(' / '):''}`;
function poolPlot(P,ax){const pts=P.points,A=on('aux')?P.aux:[],nm=P.nomem,vis=pts.filter(p=>on('p:'+p.plan)&&on('ch:'+p.group)),yk=P.yk||'gain',y0=yk==='q'?nm.q:0;
 const xs=pts.map(p=>p[ax]).concat(nm[ax],A.map(a=>a[ax])),ys=pts.map(p=>p[yk]).concat(y0,A.map(a=>a[yk]));
 let {s,X,Y}=frame(xs,ys,ax,yk==='q'?`success rate (own ${P.n_queries}-task sample)`:'pass^1 gain vs No-Memory');const lab=[],mk=[],rel=p=>p.front3&&on('relaxed');
 s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(y0)}" y2="${Y(y0)}" stroke="#8a8f87" stroke-dasharray="6 4"/>`;
 const fl=vis.filter(p=>!p.partial),sf=fl.filter(p=>!fl.some(q=>q!==p&&q[yk]>=p[yk]&&q[ax]<=p[ax]&&(q[yk]>p[yk]||q[ax]<p[ax]))).sort((a,b)=>a[ax]-b[ax]);
 if(on('strict')&&sf.length>1)s+=`<polyline fill="none" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.3" points="${sf.map(p=>X(p[ax])+','+Y(p[yk])).join(' ')}"/>`;
 A.forEach(a=>{const cx=X(a[ax]),cy=Y(a[yk]);mk.push({x:cx,y:cy,r:4});
  s+=`<circle cx="${cx}" cy="${cy}" r="4.4" fill="none" stroke="${PC[a.plan]||'#888'}" stroke-width="1.5" stroke-dasharray="1.6 1.4"><title>${e(yk==='q'?`aux (old run; GP-only input, never in the front): ${a.label}\n${a.experiment}\nsuccess ${q3(a.q)} on that run's panel = its No-Memory + paired gain ${sg(a.gain)} ± ${q3(a.gain_se)} SE\nTokens / episode ${k(a.tok)} · latency ${sec(a.lat)} s / episode (older protocol, success-only memory, concurrency 16: costs not comparable)`:`aux (old run; GP-only input, never in the front): ${a.label}\n${a.experiment}\npass^1 gain vs that run's No-Memory ${sg(a.gain)} ± ${q3(a.gain_se)} SE\nTokens / episode ${k(a.tok)} · latency ${sec(a.lat)} s / episode (older protocol: costs not comparable)`)}</title></circle>`});
 vis.slice().sort((a,b)=>rel(a)-rel(b)).forEach(p=>{const cx=X(p[ax]),cy=Y(p[yk]),c=PC[p.plan]||'#888',f=rel(p);mk.push({x:cx,y:cy,r:5});
  const at=p.partial?`fill="var(--panel)" stroke="${c}" stroke-width="1.6" stroke-dasharray="2.5 1.5"`:f?`fill="${c}" fill-opacity=".95" stroke="var(--fg)" stroke-width="1.5"`:`fill="${c}" fill-opacity=".35" stroke="${c}" stroke-width="1.2"`;
  s+=shp(p.group,cx,cy,f?6:4.8,at,ptip(P,p))});
 if(on('nomem')){lab.push({x:X(nm[ax]),y:Y(y0),t:'No-Memory',r:7});
  s+=`<rect x="${X(nm[ax])-5}" y="${Y(y0)-5}" width="10" height="10" fill="#8a8f87" transform="rotate(45 ${X(nm[ax])} ${Y(y0)})"><title>${e(yk==='q'?`No-Memory (its own stratified ${P.n_queries}-task sample, 1 run; gain 0 by definition)\nsuccess ${q3(nm.q)} ± ${q3(nm.q_se)} · Tokens / episode ${k(nm.tok)} · latency ${sec(nm.lat)} s / episode${nm.steps!=null?`\nsteps / episode ${nm.steps.toFixed(2)} · on successes ${nn(v=>v.toFixed(2),nm.steps_success)}`:''}`:`No-Memory (train178, 1 trial; gain 0 by definition)\npass^1 ${q3(nm.q)} ± ${q3(nm.q_se)} · Tokens / episode ${k(nm.tok)} · latency ${sec(nm.lat)} s / episode`)}</title></rect>`}
 vis.filter(p=>p.slo.length).forEach(p=>{const cx=X(p[ax]),cy=Y(p[yk]);p.slo.filter(n=>on(n)).forEach(n=>{s+=sloMark(n,cx,cy,ptip(P,p));lab.push({x:cx,y:cy,t:n,r:10})})});
 s+=labels(lab,mk.concat(lab.map(l=>({x:l.x,y:l.y,r:l.r-2}))));return s+'</svg>'}
const poolLeg=P=>{const pl=Object.keys(PC).filter(x=>P.points.some(p=>p.plan===x)||P.aux.some(a=>a.plan===x)),gs=Object.keys(CHG).filter(g=>P.points.some(p=>p.group===g));
 return `<div class="pp-leg" role="group" aria-label="Show / hide plans">${pl.map(x=>cb('p:'+x,dot(PC[x]),PN[x])).join('')}</div><div class="pp-leg" role="group" aria-label="Show / hide channels and marks">${gs.map(g=>cb('ch:'+g,gly(g,'fill="var(--muted)"'),CHG[g])).join('')}${cb('nomem','<i style="background:#8a8f87;border-radius:1px;transform:rotate(45deg)"></i>','No-Memory')}${cb('relaxed',gly('default','fill="var(--muted)" stroke="var(--fg)" stroke-width="1.5"'),'filled + outline = 3-objective relaxed front')}${cb('strict','┅ ','strict front (this plot, full n)')}${cb('aux','<i style="background:none;border:1.5px dashed var(--muted)"></i>',`aux (old run, ${P.aux.length})`)}${P.points.some(p=>p.slo.length)?SLOLEG:''}</div>`};
function poolStat(P){if(P.yk==='q')return alfStat(P);const f=P.points.filter(p=>p.front3).length,pa=P.points.filter(p=>p.partial),nm=P.nomem;
 return `<p class="pp-stat">τ²-bench · pool -v5 (<code>${e(P.dir)}</code>): train panel ${P.n_queries} tasks × 1 trial (pass^1), rounds 0–${P.rounds-1}, ${P.n_configs} profiled configurations: ${P.n_full} fully evaluated + ${pa.length} with partial n (hollow: ${pa.map(p=>`${p.id} ${p.status} n=${p.n}`).join(', ')}). Quality axis = paired pass^1 gain vs No-Memory on the same tasks (domain-stratified, ± SE on hover). No-Memory: pass^1 ${q3(nm.q)}, ${k(nm.tok)} tokens, ${sec(nm.lat)} s / episode. 3-objective relaxed front (gain, ln tokens, ln latency; z = ${P.relax_z} pooled SE, raw paired SE; fully evaluated configurations only): ${f} configurations. Compact memory cards, edges similar+similar_up.${P.semfilter?` SemFilter plans (${P.semfilter.configs.length} configurations: ${P.semfilter.rerun.length} rerun on all ${P.n_queries} tasks, ${P.semfilter.configs.length-P.semfilter.rerun.length} user-stopped kept stopped) were re-measured with ${e(P.semfilter.protocol)} (top-${P.semfilter.top_n} cosine prefilter, batched score-only judge, ${e(P.semfilter.rubric)}); their v1-judge episodes are superseded.`:''}${P.slo_bounds?' SLO picks of the frozen selection are marked.':''}</p>`}
function poolNotes(P){if(P.yk==='q')return alfNotes(P);const c=P.concurrency;return `<p class="pp-note"><b>Notes.</b> <b>Profiling latencies are high-load</b> (τ² 60–100 concurrent episodes) <b>and are used inside the optimizer only</b>. ${c?`Latency from round ${c.from_round} on is measured at profiling concurrency ${c.profiling_total} (vs ~${c.before.profiling_total} before), so later rounds run under higher load. `:''}Round 0 latency is inflated by load (6 arms, incl. No-Memory and the SemFilter plans, in parallel). ${P.reuse_from_round!=null?`Quality reuse (quality-reuse-v1) is on from round ${P.reuse_from_round}: a reused episode copies reward, agent tokens and episode latency from an episode with the identical injected memory (count on hover); configurations sharing reused episodes are correlated, so SEs of their differences are understated. `:''}Aux (old run): the ${P.aux.length} GP-only observations of <code>tau2:train178:bo3obj-v1</code>${P.aux_from_round!=null?` (used from round ${P.aux_from_round})`:''}, gain vs that run's own No-Memory${P.aux_nomem_q!=null?` (pass^1 ${q3(P.aux_nomem_q)})`:''}; older protocol, costs not comparable; never in the front; hidden by default.</p>`}
function poolTable(P){if(P.yk==='q')return alfTable(P);return `<details><summary>τ²-bench · pool -v5: all ${P.points.length} configurations (table)</summary><div class="pp-wrap"><table><thead><tr><th>config</th><th>channel</th><th>plan / knobs</th><th>status</th><th>n</th><th>reuse</th><th>pass^1</th><th>gain ± SE</th><th>Tokens</th><th>latency s</th><th>front</th></tr></thead><tbody>
 <tr><td>No-Memory</td><td></td><td></td><td>full</td><td>${P.n_queries}</td><td></td><td>${q3(P.nomem.q)}</td><td>0</td><td>${Math.round(P.nomem.tok).toLocaleString()}</td><td>${sec(P.nomem.lat)}</td><td></td></tr>
 ${P.points.map(p=>`<tr><td>${e(p.id)}</td><td>${e(p.channel)}${p.source?` (${e(p.source)})`:''}</td><td>${e(p.knobs)}</td><td>${e(p.status)}</td><td>${p.n}</td><td>${p.reuse}</td><td>${q3(p.q)}</td><td>${sg(p.gain)} ± ${q3(p.gain_se)}</td><td>${Math.round(p.tok).toLocaleString()}</td><td>${sec(p.lat)}</td><td>${p.front3?'relaxed':''}${p.strict3?' / strict':''}${p.slo.length?' · '+p.slo.join(' / '):''}</td></tr>`).join('')}
 ${P.aux.map(a=>`<tr><td>aux (old run)</td><td></td><td>${e(a.label)}</td><td>GP only</td><td>${P.n_queries}</td><td></td><td></td><td>${sg(a.gain)} ± ${q3(a.gain_se)}</td><td>${Math.round(a.tok).toLocaleString()}</td><td>${sec(a.lat)}</td><td></td></tr>`).join('')}</tbody></table></div></details>`}
function trainGrid(P){const yl=P.yk==='q'?'success rate':'pass^1 gain';return P.kind==='pool'?`<div class="pp-card"><h4>Train · ${yl} × tokens / episode</h4>${poolPlot(P,'tok')}</div><div class="pp-card"><h4>Train · ${yl} × latency / episode</h4>${poolPlot(P,'lat')}</div>`
 :`<div class="pp-card"><h4>训练 · 质量 × Token</h4>${trainPlot(P,'tok')}</div><div class="pp-card"><h4>训练 · 质量 × 延迟</h4>${trainPlot(P,'lat')}</div>`}
function poolTestNote(T){const P=T.pool,pend=P.pending||[],pk=P.picks||{},n=T.arms.filter(a=>a.mk&&a.mk!=='plan').length;
 return `<p class="pp-stat">Pool -v5 test arms (<code>${e(P.dir)}/test</code>, 100 tasks × ${T.trials} trial${T.trials>1?'s':''}; ${e(T.quality)} primary${T.trials>1?', pass^1 on hover':''}) next to the reused No-Memory / VGR / VGR+SemFilter arms (<code>results/tau2_uncapped_4trials_20261019/test</code>${T.trials<4?`, compared on their trial${T.trials>1?'s':''} 0${T.trials>1?'–'+(T.trials-1):''} = the same official seed${T.trials>1?'s':''}, paired gain on the same trials; their 4-trial pass^4 / pass^1 on hover as context`:''}): ${n?`${n} Opt-OneConfig / Opt-PerQuery arms drawn`:'<b>placeholder: no Opt-OneConfig SLO-1/2/3 or Opt-PerQuery (default / quality / tokens / latency) test arm yet</b>; only the reference arms are shown'} (${e(P.status)})${pend.length?`; incomplete, not drawn: ${pend.map(a=>`${e(a.label)} ${a.covered}/${a.n}`).join(', ')}`:''}${Object.keys(pk).length?`. Opt-OneConfig picks: ${Object.entries(pk).map(([s,l])=>`${s} → ${e(l||'none')}`).join('; ')}`:''}.</p>`}
function testPlot(T,ax){const A=T.arms,S=(T.slo||[]).filter(p=>p.q!=null),L=T.lam?T.lam.points.filter(p=>p.complete):[];
 const xs=A.map(a=>a[ax]).concat(S.map(p=>p[ax]),L.map(p=>p[ax])),ys=A.map(a=>a.q).concat(S.map(p=>p.q),L.map(p=>p.q));let {s,X,Y}=frame(xs,ys,ax);
 const lab=[];if(L.length&&on('tsweep')){const o=L.slice().sort((a,b)=>a.lambda-b.lambda);
  s+=`<polyline fill="none" stroke="${AC.ROUTER}" stroke-width="1.6" stroke-opacity=".75" points="${o.map(p=>X(p[ax])+','+Y(p.q)).join(' ')}"/>`;
  o.forEach((p,i)=>{const cx=X(p[ax]),cy=Y(p.q),t=`Opt-PerQuery λ = ${p.lambda}（测试${p.deployed?'，部署 λ':''}）\n质量 ${q3(p.q)}（逐题 SE ${q3(p.q_se)}）· Token ${k(p.tok)} · 延迟 ${sec(p.lat)} s\n${p.n} 题：复用已有同配置结果 ${p.reused}、定向补跑 ${p.fill}（单次运行）`;
   s+=p.deployed?`<polygon points="${star(cx,cy,7)}" fill="var(--panel)" stroke="${AC.ROUTER}" stroke-width="1.6"><title>${e(t)}</title></polygon>`
    :`<circle cx="${cx}" cy="${cy}" r="3.4" fill="${AC.ROUTER}"><title>${e(t)}</title></circle>`;
   if(!p.deployed&&(i===0||i===o.length-1))lab.push({x:cx,y:cy,t:`λ=${p.lambda}`,r:4})})}
 const V=A.filter(a=>on('a:'+a.id)),fr=V.filter(p=>!V.some(q=>q!==p&&q.q>=p.q&&q[ax]<=p[ax]&&(q.q>p.q||q[ax]<p[ax]))).sort((a,b)=>a[ax]-b[ax]);
 if(on('strict')&&fr.length>1)s+=`<polyline fill="none" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.3" points="${fr.map(p=>X(p[ax])+','+Y(p.q)).join(' ')}"/>`;
 V.slice().sort((a,b)=>Y(a.q)-Y(b.q)).forEach(a=>{const cx=X(a[ax]),cy=Y(a.q),c=a.color||AC[a.id],f=fr.includes(a)&&on('strict');
  if(a.q_std!=null){const xs_=a[ax+'_std']||0;s+=`<path d="M${cx} ${Y(a.q-a.q_std)}V${Y(a.q+a.q_std)}M${X(Math.max(a[ax]-xs_,1e-9))} ${cy}H${X(a[ax]+xs_)}" stroke="${c}" stroke-width="1.1" stroke-opacity=".75"/>`}else if(typeof a.cv==='object'&&a.q_se!=null){s+=`<path d="M${cx} ${Y(a.q-a.q_se)}V${Y(a.q+a.q_se)}" stroke="${c}" stroke-width="1.1" stroke-opacity=".75"/>`}
  const t=a.mk==='plan'?fpTip(T,a)+(fr.includes(a)?'\nstrict Pareto front (visible arms)':''):T.cv?alfTip(T,a)+(fr.includes(a)?'\nstrict Pareto front (visible arms)':''):`${a.label}（测试 ${a.n} 题）\n质量${T.quality?` (${T.quality})`:''} ${q3(a.q)} ± ${nn(q3,a.q_se)} SE · Token ${k(a.tok)} ± ${nn(k,a.tok_se)} · 延迟 ${sec(a.lat)} ± ${nn(sec,a.lat_se)} s${a.pass1!=null&&T.trials>1?`\npass^1 ${q3(a.pass1)} ± ${nn(q3,a.pass1_se)}`:''}${a.gain!=null&&(a.mk||T.pool)?`\n${T.quality||''} gain vs No-Memory ${sg(a.gain)} ± ${nn(q3,a.gain_se)}${T.trials<4&&T.pool?' (same trials)':''}`:''}${a.all4?`\ncontext, all ${a.all4.trials} trials: pass^4 ${q3(a.all4.pass4)} · pass^1 ${q3(a.all4.pass1)} · Tokens ${k(a.all4.tok)} · latency ${sec(a.all4.lat)} s`:''}${a.cfg?'\n'+a.cfg:''}${a.mk==='pq'?`\nweights (w_Q, w_L, w_T) = ${(a.weights||[]).join(' / ')} · ${a.distinct} distinct configurations`:''}${fr.includes(a)?'\n严格 Pareto 前沿（可见 arm 中）':''}`;
  s+=a.mk==='plan'?fpShape(a.plan,cx,cy,f?7.5:6.5,`fill="${f?c:'var(--panel)'}" stroke="${c}" stroke-width="2"`,`<title>${e(t)}</title>`)
   :a.id==='ROUTER'||a.mk==='pq'?`<polygon points="${star(cx,cy,a.mk?8:9)}" fill="${c}" stroke="${a.mk?'var(--panel)':c}"><title>${e(t)}</title></polygon>`
   :a.mk==='lq'?`<polygon points="${cx},${cy-7.5} ${cx+6.5},${cy} ${cx},${cy+7.5} ${cx-6.5},${cy}" fill="${f?c:'var(--panel)'}" stroke="${c}" stroke-width="2"><title>${e(t)}</title></polygon>`
   :a.mk==='oc'?`<rect x="${cx-6}" y="${cy-6}" width="12" height="12" fill="${f?c:'var(--panel)'}" stroke="${c}" stroke-width="2"><title>${e(t)}</title></rect>`
   :`<circle cx="${cx}" cy="${cy}" r="${f?6.5:5}" fill="${f?c:'var(--panel)'}" stroke="${c}" stroke-width="2"><title>${e(t)}</title></circle>`;
  lab.push({x:cx,y:cy,t:a.label,r:8})});
 S.filter(p=>on(p.slo)).forEach(p=>{const cx=X(p[ax]),cy=Y(p.q);s+=sloMark(p.slo,cx,cy,`${p.slo}：${p.cfg}（测试 arm ${p.arm}，复用已有结果，${p.n} 题）\n质量 ${q3(p.q)} · Token ${k(p.tok)} · 延迟 ${sec(p.lat)} s${p.e2e!=null?` · e2e ${sec(p.e2e)} s`:''}\n训练：质量 ${q3(p.train.q)} · Token ${k(p.train.tok)} · e2e ${sec(p.train.lat)} s\n测试上 SLO ${p.met?'满足':'未满足'}`);
  lab.push({x:cx,y:cy,t:p.slo,r:10})});
 s+=labels(lab,lab.map(l=>({x:l.x,y:l.y,r:6})));return s+'</svg>'}
// ALFWorld pool-strat200-v7 panel (alf_pool_page.train_panel): y = success rate on each configuration's OWN stratified
// 200-task sample (unpaired estimator); same markers as the τ² pool panel.
const f2=v=>v==null?'—':v.toFixed(2);
function alfStat(P){const f=P.points.filter(p=>p.front3),pa=P.points.filter(p=>p.partial),nm=P.nomem,B=P.slo_bounds,R=P.regime;
 if(R)return `<p class="pp-stat">${e(P.title)} (<code>${e(P.dir)}</code>, ${e(P.space_version)}, memory policy ${e(P.memory_policy)}) — <b>current regime only: rounds ${R.from_round}–${R.last_round}</b> (runtime ${e(R.runtime)}, concurrency ${R.concurrency}; ${e(R.semfilter)} for SemFilter plans; each configuration on its <b>own</b> task-type-stratified sample of ${R.sample_size} tasks × 1 run from the ${P.pool_size.toLocaleString()}-task pool, staged 50 → 100 with 3-D Pareto pruning). ${P.n_configs} configurations = ${P.n_full} fully evaluated + ${pa.length} pruned at 50 (hollow, partial-n means: ${pa.map(p=>p.id).join(', ')}). ${R.excluded_configs?`The ${R.excluded_configs} configurations of rounds 0–${R.from_round-1} (c128 latency, SemFilter v1) are not shown and not used by the selection. `:''}Quality axis = stratified ${e(P.qname==='success'?'success rate':P.qname)} (± SE on hover); gain = quality − No-Memory's (unpaired). No-Memory = ${e(R.nomemory_label||'its c16 rerun')} on its own ${R.nomemory_n||200}-task sample (<code>${e(R.nomemory_dir)}</code>): success ${q3(nm.q)} ± ${q3(nm.q_se)}, ${k(nm.tok)} tokens, ${sec(nm.lat)} s, ${f2(nm.steps)} steps / episode (${f2(nm.steps_success)} on successes). 3-objective relaxed front (gain, ln tokens, ln latency; z = ${P.relax_z} pooled SE; fully evaluated configurations of this regime only): ${f.map(p=>p.id).join(', ')}.${B?` SLO bounds (percentiles over these ${P.n_full} configurations): SLO-1 latency ≤ ${B['SLO-1'].lat.toFixed(1)} s; SLO-2 latency ≤ ${B['SLO-2'].lat.toFixed(1)} s and tokens ≤ ${Math.round(B['SLO-2'].tok).toLocaleString()}; SLO-3 success ≥ ${B['SLO-3'].q.toFixed(2)} (full-train Opt-OneConfig picks of the frozen catalog selection marked).`:' Selection for this catalog not frozen yet.'}</p>`;
 return `<p class="pp-stat">${e(P.title)} (<code>${e(P.dir)}</code>, ${e(P.space_version)}, memory policy ${e(P.memory_policy)}): pool of ${P.pool_size.toLocaleString()} non-test tasks; every configuration runs its <b>own</b> task-type-stratified sample of ${P.n_queries} tasks × 1 run, staged 50 → 100 → 200 with 3-D Pareto pruning. Rounds 0–${P.rounds-1}${P.paused_after_round!=null?` (profiling paused by the user after round ${P.paused_after_round})`:''}: ${P.n_configs} configurations = ${P.n_full} fully evaluated + ${pa.length} pruned (hollow: ${pa.map(p=>`${p.id} n=${p.n}`).join(', ')}); budget ${(P.budget_used||0).toLocaleString()} / ${(P.budget_episodes||0).toLocaleString()} episodes. Quality axis = stratified success rate (± SE on hover); gain on hover = success − No-Memory's success on its own 200-task sample (unpaired). No-Memory: success ${q3(nm.q)} ± ${q3(nm.q_se)}, ${k(nm.tok)} tokens, ${sec(nm.lat)} s, ${f2(nm.steps)} steps / episode (${f2(nm.steps_success)} on successes). 3-objective relaxed front (gain, ln tokens, ln latency; z = ${P.relax_z} pooled SE; fully evaluated configurations only): ${f.map(p=>p.id).join(', ')}.${P.noise_floor!=null?` Noise floor (independent VGR repeat): ${q3(P.noise_floor)}.`:''}${B?` SLO bounds: SLO-1 latency ≤ ${B['SLO-1'].lat.toFixed(1)} s; SLO-2 latency ≤ ${B['SLO-2'].lat.toFixed(1)} s and tokens ≤ ${Math.round(B['SLO-2'].tok).toLocaleString()}; SLO-3 success ≥ ${B['SLO-3'].q.toFixed(2)} (full-train Opt-OneConfig picks marked).`:''}</p>`}
function alfNotes(P){const c=P.concurrency||{},ch=(c.changes||[])[0],RL=P.round_latency||[],src=P.aux_sources||[],R=P.regime;
 if(R)return `<p class="pp-note"><b>Notes.</b> Latency = episode wall time + retrieval, measured under the <b>controlled ${e(R.runtime)} runtime (${R.concurrency} concurrent episodes, 2 halves × 8 processes)</b>, the same layout as the test runs, so profiling and test latency are on one load regime; it is still a training-pool estimate on each configuration's own tasks. Median latency of the fully evaluated configurations per round: ${RL.map(r=>`r${r.round} ${sec(r.median)} s (n=${r.n})`).join(', ')}. ${P.reuse_from_round!=null?`Quality reuse (quality-reuse-v1): a reused episode copies the outcome, agent tokens and episode latency of an episode with the identical task and injected memory, restricted to episodes of this regime (count on hover / in the table); configurations sharing reused episodes are correlated. `:''}Steps = environment steps per episode (agent LLM turns), mean over the observed tasks and over the successful ones. No aux (old-run) points: they predate this regime.${P.protocol_note?` <b>${e(P.protocol_note)}</b>`:''}</p>`;
 return `<p class="pp-note"><b>Notes.</b> <b>Profiling latencies are high-load</b> (ALFWorld 96–128 concurrent episodes) <b>and are used inside the optimizer only</b>; test latency is measured at concurrency 16. <b>Latency regimes differ across rounds</b>, so compare latency within a round rather than across rounds: ${ch?`profiling concurrency changed from ${ch.from} to ${ch.to} during round ${ch.round} (${e(String(ch.at).slice(0,16).replace('T',' '))} UTC; round-0 episodes mix both regimes), `:''}and the load differs (round 0 ran 10 configurations + No-Memory together, round 3 included heavy SemFilter / Rank configurations). Median latency of the fully evaluated configurations per round: ${RL.map(r=>`r${r.round} ${sec(r.median)} s (n=${r.n})`).join(', ')}. ${P.reuse_from_round!=null?`Quality reuse (quality-reuse-v1) is on from round ${P.reuse_from_round}: a reused episode copies the outcome, agent tokens and episode latency of an episode with the identical task and injected memory (count on hover / in the table); configurations sharing reused episodes are correlated. `:''}Steps = environment steps per episode (agent LLM turns), mean over the observed tasks and over the successful ones. Aux (old runs, toggle, hidden by default): the ${P.aux.length} GP-only observations of ${src.map(x=>`<code>${e(x.experiment)}</code> (${x.configs_used})`).join(' + ')}${P.aux_from_round!=null?`, used from round ${P.aux_from_round}`:''}; drawn at their own panel's success (that run's No-Memory + paired gain); older protocol (success-only memory, other serving load), costs not comparable; never in the front.</p>`}
function alfTable(P){const nm=P.nomem;return `<details><summary>${e(P.title)}: all ${P.points.length} configurations (table)</summary><div class="pp-wrap"><table><thead><tr><th>config</th><th>channel</th><th>plan / knobs</th><th>status</th><th>n</th><th>reuse</th><th>success ± SE</th><th>gain ± SE</th><th>Tokens</th><th>latency s</th><th>steps</th><th>steps (succ.)</th><th>front</th></tr></thead><tbody>
 <tr><td>No-Memory</td><td></td><td></td><td>full</td><td>${P.n_queries}</td><td></td><td>${q3(nm.q)} ± ${q3(nm.q_se)}</td><td>0</td><td>${Math.round(nm.tok).toLocaleString()}</td><td>${sec(nm.lat)}</td><td>${f2(nm.steps)}</td><td>${f2(nm.steps_success)}</td><td></td></tr>
 ${P.points.map(p=>`<tr><td>${e(p.id)}</td><td>${e(p.channel)}${p.source?` (${e(p.source)})`:''}</td><td>${e(p.knobs)}</td><td>${e(p.status)}${p.pruned_at?` @${p.pruned_at}`:''}</td><td>${p.n}</td><td>${p.reuse}</td><td>${q3(p.q)} ± ${q3(p.q_se)}</td><td>${sg(p.gain)} ± ${q3(p.gain_se)}</td><td>${Math.round(p.tok).toLocaleString()}</td><td>${sec(p.lat)}</td><td>${f2(p.steps)}</td><td>${f2(p.steps_success)}</td><td>${p.front3?'relaxed':''}${p.strict3?' / strict':''}${p.slo.length?' · '+p.slo.join(' / '):''}</td></tr>`).join('')}
 ${P.aux.map(a=>`<tr><td>aux (old run)</td><td></td><td>${e(a.label)}</td><td>GP only</td><td></td><td></td><td>${q3(a.q)}</td><td>${sg(a.gain)} ± ${q3(a.gain_se)}</td><td>${Math.round(a.tok).toLocaleString()}</td><td>${sec(a.lat)}</td><td>—</td><td>—</td><td></td></tr>`).join('')}</tbody></table></div></details>`}
// ALFWorld v7 test panel (alf_pool_page.test_table): outer 5-fold CV arms (mean ± std across fold routers / selections;
// error bars = std across folds) + the reused controlled-v1 references; deployed full-train arms in the table only.
function alfTestNote(T){const C=T.cv;if(C.run==='sci-v7')return `<p class="pp-stat">ScienceWorld v7 test (<code>${e(C.dir)}/test_v7c16</code>; ${T.n} AgentEval tasks × 1 run under the controlled c16 runtime, quality reuse off, SemFilter v2). The selection catalog = the ${C.catalog_configs??''} fully evaluated configurations of the training panel above (rounds ${(C.catalog_rounds||[])[0]??0}–${(C.catalog_rounds||[])[1]??6}). Points = the <b>deployed</b> arms only (selection / router fitted on all training queries; no outer cross-validation), error bars = ± SE across the ${T.n} tasks. □ = Opt-OneConfig SLO-1/2/3, ★ = Opt-PerQuery BRANE (weights default / quality / tokens / latency). Quality = score/100. References No-Memory / VGR = the original controlled-v1 reference arms (same c16 layout). Plotted latency = LLM-call span + retrieval for every arm; the episode wall time is in the table.</p>`;if(C.run==='v7')return `<p class="pp-stat">ALFWorld v7 re-test (<code>${e(C.dir)}/test_v7c16</code>; ${T.n} AgentEval tasks × 1 run under the controlled c16 runtime, quality reuse off, SemFilter v2). The selection catalog = the fully evaluated configurations of the training panel above (rounds 6–21). Points = the <b>deployed</b> arms (selection / router fitted on all training queries; primary), error bars = ± SE across the ${T.n} tasks; the outer ${C.k}-fold CV numbers (mean ± std across folds) are in the table and on hover. □ = Opt-OneConfig SLO-1/2/3, ★ = Opt-PerQuery BRANE (weights default / quality / tokens / latency). References No-Memory / VGR = the original controlled-v1 reference arms (same c16 layout). Plotted latency = LLM-call span + retrieval for every arm (the only definition the original No-Memory run has); the episode wall time is in the table.</p>`;if(C.run==='c16'){const M=C.c128_meta||{};return `<p class="pp-stat">${C.k}-fold outer-CV test arms of the pool-strat200-v7 run, <b>concurrency-16 rerun</b> (<code>${e(C.dir)}/test_c16</code>; ${T.n} AgentEval tasks × 1 run; quality reuse off; the reference runs' layout: one configuration at a time, 2 halves × 8 processes). Each fold's selection / router is fitted on 4/5 of the training queries, every fold is evaluated on the same ${T.n} tasks. Points = mean across the ${C.k} folds, error bars = ± std across folds; SE_tasks and steps on hover. □ = Opt-OneConfig SLO-1/2/3, ★ = Opt-PerQuery BRANE (weights default / quality / tokens / latency). References No-Memory / VGR = the original controlled-v1 reference arms (2026-10-15, the same c16 layout). Plotted latency = <b>LLM-call span + retrieval</b> for every arm (the only definition the original No-Memory run has), so it is like-for-like; the episode wall time is the second latency column (VGR wall 43.4 s). These test arms come from the earlier selection over profiling rounds 0–5; the train panel above now shows the rounds 6–21 configurations (c16) that the v7 re-test selects from (its test: section #alf-v7). The c128 run (quality reuse ${M.reuse_hits} / ${M.memory_pairs} memory pairs) is the secondary latency column; LimeQO ran only at c128 (table rows marked c128 only, not plotted).</p>`}return `<p class="pp-stat">${C.k}-fold outer-CV test arms of the pool-strat200-v7 run (<code>${e(C.dir)}/test</code>; ${T.n} AgentEval tasks × 1 run, concurrency ${C.concurrency}; ${C.memory_pairs} memory (configuration, task) pairs, ${C.reuse_hits} of them quality-reuse hits): each fold's selection / router is fitted on 4/5 of the training queries, every fold is evaluated on the same ${T.n} tasks. Points = mean across the ${C.k} folds, error bars = ± std across folds (router-training variability); SE_tasks (task-sampling noise, mean within-fold SE) and steps on hover. □ = Opt-OneConfig SLO-1/2/3, ★ = Opt-PerQuery BRANE, ◆ = Opt-PerQuery LimeQO (weights default / quality / tokens / latency). References No-Memory / VGR = the reused controlled-v1 test arms (single run, concurrency ${C.ref_concurrency}, memory policy success-only-v0): success comparable on the same tasks, <b>latency not comparable</b> (16 vs ${C.concurrency} concurrent episodes) and tokens only partly. The deployed full-train arms are listed in the table below.</p>`}
function fpTip(T,a){return `${a.label} (fixed plan default; ${a.n} test tasks)\n${T.quality||'quality'} ${q3(a.q)} ± ${nn(q3,a.q_se)} SE${a.gain!=null?` · gain vs No-Memory ${sg(a.gain)} ± ${nn(q3,a.gain_se)}`:''}\nTokens / episode ${k(a.tok)} · latency ${sec(a.lat)} s${a.lat_wall!=null?` (LLM span), ${sec(a.lat_wall)} s (wall)`:''}${a.steps!=null?`\nsteps / episode ${f2(a.steps)}${a.steps_se!=null?' ± '+f2(a.steps_se)+' SE':''}${a.steps_success!=null?' · on successes '+f2(a.steps_success):''}`:''}${a.pair_sources?`\npairs: ${Object.entries(a.pair_sources).map(([x,n])=>x+' '+n).join(', ')}`:''}`}
function alfTip(T,a){const s=a.se_tasks||{},cv=a.q_std!=null,c=typeof a.cv==='object'?a.cv:null;if(c)return `${a.label} (deployed, full-train fit; ${a.n} test tasks)\nsuccess ${q3(a.q)} ± ${q3(a.q_se)} SE · gain vs No-Memory ${sg(a.gain)} ± ${q3(a.gain_se)}\nTokens / episode ${k(a.tok)} · latency ${sec(a.lat)} s (span), ${sec(a.lat_wall)} s (wall)\nsteps / episode ${f2(a.steps)} ± ${f2(a.steps_se)} · on successes ${f2(a.steps_success)}${a.weights?`\nweights (w_Q, w_L, w_T) = ${a.weights.join(' / ')}`:''}\ndistinct configurations: ${a.distinct}${c.folds?`\nouter ${c.folds}-fold CV: success ${q3(c.q)} ± ${q3(c.q_std)} std · tokens ${k(c.tok)} · wall ${sec(c.lat_wall)} s · steps ${f2(c.steps)}`:'\nno outer CV (deployed arms only)'}`;return `${a.label}${cv?` (${a.folds}-fold CV, test ${a.n} tasks)`:` (reference, single run, ${a.n} tasks)`}\nsuccess ${q3(a.q)}${cv?` ± ${q3(a.q_std)} std`:''} · SE_tasks ${nn(q3,s.q)}\ngain vs No-Memory ${a.gain==null?'—':sg(a.gain)}${cv?` ± ${q3(a.gain_std)} std`:''} · SE_tasks ${nn(q3,s.gain)}\nTokens / episode ${k(a.tok)}${cv?` ± ${k(a.tok_std)}`:''} · latency ${sec(a.lat)}${cv?` ± ${sec(a.lat_std)}`:''} s\nsteps / episode ${f2(a.steps)}${cv?` ± ${f2(a.steps_std)}`:''} (SE_tasks ${f2(s.steps)}) · on successes ${f2(a.steps_success)}${a.weights?`\nweights (w_Q, w_L, w_T) = ${a.weights.join(' / ')}`:''}${a.distinct_per_fold?`\ndistinct configurations per fold: ${a.distinct_per_fold.join(' / ')}`:''}${a.protocol?'\n'+a.protocol:''}`}
function alfTestTable(T){const C=T.cv;if(C.run==='sci-v7'){const F=v=>Math.round(v).toLocaleString(),sh=a=>a.plan_shares?Object.entries(a.plan_shares).map(([p,v])=>`${e(PN[p]||p)} ${(100*v).toFixed(0)}%`).join(', '):'—';
 return `<details open><summary>${e(T.name)} test values (table; deployed arms only, ± SE across tasks; references single run)</summary><div class="pp-wrap"><table><thead><tr><th>arm</th><th>n</th><th>score/100 ± SE</th><th>gain vs No-Memory ± SE</th><th>Tokens</th><th>latency s (wall)</th><th>latency s (span)</th><th>steps ± SE</th><th>steps (succ.)</th><th>configs</th><th>routing shares (plan)</th></tr></thead><tbody>
${T.arms.map(a=>`<tr><td>${e(a.label)}</td><td>${a.n}</td><td>${q3(a.q)} ± ${nn(q3,a.q_se)}</td><td>${a.gain==null||a.id==='NoMemory'?'—':sg(a.gain)+' ± '+nn(q3,a.gain_se)}</td><td>${F(a.tok)}</td><td>${a.lat_wall==null?'—':sec(a.lat_wall)}</td><td>${sec(a.lat)}</td><td>${f2(a.steps)}${(a.steps_se??(a.se_tasks||{}).steps)!=null?' ± '+f2(a.steps_se??a.se_tasks.steps):''}</td><td>${f2(a.steps_success)}</td><td>${a.distinct??'—'}</td><td style="text-align:left;white-space:normal">${a.kind==='ref'?'—':sh(a)}</td></tr>`).join('')}</tbody></table></div></details>`}if(C.run==='v7'){const F=v=>Math.round(v).toLocaleString(),pm=(v,s,f)=>v==null?'—':f(v)+(s!=null?' ± '+f(s).replace(/^\+/,''):'');
 return `<details open><summary>${e(T.name)} test values (table; deployed arms primary ± SE across tasks; outer-CV mean ± std secondary; references single run)</summary><div class="pp-wrap"><table><thead><tr><th>arm</th><th>n</th><th>success ± SE</th><th>gain vs No-Memory ± SE</th><th>Tokens</th><th>latency s (wall)</th><th>latency s (span)</th><th>steps ± SE</th><th>steps (succ.)</th><th>configs</th><th>CV success</th><th>CV gain</th><th>CV tokens</th><th>CV wall s</th><th>CV steps</th></tr></thead><tbody>
${T.arms.map(a=>{const c=typeof a.cv==='object'?a.cv:{};return `<tr><td>${e(a.label)}</td><td>${a.n}</td><td>${q3(a.q)} ± ${nn(q3,a.q_se)}</td><td>${a.gain==null||a.id==='NoMemory'?'—':sg(a.gain)+' ± '+nn(q3,a.gain_se)}</td><td>${F(a.tok)}</td><td>${a.lat_wall==null?'—':sec(a.lat_wall)}</td><td>${sec(a.lat)}</td><td>${f2(a.steps)}${(a.steps_se??(a.se_tasks||{}).steps)!=null?' ± '+f2(a.steps_se??a.se_tasks.steps):''}</td><td>${f2(a.steps_success)}</td><td>${a.distinct??'—'}</td><td>${pm(c.q,c.q_std,q3)}</td><td>${pm(c.gain,c.gain_std,sg)}</td><td>${pm(c.tok,c.tok_std,F)}</td><td>${pm(c.lat_wall,c.lat_wall_std,sec)}</td><td>${pm(c.steps,c.steps_std,f2)}</td></tr>`}).join('')}</tbody></table></div></details>`}const C16=C.run==='c16',ms=(v,s,f)=>v==null?'—':f(v)+(s!=null?' ± '+f(s).replace(/^\+/,''):''),F=v=>Math.round(v).toLocaleString();
 return `<details open><summary>${e(T.name)} test values (table; mean ± std across folds, references single run)</summary><div class="pp-wrap"><table><thead><tr><th>arm</th><th>n</th><th>success</th><th>gain vs No-Memory</th><th>Tokens</th><th>latency s${C16?' (c16, span)':''}</th>${C16?'<th>c16 wall</th><th>c128 wall</th>':''}<th>steps</th><th>steps (succ.)</th><th>SE_tasks success / gain / steps</th></tr></thead><tbody>
${T.arms.concat(C.c128_only||[]).map(a=>{const s=a.se_tasks||{},c=a.c128;return `<tr${a.c128_only?' style="opacity:.75"':''}><td>${e(a.label)}</td><td>${a.n}</td><td>${ms(a.q,a.q_std,q3)}</td><td>${a.gain==null?'—':ms(a.gain,a.gain_std,sg)}</td><td>${ms(a.tok,a.tok_std,F)}</td><td>${a.c128_only&&C16?'—':ms(a.lat,a.lat_std,sec)}</td>${C16?`<td>${a.c128_only||a.lat_wall==null?'—':ms(a.lat_wall,a.lat_wall_std,sec)}</td><td>${a.c128_only?ms(a.lat,a.lat_std,sec):c?ms(c.lat,c.lat_std,sec):'—'}</td>`:''}<td>${ms(a.steps,a.steps_std,f2)}</td><td>${ms(a.steps_success,a.steps_success_std,f2)}</td><td>${nn(q3,s.q)} / ${nn(q3,s.gain)} / ${nn(f2,s.steps)}</td></tr>`}).join('')}
<tr><td colspan="${C16?11:9}" style="text-align:left"><b>Deployed arms</b> (router / selection fitted on the full training set; secondary; ± = SE across tasks)</td></tr>
${C.deployed.map(a=>`<tr${a.c128_only?' style="opacity:.75"':''}><td>${e(a.label)}</td><td>${a.covered}/${a.n}</td><td>${q3(a.q)} ± ${q3(a.q_se)}</td><td>${sg(a.gain)} ± ${q3(a.gain_se)}</td><td>${F(a.tok)}</td><td>${a.c128_only&&C16?'—':sec(a.lat)}</td>${C16?`<td>${a.c128_only||a.lat_wall==null?'—':sec(a.lat_wall)}</td><td>${a.c128_only?sec(a.lat):a.c128?sec(a.c128.lat):'—'}</td>`:''}<td>${f2(a.steps)} ± ${f2(a.steps_se)}</td><td>${f2(a.steps_success)}</td><td>${a.distinct} config${a.distinct>1?'s':''}</td></tr>`).join('')}</tbody></table></div></details>`}
// legend entry = <label><input type=checkbox data-k=key> swatch name</label>
const SLO_DESCRIPTION={'SLO-1':'Highest quality within a latency limit','SLO-2':'Highest quality within latency and token limits','SLO-3':'Lowest latency while meeting a minimum quality'};
const sloLegendLabel=name=>String(name).replace(/SLO-[123]/g,s=>s+' — '+SLO_DESCRIPTION[s]);
const cb=(key,sw,name,style='')=>{if(/SLO-[123]/.test(name))style+=';white-space:normal';return `<label${style?` style="${style}"`:''}><input type="checkbox" data-k="${e(key)}"${on(key)?' checked':''}>${sw}${sloLegendLabel(name)}</label>`;};
const dot=c=>`<i style="background:${c}"></i>`;
const SLOLEG=`${cb('SLO-1','□ ','SLO-1',`color:${SC['SLO-1']}`)}${cb('SLO-2','＋ ','SLO-2',`color:${SC['SLO-2']}`)}${cb('SLO-3','▽ ','SLO-3',`color:${SC['SLO-3']}`)}`;
const leg=()=>`<div class="pp-leg" role="group" aria-label="显示 / 隐藏系列">${Object.keys(PC).map(p=>cb('p:'+p,dot(PC[p]),PN[p])).join('')}${cb('nomem','<i style="background:#8a8f87;border-radius:1px;transform:rotate(45deg)"></i>','No-Memory')}${cb('qbest','◆ ','Opt-OneConfig')}${cb('knee','▲ ','Knee')}${cb('relaxed','● ','实心 = 放宽前沿')}${cb('strict','┅ ','严格前沿')}${cb('sweep','— ','Opt-PerQuery λ 扫描（★ 部署 λ）',`color:${AC.ROUTER}`)}${SLOLEG}</div>`;
function trainTable(P){const pk={};((P.slo&&P.slo.picks)||[]).forEach(x=>{(pk[x.label]=pk[x.label]||[]).push(x.slo)});
 return `<details><summary>${e(P.name)} · ${e(P.panel)} 全部 ${P.points.length} 个配置（表格）</summary><div class="pp-wrap"><table><thead><tr><th>配置</th><th>质量</th><th>± SE</th><th>增益</th><th>Token</th><th>延迟 s</th><th>放宽前沿 (Tok / 延迟 / 三目标)</th><th>选择</th></tr></thead><tbody>
 <tr><td>No-Memory</td><td>${q3(P.nomem.q)}</td><td></td><td>0</td><td>${Math.round(P.nomem.tok).toLocaleString()}</td><td>${sec(P.nomem.lat)}</td><td></td><td></td></tr>
 ${P.points.map(p=>`<tr><td>${e(p.label)}</td><td>${q3(p.q)}</td><td>${q3(p.q_se)}</td><td>${(p.gain>=0?'+':'')+q3(p.gain)} ± ${q3(p.gain_se)}</td><td>${Math.round(p.tok).toLocaleString()}</td><td>${sec(p.lat)}</td><td>${[p.front_tok,p.front_lat,p.front3].map(f=>f?'●':'·').join(' ')}</td><td>${[p.qbest?'Opt-OneConfig':'',p.knee?'Knee':''].concat(pk[p.label]||[]).filter(Boolean).join(' / ')}</td></tr>`).join('')}
 ${P.sweep?P.sweep.points.map(p=>`<tr><td>Opt-PerQuery λ = ${p.lambda}${p.lambda===P.sweep.deployed_lambda?'（部署）':''}（折外）</td><td>${q3(p.q)}</td><td>${q3(p.q_se_query)}</td><td>${(p.q-P.nomem.q>=0?'+':'')+q3(p.q-P.nomem.q)}</td><td>${Math.round(p.tok).toLocaleString()}</td><td>${sec(p.lat)}</td><td></td><td>${p.distinct} 个配置</td></tr>`).join(''):''}</tbody></table></div></details>`}
function stat(P){const f=t=>P.points.filter(p=>p[t]).length,B=P.slo&&P.slo.bounds;return `<p class="pp-stat">${e(P.name)} · ${e(P.panel)}：训练集 ${P.n_queries} 题 × ${P.n_configs} 个完整配置${P.ds==='tau2'?'（历史单 trial，质量为单次 reward，尚无训练 pass^4）':''}${P.n_configs_seen>P.n_configs?`（另有 ${P.n_configs_seen-P.n_configs} 个配置未跑满，不显示）`:''}；放宽前沿点：Token 图 ${f('front_tok')}、延迟图 ${f('front_lat')}、三目标 ${f('front3')}；${P.selection?`噪声地板 ${q3(P.noise_floor)}`:'此面板没有冻结的 Opt-OneConfig / Knee（未做噪声重复与选择）'}。${B?`SLO 界：SLO-1 延迟 ≤ ${B['SLO-1'].lat.toFixed(1)} s；SLO-2 延迟 ≤ ${B['SLO-2'].lat.toFixed(1)} s 且 Token ≤ ${Math.round(B['SLO-2'].tok).toLocaleString()}；SLO-3 质量 ≥ ${B['SLO-3'].q.toFixed(2)}。`:''}${P.sweep?`Opt-PerQuery 部署 λ = ${P.sweep.deployed_lambda}（${e(P.sweep.lambda_rule)} 规则）。`:''}</p>`}
const byDs={};D.train.forEach(P=>{(byDs[P.ds]=byDs[P.ds]||[]).push(P)});
$('pp-train').innerHTML=Object.keys(byDs).map(ds=>`<h3>${e(byDs[ds][0].name)} · 训练集 profiling</h3>${byDs[ds].length>1?`<div class="pp-tabs" data-ds="${ds}">${byDs[ds].map((P,i)=>`<button data-i="${i}" aria-pressed="${i===0}">${e(P.panel)}（${P.n_queries} 题 × ${P.n_configs} 配置）</button>`).join('')}</div>`:''}<div id="pp-tr-${ds}"></div>`).join('');
function showTrain(ds,i){const P=byDs[ds][i],pl=P.kind==='pool';$('pp-tr-'+ds).innerHTML=(pl?poolStat(P)+poolLeg(P):stat(P)+leg())+`<div class="pp-grid" data-ds="${ds}" data-i="${i}">${trainGrid(P)}</div>`+(pl?poolNotes(P)+poolTable(P):trainTable(P));
 document.querySelectorAll(`#pp-train .pp-tabs[data-ds="${ds}"] button`).forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.i===i)))}
Object.keys(byDs).forEach(ds=>showTrain(ds,0));
document.querySelectorAll('#pp-train .pp-tabs').forEach(t=>t.addEventListener('click',ev=>{const b=ev.target.closest('button');if(b)showTrain(t.dataset.ds,+b.dataset.i)}));
function sloStat(T){const S=T.slo||[];if(!S.length)return '';return `<p class="pp-stat">SLO 选择（训练面板全部配置内，阈值见上方训练统计行）：${S.map(p=>p.none?`${p.slo} 无满足的配置`:p.pending?`${p.slo} → ${e(p.cfg)}（<b>测试未完成</b>；训练 质量 ${q3(p.train.q)} · Token ${k(p.train.tok)} · e2e ${sec(p.train.lat)} s）`:`${p.slo} → ${e(p.cfg)}（复用 ${e(p.arm)}；测试${p.met?'满足':'<b>未满足</b>'} SLO）`).join('；')}。SLO 的延迟按训练口径 e2e 定义；图中横轴为本节测试口径。</p>`}
const tleg=T=>`<div class="pp-leg" role="group" aria-label="显示 / 隐藏系列">${T.arms.filter(a=>a.id!=='ROUTER').map(a=>cb('a:'+a.id,a.mk==='plan'?fpGlyph(a.plan,a.color)+' ':a.mk?`<span style="color:${a.color}">${a.mk==='pq'?'★':a.mk==='lq'?'◆':'□'} </span>`:dot(AC[a.id]),a.mk==='plan'?a.label:AN[a.id]||a.label||a.id)).join('')}${T.arms.some(a=>a.id==='ROUTER')?cb('a:ROUTER','★ ','Opt-PerQuery',`color:${AC.ROUTER}`):''}${T.lam&&T.lam.points.some(p=>p.complete)?cb('tsweep','— ','Opt-PerQuery 测试 λ 曲线（☆ 部署 λ）',`color:${AC.ROUTER}`):''}${cb('strict','┅ ','严格前沿（实心）')}${(T.slo||[]).length?SLOLEG:''}</div>`;
function lamStat(T){const L=T.lam;if(!L)return '';const c=L.points.filter(p=>p.complete),m=L.points.filter(p=>!p.complete);
 return `<p class="pp-stat">Opt-PerQuery 测试 λ 曲线（绿色连线，☆ = 部署 λ ${L.deployed_lambda}，与 ★ 同一组逐题结果）：每个 λ 下按部署路由器的测试时打分重放逐题选配置（日志中的各候选分数，代价模型为逐配置常数，故对任意 λ 精确）；每题的结果 = 已有测试 arm 中<b>同一配置</b>的该题结果（优先部署路由 arm），缺的 (配置, 题) 用 lambda-fill 定向补跑（同一旧协议：success-only-v0、semantic-rubric-v1、1 次）。测试点均为<b>单次运行</b>。已画 ${c.length} / ${L.points.length} 个 λ${m.length?`；未画（仍缺题）：${m.map(p=>`λ=${p.lambda} 缺 ${p.n-p.covered}`).join('，')}`:''}。</p>`}
const testPlots=T=>`<div class="pp-card"><h4>测试 · 质量 × Token（arm 均值）</h4>${testPlot(T,'tok')}</div><div class="pp-card"><h4>测试 · 质量 × 延迟（arm 均值）</h4>${testPlot(T,'lat')}</div>`;
$('pp-test').innerHTML=D.test.map((T,ti)=>T.pending_v7?`<h3>${e(T.name)} · 测试集 ${T.n} task × ${T.trials} trial${T.quality?' · '+e(T.quality):''}</h3><p class="pp-stat">${e(T.pending_v7)}. The training panel above is final; the deployed-arm test panel appears here once <code>${e(T.cv.dir)}/test_v7c16_report.json</code> exists.</p>${T.protocol_note?`<p class="pp-stat">${e(T.protocol_note)}</p>`:''}`:`<h3>${e(T.name)} · 测试集 ${T.n} task × ${T.trials} trial${T.quality?' · '+e(T.quality):''}</h3>${T.fixed?`<p class="pp-stat">${e(T.fixed.line)}</p>`:''}${T.protocol_note?`<p class="pp-stat">${e(T.protocol_note)}</p>`:''}${T.cv?alfTestNote(T):`<p class="pp-stat">固定 / BO arm ${T.arms.filter(a=>a.id!=='ROUTER').length} 个 + Opt-PerQuery ${T.arms.filter(a=>a.id==='ROUTER').length} 个。${T.ds==='tau2'?`Only complete ${T.trials}-trial results are drawn; incomplete Opt test arms are not plotted. `:''}路由器：<code>${e(T.router||'—')}</code></p>`}${T.pool?poolTestNote(T):''}${sloStat(T)}${lamStat(T)}<div class="pp-tl">${tleg(T)}</div>
<div class="pp-grid" data-t="${ti}">${testPlots(T)}</div>
${T.pqs?`<div class="pp-card" style="margin:10px 0"><h4>${e(T.name)} · Opt-PerQuery (BRANE) weight sweep on the test set — Q–T and Q–L curves, ${T.pqs.n_weights||11} weights (w_Q, w_L, w_T)</h4>${pqsHTML(T.pqs,'pp'+ti)}</div>`:''}
${T.cv?alfTestTable(T):`<details><summary>${e(T.name)} 测试数值（表格）</summary><div class="pp-wrap"><table><thead><tr><th>arm / 配置</th><th>题数</th><th>质量 ± SE</th><th>Token ± SE</th><th>延迟 s ± SE</th><th>配对增益 ± SE</th></tr></thead><tbody>
${T.arms.map(a=>`<tr><td>${e(a.label)}</td><td>${a.n}</td><td>${q3(a.q)} ± ${nn(q3,a.q_se)}${a.pass1!=null&&T.trials>1?` (pass^1 ${q3(a.pass1)})`:''}${a.all4?` (4-trial pass^4 ${q3(a.all4.pass4)})`:''}</td><td>${Math.round(a.tok).toLocaleString()} ± ${nn(k,a.tok_se)}</td><td>${sec(a.lat)} ± ${nn(sec,a.lat_se)}</td><td>${nn(q3,a.gain)} ± ${nn(q3,a.gain_se)}</td></tr>`).join('')}
${T.lam?T.lam.points.map(p=>`<tr><td>Opt-PerQuery λ = ${p.lambda}${p.deployed?'（部署）':''}（测试${p.complete?`；复用 ${p.reused} + 补跑 ${p.fill}`:`；<b>缺 ${p.n-p.covered} 题，未画</b>`}）</td><td>${p.covered}/${p.n}</td><td>${p.complete?q3(p.q):'—'}</td><td>${p.complete?Math.round(p.tok).toLocaleString():'—'}</td><td>${p.complete?sec(p.lat):'—'}</td></tr>`).join(''):''}
${(T.slo||[]).filter(p=>!p.none).map(p=>p.pending?`<tr><td>${p.slo} → ${e(p.cfg)}（测试未完成）</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>`:`<tr><td>${p.slo} → ${e(p.cfg)}（arm ${e(p.arm)}，复用）</td><td>${p.n}</td><td>${q3(p.q)}</td><td>${Math.round(p.tok).toLocaleString()}</td><td>${sec(p.lat)}${p.e2e!=null?`（e2e ${sec(p.e2e)}）`:''}</td></tr>`).join('')}</tbody></table></div></details>`}`).join('');
D.test.forEach((T,ti)=>{if(T.pqs)pqsDraw(T.pqs,'pp'+ti)});
const cur={};Object.keys(byDs).forEach(ds=>cur[ds]=0);
document.querySelectorAll('#pp-train .pp-tabs').forEach(t=>t.addEventListener('click',ev=>{const b=ev.target.closest('button');if(b)cur[t.dataset.ds]=+b.dataset.i}));
$('prof-pts').addEventListener('change',ev=>{const c=ev.target;if(!c.matches('input[data-k]'))return;const key=c.dataset.k;
 if(c.checked)HID.delete(key);else HID.add(key);
 document.querySelectorAll('#prof-pts input[data-k]').forEach(x=>{if(x.dataset.k===key)x.checked=c.checked});
 Object.keys(byDs).forEach(ds=>{const P=byDs[ds][cur[ds]],g=document.querySelector(`#pp-tr-${ds} .pp-grid`);
  if(g)g.innerHTML=trainGrid(P)});
 D.test.forEach((T,ti)=>{const g=document.querySelector(`#pp-test .pp-grid[data-t="${ti}"]`);if(g)g.innerHTML=testPlots(T)})});
})();</script>
<p class="small">生成时间 __TIME__ · trimem/experiments/render_profiling_points.py（只读 opt / 结果文件，未跑新 episode）· SLO：results/slo_selection_20261019/selections_v2.json · λ 扫描：router_lambda_sweep_train.json</p></section>
<!-- profiling-points:end -->"""


def collect() -> dict:
    from trimem.experiments import render_tau2_pool_points as t2

    train = [p for p in (train_panel(*t) for t in TRAIN) if p]
    for p in train:
        p["slo"], p["sweep"] = (
            train_slo(p["ds"], p["panel"]),
            train_sweep(p["ds"], p["experiment"]),
        )
    from trimem.experiments import alf_pool_page as ap
    from trimem.experiments import sci_pool_page as sp

    alf = ap.train_panel(
        regime=True
    )  # user 2026-10-22: rounds 6-21 (c16 / SemFilter v2) only
    sci = sp.train_panel()  # ScienceWorld v7 (rounds 0-6), once its profiling is paused
    if sci:  # replaces the old ScienceWorld train100 / train500 / train200 opt-DB panels (as for ALFWorld)
        train = [p for p in train if p["ds"] != "sciworld"]
        train.insert(0, sci)
    if alf:
        train.insert(0, alf)
    tau2 = t2.train_panel()
    if tau2:
        train.append(tau2)
    test_ds = ["alfworld", "sciworld", "tau2"]
    if ws_active():  # WebShop v7 (user 2026-10-23): replaces its older train200 opt-DB panel, adds the test panel
        from trimem.experiments import webshop_pool_page as wp

        ws = wp.train_panel()
        if ws:
            train = [p for p in train if p["ds"] != "webshop"]
            train.insert(
                next(
                    (i + 1 for i, p in enumerate(train) if p["ds"] == "sciworld"),
                    len(train),
                ),
                ws,
            )
        test_ds.insert(2, "webshop")
    return {
        "train": train,
        "test": [test_dataset(ds) for ds in test_ds],
    }


def _fp_js() -> str:
    from trimem.experiments import fixed_plans_page as fp

    return fp.js()


def _pqs_js() -> str:
    from trimem.experiments import perquery_sweep_page as pqs

    return pqs.js()


def main() -> int:
    data = collect()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    sec = (
        SECTION.replace("__DATA__", blob)
        .replace("__FPJS__", _fp_js())
        .replace("__PQSJS__", _pqs_js())
        .replace("__TIME__", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
    )
    with PAGE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        page = PAGE.read_text()
        if START in page:
            page = page[: page.index(START)] + sec + page[page.index(END) + len(END) :]
        elif AFTER in page:
            page = page.replace(AFTER, AFTER + "\n" + sec, 1)
        else:
            raise SystemExit("four-dataset results section not found")
        tmp = PAGE.with_suffix(".pp.tmp.html")
        from trimem.experiments.atlas_presentation import prepare_html

        tmp.write_text(prepare_html(page))
        tmp.replace(PAGE)
    from trimem.experiments.difficulty_report import write_report

    write_report()
    for p in data["train"]:
        if p.get("kind") == "pool":
            print(
                f"train {p['ds']} pool:",
                p["n_configs"],
                "configs (",
                p["n_full"],
                "full ), relaxed front:",
                [x["id"] for x in p["points"] if x["front3"]],
                "aux",
                len(p["aux"]),
            )
            continue
        print(
            "train",
            p["ds"],
            p["panel"],
            p["n_queries"],
            "q x",
            p["n_configs"],
            "configs; relaxed front tok/lat/3d:",
            sum(x["front_tok"] for x in p["points"]),
            sum(x["front_lat"] for x in p["points"]),
            sum(x["front3"] for x in p["points"]),
            "nomem",
            round(p["nomem"]["q"], 3),
        )
    for t in data["test"]:
        print(
            "test",
            t["ds"],
            [(a["id"], round(a["q"], 3)) for a in t["arms"]],
            (t.get("pool") or {}).get("status", ""),
            "SLO",
            [
                (
                    p["slo"],
                    p.get("cfg"),
                    "pending" if p.get("pending") else round(p.get("q", 0), 3),
                )
                for p in t["slo"]
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
