"""ALFWorld pool-strat200-v7 run for the qa-trajectory.html sections (user 2026-10-21).

Shared data layer of the ALFWorld blocks of render_profiling_points (train scatter + test panel),
render_atlas_results (four-dataset-results), render_cost_saving_hv (equal-cost / equal-quality + train HV curve)
and render_dataset_splits, so every ALFWorld number on the page comes from the same files. Mirrors
``render_tau2_pool_points`` / ``tau2_pool_test_table`` for the τ² pool -v5 run.

Run: results/bo_agentenv_20261016/alfworld_pool-strat200-v7__mem-sf1 (session-space-agentenv-v7, memory policy
success+failure-v1, bo_pool_driver). Profiling paused by the user after round 5 (profiling-pause-v1): 35 evaluated
configurations (30 full, 5 pruned by staged-pareto-prune-v2), each on its OWN stratified 200-task sample of the
4,066-task training pool (unpaired estimator: gain = stratified success - NoMemory's stratified success on its own
200-task sample).

* Train (``train_panel``): one point per profiled configuration of rounds 0-5, the driver's own estimates
  (``bo_pool_driver.estimates_on`` on ``observed_samples``: full sample when fully evaluated, the stage prefix when
  pruned); 3-objective relaxed / strict front over the fully evaluated configurations (``relaxed_pareto`` /
  ``pareto`` of the driver's BO points); steps = environment steps of the opt rows (mean over the observed tasks,
  and over the successful ones); reuse = quality-reuse hits among the observed rows; the 21 aux-obs-v1 points of
  the old train100 / train500 runs ride along (GP-only inputs, never in the front).
* Test (``test_table``): 200 AgentEval tasks x 1 run at concurrency 128. Primary = outer 5-fold CV
  (test_report.json ``outer_cv.test.summary``: mean ± std across the 5 fold routers / selections, SE_tasks = mean
  within-fold task SE); secondary = the deployed full-train arms (``arms``); references = the reused controlled-v1
  NoMemory / VGR test arms (``reference_arms``; concurrency 16, success-only-v0: latency not comparable). Per-task
  rows (opt rows of the test experiment + NoMemory pair files) are re-read exactly as ``bo_pool_driver.report_stage``
  and every arm's fold metrics are re-assembled and cross-checked against test_report.json (SystemExit on any
  disagreement). The per-task fold-averaged items (= routing each task with a uniformly random fold router; mean =
  the CV mean) feed the paired equal-cost / equal-quality comparison.

READ-ONLY: result files and the opt DB (read-only transaction); no episodes, no writes.
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = "alfworld"
RUN = ROOT / "results/bo_agentenv_20261016/alfworld_pool-strat200-v7__mem-sf1"
NM_TRAIN = ROOT / "results/bo_agentenv_20261016/alfworld_pool-strat200-v7/NoMemory"
SPACE_VERSION = "session-space-agentenv-v7"
TOL = 1e-6
RO = " options='-c default_transaction_read_only=on -c statement_timeout=120000'"
OPT_COLORS = {
    "Opt-OneConfig-SLO-1": "#c26a1b",
    "Opt-OneConfig-SLO-2": "#a0521a",
    "Opt-OneConfig-SLO-3": "#e09a4f",
    "Opt-PerQuery-default": "#2f8f6b",
    "Opt-PerQuery-quality": "#16624a",
    "Opt-PerQuery-tokens": "#3b6fb6",
    "Opt-PerQuery-latency": "#8a4fb0",
    "Opt-PerQuery-LimeQO-default": "#b5476b",
    "Opt-PerQuery-LimeQO-quality": "#7a2948",
    "Opt-PerQuery-LimeQO-tokens": "#d0739a",
    "Opt-PerQuery-LimeQO-latency": "#9a5b9e",
}
REF_COLORS = {"NoMemory": "#8a8f87", "VGR": "#2a6fb3"}
REF_LABELS = {"NoMemory": "No-Memory", "VGR": "VGR"}
WEIGHTS = ("default", "quality", "tokens", "latency")


def label_of(name: str) -> str:
    """Arm id -> display label (Opt-PerQuery = BRANE router, Opt-PerQuery LimeQO = the LimeQO router)."""
    if name.startswith("Opt-OneConfig-"):
        return "Opt-OneConfig " + name.removeprefix("Opt-OneConfig-")
    if name.startswith("Opt-PerQuery-LimeQO-"):
        return "Opt-PerQuery LimeQO " + name.removeprefix("Opt-PerQuery-LimeQO-")
    if name.startswith("Opt-PerQuery-"):
        return "Opt-PerQuery BRANE " + name.removeprefix("Opt-PerQuery-")
    return REF_LABELS.get(name, name)


def marker_of(name: str) -> str | None:
    """Plot marker: oc = Opt-OneConfig (square), pq = BRANE Opt-PerQuery (star), lq = LimeQO (diamond)."""
    if name.startswith("Opt-OneConfig-"):
        return "oc"
    if name.startswith("Opt-PerQuery-LimeQO-"):
        return "lq"
    if name.startswith("Opt-PerQuery-"):
        return "pq"
    return None


def _se(xs: list[float]) -> float | None:
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def _mean(xs: list[float]) -> float | None:
    return st.mean(xs) if xs else None


def _driver():
    from trimem.experiments import bo_pool_driver as d

    return d


def _connect():
    import psycopg

    return psycopg.connect(_driver().drv.OPT_DSN + RO)


def _state(run: Path | None = None, ds: str = DS) -> dict:
    run = run or RUN
    s = json.loads((run / "state.json").read_text())
    if s.get("space_version") != SPACE_VERSION or s.get("dataset") != ds:
        raise SystemExit(f"{run}/state.json is not the {SPACE_VERSION} {ds} run")
    return s


def _path(rel: str) -> Path:
    """``RUN / rel``, or its archived rounds 0-5 name after the v7 re-test archive (alf_test_v7, user 2026-10-22):
    these sections keep showing the rounds 0-5 selection / test; the re-test has its own block (alf_v7_page)."""
    from trimem.experiments import alf_test_c16 as c16

    return c16.archived(RUN, rel)


def _json(name: str, run: Path | None = None) -> dict | None:
    """``_json`` of this ALFWorld run (archived names), or plain ``run / name`` of another run (ScienceWorld)."""
    p = _path(name) if run is None or run == RUN else run / name
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ train


def train_extras(
    experiment: str, max_round: int | None = None
) -> dict[tuple[str, str], dict]:
    """(config key, query) -> environment steps, success, reuse flag of the run's train opt rows. ``max_round``
    (max-round-v1, WebShop): the derived-truncation view of every longer episode (steps n, success 0;
    episode_budget.derive_opt_row); None = as measured (ALFWorld / ScienceWorld, unchanged)."""
    d = _driver()
    with _connect() as c:
        rows = c.execute(
            "SELECT query_id, protocol->'space_config', quality, generation->'environment_steps', "
            "generation ? 'reused_from' FROM opt.session_retrieval WHERE experiment_id = %s AND split = 'train' "
            "AND status = 'evaluated'",
            (experiment,),
        ).fetchall()
    def cut(s, x) -> tuple[int | None, bool]:
        s = None if s is None else int(s)
        if max_round is not None and s is not None and s > max_round:
            return max_round, False  # not bought by turn max_round: reward 0
        return s, float(x) >= 1.0

    return {
        (d.space.key(d.space.validate(cfg)), str(q)): dict(
            zip(("steps", "ok"), cut(s, x)), reuse=bool(r)
        )
        for q, cfg, x, s, r in rows
    }


def channel_group(channel: str | None) -> str:
    """Marker group: the τ² groups (render_tau2_pool_points) + the v7 channels ``agent-gap`` (LLM analyst gap
    proposals) and ``crossover:*`` (posterior-front crossover grafts); dls-crossover stays DLS (pure)."""
    from trimem.experiments import render_tau2_pool_points as t2

    ch = channel or "default"
    if ch == "agent-gap":
        return "agent-gap"
    if ch.startswith("crossover"):
        return "crossover"
    return t2.channel_group(ch)


def step_summary(rows: list[dict]) -> dict:
    """Mean steps over the tasks with a step value, and over the successful ones (pure)."""
    s = [r["steps"] for r in rows if r.get("steps") is not None]
    ss = [r["steps"] for r in rows if r.get("steps") is not None and r.get("ok")]
    return {
        "steps": _mean(s),
        "steps_se": _se(s),
        "steps_success": _mean(ss),
        "steps_n": len(s),
    }


def train_panel(
    regime: bool = True,
    *,
    ds: str = DS,
    run: Path | None = None,
    name: str = "ALFWorld",
    qname: str = "success",
) -> dict | None:
    """Profiling-points train panel (kind 'pool', y = success rate) of the v7 run, rounds 0..last. ``regime``
    (user 2026-10-22, profiling-points panel): only the configurations of the current regime (rounds >=
    state.json catalog_regime / latency_regime from_round: controlled-c16-v1, SemFilter v2, 100 stratified tasks),
    the c16 NoMemory rerun as reference, fronts over those configurations only, the catalog selection's SLO
    picks / bounds (RUN/selection.json), no aux points. ``ds`` / ``run`` / ``name`` / ``qname``: another v7 run of
    the same space (ScienceWorld, sci_pool_page; user 2026-10-22): its own NoMemory when no latency regime is
    recorded (c16 from round 0)."""
    from trimem.experiments import render_tau2_pool_points as t2
    from trimem.optimizer import unpaired_stats as us

    d = _driver()
    run = run or RUN
    if not (run / "state.json").exists():
        return None
    state = _state(run, ds)
    pool = d.load_pool(ds)
    if pool["pool_hash"] != state["pool_hash"]:
        raise SystemExit(f"{name} pool hash differs from the v7 run's")
    exp = state["experiment"]
    r0 = 0
    nm_dir = NM_TRAIN if ds == DS else d.nomemory_dir(ds, regime=False)
    if regime:
        r0 = (state.get("catalog_regime") or state.get("latency_regime") or {}).get(
            "from_round", 0
        )
        if state.get("latency_regime"):
            nm_dir = Path(state["latency_regime"]["nomemory"]["dir"])
    mr = d.eb.view_max_round(
        state
    )  # max-round-v1 (WebShop): derived-truncation view; None elsewhere
    # nomemory-prefix-v1 (WebShop): the recorded stratified prefix; identity elsewhere (ALFWorld / ScienceWorld)
    nm_sample = d.nomemory_sample(state)
    nm, _ = d.nomemory_estimate(ds, pool, nm_sample, nm_dir, max_round=mr)
    nm_obj = d.nomemory_obj(ds, pool, state, nm_dir, max_round=mr)
    nm_ids = nm_sample["ids"]
    nm_rows = d.nomemory_rows(ds, nm_dir, nm_ids, max_round=mr)
    rows = d.opt_rows(ds, exp, state)
    obs = d.observed_samples(state)
    est = d.estimates_on(rows, obs, pool, exp)
    full = d.full_keys(state)
    if regime:
        full = {
            d.space.key(c["config"])
            for rr in state["rounds"]
            if rr["round"] >= r0
            for c in rr["configs"]
        } & full
    extra = train_extras(exp, mr)
    # semfilter-v2b (WebShop): round < v2b.from_round SemFilter configurations = old regime, drawn but kept out of
    # the fronts (as out of the router / selection catalog, bo_pool_driver.profile_table); empty elsewhere
    v2b = (state.get("semfilter") or {}).get(d.SEMFILTER_V2B_KEY)
    old_sf = (
        {
            d.space.key(c["config"])
            for rr in state["rounds"]
            if rr["round"] < v2b["from_round"]
            for c in rr["configs"]
            if c["config"]["plan"] in d.space.SEMFILTER_PLANS
        }
        if v2b
        else set()
    )
    bo = d.bo_module()
    bo_pts = [
        p
        for p in d.points(ds, pool, state, nm, state["estimator"])
        if d.space.key(p.cfg) in full - old_sf
    ]
    relaxed = {d.space.key(p.cfg) for p in bo.relaxed_pareto(bo_pts)}
    strict = {d.space.key(p.cfg) for p in bo.pareto(bo_pts)}
    sel = _json("selection.json", run) or {}
    if (
        regime
    ):  # the catalog-regime selection (in place, not the archived rounds 0-5 one)
        p_ = run / "selection.json"
        sel = json.loads(p_.read_text()) if p_.exists() else {}
        if (sel.get("catalog") or {}).get("from_round") != r0:
            sel = {}
    picks: dict[str, list[str]] = {}
    for n, s in (sel.get("selections") or {}).items():
        if s.get("opt_oneconfig"):
            picks.setdefault(s["opt_oneconfig"]["key"], []).append(n)
    pts = []
    for rr in state["rounds"]:
        if rr["round"] < r0:
            continue
        for j, c in enumerate(rr["configs"]):
            k = d.space.key(c["config"])
            e = est.get(k)
            if e is None:
                continue
            g = c.get("staging") or {}
            ids = obs[k]
            xs = [extra[(k, q)] for q in ids if (k, q) in extra]
            gn = us.diff(e["quality"], nm)
            status = (
                "full"
                if k in full
                else "pruned"
                if g.get("pruned_at")
                else "incomplete"
            )
            pts.append(
                {
                    "id": f"r{rr['round']}c{j}",
                    "key": k,
                    "round": rr["round"],
                    "channel": c.get("channel"),
                    "source": c.get("source_channel"),
                    "group": channel_group(c.get("channel")),
                    "plan": c["config"]["plan"],
                    "knobs": t2.knobs(c["config"], d.space.DISPLAY),
                    "status": status,
                    "partial": status != "full",
                    "pruned_at": g.get("pruned_at"),
                    "n": len(ids),
                    "done": len(ids),
                    "reuse": sum(x["reuse"] for x in xs),
                    "regimes": {},
                    "q": e["quality"].mean,
                    "q_se": e["quality"].se,
                    "gain": gn.mean,
                    "gain_se": gn.se,
                    "tok": e["tokens"].mean,
                    "tok_se": e["tokens"].se,
                    "lat": e["latency_ms"].mean / 1000,
                    "lat_se": e["latency_ms"].se / 1000,
                    **step_summary(xs),
                    "front3": k in relaxed,
                    "strict3": k in strict,
                    "slo": picks.get(k, []),
                    **({"old_regime": True} if k in old_sf else {}),
                }
            )
    aux = {} if regime else (state.get("aux_observations") or {})
    src_nm = {
        s["experiment"]: s.get("nomemory_quality") for s in aux.get("sources", [])
    }
    done = _json("profile_complete.json", run) or {}
    conc = state.get("concurrency") or {}
    return {
        "kind": "pool",
        "ds": ds,
        "name": name,
        "panel": "pool-strat200-v7",
        "title": f"{name} · pool-strat200-v7",
        "yk": "q",
        "qname": qname,
        "dir": str(run.relative_to(ROOT)),
        "experiment": exp,
        "space_version": state["space_version"],
        "memory_policy": state["memory_policy"],
        "pool_size": state["pool_size"],
        "n_queries": (state.get("sample_schedule") or {}).get(
            "size", state["sample_size"]
        )
        if regime
        else state["sample_size"],
        "n_configs": len(pts),
        "n_full": sum(p["status"] == "full" for p in pts),
        "rounds": len(state["rounds"]),
        "paused_after_round": done.get("last_round")
        if done.get("paused_by_user")
        else None,
        "budget_used": done.get("budget_used"),
        "budget_episodes": done.get("budget_episodes"),
        "relax_z": bo.RELAX_Z,
        "noise_floor": sel.get("noise_floor_unpaired"),
        "nomem": {
            "q": nm.mean,
            "q_se": nm.se,
            "gain": 0.0,
            "tok": nm_obj.tokens if nm_obj else None,
            "lat": nm_obj.latency / 1000 if nm_obj else None,
            **step_summary(
                [
                    {"steps": r.get("steps"), "ok": r["quality"] >= 1.0}
                    for r in nm_rows.values()
                ]
            ),
        },
        "points": pts,
        "aux": [
            {
                "label": a["label"],
                "plan": a["config"]["plan"],
                "experiment": a["experiment"],
                "gain": a["gain"],
                "gain_se": a["gain_se"],
                # aux points are drawn at their OWN success rate (source NoMemory + paired gain)
                "q": (src_nm.get(a["experiment"]) or 0.0) + a["gain"],
                "tok": a["tokens"],
                "lat": a["latency_ms"] / 1000,
            }
            for a in aux.get("points", [])
        ],
        "aux_sources": aux.get("sources", []),
        "aux_from_round": aux.get("from_round"),
        "concurrency": {"profiling_total": 16, "changes": []}
        if regime  # v7 regime: controlled c16 throughout (the c128 -> c16 change predates these rounds)
        else {
            "profiling_total": conc.get("profiling_total"),
            "changes": [
                {
                    "from": ch["from"].get("profiling_total"),
                    "to": ch["to"].get("profiling_total"),
                    "at": ch.get("at"),
                    "round": ch.get("from_round"),
                }
                for ch in conc.get("changes", [])
            ],
        },
        "round_latency": round_latency(pts),
        "reuse_from_round": (state.get("quality_reuse") or {}).get("from_round"),
        "slo_bounds": sel.get("bounds"),
        "regime": None
        if not regime
        else {
            "from_round": r0,
            "last_round": state["rounds"][-1]["round"],
            "runtime": (state.get("runtime_profile") or {}).get("name"),
            "concurrency": (state.get("runtime_profile") or {}).get(
                "total_concurrency"
            ),
            "semfilter": (state.get("semfilter") or {}).get("protocol"),
            "sample_size": (state.get("sample_schedule") or {}).get("size"),
            "nomemory_dir": str(nm_dir.relative_to(ROOT)),
            "nomemory_n": len(nm_ids),
            "nomemory_label": "c16 rerun"
            if state.get("latency_regime")
            else "own NoMemory, c16",
            "excluded_configs": sum(
                len(rr["configs"]) for rr in state["rounds"] if rr["round"] < r0
            ),
            "selection_frozen": bool(sel),
            "replay_estimator": sel.get("replay_estimator"),
        },
    }


def round_latency(pts: list[dict]) -> list[dict]:
    """Per round: median / min / max latency (s) of its fully evaluated configurations (pure)."""
    out = []
    for r in sorted({p["round"] for p in pts}):
        lat = sorted(p["lat"] for p in pts if p["round"] == r and not p["partial"])
        if lat:
            out.append(
                {
                    "round": r,
                    "n": len(lat),
                    "median": st.median(lat),
                    "min": lat[0],
                    "max": lat[-1],
                }
            )
    return out


# ------------------------------------------------------------------ test


def _test_rows(
    tab: dict,
    cv: dict | None,
    pairs_dir: Path | None = None,
    span: bool = False,
    ds: str = DS,
) -> dict[tuple[str, str], dict]:
    """(config key, task) -> {quality, tokens, e2e_ms, steps, steps_success, tsem, ret_ms, reuse}: opt rows of the
    test experiment (``tab['experiment']``) + the NoMemory router-candidate pair files of ``pairs_dir`` (default
    the c128 ``test/pairs``; as ``bo_pool_driver.report_stage``). ``span``: ``e2e_ms`` := LLM-call span + retrieval
    ms (``alf_test_c16.call_spans`` of each configuration's pair directory; the old controlled-v1 NoMemory's
    latency definition), the wall value kept as ``e2e_wall_ms``; rows without a call record are left out."""
    from trimem.experiments import alf_test_c16 as c16

    d = _driver()
    pdir0 = pairs_dir or RUN / "test" / "pairs"
    spans: dict[str, dict[str, float]] = {}

    def spanned(row: dict, cfg: dict, q: str) -> dict | None:
        if not span:
            return row
        n = d.cfg_name(cfg)
        if n not in spans:
            spans[n] = c16.call_spans(pdir0 / n, ds)
        sp = spans[n].get(q)
        return (
            None
            if sp is None
            else {**row, "e2e_wall_ms": row["e2e_ms"], "e2e_ms": sp + row["ret_ms"]}
        )

    with _connect() as c:
        raw = c.execute(
            "SELECT query_id, protocol->'space_config', quality, total_tokens, e2e_ms, generation FROM "
            "opt.session_retrieval WHERE experiment_id = %s AND split = 'test' AND status = 'evaluated'",
            (tab["experiment"],),
        ).fetchall()
    rows = {}
    for q, cfg, x, tok, e2e, g in raw:
        g = g or {}
        row = {
            **d.with_steps(
                ds,
                {"quality": float(x), "tokens": float(tok), "e2e_ms": float(e2e)},
                d.episode_steps(ds, g, None),
            ),
            "tsem": float(g.get("semantic_tokens") or 0),
            "ret_ms": float(g.get("retrieval_ms") or 0),
            "reuse": "reused_from" in g,
        }
        if (row := spanned(row, cfg, q)) is not None:
            rows[(d.space.key(cfg), q)] = row
    nm_key = d.space.key(d.space.NOMEM)
    all_arms = {**tab, "arms": {**tab["arms"], **(cv or {}).get("arms", {})}}
    nm_pairs = d.pairs_of(all_arms).get(nm_key, [])
    if nm_pairs:
        pdir = pdir0 / d.cfg_name(d.space.NOMEM)
        for q, r in d.nomemory_rows(ds, pdir, nm_pairs).items():
            row = spanned(
                {
                    **{k: v for k, v in r.items() if k != "cost_source"},
                    "tsem": 0.0,
                    "ret_ms": 0.0,
                    "reuse": False,
                },
                d.space.NOMEM,
                q,
            )
            if row is not None:
                rows[(nm_key, q)] = row
    return rows


def arm_items(routes: dict[str, dict], rows: dict, key) -> dict[str, dict]:
    """Per task of one routed arm: q, tok, lat (s, + router overhead), tsem, ret (s, + overhead), steps (pure)."""
    out = {}
    for q, r in routes.items():
        x = rows.get((key(r["config"]), q))
        if x is None:
            continue
        oh = float(r.get("overhead_ms") or 0)
        out[q] = {
            "q": x["quality"],
            "tok": x["tokens"],
            "lat": (x["e2e_ms"] + oh) / 1000,
            "tsem": x["tsem"],
            "ret": (x["ret_ms"] + oh) / 1000,
            "steps": x.get("steps"),
            "cfg": key(r["config"]),
        }
    return out


def fold_average(folds: list[dict[str, dict]]) -> dict[str, dict]:
    """Per task: mean over the folds of q / tok / lat / tsem / ret (tasks covered by every fold; pure). Equivalent
    to routing each task with a uniformly random fold router; the arm mean equals the across-fold mean."""
    common = set.intersection(*(set(f) for f in folds)) if folds else set()
    out = {}
    for q in sorted(common):
        out[q] = {
            k: st.mean(f[q][k] for f in folds)
            for k in ("q", "tok", "lat", "tsem", "ret")
        }
        s = [f[q]["steps"] for f in folds if f[q].get("steps") is not None]
        out[q]["steps"] = st.mean(s) if s else None
    return out


def _check(name: str, what: str, mine: float | None, theirs: float | None) -> None:
    if (
        mine is None
        or theirs is None
        or not math.isclose(mine, theirs, rel_tol=TOL, abs_tol=TOL)
    ):
        raise SystemExit(
            f"ALFWorld v7 {name}: {what} {mine} disagrees with test_report.json ({theirs})"
        )


def _ms(x: dict | None) -> tuple[float | None, float | None]:
    return (None, None) if not x else (x.get("mean"), x.get("std"))


C16_REPORT = "test_c16_report.json"  # concurrency-16 rerun (alf_test_c16; user 2026-10-21): PRIMARY when present


@lru_cache(maxsize=4)
def test_table_run(run: str) -> dict:
    """One test run's table: CV arms (primary), deployed arms, references + per-task items, cross-checked against
    its report. ``run`` = "c128" (test/, test_report.json; references = the old controlled-v1 arms) or "c16"
    (test_c16/, test_c16_report.json, LimeQO not rerun; references = the NoMemory / VGR reruns of the same c16
    session; gain vs the NoMemory rerun)."""
    from trimem.experiments import pool_cv_report as pcr
    from trimem.optimizer.query_router import outer_cv as ocv

    d = _driver()
    if (
        run == "v7"
    ):  # the v7 re-test (alf_test_v7; user 2026-10-22): in place, never archived
        from trimem.experiments import alf_test_v7 as v7

        p_ = RUN / v7.REPORT
        rep = json.loads(p_.read_text()) if p_.exists() else None
    else:
        rep = _json(C16_REPORT if run == "c16" else "test_report.json")
    out: dict = {
        "dir": str(RUN.relative_to(ROOT)),
        "run": run,
        "arms": [],
        "deployed": [],
        "refs": [],
    }
    if rep is None or not (_path("test/routing.json")).exists():
        out["status"] = (
            f"test stage not finished (no {C16_REPORT if run == 'c16' else 'test_report.json'})"
        )
        return out
    if run == "v7":
        tab, cv = v7.routing(RUN)
        tab = {**tab, "experiment": rep["experiment"]}
        rows = _test_rows(tab, cv, RUN / v7.SUBDIR / "pairs", span=True)
        # pairs reused from the earlier c16 test (SemFilter-free plans only; alf_test_v7.split_pairs)
        old = _test_rows(
            {**tab, "experiment": rep["reuse_source"]},
            cv,
            RUN / c16_subdir() / "pairs",
            span=True,
        )
        rows |= {
            k: x
            for k, x in old.items()
            if k not in rows and v7.reusable_plan(json.loads(k[0]))
        }
        nm_dir = Path(rep["baseline"]["dir"])
    elif run == "c16":
        from trimem.experiments import alf_test_c16 as c16

        tab, cv = c16.routing(RUN)
        tab = {**tab, "experiment": rep["experiment"]}
        rows = _test_rows(tab, cv, RUN / c16.SUBDIR / "pairs", span=True)
        nm_dir = Path(rep["baseline"]["dir"])
    else:
        tab = json.loads((_path("test/routing.json")).read_text())
        cvp = _path("test/routing_cv.json")
        cv = json.loads(cvp.read_text()) if cvp.exists() else None
        if cv is None:
            raise SystemExit("ALFWorld v7: outer-CV routing missing")
        rows = _test_rows(tab, cv)
        nm_dir = d.drv.CTRL / f"{DS}_NoMemory"
    if "outer_cv" not in rep:
        raise SystemExit("ALFWorld v7: outer-CV report missing")
    key = d.space.key
    nm_q = {
        i: d.quality_of(DS, json.load(open(f))["reward"])
        for i, f in d.drv.item_files(nm_dir, DS).items()
    }
    ids = d.test_ids(DS)
    summ = rep["outer_cv"]["test"]["summary"]
    per_fold = rep["outer_cv"]["test"]["per_fold"]
    # c16: plotted / paired latency = the LLM-call span definition (the only one the old NoMemory reference has;
    # user 2026-10-21), the wall value rides along as lat_wall
    span_ = run in ("c16", "v7")
    span_fold = rep["outer_cv"]["test"].get("per_fold_span") if span_ else None
    folds = ocv.fold_tables(cv["arms"])
    # re-assemble every fold arm from the rows and cross-check with the report
    for f, t in folds.items():
        got = d.assemble(t, rows, nm_q)
        for name, v in per_fold[str(f)].items():
            for m in (
                "quality",
                "tokens",
                "latency_s",
                "paired_gain_vs_nomemory",
                "steps",
            ):
                theirs = (
                    span_fold[str(f)][name].get(m)
                    if span_fold and m == "latency_s"
                    else v.get(m)
                )
                _check(f"fold {f} {name}", m, got[name].get(m), theirs)
    for name, s in summ.items():
        fi = [
            arm_items(folds[f]["arms"][name]["routes"], rows, key)
            for f in sorted(folds)
        ]
        if any(len(x) != len(ids) for x in fi):
            raise SystemExit(f"ALFWorld v7 {name}: incomplete fold arm")
        items = fold_average(fi)
        q, q_std = _ms(s["quality"])
        tok, tok_std = _ms(s["tokens"])
        lat, lat_std = _ms(s["latency_span_s"] if span_ else s["latency_s"])
        lat_wall, lat_wall_std = _ms(s["latency_s"])
        gain, gain_std = _ms(s["gain"])
        steps, steps_std = _ms(s.get("steps"))
        ss, ss_std = _ms(s.get("steps_success"))
        _check(
            name, "fold-averaged success", st.mean(x["q"] for x in items.values()), q
        )
        _check(
            name, "fold-averaged tokens", st.mean(x["tok"] for x in items.values()), tok
        )
        _check(
            name,
            "fold-averaged latency",
            st.mean(x["lat"] for x in items.values()),
            lat,
        )
        se = s.get("se_tasks_mean") or {}
        dep = rep["arms"].get(name) or {}
        out["arms"].append(
            {
                "id": name,
                "label": label_of(name),
                "kind": "oneconfig" if name.startswith("Opt-OneConfig") else "perquery",
                "router": "LimeQO"
                if "LimeQO" in name
                else "BRANE"
                if name.startswith("Opt-PerQuery")
                else None,
                "mk": marker_of(name),
                "color": OPT_COLORS.get(name, "#2f8f6b"),
                "weights": dep.get("weights"),
                "slo": dep.get("slo"),
                "folds": s.get("folds"),
                "n": len(items),
                "q": q,
                "q_std": q_std,
                "tok": tok,
                "tok_std": tok_std,
                "lat": lat,
                "lat_std": lat_std,
                "lat_wall": lat_wall,
                "lat_wall_std": lat_wall_std,
                "gain": gain,
                "gain_std": gain_std,
                "steps": steps,
                "steps_std": steps_std,
                "steps_success": ss,
                "steps_success_std": ss_std,
                "se_tasks": {
                    "q": se.get("quality"),
                    "gain": se.get("gain"),
                    "steps": se.get("steps"),
                },
                "tsem": st.mean(x["tsem"] for x in items.values()),
                "ret": st.mean(x["ret"] for x in items.values()),
                "distinct_per_fold": [
                    v[name].get("distinct_configs") for _, v in sorted(per_fold.items())
                ],
                "items": items,
            }
        )
    # deployed (full-train) arms
    got = d.assemble(tab, rows, nm_q)
    for name, v in rep["arms"].items():
        for m in ("quality", "tokens", "latency_s", "paired_gain_vs_nomemory", "steps"):
            mm = "latency_span_s" if span_ and m == "latency_s" else m
            _check(f"deployed {name}", m, got[name].get(m), v.get(mm))
        out["deployed"].append(
            {
                "id": name,
                "label": label_of(name) + " (deployed)",
                "q": v["quality"],
                "q_se": v["quality_se"],
                "tok": v["tokens"],
                "lat": v["latency_span_s"] if span_ else v["latency_s"],
                "lat_wall": v["latency_s"],
                "gain": v["paired_gain_vs_nomemory"],
                "gain_se": v["paired_gain_se"],
                "steps": v.get("steps"),
                "steps_se": v.get("steps_se"),
                "steps_success": v.get("steps_success"),
                "distinct": v.get("distinct_configs"),
                "weights": v.get("weights"),
                "covered": v.get("covered"),
                "n": v.get("n"),
            }
        )
    if (
        run == "v7"
    ):  # deployed arms = PRIMARY (user 2026-10-22), full arm schema + the CV numbers as ``cv``
        cvd = {a["id"]: a for a in out["arms"]}
        prim = []
        for x in out["deployed"]:
            name = x["id"]
            items = arm_items(tab["arms"][name]["routes"], rows, key)
            c = cvd.get(name) or {}
            prim.append(
                {
                    "id": name,
                    "label": label_of(name),
                    "kind": "oneconfig"
                    if name.startswith("Opt-OneConfig")
                    else "perquery",
                    "router": "BRANE" if name.startswith("Opt-PerQuery") else None,
                    "mk": marker_of(name),
                    "color": OPT_COLORS.get(name, "#2f8f6b"),
                    "weights": rep["arms"][name].get("weights"),
                    "slo": rep["arms"][name].get("slo"),
                    "folds": None,
                    "n": len(items),
                    "q": x["q"],
                    "q_std": None,
                    "q_se": x["q_se"],
                    "tok": x["tok"],
                    "tok_std": None,
                    "lat": x["lat"],
                    "lat_std": None,
                    "lat_wall": x["lat_wall"],
                    "lat_wall_std": None,
                    "gain": x["gain"],
                    "gain_std": None,
                    "gain_se": x["gain_se"],
                    "steps": x["steps"],
                    "steps_std": None,
                    "steps_se": x["steps_se"],
                    "steps_success": x["steps_success"],
                    "steps_success_std": None,
                    "se_tasks": {
                        "q": x["q_se"],
                        "gain": x["gain_se"],
                        "steps": x["steps_se"],
                    },
                    "tsem": st.mean(i["tsem"] for i in items.values()),
                    "ret": st.mean(i["ret"] for i in items.values()),
                    "distinct": x["distinct"],
                    "items": items,
                    "cv": {
                        k: c.get(k)
                        for k in (
                            "folds",
                            "q",
                            "q_std",
                            "tok",
                            "tok_std",
                            "lat",
                            "lat_std",
                            "lat_wall",
                            "lat_wall_std",
                            "gain",
                            "gain_std",
                            "steps",
                            "steps_std",
                            "steps_success",
                            "steps_success_std",
                            "se_tasks",
                        )
                    }
                    if c
                    else None,
                }
            )
        out["deployed_arms"] = prim
    out["routing_deployed"] = routing_counts(
        tab["arms"]["Opt-PerQuery-default"]["routes"], rows, key
    )
    # references (single run) = the reused controlled-v1 arms (c16: latency in the span definition)
    for name, r in _ref_metrics(rep, run).items():
        items = reference_items(name, ids, span=span_)
        _check(name, "success", st.mean(x["q"] for x in items.values()), r["quality"])
        _check(name, "tokens", st.mean(x["tok"] for x in items.values()), r["tokens"])
        _check(
            name, "latency", st.mean(x["lat"] for x in items.values()), r["latency_s"]
        )
        out["refs"].append(
            {
                "id": name,
                "label": REF_LABELS.get(name, name),
                "kind": "ref",
                "mk": None,
                "color": REF_COLORS.get(name, "#888"),
                "n": r["n"],
                "q": r["quality"],
                "q_se": r["quality_se"],
                "q_std": None,
                "tok": r["tokens"],
                "tok_std": None,
                "lat": r["latency_s"],
                "lat_std": None,
                "lat_wall": r.get("latency_wall_s"),
                "lat_definition": r.get("lat_definition"),
                "gain": r["paired_gain_vs_nomemory"],
                "gain_std": None,
                "gain_se": r["paired_gain_se"],
                "steps": r.get("steps"),
                "steps_std": None,
                "steps_success": r.get("steps_success"),
                "se_tasks": {
                    "q": r["quality_se"],
                    "gain": r["paired_gain_se"],
                    "steps": r.get("steps_se"),
                },
                "tsem": st.mean(x["tsem"] for x in items.values()),
                "ret": st.mean(x["ret"] for x in items.values()),
                "protocol": r.get("protocol"),
                "items": items,
            }
        )
    names = [a["id"] for a in out["arms"]]
    out["comparison"] = [
        {"a": a, "b": b, "diff": pcr.paired_diff(per_fold, a, b)}
        for a, b in comparisons(names)
    ]
    ex = (tab.get("experiment") or "").split(":")
    test_reuse = [x for k, x in rows.items() if k[0] != d.space.key(d.space.NOMEM)]
    out.update(
        status="complete",
        experiment=rep.get("experiment"),
        test_version=ex[2] if len(ex) > 2 else None,
        generated=rep.get("generated"),
        concurrency=rep.get("concurrency", rep.get("test_concurrency")),
        ref_concurrency=rep.get("concurrency", rep.get("reused_arms_concurrency")),
        pairs=rep.get("pairs"),
        memory_pairs=len(test_reuse),
        reuse_hits=sum(x["reuse"] for x in test_reuse),
        cv_protocol=rep["outer_cv"].get("protocol"),
        k=len(folds),
        fold_sizes=(rep["outer_cv"].get("folds") or {}).get("sizes"),
        n_test=len(ids),
        baseline=rep.get("baseline"),
        notes=rep.get("notes", []),
    )
    return out


def _ref_metrics(rep: dict, run: str) -> dict[str, dict]:
    """Reference arms in ``reference_arms`` (bo_pool_driver) fields. c16 (user 2026-10-21): the OLD controlled-v1
    runs (the reruns were stopped) with latency in the LLM-call span definition (``latency_s``; NoMemory has no
    other) and VGR's wall value as ``latency_wall_s``; gain vs the old NoMemory."""
    refs = rep.get("reference_arms") or {}
    if run not in ("c16", "v7"):
        return refs
    out = {}
    for name, r in refs.items():
        old = r if run == "v7" else (r.get("old") or {})
        sp, w = old.get("span"), old.get("wall")
        if not sp or sp.get("covered") != sp.get("n"):
            continue
        out[name] = {
            **sp,
            "latency_wall_s": (w or {}).get("latency_s"),
            "lat_definition": "LLM-call span"
            if not w
            else "LLM-call span (wall also shown)",
            "protocol": f"controlled-v1 {name} (2026-10-15, concurrency 16; original run)",
        }
    return out


C128_KEYS = (
    "q",
    "q_std",
    "tok",
    "tok_std",
    "lat",
    "lat_std",
    "gain",
    "gain_std",
    "steps",
    "steps_std",
)


def merge_runs(c16: dict, c128: dict) -> dict:
    """c16 table (primary) + the c128 values of each arm as ``c128`` (secondary column), the c128-only LimeQO arms
    (``c128_only``: table rows only, labelled) and the old references as each reference's ``old``. Pure."""
    by = {a["id"]: a for a in c128.get("arms", [])}
    dep = {a["id"]: a for a in c128.get("deployed", [])}
    have = {a["id"] for a in c16["arms"]}
    lime = [
        {**a, "label": a["label"] + " (c128 only)", "c128_only": True}
        for a in c128.get("arms", [])
        if a["id"] not in have
    ]
    return {
        **c16,
        "arms": [
            {
                **a,
                "c128": {k: by[a["id"]].get(k) for k in C128_KEYS}
                if a["id"] in by
                else None,
            }
            for a in c16["arms"]
        ],
        "deployed": [
            {
                **a,
                "c128": {"q": dep[a["id"]]["q"], "lat": dep[a["id"]]["lat"]}
                if a["id"] in dep
                else None,
            }
            for a in c16["deployed"]
        ]
        + [
            {**a, "label": a["label"] + " (c128 only)", "c128_only": True}
            for a in c128.get("deployed", [])
            if a["id"] not in {x["id"] for x in c16["deployed"]}
        ],
        "c128_only": lime,
        "comparison_c128": [
            c for c in c128.get("comparison", []) if "LimeQO" in c["a"] + c["b"]
        ],
        "c128_meta": {
            k: c128.get(k)
            for k in (
                "concurrency",
                "ref_concurrency",
                "memory_pairs",
                "reuse_hits",
                "pairs",
                "generated",
            )
        },
    }


def c16_subdir() -> str:
    from trimem.experiments import alf_test_c16 as c16

    return c16.SUBDIR


def test_table() -> dict:
    """The page's ALFWorld test table. v7 re-test (user 2026-10-22; test_v7c16_report.json present): the DEPLOYED
    full-train arms are PRIMARY (``arms``; quality ± SE across the 200 tasks, each arm's CV mean ± std as
    ``cv``), the outer-CV arms secondary (``cv_arms``); no c128 / LimeQO / rounds 0-5 data. Before it: the c16
    rerun merged with the c128 run (archived rounds 0-5 selection)."""
    v = test_table_run("v7")
    if v.get("status") == "complete":
        return {
            **v,
            "arms": v["deployed_arms"],
            "cv_arms": v["arms"],
            "primary": "deployed",
            "run": "v7",
        }
    c128 = test_table_run("c128")
    c16 = test_table_run("c16")
    if c16.get("status") != "complete" or c128.get("status") != "complete":
        return c128
    return merge_runs(c16, c128)


def comparisons(names: list[str]) -> list[tuple[str, str]]:
    """BRANE vs LimeQO per weight, then each router arm vs Opt-OneConfig SLO-1 (= the fixed r2c3 choice of most
    folds), in report.md's order (pure)."""
    out = [
        (f"Opt-PerQuery-LimeQO-{w}", f"Opt-PerQuery-{w}")
        for w in WEIGHTS
        if f"Opt-PerQuery-LimeQO-{w}" in names and f"Opt-PerQuery-{w}" in names
    ]
    for slo in ("SLO-1", "SLO-2"):
        b = f"Opt-OneConfig-{slo}"
        if b not in names:
            continue
        for w in WEIGHTS:
            for a in (f"Opt-PerQuery-{w}", f"Opt-PerQuery-LimeQO-{w}"):
                if a in names:
                    out.append((a, b))
    return out


def reference_items(
    name: str,
    ids: list[str],
    span: bool = False,
    ds: str = DS,
    max_round: int | None = None,
) -> dict[str, dict]:
    """Per-task rows of a reused controlled-v1 reference arm (as ``bo_pool_driver.reference_arms``). ``span``: VGR
    latency = LLM-call span + retrieval (``alf_test_c16.call_spans``); NoMemory has the span definition only.
    ``max_round`` (max-round-v1, WebShop; as ``webshop_test_v7.references``): the 30-turn references in the
    derived-truncation view (rows and spans); None = as measured (ALFWorld / ScienceWorld, unchanged)."""
    from trimem.experiments import alf_test_c16 as c16

    d = _driver()
    rdir = d.drv.CTRL / f"{ds}_{name}"
    if name == "NoMemory":
        rows = d.nomemory_rows(ds, rdir, ids, max_round=max_round)
        return {
            q: {
                "q": r["quality"],
                "tok": r["tokens"],
                "lat": r["e2e_ms"] / 1000,
                "tsem": 0.0,
                "ret": 0.0,
                "steps": r.get("steps"),
            }
            for q, r in rows.items()
            if "tokens" in r
        }
    with _connect() as c:
        raw = c.execute(
            "SELECT query_id, quality, total_tokens, e2e_ms, generation FROM opt.session_retrieval "
            "WHERE experiment_id = %s AND split = 'test' AND status = 'evaluated'",
            (f"{ds}:agenteval200:controlled-v1:{name}",),
        ).fetchall()
    files = d.drv.item_files(rdir, ds)
    spans = c16.call_spans(rdir, ds) if span else {}
    if span and max_round is not None:
        spans |= d.eb.truncated_spans(rdir, ds, max_round)
    want = set(ids)
    out = {}
    for q, x, t, e, g in raw:
        if q not in want:
            continue
        if max_round is not None:
            g = {**(g or {})}
            if "environment_steps" not in g and files.get(q):
                g["result_file"] = str(files[q])
            r = d.eb.derive_opt_row(
                {
                    "query_id": q,
                    "quality": x,
                    "total_tokens": t,
                    "e2e_ms": e,
                    "generation": g,
                },
                max_round,
            )
            x, t, e, g = r["quality"], r["total_tokens"], r["e2e_ms"], r["generation"]
        ret = float((g or {}).get("retrieval_ms") or 0)
        if span and q not in spans:
            continue
        out[q] = {
            "q": float(x),
            "tok": float(t),
            "lat": (spans[q] + ret if span else float(e)) / 1000,
            "tsem": float((g or {}).get("semantic_tokens") or 0),
            "ret": ret / 1000,
            "steps": d.episode_steps(ds, g, files.get(q)),
        }
    return out


def routing_counts(routes: dict[str, dict], rows: dict, key) -> list[dict]:
    """Per configuration of one routed arm: tasks routed there and their mean success (pure on its inputs)."""
    from trimem.experiments.render_atlas_results import PLAN_NAME, POLICY_SHORT

    by: dict[str, list[float]] = {}
    for q, r in routes.items():
        x = rows.get((key(r["config"]), q))
        if x is not None:
            by.setdefault(key(r["config"]), []).append(x["quality"])
    out = []
    for k, qs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cfg = json.loads(k)
        nomem = cfg["plan"] == "NoMemory"
        out.append(
            {
                "plan": cfg["plan"],
                "plan_name": "No-Memory"
                if nomem
                else PLAN_NAME.get(cfg["plan"], cfg["plan"]),
                "n": len(qs),
                "quality": st.mean(qs),
                "config": "no memory"
                if nomem
                else f"s{cfg['seed_count']} d{cfg['depth']} f{cfg['fanout']} k{cfg['top_k']} "
                + POLICY_SHORT.get(
                    cfg.get("outcome_policy", ""), cfg.get("outcome_policy", "")
                ),
            }
        )
    return out


def strip(t: dict) -> dict:
    """Page payload: the table without per-task items."""
    drop = lambda a: {k: v for k, v in a.items() if k != "items"}  # noqa: E731
    return {
        **t,
        "arms": [drop(a) for a in t.get("arms", [])],
        "refs": [drop(a) for a in t.get("refs", [])],
    }


# ------------------------------------------------------------------ train HV run (cost-saving-hv B)


def hv_run(
    *, ds: str = DS, run: Path | None = None, name: str = "ALFWorld"
) -> dict | None:
    """The v7 profiling run in render_cost_saving_hv's bo_run format. Points = the fully evaluated configurations
    (selection.json points: stratified success / tokens / latency over their own 200-task samples). Per round:
    ``evaluated`` = configuration episodes run (staging episodes, pruned configurations' prefixes included),
    ``reused`` = quality-reuse hits among them (opt rows with ``reused_from``: copied outcome, no new LLM
    generation), x = cumulative executed (= evaluated - reused). NoMemory and the noise repeat are excluded.
    ``ds`` / ``run`` / ``name``: another v7 run (ScienceWorld); the budget cross-check is ALFWorld-only (the
    ScienceWorld budget also counts superseded SemFilter v1 / beyond-prefix episodes: ``budget_used`` is reported)."""
    run = run or RUN
    if not (run / "state.json").exists():
        return None
    d = _driver()
    state = _state(run, ds)
    # v7 re-test (user 2026-10-22): the catalog selection in place (rounds >= from_round), never the archived one
    r0 = (state.get("catalog_regime") or {}).get("from_round", 0)
    p_ = run / "selection.json"
    sel = json.loads(p_.read_text()) if p_.exists() else None
    if sel is None or (sel.get("catalog") or {}).get("from_round", 0) != r0:
        return None
    reuse_by_key = Counter(
        k for (k, _), x in train_extras(state["experiment"]).items() if x["reuse"]
    )
    origin = {}
    pts = []
    for p in sel["points"]:
        r = int(p["label"].split("c", 1)[0].removeprefix("r"))
        pts.append(
            {
                "round": r,
                "label": p["label"],
                "q": p["q"],
                "tok": p["tok"],
                "lat": p["lat"],
            }
        )
    rounds, cum, cum_eval = [], 0, 0
    for rr in state["rounds"]:
        if rr["round"] < r0:
            continue
        stg = [
            (d.space.key(c["config"]), c.get("staging") or {}) for c in rr["configs"]
        ]
        ev = sum(max(g.get("episodes", 0), g.get("stage_reached", 0)) for _, g in stg)
        reused = sum(reuse_by_key[k] for k, _ in stg)
        cum += ev - reused
        cum_eval += ev
        new = sum(p["round"] == rr["round"] for p in pts)
        origin[rr["round"]] = new
        rounds.append(
            {
                "round": rr["round"],
                "proposed": len(rr["configs"]),
                "new": new,
                "incomplete": len(rr["configs"]) - new,
                "channels": sorted({c.get("channel") or "" for c in rr["configs"]}),
                "mode": "initial" if rr["round"] == 0 else "",
                "evaluated": ev,
                "reused": reused,
                "episodes_cum": cum,
                "evaluated_cum": cum_eval,
            }
        )
    done = _json("profile_complete.json", run) or {}
    if (
        ds == DS
        and r0 == 0
        and done.get("budget_used") is not None
        and cum_eval != done["budget_used"]
    ):
        raise SystemExit(
            f"ALFWorld v7: evaluated episodes {cum_eval} != budget_used {done['budget_used']}"
        )
    return {
        "ds": ds,
        "name": name,
        "panel": "pool-strat200-v7",
        "experiment": state["experiment"],
        "n_queries": (state.get("sample_schedule") or {}).get(
            "size", state["sample_size"]
        )
        if r0
        else state["sample_size"],
        "from_round": r0,
        "rounds": rounds,
        "points": pts,
        "reused_total": sum(r["reused"] for r in rounds),
        "evaluated_total": cum_eval,
        "budget_episodes": done.get("budget_episodes"),
        "budget_used": done.get("budget_used"),
        "paused_after_round": done.get("last_round")
        if done.get("paused_by_user")
        else None,
    }


def hv_note(R: dict | None) -> str:
    """English note on how the ALFWorld v7 HV curve counts episodes (rounds >= the catalog start round)."""
    if not R:
        return ""
    first, last = R["rounds"][0], R["rounds"][-1]
    pruned = sum(r["incomplete"] for r in R["rounds"])
    return (
        f"<b>ALFWorld (pool-strat200-v7, rounds {first['round']}–{last['round']})</b>: points = the "
        f"{len(R['points'])} fully evaluated configurations profiled under the controlled c16 runtime with SemFilter "
        f"v2 (each on its own stratified {R['n_queries']}-task sample of the 4,066-task pool; unpaired means); the "
        f"{pruned} configurations pruned at 50 add episodes but no point. Rounds 0–{first['round'] - 1} (c128 "
        "latency, SemFilter v1) are excluded. x = cumulative profiling episodes executed from round "
        f"{first['round']}: {R['evaluated_total']:,} evaluated − {R['reused_total']:,} quality-reuse hits (identical "
        f"task and injected memory, outcome copied) = {last['episodes_cum']:,}. NoMemory and the noise repeat are "
        "excluded."
    )


# ------------------------------------------------------------------ dataset-splits text


def splits_text() -> tuple[str, str]:
    """(profiling-panel cell, test cell) of the ALFWorld dataset-splits row (English): the v7 regime (user
    2026-10-22): catalog rounds 6-21 and the v7 re-test."""
    from trimem.experiments import alf_test_v7 as v7

    st_ = _state()
    reg = st_.get("catalog_regime") or {}
    r0 = reg.get("from_round", 6)
    last = st_["rounds"][-1]["round"]
    new = [c for r in st_["rounds"] if r["round"] >= r0 for c in r["configs"]]
    full = len(
        _driver().full_keys({"rounds": [r for r in st_["rounds"] if r["round"] >= r0]})
    )
    size = (st_.get("sample_schedule") or {}).get("size", 100)
    prof = (
        "<b>Pool-strat200-v7 profiling</b> (session-space-agentenv-v7, memory policy success+failure-v1): pool = "
        "every non-test ALFWorld task (4,066: train + valid_seen + valid_unseen, the 200 AgentEval items excluded); "
        f"shown and used: rounds {r0}–{last} under the controlled c16 runtime (2 halves × 8 processes) with "
        f"SemFilter v2 — {len(new)} configurations, each on its <b>own</b> task-type-stratified sample of {size} "
        f"tasks × 1 run (unpaired estimator vs the c16 No-Memory rerun on its own 200-task sample), staged 50 → "
        f"{size} with 3-D Pareto pruning: {full} fully evaluated, {len(new) - full} pruned. Router / selection: "
        "outer 5-fold CV over the training queries.[37]"
    )
    rep_p = RUN / v7.REPORT
    test = (
        "<br><b>Pool-v7 re-test</b>: the 200 AgentEval items × 1 run under the same controlled c16 runtime, "
        "quality reuse off; Opt-OneConfig SLO-1/2/3 + Opt-PerQuery BRANE × 4 weights (default / quality / tokens / "
        "latency); primary = the deployed full-train arms, secondary = 5-fold CV mean ± std; references No-Memory / "
        "VGR = the original controlled-v1 arms (same layout).[37]"
        if rep_p.exists()
        else "<br><b>Pool-v7 re-test</b>: pending.[37]"
    )
    return prof, test
