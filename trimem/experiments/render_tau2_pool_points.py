"""τ² pool profiling run (-v5, results/tau2_pool_v5_20261020) for the profiling-points section of qa-trajectory.html.

Data only (rendered by ``render_profiling_points``; user 2026-10-20). READ-ONLY on the live run: state.json, result
files and (when present) selection.json / test/routing.json / test_report.json are read, nothing is written, and the
driver's own helpers (``pool_profile.arm_episodes`` / ``per_task`` / ``paired_gain`` / ``summary_of`` / ``bo_point``,
``session_bo_agentenv.relaxed_pareto``) give the numbers, i.e. what the BO sees.

Train (178 tasks x 1 trial, pass^1): every profiled configuration of every round is a point. Fully evaluated ones
are on all 178 tasks; stopped / pruned / incomplete ones on the tasks of their BO observation
(``pool_profile.observed_keys``; an incomplete configuration below the first stage shows its completed tasks),
flagged ``partial``. The 3-objective relaxed front (gain, -ln tokens, -ln latency; z = RELAX_Z pooled SE, raw paired
SE as in rounds_summary.md) is computed over the fully evaluated configurations only (the selectable set). The six
aux-obs-v1 points of the earlier τ² run are passed along separately (GP-only inputs; never in the front).

Test (100 tasks; trials / quality from routing.json ``test_trials`` / ``test_quality``: 1 trial -> pass^1 since
test-1trial-v1, 4 trials -> pass^4, pass^1 too): the new Opt-OneConfig SLO-1/2/3 and Opt-PerQuery arms of
the run's test stage, assembled exactly as ``pool_profile.report_stage`` (frozen routing.json, ``test_rows``,
router overhead on latency); an arm is drawn only when every task has its 4 trials.
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

GROUPS = ("default", "sobol", "qfirst", "xover-slot", "sobol-slot", "hv", "dls")
PQ_COLORS = {
    "default": "#2f8f6b",
    "quality": "#16624a",
    "tokens": "#3b6fb6",
    "latency": "#8a4fb0",
}
OC_COLOR = "#c26a1b"


def channel_group(channel: str | None) -> str:
    """Marker group of a proposal channel: round-0 default / sobol, the round-6 slots (qfirst / xover-slot /
    sobol-slot / hv) and the DLS candidate channels of rounds 1-5 (dls-agent / dls-incumbent / dls-crossover)."""
    ch = channel or "default"
    if ch in GROUPS:
        return ch
    if ch.startswith("dls"):
        return "dls"
    return "other"


def _se(xs: list[float]) -> float | None:
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def knobs(cfg: dict, display: dict) -> str:
    sf = (
        f" · SF thr {cfg['semfilter_threshold']}"
        if "SemFilter" in display.get(cfg["plan"], cfg["plan"])
        else ""
    )
    return (
        f"{display.get(cfg['plan'], cfg['plan'])} · S/D/F/K {cfg['seed_count']}/{cfg['depth']}/"
        f"{cfg['fanout']}/{cfg['top_k']} · {cfg['outcome_policy']} · sim_gate {cfg['sim_gate']}{sf}"
    )


# ------------------------------------------------------------------ train


def train_panel() -> dict | None:
    from trimem.experiments import session_bo_agentenv as bo
    from trimem.experiments.tau2 import pool_profile as pp

    sp = pp.OUT / "state.json"
    if not sp.exists():
        return None
    state = json.loads(sp.read_text())
    panel = pp.load_panel()
    keys = pp.train_keys(panel)
    w = pp.strata_weights(keys)
    nm = pp.nomemory_tasks(keys)
    full = pp.config_tasks(state, keys)
    sel = _selection(pp.OUT)
    picks: dict[str, list[str]] = {}
    for n, s in (sel or {}).get("selections", {}).items():
        if s.get("opt_oneconfig"):
            picks.setdefault(s["opt_oneconfig"]["key"], []).append(n)
    pts, bo_pts = [], {}
    for rr in state["rounds"]:
        for j, c in enumerate(rr["configs"]):
            p, bp = _point(pp, state, rr["round"], j, c, keys, w, nm, full, panel)
            if p is None:
                continue
            p["slo"] = picks.get(p["key"], [])
            pts.append(p)
            if bp is not None:
                bo_pts[p["key"]] = bp
    relaxed = {pp.space.key(x.cfg) for x in bo.relaxed_pareto(list(bo_pts.values()))}
    strict = {pp.space.key(x.cfg) for x in bo.pareto(list(bo_pts.values()))}
    for p in pts:
        p["front3"], p["strict3"] = p["key"] in relaxed, p["key"] in strict
    nms = pp.summary_of(nm, keys, w)
    aux = state.get("aux_observations") or {}
    return {
        "kind": "pool",
        "ds": "tau2",
        "name": "τ²-bench",
        "panel": "train178 · pool -v5",
        "dir": str(pp.OUT.relative_to(pp.ROOT)),
        "experiment": state["experiment"],
        "n_queries": len(keys),
        "n_configs": len(pts),
        "n_full": sum(p["status"] == "full" for p in pts),
        "rounds": len(state["rounds"]),
        "relax_z": bo.RELAX_Z,
        "nomem": {
            "q": nms["q"],
            "q_se": nms["q_se"],
            "gain": 0.0,
            "tok": nms["tok"],
            "lat": nms["lat"],
            "steps": nms.get("steps"),
            "steps_se": nms.get("steps_se"),
            "steps_success": nms.get("steps_success"),
        },
        "points": pts,
        "aux": [
            {
                "label": a["label"],
                "plan": a["config"]["plan"],
                "experiment": a["experiment"],
                "gain": a["gain"],
                "gain_se": a["gain_se"],
                "tok": a["tokens"],
                "lat": a["latency_ms"] / 1000,
            }
            for a in aux.get("points", [])
        ],
        "aux_nomem_q": (aux.get("sources") or [{}])[0].get("nomemory_quality"),
        "aux_from_round": aux.get("from_round"),
        "concurrency": state.get("concurrency"),
        "reuse_from_round": (state.get("quality_reuse") or {}).get("from_round"),
        # semfilter-v2 rerun of the SemFilter plans (tau2.sfv2_rerun, user 2026-10-22); None = v1 judge throughout
        "semfilter": None
        if not state.get("semfilter")
        else {
            "protocol": state["semfilter"]["protocol"],
            "rubric": state["semfilter"]["rubric"],
            "top_n": state["semfilter"]["top_n"],
            "configs": state["semfilter"]["superseded"]["configs"],
            "rerun": state["semfilter"]["superseded"]["rerun"],
        },
        "slo_bounds": (sel or {}).get("bounds"),
    }


def _selection(out: Path) -> dict | None:
    p = out / "selection.json"
    return json.loads(p.read_text()) if p.exists() else None


def _point(pp, state, r, j, c, keys, w, nm, full, panel):
    cfg = c["config"]
    g = dict(c.get("staging") or {})  # copy: never mutate the state we read
    g.setdefault("stage_reached", 0)
    g.setdefault("pruned_at", None)
    k = pp.space.key(cfg)
    eps = pp.arm_episodes(pp.config_dir(cfg), keys, split="train", cfg=cfg)
    done = [x for x in keys if x in eps]
    if k in full:
        sub, status = keys, "full"
    elif g.get("stopped_by_user"):
        sub, status = pp.observed_keys(state, {"staging": g}, keys), "stopped"
    elif g["pruned_at"]:
        sub, status = pp.observed_keys(state, {"staging": g}, keys), "pruned"
    else:
        sub = (
            pp.observed_keys(state, {"staging": g}, keys)
            if g["stage_reached"]
            else done
        )
        status = "incomplete"
    if not sub:
        return None, None
    t = full.get(k) or pp.per_task(eps, sub, "train")
    if t is None:
        return None, None
    s = pp.summary_of(t, sub, w)
    gn = pp.paired_gain(t, nm, sub, w)
    recs = [eps[x][0] for x in sub]
    regimes = Counter(
        ("reused:" if e["reuse"] else "") + str(e["regime"]) for e in recs
    )
    p = {
        "id": f"r{r}c{j}",
        "key": k,
        "round": r,
        "channel": c.get("channel"),
        "source": c.get("source_channel"),
        "group": channel_group(c.get("channel")),
        "plan": cfg["plan"],
        "knobs": knobs(cfg, pp.space.DISPLAY),
        "status": status,
        "partial": status != "full",
        "n": len(sub),
        "done": len(done),
        "reuse": sum(e["reuse"] for e in recs),
        "regimes": dict(sorted(regimes.items())),
        "q": s["q"],
        "q_se": s["q_se"],
        "gain": gn.mean,
        "gain_se": gn.se,
        "tok": s["tok"],
        "tok_se": s["tok_se"],
        "lat": s["lat"],
        "lat_se": s["lat_se"],
        # agent steps (user 2026-10-21): agent calls per episode (pool_profile.episode_steps), mean over the
        # observed tasks ± SE across tasks, and over the successful trials only
        "steps": s.get("steps"),
        "steps_se": s.get("steps_se"),
        "steps_n": s.get("steps_n"),
        "steps_success": s.get("steps_success"),
    }
    bp = (
        pp.bo_point(cfg, t, nm, keys, w, state["experiment"], panel["panel_hash"])
        if status == "full"
        else None
    )
    return p, bp


# ------------------------------------------------------------------ test


def arm_summary(
    routes: dict[str, dict],
    rows: dict[tuple[str, str], dict],
    nm: dict[str, float],
    key,
) -> dict:
    """One routed test arm (pure): per task the single execution of its routed (configuration, task) pair, latency
    + router overhead (= ``bo_pool_driver.assemble``); pass^4 / pass^1 / tokens / latency mean ± SE over the
    covered tasks, paired pass^4 gain vs NoMemory."""
    got = [(q, r, rows.get((key(r["config"]), q))) for q, r in routes.items()]
    have = [(q, r, x) for q, r, x in got if x is not None]
    out = {
        "n": len(got),
        "covered": len(have),
        "distinct": len({key(r["config"]) for _, r, _ in got}),
    }
    if not have:
        return out
    q = [x["quality"] for _, _, x in have]
    p1 = [_pass(x, 1) for _, _, x in have]
    tok = [x["tokens"] for _, _, x in have]
    lat = [(x["e2e_ms"] + r["overhead_ms"]) / 1000 for _, r, x in have]
    d = [x["quality"] - nm[qid] for qid, _, x in have if qid in nm]
    out.update(
        q=st.mean(q),
        q_se=_se(q),
        pass1=st.mean(p1),
        pass1_se=_se(p1),
        tok=st.mean(tok),
        tok_se=_se(tok),
        lat=st.mean(lat),
        lat_se=_se(lat),
        gain=st.mean(d) if d else None,
        gain_se=_se(d) if d else None,
    )
    return out


def _pass(task: dict, j: int) -> float:
    ph = task["pass_hat"]
    return float(ph[j] if j in ph else ph[str(j)])


def test_arms() -> dict:
    """New test arms of the run (empty list + a pending note until the test stage produced them)."""
    from trimem.experiments.tau2 import pool_profile as pp

    out: dict = {
        "dir": str(pp.OUT.relative_to(pp.ROOT)),
        "arms": [],
        "pending": [],
        **protocol_of(None, pp.TEST_TRIALS),
    }
    sel = _selection(pp.OUT)
    if sel:
        out["bounds"] = sel.get("bounds")
        out["picks"] = {
            n: (s.get("opt_oneconfig") or {}).get("label")
            for n, s in sel["selections"].items()
        }
    rp = pp.OUT / "test" / "routing.json"
    if not rp.exists():
        out["status"] = (
            "selection frozen; test routing not frozen yet"
            if sel
            else "router / select / test stages not finished"
        )
        return out
    tab = json.loads(rp.read_text())
    out.update(protocol_of(tab, pp.TEST_TRIALS))
    if out["trials"] != pp.TEST_TRIALS:
        out["status"] = (
            f"routing.json was frozen for {out['trials']} test trial(s), the driver protocol is "
            f"{pp.TEST_TRIALS}: not drawn"
        )
        out["trials"] = pp.TEST_TRIALS
        out["quality"] = f"pass^{pp.TEST_K}"
        return out
    rows = pp.test_rows(tab)
    keys = pp.test_keys()
    nm_tasks = pp.per_task(pp.ref_episodes(pp.REUSE / "NoMemory", keys), keys, "test")
    nm = {q: t["quality"] for q, t in (nm_tasks or {}).items()}
    rep_p = pp.OUT / "test_report.json"
    rep = json.loads(rep_p.read_text())["arms"] if rep_p.exists() else {}
    for name, arm in tab["arms"].items():
        a = arm_summary(arm["routes"], rows, nm, pp.space.key)
        if (
            name in rep
            and a.get("q") is not None
            and rep[name].get("quality") is not None
        ):
            if abs(rep[name]["quality"] - a["q"]) > 1e-9:
                raise SystemExit(
                    f"{name}: test_report.json disagrees with the test rows"
                )
        if arm["kind"] == "oneconfig":
            ent = {
                "id": f"OC:{arm['slo']}",
                "label": f"Opt-OneConfig {arm['slo']}",
                "mk": "oc",
                "slo": arm["slo"],
                "color": OC_COLOR,
            }
        else:
            wn = name.removeprefix("Opt-PerQuery-")
            ent = {
                "id": f"PQ:{wn}",
                "label": f"Opt-PerQuery {wn}",
                "mk": "pq",
                "weights": arm.get("weights"),
                "color": PQ_COLORS.get(wn, "#2f8f6b"),
            }
        ent.update(a, cfg=_route_label(arm, pp))
        (out["arms"] if a["covered"] == a["n"] else out["pending"]).append(ent)
    out["status"] = (
        "complete"
        if not out["pending"]
        else f"test stage running: {len(out['pending'])} of {len(tab['arms'])} arms incomplete"
    )
    return out


def protocol_of(tab: dict | None, default_trials: int) -> dict:
    """Test protocol of the frozen routing table (``test_trials`` / ``test_quality``; tables frozen before the
    1-trial protocol carry neither and were 4 trials, pass^4); without a table, the driver's current protocol."""
    if tab is None:
        n = default_trials
        return {"trials": n, "quality": f"pass^{n}"}
    n = int(tab.get("test_trials", 4))
    return {"trials": n, "quality": tab.get("test_quality") or f"pass^{n}"}


def _route_label(arm: dict, pp) -> str | None:
    if arm["kind"] != "oneconfig":
        return None
    cfg = next(iter(arm["routes"].values()))["config"]
    return knobs(cfg, pp.space.DISPLAY)
