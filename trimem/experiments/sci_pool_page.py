"""ScienceWorld pool-strat200-v7 run for the qa-trajectory.html sections (user 2026-10-22), the ScienceWorld
counterpart of ``alf_pool_page`` (ALFWorld v7 presentation): shared data layer of the ScienceWorld blocks of
render_profiling_points (train scatter + test panel), render_atlas_results (four-dataset-results / "Three datasets"),
render_cost_saving_hv (equal-cost / equal-quality + train HV curve) and render_pool_steps.

Run: results/bo_agentenv_20261016/sciworld_pool-strat200-v7__mem-sf1 (session-space-agentenv-v7, memory policy
success+failure-v1). Profiling rounds 0-6 (50 configurations, 100 stratified tasks each, controlled-c16-v1, SemFilter
v2 for every SemFilter configuration), paused after round 6.

* ``active()``: the v7 presentation replaces the older ScienceWorld content only once profiling is paused
  (profile_complete.json); before that every renderer keeps its previous ScienceWorld content.
* Train (``train_panel``): ``alf_pool_page.train_panel`` on this run: one point per profiled configuration of
  rounds >= the catalog start (0 = all), y = stratified score/100, the run's own c16 NoMemory as reference,
  the catalog selection's SLO picks / bounds.
* Test (``test_table``): sci_test_v7 (DEPLOYED arms only, no outer CV; user 2026-10-22): the 7 deployed arms
  (Opt-OneConfig SLO-1/2/3 + Opt-PerQuery BRANE default / quality / tokens / latency) re-assembled from the opt rows
  of the test experiment + the NoMemory pair files and cross-checked against test_v7c16_report.json (SystemExit on
  any disagreement); references No-Memory / VGR = the controlled-v1 originals (LLM-call span latency). Pending
  (no report): ``status`` says so, no arms.

READ-ONLY: result files and the opt DB (read-only transaction); no episodes, no writes.
"""

from __future__ import annotations

import json
import statistics as st
from functools import lru_cache
from pathlib import Path

from trimem.experiments import alf_pool_page as ap

ROOT = ap.ROOT
DS = "sciworld"
NAME = "ScienceWorld"
RUN = ROOT / "results/bo_agentenv_20261016/sciworld_pool-strat200-v7__mem-sf1"
QNAME = "score/100"


def active(run: Path | None = None) -> bool:
    """The v7 presentation is shown once profiling is paused / complete (profile_complete.json)."""
    return ((run or RUN) / "profile_complete.json").exists()


def train_panel() -> dict | None:
    if not active():
        return None
    return ap.train_panel(True, ds=DS, run=RUN, name=NAME, qname=QNAME)


def _check(name: str, what: str, mine: float | None, theirs: float | None) -> None:
    import math

    if (
        mine is None
        or theirs is None
        or not math.isclose(mine, theirs, rel_tol=ap.TOL, abs_tol=ap.TOL)
    ):
        raise SystemExit(
            f"ScienceWorld v7 {name}: {what} {mine} disagrees with test_v7c16_report.json ({theirs})"
        )


def deployed_arm(name: str, v: dict, items: dict[str, dict]) -> dict:
    """Report arm (``bo_pool_driver.assemble`` fields + span latency) -> page arm in the ALFWorld v7 deployed
    schema (``alf_pool_page.test_table_run`` 'deployed_arms'); no CV numbers (pure)."""
    return {
        "id": name,
        "label": ap.label_of(name),
        "kind": "oneconfig" if name.startswith("Opt-OneConfig") else "perquery",
        "router": "BRANE" if name.startswith("Opt-PerQuery") else None,
        "mk": ap.marker_of(name),
        "color": ap.OPT_COLORS.get(name, "#2f8f6b"),
        "weights": v.get("weights"),
        "slo": v.get("slo"),
        "folds": None,
        "n": len(items),
        "q": v["quality"],
        "q_std": None,
        "q_se": v["quality_se"],
        "tok": v["tokens"],
        "tok_std": None,
        "lat": v["latency_span_s"],
        "lat_std": None,
        "lat_wall": v["latency_s"],
        "lat_wall_std": None,
        "gain": v["paired_gain_vs_nomemory"],
        "gain_std": None,
        "gain_se": v["paired_gain_se"],
        "steps": v.get("steps"),
        "steps_std": None,
        "steps_se": v.get("steps_se"),
        "steps_success": v.get("steps_success"),
        "steps_success_std": None,
        "se_tasks": {
            "q": v["quality_se"],
            "gain": v["paired_gain_se"],
            "steps": v.get("steps_se"),
        },
        "tsem": st.mean(i["tsem"] for i in items.values()) if items else 0.0,
        "ret": st.mean(i["ret"] for i in items.values()) if items else 0.0,
        "distinct": v.get("distinct_configs"),
        "plan_shares": v.get("plan_shares"),
        "items": items,
        "cv": None,
    }


@lru_cache(maxsize=1)
def test_table() -> dict:
    """The page's ScienceWorld v7 test table: deployed arms (primary, ± SE across the 200 tasks) + references."""
    from trimem.experiments import sci_test_v7 as stv

    d = ap._driver()
    out: dict = {
        "dir": str(RUN.relative_to(ROOT)),
        "run": "sci-v7",
        "primary": "deployed",
        "arms": [],
        "deployed": [],
        "refs": [],
        "cv_arms": [],
        "k": None,
        "n_test": 200,
    }
    rp = RUN / stv.REPORT
    if not rp.exists() or not (RUN / "test" / "routing.json").exists():
        out["status"] = (
            "v7 test pending (deployed arms only; no test_v7c16_report.json yet)"
        )
        return out
    rep = json.loads(rp.read_text())
    if rep.get("protocol") != stv.PROTOCOL:
        raise SystemExit(f"{rp}: protocol {rep.get('protocol')!r} != {stv.PROTOCOL!r}")
    tab = {**stv.routing(RUN), "experiment": rep["experiment"]}
    rows = ap._test_rows(tab, None, RUN / stv.SUBDIR / "pairs", span=True, ds=DS)
    key = d.space.key
    nm_q = {
        i: d.quality_of(DS, json.load(open(f))["reward"])
        for i, f in d.drv.item_files(Path(rep["baseline"]["dir"]), DS).items()
    }
    ids = d.test_ids(DS)
    got = d.assemble(tab, rows, nm_q)
    for name, v in rep["arms"].items():
        for m, mm in (
            ("quality", "quality"),
            ("tokens", "tokens"),
            ("latency_s", "latency_span_s"),
            ("paired_gain_vs_nomemory", "paired_gain_vs_nomemory"),
            ("steps", "steps"),
        ):
            _check(f"deployed {name}", m, got[name].get(m), v.get(mm))
        items = ap.arm_items(tab["arms"][name]["routes"], rows, key)
        if len(items) != len(ids):
            raise SystemExit(f"ScienceWorld v7 {name}: incomplete deployed arm")
        out["arms"].append(deployed_arm(name, v, items))
    for name, r in ap._ref_metrics(rep, "v7").items():
        items = ap.reference_items(name, ids, span=True, ds=DS)
        _check(name, "score", st.mean(x["q"] for x in items.values()), r["quality"])
        _check(name, "tokens", st.mean(x["tok"] for x in items.values()), r["tokens"])
        _check(
            name, "latency", st.mean(x["lat"] for x in items.values()), r["latency_s"]
        )
        out["refs"].append(
            {
                "id": name,
                "label": ap.REF_LABELS.get(name, name),
                "kind": "ref",
                "mk": None,
                "color": ap.REF_COLORS.get(name, "#888"),
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
    out["deployed"] = [
        {
            "id": a["id"],
            "label": a["label"] + " (deployed)",
            **{
                k: a[k]
                for k in ("q", "q_se", "tok", "lat", "lat_wall", "gain", "gain_se")
            },
            **{
                k: a[k]
                for k in ("steps", "steps_se", "steps_success", "distinct", "weights")
            },
            "covered": a["n"],
            "n": len(ids),
        }
        for a in out["arms"]
    ]
    pq = tab["arms"].get("Opt-PerQuery-default")
    test_mem = [x for k, x in rows.items() if k[0] != key(d.space.NOMEM)]
    cat = rep.get("catalog") or {}
    out.update(
        status="complete",
        experiment=rep.get("experiment"),
        generated=rep.get("generated"),
        concurrency=rep.get("concurrency"),
        ref_concurrency=rep.get("concurrency"),
        pairs=rep.get("pairs"),
        memory_pairs=len(test_mem),
        reuse_hits=sum(x["reuse"] for x in test_mem),
        n_test=len(ids),
        baseline=rep.get("baseline"),
        catalog_rounds=[
            (cat.get("included_rounds") or [None])[0],
            (cat.get("included_rounds") or [None])[-1],
        ],
        catalog_configs=cat.get("full_configs"),
        routing_deployed=ap.routing_counts(pq["routes"], rows, key) if pq else [],
        comparison=[],
    )
    return out


def strip(t: dict) -> dict:
    return ap.strip(t)


def hv_run() -> dict | None:
    if not active():
        return None
    return ap.hv_run(ds=DS, run=RUN, name=NAME)


def hv_note(R: dict | None) -> str:
    """English note on how the ScienceWorld v7 HV curve counts episodes."""
    if not R:
        return ""
    first, last = R["rounds"][0], R["rounds"][-1]
    pruned = sum(r["incomplete"] for r in R["rounds"])
    return (
        f"<b>ScienceWorld (pool-strat200-v7, rounds {first['round']}–{last['round']})</b>: points = the "
        f"{len(R['points'])} fully evaluated configurations, all profiled under the controlled c16 runtime with "
        f"SemFilter v2 (each on its own stratified {R['n_queries']}-task sample of the training pool; round 0 on the "
        f"stage-100 prefix of its 200-task sample; unpaired means); the {pruned} configurations pruned at 50 add "
        f"episodes but no point. x = cumulative profiling episodes executed: {R['evaluated_total']:,} evaluated − "
        f"{R['reused_total']:,} quality-reuse hits = {last['episodes_cum']:,} (the driver's budget_used "
        f"{R.get('budget_used') or '—'} also counts the superseded SemFilter v1 round-0 episodes and the round-0 "
        "episodes beyond the 100-prefix). NoMemory and the noise repeat are excluded."
    )
