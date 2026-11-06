"""WebShop pool-strat200-v7 run for the qa-trajectory.html sections (user 2026-10-23), the WebShop counterpart of
``sci_pool_page`` / ``alf_pool_page`` (v7 presentation): shared data layer of the WebShop blocks of
render_profiling_points (train scatter + test panel), render_atlas_results ("Five datasets" WebShop tab),
render_cost_saving_hv (equal-cost / equal-quality + train HV curve) and render_pool_steps.

Run: results/bo_agentenv_20261016/webshop_pool-strat200-v7__mem-sf1 (session-space-agentenv-v7, memory policy
success+failure-v1, store trimem_webshop with product-level isolation). Profiling rounds 0-5 (50 configurations, 100
stratified tasks each from round 0, controlled-c16-v1, SemFilter v2 from round 0, SemFilter v2b from round 1),
paused after round 5. Episode budget max-round-v1: max_round 10 from round 2 and for every test / fixed-plan / sweep
arm; the 30-turn rounds 0-1 / NoMemory / references enter in the derived-truncation-10 view (episode_budget.py);
NoMemory baseline = the 100-task stratified prefix (nomemory-prefix-v1). ``protocol_note`` states these facts on
every WebShop display.

* ``active()``: the v7 presentation replaces the older WebShop (controlled-v1 / train200) content only once profiling
  is paused (profile_complete.json); before that every renderer keeps its previous WebShop content, byte-identical.
* Train (``train_panel``): ``alf_pool_page.train_panel`` on this run (y = stratified mean reward, the run's own c16
  NoMemory, the catalog selection's SLO picks / bounds).
* Test (``test_table``): webshop_test_v7 = the ALF v7 layout: the 7 DEPLOYED arms (primary, ± SE across the 200
  tasks) each with its outer 5-fold CV mean ± std as ``cv``, the CV arms as ``cv_arms``, re-assembled from the opt
  rows of the test experiment + the NoMemory pair files and cross-checked against test_v7c16_report.json (SystemExit
  on disagreement); references No-Memory / VGR = the controlled-v1 originals (LLM-call span latency). Pending (no
  report): ``status`` says so, no arms.

READ-ONLY: result files and the opt DB (read-only transaction); no episodes, no writes.
"""

from __future__ import annotations

import json
import statistics as st
from functools import lru_cache
from pathlib import Path

from trimem.experiments import alf_pool_page as ap

ROOT = ap.ROOT
DS = "webshop"
NAME = "WebShop"
RUN = ROOT / "results/bo_agentenv_20261016/webshop_pool-strat200-v7__mem-sf1"
QNAME = "reward"


def active(run: Path | None = None) -> bool:
    """The v7 presentation is shown once profiling is paused / complete (profile_complete.json)."""
    return ((run or RUN) / "profile_complete.json").exists()


@lru_cache(maxsize=1)
def protocol() -> dict | None:
    """The WebShop v7 protocol facts every WebShop display states (state.json; user 2026-10-23): episode budget
    (max-round-v1), SemFilter v2b, the NoMemory prefix, the train / test product rule. None before profiling paused."""
    if not active():
        return None
    s = json.loads((RUN / "state.json").read_text())
    eb = s.get("episode_budget") or {}
    v2b = (s.get("semfilter") or {}).get("v2b") or {}
    nmu = (s.get("nomemory") or {}).get("sample_used") or {}
    prov = s.get("pool_provenance") or {}
    return {
        "max_round": eb.get("max_round"),
        "native_max_round": eb.get("native_max_round", 30),
        "budget_from_round": eb.get("from_round"),
        "derived_tag": (eb.get("derived_view") or {}).get("tag"),
        "runtime_label": eb.get("runtime_label"),
        "v2b_from_round": v2b.get("from_round"),
        "v2b_rubric": v2b.get("rubric"),
        "v2b_fallback": v2b.get("sem_fallback"),
        "v2b_thresholds": v2b.get("thresholds"),
        "nomemory_prefix": nmu.get("size"),
        "nomemory_sample": len(((s.get("nomemory") or {}).get("sample") or {}).get("ids") or []),
        "pool_size": prov.get("size") or s.get("pool_size"),
        "share_products": prov.get("train_test_share_products"),
    }


def protocol_note(scope: str = "all") -> str:
    """English one-paragraph statement of the WebShop v7 protocol (plain text, HTML-safe). ``scope``: 'train'
    (profiling panels), 'test' (test / fixed-plan / sweep arms) or 'all'. Empty before profiling paused."""
    P = protocol()
    if not P:
        return ""
    mr, nat, r2 = P["max_round"], P["native_max_round"], P["budget_from_round"]
    tag = P["derived_tag"] or f"derived-truncation-{mr}"
    parts = []
    if mr:
        parts.append(
            f"WebShop protocol: episode budget max_round {mr} (AgentGym's own WebShop evaluation limit, user decision "
            f"2026-10-23) from profiling round {r2} and for every test, fixed-plan and PerQuery-sweep arm; "
            + (
                f"rounds 0–{r2 - 1}, " if scope != "test" and r2 else ""
            )
            + f"No-Memory and the references (controlled-v1 No-Memory / VGR) ran at {nat} turns and are shown in "
            f"the exact {tag} view (an episode not finished by turn {mr} counts reward 0 and {mr} steps, with the "
            f"tokens / latency of its first {mr} turns; trimem/experiments/episode_budget.py)."
        )
    if P["v2b_from_round"] is not None:
        th = ", ".join(str(x) for x in P["v2b_thresholds"] or [])
        parts.append(
            f"SemFilter v2b (rubric {P['v2b_rubric']}, {P['v2b_fallback']} fallback, thresholds {{{th}}}) from round "
            f"{P['v2b_from_round']}"
            + (
                "; round-0 SemFilter configurations = the old SemFilter-v2 regime (drawn, but excluded from the "
                "fronts and from the router / selection catalog)."
                if scope != "test"
                else " and for every test SemFilter arm (round-0 SemFilter configurations = old regime, excluded "
                "from the selection catalog)."
            )
        )
    if P["nomemory_prefix"] and scope != "test":
        parts.append(
            f"No-Memory training baseline = its {P['nomemory_prefix']}-task stratified prefix (of "
            f"{P['nomemory_sample']})."
        )
    if P["share_products"]:
        parts.append(
            f"Train / test share products (option B: training pool = all {P['pool_size']:,} stored train goals; no "
            "test goal is in the pool; retrieval excludes each query's own product)."
        )
    if mr:
        parts.append(f"Steps = environment steps per episode (at most {mr}).")
    return " ".join(parts)


def train_panel() -> dict | None:
    if not active():
        return None
    P = ap.train_panel(True, ds=DS, run=RUN, name=NAME, qname=QNAME)
    if P is not None:
        P = {
            **P,
            "protocol_note": protocol_note("train"),
            "regime": {
                **P["regime"],
                "nomemory_label": "own NoMemory, c16 (stratified prefix, nomemory-prefix-v1)",
            },
        }
    return P


def _ref(name: str, r: dict, items: dict) -> dict:
    return {
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


CV_KEYS = (
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


@lru_cache(maxsize=1)
def test_table() -> dict:
    """The page's WebShop v7 test table (ALF v7 schema): deployed arms primary (``arms``), CV arms (``cv_arms``)."""
    from trimem.experiments import pool_cv_report as pcr
    from trimem.experiments import webshop_test_v7 as wtv
    from trimem.optimizer.query_router import outer_cv as ocv

    d = ap._driver()
    out: dict = {
        "dir": str(RUN.relative_to(ROOT)),
        "run": "v7",
        "primary": "deployed",
        "arms": [],
        "deployed": [],
        "refs": [],
        "cv_arms": [],
        "k": None,
        "n_test": 200,
    }
    rp = RUN / wtv.REPORT
    if not rp.exists() or (
        wtv.OUTER_CV and not (RUN / "test" / "routing_cv.json").exists()
    ):
        out["status"] = (
            "v7 test pending ("
            + ("deployed + outer-CV arms" if wtv.OUTER_CV else "deployed arms only")
            + "; no test_v7c16_report.json yet)"
        )
        return out
    rep = json.loads(rp.read_text())
    if rep.get("protocol") != wtv.PROTOCOL:
        raise SystemExit(f"{rp}: protocol {rep.get('protocol')!r} != {wtv.PROTOCOL!r}")
    tab, cv = wtv.routing(RUN)
    tab = {**tab, "experiment": rep["experiment"]}
    rows = ap._test_rows(tab, cv, RUN / wtv.SUBDIR / "pairs", span=True, ds=DS)
    key = d.space.key
    # max-round-v1: the 30-turn controlled-v1 NoMemory baseline in the derived-truncation-10 view (as the report)
    mr = d.test_max_round(DS)
    nm_q = {
        i: d.eb.view_quality(DS, json.load(open(f)), mr)
        for i, f in d.drv.item_files(Path(rep["baseline"]["dir"]), DS).items()
    }
    ids = d.test_ids(DS)
    oc = (
        rep.get("outer_cv") or {}
    )  # deployed-only (user 2026-10-23, latest): no outer CV -> empty
    summ = oc["test"]["summary"] if oc else {}
    per_fold = oc["test"]["per_fold"] if oc else {}
    span_fold = oc["test"].get("per_fold_span") if oc else None
    folds = ocv.fold_tables(cv["arms"]) if oc and cv else {}
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
                ap._check(f"WebShop fold {f} {name}", m, got[name].get(m), theirs)
    cv_arms = {}
    for name, s in summ.items():
        fi = [
            ap.arm_items(folds[f]["arms"][name]["routes"], rows, key)
            for f in sorted(folds)
        ]
        if any(len(x) != len(ids) for x in fi):
            raise SystemExit(f"WebShop v7 {name}: incomplete fold arm")
        items = ap.fold_average(fi)
        q, q_std = ap._ms(s["quality"])
        tok, tok_std = ap._ms(s["tokens"])
        lat, lat_std = ap._ms(s["latency_span_s"])
        lat_wall, lat_wall_std = ap._ms(s["latency_s"])
        gain, gain_std = ap._ms(s["gain"])
        steps, steps_std = ap._ms(s.get("steps"))
        ss, ss_std = ap._ms(s.get("steps_success"))
        ap._check(
            name, "fold-averaged reward", st.mean(x["q"] for x in items.values()), q
        )
        se = s.get("se_tasks_mean") or {}
        dep = rep["arms"].get(name) or {}
        cv_arms[name] = {
            "id": name,
            "label": ap.label_of(name),
            "kind": "oneconfig" if name.startswith("Opt-OneConfig") else "perquery",
            "router": "BRANE" if name.startswith("Opt-PerQuery") else None,
            "mk": ap.marker_of(name),
            "color": ap.OPT_COLORS.get(name, "#2f8f6b"),
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
    got = d.assemble(tab, rows, nm_q)
    for name, v in rep["arms"].items():
        for m, mm in (
            ("quality", "quality"),
            ("tokens", "tokens"),
            ("latency_s", "latency_span_s"),
            ("paired_gain_vs_nomemory", "paired_gain_vs_nomemory"),
            ("steps", "steps"),
        ):
            ap._check(f"WebShop deployed {name}", m, got[name].get(m), v.get(mm))
        items = ap.arm_items(tab["arms"][name]["routes"], rows, key)
        if len(items) != len(ids):
            raise SystemExit(f"WebShop v7 {name}: incomplete deployed arm")
        c = cv_arms.get(name) or {}
        out["arms"].append(
            {
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
                "tsem": st.mean(i["tsem"] for i in items.values()),
                "ret": st.mean(i["ret"] for i in items.values()),
                "distinct": v.get("distinct_configs"),
                "plan_shares": v.get("plan_shares"),
                "items": items,
                "cv": {k: c.get(k) for k in CV_KEYS} if c else None,
            }
        )
    for name, r in ap._ref_metrics(rep, "v7").items():
        items = ap.reference_items(name, ids, span=True, ds=DS, max_round=mr)
        ap._check(name, "reward", st.mean(x["q"] for x in items.values()), r["quality"])
        ap._check(
            name, "tokens", st.mean(x["tok"] for x in items.values()), r["tokens"]
        )
        ap._check(
            name, "latency", st.mean(x["lat"] for x in items.values()), r["latency_s"]
        )
        out["refs"].append(_ref(name, r, items))
    out["cv_arms"] = list(cv_arms.values())
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
    names = list(cv_arms)
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
        cv_protocol=oc.get("protocol"),
        k=len(folds) or None,
        fold_sizes=(oc.get("folds") or {}).get("sizes"),
        catalog_rounds=[
            (cat.get("included_rounds") or [None])[0],
            (cat.get("included_rounds") or [None])[-1],
        ],
        catalog_configs=cat.get("full_configs"),
        routing_deployed=ap.routing_counts(pq["routes"], rows, key) if pq else [],
        comparison=[
            {"a": a, "b": b, "diff": pcr.paired_diff(per_fold, a, b)}
            for a, b in ap.comparisons(names)
        ]
        if oc
        else [],
    )
    return out


def strip(t: dict) -> dict:
    return ap.strip(t)


def hv_run() -> dict | None:
    if not active():
        return None
    return ap.hv_run(ds=DS, run=RUN, name=NAME)


def hv_note(R: dict | None) -> str:
    """English note on how the WebShop v7 HV curve counts episodes."""
    if not R:
        return ""
    first, last = R["rounds"][0], R["rounds"][-1]
    pruned = sum(r["incomplete"] for r in R["rounds"])
    return (
        f"<b>WebShop (pool-strat200-v7, rounds {first['round']}–{last['round']})</b>: points = the "
        f"{len(R['points'])} fully evaluated configurations, all profiled under the controlled c16 runtime (each on "
        f"its own stratified {R['n_queries']}-task sample of the training pool; unpaired means of the reward); the "
        f"{pruned} configurations pruned at 50 add episodes but no point. x = cumulative profiling episodes "
        f"executed: {R['evaluated_total']:,} evaluated − {R['reused_total']:,} quality-reuse hits = "
        f"{last['episodes_cum']:,}. NoMemory and the noise repeat are excluded. "
        + protocol_note("train")
    )


def splits_text() -> tuple[str, str] | None:
    """(profiling-panel cell, test cell) of the WebShop dataset-splits row (English; user 2026-10-23); None before
    the v7 profiling is paused (the row keeps its older train200 text)."""
    if not active():
        return None
    from trimem.experiments import webshop_test_v7 as wtv

    s = json.loads((RUN / "state.json").read_text())
    done = json.loads((RUN / "profile_complete.json").read_text())
    P = protocol() or {}
    size = (s.get("sample_schedule") or {}).get("size", 100)
    last = s["rounds"][-1]["round"]
    prof = (
        "<b>Pool-strat200-v7 profiling</b> (session-space-agentenv-v7, memory policy success+failure-v1): pool = "
        f"all {P.get('pool_size') or s['pool_size']:,} stored AgentTraj-L train goals (option B: train / test "
        "share products; no test goal is in the pool; retrieval excludes each query's own product); rounds "
        f"0–{last} under the controlled c16 runtime (2 halves × 8 processes): {done.get('configs')} configurations, "
        f"each on its <b>own</b> product-category-stratified sample of {size} tasks × 1 run, staged 50 → {size} "
        f"with 3-D Pareto pruning: {done.get('full_configs')} fully evaluated, {done.get('pruned_configs')} pruned. "
        + protocol_note("train")
        + " Router / selection fitted on the full training set (no outer CV).[38]"
    )
    test = (
        "<br><b>Pool-v7 test</b>: the 200 AgentEval items × 1 run under the same controlled c16 runtime at "
        f"max_round {P.get('max_round')}, quality reuse off; Opt-OneConfig SLO-1/2/3 + Opt-PerQuery BRANE × 4 "
        "weights, all deployed (fitted on the full training set); references No-Memory / VGR = the original "
        f"controlled-v1 arms (30 turns, {P.get('derived_tag')} view); fixed-plan arms and the Opt-PerQuery "
        "weight sweep follow.[38]"
        if (RUN / wtv.REPORT).exists()
        else "<br><b>Pool-v7 test</b>: pending.[38]"
    )
    return prof, test
