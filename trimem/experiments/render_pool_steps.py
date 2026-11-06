"""qa-trajectory.html section ``pool-agent-steps`` (user 2026-10-21): agent steps of the pool-profiling runs.

A. Test: mean agent steps per test arm (outer 5-fold CV arms: mean ± std across folds; references single run) for
   every dataset with data: ALFWorld v7 (``alf_pool_page.test_table``: the concurrency-16 rerun when present, else
   the c128 run; LimeQO c128-only arms are left out), τ² pool -v5 (steps = agent LLM calls, recomputed exactly as
   ``tau2.pool_profile.report_stage`` from the frozen routing tables and the pair result files, cross-checked on
   success against test_report.json) and ScienceWorld when its pool test report exists. All episodes vs
   successful episodes only (toggle).
B. Train: success vs mean agent steps of every profiled configuration (ALFWorld v7: ``alf_pool_page.train_panel``;
   τ² pool -v5: ``render_tau2_pool_points.train_panel``), Pareto front (higher quality, fewer steps) over the fully
   evaluated configurations, partial configurations hollow, No-Memory diamond; plan legend toggles.

Read-only. Rewrites only its own block under the shared page lock.
    PYTHONPATH=. .venv/bin/python -m trimem.experiments.render_pool_steps
"""

from __future__ import annotations

import fcntl
import json
import math
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
START, END = "<!-- pool-agent-steps:start -->", "<!-- pool-agent-steps:end -->"
BEFORE = "<!-- agent-step-profiles:start -->"
TOL = 1e-6
REF_COLORS = {"NoMemory": "#8a8f87", "VGR": "#2a6fb3"}
REF_LABELS = {"NoMemory": "No-Memory", "VGR": "VGR"}


def _ms(x: dict | None) -> tuple[float | None, float | None]:
    return (None, None) if not x else (x.get("mean"), x.get("std"))


def kind_of(name: str) -> str:
    return "oc" if name.startswith("Opt-OneConfig") else "pq"


def arm_entry(name: str, s: dict, label: str, color: str) -> dict:
    """Outer-CV summary entry (``outer_cv.summarize``) -> plotted arm. Pure."""
    st, st_std = _ms(s.get("steps"))
    ss, ss_std = _ms(s.get("steps_success"))
    q, q_std = _ms(s.get("quality"))
    return {
        "id": name,
        "label": label,
        "kind": kind_of(name),
        "color": color,
        "folds": s.get("folds"),
        "q": q,
        "q_std": q_std,
        "steps": st,
        "steps_std": st_std,
        "steps_success": ss,
        "steps_success_std": ss_std,
        "steps_se_tasks": (s.get("se_tasks_mean") or {}).get("steps"),
    }


def ref_entry(
    name: str, q: float | None, steps: float | None, ss: float | None, se=None
) -> dict:
    """Single-run reference arm -> plotted arm (no fold std). Pure."""
    return {
        "id": name,
        "label": REF_LABELS.get(name, name),
        "kind": "ref",
        "color": REF_COLORS.get(name, "#888"),
        "folds": None,
        "q": q,
        "q_std": None,
        "steps": steps,
        "steps_std": None,
        "steps_success": ss,
        "steps_success_std": None,
        "steps_se_tasks": se,
    }


def _check(what: str, a: float | None, b: float | None) -> None:
    if a is None or b is None or not math.isclose(a, b, rel_tol=TOL, abs_tol=TOL):
        raise SystemExit(f"pool steps: {what} {a} != report {b}")


# ------------------------------------------------------------------ test


def v7_arms(t: dict) -> list[dict]:
    """A v7 page test table (alf_pool_page / sci_pool_page) -> plotted step arms: references + Opt arms (pure)."""
    return [
        ref_entry(
            r["id"],
            r["q"],
            r.get("steps"),
            r.get("steps_success"),
            (r.get("se_tasks") or {}).get("steps"),
        )
        for r in t["refs"]
    ] + [
        {
            "id": a["id"],
            "label": a["label"],
            "kind": "oc"
            if a["kind"] == "oneconfig"
            else "lq"
            if a.get("router") == "LimeQO"
            else "pq",
            "color": a["color"],
            "folds": a.get("folds"),
            "q": a["q"],
            "q_std": a.get("q_std"),
            "steps": a.get("steps"),
            "steps_std": a.get("steps_std"),
            "steps_success": a.get("steps_success"),
            "steps_success_std": a.get("steps_success_std"),
            "steps_se_tasks": (a.get("se_tasks") or {}).get("steps"),
        }
        for a in t["arms"]
    ]


def alf_test() -> dict:
    from trimem.experiments import alf_pool_page as ap

    t = ap.test_table()
    if t.get("status") != "complete":
        return {
            "ds": "alfworld",
            "name": "ALFWorld",
            "status": t.get("status"),
            "arms": [],
        }
    arms = v7_arms(t)
    c16 = t.get("run") == "c16"
    src = (
        f"{t['dir']}/test_v7c16 (v7 re-test, catalog rounds 6–21, controlled c16 runtime; {t['n_test']} test tasks "
        "× 1 run; deployed arms, whisker = ± SE across tasks; references = controlled-v1 originals)"
        if t.get("run") == "v7"
        else f"{t['dir']}/{'test_c16' if c16 else 'test'} ({'concurrency-16 rerun, references rerun' if c16 else 'c128'}"
        f"; {t['n_test']} test tasks × 1 run, outer {t['k']}-fold CV)"
    )
    return {
        "ds": "alfworld",
        "name": "ALFWorld",
        "status": "complete",
        "unit": "environment steps (= agent LLM turns) per episode",
        "source": src,
        "arms": arms,
    }


def tau2_test() -> dict:
    from trimem.experiments import bo_pool_driver as bpd
    from trimem.experiments import router_outer_cv as rcv
    from trimem.experiments.alf_pool_page import OPT_COLORS
    from trimem.experiments.tau2 import pool_profile as pp

    rp = pp.OUT / "test_report.json"
    if not rp.exists():
        return {
            "ds": "tau2",
            "name": "τ²-bench",
            "status": "pool -v5 test pending",
            "arms": [],
        }
    rep = json.loads(rp.read_text())
    tab = json.loads((pp.OUT / "test" / "routing.json").read_text())
    if (
        pp.load_cv_routing() is None
    ):  # deployed-only test (semfilter-v2 rerun, user 2026-10-22)
        return tau2_deployed(rep, tab)
    cv = pp.load_cv_routing()
    rows = pp.test_rows({**tab, "arms": {**tab["arms"], **cv["arms"]}})
    keys = pp.test_keys()
    nm = {
        q: t["quality"]
        for q, t in pp.per_task(
            pp.ref_episodes(pp.REUSE / "NoMemory", keys), keys, "test"
        ).items()
    }

    def fold_metrics(t: dict) -> dict:
        out = bpd.assemble(t, rows, nm, key=pp.space.key)
        return {k: v for k, v in out.items() if v.get("covered") == v.get("n")}

    summ = rcv.test_summary(cv, fold_metrics)["summary"]
    theirs = rep["outer_cv"]["test"]["summary"]
    for k, v in summ.items():
        _check(f"τ² {k} pass^1", v["quality"]["mean"], theirs[k]["quality"]["mean"])
    refs = pp.reference_arms()
    arms = [
        ref_entry(n, r["q"], r.get("steps"), r.get("steps_success"), r.get("steps_se"))
        for n, r in refs.items()
        if r and n in REF_LABELS
    ]
    for n in ("NoMemory", "VGR"):
        if refs.get(n):
            _check(f"τ² {n} pass^1", refs[n]["q"], rep["references"][n]["q"])
    from trimem.experiments.alf_pool_page import label_of

    arms += [
        arm_entry(k, v, label_of(k), OPT_COLORS.get(k, "#2f8f6b"))
        for k, v in summ.items()
    ]
    return {
        "ds": "tau2",
        "name": "τ²-bench",
        "status": "complete",
        "unit": "agent LLM calls per episode",
        "source": f"{pp.OUT.relative_to(ROOT)}/test (pool -v5, 100 test tasks × trial 0, outer 5-fold CV; references "
        "= trial 0 of the 4-trial arms)",
        "arms": arms,
    }


def deployed_entry(name: str, v: dict, label: str, color: str) -> dict:
    """Deployed arm (``bo_pool_driver.assemble`` fields) -> plotted arm: whisker = ± SE across tasks. Pure."""
    return {
        "id": name,
        "label": label,
        "kind": kind_of(name),
        "color": color,
        "folds": None,
        "q": v.get("quality"),
        "q_std": None,
        "steps": v.get("steps"),
        "steps_std": None,
        "steps_success": v.get("steps_success"),
        "steps_success_std": None,
        "steps_se_tasks": v.get("steps_se"),
    }


def tau2_deployed(rep: dict, tab: dict) -> dict:
    """τ² deployed-only test (no outer CV; tau2.sfv2_rerun): steps of the deployed arms, recomputed from the frozen
    routing and the pair files, cross-checked on pass^1 against test_report.json."""
    from trimem.experiments import bo_pool_driver as bpd
    from trimem.experiments.alf_pool_page import OPT_COLORS, label_of
    from trimem.experiments.tau2 import pool_profile as pp

    rows = pp.test_rows(tab)
    keys = pp.test_keys()
    nm = {
        q: t["quality"]
        for q, t in pp.per_task(
            pp.ref_episodes(pp.REUSE / "NoMemory", keys), keys, "test"
        ).items()
    }
    got = bpd.assemble(tab, rows, nm, key=pp.space.key)
    for k, v in got.items():
        _check(
            f"τ² {k} pass^1",
            v.get("quality"),
            (rep["arms"].get(k) or {}).get("quality"),
        )
    refs = pp.reference_arms()
    arms = [
        ref_entry(n, r["q"], r.get("steps"), r.get("steps_success"), r.get("steps_se"))
        for n, r in refs.items()
        if r and n in REF_LABELS
    ] + [
        deployed_entry(k, v, label_of(k), OPT_COLORS.get(k, "#2f8f6b"))
        for k, v in got.items()
    ]
    return {
        "ds": "tau2",
        "name": "τ²-bench",
        "status": "complete",
        "unit": "agent LLM calls per episode",
        "source": f"{pp.OUT.relative_to(ROOT)}/test (pool -v5, semfilter-v2 for SemFilter plans, 100 test tasks × "
        "trial 0; deployed arms only, whisker = ± SE across tasks; references = trial 0 of the 4-trial arms)",
        "arms": arms,
    }


def sci_test() -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import sci_pool_page as sp
    from trimem.experiments.alf_pool_page import OPT_COLORS, label_of

    if (
        sp.active()
    ):  # ScienceWorld v7 (user 2026-10-22): deployed arms only, no outer CV
        t = sp.test_table()
        if t.get("status") != "complete":
            return {
                "ds": "sciworld",
                "name": "ScienceWorld",
                "status": t.get("status"),
                "arms": [],
            }
        return {
            "ds": "sciworld",
            "name": "ScienceWorld",
            "status": "complete",
            "unit": "environment steps (= agent LLM turns) per episode",
            "source": f"{t['dir']}/test_v7c16 (v7 test, catalog rounds {t['catalog_rounds'][0]}–"
            f"{t['catalog_rounds'][1]}, controlled c16 runtime; {t['n_test']} test tasks × 1 run; deployed arms only, "
            "whisker = ± SE across tasks; references = controlled-v1 originals)",
            "arms": v7_arms(t),
        }
    p = d.run_dir("sciworld") / "test_report.json"
    if not p.exists():
        return {
            "ds": "sciworld",
            "name": "ScienceWorld",
            "status": "pool test pending",
            "arms": [],
        }
    rep = json.loads(p.read_text())
    summ = (rep.get("outer_cv") or {}).get("test", {}).get("summary") or {}
    arms = [
        ref_entry(
            n,
            r.get("quality"),
            r.get("steps"),
            r.get("steps_success"),
            r.get("steps_se"),
        )
        for n, r in (rep.get("reference_arms") or {}).items()
    ] + [
        arm_entry(k, v, label_of(k), OPT_COLORS.get(k, "#2f8f6b"))
        for k, v in summ.items()
        if "LimeQO" not in k
    ]
    return {
        "ds": "sciworld",
        "name": "ScienceWorld",
        "status": "complete" if summ else "pool test pending",
        "unit": "environment steps (= agent LLM turns) per episode",
        "source": str(p.parent.relative_to(ROOT)) + "/test",
        "arms": arms,
    }


def ws_test() -> dict:
    """WebShop v7 test (user 2026-10-23; webshop_pool_page): deployed arms (whisker ± SE across tasks) + references."""
    from trimem.experiments import webshop_pool_page as wp

    t = wp.test_table()
    if t.get("status") != "complete":
        return {
            "ds": "webshop",
            "name": "WebShop",
            "status": t.get("status"),
            "source": wp.protocol_note("test"),
            "arms": [],
        }
    return {
        "ds": "webshop",
        "name": "WebShop",
        "status": "complete",
        "unit": "environment steps (= agent LLM turns) per episode",
        "source": f"{t['dir']}/test_v7c16 (v7 test, catalog rounds {t['catalog_rounds'][0]}–"
        f"{t['catalog_rounds'][1]}, controlled c16 runtime; {t['n_test']} test tasks × 1 run; deployed arms only, whisker "
        "= ± SE across tasks; references = controlled-v1 originals). "
        + wp.protocol_note("test"),
        "arms": v7_arms(t),
    }


def with_fixed(T: dict) -> dict:
    """Fixed-plan reference arms (user 2026-10-23; ``fixed_plans_page``): each plan default's steps bar (whisker =
    ± SE across tasks), kind ``plan`` (one legend toggle per plan); ``fixed_line`` = the status line (pending until
    the fixed-plan report exists). Added only to a dataset whose test arms are complete."""
    from trimem.experiments import fixed_plans_page as fp

    t = fp.table(T["ds"])
    extra = [
        {
            "id": a["id"],
            "label": a["short"] + " (plan default)",
            "kind": "plan",
            "plan": a["plan"],
            "color": a["color"],
            "folds": None,
            "q": a["q"],
            "q_std": None,
            "steps": a.get("steps"),
            "steps_std": None,
            "steps_success": a.get("steps_success"),
            "steps_success_std": None,
            "steps_se_tasks": a.get("steps_se"),
        }
        for a in t["arms"]
    ]
    return {
        **T,
        "arms": T["arms"] + (extra if T.get("status") == "complete" else []),
        "fixed_line": fp.status_line(t),
    }


# ------------------------------------------------------------------ train


def train_points(panel: dict | None, steps_unit: str) -> dict | None:
    """A profiling panel -> the scatter's points (only configurations with a step value). Pure."""
    if not panel:
        return None
    keep = (
        "id", "round", "plan", "knobs", "status", "partial", "pruned_at", "n", "q", "q_se", "tok", "lat",
        "steps", "steps_se", "steps_n", "steps_success",
    )  # fmt: skip
    pts = [
        {k: p.get(k) for k in keep}
        for p in panel["points"]
        if p.get("steps") is not None
    ]
    nm = panel["nomem"]
    return {
        "ds": panel["ds"],
        "name": panel["name"],
        "panel": panel.get("panel"),
        "n_queries": panel["n_queries"],
        "unit": steps_unit,
        "qname": "pass^1"
        if panel["ds"] == "tau2"
        else "reward"
        if panel["ds"] == "webshop"
        else "score/100"
        if panel["ds"] == "sciworld"
        else "success rate",
        "points": pts,
        "skipped": len(panel["points"]) - len(pts),
        "nomem": {
            k: nm.get(k)
            for k in ("q", "q_se", "tok", "lat", "steps", "steps_se", "steps_success")
        },
    }


def collect() -> dict:
    from trimem.experiments import alf_pool_page as ap
    from trimem.experiments import render_tau2_pool_points as t2
    from trimem.experiments import sci_pool_page as sp

    from trimem.experiments import webshop_pool_page as wp

    ws = wp.active()  # WebShop v7 (user 2026-10-23) only once its profiling paused
    return {
        "test": [
            with_fixed(alf_test()),
            with_fixed(sci_test()),
            *([with_fixed(ws_test())] if ws else []),
            with_fixed(tau2_test()),
        ],
        "train": [
            x
            for x in (
                train_points(ap.train_panel(), "environment steps / episode"),
                train_points(sp.train_panel(), "environment steps / episode"),
                train_points(wp.train_panel(), "environment steps / episode")
                if ws
                else None,
                train_points(t2.train_panel(), "agent LLM calls / episode"),
            )
            if x
        ],
    }


SECTION = r"""<!-- pool-agent-steps:start -->
<section class="panel section" id="pool-steps"><style>
#pool-steps .ps-tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}#pool-steps .ps-tabs button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
#pool-steps .ps-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:8px 0 14px}@media (max-width:900px){#pool-steps .ps-grid{grid-template-columns:minmax(0,1fr)}}
#pool-steps .ps-card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--panel);min-width:0}
#pool-steps .ps-card h4{margin:0 0 4px;font-size:14px}#pool-steps svg{display:block;width:100%;height:auto}
#pool-steps svg text{fill:var(--muted);font-size:11px;font-family:inherit}#pool-steps svg .ps-v{fill:var(--fg);font-size:10.5px}
#pool-steps .ps-leg{display:flex;gap:6px 14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:4px 0 6px}#pool-steps .ps-leg label{cursor:pointer;white-space:nowrap}
#pool-steps .ps-ctl{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin:6px 0}
#pool-steps table{border-collapse:collapse;font-size:12.5px;width:100%}#pool-steps th,#pool-steps td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
#pool-steps th:first-child,#pool-steps td:first-child{text-align:left}#pool-steps .ps-wrap{overflow-x:auto}
</style>
<h2>Agent steps · pool-profiling runs (test arms and training configurations)</h2>
<p class="small">Steps = agent interaction steps per episode: ALFWorld / ScienceWorld / WebShop = environment steps (one per agent LLM turn; WebShop capped at max_round 10, its 30-turn references in the derived-truncation-10 view); τ²-bench = agent LLM calls. Units differ across benchmarks, so compare within a dataset. Test arms: ALFWorld / ScienceWorld / WebShop v7 = the deployed arms (selection / router fitted on all training queries), whisker = ± SE across the test tasks; τ² = outer 5-fold CV arms (mean across the 5 fold routers / selections, whisker = ± std across folds); references (No-Memory, VGR) = single runs, drawn as bars and as dashed lines. "Successful episodes only" averages over each arm's successful episodes. Fewer steps can mean earlier failures: read steps together with success.</p>
<div class="ps-ctl"><span>Episodes: <label><input type="radio" name="ps-o" value="all" checked> all</label> <label><input type="radio" name="ps-o" value="succ"> successful only</label></span></div>
<h3>A. Test · mean agent steps per arm</h3>
<div class="ps-tabs" id="ps-tabs"></div><p class="small" id="ps-src"></p>
<div class="ps-leg" id="ps-leg"></div>
<div class="ps-grid"><div class="ps-card" style="grid-column:1/-1"><h4 id="ps-bt">Mean steps per test arm</h4><div id="ps-bars"></div></div></div>
<details><summary>Test steps (table)</summary><div class="ps-wrap" id="ps-table"></div></details>
<h3>B. Train · success vs agent steps of every profiled configuration</h3>
<p class="small">One point per profiled configuration (its own stratified training sample; mean steps over the observed tasks). Filled = fully evaluated, hollow dashed = pruned / partial (n on hover); dashed polyline = Pareto front over the fully evaluated configurations (higher quality, fewer steps); ◆ = No-Memory. Colour = plan; click the legend to show / hide plans.</p>
<div class="ps-leg" id="ps-pleg"></div>
<div class="ps-grid" id="ps-train"></div>
<script id="ps-data" type="application/json">__DATA__</script>
<script>(()=>{const D=JSON.parse(document.getElementById('ps-data').textContent),$=id=>document.getElementById(id),
e=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
f2=v=>v==null?'—':v.toFixed(2),q3=v=>v==null?'—':v.toFixed(3),k=v=>v==null?'—':v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0),sec=v=>v==null?'—':v.toFixed(v<10?1:0);
const PC={VGR:'#2a6fb3','VGR+Rank':'#1f9aa6',P2:'#8f5bb5','P2+Rank':'#c0558a','P1+Rank':'#b8862b',P1:'#6b8e23','VGR+SemRank':'#3a8f8f'},
PN={VGR:'VGR','VGR+Rank':'VGR+Rank',P2:'VGR+SemFilter','P2+Rank':'VGR+SemFilter+Rank','P1+Rank':'V+SemFilter+G+Rank',P1:'V+SemFilter+G'},
KN={ref:'references (single run)',oc:'Opt-OneConfig SLO-1/2/3',pq:'Opt-PerQuery (BRANE)',lq:'Opt-PerQuery (LimeQO)'};
__FPJS__const ak=(T,a)=>a.kind==='plan'?'k:'+T.ds+':plan:'+a.id:'k:'+T.ds+':'+a.kind;
const HID=new Set(),on=x=>!HID.has(x);let cur=(D.test.find(t=>t.status==='complete')||D.test[0]).ds,succ=false;
const val=a=>succ?a.steps_success:a.steps,err=a=>succ?a.steps_success_std:(a.steps_std??(a.kind!=='ref'&&a.folds==null?a.steps_se_tasks:null));
function tw(t){let w=0;for(const ch of t)w+=ch.charCodeAt(0)>255?10.5:6.2;return w}
function bars(T){const arms=T.arms.filter(a=>val(a)!=null&&on(ak(T,a)));if(!arms.length)return `<p class="small">${e(T.status==='complete'?'No arm visible (legend).':T.status)}</p>`;
 const W=880,rh=22,mt=8,mb=38,ml=Math.min(330,Math.max(...arms.map(a=>tw(a.label)))+12),mr=56,H=mt+mb+arms.length*rh,
  mx=Math.max(...arms.map(a=>val(a)+(err(a)||0)))*1.08,X=v=>ml+v/mx*(W-ml-mr),tk=mx>40?10:mx>16?5:2;
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${e(T.name)} mean agent steps per test arm">`;
 for(let v=0;v<=mx;v+=tk)s+=`<line x1="${X(v)}" x2="${X(v)}" y1="${mt}" y2="${H-mb}" stroke="var(--line)"/><text x="${X(v)}" y="${H-mb+14}" text-anchor="middle">${v}</text>`;
 s+=`<text x="${(W+ml)/2}" y="${H-4}" text-anchor="middle">${e(T.unit)}${succ?' · successful episodes only':''}</text>`;
 T.arms.filter(a=>a.kind==='ref'&&val(a)!=null&&on('k:'+T.ds+':ref')).forEach(a=>{s+=`<line x1="${X(val(a))}" x2="${X(val(a))}" y1="${mt}" y2="${H-mb}" stroke="${a.color}" stroke-dasharray="6 4" stroke-width="1.3"><title>${e(a.label+' '+f2(val(a)))}</title></line>`});
 arms.forEach((a,i)=>{const y=mt+i*rh+3,bh=rh-8,v=val(a),sd=err(a),
  t=`${a.label}${a.kind==='ref'?' (reference, single run)':a.kind==='plan'?' (fixed plan default, ± SE across tasks)':a.folds==null?' (deployed, ± SE across tasks)':` (${a.folds}-fold CV)`}\nsteps / episode ${f2(a.steps)}${a.steps_std!=null?' ± '+f2(a.steps_std)+' std across folds':''}${a.steps_se_tasks!=null?' · SE_tasks '+f2(a.steps_se_tasks):''}\non successful episodes ${f2(a.steps_success)}${a.steps_success_std!=null?' ± '+f2(a.steps_success_std)+' std':''}\nsuccess ${q3(a.q)}${a.q_std!=null?' ± '+q3(a.q_std)+' std':''}`;
  s+=`<text x="${ml-6}" y="${y+bh-2}" text-anchor="end">${e(a.label)}</text><rect x="${ml}" y="${y}" width="${Math.max(X(v)-ml,1)}" height="${bh}" rx="2" fill="${a.color}" fill-opacity="${a.kind==='ref'?.45:a.kind==='plan'?.6:.85}" stroke="${a.color}"${a.kind==='plan'?' stroke-dasharray="4 2"':''}><title>${e(t)}</title></rect>`;
  if(a.kind==='plan')s+=fpShape(a.plan,ml+9,y+bh/2,5,`fill="${a.color}" stroke="var(--panel)" stroke-width="1"`,`<title>${e(t)}</title>`);
  if(sd!=null){const lo=X(Math.max(v-sd,0)),hi=X(v+sd),ym=y+bh/2;s+=`<path d="M${lo} ${ym}H${hi}M${lo} ${ym-4}V${ym+4}M${hi} ${ym-4}V${ym+4}" stroke="var(--fg)" fill="none"/>`}
  s+=`<text class="ps-v" x="${X(v+(sd||0))+5}" y="${y+bh-2}">${f2(v)}</text>`});
 return '<div class="ps-wrap">'+s.replace('<svg ','<svg style="min-width:600px" ')+'</svg></div>'}
function table(T){return `<table><thead><tr><th>arm</th><th>steps (all) ± std</th><th>SE_tasks steps</th><th>steps (successes) ± std</th><th>success</th></tr></thead><tbody>${T.arms.map(a=>`<tr><td>${e(a.label)}</td><td>${f2(a.steps)}${a.steps_std!=null?' ± '+f2(a.steps_std):''}</td><td>${f2(a.steps_se_tasks)}</td><td>${f2(a.steps_success)}${a.steps_success_std!=null?' ± '+f2(a.steps_success_std):''}</td><td>${q3(a.q)}${a.q_std!=null?' ± '+q3(a.q_std):''}</td></tr>`).join('')}</tbody></table>`}
function showTest(){const T=D.test.find(t=>t.ds===cur);$('ps-tabs').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.ds===cur)));
 $('ps-src').textContent=(T.status==='complete'?`${T.name}: ${T.source}. Unit: ${T.unit}.`:`${T.name}: ${T.status}.`)+(T.fixed_line?' '+T.fixed_line:'');
 $('ps-bt').textContent=`${T.name} · mean ${succ?'steps on successful episodes':'steps'} per test arm`;
 $('ps-leg').innerHTML=T.status==='complete'?['ref','oc','pq','lq'].filter(x=>T.arms.some(a=>a.kind===x)).map(x=>`<label><input type="checkbox" data-k="k:${T.ds}:${x}"${on('k:'+T.ds+':'+x)?' checked':''}> ${KN[x]}</label>`).join('')+T.arms.filter(a=>a.kind==='plan').map(a=>`<label><input type="checkbox" data-k="${e(ak(T,a))}"${on(ak(T,a))?' checked':''}> ${fpGlyph(a.plan,a.color)} ${e(a.label)}</label>`).join(''):'';
 $('ps-bars').innerHTML=bars(T);$('ps-table').innerHTML=T.status==='complete'?table(T):''}
function front(pts){return pts.filter(p=>!pts.some(q=>q!==p&&q.q>=p.q&&q.x<=p.x&&(q.q>p.q||q.x<p.x))).sort((a,b)=>a.x-b.x)}
function scatter(P){const xv=p=>succ?p.steps_success:p.steps,vis=P.points.filter(p=>on('p:'+p.plan)&&xv(p)!=null).map(p=>({...p,x:xv(p)})),nm=P.nomem,nx=xv(nm);
 const W=520,H=320,m={l:48,r:14,t:12,b:40},all=vis.map(p=>p.x).concat(nx!=null?[nx]:[]),ys=vis.map(p=>p.q).concat([nm.q]);
 if(!vis.length)return '<p class="small">No configuration visible (legend).</p>';
 let x0=Math.min(...all),x1=Math.max(...all),y0=Math.min(...ys),y1=Math.max(...ys);const px=(x1-x0)*.08||1,py=(y1-y0)*.1||.02;x0-=px;x1+=px;y0-=py;y1+=py;
 const X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${e(P.name)} success vs steps">`;
 [y0+py,(y0+y1)/2,y1-py].forEach(v=>{s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${m.l-5}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`});
 [x0+px,(x0+x1)/2,x1-px].forEach(v=>{s+=`<text x="${X(v)}" y="${H-m.b+14}" text-anchor="middle">${v.toFixed(1)}</text>`});
 s+=`<text x="${(W+m.l)/2}" y="${H-4}" text-anchor="middle">${e(P.unit)}${succ?' (successful episodes)':''} · fewer = left</text>`;
 const f=front(vis.filter(p=>!p.partial));if(f.length>1)s+=`<polyline fill="none" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.4" points="${f.map(p=>X(p.x)+','+Y(p.q)).join(' ')}"/>`;
 vis.slice().sort((a,b)=>(f.includes(a)?1:0)-(f.includes(b)?1:0)).forEach(p=>{const c=PC[p.plan]||'#888',fr=f.includes(p),
  t=`${p.id} · round ${p.round} · ${PN[p.plan]||p.plan} · ${p.status}${p.pruned_at?' at stage '+p.pruned_at:''}\n${p.knobs}\nn = ${p.n} / ${P.n_queries}\n${P.qname} ${q3(p.q)} ± ${q3(p.q_se)}\nsteps ${f2(p.steps)} ± ${f2(p.steps_se)} SE · on successes ${f2(p.steps_success)}\ntokens ${k(p.tok)} · latency ${sec(p.lat)} s / episode${fr?'\nPareto front (success × steps)':''}`;
  s+=`<circle cx="${X(p.x)}" cy="${Y(p.q)}" r="${fr?6:4.5}" fill="${p.partial?'var(--panel)':c}" fill-opacity="${fr?.95:.6}" stroke="${c}" stroke-width="${fr?2:1.5}"${p.partial?' stroke-dasharray="2 2"':''}><title>${e(t)}</title></circle>`});
 if(nx!=null){const cx=X(nx),cy=Y(nm.q);s+=`<rect x="${cx-5}" y="${cy-5}" width="10" height="10" fill="#8a8f87" transform="rotate(45 ${cx} ${cy})"><title>${e(`No-Memory (training sample)\n${P.qname} ${q3(nm.q)} ± ${q3(nm.q_se)}\nsteps ${f2(nm.steps)} · on successes ${f2(nm.steps_success)}\ntokens ${k(nm.tok)} · latency ${sec(nm.lat)} s`)}</title></rect><text class="ps-v" x="${cx>W-90?cx-8:cx+8}" y="${cy-8}" text-anchor="${cx>W-90?'end':'start'}">No-Memory</text>`}
 return s+'</svg>'}
function showTrain(){const plans=[...new Set(D.train.flatMap(P=>P.points.map(p=>p.plan)))];
 $('ps-pleg').innerHTML=plans.map(p=>`<label><input type="checkbox" data-k="p:${e(p)}"${on('p:'+p)?' checked':''}><i style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${PC[p]||'#888'};margin:0 4px;vertical-align:-1px"></i>${e(PN[p]||p)}</label>`).join('');
 $('ps-train').innerHTML=D.train.map(P=>`<div class="ps-card"><h4>${e(P.name)} · ${e(P.panel||'')} · ${P.points.length} configurations</h4>${scatter(P)}</div>`).join('')}
$('ps-tabs').innerHTML=D.test.map(T=>`<button data-ds="${T.ds}">${e(T.name)}${T.status==='complete'?'':' (pending)'}</button>`).join('');
$('ps-tabs').addEventListener('click',ev=>{const b=ev.target.closest('button');if(b){cur=b.dataset.ds;showTest()}});
$('pool-steps').addEventListener('change',ev=>{const c=ev.target;if(c.name==='ps-o'){succ=c.value==='succ';showTest();showTrain();return}
 if(c.dataset&&c.dataset.k){if(c.checked)HID.delete(c.dataset.k);else HID.add(c.dataset.k);if(c.dataset.k.startsWith('p:'))showTrain();else showTest()}});
showTest();showTrain();})();</script>
<p class="small">Generated __TIME__ · trimem/experiments/render_pool_steps.py</p></section>
<!-- pool-agent-steps:end -->"""


def main() -> int:
    data = collect()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    from trimem.experiments import fixed_plans_page as fp

    sec = (
        SECTION.replace("__DATA__", blob)
        .replace("__FPJS__", fp.js())
        .replace("__TIME__", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
    )
    with PAGE.with_suffix(".lock").open(
        "a"
    ) as lock:  # page lock shared by every section renderer
        fcntl.flock(lock, fcntl.LOCK_EX)
        page = PAGE.read_text()
        if START in page:
            page = page[: page.index(START)] + sec + page[page.index(END) + len(END) :]
        elif BEFORE in page:
            page = page.replace(BEFORE, sec + "\n" + BEFORE, 1)
        else:
            page = page.replace("<footer>", sec + "\n<footer>", 1)
        tmp = PAGE.with_suffix(".ps.tmp.html")
        tmp.write_text(page)
        tmp.replace(PAGE)
    for t in data["test"]:
        print(
            t["name"],
            t["status"],
            [(a["label"], a["steps"] and round(a["steps"], 2)) for a in t["arms"]],
        )
    for p in data["train"]:
        print(p["name"], len(p["points"]), "points,", p["skipped"], "without steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
