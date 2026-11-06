"""Fixed-plan reference arms for the qa-trajectory.html TEST panels (user request 2026-10-23).

Shared data layer of the fixed-plan points / rows / bars that render_atlas_results ("Three datasets"),
render_profiling_points (test panels), render_cost_saving_hv (equal-cost / equal-quality comparisons),
render_pool_steps (steps bars) and difficulty_report draw next to No-Memory / VGR (references) and Opt-OneConfig /
Opt-PerQuery: each plan's DEFAULT configuration on every test task (``fixed_plans_test`` for ALFWorld / ScienceWorld,
``tau2.fixed_plans_test`` for τ²). Until a dataset's ``report.json`` exists (and is complete) its table says
``pending`` and has no arms; the renderers then show a pending note only.

* ALFWorld / ScienceWorld (``agentenv_table``): per-task rows re-read with ``fixed_plans_test.pair_rows`` (own pairs
  + reused v7 / c16 pairs), latency in the LLM-call span definition (the page's plotted latency; wall kept as
  ``lat_wall``), gain vs the controlled-v1 NoMemory; every arm cross-checked against report.json (SystemExit on any
  disagreement). Schema = the v7 deployed-arm schema of ``sci_pool_page.deployed_arm`` with kind / mk ``plan``.
* τ² (``tau2_table``): per-task cost split from the pair result files (``tau2_pool_test_table.episode_costs``), paired
  vs the trial-0 NoMemory; schema = the ``tau2_pool_test_table`` arm schema + ``gain`` / ``steps``.

Markers (``STYLE`` / ``JS``): one shape per plan, the plan colour of the training scatter. READ-ONLY.
"""

from __future__ import annotations

import json
import math
import statistics as st
from functools import lru_cache

TOL = 1e-6
# plan -> colour (= the training-scatter plan colours) and marker (distinct from the □ / ★ / ◆ / ● Opt and
# reference markers): Rank variants use the rotated / sibling shape of their plan
STYLE = {
    "VGR": {"color": "#2a6fb3", "marker": "tri-up", "glyph": "▲"},
    "VGR+Rank": {"color": "#1f9aa6", "marker": "tri-down", "glyph": "▼"},
    "P2": {"color": "#8f5bb5", "marker": "hex", "glyph": "⬢"},
    "P2+Rank": {"color": "#c0558a", "marker": "pent", "glyph": "⬟"},
    "P1": {"color": "#6b8e23", "marker": "plus", "glyph": "✚"},
    "P1+Rank": {"color": "#b8862b", "marker": "xcross", "glyph": "✖"},
}
NOTE = (
    "Fixed-plan reference arms = each plan's DEFAULT configuration (seed 8 / depth 2 / fanout 4 / top-k 5, "
    "success+failure memory, SemFilter threshold 0.6, SemFilter v2) on every test task, same runtime as the test arms."
)

# JS helpers shared by the renderers: FPS (styles), fpShape(plan, cx, cy, r, attrs, inner) -> SVG polygon,
# fpGlyph(plan) -> a 13 px inline SVG icon of the plan marker. ``__FPS__`` is replaced by ``js()``.
_JS = r"""const FPS=__FPS__;
function fpPts(m,cx,cy,r){const P=a=>a.map(([x,y])=>(cx+x*r).toFixed(1)+','+(cy+y*r).toFixed(1)).join(' '),N=(n,o)=>P([...Array(n)].map((_,i)=>[1.15*Math.cos(2*Math.PI*i/n+o),1.15*Math.sin(2*Math.PI*i/n+o)])),t=.38;
 return m==='tri-up'?P([[0,-1.3],[1.15,.75],[-1.15,.75]]):m==='tri-down'?P([[0,1.3],[1.15,-.75],[-1.15,-.75]]):m==='hex'?N(6,0):m==='pent'?N(5,-Math.PI/2)
  :m==='plus'?P([[-t,-1.2],[t,-1.2],[t,-t],[1.2,-t],[1.2,t],[t,t],[t,1.2],[-t,1.2],[-t,t],[-1.2,t],[-1.2,-t],[-t,-t]])
  :P([[-1.1,-.6],[-.6,-1.1],[0,-.5],[.6,-1.1],[1.1,-.6],[.5,0],[1.1,.6],[.6,1.1],[0,.5],[-.6,1.1],[-1.1,.6],[-.5,0]])}
function fpShape(plan,cx,cy,r,attrs,inner){const s=FPS[plan]||{};return `<polygon points="${fpPts(s.marker||'tri-up',cx,cy,r)}" ${attrs}>${inner||''}</polygon>`}
function fpGlyph(plan,c){const s=FPS[plan]||{},col=c||s.color||'#888';return `<svg viewBox="-8 -8 16 16" width="13" height="13" style="display:inline-block;width:13px;height:13px;vertical-align:-2px">${fpShape(plan,0,0,5.6,`fill="${col}" stroke="var(--fg)" stroke-width=".8"`,'')}</svg>`}
"""


def js() -> str:
    return _JS.replace("__FPS__", json.dumps(STYLE, ensure_ascii=False))


def _se(xs: list[float]) -> float | None:
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def _check(ds: str, name: str, what: str, mine, theirs) -> None:
    if mine is None and theirs is None:
        return
    if (
        mine is None
        or theirs is None
        or not math.isclose(mine, theirs, rel_tol=TOL, abs_tol=TOL)
    ):
        raise SystemExit(
            f"{ds} fixed plans {name}: {what} {mine} disagrees with report.json ({theirs})"
        )


def _rel(p) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    try:
        return str(Path(p).resolve().relative_to(root))
    except ValueError:
        return str(p)


def pending(ds: str, status: str, absent: dict | None = None) -> dict:
    return {
        "ds": ds,
        "status": status,
        "arms": [],
        "absent": absent or {},
        "note": NOTE,
    }


# ------------------------------------------------------------------ ALFWorld / ScienceWorld


@lru_cache(maxsize=2)
def agentenv_table(ds: str) -> dict:
    """The page's fixed-plan arms of ALFWorld / ScienceWorld (status ``complete`` / ``pending ...``)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import fixed_plans_test as fpt

    base = fpt.run_base(ds)
    rp = fpt.out_dir(base) / "report.json"
    if not rp.exists():
        return pending(ds, "fixed-plan test pending (no report.json yet)")
    rep = json.loads(rp.read_text())
    if rep.get("protocol") != fpt.PROTOCOL:
        raise SystemExit(f"{rp}: protocol {rep.get('protocol')!r} != {fpt.PROTOCOL!r}")
    tab = json.loads((fpt.out_dir(base) / "routing.json").read_text())
    wall, span, src = fpt.pair_rows(ds, base, tab)
    _, nm = fpt.nomemory_quality(ds)
    key = d.space.key
    arms, incomplete = [], []
    for name, a in tab["arms"].items():
        r = rep["arms"].get(name) or {}
        k = key(next(iter(a["routes"].values()))["config"])
        items = {}
        for q in a["routes"]:
            x = span.get((k, q))
            if x is None:
                continue
            items[q] = {
                "q": x["quality"],
                "tok": x["tokens"],
                "lat": x["e2e_ms"] / 1000,
                "lat_wall": x["e2e_wall_ms"] / 1000,
                "tsem": x["tsem"],
                "ret": x["ret_ms"] / 1000,
                "steps": x.get("steps"),
                "cfg": k,
            }
        if len(items) != len(a["routes"]) or r.get("covered") != r.get("n"):
            incomplete.append(
                {
                    "id": name,
                    "label": a["label"],
                    "covered": len(items),
                    "n": len(a["routes"]),
                }
            )
            continue
        q = [x["q"] for x in items.values()]
        g = [x["q"] - nm[t] for t, x in items.items() if t in nm]
        stp = [x["steps"] for x in items.values() if x["steps"] is not None]
        ss = [
            x["steps"]
            for x in items.values()
            if x["steps"] is not None and x["q"] >= 1.0
        ]
        mine = {
            "q": st.mean(q),
            "tok": st.mean(x["tok"] for x in items.values()),
            "lat": st.mean(x["lat"] for x in items.values()),
            "lat_wall": st.mean(x["lat_wall"] for x in items.values()),
            "gain": st.mean(g) if g else None,
            "steps": st.mean(stp) if stp else None,
        }
        for m, mm in (
            ("q", "quality"),
            ("tok", "tokens"),
            ("lat", "latency_span_s"),
            ("lat_wall", "latency_s"),
            ("gain", "paired_gain_vs_nomemory"),
            ("steps", "steps"),
        ):
            _check(ds, name, mm, mine[m], r.get(mm))
        sty = STYLE[a["plan"]]
        arms.append(
            {
                "id": name,
                "label": a["label"] + " · plan default",
                "short": a["label"],
                "kind": "plan",
                "mk": "plan",
                "plan": a["plan"],
                "router": None,
                "color": sty["color"],
                "weights": None,
                "slo": None,
                "folds": None,
                "n": len(items),
                **mine,
                "q_std": None,
                "q_se": _se(q),
                "tok_std": None,
                "lat_std": None,
                "lat_wall_std": None,
                "gain_std": None,
                "gain_se": _se(g) if g else None,
                "steps_std": None,
                "steps_se": _se(stp) if stp else None,
                "steps_success": st.mean(ss) if ss else None,
                "steps_success_std": None,
                "se_tasks": {
                    "q": _se(q),
                    "gain": _se(g) if g else None,
                    "steps": _se(stp) if stp else None,
                },
                "tsem": st.mean(x["tsem"] for x in items.values()),
                "ret": st.mean(x["ret"] for x in items.values()),
                "distinct": 1,
                "plan_shares": {a["plan"]: 1.0},
                "pair_sources": r.get("pair_sources"),
                "cv": None,
                "items": items,
            }
        )
    status = (
        "complete"
        if not incomplete
        else f"fixed-plan test incomplete: {len(incomplete)} of {len(tab['arms'])} arms"
    )
    return {
        "ds": ds,
        "status": status,
        "arms": arms,
        "incomplete": incomplete,
        "absent": {},
        "note": NOTE,
        "dir": _rel(fpt.out_dir(base)),
        "generated": rep.get("generated"),
        "pairs": rep.get("pairs"),
        "pair_sources": rep.get("pair_sources"),
        "concurrency": rep.get("concurrency"),
        "experiment": rep.get("experiment"),
    }


# ------------------------------------------------------------------ τ²


def tau2_nomemory_items() -> dict[str, dict]:
    """Trial-0 NoMemory episodes of the reused 4-trial arm (= tau2_pool_test_table's baseline), cost split."""
    from trimem.experiments import tau2_pool_test_table as tt
    from trimem.experiments.tau2 import pool_profile as pp
    from trimem.experiments.tau2 import runner as R

    out = {}
    for k in pp.test_keys():
        r = R.read_result(R.result_path(pp.REUSE / "NoMemory", *k.split(":", 1), 0))
        if r is None:
            raise SystemExit(f"{pp.REUSE / 'NoMemory'}: trial 0 of {k} missing")
        out[k] = tt.episode_costs(r, 0.0)
    return out


@lru_cache(maxsize=1)
def tau2_table() -> dict:
    """The page's τ² fixed-plan arms (status ``complete`` / ``pending ...``); P1 listed in ``absent``."""
    from trimem.experiments import tau2_pool_test_table as tt
    from trimem.experiments.tau2 import fixed_plans_test as tfp
    from trimem.experiments.tau2 import pool_profile as pp
    from trimem.experiments.tau2 import runner as R

    rp = tfp.out_dir() / "report.json"
    if not rp.exists():
        return pending(
            "tau2", "fixed-plan test pending (no report.json yet)", tfp.ABSENT
        )
    rep = json.loads(rp.read_text())
    if rep.get("protocol") != tfp.PROTOCOL:
        raise SystemExit(f"{rp}: protocol {rep.get('protocol')!r} != {tfp.PROTOCOL!r}")
    tab = json.loads((tfp.out_dir() / "routing.json").read_text())
    src = tfp.pair_sources(tab)
    base = tau2_nomemory_items()
    arms, incomplete = [], []
    for name, a in tab["arms"].items():
        r = rep["arms"].get(name) or {}
        k = pp.space.key(next(iter(a["routes"].values()))["config"])
        items, steps = {}, {}
        for q in a["routes"]:
            if (k, q) not in src:
                continue
            raw = R.read_result(src[(k, q)][1])
            items[q] = {**tt.episode_costs(raw, 0.0), "plan": a["plan"], "cfg": k}
            steps[q] = pp.episode_steps(raw)
        if len(items) != len(a["routes"]) or r.get("covered") != r.get("n"):
            incomplete.append(
                {
                    "id": name,
                    "label": a["label"],
                    "covered": len(items),
                    "n": len(a["routes"]),
                }
            )
            continue
        s = tt.summarize(items, base)
        _check("τ²", name, "pass^1", s["q"], r.get("quality"))
        _check("τ²", name, "tokens", s["tok"], r.get("tokens"))
        _check("τ²", name, "latency", s["lat"], r.get("latency_s"))
        _check(
            "τ²",
            name,
            "paired gain",
            s["vs_nomemory"]["delta"],
            r.get("paired_gain_vs_nomemory"),
        )
        stp = [v for v in steps.values() if v is not None]
        ss = [v for q, v in steps.items() if v is not None and items[q]["q"] >= 1.0]
        sty = STYLE[a["plan"]]
        arms.append(
            {
                "id": name,
                "label": a["label"] + " · plan default",
                "short": a["label"],
                "kind": "plan",
                "mk": "plan",
                "plan": a["plan"],
                "color": sty["color"],
                "slo": None,
                "weights": None,
                "distinct_configs": 1,
                "distinct": 1,
                "items": items,
                **s,
                "gain": s["vs_nomemory"]["delta"],
                "gain_se": s["vs_nomemory"]["se"],
                "covered": len(items),
                "steps": st.mean(stp) if stp else None,
                "steps_se": _se(stp) if stp else None,
                "steps_success": st.mean(ss) if ss else None,
                "pair_sources": r.get("pair_sources"),
            }
        )
    status = (
        "complete"
        if not incomplete
        else f"fixed-plan test incomplete: {len(incomplete)} of {len(tab['arms'])} arms"
    )
    return {
        "ds": "tau2",
        "status": status,
        "arms": arms,
        "incomplete": incomplete,
        "absent": tfp.ABSENT,
        "note": NOTE,
        "dir": _rel(tfp.out_dir()),
        "generated": rep.get("generated"),
        "pairs": rep.get("pairs"),
        "pair_sources": rep.get("pair_sources"),
        "concurrency": rep.get("test_concurrency"),
        "experiment": rep.get("experiment"),
    }


def table(ds: str) -> dict:
    """Never raises for a missing run: a renderer must keep working while the fixed-plan test is pending."""
    return tau2_table() if ds == "tau2" else agentenv_table(ds)


def strip(t: dict) -> dict:
    return {
        **t,
        "arms": [
            {k: v for k, v in a.items() if k != "items"} for a in t.get("arms", [])
        ],
    }


def status_line(t: dict) -> str:
    """One English line for a panel's note: complete / pending / incomplete (+ absent plans)."""
    ab = "; ".join(f"{p} absent: {why}" for p, why in (t.get("absent") or {}).items())
    if t.get("status") == "complete":
        s = f"Fixed-plan arms: {len(t['arms'])} plan defaults ({t.get('dir')}, pair sources {t.get('pair_sources')})."
    else:
        s = f"Fixed-plan arms (VGR, VGR+Rank, VGR+SemFilter, VGR+SemFilter+Rank, V+SemFilter+G(+Rank)): {t.get('status')}."
    return s + (f" {ab}." if ab else "")
