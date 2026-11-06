"""Opt-PerQuery weight sweep for the qa-trajectory.html TEST panels (user request 2026-10-23).

Data layer + shared JS of the ALFWorld / ScienceWorld / WebShop sweep plots that render_atlas_results
("Three datasets") and render_profiling_points (test panels) draw: the deployed BRANE router at the configured weights
(``perquery_sweep``). Per dataset two plots (quality x tokens, quality x latency; latency = LLM span + retrieval +
router overhead, the page's plotted definition):

* Q-T curve (w_L = 0, ordered by w_T) and Q-L curve (w_T = 0, ordered by w_L), connected, on BOTH plots;
* all measured sweep points, with the test Pareto front highlighted (dashed);
* a full weight/metric table whose rows locate the corresponding point on both plots;
* overlays from the panel's own arms: the 4 deployed Opt-PerQuery points, Opt-OneConfig SLO points, No-Memory / VGR
  references, fixed-plan points; legend toggles per layer; tooltips with weights, metrics and plan shares.

Until ``report_pq_sweep.json`` is COMPLETE (every weight's 200 routed pairs measured) the panel shows a pending
line only (``PQ_SWEEP_PAGE_PARTIAL=1`` draws the measured weights of a partial report, for checks). READ-ONLY.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

PARTIAL_ENV = "PQ_SWEEP_PAGE_PARTIAL"
DATASETS = (
    "alfworld",
    "sciworld",
    "webshop",
)  # webshop: only attached once its v7 run is shown (2026-10-23)


def _rel(p) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    try:
        return str(Path(p).resolve().relative_to(root))
    except ValueError:
        return str(p)


def point(r: dict) -> dict:
    """report point -> compact page point (latency = span definition). Pure."""
    return {
        "arm": r["arm"],
        "w": r["weights"],
        "g": r["group"],
        "q": r["quality"],
        "q_se": r["quality_se"],
        "gain": r["paired_gain_vs_nomemory"],
        "gain_se": r["paired_gain_se"],
        "tok": r["tokens"],
        "lat": r["latency_span_s"],
        "lat_wall": r["latency_s"],
        "steps": r["steps"],
        "distinct": r["distinct_configs"],
        "shares": r["plan_shares"],
        "pred": r["pred"],
    }


def from_report(ds: str, rep: dict, partial: bool) -> dict:
    """Page table of one report (pure): complete weights only; pending unless complete (or ``partial``)."""
    pts = [point(r) for r in rep["points"].values() if r["complete"]]
    n, k = rep["weights"], len(pts)
    done = rep.get("complete") and k == n
    base = {
        "ds": ds,
        "n_weights": n,
        "complete_weights": k,
        "generated": rep.get("generated"),
        "artifact": (rep.get("artifact") or {}).get("sha", "")[:12],
        "distinct_routings": rep.get("distinct_routings"),
        "identical": rep.get("identical_routings") or [],
        "baseline": rep.get("baseline", {}).get("quality"),
        "pair_sources": rep.get("pair_sources"),
    }
    if not done and not partial:
        return {
            **base,
            "status": f"pending: {k}/{n} weights fully measured (sweep pairs not run yet)",
            "points": [],
        }
    have = {p["arm"] for p in pts}
    return {
        **base,
        "status": "complete" if done else f"PARTIAL ({k}/{n} weights; check view only)",
        "points": pts,
        "qt": [a for a in rep["slices"]["QT"] if a in have],
        "ql": [a for a in rep["slices"]["QL"] if a in have],
    }


@lru_cache(maxsize=3)
def table(ds: str) -> dict:
    """The page's three-dataset sweep data; never raises for a missing report."""
    from trimem.experiments import perquery_sweep as pq

    if ds not in DATASETS:
        return {"ds": ds, "status": "not swept", "points": []}
    p = pq.out_dir(pq.run_base(ds)) / pq.REPORT
    # Promote only a complete rescan; keep showing the completed old sweep while it runs.
    newer = pq.run_base(ds) / "test_v7c16/pq_sweep_log2" / pq.REPORT
    if newer.exists() and json.loads(newer.read_text()).get("complete"):
        p = newer
    if not p.exists():
        return {
            "ds": ds,
            "status": "pending (no report_pq_sweep.json yet)",
            "points": [],
            "dir": _rel(p.parent),
        }
    rep = json.loads(p.read_text())
    if rep.get("protocol") != pq.PROTOCOL:
        raise SystemExit(f"{p}: protocol {rep.get('protocol')!r} != {pq.PROTOCOL!r}")
    return {
        **from_report(ds, rep, os.environ.get(PARTIAL_ENV) == "1"),
        "dir": _rel(p.parent),
    }


def overlays(arms: list[dict], q: str, tok: str, lat: str) -> list[dict]:
    """A panel's arms -> overlay points (deployed Opt-PerQuery, Opt-OneConfig, references, fixed plans). Pure."""
    out = []
    for a in arms:
        kind = a.get("kind")
        if kind not in ("ref", "oneconfig", "perquery", "plan") or a.get(q) is None:
            continue
        out.append(
            {
                "id": a.get("id"),
                "label": a.get("label"),
                "kind": kind,
                "q": a[q],
                "tok": a[tok],
                "lat": a[lat],
                "color": a.get("color"),
                "plan": a.get("plan"),
                "w": a.get("weights"),
            }
        )
    return out


def attach(ds: str, arms: list[dict], q: str, tok: str, lat: str) -> dict | None:
    """``pqs`` object for a panel (None for datasets without a sweep)."""
    if ds not in DATASETS:
        return None
    t = table(ds)
    return {**t, "line": status_line(t), "over": overlays(arms, q, tok, lat)}


def status_line(t: dict) -> str:
    if t.get("status") == "complete":
        return (
            f"Opt-PerQuery weight sweep: {t['n_weights']} weights (Q-T + Q-L curves), {t['distinct_routings']} "
            f"distinct routings, deployed router {t.get('artifact')} (no refit); {t.get('dir')}."
        )
    return f"Opt-PerQuery weight sweep (Q-T + Q-L curves): {t.get('status')}."


# JS: pqsHTML(S, uid) -> legend + two plot slots; pqsDraw(S, uid) draws them; toggles via [data-pqs] checkboxes
# (event delegation, bound once). Uses fpShape / FPS of fixed_plans_page.js() when present.
_JS = r"""const PQS_HIDE={},PQS_DATA={},PQS_SELECTED={};
const pqsE=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const PQS_C={grid:'#9bb5a6',front:'#2f8f6b',tokproj:'#d65a17',qt:'#3b6fb6',ql:'#8a4fb0',dep:'#2f8f6b',oc:'#c26a1b',ref:'#8a8f87'};
const PQS_L=[['grid','all sweep points'],['front','test Pareto front of the swept points'],['tokproj','token-front points on latency plot'],['qt','Q–T curve (w_L = 0; labels = w_T)'],['ql','Q–L curve (w_T = 0; labels = w_L)'],['dep','deployed Opt-PerQuery (4 weights)'],['oc','Opt-OneConfig SLO'],['ref','No-Memory / VGR'],['plan','fixed-plan defaults']];
const PQS_PN={VGR:'VGR','VGR+Rank':'VGR+Rank',P2:'VGR+SemFilter','P2+Rank':'VGR+SemFilter+Rank',P1:'V+SemFilter+G','P1+Rank':'V+SemFilter+G+Rank',NoMemory:'No-Memory'};
const pqsNum=(v,d=3)=>v==null?'—':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pqsWeight=p=>'('+p.w.join(', ')+')';
function pqsTable(S,uid){const ft=new Set(pqsFront(S.points,'tok').map(p=>p.arm)),fl=new Set(pqsFront(S.points,'lat').map(p=>p.arm));
 return `<details open class="pqs-table-wrap" style="margin-top:12px"><summary style="cursor:pointer;font-weight:600">All ${S.points.length} measured sweep weights and results</summary><p class="small">Weights are (w_Q, w_L, w_T). Click a point ID to locate it on both plots. Coincident points retain a separate row for every weight; Q–T and Q–L indicate the two measured Pareto fronts.</p><div style="max-height:380px;overflow:auto"><table class="pqs-table" style="width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap"><thead><tr>${['Point','(w_Q, w_L, w_T)','Quality ± SE','Tokens / episode','Latency (span, s)','Steps','Q–T front','Q–L front'].map(n=>`<th scope="col" style="text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)">${n}</th>`).join('')}</tr></thead><tbody>${S.points.map((p,i)=>`<tr data-pqs-row="${uid}:${i}"><td style="padding:6px 10px"><button type="button" data-pqs-select="${uid}:${i}" aria-pressed="false" style="cursor:pointer;font:inherit">W${String(i+1).padStart(2,'0')}</button></td>${[pqsWeight(p),pqsNum(p.q)+' ± '+pqsNum(p.q_se),pqsNum(p.tok,0),pqsNum(p.lat,2),pqsNum(p.steps,2),ft.has(p.arm)?'Yes':'—',fl.has(p.arm)?'Yes':'—'].map(v=>`<td style="padding:6px 10px;border-bottom:1px solid var(--line)">${pqsE(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>`}
function pqsHTML(S,uid){PQS_DATA[uid]=S;delete PQS_SELECTED[uid];const H=PQS_HIDE[uid]=PQS_HIDE[uid]||new Set();
 if(!S.points||!S.points.length)return `<p class="small">${pqsE(S.line||S.status)}</p>`;
 const sw=k=>k==='tokproj'?`<span style="color:${PQS_C.tokproj}">◎ </span>`:k==='grid'?`<i style="background:${PQS_C.grid};border-radius:50%;opacity:.6"></i>`:k==='front'?`<i style="background:${PQS_C.front};border-radius:50%"></i>`:k==='qt'||k==='ql'?`<i style="background:${PQS_C[k]};height:3px;vertical-align:3px"></i>`:k==='dep'?`<span style="color:${PQS_C.dep}">★ </span>`:k==='oc'?`<span style="color:${PQS_C.oc}">□ </span>`:k==='ref'?`<span style="color:${PQS_C.ref}">◆ </span>`:(typeof fpGlyph==='function'?fpGlyph('VGR')+' ':'▲ ');
 return `<p class="small">${pqsE(S.line||'')}</p><div class="pqs-leg" style="display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:2px 0 6px" role="group" aria-label="weight sweep layers">${PQS_L.map(([k,n])=>`<label style="cursor:pointer;white-space:nowrap"><input type="checkbox" data-pqs="${uid}:${k}"${H.has(k)?'':' checked'}> ${sw(k)}${n}</label>`).join('')}</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:16px"><div><h4 style="margin:0 0 4px;font-size:14px">Quality × tokens / episode</h4><div id="pqs-${uid}-tok"></div></div><div><h4 style="margin:0 0 4px;font-size:14px">Quality × latency / episode (LLM span, s)</h4><div id="pqs-${uid}-lat"></div></div></div>
<p class="small" id="pqs-${uid}-selection" aria-live="polite">All ${S.points.length} sweep weights, including dominated points, are listed below. Select a point ID to highlight its weight.</p>
<p class="small">Each curve point = the deployed router (frozen artifact ${pqsE(S.artifact||'')}, no refit) routing all test tasks at one of the ${S.n_weights} weights with U = w_Q·q̂ − w_L·ln lat̂ − w_T·ln tok̂ (zscore-log-v1); each (configuration, task) pair is one c16 episode shared by every weight that routes there. ${S.identical&&S.identical.length?S.identical.length+' group(s) of weights route identically. ':''}Front = the swept points no other swept point beats on both axes of that plot. Latency = LLM span + retrieval + router overhead (wall in tooltips). Hover for weights, metrics and routing shares.</p>${pqsTable(S,uid)}`}
function pqsFront(pts,ax){return pts.filter(p=>!pts.some(o=>o!==p&&o.q>=p.q&&o[ax]<=p[ax]&&(o.q>p.q||o[ax]<p[ax]))).sort((a,b)=>a[ax]-b[ax])}
function pqsTip(S,p){const f3=v=>v==null?'—':v.toFixed(3),k=v=>Math.round(v).toLocaleString('en-US'),s1=v=>v==null?'—':v.toFixed(1);
 return `Opt-PerQuery w = (w_Q ${p.w[0]}, w_L ${p.w[1]}, w_T ${p.w[2]})${S.dep&&S.dep.has(p.arm)?' · deployed weight':''}\nquality ${f3(p.q)} ± ${f3(p.q_se)} SE · gain vs No-Memory ${p.gain>=0?'+':''}${f3(p.gain)} ± ${f3(p.gain_se)}\ntokens ${k(p.tok)} · latency ${s1(p.lat)} s (span), ${s1(p.lat_wall)} s (wall)${p.steps!=null?' · steps '+p.steps.toFixed(2):''}\n${p.distinct} configurations · routing group ${p.g}\nshares: ${Object.entries(p.shares).map(([x,v])=>(PQS_PN[x]||x)+' '+(100*v).toFixed(1)+'%').join(', ')}\npredicted: quality ${f3(p.pred.quality)} · tokens ${k(p.pred.tokens)} · wall ${s1(p.pred.latency_s)} s`}
function pqsPlot(S,uid,ax){const el=document.getElementById(`pqs-${uid}-${ax}`);if(!el)return;const H=PQS_HIDE[uid],on=k=>!H.has(k),W=440,Ht=290,m={l:46,r:14,t:10,b:36};
 const P=S.points,byA=Object.fromEntries(P.map(p=>[p.arm,p])),O=(S.over||[]).filter(o=>on(o.kind==='perquery'?'dep':o.kind==='oneconfig'?'oc':o.kind));
 // overlays far below the grid (e.g. No-Memory) would squash it: they sit clamped at the bottom edge (↓, true values in the label / tooltip)
 const gq=Math.min(...P.map(p=>p.q)),gq1=Math.max(...P.map(p=>p.q)),lo=gq-Math.max(.12,2*(gq1-gq)),out=o=>o.q<lo,Oin=O.filter(o=>!out(o));
 const all=P.map(p=>[p[ax],p.q]).concat(Oin.map(o=>[o[ax],o.q]));let x0=Math.min(...all.map(v=>v[0])),x1=Math.max(...all.map(v=>v[0])),y0=Math.min(...all.map(v=>v[1])),y1=Math.max(...all.map(v=>v[1]));
 const px=(x1-x0)*.06||1,py=(y1-y0)*.08||.02;x0-=px;x1+=px;y0-=py;y1+=py;const X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>Ht-m.b-(v-y0)/(y1-y0)*(Ht-m.t-m.b);
 const fx=ax==='tok'?(v=>v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0)):(v=>v.toFixed(1));
 let s=`<svg viewBox="0 0 ${W} ${Ht}" role="img" aria-label="weight sweep quality × ${ax==='tok'?'tokens':'latency'}" style="display:block;width:100%;height:auto">`;
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${m.l-5}" y="${Y(v)+4}" text-anchor="end" style="fill:var(--muted);font-size:11px">${v.toFixed(3)}</text>`}
 for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;s+=`<text x="${X(v)}" y="${Ht-m.b+15}" text-anchor="middle" style="fill:var(--muted);font-size:11px">${fx(v)}</text>`}
 s+=`<text x="${(W+m.l)/2}" y="${Ht-3}" text-anchor="middle" style="fill:var(--muted);font-size:11px">${ax==='tok'?'tokens / episode':'latency s / episode (LLM span)'} (left = cheaper)</text>`;
 const F=pqsFront(P,ax),fs=new Set(F.map(p=>p.arm)),tip=p=>`<title>${pqsE(pqsTip(S,p)+(fs.has(p.arm)?'\non the test Pareto front of the swept points (this plot)':''))}</title>`;
 const select=p=>`data-pqs-select="${uid}:${P.indexOf(p)}" style="cursor:pointer"`;
 if(on('grid'))s+=`<g data-pqs-all-points="true">${P.map((p,i)=>`<circle data-pqs-sweep-point="${pqsE(p.arm)}" data-pqs-select="${uid}:${i}" cx="${X(p[ax]).toFixed(1)}" cy="${Y(p.q).toFixed(1)}" r="4.7" fill="${PQS_C.grid}" stroke="var(--fg)" stroke-width=".7" style="cursor:pointer">${tip(p)}</circle>`).join('')}</g>`;
 const line=(arms,c,wi)=>{const ps=arms.map(a=>byA[a]).filter(Boolean);if(ps.length<2)return '';return ps.map(p=>p.w[wi]>0?`<text x="${(X(p[ax])+5).toFixed(1)}" y="${(Y(p.q)+(wi===2?-6:12)).toFixed(1)}" style="fill:${c};font-size:9.5px">${p.w[wi]}</text>`:'').join('')+`<polyline fill="none" stroke="${c}" stroke-width="1.8" stroke-linejoin="round" points="${ps.map(p=>X(p[ax]).toFixed(1)+','+Y(p.q).toFixed(1)).join(' ')}"/>`+ps.map(p=>`<circle ${select(p)} cx="${X(p[ax]).toFixed(1)}" cy="${Y(p.q).toFixed(1)}" r="3.4" fill="var(--panel)" stroke="${c}" stroke-width="1.6">${tip(p)}</circle>`).join('')};
 if(on('qt'))s+=line(S.qt||[],PQS_C.qt,2);if(on('ql'))s+=line(S.ql||[],PQS_C.ql,1);
 if(on('front')){s+=`<polyline fill="none" stroke="${PQS_C.front}" stroke-width="1.3" stroke-dasharray="4 3" points="${F.map(p=>X(p[ax]).toFixed(1)+','+Y(p.q).toFixed(1)).join(' ')}"/>`;F.forEach(p=>{s+=`<circle ${select(p)} cx="${X(p[ax]).toFixed(1)}" cy="${Y(p.q).toFixed(1)}" r="4.2" fill="${PQS_C.front}" stroke="var(--panel)" stroke-width="1">${tip(p)}</circle>`})}
 O.forEach(o=>{const cut=out(o),cx=Math.min(Math.max(X(o[ax]),m.l+6),W-m.r-6),cy=cut?Ht-m.b-8:Y(o.q),t=`<title>${pqsE(`${o.label}${o.w?' (w = '+o.w.join(' / ')+')':''}\nquality ${o.q.toFixed(3)} · tokens ${Math.round(o.tok).toLocaleString('en-US')} · latency ${o.lat.toFixed(1)} s (span)`)}</title>`,c=o.color||PQS_C[o.kind]||'#888';
  if(o.kind==='perquery')s+=`<text x="${cx}" y="${cy+6.5}" text-anchor="middle" style="font-size:19px;fill:${c}" stroke="var(--fg)" stroke-width=".5">★${t}</text>`;
  else if(o.kind==='oneconfig')s+=`<rect x="${cx-5.5}" y="${cy-5.5}" width="11" height="11" fill="none" stroke="${c}" stroke-width="2">${t}</rect>`;
  else if(o.kind==='ref')s+=`<rect x="${cx-5}" y="${cy-5}" width="10" height="10" transform="rotate(45 ${cx} ${cy})" fill="${c}" stroke="var(--panel)">${t}</rect>`;
  else if(o.kind==='plan'&&typeof fpShape==='function')s+=fpShape(o.plan,cx,cy,5.5,`fill="${c}" fill-opacity=".85" stroke="var(--fg)" stroke-width=".7"`,t);
  else s+=`<circle cx="${cx}" cy="${cy}" r="5" fill="${c}">${t}</circle>`;
  if(o.kind==='ref')s+=`<text x="${cx-8}" y="${cut?cy+4:cy-7}" text-anchor="end" style="fill:var(--fg);font-size:10px">${pqsE(o.label+(cut?` ↓ ${o.q.toFixed(3)}${X(o[ax])>W-m.r-6||X(o[ax])<m.l+6?' · '+fx(o[ax]):''}`:''))}</text>`});
 if(ax==='lat'&&on('tokproj')){
  const T=pqsFront(P,'tok');
  s+=`<g data-pqs-token-projection="true"><polyline fill="none" stroke="${PQS_C.tokproj}" stroke-width="1.4" stroke-dasharray="2 3" points="${T.map(p=>X(p.lat).toFixed(1)+','+Y(p.q).toFixed(1)).join(' ')}"/>`;
  T.forEach(p=>{s+=`<circle ${select(p)} data-pqs-token-front="${pqsE(p.arm)}" cx="${X(p.lat).toFixed(1)}" cy="${Y(p.q).toFixed(1)}" r="6" fill="none" stroke="${PQS_C.tokproj}" stroke-width="2"><title>${pqsE('Token Pareto-front policy shown at its measured latency; orange connection follows token order, not the latency Pareto front.\n'+pqsTip(S,p))}</title></circle>`});
  s+='</g>';
 }
 const selected=P[PQS_SELECTED[uid]];if(selected){const cx=X(selected[ax]),cy=Y(selected.q);s+=`<g data-pqs-highlight="${pqsE(selected.arm)}" style="pointer-events:none"><circle cx="${cx}" cy="${cy}" r="9" fill="none" stroke="#bd3e5d" stroke-width="2.5"/><text x="${Math.min(cx+12,W-m.r-28)}" y="${Math.max(cy-12,m.t+10)}" style="fill:#bd3e5d;font-size:11px;font-weight:700;paint-order:stroke;stroke:var(--panel);stroke-width:3px">W${String(PQS_SELECTED[uid]+1).padStart(2,'0')}</text></g>`}
 el.innerHTML=s+'</svg>'}
function pqsDraw(S,uid){if(!S||!S.points||!S.points.length)return;S.dep=new Set((S.over||[]).filter(o=>o.kind==='perquery'&&o.w).map(o=>'Opt-PerQuery-w'+o.w.map(v=>+v).join('-')));pqsPlot(S,uid,'tok');pqsPlot(S,uid,'lat')}
// Each embedding has private chart state; handle only the UIDs owned by this script.
document.addEventListener('change',ev=>{const c=ev.target;if(!c||!c.dataset||!c.dataset.pqs)return;const [uid,k]=c.dataset.pqs.split(':'),H=PQS_HIDE[uid];if(!H)return;if(c.checked)H.delete(k);else H.add(k);pqsDraw(PQS_DATA[uid],uid)});
document.addEventListener('click',ev=>{const c=ev.target.closest&&ev.target.closest('[data-pqs-select]');if(!c)return;const [uid,raw]=c.dataset.pqsSelect.split(':'),S=PQS_DATA[uid],i=Number(raw);if(!S||!Number.isInteger(i)||!S.points[i])return;PQS_SELECTED[uid]=i;pqsDraw(S,uid);document.querySelectorAll('[data-pqs-row]').forEach(row=>{if(!row.dataset.pqsRow.startsWith(uid+':'))return;const active=row.dataset.pqsRow===uid+':'+i;row.style.background=active?'rgba(189,62,93,.12)':'';row.querySelector('button').setAttribute('aria-pressed',String(active))});const label=document.getElementById(`pqs-${uid}-selection`);if(label)label.textContent=`Selected W${String(i+1).padStart(2,'0')}: (w_Q, w_L, w_T) = ${pqsWeight(S.points[i])}. Highlighted on both plots; coincident weights have separate table rows.`});
"""


def js() -> str:
    return _JS
