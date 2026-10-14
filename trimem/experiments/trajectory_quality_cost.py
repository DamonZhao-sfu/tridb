"""Checkpoint-aligned Math/ALE quality versus observed API token costs.

Private scores are read-only reporting inputs. No interpolation, cross-task
quality aggregation, or substitution of public fitness for private performance.
"""

import json
import math
import time
from pathlib import Path

from trimem.experiments import math_trajectory_run as base

ROOT = base.ROOT
MATH = ROOT / "results/trimem_v0.1.0/math_trajectory_extension_20261013"
ALE = ROOT / "results/trimem_v0.1.0/ale_trajectory_v4_20261013"
OUT = ROOT / "triMEM_optimizer/it20_summary/trajectory_quality_cost.json"
CHECKPOINTS = [10, 20, 30, 40, 50]
ALE_TASKS = ["ahc008", "ahc011", "ahc024", "ahc025", "ahc026"]
START = "<!-- trajectory-quality-cost:start -->"
END = "<!-- trajectory-quality-cost:end -->"


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def paths_of(cell):
    paths = cell.get("paths") or {"single": cell.get("path")}
    return [Path(p) for p in dict.fromkeys(paths.values()) if p]


def collect_logs(cells):
    """Ownership spans every checkpoint/stage to avoid charging shared calls twice."""
    logs, owners = {}, {}
    for cell in cells:
        key = (cell["task"], cell["plan"])
        records = {name: [] for name in ["generation", "dispatch", "retrieval"]}
        for path in paths_of(cell):
            for name, filename in [
                ("generation", "llm_telemetry.jsonl"),
                ("dispatch", "generation_prompt_index.jsonl"),
                ("retrieval", "trajectory_retrieval.jsonl"),
            ]:
                records[name].extend(base.rows(path / filename))
        logs[key] = records
        op = {"P1": "SemFilter", "P2": "SemFilter", "SemRankOnly": "SemScore"}.get(
            key[1]
        )
        for row in records["retrieval"]:
            if op and row.get("context_sha256"):
                owners.setdefault((op, row["context_sha256"]), set()).add(key)
    return logs, owners


def checkpoint_cost(key, checkpoint, records, semantic, owners):
    hashes = {}
    for row in records["dispatch"]:
        if row.get("iteration") is not None:
            hashes.setdefault(row["sha256"], set()).add(int(row["iteration"]))
    generation, ambiguous_generation = [], 0
    for row in records["generation"]:
        iterations = hashes.get(row.get("prompt_sha256"), set())
        if iterations and min(iterations) <= checkpoint:
            if max(iterations) > checkpoint:
                ambiguous_generation += 1
            else:
                generation.append(row)
    context_iterations = {}
    for row in records["retrieval"]:
        if row.get("iteration") is not None:
            context_iterations.setdefault(row.get("context_sha256"), set()).add(
                int(row["iteration"])
            )
    op = {"P1": "SemFilter", "P2": "SemFilter", "SemRankOnly": "SemScore"}.get(key[1])
    semantic_rows, ambiguous_semantic = [], 0
    if op:
        for row in semantic:
            iterations = context_iterations.get(row.get("context_sha256"), set())
            if row.get("op") != op or not iterations or min(iterations) > checkpoint:
                continue
            if (
                len(owners.get((op, row.get("context_sha256")), set())) != 1
                or max(iterations) > checkpoint
            ):
                ambiguous_semantic += 1
            else:
                semantic_rows.append(row)
    generation_unknown = sum(
        not all(finite(r.get(k)) for k in ["prompt_tokens", "completion_tokens"])
        for r in generation
    )
    semantic_unknown = sum(
        not all(
            finite((r.get("usage") or {}).get(k))
            for k in ["prompt_tokens", "completion_tokens"]
        )
        for r in semantic_rows
    )

    def tokens(rows):
        return sum(
            r.get(k, 0) or 0
            for r in rows
            for k in ["prompt_tokens", "completion_tokens"]
            if finite(r.get(k))
        )

    gen = tokens(generation)
    sem = tokens([r.get("usage") or {} for r in semantic_rows])
    seen = {r.get("prompt_sha256") for r in records["generation"]}
    missing_dispatches = sum(
        min(iterations) <= checkpoint and digest not in seen
        for digest, iterations in hashes.items()
    )
    unlinked = sum(r.get("prompt_sha256") not in hashes for r in records["generation"])
    complete = bool(generation) and not (
        generation_unknown
        or semantic_unknown
        or ambiguous_generation
        or ambiguous_semantic
        or missing_dispatches
        or unlinked
    )
    if op and not semantic_rows:
        complete = False  # Cache hit and missing telemetry cannot be distinguished.
    return {
        "generation_tokens": gen if generation else None,
        "semantic_tokens_observed": sem,
        "total_tokens_observed": gen + sem if generation else None,
        "usage_complete": complete,
        "generation_calls": len(generation),
        "semantic_calls": len(semantic_rows),
        "unknown_usage_calls": generation_unknown + semantic_unknown,
        "ambiguous_calls_excluded": ambiguous_generation + ambiguous_semantic,
        "unlinked_generation_calls": unlinked,
        "dispatches_without_telemetry": missing_dispatches,
    }


def build_data():
    math_doc = base.read(MATH / "summary.json", {})
    ale_doc = base.read(ALE / "summary.json", {})
    math_cells = math_doc.get("cells", [])
    ale_cells = ale_doc.get("cells", [])
    cells = math_cells + ale_cells
    logs, owners = collect_logs(cells)
    sem_paths = [base.OUT / "semantic_api.jsonl", MATH / "semantic_api.jsonl"]
    sem_paths += [
        Path(p) for p in ale_doc.get("semantic_logs", [str(ALE / "semantic_api.jsonl")])
    ]
    semantic = [r for path in dict.fromkeys(sem_paths) for r in base.rows(path)]
    points = []
    for family, selected in [("Math", math_cells), ("ALE", ale_cells)]:
        for cell in selected:
            key = (cell["task"], cell["plan"])
            for checkpoint in CHECKPOINTS:
                record = {
                    "family": family,
                    "task": key[0],
                    "plan": key[1],
                    "iteration": checkpoint,
                }
                if family == "Math":
                    record["quality"] = cell.get(f"quality{checkpoint}")
                    record["metric"] = "Best valid fitness (evaluator-success proxy)"
                else:
                    items = cell.get("checkpoints", [])
                    if isinstance(items, dict):
                        items = [dict(v, iteration=int(k)) for k, v in items.items()]
                    cp = next(
                        (p for p in items if p.get("iteration") == checkpoint), {}
                    )
                    record.update(cp)
                    record["quality"] = cp.get("performance")
                    record["metric"] = "Private AtCoder performance"
                if finite(record.get("quality")):
                    record.update(
                        checkpoint_cost(key, checkpoint, logs[key], semantic, owners)
                    )
                    points.append(record)
    return {
        "updated_unix": time.time(),
        "points": points,
        "tasks": {
            "Math": [
                "heilbronn_convex_13",
                "second_autocorr_ineq",
                "heilbronn_triangle",
                "circle_packing",
            ],
            "ALE": ALE_TASKS,
        },
        "checkpoints": CHECKPOINTS,
        "ale_status": base.read(ALE / "status.json", {}).get(
            "status", "queued after all Math cells reach 50"
        ),
        "cost_policy": "Observed input + output API tokens through each checkpoint, including retries. Uniquely attributed semantic calls only. Ambiguous/missing usage yields a lower bound. Frozen corpus construction, embedding compute and private evaluator CPU are outside API token cost. No cross-task quality aggregation.",
    }


def render_report():
    data = build_data()
    base.write(OUT, data)
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    section = START + TEMPLATE.replace("__PAYLOAD__", payload) + END
    # Share the dashboard writer lock with other renderers.
    import fcntl

    with base.HTML.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = base.HTML.read_text()
        if START in text and END in text:
            text = (
                text[: text.index(START)] + section + text[text.index(END) + len(END) :]
            )
        else:
            text = text.replace("</body>", section + "\n</body>")
        temp = base.HTML.with_suffix(".quality.tmp")
        temp.write_text(text)
        temp.replace(base.HTML)
    return data


TEMPLATE = r"""
<section id="trajectory-quality-cost">
<style>
#trajectory-quality-cost{margin:32px;padding:24px;background:#f9fbfd;border:1px solid #bacbd5;border-top:4px solid #199375;border-radius:10px;font:14px/1.6 system-ui;color:#173146}
#trajectory-quality-cost h2{margin:0;font-size:24px}#trajectory-quality-cost select{font:inherit;padding:7px;background:white;border:1px solid #bacbd5;border-radius:5px}
#trajectory-quality-cost .qc-controls{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}#qc-panels{display:grid;grid-template-columns:1fr 1fr;gap:20px}#qc-panels svg{width:100%;height:auto}#qc-panels text{font:12px system-ui;fill:#456074}#qc-legend{display:flex;gap:16px;flex-wrap:wrap}#qc-table{overflow:auto}#qc-table table{width:100%;border-collapse:collapse;white-space:nowrap}#qc-table td,#qc-table th{padding:8px;border-bottom:1px solid #d9e3e9;text-align:right}#qc-table th{background:#edf3f7}#trajectory-quality-cost .qc-note{font-size:12px;color:#546c7e}
@media(max-width:850px){#qc-panels{grid-template-columns:1fr}#trajectory-quality-cost{margin:16px 0;padding:14px}}
</style>
<h2>Math &amp; ALE · Quality and token cost</h2>
<p>Checkpoint comparisons at 10 / 20 / 30 / 40 / 50 rounds. ALE begins after all four Math tasks and four plans finish 50 rounds. Each private evaluation measures the program selected using public scores at that checkpoint.</p>
<div class="qc-controls"><label>Family <select id="qc-family"><option>Math</option><option>ALE</option></select></label><label>Task <select id="qc-task"></select></label><label>Checkpoints <select id="qc-cp"><option value="all">All observed</option><option>10</option><option>20</option><option>30</option><option>40</option><option>50</option></select></label><label id="qc-metric-label" hidden>Private metric <select id="qc-metric"><option value="quality">AtCoder performance</option><option value="private_abs">Raw private score</option><option value="delta_performance">Performance Δ vs seed</option></select></label><label>Token cost <select id="qc-cost"><option value="total_tokens_observed">Generation + semantic</option><option value="generation_tokens">Generation only</option></select></label></div>
<div id="qc-legend"></div><div id="qc-panels"><div id="qc-curve"></div><div id="qc-scatter"></div></div>
<p id="qc-status" class="qc-note"></p><div id="qc-table"></div>
<p class="qc-note">Color = method; labels = checkpoint. Hollow scatter points indicate incomplete token attribution (observed lower bound). Private quality can decrease even when public fitness improves. Scores from different tasks use separate axes. SemRankOnly is score-only in this experiment.</p>
<details><summary>Measurement policy and data</summary><p id="qc-policy"></p><pre id="qc-data" style="max-height:300px;overflow:auto;font-size:11px"></pre></details>
<script>(function(){const D=__PAYLOAD__,F=document.getElementById('qc-family'),T=document.getElementById('qc-task'),C=document.getElementById('qc-cp'),K=document.getElementById('qc-cost'),M=document.getElementById('qc-metric'),colors={VGR:'#177db5',SemRankOnly:'#9754b8',P1:'#cc7920',P2:'#199375'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>x==null?'Pending':typeof x==='number'?x.toLocaleString('en-US',{maximumFractionDigits:5}):esc(x);
document.getElementById('qc-legend').innerHTML=Object.entries(colors).map(([p,c])=>`<span style="color:${c}">● ${p}</span>`).join('');
function chart(rows,scatter){const valid=rows.filter(p=>Number.isFinite(p.quality)&&(!scatter||Number.isFinite(p[K.value]))),ys=valid.map(p=>p.quality);let lo=ys.length?Math.min(...ys):0,hi=ys.length?Math.max(...ys):1;const span=hi-lo||1;lo-=span*.1;hi+=span*.1;const maxX=scatter?Math.max(1,...valid.map(p=>p[K.value]))*1.12:50;const X=x=>75+x/maxX*425,Y=y=>290-(y-lo)/(hi-lo)*225;
let s=`<svg viewBox="0 0 550 350" role="img" aria-label="${scatter?'Quality versus cumulative token cost':'Quality at iteration checkpoints'}"><text x="75" y="25">${scatter?'Quality / cumulative tokens':'Quality / iteration'}</text>`;
for(let j=0;j<=4;j++){let y=lo+(hi-lo)*j/4;s+=`<line x1="75" x2="500" y1="${Y(y)}" y2="${Y(y)}" stroke="#dbe5eb"/><text x="68" y="${Y(y)+4}" text-anchor="end">${Math.abs(y)>100?Math.round(y):y.toFixed(3)}</text>`;}for(let j=0;j<=(scatter?4:5);j++){let x=maxX*j/(scatter?4:5);s+=`<text x="${X(x)}" y="311" text-anchor="middle">${scatter?(x/1000).toFixed(0)+'k':x.toFixed(0)}</text>`;}
s+=`<path d="M75 55V290H500" fill="none" stroke="#8ca2b2"/><text x="285" y="342" text-anchor="middle">${scatter?'Cumulative API tokens (input + output)':'Completed iteration'}</text>`;
Object.entries(colors).forEach(([method,color])=>{const ps=valid.filter(p=>p.plan===method).sort((a,b)=>a.iteration-b.iteration);if(!scatter&&ps.length)s+=`<path d="${ps.map((p,i)=>(i?'L':'M')+X(p.iteration)+','+Y(p.quality)).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;ps.forEach(p=>{const x=X(scatter?p[K.value]:p.iteration),y=Y(p.quality);s+=`<circle data-plan="${method}" cx="${x}" cy="${y}" r="5" stroke="${color}" stroke-width="2" fill="${scatter&&!p.usage_complete?'white':color}"><title>${method} @${p.iteration}: quality ${p.quality}; generation ${p.generation_tokens}; semantic observed ${p.semantic_tokens_observed}; ${p.usage_complete?'complete':'partial'} usage</title></circle>`;if(scatter)s+=`<text x="${x+7}" y="${y-7}" style="fill:${color};font-size:10px">${p.iteration}</text>`;});});if(!valid.length)s+='<text x="285" y="175" text-anchor="middle">Pending measured results</text>';return s+'</svg>';}
function draw(){const rows=D.points.filter(p=>p.family===F.value&&p.task===T.value&&(C.value==='all'||p.iteration===Number(C.value))).map(p=>({...p,quality:F.value==='ALE'?p[M.value]:p.quality}));document.getElementById('qc-curve').innerHTML=chart(rows,false);document.getElementById('qc-scatter').innerHTML=chart(rows,true);document.getElementById('qc-status').textContent=(F.value==='ALE'?M.selectedOptions[0].text+' · '+D.ale_status:'Best valid Math fitness')+' · '+rows.length+' measured checkpoint points';
const heads=['Method','Iteration','Quality','Private Δ vs seed','Private raw score','Private rank','Judge','Generation tokens','Semantic tokens observed','Total observed','Usage'];document.getElementById('qc-table').innerHTML='<table><thead><tr>'+heads.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(p=>'<tr>'+[p.plan,p.iteration,p.quality,F.value==='ALE'?p.delta_performance:'—',F.value==='ALE'?p.private_abs:'—',F.value==='ALE'?p.rank:'—',p.judge||'—',p.generation_tokens,p.semantic_tokens_observed,p.total_tokens_observed,p.usage_complete?'Complete':'Partial / lower bound'].map(x=>'<td>'+fmt(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';document.getElementById('qc-data').textContent=JSON.stringify(rows,null,2);}
function family(){document.getElementById('qc-metric-label').hidden=F.value!=='ALE';T.innerHTML='';D.tasks[F.value].forEach(t=>T.add(new Option(t,t)));draw();}F.onchange=family;T.onchange=C.onchange=K.onchange=M.onchange=draw;document.getElementById('qc-policy').textContent=D.cost_policy;family();})();</script>
</section>
"""


if __name__ == "__main__":
    data = render_report()
    print(json.dumps({"points": len(data["points"]), "output": str(OUT)}))
