"""Render actual trajectory curves with a frozen historical reference and checkpoints."""

import json
from pathlib import Path

START = "<!-- trajectory-v4-math:start -->"
END = "<!-- trajectory-v4-math:end -->"
TASKS = [
    "heilbronn_convex_13",
    "second_autocorr_ineq",
    "heilbronn_triangle",
    "circle_packing",
]
PLANS = ["VGR", "SemRankOnly", "P1", "P2"]


def render(data, html_path, historical_best_path):
    html_path = Path(html_path)
    history_path = Path(historical_best_path)
    history = (
        json.loads(history_path.read_text()) if history_path.exists() else {"tasks": {}}
    )
    visible = list(data)
    for task in TASKS:
        for plan in PLANS:
            if not any(r["task"] == task and r["plan"] == plan for r in visible):
                visible.append(
                    {
                        "task": task,
                        "plan": plan,
                        "status": "pending",
                        "curve": [],
                        "attempted_rounds": 0,
                        "child_count": 0,
                    }
                )
    payload = json.dumps(
        {"cells": visible, "history": history, "tasks": TASKS}, ensure_ascii=False
    ).replace("<", "\\u003c")
    section = START + TEMPLATE.replace("__PAYLOAD__", payload) + END
    import fcntl

    with html_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = html_path.read_text()
        if START in text and END in text:
            text = text[: text.index(START)] + section + text[text.index(END) + len(END) :]
        else:
            text = text.replace("</body>", section + "\n</body>")
        temp = html_path.with_name(html_path.name + ".trajectory.tmp")
        temp.write_text(text)
        temp.replace(html_path)


TEMPLATE = r"""
<section id="trajectory-v4-math" aria-labelledby="tv4title">
<style>
#trajectory-v4-math{margin:32px;padding:24px;border:1px solid #bacbd5;border-top:4px solid #177db5;border-radius:10px;background:#fafcfe;color:#173146;font:14px/1.6 system-ui,sans-serif}
#trajectory-v4-math h2{font-size:24px;color:#173146;margin:0 0 12px}#trajectory-v4-math p{max-width:100ch;color:#456074}
#trajectory-v4-math .tv4controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:18px 0 12px}
#trajectory-v4-math select{font:inherit;padding:6px 10px;border:1px solid #96adbf;border-radius:5px;background:white;color:#173146}
#tv4legend{display:flex;gap:8px 16px;flex-wrap:wrap;margin:12px 0 8px}
#tv4legend button{background:transparent;border:1px solid #ccd7e0;border-radius:5px;padding:6px 10px;display:flex;align-items:center;gap:8px;font:inherit;color:#173146;cursor:pointer}
#tv4legend button[aria-pressed="false"]{opacity:.4}#tv4legend button:focus-visible{outline:3px solid #177db5;outline-offset:2px}
#tv4legend svg{width:42px;height:16px;display:inline-block;flex:none}
#tv4chart svg{display:block;width:100%;height:auto;max-height:430px}#tv4chart svg text{font:12px system-ui,sans-serif;fill:#456074}
#tv4chart .tv4baseline{stroke:#263744;stroke-width:2;stroke-dasharray:8 5}
#tv4table{overflow-x:auto;margin-top:16px}#tv4table table{min-width:840px;width:100%;border-collapse:collapse;font-size:13px}
#tv4table th,#tv4table td{padding:9px 12px;text-align:right;border-bottom:1px solid #d9e3e9;white-space:nowrap}#tv4table th:first-child,#tv4table td:first-child{text-align:left}
#tv4table th{background:#edf3f7;color:#173146}#tv4table .tv4selected{background:#e3f0f7;font-weight:650}
#trajectory-v4-math .tv4note{font-size:12px;color:#546c7e;margin:6px 0}#tv4status{min-height:24px}
#trajectory-v4-math details{margin-top:18px}#trajectory-v4-math summary{cursor:pointer}#tv4audit{max-height:420px;overflow:auto;white-space:pre-wrap;font:11px/1.5 monospace}
@media(max-width:640px){#trajectory-v4-math{margin:16px 0;padding:14px}#tv4legend{gap:6px}#tv4legend button{font-size:12px;padding:5px 7px}#trajectory-v4-math h2{font-size:21px}}
</style>
<h2 id="tv4title">Trajectory retrieval: four Math tasks</h2>
<p>Four retrieval plans share full-code memories, task-specific prompts and seed 42. Each task runs to 20 iterations first. Once all four tasks reach that checkpoint, each run continues from its saved state to 50, with checkpoints at 30, 40 and 50.</p>
<div class="tv4controls"><label>Task <select id="tv4task"></select></label><label>Show through iteration <select id="tv4cp"><option>10</option><option selected>20</option><option>30</option><option>40</option><option>50</option></select></label></div>
<div id="tv4legend" aria-label="Method legend and visibility controls"></div>
<p class="tv4note">Click a legend entry to isolate overlapping curves. Solid curves are observed best-so-far fitness. The dashed line is the highest valid score in the imported historical corpus, not a theoretical upper bound.</p>
<div id="tv4chart"></div><p id="tv4status" class="tv4note"></p>
<div id="tv4table"></div>
<p class="tv4note">A pending checkpoint has no completed observation yet. Curves stop at observed rounds and are never extended into the future. Persisted children can lag live progress until OpenEvolve flushes its trace. SemRankOnly currently scores candidates without reordering them.</p>
<details><summary>Observed token usage and latency</summary><div id="tv4cost" style="overflow:auto"></div><p class="tv4note">Costs cover all observed calls, not only the selected checkpoint. Retrieval includes embedding and semantic waiting. The first convex_13 / VGR run was serial; later runs are parallel, so latency differences also reflect contention. Missing usage is unavailable.</p></details>
<details><summary>Historical reference, per-round audits and source paths</summary><pre id="tv4audit"></pre></details>
<p class="tv4note">Prompt contract: up to 4 retrieved programs, 16,384 inspiration tokens, 32,768 full input tokens and a separate 32,768 output limit. Historical references remain fixed as new experiments progress.</p>
<script>(function(){
const payload=__PAYLOAD__,data=payload.cells,history=payload.history.tasks||{},task=document.getElementById('tv4task'),cp=document.getElementById('tv4cp');
const names={heilbronn_convex_13:'Convex 13',second_autocorr_ineq:'Second autocorrelation',heilbronn_triangle:'Heilbronn triangle',circle_packing:'Circle packing'};
const methods=[{id:'VGR',label:'VGR',color:'#177db5',shape:'circle'},{id:'SemRankOnly',label:'SemRankOnly (score-only)',color:'#9754b8',shape:'square'},{id:'P1',label:'P1: SemFilter before graph',color:'#cc7920',shape:'triangle'},{id:'P2',label:'P2: SemFilter after graph',color:'#199375',shape:'diamond'}];
const hidden=new Set(),ck=[10,20,30,40,50];payload.tasks.forEach(t=>task.add(new Option(names[t]||t,t)));
const fmt=(v,d=5)=>v===null||v===undefined?'Pending':typeof v==='number'?v.toLocaleString('en-US',{maximumFractionDigits:d}):v;
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function marker(m,x,y,title='',size=4){let a=`fill="white" stroke="${m.color}" stroke-width="2"`,child=title?'<title>'+escape(title)+'</title>':'';
 if(m.shape==='square')return `<rect x="${x-size}" y="${y-size}" width="${size*2}" height="${size*2}" ${a}>${child}</rect>`;
 if(m.shape==='triangle')return `<polygon points="${x},${y-size-1} ${x-size-1},${y+size} ${x+size+1},${y+size}" ${a}>${child}</polygon>`;
 if(m.shape==='diamond')return `<polygon points="${x},${y-size-1} ${x-size-1},${y} ${x},${y+size+1} ${x+size+1},${y}" ${a}>${child}</polygon>`;
 return `<circle cx="${x}" cy="${y}" r="${size}" ${a}>${child}</circle>`;}
function draw(){
 const n=Number(cp.value),rs=methods.map(m=>data.find(r=>r.task===task.value&&r.plan===m.id)),base=history[task.value];
 const legend=document.getElementById('tv4legend');legend.innerHTML='';
 [...methods,{id:'history',label:'Historical best: '+fmt(base?.fitness,6),color:'#263744'}].forEach(m=>{const b=document.createElement('button');b.type='button';b.setAttribute('aria-pressed',String(!hidden.has(m.id)));b.dataset.method=m.id;b.innerHTML=`<svg viewBox="0 0 42 16" aria-hidden="true"><line x1="0" x2="42" y1="8" y2="8" stroke="${m.color}" stroke-width="2" ${m.id==='history'?'stroke-dasharray="6 4"':''}/>${m.id==='history'?'':marker(m,21,8,'',3)}</svg><span>${escape(m.label)}</span>`;b.onclick=()=>{hidden.has(m.id)?hidden.delete(m.id):hidden.add(m.id);draw()};legend.appendChild(b)});
 const points=rs.flatMap(r=>(r?.curve||[]).filter(p=>p.iteration<=n&&Number.isFinite(p.quality)));
 let values=points.map(p=>p.quality);if(Number.isFinite(base?.fitness))values.push(base.fitness);
 let lo=values.length?Math.min(...values):0,hi=values.length?Math.max(...values):1;if(!points.length)lo=0;
 const span=hi-lo||1;hi+=span*.09;lo=Math.max(0,lo-span*.04);
 const X=i=>74+i/n*836,Y=v=>324-(v-lo)/(hi-lo)*272;
 let svg=`<svg viewBox="0 0 960 382" role="img" aria-label="${escape(names[task.value])} best-so-far fitness with historical reference"><title>${escape(names[task.value])} through iteration ${n}</title>`;
 for(let j=0;j<=4;j++){const v=lo+(hi-lo)*j/4;svg+=`<line x1="74" x2="910" y1="${Y(v)}" y2="${Y(v)}" stroke="#dbe5eb"/><text x="64" y="${Y(v)+4}" text-anchor="end">${v.toFixed(3)}</text>`;}
 for(let i=0;i<=n;i+=5)svg+=`<text x="${X(i)}" y="348" text-anchor="middle">${i}</text>`;
 svg+='<path d="M74 44V324H910" fill="none" stroke="#8ca2b2"/><text x="74" y="24">Best-so-far fitness</text><text x="492" y="376" text-anchor="middle">Iteration</text>';
 if(Number.isFinite(base?.fitness)&&!hidden.has('history'))svg+=`<line class="tv4baseline" x1="74" x2="910" y1="${Y(base.fitness)}" y2="${Y(base.fitness)}"><title>Historical best valid fitness: ${base.fitness}</title></line><text x="906" y="${Y(base.fitness)-9}" text-anchor="end">Historical best ${fmt(base.fitness,6)}</text>`;
 methods.forEach((m,k)=>{if(hidden.has(m.id))return;const p=(rs[k]?.curve||[]).filter(p=>p.iteration<=n&&Number.isFinite(p.quality));if(!p.length)return;
 let d=`M${X(p[0].iteration)},${Y(p[0].quality)}`;for(let j=1;j<p.length;j++)d+=`H${X(p[j].iteration)}V${Y(p[j].quality)}`;
 svg+=`<path data-plan="${m.id}" d="${d}" fill="none" stroke="${m.color}" stroke-width="2.3"/>`;
 p.forEach((q,j)=>{if(j===p.length-1||q.iteration%4===k)svg+=marker(m,X(q.iteration),Y(q.quality),`${m.id} @${q.iteration}: ${q.quality}`)});
 });
 if(!points.length)svg+='<text x="492" y="210" text-anchor="middle">No experiment observations yet. Historical reference shown.</text>';
 document.getElementById('tv4chart').innerHTML=svg+'</svg>';
 document.getElementById('tv4status').textContent=rs.map((r,i)=>`${methods[i].id}: ${r?.status||'pending'}, ${r?.attempted_rounds||0} dispatched rounds`).join(' · ');
 let th=['Method','Run status',...ck.map(c=>'@'+c),'Persisted children'];
 let table='<table><thead><tr>'+th.map((h,i)=>`<th class="${i>=2&&ck[i-2]===n?'tv4selected':''}">${h}</th>`).join('')+'</tr></thead><tbody>';
 rs.forEach((r,i)=>{r=r||{};table+=`<tr><td style="color:${methods[i].color}">${methods[i].id}</td><td>${escape(r.status||'pending')}</td>`+ck.map(c=>`<td class="${c===n?'tv4selected':''}">${fmt(r['quality'+c],6)}</td>`).join('')+`<td>${fmt(r.child_count||0,0)}</td></tr>`});document.getElementById('tv4table').innerHTML=table+'</tbody></table>';
 let cost='<table style="width:100%;border-collapse:collapse"><thead><tr>'+['Method','Generation input','Generation output','Retrieval seconds','Generation API seconds'].map(h=>'<th style="padding:8px">'+h+'</th>').join('')+'</tr></thead><tbody>';
 rs.forEach((r,i)=>{r=r||{};cost+='<tr>'+[methods[i].id,r.generation_prompt_tokens,r.generation_completion_tokens,r.retrieval_seconds_inclusive,r.generation_api_seconds].map(x=>'<td style="padding:8px;text-align:right">'+(x===null||x===undefined?'Unavailable':escape(fmt(x,2)))+'</td>').join('')+'</tr>'});document.getElementById('tv4cost').innerHTML=cost+'</tbody></table>';
 document.getElementById('tv4audit').textContent=JSON.stringify({historical_reference:base,reference_source:payload.history.source,cells:rs},null,2);
}task.onchange=draw;cp.onchange=draw;draw();})();</script>
</section>
"""
