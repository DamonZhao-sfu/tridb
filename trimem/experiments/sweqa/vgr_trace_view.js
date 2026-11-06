/* Visualize actual native return paths; this is separate from No Memory evaluation. */
(() => {
  const d = window.QA_VGR_TRACE, host = document.getElementById('vgr-trace');
  if (!d || !host) return;
  const escape = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  host.innerHTML = `<h2>VGR 检索路径 · 真实只读查询示例</h2>
  <p class="note">这是在 1,000 条 SFT trajectories 库上执行的独立查询，未注入本次 No Memory 答题。四个 seeds 均完成有界扩展后形成完整候选池。排序与 Top-8 在候选池形成后执行。</p>
  <p id="trace-question"></p><div class="toolbar"><b>Vector top-4</b> → <b>similar ≤ 2</b> → <b>has_step ≤ 2</b> → <b>next_step × 2</b> → <b>Relational Join / Filter</b> → <b>完整 candidate pool</b> → <b>Rank / Filter</b> → <b>Top-8</b></div>
  <div class="toolbar"><label>候选路径 <select id="trace-path"><option value="">完整候选池路径</option>${d.paths.map(p=>`<option value="${p.rank}">候选 ${p.rank} · Step ID ${p.returned_step}</option>`).join('')}</select></label><button id="trace-prev">上一步</button><button id="trace-next">下一步</button><label>已展示的跳数 <input id="trace-hop" type="range" min="0" max="5" value="5"></label><span id="trace-stage" class="small"></span></div>
  <div class="toolbar" id="trace-types">${[['vector','#087c69','Vector → seeds'],['similar','#d18126','similar · 跨 session'],['has_step','#87959b','has_step · session → step'],['next_step','#3677c9','next_step · 执行顺序']].map(([t,c,l])=>`<label style="color:${c}"><input type="checkbox" data-trace-type="${t}" checked> ${l}</label>`).join('')}</div>
  <p class="small">双边框：当前所选计划最终返回的 step。所有 seeds 已完成扩展，未入选的候选仍显示在图中。</p>
  <div style="overflow:auto;border:1px solid #d8dfd8;border-radius:6px"><svg id="trace-svg" viewBox="0 0 1160 660" style="display:block;width:100%;min-width:900px;background:#f7faf5" role="img" aria-label="Vector seeds and native graph return paths"></svg></div>
  <p id="trace-count" class="small"></p><div class="cols"><div><h3>路径 / 关系记录</h3><pre id="trace-record" class="detail"></pre></div><div><h3>返回的工具观察</h3><pre id="trace-evidence" class="detail">选择一条返回路径，查看真实 observation。</pre></div></div>
  <details><summary>实际执行 SQL 与 EXPLAIN ANALYZE</summary><pre id="trace-sql" class="detail"></pre><pre id="trace-plan" class="detail"></pre></details>
  <p><a href="qa_vgr_trace/trace.json">路径与原始记录 JSON</a> · <a href="qa_vgr_trace/query.sql">查询 SQL</a></p>`;
  const $ = id => document.getElementById(id), color = {vector:'#087c69',similar:'#d18126',has_step:'#87959b',next_step:'#3677c9'};
  $('trace-question').textContent = '检索输入：'+d.source.question;
  $('trace-sql').textContent = d.sql;
  $('trace-plan').textContent = JSON.stringify(d.plan,null,2);
  const all = new Map(d.nodes.map(n=>[n.id,n])), seeds = new Set(d.seeds.map(s=>s.id)), returned = new Set(d.vgr_selected);
  d.seeds.forEach(s=>{if(!all.has(s.id))all.set(s.id,{id:s.id,kind:'session',label:'Seed '+s.id,unused:true});});
  const sessionOrder=[];
  for(const seed of d.seeds){sessionOrder.push(seed.id);d.edges.filter(e=>e.src===seed.id&&e.type==='similar').forEach(e=>{if(!sessionOrder.includes(e.dst))sessionOrder.push(e.dst);});}
  const positions=new Map();
  sessionOrder.forEach((id,i)=>positions.set(id,{x:seeds.has(id)?260:440,y:90+i*120}));
  for(const n of all.values())if(n.kind==='step'){const ss=d.nodes.filter(m=>m.session===n.session);const minimum=Math.min(...ss.map(m=>m.ordinal));positions.set(n.id,{x:620+(n.ordinal-minimum)*135,y:positions.get(n.session).y});}
  let selected=null,hop=5;
  function render(){
    const path=selected?d.paths.find(p=>p.rank===selected):null;
    const enabled=new Set([...$('trace-types').querySelectorAll('input:checked')].map(x=>x.dataset.traceType));
    const activeEdges=new Set(path?path.edges.slice(0,Math.max(0,hop-1)):d.edges.map(e=>e.key));
    const activeNodes=new Set(path?path.nodes.slice(0,hop):[...all.keys()]);
    const svg=$('trace-svg');
    let html='<defs>'+Object.entries(color).map(([t,c])=>`<marker id="trace-arrow-${t}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="${c}"/></marker>`).join('')+'</defs>';
    html+='<text x="65" y="28" font-size="14" fill="#173a3a">Question</text><text x="225" y="28" font-size="14" fill="#173a3a">Vector seeds</text><text x="420" y="28" font-size="14" fill="#173a3a">Graph expansion → tool steps</text>';
    for(const id of sessionOrder){let pos=positions.get(id);html+=`<path d="M180 ${pos.y+65}H1125" stroke="#e0e8de"/><text x="1100" y="${pos.y+55}" text-anchor="end" font-size="11" fill="#627574">Session ${id}</text>`;}
    if(enabled.has('vector'))for(const seed of d.seeds){let pos=positions.get(seed.id),active=!path||path.nodes[0]===seed.id&&hop>0;html+=`<path data-trace-edge="vector" d="M115 325C165 325,165 ${pos.y},185 ${pos.y}" stroke="${color.vector}" stroke-dasharray="5 4" fill="none" stroke-width="2" opacity="${all.get(seed.id).unused?.2:active?.8:.1}" marker-end="url(#trace-arrow-vector)"/>`;}
    for(const edge of d.edges){if(!enabled.has(edge.type))continue;let a=positions.get(edge.src),b=positions.get(edge.dst),active=activeEdges.has(edge.key),ax=a.x+(all.get(edge.src).kind==='session'?75:22),bx=b.x-(all.get(edge.dst).kind==='session'?75:22);let curve=edge.type==='has_step'&&b.x>620?-45:0;html+=`<path data-trace-edge="${edge.type}" data-trace-key="${edge.key}" d="M${ax} ${a.y}C${ax+65} ${a.y+curve},${bx-65} ${b.y+curve},${bx} ${b.y}" stroke="${color[edge.type]}" fill="none" stroke-width="${active&&path?3:2}" opacity="${active?.9:.1}" marker-end="url(#trace-arrow-${edge.type})"><title>${edge.type}: ${edge.src} → ${edge.dst}</title></path>`;}
    html+='<rect x="15" y="295" width="100" height="60" rx="7" fill="#173a3a"/><text x="65" y="321" text-anchor="middle" fill="white" font-size="12">Question</text><text x="65" y="340" text-anchor="middle" fill="white" font-size="11">Session 100000</text>';
    for(const n of all.values()){let pos=positions.get(n.id),active=activeNodes.has(n.id),fill=n.unused?'#d8dfd8':n.kind==='step'?'#3677c9':seeds.has(n.id)?'#087c69':'#d18126';let shape=n.kind==='session'?`<rect x="${pos.x-75}" y="${pos.y-22}" width="150" height="44" rx="7" fill="${fill}"/><text x="${pos.x}" y="${pos.y+4}" text-anchor="middle" fill="white" font-size="12">${escape(n.label)}</text>`:`<circle cx="${pos.x}" cy="${pos.y}" r="22" fill="${fill}"/><text x="${pos.x}" y="${pos.y+4}" text-anchor="middle" fill="white" font-size="12">${n.ordinal+1}</text><text x="${pos.x}" y="${pos.y+43}" text-anchor="middle" fill="#627574" font-size="10">${n.id}</text>`;
      if(returned.has(n.id))shape+=`<circle cx="${pos.x}" cy="${pos.y}" r="27" fill="none" stroke="#173a3a" stroke-width="2"/>`;
      if(seeds.has(n.id)){let seed=d.seeds.find(s=>s.id===n.id);shape+=`<text x="${pos.x}" y="${pos.y+43}" text-anchor="middle" fill="#627574" font-size="11">cosine ${seed.cosine.toFixed(4)}${n.unused?' · unused':''}</text>`;}
      html+=`<g data-trace-node="${n.id}" tabindex="0" role="button" aria-label="${escape(n.label)}" style="cursor:pointer" opacity="${active?1:.22}">${shape}</g>`;}
    svg.innerHTML=html;svg.querySelectorAll('[data-trace-node]').forEach(el=>{el.onclick=()=>{$('trace-record').textContent=JSON.stringify(all.get(Number(el.dataset.traceNode)),null,2);};el.onkeydown=e=>{if(e.key==='Enter')el.onclick();};});
    $('trace-hop').disabled=!path;$('trace-prev').disabled=!path||hop===0;$('trace-next').disabled=!path||hop>=path.nodes.length;
    $('trace-stage').textContent=path?`${hop} / ${path.nodes.length} 跳（含 Vector → seed）`:'全部返回路径';
    $('trace-count').textContent=`完整池 ${d.candidate_count} 条候选 · ${d.seeds.length} 个 seeds 全部扩展 · ${d.native_edges_verified} 条 native 边已核验 · 数据库阶段 ${d.plan[0]['Execution Time'].toFixed(3)} ms（不含 SemRank / SemFilter，不是 benchmark 均值）`;
    if(path){let row=d.rows[path.rank-1];$('trace-record').textContent=JSON.stringify({path:path.nodes,edge_types:path.edges.map(k=>d.edges.find(e=>e.key===k).type),returned_step:row.returned_step,task:row.task_id,session:row.session_id,repo:row.repo_label,tool:row.tool_name,arguments:row.arguments,relational_filter:"final.kind = 'tool' AND length(observation) > 0"},null,2);$('trace-evidence').textContent=row.observation;}
    else{$('trace-record').textContent='选择返回路径，查看跨 session 与 session 内的逐跳连接。\n\nRelational joins: step → session → task, step → tool_call, step → tool_observation。\nFilter: tool step 且 observation 非空。\n下方展示对完整候选池实际执行 SemRank / SemFilter 后的三个上下文。不另行生成摘要。';$('trace-evidence').textContent='尚未选择返回路径。';}
  }
  $('trace-path').onchange=e=>{selected=e.target.value?Number(e.target.value):null;hop=selected?d.paths[selected-1].nodes.length:5;$('trace-hop').max=hop;$('trace-hop').value=hop;render();};
  $('trace-hop').oninput=e=>{hop=Number(e.target.value);render();};$('trace-prev').onclick=()=>{hop--; $('trace-hop').value=hop;render();};$('trace-next').onclick=()=>{hop++;$('trace-hop').value=hop;render();};$('trace-types').onchange=render;render();window.qaTraceReady=true; window.qaTraceSelectPlan=ids=>{returned.clear();ids.forEach(id=>returned.add(id));render();};
})();
