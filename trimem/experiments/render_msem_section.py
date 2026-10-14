"""Inject the math seed-SemFilter section (msem arms, 20 iterations) into the English it20 summary page.

The section is self-contained (own data + script) and sits between `<!-- msem:start -->` and
`<!-- msem:end -->` before "Quality over Iterations"; re-running replaces it.

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.render_msem_section
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

R = Path("results/trimem_v0.1.0/math_sem/it20")
PAGE = Path("triMEM_optimizer/it20_summary/trimem_it20_summary_en.html")
TASKS = ["heilbronn_convex_13", "second_autocorr_ineq", "heilbronn_triangle"]
# Best valid fitness among the imported trajectory corpus (attempt.cell IS NULL), trimem_evo on 2026-10-13;
# drawn as a dashed reference line on each curve.
CORPUS_BEST = {'circle_packing': 0.9962039325182792, 'heilbronn_convex_13': 0.9089933515624634, 'second_autocorr_ineq': 1.0412008550157312, 'third_autocorr_ineq': 0.9968122752053968, 'heilbronn_triangle': 0.9837171488688466}
ARMS = ["vgr", "rank", "p1", "p2", "opt"]
START, END = "<!-- msem:start -->", "<!-- msem:end -->"
ANCHOR = '  <section class="panel">\n    <h2>Quality over Iterations</h2>'
IT_RE = re.compile(r"Iteration (\d+): Program \S+ \(parent: \S+\) completed")
M_RE = re.compile(r"Metrics: .*?combined_score=([-\d.e]+)(?:.*?eval_time=([\d.]+))?")
SEED_RE = re.compile(r"Island 0: 1 programs, best=([-\d.e]+)")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cell(task: str, arm: str, n: int = 20) -> dict:
    d = R / f"{task}__msem-{arm}__s42"
    seed, scores, evals, cur = None, {}, [], None
    for line in (d / "run.log").read_text().splitlines():
        if " - INFO - " not in line:
            continue
        if seed is None and (m := SEED_RE.search(line)):
            seed = float(m.group(1))
        if m := IT_RE.search(line):
            cur = int(m.group(1))
            continue
        if cur is not None and (m := M_RE.search(line)):
            scores[cur] = float(m.group(1))
            if m.group(2):
                evals.append(float(m.group(2)))
            cur = None
    best, curve = seed or 0.0, [seed or 0.0]
    for i in range(1, n + 1):
        best = max(best, scores.get(i, best))
        curve.append(best)
    ret = [r for r in _jsonl(d / "retrieval_telemetry.jsonl") if "observed" not in r]
    llm = _jsonl(d / "llm_telemetry.jsonl")
    ms = [r["total_ms"] for r in ret]
    choice = [(r.get("plan_choice") or {}).get("chosen") for r in ret]
    return {
        "task": task, "arm": arm, "seed": seed, "curve": curve, "children": len(scores),
        "valid": sum(v > 0 for v in scores.values()),
        "ret_n": len(ret), "ret_med": st.median(ms) if ms else None, "ret_mean": st.mean(ms) if ms else None,
        "ret_first": ms[0] if ms else None,
        "sem_rows": st.mean(r["sem_rows"] for r in ret) if ret else 0,
        "sem_tok": sum(r["sem_prompt_tokens"] + r["sem_completion_tokens"] for r in ret),
        "edges": st.mean(r["edges"] for r in ret) if ret else 0,
        "hops": (ret[0].get("plan_choice") or {}).get("hops") if ret else None,
        "p1": choice.count("p1"), "p2": choice.count("p2"),
        "pin_max": max((x.get("prompt_tokens") or 0) for x in llm) if llm else 0,
        "pin_mean": st.mean((x.get("prompt_tokens") or 0) for x in llm) if llm else 0,
        "out_max": max((x.get("completion_tokens") or 0) for x in llm) if llm else 0,
        "out_mean": st.mean((x.get("completion_tokens") or 0) for x in llm) if llm else 0,
        "trunc": sum(x.get("finish_reason") == "length" for x in llm), "llm_calls": len(llm),
        "gen_ms": st.mean((x.get("llm_api_ms") or 0) for x in llm) if llm else 0,
        "eval_s": st.mean(evals) if evals else None,
    }


SECTION = r"""<!-- msem:start -->
  <section class="panel" id="msem">
    <h2>Seed SemFilter + SemanticRank Plans (3 Math Tasks, 20 Iterations)</h2>
    <p class="lede" style="margin-bottom:12px">Five retrieval plans that differ only in their semantic operators. Shared settings: k = 10 memories in all 10 inspiration slots, injected as LLM change summaries; seeds = top-24 summary-bearing attempts by code embedding; every hop expands both <i>lineage</i> and <i>similar</i> edges (fanout 8); hops = the task's mean lineage depth (4 / 4 / 5); telemetry = valid, better than its parent, ≥ parent − 20%. Generator context 262,144, output cap 32,768, prompt budget 220,000.
      <b>VGR</b>: rank by gain only. <b>SemRank</b>: gain × (0.5 + sem_score). <b>P1</b>: SemFilter on the seeds <i>before</i> traversal, then SemRank. <b>P2</b>: the same SemFilter <i>after</i> traversal + telemetry (joined back on the seed id), then SemRank. P1 and P2 return identical rows. <b>Opt</b>: per retrieval, EXPLAIN P1 and P2 and run the cheaper one (plan enumeration only). The five arms of a task share one semantic cache.</p>
    <div class="kv" id="msemkpis" style="margin-bottom:16px"></div>
    <div class="treehead">Best-So-Far Score by Task (@i = Iteration i)</div>
    <div class="tablewrap"><table id="msemtable"></table></div>
    <p class="note">Bold = best arm per task at @20. Valid = children with combined_score &gt; 0. Retrieval = wall time of one retrieval (embedding + SQL incl. semantic waits). Rows judged = inputs to TriMEM semantic nodes per retrieval (cache hits included). Sem tokens = generator tokens spent on semantic calls by this cell (depends on which arm filled the shared cache first). Trunc = generations ending with finish_reason = length.</p>
    <div class="treehead" style="margin-top:16px">Per-Task Curves</div>
    <div class="legend" id="msemlegend" style="margin-bottom:8px"></div>
    <div class="curves" id="msemcurves"></div>
    <div class="grid2" style="margin-top:16px">
      <div><div class="treehead">Retrieval Latency, Median per Retrieval</div><div class="bars" id="msemlat"></div></div>
      <div><div class="treehead">Rows Judged by Semantic Operators per Retrieval</div><div class="bars" id="msemrows"></div></div>
    </div>
    <p class="note">Single seed (42), 20 iterations: quality differences between P1 / P2 / Opt come from different evolution paths (identical retrieval for the same parent), not from the plan. The plan affects cost: P2 judges 10-16× more rows than P1.</p>
  </section>
<script>
(() => {
const M = __DATA__;
const CORPUS_BEST = __CORPUS__;
const ARMS = ["vgr", "rank", "p1", "p2", "opt"], TASKS = M.tasks;
const NAME = {vgr: "VGR (no semantic op)", rank: "SemRank only", p1: "P1 · SemFilter before traversal", p2: "P2 · SemFilter after traversal", opt: "Opt · cheaper of P1/P2"};
const COL = {vgr: "--a0", rank: "--a3", p1: "--a2", p2: "--a4", opt: "--a5"};
const c = (t, a) => M.cells.find(x => x.task === t && x.arm === a);
const f = (x, d = 0) => x == null ? "–" : Number(x).toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d});
const sec = ms => ms == null ? "–" : ms >= 1000 ? f(ms / 1000, 1) + " s" : f(ms) + " ms";
const all = M.cells;
const opt = all.filter(x => x.arm === "opt"), p1 = all.filter(x => x.arm === "p1"), p2 = all.filter(x => x.arm === "p2");
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
document.getElementById("msemkpis").innerHTML = [
  [`${all.length}`, "cells (3 tasks × 5 arms)"],
  [`${all.reduce((s, x) => s + x.trunc, 0)} / ${all.reduce((s, x) => s + x.llm_calls, 0)}`, "truncated generations"],
  [`${f(Math.max(...all.map(x => x.pin_max)))}`, "max prompt tokens (budget 220,000)"],
  [`${f(Math.max(...all.map(x => x.out_max)))}`, "max output tokens (cap 32,768)"],
  [`P1 ×${opt.reduce((s, x) => s + x.p1, 0)} · P2 ×${opt.reduce((s, x) => s + x.p2, 0)}`, "opt plan choices"],
  [`${f(mean(p2.map(x => x.sem_rows)) / mean(p1.map(x => x.sem_rows)), 1)}×`, "rows judged, P2 / P1"],
].map(([b, s]) => `<div><b>${b}</b><span>${s}</span></div>`).join("");
const CK = [5, 10, 15, 20];
let h = `<tr><th>Task</th><th>Arm</th><th>Seed</th>${CK.map(i => `<th>@${i}</th>`).join("")}<th>Valid</th><th>Retrieval med / first</th><th>Rows judged</th><th>Sem tokens</th><th>Prompt max</th><th>Output max</th><th>Trunc</th></tr>`;
for (const t of TASKS) {
  const best = Math.max(...ARMS.map(a => c(t, a).curve[20]));
  ARMS.forEach((a, j) => { const x = c(t, a);
    h += `<tr><td class="l">${j ? "" : t}</td><td class="l"><span class="dot" style="background:var(${COL[a]})"></span>${NAME[a]}</td><td>${f(x.seed, 4)}</td>` +
      CK.map(i => `<td${i === 20 && x.curve[20] >= best - 1e-12 ? ' class="win"' : ""}>${f(x.curve[i], 4)}</td>`).join("") +
      `<td>${x.valid}/${x.children}</td><td>${sec(x.ret_med)} / ${sec(x.ret_first)}</td><td>${f(x.sem_rows)}</td><td>${f(x.sem_tok)}</td><td>${f(x.pin_max)}</td><td>${f(x.out_max)}</td><td>${x.trunc}</td></tr>`; });
}
document.getElementById("msemtable").innerHTML = h;
document.getElementById("msemlegend").innerHTML = ARMS.map(a => `<span><span class="sw" style="background:var(${COL[a]})"></span>${NAME[a]}</span>`).join("") +
  `<span><span class="sw" style="background:none;border-top:2px dashed var(--fg);height:0"></span>Best valid fitness in the trajectory corpus</span>`;
const W = 300, H = 170, L = 44, B = 22, Tp = 18, Rt = 8;
document.getElementById("msemcurves").innerHTML = TASKS.map(t => {
  const cb = CORPUS_BEST[t], ys = [...ARMS.flatMap(a => c(t, a).curve), ...(cb == null ? [] : [cb])], lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
  const X = i => L + (W - L - Rt) * i / 20, Y = v => H - B - (H - B - Tp) * (v - lo) / span;
  const grid = [0, 0.5, 1].map(q => { const v = lo + q * span; return `<line x1="${L}" x2="${W - Rt}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L - 4}" y="${Y(v) + 3}" text-anchor="end">${f(v, 3)}</text>`; }).join("");
  const xt = [0, 5, 10, 15, 20].map(i => `<text x="${X(i)}" y="${H - 6}" text-anchor="middle">${i}</text>`).join("");
  const lines = ARMS.map(a => `<polyline fill="none" stroke="var(${COL[a]})" stroke-width="${a === "opt" ? 2.4 : 1.6}" points="${c(t, a).curve.map((v, i) => `${X(i)},${Y(v)}`).join(" ")}"><title>${NAME[a]}: ${f(c(t, a).curve[20], 4)} @20</title></polyline>`).join("");
  const cbl = cb == null ? "" : `<line x1="${L}" x2="${W - Rt}" y1="${Y(cb)}" y2="${Y(cb)}" stroke="var(--fg)" stroke-width="1.2" stroke-dasharray="5 3"><title>Best in trajectory corpus: ${f(cb, 4)}</title></line>`;
  return `<div><div class="treehead">${t}</div><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="best-so-far curves for ${t}">${grid}${xt}${cbl}${lines}</svg></div>`;
}).join("");
const bars = (id, key, fmtv) => {
  const max = Math.max(...all.map(x => x[key] || 0)) || 1;
  document.getElementById(id).innerHTML = TASKS.map(t => `<div class="treehead" style="margin:4px 0 0">${t}</div>` + ARMS.map(a => { const v = c(t, a)[key] || 0;
    return `<div class="brow"><span>${NAME[a].split(" · ")[0]}</span><div class="stack-bar"><span style="width:${100 * v / max}%;background:var(${COL[a]})"></span></div><span class="mono">${fmtv(v)}</span></div>`; }).join("")).join("");
};
bars("msemlat", "ret_med", sec);
bars("msemrows", "sem_rows", v => f(v));
})();
</script>
<!-- msem:end -->
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=str(PAGE))
    args = ap.parse_args(argv)
    data = {"tasks": TASKS, "cells": [cell(t, a) for t in TASKS for a in ARMS]}
    page = Path(args.page)
    html = page.read_text()
    block = SECTION.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    block = block.replace("__CORPUS__", json.dumps(CORPUS_BEST))
    if START in html:
        html = html[: html.index(START)] + block + html[html.index(END) + len(END) + 1:]
    else:
        if ANCHOR not in html:
            raise SystemExit("anchor section 'Quality over Iterations' not found")
        html = html.replace(ANCHOR, block + "\n" + ANCHOR, 1)
    page.write_text(html)
    print(f"{page}: msem section with {len(data['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
