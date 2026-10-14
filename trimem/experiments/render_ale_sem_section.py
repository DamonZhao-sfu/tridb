"""Inject the ALE seed-SemFilter section (private AtCoder performance @20) into the English summary page.

Self-contained (own data + script) between `<!-- alesem:start -->` and `<!-- alesem:end -->`, placed right
after the math msem section; re-running replaces it.

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.render_ale_sem_section
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

R = Path("results/trimem_v0.1.0/ale_sem")
PAGE = Path("triMEM_optimizer/it20_summary/trimem_it20_summary_en.html")
TASKS = ["ahc008", "ahc011", "ahc024", "ahc025", "ahc026"]
ARMS = ["vgr", "rank", "p1", "p2", "opt"]
PRIVATE = ["private_it20_merged.json", "private_it20_024_026.json"]
PRIVATE50 = ["private_it50_ahc008.json", "private_it50_011_025.json", "private_it50_024_026.json", "private_it50.json"]
START, END = "<!-- alesem:start -->", "<!-- alesem:end -->"
AFTER = "<!-- msem:end -->"
IT_RE = re.compile(r"Iteration (\d+): Program \S+ \(parent: \S+\) completed")
M_RE = re.compile(r"Metrics: .*?combined_score=([-\d.e]+).*?judge_result=(\w+)")
SEED_RE = re.compile(r"Island 0: 1 programs, best=([-\d.e]+)")


def public(cell: Path) -> dict:
    """Seed / best accepted public score and judge counts over iterations 1-20 (run.log)."""
    seed, scores, cur = None, {}, None
    for line in (cell / "run.log").read_text().splitlines():
        if " - INFO - " not in line:
            continue
        if seed is None and (m := SEED_RE.search(line)):
            seed = float(m.group(1))
        if m := IT_RE.search(line):
            cur = int(m.group(1))
            continue
        if cur is not None and (m := M_RE.search(line)):
            scores[cur] = (float(m.group(1)), m.group(2))
            cur = None
    acc = [s for s, j in scores.values() if j == "ACCEPTED"]
    return {"seed": seed, "best": max([seed] + acc) if seed is not None else (max(acc) if acc else None),
            "children": len(scores), "accepted": len(acc),
            "compile_errors": sum(j == "COMPILATION_ERROR" for _, j in scores.values())}


def collect() -> dict:
    priv = {}
    for f in PRIVATE:
        for r in json.loads((R / f).read_text()):
            priv[r["cell"]] = r
    priv50 = {}
    for f in PRIVATE50:
        if (R / f).exists():
            for r in json.loads((R / f).read_text()):
                priv50[r["cell"]] = r
    cells = []
    for t in TASKS:
        for a in ARMS:
            name = f"{t}__msem-{a}__s42"
            p = priv.get(name, {})
            cells.append({"task": t, "arm": a, **public(R / "it20" / name),
                          "priv_seed": p.get("private_seed_perf"), "priv_best": p.get("private_best_perf"),
                          "delta": p.get("delta_performance"), "rank_seed": p.get("private_seed_rank"),
                          "rank_best": p.get("private_best_rank"), "judge": p.get("judge_best"),
                          "pub50": priv50.get(name, {}).get("public_best"),
                          "priv50": priv50.get(name, {}).get("private_best_perf"),
                          "delta50": priv50.get(name, {}).get("delta_performance"),
                          "judge50": priv50.get(name, {}).get("judge_best")})
    return {"tasks": TASKS, "cells": cells}


SECTION = r"""<!-- alesem:start -->
  <section class="panel" id="alesem">
    <h2>ALE: Private AtCoder Performance at 20 and 50 Iterations (5 Tasks × 5 Plans)</h2>
    <p class="lede" style="margin-bottom:12px">The same five retrieval plans as the math section (VGR / SemRank / P1 / P2 / Opt) on ALE-Bench Lite ahc008, ahc011, ahc024, ahc025, ahc026. C++, 50 public cases during search, k = 10 change summaries, lineage + similar edges, hops = mean lineage depth (3–4). After 20 iterations, the seed and each run's public-best program are re-scored on the held-out private set (ALE-Bench <code>private_eval</code>); Δ = change in AtCoder rating-point performance (the EvoTrace Table 11 metric).</p>
    <div class="kv" id="alesemkpis" style="margin-bottom:16px"></div>
    <div class="treehead">Private Δ Performance @20 (rating points; seed → best public program)</div>
    <div class="tablewrap"><table id="alesemdelta"></table></div>
    <p class="note">Cell = Δ performance (private performance of the best program). Bold = best plan per task; red = private drop; TLE = the best public program exceeds the time limit on private cases. Bottom row = sum over the 5 tasks.</p>
    <div class="treehead" style="margin-top:16px">Private Δ Performance: @20 → @50</div>
    <div class="tablewrap"><table id="alesem50"></table></div>
    <p class="note">Each run continued from its iteration-20 checkpoint to iteration 50 (ahc008 / ahc011 / ahc025: iterations 21–50 without the simplification nudge). Cell = Δ@20 → Δ@50 (private performance @50); public best @50 below. “pending” = the run has not reached 50 (ahc024 / ahc026 paused). Sum row covers tasks finished at 50.</p>
    <div class="treehead" style="margin-top:16px">Public Search Detail @20</div>
    <div class="tablewrap"><table id="alesempub"></table></div>
    <p class="note">Public = mean score over 50 public cases (ahc025 is signed: higher is better). The seed's public score varies between runs when the program seeds its RNG from the clock (ahc024, ahc025). Accepted / compile errors count the 20 children of the run.</p>
    <div class="grid2" style="margin-top:16px">
      <div><div class="treehead">Sum of Private Δ over 5 Tasks</div><div class="bars" id="alesemsum"></div></div>
      <div><div class="treehead">Notes on this round</div><p class="note" style="margin-top:0">ahc008 / ahc011 / ahc025 iterations 1–20 ran with OpenEvolve's default “Consider simplifying – code length exceeds 500 characters” nudge; ahc024 / ahc026 ran without it. All runs used OpenEvolve's exact-line diff matching, which silently drops ~17% of SEARCH/REPLACE blocks on these seeds (trailing whitespace) and causes many compile errors; fixed for the next round. EvoTrace Table 11 (100 iterations, private Δ): ahc008 ≤ +38, ahc011 −3, ahc024 −1,610…+1,606, ahc025 ≤ 0, ahc026 0…+1,735.</p></div>
    </div>
  </section>
<script>
(() => {
const A = __DATA__;
const ARMS = ["vgr", "rank", "p1", "p2", "opt"], T = A.tasks;
const NAME = {vgr: "VGR", rank: "SemRank", p1: "P1 · SemFilter first", p2: "P2 · SemFilter after", opt: "Opt"};
const COL = {vgr: "--a0", rank: "--a3", p1: "--a2", p2: "--a4", opt: "--a5"};
const c = (t, a) => A.cells.find(x => x.task === t && x.arm === a);
const f = (x, d = 0) => x == null ? "–" : Number(x).toLocaleString("en-US", {maximumFractionDigits: d, minimumFractionDigits: d});
const sgn = x => x == null ? "–" : (x > 0 ? "+" : "") + f(x);
const sum = a => T.reduce((s, t) => s + (c(t, a).delta ?? 0), 0);
const sums = Object.fromEntries(ARMS.map(a => [a, sum(a)]));
const best = ARMS.reduce((b, a) => sums[a] > sums[b] ? a : b, ARMS[0]);
const neverNeg = ARMS.filter(a => T.every(t => (c(t, a).delta ?? 0) >= 0));
document.getElementById("alesemkpis").innerHTML = [
  [`${A.cells.filter(x => x.delta != null).length} / ${A.cells.length}`, "runs scored on the private set"],
  [`${NAME[best]} (${sgn(sums[best])})`, "largest summed private Δ"],
  [neverNeg.map(a => NAME[a]).join(", ") || "none", "plans never below their seed"],
  [`${Math.min(...A.cells.map(x => x.delta ?? 0))} … ${sgn(Math.max(...A.cells.map(x => x.delta ?? 0)))}`, "range of private Δ"],
].map(([b, s]) => `<div><b>${b}</b><span>${s}</span></div>`).join("");
let h = `<tr><th>Task</th><th>Seed perf</th>${ARMS.map(a => `<th><span class="dot" style="background:var(${COL[a]})"></span>${NAME[a]}</th>`).join("")}</tr>`;
for (const t of T) {
  const top = Math.max(...ARMS.map(a => c(t, a).delta ?? -1e9));
  h += `<tr><td class="l">${t}</td><td>${f(c(t, ARMS[0]).priv_seed)}</td>` + ARMS.map(a => { const x = c(t, a);
    const cls = x.delta === top && top > 0 ? ' class="win"' : x.delta < 0 ? ' style="color:var(--a16)"' : "";
    const tle = x.judge && x.judge !== "ACCEPTED" ? ` <span class="mono">${x.judge === "TIME_LIMIT_EXCEEDED" ? "TLE" : x.judge}</span>` : "";
    return `<td${cls}>${sgn(x.delta)} <span class="mono">(${f(x.priv_best)})</span>${tle}</td>`; }).join("") + "</tr>";
}
h += `<tr><td class="l"><b>Sum</b></td><td></td>${ARMS.map(a => `<td${a === best ? ' class="win"' : ""}><b>${sgn(sums[a])}</b></td>`).join("")}</tr>`;
document.getElementById("alesemdelta").innerHTML = h;
const done50 = T.filter(t => ARMS.every(a => c(t, a).delta50 != null));
let h5 = `<tr><th>Task</th>${ARMS.map(a => `<th><span class="dot" style="background:var(${COL[a]})"></span>${NAME[a]}</th>`).join("")}</tr>`;
for (const t of T) {
  const top = Math.max(...ARMS.map(a => c(t, a).delta50 ?? -1e9));
  h5 += `<tr><td class="l">${t}</td>` + ARMS.map(a => { const x = c(t, a);
    if (x.delta50 == null) return `<td class="mono">pending</td>`;
    const cls = x.delta50 === top && top > 0 ? ' class="win"' : x.delta50 < 0 ? ' style="color:var(--a16)"' : "";
    const tle = x.judge50 && x.judge50 !== "ACCEPTED" ? ` <span class="mono">${x.judge50 === "TIME_LIMIT_EXCEEDED" ? "TLE" : x.judge50}</span>` : "";
    return `<td${cls}>${sgn(x.delta)} → <b>${sgn(x.delta50)}</b> <span class="mono">(${f(x.priv50)}; pub ${f(x.pub50, 1)})</span>${tle}</td>`; }).join("") + "</tr>";
}
h5 += `<tr><td class="l"><b>Sum (${done50.length} tasks @50)</b></td>${ARMS.map(a => { const s20 = done50.reduce((s, t) => s + (c(t, a).delta ?? 0), 0), s50 = done50.reduce((s, t) => s + (c(t, a).delta50 ?? 0), 0); return `<td><b>${sgn(s20)} → ${sgn(s50)}</b></td>`; }).join("")}</tr>`;
document.getElementById("alesem50").innerHTML = h5;
let p = `<tr><th>Task</th><th>Plan</th><th>Seed (public)</th><th>Best (public)</th><th>Gain</th><th>Accepted</th><th>Compile errors</th><th>Private rank</th></tr>`;
for (const t of T) ARMS.forEach((a, j) => { const x = c(t, a);
  const g = x.seed != null && x.best != null ? (x.best - x.seed) / Math.abs(x.seed) * 100 : null;
  p += `<tr><td class="l">${j ? "" : t}</td><td class="l"><span class="dot" style="background:var(${COL[a]})"></span>${NAME[a]}</td><td>${f(x.seed, 1)}</td><td>${f(x.best, 1)}</td><td>${g == null ? "–" : (g >= 0 ? "+" : "") + f(g, 2) + "%"}</td><td>${x.accepted}/${x.children}</td><td>${x.compile_errors}</td><td>${f(x.rank_seed)} → ${f(x.rank_best)}</td></tr>`; });
document.getElementById("alesempub").innerHTML = p;
const m = Math.max(...ARMS.map(a => Math.abs(sums[a]))) || 1;
document.getElementById("alesemsum").innerHTML = ARMS.map(a => `<div class="brow"><span>${NAME[a]}</span><div class="stack-bar" style="position:relative"><span style="position:absolute;left:50%;width:${50 * Math.abs(sums[a]) / m}%;${sums[a] < 0 ? "transform:translateX(-100%);" : ""}background:var(${COL[a]})"></span></div><span class="mono">${sgn(sums[a])}</span></div>`).join("");
})();
</script>
<!-- alesem:end -->
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=str(PAGE))
    args = ap.parse_args(argv)
    data = collect()
    page = Path(args.page)
    html = page.read_text()
    block = SECTION.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    if START in html:
        html = html[: html.index(START)] + block + html[html.index(END) + len(END) + 1:]
    elif AFTER in html:
        i = html.index(AFTER) + len(AFTER) + 1
        html = html[:i] + block + html[i:]
    else:
        raise SystemExit("msem section not found; render it first (render_msem_section)")
    page.write_text(html)
    print(f"{page}: ALE section with {sum(c['delta'] is not None for c in data['cells'])} scored runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
