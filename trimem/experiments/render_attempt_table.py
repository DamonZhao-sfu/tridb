"""Inject the heilbronn_triangle attempt table (the optimizer's frozen graph) into the English it20 summary page.

One row per attempt of the frozen snapshot: id, parent, session, fitness, gain over parent, validity, the change
summary used by SemFilter (current, original corpus text, and the regenerated detailed one when available) and the
program code (truncated for page size). The section sits between `<!-- attempts:start -->` and `<!-- attempts:end -->`
after the query-graph explorer; re-running replaces it.

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.render_attempt_table
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

PAGE = Path("triMEM_optimizer/it20_summary/trimem_it20_summary_en.html")
SNAPSHOT = Path("triMEM_optimizer/trajectory_optimizer_design/triangle_space_v5/inputs/math_heilbronn_triangle_frozen_v2.json")
SUMMARIES_V3 = Path("triMEM_optimizer/trajectory_optimizer_design/summaries_v3/summaries_v3.jsonl")
DSN = f"postgresql://{os.environ.get('USER', '')}@127.0.0.1:55441/trimem_evo"
START, END = "<!-- attempts:start -->", "<!-- attempts:end -->"
ANCHOR = "<!-- query-graph-explorer:end -->"
CODE_CHARS = 1500

SECTION = """<!-- attempts:start -->
<style>
.atab td{vertical-align:top}.atab td.txt{min-width:260px;max-width:420px;font-size:12px;line-height:1.45}
.atab pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto;background:var(--bg);padding:8px;font:11px/1.45 var(--f-mono);border-radius:5px;margin:4px 0 0}
.atab summary{cursor:pointer;color:var(--muted);font-size:11.5px}.anav{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:10px 0}
.anav input,.anav select,.anav button{font:inherit;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 9px}
.anav input{min-width:0;flex:1 1 220px}.atab th{cursor:pointer;white-space:nowrap}.atab .neg{color:var(--muted)}
</style>
<section class="panel" id="attempt-table">
<h2>Attempt Table: heilbronn_triangle Experience Graph</h2>
<p class="lede">The relational records behind the optimizer's frozen graph (snapshot __SNAP__). <b>changes</b> is the
summary SemFilter reads; <b>original</b> is the corpus text before LLM derivation (often empty or a stub);
<b>detailed</b> is the regenerated &le;1024-character summary (__V3NOTE__). Retrieval injects
<code>fitness + changes + code</code> into the prompt; code here is truncated to __CODE__ characters. Click a column to sort.</p>
<div class="anav"><input id="aq" placeholder="search id, session, summary, code…" aria-label="search attempts">
<select id="af" aria-label="filter"><option value="">all attempts</option><option value="valid">valid only</option>
<option value="improving">improving edits (seed-eligible)</option><option value="root">roots</option></select>
<button id="aprev">‹ prev</button><span id="ainfo" class="mono"></span><button id="anext">next ›</button></div>
<div class="tablewrap"><table class="atab" id="atable"></table></div>
</section>
<script>
(() => {
const A = __DATA__;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const f4 = x => x == null ? "–" : Number(x).toFixed(4);
const cols = [["id", "id"], ["parent_id", "parent"], ["session", "session"], ["fitness", "fitness"], ["gain", "gain"],
  ["is_valid", "valid"], ["changes", "changes (SemFilter input)"], ["changes_raw", "original"], ["detailed", "detailed (v3)"], ["code_chars", "code"]];
let key = "id", dir = 1, page = 0; const SIZE = 40;
function rows() {
  const q = document.getElementById("aq").value.toLowerCase().trim(), flt = document.getElementById("af").value;
  return A.filter(r => (!flt || (flt === "valid" ? r.is_valid : flt === "root" ? r.parent_id == null : r.is_valid && r.gain > 1e-9))
    && (!q || [r.id, r.session, r.changes, r.changes_raw, r.detailed, r.code].some(v => String(v ?? "").toLowerCase().includes(q))))
    .sort((a, b) => { const x = a[key], y = b[key]; return (x == null) - (y == null) || (x < y ? -dir : x > y ? dir : 0); });
}
function draw() {
  const rs = rows(), max = Math.max(0, Math.ceil(rs.length / SIZE) - 1); page = Math.min(page, max);
  document.getElementById("ainfo").textContent = `${rs.length.toLocaleString("en-US")} rows · page ${page + 1}/${max + 1}`;
  const txt = s => s ? (s.length > 260 ? `${esc(s.slice(0, 260))}… <details><summary>full (${s.length} chars)</summary>${esc(s)}</details>` : esc(s)) : '<span class="neg">–</span>';
  document.getElementById("atable").innerHTML = `<thead><tr>${cols.map(([k, l]) => `<th data-k="${k}">${l}${k === key ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead><tbody>` +
    rs.slice(page * SIZE, (page + 1) * SIZE).map(r => `<tr><td class="mono">${r.id}</td><td class="mono">${r.parent_id ?? "root"}</td>
      <td class="mono" style="max-width:180px;overflow-wrap:anywhere;font-size:11px">${esc(r.session)}${r.iteration != null ? ` #${r.iteration}` : ""}</td>
      <td>${f4(r.fitness)}</td><td class="${r.gain > 1e-9 ? "win" : "neg"}">${r.gain == null ? "–" : (r.gain > 0 ? "+" : "") + Number(r.gain).toPrecision(3)}</td>
      <td>${r.is_valid ? "✓" : "✗"}</td><td class="txt">${txt(r.changes)}</td><td class="txt">${txt(r.changes_raw)}</td><td class="txt">${txt(r.detailed)}</td>
      <td class="txt"><details><summary>${r.code_chars.toLocaleString("en-US")} chars${r.code_chars > __CODE__ ? " (truncated)" : ""}</summary><pre>${esc(r.code)}</pre></details></td></tr>`).join("") + "</tbody>";
  document.querySelectorAll("#atable th").forEach(th => th.onclick = () => { const k = th.dataset.k; dir = k === key ? -dir : 1; key = k; draw(); });
}
document.getElementById("aq").oninput = () => { page = 0; draw(); };
document.getElementById("af").onchange = () => { page = 0; draw(); };
document.getElementById("aprev").onclick = () => { page = Math.max(0, page - 1); draw(); };
document.getElementById("anext").onclick = () => { page += 1; draw(); };
draw();
})();
</script>
<!-- attempts:end -->"""


def _load_v3(path: Path) -> dict[int, str]:
    """attempt id -> regenerated summary (last row wins; rows without a summary are skipped)."""
    out: dict[int, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                text = row.get("new_summary") or row.get("summary") or row.get("new")
                if text and row.get("attempt_id", row.get("id")) is not None:
                    out[int(row.get("attempt_id", row.get("id")))] = text
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=str(PAGE))
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("--summaries", default=str(SUMMARIES_V3))
    args = ap.parse_args(argv)
    snap = json.loads(Path(args.snapshot).read_text())
    attempts = snap["attempts"]
    fit = {a["id"]: a["fitness"] for a in attempts}
    import psycopg  # read-only SELECT of session metadata that the snapshot does not carry

    with psycopg.connect(DSN) as conn:
        meta = {r[0]: r[1:] for r in conn.execute(
            "SELECT id, session, iteration, changes_raw FROM attempt WHERE id = ANY(%s)",
            ([a["id"] for a in attempts],)).fetchall()}
    v3 = _load_v3(Path(args.summaries))
    rows = []
    for a in attempts:
        session, iteration, raw = meta.get(a["id"], (None, None, None))
        pf = fit.get(a["parent_id"]) if a["parent_id"] is not None else None
        gain = a["fitness"] - pf if a["fitness"] is not None and pf is not None else None
        code = a.get("code") or ""
        rows.append({"id": a["id"], "parent_id": a["parent_id"], "session": (session or "").split(":", 1)[-1],
                     "iteration": iteration, "fitness": a["fitness"],
                     "gain": gain if gain is None or math.isfinite(gain) else None, "is_valid": bool(a["is_valid"]),
                     "changes": a.get("changes"), "changes_raw": raw or None, "detailed": v3.get(a["id"]),
                     "code": code[:CODE_CHARS], "code_chars": len(code)})
    v3note = f"{len(v3):,} of {len(rows):,} regenerated" if v3 else "regeneration in progress; column fills on re-render"
    block = (SECTION.replace("__DATA__", json.dumps(rows, separators=(",", ":")).replace("</", "<\\/"))
             .replace("__SNAP__", snap.get("provenance", {}).get("snapshot_id", Path(args.snapshot).name))
             .replace("__V3NOTE__", v3note).replace("__CODE__", str(CODE_CHARS)))
    page = Path(args.page)
    html = page.read_text()
    if START in html:
        html = html[: html.index(START)] + block + html[html.index(END) + len(END):]
    else:
        if ANCHOR not in html:
            raise SystemExit("anchor '<!-- query-graph-explorer:end -->' not found")
        html = html.replace(ANCHOR, ANCHOR + "\n" + block, 1)
    page.write_text(html)
    print(f"{page}: attempt table with {len(rows)} rows ({len(v3)} detailed summaries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
