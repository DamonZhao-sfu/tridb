"""Render a replay run (analysis.json + plan trees) as the comparison page.

    PYTHONPATH=. python -m trimem.replay.render results/trimem_v0.1.0/replay/run1 \
        --dsn postgresql://$USER@127.0.0.1:55441/trimem_evo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import threading

import psycopg

from trimem.experiments.render_plans import tree
from trimem import DEFAULT_MODEL
from trimem.llm import OpenAIJudge
from trimem.replay.memory import SEM_PREDICATE, Embedder, retrieve_sql
from trimem.worker import Worker

TEMPLATE = Path(__file__).with_name("replay_template.html")


def plan_trees(
    dsn: str,
    run: Path,
    tag: str,
    tasks: list[str],
    modes=None,
    ref_arm: str = "A2-trimem",
) -> dict:
    """The iteration-1 retrieval of each task (parent = seed), re-planned after the run under the
    planner's choice and under the forced alternative. Verdicts come from the A2 cell's cache,
    so no new LLM calls are made; actual row counts are real."""
    emb = Embedder()
    out = {}
    with psycopg.connect(dsn, autocommit=True) as conn:
        for t in tasks:
            key = t.split(":")[1]
            cell = f"{tag}:{key}:{ref_arm}"
            row = conn.execute(
                "SELECT id, code, fitness FROM attempt WHERE cell = %s AND iteration = 0",
                (cell,),
            ).fetchone()
            if not row:
                continue
            q, _, _ = emb.embed(row[1])
            params = {
                "task": t,
                "cell": cell,
                "pid": row[0],
                "floor": 0.8 * row[2],
                "q": "[" + ",".join(f"{x:.6g}" for x in q) + "]",
            }
            for stmt in (
                "SET jit = off",
                "SET hnsw.iterative_scan = relaxed_order",
                "SET hnsw.ef_search = 100",
                f"SET trimem.prompt_version = 'replay-{cell}'",
                "SET trimem.timeout_ms = 600000",
            ):
                conn.execute(stmt)
            res = {}
            for mode in modes or (("tri-modal", False), ("auto", True), ("alt", True)):
                label, sem = mode[0], mode[1]
                force = mode[2] if len(mode) > 2 else None
                sql = retrieve_sql(sem)
                conn.execute("RESET trimem.force_placement")
                if force:
                    conn.execute(f"SET trimem.force_placement = '{force}'")
                if label == "alt":
                    chosen = res["auto"]["decision"].split(" ")[0]
                    alias, where = chosen.split("=")
                    conn.execute(
                        f"SET trimem.force_placement = '{alias}={'base' if where == 'top' else 'top'}'"
                    )
                ex = conn.execute(
                    "EXPLAIN (ANALYZE, FORMAT JSON) " + sql, params
                ).fetchone()[0][0]
                res[label] = {
                    "tree": tree(ex["Plan"]),
                    "ms": ex["Execution Time"],
                    "cost": ex["Plan"]["Total Cost"],
                    "decision": _find(ex["Plan"], "Placement Decision"),
                }
            out[t] = res
    return out


def _find(p: dict, key: str):
    if key in p:
        return p[key]
    for c in p.get("Plans", []):
        v = _find(c, key)
        if v:
            return v
    return None


PRINT_CSS = """@page { size: A4 landscape; margin: 10mm; }
@media print {
  :root { --bg: #ffffff; }
  body { font-size: 11px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .wrap { max-width: none; padding-block: 0; }
  .tablewrap, .svgwrap { overflow: visible; }
  .svgwrap { break-inside: avoid; page-break-inside: avoid; }
  .svgwrap svg { max-width: 100%; max-height: 165mm; width: auto; height: auto; display: block; margin: 0 auto; }
  table { font-size: 9.5px; } th, td { padding: 4px 6px; }
  .curves > div, tr { break-inside: avoid; } h2 { break-after: avoid; }
}
</style>"""


def export_pdf(html_path: Path, pdf_path: Path) -> None:
    """Print version (all plan tabs expanded, trees scaled to a page) rendered by headless Chrome."""
    import subprocess

    s = html_path.read_text()
    s = s.replace(
        "if (PT.length) showPlan(PT[0]);",
        (
            'document.getElementById("ptabs").innerHTML = "";\n'
            'document.getElementById("pplan").innerHTML = PT.map(t => { showPlan(t); '
            'return `<h3 style="margin:14px 0 4px">${esc(t)}</h3>` + '
            'document.getElementById("pplan").innerHTML; }).join("");'
        ),
    )
    s = s.replace("</style>", PRINT_CSS, 1)
    printable = html_path.with_name(html_path.stem + "_print.html")
    printable.write_text(s)
    subprocess.run(
        [
            "google-chrome",
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf_path}",
            f"file://{printable.resolve()}",
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )


def main_multi(argv: list[str] | None = None) -> int:
    """Replicates of an A1 vs A6 run: averaged analysis, plan trees, HTML page and PDF."""
    from trimem.replay.analyze import main_multi as analyze_multi

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="offline")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    analysis = analyze_multi(args.runs, str(out / "analysis.json"))
    first = json.loads((Path(args.runs[0]) / "summary.json").read_text())
    tasks = sorted({k.split("|")[0] for k in analysis["cells"]})
    worker = Worker(args.dsn, OpenAIJudge(model=DEFAULT_MODEL), take_ms=100)
    wt = threading.Thread(target=worker.run, daemon=True)
    wt.start()
    plans = plan_trees(
        args.dsn,
        Path(args.runs[0]),
        first["tag"],
        tasks,
        modes=(
            ("tri-modal", False),
            ("auto", True),
            ("score", "score"),
            ("semfirst", "score", "d=base"),
            ("travfirst", "score", "d=top"),
        ),
        ref_arm="A6-trimem-rank",
    )
    worker.stop()
    wt.join(timeout=10)
    data = {
        "model": first.get("model"),
        "iterations": first.get("iterations"),
        "tag": first["tag"],
        "analysis": analysis,
        "plans": plans,
        "predicate": SEM_PREDICATE,
        "lede": (
            f"Offline experiment: the EvoTrace experience graph (all corpus sessions of each task, "
            f"LLM change summaries completed offline) is the memory. Greedy agent replay on {len(tasks)} "
            f"math tasks, {first.get('iterations')} iterations, {analysis['replicates']} replicates per plan, "
            f"generator {first.get('model')}. Tri-modal retrieval (ANN seeds → lineage expand → telemetry "
            f"filter → rank by improvement) vs the same retrieval with the improvement weighted by a batched "
            f"LLM relevance score (TriMEM SemScore). Values are means over replicates; per-replicate best "
            f"scores in parentheses."
        ),
    }
    html = out / f"{args.title}.html"
    html.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data)))
    export_pdf(html, out / f"{args.title}.pdf")
    print(html, out / f"{args.title}.pdf")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    run = Path(args.run)
    analysis = json.loads((run / "analysis.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    tasks = sorted({k.split("|")[0] for k in analysis["cells"]})
    # verdicts missing from a cell's cache (state moved on since iteration 1) need a worker
    worker = Worker(args.dsn, OpenAIJudge(model=DEFAULT_MODEL), take_ms=100)
    wt = threading.Thread(target=worker.run, daemon=True)
    wt.start()
    data = {
        "model": summary.get("model"),
        "iterations": summary.get("iterations"),
        "tag": summary["tag"],
        "analysis": analysis,
        "plans": plan_trees(args.dsn, run, summary["tag"], tasks),
        "predicate": SEM_PREDICATE,
    }
    worker.stop()
    wt.join(timeout=10)
    data["render_llm_calls"] = worker.stats.llm_calls
    out = Path(args.out or run / "replay.html")
    out.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data)))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
