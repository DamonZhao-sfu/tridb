"""Greedy agent vs OpenEvolve, each with TriMEM retrieval plans: one page + PDF.

    PYTHONPATH=. python -m trimem.experiments.render_compare \
        --greedy results/.../offline/greedy_it20_rep1 results/.../offline/greedy_it20_rep2 \
        --oe results/.../openevolve/it20 --dsn ... --out-dir results/.../compare --title compare_it20

Every comparison is taken at the same iteration (checkpoints 10 and 20); cells that did not
reach a checkpoint show "–" rather than a value from a different iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trimem.experiments.oe_analyze import main as oe_analyze
from trimem.replay.analyze import main_multi
from trimem.replay.memory import SEM_PREDICATE
from trimem.replay.render import TEMPLATE, export_pdf, plan_trees

NAMES = {
    "G-A1-trimodal": "Greedy · tri-modal", "G-A2-trimem": "Greedy · hard semantic filter",
    "G-A6-trimem-rank": "Greedy · rank, optimizer", "G-A7-rank-semantic-first": "Greedy · rank, semantic first",
    "G-A8-rank-traverse-first": "Greedy · rank, traverse first",
    "OE-nocontext": "OpenEvolve · no memory", "OE-none": "OpenEvolve · own population only", "OE-trimodal": "OpenEvolve · tri-modal",
    "OE-hard": "OpenEvolve · hard filter, optimizer", "OE-hard-semfirst": "OpenEvolve · hard filter, semantic first",
    "OE-hard-travfirst": "OpenEvolve · hard filter, traverse first", "OE-rank": "OpenEvolve · rank, optimizer",
    "OE-rank-semfirst": "OpenEvolve · rank, semantic first", "OE-rank-travfirst": "OpenEvolve · rank, traverse first",
}
COLORS = {
    "G-A1-trimodal": "--a1", "G-A2-trimem": "--a7", "G-A6-trimem-rank": "--a2", "G-A7-rank-semantic-first": "--a3",
    "G-A8-rank-traverse-first": "--a4", "OE-nocontext": "--muted", "OE-none": "--a0", "OE-trimodal": "--a10", "OE-hard": "--a9", "OE-hard-semfirst": "--a7", "OE-hard-travfirst": "--a3",
    "OE-rank": "--a5", "OE-rank-semfirst": "--a6", "OE-rank-travfirst": "--a8",
}


def arms_summary(cells: dict) -> dict:
    tasks = sorted({k.split("|")[0] for k in cells})
    arms = sorted({k.split("|")[1] for k in cells})
    out = {}
    for a in arms:
        cs = [cells[f"{t}|{a}"] for t in tasks if f"{t}|{a}" in cells]
        num = [k for k, v in cs[0].items() if isinstance(v, (int, float))]
        agg = {k: sum(c.get(k, 0) for c in cs) for k in num}
        agg["mean_gain"] = agg.get("gain", 0) / len(cs)
        agg["wins"] = 0
        out[a] = agg
    for t in tasks:
        here = [a for a in arms if f"{t}|{a}" in cells]
        out[max(here, key=lambda a: cells[f"{t}|{a}"]["best"])]["wins"] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", nargs="*", default=[])
    ap.add_argument("--oe", required=True)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="compare")
    ap.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = {}
    if args.greedy:
        g = main_multi(args.greedy, str(out / "greedy_analysis.json"))
        cells |= {f"{k.split('|')[0]}|G-{k.split('|')[1]}": v for k, v in g["cells"].items()}
    cells |= oe_analyze(args.oe, args.iterations)["cells"]
    analysis = {"cells": cells, "arms": arms_summary(cells), "identity": {}}
    (out / "analysis.json").write_text(json.dumps(analysis, indent=1))
    tasks = sorted({k.split("|")[0] for k in cells})
    tag = None
    if args.greedy:  # runs stopped early have no summary.json; the cell id carries the tag
        first = next(Path(args.greedy[0]).glob("*.jsonl"), None)
        if first is not None:
            tag = json.loads(first.open().readline())["cell"].split(":")[0]
    plans = plan_trees(args.dsn, Path(args.greedy[0]) if args.greedy else Path(args.oe), tag or "", tasks,
                       modes=(("tri-modal", False), ("auto", True), ("score", "score"),
                              ("semfirst", "score", "d=base"), ("travfirst", "score", "d=top")),
                       ref_arm="A6-trimem-rank") if tag else {}
    data = {
        "model": "Qwen/Qwen3.8-27B-FP8", "iterations": args.iterations, "analysis": analysis,
        "plans": plans, "predicate": SEM_PREDICATE, "names": NAMES, "colors": COLORS,
        "checkpoints": [10, 20],
        "lede": ("Offline experiment on the EvoTrace experience graph. Two agents retrieve from the same TriMEM "
                 "memory: a greedy hill-climbing agent (best program as parent, 5 retrieved programs in the prompt) "
                 "and OpenEvolve 0.3.2 (islands + MAP-Elites; 5 of its 10 inspiration slots replaced by TriMEM "
                 "results, injected every iteration). Plans: tri-modal (V+G+R), hard semantic filter, and "
                 "semantic-weighted rank with the optimizer's placement or forced semantic-first / traverse-first. "
                 "Generator Qwen3.8-27B (thinking off), 2 replicates per plan; values are means at the same "
                 "iteration. Greedy cells that were stopped early show '–' at checkpoints they did not reach."),
    }
    html = out / f"{args.title}.html"
    html.write_text(TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data)))
    export_pdf(html, out / f"{args.title}.pdf")
    print(html, out / f"{args.title}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
