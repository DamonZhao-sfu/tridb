"""Render a Figure 1 experiment directory (arm JSONs + comparison.json) as Markdown tables.

    python3 -m bench.agent_memory.fig1.report results/fig1_semantic_v0.1.0/real_slice
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ARM_ORDER = ("eager", "eager_ctl", "onread", "onread_stale", "batch")
PLAN_ORDER = ("trimodal", "naive", "planned_mat", "planned")


def _load(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = {
        a: json.loads((directory / f"{a}.json").read_text())
        for a in ARM_ORDER
        if (directory / f"{a}.json").exists()
    }
    comp_path = directory / "comparison.json"
    comp = json.loads(comp_path.read_text()) if comp_path.exists() else {}
    return arms, comp


def _ms(x: float) -> str:
    return f"{x * 1e3:.1f}"


def render(directory: Path) -> str:
    arms, comp = _load(directory)
    out: list[str] = [f"# Figure 1 experiment: `{directory}`", ""]
    if comp:
        out.append(f"LLM `{comp.get('llm')}`, {comp.get('events')} events, tag `{comp.get('tag')}`.")
        out.append("")

    out += [
        "## Ingest (per arm)",
        "",
        "| arm | mode | ingest p50 ms | p95 ms | mean ms | LLM calls on ingest path |"
        " pending at end | final materialize calls | total LLM calls | tokens (p+c) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, r in arms.items():
        ip = r["ingest_plan"]
        mode = ip["mode"] if ip["mode"] == "eager" else f"deferred/{ip['trigger']}"
        s = r["ingest_seconds"]
        tot = r["ingest_llm"]
        out.append(
            f"| {a} | {mode} | {_ms(s['p50'])} | {_ms(s['p95'])} | {_ms(s['mean'])} |"
            f" {r['ingest_llm_during_replay']['llm_calls']} | {r['pending_left_before_final']} |"
            f" {r['final_materialize']['llm']['llm_calls']} | {tot['llm_calls']} |"
            f" {tot['prompt_tokens'] + tot['completion_tokens']} |"
        )
    out.append("")

    out += [
        "## Retrieval plans",
        "",
        "| arm | plan | consistency | p50 ms | p95 ms | n_sem / retrieval | LLM calls |"
        " rows pulled / retrieval | early close rate | hints / retrieval | resolved on read |"
        " stale items seen |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, r in arms.items():
        for p in PLAN_ORDER:
            if p not in r["retrieve"]:
                continue
            rp = r["retrieve"][p]
            n = max(1.0, rp["totals"].get("retrievals", 0.0))
            out.append(
                f"| {a} | {p} | {r['consistency']} | {_ms(rp['seconds']['p50'])} |"
                f" {_ms(rp['seconds']['p95'])} | {rp['llm']['n_sem'] / n:.2f} |"
                f" {rp['llm']['llm_calls']} | {rp['totals'].get('rows_pulled', 0) / n:.1f} |"
                f" {rp['totals'].get('closed_early', 0) / n:.2f} |"
                f" {rp['totals'].get('hints', 0) / n:.2f} |"
                f" {int(rp['totals'].get('resolved', 0))} |"
                f" {int(rp['totals'].get('staleness', 0))} |"
            )
    out.append("")

    if comp.get("mat_equal_to_eager"):
        out += [
            "## Mat(S) equivalence vs eager (after final materialization)",
            "",
            "| arm | lessons equal | lessons agree | fixed_by equal | fixed_by Jaccard |",
            "|---|---|---:|---|---:|",
        ]
        for a, v in comp["mat_equal_to_eager"].items():
            out.append(
                f"| {a} | {v['lessons_equal']} | {v.get('lessons_agree', 1.0):.4f} |"
                f" {v['fixed_by_equal']} | {v['fixed_by_jaccard']:.4f} |"
            )
        out.append("")
    if comp.get("output_agreement_vs_eager_planned_mat"):
        out += [
            "## Retrieved-set agreement vs eager/planned_mat (per retrieval)",
            "",
            "| arm/plan | mean Jaccard | exact-set rate |",
            "|---|---:|---:|",
        ]
        for k, v in comp["output_agreement_vs_eager_planned_mat"].items():
            out.append(f"| {k} | {v['mean_jaccard']:.4f} | {v['exact_rate']:.4f} |")
        out.append("")
    if comp.get("offline_replay"):
        v = comp["offline_replay"]
        out.append(
            f"Offline replay of the eager arm from the cache with the LLM disabled: lessons "
            f"equal **{v['lessons_equal']}**, fixed_by equal **{v['fixed_by_equal']}**, "
            f"LLM calls **{v['llm_calls']}**."
        )
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    for d in args:
        text = render(Path(d))
        (Path(d) / "REPORT.md").write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
