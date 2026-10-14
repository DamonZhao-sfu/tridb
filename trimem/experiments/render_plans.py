"""Render results/trimem_v0.1.0/fig1b_plans.json as a Fig. 1(b)-style plan comparison page."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def modality(n: dict) -> str:
    prov = n.get("Custom Plan Provider")
    if prov == "TriMEM SemFilter":
        return "sem"
    if prov == "TriMEM Expand":
        return "graph"
    if "hnsw" in str(n.get("Index Name", "")) or "<=>" in str(n.get("Order By", "")):
        return "vector"
    if n["Node Type"] in (
        "Seq Scan",
        "Index Scan",
        "Index Only Scan",
        "Bitmap Heap Scan",
        "Bitmap Index Scan",
        "Nested Loop",
        "Hash Join",
        "Merge Join",
        "Hash",
    ):
        return "rel"
    return "op"


def label(n: dict) -> tuple[str, str]:
    t = n["Node Type"]
    prov = n.get("Custom Plan Provider")
    if prov == "TriMEM SemFilter":
        return "SemFilter", n.get("Placement", "")
    if prov == "TriMEM Expand":
        return (
            "Expand",
            f"{n.get('Edge Type')} · {n.get('Degree Model', '').split(' (')[0]}",
        )
    if t == "Sort":
        return "Rank (Sort)", ", ".join(n.get("Sort Key", []))
    if t in ("Index Scan", "Index Only Scan"):
        rel = n.get("Alias") or n.get("Relation Name", "")
        if modality(n) == "vector":
            return "ANN top-k (HNSW)", f"{n.get('Index Name')} on {rel}"
        return "Index Scan", f"{n.get('Index Name')} on {rel}"
    if t == "Seq Scan":
        return "Seq Scan", n.get("Alias") or n.get("Relation Name", "")
    if t == "Subquery Scan":
        return "Subquery", n.get("Alias", "")
    return t, ""


def detail(n: dict) -> list[str]:
    out = []
    if n.get("Filter") and n.get("Custom Plan Provider") is None:
        out.append(
            "filter "
            + n["Filter"].replace("'::double precision", "").replace("::text", "")
        )
    if n.get("Index Cond"):
        out.append("cond " + n["Index Cond"])
    if n.get("Hash Cond"):
        out.append("hash " + n["Hash Cond"])
    if n.get("Custom Plan Provider") == "TriMEM SemFilter":
        out.append(
            f"judged {n.get('Rows Judged')} · LLM calls {n.get('LLM Calls')} · "
            f"cache hits {n.get('Cache Hits')} · batches {n.get('Batches')}"
        )
        out.append(
            f"LLM wait {n.get('LLM Wait', 0):.0f} ms · {n.get('Semantic Cost Model', '').split(' (')[0]}"
        )
    if n.get("Custom Plan Provider") == "TriMEM Expand":
        out.append(f"seeds {n.get('Seeds')} · edges {n.get('Edges Emitted')}")
    if n["Node Type"] == "Memoize":
        out.append(f"hits {n.get('Cache Hits')} · misses {n.get('Cache Misses')}")
    return out


def tree(n: dict) -> dict:
    name, sub = label(n)
    return {
        "name": name,
        "sub": sub,
        "mod": modality(n),
        "rows": n.get("Actual Rows"),
        "loops": n.get("Actual Loops"),
        "est": n.get("Plan Rows"),
        "cost": n.get("Total Cost"),
        "detail": detail(n),
        "kids": [
            tree(c)
            for c in n.get("Plans", [])
            if c.get("Subplan Name") is None
            or c.get("Parent Relationship") != "InitPlan"
        ],
    }


def main(src: str, template: str, out: str) -> None:
    d = json.loads(Path(src).read_text())
    plans = []
    for p in d["plans"]:
        plans.append(
            {
                k: p[k]
                for k in (
                    "id",
                    "name",
                    "role",
                    "settings",
                    "semantic",
                    "sql",
                    "planner_total_cost",
                    "execution_ms",
                    "planning_ms",
                    "warm_wall_s",
                    "llm_wait_ms",
                    "rows_judged",
                    "llm_calls",
                    "invalid_replies",
                    "prompt_tokens",
                    "completion_tokens",
                    "result",
                    "oracle_result",
                    "matches_oracle",
                    "verdict_accuracy",
                )
            }
            | {
                "tree": tree(p["explain"]["Plan"]),
                "decision": next(
                    (
                        n.get("Placement Decision")
                        for n in _walk(p["explain"]["Plan"])
                        if n.get("Placement Decision")
                    ),
                    None,
                ),
            }
        )
    data = {"model": d["model"], "predicate": d["predicate"], "plans": plans}
    html = Path(template).read_text().replace("/*__DATA__*/null", json.dumps(data))
    Path(out).write_text(html)


def _walk(p: dict):
    yield p
    for c in p.get("Plans", []):
        yield from _walk(c)


if __name__ == "__main__":
    main(*sys.argv[1:4])
