"""Render results/trimem_v0.1.0/evotrace_plans.json as the plan-choice page."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from trimem.experiments.render_plans import tree

KEEP = (
    "candidate",
    "note",
    "planner_total_cost",
    "execution_ms",
    "planning_ms",
    "warm_wall_s",
    "llm_wait_ms",
    "rows_judged",
    "worker_requests",
    "cache_hits",
    "edges_traversed",
    "expand_calls",
    "llm_calls",
    "invalid_replies",
    "prompt_tokens",
    "completion_tokens",
    "result",
)


def slim(run: dict) -> dict:
    return {k: run[k] for k in KEEP} | {"tree": tree(run["explain"]["Plan"])}


def main(src: str, template: str, out: str) -> None:
    d = json.loads(Path(src).read_text())
    scen = []
    for s in d["scenarios"]:
        est = dict(re.findall(r"(s=\w+):(\d+)", s["auto_decision"] or ""))
        chosen = (s["auto_decision"] or "").split(" ")[0]
        scen.append(
            {k: s[k] for k in ("id", "title", "story", "predicate", "sql")}
            | {
                "decision": s["auto_decision"],
                "chosen": chosen,
                "estimates": {k: float(v) for k, v in est.items()},
                "baseline": slim(s["baseline"]),
                "runs": [slim(r) for r in s["runs"]],
            }
        )
    data = {"model": d["model"], "scenarios": scen}
    Path(out).write_text(
        Path(template).read_text().replace("/*__DATA__*/null", json.dumps(data))
    )


if __name__ == "__main__":
    main(*sys.argv[1:4])
