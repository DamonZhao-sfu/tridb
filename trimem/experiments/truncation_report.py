"""Where prompts, programs and outputs get truncated, per cell (greedy replay and OpenEvolve).

Truncation points:
* greedy prompt: task specification cut at 6,000 chars; each retrieved program's code cut at
  2,500 chars (SQL ``left(d.code, 2500)``); corpus change descriptions are not cut.
* OpenEvolve prompt: injected programs are rendered whole (no cut); evaluator artifacts are cut at
  ``max_artifact_bytes``; a generated program longer than ``max_code_length`` (65,536 chars) is
  rejected before evaluation.
* any generation: output stopped at max_tokens (``finish_reason == "length"``); a prompt over the
  32,768-token window fails the request (the iteration is lost, not truncated).

    PYTHONPATH=. python -m trimem.experiments.truncation_report --greedy ... --oe ... --out report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import psycopg

GREEDY_CODE_CUT, GREEDY_SPEC_CUT, OE_MAX_CODE = 2500, 6000, 65536


def greedy(dirs: list[str], dsn: str) -> list[dict]:
    rows = []
    with psycopg.connect(dsn) as conn:
        spec_len = dict(conn.execute("SELECT task, length(specification) FROM task").fetchall())
        for d in dirs:
            for f in sorted(Path(d).glob("*.jsonl")):
                recs = [json.loads(x) for x in f.open() if x.strip()]
                if not recs:
                    continue
                ids = sorted({i for r in recs for i in r.get("retrieved_ids", [])})
                lens = dict(conn.execute("SELECT id, length(code) FROM attempt WHERE id = ANY(%s)", (ids,)).fetchall()) if ids else {}
                cut = [(r["iteration"], i, lens.get(i, 0)) for r in recs for i in r.get("retrieved_ids", [])
                       if lens.get(i, 0) > GREEDY_CODE_CUT]
                task = recs[0]["task"]
                rows.append({
                    "agent": "greedy", "cell": f.stem, "task": task, "arm": recs[0]["arm"],
                    "iterations": len(recs),
                    "retrieved_items": sum(len(r.get("retrieved_ids", [])) for r in recs),
                    "retrieved_code_truncated": len(cut),
                    "truncated_programs": sorted({i for _, i, _ in cut}),
                    "max_retrieved_code_chars": max(lens.values(), default=0),
                    "spec_truncated": (spec_len.get(task) or 0) > GREEDY_SPEC_CUT,
                    "spec_chars": spec_len.get(task),
                    "max_prompt_tokens": max((r["generation"]["prompt_tokens"] for r in recs), default=0),
                    "output_truncated": None,  # the greedy client did not record finish_reason
                })
    return rows


def openevolve(oe_dir: str) -> list[dict]:
    rows = []
    for d in sorted(p for p in Path(oe_dir).iterdir() if p.is_dir() and "__" in p.name):
        llm = [json.loads(x) for x in (d / "llm_telemetry.jsonl").open()] if (d / "llm_telemetry.jsonl").exists() else []
        ret = [json.loads(x) for x in (d / "retrieval_telemetry.jsonl").open()] if (d / "retrieval_telemetry.jsonl").exists() else []
        log = (d / "run.log").read_text(errors="replace") if (d / "run.log").exists() else ""
        status = Counter(r.get("status") for r in llm)
        task_key, plan, seed = d.name.split("__")
        injected = [c for r in ret for c in r.get("code_chars", [])]
        rows.append({
            "agent": "openevolve", "cell": d.name, "task": f"math:{task_key}", "arm": plan, "seed": seed,
            "requests": len(llm),
            "output_truncated": sum(1 for r in llm if r.get("finish_reason") == "length"),
            "failed_requests": sum(v for k, v in status.items() if k != "success"),
            "context_overflow": len(re.findall(r"maximum context length|context length|too many tokens", log)),
            "max_prompt_tokens": max((r.get("prompt_tokens") or 0 for r in llm), default=0),
            "child_rejected_too_long": len(re.findall(r"exceeds maximum length", log)),
            "injected_programs": len(injected),
            "max_injected_code_chars": max(injected, default=None),
            "injected_over_max_code_length": sum(1 for c in injected if c > OE_MAX_CODE),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--greedy", nargs="*", default=[])
    ap.add_argument("--oe", required=True)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    report = {"greedy": greedy(args.greedy, args.dsn) if args.greedy else [], "openevolve": openevolve(args.oe)}
    Path(args.out).write_text(json.dumps(report, indent=1))
    g, o = report["greedy"], report["openevolve"]
    print(f"greedy: {sum(r['retrieved_code_truncated'] for r in g)} of {sum(r['retrieved_items'] for r in g)} "
          f"retrieved programs cut at {GREEDY_CODE_CUT} chars; specs cut: "
          f"{sorted({r['task'] for r in g if r['spec_truncated']})}")
    print(f"openevolve: {sum(r['output_truncated'] for r in o)} outputs hit max_tokens, "
          f"{sum(r['failed_requests'] for r in o)} failed requests, "
          f"{sum(r['child_rejected_too_long'] for r in o)} children rejected as too long, "
          f"max prompt {max((r['max_prompt_tokens'] for r in o), default=0)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
