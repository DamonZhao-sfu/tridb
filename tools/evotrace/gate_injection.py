"""Phase 0 gate 6 -- did the injected programs actually reach the rendered prompt?

Arms B and C claim the model saw historical code. Nothing in the pipeline verifies
that. Three ways it silently does not happen, none of which raise:

  1. `PromptSampler` deduplicates inspirations against the top/diverse sections
     (`prompt/sampler.py:449-455`), so an injected program that also appears among the
     run's own high scorers is dropped from the inspirations block.
  2. The worker resolves ids with
     `[programs[pid] for pid in inspiration_ids if pid in programs]` -- that trailing
     `if` is a SILENT filter. An injected program missing from the snapshot vanishes
     with no error and no log line.
  3. A long prompt is truncated before the inspirations section is reached.

v0.1.0 of the plan caught this with a "poisoned memory" arm: inject known dead ends and
check the outcome degrades. That arm was cut, so this assertion replaces it. It is
strictly weaker -- it proves the text REACHED the prompt, not that the model was
influenced by it -- and the report must not claim more than that.

    python3 -m tools.evotrace.gate_injection --run bench/out/oe/circle_packing_gem
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Injected ids carry this prefix; it is also what makes them greppable in a prompt.
# Imported from `constants` rather than `memory_database` so this gate does not pull in
# `openevolve`: the gates run under the repo venv, the runner under `.venv-e0`.
from bench.agent_memory.gem_oe.constants import EXTERNAL_PREFIX


def _iter_prompts(run: Path) -> dict[int, str]:
    """Rendered prompts by iteration, from the evolution trace."""
    trace = run / "evolution_trace.jsonl"
    if not trace.is_file():
        raise SystemExit(
            f"no evolution trace at {trace}; the run must set "
            "evolution_trace.enabled=true and include_prompts=true"
        )
    out: dict[int, str] = {}
    for line in trace.open():
        row = json.loads(line)
        iteration = row.get("iteration") or row.get("evolution_round")
        prompt = row.get("prompt") or {}
        if isinstance(prompt, dict):
            text = f"{prompt.get('system', '')}\n{prompt.get('user', '')}"
        else:
            text = str(prompt)
        if iteration is not None and text.strip():
            out[int(iteration)] = text
    return out


def check(run: Path, *, allow_missing_fraction: float = 0.0) -> dict[str, Any]:
    injections = [
        json.loads(line) for line in (run / "injection_trace.jsonl").open()
    ]
    prompts = _iter_prompts(run)

    checked = 0
    absent: list[dict[str, Any]] = []
    no_prompt: list[int] = []
    for row in injections:
        ids = row.get("injected_ids") or []
        if not ids:
            continue
        text = prompts.get(row["iteration"])
        if text is None:
            no_prompt.append(row["iteration"])
            continue
        for pid in ids:
            checked += 1
            # Match on the corpus uid, not the whole prefixed id: the prompt renders
            # the program, and the id may appear in a different form or not at all,
            # so the code's own identity is what must be present. The uid is embedded
            # in the id after the prefix.
            uid = pid[len(EXTERNAL_PREFIX):]
            if pid not in text and uid not in text:
                absent.append({"iteration": row["iteration"], "program_id": pid})

    total_injected = sum(len(r.get("injected_ids") or []) for r in injections)
    result = {
        "run": str(run),
        "iterations_with_injection": sum(
            1 for r in injections if r.get("injected_ids")
        ),
        "programs_injected": total_injected,
        "programs_checked": checked,
        "programs_absent_from_prompt": len(absent),
        "iterations_without_prompt": len(no_prompt),
        "absent_examples": absent[:5],
        "arms_seen": sorted({r.get("arm") for r in injections if r.get("arm")}),
        "retrieval_errors": [r["iteration"] for r in injections if r.get("error")],
    }
    fraction = (len(absent) / checked) if checked else 0.0
    result["absent_fraction"] = round(fraction, 4)
    result["passed"] = (
        checked > 0
        and not no_prompt
        and fraction <= allow_missing_fraction
        and not result["retrieval_errors"]
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument(
        "--allow-missing-fraction",
        type=float,
        default=0.0,
        help="tolerance for injected programs absent from the prompt; defaults to "
             "zero because any silent drop unbalances the token budget the arms share",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    result = check(args.run, allow_missing_fraction=args.allow_missing_fraction)
    for key, value in result.items():
        if key != "absent_examples":
            print(f"  {key:<32} {value}")
    if result["absent_examples"]:
        print("  absent examples:")
        for row in result["absent_examples"]:
            print(f"    {row}")
    print(f"\ngate: {'PASS' if result['passed'] else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"receipt: {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
