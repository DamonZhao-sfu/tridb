"""Per-iteration generation cost: join OpenEvolve's LLM telemetry to the iteration that issued it.

The evolution trace stores each iteration's prompt; llm_telemetry stores the sha256 of the same
messages, so the two join exactly. Used to train the per-candidate output-token model, the
validity model and the latency model of objective O4 (plan v0.4.0).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from bench.agent_memory.gem_oe.llm_telemetry import prompt_sha256

R = Path("results/trimem_v0.1.0")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def iteration_costs(cell_dir: Path) -> dict[int, dict]:
    """iteration -> {prompt_tokens, completion_tokens, llm_ms} (successful request of that iteration)."""
    tel: dict[str, list[dict]] = {}  # identical prompts (e.g. iterations 1 and 2) are consumed in order
    for r in sorted(_jsonl(cell_dir / "llm_telemetry.jsonl"), key=lambda x: x.get("started_at_unix") or 0):
        if r.get("status") == "success" and r.get("prompt_tokens"):
            tel.setdefault(r["prompt_sha256"], []).append(r)
    out = {}
    for r in sorted(_jsonl(cell_dir / "evolution_trace.jsonl"), key=lambda x: int(x["iteration"])):
        p = r.get("prompt")
        p = ast.literal_eval(p) if isinstance(p, str) else p
        if not isinstance(p, dict):
            continue
        h = prompt_sha256([{"role": "system", "content": p.get("system", "")},
                           {"role": "user", "content": p.get("user", "")}])
        t = tel[h].pop(0) if tel.get(h) else None
        if t:
            out[int(r["iteration"])] = {"prompt_tokens": t["prompt_tokens"],
                                        "completion_tokens": t["completion_tokens"], "llm_ms": t["llm_api_ms"],
                                        "ttft_ms": t.get("ttft_ms"), "finish_reason": t.get("finish_reason")}
    return out


def session_dirs(session: str) -> list[Path]:
    """Cell directories (iterations 1-20, resumed 21-50) of a trimem_outcome session."""
    if session.startswith("prof__"):  # offline profiling: one resumed iteration per (state, configuration)
        dom = "ale" if "__ahc" in session else "math"
        return [R / "profiling" / dom / session]
    if session.startswith("fixed:"):
        name = session.split(":", 1)[1]
        return [R / "openevolve" / "it20" / name, R / "openevolve" / "it50" / name]
    return [R / "objective" / "it20" / session, R / "objective" / "it50" / session]


def session_costs(session: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for d in session_dirs(session):
        out |= iteration_costs(d)
    return out
