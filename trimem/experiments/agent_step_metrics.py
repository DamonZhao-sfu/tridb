"""Interaction-length metrics, separate from inference calls and retrieval work.

AgentGym: each generated action is one agent step; completed client.step calls
are environment steps. The fixed three-message prompt prefix is excluded.
Tau2: generated assistant turns are agent steps; tool calls may be parallel and
user-side tool calls are excluded. Orchestrator steps count user/tool/agent
transitions and are recorded separately, never inferred from message count.
"""

from __future__ import annotations

import json
from pathlib import Path

VERSION = "agent-interaction-steps-v1"
TAU2_GREETING = "Hi! How can I help you today?"


def _missing(source="unavailable"):
    return {
        "step_metrics_version": VERSION,
        "agent_steps": None,
        "environment_steps": None,
        "agent_tool_calls": None,
        "agent_message_steps": None,
        "orchestrator_steps": None,
        "step_metrics_source": source,
    }


def agentgym_steps(result: dict) -> dict:
    messages = result.get("conversations")
    if not isinstance(messages, list) or len(messages) < 3:
        return _missing()
    if [m.get("role") for m in messages[:3]] != ["user", "assistant", "user"]:
        return _missing("unsupported-agentgym-prefix")
    tail = messages[3:]
    if any(
        m.get("role") != ("assistant" if i % 2 == 0 else "user")
        for i, m in enumerate(tail)
    ):
        return _missing("unsupported-agentgym-turn-order")
    return {
        **_missing("agentgym-api-conversation-v1"),
        "agent_steps": (len(tail) + 1) // 2,
        "environment_steps": len(tail) // 2,
    }


def tau2_steps(
    simulation: dict,
    *,
    initial_message_count: int | None = None,
    orchestrator_steps: int | None = None,
) -> dict:
    saved = (simulation.get("info") or {}).get("trimem_step_metrics")
    if (
        saved
        and saved.get("step_metrics_version") == VERSION
        and initial_message_count is None
    ):
        return dict(saved)
    messages = simulation.get("messages")
    if not isinstance(messages, list):
        return _missing()
    if initial_message_count is None:
        if (
            messages
            and messages[0].get("role") == "assistant"
            and messages[0].get("content") == TAU2_GREETING
            and not messages[0].get("tool_calls")
            and not messages[0].get("usage")
        ):
            initial_message_count = 1
        else:
            return _missing("unknown-tau2-initial-history")
    if type(initial_message_count) is not int or not 0 <= initial_message_count <= len(
        messages
    ):
        raise ValueError("invalid initial history boundary")
    turns = [
        m for m in messages[initial_message_count:] if m.get("role") == "assistant"
    ]
    return {
        **_missing("tau2-assistant-turns-v1"),
        "agent_steps": len(turns),
        "agent_tool_calls": sum(len(m.get("tool_calls") or []) for m in turns),
        "agent_message_steps": sum(not m.get("tool_calls") for m in turns),
        "orchestrator_steps": orchestrator_steps,
        "initial_message_count": initial_message_count,
    }


def annotate_agentgym_result(path: Path) -> dict:
    """Only annotate a completed evaluator result; preserve its trajectory and reward."""
    result = json.loads(path.read_text())
    metrics = agentgym_steps(result)
    if all(result.get(k) == v for k, v in metrics.items()):
        return metrics
    result.update(metrics)
    temp = path.with_suffix(".steps.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=4) + "\n")
    temp.replace(path)
    return metrics


def export_steps(roots: list[Path], destination: Path) -> dict:
    """Backfill a separate observation file from completed results; never rewrite live results."""
    import hashlib
    import math
    from collections import Counter, defaultdict
    from statistics import mean, stdev

    destination.mkdir(parents=True, exist_ok=True)
    counts, groups, failures = Counter(), defaultdict(list), []
    output = destination / "agent_steps.jsonl"
    temp = output.with_suffix(".tmp")
    seen = set()
    with temp.open("w") as out:
        for root in roots:
            for path in root.rglob("*.json"):
                if path in seen or not (
                    path.name.startswith(("alfworld_", "sciworld_"))
                    or "__" in path.name
                ):
                    continue
                seen.add(path)
                try:
                    raw = path.read_bytes()
                    r = json.loads(raw)
                    if not isinstance(r, dict) or "reward" not in r:
                        continue
                    if "simulation" in r:
                        dataset, key = "tau2", f"{r.get('domain')}:{r.get('task_id')}"
                        metrics = tau2_steps(r["simulation"])
                        success = float(r["reward"]) >= 1
                    elif "conversations" in r and r.get("item_id", "").startswith(
                        ("alfworld_", "sciworld_")
                    ):
                        key = r["item_id"]
                        dataset = key.split("_")[0]
                        metrics = agentgym_steps(r)
                        success = float(r["reward"]) >= (
                            100 if dataset == "sciworld" else 1
                        )
                    else:
                        continue
                    rec = {
                        "dataset": dataset,
                        "task_key": key,
                        "trial": r.get("trial"),
                        "result_file": str(path.resolve()),
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "arm": str(path.parent),
                        "success": success,
                        "reused": bool(r.get("reuse") or r.get("reused_from")),
                        **metrics,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[dataset] += 1
                    groups[(dataset, str(path.parent))].append(rec)
                except (OSError, ValueError, TypeError) as exc:
                    failures.append({"path": str(path), "error": repr(exc)})
    temp.replace(output)
    summaries = []
    for (dataset, arm), rows in sorted(groups.items()):
        for subset in ("all", "success", "failure"):
            selected = [
                r
                for r in rows
                if subset == "all" or r["success"] == (subset == "success")
            ]
            task_values = defaultdict(list)
            for r in selected:
                if r["agent_steps"] is not None:
                    task_values[r["task_key"]].append(r["agent_steps"])
            values = [mean(v) for v in task_values.values()]
            summaries.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "subset": subset,
                    "episodes": len(selected),
                    "counted_episodes": sum(len(v) for v in task_values.values()),
                    "tasks": len(values),
                    "mean_agent_steps": mean(values) if values else None,
                    "se_task_mean": stdev(values) / math.sqrt(len(values))
                    if len(values) > 1
                    else None,
                }
            )
    result = {
        "version": VERSION,
        "dataset_episodes": dict(counts),
        "groups": summaries,
        "read_errors": failures,
        "se_definition": "sample standard deviation of per-task trial means / sqrt(number of tasks)",
        "scope": "snapshot of completed files; reused episodes describe logical trajectory length, not fresh execution",
    }
    target = destination / "summary.json"
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temp.replace(target)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export interaction-step metrics from saved trajectories"
    )
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = export_steps(args.root, args.out)
    print(
        json.dumps(
            {
                "episodes": result["dataset_episodes"],
                "groups": len(result["groups"]),
                "read_errors": len(result["read_errors"]),
                "output": str(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
