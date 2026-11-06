"""Normalize τ²-bench simulations (official results JSON or this runner's per-task files) into
task -> session -> step rows. Pure functions, no tau2 import, no I/O beyond reading JSON.

A step is one agent turn after the hard-coded greeting: a tool call (``action`` = ``name(args)``,
``observation`` = the tool result) or a message to the user (``thought`` = the text,
``observation`` = the user's reply, including user-side device tool calls in telecom).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

GREETING = "Hi! How can I help you today?"
SUCCESS_EPS = 1e-9


def first_user_message(messages: list[dict]) -> str | None:
    """The customer's first text message: the retrieval query (the agent never sees reason_for_call)."""
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return m["content"].strip()
    return None


def _call(tc: dict) -> str:
    return f"{tc.get('name')}({json.dumps(tc.get('arguments') or {}, ensure_ascii=False, sort_keys=True)})"


def steps(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    started = False
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            if (
                not started
                and not m.get("tool_calls")
                and (m.get("content") or "").strip() == GREETING
            ):
                started = True
                continue
            started = True
            calls = m.get("tool_calls") or []
            out.append(
                {
                    "ord": len(out),
                    "role": "assistant",
                    "thought": (m.get("content") or None),
                    "action": "; ".join(_call(c) for c in calls) or None,
                    "verb": calls[0].get("name") if calls else "respond",
                    "observation": None,
                }
            )
        elif not out:
            continue
        elif (
            role == "tool"
            and out[-1]["action"]
            and m.get("requestor", "assistant") == "assistant"
        ):
            obs = out[-1]["observation"]
            out[-1]["observation"] = (obs + "\n" if obs else "") + str(
                m.get("content") or ""
            )
        elif role in (
            "user",
            "tool",
        ):  # user's reply or the user's own device tools (telecom)
            text = (
                (m.get("content") or "")
                if role == "user"
                else f"[user tool] {m.get('content') or ''}"
            )
            if role == "user" and m.get("tool_calls"):
                text = (
                    (text + " " if text else "")
                    + "[user calls] "
                    + "; ".join(_call(c) for c in m["tool_calls"])
                )
            obs = out[-1]["observation"]
            out[-1]["observation"] = (obs + "\n" if obs else "") + text.strip()
    return out


def session_from_simulation(
    domain: str, sim: dict, *, source: str, model: str | None
) -> dict:
    msgs = sim.get("messages") or []
    reward = float((sim.get("reward_info") or {}).get("reward") or 0.0)
    trial = sim.get("trial") or 0
    return {
        "key": f"{source}:{domain}:{sim['task_id']}:t{trial}:{str(sim.get('id', ''))[:8]}",
        "task_key": f"{domain}:{sim['task_id']}",
        "task_id": str(sim["task_id"]),
        "domain": domain,
        "source": source,
        "model": model,
        "item_id": sim.get("id"),
        "trial": trial,
        "success": reward >= 1.0 - SUCCESS_EPS,
        "final_score": reward,
        "termination_reason": str(sim.get("termination_reason")),
        "first_user_message": first_user_message(msgs),
        "steps": steps(msgs),
    }


def official_results(
    path: Path, *, source: str | None = None
) -> tuple[str, str, list[dict]]:
    """(domain, agent model, sessions) of one ``results/final/*.json`` file."""
    d = json.loads(Path(path).read_text())
    domain = d["info"]["environment_info"]["domain_name"]
    model = d["info"]["agent_info"]["llm"]
    src = source or f"official:{model}"
    return (
        domain,
        model,
        [
            session_from_simulation(domain, s, source=src, model=model)
            for s in d["simulations"]
        ],
    )


def runner_results(result_dir: Path, *, source: str) -> Iterator[dict]:
    """Sessions from this package's per-task result files (``runner.write_result``)."""
    for f in sorted(Path(result_dir).glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "done" or not r.get("simulation"):
            continue
        yield session_from_simulation(
            r["domain"], r["simulation"], source=source, model=r.get("model")
        )


def filter_split(sessions: Iterable[dict], task_ids: set[str]) -> list[dict]:
    return [s for s in sessions if s["task_id"] in task_ids]


def session_text(s: dict) -> str:
    """Session vector text: customer request, outcome and the agent's tool sequence."""
    acts = "; ".join(x["verb"] for x in s["steps"] if x["action"])
    out = "resolved" if s["success"] else f"failed (reward {s['final_score']:g})"
    return (
        f"Domain: {s['domain']}\nCustomer: {s['first_user_message'] or ''}\nOutcome: {out}\n"
        f"Tools: {acts}"
    )[:1500]


def task_text(domain: str, first_message: str) -> str:
    """Task-vector text; the runtime query uses the same format with the live first user message."""
    return f"{domain}: {first_message}"
