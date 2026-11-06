"""Retrieval units: what of each selected trajectory is injected (engine extension v0.1.0).

whole   : the existing card (goal, outcome, full action sequence); byte-identical to
          ``agentenv_task_adapter.render_memory``.
window  : per selected trajectory, WINDOW consecutive steps aligned to the agent's progress
          (steps [t, t+WINDOW) when injecting at agent step t; start injection t=0). When the
          trajectory is shorter than t, its last WINDOW steps are used. Each step shows the
          observation (clipped) and the action.
actions : action-level retrieval (user 2026-11-03). Every step (observation -> action) of the
          selected trajectories is a candidate; a ranker scores each against the agent's
          current observation and the ACTION_STEPS best are injected.

``render_unit`` returns (memory block, receipt). The step ranker for ``actions`` is a callable
``cards -> {id: score}`` (ColBERT in the live runs); its calls are reported in the receipt.
"""

from __future__ import annotations

from collections.abc import Callable

WINDOW = 5
ACTION_STEPS = 10
OBS_CHARS = 300
UNITS = ("whole", "window", "actions")


def _clip(text: str | None, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 3] + "..."


def _head(card: dict, i: int, policy_label: str) -> str:
    out = "success" if card["success"] else f"score {card['final_score']}"
    return (
        f"[Memory {i}] {policy_label}Goal: {card['description']}\n"
        f"Outcome: {out} in {card['n_steps']} steps\n"
    )


def _label(card: dict) -> str:
    from trimem.experiments import memory_policy as mp
    from trimem.experiments.agentenv_task_adapter import outcome_text

    if mp.labelled(card):
        return f"{mp.label(bool(card['success']), outcome_text(card))}\n"
    return ""


def window_bounds(n_steps: int, step_index: int, width: int = WINDOW) -> tuple[int, int]:
    """[a, b) of the steps to inject; the last ``width`` steps once the agent is past the end."""
    if n_steps <= 0:
        return 0, 0
    a = step_index if step_index < n_steps else max(0, n_steps - width)
    return a, min(n_steps, a + width)


def render_window_card(card: dict, i: int, cap_chars: int, step_index: int) -> tuple[str, list[int]]:
    steps = card["steps"]
    a, b = window_bounds(len(steps), step_index)
    lines = [f"Steps {a + 1}-{b} of {len(steps)}:"]
    for s in steps[a:b]:
        lines.append(f"  Observation: {_clip(s.get('observation'), OBS_CHARS)}")
        lines.append(f"  Action: {s.get('action') or ''}")
    text = _head(card, i, _label(card)) + "\n".join(lines)
    if len(text) > cap_chars:
        text = text[: cap_chars - 3] + "..."
    return text, list(range(a, b))


def step_candidates(cards: list[dict]) -> list[dict]:
    """One candidate per step with an action: id '<session id>:<step ord>'."""
    out = []
    for c in cards:
        for j, s in enumerate(c["steps"]):
            if s.get("action"):
                out.append(
                    {
                        "id": f"{c['id']}:{s.get('ord', j)}",
                        "card": c,
                        "j": j,
                        "observation": s.get("observation") or "",
                        "action": s["action"],
                        "text": f"Observation: {_clip(s.get('observation'), OBS_CHARS)}\n"
                        f"Action: {s['action']}",
                    }
                )
    return out


def render_unit(
    env: str,
    cards: list[dict],
    unit: str,
    cap_chars: int,
    policy: str,
    *,
    step_index: int = 0,
    step_ranker: Callable[[list[dict]], dict] | None = None,
) -> tuple[str, dict]:
    from trimem.experiments.agentenv_task_adapter import memory_header, render_memory

    if unit not in UNITS:
        raise ValueError(f"Unknown retrieval unit {unit!r}")
    receipt = {"retrieval_unit": unit, "step_index": step_index}
    if unit == "whole":
        return render_memory(env, cards, cap_chars, policy), receipt
    if not cards:
        return "", receipt
    if unit == "window":
        body, spans = [], {}
        for i, c in enumerate(cards):
            text, idx = render_window_card(c, i + 1, cap_chars, step_index)
            body.append(text)
            spans[str(c["id"])] = [idx[0], idx[-1] + 1] if idx else [0, 0]
        receipt.update(window=WINDOW, spans=spans)
        return memory_header(env, policy) + "\n\n".join(body), receipt
    # actions: action-level retrieval over the selected trajectories' steps
    if step_ranker is None:
        raise ValueError("actions unit requires a step ranker")
    cands = step_candidates(cards)
    if not cands:
        return "", {**receipt, "action_steps": []}
    scores = step_ranker(cands)
    if set(scores) != {c["id"] for c in cands}:
        raise ValueError("Step ranker must score every step exactly once")
    order = {c["id"]: k for k, c in enumerate(cands)}
    best = sorted(cands, key=lambda c: (-scores[c["id"]], order[c["id"]]))[:ACTION_STEPS]
    per = max(200, cap_chars * len(cards) // max(1, len(best)))
    lines = []
    for k, c in enumerate(best, 1):
        card = c["card"]
        head = f"[Step {k}] From a past episode ({_label(card).strip() or 'outcome unknown'}); goal: {card['description']}"
        text = f"{head}\n{c['text']}"
        lines.append(text if len(text) <= per else text[: per - 3] + "...")
    receipt.update(
        action_steps=[{"id": c["id"], "score": scores[c["id"]]} for c in best],
        candidates=len(cands),
        limit=ACTION_STEPS,
    )
    return memory_header(env, policy) + "\n\n".join(lines), receipt
