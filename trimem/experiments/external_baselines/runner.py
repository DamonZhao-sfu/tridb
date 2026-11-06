"""Runner side of an external-baseline arm (``agentgym_session_eval --external``): one retrieval per TEST episode
from the system's service, rendered and accounted exactly like a TRACE retrieval.

* injected block = the system's native retrieved content (G-Memory: successful shots + insights; AriGraph:
  triplets; Graphiti: facts; EXG: none, its hints are the cards) + its selected sessions rendered by ``render_memory`` (the raw cards TRACE
  injects; unlabelled under success_only, as TRACE), clipped to ``llm_budget.MEMORY_TOKENS``;
* cost: the system's retrieval-time LLM tokens -> ``semantic_tokens`` (like SemFilter), the service round trip
  -> ``retrieval_ms`` (like TRACE's retrieval), one opt row per episode (``store.start``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from trimem.experiments import llm_budget
from trimem.experiments import memory_policy as mp
from trimem.experiments.agentenv_task_adapter import family, render_memory
from trimem.experiments.memory_forms import clip_tokens

PROTOCOL = "external-baseline-v1"


@dataclass(frozen=True)
class ExternalConfig:
    system: str
    trace_label: str
    params: dict


def card_policy(outcome_policy: str) -> str:
    return mp.SUCCESS_ONLY if outcome_policy == "success_only" else mp.SUCCESS_FAILURE


def render(env: str, native: str, cards: list[dict], params: dict) -> str:
    pol = card_policy(params["outcome_policy"])
    if pol == mp.SUCCESS_ONLY:
        cards = [{k: v for k, v in c.items() if k != "memory_policy"} for c in cards]
    cap = llm_budget.per_card(params["top_k"]) * 3  # TRACE's per-card cap
    text = native + (render_memory(env, cards, cap, pol) if cards else "")
    return clip_tokens(text, llm_budget.MEMORY_TOKENS) if text else ""


def query_of(q: dict, item: str) -> dict:
    return {
        "item": item,
        "task_id": q["id"],
        "env": q["env"],
        "family": repr(family(q)),
        "query": f"{q['env']}: {q['description'] or q['key']}",
    }


def retrieve(q: dict, item: str, S: dict) -> dict:
    ext = S["external"]
    params = ext["params"]
    query = query_of(q, item)
    t0 = time.monotonic()
    rid = S["store"].start(
        experiment_id=S["experiment"],
        dataset=q["env"],
        split=S["split"],
        query_id=item,
        query_context={"task_key": q["key"], "goal": q["description"]},
        config=ExternalConfig(ext["system"], ext["trace_label"], params),
        protocol={
            "llm_budget": llm_budget.as_dict(),
            "action_format": S["action_format"],
            "isolation": "L1",
            "external": {
                "protocol": PROTOCOL,
                "system": ext["system"],
                "trace_label": ext["trace_label"],
                "params": params,
            },
            **mp.protocol(S["memory_policy"]),
            **S.get("protocol_extra", {}),
        },
    )
    try:
        r = httpx.post(
            ext["url"].rstrip("/") + "/retrieve",
            json={"query": query, "params": params},
            timeout=3600,
        )
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(body["error"])
    except Exception as exc:
        S["store"].patch(rid, status="failed", error=repr(exc))
        raise
    retrieval_ms = (time.monotonic() - t0) * 1000
    cards = body.get("cards") or []
    memory = render(q["env"], body.get("memory") or "", cards, params)
    receipt = {
        "retrieval_id": rid,
        "external": {k: v for k, v in body.items() if k not in ("memory", "cards")},
        "selected_ids": body.get("selected", []),
        "memory_chars": len(memory),
    }
    S["store"].patch(
        rid,
        status="retrieved_pending_evaluation",
        receipt=receipt,
        retrieval_ms=retrieval_ms,
    )
    return {
        "rid": rid,
        "memory": memory,
        "retrieval_ms": retrieval_ms,
        "semantic_tokens": int(body.get("prompt_tokens", 0))
        + int(body.get("completion_tokens", 0)),
        "selected": body.get("selected", []),
        "memory_policy": card_policy(params["outcome_policy"]),
        "card_outcomes": mp.card_outcomes(cards),
    }
