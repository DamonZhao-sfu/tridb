"""Deterministic content-aware responder for MockLLM in the EvoTrace workload.

It answers each semantic operator from the input it is shown, so that changing the input
(e.g. a stale read that sees the raw outcome instead of the extracted lesson) changes the
verdict, exactly as it would for a real model. Selectivities are controlled by hashes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from bench.agent_memory.semantic.ops import SPECS

_OP_BY_SYSTEM = {spec.system: name for name, spec in SPECS.items()}


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def make_responder(*, keep_rate: float = 0.4, fix_rate: float = 0.5) -> Any:
    keep_cut = int(keep_rate * 1000)
    fix_cut = int(fix_rate * 1000)

    def respond(system: str, user: str) -> str:
        op = _OP_BY_SYSTEM.get(system, "")
        payload = json.loads(user)
        if op == "sem_extract":
            o = payload.get("item") or {}
            what = o.get("change") or "no edit description"
            outcome = (
                o.get("judge")
                or o.get("error")
                or ("ok" if o.get("valid") else "invalid")
            )
            return json.dumps(
                {
                    "text": f"{o.get('language')} attempt ({what[:60]}) -> {outcome}, "
                    f"fitness {o.get('fitness')}; reuse the edit only if it improved fitness."
                }
            )
        if op == "sem_match":
            right = payload.get("right") or {}
            ok = bool(right.get("valid")) and (_h(user) % 1000) < fix_cut
            return json.dumps({"match": ok})
        if op == "sem_filter":
            return json.dumps({"keep": (_h(user) % 1000) < keep_cut})
        if op == "sem_conflict":
            return json.dumps({"conflict": (_h(user) % 1000) < 100})
        if op == "sem_rank":
            return json.dumps({"score": _h(user) % 11})
        if op == "sem_summarize":
            return json.dumps(
                {"text": f"summary of {len(payload.get('items') or [])} items"}
            )
        return json.dumps({"verdict": False})

    return respond
