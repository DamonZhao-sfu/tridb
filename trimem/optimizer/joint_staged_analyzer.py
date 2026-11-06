"""Feedback-aware 5-proposal initialization; ordinary advice remains capped at 4."""

import json
import time

from .joint_plan_analyzer import SYSTEM


def advise(space, observations, client, *, n_propose=4, purpose="ordinary"):
    if (purpose, n_propose) not in (("ordinary", 4), ("initialization", 5)):
        raise ValueError("Expected five initialization or four ordinary proposals")
    began = time.monotonic()
    body = {
        "space": space.manifest(),
        "purpose": purpose,
        "requested_proposals": n_propose,
        "complete_train_archive": [
            {
                "config": o.config,
                "quality": o.mean[0],
                "total_tokens": o.mean[1],
                "total_latency_seconds": o.mean[2],
            }
            for o in observations
        ],
    }
    system = SYSTEM
    if purpose == "initialization":
        system = SYSTEM.replace("at most four", "exactly five distinct, unmeasured")
        system += (
            " This is initialization batch 2. Diagnose the five fully measured "
            "Sobol configurations supplied below before proposing batch 2. "
            "These five proposals will be profiled directly, before fitting a GP."
        )
    receipt = {
        "status": "unavailable",
        "purpose": purpose,
        "rejected": 0,
        "proposals": [],
        "input_tokens": None,
        "output_tokens": None,
    }
    try:
        result = client.chat(system, json.dumps(body), temperature=0)
        receipt.update(
            input_tokens=result.input_tokens, output_tokens=result.output_tokens
        )
        if result.error or not result.text:
            raise ValueError(result.error or "empty analyzer response")
        content = json.loads(result.text)
        receipt["diagnosis"] = content.get("diagnosis", "")
        raw = content.get("proposals", [])
        if not isinstance(raw, list):
            raise TypeError("proposals must be an array")
        if purpose == "initialization" and len(raw) != n_propose:
            raise ValueError("Initialization requires exactly five proposals")
        seen = {space.key(o.config) for o in observations}
        for item in raw[:n_propose]:
            try:
                c = space.canonicalize(item.get("config", item))
                key = space.key(c)
                if key in seen:
                    raise ValueError("Repeated or measured proposal")
                seen.add(key)
                receipt["proposals"].append(c)
            except (ValueError, TypeError, AttributeError):
                receipt["rejected"] += 1
        receipt["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - persist failed advice without dispatch
        receipt["error"] = repr(exc)
    receipt["seconds"] = time.monotonic() - began
    return receipt["proposals"], receipt
