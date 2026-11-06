"""Schema-aware analyzer channel; advice is never a measured GP label."""

import json
import time

SYSTEM = """You are TRACE's plan-and-parameter search analyzer. First diagnose the
observed TRAIN trade-offs, citing configurations and measured evidence; then
propose at most {max_proposals} diverse legal configurations that may extend the frontier.
Quality is the raw environment score, including negative scores; maximize it
while reducing TOTAL memory+agent token and latency costs. Do not assume more
retrieval always helps. State a falsifiable rationale for each proposal. These
are hypotheses for acquisition ranking, not guaranteed improvements. Use only
the supplied capability domains. Inactive fields may be null. V is fixed task
ANN. Return JSON only: {"diagnosis":"...", "proposals":[{"config":{...},
"rationale":"...", "priority_knobs":[...]}]}. No TEST data is provided."""


def parse_json_reply(text):
    """Parse the analyzer's JSON reply. The model often wraps it in a ```json
    fenced block or adds prose; take the outermost {...} object."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(t[start : end + 1])


def advise(space, observations, client, *, max_proposals=4, initialization=False):
    if max_proposals not in (4, 5):
        raise ValueError("Expected four search proposals or five initialization probes")
    began = time.monotonic()
    body = {
        "space": space.manifest(),
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
    receipt = {
        "status": "unavailable",
        "rejected": 0,
        "proposals": [],
        "input_tokens": None,
        "output_tokens": None,
    }
    try:
        system = SYSTEM.replace("{max_proposals}", str(max_proposals))
        if initialization:
            system += (
                " This is initialization phase B. Design exactly 5 distinct new probes "
                "using the measured phase-A Sobol results. Cover informative plan/parameter "
                "trade-offs and avoid every configuration in complete_train_archive."
            )
        result = client.chat(system, json.dumps(body), temperature=0)
        receipt.update(
            input_tokens=result.input_tokens, output_tokens=result.output_tokens
        )
        receipt["raw_text"] = result.text  # audit: the exact model reply
        if result.error or not result.text:
            raise ValueError(result.error or "empty analyzer response")
        content = parse_json_reply(result.text)
        receipt["diagnosis"] = content.get("diagnosis", "")
        raw = content.get("proposals", [])
        if not isinstance(raw, list):
            raise TypeError("proposals must be an array")
        for item in raw[:max_proposals]:
            try:
                c = space.canonicalize(item.get("config", item))
                receipt["proposals"].append(c)
            except (ValueError, TypeError, AttributeError):
                receipt["rejected"] += 1
        receipt["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - analyzer failure is an optional empty source
        receipt["error"] = repr(exc)
    receipt["seconds"] = time.monotonic() - began
    return receipt["proposals"], receipt
