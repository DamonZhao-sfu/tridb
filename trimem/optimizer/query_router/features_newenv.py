"""Router query features for AppWorld and τ²-bench (same record layout as ``features.raw_record``).

Only what a deployed router knows before the episode is used, identically for training and test
queries:

* AppWorld: the task instruction. ``task_type`` = the first app the instruction names, ``object`` =
  the sorted set of named apps (``+``-joined). The store's ``apps`` / ``difficulty`` columns come from
  the released ground truth, which test tasks do not have, so they are never features.
* τ²-bench: the domain and the customer's first message (the retrieval query). ``task_type`` = the
  domain; task ids / purposes / telecom issue tags are hidden from the agent and are never features.

The probe is one VGR seed lookup through the dataset's own retrieval adapter (``AppWorldTaskAdapter``
or ``Tau2TaskAdapter``) with the retrieval's L1 isolation; the stored sessions are mapped to the same
derived fields, so ``probe.type_match`` / ``probe.object_match`` mean "named the same first app / the
same app set" (AppWorld) and "same domain" (τ², constant 1 by isolation).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from trimem.optimizer.query_router import features

DATASETS = ("appworld", "tau2")
# App names as instructions spell them (AppWorld data 0.2.0 apps, supervisor / api_docs excluded).
APP_WORDS = (
    ("amazon", r"amazon"),
    ("file_system", r"file[ _]?system"),
    ("gmail", r"gmail"),
    ("phone", r"phone"),
    ("simple_note", r"simple[ _]?note"),
    ("splitwise", r"splitwise"),
    ("spotify", r"spotify"),
    ("todoist", r"todoist"),
    ("venmo", r"venmo"),
)
_APP_RX = [(name, re.compile(rf"\b{rx}\b", re.I)) for name, rx in APP_WORDS]


def apps_mentioned(text: str | None) -> list[str]:
    """Apps named in the text, in order of first mention."""
    hits = []
    for name, rx in _APP_RX:
        m = rx.search(text or "")
        if m:
            hits.append((m.start(), name))
    return [name for _, name in sorted(hits)]


def appworld_fields(instruction: str | None) -> dict:
    apps = apps_mentioned(instruction)
    return {
        "task_type": apps[0] if apps else "none",
        "object": "+".join(sorted(apps)) or None,
        "recep": None,
        "description": instruction,
    }


def tau2_fields(domain: str, first_message: str | None) -> dict:
    return {
        "task_type": domain,
        "object": None,
        "recep": None,
        "description": first_message,
    }


def task_fields(dataset: str, row: dict) -> dict:
    """Structured fields of a store task row (AppWorld: description = instruction; τ²: env = domain,
    description = the customer's first message)."""
    if dataset == "appworld":
        return appworld_fields(row.get("description"))
    if dataset == "tau2":
        return tau2_fields(row["env"], row.get("description"))
    raise ValueError(f"unknown dataset {dataset!r}")


def query_identity(dataset: str, row: dict) -> dict:
    """What the dataset's retrieval adapter needs for isolation (same fields as the runners pass)."""
    if dataset == "appworld":
        return {"id": row.get("id"), "key": row["key"], "scenario": row["scenario"]}
    if dataset == "tau2":
        return {
            "key": row["key"],
            "env": row["env"],
            "family": row["family"],
            "entities": list(row.get("entities") or []),
        }
    raise ValueError(f"unknown dataset {dataset!r}")


def session_fields(dataset: str, s: dict) -> dict:
    """A seed session in the probe's terms (cosine and steps from the adapter, type/object derived)."""
    f = task_fields(dataset, s)
    return {
        "task_type": f["task_type"],
        "object": f["object"],
        "cosine": s["cosine"],
        "n_steps": s.get("n_steps"),
    }


def default_adapter(dataset: str, vector: str, identity: dict, dsn: str | None = None):
    if dataset == "appworld":
        from trimem.experiments.appworld.adapter import DSN, AppWorldTaskAdapter

        return AppWorldTaskAdapter(vector, identity, dsn or DSN)
    if dataset == "tau2":
        from trimem.experiments.tau2.adapter import DSN, Tau2TaskAdapter

        return Tau2TaskAdapter(vector, identity, dsn or DSN)
    raise ValueError(f"unknown dataset {dataset!r}")


def probe(
    dataset: str,
    vector: str,
    fields: dict,
    identity: dict,
    *,
    seeds: int = features.PROBE_SEEDS,
    adapter_factory: Callable | None = None,
    dsn: str | None = None,
) -> tuple[dict[str, float], float]:
    """One VGR seed lookup (no expansion, no hydration) with the dataset adapter's isolation."""
    t0 = time.monotonic()
    ad = (adapter_factory or (lambda v, q: default_adapter(dataset, v, q, dsn)))(
        vector, identity
    )
    try:
        sessions = ad.seeds(seeds)
    finally:
        ad.close()
    mapped = [session_fields(dataset, s) for s in sessions]
    return features.probe_from_sessions(mapped, fields), (time.monotonic() - t0) * 1000


def raw_record(
    dataset: str,
    query_id: str,
    fields: dict,
    identity: dict,
    vector: str | list[float] | None,
    *,
    use_probe: bool = True,
    llm: Any = None,
    questions: list[str] | None = None,
    adapter_factory: Callable | None = None,
    dsn: str | None = None,
) -> dict:
    """One raw query record in the ``runtime`` layout (``features.raw_record`` for the new datasets)."""
    rec = {
        "query_id": query_id,
        "task": dict(fields),
        "probe": None,
        "emb": None,
        "llm": None,
        "overhead": {"probe_ms": 0.0, "llm_ms": 0.0, "llm_tokens": 0.0},
    }
    if vector is not None:
        vec = json.loads(vector) if isinstance(vector, str) else list(vector)
        rec["emb"] = [float(v) for v in vec]
        if use_probe:
            rec["probe"], rec["overhead"]["probe_ms"] = probe(
                dataset,
                json.dumps(vec),
                fields,
                identity,
                adapter_factory=adapter_factory,
                dsn=dsn,
            )
    if llm is not None and questions:
        rec["llm"], rec["overhead"]["llm_tokens"], rec["overhead"]["llm_ms"] = (
            features.label_query(llm, questions, fields.get("description") or "", None)
        )
    return rec
