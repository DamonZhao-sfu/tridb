"""ALFWorld / ScienceWorld session space for the shared session-pool BO engine.

Same interface as ``session_space`` (the SWE-QA space, unchanged). Differences for the agent-env
datasets (user decisions 2026-10-16): plans VGR, VGR+Rank, P2 (shown as VGR+SemFilter), P2+Rank
(Rank = ColBERT); an ``edges`` knob over the agentenv session edges; render fixed to the agentenv
memory card. Activate with ``session_pool.use_space(session_space_agentenv)``.
"""

import itertools
import json

VERSION = "session-space-agentenv-v1"
PLANS = ("VGR", "VGR+Rank", "P2", "P2+Rank")
EDGES = ("similar", "similar_up", "similar+similar_up")
RENDERS = ("agentenv_card",)
CAT_DIMS = [4, 5]  # plan, edges
NOMEM = {"plan": "NoMemory", "seed_count": 0, "depth": 0, "fanout": 0, "top_k": 0}
DISPLAY = {
    "VGR": "VGR",
    "VGR+Rank": "VGR+Rank",
    "P2": "VGR+SemFilter",
    "P2+Rank": "VGR+SemFilter+Rank",
}


def grid():
    return {
        "seed_count": (4, 8, 16, 32),
        "depth": (1, 2, 3),
        "fanout": (2, 4, 8, 16),
        "top_k": (1, 3, 5),
        "edges": EDGES,
    }


def plan_defaults():
    return {"seed_count": 8, "depth": 1, "fanout": 4, "top_k": 3, "edges": "similar"}


def orderings(plan):
    return ("colbert_score",) if plan.endswith("+Rank") else ("query_cosine",)


def fallbacks(plan):
    return ("none",)


def v7_defaults(plan):
    """Fixed capabilities (interface name kept for the shared engine)."""
    return {
        "render": "agentenv_card",
        "ordering": orderings(plan)[0],
        "dedup": "id",
        "cadence": 1,
        "sem_fallback": "none",
    }


def domain(plan):
    if plan == "NoMemory":
        return {}
    if plan not in PLANS:
        raise ValueError("Unknown session plan")
    return grid()


def complete(cfg):
    if cfg["plan"] == "NoMemory":
        return dict(NOMEM)
    return {"plan": cfg["plan"], **plan_defaults(), **v7_defaults(cfg["plan"]), **cfg}


def validate(cfg):
    out = complete(cfg)
    if out["plan"] == "NoMemory":
        return out
    for k, levels in domain(out["plan"]).items():
        if out[k] not in levels or isinstance(out[k], bool):
            raise ValueError(f"Invalid session configuration {k}={out[k]}")
    for k, value in v7_defaults(out["plan"]).items():
        if out[k] != value:
            raise ValueError(f"Unsupported session capability {k}={out[k]}")
    if set(out) != {"plan", *grid(), *v7_defaults(out["plan"])}:
        raise ValueError("Unknown configuration fields")
    return out


def key(cfg):
    return json.dumps(validate(cfg), sort_keys=True, separators=(",", ":"))


def horizon(cfg):
    return 1  # one complete episode


def features(cfg, task=None, start_norm=0, iter_start=0, h=1):
    cfg = validate(cfg)
    if cfg["plan"] == "NoMemory":
        return [0.0, 0.0, 0.0, 0.0, float(len(PLANS)), float(len(EDGES))]
    return [
        cfg["seed_count"] / 32,
        cfg["depth"] / 3,
        cfg["fanout"] / 16,
        cfg["top_k"] / 5,
        float(PLANS.index(cfg["plan"])),
        float(EDGES.index(cfg["edges"])),
    ]


def space():
    yield dict(NOMEM)
    for plan in PLANS:
        d = domain(plan)
        for values in itertools.product(*d.values()):
            yield validate({"plan": plan, **dict(zip(d, values, strict=True))})


def runner_args(cfg) -> dict:
    """agentgym_session_eval arguments for a configuration."""
    cfg = validate(cfg)
    return {
        "plan": cfg["plan"],
        "edges": cfg["edges"],
        "memory_config": {
            k: cfg[k] for k in ("seed_count", "depth", "fanout", "top_k")
        },
    }
