"""Dataset-independent session space adapter for the latest candidate-pool BO.

Only graph capabilities actually present in SWE-QA are enabled in v1. Membership
hydration is mandatory; similarity expansion is configurable in breadth/depth.
No test-question feature is consumed by the global configuration selector.
"""

import itertools
import json

VERSION = "session-space-v1"
PLANS = ("VGR", "P1", "P2", "VGR+SemRank")
RENDERS = ("actions_only", "actions_observations", "summary")
CAT_DIMS = [4, 5]
NOMEM = {"plan": "NoMemory", "seed_count": 0, "depth": 0, "fanout": 0, "top_k": 0}


def grid():
    return {
        "seed_count": (8, 16, 24, 32),
        "depth": (1, 2, 3),
        "fanout": (2, 4, 8, 16),
        "top_k": (1, 3, 5),
    }


def plan_defaults():
    return {"seed_count": 8, "depth": 1, "fanout": 4, "top_k": 3}


def orderings(plan):
    return ("semantic_score",) if plan == "VGR+SemRank" else ("query_cosine",)


def fallbacks(plan):
    return ("none",)


def v7_defaults(plan):
    """Interface name retained to reuse the upstream pool engine verbatim."""
    return {
        "edges": "membership+similar",
        "render": "actions_observations",
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
    return {**grid(), "render": RENDERS}


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
        if k != "render" and out[k] != value:
            raise ValueError(f"Unsupported session capability {k}={out[k]}")
    if set(out) != {"plan", *grid(), *v7_defaults(out["plan"])}:
        raise ValueError("Unknown configuration fields")
    return out


def key(cfg):
    return json.dumps(validate(cfg), sort_keys=True, separators=(",", ":"))


def horizon(cfg):
    return 1  # One complete answer episode, not a Math continuation window.


def features(cfg, task=None, start_norm=0, iter_start=0, h=1):
    cfg = validate(cfg)
    if cfg["plan"] == "NoMemory":
        return [0.0, 0.0, 0.0, 0.0, float(len(PLANS)), 0.0]
    return [
        cfg["seed_count"] / 32,
        cfg["depth"] / 3,
        cfg["fanout"] / 16,
        cfg["top_k"] / 5,
        float(PLANS.index(cfg["plan"])),
        float(RENDERS.index(cfg["render"])),
    ]


def space():
    yield dict(NOMEM)
    for plan in PLANS:
        d = domain(plan)
        for values in itertools.product(*d.values()):
            yield validate({"plan": plan, **dict(zip(d, values, strict=True))})
