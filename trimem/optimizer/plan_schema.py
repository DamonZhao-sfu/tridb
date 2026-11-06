"""TRACE v0.4.2 finite typed-plan domain with fixed task-vector ANN."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

SCHEMA_VERSION = "trace-pr-v060-engine-ext-v1"
FIELDS = (
    # "none" (no SemFilter) is appended so existing categorical codes are unchanged.
    ("filter_order", "cat", ("before", "after", "both", "none")),
    ("graph_impl", "cat", ("BFS", "best-first", "beam", "MCTS", "evolutionary")),
    ("filter_pre_impl", "cat", ("LLM", "relational")),
    ("filter_post_impl", "cat", ("LLM", "relational")),
    ("rank_impl", "cat", ("ColBERT", "LLM", "cross-encoder")),
    ("retrieval_unit", "cat", ("whole", "actions", "window")),
    ("injection_timing", "cat", ("start", "periodic", "event")),
    ("seed_count", "ord", (4, 8, 16, 32, 64, 128)),
    ("depth", "ord", (2, 3, 4)),
    ("fanout", "ord", (4, 8, 16, 32, 64)),
    ("edge_threshold", "cat", ("off", "low", "high")),
    ("beam_width", "ord", (1, 2, 4, 8)),
    ("sem_threshold_pre", "ord", (0.4, 0.6, 0.8)),
    ("sem_threshold_post", "ord", (0.4, 0.6, 0.8)),
    ("retrieval_interval", "ord", (5, 10)),
    ("returned_k", "ord", (3, 5, 8, 10)),
    ("trajectory_category", "cat", ("Success", "All")),
    # Forward = similar, Backward = similar_up (reverse copies), Both = both edge types
    # (= every historical run; "Both" appended so earlier categorical codes are unchanged).
    ("graph_direction", "cat", ("Backward", "Forward", "Both")),
)
FIELD_NAMES = tuple(row[0] for row in FIELDS)
MASK_FIELDS = (
    "filter_pre_impl",
    "filter_post_impl",
    "beam_width",
    "sem_threshold_pre",
    "sem_threshold_post",
    "retrieval_interval",
)
DEFAULTS = {
    **{name: values[0] for name, _, values in FIELDS},
    "seed_count": 16,
    "fanout": 8,
    "beam_width": 2,
    "sem_threshold_pre": 0.6,
    "sem_threshold_post": 0.6,
    "returned_k": 5,
    "trajectory_category": "All",
    "graph_direction": "Both",
}
LIVE_DOMAINS = {
    # Rank is mandatory; SemFilter is optional: VGR+Rank, P1+Rank, P2+Rank.
    "filter_order": ("none", "before", "after"),
    # Engine extension v0.1.0 (user 2026-11-03).
    "graph_impl": ("BFS", "best-first", "beam", "MCTS"),
    "filter_pre_impl": ("LLM",),
    "filter_post_impl": ("LLM",),
    "rank_impl": ("ColBERT", "LLM", "cross-encoder"),
    "retrieval_unit": ("whole", "actions", "window"),
    "injection_timing": ("start", "periodic"),
    "edge_threshold": ("off",),
    "graph_direction": ("Backward", "Forward", "Both"),
}
DIRECTION_EDGES = {
    "Forward": ("similar",),
    "Backward": ("similar_up",),
    "Both": ("similar", "similar_up"),
}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PlanNode:
    role: str
    implementation: str
    inputs: tuple[str, ...]
    parameters: dict


@dataclass(frozen=True)
class PlanSpec:
    nodes: tuple[PlanNode, ...]
    injection: dict

    def describe(self) -> str:
        return " → ".join(f"{n.role}({n.implementation})" for n in self.nodes)


class JointSpace:
    """Full representable domain or a frozen, per-field capability subset.

    The full domain is representational: only ``live`` is mapped to the legacy
    executor. No declaration here claims that an extension has been implemented.
    """

    def __init__(
        self,
        capabilities: Mapping | None = None,
        *,
        execution_fingerprint: str = "representation-only",
    ):
        caps = dict(capabilities or {})
        if set(caps) - set(FIELD_NAMES):
            raise ValueError("Unknown capability fields")
        self.domains = {}
        for name, _, full in FIELDS:
            requested = tuple(caps.get(name, full))
            if not requested or len(set(requested)) != len(requested):
                raise ValueError(f"Empty or repeated capability for {name}")
            if any(v not in full for v in requested):
                raise ValueError(f"Invalid capability for {name}")
            self.domains[name] = tuple(v for v in full if v in requested)
        self.execution_fingerprint = execution_fingerprint
        self.identity = digest(self.manifest())

    @classmethod
    def live(cls, *, execution_fingerprint="legacy-task-ann-bfs-llm-colbert-v1"):
        return cls(LIVE_DOMAINS, execution_fingerprint=execution_fingerprint)

    def manifest(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "domains": self.domains,
            "fixed_vector": "task-ann",
            "judge": "execution-manifest",
            "cascade": "none",
            "execution_fingerprint": self.execution_fingerprint,
        }

    def active_mask(self, config: Mapping) -> dict[str, bool]:
        pre = config["filter_order"] in ("before", "both")
        post = config["filter_order"] in ("after", "both")
        return {
            "filter_pre_impl": pre,
            "filter_post_impl": post,
            "beam_width": config["graph_impl"] == "beam",
            "sem_threshold_pre": pre and config["filter_pre_impl"] == "LLM",
            "sem_threshold_post": post and config["filter_post_impl"] == "LLM",
            "retrieval_interval": config["injection_timing"] == "periodic",
        }

    def canonicalize(self, config: Mapping) -> dict:
        unknown = set(config) - set(FIELD_NAMES)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        c = {
            name: config.get(name)
            if config.get(name) is not None
            else (
                DEFAULTS[name]
                if DEFAULTS[name] in self.domains[name]
                else self.domains[name][0]
            )
            for name in FIELD_NAMES
        }
        active = self.active_mask(c)
        for name, kind, _ in FIELDS:
            if not active.get(name, True):
                c[name] = None
                continue
            value = c[name]
            if isinstance(value, bool) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError(f"Invalid {name}={value!r}")
            if value not in self.domains[name]:
                raise ValueError(f"Value outside capability: {name}={value!r}")
            # Recover the frozen value, including integer versus float type.
            c[name] = self.domains[name][self.domains[name].index(value)]
        return c

    def default(self, **overrides) -> dict:
        return self.canonicalize(overrides)

    def key(self, config: Mapping) -> str:
        return digest([self.identity, self.canonicalize(config)])

    def sample(self, n: int, *, seed: int, avoid=(), max_draws=4096) -> list[dict]:
        import torch

        if n < 0 or max_draws < 0:
            raise ValueError("Sample budgets must be nonnegative")
        if n == 0:
            return []
        engine = torch.quasirandom.SobolEngine(len(FIELDS), scramble=True, seed=seed)
        seen, blocked = {}, set(avoid)
        for row in engine.draw(max_draws).double().tolist():
            c = self.canonicalize(
                {
                    name: self.domains[name][
                        min(
                            int(u * len(self.domains[name])),
                            len(self.domains[name]) - 1,
                        )
                    ]
                    for (name, _, _), u in zip(FIELDS, row)
                }
            )
            k = self.key(c)
            if k not in blocked:
                seen.setdefault(k, c)
            if len(seen) == n:
                break
        return list(seen.values())

    def plan(self, config: Mapping) -> PlanSpec:
        c = self.canonicalize(config)
        nodes = [PlanNode("V", "task-ann", (), {"seed_count": c["seed_count"]})]
        for role, field, threshold in (
            ("F_pre", "filter_pre_impl", "sem_threshold_pre"),
            ("G", "graph_impl", None),
            ("F_post", "filter_post_impl", "sem_threshold_post"),
            ("Rank", "rank_impl", None),
        ):
            if c[field] is None:
                continue
            if role == "G":
                params = {
                    k: c[k]
                    for k in (
                        "depth",
                        "fanout",
                        "edge_threshold",
                        "beam_width",
                        "graph_direction",
                    )
                    if c[k] is not None
                }
            elif role == "Rank":
                params = {"returned_k": c["returned_k"]}
            else:
                params = {} if c[threshold] is None else {"threshold": c[threshold]}
            nodes.append(PlanNode(role, c[field], (nodes[-1].role,), params))
        return PlanSpec(
            tuple(nodes),
            {
                k: c[k]
                for k in (
                    "retrieval_unit",
                    "injection_timing",
                    "retrieval_interval",
                    "trajectory_category",
                )
            },
        )
