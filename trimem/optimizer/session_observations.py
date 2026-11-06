"""Strict train-panel aggregation for configuration-level surrogate observations."""

import hashlib
import json
import math
import statistics
from dataclasses import dataclass

from trimem.optimizer import session_space as space


@dataclass(frozen=True)
class Point:
    point_id: str
    experiment_id: str
    task: str  # dataset + panel hash, never a held-out test question
    cfg: dict
    start_norm: float
    iter_start: int
    h: int
    quality: float
    tokens: float
    latency_ms: float
    noise: tuple[float, float, float]


def aggregate(rows, *, panel_ids, dataset, panel_hash, experiment_id):
    """One complete panel mean per config; reject leakage, duplicates and gaps.

    Individual rows remain in opt. The GP fits up to51 panel means rather than a
    cubic dense5000-row GP, matching the intended global operating point.
    """
    expected = {str(i) for i in panel_ids}
    if len(expected) != len(panel_ids) or len(expected) < 2:
        raise ValueError("Panel must contain distinct queries")
    groups = {}
    for r in rows:
        if (
            r["split"] != "train"
            or r["dataset"] != dataset
            or r["experiment_id"] != experiment_id
        ):
            raise ValueError("Wrong split, dataset or experiment in BO input")
        if r["status"] != "evaluated" or r["protocol"]["panel_hash"] != panel_hash:
            raise ValueError("Incomplete or incompatible profiling observation")
        if str(r["query_id"]) not in expected:
            raise ValueError("Question outside the frozen training panel")
        cfg = space.validate(r["config"])
        group = groups.setdefault(space.key(cfg), {})
        if str(r["query_id"]) in group:
            raise ValueError(
                "Duplicate completed query/configuration; reconcile attempts before fit"
            )
        if any(
            r[k] is None or not math.isfinite(r[k])
            for k in ("paired_gain", "total_tokens", "e2e_ms")
        ):
            raise ValueError("Missing or non-finite objectives")
        if (
            not -1 <= r["paired_gain"] <= 1
            or r["total_tokens"] <= 0
            or r["e2e_ms"] <= 0
        ):
            raise ValueError("Objective outside its valid domain")
        group[str(r["query_id"])] = r
    points = []
    for cfg_key, group in groups.items():
        if set(group) != expected:
            continue  # A partial panel is not a completed candidate observation.
        data = [group[q] for q in sorted(expected)]
        q = [r["paired_gain"] for r in data]
        t = [r["total_tokens"] for r in data]
        lat = [r["e2e_ms"] for r in data]
        n = len(data)
        # Delta-method variance for log(mean cost), not mean(log cost).
        noise = (
            max(statistics.variance(q) / n, 1e-8),
            max(statistics.variance(t) / n / statistics.mean(t) ** 2, 1e-8),
            max(statistics.variance(lat) / n / statistics.mean(lat) ** 2, 1e-8),
        )
        points.append(
            Point(
                hashlib.sha256(cfg_key.encode()).hexdigest(),
                experiment_id,
                dataset + ":" + panel_hash,
                json.loads(cfg_key),
                0,
                0,
                1,
                statistics.mean(q),
                statistics.mean(t),
                statistics.mean(lat),
                noise,
            )
        )
    return points


def _ys(points):
    return (
        [[p.quality, -math.log(p.tokens), -math.log(p.latency_ms)] for p in points],
        [list(p.noise) for p in points],
    )
