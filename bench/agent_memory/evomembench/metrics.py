"""Pre-registered paired statistics for EvoMemBench outcome runs."""

from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean
from typing import Hashable, Iterable


def clustered_paired_effect(
    rows: Iterable[tuple[Hashable, float, float]],
    *,
    seed: int = 42,
    repetitions: int = 10_000,
) -> dict[str, float | int | list[float]]:
    """Paired arm-baseline effect with whole-cluster bootstrap resampling."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    all_deltas: list[float] = []
    for cluster, baseline, treatment in rows:
        delta = float(treatment) - float(baseline)
        grouped[cluster].append(delta)
        all_deltas.append(delta)
    if not all_deltas:
        raise ValueError("paired effect needs at least one observation")
    clusters = list(grouped)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(repetitions):
        sampled = [rng.choice(clusters) for _ in clusters]
        values = [delta for cluster in sampled for delta in grouped[cluster]]
        draws.append(mean(values))
    draws.sort()
    lower = draws[int(0.025 * (repetitions - 1))]
    upper = draws[int(0.975 * (repetitions - 1))]
    n = len(all_deltas)
    return {
        "n": n,
        "clusters": len(clusters),
        "paired_mean_delta": mean(all_deltas),
        "clustered_bootstrap_95_ci": [lower, upper],
        "positive_transfer_fraction": sum(value > 0 for value in all_deltas) / n,
        "neutral_transfer_fraction": sum(value == 0 for value in all_deltas) / n,
        "negative_transfer_fraction": sum(value < 0 for value in all_deltas) / n,
        "bootstrap_seed": seed,
        "bootstrap_repetitions": repetitions,
    }
