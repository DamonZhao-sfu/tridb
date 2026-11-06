"""Unpaired, stratified estimators for per-configuration profiling samples (pool profiling, user design
2026-10-19; docs/trimem_pool_profiling_plan_v0.1.0.md).

Every configuration is measured on its OWN stratified random sample of the profiling pool, so two
configurations share (almost) no queries and per-query paired differences are not available. Each
configuration's metric is the stratified mean over its sample (population weights ``W_h = N_h / N``
of the pool), with the stratified variance of the mean ``sum_h W_h^2 (1 - n_h/N_h) s_h^2 / n_h``;
a stratum with a single sampled query (``s_h`` undefined) uses the pooled within-stratum variance.
Differences between configurations are unpaired: ``SE(a - b) = sqrt(SE_a^2 + SE_b^2)``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    mean: float
    se: float
    n: int


def stratified(
    values: list[float],
    strata: list[str],
    weights: dict[str, float],
    sizes: dict[str, int] | None = None,
) -> Estimate:
    """Stratified mean and its standard error. ``weights`` = pool share of each stratum (sum 1);
    ``sizes`` = pool count of each stratum (finite-population correction; omitted -> no FPC).
    Strata of the pool that the sample misses are renormalized away (never happens with the
    >= 1-per-stratum allocation of ``build_agentenv_pool.draw_sample``)."""
    if len(values) != len(strata) or not values:
        raise ValueError("values and strata must be non-empty and aligned")
    by: dict[str, list[float]] = defaultdict(list)
    for v, h in zip(values, strata, strict=True):
        if h not in weights:
            raise ValueError(f"stratum {h!r} not in the pool weights")
        if not math.isfinite(v):
            raise ValueError("non-finite value")
        by[h].append(float(v))
    wsum = sum(weights[h] for h in by)
    # pooled within-stratum variance for strata with one sampled query
    ss = sum(sum((x - sum(xs) / len(xs)) ** 2 for x in xs) for xs in by.values())
    dof = sum(len(xs) - 1 for xs in by.values())
    pooled = ss / dof if dof > 0 else 0.0
    mean = var = 0.0
    for h, xs in by.items():
        w = weights[h] / wsum
        n = len(xs)
        m = sum(xs) / n
        s2 = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else pooled
        fpc = 1.0 - n / sizes[h] if sizes and sizes.get(h) else 1.0
        mean += w * m
        var += w * w * max(fpc, 0.0) * s2 / n
    return Estimate(mean, math.sqrt(var), len(values))


def simple(values: list[float], population: int | None = None) -> Estimate:
    """Simple-random-sample mean and its standard error (sample-size-v1, user 2026-10-22: configurations whose
    sample is drawn UNIFORMLY from the pool, not stratified): plain mean, SE = sqrt(fpc * s^2 / n) with the finite
    population correction 1 - n / ``population`` (omitted -> none). Unbiased for the pool mean; post-stratifying
    a 100-task uniform sample with the pool weights would renormalise away the strata it misses (biased)."""
    if not values:
        raise ValueError("values must be non-empty")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite value")
    n = len(values)
    m = sum(values) / n
    s2 = sum((x - m) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    fpc = 1.0 - n / population if population else 1.0
    return Estimate(m, math.sqrt(max(fpc, 0.0) * s2 / n), n)


def log_mean_var(est: Estimate) -> float:
    """Delta-method variance of ln(mean) (cost objectives are -ln mean tokens / latency)."""
    if est.mean <= 0:
        raise ValueError("log of a non-positive mean")
    return (est.se / est.mean) ** 2


def diff(a: Estimate, b: Estimate) -> Estimate:
    """Unpaired difference a - b (independent samples)."""
    return Estimate(a.mean - b.mean, math.sqrt(a.se**2 + b.se**2), min(a.n, b.n))


def noise_floor(a: Estimate, b: Estimate) -> float:
    """Noise floor of a configuration-mean gain from two independent runs of the SAME configuration on
    two independent samples: the larger of |mean difference| and the unpaired SE of that difference
    (the unpaired counterpart of ``session_bo_agentenv.noise_floor``)."""
    d = diff(a, b)
    return max(abs(d.mean), d.se)


def paired_overlap(a: dict[str, float], b: dict[str, float]) -> dict | None:
    """Paired comparison restricted to the queries both samples contain (diagnostic only; overlaps of two
    independent 200-samples of a ~4,000-task pool are ~10 queries)."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = [a[q] - b[q] for q in common]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
    return {"n": len(common), "mean": m, "se": se}


# ------------------------------------------------------------------ common-mode noise (gp-fix-v1, 2026-10-20)


def stratified_cov(
    a: list[float], b: list[float], strata: list[str], weights: dict[str, float]
) -> float:
    """Covariance of the stratified means of two series measured on the SAME queries (aligned):
    ``sum_h W_h^2 cov_h(a, b) / n_h`` (no FPC; a single-query stratum uses the pooled within-stratum
    covariance, as ``stratified`` does for the variance). ``stratified_cov(a, a, ...) == stratified(a).se**2``."""
    if not (len(a) == len(b) == len(strata)) or not a:
        raise ValueError("a, b and strata must be non-empty and aligned")
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for x, y, h in zip(a, b, strata, strict=True):
        if h not in weights:
            raise ValueError(f"stratum {h!r} not in the pool weights")
        by[h].append((float(x), float(y)))

    def cross(xs: list[tuple[float, float]]) -> float:
        mx, my = sum(x for x, _ in xs) / len(xs), sum(y for _, y in xs) / len(xs)
        return sum((x - mx) * (y - my) for x, y in xs)

    dof = sum(len(xs) - 1 for xs in by.values())
    pooled = sum(cross(xs) for xs in by.values()) / dof if dof > 0 else 0.0
    wsum = sum(weights[h] for h in by)
    cov = 0.0
    for h, xs in by.items():
        w, n = weights[h] / wsum, len(xs)
        cov += w * w * (cross(xs) / (n - 1) if n > 1 else pooled) / n
    return cov


def common_mode_cov(
    series: list[list[float]], strata: list[str], weights: dict[str, float]
) -> float:
    """Mean pairwise covariance of the stratified means of >= 2 series measured on the same queries: the
    common-mode part of every estimator's variance (shared queries, a shared paired baseline). A GP with a
    constant mean absorbs it, so it is not independent per-point noise. 0.0 with fewer than 2 series."""
    if len(series) < 2:
        return 0.0
    covs = [
        stratified_cov(series[i], series[j], strata, weights)
        for i in range(len(series))
        for j in range(i + 1, len(series))
    ]
    return sum(covs) / len(covs)


IDIO_FLOOR = 0.25  # idiosyncratic noise is never below this share of the full SE^2


def idiosyncratic_var(se2: float, common: float, floor: float = IDIO_FLOOR) -> float:
    """Per-point GP noise once the common-mode covariance is removed: ``max(se2 - common, floor * se2)``."""
    return max(se2 - max(common, 0.0), floor * se2, 1e-8)
