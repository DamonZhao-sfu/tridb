"""Hydra-relative objectives and response/behavior-directed local refinement.

No deployment policy, crossover generation, evaluation, or external IO lives here.
"""

from __future__ import annotations

import numpy as np

from trimem.optimizer.profiling_response import utility


class LocalObjective:
    def __init__(
        self,
        measured,
        choices,
        query_weights,
        weights,
        norm,
        reference_index=0,
        epoch_norm=None,
        labels=None,
    ):
        self.measured = np.asarray(measured)
        self.weights, self.norm = np.asarray(weights), norm
        self.epoch_norm = epoch_norm or norm
        self.p = np.asarray(query_weights, dtype=float)
        if self.p.ndim != 1 or np.any(self.p <= 0) or not np.isclose(self.p.sum(), 1):
            raise ValueError("query weights must be positive and sum to one")
        choices = np.asarray(choices, dtype=int)
        if choices.shape != (len(self.p), len(self.weights)):
            raise ValueError("router choices must have shape [queries,preferences]")
        if np.any(choices < 0) or np.any(choices >= self.measured.shape[1]):
            raise ValueError("router choice outside measured configuration catalog")
        uu = self.utility(self.measured)
        qi, wi = np.indices(choices.shape)
        self.baseline = uu[:, choices, qi, wi]
        self.reference_index = reference_index
        self.reference = self.utility(self.measured, epoch=True)[:, reference_index]
        self.labels = (
            np.zeros(len(self.p), dtype=int) if labels is None else np.asarray(labels)
        )

    def utility(self, raw, *, epoch=False):
        return utility(raw, self.weights, self.epoch_norm if epoch else self.norm)

    def hydra(self, raw, *, baseline=None, group=None):
        baseline = self.baseline if baseline is None else baseline
        delta = np.maximum(self.utility(raw) - baseline[:, None], 0)
        p = self.p.copy()
        if group is not None:
            p *= self.labels == group
            if p.sum() == 0:
                raise ValueError("empty response group")
            p /= p.sum()
        return np.einsum("mcqw,q->mcw", delta, p).mean(axis=(0, 2))

    def include(self, baseline, raw):
        if raw.shape[1] != 1:
            raise ValueError("batch inclusion requires exactly one configuration")
        return np.maximum(baseline, self.utility(raw)[:, 0])

    def descriptors(self, raw):
        diff = (self.utility(raw, epoch=True) - self.reference[:, None]).mean(axis=-1)
        return np.stack(
            [
                np.einsum("mcq,q->mc", diff, self.p * (self.labels == g))
                / self.p[self.labels == g].sum()
                for g in np.unique(self.labels)
            ],
            axis=-1,
        )

    def fitness(self, raw):
        delta = np.maximum(self.utility(raw, epoch=True) - self.reference[:, None], 0)
        return np.einsum("mcqw,q->mcw", delta, self.p).mean(axis=-1)


def behavior_regions(responses, tau, threshold=0.8):
    responses = np.asarray(responses)
    k = responses.shape[-1]
    if k == 1:
        return np.zeros(responses.shape[:-1], dtype=int)
    if tau <= 0:
        raise ValueError("positive descriptor temperature required")
    z = responses / tau
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=-1) / np.log(k)
    return np.where(entropy >= threshold, k, np.argmax(p, axis=-1))


def group_responses(signatures, weights, *, max_groups=4, min_size=25, support=None):
    """One mean clustering; member stability is assignment to fixed medoid IDs."""
    signatures, weights = np.asarray(signatures), np.asarray(weights)
    mean = signatures.mean(axis=0)
    scale = np.maximum(np.median(np.abs(mean - np.median(mean, axis=0)), axis=0), 0.1)
    allz = np.clip(signatures / scale, -3, 3)
    z = np.clip(mean / scale, -3, 3)
    dist = np.sqrt(np.mean((z[:, None] - z[None, :]) ** 2, axis=-1))
    k = max(1, min(max_groups, len(z) // min_size))
    medoids = [int(np.argmax(weights))]
    while len(medoids) < k:
        nearest = dist[:, medoids].min(axis=1)
        nearest[medoids] = -1
        new = int(np.argmax(nearest))
        if nearest[new] < 1e-10:
            break
        medoids.append(new)
    for _ in range(20):
        labels = np.argmin(dist[:, medoids], axis=1)
        small = [g for g in range(len(medoids)) if np.sum(labels == g) < min_size]
        if small and len(medoids) > 1:
            medoids.pop(small[0])
            continue
        updated = []
        for g in range(len(medoids)):
            ids = np.flatnonzero(labels == g)
            updated.append(int(ids[np.argmin(dist[np.ix_(ids, ids)] @ weights[ids])]))
        if updated == medoids:
            break
        medoids = updated
    labels = np.argmin(dist[:, medoids], axis=1)
    member_dist = np.mean((allz[:, :, None] - allz[:, None, medoids]) ** 2, axis=-1)
    probability = np.mean(np.argmin(member_dist, axis=-1) == labels[None], axis=0)
    stable = probability >= 0.8
    if support is not None:
        stable &= np.asarray(support) >= 2
    return {
        "labels": labels.tolist(),
        "medoids": medoids,
        "scale": scale.tolist(),
        "stable": stable.tolist(),
        "support": probability.tolist(),
    }


def transfer_groups(groups, utility_samples, query_weights, min_size=25):
    """Single ordered transfer pass with frozen initial group incumbents."""
    labels = np.array(groups["labels"])
    stable = np.array(groups["stable"])
    values = np.asarray(utility_samples).mean(axis=-1)  # member,config,query
    active, incumbent = [], {}
    for g in np.unique(labels):
        take = (labels == g) & stable
        if take.sum() < 10:
            continue
        p = np.asarray(query_weights) * take
        incumbent[g] = int(np.argmax(np.einsum("mcq,q->mc", values, p).mean(axis=0)))
        active.append(g)
    moved = []
    for q in range(len(labels)):
        old = labels[q]
        if not stable[q] or old not in incumbent or np.sum(labels == old) <= min_size:
            continue
        scores = []
        for new in active:
            if new == old:
                continue
            diff = values[:, incumbent[new], q] - values[:, incumbent[old], q]
            if np.mean(diff > 0.05) >= 0.8:
                scores.append((float(diff.mean()), -int(new), int(new)))
        if scores:
            new = max(scores)[2]
            labels[q] = new
            moved.append({"query_index": q, "from": int(old), "to": new})
    return {**groups, "labels": labels.tolist(), "transfers": moved}


class BehaviorArchive:
    def __init__(self, objective, measured, *, tau=None):
        self.objective = objective
        self.responses = objective.descriptors(measured)
        mean = self.responses.mean(axis=0)
        self.tau = (
            float(
                max(
                    0.1,
                    np.median(
                        np.median(
                            np.abs(mean - np.median(mean, axis=-1, keepdims=True)),
                            axis=-1,
                        )
                    ),
                )
            )
            if tau is None
            else float(tau)
        )
        self.fitness = objective.fitness(measured)
        self.regions = behavior_regions(self.responses, self.tau)
        self.elites, self.membership = {}, []
        for c in range(len(mean)):
            counts = np.bincount(self.regions[:, c], minlength=mean.shape[-1] + 1)
            region = int(np.argmax(counts))
            stable = counts[region] / len(self.regions) >= 0.8
            self.membership.append({"region": region, "stable": bool(stable)})
            if stable and (
                region not in self.elites
                or self.fitness[:, c].mean()
                > self.fitness[:, self.elites[region]].mean()
            ):
                self.elites[region] = c

    def score(self, raw):
        regions = behavior_regions(self.objective.descriptors(raw), self.tau)
        values = self.objective.fitness(raw)
        result = np.zeros_like(values)
        for region in np.unique(regions):
            elite = self.elites.get(int(region))
            ref = np.zeros(values.shape[0]) if elite is None else self.fitness[:, elite]
            result += (regions == region) * np.maximum(values - ref[:, None], 0)
        return result.mean(axis=0)

    def audit(self):
        return {
            "tau": self.tau,
            "elites": {str(r): int(c) for r, c in self.elites.items()},
            "membership": self.membership,
        }


def local_search(
    start,
    score,
    neighbors,
    key,
    avoid,
    *,
    max_steps=30,
    neutral_steps=2,
    max_evaluations=256,
):
    """Bounded neutral movement; measured starts retain their genuine score."""
    current = dict(start)
    value = float(score([current])[0])
    visited, moves, neutral = {key(current)}, [], 0
    evaluated = 1
    for _ in range(max_steps):
        candidates = [c for c in neighbors(current) if key(c) not in visited]
        candidates = sorted(candidates, key=key)[: max(0, max_evaluations - evaluated)]
        if not candidates:
            break
        evaluated += len(candidates)
        values = score(candidates)
        i = min(range(len(candidates)), key=lambda j: (-values[j], key(candidates[j])))
        new_value = float(values[i])
        if new_value < value - 1e-12 or (
            new_value <= value + 1e-12 and neutral >= neutral_steps
        ):
            break
        neutral = neutral + 1 if new_value <= value + 1e-12 else 0
        current, value = candidates[i], new_value
        visited.add(key(current))
        moves.append(key(current))
    return current, {
        "steps": len(moves),
        "score": value,
        "path": moves,
        "evaluations": evaluated,
    }
