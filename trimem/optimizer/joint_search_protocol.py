"""Staged initialization and measured-NHV escape policy for new TRACE runs.

Kept outside the source-frozen v0.4.2 modules used by the existing campaign.
"""

import math

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

from .joint_search_controller import (
    frontier_parents,
    greedy_batch,
    propose_batch,
    recombine,
)
from .plan_codec import JointCodec
from .plan_surrogate import AcquisitionSnapshot

SCHEDULE = {
    "mode": "sobol-agent-v1",
    "sobol_count": 5,
    "agent_count": 5,
    "stagnation_patience": 3,
    "stagnation_epsilon": 1e-6,
    "forced_exploration": "uncertainty-novelty-v1",
}


class EscapeSnapshot(AcquisitionSnapshot):
    def posterior_stats(self, configs):
        x = torch.tensor([self.codec.encode(c) for c in configs], dtype=torch.double)
        with torch.no_grad():
            posterior = self.model.posterior(x)
            return posterior.mean, posterior.variance


def stagnation_status(initial_nhv, events, *, patience=3, epsilon=1e-6):
    if patience < 1 or epsilon < 0:
        raise ValueError("Invalid stagnation threshold")
    best, count = initial_nhv, 0
    for event in events:
        value = event["nhv"]
        if not math.isfinite(value) or event["kind"] not in ("ordinary", "forced"):
            raise ValueError("Invalid completed search event")
        improved = value > best + epsilon
        if improved or event["kind"] == "forced":
            count = 0
        else:
            count += 1
        best = max(best, value)
    return {
        "force_next": count >= patience,
        "stagnant_configs": count,
        "best_nhv": best,
        "patience": patience,
        "epsilon": epsilon,
    }


def normalized_hv(objective, observations):
    y, _ = objective.tensors(observations)
    reference = torch.tensor(objective.ref, dtype=torch.double)
    return Hypervolume(reference).compute(y) / math.prod(1 - r for r in objective.ref)


def select_with_escape(
    space, snapshot, candidates, escape_candidates, measured, *, q=4, forbidden=()
):
    """One uncertainty/novelty slot, then pending-aware acquisition slots."""
    forbidden = set(forbidden) | {space.key(c) for c in measured}
    forbidden |= {space.key(c) for c in getattr(snapshot, "pending", ())}
    pool = {
        space.key(c): space.canonicalize(c)
        for c in candidates
        if space.key(c) not in forbidden
    }
    if q < 1 or not pool:
        return []
    preferred = {
        space.key(c): pool[space.key(c)]
        for c in escape_candidates
        if space.key(c) in pool
    }
    eligible = list((preferred or pool).values())
    _, variance = snapshot.posterior_stats(eligible)
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("Nonfinite/negative posterior uncertainty")
    uncertainty = 0.5 * variance.clamp_min(1e-18).log().sum(dim=-1)
    codec = JointCodec(space)
    observed = [codec.encode(c) for c in measured]

    def novelty(c):
        x = codec.encode(c)
        return min(
            (
                sum(
                    float(a != b) if i in codec.cat_dims else (a - b) ** 2
                    for i, (a, b) in enumerate(zip(x, y))
                )
                for y in observed
            ),
            default=1.0,
        )

    idx = max(
        range(len(eligible)),
        key=lambda i: (
            float(uncertainty[i]),
            novelty(eligible[i]),
            space.key(eligible[i]),
        ),
    )
    chosen = eligible[idx]
    key = space.key(chosen)
    first = {
        "config": chosen,
        "key": key,
        "ell": snapshot.score([chosen])[0],
        "snapshot": snapshot.identity,
        "selection_rule": "forced_uncertainty_novelty",
        "uncertainty_log_volume": float(uncertainty[idx]),
        "exploration_pool": "agent_recombination" if preferred else "whole_pool",
    }
    current = snapshot.with_pending([*getattr(snapshot, "pending", ()), chosen])
    rest = greedy_batch(
        space, current, list(pool.values()), q=q - 1, forbidden=forbidden | {key}
    )
    for item in rest:
        item["selection_rule"] = "qLogNEHVI"
    return [first, *rest]


def propose_with_escape(
    space, snapshot, measured, *, force_exploration=False, **kwargs
):
    """Reuse frozen four-channel generator; reconstruct its pool only for escape.

    The same seeds and canonical deduplication recover exactly the pool whose
    source provenance the base generator records. No additional profiling slots.
    """
    q = kwargs.get("q", 4)
    if q not in (1, 2, 3, 4):
        raise ValueError("Final batch size must be in 1..4")
    # Base generation needs only one ordinary selection when escape replaces it.
    # Its provenance retains the entire pool; pending-aware selection follows.
    generation_q = 1 if force_exploration else (4 if q == 3 else q)
    result = propose_batch(space, snapshot, measured, **{**kwargs, "q": generation_q})
    result["selected"] = result["selected"][:q]
    result["q"] = q
    if force_exploration:
        seed, q = kwargs.get("seed", 0), kwargs.get("q", 4)
        forbidden = {space.key(c) for c in [*measured, *kwargs.get("pending", ())]}
        agent = []
        for raw in list(kwargs.get("agent", ()))[:4]:
            try:
                agent.append(space.canonicalize(raw))
            except (ValueError, TypeError):
                pass
        cross = recombine(
            space, frontier_parents(snapshot, measured), seed=seed + 10000
        )
        candidates = [
            *space.sample(256, seed=seed, avoid=forbidden),
            *agent,
            *cross,
            *result["pr"]["candidates"],
            *[s["config"] for s in result["starts"]],
        ]
        actual = {space.key(c) for c in candidates} - forbidden
        expected = set(result["provenance"]) - forbidden
        if actual != expected:
            raise ValueError("Forced exploration pool/provenance mismatch")
        result["selected"] = select_with_escape(
            space,
            snapshot,
            candidates,
            [*agent, *cross],
            measured,
            q=q,
            forbidden=forbidden,
        )
        for item in result["selected"]:
            item["sources"] = result["provenance"][item["key"]]
            item["plan"] = space.plan(item["config"]).describe()
    else:
        for item in result["selected"]:
            item["selection_rule"] = "qLogNEHVI"
    result["forced_exploration"] = force_exploration
    result["unique_acquisition_inputs"] = len(snapshot.budget.keys)
    return result
