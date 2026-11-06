"""TRACE v0.4.2 four-source candidate pool and sequential pending-aware q=4."""

import random
import time

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

from .plan_codec import JointCodec
from .probabilistic_plan_search import ProposalBudget, PRStart, refine


def initial_configs(space, *, seed=0, count=10):
    defaults = [space.default(filter_order=p) for p in space.domains["filter_order"]]
    if count < len(defaults):
        raise ValueError("Initialization cannot cover the enabled plan structures")
    pool = {space.key(c): c for c in space.sample(1024, seed=seed)}
    selected = {space.key(c): c for c in defaults}
    codec = JointCodec(space)

    # Categorical mismatches, plus squared distances in scaled ordinal values.
    # Category integer ids are labels, not ordinal distances.
    def distance(a, b):
        return sum(
            float(x != y) if i in codec.cat_dims else (x - y) ** 2
            for i, (x, y) in enumerate(zip(a, b))
        )

    encoded = {k: codec.encode(c) for k, c in pool.items()}
    while len(selected) < count:
        chosen = [codec.encode(c) for c in selected.values()]
        available = [k for k in pool if k not in selected]
        if not available:
            raise ValueError("Not enough unique initialization configurations")
        key = max(
            available, key=lambda k: (min(distance(encoded[k], x) for x in chosen), k)
        )
        selected[key] = pool[key]
    return list(selected.values())


def frontier_parents(snapshot, measured):
    if not measured:
        return []
    mean = snapshot.posterior(measured)
    ids = [int(mean[:, 0].argmax()), int((0.5 * (mean[:, 1] + mean[:, 2])).argmax())]
    frontier = torch.where(is_non_dominated(mean))[0].tolist()
    frontier.sort(key=lambda j: float(mean[j, 0]))
    if frontier:
        ids += [frontier[0], frontier[len(frontier) // 2], frontier[-1]]
    return [measured[i] for i in dict.fromkeys(ids)]


BLOCKS = (
    (
        "filter_order",
        "filter_pre_impl",
        "filter_post_impl",
        "sem_threshold_pre",
        "sem_threshold_post",
    ),
    (
        "graph_impl",
        "depth",
        "fanout",
        "edge_threshold",
        "beam_width",
        "graph_direction",
    ),
    ("seed_count",),
    ("rank_impl", "returned_k"),
    ("injection_timing", "retrieval_interval", "retrieval_unit"),
    ("trajectory_category",),
)


def recombine(space, parents, *, seed=0, count=64):
    rng = random.Random(seed)
    seen = {}
    if len(parents) < 2:
        return []
    for _ in range(count * 8):
        a, b = rng.sample(parents, 2)
        c = dict(a)
        for block in rng.sample(BLOCKS, rng.choice((1, 2))):
            c.update({n: b[n] for n in block})
        c = space.canonicalize(c)
        seen.setdefault(space.key(c), c)
        if len(seen) == count:
            break
    return list(seen.values())


def select_starts(space, snapshot, measured, pool, *, agent=(), cross=(), seed=0):
    seen, starts = set(), []

    def add(source, configs, limit):
        accepted = 0
        for c in configs:
            c = space.canonicalize(c)
            key = space.key(c)
            if key not in seen:
                seen.add(key)
                starts.append(PRStart(source, c))
                accepted += 1
            if accepted >= limit:
                break

    if agent:
        add("pr-agent", agent, 4)
    elif cross:
        scores = snapshot.score(cross)
        add(
            "pr-crossover",
            [c for _, c in sorted(zip(scores, cross), key=lambda r: -r[0])],
            4,
        )
    add("pr-incumbent", frontier_parents(snapshot, measured), 5)
    known = {c["filter_order"] for c in measured}
    add(
        "pr-newplan",
        [
            space.default(filter_order=p)
            for p in space.domains["filter_order"]
            if p not in known
        ],
        3,
    )
    add("pr-random", space.sample(8, seed=seed + 200000), 8)
    if pool:
        scores = snapshot.score(pool)
        add(
            "pr-pool", [c for _, c in sorted(zip(scores, pool), key=lambda r: -r[0])], 8
        )
    return starts[:25]


def greedy_batch(space, snapshot, candidates, *, q=4, forbidden=()):
    pool = {
        space.key(c): space.canonicalize(c)
        for c in candidates
        if space.key(c) not in set(forbidden)
    }
    selected = []
    for _ in range(min(q, len(pool))):
        current = (
            snapshot.with_pending(
                [*getattr(snapshot, "pending", ()), *[r["config"] for r in selected]]
            )
            if selected
            else snapshot
        )
        configs = list(pool.values())
        scores = current.score(configs)
        index = max(
            range(len(configs)), key=lambda i: (scores[i], space.key(configs[i]))
        )
        config = configs[index]
        key = space.key(config)
        selected.append(
            {
                "config": config,
                "key": key,
                "ell": scores[index],
                "snapshot": current.identity,
            }
        )
        del pool[key]
    return selected


CHANNELS = ("llm", "recombination", "pr", "sobol")


def channel_quota_batch(space, snapshot, channels, *, q=4, forbidden=(), fill=()):
    """User 2026-11-03: each round's q=4 configurations come one from each channel,
    each the channel's highest-qLogNEHVI candidate. Selection stays sequential with
    pending: at every step the channel whose best remaining candidate scores highest
    is served first, then that pick becomes pending for the remaining channels. A
    channel with no admissible candidate is filled from ``fill`` (the whole pool) and
    recorded as such."""
    forbidden = set(forbidden)
    pools = {
        name: {space.key(c): space.canonicalize(c) for c in channels.get(name, ())
               if space.key(c) not in forbidden}
        for name in CHANNELS
    }
    fill_pool = {space.key(c): space.canonicalize(c) for c in fill if space.key(c) not in forbidden}
    selected, taken = [], set()
    remaining = [n for n in CHANNELS][:q]
    while remaining:
        current = (
            snapshot.with_pending([*getattr(snapshot, "pending", ()), *[r["config"] for r in selected]])
            if selected else snapshot
        )
        best = {}
        for name in remaining:
            cands = [c for k, c in pools[name].items() if k not in taken]
            if not cands:
                continue
            scores = current.score(cands)
            i = max(range(len(cands)), key=lambda j: (scores[j], space.key(cands[j])))
            best[name] = (scores[i], cands[i])
        if not best:  # every remaining channel is empty: fill from the whole pool
            cands = [c for k, c in fill_pool.items() if k not in taken]
            if not cands:
                break
            scores = current.score(cands)
            i = max(range(len(cands)), key=lambda j: (scores[j], space.key(cands[j])))
            name = remaining[0]
            score, config, filled = scores[i], cands[i], True
        else:
            name = max(best, key=lambda n: (best[n][0], n))
            (score, config), filled = best[name], False
        key = space.key(config)
        selected.append({"config": config, "key": key, "ell": score, "snapshot": current.identity,
                         "channel": name, "channel_fill": filled})
        taken.add(key)
        remaining.remove(name)
    return selected


def neighborhood_diagnostic(space, snapshot, configs, forbidden, *, limit=8):
    """D3 (logging only, after selection): neighborhood comparison around PR's
    best configs. Moves are classified by the number of fields whose CANONICAL
    values differ (activation changes count). For every 2-field move whose two
    single-field parts exist and which improves on the base and on every
    single-field move, ``interaction = g(ab) - g(a) - g(b)``; a positive value
    is gain an additive (factorized) view of the fields cannot explain.
    """
    import itertools

    names = [n for n in space.domains if len(space.domains[n]) > 1]
    rows = []
    for c in configs[:limit]:
        c = space.canonicalize(c)
        base_key = space.key(c)
        moves = {}
        for k in (1, 2):
            for fields in itertools.combinations(names, k):
                for values in itertools.product(*(space.domains[f] for f in fields)):
                    try:
                        d = space.canonicalize({**c, **dict(zip(fields, values))})
                    except ValueError:
                        continue
                    changed = tuple(f for f in space.domains if d[f] != c[f])
                    key = space.key(d)
                    if len(changed) in (1, 2) and key not in forbidden:
                        moves.setdefault(key, (d, changed))
        try:
            keys = list(moves)
            scores = snapshot.score([c] + [moves[x][0] for x in keys])
        except RuntimeError as exc:
            if "budget" not in str(exc):
                raise
            rows.append({"key": base_key, "skipped": str(exc)})
            break
        base, gain = scores[0], {x: v - scores[0] for x, v in zip(keys, scores[1:])}
        single = {
            (moves[x][1][0], moves[x][0][moves[x][1][0]]): gain[x]
            for x in keys
            if len(moves[x][1]) == 1
        }
        best1 = max((gain[x] for x in keys if len(moves[x][1]) == 1), default=None)
        best2 = max(
            ((gain[x], x) for x in keys if len(moves[x][1]) == 2), default=(None, None)
        )
        inter = []
        floor = max(0.0, best1 if best1 is not None else 0.0)
        for x in keys:
            d, changed = moves[x]
            # Only improving 2-field moves that beat every single-field move
            # matter (ell is a log acquisition; bad pairs are not informative).
            if len(changed) == 2 and gain[x] > floor:
                parts = [(f, d[f]) for f in changed]
                if all(p in single for p in parts):
                    inter.append((gain[x] - sum(single[p] for p in parts), list(changed)))
        top = max(inter, default=(None, None))
        rows.append(
            {
                "key": base_key,
                "ell": base,
                "best_1field_gain": best1,
                "best_2field_gain": best2[0],
                "best_2field_fields": list(moves[best2[1]][1]) if best2[1] else None,
                "max_interaction": top[0],
                "max_interaction_fields": top[1],
                "moves": {
                    k: sum(len(m[1]) == k for m in moves.values()) for k in (1, 2)
                },
            }
        )
    return rows


def propose_batch(
    space,
    snapshot,
    measured,
    *,
    agent=(),
    pending=(),
    seed=0,
    q=4,
    steps=200,
    samples=128,
    pr_seconds=None,
    selection="greedy",
):
    if q not in (1, 2, 4):
        raise ValueError("q must be recorded as 1, 2 or 4")
    began = time.monotonic()
    snapshot.budget = snapshot.budget or ProposalBudget()
    measured_keys = {space.key(c) for c in measured}
    forbidden = measured_keys | {space.key(c) for c in pending}
    usable = []
    rejected = []
    for raw in list(agent)[:4]:
        try:
            usable.append(space.canonicalize(raw))
        except (ValueError, TypeError) as e:
            rejected.append(str(e))
    sources = {}
    pool = {}

    def merge(configs, source):
        for c in configs:
            key = space.key(c)
            sources.setdefault(key, [])
            if source not in sources[key]:
                sources[key].append(source)
            if key not in forbidden:
                pool[key] = c

    sobol = space.sample(256, seed=seed, avoid=forbidden)
    cross = recombine(space, frontier_parents(snapshot, measured), seed=seed + 10000)
    merge(sobol, "sobol")
    merge(usable, "agent")
    merge(cross, "recombination")
    starts = select_starts(
        space,
        snapshot,
        measured,
        list(pool.values()),
        agent=usable,
        cross=cross,
        seed=seed,
    )
    pr = refine(
        space,
        starts,
        snapshot,
        seed=seed + 30000,
        steps=steps,
        samples=samples,
        forbidden=forbidden,
        seconds=pr_seconds,
    )
    merge(pr["candidates"], "pr")

    for start in starts:
        merge([start.config], start.source)
    if selection == "channel-quota-v1":
        selected = channel_quota_batch(
            space, snapshot,
            {"llm": usable, "recombination": cross, "pr": pr["candidates"], "sobol": sobol},
            q=q, forbidden=forbidden, fill=list(pool.values()),
        )
    else:
        selected = greedy_batch(
            space, snapshot, list(pool.values()), q=q, forbidden=forbidden
        )
    # D3 runs only after the batch is fixed, so it can never change selection.
    try:
        ranked = sorted(
            zip(snapshot.score(pr["candidates"]), range(len(pr["candidates"]))),
            reverse=True,
        ) if pr["candidates"] else []
        pr["neighborhood"] = neighborhood_diagnostic(
            space, snapshot, [pr["candidates"][i] for _, i in ranked], forbidden
        )
    except RuntimeError as exc:
        if "budget" not in str(exc):
            raise
        pr["neighborhood"] = [{"skipped": str(exc)}]
    for item in selected:
        item["sources"] = sources[item["key"]]
        item["plan"] = space.plan(item["config"]).describe()
    return {
        "selected": selected,
        "starts": [{"source": s.source, "config": s.config} for s in starts],
        "pr": pr,
        "pool_count": len(pool),
        "provenance": sources,
        "rejected_agent": rejected,
        "unique_acquisition_inputs": len(snapshot.budget.keys),
        "seconds": time.monotonic() - began,
        "seed": seed,
        "q": q,
    }
