"""Endpoint refinement and projection (spec §6, §8).

L-BFGS-B climbs the pending-aware acquisition ``ell(u)`` from one anchor inside
its trust box. The endpoint score is a diagnostic. A candidate earns a score
only after decode -> re-encode -> re-score at the canonical u. Gradients never
pass through the decoder, and a score is never fabricated.

Pre-launch fix round (Task 13 adversarial verdict): the proposal is work-based and
deterministic. Every stopping rule counts acquisition calls on ONE shared per-batch counter
(BatchBudget.calls): refinement (Endpoint.calls + 2 per anchor), projection rescoring and
per-slot pool scoring (every score request, no cache credit). Before each slot's refinement the worst-case
scoring cost of the remaining slots is reserved, so the total never exceeds the cap; when the
refinement share is used up, refinement stops (anchors "unvisited") and the scoring of the
already-built candidates still finishes inside the reserve. Wall time never changes a decision;
it only drives a safety abort (PROPOSAL_ABORT_SECONDS) that RAISES ProposalAborted, so nothing is
committed. A batch always has q picks: slots without a finite-scored legal candidate are filled
from seeded Sobol draws (source "fill", ell None); if that still gives < q, ShortBatch is raised
before the worker writes round_NNN.json. Every scored candidate must encode inside [0,1]^d (the
anchor rule); others are dropped and counted (out_of_box_dropped).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.generation.gen import gen_candidates_scipy

from .joint_search_controller import frontier_parents, recombine
from .probabilistic_plan_search import PRStart
from .latent_region import BatchBudget, TrustState, anchor_order, box

MAXITER = 50
PROPOSAL_ABORT_SECONDS = 900.0
_clock = time.monotonic  # module attribute: the tests replace it to simulate slow hosts


class ProposalAborted(Exception):
    """The proposal passed PROPOSAL_ABORT_SECONDS of wall time; nothing may be committed."""


class ShortBatch(ValueError):
    """Fewer than q legal configurations even after the Sobol fill; nothing may be committed."""


def _check_deadline(deadline) -> None:
    if deadline is not None and _clock() > deadline:
        raise ProposalAborted(f"proposal exceeded {PROPOSAL_ABORT_SECONDS:.0f} s wall time (safety abort)")


@dataclass(frozen=True)
class Endpoint:
    anchor_key: str
    anchor_source: str
    u_start: tuple
    u_end: tuple | None
    ell_start: float | None
    ell_end: float | None
    status: str  # converged | maxfun | timeout | nan | error | unvisited
    calls: int
    seconds: float


class _NonFinite(Exception):
    pass


class _CallCap(Exception):
    """The optimiser asked for one evaluation more than ``max_calls``."""


class _Counting(AcquisitionFunction):
    """Call-counting wrapper; raises on a non-finite value instead of hiding it.

    ``max_calls`` is a hard cap: scipy's L-BFGS-B ``maxfun`` is soft (the line search and
    BoTorch's closing evaluation overrun it), so the wrapper refuses evaluation number
    ``max_calls + 1`` by raising ``_CallCap``. It remembers the best point it evaluated."""

    def __init__(self, snapshot, max_calls=None, deadline=None):
        super().__init__(model=snapshot.acquisition.model)
        self.acq = snapshot.acquisition
        self.calls = 0
        self.max_calls = max_calls
        self.deadline = deadline
        self.capped = False
        self.best = None  # (ell, u) of the best evaluated point

    def forward(self, X):
        _check_deadline(self.deadline)
        if self.max_calls is not None and self.calls >= self.max_calls:
            self.capped = True
            raise _CallCap(f"acquisition call cap {self.max_calls} reached")
        self.calls += 1
        v = self.acq(X)
        if not torch.isfinite(v).all():
            raise _NonFinite("non-finite acquisition value")
        value = float(v.detach().reshape(-1)[0])
        if self.best is None or value > self.best[0]:
            self.best = (value, X.detach().reshape(-1).clone())
        return v


def _ell(acq, u) -> float:
    with torch.no_grad():
        v = acq.acq(u.reshape(1, 1, -1))
    if not torch.isfinite(v).all():
        raise _NonFinite("non-finite acquisition value")
    return float(v.reshape(-1)[0])


def refine_anchor(
    snapshot,
    u0,
    bounds,
    *,
    max_calls: int,
    seconds,
    anchor_key: str,
    anchor_source: str,
    deadline=None,
) -> Endpoint:
    """Bounded L-BFGS-B from ``u0`` inside ``bounds`` ([2, d]). ``seconds`` (L-BFGS-B
    timeout_sec) is for direct callers only; propose_latent_batch passes None (work-based).
    ``deadline`` (_clock() value): ProposalAborted propagates."""
    t0 = time.monotonic()
    u0 = torch.as_tensor(u0, dtype=torch.double).detach().clone()
    start = tuple(u0.tolist())

    def done(status, u_end=None, ell_start=None, ell_end=None):
        return Endpoint(
            anchor_key,
            anchor_source,
            start,
            None if u_end is None else tuple(u_end.tolist()),
            ell_start,
            ell_end,
            status,
            acq.calls,
            time.monotonic() - t0,
        )

    acq = _Counting(snapshot, max_calls=max_calls, deadline=deadline)
    ell_start = None
    try:
        ell_start = _ell(acq, u0)  # start and final evaluations bypass the counter (+2)
        try:
            x, _ = gen_candidates_scipy(
                u0.reshape(1, 1, -1),
                acq,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                options={"maxiter": MAXITER, "maxfun": max_calls},
                timeout_sec=seconds,
            )
            u_end = x.reshape(-1).detach().clone()
        except _CallCap:  # hard cap hit: best point evaluated so far (else the start)
            u_end = acq.best[1] if acq.best is not None else u0.clone()
        calls = acq.calls
        ell_end = _ell(acq, u_end)
        acq.calls = calls
    except _NonFinite:
        return done("nan", ell_start=ell_start)
    except ProposalAborted:
        raise
    except Exception:
        return done("error", ell_start=ell_start)
    if not (u_end.shape == u0.shape and torch.isfinite(u_end).all()):
        return done("nan", ell_start=ell_start)
    if (
        ell_end < ell_start
    ):  # kinked MC surface: keep the better of the two real evaluations
        u_end, ell_end = u0, ell_start
    if acq.capped or acq.calls >= max_calls:  # calls <= max_calls always
        status = "maxfun"
    elif seconds is not None and time.monotonic() - t0 >= seconds:
        status = "timeout"
    else:
        status = "converged"
    return done(status, u_end, ell_start, ell_end)


def _in_box(u, d) -> bool:
    return len(u) == d and all(0.0 <= float(x) <= 1.0 for x in u)


def project(codec, snapshot, endpoints) -> list[dict]:
    """Decode each usable endpoint, re-encode it, and re-score at the canonical u."""
    out = []
    for e in endpoints:
        if e.u_end is None or e.status in ("nan", "error", "unvisited"):
            continue
        try:
            u_end = torch.tensor([e.u_end], dtype=torch.double)
            config = codec.decode(u_end)[0]
            if codec.space.canonicalize(config) != config:
                raise ValueError("decoded config is not canonical")
            u_can = tuple(codec.encode(config))
            if not _in_box(u_can, codec.d):
                raise ValueError("out_of_box: canonical u outside [0,1]^d")
            ell_can = snapshot.score([config])[0]
            if not isinstance(ell_can, float) and ell_can is not None:
                ell_can = float(ell_can)
            if ell_can is None or ell_can != ell_can or abs(ell_can) == float("inf"):
                raise ValueError("non-finite canonical score")
            dist = float(torch.dist(u_end[0], torch.tensor(u_can, dtype=torch.double)))
            rt = bool(codec.roundtrip(config))
        except (ValueError, RuntimeError) as ex:
            out.append(
                {
                    "status": "rejected",
                    "reason": f"{type(ex).__name__}: {ex}",
                    "anchor_key": e.anchor_key,
                    "anchor_source": e.anchor_source,
                }
            )
            continue
        out.append(
            {
                "config": config,
                "key": codec.key(config),
                "u_end": e.u_end,
                "u_canonical": u_can,
                "ell_end": e.ell_end,
                "ell_canonical": ell_can,
                "delta": ell_can - e.ell_end,
                "distance": dist,
                "roundtrip": rt,
                "anchor_key": e.anchor_key,
                "anchor_source": e.anchor_source,
            }
        )
    return out


N_SOBOL, N_LLM, N_ANCHORS, N_LATENT, N_BACKFILL = 256, 4, 16, 25, 4
# 16 anchors (user 2026-11-04): per source at most 4, in this order; refill random Sobol to 16
ANCHOR_SOURCES = (("llm", 4), ("front", 4), ("newplan", 4), ("random", 4))
PLAN_FAMILY = ("filter_order", "graph_impl", "rank_impl")
RANDOM_SEED_OFFSET = 200000


def _front_by_hv(snapshot, configs) -> list:
    """configs ordered by exclusive hypervolume contribution of their posterior means (y_norm,
    ref = objective.ref), largest first; ties keep the input order (stable sort)."""
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    if len(configs) <= 1:
        return list(configs)
    mean = snapshot.posterior(configs).detach().double()
    ref = torch.tensor(snapshot.objective.ref, dtype=torch.double)
    hv = Hypervolume(ref_point=ref)

    def volume(rows):
        y = mean[rows]
        y = y[(y > ref).all(dim=-1)]
        return float(hv.compute(y)) if y.shape[0] else 0.0

    total = volume(list(range(len(configs))))
    contrib = [total - volume([j for j in range(len(configs)) if j != i]) for i in range(len(configs))]
    order = sorted(range(len(configs)), key=lambda i: -contrib[i])
    return [configs[i] for i in order]


def _new_plans(space, measured, seed) -> list:
    """One default config per plan family (filter_order, graph_impl, rank_impl) not yet measured,
    in a seed-shuffled deterministic order."""
    import itertools
    import random

    def family(c):
        return tuple(c[f] for f in PLAN_FAMILY)

    known = {family(space.canonicalize(c)) for c in measured}
    families = [f for f in itertools.product(*(space.domains[n] for n in PLAN_FAMILY)) if f not in known]
    random.Random(seed).shuffle(families)
    out = []
    for f in families:
        c = space.default(**dict(zip(PLAN_FAMILY, f)))
        if family(c) == f:  # canonicalization kept the family (all three fields active)
            out.append(c)
    return out


def select_latent_starts(space, snapshot, measured, *, llm, seed) -> list:
    """Anchor starts for latent refinement (user 2026-11-04; replaces the hashed select_starts):
    llm = first 4 valid LLM configs of this round; front = frontier_parents (measured configs on
    the posterior-mean front), at most 4 by largest posterior-mean HV contribution; newplan = 4
    unmeasured plan families (space.default of the family); random = space.sample(4, seed + 200000,
    avoid measured). Dedup by key in that source order. At most 16; _build_anchors refills."""
    seen, starts = set(), []
    avoid = {space.key(c) for c in measured}
    candidates = {
        "llm": list(llm),
        "front": _front_by_hv(snapshot, frontier_parents(snapshot, measured)),
        "newplan": _new_plans(space, measured, seed),
        "random": space.sample(4, seed=seed + RANDOM_SEED_OFFSET, avoid=avoid),
    }
    for source, limit in ANCHOR_SOURCES:
        accepted = 0
        for c in candidates[source]:
            if accepted >= limit:
                break
            try:
                c = space.canonicalize(c)
            except (ValueError, TypeError):
                continue
            key = space.key(c)
            if key in seen:
                continue
            seen.add(key)
            starts.append(PRStart(source, c))
            accepted += 1
    return starts


def _finite(x) -> bool:
    return x is not None and x == x and abs(x) != float("inf")


def _build_anchors(space, codec, snapshot, measured, llm, forbidden, seed, trust):
    """Starts (select_latent_starts) -> validated anchors (roundtrip, u in [0,1]^d); refill from
    Sobol (source "refill") to exactly N_ANCHORS."""
    starts = select_latent_starts(space, snapshot, measured, llm=llm, seed=seed)
    anchors, records, seen = [], [], set()

    def admit(config, source):
        key = codec.key(config)
        if key in seen:
            return
        seen.add(key)
        try:
            u = list(codec.encode(config))
            in_domain = len(u) == codec.d and all(0.0 <= x <= 1.0 for x in u)
            ok_rt = bool(codec.roundtrip(config))
        except Exception as ex:
            records.append({"key": key, "source": source, "u": None,
                            "status": f"error: {type(ex).__name__}: {ex}"})
            return
        if not in_domain:
            records.append({"key": key, "source": source, "u": u, "status": "out_of_domain"})
        elif not ok_rt:
            records.append({"key": key, "source": source, "u": u, "status": "roundtrip_failed"})
        elif len(anchors) < N_ANCHORS:
            anchors.append({"key": key, "source": source, "u": u, "config": config})
            records.append({"key": key, "source": source, "u": u, "status": "ok"})

    for s in starts:
        admit(s.config, s.source)
    if len(anchors) < N_ANCHORS:
        refill = space.sample(
            512, seed=seed + 300000 + 1000 * trust.resets, avoid=forbidden
        )
        for c in refill:
            if len(anchors) >= N_ANCHORS:
                break
            admit(c, "refill")
    return starts, anchors, records


def propose_latent_batch(
    space,
    codec,
    snapshot,
    measured: list,
    *,
    agent=(),
    seed: int,
    round_index: int,
    q: int,
    trust: TrustState,
    budget: BatchBudget,
) -> dict:
    """Four-source batch (Sobol / LLM / recombination / latent refinement).

    Candidates compete only by pending-aware ell(u(c)) of valid canonical configs;
    an endpoint score never claims a slot. Earlier picks are pending for later slots.
    """
    began = _clock()
    deadline = began + PROPOSAL_ABORT_SECONDS
    cap0 = budget.calls
    forbidden = {space.key(c) for c in measured}
    sources: dict = {}
    pool: dict = {}
    out_of_box: set = set()
    counts = {"refinement": 0, "scoring": 0}

    def in_box(config) -> bool:
        k = space.key(config)
        if k in out_of_box:
            return False
        try:
            ok = _in_box(codec.encode(config), codec.d)
        except Exception:  # noqa: BLE001 - an encoding failure is outside the domain too
            ok = False
        if not ok:
            out_of_box.add(k)
        return ok

    def merge(configs, source):
        for c in configs:
            k = space.key(c)
            if source not in sources.setdefault(k, []):
                sources[k].append(source)
            if k not in forbidden:
                pool[k] = c

    llm, rejected_agent, llm_keys = [], [], set()
    for raw in agent:  # dedup by key BEFORE the N_LLM cap
        if len(llm) >= N_LLM:
            break
        try:
            c = space.canonicalize(raw)
        except (ValueError, TypeError) as e:
            rejected_agent.append(str(e))
            continue
        if space.key(c) in llm_keys:
            continue
        llm_keys.add(space.key(c))
        llm.append(c)
    sobol = space.sample(N_SOBOL, seed=seed, avoid=forbidden)
    cross = recombine(space, frontier_parents(snapshot, measured), seed=seed + 10000)
    merge(sobol, "sobol")
    merge(llm, "llm")
    merge(cross, "recombination")

    starts, anchors, anchor_records = _build_anchors(
        space, codec, snapshot, measured, llm, forbidden, seed, trust,
    )
    merge([s.config for s in starts], "anchor")
    merge([a["config"] for a in anchors], "anchor")

    picks, selected, endpoints, projections = [], [], [], []
    fallbacks, per_slot, rejected = [], [], set()
    latent_info: dict = {}
    stopped = None

    def scoring_bound(slots_left: int) -> int:
        """Worst-case score requests of the remaining slots' projection + pool scoring."""
        per = len(pool) + N_ANCHORS + N_BACKFILL + q * N_LATENT
        return slots_left * per

    for j in range(q):
        _check_deadline(deadline)
        snap = snapshot.with_pending(picks) if picks else snapshot
        slot_t0 = _clock()
        reserve = scoring_bound(q - j)
        refine_calls = max(0, budget.calls - reserve) // (q - j)
        order = anchor_order(len(anchors), seed=seed, round_index=round_index, slot=j)
        visited, unvisited, slot_calls = 0, 0, 0
        slot_eps = []
        for n, idx in enumerate(order):
            _check_deadline(deadline)
            a = anchors[idx]
            _, calls = budget.anchor_share((None, refine_calls), len(order) - n)
            if calls < 3:  # charged = ep.calls + 2 must fit inside the share
                slot_eps.append(Endpoint(a["key"], a["source"], tuple(a["u"]), None,
                                         None, None, "unvisited", 0, 0.0))
                unvisited += 1
                continue
            u0 = torch.tensor(a["u"], dtype=torch.double)
            ep = refine_anchor(
                snap, u0, box(u0, trust.h), max_calls=calls - 2, seconds=None,
                anchor_key=a["key"], anchor_source=a["source"], deadline=deadline,
            )
            used = ep.calls + 2  # start and final evaluations are outside Endpoint.calls
            budget.charge(0.0, used)
            refine_calls -= used
            counts["refinement"] += used
            slot_calls += used
            visited += 1
            slot_eps.append(ep)
        endpoints.extend({**asdict(e), "slot": j} for e in slot_eps)
        _check_deadline(deadline)
        slot_proj = [p for p in project(codec, snap, slot_eps)]
        for p in slot_proj:
            if p.get("status") == "rejected" and str(p.get("reason", "")).startswith("ValueError: out_of_box"):
                out_of_box.add(f"projection:{j}:{p['anchor_key']}")
        projections.extend({**{k: v for k, v in p.items() if k != "config"}, "slot": j}
                           for p in slot_proj)
        taken = set(forbidden) | {c["key"] for c in selected} | rejected
        fresh = {}
        for p in sorted((p for p in slot_proj if "config" in p),
                        key=lambda p: (-p["ell_canonical"], p["key"])):
            k = p["key"]
            if k in taken or k in fresh:
                continue
            fresh[k] = p
            if len(fresh) >= N_LATENT:
                break
        merge([p["config"] for p in fresh.values()], "latent")
        for k, p in fresh.items():
            latent_info.setdefault(k, {kk: vv for kk, vv in p.items() if kk != "config"})

        def legal_configs():
            return [c for k, c in sorted(pool.items()) if k not in taken and in_box(c)]

        legal = legal_configs()
        if not legal:
            drawn = space.sample(N_BACKFILL, seed=seed + 400000 + j, avoid=taken,
                                 max_draws=4096)
            if drawn:
                merge(drawn, "backfill")
                fallbacks.append({"slot": j, "reason": "legal pool empty",
                                  "keys": [space.key(c) for c in drawn]})
                legal = legal_configs()
        _check_deadline(deadline)
        scores = snap.score(legal) if legal else []
        # every score request counts (no cache credit): deterministic, an upper bound on forwards
        proj_scored = sum(1 for p in slot_proj if "ell_canonical" in p
                          or "non-finite canonical score" in str(p.get("reason", "")))
        scored = proj_scored + len(legal)
        budget.charge(0.0, scored)
        counts["scoring"] += scored
        slot_calls += scored
        per_slot.append({"slot": j, "visited": visited, "unvisited": unvisited,
                         "calls": slot_calls, "scoring_calls": scored, "reserve": reserve,
                         "seconds": _clock() - slot_t0, "legal": len(legal)})
        bad = [space.key(c) for c, s in zip(legal, scores) if not _finite(s)]
        if bad:
            rejected.update(bad)
            keep = [(c, s) for c, s in zip(legal, scores) if _finite(s)]
            legal, scores = [c for c, _ in keep], [s for _, s in keep]
        if not legal:
            stopped = {"slot": j, "reason": "no finite-scored legal candidate; Sobol fill"}
            fallbacks.append(stopped)
            break
        i = max(range(len(legal)), key=lambda t: (scores[t], space.key(legal[t])))
        config, key = legal[i], space.key(legal[i])
        picks.append(config)
        selected.append({
            "config": config, "key": key, "ell": scores[i], "snapshot": snap.identity,
            "sources": sources[key], "plan": space.plan(config).describe(),
            "latent": latent_info.get(key),
        })
    if len(selected) < q:  # deterministic fill: seeded Sobol, unscored (never commit < q)
        taken = set(forbidden) | {c["key"] for c in selected}
        snap = snapshot.with_pending(picks) if picks else snapshot
        for c in space.sample(4 * q, seed=seed + 500000, avoid=taken, max_draws=4096):
            if len(selected) >= q:
                break
            k = space.key(c)
            if k in taken or not in_box(c):
                continue
            taken.add(k)
            picks.append(c)
            selected.append({"config": c, "key": k, "ell": None, "snapshot": snap.identity,
                             "sources": ["fill"], "plan": space.plan(c).describe(),
                             "latent": None})
            fallbacks.append({"slot": len(selected) - 1, "reason": "fill", "key": k})
    if len(selected) < q:
        raise ShortBatch(f"only {len(selected)} legal configurations for q = {q} after the Sobol fill")
    _check_deadline(deadline)
    return {
        "selected": selected,
        "starts": [{"source": s.source, "config": s.config} for s in starts],
        "anchors": [{k: v for k, v in r.items()} for r in anchor_records],
        "anchor_sources": {s: sum(a["source"] == s for a in anchors)
                           for s in dict.fromkeys(a["source"] for a in anchors)},
        "endpoints": endpoints,
        "projections": projections,
        "pool_count": len(pool),
        "provenance": sources,
        "rejected_agent": rejected_agent,
        "fallbacks": fallbacks,
        "stopped": stopped,
        "budget": {
            "seconds": None,  # work-based: wall time never stops refinement (module docstring)
            "calls": cap0 - budget.calls,
            "cap": cap0,
            "refinement_calls": counts["refinement"],
            "scoring_calls": counts["scoring"],
            "abort_seconds": PROPOSAL_ABORT_SECONDS,
            # True only when the cap is below the scoring reserve: refinement got 0 calls and the
            # scoring of the already-built candidates finished past the cap (documented policy)
            "over_cap": cap0 - budget.calls > cap0,
            "per_slot": per_slot,
        },
        "out_of_box_dropped": len(out_of_box),
        "trust": asdict(trust),
        "codec": codec.identity,
        "seconds": _clock() - began,
        "seed": seed,
        "q": q,
    }
