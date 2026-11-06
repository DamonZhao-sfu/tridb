"""TRACE v0.7.1 search worker: RAG-Stack stall rule on top of the v0.7.0 latent worker (user 2026-11-05).

Stall rule (``manifest["v071"]["stall_rule"]``): after ``patience`` (3) consecutive completed rounds without a TRAIN
hypervolume improvement greater than ``epsilon`` (1e-6, ``joint_search_protocol.SCHEDULE``), the next round is FORCED.

* Round HV: normalized HV (``measured_hv`` / prod(1 - ref), frozen objective lo / span / ref) of the complete
  never-pruned configurations of the archive and picks of that round (each on its own sample) plus complete NoMemory
  TRAIN. A round whose picks are all pruned or dominated cannot raise it, so it counts as stalled.
* Counter: ``joint_search_protocol.stagnation_status`` (a forced round resets it).
* Determinism: ``stall_state`` is a pure function of the ledger entries, the prior round files and the frozen
  objective. The flag is recorded in the round file (``force_exploration``, ``stagnant_rounds``, ``round_hv``,
  ``selection_rule``); the driver replays a committed round file without spawning a worker. ``hv_rounds.jsonl`` is
  regenerated atomically from the same function.
* Forced round: all q picks follow ``uncertainty-novelty-v1`` sequentially (max posterior log-volume, then novelty,
  then key) over the legal pool of the latent proposal WITHOUT its acquisition-driven parts (Sobol, LLM, recombination,
  anchors, ``in_box``; no refinement, no qLogNEHVI scoring). ``snapshot.with_pending`` does not change the posterior
  variance (pending enters only the acquisition), so each pick's snapshot identity comes from ``with_pending`` and the
  log-volume is conditioned on earlier picks through the GP joint covariance (Schur complement,
  ``conditioning = "schur"``; a model without joint covariance falls back to marginal variance, recorded).

The stock worker chain is unchanged: ``latent_search_worker_v071`` -> ``latent_search_worker_v070`` ->
``trace_latent_search.latent_search_worker``, with only ``trace_latent_search.propose_for_round`` wrapped in the child.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.experiments.trace_v070_runtime import (
    latent_search_worker_v070 as _stock_worker,
)
from trimem.experiments.trace_v070_runtime import v070_patches
from trimem.optimizer.joint_search_protocol import SCHEDULE, stagnation_status
from trimem.optimizer.plan_schema import MASK_FIELDS
from trimem.optimizer.plan_schema_v070 import FIELDS

PROTOCOL = "rag-stack-stall-v1"
RULE = {"protocol": PROTOCOL, "patience": 3, "epsilon": SCHEDULE["stagnation_epsilon"],
        "forced": SCHEDULE["forced_exploration"]}
FORCED_RULE = "forced_uncertainty_novelty"
GREEDY_RULE = "qLogNEHVI"
LOG2 = {"seed_count", "fanout", "beam_width"}


def stall_rule(manifest) -> dict | None:
    return (manifest.get("v071") or {}).get("stall_rule")


def _round_files(directory, before: str | None) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(Path(directory).glob("round_[0-9][0-9][0-9].json")):
        if before is None or p.stem < before:
            out.append((p.stem, json.loads(p.read_text())))
    return out


def _eligible(manifest, space, entries) -> dict:
    """key -> observation of complete never-pruned configurations on their own samples."""
    size = int(manifest["v070"]["sampling"]["size"])
    return {e["key"]: driver.aggregate_panel(space, e["config"], e["receipts"], task_ids=manifest["tasks"],
                                             protocol=manifest["protocol"],
                                             allow_synthetic=manifest.get("smoke", False))
            for e in entries if not e.get("pruned") and len(e["receipts"]) == size}


def _nomemory_obs(directory, manifest):
    from trimem.experiments.trace_v070_baseline import (
        baseline_entry,
        baseline_space,
        load_baseline,
    )
    from trimem.optimizer.plan_surrogate import aggregate_panel

    e = baseline_entry(directory, "TRAIN")
    if e is None or not e["complete"]:
        raise ValueError("the stall rule needs the complete NoMemory TRAIN ledger")
    return aggregate_panel(baseline_space(manifest), e["config"], e["receipts"], task_ids=e["tasks"],
                           protocol=load_baseline(directory, "TRAIN")["protocol"],
                           allow_synthetic=manifest.get("smoke", False))


def _hv(observations, objective) -> float:
    from trimem.optimizer.latent_plan_surrogate import measured_hv

    return measured_hv(observations, objective) / math.prod(1 - r for r in objective.ref)


def stall_state(directory, manifest, entries, objective, *, before: str | None = None) -> dict:
    """{force_exploration, stagnant_rounds, round_hv, best_hv, rounds: [...]} at the start of round ``before``."""
    rule = stall_rule(manifest)
    space = driver.get_space(manifest)
    obs, nm = _eligible(manifest, space, entries), _nomemory_obs(directory, manifest)
    init = [obs[k] for k in manifest["initial_keys"] if k in obs]
    initial = _hv([*init, nm], objective)
    events, rows = [], []
    for name, body in _round_files(directory, before):
        picks = [s["key"] for s in body["selected"]]
        members = dict.fromkeys([*body["archive_keys"], *picks])
        hv = _hv([*[obs[k] for k in members if k in obs], nm], objective)
        events.append({"nhv": hv, "kind": "forced" if body.get("force_exploration") else "ordinary"})
        st = stagnation_status(initial, events, patience=rule["patience"], epsilon=rule["epsilon"])
        rows.append({"round": name, "hv": hv, "kind": events[-1]["kind"], "stagnant_rounds": st["stagnant_configs"],
                     "force_next": st["force_next"], "best_hv": st["best_nhv"],
                     "picks_full_never_pruned": sum(k in obs for k in picks), "picks": len(picks)})
    st = stagnation_status(initial, events, patience=rule["patience"], epsilon=rule["epsilon"])
    return {"force_exploration": st["force_next"], "stagnant_rounds": st["stagnant_configs"],
            "round_hv": events[-1]["nhv"] if events else initial, "best_hv": st["best_nhv"], "initial_hv": initial,
            "patience": rule["patience"], "epsilon": rule["epsilon"], "rounds": rows}


def write_hv_rounds(directory, state: dict) -> Path:
    """<run>/hv_rounds.jsonl from a ``stall_state``: the init row + one row per round (regenerated atomically)."""
    init = {"round": "init", "hv": state["initial_hv"], "kind": "init", "stagnant_rounds": 0, "force_next": False,
            "best_hv": state["initial_hv"]}
    lines = [{**x, "includes_nomemory": True} for x in (init, *state["rounds"])]
    path = Path(directory) / "hv_rounds.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(x) + "\n" for x in lines))
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------ uncertainty-novelty-v1 (latent proposal pool)


def plan_vector(space, config) -> list[float]:
    """``plan_codec.JointCodec.encode`` on the v0.7.0 FIELDS (the stock codec's FIELDS have no rank "none")."""
    c = space.canonicalize(config)
    masks = space.active_mask(c)
    out = []
    for name, kind, full in FIELDS:
        if c[name] is None:
            out.append(0.0)
        elif kind == "cat":
            out.append(float(full.index(c[name])))
        else:
            raw = [math.log2(v) if name in LOG2 else float(v) for v in full]
            out.append((raw[full.index(c[name])] - raw[0]) / (raw[-1] - raw[0]))
    return out + [float(masks[n]) for n in MASK_FIELDS]


CAT_DIMS = {i for i, (_, kind, _) in enumerate(FIELDS) if kind == "cat"} | set(range(len(FIELDS), len(FIELDS) + 6))


def novelty(space, config, against: list) -> float:
    x = plan_vector(space, config)
    return min((sum(float(a != b) if i in CAT_DIMS else (a - b) ** 2 for i, (a, b) in enumerate(zip(x, y)))
                for y in (plan_vector(space, c) for c in against)), default=1.0)


def conditional_log_volume(snapshot, configs, pending) -> tuple[torch.Tensor, str]:
    """0.5 * sum log Var[f_m(x) | pending picks]; marginal variance when no joint covariance is available."""
    _, var = snapshot.posterior_moments(configs)
    if not torch.isfinite(var).all() or (var < 0).any():
        raise ValueError("Nonfinite/negative posterior uncertainty")
    marginal = 0.5 * var.clamp_min(1e-18).log().sum(dim=-1)
    if not pending:
        return marginal, "marginal"
    try:
        x = torch.tensor([snapshot.codec.encode(c) for c in [*pending, *configs]], dtype=torch.double)
        p, parts = len(pending), []
        with torch.no_grad():
            for m in snapshot.model.models:
                cov = m.posterior(x).distribution.covariance_matrix
                a = cov[:p, :p] + 1e-9 * torch.eye(p, dtype=cov.dtype)
                b = cov[p:, :p]
                parts.append(cov.diagonal()[p:] - (b * torch.linalg.solve(a, b.T).T).sum(dim=-1))
        return 0.5 * torch.stack(parts, dim=-1).clamp_min(1e-18).log().sum(dim=-1), "schur"
    except (AttributeError, RuntimeError, NotImplementedError):
        return marginal, "marginal-novelty-only"


def forced_picks(space, snapshot, legal, measured, q, sources) -> list[dict]:
    picks, selected = [], []
    remaining = list(legal)
    for j in range(q):
        snap = snapshot.with_pending(picks) if picks else snapshot
        logvol, how = conditional_log_volume(snapshot, remaining, picks)
        against = [*measured, *picks]
        i = max(range(len(remaining)),
                key=lambda t: (float(logvol[t]), novelty(space, remaining[t], against), space.key(remaining[t])))
        c = remaining.pop(i)
        key = space.key(c)
        selected.append({"config": c, "key": key, "ell": None, "snapshot": snap.identity,
                         "sources": sources.get(key, []), "plan": space.plan(c).describe(), "latent": None,
                         "selection_rule": FORCED_RULE, "uncertainty_log_volume": float(logvol[i]),
                         "novelty": novelty(space, c, against), "conditioning": how, "slot": j})
        picks.append(c)
    return selected


def legal_pool(space, codec, snapshot, measured, agent, seed, trust) -> tuple[list, dict, list, list, list]:
    """The latent proposal's pool without refinement: Sobol, LLM, recombination, anchors; ``in_box`` only."""
    from trimem.optimizer import latent_plan_search as lps
    from trimem.optimizer.joint_search_controller import frontier_parents, recombine

    forbidden = {space.key(c) for c in measured}
    llm, rejected = [], []
    for raw in agent:
        try:
            c = space.canonicalize(raw)
        except (ValueError, TypeError) as exc:
            rejected.append(str(exc))
            continue
        if space.key(c) not in {space.key(x) for x in llm} and len(llm) < lps.N_LLM:
            llm.append(c)
    starts, anchors, records = lps._build_anchors(space, codec, snapshot, measured, llm, forbidden, seed, trust)
    sources: dict = {}
    pool: dict = {}
    for configs, source in ((space.sample(lps.N_SOBOL, seed=seed, avoid=forbidden), "sobol"), (llm, "llm"),
                            (recombine(space, frontier_parents(snapshot, measured), seed=seed + 10000), "recombination"),
                            ([s.config for s in starts], "anchor"), ([a["config"] for a in anchors], "anchor")):
        for c in configs:
            k = space.key(c)
            if source not in sources.setdefault(k, []):
                sources[k].append(source)
            if k not in forbidden:
                pool[k] = c

    def in_box(c) -> bool:
        try:
            return lps._in_box(codec.encode(c), codec.d)
        except Exception:  # noqa: BLE001 - an encoding failure is outside the domain too
            return False

    legal = [c for k, c in sorted(pool.items()) if in_box(c)]
    return legal, sources, starts, records, rejected


def forced_proposal(directory, round_file, manifest, space, codec, snapshot, obs, objective, agent, seed):
    """Same return shape as ``trace_latent_search.propose_for_round``; no acquisition call."""
    from trimem.optimizer.latent_plan_search import ShortBatch
    from trimem.optimizer.latent_region import batch_gains, replay

    t0 = time.perf_counter()
    gains = batch_gains(tls._prior_rounds(directory, round_file), obs, objective, key=lambda o: space.key(o.config))
    trust = replay(gains)
    q = manifest["q"]
    measured = [o.config for o in obs]
    legal, sources, starts, records, rejected = legal_pool(space, codec, snapshot, measured, agent, seed, trust)
    if len(legal) < q:
        raise ShortBatch(f"only {len(legal)} legal configurations for a forced round of q = {q}")
    selected = forced_picks(space, snapshot, legal, measured, q, sources)
    result = {"selected": selected, "starts": [{"source": s.source, "config": s.config} for s in starts],
              "anchors": records, "anchor_sources": {}, "endpoints": [], "projections": [], "pool_count": len(legal),
              "provenance": sources, "rejected_agent": rejected, "fallbacks": [], "stopped": None,
              "budget": {"seconds": None, "calls": 0, "cap": 0, "refinement_calls": 0, "scoring_calls": 0,
                         "abort_seconds": None, "over_cap": False, "per_slot": []},
              "out_of_box_dropped": 0, "trust": asdict(trust), "codec": codec.identity,
              "seconds": time.perf_counter() - t0, "seed": seed, "q": q,
              "archive_keys": [space.key(o.config) for o in obs], "trust_gains": gains}
    return result, trust, max(0, (len(obs) - manifest.get("initial_configs", 10)) // 4)


def latent_search_worker_v071(directory, manifest, entries, agent, seed, round_file):
    """Spawn target (picklable by name): the v0.7.0 worker with the stall rule around its proposal."""
    with v070_patches(manifest):
        state = None
        if stall_rule(manifest):
            from trimem.optimizer.plan_surrogate import ObjectiveProtocol

            state = stall_state(directory, manifest, entries, ObjectiveProtocol(**manifest["objective"]),
                                before=round_file)
            write_hv_rounds(directory, state)
        stock_propose = tls.propose_for_round

        def propose(*args):
            forced = bool(state and state["force_exploration"])
            result, trust, idx = forced_proposal(*args) if forced else stock_propose(*args)
            rule = FORCED_RULE if forced else GREEDY_RULE
            if state:
                result.update(force_exploration=forced, stagnant_rounds=state["stagnant_rounds"],
                              round_hv=state["round_hv"], best_hv=state["best_hv"], selection_rule=rule,
                              stall_rule=stall_rule(manifest))
                for s in result["selected"]:
                    s.setdefault("selection_rule", rule)
            return result, trust, idx

        tls.propose_for_round = propose
        try:
            _stock_worker(directory, manifest, entries, agent, seed, round_file)
        finally:
            tls.propose_for_round = stock_propose
