"""TRACE latent-MOBO search worker behind the ``_search_worker`` seam (spec 7, 11).

The hashed driver ``trace_joint_plan_search`` is imported read-only. ``install()`` replaces only
its ``_search_worker`` attribute. ``guarded_search`` spawns ``latent_search_worker`` (a module
level function, so the child imports it by reference) and ``_run_round`` reserves its picks.

Surrogate ``saas-numpyro`` (option; live default is Matern x 3, user 2026-11-04): SAAS GPs fitted by NumPyro/JAX NUTS, the
three objectives in one vmapped program (nuts 512/256/16), warm-started from the previous round.

* CPU pinning: XLA sizes its CPU thread pool from the process CPU affinity mask when JAX creates
  its CPU client; OMP_NUM_THREADS / XLA_FLAGS do not bound it (measured ~21 busy cores, 342
  threads). guarded_search only sets the thread env vars, so the worker pins itself:
  ``pin_surrogate_cpus`` calls ``latent_numpyro_saas.pin_cpus(cpus, offset=cpu_offset)`` (manifest
  ``latent.surrogate``; defaults 12 and 0) before the first JAX use (importing jax does not create
  the client). It pins every thread of the worker (torch threads too). TRIMEM_JAX_CPUS is removed
  from the worker's environment before the module import, so the manifest is the only authority.
  Matern / Pyro SAAS workers are never pinned and never import JAX.
* Warm start: round R loads ``<run>/latent_warm/<P>.json``, P = the latest earlier proposal round
  file (``round_NNN.json``), with ``load_warm_state(expect = warm_meta + n)`` (checkpoint sha256,
  affine sha256, objective sha256, run id = run directory name, panel identity, n). Missing,
  invalid, or refused by the aging rule -> cold fit. After the fit the worker writes
  ``<run>/latent_warm/<R>.json`` BEFORE ``<R>.json``, so a committed round always has its warm
  state. An existing warm file is never rewritten (a worker re-run after a crash between the two
  writes keeps it and records whether the refit matches it). Replay contract (driver): when
  ``<R>.json`` exists the worker is not spawned at all, so a replayed round never refits.
  ``<R>.json["surrogate"]`` records kernel, pinned CPUs and ``fit`` = {mode cold|warm, generation,
  seconds, divergences, warm_reason, warm_load, warm_save, ...}.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

BACKEND = "latent-mobo-v1"
DECODE = "masked-greedy-v1"
DEFAULT_BUDGET = {"seconds": 300, "calls": 10000}
# user 2026-11-04: LLM 4 / front 4 / newplan 4 / random 4. One constant: validate_latent_manifest
# refuses to run if this differs from latent_plan_search.N_ANCHORS (the count the worker refines).
ANCHORS = 16
FROZEN_AFFINE_KIND = "frozen-train-init-v1"  # tools.latent_readiness freeze-affine (R6)
LATENT_KEYS = {
    "checkpoint", "checkpoint_sha256", "affine", "affine_sha256", "d", "decode",
    "proposal_budget", "anchors", "h0",
}
# live default (user 2026-11-04, later): Matern-5/2 ARD x 3 independent; saas-numpyro is an option
SURROGATE_DEFAULT = {"kernel": "matern", "structure": "independent", "nuts": [512, 256, 16],
                     "nuts_cost": None}
KERNELS = ("saas", "matern", "saas-numpyro")
LIVE_NUTS = [512, 256, 16]
# saas-numpyro only: worker CPU pinning (count, offset into the allowed CPUs) and warm warmup
NUMPYRO_DEFAULTS = {"cpus": 12, "cpu_offset": 0, "warm_warmup": 128}
MAX_CPUS = 32
# Staged pruning (user 2026-11-04 ~23:25), as the stratified TRACE-PR runs: required outside smoke.
LIVE_BUDGET = {"initial_configs": 10, "budget": 80, "new_configs": 70}  # budget = max configs
LIVE_PRUNING = {"sizes": [20, 40, 60, 80, 100], "budget_episodes": 5000}
WARM_DIR = "latent_warm"
CPU_ENV = "TRIMEM_JAX_CPUS"


def resolve_surrogate(lat: dict) -> dict:
    """latent.surrogate with defaults (saas-numpyro also gets cpus / cpu_offset / warm_warmup)."""
    sur = {**SURROGATE_DEFAULT, **(lat.get("surrogate") or {})}
    if sur["kernel"] == "saas-numpyro":
        sur = {**NUMPYRO_DEFAULTS, **sur}
    return sur


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _validate_numpyro(sur: dict, smoke: bool) -> None:
    if sur["structure"] != "independent":
        raise ValueError("surrogate kernel 'saas-numpyro' supports structure 'independent' only")
    nuts = sur["nuts"]
    if not (isinstance(nuts, (list, tuple)) and len(nuts) == 3 and all(_is_int(v) and v >= 1 for v in nuts)):
        raise ValueError(f"surrogate nuts must be 3 positive ints, got {nuts!r}")
    if not smoke and list(nuts) != LIVE_NUTS:
        raise ValueError(f"saas-numpyro requires surrogate.nuts == {LIVE_NUTS} (got {nuts!r})")
    if not (_is_int(sur["cpus"]) and 1 <= sur["cpus"] <= MAX_CPUS):
        raise ValueError(f"surrogate.cpus must be an int in [1, {MAX_CPUS}], got {sur['cpus']!r}")
    if not (_is_int(sur["cpu_offset"]) and sur["cpu_offset"] >= 0):
        raise ValueError(f"surrogate.cpu_offset must be an int >= 0, got {sur['cpu_offset']!r}")
    ww = sur["warm_warmup"]
    if not (_is_int(ww) and ww >= 1) or (not smoke and ww != NUMPYRO_DEFAULTS["warm_warmup"]):
        raise ValueError(f"surrogate.warm_warmup must be {NUMPYRO_DEFAULTS['warm_warmup']} (got {ww!r})")


def validate_latent_manifest(manifest: dict, *, running: bool = False) -> None:
    """Refuse anything but the frozen latent-MOBO protocol; check file hashes.

    running=True (worker): _run_round passes q = size of this batch, so 1 <= q <= 4 is accepted;
    the manifest-level q == 4 is checked at prepare time (running=False)."""
    from trimem.optimizer.latent_plan_codec import file_sha256, load_affine
    from trimem.optimizer.latent_plan_search import N_ANCHORS

    if ANCHORS != N_ANCHORS:
        raise RuntimeError(f"trace_latent_search.ANCHORS {ANCHORS} != latent_plan_search.N_ANCHORS {N_ANCHORS}")
    if manifest.get("search_backend") != BACKEND:
        raise ValueError(f"search_backend must be {BACKEND!r}")
    smoke = bool(manifest.get("smoke"))
    _validate_pruning(manifest, smoke)
    q = manifest.get("q", 4)
    if manifest.get("selection", "greedy") != "greedy" or not (
        (1 <= q <= 4) if running else q == 4
    ):
        raise ValueError("latent-mobo-v1 requires selection 'greedy' and q == 4")
    if not smoke and any(manifest.get(k) != v for k, v in LIVE_BUDGET.items()):
        raise ValueError(f"latent-mobo-v1 requires {LIVE_BUDGET} (staged pruning: budget = max configs)")
    lat = manifest.get("latent")
    if not isinstance(lat, dict) or not LATENT_KEYS <= set(lat):
        raise ValueError(f"manifest 'latent' block needs keys {sorted(LATENT_KEYS)}")
    if lat["decode"] != DECODE:
        raise ValueError(f"latent.decode must be {DECODE!r}")
    if not smoke and (
        lat["proposal_budget"] != DEFAULT_BUDGET or lat["h0"] != 0.4
    ):
        raise ValueError("latent proposal_budget/h0 differ from the frozen protocol")
    if lat["anchors"] != ANCHORS:  # also in smoke: the worker always refines N_ANCHORS anchors
        raise ValueError(f"latent.anchors must be {ANCHORS} (user 2026-11-04), got {lat['anchors']!r}")
    if smoke and lat["h0"] != 0.4:
        raise ValueError("latent.h0 must be 0.4")
    pb = lat["proposal_budget"]
    if not isinstance(pb, dict) or set(pb) != {"seconds", "calls"}:
        raise ValueError("latent.proposal_budget needs exactly seconds and calls")
    surrogate = lat.get("surrogate")
    if surrogate is None and not smoke:
        raise ValueError("latent.surrogate {kernel, structure, nuts, nuts_cost} is required")
    surrogate = resolve_surrogate(lat)
    if surrogate["kernel"] not in KERNELS or surrogate["structure"] not in (
        "independent", "coupled-cost"
    ):
        raise ValueError(f"unsupported surrogate {surrogate}")
    if surrogate["kernel"] == "saas-numpyro":
        _validate_numpyro(surrogate, smoke)
    if not smoke and surrogate["kernel"] == "matern" and surrogate["structure"] != "independent":
        raise ValueError("live kernel 'matern' requires structure 'independent' (user 2026-11-04)")
    for name, key in (("checkpoint", "checkpoint_sha256"), ("affine", "affine_sha256")):
        actual = file_sha256(lat[name])
        if actual != lat[key]:
            raise ValueError(f"{name} sha256 {actual} does not match manifest {lat[key]}")
    if not smoke:  # live runs use the R6 affine frozen on the shared init, never a provisional one
        kind = load_affine(lat["affine"])[1].get("kind")
        if kind != FROZEN_AFFINE_KIND:
            raise ValueError(f"latent.affine kind {kind!r}: live runs require {FROZEN_AFFINE_KIND!r} "
                             "(tools.latent_readiness freeze-affine)")


def _validate_pruning(manifest: dict, smoke: bool) -> None:
    """Staged pruning block (user 2026-11-04): staged-pareto-prune-v2, z 1.96, Agresti-Coull quality
    SE floor; outside smoke it is required with checkpoints 20..100 and the 5,000-episode budget, and
    the dataset's quality scale (ScienceWorld 100, else 1). The driver's validate_manifest checks the
    order (a permutation of the TRAIN tasks) and the stage subsets."""
    p = manifest.get("pruning")
    if p is None:
        if not smoke:
            raise ValueError("latent-mobo-v1 live arms require staged pruning (user 2026-11-04); "
                             "manifest has no 'pruning'")
        return
    from trimem.optimizer import staged_pruning as sp

    if (not isinstance(p, dict) or p.get("protocol") != sp.PROTOCOL or p.get("z") != sp.Z
            or p.get("quality_se_floor") != "agresti-coull"):
        raise ValueError(f"latent-mobo-v1 staged pruning must be {sp.PROTOCOL} (z {sp.Z}, "
                         "quality_se_floor agresti-coull)")
    if smoke:
        return
    scale = 100 if (manifest.get("fixed") or {}).get("dataset") == "sciworld" else 1
    if (list(p.get("sizes") or []) != LIVE_PRUNING["sizes"]
            or p.get("budget_episodes") != LIVE_PRUNING["budget_episodes"]
            or p.get("quality_scale") != scale):
        raise ValueError(f"latent-mobo-v1 staged pruning budget: sizes {LIVE_PRUNING['sizes']}, "
                         f"budget_episodes {LIVE_PRUNING['budget_episodes']}, quality_scale {scale}")


_NONFINITE = [0]


def _finite(x):
    """JSON-safe copy: non-finite floats become None (atomic_json forbids NaN)."""
    if isinstance(x, float):
        if math.isfinite(x):
            return x
        _NONFINITE[0] += 1
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def _prior_rounds(directory: Path, round_file: str):
    rounds = []
    for p in sorted(directory.glob("round_[0-9][0-9][0-9].json")):
        if p.stem >= round_file:
            continue
        body = json.loads(p.read_text())
        rounds.append((body["archive_keys"], [s["key"] for s in body["selected"]]))
    return rounds


def pin_surrogate_cpus(sur: dict) -> dict | None:
    """saas-numpyro only: pin this process to sur cpus/cpu_offset before the first JAX use (module
    docstring: XLA ignores the thread env vars). Returns {cpus, count, offset, ignored_env,
    jax_imported_before}; None (no pin, no JAX import) for any other kernel."""
    if sur.get("kernel") != "saas-numpyro":
        return None
    import sys

    ignored = os.environ.pop(CPU_ENV, None)  # no import-time pin: the manifest decides
    jax_before = "jax" in sys.modules
    from trimem.optimizer import latent_numpyro_saas as nps

    count = int(sur.get("cpus", NUMPYRO_DEFAULTS["cpus"]))
    offset = int(sur.get("cpu_offset", NUMPYRO_DEFAULTS["cpu_offset"]))
    cpus = nps.pin_cpus(count, offset=offset)
    # never more torch threads than pinned CPUs: oversubscribed OpenMP threads spin-wait and made
    # the (uncharged) pool scoring ~10x slower with 4 torch threads on 2 pinned CPUs (measured)
    import torch

    torch_threads = min(torch.get_num_threads(), len(cpus))
    torch.set_num_threads(torch_threads)
    return {"cpus": list(cpus), "count": count, "offset": offset, "ignored_env": ignored,
            "jax_imported_before": jax_before, "torch_threads": torch_threads}


def run_id(directory) -> str:
    """Warm-state provenance run id: the run directory name."""
    return Path(directory).resolve().name


def previous_round(directory, round_file: str) -> str | None:
    """The latest proposal round file (round_NNN.json) strictly before round_file."""
    prev = [p.stem for p in Path(directory).glob("round_[0-9][0-9][0-9].json") if p.stem < round_file]
    return max(prev) if prev else None


def warm_start_for_round(directory, round_file: str, codec, objective, observations):
    """(state | None, record) for round_file: <run>/latent_warm/<previous round>.json, loaded with
    load_warm_state(expect = warm_meta + n). Read-only (the smoke refit uses it too)."""
    from trimem.optimizer.latent_numpyro_saas import load_warm_state, warm_meta

    meta = warm_meta(codec, objective, observations, run_id=run_id(directory))
    prev = previous_round(directory, round_file)
    if prev is None:
        return None, {"source": None, "load": "no previous round"}, meta
    path = Path(directory) / WARM_DIR / f"{prev}.json"
    if not path.exists():
        return None, {"source": str(path), "load": "missing"}, meta
    state = load_warm_state(path, expect={**meta, "n": len(observations)})
    load = "loaded" if state is not None else "rejected (version/provenance/n/objectives)"
    return state, {"source": str(path), "load": load}, meta


def save_round_warm_state(directory, round_file: str, model, meta: dict) -> dict:
    """Write <run>/latent_warm/<round_file>.json unless it exists (never rewritten)."""
    from trimem.optimizer.latent_numpyro_saas import save_warm_state, warm_state

    path = Path(directory) / WARM_DIR / f"{round_file}.json"
    replaced = False
    if path.exists():
        try:
            existing = json.loads(path.read_text()).get("objectives")
        except (OSError, ValueError):
            existing = None
        if existing == json.loads(json.dumps(warm_state(model))):
            return {"path": str(path), "written": False, "existing_identical": True}
        # the file belongs to an uncommitted earlier attempt (no <round_file>.json yet, else the
        # worker would not run): replace it atomically so the next round warm-starts from the fit
        # that produced the committed proposal
        replaced = True
    path.parent.mkdir(parents=True, exist_ok=True)
    save_warm_state(path, model, meta=meta)
    return {"path": str(path), "written": True, **({"replaced_orphan": True} if replaced else {})}


def fit_round_model(directory, round_file, codec, obs, objective, sur, *, seed, on_objective=None,
                    save=True):
    """Fit this round's surrogate (warm start for saas-numpyro) -> (model, fit record)."""
    from trimem.optimizer.latent_plan_surrogate import fit_latent_model

    t = time.perf_counter()
    if sur["kernel"] != "saas-numpyro":
        model = fit_latent_model(
            codec, obs, objective, seed=seed, kernel=sur["kernel"], structure=sur["structure"],
            nuts=tuple(sur["nuts"]), nuts_cost=sur["nuts_cost"], on_objective=on_objective,
        )
        mode = "map" if sur["kernel"] == "matern" else "cold"
        return model, {"mode": mode, "wall_seconds": time.perf_counter() - t}
    state, load, meta = warm_start_for_round(directory, round_file, codec, objective, obs)
    model = fit_latent_model(
        codec, obs, objective, seed=seed, kernel="saas-numpyro", structure="independent",
        nuts=tuple(sur["nuts"]), warm_start=state, warm_warmup=int(sur["warm_warmup"]),
        on_objective=on_objective,
    )
    wall = time.perf_counter() - t
    summary = model.numpyro_summary
    first = model.models[0].numpyro_info
    record = {
        "mode": "warm" if summary["warm"] else "cold",
        "generation": int(first["state"]["generation"]),
        "cold_fit_n": int(first["state"]["cold_fit_n"]),
        "n": len(obs),
        "seconds": float(summary["seconds"]),
        "wall_seconds": wall,
        "new_compile": bool(first.get("new_compile")),
        "divergences": dict(summary["divergences"]),
        "init_tries": dict(summary["init_tries"]),
        "nuts": list(summary["nuts"]),
        "warm_reason": summary["warm_reason"],
        "warm_load": load,
    }
    if save:
        record["warm_save"] = save_round_warm_state(directory, round_file, model, meta)
    return model, record


def propose_for_round(directory, round_file, manifest, space, codec, snapshot, obs, objective,
                      agent, seed):
    """Trust replay + propose_latent_batch exactly as the worker runs it (the smoke replays it).
    Work-based: BatchBudget(seconds=None, calls=proposal_budget.calls); the manifest's 300 s is
    no stopping rule any more, latent_plan_search.PROPOSAL_ABORT_SECONDS aborts (raises)."""
    from trimem.optimizer.latent_plan_search import propose_latent_batch
    from trimem.optimizer.latent_region import BatchBudget, batch_gains, replay

    gains = batch_gains(
        _prior_rounds(directory, round_file), obs, objective,
        key=lambda o: space.key(o.config),
    )
    trust = replay(gains)
    q = manifest["q"]  # this batch's size (1..4), set by _run_round
    initial = manifest.get("initial_configs", 10)
    # same as the driver's seed offset, which divides by the manifest-level q (== 4)
    round_index = max(0, (len(obs) - initial) // 4)
    pb = manifest["latent"]["proposal_budget"]
    result = propose_latent_batch(
        space, codec, snapshot, [o.config for o in obs], agent=agent, seed=seed,
        round_index=round_index, q=q, trust=trust,
        budget=BatchBudget(seconds=None, calls=pb["calls"]),
    )
    result["archive_keys"] = [space.key(o.config) for o in obs]
    result["trust_gains"] = gains
    return result, trust, round_index


def latent_search_worker(directory, manifest, entries, agent, seed, round_file):
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    directory = Path(directory)
    from trimem.experiments.trace_joint_plan_search import (
        aggregate_panel, analyzer_client, atomic_json, get_space,
    )

    try:
        validate_latent_manifest(manifest, running=True)  # before any GP work: fails inside the startup window
        lat = manifest["latent"]
        sur = resolve_surrogate(lat)
        pin = pin_surrogate_cpus(sur)  # saas-numpyro only; before the first JAX use
        from trimem.optimizer.latent_plan_codec import load_codec
        from trimem.optimizer.latent_plan_surrogate import latent_snapshot
        from trimem.optimizer.plan_surrogate import ObjectiveProtocol

        space = get_space(manifest)
        codec = load_codec(lat["checkpoint"], lat["affine"], space)
        if codec.d != lat["d"]:
            raise ValueError(f"checkpoint d={codec.d} != manifest latent.d={lat['d']}")
        obs = [
            aggregate_panel(
                space, e["config"], e["receipts"], task_ids=manifest["tasks"],
                protocol=manifest["protocol"], allow_synthetic=manifest.get("smoke", False),
                # staged pruning: a pruned configuration enters the GP with its partial panel
                # (var / n, Agresti-Coull quality floor), as the stock worker does
                allow_partial=bool(manifest.get("pruning")),
                quality_scale=(manifest.get("pruning") or {}).get("quality_scale"),
            )
            for e in entries
        ]
        objective = ObjectiveProtocol(**manifest["objective"])
        settings = manifest.get("analyzer")
        if not agent and settings:
            from trimem.optimizer.joint_plan_analyzer import advise

            client = analyzer_client(settings)
            atomic_json(directory / "worker_phase.json", {"phase": "analyzer", "at": time.time()})
            agent, advice = advise(space, obs, client)
            atomic_json(directory / (round_file + ".analyzer.json"), advice)

        def on_objective(i, name):  # each objective fit gets its own 125 s window
            # saas-numpyro: called 3x right before the ONE vmapped fit (harmless refreshes); that
            # fit is one phase, measured <= ~30 s cold incl. compile, under the 125 s guard
            atomic_json(directory / "worker_phase.json", {"phase": "gp", "at": time.time()})

        atomic_json(directory / "worker_phase.json", {"phase": "gp", "at": time.time()})
        model, fit_record = fit_round_model(
            directory, round_file, codec, obs, objective, sur, seed=seed, on_objective=on_objective,
        )
        atomic_json(directory / "worker_phase.json", {"phase": "search", "at": time.time()})

        snapshot = latent_snapshot(codec, model, obs, objective, seed=seed)
        # raises ShortBatch (< q legal configs) or ProposalAborted (wall-clock safety abort)
        # BEFORE round_file.json is written: nothing is committed (Task 13 verdict 1-3)
        result, trust, round_index = propose_for_round(
            directory, round_file, manifest, space, codec, snapshot, obs, objective, agent, seed,
        )
        if len(result["selected"]) != manifest["q"]:
            raise RuntimeError(f"proposal has {len(result['selected'])} picks for q = {manifest['q']}")
        result["surrogate"] = {
            "kernel": sur["kernel"], "structure": sur["structure"],
            "pinned_cpus": None if pin is None else pin["cpus"], "pin": pin, "fit": fit_record,
        }
        torch.save(
            {
                "codec": codec.identity,
                "gp_state": model.state_dict(),
                "objective": asdict(objective),
                "trust": asdict(trust),
                "seed": seed,
                "round_index": round_index,
                "sources": manifest["sources"],
                "latent": manifest["latent"],
            },
            directory / (round_file + ".pt"),
        )
        _NONFINITE[0] = 0
        body = _finite(result)
        body["nonfinite_to_null"] = _NONFINITE[0]
        atomic_json(directory / (round_file + ".json"), body)
    except Exception as exc:
        atomic_json(directory / (round_file + ".error.json"), {"error": repr(exc)})
        raise


@contextlib.contextmanager
def install(module=None):
    """Patch ``module._search_worker`` (default: the hashed driver); restore on exit."""
    if module is None:
        from trimem.experiments import trace_joint_plan_search as module
    original = module._search_worker
    module._search_worker = latent_search_worker
    try:
        yield module
    finally:
        module._search_worker = original
