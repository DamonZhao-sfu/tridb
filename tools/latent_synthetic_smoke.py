"""Synthetic end-to-end smoke of the TRACE latent-MOBO arm (spec 12.3; Task 12).

No Postgres, no services, no agent episodes, CPU only. The path is the live one:

1. Shared init: ``prepare_trace_latent_campaign.shared_prepare_run`` + the real driver
   ``run_round`` (5 Sobol, then 5 "LLM" configs; ``driver.initial_advice`` is patched to a
   seeded stand-in, as ``tests/latent_fixtures.make_shared_init`` does).
2. Latent arm: ``arm_patches`` + ``arm_import_archive`` (imports the 10 complete panels) with
   ``latent_block`` of the real checkpoint and the provisional affine.
3. Exploration: ``with trace_latent_live.backend_patches(m): driver.run_round(arm, ...)`` until
   ``budget_complete`` (real ``guarded_search``, spawned latent worker, ledger).

The evaluator is a deterministic synthetic receipt (``synthetic: true``). Reaching every check
``true`` is not evidence that the method improves agent search (spec 12.3).

Crash/restart: at one chosen round, ``SearchLedger.reserve`` is patched (scoped, this process
only) to raise once *after* ``round_NNN.json`` exists. The resumed ``run_round`` must reserve
exactly the round file's keys without re-running the worker.

Surrogate ``saas-numpyro`` (option; live default Matern x 3): every worker pins itself to
``--cpus`` CPUs and warm-starts from the previous round (``<arm>/latent_warm``). The check refit
runs in this process with the same warm state under the same (scoped, then undone) CPU pin, so a
bit-identical refit proves the exact rescoring; a refit that is not bit-identical makes
``rescoring_used`` false ("not verified").

Staged pruning (``--pruning``, user 2026-11-04): the arm gets the live pruning block
(``prepare_trace_latent_campaign.add_pruning``; checkpoints every ``--stage-step`` panel tasks, strata
from ``<space-run>/train_panel.json``), runs until the driver's pruning ``budget_complete``
(``staged_batch_size == 0``), and adds the check ``pruned_at_checkpoint`` (>= 1 configuration pruned at
the first checkpoint, recorded in ``pruning_log.jsonl``, entering later rounds' archives as a partial
panel). ``--new N`` shortens the episode budget to (10 + N) x tasks (short smoke); the live
validators then fail only on that budget, which is reported.

Affine: ``--affine`` is copied into ``--out``. Without it, the checkpoint's provisional affine cache
is copied; when that cache does not exist, ``tools.latent_readiness.provisional_affine`` computes it
into ``--out`` (never next to the checkpoint).

Usage:
  PYTHONPATH=. python -m tools.latent_synthetic_smoke --out results/trace_latent_smoke_20261104
  PYTHONPATH=. python -m tools.latent_synthetic_smoke --out OUT --kernel saas-numpyro \\
      --checkpoint results/trace_latent_seqvae_v2_20261104/seqvae_v2_d16.pt
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import shutil
import threading
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE_RUN = ROOT / "results/trace_pr_alfworld_v060_strat_20261104"
DEFAULT_CHECKPOINT = ROOT / "results/trace_latent_seqvae_20261104/seqvae_d16.pt"
PROTOCOL = "trace-latent-synthetic-smoke-20261104"
CHECKS = (
    "gp_three_models", "fd_gradient_ok", "rescoring_used", "no_measured_or_pending_dup",
    "batch_sizes", "unique_configs", "proposal_budget_respected", "restart_consistent",
    # pre-launch fix round: q picks every round, shared call counter <= cap, deterministic replay
    "picks_equal_q", "total_calls_capped", "deterministic_replay",
)
PRUNING_CHECKS = ("pruned_at_checkpoint",)
# Shapes: full = 10 init + 10 x 4 (user decision); fast = 10 + 2 x 4 (unit test).
SHAPES = {
    False: {"new": 40, "proposal_budget": {"seconds": 300, "calls": 10000}, "crash_round": 5},
    # Fast: a small shared call cap keeps the unit test short (seconds never bind any more).
    True: {"new": 8, "proposal_budget": {"seconds": 8, "calls": 2000}, "crash_round": 1},
}
INITIAL, Q, ANCHORS = 10, 4, 16
# Pre-launch fix round: the proposal is work-based (one shared call counter covers refinement,
# projection rescoring and pool scoring); wall time only drives the raising safety abort.
FD_STEPS = (1e-4, 1e-5, 1e-6)
FD_STEP, FD_TOL = 1e-5, 1e-2
SCORE_TOL = 1e-9
U_TOL = 1e-5


class SimulatedCrash(RuntimeError):
    """Raised once by the patched reserve: the driver 'dies' after the round file exists."""


# ------------------------------------------------------------------ synthetic evaluator
def _jitter(task: str, salt: str, mod: int = 3) -> int:
    return int(hashlib.sha256(f"{salt}:{task}".encode()).hexdigest()[:8], 16) % mod


def synthetic_receipt(config: dict, task: str, manifest: dict) -> dict:
    """Copy of ``tests.latent_fixtures.synthetic_receipt`` for real-space configs.

    Same structure (quality up with seed_count=32 and LLM rank; tokens up with depth and
    seed_count; latency up with beam and depth). Inactive (None) fields fall back to the
    fixture's neutral values. The per-task jitter is a hash of the task id (real ids are
    not ``t<digit>``) and is applied to all three objectives so no column is noise-free."""
    depth = config.get("depth") or 2
    seeds = config.get("seed_count") or 8
    beam = (config.get("beam_width") or 1) if config.get("graph_impl") == "beam" else 1
    good = config.get("seed_count") == 32
    llm = config.get("rank_impl") == "LLM"
    return {
        "task_id": task,
        "split": manifest.get("evaluation_split", "TRAIN"),
        "protocol": manifest["protocol"],
        "synthetic": True,
        "raw_score": 40 * good + 40 * llm + 10 + _jitter(task, "q"),
        "total_tokens": 1000 * depth + 100 * (seeds // 8) + 50 * _jitter(task, "t"),
        "latency_seconds": 10 + 5 * beam + 2 * depth + _jitter(task, "l"),
        "agent_steps": 3 + depth,
    }


def _unit(*parts) -> float:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


def rugged_receipt(config: dict, task: str, manifest: dict) -> dict:
    """``--landscape rugged`` (pruning smoke): ALFWorld-like 0/1 reward with a per-configuration
    success probability and costs drawn from a hash of the canonical config (no learnable structure),
    per-task Bernoulli outcome and cost jitter. BO picks are then often dominated after 20 tasks, so
    the staged checkpoints actually prune. Synthetic; not evidence of anything (spec 12.3)."""
    c = json.dumps(config, sort_keys=True)
    p = 0.1 + 0.8 * _unit("p", c)
    return {
        "task_id": task,
        "split": manifest.get("evaluation_split", "TRAIN"),
        "protocol": manifest["protocol"],
        "synthetic": True,
        "raw_score": int(_unit("s", c, task) < p),
        "total_tokens": round((3000 + 120000 * _unit("t", c) ** 2) * (0.8 + 0.4 * _unit("tj", c, task))),
        "latency_seconds": (30 + 400 * _unit("l", c) ** 2) * (0.8 + 0.4 * _unit("lj", c, task)),
        "agent_steps": 5 + int(20 * _unit("a", c, task)),
    }


LANDSCAPES = {"structured": synthetic_receipt, "rugged": rugged_receipt}


def _evaluator(space, landscape: str = "structured"):
    receipt = LANDSCAPES[landscape]

    def evaluate(config, task, manifest):  # the importer restates config_key
        return {**receipt(config, task, manifest), "config_key": space.key(config)}

    return evaluate


# ------------------------------------------------------------------ fast-mode codec patch
def _enumerate(space) -> list[dict]:
    from trimem.optimizer.plan_schema import FIELD_NAMES

    seen: dict = {}
    for combo in itertools.product(*(space.domains[n] for n in FIELD_NAMES)):
        c = space.canonicalize(dict(zip(FIELD_NAMES, combo)))
        seen.setdefault(space.key(c), c)
    return list(seen.values())


def _patch_codec(codec):
    """Fast mode only (tiny model round-trips ~1/576): every config round-trips and decode is
    the nearest enumerated config by u, as the ``lcodec`` fixture of test_trace_latent_propose."""
    import torch

    configs = _enumerate(codec.space)
    U = codec.encode_many(configs)
    codec.roundtrip = lambda c: True
    codec.decode = lambda X: [
        configs[int(torch.cdist(x.reshape(1, -1).double(), U).argmin())]
        for x in torch.as_tensor(X)
    ]
    return codec


def _fast_worker(directory, manifest, entries, agent, seed, round_file):
    """Spawned-worker entry for fast mode: patch load_codec in the child, then the real worker."""
    from trimem.experiments import trace_latent_search as tls
    from trimem.optimizer import latent_plan_codec as lpc

    real = lpc.load_codec
    lpc.load_codec = lambda *a, **k: _patch_codec(real(*a, **k))
    try:
        return tls.latent_search_worker(directory, manifest, entries, agent, seed, round_file)
    finally:
        lpc.load_codec = real


@contextlib.contextmanager
def _worker(fast: bool):
    from trimem.experiments import trace_joint_plan_search as driver

    if not fast:
        yield
        return
    saved = driver._search_worker
    driver._search_worker = _fast_worker
    try:
        yield
    finally:
        driver._search_worker = saved


# ------------------------------------------------------------------ helpers
def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write(path: Path, body) -> None:
    from trimem.experiments.trace_joint_plan_search import atomic_json

    atomic_json(path, body)


def _clean(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _ledger(directory: Path):
    from trimem.optimizer.joint_search_ledger import SearchLedger

    return SearchLedger(directory / "ledger.sqlite", _json(directory / "manifest.json"))


class _PhaseWatch:
    """Polls worker_phase.json (written by the worker before each GP objective and the search)."""

    def __init__(self, directory: Path, period: float = 0.1):
        self.path, self.period = directory / "worker_phase.json", period
        self.events: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        seen = set()
        while not self._stop.is_set():
            try:
                p = json.loads(self.path.read_text())
                tag = (p["phase"], p["at"])
                if tag not in seen:
                    seen.add(tag)
                    self.events.append({"phase": p["phase"], "at": p["at"]})
            except (OSError, ValueError, KeyError):
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self.began = time.time()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _timings(watch: _PhaseWatch, wall: float, body: dict | None, resources: dict | None) -> dict:
    events = [e for e in watch.events if e["at"] >= watch.began]  # drop a stale earlier file
    gp = [e["at"] for e in events if e["phase"] == "gp"]
    search = [e["at"] for e in events if e["phase"] == "search"]
    fit = (search[0] - gp[0]) if gp and search else None
    marks = gp + search[:1]  # gp(start), gp(objective 0..2), search: gaps = per-objective fits
    gaps = [b - a for a, b in itertools.pairwise(marks)]
    return {
        "wall_seconds": wall,
        "startup_seconds": (gp[0] - watch.began) if gp else None,
        "fit_seconds": fit,
        "fit_gaps_seconds": gaps,
        "proposal_seconds": None if body is None else body.get("seconds"),
        "refinement_calls": None if body is None else body["budget"].get("refinement_calls"),
        "scoring_calls": None if body is None else body["budget"].get("scoring_calls"),
        "charged_calls": None if body is None else body["budget"]["calls"],
        "worker_wall_seconds": None if resources is None else resources["wall_seconds"],
        "worker_peak_rss_bytes": None if resources is None else resources["peak_rss_bytes"],
        "phase_events": events,
    }


# ------------------------------------------------------------------ setup
def _source(space_run: Path) -> dict:
    from trimem.experiments import trace_joint_plan_search as driver

    m = _json(Path(space_run) / "manifest.json")
    panel = Path(space_run) / "train_panel.json"
    return {"space": driver.get_space(m), "space_manifest": m["space"], "tasks": list(m["tasks"]),
            "holdout": list(m["holdout"]["tasks"]), "seed": int(m.get("seed", 0)),
            "dataset": m.get("fixed", {}).get("dataset", "synthetic"),
            "panel_path": str(panel) if panel.exists() else None}


def _affine(out: Path, checkpoint: Path, space, affine: Path | None) -> tuple[Path, dict]:
    """Copy the affine into ``out`` (never write next to the checkpoint)."""
    from trimem.optimizer.latent_plan_codec import file_sha256, load_affine

    sha = file_sha256(checkpoint)
    if affine is not None:
        dst = out / Path(affine).name
        shutil.copyfile(affine, dst)
        if load_affine(dst)[1].get("checkpoint_sha256") != sha:
            raise ValueError("affine meta checkpoint_sha256 differs from the checkpoint")
        return dst, {"source": str(affine), "kind": "given"}
    from tools.latent_readiness import affine_cache_path, provisional_affine
    from trimem.optimizer.latent_seq_vae import load_checkpoint

    cache = affine_cache_path(Path(checkpoint).parent, checkpoint)
    dst = affine_cache_path(out, checkpoint)
    if not cache.exists():  # compute into out; never write next to the checkpoint
        model, metrics = load_checkpoint(checkpoint, space)
        t0 = time.monotonic()
        provisional_affine(checkpoint, space, model, metrics, cache_dir=out, sha=sha)
        return dst, {"source": None, "kind": "provisional-train-v1 (computed into out)",
                     "sha256": file_sha256(dst), "seconds": time.monotonic() - t0}
    shutil.copyfile(cache, dst)
    before = file_sha256(dst)
    model, metrics = load_checkpoint(checkpoint, space)
    provisional_affine(checkpoint, space, model, metrics, cache_dir=out, sha=sha)
    if file_sha256(dst) != before:
        raise RuntimeError("provisional_affine rewrote the cache copy (cache not accepted)")
    return dst, {"source": str(cache), "kind": "provisional-train-v1 (cached copy)",
                 "sha256": before}


def _base_manifest(src: dict, protocol: str) -> dict:
    from trimem.experiments import trace_joint_plan_search as driver

    return {
        "space": src["space_manifest"], "tasks": src["tasks"], "protocol": protocol,
        "smoke": True,  # synthetic receipts are admitted only in a smoke protocol
        "fixed": {"dataset": src["dataset"], "synthetic": True},
        "live": {"artifacts": {}, **({"panel_path": src["panel_path"]} if src.get("panel_path") else {})},
        "q": Q, "seed": src["seed"], "task_concurrency": 16,
        "sources": driver.source_identity(),
        "holdout": {"tasks": src["holdout"], "selection": "deferred"},
    }


def _shared_init(out: Path, src: dict, evaluate) -> Path:
    import tools.prepare_trace_latent_campaign as pl
    from trimem.experiments import trace_joint_plan_search as driver

    run = out / "shared_init"
    m = {**_base_manifest(src, PROTOCOL + "-shared-init"), "initial_configs": INITIAL,
         "new_configs": 0, "budget": INITIAL,
         "initialization": {"method": "sobol_then_llm", "sobol": 5, "llm": 5}}
    m = pl.shared_prepare_run(run, m)

    def advice(sp, obs, settings):  # seeded stand-in for the LLM initializer (no model call)
        seen = {sp.key(o.config) for o in obs}
        return sp.sample(5, seed=src["seed"] + 555, avoid=seen), {"status": "synthetic-stand-in"}

    with mock.patch.object(driver, "initial_advice", advice):
        for _ in range(4):
            if driver.run_round(run, evaluator=evaluate)["status"] == "budget_complete":
                break
        else:
            raise RuntimeError("shared init did not reach budget_complete")
    return run


def _arm(out: Path, src: dict, shared: Path, block: dict, new: int, *, pruning: bool = False,
         stage_step: int = 20) -> tuple[Path, dict]:
    import tools.prepare_trace_latent_campaign as pl
    from trimem.experiments import trace_latent_search as tls

    arm = out / "arm_latent"
    arm.mkdir()
    m = {**_base_manifest(src, PROTOCOL + "-latent-arm"), "initial_configs": INITIAL,
         "new_configs": new, "budget": INITIAL + new, "initialization": None}
    saved = dict(pl.ARM)
    pl.ARM.update(backend=tls.BACKEND, latent=block, pruning=pruning, stage_step=stage_step,
                  import_policy="init")
    try:
        with pl.arm_patches():
            pl.arm_import_archive(arm, m, shared)
    finally:
        pl.ARM.update(saved)
    return arm, _json(arm / "manifest.json")


def _validators(m: dict) -> dict:
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments import trace_latent_search as tls

    out = {}
    for name, fn, man in (
        ("driver.validate_manifest", driver.validate_manifest, m),
        ("trace_latent_search.validate_latent_manifest", tls.validate_latent_manifest, m),
        ("driver.validate_manifest(smoke=False)", driver.validate_manifest, {**m, "smoke": False}),
        ("validate_latent_manifest(smoke=False)", tls.validate_latent_manifest,
         {**m, "smoke": False}),
    ):
        try:
            fn(man)
            out[name] = "ok"
        except ValueError as exc:
            out[name] = f"FAILED: {exc}"
    return out


# ------------------------------------------------------------------ rounds + crash
def _agent(arm: Path, space, seed: int, k: int) -> list[dict]:
    """Stand-in for the analyzer's LLM proposals (seeded draws; no model call)."""
    with _ledger(arm) as led:
        seen = {e["key"] for e in led.complete_configs()}
    return space.sample(4, seed=seed + 9000 + k, avoid=seen)


def _batch_size(arm: Path, m: dict) -> int:
    """The driver's next batch size: staged_batch_size under pruning, else next_batch_size."""
    from trimem.experiments import trace_joint_plan_search as driver

    with _ledger(arm) as led:
        if m.get("pruning"):
            return driver.staged_batch_size(led, m)
        return driver.next_batch_size(m, len(led.complete_configs()))


def _mtimes(arm: Path, name: str) -> dict:
    return {p.name: p.stat().st_mtime_ns for p in sorted(arm.glob(name + ".*"))}


def _crash_and_resume(arm: Path, m: dict, name: str, agent, evaluate) -> dict:
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.optimizer.joint_search_ledger import SearchLedger

    path = arm / (name + ".json")
    real = SearchLedger.reserve
    state = {"raised": 0, "reserve_calls": 0}

    def crashing(self, space, configs):
        state["reserve_calls"] += 1
        if path.exists() and not state["raised"]:
            state["raised"] += 1
            raise SimulatedCrash(f"simulated crash before reserving {name}")
        return real(self, space, configs)

    crashed = False
    with mock.patch.object(SearchLedger, "reserve", crashing):
        try:
            driver.run_round(arm, evaluator=evaluate, agent=agent)
        except SimulatedCrash:
            crashed = True
    body = _json(path)
    with _ledger(arm) as led:
        pending_after_crash = len(led.pending())
        complete = len(led.complete_configs())
    batch = _batch_size(arm, m)
    before = _mtimes(arm, name)
    resumed = driver.run_round(arm)  # reserve only: must replay the frozen round file
    with _ledger(arm) as led:
        reserved = sorted({p["key"] for p in led.pending()})
    after = _mtimes(arm, name)
    file_keys = sorted(s["key"] for s in body["selected"][:batch])
    rewritten = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    ok = (crashed and state["raised"] == 1 and pending_after_crash == 0
          and resumed["status"] == "reserved" and reserved == file_keys
          and len(reserved) == batch and not rewritten
          and resumed["proposal"]["selected"] == body["selected"])
    return {"pass": ok, "round_file": name, "crashed": crashed, "raised": state["raised"],
            "pending_after_crash": pending_after_crash, "batch": batch,
            "resume_status": resumed["status"], "reserved_keys": reserved,
            "round_file_keys": file_keys, "files_rewritten": rewritten,
            "files_checked": sorted(after)}


def _run_rounds(arm: Path, m: dict, space, evaluate, *, fast: bool, crash_round: int):
    from tools.trace_latent_live import backend_patches
    from trimem.experiments import trace_joint_plan_search as driver

    rounds, restart, final = [], None, None
    agents: dict = {}
    expected = m["budget"] if m.get("pruning") else (m["budget"] - INITIAL) // Q
    with backend_patches(m), _worker(fast):
        for k in range(expected + 2):
            with _ledger(arm) as led:
                complete, pending = led.complete_configs(), led.pending()
            name = f"round_{len(complete):03d}"
            batch = _batch_size(arm, m)
            if not pending and (batch == 0 if m.get("pruning") else len(complete) >= m["budget"]):
                final = driver.run_round(arm, evaluator=evaluate)
                break
            agent = _agent(arm, space, m["seed"], k)
            agents[name] = agent
            t0 = time.monotonic()
            with _PhaseWatch(arm) as watch:
                if k == crash_round:
                    restart = _crash_and_resume(arm, m, name, agent, evaluate)
                    status = driver.run_round(arm, evaluator=evaluate)
                else:
                    status = driver.run_round(arm, evaluator=evaluate, agent=agent)
            wall = time.monotonic() - t0
            body = _json(arm / (name + ".json")) if (arm / (name + ".json")).exists() else None
            res_path = arm / (name + ".resources.json")
            resources = _json(res_path) if res_path.exists() else None
            rec = {"round": k, "file": name, "complete_before": len(complete), "batch_size": batch,
                   "pending_before": len(pending), "complete_after": status["complete_configs"],
                   "status": status["status"], "crash_round": k == crash_round,
                   **_timings(watch, wall, body, resources)}
            if body is not None:
                sur = body.get("surrogate") or {}
                rec["surrogate_fit"] = sur.get("fit")
                rec["pinned_cpus"] = sur.get("pinned_cpus")
                rec["picks"] = [{"key": s["key"], "sources": s["sources"], "ell": s["ell"],
                                 "latent": s.get("latent") is not None}
                                for s in body["selected"]]
                rec["anchors_ok"] = sum(a["status"] == "ok" for a in body["anchors"])
                rec["anchor_sources"] = body.get("anchor_sources")
                rec["fallbacks"] = body["fallbacks"]
                rec["per_slot"] = body["budget"]["per_slot"]
            rounds.append(rec)
            print(json.dumps({"round": k, "file": name, "status": rec["status"],
                              "complete": rec["complete_after"], "wall": round(wall, 1),
                              "fit": rec["fit_seconds"], "proposal": rec["proposal_seconds"]}),
                  flush=True)
            if status["status"] != "batch_complete":
                raise RuntimeError(f"round {name} ended {status['status']}")
    if final is None or final["status"] != "budget_complete":
        raise RuntimeError(f"arm did not reach budget_complete: {final}")
    for r in rounds:
        r["agent"] = agents.get(r["file"])
    return rounds, restart, final


# ------------------------------------------------------------------ checks
def _check_shapes(arm, m, rounds, init_keys):
    with _ledger(arm) as led:
        complete = led.complete_configs()
        pending = led.pending()
    keys = [e["key"] for e in complete]
    observed = [r["complete_after"] - r["complete_before"] for r in rounds]
    # pruning: the driver's staged_batch_size at each round (1..4; saved episodes buy configs)
    expected = ([r["batch_size"] for r in rounds] if m.get("pruning")
                else [Q] * ((m["budget"] - INITIAL) // Q))
    proposed = [len(r["picks"]) for r in rounds]
    selected = [p["key"] for r in rounds for p in r["picks"]]
    union = set(init_keys) | set(selected)
    batch = {"pass": observed == expected and proposed == expected
             and all(1 <= b <= Q for b in expected),
             "observed": observed, "expected": expected, "selected_per_round": proposed}
    final_ok = (_batch_size(arm, m) == 0 and len(keys) <= m["budget"]) if m.get("pruning") \
        else len(keys) == m["budget"]
    unique = {"pass": len(keys) == len(set(keys)) == len(union) and final_ok and not pending,
              "final_complete": len(keys), "distinct": len(set(keys)), "budget": m["budget"],
              "init_plus_selected_distinct": len(union), "pending_at_end": len(pending)}
    return batch, unique


def _check_dups(arm, rounds):
    prior: set = set()
    bad = []
    for r in rounds:
        body = _json(arm / (r["file"] + ".json"))
        sel = [s["key"] for s in body["selected"]]
        archive = set(body["archive_keys"])
        problems = []
        if len(sel) != len(set(sel)):
            problems.append("duplicate within batch")
        if set(sel) & archive:
            problems.append("measured key selected")
        if set(sel) & prior:
            problems.append("key of an earlier batch selected")
        if r["pending_before"]:
            problems.append("proposal made while episodes were pending")
        if len(archive) != r["complete_before"]:
            problems.append("archive differs from complete configs")
        if problems:
            bad.append({"round": r["file"], "problems": problems})
        prior |= set(sel)
    return {"pass": not bad, "violations": bad, "rounds": len(rounds),
            "pending_at_each_proposal": [r["pending_before"] for r in rounds]}


def _check_budget(rounds, lat):
    """Work budget: shared calls <= cap, wall <= the safety abort, anchors <= the anchor count."""
    from trimem.optimizer.latent_plan_search import PROPOSAL_ABORT_SECONDS

    cap_c = lat["proposal_budget"]["calls"]
    rows, ok = [], True
    for r in rounds:
        row = {"round": r["file"], "wall_seconds": r["proposal_seconds"], "calls": r["charged_calls"],
               "refinement_calls": r["refinement_calls"], "scoring_calls": r["scoring_calls"],
               "anchors_ok": r["anchors_ok"],
               "max_visited_per_slot": max(s["visited"] for s in r["per_slot"])}
        row["pass"] = (row["wall_seconds"] <= PROPOSAL_ABORT_SECONDS
                       and row["calls"] <= cap_c and row["anchors_ok"] <= lat["anchors"]
                       and row["max_visited_per_slot"] <= lat["anchors"])
        ok &= row["pass"]
        rows.append(row)
    return {"pass": ok, "caps": {"calls": cap_c, "abort_seconds": PROPOSAL_ABORT_SECONDS,
                                 "anchors": lat["anchors"]},
            "max_wall_seconds": max(r["wall_seconds"] for r in rows),
            "max_calls": max(r["calls"] for r in rows), "rounds": rows}


def _check_picks(arm, rounds, q) -> dict:
    """Every round proposes exactly its batch size (q, or the staged batch size under pruning)."""
    per = [len(_json(arm / (r["file"] + ".json"))["selected"]) for r in rounds]
    want = [r.get("batch_size", q) for r in rounds]
    return {"pass": per == want, "picks_per_round": per, "batch_sizes": want, "q": q}


def _check_pruning(arm, m, rounds) -> dict:
    """>= 1 configuration pruned at the FIRST checkpoint; every pruned configuration is complete with
    exactly that many receipts and enters every later round's archive (partial panel in the GP);
    initialization configurations are never pruned."""
    p = m["pruning"]
    first = int(p["sizes"][0])
    path = arm / "pruning_log.jsonl"
    log = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    pruned = {r["key"]: r["stage"] for r in log if r["prune"]}
    with _ledger(arm) as led:
        entries = {e["key"]: e for e in led.complete_configs()}
    init = set(m["initial_keys"])
    bad = []
    for key, stage in pruned.items():
        e = entries.get(key)
        if e is None or not e["pruned"] or len(e["receipts"]) != stage:
            bad.append({"key": key, "why": "not complete with its stage receipts"})
        if key in init:
            bad.append({"key": key, "why": "initialization configuration pruned"})
        picked = next((r["round"] for r in rounds if key in {x["key"] for x in r.get("picks") or []}), None)
        missing = [r["file"] for r in rounds if picked is not None and r["round"] > picked
                   and key not in _json(arm / (r["file"] + ".json"))["archive_keys"]]
        if missing:
            bad.append({"key": key, "why": f"missing from later round archives {missing}"})
    at_first = sorted(k for k, st in pruned.items() if st == first)
    return {"pass": bool(at_first) and not bad, "first_checkpoint": first, "sizes": p["sizes"],
            "pruned_at_first_checkpoint": at_first, "pruned_by_stage": pruned,
            "decisions_logged": len(log), "violations": bad,
            "episodes_done": sum(len(e["receipts"]) for e in entries.values()),
            "budget_episodes": p["budget_episodes"], "configs": len(entries)}


def _check_calls(arm, rounds, cap) -> dict:
    rows = []
    for r in rounds:
        b = _json(arm / (r["file"] + ".json"))["budget"]
        rows.append({"round": r["file"], "calls": b["calls"], "refinement": b.get("refinement_calls"),
                     "scoring": b.get("scoring_calls"), "over_cap": b.get("over_cap")})
    ok = all(x["calls"] <= cap and x["refinement"] is not None
             and x["calls"] == x["refinement"] + x["scoring"] and not x["over_cap"] for x in rows)
    return {"pass": ok, "cap": cap, "rounds": rows, "max_calls": max(x["calls"] for x in rows)}


def _check_replay(arm, m, space, codec, model, obs, objective, body, round_file, agent) -> dict:
    """Re-run the round's proposal in THIS process (other timing) on the bit-identical refit:
    the picks, endpoints and call counts must be identical (work-based stopping)."""
    from trimem.experiments.trace_latent_search import propose_for_round
    from trimem.optimizer.latent_plan_surrogate import latent_snapshot

    seed = m["seed"] + (len(obs) - INITIAL) // Q
    snap = latent_snapshot(codec, model, obs, objective, seed=seed)
    t0 = time.monotonic()
    again, _, _ = propose_for_round(arm, round_file, {**m, "q": body["q"]}, space, codec, snap, obs,
                                    objective, agent, seed)
    secs = time.monotonic() - t0

    def sig(b):
        return {"picks": [(s["key"], s["ell"]) for s in b["selected"]],
                "endpoints": [(e["anchor_key"], e["slot"], e["status"], e["calls"],
                               None if e["u_end"] is None else list(e["u_end"])) for e in b["endpoints"]],
                "calls": b["budget"]["calls"]}

    a, b = sig(body), sig(_clean(again))
    same = {k: a[k] == b[k] for k in a}
    return {"pass": all(same.values()), "round": round_file, "equal": same,
            "worker_seconds": body.get("seconds"), "replay_seconds": secs, "calls": b["calls"]}


@contextlib.contextmanager
def _scoped_pin(sur):
    """Pin this process like the worker (saas-numpyro) for the refit, then restore every thread's
    affinity and TRIMEM_JAX_CPUS. JAX's thread pool is sized at its first use; when JAX was already
    used in this process (pin record jax_imported_before), the pool may differ from the worker's."""
    import os

    from trimem.experiments.trace_latent_search import pin_surrogate_cpus

    import torch

    if sur["kernel"] != "saas-numpyro":
        yield None
        return
    before = sorted(os.sched_getaffinity(0))
    env = os.environ.get("TRIMEM_JAX_CPUS")
    threads = torch.get_num_threads()
    try:
        yield pin_surrogate_cpus(sur)
    finally:
        torch.set_num_threads(threads)
        for tid in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(tid), before)
            except OSError:  # a thread that exited meanwhile
                pass
        if env is not None:
            os.environ["TRIMEM_JAX_CPUS"] = env


def _refit(arm, m, space, codec, round_file):
    """Re-fit the round's GP in this process (same archive, objective, seed, settings; for
    saas-numpyro the same warm state and CPU pin as the worker; nothing is written)."""
    from trimem.experiments.trace_latent_search import fit_round_model, resolve_surrogate
    from trimem.optimizer.latent_plan_surrogate import latent_snapshot
    from trimem.optimizer.plan_surrogate import ObjectiveProtocol, aggregate_panel

    body = _json(arm / (round_file + ".json"))
    with _ledger(arm) as led:
        by_key = {e["key"]: e for e in led.complete_configs()}
    entries = [by_key[k] for k in body["archive_keys"]]
    obs = [aggregate_panel(space, e["config"], e["receipts"], task_ids=m["tasks"],
                           protocol=m["protocol"], allow_synthetic=True,
                           allow_partial=bool(m.get("pruning")),
                           quality_scale=(m.get("pruning") or {}).get("quality_scale"))
           for e in entries]
    objective = ObjectiveProtocol(**_json(arm / "objective.json"))
    seed = m["seed"] + (len(entries) - INITIAL) // Q
    sur = resolve_surrogate(m["latent"])
    t0 = time.monotonic()
    with _scoped_pin(sur) as pin:
        model, record = fit_round_model(arm, round_file, codec, obs, objective, sur, seed=seed,
                                        save=False)
    fit_s = time.monotonic() - t0
    snap = latent_snapshot(codec, model, obs, objective, seed=seed)
    refit = {"round": round_file, "fit": record, "pin": pin,
             "worker_fit": (body.get("surrogate") or {}).get("fit")}
    return body, model, snap, obs, objective, fit_s, sur, refit


def _check_gp(arm, rounds, model, sur, u0):
    import torch

    from trimem.optimizer.latent_plan_surrogate import surrogate_info

    info = surrogate_info(model)
    with torch.no_grad():
        out_dim = int(model.posterior(torch.tensor([u0], dtype=torch.double)).mean.shape[-1])
    members = 3 if sur["structure"] == "independent" else 2
    per_round = []
    for r in rounds:
        state = torch.load(arm / (r["file"] + ".pt"), weights_only=False)["gp_state"]
        per_round.append(len({k.split(".")[1] for k in state if k.startswith("models.")}))
    ok = (out_dim == 3 and info["kernel"] == sur["kernel"]
          and info["structure"] == sur["structure"] and len(info["classes"]) == members
          and all(n == members for n in per_round))
    return {"pass": ok, "posterior_outputs": out_dim, "surrogate_info": info,
            "requested": {"kernel": sur["kernel"], "structure": sur["structure"]},
            "members_per_round_pt": per_round, "expected_members": members}


def _check_fd(snap, u0):
    import torch

    from trimem.optimizer.latent_plan_surrogate import acquisition_u

    u = torch.tensor([u0], dtype=torch.double, requires_grad=True)
    value = acquisition_u(snap, u).sum()
    (grad,) = torch.autograd.grad(value, u)
    g_ad = grad.reshape(-1)
    rows = {}
    for h in FD_STEPS:
        g = torch.zeros_like(g_ad)
        with torch.no_grad():
            for i in range(g.numel()):
                e = torch.zeros_like(u)
                e[0, i] = h
                g[i] = (acquisition_u(snap, u + e).sum() - acquisition_u(snap, u - e).sum()) / (2 * h)
        rel = float((g - g_ad).norm() / max(float(g_ad.norm()), 1e-12))
        rows[str(h)] = {"rel_err": rel, "max_abs_err": float((g - g_ad).abs().max())}
    rel = rows[str(FD_STEP)]["rel_err"]
    return {"pass": rel < FD_TOL, "u": list(u0), "ell": float(value.detach()), "step": FD_STEP,
            "tolerance": FD_TOL, "rel_err": rel, "autograd_norm": float(g_ad.norm()),
            "by_step": rows,
            "note": "central differences in double; qLogNEHVI with fixed Sobol QMC base "
                    "samples is deterministic and smooth (fat max/min), so no MC-kink "
                    "tolerance is used"}


def _check_rescoring(arm, rounds, codec, refit):
    """Every latent pick: decode(u_end) == pick, encode(pick) == u_canonical, canonical ell finite;
    in the re-fit round (if bit-identical) every pick's ell == pending-aware score at u(c)."""
    import torch

    bad, picks, projections = [], 0, 0
    for r in rounds:
        body = _json(arm / (r["file"] + ".json"))
        for p in body["projections"]:
            if p.get("status") == "rejected":
                continue
            projections += 1
            if p.get("ell_canonical") is None or p.get("u_canonical") is None:
                bad.append({"round": r["file"], "projection": p.get("key"), "why": "no canonical"})
        for j, s in enumerate(body["selected"]):
            if "latent" not in s["sources"]:
                continue
            picks += 1
            lat = s.get("latent") or {}
            dec = codec.decode(torch.tensor([lat.get("u_end")], dtype=torch.double))[0]
            u_c = codec.encode(s["config"])
            du = max(abs(a - b) for a, b in zip(u_c, lat.get("u_canonical") or [math.inf]))
            why = []
            if lat.get("key") != s["key"]:
                why.append("latent info key differs")
            if codec.key(dec) != s["key"]:
                why.append("decode(u_end) is not the pick")
            if not du <= U_TOL:
                why.append(f"encode(pick) differs from u_canonical by {du}")
            if lat.get("ell_canonical") is None:
                why.append("no canonical score")
            if why:
                bad.append({"round": r["file"], "slot": j, "key": s["key"], "why": why})
    body, snap, round_file = refit["body"], refit["snap"], refit["round_file"]
    identical = snap.identity == body["selected"][0]["snapshot"]
    exact: list | str = "not verified"  # Task 12 minor: no silent pass without exact evidence
    if identical:
        exact = []
        configs = [s["config"] for s in body["selected"]]
        for j, s in enumerate(body["selected"]):
            if s["ell"] is None:  # "fill" pick: unscored by design
                exact.append({"slot": j, "key": s["key"], "fill": True, "ok": "fill" in s["sources"]})
                continue
            sj = snap.with_pending(configs[:j]) if j else snap
            got = float(sj.score([s["config"]])[0])
            exact.append({"slot": j, "key": s["key"], "round_ell": s["ell"], "rescored": got,
                          "snapshot_match": sj.identity == s["snapshot"],
                          "ok": abs(got - s["ell"]) <= SCORE_TOL * max(1.0, abs(s["ell"]))})
    ok = (not bad and projections > 0 and identical and bool(exact) and all(e["ok"] for e in exact))
    return {"pass": ok, "latent_picks": picks, "projections_scored": projections,
            "violations": bad, "refit_round": round_file, "refit_bit_identical": identical,
            "exact_rescoring": exact,
            "rule": "latent pick: decode(u_end)==pick, |encode(pick)-u_canonical|_inf<=1e-5, "
                    "canonical ell finite; projections: canonical u and ell present; the refit "
                    "must be bit-identical and every pick's ell must equal its pending-aware "
                    "rescore (else 'not verified' and false)"}


def _source_mix(rounds) -> dict:
    contains: dict = {}
    exclusive: dict = {}
    for r in rounds:
        for p in r["picks"]:
            for s in p["sources"]:
                contains[s] = contains.get(s, 0) + 1
            tag = "+".join(sorted(p["sources"]))
            exclusive[tag] = exclusive.get(tag, 0) + 1
    return {"contains": contains, "combinations": exclusive,
            "latent_picks": sum(p["latent"] for r in rounds for p in r["picks"]),
            "picks": sum(len(r["picks"]) for r in rounds),
            "note": "a config can carry several sources; 'llm' = seeded stand-in agent "
                    "(no model call); 'anchor' = admitted refinement start"}


def _nhv(arm, m, space) -> dict:
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.optimizer.latent_plan_surrogate import measured_hv
    from trimem.optimizer.plan_surrogate import ObjectiveProtocol

    objective = ObjectiveProtocol(**_json(arm / "objective.json"))
    with _ledger(arm) as led:
        obs = driver.observations(led, space, m)
    init = [o for o in obs if space.key(o.config) in set(m["initial_keys"])]
    return {"init": measured_hv(init, objective) / 1.1**3,
            "final": measured_hv(obs, objective) / 1.1**3,
            "note": "synthetic objective; not evidence of improvement (spec 12.3)"}


# ------------------------------------------------------------------ main
def main(out: Path, *, space_run: Path = DEFAULT_SPACE_RUN, checkpoint: Path = DEFAULT_CHECKPOINT,
         kernel: str = "matern", structure: str = "independent", fast: bool = False,
         affine: Path | None = None, crash_round: int | None = None,
         proposal_budget: dict | None = None, cpus: int = 12, cpu_offset: int = 0,
         nuts: tuple | None = None, warm_warmup: int | None = None, pruning: bool = False,
         stage_step: int = 20, new: int | None = None, landscape: str = "structured") -> dict:
    """Run the smoke into a fresh ``out``; write and return ``smoke.json``.

    ``proposal_budget`` overrides the shape's {seconds, calls}; ``nuts`` / ``warm_warmup``
    override the saas-numpyro NUTS settings (fast mode only: the non-smoke validator requires the
    frozen 300 s / 10,000 calls and 512/256/16, 128). ``cpus`` / ``cpu_offset``: saas-numpyro pin."""
    import torch

    import tools.prepare_trace_latent_campaign as pl
    from trimem.optimizer.latent_plan_codec import file_sha256, load_codec
    from trimem.optimizer.plan_schema import digest

    out = Path(out)
    if (out / "shared_init").exists() or (out / "arm_latent").exists() or (out / "smoke.json").exists():
        raise FileExistsError(f"{out} already holds a smoke run; use a fresh --out")
    torch.set_num_threads(4)
    began = time.monotonic()
    shape = SHAPES[bool(fast)]
    crash_round = shape["crash_round"] if crash_round is None else crash_round
    out.mkdir(parents=True, exist_ok=True)
    src = _source(space_run)
    space = src["space"]
    ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if ck["domains"] != digest(space.domains):
        raise ValueError("checkpoint domain digest differs from the space run")
    affine_path, affine_meta = _affine(out, Path(checkpoint), space, affine)
    evaluate = _evaluator(space, landscape)

    t0 = time.monotonic()
    shared = _shared_init(out, src, evaluate)
    shared_s = time.monotonic() - t0
    block = pl.latent_block(Path(checkpoint).resolve(), affine_path.resolve(), ck["d"],
                            kernel=kernel, structure=structure, cpus=cpus, cpu_offset=cpu_offset)
    block["proposal_budget"] = dict(proposal_budget or shape["proposal_budget"])
    if nuts is not None:
        block["surrogate"]["nuts"] = [int(v) for v in nuts]
    if warm_warmup is not None:
        block["surrogate"]["warm_warmup"] = int(warm_warmup)
    new = shape["new"] if new is None else int(new)
    arm, m = _arm(out, src, shared, block, new, pruning=pruning, stage_step=stage_step)
    validators = _validators(m)
    # smoke runs use a provisional affine; the live-only frozen-R6 rule is the one allowed failure;
    # a short pruning smoke (--new != 40) may also fail the live episode budget only
    short = pruning and new != SHAPES[False]["new"]

    def allowed(k, v):
        return "smoke=False" in k and ("frozen-train-init-v1" in v
                                       or (short and "staged pruning budget" in v))

    if any(v != "ok" and not allowed(k, v)
           for k, v in validators.items() if not (fast and "smoke=False" in k)):
        raise ValueError(f"arm manifest failed validation: {validators}")
    with _ledger(arm) as led:
        imported = [e["key"] for e in led.complete_configs()]

    rounds, restart, final = _run_rounds(arm, m, space, evaluate, fast=fast,
                                         crash_round=crash_round)

    codec = load_codec(m["latent"]["checkpoint"], m["latent"]["affine"], space)
    if fast:
        _patch_codec(codec)
    refit_round = rounds[min(crash_round, len(rounds) - 1)]["file"]
    body, model, snap, obs_r, objective_r, refit_s, sur, refit = _refit(arm, m, space, codec, refit_round)
    agent_r = next(r["agent"] for r in rounds if r["file"] == refit_round)
    lat_pick = next((s for s in body["selected"] if s.get("latent")), None)
    u0 = list(lat_pick["latent"]["u_canonical"]) if lat_pick else codec.encode(body["selected"][0]["config"])
    batch, unique = _check_shapes(arm, m, rounds, imported)
    evidence = {
        "gp_three_models": _check_gp(arm, rounds, model, sur, u0),
        "fd_gradient_ok": {**_check_fd(snap, u0), "round": refit_round,
                           "u_source": "latent pick u_canonical" if lat_pick else "first pick u(c)"},
        "rescoring_used": _check_rescoring(arm, rounds, codec, {"body": body, "snap": snap,
                                                                 "round_file": refit_round}),
        "no_measured_or_pending_dup": _check_dups(arm, rounds),
        "batch_sizes": batch,
        "unique_configs": unique,
        "proposal_budget_respected": _check_budget(rounds, m["latent"]),
        "restart_consistent": restart or {"pass": False, "why": "crash round never reached"},
        "picks_equal_q": _check_picks(arm, rounds, Q),
        "total_calls_capped": _check_calls(arm, rounds, m["latent"]["proposal_budget"]["calls"]),
        "deterministic_replay": _check_replay(arm, m, space, codec, model, obs_r, objective_r, body,
                                              refit_round, agent_r or []),
    }
    names = CHECKS
    if m.get("pruning"):
        evidence["pruned_at_checkpoint"] = _check_pruning(arm, m, rounds)
        names = CHECKS + PRUNING_CHECKS
    checks = {k: bool(evidence[k]["pass"]) for k in names}
    report = _clean({
        "synthetic": True,
        "not_evidence_of_improvement": "spec §12.3",
        "checks": checks,
        "all_pass": all(checks.values()),
        "evidence": evidence,
        "settings": {
            "fast": bool(fast), "kernel": kernel, "structure": structure,
            "surrogate": m["latent"]["surrogate"], "space_run": str(space_run),
            "space_digest": digest(m["space"]), "domain_digest": digest(space.domains),
            "checkpoint": m["latent"]["checkpoint"],
            "checkpoint_sha256": m["latent"]["checkpoint_sha256"], "d": m["latent"]["d"],
            "affine": m["latent"]["affine"], "affine_sha256": file_sha256(affine_path),
            "affine_origin": affine_meta, "budget": m["budget"], "initial_configs": INITIAL,
            "q": m["q"], "proposal_budget": m["latent"]["proposal_budget"],
            "anchors": m["latent"]["anchors"], "h0": m["latent"]["h0"],
            "tasks": len(m["tasks"]), "crash_round": crash_round, "protocol": m["protocol"],
            "pruning": m.get("pruning") and {k: v for k, v in m["pruning"].items() if k != "order"},
            "new": new, "short_shape": short,
            "validators": validators,
            "evaluator": f"tools.latent_synthetic_smoke.{LANDSCAPES[landscape].__name__}",
            "llm": "seeded stand-in (init: space.sample seed+555; rounds: seed+9000+k)",
        },
        "rounds": rounds,
        "refit": refit,
        "source_mix": _source_mix(rounds),
        "final_status": final,
        "nhv": _nhv(arm, m, space),
        "seconds": {"shared_init": shared_s, "refit_for_checks": refit_s,
                    "total": time.monotonic() - began},
        "run_dirs": {"shared_init": str(shared), "arm": str(arm)},
    })
    _write(out / "smoke.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--space-run", type=Path, default=DEFAULT_SPACE_RUN)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--affine", type=Path, default=None,
                   help="affine JSON (default: copy of the checkpoint's provisional cache)")
    # live latent GP default (user 2026-11-04): Matern x 3 (saas-numpyro / saas: options)
    p.add_argument("--kernel", choices=("saas", "matern", "saas-numpyro"), default="matern")
    p.add_argument("--structure", choices=("independent", "coupled-cost"), default="independent")
    p.add_argument("--cpus", type=int, default=12, help="saas-numpyro: CPUs each worker pins")
    p.add_argument("--cpu-offset", type=int, default=0, help="saas-numpyro: first allowed CPU index")
    p.add_argument("--crash-round", type=int, default=None)
    p.add_argument("--pruning", action="store_true", help="staged pruning (live arm block)")
    p.add_argument("--stage-step", type=int, default=20)
    p.add_argument("--landscape", choices=tuple(LANDSCAPES), default="structured",
                   help="synthetic objective; 'rugged' = hash-random configs, 0/1 per-task reward")
    p.add_argument("--new", type=int, default=None,
                   help="exploration budget (episode budget = (10 + new) x tasks); default 40")
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    r = main(a.out, space_run=a.space_run, checkpoint=a.checkpoint, kernel=a.kernel,
             structure=a.structure, affine=a.affine, crash_round=a.crash_round, cpus=a.cpus,
             cpu_offset=a.cpu_offset, pruning=a.pruning, stage_step=a.stage_step, new=a.new,
             landscape=a.landscape)
    print(json.dumps({"all_pass": r["all_pass"], "checks": r["checks"]}, indent=2))
