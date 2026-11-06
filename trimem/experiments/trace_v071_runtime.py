"""TRACE v0.7.1 = a v0.7.0 run + SemFilter v2c + the RAG-Stack stall rule (user 2026-11-05). Additive only.

Manifest: ``v071 = {"version": "0.7.1", "semfilter": semfilter_v071.BLOCK, "stall_rule": RULE | None,
"init_protocol": None | {...}}``, ``fixed.semfilter`` = the block and ``semfilter_v071.ENVIRONMENT`` in
``live.environment``. A manifest without ``v071`` is a plain v0.7.0 run and every patch here is a no-op.

Execution fingerprint: a v2c shared init and its arm have ``digest(fixed)`` incl. ``fixed.semfilter`` (new keys, new
samples). A MID-RUN arm (ALFWorld, user 2026-11-05: init on the old SemFilter, exploration on v2c) keeps its init's
space manifest, so the init configurations keep their keys and samples; ``v071.init_protocol`` records that explicit
exemption, ``tools.prepare_trace_v071_campaign`` checks that ``fixed`` differs from the init's by ``semfilter`` only, and
every imported receipt passes the unchanged ``check_receipt``. Protocol per receipt: ``semfilter_protocol``.

``v071_patches(manifest)``:
* ``tools.trace_joint_live.subprocess`` -> proxy: the TRACE episode command runs ``tools.trace_v071_episode_worker``;
* ``tools.trace_joint_live.receipt_from_files`` -> adds ``semfilter_protocol`` (fresh and recovered episodes);
* with a stall rule: ``trace_v070_runtime.latent_search_worker_v070`` -> ``latent_search_worker_v071`` (read by
  ``v070_patches`` when it installs the spawn target) and ``driver.run_round`` -> also rewrites ``hv_rounds.jsonl`` and
  ``semfilter_protocols.json`` after each completed round.
Fail-closed companion: ``TRIMEM_SEMFILTER_JUDGE_INPUT=compact-v2c`` is refused by the stock ``semfilter_v2.settings``, so
a v0.7.1 run started through the v0.7.0 launcher stops at its first SemFilter episode instead of mixing protocols.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path

from trimem.experiments import semfilter_v071 as v2c

VERSION = "0.7.1"
STOCK_WORKER = "tools.trace_joint_episode_worker"
WORKER = "tools.trace_v071_episode_worker"
REQUIRED_ARTIFACTS = (
    "trimem/experiments/semfilter_v071.py",
    "trimem/experiments/trace_v071_runtime.py",
    "trimem/experiments/trace_v071_search.py",
    "tools/trace_v071_episode_worker.py",
    "tools/trace_v071_live.py",
    "tools/prepare_trace_v071_campaign.py",
)


def block(*, stall_rule=None, init_protocol=None) -> dict:
    return {"version": VERSION, "semfilter": dict(v2c.BLOCK), "stall_rule": stall_rule, "init_protocol": init_protocol}


def is_v071(manifest) -> bool:
    return (manifest.get("v071") or {}).get("version") == VERSION


def semfilter_mode(manifest) -> str:
    """Which SemFilter protocol a v0.7.0-family manifest runs."""
    return v2c.PROTOCOL if is_v071(manifest) else "semfilter-v2"


def validate_v071_manifest(m) -> None:
    if not is_v071(m):
        raise ValueError(f"not a TRACE v{VERSION} manifest")
    if dict(m["v071"]["semfilter"]) != dict(v2c.BLOCK):
        raise ValueError("v071.semfilter differs from the semfilter-v2c protocol")
    if dict(m["fixed"].get("semfilter") or {}) != dict(v2c.BLOCK):
        raise ValueError("fixed.semfilter must equal v071.semfilter (execution fingerprint binding)")
    rule = m["v071"].get("stall_rule")
    if rule is not None:
        from trimem.experiments.trace_v071_search import RULE

        if dict(rule) != dict(RULE):
            raise ValueError(f"v071.stall_rule must be {RULE} or null")
    env = m["live"].get("environment") or {}
    bad = sorted(k for k, v in v2c.ENVIRONMENT.items() if env.get(k) != v)
    if bad:
        raise ValueError(f"live.environment differs from semfilter-v2c: {bad}")
    if not m.get("smoke"):
        root = Path(__file__).resolve().parents[2]
        arts = {str(Path(p).resolve()) for p in m["live"]["artifacts"]}
        missing = [p for p in REQUIRED_ARTIFACTS if str((root / p).resolve()) not in arts]
        if missing:
            raise ValueError(f"v0.7.1 modules missing from live.artifacts: {missing}")


class WorkerProxy:
    """``subprocess`` for ``tools.trace_joint_live``: only ``Popen`` differs, and only for the TRACE episode worker."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def Popen(self, cmd, *args, **kwargs):
        cmd = [WORKER if x == STOCK_WORKER else x for x in cmd]
        return self._real.Popen(cmd, *args, **kwargs)


def tag_receipt(stock_receipt):
    def receipt_from_files(*args, **kwargs):
        return {**stock_receipt(*args, **kwargs), "semfilter_protocol": v2c.PROTOCOL}

    return receipt_from_files


def protocol_table(run_dir, manifest) -> dict:
    """key -> SemFilter protocols of its receipts + its source (analyses separate init / exploration protocols)."""
    from trimem.experiments.trace_v070_ledger import SampledLedger

    init = set(manifest.get("initial_keys") or [])
    out = {}
    with SampledLedger(Path(run_dir) / "ledger.sqlite", manifest) as led:
        for e in led.complete_configs():
            found = {}
            for r in e["receipts"]:
                tag = r.get("semfilter_protocol", semfilter_mode(manifest))
                found[tag] = found.get(tag, 0) + 1
            out[e["key"]] = {"source": "init" if e["key"] in init else "explored", "protocols": found}
    return out


def after_round(run_dir, manifest) -> None:
    """hv_rounds.jsonl + semfilter_protocols.json after a completed round (pure functions of the ledger)."""
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments.trace_v070_ledger import SampledLedger
    from trimem.experiments.trace_v071_search import (
        stall_rule,
        stall_state,
        write_hv_rounds,
    )
    from trimem.optimizer.plan_surrogate import ObjectiveProtocol

    run_dir = Path(run_dir)
    driver.atomic_json(run_dir / "semfilter_protocols.json", protocol_table(run_dir, manifest))
    path = run_dir / "objective.json"
    if stall_rule(manifest) and path.exists():
        with SampledLedger(run_dir / "ledger.sqlite", manifest) as led:
            entries = led.complete_configs()
        objective = ObjectiveProtocol(**json.loads(path.read_text()))
        write_hv_rounds(run_dir, stall_state(run_dir, manifest, entries, objective))


@contextlib.contextmanager
def v071_patches(manifest) -> Iterator[None]:
    if not is_v071(manifest):
        yield
        return
    from tools import trace_joint_live as live
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments import trace_v070_runtime as rt

    validate_v071_manifest(manifest)
    saved = (live.subprocess, live.receipt_from_files, rt.latent_search_worker_v070, driver.run_round)
    live.subprocess = WorkerProxy(saved[0])
    live.receipt_from_files = tag_receipt(saved[1])
    if manifest["v071"].get("stall_rule") and manifest.get("search_backend") == rt.LATENT_BACKEND:
        from trimem.experiments.trace_v071_search import latent_search_worker_v071

        rt.latent_search_worker_v070 = latent_search_worker_v071

    def run_round(directory, *args, **kwargs):
        result = saved[3](directory, *args, **kwargs)
        if result.get("status") in ("batch_complete", "budget_complete"):
            after_round(directory, json.loads((Path(directory) / "manifest.json").read_text()))
        return result

    driver.run_round = run_round
    try:
        yield
    finally:
        live.subprocess, live.receipt_from_files, rt.latent_search_worker_v070, driver.run_round = saved
