"""Prepare a TRACE v0.7.1 shared init / latent arm: ``tools.prepare_trace_v070_campaign`` + SemFilter v2c + stall rule.

``prepare_trace_v070_campaign`` is hashed by the v0.7.0 ALFWorld init, so its options cannot change on disk; this wrapper
takes the same arguments plus:

* ``--semfilter {v2c,v2}``, default v2c (user 2026-11-05: ON for new preparations). ``v2`` = exactly the v0.7.0 command.
* ``--stall-rule {on,off}``, default on (arms only): the RAG-Stack stall rule of ``trace_v071_search``.
* ``--old-protocol-init`` (arms only): the MID-RUN switch. The shared init ran the stock SemFilter (TOPN 32, raw cards,
  no reuse) and exploration runs v2c (user 2026-11-05, ALFWorld). An explicit, recorded exemption, not a loosened check:
  the arm keeps the init's space manifest (same keys, same samples); ``fixed`` may differ from the init's by
  ``semfilter`` only (else refused); ``v071.init_protocol`` records it; each imported receipt is tagged
  ``semfilter_protocol = semfilter-v2`` and passes the unchanged ``driver.check_receipt`` before it enters the ledger.
  Without the flag a v0.7.1 arm imports a v0.7.1 init only. q stays 4 (stock ``validate_latent_manifest``).

While the command runs, the v0.7.0 module attributes ``memory_identity`` (adds ``fixed.semfilter``),
``shared_prepare_run_v070`` and ``arm_import_v070`` are wrapped; the wrappers add the ``v071`` block,
``semfilter_v071.ENVIRONMENT`` and the v0.7.1 modules to ``live.artifacts`` before the manifest is frozen.

Usage: python -m tools.prepare_trace_v071_campaign shared-init|arm <v0.7.0 arguments> [--semfilter v2c]
           [--stall-rule on] [--old-protocol-init]
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import tools.prepare_trace_joint_campaign as base
import tools.prepare_trace_v070_campaign as prep
from trimem.experiments import semfilter_v071 as v2c
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_v070_baseline as tb
from trimem.experiments.trace_v070_ledger import SampledLedger
from trimem.experiments.trace_v071_runtime import (
    REQUIRED_ARTIFACTS,
    block,
    is_v071,
    validate_v071_manifest,
)
from trimem.experiments.trace_v071_search import RULE
from trimem.optimizer.plan_schema import digest

OLD_PROTOCOL = "semfilter-v2"


def apply_v071(manifest: dict, *, stall: bool = False, init_protocol=None) -> None:
    manifest["v071"] = block(stall_rule=dict(RULE) if stall else None, init_protocol=init_protocol)
    manifest["fixed"].setdefault("semfilter", dict(v2c.BLOCK))
    manifest["live"]["environment"].update(v2c.ENVIRONMENT)
    if not manifest.get("smoke"):
        prep._add(manifest, REQUIRED_ARTIFACTS)
    validate_v071_manifest(manifest)


def _no_semfilter(fixed: dict) -> dict:
    return {k: v for k, v in fixed.items() if k != "semfilter"}


def _no_semfilter_env(env: dict) -> dict:
    return {k: v for k, v in env.items() if k not in v2c.ENVIRONMENT}


def init_exemption(old: dict, manifest: dict, source_dir) -> dict:
    """The recorded mid-run exemption; refused unless only the SemFilter protocol differs from the init."""
    if digest(_no_semfilter(manifest["fixed"])) != digest(old["fixed"]):
        diff = sorted(k for k in {*old["fixed"], *_no_semfilter(manifest["fixed"])}
                      if old["fixed"].get(k) != manifest["fixed"].get(k))
        raise ValueError(f"mid-run arm: fixed differs from the shared init beyond semfilter: {diff}")
    env = old["live"].get("environment") or {}
    return {"source_run": str(source_dir), "protocol": OLD_PROTOCOL,
            "top_n": int(env.get("TRIMEM_SEMFILTER_TOPN", 32)),
            "judge_input": env.get("TRIMEM_SEMFILTER_JUDGE_INPUT", "raw"), "reuse": "off",
            "configs": "the shared init's initialization configurations",
            "exploration_protocol": v2c.PROTOCOL,
            "space": "inherited from the shared init (same keys and samples); digest(fixed) differs by semfilter only",
            "rule": "user 2026-11-05: ALFWorld init on the old SemFilter, exploration rounds on semfilter-v2c"}


@contextlib.contextmanager
def _midrun_import_patches() -> Iterator[None]:
    """During the import only: tag + check imported receipts; NoMemory parity ignores the SemFilter settings only."""
    stock_import, stock_parity = SampledLedger.import_reserved, tb.parity_violations

    def import_reserved(self, space, config, receipts):
        key = space.key(config)
        tagged = [{**r, "semfilter_protocol": OLD_PROTOCOL} for r in receipts]
        for r in tagged:
            driver.check_receipt(r, {"key": key, "task_id": r["task_id"]}, self.manifest, space)
        return stock_import(self, space, config, tagged)

    def parity(run_manifest, bm):
        bad = stock_parity(run_manifest, bm)  # NoMemory runs no SemFilter
        if _no_semfilter_env(run_manifest["live"].get("environment") or {}) == _no_semfilter_env(
                bm["live"].get("environment") or {}):
            bad = [b for b in bad if b != "live.environment"]
        if digest(_no_semfilter(run_manifest["fixed"])) == digest(_no_semfilter(bm["fixed"])):
            bad = [b for b in bad if b != "fixed"]
        return bad

    SampledLedger.import_reserved, tb.parity_violations = import_reserved, parity
    try:
        yield
    finally:
        SampledLedger.import_reserved, tb.parity_violations = stock_import, stock_parity


def arm_v071(stock_arm, directory, manifest, source_dir, *, old_protocol_init=False, stall=True):
    """The v0.7.1 arm on top of ``arm_import_v070`` (also the smoke's entry)."""
    old = json.loads((Path(source_dir) / "manifest.json").read_text())
    if is_v071(old):
        if old_protocol_init:
            raise ValueError("--old-protocol-init: the shared init already runs semfilter-v2c")
        apply_v071(manifest, stall=stall)
        return stock_arm(directory, manifest, source_dir)
    if not old_protocol_init:
        raise ValueError("a v0.7.1 arm imports a v0.7.1 shared init; a v0.7.0 init needs --old-protocol-init")
    manifest["fixed"].setdefault("semfilter", dict(v2c.BLOCK))
    exemption = init_exemption(old, manifest, source_dir)
    manifest["space"] = old["space"]
    apply_v071(manifest, stall=stall, init_protocol=exemption)
    with _midrun_import_patches():
        return stock_arm(directory, manifest, source_dir)


@contextlib.contextmanager
def v071_prepare_patches(*, old_protocol_init=False, stall=True) -> Iterator[None]:
    stock = {"memory": base.memory_identity, "shared": prep.shared_prepare_run_v070, "arm": prep.arm_import_v070}

    def memory_identity(database, env):
        return {**stock["memory"](database, env), "semfilter": dict(v2c.BLOCK)}

    def shared(directory, manifest):
        apply_v071(manifest)
        return stock["shared"](directory, manifest)

    def arm(directory, manifest, source_dir):
        return arm_v071(stock["arm"], directory, manifest, source_dir, old_protocol_init=old_protocol_init, stall=stall)

    base.memory_identity, prep.shared_prepare_run_v070, prep.arm_import_v070 = memory_identity, shared, arm
    try:
        yield
    finally:
        base.memory_identity = stock["memory"]
        prep.shared_prepare_run_v070, prep.arm_import_v070 = stock["shared"], stock["arm"]


def split_options(argv: list) -> tuple[list, dict]:
    argv, opts = list(argv), {"semfilter": "v2c", "stall": True, "old_protocol_init": False}
    if "--old-protocol-init" in argv:
        argv.remove("--old-protocol-init")
        opts["old_protocol_init"] = True
    for flag, allowed in (("--semfilter", ("v2c", "v2")), ("--stall-rule", ("on", "off"))):
        if flag in argv:
            i = argv.index(flag)
            value = argv[i + 1]
            if value not in allowed:
                raise SystemExit(f"{flag} must be one of {allowed}, got {value!r}")
            argv = argv[:i] + argv[i + 2:]
            if flag == "--semfilter":
                opts["semfilter"] = value
            else:
                opts["stall"] = value == "on"
    return argv, opts


def main(argv=None) -> None:
    rest, opts = split_options(sys.argv[1:] if argv is None else argv)
    if opts["semfilter"] == "v2":
        if opts["old_protocol_init"]:
            raise SystemExit("--old-protocol-init needs --semfilter v2c")
        prep.main(rest)
        return
    with v071_prepare_patches(old_protocol_init=opts["old_protocol_init"], stall=opts["stall"]):
        prep.main(rest)


if __name__ == "__main__":
    main()
