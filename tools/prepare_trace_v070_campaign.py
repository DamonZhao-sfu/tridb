"""Prepare a TRACE v0.7.0 shared init and its latent-MOBO arm (plan 2026-11-05 section 7, Amendment A).

The hashed ``tools.prepare_trace_joint_campaign.prepare`` is reused unchanged; only its module attributes are
patched while a command runs (as ``prepare_trace_latent_campaign`` does):
``JointSpace := JointSpaceV070``, ``PROTOCOL_VERSION := "trace-v070-rank-optional-nomemory"``,
``make_panels := v070_make_panels`` (TRAIN panel file = the whole eligible pool with strata; the manifest's
``tasks`` = that pool; the ``v070.sampling`` block freezes the per-configuration stratified sampling rule),
``prepare_run`` / ``import_archive`` := the v0.7.0 versions below.

* ``shared-init``: budget 10 = ``sobol_then_llm`` (5 Sobol + 5 LLM) in the v0.7.0 space, each configuration on
  its own stratified sample; TEST deferred; no pruning; NoMemory TRAIN ledger prepared (``baselines`` block).
* ``arm`` (``latent-mobo-v1`` only; ``explicit-pr`` is refused): imports the 10 init configurations (same space,
  same samples) and the complete NoMemory TRAIN ledger; staged pruning on own prefixes (20..100, 5,000 episodes,
  <= 80 configurations); TEST on the TRAIN Pareto set U NoMemory (``tools.trace_v070_holdout``).

Usage:
  python -m tools.prepare_trace_v070_campaign shared-init --dataset alfworld --template T --run-dir R0 \\
      [--memory-db NAME] [--seed 20261022]
  python -m tools.prepare_trace_v070_campaign arm --backend latent-mobo-v1 --shared-init R0 --dataset alfworld \\
      --template T --run-dir R1 [--checkpoint results/trace_v070_seqvae_20261105/seqvae_v2_d16.pt] --affine A \\
      [--kernel matern] [--structure independent] [--panel stratified]
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import tools.prepare_trace_joint_campaign as base
import tools.prepare_trace_joint_campaign_stratified as strat
import tools.prepare_trace_latent_campaign as pl
from tools.trace_joint_live import ROOT, sha
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments import trace_latent_search as tls
from trimem.experiments.trace_v070_baseline import (
    baselines_block,
    import_baseline,
    prepare_baseline,
)
from trimem.experiments.trace_v070_ledger import SampledLedger, write_samples_json
from trimem.experiments.trace_v070_marker import (
    check_prepared,
    mark_prepared,
    set_aside_incomplete,
)
from trimem.experiments.trace_v070_pruning import latency_cap_block
from trimem.experiments.trace_v070_runtime import (
    PLAN_GUARD,
    REQUIRED_ARTIFACTS,
    SHARED_BACKEND,
    VERSION,
    is_v070,
    v070_patches,
    validate_v070_manifest,
)
from trimem.experiments.trace_v070_samples import POLICY, sampling_block
from trimem.optimizer import staged_pruning as sp
from trimem.optimizer.plan_schema import digest
from trimem.optimizer.plan_schema_v070 import JointSpaceV070

PROTOCOL_VERSION = "trace-v070-rank-optional-nomemory"
DEFAULT_CHECKPOINT = ROOT / "results/trace_v070_seqvae_20261105/seqvae_v2_d16.pt"
INITIAL, ARM_NEW, MAX_CONFIGS, STAGE_STEP = 10, 40, 80, 20
EXTRA = ("tools/trace_joint_stratified_panel.py", "tools/prepare_trace_latent_campaign.py",
         "tools/latent_latency_pruner_v070.py", "tools/latent_latency_pruner.py", "tools/trace_v070_router.py",
         "tools/latent_readiness_v070.py")
STATE = {"sampling": None, "latent": None}


def _add(manifest, paths) -> None:
    for p in paths:
        p = Path(p) if Path(p).is_absolute() else ROOT / p
        manifest["live"]["artifacts"][str(p.resolve())] = sha(p)


def v070_artifacts(manifest) -> None:
    _add(manifest, (*REQUIRED_ARTIFACTS, *EXTRA, strat.TYPES))


def v070_make_panels(pool, test_ids, *, seed, dataset):
    """(TRAIN panel = whole eligible pool with strata, TEST panel); freezes the sampling block in STATE."""
    types = json.loads(strat.TYPES.read_text())["datasets"][dataset]
    from tools.fixed_test_panel_guard import ids_digest

    if types["ids_sha256"] != ids_digest(test_ids):
        raise ValueError(f"TEST type file does not match the fixed {dataset} TEST panel")
    block, train = sampling_block(pool, test_ids, types["task_type"], seed=seed, dataset=dataset)
    STATE["sampling"] = block
    test = {"dataset": dataset, "split": "TEST", "tasks": list(test_ids),
            "selection": "canonical fixed_test_v1; never used for search"}
    return train, test


def _settings(manifest) -> None:
    manifest["selection"] = "greedy"
    manifest["live"]["whole_card_cap"] = "none"
    manifest["train_panel_policy"] = POLICY


def v070_block(sampling) -> dict:
    return {"version": VERSION, "plan_guard": PLAN_GUARD, "sampling": sampling,
            "nomemory_runtime": "same agent model / vLLM bases / c16 / AGENT_MAX_TOKENS / env vars as TRACE"}


def shared_prepare_run_v070(directory, manifest):
    """Shared init: 5 Sobol + 5 LLM on own samples, TEST deferred, no pruning, NoMemory TRAIN ledger."""
    if "pruning" in manifest:
        raise ValueError("the shared init runs no staged pruning (manifest has 'pruning')")
    _settings(manifest)
    manifest["holdout"]["selection"] = "deferred"
    manifest["search_backend"] = SHARED_BACKEND
    manifest["v070"] = v070_block((manifest.get("v070") or {}).get("sampling") or STATE["sampling"])
    manifest["baselines"] = baselines_block(manifest)
    if not manifest.get("smoke"):
        v070_artifacts(manifest)
    with v070_patches(manifest):
        frozen = driver.prepare_run(directory, manifest)
    prepare_baseline(directory, frozen, split="TRAIN")
    write_samples_json(directory, frozen)
    return frozen


def pruning_block(manifest) -> dict:
    size = int(manifest["v070"]["sampling"]["size"])
    return {
        "protocol": sp.PROTOCOL, "z": sp.Z, "sizes": pl.stage_sizes(size, STAGE_STEP),
        "order": "per-configuration own stratified sample prefixes (trace_v070_samples.config_sample)",
        "budget_episodes": manifest["budget"] * size,
        "protect": "initialization configurations (5 Sobol + 5 LLM); NoMemory is outside the ledger",
        "quality_scale": pl.quality_scale(manifest["fixed"]["dataset"]), "quality_se_floor": "agresti-coull",
        "comparators": "full: own-sample point estimates; stage: pessimistic corners on own prefixes (unpaired)",
        "plan_guard": PLAN_GUARD,
        # fix round 1: the latency cap is applied at the same stage barrier (no separate pruner process)
        "latency_cap": latency_cap_block(manifest["fixed"]["dataset"]),
    }


def arm_import_v070(directory, manifest, source_dir):
    """Arm manifest + import of the 10 init configurations and the NoMemory TRAIN ledger (same samples)."""
    source_dir = Path(source_dir)
    old = json.loads((source_dir / "manifest.json").read_text())
    if not is_v070(old):
        raise ValueError("the v0.7.0 arm imports a v0.7.0 shared init only (nothing from v0.6.0 runs)")
    if digest(manifest["space"]) != digest(old["space"]) or list(manifest["tasks"]) != list(old["tasks"]):
        raise ValueError("Arm space / TRAIN pool differ from the shared init")
    sampling = (manifest.get("v070") or {}).get("sampling") or STATE["sampling"]
    if digest(sampling) != digest(old["v070"]["sampling"]):
        raise ValueError("Arm sampling block differs from the shared init")
    _settings(manifest)
    manifest["holdout"]["selection"] = "train_pareto"
    manifest["search_backend"] = tls.BACKEND
    manifest["v070"] = v070_block(sampling)
    manifest["baselines"] = baselines_block(manifest, source="imported", imported_from=str(source_dir))
    manifest["pruning"] = pruning_block(manifest)
    manifest["budget"], manifest["new_configs"] = MAX_CONFIGS, MAX_CONFIGS - manifest["initial_configs"]
    lat = STATE["latent"]
    if not lat:
        raise ValueError("latent-mobo-v1 needs a latent block (checkpoint, affine)")
    if not manifest.get("smoke"):
        pl.check_frozen_affine(lat["affine"], source_dir)
        v070_artifacts(manifest)
        modules = sorted((ROOT / "trimem/optimizer").glob("latent_*.py"))
        _add(manifest, (lat["checkpoint"], lat["affine"], *modules, "trimem/experiments/trace_latent_search.py",
                        "tools/trace_latent_live.py"))
    manifest["latent"] = dict(lat)
    space = JointSpaceV070(old["space"]["domains"], execution_fingerprint=old["space"]["execution_fingerprint"])
    if digest(space.manifest()) != digest(old["space"]):
        raise ValueError("shared-init space manifest is not a JointSpaceV070 manifest")
    with SampledLedger(source_dir / "ledger.sqlite", old) as led:
        init = set(driver.initialization_keys(source_dir, old, space))
        entries = [e for e in led.complete_configs() if e["key"] in init]
    if len(entries) != INITIAL or len(init) != INITIAL:
        raise ValueError(f"shared init must have {INITIAL} complete initialization configurations")
    frozen = {**manifest, "objective": None, "initial_keys": [e["key"] for e in entries],
              "imported_observations": {"source_run": str(source_dir), "source_protocol": old["protocol"],
                                        "configs": len(entries), "rule": "the 10 v0.7.0 init configurations "
                                        "with their own samples; NoMemory TRAIN imported separately"}}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with v070_patches(frozen):
        driver.validate_manifest(frozen)
        with SampledLedger(directory / "ledger.sqlite", frozen) as led:
            for e in entries:
                receipts = [{**r, "protocol": frozen["protocol"],
                             "imported_from": {"run": str(source_dir), "protocol": r["protocol"],
                                               "config_key": r.get("config_key")}} for r in e["receipts"]]
                led.import_reserved(space, e["config"], receipts)
    driver.atomic_json(directory / "manifest.json", frozen)
    import_baseline(source_dir, directory, frozen)
    write_samples_json(directory, frozen)
    return frozen


@contextlib.contextmanager
def _patched(**attrs):
    saved = {k: getattr(base, k) for k in attrs}
    for k, v in attrs.items():
        setattr(base, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(base, k, v)


def common_patches(**extra):
    return _patched(JointSpace=JointSpaceV070, PROTOCOL_VERSION=PROTOCOL_VERSION, make_panels=v070_make_panels,
                    INITIAL=INITIAL, **extra)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("shared-init", "arm"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", choices=sorted(base.PROFILES), required=True)
        p.add_argument("--template", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--memory-db", default=None)
        p.add_argument("--seed", type=int, default=None)
        if name == "arm":
            p.add_argument("--backend", required=True, help="latent-mobo-v1 (explicit-pr is refused in v0.7.0)")
            p.add_argument("--shared-init", type=Path, required=True)
            p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="default: d16 (Q7)")
            p.add_argument("--affine", type=Path, required=True)
            p.add_argument("--kernel", choices=tls.KERNELS, default="matern")
            p.add_argument("--structure", choices=("independent", "coupled-cost"), default="independent")
            p.add_argument("--panel", choices=("stratified",), default="stratified",
                           help="per-configuration stratified samples (the only v0.7.0 policy)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    aside = set_aside_incomplete(args.run_dir)  # fix round 1: replay an interrupted preparation
    if aside is not None:
        print(json.dumps({"set_aside_incomplete_preparation": str(aside)}))
    if args.cmd == "shared-init":
        with common_patches(NEW=0, prepare_run=shared_prepare_run_v070):
            m = base.prepare(args.template, args.run_dir, args.dataset, seed=args.seed or 20261022,
                             memory_db=args.memory_db)
    else:
        if args.backend != tls.BACKEND:
            raise SystemExit(f"v0.7.0 arms use {tls.BACKEND} only; {args.backend!r} is refused")
        check_prepared(args.shared_init)  # fix round 1: the init's preparation completed
        src = json.loads((args.shared_init / "manifest.json").read_text())
        if args.seed not in (None, src["seed"]) or src["fixed"]["dataset"] != args.dataset:
            raise ValueError("--seed / --dataset differ from the shared-init run")
        from trimem.optimizer.latent_plan_codec import load_affine

        d = len(load_affine(args.affine)[0].lo)
        STATE["latent"] = pl.latent_block(args.checkpoint, args.affine, d, kernel=args.kernel,
                                          structure=args.structure)
        with common_patches(NEW=ARM_NEW, import_archive=arm_import_v070):
            m = base.prepare(args.template, args.run_dir, args.dataset, seed=src["seed"],
                             memory_db=args.memory_db, import_from=args.shared_init)
        tls.validate_latent_manifest(m)
    with v070_patches(m):
        driver.validate_manifest(m)
    validate_v070_manifest(m)
    mark_prepared(args.run_dir, m, kind=args.cmd)  # completion marker, written last
    print(json.dumps({"run_dir": str(args.run_dir), "dataset": args.dataset, "budget": m["budget"],
                      "search_backend": m.get("search_backend"), "train_pool": len(m["tasks"]),
                      "sample_size": m["v070"]["sampling"]["size"], "quota": m["v070"]["sampling"]["quota"],
                      "nomemory_key": m["baselines"]["NoMemory"]["key"],
                      "holdout_selection": m["holdout"]["selection"]}))


if __name__ == "__main__":
    main()
