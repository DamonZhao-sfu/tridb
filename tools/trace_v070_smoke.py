"""CPU synthetic end-to-end smoke of TRACE v0.7.0 (plan 2026-11-05 T12; Amendment A per-config samples).

Tiny v0.7.0 space (rank_impl ColBERT | none), a 600-task synthetic pool with four strata, per-configuration
stratified 100-task samples, synthetic receipts (no Postgres, no GPU, no LLM):

1. tiny sequence VAE v2 (CPU) + provisional affine;
2. shared init (5 Sobol + 5 seeded stand-in "LLM" configs) + NoMemory TRAIN through ``v070_patches``;
3. arm import (10 init + NoMemory), then >= 2 BO rounds through the REAL ``guarded_search`` spawn of
   ``latent_search_worker_v070`` (fast nearest-config decode patched in the child), staged pruning at every
   20 own tasks;
4. ``latent_latency_pruner_v070 --once``; 5. ``run_pruned_holdout_v070`` (Pareto U NoMemory).
Writes ``<out>/smoke_report.json``; exits 1 when a check is false.

    .venv-optimizer/bin/python -m tools.trace_v070_smoke --out results/trace_v070_smoke_20261105 --new 8 \\
        --stage-step 20
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "filter_order": ("none", "before", "after"), "graph_impl": ("BFS", "beam"), "filter_pre_impl": ("LLM",),
    "filter_post_impl": ("LLM",), "rank_impl": ("ColBERT", "none"), "retrieval_unit": ("whole",),
    "injection_timing": ("start", "periodic"), "seed_count": (8, 32), "depth": (2, 3), "fanout": (8,),
    "edge_threshold": ("off",), "beam_width": (2, 4), "sem_threshold_pre": (0.6,), "sem_threshold_post": (0.6,),
    "retrieval_interval": (5,), "returned_k": (3, 5), "trajectory_category": ("All",), "graph_direction": ("Both",),
}
STRATA = ("A", "B", "C", "D")
SEED, SIZE, MAX_ROUNDS = 3, 100, 40
HOLDOUT = [f"smoke_{5000 + i}" for i in range(200)]
CHECKS = ("space_v070", "rank_none_present", "no_llm_rank", "nomemory_train_complete",
          "nomemory_not_in_main_ledger", "nomemory_never_pruned", "nomemory_test_candidate",
          "test_summary_has_nomemory", "v070_worker_used", "hashed_files_unchanged",
          "per_config_samples_distinct", "samples_follow_test_mix", "prefixes_stratified",
          "staged_checkpoint_logged", "bo_rounds_ge_2", "latency_cap_at_stage_barrier", "decisions_durable",
          "test_set_has_pareto_and_fuzzy")


def space():
    from trimem.optimizer.plan_schema_v070 import JointSpaceV070

    return JointSpaceV070(TINY, execution_fingerprint="trace-v070-smoke")


def enumerate_configs(sp) -> list[dict]:
    from trimem.optimizer.plan_schema import FIELD_NAMES

    seen: dict = {}
    for combo in itertools.product(*(sp.domains[n] for n in FIELD_NAMES)):
        c = sp.canonicalize(dict(zip(FIELD_NAMES, combo)))
        seen.setdefault(sp.key(c), c)
    return list(seen.values())


def sampling():
    from trimem.experiments.trace_v070_samples import sampling_block

    items = [{"item_id": f"smoke_{i}", "task_key": f"smoke:{i}", "split": "train", "server": "official",
              "stratum": STRATA[min(3, (i % 10) // 3)]} for i in range(600)]
    types = {t: ("A", "B", "C", "C", "D")[i % 5] for i, t in enumerate(HOLDOUT)}
    return sampling_block({"dataset": "smoke", "items": items}, HOLDOUT, types, seed=SEED, dataset="smoke",
                          size=SIZE)


def receipt(config, task, manifest):
    from trimem.optimizer.plan_schema import digest

    j = int(digest([task])[:2], 16) % 9
    base = {"task_id": task, "split": manifest.get("evaluation_split", "TRAIN"), "protocol": manifest["protocol"],
            "synthetic": True, "agent_steps": 6}
    if config.get("baseline") == "NoMemory":
        return {**base, "raw_score": 0.1 * (j > 6), "total_tokens": 900.0 + j, "latency_seconds": 13.0 + j % 3,
                "config_key": manifest["key"]}
    rank = config["rank_impl"] != "none"
    beam = (config.get("beam_width") or 1) if config["graph_impl"] == "beam" else 1
    good = (config["seed_count"] == 32) + rank + (config["filter_order"] == "after")
    return {**base, "raw_score": float(j < 2 + 2 * good), "config_key": manifest_space_key(config),
            "total_tokens": 1500.0 * config["depth"] + 900.0 * rank + 200.0 * (config["filter_order"] != "none") + j,
            "latency_seconds": 12.0 + 12.0 * beam + 3.0 * config["depth"] + 4.0 * rank + j % 3}


def manifest_space_key(config):
    return space().key(config)


def smoke_worker(directory, manifest, entries, agent, seed, round_file):
    """Spawn target: fast nearest-config decode in the child, then the real v0.7.0 worker."""
    from tools.latent_synthetic_smoke import _patch_codec
    from trimem.experiments.trace_v070_runtime import latent_search_worker_v070
    from trimem.optimizer import latent_plan_codec as lpc

    real = lpc.load_codec
    lpc.load_codec = lambda *a, **k: _patch_codec(real(*a, **k))
    try:
        latent_search_worker_v070(directory, manifest, entries, agent, seed, round_file)
    finally:
        lpc.load_codec = real


def tiny_vae(out: Path, sp) -> tuple[Path, Path]:
    from trimem.optimizer.latent_plan_codec import file_sha256, fit_affine, save_affine
    from trimem.optimizer.latent_seq_grammar import SeqGrammar
    from trimem.optimizer.latent_seq_vae_v2 import save_checkpoint_v2, train_seq_vae_v2

    model, metrics = train_seq_vae_v2(sp, 4, n_samples=200, holdout=20, epochs=2, seed=0, layers=1, batch=64,
                                      log=lambda *_: None)
    ck = out / "seqvae_v2_d4_smoke.pt"
    save_checkpoint_v2(ck, model, sp, metrics)
    aff = out / "affine_smoke.json"
    save_affine(aff, fit_affine(model, SeqGrammar(sp), enumerate_configs(sp)[:120]),
                {"kind": "provisional-train-v1", "checkpoint_sha256": file_sha256(ck)})
    return ck, aff


def shared_init(out: Path, sp) -> tuple[Path, dict]:
    import tools.prepare_trace_v070_campaign as prep
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments.trace_v070_baseline import run_baseline
    from trimem.experiments.trace_v070_runtime import v070_patches

    block, _ = sampling()
    run = out / "shared_init"
    m = {"space": sp.manifest(), "tasks": sorted(block["strata"]), "protocol": "trace-v070-smoke-shared-init",
         "smoke": True, "fixed": {"dataset": "smoke", "synthetic": True}, "live": {"artifacts": {}},
         "initial_configs": 10, "new_configs": 0, "budget": 10, "q": 4, "seed": SEED, "task_concurrency": 16,
         "initialization": {"method": "sobol_then_llm", "sobol": 5, "llm": 5}, "sources": driver.source_identity(),
         "holdout": {"tasks": HOLDOUT, "selection": "deferred"}, "v070": {"sampling": block}}
    m = prep.shared_prepare_run_v070(run, m)

    def advice(s, obs, settings):
        seen = {s.key(o.config) for o in obs}
        return s.sample(5, seed=SEED + 555, avoid=seen), {"status": "synthetic-stand-in"}

    with v070_patches(m), mock.patch.object(driver, "initial_advice", advice):
        run_baseline(run, m, "TRAIN", receipt)
        while driver.run_round(run, evaluator=receipt)["status"] != "budget_complete":
            pass
    return run, m


def arm_rounds(out: Path, sp, shared: Path, ck: Path, aff: Path, new: int, step: int) -> tuple[Path, dict, int]:
    import tools.prepare_trace_latent_campaign as pl
    import tools.prepare_trace_v070_campaign as prep
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments.trace_v070_ledger import SampledLedger
    from trimem.experiments.trace_v070_runtime import _setattrs, v070_patches

    old = json.loads((shared / "manifest.json").read_text())
    arm = out / "arm_latent"
    m = {k: v for k, v in old.items() if k not in ("initial_keys", "objective", "imported_observations")}
    m.update(protocol="trace-v070-smoke-arm", budget=10 + new, new_configs=new, initialization=None,
             live={**old["live"], "run_dir": str(arm)})
    block = pl.latent_block(ck.resolve(), aff.resolve(), 4, kernel="matern")
    block["proposal_budget"] = {"seconds": 8, "calls": 2000}
    prep.STATE["latent"], prep.STAGE_STEP = block, step
    m = prep.arm_import_v070(arm, m, shared)
    rounds = 0
    with v070_patches(m), _setattrs([(driver, "_search_worker", smoke_worker)]):
        for k in range(MAX_ROUNDS):
            with SampledLedger(arm / "ledger.sqlite", m) as led:
                seen = {e["key"] for e in led.complete_configs()}
            agent = sp.sample(4, seed=SEED + 9000 + k, avoid=seen)
            status = driver.run_round(arm, evaluator=receipt, agent=agent)
            if status["status"] == "budget_complete":
                break
            rounds += 1
        else:
            raise RuntimeError(f"arm did not reach budget_complete in {MAX_ROUNDS} rounds")
    return arm, m, rounds


def hashed_unchanged() -> bool:
    from tools.trace_v070_hash_guard import report

    return report()["ok"]


def checks(arm: Path, m: dict, rounds: int, summary: dict) -> dict:
    from trimem.experiments.trace_v070_baseline import baseline_dir, baseline_entry
    from trimem.experiments.trace_v070_ledger import SampledLedger
    from trimem.experiments.trace_v070_samples import max_prefix_deviation

    block, nm_key = m["v070"]["sampling"], m["baselines"]["NoMemory"]["key"]
    with SampledLedger(arm / "ledger.sqlite", m) as led:
        bodies = [json.loads(b) for (b,) in led.db.execute("SELECT body FROM configs")]
        keys = [k for (k,) in led.db.execute("SELECT key FROM configs")]
        samples = led.samples()
        bnm = baseline_dir(arm)
    logs = [json.loads(x) for x in (arm / "pruning_log.jsonl").read_text().splitlines() if x.strip()]
    sel = json.loads((arm / "holdout/selection.json").read_text())
    markers = [json.loads(p.read_text()) for p in sorted(arm.glob("round_*.v070.json"))]
    nm = baseline_entry(arm, "TRAIN")
    with SampledLedger(bnm / "ledger.sqlite", json.loads((bnm / "manifest.json").read_text())) as led:
        nm_pruned = led.db.execute("SELECT count(*) FROM episodes WHERE state='pruned'").fetchone()[0]
    all_samples = list(samples.values()) + [nm["tasks"]]
    with SampledLedger(arm / "ledger.sqlite", m) as led:
        recorded = led.stage_decisions()
    logged = {(d["key"], int(d["stage"])) for d in logs if "stage" in d}
    durable = bool(recorded) and set(recorded) == logged
    return {
        "space_v070": m["space"]["schema"] == "trace-pr-v070-rank-optional-v1"
        and "none" in m["space"]["domains"]["rank_impl"],
        "rank_none_present": any(b["rank_impl"] == "none" for b in bodies),
        "no_llm_rank": all(b["rank_impl"] != "LLM" for b in bodies),
        "nomemory_train_complete": nm["complete"],
        "nomemory_not_in_main_ledger": nm_key not in keys,
        "nomemory_never_pruned": nm_pruned == 0 and all(d.get("key") != nm_key for d in logs),
        "nomemory_test_candidate": sel["baselines"]["NoMemory"]["candidate"] and nm_key in sel["candidate_keys"],
        "test_summary_has_nomemory": any(r.get("baseline") == "NoMemory" for r in summary["results"]),
        "v070_worker_used": len(markers) >= 2 and all(all(x["patched"].values()) for x in markers),
        "hashed_files_unchanged": hashed_unchanged(),
        "per_config_samples_distinct": len({frozenset(s) for s in all_samples}) == len(all_samples)
        and set(samples) == set(keys),
        "samples_follow_test_mix": all(dict(Counter(block["strata"][t] for t in s)) == block["quota"]
                                       for s in all_samples),
        "prefixes_stratified": max(max_prefix_deviation(block, s, m["pruning"]["sizes"]) for s in all_samples) <= 1,
        "staged_checkpoint_logged": any(d.get("stage") == m["pruning"]["sizes"][0] for d in logs),
        "bo_rounds_ge_2": rounds >= 2,
        "latency_cap_at_stage_barrier": any(d.get("prune_rule") == "latency-cap" for d in logs)
        and all(d["latency_cap"]["front"] and "NoMemory" not in d["latency_cap"]["front"]
                for d in logs if d.get("latency_cap")),
        "decisions_durable": durable,
        "test_set_has_pareto_and_fuzzy": (set(sel["pareto_keys"]) | set(sel["fuzzy_keys"])) - {nm_key}
        == set(sel["selected_keys"]) and sel["baselines"]["NoMemory"]["tested"]
        and set(sel["router_candidates"]) == set(sel["fuzzy_keys"]) | {nm_key},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--stage-step", type=int, default=20)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the smoke writes a fresh directory")
    a.out.mkdir(parents=True)
    torch.set_num_threads(4)
    from tools import latent_latency_pruner_v070 as lp
    from tools.trace_v070_holdout import run_pruned_holdout_v070

    t0, sp = time.time(), space()
    ck, aff = tiny_vae(a.out, sp)
    shared, _ = shared_init(a.out, sp)
    arm, m, rounds = arm_rounds(a.out, sp, shared, ck, aff, a.new, a.stage_step)
    latency = lp.run_once(arm, {}, factor=2.0, step=a.stage_step, dry=True)  # offline reporter only
    with contextlib.redirect_stdout(sys.stderr):
        summary = run_pruned_holdout_v070(arm, evaluator=receipt)
    result = {"checks": checks(arm, m, rounds, summary), "bo_rounds": rounds, "seconds": time.time() - t0,
              "latency_decisions": latency, "test_summary": summary, "out": str(a.out)}
    (a.out / "smoke_report.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({"checks": result["checks"], "bo_rounds": rounds, "seconds": round(result["seconds"], 1)}))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
