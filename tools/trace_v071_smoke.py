"""CPU synthetic end-to-end smoke of TRACE v0.7.1: the ALFWorld mid-run path, q 4, at least one forced round.

Reuses the v0.7.0 smoke's tiny space, pool, receipts and tiny VAE (``tools.trace_v070_smoke``). Shared init on the old
SemFilter protocol (no ``v071`` block) -> arm with the recorded exemption and the stall rule -> rounds through the REAL
``guarded_search`` spawn of ``latent_search_worker_v071`` (fast nearest-config decode patched in the child) -> TEST.
Writes ``<out>/smoke_report.json``; exits 1 when a check is false.

    .venv-optimizer/bin/python -m tools.trace_v071_smoke --out results/trace_v071_smoke_20261105 --new 28
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

from tools import trace_v070_smoke as base

MAX_ROUNDS = 40
CHECKS = ("q_stays_4", "forced_round_present", "forced_all_picks_exploration", "no_acquisition_in_forced_round",
          "picks_conditioned_on_earlier_picks", "forced_only_at_patience", "counter_reset_after_forced",
          "replay_gives_same_flags", "hv_rounds_logged", "exemption_recorded", "init_receipts_old_protocol",
          "protocol_table_separates", "test_summary_has_nomemory", "hashed_files_unchanged")


def smoke_worker(directory, manifest, entries, agent, seed, round_file):
    """Spawn target: fast nearest-config decode in the child, then the real v0.7.1 worker."""
    from tools.latent_synthetic_smoke import _patch_codec
    from trimem.experiments.trace_v071_search import latent_search_worker_v071
    from trimem.optimizer import latent_plan_codec as lpc

    real = lpc.load_codec
    lpc.load_codec = lambda *a, **k: _patch_codec(real(*a, **k))
    try:
        latent_search_worker_v071(directory, manifest, entries, agent, seed, round_file)
    finally:
        lpc.load_codec = real


def run_arm(out: Path, sp, shared: Path, ck: Path, aff: Path, new: int) -> tuple[Path, dict, list]:
    import tools.prepare_trace_latent_campaign as pl
    import tools.prepare_trace_v070_campaign as prep
    import tools.prepare_trace_v071_campaign as p71
    from trimem.experiments import trace_joint_plan_search as driver
    from trimem.experiments.trace_v070_ledger import SampledLedger
    from trimem.experiments.trace_v070_runtime import _setattrs, v070_patches
    from trimem.experiments.trace_v071_runtime import v071_patches

    old = json.loads((shared / "manifest.json").read_text())
    arm = out / "arm_latent"
    m = {k: v for k, v in old.items() if k not in ("initial_keys", "objective", "imported_observations")}
    m.update(protocol="trace-v071-smoke-arm", budget=10 + new, new_configs=new, initialization=None,
             live={**old["live"], "run_dir": str(arm), "environment": {}}, fixed=dict(old["fixed"]))
    block = pl.latent_block(ck.resolve(), aff.resolve(), 4, kernel="matern")
    block["proposal_budget"] = {"seconds": 8, "calls": 2000}
    prep.STATE["latent"] = block
    m = p71.arm_v071(prep.arm_import_v070, arm, m, shared, old_protocol_init=True)
    timings = []
    with v071_patches(m), v070_patches(m), _setattrs([(driver, "_search_worker", smoke_worker)]):
        for k in range(MAX_ROUNDS):
            with SampledLedger(arm / "ledger.sqlite", m) as led:
                seen = {e["key"] for e in led.complete_configs()}
            t0 = time.time()
            status = driver.run_round(arm, evaluator=base.receipt, agent=sp.sample(4, seed=base.SEED + 9000 + k, avoid=seen))
            timings.append(round(time.time() - t0, 1))
            if status["status"] == "budget_complete":
                return arm, m, timings
    raise RuntimeError(f"arm did not reach budget_complete in {MAX_ROUNDS} rounds")


def checks(arm: Path, m: dict, summary: dict) -> tuple[dict, dict]:
    from tools.trace_v071_hash_guard import report as guard
    from trimem.experiments import trace_latent_search as tls
    from trimem.experiments import trace_v071_search as ts
    from trimem.experiments.trace_v070_ledger import SampledLedger
    from trimem.experiments.trace_v070_runtime import v070_patches
    from trimem.experiments.trace_v071_runtime import protocol_table
    from trimem.optimizer.plan_surrogate import ObjectiveProtocol

    tls.validate_latent_manifest(m)  # stock: q == 4
    rounds = [(p.stem, json.loads(p.read_text())) for p in sorted(arm.glob("round_[0-9][0-9][0-9].json"))]
    forced = [(n, b) for n, b in rounds if b.get("force_exploration")]
    objective = ObjectiveProtocol(**json.loads((arm / "objective.json").read_text()))
    with SampledLedger(arm / "ledger.sqlite", m) as led:
        entries = led.complete_configs()
    with v070_patches(m):
        replay = {n: ts.stall_state(arm, m, entries, objective, before=n) for n, _ in rounds}
        final = ts.stall_state(arm, m, entries, objective)
    same = all((replay[n]["force_exploration"], replay[n]["stagnant_rounds"]) == (b["force_exploration"], b["stagnant_rounds"])
               and abs(replay[n]["round_hv"] - b["round_hv"]) < 1e-12 for n, b in rounds)
    after_forced = [rounds[i + 1][1] for i, (_, b) in enumerate(rounds[:-1]) if b.get("force_exploration")]
    hv = [json.loads(x) for x in (arm / "hv_rounds.jsonl").read_text().splitlines()]
    init = set(m["initial_keys"])
    tags = {r.get("semfilter_protocol") for e in entries if e["key"] in init for r in e["receipts"]}
    table = protocol_table(arm, m)
    result = {
        "q_stays_4": m["q"] == 4 and all(len(b["selected"]) <= 4 for _, b in rounds)
        and all(len(b["selected"]) == 4 for _, b in rounds[:-1]),
        "forced_round_present": bool(forced),
        "forced_all_picks_exploration": bool(forced) and all(
            b["selection_rule"] == ts.FORCED_RULE and all(s["selection_rule"] == ts.FORCED_RULE for s in b["selected"])
            for _, b in forced),
        "no_acquisition_in_forced_round": bool(forced) and all(
            b["budget"]["scoring_calls"] == 0 and b["endpoints"] == [] and all(s["ell"] is None for s in b["selected"])
            for _, b in forced),
        "picks_conditioned_on_earlier_picks": bool(forced) and all(
            s["conditioning"] == "schur" for _, b in forced for s in b["selected"][1:]),
        "forced_only_at_patience": all(b["force_exploration"] == (b["stagnant_rounds"] >= 3) for _, b in rounds),
        "counter_reset_after_forced": bool(forced) and all(b["stagnant_rounds"] == 0 for b in after_forced),
        "replay_gives_same_flags": same,
        "hv_rounds_logged": [x["round"] for x in hv] == ["init", *[n for n, _ in rounds]]
        and all(x["includes_nomemory"] for x in hv),
        "exemption_recorded": (m["v071"]["init_protocol"] or {}).get("protocol") == "semfilter-v2",
        "init_receipts_old_protocol": tags == {"semfilter-v2"},
        "protocol_table_separates": all(("semfilter-v2" in v["protocols"]) == (v["source"] == "init")
                                        for v in table.values()),
        "test_summary_has_nomemory": any(r.get("baseline") == "NoMemory" for r in summary["results"]),
        "hashed_files_unchanged": guard()["ok"],
    }
    stats = {"rounds": len(rounds), "forced_rounds": [n for n, _ in forced], "final_stagnant": final["stagnant_rounds"],
             "hv": [round(x["hv"], 6) for x in hv], "forced_seconds": [round(b["seconds"], 3) for _, b in forced],
             "greedy_seconds": [round(b["seconds"], 3) for _, b in rounds if not b.get("force_exploration")]}
    return result, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--new", type=int, default=28)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; the smoke writes a fresh directory")
    a.out.mkdir(parents=True)
    torch.set_num_threads(4)
    from tools.trace_v070_holdout import run_pruned_holdout_v070

    t0, sp = time.time(), base.space()
    ck, aff = base.tiny_vae(a.out, sp)
    shared, _ = base.shared_init(a.out, sp)
    arm, m, timings = run_arm(a.out, sp, shared, ck, aff, a.new)
    with contextlib.redirect_stdout(sys.stderr):
        summary = run_pruned_holdout_v070(arm, evaluator=base.receipt)
    result, stats = checks(arm, m, summary)
    body = {"checks": result, **stats, "round_wall_seconds": timings, "seconds": time.time() - t0, "out": str(a.out)}
    (a.out / "smoke_report.json").write_text(json.dumps(body, indent=1, default=str))
    print(json.dumps({"checks": result, "rounds": stats["rounds"], "forced": stats["forced_rounds"],
                      "seconds": round(body["seconds"], 1)}))
    return 0 if all(result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
