"""Prepare a TRACE-PR campaign whose TRAIN panel follows the TEST task-type mix.

Same protocol as ``tools.prepare_trace_joint_campaign`` (5 Sobol + 5 LLM + 40 BO, frozen
memory clone, TEST200 on the TRAIN Pareto set), with three differences (user 2026-11-04):
* TRAIN panel: ``trace_joint_stratified_panel.make_stratified_panels`` (TEST type quotas).
* Batch selection: ``greedy`` (amendment 2026-11-03 of the v0.6.0c runs, set from round 0).
* Whole memory cards uncapped: ``live.whole_card_cap = "none"`` (amendment 2026-11-03).
* TEST deferred: the live driver stops after TRAIN (``holdout.selection = "deferred"``);
  ``tools.trace_pr_test_now`` then tests sample-mean Pareto U GP-posterior Pareto U GP
  OneConfig (user 2026-11-04, winner's-curse fix).
* Staged pruning: trimem.optimizer.staged_pruning (staged-pareto-prune-v2) on BO configurations
  at every 20 panel tasks; 5,000-episode budget kept, saved episodes buy configurations (<= 80).
Both amendments were applied mid-run to the v0.6.0c runs; here they hold from the start.

The original module is reused unchanged (its file is hashed by live runs); this wrapper
swaps its panel builder and adds the settings before the ledger is created. The wrapper,
the stratified-panel module and the frozen TEST type file enter ``live.artifacts``.

Usage:
  python -m tools.prepare_trace_joint_campaign_stratified --dataset webshop \\
      --template results/trace_pr_three_dataset_prepare/template_v060.json \\
      --run-dir results/trace_pr_webshop_v060_strat_20261104 [--memory-db NAME]
"""

import argparse
import json
from pathlib import Path

import tools.prepare_trace_joint_campaign as base
from tools.trace_joint_live import ROOT, sha
from tools.trace_joint_stratified_panel import make_stratified_panels

TYPES = ROOT / "triMEM_optimizer/test_panels/fixed_test_v1_task_types.json"
MAX_CONFIGS = 80  # = bo_pool_driver.MAX_CONFIGS (staged pruning budget cap)
EXTRA_ARTIFACTS = (
    TYPES,
    ROOT / "tools/trace_pr_test_now.py",
    ROOT / "tools/trace_pr_final_select.py",
    ROOT / "trimem/optimizer/staged_pruning.py",
    ROOT / "tools/trace_joint_stratified_panel.py",
    ROOT / "tools/prepare_trace_joint_campaign_stratified.py",
)


def stratified_make_panels(pool, test_ids, *, seed, dataset):
    types = json.loads(TYPES.read_text())["datasets"][dataset]
    from tools.fixed_test_panel_guard import ids_digest

    if types["ids_sha256"] != ids_digest(test_ids):
        raise ValueError(f"TEST type file does not match the fixed {dataset} TEST panel")
    return make_stratified_panels(pool, test_ids, types["task_type"], seed=seed, dataset=dataset)


_prepare_run = base.prepare_run


def stratified_prepare_run(directory, manifest):
    manifest["selection"] = "greedy"
    manifest["live"]["whole_card_cap"] = "none"
    manifest["train_panel_policy"] = "stratified-test-type-mix-v1"
    manifest["holdout"]["selection"] = "deferred"
    # Staged pruning (user 2026-11-04): reuse staged_pruning (staged-pareto-prune-v2) with
    # checkpoints every 20 panel tasks; the unpruned plan's episode budget is kept and saved
    # episodes buy more configurations, at most MAX_CONFIGS (as bo_pool_driver.MAX_CONFIGS).
    from trimem.optimizer import staged_pruning as sp

    panel = json.loads(Path(manifest["live"]["panel_path"]).read_text())["items"]
    stratum = {str(r["item_id"]): r["stratum"] for r in panel}
    tasks = [str(t) for t in manifest["tasks"]]
    manifest["pruning"] = {
        "protocol": sp.PROTOCOL,
        "z": sp.Z,
        "sizes": list(range(20, len(tasks) + 1, 20)),
        "order": sp.stage_order(tasks, stratum, sp.seed_of(manifest["protocol"], "stages")),
        "budget_episodes": manifest["budget"] * len(tasks),
        "protect": "initialization configurations (5 Sobol + 5 LLM)",
        # user 2026-11-04: quality SE floor = Agresti-Coull SE on the reward rescaled to [0, 1]
        "quality_scale": 100 if manifest["fixed"]["dataset"] == "sciworld" else 1,
        "quality_se_floor": "agresti-coull",
        "comparators": "full: point estimates; stage: pessimistic corners on the same prefix",
    }
    manifest["budget"] = MAX_CONFIGS
    manifest["new_configs"] = MAX_CONFIGS - manifest["initial_configs"]
    for p in EXTRA_ARTIFACTS:
        manifest["live"]["artifacts"][str(p.resolve())] = sha(p)
    return _prepare_run(directory, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(base.PROFILES), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-db", default=None)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    base.make_panels = stratified_make_panels
    base.prepare_run = stratified_prepare_run
    m = base.prepare(args.template, args.run_dir, args.dataset, seed=args.seed, memory_db=args.memory_db)
    panel = json.loads((Path(args.run_dir) / "train_panel.json").read_text())
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "dataset": args.dataset,
                "selection": m["selection"],
                "whole_card_cap": m["live"]["whole_card_cap"],
                "quota": panel["quota"],
                "uncovered_test_types": panel["uncovered_test_types"],
                "memory": {k: m["fixed"][k] for k in ("memory_database", "sessions")},
            }
        )
    )


if __name__ == "__main__":
    main()
