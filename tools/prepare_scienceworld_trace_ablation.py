"""Build a code-only draft manifest; never starts services or measurements."""

import hashlib
import json
from pathlib import Path

from trimem.experiments import scienceworld_trace_ablation as ab
from trimem.optimizer import session_space_agentenv_v2 as space

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/scienceworld_trace_ablation_20261029_prepare"
STATE = (
    ROOT
    / "results/exp2_sciworld_campaign_20261025/runs/sciworld_pool-strat200-v7-e2t610224f77fa__mem-sf1/state.json"
)
POOL = ROOT / "data/external/panels/sciworld_pool.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    state = json.loads(STATE.read_text())
    pool = json.loads(POOL.read_text())
    if pool["pool_hash"] != state["pool_hash"]:
        raise ValueError("historical state and current pool do not match")
    configs = state["rounds"][0]["configs"]
    if len(configs) != 10:
        raise ValueError(
            "historical initialization does not contain exactly ten configurations"
        )
    ids = {r["item_id"] for r in pool["items"]}
    test = {
        r["item_id"]
        for h in ("a", "b")
        for r in json.loads(
            (
                ROOT / f"results/eval_controlled_20261015/shards/sciworld_{h}.json"
            ).read_text()
        )
    }
    if len(test) != 200 or ids & test:
        raise ValueError("TRAIN pool and TEST IDs are not disjoint")
    initial_ids = set()
    for c in configs:
        sample = c["sample"]["ids"]
        if len(sample) != 100 or len(set(sample)) != 100 or not set(sample) <= ids:
            raise ValueError("invalid historical initialization task sample")
        initial_ids.update(sample)
    validation = ab.select_validation(pool["items"], initial_ids, test, size=50)
    train_pool = ab.training_pool_without_validation(pool, set(validation), initial_ids)
    manifest = ab.freeze_manifest(
        ab.protocol(),
        sha(STATE),
        sha(POOL),
        [space.key(c["config"]) for c in configs],
        validation,
        training_pool=train_pool,
    )
    ab.verify_manifest(manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "draft_manifest_v5.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise FileExistsError("refusing to overwrite a different manifest")
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    pool_target = OUT / "train_pool_excluding_validation.json"
    if pool_target.exists() and json.loads(pool_target.read_text()) != train_pool:
        raise FileExistsError("refusing to overwrite a different training pool")
    pool_target.write_text(json.dumps(train_pool, indent=1) + "\n")
    (OUT / "validation_ids.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(target),
                "sha256": manifest["manifest_sha256"],
                "test_task_ids": len(test),
                "pool_size": len(ids),
                "validation_tasks": len(validation),
                "training_pool_size": train_pool["size"],
                "runtime_identity_attested": False,
                "stage": "code_only",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
