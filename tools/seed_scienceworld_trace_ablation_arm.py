"""Seed one isolated ablation arm from a complete verified snapshot.

Default is read-only. `--commit` writes only new arm files and copied DB rows;
it never profiles episodes or starts/stops shared services.
"""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from tools.finalize_scienceworld_ablation_manifest import hash_artifact
from tools.scienceworld_ablation_source_inventory import verify_source_inventory
from trimem.experiments import bo_pool_driver as driver
from trimem.experiments import pool_branch
from trimem.experiments import scienceworld_trace_ablation as ab


def frozen_driver_arguments(arm, tag_prefix, *, stage="profile"):
    if stage not in ("profile", "router"):
        raise ValueError("unsupported ablation driver stage")
    tag = ab.arm_run_tags(tag_prefix)[arm]
    arguments = [
        stage,
        "--dataset",
        "sciworld",
        "--ablation_arm",
        arm,
        "--run_tag",
        tag,
        "--full_panel_size",
        "100",
        "--initial_budget_episodes",
        "3000",
        "--task_sample_namespace",
        "scienceworld-trace-ablation-v1",
        "--max_configs",
        "30",
        "--batch",
        "8",
        "--rounds",
        "3",
        "--stage_sizes",
        "20,40,60,80,100",
        "--seed",
        "20261022",
        "--search_design",
        "three-channel-v1",
        "--no_reuse",
    ]
    if arm == "without_pruning":
        arguments += ["--staging", "none"]
    return arguments


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "manifest",
        "snapshot",
        "source-state",
        "training-pool",
        "campaign-root",
        "artifact-paths",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--arm", choices=ab.ARMS, required=True)
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    ab.check_global_quota()
    ab.validate_ablation_environment(os.environ)

    root = args.campaign_root.resolve()
    if not root.name.startswith("scienceworld_trace_ablation_"):
        raise ValueError("campaign root must be a dedicated ablation directory")
    manifest = json.loads(args.manifest.read_text())
    snapshot = json.loads(args.snapshot.read_text())
    pool = json.loads(args.training_pool.read_text())
    source_bytes = args.source_state.read_bytes()
    source = json.loads(source_bytes)
    semfilter = (
        ab.verified_semfilter_v2(source, manifest) if "semfilter" in manifest else None
    )
    ab.verify_manifest(manifest)
    ab.verify_shared_initialization(snapshot)
    ab.verify_identity_evidence(snapshot, manifest["identity_evidence"])
    ab.validate_isolated_pool(pool, manifest)
    if (
        manifest.get("stage") != "seed_prepared"
        or manifest.get("shared_initialization_sha256") != snapshot["sha256"]
    ):
        raise ValueError(
            "manifest does not identify a verified complete initialization"
        )
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"]:
        raise ValueError("historical source state differs from frozen manifest")
    if ab._digest(sorted(snapshot["test_ids"])) != manifest.get("test_ids_sha256"):
        raise ValueError("frozen TEST identities mismatch")
    paths = json.loads(args.artifact_paths.read_text())
    verify_source_inventory(
        json.loads(Path(paths["source"]).read_text()),
        Path(__file__).resolve().parents[1],
    )
    if set(paths) != ab.FROZEN_ARTIFACTS or {
        k: hash_artifact(p) for k, p in paths.items()
    } != manifest.get("artifact_sha256"):
        raise ValueError(
            "frozen source, model, prompt, memory or other artifact changed"
        )
    tag = ab.arm_run_tags(args.tag_prefix)[args.arm]
    arguments = frozen_driver_arguments(args.arm, args.tag_prefix)
    parsed = driver.resolve_router_defaults(driver.build_parser().parse_args(arguments))
    ab.validate_driver_arguments(parsed)
    if not args.commit:
        print(
            json.dumps(
                {
                    "status": "validated_read_only",
                    "arm": args.arm,
                    "tag": tag,
                    "snapshot_sha256": snapshot["sha256"],
                    "candidate_episodes": 1000,
                    "note": "No files, database rows, or experiments were written.",
                },
                indent=2,
            )
        )
        return 0

    from tools.scienceworld_ablation_clone_memory import verify_campaign_clone

    verify_campaign_clone(root, manifest, snapshot, os.environ)
    os.environ[driver.RUN_TAG_ENV] = tag
    driver.OUT = root / "runs"
    driver.configure_ablation_pool(args.training_pool, manifest)
    if driver.mp.current() != driver.mp.SUCCESS_FAILURE:
        raise ValueError("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    driver.CONCURRENCY["sciworld"] = 16
    driver.TEST_CONCURRENCY["sciworld"] = 16
    os.environ[driver.RUNTIME_PROFILE_ENV] = driver.RUNTIME_PROFILE_C16
    state, state_path = driver.load_state("sciworld", pool, parsed)
    if state_path.exists():
        if state.get("ablation_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("existing arm state has a different manifest")
        with psycopg.connect(
            driver.drv.OPT_DSN + " options='-c default_transaction_read_only=on'",
            row_factory=dict_row,
        ) as connection:
            ab.verify_seeded_arm(state, snapshot, manifest, connection=connection)
        print(json.dumps({"status": "already_seeded", "state": str(state_path)}))
        return 0
    driver.concurrency_block("sciworld", state)
    driver.runtime_profile_block("sciworld", state)
    if semfilter is not None:
        state["semfilter"] = semfilter
    source_ids = [uuid.UUID(str(r["retrieval_id"])) for r in snapshot["receipts"]]
    with psycopg.connect(driver.drv.OPT_DSN, row_factory=dict_row) as connection:
        rows = connection.execute(
            f"SELECT {', '.join(pool_branch.ROW_COLUMNS)} FROM opt.session_retrieval WHERE retrieval_id = ANY(%s)",
            (source_ids,),
        ).fetchall()
        state = ab.seed_shared_arm(
            state,
            source["rounds"][0],
            snapshot,
            manifest,
            rows,
            campaign_root=root,
            target_dir=state_path.parent,
            source_pool_hash=source["pool_hash"],
            connection=connection,
        )
        connection.commit()
        ab.save_seeded_arm_state(state, state_path, campaign_root=root)
    print(json.dumps({"status": "seeded", "state": str(state_path), "arm": args.arm}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
