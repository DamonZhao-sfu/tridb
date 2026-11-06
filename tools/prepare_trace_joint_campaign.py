"""Prepare a 5 Sobol + 5 LLM + 40 BO campaign for one dataset; never overwrite a run.

Usage:
  python -m tools.prepare_trace_joint_campaign --dataset alfworld \\
      --template results/trace_pr_three_dataset_prepare/template.json \\
      --run-dir results/trace_pr_alfworld_50_<date> [--memory-db NAME]

Each dataset fixes its TRAIN pool, environment servers, max_round and frozen
read-only memory clone. The runtime identity (model, max_round, memory clone,
evaluator source hashes) enters ``fixed`` and therefore the execution fingerprint.
"""

import argparse
import hashlib
import json
from pathlib import Path

from tools.fixed_test_panel_guard import MANIFEST, ids_digest, validate_ids
from tools.trace_joint_live import MEMORY_ENV, ROOT, memory_writes, runtime_identity, sha
from tools.trace_joint_protocol import make_panels
from trimem.experiments.trace_joint_plan_search import (
    atomic_json,
    prepare_run,
    source_identity,
)
from trimem.optimizer.plan_schema import JointSpace, digest

SCI_CLONE = "trace_ablation_sci_48d26f8d18eba20f"  # trimem_agentenv clone: ALFWorld + ScienceWorld
PROFILES = {
    "alfworld": {
        "pool": "data/external/panels/alfworld_pool.json",
        "ports": (36001, 36011),
        "max_round": 30,
        "action_format": "function_calling",
        "memory_db": SCI_CLONE,
    },
    "sciworld": {
        "pool": "results/scienceworld_trace_ablation_20261029_prepare/train_pool_excluding_validation.json",
        "ports": (36002, 36012),
        "max_round": 30,
        "action_format": "react",
        "memory_db": SCI_CLONE,
    },
    "webshop": {
        "pool": "data/external/panels/webshop_pool.json",
        "ports": (36003, 36013),
        "max_round": 10,  # AgentGym's own WebShop limit (user 2026-10-23)
        "action_format": "react",
        "memory_db": None,  # pass --memory-db trace_ablation_web_<hex>
    },
}
INITIAL, NEW, Q = 10, 40, 4
AGENT_MAX_TOKENS = 4096  # historical controlled-c16 runner argument (all datasets)
PROTOCOL_VERSION = "trace-pr-v060-engine-ext-channel-quota"


def memory_identity(database, env):
    """Identity of the frozen memory clone for one environment.

    ``memory_writes`` (row writes in all non-temporary tables, which include the
    graph store tables) is re-checked before every round, so any write to the
    clone after freezing stops the campaign. The database default must be
    read-only (checked on a connection without session options).
    """
    import psycopg

    dsn = (
        f"host=/tmp/trimem-traj port=55450 dbname={database} "
        "options='-c default_transaction_read_only=on'"
    )
    with psycopg.connect(dsn) as conn:
        read_only = conn.execute("SHOW default_transaction_read_only").fetchone()[0]
        oid = conn.execute(
            "SELECT oid FROM pg_database WHERE datname=current_database()"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT s.key, s.success, s.final_score FROM public.session s "
            "JOIN public.task t ON t.id=s.task_id WHERE t.env=%s ORDER BY s.key",
            (env,),
        ).fetchall()
    if read_only != "on":
        raise ValueError(f"Memory clone {database} is not read-only")
    writes = memory_writes(database)  # also checks the database-level default
    if not rows:
        raise ValueError(f"Memory clone {database} has no {env} sessions")
    h = hashlib.sha256()
    for key, success, score in rows:
        h.update(json.dumps([key, success, score], default=str).encode() + b"\n")
    return {
        "memory_database": database,
        "database_oid": int(oid),
        "sessions": len(rows),
        "read_only": True,
        "memory_snapshot": h.hexdigest(),
        "memory_writes": writes,
    }


def legacy_to_v060(config):
    """A configuration of the v0.5.0 space as a v0.6.0 configuration. The v0.5.0 label
    graph_direction=Forward executed edges similar+similar_up, i.e. v0.6.0 "Both"."""
    c = dict(config)
    if c.get("graph_direction") == "Forward":
        c["graph_direction"] = "Both"
    return c


def import_archive(directory, manifest, source_dir):
    """Seed a new run with every COMPLETE configuration of an earlier run (same dataset,
    TRAIN panel and runtime), mapped into this run's space. Receipts keep every measured
    value; their run identity is restated with explicit provenance (``imported_from``)."""
    from trimem.experiments.trace_joint_plan_search import atomic_json as aj
    from trimem.experiments.trace_joint_plan_search import get_space
    from trimem.optimizer.joint_search_ledger import SearchLedger

    source_dir = Path(source_dir)
    old = json.loads((source_dir / "manifest.json").read_text())
    if old["tasks"] != manifest["tasks"] or old["fixed"]["dataset"] != manifest["fixed"]["dataset"]:
        raise ValueError("Imported run must share the dataset and the TRAIN panel")
    with SearchLedger(source_dir / "ledger.sqlite", old) as led:
        entries = led.complete_configs()
    # The source space has its own (older) schema: identify its initialization from the
    # source ledger itself (Sobol keys + the LLM phase-B configurations), not by re-keying.
    old_init = set(old["initial_keys"])
    llm_file = source_dir / "initialization_llm.json"
    if old.get("initialization") and llm_file.exists():
        llm_cfgs = [json.dumps(r["config"], sort_keys=True) for r in json.loads(llm_file.read_text())["selected"]]
        old_init |= {e["key"] for e in entries if json.dumps(e["config"], sort_keys=True) in llm_cfgs}
    space = get_space(manifest)
    mapped, init_keys = [], []
    for e in entries:
        c = space.canonicalize(legacy_to_v060(e["config"]))
        key = space.key(c)
        if e["key"] in old_init:
            init_keys.append(key)
        receipts = []
        for r in e["receipts"]:
            receipts.append({**r, "protocol": manifest["protocol"], "config_key": key,
                             "imported_from": {"run": str(source_dir), "protocol": r["protocol"],
                                               "config_key": r["config_key"],
                                               "execution_fingerprint": r.get("execution_fingerprint")}})
        mapped.append((c, receipts))
    if len(init_keys) != manifest["initial_configs"]:
        raise ValueError("Imported archive lacks its complete initialization")
    frozen = {**manifest, "objective": None, "initial_keys": init_keys,
              "imported_observations": {"source_run": str(source_dir), "source_protocol": old["protocol"],
                                        "configs": len(mapped), "mapping": "v0.5.0 Forward == v0.6.0 Both",
                                        "excluded": "configurations without a complete 100-task panel"}}
    with SearchLedger(Path(directory) / "ledger.sqlite", frozen) as led:
        led.reserve(space, [c for c, _ in mapped])
        for c, receipts in mapped:
            key = space.key(c)
            for r in receipts:
                led.claim(key, r["task_id"])
                led.finish(key, r["task_id"], r)
    aj(Path(directory) / "manifest.json", frozen)
    return frozen


def prepare(template, directory, dataset, *, seed=20261022, memory_db=None, import_from=None):
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("New run directory required; previous experiments are immutable")
    profile = PROFILES[dataset]
    memory_db = memory_db or profile["memory_db"]
    if not memory_db:
        raise ValueError(f"{dataset} requires --memory-db")
    m = json.loads(Path(template).read_text())
    pool_path = ROOT / profile["pool"]
    pool = json.loads(pool_path.read_text())
    if dataset == "webshop":
        # User decision 2026-10-23 (option B): TRAIN and TEST share products;
        # only the 200 TEST goals are excluded; retrieval drops the query's own
        # product. Freeze that the pool was built under this rule.
        if pool.get("test_goal_overlap") != 0 or "option B" not in pool.get("product_rule", ""):
            raise ValueError("WebShop pool does not follow the recorded product rule")
    fixed_test = json.loads(MANIFEST.read_text())["datasets"][dataset]
    test_ids = fixed_test["query_ids"]
    validate_ids(dataset, test_ids)
    for shard in fixed_test["shards"]:
        if sha(ROOT / shard["path"]) != shard["sha256"]:
            raise ValueError(f"Canonical {dataset} TEST shard changed")
    train, test = make_panels(pool, test_ids, seed=seed, dataset=dataset)
    directory.mkdir(parents=True)
    atomic_json(directory / "train_panel.json", train)
    atomic_json(directory / "test_panel.json", test)
    panel_path = directory / "train_panel.json"
    environment = {
        k: v for k, v in m["live"]["environment"].items() if k not in MEMORY_ENV.values()
    }
    environment[MEMORY_ENV[dataset]] = memory_db
    old_directory = Path(m["live"]["run_dir"])
    artifacts = [
        Path(p) for p in m["live"]["artifacts"] if not Path(p).is_relative_to(old_directory)
    ]
    m["live"].update(
        run_dir=str(directory),
        panel_path=str(panel_path),
        panel_sha256=sha(panel_path),
        empty_baseline_dir=str(directory / "empty_baseline"),
        environment=environment,
        environment_bases=[f"http://127.0.0.1:{p}" for p in profile["ports"]],
        max_round=profile["max_round"],
        action_format=profile["action_format"],  # as in all historical TRACE runs
        agent_max_tokens=AGENT_MAX_TOKENS,
    )
    fixed = {
        **m["fixed"],
        **memory_identity(memory_db, dataset),
        "dataset": dataset,
        "panel_sha256": sha(panel_path),
        "test_ids_sha256": ids_digest(test_ids),
        "max_round": profile["max_round"],
        "action_format": profile["action_format"],
        "agent_max_tokens": AGENT_MAX_TOKENS,
        "concurrency": 16,
        "protocol_version": PROTOCOL_VERSION,
        **({"product_rule": pool["product_rule"]} if dataset == "webshop" else {}),
    }
    m["fixed"] = fixed
    fixed["runtime"] = runtime_identity(m)
    space = JointSpace.live(execution_fingerprint=digest(fixed))
    m.update(
        space=space.manifest(),
        protocol=directory.name + "-v1",
        seed=seed,
        smoke=False,
        objective=None,
        initial_configs=INITIAL,
        new_configs=NEW,
        budget=INITIAL + NEW,
        q=Q,
        task_concurrency=16,
        max_attempts=2,
        failure_breaker=3,
        initialization=None if import_from else {"method": "sobol_then_llm", "sobol": 5, "llm": 5},
        selection="channel-quota-v1",  # user 2026-11-03: one config per channel per round
        tasks=[r["item_id"] for r in train["items"]],
        sources=source_identity(),
        holdout={"tasks": test_ids, "selection": "train_pareto", "ids_sha256": ids_digest(test_ids)},
    )
    m.pop("initial_keys", None)
    atomic_json(directory / "execution_identity.json", fixed)
    artifacts += [
        panel_path,
        directory / "test_panel.json",
        directory / "execution_identity.json",
        pool_path,
        *[ROOT / p for p in m["sources"]],
        *[ROOT / p for p in fixed["runtime"]["evaluator_sources"]],
        ROOT / "tools/trace_joint_live.py",
        ROOT / "tools/trace_joint_protocol.py",
        ROOT / "tools/prepare_trace_joint_campaign.py",
        ROOT / "tools/fixed_test_panel_guard.py",
        ROOT / "trimem/optimizer/analyzer_pool.py",
        ROOT / "trimem/optimizer/analyzer_session.py",
        # engine extension v0.1.0 modules imported by the episode worker
        ROOT / "trimem/experiments/rank_impls.py",
        ROOT / "trimem/experiments/retrieval_units.py",
        ROOT / "scripts/serve_qwen3_reranker.sh",
        ROOT / "scripts/qwen3_reranker.jinja",
        *[ROOT / s["path"] for s in fixed_test["shards"]],
    ]
    m["live"]["artifacts"] = {str(p.resolve()): sha(p) for p in dict.fromkeys(artifacts)}
    (directory / "empty_baseline").mkdir()
    frozen = import_archive(directory, m, import_from) if import_from else prepare_run(directory, m)
    atomic_json(
        directory / "preparation.json",
        {
            "dataset": dataset,
            "template": str(Path(template).resolve()),
            "template_sha256": sha(template),
            "pool": str(pool_path),
            "pool_sha256": sha(pool_path),
            "train_episodes": (INITIAL + NEW) * 100,
            "test_episode_budget": "200 * final TRAIN Pareto count",
            "seed": seed,
            "train_eligible_count": train["eligible_count"],
            "initialization": frozen["initialization"],
            "exploration_batches": [Q] * (NEW // Q),
            "space_size": 2857680,
            "imported_from": str(import_from) if import_from else None,
        },
    )
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(PROFILES), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-db", default=None)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--import-from", type=Path, default=None,
                        help="earlier run of the same dataset whose complete configurations seed this run")
    args = parser.parse_args()
    m = prepare(args.template, args.run_dir, args.dataset, seed=args.seed, memory_db=args.memory_db,
                import_from=args.import_from)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "dataset": args.dataset,
                "initialization": m["initialization"],
                "budget": m["budget"],
                "train_tasks": len(m["tasks"]),
                "test_tasks": len(m["holdout"]["tasks"]),
                "memory": {k: m["fixed"][k] for k in ("memory_database", "sessions")},
            }
        )
    )


if __name__ == "__main__":
    main()
