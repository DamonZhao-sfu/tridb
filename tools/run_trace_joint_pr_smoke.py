#!/usr/bin/env python3
"""Bounded production PR/backend smoke. Episode labels are explicitly synthetic.

Selected configurations execute the production session retrieval path against
a deterministic fixture graph; this is not a ScienceWorld quality experiment.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trimem.experiments.plan_execution_adapter import execute_plan
from trimem.experiments.trace_joint_plan_search import (
    atomic_json,
    import_initial,
    prepare_run,
    run_round,
    source_identity,
)
from trimem.optimizer.joint_search_controller import initial_configs
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.plan_codec import JointCodec
from trimem.optimizer.plan_schema import JointSpace


class FixtureAdapter:
    def __init__(self, category):
        self.policy = (
            "success-only-v0" if category == "Success" else "success+failure-v1"
        )

    def row(self, i):
        return {
            "id": i,
            "cosine": 1 - i / 1000,
            "success": i % 3 != 0,
            "score": (i % 11) / 10,
        }

    def allowed(self, row):
        return self.policy == "success+failure-v1" or row["success"]

    def seeds(self, count):
        return [self.row(i) for i in range(1, 193) if self.allowed(self.row(i))][:count]

    def neighbors(self, session, edge, fanout):
        offset = 1 if edge == "similar" else 7
        return [
            self.row((session["id"] * 3 + offset + j) % 251 + 1) for j in range(fanout)
        ]

    def hydrate(self, session):
        return session


def evaluate(config, task, manifest):
    began = time.perf_counter()
    receipt = execute_plan(
        config,
        FixtureAdapter(config["trajectory_category"]),
        judge=lambda cards: {c["id"]: c["score"] for c in cards},
        ranker=lambda cards: {c["id"]: c["score"] + c["cosine"] / 100 for c in cards},
    )
    duration = time.perf_counter() - began
    t = int(task)
    # Deliberately exercise negative quality and cost/quality trade-offs.
    score = -15 + min(
        100,
        12 * math.log2(config["seed_count"])
        + 5 * config["depth"]
        + t
        - 5 * config["returned_k"],
    )
    tokens = (
        100
        + config["seed_count"] * 8
        + config["fanout"] * config["depth"] * 10
        + config["returned_k"] * 90
        + t * 7
    )
    latency = 0.5 + tokens / 500 + (config["filter_order"] == "after") * 0.4 + t * 0.02
    return {
        "task_id": str(task),
        "split": "TRAIN",
        "protocol": manifest["protocol"],
        "synthetic": True,
        "raw_score": score,
        "total_tokens": tokens,
        "latency_seconds": latency,
        "retrieval_wall_seconds": duration,
        "retrieval": receipt,
        "quality_source": "synthetic_fixture",
        "config_key": JointSpace.live().key(config),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--initial-configs", type=int, default=10)
    parser.add_argument("--new-configs", type=int, default=4)
    parser.add_argument("--two-stage-init", action="store_true")
    args = parser.parse_args()
    directory = Path(args.out)
    if directory.exists():
        raise SystemExit(
            "Use a new smoke directory; historical artifacts are never overwritten"
        )
    space = JointSpace.live()
    manifest = {
        "space": space.manifest(),
        "tasks": [str(i) for i in range(4)],
        "protocol": "trace-pr-v042-fixture-smoke",
        "smoke": True,
        "budget": args.initial_configs + args.new_configs,
        "initial_configs": args.initial_configs,
        "new_configs": args.new_configs,
        "q": 4,
        "seed": 42,
        "task_concurrency": 4,
        "sources": source_identity(),
        "gp_maxiter": 50,
        "pr_init": "expected_changes_v1",
        "E": 1.5,
        "tau": 0.1,
        "fixed": {
            "vector": "task-ann",
            "edges": ["similar", "similar_up"],
            "judge": "fixture_scores",
            "cascade": "none",
            "pool_cap": 256,
            "expand_limit": 4096,
        },
    }
    if args.two_stage_init:
        from trimem.experiments import trace_joint_plan_search as driver

        if args.initial_configs != 10:
            raise ValueError("Two-stage initialization has exactly ten configurations")
        manifest["initialization"] = {"method": "sobol_then_llm", "sobol": 5, "llm": 5}
        prepare_run(directory, manifest)
        assert run_round(directory, evaluator=evaluate)["complete_configs"] == 5
        original = driver.initial_advice

        def fixture_advice(space, obs, settings):
            assert len(obs) == 5
            return space.sample(5, seed=43, avoid={space.key(o.config) for o in obs}), {
                "status": "synthetic_llm_fixture",
                "observed_train_configs": len(obs),
            }

        driver.initial_advice = fixture_advice
        try:
            assert run_round(directory, evaluator=evaluate)["complete_configs"] == 10
        finally:
            driver.initial_advice = original
    else:
        configs = initial_configs(space, seed=42, count=args.initial_configs)
        entries = [
            {
                "config": c,
                "receipts": [evaluate(c, t, manifest) for t in manifest["tasks"]],
            }
            for c in configs
        ]
        import_initial(directory, manifest, entries)
    # Inject a schema-valid analyzer proposal; no remote LLM is consumed in smoke.
    agent = [
        space.default(filter_order="before", seed_count=32, fanout=16, returned_k=8)
    ]
    selected_all = []
    while len(selected_all) < args.new_configs:
        result = run_round(directory, evaluator=evaluate, agent=agent)
        selected_all.extend(result["proposal"]["selected"])
    frozen = json.loads((directory / "manifest.json").read_text())
    with SearchLedger(directory / "ledger.sqlite", frozen) as ledger:
        complete = ledger.complete_configs()
        assert len(complete) == manifest["budget"] and not ledger.pending()
        assert sum(len(c["receipts"]) for c in complete) == manifest["budget"] * 4
    codec = JointCodec(space)
    for selected in selected_all:
        c = selected["config"]
        assert codec.decode(codec.encode(c)) == c
        assert "vector_impl" not in c and "failure_slots" not in c
    # Resume cannot generate or re-profile after budget completion.
    resumed = run_round(
        directory,
        evaluator=lambda *a: (_ for _ in ()).throw(
            AssertionError("duplicate dispatch")
        ),
    )
    assert resumed["status"] == "budget_complete"
    summary = {
        "status": "passed",
        "scope": "production PR/GP/controller/ledger + fixture retrieval; synthetic labels",
        "selected": selected_all,
        "complete_configs": manifest["budget"],
        "initialization": manifest.get("initialization", "imported fixture"),
        "episode_receipts": manifest["budget"] * 4,
        "resume": resumed,
        "resources": [
            json.loads(p.read_text())
            for p in sorted(directory.glob("round_*.resources.json"))
        ],
        "pr_runs": len(result["proposal"]["pr"]["runs"]),
        "unique_acquisition_inputs": result["proposal"]["unique_acquisition_inputs"],
        "pool_count": result["proposal"]["pool_count"],
        "proposal_seconds": result["proposal"]["seconds"],
    }
    atomic_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
