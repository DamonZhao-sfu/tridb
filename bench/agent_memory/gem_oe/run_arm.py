"""Run one (task, arm) cell of the agent-outcome experiment.

    python3 -m bench.agent_memory.gem_oe.run_arm \
        --task math:circle_packing --arm gem --out bench/out/oe/circle_packing_gem

One process, one task, one arm, one seed. Arms differ only in the `--arm` flag, which
selects the retriever; every other setting is frozen here so a diff between two cells
cannot come from a config drift. The frozen set is written into the receipt so that
claim is checkable rather than asserted.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
* No fitness aggregation across tasks. `combined_score` is normalised per task against
  that task's own benchmark constant, so averaging it across tasks ranks two different
  units against each other. Aggregation is a reporting step (classification rates, per
  AlphaEvolve), not something this runner should quietly do.
* No retry around retrieval. A retrieval outage that degrades arm B into arm A while
  reporting success is the failure that makes the whole comparison meaningless, so it
  propagates and kills the cell.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DSN = "postgresql://127.0.0.1:55432/evotrace_eg"
DEFAULT_SCOPE = "evotrace:349117b0"
DEFAULT_RAW = Path("data/evotrace/raw")

#: Frozen across every arm. Anything here that differed between two cells would be a
#: confound, so the receipt records the resolved values, not just the intent.
FROZEN = {
    "max_iterations": 100,
    "random_seed": 42,
    "diff_based_evolution": True,
    "num_top_programs": 3,
    "num_diverse_programs": 5,
    "num_islands": 5,
    "population_size": 1000,
    "archive_size": 100,
    "migration_interval": 50,
    "cascade_evaluation": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 16384,
}


def _load_openevolve() -> Any:
    """Import from the pinned OpenEvolve venv rather than whatever is on the path."""
    try:
        import openevolve  # noqa: F401
    except ImportError:  # pragma: no cover - environment wiring
        raise SystemExit(
            "openevolve is not importable. Run this with .venv-e0/bin/python, "
            "which is the interpreter that has openevolve 0.3.2 installed."
        ) from None
    from openevolve.config import Config
    from openevolve.controller import OpenEvolve

    return Config, OpenEvolve


def build_config(config_cls: Any, args: argparse.Namespace) -> Any:
    cfg = config_cls()
    cfg.max_iterations = FROZEN["max_iterations"]
    cfg.random_seed = FROZEN["random_seed"] if args.seed is None else args.seed
    cfg.diff_based_evolution = FROZEN["diff_based_evolution"]
    cfg.language = args.language
    cfg.checkpoint_interval = args.checkpoint_interval

    cfg.llm.api_base = args.llm_base
    cfg.llm.api_key = "EMPTY"
    cfg.llm.name = args.llm_model
    cfg.llm.temperature = FROZEN["temperature"]
    cfg.llm.top_p = FROZEN["top_p"]
    cfg.llm.max_tokens = FROZEN["max_tokens"]
    cfg.llm.models = []

    cfg.prompt.num_top_programs = FROZEN["num_top_programs"]
    cfg.prompt.num_diverse_programs = FROZEN["num_diverse_programs"]

    cfg.database.num_islands = FROZEN["num_islands"]
    cfg.database.population_size = FROZEN["population_size"]
    cfg.database.archive_size = FROZEN["archive_size"]
    cfg.database.migration_interval = FROZEN["migration_interval"]
    cfg.database.random_seed = cfg.random_seed

    # Off, and identical across arms. `cascade_evaluation` defaults to True and
    # SILENTLY degrades to direct evaluation when the evaluator defines no
    # `evaluate_stage1` -- only a log warning -- so two arms could differ in how many
    # candidates were ever fully scored without anything saying so.
    cfg.evaluator.cascade_evaluation = FROZEN["cascade_evaluation"]
    cfg.evaluator.timeout = args.eval_timeout
    cfg.evaluator.enable_artifacts = True

    # Emit the trace in EvoTrace's own schema so a completed run can be normalised
    # straight back into the Experience Graph.
    cfg.evolution_trace.enabled = True
    cfg.evolution_trace.format = "jsonl"
    cfg.evolution_trace.include_code = True
    cfg.evolution_trace.include_prompts = True
    cfg.evolution_trace.output_path = str(args.out / "evolution_trace.jsonl")
    return cfg


def build_retriever(args: argparse.Namespace) -> Any:
    from bench.agent_memory.gem_oe.memory_database import NullRetriever

    if args.arm == "none":
        return NullRetriever()

    from bench.agent_memory.gem_eg.store import EgStore
    from bench.agent_memory.gem_oe.retrievers import GemRetriever

    store = EgStore.connect(args.dsn)
    gem = GemRetriever(
        store=store,
        scope_id=args.scope,
        split=args.split,
        seed=args.seed_layer,
        m_seeds=args.m_seeds,
    )
    if args.arm == "gem":
        return gem
    if args.arm == "polyglot":
        from bench.agent_memory.gem_oe.retrievers import PolyglotRetriever

        raise SystemExit(
            "arm 'polyglot' needs a live Milvus+Neo4j+pgvector stack and a passing "
            "parity gate against GEM; wire the backend into PolyglotRetriever and "
            "run tools/evotrace/gate_polyglot_parity.py first. "
            f"({PolyglotRetriever.__name__} is written and ready.)"
        )
    raise SystemExit(f"unknown arm: {args.arm}")


def resolve_task(dsn: str, task_uid: str, raw: Path) -> dict[str, Any]:
    """Locate the evaluator and the seed program for this task."""
    import psycopg

    conn = psycopg.connect(dsn)
    row = conn.execute(
        "SELECT task_key, domain, specification_complete FROM gem_eg_task"
        " WHERE task_uid = %s",
        (task_uid,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"unknown task: {task_uid}")
    if not row[2]:
        raise SystemExit(
            f"task {task_uid} still has a stand-in specification; the ANN entry would "
            "rank over placeholder text. Re-run tools/evotrace/normalize.py."
        )

    evaluator = None
    for (run_rel,) in conn.execute(
        "SELECT run_rel FROM gem_eg_session WHERE task_uid = %s ORDER BY session_uid",
        (task_uid,),
    ).fetchall():
        candidate = (raw / run_rel / "evaluate.py").resolve()
        if candidate.is_file():
            evaluator = candidate
            break
    if evaluator is None:
        raise SystemExit(
            f"task {task_uid} ships no run-local evaluate.py; it cannot be scored here "
            "and is out of the experiment (see Phase 0 gate 2)."
        )

    # Seed program: the ROOT of the task's historical runs -- the program the corpus's
    # own sessions actually started from. Not "the worst attempt": a low-scoring
    # mid-run program is an artifact of one search trajectory, while the root is the
    # task's defined starting point and is what makes our runs comparable to the
    # recorded ones.
    #
    # Measured on the pinned revision: every root of a task is the SAME artifact (all
    # 8 roots of math:circle_packing share one sha256 and one score). That is asserted
    # rather than assumed -- if a task's roots diverge, "the root" is not well defined
    # and picking one silently would give different arms different starting points.
    roots = conn.execute(
        "SELECT n.node_uid, a.payload, n.fitness, n.artifact_uid"
        "  FROM gem_eg_node n JOIN gem_eg_artifact a ON a.artifact_uid = n.artifact_uid"
        " WHERE n.task_uid = %s AND n.parent_node_uid IS NULL"
        "   AND n.is_valid AND a.payload IS NOT NULL"
        " ORDER BY n.node_uid",
        (task_uid,),
    ).fetchall()
    if not roots:
        raise SystemExit(f"task {task_uid} has no root program with retained code")
    # One task in the pinned corpus -- math:second_autocorr_ineq -- has two roots,
    # because it carries a `strong_seed` ablation group that starts from a better
    # program (0.9677) than the default one (0.9549). That is a deliberate feature of
    # the dataset, not a defect, so the tie is broken by frequency rather than
    # rejected: the majority root is the task's ordinary starting point, and every
    # variant is recorded so the choice is auditable. A genuine tie IS rejected --
    # there would be no principled majority to take.
    counts: dict[str, int] = {}
    for row in roots:
        counts[row[3]] = counts.get(row[3], 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise SystemExit(
            f"task {task_uid} has {len(ranked)} equally common root artifacts "
            f"({[a for a, _ in ranked[:3]]}); 'the starting program' is not well "
            "defined, so arms would not share a starting point"
        )
    chosen_artifact = ranked[0][0]
    seed = next(r for r in roots if r[3] == chosen_artifact)
    return {
        "task_key": row[0],
        "domain": row[1],
        "evaluator": str(evaluator),
        "seed_node_uid": seed[0],
        "seed_code": seed[1],
        "seed_fitness": seed[2],
        "seed_artifact_uid": seed[3],
        "seed_root_count": len(roots),
        "seed_root_variants": [
            {"artifact_uid": a, "roots": n, "chosen": a == chosen_artifact}
            for a, n in ranked
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="e.g. math:circle_packing")
    ap.add_argument("--arm", required=True, choices=["none", "gem", "polyglot"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--scope", default=DEFAULT_SCOPE)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--llm-base", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--llm-model", default="qwen3.8")
    ap.add_argument("--language", default="python")
    ap.add_argument("--split", default="same_task", choices=["same_task", "cross_task"])
    ap.add_argument("--seed-layer", default="task", choices=["task", "node"])
    ap.add_argument("--m-seeds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None, help="overrides the frozen 42")
    ap.add_argument("--eval-timeout", type=int, default=900)
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything and write the receipt, run no iterations")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    config_cls, controller_cls = _load_openevolve()
    task = resolve_task(args.dsn, args.task, args.raw)
    cfg = build_config(config_cls, args)
    retriever = build_retriever(args)

    seed_path = args.out / "initial_program.py"
    seed_path.write_text(task["seed_code"], encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": "oe_arm_run_v0.1.0",
        "task_uid": args.task,
        "arm": args.arm,
        "retriever": retriever.name,
        "split": args.split,
        "seed_layer": args.seed_layer,
        "frozen": FROZEN,
        "resolved_seed": cfg.random_seed,
        "evaluator": task["evaluator"],
        "seed_node_uid": task["seed_node_uid"],
        "seed_fitness": task["seed_fitness"],
        "seed_artifact_uid": task["seed_artifact_uid"],
        "seed_root_count": task["seed_root_count"],
        "seed_root_variants": task["seed_root_variants"],
        "llm": {"base_url": args.llm_base, "model": args.llm_model},
        "scope": args.scope,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "dry_run" if args.dry_run else "running",
    }
    (args.out / "run_receipt.json").write_text(json.dumps(receipt, indent=2))
    if args.dry_run:
        print(json.dumps(receipt, indent=2))
        return 0

    from bench.agent_memory.gem_oe.memory_database import GemMemoryDatabase

    controller = controller_cls(
        initial_program_path=str(seed_path),
        evaluation_file=task["evaluator"],
        config=cfg,
        output_dir=str(args.out / "openevolve"),
    )
    # Swap BOTH references: the controller builds the Evaluator with
    # `database=self.database` in __init__, so replacing only `controller.database`
    # leaves the evaluator writing artifacts into the discarded instance.
    memory_db = GemMemoryDatabase(
        cfg.database, retriever=retriever, task_uid=args.task
    )
    controller.database = memory_db
    controller.evaluator.database = memory_db

    try:
        asyncio.run(controller.run())
        receipt["status"] = "complete"
    except BaseException as exc:  # noqa: BLE001 - the reason must reach the receipt
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        receipt["iterations_traced"] = len(memory_db.injection_trace)
        receipt["own_programs"] = len(memory_db.own_programs())
        receipt["injected_programs"] = len(memory_db.external_ids)
        (args.out / "run_receipt.json").write_text(json.dumps(receipt, indent=2))
        (args.out / "injection_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in memory_db.injection_trace),
            encoding="utf-8",
        )
        telemetry = getattr(retriever, "telemetry", None)
        if telemetry:
            (args.out / "retrieval_telemetry.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
