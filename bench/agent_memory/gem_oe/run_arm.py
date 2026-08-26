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
    "random_seed": 42,
    "diff_based_evolution": True,
    "num_top_programs": 3,
    # 10, not OpenEvolve's default 2. This is the denominator of the injection rate:
    # the sweep needs 0% / 10% / 50% to land on whole programs, and 10 slots give
    # 0 / 1 / 5. It is frozen across every arm, so the no-memory arm renders the same
    # number of slots -- it just fills all of them from its own run.
    "num_diverse_programs": 10,
    "num_islands": 5,
    "population_size": 1000,
    "archive_size": 100,
    "migration_interval": 50,
    "cascade_evaluation": False,
    # 0.7, not 0.0. Deterministic decoding was chosen first, for reproducibility, and
    # it destroyed the experiment: with T=0 and five high-scoring programs in the
    # prompt, reproducing one verbatim IS the argmax. Measured on the abandoned run
    # (bench/out/oe/ABANDONED_phase2_T0_copycollapse_*): the memory arm emitted 15
    # distinct programs in 30 iterations against the no-memory arm's 30, and its
    # child code appeared verbatim in its own prompt 10 times out of 30. It jumped
    # once and then repeated itself.
    #
    # The corpus's own runs used 1.0 / 0.7 / 0.3 -- second_autocorr_ineq carries a
    # temperature ablation group -- so T=0 was never a normal setting for this kind
    # of evolutionary search. Reproducibility here comes from the fixed random_seed,
    # which OpenEvolve propagates into the LLM config, not from collapsing the
    # sampling distribution.
    "temperature": 0.7,
    # 1.0 was chosen alongside T=0; with sampling actually on, top_p=1.0 leaves the
    # full tail in play. 0.95 is OpenEvolve's own shape for a sampling run.
    "top_p": 0.95,
    # 8192, down from 16384. The serving replica is `--max-model-len 65536`, and the
    # prompt now carries up to 10 injected programs on top of the run's own context:
    # a 16k output reservation would push long prompts past the window and fail the
    # iteration rather than truncate it. The corpus's completions are p50 3,615 and
    # p90 10,473, so 8192 covers the bulk while leaving 57k for input.
    "max_tokens": 8192,
    # 40, down from 100. Measured on the corpus: circle_packing reaches 100% of its
    # final score by iteration 40, and the four low-headroom tasks are at >=98.7% by
    # iteration 10. The heilbronn pair is still climbing at 40, which is why the
    # budget is not cut further -- the primary metric is time-to-threshold and needs
    # resolution there. Iterations beyond 40 contributed nothing to any threshold
    # event in the v0.2.0 run.
    "max_iterations": 40,
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
    cfg.llm.temperature = FROZEN["temperature"]
    cfg.llm.top_p = FROZEN["top_p"]
    cfg.llm.max_tokens = FROZEN["max_tokens"]
    # `LLMEnsemble` indexes `models_cfg` directly, so an empty list is not "use the
    # top-level settings" -- it is `list index out of range` on the first generation,
    # logged once per iteration and otherwise survivable, which is how a whole 100
    # iteration run completed in 90 seconds without calling the model at all.
    from openevolve.config import LLMModelConfig

    cfg.llm.models = [
        LLMModelConfig(
            name=args.llm_model,
            api_base=args.llm_base,
            api_key="EMPTY",
            weight=1.0,
            temperature=FROZEN["temperature"],
            top_p=FROZEN["top_p"],
            max_tokens=FROZEN["max_tokens"],
            timeout=args.llm_timeout,
            retries=3,
        )
    ]
    cfg.llm.evaluator_models = list(cfg.llm.models)

    # `nocontext` is the true zero-reference lower bound: no inspirations AND no
    # top/previous programs. Without it, the "no memory" arm still renders up to
    # `num_top_programs` of the run's own output, so a comparison against it answers
    # "which SOURCE of references is better" and never "do references help at all".
    #
    # Set here, not by returning early -- an early return would skip the LLM,
    # evaluator and trace configuration below and produce a cell that never calls a
    # model, which is the failure this file already carries a warning about.
    zero_context = args.arm == "nocontext"
    cfg.prompt.num_top_programs = 0 if zero_context else FROZEN["num_top_programs"]
    cfg.prompt.num_diverse_programs = (
        0 if zero_context else (args.num_diverse or FROZEN["num_diverse_programs"])
    )
    # `changes` renders each inspiration's `changes_description` ("switched to greedy
    # sequential placement") instead of its source. Physically unable to be copied,
    # which is the point: the retrieval predicate hands back programs that are
    # strictly BETTER than what the agent holds, and with the code visible, copying
    # one is the agent's optimal move rather than a defect. Measured at 21.8% of
    # iterations on the first full pair. Injecting the edit instead is what the
    # Experience Graph paper means by reuse, and it is the same retrieval either way,
    # so the two modes isolate "reuse the answer" from "reuse the method".
    cfg.prompt.programs_as_changes_description = args.inject_as == "changes"
    if args.inject_as == "changes":
        # The mode is a two-sided contract, and the first attempt only honoured one
        # side: it changed what the prompt RENDERS, but OpenEvolve also requires the
        # model to return a diff against the parent's changes_description and
        # DISCARDS any program whose description was not updated
        # (process_parallel.py:246-252). With the seed carrying an empty description
        # there was nothing to diff against, so all 40 iterations of all 9 cells were
        # thrown away and every cell reported its seed score as its best.
        cfg.prompt.initial_changes_description = (
            "Initial program: the task's original starting implementation, unmodified."
        )

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

    if args.arm in ("none", "nocontext"):
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
        from bench.agent_memory.gem_oe.polyglot_backend import PolyglotBackend
        from bench.agent_memory.gem_oe.retrievers import PolyglotRetriever

        receipt_path = Path(args.polyglot_receipt)
        if not receipt_path.is_file():
            raise SystemExit(
                f"no polyglot load receipt at {receipt_path}. The E0 polyglot numbers "
                "were retracted because measurement began before the loader finished; "
                "run tools/evotrace/load_polyglot.py first."
            )
        return PolyglotRetriever(backend=PolyglotBackend(), gem=gem)
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
    ap.add_argument(
        "--arm", required=True,
        choices=["nocontext", "none", "gem", "polyglot"],
        help="nocontext: no references at all (true lower bound). none: the run's own "
             "programs only, no external memory. gem/polyglot: external memory.",
    )
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
    ap.add_argument("--polyglot-receipt", default="bench/out/polyglot/load_receipt.json")
    ap.add_argument(
        "--injection-rate", type=float, default=None,
        help="fraction of the inspiration slots to fill from memory, in [0,1]. "
             "With the frozen 10 slots, 0.1 -> 1 program and 0.5 -> 5. Omit for "
             "'fill every slot'. Arm `none` is the 0.0 point by construction, so "
             "this only means anything for a memory arm.",
    )
    ap.add_argument(
        "--inject-as", default="code", choices=["code", "changes"],
        help="what the prompt renders for each injected program: its source (`code`) "
             "or its edit description (`changes`). See build_config for why this is "
             "an experimental axis rather than a formatting preference.",
    )
    ap.add_argument(
        "--num-diverse", type=int, default=None,
        help="override the frozen inspiration-slot count. Changing it changes the "
             "injection-rate denominator, so it must be identical across every cell "
             "that will appear in one table.",
    )
    ap.add_argument(
        "--injection-policy", default="fixed", choices=["fixed", "match_baseline"],
        help="fixed: render up to num_diverse_programs retrieved programs regardless "
             "of how many the run itself offers. match_baseline: render exactly as "
             "many as arm A would, which makes prompt length identical but injects "
             "NOTHING while the islands are still sparse. See memory_database.py.",
    )
    ap.add_argument("--seed", type=int, default=None, help="overrides the frozen 42")
    ap.add_argument(
        "--eval-timeout", type=int, default=240,
        help="hard deadline for one evaluation, enforced by the subprocess wrapper. "
             "240s: the corpus's own evaluations peak at ~600s, but a generated "
             "program that runs longer than 240 is pathological rather than slow, "
             "and every second past the deadline is a worker not doing anything.",
    )
    ap.add_argument(
        "--no-wrap-evaluator", dest="wrap_evaluator", action="store_false",
        help="call the task's evaluator directly. Only for comparing against the "
             "unwrapped behaviour -- OpenEvolve's timeout does not stop a running "
             "evaluation, so an expensive generated program wedges the whole cell.",
    )
    ap.add_argument("--llm-timeout", type=int, default=1800,
                    help="the corpus p99 prompt is 66k tokens; a short timeout would "
                         "silently turn long prompts into failed iterations")
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

    # Wrap the evaluator so a runaway program can actually be killed. OpenEvolve's own
    # timeout cannot do it: `evaluator.py:351` dispatches with
    # `loop.run_in_executor(None, ...)` and bounds it with `asyncio.wait_for`, which
    # cancels the awaitable while the thread keeps running the evaluator forever. Two
    # such leaks exhaust the pool and the run goes silent -- observed on four tasks
    # across three attempts, always after exactly two timeouts, always reporting
    # `status: complete` with no data.
    evaluator_path = task["evaluator"]
    if args.wrap_evaluator:
        from tools.evotrace.wrap_evaluator import TEMPLATE

        wrapped = args.out / "wrapped_evaluator.py"
        wrapped.write_text(
            TEMPLATE.format(evaluator=evaluator_path, timeout=args.eval_timeout),
            encoding="utf-8",
        )
        evaluator_path = str(wrapped)

    slots = cfg.prompt.num_diverse_programs
    max_injected = (
        None if args.injection_rate is None
        else int(round(args.injection_rate * slots))
    )

    receipt: dict[str, Any] = {
        "schema_version": "oe_arm_run_v0.2.0",
        "task_uid": args.task,
        "arm": args.arm,
        "retriever": retriever.name,
        "split": args.split,
        "seed_layer": args.seed_layer,
        "injection_policy": args.injection_policy,
        "inject_as": args.inject_as,
        "injection_slots": slots,
        "injection_rate_requested": args.injection_rate,
        "max_injected": max_injected,
        "frozen": FROZEN,
        "resolved_seed": cfg.random_seed,
        "evaluator": task["evaluator"],
        "evaluator_wrapped": bool(args.wrap_evaluator),
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
        evaluation_file=evaluator_path,
        config=cfg,
        output_dir=str(args.out / "openevolve"),
    )
    # Swap BOTH references: the controller builds the Evaluator with
    # `database=self.database` in __init__, so replacing only `controller.database`
    # leaves the evaluator writing artifacts into the discarded instance.
    memory_db = GemMemoryDatabase(
        cfg.database,
        retriever=retriever,
        task_uid=args.task,
        policy=args.injection_policy,
        max_injected=max_injected,
    )
    controller.database = memory_db
    controller.evaluator.database = memory_db

    try:
        asyncio.run(controller.run())
        receipt["status"] = "complete"
        # Provisional: the real check is in the `finally` block, because
        # `controller.run()` returning is NOT proof the run happened. When an
        # evaluator times out, OpenEvolve stops without raising, and six cells
        # recorded `status: complete` after 5 to 16 of their 100 iterations. A
        # receipt that says "complete" for a 5-iteration run is worse than one that
        # says "failed": it silently enters the results table.
    except BaseException as exc:  # noqa: BLE001 - the reason must reach the receipt
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        traced = len(memory_db.injection_trace)
        receipt["iterations_traced"] = traced
        receipt["iterations_expected"] = cfg.max_iterations
        receipt["completion_fraction"] = round(traced / cfg.max_iterations, 4)
        if receipt["status"] == "complete" and traced < cfg.max_iterations * 0.9:
            receipt["status"] = "incomplete"
            receipt["error"] = (
                f"only {traced}/{cfg.max_iterations} iterations ran. OpenEvolve "
                "returned without raising -- an evaluator timeout stops the loop "
                "silently -- so this cell must not be read as a finished run."
            )
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
