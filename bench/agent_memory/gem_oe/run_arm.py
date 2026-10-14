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
import os
import logging
import platform
import subprocess
import sys
import time
import zlib
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
    # 4096 for ALE, 8192 otherwise. The serving replica is `--max-model-len 65536`
    # and an ALE prompt already carries a 5-20k-character AtCoder problem statement
    # before any injection; with ten ~500-line C++ programs on top it reached 57,345
    # input tokens, and an 8192-token output reservation pushed the request over the
    # window (HTTP 400, four retries, iteration lost). Output is what gives way: the
    # problem statement is not negotiable and the injection count is the experiment.
    #
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
    # Same 10 slots on ALE as on math, which is only affordable because the serving
    # window was raised. At `--max-model-len 65536` ten AHC C++ programs (~5.7k
    # tokens each) drove the input alone to 61,441 tokens and 290 of ale:ahc008's
    # requests died on HTTP 400, while `nocontext` on the same task overflowed zero
    # times -- the programs were the cost, not the problem statement. Cutting ALE to
    # 4 slots fixed it (measured: 0 overflows, 4 judged) but made "100% injection"
    # mean a different number of programs per domain, which is not a comparison
    # anyone should have to caveat.
    #
    # The model's own limit is 262,144 (`text_config.max_position_embeddings`); the
    # 65,536 came from a serving flag set for a different workload. Re-served at
    # 131,072, so the slot count stays a single constant across domains.
    # 4 injected programs on ALE, 10 on math. Not a preference -- an AHC C++
    # program is ~23,000 characters at the corpus median, and ten of them keep the
    # prompt near the 131,072 window even with the run's own slots already zeroed.
    # Measured: 140 s/iteration at ten, which puts the 70-cell sweep at ~15 hours;
    # four brings it back to ~8 and leaves each injected program WHOLE, which
    # truncating them would not.
    #
    # Stated wherever ALE numbers appear: "100% injection" means four programs on
    # ALE and ten on math. Injection counts are comparable within a domain and
    # never across one -- and ALE and math scores are different units anyway.
    "num_diverse_programs_ale": 4,
    # 65536 chars, up from OpenEvolve's default 10000. The default silently REJECTS
    # a generated program before it is ever evaluated (process_parallel.py:282), and
    # an AHC C++ solution does not fit: the corpus's own ALE programs are p50 23,255
    # and max 57,079 characters. Measured on ale:ahc008 -- 52 of 25 iterations died
    # as "Generated code exceeds maximum length (20261 > 10000)" and only the seed
    # was ever judged, which reads in the results table as "the agent never improved"
    # rather than as "the agent's output was discarded unread".
    #
    # Applied to both domains, not just ALE. The math batch happened to hit it zero
    # times, but a cap that discards work without failing the run is a hazard whether
    # or not it fired this time.
    "max_code_length": 65536,
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
    # Overridable for smoke tests ONLY. Every reported cell uses the frozen value,
    # and the receipt records which was used so a short run cannot enter a table
    # unnoticed -- the completeness check below compares against this same number.
    cfg.max_iterations = args.max_iterations or FROZEN["max_iterations"]
    cfg.random_seed = FROZEN["random_seed"] if args.seed is None else args.seed
    cfg.diff_based_evolution = FROZEN["diff_based_evolution"]
    cfg.language = args.language
    cfg.checkpoint_interval = args.checkpoint_interval

    cfg.llm.api_base = args.llm_base
    cfg.llm.api_key = "EMPTY"
    cfg.llm.temperature = FROZEN["temperature"]
    cfg.llm.top_p = FROZEN["top_p"]
    is_ale = args.task.startswith("ale:")
    # One output budget for both domains. The ALE-only 4096 existed to claw back
    # room under a 65,536 window; at 131,072 a 61k ALE input plus 8192 output has
    # 60k to spare, so the special case would only make ALE completions shorter
    # than math's for no reason.
    cfg.llm.max_tokens = args.max_tokens or FROZEN["max_tokens"]  # override recorded in the receipt
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
            max_tokens=cfg.llm.max_tokens,
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
    # ALE renders 0 of the run's OWN top programs. The prompt otherwise carries up
    # to 26 full program bodies (3 previous + 13 top/diverse from the island + 10
    # injected), and an AHC C++ program is ~23,000 characters at the corpus median
    # -- 26 of them is ~187k tokens, past even the 131,072 window. Measured on
    # ale:ahc011, whose specification is the largest of the ten: 12 iterations
    # died on HTTP 400 with 122,881 input tokens.
    #
    # The earlier 65,536-window run only fit because `max_code_length` was 10,000
    # and silently truncated every program -- the very cap that was discarding
    # valid ALE work. Cutting the own-program slots instead keeps injected
    # programs whole, which is what the arms differ by. Applied to every ALE arm,
    # so the comparison stays consistent within the domain; ALE and math scores
    # are different units and are never compared across.
    cfg.prompt.num_top_programs = (
        0 if (zero_context or is_ale) else FROZEN["num_top_programs"]
    )
    default_diverse = (
        FROZEN["num_diverse_programs_ale"] if is_ale else FROZEN["num_diverse_programs"]
    )
    cfg.prompt.num_diverse_programs = (
        0 if zero_context else (args.num_diverse or default_diverse)
    )
    # External inspirations are converted to their historical change descriptions by
    # GemMemoryDatabase before the snapshot is rendered. Do not enable OpenEvolve's
    # similarly named flag: that flag also requires the model to edit the CURRENT
    # parent's changes_description and discards an otherwise valid code diff when it
    # does not. The two concerns are independent in this experiment: the treatment is
    # what external memory shows the agent, not a second output-format obligation.
    cfg.prompt.programs_as_changes_description = False
    if args.diff_ws_tolerant:
        from bench.agent_memory.gem_oe.diff_patch import enable

        enable()
    if args.no_simplify_nudge:  # OpenEvolve's default 'Consider simplifying - code length exceeds 500 characters'
        cfg.prompt.suggest_simplification_after_chars = None  # makes the LLM delete still-referenced C++ code
        cfg.prompt.code_length_threshold = None

    cfg.max_code_length = FROZEN["max_code_length"]
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


def _install_zero_context_gate() -> None:
    """Make a closed injection gate mean zero references, not own-run references.

    `sample_from_island` supplies only the prompt's `inspirations` section. Two more
    sections -- `previous_programs` and `top_programs` -- are sliced in the worker
    from `db_snapshot["islands"][parent_island]` (process_parallel.py:154-169) and
    cannot be reached from the database override. So without this hook a closed gate
    still rendered the run's OWN top and diverse programs, and p interpolated between
    the `none` arm and the memory arm rather than between `nocontext` and it.

    Emptying the island lists reproduces exactly what the `nocontext` arm achieves
    with `num_top_programs = num_diverse_programs = 0`: both slices come back empty.
    That key is read at one place in the worker and nowhere else, and the parent is
    resolved from `programs`, not from `islands`, so nothing else changes.

    Patched on the CLASS, not on an instance: `parallel_controller` is built inside
    `controller.run()` (controller.py:310), so at swap time there is no object to
    wrap. It receives `self.database` -- already the swapped memory database -- so
    the gate is read straight off it. One run_arm process runs one cell, so a
    class-level patch has no wider blast radius.
    """
    from openevolve.process_parallel import ProcessParallelController

    inner = ProcessParallelController._create_database_snapshot

    def snapshot(self: Any) -> dict[str, Any]:
        snap = inner(self)
        if not getattr(self.database, "last_gate_open", True):
            snap["islands"] = [[] for _ in snap["islands"]]
        return snap

    ProcessParallelController._create_database_snapshot = snapshot


#: Written by a matrix driver into its `--out` directory before any cell starts. A cell
#: inherits the policy of the matrix it belongs to, so a rerun into a running matrix
#: directory cannot silently switch generation policy half-way through the table.
GENERATION_POLICY_LOCK = "llm_generation_policy.json"
#: The only generation policy this runner implements: one OpenEvolve call per
#: iteration, no rescue attempt.
BASELINE_GENERATION_POLICY = "baseline"


def _locked_generation_policy(cell: Path) -> tuple[str | None, Path | None]:
    """The generation policy pinned by the nearest enclosing matrix lock, if any.

    Searches `cell` and then each ancestor for `llm_generation_policy.json` and returns
    `(policy, resolved_lock_path)` for the first one found, or `(None, None)` when no
    enclosing matrix pinned a policy. A lock that exists but cannot be read, or names
    no policy, raises: guessing a policy for a locked matrix is exactly the drift the
    lock exists to prevent.
    """
    for directory in (cell, *cell.parents):
        lock = directory / GENERATION_POLICY_LOCK
        if not lock.is_file():
            continue
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable generation-policy lock {lock}: {exc}")
        policy = payload.get("policy") if isinstance(payload, dict) else None
        if not isinstance(policy, str) or not policy:
            raise RuntimeError(f"generation-policy lock {lock} names no policy")
        return policy, lock.resolve()
    return None, None


def _require_beat_parent(args: argparse.Namespace) -> bool:
    """Whether retrieval keeps the beat-parent bound (the relational leg).

    On by default for every arm and plan -- it is the shipped behaviour, and the
    reference matrices ran with it. Only an explicit `--require-beat-parent false`
    turns it off, which is how the leg-ablation arms drop the relational leg while
    keeping every guard (see `retrievers.node_predicate`).
    """
    value = getattr(args, "require_beat_parent", None)
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "on"):
        return True
    if text in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"--require-beat-parent must be true or false, got {value!r}")


def resolve_task_trimem(dsn: str, task_uid: str) -> dict[str, Any]:
    """Evaluator + seed (majority corpus root) from the TriMEM experience graph."""
    import psycopg

    from trimem.replay.tasks import resolve

    with psycopg.connect(dsn) as conn:
        t = resolve(conn, task_uid)
    return {"task_key": t.key, "domain": task_uid.split(":")[0], "evaluator": str(t.evaluator),
            "seed_node_uid": None, "seed_code": t.seed_code, "seed_fitness": t.seed_fitness,
            "seed_artifact_uid": None, "seed_root_count": None, "seed_root_variants": None, "spec": t.spec}


def build_retriever(
    args: argparse.Namespace, source_anchor_uid: str | None = None
) -> Any:
    from bench.agent_memory.gem_oe.memory_database import NullRetriever

    if args.arm in ("none", "nocontext"):
        return NullRetriever()
    if args.arm == "trimem-adaptive":
        from bench.agent_memory.gem_oe.trimem_adaptive import TrimemAdaptiveRetriever, build_policy

        saved = Path(args.resume_from).parents[2] / "policy.pkl" if args.resume_from else None
        if saved is not None and saved.exists():  # continue the source cell's policy, posterior included
            import pickle

            policy = pickle.loads(saved.read_bytes())
        elif args.space in ("v2", "v3"):
            from bench.agent_memory.gem_oe.trimem_adaptive import build_policy_v2

            policy = build_policy_v2(args.policy, args.policy_mode, prior_path=args.policy_prior,
                                     seed=zlib.crc32(f"v2|{args.seed if args.seed is not None else FROZEN['random_seed']}"
                                                     f"|{args.task}{'|' + args.resume_from if args.resume_from else ''}".encode()),
                                     epsilon=args.policy_epsilon, lam_t=args.lambda_t, tol=args.o3_tol,
                                     space=args.space, lam_l=args.lambda_l, out_budget=args.out_budget)
            policy.fixed_name = args.fixed_config or ""
        else:
            policy = build_policy(args.policy, args.policy_mode, prior_path=args.policy_prior,
                                  # per (seed, task[, resume point]): a shared seed would repeat one choice sequence
                                  seed=zlib.crc32(f"{args.seed if args.seed is not None else FROZEN['random_seed']}|{args.task}"
                                                  f"{'|' + args.resume_from if args.resume_from else ''}".encode()),
                                  epsilon=args.policy_epsilon, lam_t=args.lambda_t, lam_a=args.lambda_a)
        return TrimemAdaptiveRetriever(
            dsn=args.trimem_dsn, policy=policy, session=args.out.name,
            horizon=args.policy_horizon,
            prompt_version=args.trimem_prompt_version or f"oe-{args.out.name}",
            telemetry_path=args.out / "llm_telemetry.jsonl", require_changes=args.inject_as == "changes")
    if args.arm == "trimem":
        if getattr(args, "trajectory_store", False):
            from bench.agent_memory.gem_oe.trajectory_retriever import TrajectoryRetriever

            return TrajectoryRetriever(
                dsn=args.trimem_dsn, plan=args.trajectory_plan, out=args.out,
                total_token_budget=args.trajectory_total_token_budget,
                objective=args.trajectory_objective,
                memory_tokens=args.trajectory_memory_tokens,
                include_code=args.trajectory_include_code,
            )
        from bench.agent_memory.gem_oe.trimem_retriever import TrimemRetriever

        return TrimemRetriever(dsn=args.trimem_dsn, mode=args.trimem_mode, force=args.trimem_force,
                               prompt_version=args.trimem_prompt_version or f"oe-{args.out.name}",
                               require_changes=args.inject_as == "changes", graph=args.trimem_graph)

    from bench.agent_memory.gem_eg.store import EgStore

    store = EgStore.connect(args.dsn)
    common: dict[str, Any] = {
        "store": store,
        "scope_id": args.scope,
        "split": args.split,
        "seed": args.seed_layer,
        "require_beat_parent": _require_beat_parent(args),
        "m_seeds": args.m_seeds,
        "embedding_endpoint": args.embedding_endpoint,
        "embedding_model": args.embedding_model,
        "query_language": args.language,
    }
    native_plan = getattr(args, "native_fused_plan", None)
    if args.physical_plan is not None:
        # The ONLY place the Python physical executor enters a cell, and only when
        # asked for: the formal native path below never imports it (ADR-0023).
        from bench.agent_memory.gem_oe.physical_retriever import PhysicalGemRetriever

        gem = PhysicalGemRetriever(**common, physical_plan=args.physical_plan)
    else:
        from bench.agent_memory.gem_oe.retrievers import GemRetriever

        gem = GemRetriever(
            **common,
            native_fused_plan=native_plan,
            source_anchor_uid=source_anchor_uid
            if native_plan == "source_anchored"
            else None,
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

    if args.arm == "cognee":
        from bench.agent_memory.gem_oe.retrievers import CogneeRetriever

        index = Path(args.cognee_index)
        if not index.exists():
            # Same rule as the polyglot receipt above: a memory arm that starts
            # before its store finished loading measures an empty index and scores
            # it as "memory had nothing useful".
            raise SystemExit(
                f"no cognee index at {index}; run tools/evotrace/cognee_eg.py ingest "
                "in the cognee virtualenv first"
            )
        return CogneeRetriever(endpoint=args.cognee_endpoint, gem=gem)
    raise SystemExit(f"unknown arm: {args.arm}")


def resolve_task(dsn: str, task_uid: str, raw: Path) -> dict[str, Any]:
    """Locate the evaluator and the seed program for this task."""
    import psycopg

    conn = psycopg.connect(dsn)
    task_row = conn.execute(
        "SELECT task_key, domain, specification_complete FROM gem_eg_task"
        " WHERE task_uid = %s",
        (task_uid,),
    ).fetchone()
    if task_row is None:
        raise SystemExit(f"unknown task: {task_uid}")
    if not task_row[2]:
        raise SystemExit(
            f"task {task_uid} still has a stand-in specification; the ANN entry would "
            "rank over placeholder text. Re-run tools/evotrace/normalize.py."
        )

    # ALE tasks get a generated evaluator. The corpus ships one per run, but it
    # imports `benchmarks.ale_bench.ale_session_helper` from SkyDiscover, which was
    # never published; the scoring contract is short enough to restate against
    # `ale_bench` directly. See tools/evotrace/ale_evaluator.py.
    if task_row[1] == "ale":
        from tools.evotrace.ale_evaluator import TEMPLATE as ALE_TEMPLATE

        generated = Path("bench/out/ale_evaluators") / f"{task_row[0]}.py"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text(
            ALE_TEMPLATE.format(
                problem=task_row[0], num_cases=10, language="cpp20", workers=8
            ),
            encoding="utf-8",
        )
        evaluator = generated.resolve()
    else:
        evaluator = None
    for (run_rel,) in conn.execute(
        "SELECT run_rel FROM gem_eg_session WHERE task_uid = %s ORDER BY session_uid",
        (task_uid,),
    ).fetchall():
        if evaluator is not None:
            break
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
        "task_key": task_row[0],
        "domain": task_row[1],
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
        "--arm",
        required=True,
        choices=["nocontext", "none", "gem", "polyglot", "cognee", "trimem", "trimem-adaptive"],
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
    ap.add_argument(
        "--physical-plan",
        choices=["vfwd", "rrev", "aivg"],
        default=None,
        help="node-seeded GEM physical plan; requires --seed-layer node",
    )
    ap.add_argument(
        "--native-fused-plan",
        choices=["graph_expand", "source_anchored", "vector_only"],
        default=None,
        help="formal native path: exactly one native fused operator call per "
        "retrieval (ADR-0022/0023). Requires --seed-layer node; exclusive with "
        "--physical-plan. source_anchored starts from the task's seed node.",
    )
    ap.add_argument(
        "--require-beat-parent",
        choices=["true", "false"],
        default=None,
        help="keep the beat-parent bound in the retrieval predicate (default: true). "
        "`false` ablates the relational leg only; every guard stays.",
    )
    ap.add_argument(
        "--embedding-endpoint",
        default="http://127.0.0.1:8001/v1/embeddings",
    )
    ap.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument(
        "--polyglot-receipt", default="bench/out/polyglot/load_receipt.json"
    )
    ap.add_argument("--cognee-endpoint", default="http://127.0.0.1:8899/retrieve")
    ap.add_argument("--cognee-index", default="bench/out/cognee/index.json")
    ap.add_argument(
        "--injection-rate",
        type=float,
        default=None,
        help="fraction of the inspiration slots to fill from memory, in [0,1]. "
        "With the frozen 10 slots, 0.1 -> 1 program and 0.5 -> 5. Omit for "
        "'fill every slot'. Arm `none` is the 0.0 point by construction, so "
        "this only means anything for a memory arm.",
    )
    ap.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Smoke tests only. Omit for the frozen budget used by every reported cell.",
    )
    ap.add_argument(
        "--gate-closed",
        choices=["empty", "own"],
        default="empty",
        help="What a closed injection gate means. empty: that iteration renders NO "
        "reference programs at all, so p interpolates between `nocontext` and "
        "the memory arm. own: it falls back to the run's own programs, so p "
        "interpolates between `none` and the memory arm.",
    )
    ap.add_argument(
        "--injection-frequency",
        type=float,
        default=0.5,
        help="arXiv:2606.29823's p: the per-step PROBABILITY that memory is injected "
        "at all. Distinct from --injection-rate, which sets how many programs "
        "one injection carries. The paper sweeps p in {0.1, 0.5}; this "
        "experiment previously ran only at p=1.0.",
    )
    ap.add_argument(
        "--inject-as",
        default="code",
        choices=["code", "changes"],
        help="what the prompt renders for each injected program: its source (`code`) "
        "or its edit description (`changes`). See build_config for why this is "
        "an experimental axis rather than a formatting preference.",
    )
    ap.add_argument(
        "--num-diverse",
        type=int,
        default=None,
        help="override the frozen inspiration-slot count. Changing it changes the "
        "injection-rate denominator, so it must be identical across every cell "
        "that will appear in one table.",
    )
    ap.add_argument(
        "--injection-policy",
        default="fixed",
        choices=["fixed", "match_baseline"],
        help="fixed: render up to num_diverse_programs retrieved programs regardless "
        "of how many the run itself offers. match_baseline: render exactly as "
        "many as arm A would, which makes prompt length identical but injects "
        "NOTHING while the islands are still sparse. See memory_database.py.",
    )
    ap.add_argument("--seed", type=int, default=None, help="overrides the frozen 42")
    ap.add_argument(
        "--eval-timeout",
        type=int,
        default=240,
        help="hard deadline for one evaluation, enforced by the subprocess wrapper. "
        "240s: the corpus's own evaluations peak at ~600s, but a generated "
        "program that runs longer than 240 is pathological rather than slow, "
        "and every second past the deadline is a worker not doing anything.",
    )
    ap.add_argument(
        "--no-wrap-evaluator",
        dest="wrap_evaluator",
        action="store_false",
        help="call the task's evaluator directly. Only for comparing against the "
        "unwrapped behaviour -- OpenEvolve's timeout does not stop a running "
        "evaluation, so an expensive generated program wedges the whole cell.",
    )
    ap.add_argument(
        "--llm-timeout",
        type=int,
        default=1800,
        help="the corpus p99 prompt is 66k tokens; a short timeout would "
        "silently turn long prompts into failed iterations",
    )
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    ap.add_argument("--trimem-dsn", default="postgresql://127.0.0.1:55441/trimem_evo",
                    help="TriMEM experience graph (arm trimem, and --task-source trimem)")
    ap.add_argument("--trajectory-store", action="store_true",
                    help="use rebuilt full-code trajectory schema and per-round agent context")
    ap.add_argument("--trajectory-plan", default="P2+Rank",
                    choices=["VGR", "P1", "P2", "SemRankOnly", "P1+Rank", "P2+Rank"])
    ap.add_argument("--trajectory-total-token-budget", type=int, default=None,
                    help="total generation-token allowance; unknown when omitted")
    ap.add_argument("--trajectory-objective", default=None,
                    help="current improvement objective; otherwise derived from actual parent feedback")
    ap.add_argument("--trajectory-memory-tokens", type=int, default=16384)
    ap.add_argument("--trajectory-prompt-tokens", type=int, default=32768,
                    help="complete generation input budget, including the local Qwen chat template; output allowance is separate")
    ap.add_argument("--trajectory-include-code", action="store_true", default=True,
                    help="complete historical code is always included for trajectory-store (compatibility flag)")
    ap.add_argument("--trimem-mode", choices=["trimodal", "hard", "rank", "filterrank", "seed-vgr", "seed-rank",
                                              "seed-p1", "seed-p2", "seed-opt"], default="rank")
    ap.add_argument("--trimem-force", default=None,
                    help="pin the semantic operator's placement, e.g. d=base (semantic first) "
                    "or d=top (traverse first); omit to let the optimizer choose")
    ap.add_argument("--trimem-graph", default=None,
                    choices=["mixed/down", "mixed/up", "lineage+similar/1", "lineage+similar/2", "lineage+similar/3"],
                    help="arm trimem: traverse the 2-hop mixed path instead of lineage/down/1")
    ap.add_argument("--trimem-prompt-version", default=None,
                    help="cache namespace for this cell's semantic calls (default: the out dir name)")
    ap.add_argument("--prompt-budget-tokens", type=int, default=None,
                    help="input-token budget of one generation request; own inspirations are dropped whole "
                         "(never truncated) to stay under it")
    ap.add_argument("--resume-from", default=None,
                    help="OpenEvolve checkpoint dir of a finished cell; runs its iterations "
                    "last+1..--max-iterations into --out (the source cell is not modified)")
    ap.add_argument("--lambda-l", type=float, default=0.1,
                    help="O4 (v3): weight of the normalised expected-latency term")
    ap.add_argument("--space", choices=["v1", "v2", "v3"], default="v1",
                    help="v1: (family, k); v2: graph configuration (edge, direction, hops, fanout) x family x k, "
                    "target = improvement over the run's best, separate 'inject nothing' estimator")
    ap.add_argument("--o3-tol", type=float, default=0.0,
                    help="O3 (v2): feasible if g(c) >= (1 - tol) * g(rank:5 on lineage/down/1/8)")
    ap.add_argument("--diff-ws-tolerant", action="store_true",
                    help="match SEARCH lines ignoring trailing whitespace (OpenEvolve drops non-matching blocks silently)")
    ap.add_argument("--no-simplify-nudge", action="store_true",
                    help="drop OpenEvolve's 'Consider simplifying - code length exceeds N characters' focus area")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="generation output cap (default FROZEN max_tokens 8192); must leave room in the context")
    ap.add_argument("--fixed-config", default=None,
                    help="--policy fixed (offline profiling): candidate to inject, e.g. 'rank:5@lineage/down/1/8' or 'none:0'")
    ap.add_argument("--out-budget", type=float, default=None,
                    help="O5: budget B of mean output tokens per call for the dual update of λ (default: prior's)")
    ap.add_argument("--policy", choices=["random", "fixed", "o1", "o2", "o3", "o4", "o5"], default="random",
                    help="arm trimem-adaptive: per-iteration choice of (plan family, k)")
    ap.add_argument("--policy-mode", choices=["offline", "online-cold", "online-warm"], default="offline")
    ap.add_argument("--policy-prior", default=None, help="pickle from trimem.objective.train")
    ap.add_argument("--policy-epsilon", type=float, default=0.1)
    ap.add_argument("--policy-horizon", type=int, default=20,
                    help="iteration scale of the 'progress' feature; the prior was trained on 20-iteration runs")
    ap.add_argument("--lambda-t", type=float, default=0.2, help="O2 token weight")
    ap.add_argument("--lambda-a", type=float, default=0.0, help="anchoring penalty (ablation)")
    ap.add_argument("--task-source", choices=["gem_eg", "trimem"], default="gem_eg",
                    help="where the task's evaluator and seed program are resolved from")
    ap.add_argument("--llm-telemetry", action="store_true",
                    help="record per-request generation tokens/latency to llm_telemetry.jsonl")
    ap.add_argument("--thinking", choices=["true", "false"], default=None,
                    help="chat_template_kwargs.enable_thinking for every generation request")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve everything and write the receipt, run no iterations",
    )
    args = ap.parse_args(argv)

    if args.physical_plan is not None and args.seed_layer != "node":
        ap.error("--physical-plan requires --seed-layer node")
    if args.native_fused_plan is not None:
        if args.seed_layer != "node":
            ap.error("--native-fused-plan requires --seed-layer node")
        if args.physical_plan is not None:
            ap.error("--native-fused-plan and --physical-plan are exclusive")
        if args.arm != "gem":
            ap.error("--native-fused-plan applies only to --arm gem")
    locked_policy, policy_lock = _locked_generation_policy(args.out)
    if locked_policy not in (None, BASELINE_GENERATION_POLICY):
        ap.error(
            f"{policy_lock} pins generation policy {locked_policy!r}; this runner "
            f"implements only {BASELINE_GENERATION_POLICY!r}"
        )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    if args.llm_telemetry or args.thinking is not None:
        from bench.agent_memory.gem_oe import llm_telemetry

        if args.thinking is not None:
            os.environ[llm_telemetry.THINKING_ENABLED_ENV] = args.thinking
        os.environ[llm_telemetry.TELEMETRY_PATH_ENV] = str((args.out / "llm_telemetry.jsonl").resolve())
        os.environ.setdefault(llm_telemetry.STREAM_ENV, "1")  # TTFT; the generated text is unchanged
        llm_telemetry.install_from_env()
    config_cls, controller_cls = _load_openevolve()
    task = (resolve_task_trimem(args.trimem_dsn, args.task) if args.task_source == "trimem"
            else resolve_task(args.dsn, args.task, args.raw))
    cfg = build_config(config_cls, args)
    if args.trajectory_store:
        if args.arm not in {"trimem", "nocontext"} or args.task_source != "trimem":
            raise ValueError("--trajectory-store requires --arm trimem/nocontext --task-source trimem")
        if min(args.trajectory_memory_tokens, args.trajectory_prompt_tokens) <= 0:
            raise ValueError("trajectory prompt and memory token budgets must be positive")
        from bench.agent_memory.gem_oe.trajectory_prompt import install

        install()
        args.inject_as = "code"
        cfg.prompt.num_top_programs = 0 if args.arm == "nocontext" else 1
        cfg.prompt.num_diverse_programs = 0 if args.arm == "nocontext" else 4
        cfg.prompt.template_dir = str(Path(__file__).parent / "prompts" / "trajectory")
        # task.spec is the original task system message, retained verbatim.
        cfg.prompt.system_message = task["spec"]
        os.environ["TRIMEM_TRAJECTORY_PROMPT_LOG"] = str((args.out / "prompt_context.jsonl").resolve())
        os.environ["TRIMEM_TRAJECTORY_PROMPT_TOKEN_BUDGET"] = str(args.trajectory_prompt_tokens)
        os.environ["TRIMEM_TRAJECTORY_MEMORY_TOKEN_BUDGET"] = str(args.trajectory_memory_tokens)
        os.environ["TRIMEM_TRAJECTORY_ENABLE_THINKING"] = str(args.thinking).lower()
    retriever = build_retriever(args, source_anchor_uid=task["seed_node_uid"])

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
        None if args.injection_rate is None else int(round(args.injection_rate * slots))
    )

    engine_identity = None
    gem_retriever = getattr(retriever, "gem", retriever)
    gem_store = getattr(gem_retriever, "store", None)
    if gem_store is not None:
        engine_identity = {
            "postgres": gem_store.conn.execute("SELECT version()").fetchone()[0],
            "extensions": dict(
                gem_store.conn.execute(
                    "SELECT extname,extversion FROM pg_extension"
                    " WHERE extname IN ('vector','graph_store_am','tjs_pg') ORDER BY 1"
                ).fetchall()
            ),
            "machine": platform.machine(),
            "platform": platform.platform(),
        }

    receipt: dict[str, Any] = {
        "schema_version": "oe_arm_run_v0.2.0",
        "task_uid": args.task,
        "language": args.language,
        "arm": args.arm,
        "policy": ({"space": args.space, "o3_tol": args.o3_tol, "lambda_l": args.lambda_l, "kind": args.policy, "mode": args.policy_mode, "prior": args.policy_prior, "epsilon": args.policy_epsilon, "lambda_t": args.lambda_t, "lambda_a": args.lambda_a} if args.arm == "trimem-adaptive" else None),
        "retriever": retriever.name,
        "split": args.split,
        "seed_layer": args.seed_layer,
        "physical_plan": args.physical_plan,
        "native_fused_plan": args.native_fused_plan,
        "native_fused_only": args.native_fused_plan is not None,
        "top_m": args.m_seeds if args.native_fused_plan is not None else None,
        "src_is_null": (args.native_fused_plan != "source_anchored")
        if args.native_fused_plan is not None
        else None,
        "source_anchor_uid": task["seed_node_uid"]
        if args.native_fused_plan == "source_anchored"
        else None,
        "logical_ranking": (
            "best_reaching_seed_distance ASC, fitness DESC, node_uid ASC"
            if args.physical_plan
            else None
        ),
        "timing_schema": "gem_plan_timing_v1" if args.physical_plan else None,
        "injection_policy": args.injection_policy,
        "trajectory_store": args.trajectory_store,
        "trajectory_plan": args.trajectory_plan if args.trajectory_store and args.arm != "nocontext" else None,
        "trajectory_memory_tokens": args.trajectory_memory_tokens if args.trajectory_store else None,
        "trajectory_prompt_tokens": args.trajectory_prompt_tokens if args.trajectory_store else None,
        "trajectory_total_token_budget": args.trajectory_total_token_budget if args.trajectory_store else None,
        "trajectory_context_contract": "trajectory-context-v4-candidate-code" if args.trajectory_store else None,
        "trajectory_top_programs": cfg.prompt.num_top_programs if args.trajectory_store else None,
        "trajectory_inspiration_target": cfg.prompt.num_diverse_programs if args.trajectory_store else None,
        "inject_as": args.inject_as,
        "injection_render_contract": "external_program_code_is_changes_description"
        if args.inject_as == "changes"
        else "external_program_code_is_source_code",
        "injection_slots": slots,
        "injection_rate_requested": args.injection_rate,
        "injection_frequency": args.injection_frequency,
        "gate_closed": args.gate_closed,
        "max_iterations_override": args.max_iterations,
        "max_tokens": args.max_tokens or FROZEN["max_tokens"],
        "simplify_nudge": not args.no_simplify_nudge,
        "diff_match": "rstrip" if args.diff_ws_tolerant else "exact",
        "max_injected": max_injected,
        "frozen": FROZEN,
        "resolved_seed": cfg.random_seed,
        "evaluator": task["evaluator"],
        "evaluator_wrapped": bool(args.wrap_evaluator),
        "seed_node_uid": task["seed_node_uid"],
        "seed_fitness": task["seed_fitness"],
        "seed_fitness_source": "stored_corpus",
        "seed_artifact_uid": task["seed_artifact_uid"],
        "seed_root_count": task["seed_root_count"],
        "seed_root_variants": task["seed_root_variants"],
        "require_beat_parent": _require_beat_parent(args),
        "llm": {
            "base_url": args.llm_base,
            "model": args.llm_model,
            "generation_policy": BASELINE_GENERATION_POLICY,
            "generation_policy_lock": str(policy_lock) if policy_lock else None,
        },
        "embedding": {
            "endpoint": args.embedding_endpoint,
            "model": args.embedding_model,
        },
        "scope": args.scope,
        "engine": engine_identity,
        "git": {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
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
        frequency=args.injection_frequency,
        frequency_seed=cfg.random_seed,
        gate_closed=args.gate_closed,
        render_mode=args.inject_as,
        # chars ~ 3.2 per token for C++/Python (measured 3.5 on ALE prompts; 3.2 keeps a margin)
        prompt_budget_chars=None if args.prompt_budget_tokens is None else int(3.2 * args.prompt_budget_tokens),
        prompt_base_chars=len(task.get("spec") or "") + 12000,
    )
    controller.database = memory_db
    controller.evaluator.database = memory_db
    if args.trajectory_store and args.arm == "trimem":
        retriever.database = memory_db
    if args.arm == "trimem-adaptive":
        from bench.agent_memory.gem_oe.trimem_adaptive import install_observer

        install_observer(memory_db, retriever)

    if args.gate_closed == "empty" and args.injection_frequency < 1.0:
        _install_zero_context_gate()

    try:
        if args.resume_from:
            # Continue a finished cell from its checkpoint into THIS (new) out dir: the original
            # cell's traces stay untouched. Iterations last+1 .. max_iterations run here.
            memory_db.load(args.resume_from)
            memory_db._iteration = memory_db.last_iteration
            to_run = cfg.max_iterations - memory_db.last_iteration
            receipt["resumed_from"] = args.resume_from
            receipt["resumed_at_iteration"] = memory_db.last_iteration
            asyncio.run(controller.run(iterations=to_run, checkpoint_path=args.resume_from))
        else:
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
        if hasattr(retriever, "close"):
            retriever.close()
        traced = len(memory_db.injection_trace)
        receipt["iterations_traced"] = traced
        expected = cfg.max_iterations - receipt.get("resumed_at_iteration", 0)
        receipt["iterations_expected"] = expected
        receipt["completion_fraction"] = round(traced / max(expected, 1), 4)
        if receipt["status"] == "complete" and traced < expected * 0.9:
            receipt["status"] = "incomplete"
            receipt["error"] = (
                f"only {traced}/{expected} iterations ran. OpenEvolve "
                "returned without raising -- an evaluator timeout stops the loop "
                "silently -- so this cell must not be read as a finished run."
            )
        own_programs = memory_db.own_programs()
        receipt["own_programs"] = len(own_programs)
        roots = [program for program in own_programs.values() if not program.parent_id]
        if len(roots) == 1:
            receipt["live_seed_metrics"] = roots[0].metrics
            receipt["live_seed_fitness"] = roots[0].metrics.get("combined_score")
            receipt["live_seed_fitness_source"] = "task_local_evaluator_this_cell"
        else:
            receipt["live_seed_metrics_error"] = (
                f"expected one run root, found {len(roots)}"
            )
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
