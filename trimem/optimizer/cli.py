"""Inspect/search/import tools; real evaluation requires an explicit factory."""

from __future__ import annotations

import argparse
import importlib
import json
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict
from pathlib import Path

from .compiler import QueryContext, compile_candidate
from .controller import JointOptimizer
from .evaluator import CohortEvaluator, CohortObservation
from .schema import CandidateSpec
from .search import SearchSettings, SpaceExhausted
from .space import SearchSpace, make_space


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _front(optimizer):
    return [
        dict(
            action=row.action.to_dict(),
            quality=row.quality_mean,
            total_llm_tokens=row.total_tokens_mean,
            quality_variance_of_mean=row.quality_variance_of_mean,
            token_variance_of_mean=row.tokens_variance_of_mean,
            n=row.n,
        )
        for row in optimizer.pareto()
    ]


def _run(optimizer, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    for _ in range(steps):
        try:
            record = optimizer.step()
        except SpaceExhausted:
            break
        observation = CohortObservation.from_dict(record["observation"])
        _print(
            {
                "trial_id": record["trial_id"],
                "plan": record["action"]["plan_id"],
                "strategy": record["proposal"]["strategy"],
                "complete_for_model": observation.complete_for_model,
                "quality": observation.quality_mean,
                "total_llm_tokens": observation.total_tokens_mean,
            }
        )
        if not observation.complete_for_model:
            _print(
                {"stopped": "incomplete evaluation retained in archive; no auto retry"}
            )
            return 2
    _print({"pareto": _front(optimizer)})
    return 0


def _publish_exclusive(path: Path, text: str) -> None:
    """Write a complete temp file, then hard-link it to a NEW destination.

    ``os.link`` fails if the destination exists, so concurrent writers cannot
    clobber each other and an interrupted write never leaves a partial file.
    """
    import os
    import tempfile

    fd, temporary = tempfile.mkstemp(prefix=".router-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("router output exists; write a new artifact path") from exc
    finally:
        os.unlink(temporary)


def _is_policy_archive(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    return manifest.get("kind") == "routed_policy_evaluation"


def _run_policy(args):
    from .evaluator import EvaluationProtocol, validate_observation
    from .router import PolicyArchive, RoutedPolicy, policy_manifest
    from .runtime import ROUTED_MODE

    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    config = json.loads(args.config.read_text())
    if config["evaluation"].get("application_mode") != ROUTED_MODE:
        raise ValueError(f"run-policy requires application_mode={ROUTED_MODE}")
    policy = RoutedPolicy.load(
        args.router, lambda_=args.lambda_, expected_sha256=args.router_sha256
    )
    module, name = args.factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    resource = factory(config["evaluation"])
    manager = (
        resource
        if isinstance(resource, AbstractContextManager)
        else nullcontext(resource)
    )
    with manager as evaluator:
        if not isinstance(evaluator, CohortEvaluator):
            raise TypeError("factory must yield CohortEvaluator")
        policy.check_deployment(
            evaluator.protocol.to_dict(),
            [asdict(unit) for unit in evaluator.units],
            config["evaluation"],
        )
        archive = PolicyArchive(
            args.archive,
            policy_manifest(evaluator, policy, config.get("provenance", {})),
        )
        for _ in range(args.repeats):
            trial = archive.reserve(
                policy,
                proposal={"strategy": "fixed_routed_policy_evaluation"},
            )
            # An exception leaves the trial pending for explicit `fail` recovery.
            observation = validate_observation(
                evaluator.evaluate(policy),
                EvaluationProtocol.from_dict(archive.manifest["protocol"]),
                evaluator.units,
            )
            archive.tell(trial["trial_id"], observation.to_dict())
            _print(
                {
                    "trial_id": trial["trial_id"],
                    "policy": policy.to_dict(),
                    "complete_for_model": observation.complete_for_model,
                    "quality": observation.quality_mean,
                    "total_llm_tokens": observation.total_tokens_mean,
                }
            )
            if not observation.complete_for_model:
                _print({"stopped": "incomplete evaluation retained in archive"})
                return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("space", help="metadata only; no DB/model calls")
    p.add_argument("--plan", action="append")
    p.add_argument("--limit", type=int, default=3)
    p = commands.add_parser("compile", help="compile only; no DB connection")
    p.add_argument("--action", type=Path, required=True)
    p.add_argument("--query", type=Path)
    p = commands.add_parser(
        "demo", help="synthetic CPU wiring smoke, never agent evidence"
    )
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--fixed-plan")
    for name in ("ask", "pareto", "status"):
        p = commands.add_parser(name)
        p.add_argument("--archive", type=Path, required=True)
    p = commands.add_parser("tell", help="validate and import actual observation JSON")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--observation", type=Path, required=True)
    p = commands.add_parser(
        "fail", help="explicit recovery; retain known cost evidence"
    )
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--evidence", type=Path)
    p = commands.add_parser("run", help="explicit real/custom evaluator plugin")
    p.add_argument(
        "--factory", required=True, help="module:callable(config) -> evaluator context"
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1)
    p = commands.add_parser(
        "preflight", help="validate frozen inputs without DB, LLM, or evaluation"
    )
    p.add_argument("--config", type=Path, required=True)
    p = commands.add_parser(
        "train-router", help="fit the per-round router on a Branch A archive (CPU)"
    )
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser(
        "run-policy", help="evaluate a fixed routed policy; no search proposal"
    )
    p.add_argument(
        "--factory", required=True, help="module:callable(config) -> evaluator context"
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--router", type=Path, required=True)
    p.add_argument("--router-sha256")
    p.add_argument("--lambda", dest="lambda_", type=float)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "space":
        space = make_space(allowed_plan_ids=args.plan)
        _print(
            {
                "count": len(space),
                "plans": space.plan_ids,
                "actions": [
                    a.to_dict() for a in space.candidates[: max(0, args.limit)]
                ],
            }
        )
    elif args.command == "compile":
        action = CandidateSpec.from_dict(json.loads(args.action.read_text()))
        query = (
            QueryContext(**json.loads(args.query.read_text())) if args.query else None
        )
        _print(asdict(compile_candidate(action, query)))
    elif args.command == "demo":
        from .demo import demo_optimizer

        return _run(
            demo_optimizer(args.archive, fixed_plan=args.fixed_plan), args.steps
        )
    elif args.command == "preflight":
        from .runtime import validate_inputs

        config = json.loads(args.config.read_text())
        checkpoints, snapshots = validate_inputs(config["evaluation"])
        space = SearchSpace(CandidateSpec.from_dict(a) for a in config["actions"])
        SearchSettings(**config["search_settings"])
        _print(
            {
                "input_validation": "passed",
                "provider_calls": 0,
                "candidate_count": len(space),
                "plans": space.plan_ids,
                "tasks": list(checkpoints),
                "snapshots": {
                    task: snap.snapshot_id for task, snap in snapshots.items()
                },
                "horizon": config["evaluation"]["protocol"]["horizon_kind"],
                "max_steps": config["evaluation"]["protocol"]["max_steps"],
            }
        )
    elif args.command == "train-router":
        from .router import train_router

        if args.out.exists():
            raise ValueError("router output exists; write a new artifact path")
        artifact = train_router(
            args.archive, json.loads(args.config.read_text()), seed=args.seed
        )
        _publish_exclusive(
            args.out, json.dumps(artifact, indent=2, allow_nan=False) + "\n"
        )
        _print(
            {
                "router": str(args.out),
                "artifact_sha256": artifact["artifact_sha256"],
                "catalog_size": len(artifact["catalog"]),
                "n_samples": artifact["training_data"]["n_samples"],
                "trials": len(artifact["training_data"]["trial_ids"]),
                "lambda": artifact["lambda"]["chosen"],
                "lambda_table": artifact["lambda"]["table"],
            }
        )
    elif args.command == "run-policy":
        return _run_policy(args)
    elif args.command == "run":
        config = json.loads(args.config.read_text())
        module, name = args.factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        resource = factory(config["evaluation"])
        manager = (
            resource
            if isinstance(resource, AbstractContextManager)
            else nullcontext(resource)
        )
        with manager as evaluator:
            if not isinstance(evaluator, CohortEvaluator):
                raise TypeError("factory must yield CohortEvaluator")
            optimizer = JointOptimizer.create(
                args.archive,
                SearchSpace(CandidateSpec.from_dict(a) for a in config["actions"]),
                SearchSettings(**config["search_settings"]),
                evaluator,
                provenance=config.get("provenance", {}),
            )
            return _run(optimizer, args.steps)
    elif args.command in ("fail", "status") and _is_policy_archive(args.archive):
        from .router import PolicyArchive

        archive = PolicyArchive(
            args.archive, json.loads((args.archive / "manifest.json").read_text())
        )
        if args.command == "status":
            _print(
                [
                    {
                        "trial_id": r["trial_id"],
                        "status": r["status"],
                        "policy": r["action"],
                    }
                    for r in archive.records()
                ]
            )
        else:
            evidence = json.loads(args.evidence.read_text()) if args.evidence else None
            _print(archive.fail(args.trial_id, args.reason, evidence=evidence))
    else:
        optimizer = JointOptimizer.restore(args.archive)
        if args.command == "ask":
            _print(optimizer.ask())
        elif args.command == "pareto":
            _print(_front(optimizer))
        elif args.command == "status":
            _print(
                [
                    {
                        "trial_id": r["trial_id"],
                        "status": r["status"],
                        "plan": r["action"]["plan_id"],
                    }
                    for r in optimizer.archive.records()
                ]
            )
        elif args.command == "tell":
            observation = CohortObservation.from_dict(
                json.loads(args.observation.read_text())
            )
            _print(optimizer.tell(args.trial_id, observation))
        elif args.command == "fail":
            evidence = json.loads(args.evidence.read_text()) if args.evidence else None
            _print(
                optimizer.archive.fail(args.trial_id, args.reason, evidence=evidence)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
