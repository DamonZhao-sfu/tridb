"""Run audited multi-family OpenEvolve cross-session-reuse sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.cross_session_reuse.workload.registry import (
    ROOT,
    TaskVariant,
    profile_variants,
    seeds_for,
)

DEFAULT_OPENEVOLVE = ROOT / ".venv-e0/bin/openevolve-run"
DEFAULT_OUTPUT = ROOT / "data/cross_session_reuse/workload_focus"
MODEL_ENDPOINT = "http://127.0.0.1:8000/v1/models"
ITERATION_ERROR = re.compile(r"Iteration (\d+) error: (.*)")


@dataclass(frozen=True)
class SessionSpec:
    variant: TaskVariant
    seed: int

    @property
    def session_id(self) -> str:
        return f"{self.variant.task_id}-seed-{self.seed}"


def session_specs(profile: str) -> list[SessionSpec]:
    return [
        SessionSpec(variant=variant, seed=seed)
        for variant in profile_variants(profile)
        for seed in seeds_for(variant, profile)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wait_for_model_api(timeout_seconds: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(MODEL_ENDPOINT, timeout=3) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"model API did not become ready: {last_error}")


def build_config(spec: SessionSpec, session_dir: Path, iterations: int) -> dict[str, Any]:
    variant = spec.variant
    return {
        "max_iterations": iterations,
        "checkpoint_interval": iterations,
        "random_seed": spec.seed,
        "log_level": "INFO",
        "log_dir": str((session_dir / "logs").resolve()),
        "llm": {
            "primary_model": "Qwen/Qwen3-32B",
            "primary_model_weight": 1.0,
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "local",
            "temperature": 0.2,
            "max_tokens": variant.max_tokens,
            "timeout": 600,
            "retries": 6,
            "retry_delay": 10,
            "random_seed": spec.seed,
        },
        "prompt": {
            "system_message": variant.system_message,
            "num_top_programs": 3,
            "num_diverse_programs": 2,
            "use_template_stochasticity": False,
        },
        "database": {
            "db_path": str((session_dir / "program_db").resolve()),
            "in_memory": True,
            "log_prompts": False,
            "population_size": 32,
            "archive_size": 16,
            "num_islands": 2,
            "elite_selection_ratio": 0.2,
            "exploration_ratio": 0.3,
            "exploitation_ratio": 0.5,
            "migration_interval": max(2, iterations // 2),
            "migration_rate": 0.1,
            "random_seed": spec.seed,
            "similarity_threshold": 0.99,
        },
        "evaluator": {
            "timeout": variant.evaluator_timeout,
            "max_retries": 0,
            "cascade_evaluation": False,
            "parallel_evaluations": 1,
            "use_llm_feedback": False,
            "enable_artifacts": True,
        },
        "evolution_trace": {
            "enabled": True,
            "format": "jsonl",
            "include_code": True,
            "include_prompts": False,
            "output_path": str((session_dir / "evolution_trace.jsonl").resolve()),
            "buffer_size": 1,
            "compress": False,
        },
        "diff_based_evolution": variant.diff_based_evolution,
        "max_code_length": 50000,
    }


def _iteration_errors(session_dir: Path) -> dict[int, str]:
    errors: dict[int, str] = {}
    paths = [session_dir / "driver.log", *sorted((session_dir / "logs").glob("*.log"))]
    for path in paths:
        if not path.exists():
            continue
        for match in ITERATION_ERROR.finditer(path.read_text(encoding="utf-8")):
            errors[int(match.group(1))] = match.group(2).strip()
    return errors


def summarize_session(
    spec: SessionSpec,
    session_dir: Path,
    *,
    run_id: str,
    effective_session_id: str,
    iterations: int,
    returncode: int,
    wall_seconds: float,
) -> dict[str, Any]:
    ledger = _read_jsonl(session_dir / "attempt_ledger.jsonl")
    traces = _read_jsonl(session_dir / "evolution_trace.jsonl")
    errors = _iteration_errors(session_dir)
    initial_rows = [row for row in ledger if row["program_role"] == "initial"]
    child_rows = [row for row in ledger if row["program_role"] == "child"]
    invalid_rows = [row for row in ledger if row["status"] != "valid"]
    unaccounted = iterations - len(traces) - len(errors)
    audit_complete = (
        returncode == 0
        and len(initial_rows) == 1
        and len(child_rows) == len(traces)
        and unaccounted == 0
    )
    generation_complete = audit_complete and len(child_rows) == iterations and not errors
    return {
        "schema_version": "csr-workload-session-v0.1.0",
        "run_id": run_id,
        "session_key": spec.session_id,
        "session_id": effective_session_id,
        "family": spec.variant.family,
        "task_id": spec.variant.task_id,
        "priority": spec.variant.priority,
        "seed": spec.seed,
        "environment": spec.variant.environment,
        "configured_generated_attempt_slots": iterations,
        "initial_program_nodes": len(initial_rows),
        "evaluated_child_nodes": len(child_rows),
        "lineage_trace_edges": len(traces),
        "generation_failures": [
            {"iteration": iteration, "error": error}
            for iteration, error in sorted(errors.items())
        ],
        "invalid_or_error_evaluations": len(invalid_rows),
        "unaccounted_iterations": unaccounted,
        "wall_seconds": wall_seconds,
        "returncode": returncode,
        "audit_complete": audit_complete,
        "generation_complete": generation_complete,
    }


def run_session(
    spec: SessionSpec,
    output_dir: Path,
    *,
    run_id: str,
    iterations: int,
    openevolve: Path,
    resume: bool,
) -> dict[str, Any]:
    session_dir = output_dir / spec.session_id
    effective_session_id = f"{run_id}:{spec.session_id}"
    summary_path = session_dir / "summary.json"
    if resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite incomplete session {session_dir}; move it aside"
        )
    session_dir.mkdir(parents=True, exist_ok=True)
    wait_for_model_api()
    config_path = session_dir / "run_config.yaml"
    config_path.write_text(
        yaml.safe_dump(build_config(spec, session_dir, iterations), sort_keys=False),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(spec.variant.environment)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else str(ROOT) + os.pathsep + existing_pythonpath
    )
    env.update(
        {
            "CSR_ATTEMPT_LEDGER": str((session_dir / "attempt_ledger.jsonl").resolve()),
            "CSR_TASK_FAMILY": spec.variant.family,
            "CSR_TASK_ID": spec.variant.task_id,
            "CSR_RUN_ID": run_id,
            "CSR_SESSION_ID": effective_session_id,
            "CSR_INITIAL_SHA256": _sha256(spec.variant.initial_program),
        }
    )
    command = [
        str(openevolve),
        str(spec.variant.initial_program),
        str(spec.variant.evaluator),
        "--config",
        str(config_path),
        "--output",
        str(session_dir / "run"),
        "--iterations",
        str(iterations),
    ]
    started = time.monotonic()
    with (session_dir / "driver.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary = summarize_session(
        spec,
        session_dir,
        run_id=run_id,
        effective_session_id=effective_session_id,
        iterations=iterations,
        returncode=completed.returncode,
        wall_seconds=time.monotonic() - started,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def aggregate(
    summaries: list[dict[str, Any]], *, run_id: str, profile: str, iterations: int
) -> dict[str, Any]:
    families = sorted({row["family"] for row in summaries})
    return {
        "schema_version": "csr-workload-manifest-v0.1.0",
        "run_id": run_id,
        "profile": profile,
        "configured_sessions": len(summaries),
        "families": families,
        "family_count": len(families),
        "configured_generated_attempt_slots": len(summaries) * iterations,
        "actual_initial_program_nodes": sum(row["initial_program_nodes"] for row in summaries),
        "actual_evaluated_child_nodes": sum(row["evaluated_child_nodes"] for row in summaries),
        "lineage_trace_edges": sum(row["lineage_trace_edges"] for row in summaries),
        "generation_failures": sum(len(row["generation_failures"]) for row in summaries),
        "invalid_or_error_evaluations": sum(
            row["invalid_or_error_evaluations"] for row in summaries
        ),
        "wall_seconds": sum(row["wall_seconds"] for row in summaries),
        "all_sessions_audit_complete": all(row["audit_complete"] for row in summaries),
        "generation_complete": all(row["generation_complete"] for row in summaries),
        "sessions": summaries,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("coverage", "focus"), default="focus")
    parser.add_argument(
        "--run-id",
        help="globally unique run namespace; defaults to the output directory name",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--openevolve", type=Path, default=DEFAULT_OPENEVOLVE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session", action="append")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not args.openevolve.is_file():
        raise FileNotFoundError(f"OpenEvolve CLI not found: {args.openevolve}")
    run_id = args.run_id or args.output.name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError("--run-id may contain only letters, digits, '.', '_', and '-'")
    selected = session_specs(args.profile)
    if args.session:
        requested = set(args.session)
        selected = [spec for spec in selected if spec.session_id in requested]
        missing = requested - {spec.session_id for spec in selected}
        if missing:
            raise ValueError(f"unknown session IDs: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for spec in selected:
        print(f"running {spec.session_id}", flush=True)
        summary = run_session(
            spec,
            args.output,
            run_id=run_id,
            iterations=args.iterations,
            openevolve=args.openevolve,
            resume=args.resume,
        )
        summaries.append(summary)
        manifest = aggregate(
            summaries,
            run_id=run_id,
            profile=args.profile,
            iterations=args.iterations,
        )
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    manifest = aggregate(
        summaries,
        run_id=run_id,
        profile=args.profile,
        iterations=args.iterations,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["generation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
