"""Run and audit the 2-task, 6-session OpenEvolve calibration matrix."""

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

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data/cross_session_reuse/calibration_a_60"
DEFAULT_OPENEVOLVE = ROOT / ".venv-e0/bin/openevolve-run"
ITERATION_ERROR = re.compile(r"Iteration (\d+) error: (.*)")
MODEL_ENDPOINT = "http://127.0.0.1:8000/v1/models"


@dataclass(frozen=True)
class SessionSpec:
    task_id: str
    seed: int
    initial_program: Path
    evaluator: Path
    system_message: str
    diff_based_evolution: bool
    max_tokens: int

    @property
    def session_id(self) -> str:
        return f"{self.task_id}-calibration-seed-{self.seed}"


def session_specs() -> list[SessionSpec]:
    f1_initial = (
        ROOT / "experiments/e0/openevolve_function_minimization/initial_program.py"
    )
    f1_evaluator = CALIBRATION_DIR / "function_f1_evaluator.py"
    c18_initial = CALIBRATION_DIR / "circle_c18_initial.py"
    c18_evaluator = CALIBRATION_DIR / "circle_c18_evaluator.py"
    specs: list[SessionSpec] = []
    for seed in (101, 102, 103):
        specs.append(
            SessionSpec(
                task_id="function_f1",
                seed=seed,
                initial_program=f1_initial,
                evaluator=f1_evaluator,
                system_message=(
                    "/no_think Improve the deterministic function-minimization "
                    "search. Modify only the EVOLVE block. Keep run_search() "
                    "bounded and reproducible; maximize combined_score."
                ),
                diff_based_evolution=True,
                max_tokens=1024,
            )
        )
        specs.append(
            SessionSpec(
                task_id="circle_c18",
                seed=seed,
                initial_program=c18_initial,
                evaluator=c18_evaluator,
                system_message=(
                    "/no_think Improve a constructor that packs exactly 18 "
                    "positive-radius, non-overlapping circles inside a unit square. "
                    "Return complete Python code defining run_packing(); maximize "
                    "the independently recomputed sum of radii."
                ),
                diff_based_evolution=False,
                max_tokens=2048,
            )
        )
    return specs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_config(
    spec: SessionSpec, session_dir: Path, iterations: int
) -> dict[str, Any]:
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
            "max_tokens": spec.max_tokens,
            "timeout": 600,
            "retries": 6,
            "retry_delay": 10,
            "random_seed": spec.seed,
        },
        "prompt": {
            "system_message": spec.system_message,
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
            "timeout": 30,
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
        "diff_based_evolution": spec.diff_based_evolution,
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


def summarize_session(
    spec: SessionSpec,
    session_dir: Path,
    *,
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
    calibration_complete = (
        audit_complete and len(child_rows) == iterations and not errors
    )
    return {
        "schema_version": "csr-calibration-session-v0.1.0",
        "session_id": spec.session_id,
        "task_id": spec.task_id,
        "seed": spec.seed,
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
        "calibration_complete": calibration_complete,
    }


def run_session(
    spec: SessionSpec,
    output_dir: Path,
    *,
    iterations: int,
    openevolve: Path,
    resume: bool,
) -> dict[str, Any]:
    session_dir = output_dir / spec.session_id
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
    env.update(
        {
            "CSR_ATTEMPT_LEDGER": str((session_dir / "attempt_ledger.jsonl").resolve()),
            "CSR_TASK_ID": spec.task_id,
            "CSR_SESSION_ID": spec.session_id,
            "CSR_INITIAL_SHA256": _sha256(spec.initial_program),
        }
    )
    command = [
        str(openevolve),
        str(spec.initial_program),
        str(spec.evaluator),
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
        iterations=iterations,
        returncode=completed.returncode,
        wall_seconds=time.monotonic() - started,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def aggregate(summaries: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    return {
        "schema_version": "csr-calibration-manifest-v0.1.0",
        "configured_sessions": len(summaries),
        "configured_generated_attempt_slots": len(summaries) * iterations,
        "expected_initial_program_nodes": len(summaries),
        "actual_initial_program_nodes": sum(
            row["initial_program_nodes"] for row in summaries
        ),
        "actual_evaluated_child_nodes": sum(
            row["evaluated_child_nodes"] for row in summaries
        ),
        "actual_program_nodes": sum(
            row["initial_program_nodes"] + row["evaluated_child_nodes"]
            for row in summaries
        ),
        "lineage_trace_edges": sum(row["lineage_trace_edges"] for row in summaries),
        "generation_failures": sum(
            len(row["generation_failures"]) for row in summaries
        ),
        "invalid_or_error_evaluations": sum(
            row["invalid_or_error_evaluations"] for row in summaries
        ),
        "wall_seconds": sum(row["wall_seconds"] for row in summaries),
        "all_sessions_audit_complete": all(row["audit_complete"] for row in summaries),
        "calibration_complete": all(row["calibration_complete"] for row in summaries),
        "sessions": summaries,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--openevolve", type=Path, default=DEFAULT_OPENEVOLVE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--session",
        action="append",
        help="run only the named session_id; may be supplied more than once",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not args.openevolve.is_file():
        raise FileNotFoundError(f"OpenEvolve CLI not found: {args.openevolve}")
    selected = session_specs()
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
            iterations=args.iterations,
            openevolve=args.openevolve,
            resume=args.resume,
        )
        summaries.append(summary)
        manifest = aggregate(summaries, args.iterations)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    manifest = aggregate(summaries, args.iterations)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["calibration_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
