"""Run the approved one-seed Math/ALE GEM physical-plan matrix serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

MATH_TASKS = (
    "math:circle_packing",
    "math:first_autocorr_ineq",
    "math:heilbronn_convex_13",
    "math:heilbronn_triangle",
    "math:second_autocorr_ineq",
    "math:third_autocorr_ineq",
    "math:uncertainty_ineq",
)
ALE_TASKS = (
    "ale:ahc008",
    "ale:ahc011",
    "ale:ahc015",
    "ale:ahc016",
    "ale:ahc024",
    "ale:ahc025",
    "ale:ahc026",
    "ale:ahc027",
    "ale:ahc039",
    "ale:ahc046",
)
TASKS = MATH_TASKS + ALE_TASKS
PLANS = ("vfwd", "rrev", "aivg")
SEED = 42
ITERATIONS = 40
INJECTION_FREQUENCY = 0.5
INJECTION_RATE = 0.5
INJECTION_SLOTS = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_path(out: Path, task: str, plan: str) -> Path:
    return out / "cells" / f"{task.replace(':', '_')}__gem__{plan}__s{SEED}"


def _complete(
    cell: Path, *, expected_iterations: int, expected_task: str, expected_plan: str
) -> bool:
    receipt_path = cell / "run_receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (
            receipt.get("status") == "complete",
            receipt.get("task_uid") == expected_task,
            receipt.get("language")
            == ("cpp" if expected_task.startswith("ale:") else "python"),
            receipt.get("resolved_seed") == SEED,
            receipt.get("physical_plan") == expected_plan,
            receipt.get("iterations_expected") == expected_iterations,
            receipt.get("iterations_traced") == expected_iterations,
            receipt.get("live_seed_fitness") is not None,
        )
    )


def _dry_complete(cell: Path, *, expected_task: str, expected_plan: str) -> bool:
    receipt_path = cell / "run_receipt.json"
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (
            receipt.get("status") == "dry_run",
            receipt.get("task_uid") == expected_task,
            receipt.get("language")
            == ("cpp" if expected_task.startswith("ale:") else "python"),
            receipt.get("physical_plan") == expected_plan,
            receipt.get("resolved_seed") == SEED,
            receipt.get("injection_frequency") == INJECTION_FREQUENCY,
            receipt.get("injection_rate_requested") == INJECTION_RATE,
            receipt.get("injection_slots") == INJECTION_SLOTS,
            receipt.get("max_injected") == 5,
            receipt.get("inject_as") == "changes",
        )
    )


def _load_gate(
    path: Path,
    *,
    expected_domain: str | None = None,
    expected_endpoint: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing gate receipt: {path}")
    receipt = json.loads(path.read_text())
    if not receipt.get("passed"):
        raise ValueError(f"gate did not pass: {path}")
    if expected_domain is not None and receipt.get("domain") != expected_domain:
        raise ValueError(
            f"gate domain {receipt.get('domain')!r}, expected {expected_domain!r}: {path}"
        )
    if expected_domain is not None:
        if receipt.get("hard_gate") != "top_m_identity":
            raise ValueError(f"unexpected embedding hard gate: {path}")
        if receipt.get("top_m") != 4 or not receipt.get("topm_identity_passed"):
            raise ValueError(f"embedding top-4 identity did not pass: {path}")
    if expected_endpoint is not None and receipt.get("endpoint") != expected_endpoint:
        raise ValueError(
            f"gate endpoint {receipt.get('endpoint')!r}, expected "
            f"{expected_endpoint!r}: {path}"
        )
    return {"path": str(path), "sha256": _sha256(path), "receipt": receipt}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        return {"status": "not_required_for_dry_run"}
    if not args.embedding_receipt_math or not args.embedding_receipt_ale:
        raise ValueError("real runs require both Math and ALE embedding gate receipts")
    if not args.correctness_receipt:
        raise ValueError("real runs require a Track-C correctness gate receipt")

    def service(url: str) -> dict[str, Any]:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)

    llm_root = args.llm_base.removesuffix("/v1")
    embedding_root = args.embedding_endpoint.removesuffix("/v1/embeddings")
    return {
        "status": "passed",
        "embedding": {
            "math": _load_gate(
                args.embedding_receipt_math,
                expected_domain="math",
                expected_endpoint=args.embedding_endpoint,
            ),
            "ale": _load_gate(
                args.embedding_receipt_ale,
                expected_domain="ale",
                expected_endpoint=args.embedding_endpoint,
            ),
        },
        "correctness": _load_gate(args.correctness_receipt),
        "services": {
            "llm_models": service(f"{args.llm_base}/models"),
            "llm_version": service(f"{llm_root}/version"),
            "embedding_models": service(f"{embedding_root}/v1/models"),
            "embedding_version": service(f"{embedding_root}/version"),
        },
    }


def _cells(
    tasks: tuple[str, ...], plans: tuple[str, ...], out: Path
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        offset = task_index % len(plans)
        rotated = plans[offset:] + plans[:offset]
        for order, plan in enumerate(rotated):
            cells.append(
                {
                    "task_uid": task,
                    "domain": task.split(":", 1)[0],
                    "physical_plan": plan,
                    "task_plan_order": order,
                    "seed": SEED,
                    "path": str(_cell_path(out, task, plan)),
                }
            )
    return cells


def _command(args: argparse.Namespace, row: dict[str, Any], cell: Path) -> list[str]:
    is_ale = row["domain"] == "ale"
    command = [
        args.python,
        "-m",
        "bench.agent_memory.gem_oe.run_arm",
        "--task",
        row["task_uid"],
        "--arm",
        "gem",
        "--out",
        str(cell),
        "--dsn",
        args.dsn,
        "--scope",
        args.scope,
        "--seed-layer",
        "node",
        "--physical-plan",
        row["physical_plan"],
        "--m-seeds",
        "4",
        "--seed",
        str(SEED),
        "--injection-frequency",
        str(INJECTION_FREQUENCY),
        "--injection-rate",
        str(INJECTION_RATE),
        "--num-diverse",
        str(INJECTION_SLOTS),
        "--injection-policy",
        "fixed",
        "--gate-closed",
        "empty",
        "--inject-as",
        "changes",
        "--eval-timeout",
        str(args.ale_eval_timeout if is_ale else args.math_eval_timeout),
        "--language",
        "cpp" if is_ale else "python",
        "--llm-base",
        args.llm_base,
        "--llm-model",
        args.llm_model,
        "--embedding-endpoint",
        args.embedding_endpoint,
        "--embedding-model",
        args.embedding_model,
    ]
    if args.max_iterations is not None:
        command.extend(("--max-iterations", str(args.max_iterations)))
    if args.dry_run:
        command.append("--dry-run")
    return command


def _run(command: list[str], log: Path, timeout: float) -> tuple[int, bool, float]:
    started = time.perf_counter()
    timed_out = False
    with log.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except KeyboardInterrupt:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            process.wait(timeout=30)
            raise
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait(timeout=30)
            handle.write(f"\n[matrix] hard timeout after {timeout:.0f}s\n")
    return returncode, timed_out, round(time.perf_counter() - started, 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=".venv-e0/bin/python")
    parser.add_argument("--dsn", default="postgresql://127.0.0.1:55432/evotrace_eg")
    parser.add_argument("--scope", default="evotrace:349117b0")
    parser.add_argument("--llm-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen3.8")
    parser.add_argument(
        "--embedding-endpoint", default="http://127.0.0.1:8011/v1/embeddings"
    )
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-receipt-math", type=Path)
    parser.add_argument("--embedding-receipt-ale", type=Path)
    parser.add_argument("--correctness-receipt", type=Path)
    parser.add_argument("--tasks", nargs="*", choices=TASKS)
    parser.add_argument("--plans", nargs="*", choices=PLANS)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--math-eval-timeout", type=int, default=240)
    parser.add_argument("--ale-eval-timeout", type=int, default=600)
    parser.add_argument("--math-cell-timeout", type=float, default=5400)
    parser.add_argument("--ale-cell-timeout", type=float, default=7200)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = tuple(args.tasks) if args.tasks else TASKS
    plans = tuple(args.plans) if args.plans else PLANS
    expected_iterations = args.max_iterations or ITERATIONS
    if expected_iterations <= 0:
        raise SystemExit("--max-iterations must be positive")
    if not tasks or not plans:
        raise SystemExit("at least one task and one plan are required")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cells").mkdir(exist_ok=True)
    cells = _cells(tasks, plans, args.out)
    try:
        preflight = _preflight(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        preflight = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    manifest: dict[str, Any] = {
        "schema_version": "gem_math_ale_physical_plan_matrix_v1",
        "plan_document": "haikaidocs/evotrace_math_ale_physical_plan_experiment_plan_v0.1.0.md",
        "tasks": list(tasks),
        "plans": list(plans),
        "cells": cells,
        "cell_count": len(cells),
        "agent_seed": SEED,
        "iterations": expected_iterations,
        "formal_iterations": ITERATIONS,
        "injection_frequency": INJECTION_FREQUENCY,
        "injection_rate": INJECTION_RATE,
        "injection_slots": INJECTION_SLOTS,
        "max_injected": round(INJECTION_RATE * INJECTION_SLOTS),
        "inject_as": "changes",
        "execution": "serial_rotated_plan_order",
        "endpoints": {
            "llm_base": args.llm_base,
            "llm_model": args.llm_model,
            "embedding_endpoint": args.embedding_endpoint,
            "embedding_model": args.embedding_model,
        },
        "implementation_sha256": {
            path: _sha256(Path(path))
            for path in (
                "experiments/e2/gem_math_ale_plan_agent_matrix.py",
                "bench/agent_memory/gem_oe/run_arm.py",
                "bench/agent_memory/gem_oe/memory_database.py",
                "bench/agent_memory/gem_oe/retrievers.py",
                "bench/agent_memory/gem_eg/physical.py",
                "tools/evotrace/ale_evaluator.py",
                "tools/evotrace/wrap_evaluator.py",
            )
        },
        "preflight": preflight,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "dry_run" if args.dry_run else "running",
        "results": [],
    }
    manifest_path = args.out / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if preflight["status"] == "failed":
        manifest["status"] = "blocked"
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(preflight["error"], file=sys.stderr)
        return 2

    failures: list[dict[str, Any]] = []
    for index, row in enumerate(cells):
        cell = Path(row["path"])
        cell.mkdir(parents=True, exist_ok=True)
        if args.skip_complete and _complete(
            cell,
            expected_iterations=expected_iterations,
            expected_task=row["task_uid"],
            expected_plan=row["physical_plan"],
        ):
            result = {**row, "status": "skipped_complete", "returncode": 0}
            manifest["results"].append(result)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue
        command = _command(args, row, cell)
        timeout = (
            args.ale_cell_timeout if row["domain"] == "ale" else args.math_cell_timeout
        )
        manifest["active_cell"] = index
        manifest["active_command"] = command
        manifest_path.write_text(json.dumps(manifest, indent=2))
        returncode, timed_out, seconds = _run(command, cell / "cell.log", timeout)
        receipt_complete = (
            _dry_complete(
                cell,
                expected_task=row["task_uid"],
                expected_plan=row["physical_plan"],
            )
            if args.dry_run
            else _complete(
                cell,
                expected_iterations=expected_iterations,
                expected_task=row["task_uid"],
                expected_plan=row["physical_plan"],
            )
        )
        if returncode == 0 and not receipt_complete:
            returncode = 3
        result = {
            **row,
            "returncode": returncode,
            "timed_out": timed_out,
            "seconds": seconds,
            "receipt_complete": receipt_complete,
            "status": "ok" if returncode == 0 else "failed",
        }
        manifest["results"].append(result)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        if returncode:
            failures.append(result)
            break

    manifest.pop("active_cell", None)
    manifest.pop("active_command", None)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["failures"] = failures
    manifest["status"] = (
        "failed" if failures else ("dry_run" if args.dry_run else "complete")
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": manifest["status"], "cells": len(cells)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
