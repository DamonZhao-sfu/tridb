"""Lossless adapter around EvoMemBench's official CrossEp-Know evaluator.

The pinned upstream evaluator deliberately constructs a score-0 result when a
judge/API/JSON failure occurs, but its concurrent driver drops that result when
``error`` is non-null.  That leaves fewer graded rows than predictions and
silently changes the denominator.  This adapter calls the pinned evaluator's
exact prompt/parser and persists every returned result, including its intended
score-0 failure rows.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


SCHEMA_VERSION = "evomembench_know_grading_receipt_v0.1.0"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _task_id(item: dict[str, Any]) -> str:
    return str(item.get("metadata", {}).get("task_id", item.get("idx", -1)))


def _score_zero(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **item,
        "idx": _task_id(item),
        "grading_rationale": reason,
        "requirement_status": [],
        "score": 0,
        "evaluator_error": reason,
    }


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as target:
        target.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_official(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evomembench_official_know_eval", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_input(rows: Sequence[dict[str, Any]]) -> None:
    ids = [_task_id(row) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("CrossEp-Know grading input must have unique task IDs")


def _write_receipt(
    receipt_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    rows: Sequence[dict[str, Any]],
    failure_ids: Sequence[str],
    mode: str,
) -> dict[str, Any]:
    observed = _rows(output_path)
    observed_ids = [_task_id(row) for row in observed]
    expected_ids = [_task_id(row) for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("graded output contains duplicate task IDs")
    if set(observed_ids) != set(expected_ids):
        raise ValueError("graded output does not cover the prediction task IDs")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mode": mode,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "predictions": len(rows),
        "graded": len(observed),
        "judge_failures_counted_as_score_zero": len(failure_ids),
        "judge_failure_task_ids": list(failure_ids),
        "denominator_preserved": True,
    }
    if receipt_path.exists():
        raise FileExistsError(f"refusing existing grading receipt: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    receipt_path = Path(args.receipt)
    if output_path.exists():
        raise FileExistsError(f"refusing existing graded output: {output_path}")
    rows = _rows(input_path)
    _validate_input(rows)
    official = _load_official(Path(args.official_eval))
    api_key, base_url, _provider = official.get_api_credentials(
        args.judge_model, args.api_key, args.base_url
    )
    if not api_key:
        raise ValueError("judge API key is required")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client_kwargs["timeout"] = official.httpx.Timeout(300.0, connect=30.0)
    client = official.OpenAI(**client_kwargs)
    tasks = [(row, client, args.judge_model, args.max_retries) for row in rows]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_rows = {
            executor.submit(official.process_single_item, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_rows):
            source = future_rows[future]
            try:
                task_id, result, error = future.result()
                if error:
                    failures.append(str(task_id))
                    result = {**result, "evaluator_error": str(error)}
            except Exception as exc:
                task_id = _task_id(source)
                failures.append(task_id)
                result = _score_zero(
                    source,
                    f"Official evaluator exception counted as score 0: "
                    f"{type(exc).__name__}: {exc}",
                )
            _append(output_path, result)
    return _write_receipt(
        receipt_path,
        input_path=input_path,
        output_path=output_path,
        rows=rows,
        failure_ids=sorted(failures),
        mode="lossless_official_evaluator",
    )


def seal_existing(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    receipt_path = Path(args.receipt)
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    rows = _rows(input_path)
    _validate_input(rows)
    observed = _rows(output_path)
    observed_ids = {_task_id(row) for row in observed}
    expected_ids = {_task_id(row) for row in rows}
    if not observed_ids <= expected_ids:
        raise ValueError("graded output contains task IDs absent from predictions")
    missing = [row for row in rows if _task_id(row) not in observed_ids]
    for row in missing:
        _append(
            output_path,
            _score_zero(
                row,
                "Pinned official evaluator omitted this item after a judge/API/JSON "
                "failure; its process_single_item contract counts the item as score 0.",
            ),
        )
    return _write_receipt(
        receipt_path,
        input_path=input_path,
        output_path=output_path,
        rows=rows,
        failure_ids=[_task_id(row) for row in missing],
        mode="seal_upstream_omitted_failures",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--official-eval")
    parser.add_argument("--judge-model", default="qwen3.8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--seal-existing", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.max_retries < 1:
        parser.error("workers and max-retries must be positive")
    if not args.seal_existing and not args.official_eval:
        parser.error("--official-eval is required unless --seal-existing is used")
    return args


def main() -> None:
    args = parse_args()
    receipt = seal_existing(args) if args.seal_existing else evaluate(args)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
