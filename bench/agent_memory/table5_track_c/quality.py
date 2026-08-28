"""Offline answer generation and LoCoMo judge gate for Search artifacts.

This phase never contributes to Search latency. Every system uses the same
retrieved-context rendering, Qwen endpoint, answer prompt, and judge prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx

from .dataset import EXPECTED_FORMAL_QUESTION_COUNT
from .protocol import _write_json, benchmark_code_sha256
from .stats import read_jsonl
from .verify import verify_run

ANSWER_PROMPT = """You are a knowledgeable and helpful memory assistant.

You have access to timestamped memories from two speakers in a conversation.
Carefully synthesize all relevant entries. Prefer the most recent memory if
facts conflict. Convert relative time references to specific dates using the
memory timestamp. Do not confuse names mentioned in memories with the two
speakers. Ground the answer in the supplied memories and keep the final answer
brief (under 5-6 words), direct, and without extra description.

Memories:
{context}

Question: {question}

Answer:"""

JUDGE_PROMPT = """You are an expert grader. Label the generated answer to the
question as CORRECT or WRONG compared with the gold answer. Be generous about
wording: it is CORRECT if it touches on the same topic. Treat equivalent date
formats or relative references to the same time period as CORRECT.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return only JSON with one key, for example {{\"label\": \"CORRECT\"}}."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_keyed(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records = read_jsonl([path])
    keyed: dict[int, dict[str, Any]] = {}
    for record in records:
        index = int(record["request_index"])
        if index in keyed:
            raise RuntimeError(f"duplicate quality record {index} in {path}")
        keyed[index] = record
    return keyed


class _AsyncJsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")
        self.lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
        async with self.lock:
            self.handle.write(encoded + "\n")
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _contexts(record: dict[str, Any]) -> tuple[list[str], str]:
    values = [
        str(value) for value in (record.get("receipt") or {}).get("contexts") or []
    ]
    return values, "\n\n".join(
        f"[{index}] {value}" for index, value in enumerate(values, start=1)
    )


async def _chat(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    response = await client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"]), payload.get("usage") or {}


async def _token_count(
    client: httpx.AsyncClient, endpoint: str, model: str, text: str
) -> int:
    base = endpoint.removesuffix("/v1").rstrip("/")
    response = await client.post(
        f"{base}/tokenize", json={"model": model, "prompt": text}
    )
    response.raise_for_status()
    return int(response.json()["count"])


async def generate_answers(
    records: Sequence[dict[str, Any]],
    *,
    output: Path,
    endpoint: str,
    model: str,
    workers: int,
) -> list[dict[str, Any]]:
    existing = _load_keyed(output)
    sink = _AsyncJsonlSink(output)
    semaphore = asyncio.Semaphore(workers)
    timeout = httpx.Timeout(60.0)
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def one(record: dict[str, Any]) -> None:
            index = int(record["request_index"])
            if index in existing:
                return
            started = time.perf_counter_ns()
            result: dict[str, Any] = {
                "schema_version": "table5_track_c_answer_v0.2.0",
                "request_index": index,
                "system": record.get("system"),
                "build_id": record.get("build_id"),
                "sample_id": record.get("sample_id"),
                "question_id": record.get("question_id"),
                "question": record.get("question"),
                "gold_answer": record.get("answer"),
                "category": record.get("category"),
                "retrieval_success": record.get("success") is True,
                "started_at": _now(),
                "success": False,
                "error": None,
            }
            try:
                if record.get("success") is not True:
                    raise RuntimeError("retrieval_failed")
                contexts, rendered = _contexts(record)
                result["context_count"] = len(contexts)
                async with semaphore:
                    result["context_tokens"] = await _token_count(
                        client, endpoint, model, rendered
                    )
                    prompt = ANSWER_PROMPT.format(
                        context=rendered, question=record.get("question")
                    )
                    answer, usage = await _chat(
                        client, endpoint, model, prompt, max_tokens=128
                    )
                result.update(
                    {"success": True, "generated_answer": answer, "usage": usage}
                )
            except Exception as exc:  # noqa: BLE001 - quality failures are data
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["completed_at"] = _now()
            result["wall_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            await sink.write(result)

        await asyncio.gather(*(one(record) for record in records))
    sink.close()
    return [value for _, value in sorted(_load_keyed(output).items())]


def _judge_label(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
        label = str(payload.get("label", "")).upper().strip()
        if label in {"CORRECT", "WRONG"}:
            return label
    except (json.JSONDecodeError, AttributeError):
        pass
    if candidate.upper() == "CORRECT":
        return "CORRECT"
    return "WRONG"


async def judge_answers(
    answers: Sequence[dict[str, Any]],
    *,
    output: Path,
    endpoint: str,
    model: str,
    workers: int,
) -> list[dict[str, Any]]:
    existing = _load_keyed(output)
    sink = _AsyncJsonlSink(output)
    semaphore = asyncio.Semaphore(workers)
    timeout = httpx.Timeout(60.0)
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def one(answer: dict[str, Any]) -> None:
            index = int(answer["request_index"])
            if index in existing:
                return
            started = time.perf_counter_ns()
            result = {
                "schema_version": "table5_track_c_judge_v0.2.0",
                "request_index": index,
                "system": answer.get("system"),
                "build_id": answer.get("build_id"),
                "sample_id": answer.get("sample_id"),
                "question_id": answer.get("question_id"),
                "category": answer.get("category"),
                "answer_success": answer.get("success") is True,
                "success": False,
                "correct": False,
                "label": "WRONG",
                "error": None,
                "started_at": _now(),
            }
            try:
                if answer.get("success") is not True:
                    raise RuntimeError("answer_generation_failed")
                prompt = JUDGE_PROMPT.format(
                    question=answer.get("question"),
                    gold_answer=answer.get("gold_answer"),
                    generated_answer=answer.get("generated_answer"),
                )
                async with semaphore:
                    raw, usage = await _chat(
                        client, endpoint, model, prompt, max_tokens=48
                    )
                label = _judge_label(raw)
                result.update(
                    {
                        "success": True,
                        "correct": label == "CORRECT",
                        "label": label,
                        "judge_raw": raw,
                        "usage": usage,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - quality failures are data
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["completed_at"] = _now()
            result["wall_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            await sink.write(result)

        await asyncio.gather(*(one(answer) for answer in answers))
    sink.close()
    return [value for _, value in sorted(_load_keyed(output).items())]


def _summary_evidence(
    answers: Sequence[dict[str, Any]], judges: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Recompute every deterministic quality statistic from raw records."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in judges:
        by_category[str(record.get("category"))].append(record)

    def score(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        successful = [record for record in records if record.get("success") is True]
        correct = sum(record.get("correct") is True for record in successful)
        return {
            "total": len(records),
            "evaluated": len(successful),
            "correct": correct,
            "score": correct / len(records) if records else None,
            "conditional_score": correct / len(successful) if successful else None,
            "coverage": len(successful) / len(records) if records else 0.0,
        }

    context_tokens = [
        int(record["context_tokens"])
        for record in answers
        if record.get("context_tokens") is not None
    ]
    context_tokens_total = sum(context_tokens)
    context_token_records = len(context_tokens)
    return {
        "schema_version": "table5_track_c_quality_summary_v0.2.0",
        "answer_prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
        "answer_prompt_source": (
            "neutralized common-context adaptation of MemOS 2.0.30 "
            "evaluation/scripts/locomo/prompts.py"
        ),
        "overall": score(judges),
        "by_category": {
            category: score(records)
            for category, records in sorted(by_category.items())
        },
        "answer_generation_success": sum(
            record.get("success") is True for record in answers
        )
        / len(answers)
        if answers
        else 0.0,
        "context_tokens_total": context_tokens_total,
        "context_token_records": context_token_records,
        "context_token_coverage": context_token_records / len(answers)
        if answers
        else 0.0,
        "mean_context_tokens": context_tokens_total / context_token_records
        if context_token_records
        else None,
    }


def _summary(
    answers: Sequence[dict[str, Any]], judges: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    return {**_summary_evidence(answers, judges), "completed_at": _now()}


def _validate_quality_records(
    records: Sequence[dict[str, Any]], *, system: str, build_id: str, label: str
) -> None:
    indices = [int(record.get("request_index", -1)) for record in records]
    if indices != list(range(EXPECTED_FORMAL_QUESTION_COUNT)):
        raise RuntimeError(f"{label} records are not the exact ordered formal set")
    if any(
        record.get("system") != system or record.get("build_id") != build_id
        for record in records
    ):
        raise RuntimeError(f"{label} system/build identity mismatch")


def _raw_quality_contracts(
    answers: Sequence[dict[str, Any]], judges: Sequence[dict[str, Any]]
) -> dict[str, bool]:
    """Validate outcome semantics and the lossless answer-to-judge linkage."""
    answers_by_index = {int(row["request_index"]): row for row in answers}
    judges_by_index = {int(row["request_index"]): row for row in judges}

    def non_empty(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    answer_outcomes = all(
        isinstance(row.get("retrieval_success"), bool)
        and non_empty(row.get("sample_id"))
        and non_empty(str(row.get("question_id") or ""))
        and (
            row.get("success") is True
            and row.get("retrieval_success") is True
            and non_empty(row.get("generated_answer"))
            and isinstance(row.get("context_count"), int)
            and row["context_count"] >= 0
            and isinstance(row.get("context_tokens"), int)
            and row["context_tokens"] >= 0
            and row.get("error") is None
            or row.get("success") is False
            and non_empty(row.get("error"))
        )
        for row in answers
    )
    judge_outcomes = all(
        isinstance(row.get("answer_success"), bool)
        and non_empty(row.get("sample_id"))
        and non_empty(str(row.get("question_id") or ""))
        and (
            row.get("success") is True
            and row.get("answer_success") is True
            and row.get("label") in {"CORRECT", "WRONG"}
            and row.get("correct") is (row.get("label") == "CORRECT")
            and row.get("error") is None
            or row.get("success") is False
            and row.get("correct") is False
            and row.get("label") == "WRONG"
            and non_empty(row.get("error"))
        )
        for row in judges
    )
    linkage = answers_by_index.keys() == judges_by_index.keys() and all(
        judges_by_index[index].get(field) == answer.get(field)
        for index, answer in answers_by_index.items()
        for field in ("system", "build_id", "sample_id", "question_id", "category")
    )
    answer_success_linkage = linkage and all(
        judges_by_index[index].get("answer_success") is (answer.get("success") is True)
        for index, answer in answers_by_index.items()
    )
    return {
        "answer_outcome_contract": answer_outcomes,
        "judge_outcome_contract": judge_outcomes,
        "answer_judge_identity_linkage": linkage,
        "answer_judge_success_linkage": answer_success_linkage,
    }


def verify_quality_tree(
    root: str | Path,
    *,
    expected_runs: set[tuple[str, str]],
    expected_code_hash: str,
) -> dict[str, Any]:
    root = Path(root)
    runs = []
    for receipt_path in sorted(root.glob("**/quality_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        summary_path = receipt_path.parent / "quality_summary.json"
        answers_path = receipt_path.parent / "answers.jsonl"
        judges_path = receipt_path.parent / "judges.jsonl"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {}
        )
        answers = read_jsonl([answers_path]) if answers_path.is_file() else []
        judges = read_jsonl([judges_path]) if judges_path.is_file() else []
        recomputed_summary = _summary_evidence(answers, judges)
        raw_contracts = _raw_quality_contracts(answers, judges)
        identity = (str(receipt.get("system")), str(receipt.get("build_id")))
        expected_indices = list(range(EXPECTED_FORMAL_QUESTION_COUNT))
        answer_indices = [record.get("request_index") for record in answers]
        judge_indices = [record.get("request_index") for record in judges]
        checks = {
            "receipt_schema_frozen": receipt.get("schema_version")
            == "table5_track_c_quality_run_v0.2.0",
            "receipt_complete": receipt.get("status") == "complete",
            "expected_identity": identity in expected_runs,
            "code_hash_frozen": receipt.get("benchmark_code_sha256")
            == expected_code_hash,
            "formal_cardinality": receipt.get("formal_records")
            == EXPECTED_FORMAL_QUESTION_COUNT,
            "model_frozen": receipt.get("model") == "Qwen/Qwen3-32B",
            "endpoint_frozen": receipt.get("endpoint") == "http://127.0.0.1:8000/v1",
            "workers_frozen": receipt.get("workers") == 32,
            "answer_prompt_frozen": receipt.get("answer_prompt_sha256")
            == hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
            "judge_prompt_frozen": receipt.get("judge_prompt_sha256")
            == hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
            "answers_hash_matches": receipt.get("answers_sha256")
            == (_sha256(answers_path) if answers_path.is_file() else None),
            "judges_hash_matches": receipt.get("judges_sha256")
            == (_sha256(judges_path) if judges_path.is_file() else None),
            "summary_hash_matches": receipt.get("summary_sha256")
            == (_sha256(summary_path) if summary_path.is_file() else None),
            "summary_identity_matches": summary.get("system") == identity[0]
            and summary.get("build_id") == identity[1],
            "summary_schema_frozen": summary.get("schema_version")
            == "table5_track_c_quality_summary_v0.2.0",
            "summary_total_exact": (summary.get("overall") or {}).get("total")
            == EXPECTED_FORMAL_QUESTION_COUNT,
            "summary_evidence_recomputed": all(
                summary.get(key) == value for key, value in recomputed_summary.items()
            ),
            "receipt_overall_matches_summary": receipt.get("overall")
            == summary.get("overall"),
            "context_token_accounting": isinstance(
                summary.get("context_tokens_total"), int
            )
            and summary["context_tokens_total"] >= 0
            and isinstance(summary.get("context_token_records"), int)
            and 0 <= summary["context_token_records"] <= EXPECTED_FORMAL_QUESTION_COUNT
            and summary.get("context_token_coverage")
            == summary["context_token_records"] / EXPECTED_FORMAL_QUESTION_COUNT
            and (
                summary.get("mean_context_tokens")
                == summary["context_tokens_total"] / summary["context_token_records"]
                if summary["context_token_records"]
                else summary.get("mean_context_tokens") is None
            ),
            "answer_records_exact": sorted(answer_indices) == expected_indices
            and len(set(answer_indices)) == len(answer_indices)
            and all(
                record.get("schema_version") == "table5_track_c_answer_v0.2.0"
                and record.get("system") == identity[0]
                and record.get("build_id") == identity[1]
                for record in answers
            ),
            "judge_records_exact": sorted(judge_indices) == expected_indices
            and len(set(judge_indices)) == len(judge_indices)
            and all(
                record.get("schema_version") == "table5_track_c_judge_v0.2.0"
                and record.get("system") == identity[0]
                and record.get("build_id") == identity[1]
                for record in judges
            ),
            **raw_contracts,
        }
        formal_path = Path(str(receipt.get("formal_path", "")))
        checks["formal_hash_matches"] = formal_path.is_file() and receipt.get(
            "formal_sha256"
        ) == _sha256(formal_path)
        runs.append(
            {
                "receipt": str(receipt_path),
                "system": identity[0],
                "build_id": identity[1],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    actual_runs = {(run["system"], run["build_id"]) for run in runs}
    return {
        "schema_version": "table5_track_c_quality_verification_v0.1.0",
        "root": str(root.resolve()),
        "runs_found": len(runs),
        "expected_runs": [list(value) for value in sorted(expected_runs)],
        "actual_runs": [list(value) for value in sorted(actual_runs)],
        "coverage_exact": actual_runs == expected_runs,
        "all_found_runs_pass": bool(runs) and all(run["passed"] for run in runs),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--expected-code-hash", required=True)
    args = parser.parse_args(argv)
    current_code_hash = benchmark_code_sha256()
    if current_code_hash != args.expected_code_hash:
        raise RuntimeError(
            "quality code does not match the frozen run hash: "
            f"{current_code_hash} != {args.expected_code_hash}"
        )
    formal_path = Path(args.formal).resolve()
    run_receipt_path = formal_path.parent / "run_receipt.json"
    if not run_receipt_path.is_file():
        raise RuntimeError(
            f"missing Search receipt beside quality input: {formal_path}"
        )
    search_verification = verify_run(run_receipt_path)
    if not search_verification["passed"]:
        raise RuntimeError("quality input Search run failed protocol verification")
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    if (
        run_receipt.get("execution", {}).get("benchmark_code_sha256")
        != args.expected_code_hash
    ):
        raise RuntimeError("quality input Search run has the wrong frozen code hash")
    records = read_jsonl([formal_path])
    indices = [record.get("request_index") for record in records]
    if len(records) != EXPECTED_FORMAL_QUESTION_COUNT:
        raise RuntimeError(
            f"quality input must contain {EXPECTED_FORMAL_QUESTION_COUNT} records, "
            f"found {len(records)}"
        )
    if sorted(indices) != list(range(EXPECTED_FORMAL_QUESTION_COUNT)) or len(
        set(indices)
    ) != len(indices):
        raise RuntimeError("quality input request indices are not exact and unique")
    records = sorted(records, key=lambda record: int(record["request_index"]))
    systems = {str(record.get("system")) for record in records}
    build_ids = {str(record.get("build_id")) for record in records}
    phases = {str(record.get("phase")) for record in records}
    if len(systems) != 1 or len(build_ids) != 1 or phases != {"formal_search"}:
        raise RuntimeError(
            f"quality input identity mismatch: systems={systems}, "
            f"build_ids={build_ids}, phases={phases}"
        )
    system = next(iter(systems))
    build_id = next(iter(build_ids))
    output = Path(args.output_dir)
    receipt_path = output / "quality_receipt.json"
    identity = {
        "system": system,
        "build_id": build_id,
        "formal_path": str(formal_path),
        "formal_sha256": _sha256(formal_path),
        "formal_records": len(records),
        "model": args.model,
        "endpoint": args.endpoint,
        "workers": args.workers,
        "benchmark_code_sha256": current_code_hash,
        "answer_prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode()).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest(),
    }
    partial_paths = [
        output / "answers.jsonl",
        output / "judges.jsonl",
        output / "quality_summary.json",
    ]
    if not receipt_path.exists() and any(path.exists() for path in partial_paths):
        raise RuntimeError("quality output exists without its identity receipt")
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            raise FileExistsError(f"quality run is already complete: {receipt_path}")
        for key, value in identity.items():
            if previous.get(key) != value:
                raise RuntimeError(
                    f"quality resume identity mismatch for {key}: "
                    f"{previous.get(key)!r} != {value!r}"
                )
    receipt = {
        "schema_version": "table5_track_c_quality_run_v0.2.0",
        **identity,
        "status": "running",
        "started_at": _now(),
    }
    _write_json(receipt_path, receipt)
    try:
        answers = asyncio.run(
            generate_answers(
                records,
                output=output / "answers.jsonl",
                endpoint=args.endpoint,
                model=args.model,
                workers=args.workers,
            )
        )
        judges = asyncio.run(
            judge_answers(
                answers,
                output=output / "judges.jsonl",
                endpoint=args.endpoint,
                model=args.model,
                workers=args.workers,
            )
        )
        _validate_quality_records(
            answers, system=system, build_id=build_id, label="answer"
        )
        _validate_quality_records(
            judges, system=system, build_id=build_id, label="judge"
        )
        summary = _summary(answers, judges)
        summary.update(identity)
        _write_json(output / "quality_summary.json", summary)
        receipt.update(
            {
                "status": "complete",
                "completed_at": _now(),
                "answers_sha256": _sha256(output / "answers.jsonl"),
                "judges_sha256": _sha256(output / "judges.jsonl"),
                "summary_sha256": _sha256(output / "quality_summary.json"),
                "overall": summary["overall"],
            }
        )
        _write_json(receipt_path, receipt)
    except BaseException as exc:
        receipt.update(
            {
                "status": "failed",
                "failed_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(receipt_path, receipt)
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # Per-request answer/judge failures are measured data and already score as
    # zero in the denominator. A complete receipt is therefore a successful
    # phase execution even when coverage is below 100%.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
