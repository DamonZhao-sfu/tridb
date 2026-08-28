"""Experiment C: LongMemEval Table-4-shaped accuracy under a local backbone.

    python3 -m experiments.e2.expc_longmemeval_table4 --systems tridb_gem ...

This is the Table 4 sibling of :mod:`experiments.e2.expc_locomo_table3`. It
reuses that module's answer/judge coroutines by import, so the prompts, the
scorer and the timeout shim are shared and cannot drift between the two
tables.

SCOPE
-----
Mandol (arXiv:2606.29778) Table 4 reports LongMemEval accuracy over 500
questions in six columns. Back-solving each published cell against the
category counts in ``longmemeval_s.json`` reproduces the paper's Overall
exactly (29/30 + 55/56 + 105/133 + 99/133 + 69/78 + 68/70 = 425/500 = 85.00%),
which confirms the paper scores the full set and does no sampling.

This driver therefore defaults to all 500. It is NOT a reproduction of the
paper's numbers: the backbone is a local Qwen3.8-27B rather than GPT-4o-mini /
GPT-4.1-mini, and the judge is that same local model. Only the *ranking* under
one common backbone is comparable.

ABSTENTION
----------
30 of the 500 ids end in ``_abs`` and are answerable only by refusing. They are
scored inside their own category like every other question -- that is what the
paper's counts imply -- but the receipt reports them separately so a reader can
see how much of a column rests on abstention behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.agent_memory.table5_track_c.longmemeval import (
    ABSTENTION_SUFFIX,
    COLUMN_ORDER,
    QUESTION_TYPE_TO_COLUMN,
)
from bench.agent_memory.table5_track_c.protocol import benchmark_code_sha256
from bench.agent_memory.table5_track_c.quality import (
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    generate_answers,
    judge_answers,
)
from bench.agent_memory.table5_track_c.stats import read_jsonl

from bench.agent_memory.table5_track_c import quality as _quality

from . import lme_alignment
from .expc_locomo_table3 import _widen_timeout

#: Where each system's LongMemEval Search artifacts live. Populated as systems
#: are run; a missing entry is a hard error rather than a silent skip.
SYSTEM_ROOTS: dict[str, Path] = {
    "tridb_gem": Path("bench/out/expc_lme_tridb_gem_2026_08_23"),
    "mem0": Path("bench/out/expc_lme_mem0_2026_08_23"),
    "memos": Path("bench/out/expc_lme_memos_2026_08_23"),
    "cognee": Path("bench/out/expc_lme_cognee_2026_08_23"),
    "graphiti": Path("bench/out/expc_lme_graphiti_2026_08_23"),
    "evermemos": Path("bench/out/expc_lme_evermemos_2026_08_23"),
}

SYSTEMS = {
    "tridb_gem": "TriDB/GEM",
    "mem0": "Mem0 2.0.18",
    "memos": "MemOS 2.0.30",
    "cognee": "Cognee 1.5.0",
    "graphiti": "Graphiti (Zep OSS proxy)",
    "evermemos": "EverMemOS (EverOS 1.2.3)",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def search_artifact(system: str, qps: int) -> Path:
    root = SYSTEM_ROOTS.get(system)
    if root is None:
        raise SystemExit(f"no LongMemEval run root registered for {system!r}")
    return root / "search" / f"qps_{qps}" / "formal.jsonl"


def load_records(system: str, qps: int) -> list[dict[str, Any]]:
    path = search_artifact(system, qps)
    if not path.is_file():
        raise SystemExit(f"missing LongMemEval Search artifact: {path}")
    records = read_jsonl([path])
    return sorted(records, key=lambda r: int(r["request_index"]))


def score(
    system: str, judges: list[dict[str, Any]], answers: list[dict[str, Any]]
) -> dict[str, Any]:
    by_column: dict[str, list[dict[str, Any]]] = defaultdict(list)
    abstention: list[dict[str, Any]] = []
    for record in judges:
        column = QUESTION_TYPE_TO_COLUMN.get(str(record.get("category")))
        if column is None:
            raise SystemExit(
                f"{system}: unmapped question_type {record.get('category')!r}"
            )
        by_column[column].append(record)
        if str(record.get("question_id", "")).endswith(ABSTENTION_SUFFIX):
            abstention.append(record)

    def acc(records: list[dict[str, Any]]) -> dict[str, Any]:
        # A retrieval or answer failure counts as wrong, never as an excluded
        # sample -- the denominator is every question in the cell.
        total = len(records)
        correct = sum(r.get("correct") is True for r in records)
        return {
            "n": total,
            "correct": correct,
            "accuracy": round(100.0 * correct / total, 2) if total else None,
        }

    tokens = [
        a.get("context_tokens")
        for a in answers
        if isinstance(a.get("context_tokens"), int)
    ]
    return {
        "system": SYSTEMS[system],
        "avg_context_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "token_records": len(tokens),
        "by_column": {name: acc(by_column.get(name, [])) for name in COLUMN_ORDER},
        "overall": acc(judges),
        "abstention": acc(abstention),
    }


async def run_system(system: str, args: argparse.Namespace) -> dict[str, Any]:
    records = load_records(system, args.qps)
    out = Path(args.output_dir) / system
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{system}] {len(records)} questions", flush=True)

    answers = await generate_answers(
        records,
        output=out / "answers.jsonl",
        endpoint=args.answer_endpoint,
        model=args.answer_model,
        workers=args.workers,
    )
    ok = sum(a.get("success") is True for a in answers)
    print(f"[{system}] answers {ok}/{len(answers)}", flush=True)

    judges = await judge_answers(
        answers,
        output=out / "judges.jsonl",
        endpoint=args.judge_endpoint,
        model=args.judge_model,
        workers=args.judge_workers,
    )
    summary = score(system, judges, answers)
    summary["judge_success"] = sum(j.get("success") is True for j in judges)
    summary["answer_success"] = ok
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[{system}] overall {summary['overall']}", flush=True)
    return summary


async def amain(args: argparse.Namespace) -> int:
    summaries = [await run_system(system, args) for system in args.systems]
    report = {
        "schema_version": "experiment_c_longmemeval_table4_v0.1.0",
        "generated_at": _now(),
        "dataset": "LongMemEval-S",
        "claim": (
            "Backbone-transfer measurement under a local answer model. NOT a "
            "reproduction of Mandol Table 4: different backbone and different "
            "judge. Only the ranking under one common backbone is comparable."
        ),
        "answer_model": args.answer_model,
        "answer_endpoint": args.answer_endpoint,
        "judge_model": args.judge_model,
        "judge_endpoint": args.judge_endpoint,
        "judge_is_answer_model": args.judge_model == args.answer_model,
        "thinking": "disabled",
        "grading_contract": "longmemeval_aligned_v1",
        "judge_prompt_source": (
            "Mandol benchmark_longmemeval/task_eval/evaluation.py "
            "MEM0_JUDGE_PROMPT, verbatim"
        ),
        "answer_prompt_source": (
            "written here: system-agnostic, carries question_date, no length "
            "cap, explicit licence to decline"
        ),
        "temperature": 0.0,
        "request_timeout_s": args.request_timeout,
        "qps_source": args.qps,
        "benchmark_code_sha256_at_scoring": benchmark_code_sha256(),
        "answer_prompt_sha": __import__("hashlib")
        .sha256(ANSWER_PROMPT.encode())
        .hexdigest()[:16],
        "judge_prompt_sha": __import__("hashlib")
        .sha256(JUDGE_PROMPT.encode())
        .hexdigest()[:16],
        "search_artifacts": {
            SYSTEMS[name]: str(search_artifact(name, args.qps))
            for name in args.systems
        },
        "systems": summaries,
    }
    path = Path(args.output_dir) / "table4_local.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    header = f"{'System':26s} {'Avg.Tok':>8s} " + " ".join(
        f"{name:>11s}" for name in COLUMN_ORDER
    ) + f" {'Overall':>8s} {'n':>6s}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        cells = " ".join(
            f"{(s['by_column'][name]['accuracy'] or 0):11.2f}" for name in COLUMN_ORDER
        )
        print(
            f"{s['system']:26s} {(s['avg_context_tokens'] or 0):8.0f} {cells} "
            f"{(s['overall']['accuracy'] or 0):8.2f} {s['overall']['n']:6d}"
        )
    print(f"\nreport: {path}")
    return 0


def _question_dates(dataset: str) -> dict[str, str]:
    """``question_id`` -> ``question_date``, joined at scoring time.

    The Search artifacts were written before this alignment existed and do not
    carry the date. Joining it here avoids re-running retrieval for every
    system just to add one field.
    """
    payload = json.loads(Path(dataset).read_text(encoding="utf-8"))
    return {
        str(r["question_id"]): str(r.get("question_date") or "")
        for r in payload
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    p.add_argument("--qps", type=int, default=1)
    p.add_argument("--output-dir", default="bench/out/expc_lme_table4_2026_08_22")
    p.add_argument("--answer-endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--answer-model", default="qwen3.8")
    p.add_argument("--judge-endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--judge-model", default="qwen3.8")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--judge-workers", type=int, default=6)
    p.add_argument(
        "--dataset",
        default=(
            "/local-scratch/localhome/hza214/tridb/data/longmemeval/"
            "longmemeval_s_cleaned.json"
        ),
        help="only read for question_date; retrieval already happened",
    )
    p.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="per-request HTTP timeout; LongMemEval contexts are larger than "
        "LoCoMo's, so quality.py's hardcoded 60 s is far too short",
    )
    args = p.parse_args()
    _widen_timeout(args.request_timeout)
    # LongMemEval is graded by its own contract, not LoCoMo's. See
    # experiments/e2/lme_alignment.py for why each piece differs.
    lme_alignment.apply(_quality, _question_dates(args.dataset))
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
