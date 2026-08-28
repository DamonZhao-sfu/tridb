"""Experiment C: LoCoMo Table-3-shaped accuracy under a local answer backbone.

    python3 -m experiments.e2.expc_locomo_table3

WHY THIS IS NOT `table5_track_c.quality`
----------------------------------------
`quality.py` is gated on `benchmark_code_sha256()` matching the frozen hash in
the Search receipt.  That gate protects Track C's *latency* claims, and it is
working as designed: `bench/agent_memory/**` has changed since the sweep (the
`gem_eg` W1 work landed), so the hash no longer matches and `quality.py` refuses.

This driver makes no Track C claim.  It reuses `quality.py`'s ANSWER_PROMPT,
JUDGE_PROMPT, answer/judge coroutines and scorer *by import*, so there is no
prompt drift, and skips only the frozen-protocol gate.  Output is labelled
`experiment_c`, never a Track C artifact.

JUDGE
-----
Answer and judge are the SAME model (qwen3.8), on the same GPU-0 endpoint.
The paper does not specify a judge model -- it adopts EverMemOS's correctness
script, with GPT-4o-mini / GPT-4.1-mini as the backbones -- so no baseline
requires a separate judge.  Every system's answers are produced by the same
backbone, so self-preference bias is uniform across rows and does not reorder
them; it can inflate all four accuracies together.  Recorded, not corrected.
Keeping the judge on port 8000 also leaves GPU 1 untouched.

SCOPE
-----
Mandol (arXiv:2606.29778) Table 3 reports n=1,540: all 1,986 LoCoMo questions
minus the 446 adversarial (category 5).  Track C's formal set is 1,787 — the
same 1,986 minus the 199 questions of the `conv-26` warmup conversation.  The
intersection is 1,388 questions over 9 conversations.  That is a strict subset
of Mandol's column definition and is reported as such; it is NOT n=1,540.

Category map (LoCoMo convention, matching Mandol's column headers):
    1 -> Multi   2 -> Temp.   3 -> Open   4 -> Single   5 -> adversarial (dropped)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.agent_memory.table5_track_c.protocol import benchmark_code_sha256
from bench.agent_memory.table5_track_c.quality import (
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    generate_answers,
    judge_answers,
)
from bench.agent_memory.table5_track_c import quality as _quality
from bench.agent_memory.table5_track_c.stats import read_jsonl


class _TimeoutShim:
    """Stand-in for the `httpx` module that only overrides `Timeout`.

    Everything else is delegated to the real module via `__getattr__`.
    """

    def __init__(self, module: Any, seconds: float) -> None:
        self._module = module
        self._seconds = seconds
        # Bind the ORIGINAL class now. Reading it lazily would resolve back to
        # whatever `Timeout` currently names -- see the recursion note below.
        self._timeout_cls = module.Timeout

    def Timeout(self, *_args: Any, **_kwargs: Any) -> Any:  # noqa: N802
        return self._timeout_cls(self._seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


def _widen_timeout(seconds: float) -> None:
    """`quality.py` hardcodes httpx.Timeout(60.0).

    Cognee returns ~8.4k-token contexts (median) against ~1.2k for Mem0 and
    ~2.1k for TriDB/GEM, so its prefill under concurrency blows past 60 s and
    every request dies with ReadTimeout -- a harness artifact scored as a wrong
    answer.  Widen the timeout inside quality's module namespace instead of
    editing Track C source, which would move `benchmark_code_sha256` again.

    Rebind only `_quality.httpx`, never `httpx.Timeout` itself. The previous
    version did the latter:

        _quality.httpx.Timeout = lambda *a, **k: _httpx.Timeout(seconds)

    `_quality.httpx` and `_httpx` are the same module object, so after the
    assignment the lambda's body resolved `_httpx.Timeout` to the lambda --
    it called itself. `httpx.AsyncClient(...)` then never returned and the
    process grew past 60 GB before the OOM killer took it, with no request
    ever reaching the endpoint.
    """
    import httpx as _httpx

    _quality.httpx = _TimeoutShim(_httpx, seconds)

SWEEP = Path("bench/out/table5_track_c_qps_sweep_snapshot_2026_08_20_v2/runs")
SYSTEMS = {
    "tridb_gem": "TriDB/GEM",
    "mem0": "Mem0 2.0.18",
    "cognee": "Cognee 1.5.0",
    "memos": "MemOS 2.0.30",
    "graphiti": "Graphiti (Zep OSS proxy)",
    "evermemos": "EverMemOS (EverOS 1.2.3)",
}
#: Systems whose Search artifacts were produced outside the frozen Track C
#: sweep and therefore do not live under SWEEP/<system>/search/qps_N. The
#: retrieval protocol and the qps_1 operating point are the same; only the
#: run directory differs, because these builds were driven separately.
SYSTEM_ROOTS = {
    "graphiti": Path("bench/out/expc_graphiti_locomo_2026_08_21"),
    "evermemos": Path("bench/out/expc_evermemos_locomo_2026_08_21"),
}
#: Mandol Table 3 column order.
COLUMNS = [(4, "Single"), (1, "Multi"), (2, "Temp."), (3, "Open")]
ADVERSARIAL = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def search_artifact(system: str, qps: int) -> Path:
    root = SYSTEM_ROOTS.get(system)
    if root is not None:
        return root / "search" / f"qps_{qps}" / "formal.jsonl"
    return SWEEP / system / "search" / f"qps_{qps}" / "formal.jsonl"


def load_records(system: str, qps: int) -> list[dict[str, Any]]:
    path = search_artifact(system, qps)
    if not path.is_file():
        raise SystemExit(f"missing Search artifact: {path}")
    records = read_jsonl([path])
    kept = [r for r in records if r.get("category") != ADVERSARIAL]
    return sorted(kept, key=lambda r: int(r["request_index"]))


def score(system: str, judges: list[dict[str, Any]], answers: list[dict[str, Any]]):
    by_cat: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in judges:
        by_cat[record.get("category")].append(record)

    def acc(records) -> dict[str, Any]:
        # Denominator is every question in the cell.  A retrieval or answer
        # failure counts as wrong, never as an excluded sample.
        total = len(records)
        correct = sum(r.get("correct") is True for r in records)
        return {
            "n": total,
            "correct": correct,
            "accuracy": round(100.0 * correct / total, 2) if total else None,
        }

    tokens = [
        a.get("context_tokens") for a in answers if isinstance(a.get("context_tokens"), int)
    ]
    return {
        "system": SYSTEMS[system],
        "avg_context_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "token_records": len(tokens),
        "by_column": {name: acc(by_cat.get(cat, [])) for cat, name in COLUMNS},
        "overall": acc(judges),
    }


async def run_system(system: str, args) -> dict[str, Any]:
    records = load_records(system, args.qps)
    out = Path(args.output_dir) / system
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{system}] {len(records)} questions (category 5 excluded)", flush=True)

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


async def amain(args) -> int:
    summaries = []
    for system in args.systems:
        summaries.append(await run_system(system, args))
    report = {
        "schema_version": "experiment_c_locomo_table3_v0.1.0",
        "generated_at": _now(),
        "claim": (
            "Backbone-transfer measurement under a local answer model. NOT a "
            "reproduction of Mandol Table 3: different backbone, different judge, "
            "and n=1388 (9 conversations) vs the paper's n=1540 (10)."
        ),
        "answer_model": args.answer_model,
        "answer_endpoint": args.answer_endpoint,
        "judge_model": args.judge_model,
        "judge_endpoint": args.judge_endpoint,
        "judge_is_answer_model": args.judge_model == args.answer_model,
        "self_preference_note": (
            "judge == answer model; bias is uniform across systems because all "
            "answers come from the same backbone, so ranking is preserved"
        ),
        "thinking": "disabled",
        "request_timeout_s": args.request_timeout,
        "temperature": 0.0,
        "qps_source": args.qps,
        "source_sweep": str(SWEEP),
        # Per-system provenance: not every row comes from the frozen sweep,
        # so a single source path would misdescribe the table.
        "search_artifacts": {
            SYSTEMS[name]: str(search_artifact(name, args.qps))
            for name in args.systems
        },
        "benchmark_code_sha256_at_scoring": benchmark_code_sha256(),
        "answer_prompt_sha": __import__("hashlib").sha256(
            ANSWER_PROMPT.encode()
        ).hexdigest()[:16],
        "judge_prompt_sha": __import__("hashlib").sha256(
            JUDGE_PROMPT.encode()
        ).hexdigest()[:16],
        "systems": summaries,
    }
    path = Path(args.output_dir) / "table3_local.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    header = f"{'System':16s} {'Avg.Tok':>8s} " + " ".join(
        f"{name:>7s}" for _, name in COLUMNS
    ) + f" {'Overall':>8s} {'n':>6s}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        cells = " ".join(
            f"{(s['by_column'][name]['accuracy'] or 0):7.2f}" for _, name in COLUMNS
        )
        print(
            f"{s['system']:16s} {(s['avg_context_tokens'] or 0):8.0f} {cells} "
            f"{(s['overall']['accuracy'] or 0):8.2f} {s['overall']['n']:6d}"
        )
    print(f"\nreport: {path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    p.add_argument("--qps", type=int, default=1)
    p.add_argument("--output-dir", default="bench/out/expc_qwen38_locomo_2026_08_21")
    p.add_argument("--answer-endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--answer-model", default="qwen3.8")
    p.add_argument("--judge-endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--judge-model", default="qwen3.8")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="per-request HTTP timeout; 60 s (quality.py's default) is too "
        "short for large-context systems such as Cognee",
    )
    p.add_argument("--judge-workers", type=int, default=8)
    args = p.parse_args()
    _widen_timeout(args.request_timeout)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
