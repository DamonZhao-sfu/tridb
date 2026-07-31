"""CLI for the GEM LongMemEval reproduction of [AM] §4.1, §4.2 and §4.8.

    python -m bench.agent_memory.gem_bench \\
      --input data/longmemeval/memoryagentbench_longmemeval_sstar.json \\
      --output-dir bench/out/gem_longmemeval

Arms run one after another against the same serving stack, and each writes its
own directory of raw records plus a ``summary.json``. The cross-arm tables land
in ``paper_sections.json`` and ``report.md`` at the top level.

The judge defaults to the LOCAL answer endpoint. That makes grading free and
offline, and it makes the protocol a VARIANT of MemoryAgentBench's (which grades
with hosted gpt-4o) — ``judge_protocol`` says so in every summary, so accuracy
here is comparable across these arms and not against [AM]'s published numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from bench.agent_memory.energy import DEFAULT_INTERVAL_SECONDS, GpuEnergySampler
from bench.agent_memory.gem.store import DEFAULT_DSN
from bench.agent_memory.gem_bench import points as pointsmod
from bench.agent_memory.gem_bench import report as reportmod
from bench.agent_memory.gem_bench.runner import run_point
from bench.agent_memory.gem_bench.summarize import paper_sections
from bench.agent_memory.serving import (
    DEFAULT_ANSWER_BASE_URL,
    DEFAULT_ANSWER_MODEL,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SOURCE,
    CallLedger,
    LongMemEvalWorkload,
    OpenAIChatClient,
    OpenAIEmbeddingClient,
    _atomic_write_json,
    _git_state,
    _sha256,
    _utc_now,
    _validate_single_model,
    judge_protocol_label,
    load_workloads,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--dsn", default=os.environ.get("TRIDB_GEM_DSN", DEFAULT_DSN))
    parser.add_argument(
        "--points",
        nargs="*",
        default=None,
        help=(
            "operating points to run (default: all). "
            f"choices: {', '.join(pointsmod.POINTS_BY_KEY)}"
        ),
    )

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-prompt-memories", type=int, default=5)
    parser.add_argument("--prompt-token-budget", type=int, default=36_000)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--chunk-tokenizer", default="gpt-4o-mini")
    parser.add_argument(
        "--term-cond",
        type=int,
        default=32,
        help="tjs_open early-termination budget for the FUSED retrieval leg",
    )

    parser.add_argument(
        "--answer-base-url",
        default=os.environ.get("VLLM_BASE_URL", DEFAULT_ANSWER_BASE_URL),
    )
    parser.add_argument(
        "--answer-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY")
    )
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument(
        "--construction-max-tokens",
        type=int,
        default=1024,
        help="cap on each construction extraction/tool response",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL),
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.environ.get("EMBEDDING_API_KEY", "EMPTY"),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=600.0)

    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--allow-nonstandard-shape", action="store_true")

    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("JUDGE_BASE_URL", DEFAULT_ANSWER_BASE_URL),
        help="defaults to the local answer endpoint",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("JUDGE_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --answer-model (the local judge)",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=10)

    parser.add_argument(
        "--max-rejection-rate",
        type=float,
        default=None,
        help=(
            "[AM] §4.4 capability floor: abort a history whose construction "
            "rejection rate exceeds this, rather than reporting a corrupted "
            "store as an accuracy datapoint"
        ),
    )
    parser.add_argument("--revise-max-hops", type=int, default=3)
    parser.add_argument(
        "--revise-max-evidence",
        type=int,
        default=0,
        help="0 = every ingested unit is revision evidence",
    )
    parser.add_argument(
        "--agentic-max-rounds",
        type=int,
        default=pointsmod.DEFAULT_AGENTIC_MAX_ROUNDS,
    )
    parser.add_argument(
        "--agentic-max-tool-calls",
        type=int,
        default=pointsmod.DEFAULT_AGENTIC_MAX_TOOL_CALLS,
    )

    parser.add_argument(
        "--energy-interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="NVML poll period in seconds",
    )
    parser.add_argument(
        "--no-energy",
        action="store_true",
        help="skip NVML sampling; Table 3's energy columns are then absent",
    )
    parser.add_argument(
        "--energy-devices",
        type=int,
        nargs="*",
        default=None,
        help="NVML device indices to attribute (default: CUDA_VISIBLE_DEVICES, else all)",
    )
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("top_k", "max_prompt_memories", "prompt_token_budget", "term_cond"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_prompt_memories > args.top_k:
        parser.error("--max-prompt-memories cannot exceed --top-k")
    if args.limit_samples is not None and args.limit_samples <= 0:
        parser.error("--limit-samples must be positive")
    if args.limit_questions is not None and args.limit_questions <= 0:
        parser.error("--limit-questions must be positive")
    if args.revise_max_evidence < 0:
        parser.error("--revise-max-evidence must be non-negative")
    if args.max_rejection_rate is not None and not 0 <= args.max_rejection_rate <= 1:
        parser.error("--max-rejection-rate must lie in [0, 1]")
    if args.judge_model is None:
        args.judge_model = args.answer_model
    shares_endpoint = args.judge_base_url.rstrip("/") == args.answer_base_url.rstrip(
        "/"
    )
    if (
        not args.skip_judge
        and shares_endpoint
        and args.judge_model != args.answer_model
    ):
        parser.error(
            "--judge-model must equal --answer-model when the judge shares the "
            "answer endpoint; that endpoint serves exactly one model"
        )


def _apply_limits(
    workloads: list[LongMemEvalWorkload], args: argparse.Namespace
) -> list[LongMemEvalWorkload]:
    if args.limit_samples is not None:
        workloads = workloads[: args.limit_samples]
    if args.limit_questions is not None:
        workloads = [
            LongMemEvalWorkload(
                scope_id=workload.scope_id,
                source=workload.source,
                context=workload.context,
                questions=workload.questions[: args.limit_questions],
            )
            for workload in workloads
        ]
    return workloads


def _check_endpoints(args: argparse.Namespace) -> dict[str, Any]:
    """Fail before the first history rather than after five hours of ingest."""
    probe = CallLedger()
    answer_client = OpenAIChatClient(
        args.answer_base_url,
        args.answer_api_key,
        timeout=args.request_timeout,
        ledger=probe,
    )
    embedding_client = OpenAIEmbeddingClient(
        args.embedding_base_url,
        args.embedding_api_key,
        args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.request_timeout,
        ledger=probe,
    )
    advertised_answer = answer_client.discover_models()
    advertised_embedding = embedding_client.discover_models()
    _validate_single_model(
        advertised_answer, args.answer_model, endpoint_name="answer endpoint"
    )
    _validate_single_model(
        advertised_embedding, args.embedding_model, endpoint_name="embedding endpoint"
    )
    return {
        "answer_endpoint_models": getattr(answer_client, "model_records", []),
        "embedding_endpoint_models": getattr(embedding_client, "model_records", []),
    }


def build_manifest(
    args: argparse.Namespace,
    *,
    workloads: Sequence[LongMemEvalWorkload],
    selected_points: Sequence[pointsmod.OperatingPoint],
    endpoints: dict[str, Any],
    sampler: GpuEnergySampler,
) -> dict[str, Any]:
    return {
        "schema_version": "tridb_gem_longmemeval_manifest_v0.1.0",
        "started_at": _utc_now(),
        "paper": {
            "reference": "arXiv:2606.06448 — Agent Memory characterization",
            "sections_reproduced": ["4.1", "4.2", "4.8"],
            "sections_not_reproduced": ["4.7"],
            "systems_compared": (
                "TriDB/GEM operating points only; the paper's other nine memory "
                "systems are not run here"
            ),
        },
        "input": {
            "path": str(args.input.resolve()),
            "sha256": _sha256(args.input),
            "source": args.source,
        },
        "workload": {
            "histories": len(workloads),
            "questions_per_history": [len(w.questions) for w in workloads],
            "selected_questions": sum(len(w.questions) for w in workloads),
            "chunk_size_tokens": args.chunk_size,
            "chunk_tokenizer": args.chunk_tokenizer,
            "top_k": args.top_k,
            "max_prompt_memories": args.max_prompt_memories,
            "prompt_token_budget": args.prompt_token_budget,
            "embedding_batch_size": args.embedding_batch_size,
            "serial_execution": True,
        },
        "operating_points": [point.to_dict() for point in selected_points],
        "models": {
            "answer": args.answer_model,
            "answer_base_url": args.answer_base_url,
            "construction": args.answer_model,
            # Structural, not configurable: the extractor is built on the answer
            # client, which is what keeps [AM] §4.3's construction/generation
            # interference measurable instead of designed away.
            "construction_colocated_with_generation": True,
            "embedding": args.embedding_model,
            "embedding_base_url": args.embedding_base_url,
            "embedding_dim": args.embedding_dim,
            "judge": None if args.skip_judge else args.judge_model,
            "judge_base_url": None if args.skip_judge else args.judge_base_url,
            "thinking_enabled": False,
            **endpoints,
        },
        "generation": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.answer_max_tokens,
            "construction_max_tokens": args.construction_max_tokens,
            "stream": True,
        },
        "evaluation": {
            "judge_enabled": not args.skip_judge,
            "judge_protocol": judge_protocol_label(
                enabled=not args.skip_judge,
                model=args.judge_model,
                base_url=args.judge_base_url,
            ),
            "judge_max_tokens": args.judge_max_tokens,
            "judge_calls_excluded_from_serving_metrics": True,
        },
        "caps": {
            "agentic_max_rounds": args.agentic_max_rounds,
            "agentic_max_tool_calls": args.agentic_max_tool_calls,
            "retrieval_term_cond": args.term_cond,
            "max_rejection_rate": args.max_rejection_rate,
            "revise_max_hops": args.revise_max_hops,
            "revise_max_evidence": args.revise_max_evidence,
        },
        "energy": sampler.describe(),
        "tridb": {
            "dsn_redacted": re.sub(r"://[^@]+@", "://***@", args.dsn),
            "graph_read_visibility": "commit_visible",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_state(),
        },
        "comparability": {
            "paper_hardware_match": False,
            "note": (
                "Current-hardware TriDB reproduction. Absolute wallclock, joules "
                "and latency must not be presented as an H100 replication; the "
                "comparable quantity is the spread across arms measured here."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(args, parser)

    selected_points = pointsmod.resolve(args.points)
    workloads = _apply_limits(
        load_workloads(
            args.input,
            source=args.source,
            strict_shape=not args.allow_nonstandard_shape,
        ),
        args,
    )
    if sum(len(workload.questions) for workload in workloads) == 0:
        raise SystemExit("no questions selected")

    endpoints = _check_endpoints(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A disabled sampler is still a sampler: window() and describe() answer, and
    # every joule comes back None, which is the same shape as "no GPU here".
    sampler = GpuEnergySampler(
        interval_seconds=args.energy_interval,
        device_indices=args.energy_devices,
        enabled=not args.no_energy,
    )

    manifest = build_manifest(
        args,
        workloads=workloads,
        selected_points=selected_points,
        endpoints=endpoints,
        sampler=sampler,
    )
    _atomic_write_json(args.output_dir / "run_manifest.json", manifest)

    summaries: list[dict[str, Any]] = []
    sampler.start()
    try:
        for point in selected_points:
            point_dir = args.output_dir / point.key
            point_dir.mkdir(parents=True, exist_ok=True)
            print(f"[gem-bench] running {point.key}", file=sys.stderr, flush=True)
            summary = run_point(
                point,
                workloads=workloads,
                args=args,
                sampler=sampler,
                output_dir=point_dir,
            )
            summary["completed_at"] = _utc_now()
            _atomic_write_json(point_dir / "summary.json", summary)
            summaries.append(summary)
    finally:
        sampler.stop()

    sections = paper_sections(summaries)
    manifest["energy"] = sampler.describe()
    _atomic_write_json(args.output_dir / "run_manifest.json", manifest)
    paths = reportmod.write(sections, manifest, args.output_dir)
    print(json.dumps(sections, ensure_ascii=False, indent=2))
    for name, path in paths.items():
        print(f"{name}: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
