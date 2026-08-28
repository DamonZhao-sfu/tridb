"""Official-handler CrossEp-Tool runner with GEM source/transfer bank lifecycle."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

from bench.agent_memory.evomembench.crossep_tool_backend import (
    CrossEpToolGemMemory,
    MemoryUsageLog,
    ToolSampleContext,
    normalize_tool_trajectory,
)
from bench.agent_memory.evomembench.manifest import (
    load_manifest,
    manifest_sha256,
    verify_assets,
)
from bench.agent_memory.evomembench.injection import pad_to_token_budget
from bench.agent_memory.evomembench.long_context import (
    CrossEpToolLongContextMemory,
    LongContextSnapshot,
)
from bench.agent_memory.evomembench.protocol import (
    ordered_transfer_pairs,
    validate_tool_protocol,
)
from bench.agent_memory.evomembench.run_pilot import VLLMTokenCounter
from bench.agent_memory.evomembench.run_systems import _footprint
from bench.agent_memory.evomembench.system_snapshot import export_gem_snapshot
from bench.agent_memory.evomembench.system_protocol import (
    balanced_arm_order,
    SystemTrace,
    require_formal_authorization,
    sha256_text,
)
from bench.agent_memory.evomembench.tool_dataset import (
    TOOL_SOURCE_RELATIVE_PATH,
    load_crossep_tool,
    load_tool_category_ids,
)
from bench.agent_memory.evomembench.vllm_metrics import (
    collect as collect_vllm_metrics,
    collect_attributable_delta,
)
from bench.agent_memory.gem.memory import TriDBGovernedMemory
from bench.agent_memory.serving import (
    CallLedger,
    OpenAIEmbeddingClient,
    PhasedEmbedder,
)

ARMS = (
    "memory_off",
    "long_context",
    "recent_fifo",
    "vector_only",
    "graph_relational",
    "gem_fused",
)
IDS_RELATIVE_DIR = (
    "Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL/"
    "bfcl_eval/scripts/cross_episode/ids"
)
UPSTREAM_RELATIVE_ROOT = "Cross-Episode-Execution/Tool-Using/CROSSEP-TOOL"


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        output.flush()


def _read_prompt_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def _upstream(source_root: Path) -> tuple[Any, Any, Any]:
    root = source_root / UPSTREAM_RELATIVE_ROOT
    sys.path.insert(0, str(root))
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.scripts.cross_episode.func_doc_loader import load_function_docs
    from bfcl_eval.scripts.cross_episode.run_batch_generate import process_one_sample

    return OpenAICompletionsHandler, load_function_docs, process_one_sample


def _allowed_functions(load_function_docs: Any, entry: dict[str, Any]) -> list[str]:
    docs = load_function_docs(entry["involved_classes"], entry.get("excluded_function"))
    return [str(doc["name"]) for doc in docs]


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"evomembench_tool_{hash(path)}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _snapshot_count(snapshot: Any) -> int:
    if isinstance(snapshot, LongContextSnapshot):
        return len(snapshot.records)
    return int(snapshot.experience_count)


def _write_context_overflow_result(
    path: Path,
    *,
    source_id: str,
    category: str,
    entry: dict[str, Any],
    usage: Any,
) -> None:
    _append_jsonl(
        path,
        {
            "id": source_id,
            "result": [],
            "input_token_count": [],
            "output_token_count": [],
            "latency": [],
            "inference_log": [
                [
                    {
                        "role": "memory_log",
                        "content": "strict long-context baseline exceeded the model window; no truncation",
                    }
                ]
            ],
            "category": category,
            "involved_classes": entry["involved_classes"],
            "num_turns": len(entry["question"]),
            "per_turn_totals": [],
            "sample_totals": {},
            "memory_usage": asdict(usage),
            "force_quit": True,
            "full_message_history": [],
            "error": "context_overflow",
        },
    )


def _last_result(path: Path, sample_id: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or str(rows[-1].get("id")) != sample_id:
        raise RuntimeError(f"missing terminal Tool result for {sample_id}")
    return rows[-1]


def _results_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indexed = {str(row.get("id")): row for row in rows}
    if len(indexed) != len(rows) or "None" in indexed:
        raise RuntimeError(f"Tool results must have unique non-null IDs: {path}")
    return indexed


def _sum_numbers(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return sum(_sum_numbers(item) for item in value)
    if isinstance(value, dict):
        return sum(_sum_numbers(item) for item in value.values())
    return 0.0


def _jsonable(value: Any) -> Any:
    """Normalize SDK/Pydantic request values without changing their JSON shape."""
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _exact_first_tool_prompt_tokens(
    handler: Any,
    entry: dict[str, Any],
    *,
    token_counter: VLLMTokenCounter,
    injection: str = "",
) -> int:
    """Render the official FC first request, including tools, without calling it."""
    probe_entry = copy.deepcopy(entry)
    inference_data: dict[str, Any] = {}
    inference_data = handler._pre_query_processing_FC(inference_data, probe_entry)
    inference_data = handler._compile_tools(inference_data, probe_entry)
    first_turn = copy.deepcopy(probe_entry["question"][0])
    inference_data = handler.add_first_turn_message_FC(inference_data, first_turn)
    messages = [dict(message) for message in inference_data["message"]]
    if injection:
        if messages and str(messages[0].get("role", "")).casefold() == "system":
            messages[0]["content"] = str(messages[0].get("content", "")) + injection
        else:
            messages.insert(0, {"role": "system", "content": injection.strip()})
    return token_counter.count_chat(
        messages,
        tools=inference_data.get("tools") or None,
        chat_template_kwargs={"enable_thinking": False},
    )


def _write_system_trace(
    path: Path,
    *,
    args: argparse.Namespace,
    arm: str,
    phase: str,
    environment: str,
    source_environment: str,
    source_id: str,
    history_size: int,
    result: dict[str, Any],
    receipt: dict[str, Any] | None,
    model_trace: dict[str, int | float | bool],
    end_to_end_ms: float,
    canonical_source_replay: bool = False,
) -> None:
    if arm not in {"memory_off", "long_context", "gem_fused"}:
        return
    probes = dict((receipt or {}).get("probes") or {})
    database_instrumentation_ms = float(
        probes.get("instrumentation_probe_read_ms") or 0.0
    )
    total_telemetry_ms = (
        float(model_trace["telemetry_overhead_ms"]) + database_instrumentation_ms
    )
    probes.update(
        {
            "phase": phase,
            "source_environment": source_environment,
            "target_environment": environment,
            "model_request_bytes_semantics": "application JSON payload before OpenAI SDK encoding",
            "model_calls_observed": model_trace["calls"],
            "prompt_tokenization_calls": model_trace["prompt_tokenization_calls"],
            "prompt_tokens_recomputed": model_trace["prompt_tokens_recomputed"],
            "prompt_token_count_matches_usage": model_trace[
                "prompt_token_count_matches_usage"
            ],
            "prefix_cache_hit_tokens_available": model_trace["cache_detail_calls"] > 0,
            "vllm_timing_attributable": model_trace["timing_attributable"],
            "model_telemetry_overhead_ms": model_trace["telemetry_overhead_ms"],
            "database_instrumentation_ms": database_instrumentation_ms,
            "telemetry_overhead_ms": total_telemetry_ms,
            "raw_end_to_end_including_telemetry_ms": end_to_end_ms,
            "canonical_source_replay": canonical_source_replay,
        }
    )
    selected_ids = tuple(
        str(value) for value in (receipt or {}).get("selected_episode_uids", [])
    )
    injection_sha = str((receipt or {}).get("injection_sha256") or sha256_text(""))
    retrieval_ms = float((receipt or {}).get("latency_ms") or 0.0)
    history_episodes = (receipt or {}).get("history_episodes")
    history_bytes = (receipt or {}).get("history_bytes")
    intermediate: dict[str, int | float | None] = {
        "final_results": len(selected_ids),
        "client_to_model_request_bytes": model_trace["request_bytes"],
        "database_candidates": None,
    }
    if arm == "long_context":
        intermediate.update(
            {
                "history_episodes_materialized": int(history_episodes or 0),
                "history_prompt_bytes": int(history_bytes or 0),
            }
        )
    elif arm == "gem_fused":
        candidates = probes.get("candidates_examined")
        graph_examined = probes.get("graph_examined")
        intermediate.update(
            {
                "vector_candidates_examined": candidates,
                "graph_edges_examined": graph_examined,
                "graph_reached": probes.get("graph_reached"),
                "relational_candidates_examined": probes.get(
                    "relational_candidates_examined"
                ),
                "relational_candidates_survived": probes.get(
                    "relational_candidates_passed"
                ),
                "peak_application_materialized_ids": len(selected_ids),
                "peak_operator_items_proxy": max(
                    [len(selected_ids)]
                    + [
                        int(value)
                        for value in (candidates, graph_examined)
                        if isinstance(value, (int, float))
                    ],
                ),
                "peak_operator_items_proxy_is_exact": False,
                "cross_process_transfer_bytes": 0,
            }
        )
    trace = SystemTrace(
        run_id=args.run_id,
        track="CrossEp-Tool",
        arm=("full_gem" if arm == "gem_fused" else arm),
        target_id=(
            f"{source_environment}__to__{environment}:{source_id}"
            if phase == "transfer"
            else f"{environment}:{source_id}"
        ),
        scope_id=f"{args.run_id}:{arm}:tool:{source_environment}",
        history_size=history_size,
        status=("complete" if not result.get("error") else str(result["error"])),
        latency_ms={
            "memory_or_prompt_assembly": retrieval_ms,
            "query_embedding": float((receipt or {}).get("query_embedding_ms") or 0.0),
            "database_retrieval": float(
                (receipt or {}).get("database_retrieval_ms") or 0.0
            ),
            "prompt_serialization": float(model_trace["prompt_serialization_ms"]),
            "prompt_tokenization": float(model_trace["prompt_tokenization_ms"]),
            "generation": _sum_numbers(result.get("latency")) * 1000,
            "model_ttft": float(model_trace["model_ttft_ms"]),
            "model_prefill": float(model_trace["model_prefill_ms"]),
            "model_decode": float(model_trace["model_decode_ms"]),
            "model_server_end_to_end": float(model_trace["model_server_end_to_end_ms"]),
            "end_to_end": max(0.0, end_to_end_ms - total_telemetry_ms),
        },
        tokens={
            "answer_prompt": int(_sum_numbers(result.get("input_token_count"))),
            "answer_completion": int(_sum_numbers(result.get("output_token_count"))),
            "model_prefill": max(
                0,
                int(_sum_numbers(result.get("input_token_count")))
                - int(model_trace["cached_tokens"]),
            ),
            "memory_injection": int(
                (receipt or {}).get("injection_tokens")
                or (receipt or {}).get("history_tokens")
                or 0
            ),
            **(
                {"prefix_cache_hit": model_trace["cached_tokens"]}
                if model_trace["cache_detail_calls"] > 0
                else {}
            ),
            "query_embedding_input": int(
                (receipt or {}).get("query_embedding_input_tokens") or 0
            ),
        },
        intermediate=intermediate,
        selected_ids=selected_ids,
        injection_sha256=injection_sha,
        probes=probes,
    )
    _append_jsonl(path, trace.as_dict())


class TokenSlotString(str):
    """FC system slot whose ``+ memory`` operation performs exact replacement."""

    def __new__(cls, value: str, codec: Any, token_budget: int) -> "TokenSlotString":
        instance = super().__new__(cls, value)
        instance.codec = codec
        instance.token_budget = token_budget
        return instance

    def __add__(self, memory: object) -> str:
        replacement = str(memory)
        if len(self.codec.encode(replacement)) != self.token_budget:
            raise ValueError("Tool memory backend returned a non-matched token slot")
        return replacement

    def __deepcopy__(self, memo: dict[int, Any]) -> "TokenSlotString":
        memo[id(self)] = self
        return self


def _token_matched_entry(
    entry: dict[str, Any], *, codec: Any, token_budget: int
) -> dict[str, Any]:
    prepared = copy.deepcopy(entry)
    filler, _ = pad_to_token_budget("", token_budget=token_budget, codec=codec)
    prepared["question"][0].insert(
        0,
        {
            "role": "system",
            "content": TokenSlotString(filler, codec, token_budget),
        },
    )
    return prepared


def _uses_fixed_token_slot(*, arm: str, phase: str, query_mode: str) -> bool:
    """Only memory-retrieval target requests use GEM's fixed injection slot."""
    return (
        phase == "transfer"
        and query_mode == "gem_task_seed"
        and arm not in {"memory_off", "long_context"}
    )


class _SourceConstructionOnlyMemory:
    """Forward source writes while making source generation memory-blind.

    CrossEp-Tool's frozen bank must be derived from the same raw trajectories
    for Long Context and GEM.  The upstream handler insists on coupling
    ``utilize`` and ``update`` through one object, so this adapter suppresses
    retrieval while preserving the backend's measured construction path.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def begin_sample(self) -> None:
        self.backend.begin_sample()

    def utilize(self, _query: str) -> str:
        return ""

    def update(self, trajectory: list[dict]) -> None:
        self.backend.update(trajectory)

    def drain_usage(self) -> Any:
        return self.backend.drain_usage()


class _CanonicalTrajectoryRecorder:
    """Capture the online SDK-shaped trajectory without injecting memory."""

    def __init__(self) -> None:
        self.trajectories: dict[str, list[dict[str, Any]]] = {}
        self._sample_id: str | None = None

    def set_sample(self, sample_id: str) -> None:
        self._sample_id = sample_id

    def begin_sample(self) -> None:
        return None

    def utilize(self, _query: str) -> str:
        return ""

    def update(self, trajectory: list[Any]) -> None:
        if self._sample_id is None:
            raise RuntimeError("canonical trajectory recorder has no sample ID")
        if self._sample_id in self.trajectories:
            raise RuntimeError(f"duplicate canonical trajectory: {self._sample_id}")
        self.trajectories[self._sample_id] = normalize_tool_trajectory(trajectory)

    def drain_usage(self) -> MemoryUsageLog:
        return MemoryUsageLog()


def _replay_canonical_source_result(
    canonical_result: dict[str, Any],
    *,
    canonical_trajectory: list[dict[str, Any]],
    backend: Any,
) -> dict[str, Any]:
    """Construct one arm's source bank from the single canonical agent run."""

    result = copy.deepcopy(canonical_result)
    if not canonical_trajectory:
        raise RuntimeError("canonical source result has no raw trajectory")
    backend.begin_sample()
    backend.update(copy.deepcopy(canonical_trajectory))
    result["memory_usage"] = asdict(backend.drain_usage())
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_formal_authorization(
        formal=bool(getattr(args, "formal", False)),
        authorization=str(getattr(args, "authorization", "")),
    )
    if args.graph_work_budget < 128:
        raise ValueError("tjs.graph_work_budget must be at least 128")
    if not 1 <= args.episodes_per_environment <= 50:
        raise ValueError("episodes-per-environment must be within 1..50")
    if "memory_off" not in args.arms:
        raise ValueError(
            "CrossEp-Tool arms must include memory_off for paired outcomes"
        )
    source_root = Path(args.source_root).resolve()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)
    manifest = load_manifest(args.manifest)
    verified = verify_assets(source_root, manifest)
    manifest_hash = manifest_sha256(manifest)
    category_by_id = load_tool_category_ids(source_root / IDS_RELATIVE_DIR)
    corpus = load_crossep_tool(
        source_root / TOOL_SOURCE_RELATIVE_PATH,
        category_by_id=category_by_id,
    )
    category_counts = validate_tool_protocol(corpus, category_by_id)
    entries = _read_prompt_rows(source_root / TOOL_SOURCE_RELATIVE_PATH)
    episodes = {episode.source_id: episode for episode in corpus.episodes}
    ids_by_category = {
        category: [
            source_id
            for source_id, observed_category in category_by_id.items()
            if observed_category == category
        ][: args.episodes_per_environment]
        for category in category_counts
    }

    OpenAIHandler, load_function_docs, process_one_sample = _upstream(source_root)
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.answer_base_url

    class SeededOpenAIHandler(OpenAIHandler):
        def begin_system_trace(self) -> None:
            self._system_trace = {
                "calls": 0,
                "request_bytes": 0,
                "cached_tokens": 0,
                "cache_detail_calls": 0,
                "model_ttft_ms": 0.0,
                "model_prefill_ms": 0.0,
                "model_decode_ms": 0.0,
                "model_server_end_to_end_ms": 0.0,
                "telemetry_overhead_ms": 0.0,
                "prompt_serialization_ms": 0.0,
                "prompt_tokenization_ms": 0.0,
                "prompt_tokenization_calls": 0,
                "prompt_tokens_recomputed": 0,
                "prompt_token_count_matches_usage": True,
                "timing_attributable": True,
            }

        def drain_system_trace(self) -> dict[str, int | float | bool]:
            trace = dict(getattr(self, "_system_trace", {}))
            self._system_trace = {
                "calls": 0,
                "request_bytes": 0,
                "cached_tokens": 0,
                "cache_detail_calls": 0,
                "model_ttft_ms": 0.0,
                "model_prefill_ms": 0.0,
                "model_decode_ms": 0.0,
                "model_server_end_to_end_ms": 0.0,
                "telemetry_overhead_ms": 0.0,
                "prompt_serialization_ms": 0.0,
                "prompt_tokenization_ms": 0.0,
                "prompt_tokenization_calls": 0,
                "prompt_tokens_recomputed": 0,
                "prompt_token_count_matches_usage": True,
                "timing_attributable": True,
            }
            return {
                "calls": int(trace.get("calls", 0)),
                "request_bytes": int(trace.get("request_bytes", 0)),
                "cached_tokens": int(trace.get("cached_tokens", 0)),
                "cache_detail_calls": int(trace.get("cache_detail_calls", 0)),
                "model_ttft_ms": float(trace.get("model_ttft_ms", 0.0)),
                "model_prefill_ms": float(trace.get("model_prefill_ms", 0.0)),
                "model_decode_ms": float(trace.get("model_decode_ms", 0.0)),
                "model_server_end_to_end_ms": float(
                    trace.get("model_server_end_to_end_ms", 0.0)
                ),
                "telemetry_overhead_ms": float(trace.get("telemetry_overhead_ms", 0.0)),
                "prompt_serialization_ms": float(
                    trace.get("prompt_serialization_ms", 0.0)
                ),
                "prompt_tokenization_ms": float(
                    trace.get("prompt_tokenization_ms", 0.0)
                ),
                "prompt_tokenization_calls": int(
                    trace.get("prompt_tokenization_calls", 0)
                ),
                "prompt_tokens_recomputed": int(
                    trace.get("prompt_tokens_recomputed", 0)
                ),
                "prompt_token_count_matches_usage": bool(
                    trace.get("prompt_token_count_matches_usage", False)
                ),
                "timing_attributable": bool(trace.get("timing_attributable", False)),
            }

        def generate_with_backoff(self, **kwargs: Any) -> Any:
            kwargs.setdefault("seed", args.seed)
            kwargs.setdefault("max_tokens", args.reserved_generation_tokens)
            extra_body = dict(kwargs.get("extra_body") or {})
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
            chat_template_kwargs["enable_thinking"] = False
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            kwargs["extra_body"] = extra_body

            trace = getattr(self, "_system_trace", None)
            serialization_started = time.perf_counter()
            request_payload = _jsonable(kwargs)
            request_bytes = len(
                json.dumps(
                    request_payload, ensure_ascii=False, allow_nan=False
                ).encode()
            )
            serialization_ms = (time.perf_counter() - serialization_started) * 1000
            tokenization_started = time.perf_counter()
            prompt_tokens = token_counter.count_chat(
                request_payload["messages"],
                tools=request_payload.get("tools") or None,
                chat_template_kwargs={"enable_thinking": False},
            )
            tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
            if trace is not None:
                trace["calls"] += 1
                trace["request_bytes"] += request_bytes
                trace["prompt_serialization_ms"] += serialization_ms
                trace["prompt_tokenization_ms"] += tokenization_ms
                trace["prompt_tokenization_calls"] += 1
                trace["prompt_tokens_recomputed"] += prompt_tokens
            before = collect_vllm_metrics(
                args.answer_base_url.removesuffix("/v1"),
                model=args.answer_model,
                timeout=args.timeout,
            )
            result = super().generate_with_backoff(**kwargs)
            timing = collect_attributable_delta(
                args.answer_base_url.removesuffix("/v1"),
                before,
                model=args.answer_model,
                expected_requests=1,
                timeout=min(2.0, args.timeout),
            )
            if (
                bool(getattr(args, "formal", False))
                and timing["attributable"] is not True
            ):
                raise RuntimeError(
                    f"vLLM per-request timing attribution failed: {timing}"
                )
            response = result[0]
            usage = getattr(response, "usage", None)
            observed_prompt_tokens = getattr(usage, "prompt_tokens", None)
            token_count_matches = (
                observed_prompt_tokens is not None
                and int(observed_prompt_tokens) == prompt_tokens
            )
            if trace is not None:
                trace["prompt_token_count_matches_usage"] = bool(
                    trace["prompt_token_count_matches_usage"] and token_count_matches
                )
            if bool(getattr(args, "formal", False)) and not token_count_matches:
                raise RuntimeError(
                    "live /tokenize prompt count does not match answer usage: "
                    f"recomputed={prompt_tokens}, usage={observed_prompt_tokens!r}"
                )
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None)
            if trace is not None and cached is not None:
                trace["cached_tokens"] += int(cached)
                trace["cache_detail_calls"] += 1
            if trace is not None:
                trace["timing_attributable"] = bool(
                    trace["timing_attributable"] and timing["attributable"]
                )
                trace["telemetry_overhead_ms"] += float(timing["telemetry_overhead_ms"])
                for source, target in (
                    ("time_to_first_token_ms", "model_ttft_ms"),
                    ("request_prefill_time_ms", "model_prefill_ms"),
                    ("request_decode_time_ms", "model_decode_ms"),
                    ("e2e_request_latency_ms", "model_server_end_to_end_ms"),
                ):
                    trace[target] += float(timing.get(source, 0.0))
            return result

    handler = SeededOpenAIHandler(
        model_name=args.answer_model,
        temperature=0,
        registry_name=args.answer_model,
        is_fc_model=True,
    )
    ledger = CallLedger()
    embedding = OpenAIEmbeddingClient(
        args.embedding_base_url,
        args.api_key,
        args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.timeout,
        ledger=ledger,
    )
    embedder = PhasedEmbedder(embedding)
    memory = TriDBGovernedMemory.connect(
        args.dsn, dim=args.embedding_dim, embedder=embedder
    )
    memory.init_schema()
    footprint_before = _footprint(memory)
    operating_point = memory.store.conn.execute(
        "SELECT set_config('tjs.graph_scoring', %s, false),"
        " set_config('tjs.graph_work_budget', %s, false)",
        (args.graph_scoring, str(args.graph_work_budget)),
    ).fetchone()
    if tuple(operating_point) != (
        args.graph_scoring,
        str(args.graph_work_budget),
    ):
        raise RuntimeError(f"failed to lock TJS operating point: {operating_point!r}")
    token_counter = VLLMTokenCounter(
        args.answer_base_url.removesuffix("/v1"),
        args.answer_model,
        timeout=args.timeout,
    )
    lock = threading.Lock()
    snapshots: dict[tuple[str, str], Any] = {}
    source_trajectory_hashes: dict[tuple[str, str, str], str] = {}
    source_errors_by_arm: dict[tuple[str, str], int] = {}
    canonical_recorder = _CanonicalTrajectoryRecorder()
    sample_errors = 0
    predictions = 0
    started = time.time()

    def execute(
        *,
        arm: str,
        environment: str,
        source_environment: str,
        phase: str,
        backend: Any | None,
        ordinal_offset: int = 0,
        source_build_only: bool = False,
        canonical_source_results: dict[str, dict[str, Any]] | None = None,
        canonical_source_trajectories: (dict[str, list[dict[str, Any]]] | None) = None,
    ) -> tuple[int, int]:
        nonlocal predictions, sample_errors
        run_dir = (
            output / "phase1" / arm / environment
            if phase == "in_env"
            else output / "phase2" / arm / f"{source_environment}__to__{environment}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "result.jsonl"
        logger = _logger(run_dir / "errors.log")
        _write_config(
            run_dir / "run_config.json",
            {
                "model": args.answer_model,
                "seed": args.seed,
                "mode": "FC",
                "run_name": run_dir.name,
                "memory": {
                    "arm": arm,
                    "query_mode": args.query_mode,
                    "token_slot_policy": (
                        "complete_history_append_no_truncation"
                        if arm == "long_context"
                        else (
                            "fixed_neutral_replacement"
                            if _uses_fixed_token_slot(
                                arm=arm,
                                phase=phase,
                                query_mode=args.query_mode,
                            )
                            else "none"
                        )
                    ),
                    "readonly": phase == "transfer",
                    "source_environment": source_environment,
                    "target_environment": environment,
                },
            },
        )
        receipt_cursor = 0
        setting_predictions = 0
        setting_errors = 0
        for local_ordinal, source_id in enumerate(ids_by_category[environment]):
            entry = entries[source_id]
            if _uses_fixed_token_slot(
                arm=arm,
                phase=phase,
                query_mode=args.query_mode,
            ):
                entry = _token_matched_entry(
                    entry,
                    codec=token_counter,
                    token_budget=args.injection_token_budget,
                )
            episode = episodes[source_id]
            allowed_functions = _allowed_functions(load_function_docs, entry)
            history_size = 0
            if backend is not None:
                backend.set_sample_context(
                    ToolSampleContext(
                        sample_id=source_id,
                        episode_uid=episode.episode_uid,
                        ordinal=ordinal_offset + local_ordinal,
                        environment=environment,
                        question=episode.online_payload()["question"],
                        involved_classes=episode.involved_classes,
                        allowed_functions=allowed_functions,
                    )
                )
                observed_snapshot = backend.freeze()
                history_size = _snapshot_count(observed_snapshot)
                before = history_size if not backend.readonly else None
                sample_started = time.perf_counter()
                handler.begin_system_trace()
                if (
                    isinstance(backend, CrossEpToolLongContextMemory)
                    and not source_build_only
                ):
                    base_prompt_tokens = _exact_first_tool_prompt_tokens(
                        handler,
                        entry,
                        token_counter=token_counter,
                    )
                    backend.set_base_prompt_tokens(base_prompt_tokens)
                    preview = backend.preview()
                    exact_input_tokens = _exact_first_tool_prompt_tokens(
                        handler,
                        entry,
                        token_counter=token_counter,
                        injection=preview.text,
                    )
                    exact_projected = (
                        exact_input_tokens
                        + args.reserved_generation_tokens
                        + args.context_safety_tokens
                    )
                    preview = replace(
                        preview,
                        projected_input_tokens=exact_projected,
                        overflow=exact_projected > args.context_window_tokens,
                    )
                    if preview.overflow:
                        backend.begin_sample()
                        backend.record_overflow(preview)
                        usage = backend.drain_usage()
                        _write_context_overflow_result(
                            result_path,
                            source_id=source_id,
                            category=environment,
                            entry=entry,
                            usage=usage,
                        )
                        predictions += 1
                        sample_errors += 1
                        setting_predictions += 1
                        setting_errors += 1
                        for receipt in backend.receipts[receipt_cursor:]:
                            _append_jsonl(run_dir / "retrieval_receipts.jsonl", receipt)
                        receipt = backend.receipts[receipt_cursor]
                        receipt_cursor = len(backend.receipts)
                        _write_system_trace(
                            output / "system_traces" / f"{arm}.jsonl",
                            args=args,
                            arm=arm,
                            phase=phase,
                            environment=environment,
                            source_environment=source_environment,
                            source_id=source_id,
                            history_size=history_size,
                            result=_last_result(result_path, source_id),
                            receipt=receipt,
                            model_trace=handler.drain_system_trace(),
                            end_to_end_ms=(time.perf_counter() - sample_started) * 1000,
                        )
                        continue
                    backend.set_finalized_preview(preview)
            else:
                sample_started = time.perf_counter()
                handler.begin_system_trace()
            if canonical_source_results is not None:
                if (
                    not source_build_only
                    or backend is None
                    or phase != "in_env"
                    or canonical_source_trajectories is None
                ):
                    raise RuntimeError(
                        "canonical source replay is only valid for memory-bank construction"
                    )
                result = _replay_canonical_source_result(
                    canonical_source_results[source_id],
                    canonical_trajectory=canonical_source_trajectories[source_id],
                    backend=backend,
                )
                _append_jsonl(result_path, result)
                error = result.get("error")
            else:
                if source_build_only and backend is not None:
                    process_backend = _SourceConstructionOnlyMemory(backend)
                elif phase == "in_env" and arm == "memory_off":
                    canonical_recorder.set_sample(source_id)
                    process_backend = canonical_recorder
                else:
                    process_backend = backend
                error = process_one_sample(
                    handler,
                    entry,
                    environment,
                    result_path,
                    lock,
                    logger,
                    memory=process_backend,
                )
            predictions += 1
            sample_errors += int(error is not None)
            setting_predictions += 1
            setting_errors += int(error is not None)
            sample_receipt: dict[str, Any] | None = None
            result = _last_result(result_path, source_id)
            if phase == "in_env" and error is None:
                trajectory = (
                    canonical_source_trajectories[source_id]
                    if canonical_source_trajectories is not None
                    else canonical_recorder.trajectories.get(source_id)
                    if arm == "memory_off"
                    else result.get("full_message_history")
                )
                if not isinstance(trajectory, list) or not trajectory:
                    raise RuntimeError(
                        f"successful source sample {source_id} has no raw trajectory"
                    )
                source_trajectory_hashes[(arm, environment, source_id)] = sha256_text(
                    json.dumps(
                        trajectory,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                )
            if backend is not None:
                expected_receipts = 0 if source_build_only else 1
                if len(backend.receipts) != receipt_cursor + expected_receipts:
                    raise RuntimeError(
                        f"sample {source_id} produced {len(backend.receipts) - receipt_cursor} "
                        "retrieval receipts; the official runner may have swallowed a "
                        "memory-utilize failure"
                    )
                if phase == "in_env" and error is None:
                    after = _snapshot_count(backend.freeze())
                    if before is None or after != before + 1:
                        raise RuntimeError(
                            f"successful sample {source_id} did not atomically admit one experience"
                        )
                    memory_usage = result.get("memory_usage") or {}
                    _append_jsonl(
                        output / "updates" / f"{arm}.jsonl",
                        {
                            "schema_version": "evomembench_tool_update_v0.1.0",
                            "source_id": source_id,
                            "episode_uid": episode.episode_uid,
                            "source_environment": environment,
                            "ordinal": local_ordinal,
                            "canonical_trajectory_sha256": source_trajectory_hashes[
                                (arm, environment, source_id)
                            ],
                            "snapshot_count_before": before,
                            "snapshot_count_after": after,
                            "construction_seconds": float(
                                memory_usage.get("latency_s", 0.0)
                            ),
                            "construction_input_tokens": int(
                                memory_usage.get("input_tokens", 0)
                            ),
                            "construction_output_tokens": int(
                                memory_usage.get("output_tokens", 0)
                            ),
                            "embedding_tokens": int(
                                memory_usage.get("embedding_tokens", 0)
                            ),
                            "llm_calls": int(memory_usage.get("n_llm_calls", 0)),
                            "embedding_calls": int(
                                memory_usage.get("n_embed_calls", 0)
                            ),
                            "updates": int(memory_usage.get("n_update", 0)),
                            "status": "committed",
                        },
                    )
                if backend.readonly:
                    backend.assert_snapshot_unchanged()
                for receipt in backend.receipts[receipt_cursor:]:
                    _append_jsonl(run_dir / "retrieval_receipts.jsonl", receipt)
                    sample_receipt = receipt
                receipt_cursor = len(backend.receipts)
            _write_system_trace(
                output / "system_traces" / f"{arm}.jsonl",
                args=args,
                arm=arm,
                phase=phase,
                environment=environment,
                source_environment=source_environment,
                source_id=source_id,
                history_size=history_size,
                result=(
                    {
                        **result,
                        "latency": [],
                        "input_token_count": [],
                        "output_token_count": [],
                    }
                    if canonical_source_results is not None
                    else result
                ),
                receipt=sample_receipt,
                model_trace=handler.drain_system_trace(),
                end_to_end_ms=(time.perf_counter() - sample_started) * 1000,
                canonical_source_replay=canonical_source_results is not None,
            )
        return setting_predictions, setting_errors

    phase1_schedule = {
        environment: ["memory_off"]
        + [
            arm
            for arm in balanced_arm_order(args.arms, block_index=index, seed=args.seed)
            if arm != "memory_off"
        ]
        for index, environment in enumerate(ids_by_category)
    }
    transfer_arms = list(args.arms)
    transfer_pairs = ordered_transfer_pairs()
    phase2_schedule = {
        f"{source}__to__{target}": list(
            balanced_arm_order(transfer_arms, block_index=index, seed=args.seed)
        )
        for index, (source, target) in enumerate(transfer_pairs)
    }
    arm_order_schedule = {
        "schema_version": "evomembench_balanced_arm_schedule_v0.1.0",
        "seed": args.seed,
        "phase1_unit": "source_environment",
        "phase1_policy": (
            "single_memory_off_generation_then_canonical_arm_construction_replay"
        ),
        "phase1": phase1_schedule,
        "phase2_unit": "directed_source_target_pair",
        "phase2": phase2_schedule,
    }
    (output / "arm_order_schedule.json").write_text(
        json.dumps(arm_order_schedule, ensure_ascii=False, indent=2) + "\n"
    )

    try:
        for environment in ids_by_category:
            _, source_errors = execute(
                arm="memory_off",
                environment=environment,
                source_environment=environment,
                phase="in_env",
                backend=None,
            )
            source_errors_by_arm[("memory_off", environment)] = source_errors
            canonical_results = _results_by_id(
                output / "phase1" / "memory_off" / environment / "result.jsonl"
            )
            if set(canonical_results) != set(ids_by_category[environment]):
                raise RuntimeError(
                    f"canonical source result IDs differ for {environment}"
                )
            canonical_trajectories = {
                source_id: canonical_recorder.trajectories[source_id]
                for source_id in ids_by_category[environment]
            }
            canonical_audit = (
                output / "canonical_source_observed" / f"{environment}.jsonl"
            )
            for source_id in ids_by_category[environment]:
                _append_jsonl(
                    canonical_audit,
                    {
                        "source_id": source_id,
                        "trajectory": canonical_trajectories[source_id],
                    },
                )
            for arm in phase1_schedule[environment]:
                if arm == "memory_off":
                    continue
                if arm == "long_context":
                    backend = CrossEpToolLongContextMemory(
                        source_environment=environment,
                        count_tokens=token_counter,
                        context_window_tokens=args.context_window_tokens,
                        reserved_generation_tokens=args.reserved_generation_tokens,
                        safety_tokens=args.context_safety_tokens,
                    )
                else:
                    scope_id = f"{args.run_id}:{arm}:tool:{environment}"
                    backend = CrossEpToolGemMemory(
                        memory=memory,
                        scope_id=scope_id,
                        source_environment=environment,
                        count_tokens=token_counter,
                        query_mode=args.query_mode,
                        arm=arm,
                        top_k=args.top_k,
                        hops=args.hops,
                        m_seeds=args.m_seeds,
                        term_cond=args.term_cond,
                        graph_scoring=args.graph_scoring,
                        graph_work_budget=args.graph_work_budget,
                        query_embedder=embedder,
                        token_budget=args.injection_token_budget,
                        token_matched_slot=args.query_mode == "gem_task_seed",
                    )
                _, source_errors = execute(
                    arm=arm,
                    environment=environment,
                    source_environment=environment,
                    phase="in_env",
                    backend=backend,
                    source_build_only=True,
                    canonical_source_results=canonical_results,
                    canonical_source_trajectories=canonical_trajectories,
                )
                source_errors_by_arm[(arm, environment)] = source_errors
                snapshot = backend.freeze()
                expected = len(ids_by_category[environment])
                if _snapshot_count(snapshot) + source_errors != expected:
                    raise RuntimeError(
                        f"source bank {environment} has unexplained missing experiences"
                    )
                snapshots[(arm, environment)] = snapshot
                snapshot_path = output / "snapshots" / arm / f"{environment}.json"
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(snapshot, LongContextSnapshot):
                    snapshot.write(snapshot_path)
                else:
                    CrossEpToolGemMemory.write_snapshot(snapshot_path, snapshot)
                    if arm == "gem_fused":
                        export_path = (
                            output
                            / "system_snapshots"
                            / "full_gem"
                            / f"{environment}.json"
                        )
                        export_path.parent.mkdir(parents=True, exist_ok=True)
                        export_gem_snapshot(memory, scope_id=snapshot.scope_id).write(
                            export_path
                        )

            for source_id in ids_by_category[environment]:
                observed = {
                    arm: source_trajectory_hashes.get((arm, environment, source_id))
                    for arm in args.arms
                }
                if (
                    any(value is None for value in observed.values())
                    or len(set(observed.values())) != 1
                ):
                    raise RuntimeError(
                        "canonical source trajectory parity failed for "
                        f"{environment}:{source_id}: {observed}"
                    )

        canonical_source_manifest = {
            "schema_version": "evomembench_tool_canonical_source_v0.1.0",
            "policy": (
                "single_memory_blind_generation_then_arm_specific_replay_construction"
            ),
            "arms_verified": list(args.arms),
            "environments": {
                environment: {
                    source_id: source_trajectory_hashes[
                        (args.arms[0], environment, source_id)
                    ]
                    for source_id in ids_by_category[environment]
                }
                for environment in ids_by_category
            },
        }
        canonical_source_manifest["digest"] = sha256_text(
            json.dumps(
                canonical_source_manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        (output / "canonical_source_trajectories.json").write_text(
            json.dumps(canonical_source_manifest, ensure_ascii=False, indent=2) + "\n"
        )

        for source_environment, target_environment in transfer_pairs:
            cell = f"{source_environment}__to__{target_environment}"
            for arm in phase2_schedule[cell]:
                if arm == "memory_off":
                    execute(
                        arm=arm,
                        environment=target_environment,
                        source_environment=source_environment,
                        phase="transfer",
                        backend=None,
                    )
                    continue
                snapshot = snapshots[(arm, source_environment)]
                if arm == "long_context":
                    backend = CrossEpToolLongContextMemory(
                        source_environment=source_environment,
                        count_tokens=token_counter,
                        context_window_tokens=args.context_window_tokens,
                        reserved_generation_tokens=args.reserved_generation_tokens,
                        safety_tokens=args.context_safety_tokens,
                        readonly=True,
                        snapshot=snapshot,
                    )
                else:
                    backend = CrossEpToolGemMemory(
                        memory=memory,
                        scope_id=snapshot.scope_id,
                        source_environment=source_environment,
                        count_tokens=token_counter,
                        query_mode=args.query_mode,
                        arm=arm,
                        top_k=args.top_k,
                        hops=args.hops,
                        m_seeds=args.m_seeds,
                        term_cond=args.term_cond,
                        graph_scoring=args.graph_scoring,
                        graph_work_budget=args.graph_work_budget,
                        query_embedder=embedder,
                        token_budget=args.injection_token_budget,
                        readonly=True,
                        snapshot=snapshot,
                        token_matched_slot=args.query_mode == "gem_task_seed",
                    )
                execute(
                    arm=arm,
                    environment=target_environment,
                    source_environment=source_environment,
                    phase="transfer",
                    backend=backend,
                    ordinal_offset=snapshot.max_ordinal + 1,
                )
                backend.assert_snapshot_unchanged()
        footprint_after = _footprint(memory)
    finally:
        memory.close()

    return {
        "schema_version": "evomembench_gem_tool_run_v0.1.0",
        "status": "complete",
        "formal": bool(getattr(args, "formal", False)),
        "run_id": args.run_id,
        "source_revision": corpus.source_revision,
        "manifest_sha256": manifest_hash,
        "verified_assets": verified,
        "arms": list(args.arms),
        "arm_order_policy": (
            "phase1_canonical_replay_then_phase2_seeded_latin_rotation"
        ),
        "arm_order_schedule": arm_order_schedule,
        "query_mode": args.query_mode,
        "generation_seed": args.seed,
        "answer_max_tokens": args.reserved_generation_tokens,
        "token_slot_policy": (
            {
                arm: (
                    "complete_history_append_no_truncation"
                    if arm == "long_context"
                    else (
                        "fixed_neutral_replacement"
                        if _uses_fixed_token_slot(
                            arm=arm,
                            phase="transfer",
                            query_mode=args.query_mode,
                        )
                        else "none"
                    )
                )
                for arm in args.arms
            }
        ),
        "token_slot_tokens": args.injection_token_budget,
        "operating_point": {
            "top_k": args.top_k,
            "m_seeds": args.m_seeds,
            "hops": args.hops,
            "term_cond": args.term_cond,
            "graph_scoring": args.graph_scoring,
            "graph_work_budget": args.graph_work_budget,
            "injection_token_budget": args.injection_token_budget,
        },
        "episodes_per_environment": args.episodes_per_environment,
        "source_building_evaluations_per_memory_arm": (
            4 * args.episodes_per_environment
        ),
        "source_building_policy": (
            "single_memory_blind_generation_then_arm_specific_replay_construction"
        ),
        "canonical_source_trajectory_digest": canonical_source_manifest["digest"],
        "source_building_errors": {
            f"{arm}:{environment}": errors
            for (arm, environment), errors in source_errors_by_arm.items()
        },
        "target_evaluations_per_arm": (
            len(transfer_pairs) * args.episodes_per_environment
        ),
        "directed_transfer_pairs": len(transfer_pairs),
        "predictions": predictions,
        "sample_errors": sample_errors,
        "snapshots": {
            f"{arm}:{environment}": asdict(snapshot)
            for (arm, environment), snapshot in snapshots.items()
        },
        "embedding_calls": ledger.summary(),
        "database_footprint_before": footprint_before,
        "database_footprint_after": footprint_after,
        "elapsed_seconds": time.time() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--manifest", default=str(Path(__file__).with_name("protocol_manifest.json"))
    )
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--answer-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--answer-model", default="qwen3.8")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--query-mode",
        choices=["upstream_parity", "gem_task_seed"],
        default="gem_task_seed",
    )
    parser.add_argument("--episodes-per-environment", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--m-seeds", type=int, default=4)
    parser.add_argument("--term-cond", type=int, default=32)
    parser.add_argument(
        "--graph-scoring", choices=("membership",), default="membership"
    )
    parser.add_argument("--graph-work-budget", type=int, default=65536)
    parser.add_argument("--injection-token-budget", type=int, default=2048)
    parser.add_argument("--context-window-tokens", type=int, default=262144)
    parser.add_argument("--reserved-generation-tokens", type=int, default=4096)
    parser.add_argument("--context-safety-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--authorization", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output_preexisted = output.exists()
    try:
        receipt = run(args)
    except BaseException as exc:
        if not output_preexisted:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_receipt.json").write_text(
                json.dumps(
                    {
                        "schema_version": "evomembench_gem_tool_run_v0.1.0",
                        "status": "failed",
                        "formal": bool(args.formal),
                        "run_id": args.run_id,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    indent=2,
                )
                + "\n"
            )
        raise
    (output / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
