"""Run LoCoMo retrieval, answer generation, judging, and metric aggregation.

The pipeline uses TriDB's LoCoMo adapter for retrieval and an OpenAI-compatible
chat endpoint for both answer generation and answer judging. Results are
checkpointed in the augmented LoCoMo JSON so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from bench.agent_memory.locomo_adapter import main as locomo_retrieval_main

CATEGORY_NAMES = {
    1: "multi",
    2: "temporal",
    3: "open",
    4: "single",
    5: "adversarial",
}
PAPER_CATEGORIES = (1, 2, 3, 4)
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_PREDICTION_KEY = "tridb_prediction"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class OpenAICompatibleClient:
    """Small dependency-light client for a vLLM/OpenAI chat endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
            self._thread_local.session = session
        return session

    def discover_model(self) -> str:
        response = self._session().get(
            f"{self.base_url}/models",
            timeout=self.timeout,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        model_ids = [str(model["id"]) for model in models if model.get("id")]
        if len(model_ids) != 1:
            raise RuntimeError(
                "model discovery requires exactly one served model; "
                f"found {model_ids!r}"
            )
        return model_ids[0]

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> tuple[str, dict[str, Any], float]:
        started = time.perf_counter()
        response = self._session().post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            },
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError(f"chat response has no choices: {payload!r}")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"chat response has no text content: {payload!r}")
        usage = payload.get("usage") or {}
        return content.strip(), dict(usage), elapsed


def build_judge_prompt(question: str, gold_answer: Any, prediction: str) -> str:
    """Build the generous binary correctness prompt used by Mandol-style runs."""
    return f"""You are an expert grader that determines whether a generated answer
matches a gold-standard answer.

Return CORRECT when the generated answer contains the same key information or
has equivalent meaning. Be generous about wording differences. For temporal
questions, accept equivalent dates or time periods. Return WRONG when required
facts are missing or contradicted.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {prediction}

Return only one JSON object, with no explanation:
{{"label": "CORRECT"}} or {{"label": "WRONG"}}
"""


def parse_judge_label(response_text: str) -> str:
    text = response_text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        label = str(parsed.get("label", "")).upper().strip()
        if label in {"CORRECT", "WRONG"}:
            return label

    match = re.search(r"\b(CORRECT|WRONG)\b", text.upper())
    if match is not None:
        return match.group(1)
    raise ValueError(f"judge did not return CORRECT or WRONG: {response_text!r}")


def _normalized_tokens(value: Any) -> list[str]:
    text = str(value).lower().translate(str.maketrans("", "", string.punctuation))
    return [token for token in text.split() if token not in {"a", "an", "the"}]


def lexical_f1(prediction: str, gold_answer: Any) -> float:
    """Return a dependency-free diagnostic token F1 score."""
    gold_values = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    prediction_tokens = _normalized_tokens(prediction)
    if not prediction_tokens:
        return 0.0

    scores = []
    for gold in gold_values:
        gold_tokens = _normalized_tokens(gold)
        if not gold_tokens:
            scores.append(float(not prediction_tokens))
            continue
        common = Counter(prediction_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(gold_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    if not collected:
        return None
    return sum(collected) / len(collected)


def _usage_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0}
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    summary: dict[str, Any] = {"count": len(items)}
    for key in keys:
        values = [float(item.get(key, 0)) for item in items]
        summary[f"total_{key}"] = int(sum(values))
        summary[f"mean_{key}"] = sum(values) / len(values)
    return summary


def retrieval_metrics(
    qas: Sequence[dict[str, Any]],
    *,
    prediction_key: str,
) -> dict[str, Any]:
    eligible = [qa for qa in qas if qa.get("evidence")]
    metrics: dict[str, Any] = {"questions_with_evidence": len(eligible)}
    if not eligible:
        return metrics

    available_k = max(
        (len(qa.get(f"{prediction_key}_context", [])) for qa in eligible),
        default=0,
    )
    for k in (1, 3, 5, 10, 20, 30, 50):
        if k > available_k:
            continue
        evidence_recall = []
        hit_any = []
        hit_all = []
        reciprocal_rank = []
        ndcg = []
        for qa in eligible:
            gold = {str(item) for item in qa["evidence"]}
            ranked = [str(item) for item in qa.get(f"{prediction_key}_context", [])[:k]]
            found = gold.intersection(ranked)
            relevances = [int(item in gold) for item in ranked]
            ideal = _dcg([1] * min(k, len(gold)))
            evidence_recall.append(len(found) / len(gold))
            hit_any.append(float(bool(found)))
            hit_all.append(float(len(found) == len(gold)))
            reciprocal_rank.append(
                next(
                    (
                        1.0 / rank
                        for rank, item in enumerate(ranked, start=1)
                        if item in gold
                    ),
                    0.0,
                )
            )
            ndcg.append(0.0 if ideal == 0.0 else _dcg(relevances) / ideal)
        metrics[f"@{k}"] = {
            "evidence_recall": _mean(evidence_recall),
            "hit_any": _mean(hit_any),
            "hit_all": _mean(hit_all),
            "mrr": _mean(reciprocal_rank),
            "ndcg": _mean(ndcg),
        }
    return metrics


def build_metrics_report(
    samples: Sequence[dict[str, Any]],
    *,
    prediction_key: str,
    categories: Sequence[int],
    answer_model: str,
    judge_model: str,
    started_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    qas = [
        qa
        for sample in samples
        for qa in sample.get("qa", [])
        if qa.get("category") in categories
    ]
    judged = [
        qa
        for qa in qas
        if isinstance(qa.get(f"{prediction_key}_judge"), dict)
        and "correct" in qa[f"{prediction_key}_judge"]
    ]
    generated = [qa for qa in qas if prediction_key in qa]
    errors = [qa for qa in qas if f"{prediction_key}_error" in qa]

    by_category: dict[str, Any] = {}
    for category in categories:
        category_qas = [qa for qa in judged if qa.get("category") == category]
        accuracy = _mean(
            float(qa[f"{prediction_key}_judge"]["correct"]) for qa in category_qas
        )
        f1 = _mean(
            lexical_f1(qa[prediction_key], qa.get("answer", "")) for qa in category_qas
        )
        by_category[CATEGORY_NAMES.get(category, str(category))] = {
            "category": category,
            "count": len(category_qas),
            "llm_judge_accuracy": accuracy,
            "diagnostic_lexical_f1": f1,
        }

    overall_accuracy = _mean(
        float(qa[f"{prediction_key}_judge"]["correct"]) for qa in judged
    )
    overall_f1 = _mean(
        lexical_f1(qa[prediction_key], qa.get("answer", "")) for qa in judged
    )
    answer_usage = [
        qa.get(f"{prediction_key}_usage", {})
        for qa in generated
        if isinstance(qa.get(f"{prediction_key}_usage"), dict)
    ]
    judge_usage = [
        qa[f"{prediction_key}_judge"].get("usage", {})
        for qa in judged
        if isinstance(qa[f"{prediction_key}_judge"].get("usage"), dict)
    ]

    paper_order = ("single", "multi", "temporal", "open")
    paper_row = {
        "backbone": answer_model,
        "judge": judge_model,
        "avg_prompt_tokens": _mean(
            float(item.get("prompt_tokens", 0)) for item in answer_usage
        ),
    }
    for name in paper_order:
        value = by_category.get(name, {}).get("llm_judge_accuracy")
        paper_row[name] = None if value is None else value * 100
    paper_row["overall"] = None if overall_accuracy is None else overall_accuracy * 100

    return {
        "schema_version": "locomo_tridb_pipeline_v0.1.0",
        "status": "completed" if len(judged) == len(qas) and not errors else "partial",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "models": {
            "answer": answer_model,
            "judge": judge_model,
            "same_model_for_answer_and_judge": answer_model == judge_model,
        },
        "scope": {
            "categories": list(categories),
            "category_names": [
                CATEGORY_NAMES.get(category, str(category)) for category in categories
            ],
            "eligible_questions": len(qas),
            "generated_questions": len(generated),
            "judged_questions": len(judged),
            "errors": len(errors),
        },
        "qa_metrics": {
            "llm_judge_accuracy": overall_accuracy,
            "diagnostic_lexical_f1": overall_f1,
            "by_category": by_category,
        },
        "retrieval_metrics": retrieval_metrics(
            qas,
            prediction_key=prediction_key,
        ),
        "token_usage": {
            "answer_generation": _usage_summary(answer_usage),
            "judging": _usage_summary(judge_usage),
        },
        "latency": {
            "mean_answer_seconds": _mean(
                float(qa.get(f"{prediction_key}_latency_seconds", 0))
                for qa in generated
            ),
            "mean_judge_seconds": _mean(
                float(qa[f"{prediction_key}_judge"].get("latency_seconds", 0))
                for qa in judged
            ),
        },
        "paper_table_percent": paper_row,
        "comparability": {
            "paper_metric_shape": True,
            "paper_protocol_match": False,
            "reason": (
                "Mandol's paper used GPT-4.1-mini/GPT-4o-mini answer backbones "
                "and a GPT-4o-mini judge. This report uses the models recorded "
                "above and should be reported as a separate backbone row."
            ),
        },
    }


def _iter_selected_qas(
    samples: Sequence[dict[str, Any]],
    categories: Sequence[int],
) -> Iterable[tuple[int, int, dict[str, Any]]]:
    allowed = set(categories)
    for sample_index, sample in enumerate(samples):
        for qa_index, qa in enumerate(sample.get("qa", [])):
            if qa.get("category") in allowed:
                yield sample_index, qa_index, qa


def _process_qa(
    qa: dict[str, Any],
    *,
    answer_client: OpenAICompatibleClient,
    answer_model: str,
    judge_client: OpenAICompatibleClient,
    judge_model: str,
    prediction_key: str,
    answer_max_tokens: int,
    judge_max_tokens: int,
    seed: int,
    force_generation: bool,
    force_judge: bool,
) -> dict[str, bool]:
    generated = False
    judged = False
    error_key = f"{prediction_key}_error"
    qa.pop(error_key, None)

    try:
        if force_generation or prediction_key not in qa:
            prompt_key = f"{prediction_key}_prompt"
            if prompt_key not in qa:
                raise KeyError(f"missing retrieval prompt {prompt_key!r}")
            prediction, usage, latency = answer_client.chat(
                model=answer_model,
                prompt=str(qa[prompt_key]),
                max_tokens=answer_max_tokens,
                temperature=0.0,
                seed=seed,
            )
            qa[prediction_key] = prediction
            qa[f"{prediction_key}_usage"] = usage
            qa[f"{prediction_key}_latency_seconds"] = latency
            qa.pop(f"{prediction_key}_judge", None)
            generated = True

        judge_key = f"{prediction_key}_judge"
        if force_judge or judge_key not in qa:
            judge_text, usage, latency = judge_client.chat(
                model=judge_model,
                prompt=build_judge_prompt(
                    str(qa.get("question", "")),
                    qa.get("answer", ""),
                    str(qa[prediction_key]),
                ),
                max_tokens=judge_max_tokens,
                temperature=0.0,
                seed=seed,
            )
            label = parse_judge_label(judge_text)
            qa[judge_key] = {
                "label": label,
                "correct": label == "CORRECT",
                "raw": judge_text,
                "model": judge_model,
                "usage": usage,
                "latency_seconds": latency,
            }
            judged = True
    except Exception as exc:
        qa[error_key] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        raise

    return {"generated": generated, "judged": judged}


def _parse_categories(value: str) -> tuple[int, ...]:
    categories = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not categories:
        raise argparse.ArgumentTypeError("at least one category is required")
    invalid = [category for category in categories if category not in CATEGORY_NAMES]
    if invalid:
        raise argparse.ArgumentTypeError(f"unknown LoCoMo categories: {invalid}")
    return categories


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--retrieval-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prediction-key", default=DEFAULT_PREDICTION_KEY)
    parser.add_argument("--dsn")
    parser.add_argument("--table", default="locomo_units")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force-retrieval", action="store_true")
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--force-judge", action="store_true")
    parser.add_argument(
        "--categories",
        type=_parse_categories,
        default=PAPER_CATEGORIES,
        help="comma-separated LoCoMo categories; default: 1,2,3,4",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-model")
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--judge-max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--limit-questions", type=int)
    return parser.parse_args(argv)


def _prepare_retrieval(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.force_retrieval or not args.retrieval_output.exists():
        retrieval_args = [
            "--input",
            str(args.input),
            "--output",
            str(args.retrieval_output),
            "--top-k",
            str(args.top_k),
            "--prediction-key",
            args.prediction_key,
            "--table",
            args.table,
            "--model",
            args.embedding_model,
            "--dim",
            str(args.dim),
            "--batch-size",
            str(args.batch_size),
        ]
        if args.dsn:
            retrieval_args.extend(["--dsn", args.dsn])
        result = locomo_retrieval_main(retrieval_args)
        if result != 0:
            raise RuntimeError(f"LoCoMo retrieval failed with exit code {result}")

    retrieval = json.loads(args.retrieval_output.read_text())
    if not isinstance(retrieval, list):
        raise ValueError("LoCoMo retrieval output must be a JSON array")
    top_ks = {
        sample.get("tridb_backend", {}).get("top_k")
        for sample in retrieval
        if sample.get("tridb_backend")
    }
    if top_ks and top_ks != {args.top_k}:
        raise ValueError(
            f"retrieval output has top-k {sorted(top_ks)!r}, "
            f"but --top-k is {args.top_k}"
        )
    return retrieval


def _prepare_output(
    args: argparse.Namespace,
    retrieval: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not args.output.exists():
        output = list(retrieval)
        _atomic_write_json(args.output, output)
        return output

    output = json.loads(args.output.read_text())
    if not isinstance(output, list):
        raise ValueError("LoCoMo pipeline output must be a JSON array")
    retrieval_ids = [sample.get("sample_id") for sample in retrieval]
    output_ids = [sample.get("sample_id") for sample in output]
    if retrieval_ids != output_ids:
        raise ValueError(
            "pipeline output does not match retrieval sample ids: "
            f"{output_ids!r} != {retrieval_ids!r}"
        )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    retrieval = _prepare_retrieval(args)
    samples = _prepare_output(args, retrieval)

    answer_client = OpenAICompatibleClient(
        args.base_url,
        args.api_key,
        timeout=args.request_timeout,
    )
    answer_model = args.model or answer_client.discover_model()
    judge_client = OpenAICompatibleClient(
        args.judge_base_url or args.base_url,
        args.judge_api_key or args.api_key,
        timeout=args.request_timeout,
    )
    judge_model = args.judge_model or answer_model

    selected = list(_iter_selected_qas(samples, args.categories))
    if args.limit_questions is not None:
        selected = selected[: args.limit_questions]
    total = len(selected)
    print(
        f"[locomo-pipeline] questions={total} workers={args.workers} "
        f"answer_model={answer_model} judge_model={judge_model}",
        file=sys.stderr,
        flush=True,
    )

    completed = 0
    generated = 0
    judged = 0
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, total, args.checkpoint_every):
            batch = selected[batch_start : batch_start + args.checkpoint_every]
            future_to_position = {
                executor.submit(
                    _process_qa,
                    qa,
                    answer_client=answer_client,
                    answer_model=answer_model,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    prediction_key=args.prediction_key,
                    answer_max_tokens=args.answer_max_tokens,
                    judge_max_tokens=args.judge_max_tokens,
                    seed=args.seed,
                    force_generation=args.force_generation,
                    force_judge=args.force_judge,
                ): (sample_index, qa_index)
                for sample_index, qa_index, qa in batch
            }
            for future in as_completed(future_to_position):
                sample_index, qa_index = future_to_position[future]
                try:
                    result = future.result()
                    generated += int(result["generated"])
                    judged += int(result["judged"])
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    print(
                        f"[locomo-pipeline] ERROR sample={sample_index + 1} "
                        f"qa={qa_index + 1}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                completed += 1
                print(
                    f"[locomo-pipeline] {completed}/{total} "
                    f"generated={generated} judged={judged}",
                    file=sys.stderr,
                    flush=True,
                )
            _atomic_write_json(args.output, samples)
            if first_error is not None:
                break

    elapsed = time.perf_counter() - started
    report = build_metrics_report(
        samples,
        prediction_key=args.prediction_key,
        categories=args.categories,
        answer_model=answer_model,
        judge_model=judge_model,
        started_at=started_at,
        elapsed_seconds=elapsed,
    )
    _atomic_write_json(args.metrics_output, report)
    if first_error is not None:
        raise RuntimeError(
            "pipeline stopped after checkpointing the first failed batch"
        ) from first_error

    print(
        json.dumps(report["paper_table_percent"], ensure_ascii=False, indent=2),
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
