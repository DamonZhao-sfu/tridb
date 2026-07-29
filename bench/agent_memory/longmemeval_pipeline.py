"""Run the MemoryAgentBench LongMemEval workload end to end on TriDB.

The paper-compatible workload shape is five independent long histories with
60 questions per history.  Each history is chunked and inserted into TriDB
once, then reused for all of its questions.  Answer generation is streamed so
effective TTFT includes query embedding, retrieval, prompt assembly, vLLM
queueing, and prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from bench.agent_memory.backend import DEFAULT_DSN, MemoryUnit, TriDBMemoryBackend

DEFAULT_ANSWER_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_ANSWER_MODEL = "Qwen/Qwen3-32B"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_SOURCE = "longmemeval_s*"
SYSTEM_MESSAGE = (
    "You are a helpful assistant that can read the context and memorize it for "
    "future retrieval."
)
QUERY_TEMPLATE = (
    "The history chats are between you and a user. Based on the relevant chat "
    "history, answer the question as concisely as you can, using a single phrase "
    "if possible.\n\n {question} \n\n Answer:"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


@dataclass(frozen=True)
class LongMemEvalQuestion:
    question_id: str
    question: str
    answer: Any
    question_type: str
    qa_pair_id: str | None


@dataclass(frozen=True)
class LongMemEvalWorkload:
    scope_id: str
    source: str
    context: str
    questions: tuple[LongMemEvalQuestion, ...]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def load_workloads(
    path: Path,
    *,
    source: str = DEFAULT_SOURCE,
    strict_shape: bool = True,
) -> list[LongMemEvalWorkload]:
    """Load the inject-once/query-many MemoryAgentBench JSON export."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError("LongMemEval input must be a JSON array or {'data': [...]}")

    selected = []
    seen_question_ids: set[str] = set()
    for row_index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"input row {row_index} is not an object")
        metadata = row.get("metadata") or {}
        row_source = str(metadata.get("source", row.get("source", "")))
        if row_source != source:
            continue
        context = row.get("context")
        if not isinstance(context, str) or not context.strip():
            raise ValueError(f"row {row_index} has no context text")

        questions = _as_list(row.get("questions"))
        answers = _as_list(row.get("answers"))
        question_ids = _as_list(metadata.get("question_ids"))
        question_types = _as_list(metadata.get("question_types"))
        qa_pair_ids = _as_list(metadata.get("qa_pair_ids"))
        lengths = {
            "questions": len(questions),
            "answers": len(answers),
            "question_ids": len(question_ids),
            "question_types": len(question_types),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"row {row_index} has mismatched QA lengths: {lengths}")
        if qa_pair_ids and len(qa_pair_ids) != len(questions):
            raise ValueError(
                f"row {row_index} has {len(qa_pair_ids)} qa_pair_ids for "
                f"{len(questions)} questions"
            )

        normalized_questions = []
        for question_index, (question, answer, question_id, question_type) in enumerate(
            zip(
                questions,
                answers,
                question_ids,
                question_types,
                strict=True,
            )
        ):
            question_id = str(question_id)
            if question_id in seen_question_ids:
                raise ValueError(f"duplicate question_id {question_id!r}")
            seen_question_ids.add(question_id)
            normalized_questions.append(
                LongMemEvalQuestion(
                    question_id=question_id,
                    question=str(question),
                    answer=answer,
                    question_type=str(question_type),
                    qa_pair_id=(
                        None if not qa_pair_ids else str(qa_pair_ids[question_index])
                    ),
                )
            )
        selected.append(
            LongMemEvalWorkload(
                scope_id=f"{source.replace('*', 'star')}_{len(selected):02d}",
                source=row_source,
                context=context,
                questions=tuple(normalized_questions),
            )
        )

    if not selected:
        raise ValueError(f"no rows found with metadata.source={source!r}")
    if strict_shape:
        counts = [len(workload.questions) for workload in selected]
        if len(selected) != 5 or counts != [60] * 5:
            raise ValueError(
                "paper workload requires five histories with 60 questions each; "
                f"found {len(selected)} histories with counts {counts}"
            )
    return selected


class TiktokenSentenceChunker:
    """MemoryAgentBench-compatible 4,096-token sentence packing."""

    _JOIN_MARGIN_TOKENS = 32

    def __init__(
        self,
        *,
        chunk_size: int = 4096,
        tokenizer_model: str = "gpt-4o-mini",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "tiktoken is required; install requirements-agent-memory.txt"
            ) from exc
        try:
            import nltk
        except ImportError as exc:
            raise RuntimeError(
                "nltk is required; install requirements-agent-memory.txt"
            ) from exc
        try:
            self.encoding = tiktoken.encoding_for_model(tokenizer_model)
        except KeyError:
            self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.nltk = nltk
        self.chunk_size = chunk_size
        self.tokenizer_model = tokenizer_model

    def _sentences(self, text: str) -> list[str]:
        try:
            return self.nltk.sent_tokenize(text)
        except LookupError as exc:
            raise RuntimeError(
                "NLTK punkt data is missing; run: "
                "python -m nltk.downloader punkt punkt_tab"
            ) from exc

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0
        for sentence in self._sentences(text):
            tokens = self.encoding.encode(
                sentence,
                allowed_special={"<|endoftext|>"},
            )
            if len(tokens) > self.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                    current_sentences = []
                    current_tokens = 0
                for start in range(0, len(tokens), self.chunk_size):
                    chunks.append(
                        self.encoding.decode(tokens[start : start + self.chunk_size])
                    )
                continue
            packing_limit = max(1, self.chunk_size - self._JOIN_MARGIN_TOKENS)
            if current_sentences and current_tokens + len(tokens) > packing_limit:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            current_sentences.append(sentence)
            current_tokens += len(tokens)
        if current_sentences:
            chunks.append(" ".join(current_sentences))
        bounded: list[str] = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            tokens = self.encoding.encode(
                chunk,
                allowed_special={"<|endoftext|>"},
            )
            for start in range(0, len(tokens), self.chunk_size):
                bounded.append(
                    self.encoding.decode(tokens[start : start + self.chunk_size])
                )
        return bounded

    def count(self, text: str) -> int:
        return len(
            self.encoding.encode(
                text,
                allowed_special={"<|endoftext|>"},
            )
        )


class CallLedger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        *,
        kind: str,
        phase: str,
        target: str,
        items: int,
        elapsed_seconds: float,
        status: str = "ok",
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.records.append(
            {
                "sequence": len(self.records) + 1,
                "kind": kind,
                "phase": phase,
                "target": target,
                "items": items,
                "elapsed_seconds": elapsed_seconds,
                "status": status,
                "detail": detail or {},
            }
        )

    def summary(self) -> dict[str, Any]:
        by_kind = Counter(record["kind"] for record in self.records)
        item_counts = Counter()
        failed = 0
        for record in self.records:
            item_counts[record["kind"]] += int(record["items"])
            failed += int(record["status"] != "ok")
        construction_embedding = by_kind["construction_embedding"]
        query_embedding = by_kind["query_embedding"]
        answers = by_kind["answer_generation"]
        return {
            "by_kind": dict(sorted(by_kind.items())),
            "items_by_kind": dict(sorted(item_counts.items())),
            "paper_model_calls": (construction_embedding + query_embedding + answers),
            "judge_calls_excluded_from_paper_total": by_kind["judge"],
            "database_calls": by_kind["tridb_insert"] + by_kind["tridb_retrieval"],
            "failed_calls": failed,
            "total_ledger_records": len(self.records),
        }


class OpenAIEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        batch_size: int,
        timeout: float,
        ledger: CallLedger,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.ledger = ledger
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def discover_models(self) -> list[str]:
        response = self.session.get(
            f"{self.base_url}/models",
            timeout=self.timeout,
        )
        response.raise_for_status()
        self.model_records = list(response.json().get("data", []))
        return [str(item["id"]) for item in self.model_records if item.get("id")]

    def encode(self, texts: Sequence[str], *, phase: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        kind = (
            "construction_embedding" if phase == "construction" else "query_embedding"
        )
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            began = time.perf_counter()
            status = "ok"
            detail: dict[str, Any] = {}
            try:
                response = self.session.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                data = sorted(payload.get("data", []), key=lambda item: item["index"])
                batch_vectors = [item.get("embedding") for item in data]
                if len(batch_vectors) != len(batch) or any(
                    not isinstance(vector, list) for vector in batch_vectors
                ):
                    raise RuntimeError(
                        "embedding response does not contain one vector per input"
                    )
                vectors.extend(
                    [[float(value) for value in vector] for vector in batch_vectors]
                )
                detail["usage"] = payload.get("usage") or {}
            except BaseException as exc:
                status = "error"
                detail["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                raise
            finally:
                self.ledger.record(
                    kind=kind,
                    phase=phase,
                    target=self.model,
                    items=len(batch),
                    elapsed_seconds=time.perf_counter() - began,
                    status=status,
                    detail=detail,
                )
        return vectors


@dataclass(frozen=True)
class StreamingChatResult:
    text: str
    usage: dict[str, Any]
    request_started: float
    first_token_at: float
    completed_at: float


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float,
        ledger: CallLedger,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.ledger = ledger
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def discover_models(self) -> list[str]:
        response = self.session.get(
            f"{self.base_url}/models",
            timeout=self.timeout,
        )
        response.raise_for_status()
        self.model_records = list(response.json().get("data", []))
        return [str(item["id"]) for item in self.model_records if item.get("id")]

    def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> StreamingChatResult:
        request_started = time.perf_counter()
        first_token_at: float | None = None
        pieces: list[str] = []
        usage: dict[str, Any] = {}
        status = "ok"
        detail: dict[str, Any] = {}
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": list(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                payload = json.loads(data)
                if payload.get("usage"):
                    usage = dict(payload["usage"])
                for choice in payload.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        pieces.append(content)
            completed_at = time.perf_counter()
            if first_token_at is None:
                raise RuntimeError("stream completed without an answer token")
            detail["usage"] = usage
            return StreamingChatResult(
                text="".join(pieces).strip(),
                usage=usage,
                request_started=request_started,
                first_token_at=first_token_at,
                completed_at=completed_at,
            )
        except BaseException as exc:
            status = "error"
            detail["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            self.ledger.record(
                kind="answer_generation",
                phase="qa",
                target=model,
                items=1,
                elapsed_seconds=time.perf_counter() - request_started,
                status=status,
                detail=detail,
            )

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int,
        temperature: float,
        seed: int,
        ledger_kind: str,
    ) -> tuple[str, dict[str, Any], float]:
        began = time.perf_counter()
        status = "ok"
        detail: dict[str, Any] = {}
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": list(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": seed,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices", [])
            if not choices:
                raise RuntimeError("chat response contains no choices")
            content = choices[0].get("message", {}).get("content")
            if not isinstance(content, str):
                raise RuntimeError("chat response contains no text")
            usage = dict(payload.get("usage") or {})
            detail["usage"] = usage
            return content.strip(), usage, time.perf_counter() - began
        except BaseException as exc:
            status = "error"
            detail["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise
        finally:
            self.ledger.record(
                kind=ledger_kind,
                phase="evaluation",
                target=model,
                items=1,
                elapsed_seconds=time.perf_counter() - began,
                status=status,
                detail=detail,
            )


def extract_retrieval_query(question: str) -> str:
    match = re.search(r"Now Answer the Question:\s*(.*)", question, re.DOTALL)
    if match is None:
        return question.strip()
    return match.group(1).strip()


def build_answer_messages(
    question: str,
    retrieved_texts: Sequence[str],
) -> list[dict[str, str]]:
    memories = "\n".join(
        f"Memory {index}:\n{text}"
        for index, text in enumerate(retrieved_texts, start=1)
    )
    prompt = f"{memories}\n{QUERY_TEMPLATE.format(question=question)}"
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]


def fit_answer_prompt(
    question: str,
    retrieved_texts: Sequence[str],
    *,
    token_counter: Any,
    token_budget: int,
) -> tuple[list[dict[str, str]], int, int]:
    """Keep ranked memories until the configured local-model prompt budget."""
    if token_budget <= 0:
        raise ValueError("prompt token budget must be positive")
    selected: list[str] = []
    selected_messages = build_answer_messages(question, selected)
    selected_tokens = sum(
        int(token_counter(message["content"])) for message in selected_messages
    )
    for text in retrieved_texts:
        candidate = build_answer_messages(question, [*selected, text])
        candidate_tokens = sum(
            int(token_counter(message["content"])) for message in candidate
        )
        if candidate_tokens > token_budget:
            break
        selected.append(text)
        selected_messages = candidate
        selected_tokens = candidate_tokens
    if not selected and retrieved_texts:
        raise RuntimeError(
            "the highest-ranked memory does not fit the prompt token budget"
        )
    return selected_messages, len(selected), selected_tokens


def build_judge_prompt(
    question_type: str,
    question: str,
    answer: Any,
    response: str,
    *,
    abstention: bool,
) -> str:
    """Reproduce MemoryAgentBench's official LongMemEval judge prompts."""
    if abstention:
        template = (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: "
            "{}\n\nModel Response: {}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )
    elif question_type in {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
    }:
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. In addition, do not penalize off-by-one errors for the number of "
            "days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still correct. "
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs "
            "the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some previous "
            "information along with an updated answer, the response should be "
            "considered as correct as long as the updated answer is the required "
            "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\nQuestion: {}\n\nRubric: "
            "{}\n\nModel Response: {}\n\nIs the model response correct? Answer "
            "yes or no only."
        )
    else:
        raise ValueError(f"unsupported LongMemEval question type: {question_type!r}")
    return template.format(question, answer, response)


def parse_judge_yes_no(text: str) -> bool:
    match = re.search(r"\b(yes|no)\b", text.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(f"judge did not return yes or no: {text!r}")
    return match.group(1).lower() == "yes"


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    collected = [float(value) for value in values]
    if not collected:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    p50 = _percentile(collected, 50)
    p95 = _percentile(collected, 95)
    return {
        "count": len(collected),
        "mean": sum(collected) / len(collected),
        "p50": p50,
        "p95": p95,
        "p99": _percentile(collected, 99),
        "max": max(collected),
        "p95_over_p50": None if not p50 else p95 / p50,
    }


def _wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [centre - radius, centre + radius]


def build_summary(
    *,
    manifest: dict[str, Any],
    predictions: Sequence[dict[str, Any]],
    judge_results: Sequence[dict[str, Any]],
    ledger: CallLedger,
    construction_records: Sequence[dict[str, Any]],
    lifecycle_seconds: float,
) -> dict[str, Any]:
    judged = [result for result in judge_results if "correct" in result]
    correct = sum(bool(result["correct"]) for result in judged)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for result in judged:
        by_type[str(result["question_type"])].append(bool(result["correct"]))

    timings = [prediction["timing"] for prediction in predictions]
    return {
        "schema_version": "tridb_longmemeval_pipeline_v0.1.0",
        "status": (
            "completed"
            if len(predictions) == manifest["workload"]["selected_questions"]
            else "partial"
        ),
        "paper_reference": {
            "configuration": "embedRAG",
            "accuracy": 0.398,
            "lifecycle_wall_seconds": 14.4 * 60,
            "model_calls": 610,
            "ttft_p50_seconds": 1.96,
            "total_time_p50_seconds": 2.78,
        },
        "accuracy": {
            "judge_model": manifest["models"].get("judge"),
            "judge_protocol": manifest["evaluation"]["judge_protocol"],
            "judged": len(judged),
            "correct": correct,
            "accuracy": None if not judged else correct / len(judged),
            "wilson_95": _wilson_interval(correct, len(judged)),
            "by_question_type": {
                question_type: {
                    "count": len(values),
                    "correct": sum(values),
                    "accuracy": sum(values) / len(values),
                }
                for question_type, values in sorted(by_type.items())
            },
        },
        "walltime": {
            "construction_plus_qa_seconds": lifecycle_seconds,
            "construction_seconds_sum": sum(
                float(record["construction_seconds"]) for record in construction_records
            ),
            "qa_seconds_sum": sum(float(timing["total_seconds"]) for timing in timings),
            "judge_seconds_excluded": sum(
                float(result.get("judge_seconds", 0)) for result in judge_results
            ),
            "by_history": list(construction_records),
        },
        "latency": {
            "effective_ttft_seconds": latency_summary(
                timing["effective_ttft_seconds"] for timing in timings
            ),
            "total_time_seconds": latency_summary(
                timing["total_seconds"] for timing in timings
            ),
            "query_embedding_seconds": latency_summary(
                timing["query_embedding_seconds"] for timing in timings
            ),
            "tridb_retrieval_seconds": latency_summary(
                timing["tridb_retrieval_seconds"] for timing in timings
            ),
            "prompt_assembly_seconds": latency_summary(
                timing["prompt_assembly_seconds"] for timing in timings
            ),
            "vllm_queue_prefill_seconds": latency_summary(
                timing["vllm_queue_prefill_seconds"] for timing in timings
            ),
            "decode_seconds": latency_summary(
                timing["decode_seconds"] for timing in timings
            ),
        },
        "calls": ledger.summary(),
        "tokens": {
            "prompt_tokens": sum(
                int(prediction.get("usage", {}).get("prompt_tokens", 0))
                for prediction in predictions
            ),
            "completion_tokens": sum(
                int(prediction.get("usage", {}).get("completion_tokens", 0))
                for prediction in predictions
            ),
        },
        "comparability": {
            "answer_model_match": (
                manifest["models"]["answer"] == DEFAULT_ANSWER_MODEL
            ),
            "embedding_model_match": (
                manifest["models"]["embedding"] == DEFAULT_EMBEDDING_MODEL
            ),
            "paper_hardware_match": False,
            "judge_protocol_match": (
                manifest["evaluation"]["judge_protocol"] == "memoryagentbench_gpt4o"
            ),
            "note": (
                "Current-hardware TriDB reproduction. Absolute walltime and latency "
                "must not be presented as an H100 hardware replication."
            ),
        },
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False))
            output.write("\n")
            output.flush()


def _validate_single_model(
    advertised: Sequence[str],
    expected: str,
    *,
    endpoint_name: str,
) -> None:
    if list(advertised) != [expected]:
        raise RuntimeError(
            f"{endpoint_name} must advertise exactly {expected!r}; "
            f"found {list(advertised)!r}"
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    workloads = load_workloads(
        args.input,
        source=args.source,
        strict_shape=not args.allow_nonstandard_shape,
    )
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
    selected_questions = sum(len(workload.questions) for workload in workloads)
    if selected_questions == 0:
        raise ValueError("no questions selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = CallLedger()
    answer_client = OpenAIChatClient(
        args.answer_base_url,
        args.answer_api_key,
        timeout=args.request_timeout,
        ledger=ledger,
    )
    embedding_client = OpenAIEmbeddingClient(
        args.embedding_base_url,
        args.embedding_api_key,
        args.embedding_model,
        batch_size=args.embedding_batch_size,
        timeout=args.request_timeout,
        ledger=ledger,
    )
    advertised_answer_models = answer_client.discover_models()
    advertised_embedding_models = embedding_client.discover_models()
    _validate_single_model(
        advertised_answer_models,
        args.answer_model,
        endpoint_name="answer endpoint",
    )
    _validate_single_model(
        advertised_embedding_models,
        args.embedding_model,
        endpoint_name="embedding endpoint",
    )

    backend = TriDBMemoryBackend.connect(
        args.dsn,
        dim=args.embedding_dim,
        table=args.table,
    )
    backend.init_schema()
    chunker = TiktokenSentenceChunker(
        chunk_size=args.chunk_size,
        tokenizer_model=args.chunk_tokenizer,
    )

    judge_enabled = not args.skip_judge
    judge_client = (
        OpenAIChatClient(
            args.judge_base_url,
            args.judge_api_key,
            timeout=args.request_timeout,
            ledger=ledger,
        )
        if judge_enabled
        else None
    )
    judge_protocol = (
        "memoryagentbench_gpt4o"
        if judge_enabled
        and args.judge_model == "gpt-4o"
        and args.judge_base_url.rstrip("/") == "https://api.openai.com/v1"
        else "protocol_variant"
        if judge_enabled
        else "not_run"
    )
    manifest = {
        "schema_version": "tridb_longmemeval_manifest_v0.1.0",
        "started_at": _utc_now(),
        "input": {
            "path": str(args.input.resolve()),
            "sha256": _sha256(args.input),
            "source": args.source,
        },
        "workload": {
            "histories": len(workloads),
            "questions_per_history": [
                len(workload.questions) for workload in workloads
            ],
            "selected_questions": selected_questions,
            "chunk_size_tokens": args.chunk_size,
            "chunk_tokenizer": args.chunk_tokenizer,
            "top_k": args.top_k,
            "max_prompt_memories": args.max_prompt_memories,
            "prompt_token_budget": args.prompt_token_budget,
            "embedding_batch_size": args.embedding_batch_size,
            "serial_execution": True,
        },
        "models": {
            "answer": args.answer_model,
            "answer_base_url": args.answer_base_url,
            "answer_endpoint_models": getattr(
                answer_client,
                "model_records",
                [{"id": model} for model in advertised_answer_models],
            ),
            "embedding": args.embedding_model,
            "embedding_base_url": args.embedding_base_url,
            "embedding_endpoint_models": getattr(
                embedding_client,
                "model_records",
                [{"id": model} for model in advertised_embedding_models],
            ),
            "embedding_dim": args.embedding_dim,
            "judge": args.judge_model if judge_enabled else None,
            "judge_base_url": args.judge_base_url if judge_enabled else None,
            "thinking_enabled": False,
        },
        "generation": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.answer_max_tokens,
            "stream": True,
        },
        "evaluation": {
            "judge_enabled": judge_enabled,
            "judge_protocol": judge_protocol,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_calls_excluded_from_serving_metrics": True,
        },
        "tridb": {
            "dsn_redacted": re.sub(r"://[^@]+@", "://***@", args.dsn),
            "table": args.table,
            "mode": "vector_with_relational_scope",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_state(),
        },
    }
    _atomic_write_json(args.output_dir / "run_manifest.json", manifest)

    predictions: list[dict[str, Any]] = []
    construction_records: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    lifecycle_started = time.perf_counter()
    try:
        for history_index, workload in enumerate(workloads):
            construction_started = time.perf_counter()
            chunk_started = time.perf_counter()
            chunks = chunker.chunk(workload.context)
            chunk_seconds = time.perf_counter() - chunk_started
            if not chunks:
                raise RuntimeError(f"{workload.scope_id} produced no chunks")

            embedding_started = time.perf_counter()
            vectors = embedding_client.encode(chunks, phase="construction")
            embedding_seconds = time.perf_counter() - embedding_started
            if any(len(vector) != args.embedding_dim for vector in vectors):
                dimensions = sorted({len(vector) for vector in vectors})
                raise RuntimeError(
                    f"embedding endpoint returned dimensions {dimensions}, "
                    f"expected {args.embedding_dim}"
                )
            units = [
                MemoryUnit(
                    scope_id=workload.scope_id,
                    external_id=f"{workload.scope_id}_chunk_{index:04d}",
                    session_id=workload.scope_id,
                    kind="chunk",
                    content=chunk,
                    event_order=index,
                    metadata={
                        "source": workload.source,
                        "history_index": history_index,
                        "chunk_index": index,
                    },
                    embedding=vector,
                )
                for index, (chunk, vector) in enumerate(
                    zip(chunks, vectors, strict=True)
                )
            ]
            insert_started = time.perf_counter()
            insert_status = "ok"
            try:
                inserted = backend.replace_scope(
                    workload.scope_id,
                    units,
                    isolated=True,
                )
            except BaseException:
                insert_status = "error"
                raise
            finally:
                insert_seconds = time.perf_counter() - insert_started
                ledger.record(
                    kind="tridb_insert",
                    phase="construction",
                    target=args.table,
                    items=len(units),
                    elapsed_seconds=insert_seconds,
                    status=insert_status,
                )
            if inserted != len(units):
                raise RuntimeError(f"TriDB inserted {inserted} of {len(units)} chunks")
            construction_seconds = time.perf_counter() - construction_started
            construction_record = {
                "history_index": history_index,
                "scope_id": workload.scope_id,
                "chunks": len(chunks),
                "questions": len(workload.questions),
                "construction_start_offset_seconds": (
                    construction_started - lifecycle_started
                ),
                "construction_end_offset_seconds": (
                    time.perf_counter() - lifecycle_started
                ),
                "chunking_seconds": chunk_seconds,
                "construction_embedding_seconds": embedding_seconds,
                "tridb_insert_and_index_seconds": insert_seconds,
                "construction_seconds": construction_seconds,
            }
            construction_records.append(construction_record)
            event_rows.append(
                {
                    "event": "construction_complete",
                    "history_index": history_index,
                    **construction_record,
                }
            )

            for question_index, question in enumerate(workload.questions):
                admitted = time.perf_counter()
                retrieval_query = extract_retrieval_query(question.question)

                query_embedding_started = time.perf_counter()
                query_vector = embedding_client.encode(
                    [retrieval_query],
                    phase="query",
                )[0]
                query_embedding_seconds = time.perf_counter() - query_embedding_started
                query_embedding_ended = time.perf_counter()
                if len(query_vector) != args.embedding_dim:
                    raise RuntimeError(
                        f"query embedding has {len(query_vector)} dimensions; "
                        f"expected {args.embedding_dim}"
                    )

                retrieval_started = time.perf_counter()
                retrieval_status = "ok"
                try:
                    hits = backend.search(
                        workload.scope_id,
                        query_embedding=query_vector,
                        k=args.top_k,
                    )
                except BaseException:
                    retrieval_status = "error"
                    raise
                finally:
                    retrieval_seconds = time.perf_counter() - retrieval_started
                    retrieval_ended = time.perf_counter()
                    ledger.record(
                        kind="tridb_retrieval",
                        phase="qa",
                        target=args.table,
                        items=1,
                        elapsed_seconds=retrieval_seconds,
                        status=retrieval_status,
                    )

                prompt_started = time.perf_counter()
                messages, prompt_hit_count, estimated_prompt_tokens = fit_answer_prompt(
                    question.question,
                    [hit.content for hit in hits[: args.max_prompt_memories]],
                    token_counter=chunker.count,
                    token_budget=args.prompt_token_budget,
                )
                prompt_seconds = time.perf_counter() - prompt_started
                prompt_ended = time.perf_counter()
                generation = answer_client.stream_chat(
                    model=args.answer_model,
                    messages=messages,
                    max_tokens=args.answer_max_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                )
                completed = generation.completed_at
                timing = {
                    "query_admitted_offset_seconds": admitted - lifecycle_started,
                    "query_embedding_start_offset_seconds": (
                        query_embedding_started - lifecycle_started
                    ),
                    "query_embedding_end_offset_seconds": (
                        query_embedding_ended - lifecycle_started
                    ),
                    "retrieval_start_offset_seconds": (
                        retrieval_started - lifecycle_started
                    ),
                    "retrieval_end_offset_seconds": (
                        retrieval_ended - lifecycle_started
                    ),
                    "prompt_assembly_start_offset_seconds": (
                        prompt_started - lifecycle_started
                    ),
                    "prompt_assembly_end_offset_seconds": (
                        prompt_ended - lifecycle_started
                    ),
                    "generation_request_sent_offset_seconds": (
                        generation.request_started - lifecycle_started
                    ),
                    "generation_first_token_offset_seconds": (
                        generation.first_token_at - lifecycle_started
                    ),
                    "generation_last_token_offset_seconds": (
                        generation.completed_at - lifecycle_started
                    ),
                    "query_embedding_seconds": query_embedding_seconds,
                    "tridb_retrieval_seconds": retrieval_seconds,
                    "prompt_assembly_seconds": prompt_seconds,
                    "vllm_queue_prefill_seconds": (
                        generation.first_token_at - generation.request_started
                    ),
                    "effective_ttft_seconds": (generation.first_token_at - admitted),
                    "decode_seconds": (
                        generation.completed_at - generation.first_token_at
                    ),
                    "total_seconds": completed - admitted,
                }
                prediction = {
                    "history_index": history_index,
                    "question_index": question_index,
                    "scope_id": workload.scope_id,
                    "question_id": question.question_id,
                    "qa_pair_id": question.qa_pair_id,
                    "question_type": question.question_type,
                    "question": question.question,
                    "answer": question.answer,
                    "prediction": generation.text,
                    "retrieval_query": retrieval_query,
                    "retrieved": [
                        {
                            "rank": rank,
                            "external_id": hit.external_id,
                            "score": hit.score,
                            "content": hit.content,
                        }
                        for rank, hit in enumerate(hits, start=1)
                    ],
                    "retrieved_count": len(hits),
                    "prompt_memory_count": prompt_hit_count,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "usage": generation.usage,
                    "timing": timing,
                }
                predictions.append(prediction)
                event_rows.append(
                    {
                        "event": "query_complete",
                        "history_index": history_index,
                        "question_index": question_index,
                        "question_id": question.question_id,
                        **timing,
                    }
                )
                _write_jsonl(
                    args.output_dir / "predictions.jsonl",
                    predictions,
                )
                _write_jsonl(args.output_dir / "events.jsonl", event_rows)
                print(
                    f"[longmemeval] {len(predictions)}/{selected_questions} "
                    f"{question.question_id} ttft={timing['effective_ttft_seconds']:.3f}s "
                    f"total={timing['total_seconds']:.3f}s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        backend.close()
    serving_completed = time.perf_counter()
    lifecycle_seconds = serving_completed - lifecycle_started

    judge_results: list[dict[str, Any]] = []
    if judge_client is not None:
        for index, prediction in enumerate(predictions, start=1):
            prompt = build_judge_prompt(
                prediction["question_type"],
                prediction["question"],
                prediction["answer"],
                prediction["prediction"],
                abstention="_abs" in prediction["question_id"],
            )
            raw, usage, judge_seconds = judge_client.chat(
                model=args.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.judge_max_tokens,
                temperature=0.0,
                seed=args.seed,
                ledger_kind="judge",
            )
            result = {
                "question_id": prediction["question_id"],
                "question_type": prediction["question_type"],
                "judge_model": args.judge_model,
                "correct": parse_judge_yes_no(raw),
                "raw": raw,
                "usage": usage,
                "judge_seconds": judge_seconds,
            }
            judge_results.append(result)
            _write_jsonl(
                args.output_dir / "judge_results.jsonl",
                judge_results,
            )
            print(
                f"[longmemeval-judge] {index}/{len(predictions)} "
                f"{prediction['question_id']} correct={result['correct']}",
                file=sys.stderr,
                flush=True,
            )

    _write_jsonl(args.output_dir / "call_ledger.jsonl", ledger.records)
    summary = build_summary(
        manifest=manifest,
        predictions=predictions,
        judge_results=judge_results,
        ledger=ledger,
        construction_records=construction_records,
        lifecycle_seconds=lifecycle_seconds,
    )
    summary["completed_at"] = _utc_now()
    _atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--dsn", default=os.environ.get("TRIDB_DSN", DEFAULT_DSN))
    parser.add_argument("--table", default="longmemeval_mab_units")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-prompt-memories",
        type=int,
        default=5,
        help=(
            "maximum retrieved chunks assembled for the local answer model; "
            "the paper uses five when ten 4,096-token chunks overflow context"
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--chunk-tokenizer", default="gpt-4o-mini")
    parser.add_argument(
        "--prompt-token-budget",
        type=int,
        default=36_000,
        help=(
            "estimated input-token budget; lower-ranked retrieved chunks are "
            "dropped before generation when they do not fit"
        ),
    )
    parser.add_argument(
        "--answer-base-url",
        default=os.environ.get("VLLM_BASE_URL", DEFAULT_ANSWER_BASE_URL),
    )
    parser.add_argument(
        "--answer-api-key",
        default=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get(
            "EMBEDDING_BASE_URL",
            DEFAULT_EMBEDDING_BASE_URL,
        ),
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.environ.get("EMBEDDING_API_KEY", "EMPTY"),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help=(
            "texts per embedding HTTP request; 64 yields about two construction "
            "calls per 360K-token history and matches the paper's call accounting"
        ),
    )
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--allow-nonstandard-shape", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get(
            "JUDGE_BASE_URL",
            "https://api.openai.com/v1",
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.environ.get("JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--judge-max-tokens", type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_prompt_memories <= 0:
        parser.error("--max-prompt-memories must be positive")
    if args.max_prompt_memories > args.top_k:
        parser.error("--max-prompt-memories cannot exceed --top-k")
    if args.prompt_token_budget <= 0:
        parser.error("--prompt-token-budget must be positive")
    if args.limit_samples is not None and args.limit_samples <= 0:
        parser.error("--limit-samples must be positive")
    if args.limit_questions is not None and args.limit_questions <= 0:
        parser.error("--limit-questions must be positive")
    if not args.skip_judge and not args.judge_api_key:
        parser.error(
            "--judge-api-key/OPENAI_API_KEY is required unless --skip-judge is used"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_pipeline(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
