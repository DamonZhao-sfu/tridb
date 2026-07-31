"""The serving layer shared by every LongMemEval arm: workload, models, judge.

Extracted verbatim from ``tridbBackend/longmemeval_pipeline.py`` for the same
reason ``chunking.py`` was extracted from it earlier — the GEM arm and the
embedRAG arm must generate, time, and grade through the SAME code, or their
numbers are not comparable and the drift is invisible. ``longmemeval_pipeline``
now imports these names rather than defining them.

What lives here is everything that is *not* a memory system:

    workload      LongMemEval_S* loading and the five-history shape gate
    telemetry     ``CallLedger`` — every model call, tagged with its phase
    embedding     ``OpenAIEmbeddingClient`` (vLLM ``/v1/embeddings``)
    generation    ``OpenAIChatClient`` — STREAMING, so effective TTFT is real
    prompting     retrieval-query extraction, prompt assembly, budget fitting
    grading       MemoryAgentBench's official LongMemEval judge prompts
    statistics    percentile / tail summaries, Wilson intervals

**Effective TTFT** ([AM] Fig. 10) is the pre-answer wait measured from query
admission, so it must include query embedding, retrieval, prompt assembly,
serving queue, and prefill. ``stream_chat`` reports ``first_token_at`` on the
caller's ``perf_counter`` timeline and the caller subtracts its own admission
stamp; nothing here may buffer the stream, or that number becomes fiction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

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


# ---------------------------------------------------------------------------
# small file / process helpers
# ---------------------------------------------------------------------------


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False))
            output.write("\n")
            output.flush()


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


# ---------------------------------------------------------------------------
# workload
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------


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
        construction_llm = by_kind["construction_llm"]
        query_embedding = by_kind["query_embedding"]
        answers = by_kind["answer_generation"]
        return {
            "by_kind": dict(sorted(by_kind.items())),
            "items_by_kind": dict(sorted(item_counts.items())),
            "paper_model_calls": (
                construction_embedding + construction_llm + query_embedding + answers
            ),
            "construction_calls": construction_embedding + construction_llm,
            "qa_calls": query_embedding + answers,
            "judge_calls_excluded_from_paper_total": by_kind["judge"],
            "database_calls": (
                by_kind["tridb_insert"]
                + by_kind["tridb_retrieval"]
                + by_kind["gem_ingest"]
                + by_kind["gem_retrieve"]
            ),
            "failed_calls": failed,
            "total_ledger_records": len(self.records),
        }

    def tokens(self) -> dict[str, int]:
        """Prompt / completion / embedding tokens, split by phase.

        [AM] Table 3 prices a whole lifecycle, so construction tokens and QA
        tokens are never pooled: a Paradigm IV arm can spend two orders of
        magnitude more on construction than on all 300 queries combined.
        """
        totals: dict[str, int] = {
            "construction_prompt_tokens": 0,
            "construction_completion_tokens": 0,
            "construction_embed_tokens": 0,
            "qa_prompt_tokens": 0,
            "qa_completion_tokens": 0,
            "qa_embed_tokens": 0,
        }
        for record in self.records:
            usage = (record.get("detail") or {}).get("usage") or {}
            if not usage:
                continue
            kind = str(record["kind"])
            if kind == "judge":
                continue  # graded, not served — never in a serving total
            bucket = "construction" if record["phase"] == "construction" else "qa"
            if kind.endswith("embedding"):
                totals[f"{bucket}_embed_tokens"] += int(usage.get("prompt_tokens", 0))
                continue
            totals[f"{bucket}_prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            totals[f"{bucket}_completion_tokens"] += int(
                usage.get("completion_tokens", 0)
            )
        return totals


# ---------------------------------------------------------------------------
# model clients
# ---------------------------------------------------------------------------


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


class PhasedEmbedder:
    """Adapts ``OpenAIEmbeddingClient`` to GEM's one-argument embedder contract.

    GEM operators call ``embedder.encode(texts)`` with no phase, because a GEM
    operator has no business knowing which [AM] phase it is being profiled in.
    The runner binds the phase around each operator call instead::

        with embedder.phase("construction"):
            memory.ingest(...)

    Serial by construction — the pipelines set ``serial_execution: True`` and a
    concurrent caller would cross-tag its neighbour's calls, so re-entry with a
    different phase is refused rather than silently mislabelled.
    """

    def __init__(self, client: OpenAIEmbeddingClient, *, default_phase: str = "query"):
        self.client = client
        self.default_phase = default_phase
        self._phase = default_phase
        self._depth = 0

    @property
    def batch_size(self) -> int:
        return self.client.batch_size

    def phase(self, name: str) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def _bind():
            if self._depth and name != self._phase:
                raise RuntimeError(
                    f"embedder phase {self._phase!r} is already bound; "
                    f"refusing to nest {name!r} — calls would be cross-tagged"
                )
            previous = self._phase
            self._phase = name
            self._depth += 1
            try:
                yield self
            finally:
                self._depth -= 1
                self._phase = previous

        return _bind()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self.client.encode(texts, phase=self._phase)


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
        phase: str = "evaluation",
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
                phase=phase,
                target=model,
                items=1,
                elapsed_seconds=time.perf_counter() - began,
                status=status,
                detail=detail,
            )


class LedgerChatExtractor:
    """``complete(system, user)`` for GEM strategies, on the shared ledger.

    ``gem.strategies.OpenAIChatExtractor`` implements the same contract but
    keeps its own session and reports to nobody. Construction LLM calls are
    exactly what [AM] §4.2 prices, so an arm whose extractor bypasses the
    ledger would report construction as free. Same endpoint as generation by
    default, which is the co-location [AM] §4.3 measures.
    """

    def __init__(
        self,
        client: OpenAIChatClient,
        *,
        model: str,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 1024,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        text, usage, _ = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            ledger_kind="construction_llm",
            phase="construction",
        )
        return text, usage


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


# ---------------------------------------------------------------------------
# prompting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------


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


def judge_protocol_label(
    *,
    enabled: bool,
    model: str,
    base_url: str,
) -> str:
    """Name the grading protocol so a local judge can never pass as the paper's.

    MemoryAgentBench grades LongMemEval with hosted ``gpt-4o``. Any other model
    or endpoint is a protocol VARIANT: the prompts are identical but the grader
    is not, so accuracy is comparable within this repository and not against
    [AM]'s published numbers.
    """
    if not enabled:
        return "not_run"
    if model == "gpt-4o" and base_url.rstrip("/") == "https://api.openai.com/v1":
        return "memoryagentbench_gpt4o"
    return "protocol_variant"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


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
