"""Alternative backends of the retrieval "Rank" stage (spec 2026-11-03 execution-engine extensions §2,
``rank_impl``), next to the existing ColBERT ranker (``colbert_rank.ColBERTRank``).

Each ranker satisfies the ``semrank`` contract of ``session_retrieval.retrieve``: ``ranker(cards) -> {id: score}``
with exactly one score per card id (or an exception), and appends one record per HTTP request to ``calls`` in the
format ``tools/trace_joint_episode_worker.semantic_tokens`` sums (``call["usage"]["total_tokens"]``, an int >= 0).

* ``CrossEncoderRank``: Qwen3-Reranker (default ``Qwen/Qwen3-Reranker-0.6B``) on a vLLM pooling server, one
  ``POST {base}/v1/rerank`` per retrieval. Counts toward retrieval latency, NOT toward LLM tokens: like ColBERT, each
  record carries ``usage.total_tokens == 0``; the reranker's own prompt-token count is kept apart in
  ``rerank_usage`` (informational).
* ``LLMPointwiseRank``: the agent LLM scores every card 0..10 through chat completions with a strict guided JSON
  schema, ``batch`` cards per call. Counts toward retrieval tokens and latency: each record carries the server's
  ``usage`` (same record shape as ``semfilter_v2.BatchScoreSemFilter``).

Both render a card's document text the way ``ColBERTRank`` does: ``render(card)[:max_chars]`` (``card_text``).
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import httpx

from trimem.experiments import semantic_rubric as sr
from trimem.experiments import semfilter_v2 as sf2

CROSS_ENCODER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
CROSS_ENCODER_BASE = (
    "http://127.0.0.1:8050"  # scripts/serve_qwen3_reranker.sh default port
)
RANK_MAX_CHARS = 12000  # = ColBERTRank's max_chars default
DEFAULT_LLM_BATCH = 8
DEFAULT_LLM_WORKERS = 4
# Answer budget for one batched LLM-rank call. SemFilter's 10 x K + 16 (= 96 for K=8) is too tight here:
# guided JSON may pretty-print, and a 5-digit id line costs ~11-12 tokens, so K=8 hit the cap and was cut
# (c16 smoke 2026-11-03, 2/4 beam+LLM-rank episodes failed). 24 x K + 32 leaves 2x headroom.
LLM_RANK_TOKENS_PER_ID, LLM_RANK_TOKENS_BASE = 24, 32


def llm_rank_max_tokens(k: int) -> int:
    return LLM_RANK_TOKENS_PER_ID * k + LLM_RANK_TOKENS_BASE

LLM_SCORE_LEVELS = tuple(
    range(11)
)  # integers 0..10 (see semfilter_v2.SCORE_LEVELS: why integers)
ZERO_USAGE = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}


def cross_encoder_version(model: str = CROSS_ENCODER_MODEL) -> str:
    return f"cross-encoder-rank-v1:{model}"


def llm_rank_version(model: str = sr.MODEL) -> str:
    return f"llm-pointwise-rank-v1:{model}"


def card_text(render, card: dict, max_chars: int = RANK_MAX_CHARS) -> str:
    """The document text of one card, exactly as ``ColBERTRank`` builds it."""
    return render(card)[:max_chars]


def _api_base(base: str) -> str:
    return base.rstrip("/").removesuffix("/v1")


def _unique_ids(cards: list[dict]) -> list:
    ids = [c["id"] for c in cards]
    if len({str(i) for i in ids}) != len(ids):
        raise ValueError("rank: duplicate card ids in the input")
    return ids


# ---------------------------------------------------------------- cross-encoder (vLLM /v1/rerank)


class CrossEncoderRank:
    """Qwen3-Reranker cross-encoder on vLLM: one ``/v1/rerank`` request scores the query against every card."""

    def __init__(
        self,
        query: str,
        render,
        base: str = CROSS_ENCODER_BASE,
        model: str = CROSS_ENCODER_MODEL,
        timeout: float = 600,
        max_chars: int = RANK_MAX_CHARS,
    ):
        self.query, self.render, self.base = query, render, _api_base(base)
        self.model, self.timeout, self.max_chars = model, timeout, max_chars
        self.version = cross_encoder_version(model)
        self.calls: list[dict] = []

    def __call__(self, cards: list[dict]) -> dict:
        if not cards:
            return {}
        ids = _unique_ids(cards)
        docs = [card_text(self.render, c, self.max_chars) for c in cards]
        t0 = time.monotonic()
        r = httpx.post(
            f"{self.base}/v1/rerank",
            json={"model": self.model, "query": self.query, "documents": docs},
            timeout=self.timeout,
        )
        r.raise_for_status()
        j = r.json()
        self.calls.append(
            {
                "usage": dict(ZERO_USAGE),  # no LLM tokens (as ColBERT)
                "seconds": round(time.monotonic() - t0, 3),
                "model": j.get("model"),
                "kind": "cross_encoder",
                "n": len(cards),
                "rerank_usage": j.get(
                    "usage"
                ),  # reranker prompt tokens, informational only
            }
        )
        return self._scores(j, ids)

    @staticmethod
    def _scores(j: dict, ids: list) -> dict:
        results = j.get("results") if isinstance(j, dict) else None
        if not isinstance(results, list):
            raise ValueError("rerank: response has no results list")  # noqa: TRY004 (malformed response)
        scores: dict = {}
        for res in results:
            idx = res.get("index") if isinstance(res, dict) else None
            if type(idx) is not int or not 0 <= idx < len(ids):
                raise ValueError(f"rerank: bad result index {idx!r}")
            if ids[idx] in scores:
                raise ValueError(f"rerank: duplicate result index {idx}")
            s = res.get("relevance_score")
            if (
                isinstance(s, bool)
                or not isinstance(s, (int, float))
                or not math.isfinite(s)
            ):
                raise ValueError(f"rerank: bad relevance_score {s!r} at index {idx}")
            scores[ids[idx]] = float(s)
        if len(scores) != len(ids):
            missing = [i for i in ids if i not in scores]
            raise ValueError(f"rerank: no score for card ids {missing}")
        return {i: scores[i] for i in ids}


# ---------------------------------------------------------------- LLM pointwise (guided JSON)

LLM_RANK_SYSTEM = """You rate how RELEVANT each past episode is to a new task. You will see the NEW TASK, then
SEVERAL past episodes, each labelled PAST EPISODE <id>. Rate EACH one independently, as if it were the only episode
shown: how useful would its content be to an agent about to solve the new task (same goal, objects, places, tools or
procedure)?

Scale (integer 0..10): 0 = unrelated; 3 = same domain but a different goal; 6 = a similar goal with partly
transferable steps; 10 = practically the same task, its steps transfer directly.

Answer JSON only: ONE object whose keys are exactly the past-episode ids (as strings), each mapped to that episode's
integer score 0..10:
{"<id>": <0..10>, "<id>": <0..10>, ...}"""


def llm_rank_schema(ids: list) -> dict:
    """Strict guided-decoding schema: an object with exactly the ids (strings) as keys, each an 0..10 integer enum."""
    keys = [str(i) for i in ids]
    level = {"enum": list(LLM_SCORE_LEVELS)}
    return {
        "type": "object",
        "properties": {k: level for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def llm_rank_response_format(ids: list) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rank_scores",
            "schema": llm_rank_schema(ids),
            "strict": True,
        },
    }


def _no_duplicate_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def parse_llm_scores(content: str | None, ids: list) -> dict | None:
    """``{id: float score}`` when ``content`` is ONE JSON object whose keys are exactly the ids (as strings, each
    once) and every value an integer 0..10; None otherwise (malformed)."""
    try:
        d = json.loads(content or "", object_pairs_hook=_no_duplicate_keys)
    except (TypeError, ValueError):
        return None
    want = {str(i): i for i in ids}
    if (
        not isinstance(d, dict)
        or {str(k).strip() for k in d} != set(want)
        or len(d) != len(want)
    ):
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if not math.isfinite(v) or v != int(v) or not 0 <= v <= 10:
            return None
        out[want[str(k).strip()]] = float(v)
    return out


class LLMPointwiseRank:
    """Agent-LLM pointwise ranker: ``batch`` cards per chat-completions call (temperature 0, thinking off, strict
    guided JSON), batches on ``workers`` threads. A malformed answer (unparseable, a missing / extra / duplicate id,
    a non-integer or out-of-range score) is retried ONCE for that batch, then raises ``ValueError``."""

    def __init__(
        self,
        query: str,
        render,
        base: str = "http://127.0.0.1:8030",
        model: str = sr.MODEL,
        batch: int = DEFAULT_LLM_BATCH,
        max_tokens: int | None = None,
        workers: int = DEFAULT_LLM_WORKERS,
        max_chars: int = RANK_MAX_CHARS,
        timeout: float = 600,
    ):
        if not 1 <= batch <= 32:
            raise ValueError("batch must be in [1, 32]")
        if max_tokens is not None and not 8 <= max_tokens <= 2048:
            raise ValueError("max_tokens must be in [8, 2048]")
        self.query, self.render, self.base = query, render, _api_base(base)
        self.model, self.batch, self.max_tokens = model, batch, max_tokens
        self.workers, self.max_chars, self.timeout = max(1, workers), max_chars, timeout
        self.version = llm_rank_version(model)
        self.calls: list[dict] = []
        self.retries = 0
        self.endpoint_pool = (
            None  # opt-in request routing (as RubricJudge), assigned by the caller
        )

    def _post(self, cards: list[dict], attempt: int) -> str | None:
        ids = [c["id"] for c in cards]
        user = sf2.batch_user(
            self.query, cards, lambda c: card_text(self.render, c, self.max_chars)
        )
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens or llm_rank_max_tokens(len(cards)),
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": llm_rank_response_format(ids),
            "messages": [
                {"role": "system", "content": LLM_RANK_SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        t0 = time.monotonic()
        route = (
            self.endpoint_pool.lease() if self.endpoint_pool else nullcontext(self.base)
        )
        with route as endpoint:
            r = httpx.post(
                f"{endpoint}/v1/chat/completions", json=body, timeout=self.timeout
            )
            r.raise_for_status()
            j = r.json()
        self.calls.append(
            {
                "usage": j.get("usage"),
                "seconds": round(time.monotonic() - t0, 3),
                "endpoint": endpoint,
                "kind": "llm_rank",
                "n": len(cards),
                "attempt": attempt,
            }
        )
        try:
            choice = j["choices"][0]
            self.calls[-1]["finish_reason"] = choice.get("finish_reason")
            return choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

    def _batch(self, cards: list[dict]) -> dict:
        ids = [c["id"] for c in cards]
        for attempt in (0, 1):
            got = parse_llm_scores(self._post(cards, attempt), ids)
            if got is not None:
                return got
            if attempt == 0:
                self.retries += 1
        raise ValueError(
            f"LLM rank: malformed scores for card ids {ids} after one retry "
            f"(finish_reason={self.calls[-1].get('finish_reason') if self.calls else None})"
        )

    def __call__(self, cards: list[dict]) -> dict:
        if not cards:
            return {}
        ids = _unique_ids(cards)
        chunks = [cards[i : i + self.batch] for i in range(0, len(cards), self.batch)]
        scores: dict = {}
        with ThreadPoolExecutor(min(self.workers, len(chunks))) as ex:
            for got in ex.map(self._batch, chunks):
                scores.update(got)
        if set(scores) != set(ids) or len(scores) != len(ids):
            raise ValueError("LLM rank: scores do not cover the card ids exactly once")
        return {i: scores[i] for i in ids}
