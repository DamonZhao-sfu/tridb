"""SemFilter v2 (protocol ``semfilter-v2``, user 2026-10-22, final spec): a faster SemFilter, OPT-IN only.

Gate: ``TRIMEM_SEMFILTER_V2=1`` (anything else / unset = the unchanged v1 SemFilter; nothing in this module runs).
Knobs (read only when the gate is on):

  ``TRIMEM_SEMFILTER_TOPN``         (default 32) similarity cutoff: only the top-N candidates of a SemFilter call
                                    (P1 / P1+Rank: the seed sessions; P2 / P2+Rank: the capped candidate pool),
                                    ordered by query cosine (desc, ties by id), are judged.
  ``TRIMEM_SEMFILTER_BATCH``        (default 8) K candidates per judge request (32 candidates -> 4 calls).
  ``TRIMEM_SEMFILTER_MAX_TOKENS``   (default 10 x K + 16) ``max_tokens`` of a batched call (single calls: 16).
  ``TRIMEM_SEMFILTER_GUIDED``       (default 1) guided decoding: a strict ``json_schema`` per request (batched:
                                    an object whose keys are exactly the K candidate ids, all required, no others,
                                    each value the score in tenths, enum 0..10; single: ``{"score": enum}``);
                                    0 = ``json_object`` mode (same prompt, parsed the same way).
  ``TRIMEM_SEMFILTER_JUDGE_INPUT``  raw | auto | compact | distilled (default raw = the v1 judge card
                                    ``semantic_rubric.env_card``); the others are an optional A/B flag only.
  ``TRIMEM_SEMFILTER_RUBRIC``       (default semantic-rubric-v3 = v2 + the CONFLICT clause) the rubric body;
                                    v1 / v2 selectable to isolate the conflict clause in an A/B.

1. Similarity cutoff. The similarity available BEFORE the filter is the query cosine every session carries (the
   ColBERT score of "+Rank" plans is computed after the filter, on the filtered pool). Candidates beyond the top-N
   are NOT PASSED: dropped from the post-filter set exactly like a "useful: false" (P1: the seed is not expanded;
   P2: the candidate is not ranked), but NOT written as semantic decisions -- ``receipt.semfilter_prefilter`` lists
   them (``dropped_ids``), so ``semantic_decisions`` keeps meaning "what the judge decided".
2. Score-only output. The LLM returns scores only (no useful / reason / transfer / evidence fields). ``useful`` is
   computed in code as ``score >= semfilter_threshold`` (the searched knob 0.4 / 0.6 / 0.8, as v1). Decisions keep
   the v1 schema (id / score / useful / reason / transfer / evidence_step / evidence_quote), text fields empty.
3. Batched calls. One request judges K candidates: system = a batch header + the rubric body of the version
   (anchors, "concrete step" rule, FAILED-ATTEMPT clause of ``semantic-rubric-v2``: byte-identical) + the batch
   answer block; user = NEW TASK, then the K ``PAST EPISODE <id>`` cards. Shared prefix order system -> NEW TASK ->
   cards, so vLLM prefix caching still covers the system + task across a retrieval's calls. The answer
   ``{"<id>": <tenths 0..10>, ...}`` (legacy 0..1-scale ``{"scores": [{id, score}]}`` / a bare list are still parsed) must name every
   candidate id with a numeric score; ids that come back missing / duplicated / non-numeric are re-judged ONE BY
   ONE with the score-only single prompt (``{"score": x}``, max_tokens 16; recorded as ``fallback_calls`` /
   ``batch_errors``). An unparseable single verdict scores 0 (= v1's "no evidence"), counted in ``parse_errors``.

Every v2 retrieval records ``protocol.semfilter`` and ``receipt.semfilter`` (``{protocol: 'semfilter-v2', top_n,
batch, judge_input, output: 'score-only', ...}`` + this retrieval's call / token counts) so measurements are
labelled.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import httpx

from trimem.experiments import semantic_rubric as sr

PROTOCOL = "semfilter-v2"
# semfilter-v2b (WebShop from round 1, user 2026-10-23): semfilter-v2 + the configuration's ``sem_fallback``
# "fill-topk" (``session_retrieval.retrieve``) + rubric semantic-rubric-v4-webshop. The label of a retrieval whose
# configuration carries the fallback; everything else of v2 (prefilter, batching, score-only guided) is unchanged.
PROTOCOL_V2B = "semfilter-v2b"
ENV_GATE = "TRIMEM_SEMFILTER_V2"
ENV_TOP_N = "TRIMEM_SEMFILTER_TOPN"
ENV_BATCH = "TRIMEM_SEMFILTER_BATCH"
ENV_JUDGE_INPUT = "TRIMEM_SEMFILTER_JUDGE_INPUT"
ENV_MAX_TOKENS = "TRIMEM_SEMFILTER_MAX_TOKENS"
ENV_RUBRIC = "TRIMEM_SEMFILTER_RUBRIC"
DEFAULT_RUBRIC = sr.VERSION_V3
DEFAULT_TOP_N = 32
DEFAULT_BATCH = 8
TOKENS_PER_SCORE = 16  # single-candidate call
BATCH_TOKENS_PER_ID, BATCH_TOKENS_BASE = 10, 16  # batched call: 10 x K + 16
ENV_GUIDED = "TRIMEM_SEMFILTER_GUIDED"
# Score levels of the guided schema = the score in TENTHS, integers 0..10 (6 -> 0.6). Measured 2026-10-22 on 8030
# (vLLM 0.23 / xgrammar): a FLOAT enum [0.0 .. 1.0] is emitted in 17-digit repr (0.1 -> 0.10000000000000001,
# ~20 tokens per id, overruns 10 x K + 16); a STRING enum ["0.0" .. "1.0"] degenerates into whitespace until
# max_tokens (the grammar allows free whitespace before a value). Integers are short and natural.
SCORE_LEVELS = tuple(range(11))
JUDGE_INPUTS = ("raw", "auto", "compact", "distilled")
JUDGE_CARD_CHARS = (
    12000  # the raw judge card's cap (semantic_rubric.env_card max_chars)
)
PREFILTER_ORDER = "query_cosine"
AGENTENV = ("alfworld", "sciworld", "webshop")

BATCH_HEADER = """You will see SEVERAL past episodes, each labelled PAST EPISODE <id>. Judge EACH one independently
with the rubric below, as if it were the only episode shown.

"""
SINGLE_ANSWER = """Answer JSON only, with the score alone, written in TENTHS as an integer 0..10 (e.g. 6 means 0.6):
{{"score": <0..10>}}"""
BATCH_ANSWER = """Answer JSON only: ONE object whose keys are exactly the past-episode ids (as strings), each mapped
to that episode's score written in TENTHS as an integer 0..10 (e.g. 6 means 0.6):
{{"<id>": <0..10>, "<id>": <0..10>, ...}}"""
EMPTY_TEXT = {"reason": "", "transfer": "", "evidence_step": None, "evidence_quote": ""}


def settings(env: Mapping[str, str] | None = None) -> dict | None:
    """The semfilter-v2 settings, or None when the gate is off (default). Invalid knob values raise."""
    env = os.environ if env is None else env
    if env.get(ENV_GATE, "").strip() != "1":
        return None

    def num(k, default):
        return int(env.get(k, "").strip() or default)

    top_n, batch = num(ENV_TOP_N, DEFAULT_TOP_N), num(ENV_BATCH, DEFAULT_BATCH)
    if top_n < 1:
        raise ValueError(f"{ENV_TOP_N} must be >= 1")
    if not 1 <= batch <= 32:
        raise ValueError(f"{ENV_BATCH} must be in [1, 32]")
    max_tokens = num(ENV_MAX_TOKENS, batch_max_tokens(batch))
    if not 8 <= max_tokens <= 2048:
        raise ValueError(f"{ENV_MAX_TOKENS} must be in [8, 2048]")
    judge_input = env.get(ENV_JUDGE_INPUT, "").strip() or "raw"
    if judge_input not in JUDGE_INPUTS:
        raise ValueError(f"{ENV_JUDGE_INPUT} must be one of {JUDGE_INPUTS}")
    guided = env.get(ENV_GUIDED, "").strip() or "1"
    if guided not in ("0", "1"):
        raise ValueError(f"{ENV_GUIDED} must be 0 or 1")
    rubric = env.get(ENV_RUBRIC, "").strip() or DEFAULT_RUBRIC
    if rubric not in sr.SYSTEMS:
        raise ValueError(f"{ENV_RUBRIC} must be one of {sorted(sr.SYSTEMS)}")
    return {
        "top_n": top_n,
        "batch": batch,
        "max_tokens": max_tokens,
        "judge_input": judge_input,
        "rubric": rubric,
        "guided": guided == "1",
    }


def batch_max_tokens(k: int) -> int:
    return BATCH_TOKENS_PER_ID * k + BATCH_TOKENS_BASE


def score_schema(ids: list | None = None) -> dict:
    """Strict JSON schema for guided decoding (vLLM / xgrammar): ``ids`` -> an object with exactly those keys
    (strings), all required, no others, each an 11-level score enum; None -> ``{"score": enum}``."""
    level = {"enum": list(SCORE_LEVELS)}
    keys = ["score"] if ids is None else [str(i) for i in ids]
    return {
        "type": "object",
        "properties": {k: level for k in keys},
        "required": keys,
        "additionalProperties": False,
    }


def response_format(ids: list | None, guided: bool) -> dict:
    if not guided:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semfilter_scores" if ids is not None else "semfilter_score",
            "schema": score_schema(ids),
            "strict": True,
        },
    }


# ---------------------------------------------------------------- 1. similarity cutoff


def prefilter(sessions: list[dict], top_n: int) -> tuple[list[dict], list[dict]]:
    """(judged, dropped): the ``top_n`` sessions by query cosine (desc, ties by id) are judged, in their ORIGINAL
    order (so the downstream filtered order is the v1 order); the rest are dropped (= not passed). Pure."""
    if len(sessions) <= top_n:
        return list(sessions), []
    keep = {
        s["id"]
        for s in sorted(sessions, key=lambda s: (-float(s["cosine"]), s["id"]))[:top_n]
    }
    return [s for s in sessions if s["id"] in keep], [
        s for s in sessions if s["id"] not in keep
    ]


# ---------------------------------------------------------------- 2./3. prompts and parsing


def _rubric_body(version: str) -> str:
    full = sr.system_for(version)
    head, sep, _ = full.rpartition("Answer JSON only:")
    if not sep:
        raise ValueError(f"rubric {version!r} has no answer block")
    return head


def single_system(version: str = sr.VERSION) -> str:
    """Score-only prompt template for ONE candidate (``{knowledge}`` placeholder kept): the version's rubric
    body, byte-identical, with the answer block replaced."""
    return _rubric_body(version) + SINGLE_ANSWER


def batch_system(version: str = sr.VERSION) -> str:
    """Score-only prompt template for K candidates: batch header + the rubric body + the batch answer block."""
    return BATCH_HEADER + _rubric_body(version) + BATCH_ANSWER


def batch_user(target: str, cards: list[dict], render) -> str:
    """NEW TASK first (shared prefix), then the K labelled cards."""
    parts = [f"NEW TASK:\n{target}"]
    parts += [f"PAST EPISODE {c['id']}:\n{render(c)}" for c in cards]
    return "\n\n".join(parts)


def _score(v) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else min(1.0, max(0.0, x))


def _tenths(v) -> float | None:
    """A score of the tenths answer (``SCORE_LEVELS``): an integer 0..10 -> /10; a non-integer <= 1 (a model in
    json_object mode answering on the 0..1 scale) is taken as is; clamped to [0, 1]."""
    x = _score_raw(v)
    if x is None:
        return None
    if x != int(x) and x <= 1:
        return min(1.0, max(0.0, x))
    return min(1.0, max(0.0, x / 10))


def _score_raw(v) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def parse_single(content: str | None) -> float | None:
    """Score of ``{"score": <tenths 0..10>}`` (``_tenths``); None when it does not parse."""
    try:
        d = json.loads(content or "")
    except (TypeError, ValueError):
        return None
    return _tenths(d.get("score")) if isinstance(d, dict) else None


def parse_batch(content: str | None, ids: list) -> tuple[dict, list]:
    """({id: score} of the candidates answered exactly once with a numeric score, [ids to re-judge]). The answer
    is ``{"<id>": score, ...}`` (guided schema); legacy shapes ``{"scores": [{id, score}]}`` (or an object with
    exactly one list value) and a bare list are still accepted. Ids compare as strings."""
    want = {str(i): i for i in ids}
    try:
        d = json.loads(content or "")
    except (TypeError, ValueError):
        return {}, list(ids)
    if isinstance(d, dict) and not any(isinstance(v, list) for v in d.values()):
        got = {}
        for k, v in d.items():
            sc = _tenths(v)
            if str(k).strip() in want and sc is not None:
                got[want[str(k).strip()]] = sc
        return got, [i for i in ids if i not in got]
    if isinstance(d, dict):
        lists = [v for v in d.values() if isinstance(v, list)]
        d = (
            d["scores"]
            if isinstance(d.get("scores"), list)
            else (lists[0] if len(lists) == 1 else None)
        )
    if not isinstance(d, list):
        return {}, list(ids)
    seen: dict[str, list] = {}
    for e in d:
        if isinstance(e, dict) and str(e.get("id")).strip() in want:
            seen.setdefault(str(e.get("id")).strip(), []).append(_score(e.get("score")))
    got = {want[k]: v[0] for k, v in seen.items() if len(v) == 1 and v[0] is not None}
    return got, [i for i in ids if i not in got]


# ---------------------------------------------------------------- judge input


def distilled_available(dataset: str) -> bool:
    from trimem.experiments import memory_forms as mf

    return mf.distilled_path(dataset).exists()


def resolve_judge_input(dataset: str, requested: str) -> str:
    if requested == "auto":
        return "distilled" if distilled_available(dataset) else "compact"
    if requested == "distilled" and not distilled_available(dataset):
        raise ValueError(f"{ENV_JUDGE_INPUT}=distilled but no sidecar for {dataset!r}")
    return requested


class JudgeView:
    """Card -> judge text: ``raw`` = ``semantic_rubric.env_card`` (v1, default); ``compact`` / ``distilled`` =
    the ``memory_forms`` cards (optional A/B flag; a session missing from the sidecar falls back to compact,
    counted). Agent-env datasets only (ALFWorld / ScienceWorld / WebShop)."""

    def __init__(self, dataset: str, judge_input: str, store=None):
        if judge_input not in ("raw", "compact", "distilled"):
            raise ValueError(
                "judge_input must be resolved to raw / compact / distilled"
            )
        if dataset not in AGENTENV:
            raise ValueError(
                f"semfilter-v2 judge view: unsupported dataset {dataset!r}"
            )
        self.dataset, self.judge_input = dataset, judge_input
        self.store = None
        if judge_input == "distilled":
            from trimem.experiments import memory_forms as mf

            self.store = store or mf.DistilledStore.for_dataset(dataset)
        self.rendered = 0
        self.distilled_missing = 0

    def __call__(self, card: dict) -> str:
        self.rendered += 1
        if self.judge_input == "raw":
            return sr.env_card(card)
        from trimem.experiments import memory_forms as mf

        if self.store is not None:
            t = mf.distilled_agentenv(card, 0, JUDGE_CARD_CHARS, self.store)
            if t is not None:
                return t.removeprefix("[Memory 0] ")
            self.distilled_missing += 1
        return mf.compact_agentenv(card, 0, JUDGE_CARD_CHARS).removeprefix(
            "[Memory 0] "
        )

    def protocol(self) -> dict:
        out = {"judge_input": self.judge_input}
        if self.store is not None:
            out |= self.store.protocol()
        return out


# ---------------------------------------------------------------- the judge


class BatchScoreSemFilter(sr.RubricSemFilter):
    """``RubricSemFilter`` with batched score-only verdicts. Same model, temperature 0, thinking off, JSON mode,
    threshold semantics, endpoint routing (``endpoint_pool``) and per-call usage log (``calls``, + ``kind`` / ``n``).
    The batches of one call run on ``workers`` threads."""

    def __init__(
        self,
        *args,
        batch: int = DEFAULT_BATCH,
        max_tokens: int | None = None,
        guided: bool = True,
        **kw,
    ):
        super().__init__(*args, max_tokens=max_tokens or batch_max_tokens(batch), **kw)
        self.batch = batch
        self.guided = guided
        v, ds = self.version, self.dataset
        self.system = sr.system_text(v, ds, batch_system(v))
        self.single = sr.system_text(v, ds, single_system(v))
        self.parse_errors = 0
        self.batch_errors = 0
        self.fallback_calls = 0

    def _post(
        self,
        system: str,
        user: str,
        max_tokens: int,
        kind: str,
        n: int,
        ids: list | None = None,
    ) -> str | None:
        body = {
            "model": sr.MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": response_format(ids, self.guided),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        t0 = time.monotonic()
        route = (
            self.endpoint_pool.lease() if self.endpoint_pool else nullcontext(self.base)
        )
        with route as endpoint:
            r = httpx.post(f"{endpoint}/v1/chat/completions", json=body, timeout=600)
            r.raise_for_status()
            j = r.json()
        self.calls.append(
            {
                "usage": j.get("usage"),
                "seconds": round(time.monotonic() - t0, 3),
                "endpoint": endpoint,
                "kind": kind,
                "n": n,
            }
        )
        try:
            return j["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

    def _one_score(self, card: dict) -> float:
        user = batch_user(self.target, [card], self.render)
        s = parse_single(self._post(self.single, user, TOKENS_PER_SCORE, "single", 1))
        if s is None:
            self.parse_errors += 1
            return 0.0
        return s

    def _batch(self, cards: list[dict]) -> dict:
        if len(cards) == 1:
            return {cards[0]["id"]: self._one_score(cards[0])}
        user = batch_user(self.target, cards, self.render)
        ids = [c["id"] for c in cards]
        content = self._post(
            self.system, user, self.max_tokens, "batch", len(cards), ids
        )
        got, redo = parse_batch(content, ids)
        if redo:
            self.batch_errors += 1
            for c in cards:
                if c["id"] in redo:
                    self.fallback_calls += 1
                    got[c["id"]] = self._one_score(c)
        return got

    def judge(self, cards: list[dict]) -> list[dict]:
        todo, seen = [], set()
        for c in cards:
            if c["id"] not in self.judged and c["id"] not in seen:
                todo.append(c)
                seen.add(c["id"])
        chunks = [todo[i : i + self.batch] for i in range(0, len(todo), self.batch)]
        with ThreadPoolExecutor(max(1, self.workers)) as ex:
            for scores in ex.map(self._batch, chunks):
                for cid, s in scores.items():
                    self.judged[cid] = {"id": cid, "score": s}
        return [self.judged[c["id"]] for c in cards]

    def __call__(self, cards: list[dict]) -> list[dict]:
        return [
            {
                "id": d["id"],
                "score": d["score"],
                "useful": d["score"] >= self.threshold,
                **EMPTY_TEXT,
            }
            for d in self.judge(cards)
        ]


# ---------------------------------------------------------------- wiring helpers (agentgym_session_eval)


def build(
    dataset: str,
    target: str,
    cfg: dict,
    *,
    base: str,
    version: str,
    workers: int,
    threshold: float,
    store=None,
    sem_fallback: str = "none",
) -> tuple[BatchScoreSemFilter, JudgeView, dict]:
    """(judge, its view, protocol block) for one retrieval under settings ``cfg`` (from ``settings()``). The judge
    uses ``cfg["rubric"]`` (default v3); ``version`` = the memory policy's rubric, recorded as ``policy_rubric``."""
    rubric = cfg.get("rubric") or version
    resolved = resolve_judge_input(dataset, cfg["judge_input"])
    view = JudgeView(dataset, resolved, store)
    sem = BatchScoreSemFilter(
        dataset,
        target,
        view,
        base=base,
        version=rubric,
        workers=workers,
        threshold=threshold,
        batch=cfg["batch"],
        max_tokens=cfg["max_tokens"],
        guided=cfg.get("guided", True),
    )
    proto = {
        "protocol": PROTOCOL if sem_fallback == "none" else PROTOCOL_V2B,
        **({"sem_fallback": sem_fallback} if sem_fallback != "none" else {}),
        "rubric": rubric,
        "policy_rubric": version,
        "top_n": cfg["top_n"],
        "prefilter_order": PREFILTER_ORDER,
        "prefilter_rest": "not passed (dropped; not a semantic decision)",
        "batch": cfg["batch"],
        "max_tokens": cfg["max_tokens"],
        "output": "score-only",
        "guided": (
            "json_schema strict (ids -> 11-level enum)"
            if cfg.get("guided", True)
            else "json_object"
        ),
        "useful_rule": "score >= semfilter_threshold",
        "fallback": "per-candidate score-only call for ids missing / duplicated / non-numeric in a batch answer",
        "judge_input": resolved,
        "judge_input_requested": cfg["judge_input"],
        **{k: v for k, v in view.protocol().items() if k != "judge_input"},
    }
    return sem, view, proto


def call_summary(calls: list[dict]) -> dict:
    """Judge calls (batched / single) and their prompt / completion / total tokens and summed seconds."""

    def tok(k):
        return sum(int((c.get("usage") or {}).get(k) or 0) for c in calls)

    return {
        "judge_calls": len(calls),
        "judge_batch_calls": sum(c.get("kind") == "batch" for c in calls),
        "judge_single_calls": sum(c.get("kind") == "single" for c in calls),
        "judge_prompt_tokens": tok("prompt_tokens"),
        "judge_completion_tokens": tok("completion_tokens"),
        "judge_total_tokens": tok("total_tokens"),
        "judge_seconds": round(sum(float(c.get("seconds") or 0) for c in calls), 3),
    }


def receipt_block(
    proto: dict, sem: BatchScoreSemFilter, view: JudgeView, receipt: dict
) -> dict:
    """``receipt.semfilter``: the protocol block + this retrieval's calls, tokens, batch errors / fallbacks, parse
    errors, sidecar misses and cutoff sizes (the dropped ids stay in ``receipt.semfilter_prefilter``)."""
    pre = receipt.get("semfilter_prefilter") or {}
    fb = receipt.get("semfilter_fallback")
    extra = (
        {
            # semfilter-v2b: what passed the judge vs what the fill-topk fallback added, per retrieval
            "passed": sum(
                bool(d.get("useful")) for d in receipt.get("semantic_decisions") or []
            ),
            "fallback_stage": fb["stage"],
            "fallback_needed": fb["needed"],
            "fallback_filled": len(fb["filled_ids"]),
        }
        if fb is not None
        else {}
    )
    return {
        **proto,
        **call_summary(sem.calls),
        "judged": len(sem.judged),
        "batch_errors": sem.batch_errors,
        "fallback_calls": sem.fallback_calls,
        "parse_errors": sem.parse_errors,
        "distilled_missing": view.distilled_missing,
        "prefilter_candidates": pre.get("candidates"),
        "prefilter_judged": len(pre.get("judged_ids") or []),
        "prefilter_dropped": len(pre.get("dropped_ids") or []),
        **extra,
    }
