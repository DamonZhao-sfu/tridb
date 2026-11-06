"""SemFilter v2c (TRACE v0.7.1, user 2026-11-05): cheaper SemFilter, applied by runtime patching only.

Opt-in gate ``TRIMEM_SEMFILTER_V2C=1`` (absent = the stock ``semfilter_v2`` path, byte-identical behaviour). Three
changes on top of ``semfilter-v2`` (whose module and every caller stay unchanged on disk):

1. TOPN 16: the stock knob ``TRIMEM_SEMFILTER_TOPN`` = 16 (was 32) -> the query-cosine prefilter judges at most 16
   candidates of a P1 (seeds) / P2 (pool) SemFilter call. Set by the v0.7.1 manifest environment and recorded in
   ``fixed.semfilter`` (so the execution fingerprint changes).
2. Compact judge card ``judge-card-compact-v1`` (``compact_card``): SUCCESSFUL / FAILED ATTEMPT tag with the
   outcome (without the label's advice sentence, which the rubric's FAILED-ATTEMPT clause already states), Goal,
   Outcome, and every step's action with its observation cut to ``OBS_CHARS`` (60); no-op observations ("Nothing happens.",
   empty, "No known action matches that input.", or the same as the previous step's) are dropped; runs of the same
   action with a no-op result are collapsed ("5-7. look (x3)"); hard cap ``CARD_CAP`` (2,000) chars, keeping the
   head and the tail steps with a "... N steps omitted ..." marker. Requested by
   ``TRIMEM_SEMFILTER_JUDGE_INPUT=compact-v2c`` -- a value the stock ``semfilter_v2.settings`` REJECTS, so a v0.7.1
   run launched without this module fails closed on its first SemFilter episode.
3. Periodic reuse within ONE episode (``TRIMEM_SEMFILTER_V2C_REUSE=episode``): a candidate session judged by an
   earlier retrieval of the same episode (initial or periodic) reuses its earlier score; only unseen candidates
   are judged. Key = (episode, candidate session id, rubric, judge input); the cached value is the
   threshold-independent score (``useful`` is recomputed with each retrieval's threshold). The episode is the
   worker process (one item per process, asserted by ``tools.trace_v071_episode_worker``). Each retrieval's
   ``receipt.semfilter`` adds ``cache_hits``, ``judged_new`` and ``tokens_saved_est`` (estimate: hits x this episode's
   mean judge prompt tokens per judged candidate); ``<episode>/attempt/semfilter_v2c.jsonl`` holds one line per retrieval.
"""

from __future__ import annotations

import json
import os
import re

PROTOCOL = "semfilter-v2c"
PROTOCOL_V2CB = "semfilter-v2c-b"  # + the configuration's sem_fallback fill-topk (as semfilter-v2b)
CARD_VERSION = "judge-card-compact-v1"
JUDGE_INPUT = "compact-v2c"
ENV_GATE = "TRIMEM_SEMFILTER_V2C"
ENV_REUSE = "TRIMEM_SEMFILTER_V2C_REUSE"
TOP_N = 16
OBS_CHARS = 60
CARD_CAP = 2000
NOOP = re.compile(r"^(nothing happens\.?|no known action matches that input\.?|ok\.?|)$", re.IGNORECASE)
BLOCK = {"protocol": PROTOCOL, "top_n": TOP_N, "card": CARD_VERSION, "judge_input": JUDGE_INPUT,
         "obs_chars": OBS_CHARS, "card_cap": CARD_CAP, "reuse": "episode",
         "reuse_key": "(episode, candidate session id, rubric, judge_input) -> threshold-independent score"}
ENVIRONMENT = {"TRIMEM_SEMFILTER_V2": "1", "TRIMEM_SEMFILTER_TOPN": str(TOP_N),
               "TRIMEM_SEMFILTER_JUDGE_INPUT": JUDGE_INPUT, ENV_GATE: "1", ENV_REUSE: "episode"}


def _obs(text) -> str:
    return " ".join(str(text or "").split())


def compact_card(card: dict) -> str:
    """The ``judge-card-compact-v1`` text of one hydrated card (agent-env datasets)."""
    from trimem.experiments import memory_policy as mp
    from trimem.experiments.agentenv_task_adapter import outcome_text

    out = "success" if card["success"] else f"score {card['final_score']}"
    head = [f"Goal: {card['description']}", f"Outcome: {out} in {card['n_steps']} steps"]
    if mp.labelled(card):  # the tag alone: the rubric's FAILED-ATTEMPT clause already says how to treat it
        tag = mp.SUCCESS_TAG if card["success"] else mp.FAILED_TAG
        head.insert(0, f"{tag} (outcome: {outcome_text(card)})")
    steps, prev = [], None
    for s in card["steps"]:
        action, obs = " ".join(str(s.get("action") or "").split()), _obs(s.get("observation"))
        noop = bool(NOOP.match(obs)) or obs == prev
        prev = obs
        n = s["ord"] + 1
        if steps and noop and steps[-1]["noop"] and steps[-1]["action"] == action:
            steps[-1]["last"], steps[-1]["count"] = n, steps[-1]["count"] + 1
            continue
        steps.append({"first": n, "last": n, "count": 1, "action": action, "noop": noop,
                      "obs": "" if noop else obs[:OBS_CHARS]})
    lines = []
    for st in steps:
        num = f"{st['first']}" if st["count"] == 1 else f"{st['first']}-{st['last']}"
        rep = f" (x{st['count']})" if st["count"] > 1 else ""
        lines.append(f"{num}. {st['action']}{rep}" + (f" -> {st['obs']}" if st["obs"] else ""))
    return _fit(head, lines)


def _fit(head: list, lines: list) -> str:
    text = "\n".join(head + lines)
    if len(text) <= CARD_CAP:
        return text
    budget = CARD_CAP - len("\n".join(head)) - 40
    keep_head, keep_tail, used = [], [], 0
    for i in range(len(lines)):  # alternate head / tail so both ends survive
        line = lines[i // 2] if i % 2 == 0 else lines[len(lines) - 1 - i // 2]
        if used + len(line) + 1 > budget or len(keep_head) + len(keep_tail) >= len(lines):
            break
        (keep_head if i % 2 == 0 else keep_tail).append(line)
        used += len(line) + 1
    omitted = len(lines) - len(keep_head) - len(keep_tail)
    text = "\n".join(head + keep_head + [f"... {omitted} steps omitted ..."] + keep_tail[::-1])
    return text[:CARD_CAP]


class CompactJudgeView:
    """Drop-in for ``semfilter_v2.JudgeView`` (render callable + ``protocol()`` + counters)."""

    def __init__(self):
        self.judge_input, self.rendered, self.distilled_missing = JUDGE_INPUT, 0, 0

    def __call__(self, card: dict) -> str:
        self.rendered += 1
        return compact_card(card)

    def protocol(self) -> dict:
        return {"judge_input": JUDGE_INPUT, "card": CARD_VERSION, "obs_chars": OBS_CHARS, "card_cap": CARD_CAP}


class EpisodeCache:
    """(candidate id, rubric, judge_input) -> score for the current episode (= this worker process)."""

    def __init__(self):
        self.scores: dict = {}
        self.prompt_tokens = 0
        self.judged = 0

    def per_candidate_tokens(self) -> float:
        return self.prompt_tokens / self.judged if self.judged else 0.0


CACHE = EpisodeCache()
_STOCK: dict = {}
LOG = {"path": None}  # one JSON line per retrieval (set by tools.trace_v071_episode_worker)


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(ENV_GATE, "").strip() == "1"


def _settings(env=None):
    from trimem.experiments import semfilter_v2 as sf

    env = dict(os.environ if env is None else env)
    if env.get(sf.ENV_JUDGE_INPUT, "").strip() == JUDGE_INPUT:
        env[sf.ENV_JUDGE_INPUT] = "raw"  # the stock view is replaced by CompactJudgeView in _build
        cfg = _STOCK["settings"](env)
        return None if cfg is None else {**cfg, "judge_input": JUDGE_INPUT}
    return _STOCK["settings"](env)


def _wrap_judge(sem, reuse: bool) -> None:
    stock_judge = sem.judge
    key = lambda c: (c["id"], sem.version, JUDGE_INPUT)
    sem.v2c = {"cache_hits": 0, "judged_new": 0, "reuse": "episode" if reuse else "off"}

    def judge(cards):
        if reuse:
            for c in cards:
                if c["id"] not in sem.judged and key(c) in CACHE.scores:
                    sem.judged[c["id"]] = {"id": c["id"], "score": CACHE.scores[key(c)]}
                    sem.v2c["cache_hits"] += 1
        before, calls_before = len(sem.judged), len(sem.calls)
        out = stock_judge(cards)
        new = len(sem.judged) - before
        sem.v2c["judged_new"] += new
        CACHE.judged += new
        CACHE.prompt_tokens += sum(int((c.get("usage") or {}).get("prompt_tokens") or 0)
                                   for c in sem.calls[calls_before:])
        for c in cards:
            CACHE.scores[key(c)] = sem.judged[c["id"]]["score"]
        return out

    sem.judge = judge


def _build(dataset, target, cfg, **kw):
    if cfg.get("judge_input") != JUDGE_INPUT:
        return _STOCK["build"](dataset, target, cfg, **kw)
    sem, _, proto = _STOCK["build"](dataset, target, {**cfg, "judge_input": "raw"}, **kw)
    view = CompactJudgeView()
    sem.render = view
    _wrap_judge(sem, os.environ.get(ENV_REUSE, "").strip() == "episode")
    proto = {**proto, **view.protocol(), "judge_input_requested": JUDGE_INPUT, "v2c": dict(BLOCK),
             "protocol": PROTOCOL if kw.get("sem_fallback", "none") == "none" else PROTOCOL_V2CB}
    return sem, view, proto


def _receipt_block(proto, sem, view, receipt):
    out = _STOCK["receipt_block"](proto, sem, view, receipt)
    stats: dict | None = getattr(sem, "v2c", None)
    if stats is not None:
        out.update(cache_hits=stats["cache_hits"], judged_new=stats["judged_new"], reuse=stats["reuse"],
                   tokens_saved_est=round(stats["cache_hits"] * CACHE.per_candidate_tokens(), 1))
        if LOG["path"]:
            line = {k: out.get(k) for k in ("protocol", "top_n", "cache_hits", "judged_new", "tokens_saved_est",
                                            "judge_prompt_tokens", "judge_total_tokens", "judge_seconds",
                                            "prefilter_candidates", "prefilter_judged")}
            with open(LOG["path"], "a") as f:
                f.write(json.dumps({"retrieval_id": receipt.get("retrieval_id"), **line}) + "\n")
    return out


def install() -> bool:
    """Patch ``semfilter_v2.settings`` / ``build`` / ``receipt_block`` (idempotent). No-op without the gate."""
    from trimem.experiments import semfilter_v2 as sf

    if not enabled() or _STOCK:
        return bool(_STOCK)
    _STOCK.update(settings=sf.settings, build=sf.build, receipt_block=sf.receipt_block)
    sf.settings, sf.build, sf.receipt_block = _settings, _build, _receipt_block
    return True


def uninstall() -> None:
    from trimem.experiments import semfilter_v2 as sf

    if _STOCK:
        sf.settings, sf.build, sf.receipt_block = _STOCK["settings"], _STOCK["build"], _STOCK["receipt_block"]
        _STOCK.clear()
    CACHE.__init__()
    LOG["path"] = None
