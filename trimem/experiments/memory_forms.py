"""Memory-card forms of the pool-profiling round (space dimension ``memory_form``; user 2026-10-20; plan
docs/trimem_pool_profiling_plan_v0.1.0.md, memory-form addendum). Protocol ``VERSION``.

  raw        the dataset's existing card (``tau2.adapter.render_card`` / ``agentenv_task_adapter.render_card``),
             capped at the per-card budget ``llm_budget.per_card(top_k)`` (x 3 characters), unchanged.
  compact    rule-based, NO LLM: the task statement / customer's first message, the action or tool-call sequence
             (tool name + key arguments; consecutive repeats collapsed to ``xN``), the outcome and the
             SUCCESSFUL / FAILED label of the memory policy; observations, tool outputs, the agent's chit-chat and
             repeated text are dropped.
  distilled  an LLM card written ONCE per stored session at ingest (``memory_distill``, temperature 0, thinking off;
             sidecar JSONL keyed by session id), rendered deterministically here. Never per episode: the
             episode pays no distillation tokens or latency. A session without a valid distilled entry falls back
             to its compact card (counted in the receipt as ``distilled_missing``).

Both compact and distilled cards are hard-capped at ``CARD_TOKENS`` tokens of the Qwen3.8 tokenizer (``TOKENIZER``
= the served model's ``tokenizer.json``; ``CHARS_PER_TOKEN`` characters per token only when that file is
unavailable -- the protocol records which counter was used) and also at the raw per-card character cap, so a
form never injects more than raw would. Labels and the header are the raw form's (``memory_policy``), so the
leakage rules (own task / L1 isolation, outcome admissibility) apply to every form identically: the form only
changes how an already-admitted card is written.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path

from trimem.experiments import memory_policy as mp

VERSION = "memory-forms-v1"
FORMS = ("raw", "compact", "distilled")
CARD_TOKENS = 600  # hard cap per compact / distilled card (Qwen3.8 tokens)
CHARS_PER_TOKEN = 3  # fallback counter (the agent-env renderers' convention)
ARG_CHARS = 24  # a tool-call argument value longer than this is shown as its name only
GOAL_CHARS = 400
ROOT = Path(__file__).resolve().parents[2]
DISTILLED_DIR = ROOT / "data/memory_forms"
TOKENIZER_GLOB = "/localhome/hza214/hf-cache/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/tokenizer.json"


# ---------------------------------------------------------------- token counting


@lru_cache(maxsize=1)
def _tokenizer():
    path = os.environ.get("TRIMEM_QWEN_TOKENIZER") or next(
        iter(sorted(glob.glob(TOKENIZER_GLOB))), None
    )
    if not path or not Path(path).exists():
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    return Tokenizer.from_file(path)


def counter() -> str:
    """Identifier of the token counter in use (recorded in the protocol)."""
    return (
        "qwen3.8-tokenizer" if _tokenizer() is not None else f"chars/{CHARS_PER_TOKEN}"
    )


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is None:
        return -(-len(text) // CHARS_PER_TOKEN)
    return len(tok.encode(text, add_special_tokens=False).ids)


def clip_tokens(text: str, cap: int = CARD_TOKENS) -> str:
    """``text`` cut to at most ``cap`` tokens (ending in '...' when cut). Deterministic."""
    tok = _tokenizer()
    if tok is None:
        n = cap * CHARS_PER_TOKEN
        return text if len(text) <= n else text[: n - 3] + "..."
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) <= cap:
        return text
    return (
        text[: enc.offsets[cap - 2][1]] + "..."
    )  # cap - 2 tokens of text + '...' (<= 2 tokens)


def _clip(text, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 3] + "..."


def fit(text: str, cap_chars: int) -> str:
    """Token cap, then the raw per-card character cap."""
    text = clip_tokens(text)
    return text if len(text) <= cap_chars else text[: cap_chars - 3] + "..."


def collapse(items: Iterable[str]) -> list[str]:
    """Consecutive duplicates -> one item with ``xN``."""
    out: list[list] = []
    for it in items:
        if out and out[-1][0] == it:
            out[-1][1] += 1
        else:
            out.append([it, 1])
    return [a if n == 1 else f"{a} x{n}" for a, n in out]


# ---------------------------------------------------------------- compact (rule-based)

_CALL = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*\((.*)\)\s*$", re.S)
_USER_CALL = re.compile(r"\[user calls\]\s*([A-Za-z_]\w*)\s*\(")


def compact_call(action: str) -> str:
    """``name({"a": "x", "long": "..."})`` -> ``name(a=x, long)``: name + key arguments (short values only)."""
    m = _CALL.match(str(action or ""))
    if not m:
        return _clip(action, 80)
    name, raw = m.groups()
    try:
        args = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return f"{name}({_clip(raw, 60)})"
    if not isinstance(args, dict):
        return f"{name}({_clip(raw, 60)})"
    parts = []
    for k, v in args.items():
        sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        parts.append(f"{k}={sv}" if len(sv) <= ARG_CHARS else k)
    return f"{name}({', '.join(parts)})"


def tau2_steps(card: dict) -> list[str]:
    """Agent tool calls and the customer's device tool calls (telecom: ``[user calls] name(...)``), in order."""
    out = []
    for s in card["steps"]:
        if s.get("action"):
            out.append(compact_call(s["action"]))
        out += [
            f"customer: {n}()"
            for n in _USER_CALL.findall(str(s.get("observation") or ""))
        ]
    return collapse(out)


def compact_tau2(card: dict, i: int, cap_chars: int) -> str:
    from trimem.experiments.tau2.adapter import outcome_text

    lab = (
        f"{mp.label(bool(card['success']), outcome_text(card))}\n"
        if mp.labelled(card)
        else ""
    )
    out = (
        "resolved (reward 1)" if card["success"] else f"reward {card['final_score']:g}"
    )
    steps = tau2_steps(card)
    text = (
        f"[Memory {i}] {lab}Customer: {_clip(card.get('initial_observation') or card.get('description'), GOAL_CHARS)}\n"
        f"Outcome: {out} in {card['n_steps']} agent turns\n"
        f"Tool calls: {' -> '.join(steps) if steps else '(none)'}"
    )
    return fit(text, cap_chars)


def compact_agentenv(card: dict, i: int, cap_chars: int) -> str:
    from trimem.experiments.agentenv_task_adapter import outcome_text

    lab = (
        f"{mp.label(bool(card['success']), outcome_text(card))}\n"
        if mp.labelled(card)
        else ""
    )
    out = "success" if card["success"] else f"score {card['final_score']}"
    acts = collapse(
        " ".join(str(s["action"]).split()) for s in card["steps"] if s.get("action")
    )
    text = (
        f"[Memory {i}] {lab}Goal: {_clip(card['description'], GOAL_CHARS)}\n"
        f"Outcome: {out} in {card['n_steps']} steps\nActions: {' -> '.join(acts)}"
    )
    return fit(text, cap_chars)


# ---------------------------------------------------------------- distilled (offline sidecar)

TAU2_FIELDS = (
    ("request", "Request"),
    ("plan", "Plan"),
    ("pitfalls", "Pitfalls"),
    ("policy_facts", "Policy"),
    ("outcome", "Outcome"),
)
ENV_FIELDS = (
    ("key_steps", "Key steps"),
    ("where", "Where"),
    ("pitfalls", "Pitfalls"),
    ("lessons", "Lessons"),
    ("outcome", "Outcome"),
)


def distilled_path(dataset: str, version: str | None = None) -> Path:
    from trimem.experiments.memory_distill import VERSION as DV

    return DISTILLED_DIR / f"distilled_{dataset}_{version or DV}.jsonl"


class DistilledStore:
    """session id -> distilled JSON of one sidecar file (``memory_distill`` output; read-only)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.cards: dict[int, dict] = {}
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                self.cards[int(r["session_id"])] = r["distilled"]

    @classmethod
    def for_dataset(cls, dataset: str) -> DistilledStore:
        return _store(str(distilled_path(dataset)))

    def protocol(self) -> dict:
        return {
            "distilled_file": str(self.path),
            "distilled_sha256": self.sha256,
            "distilled_n": len(self.cards),
        }


@lru_cache(maxsize=8)
def _store(path: str) -> DistilledStore:
    return DistilledStore(Path(path))


def _field(v) -> str:
    if isinstance(v, list | tuple):
        items = [" ".join(str(x).split()) for x in v if str(x).strip()]
        return "; ".join(f"({k}) {x}" for k, x in enumerate(items, 1))
    return " ".join(str(v or "").split())


def render_distilled(d: dict, card: dict, fields, head: str) -> str:
    lines = [head]
    for key, title in fields:
        if key == "outcome" and not mp.labelled(card):
            continue  # unlabelled policy: no outcome text beyond the raw card's
        v = _field(d.get(key))
        if v:
            lines.append(f"{title}: {v}")
    return "\n".join(lines)


def distilled_tau2(
    card: dict, i: int, cap_chars: int, store: DistilledStore
) -> str | None:
    from trimem.experiments.tau2.adapter import outcome_text

    d = store.cards.get(int(card["id"]))
    if d is None:
        return None
    lab = (
        f"{mp.label(bool(card['success']), outcome_text(card))}\n"
        if mp.labelled(card)
        else ""
    )
    head = f"[Memory {i}] {lab}Customer: {_clip(card.get('initial_observation') or card.get('description'), GOAL_CHARS)}"
    return fit(render_distilled(d, card, TAU2_FIELDS, head), cap_chars)


def distilled_agentenv(
    card: dict, i: int, cap_chars: int, store: DistilledStore
) -> str | None:
    from trimem.experiments.agentenv_task_adapter import outcome_text

    d = store.cards.get(int(card["id"]))
    if d is None:
        return None
    lab = (
        f"{mp.label(bool(card['success']), outcome_text(card))}\n"
        if mp.labelled(card)
        else ""
    )
    head = f"[Memory {i}] {lab}Goal: {_clip(card['description'], GOAL_CHARS)}"
    return fit(render_distilled(d, card, ENV_FIELDS, head), cap_chars)


# ---------------------------------------------------------------- memory blocks


def _cards(
    cards, form, cap_chars, raw: Callable, compact: Callable, distilled: Callable, store
) -> tuple[list, dict]:
    if form not in FORMS:
        raise ValueError(f"unknown memory_form {form!r}")
    if form == "distilled" and store is None:
        raise ValueError("memory_form distilled needs the distilled sidecar store")
    out, missing = [], 0
    for i, c in enumerate(cards, 1):
        if form == "raw":
            out.append(raw(c, i, cap_chars))
        elif form == "compact":
            out.append(compact(c, i, cap_chars))
        else:
            t = distilled(c, i, cap_chars, store)
            if t is None:
                missing += 1
                t = compact(c, i, cap_chars)
            out.append(t)
    receipt = {
        "memory_form": form,
        "memory_forms_version": VERSION,
        "token_counter": counter(),
        "card_tokens_cap": None if form == "raw" else CARD_TOKENS,
    }
    if form == "distilled":
        receipt |= {"distilled_missing": missing, **store.protocol()}
    return out, receipt


def render_memory_tau2(
    cards: list[dict],
    form: str,
    cap_chars: int,
    policy: str,
    store: DistilledStore | None = None,
) -> tuple[str, dict]:
    """(``<memory>`` block, receipt). ``raw`` is byte-identical to ``tau2.adapter.render_memory``."""
    from trimem.experiments.tau2.adapter import (
        MEMORY_HEADER,
        render_card,
        render_memory,
    )

    if form == "raw":
        return render_memory(cards, cap_chars, policy), {
            "memory_form": "raw",
            "memory_forms_version": VERSION,
        }
    body, receipt = _cards(
        cards, form, cap_chars, render_card, compact_tau2, distilled_tau2, store
    )
    if not cards:
        return "", receipt
    return f"<memory>\n{mp.header(MEMORY_HEADER, policy)}\n\n" + "\n\n".join(
        body
    ) + "\n</memory>", receipt


def render_memory_agentenv(
    env: str,
    cards: list[dict],
    form: str,
    cap_chars: int,
    policy: str,
    store: DistilledStore | None = None,
) -> tuple[str, dict]:
    """(memory block, receipt). ``raw`` is byte-identical to ``agentenv_task_adapter.render_memory``."""
    from trimem.experiments.agentenv_task_adapter import (
        memory_header,
        render_card,
        render_memory,
    )

    if form == "raw":
        return render_memory(env, cards, cap_chars, policy), {
            "memory_form": "raw",
            "memory_forms_version": VERSION,
        }
    body, receipt = _cards(
        cards, form, cap_chars, render_card, compact_agentenv, distilled_agentenv, store
    )
    if not cards:
        return "", receipt
    return memory_header(env, policy) + "\n\n".join(body), receipt
