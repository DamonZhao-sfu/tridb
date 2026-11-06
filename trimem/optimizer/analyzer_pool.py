"""LLM analyst candidate channel for the POOL-profiling drivers (user-approved 2026-10-20).

The RAG-PE analyst channel of ``analyzer_session`` (R^LLM: the LLM reads the profiled configurations and
proposes edits to an evaluated anchor, which only ENTER the qLogNEHVI candidate pool; arbitration is unchanged),
adapted to the pool drivers (``bo_pool_driver`` ALFWorld / ScienceWorld, ``tau2.pool_profile`` τ²):

* SPACE-AWARE: the cards list only the SEARCHED knobs of the driver's space (``space.domain(plan)``); the fixed
  knobs (``space.FIXED``: edges, memory_form since -v5) are shown as fixed and any candidate that names a fixed
  knob, an unknown knob or a value outside its grid is REJECTED (never snapped / coerced into the space).
* TRAIN profiling evaluations only: per configuration the optimizer's quality objective (mean +- SE), mean tokens,
  mean latency (with SEs) and its staged status (n evaluated of the full panel; pruned or not). No test data,
  no trajectories, no task text.
* local Qwen (vLLM 8030, then 8031) at temperature 0, ``llm_budget.OUTPUT_TOKENS`` output cap, thinking off.
* NEVER raises: a failed call or an unparseable reply returns no candidates (the round falls back to the
  other channels: Sobol breadth, crossover, DLS) and is logged.
* every call is logged under ``<run dir>/analyst/round_NN.json`` (prompt, raw response, parsed candidates,
  rejects, status, usage).

Ordinary rounds use ``analyzer_session.SYSTEM_ORDINARY``-style prompts (label ``agent``); stalled (forced)
rounds with a gap rectangle use the gap prompt (label ``agent-gap``), exactly as ``bo_agentenv_driver``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trimem.optimizer import analyzer_session as az

PROTOCOL = "analyst-pool-v1"  # bump when prompts / parsing / client change
API_BASES = ("http://127.0.0.1:8030/v1", "http://127.0.0.1:8031/v1")
TEMPERATURE = 0.0
TIMEOUT_S = 900
N_PROPOSE = 4
OFFLINE_ENV = "TRIMEM_ANALYST_OFFLINE"

SYSTEM_ORDINARY = (
    "You are an expert at tuning the trajectory-memory retrieval of an LLM agent (before each episode, past "
    "episodes of OTHER tasks are retrieved from a graph+vector store and injected into the agent's prompt). You "
    "advise a MULTI-OBJECTIVE Bayesian optimizer whose true target is the HYPERVOLUME of the (quality, tokens, "
    "latency) Pareto front. A configuration that maximizes quality but costs far more does NOT necessarily "
    "improve the hypervolume. Read the past evaluations and the current front, diagnose where configurations fall "
    "short, then propose a DIVERSE set of NEW configurations that EXTEND the front: some pushing quality, some "
    "trading a little quality for much lower cost, some filling gaps. Do NOT assume generic retrieval folklore "
    "(e.g. 'more memory always helps', 'an LLM filter or a re-ranker always helps'); let the evaluations show what "
    "drives quality HERE. Quality is noisy: use the standard errors. Evaluations marked PARTIAL were measured on a "
    "subset of the panel (staged evaluation) and are noisier; PRUNED ones were stopped because a fully evaluated "
    "configuration dominated them."
)
SYSTEM_GAP = az.SYSTEM_GAP


@dataclass
class PoolEval(az.Eval):
    """``analyzer_session.Eval`` + the staged status of the configuration."""

    n: int = 0
    n_full: int = 0
    pruned: bool = False


# ------------------------------------------------------------------ space-aware cards


def searched(space, plan: str) -> dict[str, tuple]:
    """Searched knobs of ``plan`` (fixed knobs never appear here)."""
    return dict(space.domain(plan))


def all_searched(space) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for plan in space.PLANS:
        for k, v in searched(space, plan).items():
            out.setdefault(k, v)
    return out


def knobs_text(space, cfg: dict[str, Any]) -> str:
    """Searched knobs of an evaluated configuration; a proposal-fixed knob (``PROPOSE_FIXED``, e.g. τ² sim_gate
    from round 1) is shown only when an earlier configuration measured it at another value (marked legacy)."""
    cfg = space.validate(cfg)
    legacy = [
        f"{k}={cfg[k]}(legacy, now FIXED {v})"
        for k, v in (getattr(space, "PROPOSE_FIXED", {}) or {}).items()
        if cfg["plan"] != "NoMemory" and cfg.get(k) != v
    ]
    return " ".join(
        [f"{k}={cfg[k]}" for k in ("plan", *searched(space, cfg["plan"]))] + legacy
    )


def status_text(e: PoolEval) -> str:
    if e.n_full and e.n < e.n_full:
        return f" [PARTIAL n={e.n}/{e.n_full}{', PRUNED' if e.pruned else ''}]"
    return f" [n={e.n}]" if e.n else ""


def eval_line(space, e: PoolEval) -> str:
    return (
        f"{e.eid} {knobs_text(space, e.cfg)} -> quality={e.gain:+.3f}(se {e.gain_se:.3f}) "
        f"tok={e.tokens:.0f}(ln-se {e.tok_se:.3f}) lat_s={e.latency_ms / 1000:.1f}(ln-se {e.lat_se:.3f})"
        + status_text(e)
    )


def space_card(space) -> str:
    lines = [
        f"  plan: {list(space.PLANS)}  (display names: "
        + ", ".join(f"{p}={space.DISPLAY[p]}" for p in space.PLANS)
        + ")"
    ]
    union = all_searched(space)
    for k, v in union.items():
        only = [p for p in space.PLANS if k in searched(space, p)]
        scope = (
            "" if len(only) == len(space.PLANS) else f"  (active only for plans {only})"
        )
        lines.append(f"  {k}: {list(v)}{scope}")
    fixed = getattr(space, "FIXED", {})
    if fixed:
        lines.append(
            "FIXED (NOT searchable; never output these knobs): "
            + ", ".join(f"{k}={v}" for k, v in fixed.items())
        )
    lines.append(
        "NoMemory is the baseline, never a candidate. Use ONLY knob names and values listed above."
    )
    lines.append("KNOB SEMANTICS:")
    lines += [
        f"  {k}: {v}"
        + (PLAN_DOC_P1RANK if k == "plan" and "P1+Rank" in space.PLANS else "")
        + (PLAN_DOC_P1 if k == "plan" and "P1" in space.PLANS else "")
        for k, v in KNOB_DOC.items()
        if k == "plan" or k in union
    ]
    return "\n".join(lines)


# plan-p1rank-v1 (user 2026-10-20): appended to the plan semantics only for spaces that contain P1+Rank
PLAN_DOC_P1RANK = (
    ". P1+Rank (shown as V+SemFilter+G+Rank) = the LLM judge filters the vector SEED sessions BEFORE the "
    "similarity hops (their expanded neighbours are not judged), then ColBERT ranks the expanded pool; fewer "
    "judge calls than P2+Rank when seed_count is small"
)


# plan-p1-v1 (user 2026-10-22): only for spaces / rounds where P1 is proposable
PLAN_DOC_P1 = (
    ". P1 (shown as V+SemFilter+G) = the LLM judge filters the vector SEED sessions, then the similarity hops "
    "expand them (neighbours not judged); no ColBERT rank (top_k by query cosine)"
)


KNOB_DOC = {
    **{k: v for k, v in az.KNOB_DOC.items() if k != "edges"},
    "sim_gate": "inject nothing when the best pooled session's query cosine is below this value (none = no gate)",
    "outcome_policy": (
        "which stored outcomes are retrievable: success_only (successful sessions only, unlabelled cards); "
        "success_failure (both, cards labelled SUCCESS / FAILED); success_failure_capped (both, at most "
        "floor(top_k / 2) FAILED cards among the injected top_k)"
    ),
    "semfilter_threshold": "SemFilter keep threshold on the LLM judge's rubric score (SemFilter plans only)",
}


def coverage_card(space, evals: list[PoolEval]) -> str:
    lines = []
    for knob, levels in {"plan": space.PLANS, **all_searched(space)}.items():
        parts = []
        for level in levels:
            hit = [
                e
                for e in evals
                if space.validate(e.cfg).get(knob) == level
                and knob in ("plan", *searched(space, e.cfg["plan"]))
            ]
            parts.append(
                f"{level}({len(hit)}x, best quality {max(e.gain for e in hit):+.3f}, min tok "
                f"{min(e.tokens for e in hit):.0f})"
                if hit
                else f"{level}(UNTRIED)"
            )
        lines.append(f"  {knob}: " + " ".join(parts))
    return "\n".join(lines)


def bases_of(evals: list[PoolEval]) -> dict[str, dict]:
    """Anchors = FULLY evaluated configurations when any (partial ones are noisier)."""
    full = [e for e in evals if not e.n_full or e.n >= e.n_full] or evals
    return {
        "quality": max(full, key=lambda e: e.gain).cfg,
        "cost": min(full, key=lambda e: (e.tokens, e.latency_ms)).cfg,
    }


def ordinary_prompt(
    space,
    evals: list[PoolEval],
    ctx: dict[str, Any],
    *,
    n_propose: int = N_PROPOSE,
    confidence: str | None = None,
    rect: dict[str, Any] | None = None,
) -> tuple[str, str, dict]:
    bases = bases_of(evals)
    fr = sorted(az.front(evals), key=lambda e: -e.gain)
    by_gain = sorted(evals, key=lambda e: -e.gain)
    by_cost = sorted(evals, key=lambda e: (e.tokens, e.latency_ms))[:10]
    user = [
        "OBJECTIVES (TRAIN profiling; the optimizer maximizes the HYPERVOLUME of all three):",
        f"  - quality: {ctx['quality_text']} (higher better)",
        "  - tok: LLM tokens per episode, agent generation + semantic-filter calls (lower better)",
        "  - lat_s: retrieval + full episode latency in seconds (lower better)",
        f"DATASET: {ctx['dataset']} ({ctx['panel']})",
        "SEARCH SPACE (knob: allowed values):",
        space_card(space),
        "BASE (best-quality fully evaluated configuration; default delta anchor):",
        "  " + knobs_text(space, bases["quality"]),
        "BASE_COST (cheapest fully evaluated configuration; anchor for cost-side deltas):",
        "  " + knobs_text(space, bases["cost"]),
    ]
    if confidence:
        user.append(f"MODEL CONFIDENCE: {confidence}")
    user += ["CURRENT PARETO FRONT (observed, noise-aware):"] + [
        "  " + eval_line(space, e) for e in fr
    ]
    user += [f"ALL EVALUATIONS ({len(evals)}, best quality first):"]
    user += ["  " + eval_line(space, e) for e in by_gain]
    user += ["CHEAPEST EVALUATIONS (lowest tokens first):"] + [
        "  " + eval_line(space, e) for e in by_cost
    ]
    user += [
        "GLOBAL COVERAGE (per knob value: times tried, best quality, min tokens; UNTRIED = free information):"
    ]
    user.append(coverage_card(space, evals))
    if rect:
        user.append(
            f"MAX-HV EMPTY RECTANGLE ({rect['kind']}): quality in [{rect['gain'][0]:+.3f}, {rect['gain'][1]:+.3f}], "
            f"cost score (mean of -ln tokens, -ln latency_ms) in [{rect['cost'][0]:.3f}, {rect['cost'][1]:.3f}]."
        )
    user.append(
        f"Propose {n_propose} NEW configurations that maximize HYPERVOLUME by COVERING the trade-off: (a) land at "
        "least one in the MAX-HV EMPTY RECTANGLE; (b) push an end only if it is not saturated; (c) diagnose from the "
        "data, not folklore. Avoid configurations already evaluated or dominated. For EACH candidate output ONLY the "
        'SEARCHED knobs that DIFFER from its anchor and declare the anchor with "base": "quality" (default) or '
        '"base": "cost". Never output a FIXED knob. Use knob names and values exactly as in the SEARCH SPACE. '
        'Respond with ONLY a JSON object: {"reasoning": "<1-2 sentences>", "candidates": '
        '[{"<knob>": <value>, "base": "quality"}, ...]}'
    )
    return SYSTEM_ORDINARY, "\n".join(user), bases


def gap_prompt(
    space,
    evals: list[PoolEval],
    ctx: dict[str, Any],
    rect: dict[str, Any],
    n_propose: int = N_PROPOSE,
) -> tuple[str, str, dict]:
    bases = bases_of(evals)
    a, b = space.validate(bases["quality"]), space.validate(bases["cost"])
    keys = ("plan", *all_searched(space))
    diffs = [
        f"  {k}: quality_anchor={a.get(k)!r}; cost_anchor={b.get(k)!r}"
        for k in keys
        if a.get(k) != b.get(k)
    ]
    user = [
        f"DATASET {ctx['dataset']}; quality = {ctx['quality_text']}.",
        f"GAP ({rect['kind']}): quality in [{rect['gain'][0]:+.3f}, {rect['gain'][1]:+.3f}], cost score "
        f"(mean of -ln tokens, -ln latency_ms) in [{rect['cost'][0]:.3f}, {rect['cost'][1]:.3f}].",
        "QUALITY ANCHOR:",
        "  " + knobs_text(space, a),
        "COST ANCHOR:",
        "  " + knobs_text(space, b),
        "ACTIVE KNOB DIFFERENCES BETWEEN THE ANCHORS:",
        *(diffs or ["  (none: the same configuration anchors both ends)"]),
        "TESTED CONFIGURATIONS (do not repeat):",
        *["  " + eval_line(space, e) for e in evals],
        "SEARCH SPACE:",
        space_card(space),
        f"Return at most {n_propose} diverse candidates that could land inside the GAP. For each, choose one anchor "
        "and give a non-empty set of explicit deltas containing only SEARCHED knobs that differ from it (never a "
        'FIXED knob). Respond with ONLY this JSON: {"reasoning": "<brief>", "candidates": [{"base": '
        '"quality|cost", "deltas": {"<knob>": <value>}, "hypothesis": "<why>"}]}',
    ]
    return SYSTEM_GAP, "\n".join(user), bases


# ------------------------------------------------------------------ strict parsing


META = ("base", "deltas", "hypothesis", "reason", "why")


def _level(levels: tuple, value: Any, knob: str) -> Any:
    """Exact grid level (string spellings of numbers / case accepted); anything off-grid is rejected."""
    for level in levels:
        if value == level and not isinstance(value, bool):
            return level
    if isinstance(value, str):
        for level in levels:
            if str(level).lower() == value.strip().lower():
                return level
        try:
            f = float(value)
        except ValueError:
            f = None
        if f is not None:
            for level in levels:
                if (
                    isinstance(level, (int, float))
                    and not isinstance(level, bool)
                    and float(level) == f
                ):
                    return level
    raise ValueError(f"{knob}={value!r} is outside the space {list(levels)}")


def apply_delta(space, base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    delta = {az.ALIASES.get(k, k): v for k, v in delta.items() if k not in META}
    fixed = set(getattr(space, "FIXED", {}))
    bad_fixed = sorted(set(delta) & fixed)
    if bad_fixed:
        raise ValueError(f"fixed knobs are not searchable: {bad_fixed}")
    cfg = space.validate(base)
    plan = delta.pop("plan", cfg["plan"])
    plan = az.PLAN_ALIASES.get(plan, plan)
    if plan not in space.PLANS:
        raise ValueError(f"unknown plan {plan!r}")
    dom = searched(space, plan)
    unknown = sorted(set(delta) - set(dom))
    if unknown:
        raise ValueError(f"knobs not searched for plan {plan}: {unknown}")
    out = {"plan": plan, **{k: cfg[k] for k in dom if k in cfg}}
    for knob, value in delta.items():
        out[knob] = _level(tuple(dom[knob]), value, knob)
    return space.validate(out)


def parse_candidates(
    space,
    obj: dict[str, Any] | None,
    bases: dict[str, dict],
    cap: int,
    exclude: set[str] = frozenset(),
) -> tuple[list[dict], list[str]]:
    out, errors, seen = [], [], set(exclude)
    if not obj or not isinstance(obj.get("candidates"), list):
        return [], ["no candidates list"]
    for i, cand in enumerate(obj["candidates"][: 2 * cap]):
        try:
            if not isinstance(cand, dict):
                raise TypeError("candidate is not an object")
            anchor = str(cand.get("base", "quality")).replace("_anchor", "")
            if anchor not in bases:
                raise ValueError(f"unknown base {anchor!r}")
            delta = cand.get("deltas", cand)
            if not isinstance(delta, dict):
                raise TypeError("deltas is not an object")
            cfg = apply_delta(space, bases[anchor], delta)
            k = space.key(cfg)
            if k in seen:
                raise ValueError("duplicate or already evaluated")
            seen.add(k)
            out.append(cfg)
        except Exception as e:  # noqa: BLE001 - a malformed candidate is rejected, never fatal
            errors.append(f"candidate {i}: {type(e).__name__}: {e}")
        if len(out) >= cap:
            break
    return out, errors


# ------------------------------------------------------------------ client


class QwenChat:
    """OpenAI-compatible chat on the local vLLM endpoints (first that answers), temperature 0, thinking off."""

    name = "local-qwen"

    def __init__(
        self,
        api_bases=API_BASES,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: float = TIMEOUT_S,
    ):
        from trimem.experiments import llm_budget

        self.api_bases, self.timeout = tuple(api_bases), timeout
        self.max_tokens = max_tokens or llm_budget.OUTPUT_TOKENS
        self.model = model

    def _model(self, base: str) -> str:
        if self.model:
            return self.model
        with urllib.request.urlopen(f"{base}/models", timeout=30) as r:
            return json.loads(r.read())["data"][0]["id"]

    def chat(
        self, system: str, user: str, temperature: float = TEMPERATURE
    ) -> az.ChatResult:
        if os.environ.get(OFFLINE_ENV):  # offline test runs: never call a live endpoint
            return az.ChatResult(None, error=f"{OFFLINE_ENV} set: analyst offline")
        err = None
        for base in self.api_bases:
            t0 = time.monotonic()
            try:
                body = {
                    "model": self._model(base),
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                req = urllib.request.Request(
                    f"{base}/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    res = json.loads(r.read())
                usage = res.get("usage") or {}
                return az.ChatResult(
                    res["choices"][0]["message"].get("content"),
                    input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                    latency_ms=(time.monotonic() - t0) * 1000,
                    raw={
                        "endpoint": base,
                        "model": body["model"],
                        "finish_reason": res["choices"][0].get("finish_reason"),
                    },
                )
            except Exception as e:  # noqa: BLE001 - try the next endpoint, then report
                err = f"{base}: {type(e).__name__}: {e}"
        return az.ChatResult(None, error=err)


# ------------------------------------------------------------------ one round


def evals_of(
    space, points: list, state: dict, status: dict[str, dict]
) -> list[PoolEval]:
    """Training points (state order) -> analyst evaluations; ``status`` = key -> {n, n_full, pruned}."""
    pk = {space.key(p.cfg): p for p in points}
    order = [space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]]
    out, seen = [], set()
    for k in order:
        if k in pk and k not in seen and pk[k].cfg["plan"] != "NoMemory":
            seen.add(k)
            p, st = pk[k], status.get(k, {})
            out.append(
                PoolEval(
                    f"e{len(out):02d}",
                    space.validate(p.cfg),
                    float(p.quality),
                    math.sqrt(max(p.noise[0], 0.0)),
                    float(p.tokens),
                    float(p.latency_ms),
                    {},
                    math.sqrt(max(p.noise[1], 0.0)),
                    math.sqrt(max(p.noise[2], 0.0)),
                    int(st.get("n", 0)),
                    int(st.get("n_full", 0)),
                    bool(st.get("pruned", False)),
                )
            )
    return out


def advise(
    space,
    points: list,
    state: dict,
    *,
    status: dict[str, dict],
    forced: bool,
    rnd: int,
    ctx: dict[str, Any],
    out_dir: Path,
    chat=None,
    confidence: str | None = None,
    n_propose: int = N_PROPOSE,
) -> tuple[list[dict], list | None, dict]:
    """One analyst call. Returns (agent_rows with ``_label``, realized front for a forced gap or None, summary).
    Never raises; every outcome (incl. failures) is logged to ``out_dir/analyst/round_NN.json``."""
    log_p = Path(out_dir) / "analyst" / f"round_{rnd:02d}.json"
    rec: dict[str, Any] = {
        "protocol": PROTOCOL,
        "round": rnd,
        "forced": forced,
        "started_at": datetime.now(UTC).isoformat(),
    }
    rows, front, label = [], None, "agent"
    try:
        evals = evals_of(space, points, state, status)
        if not evals:
            raise ValueError("no profiled configurations")
        rect = az.max_hv_rectangle(evals, "quality" if rnd % 2 else "cost")
        if forced and rect:
            label = "agent-gap"
            system, user, bases = gap_prompt(space, evals, ctx, rect, n_propose)
        else:
            system, user, bases = ordinary_prompt(
                space, evals, ctx, n_propose=n_propose, confidence=confidence, rect=rect
            )
        rec.update(
            label=label,
            system=system,
            user=user,
            rect=rect,
            prompt_sha256=hashlib.sha256((system + user).encode()).hexdigest(),
        )
        client = chat or QwenChat()
        try:
            res = client.chat(system, user, TEMPERATURE)
        except Exception as e:  # noqa: BLE001
            res = az.ChatResult(None, error=f"client {type(e).__name__}: {e}")
        rec.update(
            response=res.text,
            client=getattr(client, "name", "custom"),
            error=res.error,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            latency_ms=res.latency_ms,
            raw=res.raw,
        )
        tested = {space.key(e.cfg) for e in evals} | {
            space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]
        }
        obj = az.extract_json(res.text)
        cands, rejects = (
            parse_candidates(space, obj, bases, n_propose, tested)
            if res.error is None
            else ([], [res.error])
        )
        rec.update(
            parsed=obj,
            candidates=cands,
            rejects=rejects,
            status="ok" if cands else ("error" if res.error else "no_valid_candidates"),
        )
        rows = [{**c, "_label": label} for c in cands]
        front = (
            [tuple(e) for e in rect["endpoints"]]
            if (forced and rect and rows)
            else None
        )
    except Exception as e:  # noqa: BLE001 - the analyst never crashes a round
        rec.update(
            status="error",
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),  # diagnosable next time (2026-10-22: ALF v7 round 6)
            candidates=[],
            rejects=[],
        )
        rows, front = [], None
    rec["finished_at"] = datetime.now(UTC).isoformat()
    try:
        log_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = log_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, default=str, ensure_ascii=False))
        tmp.replace(log_p)
    except OSError as e:
        rec["log_error"] = str(e)
    summary = {
        "protocol": PROTOCOL,
        "label": rec.get("label", label),
        "status": rec["status"],
        "candidates": len(rec.get("candidates") or []),
        "rejects": len(rec.get("rejects") or []),
        "error": (rec.get("error") or None) and str(rec["error"])[:300],
        "log": str(log_p),
        "input_tokens": rec.get("input_tokens"),
        "output_tokens": rec.get("output_tokens"),
    }
    return rows, front, summary
