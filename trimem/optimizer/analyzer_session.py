"""LLM analyzer for the agent-env session BO: RAG-PE's stage-guided candidate channel (R^LLM).

Ported from the harness analyzer (tridb_v6_stage/trimem/optimizer/analyzer.py), itself adapted from
RAG-Stack (arXiv 2608.03487; rag_stack/optimizer/agent/llm_agent.py ``_build_messages`` and
``_build_gap_exploration_messages``). The LLM reads every profiled configuration with its per-stage
sub-metrics (``trimem.experiments.session_submetrics``) and proposes edits to an evaluated anchor.
Proposals are snapped to the agent-env grid and only ENTER the candidate pool; qLogNEHVI
arbitration decides. Sub-metrics are never GP inputs or objectives. Only configurations,
objective values and sub-metrics are sent (no trajectories, no task text). Every call is logged to
``opt.analyzer_call``. Non-strict: a failed call or unparseable reply yields no proposals.

Differences from RAG-PE: three objectives (paired gain, tokens, latency) instead of (quality,
performance); costs are measured, so there is no PERF ATTRIBUTION card; no interactive query loop;
a single-point front gets a trade-off-arm rectangle (RAG-PE has no rectangle then).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trimem.optimizer import session_space_agentenv as sv

KNOB_DOC = {
    "plan": (
        "retrieval plan. VGR = HNSW vector seeds over task embeddings -> the seed tasks' sessions -> "
        "similarity hops between sessions -> pool ranked by query cosine. VGR+Rank = VGR pool re-ranked "
        "by ColBERT (late interaction, no LLM tokens). P2 (shown as VGR+SemFilter) = an LLM judges every "
        "pooled session with a rubric (keep if score >= 0.6), one call per candidate. P2+Rank = "
        "SemFilter then ColBERT ranking"
    ),
    "seed_count": "number of vector seed TASKS (HNSW LIMIT); each contributes its stored sessions",
    "depth": "similarity hops from the seed sessions (1 = only the seeds' sessions and their direct neighbours)",
    "fanout": "neighbours per session, per edge type, per hop",
    "top_k": "sessions injected into the agent prompt (32k-token memory budget)",
    "edges": (
        "similarity edges followed: similar = each session -> its 5 nearest sessions; similar_up = the "
        "reverse (sessions that list this one among their nearest); similar+similar_up = both"
    ),
}

SYSTEM_ORDINARY = (
    "You are an expert at tuning the trajectory-memory retrieval of an LLM agent (ALFWorld / ScienceWorld "
    "household and science tasks: before each episode, past successful episodes of OTHER tasks are retrieved "
    "and injected into the agent's prompt). You advise a MULTI-OBJECTIVE Bayesian optimizer whose true "
    "target is the HYPERVOLUME of the (paired quality gain, tokens, latency) Pareto front. A configuration "
    "that maximizes gain but costs far more does NOT necessarily improve the hypervolume. Read the past "
    "evaluations, the per-stage sub-metrics and the current front, diagnose where configurations fall short, "
    "then propose a DIVERSE set of NEW configurations that EXTEND the front: some pushing gain, some trading "
    "a little gain for much lower cost, some filling gaps. Do NOT assume generic retrieval folklore (e.g. "
    "'more memory always helps', 'an LLM filter or a re-ranker always helps'); let the evaluations and "
    "sub-metrics show what drives quality HERE. Gain is noisy: each value averages 100 binary or 0-1 "
    "outcomes (standard error about 0.04)."
)
SYSTEM_GAP = (
    "You design a bounded FORCED EXPLORATION for a stalled multi-objective optimizer. The rectangle you are "
    "given is the largest unclaimed region of its realized (gain, cost) Pareto front. Explore that region; "
    "do not exploit a known endpoint and do not assume domain folklore. Every proposal must be justified by "
    "the supplied observations, valid in the declared space, and distinct from tested configurations. You "
    "only design candidates; another policy validates, ranks and may decline all of them."
)


@dataclass
class ChatResult:
    text: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float = 0.0
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ClaudeCLI:
    name = "claude-cli"

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        timeout_s: float = 600.0,
        max_budget_usd: float = 2.0,
    ):
        self.model, self.timeout_s, self.max_budget_usd = (
            model,
            timeout_s,
            max_budget_usd,
        )

    def chat(self, system: str, user: str, temperature: float) -> ChatResult:
        del (
            temperature
        )  # the CLI exposes no temperature; recorded with the call, not applied
        started = time.perf_counter()
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--max-budget-usd",
            str(self.max_budget_usd),
            "--system-prompt",
            system,
        ]
        with tempfile.TemporaryDirectory(prefix="trimem-analyzer-") as cwd:
            try:
                out = subprocess.run(
                    cmd,
                    input=user,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    cwd=cwd,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                return ChatResult(
                    None,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(e).__name__}: {e}",
                )
        ms = (time.perf_counter() - started) * 1000
        try:
            payload = json.loads(out.stdout)
        except json.JSONDecodeError:
            return ChatResult(
                None,
                latency_ms=ms,
                error=f"rc={out.returncode} non-JSON: {out.stderr[-500:]}",
            )
        usage = payload.get("usage") or {}
        tokens_in = sum(
            int(usage.get(k) or 0)
            for k in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        err = None if not payload.get("is_error") else str(payload.get("result"))[:500]
        return ChatResult(
            payload.get("result") if not err else None,
            tokens_in or None,
            usage.get("output_tokens"),
            payload.get("total_cost_usd"),
            ms,
            err,
            payload,
        )


CODEX = "/local-scratch/localhome/hza214/local/node-v22.14.0-linux-x64/bin/codex"


class CodexCLI:
    """Codex CLI headless (read-only sandbox, ephemeral, user config ignored). User decision
    2026-10-16: the analyzer uses Codex Luna; ``gpt-5.5-luna`` is rejected for ChatGPT accounts
    (HTTP 400), so the available Luna, ``gpt-5.6-luna``, is the default."""

    name = "codex-cli"

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        effort: str = "medium",
        timeout_s: float = 900.0,
        reasoning_summary: str | None = None,
    ):
        self.model, self.effort, self.timeout_s = model, effort, timeout_s
        self.reasoning_summary = reasoning_summary

    def chat(self, system: str, user: str, temperature: float) -> ChatResult:
        del temperature  # not exposed by the CLI; recorded with the call
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="trimem-analyzer-") as cwd:
            answer = Path(cwd) / "answer.txt"
            cmd = [
                CODEX,
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "-C",
                cwd,
                "-s",
                "read-only",
                "-m",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.effort}"',
                "--json",
                "-o",
                str(answer),
                "-",
            ]
            if self.reasoning_summary is not None:
                cmd[2:2] = ["-c", f'model_reasoning_summary="{self.reasoning_summary}"']
            prompt = f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER MESSAGE:\n{user}"
            try:
                out = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    cwd=cwd,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                return ChatResult(
                    None,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(e).__name__}: {e}",
                )
            text = answer.read_text() if answer.exists() else None
        ms = (time.perf_counter() - started) * 1000
        usage, err = {}, None
        for line in out.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "turn.completed":
                usage = ev.get("usage") or {}
            elif ev.get("type") == "turn.failed":
                err = str(ev.get("error"))[:500]
        if err or not text:
            return ChatResult(
                None,
                latency_ms=ms,
                error=err or f"rc={out.returncode} no answer: {out.stderr[-300:]}",
            )
        return ChatResult(
            text,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            None,
            ms,
            None,
            {"usage": usage},
        )


# ------------------------------------------------------------------ history + cards


@dataclass
class Eval:
    """One profiled configuration as the analyzer sees it (panel means per query)."""

    eid: str
    cfg: dict[str, Any]
    gain: float
    gain_se: float
    tokens: float
    latency_ms: float
    sub: dict[str, float]
    tok_se: float = 0.0  # standard error of ln(mean tokens)
    lat_se: float = 0.0  # standard error of ln(mean latency)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v == 0:
            return "0"
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def knobs_text(cfg: dict[str, Any]) -> str:
    cfg = sv.validate(cfg)
    return " ".join(f"{k}={cfg[k]}" for k in ("plan", *sv.grid()))


def eval_line(e: Eval) -> str:
    sub = " ".join(f"{k}={_fmt(v)}" for k, v in sorted(e.sub.items()))
    return (
        f"{e.eid} {knobs_text(e.cfg)} -> gain={e.gain:+.3f}(se {e.gain_se:.3f}) tok={e.tokens:.0f} "
        f"lat_s={e.latency_ms / 1000:.1f}" + (f" | {sub}" if sub else "")
    )


def cost(e: Eval) -> float:
    """The engine's cost axis (maximised): mean of -ln tokens and -ln latency."""
    return 0.5 * (-math.log(e.tokens) - math.log(e.latency_ms))


RELAX_Z = 1.0  # same noise-aware front as session_bo_agentenv.relaxed_pareto


def front(evals: list[Eval], z: float = RELAX_Z) -> list[Eval]:
    """Noise-aware front (max gain, min tokens, min latency): b dominates a only if b is at least as good
    everywhere and better by more than z pooled standard errors somewhere (z = 0: strict front)."""

    def vec(e):
        return (e.gain, -math.log(e.tokens), -math.log(e.latency_ms)), (
            e.gain_se,
            e.tok_se,
            e.lat_se,
        )

    def dom(b, a):
        (x, sx), (y, sy) = vec(b), vec(a)
        return all(i >= j for i, j in zip(x, y, strict=True)) and any(
            x[k] > y[k] + z * math.sqrt(sx[k] ** 2 + sy[k] ** 2) for k in range(3)
        )

    return [a for a in evals if not any(b is not a and dom(b, a) for b in evals)]


def max_hv_rectangle(evals: list[Eval], arm: str = "quality") -> dict[str, Any] | None:
    """Largest unclaimed (gain, cost) rectangle of the realized front.

    >= 2 front points: the largest normalised rectangle between adjacent front points (RAG-PE), over
    trade-off pairs only (higher gain at higher cost; the noise-aware front can hold pairs without one).
    1 front point, or no trade-off pair, f* = the max-gain front point (g*, c*): RAG-PE has none. We use a TRADE-OFF ARM: ``quality`` arm =
    gain in [g*, g* + 2 se*] at cost in [c* - ln 1.5, c*] (more gain for up to 1.5x cost);
    ``cost`` arm = gain in [g* - 2 se*, g*] at cost in [c*, c* + ln 1.5] (up to 1.5x cheaper for
    a little less gain). Neither region is dominated by f*, so a point landing there adds hypervolume.
    """
    fr = front(evals)
    pts = sorted({(e.gain, cost(e)) for e in fr})
    best = None
    if len(pts) >= 2:
        gs, cs = [p[0] for p in pts], [p[1] for p in pts]
        gr, cr = (max(gs) - min(gs)) or 1.0, (max(cs) - min(cs)) or 1.0
        area = 0.0
        for a, b in zip(pts, pts[1:], strict=False):
            w, h = (b[0] - a[0]) / gr, (a[1] - b[1]) / cr
            # only a genuine trade-off pair (more gain AND more cost) bounds a rectangle; the noise-aware
            # front can keep a pair where one point is better on both axes (within noise) -> no rectangle
            if w > 0 and h > 0 and w * h > area:
                area, best = w * h, (a, b)
    if best is not None:
        (g_lo, c_hi), (g_hi, c_lo) = best
        return {
            "kind": "between adjacent front points",
            "gain": (g_lo, g_hi),
            "cost": (c_lo, c_hi),
            "endpoints": [best[0], best[1]],
        }
    if not fr:
        return None
    star = max(fr, key=lambda e: e.gain)
    g, c, se, d = star.gain, cost(star), max(star.gain_se, 0.02), math.log(1.5)
    if arm == "quality":
        return {
            "kind": "single-point front, quality arm",
            "gain": (g, g + 2 * se),
            "cost": (c - d, c),
            "endpoints": [(g, c), (g + 2 * se, c - d)],
        }
    return {
        "kind": "single-point front, cost arm",
        "gain": (g - 2 * se, g),
        "cost": (c, c + d),
        "endpoints": [(g - 2 * se, c + d), (g, c)],
    }


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 6:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def coverage_card(evals: list[Eval]) -> str:
    levels_of = {"plan": sv.PLANS, **sv.grid()}
    lines = []
    for knob, levels in levels_of.items():
        parts = []
        for level in levels:
            hit = [e for e in evals if sv.validate(e.cfg)[knob] == level]
            parts.append(
                f"{level}({len(hit)}x, best gain {max(e.gain for e in hit):+.3f}, min tok "
                f"{min(e.tokens for e in hit):.0f})"
                if hit
                else f"{level}(UNTRIED)"
            )
        lines.append(f"  {knob}: " + " ".join(parts))
    return "\n".join(lines)


def reliability_card(evals: list[Eval]) -> str:
    names = sorted({k for e in evals for k in e.sub})
    parts = []
    for name in names:
        xs = [(e.sub[name], e.gain) for e in evals if name in e.sub]
        rho = _spearman([a for a, _ in xs], [b for _, b in xs])
        if rho is not None:
            parts.append(f"{name}={rho:+.2f}(n={len(xs)})")
    return (
        "  " + " ".join(parts)
        if parts
        else "  (fewer than 6 evaluations with sub-metrics)"
    )


def toggle_card(evals: list[Eval]) -> str:
    lines = []
    for knob, levels in (
        ("plan", sv.PLANS),
        ("edges", sv.EDGES),
        ("top_k", sv.grid()["top_k"]),
        ("depth", sv.grid()["depth"]),
        ("seed_count", sv.grid()["seed_count"]),
    ):
        parts = []
        for level in levels:
            hit = [e for e in evals if sv.validate(e.cfg)[knob] == level]
            if len(hit) >= 2:
                parts.append(
                    f"{level}: mean gain={sum(e.gain for e in hit) / len(hit):+.3f} "
                    f"tok={sum(e.tokens for e in hit) / len(hit):.0f} "
                    f"lat_s={sum(e.latency_ms for e in hit) / len(hit) / 1000:.1f} (n={len(hit)})"
                )
        if parts:
            lines.append(f"  {knob}: " + " | ".join(parts))
    return "\n".join(lines) or "  (fewer than 2 evaluations per level)"


def space_card() -> str:
    lines = [f"  plan: {list(sv.PLANS)}  (P2 is displayed to users as VGR+SemFilter)"]
    lines += [f"  {k}: {list(v)}" for k, v in sv.grid().items()]
    lines.append(
        "All knobs are active for every plan. NoMemory is the paired baseline, never a candidate."
    )
    lines += ["KNOB SEMANTICS:"] + [f"  {k}: {v}" for k, v in KNOB_DOC.items()]
    return "\n".join(lines)


def sub_card() -> str:
    from trimem.experiments.session_submetrics import DESCRIPTIONS

    return "\n".join(f"  - {k}: {v}" for k, v in DESCRIPTIONS.items())


def bases_of(evals: list[Eval]) -> dict[str, dict]:
    return {
        "quality": max(evals, key=lambda e: e.gain).cfg,
        "cost": min(evals, key=lambda e: (e.tokens, e.latency_ms)).cfg,
    }


def ordinary_prompt(
    evals: list[Eval],
    ctx: dict[str, Any],
    *,
    n_propose: int = 4,
    confidence: str | None = None,
    notes: str | None = None,
    rect: dict[str, Any] | None = None,
) -> tuple[str, str, dict]:
    bases = bases_of(evals)
    fr = sorted(front(evals), key=lambda e: -e.gain)
    by_gain = sorted(evals, key=lambda e: -e.gain)
    by_cost = sorted(evals, key=lambda e: (e.tokens, e.latency_ms))[:10]
    user = [
        "OBJECTIVES (per query, averaged over the same 100 training tasks; the optimizer maximizes the "
        "HYPERVOLUME of all three):",
        "  - gain: success/score with memory minus without memory on the same task (higher better; "
        f"No-Memory success on this panel = {ctx['nomemory_quality']:.2f})",
        "  - tok: LLM tokens of the episode, generation + semantic-filter calls (lower better)",
        "  - lat_s: retrieval + full episode latency (lower better)",
        f"DATASET: {ctx['dataset']} ({ctx['panel']})",
        (
            "INTERMEDIATE SUB-METRICS (per-configuration panel means; diagnostics, NOT optimized directly; stages "
            "couple, so use them as hints and let the evaluations show which knobs move them):"
        ),
        sub_card(),
        "SEARCH SPACE (knob: allowed values):",
        space_card(),
        "BASE (best-gain evaluated configuration; default delta anchor):",
        "  " + knobs_text(bases["quality"]),
        "BASE_COST (cheapest evaluated configuration; anchor for cost-side deltas):",
        "  " + knobs_text(bases["cost"]),
    ]
    if confidence:
        user.append(f"MODEL CONFIDENCE: {confidence}")
    user += [
        "CURRENT PARETO FRONT (observed, noise-aware: a configuration is dropped only if another is at least "
        "as good on all three objectives and better by more than one pooled standard error on one):"
    ] + ["  " + eval_line(e) for e in fr]
    user += [f"ALL EVALUATIONS ({len(evals)}, best gain first):"] + [
        "  " + eval_line(e) for e in by_gain
    ]
    user += [
        "CHEAPEST EVALUATIONS (lowest tokens first; the proven cheap structures for cost-side proposals):"
    ]
    user += ["  " + eval_line(e) for e in by_cost]
    user += [
        (
            "GLOBAL COVERAGE (per knob value: times tried, best gain, min tokens; UNTRIED values are free "
            "information; a single bad reading may be caused by OTHER knobs):"
        ),
        coverage_card(evals),
    ]
    user += [
        (
            "SUB-METRIC RELIABILITY (Spearman with gain over all evaluations; trust a sub-metric as a lever "
            "ONLY if it correlates here):"
        ),
        reliability_card(evals),
    ]
    user += ["KNOB EFFECTS (observed means per level):", toggle_card(evals)]
    if notes:
        user += [
            "YOUR SEARCH NOTES (written by you last round; trust the data over stale notes):",
            notes[:1200],
        ]
    if rect:
        user.append(
            f"MAX-HV EMPTY RECTANGLE ({rect['kind']}): gain in [{rect['gain'][0]:+.3f}, "
            f"{rect['gain'][1]:+.3f}], cost score (mean of -ln tokens, -ln latency_ms) in "
            f"[{rect['cost'][0]:.3f}, {rect['cost'][1]:.3f}]."
        )
    user += [
        (
            f"Propose {n_propose} NEW configurations that maximize HYPERVOLUME by COVERING the trade-off: "
            "(a) land at least one in the MAX-HV EMPTY RECTANGLE; (b) push an end only if it is not saturated; "
            "(c) diagnose from the data, not folklore: use a sub-metric as a lever only where SUB-METRIC "
            "RELIABILITY shows it correlates with gain. Avoid configurations the front already dominates and "
            "configurations already evaluated. For EACH candidate output ONLY the knobs that DIFFER from its "
            'anchor and declare the anchor with "base": "quality" (default) or "base": "cost". Cost-side '
            'proposals MUST use "base": "cost". Use knob names and values exactly as in the SEARCH SPACE. '
            'Respond with ONLY a JSON object: {"reasoning": "<1-2 sentences>", "search_notes": "<up to 120 '
            'words: validated hypotheses, dead ends, next bets>", "candidates": [{"<knob>": <value>, '
            '"base": "quality"}, ...]}'
        ),
    ]
    return SYSTEM_ORDINARY, "\n".join(user), bases


def gap_prompt(
    evals: list[Eval], ctx: dict[str, Any], rect: dict[str, Any], n_propose: int = 4
) -> tuple[str, str, dict]:
    bases = bases_of(evals)
    qa = max(evals, key=lambda e: e.gain)
    ca = min(evals, key=lambda e: (e.tokens, e.latency_ms))
    a, b = sv.validate(qa.cfg), sv.validate(ca.cfg)
    diffs = [
        f"  {k}: quality_anchor={a[k]!r}; cost_anchor={b[k]!r}"
        for k in ("plan", *sv.grid())
        if a[k] != b[k]
    ]
    user = [
        f"DATASET {ctx['dataset']}; gain is per query (noise s.e. about 0.04).",
        f"GAP ({rect['kind']}): gain in [{rect['gain'][0]:+.3f}, {rect['gain'][1]:+.3f}], cost score "
        f"(mean of -ln tokens, -ln latency_ms) in [{rect['cost'][0]:.3f}, {rect['cost'][1]:.3f}].",
        "QUALITY ANCHOR:",
        "  " + eval_line(qa),
        "COST ANCHOR:",
        "  " + eval_line(ca),
        "ACTIVE KNOB DIFFERENCES BETWEEN THE ANCHORS:",
        *(diffs or ["  (none: the same configuration anchors both ends)"]),
        "TESTED CONFIGURATIONS (do not repeat):",
        *["  " + eval_line(e) for e in evals],
        "SEARCH SPACE:",
        space_card(),
        (
            f"Return at most {n_propose} diverse candidates that could land inside the GAP. For each, choose one "
            "anchor and give a non-empty set of explicit deltas containing only knobs that differ from it. Prefer "
            "an untested interaction, an untried level, or a one-sided graft. Respond with ONLY this JSON: "
            '{"reasoning": "<brief>", "candidates": [{"base": "quality|cost", "deltas": '
            '{"<knob>": <value>}, "hypothesis": "<why>"}]}'
        ),
    ]
    return SYSTEM_GAP, "\n".join(user), bases


# ------------------------------------------------------------------ parse + snap


def extract_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "candidates" in obj:
            return obj
    return None


ALIASES = {
    "hops": "depth",
    "hop": "depth",
    "seeds": "seed_count",
    "k": "top_k",
    "topk": "top_k",
    "edge_set": "edges",
}
PLAN_ALIASES = {
    "VGR+SemFilter": "P2",
    "VGR+SemFilter+Rank": "P2+Rank",
    "V+SemFilter+G+Rank": "P1+Rank",
}


def _coerce(knob: str, value: Any) -> Any:
    levels = sv.grid()[knob]
    if value in levels:
        return value
    if isinstance(value, str):
        for level in levels:
            if str(level).lower() == value.strip().lower():
                return level
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{knob}={value!r} not allowed") from None
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and all(isinstance(x, int) for x in levels)
    ):
        return min(levels, key=lambda x: (abs(x - value), x))
    raise ValueError(f"{knob}={value!r} not allowed")


def apply_delta(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    delta = {
        ALIASES.get(k, k): v
        for k, v in delta.items()
        if k not in ("base", "hypothesis", "deltas")
    }
    cfg = sv.validate(base)
    plan = delta.pop("plan", cfg["plan"])
    plan = PLAN_ALIASES.get(plan, plan)
    if plan not in sv.PLANS:
        raise ValueError(f"unknown plan {plan!r}")
    unknown = set(delta) - set(sv.grid())
    if unknown:
        raise ValueError(f"unknown knobs {sorted(unknown)}")
    out = {**{k: cfg[k] for k in sv.grid()}, "plan": plan}
    for knob, value in delta.items():
        out[knob] = _coerce(knob, value)
    return sv.validate(out)


def parse_candidates(
    obj: dict[str, Any] | None, bases: dict[str, dict], cap: int
) -> tuple[list[dict], list[str]]:
    out, errors, seen = [], [], set()
    if not obj or not isinstance(obj.get("candidates"), list):
        return [], ["no candidates list"]
    for i, cand in enumerate(obj["candidates"][: 2 * cap]):
        try:
            if not isinstance(cand, dict):
                raise TypeError("candidate is not an object")
            anchor = str(cand.get("base", "quality")).replace("_anchor", "")
            if anchor not in bases:
                raise ValueError(f"unknown base {anchor!r}")
            cfg = apply_delta(bases[anchor], cand.get("deltas", cand))
            k = sv.key(cfg)
            if k in seen:
                raise ValueError("duplicate")
            seen.add(k)
            out.append(cfg)
        except Exception as e:  # noqa: BLE001 - a malformed candidate is rejected, never fatal
            errors.append(f"candidate {i}: {type(e).__name__}: {e}")
        if len(out) >= cap:
            break
    return out, errors


# ------------------------------------------------------------------ the analyzer

LOG_DDL = """CREATE TABLE IF NOT EXISTS opt.analyzer_call (
  call_id text PRIMARY KEY, experiment_id text NOT NULL, kind text NOT NULL, round int,
  client text, model text, system_prompt text NOT NULL, user_prompt text NOT NULL, reply text,
  parsed jsonb, proposals jsonb NOT NULL, status text NOT NULL, error text, input_tokens int,
  output_tokens int, cost_usd double precision, latency_ms double precision,
  created_at timestamptz NOT NULL DEFAULT now())"""


class Analyzer:
    def __init__(
        self,
        dsn: str | None,
        experiment_id: str,
        *,
        notes_path: Path | None = None,
        n_propose: int = 4,
        chat=None,
    ):
        self.dsn, self.experiment_id, self.n_propose = dsn, experiment_id, n_propose
        self.client = chat or CodexCLI()
        self.notes_path = notes_path

    def _notes(self) -> str | None:
        return (
            self.notes_path.read_text()
            if self.notes_path and self.notes_path.exists()
            else None
        )

    def _log(
        self, kind, rnd, system, user, res: ChatResult, parsed, proposals, status, error
    ) -> str | None:
        if not self.dsn:
            return None
        import psycopg
        from psycopg.types.json import Jsonb

        call_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        with psycopg.connect(self.dsn) as conn:
            conn.execute(LOG_DDL)
            conn.execute(
                "INSERT INTO opt.analyzer_call (call_id, experiment_id, kind, round, client, model, system_prompt, "
                "user_prompt, reply, parsed, proposals, status, error, input_tokens, output_tokens, cost_usd, "
                "latency_ms) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    call_id,
                    self.experiment_id,
                    kind,
                    rnd,
                    getattr(self.client, "name", "custom"),
                    getattr(self.client, "model", None),
                    system,
                    user,
                    res.text,
                    Jsonb(parsed) if parsed is not None else None,
                    Jsonb(proposals),
                    status,
                    error,
                    res.input_tokens,
                    res.output_tokens,
                    res.cost_usd,
                    res.latency_ms,
                ),
            )
        return call_id

    def _run(self, kind, rnd, system, user, temperature, bases):
        try:
            res = self.client.chat(system, user, temperature)
        except Exception as e:  # noqa: BLE001 - recorded in opt.analyzer_call
            res = ChatResult(None, error=f"client {type(e).__name__}: {e}")
        obj, cands, errors = None, [], []
        try:
            obj = extract_json(res.text)
            cands, errors = (
                parse_candidates(obj, bases, self.n_propose)
                if res.error is None
                else ([], [res.error])
            )
        except Exception as e:  # noqa: BLE001
            cands, errors = [], [f"parse {type(e).__name__}: {e}"]
        status = "ok" if cands else ("error" if res.error else "no_valid_candidates")
        call_id = self._log(
            kind,
            rnd,
            system,
            user,
            res,
            obj,
            cands,
            status,
            "; ".join(errors)[:2000] or None,
        )
        if obj and obj.get("search_notes") and self.notes_path:
            self.notes_path.write_text(str(obj["search_notes"])[:1200])
        return {
            "call_id": call_id,
            "status": status,
            "candidates": cands,
            "errors": errors,
            "reasoning": (obj or {}).get("reasoning"),
            "latency_ms": res.latency_ms,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cost_usd": res.cost_usd,
            "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(),
        }

    def propose(self, evals, ctx, *, rnd=None, confidence=None, rect=None):
        system, user, bases = ordinary_prompt(
            evals,
            ctx,
            n_propose=self.n_propose,
            confidence=confidence,
            notes=self._notes(),
            rect=rect,
        )
        return self._run("ordinary", rnd, system, user, 0.7, bases)

    def explore_gap(self, evals, ctx, rect, *, rnd=None):
        system, user, bases = gap_prompt(evals, ctx, rect, self.n_propose)
        return self._run("gap", rnd, system, user, 0.0, bases)


def confidence_text(model) -> str:
    """Leave-one-out RMSE of the quality GP relative to the spread of observed gain (RAG-PE card)."""
    import torch

    m = model.models[0]
    x = m.train_inputs[0]
    with torch.no_grad():
        k = m.covar_module(x).to_dense() + torch.diag(
            m.likelihood.noise.reshape(-1).expand(x.shape[0])
        )
        t = m.train_targets - m.mean_module(x)
        kinv = torch.linalg.inv(k)
        loo = (kinv @ t) / torch.diagonal(kinv)
    rel = float(loo.pow(2).mean().sqrt() / t.std().clamp_min(1e-12))
    level = "HIGH" if rel < 0.5 else "MODERATE" if rel < 0.85 else "LOW"
    return (
        f"{x.shape[0]} evaluations; gain-GP leave-one-out error is {rel:.2f} x the spread of observed gain -> "
        f"confidence {level}"
        + (
            " (treat knob contrasts as weak hints; explore untried values)"
            if level == "LOW"
            else ""
        )
    )
