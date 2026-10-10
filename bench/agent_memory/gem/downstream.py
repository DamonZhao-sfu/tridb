"""Downstream term of the retrieval objective: an estimator of E[#attempts | O].

Paper objective (``paper/sections/03_retrieval_definition.tex``)::

    cost(rho) = c_db(rho) + c_llm * n_sem(rho) + E[#attempts | O] * c_agent(|O|_tok)

This module supplies the last factor, learned from the EXG experience graphs in
``exg/results/experience_graph_v0.1.0`` (plan doc §4 row P5), and a helper
:func:`choose_k` that plugs it into the objective to pick k (or a budget B).

Features of the retrieved set O (deterministic, :func:`features_of`):
|O|, golden fraction, warning fraction, fraction of warnings whose ``fixed_by``
partner is also in O, mean query similarity, log(1 + |O|_tok).

Model: L2-regularised logistic regression for p1 = P(success on attempt 1 | O),
fitted by Newton/IRLS in numpy on standardised features. IRLS has no random
state, so the fit is a pure function of the training rows.

From p1 to E[#attempts] under an attempt cap n (:func:`expected_attempts`):
EXG retries a coding task when the tests fail, so with cap 2,
E = 1 + (1 - p1). For a general cap we use the geometric approximation where
every attempt succeeds independently with p1:
E = sum_{j<n} (1 - p1)^j = (1 - (1 - p1)^n) / p1. This is exact for n = 2. For
n > 2 it ignores that a retry sees feedback and therefore has a different
success rate.

Data and leakage. Collection was strictly sequential (protocol.json
``online_visibility``). Each ``attempt_1/prepared.json`` logs the hints that
were retrieved and shown to the agent, and it was saved before the agent call.
Training rows therefore use the logged O, which was built only from tasks
completed earlier. The label is the attempt-1 status (golden / warning).
Attempt-1 rows that contain a hint from their own task are counted and
dropped. ``infrastructure`` rows are excluded because they carry no outcome.

Caveat: the logged policy also retries on the agent's own ``needs_review`` flag
(this is the only retry trigger for QA). For that reason the observed
attempts-used are reported separately from E = 1 + (1 - p1), and a second
"retry" head is fitted for them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "n_hints",
    "frac_golden",
    "frac_warning",
    "frac_fail_with_fix",
    "mean_similarity",
    "log1p_tokens",
)
EXG_ATTEMPT_CAP = 2
DEFAULT_SEED = 42
TRAIN_FRACTION = 0.7
CHARS_PER_TOKEN_FALLBACK = 4.0
# A dataset counts as credible only with this many held-out rows and minority labels.
MIN_TEST_ROWS = 100
MIN_TEST_MINORITY = 20


# --------------------------------------------------------------------------- O


@dataclass(frozen=True)
class Hint:
    """One retrieved experience case as the estimator sees it."""

    case_uid: str
    status: str  # "golden" | "warning" | other
    repair_case_uid: str | None = None
    similarity: float | None = None
    tokens: float = 0.0


def features_of(hints: Sequence[Hint]) -> np.ndarray:
    """Return the deterministic feature vector of O (order: FEATURE_NAMES)."""
    n = len(hints)
    if n == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    uids = {h.case_uid for h in hints}
    golden = sum(h.status == "golden" for h in hints)
    warnings = [h for h in hints if h.status == "warning"]
    fixed = sum(
        h.repair_case_uid is not None and h.repair_case_uid in uids for h in warnings
    )
    sims = [h.similarity for h in hints if h.similarity is not None]
    return np.array(
        [
            float(n),
            golden / n,
            len(warnings) / n,
            fixed / len(warnings) if warnings else 0.0,
            float(np.mean(sims)) if sims else 0.0,
            math.log1p(sum(h.tokens for h in hints)),
        ],
        dtype=float,
    )


def o_tokens(hints: Sequence[Hint]) -> float:
    """Return |O|_tok."""
    return float(sum(h.tokens for h in hints))


# ----------------------------------------------------------------------- model


@dataclass(frozen=True)
class LogisticModel:
    """Standardised L2 logistic regression; coefficients are on the z-scale."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    l2: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return P(y = 1 | x) for a matrix (or one row) of raw features."""
        x2 = np.atleast_2d(np.asarray(x, dtype=float))
        z = (x2 - np.asarray(self.mean)) / np.asarray(self.scale)
        logits = z @ np.asarray(self.coef) + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))


def fit_logistic(
    x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 100, tol: float = 1e-10
) -> LogisticModel:
    """Fit by Newton/IRLS. The result depends only on (x, y, l2) and is reproducible."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
        raise ValueError("x must be (n, d) with n == len(y) > 0")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = np.hstack([np.ones((len(x), 1)), (x - mean) / scale])
    penalty = np.full(z.shape[1], l2)
    penalty[0] = 0.0  # the intercept is not penalised
    w = np.zeros(z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -35.0, 35.0)))
        grad = z.T @ (p - y) + penalty * w
        hess = (
            (z * (p * (1 - p))[:, None]).T @ z
            + np.diag(penalty)
            + 1e-9 * np.eye(len(w))
        )
        step = np.linalg.solve(hess, grad)
        w = w - step
        if float(np.max(np.abs(step))) < tol:
            break
    return LogisticModel(
        mean=tuple(mean.tolist()),
        scale=tuple(scale.tolist()),
        coef=tuple(w[1:].tolist()),
        intercept=float(w[0]),
        l2=l2,
    )


def expected_attempts(p1: float | np.ndarray, cap: int = EXG_ATTEMPT_CAP) -> np.ndarray:
    """E[#attempts] with an attempt cap; geometric approximation (exact for cap 2).

    E = sum_{j=0}^{cap-1} (1 - p1)^j. cap = 1 gives 1, and cap = 2 gives
    1 + (1 - p1), which is EXG's retry-on-failure rule.
    """
    if cap < 1:
        raise ValueError("cap must be >= 1")
    q = 1.0 - np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
    return sum(q**j for j in range(cap)) * np.ones_like(q)


@dataclass(frozen=True)
class AttemptEstimator:
    """E-hat[#attempts | O] built from a fitted p1 model."""

    p1_model: LogisticModel
    cap: int = EXG_ATTEMPT_CAP

    def p1(self, hints: Sequence[Hint]) -> float:
        """Return P(success on attempt 1 | O)."""
        return float(self.p1_model.predict_proba(features_of(hints))[0])

    def expected_attempts(self, hints: Sequence[Hint]) -> float:
        """Return E-hat[#attempts | O] under this estimator's cap."""
        return float(expected_attempts(self.p1(hints), self.cap))


def fit_estimator(
    x: np.ndarray,
    y_first_success: np.ndarray,
    l2: float = 1.0,
    cap: int = EXG_ATTEMPT_CAP,
) -> AttemptEstimator:
    """Fit an :class:`AttemptEstimator` on feature rows and attempt-1 success labels."""
    return AttemptEstimator(fit_logistic(x, y_first_success, l2=l2), cap=cap)


# -------------------------------------------------------------------- choose_k


@dataclass(frozen=True)
class Candidate:
    """The retrieved set O for one setting of k (or B) and its retrieval cost."""

    hints: tuple[Hint, ...]
    n_sem: int = 0
    c_db: float | None = None  # overrides the c_db argument of choose_k when set


@dataclass(frozen=True)
class CostBreakdown:
    """Cost terms of the paper objective for one value of k."""

    k: float
    retrieval_db: float
    retrieval_llm: float
    p1: float
    expected_attempts: float
    o_tokens: float
    agent_cost_per_attempt: float
    downstream: float
    total: float


@dataclass(frozen=True)
class KChoice:
    """Result of :func:`choose_k`: the argmin and every evaluated row."""

    k: float
    best: CostBreakdown
    table: tuple[CostBreakdown, ...] = field(default_factory=tuple)


def choose_k(
    candidates_by_k: Mapping[float, Candidate | Sequence[Hint]],
    c_db: float | Mapping[float, float] | Callable[[float], float],
    c_llm: float,
    c_agent_per_token: float,
    *,
    estimator: AttemptEstimator,
    c_agent_fixed: float = 0.0,
    base_prompt_tokens: float = 0.0,
) -> KChoice:
    """Return the k with the lowest paper objective, plus the per-term breakdown.

    cost(k) = c_db(k) + c_llm * n_sem(k)
              + E[#attempts | O_k] * (c_agent_fixed
                + c_agent_per_token * (base_prompt_tokens + |O_k|_tok))

    Keys may be result sizes k or token budgets B. A ``Candidate.c_db`` takes
    precedence over ``c_db``. Ties go to the smaller key.
    """
    if not candidates_by_k:
        raise ValueError("candidates_by_k is empty")
    rows: list[CostBreakdown] = []
    for k in sorted(candidates_by_k):
        cand = candidates_by_k[k]
        if not isinstance(cand, Candidate):
            cand = Candidate(hints=tuple(cand))
        if cand.c_db is not None:
            db = float(cand.c_db)
        elif callable(c_db):
            db = float(c_db(k))
        elif isinstance(c_db, Mapping):
            db = float(c_db[k])
        else:
            db = float(c_db)
        p1 = estimator.p1(cand.hints)
        e = float(expected_attempts(p1, estimator.cap))
        tok = o_tokens(cand.hints)
        per_attempt = c_agent_fixed + c_agent_per_token * (base_prompt_tokens + tok)
        llm = c_llm * cand.n_sem
        rows.append(
            CostBreakdown(
                k=k,
                retrieval_db=db,
                retrieval_llm=llm,
                p1=p1,
                expected_attempts=e,
                o_tokens=tok,
                agent_cost_per_attempt=per_attempt,
                downstream=e * per_attempt,
                total=db + llm + e * per_attempt,
            )
        )
    best = min(rows, key=lambda r: (r.total, r.k))
    return KChoice(k=best.k, best=best, table=tuple(rows))


# ------------------------------------------------------------------ EXG loader


@dataclass(frozen=True)
class TaskRow:
    """One training row: the logged O for attempt 1, plus outcomes."""

    task_id: str
    sequence: int
    hints: tuple[Hint, ...]
    first_success: int
    attempts_used: int
    retried: int


def _hints_from_memory(memory: dict, tokens_per_char: float) -> tuple[Hint, ...]:
    seeds = {int(s["id"]): float(s["cosine"]) for s in memory.get("query_seeds", [])}
    out = []
    for h in memory.get("hints", []):
        prov = h.get("provenance") or {}
        sim = seeds.get(prov.get("id"), seeds.get(prov.get("seed")))
        chars = len(json.dumps(h, ensure_ascii=False))
        out.append(
            Hint(
                case_uid=str(h["case_uid"]),
                status=str(h["status"]),
                repair_case_uid=h.get("repair_case_uid"),
                similarity=sim,
                tokens=chars * tokens_per_char,
            )
        )
    return tuple(out)


def _tokens_per_char(trajectory: Path, messages: list) -> float:
    """Per-row token/char ratio from the logged prompt token count (guard.count)."""
    chars = sum(len(m.get("content", "")) for m in messages)
    try:
        count = json.loads(trajectory.read_text())["call"]["guard"]["count"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        count = None
    if not count or not chars:
        return 1.0 / CHARS_PER_TOKEN_FALLBACK
    return float(count) / chars


def load_exg_rows(dataset_dir: Path) -> tuple[list[TaskRow], dict]:
    """Load attempt-1 rows of one EXG dataset; return (rows, counts of exclusions)."""
    stats = {"completed": 0, "infrastructure": 0, "self_hint_leak": 0, "missing": 0}
    rows: list[TaskRow] = []
    for done in sorted(dataset_dir.glob("tasks/*/*/completed.json")):
        stats["completed"] += 1
        comp = json.loads(done.read_text())
        attempts = comp["attempts"]
        first = attempts[0]["status"]
        if first not in ("golden", "warning"):
            stats["infrastructure"] += 1
            continue
        a1 = done.parent / "attempt_1"
        try:
            prep = json.loads((a1 / "prepared.json").read_text())
        except (OSError, json.JSONDecodeError):
            stats["missing"] += 1
            continue
        memory = prep["memory"]
        if any(
            h.get("source_task") == comp["task_id"] for h in memory.get("hints", [])
        ):
            stats["self_hint_leak"] += 1
            continue
        tpc = _tokens_per_char(a1 / "trajectory.json", prep.get("messages", []))
        rows.append(
            TaskRow(
                task_id=comp["task_id"],
                sequence=int(comp["sequence"]),
                hints=_hints_from_memory(memory, tpc),
                first_success=int(first == "golden"),
                attempts_used=len(attempts),
                retried=int(len(attempts) > 1),
            )
        )
    return rows, stats


def split_rows(
    rows: Sequence[TaskRow],
    seed: int = DEFAULT_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[list[TaskRow], list[TaskRow]]:
    """EXG-style split: sort by task id, Random(seed).shuffle, first 70% train."""
    ordered = sorted(rows, key=lambda r: r.task_id)
    random.Random(seed).shuffle(ordered)
    cut = int(round(train_fraction * len(ordered)))
    return ordered[:cut], ordered[cut:]


# ------------------------------------------------------------------ evaluation


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Return the mean squared error of probabilities."""
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Return the equal-width reliability table and the expected calibration error."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(p) * abs(conf - acc)
        table.append({"bin": b, "n": int(m.sum()), "mean_pred": conf, "frac_pos": acc})
    return {"ece": float(ece), "reliability": table}


def paired_bootstrap(
    p_model: np.ndarray, p_base: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 0
) -> dict:
    """Return the 95% CI of Brier(base) - Brier(model) (positive = model better)."""
    rng = np.random.default_rng(seed)
    d = (p_base - y) ** 2 - (p_model - y) ** 2
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {
        "brier_gain": float(d.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def _head_report(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, l2: float
) -> tuple[LogisticModel, dict]:
    model = fit_logistic(x_tr, y_tr, l2=l2)
    p_m = model.predict_proba(x_te)
    base = float(y_tr.mean())
    p_b = np.full_like(p_m, base)
    return model, {
        "train_rate": base,
        "test_rate": float(y_te.mean()),
        "brier_model": brier(p_m, y_te),
        "brier_baseline": brier(p_b, y_te),
        "brier_skill": 1.0 - brier(p_m, y_te) / max(brier(p_b, y_te), 1e-12),
        "bootstrap": paired_bootstrap(p_m, p_b, y_te),
        "calibration_model": calibration(p_m, y_te),
        "calibration_baseline": calibration(p_b, y_te),
        "coef_z": dict(zip(FEATURE_NAMES, model.coef, strict=True)),
        "intercept": model.intercept,
    }


def _choose_k_demo(est: AttemptEstimator, test: Sequence[TaskRow], costs: dict) -> dict:
    """Prefix O_k = first k logged hints, k = 0..5. Shows predicted cost only."""
    picks: dict[int, int] = {}
    tot_choice = tot_fixed = 0.0
    for r in test:
        cands = {k: Candidate(hints=r.hints[:k]) for k in range(0, len(r.hints) + 1)}
        res = choose_k(cands, estimator=est, **costs)
        picks[int(res.k)] = picks.get(int(res.k), 0) + 1
        tot_choice += res.best.total
        tot_fixed += res.table[-1].total
    n = max(len(test), 1)
    return {
        "costs_ms_illustrative": costs,
        "chosen_k_histogram": dict(sorted(picks.items())),
        "mean_predicted_cost_chosen": tot_choice / n,
        "mean_predicted_cost_full_O": tot_fixed / n,
        "note": "predicted cost under the estimator, not a measured rerun; O_k prefixes are counterfactual",
    }


DEMO_COSTS = {
    "c_db": 20.0,  # ms, EXG graph retrieval (notes: 18-22 ms)
    "c_llm": 0.0,  # EXG has no semantic operators in retrieval
    "c_agent_per_token": 0.05,  # ms per prompt token (assumed prefill rate)
    "c_agent_fixed": 2500.0,  # ms per agent call (assumed decode cost)
    "base_prompt_tokens": 500.0,
}


def evaluate_dataset(
    name: str, rows: list[TaskRow], stats: dict, l2: float, seed: int
) -> dict:
    """Run the 7:3 protocol on one dataset and return the report entry."""
    train, test = split_rows(rows, seed=seed)
    out: dict = {"dataset": name, "load": stats, "rows": len(rows), "train": len(train)}
    out["test"] = len(test)
    if not train or not test:
        out["credible"] = False
        out["reason"] = "no rows"
        return out
    x_tr = np.stack([features_of(r.hints) for r in train])
    x_te = np.stack([features_of(r.hints) for r in test])
    y_tr = np.array([r.first_success for r in train], dtype=float)
    y_te = np.array([r.first_success for r in test], dtype=float)
    minority = int(min(y_te.sum(), len(y_te) - y_te.sum()))
    out["test_minority_labels"] = minority
    out["credible"] = bool(len(test) >= MIN_TEST_ROWS and minority >= MIN_TEST_MINORITY)
    if not out["credible"]:
        out["reason"] = (
            f"held-out has {len(test)} rows and {minority} minority labels "
            f"(need >= {MIN_TEST_ROWS} and >= {MIN_TEST_MINORITY}); numbers below are not credible"
        )
    if len(set(y_tr.tolist())) < 2:
        out["reason"] = "training labels are constant; no model fitted"
        return out
    model, out["p1_head"] = _head_report(x_tr, y_tr, x_te, y_te, l2)
    est = AttemptEstimator(model, cap=EXG_ATTEMPT_CAP)
    e_hat = expected_attempts(model.predict_proba(x_te), EXG_ATTEMPT_CAP)
    used = np.array([r.attempts_used for r in test], dtype=float)
    out["expected_attempts"] = {
        "formula": "E = 1 + (1 - p1) (cap 2)",
        "mean_predicted": float(e_hat.mean()),
        "mean_retry_on_failure_rule": float((2 - y_te).mean()),
        "mean_observed_attempts_used": float(used.mean()),
        "mae_vs_retry_on_failure_rule": float(np.abs(e_hat - (2 - y_te)).mean()),
        "mae_vs_observed": float(np.abs(e_hat - used).mean()),
        "mae_baseline_vs_observed": float(np.abs(1 + (1 - y_tr.mean()) - used).mean()),
    }
    r_tr = np.array([r.retried for r in train], dtype=float)
    r_te = np.array([r.retried for r in test], dtype=float)
    if len(set(r_tr.tolist())) == 2:
        _, out["retry_head"] = _head_report(x_tr, r_tr, x_te, r_te, l2)
    out["feature_means_test"] = dict(
        zip(FEATURE_NAMES, x_te.mean(axis=0).tolist(), strict=True)
    )
    out["choose_k_demo"] = _choose_k_demo(est, test, DEMO_COSTS)
    return out


DATASETS = ("humaneval", "humaneval_plus", "musique", "hotpotqa")


def run(data: Path, datasets: Iterable[str], l2: float, seed: int) -> dict:
    """Evaluate every available dataset under ``data``."""
    report: dict = {
        "protocol": {
            "split": f"sorted task ids, Random({seed}).shuffle, 7:3",
            "rows": "attempt-1 of each completed task; O = logged prepared.json hints",
            "label": "attempt-1 status golden (1) vs warning (0)",
            "model": f"L2 logistic (l2={l2}) by IRLS on standardised features",
            "features": list(FEATURE_NAMES),
            "credible_rule": f"test >= {MIN_TEST_ROWS} rows and >= {MIN_TEST_MINORITY} minority",
            "attempt_cap": EXG_ATTEMPT_CAP,
        },
        "datasets": {},
    }
    for name in datasets:
        d = data / name
        if not d.is_dir():
            report["datasets"][name] = {
                "credible": False,
                "reason": "directory missing",
            }
            continue
        rows, stats = load_exg_rows(d)
        report["datasets"][name] = evaluate_dataset(name, rows, stats, l2, seed)
    return report


def _summary_line(name: str, e: dict) -> str:
    h = e.get("p1_head")
    if not h:
        return f"{name}: {e.get('reason', 'no model')}"
    flag = "" if e.get("credible") else "  [NOT CREDIBLE]"
    ci = h["bootstrap"]["ci95"]
    return (
        f"{name}: n_test={e['test']} brier model={h['brier_model']:.4f} "
        f"base={h['brier_baseline']:.4f} gain CI95=[{ci[0]:.4f},{ci[1]:.4f}] "
        f"ECE {h['calibration_model']['ece']:.3f} vs {h['calibration_baseline']['ece']:.3f}{flag}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--data", type=Path, default=Path("exg/results/experience_graph_v0.1.0")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("results/fig1_semantic_v0.1.0/downstream.json")
    )
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    report = run(args.data, args.datasets, args.l2, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    for name, entry in report["datasets"].items():
        print(_summary_line(name, entry))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AttemptEstimator",
    "Candidate",
    "CostBreakdown",
    "Hint",
    "KChoice",
    "LogisticModel",
    "TaskRow",
    "choose_k",
    "expected_attempts",
    "features_of",
    "fit_estimator",
    "fit_logistic",
    "load_exg_rows",
    "split_rows",
]
