"""Legacy dense log-cost/lambda ablation (training panel only).

All three BRANE predictors are query-dependent by default. This harness retains
the historical scalarization; current sparse experiment drivers use the raw-cost
three-weight utility in ``sparse.weight_curve`` and ``runtime.choose_utility``.

Outer K-fold CV over queries. In every fold the feature transform, the candidate set, the
predictors and lambda are fitted on the training queries of the fold only (lambda by an inner CV
that refits the feature transform in every inner fold); the held-out queries are routed and their
*measured* outcome under the routed configuration is read from the dense matrix (every configuration
ran every panel query, so no counterfactual is simulated). Router overhead of the enabled feature
groups (probe lookup, LLM labeling) and the measured transform + prediction time are added to the
routed tokens and latency.

References: the best single configuration under the same scalarization fitted on the fold's
training queries (what a non-routing deployment would use), the Q-best single configuration, the
per-query oracle and the uniform-random candidate.

Limitation: the configurations (matrix columns) were proposed by the BO using all 100 panel queries,
so this CV is conditional on that adaptively chosen set; it does not re-run the BO inside folds.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trimem.optimizer.query_router import brane, features, runtime
from trimem.optimizer.query_router.matrix import OutcomeMatrix, fingerprint

LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
FitFn = Callable[[np.ndarray, OutcomeMatrix], dict]
Weights = tuple[float, float]


@dataclass(frozen=True)
class FeatureConfig:
    groups: tuple[str, ...]
    emb_dims: int = 4
    llm_questions: list[str] | None = None
    llm_labeler: dict | None = None

    def spec(self, records: list[dict]) -> dict:
        return features.fit_spec(
            records,
            self.groups,
            emb_dims=self.emb_dims,
            llm_questions=self.llm_questions,
            llm_labeler=self.llm_labeler,
        )


@dataclass
class Fitted:
    spec: dict
    model: dict
    extra: dict = field(default_factory=dict)


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def fit(
    fc: FeatureConfig, fit_fn: FitFn, records: list[dict], m: OutcomeMatrix
) -> Fitted:
    spec = fc.spec(records)
    return Fitted(spec, fit_fn(features.design(spec, records), m))


def predict(f: Fitted, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Predictions (n, candidates, 3: quality, ln tokens, ln latency) and the per-query transform +
    prediction time in ms (router inference overhead)."""
    out, ms = [], []
    for r in records:
        t0 = time.perf_counter()
        x = runtime.transform(f.spec, r)
        out.append(np.stack(runtime.predict(f.model, x), axis=-1))
        ms.append((time.perf_counter() - t0) * 1000)
    return np.array(out), np.array(ms)


def route_columns(f: Fitted, pred: np.ndarray, lam: float, w: Weights) -> np.ndarray:
    s = runtime.scores(pred[..., 0], pred[..., 1], pred[..., 2], lam, *w)
    return np.asarray(f.model["candidate_columns"])[np.argmax(s, axis=1)]


def single_column(m: OutcomeMatrix, lam: float, w: Weights) -> int:
    """Best non-routed configuration on ``m``: max mean quality - lambda * cost (ties: cheaper)."""
    acc, cost = m.quality.mean(0), brane.config_cost(m.tokens, m.latency_ms, *w)
    s = acc - lam * cost
    best = np.flatnonzero(s >= s.max() - 1e-12)
    return int(best[np.argmin(cost[best])])


def realized(
    m: OutcomeMatrix,
    cols: np.ndarray,
    ovh_tok: np.ndarray | float = 0.0,
    ovh_ms: np.ndarray | float = 0.0,
) -> dict[str, np.ndarray]:
    r = np.arange(len(cols))
    return {
        "quality": m.quality[r, cols],
        "gain": m.gain[r, cols],
        "tokens": m.tokens[r, cols] + ovh_tok,
        "latency_ms": m.latency_ms[r, cols] + ovh_ms,
        "column": np.asarray(cols, dtype=float),
    }


def summary(
    per_query: dict[str, np.ndarray], w: Weights = (0.5, 0.5)
) -> dict[str, float]:
    q = per_query["quality"]
    out = {
        "quality": float(q.mean()),
        "quality_se": float(q.std(ddof=1) / math.sqrt(len(q))) if len(q) > 1 else 0.0,
        "gain": float(per_query["gain"].mean()),
        "tokens": float(per_query["tokens"].mean()),
        "latency_ms": float(per_query["latency_ms"].mean()),
    }
    out["cost"] = w[0] * math.log(out["tokens"]) + w[1] * math.log(out["latency_ms"])
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = a - b
    return {
        "mean": float(d.mean()),
        "se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0,
    }


def fixed_lambda(rule: str) -> float | None:
    return float(rule.split(":", 1)[1]) if rule.startswith("fixed:") else None


def pick_lambda(curve: list[dict], rule: str) -> float:
    """``knee``: the cheapest lambda whose routed quality is within one SE of the best;
    ``quality``: the highest routed quality (ties: larger lambda); ``fixed:<v>``: v."""
    lam = fixed_lambda(rule)
    if lam is not None:
        return lam
    best = max(c["quality"] for c in curve)
    if rule == "quality":
        return max(c["lambda"] for c in curve if c["quality"] >= best - 1e-12)
    if rule == "knee":
        se = next(c["quality_se"] for c in curve if c["quality"] >= best - 1e-12)
        near = [c for c in curve if c["quality"] >= best - se]
        return min(near, key=lambda c: (c["cost"], -c["lambda"]))["lambda"]
    raise ValueError(f"unknown lambda rule {rule!r}")


def inner_curve(
    fc: FeatureConfig,
    fit_fn: FitFn,
    records: list[dict],
    m: OutcomeMatrix,
    *,
    lambdas: tuple[float, ...],
    w: Weights,
    k: int,
    seed: int,
) -> list[dict]:
    """Routed quality / cost per lambda by K-fold CV inside the given (training) queries; the feature
    transform is refitted in every inner fold."""
    ovh_tok, ovh_ms = features.overhead(records, fc.groups)
    cols = {lam: np.zeros(len(records), dtype=int) for lam in lambdas}
    inf_ms = np.zeros(len(records))
    for te in folds(len(records), k, seed + 1):
        tr = np.setdiff1d(np.arange(len(records)), te)
        f = fit(fc, fit_fn, [records[i] for i in tr], m.subset(tr))
        pred, inf_ms[te] = predict(f, [records[i] for i in te])
        for lam in lambdas:
            cols[lam][te] = route_columns(f, pred, lam, w)
    return [
        {"lambda": lam, **summary(realized(m, cols[lam], ovh_tok, ovh_ms + inf_ms), w)}
        for lam in lambdas
    ]


def oof_curve(
    f: Fitted,
    records: list[dict],
    m: OutcomeMatrix,
    groups: tuple[str, ...],
    *,
    lambdas: tuple[float, ...],
    w: Weights,
) -> list[dict]:
    """Preference replay using nested out-of-fold predictions of all three targets.

    Legacy mean-cost artifacts may contain only quality OOF predictions.
    """
    pred, inf_ms = predict(f, records)
    pred[..., 0] = np.asarray(f.model["oof_quality"])
    for column, name in ((1, "oof_ln_tokens"), (2, "oof_ln_latency")):
        if name in f.model:
            pred[..., column] = np.asarray(f.model[name])
        elif f.model.get("params", {}).get("cost_model") == "query":
            raise ValueError("Query-cost replay requires OOF cost predictions")
    ovh_tok, ovh_ms = features.overhead(records, groups)
    return [
        {
            "lambda": lam,
            **summary(
                realized(m, route_columns(f, pred, lam, w), ovh_tok, ovh_ms + inf_ms), w
            ),
        }
        for lam in lambdas
    ]


def choose_lambda(
    fc, fit_fn, records, m, *, lambdas, w, k, seed, rule, fitted: Fitted | None = None
) -> tuple[float, list[dict]]:
    lam = fixed_lambda(rule)
    if lam is not None:  # no curve needed
        return lam, []
    if fitted is not None and "oof_quality" in fitted.model:
        curve = oof_curve(fitted, records, m, fc.groups, lambdas=lambdas, w=w)
    else:
        curve = inner_curve(
            fc, fit_fn, records, m, lambdas=lambdas, w=w, k=k, seed=seed
        )
    return pick_lambda(curve, rule), curve


def cross_validate(
    m: OutcomeMatrix,
    records: list[dict],
    fit_fn: FitFn,
    fc: FeatureConfig,
    *,
    k: int = 5,
    inner_k: int = 4,
    seed: int = 0,
    lambdas: tuple[float, ...] = LAMBDAS,
    rule: str = "knee",
    w: Weights = (0.5, 0.5),
) -> dict[str, Any]:
    if [r["query_id"] for r in records] != list(m.query_ids):
        raise ValueError("feature records must follow the matrix's query order")
    n = len(records)
    ovh_tok, ovh_ms = features.overhead(records, fc.groups)
    arms: dict[str, dict[str, np.ndarray]] = {}

    def put(name: str, te: np.ndarray, res: dict[str, np.ndarray]) -> None:
        a = arms.setdefault(
            name,
            {
                f: np.zeros(n)
                for f in ("quality", "gain", "tokens", "latency_ms", "column")
            },
        )
        for f in a:
            a[f][te] = res[f]

    chosen, fold_info = [], []
    for fi, te in enumerate(folds(n, k, seed)):
        tr = np.setdiff1d(np.arange(n), te)
        rtr, rte = [records[i] for i in tr], [records[i] for i in te]
        mtr, mte = m.subset(tr), m.subset(te)
        f = fit(fc, fit_fn, rtr, mtr)
        lam_star, curve = choose_lambda(
            fc,
            fit_fn,
            rtr,
            mtr,
            lambdas=lambdas,
            w=w,
            k=inner_k,
            seed=seed + fi,
            rule=rule,
            fitted=f,
        )
        chosen.append(lam_star)
        pred, inf_ms = predict(f, rte)
        o_tok, o_ms = ovh_tok[te], ovh_ms[te] + inf_ms
        for lam in lambdas:
            put(
                f"router@{lam}",
                te,
                realized(mte, route_columns(f, pred, lam, w), o_tok, o_ms),
            )
            put(
                f"single@{lam}",
                te,
                realized(mte, np.full(len(te), single_column(mtr, lam, w))),
            )
        put(
            "router@selected",
            te,
            realized(mte, route_columns(f, pred, lam_star, w), o_tok, o_ms),
        )
        put(
            "single@selected",
            te,
            realized(mte, np.full(len(te), single_column(mtr, lam_star, w))),
        )
        put(
            "single@qbest",
            te,
            realized(mte, np.full(len(te), single_column(mtr, 0.0, w))),
        )
        cost = np.log(mte.tokens) * w[0] + np.log(mte.latency_ms) * w[1]
        orc = np.array(
            [
                min(
                    np.flatnonzero(mte.quality[i] >= mte.quality[i].max()),
                    key=lambda c: cost[i, c],
                )
                for i in range(len(te))
            ]
        )
        put("oracle", te, realized(mte, orc))
        cands = np.array(f.model["candidate_columns"])
        put(
            "random_candidate",
            te,
            {
                "quality": mte.quality[:, cands].mean(1),
                "gain": mte.gain[:, cands].mean(1),
                "tokens": mte.tokens[:, cands].mean(1),
                "latency_ms": mte.latency_ms[:, cands].mean(1),
                "column": np.full(len(te), -1.0),
            },
        )
        fold_info.append(
            {
                "fold": fi,
                "n_test": len(te),
                "lambda": lam_star,
                "inner_curve": curve,
                "predictors": [x.get("family") for x in f.model.get("selection", [])],
                "inference_ms": float(inf_ms.mean()),
                "candidates": [m.configs[c] for c in f.model["candidate_columns"]],
            }
        )
    out = {
        "n_queries": n,
        "k": k,
        "seed": seed,
        "rule": rule,
        "w": list(w),
        "groups": list(fc.groups),
        "selected_lambdas": chosen,
        "folds": fold_info,
        "arms": {a: summary(v, w) for a, v in arms.items()},
        "router_vs_single": {
            f"{lam}": paired(
                arms[f"router@{lam}"]["quality"], arms[f"single@{lam}"]["quality"]
            )
            for lam in lambdas
        },
        "overhead": {"tokens": float(ovh_tok.mean()), "ms": float(ovh_ms.mean())},
    }
    out["selected_vs_qbest"] = paired(
        arms["router@selected"]["quality"], arms["single@qbest"]["quality"]
    )
    out["selected_vs_single_same_lambda"] = paired(
        arms["router@selected"]["quality"], arms["single@selected"]["quality"]
    )
    col = arms["router@selected"]["column"].astype(int)
    out["routing_share"] = {
        str(m.configs[c]): int((col == c).sum()) for c in np.unique(col)
    }
    out["per_query"] = {
        a: {f: v.tolist() for f, v in arr.items()}
        for a, arr in arms.items()
        if a in ("router@selected", "single@qbest", "oracle")
    }
    return out


def sweep(
    f: Fitted,
    records: list[dict],
    m: OutcomeMatrix,
    groups: tuple[str, ...],
    *,
    lambdas: tuple[float, ...],
    w: Weights = (0.5, 0.5),
) -> dict[float, dict[str, np.ndarray]]:
    """Per-query measured outcome of a fitted router on ``records`` / ``m`` for every lambda (no refit).
    Overhead of the enabled feature groups and the measured transform + prediction time are added to
    tokens / latency, as in ``cross_validate``."""
    if [r["query_id"] for r in records] != list(m.query_ids):
        raise ValueError("feature records must follow the matrix's query order")
    pred, inf_ms = predict(f, records)
    ovh_tok, ovh_ms = features.overhead(records, groups)
    return {
        lam: realized(m, route_columns(f, pred, lam, w), ovh_tok, ovh_ms + inf_ms)
        for lam in lambdas
    }


def oof_sweep(
    m: OutcomeMatrix,
    records: list[dict],
    fit_fn: FitFn,
    fc: FeatureConfig,
    *,
    k: int = 5,
    seed: int = 0,
    lambdas: tuple[float, ...] = LAMBDAS,
    w: Weights = (0.5, 0.5),
) -> dict[str, Any]:
    """Out-of-fold lambda sweep (one K-fold repeat): in every outer fold the feature transform and the
    router are fitted on the training folds and the held-out queries are routed with EVERY lambda.
    Returns ``arms`` (lambda -> per-query arrays; every query is held out exactly once) and, per fold,
    the held-out indices and the training-side curve (``oof_curve``: measured outcome on the training
    folds of routing by nested OOF predictions), which lets a caller choose lambda inside
    the fold (e.g. under a budget) without looking at the held-out queries."""
    if [r["query_id"] for r in records] != list(m.query_ids):
        raise ValueError("feature records must follow the matrix's query order")
    n = len(records)
    arms = {
        lam: {
            f: np.zeros(n)
            for f in ("quality", "gain", "tokens", "latency_ms", "column")
        }
        for lam in lambdas
    }
    out = []
    for te in folds(n, k, seed):
        tr = np.setdiff1d(np.arange(n), te)
        rtr, mtr = [records[i] for i in tr], m.subset(tr)
        f = fit(fc, fit_fn, rtr, mtr)
        res = sweep(
            f, [records[i] for i in te], m.subset(te), fc.groups, lambdas=lambdas, w=w
        )
        for lam in lambdas:
            for fld, v in res[lam].items():
                arms[lam][fld][te] = v
        curve = (
            oof_curve(f, rtr, mtr, fc.groups, lambdas=lambdas, w=w)
            if "oof_quality" in f.model
            else None
        )
        out.append(
            {
                "test": te.tolist(),
                "train_curve": curve,
                "candidates": [m.configs[c] for c in f.model["candidate_columns"]],
            }
        )
    return {"k": k, "seed": seed, "arms": arms, "folds": out}


def gate(
    diff: dict[str, float],
    routed_cost: float,
    single_cost: float,
    cost_slack: float = 0.05,
) -> str:
    """Route only if the routed quality beats the Q-best single configuration by more than one SE
    and the routed scalar cost is not higher (up to ``cost_slack`` in log units)."""
    return (
        "route"
        if diff["mean"] > diff["se"] and routed_cost <= single_cost + cost_slack
        else "single"
    )


def repeated(runs: list[dict[str, Any]], m: OutcomeMatrix) -> dict[str, Any]:
    """Pool repeated K-fold CV runs (different fold seeds)."""
    d = np.array([r["selected_vs_qbest"]["mean"] for r in runs])
    se = float(np.mean([r["selected_vs_qbest"]["se"] for r in runs]))
    arms = {
        a: {
            f: float(np.mean([r["arms"][a][f] for r in runs]))
            for f in runs[0]["arms"][a]
        }
        for a in runs[0]["arms"]
    }
    diff = {"mean": float(d.mean()), "se": se, "per_repeat": d.tolist()}
    return {
        "repeats": len(runs),
        "matrix_fingerprint": fingerprint(m),
        "selected_vs_qbest": diff,
        "arms": arms,
        "gate": gate(
            diff, arms["router@selected"]["cost"], arms["single@qbest"]["cost"]
        ),
        "selected_lambdas": [r["selected_lambdas"] for r in runs],
    }


def train(
    m: OutcomeMatrix,
    records: list[dict],
    fit_fn: FitFn,
    fc: FeatureConfig,
    *,
    dataset: str,
    policy: str,
    inner_k: int = 5,
    seed: int = 0,
    lambdas: tuple[float, ...] = LAMBDAS,
    rule: str = "knee",
    w: Weights = (0.5, 0.5),
    provenance: dict | None = None,
) -> dict[str, Any]:
    """Router artifact fitted on every training query (lambda by inner CV). ``policy`` is the CV gate's
    decision: ``single`` deploys the matched single configuration without computing features."""
    if policy not in ("route", "single"):
        raise ValueError("policy must be route or single")
    f = fit(fc, fit_fn, records, m)
    lam, curve = choose_lambda(
        fc,
        fit_fn,
        records,
        m,
        lambdas=lambdas,
        w=w,
        k=inner_k,
        seed=seed,
        rule=rule,
        fitted=f,
    )
    fallback = m.configs[single_column(m, lam, w)]
    return {
        "kind": runtime.KIND,
        "version": runtime.VERSION,
        "dataset": dataset,
        "feature_spec": f.spec,
        "model": f.model,
        "routing": {"lambda": lam, "w_tokens": w[0], "w_latency": w[1], "rule": rule},
        "policy": {"mode": policy, "fallback": fallback},
        "trained_on": {
            "queries": list(m.query_ids),
            "matrix_fingerprint": fingerprint(m),
            **(provenance or {}),
        },
        "report": {"inner_curve": curve},
    }
