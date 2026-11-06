"""BRANE router (arXiv 2605.27361), adapted to agent-memory retrieval configurations.

Training input: standardized query features ``X`` (Q, d) and the dense outcome matrix of the same
queries. Output: a JSON-serializable model for ``runtime.predict_brane``.

BRANE supplies the catalog and model-family recipe (section 4, appendix A.4).
The query-cost models and nested cross-fitting below extend that recipe:

1. Fuzzy Pareto pruning (section 4.4): on the training panel means, a configuration ``c'`` is retained
   whenever some strict-frontier vertex ``c*`` satisfies ``y(c*) - y(c') <= tau_acc`` and
   ``cost(c') <= (1 + tau_cost) cost(c*)`` (tau_acc = 0.02, tau_cost = 0.10). Cost here is the
   geometric combination ``exp(w_T ln tokens + w_L ln latency)`` of the panel means.
2. Feature cleaning: constant features and one of any pair with |correlation| > 0.99 are dropped.
3. Three query-dependent predictors per retained configuration, chosen by model selection over logistic
   regression, decision tree, random forest, gradient boosting, XGBoost and LightGBM (Optuna, 30 trials,
   3-fold inner CV, negative log-loss; ``tabular``). ScienceWorld's continuous score uses the regressor
   counterparts and squared error.
4. Costs use regressors on RAW per-query tokens / latency, with a positive prediction floor.
   ``cost_model="mean"`` and ``"ridge"`` retain historical ablations. Raw-cost min-max bounds
   come from observed query costs of retained configurations and remain frozen at deployment.
5. Nested query cross-fitting reruns inner model selection inside five outer folds. All three
   out-of-fold predictions feed preference selection; final models are refitted on all training rows.

``fit_linear`` (the earlier L2-logistic variant) is kept for comparison runs.
"""

from __future__ import annotations

import math

import numpy as np

from trimem.optimizer.query_router import predictors, runtime, tabular
from trimem.optimizer.query_router.matrix import OutcomeMatrix

TAU_ACC, TAU_COST = 0.02, 0.10
CORR_DROP = 0.99


def config_cost(
    tokens: np.ndarray,
    latency_ms: np.ndarray,
    w_tokens: float = 0.5,
    w_latency: float = 0.5,
) -> np.ndarray:
    """Scalar log cost of panel means per configuration (columns)."""
    return w_tokens * np.log(tokens.mean(0)) + w_latency * np.log(latency_ms.mean(0))


def strict_front(acc: np.ndarray, cost: np.ndarray) -> list[int]:
    return [
        i
        for i in range(len(acc))
        if not any(
            acc[j] >= acc[i]
            and cost[j] <= cost[i]
            and (acc[j] > acc[i] or cost[j] < cost[i])
            for j in range(len(acc))
        )
    ]


def fuzzy_pareto(
    acc: np.ndarray,
    cost: np.ndarray,
    *,
    tau_acc: float = TAU_ACC,
    tau_cost: float = TAU_COST,
    max_candidates: int | None = None,
) -> list[int]:
    """BRANE fuzzy Pareto set (``cost`` in log units), ordered by mean quality (desc), then cost (asc)."""
    front = strict_front(acc, cost)
    margin = math.log(1 + tau_cost)
    keep = [
        i
        for i in range(len(acc))
        if any(
            acc[s] - acc[i] <= tau_acc and cost[i] <= cost[s] + margin for s in front
        )
    ]
    keep.sort(key=lambda i: (-acc[i], cost[i], i))
    return keep[:max_candidates] if max_candidates else keep


def best_per_plan(m: OutcomeMatrix, acc: np.ndarray, cost: np.ndarray) -> list[int]:
    """Relaxed pruning (user 2026-10-17): the best configuration of every plan (highest mean quality, then
    cheapest), so the router can send some queries to plans the fuzzy front drops (e.g. SemFilter)."""
    best: dict[str, int] = {}
    for i, c in enumerate(m.configs):
        p = c["plan"]
        if p not in best or (acc[i], -cost[i]) > (acc[best[p]], -cost[best[p]]):
            best[p] = i
    return sorted(best.values(), key=lambda i: (-acc[i], cost[i], i))


def feature_mask(X: np.ndarray, corr: float = CORR_DROP) -> list[int]:
    """Indices of kept features: non-constant, and not |corr| > ``corr`` with an earlier kept one."""
    keep: list[int] = []
    for j in range(X.shape[1]):
        if np.ptp(X[:, j]) < 1e-12:
            continue
        if any(abs(np.corrcoef(X[:, j], X[:, k])[0, 1]) > corr for k in keep):
            continue
        keep.append(j)
    return keep


def quality_kind(m: OutcomeMatrix) -> str:
    return "binary" if set(np.unique(m.quality)) <= {0.0, 1.0} else "continuous"


def ridge(X: np.ndarray, y: np.ndarray, l2: float = 10.0) -> dict:
    if np.ptp(y) < 1e-12:
        return {"const": float(y.mean())}
    mu, xm = y.mean(), X.mean(0)
    Xc = X - xm
    w = np.linalg.solve(Xc.T @ Xc + l2 * np.eye(X.shape[1]), Xc.T @ (y - mu))
    return {"w": w.tolist(), "b": float(mu - xm @ w)}


def fit(
    X: np.ndarray,
    m: OutcomeMatrix,
    *,
    w_tokens: float = 0.5,
    w_latency: float = 0.5,
    tau_acc: float = TAU_ACC,
    tau_cost: float = TAU_COST,
    max_candidates: int | None = None,
    families: tuple[str, ...] = tabular.FAMILIES,
    trials: int = tabular.TRIALS,
    inner_k: int = tabular.INNER_K,
    cost_model: str = "query",
    outer_k: int = predictors.OUTER_K,
    n_jobs: int = 8,
    seed: int = 0,
    candidates: list[int] | None = None,
    keep_best_per_plan: bool = True,
    utility_norm: str = runtime.SCALE_ZLOG,
) -> dict:
    if len(X) != len(m.query_ids):
        raise ValueError("features and outcome rows differ")
    if len(set(m.query_ids)) != len(m.query_ids):
        raise ValueError(
            "Aggregate repeated trials into one row per query before fitting"
        )
    acc = m.quality.mean(0)
    cost = config_cost(m.tokens, m.latency_ms, w_tokens, w_latency)
    cand = (
        candidates
        if candidates is not None
        else fuzzy_pareto(
            acc, cost, tau_acc=tau_acc, tau_cost=tau_cost, max_candidates=max_candidates
        )
    )
    if keep_best_per_plan and candidates is None:
        cand = cand + [c for c in best_per_plan(m, acc, cost) if c not in cand]
    keep = feature_mask(X)
    Xk = X[:, keep]
    binary = quality_kind(m) == "binary"
    options = dict(
        binary=binary,
        families=families,
        trials=trials,
        inner_k=inner_k,
        outer_k=outer_k,
        seed=seed,
        cost_model=cost_model,
    )
    jobs = [
        (Xk, m.quality[:, c], m.tokens[:, c], m.latency_ms[:, c], options) for c in cand
    ]
    if n_jobs > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(n_jobs, len(jobs))) as ex:
            sel = list(ex.map(predictors.fit_job, jobs))
    else:
        sel = [predictors.fit_job(j) for j in jobs]
    oof = np.asarray([s["oof"] for s in sel]).transpose(1, 0, 2)
    if (
        utility_norm == runtime.SCALE_ZLOG
    ):  # user 2026-10-22: equal-influence scale on the dense OOF predictions
        lt, ll = (
            np.log(np.maximum(m.tokens[:, cand], 1e-6)),
            np.log(np.maximum(m.latency_ms[:, cand], 1e-6)),
        )
        cols = [m.quality[:, cand], lt, ll]
        if cost_model == "mean":
            oof = oof.copy()
            oof[:, :, 1], oof[:, :, 2] = (
                np.log(m.tokens[:, cand].mean(0)),
                np.log(m.latency_ms[:, cand].mean(0)),
            )
        norm = runtime.utility_norm(
            oof,
            np.stack([v.mean(0) for v in cols], axis=1),
            np.stack(
                [
                    v.std(0, ddof=1) / np.sqrt(len(v))
                    if len(v) > 1
                    else np.zeros(len(cand))
                    for v in cols
                ],
                axis=1,
            ),
        )
    elif utility_norm == runtime.SCALE_MINMAX:
        norm = runtime.cost_norm(
            m.tokens[:, cand].mean(0) if cost_model == "mean" else m.tokens[:, cand],
            m.latency_ms[:, cand].mean(0)
            if cost_model == "mean"
            else m.latency_ms[:, cand],
        )
    else:
        raise ValueError(f"unknown utility_norm {utility_norm!r}")
    return tabular.as_json(
        {
            "kind": "brane",
            "variant": "paper",
            "quality_kind": "binary" if binary else "continuous",
            "configs": [m.configs[c] for c in cand],
            "candidate_columns": list(cand),
            "x_keep": keep,
            "quality": [s["quality"] for s in sel],
            "ln_tokens": [s["ln_tokens"] for s in sel],
            "ln_latency": [s["ln_latency"] for s in sel],
            "selection": [s["selection"]["quality"] for s in sel],
            "selection_by_objective": {
                name: [s["selection"][name] for s in sel]
                for name in predictors.OBJECTIVES
            },
            "oof_quality": oof[:, :, 0].tolist(),
            "oof_ln_tokens": oof[:, :, 1].tolist(),
            "oof_ln_latency": oof[:, :, 2].tolist(),
            "train_means": {"quality": acc[cand].tolist(), "cost": cost[cand].tolist()},
            "cost_norm": norm,
            "predictor_protocol": predictors.PROTOCOL,
            "params": {
                "tau_acc": tau_acc,
                "tau_cost": tau_cost,
                "max_candidates": max_candidates,
                "families": list(families),
                "trials": trials,
                "inner_k": inner_k,
                "cost_model": cost_model,
                "outer_k": outer_k,
                "cost_target": "raw" if cost_model == "query" else cost_model,
                "cost_norm_scope": "configuration-means"
                if cost_model == "mean"
                else "observed-query-costs",
                "corr_drop": CORR_DROP,
                "keep_best_per_plan": keep_best_per_plan,
            },
        }
    )


# ------------------------------------------------------------------ earlier linear variant (comparison)


def logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> dict:
    """L2-penalized logistic regression (intercept unpenalized), Newton-Raphson."""
    n = len(y)
    if y.min() == y.max():
        return {"const": float((y.sum() + 0.5) / (n + 1))}
    A = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(A.shape[1])
    pen = np.full(A.shape[1], l2)
    pen[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -40, 40)))
        g = A.T @ (p - y) + pen * w
        H = (
            (A * (p * (1 - p))[:, None]).T @ A
            + np.diag(pen)
            + 1e-9 * np.eye(A.shape[1])
        )
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"w": w[:-1].tolist(), "b": float(w[-1])}


def fit_linear(
    X: np.ndarray,
    m: OutcomeMatrix,
    *,
    w_tokens: float = 0.5,
    w_latency: float = 0.5,
    max_candidates: int = 8,
    l2_quality: float = 1.0,
    l2_cost: float = 10.0,
    candidates: list[int] | None = None,
) -> dict:
    """Earlier variant (router v0.1.0 artifact 4f8fd04f): one L2 logistic regression per candidate,
    per-query ridge costs, at most ``max_candidates`` candidates."""
    acc = m.quality.mean(0)
    cost = config_cost(m.tokens, m.latency_ms, w_tokens, w_latency)
    cand = (
        candidates
        if candidates is not None
        else fuzzy_pareto(acc, cost)[:max_candidates]
    )
    kind = quality_kind(m)
    qual = [
        logistic(X, m.quality[:, c], l2_quality)
        if kind == "binary"
        else ridge(X, m.quality[:, c], l2_quality)
        for c in cand
    ]
    return {
        "kind": "brane",
        "variant": "linear",
        "quality_kind": kind,
        "configs": [m.configs[c] for c in cand],
        "candidate_columns": list(cand),
        "quality": qual,
        "ln_tokens": [ridge(X, np.log(m.tokens[:, c]), l2_cost) for c in cand],
        "ln_latency": [ridge(X, np.log(m.latency_ms[:, c]), l2_cost) for c in cand],
        "train_means": {"quality": acc[cand].tolist(), "cost": cost[cand].tolist()},
        "params": {
            "max_candidates": max_candidates,
            "l2_quality": l2_quality,
            "l2_cost": l2_cost,
        },
    }
