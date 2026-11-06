"""Per-configuration quality and raw-cost models with nested query cross-fitting.

Inputs contain one aggregated row per query, never individual repeated trials.
The outer folds are shared by all three targets. Each outer fit reruns model
selection using only its training labels; final deployment models use all rows.
The feature representation is supplied by the caller's training-only transform.
"""

from __future__ import annotations

import numpy as np

from trimem.optimizer.query_router import tabular

PROTOCOL = "query-metrics-v1"
OUTER_K = 5
COST_FLOOR = 1e-6
OBJECTIVES = ("quality", "tokens", "latency_ms")


def cost_prediction(model: dict, x: np.ndarray, fallback: float) -> float:
    """Invalid extrapolations use the training-partition mean, never a free cost."""
    return float(
        tabular.project_prediction(
            tabular.predict(model, x), "cost", fallback, COST_FLOOR
        )
    )


def _fit(
    X,
    y,
    binary,
    families,
    trials,
    inner_k,
    seed,
    mode,
    mean=None,
    domain=None,
    strata=None,
    weights=None,
):
    if mode == "mean":
        return {
            "family": "const",
            "value": float(y.mean()) if mean is None else mean,
        }, {
            "family": "const",
            "mode": "mean",
        }
    if mode == "ridge":
        # Historical log-cost ridge ablation. The default fits raw costs.
        from trimem.optimizer.query_router.brane import ridge

        return {"family": "log-ridge", "model": ridge(X, np.log(y))}, {
            "family": "log-ridge"
        }
    return tabular.select_and_fit(
        X,
        y,
        binary=binary,
        families=families,
        trials=trials,
        inner_k=inner_k,
        seed=seed,
        prediction_domain=domain,
        sample_strata=strata,
        stratum_weights=weights,
    )


def _predict(model, X, cost, fallback):
    if model["family"] == "log-ridge":
        m = model["model"]
        return np.array(
            [
                np.exp(m["const"] if "const" in m else np.dot(m["w"], x) + m["b"])
                for x in X
            ]
        )
    if cost:
        return np.array([cost_prediction(model, x, fallback) for x in X])
    return np.clip([tabular.predict(model, x) for x in X], 0.0, 1.0)


def fit(
    X,
    quality,
    tokens,
    latency,
    *,
    binary,
    families=tabular.FAMILIES,
    trials=tabular.TRIALS,
    inner_k=tabular.INNER_K,
    outer_k=OUTER_K,
    seed=0,
    cost_model="query",
    sample_strata=None,
    stratum_weights=None,
):
    """Fit three predictors and return nested OOF [quality, ln tokens, ln latency]."""
    from sklearn.model_selection import KFold

    X = np.asarray(X, dtype=float)
    targets = [np.asarray(v, dtype=float) for v in (quality, tokens, latency)]
    n = len(X)
    if n < 2 or any(y.shape != (n,) for y in targets):
        raise ValueError(
            "Predictor fitting needs at least two unique query rows with aligned targets"
        )
    if not np.isfinite(X).all() or any(not np.isfinite(y).all() for y in targets):
        raise ValueError("Predictor features and targets must be finite")
    if np.any((targets[0] < 0) | (targets[0] > 1)) or any(
        np.any(y <= 0) for y in targets[1:]
    ):
        raise ValueError("Quality must be in [0,1] and costs must be positive")
    if cost_model not in {"query", "mean", "ridge"} or min(inner_k, outer_k) < 2:
        raise ValueError("Unknown cost model or invalid cross-validation folds")
    strata = np.asarray(sample_strata) if sample_strata is not None else None
    if strata is not None and (strata.shape != (n,) or not stratum_weights):
        raise ValueError(
            "Stratified mean fitting needs aligned strata and their weights"
        )

    def training_mean(y, rows):
        # Small folds can omit a stratum. Renormalize over represented strata;
        # never borrow the held-out labels to fill the missing stratum.
        return tabular.training_mean(
            y[rows], None if strata is None else strata[rows], stratum_weights
        )

    folds = list(KFold(min(n, outer_k), shuffle=True, random_state=seed).split(X))
    oof = np.empty((n, 3))
    models, selection = [], []
    for j, y in enumerate(targets):
        mode = "query" if j == 0 else cost_model
        is_binary = binary if j == 0 else False
        fold_info = []
        for fi, (tr, te) in enumerate(folds):
            fallback = training_mean(y, tr)
            m, info = _fit(
                X[tr],
                y[tr],
                is_binary,
                families,
                trials,
                inner_k,
                seed + fi + 1,
                mode,
                fallback,
                "quality" if j == 0 else "cost",
                None if strata is None else strata[tr],
                stratum_weights,
            )
            oof[te, j] = _predict(m, X[te], j > 0, fallback)
            fold_info.append(
                {
                    "fold": fi,
                    "n_train": len(tr),
                    "n_valid": len(te),
                    "family": info["family"],
                    "params": info.get("params", {}),
                }
            )
        mean = training_mean(y, np.arange(n))
        model, info = _fit(
            X,
            y,
            is_binary,
            families,
            trials,
            inner_k,
            seed,
            mode,
            mean,
            "quality" if j == 0 else "cost",
            strata,
            stratum_weights,
        )
        models.append(model)
        selection.append(
            {
                **{k: v for k, v in info.items() if k != "oof"},
                "outer_folds": fold_info,
                "nested_oof_loss": tabular._score(y, oof[:, j], is_binary),
            }
        )
    oof[:, 1:] = np.log(np.maximum(COST_FLOOR, oof[:, 1:]))

    def export_cost(m, y):
        if m["family"] == "log-ridge":
            return m["model"]
        if cost_model == "mean":
            return {"const": float(np.log(m["value"]))}
        return {
            "target": "raw_cost",
            "predictor": m,
            "floor": COST_FLOOR,
            "fallback": training_mean(y, np.arange(n)),
        }

    return {
        "quality": models[0],
        "ln_tokens": export_cost(models[1], targets[1]),
        "ln_latency": export_cost(models[2], targets[2]),
        "selection": dict(zip(OBJECTIVES, selection, strict=True)),
        "oof": oof.tolist(),
        "outer_folds": [te.tolist() for _, te in folds],
    }


def fit_job(args):
    """Picklable entry point for per-configuration worker pools."""
    X, quality, tokens, latency, options = args
    return fit(X, quality, tokens, latency, **options)
