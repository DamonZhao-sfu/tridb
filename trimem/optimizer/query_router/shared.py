"""Inductive, masked low-rank Q/T/L router inspired by LimeQO's ALS.

One query encoder W is shared across ALL configurations AND all three metrics:
    standardized_y[q,c,h] = bias[c,h] + (standardized_x[q] @ W) @ V[c,h]
Quality uses squared loss (Brier loss for binary outcomes), costs use log loss
targets (squared error in ln(cost)). Only observed cells enter the objective.
Unlike LimeQO, query IDs are never learned parameters. See the design document
and third_party/LimeQO.SOURCE.md for the upstream revision and differences.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from trimem.optimizer.query_router import runtime, sparse

PROTOCOL = "shared-inductive-als-v1"
FIELDS = ("quality", "tokens", "latency_ms")


def _fit_core(X, values, mask, weights, *, rank, reg, iterations, seed):
    """Fit on one query partition. Empty columns get the metric-wide training mean."""
    from scipy.linalg import solve

    n, c, _ = values.shape
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (X - mean) / scale
    d = x.shape[1]
    y_mean, y_scale = np.zeros(3), np.ones(3)
    y = np.zeros_like(values)
    for h in range(3):
        obs = mask[..., h]
        vals, w = values[..., h][obs], weights[..., h][obs]
        if not len(vals):
            raise ValueError("Each training partition needs observed outcomes")
        y_mean[h] = np.average(vals, weights=w)
        y_scale[h] = max(
            float(np.sqrt(np.average((vals - y_mean[h]) ** 2, weights=w))), 1e-6
        )
        y[..., h][obs] = (vals - y_mean[h]) / y_scale[h]
    y, mask, weights = (a.reshape(n, 3 * c) for a in (y, mask, weights))
    row, col = np.nonzero(mask)
    w = weights[row, col]
    target = y[row, col]
    rank = min(rank, d, 3 * c)
    rng = np.random.default_rng(seed)
    encoder = rng.normal(0, 0.1, (d, rank))
    factors = np.zeros((3 * c, rank))
    bias = np.zeros(3 * c)
    history = []
    for _ in range(iterations):
        z = x @ encoder
        # Column factors and biases; no unobserved/imputed labels in these solves.
        for j in range(3 * c):
            ii = mask[:, j]
            if not ii.any():
                continue
            design = np.column_stack((np.ones(ii.sum()), z[ii]))
            wj = weights[ii, j]
            penalty = np.diag([0.0] + [reg] * rank)
            coef = solve(
                design.T @ (wj[:, None] * design) + penalty,
                design.T @ (wj * y[ii, j]),
                assume_a="pos",
            )
            bias[j], factors[j] = coef[0], coef[1:]
        # One shared query-feature encoder fitted against every observed column.
        if rank:
            design = (x[row, :, None] * factors[col, None, :]).reshape(
                len(row), d * rank
            )
            gram = design.T @ (w[:, None] * design) + reg * np.eye(d * rank)
            rhs = design.T @ (w * (target - bias[col]))
            encoder = solve(gram, rhs, assume_a="pos").reshape(d, rank)
        residual = (
            target - bias[col] - np.sum((x[row] @ encoder) * factors[col], axis=1)
        )
        loss = float(
            w @ (residual**2) + reg * (np.sum(encoder**2) + np.sum(factors**2))
        )
        history.append(loss)
        if len(history) > 1 and abs(history[-2] - loss) <= 1e-6 * max(1, history[-2]):
            break
    return {
        "x_mean": mean.tolist(),
        "x_scale": scale.tolist(),
        "encoder": encoder.tolist(),
        "factors": factors.reshape(c, 3, rank).tolist(),
        "bias": bias.reshape(c, 3).tolist(),
        "y_mean": y_mean.tolist(),
        "y_scale": y_scale.tolist(),
        "rank": rank,
        "loss": history,
    }


def _weights(p, qids, cand, mask):
    """Equal column mass, with population-stratum weights within each column.

    Rescale to mean observed-cell weight 1. Missing strata are not synthesized;
    available population mass is renormalized within the training partition.
    """
    weights = np.zeros_like(mask, dtype=float)
    for j, _ in enumerate(cand):
        counts = Counter(p.strata[qids[i]] for i in np.flatnonzero(mask[:, j, 0]))
        for i in np.flatnonzero(mask[:, j, 0]):
            h = p.strata[qids[i]]
            weights[i, j, :] = p.weights[h] / counts[h]
        total = weights[:, j, 0].sum()
        if total:
            weights[:, j] /= total
    if not weights.sum():
        raise ValueError("No positive training weights")
    return weights * (mask.sum() / weights.sum())


def fit(
    X: dict[str, np.ndarray],
    p: sparse.SparseProfile,
    *,
    cand=None,
    rank: int = 5,
    reg: float = 0.2,
    iterations: int = 50,
    outer_k: int = 5,
    seed: int = 0,
    cross_fit: bool = True,
) -> tuple[dict, dict]:
    """Return deployable JSON model and ALL-column query-held-out predictions.

    Fixed hyperparameters: no AutoML / per-configuration classifier selection.
    OOF excludes each held query's observations from *every* configuration. It
    returns predictions even for unobserved columns, so policy replay cannot
    mix OOF observed scores with leaked in-sample unobserved scores.
    ``cross_fit=False`` fits one model without OOF, for fixed-policy outer
    evaluation; it must not be used to produce training replay estimates.
    Feature extraction and candidate pruning supplied by the caller are held
    fixed here; independent pipeline evaluation must use outer_cv.run_fold.
    """
    if rank < 1 or iterations < 1 or reg <= 0 or not np.isfinite(reg) or outer_k < 2:
        raise ValueError(
            "rank/iterations >= 1, finite reg > 0 and outer_k >= 2 required"
        )
    qids = p.queries
    if not qids:
        raise ValueError("At least one observed query is required")
    if cross_fit and len(qids) < 2:
        raise ValueError("At least two distinct queries are required for OOF")
    if cand is None:
        observed, columns = sparse.without_empty_columns(p)
        cand = [columns[c] for c in sparse.candidates(observed)]
    else:
        cand = list(cand)
    from threadpoolctl import threadpool_limits

    if (
        not cand
        or len(set(cand)) != len(cand)
        or any(c < 0 or c >= len(p.configs) for c in cand)
    ):
        raise ValueError("Need unique, valid candidate columns")
    x = np.asarray([X[q] for q in qids], dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Query features must be finite equal-length vectors")
    values = np.zeros((len(qids), len(cand), 3))
    mask = np.zeros_like(values, dtype=bool)
    pos = {q: i for i, q in enumerate(qids)}
    for j, c in enumerate(cand):
        if not p.samples[c] or len(set(p.samples[c])) != len(p.samples[c]):
            raise ValueError(
                "Each candidate needs unique, observed query rows; aggregate repeated trials first"
            )
        for q in p.samples[c]:
            h = p.strata[q]
            if h not in p.weights or not np.isfinite(p.weights[h]) or p.weights[h] <= 0:
                raise ValueError(
                    "Observed strata need finite positive population weights"
                )
            vals = np.array([getattr(p, f)[c][q] for f in FIELDS], dtype=float)
            if (
                not np.isfinite(vals).all()
                or not 0 <= vals[0] <= 1
                or (vals[1:] < 0).any()
            ):
                raise ValueError(
                    "Need completed, finite Q in [0,1] and nonnegative T/L; exclude timeouts"
                )
            vals[1:] = np.log(np.maximum(vals[1:], 1e-6))
            values[pos[q], j], mask[pos[q], j] = vals, True
    # Imported lazily to avoid a circular module import through sparse.fit.
    from trimem.optimizer.query_router.outer_cv import stratified_folds

    folds = (
        stratified_folds(qids, p.strata, min(outer_k, len(qids)), seed)
        if cross_fit
        else {}
    )
    opts = {"rank": rank, "reg": reg, "iterations": iterations, "seed": seed}
    oof = {j: {} for j in range(len(cand))} if cross_fit else {}
    # Bound BLAS CPU use; the live profiling process may share this host.
    with threadpool_limits(limits=1, user_api="blas"):
        for k in sorted(set(folds.values())):
            train = np.array([folds[q] != k for q in qids])
            ids = [q for q, ok in zip(qids, train, strict=True) if ok]
            w = _weights(p, ids, cand, mask[train])
            params = _fit_core(x[train], values[train], mask[train], w, **opts)
            for i in np.flatnonzero(~train):
                pred = np.stack(
                    runtime.predict_shared({"shared": params}, x[i]), axis=-1
                )
                for j, predictions in oof.items():
                    predictions[qids[i]] = pred[j].tolist()
        w = _weights(p, qids, cand, mask)
        params = _fit_core(x, values, mask, w, **opts)
    obs = mask[..., 0]
    model = {
        "kind": "shared-lowrank",
        "predictor_protocol": PROTOCOL,
        "quality_kind": "binary"
        if set(values[..., 0][obs]) <= {0.0, 1.0}
        else "continuous",
        "configs": [sparse._validate(p.configs[c]) for c in cand],
        "candidate_columns": cand,
        "shared": params,
        "oof_folds": folds,
        "training_queries": qids,
        "cost_norm": runtime.cost_norm(
            np.array([p.tokens[c][q] for c in cand for q in p.samples[c]]),
            np.array([p.latency_ms[c][q] for c in cand for q in p.samples[c]]),
        ),
        "params": {
            **opts,
            "outer_k": min(outer_k, len(qids)) if cross_fit else 0,
            "cross_fit": cross_fit,
            "cost_model": "query",
            "cost_target": "log",
            "training": "observed-cells-shared-encoder",
            "oof_scope": "whole-query-all-candidates",
            "metric_order": list(FIELDS),
        },
        "training_shape": {
            "queries": len(qids),
            "configs": len(cand),
            "observed_pairs": int(obs.sum()),
            "possible_pairs": int(obs.size),
        },
    }
    return model, oof
