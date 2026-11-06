"""Three configuration-aware predictors with TRAIN-only family selection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from trimem.optimizer import session_space_agentenv_v2 as space
from trimem.optimizer.query_router import joint, runtime, sparse, tabular
from trimem.optimizer.query_router.outer_cv import stratified_folds

NAMES = ("quality", "tokens", "latency_ms")


def restore_scienceworld_scores(profile, raw_scores):
    """Restore raw scores only when every recorded label reconciles exactly."""
    columns = []
    for config, ids, old in zip(
        profile.configs, profile.samples, profile.quality, strict=True
    ):
        values = {}
        for q in ids:
            key = (space.key(config), q)
            if key not in raw_scores:
                raise ValueError(f"missing raw ScienceWorld score for {q}")
            value = float(raw_scores[key])
            if not np.isfinite(value) or not np.isclose(
                max(value, 0) / 100, float(old[q]), rtol=0, atol=1e-8
            ):
                raise ValueError(f"raw score does not match recorded quality for {q}")
            values[q] = value
        columns.append(values)
    return replace(profile, quality=tuple(columns))


def estimator(family, binary, seed):
    if family == "linear":
        return (
            LogisticRegression(C=1, max_iter=600, random_state=seed)
            if binary
            else Ridge(alpha=10)
        )
    if family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if binary else XGBRegressor
        return cls(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.05,
            n_jobs=1,
            verbosity=0,
            random_state=seed,
        )
    if family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMClassifier if binary else LGBMRegressor
        return cls(
            n_estimators=80,
            num_leaves=15,
            learning_rate=0.05,
            n_jobs=1,
            verbosity=-1,
            random_state=seed,
        )
    if family == "hist":
        cls = (
            HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
        )
        return cls(
            max_iter=100, max_leaf_nodes=15, l2_regularization=3, random_state=seed
        )
    cls = RandomForestClassifier if binary else RandomForestRegressor
    return cls(
        n_estimators=64,
        min_samples_leaf=5,
        max_features=0.85,
        n_jobs=1,
        random_state=seed,
    )


def fit_objective(X, y, groups, *, binary, cost, seed):
    if np.ptp(y) < 1e-12:
        value = np.log1p(y[0]) if cost else y[0]
        return {"family": "const", "value": float(value)}, "constant", {}
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("model selection requires at least two training queries")
    scores = {name: [] for name in ("linear", "forest", "hist", "xgboost", "lightgbm")}
    for tr, va in GroupKFold(n_splits=min(3, len(unique))).split(X, groups=groups):
        for family, losses in scores.items():
            if binary and np.unique(y[tr]).size < 2:
                pred = np.full(len(va), y[tr][0])
            else:
                m = estimator(family, binary, seed)
                m.fit(X[tr], np.log1p(y[tr]) if cost else y[tr])
                pred = m.predict_proba(X[va])[:, 1] if binary else m.predict(X[va])
                if cost:
                    pred = np.maximum(np.expm1(np.clip(pred, -10, 25)), 0)
            losses.append(
                float(
                    np.mean((pred - y[va]) ** 2)
                    if binary
                    else np.mean(np.abs(pred - y[va]))
                )
            )
    chosen = min(scores, key=lambda name: np.mean(scores[name]))
    model = estimator(chosen, binary, seed)
    model.fit(X, np.log1p(y) if cost else y)
    return (
        tabular.export(
            "logreg" if chosen == "linear" else chosen,
            model,
            binary,
            X[: min(100, len(X))],
        ),
        chosen,
        scores,
    )


def fit(X, p, *, cand=None, outer_k=5, seed=0, quality_kind=None):
    cand = (
        sparse.candidates(p, tau_acc=2.0 if quality_kind == "raw_score" else 0.02)
        if cand is None
        else list(cand)
    )
    if (
        not cand
        or len(cand) != len(set(cand))
        or any(c < 0 or c >= len(p.configs) for c in cand)
    ):
        raise ValueError("Need unique, valid candidate columns")
    qids = p.queries
    if len(qids) < 3 or outer_k < 2:
        raise ValueError("Need at least three query groups and two folds")
    cols = [c for c, sample in enumerate(p.samples) if sample]
    train_configs = [sparse._validate(p.configs[c]) for c in cols]
    configs = [sparse._validate(p.configs[c]) for c in cand]
    schema = joint.config_schema(train_configs)
    cf = [joint.encode_config(c, schema) for c in configs]
    data, targets, groups = [], [], []
    for c, config in zip(cols, train_configs, strict=True):
        for q in p.samples[c]:
            v = [p.quality[c][q], p.tokens[c][q], p.latency_ms[c][q]]
            if not np.isfinite(v).all() or min(v[1:]) <= 0:
                raise ValueError("Invalid training targets")
            data.append(joint.encode_config(config, schema) + np.asarray(X[q]).tolist())
            targets.append(v)
            groups.append(q)
    data, targets, groups = np.asarray(data), np.asarray(targets), np.asarray(groups)
    binary = (
        set(np.unique(targets[:, 0])) <= {0.0, 1.0}
        if quality_kind is None
        else quality_kind == "binary"
    )
    if binary and not set(np.unique(targets[:, 0])) <= {0.0, 1.0}:
        raise ValueError("Binary quality requires binary labels")
    raw_score = quality_kind == "raw_score"
    folds = stratified_folds(qids, p.strata, min(outer_k, len(qids)), seed)
    oof = {j: {} for j in range(len(cand))}
    with threadpool_limits(limits=1):
        for fold in sorted(set(folds.values())):
            tr = np.array([folds[q] != fold for q in groups])
            models = [
                fit_objective(
                    data[tr],
                    targets[tr, j],
                    groups[tr],
                    binary=binary and j == 0,
                    cost=j > 0,
                    seed=seed + fold,
                )[0]
                for j in range(3)
            ]
            tmp = {
                "kind": "joint-selected",
                "config_features": cf,
                "objectives": models,
                "quality_kind": "raw_score"
                if raw_score
                else "binary"
                if binary
                else "continuous",
            }
            for q in qids:
                if folds[q] == fold:
                    pred = np.stack(runtime.predict(tmp, X[q]), axis=-1)
                    for j in range(len(cand)):
                        oof[j][q] = pred[j].tolist()
        trained = [
            fit_objective(
                data,
                targets[:, j],
                groups,
                binary=binary and j == 0,
                cost=j > 0,
                seed=seed,
            )
            for j in range(3)
        ]
    return {
        "kind": "joint-selected",
        "configs": configs,
        "candidate_columns": cand,
        "config_schema": schema,
        "config_features": cf,
        "objectives": [t[0] for t in trained],
        "selected_families": dict(zip(NAMES, [t[1] for t in trained])),
        "cv_losses": dict(zip(NAMES, [t[2] for t in trained])),
        "quality_kind": "raw_score"
        if raw_score
        else "binary"
        if binary
        else "continuous",
        "oof_folds": folds,
        "training_queries": qids,
        "training_shape": {
            "observed_pairs": len(data),
            "queries": len(qids),
            "configs": len(cols),
        },
        "train_means": {
            "quality": [p.est(c, "quality").mean for c in cand],
            "tokens": [p.est(c, "tokens").mean for c in cand],
            "latency_ms": [p.est(c, "latency_ms").mean for c in cand],
            "n_per_candidate": [len(p.samples[c]) for c in cand],
        },
        "selection": [{"family": t[1]} for t in trained],
        "params": {"backend": "joint-selected", "seed": seed, "outer_k": outer_k},
    }, oof
