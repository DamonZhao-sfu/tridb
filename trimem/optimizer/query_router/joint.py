"""Three independent, configuration-aware outcome predictors over sparse observations."""

from __future__ import annotations

import math

import numpy as np

from trimem.optimizer.query_router import sparse, tabular


def config_schema(configs):
    """Freeze a JSON schema, preserving every numeric and categorical knob."""
    keys = sorted(set().union(*(c.keys() for c in configs)))
    schema = []
    for key in keys:
        values = [c.get(key) for c in configs]
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in values
        ):
            schema.append({"key": key, "kind": "number"})
        else:
            schema.append(
                {
                    "key": key,
                    "kind": "categorical",
                    "values": sorted({str(v) for v in values}),
                }
            )
    return schema


def encode_config(config, schema):
    out = []
    for entry in schema:
        value = config.get(entry["key"])
        if entry["kind"] == "number":
            out.append(float(value))
        else:
            out.extend(float(str(value) == v) for v in entry["values"])
    return out


def _train(rows, targets, binary, seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    models = []
    for j, values in enumerate(targets.T):
        is_binary = binary and j == 0
        if np.ptp(values) < 1e-12:
            models.append({"family": "const", "value": float(values[0])})
            continue
        cls = RandomForestClassifier if is_binary else RandomForestRegressor
        model = cls(
            n_estimators=64,
            min_samples_leaf=3,
            max_features=1.0,
            n_jobs=1,
            random_state=seed + j,
        )
        model.fit(rows, values)
        models.append(tabular.export("forest", model, is_binary, rows))
    return models


def fit(X, p, *, cand=None, outer_k=5, seed=0, quality_kind=None):
    from threadpoolctl import threadpool_limits

    from trimem.optimizer.query_router.outer_cv import stratified_folds

    cand = sparse.candidates(p) if cand is None else list(cand)
    if (
        not cand
        or len(set(cand)) != len(cand)
        or any(c < 0 or c >= len(p.configs) for c in cand)
    ):
        raise ValueError("Need unique, valid candidate columns")
    qids = p.queries
    if len(qids) < 2 or outer_k < 2:
        raise ValueError("Need at least two query groups and two folds")
    training_cols = [c for c, sample in enumerate(p.samples) if sample]
    training_configs = [sparse._validate(p.configs[c]) for c in training_cols]
    configs = [sparse._validate(p.configs[c]) for c in cand]
    schema = config_schema(training_configs)
    cf = [encode_config(c, schema) for c in configs]
    records, labels, groups = [], [], []
    for c, config in zip(training_cols, training_configs, strict=True):
        if len(set(p.samples[c])) != len(p.samples[c]):
            raise ValueError("Duplicate query/config observations")
        for q in p.samples[c]:
            x = np.asarray(X[q], dtype=float)
            y = [p.quality[c][q], p.tokens[c][q], p.latency_ms[c][q]]
            if (
                not np.isfinite(x).all()
                or not np.isfinite(y).all()
                or not 0 <= y[0] <= 1
                or min(y[1:]) <= 0
            ):
                raise ValueError("Invalid feature or label")
            records.append(encode_config(config, schema) + x.tolist())
            labels.append(y)
            groups.append(q)
    rows, ys = np.asarray(records), np.asarray(labels)
    if rows.ndim != 2 or len(rows) < 2:
        raise ValueError("Insufficient aligned observations")
    is_binary = (
        set(np.unique(ys[:, 0])) <= {0.0, 1.0}
        if quality_kind is None
        else quality_kind == "binary"
    )
    if is_binary and not set(np.unique(ys[:, 0])) <= {0.0, 1.0}:
        raise ValueError("Binary quality requires binary labels")
    folds = stratified_folds(qids, p.strata, min(outer_k, len(qids)), seed)
    oof = {j: {} for j in range(len(cand))}
    group_arr = np.array(groups)
    with threadpool_limits(limits=1):
        for fold in sorted(set(folds.values())):
            mask = np.array([folds[q] != fold for q in group_arr])
            if not mask.any():
                raise ValueError("Empty training fold")
            fold_models = _train(rows[mask], ys[mask], is_binary, seed + fold)
            for q in qids:
                if folds[q] != fold:
                    continue
                for j in range(len(cand)):
                    x = np.array(cf[j] + np.asarray(X[q]).tolist())
                    vals = [tabular.predict(m, x) for m in fold_models]
                    oof[j][q] = [
                        float(np.clip(vals[0], 0, 1)),
                        math.log(max(vals[1], 1e-6)),
                        math.log(max(vals[2], 1e-6)),
                    ]
        models = _train(rows, ys, is_binary, seed)
    means = ys.mean(axis=0)
    return {
        "kind": "joint-tabular",
        "configs": configs,
        "candidate_columns": cand,
        "config_schema": schema,
        "config_features": cf,
        "objectives": models,
        "fallback": means[1:].tolist(),
        "quality_kind": "binary" if is_binary else "continuous",
        "oof_folds": folds,
        "oof_scope": "whole-query-all-candidates",
        "training_queries": qids,
        "training_shape": {
            "observed_pairs": len(rows),
            "queries": len(qids),
            "configs": len(training_cols),
        },
        "train_means": {
            "quality": [p.est(c, "quality").mean for c in cand],
            "tokens": [p.est(c, "tokens").mean for c in cand],
            "latency_ms": [p.est(c, "latency_ms").mean for c in cand],
            "n_per_candidate": [len(p.samples[c]) for c in cand],
        },
        "selection": [{"family": "joint-forest"} for _ in cand],
        "params": {"backend": "joint", "seed": seed, "outer_k": outer_k},
    }, oof
