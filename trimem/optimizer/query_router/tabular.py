"""BRANE per-configuration predictor: model selection over six tabular families, exported to JSON.

BRANE (arXiv 2605.27361, section 4.2 and appendix A.4): for every configuration that survives fuzzy
Pareto pruning, "we run automated model selection independently over a small family of tabular
classifiers - logistic regression, decision tree, random forest, gradient boosting, XGBoost and
LightGBM ... and keep the model with the best inner cross-validated negative log-loss", using "an
Optuna study of 30 hyperparameter trials with 3-fold inner cross-validation".

``select_and_fit`` reproduces that (binary target: negative log-loss; continuous target such as
ScienceWorld score/100: the regressor counterparts and mean squared error). The selected model is
exported to a numpy-only JSON form (``export`` / ``predict``) because deployment runs without
sklearn / xgboost / lightgbm: logistic regression as weights, every tree model as arrays
(feature, threshold, left, right, value) plus an aggregation rule. ``predict`` is verified against
the libraries in the tests.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FAMILIES = ("logreg", "tree", "forest", "gboost", "xgboost", "lightgbm")
TRIALS, INNER_K = 30, 3


# ------------------------------------------------------------------ numpy evaluation of exported models


def _tree_value(t: dict, x: np.ndarray) -> float:
    f, thr, left, right, val = (
        t["feature"],
        t["threshold"],
        t["left"],
        t["right"],
        t["value"],
    )
    lt = t.get("op") == "lt"
    f32 = t.get(
        "f32", False
    )  # XGBoost compares float32 features with float32 thresholds
    i = 0
    while f[i] >= 0:
        v = np.float32(x[f[i]]) if f32 else x[f[i]]
        th = np.float32(thr[i]) if f32 else thr[i]
        i = left[i] if (v < th if lt else v <= th) else right[i]
    return val[i]


def predict(m: dict, x: np.ndarray) -> float:
    """Quality prediction of an exported model: probability (binary) or value (continuous)."""
    fam = m["family"]
    if fam == "const":
        return float(m["value"])
    if fam == "linear":
        z = float(np.dot(m["w"], x) + m["b"])
        return 1 / (1 + math.exp(-max(min(z, 40), -40))) if m["binary"] else z
    vals = [_tree_value(t, x) for t in m["trees"]]
    if (
        m["agg"] == "mean"
    ):  # random forest / single tree: mean leaf probability (or value)
        return float(np.mean(vals))
    z = m["base"] + m["scale"] * float(np.sum(vals))  # boosting: additive margin
    return 1 / (1 + math.exp(-max(min(z, 40), -40))) if m["binary"] else z


# ------------------------------------------------------------------ export


def _sk_tree(est, binary: bool, proba: bool) -> dict:
    t = est.tree_
    if proba:  # classifier leaf: fraction of class 1
        v = t.value[:, 0, :]
        val = (v[:, 1] / v.sum(1)).tolist() if v.shape[1] > 1 else [0.0] * len(v)
    else:
        val = t.value[:, 0, 0].tolist()
    return {
        "feature": [int(f) if f >= 0 else -1 for f in t.feature],
        "threshold": t.threshold.tolist(),
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "value": val,
        "op": "le",
    }


def _xgb_trees(model) -> list[dict]:
    import json

    out = []
    for dump in model.get_booster().get_dump(dump_format="json"):
        nodes: dict[int, dict] = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for c in n.get("children", []):
                walk(c)

        walk(json.loads(dump))
        n = max(nodes) + 1
        t = {
            "feature": [-1] * n,
            "threshold": [0.0] * n,
            "left": [-1] * n,
            "right": [-1] * n,
            "value": [0.0] * n,
            "op": "lt",
            "f32": True,
        }
        for i, nd in nodes.items():
            if "leaf" in nd:
                t["value"][i] = float(nd["leaf"])
            else:
                t["feature"][i] = int(str(nd["split"]).lstrip("f"))
                t["threshold"][i] = float(nd["split_condition"])
                t["left"][i], t["right"][i] = int(nd["yes"]), int(nd["no"])
        out.append(t)
    return out


def _lgbm_trees(model) -> list[dict]:
    out = []
    for info in model.booster_.dump_model()["tree_info"]:
        t = {
            "feature": [],
            "threshold": [],
            "left": [],
            "right": [],
            "value": [],
            "op": "le",
        }

        def add(nd) -> int:
            i = len(t["feature"])
            for k in t:
                if k != "op":
                    t[k].append(-1 if k in ("feature", "left", "right") else 0.0)
            if "leaf_value" in nd and "split_feature" not in nd:
                t["value"][i] = float(nd["leaf_value"])
                return i
            if nd.get("decision_type", "<=") != "<=":
                raise ValueError("unsupported LightGBM split")
            t["feature"][i], t["threshold"][i] = (
                int(nd["split_feature"]),
                float(nd["threshold"]),
            )
            t["left"][i] = add(nd["left_child"])
            t["right"][i] = add(nd["right_child"])
            return i

        add(info["tree_structure"])
        out.append(t)
    return out


def export(family: str, model, binary: bool, X_ref: np.ndarray) -> dict:
    """JSON form of a fitted library model. Boosting offsets (base score / init) are recovered as the
    constant difference between the library's raw margin and the exported trees' sum on ``X_ref``."""
    if family == "hist":
        if any(len(stage) != 1 for stage in model._predictors):
            raise ValueError(
                "hist exporter supports only regression or binary classification"
            )
        trees = []
        for stage in model._predictors:
            nodes = stage[0].nodes
            if np.any(nodes["is_categorical"]):
                raise ValueError("categorical histogram splits are unsupported")
            trees.append(
                {
                    "feature": [
                        -1 if n["is_leaf"] else int(n["feature_idx"]) for n in nodes
                    ],
                    "threshold": [float(n["num_threshold"]) for n in nodes],
                    "left": [int(n["left"]) for n in nodes],
                    "right": [int(n["right"]) for n in nodes],
                    "value": [float(n["value"]) for n in nodes],
                    "op": "le",
                }
            )
        return {
            "family": "trees",
            "binary": binary,
            "agg": "sum",
            "scale": 1.0,
            "base": float(model._baseline_prediction[0, 0]),
            "trees": trees,
        }
    if family == "logreg":
        coef = model.coef_.ravel() if hasattr(model, "coef_") else model.coef_
        icpt = float(np.ravel(model.intercept_)[0])
        return {
            "family": "linear",
            "binary": binary,
            "w": np.ravel(coef).tolist(),
            "b": icpt,
        }
    if family == "tree":
        return {
            "family": "trees",
            "binary": binary,
            "agg": "mean",
            "trees": [_sk_tree(model, binary, binary)],
        }
    if family == "forest":
        return {
            "family": "trees",
            "binary": binary,
            "agg": "mean",
            "trees": [_sk_tree(e, binary, binary) for e in model.estimators_],
        }
    if family == "gboost":
        trees = [_sk_tree(e, binary, False) for e in np.ravel(model.estimators_)]
        m = {
            "family": "trees",
            "binary": binary,
            "agg": "sum",
            "scale": float(model.learning_rate),
            "trees": trees,
            "base": 0.0,
        }
        raw = model.decision_function(X_ref) if binary else model.predict(X_ref)
    elif family == "xgboost":
        m = {
            "family": "trees",
            "binary": binary,
            "agg": "sum",
            "scale": 1.0,
            "trees": _xgb_trees(model),
            "base": 0.0,
        }
        raw = model.predict(X_ref, output_margin=True)
    elif family == "lightgbm":
        m = {
            "family": "trees",
            "binary": binary,
            "agg": "sum",
            "scale": 1.0,
            "trees": _lgbm_trees(model),
            "base": 0.0,
        }
        raw = model.predict(X_ref, raw_score=True)
    else:
        raise ValueError(f"unknown family {family!r}")
    sums = np.array(
        [m["scale"] * sum(_tree_value(t, x) for t in m["trees"]) for x in X_ref]
    )
    raw = np.asarray(raw, dtype=float)
    off = raw - sums
    # XGBoost accumulates leaf values in float32 (relative error ~1e-6): raw-cost regressors (tokens, latency
    # in ms; margins ~1e3-1e5) need a tolerance relative to the margin scale; probability margins keep 1e-5.
    if np.ptp(off) > 1e-5 * max(1.0, float(np.abs(raw).max())):
        raise ValueError(
            f"{family}: exported trees do not reproduce the library margin"
        )
    m["base"] = float(off.mean())
    return m


# ------------------------------------------------------------------ model selection (BRANE)


def _make(family: str, p: dict, binary: bool, seed: int):
    if family == "logreg":
        from sklearn.linear_model import LogisticRegression, Ridge

        return (
            LogisticRegression(C=p["C"], max_iter=2000)
            if binary
            else Ridge(alpha=1 / p["C"])
        )
    if family == "tree":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        cls = DecisionTreeClassifier if binary else DecisionTreeRegressor
        return cls(
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=seed,
        )
    if family == "forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if binary else RandomForestRegressor
        return cls(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=seed,
            n_jobs=1,
        )
    if family == "gboost":
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )

        cls = GradientBoostingClassifier if binary else GradientBoostingRegressor
        return cls(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            min_samples_leaf=p["min_samples_leaf"],
            random_state=seed,
        )
    if family == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if binary else xgb.XGBRegressor
        return cls(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            reg_lambda=p["reg_lambda"],
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if family == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
        return cls(
            n_estimators=p["n_estimators"],
            num_leaves=p["num_leaves"],
            learning_rate=p["learning_rate"],
            min_child_samples=p["min_child_samples"],
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    raise ValueError(family)


def _space(trial, family: str) -> dict:
    if family == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if family == "tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 6),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    if family == "forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if family == "gboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 25, 200, step=25),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if family == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 25, 200, step=25),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1e2, log=True),
        }
    return {
        "n_estimators": trial.suggest_int("n_estimators", 25, 200, step=25),
        "num_leaves": trial.suggest_int("num_leaves", 2, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 20),
    }


def _folds(y: np.ndarray, binary: bool, k: int, seed: int):
    from sklearn.model_selection import KFold, StratifiedKFold

    if binary and min(np.bincount(y.astype(int), minlength=2)) >= k:
        return list(
            StratifiedKFold(k, shuffle=True, random_state=seed).split(
                np.zeros(len(y)), y
            )
        )
    return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y))))


def _score(y: np.ndarray, p: np.ndarray, binary: bool) -> float:
    """Loss to minimize: log-loss (binary) or mean squared error."""
    if binary:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(np.mean((y - p) ** 2))


def training_mean(y, strata=None, weights=None) -> float:
    """Population-weighted mean over strata present in this training partition."""
    y = np.asarray(y, dtype=float)
    if strata is None:
        return float(y.mean())
    strata = np.asarray(strata)
    present = np.unique(strata)
    w = np.asarray([weights[s] for s in present], dtype=float)
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid stratum weights")
    return float(w @ np.array([y[strata == s].mean() for s in present]) / w.sum())


def project_prediction(values, domain=None, fallback=None, floor=1e-6):
    """Identical prediction semantics for inner selection, outer replay and serving."""
    values = np.asarray(values, dtype=float)
    if domain == "quality":
        return np.clip(values, 0.0, 1.0)
    if domain == "cost":
        if fallback is None or not np.isfinite(fallback) or fallback <= 0:
            raise ValueError("Cost fallback must be a positive training-partition mean")
        return np.maximum(
            floor, np.where(np.isfinite(values) & (values > 0), values, fallback)
        )
    if domain is not None:
        raise ValueError("Unknown prediction domain")
    return values


def _fit_predict(
    family: str,
    p: dict,
    binary: bool,
    seed: int,
    Xtr,
    ytr,
    Xte,
    prediction_domain=None,
    fallback=None,
) -> np.ndarray:
    if binary and len(np.unique(ytr)) < 2:  # a fold with one class: the fold's prior
        return np.full(len(Xte), (ytr.sum() + 0.5) / (len(ytr) + 1))
    m = _make(family, p, binary, seed).fit(Xtr, ytr)
    values = m.predict_proba(Xte)[:, 1] if binary else m.predict(Xte)
    return project_prediction(values, prediction_domain, fallback)


def select_and_fit(
    X: np.ndarray,
    y: np.ndarray,
    *,
    binary: bool,
    families: tuple[str, ...] = FAMILIES,
    trials: int = TRIALS,
    inner_k: int = INNER_K,
    seed: int = 0,
    prediction_domain=None,
    sample_strata=None,
    stratum_weights=None,
) -> tuple[dict, dict]:
    """Optuna study over (family, hyper-parameters) by ``inner_k``-fold CV loss; the winner is refitted on
    all rows and exported. Returns (exported model, selection info incl. out-of-fold predictions)."""
    n = len(y)
    if n == 0:
        raise ValueError("Cannot fit a predictor without observations")
    strata = np.asarray(sample_strata) if sample_strata is not None else None
    if (
        n < 2
        or X.shape[1] == 0
        or (binary and y.min() == y.max())
        or (not binary and np.ptp(y) < 1e-12)
    ):
        v = float((y.sum() + 0.5) / (n + 1)) if binary else float(y.mean())
        if prediction_domain == "cost":
            v = training_mean(y, strata, stratum_weights)
        return {"family": "const", "value": v}, {
            "family": "const",
            "loss": None,
            "oof": [v] * n,
        }
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = _folds(y, binary, min(n, inner_k), seed)

    def oof(family: str, p: dict) -> np.ndarray:
        out = np.zeros(n)
        for tr, te in folds:
            fallback = (
                training_mean(
                    y[tr], None if strata is None else strata[tr], stratum_weights
                )
                if prediction_domain == "cost"
                else None
            )
            out[te] = _fit_predict(
                family,
                p,
                binary,
                seed,
                X[tr],
                y[tr],
                X[te],
                prediction_domain,
                fallback,
            )
        return out

    def objective(trial):
        fam = trial.suggest_categorical("family", list(families))
        return _score(y, oof(fam, _space(trial, fam)), binary)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params)
    fam = best.pop("family")
    model = _make(fam, best, binary, seed).fit(X, y)
    info = {
        "family": fam,
        "params": best,
        "loss": float(study.best_value),
        "oof": oof(fam, best).tolist(),
        "trials": trials,
    }
    return export(fam, model, binary, X), info


def describe(m: dict) -> str:
    if m["family"] in ("const", "linear"):
        return m["family"]
    return f"{m['agg']}-{len(m['trees'])}trees"


def as_json(obj: Any) -> Any:  # numpy scalars from library params
    if isinstance(obj, dict):
        return {k: as_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
