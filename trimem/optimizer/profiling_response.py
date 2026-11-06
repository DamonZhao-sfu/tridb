"""TRAIN-only conditional performance models for offline configuration discovery.

Predictions are raw conditional means (quality, tokens, latency_ms). Ensemble
members share query bootstrap identities across outputs and all queried points;
they approximate epistemic uncertainty, never simulated episode observations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass
class TrainingData:
    inputs: np.ndarray
    outputs: np.ndarray
    query_ids: np.ndarray
    config_ids: np.ndarray
    configs: list[dict]
    X: dict
    encode: object
    config_dim: int
    identity: str

    @classmethod
    def from_rows(cls, rows, X, encode):
        if not rows:
            raise ValueError("no completed TRAIN observations")
        rows = sorted(rows, key=lambda r: (config_key(r["config"]), r["query_id"]))
        configs, positions, seen, inputs, outputs, qids, cids = (
            [],
            {},
            set(),
            [],
            [],
            [],
            [],
        )
        for row in rows:
            if row.get("split") != "train":
                raise ValueError("response model accepts TRAIN only")
            qid, key = row["query_id"], config_key(row["config"])
            if (key, qid) in seen:
                raise ValueError(
                    "duplicate configuration/query; reconcile real trials first"
                )
            seen.add((key, qid))
            if qid not in X:
                raise ValueError("missing query features")
            try:
                y = np.array(
                    [row[n] for n in ("quality", "tokens", "latency_ms")], dtype=float
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("missing real outcome") from exc
            if not np.isfinite(y).all() or not 0 <= y[0] <= 1 or np.any(y[1:] <= 0):
                raise ValueError("invalid real outcome; missing data is not zero")
            if key not in positions:
                positions[key] = len(configs)
                configs.append(dict(row["config"]))
            cf = np.asarray(encode(row["config"]), dtype=float)
            inputs.append(np.concatenate([cf, np.asarray(X[qid], dtype=float)]))
            outputs.append(y)
            qids.append(qid)
            cids.append(positions[key])
        xx = np.asarray(inputs)
        if not np.isfinite(xx).all():
            raise ValueError("non-finite features")
        ident = digest(
            {
                "rows": rows,
                "features": {q: np.asarray(X[q]).tolist() for q in sorted(set(qids))},
                "config_encoding": [np.asarray(encode(c)).tolist() for c in configs],
            }
        )
        return cls(
            xx,
            np.asarray(outputs),
            np.asarray(qids),
            np.asarray(cids),
            configs,
            X,
            encode,
            len(cf),
            ident,
        )


class ResponseModel:
    @classmethod
    def fit(
        cls, data, *, members=32, trees=64, seed=0, indices=None, config_only=False
    ):
        obj = cls()
        obj.data, obj.models, obj.seed = data, [], seed
        ids = np.arange(len(data.outputs)) if indices is None else np.asarray(indices)
        if not len(ids) or members < 1 or trees < 1:
            raise ValueError("empty training partition or invalid ensemble size")
        obj.config_only = config_only
        x, y = data.inputs[ids], data.outputs[ids]
        if config_only:
            # Leave-configuration baseline: one real mean per training config.
            cids = data.config_ids[ids]
            unique = np.unique(cids)
            x = np.array([x[cids == c][0, : data.config_dim] for c in unique])
            y = np.array([y[cids == c].mean(axis=0) for c in unique])
            units = unique.astype(str)
        else:
            units = data.query_ids[ids]
        obj.center = y.mean(axis=0)
        obj.scale = y.std(axis=0)
        obj.scale[obj.scale < 1e-12] = 1.0
        target = (y - obj.center) / obj.scale
        unique, inverse = np.unique(units, return_inverse=True)
        rng = np.random.default_rng(seed)
        for _ in range(members):
            counts = np.bincount(
                rng.integers(len(unique), size=len(unique)), minlength=len(unique)
            )
            sample_weight = counts[inverse]
            keep = sample_weight > 0
            model = ExtraTreesRegressor(
                n_estimators=trees,
                min_samples_leaf=2,
                max_features=1.0,
                n_jobs=1,
                random_state=int(rng.integers(2**31 - 1)),
            )
            model.fit(x[keep], target[keep], sample_weight=sample_weight[keep])
            obj.models.append(model)
        return obj

    def predict_matrix(self, x):
        if self.config_only:
            x = x[:, : self.data.config_dim]
        pred = np.stack([m.predict(x) * self.scale + self.center for m in self.models])
        pred[..., 0] = np.clip(pred[..., 0], 0, 1)
        pred[..., 1:] = np.maximum(pred[..., 1:], 1e-6)
        return pred

    def predict(self, configs, qids):
        if not configs or not qids:
            return np.empty((len(self.models), len(configs), len(qids), 3))
        x = np.array(
            [
                np.concatenate([self.data.encode(c), self.data.X[q]])
                for c in configs
                for q in qids
            ]
        )
        return self.predict_matrix(x).reshape(
            len(self.models), len(configs), len(qids), 3
        )


def utility(raw, weights, norm):
    """[...,3] raw Q,T,L -> [...,W], using router weights Q,L,T."""
    raw, weights = np.asarray(raw), np.asarray(weights)
    if not np.isfinite(raw).all() or np.any(raw[..., 1:] <= 0):
        raise ValueError("utility requires finite positive raw costs")
    terms = np.stack(
        [
            raw[..., 0] / norm["quality"]["s"],
            -np.log(raw[..., 2]) / norm["latency_ms"]["s"],
            -np.log(raw[..., 1]) / norm["tokens"]["s"],
        ],
        axis=-1,
    )
    return terms @ weights.T


def validation_gate(data, weights, norm, *, seed=0, members=32, trees=64):
    """Two fixed three-fold diagnostics. This is not an independent policy test."""
    if len(data.configs) < 6 or len(np.unique(data.query_ids)) < 60:
        return {
            "status": "cold_start",
            "reason": "need 6 complete configs and 60 observed TRAIN queries",
        }
    metrics = {}
    for axis, units in [("query", data.query_ids), ("configuration", data.config_ids)]:
        unique = np.unique(units)
        errors, references, cids = [], [], []
        for fold, (tr, te) in enumerate(
            KFold(3, shuffle=True, random_state=seed).split(unique)
        ):
            train = np.flatnonzero(np.isin(units, unique[tr]))
            valid = np.flatnonzero(np.isin(units, unique[te]))
            model = ResponseModel.fit(
                data, indices=train, seed=seed + fold, members=members, trees=trees
            )
            pred = model.predict_matrix(data.inputs[valid]).mean(axis=0)
            if axis == "query":
                available = np.unique(data.config_ids[train])
                keep = np.isin(data.config_ids[valid], available)
                valid, pred = valid[keep], pred[keep]
                means = {
                    c: data.outputs[train[data.config_ids[train] == c]].mean(axis=0)
                    for c in available
                }
                baseline = np.array([means[c] for c in data.config_ids[valid]])
            else:
                baseline = ResponseModel.fit(
                    data,
                    indices=train,
                    config_only=True,
                    seed=seed + fold,
                    members=members,
                    trees=trees,
                )
                baseline = baseline.predict_matrix(data.inputs[valid]).mean(axis=0)
            if not len(valid):
                continue
            truth = utility(data.outputs[valid], weights, norm)
            errors.extend(np.mean((utility(pred, weights, norm) - truth) ** 2, axis=-1))
            references.extend(
                np.mean((utility(baseline, weights, norm) - truth) ** 2, axis=-1)
            )
            cids.extend(data.config_ids[valid])
        cids, errors, references = (
            np.asarray(cids),
            np.asarray(errors),
            np.asarray(references),
        )
        mse = (
            np.mean([errors[cids == c].mean() for c in np.unique(cids)])
            if len(cids)
            else 0.0
        )
        ref = (
            np.mean([references[cids == c].mean() for c in np.unique(cids)])
            if len(cids)
            else 0.0
        )
        metrics[axis] = {
            "rows": len(cids),
            "mse": float(mse),
            "baseline_mse": float(ref),
            "ratio": float(mse / ref) if ref > 1e-12 else None,
            "improved": bool(ref > 1e-12 and mse <= 0.98 * ref),
        }
    ready = all(m["rows"] >= 30 and m["improved"] for m in metrics.values())
    return {
        "status": "ready" if ready else "degraded",
        "diagnostics": metrics,
        "threshold": 0.98,
        "seed": seed,
    }
