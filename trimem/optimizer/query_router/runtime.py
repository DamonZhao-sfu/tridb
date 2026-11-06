"""Per-query router: JSON artifact, feature transform and routing rule (numpy only).

Deployment runs in ``.venv-agentgym`` (no sklearn / torch), so everything a trained router needs at
inference time lives in one JSON artifact and is evaluated here with numpy:

* ``feature_spec``: training-panel statistics that turn a raw query record (task row fields, the
  probe signals, the task vector, optional LLM yes/no labels) into a standardized vector;
* ``model``: ``joint-tabular`` (one configuration-aware model per objective), ``brane`` (three
  per-configuration tabular predictors), ``gp`` or ``shared-lowrank``;
* the routing rule, by ``routing['rule']``:
  - ``utility3-minmax`` (Opt-PerQuery three-weight objective, user 2026-10-19; CA-RAG arXiv 2606.02581
    Eq. 1; the rule name is kept for artifact compatibility, the scale lives in ``model['cost_norm']``,
    frozen at fit time), w_Q + w_L + w_T = 1, every weight >= 0 (corners allowed):
    * ``scale == "zscore-log-v1"`` (user 2026-10-22, equal influence of the three objectives):
      ``argmax_c  w_Q q_c/s_Q - w_T ln tok_c/s_T - w_L ln lat_c/s_L`` where s_* = the median over training
      queries of the across-candidate std of the honest predictions, floored at the metric's measurement SE
      (``utility_norm``);
    * ``scale == "minmax-raw"`` (artifacts before 2026-10-22, reproducible): ``w_Q q - w_L lat~ - w_T tok~``
      with RAW predicted costs min-max normalized by training observation bounds;
  - anything else (legacy artifacts, kept as an ablation):
    ``argmax_c  q_c(x) - lambda * (w_T ln tokens_c(x) + w_L ln latency_c(x))``.

A raw query record (built by ``features.raw_record``) is a dict::

    {"query_id": str, "task": {"task_type", "object", "recep", "description"},
     "probe": {name: float}, "emb": [float] | None, "llm": [0/1] | None,
     "overhead": {"ms": float, "tokens": float}}
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

KIND = "trimem_query_router"
VERSION = 2
PROBE_KEYS = (
    "top1_cos",
    "mean_cos",
    "type_match",
    "object_match",
    "n_sessions",
    "mean_steps",
    "same_type_cos",
)


# ------------------------------------------------------------------ features


def _structured(spec: dict, task: dict) -> list[float]:
    out = [float(task.get("task_type") == t) for t in spec["task_types"]]
    out += [float(task.get("recep") == r) for r in spec["receps"]]
    obj = str(task.get("object") or "").lower()
    out.append(math.log1p(spec["object_counts"].get(obj, 0)))
    out.append(float(len(str(task.get("description") or "").split())))
    return out


def raw_vector(spec: dict, raw: dict) -> np.ndarray:
    """Unstandardized feature vector of one raw query record (layout = ``spec['names']``)."""
    v: list[float] = []
    groups = spec["groups"]
    if "structured" in groups:
        v += _structured(spec, raw["task"])
    if "probe" in groups:
        p = raw.get("probe") or {}
        v += [float(p.get(k, 0.0)) for k in PROBE_KEYS]
    if "emb" in groups:
        e = np.asarray(raw["emb"], dtype=float)
        pca = spec["emb_pca"]
        v += list((e - np.asarray(pca["mean"])) @ np.asarray(pca["components"]).T)
    if "llm" in groups:
        labels = raw.get("llm") or []
        if len(labels) != len(spec["llm_questions"]):
            raise ValueError("LLM labels do not match the router's questions")
        v += [float(x) for x in labels]
    if len(v) != len(spec["names"]):
        raise ValueError("feature layout mismatch")
    return np.asarray(v, dtype=float)


def transform(spec: dict, raw: dict) -> np.ndarray:
    x = raw_vector(spec, raw)
    return (x - np.asarray(spec["scaler"]["mean"])) / np.asarray(spec["scaler"]["std"])


# ------------------------------------------------------------------ models


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _linear(m: dict, x: np.ndarray) -> float:
    if "const" in m:
        return float(m["const"])
    return float(np.dot(np.asarray(m["w"]), x) + m["b"])


def predict_brane(
    model: dict, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per candidate: predicted quality (success probability or score), ln tokens, ln latency.
    ``variant == "paper"``: exported tabular models (``tabular.predict``) on the kept features;
    otherwise the earlier linear form (``w``/``b`` or ``const``)."""
    from trimem.optimizer.query_router import tabular

    if "x_keep" in model:
        x = x[np.asarray(model["x_keep"], dtype=int)]
    q = []
    config_features = model.get("config_features")
    rows = (
        [np.concatenate((x, np.asarray(cf, dtype=float))) for cf in config_features]
        if config_features is not None
        else [x] * len(model["quality"])
    )
    for m, row in zip(model["quality"], rows, strict=True):
        if "family" in m:
            q.append(tabular.predict(m, row))
            continue
        z = _linear(m, row)
        q.append(
            z
            if "const" in m or model["quality_kind"] == "continuous"
            else float(_sigmoid(np.array(z)))
        )
    q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)

    def log_cost(m, row):
        if m.get("target") == "raw_cost":
            value = tabular.project_prediction(
                tabular.predict(m["predictor"], row), "cost", m["fallback"], m["floor"]
            )
            return math.log(float(value))
        return _linear(m, row)  # historical log-linear / constant artifacts

    return (
        q,
        np.array(
            [log_cost(m, row) for m, row in zip(model["ln_tokens"], rows, strict=True)]
        ),
        np.array(
            [log_cost(m, row) for m, row in zip(model["ln_latency"], rows, strict=True)]
        ),
    )


def gp_mean(obj: dict, z: np.ndarray) -> np.ndarray:
    """Posterior mean of an exported RBF-ARD GP at rows ``z`` (already normalized)."""
    X, ls = np.asarray(obj["X"]), np.asarray(obj["lengthscale"])
    d = ((z[:, None, :] - X[None, :, :]) / ls) ** 2
    k = obj["outputscale"] * np.exp(-0.5 * d.sum(-1))
    return obj["const"] + k @ np.asarray(obj["alpha"])


def gp_inputs(model: dict, x: np.ndarray) -> np.ndarray:
    """Rows [config features (+) projected query features] for every candidate, min-max normalized."""
    proj = model["query_proj"]
    xq = (
        (x - np.asarray(proj["mean"])) @ np.asarray(proj["components"]).T if proj else x
    )
    cf = np.asarray(model["cfg_feats"])
    rows = np.hstack([cf, np.repeat(xq[None, :], len(cf), axis=0)])[
        :, np.asarray(model["x_keep"], dtype=bool)
    ]
    lo, hi = np.asarray(model["x_lo"]), np.asarray(model["x_hi"])
    return (rows - lo) / np.where(hi > lo, hi - lo, 1.0)


def predict_gp(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = gp_inputs(model, x)
    out = []
    for name in ("quality", "ln_tokens", "ln_latency"):
        o = model["objectives"][name]
        out.append(gp_mean(o, z) * o["y_std"] + o["y_mean"])
    return np.clip(out[0], 0.0, 1.0), out[1], out[2]


def predict_shared(
    model: dict, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All configurations and metrics from one inductive encoder (NumPy only)."""
    m = model["shared"]
    x = np.asarray(x, dtype=float)
    mean = np.asarray(m["x_mean"])
    if x.shape != mean.shape or not np.isfinite(x).all():
        raise ValueError(
            "shared router requires finite features of the trained dimension"
        )
    encoder = np.asarray(m["encoder"]).reshape(len(mean), m["rank"])
    z = ((x - mean) / np.asarray(m["x_scale"])) @ encoder
    y = np.asarray(m["bias"]) + np.einsum("chr,r->ch", np.asarray(m["factors"]), z)
    y = y * np.asarray(m["y_scale"]) + np.asarray(m["y_mean"])
    return (
        np.clip(y[:, 0], 0, 1),
        np.clip(y[:, 1], np.log(1e-6), 40),
        np.clip(y[:, 2], np.log(1e-6), 40),
    )


def predict(model: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model["kind"] == "joint-selected":
        from trimem.optimizer.query_router import tabular

        rows = [
            np.concatenate((np.asarray(c, dtype=float), x))
            for c in model["config_features"]
        ]
        values = np.asarray(
            [[tabular.predict(m, row) for m in model["objectives"]] for row in rows]
        )
        if not np.isfinite(values).all():
            raise ValueError("invalid selected router predictions")
        quality = (
            values[:, 0]
            if model.get("quality_kind") == "raw_score"
            else np.clip(values[:, 0], 0, 1)
        )
        costs = np.maximum(np.expm1(np.clip(values[:, 1:], -10, 25)), 1e-6)
        return quality, np.log(costs[:, 0]), np.log(costs[:, 1])
    if model["kind"] == "joint-staged":
        from trimem.optimizer.query_router import tabular

        rows = [
            np.concatenate((np.asarray(c, dtype=float), x))
            for c in model["config_features"]
        ]
        quality = np.asarray([tabular.predict(model["quality"], row) for row in rows])
        log_stages = np.asarray(
            [[tabular.predict(m, row) for m in model["stage_models"]] for row in rows]
        )
        if (
            log_stages.shape != (len(rows), 4)
            or not np.isfinite(log_stages).all()
            or not np.isfinite(quality).all()
        ):
            raise ValueError("invalid staged router predictions")
        stages = np.maximum(np.expm1(np.clip(log_stages, -10, 25)), 0)
        tokens = stages[:, 0] + stages[:, 1]
        latency = stages[:, 2] + stages[:, 3]
        if model.get("quality_kind") != "raw_score":
            quality = np.clip(quality, 0, 1)
        return (
            quality,
            np.log(np.maximum(tokens, 1e-6)),
            np.log(np.maximum(latency, 1e-6)),
        )
    if model["kind"] == "joint-tabular":
        from trimem.optimizer.query_router import tabular

        rows = [
            np.concatenate((np.asarray(c, dtype=float), x))
            for c in model["config_features"]
        ]
        values = np.asarray(
            [[tabular.predict(m, row) for m in model["objectives"]] for row in rows]
        )
        values[:, 0] = np.clip(values[:, 0], 0, 1)
        for j in (1, 2):
            values[:, j] = np.where(
                values[:, j] > 0, values[:, j], model["fallback"][j - 1]
            )
        return values[:, 0], np.log(values[:, 1]), np.log(values[:, 2])
    if model["kind"] == "brane":
        return predict_brane(model, x)
    if model["kind"] == "gp":
        return predict_gp(model, x)
    if model["kind"] == "shared-lowrank":
        return predict_shared(model, x)
    raise ValueError(f"unknown router model {model['kind']!r}")


# ------------------------------------------------------------------ routing rules

# Opt-PerQuery three-weight objective (user decision 2026-10-19): one weight per Opt-OneConfig objective.
RULE_UTILITY = "utility3-minmax"
DEFAULT_WEIGHTS = (
    0.6,
    0.2,
    0.2,
)  # (w_Q, w_L, w_T): CA-RAG (arXiv 2606.02581) Eq. 1 default


def scores(
    q: np.ndarray,
    ln_tok: np.ndarray,
    ln_lat: np.ndarray,
    lam: float,
    w_tokens: float = 0.5,
    w_latency: float = 0.5,
) -> np.ndarray:
    """LEGACY rule (ablation; artifacts without ``rule == utility3-minmax``): q - lambda (w_T ln tok + w_L ln lat)."""
    return q - lam * (w_tokens * ln_tok + w_latency * ln_lat)


def normalize_weights(
    weights: tuple[float, float, float] | list[float],
) -> tuple[float, float, float]:
    """(w_Q, w_L, w_T) scaled to sum 1; every weight must be finite and >= 0, not all zero (user 2026-10-22:
    w_Q = 0 allowed, so the corners (1,0,0), (0,1,0), (0,0,1) are valid)."""
    w = tuple(float(v) for v in weights)
    if len(w) != 3 or any(not math.isfinite(v) or v < 0 for v in w) or sum(w) <= 0:
        raise ValueError(
            f"weights must be (w_Q, w_L, w_T) >= 0, not all zero, got {weights!r}"
        )
    t = sum(w)
    return (w[0] / t, w[1] / t, w[2] / t)


SCALE_MINMAX = (
    "minmax-raw"  # artifacts before 2026-10-22 (kept readable / reproducible)
)
SCALE_ZLOG = (
    "zscore-log-v1"  # user 2026-10-22: equal-influence scale (``utility_norm``)
)
NORM_METRICS = (
    "quality",
    "tokens",
    "latency_ms",
)  # pred[..., 0..2] = quality, ln tokens, ln latency


def robust_spread(v: np.ndarray) -> tuple[float, str]:
    """Robust std of a small set of values: MAD * 1.4826, else IQR / 1.349, else the plain std (ddof 0)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return 0.0, "n<2"
    mad = float(np.median(np.abs(v - np.median(v)))) * 1.4826
    if mad > 0:
        return mad, "mad"
    q75, q25 = np.percentile(v, [75, 25])
    if q75 > q25:
        return float(q75 - q25) / 1.349, "iqr"
    return float(np.std(v)), "std"


def utility_norm(pred: np.ndarray, cand_means: np.ndarray, cand_se: np.ndarray) -> dict:
    """``zscore-log-v1`` scales s_Q, s_T, s_L of the three-weight utility (fit time, frozen in the model).

    ``pred`` = (training queries, candidates, [quality, ln tokens, ln latency]) honest (OOF where available)
    predictions; ``cand_means`` / ``cand_se`` = (candidates, 3) per-candidate training means and their
    stratified SEs on the same scale (quality, mean ln tokens, mean ln latency).

    Per metric: ``per_query`` = median over queries of the across-candidate std (ddof 0) of the predictions
    (used for routing); ``candidate_means`` = robust spread (MAD / IQR) of the candidate means (recorded);
    ``floor`` = median per-candidate SE of the metric's mean (measurement noise). s = max(per_query, floor)
    so a metric whose candidate differences are below noise is not amplified; when the per-query spread is 0
    or not finite the candidate-mean spread is used (``source``), and 1.0 when everything is 0.
    """
    pred, cand_means, cand_se = (
        np.asarray(a, dtype=float) for a in (pred, cand_means, cand_se)
    )
    if (
        pred.ndim != 3
        or pred.shape[2] != 3
        or cand_means.shape != (pred.shape[1], 3)
        or cand_se.shape != cand_means.shape
    ):
        raise ValueError(
            "utility_norm needs pred (n, C, 3) and per-candidate means / SEs (C, 3)"
        )
    out: dict = {
        "scale": SCALE_ZLOG,
        "n_queries": int(pred.shape[0]),
        "n_candidates": int(pred.shape[1]),
        "cost_space": "ln",
        "spread": "median over training queries of the across-candidate std "
        "(ddof 0) of the honest predictions; floor = median per-candidate SE of the mean",
    }
    for m, name in enumerate(NORM_METRICS):
        sd = (
            np.std(pred[:, :, m], axis=1)
            if pred.shape[1] > 1
            else np.zeros(pred.shape[0])
        )
        per_q = float(np.median(sd)) if len(sd) else math.nan
        cm, cm_how = robust_spread(cand_means[:, m])
        floor = float(np.median(cand_se[:, m])) if len(cand_se) else 0.0
        floor = floor if math.isfinite(floor) else 0.0
        if math.isfinite(per_q) and per_q > 0:
            spread, src = per_q, "per_query"
        elif cm > 0:
            spread, src = cm, "candidate_means"
        else:
            spread, src = 0.0, "none"
        s = max(spread, floor)
        if s <= 0:
            s, src = 1.0, "unit"
        out[name] = {
            "s": float(s),
            "per_query": per_q,
            "candidate_means": cm,
            "candidate_means_how": cm_how,
            "floor": floor,
            "floor_active": bool(floor > spread),
            "source": src,
            "per_query_iqr": [float(v) for v in np.percentile(sd, [25, 75])]
            if len(sd)
            else None,
        }
    return out


def utility_terms(
    q: np.ndarray,
    tokens: np.ndarray,
    latency_ms: np.ndarray,
    weights: tuple[float, float, float],
    norm: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three signed terms (quality, -latency, -tokens) of the utility; their sum is ``utility``."""
    wq, wl, wt = normalize_weights(weights)
    q = np.asarray(q, dtype=float)
    scale = norm.get("scale")
    if scale == SCALE_ZLOG:
        lt = np.log(np.maximum(np.asarray(tokens, dtype=float), 1e-6))
        ll = np.log(np.maximum(np.asarray(latency_ms, dtype=float), 1e-6))
        return (
            wq * q / norm["quality"]["s"],
            -wl * ll / norm["latency_ms"]["s"],
            -wt * lt / norm["tokens"]["s"],
        )
    if scale == SCALE_MINMAX:
        return (
            wq * q,
            -wl * minmax(latency_ms, norm["latency_ms"]),
            -wt * minmax(tokens, norm["tokens"]),
        )
    raise ValueError(f"unsupported cost_norm scale {scale!r}")


def cost_norm(tokens: np.ndarray, latency_ms: np.ndarray) -> dict:
    """LEGACY ``minmax-raw`` scale (before 2026-10-22; ``utility_norm`` is the default since): min-max constants
    of supplied raw training costs, never recomputed online.

    Query predictors supply all observed query costs; mean-cost ablations supply
    configuration means. Predictions outside the training range are not clipped.
    """
    tok, lat = np.asarray(tokens, dtype=float), np.asarray(latency_ms, dtype=float)
    return {
        "scale": "minmax-raw",
        "tokens": {"lo": float(tok.min()), "hi": float(tok.max())},
        "latency_ms": {"lo": float(lat.min()), "hi": float(lat.max())},
    }


def minmax(v: np.ndarray, c: dict) -> np.ndarray:
    """(v - lo) / (hi - lo); 0 everywhere when the candidates do not differ on this cost."""
    rng = c["hi"] - c["lo"]
    v = np.asarray(v, dtype=float)
    return (v - c["lo"]) / rng if rng > 0 else np.zeros_like(v)


def utility(
    q: np.ndarray,
    tokens: np.ndarray,
    latency_ms: np.ndarray,
    weights: tuple[float, float, float],
    norm: dict,
) -> np.ndarray:
    """Three-weight utility (weights normalised to sum 1) on the artifact's frozen scale: ``zscore-log-v1``
    U_c = w_Q q_c/s_Q - w_L ln lat_c/s_L - w_T ln tok_c/s_T; legacy ``minmax-raw`` U_c = w_Q q_c - w_L lat~_c -
    w_T tok~_c. ``tokens`` / ``latency_ms`` are RAW predicted costs."""
    a, b, c = utility_terms(q, tokens, latency_ms, weights, norm)
    return a + b + c


def _decision(
    model: dict, q: np.ndarray, t: np.ndarray, lat: np.ndarray, s: np.ndarray
) -> dict:
    i = int(
        np.argmax(s)
    )  # ties -> lowest index (candidates are ordered by training quality)
    return {
        "index": i,
        "config": model["configs"][i],
        "score": float(s[i]),
        "pred": {
            "quality": float(q[i]),
            "tokens": float(math.exp(t[i])),
            "latency_ms": float(math.exp(lat[i])),
        },
        "scores": [float(v) for v in s],
        "q_hat": [float(v) for v in q],
    }


def choose(
    model: dict,
    x: np.ndarray,
    lam: float,
    w_tokens: float = 0.5,
    w_latency: float = 0.5,
) -> dict:
    """LEGACY ln / lambda rule (ablation)."""
    q, t, lat = predict(model, x)
    return _decision(model, q, t, lat, scores(q, t, lat, lam, w_tokens, w_latency))


def choose_utility(
    model: dict, x: np.ndarray, weights: tuple[float, float, float]
) -> dict:
    """Three-weight rule; needs ``model['cost_norm']`` (written by ``sparse.fit`` / ``brane.fit``)."""
    if "cost_norm" not in model:
        raise ValueError(
            "router model has no cost_norm: cannot apply the three-weight rule"
        )
    q, t, lat = predict(model, x)
    return _decision(
        model,
        q,
        t,
        lat,
        utility(q, np.exp(t), np.exp(lat), weights, model["cost_norm"]),
    )


def routing_weights(routing: dict) -> tuple[float, float, float] | None:
    """(w_Q, w_L, w_T) of a three-weight routing block; None for a legacy (lambda, w) block."""
    if routing.get("rule") != RULE_UTILITY:
        return None
    return normalize_weights(
        (routing["w_quality"], routing["w_latency"], routing["w_tokens"])
    )


def choose_routing(model: dict, x: np.ndarray, routing: dict) -> dict:
    """The deployed decision of a routing block (three-weight or legacy)."""
    w = routing_weights(routing)
    if w is not None:
        return choose_utility(model, x, w)
    return choose(
        model, x, routing["lambda"], routing["w_tokens"], routing["w_latency"]
    )


# ------------------------------------------------------------------ artifact


class Router:
    def __init__(self, artifact: dict):
        if artifact.get("kind") != KIND or artifact.get("version") not in (1, VERSION):
            raise ValueError("not a TriMEM query-router artifact")
        if artifact.get("policy", {}).get("mode") not in ("route", "single"):
            raise ValueError("router artifact has no policy (route | single)")
        self.a = artifact
        self.sha = artifact_sha(artifact)

    @classmethod
    def load(cls, path: str | Path) -> Router:
        return cls(json.loads(Path(path).read_text()))

    @property
    def routes(self) -> bool:
        """False when the CV gate chose a single configuration: no features are computed at all."""
        return self.a["policy"]["mode"] == "route"

    @property
    def groups(self) -> list[str]:
        return list(self.a["feature_spec"]["groups"]) if self.routes else []

    @property
    def needs_llm(self) -> bool:
        return "llm" in self.groups

    @property
    def llm_questions(self) -> list[str]:
        return list(self.a["feature_spec"].get("llm_questions") or [])

    @property
    def llm_labeler(self) -> dict | None:
        return self.a["feature_spec"].get("llm_labeler")

    def route(self, raw: dict | None) -> dict:
        if not self.routes:
            return {
                "index": None,
                "config": self.a["policy"]["fallback"],
                "score": None,
                "pred": None,
                "scores": None,
                "policy": "single",
                "artifact_sha": self.sha,
            }
        x = transform(self.a["feature_spec"], raw)
        out = choose_routing(self.a["model"], x, self.a["routing"])
        out.update(policy="route", artifact_sha=self.sha)
        return out


def artifact_sha(artifact: dict) -> str:
    body = {k: v for k, v in artifact.items() if k != "report"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def save(path: str | Path, artifact: dict[str, Any]) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact))
    tmp.replace(path)
    return artifact_sha(artifact)
