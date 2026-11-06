"""Router over sparse profiling data with three query-dependent predictors.

The historical BRANE backend uses each configuration's own observed query sample. Model families and
hyperparameters are selected separately for quality, raw tokens and raw latency;
five outer query folds rerun three-fold inner selection for training replay.
No unobserved outcome is imputed as a label. See ``predictors``.
The joint backend instead fits three configuration-aware models over all
observed query/configuration pairs; it cross-fits by whole query.

The three-weight utility (``utility_norm="zscore-log-v1"``, user 2026-10-22) scales
quality and ln costs by the typical across-candidate spread of the honest
predictions (``fit_utility_norm``), frozen at fit time; ``utility_norm="minmax-raw"``
reproduces the earlier raw-cost min-max scale. Explicit ``cost_model="mean"``
retains the historical configuration-mean ablation. Deployment JSON supports all.

SLO preferences are selected on training replay using all three OOF predictions.
For a fixed policy, the sum of per-configuration stratified sample means of
1{policy(query)=configuration} times measured outcome estimates its pool value.
Because the catalog, representation and preferences are training-selected, this
replay is a tuning diagnostic, not an independent evaluation of the full system.
Held-out agent runs are required for the final policy comparison. Query feature
and routing overhead is charged separately by the evaluation harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trimem.optimizer import session_space_agentenv_v2 as space
from trimem.optimizer import unpaired_stats as us
from trimem.optimizer.query_router import brane, predictors, runtime, tabular


def _validate(cfg: dict) -> dict:
    """Space validation of a profiled configuration. Space -v5 fixes memory_form PER DATASET (agent-env raw,
    τ² compact; ``FIXED_MEMORY_FORM``): a configuration is validated against its own dataset's fixed form (the
    profile only holds configurations its dataset space produced)."""
    form = (
        cfg.get("memory_form", space.MEMORY_FORM)
        if cfg.get("plan") != "NoMemory"
        else space.MEMORY_FORM
    )
    if form not in set(space.FIXED_MEMORY_FORM.values()):
        raise ValueError(
            f"memory_form={form} is not a fixed per-dataset form ({space.VERSION})"
        )
    return space.validate(cfg, form)


# Router / selection protocol of the three-weight objective (bump when the rule, grid or chooser changes).
# v3 (user 2026-10-22): zscore-log-v1 utility scale + corner weights (w_Q >= 0) in the grid.
PROTOCOL = "perquery-utility3-querymetrics-v3"
DEFAULT_WEIGHTS = runtime.DEFAULT_WEIGHTS  # (w_Q, w_L, w_T)
GRID_STEP = 0.05  # simplex grid of the train weight selection
GRID_MIN_WQ = 0.0  # v3: corners included (231 points); v2 had 0.05 (210 points)
# test slices (cost share s = w_L + w_T = 1 - w_Q): latency-only (w_T = 0), token-only (w_L = 0), diagonal
# (w_L = w_T); s = 0 (pure quality) is shared by all three; the diagonal at s = 0.4 is the default weights.
SLICE_S = (0.0, 0.05, 0.1, 0.2, 0.4, 0.7)

# ---- LEGACY ln / lambda rule (ablation; artifacts with routing {"lambda", "w_tokens", "w_latency"})
# lambda grid of the SLO sweep (score = q - lambda * (w_T ln tokens + w_L ln latency))
LAMBDAS = (0.0, *[float(f"{v:.6g}") for v in np.logspace(-3, 1.5, 46)])
# Opt-PerQuery TEST lambda curve (user 2026-10-19): >= 5 lambdas spanning max quality (0) -> cheapest, equal
# cost weights; ``curve_lambdas`` adds the smallest grid lambda at which the training replay routes every query to
# the cheapest candidate when that lies above 1.0.
CURVE_LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)
CURVE_W = (0.5, 0.5)
# cost weights of the routing score per SLO: latency-only SLOs score latency only; SLO-2 bounds both.
SLO_WEIGHTS = {"SLO-1": (0.0, 1.0), "SLO-2": (0.5, 0.5), "SLO-3": (0.0, 1.0)}


@dataclass(frozen=True)
class SparseProfile:
    """Per-configuration own samples of one pool-profiling experiment (complete samples only)."""

    configs: tuple[dict, ...]
    samples: tuple[tuple[str, ...], ...]  # per config: query ids of its own sample
    quality: tuple[dict, ...]  # per config: qid -> quality in [0, 1]
    tokens: tuple[dict, ...]
    latency_ms: tuple[dict, ...]
    strata: dict  # qid -> stratum
    weights: dict  # stratum -> pool share
    sizes: dict  # stratum -> pool count

    @property
    def queries(self) -> list[str]:
        """Union of the samples (sorted, deterministic)."""
        return sorted(
            {q for s in self.samples for q in s},
            key=lambda q: (q.rsplit("_", 1)[0], int(q.rsplit("_", 1)[1])),
        )

    def est(self, c: int, field: str) -> us.Estimate:
        vals = getattr(self, field)[c]
        ids = list(self.samples[c])
        return us.stratified(
            [vals[q] for q in ids],
            [self.strata[q] for q in ids],
            self.weights,
            self.sizes,
        )


def from_estimates(
    est: dict[str, dict], strata: dict, weights: dict, sizes: dict
) -> SparseProfile:
    """``session_bo_agentenv.config_estimates`` output -> SparseProfile (config order = sorted keys)."""
    keys = sorted(est)
    return SparseProfile(
        tuple(est[k]["cfg"] for k in keys),
        tuple(tuple(est[k]["ids"]) for k in keys),
        tuple(est[k]["per_query"]["quality"] for k in keys),
        tuple(est[k]["per_query"]["total_tokens"] for k in keys),
        tuple(est[k]["per_query"]["e2e_ms"] for k in keys),
        dict(strata),
        dict(weights),
        dict(sizes),
    )


def without_empty_columns(p: SparseProfile) -> tuple[SparseProfile, list[int]]:
    """Fold-local observed catalog and its original column indices; no invented labels."""
    cols = [c for c, ids in enumerate(p.samples) if ids]
    if not cols:
        raise ValueError("No observed configurations in this training partition")
    if len(cols) == len(p.configs):
        return p, cols
    return SparseProfile(
        *(
            tuple(getattr(p, f)[c] for c in cols)
            for f in ("configs", "samples", "quality", "tokens", "latency_ms")
        ),
        p.strata,
        p.weights,
        p.sizes,
    ), cols


def means(p: SparseProfile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per configuration stratified mean quality, tokens, latency (ms)."""
    return (
        np.array([p.est(c, "quality").mean for c in range(len(p.configs))]),
        np.array([p.est(c, "tokens").mean for c in range(len(p.configs))]),
        np.array([p.est(c, "latency_ms").mean for c in range(len(p.configs))]),
    )


def candidates(
    p: SparseProfile,
    *,
    w: tuple[float, float] = (0.5, 0.5),
    tau_acc: float = brane.TAU_ACC,
    tau_cost: float = brane.TAU_COST,
    keep_best_per_plan: bool = True,
) -> list[int]:
    """BRANE fuzzy-Pareto set on the stratified means (+ best configuration of every plan)."""
    acc, tok, lat = means(p)
    cost = w[0] * np.log(tok) + w[1] * np.log(lat)
    cand = brane.fuzzy_pareto(acc, cost, tau_acc=tau_acc, tau_cost=tau_cost)
    if keep_best_per_plan:
        best: dict[str, int] = {}
        for i, c in enumerate(p.configs):
            if c["plan"] not in best or (acc[i], -cost[i]) > (
                acc[best[c["plan"]]],
                -cost[best[c["plan"]]],
            ):
                best[c["plan"]] = i
        cand += [
            i
            for i in sorted(best.values(), key=lambda i: (-acc[i], cost[i], i))
            if i not in cand
        ]
    return cand


def fit(
    X: dict[str, np.ndarray],
    p: SparseProfile,
    *,
    cand: list[int] | None = None,
    families: tuple[str, ...] = tabular.FAMILIES,
    trials: int = tabular.TRIALS,
    inner_k: int = tabular.INNER_K,
    outer_k: int = predictors.OUTER_K,
    cost_model: str = "query",
    n_jobs: int = 8,
    seed: int = 0,
    backend: str = "brane",
    shared_options: dict | None = None,
    utility_norm: str = runtime.SCALE_ZLOG,
    quality_kind: str | None = None,
) -> tuple[dict, dict[int, dict[str, list[float]]]]:
    """Router model (``runtime.predict_brane`` format) + per-candidate OOF predictions on its own sample.
    ``X`` maps every profiled query id to its transformed feature vector. ``utility_norm`` = the frozen scale
    of the three-weight utility (``zscore-log-v1`` default; ``minmax-raw`` = the pre-2026-10-22 scale)."""
    if utility_norm not in (runtime.SCALE_ZLOG, runtime.SCALE_MINMAX):
        raise ValueError(f"unknown utility_norm {utility_norm!r}")
    if quality_kind not in (None, "binary", "continuous", "raw_score"):
        raise ValueError("quality_kind must be binary, continuous, or None")
    if backend == "joint-selected":
        from trimem.optimizer.query_router import joint_selected

        if cost_model != "query" or shared_options is not None:
            raise ValueError(
                "joint-selected requires query costs and no shared options"
            )
        model, oof = joint_selected.fit(
            X, p, cand=cand, outer_k=outer_k, seed=seed, quality_kind=quality_kind
        )
        if utility_norm == runtime.SCALE_ZLOG:
            model["cost_norm"] = fit_utility_norm(model, X, p, oof)
        else:
            cols = model["candidate_columns"]
            model["cost_norm"] = runtime.cost_norm(
                np.array([p.tokens[c][q] for c in cols for q in p.samples[c]]),
                np.array([p.latency_ms[c][q] for c in cols for q in p.samples[c]]),
            )
        return model, oof
    if quality_kind == "raw_score":
        raise ValueError("raw_score quality requires the joint-selected backend")
    if backend == "joint":
        from trimem.optimizer.query_router import joint

        if cost_model != "query" or shared_options is not None:
            raise ValueError("joint backend requires query costs and no shared options")
        model, oof = joint.fit(
            X, p, cand=cand, outer_k=outer_k, seed=seed, quality_kind=quality_kind
        )
        if utility_norm == runtime.SCALE_ZLOG:
            model["cost_norm"] = fit_utility_norm(model, X, p, oof)
        else:
            cols = model["candidate_columns"]
            model["cost_norm"] = runtime.cost_norm(
                np.array([p.tokens[c][q] for c in cols for q in p.samples[c]]),
                np.array([p.latency_ms[c][q] for c in cols for q in p.samples[c]]),
            )
        return model, oof
    if backend in ("shared", "limeqo"):
        if quality_kind is not None:
            raise ValueError("explicit quality_kind is supported by the brane backend")
        from trimem.optimizer.query_router import shared

        if cost_model != "query":
            raise ValueError("shared backend supports query-dependent costs only")
        model, oof = shared.fit(
            X, p, cand=cand, outer_k=outer_k, seed=seed, **(shared_options or {})
        )
        if utility_norm == runtime.SCALE_ZLOG:
            model = {**model, "cost_norm": fit_utility_norm(model, X, p, oof)}
        return model, oof
    if backend not in ("brane", "brane-config") or shared_options is not None:
        raise ValueError(
            "backend must be brane or limeqo (alias: shared); shared_options requires limeqo"
        )
    cand = candidates(p) if cand is None else list(cand)
    config_schema = config_features = None
    if backend == "brane-config":
        from trimem.optimizer.query_router import joint

        config_schema = joint.config_schema([_validate(p.configs[c]) for c in cand])
        config_features = [
            joint.encode_config(_validate(p.configs[c]), config_schema) for c in cand
        ]
    union = p.queries
    keep = brane.feature_mask(np.array([X[q] for q in union]))
    allq = np.concatenate([np.array(list(d.values()), dtype=float) for d in p.quality])
    binary_labels = bool(set(np.unique(allq)) <= {0.0, 1.0})
    if quality_kind == "binary" and not binary_labels:
        raise ValueError("Binary quality requested for non-binary outcomes")
    binary = binary_labels if quality_kind is None else quality_kind == "binary"
    options = dict(
        binary=binary,
        families=families,
        trials=trials,
        inner_k=inner_k,
        outer_k=outer_k,
        seed=seed,
        cost_model=cost_model,
    )
    jobs = []
    for j, c in enumerate(cand):
        ids = list(p.samples[c])
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Aggregate repeated trials into one row per query before fitting"
            )
        query_rows = np.array([X[q] for q in ids])[:, keep]
        if config_features is not None:
            query_rows = np.hstack(
                (query_rows, np.tile(config_features[j], (len(ids), 1)))
            )
        jobs.append(
            (
                query_rows,
                np.array([p.quality[c][q] for q in ids], dtype=float),
                np.array([p.tokens[c][q] for q in ids]),
                np.array([p.latency_ms[c][q] for q in ids]),
                {
                    **options,
                    "sample_strata": [p.strata[q] for q in ids],
                    "stratum_weights": p.weights,
                },
            )
        )
    if n_jobs > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(n_jobs, len(jobs))) as ex:
            sel = list(ex.map(predictors.fit_job, jobs))
    else:
        sel = [predictors.fit_job(j) for j in jobs]
    acc, tok, lat = means(p)
    oof = {
        j: dict(zip(p.samples[c], sel[j]["oof"], strict=True))
        for j, c in enumerate(cand)
    }
    ln_tok = [s["ln_tokens"] for s in sel]
    ln_lat = [s["ln_latency"] for s in sel]
    if cost_model == "mean":
        # Preserve the historical stratified-mean deployment ablation.
        ln_tok = [{"const": float(math.log(tok[c]))} for c in cand]
        ln_lat = [{"const": float(math.log(lat[c]))} for c in cand]
    norm_tok = (
        tok[cand]
        if cost_model == "mean"
        else np.concatenate([[p.tokens[c][q] for q in p.samples[c]] for c in cand])
    )
    norm_lat = (
        lat[cand]
        if cost_model == "mean"
        else np.concatenate([[p.latency_ms[c][q] for q in p.samples[c]] for c in cand])
    )
    model = tabular.as_json(
        {
            "kind": "brane",
            "variant": "paper-sparse",
            "quality_kind": "binary" if binary else "continuous",
            "configs": [_validate(p.configs[c]) for c in cand],
            "candidate_columns": list(cand),
            "x_keep": keep,
            "quality": [s["quality"] for s in sel],
            "ln_tokens": ln_tok,
            "ln_latency": ln_lat,
            "cost_norm": runtime.cost_norm(norm_tok, norm_lat),
            "predictor_protocol": predictors.PROTOCOL,
            "selection": [s["selection"]["quality"] for s in sel],
            "selection_by_objective": {
                name: [s["selection"][name] for s in sel]
                for name in predictors.OBJECTIVES
            },
            "train_means": {
                "quality": acc[cand].tolist(),
                "tokens": tok[cand].tolist(),
                "latency_ms": lat[cand].tolist(),
                "n_per_candidate": [len(p.samples[c]) for c in cand],
            },
            "params": {
                "tau_acc": brane.TAU_ACC,
                "tau_cost": brane.TAU_COST,
                "families": list(families),
                "trials": trials,
                "inner_k": inner_k,
                "outer_k": outer_k,
                "cost_model": cost_model,
                "cost_target": "raw" if cost_model == "query" else cost_model,
                "cost_norm_scope": "configuration-means"
                if cost_model == "mean"
                else "observed-query-costs",
                "corr_drop": brane.CORR_DROP,
                "keep_best_per_plan": True,
                "training": "own-sample-per-config",
            },
        }
    )
    if config_features is not None:
        model.update(
            variant="brane-config",
            config_schema=config_schema,
            config_features=config_features,
        )
    if utility_norm == runtime.SCALE_ZLOG:
        model = {**model, "cost_norm": fit_utility_norm(model, X, p, oof)}
    return model, oof


def candidate_stats(p: SparseProfile, cols: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(candidates, 3) stratified means and SEs of [quality, ln tokens, ln latency] on each own sample."""
    means_, ses = [], []
    for c in cols:
        ids = list(p.samples[c])
        st = [p.strata[q] for q in ids]
        row = [us.stratified([p.quality[c][q] for q in ids], st, p.weights, p.sizes)]
        for src in (p.tokens, p.latency_ms):
            row.append(
                us.stratified(
                    [math.log(max(src[c][q], 1e-6)) for q in ids],
                    st,
                    p.weights,
                    p.sizes,
                )
            )
        means_.append([e.mean for e in row])
        ses.append([e.se for e in row])
    return np.asarray(means_, dtype=float), np.asarray(ses, dtype=float)


def fit_utility_norm(
    model: dict,
    X: dict[str, np.ndarray],
    p: SparseProfile,
    oof: dict[int, dict[str, list[float] | float]],
) -> dict:
    """``runtime.utility_norm`` on the training queries' honest predictions (``honest_predictions``) and the
    candidates' own-sample stratified means / SEs (TRAIN only; frozen in ``model['cost_norm']``)."""
    if oof:
        pred, src = (
            honest_predictions(model, X, p.queries, oof),
            "honest (own-sample OOF + full model)",
        )
    else:  # e.g. a shared model fitted without cross-fitting: in-sample predictions
        pred = np.array(
            [np.stack(runtime.predict(model, X[q]), axis=-1) for q in p.queries]
        )
        src = "in-sample (no OOF)"
    m, se = candidate_stats(p, list(model["candidate_columns"]))
    return {**runtime.utility_norm(pred, m, se), "prediction_source": src}


def objective_text(norm: dict) -> str:
    """Report text of the frozen utility scale."""
    if norm.get("scale") == runtime.SCALE_ZLOG:
        return (
            "w_Q q/s_Q - w_L ln lat/s_L - w_T ln tok/s_T (zscore-log-v1: s = median across-candidate spread of "
            "the honest predictions, floored at measurement SE; w >= 0, w_Q + w_L + w_T = 1)"
        )
    return "w_Q q - w_L lat~ - w_T tok~ (raw-scale min-max over candidates, w_Q + w_L + w_T = 1)"


def router_diagnostics(
    p: SparseProfile,
    model: dict,
    X: dict[str, np.ndarray],
    oof: dict[int, dict[str, list[float] | float]],
    weights: dict[str, tuple[float, float, float]],
) -> dict:
    """``utility_diagnostics`` of a fitted router on its training queries' honest predictions."""
    pred = honest_predictions(model, X, p.queries, oof)
    labels = [c["plan"] for c in model["configs"]]
    return {
        "n_queries": len(pred),
        "candidates": labels,
        **utility_diagnostics(pred, model["cost_norm"], weights, labels=labels),
    }


def utility_diagnostics(
    pred: np.ndarray,
    norm: dict,
    weights: dict[str, tuple[float, float, float]],
    *,
    equal: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    perturb: dict | None = None,
    labels: list[str] | None = None,
) -> dict:
    """Influence diagnostics of a frozen utility scale on predictions ``pred`` (n, C, 3):

    * ``variance_share``: under ``equal`` weights, the share of each term's across-candidate variance in the
      sum of the three (pooled over queries, and the median of the per-query shares);
    * ``flip_rate``: per weight vector, the share of queries whose decision changes when the chosen candidate's
      metric worsens by one ``perturb[metric]`` (default the norm's s; quality -s, ln costs +s);
    * ``corners``: the routing shares of (1,0,0), (0,1,0), (0,0,1) and the most-chosen candidate.
    """
    q, tok, lat = pred[..., 0], np.exp(pred[..., 1]), np.exp(pred[..., 2])
    if perturb is None:
        perturb = {k: norm[k]["s"] for k in runtime.NORM_METRICS}
    tq, tl, tt = runtime.utility_terms(q, tok, lat, equal, norm)
    var = {
        "quality": np.var(tq, axis=1),
        "tokens": np.var(tt, axis=1),
        "latency_ms": np.var(tl, axis=1),
    }
    tot = sum(var.values())
    ok = tot > 0
    pooled = {
        k: float(v.sum() / tot.sum()) if tot.sum() > 0 else None for k, v in var.items()
    }
    median = {
        k: float(np.median(v[ok] / tot[ok])) if ok.any() else None
        for k, v in var.items()
    }
    out = {
        "equal_weights": list(equal),
        "variance_share": {"pooled": pooled, "median_per_query": median},
        "perturbation": {k: float(v) for k, v in perturb.items()},
        "flip_rate": {},
        "corners": {},
    }
    ar = np.arange(len(pred))
    for name, w in {"equal": equal, **weights}.items():
        base = np.argmax(runtime.utility(q, tok, lat, w, norm), axis=1)
        rates = {}
        for m, k in enumerate(runtime.NORM_METRICS):
            pq, pt, pl = q.copy(), pred[..., 1].copy(), pred[..., 2].copy()
            if m == 0:
                pq[ar, base] -= perturb[k]
            elif m == 1:
                pt[ar, base] += perturb[k]
            else:
                pl[ar, base] += perturb[k]
            new = np.argmax(
                runtime.utility(pq, np.exp(pt), np.exp(pl), w, norm), axis=1
            )
            rates[k] = float(np.mean(new != base))
        out["flip_rate"][name] = {"weights": list(w), **rates}
    n_c = pred.shape[1]
    for name, w in (
        ("quality", (1.0, 0.0, 0.0)),
        ("latency", (0.0, 1.0, 0.0)),
        ("tokens", (0.0, 0.0, 1.0)),
    ):
        ch = np.argmax(runtime.utility(q, tok, lat, w, norm), axis=1)
        share = np.bincount(ch, minlength=n_c) / max(len(ch), 1)
        top = int(np.argmax(share))
        out["corners"][name] = {
            "weights": list(w),
            "share": share.tolist(),
            "top": top,
            "top_label": labels[top] if labels else None,
        }
    return out


def honest_predictions(
    model: dict,
    X: dict[str, np.ndarray],
    qids: list[str],
    oof: dict[int, dict[str, list[float] | float]],
) -> np.ndarray:
    """Replace all three predictions on observed queries with nested OOF values.

    Historical quality-only OOF maps remain readable with mean-cost artifacts.
    """
    normalized = {}
    for key, predictions in oof.items():
        if not isinstance(key, (int, str)) or str(key) != str(int(key)):
            raise ValueError("OOF candidate indices must be integers")
        j = int(key)  # JSON object keys are strings after loading oof.json.
        if not 0 <= j < len(model["configs"]) or j in normalized:
            raise ValueError("Invalid or duplicate OOF candidate index")
        normalized[j] = predictions
    oof = normalized
    if model.get("kind") in {
        "shared-lowrank",
        "joint-tabular",
        "joint-staged",
        "joint-selected",
    }:
        train_q = set(qids) & set(model.get("training_queries", model["oof_folds"]))
        if any(train_q - set(oof.get(j, {})) for j in range(len(model["configs"]))):
            raise ValueError(
                "Shared replay requires whole-query OOF for every candidate"
            )
    out = np.array([np.stack(runtime.predict(model, X[q]), axis=-1) for q in qids])
    for j, d in oof.items():
        for i, q in enumerate(qids):
            if q in d:
                v = np.asarray(d[q], dtype=float)
                if v.ndim == 0:  # legacy quality-only map
                    if model.get("params", {}).get("cost_model") == "query":
                        raise ValueError(
                            "Query-cost replay requires OOF predictions for all three objectives"
                        )
                    out[i, j, 0] = (
                        v
                        if model.get("quality_kind") == "raw_score"
                        else np.clip(v, 0.0, 1.0)
                    )
                elif v.shape == (3,) and np.isfinite(v).all():
                    out[i, j] = v
                    if model.get("quality_kind") != "raw_score":
                        out[i, j, 0] = np.clip(v[0], 0.0, 1.0)
                else:
                    raise ValueError("Invalid three-objective OOF prediction")
    return out


def route(pred: np.ndarray, lam: float, w: tuple[float, float]) -> np.ndarray:
    """Chosen candidate index per query (ties -> lowest index, as ``runtime.choose``)."""
    return np.argmax(
        runtime.scores(pred[..., 0], pred[..., 1], pred[..., 2], lam, *w), axis=1
    )


# Replay estimator (user 2026-10-22, ALFWorld v7 re-test). ``sum-v0`` (legacy): per metric, the SUM over candidates
# c of the stratified mean over c's own sample of 1{pi(x)=c} f_c(x) -- unbiased, but the implied routed shares
# (stratified mean of 1{pi(x)=c} over c's own sample) need not sum to 1 on sparse profiles, so a policy whose routed
# candidates happen to cover few of their own sample queries looks cheap AND bad, one covering many looks costly
# AND good (biases the SLO weight choice). ``selfnorm-v1`` (default): the Hajek / ratio form -- the same sum divided
# by the sum of those routed-share estimates (same samples, same stratified estimator); SE by linearisation
# (u_c(x) = 1{pi(x)=c} (f_c(x) - R), var = sum_c SE_strat(u_c)^2 / S^2). Identical to sum-v0 on dense profiles
# (every candidate measured on every query: S = 1 exactly); a constant metric replays to that constant exactly.
REPLAY_SELFNORM = "selfnorm-v1"
REPLAY_SUM = "sum-v0"
REPLAY_ESTIMATOR = REPLAY_SELFNORM


def routed_mean(
    parts: list[tuple[list[str], list[float], list[bool]]],
    strata: dict,
    weights: dict,
    sizes: dict,
    n: int,
    estimator: str = REPLAY_ESTIMATOR,
) -> tuple[us.Estimate, float]:
    """(estimate, routed-share sum S) of a routed policy's pool mean of one metric. ``parts`` = per candidate
    (its own sample ids, its measured values on them, whether the policy routes each of them to it); candidates
    with an empty sample contribute nothing. Pure."""
    if estimator not in (REPLAY_SELFNORM, REPLAY_SUM):
        raise ValueError(f"unknown replay estimator {estimator!r}")
    a_sum = s_sum = var_a = 0.0
    for ids, vals, hit in parts:
        if not ids:
            continue
        st = [strata[q] for q in ids]
        a = us.stratified(
            [v if h else 0.0 for v, h in zip(vals, hit, strict=True)],
            st,
            weights,
            sizes,
        )
        s = us.stratified([1.0 if h else 0.0 for h in hit], st, weights, sizes)
        a_sum, s_sum, var_a = a_sum + a.mean, s_sum + s.mean, var_a + a.se**2
    if estimator == REPLAY_SUM or s_sum <= 0.0:
        return us.Estimate(a_sum, math.sqrt(var_a), n), s_sum
    r = a_sum / s_sum
    var = 0.0
    for ids, vals, hit in parts:
        if not ids:
            continue
        u = [(v - r) if h else 0.0 for v, h in zip(vals, hit, strict=True)]
        var += us.stratified(u, [strata[q] for q in ids], weights, sizes).se ** 2
    return us.Estimate(r, math.sqrt(var) / s_sum, n), s_sum


def replay(
    p: SparseProfile,
    model: dict,
    qids: list[str],
    choice: np.ndarray,
    overhead_ms: dict[str, float] | None = None,
    estimator: str = REPLAY_ESTIMATOR,
    *,
    overhead_tokens: dict[str, float] | None = None,
) -> dict:
    """Training replay estimate conditional on the selected catalog (see module doc and ``REPLAY_ESTIMATOR``):
    per metric, over candidates c, the stratified mean over c's own sample of 1{pi(x)=c} f_c(x), summed
    (``sum-v0``) or self-normalised by the summed routed-share estimates (``selfnorm-v1``, default)."""
    pos = {q: i for i, q in enumerate(qids)}
    res, s_sum = {}, None
    for field, src in (
        ("quality", p.quality),
        ("tokens", p.tokens),
        ("latency_ms", p.latency_ms),
    ):
        parts = []
        for j, c in enumerate(model["candidate_columns"]):
            ids = list(p.samples[c])
            parts.append(
                (ids, [src[c][q] for q in ids], [choice[pos[q]] == j for q in ids])
            )
        res[field], s_sum = routed_mean(
            parts, p.strata, p.weights, p.sizes, len(qids), estimator
        )
    ovh = (
        float(np.mean([overhead_ms.get(q, 0.0) for q in qids])) if overhead_ms else 0.0
    )
    token_costs = (
        [overhead_tokens[q] for q in qids] if overhead_tokens is not None else [0.0]
    )
    if any(not math.isfinite(v) or v < 0 for v in token_costs):
        raise ValueError("Token overhead must be finite and nonnegative")
    tok_ovh = float(np.mean(token_costs))
    share = np.bincount(choice, minlength=len(model["candidate_columns"])) / len(choice)
    return {
        "quality": res["quality"].mean,
        "quality_se": res["quality"].se,
        "tokens": res["tokens"].mean + tok_ovh,
        "latency_ms": res["latency_ms"].mean + ovh,
        "latency_se_ms": res["latency_ms"].se,
        "overhead_ms": ovh,
        **({"overhead_tokens": tok_ovh} if overhead_tokens is not None else {}),
        "share": share.tolist(),
        "replay_estimator": estimator,
        "routed_share_sum": s_sum,
    }


def curve(
    p: SparseProfile,
    model: dict,
    X: dict[str, np.ndarray],
    oof: dict[int, dict[str, list[float] | float]],
    w: tuple[float, float],
    *,
    lambdas: tuple[float, ...] = LAMBDAS,
    overhead_ms: dict[str, float] | None = None,
) -> list[dict]:
    """Replay estimate of the routed policy for every lambda (fixed cost weights ``w``); also the
    direct-method (predicted) quality as a diagnostic."""
    qids = p.queries
    pred = honest_predictions(model, X, qids, oof)
    rows = []
    for lam in lambdas:
        ch = route(pred, lam, w)
        r = replay(p, model, qids, ch, overhead_ms)
        rows.append(
            {
                "lambda": lam,
                "w_tokens": w[0],
                "w_latency": w[1],
                **r,
                "q": r["quality"],
                "lat": r["latency_ms"] / 1000,
                "tok": r["tokens"],
                "direct_quality": float(pred[np.arange(len(qids)), ch, 0].mean()),
            }
        )
    return rows


def feasible(row: dict, bound: dict) -> bool:
    return all(row["q"] >= x if k == "q" else row[k] <= x for k, x in bound.items())


def choose_lambda(rows: list[dict], bound: dict, objective: str) -> dict | None:
    """SLO choice over the lambda curve, same rule and tie-breaks as ``slo_select_bo.choose``
    (max quality -> lower latency -> lower tokens; min latency -> higher quality -> lower tokens), then
    the smaller lambda."""
    ok = [r for r in rows if feasible(r, bound)]
    if not ok:
        return None
    if objective == "max_quality":
        return min(ok, key=lambda r: (-r["q"], r["lat"], r["tok"], r["lambda"]))
    return min(ok, key=lambda r: (r["lat"], -r["q"], r["tok"], r["lambda"]))


def artifact(
    *,
    dataset: str,
    spec: dict,
    model: dict,
    lam: float,
    w: tuple[float, float],
    fallback: dict,
    trained_on: dict,
    report: dict,
) -> dict:
    """Deployable router artifact (``runtime.Router``). The SLO name lives in ``report`` (outside the sha),
    so two SLOs choosing the same (lambda, w) share one artifact identity and one test run."""
    return {
        "kind": runtime.KIND,
        "version": runtime.VERSION,
        "dataset": dataset,
        "feature_spec": spec,
        "model": model,
        "routing": {
            "lambda": lam,
            "w_tokens": w[0],
            "w_latency": w[1],
            "rule": "slo-replay",
        },
        "policy": {"mode": "route", "fallback": _validate(fallback)},
        "trained_on": trained_on,
        "report": report,
    }


def curve_lambdas(
    model: dict,
    train_rows: list[dict],
    base: tuple[float, ...] = CURVE_LAMBDAS,
    w: tuple[float, float] = CURVE_W,
) -> list[float]:
    """Test-curve lambdas: ``base`` plus the first lambda of ``train_rows`` (a ``curve`` at weights ``w``) whose
    routing sends every query to the cheapest candidate, when that lambda is larger than max(base)."""
    if model.get("kind") in {
        "shared-lowrank",
        "joint-tabular",
        "joint-staged",
        "joint-selected",
    }:
        return sorted(set(base))  # shared costs have no globally cheapest candidate
    if any(
        "const" not in m for field in ("ln_tokens", "ln_latency") for m in model[field]
    ):
        return sorted(
            set(base)
        )  # query-dependent costs have no single globally cheapest configuration
    cost = [
        w[0] * a["const"] + w[1] * b["const"]
        for a, b in zip(model["ln_tokens"], model["ln_latency"], strict=True)
    ]
    cheap = min(range(len(cost)), key=lambda i: (cost[i], i))
    out = sorted(set(base))
    full = [
        r["lambda"]
        for r in sorted(train_rows, key=lambda r: r["lambda"])
        if r["share"][cheap] >= 1.0 - 1e-12
    ]
    if full and full[0] > max(out):
        out.append(full[0])
    return out


def route_test(
    art: dict,
    records: list[dict],
    lambdas_w: list[tuple[str, float, tuple[float, float]]],
    overhead_ms: dict[str, float],
) -> dict[str, dict]:
    """Offline routing of the test queries with a FROZEN artifact's spec + model: arm name -> query id ->
    {config, overhead_ms}. Exactly ``runtime.choose`` (the deployed decision rule) at each (lambda, w)."""
    out = {name: {} for name, _, _ in lambdas_w}
    for r in records:
        x = runtime.transform(art["feature_spec"], r)
        for name, lam, w in lambdas_w:
            d = runtime.choose(art["model"], x, lam, *w)
            out[name][r["query_id"]] = {
                "config": _validate(d["config"]),
                "overhead_ms": overhead_ms.get(r["query_id"], 0.0),
            }
    return out


# ------------------------------------------------------------------ three-weight objective (PROTOCOL)


def weight_grid(
    step: float = GRID_STEP, w_q_min: float = GRID_MIN_WQ
) -> list[tuple[float, float, float]]:
    """Every (w_Q, w_L, w_T) on the simplex with coordinates on multiples of ``step`` and w_Q >= ``w_q_min``
    (w_Q descending, then w_L descending). step 0.05: w_Q >= 0 (default, v3) 231 points; w_Q >= 0.05 210."""
    n = round(1 / step)
    if not math.isclose(n * step, 1.0):
        raise ValueError("1 / step must be an integer")
    lo = math.ceil(w_q_min / step - 1e-9)
    return [
        (round(iq * step, 6), round(il * step, 6), round((n - iq - il) * step, 6))
        for iq in range(n, max(lo, 0) - 1, -1)
        for il in range(n - iq, -1, -1)
    ]


def slices(
    s: tuple[float, ...] = SLICE_S,
) -> dict[str, list[tuple[float, float, float]]]:
    """Test slices, name -> weight vectors by cost share s: latency-only, token-only and diagonal."""
    r = lambda v: round(v, 6)  # noqa: E731
    return {
        "latency": [(r(1 - v), r(v), 0.0) for v in s],
        "tokens": [(r(1 - v), 0.0, r(v)) for v in s],
        "diagonal": [(r(1 - v), r(v / 2), r(v / 2)) for v in s],
    }


def route_weights(
    pred: np.ndarray, weights: tuple[float, float, float], norm: dict
) -> np.ndarray:
    """Chosen candidate per query under the three-weight utility (ties -> lowest index, as ``runtime.choose``);
    ``pred`` = (n, candidates, [quality, ln tokens, ln latency])."""
    u = runtime.utility(
        pred[..., 0], np.exp(pred[..., 1]), np.exp(pred[..., 2]), weights, norm
    )
    return np.argmax(u, axis=1)


def weight_curve(
    p: SparseProfile,
    model: dict,
    X: dict[str, np.ndarray],
    oof: dict[int, dict[str, list[float] | float]],
    weights: list[tuple[float, float, float]],
    *,
    overhead_ms: dict[str, float] | None = None,
    overhead_tokens: dict[str, float] | None = None,
) -> list[dict]:
    """Train OOF replay estimate of the routed policy at every weight vector (+ the direct-method quality as a
    diagnostic). Identical routings share one replay."""
    qids = p.queries
    pred = honest_predictions(model, X, qids, oof)
    cache: dict[bytes, dict] = {}
    rows = []
    for w in weights:
        wq, wl, wt = (round(v, 6) for v in runtime.normalize_weights(w))
        ch = route_weights(pred, (wq, wl, wt), model["cost_norm"])
        key = ch.astype(np.int32).tobytes()
        if key not in cache:
            cache[key] = replay(
                p, model, qids, ch, overhead_ms, overhead_tokens=overhead_tokens
            )
        r = cache[key]
        rows.append(
            {
                "w_quality": wq,
                "w_latency": wl,
                "w_tokens": wt,
                **r,
                "q": r["quality"],
                "lat": r["latency_ms"] / 1000,
                "tok": r["tokens"],
                "direct_quality": float(pred[np.arange(len(qids)), ch, 0].mean()),
            }
        )
    return rows


def weights_of(row: dict) -> tuple[float, float, float]:
    return (row["w_quality"], row["w_latency"], row["w_tokens"])


def slack(row: dict, bound: dict) -> float:
    """Relative headroom to the tightest SLO bound (>= 0 iff feasible)."""
    out = []
    for k, b in bound.items():
        d = row["q"] - b if k == "q" else b - row[k]
        out.append(d / abs(b) if b else d)
    return min(out) if out else math.inf


def _primary_key(objective: str):
    if objective == "max_quality":
        return lambda r, b: (
            -r["q"],
            -slack(r, b),
            -r["w_quality"],
            r["w_latency"],
            r["w_tokens"],
        )
    if objective == "min_latency":
        return lambda r, b: (
            r["lat"],
            -slack(r, b),
            -r["w_quality"],
            r["w_latency"],
            r["w_tokens"],
        )
    raise ValueError(f"unknown SLO objective {objective!r}")


def choose_weights(rows: list[dict], bound: dict, objective: str) -> dict | None:
    """Primary (frozen) SLO choice over the weight grid replay: among feasible rows, max quality (SLO
    ``min_latency``: min latency); ties -> more SLO slack -> larger w_Q (-> smaller w_L, w_T: deterministic).
    None when no row is feasible."""
    ok = [r for r in rows if feasible(r, bound)]
    if not ok:
        return None
    key = _primary_key(objective)
    return min(ok, key=lambda r: key(r, bound))


def choose_weights_robust(rows: list[dict], bound: dict, objective: str) -> dict | None:
    """1-SE variant (REPORTED, never deployed): among feasible rows within one SE of the primary pick's
    objective (quality, or latency for ``min_latency``), the one with the most SLO slack (-> larger w_Q)."""
    best = choose_weights(rows, bound, objective)
    if best is None:
        return None
    ok = [r for r in rows if feasible(r, bound)]
    if objective == "max_quality":
        near = [r for r in ok if r["q"] >= best["q"] - best["quality_se"]]
    else:
        near = [r for r in ok if r["lat"] <= best["lat"] + best["latency_se_ms"] / 1000]
    return min(
        near,
        key=lambda r: (
            -slack(r, bound),
            -r["w_quality"],
            r["w_latency"],
            r["w_tokens"],
        ),
    )


def nearest_infeasible(rows: list[dict], bound: dict) -> dict | None:
    """Row with the smallest worst relative violation (logged when an SLO has no feasible grid point)."""
    return max(rows, key=lambda r: slack(r, bound)) if rows else None


def artifact_weights(
    *,
    dataset: str,
    spec: dict,
    model: dict,
    weights: tuple[float, float, float],
    fallback: dict,
    trained_on: dict,
    report: dict,
) -> dict:
    """Deployable three-weight router artifact (``runtime.Router``); the SLO name lives in ``report``."""
    if "cost_norm" not in model:
        raise ValueError("model has no cost_norm")
    wq, wl, wt = (round(v, 6) for v in runtime.normalize_weights(weights))
    return {
        "kind": runtime.KIND,
        "version": runtime.VERSION,
        "dataset": dataset,
        "feature_spec": spec,
        "model": model,
        "routing": {
            "rule": runtime.RULE_UTILITY,
            "protocol": PROTOCOL,
            "w_quality": wq,
            "w_latency": wl,
            "w_tokens": wt,
        },
        "policy": {"mode": "route", "fallback": _validate(fallback)},
        "trained_on": trained_on,
        "report": report,
    }


def route_test_weights(
    art: dict,
    records: list[dict],
    arms: list[tuple[str, tuple[float, float, float]]],
    overhead_ms: dict[str, float],
) -> dict[str, dict]:
    """Offline routing of the test queries with a FROZEN artifact at each weight vector: arm name -> query id
    -> {config, overhead_ms}. Exactly ``runtime.choose_utility`` (the deployed decision rule)."""
    out = {name: {} for name, _ in arms}
    for r in records:
        x = runtime.transform(art["feature_spec"], r)
        for name, w in arms:
            d = runtime.choose_utility(art["model"], x, w)
            out[name][r["query_id"]] = {
                "config": _validate(d["config"]),
                "overhead_ms": overhead_ms.get(r["query_id"], 0.0),
            }
    return out


def arm_name(weights: tuple[float, float, float]) -> str:
    """Test-arm name of a weight vector, e.g. ``Opt-PerQuery-w0.6-0.2-0.2`` (w_Q-w_L-w_T)."""
    return "Opt-PerQuery-w" + "-".join(f"{v:g}" for v in weights)
