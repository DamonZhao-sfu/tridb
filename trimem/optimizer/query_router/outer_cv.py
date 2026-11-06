"""Outer 5-fold cross-validation of Opt-OneConfig and Opt-PerQuery (user decision 2026-10-21).

The TRAIN query set (the union of the profiled samples) is split into ``K`` stratified outer folds with a fixed,
recorded seed. For every fold k, using ONLY the other K - 1 folds:

* the feature transform (vocabularies, PCA, scaler), the BRANE candidate set, the per-candidate predictors
  (with their own nested 5 x 3-fold model selection, ``sparse.fit``) and the cost normalisation are fitted;
* Opt-OneConfig picks one configuration per SLO from the training-fold point estimates, with SLO bounds
  recomputed from those estimates;
* Opt-PerQuery picks a weight vector per SLO on the training-fold out-of-fold replay.

The held-out fold k is then evaluated with measured outcomes only (``heldout_eval``), and the dataset driver
routes the TEST set with fold k's router (``run_fold`` returns the artifact and the test routes). Reported
numbers are mean +- std ACROSS the K folds (``mean_std``); per-fold SEs across tasks are kept separately.

Sparse profiles (ALFWorld / ScienceWorld: every configuration has its own stratified sample) mean a held-out
query is usually measured for only some candidates. ``heldout_eval`` uses the same estimator as the training
replay (``sparse.replay``): for each candidate c, the stratified mean over c's own sample inside the held-out
fold of 1{route(q) = c} f_c(q), summed over candidates and (``sparse.REPLAY_ESTIMATOR`` selfnorm-v1, user
2026-10-22) divided by the summed routed-share estimates (Hajek form; sum-v0 = the plain sum). Routing is defined for every held-out query (features
exist for all), each sample is a stratified random draw independent of the router, so the sum is an unbiased
estimate of the policy's held-out mean that uses no imputed outcome (the self-normalised ratio trades a small
ratio bias for shares that sum to 1). The share of held-out queries whose routed
pair was directly measured (``coverage``) and the model-predicted (plug-in) quality are reported as diagnostics.
On dense profiles (tau2: every configuration on every task) coverage is 1 and the estimator is the plain
stratified mean of the realized routed outcomes.

Pure (numpy + the router modules); no DB, no episodes. See docs/trimem_query_router_plan_v0.1.0.md, addendum
"outer 5-fold CV v0.3.0".
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np

from trimem.optimizer import unpaired_stats as us
from trimem.optimizer.query_router import features, runtime, sparse

PROTOCOL = "outer-cv5-v1"
K = 5
FOLD_SEED = 20261021
METRICS = ("quality", "tokens", "latency_s", "gain")


def stratified_folds(
    qids: list[str], strata: dict[str, str], k: int = K, seed: int = FOLD_SEED
) -> dict[str, int]:
    """Query -> outer fold in [0, k). Within each stratum (sorted) the sorted queries are permuted with one seeded
    generator and dealt round-robin; the deal offset carries across strata so fold sizes differ by at most one.
    Deterministic in (set of qids, strata, k, seed)."""
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(set(qids)) != len(qids):
        raise ValueError("duplicate query ids")
    rng = np.random.default_rng(seed)
    by: dict[str, list[str]] = defaultdict(list)
    for q in sorted(qids):
        by[strata[q]].append(q)
    out, off = {}, 0
    for h in sorted(by):
        qs = by[h]
        for i, j in enumerate(rng.permutation(len(qs))):
            out[qs[int(j)]] = (off + i) % k
        off = (off + len(qs)) % k
    return out


def fold_block(
    folds: dict[str, int], strata: dict[str, str], k: int = K, seed: int = FOLD_SEED
) -> dict:
    """Recorded identity of a fold assignment (seed, sizes, per-fold strata counts, sha of the assignment)."""
    sha = hashlib.sha256(json.dumps(sorted(folds.items())).encode()).hexdigest()[:16]
    return {
        "protocol": PROTOCOL,
        "k": k,
        "seed": seed,
        "stratified_by": "stratum (task type / domain)",
        "sha": sha,
        "sizes": [sum(v == f for v in folds.values()) for f in range(k)],
        "strata": [
            dict(sorted(Counter(strata[q] for q, v in folds.items() if v == f).items()))
            for f in range(k)
        ],
    }


def restrict(p: sparse.SparseProfile, keep: set[str]) -> sparse.SparseProfile:
    """The profile on the queries in ``keep`` only (configuration order and indices unchanged)."""
    samples = tuple(tuple(q for q in s if q in keep) for s in p.samples)

    def sub(ds: tuple[dict, ...]) -> tuple[dict, ...]:
        return tuple({q: d[q] for q in s} for d, s in zip(ds, samples, strict=True))

    return sparse.SparseProfile(
        p.configs,
        samples,
        sub(p.quality),
        sub(p.tokens),
        sub(p.latency_ms),
        p.strata,
        p.weights,
        p.sizes,
    )


def _est(p: sparse.SparseProfile, ids: list[str], vals: list[float]) -> us.Estimate:
    return us.stratified(vals, [p.strata[q] for q in ids], p.weights, p.sizes)


def nomemory_est(
    p: sparse.SparseProfile, nm: dict[str, float], keep: set[str]
) -> us.Estimate | None:
    ids = [q for q in nm if q in keep]
    return _est(p, ids, [nm[q] for q in ids]) if ids else None


def points(
    p: sparse.SparseProfile,
    cols: list[int],
    nm: us.Estimate | None,
    labels: dict[int, str],
) -> list[dict]:
    """Opt-OneConfig candidates on a (restricted) profile: stratified q / tokens / latency (s), unpaired gain."""
    out = []
    for c in cols:
        ids = list(p.samples[c])
        if not ids:
            continue
        q = _est(p, ids, [p.quality[c][i] for i in ids])
        tok = _est(p, ids, [p.tokens[c][i] for i in ids])
        lat = _est(p, ids, [p.latency_ms[c][i] for i in ids])
        g = us.diff(q, nm) if nm is not None else None
        out.append(
            {
                "col": c,
                "label": labels.get(c, str(c)),
                "cfg": p.configs[c],
                "n": len(ids),
                "q": q.mean,
                "q_se": q.se,
                "tok": tok.mean,
                "lat": lat.mean / 1000,
                "gain": None if g is None else g.mean,
                "gain_se": None if g is None else g.se,
            }
        )
    return out


def heldout_eval(
    p: sparse.SparseProfile,
    cols: list[int],
    qids: list[str],
    choice: np.ndarray,
    nm: dict[str, float],
    overhead_ms: dict[str, float] | None = None,
    pred_q: np.ndarray | None = None,
) -> dict:
    """Held-out value of a routing (``choice[i]`` = index into ``cols`` for ``qids[i]``) on a profile restricted to
    the held-out fold, measured outcomes only (module doc). Gain vs NoMemory is paired when every routed pair and
    every NoMemory outcome is measured (dense), else the unpaired difference with NoMemory's held-out sample."""
    pos = {q: i for i, q in enumerate(qids)}
    res, missing, s_sum = {}, set(), None
    for field, src in (
        ("quality", p.quality),
        ("tokens", p.tokens),
        ("latency_ms", p.latency_ms),
    ):
        parts = []
        for j, c in enumerate(cols):
            ids = [q for q in p.samples[c] if q in pos]
            if not ids:
                if bool((choice == j).any()):
                    missing.add(j)
                continue
            parts.append(
                (ids, [src[c][q] for q in ids], [choice[pos[q]] == j for q in ids])
            )
        # same estimator as the training replay (sparse.REPLAY_ESTIMATOR: self-normalised by default)
        res[field], s_sum = sparse.routed_mean(
            parts, p.strata, p.weights, p.sizes, len(qids)
        )
    have = [set(p.samples[c]) for c in cols]
    direct = [q in have[int(choice[i])] for i, q in enumerate(qids)]
    ovh = (
        float(np.mean([overhead_ms.get(q, 0.0) for q in qids])) if overhead_ms else 0.0
    )
    dense = all(direct) and all(q in nm for q in qids)
    if dense:
        g = _est(
            p,
            qids,
            [p.quality[cols[int(choice[i])]][q] - nm[q] for i, q in enumerate(qids)],
        )
        gain, gain_se, kind = g.mean, g.se, "paired"
    else:
        e = nomemory_est(p, nm, set(qids))
        gain = None if e is None else res["quality"].mean - e.mean
        gain_se = None if e is None else math.sqrt(res["quality"].se ** 2 + e.se**2)
        kind = "unpaired"
    share = np.bincount(choice, minlength=len(cols)) / max(len(choice), 1)
    return {
        "quality": res["quality"].mean,
        "quality_se": res["quality"].se,
        "tokens": res["tokens"].mean,
        "tokens_se": res["tokens"].se,
        "latency_s": (res["latency_ms"].mean + ovh) / 1000,
        "latency_se_s": res["latency_ms"].se / 1000,
        "overhead_ms": ovh,
        "gain": gain,
        "gain_se": gain_se,
        "gain_kind": kind,
        "n": len(qids),
        "coverage": float(np.mean(direct)) if direct else 0.0,
        "incomplete_candidates": sorted(missing),
        "share": share.tolist(),
        "replay_estimator": sparse.REPLAY_ESTIMATOR,
        "routed_share_sum": s_sum,
        "predicted_quality": None if pred_q is None else float(np.mean(pred_q)),
    }


def _summary(row: dict) -> dict:
    return {
        k: row[k]
        for k in (
            "w_quality",
            "w_latency",
            "w_tokens",
            "q",
            "quality_se",
            "lat",
            "tok",
            "direct_quality",
            "share",
        )
        if k in row
    }


def run_fold(
    k: int,
    *,
    prof: sparse.SparseProfile,
    folds: dict[str, int],
    records: dict[str, dict],
    groups: tuple[str, ...],
    oneconfig_cols: list[int],
    labels: dict[int, str],
    nomemory: dict[str, float],
    bounds_fn: Callable[[list[dict]], dict],
    choose_fn: Callable[[list[dict], dict, str], dict | None],
    objectives: dict[str, str],
    test_weights: dict[str, tuple[float, float, float]],
    fit_kwargs: dict,
    test_records: list[dict] | None = None,
    dataset: str = "",
    fallback: dict | None = None,
    trained_on: dict | None = None,
) -> dict:
    """One outer fold (module doc). ``objectives`` = SLO name -> objective; ``test_weights`` = arm label -> (w_Q,
    w_L, w_T). Returns the fold record (JSON-able) plus ``artifact`` and ``test_routes`` (arm -> qid -> route)."""
    union = prof.queries
    train_q = [q for q in union if folds[q] != k]
    held = [q for q in union if folds[q] == k]
    tr_set, ho_set = set(train_q), set(held)
    train_records = [records[q] for q in train_q]
    spec = (
        features.fit_spec(train_records, groups, emb_dims=24)
        if fit_kwargs.get("backend") == "joint-selected"
        else features.fit_spec(train_records, groups)
    )
    t0 = time.perf_counter()
    X = {q: runtime.transform(spec, records[q]) for q in union}
    tr_ms = (time.perf_counter() - t0) * 1000 / len(union)
    Xtr = {q: X[q] for q in train_q}
    p_tr, p_ho = restrict(prof, tr_set), restrict(prof, ho_set)
    if set(p_tr.queries) & ho_set or set(Xtr) & ho_set:
        raise AssertionError("held-out queries leaked into the fold's training data")
    probe = "probe" in groups
    ovh = {
        q: (records[q]["overhead"]["probe_ms"] if probe else 0.0) + tr_ms for q in union
    }
    model, oof = sparse.fit(Xtr, p_tr, **fit_kwargs)
    cols = list(model["candidate_columns"])
    if set().union(*(set(d) for d in oof.values())) & ho_set:
        raise AssertionError("held-out queries in the fold's OOF predictions")
    # Opt-OneConfig: points and SLO bounds on the training folds only
    nm_tr = nomemory_est(p_tr, nomemory, tr_set)
    pts = points(p_tr, oneconfig_cols, nm_tr, labels)
    B = bounds_fn(pts)
    one = {}
    for n, obj in objectives.items():
        pick = choose_fn(pts, B[n], obj)
        if pick is None:
            one[n] = None
            continue
        c = pick["col"]
        ev = heldout_eval(p_ho, [c], held, np.zeros(len(held), dtype=int), nomemory)
        one[n] = {
            "col": c,
            "label": pick["label"],
            "cfg": pick["cfg"],
            "train": {
                x: pick[x] for x in ("q", "q_se", "tok", "lat", "gain", "gain_se", "n")
            },
            "heldout": ev,
        }
    # Opt-PerQuery: training-fold OOF replay over the grid (+ the test weights), SLO weights chosen on it
    grid = sparse.weight_grid()
    extra = [tuple(w) for w in test_weights.values() if tuple(w) not in set(grid)]
    rows = sparse.weight_curve(
        p_tr, model, Xtr, oof, grid + list(dict.fromkeys(extra)), overhead_ms=ovh
    )
    by_w = {sparse.weights_of(r): r for r in rows}
    grid_rows = [by_w[w] for w in grid]
    pred = sparse.honest_predictions(
        model, X, held, {}
    )  # the fold model never saw these queries
    ar = np.arange(len(held))

    def evaluate(w: tuple[float, float, float]) -> dict:
        ch = sparse.route_weights(pred, w, model["cost_norm"])
        return heldout_eval(p_ho, cols, held, ch, nomemory, ovh, pred[ar, ch, 0])

    per_w = {
        label: {
            "weights": list(w),
            "train_replay": _summary(by_w[tuple(w)]),
            "heldout": evaluate(tuple(w)),
        }
        for label, w in test_weights.items()
    }
    per_slo = {}
    for n, obj in objectives.items():
        pick = sparse.choose_weights(grid_rows, B[n], obj)
        per_slo[n] = (
            None
            if pick is None
            else {
                "weights": list(sparse.weights_of(pick)),
                "train_replay": _summary(pick),
                "heldout": evaluate(sparse.weights_of(pick)),
            }
        )
    art = sparse.artifact_weights(
        dataset=dataset,
        spec=spec,
        model=model,
        weights=sparse.DEFAULT_WEIGHTS,
        fallback=fallback if fallback is not None else model["configs"][0],
        trained_on={
            **(trained_on or {}),
            "outer_cv": {
                "protocol": PROTOCOL,
                "fold": k,
                "k": K,
                "train_queries": len(train_q),
            },
        },
        report={
            "outer_cv_fold": k,
            "slo_weights": {n: v and v["weights"] for n, v in per_slo.items()},
        },
    )
    routes = None
    if test_records is not None:
        t_ovh = {
            r["query_id"]: (r["overhead"]["probe_ms"] if probe else 0.0) + tr_ms
            for r in test_records
        }
        arms = [(label, tuple(w)) for label, w in test_weights.items()]
        routes = sparse.route_test_weights(art, test_records, arms, t_ovh)
    return {
        "fold": k,
        "n_train": len(train_q),
        "n_heldout": len(held),
        "isolation": {
            "heldout_in_fit": 0,
            "heldout_in_oof": 0,
            "heldout_in_transform_fit": 0,
            "fit_queries_sha": hashlib.sha256(
                json.dumps(sorted(p_tr.queries)).encode()
            ).hexdigest()[:16],
        },
        "bounds": B,
        "transform_ms": tr_ms,
        "router": {
            "candidates": len(cols),
            "candidate_configs": model["configs"],
            "model_kind": model["kind"],
            "families": [s.get("family") for s in model.get("selection", [])],
            "cost_norm": model.get("cost_norm"),  # frozen utility scale of this fold
        },
        "oneconfig": one,
        "perquery": per_w,
        "perquery_slo": per_slo,
        "artifact": art,
        "test_routes": routes,
    }


def mean_std(values: list[float | None]) -> dict:
    """Mean and sample standard deviation (ddof = 1) across folds of the finite values; n = folds counted."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
        "n": len(v),
    }


def heldout_arms(fold: dict) -> dict[str, dict | None]:
    """Arm name -> held-out evaluation of one fold record."""
    out = {
        f"Opt-OneConfig-{n}": v and v["heldout"] for n, v in fold["oneconfig"].items()
    }
    out |= {
        f"Opt-PerQuery-{label}": v["heldout"] for label, v in fold["perquery"].items()
    }
    out |= {
        f"Opt-PerQuery-{n}": v and v["heldout"] for n, v in fold["perquery_slo"].items()
    }
    return out


def summarize(per_fold: list[dict[str, dict | None]]) -> dict[str, dict]:
    """Arm -> metric -> mean +- std across folds, plus the mean within-fold task SE of quality and gain
    (``se_tasks_mean``; SE across tasks, not across folds). Accepts held-out or test per-fold arm metrics with
    keys quality / quality_se / tokens / latency_s / gain (or paired_gain_vs_nomemory) / gain_se."""
    names = list(dict.fromkeys(n for f in per_fold for n in f))
    out = {}
    for n in names:
        rows = [f.get(n) for f in per_fold]
        rows = [r for r in rows if r and r.get("quality") is not None]
        g = [r.get("gain", r.get("paired_gain_vs_nomemory")) for r in rows]
        gse = [r.get("gain_se", r.get("paired_gain_se")) for r in rows]
        out[n] = {
            "folds": len(rows),
            "quality": mean_std([r["quality"] for r in rows]),
            "tokens": mean_std([r["tokens"] for r in rows]),
            "latency_s": mean_std([r["latency_s"] for r in rows]),
            "gain": mean_std(g),
            # agent steps (user 2026-10-21): absent / None in a fold -> that fold is not counted (never 0)
            "steps": mean_std([r.get("steps") for r in rows]),
            "steps_success": mean_std([r.get("steps_success") for r in rows]),
            "se_tasks_mean": {
                "quality": mean_std([r.get("quality_se") for r in rows])["mean"],
                "gain": mean_std(gse)["mean"],
                "steps": mean_std([r.get("steps_se") for r in rows])["mean"],
            },
        }
        cov = [r.get("coverage") for r in rows if r.get("coverage") is not None]
        if cov:
            out[n]["coverage"] = mean_std(cov)
    return out


def test_arms(
    folds: list[dict], oneconfig: dict[int, dict[str, dict | None]] | None = None
) -> dict[str, dict]:
    """Routing-table arms of every fold: ``cv<k>/Opt-OneConfig-<SLO>`` (the fold's configuration for every test
    query) and ``cv<k>/Opt-PerQuery-<label>`` (the fold router's routes). ``oneconfig`` maps fold -> SLO ->
    {"cfg"} and needs the test query ids, so the driver passes the routes it built."""
    out = {}
    for f in folds:
        k = f["fold"]
        for n, routes in (oneconfig or {}).get(k, {}).items():
            if routes is not None:
                out[f"cv{k}/Opt-OneConfig-{n}"] = {
                    "kind": "oneconfig",
                    "slo": n,
                    "fold": k,
                    "routes": routes,
                }
        for label, routes in (f.get("test_routes") or {}).items():
            out[f"cv{k}/Opt-PerQuery-{label}"] = {
                "kind": "perquery",
                "fold": k,
                "weights": f["perquery"][label]["weights"],
                "routes": routes,
            }
    return out


def fold_tables(arms: dict[str, dict]) -> dict[int, dict]:
    """``test_arms`` output -> fold -> {"arms": {arm name without the cv<k>/ prefix: arm}} (``assemble`` input)."""
    out: dict[int, dict] = defaultdict(lambda: {"arms": {}})
    for name, arm in arms.items():
        out[arm["fold"]]["arms"][name.split("/", 1)[1]] = arm
    return dict(sorted(out.items()))


def pair_budget(
    cv_arms: dict[str, dict],
    deployed_arms: dict[str, dict],
    key: Callable[[dict], str],
    nomemory_key: str,
    completed: set[tuple[str, str]] | None = None,
    reuse_rate: float | None = None,
) -> dict:
    """Offline test-episode budget of the CV arms: distinct (config, query) pairs, those already in the deployed
    routing table (pair reuse), NoMemory pairs (reused baseline, never run), pairs already completed on disk, the
    NEW pairs to run and, given the observed quality-reuse hit rate, the expected fresh episodes."""

    def pairs(arms: dict[str, dict]) -> set[tuple[str, str]]:
        return {
            (key(r["config"]), q) for a in arms.values() for q, r in a["routes"].items()
        }

    cv, dep = pairs(cv_arms), pairs(deployed_arms)
    nm = {x for x in cv if x[0] == nomemory_key}
    new = cv - dep - nm - (completed or set())
    by_kind = {
        kind: len(
            {
                x
                for x in pairs({n: a for n, a in cv_arms.items() if a["kind"] == kind})
                if x in new
            }
        )
        for kind in sorted({a["kind"] for a in cv_arms.values()})
    }
    return {
        "cv_pairs": len(cv),
        "shared_with_deployed": len(cv & dep),
        "nomemory_pairs": len(nm),
        "already_completed": len((cv - dep - nm) & (completed or set())),
        "new_pairs": len(new),
        "new_pairs_by_kind": by_kind,
        "new_configs": len({k for k, _ in new}),
        "quality_reuse_rate": reuse_rate,
        "expected_fresh_episodes": None
        if reuse_rate is None
        else round(len(new) * (1 - reuse_rate)),
    }
