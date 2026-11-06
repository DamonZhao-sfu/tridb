"""Offline BRANE / LimeQO-inspired router training; never runs episodes.

python -m trimem.experiments.shared_router --snapshot <run>/router/training_snapshot.json \
    --model limeqo --out <new-output-directory> [--test-features <run>/router/features_test.json]

The pool router stages export this format for ALFWorld, ScienceWorld and tau2.
Existing/default routers and live profiling state are never changed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from trimem.optimizer.query_router import features, outer_cv, runtime, sparse

SNAPSHOT = "router-training-snapshot-v1"


def _identity(record):
    return record.get("store_key", record["query_id"])


def save_snapshot(
    path: Path,
    p: sparse.SparseProfile,
    records: list[dict],
    *,
    dataset: str,
    source: dict,
):
    """Called only with the existing drivers' TRAIN profile/records, before fitting."""
    ids = [r["query_id"] for r in records]
    if len(set(ids)) != len(ids) or set(ids) != set(p.queries):
        raise ValueError("Snapshot records must match the unique training query union")
    if len({_identity(r) for r in records}) != len(records):
        raise ValueError("Duplicate underlying training queries (store_key)")
    obj = {
        "protocol": SNAPSHOT,
        "split": "train",
        "dataset": dataset,
        "source": source,
        "profile": asdict(p),
        "records": records,
    }
    _save(path, obj)


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False))
    tmp.replace(path)


def load_snapshot(path: Path):
    blob = path.read_bytes()
    obj = json.loads(blob)
    if obj.get("protocol") != SNAPSHOT or obj.get("split") != "train":
        raise ValueError("Expected a versioned TRAIN-only router snapshot")
    p = sparse.SparseProfile(**obj["profile"])
    records = obj["records"]
    if len({r["query_id"] for r in records}) != len(records) or {
        r["query_id"] for r in records
    } != set(p.queries):
        raise ValueError("Snapshot feature IDs must match the training query union")
    if len({_identity(r) for r in records}) != len(records):
        raise ValueError("Duplicate underlying training queries (store_key)")
    return obj, p, {r["query_id"]: r for r in records}, hashlib.sha256(blob).hexdigest()


def train(
    snapshot: Path,
    out: Path,
    *,
    groups=("structured", "probe", "emb"),
    rank=5,
    reg=0.2,
    iterations=50,
    seed=0,
    cv_folds=5,
    weights=runtime.DEFAULT_WEIGHTS,
    test_features: Path | None = None,
    model_name="brane",
    trials=30,
    jobs=8,
):
    obj, p, records, digest = load_snapshot(snapshot)
    if model_name not in ("brane", "limeqo"):
        raise ValueError("model_name must be brane or limeqo")
    weights = runtime.normalize_weights(weights)
    if cv_folds != 0 and not 2 <= cv_folds <= len(p.queries):
        raise ValueError(
            "cv_folds must be 0 (skip) or between 2 and the number of queries"
        )
    tests = (
        []
        if test_features is None
        else json.loads(test_features.read_text())["records"]
    )

    if len({_identity(r) for r in tests}) != len(tests) or len(
        {r["query_id"] for r in tests}
    ) != len(tests):
        raise ValueError("Duplicate test queries")
    if {_identity(r) for r in tests} & {_identity(r) for r in records.values()} or {
        r["query_id"] for r in tests
    } & set(records):
        raise ValueError("Train/test feature IDs overlap")
    out.mkdir(parents=True, exist_ok=False)  # require a fresh experiment directory
    options = {"rank": rank, "reg": reg, "iterations": iterations}

    # Propagate seed to sparse.fit, not the shared_options namespace.
    def fit(profile, recs, *, cross_fit=True):
        profile, original_columns = sparse.without_empty_columns(profile)
        spec = features.fit_spec([recs[q] for q in profile.queries], groups)
        x = {q: runtime.transform(spec, recs[q]) for q in profile.queries}
        backend_opts = (
            {"shared_options": {**options, "cross_fit": cross_fit}}
            if model_name == "limeqo"
            else {"trials": trials, "n_jobs": jobs}
        )
        m, oof = sparse.fit(x, profile, backend=model_name, seed=seed, **backend_opts)
        m["candidate_columns"] = [original_columns[c] for c in m["candidate_columns"]]
        return spec, x, m, oof

    t0 = time.perf_counter()
    folds, evaluations = {}, []
    if cv_folds:
        folds = outer_cv.stratified_folds(p.queries, p.strata, cv_folds, seed)
        for k in range(cv_folds):
            print(
                f"{model_name} router: independent outer fold {k + 1}/{cv_folds}",
                flush=True,
            )
            train_ids = {q for q in p.queries if folds[q] != k}
            held = [q for q in p.queries if folds[q] == k]
            ptr, pho = outer_cv.restrict(p, train_ids), outer_cv.restrict(p, set(held))
            spec, _, model, _ = fit(ptr, records, cross_fit=False)
            choices = [
                runtime.choose_utility(
                    model, runtime.transform(spec, records[q]), weights
                )["index"]
                for q in held
            ]
            ev = outer_cv.heldout_eval(
                pho, model["candidate_columns"], held, np.array(choices), {}
            )
            ev["valid"] = not ev["incomplete_candidates"]
            if not ev["valid"]:
                # A wholly unmeasured selected column has no estimable contribution.
                # Never average its missing contribution as a zero outcome.
                for field in (
                    "quality",
                    "quality_se",
                    "tokens",
                    "tokens_se",
                    "latency_s",
                    "latency_se_s",
                ):
                    ev[field] = None
            evaluations.append(
                {
                    "fold": k,
                    "train_queries": len(train_ids),
                    "held_queries": len(held),
                    "candidate_columns": model["candidate_columns"],
                    "metrics": ev,
                }
            )
            _save(
                out / f"fold{k}.json",
                {"feature_spec": spec, "model": model, "evaluation": evaluations[-1]},
            )
    print(f"{model_name} router: final TRAIN fit + OOF replay", flush=True)
    spec, x, model, oof = fit(p, records)
    replay = sparse.weight_curve(p, model, x, oof, [weights])[0]
    summary = (
        {
            key: outer_cv.mean_std([e["metrics"][key] for e in evaluations])
            for key in ("quality", "tokens", "latency_s", "coverage")
        }
        if evaluations
        else None
    )
    shape = {
        "queries": len(p.queries),
        "configs": len(model["configs"]),
        "observed_pairs": sum(len(p.samples[c]) for c in model["candidate_columns"]),
        "possible_pairs": len(p.queries) * len(model["configs"]),
    }
    report = {
        "model": model_name,
        "protocol": model["predictor_protocol"],
        "source": obj["source"],
        "snapshot_sha256": digest,
        "training_shape": shape,
        "weights": list(weights),
        "cv_folds": folds,
        "outer_cv": summary,
        "outer_cv_validity": {
            "folds": len(evaluations),
            "valid_folds": sum(e["metrics"]["valid"] for e in evaluations),
            "partial": any(not e["metrics"]["valid"] for e in evaluations),
        },
        "per_fold": evaluations,
        "train_replay": replay,
        "fit_seconds": time.perf_counter() - t0,
        "limitations": [
            "Observed cells only; sparse held-out replay reports coverage.",
            "Feature/probe overhead is not included in these offline metric estimates.",
            "TRAIN OOF replay is a tuning diagnostic; outer CV refits the whole pipeline.",
            "Test features are routed only; no test outcome is used.",
        ],
    }
    artifact = sparse.artifact_weights(
        dataset=obj["dataset"],
        spec=spec,
        model=model,
        weights=weights,
        fallback=model["configs"][0],
        trained_on={"snapshot_sha256": digest, "source": obj["source"]},
        report=report,
    )
    runtime.save(out / "router.json", artifact)
    _save(out / "report.json", report)
    _save(out / "oof.json", oof)
    if tests:
        router = runtime.Router(artifact)
        _save(out / "test_routes.json", {r["query_id"]: router.route(r) for r in tests})
    print(
        json.dumps(
            {
                "artifact": str(out / "router.json"),
                "training_shape": shape,
                "outer_cv": summary,
                "fit_seconds": report["fit_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return artifact


def deployed_limeqo(
    snapshot: Path,
    out_dir: Path,
    *,
    groups: tuple[str, ...],
    bounds: dict[str, dict],
    objectives: dict[str, str],
    fallback: dict,
    trained_on: dict,
    seed: int = 0,
    options: dict | None = None,
) -> dict:
    """Deployed (full-TRAIN) LimeQO router of a pool driver's router stage (user 2026-10-21): fitted on the frozen
    ``training_snapshot.json`` (the same TRAIN profile + feature records as BRANE), the same candidates rule
    (``sparse.candidates`` inside ``shared.fit``), the same three-weight utility, weight grid and SLO rule on the
    OOF TRAIN replay. Saves ``router_<sha>.json`` (default weights; test weight vectors are applied with
    ``sparse.route_test_weights``) in ``out_dir`` and returns the ``router_report.json`` block."""
    from trimem.experiments.router_outer_cv import LIMEQO_OPTIONS

    options = dict(LIMEQO_OPTIONS if options is None else options)
    _, p, records, digest = load_snapshot(snapshot)
    recs = [records[q] for q in p.queries]
    spec = features.fit_spec(recs, groups)
    t0 = time.perf_counter()
    x = {q: runtime.transform(spec, records[q]) for q in p.queries}
    tr_ms = (time.perf_counter() - t0) * 1000 / len(p.queries)
    model, oof = sparse.fit(x, p, backend="limeqo", seed=seed, shared_options=options)
    ovh = {
        q: (records[q]["overhead"]["probe_ms"] if "probe" in groups else 0.0) + tr_ms
        for q in p.queries
    }
    grid = sparse.weight_grid()
    rows = sparse.weight_curve(p, model, x, oof, grid, overhead_ms=ovh)
    routers = {}
    for n, obj in objectives.items():
        pick = sparse.choose_weights(rows, bounds[n], obj)
        routers[n] = (
            None
            if pick is None
            else {
                "weights": list(sparse.weights_of(pick)),
                "replay_q": pick["q"],
                "replay_q_se": pick["quality_se"],
                "replay_lat_s": pick["lat"],
                "replay_tok": pick["tok"],
                "direct_q": pick["direct_quality"],
                "slack": sparse.slack(pick, bounds[n]),
                "n_feasible": sum(sparse.feasible(r, bounds[n]) for r in rows),
            }
        )
    art = sparse.artifact_weights(
        dataset=trained_on.get("dataset", ""),
        spec=spec,
        model=model,
        weights=sparse.DEFAULT_WEIGHTS,
        fallback=fallback,
        trained_on={**trained_on, "snapshot_sha256": digest, "backend": "limeqo"},
        report={"backend": "limeqo", "options": options},
    )
    sha = runtime.artifact_sha(art)
    path = out_dir / f"router_{sha}.json"
    if not path.exists():
        runtime.save(path, art)
    return {
        "backend": "limeqo",
        "protocol": model["predictor_protocol"],
        "options": options,
        "seed": seed,
        "snapshot": str(snapshot),
        "snapshot_sha256": digest,
        "training_shape": model["training_shape"],
        "candidates": model["configs"],
        "cost_norm": model["cost_norm"],
        "transform_ms": tr_ms,
        "routers": routers,
        "slice_router": {
            "path": str(path),
            "sha": sha,
            "default_weights": list(sparse.DEFAULT_WEIGHTS),
        },
        "weight_grid": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument(
        "--model",
        "--router-model",
        choices=("brane", "limeqo"),
        default="brane",
        help="brane (default): independent AutoML; limeqo: inductive shared ALS",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-features", type=Path)
    ap.add_argument("--groups", default="structured,probe,emb")
    ap.add_argument("--rank", type=int, default=5)
    ap.add_argument("--reg", type=float, default=0.2)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=30, help="BRANE AutoML trials")
    ap.add_argument("--jobs", type=int, default=8, help="BRANE worker processes")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=runtime.DEFAULT_WEIGHTS,
        metavar=("Q", "L", "T"),
    )
    a = ap.parse_args()
    train(
        a.snapshot,
        a.out,
        groups=tuple(a.groups.split(",")),
        rank=a.rank,
        reg=a.reg,
        iterations=a.iterations,
        seed=a.seed,
        cv_folds=a.cv_folds,
        weights=a.weights,
        test_features=a.test_features,
        model_name=a.model,
        trials=a.trials,
        jobs=a.jobs,
    )


if __name__ == "__main__":
    main()
