"""TRACE PerQuery router with the historical Opt-PerQuery classifier predictors (user 2026-11-05).

User correction 2026-11-05: candidates = ALL complete TRAIN configurations of the run (not only the TEST-measured
Pareto set); predictor = the historical trained classifiers, not kNN.

* Candidates: every configuration with a complete, never-pruned TRAIN panel (dense TRAIN matrix). No BRANE fuzzy
  Pareto pruning (user: all TRAIN configurations).
* Features: the historical router features ``structured + probe + emb`` (``features.raw_record`` with the VGR seed
  probe, ``features.fit_spec(..., emb_dims=4)``), fitted on TRAIN queries only.
* Predictors: per configuration and per metric, ``predictors.fit`` = BRANE model selection over six tabular
  families (logreg, tree, forest, gboost, xgboost, lightgbm; Optuna 30 trials, 3-fold inner CV; quality = binary
  success classifier for ALFWorld, regressors for raw tokens / latency) with 5 outer folds for honest OOF
  predictions. The best family per metric is kept (``tabular.select_and_fit``).
* Routing: the deployed three-weight rule ``runtime.utility`` with the frozen ``zscore-log-v1`` scale
  (``runtime.utility_norm`` on the honest OOF predictions). Weights = 4 named + the historical TEST sweep
  ``curves-step0.2-v1`` (Q-T: w_L = 0, Q-L: w_T = 0, step 0.2).
* Opt-OneConfig at the same weights = argmax of the same utility on the TRAIN means.

Stages:
  fit     : features (cached) + per-config predictors -> <run>/holdout/router_cls/model.json
  route   : TEST routes per weight -> routes.json + pairs.json (measured vs new (task, config) pairs).
            TEST outcomes are NOT read here except to mark which pairs already exist.
  report  : TEST metrics of the routed policies from measured pairs (refuses if a routed pair is missing).

  .venv-optimizer/bin/python -m tools.trace_perquery_classifier_router {fit|route|report} RUN --dataset alfworld
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("structured", "probe", "emb")
EMB_DIMS = 4
PROTOCOL = "trace-perquery-classifier-v1"
NAMED = {
    "default": (0.6, 0.2, 0.2),
    "quality": (0.8, 0.1, 0.1),
    "tokens": (0.1, 0.1, 0.8),
    "latency": (0.1, 0.8, 0.1),
}


def weight_set() -> dict[str, tuple[float, float, float]]:
    w = dict(NAMED)
    for q in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        w.setdefault(f"QT_w{q:g}-0-{1 - q:g}", (q, 0.0, round(1 - q, 1)))
        w.setdefault(f"QL_w{q:g}-{1 - q:g}-0", (q, round(1 - q, 1), 0.0))
    return w


def outcomes(ledger: Path, keys: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """config key -> task -> {score, tokens, latency_ms, steps} for done episodes."""
    db = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    out: dict[str, dict[str, dict]] = {}
    for cfg, task, rec in db.execute("SELECT config, task, receipt FROM episodes WHERE state='done'"):
        if keys is not None and cfg not in keys:
            continue
        r = json.loads(rec)
        out.setdefault(cfg, {})[task] = {
            "score": float(r["raw_score"]),
            "tokens": float(r["total_tokens"]),
            "latency_ms": float(r["latency_seconds"]) * 1000.0,
            "steps": float(r.get("agent_steps") or 0.0),
        }
    db.close()
    return out


def complete_keys(run: Path, n_tasks: int) -> list[str]:
    db = sqlite3.connect(f"file:{run / 'ledger.sqlite'}?mode=ro", uri=True)
    keys = []
    for (k,) in db.execute("SELECT key FROM configs ORDER BY key"):
        s = dict(db.execute("SELECT state, count(*) FROM episodes WHERE config=? GROUP BY state", (k,)))
        if s.get("done", 0) == n_tasks and len(s) == 1:
            keys.append(k)
    db.close()
    return keys


def feature_records(dataset: str, train_items: list[dict], test_ids: list[str], cache: Path) -> dict[str, dict]:
    """Raw router records (structured + probe + emb) of TRAIN items and TEST tasks; cached."""
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments.agentenv_task_adapter import dsn_for
    from trimem.experiments.query_router_agentenv import TASK_SQL
    from trimem.optimizer.query_router import features

    have = {r["query_id"]: r for r in json.loads(cache.read_text())["records"]} if cache.exists() else {}
    todo = [("train", i) for i in train_items if i["item_id"] not in have]
    todo += [("test", t) for t in test_ids if t not in have]
    if todo:
        dsn = dsn_for(dataset) + " options='-c default_transaction_read_only=on'"
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            for n, (split, it) in enumerate(todo):
                if split == "train":
                    rows = conn.execute(TASK_SQL + "key = %s AND split = 'train' AND NOT held_out",
                                        (it["task_key"],)).fetchall()
                    qid = it["item_id"]
                else:
                    rows = conn.execute(TASK_SQL + "agenteval_item = %s", (it,)).fetchall()
                    qid = it
                    if len(rows) == 1 and not rows[0]["held_out"]:
                        raise ValueError(f"TEST task {qid} is not held out")
                if len(rows) != 1:
                    raise ValueError(f"{split} task {qid}: {len(rows)} store rows")
                t = dict(rows[0])
                vec = t.pop("emb")
                have[qid] = features.raw_record(qid, t, vec, use_probe=True)
                if n % 100 == 99:
                    cache.write_text(json.dumps({"records": list(have.values())}))
        cache.write_text(json.dumps({"records": list(have.values())}))
    return have


def _fit_one(args):
    from trimem.optimizer.query_router import predictors

    key, X, q, tok, lat, binary, seed = args
    m = predictors.fit(X, q, tok, lat, binary=binary, seed=seed)
    return key, m


def cost_pred(m: dict, X: np.ndarray) -> np.ndarray:
    from trimem.optimizer.query_router import predictors

    return np.array([predictors.cost_prediction(m["predictor"], x, m["fallback"]) for x in X])


def qual_pred(m: dict, X: np.ndarray) -> np.ndarray:
    from trimem.optimizer.query_router import predictors

    return predictors._predict(m, X, False, None)


def stage_fit(run: Path, dataset: str, jobs: int, seed: int, candidates: str, out: Path) -> None:
    from trimem.optimizer.query_router import features, runtime

    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / "manifest.json").read_text())
    for k, v in manifest.get("fixed", {}).get("env", {}).items() if isinstance(manifest.get("fixed"), dict) else []:
        if k.startswith("TRIMEM_") and k.endswith("MEMORY_DB"):
            os.environ.setdefault(k, v)
    items = json.loads((run / "train_panel.json").read_text())["items"]
    test_ids = json.loads((run / "test_panel.json").read_text())["tasks"]
    tr = [i["item_id"] for i in items]
    keys = complete_keys(run, len(tr))
    if candidates == "pareto":  # user 2026-11-05: TEST-measured TRAIN sample-mean Pareto configurations
        # The TEST-measured set = the configurations of the holdout ledger (selection.json candidate_keys is
        # the pre-Pareto eligible set, not the Pareto set).
        hdb = sqlite3.connect(f"file:{run / 'holdout/ledger.sqlite'}?mode=ro", uri=True)
        pk = [k for (k,) in hdb.execute("SELECT key FROM configs")]
        hdb.close()
        if not pk or any(k not in keys for k in pk):
            raise ValueError("Pareto candidates must be complete TRAIN configurations")
        keys = sorted(pk)
    rows = outcomes(run / "ledger.sqlite", set(keys))
    if any(set(rows[k]) != set(tr) for k in keys):
        raise ValueError("every candidate must have every TRAIN task")
    cache = out / "features.json"
    shared = run / "holdout/router_cls/features.json"
    if not cache.exists() and shared.exists():
        cache.write_text(shared.read_text())
    recs = feature_records(dataset, items, test_ids, cache)
    spec = features.fit_spec([recs[t] for t in tr], GROUPS, emb_dims=EMB_DIMS)
    X_tr = np.stack([runtime.transform(spec, recs[t]) for t in tr])
    binary = dataset == "alfworld"
    scale_q = 1.0 if binary else 100.0
    args = []
    for k in keys:
        # Quality target as the historical router (query_router.matrix): ScienceWorld max(score, 0) / 100.
        q = np.array([max(rows[k][t]["score"], 0.0) / scale_q for t in tr])
        tok = np.array([rows[k][t]["tokens"] for t in tr])
        lat = np.array([rows[k][t]["latency_ms"] for t in tr])
        args.append((k, X_tr, q, tok, lat, binary, seed))
    models = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for n, (k, m) in enumerate(ex.map(_fit_one, args), 1):
            models[k] = m
            sel = {o: s.get("family") for o, s in m["selection"].items()}
            print(f"[{n}/{len(keys)}] {k[:8]} {sel}", flush=True)
    # Candidate order = TRAIN quality desc (ties in argmax -> best TRAIN quality), as the deployed router.
    tq = {k: np.mean([rows[k][t]["score"] for t in tr]) for k in keys}
    keys = sorted(keys, key=lambda k: (-tq[k], k))
    honest = np.stack([np.asarray(models[k]["oof"]) for k in keys], axis=1)  # (n_tr, C, 3)
    obs = np.stack([np.stack([
        np.array([max(rows[k][t]["score"], 0.0) / scale_q for t in tr]),
        np.log(np.array([rows[k][t]["tokens"] for t in tr])),
        np.log(np.array([rows[k][t]["latency_ms"] for t in tr])),
    ], axis=1) for k in keys], axis=1)  # (n_tr, C, 3)
    cmeans = obs.mean(axis=0)
    cse = obs.std(axis=0, ddof=1) / len(tr) ** 0.5
    norm = runtime.utility_norm(honest, cmeans, cse)
    (out / "model.json").write_text(json.dumps({
        "protocol": PROTOCOL, "dataset": dataset, "candidate_set": candidates, "groups": GROUPS, "emb_dims": EMB_DIMS, "seed": seed,
        "candidates": keys, "train_tasks": tr, "spec": spec, "norm": norm, "quality_scale": scale_q,
        "models": {k: {x: models[k][x] for x in ("quality", "ln_tokens", "ln_latency", "selection")}
                   for k in keys},
        "train_oof": honest.tolist(), "train_obs": obs.tolist(),
    }, default=float))
    print("scale:", {m: round(norm[m]["s"], 4) for m in ("quality", "tokens", "latency_ms")})


def _route(q, tok, lat, w, norm):
    from trimem.optimizer.query_router import runtime

    return np.argmax(runtime.utility(q, tok, lat, w, norm), axis=-1)


def stage_route(run: Path, out: Path) -> None:
    from trimem.optimizer.query_router import runtime

    M = json.loads((out / "model.json").read_text())
    keys, norm, spec = M["candidates"], M["norm"], M["spec"]
    recs = {r["query_id"]: r for r in json.loads((out / "features.json").read_text())["records"]}
    te = json.loads((run / "test_panel.json").read_text())["tasks"]
    X_te = np.stack([runtime.transform(spec, recs[t]) for t in te])
    pq = np.stack([qual_pred(M["models"][k]["quality"], X_te) for k in keys], axis=1)
    pt = np.stack([cost_pred(M["models"][k]["ln_tokens"], X_te) for k in keys], axis=1)
    pl = np.stack([cost_pred(M["models"][k]["ln_latency"], X_te) for k in keys], axis=1)
    oof, obs = np.asarray(M["train_oof"]), np.asarray(M["train_obs"])
    means_q, means_t, means_l = obs[:, :, 0].mean(0), np.exp(obs[:, :, 1]).mean(0), np.exp(obs[:, :, 2]).mean(0)
    measured = outcomes(run / "holdout/ledger.sqlite")
    routes, pairs = {}, set()
    for name, w in weight_set().items():
        r = _route(pq, pt, pl, w, norm)
        one = int(_route(means_q, means_t, means_l, w, norm))
        # TRAIN replay (honest OOF routing, measured TRAIN outcome of the routed config).
        rr = _route(oof[:, :, 0], np.exp(oof[:, :, 1]), np.exp(oof[:, :, 2]), w, norm)
        idx = np.arange(len(rr))
        replay = {"quality": float(obs[idx, rr, 0].mean()), "tokens": float(np.exp(obs[idx, rr, 1]).mean()),
                  "latency_ms": float(np.exp(obs[idx, rr, 2]).mean())}
        routes[name] = {
            "weights": w, "oneconfig": keys[one],
            "routes": {t: keys[c] for t, c in zip(te, r, strict=True)},
            "predicted": {"quality": float(pq[np.arange(len(te)), r].mean()),
                          "tokens": float(pt[np.arange(len(te)), r].mean()),
                          "latency_ms": float(pl[np.arange(len(te)), r].mean())},
            "train_replay": replay,
            "configs_used": {keys[c][:8]: int((r == c).sum()) for c in np.unique(r)},
        }
        pairs |= {(t, keys[c]) for t, c in zip(te, r, strict=True)}
        pairs |= {(t, keys[one]) for t in te}
    have = [p for p in pairs if p[0] in measured.get(p[1], {})]
    new = sorted(p for p in pairs if p[0] not in measured.get(p[1], {}))
    (out / "routes.json").write_text(json.dumps({"protocol": PROTOCOL, "candidates": keys, "settings": routes},
                                                indent=1))
    by_cfg: dict[str, list[str]] = {}
    for t, k in new:
        by_cfg.setdefault(k, []).append(t)
    (out / "pairs.json").write_text(json.dumps({"measured": len(have), "new": len(new), "new_by_config": by_cfg},
                                               indent=1))
    print(f"{len(keys)} candidates; pairs: {len(pairs)} total, {len(have)} measured, {len(new)} new "
          f"over {len(by_cfg)} configs")
    for name, s in routes.items():
        print(f"{name:14s} one {s['oneconfig'][:8]} used {s['configs_used']} "
              f"pred q {s['predicted']['quality']:.3f} replay q {s['train_replay']['quality']:.3f}")


def stage_report(run: Path, dataset: str, out: Path) -> None:
    R = json.loads((out / "routes.json").read_text())
    measured = outcomes(run / "holdout/ledger.sqlite")
    pair_dir = out / "pairs_ledger.sqlite"
    if pair_dir.exists():
        for k, d in outcomes(pair_dir).items():
            measured.setdefault(k, {}).update(d)
    succ_thr = 100 if dataset == "sciworld" else 1
    rep = {}
    for name, s in R["settings"].items():
        row = {}
        for arm, assign in (("router", s["routes"]), ("oneconfig", {t: s["oneconfig"] for t in s["routes"]})):
            vals = [measured.get(k, {}).get(t) for t, k in assign.items()]
            if any(v is None for v in vals):
                row[arm] = {"missing": sum(v is None for v in vals)}
                continue
            sc = np.array([v["score"] for v in vals])
            row[arm + "_scores"] = sc
            row[arm] = {"score": float(sc.mean()), "success": float((sc >= succ_thr).mean()),
                        "score_se": float(sc.std(ddof=1) / len(sc) ** 0.5),
                        "configs_used": dict(Counter(k[:8] for k in assign.values())),
                        "tokens": float(np.mean([v["tokens"] for v in vals])),
                        "latency_s": float(np.mean([v["latency_ms"] for v in vals])) / 1000,
                        "steps": float(np.mean([v["steps"] for v in vals]))}
        if "router_scores" in row and "oneconfig_scores" in row:
            d = row.pop("router_scores") - row.pop("oneconfig_scores")
            rng = np.random.default_rng(20261105)
            boots = d[rng.integers(0, len(d), (10_000, len(d)))].mean(axis=1)
            row["delta_quality_router_minus_one"] = [float(d.mean()), float(np.quantile(boots, 0.025)),
                                                     float(np.quantile(boots, 0.975))]
        row.pop("router_scores", None)
        row.pop("oneconfig_scores", None)
        row["oneconfig"]["key"] = s["oneconfig"] if "score" in row.get("oneconfig", {}) else None
        rep[name] = {"weights": {"wQ": s["weights"][0], "wL": s["weights"][1], "wT": s["weights"][2]}, **row}
        print(name, json.dumps(row))
    M = json.loads((out / "model.json").read_text())
    (out / "report.json").write_text(json.dumps({
        "protocol": PROTOCOL, "candidate_set": M.get("candidate_set"), "candidates": R["candidates"],
        "test_tasks": len(next(iter(R["settings"].values()))["routes"]),
        "predictors": {k[:8]: {o: v.get("family") for o, v in M["models"][k]["selection"].items()}
                       for k in R["candidates"]},
        "groups": M["groups"], "settings": rep}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("fit", "route", "report"))
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", required=True, choices=("alfworld", "sciworld", "webshop"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidates", choices=("all", "pareto"), default="all")
    a = ap.parse_args()
    run = a.run_dir if a.run_dir.is_absolute() else ROOT / a.run_dir
    out = run / ("holdout/router_cls" if a.candidates == "all" else "holdout/router_cls_pareto")
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "fit":
        stage_fit(run, a.dataset, a.jobs, a.seed, a.candidates, out)
    elif a.stage == "route":
        stage_route(run, out)
    else:
        stage_report(run, a.dataset, out)


if __name__ == "__main__":
    main()
