"""TRACE-PR per-query router (Opt-PerQuery style) on the measured TEST200 matrix.

Candidates: the configurations evaluated on TEST (TRAIN Pareto set), so every routed choice
has a measured TEST outcome; no outcome is simulated. Training data: the same candidates on
the 100 TRAIN tasks (dense: every candidate ran every TRAIN task).

Predictor per objective (quality, tokens, latency): kNN over the normalized task embedding,
shrunk to the candidate's global TRAIN mean (est = (sum_kNN y + m * mean) / (k + m)).
(k, m) per objective are chosen by TRAIN leave-one-out MSE. TEST is read only after routing.

Routing (as the historical Opt-PerQuery router, rule utility3-minmax with scale
zscore-log-v1, trimem.optimizer.query_router.runtime.utility_norm):
U = wQ*q/s_Q - wL*ln(lat)/s_L - wT*ln(tok)/s_T, where s_* = median over TRAIN tasks of the
across-candidate std of the honest (leave-one-out) predictions, floored at the median
per-candidate SE of the mean. Costs are predicted in ln space. argmax U per task. Weight settings:
default (0.6,0.2,0.2), quality (0.8,0.1,0.1), tokens (0.1,0.1,0.8), latency (0.1,0.8,0.1),
plus the historical Opt-PerQuery TEST sweep (curves-step0.2-v1): a Q-T curve (wL = 0) and a
Q-L curve (wT = 0) with step 0.2.
Opt-OneConfig at the same weights = argmax U of the global TRAIN means (one config for all).

Output: <run>/holdout/router_curve.json and printed table (quality, success, tokens,
latency, agent steps, paired quality delta vs OneConfig at the same weights).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.trace_pr_perquery_eval import BOOT, knn_weights, matrix, outcomes, task_features

ROOT = Path(__file__).resolve().parents[1]
KS = (5, 10, 20, 40)
SHRINK = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
NAMED = {
    "default": (0.6, 0.2, 0.2),
    "quality": (0.8, 0.1, 0.1),
    "tokens": (0.1, 0.1, 0.8),
    "latency": (0.1, 0.8, 0.1),
}


def predict(w: np.ndarray, y: np.ndarray, gmean: np.ndarray, m: float) -> np.ndarray:
    n = w.sum()
    return (w @ y + m * gmean) / (n + m)


def choose_knn(y: np.ndarray, sims: np.ndarray) -> dict:
    """(k, m) minimizing leave-one-out MSE of the per-task, per-candidate prediction."""
    n = len(y)
    best = None
    for k in KS:
        for m in SHRINK:
            err = 0.0
            for i in range(n):
                gm = (y.sum(axis=0) - y[i]) / (n - 1)
                err += float(((predict(knn_weights(sims[i], k, i), y, gm, m) - y[i]) ** 2).mean())
            if best is None or err < best[0]:
                best = (err / n, k, m)
    return {"loo_mse": best[0], "k": best[1], "m": best[2]}


def utility(pred: dict, scale: dict, w: tuple) -> np.ndarray:
    """zscore-log-v1: pred holds quality and ln costs; w = (wQ, wL, wT)."""
    return (w[0] * pred["score"] / scale["score"] - w[1] * pred["latency"] / scale["latency"]
            - w[2] * pred["tokens"] / scale["tokens"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", default="sciworld", choices=("alfworld", "sciworld", "webshop"))
    ap.add_argument("--complete-only", action="store_true",
                    help="use only candidates with all 200 TEST tasks done (interim analysis)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-tasks", type=int, default=200,
                    help="evaluate on the TEST tasks every candidate has done (>= this many)")
    args = ap.parse_args()
    run = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    import sqlite3

    hold = sqlite3.connect(f"file:{run / 'holdout/ledger.sqlite'}?mode=ro", uri=True)
    keys = [k for (k,) in hold.execute("SELECT key FROM configs ORDER BY key")]
    if args.complete_only:
        done = dict(hold.execute("SELECT config, count(*) FROM episodes WHERE state='done' GROUP BY config"))
        keys = [k for k in keys if done.get(k, 0) == 200]
    train_rows = outcomes(run / "ledger.sqlite", keys)
    test_rows = outcomes(run / "holdout/ledger.sqlite", keys)
    items = json.loads((run / "train_panel.json").read_text())["items"]
    test_ids = json.loads((run / "test_panel.json").read_text())["tasks"]
    tr = [i["item_id"] for i in items]
    if any(len(train_rows.get(t, {})) != len(keys) for t in tr):
        raise ValueError("Every candidate must have every TRAIN task")
    te = [t for t in test_ids if len(test_rows.get(t, {})) == len(keys)]
    if len(te) < args.min_tasks:
        raise ValueError(f"TEST matrix incomplete: {len(te)}/200 tasks have all candidates")
    feats = task_features(args.dataset, items, test_ids)
    vec_tr = np.stack([feats[t]["vec"] for t in tr])
    sims_tr = vec_tr @ vec_tr.T
    fields = ("score", "tokens", "latency")
    Ytr = {f: matrix(train_rows, tr, keys, f) for f in fields}
    for f in ("tokens", "latency"):  # costs routed in ln space (zscore-log-v1)
        Ytr[f] = np.log(np.maximum(Ytr[f], 1e-6))
    gmean = {f: Ytr[f].mean(axis=0) for f in fields}
    knn = {f: choose_knn(Ytr[f], sims_tr) for f in fields}
    # Honest (leave-one-out) TRAIN predictions -> per-metric scale s_* (runtime.utility_norm).
    from trimem.optimizer.query_router.runtime import utility_norm

    n_tr = len(tr)
    honest = np.stack([
        np.stack([
            predict(knn_weights(sims_tr[i], knn[f]["k"], i), Ytr[f],
                    (Ytr[f].sum(axis=0) - Ytr[f][i]) / (n_tr - 1), knn[f]["m"])
            for f in fields], axis=1)
        for i in range(n_tr)])  # (tasks, candidates, [q, ln tok, ln lat])
    cmeans = np.stack([gmean[f] for f in fields], axis=1)
    cse = np.stack([Ytr[f].std(axis=0, ddof=1) / n_tr**0.5 for f in fields], axis=1)
    norm = utility_norm(honest, cmeans, cse)  # order: quality, ln tokens, ln latency
    scale = {"score": norm["quality"]["s"], "latency": norm["latency_ms"]["s"], "tokens": norm["tokens"]["s"]}
    # Per-task predictions on TEST (TRAIN only).
    sims_te = np.stack([vec_tr @ feats[t]["vec"] for t in te])
    pred = {
        f: np.stack([predict(knn_weights(sims_te[j], knn[f]["k"]), Ytr[f], gmean[f], knn[f]["m"]) for j in range(len(te))])
        for f in fields
    }
    # Historical Opt-PerQuery TEST sweep (weight set curves-step0.2-v1, 2026-10-23): a Q-T curve
    # (wL = 0) and a Q-L curve (wT = 0), step 0.2 -> 11 distinct weights, plus the 4 named weights.
    weights = dict(NAMED)
    for q in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        weights.setdefault(f"QT_w{q:g}-0-{1 - q:g}", (q, 0.0, round(1 - q, 1)))
        weights.setdefault(f"QL_w{q:g}-{1 - q:g}-0", (q, round(1 - q, 1), 0.0))
    # ---- read TEST now
    Y = {f: matrix(test_rows, te, keys, f) for f in (*fields, "steps")}
    rng = np.random.default_rng(20261104)
    idx = np.arange(len(te))
    succ_thr = 100 if args.dataset == "sciworld" else 1

    def measure(cols):
        r = {f: float(Y[f][idx, cols].mean()) for f in (*fields, "steps")}
        r["success"] = float((Y["score"][idx, cols] >= succ_thr).mean())
        r["score_se"] = float(Y["score"][idx, cols].std(ddof=1) / len(idx) ** 0.5)
        r["configs_used"] = {keys[c][:8]: int((cols == c).sum()) for c in np.unique(cols)}
        return r

    out = {"run": str(run), "candidates": keys, "test_tasks": len(te), "knn": knn,
           "scaling": {"rule": "utility3 zscore-log-v1", "s": scale, "norm": norm}, "settings": {}}
    for name, w in weights.items():
        router = utility(pred, scale, w).argmax(axis=1)
        one = int(utility({f: gmean[f][None, :] for f in fields}, scale, w).argmax())
        rcols, ocols = router, np.full(len(te), one)
        d = Y["score"][idx, rcols] - Y["score"][idx, ocols]
        boots = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(axis=1)
        out["settings"][name] = {
            "weights": {"wQ": w[0], "wL": w[1], "wT": w[2]},
            "router": measure(rcols),
            "oneconfig": {**measure(ocols), "key": keys[one]},
            "delta_quality_router_minus_one": [float(d.mean()), float(np.quantile(boots, 0.025)),
                                               float(np.quantile(boots, 0.975))],
        }
    (args.out or run / "holdout/router_curve.json").write_text(json.dumps(out, indent=2))
    print("kNN (k, m) by TRAIN LOO:", {f: (v["k"], v["m"]) for f, v in knn.items()})
    print(f"{'setting':16s} {'router q':>8s} {'succ':>5s} {'tok':>7s} {'lat':>6s} {'steps':>5s} | "
          f"{'one q':>6s} {'tok':>7s} {'lat':>6s} | Δq [95% CI]")
    for name, s in out["settings"].items():
        r, o, (dm, dl, dh) = s["router"], s["oneconfig"], s["delta_quality_router_minus_one"]
        print(f"{name:16s} {r['score']:8.2f} {r['success']:5.3f} {r['tokens'] / 1e3:6.1f}k {r['latency']:6.1f} "
              f"{r['steps']:5.1f} | {o['score']:6.2f} {o['tokens'] / 1e3:6.1f}k {o['latency']:6.1f} | "
              f"{dm:+.2f} [{dl:+.2f},{dh:+.2f}]")


if __name__ == "__main__":
    main()
