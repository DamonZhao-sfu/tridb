"""Offline per-query routing check on a TRACE-PR TEST200 run (no new episodes).

The TEST step evaluates every TRAIN-Pareto config on every TEST task, so each TEST task has a
full row of outcomes. This tool compares, on the same TEST rows:

* OneConfig        : the config with the best TRAIN mean score (chosen before TEST).
* Hindsight-1      : the config with the best TEST mean (an upper bound for any single config).
* Type router      : per task type, a shrunk TRAIN mean per config; argmax.
* kNN router       : k nearest TRAIN tasks by task-vector cosine, shrunk mean; argmax.
* Oracle           : per TEST task, the best of the candidate configs (routing upper bound).
* Null oracle      : the oracle gain expected without task x config interaction (residual
                     permutation within config), i.e. the part of the oracle gap that is noise.

Router hyperparameters (k, shrinkage m) are chosen by leave-one-out on TRAIN only. TEST labels
are read only after every choice is fixed. Reads ledgers and the read-only memory DB clone.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHRINK = (0.0, 1.0, 2.0, 5.0, 10.0, 1e9)
KS = (5, 10, 20, 40)
BOOT = 10_000
NULL_REPS = 500
# Sensitivity routers with fixed, non-trivial settings (fixed before TEST; not LOO-chosen),
# reported because LOO may shrink both routers back to OneConfig.
SENSITIVITY = {"type_fixed": {"m": 5.0}, "knn_fixed": {"k": 20, "m": 10.0}}


def outcomes(ledger: Path, keys: list[str]) -> dict[str, dict[str, dict]]:
    """task -> config key -> {score, tokens, latency, steps} for done episodes."""
    db = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    out: dict[str, dict[str, dict]] = {}
    for key in keys:
        for task, receipt in db.execute(
            "SELECT task, receipt FROM episodes WHERE config = ? AND state = 'done'", (key,)
        ):
            r = json.loads(receipt)
            out.setdefault(task, {})[key] = {
                "score": float(r["raw_score"]),
                "tokens": float(r["total_tokens"]),
                "latency": float(r["latency_seconds"]),
                "steps": float(r["agent_steps"]),
            }
    return out


def task_features(dataset: str, train_items: list[dict], test_ids: list[str]) -> dict[str, dict]:
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments.agentenv_task_adapter import dsn_for
    from trimem.experiments.query_router_agentenv import TASK_SQL

    feats = {}
    sql_test = TASK_SQL.replace("emb::text AS emb", "agenteval_item, emb::text AS emb")
    dsn = dsn_for(dataset) + " options='-c default_transaction_read_only=on'"
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        for item in train_items:
            rows = conn.execute(
                TASK_SQL + "key = %s AND split = 'train' AND NOT held_out", (item["task_key"],)
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(f"TRAIN task {item['item_id']} is not unique in the store")
            feats[item["item_id"]] = rows[0]
        for tid in test_ids:
            rows = conn.execute(sql_test + "agenteval_item = %s", (tid,)).fetchall()
            if len(rows) != 1 or not rows[0]["held_out"]:
                raise ValueError(f"TEST task {tid} is not a unique held-out row")
            feats[tid] = rows[0]
    out = {}
    for tid, row in feats.items():
        v = np.asarray(json.loads(row["emb"]), dtype=float)
        out[tid] = {"type": row.get("task_type") or "", "vec": v / np.linalg.norm(v)}
    return out


def matrix(rows: dict[str, dict[str, dict]], tasks: list[str], keys: list[str], field: str) -> np.ndarray:
    return np.array([[rows[t][k][field] for k in keys] for t in tasks])


def route(weights: np.ndarray, y_train: np.ndarray, global_mean: np.ndarray, m: float) -> int:
    """Shrunk weighted mean per config; ties go to the better global mean."""
    n = weights.sum()
    est = (weights @ y_train + m * global_mean) / (n + m) if n + m > 0 else global_mean
    best = np.flatnonzero(np.isclose(est, est.max()))
    return int(best[np.argmax(global_mean[best])])


def type_weights(types_train: list[str], t: str, exclude: int | None = None) -> np.ndarray:
    w = np.array([1.0 if x == t else 0.0 for x in types_train])
    if exclude is not None:
        w[exclude] = 0.0
    return w


def knn_weights(sims: np.ndarray, k: int, exclude: int | None = None) -> np.ndarray:
    s = sims.copy()
    if exclude is not None:
        s[exclude] = -np.inf
    w = np.zeros_like(s)
    w[np.argsort(-s)[:k]] = 1.0
    return w


def loo_choose(y_tr, types_tr, vec_tr, global_mean):
    """Pick (policy params) by leave-one-out realized TRAIN score; TEST is never read."""
    n = len(y_tr)
    sims = vec_tr @ vec_tr.T
    best = {}
    for m in SHRINK:
        picks = []
        for i in range(n):
            gm = (global_mean * n - y_tr[i]) / (n - 1)
            picks.append(y_tr[i, route(type_weights(types_tr, types_tr[i], i), y_tr, gm, m)])
        val = float(np.mean(picks))
        if "type" not in best or val > best["type"][0]:
            best["type"] = (val, {"m": m})
    for k in KS:
        for m in SHRINK:
            picks = []
            for i in range(n):
                gm = (global_mean * n - y_tr[i]) / (n - 1)
                picks.append(y_tr[i, route(knn_weights(sims[i], k, i), y_tr, gm, m)])
            val = float(np.mean(picks))
            if "knn" not in best or val > best["knn"][0]:
                best["knn"] = (val, {"k": k, "m": m})
    return best


def paired_ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    d = a - b
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def null_oracle_gain(y: np.ndarray, rng) -> tuple[float, float]:
    """Oracle minus best single config when residuals carry no task x config interaction."""
    task_eff = y.mean(axis=1, keepdims=True)
    cfg_eff = y.mean(axis=0, keepdims=True) - y.mean()
    resid = y - task_eff - cfg_eff
    gains = []
    for _ in range(NULL_REPS):
        r = np.column_stack([rng.permutation(resid[:, j]) for j in range(y.shape[1])])
        ys = np.clip(task_eff + cfg_eff + r, y.min(), y.max())  # observed score range (ALF 0-1, Sci may be negative)
        gains.append(ys.max(axis=1).mean() - ys.mean(axis=0).max())
    return float(np.mean(gains)), float(np.quantile(gains, 0.95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", default="sciworld", choices=("alfworld", "sciworld", "webshop"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    hold = sqlite3.connect(f"file:{run / 'holdout/ledger.sqlite'}?mode=ro", uri=True)
    keys = [k for (k,) in hold.execute("SELECT key FROM configs ORDER BY key")]
    train_rows = outcomes(run / "ledger.sqlite", keys)
    test_rows = outcomes(run / "holdout/ledger.sqlite", keys)
    train_items = json.loads((run / "train_panel.json").read_text())["items"]
    test_ids = json.loads((run / "test_panel.json").read_text())["tasks"]
    tr_tasks = [i["item_id"] for i in train_items if len(train_rows.get(i["item_id"], {})) == len(keys)]
    te_tasks = [t for t in test_ids if len(test_rows.get(t, {})) == len(keys)]
    if len(tr_tasks) != len(train_items):
        raise ValueError("Some TRAIN tasks miss a candidate config")
    feats = task_features(args.dataset, train_items, test_ids)

    y_tr = matrix(train_rows, tr_tasks, keys, "score")
    gmean = y_tr.mean(axis=0)
    types_tr = [feats[t]["type"] for t in tr_tasks]
    vec_tr = np.stack([feats[t]["vec"] for t in tr_tasks])
    chosen = loo_choose(y_tr, types_tr, vec_tr, gmean)
    one = int(np.argmax(gmean))

    # ---- TEST: every choice above is fixed; read labels now.
    if not te_tasks:
        print(f"No TEST task has all {len(keys)} configs yet; LOO choice: {chosen}")
        return
    rng = np.random.default_rng(20261104)
    fields = ("score", "tokens", "latency", "steps")
    Y = {f: matrix(test_rows, te_tasks, keys, f) for f in fields}
    n = len(te_tasks)
    rows_idx = np.arange(n)

    def policy_cols(kind: str) -> np.ndarray:
        if kind == "one":
            return np.full(n, one)
        if kind == "hindsight":
            return np.full(n, int(np.argmax(Y["score"].mean(axis=0))))
        if kind == "oracle":
            return Y["score"].argmax(axis=1)
        cols = []
        for t in te_tasks:
            if kind.startswith("type"):
                m = chosen["type"][1]["m"] if kind == "type" else SENSITIVITY["type_fixed"]["m"]
                cols.append(route(type_weights(types_tr, feats[t]["type"]), y_tr, gmean, m))
            else:
                p = chosen["knn"][1] if kind == "knn" else SENSITIVITY["knn_fixed"]
                w = knn_weights(vec_tr @ feats[t]["vec"], p["k"])
                cols.append(route(w, y_tr, gmean, p["m"]))
        return np.array(cols)

    base = Y["score"][rows_idx, policy_cols("one")]
    report = {"run": str(run), "test_tasks": n, "train_tasks": len(tr_tasks), "configs": keys,
              "train_mean": gmean.round(2).tolist(), "loo": chosen, "policies": {}}
    for kind in ("one", "hindsight", "type", "knn", "type_fixed", "knn_fixed", "oracle"):
        cols = policy_cols(kind)
        res = {f: float(Y[f][rows_idx, cols].mean()) for f in fields}
        res["success"] = float((Y["score"][rows_idx, cols] >= (100 if args.dataset == "sciworld" else 1)).mean())
        res["delta_vs_one"] = paired_ci(Y["score"][rows_idx, cols], base, rng)
        res["configs_used"] = {keys[c][:8]: int((cols == c).sum()) for c in np.unique(cols)}
        report["policies"][kind] = res
    report["null_oracle_gain"] = null_oracle_gain(Y["score"], rng)
    report["oracle_gain_over_hindsight"] = (
        report["policies"]["oracle"]["score"] - report["policies"]["hindsight"]["score"]
    )
    report["per_config_test_mean"] = Y["score"].mean(axis=0).round(2).tolist()
    out = args.out or run / "holdout" / "perquery_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"TEST tasks with all {len(keys)} configs: {n}   LOO choice: {chosen}")
    for kind, r in report["policies"].items():
        d, lo, hi = r["delta_vs_one"]
        print(f"{kind:10s} score {r['score']:6.2f} succ {r['success']:.3f} tok {r['tokens'] / 1e3:5.1f}k "
              f"lat {r['latency']:6.1f}s steps {r['steps']:5.1f}  Δ {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    g, q95 = report["null_oracle_gain"]
    print(f"oracle gain over hindsight-1 {report['oracle_gain_over_hindsight']:.2f}; "
          f"no-interaction null mean {g:.2f} (95% {q95:.2f})")


if __name__ == "__main__":
    main()
