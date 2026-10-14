"""D2: fit the offline prior on logged outcomes and report leave-one-task-out (LOTO) quality.

Output per held-out task τ: ``prior_<task>.pkl`` = {'hurdle', 'X', 'delta'} fitted on every task
except τ (so an adaptive cell on τ never saw τ), plus ``estimators.json`` with, per held-out task,
AUC of P̂(Δ>0), Spearman ρ of ĝ vs Δ, calibration (mean P̂ vs observed rate) and the counts.

    PYTHONPATH=. .venv-e0/bin/python -m trimem.objective.train --out results/trimem_v0.1.0/objective
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import psycopg

from trimem.objective.models import HurdleModel
from trimem.objective.pool import vector

LOAD_SQL = """
SELECT task, policy, features, delta FROM trimem_outcome
WHERE delta IS NOT NULL AND session NOT LIKE 'smoke%%' AND (%(fixed)s OR policy NOT LIKE 'fixed-%%')
  AND coalesce((features->>'nocontext')::float, 0) = 0
"""


def load(
    dsn: str, include_fixed: bool = True
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(LOAD_SQL, {"fixed": include_fixed}).fetchall()
    tasks = [r[0] for r in rows]
    X = np.vstack([vector(r[2]) for r in rows]) if rows else np.zeros((0, 0))
    return tasks, X, np.array([float(r[3]) for r in rows]), [r[1] for r in rows]


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


V2_SQL = """
SELECT session, task, iteration, policy, features, parent_fitness, child_fitness, best_fitness, delta_best, valid
FROM trimem_outcome
WHERE delta IS NOT NULL AND session NOT LIKE 'smoke%%'
  AND coalesce((features->>'nocontext')::float, 0) = 0
ORDER BY session, iteration
"""


def load_v2(dsn: str) -> list[dict]:
    """Rows with the v2 label Δ_best = max(0, child − run best at decision). Rows logged before v2
    have no recorded best: it is rebuilt from the session's own log (seed = iteration 1's parent,
    then the best child of every earlier iteration) — exact up to OpenEvolve's concurrency, where a
    child of iteration t−1..t−4 may not have returned yet when iteration t was decided."""
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(V2_SQL).fetchall()
    out, cur, best = [], None, 0.0
    for sess, task, it, pol, f, pf, cf, bf, db, valid in rows:
        if sess != cur:
            cur, best = sess, float(pf or 0.0)
        b = float(bf) if bf is not None else max(best, float(pf or 0.0))
        y = float(db) if db is not None else max(0.0, float(cf or 0.0) - b)
        feats = dict(f, best_gap=b - float(pf or 0.0))
        out.append({"task": task, "policy": pol, "features": feats, "y": y, "session": sess, "iteration": it,
                    "valid": bool(valid)})
        best = max(best, float(cf or 0.0))
    return out


def main_v2(out: Path, dsn: str) -> dict:
    from trimem.objective.pool import vector_none, vector_v2

    rows = load_v2(dsn)
    tasks = np.array([r["task"] for r in rows])
    none = np.array([r["features"].get("fam_none", 0.0) == 1.0 for r in rows])
    Xm = np.vstack([vector_v2(r["features"]) for r in rows])
    Xn = np.vstack([vector_none(r["features"]) for r in rows])
    y = np.array([r["y"] for r in rows])
    report: dict = {"space": "v2", "label": "delta_best", "n": len(rows), "n_none": int(none.sum()),
                    "new_best_rate": float((y > 0).mean()),
                    "policies": {p: sum(r["policy"] == p for r in rows) for p in sorted({r["policy"] for r in rows})},
                    "loto": {}}
    for task in sorted(set(tasks)):
        tr, te = tasks != task, tasks == task
        mem = HurdleModel().fit(Xm[tr & ~none], y[tr & ~none])
        nmod = HurdleModel().fit(Xn[tr & none], y[tr & none])
        g = np.where(none[te], nmod.predict(Xn[te]), mem.predict(Xm[te]))
        p_hat = np.where(none[te], nmod.p_improve(Xn[te]), mem.p_improve(Xm[te]))
        yy = (y[te] > 0).astype(int)
        report["loto"][task] = {"n_test": int(te.sum()), "auc_p_new_best": _auc(yy, p_hat),
                                "spearman_g": _spearman(g, y[te]), "mean_p_hat": float(p_hat.mean()),
                                "observed_rate": float(yy.mean()),
                                "none_g_hat": float(nmod.predict(Xn[te & none]).mean()) if (te & none).any() else None,
                                "none_observed": float(y[te & none].mean()) if (te & none).any() else None,
                                "mem_observed": float(y[te & ~none].mean())}
        (out / f"prior_{task.replace(':', '_')}.pkl").write_bytes(pickle.dumps({
            "mem": mem, "none": nmod, "X_mem": Xm[tr & ~none], "y_mem": y[tr & ~none],
            "X_none": Xn[tr & none], "y_none": y[tr & none], "held_out": task}))
    (out / "estimators.json").write_text(json.dumps(report, indent=2))
    return report


def main_v3(out: Path, dsn: str, domain: str = "math", held_out: str | None = None,
            include_math: bool = False, in_task: bool = False) -> dict:
    """v3 prior: Δ_best estimators (memory / none) on vector_v3, plus the O4 cost models:
    output tokens per candidate (gradient boosting), P(valid | candidate) (calibrated logistic),
    and the latency law seconds = a + b·prompt + d·output fitted on the joined telemetry."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from trimem.objective.costs import session_costs
    from trimem.objective.pool import vector_none, vector_v3

    allrows = load_v2(dsn)
    rows = [r for r in allrows if r["task"].startswith(f"{domain}:")]
    if domain == "ale":  # v3 decisions only: their features and labels are already scale-free
        rows = [r for r in rows if r["policy"].startswith("v3")]
        if include_math:  # tasks run one at a time: math decisions stand in for unseen ALE tasks
            rows = [r for r in allrows if r["task"].startswith("math:")] + rows
    costs: dict[str, dict] = {}
    for r in rows:
        if r["session"] not in costs:
            costs[r["session"]] = session_costs(r["session"])
        r["cost"] = costs[r["session"]].get(r["iteration"])
    tasks = np.array([r["task"] for r in rows])
    none = np.array([r["features"].get("fam_none", 0.0) == 1.0 for r in rows])
    X = np.vstack([vector_v3(r["features"]) for r in rows])
    Xn = np.vstack([vector_none(r["features"]) for r in rows])
    y = np.array([r["y"] for r in rows])
    valid = np.array([r["valid"] for r in rows], dtype=int)
    has = np.array([r["cost"] is not None for r in rows])
    out_tok = np.array([r["cost"]["completion_tokens"] if r["cost"] else np.nan for r in rows])
    prm = np.array([r["cost"]["prompt_tokens"] if r["cost"] else np.nan for r in rows])
    sec = np.array([r["cost"]["llm_ms"] / 1000 if r["cost"] else np.nan for r in rows])
    A = np.column_stack([np.ones(has.sum()), prm[has], out_tok[has]])
    a, b, d = np.linalg.lstsq(A, sec[has], rcond=None)[0]
    r2 = 1 - ((sec[has] - A @ [a, b, d]) ** 2).sum() / ((sec[has] - sec[has].mean()) ** 2).sum()

    def out_model():
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)

    def valid_model():
        return CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
                                      method="isotonic", cv=3)

    var_y = float(np.var(y[~none])) or 1e-4
    var_bin = float(np.var((y[~none] > 0).astype(float))) or 0.1
    out_budget = float(np.nanmean(out_tok[has])) if has.any() else 0.0
    report: dict = {"space": "v3", "domain": domain, "label": "delta_best", "n": len(rows),
                    "n_cost_joined": int(has.sum()), "out_budget": out_budget, "var_bin": var_bin,
                    "latency_law": {"a": a, "b": b, "d": d, "r2": r2}, "online_tau2_sigma2": var_y, "loto": {}}
    for task in ([held_out] if held_out else sorted(set(tasks))):
        # in_task: BRANE-style profiling — the prior is fitted on this task's own exploration run too
        # (a separate run; the optimizer run that uses the prior starts from scratch)
        tr, te = (np.ones(len(tasks), dtype=bool) if in_task else tasks != task), tasks == task
        mem = HurdleModel().fit(X[tr & ~none], y[tr & ~none])
        nmod = HurdleModel().fit(Xn[tr & none], y[tr & none])
        om = out_model().fit(X[tr & has], out_tok[tr & has])
        vm = valid_model().fit(X[tr], valid[tr])
        prior = {"mem": mem, "none": nmod, "X_mem": X[tr & ~none], "y_mem": y[tr & ~none],
                 "X_none": Xn[tr & none], "y_none": y[tr & none], "out_model": om, "valid_model": vm,
                 "latency": (float(a), float(b), float(d)), "var_y": var_y, "var_bin": var_bin,
                 "out_budget": out_budget, "held_out": task}
        (out / f"prior_{task.replace(':', '_')}.pkl").write_bytes(pickle.dumps(prior))
        if not te.any():  # held-out task without logged decisions yet: train only
            report["loto"][task] = {"n_train": int(tr.sum()), "n_test": 0}
            continue
        g = np.where(none[te], nmod.predict(Xn[te]), mem.predict(X[te]))
        p_hat = np.where(none[te], nmod.p_improve(Xn[te]), mem.p_improve(X[te]))
        yy = (y[te] > 0).astype(int)
        oh = om.predict(X[te & has]) if (te & has).any() else np.zeros(0)
        report["loto"][task] = {
            "n_test": int(te.sum()), "auc_p_new_best": _auc(yy, p_hat), "spearman_g": _spearman(g, y[te]),
            "out_tokens_mae": float(np.mean(np.abs(oh - out_tok[te & has]))) if len(oh) else None,
            "out_tokens_spearman": _spearman(oh, out_tok[te & has]),
            "auc_valid": _auc(valid[te], vm.predict_proba(X[te])[:, 1]),
            "observed_valid": float(valid[te].mean()), "mean_p_valid": float(vm.predict_proba(X[te])[:, 1].mean())}
    (out / "estimators.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--dsn",
        default=f"postgresql://{os.environ.get('USER')}@127.0.0.1:55441/trimem_evo",
    )
    ap.add_argument(
        "--no-fixed",
        action="store_true",
        help="train on exploration (random) rows only",
    )
    ap.add_argument("--space", choices=["v1", "v2", "v3"], default="v1")
    ap.add_argument("--domain", choices=["math", "ale"], default="math")
    ap.add_argument("--held-out", default=None, help="train one prior that excludes this task (v3)")
    ap.add_argument("--include-math", action="store_true", help="ALE: add the math decisions to the training set")
    ap.add_argument("--in-task", action="store_true", help="fit the --held-out task's prior on its own data too")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.space in ("v2", "v3"):
        rep = (main_v3(out, args.dsn, args.domain, args.held_out, args.include_math, args.in_task)
               if args.space == "v3"
               else main_v2(out, args.dsn))
        print(json.dumps(rep, indent=2))
        return 0
    tasks, X, delta, policies = load(args.dsn, include_fixed=not args.no_fixed)
    t = np.array(tasks)
    report: dict = {
        "n": len(delta),
        "improve_rate": float((delta > 0).mean()) if len(delta) else None,
        "policies": {p: policies.count(p) for p in sorted(set(policies))},
        "loto": {},
    }
    for task in sorted(set(tasks)):
        tr, te = t != task, t == task
        model = HurdleModel().fit(X[tr], delta[tr])
        p, g = model.p_improve(X[te]), model.predict(X[te])
        y = (delta[te] > 0).astype(int)
        report["loto"][task] = {
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "auc_p_improve": _auc(y, p),
            "spearman_g_delta": _spearman(g, delta[te]),
            "mean_p_hat": float(p.mean()),
            "observed_rate": float(y.mean()),
            "mean_g_hat": float(g.mean()),
            "mean_delta": float(delta[te].mean()),
        }
        (out / f"prior_{task.replace(':', '_')}.pkl").write_bytes(
            pickle.dumps(
                {"hurdle": model, "X": X[tr], "delta": delta[tr], "held_out": task}
            )
        )
    full = HurdleModel().fit(X, delta)
    (out / "prior_all.pkl").write_bytes(
        pickle.dumps({"hurdle": full, "X": X, "delta": delta, "held_out": None})
    )
    (out / "estimators.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
