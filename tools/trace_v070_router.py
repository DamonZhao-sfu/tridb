"""Opt-PerQuery router and Opt-OneConfig for a TRACE v0.7.0 run: per-configuration OWN samples + NoMemory.

Candidates (user 2026-11-05): the configurations retained by BRANE fuzzy Pareto pruning on TRAIN
(``brane.fuzzy_pareto``, TAU_ACC 0.02, TAU_COST 0.10) over complete never-pruned BO configurations U NoMemory,
plus NoMemory always (user 2026-11-05), i.e. ``holdout/selection.json["router_candidates"]`` (written by
``tools.trace_v070_holdout``, whose TEST set contains them). Fit / route refuse if any candidate lacks complete TEST receipts. OneConfig candidates = the same.

The historical classifier router (``tools.trace_perquery_classifier_router``, read-only import) assumes a dense
TRAIN matrix (one shared panel). Under per-configuration samples (Amendment A.4) every candidate is fitted on its
OWN 100 TRAIN tasks (the historical sparse BRANE layout, ``query_router.sparse.SparseProfile``):

* features (``structured + probe + emb``) are fitted on the union of the candidates' TRAIN tasks;
* per candidate and metric, ``predictors.fit`` on its own rows (honest OOF on its own tasks); on the other union
  tasks the fitted model's prediction is honest as well (those tasks were never its training rows);
* ``utility_norm`` uses those honest predictions and each candidate's own-sample means / SEs;
* OneConfig = argmax of the same utility on the own-sample means; TRAIN replay averages the measured outcome
  only where the routed candidate measured the task (coverage reported).
``route`` / ``report`` read TEST outcomes from the BO holdout ledger UNION the NoMemory TEST ledger.
The dense kNN curve (``tools.trace_pr_router_curve``) needs a shared panel and is not applicable here.

  .venv-optimizer/bin/python -m tools.trace_v070_router {fit|route|report} RUN --dataset alfworld
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tools import trace_perquery_classifier_router as cr

PROTOCOL = "trace-v070-perquery-classifier-sparse-v1"


def router_candidates(run: Path) -> list[str]:
    """The BRANE fuzzy set of ``holdout/selection.json``; each must have complete TEST receipts."""
    from trimem.experiments.trace_v070_baseline import baseline_entry
    from trimem.optimizer.joint_search_ledger import SearchLedger

    sel = json.loads((run / "holdout/selection.json").read_text())
    keys = list(sel.get("router_candidates", sel["router_candidate_keys"]))
    nm_key = sel["baselines"]["NoMemory"]["key"]
    if nm_key not in keys:
        raise ValueError("NoMemory must be a router candidate (user 2026-11-05)")
    if not keys:
        raise ValueError("no router candidates in holdout/selection.json")
    tm = json.loads((run / "holdout/manifest.json").read_text())
    with SearchLedger(run / "holdout/ledger.sqlite", tm) as led:
        test_done = {e["key"]: len(e["receipts"]) for e in led.complete_configs()}
    nm = baseline_entry(run, "TEST")
    if nm is not None and nm["complete"]:
        test_done[nm["key"]] = len(nm["receipts"])
    missing = [k for k in keys if test_done.get(k) != len(tm["tasks"])]
    if missing:
        raise ValueError(f"router candidates without complete TEST receipts: {[k[:12] for k in missing]}")
    return keys


def router_rows(run: Path) -> tuple[list[str], dict, dict]:
    """(candidate keys, key -> own tasks, key -> task -> outcome) of the router candidates (TRAIN)."""
    from trimem.experiments.trace_v070_baseline import baseline_rows
    from trimem.experiments.trace_v070_candidates import candidates
    from trimem.experiments.trace_v070_ledger import SampledLedger

    allowed = set(router_candidates(run))
    rows = [r for r in candidates(run, "TRAIN", include_baselines=True, full_only=True) if r["key"] in allowed]
    if {r["key"] for r in rows} != allowed:
        raise ValueError("a router candidate is not a full, never-pruned TRAIN candidate")
    keys = [r["key"] for r in rows]
    own = {r["key"]: list(r["tasks"]) for r in rows}
    out: dict[str, dict] = {}
    m = json.loads((run / "manifest.json").read_text())
    with SampledLedger(run / "ledger.sqlite", m) as led:
        receipts = {r["key"]: led.receipts(r["key"]) for r in rows if r["source"] == "bo"}
    receipts.update({r["key"]: baseline_rows(run, "TRAIN") for r in rows if r["source"] == "baseline"})
    for r in rows:
        rec = receipts[r["key"]]
        out[r["key"]] = {str(x["task_id"]): {"score": float(x["raw_score"]), "tokens": float(x["total_tokens"]),
                                             "latency_ms": float(x["latency_seconds"]) * 1000.0,
                                             "steps": float(x.get("agent_steps") or 0.0)} for x in rec}
    return keys, own, out


def honest_matrix(union, keys, own, oof, predict) -> np.ndarray:
    """(n_union, C, 3): own-task OOF where the candidate measured the task, model prediction elsewhere."""
    H = np.zeros((len(union), len(keys), 3))
    for c, k in enumerate(keys):
        pos = {t: i for i, t in enumerate(own[k])}
        pred = None
        for i, t in enumerate(union):
            if t in pos:
                H[i, c] = oof[k][pos[t]]
            else:
                pred = predict(k) if pred is None else pred
                H[i, c] = pred[i]
    return H


def own_stats(keys, own, rows, scale_q):
    obs = {k: np.stack([[max(rows[k][t]["score"], 0.0) / scale_q, np.log(rows[k][t]["tokens"]),
                         np.log(rows[k][t]["latency_ms"])] for t in own[k]]) for k in keys}
    means = np.stack([obs[k].mean(0) for k in keys])
    se = np.stack([obs[k].std(0, ddof=1) / len(own[k]) ** 0.5 for k in keys])
    return obs, means, se


def stage_fit(run: Path, dataset: str, jobs: int, seed: int, out: Path) -> None:
    from trimem.optimizer.query_router import features, runtime

    out.mkdir(parents=True, exist_ok=True)
    keys, own, rows = router_rows(run)
    items = {i["item_id"]: i for i in json.loads((run / "train_panel.json").read_text())["items"]}
    union = sorted({t for k in keys for t in own[k]})
    test_ids = json.loads((run / "test_panel.json").read_text())["tasks"]
    recs = cr.feature_records(dataset, [items[t] for t in union], test_ids, out / "features.json")
    spec = features.fit_spec([recs[t] for t in union], cr.GROUPS, emb_dims=cr.EMB_DIMS)
    X = {t: runtime.transform(spec, recs[t]) for t in union}
    binary, scale_q = dataset == "alfworld", (1.0 if dataset == "alfworld" else 100.0)
    args = [(k, np.stack([X[t] for t in own[k]]),
             np.array([max(rows[k][t]["score"], 0.0) / scale_q for t in own[k]]),
             np.array([rows[k][t]["tokens"] for t in own[k]]),
             np.array([rows[k][t]["latency_ms"] for t in own[k]]), binary, seed) for k in keys]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        models = dict(ex.map(cr._fit_one, args))
    Xu = np.stack([X[t] for t in union])

    def predict(k):
        m = models[k]
        return np.stack([cr.qual_pred(m["quality"], Xu), np.log(cr.cost_pred(m["ln_tokens"], Xu)),
                         np.log(cr.cost_pred(m["ln_latency"], Xu))], axis=1)

    honest = honest_matrix(union, keys, own, {k: np.asarray(models[k]["oof"]) for k in keys}, predict)
    _, cmeans, cse = own_stats(keys, own, rows, scale_q)
    norm = runtime.utility_norm(honest, cmeans, cse)
    # OneConfig inputs on the router's scale: quality target max(score, 0) / scale, raw tokens, raw latency (ms)
    raw = {k: [float(np.mean([max(rows[k][t]["score"], 0.0) / scale_q for t in own[k]])),
               *(float(np.mean([rows[k][t][f] for t in own[k]])) for f in ("tokens", "latency_ms"))] for k in keys}
    (out / "model.json").write_text(json.dumps({
        "protocol": PROTOCOL, "base_protocol": cr.PROTOCOL, "dataset": dataset, "candidate_set": "brane-fuzzy-pareto",
        "groups": cr.GROUPS, "emb_dims": cr.EMB_DIMS, "seed": seed, "candidates": keys, "train_tasks": union,
        "own_samples": own, "own_means": raw, "spec": spec, "norm": norm, "quality_scale": scale_q,
        "models": {k: {x: models[k][x] for x in ("quality", "ln_tokens", "ln_latency", "selection")} for k in keys},
        "train_oof": honest.tolist(),
    }, default=float))


def stage_route(run: Path, out: Path) -> None:
    from trimem.optimizer.query_router import runtime

    M = json.loads((out / "model.json").read_text())
    keys, norm, spec = M["candidates"], M["norm"], M["spec"]
    recs = {r["query_id"]: r for r in json.loads((out / "features.json").read_text())["records"]}
    te = json.loads((run / "test_panel.json").read_text())["tasks"]
    X_te = np.stack([runtime.transform(spec, recs[t]) for t in te])
    pq = np.stack([cr.qual_pred(M["models"][k]["quality"], X_te) for k in keys], axis=1)
    pt = np.stack([cr.cost_pred(M["models"][k]["ln_tokens"], X_te) for k in keys], axis=1)
    pl = np.stack([cr.cost_pred(M["models"][k]["ln_latency"], X_te) for k in keys], axis=1)
    means = np.array([M["own_means"][k] for k in keys])
    oof = np.asarray(M["train_oof"])
    _, _, rows = router_rows(run)
    routes = {}
    for name, w in cr.weight_set().items():
        r = cr._route(pq, pt, pl, w, norm)
        one = int(cr._route(means[:, 0], means[:, 1], means[:, 2], w, norm))
        rr = cr._route(oof[:, :, 0], np.exp(oof[:, :, 1]), np.exp(oof[:, :, 2]), w, norm)
        hit = [rows[keys[c]].get(t) for t, c in zip(M["train_tasks"], rr)]
        got = [h for h in hit if h is not None]
        routes[name] = {"weights": w, "oneconfig": keys[one], "routes": {t: keys[c] for t, c in zip(te, r)},
                        "train_replay": {"coverage": len(got) / len(hit),
                                         "quality": float(np.mean([h["score"] for h in got])) if got else None},
                        "configs_used": {keys[c][:8]: int((r == c).sum()) for c in np.unique(r)}}
    (out / "routes.json").write_text(json.dumps({"protocol": PROTOCOL, "candidates": keys, "settings": routes},
                                                indent=1))


def outcomes_with_nomemory(run: Path):
    """cr.outcomes that unions the NoMemory TEST ledger into the BO holdout ledger."""
    original = cr.outcomes
    nm = run / "holdout/baselines/nomemory/ledger.sqlite"

    def outcomes(ledger, keys=None):
        res = original(ledger, keys)
        if Path(ledger).resolve() == (run / "holdout/ledger.sqlite").resolve() and nm.exists():
            for k, d in original(nm, keys).items():
                res.setdefault(k, {}).update(d)
        return res

    return original, outcomes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("fit", "route", "report"))
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--dataset", required=True, choices=("alfworld", "sciworld", "webshop"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run = a.run_dir if a.run_dir.is_absolute() else cr.ROOT / a.run_dir
    out = run / "holdout/router_cls_v070"
    router_candidates(run)  # refuse unless every fuzzy-set candidate has complete TEST receipts
    if a.stage == "fit":
        stage_fit(run, a.dataset, a.jobs, a.seed, out)
        return
    original, patched = outcomes_with_nomemory(run)
    cr.outcomes = patched
    try:
        stage_route(run, out) if a.stage == "route" else cr.stage_report(run, a.dataset, out)
    finally:
        cr.outcomes = original


if __name__ == "__main__":
    main()
