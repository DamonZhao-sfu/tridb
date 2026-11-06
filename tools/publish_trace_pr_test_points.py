"""Add TRACE-PR TEST200 points to "Paper figure · choose curve points" (pfe-data).

One point per configuration measured on the fixed 200-task TEST panel (holdout ledger of a
TRACE-PR run). Quality = mean raw score / 100 (ScienceWorld: negative scores retained),
tokens = mean total tokens, latency = mean episode latency_seconds (receipt; c16 TRACE-PR
runtime), steps = mean agent steps. Points of one run replace that run's earlier points;
nothing else in the page changes. The TRAIN-mean OneConfig and the GP-posterior OneConfig
(final_selection.json / holdout/selection.json, if present) are labelled.

Usage: python -m tools.publish_trace_pr_test_points --dataset sciworld \\
           --run-dir results/trace_pr_sciworld_v060_20261103c --label "TRACE-PR v0.6.0"
"""

import argparse
import fcntl
import hashlib
import json
import re
import sqlite3
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

from trimem.experiments import paper_frontier_editor as editor

ROOT = Path(__file__).resolve().parents[1]
STYLE = {"color": "#1b9e77", "line": "#1b9e77", "shape": "square"}
PLAN = {"none": "VGR", "before": "P1", "after": "P2"}
MIN_TASKS = 200  # set by --min-tasks
ROUTER = "knn"  # set by --router


def short(c):
    s = f"{PLAN[c['filter_order']]} {c['graph_impl']} s{c['seed_count']} d{c['depth']} f{c['fanout']} k{c['returned_k']}"
    for f in ("sem_threshold_pre", "sem_threshold_post"):
        if c.get(f) is not None:
            s += f" θ{c[f]}"
    return s + f" · {c['rank_impl']} · {c['retrieval_unit']} · {c['injection_timing']} · {c['graph_direction']}"


def points(run, dataset, label):
    from tools.fixed_test_panel_guard import validate_ids

    m = json.loads((run / "holdout/manifest.json").read_text())
    db = sqlite3.connect(f"file:{run / 'holdout/ledger.sqlite'}?mode=ro", uri=True)
    cfgs = {k: json.loads(b) for k, b in db.execute("SELECT key, body FROM configs")}
    marks = {}
    fs = run / "final_selection.json"
    if fs.exists():
        f = json.loads(fs.read_text())
        marks.setdefault(f["one_config_sample_mean"], []).append("OneConfig (TRAIN mean)")
        marks.setdefault(f["one_config"], []).append("OneConfig (GP posterior)")
    out = []
    for key, cfg in sorted(cfgs.items()):
        rs = [
            json.loads(r)
            for (r,) in db.execute("SELECT receipt FROM episodes WHERE config=? AND state='done'", (key,))
        ]
        ids = [r["task_id"] for r in rs]
        if len(rs) < MIN_TASKS:
            continue  # only (near-)complete TEST configurations are plotted; n is recorded
        validate_ids(dataset, ids, partial=len(rs) < 200)
        q = [float(r["raw_score"]) / (100 if dataset == "sciworld" else 1) for r in rs]
        tag = " · ".join(marks.get(key, []))
        out.append(
            {
                "id": f"TRACEPR-TEST-{run.name}-{key[:12]}",
                "kind": "e2",
                "method": f"{label} · TEST",
                "short": (tag + " · " if tag else "") + short(cfg) + ("" if len(rs) == 200 else f" · n={len(rs)}/200"),
                "split": "test",
                "n": len(rs),
                "q": st.mean(q),
                "q_se": st.stdev(q) / 200**0.5,
                "tok": st.mean(float(r["total_tokens"]) for r in rs),
                "lat": st.mean(float(r["latency_seconds"]) for r in rs),
                "steps": st.mean(float(r["agent_steps"]) for r in rs),
                "config": cfg,
                "protocol": m["protocol"],
                "source": f"{run.relative_to(ROOT)}/holdout/ledger.sqlite; TRAIN-Pareto config {key}; "
                "quality = raw score / 100" + (" (negative scores retained)" if dataset == "sciworld" else "")
                + "; latency = episode latency_seconds (TRACE-PR c16 runtime profile)",
            }
        )
    return out


ROUTER_STYLE = {"color": "#d95f02", "line": "#d95f02", "shape": "triangle"}
STRAT_STYLE = {"color": "#0b5d8c", "line": "#0b5d8c", "shape": "square"}
STRAT_ROUTER_STYLE = {"color": "#8c2d04", "line": "#8c2d04", "shape": "triangle"}
# TRACE-latent (latent MOBO arm, user 2026-11-05): its own series colour.
LATENT_STYLE = {"color": "#7b3294", "line": "#7b3294", "shape": "circle"}
LATENT_ROUTER_STYLE = {"color": "#c2a5cf", "line": "#c2a5cf", "shape": "triangle"}


def classifier_router_points(run, dataset, label):
    """Opt-PerQuery router with the historical per-(configuration, metric) classifiers
    (tools/trace_perquery_classifier_router.py, Pareto candidates; user 2026-11-05) on measured TEST."""
    path = run / "holdout/router_cls_pareto/report.json"
    rc = json.loads(path.read_text())
    if rc["test_tasks"] < MIN_TASKS:
        raise ValueError("Router report needs the (near-)complete TEST matrix")
    scale = 100 if dataset == "sciworld" else 1
    out = []
    for name, s in rc["settings"].items():
        r, w = s["router"], s["weights"]
        if "score" not in r:
            raise ValueError(f"router setting {name}: routed TEST pairs missing ({r})")
        d = s["delta_quality_router_minus_one"]
        out.append(
            {
                "id": f"TRACEPR-TEST-{run.name}-router-{name}",
                "kind": "e2",
                "method": f"{label} PerQuery Router · TEST",
                "short": f"Opt-PerQuery {name} (wQ {w['wQ']:.2f}, wL {w['wL']:.2f}, wT {w['wT']:.2f}) · "
                f"Δq vs OneConfig {d[0] / scale:+.3f} [{d[1] / scale:+.3f}, {d[2] / scale:+.3f}]",
                "split": "test",
                "n": rc["test_tasks"],
                "q": r["score"] / scale,
                "q_se": r["score_se"] / scale,
                "tok": r["tokens"],
                "lat": r["latency_s"],
                "steps": r["steps"],
                "config": {"router": "Opt-PerQuery: per-(configuration, metric) tabular predictor, best of "
                           "logreg/tree/forest/gboost/xgboost/lightgbm by inner CV; zscore-log-v1 utility",
                           "features": rc["groups"], "weights": w, "predictors": rc["predictors"],
                           "configs_used": r["configs_used"]},
                "source": f"{run.relative_to(ROOT)}/holdout/router_cls_pareto/report.json; candidates = "
                "TEST-measured TRAIN-Pareto configs; each task's routed config read from the measured TEST ledger",
            }
        )
    return out


def router_points(run, dataset, label):
    """Opt-PerQuery-style router points (tools/trace_pr_router_curve.py) on measured TEST."""
    if ROUTER == "classifier":
        return classifier_router_points(run, dataset, label)
    path = run / "holdout/router_curve.json"
    if not path.exists():
        return []
    rc = json.loads(path.read_text())
    if rc["test_tasks"] < MIN_TASKS:
        raise ValueError("Router curve needs the (near-)complete TEST matrix")
    scale = 100 if dataset == "sciworld" else 1
    out = []
    for name, s in rc["settings"].items():
        r, w = s["router"], s["weights"]
        out.append(
            {
                "id": f"TRACEPR-TEST-{run.name}-router-{name}",
                "kind": "e2",
                "method": f"{label} PerQuery Router · TEST",
                "short": f"router {name} (wQ {w['wQ']:.2f}, wL {w['wL']:.2f}, wT {w['wT']:.2f}) · "
                f"Δq vs OneConfig {s['delta_quality_router_minus_one'][0] / scale:+.3f}",
                "split": "test",
                "n": rc["test_tasks"],
                "q": r["score"] / scale,
                "q_se": r["score_se"] / scale,
                "tok": r["tokens"],
                "lat": r["latency"],
                "steps": r["steps"],
                "config": {"router": "kNN utility (task embedding)", "weights": w, "knn": rc["knn"],
                           "configs_used": r["configs_used"]},
                "source": f"{run.relative_to(ROOT)}/holdout/router_curve.json; candidates = TEST-measured "
                "TRAIN-Pareto configs; each task's routed config read from the measured TEST ledger",
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=("alfworld", "sciworld", "webshop"))
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--label", default="TRACE-PR v0.6.0")
    ap.add_argument("--min-tasks", type=int, default=200)
    ap.add_argument("--router", choices=("knn", "classifier"), default="knn",
                    help="classifier = Opt-PerQuery classifier router (holdout/router_cls_pareto/report.json)")
    a = ap.parse_args()
    global MIN_TASKS, ROUTER
    MIN_TASKS = a.min_tasks
    ROUTER = a.router
    run = (a.run_dir if a.run_dir.is_absolute() else ROOT / a.run_dir).resolve()
    new = points(run, a.dataset, a.label) + router_points(run, a.dataset, a.label)
    if not new:
        raise SystemExit("No complete TEST configuration yet")
    prefix = f"TRACEPR-TEST-{run.name}-"
    out_dir = ROOT / "results/trace_pr_curve_points"
    out_dir.mkdir(exist_ok=True)
    with editor.PAGE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        page = editor.PAGE.read_text()
        match = re.search(r'(<script[^>]*id="pfe-data"[^>]*>)(.*?)(</script>)', page, re.S)
        data = json.loads(match[2])
        ds = next(d for d in data["datasets"] if d["key"] == a.dataset)
        ds["points"] = [p for p in ds["points"] if not p["id"].startswith(prefix)] + new
        data["generated"] = datetime.now(timezone.utc).isoformat()
        data["fingerprint"] = hashlib.sha256(json.dumps(data["datasets"], sort_keys=True).encode()).hexdigest()
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        updated = page[: match.start(2)] + payload + page[match.end(2) :]
        marker = "const seriesOf=p=>"
        # Keep every earlier label's style; add or refresh this label's two series.
        block = re.search(r"/\* TRACE-PR-STYLES \*/\n(.*?)/\* END-TRACE-PR-STYLES \*/", updated, re.S)
        styles = dict(re.findall(r'SERIES\[("[^"]*")\]=(\{[^;]*\});', block[1])) if block else {}
        strat = "stratified" in a.label
        latent = "latent" in a.label
        styles[json.dumps(a.label)] = json.dumps(LATENT_STYLE if latent else STRAT_STYLE if strat else STYLE)
        styles[json.dumps(a.label + " PerQuery Router")] = json.dumps(
            LATENT_ROUTER_STYLE if latent else STRAT_ROUTER_STYLE if strat else ROUTER_STYLE)
        updated = re.sub(r"/\* TRACE-PR-STYLES \*/.*?/\* END-TRACE-PR-STYLES \*/\n?", "", updated, flags=re.S)
        style = "\n".join(f"SERIES[{k}]={v};" for k, v in styles.items())
        updated = updated.replace(marker, f"/* TRACE-PR-STYLES */\n{style}\n/* END-TRACE-PR-STYLES */\n{marker}", 1)
        backup = out_dir / f"qa-trajectory.before.{datetime.now():%Y%m%dT%H%M%S}.html"
        backup.write_text(page)
        (out_dir / f"{a.dataset}_{run.name}.json").write_text(json.dumps(new, indent=1, ensure_ascii=False))
        editor.atomic_write(editor.PAGE, updated)
    print(json.dumps({"dataset": a.dataset, "points": len(new), "backup": str(backup)}))
    for p in new:
        print(f"  q={p['q']:.4f}±{p['q_se']:.4f} tok={p['tok'] / 1e3:.1f}k lat={p['lat']:.1f}s steps={p['steps']:.1f}  {p['short']}")


if __name__ == "__main__":
    main()
