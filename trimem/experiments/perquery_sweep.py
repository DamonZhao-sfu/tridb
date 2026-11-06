"""Opt-PerQuery (BRANE) weight sweep on the ALFWorld / ScienceWorld v7 TEST sets (user request 2026-10-23).

The DEPLOYED Opt-PerQuery router (the frozen ``selection.json`` ``weight_slices`` artifact, sha-checked; NO refit)
is replayed at the weights of the recorded weight set ``WEIGHT_SET`` (user 2026-10-23, revised before the run:
the two curves at step 0.2 -- Q-T (w_L = 0; w_T = 0, 0.2, ..., 1) and Q-L (w_T = 0; w_L = 0.2, ..., 1) = 11
(w_Q, w_L, w_T) incl. (1, 0, 0); the first design's full 0.05 simplex grid (231) was dropped), with the deployed
decision rule (``runtime.choose_utility``: zscore-log-v1 utility on
the artifact's predictions of each test task). Predictions are computed once per task; the routing of the four
deployed TEST_WEIGHTS is checked to be IDENTICAL to ``sparse.route_test_weights`` and to the frozen
``test/routing.json`` (SystemExit otherwise). Router overhead per task = the deployed routing table's overhead_ms.

* pairs: every (configuration, task) a weight routes to runs ONCE (``bo_pool_driver.pairs_of``). A pair already
  measured under the SAME runtime (controlled-c16-v1, quality reuse off) is reused, never rerun: the v7 test
  (``test_v7c16``, ``...:pairs-c16-v7:...``), the fixed-plan arms (``test_v7c16/fixed_plans``,
  ``...:pairs-c16-v7-fixed:...``) -- SemFilter plans only when the row is labelled semfilter-v2 -- and, ALFWorld
  only, the c16 rerun (``test_c16``; SemFilter-free plans). NoMemory pairs: official-evaluator pair dirs of the v7
  test (+ ALF c16) WITH an ``episodes.jsonl`` cost record (c16 wall), never the controlled-v1 NoMemory (other
  runtime); missing NoMemory pairs run at c16 here. New pairs -> ``<run>/test_v7c16/pq_sweep/pairs``, opt experiment
  ``...:pairs-c16-v7-sweep:...``; layout = ``alf_test_c16`` (one configuration at a time, 2 halves x 8, small
  configurations share one final batch), semfilter-v2 env from state.json, quality reuse OFF.
* routing.json records ``weight_set``; one written for ANOTHER weight set is archived (``routing.<set>.<ts>.json``),
  never mixed; the same set with other routes is refused.
* report (``<run>/test_v7c16/pq_sweep/report_pq_sweep.{json,md}``): per weight quality ± SE, paired gain vs the
  controlled-v1 NoMemory ± SE, tokens, latency wall (opt e2e + overhead) and LLM span (+ retrieval + overhead; the
  page's plotted definition), steps, distinct configurations, routing shares by plan, the router's predicted
  quality / tokens / latency; weights with IDENTICAL routings are grouped; the Q-T slice (w_L = 0, by w_T) and Q-L
  slice (w_T = 0, by w_L); the test Pareto fronts of the swept points; the four deployed TEST_WEIGHTS routed and
  assembled from the same pairs as a cross-check against the v7 report (all measured there, never run here).

Stages: ``plan`` (counts new vs reused, time estimate; read-only), ``run`` (``--dry``: log only, writes nothing),
``report``.

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 POOL_SEMFILTER=v2 .venv-optimizer/bin/python -m \\
        trimem.experiments.perquery_sweep {plan|run|report} --dataset {alfworld|sciworld|webshop} [--summary]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stats
from datetime import UTC, datetime
from pathlib import Path

DATASETS = ("alfworld", "sciworld", "webshop")  # webshop: user 2026-10-23 (additive)
SUBDIR = "test_v7c16/pq_sweep"
ARM_TAG = "pairs-c16-v7-sweep"
PROTOCOL = "perquery-weight-sweep-test-v1"
WEIGHT_SET = "curves-step0.2-v1"  # recorded in routing.json / report; Q-T + Q-L slices at step 0.2
CURVE_STEP = 0.2
N_WEIGHTS = 11
CONCURRENCY = 16
RUNTIME = "controlled-c16-v1"
REPORT = "report_pq_sweep.json"
REPORT_MD = "report_pq_sweep.md"
QNAME = {"alfworld": "success rate", "sciworld": "score/100", "webshop": "reward"}
TOL = 1e-6

Weights = tuple[float, float, float]


def configure(weight_set: str, *, dataset: str | None = None) -> None:
    """Select a separately stored sweep; preserve the historical 11-point results."""
    global WEIGHT_SET, SUBDIR, ARM_TAG, CURVE_STEP, N_WEIGHTS
    if weight_set not in ("curves-step0.2-v1", "curves-log2-v1", "quality-first-v1"):
        raise ValueError(f"Unknown weight set: {weight_set}")
    if weight_set == "quality-first-v1" and dataset != "webshop":
        raise ValueError("quality-first-v1 is authorized for WebShop only")
    WEIGHT_SET = weight_set
    if weight_set == "quality-first-v1":
        SUBDIR = "test_v7c16/pq_sweep_qualityfirst"
        ARM_TAG = "pairs-c16-v7-sweep-qualityfirst-c16"
        CURVE_STEP, N_WEIGHTS = None, 22
        return
    log_grid = weight_set == "curves-log2-v1"
    SUBDIR = "test_v7c16/pq_sweep" + ("_log2" if log_grid else "")
    ARM_TAG = "pairs-c16-v7-sweep" + ("-log2" if log_grid else "")
    if log_grid and dataset in ("webshop", "sciworld"):
        ARM_TAG += "-c16"
    CURVE_STEP = None if log_grid else 0.2
    N_WEIGHTS = 25 if log_grid else 11


def out_dir(base: Path) -> Path:
    return base / SUBDIR


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


# ------------------------------------------------------------------ weights (pure)


def weights() -> list[Weights]:
    """Q-T and Q-L slices sharing pure quality; historical 11 or log-spaced 25 vectors."""
    if WEIGHT_SET == "quality-first-v1":
        # Fixed a priori quality-first weights; preserve the frozen deployment model.
        return [(1., 0., 0.)] + [
            (q, round((1-q)*share, 6), round((1-q)*(1-share), 6))
            for q in (.99, .975, .95, .925, .9, .85, .8)
            for share in (0., .5, 1.)
        ]
    if WEIGHT_SET == "curves-log2-v1":
        quality = [1.0, *[round(1 / (1 + 2.0**i), 6) for i in range(-5, 6)], 0.0]
        qt = [(q, 0.0, round(1 - q, 6)) for q in quality]
        ql = [(q, round(1 - q, 6), 0.0) for q in quality[1:]]
        return qt + ql
    n = round(1 / CURVE_STEP)
    r = lambda v: round(v, 6)  # noqa: E731
    qt = [(r(1 - i * CURVE_STEP), 0.0, r(i * CURVE_STEP)) for i in range(n + 1)]
    ql = [(r(1 - i * CURVE_STEP), r(i * CURVE_STEP), 0.0) for i in range(1, n + 1)]
    ws = qt + ql
    if len(ws) != N_WEIGHTS or len(set(ws)) != N_WEIGHTS:
        raise SystemExit(f"weight set has {len(ws)} points, expected {N_WEIGHTS}")
    return ws


def arm(w: Weights) -> str:
    """``Opt-PerQuery-w<wQ>-<wL>-<wT>`` (= ``sparse.arm_name``)."""
    return "Opt-PerQuery-w" + "-".join(f"{v:g}" for v in w)


def qt_slice(ws: list[Weights]) -> list[Weights]:
    """Quality-tokens curve: w_L = 0, ordered by w_T ascending (w_Q = 1 -> w_T = 1). Pure."""
    return sorted((w for w in ws if w[1] == 0), key=lambda w: w[2])


def ql_slice(ws: list[Weights]) -> list[Weights]:
    """Quality-latency curve: w_T = 0, ordered by w_L ascending. Pure."""
    return sorted((w for w in ws if w[2] == 0), key=lambda w: w[1])


def same_weights(a, b) -> bool:
    return all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------------ routing (deployed artifact, no refit)


def run_base(ds: str) -> Path:
    from trimem.experiments import bo_pool_driver as d

    if ds not in DATASETS:
        raise SystemExit(f"unknown dataset {ds}")
    return d.run_dir(ds)


def deployed(base: Path) -> tuple[dict, dict, dict]:
    """(artifact, test feature records by task, frozen deployed routing.json). Refuses a sha mismatch."""
    from trimem.optimizer.query_router import runtime

    sel = json.loads((base / "selection.json").read_text())
    c = sel["weight_slices"]
    art = json.loads(Path(c["path"]).read_text())
    if runtime.artifact_sha(art) != c["sha"]:
        raise SystemExit(
            f"{c['path']}: artifact sha != selection.json weight_slices.sha"
        )
    recs = {
        r["query_id"]: r
        for r in json.loads((base / "router" / "features_test.json").read_text())[
            "records"
        ]
    }
    tab = json.loads((base / "test" / "routing.json").read_text())
    return art, recs, tab


def route_all(
    model: dict, xs: dict[str, object], ids: list[str], ws: list[Weights]
) -> tuple[dict[Weights, list[int]], dict[Weights, list[float]]]:
    """Per weight: the candidate index of every task (deployed rule = argmax of ``runtime.utility`` on the frozen
    predictions, ties -> lowest index) and the routed predictions' means (quality, tokens, latency ms)."""
    import numpy as np

    from trimem.optimizer.query_router import runtime

    P = {q: runtime.predict(model, xs[q]) for q in ids}
    route: dict[Weights, list[int]] = {}
    pred: dict[Weights, list[float]] = {}
    for w in ws:
        idx, pq, pt, pl = [], [], [], []
        for q in ids:
            qq, t, lat = P[q]
            u = runtime.utility(qq, np.exp(t), np.exp(lat), w, model["cost_norm"])
            i = int(np.argmax(u))
            idx.append(i)
            pq.append(float(qq[i]))
            pt.append(float(math.exp(t[i])))
            pl.append(float(math.exp(lat[i])))
        route[w] = idx
        pred[w] = [stats.mean(pq), stats.mean(pt), stats.mean(pl)]
    return route, pred


def build(ds: str) -> tuple[Path, dict]:
    """The sweep routing (compact): configs, per-weight candidate index per task, overhead, predictions; the four
    deployed weights verified against ``sparse.route_test_weights`` and test/routing.json."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space
    from trimem.optimizer.query_router import runtime, sparse

    base = run_base(ds)
    art, recs, tab = deployed(base)
    ids = d.test_ids(ds)
    missing = [q for q in ids if q not in recs]
    if missing:
        raise SystemExit(f"{len(missing)} test tasks without router features")
    model = art["model"]
    cfgs = [sparse._validate(c) for c in model["configs"]]
    keys = [space.key(c) for c in cfgs]
    xs = {q: runtime.transform(art["feature_spec"], recs[q]) for q in ids}
    ws = weights()
    dws = [tuple(w) for w in d.TEST_WEIGHTS.values()]
    route, pred = route_all(model, xs, ids, list(dict.fromkeys(ws + dws)))
    dep = tab["arms"]["Opt-PerQuery-default"]["routes"]
    ovh = {q: float(dep[q]["overhead_ms"]) for q in ids}
    chk = sparse.route_test_weights(
        art,
        [recs[q] for q in ids],
        [(lbl, w) for lbl, w in d.TEST_WEIGHTS.items()],
        ovh,
    )
    for lbl, w in d.TEST_WEIGHTS.items():
        g = tuple(w)
        frozen = tab["arms"][f"Opt-PerQuery-{lbl}"]["routes"]
        for j, q in enumerate(ids):
            mine = keys[route[g][j]]
            if space.key(chk[lbl][q]["config"]) != mine:
                raise SystemExit(f"{ds} {lbl} {q}: sweep routing != route_test_weights")
            if space.key(frozen[q]["config"]) != mine:
                raise SystemExit(f"{ds} {lbl} {q}: sweep routing != test/routing.json")
            if not math.isclose(frozen[q]["overhead_ms"], ovh[q], abs_tol=1e-6):
                raise SystemExit(
                    f"{ds} {lbl} {q}: overhead differs between deployed arms"
                )
    R = {
        "protocol": PROTOCOL,
        "dataset": ds,
        "artifact": {
            "path": json.loads((base / "selection.json").read_text())["weight_slices"][
                "path"
            ],
            "sha": runtime.artifact_sha(art),
        },
        "weight_set": WEIGHT_SET,
        "curve_step": CURVE_STEP,
        "ids": ids,
        "configs": cfgs,
        "weights": [list(w) for w in ws],
        "routes": {arm(w): route[w] for w in ws},
        "pred": {arm(w): pred[w] for w in ws},
        "overhead_ms": ovh,
        "deployed_routes": {lbl: route[tuple(w)] for lbl, w in d.TEST_WEIGHTS.items()},
        "deployed_check": "the four TEST_WEIGHTS routings == sparse.route_test_weights == test/routing.json",
    }
    return base, R


def frozen_routing(base: Path, R: dict, write: bool) -> dict:
    """The routing (written once per weight set; a later call with the same set must describe the same tasks /
    configurations / routes). A routing.json of ANOTHER weight set (e.g. the dropped 231-point grid, which has no
    ``weight_set``) is archived to ``routing.<set>.<ts>.json`` on write and ignored otherwise."""
    p = out_dir(base) / "routing.json"
    if p.exists():
        old = json.loads(p.read_text())
        if old.get("weight_set", "grid231") != R["weight_set"]:
            if write:
                ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                p.rename(
                    p.with_name(f"routing.{old.get('weight_set', 'grid231')}.{ts}.json")
                )
                save(p, {**R, "frozen_at": datetime.now(UTC).isoformat()})
            return R
        ident = lambda x: (x["ids"], x["configs"], x["routes"])  # noqa: E731
        if ident(old) != ident(R):
            raise SystemExit(f"{p} describes other routes; refusing to mix")
        return old
    if write:
        save(p, {**R, "frozen_at": datetime.now(UTC).isoformat()})
    return R


def deployed_tab(R: dict) -> dict:
    """The four deployed TEST_WEIGHTS as a routing table (cross-check arms ``Opt-PerQuery-<label>``). Pure."""
    from trimem.experiments import bo_pool_driver as d

    return {
        "arms": {
            f"Opt-PerQuery-{lbl}": {
                "kind": "perquery",
                "weights": list(w),
                "routes": {
                    q: {"config": R["configs"][i], "overhead_ms": R["overhead_ms"][q]}
                    for q, i in zip(R["ids"], R["deployed_routes"][lbl], strict=True)
                },
            }
            for lbl, w in d.TEST_WEIGHTS.items()
        }
    }


def to_tab(R: dict) -> dict:
    """Compact routing -> the standard routing table (arm -> task -> {config, overhead_ms}). Pure."""
    arms = {}
    for w in R["weights"]:
        name = arm(tuple(w))
        arms[name] = {
            "kind": "perquery",
            "weights": list(w),
            "routes": {
                q: {"config": R["configs"][i], "overhead_ms": R["overhead_ms"][q]}
                for q, i in zip(R["ids"], R["routes"][name], strict=True)
            },
        }
    return {"protocol": PROTOCOL, "arms": arms}


def groups(R: dict) -> dict[str, int]:
    """arm -> routing-group id (weights with IDENTICAL per-task routings share an id; numbered by first weight in
    grid order). Pure."""
    seen: dict[tuple, int] = {}
    out = {}
    for w in R["weights"]:
        r = tuple(R["routes"][arm(tuple(w))])
        out[arm(tuple(w))] = seen.setdefault(r, len(seen))
    return out


# ------------------------------------------------------------------ pair reuse


def sources(ds: str) -> list[dict]:
    """Opt-row sources measured under the same runtime, in precedence order."""
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import alf_test_v7 as v7
    from trimem.experiments import fixed_plans_test as fpt

    out = [
        {
            "name": "v7_test",
            "tag": v7.ARM_TAG,
            "subdir": v7.SUBDIR,
            "semfilter_ok": fpt.semfilter_ok_label(ds),
        }
    ]
    if ds == "alfworld":
        out.append(
            {
                "name": "c16_test",
                "tag": c16.ARM_TAG,
                "subdir": c16.SUBDIR,
                "semfilter_ok": None,
            }
        )
    out.append(
        {
            "name": "fixed_plans",
            "tag": fpt.ARM_TAG,
            "subdir": fpt.SUBDIR,
            "semfilter_ok": fpt.semfilter_ok_label(ds),
        }
    )
    if WEIGHT_SET in ("curves-log2-v1", "quality-first-v1"):
        for name, tag, subdir in (
            ("prior_weight_sweep", "pairs-c16-v7-sweep", "test_v7c16/pq_sweep"),
            (
                "initial_log_sweep",
                "pairs-c16-v7-sweep-log2",
                "test_v7c16/pq_sweep_log2_initial",
            ),
        ):
            plan_path = run_base(ds) / subdir / "plan.json"
            if plan_path.exists():
                out.append(
                    {
                        "name": name,
                        "tag": tag,
                        "subdir": subdir,
                        "semfilter_ok": fpt.semfilter_ok_label(ds),
                        "excluded_configs": sorted(
                            unsafe_small_configs(json.loads(plan_path.read_text()))
                        ),
                    }
                )
    if WEIGHT_SET == "quality-first-v1":
        out.append({
            "name": "completed_log2_sweep", "tag": "pairs-c16-v7-sweep-log2-c16",
            "subdir": "test_v7c16/pq_sweep_log2",
            "semfilter_ok": fpt.semfilter_ok_label(ds),
        })
    return out


def unsafe_small_configs(plan: dict) -> set[str]:
    """Legacy scheduler gave every nonempty half >=1 worker, even above c16."""
    small = [r for r in plan["per_config"] if 0 < r["to_run"] < CONCURRENCY]
    return (
        {r["config"] for r in small}
        if sum(min(2, r["to_run"]) for r in small) > CONCURRENCY
        else set()
    )


def nomemory_sources(ds: str) -> list[tuple[str, str]]:
    """(name, subdir) of the c16 official-evaluator NoMemory pair dirs, in precedence order."""
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import alf_test_v7 as v7

    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    return (
        [("v7_test", v7.SUBDIR)]
        + ([("c16_test", c16.SUBDIR)] if ds == "alfworld" else [])
        + [
            (s["name"], s["subdir"])
            for s in sources(ds)
            if s["name"] in ("prior_weight_sweep", "initial_log_sweep", "completed_log2_sweep")
            and d.cfg_name(space.NOMEM) not in s.get("excluded_configs", [])
        ]
    )


def reusable(cfg: dict, label: str | None, src: dict) -> bool:
    from trimem.experiments import fixed_plans_test as fpt
    from trimem.experiments import bo_pool_driver as d

    return d.cfg_name(cfg) not in src.get("excluded_configs", []) and fpt.reusable(
        cfg, label, src
    )


def old_pairs(
    ds: str, base: Path, want: dict[str, list[str]]
) -> dict[tuple[str, str], str]:
    """(config key, task) -> the source that measured it (first source wins), wanted pairs only. NoMemory: pair
    files with an episodes.jsonl cost record (c16 wall)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import fixed_plans_test as fpt
    from trimem.optimizer import session_space_agentenv_v2 as space

    pairs = {(k, q) for k, qs in want.items() for q in qs}
    out: dict[tuple[str, str], str] = {}
    for src in sources(ds):
        for r in fpt.opt_rows(d.test_experiment(ds, src["tag"]), d.test_max_round(ds)):
            if not r["cfg"]:
                continue
            k = space.key(r["cfg"])
            if (k, r["query_id"]) in pairs and (k, r["query_id"]) not in out:
                if reusable(r["cfg"], r["semfilter"], src):
                    out[(k, r["query_id"])] = src["name"]
    nk = space.key(space.NOMEM)
    nm_q = want.get(nk, [])
    for name, sub in nomemory_sources(ds):
        todo = [q for q in nm_q if (nk, q) not in out]
        p = base / sub / "pairs" / d.cfg_name(space.NOMEM)
        if not todo or not p.exists():
            continue
        for q, r in d.nomemory_rows(
            ds, p, todo, max_round=d.test_max_round(ds)
        ).items():
            if r.get("cost_source") == "episodes":
                out[(nk, q)] = name
    return out


def split_pairs(
    want: dict[str, list[str]], old: dict[tuple[str, str], str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    from trimem.experiments import fixed_plans_test as fpt

    return fpt.split_pairs(want, old)


def done_here(base: Path, ds: str, run: dict[str, list[str]]) -> dict[str, int]:
    from trimem.experiments import bo_pool_driver as d

    out = {}
    for k, qs in run.items():
        have = d.drv.item_files(out_dir(base) / "pairs" / d.cfg_name(json.loads(k)), ds)
        out[k] = len(set(qs) & set(have))
    return out


# ------------------------------------------------------------------ plan


def wall_means(ds: str, base: Path) -> dict[str, float]:
    """Mean episode wall (s) per plan over the reuse sources' opt rows + this sweep's rows; NoMemory = the v7
    c16 NoMemory pair dir (episodes.jsonl)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import fixed_plans_test as fpt
    from trimem.optimizer import session_space_agentenv_v2 as space

    by: dict[str, list[float]] = {}
    for src in sources(ds) + [{"tag": ARM_TAG, "semfilter_ok": "*"}]:
        for r in fpt.opt_rows(d.test_experiment(ds, src["tag"])):
            if (
                r["cfg"]
                and r["e2e_ms"] is not None
                and (
                    src.get("semfilter_ok") == "*"
                    or reusable(r["cfg"], r["semfilter"], src)
                )
            ):
                by.setdefault(r["cfg"]["plan"], []).append(float(r["e2e_ms"]) / 1000)
    for _, sub in nomemory_sources(ds):
        p = base / sub / "pairs" / d.cfg_name(space.NOMEM)
        if p.exists():
            rows = d.nomemory_rows(ds, p, sorted(d.drv.item_files(p, ds)))
            by.setdefault("NoMemory", []).extend(
                r["e2e_ms"] / 1000
                for r in rows.values()
                if r.get("cost_source") == "episodes"
            )
    return {p: stats.mean(v) for p, v in by.items() if v}


def plan(ds: str) -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import fixed_plans_test as fpt
    from trimem.optimizer import session_space_agentenv_v2 as space

    base, R = build(ds)
    tab = to_tab(R)
    want = d.pairs_of(tab)
    old = old_pairs(ds, base, want)
    reuse, run = split_pairs(want, old)
    done = done_here(base, ds, run)
    walls = wall_means(ds, base)
    by_src: dict[str, int] = {}
    for s in old.values():
        by_src[s] = by_src.get(s, 0) + 1
    per_cfg = []
    rem_by_plan: dict[str, int] = {}
    for k in sorted(run, key=lambda k: -len(run[k])):
        cfg = json.loads(k)
        rem = len(run[k]) - done.get(k, 0)
        rem_by_plan[cfg["plan"]] = rem_by_plan.get(cfg["plan"], 0) + rem
        per_cfg.append(
            {
                "config": d.cfg_name(cfg),
                "plan": cfg["plan"],
                "to_run": len(run[k]),
                "done_here": done.get(k, 0),
                "remaining": rem,
            }
        )
    nk = space.key(space.NOMEM)
    measured = {(k, q) for k, qs in reuse.items() for q in qs}
    full = sum(
        all(
            (space.key(R["configs"][i]), q) in measured
            for q, i in zip(R["ids"], R["routes"][arm(tuple(w))], strict=True)
        )
        for w in R["weights"]
    )
    g = groups(R)
    to_run = sum(map(len, run.values()))
    remaining = to_run - sum(done.values())
    obs = fpt.observed_rate(base, ds)
    return {
        "dataset": ds,
        "protocol": PROTOCOL,
        "run": str(base),
        "out": str(out_dir(base)),
        "experiment": d.test_experiment(ds, ARM_TAG),
        "runtime": RUNTIME,
        "concurrency": CONCURRENCY,
        "quality_reuse": False,
        "weights": len(R["weights"]),
        "distinct_routings": len(set(g.values())),
        "configs_routed": len(want),
        "weights_fully_measured_before": full,
        "pairs": sum(map(len, want.values())),
        "reused": len(old),
        "reused_by_source": by_src,
        "to_run": to_run,
        "to_run_semfilter": sum(
            len(v)
            for k, v in run.items()
            if json.loads(k)["plan"] in space.SEMFILTER_PLANS
        ),
        "to_run_nomemory": len(run.get(nk, [])),
        "done_here": sum(done.values()),
        "remaining": remaining,
        "per_config": per_cfg,
        "hours_estimate": fpt.estimate_hours(rem_by_plan, walls),
        "hours_estimate_layout": layout_hours(per_cfg, walls),
        "mean_episode_wall_s": {k: round(v, 1) for k, v in walls.items()},
        "hours_estimate_observed_rate": None
        if obs is None
        else round(remaining / obs["per_hour"], 2),
        "observed_v7_test_rate": obs,
        "estimate_note": "layout = per batch ceil(remaining / 16) waves x the slowest plan's mean episode wall + "
        f"{BATCH_OVERHEAD_S:.0f} s; saturated = remaining x mean wall / 16; observed = the v7 test's pairs / hour "
        "(env-server start-up per dataset not included)",
        "reuse_rule": "reused iff measured under controlled-c16-v1 with quality reuse off: v7 test, fixed-plan arms "
        "(SemFilter plans: semfilter-v2 rows only)"
        + (", c16 rerun (SemFilter-free plans only)" if ds == "alfworld" else "")
        + "; NoMemory = c16 official-evaluator pair files with an episodes.jsonl record",
    }


BATCH_OVERHEAD_S = (
    60.0  # per batch: half start-up + the slowest episode's tail beyond the mean
)


def layout_hours(per_cfg: list[dict], walls: dict[str, float]) -> float | None:
    """Estimate bounded batches using mean episode wall plus batch startup/tail."""
    from trimem.experiments.alf_test_c16 import SMALL

    if not walls:
        return None
    fb = stats.mean(walls.values())
    todo = [c for c in per_cfg if c["remaining"] > 0]
    big = [[c] for c in todo if c["to_run"] >= SMALL]
    small = [c for c in todo if c["to_run"] < SMALL]
    sec = 0.0
    max_arms = CONCURRENCY // 2
    for b in big + [small[i : i + max_arms] for i in range(0, len(small), max_arms)]:
        n = sum(c["remaining"] for c in b)
        sec += math.ceil(n / CONCURRENCY) * max(walls.get(c["plan"], fb) for c in b)
        sec += BATCH_OVERHEAD_S
    return round(sec / 3600, 2)


def summary_lines(p: dict) -> list[str]:
    obs = p.get("hours_estimate_observed_rate")
    out = [
        f"== {p['dataset']}: {p['weights']} weights ({p['distinct_routings']} distinct routings, "
        f"{p['configs_routed']} configurations routed); {p['pairs']} pairs = {p['reused']} reused "
        f"{p['reused_by_source']} + {p['to_run']} new (SemFilter {p['to_run_semfilter']}, NoMemory "
        f"{p['to_run_nomemory']}; {p['done_here']} done here, {p['remaining']} remaining); weights already fully "
        f"measured {p['weights_fully_measured_before']}/{p['weights']}",
        f"   estimate {p['hours_estimate_layout']} h (batch layout), {p['hours_estimate']} h (saturated)"
        + (f", {obs} h (observed v7-test rate)" if obs is not None else "")
        + f"; runtime {p['runtime']}, concurrency {p['concurrency']}; experiment {p['experiment']}",
    ]
    for c in p["per_config"]:
        out.append(
            f"   {c['config']:<22} {c['plan']:<9} new {c['to_run']:>3}  done {c['done_here']:>3}  "
            f"remaining {c['remaining']:>3}"
        )
    return out


# ------------------------------------------------------------------ run


def build_arms(base: Path, ds: str, run: dict[str, list[str]], semfilter: dict) -> list:
    """One subset arm per configuration (largest first); NoMemory = official evaluator (session None)."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    baseline, _ = d.test_baseline(ds, argparse.Namespace(sampling="per-config"))
    exp = d.test_experiment(ds, ARM_TAG)
    out = out_dir(base) / "pairs"
    mr = d.test_max_round(
        ds
    )  # max-round-v1 (WebShop): the profiling run's turn budget (None = 30)
    arms = []
    for k, qs in sorted(run.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
        if cfg["plan"] == "NoMemory":
            arms.append(
                d.Arm(
                    "pq-sweep NoMemory pairs",
                    out / d.cfg_name(cfg),
                    qs,
                    None,
                    max_round=mr,
                )
            )
            continue
        session = {
            **space.runner_args(cfg),
            "runner_extra": space.runner_cli(cfg),
            "split": "test",
            "experiment": exp,
            "baseline_dir": baseline,
            "panel": None,
            "reuse": False,  # quality reuse OFF: latency must be measured, never copied
            "reuse_from": [],
            "reuse_from_dirs": [],
            "protocol_extra": {
                "panel_hash": f"test:{d.TEST_VERSION}",
                "space_version": space.VERSION,
                "space_config": cfg,
                "bo_label": "test-pq-sweep-c16-v7",
                **d.eb.protocol_extra(mr),
                "round": d.PANEL,
                "test_version": d.TEST_VERSION,
                "test_concurrency": CONCURRENCY,
                "retest": PROTOCOL,
                "semfilter": semfilter.get("protocol")
                if cfg["plan"] in space.SEMFILTER_PLANS
                else None,
            },
        }
        arms.append(
            d.Arm(
                f"pq-sweep {d.cfg_name(cfg)}",
                out / d.cfg_name(cfg),
                qs,
                session,
                max_round=mr,
            )
        )
    return arms


def bounded_batches(arms: list) -> list[list]:
    from trimem.experiments import alf_test_c16 as c16

    max_arms = CONCURRENCY // 2
    return [
        batch[i : i + max_arms]
        for batch in c16.schedule(arms)
        for i in range(0, len(batch), max_arms)
    ]


def run(ds: str, dry: bool = False) -> None:
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import alf_test_v7 as v7
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import fixed_plans_test as fpt
    from trimem.experiments import memory_policy as mp

    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    for k in ("TRIMEM_SEMANTIC_ROUTING", "TRIMEM_SEMANTIC_BASES", "MAX_ENVS"):
        if os.environ.get(k):
            raise SystemExit(f"{k} is set: {RUNTIME} runs with the defaults")
    base, R = build(ds)
    blk, env = v7.semfilter_env(
        base
    )  # refuses without POOL_SEMFILTER=v2 / with other knobs
    os.environ.update(env)
    _lock = None if dry else fpt.driver_lock(ds)
    R = frozen_routing(base, R, write=not dry)
    want = d.pairs_of(to_tab(R))
    _, todo = split_pairs(want, old_pairs(ds, base, want))
    p = plan(ds)
    rec = {
        "started": datetime.now(UTC).isoformat(),
        "layout": "one configuration at a time, 2 halves x 8 processes (reference layout); configurations with < "
        f"{c16.SMALL} pairs share batches of at most {CONCURRENCY // 2} configurations (workers <= {CONCURRENCY})",
        "semfilter": {"protocol": blk["protocol"], "env": env},
        "retrieval_slots": os.environ.get("TRIMEM_RETRIEVAL_SLOTS") or "default (16)",
        "dry": dry,
        **p,
    }
    if not dry:
        save(out_dir(base) / "plan.json", rec)
        with open(out_dir(base) / "plan_history.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    arms = build_arms(base, ds, todo, blk)
    batches = bounded_batches(arms)
    d.log(
        f"pq sweep {ds}: {p['to_run']} pairs to run ({p['remaining']} not yet done; {p['reused']} reused) over "
        f"{len(arms)} configurations in {len(batches)} batches at concurrency {CONCURRENCY}, quality reuse off, "
        f"semfilter-v2 env -> {out_dir(base)}"
    )
    if dry or not arms:
        return
    res: dict[str, int] = {}
    for b in batches:
        mon = fpt.monitor_cls(ds)(b)
        mon.start()
        try:
            res |= d.run_parallel(ds, b, CONCURRENCY)
        finally:
            mon.stop.set()
            mon.join(timeout=5)
    bad = [a.name for a in arms if res.get(a.name, 0) < len(a.ids)]
    if bad:
        raise SystemExit(f"pq sweep pairs incomplete: {bad}")
    d.log(f"PQ SWEEP DONE {ds}")


# ------------------------------------------------------------------ rows + report


def pair_rows(
    ds: str, base: Path, want: dict[str, list[str]]
) -> tuple[dict, dict, dict]:
    """(config key, task) -> wall row / span row (+ source) of every wanted pair: the reuse sources in precedence
    order, then this sweep's own rows; NoMemory from the c16 official-evaluator pair dirs (episodes.jsonl wall,
    llm_calls.jsonl span)."""
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import fixed_plans_test as fpt
    from trimem.optimizer import session_space_agentenv_v2 as space

    pairs = {(k, q) for k, qs in want.items() for q in qs}
    wall: dict = {}
    span: dict = {}
    src_of: dict = {}
    srcs = sources(ds) + [
        {"name": "new", "tag": ARM_TAG, "subdir": SUBDIR, "semfilter_ok": "*"}
    ]
    for src in srcs:
        pdir = base / src["subdir"] / "pairs"
        spans: dict[str, dict[str, float]] = {}
        files: dict[str, dict[str, Path]] = {}
        for r in fpt.opt_rows(d.test_experiment(ds, src["tag"]), d.test_max_round(ds)):
            if not r["cfg"]:
                continue
            k = space.key(r["cfg"])
            if (k, r["query_id"]) not in pairs or (k, r["query_id"]) in wall:
                continue
            if src["semfilter_ok"] != "*" and not reusable(
                r["cfg"], r["semfilter"], src
            ):
                continue
            name = d.cfg_name(r["cfg"])
            if name not in spans:
                spans[name] = c16.call_spans(pdir / name, ds)
                files[name] = d.drv.item_files(pdir / name, ds)
            w, s = fpt.row_pair(ds, r, spans[name], files[name].get(r["query_id"]))
            wall[(k, r["query_id"])] = w
            if s is not None:
                span[(k, r["query_id"])] = s
            src_of[(k, r["query_id"])] = src["name"]
    nk = space.key(space.NOMEM)
    nm_q = want.get(nk, [])
    for name, sub in nomemory_sources(ds) + [("new", SUBDIR)]:
        todo = [q for q in nm_q if (nk, q) not in wall]
        p = base / sub / "pairs" / d.cfg_name(space.NOMEM)
        if not todo or not p.exists():
            continue
        sp = c16.call_spans(p, ds)
        for q, r in d.nomemory_rows(
            ds, p, todo, max_round=d.test_max_round(ds)
        ).items():
            if r.get("cost_source") != "episodes":
                continue
            row = {k: v for k, v in r.items() if k != "cost_source"}
            row |= {"tsem": 0.0, "ret_ms": 0.0}
            wall[(nk, q)] = row
            if q in sp:
                span[(nk, q)] = {
                    **c16.with_span(row, sp[q], 0.0),
                    "e2e_wall_ms": row["e2e_ms"],
                }
            src_of[(nk, q)] = name
    return wall, span, src_of


def plan_shares(R: dict, name: str) -> dict[str, float]:
    """Routing share per plan of one weight (pure)."""
    n = len(R["ids"])
    out: dict[str, float] = {}
    for i in R["routes"][name]:
        p = R["configs"][i]["plan"]
        out[p] = out.get(p, 0.0) + 1 / n
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def pareto(points: list[dict], x: str) -> list[str]:
    """Arms on the test Pareto front (max quality, min ``x``), complete points only. Pure."""
    pts = [p for p in points if p.get("complete")]
    return [
        p["arm"]
        for p in pts
        if not any(
            o is not p
            and o["quality"] >= p["quality"]
            and o[x] <= p[x]
            and (o["quality"] > p["quality"] or o[x] < p[x])
            for o in pts
        )
    ]


def deployed_check(ds: str, base: Path, rows: dict[str, dict]) -> dict:
    """The four deployed weights, routed here and assembled from the same pair rows (``rows``: label ->
    assembled arm), vs the v7 test report's Opt-PerQuery arms (must agree: same routes, same pairs)."""
    from trimem.experiments import alf_test_v7 as v7
    from trimem.experiments import bo_pool_driver as d

    p = base / v7.REPORT
    if not p.exists():
        return {"status": "no v7 report"}
    arms = json.loads(p.read_text()).get("arms") or {}
    ks = ("quality", "tokens", "latency_s", "latency_span_s", "steps")
    out = {}
    for lbl, w in d.TEST_WEIGHTS.items():
        t = arms.get(f"Opt-PerQuery-{lbl}")
        mine = rows.get(lbl)
        if not t or not mine or mine.get("quality") is None:
            out[lbl] = None
            continue
        out[lbl] = {
            "weights": list(w),
            "sweep": {k: mine.get(k) for k in ks},
            "v7_report": {k: t.get(k) for k in ks},
            "agree": all(
                mine.get(k) is not None
                and t.get(k) is not None
                and math.isclose(mine[k], t[k], rel_tol=TOL, abs_tol=TOL)
                for k in ks
            ),
        }
    return out


def report(ds: str) -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import fixed_plans_test as fpt

    base, R = build(ds)
    R = frozen_routing(base, R, write=False)
    tab = to_tab(R)
    want = d.pairs_of(tab)
    dtab = deployed_tab(R)
    both = d.pairs_of({"arms": {**tab["arms"], **dtab["arms"]}})
    ctrl, nm = fpt.nomemory_quality(ds)
    wall, span, src_of = pair_rows(ds, base, both)
    aw, asp = d.assemble(tab, wall, nm), d.assemble(tab, span, nm)
    dw, dsp = d.assemble(dtab, wall, nm), d.assemble(dtab, span, nm)
    dep = {
        n.removeprefix("Opt-PerQuery-"): {
            **v,
            "latency_span_s": dsp[n].get("latency_s"),
        }
        for n, v in dw.items()
    }
    g = groups(R)
    key = d.space.key
    rows: dict[str, dict] = {}
    for w in R["weights"]:
        name = arm(tuple(w))
        a, s = aw[name], asp[name]
        by: dict[str, int] = {}
        for q, r in tab["arms"][name]["routes"].items():
            x = src_of.get((key(r["config"]), q))
            if x:
                by[x] = by.get(x, 0) + 1
        pq, pt, pl = R["pred"][name]
        rows[name] = {
            "arm": name,
            "weights": list(w),
            "group": g[name],
            "n": a["n"],
            "covered": a["covered"],
            "covered_span": s["covered"],
            "complete": a["covered"] == a["n"] and s["covered"] == s["n"],
            "quality": a.get("quality"),
            "quality_se": a.get("quality_se"),
            "paired_gain_vs_nomemory": a.get("paired_gain_vs_nomemory"),
            "paired_gain_se": a.get("paired_gain_se"),
            "tokens": a.get("tokens"),
            "latency_s": a.get("latency_s"),
            "latency_span_s": s.get("latency_s"),
            "steps": a.get("steps"),
            "steps_se": a.get("steps_se"),
            "steps_success": a.get("steps_success"),
            "distinct_configs": a["distinct_configs"],
            "plan_shares": plan_shares(R, name),
            "pair_sources": by,
            "pred": {"quality": pq, "tokens": pt, "latency_s": pl / 1000},
        }
    pts = list(rows.values())
    by_group: dict[int, list[str]] = {}
    for n, r in rows.items():
        by_group.setdefault(r["group"], []).append(n)
    ws = [tuple(w) for w in R["weights"]]
    pl = out_dir(base) / "plan.json"
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL,
        "dataset": ds,
        "experiment": d.test_experiment(ds, ARM_TAG),
        "runtime_profile": d.eb.runtime_label(fpt.state_of(base), RUNTIME) or RUNTIME,
        "concurrency": CONCURRENCY,
        "quality_reuse": False,
        "quality": QNAME[ds],
        "artifact": R["artifact"],
        "weight_set": R["weight_set"],
        "curve_step": R["curve_step"],
        "plan": json.loads(pl.read_text()) if pl.exists() else None,
        "latency_definitions": {
            "latency_s": "wall: episode wall time + retrieval ms (opt e2e_ms; NoMemory episodes.jsonl) + router overhead",
            "latency_span_s": "span: first .. last LLM call of the episode + retrieval ms + router overhead (the "
            "page's plotted definition, like-for-like with the controlled-v1 NoMemory reference)",
            "pred": "router-model means over the 200 routed (config, task) predictions; predicted latency = the "
            "profiling e2e wall model, without router overhead",
        },
        "baseline": {
            "dir": str(ctrl),
            "label": "controlled-v1 NoMemory (paired baseline of gain)",
            "quality": sum(nm.values()) / max(len(nm), 1),
        },
        "pairs": sum(map(len, want.values())),
        "pairs_measured": sum((k, q) in wall for k, qs in want.items() for q in qs),
        "pairs_span": sum((k, q) in span for k, qs in want.items() for q in qs),
        "pair_sources": {
            s: sum(1 for k, qs in want.items() for q in qs if src_of.get((k, q)) == s)
            for s in sorted(set(src_of.values()))
        },
        "weights": len(rows),
        "complete_weights": sum(r["complete"] for r in pts),
        "complete": all(r["complete"] for r in pts),
        "distinct_routings": len(by_group),
        "identical_routings": [v for v in by_group.values() if len(v) > 1],
        "slices": {
            "QT": [arm(w) for w in qt_slice(ws)],
            "QL": [arm(w) for w in ql_slice(ws)],
        },
        "pareto": {
            "tokens": pareto(pts, "tokens"),
            "latency_span": pareto(pts, "latency_span_s"),
            "latency_wall": pareto(pts, "latency_s"),
        },
        "deployed_check": deployed_check(ds, base, dep),
        "points": rows,
    }
    save(out_dir(base) / REPORT, rep)
    (out_dir(base) / REPORT_MD).write_text("\n".join(markdown(rep)) + "\n")
    return rep


def _f(x, fmt: str) -> str:
    return "–" if x is None else format(x, fmt)


def _row(r: dict) -> str:
    w = "/".join(f"{v:g}" for v in r["weights"])
    sh = ", ".join(f"{p} {100 * v:.1f}%" for p, v in r["plan_shares"].items())
    pr = r["pred"]
    if not r["complete"]:
        meas = "pending | | | | | "
    else:
        meas = (
            f"{r['quality']:.3f} ± {_f(r['quality_se'], '.3f')} | "
            f"{_f(r['paired_gain_vs_nomemory'], '+.3f')} ± {_f(r['paired_gain_se'], '.3f')} | "
            f"{r['tokens']:,.0f} | {r['latency_s']:.1f} | {_f(r['latency_span_s'], '.1f')} | "
            f"{_f(r['steps'], '.2f')}"
        )
    return (
        f"| {w} | {r['group']} | {r['covered']}/{r['n']} | {meas} | {r['distinct_configs']} | {sh} | "
        f"{pr['quality']:.3f} / {pr['tokens']:,.0f} / {pr['latency_s']:.1f} |"
    )


def markdown(rep: dict) -> list[str]:
    head = (
        "| w_Q/w_L/w_T | routing group | covered | quality ± SE | gain vs NoMemory ± SE | tokens | latency wall s "
        "| latency span s | steps | configs | routing shares (plan) | predicted q / tok / wall s |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"
    P = rep["points"]
    L = [
        f"# {rep['dataset']} · Opt-PerQuery (BRANE) weight sweep on the test set ({rep['runtime_profile']})",
        "",
        f"Generated {rep['generated']}. Deployed router artifact {rep['artifact']['sha'][:12]} (no refit), weight set "
        f"{rep['weight_set']} (Q-T + Q-L curves) = {rep['weights']} weights, {rep['distinct_routings']} distinct "
        f"routings. Quality = {rep['quality']}; {rep['pairs_measured']}/{rep['pairs']} (config, task) pairs measured "
        f"(sources {rep['pair_sources']}); quality reuse OFF. Complete weights: {rep['complete_weights']}/"
        f"{rep['weights']}. Gain = paired vs the controlled-v1 NoMemory "
        f"({rep['baseline']['quality']:.3f}).",
        "",
        "Deployed-weight cross-check vs the v7 test report: "
        + "; ".join(
            f"{k} {'OK' if (v or {}).get('agree') else 'n/a' if v is None else 'DIFFERS'}"
            for k, v in (rep["deployed_check"] or {}).items()
            if isinstance(v, (dict, type(None)))
        ),
        "",
        f"Test Pareto front (quality × tokens): {', '.join(rep['pareto']['tokens']) or '–'}",
        f"Test Pareto front (quality × span latency): {', '.join(rep['pareto']['latency_span']) or '–'}",
        "",
        "## Q–T curve (w_L = 0, by w_T)",
        "",
        head,
        sep,
        *[_row(P[n]) for n in rep["slices"]["QT"]],
        "",
        "## Q–L curve (w_T = 0, by w_L)",
        "",
        head,
        sep,
        *[_row(P[n]) for n in rep["slices"]["QL"]],
        "",
        "## Identical routings (weights sharing one per-task routing)",
        "",
        *[
            f"- group {P[g[0]]['group']}: "
            + ", ".join("/".join(f"{v:g}" for v in P[n]["weights"]) for n in g)
            for g in rep["identical_routings"]
        ],
        "",
        "## All weights",
        "",
        head,
        sep,
        *[_row(r) for r in P.values()],
    ]
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("plan", "run", "report"))
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument(
        "--weight-set",
        choices=("curves-step0.2-v1", "curves-log2-v1", "quality-first-v1"),
        default="curves-step0.2-v1",
    )
    ap.add_argument(
        "--dry", action="store_true", help="run: log only, write nothing, start nothing"
    )
    ap.add_argument(
        "--summary", action="store_true", help="plan / report: compact text"
    )
    a = ap.parse_args(argv)
    configure(a.weight_set, dataset=a.dataset)
    if a.stage == "plan":
        p = plan(a.dataset)
        print(
            "\n".join(summary_lines(p))
            if a.summary
            else json.dumps(p, indent=1, default=float)
        )
    elif a.stage == "run":
        run(a.dataset, a.dry)
    else:
        rep = report(a.dataset)
        if a.summary:
            print(
                f"{a.dataset}: complete weights {rep['complete_weights']}/{rep['weights']}, pairs "
                f"{rep['pairs_measured']}/{rep['pairs']}, deployed check {rep['deployed_check']}"
            )
        else:
            print("\n".join(markdown(rep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
