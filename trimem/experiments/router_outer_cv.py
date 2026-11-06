"""Dataset-independent glue of the outer 5-fold CV stage (``trimem.optimizer.query_router.outer_cv``).

Used by ``tau2.pool_profile`` and ``bo_pool_driver`` (stage ``cv``, between ``select`` and ``test``):

* ``run_folds``: K fold fits (cached per fold under ``<run>/router_cv/fold<k>.json``; the fold artifact is
  saved as ``router_cv/router_<sha>.json``) and ``router_cv/cv_report.json`` (held-out mean +- std);
* ``freeze_routing``: ``test/routing_cv.json`` -- every fold's Opt-OneConfig / Opt-PerQuery test routes, frozen
  once, next to (never inside) the deployed ``test/routing.json``; the test stage runs the union of both;
* ``test_summary`` / ``markdown``: the report's CV section (test set only, user 2026-10-21; the held-out fold
  evaluation stays an internal artifact in ``cv_report.json``).

Imported lazily by the drivers' cv / test / report stages only.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from trimem.optimizer.query_router import outer_cv as ocv
from trimem.optimizer.query_router import runtime, sparse

CV_DIR = "router_cv"
LIMEQO_CV_DIR = "router_cv_limeqo"
LIMEQO_PREFIX = "LimeQO-"  # Opt-PerQuery-LimeQO-<weights> arms (user 2026-10-21)
# LimeQO (inductive shared low-rank, trimem.optimizer.query_router.shared) fixed options: the defaults of
# trimem.experiments.shared_router / docs/trimem_shared_router_v0.1.0.md.
LIMEQO_OPTIONS = {"rank": 5, "reg": 0.2, "iterations": 50}
ROUTING = "routing_cv.json"


def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def fingerprint(
    prof: sparse.SparseProfile, groups: tuple[str, ...], fit_kwargs: dict
) -> str:
    """Identity of the fold fits' inputs (configurations, samples, groups, fit options): a cached fold is reused
    only under the same fingerprint."""
    import hashlib

    blob = json.dumps(
        [
            [json.dumps(c, sort_keys=True) for c in prof.configs],
            [list(s) for s in prof.samples],
            list(groups),
            sorted((k, repr(v)) for k, v in fit_kwargs.items()),
            # the replay estimator changes the folds' SLO weight choices (user 2026-10-22: selfnorm-v1)
            *(
                [sparse.REPLAY_ESTIMATOR]
                if sparse.REPLAY_ESTIMATOR != sparse.REPLAY_SUM
                else []
            ),
        ]
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def run_folds(
    out: Path,
    *,
    dataset: str,
    prof: sparse.SparseProfile,
    records: dict[str, dict],
    test_records: list[dict],
    groups: tuple[str, ...],
    oneconfig_cols: list[int],
    labels: dict[int, str],
    nomemory: dict[str, float],
    bounds_fn: Callable,
    choose_fn: Callable,
    objectives: dict[str, str],
    test_weights: dict[str, tuple[float, float, float]],
    fit_kwargs: dict,
    fallback: dict,
    trained_on: dict,
    log: Callable[[str], None],
    k: int = ocv.K,
    seed: int = ocv.FOLD_SEED,
    cv_dir: str = CV_DIR,
    backend: str = "brane",
) -> dict:
    """All K folds (cached) + ``cv_report.json``. Returns the report (fold records keep their test routes).
    ``backend`` (recorded; the router backend itself is chosen by ``fit_kwargs``, e.g. ``backend="limeqo"``) and
    ``cv_dir`` let a second router backend run the same folds into its own directory (LIMEQO_CV_DIR)."""
    d = out / cv_dir
    folds = ocv.stratified_folds(prof.queries, prof.strata, k, seed)
    block = ocv.fold_block(folds, prof.strata, k, seed)
    fp = fingerprint(prof, groups, fit_kwargs)
    recs = []
    for f in range(k):
        p = d / f"fold{f}.json"
        if p.exists():
            r = json.loads(p.read_text())
            if r.get("folds_sha") == block["sha"] and r.get("fingerprint") == fp:
                log(f"outer CV fold {f}: cached ({p})")
                recs.append(r)
                continue
        log(
            f"outer CV fold {f}: fitting on {block['sizes'][f]} held out / {len(folds) - block['sizes'][f]} train"
        )
        r = ocv.run_fold(
            f,
            prof=prof,
            folds=folds,
            records=records,
            groups=groups,
            oneconfig_cols=oneconfig_cols,
            labels=labels,
            nomemory=nomemory,
            bounds_fn=bounds_fn,
            choose_fn=choose_fn,
            objectives=objectives,
            test_weights=test_weights,
            fit_kwargs=fit_kwargs,
            test_records=test_records,
            dataset=dataset,
            fallback=fallback,
            trained_on=trained_on,
        )
        art = r.pop("artifact")
        sha = runtime.artifact_sha(art)
        ap = d / f"router_{sha}.json"
        if not ap.exists():
            runtime.save(ap, art)
        r |= {
            "artifact": {"path": str(ap), "sha": sha},
            "folds_sha": block["sha"],
            "fingerprint": fp,
        }
        _save(p, r)
        for n, v in r["oneconfig"].items():
            log(
                f"  fold {f} Opt-OneConfig-{n}: {v and v['label']} held-out q "
                f"{v and round(v['heldout']['quality'], 3)}"
            )
        for lab, v in r["perquery"].items():
            h = v["heldout"]
            log(
                f"  fold {f} Opt-PerQuery-{lab}: held-out q {h['quality']:.3f} tok {h['tokens']:,.0f} "
                f"lat {h['latency_s']:.1f}s coverage {h['coverage']:.2f}"
            )
        recs.append(r)
    per_fold = [ocv.heldout_arms(r) for r in recs]
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "protocol": ocv.PROTOCOL,
        "router_protocol": sparse.PROTOCOL,
        "router_backend": backend,
        "folds": block,
        "fingerprint": fp,
        "groups": list(groups),
        "fit": {k_: repr(v) for k_, v in fit_kwargs.items()},
        "estimator": "held-out replay with measured outcomes only (outer_cv.heldout_eval); mean +- std across folds",
        "replay_estimator": sparse.REPLAY_ESTIMATOR,
        "heldout": ocv.summarize(per_fold),
        "per_fold": [{x: r[x] for x in r if x != "test_routes"} for r in recs],
    }
    _save(d / "cv_report.json", rep)
    return {**rep, "fold_records": recs}


def freeze_routing(
    path: Path,
    *,
    rep: dict,
    test_ids: list[str],
    validate: Callable[[dict], dict],
    deployed: dict,
    key: Callable[[dict], str],
    nomemory_key: str,
    completed: Callable[[dict[str, dict]], set[tuple[str, str]]] | None = None,
    reuse_rate: float | None = None,
    extra: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """``routing_cv.json`` (frozen once; returned as is when present)."""
    if path.exists():
        return json.loads(path.read_text())
    # Several records per fold when a second router backend is merged in (``with_backend``): Opt-OneConfig is
    # router-independent and comes from the records that carry it.
    one: dict[int, dict] = {}
    for r in rep["fold_records"]:
        for n, v in r["oneconfig"].items():
            one.setdefault(r["fold"], {})[n] = (
                None
                if v is None
                else {
                    q: {"config": validate(v["cfg"]), "overhead_ms": 0.0}
                    for q in test_ids
                }
            )
    arms = ocv.test_arms(rep["fold_records"], one)
    done = completed(arms) if completed else None
    budget = ocv.pair_budget(
        arms, deployed.get("arms", {}), key, nomemory_key, done, reuse_rate
    )
    tab = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "protocol": ocv.PROTOCOL,
        "folds": rep["folds"],
        "fingerprint": rep["fingerprint"],
        **(extra or {}),
        "arms": arms,
        "pair_budget": budget,
    }
    _save(path, tab)
    log(f"CV routing frozen: {len(arms)} arms; pair budget {budget}")
    return tab


def with_backend(rep: dict, other: dict) -> dict:
    """The fold records of a second router backend (``other``: ``run_folds`` on the SAME folds with test-weight
    labels prefixed, e.g. ``LimeQO-default``) appended to ``rep``'s, its Opt-OneConfig dropped: Opt-OneConfig is
    router-independent, so both runs must have chosen the same configuration per fold and SLO (checked)."""
    if other["folds"]["sha"] != rep["folds"]["sha"]:
        raise ValueError("the second router backend ran on different outer folds")
    mine = {r["fold"]: r for r in rep["fold_records"]}
    extra = []
    for r in other["fold_records"]:
        a, b = mine[r["fold"]]["oneconfig"], r["oneconfig"]
        if {n: v and v["cfg"] for n, v in a.items()} != {
            n: v and v["cfg"] for n, v in b.items()
        }:
            raise ValueError(
                f"fold {r['fold']}: Opt-OneConfig differs between router backends"
            )
        extra.append({**r, "oneconfig": {}})
    return {**rep, "fold_records": rep["fold_records"] + extra}


def merged(deployed: dict, cv: dict | None) -> dict:
    """Deployed + CV arms (``pairs_of`` input: each distinct (config, query) pair runs once)."""
    return {**deployed, "arms": {**deployed["arms"], **(cv or {}).get("arms", {})}}


def test_summary(cv: dict, assemble: Callable[[dict], dict]) -> dict:
    """Per fold ``assemble`` (the deployed arms' metric function) of the fold's test arms, then mean +- std."""
    per = {k: assemble(t) for k, t in ocv.fold_tables(cv["arms"]).items()}
    return {
        "per_fold": {str(k): v for k, v in per.items()},
        "summary": ocv.summarize(list(per.values())),
    }


def _ms(m: dict, fmt: str) -> str:
    if m["mean"] is None:
        return "–"
    s = format(m["mean"], fmt)
    return (
        s
        + (f" ± {format(m['std'], fmt.lstrip('+'))}" if m["std"] is not None else "")
        + ("" if m["n"] == ocv.K else f" (n={m['n']})")
    )


def markdown(test: dict | None, quality: str, n_test: int) -> list[str]:
    """CV section lines (user 2026-10-21: TEST only): one row per arm, mean ± std across the folds' routers."""
    t = (test or {}).get("summary", {})
    lines = [
        f"## Outer 5-fold CV ({ocv.PROTOCOL}) — primary, test set",
        "",
        f"Each fold's router / selection is fitted on 4/5 of the training queries; every fold is evaluated on the "
        f"same {n_test} test tasks. Numbers are mean ± std ACROSS the {ocv.K} folds (std = router-training "
        "variability, not task sampling). `SE_tasks` = mean within-fold standard error across test tasks "
        "(task-sampling noise of one fold). `steps` = agent steps per episode (agent-env: environment steps = "
        "agent LLM turns; τ²: agent calls); `steps (succ.)` = over successful episodes only.",
        "",
        f"| arm | {quality} | tokens | latency s | steps | steps (succ.) | gain vs NoMemory | "
        f"SE_tasks {quality} | SE_tasks steps | SE_tasks gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    nil = {"mean": None, "std": None, "n": 0}
    for n, x in t.items():
        se = x.get("se_tasks_mean", {})
        cells = [
            _ms(x["quality"], ".3f"),
            _ms(x["tokens"], ",.0f"),
            _ms(x["latency_s"], ".1f"),
            _ms(x.get("steps", nil), ".2f"),
            _ms(x.get("steps_success", nil), ".2f"),
            _ms(x["gain"], "+.3f"),
            f"{se['quality']:.3f}" if se.get("quality") is not None else "–",
            f"{se['steps']:.2f}" if se.get("steps") is not None else "–",
            f"{se['gain']:.3f}" if se.get("gain") is not None else "–",
        ]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return lines
