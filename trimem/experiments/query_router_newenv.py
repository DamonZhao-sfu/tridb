"""Per-query router (BRANE) for AppWorld and τ²-bench: the ``query_router_agentenv`` stages on the
profiling outcome matrix of ``bo_newenv_driver`` (plan: docs/trimem_three_dataset_plan_v0.1.0.md §6).

Stages (``.venv-optimizer``; none reads split=test outcomes):

    features   raw query records of the profiling panel (and, with --include_test, of the test tasks:
               features only, no outcomes) -> results/query_router_20261016/<ds>/
               Task rows and task vectors come from the dataset's own store (trimem_appworld /
               trimem_tau2, read-only); the probe runs through the dataset's retrieval adapter
               (``features_newenv``). τ²'s query text is the customer's first message (store task
               ``description``), the retrieval query of the memory agent.
    cv         outer K-fold CV of a router on the panel's dense config x query matrix
    train      router artifact (JSON) fitted on all profiling queries
    route-test the routed arm on the test split with the dataset's router runner
               (``appworld.router_run`` / ``tau2.router_runner``) and the driver's arm machinery;
               same GPU / concurrency as the other test arms

    PYTHONPATH=. .venv-optimizer/bin/python -m trimem.experiments.query_router_newenv features --dataset appworld --include_test
    PYTHONPATH=. .venv-optimizer/bin/python -m trimem.experiments.query_router_newenv cv --dataset appworld --keep_per_plan
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import bo_newenv_datasets as dsets
from trimem.experiments import memory_policy as mp
from trimem.experiments import query_router_agentenv as qra

ROOT = dsets.ROOT
OUT = qra.OUT  # results/query_router_20261016 (same root: export + summaries)


def router_dir(name: str) -> Path:
    """= ``query_router_agentenv.router_dir``: features / CV / routers per memory policy."""
    return mp.tag_dir(OUT / name)


OPT_DSN = dsets.OPT_DSN
log = qra.log

TASK_SQL = {
    "appworld": "SELECT id, key, split, scenario, held_out, description, emb::text AS emb FROM task WHERE key = ANY(%s)",
    "tau2": (
        "SELECT id, key, env, task_id, split, family, entities, held_out, description, emb::text AS emb "
        "FROM task WHERE key = ANY(%s)"
    ),
}


def store_dsn(ds: str) -> str:
    if ds == "appworld":
        from trimem.experiments.appworld.adapter import DSN
    else:
        from trimem.experiments.tau2.adapter import DSN
    return DSN


def task_rows(ds: str, keys: list[str], fetch=None) -> dict[str, dict]:
    """key -> store task row (with ``emb`` as text). ``fetch(sql, keys)`` is injectable for tests."""
    if fetch is None:
        import psycopg
        from psycopg.rows import dict_row

        def fetch(sql, ks):
            with psycopg.connect(store_dsn(ds), row_factory=dict_row) as c:
                c.execute("SET TRANSACTION READ ONLY")
                return c.execute(sql, (ks,)).fetchall()

    rows = {r["key"]: r for r in fetch(TASK_SQL[ds], list(keys))}
    missing = [k for k in keys if k not in rows or rows[k].get("emb") is None]
    if missing:
        raise SystemExit(f"no store task row / vector for {missing[:5]}")
    return rows


def records_for(
    ds: str,
    keys: list[str],
    rows: dict[str, dict],
    *,
    use_probe: bool,
    llm=None,
    questions: list[str] | None = None,
    adapter_factory=None,
) -> list[dict]:
    from trimem.optimizer.query_router import features_newenv as fn

    out = []
    for k in keys:
        row = dict(rows[k])
        vec = row.pop("emb")
        out.append(
            fn.raw_record(
                ds,
                k,
                fn.task_fields(ds, row),
                fn.query_identity(ds, row),
                vec,
                use_probe=use_probe,
                llm=llm,
                questions=questions,
                adapter_factory=adapter_factory,
            )
        )
    return out


def stage_features(a) -> None:
    from trimem.optimizer.query_router import features

    d = dsets.get(a.dataset)
    out = router_dir(d.name)
    out.mkdir(parents=True, exist_ok=True)
    pnl = d.panel()
    train_keys = [x["item_id"] for x in pnl["items"]]
    test_keys = d.test_ids() if a.include_test else []
    rows = task_rows(d.name, train_keys + test_keys)
    for k in train_keys:
        if rows[k]["held_out"] and d.name == "tau2":
            raise SystemExit(f"{k}: a τ² profiling task is held out in the store")
    llm, questions, qmeta = None, None, None
    if a.llm != "none":
        llm = qra._llm(a.llm, a.llm_base)
        qp = out / "llm_questions.json"
        if qp.exists():
            questions = json.load(open(qp))["questions"]
        else:  # proposed from PROFILING queries only
            questions, qmeta = features.propose_questions(
                llm, [rows[k]["description"] or "" for k in train_keys], a.questions
            )
            qp.write_text(
                json.dumps(
                    {
                        "questions": questions,
                        "llm": a.llm,
                        "usage": qmeta,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                    indent=1,
                )
            )
        log(f"{len(questions)} LLM questions")
    for name, keys in (("train", train_keys), ("test", test_keys)):
        if not keys:
            continue
        t0 = time.monotonic()
        recs = records_for(
            d.name, keys, rows, use_probe=not a.no_probe, llm=llm, questions=questions
        )
        meta = {
            "dataset": d.name,
            "split": name,
            "panel_hash": pnl["panel_hash"] if name == "train" else None,
            "probe_seeds": features.PROBE_SEEDS,
            "llm": a.llm,
            "llm_questions": questions,
            "llm_labeler": qra._labeler_identity(llm),
            "feature_source": "features_newenv",
            "created_at": datetime.now(UTC).isoformat(),
            "wall_s": time.monotonic() - t0,
        }
        (out / f"features_{name}.json").write_text(
            json.dumps({"meta": meta, "records": recs})
        )
        log(
            f"features_{name}: {len(recs)} records, mean overhead "
            f"{sum(r['overhead']['probe_ms'] + r['overhead']['llm_ms'] for r in recs) / len(recs):.1f} ms"
        )


def nomemory_column(d: dsets.Dataset, query_ids: tuple[str, ...]):
    """No-Memory quality / tokens / latency per query from the profiling No-Memory result files."""
    import numpy as np

    files = d.result_files(d.arm_dir("NoMemory"))
    missing = [q for q in query_ids if q not in files]
    if missing:
        raise SystemExit(f"No-Memory result missing for {missing[:5]}")
    recs = [d.record(files[q]) for q in query_ids]
    return {
        k: np.array([r[k] for r in recs], dtype=float)
        for k in ("quality", "tokens", "latency_ms")
    }


def load_training(ds: str, with_nomemory: bool, rows: list[dict] | None = None):
    from trimem.optimizer.query_router import matrix

    d = dsets.get(ds)
    pnl = d.panel()
    ids = [x["item_id"] for x in pnl["items"]]
    m = matrix.build(
        matrix.fetch_rows(OPT_DSN, d.experiment_id()) if rows is None else rows,
        panel_ids=ids,
        experiment=d.experiment_id(),
    )
    if with_nomemory:
        m = matrix.with_nomemory(m, nomemory_column(d, m.query_ids))
    f = json.load(open(router_dir(ds) / "features_train.json"))
    if f["meta"]["panel_hash"] != pnl["panel_hash"]:
        raise SystemExit("features were built for a different panel")
    by_id = {r["query_id"]: r for r in f["records"]}
    return m, [by_id[q] for q in m.query_ids], f["meta"], pnl


def stage_cv(a) -> None:
    from trimem.optimizer.query_router import evaluate

    d = dsets.get(a.dataset)
    m, recs, meta, _ = load_training(d.name, a.nomemory)
    fc = qra.feature_config(a, meta)
    log(
        f"matrix {len(m.query_ids)} queries x {len(m.configs)} configurations; groups {fc.groups}; model {a.model}; "
        f"{a.repeats} x {a.folds}-fold"
    )
    runs = [
        evaluate.cross_validate(
            m,
            recs,
            qra.fit_fn(a),
            fc,
            k=a.folds,
            inner_k=a.inner_folds,
            seed=a.seed + i,
            rule=a.rule,
        )
        for i in range(a.repeats)
    ]
    res = {
        **evaluate.repeated(runs, m),
        "runs": runs,
        "model": a.model,
        "groups": list(fc.groups),
        "matrix": {
            "configs": len(m.configs),
            "queries": len(m.query_ids),
            "nomemory_column": a.nomemory,
            "experiment": d.experiment_id(),
        },
    }
    p = qra.cv_path(a, fc)
    p.write_text(json.dumps(res, indent=1))
    for arm in (
        "router@selected",
        "single@selected",
        "single@qbest",
        "oracle",
        "random_candidate",
    ):
        s = res["arms"][arm]
        log(
            f"{arm:18s} quality {s['quality']:.3f} gain {s['gain']:+.3f} "
            f"tokens {s['tokens']:,.0f} latency {s['latency_ms'] / 1000:.1f}s"
        )
    diff = res["selected_vs_qbest"]
    log(
        f"router - Q-best single: {diff['mean']:+.3f} ± {diff['se']:.3f}; gate = {res['gate']}; written {p}"
    )


def stage_train(a) -> None:
    from trimem.optimizer.query_router import evaluate, matrix, runtime

    if a.nomemory:
        raise SystemExit(
            "deployment routers must not contain NoMemory; train without --nomemory"
        )
    d = dsets.get(a.dataset)
    m, recs, meta, pnl = load_training(d.name, False)
    fc = qra.feature_config(a, meta)
    cvp = qra.cv_path(a, fc)
    if cvp.exists():
        cv = json.load(open(cvp))
        if cv["matrix_fingerprint"] != matrix.fingerprint(m):
            raise SystemExit("the outcome matrix changed since the CV report; rerun cv")
    elif a.force_route:
        cv = {"gate": None}
    else:
        raise SystemExit(
            f"run cv first ({cvp} missing): the CV gate decides route vs single"
        )
    policy = "route" if a.force_route else cv["gate"]
    art = evaluate.train(
        m,
        recs,
        qra.fit_fn(a),
        fc,
        dataset=d.name,
        policy=policy,
        inner_k=a.inner_folds,
        seed=a.seed,
        rule=a.rule,
        provenance={
            "experiment": d.experiment_id(),
            "panel_hash": pnl["panel_hash"],
            "configs_in_matrix": len(m.configs),
            "cv_report": str(cvp.relative_to(ROOT)) if cvp.exists() else None,
            "cv_gate": cv["gate"],
            "forced_route": bool(a.force_route),
            "feature_source": "features_newenv",
            "trained_at": datetime.now(UTC).isoformat(),
        },
    )
    p = (
        router_dir(d.name)
        / f"router_{'brane-paper' if a.model == 'brane' else a.model}{a.tag}.json"
    )
    sha = runtime.save(p, art)
    log(
        f"router {p} sha {sha}: policy {policy} (CV gate {cv['gate']}), lambda {art['routing']['lambda']}, "
        f"{len(art['model']['configs'])} candidates, fallback {art['policy']['fallback']}"
    )


def route_session(d: dsets.Dataset, router_path: Path, router) -> tuple[str, dict]:
    variant = router.a["model"].get("variant")
    kind = router.a["model"]["kind"] + (
        f"_{variant}" if variant and variant != "linear" else ""
    )
    name = f"Router-{kind}-{router.sha}"  # one artifact = one run identity
    return name, {
        "router": str(Path(router_path).resolve()),
        "split": "test",
        "experiment": d.test_experiment(name),
        "baseline_dir": d.test_nomemory,
        "protocol_extra": {"bo_label": name, "router_sha": router.sha},
    }


def stage_route_test(a) -> None:
    from trimem.experiments import bo_newenv_driver as drv
    from trimem.optimizer.query_router import runtime

    d = dsets.get(a.dataset)
    router = runtime.Router.load(a.router)
    if router.a["dataset"] != d.name:
        raise SystemExit("router trained for another dataset")
    if (
        router.needs_llm
        and (router.llm_labeler or {}).get("kind") != "openai-compatible"
    ):
        raise SystemExit(
            "router's LLM features were labeled by a backend deployment cannot reproduce"
        )
    held = drv.lock(d)  # noqa: F841 - never concurrently with this dataset's BO driver
    ids = d.test_ids()
    if len(set(d.result_files(d.test_nomemory)) & set(ids)) < len(ids):
        raise SystemExit(f"test No-Memory ({d.test_nomemory}) is incomplete")
    name, session = route_session(d, Path(a.router), router)
    out = d.test_dir(name)
    n = drv.run_arm(
        d,
        f"test {name}",
        out,
        ids,
        drv.parse_gpus(a.gpus) or d.test_gpus,
        stage="test",
        session=session,
        done=lambda: drv.routed_completed(d, out, ids, session),
    )
    log(
        f"ROUTE TEST {'DONE' if n == len(ids) else 'INCOMPLETE'} {d.name} {name}: {n}/{len(ids)}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("features", "cv", "train", "route-test"))
    ap.add_argument("--dataset", choices=tuple(dsets.DATASETS), required=True)
    ap.add_argument(
        "--include_test",
        action="store_true",
        help="features: also the test tasks (no outcomes)",
    )
    ap.add_argument("--no_probe", action="store_true")
    ap.add_argument("--llm", default="none", choices=("none", "qwen", "codex"))
    ap.add_argument("--llm_base", default="http://127.0.0.1:8030")
    ap.add_argument("--questions", type=int, default=8)
    ap.add_argument("--model", default="brane", choices=("brane", "brane-linear", "gp"))
    ap.add_argument(
        "--trials",
        type=int,
        default=30,
        help="brane: Optuna trials per configuration (paper: 30)",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--groups", default="structured,probe,emb")
    ap.add_argument("--emb_dims", type=int, default=4)
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument("--gp_pruned", action="store_true")
    ap.add_argument("--max_rows", type=int, default=1000)
    ap.add_argument(
        "--nomemory", action="store_true", help="cv: add NoMemory as a candidate column"
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=4)
    ap.add_argument("--force_route", action="store_true")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", default="knee", help="knee | quality | fixed:<lambda>")
    ap.add_argument("--router", default=None)
    ap.add_argument(
        "--keep_per_plan",
        action="store_true",
        default=True,
        help="brane: also keep the best configuration of every plan (relaxed pruning; default)",
    )
    ap.add_argument(
        "--strict_prune",
        dest="keep_per_plan",
        action="store_false",
        help="brane: paper pruning only (fuzzy Pareto set)",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--gpus",
        default=None,
        help="route-test: GPUs (default: the dataset's test-arm GPU)",
    )
    a = ap.parse_args(argv)
    {
        "features": stage_features,
        "cv": stage_cv,
        "train": stage_train,
        "route-test": stage_route_test,
    }[a.stage](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
