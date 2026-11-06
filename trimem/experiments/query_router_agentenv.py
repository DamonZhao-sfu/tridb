"""Per-query router for ALFWorld / ScienceWorld / WebShop (plan: docs/trimem_query_router_plan_v0.1.0.md).

Stages (run with ``.venv-optimizer``; none of them reads split=test outcomes):

    features   raw query records of the training panel (and, with --include_test, of the AgentEval
               test items: features only, no outcomes) -> results/query_router_20261016/<ds>/
    cv         outer K-fold CV of a router on the training panel's dense config x query matrix
    train      router artifact (JSON) fitted on all training queries
    route-test the routed arm on AgentEval 200 under the controlled protocol (bo_agentenv_driver
               machinery; one arm, both GPU halves)

    PYTHONPATH=. .venv-optimizer/bin/python -m trimem.experiments.query_router_agentenv features --dataset alfworld --include_test
    PYTHONPATH=. .venv-optimizer/bin/python -m trimem.experiments.query_router_agentenv cv --dataset alfworld --model brane

``--panel_name train500`` (user 2026-10-17) uses the 500-query panel and its BO experiment
``<ds>:train500:bo3obj-v2``: files get a ``_train500`` suffix in the same directory (features_train_train500.json,
cv_..._train500.json, router_brane-paper_train500.json); the default (train100) is unchanged.
``--panel_name train200`` (WebShop, user 2026-10-18) likewise: experiment ``webshop:train200:bo3obj-v2``,
BO dir ``webshop_train200/``, files suffixed ``_train200``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import memory_policy as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/query_router_20261016"
BO_OUT = ROOT / "results/bo_agentenv_20261016"
CTRL = ROOT / "results/eval_controlled_20261015"
OPT_DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa"
TASK_SQL = (
    "SELECT id, key, env, split, task_type, object, recep, scene, variation, held_out, description, "
    "initial_observation, emb::text AS emb FROM task WHERE "
)


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


BASE_PANEL = "train100"
# = session_bo_agentenv.PANEL_VERSIONS (train200: WebShop product-isolated panel, user 2026-10-18)
PANEL_VERSIONS = {
    "train100": "bo3obj-v1",
    "train500": "bo3obj-v2",
    "train200": "bo3obj-v2",
}


def experiment_id(ds: str, panel_name: str = BASE_PANEL) -> str:
    """= session_bo_agentenv.experiment_id (no botorch import here; carries the memory policy)."""
    return mp.tag(f"{ds}:{panel_name}:{PANEL_VERSIONS[panel_name]}")


def router_dir(ds: str) -> Path:
    """Features, CV reports and router artifacts of a dataset under the current memory policy (probe
    features and training rows depend on which sessions are retrievable)."""
    return mp.tag_dir(OUT / ds)


def panel(ds: str, panel_name: str = BASE_PANEL) -> dict:
    return json.load(open(ROOT / f"data/external/panels/{ds}_{panel_name}.json"))


def sfx(panel_name: str = BASE_PANEL) -> str:
    """File-name suffix of a panel's router files (none for the 100-panel)."""
    return "" if panel_name == BASE_PANEL else f"_{panel_name}"


def bo_dir(ds: str, panel_name: str = BASE_PANEL) -> Path:
    """= bo_agentenv_driver.run_dir (memory-policy tagged); NoMemory lives in ``nomemory_dir``."""
    return mp.tag_dir(
        BO_OUT / ds if panel_name == BASE_PANEL else BO_OUT / f"{ds}_{panel_name}"
    )


def nomemory_dir(ds: str, panel_name: str = BASE_PANEL) -> Path:
    """No-Memory baseline of a panel: independent of the memory policy (never tagged)."""
    return (
        BO_OUT / ds if panel_name == BASE_PANEL else BO_OUT / f"{ds}_{panel_name}"
    ) / "NoMemory"


def test_items(ds: str) -> list[str]:
    return [
        x["item_id"]
        for s in ("a", "b")
        for x in json.load(open(CTRL / "shards" / f"{ds}_{s}.json"))
    ]


def _llm(kind: str, base: str):
    from trimem.optimizer.query_router.features import OpenAIChat

    if kind == "qwen":
        return OpenAIChat(base)
    if kind == "codex":
        from trimem.optimizer.analyzer_session import CodexCLI

        return CodexCLI(effort="low")
    raise SystemExit(f"unknown LLM {kind!r}")


def _labeler_identity(llm) -> dict | None:
    if llm is None:
        return None
    if hasattr(llm, "identity"):
        return llm.identity
    return {
        "kind": "codex",
        "model": getattr(llm, "model", None),
    }  # cannot be reproduced at deployment


def stage_features(a) -> None:
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments.agentenv_task_adapter import dsn_for
    from trimem.optimizer.query_router import features

    out = router_dir(a.dataset)
    out.mkdir(parents=True, exist_ok=True)
    pnl = panel(a.dataset, a.panel_name)
    with psycopg.connect(dsn_for(a.dataset), row_factory=dict_row) as c:
        train = [
            (
                x["item_id"],
                c.execute(
                    TASK_SQL + "key = %s AND split = 'train' AND NOT held_out",
                    (x["task_key"],),
                ).fetchone(),
            )
            for x in pnl["items"]
        ]
        test = (
            [
                (i, c.execute(TASK_SQL + "agenteval_item = %s", (i,)).fetchone())
                for i in test_items(a.dataset)
            ]
            if a.include_test
            else []
        )
    missing = [i for i, t in train + test if t is None]
    if missing:
        raise SystemExit(f"no task row for {missing[:5]}")
    llm, questions, qmeta = None, None, None
    if a.llm != "none":
        llm = _llm(a.llm, a.llm_base)
        qp = out / "llm_questions.json"
        if qp.exists():
            questions = json.load(open(qp))["questions"]
        else:  # proposed from TRAINING goals only
            questions, qmeta = features.propose_questions(
                llm, [t["description"] for _, t in train], a.questions
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
    for name, items in (("train", train), ("test", test)):
        if not items:
            continue
        t0, recs = time.monotonic(), []
        for qid, task in items:
            vec = task.pop("emb")
            recs.append(
                features.raw_record(
                    qid,
                    task,
                    vec,
                    use_probe=not a.no_probe,
                    llm=llm,
                    questions=questions,
                )
            )
        meta = {
            "dataset": a.dataset,
            "split": name,
            "panel_hash": pnl["panel_hash"] if name == "train" else None,
            "probe_seeds": features.PROBE_SEEDS,
            "llm": a.llm,
            "llm_questions": questions,
            "llm_labeler": _labeler_identity(llm),
            "created_at": datetime.now(UTC).isoformat(),
            "wall_s": time.monotonic() - t0,
        }
        if a.panel_name != BASE_PANEL:
            meta["panel_name"] = a.panel_name
        (out / f"features_{name}{sfx(a.panel_name)}.json").write_text(
            json.dumps({"meta": meta, "records": recs})
        )
        log(
            f"features_{name}: {len(recs)} records, mean overhead "
            f"{sum(r['overhead']['probe_ms'] + r['overhead']['llm_ms'] for r in recs) / len(recs):.1f} ms"
        )


def load_training(ds: str, with_nomemory: bool, panel_name: str = BASE_PANEL):
    from trimem.optimizer.query_router import matrix

    pnl = panel(ds, panel_name)
    ids = [x["item_id"] for x in pnl["items"]]
    exp = experiment_id(ds, panel_name)
    m = matrix.build(matrix.fetch_rows(OPT_DSN, exp), panel_ids=ids, experiment=exp)
    if with_nomemory:
        m = matrix.with_nomemory(
            m, matrix.nomemory_column(nomemory_dir(ds, panel_name), m.query_ids, ds)
        )
    f = json.load(open(router_dir(ds) / f"features_train{sfx(panel_name)}.json"))
    if f["meta"]["panel_hash"] != pnl["panel_hash"]:
        raise SystemExit("features were built for a different panel")
    by_id = {r["query_id"]: r for r in f["records"]}
    return m, [by_id[q] for q in m.query_ids], f["meta"], pnl


def fit_fn(a):
    from trimem.optimizer.query_router import brane, contextual_gp

    if (
        a.model == "brane"
    ):  # BRANE as in the paper: model selection over six tabular families per config
        return lambda X, m: brane.fit(
            X,
            m,
            trials=a.trials,
            n_jobs=a.jobs,
            seed=a.seed,
            keep_best_per_plan=a.keep_per_plan,
        )
    if (
        a.model == "brane-linear"
    ):  # earlier variant (one L2 logistic regression per config)
        return lambda X, m: brane.fit_linear(X, m, max_candidates=a.max_candidates)
    if a.model == "gp":

        def gp(X, m):
            cand = (
                brane.fuzzy_pareto(
                    m.quality.mean(0),
                    brane.config_cost(m.tokens, m.latency_ms),
                    max_candidates=a.max_candidates,
                )
                if a.gp_pruned
                else None
            )
            return contextual_gp.fit(X, m, candidates=cand, max_rows=a.max_rows)

        return gp
    raise SystemExit(f"unknown model {a.model!r}")


def feature_config(a, meta: dict):
    from trimem.optimizer.query_router.evaluate import FeatureConfig

    g = tuple(x for x in a.groups.split(",") if x)
    if "llm" in g and not meta.get("llm_questions"):
        raise SystemExit("--groups llm needs features built with --llm")
    return FeatureConfig(
        g, a.emb_dims, meta.get("llm_questions"), meta.get("llm_labeler")
    )


def cv_path(a, fc) -> Path:
    return router_dir(a.dataset) / (
        f"cv_{a.model}_{'+'.join(fc.groups)}{'_nm' if a.nomemory else ''}"
        f"{'_perplan' if getattr(a, 'keep_per_plan', False) else ''}"
        f"{sfx(getattr(a, 'panel_name', BASE_PANEL))}.json"
    )


def stage_cv(a) -> None:
    from trimem.optimizer.query_router import evaluate

    m, recs, meta, _ = load_training(a.dataset, a.nomemory, a.panel_name)
    fc = feature_config(a, meta)
    log(
        f"matrix {len(m.query_ids)} queries x {len(m.configs)} configurations; groups {fc.groups}; model {a.model}; "
        f"{a.repeats} x {a.folds}-fold"
    )
    runs = [
        evaluate.cross_validate(
            m,
            recs,
            fit_fn(a),
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
            "experiment": experiment_id(a.dataset, a.panel_name),
        },
    }
    p = cv_path(a, fc)
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
    d = res["selected_vs_qbest"]
    log(
        f"router - Q-best single: {d['mean']:+.3f} ± {d['se']:.3f} (per repeat {d['per_repeat']}); "
        f"gate = {res['gate']}; written {p}"
    )


def stage_train(a) -> None:
    from trimem.optimizer.query_router import evaluate, matrix, runtime

    if a.nomemory:
        raise SystemExit(
            "deployment routers must not contain NoMemory; train without --nomemory"
        )
    m, recs, meta, pnl = load_training(a.dataset, False, a.panel_name)
    fc = feature_config(a, meta)
    cvp = cv_path(a, fc)
    if cvp.exists():
        cv = json.load(open(cvp))
        if cv["matrix_fingerprint"] != matrix.fingerprint(m):
            raise SystemExit("the outcome matrix changed since the CV report; rerun cv")
    elif a.force_route:  # explicit user decision: route without waiting for the CV gate
        cv = {"gate": None}
    else:
        raise SystemExit(
            f"run cv first ({cvp} missing): the CV gate decides route vs single"
        )
    policy = "route" if a.force_route else cv["gate"]
    art = evaluate.train(
        m,
        recs,
        fit_fn(a),
        fc,
        dataset=a.dataset,
        policy=policy,
        inner_k=a.inner_folds,
        seed=a.seed,
        rule=a.rule,
        provenance={
            "experiment": experiment_id(a.dataset, a.panel_name),
            "panel_hash": pnl["panel_hash"],
            **(
                {}
                if a.panel_name == BASE_PANEL
                else {"panel_name": a.panel_name, "panel_size": len(pnl["items"])}
            ),
            "configs_in_matrix": len(m.configs),
            "cv_report": str(cvp.relative_to(ROOT)) if cvp.exists() else None,
            "cv_gate": cv["gate"],
            "forced_route": bool(a.force_route),
            "trained_at": datetime.now(UTC).isoformat(),
        },
    )
    p = (
        router_dir(a.dataset)
        / f"router_{'brane-paper' if a.model == 'brane' else a.model}{sfx(a.panel_name)}{a.tag}.json"
    )
    sha = runtime.save(p, art)
    log(
        f"router {p} sha {sha}: policy {policy} (CV gate {cv['gate']}), lambda {art['routing']['lambda']}, "
        f"{len(art['model']['configs'])} candidates, fallback {art['policy']['fallback']}"
    )


def stage_route_test(a) -> None:
    """Routed arm on AgentEval 200 (controlled protocol). Uses the driver's arm runner with the
    router evaluator in place of the fixed-configuration evaluator."""
    import fcntl

    from trimem.experiments import bo_agentenv_driver as drv
    from trimem.optimizer.query_router import runtime

    router = runtime.Router.load(a.router)
    if router.a["dataset"] != a.dataset:
        raise SystemExit("router trained for another dataset")
    if (
        router.needs_llm
        and (router.llm_labeler or {}).get("kind") != "openai-compatible"
    ):
        raise SystemExit(
            "router's LLM features were labeled by a backend deployment cannot reproduce"
        )
    lock = open(BO_OUT / a.dataset / "driver.lock", "w")
    try:  # the BO driver owns the env servers; never run concurrently with it
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("the BO driver is running; route-test must wait") from None
    variant = router.a["model"].get("variant")
    kind = router.a["model"]["kind"] + (
        f"_{variant}" if variant and variant != "linear" else ""
    )
    name = f"Router-{kind}-{router.sha}"  # one artifact = one run identity
    session = {
        "router": str(Path(a.router).resolve()),
        "split": "test",
        "experiment": mp.tag(f"{a.dataset}:agenteval200:controlled-v1:{name}"),
        "baseline_dir": CTRL / f"{a.dataset}_NoMemory",
        "protocol_extra": {"bo_label": name, "router_sha": router.sha},
    }

    def half_cmd(ds, half, out, shard, *, session):
        llm, env = drv.LLM[half], drv.DS[ds]["env"][half]
        return [
            str(ROOT / ".venv-agentgym/bin/python"),
            "-m",
            "trimem.experiments.agentgym_router_eval",
            "--router",
            session["router"],
            "--split",
            "test",
            "--baseline_dir",
            str(session["baseline_dir"]),
            "--experiment",
            session["experiment"],
            "--semantic_base",
            f"http://127.0.0.1:{llm}",
            "--protocol_extra",
            json.dumps(session["protocol_extra"]),
            "--action_format",
            drv.DS[ds]["fmt"],
            *drv.COMMON,
            "--base_url",
            f"http://127.0.0.1:{llm}/v1",
            "--inference_file",
            str(shard),
            "--output_dir",
            str(out),
            "--task_name",
            ds,
            "--env_server_base",
            f"http://127.0.0.1:{env}",
        ]

    def completed(
        ds, out, expected, session
    ):  # routed rows differ per query: count rows, not configs
        import psycopg

        files = drv.item_files(out, ds)
        with psycopg.connect(OPT_DSN) as c:
            ev = dict(
                c.execute(
                    "SELECT query_id, count(*) FROM opt.session_retrieval WHERE experiment_id = %s "
                    "AND status = 'evaluated' GROUP BY 1",
                    (session["experiment"],),
                ).fetchall()
            )
            other = c.execute(
                "SELECT count(*) FROM opt.session_retrieval WHERE experiment_id = %s AND "
                "protocol->>'router_sha' IS DISTINCT FROM %s",
                (session["experiment"], session["protocol_extra"]["router_sha"]),
            ).fetchone()[0]
        if other:
            raise SystemExit(
                f"{other} rows of {session['experiment']} come from another router artifact"
            )
        dup = [q for q, n in ev.items() if n > 1]
        if dup:
            raise SystemExit(f"duplicate evaluated rows for {dup[:5]}; reconcile first")
        return set(files) & set(ev) & set(expected)

    drv.half_cmd, drv.completed = half_cmd, completed
    sa, sb = (
        CTRL / "shards" / f"{a.dataset}_a.json",
        CTRL / "shards" / f"{a.dataset}_b.json",
    )
    ids = test_items(a.dataset)
    n = drv.run_arm(
        a.dataset,
        f"test {name}",
        mp.tag_dir(CTRL / f"{a.dataset}_{name}"),
        sa,
        sb,
        ids,
        session=session,
    )
    log(
        f"ROUTE TEST {'DONE' if n == len(ids) else 'INCOMPLETE'} {a.dataset} {name}: {n}/{len(ids)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("features", "cv", "train", "route-test"))
    ap.add_argument(
        "--dataset", choices=("alfworld", "sciworld", "webshop"), required=True
    )
    ap.add_argument(
        "--include_test",
        action="store_true",
        help="features: also the AgentEval test items",
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
    ap.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="brane: parallel per-configuration model selections",
    )
    ap.add_argument("--groups", default="structured,probe,emb")
    ap.add_argument("--emb_dims", type=int, default=4)
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument(
        "--gp_pruned",
        action="store_true",
        help="gp: route among the fuzzy-Pareto candidates only",
    )
    ap.add_argument("--max_rows", type=int, default=1000)
    ap.add_argument(
        "--nomemory", action="store_true", help="add NoMemory as a candidate column"
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner_folds", type=int, default=4)
    ap.add_argument(
        "--force_route",
        action="store_true",
        help="train: route even if the CV gate chose single",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="cv: repeated K-fold with fold seeds seed..seed+n-1",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", default="knee", help="knee | quality | fixed:<lambda>")
    ap.add_argument("--router", default=None)
    # User 2026-10-17: the relaxed BRANE (best configuration of every plan kept) is the default implementation;
    # --strict_prune restores the paper's fuzzy-Pareto-only candidate set.
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
    ap.add_argument(
        "--tag",
        default="",
        help="train: suffix of the artifact file name (e.g. _lam0.1)",
    )
    ap.add_argument(
        "--panel_name",
        default=BASE_PANEL,
        choices=tuple(PANEL_VERSIONS),
        help="training panel / BO experiment (train500 / train200: files suffixed _<panel>)",
    )
    a = ap.parse_args()
    {
        "features": stage_features,
        "cv": stage_cv,
        "train": stage_train,
        "route-test": stage_route_test,
    }[a.stage](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
