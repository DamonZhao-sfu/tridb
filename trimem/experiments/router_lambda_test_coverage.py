"""Test-set lambda-curve feasibility for Opt-PerQuery (relaxed BRANE router), offline (user 2026-10-19).

For ALFWorld / ScienceWorld (AgentEval 200 test) and τ²-bench (100 test): the deployed relaxed BRANE
artifact (``render_atlas_results.relaxed_router``) routes every test query (the route-test feature file,
``features_test.json``) with each lambda of the router grid (``evaluate.LAMBDAS``; only ``routing.lambda``
changes). Each (lambda, query) is covered when an existing TEST episode of exactly that configuration
(plan, seed_count, depth, fanout, top_k, edges -- the remaining knobs are fixed per plan) exists for that
query: opt.session_retrieval rows split='test' status='evaluated' of the controlled test protocol (fixed
arms, Opt-OneConfig, BO-Knee, SLO arms, every router arm), cross-checked against the per-episode result
files. Missing pairs are counted per lambda and de-duplicated over lambdas for the fill cost.

Read-only: no episodes are run, PG is opened read-only. Writes
results/slo_selection_20261019/router_lambda_test_coverage.{json,md}.

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.router_lambda_test_coverage
"""

from __future__ import annotations

import glob
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime

import numpy as np
import psycopg

from trimem.experiments import render_atlas_results as ra
from trimem.experiments import render_profiling_points as rp
from trimem.optimizer.query_router import evaluate, runtime

ROOT = rp.ROOT
OUT = ROOT / "results/slo_selection_20261019"
QR = ROOT / "results/query_router_20261016"
DATASETS = {
    "alfworld": ("alfworld:agenteval200:controlled-v1:%", 200),
    "sciworld": ("sciworld:agenteval200:controlled-v1:%", 200),
    "tau2": ("tau2:test100:controlled-v1:%", 100),
}
# τ² fixed VGR / VGR+SemFilter test arms were logged under the session-v1 experiment ids (results/tau2_test).
TAU2_EXTRA = ("tau2:train178-test100:session-v1:%",)
KNOBS = ("plan", "seed_count", "depth", "fanout", "top_k", "edges")
LABEL_EDGES = {"similar": "sim", "similar_up": "up", "similar+similar_up": "sim+up"}


def norm(cfg: dict) -> tuple:
    """Canonical configuration key; ``edges`` as stored in opt.config (list) or space_config (str)."""
    e = cfg["edges"]
    e = "+".join(e) if isinstance(e, list) else e
    return (
        cfg["plan"],
        int(cfg["seed_count"]),
        int(cfg["depth"]),
        int(cfg["fanout"]),
        int(cfg["top_k"]),
        e,
    )


def label(k: tuple) -> str:
    return f"{ra.PLAN_NAME.get(k[0], k[0])} s{k[1]} d{k[2]} f{k[3]} k{k[4]} {LABEL_EDGES.get(k[5], k[5])}"


def arm_of(exp: str) -> str:
    return exp.split(":")[3] if exp.count(":") >= 3 else exp


def result_files(ds: str, arm: str) -> set[str]:
    """Query ids that have a per-episode result file for the arm (cross-check of the opt rows)."""
    if ds == "tau2":
        d = ra.T2 / arm
        return {
            f"{json.load(open(f))['domain']}:{json.load(open(f))['task_id']}"
            for f in glob.glob(str(d / "*.json"))
        }
    d = ra.CTRL / f"{ds}_{arm}"
    return (
        {f.stem for f in d.glob(f"{ds}_*.json") if f.stem.split("_")[-1].isdigit()}
        if d.exists()
        else set()
    )


def existing(ds: str) -> tuple[dict, dict]:
    """(config key, query) -> [arms] from evaluated test rows, and per-arm row / file counts."""
    pat, _ = DATASETS[ds]
    pats = (pat,) + (TAU2_EXTRA if ds == "tau2" else ())
    have: dict = defaultdict(list)
    arms: dict = {}
    with psycopg.connect(rp.DSN) as c:
        for p in pats:
            rows = c.execute(
                "SELECT experiment_id, query_id, config FROM opt.session_retrieval "
                "WHERE experiment_id LIKE %s AND split = 'test' AND status = 'evaluated'",
                (p,),
            ).fetchall()
            for exp, q, cfg in rows:
                if cfg.get("plan") not in ("VGR", "VGR+Rank", "P2", "P2+Rank"):
                    continue  # NoMemory / VGR+SemRank are outside the router's space
                arm = arm_of(exp)
                arms.setdefault(arm, {"experiment": exp, "rows": 0})["rows"] += 1
                have[(norm(cfg), q)].append(arm)
    for arm, a in arms.items():
        files = result_files(ds, {"VGR": "VGR", "P2": "P2"}.get(arm, arm))
        a["result_files"] = len(files)
    return have, arms


def route_all(art: dict, recs: list[dict], lam: float) -> list[dict]:
    """Mode ``features``: route the route-test feature records (features_test.json) offline."""
    spec, model, r = art["feature_spec"], art["model"], art["routing"]
    out = []
    for raw in recs:
        x = runtime.transform(spec, raw)
        ch = runtime.choose(model, x, lam, r["w_tokens"], r["w_latency"])
        out.append(
            {
                "query_id": raw["query_id"],
                "key": norm(ch["config"]),
                "config": ch["config"],
                "pred_lat_s": ch["pred"]["latency_ms"] / 1000,
            }
        )
    return out


def live_scores(ds: str, art: dict, arm: str) -> dict[str, np.ndarray]:
    """Per query: the quality predictions the deployed router computed AT TEST TIME (live features).

    The deployed arm logs every candidate's score s_i = q_i - lambda0 * c_i (protocol.router.scores).
    The cost models of these artifacts are per-configuration constants (``const`` ln_tokens /
    ln_latency, checked), so q_i = s_i + lambda0 * c_i is exact and s_i(lambda) = q_i - lambda * c_i is
    precisely what the deployed router would have chosen with routing.lambda = lambda."""
    pat = DATASETS[ds][0].replace("%", arm)
    with psycopg.connect(rp.DSN) as c:
        rows = c.execute(
            "SELECT query_id, protocol->'router' FROM opt.session_retrieval WHERE experiment_id = %s "
            "AND split = 'test' AND status = 'evaluated'",
            (pat,),
        ).fetchall()
    return {q: np.asarray(r["scores"], dtype=float) for q, r in rows}


def route_live(art: dict, live: dict[str, np.ndarray], lam: float) -> list[dict]:
    m, r = art["model"], art["routing"]
    if not all("const" in x for x in m["ln_tokens"] + m["ln_latency"]):
        raise SystemExit(
            "query-dependent cost model: live replay from logged scores is not exact"
        )
    tok = np.array([x["const"] for x in m["ln_tokens"]])
    lat = np.array([x["const"] for x in m["ln_latency"]])
    cost = r["w_tokens"] * tok + r["w_latency"] * lat
    out = []
    for q, s0 in live.items():
        qhat = s0 + r["lambda"] * cost
        i = int(np.argmax(qhat - lam * cost))  # ties -> lowest index, as runtime.choose
        out.append(
            {
                "query_id": q,
                "key": norm(m["configs"][i]),
                "config": m["configs"][i],
                "pred_lat_s": math.exp(lat[i]) / 1000,
            }
        )
    return out


def sweep(route, have: dict, dep_arm: str, lam0: float, n_test: int) -> dict:
    pts, missing_pairs = [], {}
    for lam in evaluate.LAMBDAS:
        rt = route(lam)
        cov = [r for r in rt if (r["key"], r["query_id"]) in have]
        miss = [r for r in rt if (r["key"], r["query_id"]) not in have]
        for r in miss:
            missing_pairs[(r["key"], r["query_id"])] = r
        src = Counter()
        for r in cov:
            a = have[(r["key"], r["query_id"])]
            src[dep_arm if dep_arm in a else sorted(a)[0]] += 1
        dist = Counter(r["key"] for r in rt)
        mk = {r["key"] for r in miss}
        pts.append(
            {
                "lambda": lam,
                "deployed": math.isclose(lam, lam0),
                "n": len(rt),
                "covered": len(cov),
                "missing": len(miss),
                "coverage": len(cov) / len(rt),
                "distinct_configs": len(dist),
                "distinct_missing_configs": len(mk),
                "missing_by_config": {
                    label(k): v
                    for k, v in Counter(r["key"] for r in miss).most_common()
                },
                "covered_by_arm": dict(src.most_common()),
                "routed_configs": {label(k): v for k, v in dist.most_common()},
                "fill_episodes_minimal": len(miss),
                "fill_episodes_full_arms": n_test * len(mk),
                "fill_pred_episode_hours": sum(r["pred_lat_s"] for r in miss) / 3600,
            }
        )
    miss_cfgs = Counter(k for k, _ in missing_pairs)
    return {
        "points": pts,
        "lambdas_fully_covered": [p["lambda"] for p in pts if p["missing"] == 0],
        "union_missing": {
            "pairs": len(missing_pairs),
            "distinct_configs": len(miss_cfgs),
            "by_config": {label(k): v for k, v in miss_cfgs.most_common()},
            "episodes_minimal": len(missing_pairs),
            "episodes_full_arms": n_test * len(miss_cfgs),
            "pred_episode_hours": sum(r["pred_lat_s"] for r in missing_pairs.values())
            / 3600,
            # the (configuration, query) pairs to fill, sorted (consumed by lambda_fill)
            "pair_list": [
                {
                    "query_id": q,
                    "label": label(k),
                    "config": missing_pairs[(k, q)]["config"],
                }
                for k, q in sorted(missing_pairs)
            ],
        },
    }


def dataset(ds: str) -> dict:
    sha_name = ra.relaxed_router(ds)
    art_path = next(
        p
        for p in sorted((QR / ds).glob("router_*.json"))
        if f"Router-brane_paper-{runtime.artifact_sha(json.load(open(p)))}" == sha_name
    )
    art = json.load(open(art_path))
    recs = json.load(open(QR / ds / "features_test.json"))["records"]
    n_test = DATASETS[ds][1]
    have, arms = existing(ds)
    dep_arm = sha_name
    dep = {q: k for (k, q), a in have.items() if dep_arm in a}
    lam0 = float(art["routing"]["lambda"])
    live = live_scores(ds, art, dep_arm)
    agree_live = sum(
        dep.get(r["query_id"]) == r["key"] for r in route_live(art, live, lam0)
    )
    agree_feat = sum(
        dep.get(r["query_id"]) == r["key"] for r in route_all(art, recs, lam0)
    )
    return {
        "ds": ds,
        "name": rp.DS_NAME[ds],
        "n_test": n_test,
        "router_arm": dep_arm,
        "artifact": str(art_path.relative_to(ROOT)),
        "features": str((QR / ds / "features_test.json").relative_to(ROOT)),
        "deployed_lambda": lam0,
        "candidates": len(art["model"]["configs"]),
        "validation": {
            "deployed_arm_rows": len(dep),
            "live_replay_agrees": agree_live,
            "features_file_agrees": agree_feat,
        },
        "existing_arms": arms,
        "live": sweep(
            lambda lam: route_live(art, live, lam), have, dep_arm, lam0, n_test
        ),
        "features_file": sweep(
            lambda lam: route_all(art, recs, lam), have, dep_arm, lam0, n_test
        ),
    }


def _table(d: dict, sw: dict) -> list[str]:
    L = [
        "| λ | covered | missing | coverage | distinct configs routed | distinct missing configs | "
        "fill episodes (minimal) | fill episodes (full arms) | pred. episode-hours |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in sw["points"]:
        L.append(
            f"| {p['lambda']}{' (deployed)' if p['deployed'] else ''} | {p['covered']} | {p['missing']} | "
            f"{p['coverage']:.0%} | {p['distinct_configs']} | {p['distinct_missing_configs']} | "
            f"{p['fill_episodes_minimal']} | {p['fill_episodes_full_arms']} | {p['fill_pred_episode_hours']:.1f} |"
        )
    u = sw["union_missing"]
    L += [
        f"| all λ (union) | | {u['pairs']} pairs | | | {u['distinct_configs']} | {u['episodes_minimal']} | "
        f"{u['episodes_full_arms']} | {u['pred_episode_hours']:.1f} |",
        "",
        f"λ points drawable now without new runs: {', '.join(map(str, sw['lambdas_fully_covered'])) or 'none'}.",
        "",
        (
            "Missing (configuration: queries, union over λ): "
            + "; ".join(f"{k}: {n}" for k, n in u["by_config"].items())
        )
        if u["by_config"]
        else "Nothing missing.",
        "",
    ]
    return L


def md(res: dict) -> str:
    L = [
        "# Opt-PerQuery test λ curve: coverage by existing test episodes",
        "",
        f"Generated {res['generated']}. Nothing was run. For each λ of the router grid (only `routing.lambda` "
        "changes; w_tokens = w_latency = 0.5) every test query is routed with the deployed relaxed BRANE artifact. "
        "A (λ, query) is covered when an evaluated TEST episode of exactly that configuration (plan, seeds, depth, "
        "fanout, top-k, edges) exists for the query in any arm of the same test protocol (controlled-v1 opt rows / "
        "τ² results/tau2_test; success-only-v0 memory, semantic-rubric-v1).",
        "",
        "Two routings are reported. **Live replay (primary)**: the deployed arm logged every candidate's score "
        "s_i = q̂_i − λ0·c_i computed from the LIVE test-time features; the cost models of these artifacts are "
        "per-configuration constants, so q̂_i = s_i + λ0·c_i is exact and argmax(q̂_i − λ·c_i) is what the deployed "
        "router would pick at λ (reproduces the deployed arm 100% at λ0). **features_test.json**: the same artifact "
        "on the route-test feature file — it does NOT reproduce the deployed arm (live probe / τ² first-user-message "
        "features differ), so it is shown only for reference.",
        "",
        "Fill cost: *minimal* = run only the missing (configuration, query) episodes (needs a per-query subset "
        "runner; the union row de-duplicates over λ); *full arms* = one complete test arm per missing configuration "
        "(current drivers). Hours = sum of the router's predicted per-episode latency (train e2e metric), "
        "single-stream; divide by the run's parallelism.",
        "",
    ]
    for d in res["datasets"].values():
        v = d["validation"]
        L += (
            [
                f"## {d['name']} ({d['n_test']} test, router `{d['router_arm']}`, deployed λ {d['deployed_lambda']}, "
                f"{d['candidates']} candidates)",
                "",
                f"Agreement with the deployed router arm at λ0: live replay {v['live_replay_agrees']}/"
                f"{v['deployed_arm_rows']}, features_test.json {v['features_file_agrees']}/{v['deployed_arm_rows']}.",
                "",
                "### Live replay (primary)",
                "",
            ]
            + _table(d, d["live"])
            + ["### features_test.json (reference)", ""]
            + _table(d, d["features_file"])
        )
    return "\n".join(L)


def main() -> int:
    res = {
        "generated": datetime.now(UTC).isoformat(),
        "lambdas": list(evaluate.LAMBDAS),
        "datasets": {ds: dataset(ds) for ds in DATASETS},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (
        (
            "router_lambda_test_coverage.json",
            json.dumps(res, indent=1, ensure_ascii=False, default=str),
        ),
        ("router_lambda_test_coverage.md", md(res)),
    ):
        tmp = OUT / (name + ".tmp")
        tmp.write_text(text)
        tmp.replace(OUT / name)
    print(md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
