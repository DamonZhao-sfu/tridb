"""Five-dataset held-out results for qa-trajectory.html (user request 2026-10-18; τ²-bench restored 2026-10-19).

For ALFWorld, ScienceWorld, AppWorld, τ²-bench and WebShop and every plan run on the held-out test set
(No-Memory, the four fixed plans that were run, BO Q-best, the relaxed BRANE per-query router;
displayed as "Opt-OneConfig" / "Opt-PerQuery" since 2026-10-19):
quality, latency split into retrieval + generation, tokens split into input / output / retrieval-LLM
(SemFilter), plus Pareto scatter plots (quality-latency, quality-tokens) with the non-dominated arms
highlighted, and a short list of conclusions. Data are read from the per-episode result files.

τ²-bench (user 2026-10-21): the pool -v5 test (results/tau2_pool_v5_20261020/test, 100 official test tasks x 1
trial, pass^1): Opt-OneConfig SLO-1/2/3 + Opt-PerQuery default/quality/tokens/latency, with the reused NoMemory /
VGR / VGR_SemFilter arms compared on their trial-0 episodes (``trimem.experiments.tau2_pool_test_table``). The old
4-trial pass^4 / BO-Qbest / BRANE_frozen τ² numbers are no longer shown (this section has no previous-protocol slot);
``tau2`` / ``tau2_episodes`` stay for the other renderers that read the 4-trial reference arms.

    PYTHONPATH=. .venv/bin/python -m trimem.experiments.render_atlas_results
"""

from __future__ import annotations

import fcntl
import glob
import json
import math
import statistics as st
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
CTRL = ROOT / "results/eval_controlled_20261015"
AW = ROOT / "data/external/appworld-0.2/runs/test_normal"
T2 = ROOT / "results/tau2_uncapped_4trials_20261019/test"
TAU_TRIALS = 4
TAU_ARMS = {
    "NoMemory": "NoMemory",
    "VGR": "VGR",
    "P2": "VGR_SemFilter",
    "BO-Qbest": "BO_Qbest",
    "BRANE_frozen": "BRANE_frozen",
}
QR = ROOT / "results/query_router_20261016"
START, END = "<!-- four-dataset-results:start -->", "<!-- four-dataset-results:end -->"
BEFORE = "<!-- relational-tables:start -->"

ARMS = [
    ("NoMemory", "No-Memory"),
    ("VGR", "VGR"),
    ("VGR+Rank", "VGR+Rank"),
    ("P2", "VGR+SemFilter"),
    ("P2+Rank", "VGR+SemFilter+Rank"),
    ("BO-Qbest", "Opt-OneConfig"),
    ("ROUTER", "Opt-PerQuery"),
]


def relaxed_router(ds: str) -> str | None:
    if ds == "tau2":
        # Never mix the previous one-trial router with the four-trial protocol.
        return "BRANE_frozen" if len(tau2("BRANE_frozen")) == 100 else None
    from trimem.optimizer.query_router import runtime

    for p in sorted((QR / ds).glob("router_*.json")) if (QR / ds).exists() else []:
        a = json.load(open(p))
        m = a.get("model", {})
        if (
            m.get("variant") == "paper"
            and (m.get("params") or {}).get("keep_best_per_plan")
            and "train500" not in p.name
        ):
            return f"Router-brane_paper-{runtime.artifact_sha(a)}"
    return None


def agentenv(ds: str, arm: str, *, tolerant: bool = False) -> dict[str, dict]:
    """ALFWorld / ScienceWorld: per-item quality, tokens in/out (LLM call log), retrieval (memory file)."""
    d = CTRL / f"{ds}_{arm}"
    if not d.exists():
        return {}
    calls = defaultdict(list)
    bad_log = False
    if (d / "llm_calls.jsonl").exists():
        for line in open(d / "llm_calls.jsonl"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                if not tolerant:
                    raise
                bad_log = True
                continue
            if c.get("item") is not None:
                calls[int(c["item"])].append(c)
    out = {}
    for f in d.glob(f"{ds}_*.json"):
        idx = f.stem.split("_")[-1]
        if not idx.isdigit():
            continue
        try:
            r = json.load(open(f))["reward"]
        except json.JSONDecodeError:
            if not tolerant:
                raise
            continue  # unfinished result: keep the other completed episodes
        m = d / "memory" / f"{idx}.json"
        bad_memory = False
        try:
            mem = json.load(open(m)) if m.exists() else {}
        except json.JSONDecodeError:
            if not tolerant:
                raise
            mem, bad_memory = {}, True
        cs = calls.get(int(idx), [])
        out[idx] = {
            "q": max(r, 0) / 100 if ds == "sciworld" else float(r),
            "tin": sum(c.get("prompt_tokens") or 0 for c in cs),
            "tout": sum(c.get("completion_tokens") or 0 for c in cs),
            "tsem": mem.get("semantic_tokens", 0) or 0,
            "ret": (mem.get("retrieval_ms") or 0) / 1000,
            "gen": sum(c["seconds"] for c in cs),
            "source": str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f),
            "trial": 0,
            "n_trials": 1,
        }
        if tolerant:
            out[idx]["cost_complete"] = (
                bool(cs)
                and not bad_log
                and not bad_memory
                and (arm == "NoMemory" or m.exists())
            )
    return out


def appworld(arm: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(AW / arm / "*.json")):
        r = json.load(open(f))
        if "quality" not in r:
            continue
        out[r["task_id"]] = {
            "q": r["quality"],
            "tin": r.get("prompt_tokens") or 0,
            "tout": r.get("completion_tokens") or 0,
            "tsem": r.get("semantic_tokens") or 0,
            "ret": (r.get("retrieval_ms") or 0) / 1000,
            "gen": (r.get("episode_ms") or 0) / 1000,
        }
    return out


def tau2_episodes(arm: str) -> dict[tuple[str, int], dict]:
    """Completed raw trials in the current protocol. No legacy-directory fallback."""
    out = {}
    if arm not in TAU_ARMS:
        return out
    allowed_protocols = set()
    for filename in (".run_protocol.json", ".transport_protocol.json"):
        receipt = T2 / TAU_ARMS[arm] / filename
        if receipt.exists():
            protocol = json.loads(receipt.read_text())
            if filename == ".transport_protocol.json" and not protocol.get(
                "authorization"
            ):
                raise ValueError(f"Missing transport-change authorization: {receipt}")
            allowed_protocols.add(protocol["sha256"])
    for f in sorted((T2 / TAU_ARMS[arm]).glob("*.json")):
        r = json.load(open(f))
        if r.get("status") != "done":
            continue
        if r.get("split") != "test" or r.get("num_trials") != TAU_TRIALS:
            raise ValueError(f"Wrong test/trial protocol: {f}")
        if (
            not r.get("run_protocol_sha256")
            or r["run_protocol_sha256"] not in allowed_protocols
        ):
            raise ValueError(f"Unknown run protocol: {f}")
        key = (f"{r['domain']}:{r['task_id']}", r["trial"])
        if key in out or r["trial"] not in range(TAU_TRIALS):
            raise ValueError(f"Duplicate or invalid trial: {f}")
        msgs = (r.get("simulation") or {}).get("messages") or []
        tin = sum(
            (m.get("usage") or {}).get("prompt_tokens", 0)
            for m in msgs
            if m.get("role") == "assistant"
        )
        tout = sum(
            (m.get("usage") or {}).get("completion_tokens", 0)
            for m in msgs
            if m.get("role") == "assistant"
        )
        mem = r.get("memory") or {}
        if arm != "NoMemory" and any(
            mem.get(k) is None for k in ("retrieval_ms", "semantic_tokens")
        ):
            raise ValueError(f"Missing memory cost: {f}")
        ret = (mem.get("retrieval_ms") or 0) / 1000
        if tin + tout != r["agent_tokens"]:
            raise ValueError(f"Agent usage mismatch: {f}")
        wall = r["wall_ms"] / 1000
        if not math.isfinite(wall) or not 0 <= ret <= wall or r["reward"] not in (0, 1):
            raise ValueError(f"Invalid timing or reward: {f}")
        out[key] = {
            "q": r["reward"],
            "tin": tin,
            "tout": tout,
            "tsem": mem.get("semantic_tokens") or 0,
            "ret": ret,
            "gen": wall - ret,
            "trial": r["trial"],
            "source": str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f),
            "protocol_sha256": r.get("run_protocol_sha256"),
        }
    return out


def aggregate_trials(episodes: dict[tuple[str, int], dict]) -> dict[str, dict]:
    """Task pass^4 (all trials successful), with per-episode mean costs.

    A missing trial excludes the task. The auxiliary pass^1..4 estimates use
    C(successes, k) / C(4, k); primary quality for gain, SE and Pareto stays pass^4.
    """
    grouped = defaultdict(dict)
    for (task, trial), row in episodes.items():
        grouped[task][trial] = row
    return {
        task: {
            **{
                k: st.mean(r[k] for r in rows.values())
                for k in ("tin", "tout", "tsem", "ret", "gen")
            },
            "q": float(all(r["q"] == 1 for r in rows.values())),
            "pass_k": {
                str(k): math.comb(sum(r["q"] == 1 for r in rows.values()), k)
                / math.comb(TAU_TRIALS, k)
                for k in range(1, TAU_TRIALS + 1)
            },
            "mean_reward": st.mean(r["q"] for r in rows.values()),
            "n_trials": TAU_TRIALS,
            "protocol_hashes": sorted(
                {
                    r["protocol_sha256"]
                    for r in rows.values()
                    if r.get("protocol_sha256")
                }
            ),
        }
        for task, rows in grouped.items()
        if set(rows) == set(range(TAU_TRIALS))
    }


def tau2(arm: str) -> dict[str, dict]:
    return aggregate_trials(tau2_episodes(arm))


DATASETS = [
    ("alfworld", "ALFWorld", "成功率", 200, lambda a: agentenv("alfworld", a)),
    ("sciworld", "ScienceWorld", "score/100", 200, lambda a: agentenv("sciworld", a)),
    ("appworld", "AppWorld", "TGC", 168, appworld),
    ("tau2", "τ²-bench", "pass^4", 100, tau2),
    ("webshop", "WebShop", "reward", 200, lambda a: agentenv("webshop", a)),
]
# BO Q-best arm directory when the profiling panel is not train100 (WebShop profiled on train200).
BO_ARM = {"webshop": "BO200-Qbest"}


def summarize(ep: dict[str, dict], base: dict[str, dict]) -> dict:
    ids = sorted(ep)
    mean = lambda k: st.mean(ep[i][k] for i in ids)  # noqa: E731
    row = {
        "n": len(ids),
        "quality": mean("q"),
        "tin": mean("tin"),
        "tout": mean("tout"),
        "tsem": mean("tsem"),
        "ret": mean("ret"),
        "gen": mean("gen"),
    }
    se = lambda xs: st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None  # noqa: E731
    row["quality_se"] = se([ep[i]["q"] for i in ids])
    if all("pass_k" in ep[i] for i in ids):
        row["pass_k"] = {
            str(k): {
                "mean": st.mean(ep[i]["pass_k"][str(k)] for i in ids),
                "se": se([ep[i]["pass_k"][str(k)] for i in ids]),
            }
            for k in range(1, TAU_TRIALS + 1)
        }
    for key in ("tin", "tout", "tsem", "ret", "gen"):
        row[key + "_se"] = se([ep[i][key] for i in ids])
    for key, components in (
        ("tokens", ("tin", "tout", "tsem")),
        ("latency", ("ret", "gen")),
    ):
        vals = [sum(ep[i][k] for k in components) for i in ids]
        row[key], row[key + "_se"] = st.mean(vals), se(vals)
    row["n_episodes"] = sum(ep[i].get("n_trials", 1) for i in ids)
    row["trials"] = sorted({ep[i].get("n_trials", 1) for i in ids})
    row["protocol_hashes"] = sorted(
        {h for i in ids for h in ep[i].get("protocol_hashes", [])}
    )
    common = [i for i in ids if i in base]
    row["n_paired"] = len(common)
    if common:
        d = [ep[i]["q"] - base[i]["q"] for i in common]
        row.update(gain=st.mean(d), gain_se=se(d))
    return row


PLAN_NAME = {
    "VGR": "VGR",
    "VGR+Rank": "VGR+Rank",
    "P2": "VGR+SemFilter",
    "P2+Rank": "VGR+SemFilter+Rank",
    "P1+Rank": "V+SemFilter+G+Rank",
}


def routing(arm: str) -> list[dict]:
    """Per (plan, configuration): queries routed there and their mean quality (opt rows of the router arm)."""
    import psycopg

    with psycopg.connect("host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa") as c:
        rows = c.execute(
            """SELECT protocol->'space_config' AS cfg, count(*), avg(quality) FROM opt.session_retrieval
            WHERE experiment_id LIKE %s AND split = 'test' AND status = 'evaluated' GROUP BY 1 ORDER BY 2 DESC""",
            (f"%:{arm}",),
        ).fetchall()
    out = []
    for cfg, n, q in rows:
        out.append(
            {
                "plan": cfg["plan"],
                "plan_name": PLAN_NAME.get(cfg["plan"], cfg["plan"]),
                "n": n,
                "quality": float(q),
                "config": f"s{cfg['seed_count']} d{cfg['depth']} f{cfg['fanout']} k{cfg['top_k']} "
                + {
                    "similar": "sim",
                    "similar_up": "sim_up",
                    "similar+similar_up": "sim+up",
                }.get(cfg["edges"], cfg["edges"]),
            }
        )
    return out


def collect() -> list[dict]:
    out = []
    for key, name, metric, n_test, load in DATASETS:
        if key == "tau2":
            out.append(with_fixed(tau2_pool_dataset(name, n_test)))
            continue
        if key == "alfworld":
            out.append(with_sweep(with_fixed(alf_pool_dataset(name, n_test))))
            continue
        if key == "sciworld" and sci_active():
            out.append(with_sweep(with_fixed(sci_pool_dataset(name, n_test))))
            continue
        if (
            key == "webshop" and ws_active()
        ):  # user 2026-10-23: WebShop v7 once its profiling paused
            out.append(with_sweep(with_fixed(webshop_pool_dataset(name, n_test))))
            continue
        router = relaxed_router(key)
        base = load("NoMemory")
        arms = []
        for aid, label in ARMS:
            real = (
                router
                if aid == "ROUTER"
                else BO_ARM.get(key, aid)
                if aid == "BO-Qbest"
                else aid
            )
            if not real:
                continue
            ep = load(real)
            if len(ep) != n_test or set(ep) != set(base):  # same complete task panel
                continue
            row = summarize(ep, base)
            if aid == "NoMemory":  # the paired baseline itself
                row.pop("gain", None), row.pop("gain_se", None)
            arms.append({"id": aid, "label": label, **row})
        out.append(
            {
                "key": key,
                "name": name,
                "metric": metric,
                "n": n_test,
                "arms": arms,
                "trials": TAU_TRIALS if key == "tau2" else 1,
                "pending": [
                    label
                    for aid, label in (
                        ("BO-Qbest", "Opt-OneConfig"),
                        ("ROUTER", "Opt-PerQuery"),
                    )
                    if aid not in {a["id"] for a in arms}
                ]
                if key == "tau2"
                else [],
                "routing": routing(router) if router else [],
            }
        )
    return out


def _tau2_arm(a: dict) -> dict:
    """tau2_pool_test_table arm -> this section's arm fields (quality / gain / cost split, + paired stats)."""
    v = a.get("vs_nomemory")
    return {
        "id": a["id"],
        "label": a["label"],
        "kind": a["kind"],
        "color": a["color"],
        "n": a["n"],
        "n_episodes": a["n"],  # 1 trial per task
        "quality": a["q"],
        "quality_se": a["q_se"],
        **(
            {
                "gain": v["delta"],
                "gain_se": v["se"],
                "wins": v["wins"],
                "losses": v["losses"],
                "p_sign": v["p_sign"],
                "p_z": v["p_z"],
            }
            if v
            else {}
        ),
        "tokens": a["tok"],
        "tokens_se": a["tok_se"],
        "latency": a["lat"],
        "latency_se": a["lat_se"],
        **{k: a[k] for k in ("tin", "tout", "tsem", "ret", "gen")},
        "distinct_configs": a.get("distinct_configs"),
        "reused": a.get("reused", 0),
        "all4": a.get("all4"),
    }


POLICY_SHORT = {
    "success_only": "succ",
    "success_failure": "succ+fail",
    "success_failure_capped": "succ+fail cap",
}


def tau2_routing(arm: dict) -> list[dict]:
    """Per (plan, configuration) of one routed τ² test arm: tasks routed there and their mean pass^1."""
    by: dict[str, list[dict]] = defaultdict(list)
    for row in arm["items"].values():
        by[row["cfg"]].append(row)
    out = []
    for k, rows in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cfg = json.loads(k)
        out.append(
            {
                "plan": cfg["plan"],
                "plan_name": "No-Memory"
                if cfg["plan"] == "NoMemory"
                else PLAN_NAME.get(cfg["plan"], cfg["plan"]),
                "n": len(rows),
                "quality": st.mean(r["q"] for r in rows),
                "config": "no memory"
                if cfg["plan"] == "NoMemory"
                # edges are fixed (similar+similar_up) in the -v5 space: only the searched knobs are shown
                else f"s{cfg['seed_count']} d{cfg['depth']} f{cfg['fanout']} k{cfg['top_k']} "
                + POLICY_SHORT.get(
                    cfg.get("outcome_policy", ""), cfg.get("outcome_policy", "")
                ),
            }
        )
    return out


def with_fixed(d: dict) -> dict:
    """Fixed-plan reference arms (user 2026-10-23; ``fixed_plans_page``): each plan's default configuration on the
    test set, appended to the dataset's arms as kind ``plan`` (own marker per plan, legend toggles) once its report
    exists; ``fixed`` carries the status line (pending / complete / absent plans) either way."""
    from trimem.experiments import fixed_plans_page as fp

    t = fp.table(d["key"])
    conv = _tau2_arm if d["key"] == "tau2" else _alf_arm
    arms = [
        {
            **conv(a),
            "plan": a["plan"],
            "mk": "plan",
            "short": a["short"],
            "steps": a.get("steps"),
            "steps_se": a.get("steps_se"),
            "steps_success": a.get("steps_success"),
            "pair_sources": a.get("pair_sources"),
        }
        for a in t["arms"]
    ]
    return {
        **d,
        "arms": d["arms"] + arms,
        "fixed": {
            "status": t["status"],
            "line": fp.status_line(t),
            "absent": t.get("absent") or {},
        },
    }


def with_sweep(d: dict) -> dict:
    """Opt-PerQuery weight sweep (user 2026-10-23; ``perquery_sweep_page``): ALFWorld / ScienceWorld get ``pqs``
    (11 weights: Q-T / Q-L curves at step 0.2, their test front; overlays = this dataset's arms, latency = the plotted span)."""
    from trimem.experiments import perquery_sweep_page as pqs

    return {
        **d,
        "pqs": pqs.attach(d["key"], d["arms"], "quality", "tokens", "latency"),
    }


def fixed_conclusions(d: dict) -> list[str]:
    """English bullet on the fixed-plan reference arms of one dataset (empty while pending)."""
    fx = [a for a in d["arms"] if a.get("kind") == "plan"]
    if not fx:
        return []
    opt = [a for a in d["arms"] if a.get("kind") in ("oneconfig", "perquery")]
    best = max(fx, key=lambda a: (a["quality"], -a["tokens"]))
    txt = (
        f"{d['name']} fixed-plan defaults on test: "
        + "; ".join(f"{a['short']} {a['quality']:.3f}" for a in fx)
        + f". Best plan default {best['short']} {best['quality']:.3f} ± {best['quality_se']:.3f} SE"
        + (f" (gain {best['gain']:+.3f})" if best.get("gain") is not None else "")
    )
    if opt:
        bo = max(opt, key=lambda a: (a["quality"], -a["tokens"]))
        txt += (
            f" vs best Opt arm {bo['label']} {bo['quality']:.3f} at {bo['tokens']:,.0f} tokens "
            f"(plan default {best['tokens']:,.0f} tokens)"
        )
    return [txt + "."]


def tau2_pool_dataset(name: str, n_test: int) -> dict:
    """τ² entry: the pool -v5 test table (7 Opt arms + 3 references on trial 0), pass^1."""
    from trimem.experiments import tau2_pool_test_table as tt

    t = tt.table()
    arms = [_tau2_arm(a) for a in t["arms"]]
    pq = next((a for a in t["arms"] if a["id"] == "Opt-PerQuery-default"), None)
    return {
        "key": "tau2",
        "name": name,
        "metric": "pass^1",
        "n": n_test,
        "arms": arms,
        "trials": t["trials"],
        "pending": []
        if t.get("status") == "complete"
        else ["Opt-OneConfig", "Opt-PerQuery"],
        "note": (
            f"τ² protocol: pool -v5 test ({t['dir']}/test), {n_test} official test tasks × 1 trial (trial 0, "
            f"seed 300), quality = pass^1, test concurrency {t.get('concurrency')}. No-Memory / VGR / VGR+SemFilter "
            f"= trial 0 of the reused 4-trial arms ({t['reuse_dir']}); their 4-trial numbers are context only."
            if t.get("status") == "complete"
            else f"τ² pool -v5 test: {t.get('status')}"
        ),
        "routing": tau2_routing(pq) if pq else [],
        "routing_arm": "Opt-PerQuery default (w_Q / w_L / w_T = 0.6 / 0.2 / 0.2)",
    }


def _alf_arm(a: dict) -> dict:
    """alf_pool_page arm -> this section's arm fields. Primary numbers = outer-CV mean (± std across folds);
    quality_se = SE_tasks (mean within-fold task SE); agent tokens (input + output, not split in the opt rows)
    go to ``tin``; latency = retrieval (+ router overhead) + generation."""
    se = a.get("se_tasks") or {}
    return {
        "id": a["id"],
        "label": a["label"],
        "kind": a.get("kind"),
        "router": a.get("router"),
        "color": a["color"],
        "n": a["n"],
        "n_episodes": a["n"] * (a.get("folds") or 1),
        "quality": a["q"],
        "quality_se": se.get("q") or 0.0,
        "quality_std": a.get("q_std"),
        "gain": a.get("gain"),
        "gain_se": se.get("gain") or 0.0,
        "gain_std": a.get("gain_std"),
        "tokens": a["tok"],
        "tokens_std": a.get("tok_std"),
        "latency": a["lat"],
        "latency_std": a.get("lat_std"),
        "tokens_se": 0.0,
        "latency_se": 0.0,
        "steps": a.get("steps"),
        "steps_std": a.get("steps_std"),
        "steps_success": a.get("steps_success"),
        "steps_success_std": a.get("steps_success_std"),
        "steps_se": se.get("steps"),
        "tin": a["tok"] - a["tsem"],
        "tout": 0.0,
        "tsem": a["tsem"],
        "ret": a["ret"],
        "gen": a["lat"] - a["ret"],
        "folds": a.get("folds"),
        "weights": a.get("weights"),
        "protocol": a.get("protocol"),
        # secondary (user 2026-10-21): the same arm in the c128 test run (quality reuse on), or the reference's
        # old controlled-v1 run; c128_only = LimeQO (not rerun at concurrency 16)
        "latency_wall": a.get("lat_wall"),
        "latency_wall_std": a.get("lat_wall_std"),
        "distinct": a.get("distinct"),
        # v7 (user 2026-10-22): deployed arms primary; the same arm's outer-CV mean ± std as secondary
        "cv": a.get("cv"),
        # ScienceWorld v7 (deployed only): routing share per plan of each routed arm
        "plan_shares": a.get("plan_shares"),
    }


def sci_active() -> bool:
    from trimem.experiments import sci_pool_page as sp

    return sp.active()


def ws_active() -> bool:
    from trimem.experiments import webshop_pool_page as wp

    return wp.active()


def webshop_pool_dataset(name: str, n_test: int) -> dict:
    """WebShop entry (user 2026-10-23; webshop_test_v7 / webshop_pool_page): the pool-strat200-v7 test, DEPLOYED arms
    only (± SE across the 200 tasks; no outer CV, as sci_pool_dataset); train pool = all stored train goals, sharing
    products with the test set (option B); references No-Memory / VGR = the controlled-v1 originals; training
    scatter and steps of the 50 configurations. A ``cv`` block only if a CV test was run (``t['k']``)."""
    from trimem.experiments import alf_pool_page as ap
    from trimem.experiments import webshop_pool_page as wp

    P = wp.train_panel()
    t = wp.test_table()
    common = {
        "key": "webshop",
        "name": name,
        "metric": "reward",
        "n": n_test,
        "trials": 1,
        "train_steps": alf_train_steps(P) if P else None,
        "train_scatter": alf_train_scatter(P) if P else None,
    }
    if t.get("status") != "complete":
        return {
            **common,
            "arms": [],
            "pending": ["Opt-OneConfig", "Opt-PerQuery"],
            "note": f"WebShop v7 test: {t.get('status') or 'pending'}. " + wp.protocol_note("test"),
            "routing": [],
        }
    r0, r1 = t.get("catalog_rounds") or [None, None]
    pretty = lambda n: ap.label_of(n)  # noqa: E731
    return {
        **common,
        "arms": [_alf_arm(a) for a in t["refs"]] + [_alf_arm(a) for a in t["arms"]],
        "pending": [],
        "io_merged": True,
        "primary": "deployed",
        "c16": True,
        "note": (
            f"WebShop protocol: pool-strat200-v7 test ({t['dir']}/test_v7c16), {t['n_test']} AgentEval items × 1 run "
            "under the controlled c16 runtime (one configuration at a time, 2 halves × 8 processes, quality reuse off, "
            f"SemFilter v2b). Selection catalog = the {t.get('catalog_configs')} fully evaluated configurations of "
            f"profiling rounds {r0}–{r1} (same c16 runtime, 100 training tasks each, stratified by product "
            "category, drawn from all 3,930 stored train goals; train / test share products: yes — user 2026-10-23 "
            "option B; no test goal is in the training pool and retrieval excludes each query's own product) + "
            "No-Memory as a router candidate; Opt-OneConfig SLO-1/2/3 + Opt-PerQuery (BRANE) at 4 weight vectors, all "
            "DEPLOYED (fitted on the full training set; ± SE across the 200 tasks); no outer CV. Quality = WebShop "
            "reward (0–1). No-Memory / VGR = the original controlled-v1 reference arms "
            "(same c16 layout). Plotted latency = LLM-call span + retrieval for every arm; the episode wall time (+ "
            "retrieval) is in the table. "
            + wp.protocol_note("test")
        ),
        "routing": t["routing_deployed"],
        "routing_arm": "Opt-PerQuery BRANE default, deployed full-train router (w_Q / w_L / w_T = 0.6 / 0.2 / 0.2)",
        **(
            {
                "cv": {
                    "k": t["k"],
                    "cv_arms": [_alf_arm(a) for a in t["cv_arms"]],
                    "comparison": [
                        {"a": pretty(c["a"]), "b": pretty(c["b"]), "diff": c["diff"]}
                        for c in t["comparison"]
                    ],
                }
            }
            if t.get("k")
            else {}
        ),
    }


def ws_conclusions(d: dict) -> list[str]:
    """English WebShop bullets (v7 test: deployed arms only; train / test share products, option B)."""
    arms = d["arms"]
    if not arms:
        return [d.get("note", "WebShop v7 test pending.")]
    opt = [a for a in arms if a.get("kind") in ("oneconfig", "perquery")]
    by = {a["id"]: a for a in arms}
    nm, vgr = by.get("NoMemory"), by.get("VGR")
    best = max(opt, key=lambda a: (a["quality"], -a["tokens"]))
    ref = (
        f" (No-Memory {nm['quality']:.3f}, VGR {vgr['quality']:.3f})"
        if nm and vgr
        else ""
    )
    cvq = [a["cv"]["q"] for a in opt if a.get("cv") and a["cv"].get("q") is not None]
    return [
        f"WebShop (v7 test, deployed arms only; catalog = profiling rounds 0–5 under the controlled c16 runtime; training "
        "pool shares products with the test set (option B); 200 tasks × 1 run): the "
        f"deployed Opt arms reach {min(a['quality'] for a in opt):.3f}–{max(a['quality'] for a in opt):.3f} mean "
        f"reward{ref}; best {best['label']} {best['quality']:.3f} ± {best['quality_se']:.3f} SE (gain "
        f"{best['gain']:+.3f} ± {best['gain_se']:.3f})"
        + (f"; outer 5-fold CV {min(cvq):.3f}–{max(cvq):.3f}." if cvq else "."),
        f"WebShop cost: Opt arms {min(a['tokens'] for a in opt):,.0f}–{max(a['tokens'] for a in opt):,.0f} tokens, "
        f"{min(a['latency'] for a in opt):.1f}–{max(a['latency'] for a in opt):.1f} s LLM-span latency"
        + _ws_steps(opt, " and ")
        + " per episode"
        + (
            f" (VGR {vgr['tokens']:,.0f} tokens, {vgr['latency']:.1f} s"
            + _ws_steps([vgr], ", ")
            + (
                f"; No-Memory {_ws_steps([nm], '').strip()}"
                if nm and nm.get("steps") is not None
                else ""
            )
            + ")."
            if vgr
            else "."
        )
        + _ws_budget(),
    ]


def _ws_budget() -> str:
    from trimem.experiments import webshop_pool_page as wp

    P = wp.protocol() or {}
    mr = P.get("max_round")
    return (
        f" Episode budget max_round {mr}; No-Memory / VGR (30-turn runs) in the derived-truncation-{mr} view."
        if mr
        else ""
    )


def _ws_steps(arms: list[dict], sep: str) -> str:
    """'<sep>a–b environment steps' over ``arms`` (empty when no arm has steps; pure)."""
    v = [a["steps"] for a in arms if a and a.get("steps") is not None]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    return sep + (
        f"{lo:.2f} environment steps" if lo == hi else f"{lo:.2f}–{hi:.2f} environment steps"
    )


def alf_train_steps(panel: dict | None = None) -> dict | None:
    """ALFWorld v7 profiling configurations for the per-configuration steps bar chart (user 2026-10-21):
    alf_pool_page.train_panel points (environment steps over the observed tasks ± SE across tasks, on successes).
    ``panel``: another v7 train panel (ScienceWorld, sci_pool_page)."""
    from trimem.experiments import alf_pool_page as ap
    from trimem.experiments.render_pool_steps import train_points

    return train_points(
        panel if panel is not None else ap.train_panel(), "environment steps / episode"
    )


def alf_fixed_arms(base: dict[str, dict], *, span: bool, n_test: int) -> list[dict]:
    """Completed SemFilter reference runs on exactly the current ALFWorld test panel."""
    from trimem.experiments import alf_pool_page as ap

    if len(base) != n_test:
        raise ValueError("ALFWorld reference baseline is not the complete test panel")
    out = []
    for name, color in (("P2", "#8f5bb5"), ("P2+Rank", "#c0558a")):
        items = ap.reference_items(name, sorted(base), span=span)
        wall = ap.reference_items(name, sorted(base)) if span else items
        if set(items) != set(base) or set(wall) != set(base):
            raise ValueError(f"ALFWorld {name}: incomplete or mismatched test panel")
        episodes = {
            q: {
                "q": r["q"],
                "tin": r["tok"] - r["tsem"],
                "tout": 0.0,
                "tsem": r["tsem"],
                "ret": r["ret"],
                "gen": r["lat"] - r["ret"],
            }
            for q, r in items.items()
        }
        out.append(
            {
                "id": name,
                "label": PLAN_NAME[name],
                "kind": "ref",
                "color": color,
                **summarize(episodes, base),
                **ap.step_summary(
                    [{"steps": r["steps"], "ok": r["q"] == 1} for r in items.values()]
                ),
                "latency_wall": st.mean(r["lat"] for r in wall.values()),
                "protocol": f"controlled-v1 {name} (2026-10-15, concurrency 16; original run)",
                "source": str((CTRL / f"alfworld_{name}").relative_to(ROOT)),
            }
        )
    return out


def alf_train_scatter(panel: dict | None = None) -> dict | None:
    """The ALFWorld v7 training configurations of the current regime (rounds 6-21; alf_pool_page.train_panel) for
    the quality x tokens / quality x latency scatter: success ± SE, tokens ± SE, latency ± SE, relaxed 3-D front,
    Opt-OneConfig picks, slot (channel), steps; No-Memory = its c16 rerun. ``panel``: another v7 train panel
    (ScienceWorld: rounds 0-6, its own c16 NoMemory)."""
    from trimem.experiments import alf_pool_page as ap

    P = panel if panel is not None else ap.train_panel()
    if not P:
        return None
    keep = (
        "id",
        "round",
        "channel",
        "source",
        "plan",
        "knobs",
        "status",
        "partial",
        "n",
        "q",
        "q_se",
        "gain",
        "gain_se",
        "tok",
        "tok_se",
        "lat",
        "lat_se",
        "steps",
        "steps_success",
        "front3",
        "slo",
    )
    reg = P.get("regime") or {}
    return {
        "points": [{k: p.get(k) for k in keep} for p in P["points"]],
        "nomem": P["nomem"],
        "n_full": P["n_full"],
        "n_queries": P["n_queries"],
        "rounds": [reg.get("from_round"), reg.get("last_round")],
        "slo_bounds": P.get("slo_bounds"),
        "runtime": reg.get("runtime"),
        "qname": P.get("qname"),
        "nm_label": reg.get("nomemory_label") or "c16 rerun",
    }


def alf_pool_dataset(name: str, n_test: int) -> dict:
    """ALFWorld entry: the pool-strat200-v7 RE-TEST (user 2026-10-22; alf_test_v7): catalog rounds 6-21, deployed
    full-train arms PRIMARY (± SE across the 200 tasks), outer 5-fold CV mean ± std secondary; references
    No-Memory / VGR = the controlled-v1 originals; the 70-configuration training scatter and steps."""
    from trimem.experiments import alf_pool_page as ap

    t = ap.test_table()
    if t.get("status") != "complete" or t.get("run") != "v7":
        return {
            "key": "alfworld",
            "name": name,
            "metric": "success rate",
            "n": n_test,
            "arms": [],
            "trials": 1,
            "pending": ["Opt-OneConfig", "Opt-PerQuery"],
            "note": f"ALFWorld v7 re-test: {t.get('status') or 'pending'}",
            "routing": [],
            "train_steps": alf_train_steps(),
            "train_scatter": alf_train_scatter(),
        }
    pretty = lambda n: ap.label_of(n)  # noqa: E731
    return {
        "key": "alfworld",
        "name": name,
        "metric": "success rate",
        "n": n_test,
        "arms": [_alf_arm(a) for a in t["refs"]] + [_alf_arm(a) for a in t["arms"]],
        "trials": 1,
        "pending": [],
        "io_merged": True,
        "primary": "deployed",
        "train_steps": alf_train_steps(),
        "train_scatter": alf_train_scatter(),
        "c16": True,
        "note": (
            f"ALFWorld protocol: pool-strat200-v7 re-test ({t['dir']}/test_v7c16), {t['n_test']} AgentEval items × "
            "1 run under the controlled c16 runtime (one configuration at a time, 2 halves × 8 processes, quality "
            "reuse off, SemFilter v2). Selection catalog = the 55 fully evaluated configurations of profiling rounds "
            "6–21 (same c16 runtime, SemFilter v2, 100 stratified training tasks each) + No-Memory as a router "
            "candidate; Opt-OneConfig SLO-1/2/3 + Opt-PerQuery (BRANE) at 4 weight vectors. PRIMARY = the deployed "
            "arms fitted on the full training set (± SE across the 200 tasks); secondary = outer "
            f"{t['k']}-fold CV (mean ± std across fold selections / routers). No-Memory / VGR = the original "
            "controlled-v1 reference arms (same c16 layout). Plotted latency = LLM-call span + retrieval for every "
            "arm (the only definition the original No-Memory run has); the episode wall time (+ retrieval) is in the "
            "table."
        ),
        "routing": t["routing_deployed"],
        "routing_arm": "Opt-PerQuery BRANE default, deployed full-train router (w_Q / w_L / w_T = 0.6 / 0.2 / 0.2)",
        "cv": {
            "k": t["k"],
            "cv_arms": [_alf_arm(a) for a in t["cv_arms"]],
            "comparison": [
                {"a": pretty(c["a"]), "b": pretty(c["b"]), "diff": c["diff"]}
                for c in t["comparison"]
            ],
        },
    }


def sci_pool_dataset(name: str, n_test: int) -> dict:
    """ScienceWorld entry: the pool-strat200-v7 TEST, DEPLOYED arms only (user 2026-10-22; sci_test_v7): catalog =
    profiling rounds 0-6 (controlled c16, SemFilter v2, 100 stratified tasks each); Opt-OneConfig SLO-1/2/3 +
    Opt-PerQuery BRANE × 4 weights fitted on the full training set (± SE across the 200 tasks); no outer CV;
    references No-Memory / VGR = the controlled-v1 originals; training scatter and steps of all 50 configurations."""
    from trimem.experiments import sci_pool_page as sp

    P = sp.train_panel()
    t = sp.test_table()
    common = {
        "key": "sciworld",
        "name": name,
        "metric": "score/100",
        "n": n_test,
        "trials": 1,
        "train_steps": alf_train_steps(P) if P else None,
        "train_scatter": alf_train_scatter(P) if P else None,
    }
    if t.get("status") != "complete":
        return {
            **common,
            "arms": [],
            "pending": ["Opt-OneConfig", "Opt-PerQuery"],
            "note": f"ScienceWorld v7 test: {t.get('status') or 'pending'}",
            "routing": [],
        }
    r0, r1 = t.get("catalog_rounds") or [None, None]
    return {
        **common,
        "arms": [_alf_arm(a) for a in t["refs"]] + [_alf_arm(a) for a in t["arms"]],
        "pending": [],
        "io_merged": True,
        "primary": "deployed",
        "c16": True,
        "note": (
            f"ScienceWorld protocol: pool-strat200-v7 test ({t['dir']}/test_v7c16), {t['n_test']} AgentEval items × "
            "1 run under the controlled c16 runtime (one configuration at a time, 2 halves × 8 processes, quality "
            f"reuse off, SemFilter v2). Selection catalog = the {t.get('catalog_configs')} fully evaluated "
            f"configurations of profiling rounds {r0}–{r1} (same c16 runtime, SemFilter v2, 100 stratified training "
            "tasks each) + No-Memory as a router candidate; Opt-OneConfig SLO-1/2/3 + Opt-PerQuery (BRANE) at 4 "
            "weight vectors, all DEPLOYED (fitted on the full training set; ± SE across the 200 tasks); no outer CV. "
            "Quality = score/100. No-Memory / VGR = the original controlled-v1 reference arms (same c16 layout). "
            "Plotted latency = LLM-call span + retrieval for every arm (the only definition the original No-Memory "
            "run has); the episode wall time (+ retrieval) is in the table."
        ),
        "routing": t["routing_deployed"],
        "routing_arm": "Opt-PerQuery BRANE default, deployed full-train router (w_Q / w_L / w_T = 0.6 / 0.2 / 0.2)",
    }


SECTION = r"""<!-- four-dataset-results:start -->
<section class="panel section" id="four-ds"><style>
#four-ds .fd-tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 12px}#four-ds .fd-tabs button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
#four-ds .fd-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:16px}
#four-ds .fd-card{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:var(--panel);min-width:0}
#four-ds .fd-grid>.fd-card:has(#fd-tab),#four-ds #fd-reliability{grid-column:1/-1}
#four-ds #fd-reliability[hidden],#four-ds #fd-pqs[hidden],#four-ds #fd-cv[hidden],#four-ds #fd-train[hidden],#four-ds #fd-steps[hidden],#four-ds #fd-psteps[hidden]{display:none}#four-ds #fd-pqs,#four-ds #fd-cv,#four-ds #fd-steps,#four-ds #fd-psteps,#four-ds #fd-train{grid-column:1/-1}
#four-ds .fd-leg label{cursor:pointer;white-space:nowrap}
#four-ds .fd-card h4{margin:0 0 4px;font-size:14px}#four-ds svg{display:block;width:100%;height:auto}
#four-ds svg text{fill:var(--muted);font-size:11px;font-family:inherit}#four-ds svg .fd-v{fill:var(--fg);font-size:10.5px}
#four-ds .fd-leg{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:2px 0 6px}
#four-ds .fd-leg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
#four-ds table{border-collapse:collapse;font-size:12.5px;width:100%}#four-ds th,#four-ds td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}
#four-ds th:first-child,#four-ds td:first-child{text-align:left}#four-ds .fd-wrap{overflow-x:auto}
#four-ds ol li{margin:5px 0}
</style>
<h2>五个数据集 · 各计划的质量 / 延迟 / Token 与 Pareto</h2>
<p class="small">Held-out test panels: ALFWorld 200 tasks, ScienceWorld 200 tasks、AppWorld test_normal 168、WebShop test 200, and τ² 100 official test tasks × 1 trial (pool -v5 test, trial 0). Each dataset uses its own evaluation protocol. Only arms covering the same complete task panel are compared.
Latency = retrieval (including SemFilter and routing probes) + generation (LLM calls, or episode wall time including the user simulator for τ²). Tokens = agent input + output + retrieval-side LLM usage.
Connected filled points show the Pareto front of observed means (higher quality, lower cost). SE = task-level sample standard deviation / √number of tasks. The τ² quality metric is pass^1 = the trial-0 success of each task (one trial per task); the reference arms (No-Memory, VGR, VGR+SemFilter) are the trial-0 episodes of the earlier 4-trial runs, i.e. the same official seed as the Opt arms. Gain SE uses paired per-task differences from No-Memory. All costs are per episode.</p>
<p class="small"><b>命名</b>：Opt-OneConfig = BO 在训练面板上选出一个配置，测试集每题都用它；Opt-PerQuery = 逐题路由器（BRANE 式），为每道题选一个配置。</p>
<div class="fd-tabs" id="fd-tabs"></div><p class="small" id="fd-status"></p><div class="fd-leg" id="fd-fleg" role="group" aria-label="fixed-plan reference arms"></div>
<div class="fd-grid">
<div class="fd-card"><h4 id="fd-qt">质量</h4><div id="fd-q"></div></div>
<div class="fd-card"><h4>延迟 / 题（秒）</h4><div class="fd-leg"><span><i style="background:#c26a1b"></i>检索</span><span><i style="background:#5b7290"></i>生成</span></div><div id="fd-l"></div></div>
<div class="fd-card"><h4>Token / 题</h4><div class="fd-leg" id="fd-tleg"><span><i style="background:#5b7290"></i>输入</span><span><i style="background:#2f8f6b"></i>输出</span><span><i style="background:#8f5bb5"></i>检索侧 LLM</span></div><div id="fd-t"></div></div>
<div class="fd-card"><h4>Pareto · 质量 × 延迟</h4><div id="fd-pl"></div></div>
<div class="fd-card"><h4>Pareto · 质量 × Token</h4><div id="fd-pt"></div></div>
<div class="fd-card" id="fd-reliability" hidden><h4>τ²-bench · paired comparison with No-Memory (pool -v5 test, 100 tasks × 1 trial, pass<sup>1</sup>)</h4>
<p class="small">Δ = mean per-task pass<sup>1</sup> difference from No-Memory on the same 100 tasks and the same trial ± paired SE. W / L = tasks the arm solves and No-Memory does not / the reverse. p (exact) = two-sided sign test on the discordant tasks (= exact McNemar); p (normal) = two-sided normal test of the paired mean. Opt arms are compared post hoc; p values are not corrected for multiple comparisons. Arms that route the same (configuration, task) pair share that single episode, so the Opt arms are correlated. 4-trial pass<sup>1</sup> = the reference arm's mean success over its four trials (context only, other protocol).</p>
<div class="fd-wrap" id="fd-reliability-table"></div></div>
<div class="fd-card" id="fd-train" hidden><h4 id="fd-train-t">ALFWorld · training profiling</h4><div class="fd-leg" id="fd-train-leg"></div>
<div class="fd-grid"><div><h4>Success × tokens / episode</h4><div id="fd-tr-tok"></div></div><div><h4>Success × latency / episode</h4><div id="fd-tr-lat"></div></div></div>
<p class="small">Each configuration on its own task-type-stratified training sample under the controlled c16 runtime (SemFilter v2); error bars = stratified SE; latency = episode wall + retrieval. Thick outline = 3-objective relaxed Pareto front (success, ln tokens, ln latency; full configurations); dashed = the 2-D front of each panel; ringed = the deployed Opt-OneConfig picks. Hover for round, slot / channel, knobs and steps.</p></div>
<div class="fd-card" id="fd-pqs" hidden><h4 id="fd-pqs-t">Opt-PerQuery weight sweep (test)</h4><div id="fd-pqs-b"></div></div>
<div class="fd-card" id="fd-cv" hidden><h4>ALFWorld · v7 re-test, outer 5-fold CV (secondary)</h4>
<p class="small">The same arms under outer 5-fold cross-validation: each fold's selection / router is fitted on 4/5 of the training queries and evaluated on the same 200 test tasks; mean ± std across the folds, SE<sub>tasks</sub> = mean within-fold SE. Per-fold paired differences below (a − b).</p>
<div class="fd-wrap" id="fd-cv-table"></div></div>
<div class="fd-card" id="fd-steps" hidden><h4 id="fd-st-t">Mean agent steps per test arm</h4><div class="fd-leg" id="fd-st-ctl"></div><div id="fd-st"></div></div>
<div class="fd-card" id="fd-psteps" hidden><h4 id="fd-ps-t">ALFWorld profiling · mean agent steps per configuration</h4><div class="fd-leg" id="fd-ps-ctl"></div><div id="fd-ps"></div></div>
<div class="fd-card"><h4>数值</h4><div class="fd-wrap" id="fd-tab"></div></div>
</div>
<h3>Opt-PerQuery 的路由分布</h3>
<p class="small">左：五个数据集的测试题被路由到各计划的比例（颜色 = 计划）。右：当前数据集逐配置的题数与这些题的平均质量（质量是"被路由到该配置的题"的结果，不同配置的题集不同，不能直接横向比较配置好坏）。</p>
<div class="fd-grid">
<div class="fd-card"><h4>按计划的路由比例（全部数据集）</h4><div class="fd-leg" id="fd-rleg"></div><div id="fd-rall"></div></div>
<div class="fd-card"><h4 id="fd-rct">按配置</h4><div id="fd-rcfg"></div></div>
</div>
<h3>结论</h3><ol>__CONCLUSIONS__</ol>
<script id="fd-data" type="application/json">__DATA__</script>
<script>(()=>{const D=JSON.parse(document.getElementById('fd-data').textContent),$=id=>document.getElementById(id),
e=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const COL={NoMemory:'#8a8f87',VGR:'#2a6fb3','VGR+Rank':'#1f9aa6',P2:'#8f5bb5','P2+Rank':'#c0558a','BO-Qbest':'#c26a1b',ROUTER:'#2f8f6b'};
__FPJS____PQSJS__const FH=new Set(),VA=d=>({...d,arms:d.arms.filter(a=>a.kind!=='plan'||!FH.has(a.id))});
function fleg(d){const fx=d.arms.filter(a=>a.kind==='plan'),F=d.fixed;$('fd-fleg').innerHTML=(fx.length?'<span>Fixed-plan defaults (test):</span>'+fx.map(a=>`<label><input type="checkbox" data-fk="${e(a.id)}"${FH.has(a.id)?'':' checked'}> ${fpGlyph(a.plan,a.color)} ${e(a.short||a.label)}</label>`).join(''):'')+(F?`<span class="small">${e(F.line)}</span>`:'')}
const W=440,H=250,M={l:120,r:56,t:6,b:24},CO=a=>a.color||COL[a.id];
function hbars(el,arms,segs,fmt){const L=Math.max(M.l,...arms.map(a=>tw(a.label)+10)),n=arms.length,bh=Math.min(24,(H-M.t-M.b)/n-6),max=Math.max(...arms.map(a=>segs.reduce((s,g)=>s+g.v(a),0)))*1.08||1;
 const X=v=>L+v/max*(W-L-M.r);let s=`<svg viewBox="0 0 ${W} ${H}">`;
 [0,.5,1].forEach(f=>{s+=`<line x1="${X(max*f/1.08)}" x2="${X(max*f/1.08)}" y1="${M.t}" y2="${H-M.b}" stroke="var(--line)"/><text x="${X(max*f/1.08)}" y="${H-8}" text-anchor="middle">${fmt(max*f/1.08)}</text>`});
 arms.forEach((a,i)=>{const y=M.t+i*((H-M.t-M.b)/n)+3;let x0=0;s+=`<text x="${L-6}" y="${y+bh/2+4}" text-anchor="end">${e(a.label)}</text>`;
  segs.forEach(g=>{const v=g.v(a);if(v<=0)return;s+=`<rect x="${X(x0)}" y="${y}" width="${Math.max(X(x0+v)-X(x0)-1,1)}" height="${bh}" rx="2" fill="${g.c||CO(a)}"><title>${e(a.label)} · ${e(g.n)}: ${fmt(v)}</title></rect>`;x0+=v});
  s+=`<text class="fd-v" x="${X(x0)+4}" y="${y+bh/2+4}">${fmt(x0)}</text>`});
 el.innerHTML=s+'</svg>'}
function front(pts){return pts.filter(p=>!pts.some(q=>q!==p&&q.y>=p.y&&q.x<=p.x&&(q.y>p.y||q.x<p.x))).sort((a,b)=>a.x-b.x)}
function scatter(el,arms,xv,xl,fmt){const pts=arms.map(a=>({a,x:xv(a),y:a.quality}));const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y);
 let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);const px=(x1-x0)*.12||1,py=(y1-y0)*.15||.02;x0-=px;x1+=px;y0-=py;y1+=py;
 const m={l:44,r:14,t:10,b:34},X=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),Y=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
 let s=`<svg viewBox="0 0 ${W} ${H}">`;[y0,(y0+y1)/2,y1].forEach(v=>{s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${m.l-5}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`});
 [x0,(x0+x1)/2,x1].forEach((v,i)=>{s+=`<text x="${X(v)}" y="${H-m.b+16}" text-anchor="${['start','middle','end'][i]}">${fmt(v)}</text>`});
 s+=`<text x="${(W+m.l)/2}" y="${H-1}" text-anchor="middle">${e(xl)}（越左越好）</text>`;
 const f=front(pts);s+=`<polyline fill="none" stroke="var(--muted)" stroke-dasharray="4 3" stroke-width="1.4" points="${f.map(p=>X(p.x)+','+Y(p.y)).join(' ')}"/>`;
 const lab=[];pts.forEach(p=>{const on=f.includes(p),cx=X(p.x),cy=Y(p.y);lab.push({x:cx,y:cy,t:p.a.label});
  s+=p.a.kind==='plan'?fpShape(p.a.plan,cx,cy,on?7.5:6.5,`fill="${on?CO(p.a):'var(--panel)'}" stroke="${CO(p.a)}" stroke-width="2"`,`<title>${e(p.a.label)}\nquality ${p.y.toFixed(3)} ± ${(p.a.quality_se||0).toFixed(3)} SE · ${e(xl)} ${fmt(p.x)}${p.a.gain!=null?'\ngain vs No-Memory '+(p.a.gain>=0?'+':'')+p.a.gain.toFixed(3):''}${p.a.steps!=null?'\nsteps / episode '+p.a.steps.toFixed(2):''}${on?'\nPareto 前沿':''}</title>`):`<circle cx="${cx}" cy="${cy}" r="${on?7:5.5}" fill="${on?CO(p.a):'var(--panel)'}" stroke="${CO(p.a)}" stroke-width="2"><title>${e(p.a.label)}\n质量 ${p.y.toFixed(3)}${p.a.quality_std!=null?' ± '+p.a.quality_std.toFixed(3)+' std across folds, SE_tasks '+p.a.quality_se.toFixed(3):' ± '+p.a.quality_se.toFixed(3)+' SE'} · ${e(xl)} ${fmt(p.x)}${p.a.steps!=null?'\nsteps / episode '+p.a.steps.toFixed(2)+(p.a.steps_success!=null?' · on successes '+p.a.steps_success.toFixed(2):''):''}${on?'\nPareto 前沿':''}</title></circle>`});
 s+=placeLabels(lab,m);el.innerHTML=s+'</svg>'}
// Greedy collision-free labels (same idea as the profiling section): candidates around each point, scored by
// overlaps with markers / placed labels / plot edge, then distance; a thin leader line when placed far away.
function tw(t){let w=0;for(const ch of t)w+=ch.charCodeAt(0)>255?10.5:6;return w}
function placeLabels(items,m){const boxes=items.map(o=>({x1:o.x-7,x2:o.x+7,y1:o.y-7,y2:o.y+7}));let out='';
 const hit=(b)=>boxes.reduce((n,o)=>n+(b.x1<o.x2&&o.x1<b.x2&&b.y1<o.y2&&o.y1<b.y2?1:0),0);
 items.slice().sort((a,b)=>a.y-b.y).forEach(l=>{const w=tw(l.t);let best=null,bs=1e9;
  [1,2.5,4.5].forEach(dm=>[-5,-4,-3,-2,-1,0,1,2,3,4,5].forEach(sy=>[1,-1,0].forEach(sx=>{if(!sx&&!sy)return;
   const ax=sx>0?'start':sx<0?'end':'middle',x=l.x+sx*11*dm,y=l.y+4+sy*13,x1=ax==='start'?x:ax==='end'?x-w:x-w/2,b={x1,x2:x1+w,y1:y-11,y2:y+3};
   const own=boxes.filter(o=>!(Math.abs((o.x1+o.x2)/2-l.x)<1&&Math.abs((o.y1+o.y2)/2-l.y)<1)).reduce((n,o)=>n+(b.x1<o.x2&&o.x1<b.x2&&b.y1<o.y2&&o.y1<b.y2?1:0),0);
   const sc=(b.x1<m.l+2||b.x2>W-2||b.y1<1||b.y2>H-m.b-4?1e5:0)+100*own+0.05*Math.hypot(x-l.x,y-4-l.y);
   if(sc<bs){bs=sc;best={ax,x,y,b}}})));
  boxes.push(best.b);const lead=Math.hypot(best.x-l.x,best.y-4-l.y)>22?`<line x1="${l.x}" y1="${l.y}" x2="${best.ax==='end'?best.b.x2:best.ax==='start'?best.b.x1:best.x}" y2="${best.y-4}" stroke="var(--muted)" stroke-width=".6"/>`:'';
  out+=lead+`<text class="fd-v" x="${best.x}" y="${best.y}" text-anchor="${best.ax}">${e(l.t)}</text>`});return out}
const k=v=>v>=1000?(v/1000).toFixed(v>=1e5?0:1)+'k':v.toFixed(0),sec=v=>v.toFixed(v<10?1:0),q3=v=>v.toFixed(3);
const TLEG=$('fd-tleg').innerHTML,f2=v=>v==null?'—':v.toFixed(2),sgn=v=>(v>=0?'+':'')+v.toFixed(3);
// ALFWorld v7 re-test (alf_pool_page, user 2026-10-22): deployed full-train arms PRIMARY (± SE across tasks);
// outer-CV mean ± std across folds secondary (columns "CV"); references single run.
function alfTable(d){const ms=(m,f)=>!m||m.mean==null?'—':f(m.mean)+(m.std!=null?' ± '+f(m.std).replace(/^\+/,''):''),F=v=>Math.round(v).toLocaleString('en-US'),cv=a=>a.cv||{},cm=(a,k,f)=>{const c=cv(a);return c[k]==null?'—':f(c[k])+(c[k+'_std']!=null?' ± '+f(c[k+'_std']).replace(/^\+/,''):'')};
 if(!d.cv)return sciTable(d);
 return `<table><thead><tr><th>Method</th><th>success ± SE</th><th>gain vs No-Memory ± SE</th><th>tokens / episode</th><th>latency s (wall)</th><th>latency s (LLM span)</th><th>steps ± SE</th><th>steps (succ.)</th><th>configs</th><th>retrieval s</th><th>retrieval LLM tok</th><th>CV success (mean ± std)</th><th>CV gain</th><th>CV tokens</th><th>CV latency s (wall)</th><th>CV steps</th></tr></thead><tbody>${d.arms.map(a=>`<tr><td>${e(a.label)}${a.kind==='ref'?' <span class="small">(reference)</span>':a.kind==='plan'?' <span class="small">(fixed plan)</span>':''}</td><td>${q3(a.quality)} ± ${q3(a.quality_se)}</td><td>${a.gain==null||a.id==='NoMemory'?'—':sgn(a.gain)+' ± '+q3(a.gain_se)}</td><td>${F(a.tokens)}</td><td>${a.latency_wall==null?'—':a.latency_wall.toFixed(1)}</td><td>${a.latency.toFixed(1)}</td><td>${f2(a.steps)}${a.steps_se!=null?' ± '+f2(a.steps_se):''}</td><td>${f2(a.steps_success)}</td><td>${a.distinct??'—'}</td><td>${a.ret.toFixed(1)}</td><td>${F(a.tsem)}</td><td>${cm(a,'q',q3)}</td><td>${cm(a,'gain',sgn)}</td><td>${cm(a,'tok',F)}</td><td>${cm(a,'lat_wall',v=>v.toFixed(1))}</td><td>${cm(a,'steps',f2)}</td></tr>`).join('')}</tbody></table><p class="small">Primary = deployed arms (selection / router fitted on all training queries; ± = SE across the ${d.n} test tasks). CV = the same arm under outer ${d.cv?d.cv.k:5}-fold cross-validation (each fold's selection / router fitted on 4/5 of the training queries and evaluated on the same ${d.n} tasks; mean ± std across folds). Plots use the LLM-span latency (like-for-like with the original No-Memory run).</p>`}
// ScienceWorld v7 (sci_pool_page, user 2026-10-22): deployed arms only (no outer CV); plan shares of each routed arm.
function sciTable(d){const F=v=>Math.round(v).toLocaleString('en-US'),sh=a=>a.plan_shares?Object.entries(a.plan_shares).map(([p,v])=>`${e(PN[p]||p)} ${(100*v).toFixed(0)}%`).join(', '):'—';
 return `<table><thead><tr><th>Method</th><th>${e(d.metric)} ± SE</th><th>gain vs No-Memory ± SE</th><th>tokens / episode</th><th>latency s (wall)</th><th>latency s (LLM span)</th><th>steps ± SE</th><th>steps (succ.)</th><th>configs</th><th>routing shares (plan)</th><th>retrieval s</th><th>retrieval LLM tok</th></tr></thead><tbody>${d.arms.map(a=>`<tr><td>${e(a.label)}${a.kind==='ref'?' <span class="small">(reference)</span>':a.kind==='plan'?' <span class="small">(fixed plan)</span>':''}</td><td>${q3(a.quality)} ± ${q3(a.quality_se)}</td><td>${a.gain==null||a.id==='NoMemory'?'—':sgn(a.gain)+' ± '+q3(a.gain_se)}</td><td>${F(a.tokens)}</td><td>${a.latency_wall==null?'—':a.latency_wall.toFixed(1)}</td><td>${a.latency.toFixed(1)}</td><td>${f2(a.steps)}${a.steps_se!=null?' ± '+f2(a.steps_se):''}</td><td>${f2(a.steps_success)}</td><td>${a.distinct??'—'}</td><td style="text-align:left;white-space:normal">${a.kind==='ref'?'—':sh(a)}</td><td>${a.ret.toFixed(1)}</td><td>${F(a.tsem)}</td></tr>`).join('')}</tbody></table><p class="small">Deployed arms only (selection / router fitted on all training queries; ± = SE across the ${d.n} test tasks); no outer cross-validation was run for this dataset. Plots use the LLM-span latency (like-for-like with the original No-Memory run).</p>`}
function pqsCard(d){const S=d.pqs;$('fd-pqs').hidden=!S;if(!S)return;$('fd-pqs-t').textContent=`${d.name} · Opt-PerQuery (BRANE) weight sweep on the test set — Q–T and Q–L curves, ${S.n_weights||11} weights (w_Q, w_L, w_T)`;$('fd-pqs-b').innerHTML=pqsHTML(S,'fd');pqsDraw(S,'fd')}
function cvCard(d){$('fd-cv').hidden=!d.cv;if(!d.cv)return;const C=d.cv,ms=(x,f)=>x&&x.mean!=null?f(x.mean)+' ± '+f(x.std).replace(/^\+/,''):'—',F=v=>(v>=0?'+':'')+Math.round(v).toLocaleString('en-US'),pm=(v,s,f)=>v==null?'—':f(v)+(s!=null?' ± '+f(s).replace(/^\+/,''):'');
 $('fd-cv-table').innerHTML=`<table><thead><tr><th>arm (outer ${C.k}-fold CV)</th><th>success</th><th>SE<sub>tasks</sub></th><th>gain vs No-Memory</th><th>tokens</th><th>latency s (wall)</th><th>latency s (span)</th><th>steps</th><th>steps (succ.)</th></tr></thead><tbody>${C.cv_arms.map(a=>`<tr><td>${e(a.label)}</td><td>${pm(a.quality,a.quality_std,q3)}</td><td>${q3(a.quality_se)}</td><td>${pm(a.gain,a.gain_std,sgn)}</td><td>${pm(a.tokens,a.tokens_std,v=>Math.round(v).toLocaleString('en-US'))}</td><td>${pm(a.latency_wall,a.latency_wall_std,v=>v.toFixed(1))}</td><td>${pm(a.latency,a.latency_std,v=>v.toFixed(1))}</td><td>${pm(a.steps,a.steps_std,f2)}</td><td>${pm(a.steps_success,a.steps_success_std,f2)}</td></tr>`).join('')}</tbody></table>
<h4 style="margin-top:10px">Per-fold paired differences (a − b; mean ± std across the ${C.k} folds, same 200 test tasks)</h4><table><thead><tr><th>comparison (a − b)</th><th>Δ success</th><th>Δ tokens</th><th>Δ latency s</th><th>Δ steps</th><th>Δ gain</th></tr></thead><tbody>${C.comparison.map(c=>`<tr><td>${e(c.a)} − ${e(c.b)}</td><td>${ms(c.diff.quality,sgn)}</td><td>${ms(c.diff.tokens,F)}</td><td>${ms(c.diff.latency_s,v=>(v>=0?'+':'')+v.toFixed(1))}</td><td>${ms(c.diff.steps,v=>(v>=0?'+':'')+v.toFixed(2))}</td><td>${ms(c.diff.paired_gain_vs_nomemory,sgn)}</td></tr>`).join('')}</tbody></table>`}
// ALFWorld training scatter (v7 rounds 6-21; alf_pool_page.train_panel): success × tokens / latency, ± SE bars,
// relaxed 3-D front (thick outline), 2-D front per panel (dashed), Opt-OneConfig picks, slot = marker shape.
const TPC={VGR:'#2a6fb3','VGR+Rank':'#1f9aa6',P2:'#8f5bb5','P2+Rank':'#c0558a','P1+Rank':'#b8862b',P1:'#6b8e23'},TPN={VGR:'VGR','VGR+Rank':'VGR+Rank',P2:'VGR+SemFilter','P2+Rank':'VGR+SemFilter+Rank','P1+Rank':'V+SemFilter+G+Rank',P1:'V+SemFilter+G'},
TSL={sobol:'Sobol slot',llm:'LLM-analyst slot',dls:'DLS slot',xover:'crossover slot'},TH=new Set(),tslot=p=>(p.channel||'').replace(/-slot$/,''),tsh=(sl,x,y,r)=>sl==='llm'?`<rect x="${x-r}" y="${y-r}" width="${2*r}" height="${2*r}"`:sl==='dls'?`<path d="M${x} ${y-r*1.2}L${x+r*1.1} ${y+r*.8}L${x-r*1.1} ${y+r*.8}Z"`:sl==='xover'?`<path d="M${x} ${y-r*1.25}L${x+r*1.25} ${y}L${x} ${y+r*1.25}L${x-r*1.25} ${y}Z"`:`<circle cx="${x}" cy="${y}" r="${r}"`;
function trainCard(d){const P=d.train_scatter;$('fd-train').hidden=!P;if(!P)return;
 $('fd-train-t').textContent=`${d.name} · training profiling, rounds ${P.rounds[0]}–${P.rounds[1]}: ${P.points.length} configurations (${P.n_full} full on their own ${P.n_queries}-task stratified sample, ${P.points.length-P.n_full} pruned at 50) + No-Memory (${P.nm_label||'c16 rerun'})`;
 const ic=(sl,c,f)=>`<svg width="14" height="14" viewBox="0 0 14 14" style="display:inline-block;width:14px;height:14px;vertical-align:-2px">${tsh(sl,7,7,4.5)} fill="${f?c:'none'}" stroke="${c}" stroke-width="1.5"/></svg>`,cnt=(f,v)=>P.points.filter(p=>f(p)===v).length;
 $('fd-train-leg').innerHTML=Object.keys(TPC).filter(p=>cnt(x=>x.plan,p)).map(p=>`<label><input type="checkbox" data-tk="p:${p}"${TH.has('p:'+p)?'':' checked'}> ${ic('sobol',TPC[p],1)} ${TPN[p]} (${cnt(x=>x.plan,p)})</label>`).join('')+Object.keys(TSL).map(t=>`<label><input type="checkbox" data-tk="s:${t}"${TH.has('s:'+t)?'':' checked'}> ${ic(t,'var(--fg)',0)} ${TSL[t]} (${cnt(tslot,t)})</label>`).join('')+`<label><input type="checkbox" data-tk="err"${TH.has('err')?'':' checked'}> ± SE bars</label><span>${ic('sobol','var(--muted)',1)} full · ${ic('sobol','var(--muted)',0)} pruned at 50 (partial-n mean) · thick outline = 3-D relaxed front · dashed = 2-D front · ring = Opt-OneConfig pick · ✚ No-Memory (${e(P.nm_label||'c16 rerun')})</span>`;
 $('fd-train-leg').querySelectorAll('input').forEach(el=>el.addEventListener('change',()=>{el.checked?TH.delete(el.dataset.tk):TH.add(el.dataset.tk);trainCard(d)}));
 $('fd-tr-tok').innerHTML=tscat(P,'tok','tokens / episode (log)');$('fd-tr-lat').innerHTML=tscat(P,'lat','latency s / episode, c16 (log)')}
function tscat(P,xk,xl){const W2=520,H2=330,L=50,R=12,T=10,B=40,eb=!TH.has('err'),pts=P.points.filter(p=>!TH.has('p:'+p.plan)&&!TH.has('s:'+tslot(p))),nm=P.nomem;
 const xs=pts.flatMap(p=>[p[xk]-(eb?p[xk+'_se']||0:0),p[xk]+(eb?p[xk+'_se']||0:0)]).concat([nm[xk]]).filter(v=>v>0),ys=pts.flatMap(p=>[p.q-(eb?p.q_se:0),Math.min(1,p.q+(eb?p.q_se:0))]).concat([nm.q]);
 if(!pts.length)return '<p class="small">No configuration visible.</p>';
 const x0=Math.log(Math.min(...xs)*.93),x1=Math.log(Math.max(...xs)*1.07),y0=Math.max(0,Math.min(...ys)-.02),y1=Math.min(1.005,Math.max(...ys)+.01),X=v=>L+(Math.log(Math.max(v,1e-9))-x0)/(x1-x0)*(W2-L-R),Y=v=>T+(y1-v)/(y1-y0)*(H2-T-B);
 let tk=[];for(let k=Math.floor(x0/Math.LN10)-1;k<=Math.ceil(x1/Math.LN10)+1;k++)for(const m of [1,1.5,2,3,4,5,6,8])tk.push(m*10**k);tk=tk.filter(v=>Math.log(v)>=x0&&Math.log(v)<=x1);while(tk.length>7)tk=tk.filter((_,i)=>i%2===0);
 let s=`<svg viewBox="0 0 ${W2} ${H2}">`;for(const v of tk)s+=`<line x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H2-B}" stroke="var(--line)"/><text x="${X(v)}" y="${H2-B+14}" text-anchor="middle">${v>=1000?(v/1000)+'k':v}</text>`;
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;s+=`<line x1="${L}" x2="${W2-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${L-5}" y="${Y(v)+4}" text-anchor="end">${v.toFixed(2)}</text>`}
 s+=`<text x="${(L+W2-R)/2}" y="${H2-6}" text-anchor="middle">${e(xl)}</text><text transform="translate(12,${(T+H2-B)/2}) rotate(-90)" text-anchor="middle">${e(P.qname&&P.qname!=='success'?P.qname:'success rate')} (train, own sample)</text>`;
 const c=pts.filter(p=>!p.partial).sort((a,b)=>a[xk]-b[xk]||b.q-a.q),f2d=[];let bq=-1;for(const p of c)if(p.q>bq){f2d.push(p);bq=p.q}
 if(f2d.length){let dd=`M${X(f2d[0][xk])} ${Y(f2d[0].q)}`;for(let i=1;i<f2d.length;i++)dd+=`H${X(f2d[i][xk])}V${Y(f2d[i].q)}`;s+=`<path d="${dd}" fill="none" stroke="var(--fg)" stroke-width="1.2" stroke-dasharray="4 3" opacity=".7"/>`}
 const ord=pts.slice().sort((a,b)=>(!a.partial)-(!b.partial));
 if(eb)for(const p of ord){const col=TPC[p.plan]||'#888',x=X(p[xk]),y=Y(p.q),se=p[xk+'_se']||0;s+=`<line x1="${X(Math.max(p[xk]-se,Math.exp(x0)))}" x2="${X(p[xk]+se)}" y1="${y}" y2="${y}" stroke="${col}" opacity=".45"/><line x1="${x}" x2="${x}" y1="${Y(Math.max(p.q-p.q_se,y0))}" y2="${Y(Math.min(p.q+p.q_se,y1))}" stroke="${col}" opacity=".45"/>`}
 const picks=P.points.filter(p=>p.slo&&p.slo.length);
 for(const p of ord){const col=TPC[p.plan]||'#888',x=X(p[xk]),y=Y(p.q),r=p.slo&&p.slo.length?5.5:4.5,fr=p.front3&&!p.partial;
  s+=`${tsh(tslot(p),x,y,r)} fill="${p.partial?'none':col}" stroke="${fr?'var(--fg)':p.partial?col:'var(--panel)'}" stroke-width="${fr?2.2:1.5}"><title>${e(p.id)} · round ${p.round} · ${e(TPN[p.plan]||p.plan)} · ${e(p.channel||'')}${p.source?' ('+e(p.source)+')':''}\n${p.partial?'pruned at '+p.n+' tasks (partial-n mean)':'full, n = '+p.n}${fr?' · 3-D relaxed front':''}\nsuccess ${q3(p.q)} ± ${q3(p.q_se)} · gain vs No-Memory ${sgn(p.gain)}\ntokens ${k(p.tok)} ± ${k(p.tok_se||0)} · latency ${sec(p.lat)} ± ${sec(p.lat_se||0)} s\nsteps ${f2(p.steps)} (successes ${f2(p.steps_success)})\n${e(p.knobs||'')}${p.slo&&p.slo.length?'\nOpt-OneConfig '+p.slo.join(', '):''}</title></${tslot(p)==='llm'?'rect':tslot(p)==='dls'||tslot(p)==='xover'?'path':'circle'}>`;
  if(p.slo&&p.slo.length){const j=picks.indexOf(p);s+=`<circle cx="${x}" cy="${y}" r="10.5" fill="none" stroke="var(--fg)" stroke-width="1.5"/><line x1="${x+8}" y1="${y+(j?8:-8)}" x2="${x+24}" y2="${y+(j?24:-4)}" stroke="var(--fg)"/><text class="fd-v" x="${x+26}" y="${y+(j?30:-2)}">${e(p.id)} ${e(p.slo.join('/'))}</text>`}}
 const nx=X(nm[xk]),ny=Y(nm.q);s+=`<path d="M${nx-6} ${ny}H${nx+6}M${nx} ${ny-6}V${ny+6}" stroke="var(--fg)" stroke-width="3"><title>No-Memory (${e(P.nm_label||'c16 rerun')}, own 200-task sample)\nsuccess ${q3(nm.q)} ± ${q3(nm.q_se)} · tokens ${k(nm.tok)} · latency ${sec(nm.lat)} s · steps ${f2(nm.steps)}</title></path><text class="fd-v" x="${nx+9}" y="${ny+14}">No-Memory</text>`;
 return s+'</svg>'}
function reliability(d){$('fd-reliability').hidden=d.key!=='tau2';
 if(d.key!=='tau2')return;
 const pct=v=>Number.isFinite(v)?(100*v).toFixed(1):'—',pv=p=>p==null?'—':p<1e-3?p.toExponential(1):p.toFixed(3),sg=v=>(v>=0?'+':'')+(100*v).toFixed(1);
 $('fd-reliability-table').innerHTML=`<table><thead><tr><th>Method</th><th>pass<sup>1</sup> ± SE</th><th>Δ vs No-Memory ± SE (pp)</th><th>W / L</th><th>p (exact)</th><th>p (normal)</th><th>Tokens / episode</th><th>Latency s / episode</th><th>4-trial pass<sup>1</sup> (context)</th></tr></thead><tbody>${d.arms.map(a=>`<tr><td>${e(a.label)}</td><td>${pct(a.quality)}% ± ${pct(a.quality_se)}</td><td>${a.gain==null?'—':sg(a.gain)+' ± '+pct(a.gain_se)}</td><td>${a.wins==null?'—':a.wins+' / '+a.losses}</td><td>${pv(a.p_sign)}</td><td>${pv(a.p_z)}</td><td>${Math.round(a.tokens).toLocaleString('en-US')}</td><td>${a.latency.toFixed(1)}</td><td>${a.all4&&a.all4.pass1!=null?pct(a.all4.pass1)+'%':'—'}</td></tr>`).join('')}</tbody></table>`}
// Steps (user 2026-10-21). Test: mean agent steps per test arm (CV arms: whisker = ± std across folds; references
// single run, also dashed lines). Profiling (ALFWorld v7): mean environment steps per configuration ± SE across tasks,
// ◇ = on successful episodes, sorted by mean steps, lighter dashed = pruned / partial, grey = No-Memory.
let stSucc=false;const SH=new Set(),son=x=>!SH.has(x),SPC={VGR:'#2a6fb3','VGR+Rank':'#1f9aa6',P2:'#8f5bb5','P2+Rank':'#c0558a','P1+Rank':'#b8862b',P1:'#6b8e23'},
SPN={VGR:'VGR','VGR+Rank':'VGR+Rank',P2:'VGR+SemFilter','P2+Rank':'VGR+SemFilter+Rank','P1+Rank':'V+SemFilter+G+Rank',P1:'V+SemFilter+G'},
scb=(k,t)=>`<label><input type="checkbox" data-sk="${e(k)}"${son(k)?' checked':''}> ${t}</label>`,wrap=(s,w)=>`<div class="fd-wrap">${s.replace('<svg ',`<svg style="min-width:${w}px" `)}</div>`;
function hstep(rows,unit,o){const SW=880,rh=o.rh||20,mt=8,mb=36,ml=Math.min(340,Math.max(...rows.map(r=>tw(r.label)))+12),mr=58,H=mt+mb+rows.length*rh,
 mx=Math.max(...rows.map(r=>Math.max(r.v+(r.err||0),r.m||0)))*1.07,X=v=>ml+v/mx*(SW-ml-mr),tk=mx>40?10:mx>16?5:2;let s=`<svg viewBox="0 0 ${SW} ${H}" role="img" aria-label="${e(unit)}">`;
 for(let v=0;v<=mx;v+=tk)s+=`<line x1="${X(v)}" x2="${X(v)}" y1="${mt}" y2="${H-mb}" stroke="var(--line)"/><text x="${X(v)}" y="${H-mb+14}" text-anchor="middle">${v}</text>`;
 s+=`<text x="${(SW+ml)/2}" y="${H-4}" text-anchor="middle">${e(unit)}</text>`;(o.lines||[]).forEach(l=>{s+=`<line x1="${X(l.v)}" x2="${X(l.v)}" y1="${mt}" y2="${H-mb}" stroke="${l.c}" stroke-dasharray="6 4" stroke-width="1.3"><title>${e(l.t)}</title></line>`});
 rows.forEach((r,i)=>{const y=mt+i*rh+2,bh=rh-6;s+=`<text x="${ml-6}" y="${y+bh-2}" text-anchor="end"${r.bold?' style="font-weight:600"':''}>${e(r.label)}</text><rect x="${ml}" y="${y}" width="${Math.max(X(r.v)-ml,1)}" height="${bh}" rx="2" fill="${r.c}" fill-opacity="${r.light?.3:r.ref?.45:.85}" stroke="${r.c}"${r.light?' stroke-dasharray="3 2"':''}><title>${e(r.t)}</title></rect>`;
  if(r.err!=null){const a=X(Math.max(r.v-r.err,0)),b=X(r.v+r.err),ym=y+bh/2;s+=`<path d="M${a} ${ym}H${b}M${a} ${ym-3}V${ym+3}M${b} ${ym-3}V${ym+3}" stroke="var(--fg)" fill="none"/>`}
  if(r.m!=null){const cx=X(r.m),cy=y+bh/2;s+=`<path d="M${cx} ${cy-5}L${cx+5} ${cy}L${cx} ${cy+5}L${cx-5} ${cy}Z" fill="var(--panel)" stroke="var(--fg)" stroke-width="1.3"><title>${e(r.t)}</title></path>`}
  s+=`<text class="fd-v" x="${Math.max(X(r.v+(r.err||0)),r.m!=null?X(r.m)+5:0)+5}" y="${y+bh-2}">${f2(r.v)}</text>`});return wrap(s+'</svg>',620)}
function stepsCard(d){const arms=d.arms.filter(a=>a.steps!=null);$('fd-steps').hidden=!arms.length;if(!arms.length)return;const su=stSucc;
 $('fd-st-t').textContent=`${d.name} · mean agent steps per test arm${su?' (successful episodes only)':''}`;
 $('fd-st-ctl').innerHTML=`<label><input type="checkbox" data-sk="t:succ"${stSucc?' checked':''}> successful episodes only</label>`+`<span>${d.primary==='deployed'?'deployed arms: whisker = ± SE across the test tasks':'whisker = ± std across the '+(d.cv?d.cv.k:'')+' CV folds'}; references single run (dashed lines)</span>`;
 const rows=arms.map(a=>{const v=su?a.steps_success:a.steps,sd=su?a.steps_success_std:a.steps_std,ref=a.kind==='ref';
  return v==null?null:{label:a.label,v,err:a.kind==='ref'?null:(sd!=null?sd:(su?null:a.steps_se)),c:a.color||COL[a.id]||'#888',ref,t:`${a.label}\nsteps / episode ${f2(a.steps)}${a.steps_std!=null?' ± '+f2(a.steps_std)+' std across folds':''}${a.steps_se!=null?' · SE_tasks '+f2(a.steps_se):''}\non successful episodes ${f2(a.steps_success)}${a.steps_success_std!=null?' ± '+f2(a.steps_success_std):''}\nsuccess ${q3(a.quality)}`}}).filter(Boolean);
 $('fd-st').innerHTML=hstep(rows,(d.key==='tau2'?'agent LLM calls':'environment steps')+' / episode',{lines:rows.filter(r=>r.ref).map(r=>({v:r.v,c:r.c,t:r.label+' '+f2(r.v)}))})}
function profSteps(d){const P=d.train_steps;$('fd-psteps').hidden=!P;if(!P)return;const nm=P.nomem,plans=[...new Set(P.points.map(p=>p.plan))];
 $('fd-ps-t').textContent=`${d.name} profiling (${P.panel}) · mean agent steps per configuration (${P.points.length} configurations + No-Memory, sorted by mean steps)`;
 $('fd-ps-ctl').innerHTML=plans.map(p=>scb('p:'+p,`<i style="background:${SPC[p]||'#888'}"></i>${e(SPN[p]||p)}`)).join('')+scb('succ','◇ on successful episodes')+scb('nm','No-Memory')+'<span>bar = mean over the observed tasks, whisker = ± SE across tasks; lighter dashed = pruned / partial</span>';
 const rows=(son('nm')&&nm.steps!=null?[{label:'No-Memory',bold:1,v:nm.steps,err:nm.steps_se,m:son('succ')?nm.steps_success:null,c:'#8a8f87',t:`No-Memory (training sample)\nsuccess ${q3(nm.q)} ± ${q3(nm.q_se)} · tokens ${k(nm.tok)} / episode\nsteps ${f2(nm.steps)} ± ${f2(nm.steps_se)} SE · on successes ${f2(nm.steps_success)}`}]:[])
  .concat(P.points.filter(p=>son('p:'+p.plan)).sort((a,b)=>a.steps-b.steps||a.id.localeCompare(b.id)).map(p=>({label:`${p.id} · ${SPN[p.plan]||p.plan}${p.partial?` (${p.status}, n ${p.n})`:''}`,v:p.steps,err:p.steps_se,m:son('succ')?p.steps_success:null,c:SPC[p.plan]||'#888',light:p.partial,
   t:`${p.id} · round ${p.round} · ${SPN[p.plan]||p.plan} · ${p.status}${p.pruned_at?' at stage '+p.pruned_at:''}\n${p.knobs}\nn = ${p.n} / ${P.n_queries} tasks\nsuccess ${q3(p.q)} ± ${q3(p.q_se)} · tokens ${k(p.tok)} / episode · latency ${sec(p.lat)} s\nsteps ${f2(p.steps)} ± ${f2(p.steps_se)} SE · on successes ${f2(p.steps_success)}`})));
 $('fd-ps').innerHTML=rows.length?hstep(rows,P.unit+' (sorted by mean steps)',{rh:17,lines:son('nm')&&nm.steps!=null?[{v:nm.steps,c:'#8a8f87',t:'No-Memory '+f2(nm.steps)}]:[]}):'<p class="small">No configuration visible.</p>'}
$('four-ds').addEventListener('change',ev=>{const c=ev.target;if(!c.dataset||!c.dataset.sk)return;const key=c.dataset.sk;
 if(key==='t:succ')stSucc=c.checked;else if(c.checked)SH.delete(key);else SH.add(key);
 const d=VA(D.datasets.find(x=>$('fd-tabs').querySelector(`button[data-k="${x.key}"][aria-pressed="true"]`))||D.datasets[0]);stepsCard(d);profSteps(d)});
$('fd-fleg').addEventListener('change',ev=>{const c=ev.target;if(!c.dataset||!c.dataset.fk)return;if(c.checked)FH.delete(c.dataset.fk);else FH.add(c.dataset.fk);
 const d=D.datasets.find(x=>$('fd-tabs').querySelector(`button[data-k="${x.key}"][aria-pressed="true"]`))||D.datasets[0];show(d.key)});
function show(key){const d0=D.datasets.find(x=>x.key===key)||D.datasets[0],d=VA(d0);fleg(d0);$('fd-tabs').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.k===d.key));reliability(d);
 $('fd-qt').textContent=`质量（${d.metric}，${d.n} task × ${d.trials} trial）`;
 $('fd-status').textContent=d.note?d.note:d.pending.length?'尚无完整 4-trial 测试：'+d.pending.join('、')+'。旧单次结果已退出本节。':'';
 hbars($('fd-q'),d.arms,[{n:'质量',v:a=>a.quality}],q3);
 hbars($('fd-l'),d.arms,[{n:'检索',v:a=>a.ret,c:'#c26a1b'},{n:'生成',v:a=>a.gen,c:'#5b7290'}],sec);
 if(d.io_merged){$('fd-tleg').innerHTML='<span><i style="background:#5b7290"></i>agent (input + output)</span><span><i style="background:#8f5bb5"></i>retrieval-side LLM (SemFilter)</span>';hbars($('fd-t'),d.arms,[{n:'agent (input + output)',v:a=>a.tin,c:'#5b7290'},{n:'retrieval-side LLM',v:a=>a.tsem,c:'#8f5bb5'}],k)}
 else{$('fd-tleg').innerHTML=TLEG;hbars($('fd-t'),d.arms,[{n:'输入',v:a=>a.tin,c:'#5b7290'},{n:'输出',v:a=>a.tout,c:'#2f8f6b'},{n:'检索侧 LLM',v:a=>a.tsem,c:'#8f5bb5'}],k)}
 cvCard(d);pqsCard(d0);trainCard(d);stepsCard(d);profSteps(d);
 scatter($('fd-pl'),d.arms,a=>a.ret+a.gen,'延迟 s / 题',sec);scatter($('fd-pt'),d.arms,a=>a.tin+a.tout+a.tsem,'Token / 题',k);
 if(d.cv||d.primary==='deployed'){$('fd-tab').innerHTML=alfTable(d);routeCfg(d);return}
 $('fd-tab').innerHTML=`<table><thead><tr><th>方法</th><th>质量 ± SE</th><th>增益 ± SE</th><th>总延迟 s ± SE</th><th>总 Token ± SE</th><th>task / episode</th><th>检索 s</th><th>生成 s</th><th>输入 tok</th><th>输出 tok</th><th>检索 LLM tok</th></tr></thead><tbody>${d.arms.map(a=>`<tr><td>${e(a.label)}</td><td>${q3(a.quality)} ± ${q3(a.quality_se)}</td><td>${a.gain==null?'—':(a.gain>=0?'+':'')+a.gain.toFixed(3)+' ± '+a.gain_se.toFixed(3)}</td><td>${a.latency.toFixed(1)} ± ${a.latency_se.toFixed(1)}</td><td>${k(a.tokens)} ± ${k(a.tokens_se)}</td><td>${a.n} / ${a.n_episodes}</td><td>${a.ret.toFixed(1)}</td><td>${a.gen.toFixed(1)}</td><td>${Math.round(a.tin).toLocaleString()}</td><td>${Math.round(a.tout).toLocaleString()}</td><td>${Math.round(a.tsem).toLocaleString()}</td></tr>`).join('')}</tbody></table>`;
 routeCfg(d)}
const PC={VGR:'#2a6fb3','VGR+Rank':'#1f9aa6',P2:'#8f5bb5','P2+Rank':'#c0558a','P1+Rank':'#b8862b',P1:'#6b8e23',NoMemory:'#8a8f87'},PN={VGR:'VGR','VGR+Rank':'VGR+Rank',P2:'VGR+SemFilter','P2+Rank':'VGR+SemFilter+Rank','P1+Rank':'V+SemFilter+G+Rank',P1:'V+SemFilter+G',NoMemory:'No-Memory'};
function routeAll(){const ds=D.datasets.filter(d=>d.routing.length),bh=26,h=ds.length*(bh+14)+30,w=W,ml=96;let s=`<svg viewBox="0 0 ${w} ${h}">`;
 ds.forEach((d,i)=>{const y=8+i*(bh+14),tot=d.routing.reduce((a,r)=>a+r.n,0),by={};d.routing.forEach(r=>by[r.plan]=(by[r.plan]||0)+r.n);let x=ml;
  s+=`<text x="${ml-6}" y="${y+bh/2+4}" text-anchor="end">${e(d.name)}</text>`;
  Object.keys(PC).forEach(p=>{if(!by[p])return;const ww=by[p]/tot*(w-ml-10);s+=`<rect x="${x}" y="${y}" width="${Math.max(ww-1,1)}" height="${bh}" rx="2" fill="${PC[p]}"><title>${e(d.name)} · ${PN[p]}: ${by[p]} 题 (${(100*by[p]/tot).toFixed(0)}%)</title></rect>`;
   if(ww>34)s+=`<text x="${x+ww/2}" y="${y+bh/2+4}" text-anchor="middle" style="fill:#fff;font-size:10.5px">${(100*by[p]/tot).toFixed(0)}%</text>`;x+=ww});});
 [0,50,100].forEach(v=>{const x=ml+v/100*(w-ml-10);s+=`<text x="${x}" y="${h-6}" text-anchor="${v===100?'end':'middle'}">${v}%</text>`});
 $('fd-rall').innerHTML=s+'</svg>';$('fd-rleg').innerHTML=Object.keys(PC).map(p=>`<span><i style="background:${PC[p]}"></i>${PN[p]}</span>`).join('')}
function routeCfg(d){$('fd-rct').textContent=`按配置 · ${d.name}（${d.routing.reduce((a,r)=>a+r.n,0)} 题）`+(d.routing_arm?' · '+d.routing_arm:'');
 if(!d.routing.length){$('fd-rcfg').innerHTML='<p class="small">无路由结果</p>';return}
 const rows=d.routing,n=rows.length,bh=16,h=n*(bh+8)+28,ml=Math.min(W-150,Math.max(250,...rows.map(r=>tw((PN[r.plan]||r.plan)+' · '+r.config)+10))),max=Math.max(...rows.map(r=>r.n));let s=`<svg viewBox="0 0 ${W} ${h}">`;
 rows.forEach((r,i)=>{const y=6+i*(bh+8),ww=r.n/max*(W-ml-90);s+=`<text x="${ml-6}" y="${y+bh-3}" text-anchor="end">${e(PN[r.plan]||r.plan)} · ${e(r.config)}</text>
  <rect x="${ml}" y="${y}" width="${Math.max(ww,1)}" height="${bh}" rx="2" fill="${PC[r.plan]||'#888'}"><title>${e(r.config)}: ${r.n} 题，质量 ${r.quality.toFixed(3)}</title></rect>
  <text class="fd-v" x="${ml+ww+5}" y="${y+bh-3}">${r.n} · q ${r.quality.toFixed(2)}</text>`});
 $('fd-rcfg').innerHTML=s+'</svg>'}
routeAll();
$('fd-tabs').innerHTML=D.datasets.map(d=>`<button data-k="${d.key}">${e(d.name)}</button>`).join('');
$('fd-tabs').addEventListener('click',ev=>{const b=ev.target.closest('button');if(b)show(b.dataset.k)});show(D.datasets[0].key);})();</script>
<p class="small">生成时间 __TIME__ · trimem/experiments/render_atlas_results.py</p></section>
<!-- four-dataset-results:end -->"""


def conclusions(data: list[dict]) -> list[str]:
    out = []
    for d in data:
        v = next((a for a in d["arms"] if a["id"] == "VGR"), None)
        if (
            d["key"] not in ("tau2", "alfworld")
            and not (d["key"] == "sciworld" and d.get("primary") == "deployed")
            and not (d["key"] == "webshop" and "train_scatter" in d)
            and v
            and v.get("gain") is not None
        ):  # τ²: tau2_conclusions; ALFWorld: alf_conclusions; ScienceWorld v7: sci_conclusions
            out.append(
                f"{d['name']}：VGR 质量 {v['quality']:.4f}，相对 No-Memory 的配对增益 "
                f"{100 * v['gain']:+.2f} ± {100 * v['gain_se']:.2f} pp（SE，{v['n']} 个 task）。"
            )
    alf = next((d for d in data if d["key"] == "alfworld"), None)
    sci = next((d for d in data if d["key"] == "sciworld"), None)
    sci_v7 = sci is not None and "train_scatter" in sci
    if sci_v7:
        out = sci_conclusions(sci) + out
    ws = next((d for d in data if d["key"] == "webshop"), None)
    if ws is not None and "train_scatter" in ws:  # WebShop v7 (user 2026-10-23)
        out = ws_conclusions(ws) + out
    if alf:
        out = alf_conclusions(alf) + out
    tau = next(d for d in data if d["key"] == "tau2")
    out += tau2_conclusions(tau)
    for d in data:
        out += fixed_conclusions(d)
    out.append(
        "Pareto fronts reflect observed means, without a claim of statistically significant dominance. "
        "τ² reference arms (No-Memory, VGR, VGR+SemFilter) ran on 2026-10-19 and the Opt arms on 2026-10-21, both at "
        "40 concurrent episodes; latency includes queueing and the user simulator, so wall-time differences between "
        "arms are partly load. "
        + (
            "ScienceWorld shows the pool-strat200-v7 test (deployed arms only) and "
            if sci_v7
            else "ScienceWorld still shows the completed controlled-v1 single-run tests; "
        )
        + "ALFWorld shows the pool-strat200-v7 re-test at concurrency 16 in the controlled-v1 layout of its No-Memory "
        "/ VGR references (latency like-for-like in the LLM-call-span definition)."
    )
    return out


def alf_conclusions(d: dict) -> list[str]:
    """English ALFWorld bullets (v7 re-test: deployed arms primary, outer-CV secondary)."""
    arms = d["arms"]
    if not arms:
        return [d.get("note", "ALFWorld v7 re-test pending.")]
    opt = [a for a in arms if a.get("kind") in ("oneconfig", "perquery")]
    by = {a["id"]: a for a in arms}
    best = max(opt, key=lambda a: (a["quality"], -a["tokens"]))
    lo, hi = min(a["quality"] for a in opt), max(a["quality"] for a in opt)
    nm, vgr = by.get("NoMemory"), by.get("VGR")
    st_lo, st_hi = min(a["steps"] for a in opt), max(a["steps"] for a in opt)
    cvq = [a["cv"]["q"] for a in opt if a.get("cv")]
    return [
        f"ALFWorld (v7 re-test, catalog = profiling rounds 6–21 under the controlled c16 runtime; 200 tasks × 1 "
        f"run): the deployed Opt arms reach {lo:.3f}–{hi:.3f} success (No-Memory {nm['quality']:.3f}, VGR "
        f"{vgr['quality']:.3f}); best {best['label']} {best['quality']:.3f} ± {best['quality_se']:.3f} SE (gain "
        f"{best['gain']:+.3f} ± {best['gain_se']:.3f}). Outer 5-fold CV agrees ({min(cvq):.3f}–{max(cvq):.3f}, "
        "fold std ≤ 0.014). With SE ≈ 0.02 the Opt arms are not distinguishable from each other on success.",
        f"ALFWorld cost: Opt arms {min(a['tokens'] for a in opt):,.0f}–{max(a['tokens'] for a in opt):,.0f} tokens "
        f"and {min(a['latency_wall'] for a in opt):.1f}–{max(a['latency_wall'] for a in opt):.1f} s wall per "
        f"episode (VGR {vgr['tokens']:,.0f} tokens, {vgr['latency_wall']:.1f} s wall; LLM-span latency Opt "
        f"{min(a['latency'] for a in opt):.1f}–{max(a['latency'] for a in opt):.1f} s vs No-Memory "
        f"{nm['latency']:.1f} s and VGR {vgr['latency']:.1f} s). Steps: {st_lo:.2f}–{st_hi:.2f} environment steps "
        f"per episode vs No-Memory {nm['steps']:.2f} and VGR {vgr['steps']:.2f}.",
    ]


def sci_conclusions(d: dict) -> list[str]:
    """English ScienceWorld bullets (v7 test: deployed arms only)."""
    arms = d["arms"]
    if not arms:
        return [d.get("note", "ScienceWorld v7 test pending.")]
    opt = [a for a in arms if a.get("kind") in ("oneconfig", "perquery")]
    by = {a["id"]: a for a in arms}
    nm, vgr = by.get("NoMemory"), by.get("VGR")
    best = max(opt, key=lambda a: (a["quality"], -a["tokens"]))
    ref = (
        f" (No-Memory {nm['quality']:.3f}, VGR {vgr['quality']:.3f})"
        if nm and vgr
        else ""
    )
    steps = [a["steps"] for a in opt if a.get("steps") is not None]
    return [
        f"ScienceWorld (v7 test, deployed arms only; catalog = profiling rounds 0–6 under the controlled c16 runtime; "
        f"200 tasks × 1 run): the deployed Opt arms reach {min(a['quality'] for a in opt):.3f}–"
        f"{max(a['quality'] for a in opt):.3f} score/100{ref}; best {best['label']} {best['quality']:.3f} ± "
        f"{best['quality_se']:.3f} SE (gain {best['gain']:+.3f} ± {best['gain_se']:.3f}).",
        f"ScienceWorld cost: Opt arms {min(a['tokens'] for a in opt):,.0f}–{max(a['tokens'] for a in opt):,.0f} "
        f"tokens and {min(a['latency'] for a in opt):.1f}–{max(a['latency'] for a in opt):.1f} s LLM-span latency "
        f"per episode"
        + (f" (VGR {vgr['tokens']:,.0f} tokens, {vgr['latency']:.1f} s)" if vgr else "")
        + (
            f"; {min(steps):.2f}–{max(steps):.2f} environment steps per episode"
            + (
                f" vs No-Memory {nm['steps']:.2f}"
                if nm and nm.get("steps") is not None
                else ""
            )
            if steps
            else ""
        )
        + ".",
    ]


def tau2_conclusions(tau: dict) -> list[str]:
    """English τ² bullet points (pool -v5 test, pass^1)."""
    arms = [a for a in tau["arms"] if a.get("quality") is not None]
    if not arms or tau.get("pending"):
        return [f"τ²: {tau.get('note', 'pool -v5 test pending')}."]
    opt = [a for a in arms if a["kind"] not in ("ref", "plan")]
    if not opt:
        return [f"τ²: {tau.get('note', 'pool -v5 test pending')}."]
    best = max(opt, key=lambda a: (a["quality"], -a["tokens"]))
    sig = [
        a["label"] for a in arms if a.get("p_sign") is not None and a["p_sign"] < 0.05
    ]
    out = [
        "τ² (pool -v5 test, 100 tasks × 1 trial, pass^1): "
        + "; ".join(f"{a['label']} {100 * a['quality']:.0f}%" for a in arms)
        + ". Paired Δ vs No-Memory (same tasks, same trial): "
        + "; ".join(
            f"{a['label']} {100 * a['gain']:+.0f} ± {100 * a['gain_se']:.1f} pp ({a['wins']} W / {a['losses']} L)"
            for a in arms
            if a.get("gain") is not None
        )
        + ".",
        f"Best Opt arm on test: {best['label']} {100 * best['quality']:.0f}% "
        f"(Δ {100 * best['gain']:+.0f} ± {100 * best['gain_se']:.1f} pp vs No-Memory, {best['wins']} W / "
        f"{best['losses']} L, exact sign-test p = {best['p_sign']:.3g}; chosen post hoc among 7 arms). "
        + (
            "Arms with p < 0.05 (exact, unadjusted): " + ", ".join(sig) + "."
            if sig
            else "No arm differs from No-Memory at p < 0.05 (exact sign test, unadjusted)."
        ),
    ]
    return out


def fp_js() -> str:
    from trimem.experiments import fixed_plans_page as fp

    return fp.js()


def pqs_js() -> str:
    from trimem.experiments import perquery_sweep_page as pqs

    return pqs.js()


def main() -> int:
    data = {"datasets": collect()}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    sec = (
        SECTION.replace("__DATA__", blob)
        .replace("__FPJS__", fp_js())
        .replace("__PQSJS__", pqs_js())
        .replace(
            "__CONCLUSIONS__",
            "".join(f"<li>{c}</li>" for c in conclusions(data["datasets"])),
        )
        .replace("__TIME__", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
    )
    with PAGE.with_suffix(".lock").open(
        "a"
    ) as lock:  # page lock shared by every section renderer
        fcntl.flock(lock, fcntl.LOCK_EX)
        page = PAGE.read_text()
        from trimem.experiments.atlas_presentation import prepare_html, remove_block

        # user 2026-10-22: the separate ALF v7 re-test block is merged into this section
        page = remove_block(page, "alf-v7-retest")
        if START in page:
            page = page[: page.index(START)] + sec + page[page.index(END) + len(END) :]
        elif BEFORE in page:
            page = page.replace(BEFORE, sec + "\n" + BEFORE, 1)
        else:
            page = page.replace("<footer>", sec + "\n<footer>", 1)
        tmp = PAGE.with_suffix(".fd.tmp.html")

        tmp.write_text(prepare_html(page))
        tmp.replace(PAGE)
    from trimem.experiments.difficulty_report import write_report

    write_report()
    for d in data["datasets"]:
        print(
            d["name"],
            [
                (
                    a["label"],
                    round(a["quality"], 3),
                    round(a["ret"] + a["gen"], 1),
                    round(a["tin"] + a["tout"] + a["tsem"]),
                )
                for a in d["arms"]
            ],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
