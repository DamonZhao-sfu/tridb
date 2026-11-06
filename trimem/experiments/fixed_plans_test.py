"""Fixed-PLAN reference arms on the ALFWorld / ScienceWorld TEST sets (user request 2026-10-23).

Each arm = one plan's DEFAULT configuration (the round-0 plan defaults of the dataset's pool-strat200-v7 run, P1 =
the recorded ``plan_extension_p1.plan_default``; = ``space.plan_defaults()`` + ``v7_defaults`` of
session-space-agentenv-v7: seed 8 / depth 2 / fanout 4 / top-k 5, edges similar+similar_up, raw cards, sim_gate
none, success_failure, SemFilter threshold 0.6) routed to EVERY test task. Arms (``PLANS``): VGR, VGR+Rank,
VGR+SemFilter (P2), VGR+SemFilter+Rank (P2+Rank), V+SemFilter+G (P1), V+SemFilter+G+Rank (P1+Rank).

* test set = the 200 AgentEval items (``bo_pool_driver.test_ids``), memory policy success+failure-v1 (each config's
  outcome_policy restricts what it retrieves), runtime = the dataset's current test runtime: controlled-c16-v1
  (``alf_test_c16`` layout: one configuration at a time, 2 halves x 8 processes, small configurations share one
  final batch, both-halves monitor), quality reuse OFF, the state.json semfilter-v2 knobs (top-32 cosine
  prefilter, batch 8, score-only guided, rubric v3) exported to every runner (inert for SemFilter-free plans).
* pair reuse: a (configuration, task) pair already measured under the SAME runtime is reused, never rerun:
  the v7 test (``test_v7c16``, opt experiment ``...:pairs-c16-v7:...``; SemFilter plans only when the row is
  labelled semfilter-v2) and, ALFWorld only, the earlier c16 test rerun (``test_c16``, ``...:pairs-c16:...``;
  SemFilter-free plans only: its SemFilter pairs ran the v1 judge). New pairs run into
  ``<run>/test_v7c16/fixed_plans/pairs`` under the SEPARATE opt experiment ``...:pairs-c16-v7-fixed:...``;
  profiling state / rows and the deployed test are never touched.
* report (``<run>/test_v7c16/fixed_plans/report.json`` + ``report.md``): per plan arm quality ± SE, paired gain vs
  the controlled-v1 NoMemory ± SE, tokens, wall / LLM-span latency, steps ± SE, steps on successes, pair sources;
  the controlled-v1 NoMemory / VGR references copied from the dataset's v7 test report.

Stages: ``plan`` (counts new vs reused, time estimate; nothing runs), ``run`` (``--dry``: routing + plan.json only),
``report``.

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 POOL_SEMFILTER=v2 .venv-optimizer/bin/python -m \\
        trimem.experiments.fixed_plans_test {plan|run|report} --dataset {alfworld|sciworld|webshop}
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics as stats
from datetime import UTC, datetime
from pathlib import Path

DATASETS = ("alfworld", "sciworld", "webshop")  # webshop: user 2026-10-23 (additive)
PLANS = ("VGR", "VGR+Rank", "P2", "P2+Rank", "P1", "P1+Rank")
LABELS = {
    "VGR": "VGR",
    "VGR+Rank": "VGR+Rank",
    "P2": "VGR+SemFilter (P2)",
    "P2+Rank": "VGR+SemFilter+Rank (P2+Rank)",
    "P1": "V+SemFilter+G (P1)",
    "P1+Rank": "V+SemFilter+G+Rank (P1+Rank)",
}
ARM_PREFIX = "Plan-"
SUBDIR = "test_v7c16/fixed_plans"
ARM_TAG = "pairs-c16-v7-fixed"
PROTOCOL = "fixed-plan-defaults-test-v1"
CONCURRENCY = 16
RUNTIME = "controlled-c16-v1"
QNAME = {"alfworld": "success rate", "sciworld": "score/100", "webshop": "reward"}
DONE_STATUSES = ("evaluated",)


def arm_id(plan: str) -> str:
    return ARM_PREFIX + plan


def plan_of(arm: str) -> str:
    return arm.removeprefix(ARM_PREFIX)


def out_dir(base: Path) -> Path:
    return base / SUBDIR


def run_base(ds: str) -> Path:
    from trimem.experiments import bo_pool_driver as d

    if ds not in DATASETS:
        raise SystemExit(f"unknown dataset {ds}")
    return d.run_dir(ds)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


# ------------------------------------------------------------------ plan defaults + routing (pure)


def plan_configs(state: dict) -> dict[str, dict]:
    """plan -> its DEFAULT configuration in this run: the round-0 ``default``-channel configurations (+ P1's
    recorded extension default), each checked against the canonical ``plan_defaults`` of the space. Refuses a run
    whose round 0 lacks a plan or whose recorded default differs."""
    from trimem.optimizer import session_space_agentenv_v2 as space

    r0 = next((r for r in state["rounds"] if r["round"] == 0), None)
    if r0 is None:
        raise SystemExit("state.json has no round 0")
    got = {
        c["config"]["plan"]: space.validate(c["config"])
        for c in r0["configs"]
        if c.get("channel") == "default"
    }
    ext = state.get("plan_extension_p1") or {}
    if ext.get("plan_default"):
        got.setdefault("P1", space.validate(ext["plan_default"]))
    if (state.get("round0_design") or {}).get("sampler") == "sobol-only":
        # round0-design-v1 (WebShop, user 2026-10-23): round 0 = 10 Sobol, no recorded plan defaults -> the
        # canonical plan defaults of the space (the same configurations ALF / Sci recorded in their round 0)
        for p in PLANS:
            got.setdefault(p, space.validate({"plan": p, **space.plan_defaults()}))
    out = {}
    for p in PLANS:
        if p not in got:
            raise SystemExit(
                f"plan {p}: no recorded default in this run (round 0 / plan extension)"
            )
        want = space.validate({"plan": p, **space.plan_defaults()})
        if got[p] != want:
            raise SystemExit(f"plan {p}: recorded default {got[p]} != canonical {want}")
        out[p] = got[p]
    # semfilter-v2b (WebShop, user 2026-10-23): the SemFilter plan arms run the new regime -- plan default +
    # sem_fallback fill-topk (threshold 0.6 is in the v2b set); SemFilter-free plans unchanged
    v2b = (state.get("semfilter") or {}).get("v2b")
    if v2b:
        for p in PLANS:
            if p in space.SEMFILTER_PLANS:
                if out[p]["semfilter_threshold"] not in v2b["thresholds"]:
                    raise SystemExit(
                        f"plan {p}: default threshold outside semfilter-v2b {v2b['thresholds']}"
                    )
                out[p] = space.validate({**out[p], "sem_fallback": v2b["sem_fallback"]})
    return out


def routing(cfgs: dict[str, dict], ids: list[str]) -> dict:
    """Fixed-plan routing table: every task of every arm -> the plan default, no router overhead. Pure."""
    return {
        "protocol": PROTOCOL,
        "arms": {
            arm_id(p): {
                "kind": "fixed",
                "plan": p,
                "label": LABELS[p],
                "routes": {q: {"config": cfg, "overhead_ms": 0.0} for q in ids},
            }
            for p, cfg in cfgs.items()
        },
    }


def frozen_routing(base: Path, tab: dict, write: bool) -> dict:
    """The routing table (written once; a later call must describe the same configurations)."""
    p = out_dir(base) / "routing.json"
    if p.exists():
        old = json.loads(p.read_text())
        same = {
            n: (
                a["plan"],
                next(iter(a["routes"].values()))["config"],
                sorted(a["routes"]),
            )
            for n, a in old["arms"].items()
        } == {
            n: (
                a["plan"],
                next(iter(a["routes"].values()))["config"],
                sorted(a["routes"]),
            )
            for n, a in tab["arms"].items()
        }
        if not same:
            raise SystemExit(
                f"{p} describes other configurations / tasks; refusing to mix"
            )
        return old
    if write:
        save(p, {**tab, "frozen_at": datetime.now(UTC).isoformat()})
    return tab


# ------------------------------------------------------------------ pair reuse


def semfilter_ok_label(ds: str) -> str:
    """The SemFilter label a reusable v7-test / fixed-plan row must carry: the run's ``semfilter_label`` (WebShop
    after semfilter-v2b: "semfilter-v2b"), else "semfilter-v2" (ALFWorld / ScienceWorld: unchanged)."""
    from trimem.experiments import bo_pool_driver as d

    p = d.run_dir(ds) / "state.json"
    if not p.exists():
        return "semfilter-v2"
    return d.semfilter_label(json.loads(p.read_text())) or "semfilter-v2"


def sources(ds: str) -> list[dict]:
    """Earlier test pairs measured under the SAME runtime (controlled-c16-v1, quality reuse off): the v7 test (any
    plan; SemFilter plans only with a semfilter-v2 label) and, ALFWorld, the c16 rerun (SemFilter-free plans)."""
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import alf_test_v7 as v7

    out = [
        {
            "name": "v7_test",
            "tag": v7.ARM_TAG,
            "subdir": v7.SUBDIR,
            "semfilter_ok": semfilter_ok_label(ds),
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
    return out


def reusable(cfg: dict, label: str | None, src: dict) -> bool:
    """A source row may stand for this pair: SemFilter-free plans always (same runtime); SemFilter plans only when
    the row ran the same judge version (``protocol.semfilter``). Pure."""
    from trimem.optimizer import session_space_agentenv_v2 as space

    if cfg["plan"] not in space.SEMFILTER_PLANS:
        return True
    return src["semfilter_ok"] is not None and label == src["semfilter_ok"]


ROWS_SQL = (
    "SELECT query_id, protocol->'space_config' AS cfg, protocol->>'semfilter' AS semfilter, quality, total_tokens, "
    "e2e_ms, generation FROM opt.session_retrieval WHERE experiment_id = %s AND split = 'test' AND status = 'evaluated'"
)


def opt_rows(experiment: str, max_round: int | None = None) -> list[dict]:
    """Evaluated test rows of one opt experiment (read-only transaction). ``max_round`` (max-round-v1, WebShop):
    only rows recorded at that turn budget (protocol.max_round, absent = 30) -- a 30-turn row never stands for a
    max_round-n pair; None = every row (unchanged)."""
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments import bo_pool_driver as d

    dsn = d.drv.OPT_DSN + " options='-c default_transaction_read_only=on'"
    sql = (
        ROWS_SQL
        if max_round is None
        else ROWS_SQL.replace(
            " FROM opt.session_retrieval",
            ", protocol AS row_protocol FROM opt.session_retrieval",
            1,
        )
    )
    with psycopg.connect(dsn, row_factory=dict_row) as c:
        rows = c.execute(sql, (experiment,)).fetchall()
    if max_round is None:
        return rows
    return [
        {k: v for k, v in r.items() if k != "row_protocol"}
        for r in rows
        if d.eb.row_max_round(r["row_protocol"]) == max_round
    ]


def wanted(tab: dict) -> dict[str, list[str]]:
    from trimem.experiments import bo_pool_driver as d

    return d.pairs_of(tab)


def old_pairs(ds: str, want: dict[str, list[str]]) -> dict[tuple[str, str], str]:
    """(config key, task) -> the source that measured it (first source wins), only for wanted pairs."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    pairs = {(k, q) for k, qs in want.items() for q in qs}
    out: dict[tuple[str, str], str] = {}
    mr = d.test_max_round(ds)  # max-round-v1: only same-budget rows are reused
    for src in sources(ds):
        for r in opt_rows(d.test_experiment(ds, src["tag"]), mr):
            if not r["cfg"]:
                continue
            k = space.key(r["cfg"])
            if (k, r["query_id"]) in pairs and (k, r["query_id"]) not in out:
                if reusable(r["cfg"], r["semfilter"], src):
                    out[(k, r["query_id"])] = src["name"]
    return out


def split_pairs(
    want: dict[str, list[str]], old: dict[tuple[str, str], str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(reused, to run) per config key. Pure."""
    reuse: dict[str, list[str]] = {}
    run: dict[str, list[str]] = {}
    for k, qs in want.items():
        for q in qs:
            (reuse if (k, q) in old else run).setdefault(k, []).append(q)
    return reuse, run


def done_here(base: Path, ds: str, run: dict[str, list[str]]) -> dict[str, int]:
    """Pairs of ``run`` with a result file in this test's own pair dirs (a resumed run), per config key."""
    from trimem.experiments import bo_pool_driver as d

    out = {}
    for k, qs in run.items():
        have = d.drv.item_files(out_dir(base) / "pairs" / d.cfg_name(json.loads(k)), ds)
        out[k] = len(set(qs) & set(have))
    return out


# ------------------------------------------------------------------ plan (counts + estimate)


def wall_means(ds: str) -> dict[str, float]:
    """Mean opt-row e2e seconds per plan over the reuse sources' rows (+ this test's own rows): the per-episode
    wall used by the time estimate."""
    from trimem.experiments import bo_pool_driver as d

    by: dict[str, list[float]] = {}
    for tag in [s["tag"] for s in sources(ds)] + [ARM_TAG]:
        for r in opt_rows(d.test_experiment(ds, tag)):
            if r["cfg"] and r["e2e_ms"] is not None:
                by.setdefault(r["cfg"]["plan"], []).append(float(r["e2e_ms"]) / 1000)
    return {p: stats.mean(v) for p, v in by.items() if v}


def estimate_hours(
    remaining: dict[str, int], walls: dict[str, float], concurrency: int = CONCURRENCY
) -> float | None:
    """Saturated-throughput estimate: sum over plans of remaining pairs x mean episode wall / concurrency (h).
    Plans never measured take the mean of the measured plans. Pure."""
    if not walls:
        return None
    fallback = stats.mean(walls.values())
    sec = sum(n * walls.get(p, fallback) for p, n in remaining.items())
    return round(sec / concurrency / 3600, 2)


def observed_rate(base: Path, ds: str) -> dict | None:
    """Pairs / hour the dataset's v7 test actually achieved under the same layout: result files of
    ``test_v7c16/pairs`` written after its plan.json ``started`` over (last write - started)."""
    from trimem.experiments import alf_test_v7 as v7
    from trimem.experiments import bo_pool_driver as d

    p = base / v7.SUBDIR / "plan.json"
    if not p.exists():
        return None
    t0 = datetime.fromisoformat(json.loads(p.read_text())["started"]).timestamp()
    mt = [
        f.stat().st_mtime
        for c in (base / v7.SUBDIR / "pairs").glob("cfg_*")
        for f in d.drv.item_files(c, ds).values()
    ]
    mt = [m for m in mt if m >= t0]
    if len(mt) < 50 or max(mt) <= t0:
        return None
    return {
        "pairs": len(mt),
        "hours": round((max(mt) - t0) / 3600, 2),
        "per_hour": len(mt) * 3600 / (max(mt) - t0),
    }


def state_of(base: Path) -> dict:
    return json.loads((base / "state.json").read_text())


def build(ds: str) -> tuple[Path, dict, dict]:
    from trimem.experiments import bo_pool_driver as d

    base = run_base(ds)
    cfgs = plan_configs(state_of(base))
    tab = routing(cfgs, d.test_ids(ds))
    return base, cfgs, tab


def plan(ds: str) -> dict:
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    base, cfgs, tab = build(ds)
    want = wanted(tab)
    old = old_pairs(ds, want)
    reuse, run = split_pairs(want, old)
    done = done_here(base, ds, run)
    walls = wall_means(ds)
    per = {}
    remaining_by_plan: dict[str, int] = {}
    for p, cfg in cfgs.items():
        k = space.key(cfg)
        src: dict[str, int] = {}
        for q in reuse.get(k, []):
            src[old[(k, q)]] = src.get(old[(k, q)], 0) + 1
        rem = len(run.get(k, [])) - done.get(k, 0)
        remaining_by_plan[p] = rem
        per[arm_id(p)] = {
            "label": LABELS[p],
            "config": d.cfg_name(cfg),
            "semfilter_plan": p in space.SEMFILTER_PLANS,
            "pairs": len(want.get(k, [])),
            "reused": sum(src.values()),
            "reused_by_source": src,
            "to_run": len(run.get(k, [])),
            "done_here": done.get(k, 0),
            "remaining": rem,
            "mean_episode_wall_s": walls.get(p),
        }
    tot = lambda f: sum(v[f] for v in per.values())  # noqa: E731
    obs = observed_rate(base, ds)
    return {
        "dataset": ds,
        "protocol": PROTOCOL,
        "run": str(base),
        "out": str(out_dir(base)),
        "experiment": d.test_experiment(ds, ARM_TAG),
        "runtime": RUNTIME,
        "concurrency": CONCURRENCY,
        "quality_reuse": False,
        "arms": per,
        "pairs": tot("pairs"),
        "reused": tot("reused"),
        "to_run": tot("to_run"),
        "done_here": tot("done_here"),
        "remaining": tot("remaining"),
        "hours_estimate": estimate_hours(remaining_by_plan, walls),
        "hours_estimate_observed_rate": None
        if obs is None
        else round(tot("remaining") / obs["per_hour"], 2),
        "observed_v7_test_rate": obs,
        "estimate_note": "remaining pairs x mean episode wall of the plan's earlier c16 test rows / 16 concurrent "
        "episodes (saturated throughput; the c16 layout runs one configuration at a time)",
        "reuse_rule": "a pair is reused iff measured under controlled-c16-v1 with quality reuse off in the v7 test "
        "(SemFilter plans: semfilter-v2 rows only)"
        + (" or the c16 rerun (SemFilter-free plans only)" if ds == "alfworld" else ""),
    }


# ------------------------------------------------------------------ run


def build_arms(base: Path, ds: str, run: dict[str, list[str]], semfilter: dict) -> list:
    """One subset arm per configuration (largest first), quality reuse OFF, the separate experiment / dirs."""
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    baseline, _ = d.test_baseline(ds, argparse.Namespace(sampling="per-config"))
    exp = d.test_experiment(ds, ARM_TAG)
    mr = d.test_max_round(
        ds
    )  # max-round-v1 (WebShop): the profiling run's turn budget (None = 30)
    arms = []
    for k, qs in sorted(run.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
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
                "bo_label": "test-fixed-plans-c16-v7",
                "round": d.PANEL,
                "test_version": d.TEST_VERSION,
                "test_concurrency": CONCURRENCY,
                "fixed_plan": cfg["plan"],
                "retest": PROTOCOL,
                "semfilter": semfilter.get("protocol")
                if cfg["plan"] in space.SEMFILTER_PLANS
                else None,
                **d.eb.protocol_extra(mr),
            },
        }
        arms.append(
            d.Arm(
                f"fixed {cfg['plan']} {d.cfg_name(cfg)}",
                out_dir(base) / "pairs" / d.cfg_name(cfg),
                qs,
                session,
                max_round=mr,
            )
        )
    return arms


def monitor_cls(ds: str):
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import sci_test_v7 as stv

    if ds == "webshop":  # user 2026-10-23 (additive): WebShop result files
        from trimem.experiments import webshop_test_v7 as wtv

        return wtv.HalfMonitor
    return c16.HalfMonitor if ds == "alfworld" else stv.HalfMonitor


def driver_lock(ds: str):
    """The dataset's bo_pool_driver lock (one driver per dataset: state, opt rows, env servers)."""
    from trimem.experiments import bo_pool_driver as d

    (d.OUT / ds).mkdir(parents=True, exist_ok=True)
    lock = open(d.OUT / ds / "driver.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another driver holds {d.OUT / ds / 'driver.lock'}") from None
    return lock


def run(ds: str, dry: bool = False) -> None:
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import alf_test_v7 as v7
    from trimem.experiments import bo_pool_driver as d
    from trimem.experiments import memory_policy as mp

    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    for k in ("TRIMEM_SEMANTIC_ROUTING", "TRIMEM_SEMANTIC_BASES", "MAX_ENVS"):
        if os.environ.get(k):
            raise SystemExit(f"{k} is set: {RUNTIME} runs with the defaults")
    base, cfgs, tab = build(ds)
    blk, env = v7.semfilter_env(
        base
    )  # refuses without POOL_SEMFILTER=v2 / with other knobs
    os.environ.update(env)
    _lock = None if dry else driver_lock(ds)
    tab = frozen_routing(base, tab, write=not dry)
    want = wanted(tab)
    _, todo = split_pairs(want, old_pairs(ds, want))
    p = plan(ds)
    rec = {
        "started": datetime.now(UTC).isoformat(),
        "layout": "one configuration at a time, 2 halves x 8 processes (reference layout); configurations with < "
        f"{c16.SMALL} pairs share one final batch (total <= {CONCURRENCY})",
        "semfilter": {"protocol": blk["protocol"], "env": env},
        "retrieval_slots": os.environ.get("TRIMEM_RETRIEVAL_SLOTS") or "default (16)",
        **(
            {
                "max_round": d.test_max_round(ds),
                "runtime_profile": d.eb.runtime_label(state_of(base), RUNTIME),
            }
            if d.test_max_round(ds) is not None
            else {}
        ),
        "dry": dry,
        **p,
    }
    if not dry:  # a dry run writes nothing
        save(out_dir(base) / "plan.json", rec)
        with open(out_dir(base) / "plan_history.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    arms = build_arms(base, ds, todo, blk)
    batches = c16.schedule(arms)
    d.log(
        f"fixed plans {ds}: {p['to_run']} pairs to run ({p['remaining']} not yet done; {p['reused']} reused) over "
        f"{len(arms)} configurations in {len(batches)} batches at concurrency {CONCURRENCY}, quality reuse off, "
        f"semfilter-v2 env -> {out_dir(base)}"
    )
    if dry or not arms:
        return
    res: dict[str, int] = {}
    for b in batches:
        mon = monitor_cls(ds)(b)
        mon.start()
        try:
            res |= d.run_parallel(ds, b, CONCURRENCY)
        finally:
            mon.stop.set()
            mon.join(timeout=5)
    bad = [a.name for a in arms if res.get(a.name, 0) < len(a.ids)]
    if bad:
        raise SystemExit(f"fixed-plan pairs incomplete: {bad}")
    d.log(f"FIXED PLANS TEST DONE {ds}")


# ------------------------------------------------------------------ rows + report


def row_pair(
    ds: str, r: dict, spans: dict[str, float], result_file: Path | None
) -> tuple[dict, dict | None]:
    """(wall row, span row) of one opt row: quality / tokens / e2e_ms / steps + SemFilter tokens and retrieval ms
    (the page's cost split). Span row None without an LLM call record. Pure on its inputs."""
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import bo_pool_driver as d

    g = r.get("generation") or {}
    wall = {
        **d.with_steps(
            ds,
            {
                "quality": float(r["quality"]),
                "tokens": float(r["total_tokens"]),
                "e2e_ms": float(r["e2e_ms"]),
            },
            d.episode_steps(ds, g, result_file),
        ),
        "tsem": float(g.get("semantic_tokens") or 0),
        "ret_ms": float(g.get("retrieval_ms") or 0),
    }
    sp = spans.get(r["query_id"])
    span = c16.with_span(wall, sp, wall["ret_ms"])
    if span is not None:
        span = {**span, "e2e_wall_ms": wall["e2e_ms"]}
    return wall, span


def pair_rows(ds: str, base: Path, tab: dict) -> tuple[dict, dict, dict]:
    """(config key, task) -> wall row / span row of every wanted pair: this test's own rows first, then the reuse
    sources (``reusable``), + (config key, task) -> source name."""
    from trimem.experiments import alf_test_c16 as c16
    from trimem.experiments import bo_pool_driver as d
    from trimem.optimizer import session_space_agentenv_v2 as space

    want = {(k, q) for k, qs in wanted(tab).items() for q in qs}
    wall: dict = {}
    span: dict = {}
    src_of: dict = {}
    srcs = [
        {"name": "new", "tag": ARM_TAG, "subdir": SUBDIR, "semfilter_ok": "*"}
    ] + sources(ds)
    mr = d.test_max_round(ds)  # max-round-v1: only same-budget rows
    for src in srcs:
        pdir = base / src["subdir"] / "pairs"
        spans: dict[str, dict[str, float]] = {}
        files: dict[str, dict[str, Path]] = {}
        for r in opt_rows(d.test_experiment(ds, src["tag"]), mr):
            if not r["cfg"]:
                continue
            k = space.key(r["cfg"])
            if (k, r["query_id"]) not in want or (k, r["query_id"]) in wall:
                continue
            if src["semfilter_ok"] != "*" and not reusable(
                r["cfg"], r["semfilter"], src
            ):
                continue
            name = d.cfg_name(r["cfg"])
            if name not in spans:
                spans[name] = c16.call_spans(pdir / name, ds)
                files[name] = d.drv.item_files(pdir / name, ds)
            w, s = row_pair(ds, r, spans[name], files[name].get(r["query_id"]))
            wall[(k, r["query_id"])] = w
            if s is not None:
                span[(k, r["query_id"])] = s
            src_of[(k, r["query_id"])] = src["name"]
    return wall, span, src_of


def nomemory_quality(ds: str) -> tuple[Path, dict[str, float]]:
    """The controlled-v1 NoMemory (paired baseline of gain, as the v7 test reports)."""
    from trimem.experiments import bo_pool_driver as d

    ctrl = d.drv.CTRL / f"{ds}_NoMemory"
    mr = d.test_max_round(
        ds
    )  # max-round-v1: the 30-turn reference in the derived-truncation view
    return ctrl, {
        i: d.eb.view_quality(ds, json.loads(f.read_text()), mr)
        for i, f in d.drv.item_files(ctrl, ds).items()
    }


def report(ds: str) -> dict:
    from trimem.experiments import alf_test_v7 as v7
    from trimem.experiments import bo_pool_driver as d

    base, cfgs, tab = build(ds)
    tab = frozen_routing(base, tab, write=False)
    ctrl, nm = nomemory_quality(ds)
    wall, span, src_of = pair_rows(ds, base, tab)
    aw, asp = d.assemble(tab, wall, nm), d.assemble(tab, span, nm)
    arms = {}
    for name, v in aw.items():
        routes = tab["arms"][name]["routes"]
        k = d.space.key(next(iter(routes.values()))["config"])
        by: dict[str, int] = {}
        for q in routes:
            s = src_of.get((k, q))
            if s:
                by[s] = by.get(s, 0) + 1
        arms[name] = {
            **v,
            "plan": tab["arms"][name]["plan"],
            "label": tab["arms"][name]["label"],
            "config": d.cfg_name(next(iter(routes.values()))["config"]),
            "latency_span_s": asp[name].get("latency_s"),
            "span_covered": asp[name].get("covered"),
            "pair_sources": by,
        }
    v7r = base / v7.REPORT
    refs = json.loads(v7r.read_text()).get("reference_arms") if v7r.exists() else None
    pl = out_dir(base) / "plan.json"
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL,
        "dataset": ds,
        "experiment": d.test_experiment(ds, ARM_TAG),
        "runtime_profile": d.eb.runtime_label(state_of(base), RUNTIME) or RUNTIME,
        **(
            {"episode_budget": d.eb.block(state_of(base))}
            if d.eb.block(state_of(base))
            else {}
        ),
        "concurrency": CONCURRENCY,
        "quality_reuse": False,
        "quality": QNAME[ds],
        "plan": json.loads(pl.read_text()) if pl.exists() else None,
        "semfilter": (state_of(base).get("semfilter") or {}).get("protocol"),
        "configs": {p: d.cfg_name(c) for p, c in cfgs.items()},
        "latency_definitions": {
            "latency_s": "wall: official episode wall time + retrieval ms (opt e2e_ms)",
            "latency_span_s": "span: first .. last LLM call of the episode (llm_calls.jsonl) + retrieval ms (= the "
            "controlled-v1 NoMemory reference's definition, plotted on the page)",
        },
        "baseline": {
            "dir": str(ctrl),
            "label": "controlled-v1 NoMemory (paired baseline of gain)",
            "quality": sum(nm.values()) / max(len(nm), 1),
        },
        "pairs": len(wall),
        "pairs_span": len(span),
        "pair_sources": {
            s: sum(1 for x in src_of.values() if x == s)
            for s in sorted(set(src_of.values()))
        },
        "complete": all(v.get("covered") == v.get("n") for v in arms.values()),
        "arms": arms,
        "reference_arms": refs,
        "reference_source": str(v7r),
    }
    save(out_dir(base) / "report.json", rep)
    (out_dir(base) / "report.md").write_text("\n".join(markdown(rep)) + "\n")
    return rep


def _f(x, fmt: str) -> str:
    return "–" if x is None else format(x, fmt)


def markdown(rep: dict) -> list[str]:
    L = [
        f"# {rep['dataset']} · fixed-plan reference arms on the test set ({rep['runtime_profile']})",
        "",
        f"Generated {rep['generated']}. Each arm = the plan's round-0 default configuration on every one of the 200 "
        f"test tasks; quality = {rep['quality']}; {rep['pairs']} (config, task) pairs (sources {rep['pair_sources']}); "
        f"quality reuse OFF; SemFilter {rep['semfilter']}. Complete: {rep['complete']}. Gain = paired vs the "
        "controlled-v1 NoMemory.",
        "",
        "| arm | covered | quality ± SE | gain vs NoMemory ± SE | tokens | latency wall s | latency span s | steps ± SE "
        "| steps (succ.) | pair sources |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for n, x in rep["arms"].items():
        if x.get("quality") is None:
            L.append(f"| {x['label']} | 0/{x['n']} | pending | | | | | | | |")
            continue
        L.append(
            f"| {x['label']} | {x['covered']}/{x['n']} | {x['quality']:.3f} ± {_f(x['quality_se'], '.3f')} | "
            f"{_f(x.get('paired_gain_vs_nomemory'), '+.3f')} ± {_f(x.get('paired_gain_se'), '.3f')} | "
            f"{x['tokens']:,.0f} | {x['latency_s']:.1f} | {_f(x.get('latency_span_s'), '.1f')} | "
            f"{_f(x.get('steps'), '.2f')} ± {_f(x.get('steps_se'), '.2f')} | {_f(x.get('steps_success'), '.2f')} | "
            f"{x['pair_sources']} |"
        )
    for n, r in (rep.get("reference_arms") or {}).items():
        m = r.get("wall") or r.get("span")
        if m:
            L.append(
                f"| ref {n} (controlled-v1) | {m.get('covered', '')}/{m.get('n', '')} | {_f(m['quality'], '.3f')} ± "
                f"{_f(m['quality_se'], '.3f')} | {_f(m.get('paired_gain_vs_nomemory'), '+.3f')} | "
                f"{_f(m.get('tokens'), ',.0f')} | {_f((r.get('wall') or {}).get('latency_s'), '.1f')} | "
                f"{_f((r.get('span') or {}).get('latency_s'), '.1f')} | {_f(m.get('steps'), '.2f')} | | |"
            )
    return L


def summary_lines(p: dict) -> list[str]:
    """Compact text of a ``plan`` result (also used by tau2.fixed_plans_test). Pure."""
    obs = p.get("hours_estimate_observed_rate")
    out = [
        f"== {p['dataset']}: {p['pairs']} pairs = {p['reused']} reused + {p['to_run']} new ({p['done_here']} already "
        f"done here, {p['remaining']} remaining); estimate {p['hours_estimate']} h"
        + (f" (observed v7-test rate: {obs} h)" if obs is not None else "")
        + f"; runtime {p['runtime']}, concurrency {p['concurrency']}; experiment {p['experiment']}"
    ]
    for n, a in p["arms"].items():
        out.append(
            f"   {n:<16} {a['config']}  pairs {a['pairs']:>3}  reused {a['reused']:>3} {a['reused_by_source'] or ''}  "
            f"new {a['to_run']:>3}  remaining {a['remaining']:>3}"
        )
    out += [f"   absent {k}: {why}" for k, why in (p.get("absent_plans") or {}).items()]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("plan", "run", "report"))
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument(
        "--dry", action="store_true", help="run: log only, write nothing, start nothing"
    )
    ap.add_argument(
        "--summary", action="store_true", help="plan: compact text instead of JSON"
    )
    a = ap.parse_args(argv)
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
        print("\n".join(markdown(rep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
