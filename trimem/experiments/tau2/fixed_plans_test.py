"""Fixed-PLAN reference arms on the τ²-bench TEST set (user request 2026-10-23; τ² counterpart of
``trimem.experiments.fixed_plans_test``).

Each arm = one plan's DEFAULT configuration of the τ² pool -v5 run (round-0 plan defaults of session-space-tau2-v5:
seed 8 / depth 2 / fanout 4 / top-k 5, edges similar+similar_up, memory_form compact, sim_gate none,
success_failure, SemFilter threshold 0.6; P1+Rank = the same knobs, its plan default -- never profiled in this run,
the plan extension starts at round 7 and profiling paused after round 6) routed to EVERY one of the 100 official test
tasks. Arms (``PLANS``): VGR, VGR+Rank, VGR+SemFilter (P2), VGR+SemFilter+Rank (P2+Rank), V+SemFilter+G+Rank
(P1+Rank).

P1 (V+SemFilter+G) is ABSENT for τ² (``ABSENT``): session-space-tau2-v5 pins five plans (no P1), and
``tau2.runner_pool`` builds the SemFilter judge only for the τ² space's SEMFILTER_PLANS (P2 / P2+Rank / P1+Rank), so a
P1 configuration would run WITHOUT SemFilter (silently = V+G). Supporting it needs a change to a module the live τ²
job imports; not done.

* runtime = the τ² deployed test after the semfilter-v2 rerun (``tau2.sfv2_rerun test`` = ``pool_profile.test_stage``):
  100 test tasks x 1 trial (trial 0, seed 300), pass^1, ``TAU2_TEST_CONCURRENCY`` (40) concurrent episodes, the queue's
  retrieval slots / semantic workers, the semfilter-v2 env (opt-in TAU2_SEMFILTER=v2), test quality reuse ON in its
  own index (``<experiment>:sfv2``, test split) exactly as the deployed test (identical injected memory -> the
  earlier episode's outcome).
* semfilter-v2 knobs (user 2026-10-23: the τ² sfv2 TRAIN rerun is reverted, so state.json loses its ``semfilter``
  block): this stage carries a FROZEN copy (``SEMFILTER_V2``) of the knobs the sfv2 supersede recorded
  (2026-10-23T03:10:14Z; identical to the ALFWorld / ScienceWorld v7 recorded knobs, rubric_dataset tau2 = the τ²
  KNOWLEDGE / CONFLICTS text, selected by ``tau2.runner_pool``). A state.json block, when present, must equal it
  (``semfilter_env``); without one the frozen env is exported (refused with conflicting TRIMEM_SEMFILTER_*).
* pair reuse: a (configuration, task) pair of the deployed test (``test/pairs/<cfg>``) is reused iff its result file
  validates for this configuration / trial / seed, ran at the same load regime (``c<TEST_CONCURRENCY>``) and -- for
  SemFilter plans -- carries the semfilter-v2 receipt. New pairs run into ``test_fixed_plans/pairs`` under the
  SEPARATE opt experiment ``tau2:test100-1trial:<TEST_VERSION>:fixed-plans``; profiling state / rows and the deployed
  test are never touched.
* report (``test_fixed_plans/report.json`` + ``report.md``): per plan arm pass^1 ± SE, paired gain vs the trial-0
  NoMemory ± SE, tokens, latency (episode wall), agent calls ± SE, pair sources; references from test_report.json.

Stages: ``plan`` (counts new vs reused, time estimate; nothing runs), ``run`` (``--dry``: nothing written), ``report``.

    PYTHONPATH=. TRIMEM_MEMORY_POLICY=success+failure-v1 TAU2_SEMFILTER=v2 .venv-optimizer/bin/python -m \\
        trimem.experiments.tau2.fixed_plans_test {plan|run|report}
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments.tau2 import pool_profile as pp

PLANS = ("VGR", "VGR+Rank", "P2", "P2+Rank", "P1+Rank")
ABSENT = {
    "P1": "not in the τ² space (session-space-tau2-v5 pins VGR / VGR+Rank / P2 / P2+Rank / P1+Rank) and "
    "tau2.runner_pool builds no SemFilter judge for P1 (it would run as V+G without SemFilter)"
}
LABELS = {
    "VGR": "VGR",
    "VGR+Rank": "VGR+Rank",
    "P2": "VGR+SemFilter (P2)",
    "P2+Rank": "VGR+SemFilter+Rank (P2+Rank)",
    "P1+Rank": "V+SemFilter+G+Rank (P1+Rank)",
}
ARM_PREFIX = "Plan-"
SUBDIR = "test_fixed_plans"
PROTOCOL = "tau2-fixed-plan-defaults-test-v1"
SEMFILTER_PROTOCOL = "semfilter-v2"
# frozen copy of the τ² semfilter-v2 knobs (state.json semfilter of the reverted sfv2 supersede, recorded
# 2026-10-23T03:10:14Z; = sfv2_rerun.env_of(sfv2_rerun.want_settings()) and the ALF / Sci v7 recorded env)
SEMFILTER_V2 = {
    "protocol": SEMFILTER_PROTOCOL,
    "top_n": 32,
    "batch": 8,
    "max_tokens": 96,
    "judge_input": "raw",
    "rubric": "semantic-rubric-v3",
    "guided": True,
    "rubric_dataset": "tau2",
    "env": {
        "TRIMEM_SEMFILTER_V2": "1",
        "TRIMEM_SEMFILTER_TOPN": "32",
        "TRIMEM_SEMFILTER_BATCH": "8",
        "TRIMEM_SEMFILTER_MAX_TOKENS": "96",
        "TRIMEM_SEMFILTER_JUDGE_INPUT": "raw",
        "TRIMEM_SEMFILTER_RUBRIC": "semantic-rubric-v3",
        "TRIMEM_SEMFILTER_GUIDED": "1",
    },
    "source": "frozen copy of the τ² sfv2 supersede's state.json semfilter block (2026-10-23T03:10:14Z; the τ² "
    "sfv2 train rerun was reverted 2026-10-23, user decision)",
}


def semfilter_env(state: dict, env=None) -> dict[str, str]:
    """The semfilter-v2 env this stage exports: the state.json block's when recorded (must equal the frozen
    ``SEMFILTER_V2`` env), else the frozen env. Refuses without ``TAU2_SEMFILTER=v2`` or with a conflicting
    TRIMEM_SEMFILTER_* value in ``env``. Pure (``env`` defaults to os.environ, never modified here)."""
    import os

    from trimem.experiments.tau2 import sfv2_rerun as sfv2

    env = os.environ if env is None else env
    blk = state.get("semfilter")
    if blk is not None:
        if (
            blk.get("protocol") != SEMFILTER_PROTOCOL
            or blk.get("env") != SEMFILTER_V2["env"]
        ):
            raise SystemExit(
                f"τ² state.json semfilter {blk.get('protocol')} {blk.get('env')} != the frozen fixed-plan knobs"
            )
        return sfv2.check_semfilter(state, env)
    if env.get(sfv2.SEMFILTER_ENV, "").strip() != "v2":
        raise SystemExit(
            f"run with {sfv2.SEMFILTER_ENV}=v2 (opt-in: τ² SemFilter plans use semfilter-v2)"
        )
    want = dict(SEMFILTER_V2["env"])
    diff = {
        k: (env.get(k), v)
        for k, v in want.items()
        if env.get(k) is not None and env.get(k) != v
    }
    if diff:
        raise SystemExit(
            f"TRIMEM_SEMFILTER_* differ from the frozen fixed-plan knobs: {diff}"
        )
    return want


def arm_id(plan: str) -> str:
    return ARM_PREFIX + plan


def out_dir() -> Path:
    return pp.OUT / SUBDIR


def experiment() -> str:
    from trimem.experiments import memory_policy as mp

    return mp.tag(
        f"{pp.DS}:test100-{pp.TEST_TRIALS}trial:{pp.TEST_VERSION}:fixed-plans",
        mp.SUCCESS_FAILURE,
    )


def regime() -> str:
    return f"c{pp.TEST_CONCURRENCY}"


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


# ------------------------------------------------------------------ plan defaults + routing (pure)


def plan_configs(state: dict) -> dict[str, dict]:
    """plan -> its default configuration: the round-0 ``default``-channel configurations (checked against the
    canonical plan defaults of the τ² space), P1+Rank = its canonical plan default (never profiled here)."""
    from trimem.optimizer import session_space_agentenv_v2 as v2

    r0 = next((r for r in state["rounds"] if r["round"] == 0), None)
    if r0 is None:
        raise SystemExit("τ² state.json has no round 0")
    got = {
        c["config"]["plan"]: pp.space.validate(c["config"])
        for c in r0["configs"]
        if c.get("channel") == "default"
    }
    out = {}
    for p in PLANS:
        if p not in pp.space.PLANS:
            raise SystemExit(f"plan {p} is not in {pp.space.VERSION}")
        want = pp.space.validate({"plan": p, **v2.plan_defaults()})
        if p in got and got[p] != want:
            raise SystemExit(f"plan {p}: recorded default {got[p]} != canonical {want}")
        if p not in got and p not in pp.space.ADDED_PLANS:
            raise SystemExit(f"plan {p}: no round-0 default in this run")
        out[p] = want
    return out


def routing(cfgs: dict[str, dict], keys: list[str]) -> dict:
    """Fixed-plan routing table (every task -> the plan default, no router overhead). Pure."""
    return {
        "protocol": PROTOCOL,
        "experiment": experiment(),
        "test_trials": pp.TEST_TRIALS,
        "test_quality": f"pass^{pp.TEST_K}",
        "absent_plans": ABSENT,
        "arms": {
            arm_id(p): {
                "kind": "fixed",
                "plan": p,
                "label": LABELS[p],
                "routes": {q: {"config": cfg, "overhead_ms": 0.0} for q in keys},
            }
            for p, cfg in cfgs.items()
        },
    }


def build() -> tuple[dict, dict]:
    state = json.loads((pp.OUT / "state.json").read_text())
    cfgs = plan_configs(state)
    return cfgs, routing(cfgs, pp.test_keys())


def frozen_routing(tab: dict, write: bool) -> dict:
    p = out_dir() / "routing.json"
    if p.exists():
        old = json.loads(p.read_text())
        sig = lambda t: {  # noqa: E731
            n: (
                a["plan"],
                next(iter(a["routes"].values()))["config"],
                sorted(a["routes"]),
            )
            for n, a in t["arms"].items()
        }
        if sig(old) != sig(tab):
            raise SystemExit(
                f"{p} describes other configurations / tasks; refusing to mix"
            )
        return old
    if write:
        save(p, {**tab, "frozen_at": datetime.now(UTC).isoformat()})
    return tab


# ------------------------------------------------------------------ pair sources


def source_ok(r: dict, cfg: dict, want_regime: str) -> bool:
    """A deployed-test result file may stand for this pair: same load regime and, for SemFilter plans, the
    semfilter-v2 receipt. Pure (identity / seed / trial are checked by ``pool_profile.read_episode``)."""
    if ((r.get("load") or {}).get("regime")) != want_regime:
        return False
    if cfg["plan"] in pp.space.SEMFILTER_PLANS:
        sf = (r.get("memory") or {}).get("semfilter") or {}
        return sf.get("protocol") == SEMFILTER_PROTOCOL
    return True


def episode_path(d: Path, key: str) -> Path:
    from trimem.experiments.tau2 import runner as R

    dom, t = key.split(":", 1)
    return R.result_path(d, dom, t, pp.file_trial("test", 0))


def read_pair(d: Path, key: str, cfg: dict) -> dict | None:
    """Validated trial-0 episode record of one pair file (None = absent)."""
    try:
        return pp.read_episode(
            episode_path(d, key), key=key, trial=0, split="test", cfg=cfg
        )
    except ValueError:
        return None


def pair_sources(tab: dict) -> dict[tuple[str, str], tuple[str, Path]]:
    """(config key, task) -> (source name, result file) of every wanted pair present: this test's own pair files
    first, then eligible deployed-test pair files."""
    from trimem.experiments.tau2 import runner as R

    out: dict[tuple[str, str], tuple[str, Path]] = {}
    want_regime = regime()
    for k, qs in pp.pairs_of(tab).items():
        cfg = json.loads(k)
        own = out_dir() / "pairs" / pp.cfg_name(cfg)
        dep = pp.OUT / "test" / "pairs" / pp.cfg_name(cfg)
        for q in qs:
            if read_pair(own, q, cfg) is not None:
                out[(k, q)] = ("new", episode_path(own, q))
                continue
            if read_pair(dep, q, cfg) is None:
                continue
            r = R.read_result(episode_path(dep, q))
            if r is not None and source_ok(r, cfg, want_regime):
                out[(k, q)] = ("deployed_test", episode_path(dep, q))
    return out


def split_pairs(
    tab: dict, src: dict[tuple[str, str], tuple[str, Path]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(reused from the deployed test, to run here incl. done) per config key. Pure."""
    reuse: dict[str, list[str]] = {}
    run: dict[str, list[str]] = {}
    for k, qs in pp.pairs_of(tab).items():
        for q in qs:
            s = src.get((k, q))
            (reuse if s and s[0] == "deployed_test" else run).setdefault(k, []).append(
                q
            )
    return reuse, run


# ------------------------------------------------------------------ plan


def wall_minutes() -> dict[str, float]:
    """Mean episode wall (min) per plan of the existing c40 test pair files (deployed test + this test)."""
    from trimem.experiments.tau2 import runner as R

    by: dict[str, list[float]] = {}
    for root in (pp.OUT / "test" / "pairs", out_dir() / "pairs"):
        for d in sorted(root.glob("cfg_*")) if root.exists() else []:
            a = d / "arm.json"
            ck = (
                (json.loads(a.read_text()) or {}).get("config_key")
                if a.exists()
                else None
            )
            if not ck:
                continue
            plan = json.loads(ck)["plan"]
            for f in d.glob("*.json"):
                if f.name == "arm.json":
                    continue
                r = R.read_result(f)
                if (
                    r
                    and r.get("wall_ms")
                    and (r.get("load") or {}).get("regime") == regime()
                ):
                    by.setdefault(plan, []).append(float(r["wall_ms"]) / 60000)
    return {p: stats.mean(v) for p, v in by.items() if v}


def estimate_hours(
    remaining: dict[str, int], walls: dict[str, float], conc: int
) -> float | None:
    """remaining pairs x mean episode wall (min) / concurrency (h); unmeasured plans take the mean. Pure."""
    if not walls:
        return None
    fb = stats.mean(walls.values())
    return round(sum(n * walls.get(p, fb) for p, n in remaining.items()) / conc / 60, 2)


def plan() -> dict:
    cfgs, tab = build()
    src = pair_sources(tab)
    reuse, run = split_pairs(tab, src)
    walls = wall_minutes()
    per, rem_by = {}, {}
    for p, cfg in cfgs.items():
        k = pp.space.key(cfg)
        done = sum(1 for q in run.get(k, []) if (k, q) in src)
        rem = len(run.get(k, [])) - done
        rem_by[p] = rem
        per[arm_id(p)] = {
            "label": LABELS[p],
            "config": pp.cfg_name(cfg),
            "semfilter_plan": p in pp.space.SEMFILTER_PLANS,
            "pairs": len(pp.pairs_of(tab)[k]),
            "reused": len(reuse.get(k, [])),
            "reused_by_source": {"deployed_test": len(reuse.get(k, []))}
            if reuse.get(k)
            else {},
            "to_run": len(run.get(k, [])),
            "done_here": done,
            "remaining": rem,
            "mean_episode_wall_min": walls.get(p),
        }
    tot = lambda f: sum(v[f] for v in per.values())  # noqa: E731
    state = json.loads((pp.OUT / "state.json").read_text())
    return {
        "dataset": "tau2",
        "protocol": PROTOCOL,
        "run": str(pp.OUT),
        "out": str(out_dir()),
        "experiment": experiment(),
        "runtime": f"{regime()} ({pp.TEST_TRIALS} trial, pass^{pp.TEST_K}), retrieval slots {pp.RETRIEVAL_SLOTS}, "
        f"semantic workers {pp.SEMANTIC_WORKERS}",
        "concurrency": pp.TEST_CONCURRENCY,
        "quality_reuse": "on (own test index <experiment>:sfv2, as the deployed test)",
        "semfilter": SEMFILTER_PROTOCOL,
        "semfilter_knobs": "state.json block"
        if state.get("semfilter")
        else "frozen copy (fixed_plans_test.SEMFILTER_V2)",
        "deployed_test_done": (pp.OUT / "test_report.json").exists()
        and bool(state.get("semfilter")),
        "absent_plans": ABSENT,
        "arms": per,
        "pairs": tot("pairs"),
        "reused": tot("reused"),
        "to_run": tot("to_run"),
        "done_here": tot("done_here"),
        "remaining": tot("remaining"),
        "hours_estimate": estimate_hours(rem_by, walls, pp.TEST_CONCURRENCY),
        "estimate_note": f"remaining pairs x mean {regime()} episode wall of the plan's test pair files / "
        f"{pp.TEST_CONCURRENCY} (saturated throughput; quality-reuse hits make it shorter)",
        "reuse_rule": f"a deployed-test pair is reused iff it validates for the configuration, ran at {regime()} and "
        "(SemFilter plans) carries the semfilter-v2 receipt; counts change once the deployed sfv2 test has run",
    }


# ------------------------------------------------------------------ run


def build_arms(todo: dict[str, list[str]]) -> list:
    from trimem.experiments import quality_reuse as qr
    from trimem.experiments.tau2 import sfv2_rerun as sfv2

    exp = experiment()
    arms = []
    for k, qs in sorted(todo.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cfg = json.loads(k)
        arms.append(
            pp.Arm(
                f"fixed {cfg['plan']} {pp.cfg_name(cfg)}",
                out_dir() / "pairs" / pp.cfg_name(cfg),
                "test",
                qs,
                cfg,
                exp,
                pp.REUSE / "NoMemory",
                {
                    "bo_label": "test-fixed-plans",
                    "bo_experiment": pp.experiment_id(),
                    "quality": f"pass^{pp.K}",
                    "fixed_plan": cfg["plan"],
                    "retest": PROTOCOL,
                },
                reuse_index=qr.index_path(pp.REUSE_DIR, exp + sfv2.TAG, "test"),
                regime=regime(),
            )
        )
    return arms


def run(dry: bool = False) -> None:
    from trimem.experiments import memory_policy as mp
    from trimem.experiments.tau2 import sfv2_rerun as sfv2

    if mp.current() != mp.SUCCESS_FAILURE:
        raise SystemExit("set TRIMEM_MEMORY_POLICY=success+failure-v1")
    import os

    state = json.loads((pp.OUT / "state.json").read_text())
    # the recorded block's env, else the frozen copy (the sfv2 train rerun was reverted, user 2026-10-23); refuses
    # without TAU2_SEMFILTER=v2 / with other knobs
    env = semfilter_env(state)
    os.environ.update(env)
    _lock = None if dry else sfv2.driver_lock()
    cfgs, tab = build()
    tab = frozen_routing(tab, write=not dry)
    src = pair_sources(tab)
    _, todo = split_pairs(tab, src)
    todo = {k: [q for q in qs if (k, q) not in src] for k, qs in todo.items()}
    todo = {k: v for k, v in todo.items() if v}
    p = plan()
    if not dry:
        rec = {"started": datetime.now(UTC).isoformat(), "semfilter_env": env, **p}
        save(out_dir() / "plan.json", rec)
        with open(out_dir() / "plan_history.jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
    arms = build_arms(todo)
    pp.log(
        f"τ² fixed plans: {sum(map(len, todo.values()))} pairs to run ({p['reused']} reused from the deployed test, "
        f"{p['done_here']} already here) over {len(arms)} configurations at {pp.TEST_CONCURRENCY} concurrent "
        f"episodes ({regime()}) -> {out_dir()}"
    )
    if dry or not arms:
        return
    res = pp.run_parallel(arms, pp.TEST_CONCURRENCY)
    bad = [a.name for a in arms if res[a.name] < a.expected]
    if bad:
        raise SystemExit(f"τ² fixed-plan pairs incomplete: {bad}")
    pp.log("FIXED PLANS TEST DONE τ²")


# ------------------------------------------------------------------ report


def rows_of(tab: dict, src: dict) -> dict[tuple[str, str], dict]:
    """(config key, task) -> task summary (+ e2e_ms) of every present pair (as ``pool_profile.test_rows``)."""
    rows = {}
    for (k, q), (_, path) in src.items():
        e = pp.read_episode(path, key=q, trial=0, split="test", cfg=json.loads(k))
        t = pp.per_task({q: {0: e}}, [q], "test")
        if t:
            rows[(k, q)] = {**t[q], "e2e_ms": t[q]["latency_ms"]}
    return rows


def nomemory_quality() -> dict[str, float]:
    keys = pp.test_keys()
    t = pp.per_task(pp.ref_episodes(pp.REUSE / "NoMemory", keys), keys, "test")
    if t is None:
        raise SystemExit(f"{pp.REUSE / 'NoMemory'} lacks trial 0 of some test task")
    return {q: x["quality"] for q, x in t.items()}


def report() -> dict:
    from trimem.experiments import bo_pool_driver as bpd

    cfgs, tab = build()
    tab = frozen_routing(tab, write=False)
    src = pair_sources(tab)
    rows = rows_of(tab, src)
    nm = nomemory_quality()
    got = bpd.assemble(tab, rows, nm, key=pp.space.key)
    arms = {}
    for name, v in got.items():
        a = tab["arms"][name]
        k = pp.space.key(next(iter(a["routes"].values()))["config"])
        by: dict[str, int] = {}
        for q in a["routes"]:
            if (k, q) in src:
                by[src[(k, q)][0]] = by.get(src[(k, q)][0], 0) + 1
        arms[name] = {
            **v,
            "plan": a["plan"],
            "label": a["label"],
            "config": pp.cfg_name(next(iter(a["routes"].values()))["config"]),
            "pair_sources": by,
            "reuse_records": sum(
                1 for q in a["routes"] if (k, q) in rows and _is_reuse(src[(k, q)][1])
            ),
        }
    trp = pp.OUT / "test_report.json"
    refs = json.loads(trp.read_text()).get("references") if trp.exists() else None
    pl = out_dir() / "plan.json"
    rep = {
        "generated": datetime.now(UTC).isoformat(),
        "protocol": PROTOCOL,
        "dataset": "tau2",
        "experiment": experiment(),
        "test_trials": pp.TEST_TRIALS,
        "quality": f"pass^{pp.TEST_K}",
        "test_concurrency": pp.TEST_CONCURRENCY,
        "regime": regime(),
        "semfilter": SEMFILTER_PROTOCOL,
        "semfilter_knobs": {k: v for k, v in SEMFILTER_V2.items() if k != "env"},
        "configs": {p: pp.cfg_name(c) for p, c in cfgs.items()},
        "absent_plans": ABSENT,
        "plan": json.loads(pl.read_text()) if pl.exists() else None,
        "baseline": {
            "dir": str(pp.REUSE / "NoMemory"),
            "trials_used": [0],
            "quality": sum(nm.values()) / len(nm),
        },
        "latency_definition": "episode wall time (retrieval + episode incl. the user simulator)",
        "pairs": len(rows),
        "pair_sources": {
            s: sum(1 for x in src.values() if x[0] == s)
            for s in sorted({x[0] for x in src.values()})
        },
        "complete": all(v.get("covered") == v.get("n") for v in arms.values()),
        "arms": arms,
        "references": refs,
        "reference_source": str(trp),
    }
    save(out_dir() / "report.json", rep)
    (out_dir() / "report.md").write_text("\n".join(markdown(rep)) + "\n")
    return rep


def _is_reuse(path: Path) -> bool:
    from trimem.experiments.tau2 import runner as R

    r = R.read_result(path)
    return bool(r and r.get("reuse"))


def _f(x, fmt: str) -> str:
    return "–" if x is None else format(x, fmt)


def markdown(rep: dict) -> list[str]:
    L = [
        f"# τ²-bench · fixed-plan reference arms on the test set ({rep['regime']}, {rep['quality']})",
        "",
        f"Generated {rep['generated']}. Each arm = the plan's default configuration on every one of the 100 test tasks "
        f"(trial 0); {rep['pairs']} pairs (sources {rep['pair_sources']}); SemFilter {rep['semfilter']}; complete: "
        f"{rep['complete']}. Gain = paired vs the trial-0 NoMemory. Absent: "
        + "; ".join(f"{p} ({why})" for p, why in rep["absent_plans"].items()),
        "",
        "| arm | covered | pass^1 ± SE | gain vs NoMemory ± SE | tokens | latency s | agent calls ± SE | calls (succ.) "
        "| pair sources | reuse records |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for n, x in rep["arms"].items():
        if x.get("quality") is None:
            L.append(f"| {x['label']} | 0/{x['n']} | pending | | | | | | | |")
            continue
        L.append(
            f"| {x['label']} | {x['covered']}/{x['n']} | {x['quality']:.3f} ± {_f(x['quality_se'], '.3f')} | "
            f"{_f(x.get('paired_gain_vs_nomemory'), '+.3f')} ± {_f(x.get('paired_gain_se'), '.3f')} | "
            f"{x['tokens']:,.0f} | {x['latency_s']:.1f} | {_f(x.get('steps'), '.2f')} ± {_f(x.get('steps_se'), '.2f')} | "
            f"{_f(x.get('steps_success'), '.2f')} | {x['pair_sources']} | {x['reuse_records']} |"
        )
    for n, r in (rep.get("references") or {}).items():
        if r:
            L.append(
                f"| ref {n} | {r.get('n', '')} | {_f(r.get('q'), '.3f')} ± {_f(r.get('q_se'), '.3f')} | | "
                f"{_f(r.get('tok'), ',.0f')} | {_f(r.get('lat'), '.1f')} | {_f(r.get('steps'), '.2f')} | | | |"
            )
    return L


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("plan", "run", "report"))
    ap.add_argument(
        "--dry", action="store_true", help="run: log only, write nothing, start nothing"
    )
    ap.add_argument(
        "--summary", action="store_true", help="plan: compact text instead of JSON"
    )
    a = ap.parse_args(argv)
    if a.stage == "plan":
        from trimem.experiments.fixed_plans_test import summary_lines

        p = plan()
        print(
            "\n".join(summary_lines(p))
            if a.summary
            else json.dumps(p, indent=1, default=float)
        )
    elif a.stage == "run":
        run(a.dry)
    else:
        print("\n".join(markdown(report())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
