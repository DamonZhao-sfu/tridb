"""Generic driver: three-objective session BO profiling + frozen test on AppWorld and τ²-bench
(plan: docs/trimem_three_dataset_plan_v0.1.0.md, section 6).

Same protocol as the ALFWorld / ScienceWorld driver (``bo_agentenv_driver``, which stays the record of
those runs and is only imported from here); what differs per dataset lives in ``bo_newenv_datasets``.

``profile`` (training panel only, opt split = train; AppWorld dev tasks map to split train):
  1. No-Memory on the profiling panel = the paired baseline of every profiled configuration;
  2. round 0 = the 4 plans at the default knobs + 6 Sobol configurations;
  3. rounds 1..N = 5 qLogNEHVI proposals each (``session_bo_agentenv.propose_batch``), fitted only on
     the evaluated training rows of ``<ds>:<panel>:bo3obj-v1``; optional LLM analyzer channel;
  4. the default VGR again under ``<experiment>:noise-repeat`` (noise floor; never a BO input);
  5. ``profile_complete.json``.
  Quality reuse is ON (``--reuse_identical_memory``: identical injected memory reuses the earlier
  episode of the same query within the experiment and panel), never for the noise repeat.
``test``: freeze Q-best / Knee / BO-only best from the training measurements (``selection.json``,
written before any test episode), then run ``--arms`` (default qbest) on the test split (opt split
test; never read by the optimizer), paired with the test No-Memory arm.

Completion rule (one per query): the runner's result file is valid AND, for memory arms, exactly one
evaluated opt row exists for this experiment and ``protocol.space_config``. Result files without
their row are moved to ``_orphans/`` and rerun. Stall watchdog: no new result file for the dataset's
``stall_s`` -> SIGTERM the arm's process groups and resume (3 attempts). Each arm ends with its
process groups verified gone (AppWorld / τ² environments live inside the runner processes).

Run inside trimem-eval.slice with ``.venv-optimizer``:
    scripts/in_eval_slice.sh bo-appworld results/bo_agentenv_20261016/appworld/driver.log \\
        .venv-optimizer/bin/python -m trimem.experiments.bo_newenv_driver profile --dataset appworld
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from trimem.experiments import bo_agentenv_driver as agentenv
from trimem.experiments import bo_newenv_datasets as dsets
from trimem.experiments import session_bo_agentenv as bo
from trimem.optimizer import session_observations as hb
from trimem.optimizer import session_space_agentenv as space

ROOT = dsets.ROOT
OPT_DSN = dsets.OPT_DSN
ATTEMPTS = 3
ROUNDS = 2  # user choice for ScienceWorld (rounds 0-2 = 20 configurations)
TEST_ARMS = ("qbest", "knee", "onlybest")
log, save, kill_group = agentenv.log, agentenv.save, agentenv.kill_group


# ------------------------------------------------------------------ opt rows (read-only)


def evaluated_queries(experiment: str, cfg: dict | None) -> dict[str, int]:
    """query_id -> evaluated rows of this experiment (and configuration, unless ``cfg`` is None)."""
    import psycopg

    sql = "SELECT query_id, count(*) FROM opt.session_retrieval WHERE experiment_id = %s AND status = 'evaluated'"
    args: tuple = (experiment,)
    if cfg is not None:
        sql += " AND protocol->'space_config' = %s::jsonb"
        args += (json.dumps(cfg),)
    with psycopg.connect(OPT_DSN) as c:
        return {q: n for q, n in c.execute(sql + " GROUP BY 1", args).fetchall()}


def train_rows(experiment: str) -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(OPT_DSN, row_factory=dict_row) as c:
        rows = c.execute(
            "SELECT experiment_id, dataset, split, query_id, status, config, protocol, paired_gain, "
            "total_tokens, e2e_ms, quality FROM opt.session_retrieval WHERE experiment_id = %s "
            "AND split = 'train' AND status = 'evaluated'",
            (experiment,),
        ).fetchall()
    return [
        {
            **r,
            "total_tokens": float(r["total_tokens"])
            if r["total_tokens"] is not None
            else None,
        }
        for r in rows
    ]


def load_points(
    rows: list[dict],
    *,
    panel_ids: list[str],
    dataset: str,
    panel_hash: str,
    experiment: str,
) -> list[hb.Point]:
    """``session_bo_agentenv.load_points`` for an explicit experiment id (that one is fixed to
    ``<ds>:train100:bo3obj-v1``). Training rows only; declared vs executed configuration checked."""
    mapped = []
    for r in rows:
        if r["split"] != "train":
            raise ValueError("BO input must be training rows only")
        if r["experiment_id"] != experiment:
            continue
        if r["protocol"].get("space_version") != space.VERSION:
            raise ValueError("Row profiled under a different space version")
        declared = space.validate(r["protocol"]["space_config"])
        if bo.executed(r["config"]) != space.runner_args(declared):
            raise ValueError(
                f"Executed configuration {r['config']} differs from the declared {declared}"
            )
        mapped.append({**r, "config": declared})
    return hb.aggregate(
        mapped,
        panel_ids=panel_ids,
        dataset=dataset,
        panel_hash=panel_hash,
        experiment_id=experiment,
    )


def points(
    d: dsets.Dataset, pnl: dict, rows: list[dict] | None = None
) -> list[hb.Point]:
    return load_points(
        train_rows(d.experiment_id()) if rows is None else rows,
        panel_ids=[x["item_id"] for x in pnl["items"]],
        dataset=d.name,
        panel_hash=pnl["panel_hash"],
        experiment=d.experiment_id(),
    )


# ------------------------------------------------------------------ completion + arm runner


def completed(
    d: dsets.Dataset,
    out: Path,
    expected: list[str],
    session: dict | None,
    *,
    evaluated: Callable = evaluated_queries,
) -> set[str]:
    """Done = valid result file AND (memory arms) exactly one evaluated opt row of this experiment
    and configuration. Files without their row are moved to ``_orphans/`` (episode rerun)."""
    files = d.result_files(out)
    if session is None:
        return set(files) & set(expected)
    bad = d.unretrievable(out)
    if bad:
        raise SystemExit(
            f"{len(bad)} memory episodes never retrieved (no first user message): {bad[:3]}; "
            "they can never get an opt row - inspect before resuming"
        )
    ev = evaluated(session["experiment"], session["protocol_extra"]["space_config"])
    dup = [q for q, n in ev.items() if n > 1]
    if dup:
        raise SystemExit(
            f"duplicate evaluated rows for {dup[:5]} in {session['experiment']}; reconcile first"
        )
    orphan = [q for q in files if q not in ev]
    if orphan:
        dest = d.move_orphans(
            out, files, orphan, datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        )
        log(f"moved {len(orphan)} result files without an evaluated opt row to {dest}")
    return (set(files) - set(orphan)) & set(ev) & set(expected)


def routed_completed(
    d: dsets.Dataset,
    out: Path,
    expected: list[str],
    session: dict,
    *,
    evaluated: Callable = evaluated_queries,
    foreign: Callable | None = None,
) -> set[str]:
    """Router arms: one evaluated row per query of the router experiment (configurations differ per
    query), and every row of that experiment from this artifact."""
    files = d.result_files(out)
    other = (foreign or _foreign_router_rows)(
        session["experiment"], session["protocol_extra"]["router_sha"]
    )
    if other:
        raise SystemExit(
            f"{other} rows of {session['experiment']} come from another router artifact"
        )
    ev = evaluated(session["experiment"], None)
    dup = [q for q, n in ev.items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate evaluated rows for {dup[:5]}; reconcile first")
    orphan = [q for q in files if q not in ev]
    if orphan:
        dest = d.move_orphans(
            out, files, orphan, datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        )
        log(
            f"moved {len(orphan)} routed result files without an evaluated opt row to {dest}"
        )
    return (set(files) - set(orphan)) & set(ev) & set(expected)


def _foreign_router_rows(experiment: str, sha: str) -> int:
    import psycopg

    with psycopg.connect(OPT_DSN) as c:
        return c.execute(
            "SELECT count(*) FROM opt.session_retrieval WHERE experiment_id = %s AND "
            "protocol->>'router_sha' IS DISTINCT FROM %s",
            (experiment, sha),
        ).fetchone()[0]


def arm_commands(
    d: dsets.Dataset,
    out: Path,
    items: list[str],
    gpus: tuple[int, ...],
    *,
    stage: str,
    session: dict | None,
) -> list[tuple[int, list[str]]]:
    """(gpu, command) per shard. ``full`` = one shard covering the whole split / panel."""
    shards = d.shards(items, gpus, stage=stage)
    full = len(shards) == 1 and (
        sorted(items) == sorted(d.panel_ids() if stage == "profile" else d.test_ids())
    )
    return [
        (
            s.gpu,
            d.command(
                s,
                out,
                session,
                d.selection_args(s, stage=stage, full=full, workdir=d.out / "shards"),
            ),
        )
        for s in shards
    ]


def run_arm(
    d: dsets.Dataset,
    name: str,
    out: Path,
    items: list[str],
    gpus: tuple[int, ...],
    *,
    stage: str,
    session: dict | None,
    done: Callable[[], set[str]] | None = None,
) -> int:
    """All shards of one arm with the stall watchdog; returns the number of completed queries."""
    out.mkdir(parents=True, exist_ok=True)
    done = done or (lambda: completed(d, out, items, session))
    env = {**os.environ, **d.env()}
    total = len(items)
    cmds = arm_commands(d, out, items, gpus, stage=stage, session=session)
    for attempt in range(1, ATTEMPTS + 1):
        n = len(done())
        if n >= total:
            break
        log(
            f"start {d.name} {name} (attempt {attempt}, {n}/{total} done, GPUs {list(gpus)})"
        )
        procs = []
        for gpu, cmd in cmds:
            logf = open(out / f"shard_gpu{gpu}.log", "a")
            procs.append(
                subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        last, t = len(d.result_files(out)), time.time()
        try:
            while any(p.poll() is None for p in procs):
                time.sleep(20)
                n = len(d.result_files(out))
                if n != last:
                    last, t = n, time.time()
                if time.time() - t > d.stall_s:
                    log(
                        f"stall: no new result for {d.stall_s}s at {n}/{total}; stopping the arm"
                    )
                    break
        finally:
            for p in (
                procs
            ):  # every group reaped and verified gone (environments live inside them)
                kill_group(p)
            for p in procs:
                p.wait()
        codes = [p.returncode for p in procs]
        if any(codes):
            log(
                f"{d.name} {name}: shard exit codes {codes} (RSS stop / failed tasks resume next attempt)"
            )
    n = len(done())
    log(f"done {d.name} {name}: {n}/{total}")
    return n


# ------------------------------------------------------------------ sessions


def cfg_name(cfg: dict) -> str:
    return "cfg_" + hashlib.sha256(space.key(cfg).encode()).hexdigest()[:12]


def session_args(
    d: dsets.Dataset,
    cfg: dict,
    *,
    experiment: str,
    baseline_dir: Path,
    label: str,
    pnl: dict,
    reuse: bool,
) -> dict:
    ra = space.runner_args(cfg)
    return {
        **ra,
        "split": "train",
        "experiment": experiment,
        "baseline_dir": baseline_dir,
        "reuse": reuse,
        "protocol_extra": {
            "panel_hash": pnl["panel_hash"],
            "space_version": space.VERSION,
            "space_config": cfg,
            "bo_label": label,
            "bo_experiment": d.experiment_id(),
        },
    }


def test_session(d: dsets.Dataset, cfg: dict, name: str) -> dict:
    ra = space.runner_args(cfg)
    return {
        **ra,
        "split": "test",
        "experiment": d.test_experiment(name),
        "baseline_dir": d.test_nomemory,
        "reuse": False,
        "protocol_extra": {
            "space_version": space.VERSION,
            "space_config": cfg,
            "bo_label": name,
        },
    }


def check_panel(d: dsets.Dataset) -> dict:
    pnl = d.panel()
    ids = [x["item_id"] for x in pnl["items"]]
    if len(set(ids)) != len(ids) or not ids:
        raise SystemExit(f"{d.panel_path}: empty panel or duplicate ids")
    test = set(d.test_ids())
    if test & set(ids):
        raise SystemExit(f"{d.panel_path}: test items inside the profiling panel")
    return pnl


# ------------------------------------------------------------------ analyzer (optional)

SUBMETRIC_SQL = agentenv.SUBMETRIC_SQL


def episode_submetrics(row: dict) -> dict[str, float]:
    """Dataset-neutral subset of ``session_submetrics`` (no task-type / object / 30-turn metrics,
    which are ALFWorld / ScienceWorld notions)."""
    import statistics as st

    from trimem.experiments.semantic_rubric import FILTER_THRESHOLD

    rec, gen = row["receipt"], row["generation"] or {}
    cards = rec.get("cards") or []
    out = {
        "pool_size": float(len(rec.get("pool_ids") or [])),
        "seed_sessions": float(len(rec.get("seed_ids") or [])),
        "empty_rate": float(not cards),
        "sem_tokens": float(gen.get("semantic_tokens") or 0),
        "memory_chars": float(gen.get("memory_chars") or 0),
        "llm_calls": float(gen.get("llm_calls") or 0),
    }
    if gen.get("llm_calls"):
        out["tokens_per_call"] = (
            float(gen.get("generation_tokens") or 0) / gen["llm_calls"]
        )
    if cards:
        out["sel_cosine"] = st.mean(float(c.get("cosine") or 0) for c in cards)
        out["card_steps"] = st.mean(float(c.get("n_steps") or 0) for c in cards)
    dec = rec.get("semantic_decisions") or []
    if dec:
        out["filter_keep"] = st.mean(
            float((x.get("score") or 0) >= FILTER_THRESHOLD) for x in dec
        )
    return out


def config_submetrics(rows: list[dict]) -> dict[str, float]:
    import statistics as st

    per = [episode_submetrics(r) for r in rows]
    keys = {k for p in per for k in p}
    return {
        k: st.mean(p[k] for p in per if k in p)
        for k in sorted(keys)
        if sum(k in p for p in per) >= len(per) / 2
    }


def analyzer_advice(
    d: dsets.Dataset, pnl: dict, pts: list, state: dict, forced: bool, r: int
) -> tuple:
    import psycopg
    from psycopg.rows import dict_row

    from trimem.optimizer import analyzer_session as az
    from trimem.optimizer import session_pool as pool

    with psycopg.connect(OPT_DSN, row_factory=dict_row) as c:
        rows = c.execute(SUBMETRIC_SQL, (d.experiment_id(),)).fetchall()
    by: dict[str, list] = {}
    for row in rows:
        by.setdefault(space.key(row["cfg"]), []).append(row)
    pk = {space.key(p.cfg): p for p in pts}
    order = [space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]]
    evals = [
        az.Eval(
            f"e{i:02d}",
            pk[k].cfg,
            pk[k].quality,
            math.sqrt(pk[k].noise[0]),
            pk[k].tokens,
            pk[k].latency_ms,
            config_submetrics(by.get(k, [])),
            math.sqrt(pk[k].noise[1]),
            math.sqrt(pk[k].noise[2]),
        )
        for i, k in enumerate(order)
        if k in pk
    ]
    model, _ = pool.fit(pts)
    rect = az.max_hv_rectangle(evals, "quality" if r % 2 else "cost")
    nomem = [
        d.record(f)["quality"] for f in d.result_files(d.arm_dir("NoMemory")).values()
    ]
    ctx = {
        "dataset": d.name,
        "nomemory_quality": sum(nomem) / len(nomem),
        "panel": f"frozen profiling panel of {len(pnl['items'])} tasks; quality = {d.quality_def}",
    }
    an = az.Analyzer(
        OPT_DSN, d.experiment_id(), notes_path=d.out / "analyzer_notes.txt"
    )
    if forced and rect:
        res, label = an.explore_gap(evals, ctx, rect, rnd=r), "agent-gap"
    else:
        res, label = (
            an.propose(
                evals, ctx, rnd=r, confidence=az.confidence_text(model), rect=rect
            ),
            "agent",
        )
    log(
        f"analyzer round {r} ({label}): {res['status']}, {len(res['candidates'])} candidates, "
        f"{res['latency_ms'] / 1000:.0f}s, call {res['call_id']}"
    )
    front = [tuple(e) for e in rect["endpoints"]] if (forced and rect) else None
    return (
        [{**cfg, "_label": label} for cfg in res["candidates"]],
        front,
        {
            k: res[k]
            for k in (
                "call_id",
                "status",
                "reasoning",
                "errors",
                "input_tokens",
                "output_tokens",
            )
        },
    )


def pending_user_probes(d: dsets.Dataset, state: dict, cap: int = 2) -> list[dict]:
    f = d.out / "user_probes.json"
    if not f.exists():
        return []
    planned = {space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]}
    out: list[dict] = []
    for c in json.load(open(f))["configs"]:
        cfg = space.validate(c)
        if space.key(cfg) not in planned and space.key(cfg) not in {
            space.key(x) for x in out
        }:
            out.append(cfg)
    return out[:cap]


# ------------------------------------------------------------------ stages


def propose_round(
    d: dsets.Dataset, pnl: dict, state: dict, r: int, use_analyzer: bool
) -> dict:
    if r == 0:
        picks = [
            {"config": c, "channel": "default" if i < len(bo.defaults()) else "sobol"}
            for i, c in enumerate(bo.round0())
        ]
        return {
            "round": 0,
            "proposed_at": datetime.now(UTC).isoformat(),
            "configs": picks,
            "mode": "ordinary",
            "effective_mode": ["ordinary"],
            "analyzer": None,
        }
    pts = points(d, pnl)
    measured = {space.key(p.cfg) for p in pts}
    planned = {space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]}
    if planned - measured:
        raise SystemExit(
            f"round {r}: {len(planned - measured)} planned configurations lack a complete panel"
        )
    by_key = {space.key(p.cfg): p for p in pts}
    hist = [
        [by_key[space.key(c["config"])] for c in rr["configs"]]
        for rr in state["rounds"]
    ]
    forced = bo.stalled(hist)
    agent, gap_front, advice = [], None, None
    if use_analyzer:
        agent, gap_front, advice = analyzer_advice(d, pnl, pts, state, forced, r)
    probes = pending_user_probes(d, state)
    picks = [{"config": c, "channel": "user-probe", "mode": "user"} for c in probes]
    picks += bo.propose_batch(
        pts,
        bo.context(d.name, pnl["panel_hash"]),
        k=bo.BATCH - len(probes),
        seed=1000 * r,
        forced=forced,
        agent_rows=agent or None,
        realized_front=gap_front,
        exclude={space.key(c) for c in probes},
    )
    return {
        "round": r,
        "proposed_at": datetime.now(UTC).isoformat(),
        "configs": picks,
        "mode": "forced (stagnation)" if forced else "ordinary",
        "effective_mode": sorted({p.get("mode", "ordinary") for p in picks}),
        "analyzer": advice if use_analyzer else None,
    }


def profile(
    d: dsets.Dataset,
    *,
    use_analyzer: bool = False,
    rounds: int = ROUNDS,
    gpus: tuple[int, ...] = (0, 1),
) -> None:
    pnl = check_panel(d)
    items = [x["item_id"] for x in pnl["items"]]
    exp = d.experiment_id()
    d.out.mkdir(parents=True, exist_ok=True)
    state_p = d.out / "state.json"
    state = (
        json.load(open(state_p))
        if state_p.exists()
        else {
            "dataset": d.name,
            "panel_hash": pnl["panel_hash"],
            "experiment": exp,
            "rounds": [],
        }
    )
    if state["panel_hash"] != pnl["panel_hash"] or state["experiment"] != exp:
        raise SystemExit("panel or experiment changed since this run started")
    nomem = d.arm_dir("NoMemory")
    if run_arm(d, "NoMemory", nomem, items, gpus, stage="profile", session=None) < len(
        items
    ):
        raise SystemExit("No-Memory baseline incomplete")
    for r in range(rounds + 1):
        if len(state["rounds"]) <= r:
            state["rounds"].append(propose_round(d, pnl, state, r, use_analyzer))
            save(state_p, state)
            log(
                f"round {r} configurations: "
                + "; ".join(
                    space.key(p["config"]) for p in state["rounds"][r]["configs"]
                )
            )
        for j, p in enumerate(state["rounds"][r]["configs"]):
            cfg = p["config"]
            label = f"r{r}c{j}:{space.DISPLAY[cfg['plan']]}"
            args = session_args(
                d,
                cfg,
                experiment=exp,
                baseline_dir=nomem,
                label=label,
                pnl=pnl,
                reuse=True,
            )
            if run_arm(
                d,
                label,
                d.arm_dir(cfg_name(cfg)),
                items,
                gpus,
                stage="profile",
                session=args,
            ) < len(items):
                raise SystemExit(
                    f"{label} incomplete after {ATTEMPTS} attempts; resume this driver to retry"
                )
    rep = bo.defaults()[0]
    args = session_args(
        d,
        rep,
        experiment=exp + ":noise-repeat",
        baseline_dir=nomem,
        label="noise-repeat:VGR",
        pnl=pnl,
        reuse=False,
    )
    if run_arm(
        d,
        "noise-repeat VGR",
        d.arm_dir("noise_repeat_VGR"),
        items,
        gpus,
        stage="profile",
        session=args,
    ) < len(items):
        raise SystemExit("noise repeat incomplete")
    if (
        len(state["rounds"]) > rounds + 1
    ):  # --rounds lowered after later rounds were proposed (never run)
        state["rounds"] = state["rounds"][: rounds + 1]
        save(state_p, state)
    planned = {space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]}
    measured = {space.key(p.cfg) for p in points(d, pnl)}
    expected = bo.ROUND0_TOTAL + bo.BATCH * rounds
    if (
        len(state["rounds"]) != rounds + 1
        or len(planned) != expected
        or not planned <= measured
    ):
        raise SystemExit(
            f"profiling incomplete: {len(planned & measured)} measured of {len(planned)} planned"
        )
    save(
        d.out / "profile_complete.json",
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "configs": len(planned),
            "panel_hash": pnl["panel_hash"],
            "experiment": exp,
            "space_version": space.VERSION,
            "bo_rounds": rounds,
            "gpus": list(gpus),
            "quality": d.quality_def,
        },
    )
    log(f"PROFILING DONE {d.name}: {len(planned)} complete configurations")


def per_query_gain(rows: list[dict], cfg: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        if r["protocol"].get("space_config") == cfg:
            if r["query_id"] in out:
                raise SystemExit(
                    f"duplicate evaluated row for {r['query_id']} in {r['experiment_id']}"
                )
            out[r["query_id"]] = r["paired_gain"]
    return out


def freeze_selection(
    d: dsets.Dataset, pnl: dict, state: dict, rows: list[dict], repeat_rows: list[dict]
) -> dict:
    """Q-best / Knee (within the noise floor of Q-best) / best of the BO rounds' own proposals,
    from training measurements only."""
    exp = d.experiment_id()
    pts = load_points(
        rows,
        panel_ids=[x["item_id"] for x in pnl["items"]],
        dataset=d.name,
        panel_hash=pnl["panel_hash"],
        experiment=exp,
    )
    rep = bo.defaults()[0]
    a, b = per_query_gain(rows, rep), per_query_gain(repeat_rows, rep)
    ids = [x["item_id"] for x in pnl["items"]]
    if set(a) != set(ids) or set(b) != set(ids):
        raise SystemExit("default VGR and its repeat must both cover the full panel")
    floor = bo.noise_floor([a[q] for q in ids], [b[q] for q in ids])
    sel = bo.select(pts, floor)
    bo_keys = {
        space.key(c["config"])
        for rr in state["rounds"][1:]
        for c in rr["configs"]
        if c.get("channel") != "user-probe"
    }
    bo_pts = [p for p in pts if space.key(p.cfg) in bo_keys]
    best = max(bo_pts, key=lambda p: p.quality) if bo_pts else None

    def pick(p):
        return {
            "config": p.cfg,
            "gain": p.quality,
            "tokens": p.tokens,
            "latency_ms": p.latency_ms,
        }

    return {
        "panel_hash": pnl["panel_hash"],
        "experiment": exp,
        "space_version": space.VERSION,
        "frozen_at": datetime.now(UTC).isoformat(),
        "noise_floor": floor,
        "n_configs": len(pts),
        "q_best": pick(sel["q_best"]),
        "knee": pick(sel["knee"]),
        "near_q_best": sel["near_q_best"],
        "bo_only_best": None if best is None else pick(best),
    }


def test_arm_list(sel: dict, wanted: tuple[str, ...]) -> list[tuple[str, dict]]:
    """(arm name, configuration) in order; an arm whose configuration equals an earlier one is skipped."""
    arms = []
    if "qbest" in wanted:
        arms.append(("BO-Qbest", sel["q_best"]["config"]))
    if "knee" in wanted:
        arms.append(("BO-Knee", sel["knee"]["config"]))
    if "onlybest" in wanted and sel.get("bo_only_best"):
        arms.append(("BO-onlybest", sel["bo_only_best"]["config"]))
    unknown = set(wanted) - set(TEST_ARMS)
    if unknown:
        raise SystemExit(f"unknown test arms {sorted(unknown)}")
    out, seen = [], set()
    for name, cfg in arms:
        k = space.key(cfg)
        if k in seen:
            log(f"{name} has the same configuration as an earlier test arm; not rerun")
            continue
        seen.add(k)
        out.append((name, cfg))
    return out


def test(
    d: dsets.Dataset,
    arms_wanted: tuple[str, ...] = ("qbest",),
    gpus: tuple[int, ...] | None = None,
) -> None:
    gpus = gpus or d.test_gpus
    pnl = d.panel()
    ident = {
        "panel_hash": pnl["panel_hash"],
        "experiment": d.experiment_id(),
        "space_version": space.VERSION,
    }
    done_p = d.out / "profile_complete.json"
    if not done_p.exists() or any(
        json.load(open(done_p))[k] != v for k, v in ident.items()
    ):
        raise SystemExit(
            "training profiling is not complete for this panel/experiment; no selection, no test"
        )
    sel_p = d.out / "selection.json"
    if not sel_p.exists():  # frozen before any test episode
        state = json.load(open(d.out / "state.json"))
        save(
            sel_p,
            freeze_selection(
                d,
                pnl,
                state,
                train_rows(d.experiment_id()),
                train_rows(d.experiment_id() + ":noise-repeat"),
            ),
        )
    sel = json.load(open(sel_p))
    if any(sel.get(k) != v for k, v in ident.items()):
        raise SystemExit("selection.json belongs to a different panel/experiment/space")
    test_ids = d.test_ids()
    if len(set(d.result_files(d.test_nomemory)) & set(test_ids)) < len(test_ids):
        raise SystemExit(
            f"test No-Memory ({d.test_nomemory}) is incomplete: the paired baseline must exist first"
        )
    for name, cfg in test_arm_list(sel, arms_wanted):
        if run_arm(
            d,
            f"test {name}",
            d.test_dir(name),
            test_ids,
            gpus,
            stage="test",
            session=test_session(d, cfg, name),
        ) < len(test_ids):
            raise SystemExit(f"test {name} incomplete")
    log(f"TEST DONE {d.name}")


def lock(d: dsets.Dataset):
    d.out.mkdir(parents=True, exist_ok=True)
    f = open(d.out / "driver.lock", "w")
    try:  # one driver per dataset: it owns the state and the opt rows being written
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another driver holds {d.out / 'driver.lock'}") from None
    return f


def parse_gpus(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    g = tuple(int(x) for x in text.split(","))
    if not g or len(set(g)) != len(g) or any(x not in (0, 1) for x in g):
        raise SystemExit("--gpus: comma-separated subset of 0,1")
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("profile", "test", "panel"))
    ap.add_argument("--dataset", choices=tuple(dsets.DATASETS), required=True)
    ap.add_argument(
        "--analyzer",
        action="store_true",
        help="enable the LLM analyzer channel (Codex Luna)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="profile: last BO round (default 2 = 20 configurations)",
    )
    ap.add_argument(
        "--arms", default="qbest", help="test: frozen arms to run (qbest,knee,onlybest)"
    )
    ap.add_argument(
        "--gpus",
        default=None,
        help="GPUs of every arm (profile default 0,1; test default: the "
        "dataset's test-arm GPU)",
    )
    ap.add_argument(
        "--per_gpu",
        type=int,
        default=None,
        help="AppWorld workers / τ² concurrency per GPU (default 8)",
    )
    a = ap.parse_args(argv)
    d = dsets.get(a.dataset)
    if a.per_gpu:
        d.per_gpu = a.per_gpu
    if (
        a.stage == "panel"
    ):  # τ² frozen panel file (AppWorld: trimem.experiments.appworld.build_panel)
        if not isinstance(d, dsets.Tau2):
            raise SystemExit(
                "panel: only τ² (AppWorld uses trimem.experiments.appworld.build_panel)"
            )
        p = d.write_panel()
        log(f"{d.panel_path}: {p['size']} items, hash {p['panel_hash']}")
        return 0
    held = lock(d)  # noqa: F841 - held for the process lifetime

    def stop(
        signum, frame
    ):  # SIGTERM -> exception, so run_arm's finally reaps the runner groups
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if a.stage == "profile":
        profile(
            d,
            use_analyzer=a.analyzer,
            rounds=a.rounds,
            gpus=parse_gpus(a.gpus) or (0, 1),
        )
    else:
        test(d, tuple(x for x in a.arms.split(",") if x), parse_gpus(a.gpus))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
