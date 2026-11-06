"""Driver: three-objective BO profiling on the ALFWorld / ScienceWorld / WebShop training panels, then the
frozen held-out test (plan: "BO 实验计划" in trajectory-result.html).

``test-fixed`` (WebShop, user 2026-10-18): No-Memory and the four plan defaults on the AgentEval 200 test
set under the controlled protocol, before profiling.

Per dataset, all on the frozen 100-query training panel (split=train):
  1. No-Memory (official evaluator) = the paired baseline of every profiled configuration;
  2. round 0 = the 4 plans at the default knobs + 6 Sobol configurations;
  3. rounds 1-6 = 5 qLogNEHVI proposals each (session_bo_agentenv.propose_batch), fitted only on the
     training rows of this BO experiment in opt;
  4. the default VGR again under a separate experiment id (noise floor; never a BO input).
``test``: freeze Q-best and Knee from the training measurements (selection.json, written before any
test episode), then run them, and the default VGR+Rank arm if missing, on the AgentEval 200 test set
under the controlled protocol (split=test; never read by the optimizer).

Controlled environment: one configuration at a time; panel items split by position parity, half A
on GPU0 (LLM 8030, ColBERT 8040, env A), half B on GPU1 (8031 / 8041 / env B), 8 processes each.
Stall watchdog (15 min without a new item -> SIGTERM the arm's process groups, restart env servers,
resume; at most 3 attempts). After every arm the env servers must report 0 live envs / 0 JVMs, then
they are restarted. Resumable: proposals are stored once in state.json; finished items are skipped
(official per-item files).

Run inside trimem-eval.slice with ``.venv-optimizer``:
    scripts/in_eval_slice.sh bo-alfworld results/bo_agentenv_20261016/alfworld/driver.log \\
        .venv-optimizer/bin/python -m trimem.experiments.bo_agentenv_driver profile --dataset alfworld

Larger panel (user 2026-10-17): ``--panel_name train500 --batch 10 --rounds 9`` profiles the 500-query
panel (the 100 frozen items first) under experiment ``<ds>:train500:bo3obj-v2`` in
``results/bo_agentenv_20261016/<ds>_train500/`` (own state / selection / profile_complete; round 0 = the
same 4 defaults + 6 Sobol, then 9 rounds of 10 kriging-believer picks = 100 configurations). Quality
reuse is on for both datasets and, by default, also reuses ORIGINAL episodes of the finished train100
experiment for the shared queries (``--reuse_from``; identical query, memory text, llm_budget and action
format; never the noise repeat, never test arms). The No-Memory baseline of the shared 100 items is
copied from the train100 run (same official evaluator, same items). Test arms are named BO500-*.
The driver lock stays ``<ds>/driver.lock`` for every panel (it guards the dataset's env servers).

Standalone larger panel (WebShop, user 2026-10-18): ``--panel_name train200`` = 200 product-isolated
training queries that do NOT start with the train100 items: experiment ``webshop:train200:bo3obj-v2``,
``results/bo_agentenv_20261016/webshop_train200/``, test arms BO200-*, quality reuse ON within the
experiment, its own No-Memory baseline (no seeding, no cross-experiment reuse by default), batch 5.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from trimem.experiments import memory_policy as mp
from trimem.experiments import session_bo_agentenv as bo
from trimem.optimizer import session_space_agentenv as space

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/bo_agentenv_20261016"
CTRL = ROOT / "results/eval_controlled_20261015"
OPT_DSN = "host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa"
DS = {
    "alfworld": {"fmt": "function_calling", "env": (36001, 36011)},
    "sciworld": {"fmt": "react", "env": (36002, 36012)},
    # WebShop (user 2026-10-18): AgentGym's default ReAct format (the format of AgentTraj-L), env servers
    # 36003 / 36013 (scripts/env_servers.sh webshop), same COMMON settings as ScienceWorld.
    "webshop": {"fmt": "react", "env": (36003, 36013)},
}
# Quality reuse on the 100-panel (identical injected memory -> reuse the episode): ScienceWorld (user
# 2026-10-16) and WebShop (user 2026-10-18, "same configuration as ScienceWorld"); never ALFWorld train100.
REUSE_TRAIN100 = ("sciworld", "webshop")
# Test arms at the plan defaults (``test --arms``), named as the controlled ALFWorld / ScienceWorld arms.
FIXED_ARMS = {"vgr": "VGR", "vgr_rank": "VGR+Rank", "p2": "P2", "p2_rank": "P2+Rank"}
LLM = (8030, 8031)
COMMON = [
    "--api_key",
    "EMPTY",
    "--model",
    "Qwen/Qwen3.8-27B-FP8",
    "--temperature",
    "0",
    "--max_round",
    "30",
    "--processes",
    "8",
]
STALL_S, ATTEMPTS = 900, 3
ROUNDS = 6


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


BASE_PANEL = "train100"
# Panels that start with the frozen train100 items (No-Memory seeding + default reuse of the train100
# experiment). train200 (WebShop) is standalone.
EXTENDS_BASE = ("train500",)


def panel_path(ds: str, panel_name: str = BASE_PANEL) -> Path:
    return ROOT / f"data/external/panels/{ds}_{panel_name}.json"


def panel(ds: str, panel_name: str = BASE_PANEL) -> dict:
    return json.load(open(panel_path(ds, panel_name)))


def run_dir(ds: str, panel_name: str = BASE_PANEL) -> Path:
    """Result directory of one profiling run: <ds>/ for the 100-panel, <ds>_<panel>/ otherwise; memory-policy
    tagged (``__mem-sf1`` for success+failure-v1, unchanged for success-only-v0)."""
    return mp.tag_dir(
        OUT / ds if panel_name == BASE_PANEL else OUT / f"{ds}_{panel_name}"
    )


def nomemory_dir(ds: str, panel_name: str = BASE_PANEL) -> Path:
    """No-Memory baseline of a panel: independent of the memory policy (never tagged, shared)."""
    return (
        OUT / ds if panel_name == BASE_PANEL else OUT / f"{ds}_{panel_name}"
    ) / "NoMemory"


def test_arm_dir(ds: str, name: str) -> Path:
    """Controlled test-arm directory: No-Memory untagged, memory arms memory-policy tagged."""
    d = CTRL / f"{ds}_{name}"
    return d if name == "NoMemory" else mp.tag_dir(d)


def test_experiment(ds: str, name: str) -> str:
    return mp.tag(f"{ds}:agenteval200:controlled-v1:{name}")


def arm_prefix(panel_name: str = BASE_PANEL) -> str:
    """Test-arm prefix: BO (100-panel) / BO500 (train500), so frozen arms of the two runs never collide."""
    return "BO" if panel_name == BASE_PANEL else "BO" + panel_name.removeprefix("train")


def default_batch(panel_name: str = BASE_PANEL) -> int:
    return bo.BATCH_TRAIN500 if panel_name in EXTENDS_BASE else bo.BATCH


def default_reuse_from(ds: str, panel_name: str = BASE_PANEL) -> list[str]:
    """Cross-experiment reuse sources: the finished 100-panel experiment for panels that extend it
    (train500); none for the 100-panel itself or a standalone panel (train200)."""
    return [bo.experiment_id(ds, BASE_PANEL)] if panel_name in EXTENDS_BASE else []


def source_config_dir(ds: str, experiment: str) -> Path:
    """Config directory of the run that wrote a source experiment (for rows without result_file)."""
    for pn in bo.PANEL_VERSIONS:
        if bo.experiment_id(ds, pn) == experiment:
            return run_dir(ds, pn) / "configs"
    raise SystemExit(f"unknown reuse source experiment {experiment!r}")


def seed_nomemory(ds: str, pnl: dict, dest: Path) -> int:
    """Copy the 100-panel No-Memory results (per-item files + their call-log lines) of the items a larger
    panel shares with it, so the paired baseline of a shared query is the one its reused episodes were
    paired with. Idempotent (marker file); returns the number of items copied."""
    marker = dest / "nomemory_seeded.json"
    if marker.exists():
        return json.load(open(marker))["n"]
    src = OUT / ds / "NoMemory"
    shared = [x["item_id"] for x in panel(ds, BASE_PANEL)["items"]]
    if [x["item_id"] for x in pnl["items"][: len(shared)]] != shared:
        raise SystemExit("the larger panel must start with the 100-panel items")
    files = item_files(src, ds)
    if set(shared) - set(files):
        raise SystemExit("train100 No-Memory baseline incomplete; cannot seed")
    dest.mkdir(parents=True, exist_ok=True)
    if set(item_files(dest, ds)) - set(shared):
        raise SystemExit(
            f"{dest} already holds other results; seed only into a fresh directory"
        )
    idx = {int(q.split("_")[1]) for q in shared}
    lines = (
        [
            ln
            for ln in open(src / "llm_calls.jsonl")
            if json.loads(ln).get("item") in idx
        ]
        if (src / "llm_calls.jsonl").exists()
        else []
    )
    for q in shared:
        shutil.copy2(files[q], dest / files[q].name)
    log_p = dest / "llm_calls.jsonl"
    have = (
        set(open(log_p)) if log_p.exists() else set()
    )  # idempotent if a previous seeding was interrupted
    with open(log_p, "a") as f:
        f.writelines(ln for ln in lines if ln not in have)
    save(
        marker,
        {
            "n": len(shared),
            "source": str(src),
            "items": shared,
            "call_log_lines": len(lines),
            "seeded_at": datetime.now(UTC).isoformat(),
        },
    )
    log(f"seeded {len(shared)} No-Memory results from {src}")
    return len(shared)


def shards(ds: str, items: list[str], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    a, b = out / "a.json", out / "b.json"
    a.write_text(json.dumps([{"item_id": i} for i in items[0::2]]))
    b.write_text(json.dumps([{"item_id": i} for i in items[1::2]]))
    return a, b


def item_files(d: Path, ds: str) -> dict[str, Path]:
    """Valid per-item result files (parseable JSON with the expected item id)."""
    out = {}
    for f in d.glob(f"{ds}_*.json"):
        if not f.stem.split("_")[-1].isdigit():
            continue
        try:
            if json.load(open(f))["item_id"] == f.stem:
                out[f.stem] = f
        except (ValueError, KeyError):
            continue
    return out


def evaluated_queries(experiment: str, cfg: dict) -> dict[str, int]:
    """query_id -> number of evaluated opt rows of this experiment and configuration."""
    with psycopg.connect(OPT_DSN) as c:
        rows = c.execute(
            "SELECT query_id, count(*) FROM opt.session_retrieval WHERE experiment_id = %s "
            "AND status = 'evaluated' AND protocol->'space_config' = %s::jsonb GROUP BY 1",
            (experiment, json.dumps(cfg)),
        ).fetchall()
    return {q: n for q, n in rows}


def completed(
    ds: str, out: Path, expected: list[str], session: dict | None
) -> set[str]:
    """One completion rule: a query is done iff its result file is valid AND (memory arms) exactly
    one evaluated opt row exists for it. Files without their row (a crash between the episode file
    and the database write) are moved to _orphans/ so the episode is rerun."""
    files = item_files(out, ds)
    if session is None:
        return set(files) & set(expected)
    ev = evaluated_queries(
        session["experiment"], session["protocol_extra"]["space_config"]
    )
    dup = [q for q, n in ev.items() if n > 1]
    if dup:
        raise SystemExit(
            f"duplicate evaluated rows for {dup[:5]} in {session['experiment']}; reconcile first"
        )
    orphan = [q for q in files if q not in ev]
    if orphan:
        dest = out / "_orphans" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        dest.mkdir(parents=True, exist_ok=True)
        for q in orphan:
            shutil.move(str(files[q]), dest / files[q].name)
            (out / "memory" / f"{q.split('_')[1]}.json").unlink(missing_ok=True)
        log(f"moved {len(orphan)} result files without an evaluated opt row to {dest}")
    return (set(files) - set(orphan)) & set(ev) & set(expected)


def get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def env_check_and_restart(ds: str) -> None:
    for port in DS[ds]["env"]:
        try:
            st = get(f"http://127.0.0.1:{port}/managed_stats")
            if st["live_envs"] or st["jvms"]:
                log(
                    f"WARNING env {port}: {st['live_envs']} live envs / {st['jvms']} JVMs after the arm"
                )
        except Exception as exc:
            log(f"WARNING env {port} stats unavailable: {exc!r}")
    subprocess.run(
        ["bash", str(ROOT / "scripts/env_servers.sh"), ds],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        timeout=900,
    )


def half_cmd(
    ds: str, half: int, out: Path, shard: Path, *, session: dict | None
) -> list[str]:
    llm, env = LLM[half], DS[ds]["env"][half]
    official = [
        "--action_format",
        DS[ds]["fmt"],
        *COMMON,
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
    py = [str(ROOT / ".venv-agentgym/bin/python"), "-m"]
    if session is None:
        return py + ["trimem.experiments.agentgym_official_eval", *official]
    return py + [
        "trimem.experiments.agentgym_session_eval",
        "--plan",
        session["plan"],
        "--edges",
        session["edges"],
        "--memory_config",
        json.dumps(session["memory_config"]),
        "--split",
        session["split"],
        "--baseline_dir",
        str(session["baseline_dir"]),
        "--experiment",
        session["experiment"],
        "--semantic_base",
        f"http://127.0.0.1:{llm}",
        "--protocol_extra",
        json.dumps(session["protocol_extra"]),
        *(["--panel", str(session["panel"])] if session.get("panel") else []),
        *(["--reuse_identical_memory"] if session.get("reuse") else []),
        *(
            ["--reuse_from", ",".join(session["reuse_from"])]
            if session.get("reuse_from")
            else []
        ),
        *(
            ["--reuse_from_dirs", ",".join(session["reuse_from_dirs"])]
            if session.get("reuse_from_dirs")
            else []
        ),
        *session.get(
            "runner_extra", []
        ),  # e.g. --space_v2 (pool profiling, session space v2)
        *official,
    ]


def kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the arm's whole process group (workers close their envs), then SIGKILL; verify it is gone.
    The leader is reaped while waiting: an unreaped (zombie) leader keeps the group id alive."""
    pgid = proc.pid
    for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 20)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            proc.poll()
            return
        for _ in range(wait):
            time.sleep(1)
            proc.poll()  # reap the leader if it exited
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
    log(f"WARNING process group {pgid} still has members after SIGKILL")


def run_arm(
    ds: str,
    name: str,
    out: Path,
    sa: Path,
    sb: Path,
    expected: list[str],
    *,
    session: dict | None,
) -> int:
    """Both halves of one arm with the stall watchdog; returns the number of completed queries."""
    out.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        mp.ENV_VAR: mp.current(),
    }  # runners use the driver's policy
    total = len(expected)
    for attempt in range(1, ATTEMPTS + 1):
        n = len(completed(ds, out, expected, session))
        if n >= total:
            break
        log(f"start {ds} {name} (attempt {attempt}, {n}/{total} done)")
        procs = []
        for half, shard in enumerate((sa, sb)):
            cmd = [
                "taskset",
                "-c",
                "16-31",
                "nice",
                "-n",
                "10",
                *half_cmd(ds, half, out, shard, session=session),
            ]
            logf = open(out / f"half_{'ab'[half]}.log", "a")
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
        last, t = len(item_files(out, ds)), time.time()
        try:
            while any(p.poll() is None for p in procs):
                time.sleep(20)
                n = len(item_files(out, ds))
                if n != last:
                    last, t = n, time.time()
                if time.time() - t > STALL_S:
                    log(
                        f"stall: no new item for {STALL_S}s at {n}/{total}; stopping the arm"
                    )
                    break
        finally:
            for p in procs:  # always reap every group, even after its leader exited
                kill_group(p)
            for p in procs:
                p.wait()
        env_check_and_restart(ds)
    n = len(completed(ds, out, expected, session))
    log(f"done {ds} {name}: {n}/{total}")
    return n


def cfg_dir(base: Path, cfg: dict) -> Path:
    import hashlib

    return base / ("cfg_" + hashlib.sha256(space.key(cfg).encode()).hexdigest()[:12])


def session_args(
    ds: str,
    cfg: dict,
    *,
    experiment: str,
    baseline_dir: Path,
    label: str,
    pnl: dict,
    panel_name: str = BASE_PANEL,
    reuse: bool | None = None,
    reuse_from: list[str] | tuple = (),
) -> dict:
    ra = space.runner_args(cfg)
    # Quality reuse (identical injected memory -> reuse the earlier episode): 100-panel = ScienceWorld only
    # (user, 2026-10-16); train500 = both datasets (user, 2026-10-17); never for the noise repeat.
    return {
        **ra,
        "split": "train",
        "experiment": experiment,
        "baseline_dir": baseline_dir,
        "reuse": (ds in REUSE_TRAIN100) if reuse is None else reuse,
        "reuse_from": list(reuse_from),
        "reuse_from_dirs": [str(source_config_dir(ds, e)) for e in reuse_from],
        "panel": panel_path(ds, panel_name),
        "protocol_extra": {
            "panel_hash": pnl["panel_hash"],
            "space_version": space.VERSION,
            "space_config": cfg,
            "bo_label": label,
            "bo_experiment": bo.experiment_id(ds, panel_name),
        },
    }


def train_rows(ds: str, experiment: str, *, include_retrieval_id: bool = False) -> list[dict]:
    with psycopg.connect(OPT_DSN, row_factory=dict_row) as c:
        rows = c.execute(
            ("SELECT retrieval_id, updated_at, " if include_retrieval_id else "SELECT ")
            + "experiment_id, dataset, split, query_id, status, config, protocol, paired_gain, "
            "total_tokens, e2e_ms, quality, generation FROM opt.session_retrieval WHERE experiment_id = %s "
            "AND split = 'train' AND status = 'evaluated'",
            (experiment,),
        ).fetchall()
    return [
        {
            **r,
            **({"retrieval_id": str(r["retrieval_id"])} if include_retrieval_id else {}),
            "total_tokens": float(r["total_tokens"])
            if r["total_tokens"] is not None
            else None,
        }
        for r in rows
    ]


def points(ds: str, pnl: dict, panel_name: str = BASE_PANEL):
    ids = [x["item_id"] for x in pnl["items"]]
    exp = bo.experiment_id(ds, panel_name)
    return bo.load_points(
        train_rows(ds, exp),
        panel_ids=ids,
        dataset=ds,
        panel_hash=pnl["panel_hash"],
        experiment=exp,
    )


def save(path: Path, obj) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


SUBMETRIC_SQL = """SELECT protocol->'space_config' AS cfg, query_context, generation,
  jsonb_build_object(
    'pool_ids', receipt->'pool_ids', 'seed_ids', receipt->'seed_ids',
    'semantic_decisions', CASE WHEN jsonb_typeof(receipt->'semantic_decisions') = 'array' THEN
      (SELECT jsonb_agg(jsonb_build_object('score', d->'score')) FROM jsonb_array_elements(receipt->'semantic_decisions') d)
      END,
    'cards', CASE WHEN jsonb_typeof(receipt->'cards') = 'array' THEN
      (SELECT jsonb_agg(jsonb_build_object('cosine', c->'cosine', 'task_type', c->'task_type', 'n_steps', c->'n_steps',
                                           'object', c->'object')) FROM jsonb_array_elements(receipt->'cards') c)
      END) AS receipt
FROM opt.session_retrieval WHERE experiment_id = %s AND split = 'train' AND status = 'evaluated'"""


def analyzer_advice(
    ds: str,
    pnl: dict,
    pts: list,
    state: dict,
    forced: bool,
    r: int,
    panel_name: str = BASE_PANEL,
) -> tuple[list, list | None, dict]:
    """RAG-PE's LLM analyzer channel: per-stage sub-metrics of every profiled configuration -> Codex Luna
    -> at most 4 edits, which enter the candidate pool (labelled agent / agent-gap). Training rows only."""
    import math

    from trimem.experiments import session_submetrics as sm
    from trimem.optimizer import analyzer_session as az
    from trimem.optimizer import session_pool as pool

    with psycopg.connect(OPT_DSN, row_factory=dict_row) as c:
        rows = c.execute(SUBMETRIC_SQL, (bo.experiment_id(ds, panel_name),)).fetchall()
    by = {}
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
            sm.config_submetrics(by.get(k, []), ds),
            math.sqrt(pk[k].noise[1]),
            math.sqrt(pk[k].noise[2]),
        )
        for i, k in enumerate(order)
        if k in pk
    ]
    model, _ = pool.fit(pts)
    rect = az.max_hv_rectangle(
        evals, "quality" if r % 2 else "cost"
    )  # single-point front: alternate arms
    nomem = [
        json.load(open(f))["reward"]
        for f in nomemory_dir(ds, panel_name).glob(f"{ds}_*.json")
    ]
    q0 = sum(max(x, 0) / (100 if ds == "sciworld" else 1) for x in nomem) / len(nomem)
    ctx = {
        "dataset": ds,
        "nomemory_quality": q0,
        "panel": f"frozen training panel of {len(pnl['items'])} tasks, stratified by {pnl['stratified_by']}",
    }
    an = az.Analyzer(
        OPT_DSN,
        bo.experiment_id(ds, panel_name),
        notes_path=run_dir(ds, panel_name) / "analyzer_notes.txt",
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


def pending_user_probes(
    ds: str, state: dict, cap: int = 2, base: Path | None = None
) -> list[dict]:
    """Configurations the user asked to profile (user_probes.json), not yet planned in any round."""
    f = (base or OUT / ds) / "user_probes.json"
    if not f.exists():
        return []
    planned = {space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]}
    out = []
    for c in json.load(open(f))["configs"]:
        cfg = space.validate(c)
        if space.key(cfg) not in planned and space.key(cfg) not in {
            space.key(x) for x in out
        }:
            out.append(cfg)
    return out[:cap]


def profile(
    ds: str,
    use_analyzer: bool = False,
    rounds: int = ROUNDS,
    *,
    panel_name: str = BASE_PANEL,
    batch: int | None = None,
    reuse_from: list[str] | None = None,
) -> None:
    batch = default_batch(panel_name) if batch is None else batch
    reuse_from = (
        default_reuse_from(ds, panel_name) if reuse_from is None else list(reuse_from)
    )
    # train500: quality reuse on for both datasets; train200 (WebShop): on (user 2026-10-18)
    reuse = None if panel_name == BASE_PANEL else True
    exp = bo.experiment_id(ds, panel_name)
    pnl = panel(ds, panel_name)
    items = [x["item_id"] for x in pnl["items"]]
    base = run_dir(ds, panel_name)
    sa, sb = shards(ds, items, base / "shards")
    state_p = base / "state.json"
    new_state = {
        "dataset": ds,
        "panel_hash": pnl["panel_hash"],
        "experiment": exp,
        "rounds": [],
    }
    if panel_name != BASE_PANEL:
        new_state.update(panel_name=panel_name, batch=batch, reuse_from=reuse_from)
    state = json.load(open(state_p)) if state_p.exists() else new_state
    if state["panel_hash"] != pnl["panel_hash"] or state["experiment"] != exp:
        raise SystemExit("panel changed since this run started")
    if state.get("batch", bo.BATCH) != batch:
        raise SystemExit(
            f"this run was started with batch {state.get('batch', bo.BATCH)}, not {batch}"
        )
    nomem = nomemory_dir(ds, panel_name)
    if panel_name in EXTENDS_BASE:
        seed_nomemory(ds, pnl, nomem)
    if run_arm(ds, "NoMemory", nomem, sa, sb, items, session=None) < len(items):
        raise SystemExit("No-Memory baseline incomplete")
    for r in range(rounds + 1):
        if len(state["rounds"]) <= r:
            if r == 0:
                picks = [
                    {
                        "config": c,
                        "channel": "default" if i < len(bo.defaults()) else "sobol",
                    }
                    for i, c in enumerate(bo.round0())
                ]
                forced = False
            else:
                pts = points(ds, pnl, panel_name)
                measured = {space.key(p.cfg) for p in pts}
                planned = {
                    space.key(c["config"])
                    for rr in state["rounds"]
                    for c in rr["configs"]
                }
                if planned - measured:
                    raise SystemExit(
                        f"round {r}: {len(planned - measured)} planned configurations lack a complete panel"
                    )
                by_key = {space.key(p.cfg): p for p in pts}
                measured_rounds = [
                    [by_key[space.key(c["config"])] for c in rr["configs"]]
                    for rr in state["rounds"]
                ]
                forced = bo.stalled(measured_rounds)
                agent, gap_front, advice = ([], None, None)
                if use_analyzer:
                    agent, gap_front, advice = analyzer_advice(
                        ds, pnl, pts, state, forced, r, panel_name
                    )
                # User-requested probes (<run dir>/user_probes.json): placed at the head of the next round,
                # the remaining slots are BO picks. Recorded with channel "user-probe".
                probes = pending_user_probes(ds, state, base=base)
                picks = [
                    {"config": c, "channel": "user-probe", "mode": "user"}
                    for c in probes
                ]
                picks += bo.propose_batch(
                    pts,
                    bo.context(ds, pnl["panel_hash"]),
                    k=batch - len(probes),
                    seed=1000 * r,
                    forced=forced,
                    agent_rows=agent or None,
                    realized_front=gap_front,
                    exclude={space.key(c) for c in probes},
                )
            state["rounds"].append(
                {
                    "round": r,
                    "proposed_at": datetime.now(UTC).isoformat(),
                    "configs": picks,
                    "mode": "forced (stagnation)" if forced else "ordinary",
                    "effective_mode": sorted(
                        {p.get("mode", "ordinary") for p in picks}
                    ),
                    "analyzer": advice if r > 0 and use_analyzer else None,
                }
            )
            save(state_p, state)
            log(
                f"round {r} configurations: "
                + "; ".join(space.key(p["config"]) for p in picks)
            )
        for j, p in enumerate(state["rounds"][r]["configs"]):
            cfg = p["config"]
            label = f"r{r}c{j}:{space.DISPLAY[cfg['plan']]}"
            args = session_args(
                ds,
                cfg,
                experiment=exp,
                baseline_dir=nomem,
                label=label,
                pnl=pnl,
                panel_name=panel_name,
                reuse=reuse,
                reuse_from=reuse_from,
            )
            if run_arm(
                ds, label, cfg_dir(base / "configs", cfg), sa, sb, items, session=args
            ) < len(items):
                raise SystemExit(
                    f"{label} incomplete after {ATTEMPTS} attempts; resume this driver to retry"
                )
    rep = bo.defaults()[0]
    # The noise repeat never reuses (it must be an independent second run of the default VGR).
    args = session_args(
        ds,
        rep,
        experiment=exp + ":noise-repeat",
        baseline_dir=nomem,
        label="noise-repeat:VGR",
        pnl=pnl,
        panel_name=panel_name,
        reuse=None if panel_name == BASE_PANEL else False,
    )
    if run_arm(
        ds, "noise-repeat VGR", base / "noise_repeat_VGR", sa, sb, items, session=args
    ) < len(items):
        raise SystemExit("noise repeat incomplete")
    planned = {space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]}
    measured = {space.key(p.cfg) for p in points(ds, pnl, panel_name)}
    # User 2026-10-17: --rounds may stop BO early (ScienceWorld: rounds 0-2 = 20 configurations). Rounds
    # proposed after the stop (stopped before any of their episodes) are dropped from the plan.
    if len(state["rounds"]) > rounds + 1:
        state["rounds"] = state["rounds"][: rounds + 1]
        save(state_p, state)
    planned = {space.key(c["config"]) for rr in state["rounds"] for c in rr["configs"]}
    expected = bo.expected_configs(rounds, batch)
    if (
        len(state["rounds"]) != rounds + 1
        or len(planned) != expected
        or planned != measured
    ):
        raise SystemExit(
            f"profiling incomplete: {len(measured)} measured of {len(planned)} planned"
        )
    done = {
        "completed_at": datetime.now(UTC).isoformat(),
        "configs": len(planned),
        "panel_hash": pnl["panel_hash"],
        "experiment": exp,
        "space_version": space.VERSION,
        "bo_rounds": rounds,
    }
    if panel_name != BASE_PANEL:
        done.update(panel_name=panel_name, bo_batch=batch, reuse_from=reuse_from)
    save(base / "profile_complete.json", done)
    log(f"PROFILING DONE {ds}: {len(measured)} complete configurations")


def per_query_gain(ds: str, experiment: str, cfg: dict) -> dict[str, float]:
    out = {}
    for r in train_rows(ds, experiment):
        if r["protocol"].get("space_config") == cfg:
            if r["query_id"] in out:
                raise SystemExit(
                    f"duplicate evaluated row for {r['query_id']} in {experiment}"
                )
            out[r["query_id"]] = r["paired_gain"]
    return out


def test(
    ds: str,
    arms_wanted: tuple[str, ...] = ("qbest", "knee", "onlybest", "vgr_rank"),
    panel_name: str = BASE_PANEL,
) -> None:
    pnl = panel(ds, panel_name)
    base = run_dir(ds, panel_name)
    exp = bo.experiment_id(ds, panel_name)
    pre = arm_prefix(panel_name)
    ident = {
        "panel_hash": pnl["panel_hash"],
        "experiment": exp,
        "space_version": space.VERSION,
    }
    done_p = base / "profile_complete.json"
    if not done_p.exists() or any(
        json.load(open(done_p))[k] != v for k, v in ident.items()
    ):
        raise SystemExit(
            "training profiling is not complete for this panel/experiment; no selection, no test"
        )
    sel_p = base / "selection.json"
    if not sel_p.exists():  # freeze before any test episode
        pts = points(ds, pnl, panel_name)
        rep = bo.defaults()[0]
        a, b = (
            per_query_gain(ds, exp, rep),
            per_query_gain(ds, exp + ":noise-repeat", rep),
        )
        ids = [x["item_id"] for x in pnl["items"]]
        if set(a) != set(ids) or set(b) != set(ids):
            raise SystemExit(
                "default VGR and its repeat must both cover the full panel"
            )
        floor = bo.noise_floor([a[q] for q in ids], [b[q] for q in ids])
        sel = bo.select(pts, floor)
        # User decision 2026-10-16: also test the best configuration proposed by the BO rounds themselves
        # (rounds 1-6, excluding round-0 defaults/Sobol and user probes), so the test can tell whether the
        # BO rounds beat the initial design.
        state = json.load(open(base / "state.json"))
        bo_keys = {
            space.key(c["config"])
            for rr in state["rounds"][1:]
            for c in rr["configs"]
            if c.get("channel") != "user-probe"
        }
        bo_pts = [p for p in pts if space.key(p.cfg) in bo_keys]
        bo_best = max(bo_pts, key=lambda p: p.quality) if bo_pts else None
        save(
            sel_p,
            {
                **ident,
                "frozen_at": datetime.now(UTC).isoformat(),
                "noise_floor": floor,
                "n_configs": len(pts),
                "q_best": {
                    "config": sel["q_best"].cfg,
                    "gain": sel["q_best"].quality,
                    "tokens": sel["q_best"].tokens,
                    "latency_ms": sel["q_best"].latency_ms,
                },
                "knee": {
                    "config": sel["knee"].cfg,
                    "gain": sel["knee"].quality,
                    "tokens": sel["knee"].tokens,
                    "latency_ms": sel["knee"].latency_ms,
                },
                "near_q_best": sel["near_q_best"],
                "bo_only_best": None
                if bo_best is None
                else {
                    "config": bo_best.cfg,
                    "gain": bo_best.quality,
                    "tokens": bo_best.tokens,
                    "latency_ms": bo_best.latency_ms,
                },
            },
        )
    sel = json.load(open(sel_p))
    if any(sel.get(k) != v for k, v in ident.items()):
        raise SystemExit("selection.json belongs to a different panel/experiment/space")
    sa, sb = CTRL / "shards" / f"{ds}_a.json", CTRL / "shards" / f"{ds}_b.json"
    arms = [(f"{pre}-Qbest", sel["q_best"]["config"])] if "qbest" in arms_wanted else []
    if "knee" in arms_wanted:
        arms.append((f"{pre}-Knee", sel["knee"]["config"]))
    if sel.get("bo_only_best") and "onlybest" in arms_wanted:
        arms.append((f"{pre}-onlybest", sel["bo_only_best"]["config"]))
    if (
        "vgr_rank" in arms_wanted
        and len(item_files(test_arm_dir(ds, "VGR+Rank"), ds)) < 200
    ):
        arms.append(
            ("VGR+Rank", space.validate({"plan": "VGR+Rank", **space.plan_defaults()}))
        )
    done = set()
    for name, cfg in arms:
        k = space.key(cfg)
        out = test_arm_dir(ds, name)
        if k in done:  # Knee == Q-best: one test run, linked
            log(f"{name} has the same configuration as an earlier test arm; not rerun")
            continue
        done.add(k)
        ra = space.runner_args(cfg)
        args = {
            **ra,
            "split": "test",
            "experiment": test_experiment(ds, name),
            "baseline_dir": test_arm_dir(ds, "NoMemory"),
            "panel": None,
            "protocol_extra": {
                "space_version": space.VERSION,
                "space_config": cfg,
                "bo_label": name,
            },
        }
        test_ids = [x["item_id"] for x in json.load(open(sa)) + json.load(open(sb))]
        if run_arm(ds, f"test {name}", out, sa, sb, test_ids, session=args) < len(
            test_ids
        ):
            raise SystemExit(f"test {name} incomplete")
    log(f"TEST DONE {ds}")


def test_fixed(
    ds: str, arms_wanted: tuple[str, ...] = ("nomemory", *FIXED_ARMS)
) -> None:
    """No-Memory and the plan-default arms on the AgentEval 200 test set (controlled protocol), before any
    profiling (WebShop, user 2026-10-18; the ALFWorld / ScienceWorld arms were run by run_controlled*.sh).
    Same experiment ids, result directories and opt protocol as the default VGR+Rank arm of ``test``."""
    unknown = set(arms_wanted) - {"nomemory", *FIXED_ARMS}
    if unknown:
        raise SystemExit(f"unknown fixed test arms {sorted(unknown)}")
    sa, sb = CTRL / "shards" / f"{ds}_a.json", CTRL / "shards" / f"{ds}_b.json"
    test_ids = [x["item_id"] for x in json.load(open(sa)) + json.load(open(sb))]
    nomem = test_arm_dir(ds, "NoMemory")
    if "nomemory" in arms_wanted and run_arm(
        ds, "test NoMemory", nomem, sa, sb, test_ids, session=None
    ) < len(test_ids):
        raise SystemExit("test No-Memory incomplete")
    if len(item_files(nomem, ds)) < len(test_ids):
        raise SystemExit(
            "the paired No-Memory test baseline must be complete before memory arms"
        )
    for arm in (a for a in FIXED_ARMS if a in arms_wanted):
        name = FIXED_ARMS[arm]
        cfg = space.validate({"plan": name, **space.plan_defaults()})
        args = {
            **space.runner_args(cfg),
            "split": "test",
            "experiment": test_experiment(ds, name),
            "baseline_dir": nomem,
            "panel": None,
            "protocol_extra": {
                "space_version": space.VERSION,
                "space_config": cfg,
                "bo_label": name,
            },
        }
        if run_arm(
            ds, f"test {name}", test_arm_dir(ds, name), sa, sb, test_ids, session=args
        ) < len(test_ids):
            raise SystemExit(f"test {name} incomplete")
    log(f"FIXED TEST ARMS DONE {ds}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("profile", "test", "test-fixed"))
    ap.add_argument("--dataset", choices=tuple(DS), required=True)
    ap.add_argument(
        "--analyzer",
        action="store_true",
        help="enable the LLM analyzer channel (Codex Luna)",
    )
    ap.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="profile: last BO round (default 6 = 40 configurations)",
    )
    ap.add_argument(
        "--arms",
        default=None,
        help="test: frozen arms (default qbest,knee,onlybest,vgr_rank); test-fixed: nomemory,vgr,vgr_rank,"
        "p2,p2_rank (default all)",
    )
    ap.add_argument(
        "--panel_name",
        default=BASE_PANEL,
        choices=tuple(bo.PANEL_VERSIONS),
        help="training panel (train500 = the 100 frozen items + 400; train200 = WebShop's "
        "product-isolated panel; own experiment and result dir)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=None,
        help="profile: configurations per BO round (default 10 for train500, 5 otherwise)",
    )
    ap.add_argument(
        "--reuse_from",
        default="auto",
        help="profile: comma-separated source experiments for quality reuse; 'auto' = the train100 "
        "experiment for train500, none for train100 / train200; 'none' disables",
    )
    a = ap.parse_args()
    reuse_from = (
        None
        if a.reuse_from == "auto"
        else []
        if a.reuse_from == "none"
        else [x for x in a.reuse_from.split(",") if x]
    )
    (OUT / a.dataset).mkdir(parents=True, exist_ok=True)
    lock = open(OUT / a.dataset / "driver.lock", "w")
    try:  # one driver per dataset: it owns the state, the opt rows being written and the env servers
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            f"another driver holds {OUT / a.dataset / 'driver.lock'}"
        ) from None

    def stop(
        signum, frame
    ):  # SIGTERM -> exception, so run_arm's finally reaps the eval process groups
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if a.stage == "profile":
        profile(
            a.dataset,
            use_analyzer=a.analyzer,
            rounds=a.rounds,
            panel_name=a.panel_name,
            batch=a.batch,
            reuse_from=reuse_from,
        )
    elif a.stage == "test-fixed":
        test_fixed(
            a.dataset, tuple(a.arms.split(",")) if a.arms else ("nomemory", *FIXED_ARMS)
        )
    else:
        test(
            a.dataset,
            tuple((a.arms or "qbest,knee,onlybest,vgr_rank").split(",")),
            panel_name=a.panel_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
