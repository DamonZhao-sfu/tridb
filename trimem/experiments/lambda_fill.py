"""Targeted (configuration, query) test episodes for the Opt-PerQuery TEST lambda curve (user 2026-10-19).

``router_lambda_test_coverage`` replays the deployed relaxed-BRANE router at every lambda of its grid and
lists the (configuration, query) pairs that no existing test episode covers. This module runs exactly
those pairs -- nothing else -- under the OLD test protocol of the existing results:

* ALFWorld / ScienceWorld: ``bo_agentenv_driver.run_arm`` + ``agentgym_session_eval`` (controlled-v1,
  AgentEval 200 split 'test', paired No-Memory test baseline, LLM 8030 / 8031 halves, env A / B, 8
  processes per half, stall watchdog, env check + restart), one subset arm per configuration;
* τ²-bench: ``bo_newenv_driver.run_arm`` + ``tau2.runner`` (``--domains`` / ``--task_ids`` subset of the
  official test split, GPU 1, concurrency 8, RSS guard, 1 trial, seed 300) -- the setup of results/tau2_test;
* memory policy success-only-v0 (enforced), semantic rubric v1, llm-budget v2.

Isolation from the existing arms: one new experiment id per dataset
(``<ds>:agenteval200:controlled-v1:lambda-fill``, ``tau2:test100:controlled-v1:lambda-fill``) whose rows
carry the exact ``protocol.space_config`` (+ ``bo_label`` / ``lambda_fill`` manifest); result files under
``results/eval_controlled_20261015/<ds>_lambda-fill/cfg_<hash>/`` and ``results/tau2_test/lambda-fill/
cfg_<hash>/`` (never an existing arm directory). The pair list is frozen once in a manifest
(``results/slo_selection_20261019/lambda_fill/<ds>.json``) and a rerun refuses a different list.

Resumable: the drivers' completion rule (valid result file AND exactly one evaluated opt row of this
experiment and configuration; files without their row are moved to ``_orphans/`` and rerun).

Protocol guards before the fills (stop the dataset on failure; nothing is mixed):
* ALFWorld / ScienceWorld canary: two existing (VGR configuration, query) pairs of the deployed router arm
  are re-run in ``...:lambda-fill-canary``; the injected memory text must be byte-identical to the old
  episode's (same code path, same store contents). τ²'s retrieval text is the simulated user's first
  message (LLM output), so its memory cannot be compared byte for byte;
* every dataset: the protocol fields of the new rows (llm budget, isolation, selection, action format /
  model, isolation rule, space version, memory policy) must equal those of the deployed router arm.

    TRIMEM_MEMORY_POLICY=success-only-v0 PYTHONPATH=. .venv-optimizer/bin/python -m trimem.experiments.lambda_fill \\
        {plan|run|status} --dataset {alfworld,sciworld,tau2}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from trimem.optimizer import session_space_agentenv as space

ROOT = Path(__file__).resolve().parents[2]
COVERAGE = ROOT / "results/slo_selection_20261019/router_lambda_test_coverage.json"
MANIFESTS = ROOT / "results/slo_selection_20261019/lambda_fill"
CTRL = ROOT / "results/eval_controlled_20261015"
T2 = ROOT / "results/tau2_test"
FILL, CANARY = "lambda-fill", "lambda-fill-canary"
AGENTENV = ("alfworld", "sciworld")
DATASETS = (*AGENTENV, "tau2")
N_CANARY = 2
# protocol fields that define the test protocol (compared with the deployed router arm's rows)
STABLE = (
    "llm_budget",
    "isolation",
    "selection",
    "action_format",
    "model",
    "isolation_rule",
    "strict_entities",
    "space_version",
)
POLICY = "success-only-v0"


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


# ------------------------------------------------------------------ pairs / manifest


def cfg_key(cfg: dict) -> str:
    return space.key(cfg)


def cfg_name(cfg: dict) -> str:
    """Same naming as the drivers' configuration directories (sha256 of the validated key)."""
    return "cfg_" + hashlib.sha256(cfg_key(cfg).encode()).hexdigest()[:12]


def load_pairs(ds: str, coverage: Path = COVERAGE) -> list[dict]:
    """Missing (configuration, query) pairs of the live-replay sweep, configurations validated."""
    d = json.loads(Path(coverage).read_text())["datasets"][ds]
    pairs = [
        {"query_id": p["query_id"], "config": space.validate(p["config"])}
        for p in d["live"]["union_missing"]["pair_list"]
    ]
    keys = [(cfg_key(p["config"]), p["query_id"]) for p in pairs]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{ds}: duplicate (configuration, query) pairs in {coverage}")
    return pairs


def groups(pairs: list[dict]) -> list[dict]:
    """One subset arm per configuration: [{"config", "name", "queries"}], deterministic order."""
    by: dict[str, dict] = {}
    for p in pairs:
        g = by.setdefault(
            cfg_key(p["config"]),
            {"config": p["config"], "name": cfg_name(p["config"]), "queries": []},
        )
        g["queries"].append(p["query_id"])
    out = [by[k] for k in sorted(by)]
    for g in out:
        g["queries"] = sorted(set(g["queries"]))
    return out


def manifest_path(ds: str) -> Path:
    return MANIFESTS / f"{ds}.json"


def freeze_manifest(ds: str, pairs: list[dict], path: Path | None = None) -> dict:
    """Write the pair list once; a later call must see exactly the same pairs (no silent change)."""
    path = path or manifest_path(ds)
    body = {
        "dataset": ds,
        "experiment": experiment(ds),
        "canary_experiment": experiment(ds, canary=True),
        "memory_policy": POLICY,
        "groups": groups(pairs),
        "n_pairs": len(pairs),
    }
    body["pairs_sha"] = hashlib.sha256(
        json.dumps(body["groups"], sort_keys=True).encode()
    ).hexdigest()[:16]
    if path.exists():
        old = json.loads(path.read_text())
        if old["pairs_sha"] != body["pairs_sha"]:
            raise SystemExit(
                f"{path}: frozen pair list differs from the current coverage file; refusing to mix"
            )
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    body["frozen_at"] = datetime.now(UTC).isoformat()
    body["source"] = (
        str(COVERAGE.relative_to(ROOT))
        if COVERAGE.is_relative_to(ROOT)
        else str(COVERAGE)
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=1))
    tmp.replace(path)
    return body


# ------------------------------------------------------------------ ids / directories


def experiment(ds: str, canary: bool = False) -> str:
    arm = CANARY if canary else FILL
    return (
        f"tau2:test100:controlled-v1:{arm}"
        if ds == "tau2"
        else f"{ds}:agenteval200:controlled-v1:{arm}"
    )


def out_dir(ds: str, cfg: dict, canary: bool = False) -> Path:
    arm = CANARY if canary else FILL
    return (T2 / arm if ds == "tau2" else CTRL / f"{ds}_{arm}") / cfg_name(cfg)


def session(ds: str, cfg: dict, baseline_dir: Path, *, canary: bool = False) -> dict:
    """Runner session of one subset arm (same shape as the drivers' test sessions)."""
    cfg = space.validate(cfg)
    return {
        **space.runner_args(cfg),
        "split": "test",
        "experiment": experiment(ds, canary),
        "baseline_dir": baseline_dir,
        "panel": None,
        "reuse": False,
        "protocol_extra": {
            "space_version": space.VERSION,
            "space_config": cfg,
            "bo_label": CANARY if canary else FILL,
            "lambda_fill": str(manifest_path(ds).relative_to(ROOT)),
        },
    }


def split_halves(queries: list[str]) -> tuple[list[str], list[str]]:
    """AgentEval items alternate over the two halves (LLM 8030 / env A, LLM 8031 / env B)."""
    return queries[0::2], queries[1::2]


# ------------------------------------------------------------------ guards


def memory_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(
            json.loads(Path(path).read_text())["memory"].encode()
        ).hexdigest()
    except (OSError, ValueError, KeyError):
        return None


def canary_pairs(
    ds: str, rows: list[tuple[str, dict]], n: int = N_CANARY
) -> list[dict]:
    """``rows`` = (query_id, space_config) of the deployed router arm; pick ``n`` VGR (no LLM in retrieval)
    pairs deterministically: the most frequent VGR configuration, its first queries by id."""
    vgr = [(q, space.validate(c)) for q, c in rows if c["plan"] == "VGR"]
    if not vgr:
        raise SystemExit(
            f"{ds}: no VGR configuration in the deployed router arm for the canary"
        )
    count: dict[str, int] = {}
    for _, c in vgr:
        count[cfg_key(c)] = count.get(cfg_key(c), 0) + 1
    best = max(sorted(count), key=lambda k: count[k])
    qs = sorted(q for q, c in vgr if cfg_key(c) == best)[:n]
    return [{"query_id": q, "config": json.loads(best)} for q in qs]


def compare_memory(new_dir: Path, old_dir: Path, queries: list[str]) -> dict:
    """Byte identity of the injected memory text (memory/<idx>.json of the old and new episodes)."""
    res = {}
    for q in queries:
        idx = q.rsplit("_", 1)[1]
        a, b = (
            memory_sha(old_dir / "memory" / f"{idx}.json"),
            memory_sha(new_dir / "memory" / f"{idx}.json"),
        )
        res[q] = {"old": a, "new": b, "identical": a is not None and a == b}
    return res


def compare_protocol(new: list[dict], ref: list[dict]) -> list[str]:
    """Differences of the STABLE protocol fields (+ memory policy; absent = success-only-v0)."""

    def view(p: dict) -> dict:
        v = {k: p.get(k) for k in STABLE if k in p}
        v["memory_policy"] = p.get("memory_policy", POLICY)
        return v

    refs = {json.dumps(view(p), sort_keys=True) for p in ref}
    if len(refs) != 1:
        return [f"reference rows disagree among themselves: {sorted(refs)[:2]}"]
    want = json.loads(refs.pop())
    diffs = []
    for p in new:
        got = view(p)
        for k in sorted(set(want) | set(got)):
            if want.get(k) != got.get(k):
                diffs.append(f"{k}: reference {want.get(k)!r} != fill {got.get(k)!r}")
    return sorted(set(diffs))


# ------------------------------------------------------------------ DB (opt rows)


def _rows(sql: str, args: tuple) -> list:
    import psycopg

    from trimem.experiments.bo_agentenv_driver import OPT_DSN

    with psycopg.connect(OPT_DSN) as c:
        return c.execute(sql, args).fetchall()


def deployed_arm(ds: str) -> str:
    from trimem.experiments import render_atlas_results as ra

    arm = ra.relaxed_router(ds)
    if arm is None:
        raise SystemExit(f"{ds}: no deployed relaxed router")
    return arm


def arm_rows(exp: str) -> list[tuple[str, dict, dict]]:
    """(query_id, space_config, protocol) of the evaluated test rows of an experiment."""
    return [
        (q, p.get("space_config"), p)
        for q, p in _rows(
            "SELECT query_id, protocol FROM opt.session_retrieval WHERE experiment_id = %s AND split = 'test' "
            "AND status = 'evaluated'",
            (exp,),
        )
    ]


def reference_experiment(ds: str) -> str:
    arm = deployed_arm(ds)
    return (
        f"tau2:test100:controlled-v1:{arm}"
        if ds == "tau2"
        else f"{ds}:agenteval200:controlled-v1:{arm}"
    )


# ------------------------------------------------------------------ runs


def _policy_guard() -> None:
    from trimem.experiments import memory_policy as mp

    if mp.current() != POLICY:
        raise SystemExit(
            f"the old test protocol is {POLICY}; set TRIMEM_MEMORY_POLICY={POLICY}"
        )


def run_agentenv_group(
    ds: str, cfg: dict, queries: list[str], *, canary: bool = False
) -> int:
    from trimem.experiments import bo_agentenv_driver as drv

    out = out_dir(ds, cfg, canary)
    out.mkdir(parents=True, exist_ok=True)
    a, b = split_halves(queries)
    sa, sb = out / "shard_a.json", out / "shard_b.json"
    sa.write_text(json.dumps([{"item_id": i} for i in a]))
    sb.write_text(json.dumps([{"item_id": i} for i in b]))
    nomem = drv.test_arm_dir(ds, "NoMemory")
    files = drv.item_files(nomem, ds)
    if any(q not in files for q in queries):
        raise SystemExit(
            f"{ds}: paired No-Memory test baseline lacks some fill queries"
        )
    args = session(ds, cfg, nomem, canary=canary)
    tag = f"{CANARY if canary else FILL} {cfg_name(cfg)}"
    log(f"{ds} {tag}: {len(queries)} queries -> {out} ({args['experiment']})")
    return drv.run_arm(ds, tag, out, sa, sb, queries, session=args)


def run_tau2_group(cfg: dict, queries: list[str]) -> int:
    from trimem.experiments import bo_newenv_datasets as dsets
    from trimem.experiments import bo_newenv_driver as nd

    d = dsets.get("tau2")
    missing = set(queries) - set(d.result_files(d.test_nomemory))
    if missing:
        raise SystemExit(
            f"τ² paired No-Memory test baseline lacks {sorted(missing)[:3]}"
        )
    out = out_dir("tau2", cfg)
    s = session("tau2", cfg, d.test_nomemory)
    log(
        f"tau2 {FILL} {cfg_name(cfg)}: {len(queries)} queries -> {out} ({s['experiment']})"
    )
    return nd.run_arm(
        d, f"{FILL} {cfg_name(cfg)}", out, queries, d.test_gpus, stage="test", session=s
    )


def canary(ds: str) -> dict:
    """ALFWorld / ScienceWorld: re-run two existing VGR pairs; memory text must be byte-identical."""
    from trimem.experiments import bo_agentenv_driver as drv

    arm = deployed_arm(ds)
    pairs = canary_pairs(ds, [(q, c) for q, c, _ in arm_rows(reference_experiment(ds))])
    cfg = pairs[0]["config"]
    qs = [p["query_id"] for p in pairs]
    n = run_agentenv_group(ds, cfg, qs, canary=True)
    res = {
        "pairs": pairs,
        "completed": n,
        "memory": compare_memory(out_dir(ds, cfg, True), drv.test_arm_dir(ds, arm), qs),
    }
    res["ok"] = n == len(qs) and all(v["identical"] for v in res["memory"].values())
    return res


def protocol_guard(ds: str) -> list[str]:
    new = [p for _, _, p in arm_rows(experiment(ds))] + (
        [p for _, _, p in arm_rows(experiment(ds, canary=True))]
        if ds in AGENTENV
        else []
    )
    return (
        compare_protocol(new, [p for _, _, p in arm_rows(reference_experiment(ds))])
        if new
        else []
    )


def status(ds: str) -> dict:
    m = json.loads(manifest_path(ds).read_text())
    done = {}
    for q, c, _ in arm_rows(experiment(ds)):
        done.setdefault(cfg_key(c), set()).add(q)
    gs = [
        {
            "name": g["name"],
            "n": len(g["queries"]),
            "done": len(set(g["queries"]) & done.get(cfg_key(g["config"]), set())),
        }
        for g in m["groups"]
    ]
    return {
        "dataset": ds,
        "pairs": m["n_pairs"],
        "done": sum(g["done"] for g in gs),
        "groups": gs,
    }


def run(ds: str) -> int:
    _policy_guard()
    m = freeze_manifest(ds, load_pairs(ds))
    report = {
        "dataset": ds,
        "started_at": datetime.now(UTC).isoformat(),
        "manifest": m["pairs_sha"],
    }
    rep_path = MANIFESTS / f"{ds}_run.json"
    if ds in AGENTENV:
        c = canary(ds)
        report["canary"] = c
        diffs = protocol_guard(ds)
        report["protocol_diffs"] = diffs
        rep_path.write_text(json.dumps(report, indent=1))
        if not c["ok"] or diffs:
            log(
                f"{ds}: CANARY FAILED (memory identical: {[v['identical'] for v in c['memory'].values()]}, "
                f"protocol diffs: {diffs}); no fills run"
            )
            return 2
        log(
            f"{ds}: canary ok (memory byte-identical on {len(c['memory'])} queries, protocol identical)"
        )
    incomplete = []
    for i, g in enumerate(m["groups"]):
        n = (
            run_agentenv_group(ds, g["config"], g["queries"])
            if ds in AGENTENV
            else run_tau2_group(g["config"], g["queries"])
        )
        if n < len(g["queries"]):
            incomplete.append(g["name"])
        if i == 0 or ds == "tau2":
            diffs = protocol_guard(ds)
            if diffs:
                report["protocol_diffs"] = diffs
                rep_path.write_text(json.dumps(report, indent=1))
                log(
                    f"{ds}: PROTOCOL DIFFERS from the deployed router arm: {diffs}; stopping"
                )
                return 2
    report.update(
        finished_at=datetime.now(UTC).isoformat(),
        incomplete=incomplete,
        status=status(ds),
        protocol_diffs=protocol_guard(ds),
    )
    rep_path.write_text(json.dumps(report, indent=1))
    log(
        f"{ds}: LAMBDA FILL {'DONE' if not incomplete else 'INCOMPLETE ' + str(incomplete)}"
    )
    return 0 if not incomplete else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("plan", "run", "status"))
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    a = ap.parse_args(argv)
    if a.stage == "plan":
        m = freeze_manifest(a.dataset, load_pairs(a.dataset))
        print(
            json.dumps(
                {
                    "dataset": a.dataset,
                    "pairs": m["n_pairs"],
                    "groups": [(g["name"], len(g["queries"])) for g in m["groups"]],
                }
            )
        )
        return 0
    if a.stage == "status":
        print(json.dumps(status(a.dataset)))
        return 0
    import fcntl
    import signal

    if a.dataset in AGENTENV:
        from trimem.experiments import bo_agentenv_driver as drv

        (drv.OUT / a.dataset).mkdir(parents=True, exist_ok=True)
        lock_path = drv.OUT / a.dataset / "driver.lock"
    else:
        from trimem.experiments import bo_newenv_datasets as dsets

        d = dsets.get("tau2")
        d.out.mkdir(parents=True, exist_ok=True)
        lock_path = d.out / "driver.lock"
    lock = open(lock_path, "w")
    try:  # the dataset's driver lock: no BO driver / SLO arm writes the same opt rows or env servers
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another driver holds {lock_path}") from None

    def stop(signum, frame):
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return run(a.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
