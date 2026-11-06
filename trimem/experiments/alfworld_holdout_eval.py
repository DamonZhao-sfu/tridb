"""Frozen validation -> TEST evaluation of an early-stopped E2 catalog (ALFWorld / ScienceWorld / WebShop).

The evaluator uses TRAIN transport for validation's original TRAIN tasks, with
an explicit phase identity. No TEST outcomes enter selection. No new NoMemory
episodes, optimizer proposals, or retries are admitted here.

The module name is historical: the dataset comes from the source campaign manifest
(``alfworld_e2_evidence.DATASETS``); ALFWorld identities are unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import statistics
import time
from pathlib import Path

from trimem.experiments import alfworld_e2_campaign as campaign
from trimem.experiments import alfworld_e2_evidence as ev

ROOT = campaign.ROOT
PROTOCOL = "alfworld-validation-pareto-test-v1"
EDITOR = ROOT / "paper/sigmod/revision_20261024_curve_editor"
HOLDOUT = "results/validation_holdout_20261024"


def protocol_for(dataset):
    return f"{dataset}-validation-pareto-test-v1"


def max_round_of(m):
    """Frozen runner max_round (manifests always record ``fixed``; 30 = the native/COMMON budget)."""
    return m.get("fixed", {}).get("max_round", 30)


def frontier(points, cost):
    """Keep coincident configurations; exact dominance, maximize Q/minimize cost."""
    return [
        p["key"]
        for p in points
        if not any(
            q["quality"] >= p["quality"]
            and q[cost] <= p[cost]
            and (q["quality"] > p["quality"] or q[cost] < p[cost])
            for q in points
        )
    ]


def select(validation, groups, expected=200):
    if not validation or any(
        p["phase"] != "validation" or p["n"] != expected for p in validation
    ):
        raise ValueError("selection requires complete validation measurements")
    by = {p["key"]: p for p in validation}
    if len(by) != len(validation):
        raise ValueError("duplicate validation configuration")
    selected = {}
    for method, keys in groups.items():
        if len(keys) != len(set(keys)) or not set(keys) <= by.keys():
            raise ValueError("incomplete or duplicate candidate catalog")
        rows = [by[k] for k in keys]
        selected[method] = {
            axis: sorted(frontier(rows, cost))
            for axis, cost in [("tokens", "tokens"), ("latency", "latency_span_ms")]
        }
    return selected


def serving_identity(services, proc_root=Path("/proc")):
    """Use process start ticks; vLLM model-card `created` changes per request."""
    servers = {}
    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
            argv = raw.decode().split("\0")
            if "serve" not in argv or "--port" not in argv:
                continue
            if not any(Path(a).name == "vllm" for a in argv):
                continue
            port = argv[argv.index("--port") + 1]
            if port not in ("8030", "8031"):
                continue
            start_ticks = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, UnicodeError, IndexError):
            continue
        if port in servers:
            raise ValueError("ambiguous serving process")
        servers[port] = {
            "pid": int(proc.name),
            "start_ticks": start_ticks,
            "argv_sha256": hashlib.sha256(raw).hexdigest(),
            "models": [
                {k: x.get(k) for k in ("id", "root", "max_model_len")}
                for x in services["services"][port]["data"]
            ],
        }
    if set(servers) != {"8030", "8031"}:
        raise ValueError("cannot attest both local serving processes")
    return servers


def validation_panel(items, dataset="alfworld"):
    """Keep original database splits distinct from the validation phase label.

    Whitelist per dataset: ScienceWorld's own DB ``test``/``dev`` splits are development tasks (held_out false,
    agenteval_item null; checked by ``check_validation_tasks``), never the AgentEval TEST items."""
    splits = sorted({p["source_split"] for p in items})
    if not splits or not set(splits) <= set(ev.spec(dataset)["validation_splits"]):
        raise ValueError("unexpected validation source split")
    return {"items": items, "splits": splits}


def check_validation_tasks(panel, dataset="alfworld"):
    import psycopg
    from trimem.experiments.agentenv_task_adapter import dsn_for

    items = panel["items"]
    keys = [p["task_key"] for p in items]
    if len(keys) != len(set(keys)) or panel != validation_panel(items, dataset):
        raise ValueError("duplicate tasks or inconsistent source splits")
    with psycopg.connect(dsn_for(dataset)) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        rows = conn.execute(
            "SELECT key,split,held_out,agenteval_item FROM task WHERE key=ANY(%s)",
            (keys,),
        ).fetchall()
    by_key = {r[0]: r for r in rows}
    for p in items:
        row = by_key.get(p["task_key"])
        if row is None or row[1] != p["source_split"] or row[2] or row[3] is not None:
            raise ValueError(f"{p['item_id']}: ineligible validation task")


DIRECT_SELECTION = (
    "first {limit} proposals that completed the full TRAIN stage (pruned excluded); "
    "no validation; every survivor evaluated on TEST"
)


def prepare(
    out,
    source,
    limit=20,
    run_id=None,
    label="MO-TPE",
    trace=True,
    direct=False,
    authorization=None,
):
    """``run_id`` selects the discovery run (default: the manifest's first); ``trace=False`` omits TRACE.

    ``direct``: no validation phase; TEST every pruning survivor of the first ``limit`` proposals.
    """
    if direct and not authorization:
        raise ValueError("direct TEST requires the recorded user authorization")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("preparation already frozen; use run/report")
    m = ev.read_json(source / "manifest.json")
    ds = ev.dataset_of(m)
    s = ev.spec(ds)
    run = (
        m["runs"][0]
        if run_id is None
        else next(r for r in m["runs"] if r["id"] == run_id)
    )
    base = Path(run["directory"])
    state = ev.read_json(base / "state.json")
    if ev.read_json(base / "supervisor.json")["status"] == "running":
        raise ValueError("stop discovery before freezing the checkpoint")
    proposals = [x for rr in state["rounds"] for x in rr["configs"]]
    chosen = proposals[:limit]
    first_stage = state["staging"]["sizes"][0]  # 50 for E2 (50/100); 20 for 20/40/60/80/100 campaigns
    if len(chosen) != limit or any(
        x["staging"]["stage_reached"] < first_stage for x in chosen
    ):
        raise ValueError("the requested prefix has not completed its first stage")
    d = campaign.driver()
    groups = {label: [d.space.key(x["config"]) for x in chosen]}
    configs = {d.space.key(x["config"]): x["config"] for x in chosen}
    if trace:
        catalog = [
            x
            for x in ev.read_json(source / "trace_catalog.json")
            if x["eligible"] and x["config"]["plan"] in m["plans"]
        ]
        groups["TRACE"] = [x["key"] for x in catalog]
        configs.update({x["key"]: x["config"] for x in catalog})
    holdout = ROOT / HOLDOUT / ds
    val = ev.read_json(holdout / "validation.json")
    train = ev.read_json(holdout / "train.json")
    tests = ev.read_json(holdout / "test_ids.json")
    vi = [x["item_id"] for x in val]
    ti = [x["item_id"] for x in train]
    if (
        len(vi) != 200
        or len(tests) != 200
        or len(set(vi)) != 200
        or len(set(tests)) != 200
    ):
        raise ValueError("expected unique 200-task validation and TEST")
    if set(vi) & set(tests) or set(ti) & (set(vi) | set(tests)):
        raise ValueError("overlapping train/validation/TEST")
    if ds != "alfworld" and set(tests) != set(m["test_ids"]):
        raise ValueError("holdout TEST differs from the campaign's TEST shards")
    if s["semfilter_v2b"]:
        # Holdout processes never run profile(): their exported SemFilter env must equal the discovery record.
        want = {
            k: v
            for k, v in campaign.isolated_env(ds, evaluation=True).items()
            if k.startswith("TRIMEM_SEMFILTER_")
        }
        if d.effective_semfilter_env(state["semfilter"]) != want:
            raise ValueError(
                "evaluation SemFilter env differs from the discovery record"
            )
    for name in (
        "state.json",
        "supervisor.json",
        "e2_attempts.json",
        "e2_resources.json",
    ):
        ev.write_json(out / "discovery_snapshot" / name, ev.read_json(base / name))
    panel = validation_panel(val, ds)
    check_validation_tasks(panel, ds)
    ev.write_json(out / "validation_panel.json", panel)
    receipt = ev.read_json(source / "confirmation.json")
    services = campaign.snapshot_services(ds)
    campaign.require_idle(services)
    body = {
        "protocol": protocol_for(ds),
        "created_at": campaign.now(),
        "dataset": ds,
        "proposal_limit": limit,
        "actual_proposals_at_stop": len(proposals),
        "source_prepare": str(source.resolve()),
        "source_manifest": ev.file_hash(source / "manifest.json"),
        "groups": groups,
        "configs": configs,
        "validation_ids": vi,
        "test_ids": tests,
        "split_files": {
            str(holdout / n): ev.file_hash(holdout / n)
            for n in ("train.json", "validation.json", "test_ids.json", "manifest.json")
        },
        "panel_sha256": ev.file_hash(out / "validation_panel.json"),
        "identity": {k: receipt[k] for k in ("model_revision", "memory_snapshot")},
        "serving_identity": serving_identity(services),
        "runtime": "controlled-c16-v1; one configuration, two halves x eight processes",
        "fixed": {
            "max_round": s["max_round"],
            "temperature": 0,
            "memory_policy": "success+failure-v1",
            "semfilter": m["semfilter"],
            "memory": m["fixed_memory"],
        },
        "validation_episode_cap": 0 if direct else len(configs) * 200,
        "test_episode_cap": 200 * len(direct_keys(state, chosen))
        if direct
        else len(configs) * 200,
        "wall_seconds": 48 * 3600,
        "new_nomemory_episodes": 0,
        "automatic_retries": 0,
        "authorization": authorization
        or "User: after 20 configurations, perform validation calibration, TEST, and plot against TRACE.",
        "selection": DIRECT_SELECTION.format(limit=limit)
        if direct
        else "union of both validation Pareto projections; all selected configurations evaluated on TEST",
        "direct": direct,
        "scope": "OneConfig; conditional on reused BO catalogs; TRACE historical search budget is unmatched",
        "code_sha256": campaign.source_hashes(),
    }
    body["identity_hash"] = ev.digest(body)
    campaign.freeze(out / "manifest.json", body)
    return {
        "manifest": str(out / "manifest.json"),
        "groups": {k: len(v) for k, v in groups.items()},
        "unique_configs": len(configs),
        "validation_episodes": body["validation_episode_cap"],
    }


def direct_keys(state, chosen):
    """Pruning survivors among ``chosen`` proposals, in proposal order."""
    d = campaign.driver()
    return [
        d.space.key(x["config"])
        for x in chosen
        if x["staging"].get("pruned_at") is None
        and x["staging"]["stage_reached"] >= len(x["sample"]["ids"])
    ]


def verify(out):
    m = ev.read_json(out / "manifest.json")
    unsigned = {k: v for k, v in m.items() if k != "identity_hash"}
    if ev.digest(unsigned) != m["identity_hash"]:
        raise ValueError("frozen manifest or execution code changed")
    code, panel_sha = m["code_sha256"], m["panel_sha256"]
    repair_path = out / "execution_repair.json"
    if repair_path.exists():
        repair = ev.read_json(repair_path)
        if repair["base_manifest_sha256"] != ev.file_hash(out / "manifest.json"):
            raise ValueError("repair belongs to another manifest")
        repaired_code = repair["code_sha256"]
        allowed = {"trimem/experiments/alfworld_holdout_eval.py"}
        if (
            code.keys() != repaired_code.keys()
            or {p for p in code if code[p] != repaired_code[p]} - allowed
        ):
            raise ValueError("repair changed unrelated execution code")
        code, panel_sha = repaired_code, repair["panel_sha256"]
        ds = ev.dataset_of(m)
        original_items = ev.read_json(ROOT / HOLDOUT / ds / "validation.json")
        if ev.read_json(out / "validation_panel.json") != validation_panel(
            original_items, ds
        ):
            raise ValueError("repair changed validation membership")
    if campaign.source_hashes() != code:
        raise ValueError("frozen manifest or execution code changed")
    if any(ev.file_hash(p) != h for p, h in m["split_files"].items()):
        raise ValueError("frozen split changed")
    if ev.file_hash(out / "validation_panel.json") != panel_sha:
        raise ValueError("validation panel changed")
    return m


def experiment(m, phase):
    from trimem.experiments import memory_policy

    return memory_policy.tag(
        f"{ev.dataset_of(m)}:e2holdout:{m['identity_hash'][:16]}:{phase}",
        m["fixed"]["memory_policy"],
    )


def measurements(out, m, phase):
    import psycopg
    from psycopg.rows import dict_row
    from trimem.experiments import episode_budget as eb
    from trimem.experiments.alf_test_c16 import call_spans

    d = campaign.driver()
    ds = ev.dataset_of(m)
    split = "train" if phase == "validation" else "test"
    with psycopg.connect(d.drv.OPT_DSN, row_factory=dict_row) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        rows = conn.execute(
            "SELECT query_id,protocol,quality,total_tokens,e2e_ms,"
            "generation->>'retrieval_ms' AS retrieval_ms, generation->>'steps' AS steps "
            "FROM opt.session_retrieval WHERE experiment_id=%s AND split=%s AND status='evaluated'",
            (experiment(m, phase), split),
        ).fetchall()
    wanted = set(m[phase + "_ids"])
    spans, files, result = {}, {}, {}
    for r in rows:
        p = r["protocol"]
        key = d.space.key(p["space_config"])
        identity = (key, r["query_id"])
        if key not in m["configs"] or r["query_id"] not in wanted:
            raise ValueError("measurement outside frozen matrix")
        if (
            p.get("holdout_identity") != m["identity_hash"]
            or p.get("holdout_phase") != phase
        ):
            raise ValueError("measurement identity mismatch")
        if identity in result:
            raise ValueError("duplicate measurement")
        if eb.row_max_round(p) != max_round_of(m):
            raise ValueError(
                "measurement episode budget differs from the frozen max_round"
            )
        directory = out / phase / d.cfg_name(m["configs"][key])
        if key not in spans:
            # LLM-call span per item index (dataset-generic, as bo_pool_driver.nomemory_rows)
            spans[key] = call_spans(directory, ds)
            files[key] = d.drv.item_files(directory, ds)
        if r["query_id"] not in files[key] or r["query_id"] not in spans[key]:
            raise ValueError("DB result lacks episode file or measured LLM span")
        values = {
            "quality": float(r["quality"]),
            "tokens": float(r["total_tokens"]),
            "latency_ms": float(r["e2e_ms"]),
            "latency_span_ms": spans[key][r["query_id"]] + float(r["retrieval_ms"]),
        }
        if (
            not all(math.isfinite(v) for v in values.values())
            or not 0 <= values["quality"] <= 1
        ):
            raise ValueError("invalid measured metric")
        result[identity] = {"key": key, "task": r["query_id"], "phase": phase, **values}
    return result


def aggregate(rows, keys, ids, phase):
    result = []
    for key in keys:
        sub = [rows[(key, q)] for q in ids if (key, q) in rows]
        if len(sub) != len(ids):
            raise ValueError(f"incomplete {phase} configuration: {len(sub)}/{len(ids)}")
        metrics = {
            k: statistics.mean(x[k] for x in sub)
            for k in ("quality", "tokens", "latency_ms", "latency_span_ms")
        }
        result.append(
            {
                "key": key,
                "phase": phase,
                "n": len(ids),
                **metrics,
                "quality_se": statistics.stdev(x["quality"] for x in sub)
                / math.sqrt(len(sub)),
            }
        )
    return result


def execute_arm(out, m, phase, key, status):
    from trimem.experiments import episode_budget as eb

    verify(out)
    d = campaign.driver()
    ds = ev.dataset_of(m)
    max_round = max_round_of(m)
    ids = m[phase + "_ids"]
    rows = measurements(out, m, phase)
    missing = [q for q in ids if (key, q) not in rows]
    if not missing:
        return
    ledger_path = out / "attempts.json"
    ledger = ev.read_json(ledger_path) if ledger_path.exists() else {}
    pairs = [ev.digest([phase, key, q]) for q in missing]
    if any(p in ledger for p in pairs):
        raise ValueError("unfinished reserved attempts; no automatic retry")
    ledger.update({p: {"phase": phase, "reserved_at": campaign.now()} for p in pairs})
    ev.write_json(ledger_path, ledger)
    cfg = m["configs"][key]
    dest = out / phase / d.cfg_name(cfg)
    (out / "empty_baseline").mkdir(exist_ok=True)
    session = {
        **d.space.runner_args(cfg),
        "runner_extra": d.space.runner_cli(cfg),
        "split": "train" if phase == "validation" else "test",
        "experiment": experiment(m, phase),
        "baseline_dir": out / "empty_baseline",
        "panel": out / "validation_panel.json" if phase == "validation" else None,
        "reuse": False,
        "reuse_from": [],
        "reuse_from_dirs": [],
        "protocol_extra": {
            "space_config": cfg,
            "space_version": d.space.VERSION,
            "holdout_identity": m["identity_hash"],
            "holdout_phase": phase,
            "panel_hash": ev.digest(ids),
            "runtime_profile": m["runtime"],
            # max-round-v1 row labels (WebShop 10); {} at the native 30 turns (rows unchanged)
            **eb.protocol_extra(None if max_round == 30 else max_round),
        },
    }
    arm = d.Arm(
        phase + " " + d.cfg_name(cfg), dest, missing, session, max_round=max_round
    )
    d.check_games(ds, missing)
    owned = {"procs": []}
    start = last_change = time.monotonic()
    previous = -1
    try:
        d._launch_registered(ds, arm, (8, 8), owned)
        while any(p.poll() is None for p, _ in owned["procs"]):
            n = len(d.drv.item_files(dest, ds))
            if n != previous:
                previous, last_change = n, time.monotonic()
            status.update(
                heartbeat=campaign.now(),
                phase=phase,
                config=d.cfg_name(cfg),
                current_completed=n,
                current_expected=len(ids),
            )
            ev.write_json(out / "status.json", status)
            if time.monotonic() - last_change > 900 or time.monotonic() - start > 7200:
                raise TimeoutError(
                    "episode progress stalled or configuration exceeded two hours"
                )
            time.sleep(5)
        if any(p.returncode for p, _ in owned["procs"]):
            raise RuntimeError("evaluator exited unsuccessfully; no automatic retry")
    finally:
        for p, log in owned["procs"]:
            d.drv.kill_group(p)
            p.wait()
            log.close()
    rows = measurements(out, m, phase)
    aggregate(rows, [key], ids, phase)
    d.drv.env_check_and_restart(ds)  # E2 replaces this with health-only.


def remaining_seconds(out, m, now=None):
    """A restart cannot grant another wall budget to the same frozen evaluation."""
    now = time.time() if now is None else now
    path = out / "wall_budget.json"
    if not path.exists():
        campaign.freeze(
            path,
            {
                "identity_hash": m["identity_hash"],
                "started_epoch": now,
                "deadline_epoch": now + m["wall_seconds"],
            },
        )
    budget = ev.read_json(path)
    if budget["identity_hash"] != m["identity_hash"]:
        raise ValueError("wall budget belongs to another evaluation")
    remaining = math.ceil(budget["deadline_epoch"] - now)
    if remaining <= 0:
        raise TimeoutError("holdout wall limit reached")
    return remaining


@contextlib.contextmanager
def runtime(out, m, status):
    """Frozen evaluation runtime shared by run/direct-test; writes ``status`` on exit."""
    source = Path(m["source_prepare"])
    original = ev.read_json(source / "manifest.json")
    if ev.file_hash(source / "manifest.json") != m["source_manifest"]:
        raise ValueError("source manifest changed")
    ds = ev.dataset_of(m)
    with campaign.queue_lock():
        check_validation_tasks(ev.read_json(out / "validation_panel.json"), ds)
        services = campaign.snapshot_services(ds)
        campaign.require_idle(services)
        current_serving = serving_identity(services)
        if current_serving != m["serving_identity"]:
            raise ValueError("serving instances changed since preparation")
        from functools import partial

        from scripts.exp2_memory_snapshot import snapshot as store_snapshot

        snapshot = partial(store_snapshot, dataset=ds)
        if snapshot()["id"] != m["identity"]["memory_snapshot"]:
            raise ValueError("memory snapshot changed")
        env = campaign.isolated_env_from_manifest(original)
        os.environ.clear()
        os.environ.update(env)
        d = campaign.install_runtime(original)
        d.drv.ATTEMPTS = 1

        def timeout(signum, frame):
            raise TimeoutError("holdout wall limit reached")

        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(remaining_seconds(out, m))
        status.update(
            status="running",
            started_at=campaign.now(),
            pid=os.getpid(),
            identity_hash=m["identity_hash"],
        )
        try:
            yield snapshot
        except BaseException as exc:
            status.update(status="stopped", error=repr(exc))
            raise
        finally:
            signal.alarm(0)
            status.update(finished_at=campaign.now(), heartbeat=campaign.now())
            ev.write_json(out / "status.json", status)


def run(out):
    m = verify(out)
    if m.get("direct"):
        raise ValueError("direct-TEST preparation; use direct-test")
    status = {"validation_configs": len(m["configs"])}
    with runtime(out, m, status) as snapshot:
        keys = list(dict.fromkeys(k for group in m["groups"].values() for k in group))
        for i, key in enumerate(keys):
            status.update(completed_configurations=i)
            execute_arm(out, m, "validation", key, status)
        validation = aggregate(
            measurements(out, m, "validation"),
            keys,
            m["validation_ids"],
            "validation",
        )
        campaign.freeze(out / "validation_metrics.json", validation)
        selected = select(validation, m["groups"])
        plan = {
            "identity_hash": m["identity_hash"],
            "validation_sha256": ev.digest(validation),
            "selected": selected,
            "test_ids": m["test_ids"],
        }
        campaign.freeze(out / "test_plan.json", plan)
        verify(out)
        if snapshot()["id"] != m["identity"]["memory_snapshot"]:
            raise ValueError("memory snapshot changed before TEST")
        test_keys = sorted(
            {k for v in selected.values() for ks in v.values() for k in ks}
        )
        status.update(test_configs=len(test_keys))
        for i, key in enumerate(test_keys):
            status.update(completed_configurations=i)
            execute_arm(out, m, "test", key, status)
        report(out)
        status.update(status="complete", phase="test_and_html_complete")


def direct_test(out):
    """TEST every pruning survivor of the frozen proposal prefix; no validation, no selection by outcomes."""
    m = verify(out)
    if not m.get("direct"):
        raise ValueError("validation-selected preparation; use run")
    state = ev.read_json(out / "discovery_snapshot/state.json")
    proposals = [x for rr in state["rounds"] for x in rr["configs"]]
    keys = direct_keys(state, proposals[: m["proposal_limit"]])
    if 200 * len(keys) != m["test_episode_cap"]:
        raise ValueError("survivor set differs from the frozen preparation")
    campaign.freeze(
        out / "test_plan.json",
        {
            "identity_hash": m["identity_hash"],
            "selection": m["selection"],
            "selected": {label: keys for label in m["groups"] if label != "TRACE"},
            "test_ids": m["test_ids"],
        },
    )
    status = {"mode": "direct_test", "test_configs": len(keys)}
    with runtime(out, m, status):
        for i, key in enumerate(keys):
            status.update(completed_configurations=i)
            execute_arm(out, m, "test", key, status)
        metrics = aggregate(measurements(out, m, "test"), keys, m["test_ids"], "test")
        ev.write_json(out / "test_metrics.json", metrics)
        status.update(
            status="complete",
            phase="direct_test_complete",
            completed_configurations=len(keys),
        )


def report(out):
    m = verify(out)
    plan = ev.read_json(out / "test_plan.json")
    validation = ev.read_json(out / "validation_metrics.json")
    if plan["identity_hash"] != m["identity_hash"] or plan[
        "validation_sha256"
    ] != ev.digest(validation):
        raise ValueError("frozen TEST plan provenance changed")
    if plan["selected"] != select(validation, m["groups"]):
        raise ValueError("TEST policy selection differs from validation")
    keys = sorted(
        {k for v in plan["selected"].values() for ks in v.values() for k in ks}
    )
    metrics = aggregate(measurements(out, m, "test"), keys, m["test_ids"], "test")
    ev.write_json(out / "test_metrics.json", metrics)
    by = {x["key"]: x for x in metrics}
    points = []
    for method, projections in plan["selected"].items():
        for key in sorted(set(projections["tokens"] + projections["latency"])):
            row = by[key]
            points.append(
                {
                    "id": f"E2-{method}-{m['identity_hash'][:8]}-{ev.digest(key)[:10]}",
                    "short": f"{method} {campaign.driver().cfg_name(m['configs'][key])}",
                    "kind": "e2",
                    "method": method,
                    "q": row["quality"],
                    "q_se": row["quality_se"],
                    "tok": row["tokens"],
                    "lat": row["latency_span_ms"] / 1000,
                    "lat_wall": row["latency_ms"] / 1000,
                    "n": row["n"],
                    "config": m["configs"][key],
                    "split": "test",
                    "steps": None,
                    "source": f"validation-selected E2 {method}; {out / 'test_metrics.json'}",
                }
            )
    document = {
        "schema": m["protocol"],
        "dataset": ev.dataset_of(m),
        "points": points,
        "test_plan": str(out / "test_plan.json"),
        "identity": m["identity_hash"],
    }
    ev.write_json(out / "paper_curve_points.json", document)
    plot(out, points, ev.spec(ev.dataset_of(m))["name"])
    # Publishing uses the same lock as every editor refresh; stable IDs preserve selections.
    from trimem.experiments import paper_frontier_editor as editor

    editor.publish_e2(document, (out / "test_pareto.svg").read_text())
    return {"points": len(points), "html": str(editor.PAGE)}


def plot(out, points, name="ALFWorld"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, cost, label in zip(
        axes, ("tok", "lat"), ("Tokens / episode", "Latency / episode (s)"), strict=True
    ):
        for method, color in [("MO-TPE", "#0072B2"), ("TRACE", "#D55E00")]:
            rows = [p for p in points if p["method"] == method]
            ax.scatter(
                [p[cost] for p in rows],
                [p["q"] for p in rows],
                label=method,
                color=color,
            )
            front = [
                p
                for p in rows
                if not any(
                    q["q"] >= p["q"]
                    and q[cost] <= p[cost]
                    and (q["q"] > p["q"] or q[cost] < p[cost])
                    for q in rows
                )
            ]
            front.sort(key=lambda p: p[cost])
            ax.plot(
                [p[cost] for p in front],
                [p["q"] for p in front],
                color=color,
                linewidth=1,
            )
        ax.set(xlabel=label, ylabel="TEST success rate")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(f"{name}: validation-selected configurations; observed TEST fronts")
    fig.tight_layout()
    for extension in ("svg", "pdf", "png"):
        fig.savefig(out / f"test_pareto.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "direct-test", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/exp2_alfworld_prepare_20261024_async",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--run-id")
    parser.add_argument("--label", default="MO-TPE")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--authorization")
    a = parser.parse_args()
    if a.mode == "prepare":
        result = prepare(
            a.out.resolve(),
            a.source.resolve(),
            a.limit,
            a.run_id,
            a.label,
            not a.no_trace,
            a.direct,
            a.authorization,
        )
    elif a.mode == "run":
        result = run(a.out.resolve())
    elif a.mode == "direct-test":
        result = direct_test(a.out.resolve())
    else:
        result = report(a.out.resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
