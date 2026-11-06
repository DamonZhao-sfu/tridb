"""T6: v0.7.0 staged pruning: own prefixes, executor-plan guard, optional NoMemory comparator (default off)."""

import json

import pytest

import tools.prepare_trace_v070_campaign as prep
from tests import v070_fixtures as fx
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_v070_baseline import baselines_block, run_baseline
from trimem.experiments.trace_v070_ledger import SampledLedger, own_tasks
from trimem.experiments.trace_v070_pruning import staged_evaluate_v070
from trimem.experiments.trace_v070_runtime import v070_patches
from trimem.optimizer.plan_schema_v070 import executor_plan

SIZE = 40


def arm_manifest(space, *, comparator=False, protocol="t6", factor=1000.0):
    """factor 1000 isolates the Pareto rule; the latency-cap tests pass a real factor."""
    m = fx.shared_manifest(space, size=SIZE, protocol=protocol)
    m["v070"] = prep.v070_block(m["v070"]["sampling"])
    m.update(search_backend="latent-mobo-v1", initialization=None, budget=80, new_configs=70, initial_configs=10)
    m["holdout"] = {**m["holdout"], "selection": "train_pareto"}
    m["pruning"] = {**prep.pruning_block(m), "sizes": [20, SIZE]}
    m["pruning"]["latency_cap"] = {**m["pruning"]["latency_cap"], "factor": factor}
    m["baselines"] = baselines_block(m)
    m["baselines"]["NoMemory"]["prune_comparator"] = comparator
    return m


def pick(space, **want):
    return next(c for c in fx.enumerate_space(space) if all(c[k] == v for k, v in want.items()))


def values_evaluator(space, table, nomemory=None):
    def ev(config, task, manifest):
        j = int(task.rsplit("_", 1)[1]) % 3 * 0.01
        q, tok, lat = nomemory if config.get("baseline") == "NoMemory" else table[space.key(config)]
        return {"task_id": task, "split": "TRAIN", "protocol": manifest["protocol"], "synthetic": True,
                "agent_steps": 3, "raw_score": q + j, "total_tokens": tok + j, "latency_seconds": lat + j,
                "config_key": manifest.get("key") or space.key(config)}

    return ev


def setup(tmp_path, space, m, full, batch, table, nomemory, *, protect=()):
    run = tmp_path / m["protocol"]
    run.mkdir()
    m = {**m, "initial_keys": [space.key(full), *protect], "live": {**m["live"], "run_dir": str(run)}}
    ev = values_evaluator(space, table, nomemory)
    with v070_patches(m):
        run_baseline(run, m, "TRAIN", ev)
        with SampledLedger(run / "ledger.sqlite", m) as led:
            led.reserve(space, [full])
            pending = led.pending()
        driver.evaluate_pending(run, m, space, pending, ev)
        with SampledLedger(run / "ledger.sqlite", m) as led:
            led.reserve(space, batch)
            pending = led.pending()
        staged_evaluate_v070(run, m, space, pending, ev)
    log = [json.loads(x) for x in (run / "pruning_log.jsonl").read_text().splitlines()]
    with SampledLedger(run / "ledger.sqlite", m) as led:
        done = {k: len(led.receipts(k)) for (k,) in led.db.execute("SELECT key FROM configs")}
    return run, m, {d["key"]: d for d in log}, done


def test_plan_guard_keyed_by_executor_plan(tmp_path):
    space = fx.tiny_space_v070()
    full = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=2)
    a = pick(space, filter_order="none", rank_impl="none", graph_impl="BFS", depth=3)  # plan VGR
    b = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=3)  # plan VGR+Rank
    assert executor_plan(a) == "VGR" and executor_plan(b) == executor_plan(full) == "VGR+Rank"
    k = space.key
    table = {k(full): (100, 100, 1), k(a): (1, 10000, 100), k(b): (1, 10000, 100)}
    run, m, log, done = setup(tmp_path, space, arm_manifest(space), full, [a, b], table, (0.5, 50, 0.5))
    assert log[k(b)]["prune"] and log[k(b)]["prune_rule"] == "pareto" and not log[k(b)]["guarded"]
    assert log[k(b)]["executor_plan"] == "VGR+Rank"
    assert not log[k(a)]["prune"] and log[k(a)]["guarded"] and log[k(a)]["executor_plan"] == "VGR"
    assert done == {k(full): SIZE, k(a): SIZE, k(b): 20}
    # every configuration ran its OWN prefix
    with SampledLedger(run / "ledger.sqlite", m) as led:
        got = {str(r["task_id"]) for r in led.receipts(k(b))}
    assert got == set(own_tasks(m, k(b))[:20])


@pytest.mark.parametrize("comparator", [False, True])
def test_nomemory_comparator(tmp_path, comparator):
    space = fx.tiny_space_v070()
    full = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=2)
    c = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=3)
    k = space.key
    table = {k(full): (100, 20000, 200), k(c): (1, 10000, 100)}  # full does not dominate c
    m = arm_manifest(space, comparator=comparator, protocol=f"t6-cmp-{comparator}")
    _, _, log, done = setup(tmp_path, space, m, full, [c], table, (50, 50, 0.5))
    if comparator:
        assert log[k(c)]["prune"] and log[k(c)]["dominated_by"] == "NoMemory" and done[k(c)] == 20
    else:
        assert not log[k(c)]["prune"] and done[k(c)] == SIZE


def test_latency_cap_at_the_stage_barrier(tmp_path):
    """Exact own 20-prefix; front = complete never-pruned BO only (NoMemory EXCLUDED, user 2026-11-05);
    cap = factor x fastest front latency; init protected."""
    space = fx.tiny_space_v070()
    full = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=2)
    slow = pick(space, filter_order="after", rank_impl="none", graph_impl="BFS", depth=3)
    peer = pick(space, filter_order="after", rank_impl="none", graph_impl="BFS", depth=2)  # same plan P2
    ok = pick(space, filter_order="before", rank_impl="none", graph_impl="BFS", depth=3)
    init_slow = pick(space, filter_order="after", rank_impl="ColBERT", graph_impl="BFS", depth=3)
    k = space.key
    # nothing Pareto-dominates the batch (highest quality); NoMemory (1 s) would set cap 2 if it were in the front
    table = {k(full): (50, 100, 8), k(slow): (90, 5000, 20), k(peer): (90, 5000, 10), k(ok): (90, 5000, 15),
             k(init_slow): (90, 5000, 40)}
    m = arm_manifest(space, protocol="t6-latency", factor=2.0)
    run, m, log, done = setup(tmp_path, space, m, full, [slow, peer, ok, init_slow], table, (5, 50, 1),
                              protect=[k(init_slow)])
    d = log[k(slow)]
    assert d["prune"] and d["prune_rule"] == "latency-cap" and not d["pareto_prune"]
    assert d["latency_cap"]["fastest_front_latency"] == pytest.approx(8.01, abs=0.02)
    assert d["latency_cap"]["cap"] == pytest.approx(16.02, abs=0.04) and "NoMemory" not in d["latency_cap"]["front"]
    assert not log[k(ok)]["prune"] and not log[k(peer)]["prune"] and not log[k(init_slow)]["prune"]
    assert done[k(slow)] == 20 and done[k(ok)] == SIZE and done[k(init_slow)] == SIZE
    with SampledLedger(run / "ledger.sqlite", m) as led:
        got = {str(r["task_id"]) for r in led.receipts(k(slow))}
    assert got == set(own_tasks(m, k(slow))[:20])  # decided on the exact own prefix


def test_latency_cap_keeps_the_last_config_of_an_executor_plan(tmp_path):
    """User 2026-11-05: the latency cap never prunes the last remaining configuration of an executor plan."""
    space = fx.tiny_space_v070()
    full = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=2)
    a = pick(space, filter_order="after", rank_impl="none", graph_impl="BFS", depth=3)  # P2, slower
    b = pick(space, filter_order="after", rank_impl="none", graph_impl="BFS", depth=2)  # P2, less slow
    c = pick(space, filter_order="before", rank_impl="none", graph_impl="BFS", depth=3)  # P1, alone
    k = space.key
    table = {k(full): (50, 100, 8), k(a): (90, 5000, 30), k(b): (90, 5000, 25), k(c): (90, 5000, 40)}
    m = arm_manifest(space, protocol="t6-planlast", factor=2.0)
    _, _, log, done = setup(tmp_path, space, m, full, [a, b, c], table, (5, 50, 1))
    assert log[k(a)]["prune_rule"] == "latency-cap" and done[k(a)] == 20
    assert not log[k(b)]["prune"] and log[k(b)]["guarded"] == "plan-last" and done[k(b)] == SIZE
    assert not log[k(c)]["prune"] and log[k(c)]["latency_cap"]["guarded"] == "plan-last" and done[k(c)] == SIZE


def test_decisions_commit_first_and_are_not_recomputed(tmp_path, monkeypatch):
    from tools import trace_v070_live as tl
    from trimem.experiments import trace_v070_pruning as tp

    space = fx.tiny_space_v070()
    full = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=2)
    c = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=3)  # dominated, same plan
    k = space.key
    table = {k(full): (100, 100, 1), k(c): (1, 10000, 100)}
    m = arm_manifest(space, protocol="t6-durable")
    run = tmp_path / m["protocol"]
    run.mkdir()
    m = {**m, "initial_keys": [k(full)], "live": {**m["live"], "run_dir": str(run)}}
    ev = values_evaluator(space, table, (5, 50, 8))

    def crash(*a, **kw):
        raise RuntimeError("simulated crash after the ledger commit, before the log write")

    with v070_patches(m):
        run_baseline(run, m, "TRAIN", ev)
        with SampledLedger(run / "ledger.sqlite", m) as led:
            led.reserve(space, [full])
            driver.evaluate_pending(run, m, space, led.pending(), ev)
        with SampledLedger(run / "ledger.sqlite", m) as led:
            led.reserve(space, [c])
            pending = led.pending()
        real_backfill = tp._backfill_log
        monkeypatch.setattr(tp, "_backfill_log", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            staged_evaluate_v070(run, m, space, pending, ev)
        monkeypatch.setattr(tp, "_backfill_log", real_backfill)
        assert not (run / "pruning_log.jsonl").exists()
        with SampledLedger(run / "ledger.sqlite", m) as led:
            assert led.stage_decisions(20)[(k(c), 20)]["prune_rule"] == "pareto"
            assert led.db.execute("SELECT count(*) FROM episodes WHERE config=? AND state='pruned'",
                                  (k(c),)).fetchone()[0] == SIZE - 20
            assert not led.pending()  # resume has nothing to evaluate: staged evaluation never runs again
        # fix round 2: the launcher back-fills committed decisions (startup / before each round)
        monkeypatch.setattr(tp, "decide_checkpoint", lambda *a, **kw: pytest.fail("recomputed a recorded decision"))
        assert tl.backfill_pruning_log(run, m) == 1
        assert tl.backfill_pruning_log(run, m) == 0  # idempotent
        staged_evaluate_v070(run, m, space, pending, ev)  # a replayed batch does not recompute either
    lines = [json.loads(x) for x in (run / "pruning_log.jsonl").read_text().splitlines()]
    assert [(d["key"], d["stage"], d["prune_rule"]) for d in lines] == [(k(c), 20, "pareto")]


def test_backfill_tolerates_torn_last_line(tmp_path):
    """Review round 2 item 10: a crash mid-append leaves a partial JSON line; backfill must not fail."""
    import json

    from trimem.experiments.trace_v070_pruning import _backfill_log

    log = tmp_path / "pruning_log.jsonl"
    log.write_text(json.dumps({"key": "a", "stage": 20, "prune": False}) + "\n" + '{"key": "b", "sta')
    n = _backfill_log(tmp_path, {("a", 20): {"prune": False}, ("b", 20): {"prune": True}})
    assert n == 1
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip().endswith("}")]
    assert {(r["key"], r["stage"]) for r in rows} == {("a", 20), ("b", 20)}
