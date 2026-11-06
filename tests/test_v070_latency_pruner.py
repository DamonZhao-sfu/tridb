"""T7 (fix round 1): latent_latency_pruner_v070 is an OFFLINE dry-run reporter. It reports the stock latency rule
with NoMemory in the front reference, never judges NoMemory, never changes the ledger, and refuses non-dry use.
The live latency cap runs at the stage barrier (tests/test_v070_pruning.py::test_latency_cap_at_the_stage_barrier).
"""

import json

import pytest

from tests import v070_fixtures as fx
from tests.test_v070_pruning import arm_manifest, pick, values_evaluator
from tools import latent_latency_pruner_v070 as lp
from trimem.experiments import trace_joint_plan_search as driver
from trimem.experiments.trace_v070_baseline import run_baseline
from trimem.experiments.trace_v070_ledger import SampledLedger, own_tasks
from trimem.experiments.trace_v070_pruning import FACTORS
from trimem.experiments.trace_v070_runtime import v070_patches


def make_run(tmp_path, name):
    space = fx.tiny_space_v070()
    full = pick(space, filter_order="none", rank_impl="ColBERT", graph_impl="BFS", depth=2)
    x = pick(space, filter_order="after", rank_impl="none", graph_impl="BFS", depth=3)
    k = space.key
    table = {k(full): (100, 100, 30), k(x): (50, 120, 20)}
    m = arm_manifest(space, protocol=name)
    run = tmp_path / name
    run.mkdir()
    m = {**m, "initial_keys": [k(full)], "live": {**m["live"], "run_dir": str(run)}}
    ev = values_evaluator(space, table, (5, 50, 8))
    with v070_patches(m):
        run_baseline(run, m, "TRAIN", ev)
        with SampledLedger(run / "ledger.sqlite", m) as led:
            led.reserve(space, [full, x])
            pending = led.pending()
        first = set(own_tasks(m, k(x))[:20])
        todo = [p for p in pending if p["key"] == k(full) or str(p["task_id"]) in first]
        driver.evaluate_pending(run, m, space, todo, ev)
    driver.atomic_json(run / "manifest.json", m)
    return run, m, k(x)


def pruned_count(run, m, key):
    with SampledLedger(run / "ledger.sqlite", m) as led:
        return led.db.execute("SELECT count(*) FROM episodes WHERE config=? AND state='pruned'", (key,)).fetchone()[0]


@pytest.mark.parametrize("front", [True, False])
def test_reporter_uses_nomemory_front_and_never_prunes(tmp_path, front):
    run, m, x = make_run(tmp_path, f"lat-{front}")
    decisions = lp.run_once(run, {}, factor=2.0, step=20, baseline_front=front)
    assert [d["key"] for d in decisions] == [x]  # init and NoMemory are never judged
    assert decisions[0]["cap"] == pytest.approx(2.0 * (8.0 if front else 30.0), abs=0.05)
    assert decisions[0]["prune"] is front  # what the rule WOULD decide
    assert pruned_count(run, m, x) == 0  # ... but nothing is applied
    rec = [json.loads(line) for line in (run / "latency_prune_report.jsonl").read_text().splitlines()]
    assert rec[-1]["nomemory_in_front_reference"] is front and rec[-1]["dry_run"] is True


def test_non_dry_refused_and_factors(tmp_path):
    assert FACTORS == {"alfworld": 2.0, "sciworld": 3.0, "webshop": 2.0} and lp.FACTORS is FACTORS
    run, m, _ = make_run(tmp_path, "lat-refuse")
    with pytest.raises(SystemExit, match="dry-run reporter"):
        lp.run_once(run, {}, factor=2.0, step=20, dry=False)
    (run / "manifest.json").write_text(json.dumps({k: v for k, v in m.items() if k != "v070"}))
    with pytest.raises(SystemExit):
        lp.run_once(run, {}, factor=2.0, step=20)
