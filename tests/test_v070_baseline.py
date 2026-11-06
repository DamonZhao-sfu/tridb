"""T5: NoMemory baseline ledger: key, prepare / run / import, never in the main ledger."""

import json

import pytest

from tests import v070_fixtures as fx
from trimem.experiments.trace_v070_baseline import (
    NOMEMORY,
    BaselineSpace,
    baseline_entry,
    baseline_key,
    import_baseline,
    prepare_baseline,
    summarize,
    train_sample,
)
from trimem.experiments.trace_v070_ledger import SampledLedger


def test_key_stable_and_distinct():
    a, b = BaselineSpace("fp-1"), BaselineSpace("fp-2")
    assert a.key(NOMEMORY) == BaselineSpace("fp-1").key(NOMEMORY) != b.key(NOMEMORY)
    space = fx.tiny_space_v070("fp-1")
    assert a.key(NOMEMORY) not in {space.key(c) for c in fx.enumerate_space(space)[:50]}
    with pytest.raises(ValueError):
        a.canonicalize({"baseline": "Other"})


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return fx.make_shared_init(tmp_path_factory.mktemp("baseline"), size=40)


def test_shared_init_nomemory_complete_and_outside_main_ledger(shared):
    m = json.loads((shared / "manifest.json").read_text())
    e = baseline_entry(shared, "TRAIN")
    assert e["complete"] and e["key"] == baseline_key(m) == m["baselines"]["NoMemory"]["key"]
    assert sorted(e["tasks"]) == sorted(train_sample(m)) and len(e["receipts"]) == 40
    with SampledLedger(shared / "ledger.sqlite", m) as led:
        keys = [k for (k,) in led.db.execute("SELECT key FROM configs")]
        episodes = led.db.execute("SELECT count(*) FROM episodes").fetchone()[0]
    assert len(keys) == 10 and e["key"] not in keys and episodes == 10 * 40
    assert e["key"] not in m["initial_keys"]
    s = summarize(e)
    assert s["episodes"] == 40 and s["raw_score"] < 15


def test_prepare_is_idempotent_and_refuses_a_changed_manifest(shared):
    m = json.loads((shared / "manifest.json").read_text())
    assert prepare_baseline(shared, m, split="TRAIN")["key"] == baseline_key(m)
    with pytest.raises(ValueError):
        prepare_baseline(shared, {**m, "protocol": "changed"}, split="TRAIN")


def test_import_baseline(shared, tmp_path):
    m = json.loads((shared / "manifest.json").read_text())
    dst = tmp_path / "arm"
    dm = {**m, "protocol": "arm-protocol", "live": {**m["live"], "run_dir": str(dst)}}
    bm = import_baseline(shared, dst, dm)
    e = baseline_entry(dst, "TRAIN")
    assert e["complete"] and bm["protocol"] == "arm-protocol-nomemory"
    assert all(r["protocol"] == bm["protocol"] and r["imported_from"]["run"] == str(shared) for r in e["receipts"])
    other = {**dm, "space": {**dm["space"], "execution_fingerprint": "different-runtime"}}
    with pytest.raises(ValueError, match="key differs"):
        import_baseline(shared, tmp_path / "arm2", other)
    with pytest.raises(ValueError, match="missing or incomplete"):
        import_baseline(tmp_path / "nowhere", tmp_path / "arm3", dm)


def test_parity_refused_on_import_and_on_a_complete_ledger(shared, tmp_path):
    from tools import trace_v070_live as tl

    m = json.loads((shared / "manifest.json").read_text())
    dst = tmp_path / "arm_parity"
    for key, value in (("model", "another-model"), ("llm_bases", ["http://127.0.0.1:9999/v1"]),
                       ("agent_max_tokens", 8192)):
        dm = {**m, "protocol": "arm-parity", "live": {**m["live"], "run_dir": str(dst), key: value}}
        with pytest.raises(ValueError, match="runtime differs"):
            import_baseline(shared, dst, dm)
    dm = {**m, "protocol": "arm-parity", "task_concurrency": 8, "live": {**m["live"], "run_dir": str(dst)}}
    with pytest.raises(ValueError, match="runtime differs"):
        import_baseline(shared, dst, dm)
    assert not (dst / "baselines/nomemory/ledger.sqlite").exists()  # refused before anything was restated
    assert baseline_entry(shared, "TRAIN")["complete"]
    changed = {**m, "live": {**m["live"], "model": "another-model"}}
    with pytest.raises(ValueError, match="NoMemory runtime differs"):
        tl.baseline_stage(shared, changed)  # complete ledger: parity still enforced
    assert tl.baseline_stage(shared, m)["complete"]


def test_baseline_manifest_canonical_run_dir():
    """Live bug 2026-11-05: prepare passed an absolute run dir, the launcher a relative one -> manifests differed."""
    import json
    from pathlib import Path

    import pytest

    from trimem.experiments import trace_v070_baseline as b
    from trimem.experiments.trace_joint_plan_search import digest

    run = Path("results/trace_v070_alfworld_init_20261105")
    if not (run / "manifest.json").exists():
        pytest.skip("prepared ALFWorld init not present")
    m = json.loads((run / "manifest.json").read_text())
    tasks = ["alfworld_1", "alfworld_2"]
    rel = b.baseline_manifest(m, split="TRAIN", tasks=tasks, run_dir=str(run))
    ab = b.baseline_manifest(m, split="TRAIN", tasks=tasks, run_dir=run.resolve())
    via_symlink = b.baseline_manifest(m, split="TRAIN", tasks=tasks, run_dir="/localhome/hza214/tridb/" + str(run))
    assert digest(rel) == digest(ab) == digest(via_symlink)
