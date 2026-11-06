"""Stall override: epsilon 1e-3, forced rounds from <run>/stall_override.json, stock module untouched until install."""

import json
import pickle

from trimem.experiments import trace_v071_search as ts
from trimem.experiments import trace_v071_stall_override as so
from trimem.optimizer.joint_search_protocol import stagnation_status


def _restore():
    ts.stall_rule, ts.stall_state = so._STOCK["rule"], so._STOCK["state"]


def test_rule_merges_epsilon_and_keeps_none():
    m = {"v071": {"stall_rule": dict(ts.RULE)}}
    assert so.stall_rule(m)["epsilon"] == 1e-3 and so.stall_rule(m)["patience"] == 3
    assert so.stall_rule({"v071": {"stall_rule": None}}) is None


def test_small_gain_counts_as_stalled_with_the_new_epsilon():
    hv = [0.29240252, 0.29240788, 0.2925, 0.2926]
    events = [{"nhv": v, "kind": "ordinary"} for v in hv]
    assert stagnation_status(0.2924, events, patience=3, epsilon=1e-3)["force_next"]
    assert not stagnation_status(0.2924, events, patience=3, epsilon=1e-6)["force_next"]


def test_force_rounds_force_only_the_named_round(tmp_path, monkeypatch):
    (tmp_path / so.FILE).write_text(json.dumps({"force_rounds": ["round_042"]}))
    stock = {"force_exploration": False, "stagnant_rounds": 1}
    monkeypatch.setitem(so._STOCK, "state", lambda *a, **k: dict(stock))
    assert so.stall_state(tmp_path, {}, [], None, before="round_042")["force_exploration"] is True
    assert so.stall_state(tmp_path, {}, [], None, before="round_046")["force_exploration"] is False
    assert so.stall_state(tmp_path, {}, [], None, before=None)["force_exploration"] is False


def test_no_file_means_no_forced_round(tmp_path):
    assert so.force_rounds(tmp_path) == set()


def test_worker_is_picklable_and_install_patches_the_module():
    try:
        assert pickle.loads(pickle.dumps(so.latent_search_worker)) is so.latent_search_worker
        so.install()
        assert ts.stall_rule is so.stall_rule and ts.stall_state is so.stall_state
    finally:
        _restore()
