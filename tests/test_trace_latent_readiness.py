"""Task 11: readiness gates R1-R6, gate rule and the d / variant choice (tiny space, CPU)."""

import json
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore", message=".*cache_root.*")

from tests.latent_fixtures import make_shared_init, tiny_model, tiny_space
from tools import latent_readiness as lr
from trimem.optimizer.latent_plan_codec import fit_affine, load_affine, load_codec
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import save_checkpoint

torch.set_num_threads(4)
OBJ = ("quality", "tokens", "latency")
GOOD_R = {"R1_val_exact": 1.0, "R2_family_min": 1.0, "R3_probe_valid": 1.0, "R4_collisions": 0}


def _g3(spearman=0.5, lpd=0.0):
    return {o: {"spearman": spearman, "lpd": lpd} for o in OBJ}


def _gate_entry(passes: dict, lpds: dict | None = None):
    """Per-d gates entry: one run, per-variant G3 with the given mean lpd."""
    lpds = lpds or {v: 0.0 for v in passes}
    return {
        "pass": dict(passes),
        "pass_any": any(passes.values()),
        "runs": {"r1": {"G2_spearman": 0.3, "variants": {v: {"G3": _g3(lpd=lpds[v])} for v in passes}}},
    }


def test_passes_readiness_thresholds():
    assert lr.passes_readiness(GOOD_R)
    assert lr.passes_readiness({**GOOD_R, "R1_val_exact": 0.95, "R2_family_min": 0.90})
    assert not lr.passes_readiness({**GOOD_R, "R1_val_exact": 0.949})
    assert not lr.passes_readiness({**GOOD_R, "R2_family_min": 0.89})
    assert not lr.passes_readiness({**GOOD_R, "R3_probe_valid": 0.9999})
    assert not lr.passes_readiness({**GOOD_R, "R4_collisions": 1})


def test_choose_smallest_passing_d():
    readiness = {16: {**GOOD_R, "R1_val_exact": 0.5}, 32: GOOD_R, 64: GOOD_R}
    gates = {d: _gate_entry({"A": True, "C": True}) for d in (16, 32, 64)}
    out = lr.choose_d(readiness, gates)
    assert out["d"] == 32 and out["variant"] in ("A", "C")
    assert out["missing"] == []
    assert "16" in out["reason"]


def test_choose_none_when_all_fail():
    readiness = {16: {**GOOD_R, "R4_collisions": 3}, 32: GOOD_R}
    gates = {16: _gate_entry({"A": True}), 32: _gate_entry({"A": False, "B": False})}
    out = lr.choose_d(readiness, gates)
    assert out["d"] is None and out["variant"] is None


def test_missing_checkpoint_reported():
    readiness = {16: {**GOOD_R, "R1_val_exact": 0.1}, 32: GOOD_R, 64: None}
    gates = {16: _gate_entry({"A": True}), 32: _gate_entry({"A": True}), 64: None}
    out = lr.choose_d(readiness, gates)
    assert out["d"] == 32
    assert out["missing"] == [{"d": 64, "missing": ["readiness", "gates"]}]
    # a d whose gates never ran (stop-at-first-pass) or are incomplete is missing too
    out = lr.choose_d({16: GOOD_R, 32: GOOD_R}, {16: {"runs": {}}})
    assert out["d"] is None
    assert out["missing"] == [{"d": 16, "missing": ["gates"]}, {"d": 32, "missing": ["gates"]}]


def test_choose_picks_best_lpd_variant():
    gates = {16: _gate_entry({"A": True, "B": True, "C": False}, {"A": -1.0, "B": -0.2, "C": 5.0})}
    out = lr.choose_d({16: GOOD_R}, gates)
    assert out["d"] == 16 and out["variant"] == "B"
    assert out["mean_lpd"]["B"] == pytest.approx(-0.2)


def test_family_rate_on_fixture():
    def cfg(order, impl):
        return {"filter_order": order, "graph_impl": impl}

    configs = [cfg("none", "BFS")] * 4 + [cfg("after", "beam")] * 5 + [cfg("before", "BFS")] * 2
    hits = [True, True, True, False] + [True] * 5 + [True, False]
    low, per = lr.family_rates(configs, hits)
    assert low == pytest.approx(0.5)
    assert per == {"none|BFS": 0.75, "after|beam": 1.0, "before|BFS": 0.5}


def test_collision_count():
    U = np.array([[0.1, 0.2], [0.1, 0.2], [0.9, 0.9]])
    assert lr.collisions(U, ["a", "b", "c"]) == 1
    assert lr.collisions(U, ["a", "a", "c"]) == 0  # same key counted once (distinct keys only)
    assert lr.collisions(U[[0, 2]], ["a", "c"]) == 0


def test_gate_rule_uses_margins():
    native = {"r1": _g3(0.5, -1.0), "r2": _g3(0.4, -2.0)}

    def latent(ds=0.0, dl=0.0, g2=0.2, obj="tokens"):
        out = {r: {"G2_spearman": 0.3, "G3": _g3(n["quality"]["spearman"], n["quality"]["lpd"])}
               for r, n in native.items()}
        out["r2"]["G2_spearman"] = g2
        out["r2"]["G3"][obj] = {"spearman": 0.4 + ds, "lpd": -2.0 + dl}
        return out

    assert lr.gate_pass(latent(), native)
    assert not lr.gate_pass(latent(ds=-0.11), native)
    assert lr.gate_pass(latent(ds=-0.09), native)
    assert not lr.gate_pass(latent(dl=-0.6), native)
    assert lr.gate_pass(latent(dl=-0.4), native)
    assert not lr.gate_pass(latent(g2=0.0), native)
    assert not lr.gate_pass(latent(g2=-0.1), native)
    assert not lr.gate_pass(latent(ds=float("nan")), native)
    assert not lr.gate_pass({"r1": latent()["r1"]}, native)  # every run is required
    assert lr.gate_pass(latent(ds=-0.19), native, spearman_margin=0.2)


# ---------------------------------------------------------------- tiny end to end

TINY_META = {"d": 4, "n_train": 192, "seed": 0, "heldout": {"exact": 1.0, "n": 16}}


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("ck") / "seqvae_d4.pt"
    save_checkpoint(path, tiny_model(4), tiny_space(), dict(TINY_META))
    return path


def test_corpus_rederived_as_training(tiny_ckpt):
    space = tiny_space()
    train, val = lr.corpus(space, TINY_META)
    expect = space.sample(208, seed=0, max_draws=4 * 208)
    assert train == expect[:192] and val == expect[192:]


def test_readiness_tiny_end_to_end(tiny_ckpt, tmp_path):
    space = tiny_space()
    r = lr.readiness(tiny_ckpt, space, probe=64, test=32, affine_cache=tmp_path)
    for k in ("R1_val_exact", "R2_family_min", "R3_probe_valid", "R5_test_exact", "probe_unique_rate"):
        assert 0.0 <= r[k] <= 1.0, k
    for k in ("R1_val_exact_repaired", "R5_test_exact_repaired"):
        assert 0.0 <= r[k] <= 1.0
    assert isinstance(r["R4_collisions"], int) and r["R4_collisions"] >= 0
    assert r["n_val"] == 16 and r["n_probe"] == 64 and 0 < r["n_test"] <= 32
    assert set(r["per_field_exact"]) >= {"filter_order", "graph_impl", "seed_count"}
    assert r["combination_holdout"] == "not established (uniform corpus)"
    assert r["seconds"] > 0 and r["d"] == 4
    # the provisional affine is fit_affine(train) and is cached next to the run
    aff, meta = load_affine(next(tmp_path.glob("*.provisional_affine.json")))
    train, _ = lr.corpus(space, TINY_META)
    ref = fit_affine(tiny_model(4), SeqGrammar(space), train)
    assert np.allclose(aff.lo, ref.lo) and np.allclose(aff.hi, ref.hi)
    assert meta["kind"] == lr.PROVISIONAL_KIND


def test_provisional_affine_cache_reused(tiny_ckpt, tmp_path, monkeypatch):
    space = tiny_space()
    model = tiny_model(4)
    a = lr.provisional_affine(tiny_ckpt, space, model, TINY_META, cache_dir=tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("cache not used")

    monkeypatch.setattr(lr, "_encode_z", boom)
    assert lr.provisional_affine(tiny_ckpt, space, model, TINY_META, cache_dir=tmp_path) == a


def test_freeze_affine_tiny(tiny_ckpt, tmp_path):
    run = make_shared_init(tmp_path)
    out = tmp_path / "affine_d4.json"
    rep = lr.freeze_affine(tiny_ckpt, run, out)
    side = json.loads((tmp_path / "affine_d4.json.json").read_text())
    assert side == rep
    assert rep["n_init"] == 10 and 0 <= rep["R6_init_roundtrip"] <= 10
    assert rep["checkpoint_sha256"] and rep["affine_sha256"]
    codec = load_codec(tiny_ckpt, out, tiny_space())
    assert codec.d == 4


def test_choose_cli(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"checkpoints": {"d16": GOOD_R, "d32": None}}))
    (tmp_path / "g.json").write_text(json.dumps({"d": {"d16": _gate_entry({"A": False, "C": True})}}))
    lr.main(["choose", "--readiness", str(tmp_path / "r.json"), "--gates", str(tmp_path / "g.json"),
             "--out", str(tmp_path / "c.json")])
    c = json.loads((tmp_path / "c.json").read_text())
    assert c["d"] == 16 and c["variant"] == "C"
    assert c["surrogate"] == {"kernel": "matern", "structure": "independent"}
    assert c["missing"] == [{"d": 32, "missing": ["readiness", "gates"]}]
