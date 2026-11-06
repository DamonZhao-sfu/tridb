"""Task 11: offline G2/G3 gates for the sequence VAE (--kind seq), K-fold CV shared by all models."""

import json
import math
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore", message=".*cache_root.*")

from tests.latent_fixtures import (
    enumerate_space,
    make_shared_init,
    synthetic_receipt,
    tiny_model,
    tiny_space,
)
from tools import latent_offline_gates as gates
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.latent_seq_vae import save_checkpoint
from trimem.optimizer.plan_surrogate import ObjectiveProtocol, aggregate_panel

torch.set_num_threads(4)
OBJ = ("quality", "tokens", "latency")
TASKS = ["t0", "t1", "t2", "t3"]
NUTS = (8, 4, 2)


def _ckpt(dirpath, d):
    path = dirpath / f"seqvae_d{d}.pt"
    meta = {"d": d, "n_train": 192, "seed": 0, "heldout": {"exact": 1.0, "n": 16}}
    save_checkpoint(path, tiny_model(d), tiny_space(), meta)
    return path


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    base = tmp_path_factory.mktemp("gates")
    return {"run": make_shared_init(base), "d4": _ckpt(base, 4), "d6": _ckpt(base, 6), "dir": base}


def _g3(v=0.5):
    return {o: {"lpd": -v, "spearman": v} for o in OBJ}


def _fake_cv(calls, out=None):
    def native(space, obs, objective, Y, V, folds):
        calls.append(("native", [list(map(int, f)) for f in folds], None))
        return {"G3": _g3(), "fit_seconds": {"mean": 0.0, "max": 0.0, "n": len(folds)}}

    def latent(codec, obs, objective, Y, V, folds, *, kernel, structure, seed=0, nuts=None):
        done = json.loads(out.read_text()) if out is not None and out.exists() else {}
        calls.append((f"{kernel}/{structure}/d{codec.d}", [list(map(int, f)) for f in folds], done))
        return {"G3": _g3(), "fit_seconds": {"mean": 0.0, "max": 0.0, "n": len(folds)}}

    return native, latent


def test_kfolds_deterministic():
    a, b = gates.kfolds(10, 3), gates.kfolds(10, 3)
    assert [f.tolist() for f in a] == [f.tolist() for f in b]
    ref = np.array_split(np.random.default_rng(0).permutation(10), 3)
    assert [f.tolist() for f in a] == [f.tolist() for f in ref]
    assert sorted(np.concatenate(a).tolist()) == list(range(10))


def test_cv_folds_deterministic_and_shared(shared, tmp_path, monkeypatch):
    calls = []
    out = tmp_path / "g.json"
    native, latent = _fake_cv(calls, out)
    monkeypatch.setattr(gates, "native_cv", native)
    monkeypatch.setattr(gates, "latent_cv", latent)
    rep = gates.run_seq([shared["d4"]], [shared["run"]], out, variants=("A", "B", "C"), folds=3,
                        affine_cache=shared["dir"])
    want = [f.tolist() for f in gates.kfolds(10, 3)]
    assert len(calls) == 4  # native once + 3 variants
    assert all(c[1] == want for c in calls)
    assert rep["runs"][shared["run"].name]["folds"] == want
    # JSON is written after every (run, d, variant): each later call sees the earlier variants
    seen = [sorted(((c[2].get("d") or {}).get("d4") or {}).get("runs", {}).get(shared["run"].name, {})
                   .get("variants", {})) for c in calls[1:]]
    assert seen == [[], ["A"], ["A", "B"]]
    assert json.loads(out.read_text())["d"]["d4"]["pass_any"] in (True, False)


def test_stop_at_first_pass(shared, tmp_path, monkeypatch):
    calls = []
    native, latent = _fake_cv(calls)
    monkeypatch.setattr(gates, "native_cv", native)
    monkeypatch.setattr(gates, "latent_cv", latent)
    monkeypatch.setattr(gates, "smoothness", lambda D, q: 0.5)
    ck = [shared["d6"], shared["d4"]]  # processed in ascending d
    rep = gates.run_seq(ck, [shared["run"]], tmp_path / "a.json", variants=("C",), folds=3,
                        stop_at_first_pass=True, affine_cache=shared["dir"])
    assert list(rep["d"]) == ["d4"] and rep["d"]["d4"]["pass"] == {"C": True}
    assert rep["stopped_after"] == 4
    rep = gates.run_seq(ck, [shared["run"]], tmp_path / "b.json", variants=("C",), folds=3,
                        affine_cache=shared["dir"])
    assert list(rep["d"]) == ["d4", "d6"] and rep["d"]["d6"]["pass_any"] is True


@pytest.fixture(scope="module")
def tiny_obs():
    space, model = tiny_space(), tiny_model(4)
    configs = enumerate_space(space)
    codec = LatentCodec(space, model, fit_affine(model, SeqGrammar(space), configs), checkpoint_sha256="x")
    manifest = {"protocol": "p", "evaluation_split": "TRAIN"}
    picks = configs[:: len(configs) // 12][:12]
    obs = [aggregate_panel(space, c, [synthetic_receipt(c, t, manifest) for t in TASKS], task_ids=TASKS,
                           protocol="p", allow_synthetic=True) for c in picks]
    objective = ObjectiveProtocol.from_initial(obs)
    Y, V = objective.tensors(obs)
    return codec, obs, objective, Y, V


@pytest.mark.parametrize("kernel", ["saas", "matern"])
def test_coupled_cost_posterior_has_three_objectives(tiny_obs, kernel):
    codec, obs, objective, Y, V = tiny_obs
    folds = gates.kfolds(len(obs), 3)
    r = gates.latent_cv(codec, obs, objective, Y, V, folds, kernel=kernel, structure="coupled-cost",
                        nuts=NUTS)
    assert set(r["G3"]) == set(OBJ)
    for o in OBJ:
        assert math.isfinite(r["G3"][o]["lpd"]), o
    assert r["fit_seconds"]["n"] == 3 and r["fit_seconds"]["max"] >= r["fit_seconds"]["mean"] > 0
    assert set(r["objective_seconds"]) == {"quality", "cost"}


def test_independent_saas_cv_finite(tiny_obs):
    codec, obs, objective, Y, V = tiny_obs
    r = gates.latent_cv(codec, obs, objective, Y, V, gates.kfolds(len(obs), 3), kernel="saas",
                        structure="independent", nuts=NUTS)
    assert all(math.isfinite(r["G3"][o]["lpd"]) for o in OBJ)
    assert set(r["objective_seconds"]) == set(OBJ)


def test_posterior_moments_mixture():
    class P:
        mean = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]]], dtype=torch.double)  # (S=2, n=1, 3)
        variance = torch.ones(2, 1, 3, dtype=torch.double)

    class M:
        def posterior(self, U):
            return P()

    mean, var = gates.posterior_mv(M(), torch.zeros(1, 2, dtype=torch.double))
    assert mean.shape == (1, 3) and var.shape == (1, 3)
    assert np.allclose(mean, [[1.0, 1.0, 1.0]])
    assert np.allclose(var, [[2.0, 1.0, 2.0]])  # E[var] + Var[mean]


def test_seq_gates_end_to_end_tiny(shared, tmp_path):
    out = tmp_path / "e2e.json"
    rep = gates.run_seq([shared["d4"]], [shared["run"]], out, variants=("C",), folds=3,
                        affine_cache=shared["dir"])
    on_disk = json.loads(out.read_text())  # NaN-safe comparison: structure only
    assert set(on_disk) == set(rep) and set(on_disk["d"]["d4"]) == set(rep["d"]["d4"])
    name = shared["run"].name
    r = rep["runs"][name]
    assert r["n"] == 10 and set(r["G3_native"]) == set(OBJ) and "G2_native_hamming" in r
    e = rep["d"]["d4"]
    v = e["runs"][name]["variants"]["C"]
    assert set(v["G3"]) == set(OBJ) and v["fit_seconds"]["n"] == 3
    assert "G2_spearman" in e["runs"][name]
    assert set(e["pass"]) == {"C"} and isinstance(e["pass_any"], bool)
    assert e["checkpoint_sha256"] and e["affine"]["kind"]
    assert len(rep["loadavg_start"]) == 3 and rep["folds"] == 3
    assert rep["margins"] == {"spearman": 0.10, "lpd": 0.5}


def test_slot_is_default_kind():
    a = gates.build_parser().parse_args(["--checkpoints", "x", "--runs", "y", "--out", "z"])
    assert a.kind == "slot" and a.folds == 10 and a.variants == "A,B,C" and not a.stop_at_first_pass
