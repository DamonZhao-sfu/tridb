"""SAAS GP with NumPyro/JAX NUTS, loaded into BoTorch (user 2026-11-04 extra task)."""

import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", message=".*cache_root.*")

import numpy as np
import pytest
import torch

from tests.latent_fixtures import (
    enumerate_space,
    synthetic_receipt,
    tiny_model,
    tiny_space,
)
from tools import latent_offline_gates as gates
from tools.latent_readiness import VARIANTS
from trimem.optimizer import latent_numpyro_saas as nps
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_plan_surrogate import (
    acquisition_u,
    fit_latent_model,
    latent_snapshot,
)
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.plan_surrogate import ObjectiveProtocol, aggregate_panel

torch.set_num_threads(4)
OBJ = ("quality", "tokens", "latency")
TASKS = ["t0", "t1", "t2", "t3"]
MANIFEST = {"protocol": "p", "evaluation_split": "TRAIN"}
SMALL = {"warmup": 32, "samples": 16, "thinning": 4}  # 4 retained samples
META = {"checkpoint_sha256": "ck", "affine_sha256": "af", "objective_sha256": "ob", "run_id": "run",
        "panel_identity": "p"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module", autouse=True)
def jax_cache_dir(tmp_path_factory):
    """Persistent compilation cache in a temp dir for this module; removed afterwards. No CPU pinning
    here (pinning is tested in subprocesses); run pytest under taskset to bound the CPUs."""
    path = tmp_path_factory.mktemp("jaxcache")
    nps.enable_compilation_cache(path)
    yield path
    nps.enable_compilation_cache(None)
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def world():
    space, model = tiny_space(), tiny_model(4)
    configs = enumerate_space(space)
    codec = LatentCodec(space, model, fit_affine(model, SeqGrammar(space), configs), checkpoint_sha256="x")
    picks = configs[:: len(configs) // 12][:12]
    obs = [aggregate_panel(space, c, [synthetic_receipt(c, t, MANIFEST) for t in TASKS], task_ids=TASKS,
                           protocol="p", allow_synthetic=True) for c in picks]
    objective = ObjectiveProtocol.from_initial(obs)
    Y, V = objective.tensors(obs)
    return codec, obs, objective, Y, V, configs


def _synthetic(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g, dtype=torch.double)
    f = torch.sin(2 * math.pi * x[:, 0]) + 0.5 * x[:, 1] ** 2
    v = torch.full((n, 1), 0.01, dtype=torch.double)
    y = (f + 0.1 * torch.randn(n, generator=g, dtype=torch.double)).unsqueeze(-1)
    return x, y, v


def _mixture(model, X):
    with torch.no_grad():
        p = model.posterior(X)
        m, v = p.mean, p.variance
    return m.mean(0), v.mean(0) + m.var(0, unbiased=False)


# ------------------------------------------------------------ exactness of the NumPyro model
def test_log_density_matches_pyro_site_by_site():
    """log p(params, Y) of the NumPyro model == Pyro trace of SaasPyroModel.sample, constrained
    space, same transformed data (Standardize, clamped Yvar)."""
    import pyro
    from botorch.models import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms.outcome import Standardize

    x, y, v = _synthetic(20, 4, 0)
    m = SaasFullyBayesianSingleTaskGP(x, y, train_Yvar=v, outcome_transform=Standardize(m=1))
    pm = m.pyro_model
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        params = {
            "outputscale": torch.rand((), generator=g, dtype=torch.double) * 5 + 0.1,
            "mean": torch.randn((), generator=g, dtype=torch.double),
            "kernel_tausq": torch.rand((), generator=g, dtype=torch.double) + 0.01,
            "_kernel_inv_length_sq": torch.rand(4, generator=g, dtype=torch.double) * 3 + 0.01,
        }
        tr = pyro.poutine.trace(pyro.poutine.condition(pm.sample, data=params)).get_trace()
        lp_pyro = float(tr.log_prob_sum())
        lp_np = nps.log_density({k: t.numpy() for k, t in params.items()}, pm.train_X.numpy(),
                                pm.train_Y.squeeze(-1).numpy(), pm.train_Yvar.squeeze(-1).numpy())
        assert abs(lp_np - lp_pyro) <= 1e-8 * max(1.0, abs(lp_pyro)), (lp_np, lp_pyro)


def test_padding_shifts_log_density_by_a_constant():
    """Padding n -> n_pad (masked rows, unit noise, zero residual) adds exactly
    -0.5*log(2*pi) per padded row, so the hyperparameter posterior and NUTS are unchanged."""
    x, y, v = _synthetic(13, 3, 2)
    rng = np.random.default_rng(0)
    for _ in range(4):
        params = {"outputscale": rng.uniform(0.1, 4), "mean": rng.normal(), "kernel_tausq": rng.uniform(0.01, 1),
                  "_kernel_inv_length_sq": rng.uniform(0.01, 3, size=3)}
        a = nps.log_density(params, x.numpy(), y[:, 0].numpy(), v[:, 0].numpy())
        Xp, Yp, Vp, M = nps.pad_inputs(x.numpy(), y[:, 0].numpy(), v[:, 0].numpy(), 16)
        b = nps.log_density(params, Xp, Yp, Vp, M)
        assert Xp.shape == (16, 3) and M.sum() == 13
        assert b - a == pytest.approx(-0.5 * math.log(2 * math.pi) * 3, abs=1e-9)


# ------------------------------------------------------------ statistical agreement with Pyro
def test_numpyro_saas_matches_pyro_posterior():
    """Pyro NUTS (BoTorch) vs NumPyro NUTS, same model and NUTS sizes (128/64/4 -> 16 kept).
    Tolerance (MC noise of two independent short chains): mean |mu_np - mu_pyro| over 10 test
    points <= 0.25 * mean posterior sd, max <= 0.75 * sd; variance ratio within [0.5, 2] on
    average."""
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.models import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms.outcome import Standardize

    x, y, v = _synthetic(20, 4, 0)
    xt = _synthetic(10, 4, 99)[0]
    ref = SaasFullyBayesianSingleTaskGP(x, y, train_Yvar=v, outcome_transform=Standardize(m=1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        fit_fully_bayesian_model_nuts(ref, warmup_steps=128, num_samples=64, thinning=4, disable_progbar=True)
    got = nps.saas_single_numpyro(x, y, v, seed=0, warmup=128, samples=64, thinning=4)
    assert got.num_mcmc_samples == ref.num_mcmc_samples == 16
    mr, vr = _mixture(ref, xt)
    mg, vg = _mixture(got, xt)
    sd = vr.sqrt()
    diff = (mg - mr).abs()
    assert float(diff.mean()) <= 0.25 * float(sd.mean()), (diff.squeeze(), sd.squeeze())
    assert float(diff.max()) <= 0.75 * float(sd.mean())
    ratio = float((vg / vr).mean())
    assert 0.5 <= ratio <= 2.0, ratio
    # hyperparameter summary: median log lengthscale per dim; dims 2, 3 do not enter f -> shrink
    lr = ref.covar_module.base_kernel.lengthscale.squeeze(-2).log().median(0).values
    lg = got.covar_module.base_kernel.lengthscale.squeeze(-2).log().median(0).values
    assert (lr[:2] - lg[:2]).abs().max() <= 0.75, (lr, lg)
    for ll in (lr, lg):
        assert ll[2:].min() > ll[:2].max(), ll


def _pyro_fit(x, y1, v1, seed):
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.models import SaasFullyBayesianSingleTaskGP
    from botorch.models.transforms.outcome import Standardize

    ref = SaasFullyBayesianSingleTaskGP(x, y1, train_Yvar=v1, outcome_transform=Standardize(m=1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fit_fully_bayesian_model_nuts(ref, warmup_steps=128, num_samples=64, thinning=4, disable_progbar=True)
    return ref


def _agree(got, ref, xt):
    """Same tolerance as test_numpyro_saas_matches_pyro_posterior."""
    mr, vr = _mixture(ref, xt)
    mg, vg = _mixture(got, xt)
    sd, diff = vr.sqrt(), (mg - mr).abs()
    ratio = float((vg / vr).mean())
    return (float(diff.mean()) <= 0.25 * float(sd.mean()) and float(diff.max()) <= 0.75 * float(sd.mean())
            and 0.5 <= ratio <= 2.0), (float(diff.mean() / sd.mean()), float(diff.max() / sd.mean()), ratio)


def test_parallel_matches_pyro_posterior():
    """The vmapped 3-objective fit (default) vs Pyro NUTS per objective, the same tolerance."""
    x, y0, v = _synthetic(20, 4, 0)
    y = torch.cat([y0, (3 * x[:, 2:3] + x[:, 3:4]), torch.cos(3 * x[:, :1]) + x[:, 2:3]], dim=-1)
    y[:, 1:] += 0.1 * torch.randn(20, 2, generator=torch.Generator().manual_seed(5), dtype=torch.double)
    vv = v.expand(20, 3).clone()
    xt = _synthetic(10, 4, 99)[0]
    got = nps.saas_multi_numpyro(x, y, vv, seed=0, warmup=128, samples=64, thinning=4)
    assert [m.num_mcmc_samples for m in got] == [16, 16, 16]
    assert all(m.numpyro_info["mode"] == "vmap" for m in got)
    for i in range(3):
        ok, stats = _agree(got[i], _pyro_fit(x, y[:, i : i + 1], vv[:, i : i + 1], i), xt)
        assert ok, (i, stats)


def test_parallel_is_default_and_matches_sequential_shapes(world):
    codec, obs, objective, _, _, configs = world
    calls = []
    par = nps.fit_saas_numpyro(codec, obs, objective, seed=0, on_objective=lambda i, n: calls.append(i), **SMALL)
    seq = nps.fit_saas_numpyro(codec, obs, objective, seed=0, parallel=False, **SMALL)
    assert calls == [0, 1, 2]
    assert [m.numpyro_info["mode"] for m in par.models] == ["vmap"] * 3
    assert [m.numpyro_info["mode"] for m in seq.models] == ["sequential"] * 3
    U = codec.encode_many(configs[:5])
    with torch.no_grad():
        assert par.posterior(U).mean.shape == seq.posterior(U).mean.shape == torch.Size([4, 5, 3])
    again = nps.fit_saas_numpyro(codec, obs, objective, seed=0, **SMALL)
    for a, b in zip(par.models, again.models):
        assert torch.equal(a.covar_module.base_kernel.lengthscale, b.covar_module.base_kernel.lengthscale)


# ------------------------------------------------------------ loaded model is a normal BoTorch SAAS GP
def test_loaded_model_shapes_and_acquisition(world):
    from botorch.models import ModelListGP, SaasFullyBayesianSingleTaskGP

    codec, obs, objective, _, _, configs = world
    gp = nps.fit_saas_numpyro(codec, obs, objective, seed=0, **SMALL)
    assert isinstance(gp, ModelListGP) and len(gp.models) == 3
    assert all(isinstance(m, SaasFullyBayesianSingleTaskGP) for m in gp.models)
    assert [m.num_mcmc_samples for m in gp.models] == [4, 4, 4]
    ref = fit_latent_model(codec, obs, objective, seed=0, nuts=(8, 16, 4))
    U = codec.encode_many(configs[:5])
    with torch.no_grad():
        assert gp.posterior(U).mean.shape == ref.posterior(U).mean.shape == torch.Size([4, 5, 3])
    snap = latent_snapshot(codec, gp, obs, objective, seed=3)
    U = codec.encode_many(configs[:6]).clone().requires_grad_(True)
    acquisition_u(snap, U).sum().backward()
    assert torch.isfinite(U.grad).all() and U.grad.abs().max() > 0
    seen = {codec.key(o.config) for o in obs}
    cands = [c for c in configs if codec.key(c) not in seen][:20]
    assert all(math.isfinite(float(s)) for s in snap.score(cands))


def test_seed_determinism(world):
    codec, obs, objective, _, _, _ = world
    a = nps.fit_saas_numpyro(codec, obs, objective, seed=7, **SMALL)
    b = nps.fit_saas_numpyro(codec, obs, objective, seed=7, **SMALL)
    c = nps.fit_saas_numpyro(codec, obs, objective, seed=8, **SMALL)
    for ma, mb, mc in zip(a.models, b.models, c.models):
        la, lb, lc = (m.covar_module.base_kernel.lengthscale for m in (ma, mb, mc))
        assert torch.equal(la, lb) and torch.equal(ma.mean_module.constant, mb.mean_module.constant)
        assert torch.equal(ma.covar_module.outputscale, mb.covar_module.outputscale)
        assert not torch.equal(la, lc)


def test_on_objective_order_and_validation(world):
    codec, obs, objective, _, _, _ = world
    calls = []
    nps.fit_saas_numpyro(codec, obs, objective, on_objective=lambda i, n: calls.append((i, n)), **SMALL)
    assert calls == [(0, "quality"), (1, "tokens"), (2, "latency")]
    with pytest.raises(ValueError, match="Repeated"):
        nps.fit_saas_numpyro(codec, obs + obs[:1], objective, **SMALL)
    bad = dataclasses.replace(obs[0], panel_identity="other-panel")
    with pytest.raises(ValueError, match="panel"):
        nps.fit_saas_numpyro(codec, obs[1:] + [bad], objective, **SMALL)
    with pytest.raises(ValueError, match="thinning must be"):
        nps.fit_saas_numpyro(codec, obs, objective, warmup=8, samples=2, thinning=0)


def test_padded_bucket_reuses_compiled_sampler(world):
    """Two fits whose n differs inside one padding bucket compile the sampler once."""
    codec, obs, objective, _, _, _ = world
    x = torch.tensor([codec.encode(o.config) for o in obs], dtype=torch.double)
    y, noise = objective.tensors(obs)
    nps.saas_single_numpyro(x, y[:, :1], noise[:, :1], seed=0, **SMALL)
    before = nps.compiled_samplers()
    nps.saas_single_numpyro(x[:-1], y[:-1, :1], noise[:-1, :1], seed=1, **SMALL)
    assert nps.compiled_samplers() == before
    assert nps.bucket(11) == nps.bucket(12)


# ------------------------------------------------------------ gates dispatch
def test_variants_F_G_registered():
    assert VARIANTS["F"] == {"kernel": "saas-numpyro", "structure": "independent"}
    assert VARIANTS["G"] == {"kernel": "saas-numpyro", "structure": "independent", "nuts": [256, 128, 16]}


def test_gates_dispatch_F(world, monkeypatch):
    codec, obs, objective, Y, V, _ = world
    seen = []
    real = nps.fit_saas_numpyro

    def spy(*a, **k):
        seen.append({key: k[key] for key in ("warmup", "samples", "thinning", "seed")})
        return real(*a, **k)

    monkeypatch.setattr(nps, "fit_saas_numpyro", spy)
    r = gates.latent_cv(codec, obs, objective, Y, V, gates.kfolds(len(obs), 3),
                        **{"nuts": (32, 16, 4), **VARIANTS["F"]})
    assert len(seen) == 3 and seen[0] == {"warmup": 32, "samples": 16, "thinning": 4, "seed": 0}
    assert all(math.isfinite(r["G3"][o]["lpd"]) for o in OBJ)
    assert set(r["objective_seconds"]) == {"joint"}  # one vmapped program for the 3 objectives
    assert r["fit_seconds"]["n"] == 3


def test_gates_dispatch_F_defaults_to_live_nuts(world, monkeypatch):
    codec, obs, objective, Y, V, _ = world
    seen = []

    def fake(codec_, obs_, objective_, **k):
        seen.append((k["warmup"], k["samples"], k["thinning"]))
        return real(codec_, obs_, objective_, **{**k, **SMALL})

    real = nps.fit_saas_numpyro
    monkeypatch.setattr(nps, "fit_saas_numpyro", fake)
    gates.latent_cv(codec, obs, objective, Y, V, gates.kfolds(len(obs), 2), nuts=None, **VARIANTS["F"])
    assert seen == [(512, 256, 16)] * 2


# ------------------------------------------------------------ speed option 2a: compilation cache
def test_compilation_cache_dir_configured(jax_cache_dir):
    import jax

    assert nps.COMPILATION_CACHE == str(jax_cache_dir)
    assert jax.config.jax_compilation_cache_dir == str(jax_cache_dir)
    assert jax.config.jax_persistent_cache_min_compile_time_secs == nps.CACHE_MIN_COMPILE_SECS == 1.0
    assert jax.config.jax_compilation_cache_max_size == nps.CACHE_MAX_BYTES == 2 * 1024**3
    x, y, v = _synthetic(9, 2, 3)
    nps.saas_multi_numpyro(x, y.expand(9, 3).clone(), v.expand(9, 3).clone(), seed=0, warmup=8, samples=4, thinning=2)
    assert any(os.scandir(jax_cache_dir))  # the NUTS program compiles for > 1 s


# ------------------------------------------------------------ speed option 2b: warm start
def _three(n, d, seed):
    x, y0, v = _synthetic(n, d, seed)
    g = torch.Generator().manual_seed(seed + 100)
    y = torch.cat([y0, 3 * x[:, 1:2] + x[:, :1], torch.cos(3 * x[:, :1]) + x[:, 1:2]], dim=-1)
    y[:, 1:] += 0.1 * torch.randn(n, 2, generator=g, dtype=torch.double)
    return x, y, v.expand(n, 3).clone()


def test_warm_start_matches_cold_512(tmp_path):
    """Previous round (18 points, cold) -> saved state -> warm fit on 20 points (warmup 128,
    256/16) vs a cold 512/256/16 fit on the same 20 points: the same tolerance as the Pyro test."""
    x, y, v = _three(20, 4, 0)
    xt = _synthetic(10, 4, 99)[0]
    prev = nps.saas_multi_numpyro(x[:18], y[:18], v[:18], seed=1, warmup=128, samples=64, thinning=4)
    path = nps.save_warm_state(tmp_path / "warm.json", prev, meta=META)
    state = nps.load_warm_state(path, expect={**META, "n": 20})
    assert set(state) == set(OBJ)
    warm = nps.saas_multi_numpyro(x, y, v, seed=2, warmup=512, samples=256, thinning=16, warm_start=state)
    cold = nps.saas_multi_numpyro(x, y, v, seed=3, warmup=512, samples=256, thinning=16)
    assert all(m.numpyro_info["warm"] and m.numpyro_info["nuts"] == [128, 256, 16] for m in warm)
    assert not any(m.numpyro_info["warm"] for m in cold)
    for i in range(3):
        ok, stats = _agree(warm[i], cold[i], xt)
        assert ok, (i, stats)


def test_warm_start_invalid_falls_back_to_cold(tmp_path, world):
    codec, obs, objective, _, _, _ = world
    assert nps.load_warm_state(tmp_path / "missing.json", expect={**META, "n": 12}) is None
    (tmp_path / "bad.json").write_text("{not json")
    assert nps.load_warm_state(tmp_path / "bad.json", expect={**META, "n": 12}) is None
    x, y, v = _three(12, 3, 4)
    prev = nps.saas_multi_numpyro(x, y, v, seed=0, **SMALL)
    st = nps.warm_state(prev)
    m = nps.fit_saas_numpyro(codec, obs, objective, seed=0, warm_start=st, **SMALL)  # d=4 != 3
    assert [mm.numpyro_info["warm"] for mm in m.models] == [False] * 3
    assert "mismatch" in m.models[0].numpyro_info["warm_reason"]
    ok = nps.fit_saas_numpyro(codec, obs, objective, seed=0, **SMALL)
    again = nps.fit_saas_numpyro(codec, obs, objective, seed=1, warm_start=nps.warm_state(ok), **SMALL)
    assert all(mm.numpyro_info["warm"] for mm in again.models)
    with pytest.raises(ValueError, match="parallel"):
        nps.fit_saas_numpyro(codec, obs, objective, parallel=False, warm_start=nps.warm_state(ok), **SMALL)


# ------------------------------------------------------------ fix round 1: provenance
@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    x, y, v = _three(12, 3, 4)
    prev = nps.saas_multi_numpyro(x, y, v, seed=0, **SMALL)
    path = nps.save_warm_state(tmp_path_factory.mktemp("warm") / "w.json", prev, meta=META)
    return prev, path


def test_save_warm_state_requires_meta(saved, tmp_path):
    prev, _ = saved
    with pytest.raises(TypeError):
        nps.save_warm_state(tmp_path / "x.json", prev)
    with pytest.raises(ValueError, match="run_id"):
        nps.save_warm_state(tmp_path / "x.json", prev, meta={k: v for k, v in META.items() if k != "run_id"})


def test_load_warm_state_checks_provenance(saved, tmp_path):
    _, path = saved
    with pytest.raises(TypeError):
        nps.load_warm_state(path)  # expect is required
    assert set(nps.load_warm_state(path, expect={**META, "n": 12})) == set(OBJ)
    assert set(nps.load_warm_state(path, expect={**META, "n": 16})) == set(OBJ)
    for key in nps.PROVENANCE_KEYS:
        assert nps.load_warm_state(path, expect={**META, key: "other", "n": 12}) is None, key
    assert nps.load_warm_state(path, expect={**META, "n": 11}) is None  # n_prev > n
    doc = json.loads(path.read_text())
    doc["objectives"]["quality"]["objective"] = "tokens"
    bad = tmp_path / "swapped.json"
    bad.write_text(json.dumps(doc))
    assert nps.load_warm_state(bad, expect={**META, "n": 12}) is None
    doc = json.loads(path.read_text())
    doc["version"] = "old"
    (tmp_path / "v.json").write_text(json.dumps(doc))
    assert nps.load_warm_state(tmp_path / "v.json", expect={**META, "n": 12}) is None


def test_warm_meta_and_hashes(world):
    codec, obs, objective, _, _, _ = world
    meta = nps.warm_meta(codec, objective, obs, run_id="r1")
    assert meta == {"checkpoint_sha256": "x", "affine_sha256": nps.affine_sha256(codec.affine),
                    "objective_sha256": nps.objective_sha256(objective), "run_id": "r1",
                    "panel_identity": obs[0].panel_identity}
    other = ObjectiveProtocol(objective.lo, tuple(s * 2 for s in objective.span))
    assert nps.objective_sha256(other) != nps.objective_sha256(objective)
    assert len(nps.objective_sha256(objective)) == 64


# ------------------------------------------------------------ fix round 1: mass-matrix aging
def test_aging_rule_forces_cold(saved):
    prev, _ = saved
    st = nps.warm_state(prev)
    assert all(s["generation"] == 0 and s["cold_fit_n"] == 12 and s["n"] == 12 for s in st.values())
    x, y, v = _three(12, 3, 4)
    w1 = nps.saas_multi_numpyro(x, y, v, seed=1, warm_start=st, **SMALL)
    s1 = nps.warm_state(w1)
    assert all(m.numpyro_info["warm"] for m in w1)
    assert all(s["generation"] == 1 and s["cold_fit_n"] == 12 for s in s1.values())
    old = {k: {**s, "generation": nps.MAX_WARM_GENERATIONS} for k, s in st.items()}
    r = nps.saas_multi_numpyro(x, y, v, seed=2, warm_start=old, **SMALL)
    assert not r[0].numpyro_info["warm"] and "generation" in r[0].numpyro_info["warm_reason"]
    assert nps.warm_state(r)["quality"]["generation"] == 0
    x2, y2, v2 = _three(19, 3, 4)  # 19 > 1.5 * 12
    r = nps.saas_multi_numpyro(x2, y2, v2, seed=3, warm_start=st, **SMALL)
    assert not r[0].numpyro_info["warm"] and "cold_fit_n" in r[0].numpyro_info["warm_reason"]
    assert nps.warm_state(r)["quality"]["cold_fit_n"] == 19


def test_warm_chain_10_to_50_matches_cold():
    """4-round chain after a cold n=10 fit: n = 20, 30, 40, 50, each round warm-started from the
    previous round. Aging rule -> pattern cold, cold, warm, cold, warm. Every round agrees with an
    independent cold 512/256/16 fit on the same data (the Pyro-test tolerance)."""
    x, y, v = _three(50, 4, 7)
    xt = _synthetic(10, 4, 99)[0]
    state, pattern = None, []
    for k, n in enumerate((10, 20, 30, 40, 50)):
        got = nps.saas_multi_numpyro(x[:n], y[:n], v[:n], seed=10 + k, warmup=512, samples=256, thinning=16,
                                     warm_start=state)
        pattern.append(got[0].numpyro_info["warm"])
        state = nps.warm_state(got)
        if got[0].numpyro_info["warm"]:
            cold = nps.saas_multi_numpyro(x[:n], y[:n], v[:n], seed=50 + k, warmup=512, samples=256, thinning=16)
            for i in range(3):
                ok, stats = _agree(got[i], cold[i], xt)
                assert ok, (n, i, stats)
    assert pattern == [False, False, True, False, True]


# ------------------------------------------------------------ fix round 1: fallbacks
def _mutate(st, **kw):
    return {k: {**s, **kw} for k, s in st.items()}


@pytest.mark.parametrize("case", ["non_pd", "step0", "nonfinite", "shape", "missing_objective"])
def test_invalid_warm_state_falls_back(saved, case):
    prev, _ = saved
    st = nps.warm_state(prev)
    dim = len(st["quality"]["inverse_mass_matrix"])
    if case == "non_pd":
        bad = _mutate(st, inverse_mass_matrix=(-np.eye(dim)).tolist())
    elif case == "step0":
        bad = _mutate(st, step_size=0.0)
    elif case == "nonfinite":
        bad = _mutate(st, inverse_mass_matrix=np.full((dim, dim), np.nan).tolist())
    elif case == "shape":
        bad = {k: {**s, "constrained": {**s["constrained"], "_kernel_inv_length_sq": [1.0, 2.0]}} for k, s in st.items()}
    else:
        bad = {k: s for k, s in st.items() if k != "latency"}
    x, y, v = _three(12, 3, 4)
    r = nps.saas_multi_numpyro(x, y, v, seed=0, warm_start=bad, **SMALL)
    assert not any(m.numpyro_info["warm"] for m in r), case


# ------------------------------------------------------------ fix round 1: fit health
def test_init_failure_raises():
    Xp, _, Vp, M = nps.pad_inputs(np.random.default_rng(0).random((9, 2)), np.zeros(9), np.full(9, 0.01), 16)
    Ys = np.stack([np.full(16, np.nan)] * 3)
    with pytest.raises(RuntimeError, match="initial"):
        nps.run_vmapped(Xp, Ys, np.stack([Vp] * 3), M, seeds=[0, 1, 2], warmup=8, samples=4)


def test_nonfinite_samples_raise():
    with pytest.raises(RuntimeError, match="non-finite"):
        nps.check_finite_samples({"mean": np.array([0.0, np.nan]), "outputscale": np.ones(2)}, "quality")


def test_result_records_divergences(world):
    codec, obs, objective, _, _, _ = world
    gp = nps.fit_saas_numpyro(codec, obs, objective, seed=0, **SMALL)
    summary = gp.numpyro_summary
    assert set(summary["divergences"]) == set(OBJ)
    assert all(isinstance(d, int) and d >= 0 for d in summary["divergences"].values())
    assert summary["mode"] == "vmap" and summary["warm"] is False
    assert summary["init_tries"] == {o: -1 for o in OBJ}  # -1 = valid initial point found


# ------------------------------------------------------------ fix round 1: CPU pinning API
def _run(code, env=None):
    e = {**os.environ, "PYTHONPATH": REPO, "TRIMEM_JAX_CACHE_DIR": "off"}
    e.pop("TRIMEM_JAX_CPUS", None)
    e.update(env or {})
    return subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300,
                          cwd=REPO, check=False)


def test_pin_cpus_api_subprocess():
    code = (
        "import os, json\n"
        "from trimem.optimizer import latent_numpyro_saas as n\n"
        "allowed = sorted(os.sched_getaffinity(0))\n"
        "assert n.PINNED_CPUS is None\n"
        "got = n.pin_cpus(2, offset=1)\n"
        "tasks = {tuple(sorted(os.sched_getaffinity(int(t)))) for t in os.listdir('/proc/self/task')}\n"
        "one = n.pin_cpus([got[0]])\n"
        "err = ''\n"
        "try:\n    n.pin_cpus([max(allowed) + 1000])\nexcept ValueError as e:\n    err = str(e)\n"
        "print(json.dumps({'allowed': allowed, 'got': got, 'tasks': [list(t) for t in tasks], 'one': one, 'err': err}))\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["got"] == out["allowed"][1:3] and out["tasks"] == [out["got"]]
    assert out["one"] == [out["got"][0]] and "not allowed" in out["err"]


def test_env_pinning_opt_in_and_bad_value_raises_at_use():
    r = _run("from trimem.optimizer import latent_numpyro_saas as n; print(n.PINNED_CPUS)", {"TRIMEM_JAX_CPUS": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert len(eval(r.stdout.strip().splitlines()[-1])) == 1
    code = (
        "import torch\n"
        "from trimem.optimizer import latent_numpyro_saas as n\n"
        "assert n.PINNED_CPUS is None\n"
        "x = torch.rand(6, 2, dtype=torch.double)\n"
        "try:\n    n.saas_multi_numpyro(x, torch.rand(6, 3, dtype=torch.double), torch.full((6, 3), 0.01, dtype=torch.double), warmup=4, samples=2, thinning=1)\n"
        "except ValueError as e:\n    print('ERR', e)\n"
    )
    r = _run(code, {"TRIMEM_JAX_CPUS": "four"})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ERR" in r.stdout and "TRIMEM_JAX_CPUS" in r.stdout
