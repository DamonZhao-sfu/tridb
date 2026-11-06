import pytest
import warnings
warnings.filterwarnings("ignore", message=".*cache_root.*")
import torch

from tests.latent_fixtures import enumerate_space, synthetic_receipt, tiny_model, tiny_space
from trimem.optimizer.latent_plan_codec import LatentCodec, fit_affine
from trimem.optimizer.latent_plan_surrogate import (
    acquisition_u,
    fit_latent_model,
    latent_snapshot,
    measured_hv,
    surrogate_info,
)
from trimem.optimizer.latent_seq_grammar import SeqGrammar
from trimem.optimizer.plan_surrogate import (
    Observation,
    ObjectiveProtocol,
    aggregate_panel,
)

torch.set_num_threads(4)
TASKS = ["t0", "t1", "t2", "t3"]
MANIFEST = {"protocol": "p", "evaluation_split": "TRAIN"}


@pytest.fixture(scope="module")
def world():
    space, model = tiny_space(), tiny_model(4)
    configs = enumerate_space(space)
    codec = LatentCodec(
        space, model, fit_affine(model, SeqGrammar(space), configs), checkpoint_sha256="x"
    )
    picks = configs[:: len(configs) // 12][:12]
    obs = [
        aggregate_panel(
            space,
            c,
            [synthetic_receipt(c, t, MANIFEST) for t in TASKS],
            task_ids=TASKS,
            protocol="p",
            allow_synthetic=True,
        )
        for c in picks
    ]
    objective = ObjectiveProtocol.from_initial(obs)
    gp = fit_latent_model(codec, obs, objective, seed=0, maxiter=60, timeout=60, kernel="matern")
    return codec, obs, objective, gp, configs


def _obs(y):
    return Observation({}, (0,) * 3, tuple(y), (1e-4,) * 3, "p", True)


def _unit():
    return ObjectiveProtocol((0.0,) * 3, (1.0,) * 3)


def test_hv_hand_computed():
    hv = measured_hv([_obs((1, 0, 0)), _obs((0, 1, 0))], _unit())
    assert hv == pytest.approx(0.021, abs=1e-12)


def test_point_below_ref_adds_zero_hv():
    base = [_obs((1, 0, 0)), _obs((0, 1, 0))]
    assert measured_hv(base + [_obs((-0.5, 2, 2))], _unit()) == pytest.approx(
        measured_hv(base, _unit()), abs=1e-12
    )
    assert measured_hv([_obs((-0.5, 2, 2))], _unit()) == 0.0


def test_posterior_in_ynorm_units(world):
    codec, obs, objective, gp, _ = world
    y, _ = objective.tensors(obs)
    U = codec.encode_many([o.config for o in obs])
    with torch.no_grad():
        mean = gp.posterior(U).mean
    assert (mean - y).abs().max() < 0.05


def test_yvar_is_scaled_by_span_squared(world):
    codec, obs, objective, gp, _ = world
    _, v = objective.tensors(obs)
    for i, m in enumerate(gp.models):
        # GP-internal noise is in Standardize units: noise * std^2 == v/span^2, floored at 1e-6 by gpytorch.
        std2 = m.outcome_transform.stdvs.square().reshape(())
        expect = torch.clamp(v[:, i] / std2, min=1e-6)
        assert torch.allclose(m.likelihood.noise.detach().reshape(-1), expect, rtol=1e-6)
        unfloored = v[:, i] / std2 > 1e-6
        assert torch.allclose(
            (m.likelihood.noise.detach().reshape(-1) * std2)[unfloored], v[unfloored, i], rtol=1e-6
        )
        assert m.train_inputs[0].shape == (len(obs), codec.d)


def test_gradient_matches_finite_difference(world):
    codec, obs, objective, gp, configs = world
    snap = latent_snapshot(codec, gp, obs, objective, seed=3)
    U = codec.encode_many(configs[:40])
    u0 = (0.3 * U[2] + 0.7 * U[9]).clone()  # smooth interior point (0.5*U0+0.5*U17 sits on a kink)
    x = u0.clone().requires_grad_(True)
    val = acquisition_u(snap, x.unsqueeze(0))
    assert val.shape == (1,)
    val.sum().backward()
    g = x.grad
    assert torch.isfinite(g).all() and g.abs().max() > 0
    eps = 1e-4
    fd = torch.zeros_like(u0)
    with torch.no_grad():
        for j in range(codec.d):
            e = torch.zeros_like(u0)
            e[j] = eps
            fd[j] = (
                acquisition_u(snap, (u0 + e).unsqueeze(0)) - acquisition_u(snap, (u0 - e).unsqueeze(0))
            ).item() / (2 * eps)
    assert torch.allclose(g, fd, rtol=2e-2, atol=1e-6)


def test_pending_lowers_duplicate_score(world):
    codec, obs, objective, gp, configs = world
    seen = {codec.key(o.config) for o in obs}
    s = latent_snapshot(codec, gp, obs, objective, seed=1)
    cands = [c for c in configs if codec.key(c) not in seen]
    base = s.score(cands[:60])
    c = cands[max(range(60), key=lambda i: base[i])]
    assert s.with_pending([c]).score([c])[0] < s.score([c])[0]


def test_mixed_panels_rejected(world):
    codec, obs, objective, _, _ = world
    bad = [obs[0], Observation(**{**obs[1].__dict__, "panel_identity": "other"}), *obs[2:]]
    with pytest.raises(ValueError):
        fit_latent_model(codec, bad, objective)
    with pytest.raises(ValueError):
        fit_latent_model(codec, [obs[0], obs[0]], objective)


NUTS = (32, 16, 4)


@pytest.fixture(scope="module")
def saas_world(world):
    codec, obs, objective, _, configs = world
    gp = fit_latent_model(codec, obs, objective, seed=0, nuts=NUTS)
    return codec, obs, objective, gp, configs


def test_saas_is_default(saas_world):
    from botorch.models import ModelListGP, SaasFullyBayesianSingleTaskGP

    gp = saas_world[3]
    assert isinstance(gp, ModelListGP) and len(gp.models) == 3
    assert all(isinstance(m, SaasFullyBayesianSingleTaskGP) for m in gp.models)


def test_saas_snapshot_scores_and_pending(saas_world):
    codec, obs, objective, gp, configs = saas_world
    seen = {codec.key(o.config) for o in obs}
    s = latent_snapshot(codec, gp, obs, objective, seed=1)
    cands = [c for c in configs if codec.key(c) not in seen][:60]
    base = s.score(cands)
    assert all(torch.isfinite(torch.as_tensor(float(b))) for b in base)
    c = cands[max(range(len(cands)), key=lambda i: base[i])]
    assert s.with_pending([c]).score([c])[0] < s.score([c])[0]


def test_saas_acquisition_gradient_finite(saas_world):
    codec, obs, objective, gp, configs = saas_world
    snap = latent_snapshot(codec, gp, obs, objective, seed=3)
    U = codec.encode_many(configs[:6]).clone().requires_grad_(True)
    acquisition_u(snap, U).sum().backward()
    assert torch.isfinite(U.grad).all() and U.grad.abs().max() > 0


def test_saas_seed_deterministic(world):
    codec, obs, objective, _, _ = world
    a = fit_latent_model(codec, obs, objective, seed=7, nuts=NUTS)
    b = fit_latent_model(codec, obs, objective, seed=7, nuts=NUTS)
    U = codec.encode_many([o.config for o in obs[:5]])
    with torch.no_grad():
        assert (a.posterior(U).mean - b.posterior(U).mean).abs().max() <= 1e-10


def test_on_objective_called_in_order(world):
    codec, obs, objective, _, _ = world
    calls = []
    fit_latent_model(codec, obs, objective, nuts=(8, 4, 2), on_objective=lambda i, n: calls.append((i, n)))
    assert calls == [(0, "quality"), (1, "tokens"), (2, "latency")]


def test_unknown_kernel_raises(world):
    codec, obs, objective, _, _ = world
    with pytest.raises(ValueError):
        fit_latent_model(codec, obs, objective, kernel="rbf")


@pytest.fixture(scope="module")
def coupled_world(world):
    codec, obs, objective, _, configs = world
    gp = fit_latent_model(codec, obs, objective, seed=0, structure="coupled-cost", nuts=NUTS)
    return codec, obs, objective, gp, configs


def test_coupled_posterior_has_three_outputs(coupled_world):
    codec, obs, objective, gp, _ = coupled_world
    y, _ = objective.tensors(obs)
    U = codec.encode_many([o.config for o in obs])
    with torch.no_grad():
        mean = gp.posterior(U).mean
    assert mean.shape[-2:] == (len(obs), 3)
    # order (quality, tokens, latency): each column tracks its own objective
    for j in range(3):
        assert (mean[..., j].mean(0) - y[:, j]).abs().max() < 0.15


def test_coupled_snapshot_scores_pending_gradient(coupled_world):
    codec, obs, objective, gp, configs = coupled_world
    seen = {codec.key(o.config) for o in obs}
    s = latent_snapshot(codec, gp, obs, objective, seed=1)
    cands = [c for c in configs if codec.key(c) not in seen][:60]
    base = s.score(cands)
    assert all(torch.isfinite(torch.as_tensor(float(b))) for b in base)
    c = cands[max(range(len(cands)), key=lambda i: base[i])]
    assert s.with_pending([c]).score([c])[0] < s.score([c])[0]
    U = codec.encode_many(configs[:6]).clone().requires_grad_(True)
    acquisition_u(latent_snapshot(codec, gp, obs, objective, seed=3), U).sum().backward()
    assert torch.isfinite(U.grad).all() and U.grad.abs().max() > 0


def test_coupled_learns_positive_cost_correlation(coupled_world):
    info = surrogate_info(coupled_world[3])
    assert info["structure"] == "coupled-cost" and info["kernel"] == "saas"
    assert info["cost_task_correlation"] > 0
    assert surrogate_info(fit_latent_model(*coupled_world[:3], kernel="matern", nuts=NUTS))[
        "cost_task_correlation"
    ] is None


def test_coupled_matern_path(world):
    codec, obs, objective, _, _ = world
    calls = []
    gp = fit_latent_model(
        codec, obs, objective, kernel="matern", structure="coupled-cost", maxiter=60, timeout=60,
        on_objective=lambda i, n: calls.append((i, n)),
    )
    assert calls == [(0, "quality"), (1, "cost")]
    info = surrogate_info(gp)
    assert info["kernel"] == "matern" and "MultiTaskGP" in info["classes"]
    assert -1 <= info["cost_task_correlation"] <= 1
    U = codec.encode_many([o.config for o in obs])
    with torch.no_grad():
        assert gp.posterior(U).mean.shape == (len(obs), 3)


def test_unknown_structure_raises(world):
    codec, obs, objective, _, _ = world
    with pytest.raises(ValueError):
        fit_latent_model(codec, obs, objective, structure="joint")


def test_coupled_unstandardization_distinct_scales(world):
    """tokens ~0.2 +- 0.05, latency ~0.8 +- 0.3 in y_norm: a shared-scale bug breaks this."""
    codec, obs, objective, _, _ = world
    n = len(obs)
    U = codec.encode_many([o.config for o in obs])
    g = torch.Generator().manual_seed(0)
    z = torch.randn(n, generator=g, dtype=torch.double)
    z = (z - z.mean()) / z.std()
    y = torch.stack([0.5 + 0.2 * U[:, 0], 0.2 + 0.05 * z, 0.8 + 0.3 * z], 1)
    noise = torch.full((n, 3), 1e-6, dtype=torch.double)
    from trimem.optimizer.latent_plan_surrogate import _fit_coupled

    for kern in ("matern", "saas"):
        gp = _fit_coupled(U, y, noise, 0, NUTS, kern, 150, 120, None, NUTS)
        with torch.no_grad():
            mean = gp.posterior(U).mean
        mean = mean.mean(0) if mean.dim() == 3 else mean
        assert (mean - y).abs().max() < 0.05, kern
        assert (mean[:, 2].std() / mean[:, 1].std()) > 3.0, kern


def test_task_standardize_untransform_posterior_unit():
    from botorch.posteriors import GPyTorchPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal

    from trimem.optimizer.latent_plan_surrogate import _TaskStandardize

    t = _TaskStandardize(stratification_idx=1, all_task_values=torch.tensor([0.0, 1.0], dtype=torch.double))
    X = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 1.0], [0.4, 1.0]], dtype=torch.double)
    Y = torch.tensor([[0.1], [0.3], [1.0], [3.0]], dtype=torch.double)
    t(Y, X=X)  # train: per-task mean/std
    mu, sd = t.means.reshape(-1), t.stdvs.reshape(-1)
    assert torch.allclose(mu, torch.tensor([0.2, 2.0], dtype=torch.double))
    m = torch.tensor([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]], dtype=torch.double)
    cov = torch.diag_embed(torch.tensor([1.0, 4.0], dtype=torch.double).repeat(3))
    post = GPyTorchPosterior(MultitaskMultivariateNormal(m, cov, interleaved=True))
    out = t.untransform_posterior(post)
    assert torch.allclose(out.mean, mu + sd * m)
    assert torch.allclose(out.variance, sd.square() * torch.tensor([1.0, 4.0], dtype=torch.double).expand(3, 2))


def test_nuts_cost_default_matches_quality_sample_count():
    from trimem.optimizer.latent_plan_surrogate import _resolve_nuts_cost

    assert _resolve_nuts_cost((512, 256, 16), None) == (192, 96, 6)
    assert _resolve_nuts_cost((32, 16, 4), None) == (32, 16, 4)
    with pytest.raises(ValueError):
        _resolve_nuts_cost((512, 256, 16), (192, 96, 16))


# ---- Task 12 fix 1: MCMC-mixture posterior for fully Bayesian snapshots
def test_saas_snapshot_posterior_is_mixture_moments(saas_world):
    codec, obs, objective, gp, configs = saas_world
    snap = latent_snapshot(codec, gp, obs, objective, seed=1)
    cfgs = configs[:7]
    mean, var = snap.posterior_moments(cfgs)
    assert mean.shape == var.shape == (7, 3)
    assert torch.equal(snap.posterior(cfgs), mean)
    with torch.no_grad():
        p = gp.posterior(codec.encode_many(cfgs))
    assert p.mean.dim() == 3  # [S, n, 3]: the batched shape that broke frontier_parents
    mu_s, var_s = p.mean, p.variance
    assert torch.allclose(mean, mu_s.mean(0), atol=1e-12)
    manual = var_s.mean(0) + ((mu_s - mu_s.mean(0)) ** 2).mean(0)
    assert torch.allclose(var, manual, atol=1e-12) and (var > 0).all()
    pend = snap.with_pending([cfgs[0]])
    assert type(pend) is type(snap) and pend.posterior(cfgs).shape == (7, 3)


def test_saas_snapshot_feeds_frontier_parents_and_recombine(saas_world):
    from trimem.optimizer.joint_search_controller import frontier_parents, recombine

    codec, obs, objective, gp, _ = saas_world
    snap = latent_snapshot(codec, gp, obs, objective, seed=1)
    measured = [o.config for o in obs]
    parents = frontier_parents(snap, measured)  # raised ValueError before the fix
    assert 1 <= len(parents) <= 5 and all(p in measured for p in parents)
    assert recombine(codec.space, parents, seed=0)


def test_coupled_snapshot_posterior_is_mixture(coupled_world):
    codec, obs, objective, gp, configs = coupled_world
    mean, var = latent_snapshot(codec, gp, obs, objective, seed=1).posterior_moments(configs[:5])
    assert mean.shape == var.shape == (5, 3) and (var > 0).all()


def test_matern_snapshot_posterior_unchanged(world):
    from trimem.optimizer.plan_surrogate import AcquisitionSnapshot

    codec, obs, objective, gp, configs = world
    snap = latent_snapshot(codec, gp, obs, objective, seed=1)
    base = AcquisitionSnapshot(codec, gp, obs, objective, seed=1, mc_samples=64, budget=None)
    assert snap.identity == base.identity
    assert torch.equal(snap.posterior(configs[:9]), base.posterior(configs[:9]))
    with torch.no_grad():
        want = gp.posterior(codec.encode_many(configs[:9])).variance
    assert torch.equal(snap.posterior_moments(configs[:9])[1], want)
    assert snap.score(configs[:9]) == base.score(configs[:9])


# ---- live wiring (goal Step 3): kernel="saas-numpyro" -> latent_numpyro_saas.fit_saas_numpyro
NP_NUTS = (32, 16, 4)  # 4 retained MCMC samples


@pytest.fixture(scope="module")
def np_cache(tmp_path_factory):
    """Compilation cache in a temp dir (never the repo default results/.jax_cache)."""
    from trimem.optimizer import latent_numpyro_saas as nps

    previous = nps.COMPILATION_CACHE
    nps.enable_compilation_cache(tmp_path_factory.mktemp("jaxcache"))
    yield nps
    nps.enable_compilation_cache(previous)


@pytest.fixture(scope="module")
def numpyro_world(world, np_cache):
    codec, obs, objective, _, configs = world
    gp = fit_latent_model(codec, obs, objective, seed=0, kernel="saas-numpyro", nuts=NP_NUTS)
    return codec, obs, objective, gp, configs


def test_saas_numpyro_dispatches_to_fit_saas_numpyro(world, np_cache, monkeypatch):
    codec, obs, objective, _, _ = world
    calls = []

    def fake(codec_, obs_, objective_, **kw):
        calls.append((codec_, obs_, objective_, kw))
        return "MODEL"

    monkeypatch.setattr(np_cache, "fit_saas_numpyro", fake)
    hook = lambda i, n: None  # noqa: E731
    state = {"quality": {}}
    out = fit_latent_model(codec, obs, objective, seed=5, kernel="saas-numpyro", nuts=(64, 32, 8),
                           on_objective=hook, warm_start=state, warm_warmup=16)
    assert out == "MODEL" and len(calls) == 1
    c, o, ob, kw = calls[0]
    assert (c, o, ob) == (codec, obs, objective)
    assert kw == {"seed": 5, "warmup": 64, "samples": 32, "thinning": 8, "parallel": True,
                  "warm_start": state, "warm_warmup": 16, "on_objective": hook}


def test_saas_numpyro_rejects_coupled_cost_and_warm_start_elsewhere(world):
    codec, obs, objective, _, _ = world
    with pytest.raises(ValueError, match="independent"):
        fit_latent_model(codec, obs, objective, kernel="saas-numpyro", structure="coupled-cost")
    with pytest.raises(ValueError, match="warm_start"):
        fit_latent_model(codec, obs, objective, kernel="matern", warm_start={"quality": {}})


def test_saas_numpyro_snapshot_shapes_and_gradient(numpyro_world):
    from botorch.models import ModelListGP, SaasFullyBayesianSingleTaskGP

    codec, obs, objective, gp, configs = numpyro_world
    assert isinstance(gp, ModelListGP) and len(gp.models) == 3
    assert all(isinstance(m, SaasFullyBayesianSingleTaskGP) for m in gp.models)
    assert gp.numpyro_summary["mode"] == "vmap" and gp.numpyro_summary["warm"] is False
    snap = latent_snapshot(codec, gp, obs, objective, seed=1)
    post = snap.posterior(configs[:7])
    assert post.shape == (7, 3)  # mixture moments for posterior consumers
    mean, var = snap.posterior_moments([o.config for o in obs])
    assert mean.shape == var.shape == (len(obs), 3) and (var > 0).all()
    with torch.no_grad():
        assert gp.posterior(codec.encode_many(configs[:7])).mean.dim() == 3  # full FB model kept
    U = codec.encode_many(configs[:6]).clone().requires_grad_(True)
    acquisition_u(snap, U).sum().backward()
    assert torch.isfinite(U.grad).all() and U.grad.abs().max() > 0
    info = surrogate_info(gp)
    assert info["kernel"] == "saas-numpyro" and info["structure"] == "independent"


def test_saas_numpyro_same_seed_is_bit_identical(numpyro_world):
    codec, obs, objective, gp, _ = numpyro_world
    again = fit_latent_model(codec, obs, objective, seed=0, kernel="saas-numpyro", nuts=NP_NUTS)
    a = latent_snapshot(codec, gp, obs, objective, seed=1)
    b = latent_snapshot(codec, again, obs, objective, seed=1)
    assert a.identity == b.identity  # identity hashes every MCMC sample in the state dict


def test_saas_numpyro_warm_start_through_fit_latent_model(numpyro_world, np_cache):
    codec, obs, objective, gp, _ = numpyro_world
    state = np_cache.warm_state(gp)
    warm = fit_latent_model(codec, obs, objective, seed=0, kernel="saas-numpyro", nuts=NP_NUTS,
                            warm_start=state, warm_warmup=8)
    s = warm.numpyro_summary
    assert s["warm"] is True and s["nuts"] == [8, 16, 4]
    assert all(m.numpyro_info["state"]["generation"] == 1 for m in warm.models)
