"""Latent-space GP (three SingleTaskGP on u), gradient acquisition and measured HV."""

import torch
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models import ModelListGP, MultiTaskGP, SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.transforms.outcome import StratifiedStandardize, Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.multi_objective.hypervolume import Hypervolume
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from .plan_surrogate import AcquisitionSnapshot


OBJECTIVE_NAMES = ("quality", "tokens", "latency")
STRUCTURES = ("independent", "coupled-cost")
KERNELS = ("saas", "matern", "saas-numpyro")


def fit_latent_model(
    codec,
    observations,
    objective,
    *,
    seed=0,
    maxiter=150,
    timeout=120,
    kernel="saas",
    structure="independent",
    nuts=(512, 256, 16),
    on_objective=None,
    nuts_cost=None,
    warm_start=None,
    warm_warmup=128,
):
    """nuts_cost: NUTS settings for the joint cost fit only (coupled-cost + saas); nuts governs
    quality and the independent path. Both must retain the same number of MCMC samples
    (num_samples // thinning) because ModelListGP stacks them. Default None derives
    (min(warmup,192), min(samples,96), thinning matching quality's retained count):
    (192, 96, 6) for the default nuts.
    structure="independent": three independent GPs on u.
    structure="coupled-cost": quality GP + ONE joint ICM two-output GP over (tokens, latency);
    on_objective is then called as (0,"quality") and (1,"cost"). Output order is always
    (quality, tokens, latency).

    Independent structure: kernel="saas" (default): fully Bayesian SAAS fitted by NUTS
    (warmup, samples, thinning = nuts). kernel="matern": Matern-5/2 ARD, MAP fit (Task 3 path).
    on_objective(i, name) is called immediately before fitting objective i.

    kernel="saas-numpyro" (live default, user 2026-11-04): the same SAAS model fitted by NumPyro/JAX
    NUTS (latent_numpyro_saas.fit_saas_numpyro): the three objectives in ONE vmapped program
    (parallel=True; on_objective fires for all three immediately before that fit), warmup/samples/
    thinning = nuts, seed + i per objective. warm_start = {objective: state} from
    latent_numpyro_saas.load_warm_state (or None); a valid state fits with warmup=warm_warmup.
    Structure "independent" only. The ModelListGP of SaasFullyBayesianSingleTaskGP is consumed
    exactly like the Pyro SAAS model (latent_snapshot: mixture moments for posterior consumers,
    acquisition on the full fully Bayesian model). Its ``numpyro_summary`` records the fit."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure {structure!r}; expected one of {STRUCTURES}")
    if kernel == "saas-numpyro":
        if structure != "independent":
            raise ValueError("kernel 'saas-numpyro' supports structure 'independent' only")
        # imported here: the JAX import stays out of the Matern/Pyro paths, and the live worker
        # pins its CPUs before the first JAX use (trace_latent_search.pin_surrogate_cpus)
        from . import latent_numpyro_saas as nps

        warmup, samples, thinning = nuts
        return nps.fit_saas_numpyro(
            codec, observations, objective, seed=seed, warmup=warmup, samples=samples,
            thinning=thinning, parallel=True, warm_start=warm_start, warm_warmup=warm_warmup,
            on_objective=on_objective,
        )
    if warm_start is not None:
        raise ValueError("warm_start is supported only by kernel 'saas-numpyro'")
    if len({o.panel_identity for o in observations}) != 1:
        raise ValueError("GP observations must share the exact TRAIN panel and protocol")
    if len({codec.key(o.config) for o in observations}) != len(observations):
        raise ValueError("Repeated complete config observations")
    x = torch.tensor([codec.encode(o.config) for o in observations], dtype=torch.double)
    y, noise = objective.tensors(observations)
    if structure == "coupled-cost":
        nuts_cost = _resolve_nuts_cost(nuts, nuts_cost)
        return _fit_coupled(
            x, y, noise, seed, nuts, kernel, maxiter, timeout, on_objective, nuts_cost
        )
    if kernel == "saas":
        return _fit_saas(x, y, noise, seed, nuts, on_objective)
    if on_objective is not None:
        on_objective(0, OBJECTIVE_NAMES[0])
    models = []
    for i in range(3):
        covar = ScaleKernel(
            get_covar_module_with_dim_scaled_prior(ard_num_dims=codec.d, use_rbf_kernel=False)
        )
        covar.register_prior("outputscale_prior", GammaPrior(2.0, 1.0), "outputscale")
        models.append(
            SingleTaskGP(
                x,
                y[:, i : i + 1],
                train_Yvar=noise[:, i : i + 1],
                covar_module=covar,
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fit_gpytorch_mll(
            SumMarginalLogLikelihood(model.likelihood, model),
            optimizer_kwargs={"options": {"maxiter": maxiter}, "timeout_sec": timeout},
        )
    model.eval()
    return model


class _TaskStandardize(StratifiedStandardize):
    """Per-task standardization for the 2-output cost pair. StratifiedStandardize cannot
    un-transform a multi-output posterior (it looks up the task column in X, which a
    2-output MultiTaskGP posterior does not pass), so per-input stats are replaced by per
    output column stats (output j == task j, as output_tasks=[0, 1])."""

    _posterior_mode = False

    def _get_per_input_means_stdvs(self, X, include_stdvs_sq):
        if not self._posterior_mode:
            return super()._get_per_input_means_stdvs(X, include_stdvs_sq)
        mu, sd = self.means.reshape(1, -1), self.stdvs.reshape(1, -1)
        return mu, sd, (sd.square() if include_stdvs_sq else None)

    def untransform_posterior(self, posterior, X=None):
        self._posterior_mode = True
        try:
            return Standardize.untransform_posterior(self, posterior, X=X)
        finally:
            self._posterior_mode = False


def _long_cost_data(x, y, noise):
    """Long format for the cost pair: rows [x, task] with task 0 = tokens, 1 = latency."""
    n = x.shape[0]
    tasks = torch.cat([torch.zeros(n, 1), torch.ones(n, 1)]).to(x)
    lx = torch.cat([torch.cat([x, x]), tasks], dim=-1)
    ly = torch.cat([y[:, 1:2], y[:, 2:3]])
    lv = torch.cat([noise[:, 1:2], noise[:, 2:3]])
    return lx, ly, lv


def _resolve_nuts_cost(nuts, nuts_cost):
    kept = nuts[1] // nuts[2]
    if nuts_cost is None:
        n_s = min(nuts[1], 96)
        return (min(nuts[0], 192), n_s, max(1, n_s // kept))
    if nuts_cost[1] // nuts_cost[2] != kept:
        raise ValueError(
            f"nuts_cost keeps {nuts_cost[1] // nuts_cost[2]} MCMC samples but nuts keeps {kept}"
        )
    return tuple(nuts_cost)


def _fit_coupled(x, y, noise, seed, nuts, kernel, maxiter, timeout, on_objective, nuts_cost):
    d = x.shape[-1]
    if on_objective is not None:
        on_objective(0, "quality")
    if kernel == "saas":
        quality = _saas_single(x, y[:, 0:1], noise[:, 0:1], seed, nuts)
    else:
        quality = _matern_single(x, y[:, 0:1], noise[:, 0:1], d, seed, maxiter, timeout)
    if on_objective is not None:
        on_objective(1, "cost")
    lx, ly, lv = _long_cost_data(x, y, noise)
    # Per-task standardization: tokens and latency live on different scales.
    strat = _TaskStandardize(
        stratification_idx=d, all_task_values=torch.tensor([0.0, 1.0], dtype=torch.double)
    )
    if kernel == "saas":
        cost = SaasFullyBayesianMultiTaskGP(
            lx, ly, task_feature=d, train_Yvar=lv, output_tasks=[0, 1], outcome_transform=strat
        )
        warmup, samples, thinning = nuts_cost
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            fit_fully_bayesian_model_nuts(
                cost, warmup_steps=warmup, num_samples=samples, thinning=thinning, disable_progbar=True
            )
    else:
        cost = MultiTaskGP(
            lx, ly, task_feature=d, train_Yvar=lv, output_tasks=[0, 1], outcome_transform=strat
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(cost.likelihood, cost),
                optimizer_kwargs={"options": {"maxiter": maxiter}, "timeout_sec": timeout},
            )
    model = ModelListGP(quality, cost)
    model.eval()
    return model


def _saas_single(x, y1, v1, seed, nuts):
    warmup, samples, thinning = nuts
    m = SaasFullyBayesianSingleTaskGP(
        train_X=x, train_Y=y1, train_Yvar=v1, outcome_transform=Standardize(m=1)
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fit_fully_bayesian_model_nuts(
            m, warmup_steps=warmup, num_samples=samples, thinning=thinning, disable_progbar=True
        )
    return m


def _matern_single(x, y1, v1, d, seed, maxiter, timeout):
    covar = ScaleKernel(get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel=False))
    covar.register_prior("outputscale_prior", GammaPrior(2.0, 1.0), "outputscale")
    m = SingleTaskGP(x, y1, train_Yvar=v1, covar_module=covar, outcome_transform=Standardize(m=1))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(m.likelihood, m),
            optimizer_kwargs={"options": {"maxiter": maxiter}, "timeout_sec": timeout},
        )
    return m


def surrogate_info(model):
    """Structure, kernel family, member classes and the learned tokens-latency task correlation
    (ICM task covariance; mean over MCMC samples when fully Bayesian)."""
    classes = [type(m).__name__ for m in model.models]
    cost = next((m for m in model.models if isinstance(m, MultiTaskGP)), None)
    coupled = cost is not None
    fully_bayesian = any(getattr(m, "_is_fully_bayesian", False) for m in model.models)
    info = {
        "structure": "coupled-cost" if coupled else "independent",
        # fit_saas_numpyro tags its ModelListGP with numpyro_summary
        "kernel": ("saas-numpyro" if hasattr(model, "numpyro_summary") else "saas")
        if fully_bayesian else "matern",
        "classes": classes,
        "cost_task_correlation": None,
    }
    if coupled:
        with torch.no_grad():
            k = cost.covar_module.kernels[1]
            cov = k.covar_matrix.detach()
            corr = cov[..., 0, 1] / (cov[..., 0, 0] * cov[..., 1, 1]).sqrt()
            info["cost_task_correlation"] = float(corr.mean())
    return info


def _fit_saas(x, y, noise, seed, nuts, on_objective):
    warmup, samples, thinning = nuts
    models = []
    for i, name in enumerate(OBJECTIVE_NAMES):
        if on_objective is not None:
            on_objective(i, name)
        m = SaasFullyBayesianSingleTaskGP(
            train_X=x,
            train_Y=y[:, i : i + 1],
            train_Yvar=noise[:, i : i + 1],
            outcome_transform=Standardize(m=1),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + i)
            fit_fully_bayesian_model_nuts(
                m, warmup_steps=warmup, num_samples=samples, thinning=thinning, disable_progbar=True
            )
        models.append(m)
    model = ModelListGP(*models)
    model.eval()
    return model


def _fully_bayesian(model) -> bool:
    members = getattr(model, "models", None) or [model]
    return any(getattr(m, "_is_fully_bayesian", False) for m in members)


class LatentSnapshot(AcquisitionSnapshot):
    """AcquisitionSnapshot whose non-acquisition posterior is MCMC-mixture aware.

    A fully Bayesian (SAAS) posterior is batched over MCMC samples, [S, n, 3]. Consumers of
    ``posterior()`` (e.g. the hashed ``joint_search_controller.frontier_parents``) expect
    [n, 3], so ``posterior()`` returns the mixture mean E_s[mu_s] and ``posterior_moments()``
    adds the mixture variance E_s[var_s] + Var_s[mu_s]. The qLogNEHVI acquisition (scores,
    gradients) keeps the full fully Bayesian model; BoTorch averages over MCMC samples there.
    For a non-fully-Bayesian model both methods return the plain posterior moments. The
    snapshot identity is unchanged (the class is not part of it)."""

    def posterior_moments(self, configs):
        x = torch.tensor([self.codec.encode(c) for c in configs], dtype=torch.double)
        with torch.no_grad():
            post = self.model.posterior(x)
            mean, var = post.mean, post.variance
        if _fully_bayesian(self.model) and mean.dim() == 3:  # [S, n, m] MCMC batch
            return mean.mean(0), var.mean(0) + mean.var(0, unbiased=False)
        return mean, var

    def posterior(self, configs):
        return self.posterior_moments(configs)[0]

    def with_pending(self, pending):
        return type(self)(
            self.codec, self.model, self.observations, self.objective, seed=self.seed,
            mc_samples=self.mc_samples, pending=pending, budget=self.budget,
        )


def latent_snapshot(codec, model, observations, objective, *, seed, pending=()):
    """Pending-aware qLogNEHVI snapshot (64 Sobol QMC samples, budget=None)."""
    return LatentSnapshot(
        codec, model, observations, objective, seed=seed, mc_samples=64, pending=pending, budget=None
    )


def acquisition_u(snapshot, U):
    """ell(u) for U of shape [N, d]; gradient-enabled (flows to U only through the GP)."""
    return snapshot.acquisition(U.unsqueeze(-2))


def measured_hv(observations, objective):
    """HV (y_norm units) of observed means strictly above ref in every objective."""
    if not observations:
        return 0.0
    y, _ = objective.tensors(observations)
    ref = torch.tensor(objective.ref, dtype=torch.double)
    y = y[(y > ref).all(dim=-1)]
    if y.shape[0] == 0:
        return 0.0
    return float(Hypervolume(ref_point=ref).compute(y))
