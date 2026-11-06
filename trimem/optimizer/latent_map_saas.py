"""MAP-SAAS latent GP (user 2026-11-04: speed up SAAS; "先用1+2").

Same SAAS sparsity prior as SaasFullyBayesianSingleTaskGP (half-Cauchy global shrinkage tau,
half-Cauchy per-dimension inverse lengthscales; Matern-5/2 kernel) but fitted by MAP
(L-BFGS-B on the log posterior) instead of NUTS: one point estimate of the hyperparameters,
seconds instead of minutes. BoTorch ``botorch.models.map_saas.get_map_saas_model``.

Three independent single-output models on u, output order (quality, tokens, latency), fixed
per-config noise (``train_Yvar``) and ``Standardize`` exactly as ``fit_latent_model``. Returns a
``ModelListGP`` with an ordinary (non-MCMC) posterior, so every consumer of the Matern path
(qLogNEHVI, frontier_parents, measured HV) works unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP
from botorch.models.map_saas import get_map_saas_model
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

OBJECTIVE_NAMES = ("quality", "tokens", "latency")


def fit_map_saas_model(
    codec,
    observations,
    objective,
    *,
    seed: int = 0,
    maxiter: int = 150,
    timeout: float = 120.0,
    on_objective: Callable[[int, str], None] | None = None,
) -> ModelListGP:
    """MAP-SAAS GP per objective on u = codec.encode(config)."""
    if len({o.panel_identity for o in observations}) != 1:
        raise ValueError("GP observations must share the exact TRAIN panel and protocol")
    if len({codec.key(o.config) for o in observations}) != len(observations):
        raise ValueError("Repeated complete config observations")
    x = torch.tensor([codec.encode(o.config) for o in observations], dtype=torch.double)
    y, noise = objective.tensors(observations)
    models = []
    for i, name in enumerate(OBJECTIVE_NAMES):
        if on_objective is not None:
            on_objective(i, name)
        m = get_map_saas_model(
            x, y[:, i : i + 1], train_Yvar=noise[:, i : i + 1], outcome_transform=Standardize(m=1)
        )
        with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(seed + i)
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(m.likelihood, m),
                optimizer_kwargs={"options": {"maxiter": maxiter}, "timeout_sec": timeout},
            )
        m.eval()
        models.append(m)
    model = ModelListGP(*models)
    model.eval()
    return model
