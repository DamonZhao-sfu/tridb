"""Contextual GP router: one joint surrogate per objective over (configuration, query).

Input row = [configuration features (plan one-hot, edges one-hot, normalized knobs) (+) query
features projected on a PCA of the training queries], min-max normalized. Targets: per-query
quality, ln tokens, ln latency (standardized). One exact GP per objective (ScaleKernel(RBF-ARD),
constant mean, Gaussian noise) fitted with botorch; rows are subsampled to ``max_rows`` for the
O(n^3) fit. The fitted GP is exported (inputs, alpha, hyper-parameters) so that
``runtime.predict_gp`` evaluates the posterior mean with numpy only.

Configurations share information through the kernel, so the router can choose among every measured
configuration (no pruning needed); a candidate subset can still be passed.
"""

from __future__ import annotations

import numpy as np

from trimem.optimizer import session_space_agentenv as space
from trimem.optimizer.query_router.matrix import OutcomeMatrix

PLAN_LEVELS = ("NoMemory", *space.PLANS)
KNOBS = (("seed_count", 32), ("depth", 3), ("fanout", 16), ("top_k", 5))
MAX_ROWS, QUERY_DIMS = 1000, 6
NOISE_FLOOR = 1e-3  # standardized units; keeps K well conditioned with repeated inputs


def config_features(cfg: dict) -> list[float]:
    plan = cfg["plan"]
    out = [float(plan == p) for p in PLAN_LEVELS]
    out += [float(cfg.get("edges") == e) for e in space.EDGES]
    out += [float(cfg.get(k, 0)) / hi for k, hi in KNOBS]
    return out


def _query_projection(X: np.ndarray, dims: int) -> dict | None:
    if X.shape[1] <= dims:
        return None
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return {"mean": mu.tolist(), "components": vt[:dims].tolist()}


def _fit_one(Z: np.ndarray, y: np.ndarray, seed: int, return_model: bool = False):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import RBFKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from botorch.exceptions.errors import ModelFittingError
    from gpytorch.constraints import GreaterThan
    from gpytorch.likelihoods import GaussianLikelihood

    torch.manual_seed(seed)
    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
    tx = torch.tensor(Z, dtype=torch.double)
    ty = torch.tensor(((y - y_mean) / y_std)[:, None], dtype=torch.double)
    gp = SingleTaskGP(
        tx,
        ty,
        covar_module=ScaleKernel(RBFKernel(ard_num_dims=Z.shape[1])),
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(NOISE_FLOOR)),
        outcome_transform=None,
        input_transform=None,
    )
    fit_status = "ok"
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    except ModelFittingError:
        # botorch restores the initial hyper-parameters after failed attempts; keep them (lengthscale
        # ln 2 per normalized input, unit output scale) rather than failing the whole router.
        fit_status = "failed: initial hyper-parameters"
    ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    os_ = float(gp.covar_module.outputscale.detach())
    noise = float(gp.likelihood.noise.detach().ravel()[0])
    const = float(gp.mean_module.constant.detach().ravel()[0])
    d = ((Z[:, None, :] - Z[None, :, :]) / ls) ** 2
    K = os_ * np.exp(-0.5 * d.sum(-1)) + noise * np.eye(len(Z))
    alpha = np.linalg.solve(K, (y - y_mean) / y_std - const)
    out = {
        "X": Z.tolist(),
        "alpha": alpha.tolist(),
        "lengthscale": ls.tolist(),
        "outputscale": os_,
        "noise": noise,
        "const": const,
        "y_mean": y_mean,
        "y_std": y_std,
        "fit": fit_status,
    }
    return (out, gp) if return_model else out


def fit(
    X: np.ndarray,
    m: OutcomeMatrix,
    *,
    candidates: list[int] | None = None,
    max_rows: int = MAX_ROWS,
    query_dims: int = QUERY_DIMS,
    seed: int = 0,
) -> dict:
    if len(X) != len(m.query_ids):
        raise ValueError("features and outcome rows differ")
    cand = list(range(len(m.configs))) if candidates is None else list(candidates)
    proj = _query_projection(X, query_dims)
    Xq = (
        (X - np.asarray(proj["mean"])) @ np.asarray(proj["components"]).T if proj else X
    )
    cf_all = np.array([config_features(m.configs[c]) for c in range(len(m.configs))])
    pairs = [(q, c) for q in range(len(X)) for c in range(len(m.configs))]
    rng = np.random.default_rng(seed)
    if len(pairs) > max_rows:
        pairs = [
            pairs[i] for i in sorted(rng.choice(len(pairs), max_rows, replace=False))
        ]
    rows = np.array([np.concatenate([cf_all[c], Xq[q]]) for q, c in pairs])
    keep = rows.max(0) > rows.min(
        0
    )  # constant inputs carry no information and degrade the fit
    rows = rows[:, keep]
    lo, hi = rows.min(0), rows.max(0)
    # Query features of a new query may fall outside [0, 1] after this normalization; the RBF kernel
    # then decays toward the constant mean (no clipping).
    Z = (rows - lo) / np.where(hi > lo, hi - lo, 1.0)
    targets = {
        "quality": np.array([m.quality[q, c] for q, c in pairs]),
        "ln_tokens": np.log([m.tokens[q, c] for q, c in pairs]),
        "ln_latency": np.log([m.latency_ms[q, c] for q, c in pairs]),
    }
    return {
        "kind": "gp",
        "configs": [m.configs[c] for c in cand],
        "candidate_columns": cand,
        "cfg_feats": cf_all[cand].tolist(),
        "query_proj": proj,
        "x_keep": keep.tolist(),
        "x_lo": lo.tolist(),
        "x_hi": hi.tolist(),
        "objectives": {k: _fit_one(Z, y, seed) for k, y in targets.items()},
        "params": {
            "max_rows": max_rows,
            "query_dims": query_dims,
            "rows": len(pairs),
            "seed": seed,
        },
    }
