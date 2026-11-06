"""Offline gates for TRACE latent BO (plan 2026-11-04-trace-latent-bo, Task 4; G1-G3).

G1  reconstruction: from the checkpoint metrics (held-out and measured exact match).
G2  latent smoothness: Spearman between pairwise latent distance ||z_i - z_j|| and the
    pairwise quality difference |q_i - q_j| over measured configurations (> 0 required).
    Baseline: the same statistic with the native Hamming distance between canonical tokens.
G3  surrogate quality: leave-one-out GP on the run's frozen objective scale. For each
    objective (quality, -log1p tokens, -log1p latency): mean LOO log predictive density
    (posterior variance + the held-out observation's own noise) and Spearman(LOO mean, truth).
    Native = plan_surrogate.fit_model (mixed GP on the codec); latent = SingleTaskGP on the
    encoder mean z (Matern-5/2 ARD, Standardize, fixed per-config noise).
Reads finished ledgers only. Usage:
  python -m tools.latent_offline_gates --checkpoints <vae_d12.pt> ... --runs <run> ... --out <json>

--kind seq (TRACE latent-MOBO Task 11): sequence-VAE checkpoints, u from LatentCodec with the
provisional affine (fit on the checkpoint's train u, as tools.latent_readiness), deterministic
K-fold CV (numpy default_rng(0) permutation, the SAME folds for native and every variant) instead
of LOO, surrogate variants A saas/independent, B saas/coupled-cost, C matern/independent via
fit_latent_model with default NUTS. Fully Bayesian posteriors are moment-matched over MCMC
samples (mean of means; mean of variances + variance of means). Per d: pass per variant
(tools.latent_readiness.gate_pass) and pass_any. The JSON is rewritten after every variant.
  python -m tools.latent_offline_gates --kind seq --checkpoints <seqvae_d16.pt> ... --runs <run> ...
      --out <json> [--variants A,B,C] [--folds 10] [--stop-at-first-pass] [--affine-cache DIR]
"""

import argparse
import json
import math
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from trimem.experiments.trace_joint_plan_search import get_space, observations
from trimem.optimizer.joint_search_ledger import SearchLedger
from trimem.optimizer.latent_grammar import encode_tokens
from trimem.optimizer.latent_vae import load_checkpoint
from trimem.optimizer.plan_codec import JointCodec
from trimem.optimizer.plan_surrogate import ObjectiveProtocol, fit_model

OBJ = ("quality", "tokens", "latency")


def load_run(run):
    m = json.loads((run / "manifest.json").read_text())
    space = get_space(m)
    with SearchLedger(run / "ledger.sqlite", m) as led:
        obs = observations(led, space, m)
    objective = ObjectiveProtocol(**json.loads((run / "objective.json").read_text()))
    y, v = objective.tensors(obs)
    return m, space, obs, objective, y, v


def fit_latent(Z, Y, V):
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from gpytorch.mlls import SumMarginalLogLikelihood

    models = [
        SingleTaskGP(Z, Y[:, i:i + 1], train_Yvar=V[:, i:i + 1],
                     input_transform=Normalize(Z.shape[1]), outcome_transform=Standardize(1))
        for i in range(Y.shape[1])
    ]
    model = ModelListGP(*models)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def loo(predict_one, Y, V):
    """predict_one(i) -> (mean[3], var[3]) for the held-out index i."""
    n = len(Y)
    mean, var = np.zeros((n, 3)), np.zeros((n, 3))
    for i in range(n):
        mean[i], var[i] = predict_one(i)
    return _scores(mean, var, Y, V)


def _scores(mean, var, Y, V):
    """Per objective: mean log predictive density (posterior var + observation noise) and
    Spearman(prediction, truth)."""
    y, v = Y.numpy(), V.numpy()
    s2 = var + v
    lpd = -0.5 * (np.log(2 * math.pi * s2) + (y - mean) ** 2 / s2)
    return {
        o: {"lpd": float(lpd[:, j].mean()), "spearman": float(spearmanr(mean[:, j], y[:, j]).statistic)}
        for j, o in enumerate(OBJ)
    }


def native_loo(space, obs, objective, Y, V):
    codec = JointCodec(space)
    X = torch.tensor([codec.encode(o.config) for o in obs], dtype=torch.double)

    def one(i):
        keep = [j for j in range(len(obs)) if j != i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_model(codec, [obs[j] for j in keep], objective, seed=0)
        with torch.no_grad():
            p = model.posterior(X[i:i + 1])
        return p.mean[0].numpy(), p.variance[0].numpy()

    return loo(one, Y, V)


def latent_loo(Z, Y, V):
    def one(i):
        keep = torch.tensor([j for j in range(len(Z)) if j != i])
        model = fit_latent(Z[keep], Y[keep], V[keep])
        with torch.no_grad():
            p = model.posterior(Z[i:i + 1])
        return p.mean[0].numpy(), p.variance[0].numpy()

    return loo(one, Y, V)


def smoothness(D, q):
    iu = np.triu_indices(len(q), 1)
    dq = np.abs(q[:, None] - q[None, :])[iu]
    return float(spearmanr(D[iu], dq).statistic)


# ------------------------------------------------------------------ --kind seq (Task 11)
def kfolds(n: int, k: int) -> list[np.ndarray]:
    """Deterministic K folds: default_rng(0).permutation(n) split into k near-equal parts."""
    if not 2 <= k <= n:
        raise ValueError(f"need 2 <= folds <= n, got folds={k}, n={n}")
    return np.array_split(np.random.default_rng(0).permutation(n), k)


def posterior_mv(model, U: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(mean, var), each (n, 3) in (quality, tokens, latency) order. An MCMC batch dimension
    (fully Bayesian SAAS, also inside the coupled-cost 2-model list) is moment-matched."""
    with torch.no_grad():
        p = model.posterior(U)
        m, v = p.mean, p.variance
    if m.dim() == 3:
        v = v.mean(0) + m.var(0, unbiased=False)
        m = m.mean(0)
    want = (len(U), 3)
    if tuple(m.shape) != want or tuple(v.shape) != want:
        raise ValueError(f"posterior shape {tuple(m.shape)}/{tuple(v.shape)}, expected {want}")
    return m.numpy(), v.numpy()


def _cv(fit_predict, Y, V, folds) -> dict:
    """fit_predict(train_idx, test_idx) -> (mean, var, fit_seconds, {part: seconds})."""
    n = len(Y)
    mean, var = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    secs, parts = [], {}
    for f in folds:
        test = np.asarray(f, dtype=int)
        train = np.setdiff1d(np.arange(n), test)
        mean[test], var[test], s, part = fit_predict(train, test)
        secs.append(s)
        for k, x in part.items():
            parts.setdefault(k, []).append(x)
    return {
        "G3": _scores(mean, var, Y, V),
        "fit_seconds": {"mean": float(np.mean(secs)), "max": float(np.max(secs)), "n": len(secs)},
        "objective_seconds": {k: float(np.mean(x)) for k, x in parts.items()},
    }


def native_cv(space, obs, objective, Y, V, folds) -> dict:
    codec = JointCodec(space)
    X = torch.tensor([codec.encode(o.config) for o in obs], dtype=torch.double)

    def fit_predict(train, test):
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_model(codec, [obs[j] for j in train], objective, seed=0)
        s = time.perf_counter() - t
        return (*posterior_mv(model, X[test]), s, {})

    return _cv(fit_predict, Y, V, folds)


def latent_cv(codec, obs, objective, Y, V, folds, *, kernel, structure, seed=0, nuts=None) -> dict:
    """K-fold CV of fit_latent_model (the live GP); nuts=None keeps the live NUTS defaults."""
    from trimem.optimizer.latent_plan_surrogate import fit_latent_model

    U = codec.encode_many([o.config for o in obs])
    extra = {} if nuts is None else {"nuts": tuple(nuts)}
    if kernel == "map-saas":  # MAP-SAAS (trimem/optimizer/latent_map_saas.py)
        from trimem.optimizer.latent_map_saas import fit_map_saas_model

        def fit_latent_model(codec, obs_, objective_, *, seed, kernel, structure, on_objective, **_):
            return fit_map_saas_model(codec, obs_, objective_, seed=seed, on_objective=on_objective)
    if kernel == "saas-numpyro":  # SAAS with NumPyro/JAX NUTS (trimem/optimizer/latent_numpyro_saas.py)
        if structure != "independent":
            raise ValueError("kernel 'saas-numpyro' supports structure 'independent' only")
        from trimem.optimizer import latent_numpyro_saas as nps

        warmup, samples, thinning = nps.DEFAULT_NUTS if nuts is None else tuple(nuts)

        def fit_latent_model(codec, obs_, objective_, *, seed, kernel, structure, on_objective, **_):
            return nps.fit_saas_numpyro(codec, obs_, objective_, seed=seed, warmup=warmup, samples=samples,
                                        thinning=thinning, on_objective=on_objective)

    def fit_predict(train, test):
        marks = []
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_latent_model(
                codec, [obs[j] for j in train], objective, seed=seed, kernel=kernel,
                structure=structure, on_objective=lambda i, name: marks.append((name, time.perf_counter())),
                **extra,
            )
        end = time.perf_counter()
        ends = [m[1] for m in marks[1:]] + [end]
        parts = {name: e - s for (name, s), e in zip(marks, ends)}
        if structure == "independent" and kernel == "matern":
            parts = {"joint": end - t}  # one joint MAP fit; on_objective fires once
        if kernel == "saas-numpyro":
            parts = {"joint": end - t}  # one vmapped NUTS program for the 3 objectives
        return (*posterior_mv(model, U[test]), end - t, parts)

    return _cv(fit_predict, Y, V, folds)


def _safe_roundtrip(codec, config) -> bool:
    """A decoder exception counts as a failed round trip (as in readiness / freeze-affine)."""
    try:
        return bool(codec.roundtrip(config))
    except Exception:  # noqa: BLE001 - any decode failure is a failed round trip
        return False


def _dump(path: Path, report: dict) -> None:
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, path)


def _seq_d(path) -> int:
    m = re.search(r"d(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


def run_seq(checkpoints, runs, out, *, variants=("A", "B", "C"), folds=10, stop_at_first_pass=False,
            affine_cache=None, nuts=None) -> dict:
    """G2/G3 for sequence-VAE checkpoints (ascending d) against the native mixed GP."""
    from tools.latent_readiness import (
        PROVISIONAL_KIND,
        VARIANTS,
        affine_cache_path,
        gate_pass,
        mean_lpd,
        provisional_affine,
    )
    from trimem.optimizer.latent_plan_codec import LatentCodec, file_sha256
    from trimem.optimizer.latent_seq_vae import load_checkpoint as load_seq

    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; expected a subset of {sorted(VARIANTS)}")
    out = Path(out)
    report = {"kind": "seq", "folds": folds, "fold_seed": 0, "variants": {v: VARIANTS[v] for v in variants},
              "nuts": "fit_latent_model defaults" if nuts is None else list(nuts),
              "margins": {"spearman": 0.10, "lpd": 0.5}, "loadavg_start": list(os.getloadavg()),
              "runs": {}, "d": {}}
    loaded = {Path(r).name: load_run(Path(r)) for r in runs}
    spaces = [x[1] for x in loaded.values()]
    if any(s.domains != spaces[0].domains for s in spaces):
        raise ValueError("runs have different field domains")
    fold_of = {}
    for name, (m, space, obs, objective, Y, V) in loaded.items():
        fold_of[name] = kfolds(len(obs), folds)
        tok = np.array([encode_tokens(space, o.config) for o in obs])
        ham = (tok[:, None, :] != tok[None, :, :]).sum(-1).astype(float)
        nat = native_cv(space, obs, objective, Y, V, fold_of[name])
        report["runs"][name] = {"n": len(obs), "folds": [f.tolist() for f in fold_of[name]],
                                "G2_native_hamming": smoothness(ham, Y[:, 0].numpy()),
                                "G3_native": nat["G3"], "native_fit_seconds": nat["fit_seconds"]}
        print(name, "native", json.dumps(report["runs"][name]["G3_native"]), flush=True)
        _dump(out, report)
    cks = []
    for ck in map(Path, checkpoints):
        if not ck.exists():
            report["d"][f"d{_seq_d(ck)}"] = None
            continue
        model, met = load_seq(ck, spaces[0])
        cks.append((model.d, ck, model, met))
    for d, ck, model, met in sorted(cks, key=lambda t: t[0]):
        sha = file_sha256(ck)
        cache = ck.parent if affine_cache is None else Path(affine_cache)
        aff = provisional_affine(ck, spaces[0], model, met, cache_dir=cache, sha=sha)
        entry = {"checkpoint": str(ck), "checkpoint_sha256": sha,
                 "affine": {"kind": PROVISIONAL_KIND, "cache": str(affine_cache_path(cache, ck))}, "runs": {}}
        report["d"][f"d{d}"] = entry
        for name, (_m, space, obs, objective, Y, V) in loaded.items():
            codec = LatentCodec(space, model, aff, checkpoint_sha256=sha)
            U = codec.encode_many([o.config for o in obs])
            rr = {"G2_spearman": smoothness(torch.cdist(U, U).numpy(), Y[:, 0].numpy()),
                  "G1_measured_roundtrip": float(np.mean([_safe_roundtrip(codec, o.config) for o in obs])),
                  "variants": {}}
            entry["runs"][name] = rr
            for v in variants:
                rr["variants"][v] = latent_cv(codec, obs, objective, Y, V, fold_of[name], **{"nuts": nuts, **VARIANTS[v]})
                print(name, f"d{d}", v, json.dumps(rr["variants"][v]), flush=True)
                _dump(out, report)
        native = {n: r["G3_native"] for n, r in report["runs"].items()}
        entry["pass"] = {v: gate_pass({n: {"G2_spearman": r["G2_spearman"], "G3": r["variants"][v]["G3"]}
                                       for n, r in entry["runs"].items()}, native) for v in variants}
        entry["pass_any"] = any(entry["pass"].values())
        entry["mean_lpd"] = {v: mean_lpd(entry, v) for v in variants}
        _dump(out, report)
        print(f"d{d}", "pass", json.dumps(entry["pass"]), flush=True)
        if stop_at_first_pass and entry["pass_any"]:
            report["stopped_after"] = d
            _dump(out, report)
            break
    return report


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("slot", "seq"), default="slot")
    ap.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variants", default="A,B,C", help="seq only")
    ap.add_argument("--folds", type=int, default=10, help="seq only")
    ap.add_argument("--stop-at-first-pass", action="store_true", help="seq only")
    ap.add_argument("--affine-cache", type=Path, default=None,
                    help="seq only; provisional affine cache (default: the checkpoint's directory)")
    return ap


def main():
    a = build_parser().parse_args()
    if a.kind == "seq":
        torch.set_num_threads(4)
        run_seq(a.checkpoints, a.runs, a.out, variants=tuple(a.variants.split(",")), folds=a.folds,
                stop_at_first_pass=a.stop_at_first_pass, affine_cache=a.affine_cache)
        return
    torch.set_num_threads(8)
    report = {"runs": {}}
    for run in a.runs:
        m, space, obs, objective, Y, V = load_run(run)
        tok = np.array([encode_tokens(space, o.config) for o in obs])
        q = Y[:, 0].numpy()
        ham = (tok[:, None, :] != tok[None, :, :]).sum(-1).astype(float)
        r = {"n": len(obs), "G2_native_hamming": smoothness(ham, q),
             "G3_native": native_loo(space, obs, objective, Y, V), "latent": {}}
        for ck in a.checkpoints:
            vae, met = load_checkpoint(ck, space)
            with torch.no_grad():
                Z = vae.encode_mean(torch.as_tensor(tok)).double()
            D = torch.cdist(Z, Z).numpy()
            r["latent"][f"d{vae.d}"] = {
                "G1_heldout_exact": met["heldout"]["exact"],
                "G1_measured_exact": met.get("measured", {}).get("exact"),
                "G2_spearman": smoothness(D, q),
                "G3": latent_loo(Z, Y, V),
            }
            print(run.name, f"d{vae.d}", json.dumps(r["latent"][f"d{vae.d}"]), flush=True)
        print(run.name, "native", json.dumps({"G2": r["G2_native_hamming"], "G3": r["G3_native"]}), flush=True)
        report["runs"][run.name] = r
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
