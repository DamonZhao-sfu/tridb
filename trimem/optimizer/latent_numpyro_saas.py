"""SAAS fully Bayesian GP fitted by NumPyro/JAX NUTS, loaded into BoTorch (user 2026-11-04).

BoTorch's ``fit_fully_bayesian_model_nuts`` runs Pyro NUTS (70-90 s per objective at n~50, d=16,
512/256/16 on this host). This module runs the same NUTS on a NumPyro model that is identical to
``botorch.models.fully_bayesian.SaasPyroModel`` (fixed-noise branch) and loads the thinned
samples into an ordinary ``SaasFullyBayesianSingleTaskGP`` with its own
``pyro_model.postprocess_mcmc_samples`` + ``load_mcmc_samples``. The result is a normal BoTorch
fully Bayesian model: posterior batch shape [S, n, 1], qLogNEHVI, gates, snapshots unchanged.

Site-by-site mapping (BoTorch 0.17.2 SaasPyroModel.sample -> ``_saas_model``):
  outputscale            Gamma(2.0, rate 0.15)                 -> same site, same prior
  mean                   Normal(0, 1)                          -> same
  noise                  train_Yvar given: no site, K += diag(train_Yvar) (Standardize-scaled,
                         clamped >= 1e-4 by the BoTorch constructor; read from pyro_model)
  kernel_tausq           HalfCauchy(0.1)                       -> same
  _kernel_inv_length_sq  HalfCauchy(ones(d))                   -> same
  lengthscale            deterministic rsqrt(tausq * inv)      -> computed inline
  K                      outputscale * Matern52(X / lengthscale), dist clamped at 1e-30 before
                         sqrt, + diag(noise)                    -> same formula
  Y                      MultivariateNormal(mean, K), obs = Standardize(train_Y)
NUTS: dense mass (Pyro full_mass=True), max_tree_depth=6, target_accept 0.8, step-size and mass
adaptation on, heuristic initial step size (Pyro does this at warmup start), init_to_uniform,
one chain; samples are post-processed then thinned [::thinning] exactly as BoTorch does.

Padding (compile once per bucket): JAX compiles the NUTS program per data shape, so n is padded
to ``bucket(n)`` (multiple of 8). A padded row has zero kernel covariance with every real row,
unit noise, zero residual and zero mean: its factor is N(0 | 0, 1), a constant. The hyperparameter
posterior and every NUTS acceptance decision are unchanged (only floating-point rounding
differs). ``tests/test_trace_latent_numpyro_saas.py`` checks both facts.

CPU threads: XLA sizes its CPU thread pool from the process CPU affinity mask, and the XLA_FLAGS
``--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`` do not limit it (measured:
342 threads, ~21 busy cores). The only limit that works is the affinity mask:
  * API (live worker): call ``pin_cpus(cpus, offset=0)`` BEFORE the first JAX use in the process
    (the XLA pool is sized when JAX creates its CPU client). ``cpus`` is an explicit CPU list or a
    count; a count takes the allowed CPUs [offset, offset + count). It pins every thread of the
    process (torch threads included).
  * Env (opt-in, gate scripts): ``TRIMEM_JAX_CPUS=N`` pins at import to the N highest-numbered
    allowed CPUs; ``TRIMEM_JAX_CPUS=28-31,36`` pins to that explicit list. Unset or 0: no pinning
    (then bound the process with pin_cpus, taskset or a CPUQuota scope). A malformed value does not
    break the import; every fit then raises ValueError naming TRIMEM_JAX_CPUS.
``JAX_PLATFORMS`` defaults to ``cpu``; float64 is enabled globally.

Warm start (speed option 2b) and its provenance / aging rules: see latent_numpyro_warm.
Fit health: a lane that finds no valid initial point in 100 draws, or any non-finite sample,
raises RuntimeError; divergences are recorded per objective (model.numpyro_summary).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path

CPU_ENV = "TRIMEM_JAX_CPUS"
DEFAULT_NUTS = (512, 256, 16)
OBJECTIVE_NAMES = ("quality", "tokens", "latency")
BUCKET = 8
MAX_TREE_DEPTH = 6


def _pin_all_threads(keep: list[int]) -> list[int]:
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), set(keep))
        except OSError:  # a thread that exited between listdir and setaffinity
            pass
    return sorted(keep)


def pin_cpus(cpus: list[int] | int, *, offset: int = 0) -> list[int]:
    """Pin every thread of this process to a caller-chosen CPU set: an explicit list (must be a
    subset of the currently allowed CPUs) or a count taking allowed CPUs [offset, offset + count).
    Call it before the first JAX use (module docstring). Returns the pinned CPUs."""
    global PINNED_CPUS
    allowed = sorted(os.sched_getaffinity(0))
    if isinstance(cpus, bool) or not isinstance(cpus, (int, list, tuple)):
        raise TypeError(f"pin_cpus expects a CPU count or a list of CPU ids, got {cpus!r}")
    if isinstance(cpus, int):
        if cpus <= 0 or offset < 0 or offset + cpus > len(allowed):
            raise ValueError(f"pin_cpus({cpus}, offset={offset}): only {len(allowed)} CPUs are allowed")
        keep = allowed[offset : offset + cpus]
    else:
        keep = sorted({int(c) for c in cpus})
        if not keep or not set(keep) <= set(allowed):
            raise ValueError(f"pin_cpus({list(cpus)}): CPUs not allowed for this process (allowed {allowed})")
    PINNED_CPUS = _pin_all_threads(keep)
    return PINNED_CPUS


def limit_process_cpus(n: int) -> list[int] | None:
    """Pin to the n highest-numbered allowed CPUs (the TRIMEM_JAX_CPUS=N behaviour); n <= 0: no-op."""
    if n <= 0:
        return None
    allowed = sorted(os.sched_getaffinity(0))
    return pin_cpus(allowed[-n:])


def _parse_cpu_env(value: str | None) -> tuple[list[int] | int | None, str | None]:
    """TRIMEM_JAX_CPUS -> (count | explicit list | None, error)."""
    v = (value or "").strip()
    if v in ("", "0"):
        return None, None
    try:
        if v.isdigit():
            return int(v), None
        cpus = []
        for part in v.split(","):
            a, _, b = part.strip().partition("-")
            cpus.extend(range(int(a), int(b) + 1) if b else [int(a)])
        return cpus, None
    except ValueError:
        return None, f"{CPU_ENV}={value!r} is not a CPU count (e.g. 4) or a CPU list (e.g. 28-31,36)"


# Opt-in (coordinator 2026-11-04): a live worker must not silently re-pin its torch threads, so the
# import pins only when TRIMEM_JAX_CPUS is set (the gate scripts set it). A malformed value is
# reported at use (_check_env), not at import.
PINNED_CPUS: list[int] | None = None
_CPU_SPEC, _CPU_ENV_ERROR = _parse_cpu_env(os.environ.get(CPU_ENV))
if _CPU_SPEC is not None:
    try:
        PINNED_CPUS = limit_process_cpus(_CPU_SPEC) if isinstance(_CPU_SPEC, int) else pin_cpus(_CPU_SPEC)
    except ValueError as _exc:
        _CPU_ENV_ERROR = f"{CPU_ENV}={os.environ.get(CPU_ENV)!r}: {_exc}"


def _check_env() -> None:
    if _CPU_ENV_ERROR is not None:
        raise ValueError(_CPU_ENV_ERROR)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# imported after the affinity / platform setup above
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
from botorch.models import ModelListGP, SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from jax import lax
from numpyro.infer import MCMC, NUTS
from numpyro.infer.hmc import hmc
from numpyro.infer.util import ParamInfo, initialize_model
from numpyro.infer.util import log_density as _np_log_density

from .latent_numpyro_warm import (
    MAX_N_GROWTH,
    MAX_WARM_GENERATIONS,
    PROVENANCE_KEYS,
    WARM_STATE_VERSION,
    affine_sha256,
    load_states,
    objective_sha256,
    save_states,
    warm_meta,
)

# re-exported warm-start API (defined in latent_numpyro_warm)
__all__ = [
    "MAX_N_GROWTH",
    "MAX_WARM_GENERATIONS",
    "PROVENANCE_KEYS",
    "WARM_STATE_VERSION",
    "affine_sha256",
    "check_finite_samples",
    "enable_compilation_cache",
    "fit_saas_numpyro",
    "load_warm_state",
    "log_density",
    "objective_sha256",
    "pad_inputs",
    "pin_cpus",
    "run_vmapped",
    "saas_multi_numpyro",
    "saas_single_numpyro",
    "save_warm_state",
    "warm_meta",
    "warm_state",
    "warm_state_valid",
]

jax.config.update("jax_enable_x64", True)

# Persistent XLA compilation cache (speed option 2a, user 2026-11-04). DEFAULT ON, repo-local:
# <repo>/results/.jax_cache. TRIMEM_JAX_CACHE_DIR=<dir> moves it; TRIMEM_JAX_CACHE_DIR=off disables.
# A fresh process (e.g. a newly spawned live worker) then loads the compiled NUTS program from disk
# instead of compiling it again (tracing/lowering still run). Cache keys include the HLO, the XLA
# flags, the jax/jaxlib version and the device, so a stale entry is never reused for a new program.
# Only programs that take >= CACHE_MIN_COMPILE_SECS to compile are stored (the NUTS programs take
# 5-10 s; small helper jits are not cached); the directory is capped at CACHE_MAX_BYTES (LRU eviction
# by JAX). JAX creates the directory on the first write.
CACHE_ENV = "TRIMEM_JAX_CACHE_DIR"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / ".jax_cache"
CACHE_MIN_COMPILE_SECS = 1.0
CACHE_MAX_BYTES = 2 * 1024**3
COMPILATION_CACHE: str | None = None


def _reset_jax_cache() -> None:
    try:  # private API: re-read the cache dir on the next compile (no-op before the first compile)
        from jax._src import compilation_cache as _cc

        _cc.reset_cache()
    except (ImportError, AttributeError):
        pass


def enable_compilation_cache(path: str | os.PathLike | None) -> str | None:
    """Point the persistent cache at path; None / "" / "0" / "off" disables it."""
    global COMPILATION_CACHE
    if path is None or str(path).strip().lower() in ("", "0", "off", "none"):
        jax.config.update("jax_compilation_cache_dir", None)
        COMPILATION_CACHE = None
        _reset_jax_cache()
        return None
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", CACHE_MIN_COMPILE_SECS)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_compilation_cache_max_size", CACHE_MAX_BYTES)
    _reset_jax_cache()
    COMPILATION_CACHE = str(path)
    return COMPILATION_CACHE


enable_compilation_cache(os.environ.get(CACHE_ENV, str(DEFAULT_CACHE_DIR)))
_SQRT5 = math.sqrt(5.0)
_SAMPLERS: dict[tuple, object] = {}  # MCMC (sequential) or jitted vmapped fn


def _matern52(X, inv_length_sq):
    Xs = X / jax.lax.rsqrt(inv_length_sq)  # X / lengthscale, as BoTorch compute_dists
    sq = jnp.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(jnp.maximum(sq, 1e-30))
    return (_SQRT5 * r + 1.0 + 5.0 / 3.0 * r**2) * jnp.exp(-_SQRT5 * r)


def _saas_model(X, Y, Yvar, mask):
    """NumPyro twin of SaasPyroModel.sample with train_Yvar given (see module docstring)."""
    d = X.shape[-1]
    outputscale = numpyro.sample("outputscale", dist.Gamma(2.0, 0.15))
    mean = numpyro.sample("mean", dist.Normal(0.0, 1.0))
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(0.1))
    inv = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(d)))
    k = outputscale * _matern52(X, tausq * inv) * (mask[:, None] * mask[None, :])
    K = k + jnp.diag(jnp.where(mask > 0, Yvar, 1.0))
    numpyro.sample("Y", dist.MultivariateNormal(mean * mask, covariance_matrix=K), obs=Y)


def bucket(n: int) -> int:
    return BUCKET * math.ceil(n / BUCKET)


def pad_inputs(X, Y, Yvar, n_pad: int):
    """(X, Y, Yvar, mask) padded to n_pad rows; padded rows are masked out (module docstring)."""
    X, Y, Yvar = (np.asarray(a, dtype=np.float64) for a in (X, Y, Yvar))
    n, d = X.shape
    if n_pad < n:
        raise ValueError(f"n_pad {n_pad} < n {n}")
    Xp = np.zeros((n_pad, d))
    Xp[:n] = X
    Yp, Vp, M = np.zeros(n_pad), np.ones(n_pad), np.zeros(n_pad)
    Yp[:n], Vp[:n], M[:n] = Y, Yvar, 1.0
    return Xp, Yp, Vp, M


def log_density(params: dict, X, Y, Yvar, mask=None) -> float:
    """log p(params, Y) of the NumPyro model in constrained space (tests: equals Pyro's trace)."""
    X = jnp.asarray(X, dtype=jnp.float64)
    mask = jnp.ones(X.shape[0]) if mask is None else jnp.asarray(mask, dtype=jnp.float64)
    p = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
    lp, _ = _np_log_density(
        _saas_model, (X, jnp.asarray(Y, dtype=jnp.float64), jnp.asarray(Yvar, dtype=jnp.float64), mask),
        {}, p,
    )
    return float(lp)


def compiled_samplers() -> int:
    """Number of distinct compiled NUTS programs (one per (bucket, d, warmup, samples))."""
    return len(_SAMPLERS)


def _sampler(n_pad: int, d: int, warmup: int, samples: int) -> MCMC:
    key = (n_pad, d, warmup, samples)
    if key not in _SAMPLERS:
        kernel = NUTS(_saas_model, dense_mass=True, max_tree_depth=MAX_TREE_DEPTH,
                      target_accept_prob=0.8, find_heuristic_step_size=True)
        _SAMPLERS[key] = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=1,
                              progress_bar=False, jit_model_args=True)
    return _SAMPLERS[key]


def _check_nuts(warmup: int, samples: int, thinning: int) -> None:
    if min(warmup, samples, thinning) < 1:
        raise ValueError(f"NUTS warmup/samples/thinning must be >= 1, got {(warmup, samples, thinning)}")


SITES = ("outputscale", "mean", "kernel_tausq", "_kernel_inv_length_sq")


def _new_model(x, y1, v1) -> SaasFullyBayesianSingleTaskGP:
    m = SaasFullyBayesianSingleTaskGP(train_X=x, train_Y=y1, train_Yvar=v1, outcome_transform=Standardize(m=1))
    m.train()
    return m


def _padded(m):
    """(X, Y, Yvar, mask) of the data exactly as SaasPyroModel sees it (transformed, clamped), padded."""
    pm = m.pyro_model
    return pad_inputs(pm.train_X.numpy(), pm.train_Y.squeeze(-1).numpy(), pm.train_Yvar.squeeze(-1).numpy(),
                      bucket(pm.train_X.shape[0]))


def _load(m, raw: dict, thinning: int, info: dict) -> SaasFullyBayesianSingleTaskGP:
    """BoTorch's own postprocess -> [::thinning] -> load_mcmc_samples, as fit_fully_bayesian_model_nuts."""
    pm = m.pyro_model
    raw = {k: torch.as_tensor(np.asarray(raw[k]), dtype=pm.train_X.dtype) for k in SITES}
    mcmc_samples = pm.postprocess_mcmc_samples(mcmc_samples=raw)
    m.load_mcmc_samples({k: v[::thinning] for k, v in mcmc_samples.items()})
    m.eval()
    m.numpyro_info = info
    return m


def saas_single_numpyro(x, y1, v1, *, seed: int = 0, warmup: int = 512, samples: int = 256,
                        thinning: int = 16) -> SaasFullyBayesianSingleTaskGP:
    """One SaasFullyBayesianSingleTaskGP (Standardize(m=1), fixed train_Yvar) fitted by NumPyro MCMC."""
    _check_env()
    _check_nuts(warmup, samples, thinning)
    m = _new_model(x, y1, v1)
    Xp, Yp, Vp, M = _padded(m)
    compiled = len(_SAMPLERS)
    mcmc = _sampler(Xp.shape[0], Xp.shape[1], warmup, samples)
    t = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(Xp), jnp.asarray(Yp), jnp.asarray(Vp), jnp.asarray(M),
             extra_fields=("diverging",))
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    info = {"seconds": time.perf_counter() - t, "new_compile": len(_SAMPLERS) > compiled, "mode": "sequential",
            "n": int(m.pyro_model.train_X.shape[0]), "n_pad": int(Xp.shape[0]), "divergences": divergences,
            "nuts": [warmup, samples, thinning], "init_tries": None}
    raw = mcmc.get_samples()
    check_finite_samples({k: raw[k] for k in SITES}, "objective")
    return _load(m, raw, thinning, info)


# ------------------------------------------------------------ parallel: one vmapped NUTS program
def _vmapped_sampler(n_pad: int, d: int, warmup: int, samples: int, warm: bool = False):
    """jit(vmap(lane)) over objectives, X and mask shared. Each lane is the NUTS that MCMC runs for
    one objective (numpyro.infer.hmc.hmc NUTS kernel, same settings as _sampler).
    Cold lane: init_to_uniform (uniform(-2, 2) in unconstrained space, redrawn until potential and
    gradient are finite, <= 100 tries, as find_valid_initial_params), warmup with step-size +
    dense-mass adaptation and a heuristic initial step size, then `samples` draws.
    Warm lane (speed option 2b): init at the given unconstrained z with the given step size and
    dense inverse mass matrix; warmup adapts the step size only (mass matrix fixed), then `samples`
    draws. Every lane returns its constrained samples, divergences, its final (z, step size,
    inverse mass matrix) for the next warm start, and its init tries (-1 = valid initial point). Under vmap a NUTS step costs the largest tree of
    the lanes."""
    key = ("vmap", n_pad, d, warmup, samples, warm)
    if key in _SAMPLERS:
        return _SAMPLERS[key]
    dummy = (jnp.zeros((n_pad, d)), jnp.zeros(n_pad), jnp.ones(n_pad), jnp.ones(n_pad))
    info, potential_gen, postprocess_gen, _ = initialize_model(
        jax.random.PRNGKey(0), _saas_model, model_args=dummy, dynamic_args=True)
    names = tuple(sorted(info.z))
    shapes = {k: jnp.shape(info.z[k]) for k in names}
    init_kernel, sample_kernel = hmc(potential_fn_gen=potential_gen, algo="NUTS")

    def lane(rng_key, X, Y, V, M, z_warm, step_warm, imm_warm):
        args = (X, Y, V, M)
        rng_key, key_init = jax.random.split(rng_key)  # as NUTS.init
        value_and_grad = jax.value_and_grad(potential_gen(*args))
        if warm:
            z = z_warm
            pe, g = value_and_grad(z)
            ok = jnp.isfinite(pe) & jnp.all(jnp.array([jnp.all(jnp.isfinite(g[nm])) for nm in names]))
            tries = jnp.where(ok, -1, 100)
            state = init_kernel(ParamInfo(z, pe, g), num_warmup=warmup, step_size=step_warm,
                                inverse_mass_matrix={names: imm_warm}, adapt_step_size=True,
                                adapt_mass_matrix=False, dense_mass=[names], target_accept_prob=0.8,
                                max_tree_depth=MAX_TREE_DEPTH, find_heuristic_step_size=False,
                                model_args=args, rng_key=rng_key)
        else:
            def draw(carry):
                k, _, _, _, tries = carry
                k, sub = jax.random.split(k)
                ks = jax.random.split(sub, len(names))
                z = {nm: jax.random.uniform(kk, shapes[nm], minval=-2.0, maxval=2.0) for nm, kk in zip(names, ks)}
                pe, g = value_and_grad(z)
                ok = jnp.isfinite(pe) & jnp.all(jnp.array([jnp.all(jnp.isfinite(g[nm])) for nm in names]))
                return k, z, pe, g, jnp.where(ok, -1, tries + 1)

            z0 = {nm: jnp.zeros(shapes[nm]) for nm in names}
            carry = (key_init, z0, jnp.array(jnp.inf), z0, jnp.array(0))
            _, z, pe, g, tries = lax.while_loop(lambda c: (c[4] >= 0) & (c[4] < 100), draw, carry)
            state = init_kernel(ParamInfo(z, pe, g), num_warmup=warmup, dense_mass=[names],
                                target_accept_prob=0.8, max_tree_depth=MAX_TREE_DEPTH,
                                find_heuristic_step_size=True, model_args=args, rng_key=rng_key)
        state = lax.fori_loop(0, warmup, lambda i, s: sample_kernel(s, model_args=args), state)

        def body(s, _):
            s = sample_kernel(s, model_args=args)
            return s, (s.z, s.diverging)

        last, (zs, diverging) = lax.scan(body, state, None, length=samples)
        final = (last.z, last.adapt_state.step_size, last.adapt_state.inverse_mass_matrix[names])
        return jax.vmap(postprocess_gen(*args))(zs), diverging, final, tries

    fn = jax.jit(jax.vmap(lane, in_axes=(0, None, 0, 0, None, 0, 0, 0)))
    _SAMPLERS[key] = (fn, names, shapes)
    return _SAMPLERS[key]


def _unconstrained(name: str, value):
    """Constrained site value -> NUTS unconstrained space (exp transform for the positive sites)."""
    return np.asarray(value, dtype=np.float64) if name == "mean" else np.log(np.asarray(value, dtype=np.float64))


def check_finite_samples(raw: dict, name: str) -> None:
    """Raise before load_mcmc_samples if any NUTS sample is NaN / inf."""
    bad = [k for k, v in raw.items() if not np.all(np.isfinite(np.asarray(v)))]
    if bad:
        raise RuntimeError(f"non-finite NUTS samples for {name}: sites {bad}")


def warm_state_valid(state: dict | None, d: int, lanes) -> tuple[bool, str]:
    """A warm state is usable when it has every objective (with matching names), the same d, site
    layout and site shapes, generation < MAX_WARM_GENERATIONS, the new n <= MAX_N_GROWTH x
    cold_fit_n (aging rule, latent_numpyro_warm), a finite step size > 0, a finite positive-definite
    inverse mass matrix, and a finite log density on the NEW (padded) data of every objective."""
    if not state:
        return False, "no warm state"
    n_new = int(np.asarray(lanes[0][3]).sum())
    try:
        for i, name in enumerate(OBJECTIVE_NAMES):
            st = state[name]
            if st.get("objective") != name:
                return False, f"{name}: objective name mismatch"
            if int(st["d"]) != d or tuple(st["sites"]) != tuple(sorted(SITES)):
                return False, f"{name}: d/site layout mismatch"
            if int(st["generation"]) >= MAX_WARM_GENERATIONS:
                return False, f"{name}: generation {st['generation']} >= {MAX_WARM_GENERATIONS} (aging rule)"
            if n_new > MAX_N_GROWTH * int(st["cold_fit_n"]):
                return False, f"{name}: n {n_new} > {MAX_N_GROWTH} x cold_fit_n {st['cold_fit_n']} (aging rule)"
            constrained = {k: np.asarray(st["constrained"][k], dtype=np.float64) for k in SITES}
            want = {"_kernel_inv_length_sq": d}
            if any(constrained[k].size != want.get(k, 1) for k in SITES):
                return False, f"{name}: site shape mismatch"
            imm = np.asarray(st["inverse_mass_matrix"], dtype=np.float64)
            dim = 3 + d
            step = float(st["step_size"])
            if imm.shape != (dim, dim) or not np.all(np.isfinite(imm)) or not (math.isfinite(step) and step > 0):
                return False, f"{name}: bad step size / inverse mass matrix"
            if not np.all(np.linalg.eigvalsh(0.5 * (imm + imm.T)) > 0):
                return False, f"{name}: inverse mass matrix not positive definite"
            if not math.isfinite(log_density(constrained, *lanes[i])):
                return False, f"{name}: warm point has non-finite density on the new data"
    except (AttributeError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return False, f"malformed warm state: {exc}"
    return True, "ok"


def run_vmapped(Xp, Ys, Vs, M, *, seeds, warmup: int, samples: int, warm_inputs=None):
    """Run the vmapped NUTS program on padded data (one lane per row of Ys / Vs).
    Returns (raw constrained samples {site: [L, S, ...]}, divergences [L], final states, init tries
    [L]). Raises RuntimeError when a lane finds no valid initial point."""
    _check_env()
    Xp, M = np.asarray(Xp, dtype=np.float64), np.asarray(M, dtype=np.float64)
    Ys, Vs = np.asarray(Ys, dtype=np.float64), np.asarray(Vs, dtype=np.float64)
    L, d = Ys.shape[0], Xp.shape[1]
    warm = warm_inputs is not None
    fn, names, shapes = _vmapped_sampler(Xp.shape[0], d, warmup, samples, warm)
    if warm:
        z_w, step_w, imm_w = warm_inputs
    else:  # placeholders (unused by the cold lane)
        z_w = {nm: jnp.zeros((L, *shapes[nm])) for nm in names}
        step_w = jnp.ones(L)
        imm_w = jnp.broadcast_to(jnp.eye(3 + d), (L, 3 + d, 3 + d))
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    raw, diverging, final, tries = fn(keys, jnp.asarray(Xp), jnp.asarray(Ys), jnp.asarray(Vs), jnp.asarray(M),
                                      z_w, step_w, imm_w)
    raw = jax.block_until_ready(raw)
    tries = np.asarray(tries)
    failed = [i for i in range(L) if int(tries[i]) != -1]
    if failed:
        raise RuntimeError(f"NUTS found no valid initial point (100 draws) for lane(s) {failed}")
    return raw, np.asarray(diverging).sum(axis=1), final, tries


def _warm_inputs(warm_start: dict, names, shapes, L: int):
    sts = [warm_start[n] for n in OBJECTIVE_NAMES]
    z_w = {nm: jnp.asarray(np.stack([_unconstrained(nm, st["constrained"][nm]) for st in sts]).reshape((L, *shapes[nm])))
           for nm in names}
    step_w = jnp.asarray([float(st["step_size"]) for st in sts])
    imm_w = jnp.asarray(np.stack([np.asarray(st["inverse_mass_matrix"], dtype=np.float64) for st in sts]))
    return z_w, step_w, imm_w


def saas_multi_numpyro(x, y, v, *, seed: int = 0, warmup: int = 512, samples: int = 256,
                       thinning: int = 16, warm_start: dict | None = None,
                       warm_warmup: int = 128) -> list[SaasFullyBayesianSingleTaskGP]:
    """One SAAS GP per column of y (columns = OBJECTIVE_NAMES), all fitted in one vmapped NUTS
    program; column i uses PRNGKey(seed + i). Shared X (no input transform), so the lanes differ only
    in (Y, Yvar). warm_start (speed option 2b): a state from ``warm_state(models)`` /
    ``load_warm_state``; when ``warm_state_valid`` accepts it, the fit uses warmup=warm_warmup with
    the stored z / step size / inverse mass matrix; otherwise it falls back to the cold fit.
    Each model's ``numpyro_info`` records ``warm`` (bool), ``warm_reason``, divergences, init tries
    and the state for the next round (generation, cold_fit_n: aging rule)."""
    _check_env()
    _check_nuts(warmup, samples, thinning)
    models = [_new_model(x, y[:, i : i + 1], v[:, i : i + 1]) for i in range(y.shape[1])]
    lanes = [_padded(m) for m in models]
    Xp, M = lanes[0][0], lanes[0][3]
    if any(not np.array_equal(lane[0], Xp) for lane in lanes):
        raise ValueError("vmapped SAAS fit needs the same transformed X for every objective")
    d, L, n = Xp.shape[1], len(models), int(x.shape[0])
    warm, reason = (False, "not requested") if warm_start is None else warm_state_valid(warm_start, d, lanes)
    if warm and L != len(OBJECTIVE_NAMES):
        warm, reason = False, "warm start needs the 3 objectives"
    W = warm_warmup if warm else warmup
    _check_nuts(W, samples, thinning)
    compiled = len(_SAMPLERS)
    _, names, shapes = _vmapped_sampler(Xp.shape[0], d, W, samples, warm)
    t = time.perf_counter()
    raw, div, final, tries = run_vmapped(
        Xp, np.stack([lane[1] for lane in lanes]), np.stack([lane[2] for lane in lanes]), M,
        seeds=[seed + i for i in range(L)], warmup=W, samples=samples,
        warm_inputs=_warm_inputs(warm_start, names, shapes, L) if warm else None)
    info = {"seconds": time.perf_counter() - t, "new_compile": len(_SAMPLERS) > compiled, "mode": "vmap",
            "n": n, "n_pad": int(Xp.shape[0]), "nuts": [W, samples, thinning],
            "warm": warm, "warm_reason": reason}
    _z_last, step_last, imm_last = final  # last z == last raw sample (stored constrained below)
    out = []
    for i, m in enumerate(models):
        name = OBJECTIVE_NAMES[i] if L == len(OBJECTIVE_NAMES) else f"column{i}"
        lane_raw = {k: np.asarray(raw[k][i]) for k in SITES}
        check_finite_samples(lane_raw, name)
        prev = warm_start[name] if warm else None
        state = {"objective": name, "n": n, "d": d, "sites": list(names),
                 "generation": int(prev["generation"]) + 1 if warm else 0,
                 "cold_fit_n": int(prev["cold_fit_n"]) if warm else n,
                 "constrained": {nm: lane_raw[nm][-1].tolist() for nm in SITES},
                 "step_size": float(step_last[i]), "inverse_mass_matrix": np.asarray(imm_last[i]).tolist()}
        out.append(_load(m, lane_raw, thinning, {**info, "divergences": int(div[i]), "init_tries": int(tries[i]),
                                                 "state": state}))
    return out


def warm_state(models) -> dict:
    """{objective: final NUTS state} of a vmapped fit (ModelListGP or list), for the next round."""
    members = getattr(models, "models", models)
    return {name: m.numpyro_info["state"] for name, m in zip(OBJECTIVE_NAMES, members)}


def save_warm_state(path: str | os.PathLike, models, *, meta: dict) -> Path:
    """Persist the warm state next to a round's model artifacts. meta (required) holds the
    provenance keys PROVENANCE_KEYS (see warm_meta); JSON, temp file + atomic replace."""
    return save_states(path, warm_state(models), meta=meta)


def load_warm_state(path: str | os.PathLike, *, expect: dict) -> dict | None:
    """{objective: state} or None (-> cold fit). expect = warm_meta(...) plus "n" (current n);
    every provenance key must match and the stored n must be <= expect["n"]."""
    return load_states(path, expect=expect)


def fit_saas_numpyro(
    codec,
    observations,
    objective,
    *,
    seed: int = 0,
    warmup: int = 512,
    samples: int = 256,
    thinning: int = 16,
    on_objective: Callable[[int, str], None] | None = None,
    parallel: bool = True,
    warm_start: dict | None = None,
    warm_warmup: int = 128,
) -> ModelListGP:
    """Three independent SAAS GPs (quality, tokens, latency) on u = codec.encode(config); objective
    i uses NUTS seed ``seed + i`` (as latent_plan_surrogate._fit_saas).

    parallel=True (default, user 2026-11-04): the three NUTS chains run in one vmapped JAX program
    (saas_multi_numpyro); on_objective is then called for all three objectives immediately before
    the joint fit. parallel=False: one MCMC per objective, on_objective before each.
    warm_start (parallel only): {objective: state} from load_warm_state / warm_state of the previous
    round; a valid state fits with warmup=warm_warmup (default 128), otherwise the cold `warmup`.
    The returned ModelListGP carries ``numpyro_summary``: mode, warm, warm_reason, nuts, seconds and
    per-objective divergences / init tries."""
    _check_env()
    if len({o.panel_identity for o in observations}) != 1:
        raise ValueError("GP observations must share the exact TRAIN panel and protocol")
    if len({codec.key(o.config) for o in observations}) != len(observations):
        raise ValueError("Repeated complete config observations")
    _check_nuts(warmup, samples, thinning)
    x = torch.tensor([codec.encode(o.config) for o in observations], dtype=torch.double)
    y, noise = objective.tensors(observations)
    nuts = {"warmup": warmup, "samples": samples, "thinning": thinning}
    if parallel:
        for i, name in enumerate(OBJECTIVE_NAMES):
            if on_objective is not None:
                on_objective(i, name)
        models = saas_multi_numpyro(x, y[:, :3], noise[:, :3], seed=seed, warm_start=warm_start,
                                    warm_warmup=warm_warmup, **nuts)
    else:
        if warm_start is not None:
            raise ValueError("warm_start needs parallel=True")
        models = []
        for i, name in enumerate(OBJECTIVE_NAMES):
            if on_objective is not None:
                on_objective(i, name)
            models.append(saas_single_numpyro(x, y[:, i : i + 1], noise[:, i : i + 1], seed=seed + i, **nuts))
    model = ModelListGP(*models)
    model.eval()
    first = models[0].numpyro_info
    model.numpyro_summary = {
        "mode": first["mode"], "warm": bool(first.get("warm", False)), "warm_reason": first.get("warm_reason"),
        "nuts": first["nuts"], "seconds": float(sum(m.numpyro_info["seconds"] for m in models))
        if first["mode"] == "sequential" else float(first["seconds"]),
        "divergences": {n: int(m.numpyro_info["divergences"]) for n, m in zip(OBJECTIVE_NAMES, models)},
        "init_tries": {n: m.numpyro_info.get("init_tries") for n, m in zip(OBJECTIVE_NAMES, models)},
    }
    return model
