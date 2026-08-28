# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.special import hermite

def _objective(params, hb, h0, xg):
    co, lcl = params[:-1], params[-1]
    cl = jnp.exp(lcl)
    c0 = -(jnp.sum(co * h0[1:-1]) + cl * h0[-1]) / h0[0]
    hc = jnp.concatenate([jnp.array([c0]), co, jnp.array([cl])])
    pc = jnp.sum(hc[:, None] * hb, axis=0)
    pv = jnp.polyval(pc, xg)
    # Simpler weight: emphasize region near zero where C4 is determined
    w = 1.0 + xg
    return jnp.sum(w * jax.nn.relu(-pv)) + 1e-8 * jnp.sum(jnp.square(co))

def _step(params, os_, opt, hb, h0, xg):
    l, g = jax.value_and_grad(_objective)(params, hb, h0, xg)
    u, os_ = opt.update(g, os_, params)
    return optax.apply_updates(params, u), os_, l

def _run(key, opt, hb, h0, xg, steps):
    base = jnp.array([-0.01158510802599293, -8.921606035407065e-05, np.log(1e-6)], dtype=jnp.float32)
    p = base + jax.random.normal(key, (3,)) * 5e-3
    os_ = opt.init(p)
    js = jax.jit(_step, static_argnums=(2,))
    for _ in range(steps):
        p, os_, _ = js(p, os_, opt, hb, h0, xg)
    return p

def _c4_from(hc):
    degs = [0, 4, 8, 12]
    hp = [hermite(d) for d in degs]
    pc = np.zeros(13)
    for i, c in enumerate(hc):
        pad = 12 - hp[i].order
        pc[pad:] += c * hp[i].coef
    if pc[0] < 0:
        pc = -pc
    P = np.poly1d(pc)
    Q, R = np.polydiv(P, np.poly1d([1.0, 0.0, 0.0]))
    if np.max(np.abs(R.c)) > 1e-10:
        return None, None
    roots = Q.r
    rp = roots[(np.isreal(roots)) & (roots.real > 0)].real
    if rp.size == 0:
        return None, None
    rc = np.sort(rp)
    rm = None
    for r in rc:
        e = 1e-10 * max(1.0, abs(r))
        if np.polyval(Q, r - e) * np.polyval(Q, r + e) < 0:
            rm = float(r)
    if rm is None:
        rm = float(rc[-1])
    return (rm**2) / (2 * np.pi), rm

def _get_c4(params):
    co, lcl = params[:-1], params[-1]
    cl = np.exp(lcl)
    hp = [hermite(d) for d in [0, 4, 8, 12]]
    h0 = np.array([p(0) for p in hp])
    c0 = -(np.sum(co * h0[1:-1]) + cl * h0[-1]) / h0[0]
    hc = np.concatenate([[c0], np.array(co), [cl]])
    c4, rm = _c4_from(hc)
    if c4 is None:
        return None, None, None
    return hc, c4, rm

def run():
    degs = [0, 4, 8, 12]
    hp = [hermite(d) for d in degs]
    hb = jnp.stack([jnp.array(np.pad(p.coef, (12 - p.order, 0))) for p in hp])
    h0 = jnp.array([p(0) for p in hp])
    xg = jnp.linspace(0.0, 8.0, 500)
    opt = optax.adam(0.005)
    key = jax.random.PRNGKey(42)
    best_c4, best_hc, best_rm = float("inf"), None, None
    for i in range(5):
        key, rk = jax.random.split(key)
        p = _run(rk, opt, hb, h0, xg, 10000)
        hc, c4, rm = _get_c4(np.array(p))
        if c4 is not None and c4 < best_c4:
            best_c4, best_hc, best_rm = c4, hc, rm
    if best_hc is None:
        raise RuntimeError("No valid solution found")
    print(f"Best Hermite coeffs: {best_hc}")
    print(f"Best r_max: {best_rm:.8f}")
    print(f"Best C4 bound: {best_c4:.8f}")
    return best_hc, best_c4, best_rm
# EVOLVE-BLOCK-END
