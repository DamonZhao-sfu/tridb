import jax
import jax.numpy as jnp
import optax
import numpy as np
from scipy.special import hermite

# EVOLVE-BLOCK-START
DEGREES = [0, 4, 8, 12]
MAX_DEG = 12
HERMITE_POLYS = [hermite(d) for d in DEGREES]
H0_VALS = np.array([p(0) for p in HERMITE_POLYS])
X_GRID = jnp.linspace(0.0, 10.0, 2000)
W = 1.0 + X_GRID / (X_GRID[-1] + 1e-12)

# Precompute basis in JAX
_basis = []
for poly in HERMITE_POLYS:
    pad = MAX_DEG - poly.order
    _basis.append(jnp.array(np.pad(poly.coef, (pad, 0))))
B = jnp.stack(_basis)
H0J = jnp.array(H0_VALS)

# Precompute coefficient-to-polynomial mapping for fast numpy assembly
COEF_MAP = np.zeros((len(DEGREES), MAX_DEG + 1))
for i, poly in enumerate(HERMITE_POLYS):
    pad = MAX_DEG - poly.order
    COEF_MAP[i, pad:] = poly.coef


def _obj(params):
    c_o, lc = params[:-1], params[-1]
    cl = jnp.exp(lc)
    c0 = -(jnp.dot(c_o, H0J[1:-1]) + cl * H0J[-1]) / H0J[0]
    hc = jnp.concatenate([jnp.array([c0]), c_o, jnp.array([cl])])
    pc = jnp.sum(hc[:, None] * B, axis=0)
    pv = jnp.polyval(pc, X_GRID)
    return jnp.sum(W * jax.nn.relu(-pv))


def _step(params, state, opt):
    loss, g = jax.value_and_grad(_obj)(params)
    u, state = opt.update(g, state, params)
    return optax.apply_updates(params, u), state


def _trial(key):
    base = jnp.array([-0.0115851, -8.9216e-05, np.log(1e-6)], jnp.float32)
    p = base + jax.random.normal(key, (3,)) * 1e-3
    opt = optax.adam(0.001)
    st = opt.init(p)
    jstep = jax.jit(_step, static_argnums=(2,))
    for _ in range(5000):
        p, st = jstep(p, st, opt)
    return np.array(p)


def _c4(params):
    c_o, lc = params[:-1], params[-1]
    cl = np.exp(lc)
    c0 = -(np.dot(c_o, H0_VALS[1:-1]) + cl * H0_VALS[-1]) / H0_VALS[0]
    hc = np.concatenate([[c0], c_o, [cl]])
    pc = hc @ COEF_MAP
    if pc[0] < 0:
        pc, hc = -pc, -hc
    Q, R = np.polydiv(pc, [1.0, 0.0, 0.0])
    if np.max(np.abs(R)) > 1e-10:
        return None, None, None
    roots = np.roots(Q)
    rp = roots[np.isreal(roots) & (roots.real > 0)].real
    if rp.size == 0:
        return None, None, None
    rmax = float(rp.max())
    return hc, rmax**2 / (2 * np.pi), rmax


def run():
    key = jax.random.PRNGKey(42)
    best_c4, best_coeffs, best_r = float('inf'), None, None
    for i in range(10):
        key, k = jax.random.split(key)
        p = _trial(k)
        hc, c4, rm = _c4(p)
        if c4 is not None and c4 < best_c4:
            best_c4, best_coeffs, best_r = c4, hc, rm
    if best_coeffs is None:
        raise RuntimeError("No valid solution found")
    print(f"Best Hermite coeffs: {best_coeffs}")
    print(f"Best r_max: {best_r:.8f}")
    print(f"Best C4 bound: {best_c4:.8f}")
    return best_coeffs, best_c4, best_r

# EVOLVE-BLOCK-END
