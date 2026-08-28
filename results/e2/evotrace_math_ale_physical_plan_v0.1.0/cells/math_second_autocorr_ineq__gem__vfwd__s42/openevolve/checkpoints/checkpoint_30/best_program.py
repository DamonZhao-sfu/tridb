# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np

N = 50
STEPS = 10000
WARMUP = 400
LR = 0.025

def objective(f):
    f = jax.nn.relu(f)
    padded = jnp.pad(f, (0, N))
    conv = jnp.fft.ifft(jnp.fft.fft(padded)**2).real
    h = 1.0 / (len(conv) + 1)
    y = jnp.concatenate([jnp.array([0.0]), conv, jnp.array([0.0])])
    y1, y2 = y[:-1], y[1:]
    l2_sq = jnp.sum((h/3)*(y1**2 + y1*y2 + y2**2))
    l1 = jnp.sum(jnp.abs(conv)) / (len(conv) + 1)
    linf = jnp.max(jnp.abs(conv))
    return -(l2_sq / (l1 * linf + 1e-8))

def run():
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, WARMUP, STEPS - WARMUP, LR*1e-3)
    opt = optax.adam(sched)
    f = jax.random.uniform(jax.random.PRNGKey(42), (N,))
    state = opt.init(f)
    def step(f, s):
        g = jax.grad(objective)(f)
        updates, new_s = opt.update(g, s)
        new_f = optax.apply_updates(f, updates)
        return new_f, new_s
    step_fn = jax.jit(step)
    for _ in range(STEPS):
        f, state = step_fn(f, state)
    c2 = -objective(f)
    return np.array(jax.nn.relu(f)), float(c2), float(-c2), N

# EVOLVE-BLOCK-END
