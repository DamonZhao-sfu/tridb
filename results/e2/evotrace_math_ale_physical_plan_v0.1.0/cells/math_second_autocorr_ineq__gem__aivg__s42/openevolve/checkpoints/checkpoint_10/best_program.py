# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np

N, LR, STEPS, WARMUP = 50, 0.01, 15000, 1000

def _objective(f):
    f = jax.nn.relu(f)
    conv = jnp.fft.ifft(jnp.fft.fft(jnp.pad(f, (0, N))) ** 2).real
    n = len(conv)
    h = 1.0 / (n + 1)
    y = jnp.concatenate([jnp.array([0.0]), conv, jnp.array([0.0])])
    y1, y2 = y[:-1], y[1:]
    l2_sq = jnp.sum((h / 3) * (y1**2 + y1 * y2 + y2**2))
    n1 = jnp.sum(jnp.abs(conv)) / (n + 1)
    ninf = jnp.max(jnp.abs(conv))
    return -(l2_sq / (n1 * ninf))

def _smooth_init(key, n):
    """Generate a smooth bump-function-like initial guess."""
    x = jnp.linspace(0, 1, n)
    # Gaussian bump centered at 0.5 with some width
    sigma = 0.2
    f = jnp.exp(-0.5 * ((x - 0.5) / sigma) ** 2)
    # Add small random perturbation
    noise = jax.random.normal(key, (n,)) * 0.05
    return jnp.maximum(f + noise, 0.0)

def run():
    sched = optax.warmup_cosine_decay_schedule(
        0.0, LR, WARMUP, STEPS - WARMUP, LR * 1e-4)
    opt = optax.adam(sched)

    @jax.jit
    def step(f, state):
        loss, grads = jax.value_and_grad(_objective)(f)
        updates, state = opt.update(grads, state, f)
        return optax.apply_updates(f, updates), state, loss

    # Multi-start: try a few different initializations
    best_f, best_c2 = None, -1e10
    for seed in [42, 123, 7]:
        key = jax.random.PRNGKey(seed)
        f = _smooth_init(key, N)
        state = opt.init(f)

        for s in range(STEPS):
            f, state, loss = step(f, state)

        c2 = -_objective(f)
        if c2 > best_c2:
            best_c2 = c2
            best_f = f

    print(f"Final C2 lower bound found: {best_c2:.8f}")
    return np.array(jax.nn.relu(best_f)), float(best_c2), float(-best_c2), N


# EVOLVE-BLOCK-END
