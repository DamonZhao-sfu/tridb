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
    l2_sq = h * (jnp.sum(conv**2) + 0.5 * (conv[0]**2 + conv[-1]**2))
    n1 = jnp.sum(jnp.abs(conv)) / (n + 1)
    ninf = jnp.max(jnp.abs(conv))
    return -(l2_sq / (n1 * ninf))

def run():
    sched = optax.warmup_cosine_decay_schedule(
        0.0, LR, WARMUP, STEPS - WARMUP, LR * 1e-4)
    opt = optax.adam(sched)
    key = jax.random.PRNGKey(42)
    x = jnp.linspace(-1.0, 1.0, N)
    f = 1.0 - x ** 2
    state = opt.init(f)

    @jax.jit
    def step(f, state):
        loss, grads = jax.value_and_grad(_objective)(f)
        updates, state = opt.update(grads, state, f)
        return optax.apply_updates(f, updates), state, loss

    for s in range(STEPS):
        f, state, loss = step(f, state)
        if s % 1000 == 0 or s == STEPS - 1:
            print(f"Step {s:5d} | C2 ≈ {-loss:.8f}")

    c2 = -_objective(f)
    print(f"Final C2 lower bound found: {c2:.8f}")
    return np.array(jax.nn.relu(f)), float(c2), float(-c2), N


# EVOLVE-BLOCK-END
