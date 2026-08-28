# EVOLVE-BLOCK-START
import jax, jax.numpy as jnp, optax, numpy as np

N, STEPS, DX = 600, 40000, 0.5 / 600

def objective(f):
    f = jax.nn.relu(f)
    I = jnp.maximum(jnp.sum(f) * DX, 1e-9)
    p = jnp.pad(f, (0, N))
    c = jnp.fft.ifft(jnp.fft.fft(p)**2).real * DX
    return jnp.max(c) / I**2

def run():
    sched = optax.warmup_cosine_decay_schedule(0.0, 0.006, 2000, 38000, 5e-7)
    opt = optax.adam(learning_rate=sched)
    f = jnp.zeros(N).at[150:450].set(1.0)
    f += 0.05 * jax.random.uniform(jax.random.PRNGKey(42), (N,))
    st = opt.init(f)
    @jax.jit
    def step(f, st):
        l, g = jax.value_and_grad(objective)(f)
        u, st = opt.update(g, st, f)
        return optax.apply_updates(f, u), st, l
    loss = jnp.inf
    for s in range(STEPS):
        f, st, loss = step(f, st)
        if s % 10000 == 0:
            print(f"Step {s:5d} | C1 = {loss:.8f}")
    print(f"Final C1: {float(loss):.8f}")
    return np.array(jax.nn.relu(f)), float(loss), loss, N

# EVOLVE-BLOCK-END
