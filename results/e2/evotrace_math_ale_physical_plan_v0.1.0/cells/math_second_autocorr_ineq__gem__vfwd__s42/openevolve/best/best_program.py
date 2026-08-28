# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np

N, LR, STEPS = 50, 0.01, 15000

def _obj(f):
    f = jax.nn.relu(f)
    conv = jnp.fft.ifft(jnp.fft.fft(jnp.pad(f, (0, N)))**2).real
    m = len(conv)
    y = jnp.concatenate([jnp.zeros(1), conv, jnp.zeros(1)])
    y1, y2 = y[:-1], y[1:]
    l2 = jnp.sum((y1**2 + y1*y2 + y2**2))/(3*(m+1))
    n1 = jnp.sum(jnp.abs(conv))/(m+1)
    ninf = jnp.max(jnp.abs(conv))
    return -(l2/(n1*ninf))

def run():
    sched = optax.warmup_cosine_decay_schedule(0.0, LR, 500, STEPS-500, LR*1e-4)
    opt = optax.adam(sched)
    x = jnp.linspace(0, 2*jnp.pi, N)
    f = jnp.sin(x) + jnp.cos(2*x)*0.5
    state = opt.init(f)

    @jax.jit
    def step(f, state):
        loss, g = jax.value_and_grad(_obj)(f)
        u, state = opt.update(g, state, f)
        return optax.apply_updates(f, u), state, loss

    for _ in range(STEPS):
        f, state, _ = step(f, state)

    c2 = -_obj(f)
    return np.array(jax.nn.relu(f)), float(c2), float(-c2), N


# EVOLVE-BLOCK-END
