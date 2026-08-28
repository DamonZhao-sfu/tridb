# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np

N = 600
dx = 0.5 / N
lr = 0.01
steps = 40000
warmup = 2000

def objective(f):
    fn = jax.nn.relu(f)
    int_f = jnp.sum(fn) * dx
    int_safe = jnp.maximum(int_f, 1e-9)
    padded = jnp.pad(fn, (0, N))
    fft_f = jnp.fft.fft(padded)
    conv = jnp.fft.ifft(fft_f * fft_f).real * dx
    return jnp.max(conv) / int_safe**2

def train_step(f, opt_state, optimizer):
    loss, grads = jax.value_and_grad(objective)(f)
    updates, opt_state = optimizer.update(grads, opt_state, f)
    return optax.apply_updates(f, updates), opt_state, loss

def run():
    sched = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup, steps - warmup, lr * 1e-4)
    optimizer = optax.adam(learning_rate=sched)
    key = jax.random.PRNGKey(42)
    x = jnp.linspace(-0.25, 0.25, N)
    f = jnp.exp(-4.0 * x**2)
    f += 0.01 * jax.random.uniform(key, (N,))
    opt_state = optimizer.init(f)
    ts = jax.jit(lambda f, s: train_step(f, s, optimizer))
    loss = jnp.inf
    for step in range(steps):
        f, opt_state, loss = ts(f, opt_state)
    return np.array(jax.nn.relu(f)), float(loss), loss, N
# EVOLVE-BLOCK-END
