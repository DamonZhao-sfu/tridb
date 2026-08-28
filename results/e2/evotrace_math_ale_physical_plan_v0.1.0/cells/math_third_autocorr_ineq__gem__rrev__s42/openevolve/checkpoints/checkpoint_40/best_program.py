# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np

N = 400
WIDTH = 0.5
DX = WIDTH / N
NUM_STEPS = 15000
WARMUP = 1500
LR = 0.01

def autoconv_max(f):
    """Compute max |f*f| using FFT-based autoconvolution."""
    padded = jnp.pad(f, (0, N))
    F = jnp.fft.fft(padded)
    conv = jnp.fft.ifft(F * F).real
    return jnp.max(jnp.abs(conv * DX))

def objective(f):
    """C3 ratio: max|f*f| / (integral f)^2. We minimize this."""
    integral = jnp.sum(f) * DX
    eps = 1e-8
    denom = jnp.maximum(integral ** 2, eps)
    return autoconv_max(f) / denom

def make_train_step(optimizer):
    """Create a jit-able train step with optimizer captured in closure."""
    @jax.jit
    def _step(f, opt_state):
        loss, grads = jax.value_and_grad(objective)(f)
        updates, opt_state = optimizer.update(grads, opt_state, f)
        f = optax.apply_updates(f, updates)
        return f, opt_state, loss
    return _step

def run():
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR, warmup_steps=WARMUP,
        decay_steps=NUM_STEPS - WARMUP, end_value=LR * 1e-3
    )
    optimizer = optax.adam(learning_rate=schedule)

    # Initialize with a smooth positive bump (Gaussian-like)
    x = jnp.linspace(-WIDTH / 2, WIDTH / 2, N)
    f0 = jnp.exp(-20.0 * x ** 2)
    # Normalize so integral = 1
    f0 = f0 / (jnp.sum(f0) * DX)

    opt_state = optimizer.init(f0)
    train_step_jit = make_train_step(optimizer)

    f, opt_state, loss = f0, opt_state, jnp.inf
    for step in range(NUM_STEPS):
        f, opt_state, loss = train_step_jit(f, opt_state)
        if step % 5000 == 0 or step == NUM_STEPS - 1:
            print(f"Step {step:5d} | C3 = {loss:.8f}")

    f_np = np.array(f)
    return f_np, float(loss), float(loss), N

# EVOLVE-BLOCK-END
