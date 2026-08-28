# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import numpy as np

def _c3_ratio(f, dx):
    """Compute C3 = max|f*f| / (int f)^2."""
    N = len(f)
    integral = jnp.sum(f) * dx
    eps = 1e-9
    int_sq = jnp.maximum(integral**2, eps)
    # Linear convolution via FFT
    padded = jnp.pad(f, (0, N))
    F = jnp.fft.fft(padded)
    conv = jnp.fft.ifft(F * F).real * dx
    return jnp.max(jnp.abs(conv)) / int_sq

def run():
    N = 200
    dx = 0.5 / N
    key = jax.random.PRNGKey(0)
    x = jnp.linspace(-0.25, 0.25, N)
    
    # Build initial function: Gaussian bump + random sinusoids for richer exploration
    subkey1, subkey2 = jax.random.split(key)
    f0 = jnp.exp(-15 * x**2)
    # Add 5 random sinusoids
    for i in range(5):
        subkey, subkey2 = jax.random.split(subkey2)
        amp = jax.random.uniform(subkey, (), minval=0.1, maxval=0.5)
        freq = jax.random.uniform(subkey2, (), minval=2.0, maxval=20.0)
        phase = jax.random.uniform(subkey2, (), minval=0.0, maxval=2 * jnp.pi)
        f0 = f0 + amp * jnp.sin(freq * x + phase)
    
    # Normalize: ensure integral is 1
    integral = jnp.sum(f0) * dx
    f0 = f0 / integral
    
    # Gradient descent with cosine-decay learning rate
    lr0 = 0.05
    lr_end = 0.001
    num_steps = 5000
    opt = jax.jit(lambda f, lr: f - lr * jax.grad(_c3_ratio)(f, dx))
    
    f = f0
    best_c3 = _c3_ratio(f, dx)
    for step in range(num_steps):
        t = step / num_steps
        lr = lr_end + 0.5 * (lr0 - lr_end) * (1 + jnp.cos(jnp.pi * t))
        f = opt(f, lr)
        c3 = _c3_ratio(f, dx)
        if c3 < best_c3:
            best_c3 = c3
        if step % 1000 == 0:
            print(f"Step {step}: C3={c3:.6f}, best={best_c3:.6f}")
    
    f_np = np.array(f)
    return f_np, float(best_c3), float(best_c3), N
# EVOLVE-BLOCK-END
