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
    # Start with a simple bump-like function
    x = jnp.linspace(-0.25, 0.25, N)
    f0 = jnp.exp(-10 * x**2) - 0.5 * jnp.exp(-10 * (x - 0.1)**2)
    f0 = f0 - jnp.mean(f0) * dx / (jnp.sum(f0**2) * dx)  # normalize integral
    
    # Use simple gradient descent with decaying LR
    lr0 = 0.01
    opt = jax.jit(lambda f, lr: f - lr * jax.grad(_c3_ratio)(f, dx))
    
    f = f0
    best_c3 = _c3_ratio(f, dx)
    for step in range(3000):
        lr = lr0 / (1 + step * 0.001)
        f = opt(f, lr)
        c3 = _c3_ratio(f, dx)
        if c3 < best_c3:
            best_c3 = c3
        if step % 500 == 0:
            print(f"Step {step}: C3={c3:.6f}, best={best_c3:.6f}")
    
    f_np = np.array(f)
    return f_np, float(best_c3), float(best_c3), N
# EVOLVE-BLOCK-END
