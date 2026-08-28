# EVOLVE-BLOCK-START
import jax, jax.numpy as jnp, optax, numpy as np
N, LR, STEPS, WARMUP = 50, 0.01, 15000, 1000

def _obj(f):
    f = jax.nn.relu(f)
    c = jnp.fft.ifft(jnp.fft.fft(jnp.pad(f,(0,N)))**2).real
    n = len(c); h = 1.0/(n+1)
    y = jnp.concatenate([jnp.array([0.0]),c,jnp.array([0.0])])
    y1,y2 = y[:-1],y[1:]
    l2 = jnp.sum(h/3*(y1**2+y1*y2+y2**2))
    n1 = jnp.sum(jnp.abs(c))/(n+1)
    ninf = jnp.max(jnp.abs(c))
    return -(l2/(n1*ninf+1e-12))

def run():
    opt = optax.adam(optax.warmup_cosine_decay_schedule(0.0,LR,WARMUP,STEPS-WARMUP,LR*1e-5))
    f = 4*jnp.linspace(0,1,N)*(1-jnp.linspace(0,1,N))
    st = opt.init(f)
    @jax.jit
    def step(f,st):
        _,g = jax.value_and_grad(_obj)(f)
        u,st = opt.update(g,st,f)
        return optax.apply_updates(f,u),st
    for _ in range(STEPS): f,st = step(f,st)
    c2 = -_obj(f)
    return np.array(jax.nn.relu(f)),float(c2),float(-c2),N
# EVOLVE-BLOCK-END
