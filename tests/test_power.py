import numpy as np
import time
import haiku as hk
from jax import numpy as jnp
from jax import grad
from jax import jacfwd
from jax import jit
from jax.nn import swish
from jax.random import PRNGKey
from neural_pde.neural_ivp import M_estimate
from gr.linalg import get_max_eigenvalue
from utils.general import print_time_taken

max_iters, tolerance = 10_000, 1e-10
reg, bs = 0, 0
key = PRNGKey(seed=21)
N = 200
dtype = jnp.float32
xgrid = np.linspace(-1, 1, N + 2)[1:-1]
xygrid = np.stack(np.meshgrid(xgrid, xgrid), 0).reshape(2, N * N).T


def nn(x):
    k = 20
    act = swish
    y = x
    x_norm = jnp.linalg.norm(x)[None]
    y_norm = x_norm
    y = jnp.concatenate([y, y_norm])
    mlp = hk.Sequential([hk.Linear(k), act, hk.Linear(k), act])
    mlp_deriv = hk.Sequential([hk.Linear(k), act, hk.Linear(k), act])
    bound = (1. - x[0]**2.) * (1. - x[1]**2.)
    out1 = hk.Linear(1)(mlp(y)) * bound
    out2 = hk.Linear(1)(mlp_deriv(y)) * bound
    uut = jnp.concatenate([out1, out2])
    return uut


nn = hk.without_apply_rng(hk.transform(nn))
params = nn.init(PRNGKey(seed=0), jnp.zeros(shape=(2, )))


def sampler(*_):
    return jnp.array(xygrid, dtype=dtype)


def pde_f(nnfn, x):
    du = grad(lambda x: nnfn(x)[0])
    laplacian_u = jnp.trace(jacfwd(du)(x))
    ut = nnfn(x)[1]
    return jnp.array([ut, 0.1 * laplacian_u])


M = M_estimate(nn.apply, params, sampler, pde_f, key, N, bs, reg)
fn = jit(get_max_eigenvalue, static_argnums=(0, ))
t0 = time.time()
max_eigen, iters = fn(M, key, max_iters, tolerance)
t1 = time.time()
print_time_taken(t1 - t0)
print(f"Max eigen: {max_eigen:1.5e}")
print(f"Iters: {iters:5d}")

t0 = time.time()
I_parts = jnp.array_split(jnp.eye(M.shape[0]), 10, -1)
M_dense = jnp.concatenate([M @ I for I in I_parts], -1)
eigen, _ = jnp.linalg.eigh(M_dense)
actual = eigen[-1]
t1 = time.time()
print_time_taken(t1 - t0)
print(f"Max eigen: {eigen[-1]:1.5e}")

diff = jnp.linalg.norm(actual - max_eigen)
print(f"Abs Diff: {diff:1.3e}")
rel_diff = diff / jnp.linalg.norm(actual)
print(f"Rel Diff: {rel_diff:1.3e}")
