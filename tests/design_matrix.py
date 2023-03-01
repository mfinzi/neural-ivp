from jax import numpy as jnp
import time
from jax import jit
from neural_pde.Preconditioners import NysPrecondLO
from neural_pde.CGSolver import get_cg_solver
from neural_pde.CGSolver import run_mbcg_unclipped
from jax.scipy.sparse.linalg import cg as cg_solver
from utils.general import print_time_taken
import numpy as np

diag = jnp.array([10., 2., 2.3, 2.2, 1., 1.1, 1.3, 1.2, 0.5, 0.1])
cg_tol = 1.e-5
maxiter = int(5.e2)
dtype = jnp.float32
mu = jnp.array(1.e-3, dtype=dtype)
rank = 7
N = diag.shape[0]
np.random.seed(seed=21)
L = np.zeros(shape=(N, N))
for i in range(N):
    for j in range(N):
        if i > j:
            L[i, j] = np.random.normal(size=1)
        if i == j:
            L[i, i] = diag[i]
A = jnp.array(L.T @ L + 0.001 + np.eye(N))
# A = jnp.array(L)
# cond = jnp.linalg.cond(A)
# print(f'Cond {cond:1.2e}')
vec = jnp.ones(shape=N)

precond = NysPrecondLO(lambda x: A @ x, rank=rank, mu=mu, dtype=dtype, shape=(N, N))
precond.construct_precond()
precond = jit(precond.precond)
# precond = precond.precond
for _ in range(5):
    t0 = time.time()
    # out = cg_solver(lambda x: A @ x, vec, maxiter=maxiter, tol=cg_tol)[0]
    out = get_cg_solver(lambda x: A @ x, vec, maxiter, cg_tol)
    # out = run_mbcg_unclipped(
    #         lambda x: A @ x, b=vec, x0=jnp.zeros_like(vec),
    #         preconditioner=precond, max_iters=maxiter, tolerance=cg_tol)
    t1 = time.time()
    print_time_taken(t1 - t0)
    res = A @ out - vec
    print(f'Residual {jnp.linalg.norm(res):1.3e}')
