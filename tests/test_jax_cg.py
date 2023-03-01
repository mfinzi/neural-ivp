import time
from jax import numpy as jnp
from jax.config import config
from jax.random import PRNGKey
from jax.random import normal
from jax.scipy.sparse.linalg import cg as cg_solver
from neural_pde.Preconditioners import PivCholPrecondLO
from neural_pde.Preconditioners import JaxPivCholPrecond
from neural_pde.Preconditioners import CholPrecond
from neural_pde.Preconditioners import JacobiPrecond
from neural_pde.Preconditioners import IdentityJAXPrecond
from neural_pde.CGSolver import CGSolver
from neural_pde.CGSolver import run_mbcg_unclipped
config.update("jax_enable_x64", True)

# dtype = jnp.float64
dtype = jnp.float32
prcond_num = 3
cgtol = 1.e-5
maxcgiters = int(1 * 1.e3)
N = int(1 * 1.e3)
# N = int(1 * 1.e2)
# N = int(2 * 1.e1)
# rank = N
# rank = N // 10
key = PRNGKey(seed=21)
L = normal(key, shape=(N, N), dtype=dtype)
matrix = L @ L.T + 1. * jnp.eye(N, dtype=dtype)
vec = normal(key, shape=(N,), dtype=dtype)
precond_classes = (
    JaxPivCholPrecond, CholPrecond, PivCholPrecondLO, JacobiPrecond,
    IdentityJAXPrecond,
)
# precond = precond_classes[prcond_num](matrix)
# precond = precond_classes[prcond_num](rank=rank)
precond = precond_classes[prcond_num](get_diag=lambda: jnp.diag(matrix))
v, _ = cg_solver(A=lambda x: matrix @ x, b=vec, tol=cgtol, M=precond, maxiter=maxcgiters)

print('Test CG solve')
diff = jnp.linalg.norm(matrix @ v - vec)
print(f'Diff {diff:1.3e}')


def A_fn(x): return matrix @ x


tic = time.time()
cg = CGSolver(tolerance=cgtol, max_iters=maxcgiters, preconditioner=precond)
cg.set_matrix_and_probes(A_fn, vec)
solve = cg.run_mbcg_with_tracking()
# solver_fn = cg.construct_solver(A_fn, preconditioner)
# solve = solver_fn(rhs)

toc = time.time()
# print_time_taken(toc - tic)
out = A_fn(solve) - vec
aux = jnp.linalg.norm(out, axis=0)
print(f'Residual norm: {aux}')
print(f'Residual: {float(jnp.mean(aux)):1.3e}')

sol = run_mbcg_unclipped(A_fn, vec, jnp.zeros_like(vec), maxcgiters, cgtol, precond)
out = A_fn(sol) - vec
aux = jnp.linalg.norm(out, axis=0)
print(f'Residual norm: {aux}')
print(f'Residual: {float(jnp.mean(aux)):1.3e}')
