import numpy as np
import time
from jax import numpy as jnp
from jax.config import config
from jax.random import PRNGKey
from jax.random import normal
# from scipy.sparse.linalg import cg as cg_solver
# from gr.CGSolver import run_mbcg_log_unclipped as cg_solver
from neural_pde.CGSolver import run_mbcg_unclipped as cg_solver
from gr.Preconditioners import MatrixOp
from gr.Preconditioners import IdentityPrecond
from gr.Preconditioners import JacobiPrecondLO
from gr.Preconditioners import NysPrecondLO
from utils.general import print_time_taken

config.update("jax_enable_x64", True)

key = PRNGKey(seed=21)
dtype = jnp.float64
# dtype = jnp.float32
cgtol = 1.e-6
maxcgiters = int(1 * 1.e4)
N = int(1 * 1.e3)
prcond_num = -1
rank = 100
mu = 1.e-8

vec = np.array(normal(key, shape=(N, ), dtype=dtype))
L = normal(key, shape=(N, N), dtype=dtype)
matrix = np.array(L.T @ L + mu * jnp.eye(N, dtype=dtype))
A_fn = MatrixOp(matrix)
precond_classes = (IdentityPrecond, JacobiPrecondLO, NysPrecondLO)
precond = precond_classes[prcond_num](A=A_fn, rank=rank, mu=mu, dtype=matrix.dtype,
                                      shape=matrix.shape)
x0 = jnp.zeros_like(vec)
t0 = time.time()
# v, _ = cg_solver(A=A_fn, b=vec, tol=cgtol, M=precond, maxiter=maxcgiters)
v = cg_solver(A_fn, b=vec, max_iters=maxcgiters, tolerance=cgtol, preconditioner=precond, x0=x0)
t1 = time.time()
print_time_taken(delta=t1 - t0)

print('Test CG solve')
diff = jnp.linalg.norm(matrix @ v - vec)
print(f'Diff {diff:1.3e}')

sol = jnp.linalg.solve(matrix, vec)
diff = jnp.linalg.norm(matrix @ sol - vec)
print(f'Diff {diff:1.3e}')
