from jax import numpy as jnp
import time
from gr.Preconditioners import IdentityPrecond
from gr.Preconditioners import JacobiPrecondLO
from gr.Preconditioners import NysPrecondLO
from neural_pde.CGSolver import run_mbcg_unclipped as cg_solver
from utils.general import print_time_taken

dtype = jnp.float32
mu = 1.e-8
rank = 2
cgtol = 1.e-6
maxcgiters = int(1 * 1.e4)
precond_num = 0
lam_min = 0.5
lam_max = 10.
L = jnp.array([[1., 0., 0.], [3., lam_min, 0.], [1., 1., lam_max]])
# b = jnp.array([[1., 2., 3.], [-1., 1., 0.], [-2., 1., 4.]]).T
b = jnp.array([-1., 1., 0.])
A = L @ L.T + mu * jnp.eye(L.shape[0])


def A_fn(z):
    return A @ z


cond = (lam_max / lam_min)**2.
print(f'Cond: {cond:1.3e}')
cond = jnp.linalg.cond(A)
print(f'Estimated Cond: {cond:1.3e}')

sol = jnp.linalg.solve(A, b)
diff = jnp.linalg.norm(A @ sol - b)
print(f'Diff {diff:1.3e}')

precond_classes = (IdentityPrecond, JacobiPrecondLO, NysPrecondLO)
precond = precond_classes[precond_num](A=A_fn, rank=rank, mu=mu, dtype=A.dtype, shape=A.shape)
x0 = jnp.zeros_like(b)
t0 = time.time()
# v, _ = cg_solver(A=A_fn, b=vec, tol=cgtol, M=precond, maxiter=maxcgiters)
v = cg_solver(A_fn, b=b, max_iters=maxcgiters, tolerance=cgtol, preconditioner=precond, x0=x0)
t1 = time.time()
print_time_taken(delta=t1 - t0)

print('Test CG solve')
diff = jnp.linalg.norm(A @ sol - b)
print(f'Diff {diff:1.3e}')
