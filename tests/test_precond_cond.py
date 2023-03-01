from jax import numpy as jnp
dtype = jnp.float32
# N = int(1.e2)
N = 4
mu = 1.e-8

Lambda = jnp.array([10., 1., 0.1, 0.001])
U = jnp.eye(N, dtype=dtype)

l_ell, mu = jnp.min(Lambda), 1.e-6
identity = jnp.eye(N, dtype=dtype)
cons = (l_ell + mu)
center = jnp.diag(Lambda) + mu * identity
inv_center = jnp.diag(1. / (Lambda + mu))

P = (1. / cons) * U @ center @ U.T
P += identity - U @ U.T
P_inv = cons * U @ inv_center  @ U.T
P_inv += identity - U @ U.T

cond = jnp.linalg.cond(jnp.diag(Lambda))
cond = jnp.linalg.cond(P_inv ** 0.5 @ jnp.diag(Lambda) @ P_inv ** 0.5)
