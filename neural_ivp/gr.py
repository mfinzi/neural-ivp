import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
#from jax.ops import index, index_add, index_update
import numpy as np
from functools import partial
from jax import jacfwd, jacrev, jvp

D = jacfwd

@jit
def cartesian_s_metric(x, M=1 / 2):
    """ Computes the schwarzchild metric in cartesian like coordinates"""
    r = jnp.linalg.norm(x[1:])
    rhat = x[1:] / r
    rs = 2 * M
    a = (1 - rs / r)
    g = jnp.zeros((4, 4), dtype=r.dtype)
    g = g.at[0, 0].set(-a)
    g = g.at[1:, 1:].set(jnp.eye(3) + (1 / a - 1) * jnp.outer(rhat, rhat))
    return g

@jit
def christoffel_symbols(dg, ginv):
    """ Computes the christoffel symbols of the metric
        Γₐₑᵍ (lower lower upper)"""
    Γ_lower = (dg.transpose((2, 0, 1)) + dg.transpose((1, 2, 0)) - dg.transpose((0, 1, 2))) / 2
    return jnp.einsum('abd,cd->abc', Γ_lower, ginv)  #jnp.dot(Γ_lower.transpose((1,2,0)),ginv)