from jax import numpy as jnp


def compute_rbf(x):
    distances = compute_sq_distances(x, x)
    cov = compute_rbf_cov(jnp.sum(distances, axis=-1))
    return cov


def compute_sq_distances(xi, xj):
    xi, xj = jnp.expand_dims(xi, -2), jnp.expand_dims(xj, -3)
    distances = (xi - xj) ** 2.
    return distances


def compute_rbf_cov(distances):
    constant = jnp.array(0.5, dtype=distances.dtype)
    res = jnp.exp(-constant * distances)
    return res
