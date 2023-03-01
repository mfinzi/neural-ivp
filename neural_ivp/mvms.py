from jax import numpy as jnp
from jax.lax import fori_loop
from jax.random import split


def compute_chunked_loop(fn, sampler, shape, dtype, key, grid_size, batch_size):
    blocks_n = grid_size // batch_size

    def body_fun(_, state):
        partial_sum, key = state
        key = split(key)[0]
        xi = sampler(key, batch_size)
        partial_sum += fn(xi) * batch_size
        return (partial_sum, key)

    init_val = (jnp.zeros(shape=shape, dtype=dtype), key)
    mid_val = fori_loop(0, blocks_n, body_fun, init_val)
    # mid_val = check_fori_loop(0, blocks_n, body_fun, init_val)
    output = mid_val[0] / grid_size

    return output


def compute_mvm_chunked(mvm_fn, sampler, key, vec, grid_size, batch_size):
    blocks_n = grid_size // batch_size

    def body_fun(_, state):
        partial_sum, key = state
        key = split(key)[0]
        xi = sampler(key, batch_size)
        partial_sum += mvm_fn(xi, vec) * batch_size
        return (partial_sum, key)

    init_val = (jnp.zeros(shape=vec.shape, dtype=vec.dtype), key)
    mid_val = fori_loop(0, blocks_n, body_fun, init_val)
    # mid_val = check_fori_loop(0, blocks_n, body_fun, init_val)
    output = mid_val[0] / grid_size

    return output


def check_fori_loop(lower, upper, body_fun, init_val):
    for idx in range(lower, upper):
        init_val = body_fun(idx, init_val)
    return init_val
