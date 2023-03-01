from jax import numpy as jnp


def propagate_mvm(A_fn, vec, block_size=1):
    N = vec.shape[0] // block_size
    aux = jnp.reshape(vec, newshape=(N, block_size), order="F")
    out = A_fn(aux)
    out = out.reshape(-1, order="F")
    return out


def evaluate_propagate_mvm(matrix, vector, block_size=1):
    size = matrix.shape[0]
    blocks_vector = jnp.reshape(vector, newshape=(size, block_size), order="F")
    mvm = matrix @ blocks_vector
    mvm = mvm.reshape(-1, order="F")
    return mvm
