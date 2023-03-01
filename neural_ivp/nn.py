import numpy as np
import jax.numpy as jnp
from functools import partial
from jax.random import PRNGKey
from jax.nn import tanh
from jax.nn import swish
from jax.nn import relu
from jax.flatten_util import ravel_pytree
import haiku as hk
from utils.general import load_object


def sinusoidal_embedding(x, L=10, scaling=2):
    w = (jnp.pi / 2.) * 2**jnp.arange(L)
    sins = 4 * (jnp.sin(w[:, None] * x[None, :]) / w[:, None]**scaling).reshape(-1)
    coss = 4 * (jnp.cos(w[:, None] * x[None, :]) / w[:, None]**scaling).reshape(-1)
    return jnp.concatenate([sins, coss], axis=-1)


def base_mlp2(x, k=200, L=6, scaling=.5):
    act = swish
    y = sinusoidal_embedding(x, L=L, scaling=scaling) * 1.5
    mlp = hk.Sequential([hk.Linear(k), act, hk.Linear(k), act, hk.Linear(k), act])
    return mlp(y)


def multi_output_nn2(x, n_outputs=2, k=200, L=5, scaling=1, feats_only=False):
    bound = jnp.prod(1. - x**2)
    feats = [base_mlp2(x, k=k, L=L, scaling=scaling) for _ in range(n_outputs)]
    if feats_only:
        return feats, bound
    return jnp.concatenate([hk.Linear(1)(feat) for feat in feats], -1) * bound


def base_mlp(x, k=200, L=6, scaling=.5):
    act = swish  # tanh
    # y = x
    # x_norm = jnp.linalg.norm(x)[None]
    # y_norm = x_norm
    #y = jnp.concatenate([x, sinusoidal_embedding(x, L=L, scaling=scaling) * 1.5])
    y = sinusoidal_embedding(x, L=L, scaling=scaling) * 1.5
    #y = jnp.concatenate([y, y_norm])
    mlp = hk.Sequential([hk.Linear(k), act, hk.Linear(k), act, hk.Linear(k), act])
    return mlp(y)


def variable_nn(x, k=200, L=6, feats_only=False):
    bound = jnp.prod(1. - x**2)
    feats = base_mlp(x, k=k, L=L)
    if feats_only:
        return (feats, ), bound
    return hk.Linear(1)(feats) * bound


def multi_output_nn(x, n_outputs=2, k=200, L=5, scaling=1, feats_only=False):
    bound = jnp.prod(1. - x**2)
    feats = [base_mlp(x, k=k, L=L, scaling=scaling) for _ in range(n_outputs)]
    if feats_only:
        return feats, bound
    return jnp.concatenate([hk.Linear(1)(feat) for feat in feats], -1) * bound


def ednn_base(x):
    # base network
    bound = jnp.prod(1. - x**2)
    k = 30
    act = tanh
    mlp = hk.Sequential(
        [hk.Linear(k), act,
         hk.Linear(k), act,
         hk.Linear(k), act,
         hk.Linear(k), act,
         hk.Linear(1)])
    return mlp(x) * bound


def ednn(x, n_outputs=2):
    bound = jnp.prod(1. - x**2)
    feats = [ednn_base(x) for _ in range(n_outputs)]
    return jnp.concatenate([hk.Linear(1)(feat) for feat in feats], -1) * bound