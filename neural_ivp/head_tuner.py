from functools import partial
from datetime import datetime
from jax import numpy as jnp
from jax import vmap
from jax import grad
from jax import jit
from jax.random import PRNGKey
from jax.random import split
from jax.random import choice
from optax import adam
from optax import apply_updates
import haiku as hk
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm
from utils.fns import convert_to_np
from utils.fns import convert_to_jax
from utils.general import print_time_taken
import logging
import optax
import jax
import numpy as np
import copy


def ls_solve(target, fn, init_params, sampler, epochs=10000, lr=1e-3, bs=10000):
    opt = optax.adam(lr)
    opt_state = opt.init(init_params)
    params = init_params
    key = PRNGKey(0)

    @jit
    def loss(params, X):
        err = vmap(partial(fn, params))(X) - vmap(target)(X)
        return (err**2).mean()

    @jit
    def update(params, opt_state, batch):
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for step in tqdm(range(epochs), desc="LS solve for initial conditions"):
        key, _ = split(key)
        batch = sampler(key, bs)
        if not step % (epochs // 10):
            # print(f"LS soln error norm: {np.sqrt(loss(params, batch))}")
            logging.info(f"LS soln error norm: {np.sqrt(loss(params, batch))}")
        params, opt_state = update(params, opt_state, batch)
    return params


def optimize_loss(loss, init_params, sampler, epochs=10000, lr=1e-3, bs=10000):
    opt = optax.adam(lr)
    opt_state = opt.init(init_params)
    params = init_params
    key = PRNGKey(13)

    @jit
    def update(params, opt_state, batch):
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for step in tqdm(range(epochs), desc="LS solve for initial conditions"):
        key, _ = split(key)
        batch = sampler(key, bs)
        if not step % (epochs // 10):
            # print(f"LS soln error norm: {np.sqrt(loss(params, batch))}")
            logging.info(f"sqrt loss: {np.sqrt(loss(params, batch))}")
        params, opt_state = update(params, opt_state, batch)
    return params


def tune_head(target, fn, params, sampler, N=100000):
    theta0 = copy.deepcopy(params)
    X = sampler(PRNGKey(0), N)
    Phi = construct_features2(theta0, fn, X=X)
    loc = list(theta0.keys())[-len(Phi):]
    for i in range(len(Phi)):
        rhs = vmap(target)(X)[:, [i]]
        reg = 1e-5 * len(rhs) / Phi[i].shape[-1]
        weight, bias = solve_head(Phi=Phi[i], y=rhs, reg=reg)
        theta0[loc[i]]["w"] = weight.astype(theta0[loc[i]]["w"].dtype)
        theta0[loc[i]]["b"] = bias.astype(theta0[loc[i]]["b"].dtype)
    return theta0


def construct_features2(theta_wo, nnfn, X):
    def phi(x):
        feats, bound = nnfn(theta_wo, x, feats_only=True)
        output = list(feats)
        for j in range(len(output)):
            output[j] = jnp.concatenate([jnp.array([1.0]), output[j]])
            output[j] *= bound
        return output

    Phi = vmap(phi)(X)
    return Phi


def solve_head(Phi, y, reg=0.):
    # linear = LinearRegression(fit_intercept=True).fit(Phi, y)
    linear = Ridge(alpha=reg, fit_intercept=True).fit(Phi, y)
    loss = jnp.sqrt(jnp.mean((y - Phi @ linear.coef_.T)**2))
    logging.info(f"Head RMSE {loss:1.3e}")
    sol = jnp.array(linear.coef_.T)
    return sol[1:, :], sol[0, :]