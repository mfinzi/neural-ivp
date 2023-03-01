import pickle
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from .general import load_object


def merge_thetas_snaps(path):
    thetas = []
    files = [f"wave_kMT_{i}_theta.pkl" for i in range(10)]
    for file in files:
        theta = load_object(path + file)
        theta_flat = ravel_pytree(theta)[0]
        thetas.append(theta_flat[None])
    thetas = np.concatenate(thetas, axis=0)
    return thetas


def rename_dict(new_keys, results):
    new_results = {}
    for k, v in results.items():
        new_results[new_keys[k]] = v
    return new_results


def reshape_finer_coarser(coarse, fine, nT, grid_size, double_grid_size):
    fine = fine.reshape(nT, 2, -1)
    fine = fine[:, 0].reshape(-1, double_grid_size, double_grid_size)
    fine = fine[:, 1:-1:2, 1:-1:2]

    coarse = coarse.reshape(nT, 2, -1)
    coarse = coarse[:, 0].reshape(-1, grid_size, grid_size)
    return coarse, fine


def construct_grid(grid_size, spatial_dim, return_dx=False):
    single_grid = np.linspace(-1, 1, grid_size + 2)[1:-1]
    dx = single_grid[1] - single_grid[0]
    grid_tup = (*[single_grid for _ in range(spatial_dim)], )
    grid = np.stack(np.meshgrid(*grid_tup), axis=0)
    grid = grid.reshape(spatial_dim, int(grid_size**spatial_dim)).T
    if return_dx:
        return grid, dx
    return grid


def get_T_from_defaults(defaults):
    nT = defaults['nT']
    tf = defaults['tf']
    T = np.arange(nT) * (tf / nT)
    return T


def convert_to_np(params):
    params_np = {}
    for k, v in params.items():
        params_np[k] = {'w': np.array(v['w']), 'b': np.array(v['b'])}
    return params_np


def convert_to_jax(params):
    params_jax = {}
    for k, v in params.items():
        params_jax[k] = {'w': jnp.array(v['w']), 'b': jnp.array(v['b'])}
    return params_jax


def load_theta(file_path):
    with open(file=file_path, mode='rb') as f:
        obj = pickle.load(file=f)
    return obj


def save_theta(theta, file_path):
    with open(file=file_path, mode='wb') as f:
        pickle.dump(theta, f)
