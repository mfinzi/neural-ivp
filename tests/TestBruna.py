import unittest
import numpy as np
from jax import grad
from jax import jacfwd
from jax import flatten_util
from jax.random import PRNGKey
from jax.random import uniform
from jax.random import normal
from jax.nn import tanh
from jax import numpy as jnp
import haiku as hk
from gr.bruna import compute_quadratic_bruna
from gr.bruna import compute_solution_from_loss
from gr.brunav2 import batch_M_diag
from gr.brunav2 import M_estimate
# from gr.bruna import bruna_architecture


class TestBruna(unittest.TestCase):

    def test_diagonal_extraction(self):
        test_tol = 1.e-5
        key = PRNGKey(0)
        N = 10
        d = 2
        x = uniform(key, (N, d))
        x = 2. * x - 1.
        batch = x
        f = heat_f
        model = hk.without_apply_rng(hk.transform(baby_nn))
        nn = model.apply
        params = model.init(PRNGKey(0), jnp.zeros(d))
        flat_params, _ = flatten_util.ravel_pytree(params)
        def sampler(key): return batch
        M = M_estimate(nn, params, sampler, f, key)
        aux1 = batch_M_diag(nn, f, params, batch)
        print('\nTEST: Diagonal extraction')
        aux = M._get_diag()
        diff = jnp.linalg.norm(aux - aux1)
        print(f'Diff {diff:1.3e}')
        check = np.zeros(shape=(aux.shape[0],))
        for i in range(aux.shape[0]):
            row_vec = np.zeros_like(flat_params)
            row_vec = np.expand_dims(row_vec, axis=-1)
            row_vec[i] = 1.
            row_vec = jnp.array(row_vec)
            output = M._matmat(row_vec)
            check[i] = output[i, 0]
        diff = jnp.linalg.norm(aux - check)
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tol)

    def test_compute_solution_from_loss(self):
        test_tol = 1.e-2
        key = PRNGKey(0)
        d = 5
        N = 100
        batch = uniform(key, shape=(N, d))
        x_manual = jnp.arange(d) + 1.
        x0 = normal(key, shape=(d,))
        y = batch @ x_manual + 0.01 * normal(key, shape=(N,))
        def loss_fn(x, batch): return jnp.sum((y - batch @ x) ** 2.)
        def sampler(key): return batch
        x = compute_solution_from_loss(
            loss_fn=loss_fn, x0=x0, sampler=sampler, epochs=1000, lr=1.e-1)
        diff = jnp.linalg.norm(x - x_manual)
        # diff_rel = jnp.linalg.norm(x - x_manual) / jnp.linalg.norm(x_manual)
        print('\nTEST: General Adam solver')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tol)

    def test_quadratic_bruna(self):
        test_tol = 1.e-7
        key = PRNGKey(0)
        N = 10
        d = 2
        x = jnp.zeros(shape=())
        x = uniform(key, (N, d))
        x = 2. * x - 1.
        x = x[0, :]
        f = heat_f
        model = hk.without_apply_rng(hk.transform(nn1))
        nn = model.apply
        params = model.init(PRNGKey(0), jnp.zeros(d))
        flat_params, _ = flatten_util.ravel_pytree(params)
        v = jnp.ones_like(flat_params)

        def get_grad(x):
            g = grad(nn)(params, x)
            aux, _ = flatten_util.ravel_pytree(g)
            return aux

        aux = compute_quadratic_bruna(v, x, params, f, get_grad, nn)
        diff = 0
        print('\nTEST: Solves')
        self.assertTrue(expr=diff < test_tol)


def heat_f(nnfn, x):
    du = grad(nnfn)
    laplacian_u = jnp.trace(jacfwd(du)(x))
    return laplacian_u


def baby_nn(x):
    k = 5
    act = tanh
    mlp = hk.Sequential([
        hk.Linear(k), act,
        hk.Linear(1)])
    u = mlp(x)
    return u


def nn1(x):
    k = 12
    act = tanh
    mlp = hk.Sequential([
        hk.Linear(k), act,
        hk.Linear(1)])
    u = mlp(x)[0]
    return u


if __name__ == '__main__':
    unittest.main()
