import unittest
import numpy as np
from functools import partial
from jax import numpy as jnp
from jax import jit
from jax import grad
from jax import jvp
from jax import vmap
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.nn import swish
from jax.random import PRNGKey, normal, split
import haiku as hk
from emlp.reps.linear_operator_base import LinearOperator
from neural_pde.mvms import compute_mvm_chunked
from neural_pde.mvms import compute_chunked_loop
from jax.config import config

config.update("jax_enable_x64", True)


class TestMVMs(unittest.TestCase):
    def test_compare_precision(self):
        dtype = jnp.float32
        dtype2 = jnp.float64
        grid_size = 10_000
        batch_size = 10_000
        key = PRNGKey(seed=21)
        nn = get_nn()
        theta = nn.init(key, jnp.zeros(shape=(2, ), dtype=dtype))
        unflat_theta = ravel_pytree(theta)[0]
        vec = normal(key, shape=(unflat_theta.shape[0], 1), dtype=dtype)

        def sampler(key, batch_size):
            return sample_fn(key, batch_size, dtype=dtype)

        mvm_fn = M_estimate(
            nn=nn.apply,
            reg=0,
            N=grid_size,
            bs=batch_size,
            params=theta,
            sampler=sampler,
            pde_f=pde_f,
            key=key,
            is_adaptive=False,
            cache_v0=None,
        )
        soln_mvm = mvm_fn(vec)

        def compute_bMv(X, V):
            vBMv = jit(vmap(batch_Mv, (None, None, None, None, 1), 1), static_argnums=(0, 1))
            out = vBMv(nn.apply, pde_f, theta, X, V)
            return out

        def mvm_chunked(vec):
            out = compute_mvm_chunked(mvm_fn=compute_bMv, sampler=sampler, key=key, vec=vec,
                                      grid_size=grid_size, batch_size=batch_size)
            return out

        soln_jit = jit(mvm_chunked)(vec)

        vec = normal(key, shape=(unflat_theta.shape[0], 1), dtype=dtype2)
        theta = tree_map(lambda x: x.astype(dtype2), theta)

        def sampler(key, batch_size):
            return sample_fn(key, batch_size, dtype=dtype2)

        mvm_fn = M_estimate(
            nn=nn.apply,
            reg=0,
            N=grid_size,
            bs=batch_size,
            params=theta,
            sampler=sampler,
            pde_f=pde_f,
            key=key,
            is_adaptive=False,
            cache_v0=None,
        )
        soln_2 = mvm_fn(vec)

        def compute_bMv(X, V):
            vBMv = jit(vmap(batch_Mv, (None, None, None, None, 1), 1), static_argnums=(0, 1))
            out = vBMv(nn.apply, pde_f, theta, X, V)
            return out

        def mvm_chunked(vec):
            out = compute_mvm_chunked(mvm_fn=compute_bMv, sampler=sampler, key=key, vec=vec,
                                      grid_size=grid_size, batch_size=batch_size)
            return out

        soln2_jit = jit(mvm_chunked)(vec)

        print("TEST:")
        diff = jnp.linalg.norm(soln_mvm.astype(dtype2) - soln_2)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / jnp.linalg.norm(soln_2)
        print(f"Rel Diff: {rel_diff:1.3e}")
        diff = jnp.linalg.norm(soln_mvm.astype(dtype2) - soln_jit)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / jnp.linalg.norm(soln_jit)
        print(f"Rel Diff: {rel_diff:1.3e}")
        diff = jnp.linalg.norm(soln_2 - soln2_jit)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / jnp.linalg.norm(soln_2)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-5)
        self.assertLessEqual(rel_diff, 1e-5)

    def test_against_MVM(self):
        dtype = jnp.float64
        grid_size = 100
        batch_size = 10
        key = PRNGKey(seed=21)
        nn = get_nn()
        theta = nn.init(key, jnp.zeros(shape=(2, )))
        unflat_theta = ravel_pytree(theta)[0]
        shape = (unflat_theta.shape[0], )

        def compute_bMv(X, V):
            vBMv = jit(vmap(batch_Mv, (None, None, None, None, 1), 1), static_argnums=(0, 1))
            out = vBMv(nn.apply, pde_f, theta, X, V)
            return out

        def compute_bF(X):
            out = batch_F(nn.apply, pde_f, theta, X)
            return out

        vec = normal(key, shape=(unflat_theta.shape[0], 1), dtype=dtype)

        def sampler(key, batch_size):
            return sample_fn(key, batch_size, dtype=dtype)

        mvm_fn = M_estimate(
            nn=nn.apply,
            reg=0,
            N=grid_size,
            bs=batch_size,
            params=theta,
            sampler=sampler,
            pde_f=pde_f,
            key=key,
            is_adaptive=False,
            cache_v0=None,
        )
        soln_mvm = mvm_fn(vec)
        soln_f = mvm_fn.estimate_F()

        def mvm_chunked(vec):
            out = compute_mvm_chunked(mvm_fn=compute_bMv, sampler=sampler, key=key, vec=vec,
                                      grid_size=grid_size, batch_size=batch_size)
            return out

        def f_chunked():
            out = compute_chunked_loop(fn=compute_bF, sampler=sampler, shape=shape, dtype=dtype,
                                       key=key, grid_size=grid_size, batch_size=batch_size)
            return out

        # check = mvm_chunked(vec)
        check_mvm = jit(mvm_chunked)(vec)
        check_f = f_chunked()
        print("\nTEST: against M_estimate")
        diff = jnp.linalg.norm(check_mvm - soln_mvm)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / jnp.linalg.norm(soln_mvm)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-5)

        print("\nTEST: against F_estimate")
        diff = jnp.linalg.norm(check_f - soln_f)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / jnp.linalg.norm(soln_f)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-5)

    def test_compute_mvm_chunked(self):
        dtype = jnp.float32
        grid_size = 100
        batch_size = 10
        key = PRNGKey(seed=21)
        vec = normal(key, shape=(grid_size, ), dtype=dtype)

        def sampler(key, batch_size):
            return sample_fn(key, batch_size, dtype=dtype)

        def mvm_chunked(vec):
            out = compute_mvm_chunked(mvm_fn=compute_mvm, sampler=sampler, key=key, vec=vec,
                                      grid_size=grid_size, batch_size=batch_size)
            return out

        check = jit(mvm_chunked)(vec)
        # check = mvm_chunked(vec)
        soln = get_mvm_chunked(key, batch_size, grid_size, vec)
        print("\nTEST: MVM chunked")
        diff = jnp.linalg.norm(check - soln)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / jnp.linalg.norm(soln)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-0)


def batch_Mv(nn, _, params, X, v):
    _, unflatten = ravel_pytree(params)

    def vMv2(x, v):
        _, dudv = jvp((lambda p: nn(p, x)), (params, ), (unflatten(v), ))
        vMiv = (dudv**2).sum()
        return vMiv / 2.0

    return grad(lambda v: vmap(vMv2, (0, None), 0)(X, v).mean(0))(v)


vBMv = jit(vmap(batch_Mv, (None, None, None, None, 1), 1), static_argnums=(0, 1))


def batch_F(nn, pde_f, params, X):
    flat_params, unflatten = ravel_pytree(params)

    def Fv(x, v):
        u, dudv = jvp((lambda p: nn(p, x)), (params, ), (unflatten(v), ))
        return (dudv * pde_f(partial(nn, params), x)).sum()

    return grad(lambda v: vmap(Fv, (0, None), 0)(X, v).mean(0))(jnp.zeros(len(flat_params)))


def get_nn():
    nn = hk.without_apply_rng(hk.transform(nn1))
    return nn


def nn1(x):
    units = 10
    act = swish
    mlp = hk.Sequential([hk.Linear(units), act, hk.Linear(units), act])
    out = hk.Linear(1)(mlp(x))
    return out


def pde_f(nnfn, x):
    du = grad(lambda x: nnfn(x)[0])
    return du(x).sum()[None]


def sample_fn(key, batch_size, dtype):
    x = normal(key, shape=(batch_size, 2), dtype=dtype)
    return x


def compute_mvm(x, v):
    return v * jnp.sum(x)


def get_mvm_chunked(key, batch_size, grid_size, vec):
    N = 0
    partial_sum = 0.0 * vec
    while N < grid_size:
        key = split(key)[0]
        x = sample_fn(key, batch_size, dtype=vec.dtype)
        partial_sum += compute_mvm(x, vec) * x.shape[0]
        N += x.shape[0]
    return partial_sum / N


class M_estimate(LinearOperator):
    def __init__(
        self,
        nn,
        params,
        sampler,
        pde_f,
        key,
        N,
        bs,
        reg,
        is_adaptive=True,
        cache_v0=None,
    ):
        flat_params, self.unflatten = ravel_pytree(params)
        d = len(flat_params)
        self.shape = (d, d)
        super().__init__(np.float32, self.shape)
        self.nn = nn
        self.bMv = partial(vBMv, nn, pde_f, params)
        self.bF = partial(batch_F, nn, pde_f, params)
        self.sampler = sampler
        self.key = key
        self.N = N
        self.evals = 0
        self.reg = reg
        self.batch_size = bs
        self.is_adaptive = is_adaptive
        self.cache_v0 = cache_v0
        # self.mvm_chunked = partial(compute_mvm_chunked, self.bMv, self.sampler, self.key,
        #                            grid_size=N, batch_size=bs)

    def sample(self, key, bs):
        return self.sampler(key, bs)

    # def _matmat(self, V):
    #     return self.mvm_chunked(V)

    def estimate_F(self):
        N = 0
        key = self.key
        partial_sum = 0.0
        while N < self.N:
            key = split(key)[0]
            X = self.sample(key, self.batch_size)
            partial_sum += self.bF(X) * X.shape[0]
            N += X.shape[0]
        return partial_sum / N

    def _matmat(self, V):
        self.evals += 1
        V = jnp.array(V)
        N = 0
        key = self.key
        partial_sum = 0.0 * V
        while N < self.N:
            key = split(key)[0]
            X = self.sample(key, self.batch_size)
            partial_sum += self.bMv(X, V) * X.shape[0]
            N += X.shape[0]
        return partial_sum / N + self.reg * V


if __name__ == "__main__":
    unittest.main()
