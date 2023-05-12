import logging
from functools import partial
import numpy as np
import haiku as hk
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
from jax.tree_util import tree_map
from jax.random import split
from jax.random import PRNGKey
from jax import grad
from jax import jit
from jax import vmap
from jax import jacfwd

from neural_ivp.mvms import compute_mvm_chunked
from neural_ivp.mvms import compute_chunked_loop
#from linops.linalg import inverse
#import linops as lo
from linops.operator_base import get_library_fns
from linops.linalg import solve_symmetric
from linops.algorithms.preconditioners import NystromPrecond
from linops import LinearOperator, I_like#, Symmetric  #,Identity

from neural_ivp.head_tuner import ls_solve, tune_head, optimize_loss
from neural_ivp.ode_solver import solve_ivp_rk

from jax import numpy as jnp
from jax.random import uniform
import jax
from neural_ivp.utils import LogTimer


def sample_ids_from_probs(probs, num_samples, key):
    ids = jnp.arange(probs.shape[0])
    return jax.random.choice(key, ids, shape=(num_samples, ), p=probs, replace=False)


# (1/n) JT(Jv)
@partial(jit, static_argnums=(0, 1))
def batch_Mv(nn, pde_f, params, X, v):
    _, unflatten = jax.flatten_util.ravel_pytree(params)

    def vMv2(x, v):
        f, dudv = jax.jvp((lambda p: nn(p, x)), (params, ), (unflatten(v), ))
        # dudv has shape d
        vMiv = (dudv**2).sum()
        return vMiv / 2.0

    return grad(lambda v: vmap(vMv2, (0, None), 0)(X, v).mean(0))(v)


# time O((n+p)r) # 500k x 250
vBMv = jit(vmap(batch_Mv, (None, None, None, None, 1), 1), static_argnums=(0, 1))


@partial(jit, static_argnums=(0, 1))
def batch_F(nn, pde_f, params, t, X):
    flat_params, unflatten = jax.flatten_util.ravel_pytree(params)

    def Fv(x, v):
        f, dudv = jax.jvp((lambda p: nn(p, x)), (params, ), (unflatten(v), ))
        return ((dudv * pde_f(partial(nn, params), x, t))).sum()

    return grad(lambda v: vmap(Fv, (0, None), 0)(X, v).mean(0))(jnp.zeros(
        len(flat_params), dtype=X.dtype))


@partial(jit, static_argnums=(0, 1))
def batch_L(nn, pde_f, params, t, v, X):
    res = batch_residuals(nn=nn, pde_f=pde_f, params=params, t=t, v=v, X=X)
    return res.mean()


@partial(jit, static_argnums=(0, 1))
def batch_residuals(nn, pde_f, params, t, v, X):
    pflat, unflatten = jax.flatten_util.ravel_pytree(params)

    def L(x):
        f = pde_f(partial(nn, params), x, t)
        _, dudt = jax.jvp((lambda p: nn(p, x)), (params, ), (unflatten(v.astype(pflat.dtype)), ))
        return ((dudt - f)**2).sum()

    return vmap(L)(X)


@partial(jit, static_argnums=(0, 1))
def batch_dl_dtheta(nn, pde_f, params, t, v, X):
    flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
    g = grad(lambda p: batch_L(nn, pde_f, unflatten(p), t, v, X))(flat_params)
    return g


class Casted(LinearOperator):
    def __init__(self, A, dtype):
        self.A = A
        super().__init__(dtype, A.shape)

    def _matmat(self, X):
        return (self.A @ X.astype(self.A.dtype)).astype(self.dtype)


class M_estimate(LinearOperator):
    def __init__(self, nn, params, sampler, pde_f, t, key, N, bs, is_adaptive=False, cache_v0=None,
                 chunk=False):
        self.nn = nn
        flat_params, self.unflatten = jax.flatten_util.ravel_pytree(params)
        self.batch_residuals = partial(batch_residuals, nn, pde_f, params, t)
        self.bMv = partial(vBMv, nn, pde_f, params)
        self.bF = partial(batch_F, nn, pde_f, params, t)
        self.batch_gradL = partial(batch_dl_dtheta, nn, pde_f, params, t)
        self.sampler = partial(sampler, t=t)
        self.key = key
        self.N = N
        d = len(flat_params)
        self.shape = (d, d)
        self.params_dtype = flat_params.dtype
        #print(f"params dtype {self.params_dtype}")
        super().__init__(self.params_dtype, self.shape)
        import linops.jax_fns as fns
        self.ops = fns
        print(self.ops)
        self.evals = 0
        self.batch_size = bs
        self.is_adaptive = is_adaptive
        self.cache_v0 = cache_v0.astype(flat_params.dtype)
        self.chunk = chunk
        self.mvm_chunked = partial(compute_mvm_chunked, self.bMv, self.sample, self.key,
                                   grid_size=N, batch_size=bs)
        self.f_chunked = partial(compute_chunked_loop, self.bF, self.sample, shape=(d, ),
                                 dtype=flat_params.dtype, key=self.key, grid_size=N, batch_size=bs)
        self.X = self.sample(self.key, N)

    def sample(self, key, bs):
        if not self.is_adaptive:
            return self.sampler(key, bs, self.params_dtype)
        else:
            div_n = 10
            oversampling_ratio = 10
            X = self.sampler(key, oversampling_ratio * bs, self.params_dtype)
            res = self.batch_residuals(self.cache_v0, X)

            probs = res.reshape((div_n, -1))
            probs = probs / probs.sum(-1, keepdims=True)
            keys = split(key, div_n)

            ind = vmap(sample_ids_from_probs, (0, None, 0), 0)(probs, bs // div_n, keys)
            X_new = X[ind.reshape(-1)]
            return X_new

    def cg_loss(self, v, i, n):
        #print(hash(X.tobytes()))
        #X = self.sample(key, self.N//n)
        v2 = v.astype(self.dtype)
        bs = self.N // n
        #X = self.X[i*bs:(i+1)*bs]
        X = jax.lax.dynamic_slice(self.X, (i * bs, 0), (bs, self.X.shape[1]))
        X = X.astype(self.dtype)
        return (v2 * ((self.bMv(X, v2[:, None])[:, 0]) / 2 - self.bF(X))).sum().astype(v.dtype)

    def estimate_F(self):
        if self.chunk:
            return self.f_chunked()
        N = 0
        key = self.key
        partial_sum = 0.0
        while N < self.N:
            key = split(key)[0]
            X = self.X[N:N + self.batch_size].astype(self.dtype)  #self.sample(key, self.batch_size)
            partial_sum += self.bF(X) * X.shape[0]
            N += X.shape[0]
        return (partial_sum / N).astype(self.dtype)

    def _matmat(self, V):
        self.evals += 1
        V2 = jnp.array(V).astype(self.dtype)
        if self.chunk:
            return (self.mvm_chunked(V2)).astype(V.dtype)
        N = 0
        key = self.key
        partial_sum = 0.0 * V2
        while N < self.N:
            key = split(key)[0]
            X = self.X[N:N + self.batch_size].astype(self.dtype)  #self.sample(key, self.batch_size)
            partial_sum += self.bMv(X, V2) * X.shape[0]
            N += X.shape[0]
        return (partial_sum / N).astype(V.dtype)

    def __call__(self, v):
        raise NotImplementedError


class IVPSolver:
    def __init__(self, nn_name, log_dir='.', dtype=jnp.float32, tune_head=True, epochs=100_000,
                 ics=None, first_fit=False):
        self.log_dir = log_dir
        self.dtype = dtype
        self.ics = ics if ics is not None else self.__class__.default_ics
        self.nn = hk.without_apply_rng(hk.transform(nn_name))
        self.epochs = epochs
        self.first_fit = first_fit
        self.tune_head = tune_head
        self.restart_timer = LogTimer(minPeriod=0.0, timeFrac=1 / 2)
        with self.restart_timer:
            self.z0, self.unflatten, self.v0 = self.setup(self.nn, self.tune_head, self.epochs,
                                                          first_fit=self.first_fit)
        self.theta_subspace = self.z0.copy()

    @staticmethod
    def pde_f(nn, x, t):
        del nn, x
        raise NotImplementedError

    def sampler(self, key, bs, t=0, dtype=jnp.float32):
        del key, bs, dtype
        raise NotImplementedError

    @staticmethod
    def ics(x):
        del x
        raise NotImplementedError

    def get_fns(self, thetas):
        output = [partial(self.nn.apply, self.unflatten(theta)) for theta in thetas]
        return output

    @partial(jit, static_argnums=(0, 4, 5))
    def pde_residual(self, theta, theta_dot, t, grid_size, batch_size, key):
        N, num, denom = 0, 0, 0
        while N < grid_size:
            X = self.sampler(key, batch_size, dtype=theta.dtype, t=t)
            unflat_theta = self.unflatten(theta)
            out = batch_L(self.nn.apply, self.pde_f, unflat_theta, t, theta_dot, X)
            res = out * X.shape[0]
            F_fn = lambda x: self.pde_f(partial(self.nn.apply, unflat_theta), x, t)
            F = vmap(F_fn)(X)
            N += X.shape[0]
            num += res
            denom += jnp.sum(F**2)
            key = split(key)[0]
        return jnp.sqrt(num / (denom * N))

    def log_info(self, z, v0, t, cg_residuals):
        res = cg_residuals
        logger = logging.getLogger(self.log_dir)
        rs = np.asarray(res)
        loc = np.argmin(rs > 0.0) - 1
        total_cg_iters = loc if loc > -1 else len(rs)
        text = f"Last RS norm: {np.linalg.norm(rs[loc]):1.5e} | "
        text += f"CG iters: {total_cg_iters:,d} | "
        text += f"Time: {t:2.4f}"
        logger.info(text)
        pde_residual = self.pde_residual(theta=z, theta_dot=v0, t=t, grid_size=10000,
                                         batch_size=10000, key=PRNGKey(38))
        text = f"PDE residual: {pde_residual:1.5e}"
        logger.info(text)

    def solve_ode(self, T, cgtol=1e-7, maxcgiter=500, mu=1e-6, rank=250, odetol=1e-4, method='rk23',
                  grid_size=5000, batch_size=5000, is_adaptive=False, run_chunks=False):
        pbar = tqdm(total=100, desc="ODE solve")

        def vfn_wlog(t, z, key):
            pbar.refresh()
            pbar.update(max(int(100 * (t - T[0]) / T[-1]) - pbar.n, 0))
            v0, res = self.v_fn(t=t, theta=z, v0=self.v0, cgtol=cgtol, maxcgiter=maxcgiter,
                                grid_size=grid_size, batch_size=batch_size, mu=mu, rank=rank,
                                is_adaptive=is_adaptive, key=key,
                                chunk=run_chunks)
            self.log_info(z, v0, t, res)
            self.v0 = jnp.zeros_like(v0)
            return v0

        def maybe_restart(z, key):
            with self.restart_timer as go:
                if go:
                    params = self.unflatten(z)
                    u0 = lambda x: self.nn.apply(params, x)
                    z, _, v0 = self.setup(self.nn, self.tune_head, self.epochs,
                                          first_fit=self.first_fit, u0=u0)
            return z, key
        output_zt, nevals = solve_ivp_rk(vfn_wlog, self.z0, T, rtol=odetol, method=method,
                                         rng=PRNGKey(32), transform=maybe_restart)
        pbar.close()
        logging.info(f"# evals {nevals}, # restarts: {self.restart_timer.numLogs}")
        return output_zt

    @partial(jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 10, 12))
    def v_fn(self, t, theta, v0, cgtol, rank, mu, maxcgiter, grid_size, batch_size, is_adaptive, key, chunk):
        theta = self.unflatten(theta)
        theta = tree_map(lambda x: x.astype(self.dtype), theta)
        M = M_estimate(nn=self.nn.apply, N=grid_size, bs=batch_size, params=theta, t=t,
                       sampler=self.sampler, pde_f=self.pde_f, key=key, is_adaptive=is_adaptive,
                       cache_v0=v0, chunk=chunk)
        precond = NystromPrecond(M, rank, mu=mu)
        F = M.estimate_F().astype(jnp.float64)
        A = Casted(M + precond.adjusted_mu * I_like(M), jnp.float64)
        v,info = solve_symmetric(A,F,x0=v0, tol=cgtol, max_iters=maxcgiter, P=precond, info=True,pbar=False)
        #Ainv = lo.linalg.inverse(A,x0=v0, tol=cgtol, max_iters=maxcgiter, P=precond, info=True, pbar=False)
        #v = Ainv @ F
        out = jax.flatten_util.ravel_pytree(v)[0]
        return out, info['residuals']

    def setup(self, model, do_head_tuning=True, epochs=50000, lr=3e-4, bs=10_000, first_fit=False,
              u0=None):
        theta = model.init(PRNGKey(0), jnp.zeros(self.spatial_dim, dtype=self.dtype))
        u0 = self.ics if u0 is None else u0
        sample_fn = partial(self.sampler, dtype=self.dtype, t=0)
        if first_fit:

            def combined_loss(combined_params, X):
                #alpha = .00003
                alpha = 3e-3
                v, params = combined_params
                fn = model.apply
                fit_mse = ((vmap(partial(fn, params))(X) - vmap(u0)(X))**2).mean()
                pde_mse = batch_L(model.apply, self.pde_f, params, v, X)
                s1, s2 = 1 / jax.lax.stop_gradient(fit_mse), 1 / jax.lax.stop_gradient(pde_mse)
                return fit_mse * s1 + alpha * pde_mse * s2

            vtheta = (jnp.zeros_like(jax.flatten_util.ravel_pytree(theta)[0]), theta)
            v, theta = optimize_loss(jit(combined_loss), vtheta, sample_fn, epochs, lr, bs)
        else:
            theta = ls_solve(target=u0, fn=model.apply, init_params=theta, sampler=sample_fn,
                             epochs=epochs, lr=lr, bs=bs)
            v = jnp.zeros_like(jax.flatten_util.ravel_pytree(theta)[0])
        if do_head_tuning:
            theta = tune_head(target=u0, fn=model.apply, params=theta, sampler=sample_fn, N=200000)

        print_error(model, theta, u0, sample_fn, bs)

        flat_theta0, unflatten_fn = jax.flatten_util.ravel_pytree(theta)
        return (flat_theta0, unflatten_fn, v.astype(jnp.float64))


def print_error(model, theta, u0, sampler, bs):
    X = sampler(PRNGKey(0), bs)
    err = vmap(partial(model.apply, theta))(X) - vmap(u0)(X)
    diff = jnp.sqrt((err**2.0).mean())
    print(f"Error {diff:1.3e}")
