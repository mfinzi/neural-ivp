import numpy as np
import jax.numpy as jnp
import collections
from jax import jit
from functools import partial
import jax

ButcherTableau = collections.namedtuple('ButcherTableau', 'a, b_sol, b_err, c, order')

integrator_tableaus = {
    'heun':
    ButcherTableau(a=(np.array([1.0]), ), b_sol=np.array([0.5, 0.5]), b_err=np.array([1., 0.]),
                   c=np.array([1.0]), order=2),
    'rk23':
    ButcherTableau(a=(np.array([1 / 2]), np.array([0., 3 / 4]), np.array([2 / 9, 1 / 3, 4 / 9])),
                   b_sol=np.array([2 / 9, 1 / 3, 4 / 9, 0.]),
                   b_err=np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8]), c=np.array([1 / 2, 3 / 4,
                                                                              1.]), order=3)
}


#@partial(jit, static_argnums=(0, ))
def rk_step(fn, y, t, h, tableau, key):
    a, b_sol, b_error, c, _ = tableau
    k = [fn(t, y, key)]
    for i in range(len(tableau.a)):
        k_i = fn(t + tableau.c[i] * h, y + h * sum((a * ki for a, ki in zip(tableau.a[i], k))), key)
        k.append(k_i)
    k = jnp.stack(k)
    y = y + h * jnp.dot(tableau.b_sol, k)
    error = h * jnp.dot(tableau.b_err - tableau.b_sol, k)
    return y, jnp.linalg.norm(error)


def while_loop(cond_fn, body_fn, init_val):
    val = init_val
    while cond_fn(val):
        val = body_fn(val)
    return val


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


#@partial(jit, static_argnums=(0, 3, 4))
def solve_ivp_rk(fn, y0, teval, rtol=1e-4, method='rk23', rng=None, transform=None):
    """ adaptive step size rk solver. TODO: make jit compilable """
    tableau = integrator_tableaus[method]
    err_target = rtol * jnp.linalg.norm(y0)
    keyfn = (lambda t, y, key: fn(t, y)) if rng is None else fn
    key = jax.random.PRNGKey(0) if rng is None else rng
    transform = (lambda x, key: x) if transform is None else transform

    #@jit
    def body_fn(x):
        y, t, h, nevals, key, tnext = x
        y, key = transform(y, key)
        step = jnp.minimum(h, tnext - t)
        yout, error = rk_step(keyfn, y, t, step, tableau, key)
        cond = error < err_target
        yout = jnp.where(cond, yout, y)
        tout = jnp.where(cond, t + step, t)
        # adaptive step size rescaling based on method order
        k = tableau.order
        E = err_target / error
        hout = jnp.where(cond, jnp.where(h <= tnext - t, h * 0.97 * E**(1. / k), h),
                         step * 0.97 * E**(1 / k))
        nevals += 1
        nextkey, _ = jax.random.split(key)
        return yout, tout, hout, nevals, nextkey, tnext

    #@jit
    def scan_fn(x, tstop):
        x = x[:-1] + (tstop, )
        out = while_loop(lambda x: x[1] < x[-1] - 1e-7, body_fn, x)
        return out, out[0]

    val = y0, teval[0], 20 * (teval[-1] - teval[0]) * rtol, 0, key, teval[1]
    carry, ys = scan(scan_fn, val, teval)
    return ys, carry[3]

    # while t_sol[-1] < t1:
    #     y, error = rk_step(fn, y, t_sol[-1], h, tableau)
    #     if error < err_target:
    #         h *= 0.99 * (err_target / error)**0.2
    #         y_sol.append(y)
    #         t_sol.append(t_sol[-1] + h)
    #     else:
    #         h *= 0.99 * (err_target / error)**0.25

    # val = y, t_sol[-1], h, 0
    # outval = jax.lax.while_loop(cond_fn, body_fn, val)
    # while cond_fn(val):
    #     val = body_fn(val)
    # return val[0], val[1]
    #return jnp.array(t_sol), jnp.array(y_sol)


# def solve_ivp_rk4(fn, y0, t_span, step_size):
#     total_iters = int((t_span[-1] - t_span[0]) // step_size)
#     ys = np.zeros(shape=(total_iters + 1, ) + y0.shape)
#     ts = np.zeros(shape=(total_iters + 1, ))
#     ys[0], ts[0] = y0, np.array(t_span[0])
#     for i in range(total_iters):
#         ys[i + 1], ts[i + 1] = take_rk4_step(fn, ys[i], ts[i], step_size=step_size)
#     return ys, ts

# def take_rk4_step(fn, y0, t0, step_size):
#     half_step_size = step_size / 2.
#     k1 = fn(t0, y0)
#     k2 = fn(t0 + half_step_size, y0 + half_step_size * k1)
#     k3 = fn(t0 + half_step_size, y0 + half_step_size * k2)
#     k4 = fn(t0 + step_size, y0 + step_size * k3)
#     y1 = y0 + (step_size / 6) * (k1 + 2. * k2 + 2. * k3 + k4)
#     if np.sum(np.isnan(y1)) > 0:
#         breakpoint()
#     t1 = t0 + step_size
#     return y1, t1