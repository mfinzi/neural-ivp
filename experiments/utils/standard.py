import numpy as np
from jax import numpy as jnp
from jax import vmap
from jax import jit
from scipy.signal import convolve2d
from scipy.integrate import solve_ivp
from scipy.ndimage import laplace
from jax.scipy.signal import correlate
from neural_ivp.gr import cartesian_s_metric, christoffel_symbols, D


def wave_maps_finite_diff(u0, T, xygrid, dx, N=50, rtol=1e-3):
    deriv = np.array([-1 / 2, 0., 1 / 2]) / dx
    jderiv = lambda x: correlate(x, deriv, mode='same')
    di = lambda x, i: jnp.apply_along_axis(jderiv, i, x)
    grad = lambda x: jnp.stack([di(x, i) for i in [-3, -2, -1]], axis=0)
    Gamma_all = vmap(Gammax)(xygrid).T.reshape(4, N, N, N)
    ginv_all = vmap(ginvx)(xygrid).T.reshape(4, 4, N, N, N)
    u0s = vmap(u0)(xygrid)
    z0 = jnp.concatenate([u0s[:, 0], u0s[:, 1]], axis=-1)

    @jit
    def F(_, z):
        _, ut = z.reshape(2, N, N, N)
        du, dut = jnp.transpose(grad(z.reshape(2, N, N, N)), (1, 0, 2, 3, 4))
        Hu = grad(du)
        Deltau = -(ginv_all[1:, 1:] * Hu).sum((0, 1))
        gammadphi = (Gamma_all[1:] * du).sum(0)
        gammaphit = Gamma_all[0] * ut
        g0i = -2 * (ginv_all[0, 1:] * dut).sum(0)
        utt = (Deltau + gammadphi + gammaphit + g0i) / ginv_all[0, 0]
        return jnp.concatenate([ut.reshape(-1), utt.reshape(-1)])

    zt = solve_ivp(F, (T[0], T[-1]), jnp.asarray(z0), t_eval=T, rtol=rtol).y.T
    return zt


gfunc = lambda z: cartesian_s_metric(z + 2 * jnp.eye(4)[1])
ginv = lambda z: jnp.linalg.inv(gfunc(z))
Γfunc = lambda z: christoffel_symbols(D(gfunc)(z), ginv(z))
Gamma = lambda z: (ginv(z)[..., None] * Γfunc(z)).sum((-3, -2))
ginvx = lambda x: ginv(jnp.concatenate([0 * x[:1], x]))
Gammax = lambda x: Gamma(jnp.concatenate([0 * x[:1], x]))


def advec_eq_soln(u0, T, xygrid, dx, N=50, rtol=1e-3):

    def dux(x):
        return convolve2d(x, -np.array([[-1, 0, 1]]) / (2 * dx), mode='same')

    def duy(x):
        return convolve2d(x, -np.array([[-1], [0], [1]]) / (2 * dx), mode='same')

    u0s = vmap(u0)(xygrid)[:, 0]

    def F(_, u):
        uimg = u.reshape(N, N)
        return (duy(uimg).reshape(-1) + dux(uimg).reshape(-1))

    zt = solve_ivp(F, (T[0], T[-1]), jnp.asarray(u0s), t_eval=T, rtol=rtol).y.T
    return zt


def wave_eq3d_finite_diff(u0, T, xygrid, dx, N=50, rtol=1e-3):

    def laplacian(x):
        return laplace(x, mode='constant') / dx**2

    u0s = vmap(u0)(xygrid)
    z0 = jnp.concatenate([u0s[:, 0], u0s[:, 1]], axis=-1)

    def F(_, z):
        u, ut = z.reshape(2, -1)
        Deltau = laplacian(u.reshape(N, N, N)).reshape(-1)
        return np.concatenate([ut, Deltau])

    zt = solve_ivp(F, (T[0], T[-1]), jnp.asarray(z0), t_eval=T, rtol=rtol).y.T
    return zt


def wave_ode_soln(u0, T, xygrid, dx, N=50, rtol=1e-6, mult=0.1):
    boundaries = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / dx**2.

    def laplacian(x):
        return convolve2d(x, boundaries, mode="same")

    u0 = vmap(u0)(xygrid)
    z0 = jnp.concatenate([u0[:, 0], u0[:, 1]], axis=-1)

    def F(_, z):
        u, ut = z.reshape(2, -1)
        Deltau = mult * laplacian(u.reshape(N, N)).reshape(-1)
        return np.concatenate([ut, Deltau])

    zt = solve_ivp(F, (T[0], T[-1]), jnp.asarray(z0), t_eval=T, rtol=rtol).y.T
    return zt
