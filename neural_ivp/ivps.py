import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd
from jax.random import uniform
from jax.random import split
from jax.random import PRNGKey
from neural_ivp.neural_ivp import IVPSolver


def u0(x):
    s = 0.05
    output = jnp.exp(-(x**2).sum() / (2 * s**2)) / (2 * jnp.pi * s**2) / 30.0
    return output[None]


def uu0(x):
    return jnp.array([u0(x)[0], 0.0])


def sample_batch(d, key, bs, dtype):
    X = uniform(key, (bs, d), dtype=dtype)
    return 2 * X - 1


class HeatSolver(IVPSolver):
    spatial_dim = 2
    default_ics = u0

    @staticmethod
    def pde_f(nnfn, x, t):
        du = grad(lambda x: nnfn(x)[0])
        laplacian_u = jnp.trace(jacfwd(du)(x))
        return 0.05 * laplacian_u

    def sampler(self, key, bs, dtype, t):
        return sample_batch(self.spatial_dim, key, bs, dtype=dtype)


class AdvectionSolver(IVPSolver):
    spatial_dim = 2
    default_ics = u0

    @staticmethod
    def pde_f(nnfn, x, t):
        du = grad(lambda x: nnfn(x)[0])
        return du(x).sum()[None]

    def sampler(self, key, bs, dtype, t):
        return sample_batch(self.spatial_dim, key, bs, dtype=dtype)


class WaveEqSolver(IVPSolver):
    spatial_dim = 2
    default_ics = lambda x: jnp.array([u0(x)[0], 0.0])

    @staticmethod
    def pde_f(nnfn, x, t):
        # wave equation ∂ₜ∂ₜu = Δu
        du = grad(lambda x: nnfn(x)[0])
        laplacian_u = jnp.trace(jacfwd(du)(x))
        ut = nnfn(x)[1]
        return jnp.array([ut, 0.1 * laplacian_u])

    def sampler(self, key, bs, dtype, t):
        return sample_batch(self.spatial_dim, key, bs, dtype=dtype)


def maxwellboltzmann_dist(v, T=1):
    return jnp.exp(-0.5 * (v * v).sum() / T) / jnp.sqrt(2 * jnp.pi * T)**3


def gaussian(x, mu=0, sigma=1):
    return jnp.exp(-0.5 *
                   (((x - mu) / sigma)**2).sum()) / jnp.sqrt(2 * jnp.pi * sigma**2)**x.shape[-1]


class VlasovEqSolver(IVPSolver):
    spatial_dim = 6
    default_ics = lambda z: (gaussian(z[:3], mu=0, sigma=.3) * gaussian(z[3:], sigma=.3))[None]
    default_E_field = grad(lambda x: jnp.exp(-(x**2).sum()))

    @staticmethod
    def pde_f(nnfn, z, t):
        du = grad(lambda x: nnfn(x)[0])(z)
        x, v = z[:3], z[3:]
        F = VlasovEqSolver.default_E_field(x)
        vz = jnp.concatenate([v, F])
        return -(vz * du).sum()[None]

    def sampler(self, key, bs, dtype, t):
        return sample_batch(self.spatial_dim, key, bs, dtype=dtype)


class HarmonicFPSolver(IVPSolver):  # Harmonic trap Focker Planck equation
    spatial_dim = 5  #TODO: set the dimension
    default_ics = lambda x: jnp.prod(1. - x**2)[None] * (
        3 / 4)**HarmonicFPSolver.spatial_dim  # uniform distribution

    @staticmethod
    def pde_f(nnfn, x, t):
        alpha = 1 / 4
        d = HarmonicFPSolver.spatial_dim
        D = 1e-2
        a = .2  # static trap center rather than as a function of time
        # compute diffusion term
        du = grad(lambda x: nnfn(x)[0])
        laplacian_u = jnp.trace(jacfwd(du)(x))
        # compute drift term
        h = lambda x: (a - x) + (alpha / d) * (x.sum() - d * x)
        v = lambda x: nnfn(x)[0] * h(x)
        div_v = jnp.trace(jacfwd(v)(x))
        ut = D * laplacian_u - div_v
        return ut[None]

    def sampler(self, key, bs, dtype, t):
        return sample_batch(self.spatial_dim, key, bs, dtype=dtype)


from neural_ivp.gr import cartesian_s_metric, christoffel_symbols, D


class WaveMapsSolver(IVPSolver):
    spatial_dim = 3
    default_ics = None

    def __init__(self, *args, g=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = g if g is not None else (lambda z: cartesian_s_metric(z + 2 * jnp.eye(4)[1]))
        self.ginv = lambda z: jnp.linalg.inv(self.g(z))
        self.Γ = lambda z: christoffel_symbols(D(self.g)(z), self.ginv(z))
        self.Gamma = lambda z: (self.ginv(z)[..., None] * self.Γ(z)).sum((-3, -2))
        self.ginvx = lambda x: self.ginv(jnp.concatenate([0 * x[:1], x]))
        self.Gammax = lambda x: self.Gamma(jnp.concatenate([0 * x[:1], x]))

    #@staticmethod
    def pde_f(self, nnfn, x, t):
        # wave maps equation ☐_gu = 0
        ut = nnfn(x)[1]
        ginvx = self.ginvx(x)
        Gammax = self.Gammax(x)
        du = grad(lambda x: nnfn(x)[0])
        Hu = jacfwd(du)(x)
        Deltau = -(ginvx[1:, 1:] * Hu).sum()
        gammadphi = (Gammax[1:] * du(x)).sum(0)
        gammaphit = Gammax[0] * ut
        dut = grad(lambda x: nnfn(x)[1])(x)
        g0i = -2 * (ginvx[0, 1:] * dut).sum(0)
        utt = (Deltau + gammadphi + gammaphit + g0i) / ginvx[0, 0]
        ut = nnfn(x)[1]
        return jnp.concatenate([ut[None], utt[None]])

    def sampler(self, key, bs, dtype, t):
        return sample_batch(self.spatial_dim, key, bs, dtype=dtype)


class WaveEqSolver3D(WaveEqSolver):
    spatial_dim = 3
    default_ics = lambda x: jnp.array([u0(x)[0], 0.0])

    @staticmethod
    def pde_f(nnfn, x, t):
        # wave equation ∂ₜ∂ₜu = Δu
        du = grad(lambda x: nnfn(x)[0])
        laplacian_u = jnp.trace(jacfwd(du)(x))
        ut = nnfn(x)[1]
        return jnp.array([ut, laplacian_u])

    def sampler(self, key, bs, dtype, t):
        return sample_batch(self.spatial_dim, key, bs, dtype=dtype)
