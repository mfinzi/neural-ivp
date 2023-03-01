import pytest
import jax.numpy as jnp
import jax
from gr.diffgeo import christoffel_symbols,spherical_s_metric,cartesian_s_metric,GP_s_metric
from gr.diffgeo import covariant_derivative, lie_derivative, applytransform2tensorfield
from gr.adm import four_momentum_constraint, n, extract_gamma_K,extract_gamma_K2
from jax import device_put
from jax import jacfwd, jacrev
from jax import grad, jit, vmap
from jax import random
from functools import partial
from gr.initial_data import scalar_Tμν, boost, shift, compose

def rel_error(t1,t2):
    error = jnp.sqrt(jnp.mean(jnp.abs(t1-t2)**2))
    scale = jnp.sqrt(jnp.mean(jnp.abs(t1)**2)) + jnp.sqrt(jnp.mean(jnp.abs(t2)**2))
    return error/jnp.maximum(scale,1e-3)

def test_schwarschild_christoffel_symbols(tol=1e-5):
    key = random.PRNGKey(0)
    X = 10*random.normal(key,(10000,4))
    X=X[X[:,1]>1.2]
    X = device_put(X)
    x=X[0]
    Γ = christoffel_symbols(jacfwd(spherical_s_metric)(x),jnp.linalg.inv(spherical_s_metric(x)))
    t,r,theta,phi = x
    rs=1
    assert rel_error(Γ[0,1,0],rs/(2*r*(r-rs)))<tol, "Γtrt"
    assert rel_error(Γ[1,1,1],-rs/(2*r*(r-rs)))<tol, "Γrrr"
    assert rel_error(Γ[0,0,1],rs*(r-rs)/(2*r**3))<tol, "Γttr"
    assert rel_error(Γ[3,3,1],(rs-r)*jnp.sin(theta)**2)<tol, "Γφφr"
    assert rel_error(Γ[2,2,1],(rs-r))<tol, "Γθθr"
    assert rel_error(Γ[1,2,2],1/r)<tol, "Γrθθ"
    assert rel_error(Γ[1,3,3],1/r)<tol, "Γrφφ"
    assert rel_error(Γ[3,3,2],-jnp.sin(theta)*jnp.cos(theta))<tol, "Γφφθ"
    assert rel_error(Γ[2,3,3],jnp.cos(theta)/jnp.sin(theta))<tol, "Γθφφ"


def sample(key,N):
    z = jax.random.normal(key,(N,4))
    r = jnp.sqrt((z[:,1:]**2).sum(-1))
    #new_r = loguniform(key,1.1,20,N)
    new_r = jax.random.uniform(key,(N,),minval=1.1,maxval=20)
    zout= jnp.concatenate([z[:,:1],z[:,1:]*new_r[:,None]/r[:,None]],axis=-1)
    return zout

def test_covariant_derivative(tol=1e-5):
    """ Covariant derivative of the metric should be 0"""
    key = random.PRNGKey(0)
    X = sample(key,100)
    gfunc = cartesian_s_metric
    Γ = lambda x: christoffel_symbols(jacfwd(gfunc)(x),jnp.linalg.inv(gfunc(x)))
    deriv_g = vmap(lambda x: covariant_derivative(gfunc,(0,0),x,Γ(x)))(X)
    assert rel_error(deriv_g,0*deriv_g)<tol, f"deriv_g error {deriv_g}"

def test_lie_derivative(tol=1e-5):
    X = random.normal(random.PRNGKey(0),(100,3))
    r = lambda x: x
    A,B,C = jnp.array(
        [[[0,-1,0],[1,0,0],[0,0,0]],
        [[0,0,-1],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,1],[0,-1,0]]])
    omega = lambda x: A@x
    omega2 = lambda x: B@x
    omega3 = lambda x: C@x
    Lo2o1 = vmap(lambda x: lie_derivative(omega2,(1,),x,omega))(X)
    assert rel_error(Lo2o1,vmap(omega3)(X))<tol, f"Rotations [Ωx,Ωy]=Ωz. Err"
    Lo1r = vmap(lambda x: lie_derivative(r,(0,),x,omega))(X)
    assert rel_error(Lo1r,0*Lo1r)<tol, "Rotations [Ωx,r]=0"

def test_hamiltonian_and_momentum_constraints(tol=1e-4):
    g = cartesian_s_metric
    ϕ = lambda x:0.0
    def constraints(x):
        r = jnp.linalg.norm(x[1:])
        α=jnp.sqrt(1-1/r)
        β=jnp.zeros(3)
        
        na = n(α,β)
        
        ginv = jnp.linalg.inv(g(x))
        #nlower = jnp.array([-α,0,0,0])
        γ = lambda y: g(jnp.insert(y,0,x[0]))[1:,1:]
        K = lambda y: jnp.zeros((3,3))
        Tμν = scalar_Tμν(ϕ,ginv,g(x),x)
        ρ = na@Tμν@na
        Si = -(na@Tμν)[1:]
        return four_momentum_constraint(γ,K,jnp.insert(Si,0,ρ),x[1:])
    key = random.PRNGKey(0)
    X = sample(key,100)
    mom_constraint = vmap(constraints)(X)
    assert rel_error(mom_constraint,0*mom_constraint)<tol, "Momentum constraint"

def test_apply_tensor_diffeomorphism(tol=1e-5):
    e0 = jnp.eye(4)[0]
    eta = jnp.eye(4) -2*e0[None]*e0[:,None]
    eta2 = applytransform2tensorfield(boost(.5),lambda x:eta,(1,1))(jnp.ones(4))
    assert rel_error(eta2,eta)<tol, "applytensor"

def four_momentum(g,xt):
    γ = lambda x: extract_gamma_K(g,jnp.insert(x,0,xt[0]))[0]
    K = lambda x: extract_gamma_K(g,jnp.insert(x,0,xt[0]))[1]
    return four_momentum_constraint(γ,K,jnp.zeros(4),xt[1:])

def test_diffeo_g_satisfies_constraints(tol=1e-5):
    g = GP_s_metric
    key = random.PRNGKey(0)
    d=3
    T = compose(shift(d,1),boost(.5/d,2))
    Tg = applytransform2tensorfield(T,g,(0,0))
    X = sample(key,500)
    err = vmap(partial(four_momentum,Tg))(X)
    assert all(vmap(jnp.linalg.norm)(err)<tol), "diffeo_g_satisfies_constraints"


def test_extract_gamma_K_equivalent(tol=1e-5):
    X = sample(random.PRNGKey(0),500)
    g1,k1 = vmap(partial(extract_gamma_K,GP_s_metric))(X)
    g2,k2 = vmap(partial(extract_gamma_K2,GP_s_metric))(X)
    assert rel_error(g1,g2)<tol, "extract_gamma_equivalent"
    assert rel_error(k1,k2)<tol, "extract_K_equivalent"
