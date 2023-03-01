from functools import partial
import numpy as np
from jax import vmap


def compute_rel_err_analytical(zt, fns, T, xygrid, N=100, logger=None):
    zt_t = lambda t: vmap(partial(zt, T[t]))(xygrid)
    f_t = lambda t: vmap(fns[t])(xygrid)
    reshape_fn = partial(reshape_for_analytical, N=N, idx=0)
    rel_err = compute_rel_err(zt=zt_t, fns=f_t, T=T, reshape_fn=reshape_fn, logger=logger)
    return rel_err


def compute_rel_err_maps(zt, fns, T, xygrid, N=100, logger=None):
    zt_t = lambda t: zt[t]
    f_t = lambda t: vmap(fns[t])(xygrid)
    reshape_fn = partial(reshape_for_maps, idx=0, N=N)
    rel_err = compute_rel_err(zt=zt_t, fns=f_t, T=T, reshape_fn=reshape_fn, logger=logger)
    return rel_err


def compute_rel_err_wave(zt, fns, T, xygrid, N=1_000, logger=None):
    zt_t = lambda t: zt[t]
    f_t = lambda t: vmap(fns[t])(xygrid)
    reshape_fn = partial(reshape_for_wave, N=N, idx=0)
    rel_err = compute_rel_err(zt=zt_t, fns=f_t, T=T, reshape_fn=reshape_fn, logger=logger)
    return rel_err


def compute_rel_err_ad(zt, fns, T, xygrid, logger=None):
    zt_t = lambda t: zt[t]
    f_t = lambda t: vmap(fns[t])(xygrid)
    reshape_fn = reshape_for_ad
    rel_err = compute_rel_err(zt=zt_t, fns=f_t, T=T, reshape_fn=reshape_fn, logger=logger)
    return rel_err


def compute_rel_err(zt, fns, T, reshape_fn, logger):
    rel_errs = []
    for t in range(len(T)):
        fdsol, nnsol = reshape_fn(zt(t), fns(t))
        diff = np.mean((fdsol - nnsol)**2.)
        norm = np.mean(fdsol**2.)
        rel_err = np.sqrt(diff / norm)
        print(f"Rel err: {rel_err:1.3e} | T: {T[t]:2.4f}")
        if logger is not None:
            logger.info(f"Rel err: {rel_err:1.3e} | T: {T[t]:2.4f}")
        rel_errs.append(rel_err)
    return np.array(rel_errs)


def compute_rel_error_from_arrays(zt, zt_hat, T, logger):
    rel_errs = []
    for t in range(len(T)):
        diff = np.mean((zt[t] - zt_hat[t])**2.)
        norm = np.mean(zt[t]**2.)
        rel_err = np.sqrt(diff / norm)
        print(f"Rel err: {rel_err:1.3e} | T: {T[t]:2.4f}")
        if logger is not None:
            logger.info(f"Rel err: {rel_err:1.3e} | T: {T[t]:2.4f}")
        rel_errs.append(rel_err)
    return np.array(rel_errs)


def reshape_for_analytical(zt_t, f_t, idx, N):
    fdsol = zt_t.reshape(N, N, N)
    nnsol = f_t.reshape(N, N, N, 2)[:, :, :, idx]
    return fdsol, nnsol


def reshape_for_maps(zt_t, f_t, idx, N):
    fdsol = zt_t.reshape(2, N, N, N)[idx, :]
    nnsol = f_t.reshape(N, N, N, 2)[:, :, :, idx]
    return fdsol, nnsol


def reshape_for_wave(zt_t, f_t, N, idx):
    out = zt_t.reshape(2, int(N**2)).T
    nnsol = f_t[:, 0]
    fdsol = out[:, idx]
    return fdsol, nnsol


def reshape_for_ad(zt_t, f_t):
    return zt_t, f_t[:, 0]
