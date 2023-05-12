import logging
import time
import numpy as np
import fire
from sys import argv
from jax import numpy as jnp
from functools import partial
from neural_ivp.ivps import AdvectionSolver
from neural_ivp.nn import multi_output_nn
from neural_ivp.ivps import u0
from utils.general import add_timestamp_with_random
from utils.general import save_object
from utils.general import generate_log_dir
from utils.general import log_inputs
from utils.general import get_default_args
from utils.general import prepare_logger
from utils.general import print_time_taken
from utils.metrics import compute_rel_err_ad
from utils.standard import advec_eq_soln
from utils.fns import construct_grid
from utils.log_fn import get_vars_from_logs
from jax.config import config

config.update("jax_enable_x64", True)


def main(
        cgtol=1e-8,
        maxcgiter=int(1e3),
        mu=1e-8,
        rank=200,
        odetol=1e-3,
        grid_size=int(1e4),
        batch_size=int(1e4),
        method="rk23",
        dtype="single",
        is_adaptive=False,
        log_dir="",
        tune_head=False,
        epochs=int(1e5),
        nT=2,
        tf=0.5,
        nn_units=100,
        nn_freq=3,
        run_rel_err=True,
        run_chunks=False,
):
    T = np.arange(nT) * (tf / nT)
    dtype = jnp.float32 if dtype == "single" else jnp.float64
    ics = u0
    nn_name = partial(multi_output_nn, n_outputs=1, k=nn_units, L=nn_freq)
    Solver = AdvectionSolver(nn_name, log_dir=log_dir, dtype=dtype, tune_head=tune_head,
                             epochs=epochs, ics=ics)
    t0 = time.time()
    logger = logging.getLogger(log_dir)
    # thetas = [Solver.z0 for _ in range(len(T))]
    thetas = Solver.solve_ode(T=T, cgtol=cgtol, maxcgiter=maxcgiter, mu=mu, rank=rank,
                              odetol=odetol, grid_size=grid_size, batch_size=batch_size,
                              method=method, is_adaptive=is_adaptive, run_chunks=run_chunks)
    t1 = time.time()
    print_time_taken(t1 - t0, logger=logging.getLogger(log_dir))
    output_path = log_dir + "thetas_"
    save_object(thetas, add_timestamp_with_random(beginning=output_path))
    if run_rel_err:
        tt0 = time.time()
        grid_size = 1_000
        xygrid, dx = construct_grid(grid_size=grid_size, spatial_dim=2, return_dx=True)
        zt = advec_eq_soln(u0=ics, T=T, xygrid=xygrid, dx=dx, N=grid_size, rtol=1e-6)
        fns = Solver.get_fns(thetas)
        rel_errs = compute_rel_err_ad(zt=zt, fns=fns, xygrid=xygrid, logger=logger, T=T)
        save_object(rel_errs, filepath=log_dir + "rel_err.pkl")
        tt1 = time.time()
        print_time_taken(tt1 - tt0, logger=logger, text="Rel Err took:")

    results = get_vars_from_logs(log_dir + "info.log")
    results["time_taken"] = t1 - t0
    save_object(results, filepath=log_dir + "results.pkl")


def entrypoint(**kwargs):
    defaults = get_default_args(main)
    kwargs['log_dir'] = generate_log_dir()
    prepare_logger(kwargs["log_dir"])
    log_inputs(dict(defaults, **kwargs), log_dir=kwargs['log_dir'])
    logger = logging.getLogger(kwargs["log_dir"])
    logger.info("py " + " ".join(argv))
    main(**kwargs)


if __name__ == "__main__":
    fire.Fire(entrypoint)
