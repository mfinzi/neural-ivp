# Neural PDEs

## Notes
* Moving eps=1e-16 to 1e-8 helped on the single precision issue of the Wave Eq not
  running the SVD decomposition

Some recent changes are:
* different ODE integrator
* random seed for samples changing across timesteps (improves soln quality)
* changing regularization scale to be proportional to maximum eigenvalue of M. Seems more
  natural that what we did before but it's less clear exactly how this impacts the
  tradeoff between # of CG iterations and soln quality, and still not 100% clear which is
  better to do
* restarts as fraction of compute time (in many cases with how it's set right now it
  seems like the restarts are too infrequent). On the flip side, because regularization
  scale is set proportional to max eigenval, restarts actually increase # of cg iters,
  but trading off with solving the original problem better

## Files
* `experiments/advection` command line interface to solve the advection equation

## Commands
Advection equation (error aroud (1.e-3, 4.e-3) and it runs fast.
```shell
py experiments/advection.py --cgtol=1e-8 --maxcgiter=100 --mu=1e-8 --odetol=1e-3 --method=rk23 --grid-size=10_000 --batch-size=10_000 --dtype=single --epochs=100_000
```
