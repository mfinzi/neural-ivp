import time
import os
from jax.random import PRNGKey
from jax.random import uniform
from jax import numpy as jnp
from matplotlib import pyplot as plt
from utils.general import print_time_taken
from gr.samplers import sample_from_residuals

key = PRNGKey(seed=21)
dim_n = 2
# obs_n, num_samples = int(1.e1), 10
obs_n, num_samples = int(1.e1), int(1.e4)
# obs_n, num_samples = int(1.e4), 100
# obs_n, num_samples = int(1.e5), int(1.e4)
x = uniform(key, shape=(obs_n, dim_n))
residuals = jnp.sum(x ** 2, axis=-1, keepdims=True)
t0 = time.time()
samples = sample_from_residuals(residuals, num_samples=num_samples, key=key)
t1 = time.time()
print_time_taken(t1 - t0)
plot_res = True if obs_n < int(1.e3) else False

if plot_res:
    residuals_norm = residuals / jnp.sum(residuals)
    plt.figure()
    plt.title('Samples')
    values, counts = jnp.unique(samples, return_counts=True)
    plt.vlines(values, 0, ymax=counts, lw=10)
    plt.xlim([0, obs_n])
    plt.xticks(jnp.arange(obs_n))
    file_path = os.environ['HOME'] + '/samples.png'
    plt.savefig(file_path)
    # plt.show()

    plt.figure()
    plt.title('Actual')
    plt.bar(jnp.arange(residuals.shape[0]), residuals_norm[:, 0])
    file_path = os.environ['HOME'] + '/hist.png'
    plt.xticks(jnp.arange(obs_n))
    plt.savefig(file_path)
    # plt.show()
    print('PLOTTED!')
