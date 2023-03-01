import numpy as np
from gr.np_linalg import generate_spectrum
from gr.np_linalg import generate_pd_from_diag
from gr.np_linalg import run_randomized_lanczos
from gr.np_linalg import compute_power_method
from gr.np_linalg import LinearOp
from matplotlib import pyplot as plt

dtype = np.float64
coeff, scale, size = 0.15, 5., 1000
diag = generate_spectrum(coeff, scale, size, dtype)
# np.random.shuffle(diag)
A = generate_pd_from_diag(diag, dtype=dtype)
A_fn = LinearOp(A, dtype)

max_iter = 15
soln_lan, iters_lan = run_randomized_lanczos(A_fn, dtype=dtype, max_iter=max_iter, tolerance=1e-5)
print(soln_lan)
eig, _ = np.linalg.eigh(A)
eig = np.flip(eig)
soln_pow, iters_pow = compute_power_method(A_fn, dtype=dtype, max_iters=max_iter, tolerance=1e-10)
print(soln_pow)

plt.figure()
plt.plot(np.arange(diag.shape[0]) + 1, diag)
plt.plot(np.arange(eig.shape[0]) + 1, eig)
plt.show()
