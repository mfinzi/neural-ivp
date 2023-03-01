import unittest
import numpy as np
from jax import numpy as jnp
from jax import jit
from jax.config import config
from jax.random import PRNGKey
from gr.linalg import compute_power_method
from gr.linalg import get_max_eigenvalue

config.update("jax_enable_x64", True)


class TestLinAlg(unittest.TestCase):
    def test_power_jnp(self):
        dtype = jnp.float32
        key = PRNGKey(seed=21)
        max_iters, tolerance = 200, 1.e-6
        A = jnp.diag(jnp.array([10., 9.5, 3., 0.1], dtype=dtype))
        A_fn = LinearOp(A, dtype=dtype)
        actual = jnp.array(10., dtype=dtype)
        fn = jit(get_max_eigenvalue, static_argnums=(0, ))
        fn = get_max_eigenvalue
        soln, iters = fn(A_fn, key, max_iters, tolerance)
        print("TEST: Power Method")
        print(f"Iters: {iters:4d}")
        print(f"Max_eigen: {soln:1.3e}")
        diff = jnp.linalg.norm(actual - soln)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / jnp.linalg.norm(actual)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-4)
        self.assertLessEqual(rel_diff, 1e-4)

    def test_power_np(self):
        np.random.seed(seed=21)
        max_iters, tolerance = 200, 1.e-7
        A = np.diag(np.array([10., 9.5, 3.]))
        A_fn = LinearOp(A, dtype=np.float32)
        actual = 10.
        soln, iters = compute_power_method(A_fn, max_iters, tolerance)
        print("TEST: Power Method NumPy")
        print(f"Iters: {iters:4d}")
        print(f"Max_eigen: {soln:1.3e}")
        diff = np.linalg.norm(actual - soln)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / np.linalg.norm(actual)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-5)
        self.assertLessEqual(rel_diff, 1e-5)


class LinearOp:
    def __init__(self, A, dtype):
        self.A = A
        self.shape = (A.shape[0], A.shape[0])
        self.dtype = dtype

    def __call__(self, x):
        return self.A @ x


if __name__ == "__main__":
    unittest.main()
