import unittest
import numpy as np
from gr.np_linalg import compute_power_method
from gr.np_linalg import construct_tri
from gr.np_linalg import do_double_gram
from gr.np_linalg import get_max_eigenval_and_vec
from gr.np_linalg import run_randomized_lanczos_full
from gr.np_linalg import run_randomized_lanczos
from gr.np_linalg import LinearOp


class TestNPLinAlg(unittest.TestCase):
    def test_randomized_lanczos(self):
        dtype = np.float32
        actual = 10.
        max_iter, tolerance = 100, 1.e-7
        A = np.diag(np.array([actual, 9.5, 3.])).astype(dtype)
        A_fn = LinearOp(A, dtype=dtype)
        print("TEST: Randomized Lanczos")
        soln, iters = run_randomized_lanczos(A_fn, max_iter, dtype, tolerance)
        print(f"Iters: {iters:4d}")
        print(f"Max_eigen: {soln:1.3e}")
        diff = np.abs(actual - soln)
        print(f"Abs Diff: {diff:1.3e}")

    def test_randomized_lanczos_full(self):
        dtype = np.float32
        actual = 10.
        max_iter, tolerance = 100, 1.e-8
        A = np.diag(np.array([actual, 9.5, 3.])).astype(dtype)
        A_fn = LinearOp(A, dtype=dtype)
        print("TEST: Randomized Lanczos Full")
        soln, _, iters = run_randomized_lanczos_full(A_fn, max_iter, dtype, tolerance)
        print(f"Iters: {iters:4d}")
        print(f"Max_eigen: {soln:1.3e}")
        diff = np.abs(actual - soln)
        print(f"Abs Diff: {diff:1.3e}")

    def test_retrieval_from_tri(self):
        dtype = np.float32
        alpha = np.array([1.2, 3.4, 5.5, 3.1], dtype=dtype)
        beta = np.array([0.1, 0.04, 0.3], dtype=dtype)
        N, d = 10, 4
        vec = np.random.normal(size=(N, d)).astype(dtype)
        xi, y = get_max_eigenval_and_vec(vec, alpha, beta)
        print("TEST: Max Eigenval from T")
        self.assertTrue(y.shape == (N,))
        self.assertTrue(y.dtype == dtype)
        self.assertTrue(xi > 0)

    def test_double_gram(self):
        np.random.seed(seed=21)
        dtype = np.float32
        # dtype = np.float64
        N, d = 10, 5
        vec = np.random.normal(size=(N, d)).astype(dtype)
        vec[:, 0] = vec[:, 0] / np.linalg.norm(vec[:, 0])
        for j in range(1, d):
            vec = do_double_gram(vec, ind=j)
            vec[:, j] = vec[:, j] / np.linalg.norm(vec[:, j])
        print("TEST: Double Gram-Schmidt")
        cases = [(-1, 0), (-1, 1), (1, 3), (3, 4)]
        for idx, jdx in cases:
            diff = np.abs(vec[:, idx].T @ vec[:, jdx])
            print(f"Abs Diff: {diff:1.3e}")
            self.assertLessEqual(diff, 1e-7)

    def test_construct_tri(self):
        alpha = np.array([1.2, 3.4, 5.5, 0.1])
        beta = np.array([0.1, 6.4, 1.3])
        actual = np.array([
            [1.2, 0.1, 0., 0.],
            [0.1, 3.4, 6.4, 0.],
            [0., 6.4, 5.5, 1.3],
            [0., 0., 1.3, 0.1]
        ])
        soln = construct_tri(alpha, beta)
        print("TEST: Tridiag construction")
        diff = np.linalg.norm(actual - soln)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / np.linalg.norm(actual)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-4)
        self.assertLessEqual(rel_diff, 1e-4)

    def test_power(self):
        np.random.seed(seed=21)
        dtype = np.float32
        max_iters, tolerance = 200, 1.e-6
        actual = 10.
        A = np.diag(np.array([actual, 9.5, 3.])).astype(dtype)
        A_fn = LinearOp(A, dtype=dtype)
        soln, iters = compute_power_method(A_fn, max_iters, tolerance, dtype)
        print("TEST: Power Method NumPy")
        print(f"Iters: {iters:4d}")
        print(f"Max_eigen: {soln:1.3e}")
        diff = np.linalg.norm(actual - soln)
        print(f"Abs Diff: {diff:1.3e}")
        rel_diff = diff / np.linalg.norm(actual)
        print(f"Rel Diff: {rel_diff:1.3e}")
        self.assertLessEqual(diff, 1e-4)
        self.assertLessEqual(rel_diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
