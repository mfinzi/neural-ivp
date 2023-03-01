import unittest
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey, normal
from jax.scipy.linalg import solve_triangular
from jax.config import config
from gr.Preconditioners import pivoted_chol
from gr.Preconditioners import JaxPivCholPrecond
from gr.Preconditioners import pivoted_chol_np
from gr.Preconditioners import get_random_nys_approx
from gr.Preconditioners import construct_random_nys_precond
from gr.Preconditioners import get_error_norm_estimate
from gr.Preconditioners import determine_rank_adaptively

config.update("jax_enable_x64", True)


class TestPreconditioners(unittest.TestCase):
    def test_determine_rank_adaptively(self):
        dtype = jnp.float32
        N = int(1 * 1e2)
        key = PRNGKey(seed=21)
        low_rank = int(N * 0.3)
        approx_args = (
            jnp.array(N, dtype=jnp.int32),
            jnp.array(1e-16, dtype=dtype),
            dtype,
        )
        adaptive_args = (
            jnp.array(16, dtype=jnp.int32),
            jnp.array(int(N * 0.9), dtype=jnp.int32),
            jnp.array(20, dtype=jnp.int32),
            jnp.array(1e-1 * N, dtype=dtype),
        )
        L = normal(key, shape=(low_rank, N), dtype=dtype)
        A = L.T @ L + 1e-2 * jnp.eye(N, dtype=dtype)

        def A_fn(x):
            return A @ x

        rank = determine_rank_adaptively(
            A=A_fn, key=key, approx_args=approx_args, adaptive_args=adaptive_args
        )
        print("\nTEST: Adaptive Rank Determination")
        print(f"Rank: {rank:1.5e}")
        diff = 1.0
        print(f"Diff: {diff:1.3e}")
        self.assertLessEqual(diff, 1e-0)

    def test_error_norm_estimate(self):
        dtype = jnp.float32
        N, rank = int(1 * 1e2), 90
        key = PRNGKey(seed=21)
        L = normal(key, shape=(N, N), dtype=dtype)
        A = L.T @ L + 1e-2 * jnp.eye(N, dtype=dtype)
        Lambda_full, U_full = jnp.linalg.eigh(A)
        Lambda, U = Lambda_full[-rank:], U_full[:, -rank:]
        E = A - U @ jnp.diag(Lambda) @ U.T
        aux, _ = jnp.linalg.eigh(E)
        max_eigen = aux[-1]

        def A_fn(x):
            return A @ x

        pow_iters = jnp.array(int(2e1), dtype=jnp.int32)
        norm_estimate = get_error_norm_estimate(
            key=key, A=A_fn, Lambda=Lambda, U=U, pow_iters=pow_iters
        )
        print("\nTEST: Error Norm Estimate")
        print(f"Actual norm:    {max_eigen:1.5e}")
        print(f"Estimated norm: {norm_estimate:1.5e}")
        diff = jnp.linalg.norm(norm_estimate - max_eigen)
        print(f"Diff: {diff:1.3e}")
        self.assertLessEqual(diff, 1e-0)

    def test_precond(self):
        # dtype = jnp.float32
        dtype = jnp.float64
        N, rank = int(5 * 1e2), 500
        key = PRNGKey(seed=21)
        L = normal(key, shape=(N, N), dtype=dtype)
        matrix = L.T @ L

        def A_fn(x):
            return matrix @ x

        Lambda, U = get_random_nys_approx(
            A=A_fn, rank=rank, obs_n=N, key=key, dtype=dtype
        )

        l_ell, mu = jnp.min(Lambda), 1e-6
        identity = jnp.eye(N, dtype=dtype)
        cons = l_ell + mu
        center = jnp.diag(Lambda) + mu * identity
        inv_center = jnp.diag(1.0 / (Lambda + mu))

        P = (1.0 / cons) * U @ center @ U.T
        P += identity - U @ U.T
        P_inv = cons * U @ inv_center @ U.T
        P_inv += identity - U @ U.T
        z = normal(key, shape=(N, 1), dtype=dtype)
        precond_fn = construct_random_nys_precond(Lambda, U, mu)
        print("\nTEST: Precond")
        diff = jnp.linalg.norm(P_inv @ z - precond_fn(z))
        print(f"Rel Diff: {diff:1.3e}")
        self.assertLessEqual(diff, 1e-5)
        diff = jnp.linalg.norm(z - P @ precond_fn(z))
        # aux = P @ precond_fn(z)
        print(f"Rel Diff: {diff:1.3e}")
        self.assertLessEqual(diff, 1e-1)

    def test_random_nys_approx(self):
        # dtype = jnp.float64
        dtype = jnp.float32
        N, rank = int(5 * 1e2), 500
        key = PRNGKey(seed=21)
        L = normal(key, shape=(N, N), dtype=dtype)
        # matrix = L @ L.T + 1. * jnp.eye(N, dtype=dtype)
        matrix = L.T @ L
        # e, _ = jnp.linalg.eig(matrix)

        def A_fn(x):
            return matrix @ x

        Lambda, U = get_random_nys_approx(
            A=A_fn, rank=rank, obs_n=N, key=key, dtype=dtype
        )
        matrix_nys = (U * Lambda) @ U.T
        print("\nTEST: Random matrix approx")
        diff = jnp.linalg.norm(matrix - matrix_nys) / jnp.linalg.norm(matrix)
        print(f"Rel Diff: {diff:1.3e}")
        self.assertLessEqual(diff, 1e-4)
        diff = jnp.linalg.norm(matrix - matrix_nys)
        print(f"Abs Diff: {diff:1.3e}")
        self.assertLessEqual(diff, 1e-1)

    def test_jax_solves(self):
        test_tol = 1e-2
        key = PRNGKey(seed=21)
        dtype = jnp.float32
        data_n = int(1e2)
        A = normal(key, shape=(150, data_n), dtype=dtype)
        rhs = normal(key, shape=(data_n,))
        matrix = A.T @ A + 1.0 * jnp.eye(data_n)
        precond = JaxPivCholPrecond(rank=data_n)
        precond.construct_matrix(
            get_diag=lambda: jnp.diag(matrix), get_row=lambda i: matrix[i, :]
        )
        # check = precond(rhs)
        lr = precond.piv_chol[:, precond.indices].T
        aux = jnp.linalg.solve(matrix, rhs)
        vec_mod = rhs[precond.indices]
        check = solve_triangular(lr.T, b=solve_triangular(lr, vec_mod, lower=True))
        check = check[precond.inv_indices]
        aux = jnp.linalg.solve(matrix, rhs)

        abs_err = jnp.linalg.norm(aux - check)
        rel_err = abs_err / jnp.linalg.norm(aux)
        print("\nTEST: JAX Solves")
        print(f"Abs error {abs_err:1.3e}")
        print(f"Relative error {rel_err:1.3e}")
        self.assertLessEqual(abs_err, test_tol)
        print("Check approx")
        diff = precond.piv_chol.T @ precond.piv_chol - matrix
        print(f"Rank approx {jnp.linalg.norm(diff):1.3e}")

    def test_pivoted_numpy(self):
        test_tol = 1e-4
        key = PRNGKey(seed=21)
        dtype = jnp.float32
        data_n = int(1e1)
        A = normal(key, shape=(150, data_n), dtype=dtype)
        matrix = A.T @ A + jnp.eye(data_n)

        def get_diag():
            return np.diag(matrix).copy()

        def get_row(i):
            return matrix[i, :]

        lr = pivoted_chol_np(get_diag, get_row, M=data_n)
        abs_err = jnp.linalg.norm(np.matmul(lr.T, lr) - matrix)
        rel_err = abs_err / jnp.linalg.norm(matrix)
        print("\nTEST: NumPy")
        print(f"Abs error {abs_err:1.3e}")
        print(f"Relative error {rel_err:1.3e}")
        self.assertLessEqual(rel_err, test_tol)

    def test_pivoted_chol(self):
        test_tol = 1e-4
        key = PRNGKey(seed=21)
        # data_n = int(1e2)
        data_n = int(1e1)
        A = normal(key, shape=(150, data_n))
        matrix = A.T @ A + jnp.eye(data_n)
        diagonal = jnp.diag(matrix)

        lr, ind = pivoted_chol(
            get_diag=lambda: diagonal.copy(),
            get_row=lambda m: matrix[m, :],
            max_rank=data_n,
            err_tol=1e-10,
        )
        abs_err = jnp.linalg.norm(lr.T @ lr - matrix)
        rel_err = abs_err / jnp.linalg.norm(matrix)
        print("\nTEST: JAXPivCholesky")
        print(f"Abs error {abs_err:1.3e}")
        print(f"Relative error {rel_err:1.3e}")
        self.assertLessEqual(rel_err, test_tol)


if __name__ == "__main__":
    unittest.main()


def get_random_nys_approx_np(A, rank, obs_n, eps=1e-6):
    Omega = np.random.normal(size=(obs_n, rank))
    Omega, _ = np.linalg.qr(Omega, mode="reduced")
    Y = np.copy(A(Omega))
    nu = eps * np.linalg.norm(Y, ord="fro")
    Y += nu * Omega
    C = np.linalg.cholesky(Omega.T @ Y)
    aux = solve_triangular(C, Y.T, lower=True)
    B = aux.T
    U, Sigma, _ = np.linalg.svd(B, full_matrices=False)
    Lambda = np.clip(Sigma**2.0 - nu, a_min=0.0, a_max=np.inf)
    return Lambda, U
