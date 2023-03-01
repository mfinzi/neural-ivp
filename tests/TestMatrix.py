import unittest
from functools import partial
from jax import numpy as jnp
from utils.matrix_transforms import evaluate_propagate_mvm


class TestMatrix(unittest.TestCase):

    def test_precond(self):
        dtype = jnp.float32
        block_size = 3
        A = [[1., 0., 0.], [1., 1., 0.], [1., 0., 2.]]
        A = jnp.array(A, dtype=dtype)
        vector = jnp.array([1., 1., 1., 2., 2., 2., 3., 3., 3.], dtype=dtype)
        aux = [1., 2., 3., 2., 4., 6., 3., 6., 9.]
        aux = jnp.array(aux, dtype=dtype)
        A_fn = partial(evaluate_propagate_mvm, A, block_size=block_size)
        check = A_fn(vector)
        print('\nTEST: Matrix Transform')
        diff = jnp.linalg.norm(check - aux)
        print(f'Rel Diff: {diff:1.3e}')
        self.assertLessEqual(diff, 1.e-5)


if __name__ == '__main__':
    unittest.main()
