import unittest
import numpy as np
from gr.ode_solver import solve_ivp_rk4


class TestODE(unittest.TestCase):

    def test_ivp_rk4(self):
        step_size = 0.0001
        nT = 10
        tf = 0.001
        y0 = np.array([3.])
        times = np.arange(nT) * (tf / nT) + 2.
        def fn(t0, y0): return 7. * y0 ** 2. * t0 ** 3.
        t_span = (times[0], times[-1])
        ys, ts = solve_ivp_rk4(fn, y0, t_span, step_size=step_size)
        ans = -1. / ((7. / 4.) * times ** 4 - 85. / 3.)

        abs_err = np.linalg.norm(ys.reshape(-1) - ans)
        rel_err = abs_err / np.linalg.norm(ans)
        print("\nTEST: ODE")
        print(f"Abs error {abs_err:1.3e}")
        print(f"Relative error {rel_err:1.3e}")
        self.assertLessEqual(rel_err, 1e-4)


if __name__ == "__main__":
    unittest.main()
