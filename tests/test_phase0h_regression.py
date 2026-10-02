import unittest

import numpy as np

from ravel.metrics import score_session_model
from ravel.model import fit_session_model, refit_recruitment_only
from ravel.synthetic import make_dynamic_route_session


class Phase0HRegressionTest(unittest.TestCase):
    """Numerical fixture frozen from notebook code hash 9ab1933068..."""

    def test_dynamic_synthetic_fixture(self):
        data, atoms, _ = make_dynamic_route_session()
        static = fit_session_model(data, atoms, "active", dynamic=False)
        dynamic = fit_session_model(data, atoms, "active", dynamic=True)
        self.assertAlmostEqual(static.train_mse, 0.32352170299129307, places=9)
        self.assertAlmostEqual(dynamic.train_mse, 0.08692450061192623, places=9)
        self.assertLess(dynamic.train_mse, static.train_mse)
        nll = score_session_model(data, dynamic, "active", "test")["trial_nll"]
        np.testing.assert_allclose(
            nll,
            np.array([-3.32584503, -3.30586654, -3.32173664]),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_recruitment_only_does_not_change_residues(self):
        data, atoms, _ = make_dynamic_route_session()
        base = fit_session_model(data, atoms, "active", dynamic=True)
        adapted = refit_recruitment_only(data, base, "passive")
        np.testing.assert_allclose(adapted.coefficients, base.coefficients, atol=0.0)


if __name__ == "__main__":
    unittest.main()

