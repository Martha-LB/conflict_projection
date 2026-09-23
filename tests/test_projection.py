import unittest

import numpy as np

from conflict_projection.projection import information_projection


class ProjectionTests(unittest.TestCase):
    def test_lower_bound_moves_mass_to_high_feature_value(self):
        prior = np.array([0.5, 0.5])
        matrix = np.array([[0.0], [1.0]])
        result = information_projection(
            prior,
            matrix,
            np.array([0.75]),
            np.array([1.0]),
            feature_names=("quality",),
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.distribution[1], 0.75 - 1e-6)
        self.assertLess(result.max_violation, 1e-6)

    def test_satisfied_inequality_stops_at_prior(self):
        prior = np.array([0.5, 0.5])
        matrix = np.array([[0.0], [1.0]])
        result = information_projection(
            prior,
            matrix,
            np.array([0.25]),
            np.array([1.0]),
            feature_names=("quality",),
        )
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.distribution, prior, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

