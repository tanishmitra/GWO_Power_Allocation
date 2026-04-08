from __future__ import annotations

import unittest

import numpy as np

from isac_power_allocation.constraints import project_power_allocation


class ConstraintTests(unittest.TestCase):
    def test_simplex_projection_preserves_budget(self) -> None:
        allocation = project_power_allocation(np.array([1.0, -2.0, 3.0]), total_power_w=5.0)
        self.assertAlmostEqual(float(allocation.sum()), 5.0, places=8)
        self.assertTrue(np.all(allocation >= 0.0))

    def test_upper_bounds_are_respected(self) -> None:
        allocation = project_power_allocation(
            np.array([10.0, 0.1, 0.1, 0.1]),
            total_power_w=2.0,
            per_subcarrier_max_power_w=0.75,
        )
        self.assertTrue(np.all(allocation <= 0.75 + 1e-10))
        self.assertLessEqual(float(allocation.sum()), 2.0 + 1e-10)


if __name__ == "__main__":
    unittest.main()
