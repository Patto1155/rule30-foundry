import unittest

import numpy as np

from experiments.periodic_fiber_invariants import (
    _solve_gf2,
    check_affine_recurrence,
    check_polynomial_recurrence,
    first_escape_depth,
    fit_affine_recurrence,
    fit_polynomial_recurrence,
    horizon_row,
    linear_bound_candidates,
)


class PeriodicFiberInvariantTest(unittest.TestCase):
    def test_first_escape_depth(self):
        left = np.array([[1, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
        self.assertEqual(first_escape_depth(left, 1).tolist(), [3, 4])
        self.assertEqual(first_escape_depth(left, 4).tolist(), [5, 5])

    def test_horizon_matches_known_small_values(self):
        expected = {1: 7, 2: 7, 3: 7, 4: 7, 5: 9}
        for support, want in expected.items():
            rows = [horizon_row(support, 24, phase)[0] for phase in (0, 1)]
            self.assertEqual(max(r["maximum_first_escape_depth"] for r in rows), want)
            self.assertTrue(all(r["saturated_count"] == 0 for r in rows))

    def test_linear_bounds_use_a_real_holdout(self):
        rows = [
            {"support": 1, "maximum_first_escape_depth": 3},
            {"support": 2, "maximum_first_escape_depth": 4},
            {"support": 3, "maximum_first_escape_depth": 8},
        ]
        candidates = linear_bound_candidates(rows, train_max=2)
        slope_one = next(x for x in candidates if x["slope"] == 1)
        self.assertEqual(slope_one["intercept"], 2)
        self.assertFalse(slope_one["holdout_survives"])

    def test_gf2_solver(self):
        # x0 xor x1 = 1; x1 = 0.
        solution = _solve_gf2([(1 << 0) | (1 << 1) | (1 << 2), 1 << 1], 2)
        self.assertEqual(solution, 1)
        self.assertIsNone(_solve_gf2([1, 1 | (1 << 1)], 1))

    def test_affine_recurrence_fit_and_holdout(self):
        # Fibonacci recurrence over GF(2), duplicated across two rows.
        seq = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        left = np.vstack([seq, seq])
        solution = fit_affine_recurrence(left, 1, memory=2, modulus=1, end_depth=8)
        self.assertIsNotNone(solution)
        checked = check_affine_recurrence(left, 1, 2, 1, solution, 9)
        self.assertEqual(checked["violations"], 0)

    def test_polynomial_recurrence_includes_affine_controls(self):
        seq = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        left = np.vstack([seq, seq])
        solution, terms = fit_polynomial_recurrence(
            left, 1, memory=2, degree=2, modulus=1, end_depth=8)
        self.assertIsNotNone(solution)
        checked = check_polynomial_recurrence(
            left, 1, 2, 1, solution, terms, 9)
        self.assertEqual(checked["violations"], 0)


if __name__ == "__main__":
    unittest.main()
