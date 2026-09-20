import unittest

from experiments.alternating_left_weight import (
    induced_left_prefix,
    minimum_weight_profile,
    self_test,
    vertical_window_zero_distance,
)


class AlternatingLeftWeightTest(unittest.TestCase):
    def test_known_symbolic_prefixes(self):
        # s=000 gives e1=1, e2=0, e3=1, e4=0, e5=1, e6=0.
        self.assertEqual(induced_left_prefix(0, 3).tolist(), [1, 0, 1, 0, 1, 0])

    def test_small_exact_minima(self):
        profile = minimum_weight_profile(4)
        self.assertEqual(
            [profile[2 * m - 1]["minimum_weight"] for m in range(1, 5)],
            [1, 1, 1, 2])
        self.assertTrue(all(row["cone_charge_survives"] for row in profile))

    def test_spacetime_crosscheck(self):
        check = self_test()
        self.assertEqual(check["sideways_vs_spacetime_bit_mismatches"], 0)
        self.assertTrue(check["small_profile_matches"])

    def test_vertical_window_relaxation_has_zero_cycle(self):
        self.assertEqual(
            [vertical_window_zero_distance(height) for height in range(2, 6)],
            [2, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
