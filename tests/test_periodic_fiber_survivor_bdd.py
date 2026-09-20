import unittest

import numpy as np

from experiments.periodic_fiber_survivor_bdd import (
    robdd_stats,
    survivor_profile,
)


class PeriodicFiberSurvivorBddTest(unittest.TestCase):
    def test_constants_have_no_nonterminal_nodes(self):
        for value in (0, 1):
            stats = robdd_stats(np.full(8, value, dtype=np.uint8), (0, 1, 2))
            self.assertEqual(stats["nonterminal_nodes"], 0)
            self.assertTrue(stats["root_is_terminal"])
            self.assertEqual(stats["root_terminal_value"], value)

    def test_reduction_and_order_are_exact(self):
        assignments = np.arange(4, dtype=np.uint8)
        x0 = assignments & 1
        x1 = (assignments >> 1) & 1
        projection = robdd_stats(x0, (0, 1))
        parity = robdd_stats(x0 ^ x1, (0, 1))
        self.assertEqual(projection["nonterminal_nodes"], 1)
        self.assertEqual(parity["nonterminal_nodes"], 3)
        self.assertEqual(parity["max_layer_width"], 2)

    def test_small_rule30_profile_matches_known_horizon(self):
        profiles = [survivor_profile(1, 12, phase) for phase in (0, 1)]
        self.assertEqual(max(p["escape_depth"] for p in profiles), 7)
        self.assertTrue(all(p["verification_mismatches"] == 0 for p in profiles))

    def test_rule15_survivor_is_constant_true(self):
        profile = survivor_profile(1, 12, 0, rule=15)
        self.assertTrue(profile["right_censored"])
        self.assertEqual(profile["last_nonempty_survivors"], 2)
        self.assertEqual(profile["natural_order_robdd"]["nonterminal_nodes"], 0)
        self.assertEqual(profile["natural_order_robdd"]["root_terminal_value"], 1)


if __name__ == "__main__":
    unittest.main()
