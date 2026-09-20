import unittest

from experiments.alternating_left_transfer import (
    first_failure_macrotime,
    half_step,
    lifetime_profile,
    macro_step,
    period_constraint_step,
    scan_period_words,
    self_test,
    survivor_condition,
)


class AlternatingLeftTransferTest(unittest.TestCase):
    def test_controls(self):
        self.assertEqual(self_test(), {
            "half_step_identity_mismatches": 0,
            "low_bit_condition_mismatches": 0,
            "alternating_generalization_mismatches": 0,
        })

    def test_low_pair_condition(self):
        self.assertFalse(survivor_condition(0b00))
        self.assertTrue(survivor_condition(0b01))
        self.assertTrue(survivor_condition(0b10))
        self.assertFalse(survivor_condition(0b11))

    def test_two_step_composition(self):
        for n in range(64):
            self.assertEqual(macro_step(n), half_step(half_step(n, 0), 1))
            state, valid = period_constraint_step(n, "01")
            self.assertEqual(state, macro_step(n))
            self.assertEqual(valid, survivor_condition(n))

    def test_failure_time(self):
        self.assertEqual(first_failure_macrotime(0, 10), 0)
        self.assertEqual(first_failure_macrotime(1, 10), 3)

    def test_exhaustive_small_profiles(self):
        maxima = [lifetime_profile(w)["maximum_first_failure_macrotime"]
                  for w in range(1, 5)]
        self.assertEqual(maxima, [3, 3, 3, 3])
        self.assertTrue(all(lifetime_profile(w)["right_censored_count"] == 0
                            for w in range(1, 5)))

    def test_small_general_period_scan(self):
        scan = scan_period_words(4, 4, 30)
        self.assertEqual([row["period"] for row in scan], [2, 3, 4])
        self.assertTrue(all(row["right_censored_count"] == 0 for row in scan))


if __name__ == "__main__":
    unittest.main()
