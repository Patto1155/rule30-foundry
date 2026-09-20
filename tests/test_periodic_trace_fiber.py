import unittest

import numpy as np

from experiments.periodic_trace_fiber import (
    forced_left_prefix,
    is_left_permutive,
    maximum_zero_run,
    periodic_target,
    self_test,
    verify_trace,
)


class PeriodicTraceFiberTest(unittest.TestCase):
    def test_left_permutivity_gate(self):
        self.assertTrue(is_left_permutive(30))
        self.assertTrue(is_left_permutive(15))
        self.assertTrue(is_left_permutive(90))
        self.assertFalse(is_left_permutive(0))
        with self.assertRaises(ValueError):
            forced_left_prefix(8, 3, "01", rule=0)

    def test_arbitrary_words_and_rotations_are_realized(self):
        for word in ("0", "1", "01", "001", "0110"):
            for rotation in range(len(word)):
                with self.subTest(word=word, rotation=rotation):
                    rows, _, target, values = forced_left_prefix(
                        9, 4, word, rotation)
                    self.assertEqual(len(values), 16)
                    self.assertEqual(verify_trace(rows, target), 0)
                    self.assertTrue(np.array_equal(
                        target, periodic_target(word, 10, rotation)))

    def test_selected_right_halves_preserve_integer_encoding(self):
        rows, left, target, values = forced_left_prefix(
            10, 5, "01", right_values=[0, 3, 17])
        self.assertEqual(values.tolist(), [0, 3, 17])
        self.assertEqual(left.shape, (3, 10))
        self.assertEqual(verify_trace(rows, target), 0)

    def test_rule90_negative_control_has_finite_periodic_fibre(self):
        rows, left, target, _ = forced_left_prefix(
            12, 1, "0", rule=90, right_values=[1])
        self.assertEqual(verify_trace(rows, target, 90), 0)
        self.assertEqual(left[0].tolist(), [1] + [0] * 11)

    def test_rule15_negative_control_has_finite_alternating_fibre(self):
        rows, left, target, _ = forced_left_prefix(
            12, 1, "01", rule=15, right_values=[1])
        self.assertEqual(verify_trace(rows, target, 15), 0)
        self.assertEqual(left[0].tolist(), [0] * 12)

    def test_zero_run_report_is_one_based(self):
        left = np.array([[1, 0, 0, 1], [0, 0, 0, 1]], dtype=np.uint8)
        self.assertEqual(maximum_zero_run(left), {
            "length": 3, "row": 1, "start_depth": 1, "end_depth": 3})
        self.assertEqual(maximum_zero_run(left, 2)["start_depth"], 2)

    def test_full_self_test(self):
        check = self_test()
        self.assertEqual(check["rule30_stepper_mismatches"], 0)
        self.assertEqual(check["constructed_trace_mismatches"], 0)
        self.assertEqual(check["rule90_trace_mismatches"], 0)
        self.assertTrue(check["rule90_finite_periodic_fibre_control"])
        self.assertEqual(check["rule15_trace_mismatches"], 0)
        self.assertTrue(check["rule15_finite_alternating_fibre_control"])


if __name__ == "__main__":
    unittest.main()
