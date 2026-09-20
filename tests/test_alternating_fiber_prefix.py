import unittest

import numpy as np

from experiments.alternating_fiber_prefix import forced_left_prefix, phase_report, verify


class AlternatingFiberPrefixTest(unittest.TestCase):
    def test_constructed_rows_have_the_requested_trace(self):
        for phase in (0, 1):
            with self.subTest(phase=phase):
                rows, _, target = forced_left_prefix(8, 4, phase)
                self.assertEqual(verify(rows, target), 0)

    def test_near_centre_constraints_are_reproduced(self):
        phase_zero = phase_report(8, 4, 0)
        phase_one = phase_report(8, 4, 1)
        self.assertEqual(phase_zero["stepper_mismatches"], 0)
        self.assertEqual(phase_one["stepper_mismatches"], 0)
        self.assertTrue(phase_zero["near_centre"]["L1_ne_L2"])
        self.assertTrue(phase_one["near_centre"]["L2_ne_L3"])

    def test_every_right_prefix_has_one_constructed_left_prefix(self):
        _, left, _ = forced_left_prefix(7, 5, 1)
        self.assertEqual(left.shape, (1 << 5, 7))
        self.assertTrue(np.isin(left, (0, 1)).all())


if __name__ == "__main__":
    unittest.main()
