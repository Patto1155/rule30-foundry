import unittest

from tools.verify_golden_mean_phase_slip import (
    verify, verify_physical_reset_failure,
)


class GoldenMeanPhaseSlipTests(unittest.TestCase):
    def test_truth_table_identity_and_physical_counterexample(self):
        report = verify()
        self.assertEqual(report["identities"]["golden_mean_eight_bit_words"], 55)
        self.assertFalse(report["identities"]["spatial_transport_proved"])
        witness = report["physical_reset_failure"]
        self.assertEqual(witness["centre_trace"], "01" * 14 + "0")
        self.assertEqual(witness["right_even_times_0_through_26"], "10100001001001")

    def test_wrong_right_half_is_rejected_by_full_orbit_replay(self):
        with self.assertRaises(AssertionError):
            verify_physical_reset_failure(right=0)


if __name__ == "__main__":
    unittest.main()
