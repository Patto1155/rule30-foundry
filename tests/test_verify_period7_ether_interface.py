"""Independent and hostile checks for the period-7 interface certificate."""

import copy
import json
from pathlib import Path
import unittest

from tools.verify_period7_ether_interface import verify


class Period7EtherInterfaceVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((Path(__file__).resolve().parents[1] /
            "runs/theory/period7-ether-interface.json").read_text())

    def test_exact_interface_cycle_and_all_r_bounds(self):
        result = verify(self.report)
        self.assertEqual(result["combined_state_first_return_samples"], 512)
        self.assertEqual(result["arrival_prefix_bits"], 18)
        self.assertEqual(result["boundary_switch"],
                         "applied only after arrival when testing delays")
        self.assertTrue(result["odd_r_exact"])
        self.assertIn("left permutivity", result["odd_r_proof"])
        self.assertEqual(result["even_r_bound"], "q(n_r) <= 7r/2 + 8")
        self.assertEqual(result["all_r_bound"], "q(n_r) <= 4r")

    def test_exact_failure_delay_distribution(self):
        self.assertEqual(verify(self.report)["failure_delay_counts"], {
            "0": 256, "1": 128, "2": 64, "4": 32,
            "5": 24, "6": 4, "7": 2, "8": 2,
        })

    def test_rejects_tampered_truth_table_circuit(self):
        damaged = copy.deepcopy(self.report)
        damaged["ether_phases_low_index_first"][2] = "0010011"
        with self.assertRaisesRegex(ValueError, "ether circuit"):
            verify(damaged)

    def test_rejects_tampered_delay_table(self):
        damaged = copy.deepcopy(self.report)
        damaged["even_r_delay_by_m_mod_512_for_r_2m"][37] = 0
        with self.assertRaisesRegex(ValueError, "delay table"):
            verify(damaged)

    def test_rejects_period_that_is_not_initial_state_return(self):
        damaged = copy.deepcopy(self.report)
        damaged["pure_sample_cycle_length"] = 256
        with self.assertRaisesRegex(ValueError, "period"):
            verify(damaged)

    def test_rejects_tampered_direct_odd_radius_value(self):
        damaged = copy.deepcopy(self.report)
        damaged["direct_first_failures"][0] += 1
        with self.assertRaisesRegex(ValueError, "direct finite"):
            verify(damaged)

    def test_rejects_broadened_claim_scope(self):
        damaged = copy.deepcopy(self.report)
        damaged["claim_limit"] = "All finite left halves are excluded."
        with self.assertRaisesRegex(ValueError, "scope"):
            verify(damaged)


if __name__ == "__main__":
    unittest.main()
