"""Hostile tests for the independent alternating-charge verifier."""

import copy
import json
from pathlib import Path
import unittest

from tools.verify_alternating_charge_counterexample import verify


class AlternatingChargeVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (Path(__file__).resolve().parents[1] /
                "runs/theory/alternating-charge-counterexample.json")
        cls.report = json.loads(path.read_text(encoding="utf-8"))

    def test_exact_witness_and_fixed_weight_maximum(self):
        result = verify(self.report)
        self.assertTrue(result["verified"])
        self.assertEqual(result["support_zero_based"], [0, 7, 17, 19, 25, 32])
        self.assertEqual(result["weight"], 6)
        self.assertEqual(result["first_failure_macrotime_q"], 26)
        self.assertEqual(result["fixed_weight_six_maximum_q"], 26)
        self.assertEqual(result["prefix_charges"], {"N_50": -2, "N_52": -4})

    def test_rule_controls_are_retained_with_correct_scope(self):
        result = verify(self.report)
        self.assertTrue(result["rule_15_zero_prefix_survives"])
        self.assertTrue(result["rule_90_zero_prefix_survives_necessary_constraints"])

    def test_rejects_tampered_witness_failure(self):
        damaged = copy.deepcopy(self.report)
        damaged["witness"]["first_failure_macrotime_q"] = 25
        with self.assertRaisesRegex(ValueError, "truth-table replay"):
            verify(damaged)

    def test_rejects_tampered_sparse_enumeration(self):
        damaged = copy.deepcopy(self.report)
        damaged["fixed_weight_six_theorem"]["surviving_prefix_counts"][25] = 0
        with self.assertRaisesRegex(ValueError, "enumeration"):
            verify(damaged)

    def test_rejects_tampered_free_boundary(self):
        damaged = copy.deepcopy(self.report)
        damaged["witness"]["free_boundary_s_26"] ^= 1
        with self.assertRaisesRegex(ValueError, "truth-table replay|free boundary"):
            verify(damaged)

    def test_rejects_control_and_broadened_scope(self):
        damaged = copy.deepcopy(self.report)
        damaged["negative_controls"]["rule_90"]["verified_bound"] = True
        with self.assertRaisesRegex(ValueError, "Rule 90"):
            verify(damaged)
        damaged = copy.deepcopy(self.report)
        damaged["claim_limit"] = "All finite left halves are excluded."
        with self.assertRaisesRegex(ValueError, "claim limit"):
            verify(damaged)

    def test_rejects_extra_schema_field(self):
        damaged = copy.deepcopy(self.report)
        damaged["witness"]["unsupported_claim"] = True
        with self.assertRaisesRegex(ValueError, "witness schema"):
            verify(damaged)


if __name__ == "__main__":
    unittest.main()
