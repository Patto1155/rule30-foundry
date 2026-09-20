import unittest

from experiments.alternating_charge_counterexample import PARENT, WITNESS, run
from experiments.alternating_left_transfer import first_failure_macrotime
from experiments.alternating_renewal import admits_latches


class AlternatingChargeCounterexampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run()

    def test_counterexample_and_prefix_charges(self):
        witness = self.report["witness"]
        self.assertEqual(witness["support_zero_based"], [0, 7, 17, 19, 25, 32])
        self.assertEqual(witness["weight"], 6)
        self.assertEqual(witness["first_failure_macrotime_q"], 26)
        self.assertEqual(witness["q_minus_four_weight"], 2)
        self.assertEqual(witness["prefix_charges"], {"N_50": -2, "N_52": -4})

    def test_independent_truth_table_replay(self):
        self.assertTrue(admits_latches(WITNESS, 26))
        self.assertFalse(admits_latches(WITNESS, 27))

    def test_highest_one_jump(self):
        deletion = self.report["highest_one_deletion"]
        self.assertEqual(deletion["parent"], PARENT)
        self.assertEqual(first_failure_macrotime(PARENT, 27), 16)
        self.assertEqual(deletion["q_jump"], 10)
        self.assertEqual(deletion["parent_unused_four_weight_credit"], 4)
        self.assertEqual(deletion["child_four_weight_deficit"], -2)

    def test_fixed_weight_six_bound_is_sharp(self):
        theorem = self.report["fixed_weight_six_theorem"]
        self.assertTrue(theorem["verified_bound"])
        self.assertTrue(theorem["sharp"])
        self.assertEqual(theorem["witness_surviving_bound_latches"], WITNESS)
        self.assertEqual(theorem["surviving_prefix_counts"][-1], 0)

    def test_rule_controls_remain_alive(self):
        for rule in ("rule_15", "rule_90"):
            control = self.report["negative_controls"][rule]
            self.assertFalse(control["verified_bound"])
            self.assertEqual(control["surviving_prefix_counts"], [1] * 9)


if __name__ == "__main__":
    unittest.main()
