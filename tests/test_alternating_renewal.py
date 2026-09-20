"""Independent cone checks and failures of uncredited block renewal."""
import unittest

from experiments.alternating_left_transfer import first_failure_macrotime
from experiments.alternating_renewal import admits_latches, check_bound


class AlternatingRenewalTests(unittest.TestCase):
    def test_truth_table_cone_matches_integer_latches(self):
        for n in range(256):
            for m in range(1, 5):
                failure = first_failure_macrotime(n, m - 1)
                self.assertEqual(admits_latches(n, m), failure is None)

    def test_existing_fixed_weight_bounds_independently(self):
        for r, bound in enumerate((0, 3, 8, 11, 16, 19)):
            report = check_bound(r, bound)
            self.assertTrue(report['verified_bound'])
            self.assertTrue(report['sharp'])
            witness = report['witness_surviving_bound_latches']
            if witness is not None:
                self.assertLessEqual(witness.bit_count(), r)
                self.assertEqual(first_failure_macrotime(witness, bound), bound)

    def test_zero_cost_control_survivors_are_retained(self):
        # Rule 15 is an actual finite alternating fibre. Rule 90 here merely
        # passes the necessary left-only constraints; this asserts no fibre.
        for rule in (15, 90):
            report = check_bound(0, 8, rule)
            self.assertFalse(report['verified_bound'])
            self.assertEqual(report['surviving_prefix_counts'], [1] * 9)

    def test_uncharged_eight_cell_renewal_is_false(self):
        n = 1 + (1 << 7)
        self.assertTrue(admits_latches(n, 8))
        self.assertEqual((n >> 8) & 255, 0)
        self.assertEqual(first_failure_macrotime(n, 8), 8)
        self.assertEqual(8 * n.bit_count() - 16, 0)

    def test_paired_one_plain_deletion_is_false(self):
        n = 1075855489
        self.assertEqual(first_failure_macrotime(n, 19), 19)
        outputs = set()
        for start in range(n.bit_length()):
            if ((n >> start) & 65535).bit_count() != 2:
                continue
            shortened = ((n & ((1 << start) - 1))
                         | ((n >> (start + 16)) << start))
            outputs.add(shortened)
        self.assertEqual(outputs, {16417, 16513})
        self.assertEqual({first_failure_macrotime(n, 11) for n in outputs},
                         {2, 10})

    def test_invalid_bounds(self):
        for weight, bound in ((-1, 0), (0, -1)):
            with self.assertRaises(ValueError):
                check_bound(weight, bound)


if __name__ == '__main__':
    unittest.main()
