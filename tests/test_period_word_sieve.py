import copy
import unittest

from experiments import period_word_sieve as sieve


class PeriodWordSieveTests(unittest.TestCase):
    def test_necklace_counts_and_rotation_coverage(self):
        words = sieve.necklaces(6)
        self.assertEqual([sum(len(w) == n for w in words) for n in range(1, 7)],
                         [2, 1, 2, 3, 6, 9])
        self.assertEqual(len(set(words)), 23)
        self.assertIn("01", words)
        self.assertNotIn("0101", words)
        self.assertNotIn("10", words)

    def test_independent_controls(self):
        for word in ("0", "1"):
            result = sieve.independent_audit(5, word)
            self.assertEqual(result["classification"], "forces_left_period")
            self.assertGreater(result["recurrent_states"], 0)
        self.assertEqual(sieve.independent_audit(5, "01")["classification"],
                         "does_not_force_left_period")

    def test_failed_implication_has_directly_checked_loops(self):
        artifact = sieve.run(2, [5])
        record = artifact["records"][-1]
        self.assertTrue(record["output_loops"]["found"])
        self.assertTrue(sieve.strip.verify_output_loops(record["output_loops"])["verified"])
        self.assertEqual(sieve.verify(artifact)["records"], 3)

    def test_rotation_invariance(self):
        audits = [sieve.independent_audit(5, word) for word in ("001", "010", "100")]
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(audits[0], audits[2])

    def test_missing_duplicate_and_tampered_records_rejected(self):
        artifact = sieve.run(1, [5])
        variants = []
        missing = copy.deepcopy(artifact)
        missing["records"].pop()
        variants.append(missing)
        duplicate = copy.deepcopy(artifact)
        duplicate["records"][1] = copy.deepcopy(duplicate["records"][0])
        variants.append(duplicate)
        tampered = copy.deepcopy(artifact)
        tampered["records"][0]["audit"]["classification"] = "center_impossible"
        variants.append(tampered)
        for variant in variants:
            with self.assertRaises(ValueError):
                sieve.verify(variant)

    def test_configuration_bounds(self):
        for period in (0, 7, True, 1.0):
            with self.assertRaises(ValueError):
                sieve.necklaces(period)
        for widths in ([], [4], [11], [5, 3], [3, 3], [True]):
            with self.assertRaises(ValueError):
                sieve.validate_widths(widths)


if __name__ == "__main__":
    unittest.main()
