"""Regression checks using independent cell evolution and old triangular solver."""
import unittest
from experiments.support_growth import completion, self_test


class SupportGrowthTests(unittest.TestCase):
    def test_independent_reference_and_controls(self):
        self.assertEqual(self_test()['cases'], 192)

    def test_rejects_invalid_input(self):
        for args in [(-1, 4), (0, -1), (0, 4, 'typo'), (0, 4, 'zero', 110)]:
            with self.assertRaises(ValueError):
                completion(*args)

    def test_empty_prefix(self):
        self.assertEqual(completion(17, 0), 0)


if __name__ == '__main__':
    unittest.main()
