#!/usr/bin/env python3

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT))

import nersissian_audit as na
from prize_lab import center_bits_int


class PublishedSupportRegressionTest(unittest.TestCase):
    EXPECTED = {
        1: {0},
        2: {1},
        3: {1},
        4: {2},
        5: {2, 3, 4},
        6: {3, 5, 7},
        7: {3, 5, 8},
        8: {4, 6, 8, 9, 12, 14, 16},
    }

    def test_supports_match_published_small_cases(self):
        for m, expected in self.EXPECTED.items():
            with self.subTest(m=m):
                got, _ = na.build_support(m)
                self.assertEqual(got, expected)


class LucasTest(unittest.TestCase):
    def test_lucas_parity_examples(self):
        self.assertEqual(na.lucas_binomial_parity(5, 1), 1)
        self.assertEqual(na.lucas_binomial_parity(5, 2), 0)
        self.assertEqual(na.lucas_binomial_parity(7, 3), 1)
        self.assertEqual(na.lucas_binomial_parity(3, 4), 0)


class CenterColumnValidationTest(unittest.TestCase):
    def test_explicit_support_method_matches_independent_center_engine(self):
        expected = center_bits_int(40, margin=64)
        state = na.SequentialSupport()
        got = [state.center_bit(n)[0] for n in range(41)]
        self.assertEqual(got, expected)

    def test_cold_and_warm_return_same_value(self):
        state = na.SequentialSupport()
        for n in (0, 1, 2, 4, 8, 12, 16, 20):
            with self.subTest(n=n):
                cold = na.center_bit_cold(n)[0]
                warm = state.center_bit(n)[0]
                self.assertEqual(cold, warm)


class AccountingTest(unittest.TestCase):
    def test_warm_reuse_reports_no_new_layers_for_existing_support(self):
        state = na.SequentialSupport()
        state.ensure(12)
        stats = state.ensure(8)
        self.assertEqual(stats.layers_built, 0)
        self.assertEqual(stats.or_pairs, 0)

    def test_cold_construction_accounts_for_all_layers(self):
        _, stats = na.build_support(10)
        self.assertEqual(stats.layers_built, 8)
        self.assertGreater(stats.or_pairs, 0)
        self.assertGreater(stats.final_support_size, 0)

    def test_explicit_query_cost_is_support_size(self):
        support, _ = na.build_support(12)
        query = na.evaluate_from_support(support, 11)
        self.assertEqual(query.lucas_tests, len(support))


if __name__ == "__main__":
    unittest.main()
