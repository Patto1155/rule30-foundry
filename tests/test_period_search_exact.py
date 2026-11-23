"""Pure vs eventual periodicity in experiments/period_search_exact.py.

The position-0 test decides pure periodicity only. These tests pin that blind
spot and check that the tail-anchored test recovers the preperiod exactly.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from period_search_exact import (  # noqa: E402
    find_eventual_periods, find_periods, overlap_length)


def naive_min_preperiod(bits, p):
    last = -1
    for i in range(bits.size - p):
        if bits[i] != bits[i + p]:
            last = i
    return last + 1


class EventualPeriodTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.block = rng.integers(0, 2, 97, dtype=np.uint8)

    def stream(self, transient, n=20_000):
        s = np.resize(self.block, n).copy()
        s[:transient] ^= 1
        return s

    def test_pure_period_found_by_both(self):
        s = self.stream(0)
        self.assertEqual(find_periods(s, 64)["confirmed_periods"][0]["period"], 97)
        hit = find_eventual_periods(s, 64)["eventual_periods_confirmed_by_window"][0]
        self.assertEqual((hit["period"], hit["preperiod"]), (97, 0))

    def test_position_zero_test_misses_one_bit_transient(self):
        s = self.stream(1)
        self.assertEqual(find_periods(s, 64)["confirmed_periods"], [])
        hit = find_eventual_periods(s, 64)["eventual_periods_confirmed_by_window"][0]
        self.assertEqual((hit["period"], hit["preperiod"]), (97, 1))

    def test_preperiod_matches_naive(self):
        for k in (1, 5, 40, 300):
            s = self.stream(k)
            hits = find_eventual_periods(s, 64)["eventual_periods_confirmed_by_window"]
            self.assertEqual(hits[0]["period"], 97)
            self.assertEqual(hits[0]["preperiod"], naive_min_preperiod(s, 97))

    def test_tail_agreement_is_exact_on_random(self):
        rng = np.random.default_rng(3)
        s = rng.integers(0, 2, 200_000, dtype=np.uint8)
        res = find_eventual_periods(s, 64)
        self.assertEqual(res["eventual_periods_confirmed_by_window"], [])
        rev = np.ascontiguousarray(s[::-1])
        brute = max(overlap_length(rev, p, cap=64) for p in range(1, s.size))
        # The narrow 16-bit scan is exact at or above 16 and reports 0 below.
        self.assertGreaterEqual(brute, 16)
        self.assertEqual(res["longest_tail_agreement_bits"], brute)


if __name__ == "__main__":
    unittest.main()
