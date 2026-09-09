#!/usr/bin/env python3
"""Unit tests for experiments/compress_probe.py — pure functions only.

The two tested functions, `ratio` and `geometric_expected_hist`, need no
data file, so these tests run on a fresh clone where the canonical
bitstreams (data/center_col_*.bin) are gitignored and absent. Everything
else in the module (`main`, data loading, plotting) is deliberately not
covered here: it requires the absent 46M-bit stream.

Bit-order note: the module itself unpacks its center-column bytes with
`bitorder="little"` (the repo-wide convention for LSB-first dumps; a bare
np.unpackbits reverses every 8-bit block). This test file does not pack or
unpack anything, so it has no bit-order surface of its own.
"""

import os
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import compress_probe as cp

N = 10_000


class RatioTest(unittest.TestCase):
    """`ratio(data, comp)` = compressed size / original size."""

    def test_empty_input_has_ratio_one(self):
        """Original-size division is guarded: len 0 would be a divide-by-zero."""
        self.assertEqual(cp.ratio(b"", "gzip"), 1.0)
        self.assertEqual(cp.ratio(b"", "bz2"), 1.0)

    def test_ratio_is_far_below_one_for_runs(self):
        """Measured value on b'\\x00' * 10000: gzip 0.0045, bz2 0.0046."""
        for comp in ("gzip", "bz2"):
            with self.subTest(comp=comp):
                self.assertLess(cp.ratio(b"\x00" * N, comp), 0.01)

    def test_runs_compress_far_below_random(self):
        """The ordering the experiment's matched-baseline design depends on:
        highly compressible input must compress far better than os.urandom
        input of the same length. Assert the ordering, not a magic constant.
        Measured on this interpreter: gzip zero=0.0045 vs rand=1.0023;
        bz2 zero=0.0046 vs rand=1.0467 — r_zero stays under r_rand/4."""
        for comp in ("gzip", "bz2"):
            with self.subTest(comp=comp):
                r_rand = cp.ratio(os.urandom(N), comp)
                r_zero = cp.ratio(b"\x00" * N, comp)
                self.assertLess(r_zero, r_rand / 4.0,
                                f"{comp}: runs {r_zero:.4f} must be far below "
                                f"random {r_rand:.4f} on {N} bytes")

    def test_unknown_compressor_raises(self):
        with self.assertRaises(ValueError):
            cp.ratio(b"abc", "xz")


class GeometricExpectedHistTest(unittest.TestCase):
    """`geometric_expected_hist(total_runs)` — P(run=k) = (0.5)**k for
    k = 1..6, plus a '7+' tail bin with the same expected count as k=6."""

    def test_sum_matches_total_within_float_tolerance(self):
        for total in (0, 1, 7, 4096, 1_000_000, 5_750_001):
            with self.subTest(total=total):
                hist = cp.geometric_expected_hist(total)
                self.assertAlmostEqual(sum(hist.values()), float(total),
                                       places=6, msg=f"total={total}")

    def test_halves_at_each_successive_run_length(self):
        for total in (4096, 1_000_000):
            with self.subTest(total=total):
                hist = cp.geometric_expected_hist(total)
                for k in range(1, 6):
                    self.assertEqual(hist[str(k + 1)], hist[str(k)] / 2.0,
                                     msg=f"k={k}: hist[k+1] should be "
                                         f"hist[k]/2, total={total}")

    def test_value_for_a_literal_total(self):
        """Pinned to numbers derived by running the function: total=4096
        gives 2048.0 at run length 1, 64.0 in the '7+' tail, summing to
        the 4096.0 it was handed."""
        hist = cp.geometric_expected_hist(4096)
        self.assertEqual(hist["1"], 2048.0)
        self.assertEqual(hist["7+"], 64.0)
        self.assertEqual(sum(hist.values()), 4096.0)

    def test_tail_bin_matches_the_k6_count(self):
        hist = cp.geometric_expected_hist(1024)
        self.assertEqual(hist["6"], hist["7+"])

    def test_keys_are_the_expected_bins(self):
        self.assertEqual(set(cp.geometric_expected_hist(100).keys()),
                         {"1", "2", "3", "4", "5", "6", "7+"})


if __name__ == "__main__":
    unittest.main()