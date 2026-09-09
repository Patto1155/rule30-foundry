#!/usr/bin/env python3
"""Unit tests for the pure helpers in experiments/block_frequency.py.

Covers `count_blocks` (overlapping k-bit window histograms) and
`chi_squared_test` only. `load_center_column` is deliberately NOT exercised:
it opens `data/center_col_10M.bin`, which is gitignored and absent in a fresh
checkout, so any test that reaches for it cannot run. Bit arrays are either
literal (with a histogram that can be written down by hand) or generated from
the repo's own naive CPU Rule 30 simulator (single-spike initial condition, the
canonical seed for all three prizes), never unpacked from `.bin` files.
"""
import pathlib
import sys
import types
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

# experiments/block_frequency.py imports tqdm at module scope for main()'s
# progress bar, but tqdm is not in requirements-ci.txt. Use the real module
# when present; otherwise a minimal stand-in so the pure helpers still import
# and run on a minimal CPU-only machine.
try:
    import tqdm  # noqa: F401
except ImportError:
    _fake_tqdm = types.ModuleType("tqdm")

    def _tqdm(iterable=None, **_kwargs):  # pragma: no cover - stand-in only
        return iterable if iterable is not None else iter(())

    _fake_tqdm.tqdm = _tqdm
    sys.modules.setdefault("tqdm", _fake_tqdm)

try:
    from scipy import stats  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import block_frequency as bf
from rule30_open_utils import simulate_naive_center_columns


def reference_count_blocks(bits: np.ndarray, k: int) -> np.ndarray:
    """Independent second implementation of count_blocks.

    Same convention (window value = sum bits[i+j] << j, LSB-first), but built
    by plain numpy slicing and int shifts instead of sliding_window_view @
    powers. Used to check the chunked vectorised path, including across the
    internal 2,000,000-window chunk boundary.
    """
    bits = np.asarray(bits, dtype=np.int64)
    n = len(bits) - k + 1
    pats = np.zeros(n, dtype=np.int64)
    for j in range(k):
        pats += bits[j:j + n] << j
    return np.bincount(pats, minlength=1 << k)


def canonical_center_column(n_steps: int, half_width: int) -> np.ndarray:
    """Rule 30 center column from a single black cell, via the repo's naive CPU
    simulator (open boundaries, center cell at index `half_width`)."""
    row = np.zeros(2 * half_width + 1, dtype=np.uint8)
    row[half_width] = 1
    return simulate_naive_center_columns(
        row.reshape(1, -1), n_steps, center_cell=half_width
    )[0]


class TestCountBlocksLiteralHistograms(unittest.TestCase):
    """count_blocks on literal bit arrays, with hand-writable histograms.

    Pattern value convention (from the module docstring):
    bits[i]*2^0 + bits[i+1]*2^1 + ... + bits[i+k-1]*2^(k-1).
    For bits = [1, 0, 1, 1, 0]:
      k=1 windows 1,0,1,1,0 -> {0:2, 1:3}
      k=2 windows [1,0]=1, [0,1]=2, [1,1]=3, [1,0]=1 -> {0:0, 1:2, 2:1, 3:1}
      k=3 windows [1,0,1]=5, [0,1,1]=6, [1,1,0]=3 -> {3:1, 5:1, 6:1}
    """

    def setUp(self):
        self.bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)

    def test_k1_histogram(self):
        counts = bf.count_blocks(self.bits, 1)
        self.assertEqual(counts.shape, (2,))
        np.testing.assert_array_equal(counts, np.array([2, 3]))

    def test_k2_histogram(self):
        counts = bf.count_blocks(self.bits, 2)
        self.assertEqual(counts.shape, (4,))
        np.testing.assert_array_equal(counts, np.array([0, 2, 1, 1]))

    def test_k3_histogram(self):
        counts = bf.count_blocks(self.bits, 3)
        self.assertEqual(counts.shape, (8,))
        np.testing.assert_array_equal(counts, np.array([0, 0, 0, 1, 0, 1, 1, 0]))


class TestCountBlocksProperties(unittest.TestCase):
    def test_all_zeros_land_in_pattern_zero(self):
        counts = bf.count_blocks(np.zeros(8, dtype=np.uint8), 3)
        np.testing.assert_array_equal(
            counts, np.array([6, 0, 0, 0, 0, 0, 0, 0])
        )

    def test_all_ones_land_in_pattern_2km1(self):
        counts = bf.count_blocks(np.ones(7, dtype=np.uint8), 3)
        expected = np.zeros(8, dtype=np.int64)
        expected[7] = 5
        np.testing.assert_array_equal(counts, expected)

    def test_k_equal_to_length(self):
        # A single window: entire array as one pattern value.
        counts = bf.count_blocks(np.array([1, 0, 1], dtype=np.uint8), 3)
        expected = np.zeros(8, dtype=np.int64)
        expected[5] = 1  # 1*1 + 0*2 + 1*4 = 5
        np.testing.assert_array_equal(counts, expected)

    def test_dtype_is_int64_and_shape_is_2_pow_k(self):
        rng = np.random.default_rng(20260908)
        bits = rng.integers(0, 2, size=97, dtype=np.uint8)
        for k in (1, 5, 20):
            counts = bf.count_blocks(bits, k)
            self.assertEqual(counts.dtype, np.int64)
            self.assertEqual(counts.shape, (1 << k,))

    def test_sum_equals_number_of_windows(self):
        rng = np.random.default_rng(20260908)
        for length, k in ((5, 1), (5, 2), (10, 3), (97, 20), (2_000_004, 4)):
            bits = rng.integers(0, 2, size=length, dtype=np.uint8)
            counts = bf.count_blocks(bits, k)
            self.assertEqual(int(counts.sum()), length - k + 1)


class TestCountBlocksAgainstReference(unittest.TestCase):
    def test_matches_independent_reference_across_chunk_boundary(self):
        # 2_000_004 bits, k=4 -> 2_000_001 windows, which splits the internal
        # 2_000_000-window chunk into two. Deterministic seed, not data files.
        rng = np.random.default_rng(20260908)
        bits = rng.integers(0, 2, size=2_000_004, dtype=np.uint8)
        np.testing.assert_array_equal(
            bf.count_blocks(bits, 4), reference_count_blocks(bits, 4)
        )

    def test_k20_all_ones_across_chunk_boundary(self):
        # 2_000_021 bits, k=20 -> 2_000_002 windows: every window is all ones
        # and must land in bin 2^20 - 1. Crosses the chunk boundary too.
        bits = np.ones(2_000_021, dtype=np.uint8)
        counts = bf.count_blocks(bits, 20)
        self.assertEqual(int(counts.sum()), 2_000_002)
        np.testing.assert_array_equal(
            counts, reference_count_blocks(bits, 20)
        )

    def test_canonical_center_column_matches_reference(self):
        # Single-black-cell initial condition (the prize-relevant seed), 8
        # steps -> 9 center-column bits, generated by the module's own CPU
        # simulator; verify count_blocks agrees with the independent reference.
        bits = canonical_center_column(n_steps=8, half_width=8)
        self.assertEqual(len(bits), 9)
        for k in (1, 2, 3):
            np.testing.assert_array_equal(
                bf.count_blocks(bits, k), reference_count_blocks(bits, k)
            )


class TestChiSquaredTest(unittest.TestCase):
    """chi_squared_test needs scipy, which is deliberately absent from
    requirements-ci.txt (CPU CI has no GPU and experiment modules are not CI
    stages). Skip the p-value checks there; the harness environment runs them.
    """

    @unittest.skipUnless(HAS_SCIPY, "chi_squared_test requires scipy")
    def test_exact_fit_gives_zero_chi2(self):
        # 10 bits, 5 zeros / 5 ones, expected 5 per bin: every bin matches the
        # expectation exactly, so chi-squared must be exactly 0, p exactly 1.
        counts = np.array([5, 5], dtype=np.int64)
        chi2, p, df = bf.chi_squared_test(counts, 5.0)
        self.assertEqual(chi2, 0.0)
        self.assertEqual(p, 1.0)
        self.assertEqual(df, 1)

    @unittest.skipUnless(HAS_SCIPY, "chi_squared_test requires scipy")
    def test_exact_fit_many_bins(self):
        counts = np.ones(8, dtype=np.int64)
        chi2, p, df = bf.chi_squared_test(counts, 1.0)
        self.assertEqual(chi2, 0.0)
        self.assertEqual(p, 1.0)
        self.assertEqual(df, 7)

    @unittest.skipUnless(HAS_SCIPY, "chi_squared_test requires scipy")
    def test_one_bin_off_by_four(self):
        # 8 bins, expected 1 each, one bin holds 5: chi2 = 4^2/1 = 16, df = 7.
        counts = np.array([1, 1, 1, 1, 1, 1, 1, 5], dtype=np.int64)
        chi2, p, df = bf.chi_squared_test(counts, 1.0)
        self.assertAlmostEqual(chi2, 16.0, places=9)
        self.assertEqual(df, 7)
        self.assertAlmostEqual(p, 0.02511636074685275, places=9)

    @unittest.skipUnless(HAS_SCIPY, "chi_squared_test requires scipy")
    def test_fractional_expected_per_bin(self):
        # k=2 on 3 bits: 3 windows over 4 patterns, expected = 3/4 per bin.
        counts = bf.count_blocks(np.array([1, 1, 0], dtype=np.uint8), 2)
        np.testing.assert_array_equal(counts, np.array([0, 1, 0, 1]))
        chi2, p, df = bf.chi_squared_test(counts, 3 / 4)
        self.assertAlmostEqual(chi2, 5 / 3, places=9)
        self.assertEqual(df, 3)
        self.assertAlmostEqual(p, 0.6443698056370253, places=9)

    @unittest.skipUnless(HAS_SCIPY, "chi_squared_test requires scipy")
    def test_uneven_counts(self):
        counts = np.array([6, 4], dtype=np.int64)
        chi2, p, df = bf.chi_squared_test(counts, 5.0)
        self.assertAlmostEqual(chi2, 0.4, places=9)
        self.assertEqual(df, 1)
        self.assertAlmostEqual(p, 0.5270892568655381, places=9)


if __name__ == "__main__":
    unittest.main()