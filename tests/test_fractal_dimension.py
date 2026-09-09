#!/usr/bin/env python3
"""Unit tests for experiments/fractal_dimension.py (Experiment O).

Covers the four pure functions a test can exercise without a data file:

  - box_count      hand-built arrays with hand-computed occupied-box counts
  - centered_corr  exact +1 against itself, -1 against its negation
  - causal_mask    the light cone is a triangle: row sums 1, 3, 5, ...
  - diagonal_corr  causal-masked lag correlation, including the zero-valid
                   guard and the affine-invariance that centered correlations
                   promise

Everything that needs bits builds them with the module's own CPU simulator
(`simulate_spacetime(n_steps, gpu=False)`), which is pure NumPy, so the suite
runs on a machine with no GPU and no canonical bitstream. The module itself
uses `np.unpackbits(..., bitorder="little")` and `verify_kernel()` cross-checks
the packed path against a naive per-cell implementation; the test asserts the
same naive agreement directly at the verify_kernel size.
"""

import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments import fractal_dimension as fd


class BoxCountTest(unittest.TestCase):
    """box_count: rebuild a grid and count occupied non-overlapping boxes."""

    @staticmethod
    def _expected(arr, box_size):
        """Reference: mark each box anchor occupied iff any cell in it is set."""
        boxed = np.zeros_like(arr)
        for i in range(0, arr.shape[0], box_size):
            for j in range(0, arr.shape[1], box_size):
                boxed[i, j] = np.any(arr[i:i + box_size, j:j + box_size])
        return int(boxed.sum())

    def test_scattered_points_in_one_box(self):
        arr = np.zeros((1, 8), dtype=np.uint8)
        arr[0, 0] = 1
        arr[0, 3] = 1
        self.assertEqual(fd.box_count(arr, 8), 1)

    def test_spread_over_two_boxes(self):
        arr = np.zeros((2, 16), dtype=np.uint8)
        arr[0, 1] = 1
        arr[1, 9] = 1
        self.assertEqual(fd.box_count(arr, 8), 2)

    def test_rectangle_not_multiple_of_box_size(self):
        arr = np.zeros((5, 7), dtype=np.uint8)
        arr[0, 0] = 1
        arr[0, 1] = 1
        arr[4, 6] = 1
        for box_size in (1, 2, 3, 4, 5, 6, 7, 8):
            with self.subTest(box_size=box_size):
                self.assertEqual(fd.box_count(arr, box_size),
                                 self._expected(arr, box_size),
                                 f"box_size={box_size}")

    def test_box_size_larger_than_array_is_one_box(self):
        arr = np.zeros((5, 7), dtype=np.uint8)
        arr[0, 0] = 1
        self.assertEqual(fd.box_count(arr, 8), 1)
        self.assertEqual(fd.box_count(arr, 100), 1)

    def test_full_block_packs_into_every_box(self):
        arr = np.ones((5, 5), dtype=np.uint8)
        self.assertEqual(fd.box_count(arr, 1), 25)
        self.assertEqual(fd.box_count(arr, 2), 9)   # 3x3 padded, all occupied
        self.assertEqual(fd.box_count(arr, 4), 4)   # 2x2 padded, all occupied
        self.assertEqual(fd.box_count(arr, 5), 1)
        self.assertEqual(fd.box_count(arr, 8), 1)

    def test_empty_array_counts_zero(self):
        self.assertEqual(fd.box_count(np.zeros((6, 8), dtype=np.uint8), 2), 0)
        self.assertEqual(fd.box_count(np.zeros((1, 1), dtype=np.uint8), 1), 0)

    def test_random_array_matches_a_brute_force_reference(self):
        rng = np.random.default_rng(11)
        arr = rng.integers(0, 2, size=(37, 53)).astype(np.uint8)
        for box_size in (1, 2, 4, 8, 16):
            with self.subTest(box_size=box_size):
                self.assertEqual(fd.box_count(arr, box_size),
                                 self._expected(arr, box_size))

    def test_non_bool_dtypes_are_accepted(self):
        arr = np.arange(6 * 8).reshape(6, 8) % 2
        self.assertEqual(fd.box_count(arr, 2), self._expected(arr, 2))


class CenteredCorrTest(unittest.TestCase):
    """centered_corr: Pearson correlation on demeaned series.

    These are exact algebraic properties, so they are asserted with ==, not
    approximate, and the random series is chosen to have nonzero variance so
    the +1 / -1 results are real signal rather than a degenerate 0/0.
    """

    def _signal(self):
        rng = np.random.default_rng(7)
        return rng.integers(0, 2, 1000).astype(np.uint8)

    def test_signal_against_itself_is_exactly_one(self):
        sig = self._signal()
        self.assertNotEqual(float(sig.mean()), 0.0)
        self.assertNotEqual(float(sig.mean()), 1.0)
        self.assertEqual(fd.centered_corr(sig, sig), 1.0)

    def test_signal_against_its_negation_is_exactly_minus_one(self):
        sig = self._signal()
        self.assertEqual(fd.centered_corr(sig, 1 - sig), -1.0)
        self.assertEqual(fd.centered_corr(1 - sig, sig), -1.0)

    def test_negation_against_negation_is_exactly_one(self):
        sig = self._signal()
        self.assertEqual(fd.centered_corr(1 - sig, 1 - sig), 1.0)

    def test_strictly_positive_affine_scale_is_invariant(self):
        sig = self._signal()
        scaled = (2 * sig).astype(np.uint8)
        self.assertEqual(fd.centered_corr(scaled, sig), 1.0)

    def test_random_pairs_are_uncorrelated_in_expectation(self):
        """Two independent random series should sit near zero, not at +/-1."""
        rng = np.random.default_rng(123)
        a = rng.integers(0, 2, 4096).astype(np.uint8)
        b = rng.integers(0, 2, 4096).astype(np.uint8)
        c = fd.centered_corr(a, b)
        self.assertGreater(c, -0.2)
        self.assertLess(c, 0.2)

    def test_explicit_pair_matches_manual_value(self):
        x = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8)
        y = np.array([0, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        self.assertEqual(fd.centered_corr(x, y), 0.0)

    def test_constant_inputs_return_zero_not_nan(self):
        const = np.ones(50, dtype=np.uint8)
        self.assertIsInstance(fd.centered_corr(const, const), float)
        self.assertEqual(fd.centered_corr(const, const), 0.0)


class CausalMaskTest(unittest.TestCase):
    """causal_mask: |col - center| <= row, i.e. a solid triangle.

    The mask must be a contiguous triangle symmetric about the center column:
    row r contains exactly the 2r+1 columns within r of the center. Row sums
    [1, 3, 5, ..., 2n-1] assert that; the symmetry asserts the center column
    is where the initial 1 lives.
    """

    def test_row_sums_are_odd_integers(self):
        for n in (1, 5, 16, 64):
            mask = fd.causal_mask(n, 2 * n + 1)
            self.assertEqual(mask.shape, (n, 2 * n + 1))
            sums = mask.sum(axis=1).tolist()
            self.assertEqual(sums, [2 * r + 1 for r in range(n)])

    def test_total_area_is_n_squared(self):
        for n in (1, 5, 16):
            mask = fd.causal_mask(n, 2 * n + 1)
            self.assertEqual(int(mask.sum()), n * n)

    def test_center_column_is_all_true(self):
        n = 16
        mask = fd.causal_mask(n, 2 * n + 1)
        self.assertTrue(mask[:, n].all())

    def test_mask_is_symmetric_about_the_center_column(self):
        n = 16
        mask = fd.causal_mask(n, 2 * n + 1)
        np.testing.assert_array_equal(mask, mask[:, ::-1])

    def test_contiguity_within_each_row(self):
        """Each row is one contiguous run of True around the center."""
        n = 16
        mask = fd.causal_mask(n, 2 * n + 1)
        for r in range(n):
            cols = np.flatnonzero(mask[r])
            self.assertEqual(cols.tolist(),
                             list(range(n - r, n + r + 1)),
                             f"row {r}")

    def test_top_row_contains_only_the_center_cell(self):
        n = 16
        mask = fd.causal_mask(n, 2 * n + 1)
        row0 = mask[0]
        self.assertEqual(int(row0.sum()), 1)
        self.assertTrue(row0[n])  # center of the parity array

    def test_dtype_is_bool(self):
        self.assertEqual(fd.causal_mask(16, 33).dtype, np.bool_)


class DiagonalCorrTest(unittest.TestCase):
    """diagonal_corr: lag-shifted centered correlation inside the light cone."""

    def test_slope_plus_one_of_a_translation_invariant_pattern_is_one(self):
        """arr[t][x] depends on (x - t): the t+1 row is the t row shifted right."""
        n, n_cells = 10, 21
        t = np.arange(n)[:, None]
        x = np.arange(n_cells)[None, :]
        arr = ((x - t) % 3).astype(np.uint8)
        mask = fd.causal_mask(n, n_cells)
        for lag in (1, 2, 4):
            with self.subTest(lag=lag):
                self.assertAlmostEqual(
                    fd.diagonal_corr(arr, mask, lag, +1), 1.0, places=12)

    def test_slope_plus_one_differs_from_slope_minus_one(self):
        """The two diagonals must be distinguishable, not both trivially 1."""
        n, n_cells = 10, 21
        t = np.arange(n)[:, None]
        x = np.arange(n_cells)[None, :]
        arr = ((x - t) % 3).astype(np.uint8)
        mask = fd.causal_mask(n, n_cells)
        c_neg = fd.diagonal_corr(arr, mask, 1, -1)
        self.assertLess(c_neg, -0.4)
        self.assertNotAlmostEqual(c_neg, -1.0, places=8)

    def test_negative_slope_pairwise_antitone_gives_minus_one(self):
        """Anti-diagonal translation invariance: arr[t][x] depends on (t + x)."""
        n, n_cells = 10, 21
        t = np.arange(n)[:, None]
        x = np.arange(n_cells)[None, :]
        arr = ((t + x) % 3).astype(np.uint8)
        mask = fd.causal_mask(n, n_cells)
        for lag in (1, 2):
            with self.subTest(lag=lag):
                self.assertAlmostEqual(
                    fd.diagonal_corr(arr, mask, lag, -1), 1.0, places=10)

    def test_real_rule30_diagonals_are_not_identical(self):
        """A smoke test on genuinely 2D data from the module's own simulator.

        The 16-step slice is a real 2D object (rows differ; diag+ != diag-).
        This pins current values so a refactor that flattens the pattern to
        something trivially isotropic (as a 1D projection would) fails here.
        """
        bits = fd.simulate_spacetime(16, gpu=False)
        mask = fd.causal_mask(16, bits.shape[1])
        c_pos = fd.diagonal_corr(bits, mask, 3, +1)
        c_neg = fd.diagonal_corr(bits, mask, 3, -1)
        self.assertEqual(c_pos, -0.20475348513374347)
        self.assertEqual(c_neg, 0.2927849927849928)
        self.assertNotEqual(c_pos, c_neg)
        # a lag-shifted comparison must not be the identity correlation:
        # if a refactor drops the shift (correlating a block with itself),
        # every slope gives 1.0 and this catches it.
        self.assertNotAlmostEqual(c_pos, 1.0, places=8)

    def test_missing_overlap_returns_zero_not_nan(self):
        arr = np.ones((4, 4), dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        # lag 5 disjoints every compared position -> no valid cells
        for slope in (+1, -1):
            with self.subTest(slope=slope):
                self.assertIsInstance(
                    fd.diagonal_corr(arr, mask, 5, slope), float)
                self.assertEqual(fd.diagonal_corr(arr, mask, 5, slope), 0.0)


class SimulatedSpacetimeTest(unittest.TestCase):
    """The CPU simulator the tests rely on, pinned against the naive rule.

    fractal_dimension sets this up, but it runs the actual LSB-first packed
    kernel, so a bit-order regression in the packing surfaces here before it
    can corrupt the diagonal-correlation expectations above.
    """

    def test_simulated_matches_naive_at_the_verifier_size(self):
        packed = fd.simulate_spacetime(96, gpu=False)
        naive = fd.naive_spacetime(96)
        self.assertEqual(packed.shape, (96, 193))
        np.testing.assert_array_equal(packed, naive)

    def test_center_column_matches_the_oeis_prefix(self):
        """a051023: middle column of Rule 30 from a single 1 cell."""
        bits = fd.simulate_spacetime(16, gpu=False)
        center = bits[:, bits.shape[1] // 2]
        self.assertEqual(
            center[:15].tolist(), [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0])

    def test_first_row_is_a_single_one_at_the_center(self):
        bits = fd.simulate_spacetime(8, gpu=False)
        self.assertEqual(int(bits[0].sum()), 1)
        self.assertEqual(int(bits[0, bits.shape[1] // 2]), 1)

    def test_verify_kernel_passes(self):
        """verify_kernel reruns the same check and raises on mismatch."""
        fd.verify_kernel()

    def test_cpu_path_does_not_hit_cupy(self):
        """gpu=False must not require import cupy (returns sentinel when absent)."""
        self.assertFalse(
            fd.GPU and fd.cp is None,
            "GPU flag set without cupy imported is an inconsistent state")
        # the real assertion: the CPU path runs without a GPU module at all
        bits = fd.simulate_spacetime(4, gpu=False)
        self.assertEqual(bits.shape, (4, 9))


if __name__ == "__main__":
    unittest.main()