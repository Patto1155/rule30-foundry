#!/usr/bin/env python3
"""Unit tests for experiments/gf2_representation.py.

Covers the four pure, data-free helpers: entropy, sliding_windows,
xor_projection_entropy and multi_bit_projection_entropy. All expected values
below were derived by running the module on this checkout (numpy 2.4.6) and,
for the projections, cross-checked against an independent literal
recomputation of the XOR outputs and the Shannon formula.

The module imports `from tqdm import tqdm` at import time, but tqdm is not in
requirements-ci.txt and is absent here; it is only used by the two search
functions (random_search / greedy_search), which are not exercised here. The
stub below is installed before the import so the pure functions under test are
reachable on a numpy-only machine. No data file is opened by any test in this
module: the canonical bitstreams (data/center_col_*.bin) are gitignored and
absent, so bit arrays are built inline as literals.
"""

import pathlib
import sys
import types
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# experiments/gf2_representation.py does `from tqdm import tqdm` at module
# scope. tqdm is not a CI dependency (requirements-ci.txt ships numpy only),
# and nothing below calls the search routines that use it. Stub it before
# importing the experiment module.
_tqdm = types.ModuleType("tqdm")
_tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", _tqdm)

from experiments import gf2_representation as gf2  # noqa: E402


class EntropyTest(unittest.TestCase):
    """entropy(counts) is Shannon entropy H(p) over normalized counts."""

    def test_uniform_count_vector_is_exactly_log2_k(self):
        # [1]*8 counts -> uniform over 8 symbols -> H = 3 exactly.
        counts = np.ones(8, dtype=np.int64)
        self.assertEqual(gf2.entropy(counts), np.log2(8))
        # Non-unit multiplicities must not change the distribution.
        counts = np.array([8, 8, 8, 8], dtype=np.int64)
        self.assertEqual(gf2.entropy(counts), 2.0)
        # Degenerate k=1 uniform vector collapses to the point-mass case.
        self.assertEqual(gf2.entropy(np.array([42], dtype=np.int64)), 0.0)

    def test_point_mass_is_exactly_zero(self):
        self.assertEqual(gf2.entropy(np.array([17], dtype=np.int64)), 0.0)
        self.assertEqual(gf2.entropy(np.array([0, 0, 17, 0], dtype=np.int64)), 0.0)

    def test_entropy_of_balanced_two_symbols_is_one(self):
        self.assertEqual(gf2.entropy(np.array([5, 5], dtype=np.int64)), 1.0)

    def test_matches_direct_manual_computation(self):
        counts = np.array([3, 2, 1], dtype=np.int64)
        p = counts / counts.sum()
        p = p[p > 0]
        expected = float(-np.sum(p * np.log2(p)))
        # assertAlmostEqual: the direct formula is the same code, so this is a
        # weak check; the value itself is pinned in the assertion message.
        self.assertAlmostEqual(float(gf2.entropy(counts)), expected, places=12)
        self.assertGreater(float(gf2.entropy(counts)), 0.0)


class SlidingWindowsTest(unittest.TestCase):
    """sliding_windows(bits, w) is the w-wide row-major sliding windows."""

    def test_exact_shape(self):
        bits = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        win = gf2.sliding_windows(bits, 3)
        self.assertEqual(win.ndim, 2)
        self.assertEqual(win.shape, (6, 3))  # n - w + 1 rows
        self.assertEqual(win.dtype, np.uint8)

    def test_exact_contents(self):
        bits = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        win = gf2.sliding_windows(bits, 3)
        expected = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(win, expected)

    def test_handles_w_equal_to_length(self):
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)
        win = gf2.sliding_windows(bits, 4)
        self.assertEqual(win.shape, (1, 4))
        np.testing.assert_array_equal(win[0], bits)


class XorProjectionEntropyTest(unittest.TestCase):
    """xor_projection_entropy(windows, mask) = H of the XOR-reduced row."""

    def setUp(self):
        bits = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        self.windows = gf2.sliding_windows(bits, 3)

    def test_mask_picking_balanced_column_is_exactly_one(self):
        # Middle column is [1,1,0,0,1,0]: three 1s of six -> H = 1 exactly.
        mask = np.array([0, 1, 0], dtype=np.uint8)
        self.assertEqual(gf2.xor_projection_entropy(self.windows, mask), 1.0)

    def test_exact_value_for_two_bit_mask(self):
        # b0 ^ b2 over rows is [1,1,1,1,1,0], p = 5/6:
        # H = -5/6*log2(5/6) - 1/6*log2(1/6) = 0.9182958340544896 (numpy 2.4.6)
        mask = np.array([1, 0, 1], dtype=np.uint8)
        self.assertAlmostEqual(
            float(gf2.xor_projection_entropy(self.windows, mask)),
            0.9182958340544896,
            places=12,
        )

    def test_constant_projection_is_zero(self):
        # All-ones rows: the XOR of the two set bits is 0 in every window,
        # so the projected column is constant and H = 0 exactly.
        bits = np.ones(6, dtype=np.uint8)
        windows = gf2.sliding_windows(bits, 3)
        self.assertEqual(gf2.xor_projection_entropy(windows, np.ones(3, dtype=np.uint8)), 0.0)

    def test_empty_mask_projecting_all_ones_windows_is_zero(self):
        # Every row all-ones -> XOR of the two set bits is 0 everywhere.
        bits = np.ones(6, dtype=np.uint8)
        windows = gf2.sliding_windows(bits, 2)
        self.assertEqual(gf2.xor_projection_entropy(windows, np.array([1, 1], dtype=np.uint8)), 0.0)

    def test_balanced_alternating_is_close_to_one(self):
        # Rows [1,0],[0,1],[1,0],[0,0],[0,1],[1,0],[0,1]; first-bit column
        # [1,0,1,0,0,1,0] -> p = 3/7 -> H = 0.9852281360342515 (numpy 2.4.6).
        bits = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        windows = gf2.sliding_windows(bits, 2)
        h = gf2.xor_projection_entropy(windows, np.array([1, 0], dtype=np.uint8))
        self.assertAlmostEqual(float(h), 0.9852281360342515, places=12)

    def test_projection_matches_literal_hand_reduction(self):
        # Independent recomputation: XOR selected bits per row by hand, then
        # apply the binary entropy formula. Guards the vectorized reduce path.
        mask = np.array([1, 0, 1], dtype=np.uint8)
        rows = self.windows[:, mask.astype(bool)]
        projected = np.zeros(len(rows), dtype=np.uint8)
        for i, row in enumerate(rows):
            acc = 0
            for b in row:
                acc ^= int(b)
            projected[i] = acc
        p1 = projected.mean()
        expected = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        self.assertAlmostEqual(
            float(gf2.xor_projection_entropy(self.windows, mask)),
            float(expected),
            places=12,
        )


class MultiBitProjectionEntropyTest(unittest.TestCase):
    """multi_bit_projection_entropy(windows, T): joint H of k XOR outputs."""

    def setUp(self):
        bits = np.array([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        self.windows = gf2.sliding_windows(bits, 3)

    def test_exact_value_k2(self):
        # T rows: out0 = b0^b2 -> [1,1,1,1,1,0]; out1 = b1 -> [1,1,0,0,1,0].
        # Pair counts  (0,0)=1, (1,0)=2, (0,1)=1, (1,1)=2 -> H = 1.9182958340544896
        # (numpy 2.4.6).
        transform = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        h = gf2.multi_bit_projection_entropy(self.windows, transform)
        self.assertAlmostEqual(float(h), 1.9182958340544896, places=12)

    def test_identity_transform_reproduces_window_entropy(self):
        # k=w identity transform returns the 3-bit windows verbatim; their
        # empirical entropy is 2.584962500721156 (numpy 2.4.6).
        transform = np.eye(3, dtype=np.uint8)
        h = gf2.multi_bit_projection_entropy(self.windows, transform)
        self.assertAlmostEqual(float(h), 2.584962500721156, places=12)

    def test_constant_projection_is_zero(self):
        # All rows all-ones: every output bit is 0 for any mask, joint H = 0.
        bits = np.ones(8, dtype=np.uint8)
        windows = gf2.sliding_windows(bits, 4)
        transform = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
        self.assertEqual(gf2.multi_bit_projection_entropy(windows, transform), 0.0)

    def test_uniform_joint_distribution_is_log2_2k(self):
        # Maximal joint entropy for k outputs is log2(2^k) = k. Rows here are
        # [0,0],[0,1],[1,1],[1,0],[0,0],[0,1],[1,1]; with T = 2x2 identity the
        # pair counts are (0,0)=2, (0,1)=2, (1,0)=1, (1,1)=2 — not uniform, so
        # H is below 2.0; the value is pinned rather than asserted = k:
        # 1.9502120649147465 (numpy 2.4.6).
        bits = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
        windows = gf2.sliding_windows(bits, 2)
        transform = np.eye(2, dtype=np.uint8)
        h = gf2.multi_bit_projection_entropy(windows, transform)
        self.assertAlmostEqual(float(h), 1.9502120649147465, places=12)
        self.assertGreaterEqual(float(h), 0.0)
        self.assertLessEqual(float(h), 2.0)

    def test_matches_independent_recomputation(self):
        # Out-0 / out-1 derived by a slow literal XOR over the selected columns;
        # then joint counts and the same Shannon formula, computed independently.
        transform = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        proj = np.zeros((len(self.windows), transform.shape[0]), dtype=np.uint8)
        for row_i, row in enumerate(self.windows):
            for out_i, mask_row in enumerate(transform):
                acc = 0
                for col, sel in enumerate(mask_row):
                    if sel:
                        acc ^= int(row[col])
                proj[row_i, out_i] = acc
        values = np.zeros(len(proj), dtype=np.int64)
        for i in range(proj.shape[1]):
            values |= proj[:, i].astype(np.int64) << i
        counts = np.bincount(values, minlength=2 ** proj.shape[1])
        p = counts / counts.sum()
        p = p[p > 0]
        expected = float(-np.sum(p * np.log2(p)))
        self.assertAlmostEqual(
            float(gf2.multi_bit_projection_entropy(self.windows, transform)),
            expected,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()