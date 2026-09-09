#!/usr/bin/env python3
"""Unit tests for the pure helpers of experiments/invariant_measure.py.

Covers pack_row, unpack_rows, block_entropy and kmer_counts -- the four
functions in Experiment P that need no artifact, no GPU and no data file.
The canonical bitstreams (data/center_col_*.bin) are gitignored and absent
in a fresh checkout, so nothing here opens one. Bit order is pinned to
LSB-first (bitorder='little'), the convention gpu/rule30_sim.py writes and
tools/lint_bitorder.py mandates; the literal-value tests below would break
under a bare bitorder="big" (or a missing bitorder=) call.
"""

import math
import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.invariant_measure import (  # noqa: E402
    block_entropy,
    kmer_counts,
    naive_periodic,
    pack_row,
    simulate_rows,
    unpack_rows,
)

# OEIS A051023 prefix -- middle column of Rule 30 from a single 1 cell.
# Used here purely as a fixed literal bit pattern for the packing test.
OEIS_PREFIX = (1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0)


def _bits(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.uint8)


def _brute_kmer_counts(rows: np.ndarray, k: int) -> np.ndarray:
    """Reference k-mer histogram by explicit window enumeration."""
    counts = np.zeros(1 << k, dtype=np.int64)
    for row in rows:
        n = len(row) - k + 1
        for j in range(n):
            code = 0
            for bit in row[j:j + k]:
                code = (code << 1) | int(bit)
            counts[code] += 1
    return counts


class PackRowTest(unittest.TestCase):
    def test_literal_row_packs_to_the_lsb_first_uint64(self):
        # OEIS A051023 prefix, zero-padded to a full 64-bit word.
        bits = np.zeros(64, dtype=np.uint8)
        bits[:len(OEIS_PREFIX)] = OEIS_PREFIX
        packed = pack_row(bits)
        self.assertEqual(packed.shape, (1,))
        self.assertEqual(packed.dtype, np.uint64)
        # LSB-first: bit i of the row is bit (i % 8) of byte (i // 8), and
        # the uint64 view of the 8 bytes is little-endian. Bits 0-7 are
        # 00110111 = 0x3B; bits 8-15 are 11000100 read LSB-first =>
        # byte 0x23. The whole word is 0x233B. A bitorder="big" decode or
        # a missing bitorder= would reverse every byte and fail both checks.
        self.assertEqual(int(packed[0]), 0x233B)
        self.assertEqual(packed.view(np.uint8).tolist(),
                         [0x3B, 0x23, 0, 0, 0, 0, 0, 0])

    def test_round_trip_single_word(self):
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, size=64, dtype=np.uint8)
        back = unpack_rows(pack_row(bits).reshape(1, -1), 64)[0]
        np.testing.assert_array_equal(back, bits)

    def test_round_trip_two_words(self):
        rng = np.random.default_rng(13)
        bits = rng.integers(0, 2, size=128, dtype=np.uint8)
        back = unpack_rows(pack_row(bits).reshape(1, -1), 128)[0]
        np.testing.assert_array_equal(back, bits)

    def test_pack_row_returns_one_word_per_64_cells(self):
        self.assertEqual(pack_row(np.zeros(64, dtype=np.uint8)).shape, (1,))
        self.assertEqual(pack_row(np.zeros(128, dtype=np.uint8)).shape, (2,))
        self.assertEqual(pack_row(np.zeros(192, dtype=np.uint8)).shape, (3,))


class UnpackRowsTest(unittest.TestCase):
    """unpack_rows is where the not-a-multiple-of-8 boundary bites."""

    def _pad_to_words(self, bits: np.ndarray) -> np.ndarray:
        pad = (-len(bits)) % 64
        return np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    def test_round_trip_when_n_cells_is_a_multiple_of_8(self):
        for n in (8, 32, 64, 128):
            rng = np.random.default_rng(n)
            bits = rng.integers(0, 2, size=n, dtype=np.uint8)
            packed = pack_row(self._pad_to_words(bits)).reshape(1, -1)
            back = unpack_rows(packed, n)[0]
            self.assertEqual(back.shape, (n,))
            np.testing.assert_array_equal(back, bits)

    def test_round_trip_when_n_cells_is_not_a_multiple_of_8(self):
        # 63 stays inside word 0 but cuts mid-byte; 65 and 71 cross into
        # word 1. All three are where a shift-by-8 packing bug hides.
        for n in (1, 7, 63, 65, 71):
            rng = np.random.default_rng(n)
            full = rng.integers(0, 2, size=128, dtype=np.uint8)
            back = unpack_rows(pack_row(full).reshape(1, -1), n)[0]
            self.assertEqual(back.shape, (n,))
            np.testing.assert_array_equal(back, full[:n])

    def test_truncation_excludes_padding_bits(self):
        # 63 real cells padded to a full word: the padding bits must not
        # leak into the returned row.
        rng = np.random.default_rng(23)
        bits = rng.integers(0, 2, size=63, dtype=np.uint8)
        back = unpack_rows(pack_row(bits).reshape(1, -1), 63)[0]
        self.assertEqual(back.shape, (63,))
        np.testing.assert_array_equal(back, bits)

    def test_unpacks_multiple_rows(self):
        rng = np.random.default_rng(4)
        rows = rng.integers(0, 2, size=(3, 64), dtype=np.uint8)
        packed = np.stack([pack_row(r) for r in rows])
        back = unpack_rows(packed, 64)
        self.assertEqual(back.shape, (3, 64))
        np.testing.assert_array_equal(back, rows)


class BlockEntropyTest(unittest.TestCase):
    def test_uniform_counts_give_log2_of_support(self):
        for k in (2, 4, 8, 12, 16, 32):
            counts = np.full(k, 100, dtype=np.int64)
            h, _ = block_entropy(counts)
            self.assertAlmostEqual(h, math.log2(k), places=12,
                                   msg=f"block_entropy of {k} uniform bins")

    def test_uniform_bins_exact_values(self):
        h, _ = block_entropy(np.array([3, 3], dtype=np.int64))
        self.assertEqual(h, 1.0)
        h, _ = block_entropy(np.full(8, 100, dtype=np.int64))
        self.assertEqual(h, 3.0)

    def test_nonuniform_value_derived_from_running(self):
        counts = np.array([1, 3, 1], dtype=np.int64)
        h, _ = block_entropy(counts)
        self.assertAlmostEqual(h, 1.3709505944546687, places=12)

    def test_miller_madow_correction_matches_closed_form(self):
        counts = np.array([3, 3], dtype=np.int64)  # m=2, total=6
        h, mm = block_entropy(counts)
        expected = h + (2 - 1) / (2.0 * 6 * math.log(2.0))
        self.assertAlmostEqual(mm, expected, places=12)
        self.assertAlmostEqual(mm, 1.120224586740747, places=10)
        self.assertGreaterEqual(mm, h)

    def test_empty_counts_return_zero(self):
        h, mm = block_entropy(np.zeros(8, dtype=np.int64))
        self.assertEqual((h, mm), (0.0, 0.0))

    def test_single_support_has_zero_entropy(self):
        h, _ = block_entropy(np.array([7], dtype=np.int64))
        self.assertEqual(h, 0.0)
        h, _ = block_entropy(np.array([0, 4, 0], dtype=np.int64))
        self.assertEqual(h, 0.0)


class KmerCountsTest(unittest.TestCase):
    def test_literal_row_exact_histogram(self):
        row = _bits(0, 1, 0, 1, 1).reshape(1, -1)
        np.testing.assert_array_equal(kmer_counts(row, 1), [2, 3])
        np.testing.assert_array_equal(kmer_counts(row, 2), [0, 2, 1, 1])
        np.testing.assert_array_equal(kmer_counts(row, 3),
                                      [0, 0, 1, 1, 0, 1, 0, 0])

    def test_all_zero_row_concentrates_in_code_zero(self):
        row = np.zeros(8, dtype=np.uint8).reshape(1, -1)
        counts = kmer_counts(row, 3)
        self.assertEqual(counts.shape, (8,))
        self.assertEqual(int(counts[0]), 6)
        self.assertEqual(int(counts.sum()), 6)

    def test_all_ones_row(self):
        counts = kmer_counts(np.ones(4, dtype=np.uint8).reshape(1, -1), 1)
        np.testing.assert_array_equal(counts, [0, 4])

    def test_k_larger_than_row_returns_zero_counts(self):
        counts = kmer_counts(_bits(0, 1).reshape(1, -1), 5)
        self.assertEqual(counts.shape, (32,))
        self.assertEqual(int(counts.sum()), 0)

    def test_matches_brute_force_over_row_widths_not_multiple_of_8(self):
        rng = np.random.default_rng(5)
        rows = rng.integers(0, 2, size=(3, 45), dtype=np.uint8)
        for k in (1, 2, 3, 5):
            np.testing.assert_array_equal(kmer_counts(rows, k),
                                          _brute_kmer_counts(rows, k),
                                          err_msg=f"k={k}")
            self.assertEqual(int(kmer_counts(rows, k).sum()),
                             len(rows) * (45 - k + 1),
                             msg=f"window count at k={k}")


class SimulateRowsCpuTest(unittest.TestCase):
    """pack_row -> CPU step loop -> unpack_rows against the module's own
    naive per-cell reference. No GPU, no data file."""

    def test_cpu_path_matches_naive_periodic_reference(self):
        # 128 cells = two 64-bit words, so the periodic wrap at each word
        # boundary is actually exercised. A single-word test cannot see a
        # missed np.roll wrap (np.roll of a 1-element array is a no-op).
        init = _bits(
            0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0,
            0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0,
            1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0,
            1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1,
            1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
            0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
            0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
            1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
        )
        sampled = simulate_rows(init, 24, sample_every=2, gpu=False)
        reference = naive_periodic(init, 24)[::2]
        self.assertEqual(sampled.shape, (12, 128))
        np.testing.assert_array_equal(sampled, reference)
        np.testing.assert_array_equal(sampled[0], init)


if __name__ == "__main__":
    unittest.main()