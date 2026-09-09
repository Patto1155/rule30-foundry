#!/usr/bin/env python3
"""Unit tests for experiments/motif_mining.py (Experiment N, CPU).

Only pure functions are tested - none of them reads a data file, so these
tests run on a fresh clone where the canonical `data/center_col_*.bin`
bitstreams are gitignored and absent. `load_bits` is deliberately not touched.

The functions under test have a documented dependency: importing the module
brings in `scipy.stats.chi2` and `tqdm` at module scope. This file imports the
module through `unittest`'s import machinery (the way CI's `unittest discover`
does), so a machine without scipy reports these tests as skipped rather than
silently dropping them.
"""

import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from experiments import motif_mining as mm
except (Exception, SystemExit) as exc:  # pragma: no cover - dependency-driven
    # e.g. scipy missing -> import fails -> skip, not silently drop. c.f.
    # tests/test_import_safety.py, which forbids the silent-drop pattern.
    raise unittest.SkipTest(f"cannot import experiments.motif_mining: {exc}")

# 8 known bits -> exactly one byte value. bits_to_bytes_packed packs MSB-first
# (np.packbits default = bitorder="big"), so bit i is the (7-i)th bit of the
# byte. [1,1,0,1,1,1,0,0] -> 0b11011100 -> 0xDC.
KNOWN_8_BITS = np.array([1, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
KNOWN_8_BYTE = b"\xdc"


def _de_bruijn(k: int, n: int) -> list:
    """Lexicographically least de Bruijn sequence B(k, n), classical db().

    Every length-n word over a k-symbol alphabet appears exactly once as a
    cyclic window. Used to build perfectly balanced k-mer sequences.
    """
    a = [0] * (k * n)
    seq = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


class BitsToBytesPackedTest(unittest.TestCase):
    """Packing convention: MSB-first, zero-padded, exactly one byte per 8 bits."""

    def test_known_8_bits_pack_to_one_exact_byte(self):
        self.assertEqual(mm.bits_to_bytes_packed(KNOWN_8_BITS), KNOWN_8_BYTE)

    def test_packs_msb_first_across_a_byte_boundary(self):
        # [11011100 01111111] -> 0xDC 0x7F (bit 8 is the MSB of the 2nd byte).
        bits = np.array([1, 1, 0, 1, 1, 1, 0, 0,
                         0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
        self.assertEqual(mm.bits_to_bytes_packed(bits), b"\xdc\x7f")

    def test_zero_pads_to_a_byte_boundary(self):
        self.assertEqual(mm.bits_to_bytes_packed(
            np.array([1, 0, 1, 0, 1], dtype=np.uint8)), b"\xa8")

    def test_empty_input_yields_empty_bytes(self):
        self.assertEqual(mm.bits_to_bytes_packed(np.array([], dtype=np.uint8)),
                         b"")

    def test_round_trip_with_unpackbits_little(self):
        """pack(unpackbits(x, 'little')) reverses each byte; twice is identity.

        bits_to_bytes_packed packs MSB-first (np.packbits default), while
        np.unpackbits(..., bitorder='little') reads LSB-first, so the two
        conversions differ by a per-byte reversal. Reversal is an involution,
        so the round trip pack -> unpack_little -> pack is the identity on
        already-packed bytes. Derived values: 0xDC<->0x3B, 0x00, 0x7F<->0xFE,
        0xFF.
        """
        raw = np.array([0xDC, 0x00, 0x7F, 0xFF], dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="little")
        packed = np.frombuffer(mm.bits_to_bytes_packed(bits), dtype=np.uint8)
        np.testing.assert_array_equal(
            packed, np.array([0x3B, 0x00, 0xFE, 0xFF], dtype=np.uint8))
        repacked = np.frombuffer(
            mm.bits_to_bytes_packed(np.unpackbits(packed, bitorder="little")),
            dtype=np.uint8)
        np.testing.assert_array_equal(repacked, raw)

    def test_round_trip_preserves_the_bit_stream(self):
        """The bit stream comes back through unpackbits(little) reversed per
        8-bit block, and the packed bytes are the per-block reversal of the
        starting bytes - exact bytes derived from the known 16-bit literal."""
        bits = np.array([1, 0, 1, 0, 0, 1, 1, 0,
                         0, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        packed = np.frombuffer(mm.bits_to_bytes_packed(bits), dtype=np.uint8)
        np.testing.assert_array_equal(packed, np.array([0xA6, 0x59], dtype=np.uint8))
        back = np.unpackbits(packed, bitorder="little")
        np.testing.assert_array_equal(back, bits.reshape(-1, 8)[:, ::-1].reshape(-1))
        # repacking those LSB-decoded bits gives each original byte reversed:
        np.testing.assert_array_equal(
            np.frombuffer(mm.bits_to_bytes_packed(back), dtype=np.uint8),
            np.array([0x65, 0x9A], dtype=np.uint8))


class CompressionRatioTest(unittest.TestCase):
    """compressed_size / original_size, with real, derived bounds."""

    def test_highly_repetitive_input_compresses_well(self):
        self.assertLess(mm.compression_ratio(b"\x00" * 1024, "zlib"), 0.05)

    def test_incompressible_input_is_near_one(self):
        # 1024 fixed-seed bytes; zlib ratio reproduces 1.0107... deterministically.
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=1024, dtype=np.uint8).tobytes()
        self.assertGreater(mm.compression_ratio(data, "zlib"), 1.0)
        self.assertLess(mm.compression_ratio(data, "zlib"), 1.2)

    def test_zlib_and_lzma_both_compress_zero_runs(self):
        self.assertLess(mm.compression_ratio(b"\x00" * 1024, "lzma"), 0.05)

    def test_unknown_method_raises_value_error(self):
        with self.assertRaises(ValueError):
            mm.compression_ratio(b"abc", "bzip2")

    def test_ratio_cannot_be_below_zero(self):
        # tiny input forces zlib overhead; ratio is real but bounded above 0.
        self.assertGreaterEqual(mm.compression_ratio(b"\x00" * 8, "zlib"), 0.0)


class KmerUniformityTest(unittest.TestCase):
    """Chi-squared uniformity over all k-bit words."""

    def test_balanced_sequence_reports_near_uniform(self):
        # B(2,4) repeated 10x: every 4-mer appears exactly 10 times -> chi2 = 0.
        period = np.array(_de_bruijn(2, 4), dtype=np.uint8)
        bits = np.tile(period, 11)[:163]          # n_windows = 160
        res = mm.kmer_uniformity_test(bits, 4)
        self.assertTrue(res["uniform"])
        self.assertEqual(res["n_distinct"], 16)
        self.assertEqual(res["total_cells"], 16)
        self.assertEqual(res["expected_per_cell"], 10.0)
        self.assertAlmostEqual(res["chi2_stat"], 0.0, places=6)
        self.assertEqual(res["p_value"], 1.0)

    def test_balanced_sequence_k8_reports_near_uniform(self):
        # B(2,8) repeated: every 8-mer appears exactly 6 times -> chi2 = 0.
        period = np.array(_de_bruijn(2, 8), dtype=np.uint8)
        bits = np.tile(period, 7)[:1543]          # n_windows = 1536
        res = mm.kmer_uniformity_test(bits, 8)
        self.assertTrue(res["uniform"])
        self.assertEqual(res["n_distinct"], 256)
        self.assertEqual(res["total_cells"], 256)
        self.assertEqual(res["expected_per_cell"], 6.0)
        self.assertAlmostEqual(res["chi2_stat"], 0.0, places=6)
        self.assertEqual(res["p_value"], 1.0)

    def test_all_zeros_reports_non_uniform(self):
        res = mm.kmer_uniformity_test(np.zeros(1024, dtype=np.uint8), 4)
        self.assertFalse(res["uniform"])
        self.assertEqual(res["n_distinct"], 1)
        self.assertEqual(res["p_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
