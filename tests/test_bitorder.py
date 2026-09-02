#!/usr/bin/env python3
"""Pin the LSB/MSB bit-packing convention that cost this repo five months.

`gpu/rule30_sim.py` writes center-column dumps with `bitorder='little'`
(LSB-first). NumPy's `unpackbits` defaults to MSB-first. Decoding one with the
other's convention returns the true stream with every consecutive 8-bit block
reversed - 49.95% of positions differ - while the bit *mean* is unchanged.
Every aggregate check the repo had was therefore blind to it, and README
experiments I-L were computed on the reversed stream.

These tests encode that trap as executable facts. The ones that need a
canonical bitstream skip cleanly when it is absent, so a fresh clone still
passes; the algebraic and golden-reference ones always run.
"""

import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# OEIS A051023: middle column of Rule 30 from a single 1 cell.
OEIS_A051023_PREFIX = (1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0)

GOLDEN = REPO_ROOT / "data" / "golden" / "center_col_golden_1M.bin"
BITSTREAMS = (
    REPO_ROOT / "data" / "center_col_10M.bin",
    REPO_ROOT / "data" / "center_col_46M.bin",
)
# Enough to be decisive without reading gigabytes.
SAMPLE_BYTES = 125_000


def _reverse_each_byte(bits: np.ndarray) -> np.ndarray:
    """Reverse every consecutive 8-bit block of a flat bit array."""
    assert bits.size % 8 == 0
    return bits.reshape(-1, 8)[:, ::-1].reshape(-1)


class RoundTripTest(unittest.TestCase):
    """A convention only round-trips against itself."""

    def setUp(self):
        rng = np.random.default_rng(30)
        self.bits = rng.integers(0, 2, size=8 * 4096, dtype=np.uint8)

    def test_little_round_trips(self):
        packed = np.packbits(self.bits, bitorder="little")
        back = np.unpackbits(packed, bitorder="little")
        np.testing.assert_array_equal(back, self.bits)

    def test_big_round_trips(self):
        packed = np.packbits(self.bits, bitorder="big")
        back = np.unpackbits(packed, bitorder="big")
        np.testing.assert_array_equal(back, self.bits)

    def test_mixing_conventions_reverses_each_byte(self):
        """The exact defect: decode LSB-packed bytes as MSB-first."""
        packed = np.packbits(self.bits, bitorder="little")
        wrong = np.unpackbits(packed, bitorder="big")
        np.testing.assert_array_equal(wrong, _reverse_each_byte(self.bits))

    def test_the_reversal_preserves_the_bit_mean(self):
        """Why no aggregate check caught it: the mean is invariant."""
        packed = np.packbits(self.bits, bitorder="little")
        right = np.unpackbits(packed, bitorder="little")
        wrong = np.unpackbits(packed, bitorder="big")
        self.assertEqual(right.sum(), wrong.sum())
        self.assertEqual(right.mean(), wrong.mean())
        # ...and yet the streams disagree on ~half of all positions.
        differing = int(np.count_nonzero(right != wrong))
        self.assertGreater(differing / right.size, 0.4)


class GoldenReferenceConventionTest(unittest.TestCase):
    """`tools/gen_golden_reference.py` is MSB-first *by design*.

    Its independence from `gpu/` is the whole reason it can catch a kernel
    bug, so this asserts the exception rather than trying to remove it.
    """

    def setUp(self):
        if not GOLDEN.exists():
            raise unittest.SkipTest(f"golden reference absent: {GOLDEN}")
        self.raw = np.fromfile(GOLDEN, dtype=np.uint8, count=64)

    def test_msb_decode_matches_oeis(self):
        bits = np.unpackbits(self.raw, bitorder="big")[:len(OEIS_A051023_PREFIX)]
        self.assertEqual(tuple(int(b) for b in bits), OEIS_A051023_PREFIX)

    def test_lsb_decode_does_not_match_oeis(self):
        bits = np.unpackbits(self.raw, bitorder="little")[:len(OEIS_A051023_PREFIX)]
        self.assertNotEqual(tuple(int(b) for b in bits), OEIS_A051023_PREFIX)


class CanonicalBitstreamTest(unittest.TestCase):
    """The kernel's dumps are LSB-first. Assert it on the real artifacts."""

    def _sample(self, path):
        if not path.exists():
            raise unittest.SkipTest(
                f"{path.relative_to(REPO_ROOT).as_posix()} absent "
                "(not tracked in git; regenerate with gpu/rule30_sim.py)")
        return np.fromfile(path, dtype=np.uint8, count=SAMPLE_BYTES)

    def test_lsb_decode_matches_oeis(self):
        for path in BITSTREAMS:
            with self.subTest(path=path.name):
                raw = self._sample(path)
                bits = np.unpackbits(raw, bitorder="little")
                self.assertEqual(
                    tuple(int(b) for b in bits[:len(OEIS_A051023_PREFIX)]),
                    OEIS_A051023_PREFIX,
                    "center_col dumps are LSB-first; a mismatch here means "
                    "either the file or the convention changed")

    def test_msb_decode_is_exactly_the_per_byte_reversal(self):
        for path in BITSTREAMS:
            with self.subTest(path=path.name):
                raw = self._sample(path)
                lsb = np.unpackbits(raw, bitorder="little")
                msb = np.unpackbits(raw, bitorder="big")
                np.testing.assert_array_equal(msb, _reverse_each_byte(lsb))
                self.assertEqual(lsb.sum(), msb.sum())

    def test_the_two_decodes_disagree_on_about_half_the_positions(self):
        for path in BITSTREAMS:
            with self.subTest(path=path.name):
                raw = self._sample(path)
                lsb = np.unpackbits(raw, bitorder="little")
                msb = np.unpackbits(raw, bitorder="big")
                frac = np.count_nonzero(lsb != msb) / lsb.size
                self.assertGreater(frac, 0.45)
                self.assertLess(frac, 0.55)


class BitstreamAgreesWithGoldenTest(unittest.TestCase):
    """LSB-decoded kernel output equals the MSB-decoded golden reference."""

    def test_first_sample_bits_agree(self):
        if not GOLDEN.exists():
            raise unittest.SkipTest(f"golden reference absent: {GOLDEN}")
        present = [p for p in BITSTREAMS if p.exists()]
        if not present:
            raise unittest.SkipTest("no canonical bitstream present")
        gold = np.unpackbits(
            np.fromfile(GOLDEN, dtype=np.uint8, count=SAMPLE_BYTES),
            bitorder="big")
        for path in present:
            with self.subTest(path=path.name):
                bits = np.unpackbits(
                    np.fromfile(path, dtype=np.uint8, count=SAMPLE_BYTES),
                    bitorder="little")
                np.testing.assert_array_equal(bits, gold)


if __name__ == "__main__":
    unittest.main()
