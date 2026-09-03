#!/usr/bin/env python3
"""Tests for Experiment C2, the algebraic annihilator search.

Two things have to be true for the headline negative to mean anything, and
neither is checked by the run itself succeeding:

1. The GF(2) linear algebra is correct. A rank routine that overcounts rank
   reports "no annihilator" on everything, which is the failure mode that looks
   most like success.
2. The counting-bound gates fire. The repo has already lost a certificate to a
   negative that any sequence would have produced; this experiment's model
   class can do that in both directions.
"""

import importlib.util
import pathlib
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


aa = _load("algebraic_annihilator", "experiments/algebraic_annihilator.py")
cb = _load("counting_bound", "experiments/counting_bound.py")


class TestGF2Rank(unittest.TestCase):
    """The rank routine against cases whose answer is known independently."""

    def _rank(self, dense):
        dense = np.asarray(dense, dtype=np.uint8)
        n_rows, n_cols = dense.shape
        rows = np.zeros((n_rows, (n_cols + 63) // 64), dtype=np.uint64)
        for c in range(n_cols):
            rows[:, c >> 6] |= dense[:, c].astype(np.uint64) << np.uint64(c & 63)
        return len(aa.gf2_rref(rows, n_cols)), rows

    def test_identity_is_full_rank(self):
        rank, _ = self._rank(np.eye(10, dtype=np.uint8))
        self.assertEqual(rank, 10)

    def test_zero_matrix_has_rank_zero(self):
        rank, _ = self._rank(np.zeros((6, 6), dtype=np.uint8))
        self.assertEqual(rank, 0)

    def test_duplicate_rows_do_not_add_rank(self):
        m = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        rank, _ = self._rank(m)
        self.assertEqual(rank, 2)

    def test_xor_dependency_is_detected(self):
        """Third row is the XOR of the first two, so rank is 2, not 3."""
        m = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
        rank, _ = self._rank(m)
        self.assertEqual(rank, 2)

    def test_rank_spans_multiple_words(self):
        """Columns beyond 64 exercise the word indexing, where an off-by-one
        would silently drop the high monomials."""
        n = 200
        rank, _ = self._rank(np.eye(n, dtype=np.uint8))
        self.assertEqual(rank, n)

    def test_kernel_vectors_actually_annihilate(self):
        """Every returned kernel vector must satisfy M c = 0 on the original.

        Rank alone does not prove the kernel extraction is right, and the
        experiment reports a kernel vector as a shortcut candidate.
        """
        rng = np.random.default_rng(7)
        n_cols = 40
        dense = rng.integers(0, 2, (25, n_cols), dtype=np.uint8)

        packed = np.zeros((dense.shape[0], (n_cols + 63) // 64), dtype=np.uint64)
        for c in range(n_cols):
            packed[:, c >> 6] |= dense[:, c].astype(np.uint64) << np.uint64(c & 63)

        pivots = aa.gf2_rref(packed, n_cols)
        kernel = aa.kernel_basis(packed, pivots, n_cols)

        self.assertEqual(len(kernel), n_cols - len(pivots))
        self.assertGreater(len(kernel), 0, "25 rows in 40 columns must leave a kernel")
        for vec in kernel:
            product = (dense.astype(np.uint32) @ vec.astype(np.uint32)) % 2
            self.assertTrue(np.all(product == 0),
                            "kernel vector does not annihilate the matrix")


class TestMonomials(unittest.TestCase):

    def test_count_matches_the_counting_bound(self):
        for w, d in ((8, 2), (12, 3), (20, 2), (24, 3)):
            with self.subTest(w=w, d=d):
                self.assertEqual(len(aa.monomials(w, d)),
                                 cb.annihilator_dimension(w, d))

    def test_constant_term_is_first(self):
        self.assertEqual(aa.monomials(5, 2)[0], ())

    def test_evaluate_matches_a_hand_computed_polynomial(self):
        """f = 1 + x0 + x1*x2 on every 3-bit window, checked exhaustively."""
        monos = aa.monomials(3, 2)
        coeffs = np.zeros(len(monos), dtype=np.uint8)
        coeffs[monos.index(())] = 1
        coeffs[monos.index((0,))] = 1
        coeffs[monos.index((1, 2))] = 1
        codes = np.arange(8, dtype=np.uint64)
        got = aa.evaluate(codes, monos, coeffs)
        for v in range(8):
            x0, x1, x2 = v & 1, (v >> 1) & 1, (v >> 2) & 1
            self.assertEqual(int(got[v]), (1 ^ x0 ^ (x1 & x2)))


class TestReedMullerGate(unittest.TestCase):
    """The forced-negative gate, which is the one with a proof behind it."""

    def test_bound_matches_reed_muller_minimum_distance(self):
        for w, d in ((20, 2), (24, 3), (32, 2)):
            with self.subTest(w=w, d=d):
                self.assertEqual(cb.max_zeros_of_degree(w, d),
                                 2**w - 2**(w - d))

    def test_full_coverage_is_always_vacuous(self):
        """If every window occurs, only the zero polynomial vanishes on them."""
        for w, d in ((12, 2), (18, 3)):
            v = cb.annihilator_verdict(w, d, 2**w)
            with self.subTest(w=w, d=d):
                self.assertFalse(v["informative"])
                self.assertTrue(v["covers_window_space"])

    def test_measured_rule30_regime(self):
        """The parameters this experiment actually ran, and their verdicts.

        Distinct-window counts measured on data/golden/center_col_golden_10M.bin.
        w <= 22 exceeds the Reed-Muller ceiling, so those negatives are forced
        and must be refused however tempting the clean 'no annihilator' looks.
        """
        observed = {20: 1048499, 22: 3807262, 24: 7531795,
                    26: 9290394, 28: 9815389, 32: 9987988}
        for w, n in observed.items():
            informative = cb.annihilator_verdict(w, 2, n)["informative"]
            with self.subTest(w=w):
                self.assertEqual(informative, w >= 24)

    def test_too_few_rows_is_vacuous_in_the_other_direction(self):
        """Fewer distinct windows than monomials guarantees a kernel."""
        v = cb.annihilator_verdict(32, 3, 100)
        self.assertFalse(v["informative"])
        self.assertIn("dimension alone", v["reading"])

    def test_informative_window_exists_and_is_reported(self):
        v = cb.annihilator_verdict(24, 2, 7531795)
        self.assertTrue(v["informative"])
        self.assertGreater(v["surplus_rows"], 0)
        self.assertGreater(v["headroom_below_ceiling"], 0)


class TestControls(unittest.TestCase):
    """A control that cannot fail is not a control."""

    def test_positive_control_relation_holds_by_construction(self):
        bits = aa.positive_control_bits(50_000)
        k = aa.CONTROL_ORDER
        n = bits.size - k
        expect = np.zeros(n, dtype=np.uint8)
        for t in aa.CONTROL_TAPS:
            expect ^= bits[t:t + n]
        expect ^= bits[aa.CONTROL_QUAD[0]:aa.CONTROL_QUAD[0] + n] & \
            bits[aa.CONTROL_QUAD[1]:aa.CONTROL_QUAD[1] + n]
        self.assertTrue(np.array_equal(expect, bits[k:k + n]),
                        "the planted relation does not hold in the control")

    def test_search_finds_the_planted_relation(self):
        """The headline negative is worthless if this fails."""
        bits = aa.positive_control_bits(120_000)
        rec = aa.search(bits, aa.CONTROL_WINDOW, 2, margin_bits=64)
        self.assertEqual(rec["status"], "annihilator_found")
        self.assertTrue(any(a["holds_everywhere"] for a in rec["annihilators"]))

    def test_random_stream_gives_no_annihilator(self):
        bits = aa.random_control_bits(1_500_000)
        rec = aa.search(bits, 24, 2, margin_bits=64)
        self.assertEqual(rec["status"], "no_annihilator")

    def test_vacuous_parameters_are_skipped_not_reported(self):
        bits = aa.random_control_bits(400_000)
        rec = aa.search(bits, 12, 2, margin_bits=64)
        self.assertEqual(rec["status"], "skipped_vacuous")
        self.assertNotIn("rank", rec)


class TestWindowCodes(unittest.TestCase):

    def test_codes_are_little_endian_within_the_window(self):
        bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        codes = aa.window_codes(bits, 3)
        # window 0 = (1,0,1) -> bit0=1, bit1=0, bit2=1 -> 0b101 = 5
        self.assertEqual(int(codes[0]), 5)
        self.assertEqual(codes.size, 3)

    def test_oversized_window_is_rejected(self):
        with self.assertRaises(ValueError):
            aa.window_codes(np.zeros(100, dtype=np.uint8), aa.MAX_WINDOW + 1)


if __name__ == "__main__":
    unittest.main()
