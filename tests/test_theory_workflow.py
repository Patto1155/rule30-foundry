"""Gates for the theory-first loop: tools/theory_triage.py and the F_2 instrument.

These guard the two things that make the loop worth having -- a vacuity gate
that reproduces the repo's own published verdicts, and an instrument that can
actually detect the structure it claims to exclude.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.algebraic_relation import (bits_to_poly, budget_curve, clmul,
                                            complexity_point, kernel_vector,
                                            relation_for, residual_bits)
from prize_lab import sequence_bits
from tools.theory_triage import MECHANISMS, load_queue, mechanism_gate, quantitative_gate


class QuantitativeGateTest(unittest.TestCase):
    """The gate must reproduce verdicts the repo already paid to learn."""

    def test_retracted_certificate_is_refused(self):
        # docs/CLAIM_LEDGER.md: "no 1-5 state DFAO fits the first 128 center
        # bits" was retracted as vacuous by 2^-100. The gate must catch it.
        g = quantitative_gate({"class": "dfao", "states": 5, "base": 2, "n": 128})
        self.assertEqual(g["verdict"], "FAIL")
        self.assertIn("vacuous", g["reason"])
        self.assertLess(g["log2_class_size"], 128)

    def test_informative_dfao_search_passes(self):
        g = quantitative_gate({"class": "dfao", "states": 20, "base": 2, "n": 128})
        self.assertEqual(g["verdict"], "PASS")
        self.assertGreater(g["margin_bits"], 0)

    def test_annihilator_at_narrow_window_is_refused(self):
        # theory/README section 5: the Reed-Muller bound voids w <= 22 on 10M.
        g = quantitative_gate({"class": "annihilator", "window": 22, "degree": 3,
                               "n": 10_000_000})
        self.assertEqual(g["verdict"], "FAIL")

    def test_algebraic_relation_forced_positive_is_refused(self):
        g = quantitative_gate({"class": "algebraic-relation", "degree": 99,
                               "ext_degree": 99, "n": 8192})
        self.assertEqual(g["verdict"], "FAIL")
        self.assertIn("forced positive", g["reason"])

    def test_extrapolation_evidence_escapes_the_counting_squeeze(self):
        # A bare negative in this class is uninformative at every budget, since
        # discriminating wants C >= n and not-forced wants C <= n. Extrapolation
        # is what makes the design admissible, so the gate must allow it.
        spec = {"class": "algebraic-relation", "degree": 16, "ext_degree": 16, "n": 8192}
        self.assertEqual(quantitative_gate(spec)["verdict"], "FAIL")
        self.assertEqual(quantitative_gate({**spec, "evidence": "extrapolation",
                                            "holdout": 8192})["verdict"], "PASS")

    def test_extrapolation_without_holdout_is_refused(self):
        g = quantitative_gate({"class": "algebraic-relation", "degree": 16,
                               "ext_degree": 16, "n": 8192,
                               "evidence": "extrapolation", "holdout": 0})
        self.assertEqual(g["verdict"], "FAIL")

    def test_curve_shape_needs_sizes_and_controls(self):
        base = {"class": "algebraic-relation", "degree": 16, "ext_degree": 16,
                "n": 8192, "evidence": "curve-shape"}
        self.assertEqual(quantitative_gate(
            {**base, "sizes": [1, 2], "controls": ["random", "thue-morse"]})["verdict"], "FAIL")
        self.assertEqual(quantitative_gate(
            {**base, "sizes": [1, 2, 3], "controls": []})["verdict"], "FAIL")
        self.assertEqual(quantitative_gate(
            {**base, "sizes": [1024, 2048, 4096], "controls": ["random", "thue-morse"]}
        )["verdict"], "PASS")

    def test_symbolic_obligation_needs_no_numbers(self):
        self.assertIsNone(quantitative_gate(None))


class MechanismGateTest(unittest.TestCase):
    def good(self):
        return {"id": "x", "mechanism_check": {
            "forced-outcome": "The class is larger than the prefix and the design "
                              "reports a curve against a matched null rather than "
                              "a single point."}}

    def test_unknown_mechanism_is_refused(self):
        o = {"id": "x", "mechanism_check": {"vibes": "a" * 60}}
        self.assertEqual(mechanism_gate(o)["verdict"], "FAIL")

    def test_empty_check_is_refused(self):
        self.assertEqual(mechanism_gate({"id": "x", "mechanism_check": {}})["verdict"], "FAIL")

    def test_thin_rebuttal_is_refused(self):
        o = {"id": "x", "mechanism_check": {"forced-outcome": "it is fine"}}
        self.assertEqual(mechanism_gate(o)["verdict"], "FAIL")

    def test_substantive_rebuttal_passes(self):
        self.assertEqual(mechanism_gate(self.good())["verdict"], "PASS")

    def test_prose_punctuation_does_not_break_the_gate(self):
        # A rebuttal is prose and contains semicolons, colons and commas. A gate
        # that blocks work must never refuse a well-formed obligation over them.
        o = {"id": "x", "mechanism_check": {"forced-outcome":
             "log2|M| = 289.0; n = 8192: the design reports C*(N) as a curve, "
             "against a null, and requires held-out prediction."}}
        self.assertEqual(mechanism_gate(o)["verdict"], "PASS")

    def test_shipped_queue_validates_and_every_open_item_clears_both_gates(self):
        data = load_queue(ROOT / "queue/theory/obligations.json")
        open_items = [o for o in data["obligations"] if o.get("status", "open") == "open"]
        self.assertTrue(open_items)
        for o in open_items:
            with self.subTest(obligation=o["id"]):
                self.assertEqual(mechanism_gate(o)["verdict"], "PASS")
                self.assertIn(o["prize"], ("1", "2", "3"))
                self.assertTrue(set(o["mechanism_check"]) <= set(MECHANISMS))


class F2ArithmeticTest(unittest.TestCase):
    def test_clmul_is_carry_less(self):
        # (1+x)^2 = 1 + x^2 over F_2, not 1 + 2x + x^2.
        self.assertEqual(clmul(0b11, 0b11, 8), 0b101)

    def test_kernel_finds_a_dependency(self):
        combo = kernel_vector([0b0011, 0b0101, 0b0110], 4)
        self.assertEqual(combo, 0b111)          # the three columns XOR to zero

    def test_kernel_is_empty_for_independent_columns(self):
        self.assertEqual(kernel_vector([0b0001, 0b0010, 0b0100], 4), 0)

    def test_bits_to_poly_is_little_endian_in_the_exponent(self):
        self.assertEqual(bits_to_poly([1, 0, 1]), 0b101)


class InstrumentTest(unittest.TestCase):
    """An instrument that cannot detect known structure excludes nothing."""

    def test_thue_morse_relation_is_the_textbook_one(self):
        r = budget_curve("thue-morse", 256, 6, 6, 256, 0)
        self.assertIsNotNone(r)
        # (1+x)^3 f^2 + (1+x)^2 f + x = 0  ->  D=2, E=3, C=(2+1)(3+1)=12
        self.assertEqual((r["degree"], r["ext_degree"], r["coefficients"]), (2, 3, 12))
        self.assertTrue(r["extrapolates"])

    def test_thue_morse_budget_plateaus_in_n(self):
        # Automaticity means one relation serves every length, so C* must not
        # grow. This is the signature a real shortcut in the center would show.
        small = budget_curve("thue-morse", 256, 6, 6, 256, 0)
        large = budget_curve("thue-morse", 1024, 6, 6, 1024, 0)
        self.assertEqual(small["coefficients"], large["coefficients"])

    def test_random_admits_no_extrapolating_relation(self):
        r = budget_curve("random", 256, 6, 6, 256, 30)
        self.assertFalse(bool(r and r["extrapolates"]))

    def test_center_admits_no_extrapolating_relation_at_this_budget(self):
        r = budget_curve("center", 256, 6, 6, 256, 30)
        self.assertFalse(bool(r and r["extrapolates"]))

    def test_a_found_relation_actually_annihilates_the_series(self):
        r = budget_curve("thue-morse", 256, 6, 6, 256, 0)
        terms = relation_for(sequence_bits("thue-morse", 256), r["degree"],
                             r["ext_degree"], 256)
        self.assertEqual(residual_bits(sequence_bits("thue-morse", 512), terms, 512), 0)




class ComplexityCurveTest(unittest.TestCase):
    """C*(N) is always defined, which is what makes it a curve rather than an absence."""

    def test_thue_morse_budget_is_flat_across_lengths(self):
        for n in (128, 256, 512):
            with self.subTest(n=n):
                self.assertEqual(complexity_point("thue-morse", n, 6, 0)["coefficients"], 12)

    def test_random_null_is_maximal(self):
        # A fit is forced once the column count passes N, so the null must sit
        # just under N. Anything well below would mean the search is broken.
        for n in (128, 256):
            with self.subTest(n=n):
                self.assertGreater(complexity_point("random", n, 6, 30)["ratio_to_n"], 0.9)

    def test_center_sits_with_the_null_not_with_the_positive_control(self):
        point = complexity_point("center", 256, 6, 0)
        self.assertGreater(point["ratio_to_n"], 0.9)

    def test_curve_is_defined_for_every_sequence(self):
        # The original budget search reported None for center and random, which
        # is an absence. This quantity always exists.
        for kind, seed in (("thue-morse", 0), ("random", 30), ("center", 0)):
            with self.subTest(kind=kind):
                self.assertIsNotNone(complexity_point(kind, 128, 6, seed))


class IndependentRankCrossCheckTest(unittest.TestCase):
    """Re-decide C* with numpy row reduction, sharing no code with the packed path.

    The packed implementation builds powers with a hand-written carry-less
    multiply and decides rank on row-bitmaps. This rebuilds both from numpy
    convolution and uint8 Gaussian elimination, so agreement is a real check on
    the instrument rather than a restatement of it.
    """

    @staticmethod
    def cstar_numpy(kind, n, max_degree, seed):
        import numpy as np

        bits = sequence_bits(kind, n, seed=seed)
        best = None
        for degree in range(1, max_degree + 1):
            f = np.array(bits[:n], dtype=np.int64)
            power = np.zeros(n, dtype=np.int64)
            power[0] = 1
            fp = [power]
            for _ in range(degree):
                fp.append(np.convolve(fp[-1], f)[:n] % 2)
            columns = []
            for ext in range(n // (degree + 1) + 2):
                for i in range(degree + 1):
                    column = np.zeros(n, dtype=np.int64)
                    if ext < n:
                        column[ext:] = fp[i][:n - ext]
                    columns.append(column)
                matrix = (np.array(columns).T % 2).copy()
                rows, cols = matrix.shape
                rank = 0
                for c in range(cols):
                    pivot = next((r for r in range(rank, rows) if matrix[r, c]), None)
                    if pivot is None:
                        continue
                    matrix[[rank, pivot]] = matrix[[pivot, rank]]
                    hits = np.nonzero(matrix[rank + 1:, c])[0] + rank + 1
                    matrix[hits] = (matrix[hits] + matrix[rank]) % 2
                    rank += 1
                    if rank == rows:
                        break
                if rank < cols:
                    total = (degree + 1) * (ext + 1)
                    if best is None or total < best:
                        best = total
                    break
        return best

    def test_two_implementations_agree(self):
        for kind, seed in (("thue-morse", 0), ("center", 0), ("random", 30)):
            with self.subTest(kind=kind):
                packed = complexity_point(kind, 64, 5, seed)["coefficients"]
                self.assertEqual(packed, self.cstar_numpy(kind, 64, 5, seed))


if __name__ == "__main__":
    unittest.main()
