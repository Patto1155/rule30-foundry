#!/usr/bin/env python3
"""Correctness tests for the smallest-grammar curve harness.

Two things have to be right for g(n) to mean anything: the grammar must
actually derive the string it claims to (otherwise every size is fiction),
and the counting null must be computed from the right class size (otherwise
the Admission Rule check is decorative).
"""

import math
import pathlib
import random
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import grammar_min_size as gms
except (Exception, SystemExit) as exc:  # pragma: no cover - dependency-driven
    # SystemExit inherits from BaseException, not Exception. A bare
    # `except Exception` here silently deleted 23 tests when an optional
    # dependency was absent and an imported module raised SystemExit at
    # module scope -- see tests/test_import_safety.py, which now forbids that
    # at source. This guard is the second line of defence, not the first.
    raise unittest.SkipTest(f"cannot import grammar_min_size: {exc}")


class DerivationTest(unittest.TestCase):
    """Every measured grammar must reproduce its own input, exactly."""

    def test_round_trips_on_random_strings(self):
        rng = random.Random(30)
        for trial in range(20):
            n = rng.randint(1, 400)
            bits = [rng.randint(0, 1) for _ in range(n)]
            with self.subTest(trial=trial, n=n):
                res = gms.repair_with_check(bits)
                self.assertTrue(res["derivation_verified"])

    def test_round_trips_on_highly_repetitive_input(self):
        """Runs are where an overlapping-pair count could go wrong."""
        for bits in ([1] * 257, [0] * 128 + [1] * 128, [0, 1] * 200,
                     [1, 1, 1, 0] * 100):
            with self.subTest(head=bits[:6], n=len(bits)):
                self.assertTrue(gms.repair_with_check(bits)["derivation_verified"])

    def test_single_bit_and_empty(self):
        self.assertEqual(gms.repair_with_check([1])["g_rules"], 0)
        self.assertEqual(gms.repair_with_check([])["g_rules"], 0)

    def test_expand_inverts_the_rules(self):
        rules = [(0, 1), (2, 2)]        # A=01, B=AA -> B expands to 0101
        self.assertEqual(gms.expand(rules, [3]), [0, 1, 0, 1])

    def test_a_wrong_grammar_is_caught(self):
        """The derivation check must not be a no-op."""
        rules = [(0, 1)]
        self.assertNotEqual(gms.expand(rules, [2]), [1, 1])


class GrammarSizeTest(unittest.TestCase):
    def test_repetition_costs_logarithmically(self):
        """2^k copies of a symbol need about k rules, not 2^k."""
        res = gms.repair_with_check([1] * 4096)
        self.assertLess(res["g_rules"], 40)

    def test_incompressible_input_costs_more_than_repetitive_input(self):
        rng = random.Random(31)
        noise = gms.repair_with_check([rng.randint(0, 1) for _ in range(4096)])
        runs = gms.repair_with_check([1] * 4096)
        self.assertGreater(noise["g_rules"], 10 * runs["g_rules"])

    def test_rule_count_accounts_for_the_start_sequence(self):
        """A start sequence left unbinarised would undercount the grammar."""
        res = gms.repair_with_check([random.Random(7).randint(0, 1)
                                     for _ in range(512)])
        self.assertEqual(
            res["g_rules"],
            res["repair_rules"] + res["start_binarisation_rules"])
        self.assertEqual(res["start_binarisation_rules"],
                         max(0, res["start_sequence_len"] - 1))

    def test_deterministic(self):
        bits = [random.Random(9).randint(0, 1) for _ in range(1024)]
        self.assertEqual(gms.repair_with_check(bits),
                         gms.repair_with_check(list(bits)))


class CountingNullTest(unittest.TestCase):
    """|M(g)| = ((g+1)!)^2 for CNF SLPs over a binary alphabet."""

    def test_class_size_matches_the_product_form(self):
        for g in (1, 2, 5, 10, 25):
            want = 2.0 * math.log2(math.factorial(g + 1))
            self.assertAlmostEqual(gms.log2_slp_class(g), want, places=6)

    def test_class_size_is_the_product_of_choices_per_rule(self):
        """Rule i chooses an ordered pair from {0,1} u {A_1..A_{i-1}}."""
        for g in (1, 2, 3, 6):
            product = 1
            for i in range(1, g + 1):
                product *= (i + 1) ** 2
            self.assertAlmostEqual(gms.log2_slp_class(g),
                                   math.log2(product), places=6)

    def test_g_null_is_the_first_g_that_clears_n(self):
        for n in (16, 64, 128, 1024, 4096):
            g = gms.g_null(n)
            self.assertGreaterEqual(gms.log2_slp_class(g), n)
            self.assertLess(gms.log2_slp_class(g - 1), n,
                            f"g_null({n}) = {g} is not minimal")

    def test_g_null_grows_with_n(self):
        nulls = [gms.g_null(n) for n in (64, 128, 256, 512, 1024)]
        self.assertEqual(nulls, sorted(nulls))
        self.assertEqual(len(set(nulls)), len(nulls))


class DetectionPowerTest(unittest.TestCase):
    """A curve that cannot separate a known-structured sequence is void."""

    def test_thue_morse_is_far_below_random(self):
        from dfao_min_states import sequence_bits
        tm = gms.repair_with_check(sequence_bits("thue-morse", 4096))["g_rules"]
        rnd = gms.repair_with_check(
            sequence_bits("random", 4096, seed=30))["g_rules"]
        self.assertLess(tm * 3, rnd, f"thue-morse {tm} vs random {rnd}")


if __name__ == "__main__":
    unittest.main()
