"""A characterisation is worthless if the instrument cannot see structure.

The result here is that the seed's row language is FULL at every tested window
width, so no subshift-of-finite-type invariant can exclude anything. That is
only a statement about Rule 30 if the same measurement reports a restricted
language for a rule that has one -- rule 90 is the positive control, and its
count must come back far below 2^k or the negative means nothing.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sis = _load("seed_invariant_search")


class TestRuleArithmetic(unittest.TestCase):
    def test_rule30_image_on_every_triple(self):
        """a XOR (b OR c), with bit 0 the LEFT cell."""
        for u in range(8):
            a, b, c = u & 1, (u >> 1) & 1, (u >> 2) & 1
            with self.subTest(u=format(u, "03b")):
                self.assertEqual(sis.rule30_image(u, 3), a ^ (b | c))

    def test_seed_rows_are_the_rule30_cone(self):
        """Bit 0 is the LEFTMOST cone cell, so the int reads right-to-left.
        Asserting the left-to-right string keeps that from being mistaken for
        a bug the next time someone reads a row as a number."""
        rows = sis.seed_rows(3)
        as_text = ["".join(str((r >> i) & 1) for i in range(2 * t + 1))
                   for t, r in enumerate(rows[:3])]
        self.assertEqual(as_text, ["1", "111", "11001"])

    def test_windows_of_a_short_row_is_empty(self):
        self.assertEqual(sis.windows_of(0b1, 1, 4), set())


class TestObligations(unittest.TestCase):
    def test_a_full_window_set_is_preserved_and_vacuous(self):
        for k in (2, 3, 4):
            with self.subTest(k=k):
                self.assertTrue(sis.preserved(set(range(1 << k)), k)["ok"])

    def test_a_restricted_set_can_fail_preservation(self):
        """Preservation must be capable of failing, or it tests nothing.

        W = {00, 10, 01} at k=2. The window 1001 has all three of its
        2-sub-windows in W, and its image is 11, which is not -- so the
        subshift is not forward-closed. (W = {00} alone is a poor probe: the
        all-zero row really is preserved by Rule 30.)
        """
        r = sis.preserved({0b00, 0b01, 0b10}, 2)
        self.assertFalse(r["ok"])
        self.assertIn({"window": 0b1001, "image": 0b11}, r["violations"])

    def test_the_all_zero_subshift_is_genuinely_preserved(self):
        """Preservation is not a rubber stamp in the other direction either."""
        self.assertTrue(sis.preserved({0}, 2)["ok"])

    def test_exclusion_needs_a_window_the_seed_never_shows(self):
        loops = [{"width": 9, "rows": [0b101010101]}]
        self.assertTrue(sis.excludes(set(), 3, loops)["excludes_any"])
        self.assertFalse(
            sis.excludes(set(range(8)), 3, loops)["excludes_any"])

    def test_windows_touching_the_free_boundary_are_not_counted(self):
        """Boundary cells are supplied freely, so they say nothing about the
        rule; only the strip interior may carry an exclusion."""
        loops = [{"width": 5, "rows": [0b10001]}]
        r = sis.excludes(set(range(8)), 3, loops)
        self.assertEqual(r["widths_tested"], [5])
        self.assertFalse(r["excludes_any"])


class TestControls(unittest.TestCase):
    def test_rule90_language_is_detectably_restricted(self):
        """The positive control. Without this, "rule 30 is full" is not a
        finding about rule 30 -- it is an untested instrument."""
        self.assertLess(sis.control_rule(90, 160, 10), 1 << 10)
        self.assertLess(sis.control_rule(90, 160, 10), 200)

    def test_rule30_language_is_full_at_small_widths(self):
        for k in (4, 6, 8):
            with self.subTest(k=k):
                self.assertEqual(len(sis.seed_window_set(300, k)), 1 << k)


class TestResult(unittest.TestCase):
    def test_no_invariant_is_found_and_the_verdict_says_vacuous(self):
        art = sis.run(steps=400, kmax=8, controls=(90,))
        self.assertFalse(art["any_invariant_found"])
        for row in art["rows"]:
            self.assertTrue(row["full_language"])
            self.assertIn("VACUOUS", row["verdict"])

    def test_the_loop_rows_are_actually_loaded(self):
        art = sis.run(steps=200, kmax=4, controls=())
        self.assertEqual(art["loop_widths"], [3, 5, 7, 9, 11, 13, 15])


if __name__ == "__main__":
    unittest.main()
