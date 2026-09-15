"""The hybrid accounting is only meaningful if the bit it produces is right.

docs/theory/README.md §3 says the centre bit at time T lives on diagonal d = T
and that settle(T) ~ 1.34*T > T, so the target is never settled. That is the
claim the cost measurement rests on, so it is pinned here rather than assumed.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


whc = _load("wedge_hybrid_cost")


class TestOracle(unittest.TestCase):
    def test_the_diagonal_view_reproduces_the_centre_column(self):
        dg = whc.left_diagonals(64, 65)
        for T in range(0, 65, 7):
            with self.subTest(T=T):
                self.assertEqual(int(dg[T, T]), whc.centre_column(T))

    def test_the_first_centre_bits_match_oeis_a051023(self):
        """1101110011000101 -- the published prefix, as in diagonal_recursion."""
        bits = "".join(str(whc.centre_column(t)) for t in range(16))
        self.assertEqual(bits, "1101110011000101")

    def test_cone_edge_base_cases(self):
        dg = whc.left_diagonals(40, 41)
        self.assertTrue(dg[:, 0].all())
        self.assertTrue(dg[1:, 1].all())


class TestConeGeometry(unittest.TestCase):
    def test_the_band_widens_by_two_per_step_back(self):
        band = whc.cone_cells(10)
        self.assertEqual(list(band[10]), [10, 10])
        self.assertEqual(list(band[9]), [8, 10])
        self.assertEqual(list(band[8]), [6, 10])
        self.assertEqual(list(band[0]), [0, 10])

    def test_the_cone_fits_inside_its_rectangle(self):
        band = whc.cone_cells(32)
        self.assertLessEqual(int((band[:, 1] - band[:, 0] + 1).sum()), 33 * 33)


class TestSettling(unittest.TestCase):
    def test_a_constant_column_settles_at_zero(self):
        dg = np.ones((256, 1), dtype=np.uint8)
        self.assertEqual(int(whc.settle_times(dg, 64)[0]), 0)

    def test_a_column_with_a_transient_settles_after_it(self):
        col = np.zeros((256, 1), dtype=np.uint8)
        col[:9, 0] = 1
        self.assertEqual(int(whc.settle_times(col, 64)[0]), 9)

    def test_an_unsettled_column_is_never_free(self):
        rng = np.random.default_rng(0)
        col = rng.integers(0, 2, size=(512, 1), dtype=np.uint8)
        self.assertEqual(int(whc.settle_times(col, 128)[0]),
                         np.iinfo(np.int64).max)


class TestResult(unittest.TestCase):
    def test_the_target_cell_is_never_settled(self):
        """§3's claim, as a test rather than a citation."""
        art = whc.run([32, 64, 128], tail=64)
        self.assertFalse(art["target_ever_settled"])
        for r in art["rows"]:
            self.assertFalse(r["target_is_settled"])

    def test_the_hybrid_bit_equals_the_simulated_bit(self):
        art = whc.run([32, 64, 128], tail=64)
        self.assertTrue(art["oracle"]["ok"])
        self.assertEqual(art["oracle"]["mismatches"], [])

    def test_free_and_work_partition_the_cone(self):
        for r in whc.run([32, 64], tail=64)["rows"]:
            self.assertEqual(r["cells_free"] + r["cells_work"], r["cells_cone"])

    def test_growth_reading_names_the_direction(self):
        rows = [{"T": 2 ** k, "cells_work": 4 ** k, "cells_cone": 4 ** k,
                 "free_fraction": 0.2} for k in range(4, 8)]
        self.assertIn("constant factor", whc.growth(rows)["reading"])


if __name__ == "__main__":
    unittest.main()
