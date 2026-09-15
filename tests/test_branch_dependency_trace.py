"""The backward trace must agree with simulation, and pruning must be sound.

The short-circuit in (R) is an exact algebraic cancellation, not a heuristic:
`D_d(t)` sits inside the OR, so `D_{d-1}(t) = 1` makes it irrelevant. If that
reasoning were wrong the traced value would silently differ from the rule, and
the cost numbers would be measuring a different function. These tests pin the
value first and the cost second.
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


bdt = _load("branch_dependency_trace")


class TestAgainstSimulation(unittest.TestCase):
    def test_every_cell_matches_the_simulator(self):
        steps, diagonals = 48, 30
        dg = bdt.reference_diagonals(steps, diagonals)
        for t in range(1, steps + 1, 5):
            for d in range(min(diagonals, 2 * t + 1)):
                with self.subTest(d=d, t=t):
                    self.assertEqual(bdt.trace(d, t)["value"], int(dg[t, d]))

    def test_short_circuit_never_changes_the_value(self):
        for t in (7, 15, 31):
            for d in (0, 1, 2, t // 2, t):
                with self.subTest(d=d, t=t):
                    self.assertEqual(bdt.trace(d, t, True)["value"],
                                     bdt.trace(d, t, False)["value"])

    def test_cone_edge_base_cases(self):
        """D_0 == 1 and D_1(t>=1) == 1 -- docs/theory/README.md §3."""
        for t in range(6):
            self.assertEqual(bdt.trace(0, t)["value"], 1)
        for t in range(1, 6):
            self.assertEqual(bdt.trace(1, t)["value"], 1)
        self.assertEqual(bdt.trace(1, 0)["value"], 0)

    def test_outside_the_cone_is_zero(self):
        self.assertEqual(bdt.trace(9, 3)["value"], 0)


class TestCost(unittest.TestCase):
    def test_pruning_never_costs_more(self):
        for t in (16, 32, 64):
            c = bdt.costs(t, t)
            with self.subTest(t=t):
                self.assertLessEqual(c["cells_evaluated"], c["cells_cone"])
                self.assertLessEqual(c["cells_cone"], c["cells_rect"])

    def test_the_cone_is_about_half_the_rectangle(self):
        """A triangle inside its bounding box: a gross error in the closure
        would show up here before it showed up in a growth exponent."""
        c = bdt.costs(128, 128)
        self.assertAlmostEqual(c["cone_over_rect"], 0.5, delta=0.02)

    def test_the_artifact_verifies_against_a_recomputation(self):
        art = bdt.run([8, 16, 32])
        self.assertTrue(art["oracle"]["ok"])
        self.assertEqual(art["oracle"]["mismatches"], [])


class TestGrowth(unittest.TestCase):
    def test_exponents_are_reported_for_both_cost_models(self):
        g = bdt.run([16, 32, 64, 128])["growth"]
        self.assertIsNotNone(g["exponent_evaluated"])
        self.assertIsNotNone(g["exponent_cone"])

    def test_two_depths_are_the_minimum_for_a_fit(self):
        self.assertIsNone(bdt.growth([{"t": 16, "cells_evaluated": 1,
                                       "cells_cone": 1}])["exponent"])


if __name__ == "__main__":
    unittest.main()
