"""A falsifier is only worth its verdict if it CAN fail and CAN pass.

Two failure modes this file exists to catch, both of which the first draft of
the experiment had. Feeding the same history both of its branches guarantees a
collision and refutes everything, including a summary that is in fact correct.
And a summary whose values are all distinct cannot collide for want of
opportunity, so reporting it as "survived" is a forced positive.
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


bss = _load("branch_summary_search")


class TestRecursion(unittest.TestCase):
    def test_step_column_reproduces_the_simulator(self):
        steps, ndiag = 120, 10
        width = 2 * steps + 3
        centre = steps + 1
        mask = (1 << width) - 1
        state = 1 << centre
        ref = np.zeros((steps + 1, ndiag), dtype=np.uint8)
        for t in range(steps + 1):
            win = state >> (centre - t)
            for d in range(ndiag):
                ref[t, d] = (win >> d) & 1
            state = ((state << 1) ^ (state | (state >> 1))) & mask
        cols = [ref[:, 0], ref[:, 1]]
        for d in range(2, ndiag):
            nxt = bss.step_column(cols[-2], cols[-1], int(ref[0, d]))
            with self.subTest(d=d):
                self.assertTrue(np.array_equal(nxt, ref[:, d]))
            cols.append(nxt)

    def test_a_zero_word_admits_two_distinct_successors(self):
        u = np.zeros(64, dtype=np.uint8)
        z = np.zeros(64, dtype=np.uint8)
        self.assertFalse(np.array_equal(bss.step_column(u, z, 0),
                                        bss.step_column(u, z, 1)))


class TestFalsifierIsHonest(unittest.TestCase):
    def test_a_perfect_summary_is_not_refuted(self):
        """Give the collider a summary that always determines the branch."""
        events = [{"summaries": {"s": (i % 3,)}, "branch": (i % 3) & 1}
                  for i in range(60)]
        r = bss.collide(events, "s")
        self.assertTrue(r["determines_branch"])
        self.assertTrue(r["tested"])
        self.assertIn("survived", r["verdict"])

    def test_a_useless_summary_is_refuted_with_a_counterexample(self):
        events = [{"summaries": {"s": (0,)}, "branch": i & 1} for i in range(20)]
        r = bss.collide(events, "s")
        self.assertFalse(r["determines_branch"])
        self.assertIsNotNone(r["counterexample"])
        self.assertIn("REFUTED", r["verdict"])

    def test_all_unique_keys_is_reported_untested_not_passed(self):
        """The forced-positive control. A pass here would be meaningless."""
        events = [{"summaries": {"s": (i,)}, "branch": i & 1} for i in range(50)]
        r = bss.collide(events, "s")
        self.assertTrue(r["determines_branch"])
        self.assertFalse(r["tested"])
        self.assertEqual(r["events_with_a_twin"], 0)
        self.assertIn("UNTESTED", r["verdict"])

    def test_twin_count_matches_the_events_that_share_a_key(self):
        events = ([{"summaries": {"s": (0,)}, "branch": 0} for _ in range(3)]
                  + [{"summaries": {"s": (1,)}, "branch": 0}])
        self.assertEqual(bss.collide(events, "s")["events_with_a_twin"], 3)


class TestPopulations(unittest.TestCase):
    def test_a_branch_event_records_one_branch_not_both(self):
        u = bss._settled_column(0b1011, 4, np.array([1, 0], dtype=np.uint8), 96)
        z = bss._settled_column(0, 4, np.array([1, 1, 0], dtype=np.uint8), 96)
        ev = bss.branch_event(u, z, 32)
        self.assertIsNotNone(ev)
        self.assertIn(ev["branch"], (0, 1))

    def test_the_seed_population_is_small_and_says_so(self):
        art = bss.run(n_histories=60, length=192, depth=24, tail=64,
                      steps=2048, seed=30)
        self.assertIn("low power", art["seed_orbit"]["note"])
        self.assertGreater(art["arbitrary"]["n_events"], 0)


if __name__ == "__main__":
    unittest.main()
