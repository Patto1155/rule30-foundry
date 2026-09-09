#!/usr/bin/env python3
"""Unit tests for experiments/period16_walk.walk (pure function, no data files).

`walk` is a CPU-only big-integer Rule 30 simulation with no data-file input and
no randomness: the single-black-cell seed is hardcoded. These tests call it with
small `diagonals` / `steps` so they run in milliseconds and pin the exact values
the current implementation produces, plus the two structural contracts that
matter (the recorded-tail width is bounded by `keep`, and the result is
deterministic under identical arguments).
"""

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from period16_walk import walk  # noqa: E402


# A config where every diagonal settles (steps >> 1.34 * diagonals) and the
# recorded tail excludes the early transient: all 30 diagonals are settled, so
# the verdict and word/collision lists are meaningful, not all-empty.
D = 30
STEPS = 600
KEEP = 512


class WalkResultStructureTest(unittest.TestCase):
    def test_returns_a_dict_with_the_expected_schema(self):
        res = walk(D, STEPS, KEEP)
        self.assertIsInstance(res, dict)
        for key in ("params", "timing", "settled", "zero_words", "collisions",
                    "doubling_events", "verdict", "ok"):
            self.assertIn(key, res)

    def test_dtype_of_scalar_fields(self):
        res = walk(D, STEPS, KEEP)
        self.assertIsInstance(res["params"]["diagonals"], int)
        self.assertIsInstance(res["params"]["steps"], int)
        self.assertIsInstance(res["params"]["keep_rows"], int)
        self.assertIsInstance(res["params"]["base_row"], int)
        self.assertIsInstance(res["params"]["period"], int)
        self.assertIsInstance(res["settled"]["count"], int)
        self.assertIsInstance(res["ok"], bool)
        self.assertIsInstance(res["zero_words"], list)
        self.assertTrue(all(isinstance(z, int) for z in res["zero_words"]))
        self.assertIsInstance(res["collisions"], list)
        self.assertIsInstance(res["verdict"], str)

    def test_every_diagonal_settles_in_this_config(self):
        res = walk(D, STEPS, KEEP)
        self.assertEqual(res["settled"]["count"], D)
        self.assertEqual(res["settled"]["fraction"], 1.0)
        self.assertEqual(res["settled"]["first_unsettled_d"], D)


class WalkValuesTest(unittest.TestCase):
    """Real values derived by running the current implementation (pinned)."""

    def test_pinned_values_for_small_config(self):
        res = walk(D, STEPS, KEEP)
        self.assertEqual(res["params"], {
            "diagonals": D, "steps": STEPS, "keep_rows": 521,
            "base_row": 80, "period": 16,
        })
        self.assertEqual(res["zero_words"], [2, 7, 28])
        self.assertEqual(res["n_zero_words"], 3)
        # no collisions at this small scale: parity fields are empty, not a lie
        self.assertEqual(res["n_collisions"], 0)
        self.assertEqual(res["odd_parity_collisions"], [])
        self.assertEqual(res["n_doubling_events"], 3)
        self.assertEqual(res["lemma_b_failures"], [])
        self.assertEqual(res["lemma_a_violations"], [])
        self.assertTrue(res["ok"])
        self.assertEqual(
            res["verdict"], "period-16 holds for every settled diagonal tested")

    def test_period_histogram_counts_are_exact(self):
        res = walk(D, STEPS, KEEP)
        hist = res["settled"]["period_histogram"]
        self.assertEqual(sum(hist.values()), D)
        # every settled diagonal is periodic at some tested candidate
        self.assertEqual(hist[0], 0)

    def test_censoring_note_is_explicit(self):
        """Right-censoring must be stated, never silently dropped."""
        res = walk(D, STEPS, KEEP)
        self.assertIn("right-censored", res["settled"]["censoring_note"])


class KeepBoundsTest(unittest.TestCase):
    def test_keep_bounds_the_recorded_tail_width(self):
        """keep_rows is `keep`, rounded up to the next multiple of PERIOD.

        base_row = steps - keep + 1 is floored to a multiple of 16, so
        keep <= keep_rows <= keep + (PERIOD - 1).
        """
        cases = [
            (D, STEPS, KEEP),        # keep=512 -> keep_rows=521
            (D, STEPS, 256),         # keep=256 -> keep_rows=269
            (5, 50, 32),             # keep=32  -> keep_rows=35
            (20, 300, 64),           # keep=64  -> keep_rows=77
        ]
        for diagonals, steps, keep in cases:
            with self.subTest(diagonals=diagonals, steps=steps, keep=keep):
                res = walk(diagonals, steps, keep)
                kr = res["params"]["keep_rows"]
                self.assertGreaterEqual(kr, keep)
                self.assertLessEqual(kr, keep + 16 - 1)

    def test_smaller_keep_records_no_more_rows(self):
        res_big = walk(D, STEPS, 512)
        res_small = walk(D, STEPS, 256)
        self.assertGreaterEqual(
            res_big["params"]["keep_rows"], res_small["params"]["keep_rows"])


class DeterminismTest(unittest.TestCase):
    def test_identical_arguments_give_identical_results(self):
        """Single seed, no randomness: two calls must be byte-for-byte equal.

        The `timing` key is wall-clock and legitimately differs between runs, so
        it is excluded; every result key that is a function of the simulation
        must match exactly.
        """
        a = walk(D, STEPS, KEEP)
        b = walk(D, STEPS, KEEP)
        self.assertEqual(
            {k: v for k, v in a.items() if k != "timing"},
            {k: v for k, v in b.items() if k != "timing"})

    def test_smaller_keep_does_not_change_the_underlying_verdict(self):
        """The tail width only changes which rows are examined, not the CA."""
        for keep in (256, 512):
            with self.subTest(keep=keep):
                res = walk(D, STEPS, keep)
                self.assertEqual(res["zero_words"], [2, 7, 28])


if __name__ == "__main__":
    unittest.main()
