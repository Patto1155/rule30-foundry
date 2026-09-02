#!/usr/bin/env python3
"""Tape sizing rules for the GPU kernels.

A tape too short for its step count does not crash and does not look wrong:
the bit mean stays 0.5, and the `--center` sanity check on the first 20 bits
still passes. It fails *late* — the same signature the repo treats as a real
kernel bug. Sizing a rented run to fit VRAM without this check produces a
plausible, wrong bitstream after hours of compute, so the arithmetic is pinned
here where it runs without a GPU.
"""

import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu"))

import tape_geometry as tg


class WordRoundingTest(unittest.TestCase):
    def test_rounds_up_to_whole_words(self):
        self.assertEqual(tg.word_count(1), 1)
        self.assertEqual(tg.word_count(64), 1)
        self.assertEqual(tg.word_count(65), 2)
        self.assertEqual(tg.rounded_cells(65), 128)

    def test_rejects_non_positive(self):
        for bad in (0, -64):
            with self.subTest(n_cells=bad):
                self.assertRaises(ValueError, tg.word_count, bad)

    def test_seed_matches_the_kernel_placement(self):
        """rule30_sim seeds bit 32 of word n_words//2."""
        for cells in (64, 1_000_000, 21_000_000):
            with self.subTest(cells=cells):
                self.assertEqual(tg.seed_cell(cells),
                                 (tg.word_count(cells) // 2) * 64 + 32)


class ConeBoundTest(unittest.TestCase):
    def test_safe_steps_is_the_nearer_edge(self):
        for cells in (1024, 1_000_000, 21_000_000, 96_000_000):
            with self.subTest(cells=cells):
                c = tg.seed_cell(cells)
                self.assertEqual(tg.max_safe_steps(cells),
                                 min(c, tg.rounded_cells(cells) - 1 - c))

    def test_safe_steps_is_about_half_the_tape(self):
        for cells in (1_000_000, 21_000_000):
            with self.subTest(cells=cells):
                ratio = tg.max_safe_steps(cells) / cells
                self.assertGreater(ratio, 0.49)
                self.assertLess(ratio, 0.51)

    def test_min_cells_is_minimal(self):
        """One word narrower must fail, or the bound is not tight.

        Skipped at the floor: a one-word tape already supports 31 steps, so
        for tiny step counts there is no narrower tape to compare against.
        """
        for steps in (1, 1000, 10_000_000, 46_000_000):
            with self.subTest(steps=steps):
                cells = tg.min_cells_for_steps(steps)
                self.assertGreaterEqual(tg.max_safe_steps(cells), steps)
                if cells > tg.WORD_BITS:
                    self.assertLess(tg.max_safe_steps(cells - 64), steps)

    def test_the_floor_is_one_word(self):
        self.assertEqual(tg.min_cells_for_steps(1), tg.WORD_BITS)
        self.assertEqual(tg.max_safe_steps(tg.WORD_BITS), 31)

    def test_round_trip(self):
        for steps in (10, 5000, 1_000_000):
            with self.subTest(steps=steps):
                self.assertGreaterEqual(
                    tg.max_safe_steps(tg.min_cells_for_steps(steps)), steps)


class DocumentedRunsTest(unittest.TestCase):
    """The commands in the handovers and README must pass the guard.

    If one of these ever fails, either a documented command is unsound or the
    bound has been made too strict.
    """

    CASES = [
        (21_000_000, 10_000_000, "10M center column"),
        (96_000_000, 46_000_000, "46M center column"),
    ]

    def test_documented_commands_are_sound(self):
        for cells, steps, label in self.CASES:
            with self.subTest(label=label):
                info = tg.check(cells, steps)
                self.assertTrue(info["cone_fits"])

    def test_they_are_not_passing_by_luck(self):
        """Each should have real margin, not squeak past by a few steps."""
        for cells, steps, label in self.CASES:
            with self.subTest(label=label):
                margin = tg.max_safe_steps(cells) - steps
                self.assertGreater(margin, 100_000, f"{label} margin {margin}")


class RejectionTest(unittest.TestCase):
    def test_rejects_a_tape_that_is_too_short(self):
        with self.assertRaises(tg.ConeTooLarge):
            tg.check(1_000_000, 10_000_000)

    def test_the_message_names_a_working_tape_size(self):
        try:
            tg.check(1_000_000, 10_000_000)
        except tg.ConeTooLarge as exc:
            msg = str(exc)
            self.assertIn("--cells", msg)
            self.assertIn("wrong LATE", msg)
            # the size it recommends must actually work
            self.assertGreaterEqual(
                tg.max_safe_steps(tg.min_cells_for_steps(10_000_000)),
                10_000_000)
        else:
            self.fail("expected ConeTooLarge")

    def test_exactly_at_the_bound_is_accepted(self):
        cells = 1_000_000
        tg.check(cells, tg.max_safe_steps(cells))

    def test_one_step_past_the_bound_is_rejected(self):
        cells = 1_000_000
        with self.assertRaises(tg.ConeTooLarge):
            tg.check(cells, tg.max_safe_steps(cells) + 1)


class ScalingFactsTest(unittest.TestCase):
    """The numbers that drive rented-compute planning.

    The 2026-08-30 plan asserted item 13 is VRAM-bound and that a 24 GB card
    "quadruples the reachable horizon". These pin the arithmetic that shows
    otherwise, so the claim cannot quietly come back.
    """

    def test_a_1e8_step_run_is_small_in_memory(self):
        info = tg.describe(tg.min_cells_for_steps(100_000_000), 100_000_000)
        total = info["tape_bytes"] + info["center_buffer_bytes"]
        self.assertLess(total, 200 * 1024 * 1024,
                        "1e8 steps should need well under 200 MB of VRAM")

    def test_even_1e9_steps_fits_a_6gb_card(self):
        info = tg.describe(tg.min_cells_for_steps(1_000_000_000),
                           1_000_000_000)
        total = info["tape_bytes"] + info["center_buffer_bytes"]
        self.assertLess(total, 6 * 1024**3)

    def test_memory_is_dominated_by_the_center_buffer(self):
        """Which means streaming it to host removes the VRAM question."""
        info = tg.describe(tg.min_cells_for_steps(100_000_000), 100_000_000)
        self.assertGreater(info["center_buffer_bytes"], info["tape_bytes"])


if __name__ == "__main__":
    unittest.main()
