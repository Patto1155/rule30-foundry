#!/usr/bin/env python3
"""Unit tests for experiments/eca_sim.py — the rule-parameterized ECA simulator.

Covers the three pure, data-free entry points:

  step_naive_rule        cell-by-cell ground truth (open boundary)
  step_packed_cpu_rule   bit-packed CPU step of the same local rule
  simulate_spacetime_rule_cpu  packed spacetime field, row 0 = initial row

The two imperative facts this file pins are the two that cost this repo months:

  * bit order. `pack_rows` / `unpack_rows` in rule30_open_utils are LSB-first
    (`bitorder="little"`). A test that builds or decodes a packed row with the
    numpy default (MSB-first) disagrees with the naive reference on ~49.95% of
    positions while leaving every aggregate (mean, monobit, block frequency)
    unchanged, so the agreement tests below compare BIT FOR BIT against the
    naive reference, never an aggregate.
  * single-seed correctness. Rule 30's center column from the one
    single-black-cell initial condition is pinned (OEIS A051023 prefix), and
    rules 0 and 255 are pinned as both per-step and whole-field controls.

Widths are chosen as multiples of 64 (no padding cells -> packed and naive tapes
have identical open boundaries) in the agreement tests, with one deliberate
non-multiple-of-64 case to prove padding cells do not leak back into the real
boundary. No canonical bitstream (data/center_col_*.bin) is opened here — those
files are gitignored and absent on a fresh clone; every expected value below was
derived by running the module itself.

The GPU path `simulate_spacetime_rule(gpu=True)` needs a CUDA device and is the
one non-pure function in the module, so it is not exercised here. On a machine
with cupy and a GPU, `eca_sim.verify()` is the canonical CPU/GPU cross-check.
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

from eca_sim import (  # noqa: E402
    simulate_spacetime_rule_cpu,
    step_naive_rule,
    step_packed_cpu_rule,
)
from rule30_open_utils import pack_rows, unpack_rows  # noqa: E402

# OEIS A051023: center column of Rule 30 from a single black cell, steps 0..13.
RULE30_CENTER_PREFIX = (1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1)


def single_spike(n_cells: int, center: int) -> np.ndarray:
    """The one deterministic initial condition all three prizes concern."""
    row = np.zeros(n_cells, dtype=np.uint8)
    row[center] = 1
    return row


def naive_spacetime(row0: np.ndarray, n_steps: int, rule: int) -> np.ndarray:
    rows = []
    cur = np.asarray(row0, dtype=np.uint8).copy()
    for _ in range(n_steps):
        rows.append(cur)
        cur = step_naive_rule(cur, rule)
    return np.array(rows, dtype=np.uint8)


class StepNaiveRuleTest(unittest.TestCase):
    """Wolfram rule-table indexing: out_bit = (rule >> (4l + 2c + r)) & 1."""

    def test_rule30_hand_derived_rows(self):
        # Neighborhood 011 = index 4l+2c+r = 0+2*1+1 = 3 -> bit 3 of 30 is 1.
        np.testing.assert_array_equal(
            step_naive_rule(np.array([0, 1, 0, 0, 0], dtype=np.uint8), 30),
            np.array([1, 1, 1, 0, 0], dtype=np.uint8),
        )
        # Neighborhood 100 = index 4 -> bit 4 of 30 is 1.
        np.testing.assert_array_equal(
            step_naive_rule(np.array([1, 1, 0, 0, 0], dtype=np.uint8), 30),
            np.array([1, 0, 1, 0, 0], dtype=np.uint8),
        )

    def test_rule8_isolates_the_011_neighborhood(self):
        # 8 = 0b00001000: out=1 iff (l,c,r) == (0,1,1), i.e. index 3.
        np.testing.assert_array_equal(
            step_naive_rule(np.array([1, 0, 1, 1, 0], dtype=np.uint8), 8),
            np.array([0, 0, 1, 0, 0], dtype=np.uint8),
        )

    def test_boundary_cells_see_zeros_outside_the_row(self):
        # Row [1,0,1,1]: the four neighborhoods are (0,1,0) idx 2->1,
        # (1,0,1) idx 5->0, (0,1,1) idx 3->1, (1,1,0) idx 6->0.
        np.testing.assert_array_equal(
            step_naive_rule(np.array([1, 0, 1, 1], dtype=np.uint8), 30),
            np.array([1, 0, 1, 0], dtype=np.uint8),
        )

    def test_does_not_mutate_its_input(self):
        row = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        before = row.copy()
        step_naive_rule(row, 30)
        np.testing.assert_array_equal(row, before)


class StepPackedCpuRuleTest(unittest.TestCase):
    """Packed stepper vs naive, bit for bit, on words-length tapes and across
    64-bit word boundaries. Rule 30 and the trivial controls are here; the
    multi-rule sweep in the verification entry point (eca_sim.verify) covers
    the other rules the module claims."""

    def _assert_packed_matches_naive(self, row0, rule, n_steps):
        n_cells = len(row0)
        cur_p = pack_rows(row0)[0]
        cur_n = np.asarray(row0, dtype=np.uint8).copy()
        for step in range(n_steps):
            got = unpack_rows(cur_p, n_cells)[0]
            np.testing.assert_array_equal(
                got, cur_n,
                err_msg=f"packed != naive on step {step} of rule {rule}",
            )
            cur_p = step_packed_cpu_rule(cur_p, rule)
            cur_n = step_naive_rule(cur_n, rule)

    def test_rule30_single_spike_across_word_boundaries(self):
        # 192 cells = 3 words; the spike is in the middle word, so the wave
        # must cross both internal word boundaries before the tape is done.
        self._assert_packed_matches_naive(single_spike(192, 96), 30, 120)

    def test_rule30_single_spike_padded_width_does_not_leak(self):
        # 65 cells -> 64 real bits + one padding bit in the last word. The
        # packed stepper must not let the padding cell evolve and leak back
        # into the open boundary.
        self._assert_packed_matches_naive(single_spike(65, 32), 30, 40)

    def test_rules_zero_and_255(self):
        # In a rule-0/rule-255 world every packed-bit pattern is a fixed point,
        # but the step must still agree cell-for-cell with the naive reference.
        self._assert_packed_matches_naive(single_spike(192, 96), 0, 8)
        self._assert_packed_matches_naive(single_spike(192, 96), 255, 8)

    def test_does_not_mutate_its_input(self):
        cur_p = pack_rows(single_spike(192, 96))[0]
        before = cur_p.copy()
        step_packed_cpu_rule(cur_p, 30)
        np.testing.assert_array_equal(cur_p, before)


class SimulateSpacetimeRuleCpuTest(unittest.TestCase):
    """The packed spacetime path returns rows 0..n_steps-1 with row0 = input."""

    def test_rule30_matches_naive_spacetime_bit_for_bit(self):
        row = single_spike(192, 96)
        got = simulate_spacetime_rule_cpu(row, 120, 30)
        ref = naive_spacetime(row, 120, 30)
        np.testing.assert_array_equal(got, ref)
        self.assertEqual(got.shape, (120, 192))

    def test_rule30_center_column_prefix_is_pinned(self):
        # OEIS A051023, steps 0..13 — the single-seed quantity the prizes ask
        # about. Any packing/bit-order slip changes these bits.
        row = single_spike(192, 96)
        field = simulate_spacetime_rule_cpu(row, 14, 30)
        prefix = tuple(int(field[s, 96]) for s in range(14))
        self.assertEqual(prefix, RULE30_CENTER_PREFIX)

    def test_rule0_row0_is_the_seed_then_trivially_zero(self):
        row = single_spike(64, 32)
        field = simulate_spacetime_rule_cpu(row, 8, 0)
        np.testing.assert_array_equal(field[0], row)
        np.testing.assert_array_equal(field[1:], np.zeros((7, 64), dtype=np.uint8))

    def test_rule255_row0_is_the_seed_then_trivially_one(self):
        row = single_spike(64, 32)
        field = simulate_spacetime_rule_cpu(row, 8, 255)
        np.testing.assert_array_equal(field[0], row)
        np.testing.assert_array_equal(field[1:], np.ones((7, 64), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()