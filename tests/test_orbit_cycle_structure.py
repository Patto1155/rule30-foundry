#!/usr/bin/env python3
"""Unit tests for experiments/orbit_cycle_structure.py.

Covers the three pure functions:

- f_vec          : the vectorized period-word map must agree bit-exactly with
                   the trusted scalar pattern_map_step (gate_vec_matches_reference)
- step           : one step of the partial state map (u, v) -> (v, w), None on
                   the v == 0 dead end; a handful of hand-enumerable literals:
                   Lemma A (u == v -> w == 0 -> dead end next step), and the
                   real-orbit chain (1, 3) -> (3, 0xfffd) -> (0xfffd, 0) of
                   length 3.
- floyd          : tortoise-and-hare cycle detection. The module's own premise
                   is that the REAL orbit never closes a cycle (F is a partial
                   map; orbits exit the deterministic region), so exact mu/lambda
                   reports are pinned by patching the module's `step` with a
                   hand-enumerated tiny orbit whose cycle is listed in the test
                   itself. `floyd` reads every state through that same `step`
                   function (including the mu/lambda locating phases), so the
                   reported (mu, lambda) is exactly the cycle of the control
                   map. A genuine dead-end run through floyd's own recount loop
                   is also pinned on the real map from the tiny start (1, 3).

This test uses no data files (data/center_col_*.bin is gitignored/absent),
needs no GPU, and never calls np.unpackbits.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np

import orbit_cycle_structure as ocs
from diagonal_recursion import pattern_map_step

PERIOD = ocs.PERIOD  # 16
SPACE = ocs.SPACE    # 2^16: every period-bits word


def _step_patched(f, start, max_steps):
    """Run ocs.floyd with ocs.step replaced by a lookup on a dict map.

    `f` is the transition map of a tiny oracle tracked in the test; values are
    whatever the oracle uses - real (u, v) tuples for real-map tests, plain
    integers for hand-built cycle tests. ocs.floyd only ever touches states
    through ocs.step, so patching it makes the algorithm follow the oracle
    exactly without touching any production code.
    """
    real_step = ocs.step
    ocs.step = lambda s: f.get(s)
    try:
        return ocs.floyd(start, max_steps)
    finally:
        ocs.step = real_step


class FVecMatchesScalarReferenceTest(unittest.TestCase):

    def test_gate_passes_with_small_sample_count(self):
        """The vectorized map must match pattern_map_step bit-exactly.

        Drawn values: u uniform over [0, 2^16), v uniform over [1, 2^16) so
        v != 0 and pattern_map_step returns ("ok", w) for every pair.
        """
        gate = ocs.gate_vec_matches_reference(samples=200, seed=0)
        self.assertEqual(gate["mismatches"], 0)
        self.assertTrue(gate["ok"])
        self.assertEqual(gate["tested"], 200)

    def test_vectorized_map_agrees_with_scalar_on_literals(self):
        """Four pairs by hand; v != 0 in each so the map is defined."""
        pairs = [
            (0b0011, 0b0001, 0xFFFD),   # t0 forced to NOT 1 in second period
            (0x0000, 0xFFFF, 0xFFFF),   # NOT 0 -> all ones
            (0x9F60, 0x6110, 0xD5A0),   # the module's worked d=399 example
            (0x1234, 0xFFFF, 0xDB97),
        ]
        for u, v, expected in pairs:
            with self.subTest(u=hex(u), v=hex(v)):
                w, flag = pattern_map_step(u, v, PERIOD)
                self.assertEqual(flag, "ok")
                self.assertEqual(w, expected)
                fast = int(ocs.f_vec(
                    np.array([u], dtype=np.int64),
                    np.array([v], dtype=np.int64))[0])
                self.assertEqual(fast, expected)

    def test_random_literal_rounds_agree(self):
        """f_vec(pair) == pattern_map_step(pair) for one self-checking pair."""
        u, v = 0xFACE, 0x00F0
        expected = pattern_map_step(u, v, PERIOD)[0]
        self.assertEqual(int(ocs.f_vec(
            np.array([u], dtype=np.int64),
            np.array([v], dtype=np.int64))[0]), expected)


class StepTest(unittest.TestCase):

    def test_returns_none_when_v_is_zero(self):
        for u in (0, 5, 0x1234):
            with self.subTest(u=hex(u)):
                self.assertIsNone(ocs.step((u, 0)))

    def test_step_is_the_pattern_map(self):
        """step((u, v)) == (v, pattern_map_step(u, v)) for v != 0."""
        for u, v in [(0b0011, 0b0001), (0x9F60, 0x6110), (0x0000, 0xFFFF)]:
            with self.subTest(u=hex(u), v=hex(v)):
                w, flag = pattern_map_step(u, v, PERIOD)
                self.assertEqual(flag, "ok")
                self.assertEqual(ocs.step((u, v)), (v, w))

    def test_step_is_identity_fixed_point(self):
        """v != 0 and u == v -> w == 0 -> (v, 0): the map then dies."""
        self.assertEqual(ocs.step((0x0001, 0x0001)), (0x0001, 0x0000))
        self.assertEqual(ocs.step((0xFFFF, 0xFFFF)), (0xFFFF, 0x0000))

    def test_real_orbit_dies_after_three_steps(self):
        """(1, 3) -> (3, 0xFFFD) -> (0xFFFD, 0xFFFD) -> (0xFFFD, 0): traced."""
        chain = []
        s = (0x0001, 0x0003)
        while s is not None:
            chain.append(s)
            s = ocs.step(s)
        self.assertEqual(chain, [(0x0001, 0x0003),
                                 (0x0003, 0xFFFD),
                                 (0xFFFD, 0xFFFD),
                                 (0xFFFD, 0x0000)])


class FloydTest(unittest.TestCase):
    """Exact mu/lambda from hand-enumerated tiny orbits.

    These uses of ocs.floyd deliberately do NOT require any real F-value; the
    algorithm reads every state through ocs.step, so a patched step with a
    table written out in the test is a complete executable specification of
    the orbit. The state *values* are irrelevant to the algorithm - only the
    step structure is - so plain ints stand in for (u, v) pairs.
    """

    def _run_orbit(self, f, start, max_steps=50):
        result = _step_patched(f, start, max_steps)
        self.assertEqual(result["outcome"], "cycle")
        return result["mu"], result["lambda"]

    def test_cycle_with_tail(self):
        """0 -> 1 -> 2 -> 3 -> 4 -> 1: tail length 1, period 4."""
        f = {0: 1, 1: 2, 2: 3, 3: 4, 4: 1}
        self.assertEqual(self._run_orbit(f, 0), (1, 4))

    def test_cycle_with_longer_tail(self):
        """0 -> 1 -> 2 -> 3 -> 4 -> 2: tail length 2, period 3."""
        f = {0: 1, 1: 2, 2: 3, 3: 4, 4: 2}
        self.assertEqual(self._run_orbit(f, 0), (2, 3))

    def test_pure_cycle_no_tail(self):
        """0 -> 1 -> 2 -> 0: start on the cycle, tail 0, period 3."""
        f = {0: 1, 1: 2, 2: 0}
        self.assertEqual(self._run_orbit(f, 0), (0, 3))

    def test_step_cap_is_reported(self):
        """max_steps=1 cuts the search inside the while loop."""
        f = {0: 1, 1: 2, 2: 0}
        result = _step_patched(f, 0, max_steps=1)
        self.assertEqual(result["outcome"], "step_cap")
        self.assertEqual(result["steps"], 1)

    def test_dead_end_chain_on_real_map(self):
        """floyd's recount loop on the real partial map: (1, 3) dies in 3."""
        result = ocs.floyd((0x0001, 0x0003), max_steps=50)
        self.assertEqual(result["outcome"], "left_deterministic_region")
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["final_state"], [0xFFFD, 0x0000])
        self.assertEqual(result["reason"], "v_is_zero")

    def test_dead_end_chain_via_patched_map(self):
        """0 -> 1 -> 2 -> None: the recount loop distance must be exact."""
        f = {(0,): (1,), (1,): (2,), (2,): None}
        result = _step_patched(f, (0,), max_steps=50)
        self.assertEqual(result["outcome"], "left_deterministic_region")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["final_state"], [2])
        self.assertEqual(result["reason"], "v_is_zero")


if __name__ == "__main__":
    unittest.main()