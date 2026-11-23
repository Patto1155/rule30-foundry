"""experiments/dfao_state_lower_bound.py: soundness of the Myhill-Nerode bound.

The bound is only worth anything if it can never exceed the true minimum, so
the reference here is a brute-force exact minimum by exhaustive enumeration of
every s-state table - sharing no code with the distinguishability count.
"""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import dfao_state_lower_bound as lb  # noqa: E402


def digits(i, direction):
    d = [int(ch) for ch in bin(i)[2:]]
    return d[::-1] if direction == "lsd" else d


def brute_min_states(seq, direction, max_states=3):
    """Exact s*(n) by enumerating every table with up to max_states states."""
    words = [digits(i, direction) for i in range(len(seq))]
    for s in range(1, max_states + 1):
        for flat in itertools.product(range(s), repeat=2 * s):
            for out in itertools.product((0, 1), repeat=s):
                ok = True
                for w, bit in zip(words, seq):
                    q = 0
                    for d in w:
                        q = flat[2 * q + d]
                    if out[q] != bit:
                        ok = False
                        break
                if ok:
                    return s
    return None


class LowerBoundTests(unittest.TestCase):
    def test_never_exceeds_brute_force_minimum(self):
        rng = np.random.default_rng(11)
        compared = 0
        for _ in range(60):
            n = int(rng.integers(6, 15))
            seq = rng.integers(0, 2, size=n, dtype=np.uint8)
            got = lb.lower_bounds(seq)
            for direction in ("msd", "lsd"):
                exact = brute_min_states(seq.tolist(), direction)
                bound = got[f"{direction}_lower_bound"]
                if exact is None:        # true minimum is >= 4
                    continue
                compared += 1
                self.assertLessEqual(bound, exact, (seq.tolist(), direction))
        self.assertGreater(compared, 30)

    def test_bound_is_not_trivially_one(self):
        # a sequence needing 3 MSD states must be seen to need more than 1
        seq = np.array(lb.naive_center(64), dtype=np.uint8)
        got = lb.lower_bounds(seq)
        self.assertEqual(got["msd_lower_bound"], 15)   # = SAT-certified s*(64)
        self.assertEqual(lb.lower_bounds(seq[:56])["msd_lower_bound"], 13)

    def test_thue_morse_is_two(self):
        tm = np.array([bin(i).count("1") & 1 for i in range(4096)],
                      dtype=np.uint8)
        got = lb.lower_bounds(tm)
        self.assertEqual((got["msd_lower_bound"], got["lsd_lower_bound"]),
                         (2, 2))

    def test_naive_center_is_oeis_a051023(self):
        self.assertEqual(lb.naive_center(15),
                         [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0])

    def test_self_test_passes(self):
        self.assertTrue(lb.self_test()["passed"])


if __name__ == "__main__":
    unittest.main()
