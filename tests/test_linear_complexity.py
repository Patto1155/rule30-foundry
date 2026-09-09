#!/usr/bin/env python3
"""Unit tests for experiments/linear_complexity.py -- bm_profile_gf2.

Berlekamp-Massey has exact, checkable answers on short sequences, which makes
it a good target for regression tests:

    * all zeros                -> L(n) = 0 for every n
    * all ones                 -> L(n) = 1 for every n (s_n = s_{n-1})
    * a single 1 at position t -> L(n) = 0 for n <= t, then t + 1 forever
    * an order-4 LFSR          -> L(n) = 4 once n >= 5, and non-decreasing

and the profile is non-decreasing on every sequence by the definition of
linear complexity.

The single-seed center column (the actual Prize-2 object) is reachable here
only through the module's own CPU simulator: the canonical
data/center_col_*.bin files are gitignored and absent on machines that have
not regenerated them, so a test that opens one would be a test that cannot
run. One test below builds 512 bits of the single-seed center column with
gpu=False and checks the experiment's headline claim (L(n) tracks n/2, i.e.
maximal linear complexity) at three scales.

No data files, no GPU, no cupy/torch imports, no bare np.unpackbits.
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))

from linear_complexity import bm_profile_gf2  # noqa: E402
from rule30_open_utils import (  # noqa: E402
    make_single_spike_row,
    simulate_center_columns_batch,
)


def lfsr_bits(feedback_lags, seed, n):
    """First n bits of the GF(2) LFSR s_k = XOR_{i in feedback_lags} s_{k-i}.

    feedback_lags are positive lags; e.g. [3, 4] gives the recurrence
    s_k = s_{k-3} XOR s_{k-4}, whose characteristic polynomial is
    x^4 + x^3 + 1.
    """
    bits = list(seed)
    for k in range(len(seed), n):
        nxt = 0
        for lag in feedback_lags:
            nxt ^= bits[k - lag]
        bits.append(nxt)
    return np.array(bits[:n], dtype=np.uint8)


def bm_reference(bits):
    """Textbook Berlekamp-Massey over GF(2) written from the algorithm
    statement (C = connection poly, lowest index = constant term).

    Used only to cross-check the module's word-parallel big-int version on
    short random arrays; not part of the module under test.
    """
    c = [1]  # C(x); c[0] = 1
    b = [1]  # B(x)
    l = 0
    m = 1
    prof = []
    for k, s in enumerate(bits):
        d = 0
        for i, ci in enumerate(c):
            if ci and k - i >= 0:
                d ^= bits[k - i]
        if d:
            t = c[:]
            padded = [0] * m + b
            if len(padded) > len(c):
                c.extend([0] * (len(padded) - len(c)))
            for i in range(len(padded)):
                c[i] ^= padded[i]
            if 2 * l <= k:
                l = k + 1 - l
                b = t
                m = 1
            else:
                m += 1
        else:
            m += 1
        prof.append(l)
    return prof


class BmProfileExactAnswersTest(unittest.TestCase):
    """BM has known exact answers on these four canonical sequences."""

    def test_all_zeros_profile_is_identically_zero(self):
        prof = bm_profile_gf2(np.zeros(64, dtype=np.uint8))
        self.assertTrue(np.all(prof == 0))
        self.assertEqual(int(prof[-1]), 0)

    def test_all_ones_profile_is_identically_one(self):
        prof = bm_profile_gf2(np.ones(64, dtype=np.uint8))
        self.assertTrue(np.all(prof == 1))

    def test_impulse_at_position_zero(self):
        bits = np.zeros(64, dtype=np.uint8)
        bits[0] = 1
        prof = bm_profile_gf2(bits)
        # A single 1 at position 0 has linear complexity 1 from the start.
        self.assertEqual(int(prof[0]), 1)
        self.assertTrue(np.all(prof == 1))

    def test_impulse_reaches_t_plus_one(self):
        t = 5
        bits = np.zeros(64, dtype=np.uint8)
        bits[t] = 1
        prof = bm_profile_gf2(bits)
        # All-zero prefix: complexity 0. Once the 1 arrives the sequence is
        # the impulse response of a first-order recurrence: L jumps to t + 1
        # and stays there (verified by running the module: profile[4] == 0,
        # profile[5] == 6 and constant thereafter).
        self.assertTrue(np.all(prof[:t] == 0))
        self.assertEqual(int(prof[t]), t + 1)
        self.assertTrue(np.all(prof[t:] == t + 1))

    def test_lfsr_order_four_known_feedback_polynomial(self):
        # x^4 + x^3 + 1 (primitive), recurrence s_k = s_{k-3} XOR s_{k-4},
        # non-zero seed 1000. Verified by executing the module: the profile is
        # [1,1,1,1,4,4,4,4,...] and holds at 4 through n = 40.
        bits = lfsr_bits([3, 4], seed=[1, 0, 0, 0], n=40)
        prof = bm_profile_gf2(bits)
        self.assertTrue(np.all(np.diff(prof) >= 0))
        self.assertEqual(int(prof[3]), 1)
        self.assertEqual(int(prof[4]), 4)  # 5 bits processed: complexity 4
        self.assertEqual(int(prof[-1]), 4)  # still 4 at n = 40
        self.assertEqual(prof[:8].tolist(), [1, 1, 1, 1, 4, 4, 4, 4])


class BmProfileStructuralTest(unittest.TestCase):
    """Properties that any correct BM implementation must satisfy."""

    def test_profile_is_non_decreasing_on_random_bits(self):
        rng = np.random.default_rng(20260908)
        for n in (1, 2, 7, 33, 128):
            bits = rng.integers(0, 2, size=n).astype(np.uint8)
            prof = bm_profile_gf2(bits)
            self.assertTrue(np.all(np.diff(prof) >= 0), f"profile decreased at n={n}")
            self.assertTrue(np.all(prof <= n), f"L(n) > n at n={n}")

    def test_matches_independent_reference_implementation(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 5, 13, 33, 64):
            for _ in range(10):
                bits = rng.integers(0, 2, size=n).astype(np.uint8)
                self.assertEqual(
                    bm_profile_gf2(bits).tolist(),
                    bm_reference(bits.tolist()),
                    f"module disagrees with reference BM at n={n}",
                )


class SingleSeedCenterColumnTest(unittest.TestCase):
    """The module exists to score the single-seed center column; build it the
    only way a data-free test can (the module's own CPU simulator, gpu=False)
    and check the experiment's headline claim at small scale."""

    def test_profile_tracks_n_over_2(self):
        n_steps = 512
        center = n_steps + 8
        base = make_single_spike_row(2 * center + 1, center)
        col = simulate_center_columns_batch(base, n_steps, center, gpu=False)[0].astype(np.uint8)
        self.assertEqual(len(col), n_steps + 1)

        prof = bm_profile_gf2(col)
        self.assertTrue(np.all(np.diff(prof) >= 0))

        # Derived by executing this exact setup (module + CPU sim): L(128)=64,
        # L(256)=128, L(512)=255. Assert they sit within +/- 8 of n/2, which
        # catches L stuck at 0, L = n (not compressible at all), a constant
        # offset, or a gross deviation, without over-claiming exact equality
        # at small n (the experiment's L(n) = n/2 exactly claim is for 4k+).
        for n in (128, 256, 512):
            L = int(prof[n - 1])
            self.assertGreaterEqual(L, n // 2 - 8, f"L({n}) far below n/2")
            self.assertLessEqual(L, n // 2 + 8, f"L({n}) far above n/2")
        self.assertGreater(int(prof[-1]), 200)  # linear structure would show as << n/2


if __name__ == "__main__":
    unittest.main()