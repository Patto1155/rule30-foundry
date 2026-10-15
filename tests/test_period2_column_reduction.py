"""Independent checks of phase-zero column inversion, not minimisation proofs."""
import random
import unittest
from experiments.period2_column_reduction import (
    column1, completion, completion_from_column1, completion_from_u, forward_check,
)
from experiments.support_growth import completion as support_completion
from experiments.period2_identity_check import forced_left


class ColumnReductionTests(unittest.TestCase):
    def test_checkpoint_witnesses(self):
        free = tuple(map(int, '01010100110101101101010010010001100100010'))
        l1 = tuple(map(int, '01' * 17 + '0' * 6))
        self.assertEqual(completion_from_u(free, 81).bit_count(), 9)
        self.assertEqual(completion_from_u(l1, 79).bit_count(), 10)
        left = completion(0x98f9d8a0, 190)
        self.assertEqual(left.bit_count(), 26)
        self.assertTrue(forward_check(0x98f9d8a0, left, 190))

    def test_three_independent_implementations(self):
        for r in range(32):
            n = 16
            left = completion(r, n)
            self.assertEqual(left, support_completion(r, n))
            target = [t % 2 for t in range(n + 1)]
            self.assertEqual(left, sum(v << j for j, v in enumerate(forced_left(r, target, n))))
            self.assertTrue(forward_check(r, left, n))

    def test_odd_right_bits_are_erased(self):
        rng = random.Random(300)
        for _ in range(25):
            n = 80
            d = column1(rng.getrandbits(128), n)
            randomized = [v if t % 2 == 0 else rng.randrange(2) for t, v in enumerate(d)]
            self.assertEqual(completion_from_column1(d, n), completion_from_column1(randomized, n))

    def test_left_odd_latch_and_right_counterexample(self):
        self.assertEqual(column1(0, 1), (0, 0))
        for r in range(16):
            d = column1(r, 20)
            for t in range(1, 20, 2):
                c, c_next = t % 2, (t + 1) % 2
                self.assertEqual(c_next ^ (c | d[t]), 1)

    def test_prefixes_and_constraint_scope(self):
        for r in range(256):
            u = column1(r, 80)[::2]
            word = ''.join(map(str, u))
            self.assertNotIn('11', word)
            self.assertNotIn('00000', word)
            self.assertEqual(completion(r, 13), completion(r, 80) & ((1 << 13) - 1))

    def test_complete_five_output_cone(self):
        actual = {''.join(map(str, column1(r, 8)[::2])) for r in range(512)}
        expected = set('00001 00010 00100 00101 01000 01001 01010 10000 10001 10010 10100 10101'.split())
        self.assertEqual(actual, expected)

    def test_empty_and_bad_inputs(self):
        self.assertEqual(completion_from_u([], 0), 0)
        self.assertEqual(completion(0, 0), 0)
        self.assertEqual(column1(0, 0), (0,))
        for args in (([], 1), ([2], 1), ([0], -1)):
            with self.assertRaises(ValueError):
                completion_from_u(*args)
        with self.assertRaises(ValueError):
            column1(-1, 2)
