"""Focused finite transient and exact cyclic-recurrence tests."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'experiments/period2_transient.py'
SPEC = importlib.util.spec_from_file_location('transient', PATH)
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)


class TransientTests(unittest.TestCase):
    def test_period_four_return(self):
        columns = t.periodic_columns()
        self.assertEqual(columns[-2:], columns[:2])
        self.assertEqual([c[0] for c in columns[2:]], [1, 0, 0, 0, 0, 0, 0])

    def test_rule90_does_not_share_return(self):
        columns = t.periodic_columns(90)
        self.assertNotEqual(columns[-2:], columns[:2])

    def test_interval_convention(self):
        packed = sum(1 << (j - 1) for j in [1, 8, 15, 30])
        self.assertEqual(t.first_transient(packed, 30), {
            'last_sparse_one': 15, 'next_one': 30, 'zero_start': 16,
            'zero_end': 29, 'zeros': 14, 'censored': False})

    def test_censoring_is_not_a_bound(self):
        self.assertTrue(t.first_transient(1, 20)['censored'])

    def test_counterexample_language(self):
        word = '01' * 45 + '000101'
        self.assertTrue(t.allowed(word, 'L3'))
        self.assertFalse(t.allowed('01' * 17 + '000000', 'L2'))

    def test_phase_zero_required(self):
        with self.assertRaises(ValueError):
            t.first_transient(0, 20)


if __name__ == '__main__':
    unittest.main()
