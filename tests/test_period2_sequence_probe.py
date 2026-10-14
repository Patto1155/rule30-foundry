"""Controls for exact recurrence rejection, not checks of Rule 30 claims."""
import unittest
from experiments.period2_sequence_probe import recurrence, solve, nullspace, interpolate_plateaus
from experiments.period_horizon_sat import target_word

class SequenceProbeTests(unittest.TestCase):
    def test_singular_and_inconsistent_systems(self):
        self.assertEqual(solve([[1,2],[2,4]],[3,6]),[3,0])
        with self.assertRaises(ValueError): solve([[1,2],[2,4]],[3,7])
        self.assertEqual(nullspace([[1,2],[2,4]]),[[-2,1]])

    def test_positive_recurrence_control_and_heldout_kill(self):
        fib=[1,1]
        for _ in range(58):fib.append(fib[-1]+fib[-2])
        record=next(r for r in recurrence(fib,8) if (r['order'],r['degree'])==(2,0))
        self.assertEqual(len(record['survived']),1)
        fib[-1]+=1
        record=next(r for r in recurrence(fib,8) if (r['order'],r['degree'])==(2,0))
        self.assertEqual(len(record['killed']),1)
        self.assertEqual(record['killed'][0]['first_heldout_violation'],65)

    def test_monotonic_interpolation_keeps_sources(self):
        def row(n,r):return {'length':n,'radius':r,'source':'test.json','query_id':n}
        rows={('alternating',64):row(64,54),('alternating',66):row(66,54)}
        interpolate_plateaus(rows)
        self.assertEqual(rows['alternating',65]['radius'],54)
        self.assertEqual([r['length'] for r in rows['alternating',65]['inference_sources']],[64,66])
        rows={('alternating',64):row(64,54),('alternating',66):row(66,55)}
        interpolate_plateaus(rows)
        self.assertNotIn(('alternating',65),rows)

    def test_phase_conventions(self):
        self.assertEqual(target_word('alternating',4),[0,1,0,1])
        self.assertEqual(target_word('alternating-one',4),[1,0,1,0])
        with self.assertRaises(ValueError):target_word('bad',4)
