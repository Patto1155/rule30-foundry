import copy
import json
import itertools
from pathlib import Path
import unittest

from experiments.slanted_density_certificate import verify_report
from tools.verify_slanted_density import verify


class DensityObstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((Path(__file__).resolve().parents[1] /
            'runs/theory/slanted-density-obstructions.json').read_text())

    def test_all_exact_obstructions(self):
        self.assertEqual(verify_report(self.report)['verified_obstructions'], 14)
        for item in self.report['candidates']:
            self.assertGreater(verify(item), 0)

    def test_changed_multiplier_rejected(self):
        item = copy.deepcopy(self.report['candidates'][0])
        item['contributing_constraints'][0]['multiplier'] += 1
        with self.assertRaises(ValueError):
            verify(item)

    def test_illegal_transition_rejected(self):
        item = copy.deepcopy(self.report['candidates'][-1])
        row = next(r for r in item['contributing_constraints'] if r['label'][0] == 'interior')
        row['label'][4][2] ^= 1
        with self.assertRaises(ValueError):
            verify(item)

    def test_negative_multiplier_rejected(self):
        item = copy.deepcopy(self.report['candidates'][-1])
        item['contributing_constraints'][0]['multiplier'] = -1
        with self.assertRaises(ValueError):
            verify(item)

    def test_two_step_predecessor_coherence(self):
        # Independent Rule 30 truth table on seven original cells. No fresh
        # predecessor bits may be substituted on the second time step.
        def advance(row):
            return tuple((30 >> (4*row[j] + 2*row[j-1] + row[j-2])) & 1
                         for j in range(2, len(row)))
        genuine = set()
        single = set()
        for row in itertools.product((0, 1), repeat=7):
            middle = advance(row)
            genuine.add((row[-3:], middle[-3:], advance(middle)))
            single.add((row[-3:], middle[-3:]))
        forbidden = ((0, 0, 0), (1, 0, 0), (1, 1, 1))
        self.assertIn(forbidden[:2], single)
        self.assertIn(forbidden[1:], single)
        self.assertNotIn(forbidden, genuine)
        relaxed = {(a, b, d) for a, b in single for c, d in single if b == c}
        self.assertEqual(len(relaxed - genuine), 4)
