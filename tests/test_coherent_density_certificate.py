import copy
import itertools
import json
from pathlib import Path
import unittest

from experiments.coherent_density_certificate import (
    check_farkas, coherent_relation, inequality_system, verify_report,
)


class CoherentDensityCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((Path(__file__).resolve().parents[1] /
            'runs/theory/coherent-density-obstructions.json').read_text())

    def test_exact_obstructions(self):
        checked = verify_report(self.report)
        self.assertEqual(checked['verified_candidates'], 4)
        self.assertTrue(all(item['outcome'] == 'certificate_language_refuted'
                            for item in self.report['candidates']))

    def test_shared_seven_bit_relation(self):
        # Independent enumeration: the same seven original bits produce all
        # three rows.  In particular no fresh z,w are sampled at update two.
        def step(x, left, right):
            # Characteristic-coordinate form derived from Rule 30.
            return x ^ (left | right)

        triples = set()
        for u, v, w, z, a, b, c in itertools.product((0, 1), repeat=7):
            r = (a, b, c)
            rp = (step(a, z, w), step(b, a, z), step(c, b, a))
            zp, wp = step(z, w, v), step(w, v, u)
            rpp = (step(rp[0], zp, wp), step(rp[1], rp[0], zp),
                   step(rp[2], rp[1], rp[0]))
            triples.add((r, rp, rpp))
        states, edges = coherent_relation()
        self.assertEqual((len(states), len(edges)), (20, 44))
        self.assertEqual({(state[0], state[1], following[1])
                          for state, following in edges}, triples)
        forbidden = ((0, 0, 0), (1, 0, 0), (1, 1, 1))
        self.assertNotIn(forbidden, triples)

    def test_exact_four_resampled_walks_removed(self):
        _, edges = coherent_relation()
        genuine = {(state[0], state[1], following[1]) for state, following in edges}
        one_step = {(a, b) for a, b, _ in genuine} | {(b, c) for _, b, c in genuine}
        relaxed = {(a, b, c) for a, b in one_step for bb, c in one_step if b == bb}
        removed = relaxed - genuine
        self.assertEqual(removed, {
            ((0, 0, 0), (1, 0, 0), (1, 1, 1)),
            ((0, 0, 1), (1, 0, 1), (1, 1, 0)),
            ((0, 1, 0), (1, 1, 1), (1, 0, 0)),
            ((0, 1, 1), (1, 1, 0), (1, 0, 1)),
        })

    def test_changed_multiplier_rejected(self):
        item = copy.deepcopy(self.report['candidates'][-1])
        item['farkas_multipliers'][next(i for i, x in enumerate(item['farkas_multipliers']) if x)] += 1
        _, rows, rhs, _ = inequality_system(True, None)
        with self.assertRaises(ValueError):
            check_farkas(rows, rhs, item['farkas_multipliers'])

    def test_labels_identify_every_contributing_row(self):
        for item in self.report['candidates']:
            alpha = None if item['alpha'] == 'variable' else item['alpha']
            _, _, _, labels = inequality_system(item['time_parity'], alpha)
            self.assertEqual(len(labels), len({json.dumps(label) for label in labels}))
            expected = [(labels[i], value) for i, value in enumerate(item['farkas_multipliers']) if value]
            actual = [(entry['label'], entry['multiplier'])
                      for entry in item['contributing_constraints']]
            self.assertEqual(actual, expected)

    def test_obstructions_do_not_depend_on_alpha_or_endpoint_allowance(self):
        for item in self.report['candidates']:
            labels = [entry['label'] for entry in item['contributing_constraints']]
            self.assertFalse({'slope_floor', 'initial', 'terminal_bound'} &
                             {label[0] for label in labels})
            # Alpha occurs only in bottom rows with emitted bit C=1.  Every
            # bottom row used by these witnesses has C=0.
            self.assertTrue(all(label[2][2] == 0 for label in labels
                                if label[0] == 'bottom'))


if __name__ == '__main__':
    unittest.main()
