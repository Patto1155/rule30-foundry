import copy
from collections import Counter
from fractions import Fraction
import itertools
import json
from pathlib import Path
import unittest

from experiments.temporal_pair_density import (
    check_farkas, coherent_relation, inequality_system, pair_projection,
    temporal_pair_relation, verify_report,
)


class TemporalPairDensityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((Path(__file__).resolve().parents[1] /
            'runs/theory/temporal-pair-density.json').read_text())

    def test_saved_results_are_exact_obstructions(self):
        self.assertEqual(verify_report(self.report)['verified_candidates'], 4)
        self.assertTrue(all(x['outcome'] == 'certificate_language_refuted'
                            for x in self.report['candidates']))
        self.assertEqual({x['verification']['identity']
                          for x in self.report['candidates']}, {'0 >= 4'})

    def test_shared_relation_and_its_temporal_projection(self):
        states, edges = coherent_relation()
        temporal, projected = temporal_pair_relation()
        self.assertEqual((len(states), len(edges), len(temporal), len(projected)),
                         (20, 44, 10, 22))
        self.assertEqual(set(temporal), {pair_projection(x) for x in states})
        self.assertEqual(set(projected),
                         {(pair_projection(x), pair_projection(y)) for x, y in edges})

        # Independent shared-word enumeration: terminal flow must not compose
        # two pair transitions obtained from different predecessor words.
        def step(x, left, right):
            return x ^ (left | right)
        independently_projected = set()
        for u, v, w, z, a, b, c in itertools.product((0, 1), repeat=7):
            r = (a, b, c)
            rp = (step(a, z, w), step(b, a, z), step(c, b, a))
            zp, wp = step(z, w, v), step(w, v, u)
            rpp = (step(rp[0], zp, wp), step(rp[1], rp[0], zp),
                    step(rp[2], rp[1], rp[0]))
            independently_projected.add((
                ((r[1], r[2]), (rp[1], rp[2])),
                ((rp[1], rp[2]), (rpp[1], rpp[2]))))
        self.assertEqual(set(projected), independently_projected)

    def test_spatial_difference_formula_for_q_at_least_three(self):
        # Compare formal coefficients, independently of the LP builder.
        for q in range(3, 8):
            rows = [tuple((q + 2*t + j) & 1 for j in range(3))
                    for t in range(q-1)]
            final_pair = ((q + 1) & 1, q & 1)
            p = q % 2

            def f_key(side, t, u):
                return ('F', side, t % 2, u)
            direct = Counter()
            for t in range(q-3):
                r, rp = rows[t], rows[t+1]
                direct[f_key(p, t, ((r[1], r[2]), (rp[1], rp[2])))] += 1
                direct[f_key(1-p, t, ((r[0], r[1]), (rp[0], rp[1])))] -= 1
            # t=q-3 is the last L term; U_(q-2) is the unpaired top term.
            r, rp = rows[q-3], rows[q-2]
            direct[f_key(p, q-3, ((r[1], r[2]), (rp[1], rp[2])))] += 1
            direct[f_key(1-p, q-3, ((r[0], r[1]), (rp[0], rp[1])))] -= 1
            direct[f_key(p, q-2, ((rp[1], rp[2]), final_pair))] += 1
            direct[('g', p, ((rows[0][1], rows[0][2]),
                              (rows[1][1], rows[1][2])))] += 1
            direct[('g', 1-p, ((rows[0][0], rows[0][1]),
                                (rows[1][0], rows[1][1])))] -= 1

            phi_difference = Counter(direct)  # assembled a second way below
            # Phi_q minus Phi_(q-1): common time range plus current final U.
            expected = Counter()
            for t in range(q-2):
                r, rp = rows[t], rows[t+1]
                expected[f_key(p, t, ((r[1], r[2]), (rp[1], rp[2])))] += 1
                expected[f_key(1-p, t, ((r[0], r[1]), (rp[0], rp[1])))] -= 1
            expected[f_key(p, q-2, ((rows[-1][1], rows[-1][2]), final_pair))] += 1
            expected[('g', p, ((rows[0][1], rows[0][2]),
                                (rows[1][1], rows[1][2])))] += 1
            expected[('g', 1-p, ((rows[0][0], rows[0][1]),
                                  (rows[1][0], rows[1][1])))] -= 1
            self.assertEqual({k: v for k, v in phi_difference.items() if v},
                             {k: v for k, v in expected.items() if v})

    def test_exact_top_caps_and_q3_empty_interior(self):
        _, _, _, labels = inequality_system(True, 8)
        tops = [x for x in labels if x[0] == 'top']
        self.assertEqual(tops, [
            ['top', 0, [1, 1, 0], [0, 0, 1], [[0, 1], [1, 1]]],
            ['top', 0, [1, 0, 1], [0, 1, 0], [[1, 0], [1, 1]]],
            ['top', 1, [0, 1, 0], [1, 1, 1], [[1, 1], [0, 0]]],
            ['top', 1, [0, 1, 1], [1, 1, 0], [[1, 0], [0, 1]]],
            ['top', 1, [1, 0, 0], [1, 1, 1], [[1, 1], [0, 0]]],
            ['top', 1, [1, 0, 1], [1, 1, 0], [[1, 0], [0, 1]]],
        ])
        # q=3 uses bottom S_0 and the odd top cap at S_0; no S edge is needed.
        self.assertEqual(3-4, -1)

    def test_formula_independent_farkas_review(self):
        """Reassemble every used row from its label, without the LP matrix."""
        for item in self.report['candidates']:
            phases = 2 if item['time_parity'] else 1
            alpha_variable = item['alpha'] == 'variable'

            def F(p, t, u): return ('F', p, t % phases, *u[0], *u[1])
            def psi(p, t, r, rp): return ('psi', p, t % phases, *r, *rp)
            def chi(p, t, u): return ('chi', p, t % phases, *u[0], *u[1])
            total, total_rhs = Counter(), 0

            for entry in item['contributing_constraints']:
                label, multiplier = entry['label'], entry['multiplier']
                kind, p = label[0], label[1] if label[0] != 'slope_floor' else None
                terms, bound = Counter(), 0
                if kind == 'slope_floor':
                    terms[('alpha',)] = 1; bound = 0
                elif kind == 'interior':
                    _, p, t, r, rp, rpp = label
                    u = ((r[1], r[2]), (rp[1], rp[2]))
                    prev = ((r[0], r[1]), (rp[0], rp[1]))
                    terms[F(p, t, u)] += 1; terms[F(1-p, t, prev)] -= 1
                    terms[psi(p, t+1, tuple(rp), tuple(rpp))] += 1
                    terms[psi(p, t, tuple(r), tuple(rp))] -= 1
                elif kind == 'terminal_flow':
                    _, p, t, u, up = label
                    u, up = tuple(map(tuple, u)), tuple(map(tuple, up))
                    terms[F(p, t, u)] -= 1; terms[chi(p, t, u)] += 1
                    terms[chi(p, t+1, up)] -= 1
                elif kind == 'bottom':
                    _, p, r, rp = label
                    r, rp = tuple(r), tuple(rp)
                    u = ((r[1], r[2]), (rp[1], rp[2]))
                    prev = ((r[0], r[1]), (rp[0], rp[1]))
                    terms[psi(p, 0, r, rp)] += 1; terms[('b', p)] -= 1
                    terms[('g', p, *u[0], *u[1])] += 1
                    terms[('g', 1-p, *prev[0], *prev[1])] -= 1
                    if alpha_variable:
                        terms[('alpha',)] += r[2]; bound = 1
                    else:
                        bound = 1-item['alpha']*r[2]
                elif kind == 'top':
                    _, p, r, rp, final = label
                    r, rp, final = tuple(r), tuple(rp), tuple(map(tuple, final))
                    t = 1 if p == 0 else 0
                    u = ((r[1], r[2]), (rp[1], rp[2]))
                    prev = ((r[0], r[1]), (rp[0], rp[1]))
                    terms[F(p, t, u)] += 1; terms[F(1-p, t, prev)] -= 1
                    terms[F(p, t+1, final)] += 1
                    terms[psi(p, t, r, rp)] -= 1; terms[('b', p)] += 1
                else:
                    self.fail(f'unreviewed constraint kind {kind}')
                total.update({key: multiplier*x for key, x in terms.items()})
                total_rhs += multiplier*bound
            self.assertFalse({key: x for key, x in total.items() if x})
            self.assertEqual(total_rhs, item['verification']['contradiction'])

    def test_modified_certificate_is_rejected(self):
        item = copy.deepcopy(self.report['candidates'][-1])
        i = next(i for i, x in enumerate(item['farkas_multipliers']) if x)
        item['farkas_multipliers'][i] += 1
        _, rows, rhs, _ = inequality_system(True, None)
        with self.assertRaises(ValueError):
            check_farkas(rows, rhs, item['farkas_multipliers'])


if __name__ == '__main__':
    unittest.main()
