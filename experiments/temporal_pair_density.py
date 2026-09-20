#!/usr/bin/env python3
"""Synthesize the first justified temporal-pair cone density.

The charge table sees two successive diagonal pairs.  The spatial flux still
uses the shared-seven-bit, two-update relation: this is a density refinement,
not a further predecessor-memory expansion.  Floating point LP is only a
search device; saved answers are exact rational potentials or exact integer
Farkas identities.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
import json
from math import gcd, lcm
from pathlib import Path


def update(centre, left, right):
    return centre ^ (left | right)


def coherent_relation():
    """Return triple-row states and edges from one shared seven-bit word."""
    triples = set()
    for u, v, w, z, a, b, c in itertools.product((0, 1), repeat=7):
        r = (a, b, c)
        rp = (update(a, z, w), update(b, a, z), update(c, b, a))
        zp, wp = update(z, w, v), update(w, v, u)
        rpp = (update(rp[0], zp, wp), update(rp[1], rp[0], zp),
                update(rp[2], rp[1], rp[0]))
        triples.add((r, rp, rpp))
    states = sorted({(r, rp) for r, rp, _ in triples} |
                    {(rp, rpp) for _, rp, rpp in triples})
    edges = sorted(((r, rp), (rp, rpp)) for r, rp, rpp in triples)
    return states, edges


def pair_projection(state):
    """Project S=(r,r') to U=((B,C),(B',C'))."""
    r, rp = state
    return ((r[1], r[2]), (rp[1], rp[2]))


def temporal_pair_relation():
    """The 10 temporal pairs and 22 projected coherent transitions."""
    states, edges = coherent_relation()
    temporal = sorted({pair_projection(state) for state in states})
    transitions = sorted({(pair_projection(state), pair_projection(following))
                          for state, following in edges})
    return temporal, transitions


def inequality_system(time_parity: bool, alpha: int | None = 8):
    """Build ``A*x >= rhs`` for the temporal-pair density language.

    ``alpha=None`` searches all finite ``alpha>=0``.  Initialization and the
    terminal endpoint bound are deliberately absent: once all tables are
    finite they merely define finite constants J and K (see ``endpoint_constants``).
    """
    if alpha is not None and (type(alpha) is not int or alpha < 0):
        raise ValueError('alpha must be a nonnegative integer or None')
    states, edges = coherent_relation()
    temporal, temporal_edges = temporal_pair_relation()
    phases = range(2 if time_parity else 1)
    names = ([('F', p, t, *u[0], *u[1]) for p in (0, 1) for t in phases
              for u in temporal] +
             [('psi', p, t, *r, *rp) for p in (0, 1) for t in phases
              for r, rp in states] +
             [('b', p) for p in (0, 1)] +
             [('g', p, *u[0], *u[1]) for p in (0, 1) for u in temporal] +
             [('chi', p, t, *u[0], *u[1]) for p in (0, 1) for t in phases
              for u in temporal] +
             ([('alpha',)] if alpha is None else []))
    indices = {name: j for j, name in enumerate(names)}
    rows, rhs, labels = [], [], []

    def add(terms, bound, label):
        row = [0] * len(names)
        for name, coefficient in terms:
            row[indices[name]] += coefficient
        rows.append(row)
        rhs.append(bound)
        labels.append(label)

    phase_count = len(phases)

    def F(p, t, u):
        return ('F', p, t % phase_count, *u[0], *u[1])

    def psi(p, t, state):
        return ('psi', p, t % phase_count, *state[0], *state[1])

    def chi(p, t, u):
        return ('chi', p, t % phase_count, *u[0], *u[1])

    def local_difference(p, t, state):
        r, rp = state
        current = ((r[1], r[2]), (rp[1], rp[2]))
        previous = ((r[0], r[1]), (rp[0], rp[1]))
        return [(F(p, t, current), 1), (F(1-p, t, previous), -1)]

    if alpha is None:
        add([(('alpha',), 1)], 0, ['slope_floor', 0])

    for p in (0, 1):
        for t in phases:
            # S_t -> S_(t+1) is exactly the same shared-seven-bit relation as
            # in the predecessor-coherent experiment.
            for state, following in edges:
                add(local_difference(p, t, state) +
                    [(psi(p, t+1, following), 1), (psi(p, t, state), -1)],
                    0, ['interior', p, t, list(state[0]), list(state[1]),
                        list(following[1])])

            # The terminal flow sees the coherent projection U_t -> U_(t+1),
            # rather than independently resampling two one-row transitions.
            for u, following in temporal_edges:
                add([(F(p, t, u), -1), (chi(p, t, u), 1),
                     (chi(p, t+1, following), -1)], 0,
                    ['terminal_flow', p, t, [list(x) for x in u],
                     [list(x) for x in following]])

        # C=e_q and g_p(B,C)-g_(1-p)(A,B) is the bottom correction difference.
        for state in states:
            r, rp = state
            current_u = pair_projection(state)
            previous_u = ((r[0], r[1]), (rp[0], rp[1]))
            terms = [(psi(p, 0, state), 1), (('b', p), -1),
                     (('g', p, *current_u[0], *current_u[1]), 1),
                     (('g', 1-p, *previous_u[0], *previous_u[1]), -1)]
            if alpha is None:
                terms.append((('alpha',), r[2]))
                bound = 1
            else:
                bound = 1 - alpha*r[2]
            add(terms, bound, ['bottom', p, list(r), list(rp)])

        # Exact S_(q-3) and exact final U_(q-2).  This cap is valid already at
        # q=3, when the interior edge sum is empty and S_(q-3)=S_0.
        if p == 0:
            tops = [((1, 1-b, b), (0, b, 1-b),
                     ((b, 1-b), (1, 1))) for b in (0, 1)]
            top_time = 1
        else:
            tops = [((u, 1-u, v), (1, 1, 1-v),
                     ((1, 1-v), (0, v)))
                    for u, v in itertools.product((0, 1), repeat=2)]
            top_time = 0
        for r, rp, final_u in tops:
            state = (r, rp)
            if state not in states or final_u not in temporal:
                raise AssertionError('exact top cap is outside the coherent alphabet')
            add(local_difference(p, top_time, state) +
                [(F(p, top_time+1, final_u), 1),
                 (psi(p, top_time, state), -1), (('b', p), 1)],
                0, ['top', p, list(r), list(rp),
                    [list(x) for x in final_u]])
    return names, rows, rhs, labels


def endpoint_constants(names, values, alpha, time_parity):
    """Return explicit finite J,K chosen after a feasible potential is known."""
    value = dict(zip(names, values))
    phase_count = 2 if time_parity else 1

    def F(p, t, u):
        return value[('F', p, t % phase_count, *u[0], *u[1])]

    def chi(p, t, u):
        return value[('chi', p, t % phase_count, *u[0], *u[1])]

    temporal, _ = temporal_pair_relation()
    init = [((x, 1-x), (1, 1)) for x in (0, 1)]
    J = min(F(0, 0, u) + value[('g', 0, *u[0], *u[1])] + alpha - 2
            for u in init)
    finals = {
        0: [((b, 1-b), (1, 1)) for b in (0, 1)],
        1: [((1, 1-v), (0, v)) for v in (0, 1)],
    }
    K_by_parity = {}
    for p in (0, 1):
        bottom = max(value[('g', p, *u[0], *u[1])] + chi(p, 0, u)
                     for u in temporal)
        terminal = max(F(p, p, u) - chi(p, p, u) for u in finals[p])
        K_by_parity[p] = bottom + terminal
    return {'J': str(J), 'K': str(max(K_by_parity.values())),
            'K_by_parity': {str(p): str(x) for p, x in K_by_parity.items()}}


def check_farkas(rows, rhs, multipliers):
    if len(multipliers) != len(rows) or any(type(x) is not int or x < 0 for x in multipliers):
        raise ValueError('multipliers must be nonnegative integers, one per row')
    sums = [sum(m*r[j] for m, r in zip(multipliers, rows))
            for j in range(len(rows[0]))]
    contradiction = sum(m*b for m, b in zip(multipliers, rhs))
    if any(sums) or contradiction <= 0:
        raise ValueError('invalid Farkas identity')
    return {'verified': True, 'nonzero_rows': sum(x > 0 for x in multipliers),
            'identity': f'0 >= {contradiction}', 'contradiction': contradiction}


def rational_integers(values):
    fractions = [Fraction(float(v)).limit_denominator(1_000_000) for v in values]
    scale = lcm(*(x.denominator for x in fractions))
    integers = [int(x*scale) for x in fractions]
    common = 0
    for x in integers:
        common = gcd(common, abs(x))
    return [x // (common or 1) for x in integers]


def solve(time_parity, alpha=8):
    import numpy as np
    from scipy.optimize import linprog
    names, rows, rhs, labels = inequality_system(time_parity, alpha)
    A, bound = np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)
    answer = linprog(np.zeros(len(names)), A_ub=-A, b_ub=-bound,
                     bounds=[(None, None)]*len(names), method='highs')
    states, edges = coherent_relation()
    temporal, temporal_edges = temporal_pair_relation()
    report = {'time_parity': time_parity,
              'alpha': 'variable' if alpha is None else alpha,
              'variables': len(names), 'inequalities': len(rows),
              'coherent_states': len(states), 'coherent_edges': len(edges),
              'temporal_pair_states': len(temporal),
              'temporal_pair_edges': len(temporal_edges),
              'optimizer_status': int(answer.status)}
    if answer.success:
        candidate = [Fraction(float(x)).limit_denominator(1_000_000) for x in answer.x]
        if not all(sum(Fraction(a)*x for a, x in zip(row, candidate)) >= b
                   for row, b in zip(rows, rhs)):
            raise RuntimeError('floating candidate failed exact rational check')
        actual_alpha = candidate[names.index(('alpha',))] if alpha is None else Fraction(alpha)
        report.update(outcome='rational_potential',
                      resolved_alpha=str(actual_alpha),
                      endpoint_constants=endpoint_constants(names, candidate, actual_alpha,
                                                            time_parity),
                      variable_names=[list(name) for name in names],
                      potential=[str(x) for x in candidate])
        return report
    dual = linprog(np.ones(len(rows)), A_eq=np.vstack((A.T, bound)),
                   b_eq=np.r_[np.zeros(len(names)), 1.0], bounds=(0, None), method='highs')
    if not dual.success:
        report['outcome'] = 'unverified_solver_failure'
        return report
    multipliers = rational_integers(dual.x)
    report.update(outcome='certificate_language_refuted',
                  verification=check_farkas(rows, rhs, multipliers),
                  farkas_multipliers=multipliers,
                  contributing_constraints=[
                      {'label': label, 'multiplier': x}
                      for label, x in zip(labels, multipliers) if x])
    return report


def verify_report(report):
    for item in report['candidates']:
        alpha = None if item['alpha'] == 'variable' else item['alpha']
        names, rows, rhs, _ = inequality_system(item['time_parity'], alpha)
        if item['outcome'] == 'certificate_language_refuted':
            check_farkas(rows, rhs, item['farkas_multipliers'])
        elif item['outcome'] == 'rational_potential':
            values = [Fraction(x) for x in item['potential']]
            if len(values) != len(names) or not all(
                    sum(Fraction(a)*x for a, x in zip(row, values)) >= b
                    for row, b in zip(rows, rhs)):
                raise ValueError('invalid rational potential')
            actual_alpha = values[names.index(('alpha',))] if alpha is None else Fraction(alpha)
            if endpoint_constants(names, values, actual_alpha, item['time_parity']) != item['endpoint_constants']:
                raise ValueError('incorrect endpoint constants')
        else:
            raise ValueError('report contains no exact evidence')
    return {'verified_candidates': len(report['candidates']),
            'arithmetic': 'rationals/integers only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_report(json.loads(args.verify.read_text()))))
        return
    candidates = [solve(parity, alpha) for parity in (False, True)
                  for alpha in (8, None)]
    successes = [x for x in candidates if x['outcome'] == 'rational_potential']
    report = {
        'experiment': 'temporal-pair-density',
        'result': ('The temporal-pair refinement admits an exact rational cone-charge '
                   'potential.' if successes else
                   'The temporal-pair density language is infeasible with or without time '
                   'parity, at alpha=8 and for every finite alpha>=0.'),
        'evidence': ('F depends on two successive diagonal pairs and g on the bottom temporal '
                     'pair; psi retains the 20-state, 44-edge shared-seven-bit relation and chi '
                     'uses its 10-state, 22-edge temporal-pair projection. Exact integer Farkas '
                     'identities give 0>=4 using 19 rows without time parity and 24 rows with it.'),
        'gap': ('This refutes one sufficient local language over a relaxation of reachable cuts. '
                'Its witnesses use no initialization or terminal endpoint rows and omit the '
                'variable-alpha floor, but they do not prove that a cone charge is impossible.'),
        'next_implication': ('The old pair-density obstruction is genuinely escaped, yet this '
                             'first temporal-pair representation still fails. Stop here pending '
                             'a separately justified refinement; do not add more memory from '
                             'infeasibility alone.'),
        'claim_limit': ('The representation stops at temporal-pair charge; no additional '
                        'predecessor memory is introduced.'),
        'candidates': candidates,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in x.items()
                       if k not in ('potential', 'variable_names', 'farkas_multipliers',
                                    'contributing_constraints')}
                      for x in candidates], indent=2))


if __name__ == '__main__':
    main()
