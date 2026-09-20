#!/usr/bin/env python3
"""Synthesize or exactly refute small characteristic density/flux languages.

The integer inequality table follows docs/theory/slanted-cone-geometry.md.
Floating point optimization proposes answers only: rational Farkas witnesses
are converted to integers and checked exactly. Infeasibility rejects this
language, never the cone-charge conjecture. No solver status is a proof.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import itertools
import json
from math import gcd, lcm
from pathlib import Path


def inequality_system(mode: str, time_parity: bool, bottom_correction: bool = False,
                      terminal_flux: bool = False):
    if mode not in ('newest', 'two', 'three'):
        raise ValueError('unknown transition refinement')
    bits = list(itertools.product((0, 1), repeat=3))
    phases = range(2 if time_parity else 1)
    names = ([('F', p, t, a, b) for p in (0, 1) for t in phases
              for a, b in itertools.product((0, 1), repeat=2)] +
             [('psi', p, t, *r) for p in (0, 1) for t in phases for r in bits] +
             [('b', p) for p in (0, 1)] +
             ([('g', p, a, b) for p in (0, 1) for a, b in itertools.product((0, 1), repeat=2)]
              if bottom_correction else []) +
             ([('chi', p, t, a, b) for p in (0, 1) for t in phases
               for a, b in itertools.product((0, 1), repeat=2)] if terminal_flux else []))
    indices = {name: j for j, name in enumerate(names)}
    rows, rhs, labels = [], [], []

    def add(terms, bound, label):
        row = [0] * len(names)
        for name, coefficient in terms:
            row[indices[name]] += coefficient
        rows.append(row); rhs.append(bound); labels.append(label)

    def F(p, t, a, b):
        return ('F', p, t % len(phases), a, b)

    def P(p, t, r):
        return ('psi', p, t % len(phases), *r)

    def local_difference(p, t, r):
        a, b, c = r
        return [(F(p, t, b, c), 1), (F(1-p, t, a, b), -1)]

    for name in names:
        if name[0] in ('F', 'g') and not terminal_flux:
            add([(name, -1)], 0, ['nonpositive', list(name)])
    if terminal_flux:
        def H(p, t, a, b):
            return ('chi', p, t % len(phases), a, b)
        for p in (0, 1):
            for t in phases:
                for a, b in itertools.product((0, 1), repeat=2):
                    successors = {(a ^ (z | w), b ^ (a | z))
                                  for z, w in itertools.product((0, 1), repeat=2)}
                    for aa, bb in sorted(successors):
                        add([(F(p, t, a, b), -1), (H(p, t, a, b), 1),
                             (H(p, t+1, aa, bb), -1)], 0,
                            ['terminal_flow', p, t, [a, b], [aa, bb]])
            for a, b, v in itertools.product((0, 1), repeat=3):
                top = (1, 1) if p == 0 else (0, v)
                correction = [(('g', p, a, b), -1)] if bottom_correction else []
                add([(H(p, 0, a, b), -1), (F(p, 1-p, *top), -1),
                     (H(p, 1-p, *top), 1)] + correction, 0,
                    ['terminal_bound', p, [a, b], list(top)])
    for p in (0, 1):
        for t in phases:
            for r in bits:
                a, b, c = r
                if mode == 'newest':
                    successors = {(aa, bb, c ^ (a | b)) for aa, bb in itertools.product((0, 1), repeat=2)}
                elif mode == 'two':
                    successors = {(aa, b ^ (a | z), c ^ (a | b)) for aa, z in itertools.product((0, 1), repeat=2)}
                else:
                    successors = {(a ^ (z | w), b ^ (a | z), c ^ (a | b)) for z, w in itertools.product((0, 1), repeat=2)}
                for successor in sorted(successors):
                    add(local_difference(p, t, r) + [(P(p, t+1, successor), 1), (P(p, t, r), -1)],
                        0, ['interior', p, t, list(r), list(successor)])
        for r in bits:
            correction = [(('g', p, r[1], r[2]), 1), (('g', 1-p, r[0], r[1]), -1)] if bottom_correction else []
            add([(P(p, 0, r), 1), (('b', p), -1)] + correction, 1-8*r[2], ['bottom', p, list(r)])
        for v in (0, 1):
            r = (0, v, 1 ^ v) if p == 0 else (1, 1, 1 ^ v)
            top = (1, 1) if p == 0 else (0, v)
            add(local_difference(p, p, r) + [(F(p, 1-p, *top), 1),
                 (P(p, p, r), -1), (('b', p), 1)], 0, ['top', p, v])
    for x in (0, 1):
        correction = [(('g', 0, x, 1-x), 1)] if bottom_correction else []
        add([(F(0, 0, x, 1-x), 1), (F(0, 1, 1, 1), 1)] + correction, -7, ['initial', x])
    return names, rows, rhs, labels


def check_farkas(rows, rhs, multipliers):
    """Exact arithmetic verifier, no solver or tolerance; A*x>=b convention."""
    if len(multipliers) != len(rows) or any(type(x) is not int or x < 0 for x in multipliers):
        raise ValueError('multipliers must be nonnegative integers, one per row')
    column_sums = [sum(multiplier * row[j] for multiplier, row in zip(multipliers, rows))
                   for j in range(len(rows[0]))]
    contradiction = sum(a*b for a, b in zip(multipliers, rhs))
    if any(column_sums) or contradiction <= 0:
        raise ValueError('invalid Farkas identity')
    return {'verified': True, 'nonzero_rows': sum(x > 0 for x in multipliers),
            'identity': f'0 >= {contradiction}', 'contradiction': contradiction}


def rational_integers(values):
    fractions = [Fraction(float(v)).limit_denominator(1000000) for v in values]
    scale = lcm(*(x.denominator for x in fractions))
    integers = [int(x * scale) for x in fractions]
    common = 0
    for value in integers:
        common = gcd(common, abs(value))
    return [x // (common or 1) for x in integers]


def solve(mode, time_parity, bottom_correction=False, terminal_flux=False):
    import numpy as np
    from scipy.optimize import linprog
    names, rows, rhs, labels = inequality_system(mode, time_parity, bottom_correction, terminal_flux)
    A, b = np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)
    answer = linprog(np.zeros(len(names)), A_ub=-A, b_ub=-b,
                     bounds=[(None, None)] * len(names), method='highs')
    report = {'mode': mode, 'time_parity': time_parity, 'bottom_correction': bottom_correction,
              'terminal_flux': terminal_flux, 'variables': len(names),
              'inequalities': len(rows), 'optimizer_status': int(answer.status)}
    if answer.success:
        candidate = [Fraction(float(x)).limit_denominator(1000000) for x in answer.x]
        if not all(sum(Fraction(a)*x for a, x in zip(row, candidate)) >= bound for row, bound in zip(rows, rhs)):
            raise RuntimeError('floating candidate failed exact rational check')
        report.update(outcome='rational_potential_candidate', potential=[str(x) for x in candidate])
        return report
    # Independently solve the Farkas alternative. An optimizer's infeasibility
    # status alone is not accepted as a mathematical exclusion.
    dual = linprog(np.ones(len(rows)), A_eq=np.vstack((A.T, b)),
                   b_eq=np.r_[np.zeros(len(names)), 1.0], bounds=(0, None), method='highs')
    if not dual.success:
        report['outcome'] = 'unverified_solver_failure'
        return report
    multipliers = rational_integers(dual.x)
    checked = check_farkas(rows, rhs, multipliers)
    report.update(outcome='certificate_language_refuted', farkas_multipliers=multipliers,
                  verification=checked, contributing_constraints=[{'label': label, 'multiplier': value}
                    for label, value in zip(labels, multipliers) if value])
    return report


def verify_report(report):
    for item in report['candidates']:
        _, rows, rhs, _ = inequality_system(item['mode'], item['time_parity'],
                 item.get('bottom_correction', False), item.get('terminal_flux', False))
        if item['outcome'] != 'certificate_language_refuted':
            raise ValueError('this verifier expects an obstruction certificate')
        check_farkas(rows, rhs, item['farkas_multipliers'])
    return {'verified_obstructions': len(report['candidates']), 'arithmetic': 'integers only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_report(json.loads(args.verify.read_text())))); return
    report = {'experiment': 'slanted-local-density-flux',
              'claim_limit': 'Refutes specified sufficient certificate languages, not the cone-charge conjecture.',
              'candidates': ([solve(mode, parity, correction) for correction in (False, True) for parity in (False, True)
                             for mode in ('newest', 'two', 'three')] +
                             [solve('three', parity, True, True) for parity in (False, True)])}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in item.items() if k not in ('farkas_multipliers', 'contributing_constraints')}
                      for item in report['candidates']], indent=2))


if __name__ == '__main__':
    main()
