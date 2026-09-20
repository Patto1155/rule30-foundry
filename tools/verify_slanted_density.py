#!/usr/bin/env python3
"""Standalone integer checker for labeled local-density obstruction proofs.

No optimizer, numpy, synthesis module, or floating arithmetic is imported.
Constraint labels are reconstructed from the documented local inequalities.
This checks language infeasibility, not Rule 30 period exclusion.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path


def constraint(item, label):
    pcount = 2 if item['time_parity'] else 1
    mode = item['mode']
    if mode not in ('newest', 'two', 'three'):
        raise ValueError('unknown mode')

    def density(p, t, a, b):
        return ('F', p, t % pcount, a, b)

    def flux(p, t, r):
        return ('psi', p, t % pcount, *r)

    def terminal(p, t, a, b):
        return ('chi', p, t % pcount, a, b)

    def rule(l, c, r):
        return (30 >> (4*l+2*c+r)) & 1

    def binary(values):
        if any(type(x) is not int or x not in (0, 1) for x in values):
            raise ValueError('nonbinary label')

    kind = label[0]
    if kind == 'nonpositive':
        name = tuple(label[1])
        if item.get('terminal_flux', False) or name[0] not in ('F', 'g'):
            raise ValueError('invalid sign constraint')
        if name[0] == 'g' and not item.get('bottom_correction', False):
            raise ValueError('unknown bottom correction')
        binary(name[1:])
        if len(name) != (5 if name[0] == 'F' else 4):
            raise ValueError('invalid variable arity')
        if name[0] == 'F' and name[2] >= pcount:
            raise ValueError('invalid time phase')
        return [(name, -1)], 0
    if kind == 'interior':
        _, p, t, r, following = label
        binary([p, t, *r, *following])
        if len(r) != 3 or len(following) != 3 or t >= pcount:
            raise ValueError('invalid interior label')
        a, b, c = r
        legal = set()
        for z, w, arbitrary in itertools.product((0, 1), repeat=3):
            aa = arbitrary if mode != 'three' else rule(a, z, w)
            bb = arbitrary if mode == 'newest' else rule(b, a, z)
            if mode == 'newest':
                for free in (0, 1):
                    legal.add((free, arbitrary, rule(c, b, a)))
            else:
                legal.add((aa, bb, rule(c, b, a)))
        if tuple(following) not in legal:
            raise ValueError('illegal local transition')
        return [(density(p, t, b, c), 1), (density(1-p, t, a, b), -1),
                (flux(p, t+1, following), 1), (flux(p, t, r), -1)], 0
    if kind == 'bottom':
        _, p, r = label
        binary([p, *r])
        if len(r) != 3:
            raise ValueError('invalid bottom label')
        a, b, c = r
        terms = [(flux(p, 0, r), 1), (('b', p), -1)]
        if item.get('bottom_correction', False):
            terms += [(('g', p, b, c), 1), (('g', 1-p, a, b), -1)]
        return terms, 1-8*c
    if kind == 'top':
        _, p, v = label
        binary([p, v])
        a, b, c = (0, v, 1-v) if p == 0 else (1, 1, 1-v)
        top = (1, 1) if p == 0 else (0, v)
        return [(density(p, p, b, c), 1), (density(1-p, p, a, b), -1),
                (density(p, 1-p, *top), 1), (flux(p, p, (a, b, c)), -1), (('b', p), 1)], 0
    if kind == 'initial':
        _, x = label
        binary([x])
        terms = [(density(0, 0, x, 1-x), 1), (density(0, 1, 1, 1), 1)]
        if item.get('bottom_correction', False):
            terms += [(('g', 0, x, 1-x), 1)]
        return terms, -7
    if kind == 'terminal_flow':
        _, p, t, r, following = label
        binary([p, t, *r, *following])
        if not item.get('terminal_flux', False) or t >= pcount or len(r) != 2 or len(following) != 2:
            raise ValueError('invalid terminal-flow label')
        a, b = r
        legal = {(rule(a, z, w), rule(b, a, z)) for z, w in itertools.product((0, 1), repeat=2)}
        if tuple(following) not in legal:
            raise ValueError('illegal terminal transition')
        return [(density(p, t, a, b), -1), (terminal(p, t, a, b), 1),
                (terminal(p, t+1, *following), -1)], 0
    if kind == 'terminal_bound':
        _, p, r, top = label
        binary([p, *r, *top])
        if not item.get('terminal_flux', False) or len(r) != 2 or tuple(top) not in ([(1, 1)] if p == 0 else [(0, 0), (0, 1)]):
            raise ValueError('invalid terminal endpoint')
        terms = [(terminal(p, 0, *r), -1), (density(p, 1-p, *top), -1),
                 (terminal(p, 1-p, *top), 1)]
        if item.get('bottom_correction', False):
            terms += [(('g', p, *r), -1)]
        return terms, 0
    raise ValueError('unknown constraint kind')


def verify(item):
    sums = defaultdict(int)
    bound = 0
    for entry in item['contributing_constraints']:
        weight = entry['multiplier']
        if type(weight) is not int or weight <= 0:
            raise ValueError('multipliers must be positive integers')
        terms, rhs = constraint(item, entry['label'])
        for name, coefficient in terms:
            sums[name] += weight * coefficient
        bound += weight * rhs
    if any(sums.values()) or bound <= 0:
        raise ValueError('weighted constraints do not give a contradiction')
    return bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate', type=Path)
    args = parser.parse_args()
    report = json.loads(args.certificate.read_text())
    results = [verify(item) for item in report['candidates']]
    if not results:
        raise ValueError('empty certificate collection')
    print(json.dumps({'verified': len(results), 'integer_contradictions': results,
                      'claim': 'Only the specified sufficient certificate languages are refuted.'}))


if __name__ == '__main__':
    main()
