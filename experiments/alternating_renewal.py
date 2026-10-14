#!/usr/bin/env python3
"""Independent sparse-prefix check of specified fixed-weight renewal bounds.

This checks prescribed finite bounds; it does not establish a bound for every
weight. The local truth-table evaluator deliberately does not use the packed
transfer map or the sideways reconstruction used by the original search.
"""
from __future__ import annotations

import argparse
import json


def local(rule: int, left: int, centre: int, right: int) -> int:
    return (rule >> (4 * left + 2 * centre + right)) & 1


def admits_latches(prefix: int, macrosteps: int, rule: int = 30) -> bool:
    """Test all necessary centre constraints in the exact finite left cone.

    At each time an arbitrary right neighbour is allowed. Passing is only
    necessary for a full centre trace; failure suffices for exclusion. The
    initial cone has 2*macrosteps cells, nearest cell first.
    """
    cells = [(prefix >> j) & 1 for j in range(2 * macrosteps)]
    for time in range(2 * macrosteps):
        centre = time & 1
        following = centre ^ 1
        if not any(local(rule, cells[0], centre, right) == following
                   for right in (0, 1)):
            return False
        cells = [local(rule, cells[j + 1], cells[j],
                       centre if j == 0 else cells[j - 1])
                 for j in range(len(cells) - 1)]
    return True


def check_bound(weight: int, bound: int, rule: int = 30) -> dict:
    """Certify q(n)<=bound for all n of weight<=weight, if last set is empty.

    Bits outside the finite initial cone cannot change any tested condition.
    Thus the conclusion has unrestricted support width but fixed weight.
    """
    if weight < 0 or bound < 0:
        raise ValueError('weight and bound must be nonnegative')
    prefixes = {0}
    counts = []
    witness = None
    for m in range(1, bound + 2):
        candidates = (n | (pair << (2 * (m - 1)))
                      for n in prefixes for pair in range(4))
        prefixes = {n for n in candidates
                    if n.bit_count() <= weight and admits_latches(n, m, rule)}
        counts.append(len(prefixes))
        if m == bound and prefixes:
            witness = min(prefixes)
    return {
        'rule': rule,
        'weight_cap': weight,
        'first_failure_bound': bound,
        'initial_cone_bits': 2 * (bound + 1),
        'surviving_prefix_counts': counts,
        'verified_bound': not prefixes,
        'witness_surviving_bound_latches': witness,
        'sharp': not prefixes and (bound == 0 or witness is not None),
        'scope': 'Finite certificate at a fixed weight, unrestricted support width; '
                 'no assertion that such a bound exists for every weight.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=int, default=2)
    parser.add_argument('--bound', type=int, default=8)
    parser.add_argument('--rule', type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(check_bound(args.weight, args.bound, args.rule), indent=2))


if __name__ == '__main__':
    main()
