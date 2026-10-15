#!/usr/bin/env python3
"""Exact finite alternating-prefix transients; no physical reachability claim."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'free': (), 'L1': ('11',), 'L2': ('11', '00000'),
             'L3': ('11', '00000', '101001', '0100101'),
             'L4': ('11', '00000', '101001', '0100101', '010010001',
                    '0101000101', '0101010000', '01010001001', '10010001001',
                    '010010000101', '010100010001', '100100010000')}


def reference_module(path=None):
    path = Path(path) if path else ROOT / 'experiments/period2_column_reduction.py'
    spec = importlib.util.spec_from_file_location('transient_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allowed(text, language):
    return not any(f in text for f in FORBIDDEN[language])


def first_transient(word, depth):
    """First non-seven gap in initial one positions; missing endpoint is censored.

    The first one must be at depth1. If all observed gaps are seven, no
    transient is identified. Positions use one-based initial left depth.
    """
    ones = [j for j in range(1, depth + 1) if word >> (j - 1) & 1]
    if not ones or ones[0] != 1:
        raise ValueError('expected the phase-zero sparse orbit initial one')
    for left, right in zip(ones, ones[1:]):
        if right - left != 7:
            return {'last_sparse_one': left, 'next_one': right,
                    'zero_start': left + 1, 'zero_end': right - 1,
                    'zeros': right - left - 1, 'censored': False}
    return {'last_sparse_one': ones[-1], 'zeros_observed': depth - ones[-1],
            'censored': True}


def periodic_columns(rule=30):
    """Exact cyclic temporal recurrence, residues t modulo4."""
    previous, current = (0, 0, 1, 1), (0, 1, 0, 1)
    columns = [previous, current]
    for _ in range(7):
        following = tuple(current[(t + 1) % 4] ^
                          ((current[t] | previous[t]) if rule == 30 else previous[t])
                          for t in range(4))
        columns.append(following)
        previous, current = current, following
    return columns


def tail_table(ref, max_m=48, tail_length=12):
    table = {}
    for language in FORBIDDEN:
        count = censored = 0
        best = None
        by_m = []
        for m in range(1, max_m + 1):
            local_best = None
            for tail in itertools.product('01', repeat=tail_length):
                w = ''.join(tail)
                text = '01' * m + w
                if not allowed(text, language):
                    continue
                count += 1
                depth = 2 * len(text)
                packed = ref.completion_from_u(tuple(map(int, text)), depth)
                transient = first_transient(packed, depth)
                if transient['censored']:
                    censored += 1
                    continue
                record = dict(transient, m=m, tail=w, depth=depth)
                if local_best is None or record['zeros'] > local_best['zeros']:
                    local_best = record
                if best is None or record['zeros'] > best['zeros']:
                    best = record
            by_m.append({'m': m, 'maximum_uncensored': local_best})
        table[language] = {'accepted_words': count, 'censored': censored,
                           'maximum_uncensored': best, 'by_m': by_m}
    return table


def verify(ref):
    sys.path.insert(0, str(ROOT / 'experiments'))
    from support_growth import completion, forward_check
    r = 0x98f9d8a0
    physical = ref.completion(r, 192)
    assert physical == completion(r, 192)
    forward_check(r, physical, 192)
    actual = first_transient(physical, 192)
    assert actual == {'last_sparse_one': 176, 'next_one': 191, 'zero_start': 177,
                      'zero_end': 190, 'zeros': 14, 'censored': False}
    assert (physical & ((1 << 190) - 1)).bit_count() == 26
    u = '01' * 45 + '000101'
    assert allowed(u, 'L3')
    relaxed = ref.completion_from_u(tuple(map(int, u)), 192)
    killed = first_transient(relaxed, 192)
    assert killed['zeros'] == 15 and killed['next_one'] == 192
    assert (relaxed & ((1 << 191) - 1)).bit_count() == 26
    # Independent dictionary forward simulation applies to physical r only.
    rng = random.Random(30715)
    for _ in range(64):
        r0 = rng.getrandbits(48)
        assert ref.completion(r0, 96) == completion(r0, 96)
    cols = periodic_columns()
    assert cols[-2:] == cols[:2]
    assert tuple(col[0] for col in cols[2:]) == (1, 0, 0, 0, 0, 0, 0)
    assert periodic_columns(90)[-2:] != periodic_columns(90)[:2]
    assert completion(0, 96, 'zero') == 0
    assert completion(1, 96, 'zero', 90) == 1
    assert completion(17, 28) >> 21 == 0
    return {'physical': {'right_hex': hex(r), 'N': 190, 'W': 26, 'deficit': 8,
                         'transient': actual, 'forward_simulation': 'PASS'},
            'L3_counterexample': {'physically_forbidden_factor': '0101000101', 'u': u, 'N': 191, 'W': 26, 'deficit': 9,
                                  'transient': killed, 'physical_realisability': 'unknown'},
            'conditional_exact7': {'time_modulus': 4, 'columns_minus1_through7': cols,
                                   'returned_adjacent_pair': True},
            'controls': {'rule90_pair_return': False, 'rule30_zero_completion': 0,
                         'rule90_zero_target_right1_completion': 1,
                         'r17_depth22_through28_all_zero': True},
            'm45_six_bit_quiet_tails': [dict(tail=w, **first_transient(
                ref.completion_from_u(tuple(map(int, '01' * 45 + w)), 192), 192))
                for w in ('000010', '000100', '000101', '001000', '001001', '001010')],
            'random_physical_crosschecks': {'count': 64, 'depth': 96, 'seed': 30715}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-file', type=Path)
    parser.add_argument('--max-m', type=int, default=48)
    parser.add_argument('--tail-length', type=int, default=12)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ref = reference_module(args.reference_file)
    result = {'result': 'PASS', 'verification': verify(ref),
              'scope': {'max_m': args.max_m, 'tail_length': args.tail_length,
                        'meaning': 'Exact finite words, not physical language enumeration'},
              'tail_table': tail_table(ref, args.max_m, args.tail_length),
              'not_run': ['full verifier', 'Lean', 'SAT/DRAT', 'physical reachability of L3 witness',
                          'uniform deficit theorem', 'infinite seed32 boundary proof']}
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(json.dumps({'result': result['result'], 'maxima': {
        key: val['maximum_uncensored'] for key, val in result['tail_table'].items()}}, indent=2))


if __name__ == '__main__':
    main()
