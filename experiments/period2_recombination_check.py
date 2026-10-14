#!/usr/bin/env python3
"""Exact focused recombination checks; passing prefixes are not proofs."""
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from verify_golden_mean_phase_slip import verify as phase_slip_verify
from support_growth import completion


def prefix_check(r, depth):
    word = completion(r, depth)
    weight = 0
    h = r.bit_length()
    minimum = None
    for j in range(1, depth + 1):
        weight += (word >> (j - 1)) & 1
        slack = 7 * weight - j + 7 * h
        minimum = slack if minimum is None else min(minimum, slack)
        if slack < 0:
            return {'right': str(r), 'N': j, 'weight': weight, 'h': h}
    return minimum


def main():
    old = phase_slip_verify()
    smallest = None
    for r in range(1 << 16):
        out = prefix_check(r, 256)
        if isinstance(out, dict):
            raise AssertionError(out)
        smallest = out if smallest is None else min(smallest, out)
    rng = random.Random(307)
    sampled = [rng.getrandbits(96) for _ in range(128)]
    for r in sampled:
        out = prefix_check(r, 4096)
        if isinstance(out, dict):
            raise AssertionError(out)
    # The physical seven-zero block kills aligned seven-cell renewal reset.
    assert completion(17, 28) == 857638
    assert (completion(17, 28) >> 21) & 127 == 0
    # Confirm the separating controls with the same instruments.
    assert completion(0, 256, 'zero') == 0
    assert completion(1, 256, 'zero', 90) == 1
    rule90_weight = completion(0, 256, rule=90).bit_count()
    # Prefix7 itself survives this Rule90 case: do not invent a separating
    # control. The base32 comparison separates only before replacing its
    # baseline by the Rule30-specific period7 formula.
    rule90_base32 = completion(32, 256, rule=90).bit_count()
    assert rule90_weight < rule90_base32
    result = {
        'result': 'PASS', 'recovered_phase_slip_replay': old,
        'prefix7_scope': {'enumerated_seeds': 65536, 'all_prefixes_through': 256,
                         'sampled_96bit_seeds': 128, 'sample_all_prefixes_through': 4096,
                         'random_seed': 307, 'minimum_enumerated_slack': smallest},
        'rule90_control': {'right': 0, 'N': 256, 'weight': rule90_weight,
                          'base32_weight': rule90_base32,
                          'prefix7_separates': False, 'base32_comparison_separates': True},
        'zero_control': 'Rule30 zero target/right zero has completion zero',
        'meaning': 'Exact bounded kill tests only; no infinite density or support theorem.',
        'not_run': ['infinite physical right-alt32 proof', 'full verifier', 'Lean',
                    'SAT/DRAT', 'all65536 seeds beyond256', 'external-lineage audit']}
    path = ROOT / 'runs/period2-recombination-2026-10-14/results.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
