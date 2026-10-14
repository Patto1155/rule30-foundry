#!/usr/bin/env python3
"""Replay recovered exact certificates; no new universal support bound."""
from __future__ import annotations
import argparse
import json
from math import comb, log2
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import alternating_renewal as renewal
from experiments import alternating_right_coherence as coherence
from experiments import right_coherence_followup as invariant
from experiments import right_invariant_entry as entry
from experiments import weighted_renewal_transport as transport
from experiments.alternating_left_transfer import macro_step, survivor_condition


def lifetime(n):
    for q in range(100):
        if not survivor_condition(n):
            return q
        n = macro_step(n)
    raise AssertionError("bounded witness unexpectedly survived 100 latches")


def run():
    # Count the actual tested sparse-prefix class before any empty-layer check.
    counting = []
    for r,b in enumerate((0,3,8,11,16,19)):
        n = 2*(b+1)
        size = sum(comb(n,j) for j in range(r+1))
        counting.append({'weight_cap':r, 'cone_bits':n, 'class_size':size,
                         'log2_class_size':log2(size), 'target_bits_after_time0':n,
                         'purpose':'exact-exclusion',
                         'statistical_interpretation':'not complexity evidence; necessary left-only constraints allow free right bits'})
    bounds = [renewal.check_bound(r, b) for r,b in enumerate((0,3,8,11,16,19))]
    assert all(x['sharp'] for x in bounds)
    # Controls concern necessary left-only constraints, not full alternating rows.
    assert renewal.admits_latches(0, 12, rule=90)
    assert renewal.admits_latches(0, 12, rule=15)
    assert not renewal.admits_latches(0, 1, rule=30)
    assert lifetime(129) == 8
    assert lifetime(1075855489) == 19
    assert [lifetime(n) for n in (16417,16513)] == [2,10]
    charge_kills = []
    for n,q in ((4329177217,26), (1158551008470237313,40)):
        assert lifetime(n) == q
        assert renewal.admits_latches(n,q) and not renewal.admits_latches(n,q+1)
        charge_kills.append({'left':n, 'weight':n.bit_count(), 'lifetime':q,
                             'charge_at_2q':8*n.bit_count()-2*q})
    weighted = transport.verify(transport.fixed_certificate())
    weighted['nonzero_density_drifts_count'] = len(weighted.pop('nonzero_density_drifts'))
    assert weighted['exact_rows'] == 45
    assert weighted['nonzero_density_drifts_count'] == 41
    # Exact bounded classification of the support agent's slow row r=32.
    edges = invariant.transition_table(16)
    kernel, _ = invariant.greatest_kernel(edges, invariant.safe_states(16))
    cyclic, _, _ = entry.components(edges, ~kernel)
    exceptional = set(map(int, cyclic[-1]))
    right = 32
    seed32_states = []
    for k in range(64):
        low = right & 65535
        assert low in exceptional and not kernel[low]
        assert low & 1 == k & 1
        seed32_states.append(low)
        following = int(invariant.packed_return(right))
        direct = invariant.independent_return(np.array([right & ((1<<18)-1)],dtype=np.uint64),16)
        assert following & 65535 == int(direct[0])
        right = following
    return {
        'source_ref': '44d11eebaab4e8486c195dee1e5c61a36860851f',
        'purpose': 'exact-exclusion and exact-certificate-replay; not statistical complexity',
        'counting_before_sparse_replay': counting,
        'fixed_weight_bounds': bounds,
        'controls': {'rule90_zero_left_survives_12': True,
                     'rule15_zero_left_survives_12': True,
                     'rule30_zero_left_fails_first_latch': True},
        'spatial_deletion_counterexamples': {'weight2_lifetime':8,
                                            'weight5_lifetime':19,
                                            'weight3_deletion_lifetimes':[2,10]},
        'unrestricted_charge_counterexamples': charge_kills,
        'right_boundary_constraints': coherence.run(),
        'right_invariant': invariant.run(),
        'right_entry': entry.run(),
        'weighted_local_credit_obstruction': weighted,
        'seed32_bounded_membership': {'macrosteps':64, 'all_in_Q':True,
                                      'all_outside_K':True, 'output':'01'*32,
                                      'first14_low16_states':seed32_states[:14],
                                      'scope':'bounded physical forced-right prefix; no asymptotic assertion'},
        'reported_740_core': 'not reproduced; source absent from fetched refs',
        'scope': 'Exact finite certificates and analytic implications. No support-growth theorem.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end='')

if __name__ == '__main__':
    main()
