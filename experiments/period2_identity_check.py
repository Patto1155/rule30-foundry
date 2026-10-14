#!/usr/bin/env python3
"""Small independent checks for the Condrey audit and period-two failure note.

No searched-class negative is reported. Exhaustive p=1 checks corroborate
analytical identities; they do not substitute for the unbounded proofs.
"""
from itertools import product
import json


RULE30 = (0, 1, 1, 1, 1, 0, 0, 0)
RULE90 = (0, 1, 0, 1, 1, 0, 1, 0)


def step_truth(word, rule=RULE30):
    """Position increases to the right; finite zero extension on both sides."""
    out = 0
    for i in range(word.bit_length() + 1):
        left = (word >> (i - 1)) & 1 if i else 0
        centre = (word >> i) & 1
        right = (word >> (i + 1)) & 1
        out |= rule[(left << 2) | (centre << 1) | right] << i
    return out


def center_trace(word, offset, length, rule=RULE30):
    values = []
    for _ in range(length):
        values.append((word >> offset) & 1)
        word = step_truth(word, rule)
    return values


def forced_left(right, target, depth):
    """Triangular inversion, with all exterior choices kept fixed."""
    offset = 2 * depth + 4
    word = (right << (offset + 1)) | (target[0] << offset)
    for t in range(1, depth + 1):
        observed = center_trace(word, offset, t + 1)[-1]
        if observed != target[t]:
            word ^= 1 << (offset - t)
    assert center_trace(word, offset, depth + 1) == target
    return [(word >> (offset - j)) & 1 for j in range(1, depth + 1)]


def zero_completion(right, depth):
    if not right:
        return [0] * depth
    first = (right & -right).bit_length()
    return [0 if j < first else 1 if j == first else j % 2
            for j in range(1, depth + 1)]


def check_fibers(depth=32, right_width=5):
    for right in range(1 << right_width):
        for centre in (0, 1):
            target = [centre] * (depth + 1)
            actual = forced_left(right, target, depth)
            expected = zero_completion(right, depth) if not centre else [
                int(j % 2 == 0) for j in range(1, depth + 1)]
            assert actual == expected, (right, centre, actual, expected)
            if not centre:
                for k in range((depth + 1) // 2):
                    # Condrey Eq.4, with positive coordinate j in bit j-1.
                    assert actual[2 * k] == int(bool(right & ((1 << (2 * k + 1)) - 1)))
                for k in range(1, depth // 2 + 1):
                    # Condrey Eq.5: first one is exactly at even coordinate.
                    expected_even = ((right >> (2 * k - 1)) & 1) and not (
                        right & ((1 << (2 * k - 1)) - 1))
                    assert actual[2 * k - 1] == expected_even
    return (1 << right_width) * 2


def check_horizons(max_radius=6):
    rows = []
    for radius in range(1, max_radius + 1):
        offset = 2 * radius + 5
        max_index = [-1, -1]
        counts = [0, 0]
        for initial in range(1, 1 << (2 * radius + 1)):
            centre = (initial >> radius) & 1
            trace = center_trace(initial << (offset - radius), offset, radius + 3)
            mismatch = next((t for t, value in enumerate(trace) if value != centre), None)
            assert mismatch is not None, (radius, initial)
            index = mismatch - 1
            if index > max_index[centre]:
                max_index[centre] = index
                counts[centre] = 1
            elif index == max_index[centre]:
                counts[centre] += 1
        assert max_index == [2 * ((radius + 1) // 2), 2 * (radius // 2) + 1]
        assert counts == [(1 << radius) - 1, 1 << radius]
        rows.append({'radius': radius, 'horizon_indices': max_index,
                     'extremizer_counts': counts})
    return rows


def check_two_step():
    # A complete radius-two light cone checks the exact forced-right return.
    for s, u, v, w, z in product((0, 1), repeat=5):
        offset = 8
        right = s | (u << 1) | (v << 2) | (w << 3) | (z << 4)
        row = right << (offset + 1)  # centre=0 at even time
        after_one = step_truth(row)
        after_one |= 1 << offset  # prescribe next centre=1, not a full orbit
        after_two = step_truth(after_one)
        assert ((after_two >> (offset + 1)) & 1) == (1 - s) * (1 - u) * (1 - v)
    # Left return G=T_1 o T_0, compared with independent cell truth table.
    for n in range(1 << 6):
        offset = 12
        row = sum(((n >> (j - 1)) & 1) << (offset - j) for j in range(1, 7))
        after_one = step_truth(row) | (1 << offset)
        after_two = step_truth(after_one)
        def left_return(a, b):
            return (a >> 1) ^ (a | (a << 1) | b)
        returned = left_return(left_return(n, 0), 1)
        low_six = sum(((after_two >> (offset - j)) & 1) << (j - 1) for j in range(1, 7))
        assert low_six == returned & 63
        # Centre constraints with a free physical right bit: exactly residues1,2.
        odd_left = ((after_one >> (offset - 1)) & 1)
        assert (odd_left == 1) == (n % 4 in (1, 2))
    return {'right_cones': 32, 'left_prefixes': 64}


def main():
    fibers = check_fibers()
    horizons = check_horizons()
    local = check_two_step()
    target = [t % 2 for t in range(17)]
    e1 = forced_left(1, target, 16)
    e7 = forced_left(7, target, 16)
    first = next(j for j, pair in enumerate(zip(e1, e7), 1) if pair[0] != pair[1])
    assert first == 9
    # The p=2 first-one statistic is insufficient; no exclusion is inferred.
    assert e1[:8] == e7[:8] == [0, 1, 1, 0, 1, 0, 1, 0]
    assert (e1[8], e7[8]) == (0, 1)
    offset = 32
    rule90_single = center_trace(1 << offset, offset, 16, RULE90)
    rule90_pair = center_trace((1 << (offset - 1)) | (1 << (offset + 1)), offset, 16, RULE90)
    assert rule90_single == [1] + [0] * 15
    assert rule90_pair == [0] * 16
    assert center_trace(0, offset, 16) == [0] * 16
    print(json.dumps({'result': 'PASS', 'constant_fibers_checked': fibers,
                      'fiber_depth': 32, 'horizon_checks': horizons, 'local_checks': local,
                      'period2_first_one_counterexample': {
                          'right_integers': [1, 7], 'first_divergent_left_depth': first,
                          'left_prefixes_16': [''.join(map(str, e1)), ''.join(map(str, e7))]},
                      'rule90_and_zero_controls': 'PASS',
                      'claim_scope': 'identity checks and external p1 audit, not p2 exclusion'}, indent=2))


if __name__ == '__main__':
    main()
