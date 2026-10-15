#!/usr/bin/env python3
"""Reference phase-zero column reduction (finite exact computations only).

Position +1 is bit zero of r; bit j-1 of every completion is x[-j,0].
No optimisation minimum is established by the checkpoint witnesses.
"""
import argparse
import json
import random


def column1(r, T):
    """Return x[1,t] for 0 <= t <= T, centre externally fixed to t % 2."""
    if r < 0 or T < 0:
        raise ValueError('nonnegative right integer and horizon required')
    result = []
    for t in range(T + 1):
        result.append(r & 1)
        r = ((r << 1) | (t & 1)) ^ (r | (r >> 1))
    return tuple(result)


def completion_from_column1(d, N):
    """Triangular inversion using d[t]=x[1,t], t<N (arbitrary input allowed)."""
    if N < 0:
        raise ValueError('nonnegative depth required')
    d = tuple(d)
    if len(d) < N or any(x not in (0, 1) for x in d):
        raise ValueError('need at least N binary temporal entries')
    right = sum(int(b) << t for t, b in enumerate(d[:N]))
    centre = sum(1 << t for t in range(1, N + 1, 2))
    result = 0
    mask = (1 << N) - 1
    for j in range(N):
        left = ((centre >> 1) ^ (centre | right)) & mask
        result |= (left & 1) << j
        right, centre = centre, left
        mask >>= 1
    return result


def completion_from_u(u, N):
    """Return packed e_1..e_N from chronological even column bits u_s.

    Supply at least ceil(N/2) binary bits. Odd right-column bits are erased
    by the first inversion; zero is a convenient arbitrary representative.
    """
    if N < 0:
        raise ValueError('nonnegative depth required')
    u = tuple(u)
    if len(u) < (N + 1) // 2 or any(x not in (0, 1) for x in u):
        raise ValueError('need ceil(N/2) binary even-time entries')
    d = [u[t // 2] if t % 2 == 0 else 0 for t in range(N)]
    return completion_from_column1(d, N)


def completion(r, N):
    """Packed initial negative half forcing the alternating centre through N."""
    if r < 0 or N < 0:
        raise ValueError('nonnegative right integer and depth required')
    return completion_from_u(column1(r, max(0, N - 1))[::2], N)


def forward_check(r, left, N):
    """Independent dictionary/truth-table full physical CA simulation."""
    table = (0, 1, 1, 1, 1, 0, 0, 0)
    row = {0: 0}
    row.update({i + 1: (r >> i) & 1 for i in range(r.bit_length())})
    row.update({-j: (left >> (j - 1)) & 1 for j in range(1, N + 1)})
    for t in range(N + 1):
        assert row.get(0, 0) == t % 2, (r, N, t)
        if t == N:
            break
        row = {i: table[4 * row.get(i - 1, 0) + 2 * row.get(i, 0) + row.get(i + 1, 0)]
               for i in range(min(row) - 1, max(row) + 2)}
    return True


def verify(exhaustive_bits=0, horizon=80, random_rows=200):
    try:
        from .support_growth import completion as old_completion
    except ImportError:
        from support_growth import completion as old_completion
    free = tuple(map(int, '01010100110101101101010010010001100100010'))
    l1 = tuple(map(int, '01' * 17 + '0' * 6))
    assert completion_from_u(free, 81).bit_count() == 9
    assert completion_from_u(l1, 79).bit_count() == 10
    r = 0x98f9d8a0
    physical = completion(r, 190)
    assert physical.bit_count() == 26 and 190 - 7 * physical.bit_count() == 8
    assert physical == old_completion(r, 190)
    forward_check(r, physical, 190)
    rng = random.Random(30)
    for _ in range(random_rows):
        r = rng.getrandbits(128)
        d = column1(r, horizon)
        randomized = [bit if t % 2 == 0 else rng.randrange(2) for t, bit in enumerate(d)]
        actual = completion(r, horizon)
        assert actual == completion_from_column1(randomized, horizon)
        assert actual == old_completion(r, horizon)
        forward_check(r, actual, horizon)
    # Counterexample to the erroneous *right* odd-column assertion: r=0.
    assert column1(0, 1)[1] == 0
    for r in range(1 << exhaustive_bits):
        u = column1(r, horizon)[::2]
        text = ''.join(map(str, u))
        assert '11' not in text and '00000' not in text, (r, text)
    return {'result': 'PASS', 'checkpoint_weights_witness_only': {'free_N81': 9, 'L1_N79': 10},
            'physical': {'right': '0x98f9d8a0', 'N': 190, 'W': 26, 'deficit': 8},
            'odd_erasure_random_rows': random_rows, 'random_horizon': horizon,
            'no11_no00000_right_integers': 1 << exhaustive_bits,
            'constraint_time_scope': [0, 2 * (horizon // 2)],
            'minimum_claims': 'none; witnesses alone do not prove minima'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--exhaustive-bits', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=80)
    parser.add_argument('--random-rows', type=int, default=200)
    args = parser.parse_args()
    if not args.verify:
        parser.error('choose --verify (optionally --exhaustive-bits 16)')
    print(json.dumps(verify(args.exhaustive_bits, args.horizon, args.random_rows), indent=2))


if __name__ == '__main__':
    main()
