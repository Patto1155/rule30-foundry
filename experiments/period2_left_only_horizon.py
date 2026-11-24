#!/usr/bin/env python
"""Left-half-only horizon for an alternating centre (Prize 1, period two).

The left half-line evolves autonomously once the centre column is prescribed.
An alternating centre 0,1,0,1,... requires the nearest left cell to be 1 at
every odd time, i.e. L1 != L2 at every even time - a condition on the left
half alone. For every left half of exact radius R this prints the largest
number K of double-steps survived. If K_max(R) were finite for every R,
period two would be excluded with no reference to the right half.

Result (2026-11-24): K_max tracks R with O(1) scatter, which is what 2^R
independent geometric lifetimes give. No extremal structure to prove from.
Usage: python experiments/period2_left_only_horizon.py [max_radius]
"""

import random
import sys


def T(n: int, b: int) -> int:
    """One Rule 30 step of the left half (bit j-1 = depth j), centre bit b."""
    return (n >> 1) ^ (n | (n << 1) | b)


def survive(n: int, cap: int = 10**6) -> int:
    """Double-steps for which the left half is compatible with centre 0,1,0,1."""
    k = 0
    while k < cap:
        if (n & 3) not in (1, 2):
            return k
        n = T(T(n, 0), 1)
        k += 1
    return k


def self_check() -> None:
    """T against a direct cell-by-cell Rule 30 with a driven centre."""
    rng = random.Random(1)
    for _ in range(200):
        n = rng.getrandbits(rng.randint(1, 12))
        cells = {j: (n >> (j - 1)) & 1 for j in range(1, 60)}
        m = n
        for t in range(10):
            b = t & 1
            cells = {j: cells.get(j + 1, 0) ^ (cells[j] | (cells[j - 1] if j > 1 else b))
                     for j in range(1, 50)}
            m = T(m, b)
            assert all(((m >> (j - 1)) & 1) == cells[j] for j in range(1, 40))


def main() -> int:
    self_check()
    max_radius = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    for R in range(1, max_radius + 1):
        best, arg = 0, []
        for n in range(1 << (R - 1), 1 << R):
            k = survive(n)
            if k > best:
                best, arg = k, [n]
            elif k == best:
                arg.append(n)
        print(R, best, len(arg), [bin(a)[2:][::-1] for a in arg[:4]], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
