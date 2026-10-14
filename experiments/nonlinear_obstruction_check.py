#!/usr/bin/env python3
"""Check the symbolic identities in docs/theory/nonlinear_obstruction.md.

Exhaustive means every 5-cell initial row, with zero exterior, not every
space-time diagram. Each of these 32 seeds is checked at every time 0..64.
Naive dynamics use Wolfram truth-table bits; polynomial identities use
independent GF(2) convolution and closed-form binomial/product kernels.
No numerical packages, external tools, or finite-ring boundaries are used.
"""

from __future__ import annotations

import json
from math import comb


MAX_TIME = 64
SEED_WIDTH = 5
OFFSET = 2 * MAX_TIME + SEED_WIDTH


def naive_step(row: set[int], rule: int) -> set[int]:
    """Unbounded integer lattice with a quiescent exterior."""
    if not row:
        return set()
    result = set()
    for i in range(min(row) - 1, max(row) + 2):
        neighborhood = (4 * (i - 1 in row) + 2 * (i in row)
                        + (i + 1 in row))
        if (rule >> neighborhood) & 1:
            result.add(i)
    return result


def encode(row: set[int]) -> int:
    return sum(1 << (OFFSET + i) for i in row)


def gf2_multiply(a: int, b: int) -> int:
    """Polynomial multiplication, with bit j the coefficient of z**j."""
    if a.bit_count() > b.bit_count():
        a, b = b, a
    result = 0
    while a:
        low = a & -a
        result ^= b << (low.bit_length() - 1)
        a ^= low
    return result


def kernel90(t: int) -> int:
    """z**t (z + z**-1)**t, expanded using actual binomial coefficients."""
    return sum(1 << (2 * k) for k in range(t + 1) if comb(t, k) & 1)


def kernel_product(t: int, with_center: bool) -> int:
    """z**t times the Laurent binary-time Frobenius product."""
    result = 1
    bit = 1
    remaining = t
    while remaining:
        if remaining & 1:
            factor = (1 << (2 * bit)) | 1
            if with_center:
                factor |= 1 << bit
            result = gf2_multiply(result, factor)
        bit <<= 1
        remaining >>= 1
    return result


def propagate(row: int, kernel: int, elapsed: int) -> int:
    product = gf2_multiply(row, kernel)
    # OFFSET is larger than every backwards light cone checked here.
    assert not (product & ((1 << elapsed) - 1)), "coordinate truncation"
    return product >> elapsed


def check_local_identities() -> int:
    cases = 0
    for neighborhood in range(8):
        left, center, right = ((neighborhood >> j) & 1 for j in (2, 1, 0))
        assert ((30 >> neighborhood) & 1) == (left ^ center ^ right
                                              ^ (center * right))
        assert ((150 >> neighborhood) & 1) == (left ^ center ^ right)
        assert ((90 >> neighborhood) & 1) == (left ^ right)
        cases += 3
    # q(a+b)+q(a)+q(b)=a_i*b_(i+1)+b_i*a_(i+1).
    for bits in range(16):
        a, ar, b, br = ((bits >> j) & 1 for j in range(4))
        assert ((a ^ b) * (ar ^ br)) ^ (a * ar) ^ (b * br) == (
            (a * br) ^ (b * ar))
        cases += 1
    return cases


def main() -> None:
    local_cases = check_local_identities()
    k90 = [kernel90(t) for t in range(MAX_TIME + 1)]
    k150 = [kernel_product(t, True) for t in range(MAX_TIME + 1)]
    for t in range(MAX_TIME + 1):
        assert k90[t] == kernel_product(t, False), ("Lucas", t)
    for exponent in range(7):
        t = 1 << exponent
        assert k90[t] == 1 | (1 << (2 * t)), ("90 Frobenius", t)
        assert k150[t] == 1 | (1 << t) | (1 << (2 * t)), (
            "150 Frobenius", t)

    rows_checked = 0
    for seed in range(1 << SEED_WIDTH):
        initial = {i - SEED_WIDTH // 2 for i in range(SEED_WIDTH)
                   if (seed >> i) & 1}
        initial_poly = encode(initial)
        rows = {rule: initial.copy() for rule in (30, 90, 150)}
        forcing: list[int] = []
        for t in range(MAX_TIME + 1):
            poly30 = encode(rows[30])
            predicted90 = propagate(initial_poly, k90[t], t)
            predicted150 = propagate(initial_poly, k150[t], t)
            assert encode(rows[90]) == predicted90, (seed, t, "90")
            assert encode(rows[150]) == predicted150, (seed, t, "150")
            duhamel = predicted150
            for s, q_s in enumerate(forcing):
                elapsed = t - 1 - s
                duhamel ^= propagate(q_s, k150[elapsed], elapsed)
            assert poly30 == duhamel, (seed, t, "Duhamel")
            # The homogeneous two-power formula therefore fails exactly by
            # the stated forcing remainder; this checks it for every t,
            # including all dyadic times in range.
            remainder = poly30 ^ predicted150
            assert remainder == (duhamel ^ predicted150)
            if seed == 0:
                assert poly30 == predicted90 == predicted150 == remainder == 0
            if t < MAX_TIME:
                q_s = {i for i in rows[30] if i + 1 in rows[30]}
                forcing.append(encode(q_s))
                for rule in rows:
                    rows[rule] = naive_step(rows[rule], rule)
            rows_checked += 1

    # Earliest centre obstruction on the actual prize seed, and explicit
    # counterexample to superposition, independently through truth tables.
    singleton = {0}
    one_step = naive_step(singleton, 30)
    assert one_step == {-1, 0, 1}
    q1 = {i for i in one_step if i + 1 in one_step}
    assert q1 == {-1, 0}
    two30 = naive_step(one_step, 30)
    two150 = naive_step(naive_step(singleton, 150), 150)
    assert two30 == {-2, -1, 2}
    assert two150 == {-2, 0, 2}
    assert two30 ^ two150 == q1
    superposition_defect = (naive_step({0, 1}, 30)
                            ^ naive_step({0}, 30)
                            ^ naive_step({1}, 30))
    assert superposition_defect == {0}
    assert (naive_step({0, 1}, 90) ^ naive_step({0}, 90)
            ^ naive_step({1}, 90)) == set()
    assert (naive_step({0, 1}, 150) ^ naive_step({0}, 150)
            ^ naive_step({1}, 150)) == set()
    # Rule90: remove the centre linear term AND the quadratic switch.
    # Rule150: remove only the quadratic switch. In each the Duhamel
    # remainder is identically absent, not an unmeasured statistical null.
    print(json.dumps({
        "result": "PASS",
        "initial_rows": 1 << SEED_WIDTH,
        "seed_width": SEED_WIDTH,
        "times": [0, MAX_TIME],
        "seed_time_pairs": rows_checked,
        "local_truth_table_cases": local_cases,
        "rules": [30, 90, 150],
        "boundary": "infinite zero exterior; no wraparound",
        "checks": ["ANF", "additivity defect", "Lucas kernel",
                   "Frobenius kernels", "additive closed forms",
                   "Rule30 Duhamel", "zero", "single-seed first correction"],
        "limitation": "Finite checks support the separate all-time algebraic proof; no complexity bound."
    }, indent=2))


if __name__ == "__main__":
    main()
