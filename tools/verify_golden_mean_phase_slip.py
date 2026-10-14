#!/usr/bin/env python3
"""Truth-table audit of the seven-column identity and physical reset failure.

This checks the finite Boolean identities underlying the handwritten uniform
telescoping argument.  It does not certify the missing spatial transport lemma.
"""
from __future__ import annotations

import json


def cell(left: int, centre: int, right: int) -> int:
    return (30 >> (4 * left + 2 * centre + right)) & 1


def inverse_left(future: int, centre: int, right: int) -> int:
    choices = [left for left in (0, 1)
               if cell(left, centre, right) == future]
    if len(choices) != 1:
        raise AssertionError("Rule 30 is not left permutive")
    return choices[0]


def columns(s: list[int], depth: int) -> list[tuple[list[int], list[int]]]:
    """Reconstruct columns using only the local truth-table inverse."""
    ep, even = s[:], [0] * len(s)
    op, odd = [0] * len(s), [1] * len(s)
    result = []
    for _ in range(depth):
        en = [inverse_left(odd[k], even[k], ep[k]) for k in range(len(s))]
        on = [inverse_left(even[k + 1] if k + 1 < len(s) else 0,
                           odd[k], op[k]) for k in range(len(s))]
        result.append((en, on))
        ep, even, op, odd = even, en, odd, on
    return result


def verify_identities() -> dict:
    words = 0
    for value in range(1 << 8):
        if value & (value >> 1):
            continue
        words += 1
        s = [(value >> j) & 1 for j in range(8)]
        reconstructed = columns(s, 8)
        for k in range(3):
            a, b, c, d = s[k:k + 4]
            expected = [
                (1-a, 1), (a, b), (1-b, 1-b), (0, c),
                (1-b-c, 1-b), (c, b ^ c ^ d),
                (1-c-d, 1-d+b*d),
            ]
            for j, pair in enumerate(expected):
                actual = (reconstructed[j][0][k], reconstructed[j][1][k])
                if actual != pair:
                    raise AssertionError(f"column formula failed: {value}, {k}, {j}")
            if reconstructed[7][0][k] != b*d:
                raise AssertionError("depth-eight even formula failed")
            delta = sum(expected[6]) - 1
            d1, d2 = int(b == c == 0), int(c == d == 0)
            if delta != d2 - d1 + d1*d2:
                raise AssertionError("coboundary/production identity failed")
        first_weight = sum(pair[0][0] for pair in reconstructed[:7])
        if first_weight != 4 - 2*s[1] - s[2] - s[3] or first_weight < 1:
            raise AssertionError("first-slab weight identity failed")

    # This is the exact obstruction to propagating temporal excess alone.
    zero_columns = columns([0] * 12, 14)
    if (zero_columns[6][0][0], zero_columns[6][1][0]) != (1, 1):
        raise AssertionError("zero-word first-cut excess changed")
    if (zero_columns[13][0][0], zero_columns[13][1][0]) != (0, 0):
        raise AssertionError("zero-word second-cut deficit changed")
    return {"golden_mean_eight_bit_words": words,
            "depth_formulas": "verified",
            "identity": "Delta_k=d_(k+2)-d_(k+1)+d_(k+1)*d_(k+2)",
            "spatial_transport_proved": False}


def verify_physical_reset_failure(left: int = 857638, right: int = 17) -> dict:
    horizon = 28
    bits = [(left >> j) & 1 for j in range(horizon)]
    if any(bits[21:28]) or not any(bits[:21]):
        raise AssertionError("specified aligned zero block is absent")
    row = [((left >> (-x-1)) & 1) if x < 0 else
           ((right >> (x-1)) & 1) if x > 0 else 0
           for x in range(-horizon, horizon + 1)]
    trace, boundary = [], []
    for time in range(horizon + 1):
        centre = horizon-time
        trace.append(row[centre])
        if time < horizon and time % 2 == 0:
            boundary.append(row[centre + 1])
        row = [cell(row[j-1], row[j], row[j+1])
               for j in range(1, len(row)-1)]
    if trace != [t % 2 for t in range(horizon + 1)]:
        raise AssertionError("finite full row does not realize the claimed trace")
    return {"left": left, "right": right,
            "initial_left_low_first": "".join(map(str, bits)),
            "zero_block_indices": [21, 27],
            "centre_trace": "".join(map(str, trace)),
            "right_even_times_0_through_26": "".join(map(str, boundary)),
            "scope": "A finite physical prefix; not eventual alternation."}


def verify() -> dict:
    return {"verified": True, "identities": verify_identities(),
            "physical_reset_failure": verify_physical_reset_failure()}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
