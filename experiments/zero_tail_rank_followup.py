#!/usr/bin/env python3
"""Exact audit of the finite-range additive zero-tail rank obstruction.

No survival scan, floating-point LP, or period-two exclusion. The all-range
bulk-conservation implication is handwritten in the accompanying note; the
finite-ring rank computation proves the classification for ranges 1..8.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

PRIME = 65521


def cell(left: int, centre: int, right: int) -> int:
    return (30 >> (4 * left + 2 * centre + right)) & 1


def ring_step(word: int, length: int) -> int:
    """Bit index increases toward the physical left, as in left integers."""
    mask = (1 << length) - 1
    left = (word >> 1) | ((word & 1) << (length - 1))
    right = ((word << 1) & mask) | (word >> (length - 1))
    return left ^ (word | right)


def ring_step_reference(word: int, length: int) -> int:
    return sum(cell((word >> ((i + 1) % length)) & 1,
                    (word >> i) & 1,
                    (word >> ((i - 1) % length)) & 1) << i
               for i in range(length))


def block_counts(word: int, length: int, width: int) -> np.ndarray:
    counts = np.zeros(1 << width, dtype=np.int64)
    for i in range(length):
        block = sum(((word >> ((i + j) % length)) & 1) << j
                    for j in range(width))
        counts[block] += 1
    return counts


def classify(width: int) -> dict:
    dimension = 1 << width
    # Constants plus gradients on the binary de Bruijn graph have this
    # dimension: one constant + (2**(width-1)-1) independent gradients.
    trivial_dimension = 1 << (width - 1)
    target_rank = dimension - trivial_dimension
    basis: dict[int, np.ndarray] = {}
    rows_checked = 0
    stopping_ring = None
    gradient_mask = (1 << (width - 1)) - 1
    for length in range(1, width + 5):
        for word in range(1 << length):
            first = ring_step(word, length)
            second = ring_step(first, length)
            assert first == ring_step_reference(word, length)
            assert second == ring_step_reference(first, length)
            row = block_counts(word, length, width) - block_counts(second, length, width)
            # Each conservation equation annihilates constants and every
            # vertex-gradient density, by explicit integer calculation.
            assert int(row.sum()) == 0
            divergence = [0] * trivial_dimension
            for block, coefficient in enumerate(row):
                divergence[block & gradient_mask] += int(coefficient)
                divergence[block >> 1] -= int(coefficient)
            assert not any(divergence)
            rows_checked += 1
            reduced = row % PRIME
            for pivot, vector in basis.items():
                coefficient = int(reduced[pivot])
                if coefficient:
                    reduced = (reduced - coefficient * vector) % PRIME
            nonzero = np.flatnonzero(reduced)
            if len(nonzero):
                pivot = int(nonzero[0])
                inverse = pow(int(reduced[pivot]), -1, PRIME)
                basis[pivot] = (reduced * inverse) % PRIME
            if len(basis) == target_rank:
                stopping_ring = {"length": length, "word": word}
                break
        if stopping_ring is not None:
            break
    assert len(basis) == target_rank, "classification not established at requested range"
    return {"range": width, "density_variables": dimension,
            "rank_mod_prime": len(basis),
            "rational_kernel_dimension": trivial_dimension,
            "kernel": "constants plus spatial coboundaries",
            "ring_equations_checked": rows_checked,
            "last_ring": stopping_ring}


def transfer(word: int, boundary: int) -> int:
    return (word >> 1) ^ (word | (word << 1) | boundary)


def macrostep(word: int) -> int:
    return transfer(transfer(word, 0), 1)


def verify_ether() -> dict:
    row = [int(i == 6) for i in range(7)]
    initial = row[:]
    centre = []
    for _ in range(4):
        centre.append(row[0])
        row = [cell(row[(i - 1) % 7], row[i], row[(i + 1) % 7])
               for i in range(7)]
    assert centre == [0, 1, 0, 1] and row == initial
    # Distinct from the periodic-ring audit: forward full finite rows check
    # arbitrarily long-prefix family coordinates and finite-left length gain.
    for repetitions in (2, 3, 5, 8):
        ones = {-(7*j + 1) for j in range(repetitions)}
        ones |= {7*j + 6 for j in range(repetitions)}
        left = sum(1 << (7*j) for j in range(repetitions))
        first_missing_distance = 7 * repetitions + 1
        for time in range(first_missing_distance + 1):
            expected = time & 1
            assert int(0 in ones) == (expected if time < first_missing_distance else 1-expected)
            if time < first_missing_distance and time % 2 == 0:
                actual_left = sum(1 << (-x-1) for x in ones if x < 0)
                assert left == actual_left
                next_left = macrostep(left)
                assert next_left.bit_length() == left.bit_length() + 2
                left = next_left
            low, high = min(ones)-1, max(ones)+1
            ones = {x for x in range(low, high+1)
                    if cell(int(x-1 in ones), int(x in ones), int(x+1 in ones))}
    return {"spatial_period": 7, "temporal_period": 4,
            "centre_period": 2, "finite_rows_checked": 4}


def centre_after(left: int, right: int, time: int) -> int:
    row = [((left >> (-x-1)) & 1) if x < 0 else
           ((right >> (x-1)) & 1) if x > 0 else 0
           for x in range(-time, time+1)]
    for _ in range(time):
        row = [cell(row[i-1], row[i], row[i+1])
               for i in range(1, len(row)-1)]
    assert len(row) == 1
    return row[0]


def verify_finite_prefix_realisation() -> dict:
    # 167 is a proved entry cylinder for the separate right invariant.
    # This check uses only truth-table cone inversion, not that invariant's
    # graph or any assumption about eventual survival.
    right = 167
    witnesses = []
    for horizon in (8, 16, 24):
        left = 0
        for time in range(1, horizon+1):
            if centre_after(left, right, time) != time % 2:
                left |= 1 << (time-1)
            assert centre_after(left, right, time) == time % 2
        assert all(centre_after(left, right, time) == time % 2
                   for time in range(horizon+1))
        witnesses.append({"halfstep_horizon": horizon, "left": left,
                          "right": right})
    return {"construction": "successive extreme-left-input inversion",
            "witnesses": witnesses}


def verify(max_range: int = 8) -> dict:
    if not 1 <= max_range <= 8:
        raise ValueError("certified scope is max_range in 1..8")
    assert all(PRIME % divisor for divisor in range(2, int(PRIME**0.5)+1))
    return {"verified": True, "prime": PRIME,
            "classifications": [classify(width) for width in range(1, max_range+1)],
            "ether": verify_ether(),
            "conditional_prefix_realisation": verify_finite_prefix_realisation(),
            "scope": "Exact finite-range rank obstruction; no all-range classification or period-two exclusion.",
            "period_two_exclusion": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-range", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(verify(args.max_range), indent=2))


if __name__ == "__main__":
    main()
