#!/usr/bin/env python3
"""Audit the periodic-even-right-word zero-gap lemma for Rule 30.

Finite identities and examples supplement the handwritten all-period argument
in docs/theory/weak-weight-growth-followup.md. This is not period-two exclusion.
The independent ECA truth table is used throughout; no packed transfer map.
"""
from __future__ import annotations

import argparse
import json


def cell(left: int, centre: int, right: int) -> int:
    return (30 >> (4 * left + 2 * centre + right)) & 1


def inverse_left(future: int, centre: int, right: int) -> int:
    choices = [bit for bit in (0, 1) if cell(bit, centre, right) == future]
    if len(choices) != 1:
        raise AssertionError("local rule is not left permutive")
    return choices[0]


def periodic_columns(word: tuple[int, ...], depth: int) -> list[tuple[int, ...]]:
    """Columns at depths -1 (right), 0 (centre), then 1..depth (left)."""
    period = 2 * len(word)
    right = tuple(word[t // 2] if t % 2 == 0 else 0 for t in range(period))
    centre = tuple(t % 2 for t in range(period))
    columns = [right, centre]
    for _ in range(depth):
        previous, current = columns[-2:]
        columns.append(tuple(inverse_left(current[(t + 1) % period],
                                          current[t], previous[t])
                             for t in range(period)))
    return columns


def check_word(word: tuple[int, ...], depth: int) -> dict:
    columns = periodic_columns(word, depth)
    period = 2 * len(word)
    initial = [column[0] for column in columns[2:]]
    gap = maximum_gap = 0
    for bit in initial:
        gap = 0 if bit else gap + 1
        maximum_gap = max(gap, maximum_gap)
    if maximum_gap >= 2 * period:
        raise AssertionError("periodic boundary produced forbidden zero block")
    if sum(initial) < depth // (2 * period):
        raise AssertionError("periodic boundary density bound failed")

    # Forward evolution of a shrinking interior cone, separate from the
    # inverse construction, checks all spatial/temporal coordinate choices.
    row = [column[0] for column in columns[::-1]]
    first_depth = depth
    for time in range(1, period + 1):
        row = [cell(row[i-1], row[i], row[i+1])
               for i in range(1, len(row)-1)]
        first_depth -= 1
        expected = [columns[first_depth-i+1][time % period]
                    for i in range(len(row))]
        if row != expected:
            raise AssertionError("forward cone disagrees with periodic columns")

    # Two different periodic continuations of an identical M-bit boundary
    # prefix must reconstruct the same first 2M initial-left cells.
    prefix_length = len(word)
    extension = word + tuple(1-bit for bit in word) + (0, 1)
    other = periodic_columns(extension, 2 * prefix_length)
    if initial[:2 * prefix_length] != [column[0] for column in other[2:]]:
        raise AssertionError("claimed boundary-prefix locality failed")
    return {"period_parameter": len(word), "initial_weight": sum(initial),
            "maximum_zero_run": maximum_gap}


def verify(max_period: int = 7) -> dict:
    if not 1 <= max_period <= 10:
        raise ValueError("max_period must be in 1..10")
    rows = []
    total = 0
    for p in range(1, max_period + 1):
        depth = 32 * p
        results = []
        for value in range(1 << p):
            word = tuple((value >> i) & 1 for i in range(p))
            results.append(check_word(word, depth))
            total += 1
        rows.append({"period_parameter": p, "words": 1 << p,
                     "initial_depth": depth,
                     "largest_observed_zero_run": max(r["maximum_zero_run"] for r in results),
                     "proved_forbidden_zero_run_length": 4 * p})
    return {"verified": True, "words_checked": total, "rows": rows,
            "scope": "Finite identity/cone audit; all-period proof is in the accompanying note.",
            "period_two_exclusion": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-period", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(verify(args.max_period), indent=2))


if __name__ == "__main__":
    main()
