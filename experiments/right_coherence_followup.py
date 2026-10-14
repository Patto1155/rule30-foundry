#!/usr/bin/env python3
"""Exact robust right-boundary invariant; no all-finite entry claim.

The finite abstraction forgets outer predecessor coherence adversarially.
Closure under *all* its edges is therefore a sufficient invariant for actual
Rule-30 right halves, including infinite ones.  This is not a trace-language
characterization and does not exclude an alternating centre.
"""
from __future__ import annotations

import json
import numpy as np

WIDTH = 16
TRIGGERS = (167, 168, 170)


def packed_return(x):
    y = (x << 1) ^ (x | (x >> 1))
    return ((y << 1) | 1) ^ (y | (y >> 1))


def independent_return(x, width):
    """Shrinking cone using ECA truth-table lookups, vectorized over x."""
    cells = [(x >> j) & 1 for j in range(width + 2)]
    table = np.array([(30 >> i) & 1 for i in range(8)], dtype=np.uint64)
    for centre in (0, 1):
        new = []
        for j in range(len(cells) - 1):
            left = centre if j == 0 else cells[j - 1]
            neighbourhood = (left << 2) | (cells[j] << 1) | cells[j + 1]
            new.append(table[neighbourhood])
        cells = new
    result = np.zeros_like(x)
    for j, cell in enumerate(cells):
        result |= cell << j
    return result


def transition_table(width):
    n = 1 << width
    x = np.arange(4 * n, dtype=np.uint64)
    packed = packed_return(x) & (n - 1)
    direct = independent_return(x, width)
    if not np.array_equal(packed, direct):
        raise AssertionError("packed map disagrees with ECA cone")
    # columns are low-width states; rows are adversarial next outer bit pairs.
    return direct.astype(np.int64).reshape(4, n)


def safe_states(width):
    states = np.arange(1 << width)
    return (states % 2 == 0) | (states % 8 == 7)


def greatest_kernel(edges, safe):
    kernel = safe.copy()
    deletions = []
    while True:
        new = kernel & kernel[edges].all(axis=0)
        removed = int(kernel.sum() - new.sum())
        if removed == 0:
            return kernel, deletions
        deletions.append(removed)
        kernel = new


def forward_closure(edges, initial):
    reachable = initial.copy()
    rounds = 0
    while True:
        new = reachable.copy()
        new[edges[:, reachable].ravel()] = True
        if np.array_equal(new, reachable):
            return reachable, rounds
        reachable = new
        rounds += 1


def first_kernel_entry(value, kernel, limit=500):
    mask = len(kernel) - 1
    for k in range(limit + 1):
        if kernel[value & mask]:
            return k
        value = packed_return(value)
    return None  # Censored, never an assertion of eternal nonentry.


def verify_one_spacings():
    """Low-triple first return from 111: universally 3 or 5 macrosteps."""
    # Five macrosteps can inspect bits through index12 from low triple.
    gaps = set()
    for upper in range(1 << 10):
        value = 7 | (upper << 3)
        for gap in range(1, 6):
            value = packed_return(value)
            if value & 1:
                gaps.add(gap)
                break
        else:
            raise AssertionError("no one within five returns")
    if gaps != {3, 5}:
        raise AssertionError(gaps)
    return sorted(gaps)


def run():
    edges = transition_table(WIDTH)
    safe = safe_states(WIDTH)
    kernel, deletions = greatest_kernel(edges, safe)
    low8 = np.arange(1 << WIDTH) % 256
    initial = np.isin(low8, TRIGGERS)
    closure, rounds = forward_closure(edges, initial)
    assert int(kernel.sum()) == 13526
    assert int(closure.sum()) == 1452
    assert safe[closure].all()
    assert closure[edges[:, closure]].all()
    assert kernel[initial].all()
    assert kernel[edges[:, kernel]].all()
    return {
        "width": WIDTH,
        "independent_truth_table_rows": 1 << (WIDTH + 2),
        "kernel_size": int(kernel.sum()),
        "strict_kernel_deletion_rounds": len(deletions),
        "trigger_low8_integers": list(TRIGGERS),
        "trigger_nearest_cell_first": [format(v, "08b")[::-1] for v in TRIGGERS],
        "trigger_initial_states": int(initial.sum()),
        "trigger_forward_closure_size": int(closure.sum()),
        "strict_forward_expansion_rounds": rounds,
        "universal_closure_edges": 4 * int(closure.sum()),
        "future_one_spacings_macrosteps": verify_one_spacings(),
        "finite_right_example_first_kernel_entries": {
            str(v): first_kernel_entry(v, kernel)
            for v in (0, 1, 2, 3, 4, 17, 12345)
        },
        "scope": (
            "Every actual right trajectory entering K stays in K and has "
            "future one spacings 3 or 5. The three low8 triggers suffice for "
            "entry regardless of the entire remaining right half. No claim "
            "that every finite right half enters K; no left-growth bound."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
