#!/usr/bin/env python3
"""Exact Hamming-weight search for the alternating left-half cone charge.

The unrestricted uniform conjecture stated below was subsequently refuted at
N=50.  See alternating_charge_counterexample.py.  This scanner still checks
the original finite horizons; physical right-boundary coherence is a separate
restriction implemented in alternating_right_coherence.py.

Under an alternating centre, let ``s_k`` be the right neighbour at even time
``2k``.  Sideways Rule 30 reconstruction maps the free bits ``s`` to a unique
initial left prefix ``e``.  The first ``N`` bits of ``e`` depend on only
``ceil(N/2)`` bits of ``s``, so all possibilities can be minimized exactly.

The proof target is the cone-charge inequality

    wt(e[0:N]) >= floor((N + 6) / 8).

At ``N=2m`` this says that ``m`` necessary alternating-latch events force at
least ``ceil(m/4)`` initial left ones.  If true for every ``m``, an eternal
alternating centre has an infinite-support initial left half.  Finite exhaustive
agreement is a proof target only; the missing object is a local telescoping
potential or equivalent induction.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.periodic_trace_fiber import (  # noqa: E402
    forced_left_prefix,
    step_eca,
)


def induced_left_prefix(s_value: int, macrosteps: int) -> np.ndarray:
    """Return the first ``2*macrosteps`` forced initial-left bits."""
    if not 1 <= macrosteps <= 32 or not 0 <= s_value < (1 << macrosteps):
        raise ValueError("invalid macrostep count or s value")
    s = ((s_value >> np.arange(macrosteps, dtype=np.uint64)) & 1).astype(np.uint8)
    even_previous = s
    even = np.zeros(macrosteps, dtype=np.uint8)  # centre: 0 at even times
    odd_previous = np.zeros(macrosteps, dtype=np.uint8)  # value is immaterial
    odd = np.ones(macrosteps, dtype=np.uint8)  # centre: 1 at odd times
    prefix = []
    for _ in range(2 * macrosteps):
        even_next = odd ^ (even | even_previous)
        shifted_even = np.empty_like(even)
        shifted_even[:-1] = even[1:]
        shifted_even[-1] = 0  # outside the dependency cone of e_1..e_2m
        odd_next = shifted_even ^ (odd | odd_previous)
        prefix.append(int(even_next[0]))
        even_previous, even = even, even_next
        odd_previous, odd = odd, odd_next
    return np.asarray(prefix, dtype=np.uint8)


def minimum_weight_profile(macrosteps: int) -> list[dict]:
    """Minimize every initial-left prefix over all ``2^macrosteps`` s words."""
    if not 1 <= macrosteps <= 22:
        raise ValueError("macrosteps must be in 1..22")
    count = 1 << macrosteps
    values = np.arange(count, dtype=np.uint32)
    s = ((values[:, None] >> np.arange(macrosteps, dtype=np.uint32)) & 1
         ).astype(np.uint8)
    even_previous = s
    even = np.zeros((count, macrosteps), dtype=np.uint8)
    odd_previous = np.zeros((count, macrosteps), dtype=np.uint8)
    odd = np.ones((count, macrosteps), dtype=np.uint8)
    weights = np.zeros(count, dtype=np.uint8)
    rows = []

    for offset in range(2 * macrosteps):
        even_next = odd ^ (even | even_previous)
        shifted_even = np.empty_like(even)
        shifted_even[:, :-1] = even[:, 1:]
        shifted_even[:, -1] = 0
        odd_next = shifted_even ^ (odd | odd_previous)
        weights += even_next[:, 0]

        length = offset + 1
        minimum = int(weights.min())
        used_s_bits = (length + 1) // 2
        extension_factor = 1 << (macrosteps - used_s_bits)
        extremizers = int(np.count_nonzero(weights == minimum) // extension_factor)
        example_full = int(np.flatnonzero(weights == minimum)[0])
        example_s = example_full & ((1 << used_s_bits) - 1)
        example_prefix = induced_left_prefix(example_s, used_s_bits)[:length]
        conjectured = (length + 6) // 8
        rows.append({
            "prefix_length": length,
            "free_s_bits": used_s_bits,
            "minimum_weight": minimum,
            "extremizer_prefixes": extremizers,
            "example_s_lsb_first": format(example_s, f"0{used_s_bits}b")[::-1],
            "example_one_depths": [
                j + 1 for j, bit in enumerate(example_prefix) if bit],
            "conjectured_lower_bound": conjectured,
            "cone_charge_survives": minimum >= conjectured,
            "minimum_8weight_minus_length": 8 * minimum - length,
        })
        even_previous, even = even, even_next
        odd_previous, odd = odd, odd_next
    return rows


def self_test() -> dict:
    # Cross-check the sideways recurrence against actual spacetime rows from
    # the independently forward-verified periodic-fibre constructor.
    macrosteps = 5
    rows, left, _, _ = forced_left_prefix(
        depth=2 * macrosteps, right_support=4, word="01")
    current = rows.copy()
    centre = rows.shape[1] // 2
    s_values = np.zeros(len(rows), dtype=np.uint32)
    for time in range(2 * macrosteps):
        if time % 2 == 0:
            s_values |= current[:, centre + 1].astype(np.uint32) << (time // 2)
        current = step_eca(current, 30)
    mismatches = 0
    for index, s_value in enumerate(s_values):
        mismatches += int(np.count_nonzero(
            induced_left_prefix(int(s_value), macrosteps) != left[index]))

    small = minimum_weight_profile(4)
    even_minima = [small[2 * m - 1]["minimum_weight"] for m in range(1, 5)]
    return {
        "sideways_vs_spacetime_bit_mismatches": mismatches,
        "small_even_prefix_minima": even_minima,
        "small_profile_matches": even_minima == [1, 1, 1, 2],
    }


def vertical_window_zero_distance(height: int) -> int | None:
    """Shortest relaxed spatial path from the alternating centre to (0,0).

    A state is a pair of adjacent temporal words of fixed height.  Sideways
    reconstruction fixes all but the newest top bit of the next word.  Reusing
    that forgotten bit at every spatial step is a sound over-approximation, but
    reaching ``(0,0)`` exposes a spurious zero-cost cycle and rules out this
    naive fixed vertical cut as a cone-charge proof.
    """
    if not 2 <= height <= 12:
        raise ValueError("height must be in 2..12")
    mask = (1 << height) - 1
    alternating = sum((time & 1) << time for time in range(height))
    starts = [right | (alternating << height) for right in range(1 << height)]
    seen = set(starts)
    queue = deque((state, 0) for state in starts)
    while queue:
        state, distance = queue.popleft()
        if state == 0:
            return distance
        first = state & mask
        second = state >> height
        next_low = ((second >> 1) ^ (second | first)) & ((1 << (height - 1)) - 1)
        for top in (0, 1):
            third = next_low | (top << (height - 1))
            successor = second | (third << height)
            if successor not in seen:
                seen.add(successor)
                queue.append((successor, distance + 1))
    return None


def run(macrosteps: int) -> dict:
    check = self_test()
    if check["sideways_vs_spacetime_bit_mismatches"] or not check["small_profile_matches"]:
        raise RuntimeError(f"self-test failed: {check}")
    profile = minimum_weight_profile(macrosteps)
    return {
        "experiment": "alternating-left-weight",
        "macrosteps": macrosteps,
        "initial_left_prefix_length": 2 * macrosteps,
        "free_boundary_words": 1 << macrosteps,
        "self_test": check,
        "profile": profile,
        "naive_vertical_window_relaxation": [
            {"height": height,
             "distance_to_spurious_all_zero_cycle": vertical_window_zero_distance(height)}
            for height in range(2, 11)
        ],
        "cone_charge_survives_all_prefixes": all(
            row["cone_charge_survives"] for row in profile),
        "claim_limit": (
            "Exact exhaustive minima over the stated finite prefix only. The "
            "cone-charge inequality remains a conjecture until a uniform local "
            "potential, induction, or other proof is supplied. The vertical-"
            "window result is a falsifier for that relaxation, not for slanted "
            "cut potentials or the cone-charge conjecture."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macrosteps", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.macrosteps)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
