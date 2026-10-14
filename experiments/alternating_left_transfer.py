#!/usr/bin/env python3
"""Exact left-half transfer reduction for an alternating Rule 30 centre.

Encode the left half at an even (centre-zero) phase as a nonnegative integer
``n`` whose bit ``j`` is the cell at coordinate ``-j-1``.  If the prescribed
centre boundary is ``b``, one Rule 30 step of that half-line is exactly

    T_b(n) = (n >> 1) XOR (n OR ((n << 1) OR b)).

For the boundary word ``01``, the two-step map is ``G = T_1 o T_0``.  At the
intervening centre-one phase the neighbour at -1 must equal one, otherwise the
next centre bit cannot be zero.  That condition is equivalent to the low two
bits of ``G^k(n)`` being different.  Thus an eternal alternating centre from a
finite configuration would require an ordinary integer orbit satisfying

    G^k(n) mod 4 in {1, 2}  for every k >= 0.

The reduction is exact as a necessary condition and removes the initial right
half entirely.  Finite exhaustive lifetimes are evidence for the remaining
integer-orbit theorem, not a proof of it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def half_step(n: int, boundary: int) -> int:
    """Advance an eventually-zero left half by one prescribed-boundary step."""
    if n < 0 or boundary not in (0, 1):
        raise ValueError("n must be nonnegative and boundary must be 0 or 1")
    return (n >> 1) ^ (n | ((n << 1) | boundary))


def macro_step(n: int) -> int:
    """Advance from one centre-zero phase to the next."""
    return half_step(half_step(n, 0), 1)


def survivor_condition(n: int) -> bool:
    """Whether the intervening centre-one phase can return to centre zero."""
    return (n & 3) in (1, 2)


def period_constraint_step(n: int, word: str) -> tuple[int, bool]:
    """Apply one boundary period and test every centre-one latch constraint."""
    if not word or any(bit not in "01" for bit in word):
        raise ValueError("word must be a nonempty binary string")
    bits = tuple(int(bit) for bit in word)
    state = n
    valid = True
    for phase, boundary in enumerate(bits):
        if boundary == 1:
            required_left = 1 ^ bits[(phase + 1) % len(bits)]
            valid &= (state & 1) == required_left
        state = half_step(state, boundary)
    return state, bool(valid)


def first_period_failure(n: int, word: str, limit: int) -> int | None:
    """First macroperiod whose centre-one latch constraint fails."""
    state = n
    for k in range(limit + 1):
        state, valid = period_constraint_step(state, word)
        if not valid:
            return k
    return None


def first_failure_macrotime(n: int, limit: int) -> int | None:
    """First k with G^k(n) failing the low-bit condition, or None by limit."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    state = n
    for k in range(limit + 1):
        if not survivor_condition(state):
            return k
        state = macro_step(state)
    return None


def _half_step_u64(values: np.ndarray, boundary: int) -> np.ndarray:
    return (values >> np.uint64(1)) ^ (
        values | ((values << np.uint64(1)) | np.uint64(boundary)))


def lifetime_profile(width: int, limit: int | None = None) -> dict:
    """Exhaust every initial left half supported in the first ``width`` bits."""
    if not 1 <= width <= 20:
        raise ValueError("width must be in 1..20")
    if limit is None:
        limit = width + 2
    # A surviving state grows by exactly two high bits per macrostep.  Keep the
    # vectorized calculation inside uint64; callers can use the scalar map for
    # deeper targeted checks.
    if width + 2 * limit > 64:
        raise ValueError("width + 2*limit must not exceed 64")

    states = np.arange(1 << width, dtype=np.uint64)
    unresolved = np.ones(len(states), dtype=bool)
    failures = np.full(len(states), -1, dtype=np.int16)
    survivor_counts = []
    for k in range(limit + 1):
        low = states & np.uint64(3)
        valid = (low == 1) | (low == 2)
        just_failed = unresolved & ~valid
        failures[just_failed] = k
        unresolved &= valid
        survivor_counts.append(int(unresolved.sum()))
        if not np.any(unresolved):
            break
        active = states[unresolved]
        active = _half_step_u64(_half_step_u64(active, 0), 1)
        states[unresolved] = active

    observed = failures[failures >= 0]
    maximum = int(observed.max()) if len(observed) else None
    return {
        "left_support_width": width,
        "initial_halves": 1 << width,
        "search_limit": limit,
        "maximum_first_failure_macrotime": maximum,
        "extremizer_count": int(np.count_nonzero(failures == maximum))
        if maximum is not None else 0,
        "right_censored_count": int(unresolved.sum()),
        "survivor_counts": survivor_counts,
        "candidate_q_le_w_plus_2_survives": bool(
            not np.any(unresolved) and maximum is not None and maximum <= width + 2),
    }


def self_test() -> dict:
    # Direct bitwise evaluation of Rule 30 on the left half, including the new
    # expanding outer cell, independently checks the integer identity.
    mismatches = 0
    for n in range(256):
        for boundary in (0, 1):
            bits = [(n >> j) & 1 for j in range(10)]
            direct = 0
            for j in range(10):
                deeper = bits[j + 1] if j + 1 < len(bits) else 0
                nearer = boundary if j == 0 else bits[j - 1]
                output = deeper ^ (bits[j] | nearer)
                direct |= output << j
            exact = half_step(n, boundary)
            mismatches += direct != exact

    low_bit_condition_mismatches = 0
    for n in range(256):
        odd_left_neighbour = half_step(n, 0) & 1
        low_bit_condition_mismatches += (
            bool(odd_left_neighbour) != survivor_condition(n))
    alternating_generalization_mismatches = 0
    for n in range(256):
        state, valid = period_constraint_step(n, "01")
        alternating_generalization_mismatches += state != macro_step(n)
        alternating_generalization_mismatches += valid != survivor_condition(n)
    return {
        "half_step_identity_mismatches": mismatches,
        "low_bit_condition_mismatches": low_bit_condition_mismatches,
        "alternating_generalization_mismatches": alternating_generalization_mismatches,
    }


def _primitive(word: str) -> bool:
    return all(word != word[:d] * (len(word) // d)
               for d in range(1, len(word)) if len(word) % d == 0)


def scan_period_words(max_period: int, width: int, limit: int) -> list[dict]:
    """Bounded falsifier for the generalized left-only period-word framework."""
    if not 2 <= max_period <= 10 or not 1 <= width <= 12 or limit < 1:
        raise ValueError("unsupported period-word scan bounds")
    report = []
    for period in range(2, max_period + 1):
        words = [format(value, f"0{period}b")
                 for value in range(1, (1 << period) - 1)
                 if _primitive(format(value, f"0{period}b"))]
        capped = 0
        maximum = -1
        extremal_word = None
        for word in words:
            for n in range(1 << width):
                failure = first_period_failure(n, word, limit)
                if failure is None:
                    capped += 1
                elif failure > maximum:
                    maximum = failure
                    extremal_word = word
        report.append({
            "period": period,
            "primitive_words_including_rotations": len(words),
            "initial_left_halves_per_word": 1 << width,
            "maximum_first_failure_macroperiod": maximum,
            "one_extremal_word": extremal_word,
            "right_censored_count": capped,
        })
    return report


def run(max_width: int, scan_period_max: int = 0,
        scan_width: int = 10, scan_limit: int = 200) -> dict:
    check = self_test()
    if any(check.values()):
        raise RuntimeError(f"self-test failed: {check}")
    profiles = [lifetime_profile(width) for width in range(1, max_width + 1)]
    report = {
        "experiment": "alternating-left-transfer",
        "map": "T_b(n)=(n>>1) XOR (n OR ((n<<1) OR b)); G=T_1(T_0(n))",
        "survivor_condition": "G^k(n) mod 4 in {1,2}",
        "self_test": check,
        "profiles": profiles,
        "claim_limit": (
            "The half-line reduction is exact. Exhaustive finite widths do not "
            "prove that every ordinary integer orbit exits the survivor set; "
            "a width-independent ranking, induction, or certificate is still "
            "required."),
    }
    if scan_period_max:
        report["general_period_word_scan"] = scan_period_words(
            scan_period_max, scan_width, scan_limit)
        report["general_period_claim_limit"] = (
            "Necessary left-latch constraints only, at the stated finite word "
            "length, support width, and iteration cap. Zero capped survivors "
            "does not prove any all-period theorem.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-width", type=int, default=20)
    parser.add_argument("--scan-period-max", type=int, default=0)
    parser.add_argument("--scan-width", type=int, default=10)
    parser.add_argument("--scan-limit", type=int, default=200)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_width <= 20:
        parser.error("--max-width must be in 1..20")
    report = run(args.max_width, args.scan_period_max,
                 args.scan_width, args.scan_limit)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
