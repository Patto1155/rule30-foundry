#!/usr/bin/env python3
"""Construct exact finite prefixes of periodic-trace fibres of left-permutive ECAs.

Fix an initial right half and a desired centre trace.  Left permutivity makes
the initial left half unique, one bit at a time: the bit at ``-d`` first reaches
the centre at time ``d`` and enters there with coefficient one.  This module
constructs that inverse prefix for Rule 30 and, as a control, any other
left-permutive elementary cellular automaton.

The construction is exact for the requested finite depth.  It does *not* show
that the compatible infinite left half has infinite support.  For Prize 1 the
missing statement is precisely that an eventually periodic target cannot have
an eventually-zero inverse left half when the right half is finitely supported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.rule30_open_utils import step_naive_open  # noqa: E402


def _binary_word(word: str) -> tuple[int, ...]:
    if not word or any(ch not in "01" for ch in word):
        raise ValueError("word must be a nonempty binary string")
    return tuple(int(ch) for ch in word)


def is_left_permutive(rule: int) -> bool:
    """Whether flipping the left input flips the output for every (c, r)."""
    if not 0 <= rule <= 255:
        return False
    for centre in (0, 1):
        for right in (0, 1):
            low = (rule >> ((centre << 1) | right)) & 1
            high = (rule >> (4 | (centre << 1) | right)) & 1
            if low == high:
                return False
    return True


def step_eca(rows: np.ndarray, rule: int = 30) -> np.ndarray:
    """One open-boundary ECA step over one row or a batch of rows."""
    if not 0 <= rule <= 255:
        raise ValueError("rule must be in 0..255")
    left = np.zeros_like(rows)
    left[..., 1:] = rows[..., :-1]
    right = np.zeros_like(rows)
    right[..., :-1] = rows[..., 1:]
    neighbourhood = (left << 2) | (rows << 1) | right
    return ((rule >> neighbourhood) & 1).astype(np.uint8)


def periodic_target(word: str, length: int, rotation: int = 0) -> np.ndarray:
    bits = _binary_word(word)
    if length < 1:
        raise ValueError("length must be positive")
    rotation %= len(bits)
    return np.fromiter((bits[(rotation + t) % len(bits)] for t in range(length)),
                       dtype=np.uint8, count=length)


def _right_values(right_support: int, values: Iterable[int] | None) -> np.ndarray:
    limit = 1 << right_support
    if values is None:
        return np.arange(limit, dtype=np.uint32)
    out = np.asarray(list(values), dtype=np.uint32)
    if out.ndim != 1 or len(out) == 0 or np.any(out >= limit):
        raise ValueError("right values must be a nonempty subset of the support range")
    return out


def forced_left_prefix(
    depth: int,
    right_support: int,
    word: str,
    rotation: int = 0,
    rule: int = 30,
    right_values: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return constructed rows, left prefixes, target trace, and right values.

    ``left[:, d-1]`` is the initial bit at spatial coordinate ``-d``.  Right
    value bit zero is coordinate ``+1``.  The entire right half beyond
    ``right_support`` is fixed to zero.
    """
    if not is_left_permutive(rule):
        raise ValueError(f"rule {rule} is not left permutive")
    if not 1 <= right_support <= 18:
        raise ValueError("right_support must be in 1..18")
    if not right_support <= depth <= 192:
        raise ValueError("require right_support <= depth <= 192")
    values = _right_values(right_support, right_values)
    target = periodic_target(word, depth + 1, rotation)
    width = 2 * depth + 1
    centre = depth
    rows = np.zeros((len(values), width), dtype=np.uint8)
    rows[:, centre] = target[0]
    rows[:, centre + 1:centre + right_support + 1] = (
        (values[:, None] >> np.arange(right_support, dtype=np.uint32)) & 1
    ).astype(np.uint8)
    left = np.empty((len(values), depth), dtype=np.uint8)

    for d in range(1, depth + 1):
        evolved = rows.copy()
        for _ in range(d):
            evolved = step_eca(evolved, rule)
        # Left permutivity makes the extreme-left input enter with coefficient
        # one, so this is the unique choice correcting the time-d output.
        bit = evolved[:, centre] ^ target[d]
        rows[:, centre - d] = bit
        left[:, d - 1] = bit
    return rows, left, target, values


def verify_trace(rows: np.ndarray, target: np.ndarray, rule: int = 30) -> int:
    centre = rows.shape[1] // 2
    current = rows.copy()
    mismatches = 0
    for bit in target:
        mismatches += int(np.count_nonzero(current[:, centre] != bit))
        current = step_eca(current, rule)
    return mismatches


def maximum_zero_run(left: np.ndarray, start_depth: int = 1) -> dict:
    """Longest observed zero run at depths >= ``start_depth`` (one based)."""
    if not 1 <= start_depth <= left.shape[1]:
        raise ValueError("start_depth outside the constructed prefix")
    best = {"length": 0, "row": None, "start_depth": None, "end_depth": None}
    for row_index, row in enumerate(left):
        run = 0
        for offset, bit in enumerate(row[start_depth - 1:], start=start_depth):
            run = run + 1 if bit == 0 else 0
            if run > best["length"]:
                best = {"length": run, "row": row_index,
                        "start_depth": offset - run + 1, "end_depth": offset}
    return best


def self_test() -> dict:
    rng = np.random.default_rng(30)
    rule30_mismatches = 0
    for _ in range(8):
        row = rng.integers(0, 2, size=65, dtype=np.uint8)
        rule30_mismatches += int(np.count_nonzero(
            step_eca(row, 30) != step_naive_open(row)))

    fibre_mismatches = 0
    for word in ("0", "1", "01", "001", "0110"):
        for rotation in range(len(word)):
            rows, _, target, _ = forced_left_prefix(
                depth=10, right_support=4, word=word, rotation=rotation)
            fibre_mismatches += verify_trace(rows, target)

    # Rule 90 is left permutive but the finite row 101 (coordinates -1..1)
    # produces centre trace 0 forever after the chosen onset.  This prevents a
    # future proof from silently using left permutivity as if it were Rule-30
    # specific.
    rows90, left90, target90, _ = forced_left_prefix(
        depth=10, right_support=1, word="0", rule=90, right_values=[1])
    rule90_mismatches = verify_trace(rows90, target90, 90)
    rule90_finite_control = bool(
        left90[0, 0] == 1 and np.count_nonzero(left90[0, 1:]) == 0)

    # Rule 15 is the sharper control for the alternating target.  Its local
    # rule is F(l,c,r) = 1 XOR l, so an all-zero initial left half makes the
    # centre alternate forever, independently of the right half.  Any future
    # period-two argument based only on left permutivity, triangular inversion,
    # or low-complexity survivor sets must therefore fail this control.
    rows15, left15, target15, _ = forced_left_prefix(
        depth=10, right_support=1, word="01", rule=15, right_values=[1])
    rule15_mismatches = verify_trace(rows15, target15, 15)
    rule15_finite_control = bool(np.count_nonzero(left15) == 0)
    return {
        "rule30_stepper_mismatches": rule30_mismatches,
        "constructed_trace_mismatches": fibre_mismatches,
        "rule90_trace_mismatches": rule90_mismatches,
        "rule90_finite_periodic_fibre_control": rule90_finite_control,
        "rule15_trace_mismatches": rule15_mismatches,
        "rule15_finite_alternating_fibre_control": rule15_finite_control,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word", default="01")
    parser.add_argument("--rotation", type=int, default=0)
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--right-support", type=int, default=12)
    parser.add_argument("--rule", type=int, default=30)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    check = self_test()
    if (check["rule30_stepper_mismatches"] or
            check["constructed_trace_mismatches"] or
            check["rule90_trace_mismatches"] or
            not check["rule90_finite_periodic_fibre_control"] or
            check["rule15_trace_mismatches"] or
            not check["rule15_finite_alternating_fibre_control"]):
        print(json.dumps(check, indent=2))
        return 2
    if args.self_test:
        print(json.dumps(check, indent=2))
        return 0

    rows, left, target, values = forced_left_prefix(
        args.depth, args.right_support, args.word, args.rotation, args.rule)
    mismatches = verify_trace(rows, target, args.rule)
    report = {
        "rule": args.rule,
        "word": args.word,
        "rotation": args.rotation % len(args.word),
        "depth": args.depth,
        "right_support": args.right_support,
        "right_halves": len(values),
        "verification_mismatches": mismatches,
        "maximum_zero_run_beyond_right_support": maximum_zero_run(
            left, args.right_support + 1 if args.right_support < args.depth else 1),
        "left_prefixes_ending_in_zeros": {
            str(q): int(np.all(left[:, -q:] == 0, axis=1).sum())
            for q in (4, 8, 16, 32) if q <= args.depth
        },
        "claim_limit": (
            "Exact finite prefixes of the stated fibre only. A zero suffix at "
            "finite depth is not an eventually-zero left half; absence of one "
            "does not prove infinite support or exclude a periodic trace."),
    }
    if mismatches:
        raise RuntimeError("constructed rows do not realize the requested trace")
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
