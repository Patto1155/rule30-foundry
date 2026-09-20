#!/usr/bin/env python3
"""Construct finite prefixes of Rule 30's alternating-trace fiber.

Fix an alternating centre trace and an initial right half. Left permutivity
determines the initial left half one cell at a time: initial cell -d first
reaches the centre at time d, by a unique path on which its coefficient is 1.
Set all still-unknown deeper cells to zero, simulate d steps, and choose cell
-d to correct the centre bit. Earlier trace bits cannot change.

This constructs a finite prefix of the unique compatible left half. It does
not prove that the left half has infinite support. The intended use is to find
or falsify a precise tail invariant before attempting the period-two proof.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.periodic_trace_fiber import (  # noqa: E402
    forced_left_prefix as periodic_forced_left_prefix,
    step_eca,
    verify_trace,
)
from experiments.rule30_open_utils import step_naive_open  # noqa: E402


def step(rows: np.ndarray) -> np.ndarray:
    return step_eca(rows, 30)


def forced_left_prefix(depth: int, right_support: int, phase: int):
    if phase not in (0, 1):
        raise ValueError("phase must be 0 or 1")
    if not 1 <= right_support <= 16 or not right_support <= depth <= 96:
        raise ValueError("require 1 <= right-support <= 16 and support <= depth <= 96")
    rows, left, target, _ = periodic_forced_left_prefix(
        depth=depth, right_support=right_support, word="01", rotation=phase)
    return rows, left, target


def verify(rows: np.ndarray, target: np.ndarray) -> int:
    return verify_trace(rows, target, 30)


def phase_report(depth: int, right_support: int, phase: int) -> dict:
    rows, left, target = forced_left_prefix(depth, right_support, phase)
    sample = sorted({0, len(rows) // 2, len(rows) - 1})
    stepper_mismatches = sum(
        int(np.count_nonzero(step(rows[i:i + 1])[0] != step_naive_open(rows[i])))
        for i in sample)
    if stepper_mismatches:
        raise RuntimeError(f"vector stepper disagrees with naive stepper: {stepper_mismatches}")
    mismatches = verify(rows, target)
    if mismatches:
        raise RuntimeError(f"constructed fiber prefix failed verification: {mismatches}")
    suffix_lengths = sorted(set(q for q in (4, 8, 12, 16, 24, 32) if q <= depth))
    suffix_zero = {str(q): int(np.all(left[:, -q:] == 0, axis=1).sum())
                   for q in suffix_lengths}
    trailing = np.empty(len(left), dtype=np.int16)
    for i, row in enumerate(left):
        ones = np.flatnonzero(row)
        trailing[i] = depth if len(ones) == 0 else depth - int(ones[-1]) - 1
    fixed = []
    for d in range(depth):
        mean = float(left[:, d].mean())
        if mean in (0.0, 1.0):
            fixed.append({"depth": d + 1, "bit": int(mean)})
    return {"phase": phase, "right_halves": len(left), "depth": depth,
            "right_support": right_support, "verification_mismatches": mismatches,
            "stepper_mismatches": stepper_mismatches,
            "fixed_left_bits": fixed, "suffix_all_zero_counts": suffix_zero,
            "maximum_trailing_zero_run": int(trailing.max()),
            "rows_attaining_maximum": int(np.count_nonzero(trailing == trailing.max())),
            "right_half_examples_at_maximum":
                np.flatnonzero(trailing == trailing.max())[:5].astype(int).tolist(),
            "near_centre": {
                "L1_ne_L2": bool(np.all(left[:, 0] != left[:, 1])) if depth >= 2 else None,
                "L2_ne_L3": bool(np.all(left[:, 1] != left[:, 2])) if depth >= 3 else None}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=48)
    parser.add_argument("--right-support", type=int, default=12)
    parser.add_argument("--phase", choices=("0", "1", "both"), default="both")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    phases = (0, 1) if args.phase == "both" else (int(args.phase),)
    report = {"claim_limit": "Finite exhaustive evidence over the stated right-support "
                             "and left-prefix depth; no infinite-support claim is proved.",
              "phases": [phase_report(args.depth, args.right_support, p) for p in phases]}
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
