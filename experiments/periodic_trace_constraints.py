"""Exact finite controls for prescribed center traces and time-shift defects.

Tail scanning compares finite prefixes only. It does not exclude eventual
periodicity: untested onsets, periods, and unresolved finite tails remain open.
The prescribed-trace construction changes the initial condition and is a
control demonstrating why local feasibility alone cannot settle the prize.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _bit(value: int) -> int:
    if not isinstance(value, int) or value not in (0, 1):
        raise ValueError("bits must be integers 0 or 1")
    return value


def _nonnegative(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def rule30(a: int, b: int, c: int) -> int:
    return _bit(a) ^ (_bit(b) | _bit(c))


def center_trace(initial: Mapping[int, int], steps: int) -> list[int]:
    """Return center values at times 0 through steps, inclusive.

    Missing coordinates are zero. The shrinking light cone is exact and
    avoids imposing any artificial boundary on the requested output.
    """
    _nonnegative(steps, "steps")
    for position, bit in initial.items():
        if not isinstance(position, int) or isinstance(position, bool):
            raise ValueError("initial coordinates must be integers")
        _bit(bit)
    row = {i: initial.get(i, 0) for i in range(-steps, steps + 1)}
    trace = [row[0]]
    for time in range(1, steps + 1):
        row = {
            i: rule30(row[i - 1], row[i], row[i + 1])
            for i in range(-steps + time, steps - time + 1)
        }
        trace.append(row[0])
    return trace


def realize_trace(target: Sequence[int]) -> dict[int, int]:
    """Construct the unique left initial segment for a finite target.

    Coordinates >0 are zero. At step t, the still unset bit at -t affects
    c_t with XOR coefficient one by left permutivity; it cannot affect any
    earlier center value. This elementary implementation costs O(len^3).
    """
    target = [_bit(bit) for bit in target]
    if not target:
        raise ValueError("target must contain at least one bit")
    initial = {0: target[0]}
    for time in range(1, len(target)):
        initial[-time] = 0
        initial[-time] = center_trace(initial, time)[-1] ^ target[time]
    return initial


def defect_update(background: Sequence[int], defect: Sequence[int]) -> int:
    """Difference of two Rule 30 outputs, with second input = first XOR d."""
    if len(background) != 3 or len(defect) != 3:
        raise ValueError("background and defect must each contain three bits")
    _, center, right = [_bit(bit) for bit in background]
    dleft, dcenter, dright = [_bit(bit) for bit in defect]
    return (dleft ^ ((1 ^ right) * dcenter)
            ^ ((1 ^ center) * dright) ^ (dcenter * dright))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def self_test() -> dict:
    pairs = 0
    for background in itertools.product((0, 1), repeat=3):
        for defect in itertools.product((0, 1), repeat=3):
            second = tuple(a ^ d for a, d in zip(background, defect))
            _require(defect_update(background, defect)
                     == rule30(*background) ^ rule30(*second), "defect identity")
            pairs += 1
    words = 0
    for length in range(1, 9):
        for target in itertools.product((0, 1), repeat=length):
            _require(center_trace(realize_trace(target), length - 1)
                     == list(target), "trace inversion")
            words += 1
    seed = center_trace({0: 1}, 80)
    recovered = realize_trace(seed)
    _require(recovered[0] == 1 and not any(
        bit for i, bit in recovered.items() if i < 0), "single seed inversion")
    return {"status": "pass", "paired_triples": pairs,
            "prescribed_words": words, "seed_steps": 80,
            "seed_left_tail_all_zero": True}


def tail_scan(steps: int = 512, max_period: int = 16,
              max_onset: int = 64) -> dict:
    _nonnegative(steps, "steps")
    _nonnegative(max_period, "max_period")
    _nonnegative(max_onset, "max_onset")
    if max_period == 0:
        raise ValueError("max_period must be positive")
    if max_onset + max_period > steps + 1:
        raise ValueError("every candidate's first period must fit in the horizon")
    seed = center_trace({0: 1}, steps)
    rows = []
    for onset in range(max_onset + 1):
        for period in range(1, max_period + 1):
            word = seed[onset:onset + period]
            mismatch = next((t for t in range(onset + period, steps + 1)
                             if seed[t] != word[(t - onset) % period]), None)
            rows.append({"onset": onset, "period": period,
                         "word": word, "first_mismatch_time": mismatch,
                         "status": "unresolved" if mismatch is None else "mismatch"})
    # Fixed, reproducible changed-initial-condition control, independent of
    # scan dimensions. The modified seed realizes this tail only through t=64.
    witness_seed = center_trace({0: 1}, 64)
    onset, period = 8, 3
    target = witness_seed[:onset] + [
        witness_seed[onset + (t - onset) % period] for t in range(onset, 65)]
    initial = realize_trace(target)
    mismatch = next((t for t in range(65) if target[t] != witness_seed[t]), None)
    nonzero_left = sorted((i for i, bit in initial.items() if i < 0 and bit),
                          reverse=True)
    modified = center_trace(initial, 64)
    _require(modified == target, "modified-seed forward witness")
    _require(bool(nonzero_left) and -nonzero_left[0] == mismatch,
             "first initial difference reaches center at its distance")
    return {"scope": "finite prefix comparison; no eventual-period exclusion",
            "time_range_inclusive": [0, steps], "max_period": max_period,
            "max_onset": max_onset, "candidate_count": len(rows),
            "mismatch_count": sum(r["status"] == "mismatch" for r in rows),
            "unresolved_count": sum(r["status"] == "unresolved" for r in rows),
            "candidates": rows,
            "changed_seed_witness": {
                "onset": onset, "period": period, "horizon_inclusive": 64,
                "seed_trace": witness_seed, "target_trace": target,
                "modified_seed_trace": modified,
                "initial_nonzero_coordinates": sorted(i for i, b in initial.items() if b),
                "nearest_nonzero_negative_coordinate": nonzero_left[0],
                "first_output_mismatch_time": mismatch,
                "modified_seed_forward_check": modified == target,
                "seed_forward_check": witness_seed == center_trace({0: 1}, 64)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--tail-scan", action="store_true")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--max-period", type=int, default=16)
    parser.add_argument("--max-onset", type=int, default=64)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = self_test() if args.self_test else tail_scan(
            args.steps, args.max_period, args.max_onset)
    except ValueError as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8", newline="")
    print(encoded, end="")


if __name__ == "__main__":
    main()
