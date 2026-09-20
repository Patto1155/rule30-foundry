#!/usr/bin/env python3
"""Exact spatial-cycle test for periodic free boundaries of the alternating fibre.

For a period-``p`` even-time right-neighbour word ``s``, sideways Rule 30
reconstruction is a deterministic map on four ``p``-bit temporal words.  This
script follows that finite map until its first repeated state and computes the
exact rational density of initial-left ones on the eventual spatial cycle.

The declared search is all boundary words for ``1 <= p <= 6``.  Each orbit is
right-censored after 100,000 distinct states; the horizon is never enlarged.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.periodic_trace_fiber import (  # noqa: E402
    forced_left_prefix,
    step_eca,
    verify_trace,
)

State = tuple[int, int, int, int]


def rotate_time(word: int, period: int) -> int:
    """Return the word whose phase-k bit is the input phase-(k+1) bit."""
    if not 1 <= period <= 30 or not 0 <= word < (1 << period):
        raise ValueError("invalid period or word")
    return (word >> 1) | ((word & 1) << (period - 1))


def rule30_spatial_step(state: State, period: int) -> tuple[State, int]:
    """Move one column left and return ``(new_state, initial_row_bit)``."""
    mask = (1 << period) - 1
    even_previous, even, odd_previous, odd = state
    if any(not 0 <= word <= mask for word in state):
        raise ValueError("state word does not fit period")
    even_new = odd ^ (even | even_previous)
    odd_new = rotate_time(even, period) ^ (odd | odd_previous)
    return (even, even_new & mask, odd, odd_new & mask), even_new & 1


def initial_state(boundary_word: int, period: int) -> State:
    """Alternating centre with free even-right word and normalized odd-right word."""
    if not 0 <= boundary_word < (1 << period):
        raise ValueError("boundary word does not fit period")
    return boundary_word, 0, 0, (1 << period) - 1


def orbit_report(boundary_word: int, period: int, max_states: int) -> dict:
    """Follow one exact orbit, reporting a cycle or an honest censor."""
    if max_states < 1:
        raise ValueError("max_states must be positive")
    state = initial_state(boundary_word, period)
    seen: dict[State, int] = {}
    outputs: list[int] = []
    while state not in seen and len(seen) < max_states:
        seen[state] = len(outputs)
        state, output = rule30_spatial_step(state, period)
        outputs.append(output)
    base = {
        "boundary_lsb_first": format(boundary_word, f"0{period}b")[::-1],
        "visited_states": len(seen),
    }
    if state not in seen:
        return {**base, "status": "right_censored", "max_states": max_states}
    transient = seen[state]
    cycle = outputs[transient:]
    ones = sum(cycle)
    density = Fraction(ones, len(cycle))
    return {
        **base,
        "status": "cycle",
        "transient_length": transient,
        "cycle_length": len(cycle),
        "cycle_ones": ones,
        "cycle_density": {
            "numerator": density.numerator,
            "denominator": density.denominator,
        },
        "cycle_output_lsb_spatial_order": "".join(map(str, cycle)),
        "charge_slope_numerator": 8 * ones - len(cycle),
        "charge_slope_denominator": len(cycle),
    }


def _inverse_left(rule: int, future: int, centre: int, right: int) -> int:
    choices = [left for left in (0, 1)
               if ((rule >> (4 * left + 2 * centre + right)) & 1) == future]
    if len(choices) != 1:
        raise ValueError(f"rule {rule} is not left permutive")
    return choices[0]


def rule30_ring_step(row: str) -> str:
    """Independent cell-by-cell Rule 30 step with periodic spatial boundary."""
    if not row or set(row) - set("01"):
        raise ValueError("row must be a nonempty binary string")
    width = len(row)
    return "".join(str((30 >> (4 * int(row[(index - 1) % width])
                                  + 2 * int(row[index])
                                  + int(row[(index + 1) % width]))) & 1)
                   for index in range(width))


def period_four_ether_witness() -> dict:
    """Return and independently verify the sharp 7-by-4 spacetime witness."""
    rows = ["1000000", "1100001", "0010011", "1111110", "1000000"]
    mismatches = sum(rule30_ring_step(first) != second
                     for first, second in zip(rows, rows[1:]))
    temporal_columns = ["".join(row[index] for row in rows[:-1])
                        for index in range(7)]
    return {
        "spatial_period": 7,
        "temporal_period": 4,
        "rows": rows,
        "truth_table_step_mismatches": mismatches,
        "alternating_centre_column_index": 1,
        "alternating_centre_column": temporal_columns[1],
        "initial_row_density": {"numerator": 1, "denominator": 7},
        "uniform_alpha_lower_bound": 7,
    }


def direct_period_wrap_mismatches(max_period: int = 4) -> int:
    """Compare the word recurrence with the Rule-30 truth table phase by phase."""
    mismatches = 0
    for period in range(1, max_period + 1):
        limit = 1 << period
        for ep in range(limit):
            for even in range(limit):
                for op in range(limit):
                    for odd in range(limit):
                        successor, _ = rule30_spatial_step((ep, even, op, odd), period)
                        enew, onew = successor[1], successor[3]
                        for phase in range(period):
                            get = lambda word, k: (word >> (k % period)) & 1
                            expected_even = _inverse_left(
                                30, get(odd, phase), get(even, phase), get(ep, phase))
                            expected_odd = _inverse_left(
                                30, get(even, phase + 1), get(odd, phase), get(op, phase))
                            mismatches += get(enew, phase) != expected_even
                            mismatches += get(onew, phase) != expected_odd
    return int(mismatches)


def finite_cone_mismatches(max_period: int = 4) -> int:
    """Cross-check early columns against the exact finite-cone constructor."""
    mismatches = 0
    for period in range(1, max_period + 1):
        depth = 2 * period
        right_support = period
        rows, left, _, _ = forced_left_prefix(depth, right_support, "01")
        current = rows.copy()
        centre = depth
        boundary_words = np.zeros(len(rows), dtype=np.uint32)
        for time in range(2 * period):
            if time % 2 == 0:
                boundary_words |= current[:, centre + 1].astype(np.uint32) << (time // 2)
            current = step_eca(current, 30)
        for row_index, boundary_word in enumerate(boundary_words):
            state = initial_state(int(boundary_word), period)
            emitted = []
            for _ in range(period):
                state, bit = rule30_spatial_step(state, period)
                emitted.append(bit)
            mismatches += int(np.count_nonzero(
                np.asarray(emitted, dtype=np.uint8) != left[row_index, :period]))
    return mismatches


def controls() -> dict:
    """Finite-cone Rule 15 and Rule 90 baselines, not periodic-map tests."""
    rows15, left15, target15, _ = forced_left_prefix(
        depth=12, right_support=2, word="01", rule=15)
    rows90, left90, target90, values90 = forced_left_prefix(
        depth=12, right_support=2, word="01", rule=90)
    words90 = ["".join(map(str, row)) for row in left90]
    return {
        "scope": (
            "Exact finite-cone constructor baselines only; these controls do "
            "not exercise the Rule 30 periodic word map or orbit scanner."),
        "rule15": {
            "trace_mismatches": verify_trace(rows15, target15, 15),
            "all_boundaries_have_zero_left_prefix": bool(np.count_nonzero(left15) == 0),
            "cases": len(left15),
        },
        "rule90": {
            "trace_mismatches": verify_trace(rows90, target90, 90),
            "right_initial_values": [int(value) for value in values90],
            "left_prefixes": words90,
            "distinct_left_prefixes": len(set(words90)),
            "boundary_constraint_is_distinct": len(set(words90)) > 1,
        },
    }


def verify_report(path: Path) -> dict:
    """Recompute the canonical declared scan and exactly compare saved JSON."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read report {path}: {exc}") from exc
    required_limits = {
        "max_period": 6,
        "max_states_per_word": 100_000,
        "automatic_horizon_growth": False,
    }
    if saved.get("declared_limits") != required_limits:
        raise ValueError(
            f"declared_limits mismatch: expected {required_limits!r}, "
            f"got {saved.get('declared_limits')!r}")
    recomputed = run(max_period=6, max_states=100_000)
    if saved != recomputed:
        def first_difference(expected, actual, location="report"):
            if type(expected) is not type(actual):
                return location
            if isinstance(expected, dict):
                if expected.keys() != actual.keys():
                    return location
                for key in expected:
                    difference = first_difference(
                        expected[key], actual[key], f"{location}.{key}")
                    if difference:
                        return difference
            elif isinstance(expected, list):
                if len(expected) != len(actual):
                    return location
                for index, (left, right) in enumerate(zip(expected, actual)):
                    difference = first_difference(
                        left, right, f"{location}[{index}]")
                    if difference:
                        return difference
            elif expected != actual:
                return location
            return None

        location = first_difference(recomputed, saved)
        raise ValueError(f"saved report differs from exact replay at {location}")
    return {
        "verified": True,
        "path": str(path),
        "max_period": 6,
        "boundary_words": sum(row["boundary_words"] for row in recomputed["periods"]),
        "right_censored": recomputed["total_right_censored"],
        "global_minimum_cycle_density": recomputed["global_minimum_cycle_density"],
    }


def run(max_period: int = 6, max_states: int = 100_000) -> dict:
    if not 1 <= max_period <= 6:
        raise ValueError("declared scan requires max_period in 1..6")
    if max_states != 100_000:
        raise ValueError("declared scan fixes max_states at 100000")
    wrap_mismatches = direct_period_wrap_mismatches()
    cone_mismatches = finite_cone_mismatches()
    control = controls()
    ether = period_four_ether_witness()
    if (wrap_mismatches or cone_mismatches
            or ether["truth_table_step_mismatches"]
            or ether["alternating_centre_column"] != "0101"):
        raise RuntimeError("sideways recurrence cross-check failed")
    periods = []
    global_minimum: Fraction | None = None
    low_density_cycles = []
    total_censored = 0
    for period in range(1, max_period + 1):
        reports = [orbit_report(word, period, max_states)
                   for word in range(1 << period)]
        resolved = [row for row in reports if row["status"] == "cycle"]
        censored = len(reports) - len(resolved)
        total_censored += censored
        minimum = min((Fraction(row["cycle_density"]["numerator"],
                                row["cycle_density"]["denominator"])
                       for row in resolved), default=None)
        minimizers = [row for row in resolved
                      if Fraction(row["cycle_density"]["numerator"],
                                  row["cycle_density"]["denominator"]) == minimum]
        below = [row for row in resolved if 8 * row["cycle_ones"] < row["cycle_length"]]
        low_density_cycles.extend({"period": period, **row} for row in below)
        if minimum is not None and (global_minimum is None or minimum < global_minimum):
            global_minimum = minimum
        periods.append({
            "period": period,
            "boundary_words": len(reports),
            "resolved_cycles": len(resolved),
            "right_censored": censored,
            "minimum_cycle_density": None if minimum is None else {
                "numerator": minimum.numerator, "denominator": minimum.denominator},
            "minimum_example_count": len(minimizers),
            "minimum_examples": minimizers[:3],
            "cycles_below_one_eighth": len(below),
            "maximum_visited_states": max(row["visited_states"] for row in reports),
        })
    return {
        "experiment": "periodic-boundary-density",
        "rule": 30,
        "target_trace": "01",
        "boundary_family": "all p-bit free even-right words; odd-right normalized to zero",
        "declared_limits": {"max_period": max_period, "max_states_per_word": max_states,
                            "automatic_horizon_growth": False},
        "cross_checks": {
            "truth_table_period_wrap_mismatches": wrap_mismatches,
            "finite_cone_mismatches": cone_mismatches,
        },
        "controls": control,
        "period_four_space_time_ether": ether,
        "periods": periods,
        "total_right_censored": total_censored,
        "global_minimum_cycle_density": None if global_minimum is None else {
            "numerator": global_minimum.numerator,
            "denominator": global_minimum.denominator,
        },
        "cycles_below_one_eighth": low_density_cycles,
        "falsifies_uniform_charge_bound": bool(low_density_cycles),
        "periodic_nonzero_density_lemma": (
            "On a spatial cycle, zero emitted phase-0 E bits force phase-0 O "
            "bits to zero by the E recurrence. The O recurrence then forces "
            "phase-1 E to zero, and induction around the p temporal phases "
            "forces E=O=0. The all-zero four-word state has only itself as a "
            "predecessor, so it is unreachable from the prescribed initial "
            "state, whose O word is all ones. Thus every finite-period boundary "
            "has positive eventual density; this gives no lower bound uniform in p."),
        "claim_limit": (
            "Exact only for periodic free even-right boundary words in the declared "
            "period range and for resolved orbits. A finite negative charge prefix "
            "does not refute a bound with an unspecified additive constant; only an "
            "eventual cycle density below 1/8 does so."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-period", type=int, default=6)
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path,
                        help="recompute and exactly verify a canonical saved report")
    args = parser.parse_args()
    if args.verify:
        if args.out:
            parser.error("--verify and --out are mutually exclusive")
        try:
            print(json.dumps(verify_report(args.verify), indent=2))
        except ValueError as exc:
            parser.error(str(exc))
        return 0
    report = run(args.max_period, args.max_states)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
