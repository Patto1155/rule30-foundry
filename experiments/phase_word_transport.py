#!/usr/bin/env python3
"""Check one-event right-boundary transport for arbitrary centre histories.

The all-parameter proof is in docs/theory/phase-word-transport.md.  These
checks use a separate elementary-rule cone evaluator, not the packed update,
for the supporting finite identities and the period-three invariant.
"""
from __future__ import annotations

import json


def right_step(state: int, boundary: int) -> int:
    return ((state << 1) | boundary) ^ (state | (state >> 1))


def event_return(state: int, zero_run: int) -> int:
    if zero_run < 0:
        raise ValueError("zero_run must be nonnegative")
    for _ in range(zero_run):
        state = right_step(state, 0)
    return right_step(state, 1)


def truth_table_cone(state: int, width: int, boundary: list[int]) -> list[int]:
    """Exact low cells after each step, dropping unknown outer dependencies."""
    if width <= len(boundary):
        raise ValueError("cone must retain at least one output bit")
    cells = [(state >> j) & 1 for j in range(width)]
    for bit in boundary:
        cells = [
            (30 >> (4 * (bit if j == 0 else cells[j - 1])
                    + 2 * cells[j] + cells[j + 1])) & 1
            for j in range(len(cells) - 1)
        ]
    return cells


def encode(bits: list[int]) -> int:
    return sum(bit << j for j, bit in enumerate(bits))


def verify_run_identities(max_run: int = 10) -> dict[str, int]:
    or_checks = nor_checks = profile_checks = left_checks = 0
    for run in range(max_run + 1):
        for state in range(1 << (run + 1)):
            output = truth_table_cone(state, run + 1, [0] * run)[0]
            assert output == int(state != 0)
            or_checks += 1
        for state in range(1 << (run + 2)):
            output = truth_table_cone(state, run + 2, [0] * run + [1])[0]
            assert output == int(state == 0)
            assert output == (event_return(state, run) & 1)
            # This implication also applies at zero-length runs (consecutive ones).
            assert not ((state & 1) and output)
            nor_checks += 1
            threshold = ((state & -state).bit_length() - 1
                         if state else run + 2)
            for phase in range(run + 1):
                near = truth_table_cone(state, run + 2, [0] * phase)[0]
                assert near == int(threshold <= phase)
                profile_checks += 1
                if phase < run:
                    left = int(phase == run - 1) ^ int(threshold <= phase)
                    next_centre = int(phase == run - 1)
                    assert ((30 >> (4 * left + near)) & 1) == next_centre
                    left_checks += 1
                else:
                    for next_gap_positive in (False, True):
                        left = int(next_gap_positive)
                        next_centre = int(not next_gap_positive)
                        assert ((30 >> (4 * left + 2 + near)) & 1) == next_centre
                        left_checks += 1
            assert output == int(threshold == run + 2)
    return {"max_zero_run_checked": max_run,
            "zero_run_or_cones": or_checks, "event_nor_cones": nor_checks,
            "within_run_threshold_checks": profile_checks,
            "nearest_left_truth_table_checks": left_checks}


def verify_period_three_trap() -> dict[str, object]:
    """Universal low-four closure: 7 current states times 8 outer prefixes."""
    expected = {
        2: {2, 10, 14}, 4: {4, 12}, 6: {4, 14}, 8: {2, 10},
        10: {14}, 12: {8, 12}, 14: {4, 12},
    }
    trap = set(expected)
    checks = 0
    for state, targets in expected.items():
        actual = set()
        for outer in range(8):
            cone = state | (outer << 4)
            output = encode(truth_table_cone(cone, 7, [0, 0, 1]))
            assert output == event_return(cone, 2) & 15
            actual.add(output)
            checks += 1
        assert actual == targets
        assert actual <= trap
    assert event_return(0, 2) == 1
    assert event_return(1, 2) == 12
    # Compare complete within-event histories with the same endpoint symbols.
    histories = []
    for initial in (2, 4):
        state = initial
        history = [state & 1]
        for boundary in (0, 0, 1):
            state = right_step(state, boundary)
            history.append(state & 1)
        histories.append(history)
    assert histories == [[0, 1, 1, 0], [0, 0, 1, 0]]
    return {
        "centre_word": "001", "low_four_trap": sorted(trap),
        "universal_dependency_cones": checks,
        "exact_entry_from_finite_right_zero": [0, 1, 12],
        "sampled_trace_from_zero": "0,1,0,0,0,... (all later zeros by closure)",
        "same_event_symbols_different_within_run_histories": histories,
        "scope": "forced right half only; no finite full-row periodic orbit",
    }


def main() -> None:
    print(json.dumps({
        "scope": "all-history one-event algebra and a universally closed period-three right trap",
        "run_identities": verify_run_identities(),
        "period_three_rank_extension_counterexample": verify_period_three_trap(),
        "failures": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
