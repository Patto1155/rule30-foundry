#!/usr/bin/env python3
"""Independent exact verifier for the period-7 ether interface certificate."""

import argparse
from collections import Counter
import json
from pathlib import Path


WIDTH = 18
HORIZON = 8
SAMPLE_SPACING = 14
PHASE_PERIOD = 28
CYCLE_LENGTH = 512
EXPECTED_DELAYS = {0: 256, 1: 128, 2: 64, 4: 32, 5: 24, 6: 4, 7: 2, 8: 2}


def rule30(left, centre, right):
    return (30 >> (4 * left + 2 * centre + right)) & 1


def ether_orbit():
    row = tuple(int(bit) for bit in "1000000")
    rows = []
    for _ in range(4):
        rows.append(row)
        row = tuple(rule30(row[(j + 1) % 7], row[j], row[(j - 1) % 7])
                    for j in range(7))
    if row != rows[0]:
        raise AssertionError("ether does not close after four half-steps")
    return tuple(rows)


ETHER = ether_orbit()


def _bit(word, index):
    return (word >> index) & 1 if index >= 0 else 0


def interface_step(delta, time, width=WIDTH):
    """Advance paired physical cells, then XOR their truth-table outputs."""
    following = 0
    phase = ETHER[time % 4]
    for d in range(width):
        # The three old physical cells are consecutive.  The left input uses
        # lag d; centre and right use lags d-1 and d-2.  Both configurations
        # are stepped separately through Rule 30's truth table.
        left_e = phase[(-time + d) % 7]
        centre_e = phase[(-time - 1 + d) % 7]
        right_e = phase[(-time - 2 + d) % 7]
        ether_out = rule30(left_e, centre_e, right_e)
        actual_out = rule30(left_e ^ _bit(delta, d),
                            centre_e ^ _bit(delta, d - 1),
                            right_e ^ _bit(delta, d - 2))
        following |= (actual_out ^ ether_out) << d
    return following


def halfline_step(word, time):
    """Advance the finite left half beside centre bit ``time mod 2``."""
    return (word >> 1) ^ (word | ((word << 1) | (time & 1)))


def failure_delay(arrival_word, arrival_time, horizon=HORIZON):
    # Use a shrinking dependency cone and the elementary truth table here,
    # independently of the packed map used by the synthesizer. No artificial
    # exterior zero is consulted at the decisive post-arrival boundary.
    cells = [_bit(arrival_word, j) for j in range(WIDTH)]
    time = arrival_time
    for delay in range(horizon + 1):
        if cells[0] == cells[1]:
            return delay
        if delay < horizon:
            for _ in range(2):
                cells = [rule30(cells[j + 1], cells[j],
                                (time & 1) if j == 0 else cells[j - 1])
                         for j in range(len(cells) - 1)]
                time += 1
    raise AssertionError("arrival state survives beyond the certified horizon")


def interface_certificate():
    initial = sum(1 << d for d in range(WIDTH) if d % 7 == 0)
    delta, time = initial, 0
    sampled_states = []
    delays = []
    seen = {}
    for sample in range(CYCLE_LENGTH):
        state = (delta, time % PHASE_PERIOD)
        if state in seen:
            raise AssertionError("combined interface state repeats before sample 512")
        seen[state] = sample
        sampled_states.append(state)
        arrival = delta
        for d in range(WIDTH):
            arrival ^= ETHER[time % 4][d % 7] << d
        delays.append(failure_delay(arrival, time))
        for _ in range(SAMPLE_SPACING):
            delta = interface_step(delta, time)
            time += 1
    if (delta, time % PHASE_PERIOD) != (initial, 0):
        raise AssertionError("sample 512 does not return to the initial combined state")

    # Sixteen half-steps cover delay eight.  Bit 18 cannot reach output bits
    # zero or one in that time, an exact finite-speed coverage check.
    for delta, phase in sampled_states:
        arrival = delta
        for d in range(WIDTH):
            arrival ^= ETHER[phase % 4][d % 7] << d
        a, b = arrival, arrival ^ (1 << WIDTH)
        for step in range(2 * HORIZON + 1):
            if (a & 3) != (b & 3):
                raise AssertionError("18 bits do not cover the failure horizon")
            a = halfline_step(a, phase + step)
            b = halfline_step(b, phase + step)
    return delays


def truncation(r):
    return sum(1 << (7 * j) for j in range(r))


def direct_first_failure(r):
    word = truncation(r)
    for macro in range(4 * r + 1):
        if word & 3 not in (1, 2):
            return macro
        word = halfline_step(word, 2 * macro)
        word = halfline_step(word, 2 * macro + 1)
    raise AssertionError("direct truncation did not fail within 4r")


def verify(report):
    expected_keys = {
        "experiment", "ether_phases_low_index_first", "interface_prefix_width",
        "macro_failure_horizon_after_arrival", "even_r_sample_spacing_halfsteps",
        "combined_phase_period_halfsteps", "pure_sample_cycle_length",
        "failure_delay_counts", "even_r_delay_by_m_mod_512_for_r_2m",
        "maximum_failure_delay", "small_even_exact_first_failures", "theorem", "claim_limit",
        "direct_cross_check_max_r", "direct_first_failures",
    }
    if type(report) is not dict or set(report) != expected_keys:
        raise ValueError("unexpected period-7 certificate schema")
    phases = ["".join(map(str, row)) for row in ETHER]
    if report["experiment"] != "period7-ether-interface" or report["ether_phases_low_index_first"] != phases:
        raise ValueError("ether circuit or coordinate convention is incorrect")
    if any(rule30(0, centre, right) == rule30(1, centre, right)
           for centre in (0, 1) for right in (0, 1)):
        raise AssertionError("Rule 30 is not left permutive")
    for phase in (0, 2):
        low_pair = ETHER[phase][:2]
        if low_pair[0] == low_pair[1] or low_pair[0] != 1 - low_pair[1]:
            raise AssertionError("ether or one-bit odd-r arrival does not force failure")
    scalar_checks = {
        "interface_prefix_width": WIDTH,
        "macro_failure_horizon_after_arrival": HORIZON,
        "even_r_sample_spacing_halfsteps": SAMPLE_SPACING,
        "combined_phase_period_halfsteps": PHASE_PERIOD,
        "pure_sample_cycle_length": CYCLE_LENGTH,
        "maximum_failure_delay": HORIZON,
        "direct_cross_check_max_r": 64,
    }
    if any(type(report.get(key)) is not int or report[key] != value
           for key, value in scalar_checks.items()):
        raise ValueError("incorrect circuit width, horizon, phase, or period")

    delays = interface_certificate()
    if report["even_r_delay_by_m_mod_512_for_r_2m"] != delays:
        raise ValueError("saved even-r delay table does not match the circuit")
    counts = Counter(delays)
    saved_counts = report["failure_delay_counts"]
    if ({str(key): value for key, value in sorted(counts.items())} != saved_counts or
            dict(counts) != EXPECTED_DELAYS):
        raise ValueError("failure-delay distribution is incorrect")

    direct = [direct_first_failure(r) for r in range(1, 65)]
    if report["direct_first_failures"] != direct:
        raise ValueError("direct finite truncation cross-check is incorrect")
    for r in range(1, 65, 2):
        if direct[r - 1] != (7 * r - 1) // 2:
            raise AssertionError("odd-r nearest-front formula failed")
    for r in range(2, 65, 2):
        m = r // 2
        if direct[r - 1] != 7 * r // 2 + delays[m % CYCLE_LENGTH]:
            raise AssertionError("even-r interface delay disagrees with direct evolution")
    for r in range(2, 16, 2):
        if direct[r - 1] > 4 * r:
            raise AssertionError("small even-r bridge to 4r failed")
    small_even = {str(r): direct[r - 1] for r in range(2, 16, 2)}
    if report["small_even_exact_first_failures"] != small_even:
        raise ValueError("small even-r bridge table is incorrect")
    if any(delay > HORIZON for delay in delays) or 7 * 16 // 2 + HORIZON > 4 * 16:
        raise AssertionError("large even-r bound does not imply 4r")

    expected_theorem = ("For n_r=sum_{j=0}^{r-1}2^(7j), odd r has exact first failure "
                        "(7r-1)/2; for even r=2m the first failure is 7r/2 plus the "
                        "table entry at m mod 512.  In particular q(n_r)<=4r.")
    expected_limit = ("This proves forced-boundary latch termination starting from n_r "
                      "at a centre-zero phase. It does not exclude arbitrary finite "
                      "left halves or later eventual-alternation onsets.")
    if report["theorem"] != expected_theorem or report["claim_limit"] != expected_limit:
        raise ValueError("theorem scope is incorrect")
    return {
        "verified": True,
        "ether_period": 4,
        "combined_state_first_return_samples": 512,
        "arrival_prefix_bits": 18,
        "boundary_switch": "applied only after arrival when testing delays",
        "failure_delay_counts": saved_counts,
        "odd_r_exact": True,
        "odd_r_proof": "left permutivity moves the first missing bit one cell per half-step",
        "even_r_bound": "q(n_r) <= 7r/2 + 8",
        "all_r_bound": "q(n_r) <= 4r",
        "scope": expected_limit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.certificate.read_text()))))


if __name__ == "__main__":
    main()
