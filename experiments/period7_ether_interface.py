#!/usr/bin/env python3
"""Exact interface certificate for finite truncations of a period-7 ether.

The initial left half ``n_r`` has ones at 0, 7, ..., 7(r-1).  Its infinite
periodic continuation is compatible with the alternating boundary.  Odd
truncation lengths are eliminated by the nearest-discrepancy lemma.  For even
lengths this program certifies the finite moving-interface return map used in
``docs/theory/period7-ether-interface.md``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from experiments.alternating_left_transfer import (
        half_step,
        first_failure_macrotime,
        survivor_condition,
    )
except ModuleNotFoundError:  # direct ``python experiments/...py`` execution
    from alternating_left_transfer import (  # type: ignore[no-redef]
        half_step,
        first_failure_macrotime,
        survivor_condition,
    )


def rule30(left: int, center: int, right: int) -> int:
    return left ^ (center | right)


def ether_orbit() -> tuple[tuple[int, ...], ...]:
    """Four temporal phases, indexed by half-line coordinate modulo seven."""
    phases = []
    word = 1
    for _ in range(4):
        phases.append(tuple((word >> j) & 1 for j in range(7)))
        nxt = 0
        for j in range(7):
            nxt |= rule30(
                (word >> ((j + 1) % 7)) & 1,
                (word >> j) & 1,
                (word >> ((j - 1) % 7)) & 1,
            ) << j
        word = nxt
    if word != 1:
        raise AssertionError("ether does not close after four steps")
    return tuple(phases)


ETHER = ether_orbit()


def interface_initial(width: int) -> int:
    return sum(1 << j for j in range(0, width, 7))


def interface_step(delta: int, time: int, width: int) -> int:
    """Advance a closed prefix in coordinates moving inward with the front.

    After translating by the initial cut index, lag ``d`` at time ``time`` is
    half-line index ``-time+d``.  Negative-lag defects are zero.  Output lag
    ``d`` depends only on old lags ``d,d-1,d-2``, so every prefix is an exact
    closed triangular system.
    """
    phase = ETHER[time % 4]
    out = 0
    for d in range(width):
        dm1 = (delta >> (d - 1)) & 1 if d >= 1 else 0
        dm2 = (delta >> (d - 2)) & 1 if d >= 2 else 0
        # The output coordinate is -time-1+d.  These are its centre and
        # boundary-side neighbours in the old ether phase.
        e0 = phase[(-time - 1 + d) % 7]
        em1 = phase[(-time - 2 + d) % 7]
        or_difference = ((e0 ^ dm1) | (em1 ^ dm2)) ^ (e0 | em1)
        out |= (((delta >> d) & 1) ^ or_difference) << d
    return out


def arrival_half(delta: int, time: int, width: int) -> int:
    """Recover the finite half's low prefix when the interface arrives."""
    phase = ETHER[time % 4]
    ether = sum(phase[j % 7] << j for j in range(width))
    return ether ^ delta


def failure_delay(prefix: int, horizon: int) -> int | None:
    """Failure macrotime relative to interface arrival, through ``horizon``."""
    state = prefix
    for delay in range(horizon + 1):
        if not survivor_condition(state):
            return delay
        state = half_step(half_step(state, 0), 1)
    return None


def certificate(width: int = 18, horizon: int = 8) -> dict:
    if width < 2 * horizon + 2:
        raise ValueError("width must be at least 2*horizon+2")
    initial = interface_initial(width)
    delta = initial
    seen = {(delta, 0): 0}
    delays: Counter[int] = Counter()
    delay_table = []
    sample = 0
    while True:
        delay = failure_delay(arrival_half(delta, 14 * sample, width), horizon)
        if delay is None:
            raise AssertionError(f"sample {sample} survives beyond horizon")
        delays[delay] += 1
        delay_table.append(delay)

        for phase_offset in range(14):
            delta = interface_step(delta, 14 * sample + phase_offset, width)
        sample += 1
        key = (delta, (14 * sample) % 28)
        if key in seen:
            cycle_start = seen[key]
            cycle_length = sample - cycle_start
            break
        seen[key] = sample

    if cycle_start != 0 or delta != initial:
        raise AssertionError("expected a pure cycle returning to its initial state")
    # For r=2m, q=7m+D(m).  The uniform q<=4r bound is D(m)<=m.
    # The cycle maximum proves this for m>=8; retain the seven small checks in
    # the certificate itself rather than relying on the optional direct scan.
    small_even = {str(2 * m): 7 * m + delay_table[m] for m in range(1, 8)}
    if max(delays) > 8 or any(delay_table[m] > m for m in range(1, 8)):
        raise AssertionError("q(n_r)<=4r corollary check failed")
    return {
        "experiment": "period7-ether-interface",
        "ether_phases_low_index_first": ["".join(map(str, row)) for row in ETHER],
        "interface_prefix_width": width,
        "macro_failure_horizon_after_arrival": horizon,
        "even_r_sample_spacing_halfsteps": 14,
        "combined_phase_period_halfsteps": 28,
        "pure_sample_cycle_length": cycle_length,
        "failure_delay_counts": {str(k): delays[k] for k in sorted(delays)},
        "even_r_delay_by_m_mod_512_for_r_2m": delay_table,
        "maximum_failure_delay": max(delays),
        "small_even_exact_first_failures": small_even,
        "theorem": (
            "For n_r=sum_{j=0}^{r-1}2^(7j), odd r has exact first failure "
            "(7r-1)/2; for even r=2m the first failure is 7r/2 plus "
            "the table entry at m mod 512.  In particular q(n_r)<=4r."
        ),
        "claim_limit": (
            "This proves forced-boundary latch termination starting from n_r "
            "at a centre-zero phase. It does not exclude arbitrary finite "
            "left halves or later eventual-alternation onsets."
        ),
    }


def direct_cross_check(max_r: int = 64) -> list[int]:
    values = []
    for r in range(1, max_r + 1):
        n = ((1 << (7 * r)) - 1) // 127
        failure = first_failure_macrotime(n, 4 * r + 16)
        if failure is None:
            raise AssertionError(f"direct check was censored at r={r}")
        values.append(failure)
        if r % 2 and failure != (7 * r - 1) // 2:
            raise AssertionError(f"odd-r formula failed at r={r}")
        if r % 2 == 0 and failure > 7 * r // 2 + 8:
            raise AssertionError(f"even-r bound failed at r={r}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--direct-max-r", type=int, default=64)
    args = parser.parse_args()
    report = certificate()
    report["direct_cross_check_max_r"] = args.direct_max_r
    report["direct_first_failures"] = direct_cross_check(args.direct_max_r)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
