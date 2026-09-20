#!/usr/bin/env python3
"""Exact finite identities for the second phase-slip slab.

No global spatial charge is claimed. Exhaustive Boolean checks establish local
identities; telescoping their displayed coboundaries is an all-length argument.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verify_golden_mean_phase_slip import columns


FORBIDDEN = ("11", "101", "10001", "00000")
POTENTIAL = {
    "00100": 0, "01000": 1, "01001": 0, "10000": 1,
    "10010": 0, "00001": 0, "00010": 1,
}


def words(length: int, forbidden: tuple[str, ...]) -> list[str]:
    out = [""]
    for _ in range(length):
        out = [w + b for w in out for b in "01"
               if not any((w + b).endswith(block) for block in forbidden)]
    return out


def second_weight(s: list[int]) -> int:
    a, b, c, d, e, f = s[1:7]
    return (3 - 2*c + 4*a*c - b*d - 2*e + 3*b*e + 6*c*e
            - 8*a*c*e + 2*c*f - 3*a*c*f + d*f)


def periodic_columns(s: list[int], depth: int):
    """Independent periodic-time recurrence using literal Rule-30 truth table."""
    right = [bit for value in s for bit in (value, 0)]
    current = [bit for _ in s for bit in (0, 1)]
    result = []
    for _ in range(depth):
        next_column = []
        for t, centre in enumerate(current):
            future = current[(t + 1) % len(current)]
            candidates = [left for left in (0, 1)
                          if ((30 >> (4*left + 2*centre + right[t])) & 1)
                          == future]
            assert len(candidates) == 1
            next_column.append(candidates[0])
        result.append(next_column)
        right, current = current, next_column
    return result


def verify() -> dict:
    golden = words(9, ("11",))
    for word in golden:
        s = list(map(int, word))
        cs = columns(s, 14)
        actual = sum(pair[0][0] for pair in cs[7:14])
        assert actual == second_weight(s)
        assert actual >= 1

    renewal = words(9, FORBIDDEN)
    delta_table = {}
    for word in renewal:
        cs = columns(list(map(int, word)), 14)
        delta = cs[-1][0][0] + cs[-1][1][0] - 1
        s = list(map(int, word))
        assert delta == (-1 + s[3] + 2*s[4] + 2*s[6] + s[7]
                         + s[2]*s[5] - 2*s[3]*s[6] - 2*s[4]*s[7])
        local = word[2:8]
        assert delta == (int(local == "100001")
                         + POTENTIAL[local[1:]] - POTENTIAL[local[:-1]])
        if local in delta_table:
            assert delta_table[local] == delta
        delta_table[local] = delta
    assert set(delta_table) == set(words(6, FORBIDDEN))

    # The two local obstruction witnesses are exact, not horizon claims.
    cs = periodic_columns([1, 0, 0, 0, 0], 21)
    delta21 = [cs[-1][2*k] + cs[-1][2*k + 1] - 1 for k in range(5)]
    assert sum(delta21) == -2
    for phase in range(5):
        s = [int((phase + k) % 5 == 0) for k in range(20)]
        reference = columns(s, 21)
        for depth, temporal in enumerate(cs):
            assert reference[depth][0][0] == temporal[2*phase]
            assert reference[depth][1][0] == temporal[2*phase + 1]

    empty_word = "0010000100100100001000010"
    assert all(block not in empty_word for block in FORBIDDEN)
    empty_columns = columns(list(map(int, empty_word)), 28)
    emitted = [pair[0][0] for pair in empty_columns]
    assert emitted[21:28] == [0] * 7

    return {
        "golden_mean_nine_bit_words": len(golden),
        "second_slab_weight_minimum": min(second_weight(list(map(int, w)))
                                         for w in golden),
        "gap_3_5_nine_bit_words": len(renewal),
        "depth14_delta_table": dict(sorted(delta_table.items())),
        "depth14_potential": POTENTIAL,
        "depth14_production_word": "100001",
        "depth14_temporal_prefix_lower_bound": -1,
        "period5_depth21_delta": delta21,
        "period5_depth21_total_delta": sum(delta21),
        "gap_3_5_empty_fourth_slab": {
            "boundary_prefix": empty_word,
            "initial_left_prefix": "".join(map(str, emitted)),
        },
        "spatial_charge_proved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
