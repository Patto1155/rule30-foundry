#!/usr/bin/env python3
"""Check algebra supporting the all-period bridge follow-up.

The symbolic proofs, not the bounded checks here, carry all-period scope.
The loop construction concerns an open five-cell strip, never a full orbit.
"""
from __future__ import annotations

import argparse
import itertools
import json


def rule30(left: int, centre: int, right: int) -> int:
    """Independent elementary-rule truth-table evaluator."""
    return (30 >> (4 * left + 2 * centre + right)) & 1


def loop_pair(word: str) -> tuple[list[list[int]], list[list[int]]]:
    """Two closed 3p-step strip walks with different left-neighbour words."""
    if not word or set(word) != {"0", "1"}:
        raise ValueError("need a nonconstant binary word")
    period = len(word)
    length = 3 * period
    start = period + word.index("0")
    stop = next(t for t in range(start, start + period)
                if word[t % period] == "1")

    def make(pulse: bool) -> list[list[int]]:
        centre = [int(word[t % period]) for t in range(length + 3)]
        right = [int(pulse and start <= t <= stop)
                 for t in range(length + 3)]
        left = [centre[t + 1] ^ (centre[t] | right[t])
                for t in range(length + 2)]
        rows = []
        for t in range(length + 1):
            exterior_left = left[t + 1] ^ (left[t] | centre[t])
            exterior_right = (centre[t] ^ right[t + 1]) if right[t] == 0 else 0
            rows.append([exterior_left, left[t], centre[t], right[t], exterior_right])
        return rows

    return make(False), make(True)


def verify_pair(word: str, pair: tuple[list[list[int]], list[list[int]]]) -> int:
    """Verify interiors via truth table, without the producer's XOR formulas."""
    first, second = pair
    if first[0] != first[-1] or second[0] != second[-1] or first[0] != second[0]:
        raise AssertionError("loops do not close at a common full state")
    if [row[1] for row in first[:-1]] == [row[1] for row in second[:-1]]:
        raise AssertionError("left words coincide")
    checks = 0
    for rows in pair:
        for t, row in enumerate(rows[:-1]):
            if row[2] != int(word[t % len(word)]):
                raise AssertionError("wrong centre phase")
            for index in (1, 2, 3):
                if rule30(*row[index - 1:index + 2]) != rows[t + 1][index]:
                    raise AssertionError((word, t, index))
                checks += 1
    return checks


def local_checks() -> int:
    checks = 0
    for left, centre, right in itertools.product((0, 1), repeat=3):
        output = rule30(left, centre, right)
        assert left == (output ^ (centre | right))
        checks += 1
    # Constant-zero centre gives monotone right neighbour.
    for near, far in itertools.product((0, 1), repeat=2):
        assert rule30(0, near, far) == (near | far)
        checks += 1
    # A nonzero finite row cannot die: its leftmost one advances left.
    assert rule30(0, 0, 0) == 0
    assert rule30(0, 0, 1) == 1
    return checks + 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-period", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.max_period <= 14:
        parser.error("max period must be 2..14")
    words = transitions = 0
    for period in range(2, args.max_period + 1):
        for bits in itertools.product("01", repeat=period):
            word = "".join(bits)
            if len(set(word)) == 1:
                continue
            transitions += verify_pair(word, loop_pair(word))
            words += 1
    print(json.dumps({
        "scope": "local identities and open-strip loop-construction check; not full orbits",
        "max_period_checked": args.max_period,
        "local_truth_table_checks": local_checks(),
        "nonconstant_words_checked": words,
        "interior_truth_table_checks": transitions,
        "failures": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
