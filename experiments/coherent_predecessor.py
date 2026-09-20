#!/usr/bin/env python3
"""Enumerate shared-predecessor transitions between slanted-cut triples.

The state ``(A, B, C)`` is stored in the geometry note's order.  An original
word is stored in increasing spatial order ``(C, B, A, z, w, ...)``.  One CA
step replaces every consecutive truth-table triple, so ``3 + 2*k`` original
bits determine the three observed state bits through ``k`` updates.

The experiment is deliberately bounded.  It compares one independently
existential update with two updates sharing one seven-bit predecessor, then
checks paths made from overlapping two-update relations only until their first
failure to describe a real temporal path.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

Bit = int
State = tuple[Bit, Bit, Bit]
PathStates = tuple[State, ...]


def truth_table_cell(rule: int, left: Bit, centre: Bit, right: Bit) -> Bit:
    """Return one ECA update by direct Wolfram truth-table indexing."""
    if not 0 <= rule <= 255:
        raise ValueError(f"rule must be in [0, 255], got {rule}")
    if any(bit not in (0, 1) for bit in (left, centre, right)):
        raise ValueError("cell inputs must be bits")
    return (rule >> (4 * left + 2 * centre + right)) & 1


def step_word(rule: int, word: tuple[Bit, ...]) -> tuple[Bit, ...]:
    """Update a finite dependency window; no exterior boundary is consulted."""
    if len(word) < 3:
        raise ValueError("a dependency window needs at least three bits")
    return tuple(
        truth_table_cell(rule, word[i], word[i + 1], word[i + 2])
        for i in range(len(word) - 2)
    )


def observed_state(word: tuple[Bit, ...]) -> State:
    """Read ``(A,B,C)`` from a dependency word beginning ``(C,B,A)``."""
    if len(word) < 3:
        raise ValueError("an observed state needs at least three bits")
    return word[2], word[1], word[0]


def trace_original(rule: int, word: tuple[Bit, ...], updates: int) -> PathStates:
    """Trace a triple through ``updates`` using one shared original word."""
    if updates < 0:
        raise ValueError("updates must be nonnegative")
    expected = 3 + 2 * updates
    if len(word) != expected:
        raise ValueError(
            f"{updates} updates require exactly {expected} original bits"
        )
    current = word
    states = [observed_state(current)]
    for _ in range(updates):
        current = step_word(rule, current)
        states.append(observed_state(current))
    return tuple(states)


def realized_paths(rule: int, updates: int) -> frozenset[PathStates]:
    """All paths produced by the ``2**(3+2*updates)`` original words."""
    if updates < 0:
        raise ValueError("updates must be nonnegative")
    width = 3 + 2 * updates
    return frozenset(
        trace_original(rule, word, updates)
        for word in itertools.product((0, 1), repeat=width)
    )


def compose_edges(edges: Iterable[PathStates], updates: int) -> frozenset[PathStates]:
    """Compose an existential one-update relation for ``updates`` steps."""
    if updates < 1:
        raise ValueError("updates must be positive")
    relation = frozenset(edges)
    if any(len(edge) != 2 for edge in relation):
        raise ValueError("edge relation must contain two-state paths")
    paths = relation
    for _ in range(1, updates):
        paths = frozenset(
            path + (edge[1],)
            for path in paths
            for edge in relation
            if path[-1] == edge[0]
        )
    return paths


def compose_two_update_windows(
    relation: Iterable[PathStates], updates: int
) -> frozenset[PathStates]:
    """Compose overlapping shared-predecessor two-update relations."""
    if updates < 2:
        raise ValueError("two-update windows require at least two updates")
    windows = frozenset(relation)
    if any(len(window) != 3 for window in windows):
        raise ValueError("two-update relation must contain three-state paths")
    paths = windows
    for _ in range(2, updates):
        paths = frozenset(
            path + (window[2],)
            for path in paths
            for window in windows
            if path[-2:] == window[:2]
        )
    return paths


def bits(values: Iterable[Bit]) -> str:
    return "".join(str(value) for value in values)


def path_text(path: PathStates) -> str:
    return "->".join(bits(state) for state in path)


def witnesses(rule: int, target: PathStates) -> list[str]:
    updates = len(target) - 1
    width = 3 + 2 * updates
    return [
        bits(word)
        for word in itertools.product((0, 1), repeat=width)
        if trace_original(rule, word, updates) == target
    ]


def relation_summary(rule: int) -> dict:
    """Return exact one/two-update counts for one actual ECA truth table."""
    one = realized_paths(rule, 1)
    independent_two = compose_edges(one, 2)
    coherent_two = realized_paths(rule, 2)
    spurious = sorted(independent_two - coherent_two)
    return {
        "rule": rule,
        "one_update_relation_count": len(one),
        "independently_composed_two_update_count": len(independent_two),
        "shared_predecessor_two_update_count": len(coherent_two),
        "spurious_independent_walk_count": len(spurious),
        "spurious_independent_walks": [path_text(path) for path in spurious],
    }


def build_report() -> dict:
    rule = 30
    one = realized_paths(rule, 1)
    independent_two = compose_edges(one, 2)
    coherent_two = realized_paths(rule, 2)

    # This is the named bounded temporal claim: stop at the first failure.
    actual_three = realized_paths(rule, 3)
    local_three = compose_two_update_windows(coherent_two, 3)
    actual_four = realized_paths(rule, 4)
    local_four = compose_two_update_windows(coherent_two, 4)
    spurious_four = sorted(local_four - actual_four)
    first = spurious_four[0]

    window_witnesses = [
        witnesses(rule, first[start:start + 3])
        for start in range(len(first) - 2)
    ]
    forced_prefix = witnesses(rule, first[:3])[0]
    compatible_nine_bit_words = witnesses(rule, first[:4])
    continuations = [
        word
        for word in itertools.product((0, 1), repeat=11)
        if bits(word).startswith(forced_prefix)
    ]
    final_counts: dict[str, int] = {}
    for word in continuations:
        final = bits(trace_original(rule, word, 4)[-1])
        final_counts[final] = final_counts.get(final, 0) + 1

    return {
        "experiment": "coherent-predecessor",
        "result": {
            "rule30_relations": relation_summary(30),
            "two_update_coherence_is_temporally_sufficient": False,
            "shortest_tested_and_proved_failure_updates": 4,
            "lexicographically_first_four_update_spurious_path": path_text(first),
            "all_four_update_spurious_paths": [
                path_text(path) for path in spurious_four
            ],
        },
        "evidence": {
            "method": "direct Wolfram truth-table stepping of every bounded original word",
            "two_updates": {
                "original_bit_width": 7,
                "original_words_checked": 128,
                "independent_walk_count": len(independent_two),
                "shared_predecessor_walk_count": len(coherent_two),
            },
            "three_updates": {
                "original_bit_width": 9,
                "original_words_checked": 512,
                "two_update_window_path_count": len(local_three),
                "realized_path_count": len(actual_three),
                "spurious_path_count": len(local_three - actual_three),
            },
            "four_updates": {
                "original_bit_width": 11,
                "original_words_checked": 2048,
                "two_update_window_path_count": len(local_four),
                "realized_path_count": len(actual_four),
                "spurious_path_count": len(spurious_four),
            },
            "first_obstruction_certificate": {
                "two_update_window_witnesses": window_witnesses,
                "first_two_updates_force_seven_bit_original": forced_prefix,
                "compatible_nine_bit_originals_through_three_updates": (
                    compatible_nine_bit_words
                ),
                "four_bit_continuations_checked": len(continuations),
                "forced_fourth_state_counts": dict(sorted(final_counts.items())),
                "requested_fourth_state": bits(first[-1]),
                "truth_table_contradiction": [
                    "The forced original 0000001 gives time-two prefix 00110de for some bits d,e.",
                    "At time three, positions 2,3,4 are 0, 1 XOR d, and d OR e.",
                    "Therefore fourth-state A is 0 XOR ((1 XOR d) OR d OR e) = 1, but state 010 requires A=0.",
                ],
            },
            "truth_table_controls": {
                "rule15": relation_summary(15),
                "rule90": relation_summary(90),
                "scope": "Each control uses its own ECA truth table; no Rule 30 latch or boundary condition is imposed.",
            },
        },
        "gap": (
            "Two-update predecessor coherence removes the four rowwise-spurious "
            "walks, but overlapping coherent windows can still resample older "
            "predecessors. This finite result does not establish a sufficient "
            "state for paths of unbounded duration or a cone-charge potential."
        ),
        "next_implication": (
            "A sound temporal state must retain enough predecessor information "
            "to distinguish the 16 four-update obstructions; merely composing "
            "the 44 coherent two-update triples is insufficient."
        ),
        "claim_limit": (
            "Exhaustive only for the named widths: seven bits/two updates, nine "
            "bits/three updates, and eleven bits/four updates. The search stops "
            "at the first failure of overlapping two-update coherence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="")


if __name__ == "__main__":
    main()
