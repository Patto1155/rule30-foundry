#!/usr/bin/env python3
"""Small exact checks for the right boundary of an alternating Rule-30 orbit.

At even time write the right half as an integer whose bit zero is the cell
adjacent to the centre.  The centre is 0 and then 1, so its two-step return
map is ``H = R_1 o R_0``.  All enumerations below cover exactly the finite
dependency cone of the requested boundary word.
"""
from __future__ import annotations

import argparse
import json

try:
    from experiments.alternating_left_transfer import macro_step, survivor_condition
    from experiments.alternating_renewal import admits_latches
except ModuleNotFoundError:  # direct ``python experiments/...py`` execution
    from alternating_left_transfer import macro_step, survivor_condition
    from alternating_renewal import admits_latches


COUNTEREXAMPLE_BOUNDARY = "01010100110101101101010011"
LOW_TRIPLE_GRAPH = {
    0: {1, 5, 7}, 1: {0, 4}, 2: {0}, 3: {4, 6},
    4: {2}, 5: {4}, 6: {0, 4}, 7: {2, 6},
}


def right_step(value: int, centre: int) -> int:
    """Advance a one-sided Rule-30 right half with prescribed centre bit."""
    if value < 0 or centre not in (0, 1):
        raise ValueError("invalid right half or centre bit")
    return ((value << 1) | centre) ^ (value | (value >> 1))


def right_macrostep(value: int) -> int:
    """Advance from one even time to the next under centre trace 0,1."""
    return right_step(right_step(value, 0), 1)


def boundary_word(value: int, length: int) -> str:
    """Return ``s_0...s_(length-1)`` in chronological order."""
    if value < 0 or length < 1:
        raise ValueError("invalid value or length")
    out = []
    for _ in range(length):
        out.append(str(value & 1))
        value = right_macrostep(value)
    return "".join(out)


def allowed_boundary_words(length: int) -> set[str]:
    """Enumerate the exact length-``length`` right-boundary language.

    The last observation occurs after ``2*(length-1)`` radius-one steps, so
    it depends on exactly the first ``2*length-1`` initial right cells.
    """
    if not 1 <= length <= 10:
        raise ValueError("length must be in 1..10")
    return {
        boundary_word(value, length)
        for value in range(1 << (2 * length - 1))
    }


def low_triple(value: int) -> str:
    return format(value & 7, "03b")[::-1]


def verify_five_bit_transitions() -> dict:
    """Truth-table check of the low-bit identities used in the proof."""
    nor_mismatches = 0
    chain_mismatches = 0
    map_mismatches = 0
    for value in range(32):
        image = right_macrostep(value)
        # Independently evolve the five-cell cone through the ECA truth table.
        cells = [(value >> j) & 1 for j in range(5)]
        for boundary in (0, 1):
            cells = [rule_output(30, boundary if j == 0 else cells[j - 1],
                                 cells[j], cells[j + 1])
                     for j in range(len(cells) - 1)]
        direct = sum(bit << j for j, bit in enumerate(cells))
        map_mismatches += (image & 7) != direct
        expected_first = int((value & 7) == 0)
        nor_mismatches += (image & 1) != expected_first

        triple = low_triple(value)
        if triple == "010":
            expected = "000"
        elif triple == "001":
            expected = "010"
        elif triple == "011":
            expected = "00" + str(int(bool(value & 0b11000)))
        else:
            continue
        chain_mismatches += low_triple(image) != expected
    actual_graph = {
        triple: {
            right_macrostep(triple | (outer << 3)) & 7
            for outer in range(4)
        }
        for triple in range(8)
    }
    ranks = {0: 0, 2: 1, 4: 2, 6: 3}
    rank_mismatches = sum(
        successor % 2 == 0 and ranks[successor] >= ranks[state]
        for state in ranks for successor in actual_graph[state]
    )
    return {
        "five_bit_rows": 32,
        "truth_table_map_mismatches": map_mismatches,
        "nor_mismatches": nor_mismatches,
        "forced_chain_mismatches": chain_mismatches,
        "factor_graph_mismatches": actual_graph != LOW_TRIPLE_GRAPH,
        "zero_rank_mismatches": rank_mismatches,
    }


def rule_output(rule: int, left: int, centre: int, right: int) -> int:
    if not 0 <= rule <= 255 or any(bit not in (0, 1) for bit in (left, centre, right)):
        raise ValueError("invalid elementary-rule input")
    return (rule >> ((left << 2) | (centre << 1) | right)) & 1


def required_left_bits(rule: int, centre: int, next_centre: int, right: int) -> tuple[int, ...]:
    """All left-neighbour bits compatible with one centre update."""
    return tuple(
        left for left in (0, 1)
        if rule_output(rule, left, centre, right) == next_centre
    )


def admits_coherent_latches(prefix: int, macrosteps: int) -> bool:
    """Whether a left prefix passes ``macrosteps`` latches with no ``11`` in s.

    At even time the physical Rule-30 centre equation infers the adjacent
    right bit as ``s=1-left``.  Avoiding ``11`` is necessary, but not
    sufficient, for that inferred word to be a realizable right trace.
    """
    if prefix < 0 or macrosteps < 1:
        raise ValueError("invalid prefix or macrostep count")
    if not admits_latches(prefix, macrosteps):
        return False
    state = prefix
    previous_s = 0
    for _ in range(macrosteps):
        if not survivor_condition(state):
            return False
        s = 1 - (state & 1)
        if previous_s and s:
            return False
        previous_s = s
        state = macro_step(state)
    return True


def check_coherent_bound(weight: int, bound: int) -> dict:
    """Certify ``ell(n)<=bound`` for all integers of weight at most ``weight``.

    ``ell`` counts initial passed alternating latches for which the inferred
    even-time right word avoids ``11``.  It is at most the unrestricted
    left-only lifetime ``q``.  The enumeration retains only exact dependency-
    cone prefixes and is therefore unrestricted in support width.
    """
    if not 0 <= weight <= 14 or bound < 0:
        raise ValueError("require weight in 0..14 and a nonnegative bound")
    prefixes = {0}
    counts = []
    witness = None
    for m in range(1, bound + 2):
        candidates = (
            prefix | (pair << (2 * (m - 1)))
            for prefix in prefixes for pair in range(4)
        )
        prefixes = {
            prefix for prefix in candidates
            if prefix.bit_count() <= weight
            and admits_coherent_latches(prefix, m)
        }
        counts.append(len(prefixes))
        if m == bound and prefixes:
            witness = min(prefixes)
    return {
        "weight_cap": weight,
        "coherent_lifetime_bound": bound,
        "initial_cone_bits": 2 * (bound + 1),
        "surviving_prefix_counts": counts,
        "verified_bound": not prefixes,
        "witness_surviving_bound_latches": witness,
        "sharp": not prefixes and (bound == 0 or witness is not None),
        "scope": (
            "Locality-complete at the stated weight cap for the no-11 "
            "necessary relaxation; ell is not the unrestricted lifetime q."
        ),
    }


def run() -> dict:
    local = verify_five_bit_transitions()
    if any(local[key] for key in (
            "truth_table_map_mismatches", "nor_mismatches",
            "forced_chain_mismatches", "factor_graph_mismatches",
            "zero_rank_mismatches")):
        raise AssertionError(f"local identity changed: {local}")

    words2 = allowed_boundary_words(2)
    words5 = allowed_boundary_words(5)
    words6 = allowed_boundary_words(6)
    if "11" in words2 or "00000" in words5 or "101001" in words6:
        raise AssertionError("forbidden right-boundary word became realizable")

    witness_11_positions = [
        i for i in range(len(COUNTEREXAMPLE_BOUNDARY) - 1)
        if COUNTEREXAMPLE_BOUNDARY[i:i + 2] == "11"
    ]
    return {
        "experiment": "alternating-right-coherence",
        "right_map": "R_b(n)=((n<<1)|b) XOR (n|(n>>1)); H=R_1(R_0(n))",
        "local_truth_table": local,
        "language": {
            "allowed_counts": {"2": len(words2), "5": len(words5), "6": len(words6)},
            "forbidden": ["11", "00000", "101001"],
            "scope": (
                "11 and 00000 have analytic local proofs.  The missing word "
                "101001 shows that avoiding those two words is not an exact "
                "description of the right-boundary language."
            ),
        },
        "charge_counterexample": {
            "boundary": COUNTEREXAMPLE_BOUNDARY,
            "adjacent_one_positions_zero_based": witness_11_positions,
            "full_orbit_compatible": not witness_11_positions,
        },
        "rule30_centre_constraints": {
            "even_0_to_odd_1": "left = 1 - right",
            "odd_1_to_even_0": "left = 1",
        },
        "controls": {
            "rule15": "Its alternating-centre condition is left=centre; the Rule-30 right recurrence does not apply.",
            "rule90": "Its centre condition is left XOR right=next_centre; the Rule-30 right recurrence does not apply.",
        },
        "claim_limit": (
            "Finite exact truth tables and an incompatibility check only.  "
            "They do not prove a uniform cone-charge inequality."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=int)
    parser.add_argument("--bound", type=int)
    args = parser.parse_args()
    if (args.weight is None) != (args.bound is None):
        parser.error("--weight and --bound must be supplied together")
    report = (run() if args.weight is None
              else check_coherent_bound(args.weight, args.bound))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
