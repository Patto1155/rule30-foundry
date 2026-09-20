#!/usr/bin/env python3
"""Independent verifier for the alternating-charge counterexample.

The verifier uses only the elementary ECA truth table and shrinking dependency
cones.  It deliberately does not import either the certificate producer or
the packed half-line transfer implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WITNESS = 4_329_177_217
PARENT = WITNESS ^ (1 << 32)
BOUND = 26
WEIGHT = 6
SCOPE = (
    "Finite certificate at a fixed weight, unrestricted support width; "
    "no assertion that such a bound exists for every weight."
)


def cell(rule: int, left: int, centre: int, right: int) -> int:
    """Evaluate one elementary-CA neighbourhood from its truth table."""
    return (rule >> (4 * left + 2 * centre + right)) & 1


def survives(prefix: int, macrosteps: int, rule: int = 30) -> bool:
    """Replay the prescribed 01 centre boundary in its shrinking left cone."""
    cells = [(prefix >> j) & 1 for j in range(2 * macrosteps)]
    for time in range(2 * macrosteps):
        centre = time & 1
        if not any(cell(rule, cells[0], centre, right) == (centre ^ 1)
                   for right in (0, 1)):
            return False
        cells = [cell(rule, cells[j + 1], cells[j],
                      centre if j == 0 else cells[j - 1])
                 for j in range(len(cells) - 1)]
    return True


def first_failure(prefix: int, limit: int, rule: int = 30) -> int | None:
    """Return the first failed macro-latch, independently by cone replay."""
    for q in range(limit + 1):
        if not survives(prefix, q + 1, rule):
            return q
    return None


def sparse_enumeration(weight: int, bound: int, rule: int = 30) -> dict:
    """Enumerate only prefixes surviving each newly exposed two-cell layer."""
    survivors = {0}
    counts = []
    sharp_witness = None
    for depth in range(1, bound + 2):
        shift = 2 * (depth - 1)
        candidates = {
            prefix | (pair << shift)
            for prefix in survivors
            for pair in range(4)
            if prefix.bit_count() + pair.bit_count() <= weight
        }
        survivors = {prefix for prefix in candidates
                     if survives(prefix, depth, rule)}
        counts.append(len(survivors))
        if depth == bound:
            sharp_witness = min(survivors) if survivors else None
    return {
        "rule": rule,
        "weight_cap": weight,
        "first_failure_bound": bound,
        "initial_cone_bits": 2 * (bound + 1),
        "surviving_prefix_counts": counts,
        "verified_bound": not survivors,
        "witness_surviving_bound_latches": sharp_witness,
        "sharp": not survivors and bound > 0 and sharp_witness is not None,
        "scope": SCOPE,
    }


def forced_halfline_state(prefix: int, halfsteps: int) -> int:
    """Evolve a finite left half directly, with centre boundary 0101..."""
    width = max(1, prefix.bit_length()) + halfsteps + 2
    cells = [(prefix >> j) & 1 for j in range(width)]
    for time in range(halfsteps):
        old = cells
        boundary = time & 1
        cells = [cell(30, old[j + 1] if j + 1 < width else 0, old[j],
                      boundary if j == 0 else old[j - 1])
                 for j in range(width)]
    return sum(value << j for j, value in enumerate(cells))


def reconstruct_from_free_boundary(s_value: int, macrosteps: int) -> int:
    """Invert Rule 30 cell by cell, using its truth table rather than a map."""
    def unique_left(centre: int, right: int, following: int) -> int:
        choices = [left for left in (0, 1)
                   if cell(30, left, centre, right) == following]
        if len(choices) != 1:
            raise AssertionError("Rule 30 truth table is not left permutive")
        return choices[0]

    even_previous = [(s_value >> j) & 1 for j in range(macrosteps)]
    even = [0] * macrosteps
    odd_previous = [0] * macrosteps
    odd = [1] * macrosteps
    prefix = 0
    for depth in range(2 * macrosteps):
        even_next = [unique_left(even[j], even_previous[j], odd[j])
                     for j in range(macrosteps)]
        shifted_even = even[1:] + [0]
        odd_next = [unique_left(odd[j], odd_previous[j], shifted_even[j])
                    for j in range(macrosteps)]
        prefix |= even_next[0] << depth
        even_previous, even = even, even_next
        odd_previous, odd = odd, odd_next
    return prefix


def _schema(value: object, keys: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"unexpected {label} schema")
    return value


def _equal(actual: object, expected: object, label: str) -> None:
    if not _same_typed_value(actual, expected):
        raise ValueError(f"incorrect {label}")


def _same_typed_value(actual: object, expected: object) -> bool:
    """Compare decoded JSON without treating booleans as integers."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(actual) == set(expected) and
                all(_same_typed_value(actual[key], expected[key])
                    for key in expected))
    if type(expected) is list:
        return (len(actual) == len(expected) and
                all(_same_typed_value(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def verify(report: object) -> dict:
    """Verify the complete saved certificate and return a compact verdict."""
    report = _schema(report, {
        "experiment", "map", "witness", "highest_one_deletion",
        "fixed_weight_six_theorem", "negative_controls", "conclusion",
        "claim_limit",
    }, "certificate")
    _equal(report["experiment"], "alternating-charge-counterexample", "experiment")
    _equal(report["map"],
           "T_b(n)=(n>>1) XOR (n OR ((n<<1) OR b)); G=T_1(T_0(n))",
           "coordinate map")

    witness = _schema(report["witness"], {
        "n", "hex", "support_zero_based", "weight",
        "first_failure_macrotime_q", "four_weight", "q_minus_four_weight",
        "truth_table_latch_replay", "free_boundary_s_26",
        "free_boundary_s_26_lsb_first", "prefix_charges",
    }, "witness")
    support = [j for j in range(WITNESS.bit_length()) if (WITNESS >> j) & 1]
    replay = {str(m): survives(WITNESS, m) for m in (24, 25, 26, 27)}
    expected_witness = {
        "n": WITNESS,
        "hex": hex(WITNESS),
        "support_zero_based": support,
        "weight": WEIGHT,
        "first_failure_macrotime_q": first_failure(WITNESS, 27),
        "four_weight": 4 * WEIGHT,
        "q_minus_four_weight": BOUND - 4 * WEIGHT,
        "truth_table_latch_replay": replay,
        "free_boundary_s_26": 53_177_130,
        "free_boundary_s_26_lsb_first": format(53_177_130, "026b")[::-1],
        "prefix_charges": {"N_50": 8 * WEIGHT - 50,
                           "N_52": 8 * WEIGHT - 52},
    }
    if not _same_typed_value(witness, expected_witness):
        raise ValueError("witness data disagrees with truth-table replay")
    if reconstruct_from_free_boundary(witness["free_boundary_s_26"], 26) != WITNESS:
        raise ValueError("free boundary does not reconstruct the witness")

    deletion = _schema(report["highest_one_deletion"], {
        "deleted_bit", "parent", "parent_hex", "parent_support_zero_based",
        "parent_weight", "parent_q", "q_jump",
        "parent_unused_four_weight_credit", "child_four_weight_deficit",
        "arrival_macrotime", "child_arrival_state",
        "child_arrival_residue_mod_4", "parent_arrival_state",
        "parent_arrival_residue_mod_4",
    }, "deletion")
    parent_q = first_failure(PARENT, 27)
    child_arrival = forced_halfline_state(WITNESS, 32)
    parent_arrival = forced_halfline_state(PARENT, 32)
    expected_deletion = {
        "deleted_bit": 32,
        "parent": PARENT,
        "parent_hex": hex(PARENT),
        "parent_support_zero_based": [j for j in range(PARENT.bit_length())
                                      if (PARENT >> j) & 1],
        "parent_weight": 5,
        "parent_q": parent_q,
        "q_jump": BOUND - parent_q,
        "parent_unused_four_weight_credit": 20 - parent_q,
        "child_four_weight_deficit": 24 - BOUND,
        "arrival_macrotime": 16,
        "child_arrival_state": child_arrival,
        "child_arrival_residue_mod_4": child_arrival & 3,
        "parent_arrival_state": parent_arrival,
        "parent_arrival_residue_mod_4": parent_arrival & 3,
    }
    if not _same_typed_value(deletion, expected_deletion):
        raise ValueError("highest-one deletion data is incorrect")

    theorem = sparse_enumeration(WEIGHT, BOUND)
    if not _same_typed_value(report["fixed_weight_six_theorem"], theorem):
        raise ValueError("fixed-weight-six enumeration is incorrect")
    if theorem["witness_surviving_bound_latches"] != WITNESS:
        raise AssertionError("independent enumeration found a different extremizer")

    controls = _schema(report["negative_controls"],
                       {"rule_15", "rule_90", "scope"}, "negative controls")
    for rule in (15, 90):
        if not _same_typed_value(controls[f"rule_{rule}"],
                                 sparse_enumeration(0, 8, rule)):
            raise ValueError(f"Rule {rule} negative control is incorrect")
    control_scope = (
        "Rule 15 has a genuine zero-left alternating fibre.  For Rule 90, "
        "the zero word passes only the necessary left-cone constraints; "
        "no full alternating fibre is asserted."
    )
    _equal(controls["scope"], control_scope, "negative-control scope")

    conclusion = (
        "The witness has q=26>4*weight=24.  Its valid 50- and 52-bit "
        "prefixes have charges -2 and -4, refuting 8W_N-N>=-1.  Exact "
        "locality-complete enumeration also proves q<=26 for every ordinary "
        "integer of weight at most six, so this fixed-weight maximum is sharp."
    )
    claim_limit = (
        "This refutes the zero-additive-constant charge bound.  It does not "
        "refute q(n)<=4*weight(n)+C for C>=2 or any eventual-alternation claim."
    )
    _equal(report["conclusion"], conclusion, "conclusion")
    _equal(report["claim_limit"], claim_limit, "claim limit")
    return {
        "verified": True,
        "witness": WITNESS,
        "support_zero_based": support,
        "weight": WEIGHT,
        "first_failure_macrotime_q": BOUND,
        "fixed_weight_six_maximum_q": BOUND,
        "exact_weight_reason": (
            "The cap-six enumeration bounds every exact-weight-six integer, "
            "and the displayed extremizer itself has weight six."
        ),
        "prefix_charges": expected_witness["prefix_charges"],
        "rule_15_zero_prefix_survives": True,
        "rule_90_zero_prefix_survives_necessary_constraints": True,
        "scope": claim_limit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    report = json.loads(args.certificate.read_text(encoding="utf-8"))
    print(json.dumps(verify(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
