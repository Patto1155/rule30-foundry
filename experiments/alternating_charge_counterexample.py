#!/usr/bin/env python3
"""Exact weight-six counterexample to the proposed alternating cone charge.

The fixed-weight enumeration is locality-complete: testing 27 latches reads
only the first 54 initial cells, so an initial one outside that cone cannot
repair a failure.  The elementary-rule replay in ``alternating_renewal`` is
independent of the packed integer transfer map used to compute ``q``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.alternating_left_transfer import first_failure_macrotime, macro_step
    from experiments.alternating_left_weight import induced_left_prefix
    from experiments.alternating_renewal import admits_latches, check_bound
except ModuleNotFoundError:  # direct ``python experiments/...py`` execution
    from alternating_left_transfer import first_failure_macrotime, macro_step
    from alternating_left_weight import induced_left_prefix
    from alternating_renewal import admits_latches, check_bound


WITNESS = 4_329_177_217
HIGHEST_BIT = 32
PARENT = WITNESS ^ (1 << HIGHEST_BIT)
FREE_BOUNDARY = 53_177_130


def run() -> dict:
    q = first_failure_macrotime(WITNESS, 27)
    parent_q = first_failure_macrotime(PARENT, 27)
    if q != 26 or parent_q != 16:
        raise AssertionError("packed transfer-map witness replay changed")

    latch_replay = {
        str(m): admits_latches(WITNESS, m) for m in (24, 25, 26, 27)
    }
    if latch_replay != {"24": True, "25": True, "26": True, "27": False}:
        raise AssertionError("elementary truth-table witness replay changed")
    sideways = induced_left_prefix(FREE_BOUNDARY, 26)
    reconstructed = sum(int(bit) << j for j, bit in enumerate(sideways))
    if reconstructed != WITNESS:
        raise AssertionError("sideways boundary word no longer reconstructs witness")

    child_arrival = WITNESS
    parent_arrival = PARENT
    for _ in range(16):
        child_arrival = macro_step(child_arrival)
        parent_arrival = macro_step(parent_arrival)

    fixed_weight = check_bound(weight=6, bound=26)
    if not fixed_weight["verified_bound"] or not fixed_weight["sharp"]:
        raise AssertionError("weight-six exhaustive bound did not close")
    if fixed_weight["witness_surviving_bound_latches"] != WITNESS:
        raise AssertionError("weight-six extremizer changed")

    controls = {
        str(rule): check_bound(weight=0, bound=8, rule=rule)
        for rule in (15, 90)
    }
    for control in controls.values():
        if control["verified_bound"] or control["surviving_prefix_counts"] != [1] * 9:
            raise AssertionError("negative control was incorrectly eliminated")

    weight = WITNESS.bit_count()
    parent_weight = PARENT.bit_count()
    return {
        "experiment": "alternating-charge-counterexample",
        "map": "T_b(n)=(n>>1) XOR (n OR ((n<<1) OR b)); G=T_1(T_0(n))",
        "witness": {
            "n": WITNESS,
            "hex": hex(WITNESS),
            "support_zero_based": [
                j for j in range(WITNESS.bit_length()) if (WITNESS >> j) & 1
            ],
            "weight": weight,
            "first_failure_macrotime_q": q,
            "four_weight": 4 * weight,
            "q_minus_four_weight": q - 4 * weight,
            "truth_table_latch_replay": latch_replay,
            "free_boundary_s_26": FREE_BOUNDARY,
            "free_boundary_s_26_lsb_first": format(FREE_BOUNDARY, "026b")[::-1],
            "prefix_charges": {
                "N_50": 8 * weight - 50,
                "N_52": 8 * weight - 52,
            },
        },
        "highest_one_deletion": {
            "deleted_bit": HIGHEST_BIT,
            "parent": PARENT,
            "parent_hex": hex(PARENT),
            "parent_support_zero_based": [
                j for j in range(PARENT.bit_length()) if (PARENT >> j) & 1
            ],
            "parent_weight": parent_weight,
            "parent_q": parent_q,
            "q_jump": q - parent_q,
            "parent_unused_four_weight_credit": 4 * parent_weight - parent_q,
            "child_four_weight_deficit": 4 * weight - q,
            "arrival_macrotime": 16,
            "child_arrival_state": child_arrival,
            "child_arrival_residue_mod_4": child_arrival & 3,
            "parent_arrival_state": parent_arrival,
            "parent_arrival_residue_mod_4": parent_arrival & 3,
        },
        "fixed_weight_six_theorem": fixed_weight,
        "negative_controls": {
            "rule_15": controls["15"],
            "rule_90": controls["90"],
            "scope": (
                "Rule 15 has a genuine zero-left alternating fibre.  For Rule 90, "
                "the zero word passes only the necessary left-cone constraints; "
                "no full alternating fibre is asserted."
            ),
        },
        "conclusion": (
            "The witness has q=26>4*weight=24.  Its valid 50- and 52-bit "
            "prefixes have charges -2 and -4, refuting 8W_N-N>=-1.  Exact "
            "locality-complete enumeration also proves q<=26 for every ordinary "
            "integer of weight at most six, so this fixed-weight maximum is sharp."
        ),
        "claim_limit": (
            "This refutes the zero-additive-constant charge bound.  It does not "
            "refute q(n)<=4*weight(n)+C for C>=2 or any eventual-alternation claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--out", type=Path)
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()

    report = run()
    payload = json.dumps(report, indent=2) + "\n"
    if args.verify:
        saved = json.loads(args.verify.read_text(encoding="utf-8"))
        if saved != report:
            raise SystemExit("saved report does not match exact recomputation")
        print(f"verified {args.verify}")
    elif args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8", newline="")
        print(payload, end="")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
